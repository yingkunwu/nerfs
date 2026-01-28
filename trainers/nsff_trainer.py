import os
import torch
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import imageio
from concurrent.futures import ThreadPoolExecutor

from .factory import BaseTrainer
from models.nsff import NeRF_Static, NeRF_Dynamic, Embedding
from losses.nsff_loss import NSFFLoss
from utils.nsff_rendering import render_rays, interpolate
from utils.metrics import psnr
from utils.ray_utils import get_ray_directions, get_rays, get_ndc_rays
from utils.misc import visualize_depth, create_spiral_poses_from_pose
from utils.flowlib import flow_to_image


class NSFFTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.criterion = NSFFLoss()

    def create_nerf(self, cfg):
        """Create NeRF model and embeddings."""
        # Embeddings
        embeddings = {
            'xyz': Embedding(
                input_dim=3,
                max_freq_log2=cfg.model.xyz_embed_dim - 1,
                num_freqs=cfg.model.xyz_embed_dim,
            ),
            'dir': Embedding(
                input_dim=3,
                max_freq_log2=cfg.model.dir_embed_dim - 1,
                num_freqs=cfg.model.dir_embed_dim,
            ),
            'transient': torch.nn.Embedding(
                cfg.N_vocab, cfg.model.tra_embed_dim)
        }

        # Models
        model_static = NeRF_Static(
            depth=cfg.model.depth,
            width=cfg.model.width,
            skips=cfg.model.skips,
            in_ch_xyz=embeddings['xyz'].output_dim,
            in_ch_dir=embeddings['dir'].output_dim,
        )
        model_dynamic = NeRF_Dynamic(
            depth=cfg.model.depth,
            width=cfg.model.width,
            skips=cfg.model.skips,
            in_ch_xyz=embeddings['xyz'].output_dim,
            in_ch_dir=embeddings['dir'].output_dim,
            in_ch_t=cfg.model.tra_embed_dim,
        )
        models = {'static': model_static, 'dynamic': model_dynamic}

        return embeddings, models

    def forward(self, inputs, step, infer_only=False):
        """Do batched inference on rays using chunk."""
        rays, rays_t, max_t = inputs['rays'], inputs['rays_t'], inputs['max_t']

        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.cfg.chunk):
            results_chunk = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.cfg.chunk],
                rays_t[i:i+self.cfg.chunk],
                max_t,
                self.cfg.N_samples,
                self.cfg.perturb,
                self.cfg.noise_std)

            for k, v in results_chunk.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        if infer_only:
            return results

        log = self.criterion(results, inputs, step)

        return results, log

    def extract_from_sample(self, sample, batch_size=None):
        rays = sample['rays'].to(self.device)  # [N_rays, 6]
        rays_t = sample['rays_t'].to(self.device)  # [N_rays,]
        rgbs = sample['rgbs'].to(self.device)  # [N_rgbs, 3]
        depth = sample['depth'].to(self.device)  # [N_rays,]
        uv_fw = sample['uv_fw'].to(self.device)  # [N_rays, 2]
        uv_bw = sample['uv_bw'].to(self.device)  # [N_rays, 2]
        uv = sample['uv'].to(self.device)  # [N_rays, 2]

        # all the data share the same camera intrinsics and extrinsics
        if isinstance(sample['Ks'], list):
            Ks = sample['Ks'][0]
            Ps = sample['Ps'][0]
        else:
            Ks = sample['Ks']
            Ps = sample['Ps']
        Ks = Ks.to(self.device)
        Ps = Ps.to(self.device)

        if batch_size is not None:
            num_rgb_rays = rays.shape[0]
            idx = torch.randint(0, num_rgb_rays,
                                (batch_size,),
                                device=self.device)
            rays = rays[idx]
            rays_t = rays_t[idx]
            rgbs = rgbs[idx]
            depth = depth[idx]
            uv_fw = uv_fw[idx]
            uv_bw = uv_bw[idx]
            uv = uv[idx]

        inputs = {
            "rays": rays,
            "rays_t": rays_t,
            "rgbs": rgbs,
            "depth": depth,
            "uv_fw": uv_fw,
            "uv_bw": uv_bw,
            "uv": uv,
            'Ks': Ks,
            'Ps': Ps
        }

        return inputs

    def fit(self, train_dataset, val_dataset):
        print("Starting training loop")
        best_psnr = 0.0
        pbar = tqdm(range(self.cfg.iters), total=self.cfg.iters)
        for step in pbar:
            for m in self.models.values():
                m.train()
            # Sample batch

            def get_sample():
                return train_dataset.sample(shuffle=True)

            # you can increase the number of workers if memory allows
            with ThreadPoolExecutor(max_workers=8) as executor:
                samples = list(executor.map(lambda _: get_sample(), range(8)))

            # a list of 16 sampled batches
            sample = {}
            for key in samples[0].keys():
                if key in ['Ks', 'Ps']:
                    sample[key] = [s[key] for s in samples]
                elif isinstance(samples[0][key], torch.Tensor):
                    sample[key] = torch.cat([s[key] for s in samples], dim=0)
                elif isinstance(samples[0][key], list):
                    sample[key] = []
                    for s in samples:
                        sample[key].extend(s[key])
                else:
                    sample[key] = [s[key] for s in samples]

            max_t = len(train_dataset)
            inputs = self.extract_from_sample(sample, self.cfg.batch_size)
            inputs["max_t"] = max_t

            # advance the batch pointer
            self.optimizer.zero_grad()

            results, log = self.forward(inputs, step)

            log['train/psnr'] = psnr(
                results['rgb_map_ref'], inputs['rgbs'])
            log['train/loss'] = sum(
                [v for k, v in log.items() if 'loss' in k])

            # Backpropagation and optimizer step
            log['train/loss'].backward()
            self.optimizer.step()
            self.scheduler.step()

            # TensorBoard logging
            for k, v in log.items():
                self.writer.add_scalar(k, v.item(), step)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', current_lr, step)

            # Update tqdm with readable metrics
            loss_val = log['train/loss'].item()
            psnr_val = log['train/psnr'].item()
            pbar.set_postfix({'loss': f'{loss_val:.6f}',
                              'psnr': f'{psnr_val:.3f}'})

            # Validation
            if (step + 1) % 1000 == 0:
                for m in self.models.values():
                    m.eval()

                with torch.no_grad():
                    max_t = len(val_dataset)
                    sample = val_dataset.sample(shuffle=False)
                    inputs = self.extract_from_sample(sample)
                    inputs["max_t"] = max_t

                    results, log = self.forward(inputs, step)

                    log['val/psnr'] = psnr(
                        results['rgb_map_ref'], inputs['rgbs'])
                    log['val/loss'] = sum(
                        [v for k, v in log.items() if 'loss' in k])

                    # TensorBoard logging
                    for k, v in log.items():
                        self.writer.add_scalar(k, v.item(), step)

                    H, W = sample['image_size']

                    img_rgb = results['rgb_map_ref']\
                        .view(H, W, 3).permute(2, 0, 1).cpu()
                    img_rgb_rig = results['rgb_map_static']\
                        .view(H, W, 3).permute(2, 0, 1).cpu()
                    img_rgb_dy = results['rgb_map_ref_dynamic']\
                        .view(H, W, 3).permute(2, 0, 1).cpu()
                    img_gt = sample['rgbs']\
                        .view(H, W, 3).permute(2, 0, 1).cpu()

                    depth_gt = visualize_depth(inputs['depth'].view(H, W))
                    depth = visualize_depth(
                        -results['depth_map_ref'].view(H, W))
                    depth_rig = visualize_depth(
                        -results['depth_map_static'].view(H, W))
                    depth_dy = visualize_depth(
                        -results['depth_map_ref_dynamic'].view(H, W))

                    # Create a 2x4 grid
                    row1 = torch.cat(
                        [img_gt, img_rgb, img_rgb_rig, img_rgb_dy], -1)
                    row2 = torch.cat(
                        [depth_gt, depth, depth_rig, depth_dy], -1)
                    grid = torch.cat([row1, row2], -2)
                    self.writer.add_image('val/visualization', grid, step)

                    save_name = os.path.join(
                        self.save_vis_path, f'val_{step:06d}.png')
                    save_image(grid, save_name, nrow=2)

                    def get_img(target):
                        diff = (target - inputs['uv']).view(H, W, 2)
                        return flow_to_image(diff.cpu().numpy())

                    flow_fw_gt = get_img(inputs['uv_fw'])
                    flow_bw_gt = get_img(inputs['uv_bw'])
                    flow_fw_pred = get_img(results['uv_fw'])
                    flow_bw_pred = get_img(results['uv_bw'])

                    grid_np = np.stack([
                        flow_fw_gt, flow_bw_gt, flow_fw_pred, flow_bw_pred])
                    grid_tensor = \
                        torch.from_numpy(grid_np).permute(0, 3, 1, 2).float()
                    if grid_tensor.max() > 1.0:
                        grid_tensor /= 255.0
                    save_name = os.path.join(
                        self.save_vis_path, f'flow_{step:06d}.png')
                    save_image(grid_tensor, save_name, nrow=2)

                    if log['val/psnr'].item() > best_psnr:
                        # save model weight
                        self.save_model()
                        best_psnr = log['val/psnr'].item()

    def inference(self, val_dataset):
        save_dir = os.path.join(self.log_dir, "inference")
        os.makedirs(save_dir, exist_ok=True)

        max_trans = np.percentile(
            np.abs(np.diff(val_dataset.poses[:, 0, 3])), 10)
        radii = np.array([max_trans, max_trans, 0])
        poses = create_spiral_poses_from_pose(
            val_dataset.poses, radii, n_poses=6 * len(val_dataset))

        W, H = val_dataset.img_wh
        K = torch.FloatTensor(val_dataset.K)
        directions = get_ray_directions(H, W, K)

        for m in self.models.values():
            m.eval()

        imgs = []
        total = np.linspace(
            0, len(val_dataset) - 1, 6 * len(val_dataset)
        ).tolist()[:-1]
        for i, cur_time in tqdm(enumerate(total),
                                total=len(total),
                                desc="Rendering spiral poses"):
            flow_time = int(np.floor(cur_time))
            dt = cur_time - np.floor(cur_time)

            c2w = torch.FloatTensor(poses[i])
            rays_o, rays_d = get_rays(directions, c2w)
            rays_o, rays_d = get_ndc_rays(K, 1.0, rays_o, rays_d)

            near, far = 0, 1

            near_ = near * torch.ones_like(rays_o[:, :1])
            far_ = far * torch.ones_like(rays_o[:, :1])
            rays = torch.cat([rays_o, rays_d, near_, far_], dim=1)
            rays_t = flow_time * torch.ones(len(rays_o), dtype=torch.long)

            inputs = {
                'rays': rays.to(self.device),
                "rays_t": rays_t.to(self.device),
                "max_t": len(val_dataset)
            }
            with torch.no_grad():
                results = self.forward(inputs, i, infer_only=True)

            inputs['rays_t'] += 1
            with torch.no_grad():
                results_tp1 = self.forward(inputs, i, infer_only=True)

            img, depth = interpolate(
                results,
                results_tp1,
                dt,
                K.to(self.device),
                c2w.to(self.device),
                val_dataset.img_wh)

            img = torch.clip(img, 0, 1)
            img = (img.numpy() * 255).astype(np.uint8)
            depth = visualize_depth(depth.view(H, W))
            depth = depth.permute(1, 2, 0).numpy()
            depth = (depth * 255).astype(np.uint8)

            stack = np.concatenate([img, depth], axis=1)
            imgs += [stack]
            imageio.imwrite(os.path.join(save_dir, f'{i:03d}.png'), stack)

            del results
            del results_tp1

        imageio.mimsave(os.path.join(save_dir, 'animation.gif'), imgs, fps=30)

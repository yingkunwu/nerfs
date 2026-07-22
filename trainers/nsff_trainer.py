import os
import torch
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import imageio

from .factory import BaseTrainer
from models.nsff import NeRF_Static, NeRF_Dynamic, Embedding
from losses.nsff_loss import NSFFLoss
from utils.nsff_rendering import render_rays, interpolate
from utils.metrics import psnr
from utils.ray_utils import get_ray_directions, get_rays, get_ndc_rays
from utils.misc import visualize_depth, create_spiral_poses_from_pose, save_gif
from utils.flowlib import flow_to_image


class NSFFTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.criterion = NSFFLoss(
            decay_iteration=cfg.get('decay_iteration', 30))

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

    def extract_from_sample(self, sample):
        inputs = {
            "rays": sample['rays'].to(self.device),  # [N_rays, 8]
            "rays_t": sample['rays_t'].to(self.device),  # [N_rays,]
            "rgbs": sample['rgbs'].to(self.device),  # [N_rays, 3]
            "depth": sample['depth'].to(self.device),  # [N_rays,]
            "mask": sample['mask'].to(self.device),  # [N_rays,] 1 = dynamic
            "uv_fw": sample['uv_fw'].to(self.device),  # [N_rays, 2]
            "uv_bw": sample['uv_bw'].to(self.device),  # [N_rays, 2]
            "uv": sample['uv'].to(self.device),  # [N_rays, 2]
            "Ks": sample['Ks'].to(self.device),  # [3, 3]
            "Ps": sample['Ps'].to(self.device),  # [N_frames, 3, 4]
        }

        return inputs

    def fit(self, train_dataset, val_dataset):
        print("Starting training loop")
        best_psnr = 0.0
        num_extra = self.cfg.get('num_extra_samples', 512)
        decay_steps = self.criterion.decay_iteration * 1000
        pbar = tqdm(range(self.cfg.iters), total=self.cfg.iters)
        for step in pbar:
            for m in self.models.values():
                m.train()

            # Sample rays from one random frame. During the early stage,
            # additionally hard-mine rays from the dynamic region (motion
            # mask) like the original NSFF.
            n_extra = num_extra if step < decay_steps else 0
            sample = train_dataset.sample(
                shuffle=True, batch_size=self.cfg.batch_size,
                num_extra=n_extra)

            max_t = len(train_dataset)
            inputs = self.extract_from_sample(sample)
            inputs["max_t"] = max_t
            # extra hard-mined rays are excluded from the union render loss
            inputs["n_uniform"] = self.cfg.batch_size

            # advance the batch pointer
            self.optimizer.zero_grad()

            results, log = self.forward(inputs, step)

            log['train/psnr'] = psnr(
                results['rgb_map_ref'][:self.cfg.batch_size],
                inputs['rgbs'][:self.cfg.batch_size])
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

    def _render_novel_frame(self, c2w, cur_time, K, directions, val_dataset):
        """
        Render a single novel (view, time) frame.

        @c2w is the camera pose to render from and @cur_time is a continuous
        time index. Integer times render the scene directly; fractional times
        are produced by warping the dynamic field along the predicted scene
        flow and splatting it (softsplat) between the two bracketing integer
        times. Returns an (H, W*2, 3) uint8 array of [rgb | depth].
        """
        W, H = val_dataset.img_wh
        flow_time = int(np.floor(cur_time))
        dt = float(cur_time - np.floor(cur_time))

        rays_o, rays_d = get_rays(directions, c2w)
        shift_near = -min(-1.0, float(c2w[2, 3]))
        rays_o, rays_d = get_ndc_rays(
            K, 1.0, rays_o, rays_d, shift_near=shift_near)
        near_ = torch.zeros_like(rays_o[:, :1])
        far_ = torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near_, far_], dim=1)
        rays_t = flow_time * torch.ones(len(rays_o), dtype=torch.long)

        inputs = {
            'rays': rays.to(self.device),
            'rays_t': rays_t.to(self.device),
            'max_t': len(val_dataset),
        }
        with torch.no_grad():
            results = self.forward(inputs, 0, infer_only=True)

        if dt > 0:
            # need the t+1 render to interpolate the in-between time
            inputs['rays_t'] = inputs['rays_t'] + 1
            with torch.no_grad():
                results_tp1 = self.forward(inputs, 0, infer_only=True)
        else:
            # exact integer time: interpolate() weights t+1 by dt = 0, so we
            # can reuse the current results and skip a forward pass (this also
            # avoids querying a time embedding at t+1 for the last frame)
            results_tp1 = results

        img, depth = interpolate(
            results, results_tp1, dt,
            K.to(self.device), c2w.to(self.device), val_dataset.img_wh)

        img = torch.clip(img, 0, 1)
        img = (img.numpy() * 255).astype(np.uint8)
        depth = visualize_depth(depth.view(H, W))
        depth = (depth.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return np.concatenate([img, depth], axis=1)

    def _run_inference(self, val_dataset, cam_poses, times, save_dir, desc):
        """Render a sequence of (camera pose, time) pairs into a looping GIF."""
        os.makedirs(save_dir, exist_ok=True)
        W, H = val_dataset.img_wh
        K = torch.FloatTensor(val_dataset.K)
        directions = get_ray_directions(H, W, K)

        for m in self.models.values():
            m.eval()

        imgs = []
        for i, (pose, cur_time) in tqdm(
                enumerate(zip(cam_poses, times)),
                total=len(times), desc=desc):
            c2w = torch.FloatTensor(pose)
            stack = self._render_novel_frame(
                c2w, cur_time, K, directions, val_dataset)
            imgs += [stack]
            imageio.imwrite(os.path.join(save_dir, f'{i:03d}.png'), stack)

        save_gif(os.path.join(save_dir, 'animation.gif'), imgs, fps=30)
        return imgs

    def inference(self, val_dataset):
        """Spiral camera path AND advancing time (the default demo)."""
        N = len(val_dataset)
        max_trans = np.percentile(
            np.abs(np.diff(val_dataset.poses[:, 0, 3])), 10)
        radii = np.array([max_trans, max_trans, 0])
        cam_poses = create_spiral_poses_from_pose(
            val_dataset.poses, radii, n_poses=6 * N)
        times = np.linspace(0, N - 1, 6 * N).tolist()[:-1]
        self._run_inference(
            val_dataset, cam_poses, times,
            os.path.join(self.log_dir, "inference"),
            "Rendering spiral poses (view + time)")

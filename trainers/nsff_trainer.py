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
from utils.ray_utils import (
    get_ray_directions, get_rays, get_ndc_rays, ndc2world
)
from utils.misc import (
    visualize_depth, create_spiral_poses_from_pose, create_wander_path,
    export_animation
)
from utils.flowlib import flow_to_image


class NSFFTrainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.criterion = NSFFLoss(
            lambda_geo=cfg.get('lambda_geo', 0.04),
            lambda_reg=cfg.get('lambda_reg', 0.1),
            thickness=cfg.get('thickness', 15),
        ).to(self.device)

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
            in_ch_t=cfg.model.tra_embed_dim,
            flow_scale=cfg.model.get('flow_scale', 0.2),
        )
        models = {'static': model_static, 'dynamic': model_dynamic}

        return embeddings, models

    def forward(self, inputs, test_time=False,
                output_transient_flow=('fw', 'bw', 'disocc'), dataset=None):
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
                self.cfg.perturb if not test_time else 0.0,
                self.cfg.noise_std if not test_time else 0.0,
                test_time=test_time,
                output_transient_flow=output_transient_flow,
                dataset=dataset)

            for k, v in results_chunk.items():
                if test_time:
                    v = v.cpu()
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def extract_from_sample(self, sample):
        inputs = {
            "rays": sample['rays'].to(self.device),  # [N_rays, 8]
            "rays_t": sample['rays_t'].to(self.device),  # [N_rays,]
            "rgbs": sample['rgbs'].to(self.device),  # [N_rays, 3]
            "depth": sample['depth'].to(self.device),  # [N_rays,] disparity
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

        # nsff_pl schedule: one "epoch" is W*H*N_frames/1000 steps; the
        # monodepth/flow losses decay x0.1 and the cross-entropy weight
        # ramps up on a 10-epoch cadence
        w, h = train_dataset.img_wh
        N_frames = len(train_dataset)
        steps_per_epoch = w * h * N_frames // 1000
        num_epochs = self.cfg.get('num_epochs', 50)
        total_steps = num_epochs * steps_per_epoch
        val_every = self.cfg.get('val_every', 2000)
        print(f"steps/epoch: {steps_per_epoch}, epochs: {num_epochs}, "
              f"total steps: {total_steps}")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps,
            eta_min=self.cfg.get('lr_min', 1e-8))

        max_t = N_frames - 1
        pbar = tqdm(range(total_steps), total=total_steps)
        for step in pbar:
            for m in self.models.values():
                m.train()

            self.criterion.set_epoch(step // steps_per_epoch)

            # Sample rays from one random frame; consecutive frames are kept
            # apart (anti-correlated in time) so the static model cannot
            # explain the dynamic object away.
            sample = train_dataset.sample(
                shuffle=True, batch_size=self.cfg.batch_size,
                t_window=self.cfg.get('t_window', 5))

            inputs = self.extract_from_sample(sample)
            inputs["max_t"] = max_t

            self.optimizer.zero_grad()

            results = self.forward(inputs)
            log = self.criterion(results, inputs)
            log = {f'train/{k}': v for k, v in log.items()}

            log['train/psnr'] = psnr(results['rgb_fine'], inputs['rgbs'])
            log['train/loss'] = sum(
                [v for k, v in log.items() if k.endswith('_l')])

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

            # Validation (always the middle frame, so PSNRs are comparable
            # across validations)
            if (step + 1) % val_every == 0 or step == total_steps - 1:
                val_psnr = self._validate(val_dataset, step)
                if val_psnr > best_psnr:
                    self.save_model()
                    best_psnr = val_psnr

    def _validate(self, val_dataset, step):
        for m in self.models.values():
            m.eval()

        with torch.no_grad():
            N_frames = len(val_dataset)
            sample = val_dataset.sample(idx=N_frames // 2)
            inputs = self.extract_from_sample(sample)
            inputs["max_t"] = N_frames - 1

            # no warped renderings at val, but keep the flow heads so the
            # integrated scene flow can be visualized
            results = self.forward(
                inputs, test_time=True, output_transient_flow=('fw', 'bw'))

            val_psnr = psnr(results['rgb_fine'], inputs['rgbs'].cpu())
            self.writer.add_scalar('val/psnr', val_psnr.item(), step)

            H, W = sample['image_size']

            def to_img(t):
                return t.view(H, W, 3).permute(2, 0, 1).clamp(0, 1).cpu()

            img_gt = to_img(sample['rgbs'])
            img_rgb = to_img(results['rgb_fine'])
            img_static = to_img(results['_static_rgb_fine'])
            img_transient = to_img(results['transient_rgb_fine'])

            depth_gt = visualize_depth(inputs['depth'].view(H, W))
            depth = visualize_depth(-results['depth_fine'].view(H, W))
            depth_static = visualize_depth(
                -results['_static_depth_fine'].view(H, W))
            transient_alpha = results['transient_alpha_fine']\
                .view(1, H, W).clamp(0, 1).expand(3, H, W).cpu()

            # 2x4 grid: [gt, composed, static-only, transient] over
            # [disp gt, composed depth, static depth, transient alpha]
            row1 = torch.cat([img_gt, img_rgb, img_static, img_transient], -1)
            row2 = torch.cat(
                [depth_gt, depth, depth_static, transient_alpha], -1)
            grid = torch.cat([row1, row2], -2)
            self.writer.add_image('val/visualization', grid, step)

            save_name = os.path.join(
                self.save_vis_path, f'val_{step:06d}.png')
            save_image(grid, save_name, nrow=2)

            # flow visualization: project the integrated scene flow with the
            # neighboring frames' projection matrices (same as the loss)
            Ps, Ks = inputs['Ps'].cpu(), inputs['Ks'].cpu()
            ts = inputs['rays_t'].cpu()

            def project(xyz_ndc, ts_nb):
                xyz_w = ndc2world(xyz_ndc, Ks)
                P = Ps[ts_nb]
                uvd = P[:, :3, :3] @ xyz_w.unsqueeze(-1) + P[:, :3, 3:]
                return uvd[:, :2, 0] / (torch.abs(uvd[:, 2:, 0]) + 1e-8)

            uv_fw = project(results['xyz_fw'],
                            torch.clamp(ts + 1, max=N_frames - 1))
            uv_bw = project(results['xyz_bw'], torch.clamp(ts - 1, min=0))

            uv = inputs['uv'].cpu()

            def get_img(target):
                diff = (target - uv).view(H, W, 2)
                return flow_to_image(diff.numpy())

            grid_np = np.stack([
                get_img(inputs['uv_fw'].cpu()), get_img(inputs['uv_bw'].cpu()),
                get_img(uv_fw), get_img(uv_bw)])
            grid_tensor = \
                torch.from_numpy(grid_np).permute(0, 3, 1, 2).float()
            if grid_tensor.max() > 1.0:
                grid_tensor /= 255.0
            save_name = os.path.join(
                self.save_vis_path, f'flow_{step:06d}.png')
            save_image(grid_tensor, save_name, nrow=2)

        return val_psnr.item()

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
        rays = torch.cat([rays_o, rays_d], dim=1)
        rays_t = flow_time * torch.ones(len(rays_o), dtype=torch.long)

        inputs = {
            'rays': rays.to(self.device),
            'rays_t': rays_t.to(self.device),
            'max_t': len(val_dataset) - 1,
        }
        with torch.no_grad():
            results = self.forward(
                inputs, test_time=True, output_transient_flow=('fw', 'bw'),
                dataset=val_dataset)

            if dt == 0:
                # exact integer time: use the direct volume rendering
                img = results['rgb_fine'].view(H, W, 3)
                depth = results['depth_fine'].view(H, W)
            else:
                inputs['rays_t'] = inputs['rays_t'] + 1
                results_tp1 = self.forward(
                    inputs, test_time=True,
                    output_transient_flow=('fw', 'bw'), dataset=val_dataset)
                img, depth = interpolate(
                    results, results_tp1, dt,
                    K.to(self.device), c2w.to(self.device),
                    val_dataset.img_wh)

        img = torch.clip(img, 0, 1)
        img = (img.numpy() * 255).astype(np.uint8)
        depth = visualize_depth(-depth.view(H, W))
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

        export_animation(save_dir, imgs, fps=30)
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

    def inference_fixed_time(self, val_dataset, t_fixed=None, n_poses=60):
        """
        Bullet time: freeze the scene at one instant (the kid stops moving)
        and wander the camera around that viewpoint. Uses the same sinusoidal
        wander path as nsff_pl / the original NSFF demos, with amplitude
        1/5 of the trajectory's x-extent.
        """
        N = len(val_dataset)
        if t_fixed is None:
            t_fixed = N // 2
        t_fixed = int(np.clip(t_fixed, 0, N - 1))

        max_trans = np.abs(
            val_dataset.poses[0, 0, 3] - val_dataset.poses[-1, 0, 3]) / 5
        cam_poses = create_wander_path(
            val_dataset.poses[t_fixed], max_trans=max_trans, n_poses=n_poses)

        times = [t_fixed] * n_poses
        self._run_inference(
            val_dataset, cam_poses, times,
            os.path.join(self.log_dir, f"inference_fixtime_{t_fixed:03d}"),
            f"Rendering bullet time @ t={t_fixed} (wander path)")

    def inference_fixed_view(self, val_dataset, view_idx=None, n_frames=None):
        """
        Fixed camera, advancing time: hold one viewpoint and let the kid
        move (with fractional-time interpolation for smooth slow motion).
        """
        N = len(val_dataset)
        if view_idx is None:
            view_idx = N // 2
        view_idx = int(np.clip(view_idx, 0, N - 1))
        if n_frames is None:
            n_frames = 4 * N
        cam_poses = np.tile(val_dataset.poses[view_idx], (n_frames, 1, 1))
        times = np.linspace(0, N - 1, n_frames).tolist()[:-1]
        cam_poses = cam_poses[:len(times)]
        self._run_inference(
            val_dataset, cam_poses, times,
            os.path.join(self.log_dir, f"inference_fixview_{view_idx:03d}"),
            f"Rendering fixed view {view_idx} (moving time)")

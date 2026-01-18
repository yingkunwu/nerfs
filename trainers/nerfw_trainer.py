import os
import torch
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .factory import BaseTrainer
from models.nerfw import NeRFW, Embedding
from losses.nerfw_loss import NeRFWLoss
from utils.nerfw_rendering import render_rays
from utils.metrics import psnr
from utils.misc import visualize_depth


class NeRFWTrainer(BaseTrainer):
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)
        self.criterion = NeRFWLoss(cfg.model.lambda_u)

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
            'appearance': torch.nn.Embedding(
                cfg.N_vocab,
                cfg.model.app_embed_dim),
            'transient': torch.nn.Embedding(
                cfg.N_vocab,
                cfg.model.tra_embed_dim),
        }

        # Models
        model_coarse = NeRFW(
            typ='coarse',
            output_transient=False,
            depth=cfg.model.depth,
            width=cfg.model.width,
            skips=cfg.model.skips,
            in_ch_xyz=embeddings['xyz'].output_dim,
            in_ch_dir=embeddings['dir'].output_dim,
            in_ch_a=cfg.model.app_embed_dim
        )

        model_fine = NeRFW(
            typ='fine',
            output_transient=True,
            depth=cfg.model.depth,
            width=cfg.model.width,
            skips=cfg.model.skips,
            in_ch_xyz=embeddings['xyz'].output_dim,
            in_ch_dir=embeddings['dir'].output_dim,
            in_ch_a=cfg.model.app_embed_dim,
            in_ch_t=cfg.model.tra_embed_dim,
            beta_min=cfg.model.beta_min
        )

        models = {'coarse': model_coarse, 'fine': model_fine}

        return embeddings, models

    def forward(self, inputs):
        """Do batched inference on rays using chunk."""
        rays, rays_t = inputs['rays'], inputs['rays_t']

        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.cfg.chunk):
            results_chunk = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.cfg.chunk],
                rays_t[i:i+self.cfg.chunk],
                self.cfg.N_samples,
                self.cfg.perturb,
                self.cfg.noise_std,
                self.cfg.N_importance,
                self.cfg.white_back)

            for k, v in results_chunk.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        log = self.criterion(results, inputs)

        return results, log

    def extract_from_sample(self, sample, batch_size=None):
        rays = sample['rays'].to(self.device)  # [N_rays, 3]
        rays_t = sample['rays_t'].to(self.device)  # [N_rays,]
        rgbs = sample['rgbs'].to(self.device)  # [N_rgbs, 3]
        num_rgb_rays = rays.shape[0]

        if batch_size is not None:
            idx = torch.randint(0, num_rgb_rays,
                                (batch_size,),
                                device=self.device)
            rays = rays[idx]
            rays_t = rays_t[idx]
            rgbs = rgbs[idx]

        inputs = {
            "rays": rays,
            "rays_t": rays_t,
            "rgbs": rgbs
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

            with ThreadPoolExecutor(max_workers=16) as executor:
                samples = list(executor.map(lambda _: get_sample(), range(16)))

            # a list of 16 sampled batches
            sample = {}
            for key in samples[0].keys():
                if isinstance(samples[0][key], torch.Tensor):
                    sample[key] = torch.cat([s[key] for s in samples], dim=0)
                elif isinstance(samples[0][key], list):
                    sample[key] = []
                    for s in samples:
                        sample[key].extend(s[key])
                else:
                    sample[key] = [s[key] for s in samples]

            inputs = self.extract_from_sample(sample, self.cfg.batch_size)

            # advance the batch pointer
            self.optimizer.zero_grad()

            results, log = self.forward(inputs)

            log['train/psnr_coarse'] = psnr(
                results['rgb_coarse'], inputs['rgbs'])
            if 'rgb_fine' in results:
                log['train/psnr_fine'] = psnr(
                    results['rgb_fine'], inputs['rgbs'])
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
            if 'rgb_fine' in results:
                psnr_val = log['train/psnr_fine'].item()
            else:
                psnr_val = log['train/psnr_coarse'].item()
            pbar.set_postfix({'loss': f'{loss_val:.6f}',
                              'psnr': f'{psnr_val:.3f}'})

            # Validation
            if (step + 1) % 1000 == 0:
                with torch.no_grad():
                    for m in self.models.values():
                        m.eval()
                    sample = val_dataset.sample(shuffle=False)
                    inputs = self.extract_from_sample(sample)

                    results, log = self.forward(inputs)

                    log['val/psnr_coarse'] = psnr(
                        results['rgb_coarse'], inputs['rgbs'])
                    if 'rgb_fine' in results:
                        log['val/psnr_fine'] = psnr(
                            results['rgb_fine'], inputs['rgbs'])
                    log['val/loss'] = sum(
                        [v for k, v in log.items() if 'loss' in k])

                    # TensorBoard logging
                    for k, v in log.items():
                        self.writer.add_scalar(k, v.item(), step)

                    typ = 'fine' if 'rgb_fine' in results else 'coarse'

                    H, W = sample['image_size']

                    img_rgb = results[f'rgb_{typ}']\
                        .view(H, W, 3).cpu().permute(2, 0, 1)
                    img_static = results[f'static_rgb_{typ}']\
                        .view(H, W, 3).cpu().permute(2, 0, 1)
                    img_transient = results[f'transient_rgb_{typ}']\
                        .view(H, W, 3).cpu().permute(2, 0, 1)
                    img_gt = inputs['rgbs']\
                        .view(H, W, 3).permute(2, 0, 1).cpu()

                    depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
                    beta_vis = visualize_depth(  # uncertainty visualization
                        results[f'transient_beta_{typ}'].view(H, W))

                    top_row = torch.cat(
                        [img_gt, img_rgb, depth], dim=2)
                    bottom_row = torch.cat(
                        [img_static, img_transient, beta_vis], dim=2)
                    grid = torch.cat([top_row, bottom_row], dim=1).unsqueeze(0)
                    self.writer.add_images('val/visualization', grid, step)

                    save_name = os.path.join(
                        self.save_vis_path, f'val_{step:06d}.png')
                    save_image(grid, save_name, nrow=3)

                    if log[f'val/psnr_{typ}'].item() > best_psnr:
                        # save model weight
                        self.save_model()
                        best_psnr = log[f'val/psnr_{typ}'].item()

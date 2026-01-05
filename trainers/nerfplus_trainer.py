import os
import torch
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .factory import BaseTrainer
from models.nerf import NeRF, Embedding
from losses.nerf_loss import NeRFLoss
from utils.nerfplus_rendering import render_rays
from utils.metrics import psnr
from utils.misc import visualize_depth


class NeRFPlusPlusTrainer(BaseTrainer):
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)
        self.criterion = NeRFLoss()

    def create_nerf(self, cfg):
        """Create NeRF model and embeddings."""
        # Embeddings
        embeddings = {
            'fg_xyz': Embedding(
                input_dim=3,
                max_freq_log2=cfg.model.xyz_embed_dim - 1,
                num_freqs=cfg.model.xyz_embed_dim,
            ),
            'fg_dir': Embedding(
                input_dim=3,
                max_freq_log2=cfg.model.dir_embed_dim - 1,
                num_freqs=cfg.model.dir_embed_dim,
            ),
            'bg_xyz': Embedding(
                input_dim=4,
                max_freq_log2=cfg.model.xyz_embed_dim - 1,
                num_freqs=cfg.model.xyz_embed_dim,
            ),
            'bg_dir': Embedding(
                input_dim=3,
                max_freq_log2=cfg.model.dir_embed_dim - 1,
                num_freqs=cfg.model.dir_embed_dim,
            ),
        }

        # Models
        models = {
            'fg_coarse': NeRF(
                depth=cfg.model.depth,
                width=cfg.model.width,
                in_ch_xyz=embeddings['fg_xyz'].output_dim,
                in_ch_dir=embeddings['fg_dir'].output_dim,
                skips=cfg.model.skips
            ),
            'fg_fine': NeRF(
                depth=cfg.model.depth,
                width=cfg.model.width,
                in_ch_xyz=embeddings['fg_xyz'].output_dim,
                in_ch_dir=embeddings['fg_dir'].output_dim,
                skips=cfg.model.skips
            ),
            'bg_coarse': NeRF(
                depth=cfg.model.depth,
                width=cfg.model.width,
                in_ch_xyz=embeddings['bg_xyz'].output_dim,
                in_ch_dir=embeddings['bg_dir'].output_dim,
                skips=cfg.model.skips
            ),
            'bg_fine': NeRF(
                depth=cfg.model.depth,
                width=cfg.model.width,
                in_ch_xyz=embeddings['bg_xyz'].output_dim,
                in_ch_dir=embeddings['bg_dir'].output_dim,
                skips=cfg.model.skips
            )
        }

        return embeddings, models

    def forward(self, inputs):
        """Do batched inference on rays using chunk."""
        rays = inputs['rays']

        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.cfg.chunk):
            results_chunk = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.cfg.chunk],
                self.cfg.N_samples,
                self.cfg.perturb,
                self.cfg.noise_std,
                self.cfg.N_importance)

            for k, v in results_chunk.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def fit(self, train_dataset, val_dataset):
        # training hyperparams
        batch_size = int(getattr(self.cfg, "batch_size", 1024))

        print("Starting training loop")
        best_psnr = 0.0
        pbar = tqdm(range(self.cfg.iters), total=self.cfg.iters)
        for step in pbar:
            for m in self.models.values():
                m.train()

            def get_sample():
                return train_dataset.sample(shuffle=True)

            with ThreadPoolExecutor(max_workers=4) as executor:
                samples = list(executor.map(lambda _: get_sample(), range(4)))

            # a list of 4 sampled batches
            sample = {}
            for key in samples[0].keys():
                if isinstance(samples[0][key], torch.Tensor):
                    sample[key] = torch.cat([s[key] for s in samples], dim=0)

            rays = sample['rays'].to(self.device)  # [N_rays, 3]
            rgbs = sample['rgbs'].to(self.device)  # [N_rgbs, 3]
            idx = torch.randint(0, rays.shape[0], (batch_size,),
                                device=self.device)
            rays = rays[idx]
            rgbs = rgbs[idx]

            # advance the batch pointer
            self.optimizer.zero_grad()

            inputs = {'rays': rays}
            targets = {'rgbs': rgbs}

            results = self.forward(inputs)
            log = self.criterion(results, targets)

            log['train/psnr_coarse'] = psnr(
                results['rgb_coarse'], targets['rgbs'])
            if 'rgb_fine' in results:
                log['train/psnr_fine'] = psnr(
                    results['rgb_fine'], targets['rgbs'])
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
                    rays = sample['rays'].to(self.device)  # [N_rays, 3]
                    rgbs = sample['rgbs'].to(self.device)  # [N_rgbs, 3]

                    inputs = {'rays': rays}
                    targets = {'rgbs': rgbs}

                    results = self.forward(inputs)
                    log = self.criterion(results, targets)

                    log['val/psnr_coarse'] = psnr(
                        results['rgb_coarse'], targets['rgbs'])
                    if 'rgb_fine' in results:
                        log['val/psnr_fine'] = psnr(
                            results['rgb_fine'], targets['rgbs'])
                    log['val/loss'] = sum(
                        [v for k, v in log.items() if 'loss' in k])

                    # TensorBoard logging
                    for k, v in log.items():
                        self.writer.add_scalar(k, v.item(), step)

                    typ = 'fine' if 'rgb_fine' in results else 'coarse'

                    # visualization
                    H, W = sample['image_size']
                    img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
                    img = img.permute(2, 0, 1)
                    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
                    depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
                    stack = torch.stack([img_gt, img, depth])
                    self.writer.add_images('val/visualization', stack, step)

                    save_name = os.path.join(
                        self.save_vis_path, f'val_{step:06d}.png')
                    save_image(stack, save_name, nrow=3)

                    # get numpy arrays
                    fg_img_pred = results[f'fg_rgb_{typ}'].view(H, W, 3).cpu()
                    fg_img_pred = fg_img_pred.permute(2, 0, 1)
                    fg_depth_pred = results[f'fg_depth_{typ}'].view(H, W).cpu()
                    bg_img_pred = results[f'bg_rgb_{typ}'].view(H, W, 3).cpu()
                    bg_img_pred = bg_img_pred.permute(2, 0, 1)
                    bg_depth_pred = results[f'bg_depth_{typ}'].view(H, W).cpu()

                    # convert rgb to uint8
                    fg_depth_img = visualize_depth(fg_depth_pred)
                    bg_depth_img = visualize_depth(bg_depth_pred)

                    # concatenate horizontally
                    fg_img = torch.cat([fg_img_pred, fg_depth_img], dim=2)
                    bg_img = torch.cat([bg_img_pred, bg_depth_img], dim=2)
                    img = torch.cat([fg_img, bg_img], dim=1)

                    save_name = os.path.join(
                        self.save_vis_path, f'val_{step:06d}_debug.png')
                    save_image(img, save_name, nrow=3)

                    # save model weight
                    if log[f'val/psnr_{typ}'].item() > best_psnr:
                        self.save_model()
                        best_psnr = log[f'val/psnr_{typ}'].item()

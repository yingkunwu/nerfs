import os
import torch
from collections import defaultdict
from torchvision.utils import save_image
from tqdm import tqdm

from .factory import BaseTrainer
from models.nerf import NeRF, Embedding
from losses.nerf_loss import NeRFLoss, DepthLoss
from utils.nerf_rendering import render_rays
from utils.metrics import psnr
from utils.misc import visualize_depth


class NeRFTrainer(BaseTrainer):
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)
        self.criterion = NeRFLoss()
        self.depth_loss = None
        if cfg.use_depth_loss:
            self.depth_loss = DepthLoss()

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
        }

        # Models
        model_coarse = NeRF(
            depth=cfg.model.depth,
            width=cfg.model.width,
            in_ch_xyz=embeddings['xyz'].output_dim,
            in_ch_dir=embeddings['dir'].output_dim,
            skips=cfg.model.skips
        )

        model_fine = NeRF(
            depth=cfg.model.depth,
            width=cfg.model.width,
            in_ch_xyz=embeddings['xyz'].output_dim,
            in_ch_dir=embeddings['dir'].output_dim,
            skips=cfg.model.skips
        )

        models = {'coarse': model_coarse, 'fine': model_fine}

        return embeddings, models

    def forward(self, inputs):
        """Do batched inference on rays using chunk."""
        rays = inputs['rays']
        rays_batch_size = rays.shape[0]

        if self.depth_loss is not None:
            assert 'depth_rays' in inputs, \
                "depth_rays not found in inputs. Please make sure your data " \
                "folder includes colmap's output and set dataset.load_depth " \
                "to True."

            depth_rays = inputs['depth_rays']
            rays = torch.cat([rays, depth_rays], dim=0)

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
                self.cfg.N_importance,
                self.cfg.white_back)

            for k, v in results_chunk.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        depth_results = defaultdict(list)
        if self.depth_loss is not None:
            for k, v in results.items():
                results[k] = v[:rays_batch_size]
                depth_results[k] = v[rays_batch_size:]

        log = self.criterion(results, inputs)
        if self.depth_loss is not None:
            log.update(self.depth_loss(depth_results, inputs))

        return results, log

    def extract_from_sample(self, sample, batch_size=None):
        rays = sample['rays'].to(self.device)  # [N_rays, 3]
        rgbs = sample['rgbs'].to(self.device)  # [N_rgbs, 3]

        if batch_size is not None:
            idx = torch.randint(0, rays.shape[0],
                                (batch_size,),
                                device=self.device)
            rays = rays[idx]
            rgbs = rgbs[idx]

        inputs = {
            "rays": rays,
            "rgbs": rgbs
        }

        if "depth_rays" in sample:
            depth_rays = sample['depth_rays'].to(self.device)
            depth_values = sample["depth_values"].to(self.device)
            depth_weights = sample["depth_weights"].to(self.device)

            if batch_size is not None:
                idx = torch.randint(0, depth_rays.shape[0],
                                    (batch_size // 4,),
                                    device=self.device)
                depth_rays = depth_rays[idx]
                depth_values = depth_values[idx]
                depth_weights = depth_weights[idx]

            inputs.update({
                "depth_rays": depth_rays,
                "depth_values": depth_values,
                "depth_weights": depth_weights
            })

        return inputs

    def fit(self, train_dataset, val_dataset):
        print("Starting training loop")
        best_psnr = 0.0
        pbar = tqdm(range(self.cfg.iters), total=self.cfg.iters)
        for step in pbar:
            for m in self.models.values():
                m.train()
            # Sample batch
            sample = train_dataset.sample(shuffle=True)
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

                    # visualization
                    H, W = sample['image_size']
                    img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
                    img = img.permute(2, 0, 1)
                    img_gt = inputs['rgbs'].view(H, W, 3).cpu()
                    img_gt = img_gt.permute(2, 0, 1)
                    depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
                    stack = torch.stack([img_gt, img, depth])
                    self.writer.add_images('val/visualization', stack, step)

                    save_name = os.path.join(
                        self.save_vis_path, f'val_{step:06d}.png')
                    save_image(stack, save_name, nrow=3)

                    # save model weight
                    if log[f'val/psnr_{typ}'].item() > best_psnr:
                        self.save_model()
                        best_psnr = log[f'val/psnr_{typ}'].item()

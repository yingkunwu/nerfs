import torch
from torch import nn

from .base_loss import compute_mse


class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        coarse_loss = compute_mse(inputs['rgb_coarse'], targets['rgbs'])
        loss = {'coarse_loss': coarse_loss}

        if 'rgb_fine' in inputs:
            fine_loss = compute_mse(inputs['rgb_fine'], targets['rgbs'])
            loss['fine_loss'] = fine_loss

        return loss


class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_lambda = 0.1

    def forward(self, results, inputs):
        depth_values = inputs['depth_values']
        depth_weights = inputs['depth_weights']

        weights_coarse = results['weights_coarse']
        z_vals_coarse = results['z_vals_coarse']
        deltas_coarse = results['deltas_coarse'][..., :-1]
        # Here we get rid of the last infinite interval and replicate the
        # second last delta to keep the integral approximation reasonable
        # locally.
        last_delta = deltas_coarse[:, -1:]
        deltas_coarse = torch.cat([deltas_coarse, last_delta], dim=-1)

        coarse_loss = -torch.log(weights_coarse + 1e-5) \
            * torch.exp(
                -(z_vals_coarse - depth_values) ** 2 / (2 * depth_weights)
            ) \
            * deltas_coarse
        loss = {'depth_coarse_loss': self.sigma_lambda * coarse_loss.mean()}

        if 'rgb_fine' in results:
            weights_fine = results['weights_fine']
            z_vals_fine = results['z_vals_fine']
            deltas_fine = results['deltas_fine'][..., :-1]
            last_delta = deltas_fine[:, -1:]
            deltas_fine = torch.cat([deltas_fine, last_delta], dim=-1)

            fine_loss = -torch.log(weights_fine + 1e-5) \
                * torch.exp(
                    -(z_vals_fine - depth_values) ** 2 / (2 * depth_weights)
                ) \
                * deltas_fine
            loss['depth_fine_loss'] = self.sigma_lambda * fine_loss.mean()

        return loss

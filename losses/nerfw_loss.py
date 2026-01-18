import torch
from torch import nn

from .base_loss import compute_mse


class NeRFWLoss(nn.Module):
    def __init__(self, lambda_u):
        super().__init__()
        self.lambda_u = lambda_u

    def forward(self, inputs, targets):
        coarse_loss = compute_mse(inputs['rgb_coarse'], targets['rgbs'])
        loss = {'coarse_loss': coarse_loss}

        if 'rgb_fine' in inputs:
            # Compute the transient loss term for the fine output
            mse_term = (inputs['rgb_fine'] - targets['rgbs']) ** 2
            beta_sq = inputs['transient_beta_fine'] ** 2
            transient_loss = (mse_term / (2 * beta_sq[:, None])).mean()
            log_beta = torch.log(inputs['transient_beta_fine']).mean()
            sigma_term = self.lambda_u * inputs['transient_sigma_fine'].mean()

            loss['fine_loss'] = transient_loss + log_beta + sigma_term

        return loss

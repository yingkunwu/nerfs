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

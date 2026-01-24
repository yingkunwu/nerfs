import torch
from torch import nn

from .base_loss import compute_mse, compute_mae, compute_depth_loss
from utils.ray_utils import ndc2world


class NSFFLoss(nn.Module):
    def __init__(self, decay_iteration=30):
        super().__init__()
        self.decay_iteration = decay_iteration
        self.decay_rate = 10

    def forward(self, inputs, targets, global_step):
        # photo consistency loss
        render_loss = compute_mse(inputs['rgb_map_ref'], targets["rgbs"])
        render_loss += compute_mse(inputs['rgb_bw'],
                                   targets["rgbs"],
                                   inputs["prob_ref2prev"].unsqueeze(-1))
        render_loss += compute_mse(inputs['rgb_fw'],
                                   targets["rgbs"],
                                   inputs["prob_ref2post"].unsqueeze(-1))

        # cycle consistency loss
        sf_cycle_loss = 0.1 * compute_mae(
            inputs['raw_sf_ref2prev'],
            -inputs['raw_sf_prev2ref'],
            inputs['raw_prob_ref2prev'].unsqueeze(-1)
        )
        sf_cycle_loss += 0.1 * compute_mae(
            inputs['raw_sf_ref2post'],
            -inputs['raw_sf_post2ref'],
            inputs['raw_prob_ref2post'].unsqueeze(-1)
        )

        # Encourage weight_post and weight_prev to be close to 1
        # try to aboid the trivial solution
        weight_post_loss = torch.mean((1 - inputs['raw_prob_ref2prev']) ** 2)
        weight_prev_loss = torch.mean((1 - inputs['raw_prob_ref2post']) ** 2)
        weight_close_loss = 0.001 * (weight_post_loss + weight_prev_loss)

        # scene flow regularization loss
        N = inputs['raw_pts_ref'].shape[1]
        Ks = targets['Ks']
        xyzs_w = ndc2world(inputs['raw_pts_ref'][:, :int(N * 0.95)], Ks)
        xyzs_fw_w = ndc2world(inputs['raw_xyz_fw'][:, :int(N * 0.95)], Ks)
        xyzs_bw_w = ndc2world(inputs['raw_xyz_bw'][:, :int(N * 0.95)], Ks)

        # temporal smoothness
        reg_temp_sm_loss = 0.1 * torch.mean(
            torch.abs(xyzs_fw_w + xyzs_bw_w - 2 * xyzs_w))

        # min (encourage scene flow to be minimal in most of 3D space)
        reg_min_loss = 0.1 * torch.mean(
            torch.abs(xyzs_fw_w - xyzs_w) + torch.abs(xyzs_bw_w - xyzs_w))

        # spacial smoothness
        d = torch.norm(xyzs_w[:, 1:] - xyzs_w[:, :-1], dim=-1, keepdim=True)
        sp_w = torch.exp(-2 * d)  # weight decreases as the distance increases
        sf_fw_w = xyzs_fw_w - xyzs_w  # forward scene flow in world coordinate
        sf_bw_w = xyzs_bw_w - xyzs_w  # backward scene flow in world coordinate
        reg_sp_sm_loss = 0.1 * torch.mean(
            torch.abs(sf_fw_w[:, 1:] - sf_fw_w[:, :-1]) * sp_w
            + torch.abs(sf_bw_w[:, 1:] - sf_bw_w[:, :-1]) * sp_w)

        # scale-invariant depth loss
        depth_loss = 0.04 * compute_depth_loss(
            inputs['depth_map_ref'], -targets["disp"])

        # scene flow loss
        max_t = targets['max_t']
        xyz_fw_w = ndc2world(inputs['xyz_fw'], Ks)
        xyz_bw_w = ndc2world(inputs['xyz_bw'], Ks)

        ts_bw = torch.clamp(targets['rays_t'] - 1, min=0)
        Ps_bw = targets['Ps'][ts_bw]  # (N_rays, 3, 4)
        uvd_bw = Ps_bw[:, :3, :3] @ xyz_bw_w.unsqueeze(-1) + Ps_bw[:, :3, 3:]
        uv_bw = uvd_bw[:, :2, 0] / (torch.abs(uvd_bw[:, 2:, 0]) + 1e-8)

        ts_fw = torch.clamp(targets['rays_t'] + 1, max=max_t - 1)
        Ps_fw = targets['Ps'][ts_fw]  # (N_rays, 3, 4)
        uvd_fw = Ps_fw[:, :3, :3] @ xyz_fw_w.unsqueeze(-1) + Ps_fw[:, :3, 3:]
        uv_fw = uvd_fw[:, :2, 0] / (torch.abs(uvd_fw[:, 2:, 0]) + 1e-8)

        # disable geo loss for the first and last frames (no gt for fw/bw)
        # also projected depth must > 0 (must be in front of the camera)
        valid_geo_fw = (uvd_fw[:, 2, 0] > 0) & (targets['rays_t'] <= max_t - 1)
        valid_geo_bw = (uvd_bw[:, 2, 0] > 0) & (targets['rays_t'] > 0)

        flow_loss = torch.tensor(0.0, device=uv_fw.device)
        if valid_geo_fw.any():
            flow_loss += 0.02 * compute_mae(uv_fw[valid_geo_fw],
                                            targets['uv_fw'][valid_geo_fw])
        if valid_geo_bw.any():
            flow_loss += 0.02 * compute_mae(uv_bw[valid_geo_bw],
                                            targets['uv_bw'][valid_geo_bw])

        divsor = global_step // (self.decay_iteration * 1000)

        loss = {
            'render_loss': render_loss,
            'sf_cycle_loss': sf_cycle_loss,
            'reg_temp_sm_loss': reg_temp_sm_loss,
            'reg_min_loss': reg_min_loss,
            'reg_sp_sm_loss': reg_sp_sm_loss,
            'weight_close_loss': weight_close_loss,
            'depth_loss': depth_loss / (self.decay_rate ** divsor),
            'flow_loss': flow_loss / (self.decay_rate ** divsor)
        }

        return loss

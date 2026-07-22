import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce, rearrange

from .base_loss import compute_depth_loss
from utils.ray_utils import ndc2world


class NSFFLoss(nn.Module):
    """
    Port of nsff_pl's NeRFWLoss.

    col_l: color loss on the composed (static+transient) rendering
    disp_l: shift/scale-invariant monodepth loss on the composite depth
    entropy_l: entropy loss to concentrate the transient weights
    cross_entropy_l: push static weights away from (dilated) transient peaks;
                     @thickness sets the dilation window (in samples), its
                     weight ramps linearly from 0 to 2e-4 over 10 epochs
    pho_l: forward+backward warped color loss, gated by inferred disocclusion
    cyc_l: forward-backward scene-flow cycle consistency
    flow_fw_l / flow_bw_l: 2D projection of scene flow vs optical flow
    reg_min_l / reg_temp_sm_l / reg_sp_sm_l: scene flow regularizers

    lambda_geo_d / lambda_geo_f decay x0.1 every 10 epochs (set by the
    trainer each step via set_epoch()).
    """

    def __init__(self, lambda_geo=0.04, lambda_reg=0.1, thickness=15):
        super().__init__()
        self.lambda_geo_init = lambda_geo
        self.lambda_geo_d = self.lambda_geo_f = lambda_geo
        self.lambda_reg = lambda_reg
        self.lambda_ent = 1e-3
        self.z_far = 0.95
        assert thickness % 2 == 1, "thickness must be odd (symmetric dilation)"
        self.register_buffer(
            'thickness_filter', torch.ones(1, 1, max(thickness, 1)))
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.lambda_geo_d = self.lambda_geo_init * 0.1 ** (epoch // 10)
        self.lambda_geo_f = self.lambda_geo_init * 0.1 ** (epoch // 10)

    def forward(self, inputs, targets, output_transient_flow=True):
        ret = {}
        ret['col_l'] = reduce((inputs['rgb_fine'] - targets['rgbs']) ** 2,
                              'n1 c -> n1', 'mean').mean()
        ret['disp_l'] = self.lambda_geo_d * compute_depth_loss(
            inputs['depth_fine'], -targets['depth'])

        if not output_transient_flow:
            return ret

        tr_w_raw = inputs['transient_weights_fine']  # (N_rays, N_samples)
        ret['entropy_l'] = self.lambda_ent * reduce(
            -tr_w_raw * torch.log(tr_w_raw + 1e-8), 'n1 n2 -> n1', 'sum'
        ).mean()

        # linearly increase the weight from 0 to lambda_ent/5 in 10 epochs
        cross_entropy_w = self.lambda_ent / 5 * min(self.epoch / 10, 1.0)
        # dilate transient weights with the @thickness window (1D box filter
        # along the sample dimension, zero padding)
        tr_w = tr_w_raw.detach()
        tr_w = F.conv1d(
            rearrange(tr_w, 'n1 n2 -> n1 1 n2'),
            self.thickness_filter,
            padding=self.thickness_filter.shape[-1] // 2)
        tr_w = rearrange(tr_w, 'n1 1 n2 -> n1 n2')
        ret['cross_entropy_l'] = cross_entropy_w * reduce(
            tr_w * torch.log(inputs['static_weights_fine'] + 1e-8),
            'n1 n2 -> n1', 'sum').mean()

        Ks = targets['Ks']  # (3, 3), single camera
        max_t = targets['max_t']
        ts = targets['rays_t']
        xyz_fw_w = ndc2world(inputs['xyz_fw'], Ks)  # (N_rays, 3)
        xyz_bw_w = ndc2world(inputs['xyz_bw'], Ks)  # (N_rays, 3)

        ts_fw = torch.clamp(ts + 1, max=max_t)
        Ps_fw = targets['Ps'][ts_fw]  # (N_rays, 3, 4)
        uvd_fw = Ps_fw[:, :3, :3] @ xyz_fw_w.unsqueeze(-1) + Ps_fw[:, :3, 3:]
        uv_fw = uvd_fw[:, :2, 0] / (torch.abs(uvd_fw[:, 2:, 0]) + 1e-8)

        ts_bw = torch.clamp(ts - 1, min=0)
        Ps_bw = targets['Ps'][ts_bw]  # (N_rays, 3, 4)
        uvd_bw = Ps_bw[:, :3, :3] @ xyz_bw_w.unsqueeze(-1) + Ps_bw[:, :3, 3:]
        uv_bw = uvd_bw[:, :2, 0] / (torch.abs(uvd_bw[:, 2:, 0]) + 1e-8)

        # disable geo loss for the first and last frames (no gt for fw/bw)
        # also projected depth must > 0 (must be in front of the camera)
        valid_geo_fw = (uvd_fw[:, 2, 0] > 0) & (ts < max_t)
        valid_geo_bw = (uvd_bw[:, 2, 0] > 0) & (ts > 0)
        if valid_geo_fw.any():
            ret['flow_fw_l'] = self.lambda_geo_f / 2 * torch.abs(
                uv_fw[valid_geo_fw] - targets['uv_fw'][valid_geo_fw]).mean()
        if valid_geo_bw.any():
            ret['flow_bw_l'] = self.lambda_geo_f / 2 * torch.abs(
                uv_bw[valid_geo_bw] - targets['uv_bw'][valid_geo_bw]).mean()

        # exported for flow visualization during validation/debugging
        inputs['uv_fw'] = uv_fw
        inputs['uv_bw'] = uv_bw

        pho_w = cyc_w = 1.0
        pho_l = pho_w * inputs['disocc_fw'] * \
            (inputs['rgb_fw'] - targets['rgbs']) ** 2 / \
            inputs['disocc_fw'].mean()
        pho_l = pho_l + pho_w * inputs['disocc_bw'] * \
            (inputs['rgb_bw'] - targets['rgbs']) ** 2 / \
            inputs['disocc_bw'].mean()
        ret['pho_l'] = reduce(pho_l, 'n1 c -> n1', 'mean').mean()

        cyc_l = cyc_w * inputs['disoccs_fw'] * \
            torch.abs(inputs['xyzs_fw_bw'] - inputs['xyzs_fine']) / \
            inputs['disoccs_fw'].mean()
        cyc_l = cyc_l + cyc_w * inputs['disoccs_bw'] * \
            torch.abs(inputs['xyzs_bw_fw'] - inputs['xyzs_fine']) / \
            inputs['disoccs_bw'].mean()
        ret['cyc_l'] = reduce(cyc_l, 'n1 n2 c -> n1', 'mean').mean()

        N = inputs['xyzs_fine'].shape[1]
        xyzs_w = ndc2world(inputs['xyzs_fine'][:, :int(N * self.z_far)], Ks)
        xyzs_fw_w = ndc2world(inputs['xyzs_fw'][:, :int(N * self.z_far)], Ks)
        xyzs_bw_w = ndc2world(inputs['xyzs_bw'][:, :int(N * self.z_far)], Ks)

        # temporal smoothness (flow should be linear in time)
        ret['reg_temp_sm_l'] = self.lambda_reg * torch.abs(
            xyzs_fw_w + xyzs_bw_w - 2 * xyzs_w).mean()
        # small flow prior (most of the scene is static)
        ret['reg_min_l'] = self.lambda_reg * (
            torch.abs(xyzs_fw_w - xyzs_w) + torch.abs(xyzs_bw_w - xyzs_w)
        ).mean()

        # spatial smoothness
        d = torch.norm(xyzs_w[:, 1:] - xyzs_w[:, :-1], dim=-1, keepdim=True)
        sp_w = torch.exp(-2 * d)  # weight decreases as the distance increases
        sf_fw_w = xyzs_fw_w - xyzs_w  # forward scene flow in world coordinate
        sf_bw_w = xyzs_bw_w - xyzs_w  # backward scene flow in world coordinate
        ret['reg_sp_sm_l'] = self.lambda_reg * (
            torch.abs(sf_fw_w[:, 1:] - sf_fw_w[:, :-1]) * sp_w
            + torch.abs(sf_bw_w[:, 1:] - sf_bw_w[:, :-1]) * sp_w
        ).mean()

        return ret

import cv2
import torch
import glob
import os
import numpy as np
from scipy.stats import linregress
from PIL import Image
import random

from .factory import DataLoader
from utils.ray_utils import get_ray_directions, get_rays, get_ndc_rays
from utils.colmap_utils import (
    read_cameras_binary, read_images_binary, read_points3d_binary, center_poses
)
from utils.flowlib import read_flow, resize_flow


class DynamicDataLoader(DataLoader):
    def __init__(self,
                 name,
                 root_dir,
                 split='train',
                 resolution=1,
                 start_end=(0, 30)):
        super().__init__()
        assert resolution <= 1, "resolution must be <= 1"
        self.name = name
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.start_frame = start_end[0]
        self.end_frame = start_end[1]

        self.cam_train = [0]
        self.cam_test = 1

        self.frames = []
        self.read_meta()
        self.count = 0

    def _resize_intrinsics(self, K, orig_hw):
        """Downscale intrinsics to match self.resolution (integer factor)."""
        h0, w0 = orig_hw
        if self.resolution <= 1:
            return K.astype(np.float32), (h0, w0)
        new_h = int(h0 * self.resolution)
        new_w = int(w0 * self.resolution)
        Kd = K.copy().astype(np.float32)
        Kd[0, 0] *= self.resolution
        Kd[1, 1] *= self.resolution
        Kd[0, 2] *= self.resolution
        Kd[1, 2] *= self.resolution
        return Kd, (new_h, new_w)

    def read_meta(self):
        # read inputs
        self.image_paths = sorted(
            glob.glob(os.path.join(self.root_dir, 'images/*'))
        )[self.start_frame:self.end_frame]
        self.disp_paths = sorted(
            glob.glob(os.path.join(self.root_dir, 'disps/*'))
        )[self.start_frame:self.end_frame]
        self.mask_paths = sorted(
            glob.glob(os.path.join(self.root_dir, 'masks/*'))
        )[self.start_frame:self.end_frame]
        self.flow_fw_paths = (
            sorted(
                glob.glob(os.path.join(self.root_dir, 'flow_fw/*.flo'))
            )[self.start_frame:self.end_frame] + ['dummy']
        )
        self.flow_bw_paths = (
            ['dummy'] + sorted(
                glob.glob(os.path.join(self.root_dir, 'flow_bw/*.flo'))
            )[self.start_frame:self.end_frame]
        )
        self.N_frames = len(self.image_paths)

        camdata = read_cameras_binary(
            os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        f, cx, cy, _ = camdata[1].params

        self.K = np.array([[f, 0, cx],
                           [0, f, cy],
                           [0,  0,  1]], dtype=np.float32)
        self.K, (new_h, new_w) = self._resize_intrinsics(self.K, (H, W))
        self.img_wh = (new_w, new_h)

        # read extrinsics
        imdata = read_images_binary(
            os.path.join(self.root_dir, 'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])

        w2c_mats = []
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            M = np.eye(4, dtype=R.dtype)
            M[:3, :3] = R
            M[:3,  3] = t.reshape(3)
            w2c_mats.append(M)
        w2c_mats = np.stack(w2c_mats, 0)[perm]
        w2c_mats = w2c_mats[self.start_frame:self.end_frame]
        poses = np.linalg.inv(w2c_mats)[:, :3]

        # read bounds
        pts3d = read_points3d_binary(
            os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        # (1, 3, N_points)
        pts_w = np.zeros((1, 3, len(pts3d)))
        # (N_frames, N_points)
        visibilities = np.zeros((len(poses), len(pts3d)))
        for i, k in enumerate(pts3d):
            pts_w[0, :, i] = pts3d[k].xyz
            for j in pts3d[k].image_ids:
                if self.start_frame <= j-1 < self.end_frame:
                    visibilities[j-1-self.start_frame, i] = 1

        min_depth = 1e8
        for i in range(self.N_frames):
            # For each image, compute the nearest depth according to real
            # depth from COLMAP and the disparity estimated by monodepth.
            # Use linear regression to find the best scale and shift.
            disp = cv2.imread(
                self.disp_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float32)
            disp = cv2.resize(
                disp, self.img_wh, interpolation=cv2.INTER_NEAREST)

            pts_xyz = pts_w[0]  # (3, N_points)
            ones_row = np.ones((1, pts_xyz.shape[1]))  # (1, N_points)
            pts_w_homo = np.vstack([pts_xyz, ones_row])  # (4, N_points)

            visibility_i = visibilities[i]  # (N_points) 1 if visible
            pts_w_v = pts_w_homo[:, visibility_i == 1]  # (4, N_points_v)
            pts_c_v = (w2c_mats[i] @ pts_w_v)[:3]  # (3, N_points_v)
            pts_uvd_v = self.K @ pts_c_v
            pts_uv_v = (pts_uvd_v[:2] / pts_uvd_v[2:]).T  # (N_points_v, 2)
            pts_uv_v = pts_uv_v.astype(int)  # to integer pixel coordinates
            pts_uv_v[:, 0] = np.clip(pts_uv_v[:, 0], 0, self.img_wh[0] - 1)
            pts_uv_v[:, 1] = np.clip(pts_uv_v[:, 1], 0, self.img_wh[1] - 1)
            pts_d_v = pts_uvd_v[2]

            # disp = a * (1 / z) + b -> solve a and b
            # use this regression to estimate the scale and shift of monodepth
            # predictions
            try:
                y_vals = disp[pts_uv_v[:, 1], pts_uv_v[:, 0]]
                x_vals = 1 / pts_d_v
                reg = linregress(x_vals, y_vals)
            except Exception:
                reg = None

            if reg is not None and reg.rvalue ** 2 > 0.9:
                # z = a / (disp - b)
                d = np.percentile(disp, 95)
                scale_est = reg.slope / (d - reg.intercept)
                min_depth = min(min_depth, scale_est)
            else:
                min_depth = min(min_depth, np.percentile(pts_d_v, 5))

        # correct poses
        # change "right down front" of COLMAP to "right up back"
        self.poses = np.concatenate(
            [poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses = center_poses(self.poses)

        # correct scale so that the nearest depth is at a little more than 1.0
        self.scale_factor = min_depth * 0.75
        self.poses[..., 3] /= self.scale_factor

        # create projection matrix, used to compute optical flow
        bottom = np.zeros((self.N_frames, 1, 4))
        bottom[..., -1] = 1
        rt = np.linalg.inv(np.concatenate([self.poses, bottom], 1))[:, :3]
        rt[:, 1:] *= -1  # change to "right up back"

        self.Ps = self.K @ rt
        self.Ps = torch.from_numpy(self.Ps).to(torch.float32)
        self.Ks = torch.from_numpy(self.K).to(torch.float32)

        self.frames = list(range(len(self.poses)))

        self.directions, self.uv = get_ray_directions(
            self.img_wh[1], self.img_wh[0], self.K, return_uv=True)

    def __len__(self):
        return len(self.frames)

    def sample(self, shuffle=False, idx=None):
        if shuffle:
            i = random.choice(self.frames)
        else:
            if self.count >= len(self.frames):
                self.count = 0
            i = self.count
            self.count += 1

        if idx is not None:
            i = idx

        c2w = torch.from_numpy(self.poses[i]).to(torch.float32)
        rays_o, rays_d = get_rays(self.directions, c2w)
        rays_o, rays_d = get_ndc_rays(self.K, 1.0, rays_o, rays_d)

        rays_t = i * torch.ones(len(rays_o), dtype=torch.long)  # (h*w)

        # since we are using ndc rays, the near and far interval is limited to
        # [0, 1] by default
        near, far = 0, 1
        near_ = near * torch.ones_like(rays_o[:, :1])
        far_ = far * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near_, far_], dim=1)

        sample = {'img_i': i, 'rays': rays, 'rays_t': rays_t, 'c2w': c2w}

        img = Image.open(self.image_paths[i]).convert('RGB')
        if self.resolution < 1:
            img = img.resize(self.img_wh, Image.LANCZOS)

        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        img_t = img_t.view(self.img_wh[1] * self.img_wh[0], 3)
        sample['rgbs'] = img_t

        disp = cv2.imread(
            self.disp_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
        sample['disp'] = torch.from_numpy(disp.flatten())

        if i < self.N_frames - 1:
            flow_fw = read_flow(self.flow_fw_paths[i])
            flow_fw = resize_flow(
                flow_fw, self.img_wh[0], self.img_wh[1])
            flow_fw = torch.from_numpy(flow_fw)
        else:
            flow_fw = torch.zeros(self.img_wh[1], self.img_wh[0], 2)
        sample['uv_fw'] = self.uv + flow_fw
        sample['uv_fw'] \
            = sample['uv_fw'].view(self.img_wh[1] * self.img_wh[0], 2)

        if i >= 1:
            flow_bw = read_flow(self.flow_bw_paths[i])
            flow_bw = resize_flow(
                flow_bw, self.img_wh[0], self.img_wh[1])
            flow_bw = torch.from_numpy(flow_bw)
        else:
            flow_bw = torch.zeros(self.img_wh[1], self.img_wh[0], 2)
        sample['uv_bw'] = self.uv + flow_bw
        sample['uv_bw'] \
            = sample['uv_bw'].view(self.img_wh[1] * self.img_wh[0], 2)

        sample['Ks'] = self.Ks.clone()
        sample['Ps'] = self.Ps.clone()
        sample['image_size'] = (self.img_wh[1], self.img_wh[0])

        return sample

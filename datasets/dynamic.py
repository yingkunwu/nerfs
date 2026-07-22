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
        self.last_t = -1

    def _resize_intrinsics(self, K, orig_hw):
        """Downscale intrinsics to match self.resolution (integer factor)."""
        h0, w0 = orig_hw
        if self.resolution >= 1:
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
            glob.glob(os.path.join(
                self.root_dir, 'images_undistorted/images/*'))
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
            # what we read from file is actually depth

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

            # the files in disps/ hold DISPARITY (inverse depth) up to an
            # unknown scale and shift: disp = a * (1/depth) + b.
            # Regress against the COLMAP inverse depths to recover a and b,
            # then convert the 95th disparity percentile (= nearest content)
            # back to a metric depth.
            try:
                y_vals = disp[pts_uv_v[:, 1], pts_uv_v[:, 0]]
                x_vals = 1.0 / pts_d_v
                reg = linregress(x_vals, y_vals)
            except Exception:
                reg = None

            if reg is not None and reg.rvalue ** 2 > 0.9:
                # depth = a / (disp - b)
                d = np.percentile(disp, 95)
                depth_est = reg.slope / (d - reg.intercept)
                min_depth = min(min_depth, depth_est)
            else:
                min_depth = min(min_depth, np.percentile(pts_d_v, 5))

        # change "right down front" of COLMAP to "right up back"
        self.poses = np.concatenate(
            [poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)

        # recenter poses
        self.poses = center_poses(self.poses)

        # correct scale so that the nearest depth is at a little more than 1.0
        self.scale_factor = min_depth * 0.75
        self.poses[..., 3] /= self.scale_factor

        # create projection matrix, used to compute optical flow
        bottom = np.zeros((self.N_frames, 1, 4))
        bottom[..., -1] = 1
        rt = np.linalg.inv(np.concatenate([self.poses, bottom], 1))[:, :3]
        rt[:, 1:] *= -1  # change back to c2w under opencv coordinate system
        # Since we are comparing optical flow on the opencv coordinate system,
        # we require the opencv representations for world2camera projection.

        self.Ps = self.K @ rt
        self.Ps = torch.from_numpy(self.Ps).to(torch.float32)
        self.Ks = torch.from_numpy(self.K).to(torch.float32)

        self.frames = list(range(len(self.poses)))

        self.directions, self.uv = get_ray_directions(
            self.img_wh[1], self.img_wh[0], self.K, return_uv=True)
        self.uv = self.uv.reshape(-1, 2)

        # Prebuild per-frame ray buffers once (loading images/flows/disps from
        # disk at every training step is prohibitively slow).
        self.rays_dict = {}
        self.dynamic_idxs = {}
        for t in range(self.N_frames):
            self.rays_dict[t] = self._build_frame_buffer(t)

    def _build_frame_buffer(self, t):
        """Build all per-ray tensors for frame t."""
        w, h = self.img_wh

        c2w = torch.from_numpy(self.poses[t]).to(torch.float32)
        rays_o, rays_d = get_rays(self.directions, c2w)
        # if the camera lies in front of the global near plane (z < -1),
        # shift ray origins to the camera plane instead so they do not start
        # behind the camera
        shift_near = -min(-1.0, float(self.poses[t, 2, 3]))
        rays_o, rays_d = get_ndc_rays(
            self.K, 1.0, rays_o, rays_d, shift_near=shift_near)

        # since we are using ndc rays, the near and far interval is limited to
        # [0, 1] by default
        near_ = torch.zeros_like(rays_o[:, :1])
        far_ = torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near_, far_], dim=1)

        img = Image.open(self.image_paths[t]).convert('RGB')
        if self.resolution < 1:
            img = img.resize(self.img_wh, Image.LANCZOS)
        rgbs = torch.from_numpy(np.array(img)).float() / 255.0
        rgbs = rgbs.view(h * w, 3)

        disp = cv2.imread(
            self.disp_paths[t], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
        depth = torch.from_numpy(disp.flatten())

        # motion mask; in this dataset 0 = dynamic region, 255 = static
        mask = Image.open(self.mask_paths[t]).convert('L')
        mask = mask.resize(self.img_wh, Image.NEAREST)
        dynamicness = 1.0 - torch.from_numpy(
            np.array(mask)).float().flatten() / 255.0
        self.dynamic_idxs[t] = torch.nonzero(
            dynamicness > 0.5, as_tuple=False).squeeze(-1)
        mask = dynamicness

        if t < self.N_frames - 1:
            flow_fw = read_flow(self.flow_fw_paths[t])
            flow_fw = resize_flow(flow_fw, w, h)
            flow_fw = torch.from_numpy(flow_fw).view(h * w, 2)
        else:
            flow_fw = torch.zeros(h * w, 2)
        uv_fw = self.uv + flow_fw

        if t >= 1:
            flow_bw = read_flow(self.flow_bw_paths[t])
            flow_bw = resize_flow(flow_bw, w, h)
            flow_bw = torch.from_numpy(flow_bw).view(h * w, 2)
        else:
            flow_bw = torch.zeros(h * w, 2)
        uv_bw = self.uv + flow_bw

        return {
            'rays': rays,
            'rgbs': rgbs,
            'depth': depth,
            'mask': mask,  # 1 = dynamic region, 0 = static
            'uv_fw': uv_fw,
            'uv_bw': uv_bw,
        }

    def __len__(self):
        return len(self.frames)

    def sample(self, shuffle=False, idx=None, batch_size=None, num_extra=0,
               t_window=None):
        """
        Sample rays from one frame.

        Inputs:
            shuffle: pick a random frame, otherwise iterate sequentially
            idx: force a specific frame
            batch_size: if None, return all rays of the frame (full image);
                otherwise return @batch_size uniformly sampled rays
            num_extra: number of extra rays sampled from the dynamic region
                (motion mask), appended AFTER the uniform rays. Used for hard
                mining in early training. Only valid with batch_size.
            t_window: if set (with shuffle), the next frame is sampled
                OUTSIDE +-t_window of the previously sampled frame, so the
                static model cannot explain the dynamic object away
                (anti-correlated time sampling, cf. nsff_pl).
        """
        if shuffle:
            if t_window is None or self.last_t == -1:
                i = random.choice(self.frames)
            else:
                valid_t = list(
                    set(self.frames)
                    - set(range(self.last_t - t_window,
                                self.last_t + t_window + 1)))
                i = random.choice(valid_t)
            self.last_t = i
        else:
            if self.count >= len(self.frames):
                self.count = 0
            i = self.count
            self.count += 1

        if idx is not None:
            i = idx

        buf = self.rays_dict[i]
        N_rays = buf['rays'].shape[0]

        if batch_size is None:
            sel = torch.arange(N_rays)
        else:
            sel = torch.randint(0, N_rays, (batch_size,))
            if num_extra > 0 and len(self.dynamic_idxs[i]) > 0:
                dyn = self.dynamic_idxs[i]
                extra = dyn[torch.randint(0, len(dyn), (num_extra,))]
                sel = torch.cat([sel, extra])

        c2w = torch.from_numpy(self.poses[i]).to(torch.float32)
        rays_t = i * torch.ones(len(sel), dtype=torch.long)

        sample = {
            'img_i': i,
            'c2w': c2w,
            'rays': buf['rays'][sel],
            'rays_t': rays_t,
            'rgbs': buf['rgbs'][sel],
            'depth': buf['depth'][sel],
            'mask': buf['mask'][sel],
            'uv_fw': buf['uv_fw'][sel],
            'uv_bw': buf['uv_bw'][sel],
            'uv': self.uv[sel],
            'Ks': self.Ks.clone(),
            'Ps': self.Ps.clone(),
            'image_size': (self.img_wh[1], self.img_wh[0]),
        }

        return sample

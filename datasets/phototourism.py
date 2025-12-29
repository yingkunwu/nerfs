import os
import pandas as pd
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import glob

from .factory import DataLoader
from utils.ray_utils import get_ray_directions, get_rays
from utils.colmap_utils import (
    read_cameras_binary, read_images_binary, read_points3d_binary
)


class PhototourismDataLoader(DataLoader):
    def __init__(self, root_dir, split, resolution=1):
        super().__init__()
        assert resolution <= 1, "resolution must be <= 1"
        self.root_dir = Path(root_dir)
        self.split = split
        self.resolution = resolution

        # State for round-robin sampling
        self.frames = []
        self.count = 0

        # File lists (parallel arrays by index)
        self.img_files = []
        self.ids = []

        # COLMAP content
        self._camdata = None
        self._imdata = None
        self._pts3d = None

        # Per-frame near/far (by frame index)
        self._nears = []
        self._fars = []

        self._discover_scene()
        self._load_colmap()
        self._build_index()
        self._compute_near_far()

        if split == 'val':
            # Use only first 20 images for validation
            self.frames = list(range(len(self.img_files[:20])))
        else:
            self.frames = list(range(len(self.img_files)))

    def _discover_scene(self):
        self.images_dir = self.root_dir / "dense" / "images"
        self.sparse_dir = self.root_dir / "dense" / "sparse"
        for p in (self.images_dir, self.sparse_dir):
            if not p.exists():
                raise FileNotFoundError(f"Expected path missing: {p}")

        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()]
        self.files.reset_index(inplace=True, drop=True)

    def _load_colmap(self):
        self._camdata = read_cameras_binary(
            str(self.sparse_dir / "cameras.bin")
        )
        self._imdata = read_images_binary(
            str(self.sparse_dir / "images.bin")
        )
        self._pts3d = read_points3d_binary(
            str(self.sparse_dir / "points3D.bin")
        )

        # Build mapping from image filename -> COLMAP id
        self._name_to_id = {}
        for v in self._imdata.values():
            self._name_to_id[v.name] = v.id

    def _build_index(self):
        for filename in list(self.files['filename']):
            im_id = self._name_to_id[filename]
            self.ids.append(im_id)
            self.img_files.append(str(self.images_dir / filename))

        if len(self.img_files) == 0:
            raise RuntimeError(
                f"No images for split in {self.images_dir}"
            )

    def _to_homo(self, xyz):
        return np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)

    def _colmap_intrinsics(self, im_id):
        """Return K (3x3) and the original (H, W) for this image id."""
        cam = self._camdata[self._imdata[im_id].camera_id]
        K = np.zeros((3, 3), dtype=np.float32)
        img_w = int(cam.params[2] * 2)
        img_h = int(cam.params[3] * 2)
        K[0, 0] = cam.params[0]
        K[1, 1] = cam.params[1]
        K[0, 2] = cam.params[2]
        K[1, 2] = cam.params[3]
        K[2, 2] = 1.0
        return K, (img_h, img_w)

    def _colmap_pose_w2c(self, im_id):
        im = self._imdata[im_id]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3:] = t
        return w2c

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

    def _compute_near_far(self):
        """Compute per-frame near/far from triangulated 3D points."""
        if len(self._pts3d) == 0:
            self._nears = [0.1] * len(self.img_files)
            self._fars = [5.0] * len(self.img_files)
            return

        xyz_world = np.array([p.xyz for p in self._pts3d.values()],
                             dtype=np.float32)
        xyz_world_h = self._to_homo(xyz_world)

        self._nears = []
        self._fars = []
        for im_id in self.ids:
            w2c = self._colmap_pose_w2c(im_id)
            xyz_cam = (xyz_world_h @ w2c.T)[:, :3]
            xyz_cam = xyz_cam[xyz_cam[:, 2] > 0]
            assert len(xyz_cam) > 0, f"Empty xyz_cam for image {im_id}"
            near = float(np.percentile(xyz_cam[:, 2], 0.1))
            far = float(np.percentile(xyz_cam[:, 2], 99.9))
            self._nears.append(near)
            self._fars.append(far)

        max_far = max(self._fars) if self._fars else 5.0
        self._scale = (max_far / 5.0) if max_far > 0 else 1.0
        self._nears = [n / self._scale for n in self._nears]
        self._fars = [f / self._scale for f in self._fars]

    def __len__(self):
        return len(self.frames)

    def sample(self, shuffle=False, idx=None):
        """
        Returns a dict with the TNT keys:
          - image_path: str
          - pose: (3,4) torch.FloatTensor (camera-to-world, scaled)
          - rays: (H*W, 8) torch.FloatTensor [r_o(3), r_d(3), near(1), far(1)]
          - rgbs: (H*W, 3) torch.FloatTensor in [0,1]
          - raw_d: (H*W, 3) torch.FloatTensor (ray directions in camera frame)
          - image_size: (H, W)
          - rays_t: (H*W) torch.LongTensor (containing the frame index)
        """
        if shuffle:
            i = random.choice(self.frames)
        else:
            if self.count >= len(self.frames):
                self.count = 0
            i = self.count
            self.count += 1

        if idx is not None:
            i = idx

        im_id = self.ids[i]
        img_path = self.img_files[i]

        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        K, (orig_h, orig_w) = self._colmap_intrinsics(im_id)
        assert (orig_h, orig_w) == (h0, w0), (
            f"COLMAP size {(orig_w, orig_h)} != image size "
            f"{(w0, h0)} for {img_path}"
        )

        Kd, (new_h, new_w) = self._resize_intrinsics(K, (orig_h, orig_w))
        if self.resolution < 1:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        img_t = img_t.view(new_h * new_w, 3)

        w2c = torch.from_numpy(self._colmap_pose_w2c(im_id)).float()
        c2w = torch.linalg.inv(w2c)[:3, :4]
        c2w[:, 3] /= float(self._scale)  # Scale the translation

        raw_d = get_ray_directions(new_h, new_w, Kd)
        rays_o, rays_d = get_rays(raw_d, c2w)
        near = float(self._nears[i])
        far = float(self._fars[i])
        near_t = torch.full((rays_o.shape[0], 1), near, dtype=rays_o.dtype)
        far_t = torch.full((rays_o.shape[0], 1), far, dtype=rays_o.dtype)
        rays = torch.cat([rays_o, rays_d, near_t, far_t], dim=1)

        return {
            "image_path": img_path,
            "pose": c2w,
            "rays": rays,
            "rgbs": img_t,
            "raw_d": raw_d,
            "image_size": (new_h, new_w),
            'rays_t': i * torch.ones(len(rays), dtype=torch.long)
        }

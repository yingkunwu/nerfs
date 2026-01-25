import os
import random
import glob
import numpy as np
from PIL import Image
import torch

from .factory import DataLoader
from utils.ray_utils import get_ray_directions, get_rays


def convert_pose(C2W):
    flip_yz = torch.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


class TNTDataLoader(DataLoader):
    def __init__(self, name, root_dir, split, resolution=1):
        super().__init__()
        assert resolution <= 1, "resolution must be <= 1"
        self.name = name
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution

        assert split in ['train', 'val'], "split must be 'train' or 'val'"
        self.img_files = self._find_files(
            f'{root_dir}/{split}/images', exts=['*.png', '*.jpg'])
        self.intrinsics_files = self._find_files(
            f'{root_dir}/{split}/intrinsics', exts=['*.txt'])
        self.extrinsics_files = self._find_files(
            f'{root_dir}/{split}/extrinsics', exts=['*.txt'])

        self.frames = range(len(self.img_files))
        self.count = 0

        self.near = 0.01
        self.far = 1

    def __len__(self):
        return len(self.frames)

    def _find_files(self, dir, exts):
        if os.path.isdir(dir):
            # types should be ['*.png', '*.jpg']
            files_grabbed = []
            for ext in exts:
                files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
            if len(files_grabbed) > 0:
                files_grabbed = sorted(files_grabbed)
            return files_grabbed
        else:
            return []

    def _parse_txt(self, filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4])

    def sample(self, shuffle=False, idx=None):
        """
        Returns a dict with the TNT keys:
          - image_path: str
          - pose: (3,4) torch.FloatTensor (camera-to-world, scaled)
          - rays: (H*W, 8) torch.FloatTensor [r_o(3), r_d(3), near(1), far(1)]
          - rgbs: (H*W, 3) torch.FloatTensor in [0,1]
          - image_size: (H, W)
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

        intrinsics = self._parse_txt(self.intrinsics_files[i])
        extrinsics = self._parse_txt(self.extrinsics_files[i])
        # w2c is the camera extrinsic matrix
        w2c = torch.from_numpy(extrinsics).float()
        c2w = torch.linalg.inv(w2c)
        c2w = convert_pose(c2w)[:3]

        img = Image.open(self.img_files[i])
        w, h = img.size  # PIL: (width, height)
        new_w = int(w * self.resolution)
        new_h = int(h * self.resolution)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        # scale intrinsics to match resized image
        K = intrinsics[:3, :3]
        K[0, 0] *= self.resolution
        K[0, 2] *= self.resolution
        K[1, 1] *= self.resolution
        K[1, 2] *= self.resolution
        K = torch.from_numpy(K).float()

        directions = get_ray_directions(new_h, new_w, K)
        rays_o, rays_d = get_rays(directions, c2w)
        near = self.near * torch.ones_like(rays_o[:, :1])
        far = self.far * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near, far], dim=1)

        return {
            'image_path': self.img_files[i],
            'pose': c2w,
            'rays': rays,
            'rgbs': img,
            'image_size': (new_h, new_w),
            'intrinsics': K
        }

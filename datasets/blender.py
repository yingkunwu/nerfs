import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import torch

from .factory import DataLoader
from utils.ray_utils import get_ray_directions, get_rays


def add_perturbation(img, perturbation, seed):
    """Apply color and occlusion perturbations to a PIL Image.

    perturbation: iterable containing 'color' and/or 'occ'.
    seed: int seed for deterministic random perturbations.
    """
    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img) / 255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s * img_np[..., :3] + b, 0, 1)
        img = Image.fromarray((255 * img_np).astype(np.uint8))

    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)

        for i in range(10):
            np.random.seed(10 * seed + i)
            random_color = tuple(np.random.choice(range(256), 3))

            x0 = left + 20 * i
            x1 = left + 20 * (i + 1)
            y1 = top + 200
            draw.rectangle(((x0, top), (x1, y1)), fill=random_color)

    return img


class BlenderDataLoader(DataLoader):
    def __init__(self, name, root_dir, split, resolution=1):
        super().__init__()
        assert resolution <= 1, "resolution must be <= 1"
        self.name = name
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.read_meta(root_dir, split, resolution)

        self.frames = range(len(self.meta['frames']))
        self.count = 0

    def __len__(self):
        return len(self.meta['frames'])

    def read_meta(self, root_dir, split, resolution):
        meta_path = os.path.join(root_dir, f'transforms_{split}.json')
        with open(meta_path, 'r') as meta_file:
            self.meta = json.load(meta_file)

        angle_x = self.meta['camera_angle_x']
        self.focal = 0.5 * 800 / np.tan(0.5 * angle_x)
        self.focal *= resolution

        self.W = int(800 * resolution)
        self.H = int(800 * resolution)

        # Assumes a simple pinhole camera with square pixels, zero skew, and
        # the principal point at image center.
        # Under those assumptions the intrinsic matrix is:
        # K =
        # [[f & 0 & W / 2]
        #  [0 & f & H / 2]
        #  [0 & 0 & 1    ]]
        self.K = np.array([
            [self.focal, 0.0, self.W / 2.0],
            [0.0, self.focal, self.H / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

    def sample(self, shuffle=False, idx=None, perturbation=[]):
        """
        Returns a dict with the TNT keys:
          - image_path: str
          - pose: (3,4) torch.FloatTensor (camera-to-world, scaled)
          - rays: (H*W, 8) torch.FloatTensor [r_o(3), r_d(3), near(1), far(1)]
          - rgbs: (H*W, 3) torch.FloatTensor in [0,1]
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

        frame = self.meta['frames'][i]
        pose = np.array(frame['transform_matrix'])[:3, :4]
        c2w = torch.from_numpy(pose).float()
        image_path = os.path.join(self.root_dir, f'{frame["file_path"]}.png')

        img = Image.open(image_path)
        if len(perturbation) > 0:
            img = add_perturbation(img, perturbation, i)

        new_size = (self.W, self.H)
        img = img.resize(new_size, Image.LANCZOS)
        img = self.transform(img)
        img = img.view(4, -1).permute(1, 0)
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])

        directions = get_ray_directions(self.H, self.W, self.K)
        # Convert from Blender coordinates (OpenGL) to opencv convention
        # : flip y and z axes
        directions = directions * torch.tensor([1.0, -1.0, -1.0]).float()
        rays_o, rays_d = get_rays(directions, c2w)
        near = self.near * torch.ones_like(rays_o[:, :1])
        far = self.far * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near, far], dim=1)

        return {
            'image_path': image_path,
            'pose': c2w,
            'rays': rays,  # shape (H*W, 8)
            'rgbs': img,   # shape (H*W, 3)
            'image_size': (self.H, self.W),
            'rays_t': i * torch.ones(len(rays), dtype=torch.long)  # for nerfw
        }

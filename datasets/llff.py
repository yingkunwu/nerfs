import os
import random
import glob
import numpy as np
from PIL import Image
import torch

from .factory import DataLoader
from utils.ray_utils import get_ray_directions, get_rays, get_ndc_rays


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate

    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4)

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered


class LIFFDataLoader(DataLoader):
    def __init__(self, root_dir, split, resolution=1):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution

        assert split in ['train', 'val'], 'split must be either train or val'

        poses_bounds = np.load(os.path.join(
            self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(
            os.path.join(self.root_dir, 'images_undistorted/images/*')))

        assert len(poses_bounds) == len(self.image_paths), \
            'Mismatch between number of images and number of poses! ' \
            'Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics

        self.img_wh = (int(W * self.resolution), int(H * self.resolution))
        self.focal *= self.img_wh[0] / W

        # Original poses has rotation in form "down right back",
        # change to "right up back"
        # [-u, r, -t] -> [r, u, -t]
        poses = np.concatenate(
            [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        self.poses = center_poses(poses)

        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        # 這邊的意思就是找到整個場景中最接近相機的點，根據這個點的深度來調整整個場景的尺度
        # 調整過後這個最接近相機的點的深度就會位於 1.0 / 0.75 = 1.3333... 的位置
        # 因此這個時候相機的姿態也要跟著做調整
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        self.K = np.array([
            [self.focal, 0.0, self.img_wh[0] / 2.0],
            [0.0, self.focal, self.img_wh[1] / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        if self.split == 'train':
            self.frames = range(1, len(self.image_paths))
        else:
            self.frames = [0]
        self.count = 0

        # assuming that we are using normalized device coordinate (NDC)
        self.near = 0
        self.far = 1

    def __len__(self):
        return len(self.frames)

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

        c2w = torch.from_numpy(self.poses[i]).to(torch.float32)

        img = Image.open(self.image_paths[i])
        new_w, new_h = self.img_wh
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

        # scale intrinsics to match resized image
        K = torch.from_numpy(self.K).float()

        directions = get_ray_directions(new_h, new_w, K)
        rays_o, rays_d = get_rays(directions, c2w)

        # 因為我們在準備資料的時候就把整個空間最接近相機的點調整到深度1.3333...的位置了
        # 所以這邊我們在生成ndc ray的時候，可以直接把ndc場景的near設為1.0，讓所有射線
        # 都是以near plane為原點向外部射出，這樣就能確保整個場景都是在ndc的範圍內。
        # 也就是說當我們在做ndc ray生成的時候，如果確保了ndc space的near plane位置在
        # 1.0，那麼我們在ndc space做sampling時就不會有任何的ray在near plane之前被截
        # 斷掉，同時也不會距離near plane太遠而導致數值不穩定的問題。
        # 注意這裏的near跟nerf的near是不同的概念！
        # nerf的near是用來做sampling的，而這裡的near是ndc空間的near plane位置。
        # 根據ndc ray的轉換，當我們在ndc空間做 [near, far]=[0, 1] 的sampling時，
        # 實際上對應到原本的世界座標系統會是在 [0, 無限大] 的範圍內做sampling。
        rays_o, rays_d = get_ndc_rays(self.K, 1.0, rays_o, rays_d)
        near = self.near * torch.ones_like(rays_o[:, :1])
        far = self.far * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near, far], dim=1)

        return {
            'image_path': self.image_paths[i],
            'pose': c2w,
            'rays': rays,
            'rgbs': img,
            'image_size': (new_h, new_w),
        }

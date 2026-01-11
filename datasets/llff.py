import os
import random
import glob
import numpy as np
from PIL import Image
import torch

from .factory import DataLoader
from utils.ray_utils import get_ray_directions, get_rays, get_ndc_rays
from utils.colmap_utils import (
    center_poses, read_images_binary, read_points3d_binary
)


def get_poses(images):
    w2c_list = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_list.append(w2c)
    return np.array(w2c_list)


def load_colmap_depth(basedir, bounds, sc):
    """
    Load per-pixel depths from COLMAP sparse reconstruction and return
    per-image depth/coordinate/error arrays.

    This function reads COLMAP binary files (images.bin and points3D.bin) from
    a dataset directory, projects 3D points into each image to obtain depth
    samples, filters samples by per-image depth bounds, computes a confidence
    weight per sample from COLMAP reprojection errors, and aggregates results
    into a list of per-image dictionaries.

    Parameters
    - basedir (str):
            Path to the COLMAP dataset folder that contains the
            'sparse/0/images.bin' and 'sparse/0/points3D.bin' files.
    - bounds (np.ndarray):
            Array of shape (N_images, 2) providing per-image [near, far] depth
            bounds in the same scale as the original reconstruction.
            bounds[i, 0] is the near bound and bounds[i, 1] is the far bound
            for image i.
    - sc (float):
            Scale factor used to normalize the depth values from the original
            reconstruction scale.

    Return value
    - list of dicts, one entry per image that had at least one valid sample.
        Each dict has keys:
            - "depth": np.ndarray of shape (M,) containing scaled depths for
                the M kept samples
            - "coord": np.ndarray of shape (M, 2) containing pixel coordinates
                for each sample
            - "error": np.ndarray of shape (M,) containing the computed
                confidence weight per sample

    """
    images_path = os.path.join(basedir, 'sparse', '0', 'images.bin')
    points_path = os.path.join(basedir, 'sparse', '0', 'points3D.bin')

    images = read_images_binary(images_path)
    points = read_points3d_binary(points_path)

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    poses_inv = get_poses(images)  # w2c -> inverse of the camera pose
    print("Depth points' statistics from Colmap:")

    data_list = []
    for id_im in range(1, len(images) + 1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D_world = points[id_3D].xyz

            point3D_hom = np.array([*point3D_world, 1]).reshape(4, 1)
            point3D_cam = poses_inv[id_im - 1, :3] @ point3D_hom
            depth = point3D_cam[2, 0] / sc

            if (depth < bounds[id_im - 1, 0] or depth > bounds[id_im - 1, 1]):
                continue

            err = points[id_3D].error
            weight = 2 * np.exp(-((err / Err_mean) ** 2))
            depth_list.append(depth)
            coord_list.append(point2D)
            weight_list.append(weight)
        if len(depth_list) > 0:
            # Calculate useful stats
            depth_std = np.std(depth_list)
            mean_weight = np.mean(weight_list)
            total_points = len(images[id_im].xys)
            retention_rate = len(depth_list) / total_points * 100

            print(
                f"ID: {id_im:3d} | "
                f"Kept: {len(depth_list):4d}/{total_points:4d} "
                f"({retention_rate:.1f}%) | "
                f"Depth: {np.min(depth_list):.2f}-{np.max(depth_list):.2f} "
                f"(μ={np.mean(depth_list):.2f}, sigma={depth_std:.2f}) | "
                f"Conf: {mean_weight:.3f}"
            )

            data_list.append(
                {
                    "depth": np.array(depth_list),
                    "coord": np.array(coord_list),
                    "error": np.array(weight_list),
                }
            )
        else:
            print(id_im, len(depth_list))

    return data_list


class LIFFDataLoader(DataLoader):
    def __init__(self,
                 name,
                 root_dir,
                 split,
                 resolution=1,
                 bd_factor=0.75,
                 use_ndc=False,
                 load_depth=False):
        super().__init__()
        self.name = name
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.bd_factor = bd_factor
        self.use_ndc = use_ndc
        self.load_depth = load_depth

        assert split in ['train', 'val'], 'split must be either train or val'

        poses_bounds = np.load(os.path.join(root_dir, 'poses_bounds.npy'))
        self.image_paths = sorted(glob.glob(
            os.path.join(root_dir, 'images_undistorted/images/*')))

        assert len(poses_bounds) == len(self.image_paths), \
            'Mismatch between number of images and number of poses! ' \
            'Please rerun COLMAP!'

        # poses_bounds.shape -> (N_images, 17)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)

        # rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics

        # scale intrinsics to match resized image
        self.img_wh = (int(W * self.resolution), int(H * self.resolution))
        self.focal *= self.img_wh[0] / W

        # Original poses has rotation in form "down right back",
        # change to "right up back"
        # [-u, r, -t] -> [r, u, -t] (OpenGL's convention)
        poses = np.concatenate(
            [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        self.poses = center_poses(poses)

        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        # 這邊的意思就是找到整個場景中最接近相機的點，根據這個點的深度來調整整個場景的尺度
        # 調整過後這個最接近相機的點的深度就會位於 1.0 / 0.75 = 1.3333... 的位置
        # 因此這個時候相機的姿態也要跟著做調整
        near_original = self.bounds.min()
        scale_factor = near_original * bd_factor
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        if load_depth:
            self.depth_info = load_colmap_depth(
                root_dir, bounds=self.bounds, sc=scale_factor)

        K = np.array([
            [self.focal, 0.0, self.img_wh[0] / 2.0],
            [0.0, self.focal, self.img_wh[1] / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.K = torch.from_numpy(K).float()
        self.directions = get_ray_directions(
            self.img_wh[1], self.img_wh[0], self.K)

        if self.split == 'train':
            self.frames = range(1, len(self.image_paths))
        else:
            self.frames = [0]
        self.count = 0

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

        img = Image.open(self.image_paths[i])
        new_w, new_h = self.img_wh
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)

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
        c2w = torch.from_numpy(self.poses[i]).float()
        rays_o, rays_d = get_rays(self.directions, c2w)

        if self.use_ndc:
            rays_o, rays_d = get_ndc_rays(self.K, 1.0, rays_o, rays_d)
            near, far = 0, 1
        else:
            near = self.bounds.min() * 0.9
            far = self.bounds.max() * 1.0

        near_ = near * torch.ones_like(rays_o[:, :1])
        far_ = far * torch.ones_like(rays_o[:, :1])
        rays = torch.cat([rays_o, rays_d, near_, far_], dim=1)

        data = {
            'image_path': self.image_paths[i],
            'pose': c2w,
            'rays': rays,
            'rgbs': img,
            'image_size': (new_h, new_w),
        }

        if self.load_depth:
            # note that we do not use ndc coordinate for depth rays
            depths = torch.from_numpy(self.depth_info[i]['depth']).float()
            weights = torch.from_numpy(self.depth_info[i]['error']).float()
            coords = torch.from_numpy(self.depth_info[i]['coord']).float()
            rays_directions = get_ray_directions(
                self.img_wh[1], self.img_wh[0], self.K, coords=coords)  # N x 3
            rays_o, rays_d = get_rays(rays_directions, c2w)

            near_ = near * torch.ones_like(rays_o[:, :1])
            far_ = far * torch.ones_like(rays_o[:, :1])

            data.update({
                "depth_rays": torch.cat([rays_o, rays_d, near_, far_], dim=-1),
                "depth_values": depths[:, None],
                "depth_weights": weights[:, None]
            })

        return data

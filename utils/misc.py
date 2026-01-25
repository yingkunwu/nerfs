import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .colmap_utils import normalize


def visualize_depth(depth, return_numpy=False, cmap=cv2.COLORMAP_JET):
    """
    Convert a depth map to a color image tensor.

    Args:
        depth (Tensor): depth map of shape (H, W).
        cmap (int): OpenCV colormap.

    Returns:
        Tensor: color image tensor of shape (3, H, W).
    """
    # To numpy and replace NaNs.
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth)
    depth_np = np.nan_to_num(depth_np)

    # Normalize to [0, 1].
    min_val = np.min(depth_np)
    max_val = np.max(depth_np)
    norm = (depth_np - min_val) / (max_val - min_val + 1e-8)

    # To 8‐bit and apply colormap.
    depth_8u = (norm * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(depth_8u, cmap)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    if return_numpy:
        return color_rgb

    # PIL then to tensor.
    pil_img = Image.fromarray(color_rgb)
    tensor_img = T.ToTensor()(pil_img)

    return tensor_img


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


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def create_spiral_poses_from_pose(original_poses, radii, n_poses=120):
    """
    Create spiral poses around given poses for novel view rendering purpose.

    Inputs:
        original_poses: (N_frames, 3, 4)
            original poses around which to generate the spiral.
        radii: (3)
            radii of the spiral for each axis (only x, y are used)
        n_poses: int
            number of poses to create

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """
    N_frames = len(original_poses)
    # interpolation rotations
    rot_slerp = Slerp(
        range(N_frames),
        R.from_matrix(original_poses[..., :3])
    )
    interp_rots = rot_slerp(
        np.linspace(0, N_frames - 1, n_poses + 1)
    )[:-1]
    interp_rots = interp_rots.as_matrix()
    # interpolation positions
    interp_xyzs = np.stack([
        np.interp(
            np.linspace(0, N_frames - 1, n_poses + 1)[:-1],
            range(N_frames),
            original_poses[:, i, 3]
        ) for i in range(3)
    ], -1)

    poses_spiral = []
    for i, t in enumerate(
        np.linspace(0, 8 * np.pi, n_poses + 1)[:-1]
    ):  # rotate 8pi (4 rounds)
        pose = np.zeros((3, 4))
        pose[:, :3] = interp_rots[i]
        pose[:, 3] = (
            interp_xyzs[i] +
            radii * np.array([np.cos(t), -np.sin(t), 0])
        )
        poses_spiral.append(pose)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def quat_from_rotm(R):
    R = R.reshape(-1, 3, 3)
    w = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]))
    x = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    y = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    z = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))
    q0 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q0[:, 0] = w
    q0[:, 1] = x * np.sign(x * (R[:, 2, 1] - R[:, 1, 2]))
    q0[:, 2] = y * np.sign(y * (R[:, 0, 2] - R[:, 2, 0]))
    q0[:, 3] = z * np.sign(z * (R[:, 1, 0] - R[:, 0, 1]))
    q1 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q1[:, 0] = w * np.sign(w * (R[:, 2, 1] - R[:, 1, 2]))
    q1[:, 1] = x
    q1[:, 2] = y * np.sign(y * (R[:, 1, 0] + R[:, 0, 1]))
    q1[:, 3] = z * np.sign(z * (R[:, 0, 2] + R[:, 2, 0]))
    q2 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q2[:, 0] = w * np.sign(w * (R[:, 0, 2] - R[:, 2, 0]))
    q2[:, 1] = x * np.sign(x * (R[:, 0, 1] + R[:, 1, 0]))
    q2[:, 2] = y
    q2[:, 3] = z * np.sign(z * (R[:, 1, 2] + R[:, 2, 1]))
    q3 = np.empty((R.shape[0], 4), dtype=R.dtype)
    q3[:, 0] = w * np.sign(w * (R[:, 1, 0] - R[:, 0, 1]))
    q3[:, 1] = x * np.sign(x * (R[:, 0, 2] + R[:, 2, 0]))
    q3[:, 2] = y * np.sign(y * (R[:, 1, 2] + R[:, 2, 1]))
    q3[:, 3] = z
    q = q0 * (w[:, None] > 0) + (w[:, None] == 0) * (
        q1 * (x[:, None] > 0)
        + (x[:, None] == 0) * (
            q2 * (y[:, None] > 0) + (y[:, None] == 0) * (q3)
        )
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.squeeze()


def cameracenter_from_translation(R, t):
    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def translation_from_cameracenter(R, C):
    C = C.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    t = -R @ C
    return t.squeeze()


def quat_slerp_space(q0, q1, t=None, num=100, endpoint=True):
    q0 = q0.reshape(-1, 4)
    q1 = q1.reshape(-1, 4)
    dot = (q0 * q1).sum(axis=1)

    ma = dot < 0
    q1[ma] *= -1
    dot[ma] *= -1

    if t is None:
        t = np.linspace(0, 1, num=num, endpoint=endpoint, dtype=q0.dtype)
    t = t.reshape((-1, 1))
    num = t.shape[0]

    res = np.empty((q0.shape[0], num, 4), dtype=q0.dtype)

    ma = dot > 0.9995
    if np.any(ma):
        res[ma] = (
            q0[ma] + t[..., None] * (q1[ma] - q0[ma])
        ).transpose(1, 0, 2)

    ma = ~ma
    if np.any(ma):
        q0 = q0[ma]
        q1 = q1[ma]
        dot = dot[ma]

        dot = np.clip(dot, -1, 1)
        theta0 = np.arccos(dot)
        theta = theta0 * t
        s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
        s1 = np.sin(theta) / np.sin(theta0)
        res[ma] = ((s0[..., None] * q0) + (s1[..., None] * q1)).transpose(
            1, 0, 2
        )
    return res.squeeze()


def rotm_from_quat(q):
    q = q.reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w,
            ],
            [
                2 * x * y + 2 * z * w,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * x * w,
            ],
            [
                2 * x * z - 2 * y * w,
                2 * y * z + 2 * x * w,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=q.dtype,
    )
    R = R.transpose((2, 0, 1))
    return R.squeeze()


def interpolate_waypoints(wpRs, wpts, steps=25):
    wpRs = np.array(wpRs)
    wpts = np.array(wpts)
    wpqs = quat_from_rotm(wpRs)
    wpCs = cameracenter_from_translation(wpRs, wpts)
    qs, Cs = [], []
    for idx in range(wpRs.shape[0] - 1):
        q0, q1 = wpqs[idx], wpqs[idx + 1]
        C0, C1 = wpCs[idx], wpCs[idx + 1]
        alphas = np.linspace(0, 1, num=steps, endpoint=False)
        Cs.append((1 - alphas[:, None]) * C0 + alphas[:, None] * C1)
        qs.append(quat_slerp_space(q0, q1, t=alphas))
    Rs = rotm_from_quat(np.vstack(qs))
    ts = translation_from_cameracenter(Rs, np.vstack(Cs))
    return Rs, ts

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T


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

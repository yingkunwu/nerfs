import torch
from typing import Union
from einops import rearrange


def create_meshgrid(H: int,
                    W: int,
                    normalized_coordinates: bool = True,
                    dtype: torch.dtype = None,
                    device: Union[torch.device, str] = None) -> torch.Tensor:
    """
    Create a meshgrid in the Kornia format.

    Returns:
        grid: (1, H, W, 2) where grid[..., 0] = x, grid[..., 1] = y.
              If normalized_coordinates:
                  x in [-1, 1] across width, y in [-1, 1] across height.
              Else:
                  x in [0, W-1], y in [0, H-1].
    """
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, W, dtype=dtype, device=device)
        ys = torch.linspace(-1.0, 1.0, H, dtype=dtype, device=device)
    else:
        xs = torch.linspace(0, W - 1, W, dtype=dtype, device=device)
        ys = torch.linspace(0, H - 1, H, dtype=dtype, device=device)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2) -> [..., 0]=x, [..., 1]=y
    return grid.unsqueeze(0)  # (1, H, W, 2)


def get_ray_directions(H, W, K, return_uv=False):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference:
      https://www.scratchapixel.com/lessons/3d-basic-rendering/
      ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, K: image height, width and camera intrinsics

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coord.
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)

    assert K.shape == (3, 3), "K must be of shape (3, 3)"
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    fx = torch.as_tensor(fx, dtype=i.dtype, device=i.device)
    fy = torch.as_tensor(fy, dtype=i.dtype, device=i.device)
    cx = torch.as_tensor(cx, dtype=i.dtype, device=i.device)
    cy = torch.as_tensor(cy, dtype=i.dtype, device=i.device)

    directions = torch.stack(
        [
            (i - cx) / fx,
            (j - cy) / fy,
            torch.ones_like(i),
        ],
        dim=-1,
    )
    if return_uv:
        return directions, grid

    return directions


def get_rays(directions, c2w):
    """
    Get ray origins and normalized directions in world coord.
    Reference:
      https://www.scratchapixel.com/lessons/3d-basic-rendering/
      ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3)
        c2w: (3, 4) camera-to-world transform

    Outputs:
        rays_o: (H*W, 3)
        rays_d: (H*W, 3)
    """
    # Rotate ray directions to world coord
    rays_d = directions @ c2w[:, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Origin is camera origin in world coord
    rays_o = c2w[:, 3].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(K, near, shift_near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded
    (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        K: (3, 3) camera intrinsics
        near: (N_rays) or float, the depths of the near plane
        shift_near: (N_rays) or float, the amount to shift the origin
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Shift ray origins to near plane
    t = -(shift_near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1./(cx/fx) * ox_oz
    o1 = -1./(cy/fy) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(cx/fx) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1./(cy/fy) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


# NOTE: WE DO IN COLMAP/OPENCV FORMAT, BUT INPUT IS OPENGL FORMAT!!!!!
def perspective_projection(pts_3d, h, w, f):
    x = pts_3d[..., 0:1] * f / -pts_3d[..., 2:3] + w / 2.0
    y = -pts_3d[..., 1:2] * f / -pts_3d[..., 2:3] + h / 2.0
    pts_2d = torch.cat([x, y], dim=-1)
    return pts_2d


def ndc2world(xyz, K, eps=1e-6):
    """
    Convert NDC coordinates into world coordinates.
    Inputs:
        xyz: (N, 3) or (N, M, 3) NDC coordinates
             in case len(K.shape)==3, the first dimension size must match
        K: (3, 3) camera intrinsics
        eps: float to prevent division by zero

    Outputs:
        world: (N, 3) or (N, M, 3) world coordinates
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    Rz = 2 / (xyz[..., 2] - 1 - eps)
    Rx = -Rz * xyz[..., 0] * cx / fx
    Ry = -Rz * xyz[..., 1] * cy / fy

    if len(xyz.shape) == 2:
        world = rearrange([Rx, Ry, Rz], 'c n -> n c', c=3)
    elif len(xyz.shape) == 3:
        world = rearrange([Rx, Ry, Rz], 'c n m -> n m c', c=3)
    return world


def perturb_samples(z_vals, perturb):
    mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
    upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
    lower = torch.cat([z_vals[:, :1], mids], dim=-1)
    rand = perturb * torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * rand

    return z_vals


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample N_importance samples from bins with distribution defined by weights.
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)
    inds_g = torch.stack([below, above], dim=-1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_g).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_g).view(N_rays, N_importance, 2)

    # Linear interpolation inside the bin:
    # map u from its CDF interval to the corresponding z-interval.
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1
    samples = bins_g[..., 0] + (
        (u - cdf_g[..., 0]) / denom
    ) * (bins_g[..., 1] - bins_g[..., 0])
    return samples

import torch
from einops import rearrange, repeat

from utils.ray_utils import perturb_samples, sample_pdf


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit
    sphere: ||ray_o + (d * ray_d)||^2 = 1, where d = d1 + d2
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception('Not all your cameras are bounded by the unit sphere; '
                        'please make sure the cameras are normalized!')
    d2 = (
        torch.sqrt(torch.sum(ray_d * ray_o, dim=-1) ** 2
                   - torch.sum(ray_d * ray_d, dim=-1)
                   * (torch.sum(ray_o * ray_o, dim=-1) - 1.))
    ) / torch.sum(ray_d * ray_d, dim=-1)

    return d1 + d2


def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    d = intersect_sphere(ray_o, ray_d)
    p_sphere = ray_o + d.unsqueeze(-1) * ray_d

    N_sample = depth.shape[-1]
    ray_o = repeat(ray_o, '... d -> ... n d', n=N_sample)
    ray_d = repeat(ray_d, '... d -> ... n d', n=N_sample)
    p_sphere = repeat(p_sphere, '... d -> ... n d', n=N_sample)

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)

    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d

    p_mid_norm = torch.norm(p_mid, dim=-1)  # not sure why we need norm here
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is already inverse here
    rot_angle = (phi - theta).unsqueeze(-1)

    # rotate p_sphere by rot_angle using Rodrigues formula
    p_sphere = p_sphere * torch.cos(rot_angle) \
        + torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) \
        + rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) \
        * (1. - torch.cos(rot_angle))
    p_sphere = p_sphere / torch.norm(p_sphere, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere, depth.unsqueeze(-1)), dim=-1)

    return pts


def inference(
    fg_model,
    bg_model,
    embeddings,
    rays_o,
    rays_d,
    fg_z_vals,
    bg_z_vals,
    fg_z_max,
    noise_std,
):
    """
    Perform NeRF model inference on sampled points.

    Args:
        model: NeRF model (coarse or fine).
        embeddings: positional embedding module for rgb and dirs.
        rays_o: tensor, shape (N_rays, 3), ray origins.
        rays_d: tensor, shape (N_rays, 3), ray directions.
        fg_z_vals: tensor, shape (N_rays, N_samples), foreground depths.
        bg_z_vals: tensor, shape (N_rays, N_samples), background depths.
        fg_z_max: tensor, shape (N_rays,), the intersection depth with sphere.
        noise_std: float, std for noise added to sigma.
    """
    N_rays, N_samples_ = fg_z_vals.shape[:2]

    # Foreground inputs
    fg_pts = (
        rearrange(rays_o, 'r d -> r 1 d')
        + rearrange(rays_d, 'r d -> r 1 d')
        * rearrange(fg_z_vals, 'r n -> r n 1')
    )
    rays_d_rep = repeat(rays_d, 'r d -> r n d', n=N_samples_)

    inp = torch.cat([embeddings['fg_xyz'](fg_pts),
                     embeddings['fg_dir'](rays_d_rep)], dim=-1)

    fg_out = fg_model(inp)

    # Background inputs
    # near_depth: physical far; far_depth: physical near
    bg_z_vals = torch.flip(bg_z_vals, dims=[-1,])
    bg_pts = depth2pts_outside(rays_o, rays_d, bg_z_vals)

    inp = torch.cat([embeddings['bg_xyz'](bg_pts),
                     embeddings['bg_dir'](rays_d_rep)], dim=-1)

    bg_out = bg_model(inp)

    fg_rgbs, fg_sigmas = fg_out[..., :3], fg_out[..., 3]
    bg_rgbs, bg_sigmas = bg_out[..., :3], bg_out[..., 3]

    # Foreground Volume rendering
    deltas = fg_z_vals[:, 1:] - fg_z_vals[:, :-1]
    delta_inf = fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]
    deltas = torch.cat([deltas, delta_inf], dim=-1)
    deltas = deltas * torch.norm(rays_d, dim=-1, keepdim=True)

    noise = torch.randn(fg_sigmas.shape, device=fg_sigmas.device) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(fg_sigmas + noise))
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10],
        dim=-1
    )
    T = torch.cumprod(alphas_shifted, dim=-1)
    # the probability the ray was not terminated by any foreground density
    bg_lambda = T[..., -1].clone()
    fg_weights = alphas * T[:, :-1]

    # Background Volume rendering
    deltas = bg_z_vals[:, :-1] - bg_z_vals[:, 1:]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], dim=-1)

    noise = torch.randn(bg_sigmas.shape, device=bg_sigmas.device) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(bg_sigmas + noise))
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10],
        dim=-1
    )
    bg_weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :-1]

    # sum over n_sample dimension
    fg_rgb = torch.sum(fg_weights[..., None] * fg_rgbs, dim=1)
    fg_depth = torch.sum(fg_weights * fg_z_vals, dim=1)

    bg_rgb = torch.sum(bg_weights[..., None] * bg_rgbs, dim=1)
    bg_depth = torch.sum(bg_weights * bg_z_vals, dim=1)

    bg_rgb = bg_rgb * bg_lambda[:, None]
    bg_depth = bg_depth * bg_lambda

    return {
        'rgb': fg_rgb + bg_rgb,
        'depth': fg_depth + bg_depth,
        'fg_rgb': fg_rgb,
        'fg_depth': fg_depth,
        'bg_rgb': bg_rgb,
        'bg_depth': bg_depth,
        'fg_weights': fg_weights,
        'bg_weights': bg_weights
    }


def render_rays(
    models,
    embeddings,
    rays,
    N_samples=64,
    perturb=0,
    noise_std=1,
    N_importance=0,
):
    """
    Render rays by computing NeRF model outputs on rays.

    Args:
        models: list of NeRF models [coarse, fine].
        embeddings: [positional embedding, directional embedding].
        rays: tensor, shape (N_rays, 8), (o, d, near, far).
        N_samples: int, number of coarse samples.
        perturb: float, perturb factor.
        noise_std: float, sigma noise std.
        N_importance: int, number of fine samples.
    """
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]

    # foreground depth
    fg_far = intersect_sphere(rays_o, rays_d)
    fg_near = 1e-4 * torch.ones_like(fg_far)

    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    fg_depth = fg_near[..., None] * (1 - z_steps) + fg_far[..., None] * z_steps

    # background depth
    bg_depth = torch.linspace(0., 1., N_samples, device=rays.device)
    bg_depth = repeat(bg_depth, 'n -> r n', r=rays.shape[0])

    if perturb > 0:
        fg_depth = perturb_samples(fg_depth, perturb)
        bg_depth = perturb_samples(bg_depth, perturb)

    result = {}

    result_coarse = inference(
        models['fg_coarse'], models['bg_coarse'], embeddings, rays_o, rays_d,
        fg_depth, bg_depth, fg_far, noise_std
    )
    result.update({
        'rgb_coarse': result_coarse['rgb'],
        'depth_coarse': result_coarse['depth'],
        'fg_rgb_coarse': result_coarse['fg_rgb'],
        'fg_depth_coarse': result_coarse['fg_depth'],
        'bg_rgb_coarse': result_coarse['bg_rgb'],
        'bg_depth_coarse': result_coarse['bg_depth'],
        'fg_weights': result_coarse['fg_weights'],
        'bg_weights': result_coarse['bg_weights']
    })

    if N_importance > 0:
        fg_weights = result['fg_weights'].detach()
        # To build a piecewise-constant PDF, we need bin centers.
        mids = 0.5 * (fg_depth[:, :-1] + fg_depth[:, 1:])
        z_fine = sample_pdf(mids, fg_weights[:, 1:-1], N_importance,
                            det=(perturb == 0))
        fg_depth = torch.sort(torch.cat([fg_depth, z_fine], dim=-1), dim=-1)[0]

        bg_weights = result['bg_weights'].detach()
        # To build a piecewise-constant PDF, we need bin centers.
        mids = 0.5 * (bg_depth[:, :-1] + bg_depth[:, 1:])
        z_fine = sample_pdf(mids, bg_weights[:, 1:-1], N_importance,
                            det=(perturb == 0))
        bg_depth = torch.sort(torch.cat([bg_depth, z_fine], dim=-1), dim=-1)[0]

        result_fine = inference(
            models['fg_fine'], models['bg_fine'], embeddings, rays_o, rays_d,
            fg_depth, bg_depth, fg_far, noise_std
        )
        result.update({
            'rgb_fine': result_fine['rgb'],
            'depth_fine': result_fine['depth'],
            'fg_rgb_fine': result_fine['fg_rgb'],
            'fg_depth_fine': result_fine['fg_depth'],
            'bg_rgb_fine': result_fine['bg_rgb'],
            'bg_depth_fine': result_fine['bg_depth']
        })

    return result

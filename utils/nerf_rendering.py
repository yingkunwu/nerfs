import torch
from einops import rearrange, repeat

from utils.ray_utils import perturb_samples, sample_pdf


def inference(
    model,
    embeddings,
    ray_pts,
    ray_d,
    z_vals,
    noise_std,
    white_back,
):
    """
    Perform NeRF model inference on sampled points.

    Args:
        model: NeRF model (coarse or fine).
        embeddings: positional embedding module for rgb and dirs.
        ray_pts: tensor, shape (N_rays, N_samples, 3).
        ray_d: tensor, shape (N_rays, N_samples).
        noise_std: float, std for noise added to sigma.
        white_back: bool, whether to use white background.

    Returns:
        If weights_only:
            weights: (N_rays, N_samples)
        else:
            rgb_final: (N_rays, 3)
            depth_final: (N_rays,)
            weights: (N_rays, N_samples)
    """
    N_rays, N_samples_ = ray_pts.shape[:2]

    ray_d_rep = repeat(ray_d, 'r d -> r n d', n=N_samples_)
    inp = torch.cat([embeddings['xyz'](ray_pts),
                     embeddings['dir'](ray_d_rep)], dim=-1)
    out = model(inp)

    rgbsigma = out.view(N_rays, N_samples_, 4)
    rgbs = rgbsigma[..., :3]
    sigmas = rgbsigma[..., 3]

    # Volume rendering
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], dim=-1)
    deltas = deltas * torch.norm(ray_d, dim=-1, keepdim=True)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10],
        dim=-1
    )
    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[:, :-1]

    weights_sum = weights.sum(dim=1)

    # sum over n_sample dimension
    rgb_final = torch.sum(weights[..., None] * rgbs, dim=1)
    depth_final = torch.sum(weights * z_vals, dim=1)

    # ignore background (rays that didn’t intersect any surface)
    if white_back:
        # adding this term fills remaining transparency with white color.
        rgb_final = rgb_final + (1 - weights_sum[..., None])

    return rgb_final, depth_final, weights


def render_rays(
    models,
    embeddings,
    rays,
    N_samples=64,
    perturb=0,
    noise_std=1,
    N_importance=0,
    white_back=False
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
        white_back: bool, white background.
    """
    model_coarse = models['coarse']
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)

    # z_vals.shape -> [num_rays, N_samples]
    z_vals = near * (1 - z_steps) + far * z_steps
    z_vals = z_vals.expand(rays.shape[0], N_samples)

    if perturb > 0:
        z_vals = perturb_samples(z_vals, perturb)

    pts_coarse = (
        rearrange(rays_o, 'r d -> r 1 d')
        + rearrange(rays_d, 'r d -> r 1 d')
        * rearrange(z_vals, 'r n -> r n 1')
    )
    rgb_coarse, depth_coarse, weights_coarse = inference(
        model_coarse, embeddings, pts_coarse, rays_d, z_vals,
        noise_std, white_back
    )
    result = {
        'rgb_coarse': rgb_coarse,
        'depth_coarse': depth_coarse,
        'opacity_coarse': weights_coarse.sum(dim=1),
    }

    if N_importance > 0:
        model_fine = models['fine']
        # To build a piecewise-constant PDF, we need bin centers.
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        z_fine = sample_pdf(
            mids, weights_coarse[:, 1:-1], N_importance,
            det=(perturb == 0)
        ).detach()
        z_vals = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)[0]

        pts_fine = (
            rearrange(rays_o, 'r d -> r 1 d')
            + rearrange(rays_d, 'r d -> r 1 d')
            * rearrange(z_vals, 'r n -> r n 1')
        )
        rgb_fine, depth_fine, weights_fine = inference(
            model_fine, embeddings, pts_fine, rays_d, z_vals,
            noise_std, white_back
        )
        result.update({
            'rgb_fine': rgb_fine,
            'depth_fine': depth_fine,
            'opacity_fine': weights_fine.sum(dim=1)
        })
    return result

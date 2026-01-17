import torch
from einops import rearrange, repeat

from utils.ray_utils import perturb_samples, sample_pdf


def inference(
    model,
    embeddings,
    ray_pts,
    ray_d,
    rays_t,
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
    rays_t_rep = repeat(rays_t, 'r -> r n', n=N_samples_)
    inp = torch.cat([embeddings['xyz'](ray_pts),
                     embeddings['dir'](ray_d_rep)], dim=-1)

    if model.output_transient:
        inp = torch.cat([embeddings['xyz'](ray_pts),
                         embeddings['dir'](ray_d_rep),
                         embeddings['appearance'](rays_t_rep),
                         embeddings['transient'](rays_t_rep)], dim=-1)
    else:
        inp = torch.cat([embeddings['xyz'](ray_pts),
                         embeddings['dir'](ray_d_rep),
                         embeddings['appearance'](rays_t_rep)], dim=-1)

    out = model(inp)

    static_rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
    static_sigmas = out[..., 3]  # (N_rays, N_samples_)

    if model.output_transient:
        tra_rgbs = out[..., 4:7]  # (N_rays, N_samples_, 3)
        tra_sigmas = out[..., 7]  # (N_rays, N_samples_)
        tra_betas = out[..., 8]  # (N_rays, N_samples_)

    # Volume rendering
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 1e2 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], dim=-1)
    deltas = deltas * torch.norm(ray_d, dim=-1, keepdim=True)

    noise = torch.randn(static_sigmas.shape,
                        device=static_sigmas.device) * noise_std

    if model.output_transient:
        static_alphas = 1 - torch.exp(-deltas * (static_sigmas + noise))
        transient_alphas = 1 - torch.exp(-deltas * (tra_sigmas + noise))
        alphas = 1 - torch.exp(-deltas * (static_sigmas + tra_sigmas + noise))
    else:
        alphas = 1 - torch.exp(-deltas * (static_sigmas + noise))

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10],
        dim=-1
    )
    transmittance = torch.cumprod(alphas_shifted[:, :-1], -1)

    if model.output_transient:
        static_weights = static_alphas * transmittance
        transient_weights = transient_alphas * transmittance
    weights = alphas * transmittance
    weights_sum = weights.sum(dim=1)

    if model.output_transient:
        # sum over n_sample dimension
        static_rgb = torch.sum(static_weights[..., None] * static_rgbs, dim=1)

        tra_rgb = torch.sum(transient_weights[..., None] * tra_rgbs, dim=1)
        tra_beta = torch.sum(transient_weights * tra_betas, dim=1)
        # Add beta_min AFTER the beta composition. Different from the paper.
        tra_beta += model.beta_min
    else:
        static_rgb = torch.sum(weights[..., None] * static_rgbs, dim=1)

    # ignore background (rays that didn’t intersect any surface)
    if white_back:
        # adding this term fills remaining transparency with white color.
        static_rgb = static_rgb + (1 - weights_sum[..., None])

    depth = torch.sum(weights * z_vals, dim=1)

    if model.output_transient:
        result = {
            f'rgb_{model.typ}': static_rgb + tra_rgb,
            f'depth_{model.typ}': depth,
            f'static_rgb_{model.typ}': static_rgb,
            f'transient_rgb_{model.typ}': tra_rgb,
            f'transient_sigma_{model.typ}': tra_sigmas,
            f'transient_beta_{model.typ}': tra_beta,
            'weights': weights
        }
    else:
        result = {
            f'rgb_{model.typ}': static_rgb,
            f'depth_{model.typ}': depth,
            'weights': weights
        }

    return result


def render_rays(
    models,
    embeddings,
    rays,
    rays_t,
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
    result = inference(
        model_coarse, embeddings, pts_coarse, rays_d, rays_t, z_vals,
        noise_std, white_back
    )

    if N_importance > 0 and 'fine' in models:
        model_fine = models['fine']
        # To build a piecewise-constant PDF, we need bin centers.
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        z_fine = sample_pdf(
            mids, result['weights'][:, 1:-1].detach(), N_importance,
            det=(perturb == 0))
        z_vals = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)[0]

        pts_fine = (
            rearrange(rays_o, 'r d -> r 1 d')
            + rearrange(rays_d, 'r d -> r 1 d')
            * rearrange(z_vals, 'r n -> r n 1')
        )

        result_fine = inference(
            model_fine, embeddings, pts_fine, rays_d, rays_t, z_vals,
            noise_std, white_back
        )
        result.update({k: v for k, v in result_fine.items()
                       if k.endswith('_fine')})

    return result

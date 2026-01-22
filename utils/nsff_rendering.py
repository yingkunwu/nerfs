import torch
import torch.nn.functional as F
from einops import repeat, rearrange

from utils.ray_utils import perturb_samples


def compute_deltas(z_vals, rays_d):
    """Computes distances between samples along the ray."""
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    # Append distance to infinity for the last sample
    delta_inf = torch.empty_like(deltas[:, :1]).fill_(1e10)
    deltas = torch.cat([deltas, delta_inf], dim=-1)
    # Scale by ray direction norm to get real world distances
    return deltas * torch.norm(rays_d, dim=-1, keepdim=True)


def raw2outputs(
    rgb,
    sigma,
    deltas,
    z_vals,
    noise_std=0.0
):
    """
    Standard NeRF Volume Rendering integration.
    Transforms raw model outputs (rgb, sigma) into image space
    (rgb_map, depth_map).
    """
    # Add noise to sigma during training for regularization
    if noise_std > 0:
        noise = torch.randn_like(sigma) * noise_std
        sigma = sigma + noise

    alphas = 1.0 - torch.exp(-deltas * F.relu(sigma))
    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1.0 - alphas + 1e-10], dim=-1)
    transmittance = torch.cumprod(alphas_shifted, dim=-1)[:, :-1]
    weights = alphas * transmittance

    # Integrate RGB and Depth
    rgb_map = torch.sum(weights[..., None] * rgb, dim=1)
    depth_map = torch.sum(weights * z_vals, dim=1)

    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "weights": weights
    }


def inference_static(
    model,
    embeddings,
    xyz,
    rays_d,
    z_vals,
    noise_std=1.0
):
    """
    Inference for the static (rigid) NeRF model.
    """
    N_rays, N_samples = xyz.shape[:2]

    # Prepare inputs
    dir_rep = repeat(rays_d, 'r d -> r n d', n=N_samples)
    inp = torch.cat([embeddings['xyz'](xyz),
                     embeddings['dir'](dir_rep)], dim=-1)

    # Model Query
    raw_output = model(inp)
    rgb = raw_output[..., :3]
    sigma = raw_output[..., 3]

    # Rendering
    deltas = compute_deltas(z_vals, rays_d)
    results = raw2outputs(rgb, sigma, deltas, z_vals, noise_std)

    return results, raw_output


def warp_and_render(
    model,
    embeddings,
    xyz_start,
    rays_d,
    z_vals,
    time_emb,
    flow_type,
    noise_std
):
    """
    Helper to warp points based on flow and render the view from the new
    position. Used for temporal consistency (forward/backward warping).
    """
    assert flow_type in ['fw', 'bw']

    N_samples = xyz_start.shape[1]
    dir_rep = repeat(rays_d, 'r d -> r n d', n=N_samples)

    inp = torch.cat([embeddings['xyz'](xyz_start),
                     embeddings['transient'](time_emb),
                     embeddings['dir'](dir_rep)], dim=-1)
    out = model(inp)

    # 2. Extract outputs
    rgb = out[..., :3]
    sigma = out[..., 3]

    # Extract flow relative to direction
    # indices [4:7] are backward flow, [7:10] are forward flow
    if flow_type == 'bw':
        flow = out[..., 7:10]
    else:
        flow = out[..., 4:7]

    # Mask flow at far depth
    mask = (z_vals <= 0.95)[..., None]
    flow = flow * mask

    # 3. Render
    deltas = compute_deltas(z_vals, rays_d)
    render_results = raw2outputs(rgb, sigma, deltas, z_vals, noise_std)

    return render_results, flow


def inference_blending(
    model,
    embeddings,
    xyz,
    rays_d,
    t,
    max_t,
    z_vals,
    static_raw_output,
    noise_std,
    output_transient=True,
):
    """
    Complex inference that blends a Static NeRF with a Dynamic NeRF.
    """
    N_rays, N_samples = xyz.shape[:2]

    dir_rep = repeat(rays_d, 'r d -> r n d', n=N_samples)
    t_rep = repeat(t, 'r -> r n', n=N_samples)

    inp = torch.cat([embeddings['xyz'](xyz),
                     embeddings['transient'](t_rep),
                     embeddings['dir'](dir_rep)], dim=-1)

    out_dy = model(inp)

    # Dynamic components
    rgb_dy = out_dy[..., :3]
    sigma_dy = out_dy[..., 3]

    # Extract Static components
    rgb_rigid = static_raw_output[..., :3]
    sigma_rigid = static_raw_output[..., 3]
    blend_w = static_raw_output[..., 4]

    # Add noise
    if noise_std > 0:
        noise_dy = torch.randn_like(sigma_dy) * noise_std
        noise_rig = torch.randn_like(sigma_rigid) * noise_std
    else:
        noise_dy, noise_rig = 0.0, 0.0

    # Alpha composite: Blending dynamic and rigid density based on blend_w
    deltas = compute_deltas(z_vals, rays_d)
    alpha_dy = (
        1 - torch.exp(-deltas * F.relu(sigma_dy + noise_dy))
    ) * blend_w
    alpha_rigid = (
        1 - torch.exp(-deltas * F.relu(sigma_rigid + noise_rig))
    ) * (1.0 - blend_w)

    # Combined Transmittance
    # The probability of NOT hitting either is (1-a_dy)*(1-a_rigid)
    alphas_shifted = torch.cat([
        torch.ones_like(alpha_dy[:, :1]),
        (1.0 - alpha_dy) * (1.0 - alpha_rigid) + 1e-10
    ], dim=-1)
    transmittance = torch.cumprod(alphas_shifted, dim=-1)[:, :-1]

    weights_dy = transmittance * alpha_dy
    weights_rigid = transmittance * alpha_rigid
    weights_mix = weights_dy + weights_rigid

    # Integrated outputs
    rgb_map_ref = torch.sum(weights_dy[..., None] * rgb_dy
                            + weights_rigid[..., None] * rgb_rigid, dim=1)
    depth_map_ref = torch.sum(weights_mix * z_vals, dim=-1)

    # Standard rendering of just the dynamic component
    results_dy = raw2outputs(rgb_dy, sigma_dy, deltas, z_vals, noise_std)

    # Transient Warping
    res_warp = {}
    if output_transient:
        raw_sf_ref2prev, raw_sf_ref2post = out_dy[..., 4:7], out_dy[..., 7:10]
        raw_prob_ref2prev, raw_prob_ref2post = out_dy[..., 10], out_dy[..., 11]

        # Mask flow
        mask = (z_vals <= 0.95)[..., None]
        sf_ref2prev = raw_sf_ref2prev * mask
        sf_ref2post = raw_sf_ref2post * mask

        # Warp and Render: Backward (t - 1)
        xyz_bw = xyz + sf_ref2prev
        t_prev = torch.clamp(t_rep - 1, min=0)
        results_bw, sf_prev2ref = warp_and_render(
            model, embeddings, xyz_bw, rays_d, z_vals, t_prev, 'bw', noise_std
        )

        # Warp and Render: Forward (t + 1)
        xyz_fw = xyz + sf_ref2post
        t_next = torch.clamp(t_rep + 1, max=max_t - 1)
        results_fw, sf_post2ref = warp_and_render(
            model, embeddings, xyz_fw, rays_d, z_vals, t_next, 'fw', noise_std
        )

        # Calculate centroids for loss
        res_warp = {
            'sf_ref2prev': sf_ref2prev,
            'sf_ref2post': sf_ref2post,
            'sf_prev2ref': sf_prev2ref,
            'sf_post2ref': sf_post2ref,
            'raw_prob_ref2prev': raw_prob_ref2prev,
            'raw_prob_ref2post': raw_prob_ref2post,
            'prob_ref2prev': torch.sum(
                weights_mix.detach() * raw_prob_ref2prev, dim=-1),
            'prob_ref2post': torch.sum(
                weights_mix.detach() * raw_prob_ref2post, dim=-1),
            'rgb_bw': results_bw['rgb_map'],
            'rgb_fw': results_fw['rgb_map'],
            'xyz_bw': torch.sum(
                results_dy['weights'][..., None] * xyz_bw, dim=-2),
            'xyz_fw': torch.sum(
                results_dy['weights'][..., None] * xyz_fw, dim=-2),
            'raw_xyz_bw': xyz_bw,
            'raw_xyz_fw': xyz_fw,
            'raw_pts_ref': xyz
        }

    return {
        'rgb_map': rgb_map_ref,
        'depth_map': depth_map_ref,
        'rgb_map_dy': results_dy['rgb_map'],
        'depth_map_dy': results_dy['depth_map'],
        **res_warp
    }


def render_rays(
    models,
    embeddings,
    rays,
    rays_t,
    max_t,
    N_samples=64,
    perturb=0.0,
    noise_std=1.0
):
    """
    Main entry point: Generates samples along rays and runs inference.
    """
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)

    # z_vals.shape -> [num_rays, N_samples]
    z_vals = near * (1 - z_steps) + far * z_steps
    z_vals = z_vals.expand(rays.shape[0], N_samples)

    if perturb > 0:
        z_vals = perturb_samples(z_vals, perturb)

    # pts = o + d * z
    pts_coarse = (
        rearrange(rays_o, 'r d -> r 1 d')
        + rearrange(rays_d, 'r d -> r 1 d')
        * rearrange(z_vals, 'r n -> r n 1')
    )

    # 2. Run Static Inference
    static_results, static_raw = inference_static(
        models['static'], embeddings, pts_coarse, rays_d, z_vals, noise_std
    )

    # 3. Run Dynamic/Blending Inference
    blending_results = inference_blending(
        models['dynamic'], embeddings, pts_coarse, rays_d, rays_t, max_t,
        z_vals, static_raw, noise_std
    )

    return {
        'rgb_map_static': static_results['rgb_map'],
        'depth_map_static': static_results['depth_map'],
        'rgb_map_ref_dynamic': blending_results['rgb_map_dy'],
        'depth_map_ref_dynamic': blending_results['depth_map_dy'],
        'rgb_map_ref': blending_results['rgb_map'],
        'depth_map_ref': blending_results['depth_map'],
        'rgb_bw': blending_results['rgb_bw'],
        'rgb_fw': blending_results['rgb_fw'],
        'prob_ref2prev': blending_results['prob_ref2prev'],
        'prob_ref2post': blending_results['prob_ref2post'],
        'raw_sf_ref2prev': blending_results['sf_ref2prev'],
        'raw_sf_ref2post': blending_results['sf_ref2post'],
        'raw_sf_prev2ref': blending_results['sf_prev2ref'],
        'raw_sf_post2ref': blending_results['sf_post2ref'],
        'raw_prob_ref2prev': blending_results['raw_prob_ref2prev'],
        'raw_prob_ref2post': blending_results['raw_prob_ref2post'],
        'xyz_bw': blending_results['xyz_bw'],
        'xyz_fw': blending_results['xyz_fw'],
        'raw_xyz_bw': blending_results['raw_xyz_bw'],
        'raw_xyz_fw': blending_results['raw_xyz_fw'],
        'raw_pts_ref': blending_results['raw_pts_ref']
    }

import torch
from einops import repeat, rearrange, reduce

from utils.ray_utils import (
    perturb_samples, create_meshgrid, ndc2world, compute_world_visibility
)

# explicitly zero the scene flow if NDC z exceeds this value
Z_FAR = 0.95


def render_rays(
    models,
    embeddings,
    rays,
    ts,
    max_t,
    N_samples=128,
    perturb=0.0,
    noise_std=1.0,
    test_time=False,
    output_transient_flow=('fw', 'bw', 'disocc'),
    dataset=None,
):
    """
    Volume-render a batch of rays with the static + dynamic (transient)
    models, composited additively like NeRF-W / nsff_pl.

    Inputs:
        models: {'static': NeRF_Static, 'dynamic': NeRF_Dynamic}
        embeddings: {'xyz': Embedding, 'dir': Embedding,
                     'transient': nn.Embedding}
        rays: (N_rays, 6+) ray origins and directions (extra columns ignored)
        ts: (N_rays) integer frame indices
        max_t: N_frames - 1 (largest valid frame index)
        N_samples: samples per ray (uniform in NDC z)
        perturb: stratified-sampling jitter factor (training only)
        noise_std: sigma noise std (training only)
        test_time: if True, skip the warped renderings (used for losses) and
            additionally store the per-sample buffers interpolate() needs
        output_transient_flow: subset of ('fw', 'bw', 'disocc')
        dataset: if given at test time, dynamic sigmas of sample points that
            are not visible from the training camera at time ts[0] are
            suppressed (visibility culling, cf. nsff_pl)
    """
    results = {}
    act = torch.nn.Softplus()  # sigma activation
    model_static, model_dynamic = models['static'], models['dynamic']
    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]

    zs = torch.linspace(0, 1, N_samples, device=rays.device)
    zs = zs.expand(N_rays, N_samples)
    if perturb > 0:
        zs = perturb_samples(zs, perturb)
    results['zs_fine'] = zs

    xyz = (
        rearrange(rays_o, 'n1 c -> n1 1 c')
        + rearrange(rays_d, 'n1 c -> n1 1 c')
        * rearrange(zs, 'n1 n2 -> n1 n2 1')
    )
    results['xyzs_fine'] = xyz

    xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c')
    xyz_embedded = embedding_xyz(xyz_)

    # ---- static model (view-dependent rgb + sigma) ----
    dir_embedded_ = repeat(
        embedding_dir(rays_d), 'n1 c -> (n1 n2) c', n2=N_samples)
    out = model_static(torch.cat([xyz_embedded, dir_embedded_], 1))
    out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples)
    results['static_rgbs_fine'] = static_rgbs = out[..., :3]
    static_sigmas = out[..., 3]

    # ---- dynamic model (view-independent rgb + sigma + scene flow) ----
    flow_heads = tuple(f for f in ('fw', 'bw') if f in output_transient_flow)
    t_embedded = embeddings['transient'](ts)  # (N_rays, in_ch_t)
    t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples)
    out = model_dynamic(
        torch.cat([xyz_embedded, t_embedded_], 1), output_flow=flow_heads)
    out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples)
    results['transient_rgbs_fine'] = transient_rgbs = out[..., :3]
    transient_sigmas = out[..., 3]
    if flow_heads:
        results['transient_flows_fw'] = transient_flows_fw = out[..., 4:7]
        results['transient_flows_bw'] = transient_flows_bw = out[..., 7:10]
        transient_flows_fw[zs > Z_FAR] = 0
        transient_flows_bw[zs > Z_FAR] = 0

    # visibility culling: suppress dynamic content that no training camera
    # sees at this time (removes ghosting when the render pose leaves the
    # training trajectory)
    if test_time and dataset is not None:
        K = torch.FloatTensor(dataset.K).to(xyz.device)
        W, H = dataset.img_wh
        visibility = torch.zeros(len(xyz_), device=xyz.device)
        xyz_w = ndc2world(xyz_, K)
        c2w_t = torch.FloatTensor(dataset.poses[int(ts[0])]).to(xyz.device)
        compute_world_visibility(visibility, xyz_w, K, H, W, c2w_t)
        transient_sigmas = transient_sigmas.clone()
        transient_sigmas[visibility.view_as(transient_sigmas) == 0] = -10

    # ---- composite ----
    deltas = zs[:, 1:] - zs[:, :-1]  # (N_rays, N_samples-1)
    # the static background is opaque at infinity, the dynamic object is thin
    static_deltas = torch.cat([deltas, 100 * torch.ones_like(deltas[:, :1])],
                              -1)
    transient_deltas = torch.cat(
        [deltas, 1e-3 * torch.ones_like(deltas[:, :1])], -1)

    static_sigmas = act(
        static_sigmas + torch.randn_like(static_sigmas) * noise_std)
    results['static_sigmas_fine'] = static_sigmas
    static_alphas = 1 - torch.exp(-static_deltas * static_sigmas)

    transient_sigmas = act(
        transient_sigmas + torch.randn_like(transient_sigmas) * noise_std)
    results['transient_sigmas_fine'] = transient_sigmas
    transient_alphas = 1 - torch.exp(-transient_deltas * transient_sigmas)

    alphas = 1 - (1 - static_alphas) * (1 - transient_alphas)

    # warped renderings for the flow/photometric losses (training only)
    if (not test_time) and flow_heads:

        def render_transient_warping(xyz_warped, t_emb, flow):
            """
            Render the dynamic model at @xyz_warped / time embedding @t_emb,
            composited with the CURRENT time's static field. Also returns the
            warped points' fw/bw flow and the warped transient weights (used
            to infer occlusion).
            """
            inp = torch.cat([
                embedding_xyz(rearrange(xyz_warped, 'n1 n2 c -> (n1 n2) c')),
                repeat(t_emb, 'n1 c -> (n1 n2) c', n2=N_samples),
            ], 1)
            out_w = model_dynamic(inp, output_flow=(flow,))
            out_w = rearrange(
                out_w, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples)
            rgbs_w = out_w[..., :3]
            sigmas_w = out_w[..., 3]
            flows_w = out_w[..., 4:7]
            flows_w[zs > Z_FAR] = 0

            noise = torch.randn_like(sigmas_w) * noise_std
            alphas_w = 1 - torch.exp(-transient_deltas * act(sigmas_w + noise))
            alphas_comp = 1 - (1 - static_alphas) * (1 - alphas_w)
            alphas_comp_sh = torch.cat(
                [torch.ones_like(alphas_comp[:, :1]), 1 - alphas_comp], -1)
            transmittance_w = torch.cumprod(alphas_comp_sh[:, :-1], -1)
            static_weights_w = static_alphas * transmittance_w
            transient_weights_w = alphas_w * transmittance_w
            rgb_map_w = (
                reduce(static_weights_w[..., None] * static_rgbs,
                       'n1 n2 c -> n1 c', 'sum')
                + reduce(transient_weights_w[..., None] * rgbs_w,
                         'n1 n2 c -> n1 c', 'sum')
            )
            return rgb_map_w, flows_w, transient_weights_w

        results['xyzs_fw'] = xyz_fw = xyz + transient_flows_fw
        tp1_embedded = embeddings['transient'](torch.clamp(ts + 1, max=max_t))
        results['rgb_fw'], transient_flows_fw_bw, transient_weights_fw = \
            render_transient_warping(xyz_fw, tp1_embedded, 'bw')

        results['xyzs_bw'] = xyz_bw = xyz + transient_flows_bw
        tm1_embedded = embeddings['transient'](torch.clamp(ts - 1, min=0))
        results['rgb_bw'], transient_flows_bw_fw, transient_weights_bw = \
            render_transient_warping(xyz_bw, tm1_embedded, 'fw')

        # to compute fw-bw cycle consistency
        results['xyzs_fw_bw'] = xyz_fw + transient_flows_fw_bw
        results['xyzs_bw_fw'] = xyz_bw + transient_flows_bw_fw

    alphas_sh = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas], -1)
    transmittance = torch.cumprod(alphas_sh[:, :-1], -1)

    static_weights = static_alphas * transmittance
    transient_weights = transient_alphas * transmittance
    weights = alphas * transmittance
    weights_ = rearrange(weights, 'n1 n2 -> n1 n2 1')

    results['static_weights_fine'] = static_weights
    results['transient_weights_fine'] = transient_weights
    results['weights_fine'] = weights
    if test_time:
        results['static_alphas_fine'] = static_alphas
        results['transient_alphas_fine'] = transient_alphas

    results['depth_fine'] = reduce(weights * zs, 'n1 n2 -> n1', 'sum')
    static_rgb_map = reduce(
        static_weights[..., None] * static_rgbs, 'n1 n2 c -> n1 c', 'sum')
    transient_rgb_map = reduce(
        transient_weights[..., None] * transient_rgbs,
        'n1 n2 c -> n1 c', 'sum')
    results['rgb_fine'] = static_rgb_map + transient_rgb_map
    results['transient_alpha_fine'] = \
        reduce(transient_weights, 'n1 n2 -> n1', 'sum')
    results['transient_rgb_fine'] = transient_rgb_map + \
        0.8 * (1 - rearrange(results['transient_alpha_fine'], 'n1 -> n1 1'))

    # depth and rgb of the static field alone (its own transmittance)
    static_alphas_sh = torch.cat(
        [torch.ones_like(static_alphas[:, :1]), 1 - static_alphas], -1)
    static_transmittance = torch.cumprod(static_alphas_sh[:, :-1], -1)
    _static_weights = static_alphas * static_transmittance
    results['_static_rgb_fine'] = reduce(
        _static_weights[..., None] * static_rgbs, 'n1 n2 c -> n1 c', 'sum')
    results['_static_depth_fine'] = reduce(
        _static_weights * zs, 'n1 n2 -> n1', 'sum')

    if flow_heads:
        results['xyz_fine'] = reduce(weights_ * xyz, 'n1 n2 c -> n1 c', 'sum')
        results['transient_flow_fw'] = reduce(
            weights_ * transient_flows_fw, 'n1 n2 c -> n1 c', 'sum')
        results['xyz_fw'] = results['xyz_fine'] + results['transient_flow_fw']
        results['transient_flow_bw'] = reduce(
            weights_ * transient_flows_bw, 'n1 n2 c -> n1 c', 'sum')
        results['xyz_bw'] = results['xyz_fine'] + results['transient_flow_bw']

        if (not test_time) and 'disocc' in output_transient_flow:
            # occlusion inferred from the difference between warped and
            # reference transient weights (no learned head)
            occ_fw = (transient_weights_fw - transient_weights).detach()
            occ_bw = (transient_weights_bw - transient_weights).detach()
            results['disocc_fw'] = \
                1 - torch.abs(reduce(occ_fw, 'n1 n2 -> n1 1', 'sum'))
            results['disoccs_fw'] = \
                1 - torch.abs(rearrange(occ_fw, 'n1 n2 -> n1 n2 1'))
            results['disocc_bw'] = \
                1 - torch.abs(reduce(occ_bw, 'n1 n2 -> n1 1', 'sum'))
            results['disoccs_bw'] = \
                1 - torch.abs(rearrange(occ_bw, 'n1 n2 -> n1 n2 1'))

    return results


def interpolate(results_t, results_tp1, dt, K, c2w, img_wh):
    """
    Interpolate between two results t and t+1 to produce t+dt, dt in (0, 1).
    For each sample on the ray (the sample points lie on the same distances,
    so they actually form planes), compute the optical flow on this plane,
    then use softsplat to splat the flows. Finally use MPI technique to
    compute the composite image. Used in test time only.

    Inputs:
        results_t, results_tp1: dictionaries of the @render_rays function
            (with test_time=True, buffers on cpu).
        dt: float in (0, 1)
        K: (3, 3)
            intrinsics matrix (MUST BE THE SAME for results_t and results_tp1!)
        c2w: (3, 4)
            current pose (MUST BE THE SAME for results_t and results_tp1!)
        img_wh: image width and height

    Outputs:
        (img_wh[1], img_wh[0], 3) rgb interpolation result
        (img_wh[1], img_wh[0]) depth of the interpolation (in NDC)
    """
    # softsplat needs cupy, which is only required for test-time
    # interpolation; import lazily so training works without it
    from utils.softsplat import FunctionSoftsplat

    device = K.device
    N_rays, N_samples = results_t['xyzs_fine'].shape[:2]
    w, h = img_wh
    rgba = torch.zeros((h, w, 4))
    depth = torch.zeros((h, w))

    c2w_ = torch.eye(4, device=device)
    c2w_[:3] = c2w
    w2c = torch.inverse(c2w_)[:3]
    w2c[1:] *= -1  # "right up back" to "right down forward" for cam projection
    P = K @ w2c  # (3, 4) projection matrix
    grid = create_meshgrid(h, w, normalized_coordinates=False).to(device)

    xyzs = results_t['xyzs_fine'].to(device)
    zs = rearrange(results_t['zs_fine'], '(h w) n2 -> h w n2', w=w, h=h)

    # static buffers
    static_rgb = rearrange(results_t['static_rgbs_fine'],
                           '(h w) n2 c -> h w n2 c', w=w, h=h, c=3)
    static_a = rearrange(results_t['static_alphas_fine'],
                         '(h w) n2 -> h w n2 1', w=w, h=h)

    # compute forward buffers
    xyzs_w = ndc2world(rearrange(xyzs, 'n1 n2 c -> (n1 n2) c'), K)
    xyzs_fw_w = ndc2world(
        rearrange(xyzs + results_t['transient_flows_fw'].to(device),
                  'n1 n2 c -> (n1 n2) c'), K)  # fw points with full flow
    xyzs_fw_w = xyzs_w + dt * (xyzs_fw_w - xyzs_w)  # scale the flow with dt
    uvds_fw = P[:3, :3] @ rearrange(xyzs_fw_w, 'n c -> c n') + P[:3, 3:]
    uvs_fw = uvds_fw[:2] / uvds_fw[2]
    uvs_fw = rearrange(uvs_fw, 'c (n1 n2) -> c n1 n2', n1=N_rays, n2=N_samples)
    uvs_fw = rearrange(uvs_fw, 'c (h w) n2 -> n2 h w c', w=w, h=h)
    of_fw = rearrange(uvs_fw - grid, 'n2 h w c -> n2 c h w', c=2)

    transient_rgb_t = rearrange(results_t['transient_rgbs_fine'],
                                '(h w) n2 c -> n2 c h w', w=w, h=h, c=3)
    transient_a_t = rearrange(results_t['transient_alphas_fine'],
                              '(h w) n2 -> n2 1 h w', w=w, h=h)
    transient_rgba_t = torch.cat([transient_rgb_t, transient_a_t], 1)

    # compute backward buffers
    xyzs_bw_w = ndc2world(
        rearrange(xyzs + results_tp1['transient_flows_bw'].to(device),
                  'n1 n2 c -> (n1 n2) c'), K)  # bw points with full flow
    xyzs_bw_w = xyzs_w + (1 - dt) * (xyzs_bw_w - xyzs_w)
    uvds_bw = P[:3, :3] @ rearrange(xyzs_bw_w, 'n c -> c n') + P[:3, 3:]
    uvs_bw = uvds_bw[:2] / uvds_bw[2]
    uvs_bw = rearrange(uvs_bw, 'c (n1 n2) -> c n1 n2', n1=N_rays, n2=N_samples)
    uvs_bw = rearrange(uvs_bw, 'c (h w) n2 -> n2 h w c', w=w, h=h)
    of_bw = rearrange(uvs_bw - grid, 'n2 h w c -> n2 c h w', c=2)

    transient_rgb_tp1 = rearrange(results_tp1['transient_rgbs_fine'],
                                  '(h w) n2 c -> n2 c h w', w=w, h=h, c=3)
    transient_a_tp1 = rearrange(results_tp1['transient_alphas_fine'],
                                '(h w) n2 -> n2 1 h w', w=w, h=h)
    transient_rgba_tp1 = torch.cat([transient_rgb_tp1, transient_a_tp1], 1)

    for s in range(N_samples):  # compute MPI planes (front to back)
        transient_rgba_fw = FunctionSoftsplat(
            tenInput=transient_rgba_t[s:s+1].to(device).contiguous(),
            tenFlow=of_fw[s:s+1].contiguous(),
            tenMetric=None,
            strType='average').cpu()
        transient_rgba_fw = rearrange(transient_rgba_fw, '1 c h w -> h w c')

        transient_rgba_bw = FunctionSoftsplat(
            tenInput=transient_rgba_tp1[s:s+1].to(device).contiguous(),
            tenFlow=of_bw[s:s+1].contiguous(),
            tenMetric=None,
            strType='average').cpu()
        transient_rgba_bw = rearrange(transient_rgba_bw, '1 c h w -> h w c')

        composed_rgb = \
            transient_rgba_fw[..., :3] * transient_rgba_fw[..., 3:] * (1-dt) \
            + transient_rgba_bw[..., :3] * transient_rgba_bw[..., 3:] * dt \
            + static_rgb[:, :, s] * static_a[:, :, s]
        composed_a = 1 - (
            1 - (transient_rgba_fw[..., 3:] * (1 - dt)
                 + transient_rgba_bw[..., 3:] * dt)
        ) * (1 - static_a[:, :, s])
        rgba[..., :3] += (1 - rgba[..., 3:]) * composed_rgb
        depth += (1 - rgba[..., 3]) * composed_a[..., 0] * zs[..., s]
        rgba[..., 3:] += (1 - rgba[..., 3:]) * composed_a

    return rgba[..., :3], depth

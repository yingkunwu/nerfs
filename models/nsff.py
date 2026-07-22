import torch
from torch import nn


class Embedding(nn.Module):
    """
    Embeds input to (x, sin(2^k x), cos(2^k x), ...).
    """

    def __init__(self,
                 input_dim,
                 max_freq_log2,
                 num_freqs,
                 logscale=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.funcs = [torch.sin, torch.cos]
        self.output_dim = input_dim * (len(self.funcs) * num_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** max_freq_log2, num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: Tensor of shape (B, input_dim)
        Outputs:
            Tensor of shape (B, output_dim)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)


class NeRF_Static(nn.Module):
    """
    NeRF network of the static (time-invariant) scene representation.
    Follows nsff_pl's static branch: sigma from the xyz features, rgb from a
    single view-dependent layer. There is NO learned blending weight — the
    static and dynamic fields are composited additively (NeRF-W style) in the
    renderer.
    """

    def __init__(self,
                 depth=8,
                 width=256,
                 in_ch_xyz=63,
                 in_ch_dir=27,
                 skips=(4,)):
        super().__init__()
        self.depth = depth
        self.width = width
        self.in_ch_xyz = in_ch_xyz
        self.in_ch_dir = in_ch_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(depth):
            if i == 0:
                lin = nn.Linear(in_ch_xyz, width)
            elif i in skips:
                lin = nn.Linear(width + in_ch_xyz, width)
            else:
                lin = nn.Linear(width, width)
            block = nn.Sequential(lin, nn.ReLU(inplace=True))
            setattr(self, f"xyz_encoding_{i}", block)

        self.xyz_final = nn.Linear(width, width)
        self.dir_encoding = nn.Sequential(
            nn.Linear(width + in_ch_dir, width),
            nn.ReLU(inplace=True),
        )

        # outputs (raw sigma; Softplus is applied in the renderer)
        self.sigma = nn.Linear(width, 1)
        self.rgb = nn.Sequential(
            nn.Linear(width, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xyz, dirs = torch.split(
            x,
            [self.in_ch_xyz, self.in_ch_dir],
            dim=-1
        )

        h = xyz
        for i in range(self.depth):
            if i in self.skips:
                h = torch.cat((xyz, h), dim=-1)
            h = getattr(self, f"xyz_encoding_{i}")(h)

        sigma = self.sigma(h)

        h_final = self.xyz_final(h)
        h_dir = self.dir_encoding(torch.cat((h_final, dirs), dim=-1))
        rgb = self.rgb(h_dir)

        return torch.cat((rgb, sigma), dim=-1)


class NeRF_Dynamic(nn.Module):
    """
    NeRF network of the dynamic (time-variant) scene representation.
    Follows nsff_pl's transient branch: conditioned on a learned per-frame
    time embedding, VIEW-INDEPENDENT rgb, raw sigma (Softplus in the
    renderer), and forward/backward scene flow through separate
    `flow_scale * tanh` heads. There is no learned disocclusion head —
    occlusion weights are inferred from the warped rendering weights.
    """

    def __init__(self,
                 depth=8,
                 width=256,
                 in_ch_xyz=63,
                 in_ch_t=48,
                 skips=(4,),
                 flow_scale=0.2):
        super().__init__()
        self.depth = depth
        self.width = width
        self.in_ch_xyz = in_ch_xyz
        self.in_ch_t = in_ch_t
        self.skips = skips
        self.flow_scale = flow_scale

        # xyz+t encoding layers
        for i in range(depth):
            if i == 0:
                lin = nn.Linear(in_ch_xyz + in_ch_t, width)
            elif i in skips:
                lin = nn.Linear(width + in_ch_xyz + in_ch_t, width)
            else:
                lin = nn.Linear(width, width)
            block = nn.Sequential(lin, nn.ReLU(inplace=True))
            setattr(self, f"xyz_encoding_{i}", block)

        self.xyz_final = nn.Linear(width, width)

        # outputs
        self.sigma = nn.Linear(width, 1)
        self.rgb = nn.Sequential(
            nn.Linear(width, 3),
            nn.Sigmoid(),
        )
        self.flow_fw = nn.Sequential(nn.Linear(width, 3), nn.Tanh())
        self.flow_bw = nn.Sequential(nn.Linear(width, 3), nn.Tanh())

    def forward(self, x, output_flow=('fw', 'bw')):
        """
        Inputs:
            x: (B, in_ch_xyz + in_ch_t) embedded position and time
            output_flow: which scene-flow heads to evaluate; the outputs are
                concatenated in the given order after (rgb, sigma)
        """
        h = x
        for i in range(self.depth):
            if i in self.skips:
                h = torch.cat((x, h), dim=-1)
            h = getattr(self, f"xyz_encoding_{i}")(h)

        h_final = self.xyz_final(h)
        sigma = self.sigma(h_final)
        rgb = self.rgb(h_final)

        out = [rgb, sigma]
        if 'fw' in output_flow:
            out.append(self.flow_scale * self.flow_fw(h_final))
        if 'bw' in output_flow:
            out.append(self.flow_scale * self.flow_bw(h_final))

        return torch.cat(out, dim=-1)

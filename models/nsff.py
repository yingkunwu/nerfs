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
    NeRF network of static (time-invariant) scene representation.
    We predict an extra blending weight field 'v' from intermediate features
    along with opacity 'sigma' for blending with transient components.
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

        # outputs
        self.sigma = nn.Sequential(
            nn.Linear(width, 1),
            nn.Softplus()
        )
        self.rgb = nn.Sequential(
            nn.Linear(width + in_ch_dir, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),
        )
        self.v = nn.Sequential(
            nn.Linear(width, 1),
            nn.Sigmoid()
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
        v = self.v(h)

        h_final = self.xyz_final(h)
        d_in = torch.cat((h_final, dirs), dim=-1)
        rgb = self.rgb(d_in)

        return torch.cat((rgb, sigma, v), dim=-1)


class NeRF_Dynamic(nn.Module):
    """
    NeRF network of dynamic (time-variant) scene representation.
    We encode an input time indices 'i' into the MLP and predict time-dependent
    scene flow fields F_i and disocclusionweight fields W_i from the
    intermediate features along with opacity 'sigma'.
    """

    def __init__(self,
                 depth=4,
                 width=128,
                 in_ch_xyz=63,
                 in_ch_dir=27,
                 in_ch_t=16,
                 skips=(4,)):
        super().__init__()
        self.depth = depth
        self.width = width
        self.in_ch_xyz = in_ch_xyz
        self.in_ch_dir = in_ch_dir
        self.in_ch_t = in_ch_t
        self.skips = skips

        # xyz encoding layers
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
        self.sigma = nn.Sequential(
            nn.Linear(width, 1),
            nn.Softplus()
        )
        self.rgb = nn.Sequential(
            nn.Linear(width + in_ch_dir, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),
        )

        self.sf = nn.Sequential(
            nn.Linear(width, 6),
            nn.Tanh()
        )
        self.prob = nn.Sequential(
            nn.Linear(width, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        xyz_t, dirs = torch.split(
            x,
            [self.in_ch_xyz + self.in_ch_t, self.in_ch_dir],
            dim=-1
        )

        h = xyz_t
        for i in range(self.depth):
            if i in self.skips:
                h = torch.cat((xyz_t, h), dim=-1)
            h = getattr(self, f"xyz_encoding_{i}")(h)

        sigma = self.sigma(h)
        sf = self.sf(h)
        prob = self.prob(h)

        h_final = self.xyz_final(h)
        d_in = torch.cat((h_final, dirs), dim=-1)
        rgb = self.rgb(d_in)

        return torch.cat((rgb, sigma, sf, prob), dim=-1)

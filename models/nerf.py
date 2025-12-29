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


class NeRF(nn.Module):
    """
    NeRF model that predicts density (sigma) and RGB.
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
        self.sigma = nn.Linear(width, 1)
        self.rgb = nn.Sequential(
            nn.Linear(width + in_ch_dir, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, sigma_only=False):
        """
        Encodes x and returns sigma (and rgb if requested).

        Inputs:
            x: Tensor of shape (B, in_ch_xyz [+ in_ch_dir])
            sigma_only: bool, if True returns only sigma
        Outputs:
            sigma or concat(rgb, sigma)
        """
        if sigma_only:
            xyz = x
        else:
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
        if sigma_only:
            return sigma

        h_final = self.xyz_final(h)
        d_in = torch.cat((h_final, dirs), dim=-1)
        rgb = self.rgb(d_in)

        return torch.cat((rgb, sigma), dim=-1)

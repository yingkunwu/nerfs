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


class NeRFW(nn.Module):
    def __init__(self,
                 typ='coarse',
                 output_transient=True,
                 depth=8,
                 width=256,
                 skips=(4,),
                 in_ch_xyz=63,
                 in_ch_dir=27,
                 in_ch_a=48,
                 in_ch_t=16,
                 beta_min=0.03):
        """
        depth: number of layers for density (sigma) encoder
        width: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_ch_xyz: number of input channels for xyz
        in_ch_dir: number of input channels for direction
        in_ch_a: appearance embedding dimension. n^(a) in the paper
        in_ch_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.typ = typ
        self.output_transient = output_transient
        self.depth = depth
        self.width = width
        self.skips = skips
        self.in_ch_xyz = in_ch_xyz
        self.in_ch_dir = in_ch_dir
        self.in_ch_a = in_ch_a
        self.in_ch_t = in_ch_t
        self.beta_min = beta_min

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
        self.static_sigma = nn.Sequential(
            nn.Linear(width, 1),
            nn.Softplus()
        )
        self.static_rgb = nn.Sequential(
            nn.Linear(width + in_ch_dir + self.in_ch_a, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),
            nn.Sigmoid(),
        )

        if output_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                nn.Linear(width + in_ch_t, width // 2),
                nn.ReLU(True),
                nn.Linear(width // 2, width // 2),
                nn.ReLU(True),
                nn.Linear(width // 2, width // 2),
                nn.ReLU(True),
                nn.Linear(width // 2, width // 2),
                nn.ReLU(True)
            )
            # transient output layers
            self.transient_sigma = nn.Sequential(
                nn.Linear(width // 2, 1),
                nn.Softplus()
            )
            self.transient_rgb = nn.Sequential(
                nn.Linear(width // 2, 3),
                nn.Sigmoid()
            )
            self.transient_beta = nn.Sequential(
                nn.Linear(width // 2, 1),
                nn.Softplus()
            )

    def forward(self, x, sigma_only=False):
        """
        Encode input (xyz + dir) to rgb and sigma.

        x: embedded vector of position (+ direction + appearance + transient)
        sigma_only: if True, infer sigma only
        """
        if sigma_only:
            input_xyz = x
        elif self.output_transient:
            input_xyz, input_dir_a, input_t = torch.split(
                x,
                [self.in_ch_xyz, self.in_ch_dir + self.in_ch_a, self.in_ch_t],
                dim=-1,
            )
        else:
            input_xyz, input_dir_a = torch.split(
                x,
                [self.in_ch_xyz, self.in_ch_dir + self.in_ch_a],
                dim=-1,
            )

        xyz_ = input_xyz
        for i in range(self.depth):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], dim=-1)
            xyz_ = getattr(self, f"xyz_encoding_{i}")(xyz_)

        static_sigma = self.static_sigma(xyz_)  # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_final(xyz_)

        dir_encoding_input = torch.cat(
            [xyz_encoding_final, input_dir_a], dim=-1
        )
        static_rgb = self.static_rgb(dir_encoding_input)
        static = torch.cat([static_rgb, static_sigma], dim=-1)  # (B, 4)

        if not self.output_transient:
            return static

        tra_encoding_input = torch.cat(
            [xyz_encoding_final, input_t], dim=-1
        )
        tra_encoding = self.transient_encoding(tra_encoding_input)
        transient_sigma = self.transient_sigma(tra_encoding)  # (B, 1)
        transient_rgb = self.transient_rgb(tra_encoding)  # (B, 3)
        transient_beta = self.transient_beta(tra_encoding)  # (B, 1)

        transient = torch.cat(
            [transient_rgb, transient_sigma, transient_beta], dim=-1
        )  # (B, 5)
        return torch.cat([static, transient], dim=-1)  # (B, 9)

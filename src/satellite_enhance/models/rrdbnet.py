"""RRDB-based generator for 2D satellite image spatial super-resolution.

Architecture follows the Residual-in-Residual Dense Block design popularized by
ESRGAN (Wang et al., 2018) and Real-ESRGAN (Wang et al., 2021), adapted for
multi-band (RGB or multispectral) satellite imagery. The network is fully
convolutional so it accepts arbitrary tile sizes and any integer upscale
factor that is a power of two (2x, 4x, 8x).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """5-conv dense block with residual scaling, the core unit of an RRDB."""

    def __init__(self, channels: int = 64, growth: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        convs = []
        for i in range(5):
            in_ch = channels + i * growth
            out_ch = growth if i < 4 else channels
            convs.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.convs = nn.ModuleList(convs)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for i, conv in enumerate(self.convs):
            inp = torch.cat(feats, dim=1)
            out = conv(inp)
            if i < 4:
                out = self.act(out)
            feats.append(out)
        return feats[-1] * self.res_scale + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: three stacked dense blocks + long skip."""

    def __init__(self, channels: int = 64, growth: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.blocks = nn.Sequential(
            DenseBlock(channels, growth, res_scale),
            DenseBlock(channels, growth, res_scale),
            DenseBlock(channels, growth, res_scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x) * self.res_scale + x


class PixelShuffleUpsample(nn.Module):
    """One 2x upsampling stage via sub-pixel convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class RRDBNet(nn.Module):
    """Generator network: shallow feature extraction -> RRDB trunk -> upsampling -> reconstruction.

    Args:
        in_channels: number of input spectral bands (3 for RGB, more for multispectral).
        out_channels: number of output bands (usually same as in_channels).
        channels: base feature width.
        num_blocks: number of stacked RRDB units in the trunk.
        growth: growth channels inside each dense block.
        scale: spatial upscale factor, must be 1, 2, 4, or 8.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 16,
        growth: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        if scale not in (1, 2, 4, 8):
            raise ValueError("scale must be one of 1, 2, 4, 8")
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(channels, growth) for _ in range(num_blocks)])
        self.conv_trunk = nn.Conv2d(channels, channels, 3, 1, 1)

        n_up = 0 if scale == 1 else {2: 1, 4: 2, 8: 3}[scale]
        self.upsample = nn.Sequential(*[PixelShuffleUpsample(channels) for _ in range(n_up)])

        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act_hr = nn.LeakyReLU(0.2, inplace=True)
        self.conv_last = nn.Conv2d(channels, out_channels, 3, 1, 1)

        # Bicubic-upsampled skip connection: the network only has to learn a
        # residual correction on top of a safe, smooth baseline, the same
        # design used by DEMSRNet (models/dem_srnet.py). Without this, a
        # network trained on a small/narrow corpus has no floor forcing it
        # back toward "at least as good as bicubic" on out-of-distribution
        # input -- measured evidence (scripts/evaluate.py on genuinely
        # held-out imagery) showed the un-skipped version actively making
        # images worse than plain bicubic upscaling instead of leaving
        # unfamiliar content alone.
        self.baseline_upsample = (
            nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False) if scale > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = self.baseline_upsample(x)
        feat = self.conv_first(x)
        trunk_out = self.conv_trunk(self.trunk(feat))
        feat = feat + trunk_out
        feat = self.upsample(feat)
        residual = self.conv_last(self.act_hr(self.conv_hr(feat)))
        return baseline + residual

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

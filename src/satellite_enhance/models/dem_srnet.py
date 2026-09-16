"""Digital Elevation Model (DEM) spatial super-resolution network.

'3D spatial enhancement' for satellite products means increasing the spatial
resolution of the terrain elevation raster (e.g. going from a coarse 90m/30m
DEM such as SRTM to a finer effective grid spacing), so that when combined
with an enhanced 2D optical image it produces a materially more detailed 3D
terrain reconstruction. The network below is a compact residual CNN operating
on a single elevation channel; it is intentionally shallower than the RGB
generator since elevation fields are smoother and lower-entropy than optical
imagery, and it is trained with an added slope/gradient consistency loss
(see losses/losses.py) so ridgelines and drainage patterns stay sharp instead
of being smoothed away.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class PixelShuffleUpsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shuffle(self.conv(x)))


class DEMSRNet(nn.Module):
    """Single-channel elevation super-resolution network.

    Input: low-resolution DEM tile, shape (B, 1, H, W), elevation in meters
    (caller should normalize/denormalize -- see utils.normalize_dem).
    Output: high-resolution DEM tile, shape (B, 1, H*scale, W*scale).
    """

    def __init__(self, channels: int = 48, num_blocks: int = 8, scale: int = 4):
        super().__init__()
        if scale not in (1, 2, 4, 8):
            raise ValueError("scale must be one of 1, 2, 4, 8")
        self.scale = scale
        self.head = nn.Conv2d(1, channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        n_up = 0 if scale == 1 else {2: 1, 4: 2, 8: 3}[scale]
        self.upsample = nn.Sequential(*[PixelShuffleUpsample(channels) for _ in range(n_up)])
        self.tail = nn.Conv2d(channels, 1, 3, 1, 1)

        # Bicubic-like bilinear skip connection so the network only needs to
        # learn a residual correction on top of a smooth upsampled baseline,
        # which stabilizes training on small terrain datasets.
        self.baseline_upsample = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = self.baseline_upsample(x) if self.scale > 1 else x
        feat = self.head(x)
        feat = feat + self.body_conv(self.body(feat))
        feat = self.upsample(feat)
        residual = self.tail(feat)
        return baseline + residual

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

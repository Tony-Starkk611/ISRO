"""PatchGAN-style discriminator used for optional adversarial fine-tuning of the
2D super-resolution generator (sharper high-frequency detail than pixel-loss-only
training, at the cost of training stability)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_ch: int, out_ch: int, stride: int, use_bn: bool = True) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 3, stride, 1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class VGGStyleDiscriminator(nn.Module):
    """Convolutional discriminator classifying HR-sized patches as real/fake."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        layers: list[nn.Module] = _block(in_channels, c, 1, use_bn=False)
        layers += _block(c, c, 2)
        layers += _block(c, c * 2, 1)
        layers += _block(c * 2, c * 2, 2)
        layers += _block(c * 2, c * 4, 1)
        layers += _block(c * 4, c * 4, 2)
        layers += _block(c * 4, c * 8, 1)
        layers += _block(c * 8, c * 8, 2)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8 * 4 * 4, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.pool(feat)
        return self.classifier(feat)

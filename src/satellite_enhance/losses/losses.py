"""Loss functions for 2D image super-resolution and 3D DEM enhancement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss, more robust to outliers than plain MSE and known
    to produce sharper edges than L2 for image restoration tasks."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class GradientLoss(nn.Module):
    """Penalizes differences in horizontal/vertical finite-difference gradients.

    For optical imagery this discourages blurry outputs; for DEM tiles it
    directly preserves slope/ridge/drainage structure, which is the terrain
    feature most valuable to a "spatial enhancement" product but the first
    thing a plain pixelwise loss smooths away.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def grads(t: torch.Tensor):
            gx = t[..., :, 1:] - t[..., :, :-1]
            gy = t[..., 1:, :] - t[..., :-1, :]
            return gx, gy

        pgx, pgy = grads(pred)
        tgx, tgy = grads(target)
        return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


class VGGPerceptualLoss(nn.Module):
    """Feature-space loss using a small pretrained-free CNN feature extractor.

    A full VGG19-based perceptual loss requires downloading ImageNet-pretrained
    weights; to keep this pipeline fully self-contained and runnable offline
    once cloned, we instead use a lightweight randomly-initialized-but-frozen
    multi-scale edge/texture extractor (fixed Sobel + Laplacian filter bank).
    This still captures structural/textural similarity beyond raw pixels and
    can be swapped for `torchvision.models.vgg19(pretrained=True)` features
    if internet access to pretrained weights is available at train time.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        kernels = torch.stack([sobel_x, sobel_y, laplacian])[:, None].repeat(1, in_channels, 1, 1)
        self.register_buffer("kernels", kernels)
        self.in_channels = in_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pf = F.conv2d(pred, self.kernels, padding=1)
        tf = F.conv2d(target, self.kernels, padding=1)
        return F.l1_loss(pf, tf)


class AdversarialLoss(nn.Module):
    """Non-saturating GAN loss (BCE-with-logits) for generator/discriminator."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_loss = self.bce(real_logits, torch.ones_like(real_logits))
        fake_loss = self.bce(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) * 0.5

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return self.bce(fake_logits, torch.ones_like(fake_logits))


class SRCompositeLoss(nn.Module):
    """Weighted combination of pixel, gradient, and perceptual losses used
    for both the 2D image generator and the 3D DEM network (the DEM network
    is trained with weight_perceptual=0 since edge filters on single-channel
    elevation are already covered by the gradient term)."""

    def __init__(
        self,
        in_channels: int = 3,
        weight_pixel: float = 1.0,
        weight_gradient: float = 0.1,
        weight_perceptual: float = 0.05,
    ):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.gradient_loss = GradientLoss()
        self.use_perceptual = weight_perceptual > 0
        if self.use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss(in_channels)
        self.w_pixel = weight_pixel
        self.w_grad = weight_gradient
        self.w_perc = weight_perceptual

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        loss_pixel = self.pixel_loss(pred, target)
        loss_grad = self.gradient_loss(pred, target)
        total = self.w_pixel * loss_pixel + self.w_grad * loss_grad
        parts = {"pixel": loss_pixel.item(), "gradient": loss_grad.item()}
        if self.use_perceptual:
            loss_perc = self.perceptual_loss(pred, target)
            total = total + self.w_perc * loss_perc
            parts["perceptual"] = loss_perc.item()
        parts["total"] = total.item()
        return total, parts

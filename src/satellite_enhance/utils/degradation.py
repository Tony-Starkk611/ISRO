"""Synthetic degradation pipeline for generating paired (LR, HR) training data
from high-resolution-only imagery.

Paired real low/high resolution satellite scenes of the exact same ground
footprint are rare and hard to source without licensing. The standard
practice in the super-resolution literature (SRCNN, ESRGAN, Real-ESRGAN,
BSRGAN) is instead to start from real high-resolution imagery and
synthesize a realistic low-resolution counterpart by composing:
blur -> downsample -> sensor noise -> mild JPEG-style compression artifacts.
Training on this synthetic-but-realistic degradation model transfers well to
real low-resolution satellite input at inference time.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def random_degrade(hr: np.ndarray, scale: int, rng: random.Random | None = None) -> np.ndarray:
    """Turn an HR image (H, W, C) float32 in [0, 1] into a degraded LR image
    of shape (H/scale, W/scale, C), simulating sensor blur, downsampling,
    noise, and compression.
    """
    rng = rng or random.Random()
    h, w = hr.shape[:2]

    sigma = rng.uniform(0.6, 1.8)
    blurred = gaussian_filter(hr, sigma=(sigma, sigma, 0))

    lr_h, lr_w = h // scale, w // scale
    img = Image.fromarray((np.clip(blurred, 0, 1) * 255).astype(np.uint8))
    img = img.resize((lr_w, lr_h), Image.BICUBIC)

    if rng.random() < 0.7:
        quality = rng.randint(45, 90)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert(img.mode)

    lr = np.asarray(img).astype(np.float32) / 255.0

    noise_sigma = rng.uniform(0.002, 0.02)
    lr = lr + np.random.normal(0, noise_sigma, size=lr.shape)
    lr = np.clip(lr, 0.0, 1.0).astype(np.float32)
    return lr


def make_lr_hr_pair(hr: np.ndarray, scale: int, rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Crop HR to a multiple of `scale` and produce its degraded LR pair."""
    h, w = hr.shape[:2]
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    hr_cropped = hr[:h2, :w2]
    lr = random_degrade(hr_cropped, scale, rng)
    return lr, hr_cropped


def degrade_dem(hr_dem: np.ndarray, scale: int, rng: random.Random | None = None) -> np.ndarray:
    """Simulate a coarse-resolution DEM from a fine one: low-pass filter then
    block-average downsample (matches how real coarse DEM products such as
    SRTM90 relate to finer sources -- an averaging resample, not just blur).
    """
    rng = rng or random.Random()
    h, w = hr_dem.shape
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    dem = hr_dem[:h2, :w2]
    sigma = rng.uniform(0.5, 1.2)
    smoothed = gaussian_filter(dem, sigma=sigma)
    lr = smoothed.reshape(h2 // scale, scale, w2 // scale, scale).mean(axis=(1, 3))
    noise_sigma = rng.uniform(0.1, 0.6)
    lr = lr + np.random.normal(0, noise_sigma, size=lr.shape)
    return lr.astype(np.float32)

"""Regression tests guarding against the exact failure mode diagnosed and
fixed in this project: a 2D super-resolution checkpoint that makes images
worse than plain bicubic upscaling.

These tests are hermetic (no network access, no dependency on the
downloaded sample/held-out images) -- they build a small synthetic test
image procedurally so they run the same way in any clone of this repo.
The image deliberately mixes a smooth gradient region with a textured
region, since the diagnosed failure mode (hallucinating noise into smooth
content) specifically depends on having a smooth region to check.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from satellite_enhance.infer_2d import enhance_tiled, load_model
from satellite_enhance.metrics import psnr, ssim
from satellite_enhance.utils.degradation import random_degrade

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "checkpoints" / "rrdb_x4_best.pth",
    REPO_ROOT / "checkpoints" / "rrdb_x4.pth",
]


def _find_checkpoint() -> Path | None:
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return None


def _synthetic_hr_image(size: int = 128, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Smooth half: a soft gradient, like open water or a cloud deck -- flat
    # regions the model must NOT inject texture into.
    ys, xs = np.mgrid[0:size, 0 : size // 2]
    img[:, : size // 2, 0] = 0.15 + 0.05 * (ys / size)
    img[:, : size // 2, 1] = 0.25 + 0.05 * (xs / (size // 2))
    img[:, : size // 2, 2] = 0.45

    # Textured half: structured edges (like fields/urban blocks), the kind
    # of content the model should sharpen.
    block = rng.integers(0, 2, size=(size // 8, size // 16)).astype(np.float32)
    block = np.kron(block, np.ones((8, 8), dtype=np.float32))
    img[:, size // 2 :, 0] = 0.3 + 0.4 * block[:, : size // 2]
    img[:, size // 2 :, 1] = 0.35 + 0.3 * block[:, : size // 2]
    img[:, size // 2 :, 2] = 0.2

    return np.clip(img, 0, 1)


@pytest.mark.skipif(_find_checkpoint() is None, reason="no trained 2D checkpoint present")
def test_2d_model_beats_bicubic_on_synthetic_scene():
    checkpoint = _find_checkpoint()
    device = torch.device("cpu")
    model, scale = load_model(str(checkpoint), device)

    hr = _synthetic_hr_image(size=128)
    h2, w2 = (hr.shape[0] // scale) * scale, (hr.shape[1] // scale) * scale
    hr = hr[:h2, :w2]
    lr = random_degrade(hr, scale, rng=random.Random(42))

    from PIL import Image

    bicubic = np.asarray(
        Image.fromarray((lr * 255).astype(np.uint8)).resize((w2, h2), Image.BICUBIC)
    ).astype(np.float32) / 255.0

    with torch.no_grad():
        model_out = enhance_tiled(model, lr, scale, device, tile=256, overlap=16)

    assert model_out.shape == hr.shape
    assert model_out.dtype == np.float32
    assert model_out.min() >= 0.0 and model_out.max() <= 1.0

    bicubic_psnr = psnr(bicubic, hr)
    model_psnr = psnr(model_out, hr)
    bicubic_ssim = ssim(bicubic, hr)
    model_ssim = ssim(model_out, hr)

    # The regression this guards against: a checkpoint that scores *worse*
    # than doing nothing but a bicubic resize. A small tolerance (0.3 dB)
    # avoids flakiness from run-to-run checkpoint/hardware nondeterminism
    # without hiding a real regression.
    assert model_psnr >= bicubic_psnr - 0.3, (
        f"model PSNR ({model_psnr:.2f} dB) is worse than bicubic ({bicubic_psnr:.2f} dB) "
        "by more than the allowed tolerance -- this is the exact failure mode this project "
        "was fixed for; a new regression has likely been introduced."
    )
    assert model_ssim >= bicubic_ssim - 0.02

    # Specifically check the smooth (left) half for hallucinated texture:
    # the model's output there should not have drastically higher local
    # variance than the ground truth -- a proxy for "injecting noise into
    # flat regions" without needing a full no-reference IQA model.
    smooth_gt_std = hr[:, : w2 // 2].std()
    smooth_model_std = model_out[:, : w2 // 2].std()
    assert smooth_model_std <= smooth_gt_std * 2.5 + 0.02, (
        f"model injects much higher variance ({smooth_model_std:.4f}) than ground truth "
        f"({smooth_gt_std:.4f}) into a smooth region -- likely hallucinating texture into "
        "flat/uniform content (e.g. open water)."
    )

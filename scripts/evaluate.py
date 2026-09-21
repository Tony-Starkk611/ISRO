"""Quantitative evaluation of the 2D super-resolution model against a
bicubic-upscale baseline, using real PSNR/SSIM measured against genuine
high-resolution ground truth (never fabricated).

Methodology (standard practice in the SR literature -- EDSR/ESRGAN/etc.
evaluate the same way on Set5/Set14/Urban100): take a real HR image, apply
a defined, reproducible degradation to synthesize its LR counterpart, run
the candidate method to reconstruct an HR estimate, and compare that
estimate to the *original* HR image with PSNR/SSIM.

Two degradation modes are evaluated, deliberately, because they answer
different questions:
  - "matched"      -- the same blur+downsample+noise+JPEG pipeline used to
                       generate training pairs (utils/degradation.py). This
                       measures how well the model does on inputs shaped
                       like what it was trained on.
  - "bicubic_only" -- a plain bicubic downsample with no other degradation,
                       the classic academic SR-benchmark protocol. This
                       measures generalization to a *different, simpler*
                       degradation than training used -- a model overfit to
                       its training degradation can do worse than plain
                       bicubic upscaling here, which is a real, meaningful
                       failure mode worth surfacing rather than hiding.

Test images are split into:
  - "held_out"  -- never seen during training of the checkpoint(s) here.
  - "in_sample" -- were part of the training corpus; reported separately
                   and clearly labeled, never blended into the headline
                   held-out numbers, since an in-sample score is not
                   evidence of generalization.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/rrdb_x4.pth
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from satellite_enhance.infer_2d import enhance_tiled, load_model
from satellite_enhance.metrics import psnr, ssim
from satellite_enhance.utils.degradation import random_degrade
from satellite_enhance.utils.image_io import load_image

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TestImage:
    path: Path
    label: str  # "held_out" or "in_sample"
    name: str


TEST_IMAGES = [
    TestImage(REPO_ROOT / "data/heldout/gdal_small_world.png", "held_out", "gdal_small_world"),
    TestImage(REPO_ROOT / "data/heldout/gdal_rgbsmall.png", "held_out", "gdal_rgbsmall"),
    TestImage(REPO_ROOT / "data/sample/landsat_rockies_truecolor.png", "in_sample", "landsat_rockies"),
    TestImage(REPO_ROOT / "data/sample/landsat_sfbay_falsecolor.png", "in_sample", "landsat_sfbay"),
]


def bicubic_degrade(hr: np.ndarray, scale: int) -> np.ndarray:
    h, w = hr.shape[:2]
    img = Image.fromarray((np.clip(hr, 0, 1) * 255).astype(np.uint8))
    lr = img.resize((w // scale, h // scale), Image.BICUBIC)
    return np.asarray(lr).astype(np.float32) / 255.0


def bicubic_upscale(lr: np.ndarray, scale: int) -> np.ndarray:
    h, w = lr.shape[:2]
    img = Image.fromarray((np.clip(lr, 0, 1) * 255).astype(np.uint8))
    hr = img.resize((w * scale, h * scale), Image.BICUBIC)
    return np.asarray(hr).astype(np.float32) / 255.0


def make_crops(hr: np.ndarray, crop_size: int, n_crops: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    h, w = hr.shape[:2]
    if h < crop_size or w < crop_size:
        return [hr]
    crops = []
    for _ in range(n_crops):
        top = rng.randint(0, h - crop_size)
        left = rng.randint(0, w - crop_size)
        crops.append(hr[top : top + crop_size, left : left + crop_size])
    return crops


def evaluate(
    checkpoint: str,
    device: torch.device,
    scale: int,
    crop_size: int = 128,
    n_crops: int = 4,
    seed: int = 123,
) -> dict:
    """Evaluate one checkpoint. Crops and their synthetic degradation are
    fully determined by (crop_size, n_crops, seed) -- calling this again
    with the same three values for a different checkpoint evaluates it on
    the exact same LR inputs, which is what makes cross-checkpoint
    comparisons in `compare()` below fair rather than apples-to-oranges.
    """
    model, ckpt_scale = load_model(checkpoint, device)
    if ckpt_scale != scale:
        print(f"warning: checkpoint scale ({ckpt_scale}) != requested scale ({scale}); using checkpoint's scale")
        scale = ckpt_scale

    results: dict[str, list[dict]] = {"matched": [], "bicubic_only": []}

    for test_img in TEST_IMAGES:
        if not test_img.path.exists():
            print(f"skip {test_img.name}: file not found at {test_img.path}")
            continue
        hr_full = load_image(test_img.path)
        crops = make_crops(hr_full, crop_size, n_crops, seed)

        for mode in ("matched", "bicubic_only"):
            for i, hr in enumerate(crops):
                h2, w2 = (hr.shape[0] // scale) * scale, (hr.shape[1] // scale) * scale
                hr = hr[:h2, :w2]

                if mode == "matched":
                    lr = random_degrade(hr, scale, rng=random.Random(seed + i))
                else:
                    lr = bicubic_degrade(hr, scale)

                bicubic_out = bicubic_upscale(lr, scale)
                with torch.no_grad():
                    model_out = enhance_tiled(model, lr, scale, device, tile=max(256, crop_size), overlap=16)

                results[mode].append(
                    {
                        "image": test_img.name,
                        "label": test_img.label,
                        "crop": i,
                        "bicubic_psnr": psnr(bicubic_out, hr),
                        "bicubic_ssim": ssim(bicubic_out, hr),
                        "model_psnr": psnr(model_out, hr),
                        "model_ssim": ssim(model_out, hr),
                    }
                )

    return results


def summarize(results: dict, model_name: str = "Model") -> None:
    for mode, rows in results.items():
        print(f"\n=== Degradation mode: {mode} ===")
        for label in ("held_out", "in_sample"):
            subset = [r for r in rows if r["label"] == label]
            if not subset:
                continue
            n = len(subset)
            bic_psnr = sum(r["bicubic_psnr"] for r in subset) / n
            bic_ssim = sum(r["bicubic_ssim"] for r in subset) / n
            mod_psnr = sum(r["model_psnr"] for r in subset) / n
            mod_ssim = sum(r["model_ssim"] for r in subset) / n
            tag = "HELD-OUT (generalization)" if label == "held_out" else "in-sample (seen during training)"
            print(f"  [{tag}] n={n} crops")
            print(f"    {'Method':<20}{'PSNR':>10}{'SSIM':>10}")
            print(f"    {'Bicubic':<20}{bic_psnr:>10.3f}{bic_ssim:>10.4f}")
            print(f"    {model_name:<20}{mod_psnr:>10.3f}{mod_ssim:>10.4f}")
            delta_psnr = mod_psnr - bic_psnr
            verdict = "BETTER than bicubic" if delta_psnr > 0 else "WORSE than bicubic"
            print(f"    -> {model_name} {verdict} by {delta_psnr:+.3f} dB PSNR")


def compare(
    checkpoints: dict[str, str],
    device: torch.device,
    scale: int,
    crop_size: int = 128,
    n_crops: int = 4,
    seed: int = 123,
) -> None:
    """Evaluate several named checkpoints on identical inputs and print one
    combined Bicubic / <name1> / <name2> / ... table per (mode, label)."""
    all_results = {name: evaluate(path, device, scale, crop_size, n_crops, seed) for name, path in checkpoints.items()}

    modes = next(iter(all_results.values())).keys()
    for mode in modes:
        print(f"\n=== Degradation mode: {mode} ===")
        for label in ("held_out", "in_sample"):
            print(f"  [{'HELD-OUT (generalization)' if label == 'held_out' else 'in-sample'}]")
            header_printed = False
            for name, results in all_results.items():
                subset = [r for r in results[mode] if r["label"] == label]
                if not subset:
                    continue
                n = len(subset)
                if not header_printed:
                    print(f"    {'Method':<20}{'PSNR':>10}{'SSIM':>10}   (n={n} crops)")
                    bic_psnr = sum(r["bicubic_psnr"] for r in subset) / n
                    bic_ssim = sum(r["bicubic_ssim"] for r in subset) / n
                    print(f"    {'Bicubic':<20}{bic_psnr:>10.3f}{bic_ssim:>10.4f}")
                    header_printed = True
                mod_psnr = sum(r["model_psnr"] for r in subset) / n
                mod_ssim = sum(r["model_ssim"] for r in subset) / n
                print(f"    {name:<20}{mod_psnr:>10.3f}{mod_ssim:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints/rrdb_x4.pth", help="Single checkpoint to evaluate")
    ap.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Compare several checkpoints in one table, each as NAME=path/to.pth "
        "(e.g. --compare 'Previous=checkpoints/rrdb_x4_v1_baseline.pth' 'Improved=checkpoints/rrdb_x4_best.pth')",
    )
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--n-crops", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.compare:
        checkpoints = dict(item.split("=", 1) for item in args.compare)
        compare(checkpoints, device, args.scale, args.crop_size, args.n_crops, args.seed)
        return

    results = evaluate(args.checkpoint, device, args.scale, args.crop_size, args.n_crops, args.seed)
    summarize(results)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        print(f"\nRaw per-crop results written to {args.out_json}")


if __name__ == "__main__":
    main()

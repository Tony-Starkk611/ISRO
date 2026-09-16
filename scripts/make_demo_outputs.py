"""Assemble a small set of committed before/after demo outputs (2D image
enhancement side-by-side, 3D terrain preview) into outputs/demo/, using the
trained checkpoints. Run after train_2d.py / train_3d.py have produced
checkpoints/rrdb_x4.pth and checkpoints/dem_sr_x4.pth.

Usage:
    python scripts/make_demo_outputs.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from satellite_enhance.infer_2d import enhance_tiled, load_model as load_2d_model
from satellite_enhance.infer_3d import enhance_dem, load_model as load_3d_model
from satellite_enhance.utils.degradation import random_degrade
from satellite_enhance.utils.image_io import load_dem, load_image
from satellite_enhance.utils.terrain3d import render_3d_preview


def make_2d_demo(checkpoint: str, image_path: Path, out_dir: Path):
    device = torch.device("cpu")
    model, scale = load_2d_model(checkpoint, device)

    hr_like = load_image(image_path)
    # Degrade with the same blur/downsample/noise/JPEG pipeline used during
    # training (utils/degradation.py) so the model sees in-distribution
    # input -- feeding it a plain bicubic downsample it never saw the
    # inverse of during training would understate/misrepresent quality.
    h, w = hr_like.shape[:2]
    h2, w2 = (h // scale) * scale, (w // scale) * scale
    hr_like = hr_like[:h2, :w2]
    lr_img = random_degrade(hr_like, scale, rng=random.Random(0))

    enhanced = enhance_tiled(model, lr_img, scale, device, tile=256, overlap=16)

    bicubic_upsampled = np.asarray(
        Image.fromarray((lr_img * 255).astype(np.uint8)).resize((w2, h2), Image.BICUBIC)
    ).astype(np.float32) / 255.0

    side_by_side = np.concatenate([bicubic_upsampled, enhanced], axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(side_by_side, 0, 1) * 255).astype(np.uint8)).save(
        out_dir / f"{image_path.stem}_bicubic_vs_enhanced_x{scale}.png"
    )
    Image.fromarray((np.clip(lr_img, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{image_path.stem}_input_lr.png")
    Image.fromarray((np.clip(enhanced, 0, 1) * 255).astype(np.uint8)).save(
        out_dir / f"{image_path.stem}_enhanced_x{scale}.png"
    )
    print(f"2D demo written for {image_path.name} (bicubic vs. model, x{scale})")


def make_3d_demo(checkpoint: str, dem_path: Path, z_min: float, z_max: float, out_dir: Path):
    device = torch.device("cpu")
    model, scale = load_3d_model(checkpoint, device)

    raw = load_dem(dem_path)
    dem_m = z_min + (raw / 65535.0) * (z_max - z_min)
    enhanced = enhance_dem(model, dem_m, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    render_3d_preview(dem_m, out_dir / f"{dem_path.stem}_before_3d.png", title="Before (low-res DEM)")
    render_3d_preview(enhanced, out_dir / f"{dem_path.stem}_after_3d.png", title=f"After (x{scale} enhanced)")
    print(f"3D demo written for {dem_path.name} (before/after, x{scale})")


def main():
    demo_dir = Path("outputs/demo")

    ckpt_2d = Path("checkpoints/rrdb_x4.pth")
    if ckpt_2d.exists():
        sample_images = sorted(Path("data/sample").glob("*.png"))[:2]
        for img_path in sample_images:
            make_2d_demo(str(ckpt_2d), img_path, demo_dir)
    else:
        print(f"skip 2D demo: {ckpt_2d} not found (run train_2d.py first)")

    ckpt_3d = Path("checkpoints/dem_sr_x4.pth")
    if ckpt_3d.exists():
        dem_files = sorted(Path("data/dem").glob("terrain_*.png"))[:2]
        for dem_path in dem_files:
            # filename pattern: terrain_00_z0-300m.png
            zrange = dem_path.stem.split("_z")[-1].replace("m", "")
            z_min, z_max = (float(v) for v in zrange.split("-"))
            make_3d_demo(str(ckpt_3d), dem_path, z_min, z_max, demo_dir)
    else:
        print(f"skip 3D demo: {ckpt_3d} not found (run train_3d.py first)")


if __name__ == "__main__":
    main()

"""Run the trained DEM super-resolution model on a low-resolution elevation
tile and reconstruct a 3D terrain preview (optionally textured with an
enhanced 2D optical image).

Example:
    python -m satellite_enhance.infer_3d --checkpoint checkpoints/dem_sr_x4.pth \
        --input-dem data/dem/lowres_tile.png --z-min 0 --z-max 800 \
        --texture outputs/scene_x4.png --output-prefix outputs/terrain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from satellite_enhance.models.dem_srnet import DEMSRNet
from satellite_enhance.utils.image_io import denormalize_dem, load_dem, load_image, normalize_dem, save_dem_png16
from satellite_enhance.utils.terrain3d import dem_to_mesh, render_3d_preview, save_obj


def parse_args():
    p = argparse.ArgumentParser(description="3D DEM super-resolution + terrain reconstruction")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input-dem", type=str, required=True, help="Low-res elevation raster (16-bit PNG or TIFF)")
    p.add_argument("--z-min", type=float, default=None, help="Elevation (m) mapped from pixel value 0, if 16-bit PNG")
    p.add_argument("--z-max", type=float, default=None, help="Elevation (m) mapped from pixel value 65535")
    p.add_argument("--texture", type=str, default=None, help="Optional co-registered optical image to drape as texture")
    p.add_argument("--pixel-size-m", type=float, default=30.0)
    p.add_argument("--output-prefix", type=str, default="outputs/terrain")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DEMSRNet, int]:
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state.get("args", {})
    scale = cfg.get("scale", 4)
    model = DEMSRNet(channels=cfg.get("channels", 48), num_blocks=cfg.get("num_blocks", 8), scale=scale)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, scale


@torch.no_grad()
def enhance_dem(model: DEMSRNet, dem: np.ndarray, device) -> np.ndarray:
    dem_norm, mean, scale = normalize_dem(dem)
    x = torch.from_numpy(dem_norm[None, None]).float().to(device)
    pred = model(x).cpu().numpy()[0, 0]
    return denormalize_dem(pred, mean, scale)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, scale_factor = load_model(args.checkpoint, device)

    raw = load_dem(args.input_dem)
    if args.z_min is not None and args.z_max is not None and raw.max() > 1.5:
        # input was stored as a scaled 16-bit PNG -- recover physical elevation
        dem_m = args.z_min + (raw / 65535.0) * (args.z_max - args.z_min)
    else:
        dem_m = raw

    enhanced = enhance_dem(model, dem_m, device)
    print(f"DEM enhanced: {dem_m.shape} -> {enhanced.shape} (x{scale_factor})")

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z_min, z_max = float(enhanced.min()), float(enhanced.max())
    save_dem_png16(enhanced, out_prefix.with_suffix(".dem16.png"), z_min, z_max)

    texture = None
    if args.texture:
        texture = load_image(args.texture)

    vertices, faces = dem_to_mesh(enhanced, pixel_size_m=args.pixel_size_m / scale_factor)
    save_obj(out_prefix.with_suffix(".obj"), vertices, faces, texture=texture)

    render_3d_preview(enhanced, out_prefix.with_suffix(".png"), texture=texture)
    print(f"Wrote: {out_prefix}.obj, {out_prefix}.dem16.png, {out_prefix}.png")


if __name__ == "__main__":
    main()

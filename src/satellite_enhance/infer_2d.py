"""Run the trained 2D super-resolution model on an input image (or a whole
directory of images), tiling large inputs to bound memory use.

Example:
    python -m satellite_enhance.infer_2d --checkpoint checkpoints/rrdb_x4.pth \
        --input data/sample/scene.png --output outputs/scene_x4.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from satellite_enhance.models.rrdbnet import RRDBNet
from satellite_enhance.utils.image_io import load_image, save_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="2D satellite image super-resolution inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Image file or directory")
    p.add_argument("--output", type=str, required=True, help="Output file or directory")
    p.add_argument("--tile-size", type=int, default=256, help="0 disables tiling")
    p.add_argument("--tile-overlap", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[RRDBNet, int]:
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state.get("args", {})
    scale = cfg.get("scale", 4)
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        channels=cfg.get("channels", 64),
        num_blocks=cfg.get("num_blocks", 16),
        scale=scale,
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, scale


def _feather_window(size: int, ramp: int) -> np.ndarray:
    """1D weight ramping linearly from ~0 to 1 over `ramp` pixels at each
    end and staying at 1 in the middle. Used to blend overlapping tiles
    smoothly instead of averaging them with equal weight everywhere --
    flat averaging leaves a visible seam at the tile boundary because each
    tile's own edge pixels are reconstructed with less spatial context
    than its interior pixels, so neighboring tiles disagree slightly right
    at the seam. Weighting the overlap zone down to ~0 at each tile's own
    edge (and normalizing by the accumulated weight afterwards) makes the
    output dominated by each tile's higher-context interior instead.
    """
    ramp = max(1, min(ramp, size // 2))
    w = np.ones(size, dtype=np.float32)
    taper = (np.arange(ramp, dtype=np.float32) + 1) / (ramp + 1)
    w[:ramp] = taper
    w[-ramp:] = taper[::-1]
    return w


@torch.no_grad()
def enhance_tiled(model: RRDBNet, img: np.ndarray, scale: int, device, tile: int = 256, overlap: int = 16) -> np.ndarray:
    h, w, c = img.shape
    if tile <= 0 or (h <= tile and w <= tile):
        x = torch.from_numpy(img.transpose(2, 0, 1)[None]).float().to(device)
        out = model(x).clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
        return out

    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

    stride = tile - overlap
    feather_px = max(1, overlap * scale // 2)

    for y in range(0, h, stride):
        for x0 in range(0, w, stride):
            y_end = min(y + tile, h)
            x_end = min(x0 + tile, w)
            y_start = max(0, y_end - tile)
            x_start = max(0, x_end - tile)

            patch = img[y_start:y_end, x_start:x_end]
            t = torch.from_numpy(patch.transpose(2, 0, 1)[None]).float().to(device)
            pred = model(t).clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)

            oy, ox = y_start * scale, x_start * scale
            oh, ow = pred.shape[:2]

            wy = _feather_window(oh, feather_px)
            wx = _feather_window(ow, feather_px)
            tile_weight = (wy[:, None] * wx[None, :])[..., None]

            output[oy : oy + oh, ox : ox + ow] += pred * tile_weight
            weight[oy : oy + oh, ox : ox + ow] += tile_weight

            if x_end >= w:
                break
        if y_end >= h:
            break

    weight = np.maximum(weight, 1e-6)
    return output / weight


def process_one(model, scale, device, in_path: Path, out_path: Path, tile: int, overlap: int):
    img = load_image(in_path)
    enhanced = enhance_tiled(model, img, scale, device, tile=tile, overlap=overlap)
    save_image(enhanced, out_path)
    print(f"{in_path} -> {out_path}  ({img.shape[1]}x{img.shape[0]} -> {enhanced.shape[1]}x{enhanced.shape[0]})")


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, scale = load_model(args.checkpoint, device)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        for f in sorted(in_path.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                process_one(model, scale, device, f, out_path / f.name, args.tile_size, args.tile_overlap)
    else:
        process_one(model, scale, device, in_path, out_path, args.tile_size, args.tile_overlap)


if __name__ == "__main__":
    main()

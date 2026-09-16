"""Fetch a small set of real satellite/aerial imagery to use as demo and
smoke-test data for the 2D super-resolution pipeline, and convert it to
standard 8-bit RGB PNGs under data/sample/.

Sources are real Landsat/NAIP-derived test fixtures from well-known
open-source geospatial Python projects (rasterio, rio-tiler, torchgeo),
served with no authentication required via raw.githubusercontent.com. They
are small enough to be practical demo/smoke-test assets while still being
genuine satellite/aerial pixel data (not synthetic placeholders).

For real model training at scale, point --data-dir in train_2d.py at a much
larger corpus instead -- see README.md "Datasets" for options (SpaceNet,
Copernicus Sentinel-2, WorldStrat, AID / UC-Merced / NWPU-RESISC45, ISRO
Bhuvan, USGS EarthExplorer) which require free registration/API keys that
this sandboxed environment does not have.

Usage:
    python scripts/download_sample_data.py --out-dir data/sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# (filename, url, description) -- real Landsat/NAIP test fixtures used across
# the open-source geospatial Python ecosystem.
SOURCE_RASTERS = [
    (
        "landsat_rgb_1.tif",
        "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif",
        "Landsat 7 scene (rasterio test fixture)",
    ),
    (
        "landsat_rgb_2.tif",
        "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB2.byte.tif",
        "Landsat 7 scene, alternate band order (rasterio test fixture)",
    ),
    (
        "sentinel_cog.tif",
        "https://raw.githubusercontent.com/cogeotiff/rio-tiler/main/tests/fixtures/cog.tif",
        "Multi-band 16-bit cloud-optimized GeoTIFF (rio-tiler test fixture)",
    ),
    (
        "naip_aerial.tif",
        "https://raw.githubusercontent.com/microsoft/torchgeo/main/tests/data/naip/m_3807511_ne_18_060_20181104.tif",
        "NAIP aerial imagery tile (torchgeo test fixture)",
    ),
]


def raster_to_rgb_png(src_path: Path, dst_path: Path) -> bool:
    """Convert an arbitrary-depth GeoTIFF to an 8-bit RGB PNG using the first
    three bands, percentile-stretched for visibility."""
    img = Image.open(src_path)
    arr = np.asarray(img)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        return False

    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (arr * 255).astype(np.uint8)

    if rgb.shape[0] < 32 or rgb.shape[1] < 32:
        return False

    Image.fromarray(rgb).save(dst_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/sample")
    ap.add_argument("--raw-dir", type=str, default="data/raw")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for name, url, desc in SOURCE_RASTERS:
        raw_path = raw_dir / name
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "satellite-enhance-demo/0.1"})
            resp.raise_for_status()
            raw_path.write_bytes(resp.content)

            png_path = out_dir / (raw_path.stem + ".png")
            if raster_to_rgb_png(raw_path, png_path):
                print(f"OK   {name:20s} -> {png_path.name}   ({desc})")
                ok += 1
            else:
                print(f"skip {name}: converted image too small/unsuitable")
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: {exc}")

    if ok == 0:
        print(
            "No images could be downloaded (offline environment?). "
            "Run scripts/make_synthetic_dataset.py instead, or supply your own "
            "images under data/sample/."
        )
    else:
        print(f"\n{ok} real satellite/aerial images ready in {out_dir}/")


if __name__ == "__main__":
    main()

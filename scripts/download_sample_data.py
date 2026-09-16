"""Fetch a small set of real satellite/aerial imagery to use as demo and
smoke-test data for the 2D super-resolution pipeline, and convert it to
standard 8-bit RGB PNGs under data/sample/.

Sources are real Landsat test/example rasters from well-known open-source
geospatial Python projects, served with no authentication required via
raw.githubusercontent.com:

- rasterio's classic "RGB.byte.tif" test fixture: a true-color Landsat 7
  scene over the Rocky Mountains.
- leafmap/opengeos's "landsat7.tif" example raster: a larger false-color
  (NIR-Red-Green) Landsat 7 scene over the San Francisco Bay Area.

Both are genuine satellite pixel data, not synthetic placeholders -- which
matters here because several *other* well-known test-fixture repos turned
out to be unusable for this purpose on inspection: torchgeo's bundled
NAIP/UCMerced/RESISC45 "test data" are randomly generated noise images (a
common, legitimate practice for keeping ML test suites small, but not real
imagery), and rio-tiler's "cog.tif" fixture is a single-band raster with
multiple pyramid/overview levels rather than a 3-band RGB image -- treating
it as RGB by taking the first 3 "bands" silently produces a washed-out
near-grayscale image. Mixing either of those into training data measurably
degrades color fidelity, so they are deliberately not included here.

Note the two sources use different band-to-color mappings (true color vs.
false color) -- both real, but not photometrically consistent with each
other, same as different real-world satellite products often aren't.

For real model training at scale, point --data-dir in train_2d.py at a much
larger, consistent corpus instead -- see README.md "Datasets" for options
(SpaceNet, Copernicus Sentinel-2, WorldStrat, AID / UC-Merced / NWPU-
RESISC45, ISRO Bhuvan, USGS EarthExplorer) which require free registration/
API keys that this sandboxed environment does not have.

Usage:
    python scripts/download_sample_data.py --out-dir data/sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# (filename, url, description) -- real Landsat rasters, no auth required.
SOURCE_RASTERS = [
    (
        "landsat_rockies_truecolor.tif",
        "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif",
        "Landsat 7 true-color scene, Rocky Mountains (rasterio test fixture)",
    ),
    (
        "landsat_sfbay_falsecolor.tif",
        "https://raw.githubusercontent.com/opengeos/data/main/raster/landsat7.tif",
        "Landsat 7 false-color (NIR-Red-Green) scene, San Francisco Bay Area (leafmap/opengeos example data)",
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
                print(f"OK   {name:32s} -> {png_path.name}   ({desc})")
                ok += 1
            else:
                print(f"skip {name}: converted image too small/unsuitable")
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: {exc}")

    if ok == 0:
        print(
            "No images could be downloaded (offline environment?). "
            "Run scripts/make_synthetic_dem.py instead, or supply your own "
            "images under data/sample/."
        )
    else:
        print(f"\n{ok} real satellite images ready in {out_dir}/")


if __name__ == "__main__":
    main()

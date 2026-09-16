"""Generate a small set of synthetic terrain DEM tiles for demoing/testing
the 3D enhancement pipeline without needing a real DEM download (real DEM
sources such as SRTM/CartoSat/Bhuvan typically require registration/API
keys not available in this environment -- see README.md "Datasets").

Each tile is saved as a 16-bit grayscale PNG (elevation linearly scaled into
[0, 65535] between the tile's own min/max, recorded in the filename), plus a
matplotlib preview.

Usage:
    python scripts/make_synthetic_dem.py --out-dir data/dem --count 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from satellite_enhance.utils.image_io import save_dem_png16  # noqa: E402
from satellite_enhance.utils.synthetic_terrain import generate_terrain  # noqa: E402
from satellite_enhance.utils.terrain3d import render_3d_preview  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/dem")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--size", type=int, default=257, help="tile side length (2**k + 1)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Previews are RGB renders, not elevation rasters -- keep them outside
    # out_dir so DEMDataset's raster glob never mistakes them for DEM tiles.
    preview_dir = out_dir.parent / f"{out_dir.name}_previews"
    preview_dir.mkdir(exist_ok=True)

    for i in range(args.count):
        roughness = 0.35 + 0.35 * (i / max(args.count - 1, 1))
        elev_range = 300.0 + 1200.0 * (i / max(args.count - 1, 1))
        dem = generate_terrain(size=args.size, roughness=roughness, elevation_range_m=elev_range, seed=i)

        z_min, z_max = float(dem.min()), float(dem.max())
        name = f"terrain_{i:02d}_z{z_min:.0f}-{z_max:.0f}m.png"
        save_dem_png16(dem, out_dir / name, z_min, z_max)
        render_3d_preview(dem, preview_dir / name.replace(".png", "_3d.png"), title=f"Synthetic terrain {i}")
        print(f"wrote {name}  (roughness={roughness:.2f}, range={z_max - z_min:.0f}m)")

    print(f"\n{args.count} synthetic DEM tiles ready in {out_dir}/")


if __name__ == "__main__":
    main()

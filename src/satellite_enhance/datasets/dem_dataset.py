"""PyTorch Dataset for 3D DEM (elevation) super-resolution.

Produces (LR, HR) elevation tile pairs either from real DEM rasters placed
under a directory, or synthetically via fractal terrain generation
(utils/synthetic_terrain.py) when no real DEM is available -- letting the
same training script run out-of-the-box for demos and unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from satellite_enhance.utils.degradation import degrade_dem
from satellite_enhance.utils.image_io import load_dem, normalize_dem
from satellite_enhance.utils.synthetic_terrain import generate_terrain

DEM_EXTS = {".tif", ".tiff", ".png"}


class DEMDataset(Dataset):
    def __init__(
        self,
        dem_dir: str | Path | None,
        scale: int = 4,
        patch_size: int = 128,
        synthetic_count: int = 200,
        elevation_range_m: tuple[float, float] = (200.0, 1500.0),
        seed: int | None = 0,
        patches_per_tile: int = 1,
    ):
        self.scale = scale
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)
        self.elevation_range_m = elevation_range_m

        self.paths: list[Path] = []
        if dem_dir is not None:
            dem_dir = Path(dem_dir)
            if dem_dir.exists():
                self.paths = sorted(p for p in dem_dir.rglob("*") if p.suffix.lower() in DEM_EXTS)

        self.synthetic_count = synthetic_count if not self.paths else 0
        self.patches_per_tile = max(1, patches_per_tile)

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_tile if self.paths else self.synthetic_count

    def _get_hr_tile(self, idx: int) -> np.ndarray:
        if self.paths:
            dem = load_dem(self.paths[idx % len(self.paths)])
            h, w = dem.shape
            ps = self.patch_size
            if h < ps or w < ps:
                dem = np.pad(dem, ((0, max(0, ps - h)), (0, max(0, ps - w))), mode="reflect")
                h, w = dem.shape
            top = int(self.rng.integers(0, h - ps + 1))
            left = int(self.rng.integers(0, w - ps + 1))
            return dem[top : top + ps, left : left + ps]

        elev_range = float(self.rng.uniform(*self.elevation_range_m))
        seed = int(self.rng.integers(0, 2**31 - 1))
        size = self.patch_size
        n = 2 ** int(np.ceil(np.log2(size - 1))) + 1
        terrain = generate_terrain(size=n, roughness=0.5, elevation_range_m=elev_range, seed=seed)
        return terrain[:size, :size]

    def __getitem__(self, idx: int):
        hr = self._get_hr_tile(idx)
        h, w = hr.shape
        h2, w2 = (h // self.scale) * self.scale, (w // self.scale) * self.scale
        hr = hr[:h2, :w2]
        lr = degrade_dem(hr, self.scale, rng=None)

        hr_norm, mean, scale = normalize_dem(hr)
        lr_norm = (lr - mean) / scale

        lr_t = torch.from_numpy(lr_norm[None]).float()
        hr_t = torch.from_numpy(hr_norm[None]).float()
        return lr_t, hr_t

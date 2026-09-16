"""Image and raster I/O helpers shared across the pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as float32 array in [0, 1], shape (H, W, C)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_image(arr: np.ndarray, path: str | Path) -> None:
    """Save a float32 [0, 1] or uint8 array as an image file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_dem(path: str | Path) -> np.ndarray:
    """Load a single-band elevation raster. Supports plain image formats
    (PNG/TIFF encoded as 16-bit grayscale) directly via Pillow; for real
    GeoTIFF DEM products with georeferencing, install and use `rasterio`
    instead (kept optional/out of core deps to avoid a heavy GDAL dependency
    for users who only want the synthetic/demo pipeline).
    """
    img = Image.open(path)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr


def save_dem_png16(dem: np.ndarray, path: str | Path, z_min: float, z_max: float) -> None:
    """Save a float elevation array as a 16-bit PNG, linearly scaled to
    [z_min, z_max] -> [0, 65535]. Store z_min/z_max alongside (e.g. in a
    sidecar filename or config) to recover absolute elevation later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    span = max(z_max - z_min, 1e-6)
    scaled = np.clip((dem - z_min) / span, 0, 1)
    arr16 = (scaled * 65535).round().astype(np.uint16)
    Image.fromarray(arr16).save(path)


def normalize_dem(dem: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize a DEM tile to roughly [-1, 1] for network stability.
    Returns (normalized, mean, scale) so it can be denormalized later.
    """
    mean = float(dem.mean())
    scale = float(dem.std() + 1e-6)
    return (dem - mean) / scale, mean, scale


def denormalize_dem(dem_norm: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return dem_norm * scale + mean

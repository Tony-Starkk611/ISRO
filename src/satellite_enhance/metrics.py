"""Evaluation metrics for 2D image and 3D DEM spatial enhancement quality."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range**2) / mse)


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    channel_axis = -1 if pred.ndim == 3 else None
    return float(ssim_fn(target, pred, data_range=data_range, channel_axis=channel_axis))


def dem_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root-mean-square elevation error in the same units as the DEM (meters)."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def dem_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def slope_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error of terrain slope (gradient magnitude), which
    reflects how well ridgelines/drainage structure are preserved -- a more
    'spatial-detail-aware' metric than raw elevation RMSE."""
    def slope(dem: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(dem)
        return np.sqrt(gx**2 + gy**2)

    return float(np.mean(np.abs(slope(pred) - slope(target))))

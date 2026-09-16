"""Classical stereo/photoclinometry depth recovery for satellite image pairs.

When no DEM product is available at all (only optical imagery), a coarse
elevation estimate can still be produced from:

1. A stereo pair (e.g. along-track/across-track satellite stereo, or two
   overlapping passes) via semi-global block matching -> disparity -> depth.
2. A single image via shape-from-shading (photoclinometry), which is far
   coarser but works with just one panchromatic frame and a sun-elevation
   angle -- a technique actually used operationally for planetary/lunar DEMs
   (e.g. from Chandrayaan/LRO imagery) when stereo coverage is unavailable.

The resulting coarse elevation map is meant to be fed into DEMSRNet
(dem_srnet.py) as its low-resolution input, unifying the "no DEM data"
case with the "have a coarse DEM" case behind the same 3D-enhancement
pipeline.
"""

from __future__ import annotations

import numpy as np


def stereo_disparity_to_depth(
    left: np.ndarray,
    right: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    num_disparities: int = 64,
    block_size: int = 7,
) -> np.ndarray:
    """Compute a metric depth map from a rectified stereo pair using SGBM.

    Args:
        left, right: grayscale uint8 rectified stereo images, same shape.
        focal_length_px: camera focal length in pixels.
        baseline_m: distance between the two camera/satellite positions in meters.
        num_disparities: must be divisible by 16.
        block_size: odd matching window size.

    Returns:
        depth map in meters, shape == left.shape, float32. Pixels with no
        valid disparity are set to NaN.
    """
    import cv2

    if left.shape != right.shape:
        raise ValueError("left and right images must have the same shape")
    num_disparities = max(16, (num_disparities // 16) * 16)

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
    )
    disparity = matcher.compute(left, right).astype(np.float32) / 16.0
    depth = np.full_like(disparity, np.nan, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal_length_px * baseline_m) / disparity[valid]
    return depth


def shape_from_shading(
    image: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    albedo: float = 0.3,
    pixel_size_m: float = 30.0,
    iterations: int = 400,
    step: float = 0.15,
) -> np.ndarray:
    """Recover a coarse relative elevation surface from single-image shading.

    Implements a simple gradient-descent Shape-from-Shading solver under a
    Lambertian reflectance model with a directional light source. This is a
    classical photoclinometry technique used to derive elevation from single
    panchromatic frames when stereo coverage is unavailable.

    Args:
        image: single-band grayscale image, float in [0, 1] or uint8.
        sun_azimuth_deg: sun azimuth angle (degrees from north, clockwise).
        sun_elevation_deg: sun elevation angle above horizon (degrees).
        albedo: assumed surface reflectance (0-1).
        pixel_size_m: ground sample distance of one pixel, for gradient scaling.
        iterations: number of gradient-descent refinement steps.
        step: gradient-descent step size.

    Returns:
        relative elevation surface (float32, same H,W as image), in meters,
        arbitrary vertical datum (only relative relief is physically meaningful).
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    az = np.deg2rad(sun_azimuth_deg)
    el = np.deg2rad(sun_elevation_deg)
    light = np.array(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
        dtype=np.float32,
    )

    h, w = img.shape
    z = np.zeros((h, w), dtype=np.float32)

    for _ in range(iterations):
        gy, gx = np.gradient(z, pixel_size_m)
        normal = np.stack([-gx, -gy, np.ones_like(z)], axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
        shaded = albedo * np.clip(normal @ light, 0, 1)
        residual = img - shaded
        z += step * residual
        z -= z.mean()

    return z

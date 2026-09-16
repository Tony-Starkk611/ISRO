"""Synthetic terrain generation for training/testing the DEM enhancement
pipeline when no real elevation data is available offline.

Uses the midpoint-displacement ("diamond-square") fractal algorithm, which
produces terrain with realistic 1/f-noise statistics -- the same power-law
roughness spectrum observed in real topography -- making it a reasonable
stand-in for training and unit-testing the super-resolution network's
ability to hallucinate plausible fine-scale relief from a coarse input.
"""

from __future__ import annotations

import random

import numpy as np


def diamond_square(size_pow2_plus_1: int, roughness: float = 0.55, seed: int | None = None) -> np.ndarray:
    """Generate a fractal heightmap of shape (n, n) where n = 2**k + 1.

    Args:
        size_pow2_plus_1: output side length, must equal 2**k + 1 for some k.
        roughness: in (0, 1); higher values produce rougher/noisier terrain.
        seed: RNG seed for reproducibility.
    """
    n = size_pow2_plus_1
    if (n - 1) & (n - 2) != 0 and n != 2:
        # (n-1) must be a power of two
        k = int(np.ceil(np.log2(n - 1)))
        n = 2**k + 1

    rng = random.Random(seed)
    grid = np.zeros((n, n), dtype=np.float32)
    grid[0, 0] = rng.uniform(-1, 1)
    grid[0, -1] = rng.uniform(-1, 1)
    grid[-1, 0] = rng.uniform(-1, 1)
    grid[-1, -1] = rng.uniform(-1, 1)

    step = n - 1
    scale = 1.0
    while step > 1:
        half = step // 2

        # diamond step
        for y in range(half, n - 1, step):
            for x in range(half, n - 1, step):
                avg = (
                    grid[y - half, x - half]
                    + grid[y - half, x + half]
                    + grid[y + half, x - half]
                    + grid[y + half, x + half]
                ) / 4.0
                grid[y, x] = avg + rng.uniform(-1, 1) * scale

        # square step
        for y in range(0, n, half):
            for x in range((y + half) % step, n, step):
                total, count = 0.0, 0
                if y - half >= 0:
                    total += grid[y - half, x]
                    count += 1
                if y + half < n:
                    total += grid[y + half, x]
                    count += 1
                if x - half >= 0:
                    total += grid[y, x - half]
                    count += 1
                if x + half < n:
                    total += grid[y, x + half]
                    count += 1
                grid[y, x] = total / count + rng.uniform(-1, 1) * scale

        step = half
        scale *= 2 ** (-roughness)

    grid -= grid.min()
    grid /= (grid.max() + 1e-8)
    return grid


def generate_terrain(size: int = 257, roughness: float = 0.55, elevation_range_m: float = 800.0, seed: int | None = None) -> np.ndarray:
    """Convenience wrapper returning a terrain patch scaled to meters, cropped
    to exactly `size` x `size` (diamond_square rounds up internally)."""
    heightmap = diamond_square(size, roughness=roughness, seed=seed)
    heightmap = heightmap[:size, :size] * elevation_range_m
    return heightmap.astype(np.float32)

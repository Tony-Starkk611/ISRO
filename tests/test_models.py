"""Unit tests for model architectures and core numerical utilities."""

import numpy as np
import torch

from satellite_enhance.models.dem_srnet import DEMSRNet
from satellite_enhance.models.discriminator import VGGStyleDiscriminator
from satellite_enhance.models.rrdbnet import RRDBNet
from satellite_enhance.models.stereo_depth import shape_from_shading, stereo_disparity_to_depth
from satellite_enhance.utils.degradation import degrade_dem, make_lr_hr_pair
from satellite_enhance.utils.synthetic_terrain import generate_terrain
from satellite_enhance.utils.terrain3d import dem_to_mesh


def test_rrdbnet_shapes():
    for scale in (1, 2, 4, 8):
        model = RRDBNet(channels=16, num_blocks=2, scale=scale)
        x = torch.randn(1, 3, 16, 16)
        y = model(x)
        assert y.shape == (1, 3, 16 * scale, 16 * scale)


def test_dem_srnet_shapes():
    for scale in (1, 2, 4):
        model = DEMSRNet(channels=8, num_blocks=2, scale=scale)
        x = torch.randn(2, 1, 20, 20)
        y = model(x)
        assert y.shape == (2, 1, 20 * scale, 20 * scale)


def test_discriminator_output():
    disc = VGGStyleDiscriminator(base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    out = disc(x)
    assert out.shape == (2, 1)


def test_degradation_pair_shapes():
    hr = np.random.rand(64, 64, 3).astype(np.float32)
    lr, hr_out = make_lr_hr_pair(hr, scale=4)
    assert lr.shape == (16, 16, 3)
    assert hr_out.shape == (64, 64, 3)


def test_degrade_dem_shape():
    hr_dem = np.random.rand(64, 64).astype(np.float32) * 100
    lr_dem = degrade_dem(hr_dem, scale=4)
    assert lr_dem.shape == (16, 16)


def test_synthetic_terrain_range():
    terrain = generate_terrain(size=33, elevation_range_m=500.0, seed=42)
    assert terrain.shape == (33, 33)
    assert terrain.min() >= 0.0
    assert terrain.max() <= 500.0 + 1e-3


def test_dem_to_mesh_counts():
    dem = np.zeros((5, 4), dtype=np.float32)
    vertices, faces = dem_to_mesh(dem)
    assert vertices.shape == (20, 3)
    assert faces.shape == (2 * 4 * 3, 3)


def test_shape_from_shading_runs():
    img = np.random.rand(32, 32).astype(np.float32)
    z = shape_from_shading(img, sun_azimuth_deg=135, sun_elevation_deg=45, iterations=5)
    assert z.shape == (32, 32)
    assert np.isfinite(z).all()


def test_stereo_disparity_to_depth_runs():
    left = (np.random.rand(48, 64) * 255).astype(np.uint8)
    shift = 3
    right = np.roll(left, shift, axis=1)
    depth = stereo_disparity_to_depth(left, right, focal_length_px=500.0, baseline_m=1.0, num_disparities=16, block_size=5)
    assert depth.shape == left.shape

"""End-to-end smoke tests for the training-data pipeline (dataset + losses)."""

import torch

from satellite_enhance.datasets.dem_dataset import DEMDataset
from satellite_enhance.datasets.sr_dataset import SatelliteSRDataset
from satellite_enhance.losses.losses import SRCompositeLoss
from satellite_enhance.models.dem_srnet import DEMSRNet
from satellite_enhance.models.rrdbnet import RRDBNet


def test_sr_dataset_and_training_step(tmp_path):
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(2):
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.png")

    dataset = SatelliteSRDataset(img_dir, scale=4, hr_patch_size=32, patches_per_image=3)
    assert len(dataset) == 6
    lr, hr = dataset[0]
    assert lr.shape == (3, 8, 8)
    assert hr.shape == (3, 32, 32)

    model = RRDBNet(channels=8, num_blocks=1, scale=4)
    criterion = SRCompositeLoss(in_channels=3)
    pred = model(lr.unsqueeze(0))
    loss, parts = criterion(pred, hr.unsqueeze(0))
    assert torch.isfinite(loss)
    assert "pixel" in parts


def test_dem_dataset_synthetic_and_training_step():
    dataset = DEMDataset(None, scale=4, patch_size=32, synthetic_count=5)
    assert len(dataset) == 5
    lr, hr = dataset[0]
    assert lr.shape == (1, 8, 8)
    assert hr.shape == (1, 32, 32)

    model = DEMSRNet(channels=8, num_blocks=1, scale=4)
    criterion = SRCompositeLoss(in_channels=1, weight_perceptual=0.0)
    pred = model(lr.unsqueeze(0))
    loss, parts = criterion(pred, hr.unsqueeze(0))
    assert torch.isfinite(loss)

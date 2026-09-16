"""PyTorch Dataset for 2D satellite image super-resolution.

Reads arbitrary high-resolution image tiles from a directory and produces
(LR, HR) training pairs on the fly using the synthetic degradation pipeline
in utils/degradation.py. This design lets the same dataset class train on
any HR-only satellite image collection (drone orthomosaics, Sentinel-2 RGB
composites, aerial imagery, etc.) without needing pre-paired LR/HR data.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from satellite_enhance.utils.degradation import make_lr_hr_pair
from satellite_enhance.utils.image_io import load_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SatelliteSRDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        scale: int = 4,
        hr_patch_size: int = 128,
        augment: bool = True,
        seed: int | None = None,
        patches_per_image: int = 1,
    ):
        """
        Args:
            patches_per_image: virtual multiplier on dataset length so one
                training "epoch" draws this many random crops per source
                image instead of exactly one -- source satellite scenes are
                large relative to a training patch, so a handful of raw
                files still yields many distinct training samples per epoch.
        """
        self.image_dir = Path(image_dir)
        self.paths = sorted(p for p in self.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {image_dir}")
        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.augment = augment
        self.rng = random.Random(seed)
        self.patches_per_image = max(1, patches_per_image)

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_image

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ps = self.hr_patch_size
        if h < ps or w < ps:
            pad_h, pad_w = max(0, ps - h), max(0, ps - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w = img.shape[:2]
        top = self.rng.randint(0, h - ps)
        left = self.rng.randint(0, w - ps)
        return img[top : top + ps, left : left + ps]

    def _augment(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            lr, hr = lr[:, ::-1].copy(), hr[:, ::-1].copy()
        if self.rng.random() < 0.5:
            lr, hr = lr[::-1, :].copy(), hr[::-1, :].copy()
        if self.rng.random() < 0.5:
            lr, hr = lr.transpose(1, 0, 2).copy(), hr.transpose(1, 0, 2).copy()
        return lr, hr

    def __getitem__(self, idx: int):
        img = load_image(self.paths[idx % len(self.paths)])
        hr_patch = self._random_crop(img)
        lr, hr = make_lr_hr_pair(hr_patch, self.scale, self.rng)
        if self.augment:
            lr, hr = self._augment(lr, hr)
        lr_t = torch.from_numpy(lr.transpose(2, 0, 1).copy()).float()
        hr_t = torch.from_numpy(hr.transpose(2, 0, 1).copy()).float()
        return lr_t, hr_t

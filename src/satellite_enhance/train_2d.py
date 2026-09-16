"""Train the 2D satellite image super-resolution generator (RRDBNet).

Example:
    python -m satellite_enhance.train_2d --data-dir data/sample --epochs 5 \
        --scale 4 --patch-size 96 --batch-size 4 --checkpoint checkpoints/rrdb_x4.pth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satellite_enhance.datasets.sr_dataset import SatelliteSRDataset
from satellite_enhance.losses.losses import SRCompositeLoss
from satellite_enhance.models.rrdbnet import RRDBNet


def parse_args():
    p = argparse.ArgumentParser(description="Train 2D satellite image super-resolution model")
    p.add_argument("--data-dir", type=str, required=True, help="Directory of HR training images")
    p.add_argument("--val-dir", type=str, default=None, help="Optional directory of HR validation images")
    p.add_argument("--scale", type=int, default=4, choices=[1, 2, 4, 8])
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--patches-per-image", type=int, default=1, help="Random crops drawn per source image per epoch")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=None, help="Cap steps/epoch (useful for smoke tests)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=16)
    p.add_argument("--checkpoint", type=str, default="checkpoints/rrdb_x4.pth")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def build_model(args) -> RRDBNet:
    return RRDBNet(
        in_channels=3,
        out_channels=3,
        channels=args.channels,
        num_blocks=args.num_blocks,
        scale=args.scale,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = SatelliteSRDataset(
        args.data_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = build_model(args).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Resumed weights from {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = SRCompositeLoss(in_channels=3)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training on {len(dataset)} images, device={device}, scale={args.scale}x")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = {"total": 0.0, "pixel": 0.0, "gradient": 0.0, "perceptual": 0.0}
        n_batches = 0

        for step, (lr_img, hr_img) in enumerate(loader):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            pred = model(lr_img)
            loss, parts = criterion(pred, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1
            global_step += 1

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss={parts['total']:.4f}")

        elapsed = time.time() - epoch_start
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        print(
            f"[epoch {epoch}/{args.epochs}] avg_total={avg['total']:.4f} "
            f"avg_pixel={avg['pixel']:.4f} avg_grad={avg['gradient']:.4f} "
            f"({elapsed:.1f}s, {n_batches} batches)"
        )

        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step,
            },
            ckpt_path,
        )
    print(f"Saved final checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()

"""Train the 2D satellite image super-resolution generator (RRDBNet).

Example:
    python -m satellite_enhance.train_2d --data-dir data/sample --epochs 5 \
        --scale 4 --patch-size 96 --batch-size 4 --checkpoint checkpoints/rrdb_x4.pth

Includes a genuine train/val split (spatial, within each source image --
see datasets/sr_dataset.py) so model selection is based on measured
held-out PSNR/SSIM rather than blindly trusting the last epoch. This
matters most exactly when the training corpus is small: without it there
is no signal for whether a given epoch generalizes or has started
overfitting/degrading on content it wasn't directly trained on.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satellite_enhance.datasets.sr_dataset import SatelliteSRDataset
from satellite_enhance.losses.losses import SRCompositeLoss
from satellite_enhance.metrics import psnr, ssim
from satellite_enhance.models.rrdbnet import RRDBNet


def parse_args():
    p = argparse.ArgumentParser(description="Train 2D satellite image super-resolution model")
    p.add_argument("--data-dir", type=str, required=True, help="Directory of HR training images")
    p.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of each image's width held out for validation")
    p.add_argument("--scale", type=int, default=4, choices=[1, 2, 4, 8])
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--patches-per-image", type=int, default=1, help="Random crops drawn per source image per epoch")
    p.add_argument("--val-patches-per-image", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=None, help="Cap steps/epoch (useful for smoke tests)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=16)
    p.add_argument("--weight-pixel", type=float, default=1.0)
    p.add_argument("--weight-gradient", type=float, default=0.05)
    p.add_argument(
        "--weight-perceptual",
        type=float,
        default=0.0,
        help="Weight for the hand-rolled edge-filter 'perceptual' loss. Default 0 -- empirically this term "
        "pushed the model to inject high-frequency texture into naturally smooth regions (e.g. open water) "
        "it wasn't trained on, hurting both PSNR/SSIM and visual fidelity there. Enable only if you have "
        "verified it helps on your own held-out data.",
    )
    p.add_argument("--checkpoint", type=str, default="checkpoints/rrdb_x4.pth")
    p.add_argument("--best-checkpoint", type=str, default=None, help="Defaults to <checkpoint>.best.pth")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_model(args) -> RRDBNet:
    return RRDBNet(
        in_channels=3,
        out_channels=3,
        channels=args.channels,
        num_blocks=args.num_blocks,
        scale=args.scale,
    )


@torch.no_grad()
def validate(model: RRDBNet, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnrs, ssims = [], []
    for lr_img, hr_img in loader:
        lr_img = lr_img.to(device)
        pred = model(lr_img).clamp(0, 1).cpu().numpy()
        hr_np = hr_img.numpy()
        for p, h in zip(pred, hr_np):
            p = p.transpose(1, 2, 0)
            h = h.transpose(1, 2, 0)
            psnrs.append(psnr(p, h))
            ssims.append(ssim(p, h))
    model.train()
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_dataset = SatelliteSRDataset(
        args.data_dir,
        scale=args.scale,
        hr_patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_loader = None
    if args.val_fraction > 0:
        val_dataset = SatelliteSRDataset(
            args.data_dir,
            scale=args.scale,
            hr_patch_size=args.patch_size,
            patches_per_image=args.val_patches_per_image,
            split="val",
            val_fraction=args.val_fraction,
            augment=False,
            seed=12345,  # fixed regardless of --seed: same val crops across runs/epochs for comparable curves
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Resumed weights from {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = SRCompositeLoss(
        in_channels=3,
        weight_pixel=args.weight_pixel,
        weight_gradient=args.weight_gradient,
        weight_perceptual=args.weight_perceptual,
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training on {len(train_dataset)} patches, device={device}, scale={args.scale}x")
    if val_loader is not None:
        print(f"Validating on {len(val_dataset)} held-out patches (val_fraction={args.val_fraction})")

    best_ckpt_path = Path(args.best_checkpoint or f"{args.checkpoint}.best.pth")
    best_val_psnr = float("-inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = {"total": 0.0, "pixel": 0.0, "gradient": 0.0, "perceptual": 0.0}
        n_batches = 0

        for step, (lr_img, hr_img) in enumerate(train_loader):
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
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={parts['total']:.4f}")

        elapsed = time.time() - epoch_start
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        val_str = ""
        val_psnr = None
        if val_loader is not None:
            val_psnr, val_ssim = validate(model, val_loader, device)
            val_str = f" val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f}"

        print(
            f"[epoch {epoch}/{args.epochs}] avg_total={avg['total']:.4f} "
            f"avg_pixel={avg['pixel']:.4f} avg_grad={avg['gradient']:.4f}"
            f"{val_str} ({elapsed:.1f}s, {n_batches} batches)"
        )

        ckpt_path = Path(args.checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "global_step": global_step,
            "val_psnr": val_psnr,
        }
        torch.save(checkpoint, ckpt_path)

        if val_psnr is not None and val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_ckpt_path)
            print(f"  -> new best val_psnr={val_psnr:.3f}, saved to {best_ckpt_path}")

    print(f"Saved final checkpoint to {args.checkpoint}")
    if val_loader is not None:
        print(f"Best checkpoint (val_psnr={best_val_psnr:.3f}) saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()

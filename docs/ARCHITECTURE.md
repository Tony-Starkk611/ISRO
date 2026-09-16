# Architecture Notes

## 2D pipeline: RRDBNet

`models/rrdbnet.py` implements the generator architecture from ESRGAN/Real-ESRGAN:

```
input (LR image)
  -> Conv3x3                                    (shallow feature extraction)
  -> N x RRDB (Residual-in-Residual Dense Block)  (deep feature trunk)
  -> Conv3x3 + long skip from shallow features
  -> log2(scale) x [Conv3x3 -> PixelShuffle(2) -> LeakyReLU]  (learned upsampling)
  -> Conv3x3 -> LeakyReLU -> Conv3x3             (HR reconstruction)
output (HR image)
```

Each RRDB is three stacked 5-layer dense blocks with residual scaling
(`res_scale=0.2`), which is what lets this architecture train stably at
16-23 blocks deep without batch normalization (BN is deliberately omitted,
following the ESRGAN finding that BN introduces artifacts on GAN-trained
SR networks).

Sub-pixel convolution (`PixelShuffle`) is used for upsampling rather than
transposed convolution to avoid checkerboard artifacts.

## 3D pipeline: DEMSRNet + terrain reconstruction

Two complementary ways to get an elevation input for 3D enhancement:

1. **You already have a coarse DEM** (SRTM, CartoDEM, etc.) → feed it
   directly into `DEMSRNet`.
2. **You only have optical imagery** → recover a coarse elevation estimate
   first via `models/stereo_depth.py` (stereo SGBM for image pairs, or
   shape-from-shading/photoclinometry for a single frame), then feed that
   into `DEMSRNet`.

`DEMSRNet` (`models/dem_srnet.py`) is a lighter residual CNN than RRDBNet
(elevation fields have far less high-frequency content than optical
imagery) with one deliberate structural choice: a **bicubic-upsampled skip
connection** around the whole network, so the network only has to learn a
*residual correction* on top of a smooth baseline. This matters a lot for
training stability on small terrain datasets — a plain regression network
has to relearn "smoothly interpolate" from scratch, whereas the residual
formulation only has to learn where terrain deviates from smooth
interpolation (ridges, incised drainage channels, etc).

Training uses `GradientLoss` in addition to the pixel loss specifically for
this network (`weight_gradient=0.3`, higher than the 2D default of 0.1)
because slope/ridge preservation, not raw per-pixel elevation accuracy, is
the feature that makes a "spatially enhanced" DEM actually look higher
resolution rather than merely smoother.

Once enhanced, `utils/terrain3d.py` converts the elevation grid into an
explicit triangle mesh (two triangles per grid cell) and can drape a
co-registered — ideally *also enhanced* — optical image over it as a UV
texture, producing a genuinely higher-detail 3D reconstruction in both the
horizontal (from the 2D model) and vertical (from the 3D model) sense.

## Loss design (`losses/losses.py`)

- **CharbonnierLoss**: smooth-L1-like pixel loss, more robust to outlier
  pixels (cloud edges, sensor noise) than plain MSE while still producing
  sharper results than L1 alone in the low-residual regime.
- **GradientLoss**: L1 on finite-difference gradients. For optical images
  this fights blur; for DEMs it directly targets slope fidelity.
- **VGGPerceptualLoss**: a fixed (non-trainable) Sobel + Laplacian filter
  bank used as a lightweight, dependency-free stand-in for a real VGG
  perceptual loss — this keeps the repo runnable fully offline. Swap in
  `torchvision.models.vgg19(pretrained=True)` features when pretrained
  weights are reachable.
- **AdversarialLoss**: standard non-saturating GAN loss, provided for an
  optional second fine-tuning stage (see README "Scaling up training").

## Why synthetic degradation instead of requiring paired data

Real paired low/high-resolution satellite imagery of the *same* footprint
is operationally rare (it requires either two sensors imaging the same
scene at different resolutions, or a resampled-down copy of the same
frame, which teaches the network nothing it doesn't already know). Instead,
`utils/degradation.py` implements the standard SR-research recipe: take
real HR imagery, apply blur → downsample → noise → compression to
synthesize a realistic LR counterpart. This is exactly how SRCNN, ESRGAN,
Real-ESRGAN, and BSRGAN are trained, and it generalizes to real
low-resolution satellite input at inference time because the degradation
model is designed to mimic real sensor/compression behavior rather than
being an arbitrary transform.

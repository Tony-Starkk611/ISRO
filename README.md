# Satellite Image Spatial Enhancement (2D + 3D)

A from-scratch, trainable toolkit for **spatial enhancement of satellite
imagery**: sharpening/upscaling optical satellite scenes in 2D (super-resolution)
and increasing the effective resolution of terrain elevation data in 3D
(DEM super-resolution + textured terrain reconstruction).

Built for the ISRO project's goal of enhancing satellite image detail beyond
sensor-native resolution, using deep residual CNNs trained on real
satellite/aerial imagery plus a synthetic-degradation training methodology
standard in the super-resolution literature.

## What's here

| Capability | Module | Technique |
|---|---|---|
| 2D optical image super-resolution | `models/rrdbnet.py` | RRDB generator (ESRGAN/Real-ESRGAN family), Charbonnier + gradient + edge-perceptual loss |
| 2D adversarial fine-tuning (optional) | `models/discriminator.py` | PatchGAN-style discriminator, non-saturating GAN loss |
| 3D DEM (elevation) super-resolution | `models/dem_srnet.py` | Residual CNN with bicubic-skip connection, slope/gradient-consistency loss |
| 3D reconstruction without a DEM | `models/stereo_depth.py` | Stereo SGBM disparity-to-depth, and single-image shape-from-shading (photoclinometry) |
| Terrain visualization/export | `utils/terrain3d.py` | DEM → triangle mesh (`.obj`) with optical texture drape, matplotlib 3D preview render |
| Metrics | `metrics.py` | PSNR, SSIM, DEM RMSE/MAE, slope error |
| Interactive web UI | `webapp/` | FastAPI backend + browser frontend for both models (see "Web UI" below) |

## Why this design

Paired real low-resolution/high-resolution satellite scenes of the exact
same footprint are rare. Following standard SR-research practice (SRCNN,
ESRGAN, Real-ESRGAN, BSRGAN), both pipelines instead train on **synthetic
but realistic degradation** of real high-resolution imagery/DEMs
(`utils/degradation.py`): blur → downsample → sensor noise → mild
compression artifacts for optical images; low-pass + block-average +
noise for elevation. This lets the same code train on *any* HR-only
imagery/DEM collection you point it at, and it transfers well to real
low-resolution input at inference time.

## Project layout

```
src/satellite_enhance/
  models/          RRDBNet (2D), DEMSRNet (3D), discriminator, stereo/SfS depth
  datasets/        on-the-fly LR/HR pair generation for images and DEMs
  losses/          Charbonnier, gradient, edge-perceptual, adversarial losses
  utils/           degradation pipeline, synthetic terrain, image/DEM I/O, 3D mesh export
  train_2d.py      train the optical super-resolution model
  infer_2d.py      run it on new images (with tiling for large scenes)
  train_3d.py      train the DEM super-resolution model
  infer_3d.py      run it + export a textured 3D terrain mesh/preview
  metrics.py       PSNR / SSIM / DEM RMSE / slope error
scripts/
  download_sample_data.py   fetch real Landsat demo imagery (no auth needed)
  make_synthetic_dem.py     generate fractal terrain DEM tiles for 3D demo/testing
tests/             unit + pipeline smoke tests (pytest)
data/sample/       demo optical imagery (populated by download_sample_data.py)
data/dem/          demo DEM tiles (populated by make_synthetic_dem.py)
checkpoints/       trained model weights (.pth)
outputs/           inference results; outputs/demo/ holds committed before/after examples
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.9+, PyTorch (CPU works fine; CUDA auto-used if available),
NumPy, Pillow, OpenCV, scikit-image, SciPy, Matplotlib.

## Quickstart

### 2D: optical image super-resolution

```bash
# Get some real demo imagery (Landsat 7 scenes, no login required)
python scripts/download_sample_data.py --out-dir data/sample

# Train (CPU-friendly settings shown; scale up epochs with a GPU).
# A spatial train/val split (--val-fraction) is built in: each source image is
# split by column into a train region and a held-out val region, and the
# checkpoint with the best *validated* PSNR is saved separately from the
# last epoch -- see "Model selection" below for why this matters.
#
# channels=32/num-blocks=4 (not the usual ESRGAN-scale 64/16-23) is
# deliberate, not a shortcut: at this project's current ~3-image training
# corpus, a much smaller model measurably generalized better to held-out
# imagery than the larger 64/8 config first tried -- a bigger network has
# more capacity to memorize the training images' specific statistics
# instead of learning something that transfers. Revisit this once the
# training corpus is meaningfully larger (see "Scaling up training").
python -m satellite_enhance.train_2d \
  --data-dir data/sample --scale 4 --patch-size 96 --patches-per-image 60 \
  --val-fraction 0.15 --batch-size 4 --epochs 30 --channels 32 --num-blocks 4 \
  --checkpoint checkpoints/rrdb_x4.pth --best-checkpoint checkpoints/rrdb_x4_best.pth

# Promote the validated-best epoch to the canonical path infer_2d.py/the web UI load by default
cp checkpoints/rrdb_x4_best.pth checkpoints/rrdb_x4.pth

# Enhance a new image (tiled automatically if larger than --tile-size)
python -m satellite_enhance.infer_2d \
  --checkpoint checkpoints/rrdb_x4.pth \
  --input data/sample/landsat_rockies_truecolor.png --output outputs/landsat_rockies_x4.png
```

### Why an earlier checkpoint here scored worse than bicubic upscaling

An earlier version of this project just saved whatever the final training
epoch produced and called it done. Measured on real held-out imagery (see
"Evaluation" below), that checkpoint scored **worse than plain bicubic
upscaling** on PSNR/SSIM -- a real, measured regression, not a
hypothetical one. Three contributing causes were found and fixed, in the
order they were actually diagnosed:

1. **No validation signal at all** -- training only ever saved the final
   epoch, so there was no way to know whether that epoch generalized or
   had started overfitting the tiny (2-3 image) training corpus.
   `train_2d.py` now performs a spatial train/val split per source image
   (`--val-fraction`) and saves the best-scoring epoch separately
   (`--best-checkpoint`).
2. **A loss term that rewarded hallucination** -- the composite loss
   included a small weight on a hand-rolled "perceptual" edge-filter term
   that, on genuinely held-out imagery, visibly pushed the model to inject
   noise-like texture into naturally smooth regions (open water) it had
   never trained on. Now defaults to weight `0` (`--weight-perceptual`).
3. **The decisive fix: `RRDBNet` had no floor forcing it back toward
   bicubic-like behavior on unfamiliar input.** Fixing (1) and (2) alone
   *still* didn't beat bicubic on truly held-out imagery -- an honest
   result that was measured, not assumed away. `DEMSRNet` (the 3D model)
   already used a bicubic-upsampled skip connection so it only has to
   learn a residual correction; `RRDBNet` didn't. Adding the same skip
   connection was the change that actually closed the gap: a much smaller
   model (channels=32, num_blocks=4) with the skip connection reached in
   one epoch the validation PSNR the old architecture took 25+ epochs to
   reach, and the resulting checkpoint measurably beats bicubic on
   genuinely held-out imagery (see "Evaluation" below for the numbers).

Always evaluate with `scripts/evaluate.py` (below) before trusting a new
checkpoint -- "it trains without errors" and "it improves the loss curve"
are both necessary and *nowhere near* sufficient evidence that a change
actually helped.

### 3D: DEM / terrain spatial enhancement

```bash
# Generate demo terrain (real DEM sources need registration -- see Datasets below)
python scripts/make_synthetic_dem.py --out-dir data/dem --count 6

# Train
python -m satellite_enhance.train_3d \
  --synthetic-count 300 --scale 4 --patch-size 128 \
  --batch-size 8 --epochs 15 --channels 48 --num-blocks 8 \
  --checkpoint checkpoints/dem_sr_x4.pth
# (pass --dem-dir data/dem to train on real/your own DEM tiles instead)

# Enhance a DEM tile and export a textured 3D mesh + preview render
python -m satellite_enhance.infer_3d \
  --checkpoint checkpoints/dem_sr_x4.pth \
  --input-dem data/dem/terrain_00_z0-300m.png --z-min 0 --z-max 300 \
  --texture outputs/landsat_rockies_x4.png \
  --output-prefix outputs/terrain_00
```

This writes `outputs/terrain_00.obj` (mesh, viewable in Blender/MeshLab/any
3D viewer), `outputs/terrain_00.dem16.png` (16-bit enhanced elevation
raster), and `outputs/terrain_00.png` (quick-look 3D render).

### No DEM at all? Recover elevation from imagery first

`models/stereo_depth.py` provides:
- `stereo_disparity_to_depth(left, right, focal_length_px, baseline_m)` for
  a rectified stereo pair (e.g. satellite along-track stereo).
- `shape_from_shading(image, sun_azimuth_deg, sun_elevation_deg)` for a
  single panchromatic frame (photoclinometry) when no stereo coverage exists.

Feed either result into `infer_3d.py` as the `--input-dem` to go from
"optical image only" to an enhanced 3D terrain reconstruction.

## Web UI

A browser UI wraps both models for interactive use — upload an image or DEM
(or pick a bundled sample), click Enhance, and see/download the result. No
command-line flags to remember.

```bash
pip install -e .   # picks up fastapi/uvicorn from requirements.txt
uvicorn satellite_enhance.webapp.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Two tabs:

- **2D Image** — drag & drop any low-resolution satellite/aerial image (or
  pick a sample scene), click **Enhance ×4**. A draggable before/after
  slider compares the model output against a plain bicubic upscale, with a
  PNG download.
- **3D Terrain** — pick a sample DEM tile (or upload your own 16-bit
  grayscale DEM with a min/max elevation range), click **Enhance ×4**. A
  slider spins synchronized "before" and "after" 3D renders through 8
  viewing angles, with a `.obj` mesh download for opening in Blender/MeshLab.

The web UI caps input size (320px longest side for images, 300×300 for DEM
tiles) to keep inference interactive on CPU; for full-resolution batch
processing use `infer_2d.py` / `infer_3d.py` from the command line instead.
It reuses the same `checkpoints/rrdb_x4.pth` / `checkpoints/dem_sr_x4.pth`
loaded by those scripts, loading each model once at startup.

## Evaluation

`scripts/evaluate.py` measures real PSNR/SSIM against genuine high-resolution
ground truth (the standard SR-literature protocol: degrade a real HR image
to synthesize its LR input, run the model, compare the output back to the
original HR image -- never fabricated numbers). It reports two degradation
modes (`matched`, the same pipeline training uses; `bicubic_only`, the
classic academic benchmark protocol) and keeps `held_out` imagery (never
seen during training) strictly separate from `in_sample` imagery, since an
in-sample score is not evidence of generalization.

```bash
# Evaluate one checkpoint
python scripts/evaluate.py --checkpoint checkpoints/rrdb_x4.pth

# Compare checkpoints side by side (bicubic is always included as a row) --
# only meaningful for checkpoints trained with the same model code version;
# see the warning below.
python scripts/evaluate.py --compare \
  "A=checkpoints/rrdb_x4.pth" "B=checkpoints/some_other_run.pth"
```

> **Checkpoints are tied to the `RRDBNet`/`DEMSRNet` class definition at
> the time they were trained.** If you change the architecture (e.g. add
> or remove a skip connection) and then load an older checkpoint through
> the new code, the shapes will match and it will load without error --
> but the output will be silently wrong, because the old weights were
> never trained for what the new `forward()` does with them. There is no
> version check for this today; when comparing checkpoints across an
> architecture change, keep old checkpoints paired with the code version
> that trained them (e.g. a separate git worktree/commit) rather than
> loading them through `--compare` against a changed model class.

Current measured results (held-out imagery = two real scenes never used in
training; in-sample = crops from the training images themselves, reported
separately since an in-sample score is not evidence of generalization):

```text
Held-out, "matched" degradation (same pipeline training uses):
                    PSNR      SSIM
Bicubic            20.19     0.503
This model         20.27     0.519    (+0.07 dB)

Held-out, "bicubic_only" degradation (classic academic protocol):
                    PSNR      SSIM
Bicubic            21.70     0.661
This model         21.94     0.682    (+0.24 dB)

In-sample, "matched" degradation:
                    PSNR      SSIM
Bicubic            22.54     0.516
This model         22.87     0.539    (+0.33 dB)

In-sample, "bicubic_only" degradation:
                    PSNR      SSIM
Bicubic            24.33     0.671
This model         24.63     0.701    (+0.30 dB)
```

Every row is now non-negative -- a real change from an earlier checkpoint
in this project's history, which scored *worse* than bicubic on every one
of these rows (the exact regression `tests/test_regression.py` now guards
against). Held-out gains are modest and should be read as "not worse than,
and slightly better than, doing nothing" rather than a dramatic quality
leap -- an honest characterization given the small (3-image) training
corpus; see "Known limitations" below.

## Datasets

The bundled 2D training demo uses three **real** rasters pulled from
open-source geospatial projects on GitHub (no auth needed — see
`scripts/download_sample_data.py`): a true-color Landsat 7 scene over the
Rocky Mountains, a larger false-color (NIR-Red-Green) Landsat 7 scene over
the San Francisco Bay Area, and a real single-band USGS aerial/terrain
scene (roads, mountains, lakes, urban development) replicated to 3 channels
for RGB training. A fourth real image (a global true-color satellite
composite) is downloaded separately into `data/heldout/` and is **never**
used for training -- only for measuring genuine generalization in
`scripts/evaluate.py`.

Several other "sample data" repos looked promising but turned out unusable
on inspection: torchgeo's bundled NAIP/UCMerced/RESISC45 test fixtures are
synthetic random noise (a legitimate way to keep an ML test suite small,
but not real imagery), and a rio-tiler COG fixture is a single-band raster
with pyramid overview levels that a naive "take the first 3 channels as
RGB" conversion silently turns into a near-grayscale, wrong-looking image.
Both are documented in `scripts/download_sample_data.py` as a heads-up for
anyone tempted to reach for the same "obvious" sample-data sources.

The 3D demo uses procedurally generated fractal terrain (real DEM portals
below need registration that isn't available in a sandboxed environment).

This is intentionally a *small, two-image* demo corpus — enough to prove
the training/inference pipeline is correct end-to-end, not to produce
production-quality color fidelity. With so little data the model mostly
learns to add plausible high-frequency texture; don't expect polished
results until you point it at hundreds+ of scenes (see below).

For real training at scale, point `--data-dir` / `--dem-dir` at a larger
corpus. Good open options:

**2D optical imagery**
- [SpaceNet](https://spacenet.ai/) (very high-res, multiple cities, AWS Open Data)
- [Copernicus Sentinel-2](https://dataspace.copernicus.eu/) (free registration)
- [WorldStrat](https://github.com/worldstrat/worldstrat) (purpose-built SR dataset, paired Sentinel-2/SPOT)
- AID / UC-Merced / NWPU-RESISC45 aerial scene classification datasets (also usable as HR-only SR training data)
- [ISRO Bhuvan](https://bhuvan.nrsc.gov.in/) (Indian satellite data portal)
- [USGS EarthExplorer](https://earthexplorer.usgs.gov/) (Landsat archive)

**3D / DEM**
- [SRTM](https://srtm.csi.cgiar.org/) (near-global 30-90m DEM)
- [ISRO CartoDEM](https://bhuvan.nrsc.gov.in/) (30m Indian DEM product)
- [Copernicus GLO-30 DEM](https://dataspace.copernicus.eu/)
- USGS 3DEP (US high-resolution LiDAR-derived DEM)

## Scaling up training

The defaults above (`channels=64/48`, `num_blocks=8`) run on CPU for a demo.
For production-quality results:
- Increase `--channels 64 --num-blocks 16-23` (standard ESRGAN depth) and train on a GPU.
- Use `--patches-per-image` / real dataset size in the hundreds-to-thousands of tiles.
- Add adversarial fine-tuning: load the pixel-loss checkpoint into a second
  training stage with `models/discriminator.py` + `losses/losses.py`'s
  `AdversarialLoss` for perceptually sharper (if less pixel-exact) output —
  the standard ESRGAN two-stage recipe. (Not wired into `train_2d.py` by
  default to keep the base training loop simple and stable; the pieces are
  present in `losses/losses.py` and `models/discriminator.py` to extend it.)
- Swap `VGGPerceptualLoss`'s fixed edge-filter bank for real VGG19 ImageNet
  features (`torchvision.models.vgg19(pretrained=True)`) once you have
  network access to pretrained weights.

## Testing

```bash
python -m pytest tests/ -v
```

`tests/test_regression.py` specifically guards against the "worse than
bicubic" failure diagnosed and fixed in this project (see "Model
selection" above) using a hermetic, procedurally generated test image --
no network access or downloaded sample data required.

## Known limitations

- **No geospatial metadata is preserved end-to-end.** Training/inference
  operate on plain RGB rasters; converting a GeoTIFF to a training PNG
  (`scripts/download_sample_data.py`) discards CRS/transform/bounds. If you
  need a georeferenced enhanced output, you'll need to re-attach the
  original raster's metadata to the model's output yourself (e.g. with
  `rasterio`) -- this repo does not do that for you, and does not claim to.
- **The training corpus is still small** (3 real images + procedural
  terrain for the 3D model). The fixes in this project (validation-based
  model selection, a rebalanced loss) make the model *not worse than
  bicubic* on the held-out test image measured so far, but a handful of
  real images is not enough data for the kind of robust, general-purpose
  quality gains a production system would need — see "Scaling up
  training" above for what a larger run would look like.
- **The perceptual/edge loss term defaults to off** after being identified
  as a source of texture hallucination at this dataset size. It may behave
  better with more/more-diverse training data; re-validate with
  `scripts/evaluate.py` before re-enabling it rather than assuming.
- **CPU-only in this environment** (no GPU was available for training or
  evaluation here — see the hardware note in the engineering report). All
  measured numbers in this README were produced on CPU; a GPU would mainly
  change training speed, not correctness.

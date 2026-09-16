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
  download_sample_data.py   fetch real Landsat/NAIP demo imagery (no auth needed)
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
# Get some real demo imagery (Landsat/NAIP test fixtures, no login required)
python scripts/download_sample_data.py --out-dir data/sample

# Train (CPU-friendly settings shown; scale up channels/blocks/epochs with a GPU)
python -m satellite_enhance.train_2d \
  --data-dir data/sample --scale 4 --patch-size 96 --patches-per-image 20 \
  --batch-size 4 --epochs 15 --channels 64 --num-blocks 8 \
  --checkpoint checkpoints/rrdb_x4.pth

# Enhance a new image (tiled automatically if larger than --tile-size)
python -m satellite_enhance.infer_2d \
  --checkpoint checkpoints/rrdb_x4.pth \
  --input data/sample/landsat_rgb_1.png --output outputs/landsat_rgb_1_x4.png
```

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
  --texture outputs/landsat_rgb_1_x4.png \
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

## Datasets

The bundled demo uses small **real** Landsat/NAIP raster fixtures pulled
from open-source geospatial projects on GitHub (no auth needed — see
`scripts/download_sample_data.py`) plus procedurally generated fractal
terrain for the 3D demo (real DEM portals below need registration that
isn't available in a sandboxed environment).

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

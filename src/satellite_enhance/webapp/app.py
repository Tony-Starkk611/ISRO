"""Web UI + REST API for interactively running the 2D/3D satellite image
spatial enhancement models in a browser.

Run with:
    uvicorn satellite_enhance.webapp.app:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in a browser.
"""

from __future__ import annotations

import base64
import io
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from satellite_enhance.infer_2d import enhance_tiled, load_model as load_2d_model
from satellite_enhance.infer_3d import enhance_dem, load_model as load_3d_model
from satellite_enhance.utils.image_io import load_dem
from satellite_enhance.utils.terrain3d import dem_to_mesh, render_3d_preview, save_obj

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_2D_DIR = REPO_ROOT / "data" / "sample"
SAMPLE_3D_DIR = REPO_ROOT / "data" / "dem"
CHECKPOINT_2D = REPO_ROOT / "checkpoints" / "rrdb_x4.pth"
CHECKPOINT_3D = REPO_ROOT / "checkpoints" / "dem_sr_x4.pth"
TMP_DIR = REPO_ROOT / "outputs" / "webapp_tmp"
if TMP_DIR.exists():
    import shutil

    shutil.rmtree(TMP_DIR)
TMP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPIN_AZIMUTHS = [-120, -75, -30, 15, 60, 105, 150, -165]

app = FastAPI(title="Satellite Image Spatial Enhancement")


@lru_cache(maxsize=1)
def get_2d_model():
    if not CHECKPOINT_2D.exists():
        raise HTTPException(500, f"2D checkpoint not found at {CHECKPOINT_2D}. Run train_2d.py first.")
    model, scale = load_2d_model(str(CHECKPOINT_2D), DEVICE)
    return model, scale


@lru_cache(maxsize=1)
def get_3d_model():
    if not CHECKPOINT_3D.exists():
        raise HTTPException(500, f"3D checkpoint not found at {CHECKPOINT_3D}. Run train_3d.py first.")
    model, scale = load_3d_model(str(CHECKPOINT_3D), DEVICE)
    return model, scale


def image_to_data_uri(arr: np.ndarray) -> str:
    """Encode a float [0,1] or uint8 HxWx3 array as a PNG data URI."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_z_range(filename: str) -> tuple[float, float] | None:
    match = re.search(r"z(-?[\d.]+)-(-?[\d.]+)m", filename)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


# ---------------------------------------------------------------- samples --

@app.get("/api/samples/2d")
def list_2d_samples():
    return [
        {"name": p.name, "url": f"/data/sample/{p.name}"}
        for p in sorted(SAMPLE_2D_DIR.glob("*.png"))
    ]


@app.get("/api/samples/3d")
def list_3d_samples():
    out = []
    for p in sorted(SAMPLE_3D_DIR.glob("terrain_*.png")):
        zrange = parse_z_range(p.name)
        if zrange is None:
            continue
        out.append({"name": p.name, "z_min": zrange[0], "z_max": zrange[1]})
    return out


@app.get("/api/samples/3d/{name}/preview")
def sample_3d_preview(name: str):
    """On-the-fly quick-look render of a bundled DEM sample (for the picker)."""
    path = _safe_sample_path(SAMPLE_3D_DIR, name)
    zrange = parse_z_range(name)
    if zrange is None:
        raise HTTPException(400, "Could not parse elevation range from filename")
    raw = load_dem(path)
    dem_m = zrange[0] + (raw / 65535.0) * (zrange[1] - zrange[0])
    out_path = TMP_DIR / f"preview_{name}.png"
    if not out_path.exists():
        render_3d_preview(dem_m, out_path, title=name, max_grid=150)
    return FileResponse(out_path, media_type="image/png")


def _safe_sample_path(base_dir: Path, name: str) -> Path:
    path = (base_dir / name).resolve()
    if base_dir.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Sample not found")
    return path


# ------------------------------------------------------------- 2D enhance --

@app.post("/api/enhance/2d")
async def enhance_2d(file: UploadFile | None = File(None), sample: str | None = Form(None)):
    if file is not None:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    elif sample:
        path = _safe_sample_path(SAMPLE_2D_DIR, sample)
        img = Image.open(path).convert("RGB")
    else:
        raise HTTPException(400, "Provide either a file upload or a sample name")

    max_side = 320
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.BICUBIC)

    lr = np.asarray(img).astype(np.float32) / 255.0

    model, scale = get_2d_model()
    t0 = time.time()
    # tile size >= max_side ensures demo-sized (capped) inputs run in a
    # single forward pass instead of paying redundant overlap compute.
    enhanced = enhance_tiled(model, lr, scale, DEVICE, tile=max_side, overlap=16)
    elapsed = time.time() - t0

    bicubic = np.asarray(
        img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    ).astype(np.float32) / 255.0

    return JSONResponse(
        {
            "scale": scale,
            "elapsed_sec": round(elapsed, 2),
            "input_size": list(img.size),
            "output_size": [img.width * scale, img.height * scale],
            "input_png": image_to_data_uri(lr),
            "bicubic_png": image_to_data_uri(bicubic),
            "enhanced_png": image_to_data_uri(enhanced),
        }
    )


# ------------------------------------------------------------- 3D enhance --

@app.post("/api/enhance/3d")
async def enhance_3d(
    file: UploadFile | None = File(None),
    sample: str | None = Form(None),
    z_min: float | None = Form(None),
    z_max: float | None = Form(None),
):
    if file is not None:
        if z_min is None or z_max is None:
            raise HTTPException(400, "z_min and z_max are required when uploading a custom DEM")
        raw_bytes = await file.read()
        img = Image.open(io.BytesIO(raw_bytes))
        raw = np.asarray(img).astype(np.float32)
        if raw.ndim == 3:
            raw = raw.mean(axis=-1)
        max_val = 65535.0 if raw.max() > 255 else 255.0
        dem_m = z_min + (raw / max_val) * (z_max - z_min)
    elif sample:
        path = _safe_sample_path(SAMPLE_3D_DIR, sample)
        zrange = parse_z_range(sample)
        if zrange is None:
            raise HTTPException(400, "Could not parse elevation range from sample filename")
        raw = load_dem(path)
        dem_m = zrange[0] + (raw / 65535.0) * (zrange[1] - zrange[0])
    else:
        raise HTTPException(400, "Provide either a file upload or a sample name")

    if max(dem_m.shape) > 300:
        raise HTTPException(400, "DEM too large for interactive demo (max 300x300 px) -- use infer_3d.py for large tiles")

    model, scale = get_3d_model()
    t0 = time.time()
    enhanced = enhance_dem(model, dem_m, DEVICE)
    elapsed = time.time() - t0

    before_frames = [
        image_to_data_uri(_render_frame(dem_m, az)) for az in SPIN_AZIMUTHS
    ]
    after_frames = [
        image_to_data_uri(_render_frame(enhanced, az)) for az in SPIN_AZIMUTHS
    ]

    mesh_id = uuid.uuid4().hex[:16]
    vertices, faces = dem_to_mesh(enhanced, pixel_size_m=1.0)
    save_obj(TMP_DIR / f"{mesh_id}.obj", vertices, faces)

    return JSONResponse(
        {
            "scale": scale,
            "elapsed_sec": round(elapsed, 2),
            "input_shape": list(dem_m.shape),
            "output_shape": list(enhanced.shape),
            "elevation_range_m": [float(dem_m.min()), float(dem_m.max())],
            "before_frames": before_frames,
            "after_frames": after_frames,
            "mesh_id": mesh_id,
        }
    )


def _render_frame(dem: np.ndarray, azimuth: float) -> np.ndarray:
    buf_path = TMP_DIR / f"_frame_{uuid.uuid4().hex}.png"
    render_3d_preview(dem, buf_path, z_exaggeration=2.0, azim=azimuth, max_grid=140, title="")
    arr = np.asarray(Image.open(buf_path).convert("RGB"))
    buf_path.unlink(missing_ok=True)
    return arr


@app.get("/api/mesh/{mesh_id}.obj")
def download_mesh(mesh_id: str):
    if not re.fullmatch(r"[0-9a-f]{16}", mesh_id):
        raise HTTPException(400, "Invalid mesh id")
    path = TMP_DIR / f"{mesh_id}.obj"
    if not path.exists():
        raise HTTPException(404, "Mesh not found (it may have expired)")
    return FileResponse(path, media_type="text/plain", filename="enhanced_terrain.obj")


# ------------------------------------------------------------------ pages --

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/data/sample", StaticFiles(directory=str(SAMPLE_2D_DIR)), name="sample2d")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()

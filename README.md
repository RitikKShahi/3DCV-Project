# 3DCV-Project — Incremental 3D Gaussian Splatting Reconstruction Platform

A full-stack web application and incremental 3D Gaussian Splatting (3DGS) reconstruction engine that converts videos or photo sets into 3D Gaussian Splatting (`.ply`, `.splat`) and mesh (`.obj`) scenes—supporting CPU-only execution as well as PyTorch GPU acceleration (`MPS` on Apple Silicon and `CUDA` on NVIDIA).

## Key Features

- **Flexible Input Ingestion**:
  - **Video Upload** (`POST /api/upload`): MP4, MOV, AVI, WebM, MKV with FFmpeg / OpenCV frame extraction, Laplacian blur filtering, and frame-difference keyframe selection (`reconstruction/preprocessing/`).
  - **Photo Set Upload** (`POST /api/upload/photos`): Supports multi-image collections (up to **500 keyframes**).
- **CPU & GPU Compute Device Support (`auto | cpu | mps | cuda`)**:
  - Pure PyTorch differentiable EWA Gaussian splatting rasterizer (`reconstruction/gaussian/rasterizer.py`) with no mandatory CUDA compiler extensions (`diff-gaussian-rasterization` or `simple-knn` not required).
  - Automatic device resolution with a safe subprocess MPS kernel probe (`probe_mps_support()` in `reconstruction/config.py`) to detect OS/PyTorch Metal compatibility before enabling MPS, cleanly falling back to CPU when needed.
  - Optional CUDA acceleration for `pycolmap` feature extraction, matching, and bundle adjustment when running on CUDA GPUs.
- **Custom Camera FOV Selection & Preview**:
  - Configurable `camera_fov_deg` parameter (`0` for auto heuristic, or `20°–140°` explicit pinhole horizontal FoV) with an interactive live framing preview widget (`frontend/src/components/FovPreviewWidget.tsx`).
- **Incremental Structure-from-Motion (SfM)**:
  - **Primary**: Isolated `pycolmap` subprocess worker (`reconstruction/sfm/pycolmap_worker.py`) supporting exhaustive matching (≤ 120 images) and sequential overlap matching (> 120 images) with `fix_existing_frames=True` incremental registration.
  - **Fallback**: OpenCV + SciPy sparse Bundle Adjustment (`reconstruction/sfm/opencv_sfm.py`).
- **Incremental 3DGS Training & Export**:
  - `cKDTree`-based scale initialization (`reconstruction/gaussian/model.py`), L1 + SSIM photometric loss (`reconstruction/gaussian/losses.py`), adaptive densification/pruning with screen-space oversize & isolated-outlier cleanup (`reconstruction/gaussian/densification.py`), checkpoint persistence (`reconstruction/incremental/checkpoint.py`), and exports to `.ply`, `.splat`, and `.obj` (`reconstruction/export/`).
- **Full-Stack Web Platform**:
  - **Backend**: FastAPI + async SQLAlchemy (SQLite by default, PostgreSQL via `GS_PLATFORM_DATABASE_URL`) + background `ThreadPoolExecutor` worker + WebSocket live progress streaming (`/ws/jobs/{job_id}`).
  - **Frontend**: Next.js 14 + TypeScript + Tailwind CSS + Three.js + `@mkkellogg/gaussian-splats-3d` multi-mode 3D viewer (`Splat`, `Geometry`, `Mesh`, `Triangle Splats`) with live incremental batch previews.

---

## Project Structure

```text
├── backend/                  # FastAPI routes, SQLAlchemy models, async worker & WebSocket manager
├── reconstruction/           # Core 3DGS & SfM pipeline (preprocessing, SfM, Gaussian model/rasterizer, export)
├── frontend/                 # Next.js 14 TypeScript UI & Three.js / WebGL 3DGS viewer
├── scripts/                  # Standalone CPU benchmark runner (benchmark_cpu.py)
├── tests/                    # Unit, API integration, and end-to-end synthetic pipeline tests
├── docker-compose.yml        # Optional FastAPI + Next.js + PostgreSQL stack
├── pyproject.toml            # Python project & pytest configuration
└── requirements.txt          # Python backend & reconstruction dependencies
```

---

## Installation & Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- FFmpeg (`brew install ffmpeg` on macOS or `apt install ffmpeg` on Linux)

### 1. Backend & Reconstruction Engine

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU wheel or standard PyTorch for MPS/CUDA)
pip install --upgrade pip
pip install torch torchvision

# Install remaining Python dependencies
pip install -r requirements.txt
```

Start the FastAPI backend:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Check detected hardware and device availability:
```bash
curl http://localhost:8000/api/health
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

### 3. Docker Compose (Optional)

```bash
docker compose up --build
```

### 4. Standalone CLI Reconstruction

```bash
PYTHONPATH=. .venv/bin/python -m reconstruction.pipeline path/to/video.mp4 \
  -o output_recon \
  --device auto \
  --max-res 384 \
  --max-keyframes 20 \
  --max-gaussians 5000 \
  --iters 200 \
  --final-iters 150
```

---

## Testing & Build Verification

Run the test suite:
```bash
.venv/bin/pytest -v tests/
```

Verify the Next.js production build:
```bash
cd frontend && npm run build
```

Run multi-scene benchmarks:
```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_cpu.py
```

### Measured CPU Benchmarks (Baseline)
Measured on an 8-core Apple Silicon CPU (8 GB RAM, Python 3.11, PyTorch 2.15.0 CPU, `pycolmap` CPU, `320×240`, 12 keyframes across 3 incremental batches, 60 iters/batch + 40 final refinement iters):

| Scene Type | Resolution | Keyframes | Registered Cameras | Gaussians | SfM Backend | Final PSNR | Total Time | Peak RSS Memory |
|---|---|---|---|---|---|---|---|---|
| **Object Orbit** | 320×240 | 12 | 12 / 12 | 1,094 | `pycolmap` | 18.03 dB | 162.9s (~2.7 min) | 567.0 MB |
| **Indoor Corner** | 320×240 | 12 | 12 / 12 | 1,492 | `pycolmap` | 18.78 dB | 173.1s (~2.9 min) | 683.1 MB |
| **Outdoor Facade** | 320×240 | 12 | 12 / 12 | 1,239 | `pycolmap` | 18.62 dB | 170.4s (~2.8 min) | 698.5 MB |


Sample Dataset Download Link- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

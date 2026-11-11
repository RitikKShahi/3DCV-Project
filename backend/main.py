"""FastAPI application entry point for CPU-only 3DGS Reconstruction Platform."""

from contextlib import asynccontextmanager
import logging
import os

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import export, projects, reconstruction, upload
from backend.api.websocket import ws_manager
from backend.config import settings
from backend.models.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(projects.router)
app.include_router(reconstruction.router)
app.include_router(export.router)


@app.get("/api/health")
async def health_check() -> dict:
    from reconstruction.config import probe_mps_support

    cuda_available = bool(torch.cuda.is_available())
    mps_detected = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    mps_available, mps_note = probe_mps_support()
    available_devices = ["cpu"]
    if mps_available:
        available_devices.append("mps")
    if cuda_available:
        available_devices.append("cuda")
    recommended = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
    return {
        "status": "ok",
        "device": recommended,
        "recommended_device": recommended,
        "available_devices": available_devices,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "mps_detected": mps_detected,
        "mps_note": mps_note,
        "torch_version": torch.__version__,
        "cpu_cores": os.cpu_count() or 1,
    }


@app.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            # Keep socket alive; client may send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(job_id, websocket)


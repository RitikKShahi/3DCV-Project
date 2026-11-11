"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class ReconstructionParams(BaseModel):
    max_resolution: int = Field(default=384, ge=160, le=1280)
    frame_interval: int = Field(default=10, ge=1, le=120)
    max_keyframes: int = Field(default=20, ge=4, le=500)
    max_gaussians: int = Field(default=5000, ge=500, le=50000)
    iterations_per_batch: int = Field(default=200, ge=20, le=3000)
    final_iterations: int = Field(default=150, ge=0, le=2000)
    initial_batch_size: int = Field(default=8, ge=3, le=100)
    incremental_batch_size: int = Field(default=4, ge=1, le=50)
    cpu_threads: int = Field(default=4, ge=1, le=16)
    device: str = Field(default="auto", pattern="^(auto|cpu|mps|cuda)$")
    sfm_backend: str = Field(default="auto", pattern="^(auto|pycolmap|opencv)$")
    sh_degree: int = Field(default=0, ge=0, le=3)
    camera_fov_deg: float = Field(default=0.0, ge=0.0, le=140.0)
    adaptive_early_stop: bool = True
    resume: bool = True


class JobResponse(BaseModel):
    id: str
    project_id: str
    status: str
    stage: str
    progress: float
    message: str
    batch_index: int
    total_batches: int
    num_keyframes: int
    num_cameras: int
    num_gaussians: int
    psnr: float
    loss: float
    sfm_backend: str
    elapsed_sec: float
    peak_memory_mb: float
    has_ply: bool
    has_splat: bool
    has_preview: bool
    preview_batches: list[int] = []
    metrics_history: list[dict[str, Any]] = []
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    video_filename: str
    video_width: int
    video_height: int
    video_duration_sec: float
    status: str
    created_at: datetime
    updated_at: datetime
    latest_job: Optional[JobResponse] = None


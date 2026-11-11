"""Background CPU worker orchestrating non-blocking 3DGS reconstruction jobs."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import time
from typing import Any

from sqlalchemy import select

from backend.api.websocket import ws_manager
from backend.config import settings
from backend.models.database import AsyncSessionLocal, Project, ReconstructionJob
from reconstruction.config import ReconstructionConfig
from reconstruction.pipeline import ProgressEvent, ReconstructionPipeline

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="3dgs-cpu")
_active_futures: dict[str, asyncio.Task] = {}


def _build_config(params: dict[str, Any]) -> ReconstructionConfig:
    return ReconstructionConfig(
        max_resolution=int(params.get("max_resolution", 384)),
        frame_interval=int(params.get("frame_interval", 10)),
        max_keyframes=int(params.get("max_keyframes", 20)),
        max_gaussians=int(params.get("max_gaussians", 5000)),
        iterations_per_batch=int(params.get("iterations_per_batch", 200)),
        final_iterations=int(params.get("final_iterations", 150)),
        initial_batch_size=int(params.get("initial_batch_size", 8)),
        incremental_batch_size=int(params.get("incremental_batch_size", 4)),
        cpu_threads=int(params.get("cpu_threads", 4)),
        device=str(params.get("device", "auto")),
        sfm_backend=str(params.get("sfm_backend", "auto")),
        sh_degree=int(params.get("sh_degree", 0)),
        camera_fov_deg=float(params.get("camera_fov_deg", 0.0)),
        adaptive_early_stop=bool(params.get("adaptive_early_stop", True)),
    ).clamp_for_safety()


async def _update_job_db(job_id: str, **fields: Any) -> None:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(ReconstructionJob).where(ReconstructionJob.id == job_id))
        job = res.scalar_one_or_none()
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        await session.commit()


async def _update_project_status(project_id: str, status: str) -> None:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Project).where(Project.id == project_id))
        proj = res.scalar_one_or_none()
        if proj is not None:
            proj.status = status
            await session.commit()


async def run_reconstruction_job(
    job_id: str,
    project_id: str,
    video_path: Path,
    work_dir: Path,
    params: dict[str, Any],
) -> None:
    """Run CPU reconstruction in ThreadPoolExecutor while streaming DB & WebSocket updates."""
    loop = asyncio.get_running_loop()
    config = _build_config(params)
    resume = bool(params.get("resume", True))
    t0 = time.perf_counter()
    last_emit = [0.0]

    await _update_job_db(job_id, status="running", stage="starting", message="Initializing CPU pipeline...")
    await _update_project_status(project_id, "reconstructing")

    def on_progress(ev: ProgressEvent) -> None:
        now = time.perf_counter()
        # Throttle frequent step events to ~2/sec except major stage transitions
        is_major = ev.stage in ("preprocessing", "sfm", "incremental_preview", "refining", "exporting", "completed", "resumed")
        if not is_major and (now - last_emit[0]) < 0.5:
            return
        last_emit[0] = now

        elapsed = now - t0
        preview_url = (
            f"/api/export/{job_id}/preview/latest"
            if (work_dir / "previews" / "latest_preview.splat").exists()
            else None
        )
        ws_payload = {
            "job_id": job_id,
            "project_id": project_id,
            "status": "completed" if ev.stage == "completed" else "running",
            "stage": ev.stage,
            "progress": float(ev.progress),
            "message": ev.message,
            "batch_index": ev.batch_index,
            "total_batches": ev.total_batches,
            "num_gaussians": ev.num_gaussians,
            "num_cameras": ev.num_cameras,
            "psnr": float(ev.psnr),
            "loss": float(ev.loss),
            "elapsed_sec": round(elapsed, 2),
            "preview_url": preview_url,
            "has_new_preview": ev.stage == "incremental_preview",
        }

        async def _sync() -> None:
            db_fields: dict[str, Any] = {
                "stage": ev.stage,
                "progress": float(ev.progress),
                "message": ev.message,
                "batch_index": ev.batch_index,
                "total_batches": ev.total_batches,
                "num_gaussians": ev.num_gaussians,
                "num_cameras": ev.num_cameras,
                "psnr": float(ev.psnr),
                "loss": float(ev.loss),
                "elapsed_sec": elapsed,
            }
            if ev.preview_splat_path:
                db_fields["latest_preview_path"] = ev.preview_splat_path
            await _update_job_db(job_id, **db_fields)
            await ws_manager.broadcast(job_id, ws_payload)

        asyncio.run_coroutine_threadsafe(_sync(), loop)

    def _run_blocking():
        pipeline = ReconstructionPipeline(config)
        return pipeline.run(
            video_path=video_path,
            work_dir=work_dir,
            progress_callback=on_progress,
            resume=resume,
        )

    try:
        result = await loop.run_in_executor(_executor, _run_blocking)
        await _update_job_db(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            message=f"Completed: {result.num_gaussians} Gaussians, PSNR={result.final_psnr:.2f} dB",
            num_keyframes=result.num_keyframes,
            num_cameras=result.num_registered_cameras,
            num_gaussians=result.num_gaussians,
            psnr=result.final_psnr,
            loss=result.final_loss,
            sfm_backend=result.sfm_backend,
            elapsed_sec=result.total_time_sec,
            peak_memory_mb=result.peak_memory_mb,
            ply_path=str(result.ply_path),
            splat_path=str(result.splat_path),
            latest_preview_path=str(result.preview_dir / "latest_preview.splat"),
            metrics_history_json=json.dumps(result.batch_metrics),
        )
        await _update_project_status(project_id, "completed")
        await ws_manager.broadcast(
            job_id,
            {
                "job_id": job_id,
                "project_id": project_id,
                "status": "completed",
                "stage": "completed",
                "progress": 1.0,
                "message": f"Completed: {result.num_gaussians} Gaussians, PSNR={result.final_psnr:.2f} dB",
                "num_gaussians": result.num_gaussians,
                "num_cameras": result.num_registered_cameras,
                "psnr": result.final_psnr,
                "elapsed_sec": round(result.total_time_sec, 2),
                "peak_memory_mb": round(result.peak_memory_mb, 1),
                "preview_url": f"/api/export/{job_id}/splat",
                "has_new_preview": True,
            },
        )
    except Exception as exc:
        logger.exception("Reconstruction job %s failed", job_id)
        await _update_job_db(
            job_id,
            status="failed",
            stage="failed",
            message=f"Failed: {exc}",
            error_message=str(exc),
        )
        await _update_project_status(project_id, "failed")
        await ws_manager.broadcast(
            job_id,
            {
                "job_id": job_id,
                "project_id": project_id,
                "status": "failed",
                "stage": "failed",
                "progress": 0.0,
                "message": f"Reconstruction failed: {exc}",
                "error_message": str(exc),
            },
        )
    finally:
        _active_futures.pop(job_id, None)


def enqueue_reconstruction_job(
    job_id: str,
    project_id: str,
    video_path: Path,
    work_dir: Path,
    params: dict[str, Any],
) -> None:
    """Schedule background reconstruction task on the current FastAPI event loop."""
    task = asyncio.create_task(
        run_reconstruction_job(job_id, project_id, video_path, work_dir, params)
    )
    _active_futures[job_id] = task


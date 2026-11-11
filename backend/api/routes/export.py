"""Scene export, progressive preview, and derived representation endpoints."""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select

from backend.models.database import AsyncSessionLocal, ReconstructionJob
from reconstruction.export.representations import (
    export_mesh_obj,
    extract_geometry_data,
    extract_triangle_mesh,
    extract_triangle_splats,
    load_gaussian_model_from_artifacts,
)
from reconstruction.sfm.types import CameraPose

router = APIRouter(prefix="/api/export", tags=["export"])


async def _get_job(job_id: str) -> ReconstructionJob:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(ReconstructionJob).where(ReconstructionJob.id == job_id))
        job = res.scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


def _load_poses_from_checkpoint(work_dir: Path) -> dict[str, CameraPose] | None:
    state_path = work_dir / "checkpoint" / "state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
        return {name: CameraPose.from_dict(p) for name, p in state.get("poses", {}).items()}
    except Exception:
        return None


def _build_representation_sync(job: ReconstructionJob, mode: str) -> dict:
    work_dir = Path(job.work_dir)
    preview_dir = work_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cache_path = preview_dir / f"latest_{mode}.json"
    splat_ref = preview_dir / "latest_preview.splat"

    if cache_path.exists():
        # Reuse cache if up to date with latest preview
        if not splat_ref.exists() or cache_path.stat().st_mtime >= splat_ref.stat().st_mtime:
            return json.loads(cache_path.read_text())

    ckpt_pt = work_dir / "checkpoint" / "gaussians.pt"
    ply_path = Path(job.ply_path) if job.ply_path else (work_dir / "exports" / "scene.ply")
    model = load_gaussian_model_from_artifacts(checkpoint_pt=ckpt_pt, ply_path=ply_path)
    poses = _load_poses_from_checkpoint(work_dir)

    if mode == "geometry":
        data = extract_geometry_data(model, poses=poses)
    elif mode == "mesh":
        data = extract_triangle_mesh(model, grid_res=48)
    elif mode == "triangle_splats":
        data = extract_triangle_splats(model)
    else:
        raise ValueError(f"Unsupported representation mode: {mode}")

    cache_path.write_text(json.dumps(data))
    return data


@router.get("/{job_id}/ply")
async def download_ply(job_id: str) -> FileResponse:
    job = await _get_job(job_id)
    if not job.ply_path or not Path(job.ply_path).exists():
        raise HTTPException(status_code=404, detail="PLY scene not ready yet")
    return FileResponse(
        path=job.ply_path,
        filename=f"scene_{job_id}.ply",
        media_type="application/octet-stream",
    )


@router.get("/{job_id}/splat")
async def download_splat(job_id: str) -> FileResponse:
    job = await _get_job(job_id)
    if not job.splat_path or not Path(job.splat_path).exists():
        raise HTTPException(status_code=404, detail="SPLAT scene not ready yet")
    return FileResponse(
        path=job.splat_path,
        filename=f"scene_{job_id}.splat",
        media_type="application/octet-stream",
    )


@router.get("/{job_id}/mesh")
async def download_mesh_obj(job_id: str) -> FileResponse:
    job = await _get_job(job_id)
    work_dir = Path(job.work_dir)
    obj_path = work_dir / "exports" / "scene_mesh.obj"
    if not obj_path.exists():
        try:
            mesh_data = await asyncio.to_thread(_build_representation_sync, job, "mesh")
            await asyncio.to_thread(export_mesh_obj, mesh_data, obj_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Reconstruction data not ready yet") from exc
    return FileResponse(path=str(obj_path), filename=f"scene_mesh_{job_id}.obj", media_type="text/plain")


@router.get("/{job_id}/representation/{mode}")
async def get_scene_representation(job_id: str, mode: str) -> JSONResponse:
    """Return Geometry, Triangle Mesh, or Triangle Splatting data derived from Gaussians."""
    if mode not in ("geometry", "mesh", "triangle_splats"):
        raise HTTPException(status_code=400, detail="Mode must be 'geometry', 'mesh', or 'triangle_splats'")
    job = await _get_job(job_id)
    try:
        data = await asyncio.to_thread(_build_representation_sync, job, mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Preview geometry not ready yet") from exc
    return JSONResponse(content=data)


@router.get("/{job_id}/preview/latest")
async def get_latest_preview(job_id: str) -> FileResponse:
    """Serve the latest incremental `.splat` preview (or final `.splat` if done)."""
    job = await _get_job(job_id)
    work_dir = Path(job.work_dir)
    latest = work_dir / "previews" / "latest_preview.splat"
    if latest.exists():
        return FileResponse(path=str(latest), filename="preview.splat", media_type="application/octet-stream")
    if job.splat_path and Path(job.splat_path).exists():
        return FileResponse(path=job.splat_path, filename="scene.splat", media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="No preview available yet")


@router.get("/{job_id}/preview/batch/{batch_idx}")
async def get_batch_preview(job_id: str, batch_idx: int) -> FileResponse:
    job = await _get_job(job_id)
    path = Path(job.work_dir) / "previews" / f"preview_batch_{batch_idx}.splat"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Preview for batch {batch_idx} not found")
    return FileResponse(path=str(path), filename=path.name, media_type="application/octet-stream")


@router.get("/{job_id}/preview/image/{batch_idx}")
async def get_batch_render_image(job_id: str, batch_idx: int) -> FileResponse:
    job = await _get_job(job_id)
    path = Path(job.work_dir) / "previews" / f"render_batch_{batch_idx}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Preview render image for batch {batch_idx} not found")
    return FileResponse(path=str(path), media_type="image/jpeg")


@router.get("/{job_id}/benchmarks")
async def get_benchmarks(job_id: str) -> FileResponse:
    job = await _get_job(job_id)
    path = Path(job.work_dir) / "exports" / "benchmarks.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Benchmarks not generated yet")
    return FileResponse(path=str(path), filename=f"benchmarks_{job_id}.json", media_type="application/json")

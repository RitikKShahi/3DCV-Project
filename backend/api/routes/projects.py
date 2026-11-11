"""Project management endpoints (list, inspect, keyframes, delete)."""

import json
from pathlib import Path
import re

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.models.database import AsyncSessionLocal, Project, ReconstructionJob
from backend.models.schemas import JobResponse, ProjectResponse
from backend.storage.file_manager import delete_project_files

router = APIRouter(prefix="/api/projects", tags=["projects"])


def job_to_schema(job: ReconstructionJob) -> JobResponse:
    work_dir = Path(job.work_dir)
    previews_dir = work_dir / "previews"
    preview_batches: list[int] = []
    if previews_dir.exists():
        for p in sorted(previews_dir.glob("preview_batch_*.splat")):
            m = re.search(r"preview_batch_(\d+)\.splat$", p.name)
            if m:
                preview_batches.append(int(m.group(1)))

    try:
        metrics_history = json.loads(job.metrics_history_json or "[]")
    except Exception:
        metrics_history = []

    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        batch_index=job.batch_index,
        total_batches=job.total_batches,
        num_keyframes=job.num_keyframes,
        num_cameras=job.num_cameras,
        num_gaussians=job.num_gaussians,
        psnr=job.psnr,
        loss=job.loss,
        sfm_backend=job.sfm_backend,
        elapsed_sec=job.elapsed_sec,
        peak_memory_mb=job.peak_memory_mb,
        has_ply=bool(job.ply_path and Path(job.ply_path).exists()),
        has_splat=bool(job.splat_path and Path(job.splat_path).exists()),
        has_preview=bool(
            (previews_dir / "latest_preview.splat").exists()
            or (job.splat_path and Path(job.splat_path).exists())
        ),
        preview_batches=sorted(preview_batches),
        metrics_history=metrics_history,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def project_to_schema(project: Project) -> ProjectResponse:
    latest = job_to_schema(project.jobs[0]) if project.jobs else None
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        video_filename=project.video_filename,
        video_width=project.video_width,
        video_height=project.video_height,
        video_duration_sec=project.video_duration_sec,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        latest_job=latest,
    )


@router.get("", response_model=list[ProjectResponse])
async def list_projects() -> list[ProjectResponse]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(Project).options(selectinload(Project.jobs)).order_by(Project.created_at.desc())
        )
        projects = res.scalars().all()
        return [project_to_schema(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str) -> ProjectResponse:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(Project).options(selectinload(Project.jobs)).where(Project.id == project_id)
        )
        project = res.scalar_one_or_none()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return project_to_schema(project)


@router.delete("/{project_id}")
async def delete_project(project_id: str) -> dict[str, str]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Project).where(Project.id == project_id))
        project = res.scalar_one_or_none()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        await session.delete(project)
        await session.commit()

    delete_project_files(project_id)
    return {"status": "deleted", "project_id": project_id}


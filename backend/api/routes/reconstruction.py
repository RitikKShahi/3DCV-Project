"""Reconstruction job creation, status, and resume endpoints."""

import json
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from backend.api.routes.projects import job_to_schema
from backend.models.database import AsyncSessionLocal, Project, ReconstructionJob
from backend.models.schemas import JobResponse, ReconstructionParams
from backend.storage.file_manager import job_work_dir
from backend.workers.reconstruction_worker import enqueue_reconstruction_job

router = APIRouter(prefix="/api/reconstruction", tags=["reconstruction"])


@router.post("/{project_id}/start", response_model=JobResponse)
async def start_reconstruction(project_id: str, params: ReconstructionParams) -> JobResponse:
    """Create and asynchronously start a CPU-only incremental 3DGS reconstruction job."""
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(Project).where(Project.id == project_id))
        project = res.scalar_one_or_none()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        video_path = Path(project.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=400, detail="Uploaded video file missing on disk")

        # Check if resuming the latest failed/interrupted job
        existing_job = None
        if params.resume:
            jres = await session.execute(
                select(ReconstructionJob)
                .where(ReconstructionJob.project_id == project_id)
                .order_by(ReconstructionJob.created_at.desc())
            )
            latest = jres.scalars().first()
            if latest and latest.status == "failed" and (Path(latest.work_dir) / "checkpoint" / "state.json").exists():
                existing_job = latest

        if existing_job is not None:
            job = existing_job
            job.status = "queued"
            job.stage = "resuming"
            job.message = "Resuming from saved checkpoint..."
            job.error_message = None
            job.config_json = json.dumps(params.model_dump())
            await session.commit()
            await session.refresh(job)
        else:
            job_id = uuid.uuid4().hex[:12]
            work_dir = job_work_dir(project_id, job_id)
            job = ReconstructionJob(
                id=job_id,
                project_id=project_id,
                status="queued",
                stage="queued",
                message="Job queued for CPU reconstruction",
                config_json=json.dumps(params.model_dump()),
                work_dir=str(work_dir),
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)

    enqueue_reconstruction_job(
        job_id=job.id,
        project_id=project_id,
        video_path=Path(project.video_path),
        work_dir=Path(job.work_dir),
        params=params.model_dump(),
    )
    return job_to_schema(job)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str) -> JobResponse:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(ReconstructionJob).where(ReconstructionJob.id == job_id))
        job = res.scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=404, detail="Reconstruction job not found")
        return job_to_schema(job)


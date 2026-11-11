"""Video upload endpoint creating a new reconstruction project."""

import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.models.database import AsyncSessionLocal, Project
from backend.models.schemas import ProjectResponse
from backend.storage.file_manager import save_uploaded_photos, save_uploaded_video
from reconstruction.preprocessing.frame_extractor import get_video_info

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("", response_model=ProjectResponse)
async def upload_video(
    file: UploadFile = File(...),
    name: str = Form(""),
    description: str = Form(""),
) -> ProjectResponse:
    """Upload a real-world video (MP4, MOV, AVI, WebM, MKV) and create a project."""
    project = Project(
        id=uuid.uuid4().hex[:12],
        name=(name.strip() or (file.filename or "Untitled Scene").rsplit(".", 1)[0]),
        description=description.strip(),
        video_filename=file.filename or "video.mp4",
        video_path="",
        status="uploaded",
    )

    try:
        filename, saved_path = await save_uploaded_video(project.id, file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        info = get_video_info(saved_path)
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid or unreadable video: {exc}") from exc

    project.video_filename = filename
    project.video_path = str(saved_path)
    project.video_width = info.width
    project.video_height = info.height
    project.video_duration_sec = info.duration_sec

    async with AsyncSessionLocal() as session:
        session.add(project)
        await session.commit()
        await session.refresh(project)

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
        latest_job=None,
    )


@router.post("/photos", response_model=ProjectResponse)
async def upload_photos(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    description: str = Form(""),
) -> ProjectResponse:
    """Upload a set of overlapping photos (JPG, PNG, WebP, etc.) and create a project."""
    project_id = uuid.uuid4().hex[:12]
    try:
        label, photos_dir, (w, h) = await save_uploaded_photos(project_id, files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project = Project(
        id=project_id,
        name=(name.strip() or f"Photo Set ({label})"),
        description=description.strip(),
        video_filename=label,
        video_path=str(photos_dir),
        video_width=w,
        video_height=h,
        video_duration_sec=0.0,
        status="uploaded",
    )

    async with AsyncSessionLocal() as session:
        session.add(project)
        await session.commit()
        await session.refresh(project)

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
        latest_job=None,
    )


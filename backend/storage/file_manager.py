"""Storage and file management for uploaded videos, jobs, and exports."""

import shutil
from pathlib import Path

from fastapi import UploadFile

from backend.config import settings

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def project_dir(project_id: str) -> Path:
    path = settings.data_dir / "projects" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_work_dir(project_id: str, job_id: str) -> Path:
    path = project_dir(project_id) / "jobs" / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_uploaded_video(project_id: str, upload: UploadFile) -> tuple[str, Path]:
    """Stream uploaded video file to disk without exhausting RAM."""
    filename = Path(upload.filename or "video.mp4").name
    suffix = Path(filename).suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}")

    dest_dir = project_dir(project_id)
    dest_path = dest_dir / f"input{suffix}"

    max_bytes = settings.max_upload_mb * 1024 * 1024
    total_bytes = 0
    with dest_path.open("wb") as out:
        while chunk := await upload.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                dest_path.unlink(missing_ok=True)
                raise ValueError(f"File exceeds {settings.max_upload_mb} MB upload limit")
            out.write(chunk)

    return filename, dest_path


async def save_uploaded_photos(
    project_id: str,
    uploads: list[UploadFile],
) -> tuple[str, Path, tuple[int, int]]:
    """Stream multiple uploaded photos to a project `photos/` directory."""
    from PIL import Image, ImageOps

    if len(uploads) < 3:
        raise ValueError("Please upload at least 3 overlapping photos for 3D reconstruction.")

    photos_dir = project_dir(project_id) / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    total_bytes = 0
    saved_paths: list[Path] = []
    sample_dims: tuple[int, int] = (0, 0)

    # Sort by original filename so ordered camera sequences stay ordered
    ordered = sorted(uploads, key=lambda u: (u.filename or "").lower())

    for idx, upload in enumerate(ordered, start=1):
        orig_name = Path(upload.filename or f"photo_{idx:04d}.jpg").name
        suffix = Path(orig_name).suffix.lower() or ".jpg"
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            shutil.rmtree(photos_dir, ignore_errors=True)
            raise ValueError(
                f"Unsupported photo format '{suffix}' in '{orig_name}'. "
                f"Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            )

        dest_path = photos_dir / f"photo_{idx:04d}{suffix}"
        with dest_path.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    shutil.rmtree(photos_dir, ignore_errors=True)
                    raise ValueError(f"Total photo upload exceeds {settings.max_upload_mb} MB limit")
                out.write(chunk)

        if sample_dims == (0, 0):
            try:
                with Image.open(dest_path) as im:
                    im = ImageOps.exif_transpose(im)
                    sample_dims = (int(im.width), int(im.height))
            except Exception:
                pass
        saved_paths.append(dest_path)

    return f"{len(saved_paths)} photos", photos_dir, sample_dims


def delete_project_files(project_id: str) -> None:
    pdir = settings.data_dir / "projects" / project_id
    if pdir.exists():
        shutil.rmtree(pdir, ignore_errors=True)


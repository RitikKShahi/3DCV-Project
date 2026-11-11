"""Video frame extraction using FFmpeg (primary) and OpenCV (fallback)."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Metadata for an input video file."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def get_video_info(video_path: Path) -> VideoInfo:
    """Inspect video metadata via OpenCV / ffprobe."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if frame_count <= 0:
        # Fallback to ffprobe if container doesn't report frame count
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", str(video_path),
            ]
            out = subprocess.check_output(cmd, text=True)
            streams = json.loads(out).get("streams", [])
            if streams:
                s = streams[0]
                duration = float(s.get("duration", 0.0))
                frame_count = int(s.get("nb_frames") or (duration * fps))
        except Exception:
            pass

    duration_sec = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def _compute_target_size(width: int, height: int, max_resolution: int) -> tuple[int, int]:
    """Compute even dimensions scaled so max(w, h) <= max_resolution."""
    if max(width, height) <= max_resolution:
        w, h = width, height
    elif width >= height:
        w = max_resolution
        h = int(round(height * (max_resolution / width)))
    else:
        h = max_resolution
        w = int(round(width * (max_resolution / height)))
    # Ensure even dimensions for encoders/filters
    w = max(2, w - (w % 2))
    h = max(2, h - (h % 2))
    return w, h


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 12,
    max_resolution: int = 480,
    max_raw_frames: int = 500,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[Path]:
    """Extract and resize frames from a video using FFmpeg with OpenCV fallback.

    Returns sorted list of extracted JPEG frame paths.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    target_w, target_h = _compute_target_size(info.width, info.height, max_resolution)

    # Adapt interval if video is very long so we don't extract thousands of frames
    effective_interval = frame_interval
    if info.frame_count > 0 and (info.frame_count / effective_interval) > max_raw_frames:
        effective_interval = max(1, int(info.frame_count // max_raw_frames))
        logger.info(
            "Long video (%d frames): adjusting frame_interval from %d to %d",
            info.frame_count, frame_interval, effective_interval,
        )

    if progress_cb:
        progress_cb(0.05, f"Extracting frames ({target_w}x{target_h}, every {effective_interval} frames)...")

    # Try FFmpeg first (fast, handles MOV/MP4/HEVC rotation metadata cleanly)
    pattern = str(output_dir / "frame_%05d.jpg")
    ffmpeg_ok = False
    try:
        vf = f"select=not(mod(n\\,{effective_interval})),scale={target_w}:{target_h}"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", vf,
            "-vsync", "vfr",
            "-q:v", "2",
            "-frames:v", str(max_raw_frames),
            pattern,
        ]
        subprocess.run(cmd, check=True, timeout=180)
        extracted = sorted(output_dir.glob("frame_*.jpg"))
        if len(extracted) > 0:
            ffmpeg_ok = True
    except Exception as exc:
        logger.warning("FFmpeg extraction failed (%s), falling back to OpenCV", exc)

    if not ffmpeg_ok:
        extracted = _extract_with_opencv(
            video_path, output_dir, effective_interval, target_w, target_h, max_raw_frames
        )

    if not extracted:
        raise RuntimeError(f"Failed to extract any frames from {video_path}")

    if progress_cb:
        progress_cb(0.15, f"Extracted {len(extracted)} candidate frames")

    logger.info("Extracted %d frames to %s", len(extracted), output_dir)
    return extracted


def _extract_with_opencv(
    video_path: Path,
    output_dir: Path,
    interval: int,
    target_w: int,
    target_h: int,
    max_raw_frames: int,
) -> list[Path]:
    """OpenCV fallback frame extractor."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {video_path}")

    extracted: list[Path] = []
    idx = 0
    saved = 0
    while saved < max_raw_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            out_path = output_dir / f"frame_{saved + 1:05d}.jpg"
            cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(out_path)
            saved += 1
        idx += 1

    cap.release()
    return extracted


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def prepare_images_from_dir(
    images_dir: Path,
    output_dir: Path,
    max_resolution: int = 720,
    max_images: int = 500,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[Path]:
    """Normalize and resize an uploaded directory of photos into sequential JPEGs."""
    from PIL import Image, ImageOps

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[:max_images]

    if not src_files:
        raise RuntimeError(f"No valid image files found in {images_dir}")

    if progress_cb:
        progress_cb(0.05, f"Preparing {len(src_files)} uploaded photos...")

    prepared: list[Path] = []
    for idx, src in enumerate(src_files, start=1):
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                target_w, target_h = _compute_target_size(im.width, im.height, max_resolution)
                if (im.width, im.height) != (target_w, target_h):
                    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
                out_path = output_dir / f"frame_{idx:05d}.jpg"
                im.save(out_path, format="JPEG", quality=95)
                prepared.append(out_path)
        except Exception as exc:
            logger.warning("Skipping unreadable image %s (%s)", src, exc)

    if not prepared:
        raise RuntimeError(f"Failed to read any uploaded photos in {images_dir}")

    if progress_cb:
        progress_cb(0.15, f"Prepared {len(prepared)} photos for reconstruction")

    logger.info("Prepared %d photos into %s", len(prepared), output_dir)
    return prepared


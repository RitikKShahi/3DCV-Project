"""Quality filtering: blur detection and redundant frame removal."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameQuality:
    """Quality metrics for a single extracted frame."""

    path: Path
    index: int
    blur_score: float
    mean_brightness: float
    motion_diff: float
    kept: bool = True
    reject_reason: str = ""


def compute_blur_score(image_gray: np.ndarray) -> float:
    """Compute Laplacian variance sharpness metric (higher = sharper)."""
    return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())


def compute_frame_difference(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Compute normalized mean absolute difference in [0, 1] between two small grayscale frames."""
    small_a = cv2.resize(gray_a, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_b = cv2.resize(gray_b, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.mean(np.abs(small_a - small_b)) / 255.0)


def filter_frames(
    frame_paths: list[Path],
    blur_threshold: float = 40.0,
    min_frame_diff: float = 0.04,
    min_keep: int = 6,
) -> tuple[list[Path], list[FrameQuality]]:
    """Filter out blurry, poorly exposed, and near-duplicate frames.

    Guarantees at least `min_keep` frames (sorted by sharpness) if enough frames exist,
    so slightly soft videos don't fail completely.
    """
    if not frame_paths:
        return [], []

    metrics: list[FrameQuality] = []
    prev_kept_gray: np.ndarray | None = None

    for idx, path in enumerate(frame_paths):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            metrics.append(FrameQuality(path, idx, 0.0, 0.0, 0.0, False, "unreadable"))
            continue

        blur = compute_blur_score(img)
        brightness = float(np.mean(img))
        diff = 1.0 if prev_kept_gray is None else compute_frame_difference(prev_kept_gray, img)

        reason = ""
        kept = True
        if brightness < 15.0 or brightness > 245.0:
            kept = False
            reason = f"extreme_exposure({brightness:.1f})"
        elif blur < blur_threshold:
            kept = False
            reason = f"blurry({blur:.1f}<{blur_threshold:.1f})"
        elif diff < min_frame_diff:
            kept = False
            reason = f"redundant(diff={diff:.3f}<{min_frame_diff:.3f})"

        if kept:
            prev_kept_gray = img

        metrics.append(
            FrameQuality(
                path=path,
                index=idx,
                blur_score=blur,
                mean_brightness=brightness,
                motion_diff=diff,
                kept=kept,
                reject_reason=reason,
            )
        )

    kept_paths = [m.path for m in metrics if m.kept]

    # Safety fallback: if too few frames passed strict thresholds, relax by picking sharpest evenly spaced frames
    if len(kept_paths) < min(min_keep, len(frame_paths)):
        logger.warning(
            "Only %d frames passed strict quality filter; relaxing filter to keep at least %d frames",
            len(kept_paths), min(min_keep, len(frame_paths)),
        )
        valid = [m for m in metrics if m.reject_reason != "unreadable"]
        # Pick sharpest frames across temporal bins
        target = min(max(min_keep, len(frame_paths) // 3), len(valid))
        bins = np.array_split(valid, target)
        kept_paths = []
        for b in bins:
            if len(b) == 0:
                continue
            best = max(b, key=lambda m: m.blur_score)
            best.kept = True
            best.reject_reason = ""
            kept_paths.append(best.path)

    logger.info("Quality filter kept %d / %d frames", len(kept_paths), len(frame_paths))
    return kept_paths, metrics

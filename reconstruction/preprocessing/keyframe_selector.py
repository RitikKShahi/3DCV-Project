"""Keyframe selection for maximum scene coverage with minimal redundancy."""

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from reconstruction.preprocessing.quality_filter import compute_blur_score

logger = logging.getLogger(__name__)


def select_keyframes(
    candidate_paths: list[Path],
    output_dir: Path,
    max_keyframes: int = 20,
) -> list[Path]:
    """Select up to `max_keyframes` sharp, temporally well-distributed keyframes.

    Copies selected keyframes into `output_dir` with clean sequential names
    (`keyframe_0001.jpg`, ...) so SfM operates on a clean, bounded directory.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not candidate_paths:
        raise ValueError("No candidate frames provided for keyframe selection")

    if len(candidate_paths) <= max_keyframes:
        selected = list(candidate_paths)
    else:
        # Divide timeline into `max_keyframes` uniform windows and pick the sharpest frame in each
        # weighted slightly toward visual diversity from the previously chosen keyframe.
        indices = np.array_split(np.arange(len(candidate_paths)), max_keyframes)
        selected = []
        prev_hist: np.ndarray | None = None

        for window in indices:
            best_path = candidate_paths[int(window[0])]
            best_score = -1.0
            best_hist = None

            for idx in window:
                p = candidate_paths[int(idx)]
                img = cv2.imread(str(p))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharpness = compute_blur_score(gray)
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
                hist /= (hist.sum() + 1e-6)

                diversity = 1.0
                if prev_hist is not None:
                    diversity = float(0.5 + np.sum(np.abs(hist - prev_hist)))

                score = sharpness * diversity
                if score > best_score:
                    best_score = score
                    best_path = p
                    best_hist = hist

            selected.append(best_path)
            if best_hist is not None:
                prev_hist = best_hist

    # Copy to keyframe directory with deterministic ordering
    keyframe_paths: list[Path] = []
    for i, src in enumerate(selected, start=1):
        dst = output_dir / f"keyframe_{i:04d}.jpg"
        shutil.copy2(src, dst)
        keyframe_paths.append(dst)

    logger.info("Selected %d keyframes in %s", len(keyframe_paths), output_dir)
    return keyframe_paths


"""SIFT feature extraction and FLANN matching for CPU OpenCV SfM."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ImageFeatures:
    """Extracted SIFT keypoints, descriptors, and image metadata."""

    name: str
    path: Path
    width: int
    height: int
    keypoints: np.ndarray     # [K, 2] float32 (x, y)
    descriptors: np.ndarray   # [K, 128] float32
    colors: np.ndarray        # [K, 3] float32 RGB in [0, 1]


def extract_sift_features(image_path: Path, max_features: int = 4096) -> ImageFeatures:
    """Extract SIFT keypoints, descriptors, and sampled RGB colors."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    sift = cv2.SIFT_create(nfeatures=max_features)
    kps, desc = sift.detectAndCompute(gray, None)

    if not kps or desc is None:
        return ImageFeatures(
            name=image_path.name,
            path=image_path,
            width=w,
            height=h,
            keypoints=np.zeros((0, 2), dtype=np.float32),
            descriptors=np.zeros((0, 128), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
        )

    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    colors = img_rgb[yi, xi]

    return ImageFeatures(
        name=image_path.name,
        path=image_path,
        width=w,
        height=h,
        keypoints=pts,
        descriptors=desc.astype(np.float32),
        colors=colors,
    )


def match_features(
    feat_a: ImageFeatures,
    feat_b: ImageFeatures,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match SIFT descriptors with FLANN + Lowe's ratio test + Fundamental RANSAC.

    Returns (idx_a, idx_b) arrays of verified inlier keypoint indices.
    """
    if len(feat_a.keypoints) < 8 or len(feat_b.keypoints) < 8:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50),
    )
    raw = flann.knnMatch(feat_a.descriptors, feat_b.descriptors, k=2)

    idx_a_list: list[int] = []
    idx_b_list: list[int] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            idx_a_list.append(m.queryIdx)
            idx_b_list.append(m.trainIdx)

    if len(idx_a_list) < 8:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    idx_a = np.asarray(idx_a_list, dtype=np.int64)
    idx_b = np.asarray(idx_b_list, dtype=np.int64)

    pts_a = feat_a.keypoints[idx_a]
    pts_b = feat_b.keypoints[idx_b]
    _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, ransac_thresh, 0.99)
    if mask is None:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    inliers = mask.ravel().astype(bool)
    return idx_a[inliers], idx_b[inliers]


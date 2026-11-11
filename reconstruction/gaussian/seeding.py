"""Dense epipolar-verified point seeding with Statistical Outlier Removal (SOR).

Bridges sparse SIFT points to a clean, noise-free colored Gaussian initialization
without creating far-away depth-explosion floaters.
"""

import logging
import re

import cv2
import numpy as np
from scipy.spatial import cKDTree

from reconstruction.sfm.triangulation import triangulate_two_view
from reconstruction.sfm.types import CameraPose

logger = logging.getLogger(__name__)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _fundamental_from_poses(
    K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
    K2: np.ndarray, R2: np.ndarray, t2: np.ndarray,
) -> np.ndarray:
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    E = _skew(t_rel) @ R_rel
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def _frame_index(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def _statistical_outlier_filter(
    points: np.ndarray,
    colors: np.ndarray,
    k_neighbors: int = 8,
    mad_thresh: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove isolated 3D floater noise while preserving distant outdoor background structures."""
    n = len(points)
    if n < k_neighbors + 2:
        return points, colors

    # 1. Remove extreme far-field spikes beyond 14x median distance (preserves trees/buildings)
    med_center = np.median(points, axis=0)
    center_dists = np.linalg.norm(points - med_center[None, :], axis=1)
    med_rad = max(0.25, float(np.median(center_dists)))
    in_bounds = center_dists <= (14.0 * med_rad)
    points = points[in_bounds]
    colors = colors[in_bounds]

    if len(points) < k_neighbors + 2:
        return points, colors

    # 2. Statistical Outlier Removal on k-nearest-neighbor mean distances
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k_neighbors + 1)
    knn_mean = np.mean(dists[:, 1:], axis=1)
    med_knn = float(np.median(knn_mean))
    mad = float(np.median(np.abs(knn_mean - med_knn))) + 1e-6
    keep = knn_mean <= (med_knn + mad_thresh * 1.4826 * mad)

    if int(np.sum(keep)) >= 16:
        points = points[keep]
        colors = colors[keep]

    return points, colors


def densify_sparse_points_from_poses(
    poses: dict[str, CameraPose],
    max_new_points: int = 3000,
    grid_step: int = 10,
    epipolar_thresh_px: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean, geometrically verified 3D surface points from calibrated camera pairs.

    Enforces temporal proximity, camera viewing-angle overlap, tight forward-backward
    optical-flow consistency, epipolar verification, and 3D Statistical Outlier Removal.
    """
    pose_list = sorted(poses.values(), key=lambda p: p.image_name)
    if len(pose_list) < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Load grayscale & RGB images at their pose resolution
    imgs_gray: dict[str, np.ndarray] = {}
    imgs_rgb: dict[str, np.ndarray] = {}
    for p in pose_list:
        bgr = cv2.imread(p.image_path)
        if bgr is None:
            continue
        if bgr.shape[1] != p.width or bgr.shape[0] != p.height:
            bgr = cv2.resize(bgr, (p.width, p.height), interpolation=cv2.INTER_AREA)
        imgs_gray[p.image_name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        imgs_rgb[p.image_name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Pair only nearby temporal frames with modest view-angle change (<= 18 deg)
    pairs: list[tuple[int, int]] = []
    for i in range(len(pose_list)):
        idx_i = _frame_index(pose_list[i].image_name)
        z_i = pose_list[i].R[2, :]  # camera optical axis in world space
        for j in range(i + 1, min(len(pose_list), i + 3)):
            idx_j = _frame_index(pose_list[j].image_name)
            if abs(idx_j - idx_i) > 3:
                continue
            z_j = pose_list[j].R[2, :]
            cos_ang = float(np.clip(np.dot(z_i, z_j), -1.0, 1.0))
            if np.degrees(np.arccos(cos_ang)) <= 18.0:
                pairs.append((i, j))

    if not pairs:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    per_pair_budget = max(120, max_new_points // max(1, len(pairs)))
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for i, j in pairs:
        p1 = pose_list[i]
        p2 = pose_list[j]
        if p1.image_name not in imgs_gray or p2.image_name not in imgs_gray:
            continue

        baseline = float(np.linalg.norm(p1.camera_center - p2.camera_center))
        if baseline < 1e-3:
            continue

        g1 = imgs_gray[p1.image_name]
        g2 = imgs_gray[p2.image_name]
        rgb1 = imgs_rgb[p1.image_name]
        h1, w1 = g1.shape
        h2, w2 = g2.shape

        # Pad to common canvas if images have different dimensions (e.g. mixed portrait/landscape photos)
        if (h1, w1) != (h2, w2):
            hc, wc = max(h1, h2), max(w1, w2)
            c1 = np.zeros((hc, wc), dtype=g1.dtype)
            c2 = np.zeros((hc, wc), dtype=g2.dtype)
            c1[:h1, :w1] = g1
            c2[:h2, :w2] = g2
            g1_flow, g2_flow = c1, c2
        else:
            g1_flow, g2_flow = g1, g2

        ys = np.arange(grid_step // 2, h1 - grid_step // 2, grid_step, dtype=np.float32)
        xs = np.arange(grid_step // 2, w1 - grid_step // 2, grid_step, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        corners = cv2.goodFeaturesToTrack(g1, maxCorners=450, qualityLevel=0.008, minDistance=6)
        if corners is not None and len(corners) > 0:
            pts1 = np.vstack([corners.reshape(-1, 2).astype(np.float32), grid_pts])
        else:
            pts1 = grid_pts

        if len(pts1) == 0:
            continue

        p1_in = pts1.reshape(-1, 1, 2)
        p2_tracked, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
            g1_flow, g2_flow, p1_in, None, winSize=(21, 21), maxLevel=3
        )
        p1_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            g2_flow, g1_flow, p2_tracked, None, winSize=(21, 21), maxLevel=3
        )

        fb_err = np.linalg.norm(p1_in.reshape(-1, 2) - p1_back.reshape(-1, 2), axis=1)
        pts2 = p2_tracked.reshape(-1, 2)
        ok = (
            (st_fwd.ravel() == 1)
            & (st_bwd.ravel() == 1)
            & (fb_err < 1.2)
            & (pts2[:, 0] >= 1)
            & (pts2[:, 0] < w2 - 1)
            & (pts2[:, 1] >= 1)
            & (pts2[:, 1] < h2 - 1)
        )
        if not np.any(ok):
            continue

        pts1_ok = pts1[ok]
        pts2_ok = pts2[ok]

        F = _fundamental_from_poses(p1.K, p1.R, p1.t, p2.K, p2.R, p2.t)
        homog1 = np.hstack([pts1_ok, np.ones((len(pts1_ok), 1))])
        homog2 = np.hstack([pts2_ok, np.ones((len(pts2_ok), 1))])
        lines2 = (F @ homog1.T).T
        denom = np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2) + 1e-8
        epi_err = np.abs(np.sum(lines2 * homog2, axis=1)) / denom
        epi_ok = epi_err < epipolar_thresh_px

        if not np.any(epi_ok):
            continue

        pts1_epi = pts1_ok[epi_ok]
        pts2_epi = pts2_ok[epi_ok]

        tri_pts, tri_mask = triangulate_two_view(
            p1.K, p1.R, p1.t, p2.R, p2.t,
            pts1_epi, pts2_epi,
            max_reproj_error=1.8,
            min_parallax_deg=0.6,
        )
        if len(tri_pts) == 0:
            continue

        valid_uv = pts1_epi[tri_mask]
        xi = np.clip(np.round(valid_uv[:, 0]).astype(int), 0, w1 - 1)
        yi = np.clip(np.round(valid_uv[:, 1]).astype(int), 0, h1 - 1)
        cols = rgb1[yi, xi]

        if len(tri_pts) > per_pair_budget:
            rng = np.random.default_rng(i * 31 + j)
            sel = rng.choice(len(tri_pts), size=per_pair_budget, replace=False)
            tri_pts = tri_pts[sel]
            cols = cols[sel]

        all_pts.append(tri_pts)
        all_cols.append(cols)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts_merged = np.vstack(all_pts).astype(np.float32)
    cols_merged = np.vstack(all_cols).astype(np.float32)

    # Filter out 3D depth-explosion floaters and isolated noise points
    pts_merged, cols_merged = _statistical_outlier_filter(pts_merged, cols_merged)

    if len(pts_merged) > max_new_points:
        rng = np.random.default_rng(99)
        sel = rng.choice(len(pts_merged), size=max_new_points, replace=False)
        pts_merged = pts_merged[sel]
        cols_merged = cols_merged[sel]

    logger.info("Clean epipolar seeding generated %d filtered 3D points across %d pairs", len(pts_merged), len(pairs))
    return pts_merged, cols_merged

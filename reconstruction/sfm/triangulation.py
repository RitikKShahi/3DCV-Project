"""Multi-view point triangulation with cheirality, parallax, and reprojection filtering."""

import cv2
import numpy as np


def triangulate_two_view(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    max_reproj_error: float = 3.0,
    min_parallax_deg: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate 3D points from two calibrated camera views.

    Returns (points_3d [M, 3], valid_mask [N] bool).
    """
    if len(pts1) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=bool)

    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

    homog = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
    w = homog[3, :]
    valid_w = np.abs(w) > 1e-8
    pts3d = np.zeros((len(pts1), 3), dtype=np.float64)
    pts3d[valid_w] = (homog[:3, valid_w] / w[valid_w]).T

    # Cheirality check (positive depth in both cameras)
    cam1 = (R1 @ pts3d.T + t1.reshape(3, 1)).T
    cam2 = (R2 @ pts3d.T + t2.reshape(3, 1)).T
    valid = valid_w & (cam1[:, 2] > 0.05) & (cam2[:, 2] > 0.05)

    # Reprojection error filter
    proj1 = (P1 @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T
    proj2 = (P2 @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T
    uv1 = proj1[:, :2] / np.where(np.abs(proj1[:, 2:3]) > 1e-8, proj1[:, 2:3], 1e-8)
    uv2 = proj2[:, :2] / np.where(np.abs(proj2[:, 2:3]) > 1e-8, proj2[:, 2:3], 1e-8)
    err1 = np.linalg.norm(uv1 - pts1, axis=1)
    err2 = np.linalg.norm(uv2 - pts2, axis=1)
    valid &= (err1 < max_reproj_error) & (err2 < max_reproj_error)

    # Parallax angle filter to reject unstable near-collinear points
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2
    ray1 = pts3d - c1[None, :]
    ray2 = pts3d - c2[None, :]
    n1 = np.linalg.norm(ray1, axis=1, keepdims=True) + 1e-8
    n2 = np.linalg.norm(ray2, axis=1, keepdims=True) + 1e-8
    cos_ang = np.clip(np.sum((ray1 / n1) * (ray2 / n2), axis=1), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cos_ang))
    valid &= ang_deg >= min_parallax_deg

    # Reject extreme outliers far from camera median
    if np.any(valid):
        dists = np.linalg.norm(pts3d[valid] - c1[None, :], axis=1)
        med = np.median(dists)
        valid &= np.linalg.norm(pts3d - c1[None, :], axis=1) < max(50.0, med * 10.0)

    return pts3d[valid].astype(np.float32), valid


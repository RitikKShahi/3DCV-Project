"""Relative and absolute camera pose estimation (Essential matrix & PnP)."""

import math
import cv2
import numpy as np


def default_intrinsics(
    width: int,
    height: int,
    fov_factor: float = 1.0,
    fov_deg: float = 0.0,
) -> np.ndarray:
    """Estimate pinhole intrinsics K from image dimensions or explicit FOV in degrees."""
    if fov_deg > 0.0:
        half_rad = math.radians(max(20.0, min(140.0, fov_deg))) * 0.5
        focal = (0.5 * float(max(width, height))) / math.tan(half_rad)
    else:
        focal = float(max(width, height)) * fov_factor
    return np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_relative_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    ransac_prob: float = 0.999,
    ransac_thresh: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate relative pose (R, t) from view 1 to view 2 via Essential matrix.

    Returns (R, t, inlier_mask_bool).
    """
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=ransac_prob, threshold=ransac_thresh
    )
    if E is None or mask is None:
        raise RuntimeError("Essential matrix estimation failed")

    # Handle case where findEssentialMat returns multiple stacked solutions
    if E.shape[0] > 3:
        E = E[:3, :3]

    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inliers = pose_mask.ravel() > 0
    return R.astype(np.float64), t.reshape(3).astype(np.float64), inliers


def register_pose_pnp(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    reproj_thresh: float = 4.0,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Register a new camera into the existing world coordinate system via PnP-RANSAC.

    Returns (success, R, t, inlier_indices).
    """
    if len(pts3d) < 6:
        return False, np.eye(3), np.zeros(3), np.zeros(0, dtype=np.int64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3d.astype(np.float64),
        imagePoints=pts2d.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=None,
        iterationsCount=500,
        reprojectionError=reproj_thresh,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok or inliers is None or len(inliers) < 6:
        return False, np.eye(3), np.zeros(3), np.zeros(0, dtype=np.int64)

    inlier_idx = inliers.ravel().astype(np.int64)
    # Refine pose with Levenberg-Marquardt on inliers
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            pts3d[inlier_idx].astype(np.float64),
            pts2d[inlier_idx].astype(np.float64),
            K.astype(np.float64),
            None,
            rvec,
            tvec,
        )
    except Exception:
        pass

    R, _ = cv2.Rodrigues(rvec)
    return True, R.astype(np.float64), tvec.reshape(3).astype(np.float64), inlier_idx


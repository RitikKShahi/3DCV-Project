"""Sparse CPU Bundle Adjustment using scipy.optimize.least_squares."""

import logging

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

logger = logging.getLogger(__name__)


def run_bundle_adjustment(
    K: np.ndarray,
    camera_poses: list[tuple[np.ndarray, np.ndarray]],
    points_3d: np.ndarray,
    observations: list[tuple[int, int, np.ndarray]],
    fix_first_n_cameras: int = 1,
    max_nfev: int = 50,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, float]:
    """Refine camera poses (R, t) and 3D points via sparse Levenberg-Marquardt/TRF BA.

    Args:
        K: [3, 3] intrinsics matrix.
        camera_poses: List of (R [3,3], t [3]) world-to-camera poses.
        points_3d: [M, 3] 3D points.
        observations: List of (cam_idx, pt_idx, uv [2]).
        fix_first_n_cameras: Number of leading cameras held fixed to preserve coordinate frame.
        max_nfev: Max function evaluations for CPU budget.

    Returns:
        (refined_poses, refined_points_3d, mean_reproj_error_px)
    """
    n_cams = len(camera_poses)
    n_pts = len(points_3d)
    n_obs = len(observations)

    if n_cams < 2 or n_pts < 6 or n_obs < 12:
        return camera_poses, points_3d, 0.0

    fixed_cams = min(fix_first_n_cameras, n_cams)
    opt_cams = n_cams - fixed_cams
    if opt_cams <= 0 and n_pts == 0:
        return camera_poses, points_3d, 0.0

    # Pack optimizable camera parameters (rvec[3], t[3])
    cam_params_fixed = []
    cam_params_opt = []
    for idx, (R, t) in enumerate(camera_poses):
        rvec, _ = cv2.Rodrigues(R.astype(np.float64))
        vec = np.concatenate([rvec.ravel(), t.astype(np.float64).ravel()])
        if idx < fixed_cams:
            cam_params_fixed.append(vec)
        else:
            cam_params_opt.append(vec)

    x_cam = np.concatenate(cam_params_opt) if cam_params_opt else np.zeros(0, dtype=np.float64)
    x_pts = points_3d.astype(np.float64).ravel()
    x0 = np.concatenate([x_cam, x_pts])

    cam_indices = np.array([o[0] for o in observations], dtype=np.int64)
    pt_indices = np.array([o[1] for o in observations], dtype=np.int64)
    obs_uv = np.array([o[2] for o in observations], dtype=np.float64)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Build sparse Jacobian pattern
    n_cam_vars = opt_cams * 6
    n_vars = n_cam_vars + n_pts * 3
    sparsity = lil_matrix((2 * n_obs, n_vars), dtype=int)
    obs_range = np.arange(n_obs)

    for i, c_idx in enumerate(cam_indices):
        if c_idx >= fixed_cams:
            base = (c_idx - fixed_cams) * 6
            sparsity[2 * i, base : base + 6] = 1
            sparsity[2 * i + 1, base : base + 6] = 1

    pt_base = n_cam_vars + pt_indices * 3
    for s in range(3):
        sparsity[2 * obs_range, pt_base + s] = 1
        sparsity[2 * obs_range + 1, pt_base + s] = 1

    fixed_arr = np.array(cam_params_fixed, dtype=np.float64) if cam_params_fixed else np.zeros((0, 6))

    def _rodrigues_rotate(pts: np.ndarray, rvecs: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
        small = theta[:, 0] < 1e-8
        k = np.where(small[:, None], 0.0, rvecs / np.where(small[:, None], 1.0, theta))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cross = np.cross(k, pts)
        dot = np.sum(k * pts, axis=1, keepdims=True)
        rot = pts * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)
        # First-order approximation for near-zero angles
        rot[small] = pts[small] + np.cross(rvecs[small], pts[small])
        return rot

    def residuals(params: np.ndarray) -> np.ndarray:
        opt_c = params[:n_cam_vars].reshape(opt_cams, 6) if opt_cams > 0 else np.zeros((0, 6))
        all_c = np.vstack([fixed_arr, opt_c]) if fixed_cams > 0 else opt_c
        pts = params[n_cam_vars:].reshape(n_pts, 3)

        c_per_obs = all_c[cam_indices]
        p_per_obs = pts[pt_indices]

        cam_pts = _rodrigues_rotate(p_per_obs, c_per_obs[:, :3]) + c_per_obs[:, 3:6]
        z = np.where(np.abs(cam_pts[:, 2:3]) < 1e-6, 1e-6, cam_pts[:, 2:3])
        u = fx * (cam_pts[:, 0:1] / z) + cx
        v = fy * (cam_pts[:, 1:2] / z) + cy
        proj = np.hstack([u, v])
        return (proj - obs_uv).ravel()

    res = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        loss="huber",
        f_scale=2.0,
        max_nfev=max_nfev,
        x_scale="jac",
    )

    opt_c = res.x[:n_cam_vars].reshape(opt_cams, 6) if opt_cams > 0 else np.zeros((0, 6))
    all_c = np.vstack([fixed_arr, opt_c]) if fixed_cams > 0 else opt_c
    refined_poses: list[tuple[np.ndarray, np.ndarray]] = []
    for vec in all_c:
        R, _ = cv2.Rodrigues(vec[:3])
        refined_poses.append((R.astype(np.float64), vec[3:6].astype(np.float64)))

    refined_pts = res.x[n_cam_vars:].reshape(n_pts, 3).astype(np.float32)
    final_res = res.fun.reshape(-1, 2)
    mean_err = float(np.mean(np.linalg.norm(final_res, axis=1)))
    logger.info("SciPy BA finished (%d nfev): mean reproj error = %.3f px", res.nfev, mean_err)
    return refined_poses, refined_pts, mean_err


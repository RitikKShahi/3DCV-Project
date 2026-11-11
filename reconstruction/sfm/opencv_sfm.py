"""Incremental CPU-only SfM fallback implemented purely with OpenCV + SciPy."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from reconstruction.sfm.bundle_adjustment import run_bundle_adjustment
from reconstruction.sfm.feature_matching import ImageFeatures, extract_sift_features, match_features
from reconstruction.sfm.pose_estimation import default_intrinsics, estimate_relative_pose, register_pose_pnp
from reconstruction.sfm.triangulation import triangulate_two_view
from reconstruction.sfm.types import CameraPose, SfMBatchResult, SparsePointCloud

logger = logging.getLogger(__name__)


class OpenCvSfM:
    """Pure OpenCV + SciPy incremental Structure-from-Motion engine."""

    def __init__(self, max_features: int = 4096, min_inliers_pnp: int = 12, camera_fov_deg: float = 0.0):
        self.max_features = max_features
        self.min_inliers_pnp = min_inliers_pnp
        self.camera_fov_deg = camera_fov_deg

        self.K: Optional[np.ndarray] = None
        self.features: dict[str, ImageFeatures] = {}
        self.poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.pose_order: list[str] = []

        self.points3d: list[np.ndarray] = []
        self.colors3d: list[np.ndarray] = []
        # Map (image_name, kp_idx) -> point3d_idx
        self.kp_to_pt3d: dict[tuple[str, int], int] = {}
        # Observations: (image_name, kp_idx, pt3d_idx)
        self.observations: list[tuple[str, int, int]] = []
        self.known_point_count = 0

    def process_batch(self, frame_paths: list[Path]) -> SfMBatchResult:
        """Process initial or incremental batch of frames into consistent world frame."""
        new_feats: list[ImageFeatures] = []
        for p in frame_paths:
            p = Path(p)
            if p.name not in self.features:
                feat = extract_sift_features(p, max_features=self.max_features)
                self.features[p.name] = feat
                new_feats.append(feat)
                if self.K is None:
                    self.K = default_intrinsics(feat.width, feat.height, fov_deg=self.camera_fov_deg)

        assert self.K is not None, "No frames provided to OpenCvSfM"
        prev_names = set(self.poses.keys())
        prev_pt_count = len(self.points3d)

        if not self.poses:
            self._initialize_first_batch(new_feats)
        else:
            for feat in new_feats:
                self._register_and_triangulate(feat)

        mean_err = self._bundle_adjust(fix_existing=len(prev_names) > 0, prev_count=len(prev_names))

        all_pose_objs: dict[str, CameraPose] = {}
        for name in self.pose_order:
            R, t = self.poses[name]
            f = self.features[name]
            all_pose_objs[name] = CameraPose(
                image_name=name,
                image_path=str(f.path),
                width=f.width,
                height=f.height,
                K=self.K.copy(),
                R=R.copy(),
                t=t.copy(),
            )

        new_pose_objs = {k: v for k, v in all_pose_objs.items() if k not in prev_names}
        pts_arr = np.asarray(self.points3d, dtype=np.float32) if self.points3d else np.zeros((0, 3), dtype=np.float32)
        cols_arr = np.asarray(self.colors3d, dtype=np.float32) if self.colors3d else np.zeros((0, 3), dtype=np.float32)
        pids = list(range(len(pts_arr)))

        new_pts_arr = pts_arr[prev_pt_count:]
        new_cols_arr = cols_arr[prev_pt_count:]
        new_pids = pids[prev_pt_count:]

        return SfMBatchResult(
            all_poses=all_pose_objs,
            new_poses=new_pose_objs,
            all_points=SparsePointCloud(points=pts_arr, colors=cols_arr, point_ids=pids),
            new_points=SparsePointCloud(points=new_pts_arr, colors=new_cols_arr, point_ids=new_pids),
            backend_used="opencv",
            mean_reprojection_error=mean_err,
        )

    def _initialize_first_batch(self, batch_feats: list[ImageFeatures]) -> None:
        if len(batch_feats) < 2:
            raise RuntimeError("Need at least 2 frames to initialize OpenCvSfM")

        assert self.K is not None
        # Choose best initial pair (0, j) with enough matches and non-trivial baseline
        best_pair = None
        best_score = -1.0
        for i in range(len(batch_feats) - 1):
            for j in range(i + 1, min(len(batch_feats), i + 4)):
                ia, ib = match_features(batch_feats[i], batch_feats[j])
                if len(ia) < 25:
                    continue
                disp = np.median(
                    np.linalg.norm(batch_feats[i].keypoints[ia] - batch_feats[j].keypoints[ib], axis=1)
                )
                score = len(ia) * min(disp, 80.0)
                if disp > 8.0 and score > best_score:
                    best_score = score
                    best_pair = (i, j, ia, ib)

        if best_pair is None:
            # Fallback to (0, 1)
            ia, ib = match_features(batch_feats[0], batch_feats[1])
            if len(ia) < 12:
                raise RuntimeError("Insufficient feature matches in initial frames for OpenCV SfM")
            best_pair = (0, 1, ia, ib)

        i0, i1, idx0, idx1 = best_pair
        f0, f1 = batch_feats[i0], batch_feats[i1]
        pts0 = f0.keypoints[idx0]
        pts1 = f1.keypoints[idx1]

        R_rel, t_rel, E_mask = estimate_relative_pose(pts0, pts1, self.K)
        idx0, idx1 = idx0[E_mask], idx1[E_mask]
        pts0, pts1 = pts0[E_mask], pts1[E_mask]

        R0, t0 = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        R1, t1 = R_rel, t_rel

        tri_pts, tri_mask = triangulate_two_view(self.K, R0, t0, R1, t1, pts0, pts1)
        if len(tri_pts) < 10:
            raise RuntimeError(f"Initial two-view triangulation produced only {len(tri_pts)} points")

        self.poses[f0.name] = (R0, t0)
        self.poses[f1.name] = (R1, t1)
        self.pose_order.extend([f0.name, f1.name])

        valid_idx0 = idx0[tri_mask]
        valid_idx1 = idx1[tri_mask]
        for k, p3 in enumerate(tri_pts):
            pid = len(self.points3d)
            self.points3d.append(p3)
            c = 0.5 * (f0.colors[valid_idx0[k]] + f1.colors[valid_idx1[k]])
            self.colors3d.append(c)
            self.kp_to_pt3d[(f0.name, int(valid_idx0[k]))] = pid
            self.kp_to_pt3d[(f1.name, int(valid_idx1[k]))] = pid
            self.observations.append((f0.name, int(valid_idx0[k]), pid))
            self.observations.append((f1.name, int(valid_idx1[k]), pid))

        # Register remaining frames in batch 0
        for idx, feat in enumerate(batch_feats):
            if idx in (i0, i1):
                continue
            self._register_and_triangulate(feat)

    def _register_and_triangulate(self, feat: ImageFeatures) -> bool:
        assert self.K is not None
        pts3d_corr: list[np.ndarray] = []
        pts2d_corr: list[np.ndarray] = []
        kp_pids: list[tuple[int, int]] = []
        cached_matches: list[tuple[str, np.ndarray, np.ndarray]] = []

        # Match against up to 6 most recent registered frames
        recent_names = self.pose_order[-6:]
        for ref_name in recent_names:
            ref_feat = self.features[ref_name]
            ia, ib = match_features(ref_feat, feat)
            if len(ia) == 0:
                continue
            cached_matches.append((ref_name, ia, ib))
            for ka, kb in zip(ia, ib):
                pid = self.kp_to_pt3d.get((ref_name, int(ka)))
                if pid is not None:
                    pts3d_corr.append(self.points3d[pid])
                    pts2d_corr.append(feat.keypoints[int(kb)])
                    kp_pids.append((int(kb), pid))

        if len(pts3d_corr) < self.min_inliers_pnp:
            logger.warning("Skipping %s: only %d 2D-3D correspondences", feat.name, len(pts3d_corr))
            return False

        ok, R_new, t_new, inlier_idx = register_pose_pnp(
            np.asarray(pts3d_corr), np.asarray(pts2d_corr), self.K
        )
        if not ok or len(inlier_idx) < self.min_inliers_pnp:
            logger.warning("PnP failed for %s (%d inliers)", feat.name, len(inlier_idx))
            return False

        self.poses[feat.name] = (R_new, t_new)
        self.pose_order.append(feat.name)

        # Record 2D-3D observations for PnP inliers
        for idx in inlier_idx:
            kb, pid = kp_pids[int(idx)]
            if (feat.name, kb) not in self.kp_to_pt3d:
                self.kp_to_pt3d[(feat.name, kb)] = pid
                self.observations.append((feat.name, kb, pid))

        # Triangulate new 3D points from unmatched correspondences
        for ref_name, ia, ib in cached_matches:
            ref_feat = self.features[ref_name]
            R_ref, t_ref = self.poses[ref_name]
            unmatched_a = []
            unmatched_b = []
            for ka, kb in zip(ia, ib):
                if (ref_name, int(ka)) not in self.kp_to_pt3d and (feat.name, int(kb)) not in self.kp_to_pt3d:
                    unmatched_a.append(int(ka))
                    unmatched_b.append(int(kb))
            if len(unmatched_a) < 5:
                continue

            ua = np.asarray(unmatched_a, dtype=np.int64)
            ub = np.asarray(unmatched_b, dtype=np.int64)
            tri_pts, tri_mask = triangulate_two_view(
                self.K, R_ref, t_ref, R_new, t_new,
                ref_feat.keypoints[ua], feat.keypoints[ub],
            )
            va = ua[tri_mask]
            vb = ub[tri_mask]
            for k, p3 in enumerate(tri_pts):
                pid = len(self.points3d)
                self.points3d.append(p3)
                c = 0.5 * (ref_feat.colors[va[k]] + feat.colors[vb[k]])
                self.colors3d.append(c)
                self.kp_to_pt3d[(ref_name, int(va[k]))] = pid
                self.kp_to_pt3d[(feat.name, int(vb[k]))] = pid
                self.observations.append((ref_name, int(va[k]), pid))
                self.observations.append((feat.name, int(vb[k]), pid))

        logger.info("Registered %s via PnP (%d inliers), total points=%d", feat.name, len(inlier_idx), len(self.points3d))
        return True

    def _bundle_adjust(self, fix_existing: bool, prev_count: int) -> float:
        if self.K is None or len(self.pose_order) < 2 or len(self.points3d) < 10:
            return 0.0

        name_to_cam_idx = {name: idx for idx, name in enumerate(self.pose_order)}
        cam_list = [self.poses[name] for name in self.pose_order]
        pts_arr = np.asarray(self.points3d, dtype=np.float32)

        ba_obs = [
            (name_to_cam_idx[name], pid, self.features[name].keypoints[kp_idx])
            for name, kp_idx, pid in self.observations
            if name in name_to_cam_idx
        ]
        fix_n = max(1, prev_count) if fix_existing else 1
        refined_poses, refined_pts, mean_err = run_bundle_adjustment(
            self.K, cam_list, pts_arr, ba_obs, fix_first_n_cameras=fix_n, max_nfev=35
        )
        for idx, name in enumerate(self.pose_order):
            self.poses[name] = refined_poses[idx]
        self.points3d = [refined_pts[i] for i in range(len(refined_pts))]
        return mean_err


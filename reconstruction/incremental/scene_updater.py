"""Incremental scene updater: preserves coordinate frame and refines Gaussians."""

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from reconstruction.config import ReconstructionConfig
from reconstruction.export.representations import generate_and_cache_representations
from reconstruction.export.splat_exporter import export_splat
from reconstruction.gaussian.model import GaussianModel
from reconstruction.gaussian.optimizer import GaussianOptimizer, TrainingView, load_training_views
from reconstruction.gaussian.rasterizer import render_gaussians
from reconstruction.gaussian.seeding import densify_sparse_points_from_poses
from reconstruction.sfm.types import SfMBatchResult

logger = logging.getLogger(__name__)


class IncrementalSceneUpdater:
    """Maintains a single persistent GaussianModel updated across frame batches."""

    def __init__(self, model: GaussianModel, config: ReconstructionConfig):
        self.model = model
        self.config = config
        self.optimizer = GaussianOptimizer(model, config)

    def apply_sfm_batch(
        self,
        sfm_res: SfMBatchResult,
        batch_index: int,
        preview_dir: Path,
        progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> dict:
        """Update Gaussian scene from an initial or incremental SfM batch and generate preview.

        1. On initial batch: initialize Gaussians from sparse points.
        2. On subsequent batches: add only novel sparse points into existing scene.
        3. Optimize existing + newly added Gaussians jointly on all registered views
           (prioritizing new views + replay of previous views).
        4. Export intermediate `.splat` preview and rendered comparison image.
        """
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

        # Augment sparse SfM points with dense epipolar-verified optical-flow 3D surface points
        seeded_pts, seeded_cols = densify_sparse_points_from_poses(
            sfm_res.all_poses,
            max_new_points=max(1600, int(self.config.max_gaussians * 0.7)),
            grid_step=10,
        )

        if self.model.num_gaussians == 0:
            if len(seeded_pts) > 0:
                init_pts = np.vstack([sfm_res.all_points.points, seeded_pts])
                init_cols = np.vstack([sfm_res.all_points.colors, seeded_cols])
            else:
                init_pts = sfm_res.all_points.points
                init_cols = sfm_res.all_points.colors

            self.model.init_from_sparse_points(
                init_pts,
                init_cols,
                max_gaussians=int(self.config.max_gaussians * 0.75),
            )
            added = self.model.num_gaussians
        else:
            if len(seeded_pts) > 0 and len(sfm_res.new_poses) > 0:
                comb_pts = np.vstack([sfm_res.new_points.points, seeded_pts])
                comb_cols = np.vstack([sfm_res.new_points.colors, seeded_cols])
            else:
                comb_pts = sfm_res.new_points.points
                comb_cols = sfm_res.new_points.colors

            added = self.model.add_new_points(
                comb_pts,
                comb_cols,
                max_gaussians=self.config.max_gaussians,
            )

        new_names = set(sfm_res.new_poses.keys()) if batch_index > 0 else set(sfm_res.all_poses.keys())
        views: list[TrainingView] = load_training_views(
            sfm_res.all_poses,
            new_pose_names=new_names,
            device=self.model.device,
        )

        metrics = self.optimizer.train_steps(
            views=views,
            num_iterations=self.config.iterations_per_batch,
            allow_densify=True,
            progress_cb=progress_cb,
        )

        # Export intermediate .splat preview + derived Geometry / Mesh / Triangle Splatting views
        preview_splat = preview_dir / f"preview_batch_{batch_index}.splat"
        latest_splat = preview_dir / "latest_preview.splat"
        export_splat(self.model, preview_splat)
        latest_splat.write_bytes(preview_splat.read_bytes())
        try:
            generate_and_cache_representations(
                preview_dir=preview_dir,
                model=self.model,
                poses=sfm_res.all_poses,
                mesh_grid_res=40,
            )
        except Exception as exc:
            logger.warning("Failed generating derived preview representations: %s", exc)

        # Render a thumbnail image from the newest registered camera for dashboard display
        preview_img_path = preview_dir / f"render_batch_{batch_index}.jpg"
        if views:
            ref_view = next((v for v in views if v.is_new), views[-1])
            with torch.no_grad():
                rendered = render_gaussians(
                    self.model,
                    ref_view.camera,
                    bg_color=self.config.background_color,
                    tile_size=self.config.tile_size,
                )["image"]
            img_u8 = (rendered.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            cv2.imwrite(str(preview_img_path), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

        return {
            "batch_index": batch_index,
            "num_cameras": len(sfm_res.all_poses),
            "new_cameras": len(sfm_res.new_poses),
            "num_gaussians": self.model.num_gaussians,
            "gaussians_added": added,
            "sfm_backend": sfm_res.backend_used,
            "sfm_reproj_error": sfm_res.mean_reprojection_error,
            "preview_splat": str(preview_splat),
            "preview_image": str(preview_img_path) if preview_img_path.exists() else None,
            **metrics,
        }

    def final_refinement(
        self,
        sfm_res: SfMBatchResult,
        progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> dict:
        """Run final global refinement across all registered cameras and prune floaters."""
        if self.config.final_iterations <= 0:
            return {"final_loss": 0.0, "mean_psnr": 0.0, "iterations_run": 0}
        from reconstruction.gaussian.densification import densify_and_prune
        from reconstruction.gaussian.optimizer import compute_scene_extent

        views = load_training_views(sfm_res.all_poses, new_pose_names=None, device=self.model.device)
        metrics = self.optimizer.train_steps(
            views=views,
            num_iterations=self.config.final_iterations,
            allow_densify=False,
            progress_cb=progress_cb,
        )
        # Final floater & oversized-Gaussian cleanup before export (no new cloning/splitting)
        densify_and_prune(
            self.model,
            max_gaussians=self.config.max_gaussians,
            grad_threshold=1e9,
            min_opacity=self.config.prune_opacity_threshold,
            scene_extent=compute_scene_extent(views),
        )
        return metrics


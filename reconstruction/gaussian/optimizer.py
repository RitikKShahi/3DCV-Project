"""CPU Gaussian optimization loop with adaptive LR scheduling and early stopping."""

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from reconstruction.config import ReconstructionConfig
from reconstruction.gaussian.densification import densify_and_prune
from reconstruction.gaussian.losses import photometric_loss, psnr
from reconstruction.gaussian.model import GaussianModel
from reconstruction.gaussian.rasterizer import Camera, render_gaussians
from reconstruction.sfm.types import CameraPose

logger = logging.getLogger(__name__)


@dataclass
class TrainingView:
    """Loaded ground-truth image paired with its differentiable Camera."""

    camera: Camera
    target_image: torch.Tensor  # [H, W, 3] float32 in [0, 1]
    is_new: bool = True


def load_training_views(
    poses: dict[str, CameraPose],
    new_pose_names: set[str] | None = None,
    device: torch.device | str = "cpu",
) -> list[TrainingView]:
    """Load target RGB images and convert CameraPoses into PyTorch Cameras."""
    views: list[TrainingView] = []
    for name, pose in poses.items():
        img_bgr = cv2.imread(pose.image_path)
        if img_bgr is None:
            logger.warning("Could not load training view image %s", pose.image_path)
            continue
        h, w = img_bgr.shape[:2]
        if w != pose.width or h != pose.height:
            img_bgr = cv2.resize(img_bgr, (pose.width, pose.height), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cam = Camera(
            image_name=name,
            width=pose.width,
            height=pose.height,
            fx=pose.fx,
            fy=pose.fy,
            cx=pose.cx,
            cy=pose.cy,
            R=torch.tensor(pose.R, dtype=torch.float32, device=device),
            t=torch.tensor(pose.t, dtype=torch.float32, device=device),
        )
        views.append(
            TrainingView(
                camera=cam,
                target_image=torch.tensor(img_rgb, dtype=torch.float32, device=device),
                is_new=(True if new_pose_names is None else name in new_pose_names),
            )
        )
    return views


def compute_scene_extent(views: list[TrainingView]) -> float:
    """Estimate spatial extent from camera centers for densification scaling."""
    if len(views) < 2:
        return 1.0
    centers = torch.stack([v.camera.center_world for v in views], dim=0)
    center_mean = centers.mean(dim=0, keepdim=True)
    dists = torch.norm(centers - center_mean, dim=-1)
    return float(max(0.5, dists.max().item() * 1.1))


class GaussianOptimizer:
    """Optimizes GaussianModel parameters over training views on CPU."""

    def __init__(self, model: GaussianModel, config: ReconstructionConfig):
        self.model = model
        self.config = config
        self.optimizer: Optional[torch.optim.Adam] = None
        self._max_radii: Optional[torch.Tensor] = None
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        param_groups = [
            {"params": [self.model._xyz], "lr": self.config.lr_position, "name": "xyz"},
            {"params": [self.model._features_dc], "lr": self.config.lr_feature_dc, "name": "f_dc"},
            {"params": [self.model._features_rest], "lr": self.config.lr_feature_rest, "name": "f_rest"},
            {"params": [self.model._opacity], "lr": self.config.lr_opacity, "name": "opacity"},
            {"params": [self.model._scales], "lr": self.config.lr_scale, "name": "scales"},
            {"params": [self.model._rotations], "lr": self.config.lr_rotation, "name": "rotations"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def _update_position_lr(self, step: int, total_steps: int, scene_extent: float) -> None:
        """Exponential decay of position LR, scaled by scene extent (as in original 3DGS)."""
        if self.optimizer is None or total_steps <= 1:
            return
        t = min(1.0, step / max(1, total_steps - 1))
        lr_start = self.config.lr_position * scene_extent
        lr_end = self.config.lr_position_final * scene_extent
        lr = math.exp(math.log(lr_start) * (1.0 - t) + math.log(lr_end) * t)
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = lr

    def _reset_max_radii(self) -> None:
        self._max_radii = torch.zeros(
            self.model.num_gaussians,
            dtype=torch.float32,
            device=self.model.device,
        )

    def _track_max_radii(self, radii: torch.Tensor, visible_indices: torch.Tensor) -> None:
        """Keep the largest screen-space radius seen per Gaussian since the last densify."""
        if self._max_radii is None or self._max_radii.shape[0] != self.model.num_gaussians:
            self._reset_max_radii()
        assert self._max_radii is not None
        r = radii.detach().float()
        if visible_indices is not None and r.shape[0] == visible_indices.shape[0]:
            idx = visible_indices.long()
            self._max_radii[idx] = torch.maximum(self._max_radii[idx], r)
        elif r.shape[0] == self._max_radii.shape[0]:
            self._max_radii = torch.maximum(self._max_radii, r)

    def train_steps(
        self,
        views: list[TrainingView],
        num_iterations: int,
        allow_densify: bool = True,
        progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> dict[str, float]:
        """Run `num_iterations` optimization steps sampling across `views`.

        Prioritizes newly added views while replaying earlier views to prevent
        catastrophic forgetting during incremental reconstruction.
        """
        if not views or self.model.num_gaussians == 0 or num_iterations <= 0:
            return {"final_loss": 0.0, "mean_psnr": 0.0, "iterations_run": 0}

        self._build_optimizer()
        self._reset_max_radii()
        scene_extent = compute_scene_extent(views)
        logger.info("scene_extent=%.4f | position LR start=%.3e", scene_extent, self.config.lr_position * scene_extent)
        new_views = [v for v in views if v.is_new] or views
        old_views = [v for v in views if not v.is_new]

        rng = np.random.default_rng(123 + self.model.num_gaussians)
        best_ema_loss = float("inf")
        ema_loss = 0.0
        ema_decay = getattr(self.config, "loss_ema_decay", 0.99)
        stagnant_steps = 0
        last_psnr = 0.0
        t_start = time.perf_counter()

        iters_done = 0
        for step in range(1, num_iterations + 1):
            assert self.optimizer is not None
            self._update_position_lr(step, num_iterations, scene_extent)

            # Sample view: mix new views and replay views
            if old_views and rng.random() < self.config.replay_ratio:
                view = old_views[int(rng.integers(0, len(old_views)))]
            else:
                view = new_views[int(rng.integers(0, len(new_views)))]

            self.optimizer.zero_grad(set_to_none=True)
            out = render_gaussians(
                self.model,
                view.camera,
                bg_color=self.config.background_color,
                tile_size=self.config.tile_size,
            )
            pred = out["image"]
            loss = photometric_loss(pred, view.target_image, lambda_dssim=0.2)
            # Gentle scale & opacity sparsity regularization suppresses floating noise
            if self.model.num_gaussians > 0:
                loss = (
                    loss
                    + 0.01 * self.model.get_scales.mean()
                    + 0.002 * self.model.get_opacity.mean()
                )
            loss.backward()

            # Track 2D gradient stats before step (NDC-normalized)
            self.model.add_densification_stats(
                out["means2d"].grad,
                out["visible_indices"],
                out["radii"],
                image_size=(view.camera.width, view.camera.height),
            )
            self._track_max_radii(out["radii"], out["visible_indices"])

            self.optimizer.step()
            loss_val = float(loss.item())
            ema_loss = loss_val if step == 1 else (ema_decay * ema_loss + (1.0 - ema_decay) * loss_val)
            with torch.no_grad():
                last_psnr = psnr(pred, view.target_image)

            iters_done = step

            # Periodic densification & pruning (rebuild optimizer after tensor shape changes)
            if (
                allow_densify
                and step < int(num_iterations * 0.8)
                and step % self.config.densify_interval == 0
            ):
                # Screen-space prune mask (aligned with the model BEFORE clone/split appends)
                extra_prune = None
                if (
                    self._max_radii is not None
                    and self._max_radii.shape[0] == self.model.num_gaussians
                    and step >= int(num_iterations * self.config.screen_prune_start)
                ):
                    px_limit = self.config.screen_prune_frac * max(view.camera.width, view.camera.height)
                    extra_prune = self._max_radii > px_limit

                cloned, split, pruned = densify_and_prune(
                    self.model,
                    max_gaussians=self.config.max_gaussians,
                    grad_threshold=self.config.densify_grad_threshold,
                    min_opacity=self.config.prune_opacity_threshold,
                    scene_extent=scene_extent,
                    extra_prune_mask=extra_prune,
                )
                if cloned or split or pruned:
                    self._build_optimizer()
                    if self.model.device.type == "mps" and hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                self._reset_max_radii()

            if progress_cb and (step == 1 or step % 10 == 0 or step == num_iterations):
                progress_cb(step, num_iterations, ema_loss, last_psnr)

            if step % 500 == 0:
                logger.info(
                    "step %d/%d | EMA loss=%.4f | last PSNR=%.2f | Gaussians=%d",
                    step, num_iterations, ema_loss, last_psnr, self.model.num_gaussians,
                )

            # Adaptive early stopping once loss plateaus (off by default)
            if self.config.adaptive_early_stop and step > max(40, self.config.early_stop_patience):
                if ema_loss < best_ema_loss - 1e-4:
                    best_ema_loss = ema_loss
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
                    if stagnant_steps >= self.config.early_stop_patience:
                        logger.info("Early stopping at iter %d/%d (EMA loss=%.4f)", step, num_iterations, ema_loss)
                        break

        elapsed = time.perf_counter() - t_start
        mean_psnr = self.evaluate_psnr(views[: min(6, len(views))])
        logger.info(
            "Optimization complete: %d iters in %.1fs | Gaussians=%d | EMA loss=%.4f | Mean PSNR=%.2f dB",
            iters_done, elapsed, self.model.num_gaussians, ema_loss, mean_psnr,
        )
        return {
            "final_loss": float(ema_loss),
            "mean_psnr": float(mean_psnr),
            "iterations_run": float(iters_done),
            "elapsed_sec": float(elapsed),
        }

    def evaluate_psnr(self, views: list[TrainingView]) -> float:
        """Evaluate average PSNR across a subset of views without gradients."""
        if not views or self.model.num_gaussians == 0:
            return 0.0
        scores: list[float] = []
        with torch.no_grad():
            for v in views:
                out = render_gaussians(
                    self.model,
                    v.camera,
                    bg_color=self.config.background_color,
                    tile_size=self.config.tile_size,
                )
                scores.append(psnr(out["image"], v.target_image))
        return float(np.mean(scores)) if scores else 0.0
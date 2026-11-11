"""3D Gaussian Scene Representation (CPU-only PyTorch)."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from reconstruction.gaussian.sh_utils import rgb_to_sh

logger = logging.getLogger(__name__)


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


class GaussianModel(nn.Module):
    """Holds differentiable 3D Gaussian primitives and densification statistics."""

    def __init__(self, sh_degree: int = 0):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_rest = (sh_degree + 1) ** 2 - 1

        self._xyz = nn.Parameter(torch.zeros((0, 3), dtype=torch.float32))
        self._scales = nn.Parameter(torch.zeros((0, 3), dtype=torch.float32))
        self._rotations = nn.Parameter(torch.zeros((0, 4), dtype=torch.float32))
        self._opacity = nn.Parameter(torch.zeros((0, 1), dtype=torch.float32))
        self._features_dc = nn.Parameter(torch.zeros((0, 1, 3), dtype=torch.float32))
        self._features_rest = nn.Parameter(
            torch.zeros((0, self.num_sh_rest, 3), dtype=torch.float32)
        )

        # Accumulated screen-space gradient norms for densification
        self.register_buffer("xyz_gradient_accum", torch.zeros((0, 1), dtype=torch.float32))
        self.register_buffer("denom", torch.zeros((0, 1), dtype=torch.float32))
        self.register_buffer("max_radii2d", torch.zeros((0,), dtype=torch.float32))

    @property
    def device(self) -> torch.device:
        return self._xyz.device

    @property
    def num_gaussians(self) -> int:
        return int(self._xyz.shape[0])

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scales(self) -> torch.Tensor:
        return torch.exp(self._scales.clamp(-10.0, 4.0))

    @property
    def get_rotations(self) -> torch.Tensor:
        return F.normalize(self._rotations, dim=-1, eps=1e-8)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        if self.num_sh_rest == 0:
            return self._features_dc
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def _compute_initial_scales(self, points_np: np.ndarray, ref_points_np: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate isotropic initial Gaussian log-scales from nearest-neighbor distances via cKDTree."""
        n = len(points_np)
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)

        cloud = ref_points_np if (ref_points_np is not None and len(ref_points_np) >= 4) else points_np
        if len(cloud) < 4:
            return np.full((n, 3), np.log(0.05), dtype=np.float32)

        tree = cKDTree(cloud)
        k = min(4, len(cloud))
        dists, _ = tree.query(points_np, k=k)
        # Skip self-distance column if querying the same cloud
        start_col = 1 if (cloud is points_np and k > 1) else 0
        mean_dist = np.mean(dists[:, start_col:], axis=1)
        mean_dist = np.clip(mean_dist * 0.4, 1e-3, 0.15)
        return np.log(np.repeat(mean_dist[:, None], 3, axis=1)).astype(np.float32)

    def init_from_sparse_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        max_gaussians: int = 5000,
        init_opacity: float = 0.45,
    ) -> None:
        """Initialize Gaussians from SfM sparse 3D point cloud."""
        if len(points) == 0:
            raise ValueError("Cannot initialize GaussianModel from empty point cloud")

        # Subsample uniformly if sparse point cloud exceeds budget
        if len(points) > max_gaussians:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(points), size=max_gaussians, replace=False)
            points = points[idx]
            colors = colors[idx]

        dev = self.device
        xyz = torch.tensor(points, dtype=torch.float32, device=dev)
        rgb = torch.tensor(np.clip(colors, 0.0, 1.0), dtype=torch.float32, device=dev)
        dc = rgb_to_sh(rgb).unsqueeze(1)  # [N, 1, 3]
        rest = torch.zeros((len(points), self.num_sh_rest, 3), dtype=torch.float32, device=dev)

        log_scales = torch.tensor(self._compute_initial_scales(points), dtype=torch.float32, device=dev)
        rots = torch.zeros((len(points), 4), dtype=torch.float32, device=dev)
        rots[:, 0] = 1.0  # Identity quaternion (w=1, x=0, y=0, z=0)
        opacities = _inverse_sigmoid(torch.full((len(points), 1), init_opacity, dtype=torch.float32, device=dev))

        self._set_parameters(xyz, log_scales, rots, opacities, dc, rest)
        logger.info("Initialized %d Gaussians from sparse point cloud", self.num_gaussians)

    def add_new_points(
        self,
        new_points: np.ndarray,
        new_colors: np.ndarray,
        max_gaussians: int = 5000,
        min_spacing_factor: float = 0.35,
        init_opacity: float = 0.45,
    ) -> int:
        """Incrementally add newly triangulated 3D points without duplicating existing Gaussians."""
        if len(new_points) == 0:
            return 0

        if self.num_gaussians == 0:
            self.init_from_sparse_points(new_points, new_colors, max_gaussians=max_gaussians, init_opacity=init_opacity)
            return self.num_gaussians

        room = max_gaussians - self.num_gaussians
        if room <= 0:
            logger.info("Gaussian budget reached (%d); skipping new sparse points", max_gaussians)
            return 0

        existing_np = self._xyz.detach().cpu().numpy()
        tree = cKDTree(existing_np)

        # Estimate typical scene spacing
        sample_dists, _ = tree.query(existing_np[: min(500, len(existing_np))], k=min(2, len(existing_np)))
        if sample_dists.ndim == 2 and sample_dists.shape[1] > 1:
            med_spacing = float(np.median(sample_dists[:, 1]))
        else:
            med_spacing = 0.05
        min_dist = max(1e-4, med_spacing * min_spacing_factor)

        dists_to_existing, _ = tree.query(new_points, k=1)
        novel_mask = dists_to_existing > min_dist
        novel_pts = new_points[novel_mask]
        novel_cols = new_colors[novel_mask]

        if len(novel_pts) == 0:
            return 0

        if len(novel_pts) > room:
            rng = np.random.default_rng(self.num_gaussians)
            sel = rng.choice(len(novel_pts), size=room, replace=False)
            novel_pts = novel_pts[sel]
            novel_cols = novel_cols[sel]

        dev = self.device
        xyz_add = torch.tensor(novel_pts, dtype=torch.float32, device=dev)
        rgb_add = torch.tensor(np.clip(novel_cols, 0.0, 1.0), dtype=torch.float32, device=dev)
        dc_add = rgb_to_sh(rgb_add).unsqueeze(1)
        rest_add = torch.zeros((len(novel_pts), self.num_sh_rest, 3), dtype=torch.float32, device=dev)
        scales_add = torch.tensor(
            self._compute_initial_scales(novel_pts, ref_points_np=existing_np), dtype=torch.float32, device=dev
        )
        rots_add = torch.zeros((len(novel_pts), 4), dtype=torch.float32, device=dev)
        rots_add[:, 0] = 1.0
        opa_add = _inverse_sigmoid(torch.full((len(novel_pts), 1), init_opacity, dtype=torch.float32, device=dev))

        self.append_tensors(xyz_add, scales_add, rots_add, opa_add, dc_add, rest_add)
        logger.info("Incrementally added %d new Gaussians (total=%d)", len(novel_pts), self.num_gaussians)
        return len(novel_pts)

    def _set_parameters(
        self,
        xyz: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
    ) -> None:
        self._xyz = nn.Parameter(xyz.detach().clone().requires_grad_(True))
        self._scales = nn.Parameter(scales.detach().clone().requires_grad_(True))
        self._rotations = nn.Parameter(rotations.detach().clone().requires_grad_(True))
        self._opacity = nn.Parameter(opacity.detach().clone().requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.detach().clone().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.detach().clone().requires_grad_(True))

        n = xyz.shape[0]
        dev = xyz.device
        self.xyz_gradient_accum = torch.zeros((n, 1), dtype=torch.float32, device=dev)
        self.denom = torch.zeros((n, 1), dtype=torch.float32, device=dev)
        self.max_radii2d = torch.zeros((n,), dtype=torch.float32, device=dev)

    def append_tensors(
        self,
        xyz: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
    ) -> None:
        """Append new Gaussian tensors to existing model parameters."""
        dev = self.device
        new_xyz = torch.cat([self._xyz.detach(), xyz.detach().to(dev)], dim=0)
        new_scales = torch.cat([self._scales.detach(), scales.detach().to(dev)], dim=0)
        new_rots = torch.cat([self._rotations.detach(), rotations.detach().to(dev)], dim=0)
        new_opa = torch.cat([self._opacity.detach(), opacity.detach().to(dev)], dim=0)
        new_dc = torch.cat([self._features_dc.detach(), features_dc.detach().to(dev)], dim=0)
        new_rest = torch.cat([self._features_rest.detach(), features_rest.detach().to(dev)], dim=0)

        self._set_parameters(new_xyz, new_scales, new_rots, new_opa, new_dc, new_rest)

    def keep_mask(self, mask: torch.Tensor) -> None:
        """Prune Gaussians where `mask` is False."""
        with torch.no_grad():
            self._set_parameters(
                self._xyz[mask],
                self._scales[mask],
                self._rotations[mask],
                self._opacity[mask],
                self._features_dc[mask],
                self._features_rest[mask],
            )

    def add_densification_stats(
        self,
        means2d_grad: Optional[torch.Tensor],
        visible_indices: torch.Tensor,
        radii: torch.Tensor,
        image_size: tuple[int, int] = (384, 384),
    ) -> None:
        """Accumulate NDC-normalized 2D screen-space gradient magnitudes for visible Gaussians."""
        if means2d_grad is None or len(visible_indices) == 0:
            return
        with torch.no_grad():
            # Convert pixel-space gradients to NDC [-1, 1] scale matching standard 3DGS thresholds
            ndc_scale = max(image_size) * 0.5
            grad_norm = torch.norm(means2d_grad[visible_indices, :2], dim=-1, keepdim=True) * ndc_scale
            self.xyz_gradient_accum[visible_indices] += grad_norm
            self.denom[visible_indices] += 1.0
            self.max_radii2d[visible_indices] = torch.maximum(
                self.max_radii2d[visible_indices], radii.detach()
            )

    def reset_densification_stats(self) -> None:
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2d.zero_()

    def export_checkpoint_dict(self) -> dict:
        return {
            "sh_degree": self.sh_degree,
            "xyz": self._xyz.detach().cpu(),
            "scales": self._scales.detach().cpu(),
            "rotations": self._rotations.detach().cpu(),
            "opacity": self._opacity.detach().cpu(),
            "features_dc": self._features_dc.detach().cpu(),
            "features_rest": self._features_rest.detach().cpu(),
        }

    def load_checkpoint_dict(self, data: dict) -> None:
        dev = self.device
        self.sh_degree = int(data["sh_degree"])
        self.num_sh_rest = (self.sh_degree + 1) ** 2 - 1
        self._set_parameters(
            data["xyz"].to(dev),
            data["scales"].to(dev),
            data["rotations"].to(dev),
            data["opacity"].to(dev),
            data["features_dc"].to(dev),
            data["features_rest"].to(dev),
        )


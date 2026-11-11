"""Adaptive Gaussian densification (clone/split) and pruning on CPU."""

import logging
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
import torch

from reconstruction.gaussian.model import GaussianModel

logger = logging.getLogger(__name__)


def _sample_split_offsets(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Sample 3D offsets along the Gaussian's oriented ellipsoid axes."""
    from reconstruction.gaussian.rasterizer import quaternion_to_rotation_matrix

    R = quaternion_to_rotation_matrix(rotations)  # [M, 3, 3]
    stds = scales.unsqueeze(-1)
    samples = torch.randn((scales.shape[0], 3, 1), dtype=torch.float32, device=scales.device) * (0.5 * stds)
    return torch.bmm(R, samples).squeeze(-1)


def densify_and_prune(
    model: GaussianModel,
    max_gaussians: int,
    grad_threshold: float = 0.0003,
    min_opacity: float = 0.02,
    scene_extent: float = 1.0,
    percent_dense: float = 0.02,
    extra_prune_mask: Optional[torch.Tensor] = None,
) -> tuple[int, int, int]:
    """Perform 3DGS adaptive density control (clone, split, prune) within CPU budget.

    `extra_prune_mask` is an optional bool tensor aligned with the model's Gaussians
    BEFORE this call (e.g. screen-space oversize). Clones/splits are appended at the
    end, so it is padded with False for them.

    Returns (num_cloned, num_split, num_pruned).
    """
    if model.num_gaussians == 0:
        return 0, 0, 0

    with torch.no_grad():
        grads = model.xyz_gradient_accum / torch.clamp(model.denom, min=1.0)
        grads = grads.squeeze(-1)
        grads[torch.isnan(grads)] = 0.0

        scales = model.get_scales
        max_scale = torch.max(scales, dim=-1).values
        high_grad = grads >= grad_threshold

        size_thresh = max(1e-3, percent_dense * scene_extent)
        room = max(0, max_gaussians - model.num_gaussians)

        num_cloned = 0
        num_split = 0

        if room > 0 and torch.any(high_grad):
            # 1. Clone small high-gradient Gaussians
            clone_mask = high_grad & (max_scale <= size_thresh)
            clone_idx = torch.nonzero(clone_mask, as_tuple=False).squeeze(-1)
            if len(clone_idx) > room:
                top = torch.argsort(grads[clone_idx], descending=True)[:room]
                clone_idx = clone_idx[top]

            if len(clone_idx) > 0:
                model.append_tensors(
                    model._xyz[clone_idx],
                    model._scales[clone_idx],
                    model._rotations[clone_idx],
                    model._opacity[clone_idx],
                    model._features_dc[clone_idx],
                    model._features_rest[clone_idx],
                )
                num_cloned = int(len(clone_idx))
                room = max(0, max_gaussians - model.num_gaussians)

            # 2. Split large high-gradient Gaussians into smaller children
            if room > 0:
                split_mask = high_grad[: len(max_scale)] & (max_scale > size_thresh)
                split_idx = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)
                if len(split_idx) > room:
                    top = torch.argsort(grads[split_idx], descending=True)[:room]
                    split_idx = split_idx[top]

                if len(split_idx) > 0:
                    s_orig = model.get_scales[split_idx]
                    r_orig = model.get_rotations[split_idx]
                    offset = _sample_split_offsets(s_orig, r_orig)

                    new_scales = torch.log(torch.clamp(s_orig / 1.6, min=1e-4))
                    # Update parent in-place to one child, append second child
                    model._xyz[split_idx] = model._xyz[split_idx] - offset
                    model._scales[split_idx] = new_scales

                    model.append_tensors(
                        model._xyz[split_idx] + 2.0 * offset,
                        new_scales,
                        model._rotations[split_idx],
                        model._opacity[split_idx],
                        model._features_dc[split_idx],
                        model._features_rest[split_idx],
                    )
                    num_split = int(len(split_idx))

        # 3. Prune transparent, oversized, far-field, or isolated floater Gaussians
        opacities = model.get_opacity.squeeze(-1)
        scales_now = model.get_scales
        max_scale_now = torch.max(scales_now, dim=-1).values

        xyz_now = model._xyz.detach()
        centroid = torch.median(xyz_now, dim=0).values
        dist_from_center = torch.norm(xyz_now - centroid.unsqueeze(0), dim=-1)

        # Scale-adaptive bounds: keep foreground crisp while retaining distant background structures
        max_dist = max(1e-2, scene_extent * 12.0)
        fg_scale_limit = max(1e-3, scene_extent * 0.08)
        max_allowed_scale = fg_scale_limit + 0.08 * dist_from_center

        prune_mask = (
            (opacities < max(0.03, min_opacity))
            | (max_scale_now > max_allowed_scale)
            | (dist_from_center > max_dist)
        )

        # NEW: screen-space oversize prune (mask is aligned with pre-densify model)
        if extra_prune_mask is not None:
            n_now = model.num_gaussians
            padded = torch.zeros(n_now, dtype=torch.bool, device=xyz_now.device)
            n_copy = min(n_now, extra_prune_mask.shape[0])
            padded[:n_copy] = extra_prune_mask[:n_copy].to(device=xyz_now.device, dtype=torch.bool)
            prune_mask = prune_mask | padded

        # Also prune extreme isolated spatial outlier Gaussians
        if model.num_gaussians >= 64:
            xyz_np = xyz_now.cpu().numpy()
            tree = cKDTree(xyz_np)
            nn_dists, _ = tree.query(xyz_np, k=4)
            knn_mean = np.mean(nn_dists[:, 1:], axis=1)
            med_nn = max(1e-4, float(np.median(knn_mean)))
            isolated = torch.from_numpy(knn_mean > (6.0 * med_nn)).to(xyz_now.device)
            prune_mask = prune_mask | isolated

        keep = ~prune_mask
        num_pruned = 0
        if int(keep.sum().item()) >= min(32, model.num_gaussians):
            num_pruned = int(prune_mask.sum().item())
            if num_pruned > 0:
                model.keep_mask(keep)

        model.reset_densification_stats()

    if num_cloned or num_split or num_pruned:
        logger.debug(
            "Densify/Prune: +%d cloned, +%d split, -%d pruned (total=%d)",
            num_cloned, num_split, num_pruned, model.num_gaussians,
        )
    return num_cloned, num_split, num_pruned
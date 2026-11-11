"""Pure PyTorch CPU-only differentiable Gaussian Splatting rasterizer.

Implements EWA splatting (3D->2D covariance projection), depth sorting, and
tile-based sparse alpha compositing using only vectorized PyTorch CPU ops.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch

from reconstruction.gaussian.model import GaussianModel
from reconstruction.gaussian.sh_utils import eval_sh


@dataclass
class Camera:
    """Differentiable rendering camera parameters."""

    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: torch.Tensor  # [3, 3] world-to-camera
    t: torch.Tensor  # [3] world-to-camera

    @property
    def center_world(self) -> torch.Tensor:
        return -self.R.T @ self.t


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized quaternions [N, 4] (w, x, y, z) to rotation matrices [N, 3, 3]."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)

    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)

    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=1,
    )


_TILE_GRID_CACHE: dict[tuple[int, int, int, str], list[list[tuple[torch.Tensor, torch.Tensor]]]] = {}


def _get_tile_grids(
    H: int,
    W: int,
    tile_size: int,
    device: torch.device,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Cache [1, P] pixel coordinate grids per tile & device."""
    key = (H, W, tile_size, str(device))
    cached = _TILE_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    nx = math.ceil(W / tile_size)
    ny = math.ceil(H / tile_size)
    grids: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for ty in range(ny):
        y0 = ty * tile_size
        y1 = min(H, y0 + tile_size)
        py = torch.arange(y0, y1, dtype=torch.float32, device=device) + 0.5
        row_grids: list[tuple[torch.Tensor, torch.Tensor]] = []
        for tx in range(nx):
            x0 = tx * tile_size
            x1 = min(W, x0 + tile_size)
            px = torch.arange(x0, x1, dtype=torch.float32, device=device) + 0.5
            gy, gx = torch.meshgrid(py, px, indexing="ij")
            row_grids.append((gx.reshape(1, -1), gy.reshape(1, -1)))
        grids.append(row_grids)

    if len(_TILE_GRID_CACHE) > 16:
        _TILE_GRID_CACHE.clear()
    _TILE_GRID_CACHE[key] = grids
    return grids


def render_gaussians(
    model: GaussianModel,
    camera: Camera,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tile_size: int = 24,
    max_splats_per_tile: int = 4096,
    active_sh_degree: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Differentiably render 3D Gaussians to an RGB image [H, W, 3] on CPU or GPU (MPS/CUDA)."""
    H, W = camera.height, camera.width
    dev = model.device
    bg = torch.tensor(bg_color, dtype=torch.float32, device=dev)

    if model.num_gaussians == 0:
        img = bg.view(1, 1, 3).expand(H, W, 3).clone()
        return {
            "image": img,
            "means2d": torch.zeros((0, 2), dtype=torch.float32, device=dev),
            "visible_indices": torch.zeros((0,), dtype=torch.long, device=dev),
            "radii": torch.zeros((0,), dtype=torch.float32, device=dev),
        }

    means = model.get_xyz                      # [N, 3]
    scales = model.get_scales                  # [N, 3]
    rotations = model.get_rotations            # [N, 4]
    opacities = model.get_opacity.squeeze(-1)  # [N]
    sh_feats = model.get_features              # [N, K, 3]

    # 1. Transform 3D means to camera frame
    cam_R = camera.R.to(dev)
    cam_t = camera.t.to(dev)
    means_cam = means @ cam_R.T + cam_t  # [N, 3]
    z = means_cam[:, 2]

    # Near-plane cull
    in_front = z > 0.1
    if not torch.any(in_front):
        img = bg.view(1, 1, 3).expand(H, W, 3).clone()
        return {
            "image": img,
            "means2d": torch.zeros((model.num_gaussians, 2), dtype=torch.float32, device=dev),
            "visible_indices": torch.zeros((0,), dtype=torch.long, device=dev),
            "radii": torch.zeros((0,), dtype=torch.float32, device=dev),
        }

    # 2. Project to 2D pixel space (keep full [N, 2] tensor for densification gradient tracking)
    z_safe = torch.where(in_front, z, torch.ones_like(z))
    u_all = camera.fx * (means_cam[:, 0] / z_safe) + camera.cx
    v_all = camera.fy * (means_cam[:, 1] / z_safe) + camera.cy
    means2d = torch.stack([u_all, v_all], dim=-1)  # [N, 2]
    if means2d.requires_grad:
        means2d.retain_grad()

    # 3. 3D covariance matrices for front Gaussians: Sigma = R_g S^2 R_g^T
    idx_front = torch.nonzero(in_front, as_tuple=False).squeeze(-1)
    mc = means_cam[idx_front]
    zf = z[idx_front]
    sc = scales[idx_front]
    R_g = quaternion_to_rotation_matrix(rotations[idx_front])  # [M, 3, 3]
    M_mat = R_g * sc.unsqueeze(1)  # R_g @ diag(s)
    cov3d = M_mat @ M_mat.transpose(-1, -2)  # [M, 3, 3]

    # 4. EWA Splatting Jacobian J [M, 2, 3]
    lim_x = 1.3 * (W * 0.5 / camera.fx)
    lim_y = 1.3 * (H * 0.5 / camera.fy)
    tx = torch.clamp(mc[:, 0] / zf, -lim_x, lim_x) * zf
    ty = torch.clamp(mc[:, 1] / zf, -lim_y, lim_y) * zf

    inv_z = 1.0 / zf
    inv_z2 = inv_z * inv_z
    zeros = torch.zeros_like(zf)

    J_row0 = torch.stack([camera.fx * inv_z, zeros, -camera.fx * tx * inv_z2], dim=-1)
    J_row1 = torch.stack([zeros, camera.fy * inv_z, -camera.fy * ty * inv_z2], dim=-1)
    J = torch.stack([J_row0, J_row1], dim=1)  # [M, 2, 3]

    # Project covariance: T = J @ R_cam, cov2d = T @ cov3d @ T^T + 0.3 * I
    T_mat = J @ cam_R.unsqueeze(0)  # [M, 2, 3]
    cov2d = T_mat @ cov3d @ T_mat.transpose(-1, -2)  # [M, 2, 2]
    a = cov2d[:, 0, 0] + 0.3
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + 0.3

    det = a * c - b * b
    valid_cov = det > 1e-6

    # 5. Eigenvalue-based 3.5-sigma pixel radius (sub-1/255 cutoff prevents tile edge seams)
    mid = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
    lambda_max = mid + disc
    radii_f = torch.ceil(3.5 * torch.sqrt(torch.clamp(lambda_max, min=1e-4)))
    radii_f = torch.clamp(radii_f, min=1.0, max=float(max(H, W)))

    u_f = means2d[idx_front, 0]
    v_f = means2d[idx_front, 1]
    in_screen = (
        valid_cov
        & ((u_f + radii_f) >= 0.0)
        & ((u_f - radii_f) < W)
        & ((v_f + radii_f) >= 0.0)
        & ((v_f - radii_f) < H)
        & (opacities[idx_front] > 1.0 / 255.0)
    )

    if not torch.any(in_screen):
        img = bg.view(1, 1, 3).expand(H, W, 3).clone()
        return {
            "image": img,
            "means2d": means2d,
            "visible_indices": torch.zeros((0,), dtype=torch.long, device=dev),
            "radii": torch.zeros((0,), dtype=torch.float32, device=dev),
        }

    # Filter to visible Gaussians and sort front-to-back by depth
    vis_sub = torch.nonzero(in_screen, as_tuple=False).squeeze(-1)
    sort_order = torch.argsort(zf[vis_sub], descending=False)
    vis_sub = vis_sub[sort_order]

    vis_indices = idx_front[vis_sub]  # Indices in full model [K_vis]
    u_vis = u_f[vis_sub]
    v_vis = v_f[vis_sub]
    r_vis = radii_f[vis_sub]

    det_v = det[vis_sub]
    inv_det = 1.0 / det_v
    cxx = c[vis_sub] * inv_det
    cxy = -b[vis_sub] * inv_det
    cyy = a[vis_sub] * inv_det
    opa_vis = opacities[vis_indices]

    # 6. Evaluate SH colors
    sh_deg = model.sh_degree if active_sh_degree is None else min(active_sh_degree, model.sh_degree)
    if sh_deg > 0:
        cam_center = (-cam_R.T @ cam_t)
        dirs = means[vis_indices] - cam_center.unsqueeze(0)
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)
    else:
        dirs = None
    colors_vis = eval_sh(sh_deg, sh_feats[vis_indices], dirs)  # [K_vis, 3]

    # 7. Sparse Tile-Based Rasterization (memory-bounded & CPU cache friendly)
    # Precompute tile coverage bounding boxes
    with torch.no_grad():
        tx_min = torch.clamp(torch.floor((u_vis - r_vis) / tile_size).to(torch.long), 0, (W - 1) // tile_size)
        tx_max = torch.clamp(torch.floor((u_vis + r_vis) / tile_size).to(torch.long), 0, (W - 1) // tile_size)
        ty_min = torch.clamp(torch.floor((v_vis - r_vis) / tile_size).to(torch.long), 0, (H - 1) // tile_size)
        ty_max = torch.clamp(torch.floor((v_vis + r_vis) / tile_size).to(torch.long), 0, (H - 1) // tile_size)

    nx = math.ceil(W / tile_size)
    ny = math.ceil(H / tile_size)
    tile_grids = _get_tile_grids(H, W, tile_size, dev)

    row_strips: list[torch.Tensor] = []
    for ty in range(ny):
        y0 = ty * tile_size
        y1 = min(H, y0 + tile_size)
        th = y1 - y0
        row_mask = (ty_min <= ty) & (ty_max >= ty)

        col_tiles: list[torch.Tensor] = []
        for tx in range(nx):
            x0 = tx * tile_size
            x1 = min(W, x0 + tile_size)
            tw = x1 - x0

            tile_mask = row_mask & (tx_min <= tx) & (tx_max >= tx)
            if not torch.any(tile_mask):
                col_tiles.append(bg.view(1, 1, 3).expand(th, tw, 3))
                continue

            t_idx = torch.nonzero(tile_mask, as_tuple=False).squeeze(-1)
            if len(t_idx) > max_splats_per_tile:
                t_idx = t_idx[:max_splats_per_tile]

            gx_flat, gy_flat = tile_grids[ty][tx]

            dx = gx_flat - u_vis[t_idx].unsqueeze(1)  # [Kt, P]
            dy = gy_flat - v_vis[t_idx].unsqueeze(1)  # [Kt, P]

            power = -0.5 * (
                cxx[t_idx].unsqueeze(1) * dx * dx
                + 2.0 * cxy[t_idx].unsqueeze(1) * dx * dy
                + cyy[t_idx].unsqueeze(1) * dy * dy
            )
            alpha = opa_vis[t_idx].unsqueeze(1) * torch.exp(torch.clamp(power, max=0.0))
            alpha = torch.clamp(alpha, max=0.99)
            # Zero out tiny contributions cleanly
            alpha = torch.where(alpha >= (1.0 / 255.0), alpha, torch.zeros_like(alpha))

            one_minus_a = 1.0 - alpha  # [Kt, P]
            ones_top = torch.ones((1, one_minus_a.shape[1]), dtype=torch.float32, device=dev)
            T_cum = torch.cumprod(torch.cat([ones_top, one_minus_a], dim=0), dim=0)
            T = T_cum[:-1]          # [Kt, P]
            T_final = T_cum[-1]     # [P]

            weights = alpha * T     # [Kt, P]
            tile_rgb = (
                torch.matmul(weights.T, colors_vis[t_idx])  # [P, 3]
                + T_final.unsqueeze(1) * bg.unsqueeze(0)
            )
            col_tiles.append(tile_rgb.view(th, tw, 3))

        row_strips.append(torch.cat(col_tiles, dim=1))

    image = torch.clamp(torch.cat(row_strips, dim=0), 0.0, 1.0)
    return {
        "image": image,
        "means2d": means2d,
        "visible_indices": vis_indices,
        "radii": r_vis,
    }

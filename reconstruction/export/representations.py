"""Derived 3D representations from GaussianModel: Geometry, Marching-Cubes Mesh, and Triangle Splats."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from skimage import measure
import torch

from reconstruction.gaussian.model import GaussianModel
from reconstruction.gaussian.rasterizer import quaternion_to_rotation_matrix
from reconstruction.gaussian.sh_utils import sh_to_rgb
from reconstruction.sfm.types import CameraPose

logger = logging.getLogger(__name__)


def load_gaussian_model_from_artifacts(
    checkpoint_pt: Optional[Path] = None,
    ply_path: Optional[Path] = None,
) -> GaussianModel:
    """Reconstruct a GaussianModel from checkpoint `gaussians.pt` or exported `scene.ply`."""
    if checkpoint_pt is not None and checkpoint_pt.exists():
        data = torch.load(checkpoint_pt, map_location="cpu", weights_only=True)
        model = GaussianModel(sh_degree=int(data.get("sh_degree", 0)))
        model.load_checkpoint_dict(data)
        return model

    if ply_path is not None and ply_path.exists():
        ply = PlyData.read(str(ply_path))["vertex"]
        n = len(ply)
        xyz = np.column_stack([ply["x"], ply["y"], ply["z"]]).astype(np.float32)
        f_dc = np.column_stack([ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]]).astype(np.float32)
        opacity = np.asarray(ply["opacity"], dtype=np.float32).reshape(n, 1)
        scales = np.column_stack([ply["scale_0"], ply["scale_1"], ply["scale_2"]]).astype(np.float32)
        rots = np.column_stack([ply["rot_0"], ply["rot_1"], ply["rot_2"], ply["rot_3"]]).astype(np.float32)

        model = GaussianModel(sh_degree=0)
        model._set_parameters(
            torch.from_numpy(xyz),
            torch.from_numpy(scales),
            torch.from_numpy(rots),
            torch.from_numpy(opacity),
            torch.from_numpy(f_dc).unsqueeze(1),
            torch.zeros((n, 0, 3), dtype=torch.float32),
        )
        return model

    raise FileNotFoundError("Neither gaussians.pt nor scene.ply exists for this job")


def extract_geometry_data(
    model: GaussianModel,
    poses: Optional[dict[str, CameraPose]] = None,
    max_points: int = 12000,
) -> dict[str, Any]:
    """Extract 3D point cloud, Gaussian ellipsoid principal extents, and camera frustums."""
    n = model.num_gaussians
    if n == 0:
        return {"count": 0, "positions": [], "colors": [], "scales": [], "rotations": [], "opacities": [], "cameras": []}

    with torch.no_grad():
        xyz = model._xyz.detach().cpu().numpy().astype(np.float32)
        rgb = torch.clamp(sh_to_rgb(model._features_dc.detach()[:, 0, :]), 0.0, 1.0).cpu().numpy().astype(np.float32)
        scales = model.get_scales.detach().cpu().numpy().astype(np.float32)
        rots = model.get_rotations.detach().cpu().numpy().astype(np.float32)
        opacities = model.get_opacity.detach().cpu().numpy().ravel().astype(np.float32)

    if n > max_points:
        order = np.argsort(-opacities)[:max_points]
        xyz = xyz[order]
        rgb = rgb[order]
        scales = scales[order]
        rots = rots[order]
        opacities = opacities[order]

    cameras_list: list[dict[str, Any]] = []
    if poses:
        for name, p in sorted(poses.items()):
            R_c2w = p.R.T
            center = (-R_c2w @ p.t).astype(float).tolist()
            forward = R_c2w[:, 2].astype(float).tolist()
            up = (-R_c2w[:, 1]).astype(float).tolist()
            right = R_c2w[:, 0].astype(float).tolist()
            cameras_list.append(
                {
                    "name": name,
                    "center": center,
                    "forward": forward,
                    "up": up,
                    "right": right,
                }
            )

    return {
        "mode": "geometry",
        "count": int(len(xyz)),
        "positions": np.round(xyz, 4).ravel().tolist(),
        "colors": np.round(rgb, 4).ravel().tolist(),
        "scales": np.round(scales, 4).ravel().tolist(),
        "rotations": np.round(rots, 4).ravel().tolist(),
        "opacities": np.round(opacities, 4).tolist(),
        "cameras": cameras_list,
    }


def extract_triangle_mesh(
    model: GaussianModel,
    grid_res: int = 48,
    iso_fraction: float = 0.22,
) -> dict[str, Any]:
    """Extract a colored 3D triangle mesh from the Gaussian opacity density field via Marching Cubes."""
    n = model.num_gaussians
    if n < 8:
        return {"mode": "mesh", "vertex_count": 0, "face_count": 0, "vertices": [], "faces": [], "normals": [], "colors": []}

    with torch.no_grad():
        xyz = model._xyz.detach().cpu().numpy().astype(np.float32)
        scales = model.get_scales.detach().cpu().numpy().astype(np.float32)
        opa = model.get_opacity.detach().cpu().numpy().ravel().astype(np.float32)
        rgb = torch.clamp(sh_to_rgb(model._features_dc.detach()[:, 0, :]), 0.0, 1.0).cpu().numpy().astype(np.float32)

    # Filter out nearly invisible floaters before meshing
    active = opa > 0.04
    if int(np.sum(active)) >= 8:
        xyz_a = xyz[active]
        scales_a = scales[active]
        opa_a = opa[active]
        rgb_a = rgb[active]
    else:
        xyz_a, scales_a, opa_a, rgb_a = xyz, scales, opa, rgb

    # Robust bounding box (2nd-98th percentile + padding) so single outliers don't stretch the voxel grid
    bmin = np.percentile(xyz_a, 2, axis=0)
    bmax = np.percentile(xyz_a, 98, axis=0)
    extent = np.maximum(bmax - bmin, 0.2)
    pad = extent * 0.12
    bmin = bmin - pad
    bmax = bmax + pad
    span = bmax - bmin
    voxel_size = span / float(grid_res - 1)

    density = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)

    # Splat each Gaussian locally into its 2.5-sigma voxel bounding box (O(N * r^3))
    r_world = np.clip(scales_a.max(axis=1) * 2.5, voxel_size.max() * 1.1, span.max() * 0.25)
    grid_coords = ((xyz_a - bmin[None, :]) / voxel_size[None, :])
    r_vox = r_world[:, None] / voxel_size[None, :]

    for idx in range(len(xyz_a)):
        cx, cy, cz = grid_coords[idx]
        rx, ry, rz = r_vox[idx]
        x0 = max(0, int(np.floor(cx - rx)))
        x1 = min(grid_res, int(np.ceil(cx + rx)) + 1)
        y0 = max(0, int(np.floor(cy - ry)))
        y1 = min(grid_res, int(np.ceil(cy + ry)) + 1)
        z0 = max(0, int(np.floor(cz - rz)))
        z1 = min(grid_res, int(np.ceil(cz + rz)) + 1)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue

        xs = np.arange(x0, x1, dtype=np.float32) * voxel_size[0] + bmin[0]
        ys = np.arange(y0, y1, dtype=np.float32) * voxel_size[1] + bmin[1]
        zs = np.arange(z0, z1, dtype=np.float32) * voxel_size[2] + bmin[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        dx = gx - xyz_a[idx, 0]
        dy = gy - xyz_a[idx, 1]
        dz = gz - xyz_a[idx, 2]
        sigma = max(float(scales_a[idx].mean()), float(voxel_size.max()) * 0.75)
        val = opa_a[idx] * np.exp(-0.5 * (dx * dx + dy * dy + dz * dz) / (sigma * sigma))
        density[x0:x1, y0:y1, z0:z1] += val

    d_max = float(density.max())
    if d_max <= 1e-4:
        return {"mode": "mesh", "vertex_count": 0, "face_count": 0, "vertices": [], "faces": [], "normals": [], "colors": []}

    iso_level = max(1e-3, min(d_max * 0.85, float(np.percentile(density[density > 0], 60)) * 0.8 + d_max * iso_fraction * 0.2))
    try:
        verts_vox, faces, normals, _ = measure.marching_cubes(density, level=iso_level, spacing=tuple(voxel_size))
    except Exception:
        iso_level = d_max * 0.35
        verts_vox, faces, normals, _ = measure.marching_cubes(density, level=iso_level, spacing=tuple(voxel_size))

    verts_world = verts_vox.astype(np.float32) + bmin[None, :]

    # Sample smooth vertex colors from nearest active Gaussians
    tree = cKDTree(xyz_a)
    k = min(3, len(xyz_a))
    dists, nn_idx = tree.query(verts_world, k=k)
    if k == 1:
        vert_colors = rgb_a[nn_idx]
    else:
        w = 1.0 / np.maximum(dists, 1e-4)
        w /= np.sum(w, axis=1, keepdims=True)
        vert_colors = np.sum(rgb_a[nn_idx] * w[:, :, None], axis=1)

    return {
        "mode": "mesh",
        "vertex_count": int(len(verts_world)),
        "face_count": int(len(faces)),
        "vertices": np.round(verts_world, 4).ravel().tolist(),
        "faces": faces.astype(np.int32).ravel().tolist(),
        "normals": np.round(normals, 4).ravel().tolist(),
        "colors": np.round(np.clip(vert_colors, 0.0, 1.0), 4).ravel().tolist(),
    }


def export_mesh_obj(mesh_data: dict[str, Any], output_path: Path) -> Path:
    """Write extracted triangle mesh to Wavefront .OBJ with per-vertex RGB."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(mesh_data["vertices"], dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(mesh_data["colors"], dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh_data["faces"], dtype=np.int32).reshape(-1, 3)

    lines = ["# CPU-3DGS Extracted Triangle Mesh"]
    for v, c in zip(verts, cols):
        lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
    for f in faces:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
    output_path.write_text("\n".join(lines))
    return output_path


def extract_triangle_splats(
    model: GaussianModel,
    max_triangles: int = 12000,
) -> dict[str, Any]:
    """Convert 3D Gaussians into oriented 3D Triangle Surfels (Triangle Splatting).

    Each Gaussian becomes a 3D triangle aligned with its two largest principal tangent axes,
    carrying barycentric UVs so WebGL shaders can render either flat surfels or soft radial splats.
    """
    n = model.num_gaussians
    if n == 0:
        return {"mode": "triangle_splats", "triangle_count": 0, "positions": [], "normals": [], "colors": [], "opacities": [], "barycentrics": []}

    with torch.no_grad():
        xyz = model._xyz.detach()
        scales = model.get_scales.detach()
        rots = model.get_rotations.detach()
        opa = model.get_opacity.detach().squeeze(-1)
        rgb = torch.clamp(sh_to_rgb(model._features_dc.detach()[:, 0, :]), 0.0, 1.0)
        R = quaternion_to_rotation_matrix(rots)  # [N, 3, 3]

    # Filter invisible Gaussians and sort by visual importance
    active = opa > 0.03
    if torch.any(active):
        xyz = xyz[active]
        scales = scales[active]
        opa = opa[active]
        rgb = rgb[active]
        R = R[active]

    m = len(xyz)
    if m > max_triangles:
        imp = opa * scales.prod(dim=-1)
        top = torch.argsort(imp, descending=True)[:max_triangles]
        xyz, scales, opa, rgb, R = xyz[top], scales[top], opa[top], rgb[top], R[top]
        m = max_triangles

    # Sort axis indices per Gaussian by scale descending so [0, 1] are tangent plane axes, [2] is normal
    axis_order = torch.argsort(scales, dim=-1, descending=True)  # [M, 3]
    s_sorted = torch.gather(scales, 1, axis_order)               # [M, 3]
    idx_exp = axis_order.unsqueeze(1).expand(-1, 3, -1)
    R_sorted = torch.gather(R, 2, idx_exp)                       # [M, 3, 3]

    e1 = R_sorted[:, :, 0] * (s_sorted[:, 0:1] * 2.1)            # Major tangent vector
    e2 = R_sorted[:, :, 1] * (s_sorted[:, 1:2] * 2.1)            # Minor tangent vector
    normal = R_sorted[:, :, 2]                                   # Surface normal

    # Equilateral-in-tangent-ellipse triangle vertices around center xyz
    # v0 = c + e2; v1 = c - 0.866*e1 - 0.5*e2; v2 = c + 0.866*e1 - 0.5*e2
    v0 = xyz + e2
    v1 = xyz - 0.8660254 * e1 - 0.5 * e2
    v2 = xyz + 0.8660254 * e1 - 0.5 * e2

    tri_pos = torch.stack([v0, v1, v2], dim=1).cpu().numpy().astype(np.float32)       # [M, 3, 3]
    tri_norm = normal.unsqueeze(1).expand(-1, 3, -1).cpu().numpy().astype(np.float32)  # [M, 3, 3]
    tri_col = rgb.unsqueeze(1).expand(-1, 3, -1).cpu().numpy().astype(np.float32)      # [M, 3, 3]
    tri_opa = opa.unsqueeze(1).expand(-1, 3).cpu().numpy().astype(np.float32)          # [M, 3]

    # Local 2D coordinates inside triangle for smooth radial Gaussian falloff in fragment shader:
    # (0, 1), (-0.866, -0.5), (0.866, -0.5) -> distance^2 from centroid (0,0) = 1 at vertices, 0.25 at edges
    local_uv = np.array([[0.0, 1.0], [-0.8660254, -0.5], [0.8660254, -0.5]], dtype=np.float32)
    tri_uv = np.broadcast_to(local_uv[None, :, :], (m, 3, 2)).copy()

    return {
        "mode": "triangle_splats",
        "triangle_count": int(m),
        "positions": np.round(tri_pos, 4).ravel().tolist(),
        "normals": np.round(tri_norm, 4).ravel().tolist(),
        "colors": np.round(tri_col, 4).ravel().tolist(),
        "opacities": np.round(tri_opa, 4).ravel().tolist(),
        "uvs": tri_uv.ravel().tolist(),
    }


def generate_and_cache_representations(
    preview_dir: Path,
    model: GaussianModel,
    poses: Optional[dict[str, CameraPose]] = None,
    mesh_grid_res: int = 44,
    export_obj_path: Optional[Path] = None,
) -> None:
    """Generate and persist JSON caches for Geometry, Mesh, and Triangle Splatting."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    geom = extract_geometry_data(model, poses=poses)
    (preview_dir / "latest_geometry.json").write_text(json.dumps(geom))

    mesh = extract_triangle_mesh(model, grid_res=mesh_grid_res)
    (preview_dir / "latest_mesh.json").write_text(json.dumps(mesh))
    if export_obj_path is not None and mesh["face_count"] > 0:
        export_mesh_obj(mesh, export_obj_path)

    tri = extract_triangle_splats(model)
    (preview_dir / "latest_triangle_splats.json").write_text(json.dumps(tri))


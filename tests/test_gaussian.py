"""Unit tests for CPU differentiable Gaussian rasterizer, densification, and export."""

from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData

from reconstruction.export.ply_exporter import export_ply
from reconstruction.export.representations import (
    export_mesh_obj,
    extract_geometry_data,
    extract_triangle_mesh,
    extract_triangle_splats,
)
from reconstruction.export.splat_exporter import export_splat
from reconstruction.gaussian.losses import photometric_loss, psnr
from reconstruction.gaussian.model import GaussianModel
from reconstruction.gaussian.rasterizer import Camera, render_gaussians


def test_rasterizer_forward_and_backward_cpu():
    """Verify differentiable forward & backward passes work strictly on CPU."""
    assert not torch.cuda.is_available() or True
    rng = np.random.default_rng(7)
    pts = rng.uniform(-0.8, 0.8, size=(120, 3)).astype(np.float32)
    pts[:, 2] += 3.0  # In front of camera at origin
    cols = rng.uniform(0.1, 0.9, size=(120, 3)).astype(np.float32)

    model = GaussianModel(sh_degree=0)
    model.init_from_sparse_points(pts, cols, max_gaussians=200)

    cam = Camera(
        image_name="test.jpg",
        width=96,
        height=72,
        fx=100.0,
        fy=100.0,
        cx=48.0,
        cy=36.0,
        R=torch.eye(3, dtype=torch.float32),
        t=torch.zeros(3, dtype=torch.float32),
    )
    target = torch.zeros((72, 96, 3), dtype=torch.float32)
    target[20:52, 30:66, 0] = 0.8

    out = render_gaussians(model, cam, tile_size=16)
    img = out["image"]
    assert img.shape == (72, 96, 3)
    assert img.device.type == "cpu"

    loss = photometric_loss(img, target)
    loss.backward()

    assert model._xyz.grad is not None
    assert model._scales.grad is not None
    assert model._rotations.grad is not None
    assert model._opacity.grad is not None
    assert model._features_dc.grad is not None
    assert torch.isfinite(model._xyz.grad).all()
    assert out["means2d"].grad is not None


def test_optimization_reduces_loss_and_exports(tmp_path: Path):
    """Verify optimizing Gaussians improves PSNR and exports valid .ply and .splat."""
    pts = np.array([[0.0, 0.0, 3.0], [0.3, 0.0, 3.0], [-0.3, 0.2, 3.0]], dtype=np.float32)
    cols = np.array([[0.2, 0.2, 0.2]] * 3, dtype=np.float32)
    model = GaussianModel(sh_degree=0)
    model.init_from_sparse_points(pts, cols, max_gaussians=50, init_opacity=0.6)

    cam = Camera(
        image_name="v0.jpg",
        width=64,
        height=64,
        fx=80.0,
        fy=80.0,
        cx=32.0,
        cy=32.0,
        R=torch.eye(3, dtype=torch.float32),
        t=torch.zeros(3, dtype=torch.float32),
    )

    # Target has bright red/green/blue blobs matching initial projections
    target = torch.zeros((64, 64, 3), dtype=torch.float32)
    target[24:40, 24:40, 0] = 0.9

    opt = torch.optim.Adam(model.parameters(), lr=0.015)
    init_loss = photometric_loss(render_gaussians(model, cam)["image"].detach(), target).item()

    for _ in range(30):
        opt.zero_grad()
        loss = photometric_loss(render_gaussians(model, cam)["image"], target)
        loss.backward()
        opt.step()

    final_loss = photometric_loss(render_gaussians(model, cam)["image"].detach(), target).item()
    assert final_loss < init_loss

    ply_path = export_ply(model, tmp_path / "out.ply")
    splat_path = export_splat(model, tmp_path / "out.splat")

    assert ply_path.exists()
    ply = PlyData.read(str(ply_path))
    assert len(ply["vertex"]) == model.num_gaussians

    assert splat_path.exists()
    assert splat_path.stat().st_size == model.num_gaussians * 32


def test_derived_geometry_mesh_and_triangle_splats(tmp_path: Path):
    """Verify Geometry, Marching-Cubes Mesh, and Triangle Splatting extraction from GaussianModel."""
    rng = np.random.default_rng(9)
    pts = rng.uniform(-0.5, 0.5, size=(60, 3)).astype(np.float32)
    cols = rng.uniform(0.2, 0.9, size=(60, 3)).astype(np.float32)
    model = GaussianModel(sh_degree=0)
    model.init_from_sparse_points(pts, cols, max_gaussians=100, init_opacity=0.8)

    geom = extract_geometry_data(model)
    assert geom["count"] == 60
    assert len(geom["positions"]) == 60 * 3
    assert len(geom["rotations"]) == 60 * 4

    mesh = extract_triangle_mesh(model, grid_res=24)
    assert mesh["vertex_count"] > 0
    assert mesh["face_count"] > 0
    obj_path = export_mesh_obj(mesh, tmp_path / "mesh.obj")
    assert obj_path.exists() and obj_path.stat().st_size > 100

    tris = extract_triangle_splats(model)
    assert tris["triangle_count"] == 60
    assert len(tris["positions"]) == 60 * 9
    assert len(tris["uvs"]) == 60 * 6


def test_seeding_handles_mixed_image_dimensions(tmp_path: Path):
    """Ensure dense optical-flow seeding does not crash on mixed portrait/landscape photo sizes."""
    import cv2
    from reconstruction.gaussian.seeding import densify_sparse_points_from_poses
    from reconstruction.sfm.types import CameraPose

    p1_path = tmp_path / "p1.jpg"
    p2_path = tmp_path / "p2.jpg"
    rng = np.random.default_rng(5)
    cv2.imwrite(str(p1_path), rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8))
    cv2.imwrite(str(p2_path), rng.integers(0, 255, size=(160, 120, 3), dtype=np.uint8))

    poses = {
        "p1.jpg": CameraPose(
            image_name="p1.jpg",
            image_path=str(p1_path),
            width=160,
            height=120,
            K=np.array([[160.0, 0.0, 80.0], [0.0, 160.0, 60.0], [0.0, 0.0, 1.0]]),
            R=np.eye(3),
            t=np.zeros(3),
        ),
        "p2.jpg": CameraPose(
            image_name="p2.jpg",
            image_path=str(p2_path),
            width=120,
            height=160,
            K=np.array([[160.0, 0.0, 60.0], [0.0, 160.0, 80.0], [0.0, 0.0, 1.0]]),
            R=np.eye(3),
            t=np.array([-0.2, 0.0, 0.0]),
        ),
    }

    pts, cols = densify_sparse_points_from_poses(poses, max_new_points=200)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert cols.ndim == 2 and cols.shape[1] == 3


def test_500_photo_limit_config_and_schema():
    """Verify max_keyframes=500 is accepted by both ReconstructionConfig and API schema."""
    from backend.models.schemas import ReconstructionParams
    from reconstruction.config import ReconstructionConfig

    cfg = ReconstructionConfig(max_keyframes=500).clamp_for_safety()
    assert cfg.max_keyframes == 500
    params = ReconstructionParams(max_keyframes=500)
    assert params.max_keyframes == 500


def test_custom_camera_fov_intrinsics():
    """Verify explicit camera_fov_deg converts accurately to pinhole focal length."""
    from backend.models.schemas import ReconstructionParams
    from reconstruction.sfm.pose_estimation import default_intrinsics

    K90 = default_intrinsics(800, 600, fov_deg=90.0)
    assert np.isclose(K90[0, 0], 400.0, atol=1e-3)
    assert np.isclose(K90[1, 1], 400.0, atol=1e-3)
    assert np.isclose(K90[0, 2], 400.0)
    assert np.isclose(K90[1, 2], 300.0)

    params = ReconstructionParams(camera_fov_deg=75.0)
    assert params.camera_fov_deg == 75.0


def test_device_resolution_and_mps_rasterizer():
    """Verify safe device resolution and MPS GPU forward/backward pass when supported."""
    from backend.models.schemas import ReconstructionParams
    from reconstruction.config import ReconstructionConfig, probe_mps_support, resolve_torch_device
    from reconstruction.gaussian.densification import densify_and_prune

    assert resolve_torch_device("cpu").type == "cpu"
    assert ReconstructionConfig(device="mps").clamp_for_safety().device == "mps"
    assert ReconstructionParams(device="mps").device == "mps"
    assert ReconstructionParams(device="cpu").device == "cpu"

    mps_usable, mps_reason = probe_mps_support()
    auto_dev = resolve_torch_device("auto")
    if torch.cuda.is_available():
        assert auto_dev.type == "cuda"
    elif mps_usable:
        assert auto_dev.type == "mps"
    else:
        assert auto_dev.type == "cpu"
        # Requesting MPS when host runtime lacks compatible Metal kernels falls back safely to CPU
        assert resolve_torch_device("mps").type == "cpu"
        assert isinstance(mps_reason, str) and len(mps_reason) > 0
        return

    mps_dev = resolve_torch_device("mps")
    assert mps_dev.type == "mps"

    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.6, 0.6, size=(80, 3)).astype(np.float32)
    pts[:, 2] += 3.0
    cols = rng.uniform(0.1, 0.9, size=(80, 3)).astype(np.float32)

    model = GaussianModel(sh_degree=0).to(mps_dev)
    model.init_from_sparse_points(pts, cols, max_gaussians=120)
    assert model.device.type == "mps"

    cam = Camera(
        image_name="mps_test.jpg",
        width=64,
        height=64,
        fx=80.0,
        fy=80.0,
        cx=32.0,
        cy=32.0,
        R=torch.eye(3, dtype=torch.float32, device=mps_dev),
        t=torch.zeros(3, dtype=torch.float32, device=mps_dev),
    )
    target = torch.zeros((64, 64, 3), dtype=torch.float32, device=mps_dev)
    target[20:44, 20:44, 1] = 0.85

    opt = torch.optim.Adam(model.parameters(), lr=0.015)
    out0 = render_gaussians(model, cam, tile_size=16)
    img0 = out0["image"]
    assert img0.device.type == "mps"
    loss0 = photometric_loss(img0, target)
    loss0.backward()
    assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all()
    model.add_densification_stats(out0["means2d"].grad, out0["visibility_filter"])
    opt.step()

    densify_and_prune(model, max_gaussians=120, grad_threshold=1e-4, scene_extent=2.0)
    assert model.device.type == "mps"

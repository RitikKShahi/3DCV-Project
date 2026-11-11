"""End-to-end tests for video preprocessing, incremental SfM, and full CPU 3DGS pipeline."""

import math
from pathlib import Path

import cv2
import numpy as np

from reconstruction.config import ReconstructionConfig
from reconstruction.pipeline import ReconstructionPipeline


def _make_rich_texture(size: int = 256, seed: int = 42) -> np.ndarray:
    """Create a multi-scale high-frequency textured pattern with rich SIFT features."""
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    # Base gradient + checkerboard blocks + random textured shapes + crisp text/corners
    for y in range(0, size, 32):
        for x in range(0, size, 32):
            col = rng.integers(50, 230, size=3).tolist()
            cv2.rectangle(tex, (x, y), (x + 31, y + 31), col, -1)
    for _ in range(180):
        p1 = tuple(rng.integers(0, size, size=2).tolist())
        p2 = tuple(np.clip(np.array(p1) + rng.integers(-28, 29, size=2), 0, size - 1).tolist())
        col = rng.integers(20, 255, size=3).tolist()
        if rng.random() < 0.5:
            cv2.line(tex, p1, p2, col, int(rng.integers(1, 4)))
        else:
            cv2.rectangle(tex, p1, p2, col, int(rng.integers(1, 3)))
    return tex


def _generate_textured_orbit_video(video_path: Path, num_frames: int = 36, width: int = 320, height: int = 240) -> None:
    """Render a 3D textured scene (cube + floor) with an orbiting camera using perspective warps."""
    tex_size = 256
    textures = [
        _make_rich_texture(tex_size, seed=10),  # Front face
        _make_rich_texture(tex_size, seed=20),  # Right face
        _make_rich_texture(tex_size, seed=30),  # Left face
        _make_rich_texture(tex_size, seed=40),  # Floor plane
    ]

    # Define 4 planar quads in 3D world space: [top-left, top-right, bottom-right, bottom-left]
    s = 0.75
    quads = [
        np.array([[-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s]], dtype=np.float64),   # Front (-Z)
        np.array([[s, -s, -s], [s, -s, s], [s, s, s], [s, s, -s]], dtype=np.float64),       # Right (+X)
        np.array([[-s, -s, s], [-s, -s, -s], [-s, s, -s], [-s, s, s]], dtype=np.float64),   # Left (-X)
        np.array([[-2.5, s, 2.5], [2.5, s, 2.5], [2.5, s, -2.5], [-2.5, s, -2.5]], dtype=np.float64),  # Floor (+Y)
    ]
    src_pts = np.array([[0, 0], [tex_size - 1, 0], [tex_size - 1, tex_size - 1], [0, tex_size - 1]], dtype=np.float32)

    fx = fy = float(max(width, height))
    cx, cy = width * 0.5, height * 0.5
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (width, height))

    for i in range(num_frames):
        theta = 0.75 * (i / max(1, num_frames - 1)) - 0.375
        cam_pos = np.array([3.0 * math.sin(theta), -0.35 - 0.15 * math.sin(theta * 2.0), -3.0 * math.cos(theta)])
        target = np.array([0.0, 0.1, 0.0])
        forward = (target - cam_pos)
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)
        R_cw = np.stack([right, true_up, forward], axis=0)
        t_cw = -R_cw @ cam_pos

        canvas = np.full((height, width, 3), 22, dtype=np.uint8)
        # Render back-to-front by quad center depth
        quad_order = sorted(
            range(len(quads)),
            key=lambda qi: -float(np.mean((R_cw @ quads[qi].T + t_cw.reshape(3, 1))[2, :])),
        )
        for qi in quad_order:
            q3d = quads[qi]
            cam_q = (R_cw @ q3d.T + t_cw.reshape(3, 1)).T
            if np.any(cam_q[:, 2] <= 0.2):
                continue
            # Back-face culling
            v1 = cam_q[1] - cam_q[0]
            v2 = cam_q[3] - cam_q[0]
            normal = np.cross(v1, v2)
            if np.dot(normal, cam_q[0]) >= 0 and qi != 3:
                continue

            proj = (K @ cam_q.T).T
            dst_pts = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(textures[qi], H, (width, height), flags=cv2.INTER_LINEAR)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst_pts).astype(np.int32), 255)
            canvas[mask > 0] = warped[mask > 0]

        writer.write(canvas)

    writer.release()


def test_full_incremental_pipeline_on_synthetic_video(tmp_path: Path):
    """Run full video -> preprocessing -> incremental SfM -> 3DGS -> .ply/.splat on CPU."""
    video_path = tmp_path / "orbit.mp4"
    _generate_textured_orbit_video(video_path, num_frames=36, width=320, height=240)

    cfg = ReconstructionConfig(
        max_resolution=320,
        frame_interval=3,
        max_keyframes=10,
        initial_batch_size=5,
        incremental_batch_size=3,
        max_gaussians=600,
        iterations_per_batch=25,
        final_iterations=20,
        sfm_backend="auto",
        blur_threshold=5.0,
        min_frame_diff=0.005,
        adaptive_early_stop=False,
    )

    events = []
    pipeline = ReconstructionPipeline(cfg)
    result = pipeline.run(
        video_path=video_path,
        work_dir=tmp_path / "work",
        progress_callback=events.append,
        resume=False,
    )

    assert result.ply_path.exists()
    assert result.splat_path.exists()
    assert result.num_gaussians > 10
    assert result.num_registered_cameras >= 4
    assert len(result.batch_metrics) >= 2  # Incremental multi-batch verified
    assert any(e.stage == "incremental_preview" for e in events)
    assert events[-1].stage == "completed"
    assert (tmp_path / "work" / "previews" / "preview_batch_0.splat").exists()
    assert (tmp_path / "work" / "checkpoint" / "state.json").exists()

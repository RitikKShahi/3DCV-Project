"""CPU Performance & Quality Benchmark Suite for Incremental 3DGS Reconstruction."""

import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np

from reconstruction.config import ReconstructionConfig
from reconstruction.pipeline import ReconstructionPipeline
from tests.test_pipeline import _make_rich_texture


def generate_scene_video(
    out_path: Path,
    scene_type: str,
    num_frames: int = 36,
    width: int = 320,
    height: int = 240,
) -> None:
    """Generate distinct realistic textured 3D test videos (object, indoor, outdoor)."""
    tex_size = 256
    t0 = _make_rich_texture(tex_size, seed=101 if scene_type == "object" else 202)
    t1 = _make_rich_texture(tex_size, seed=303 if scene_type == "indoor" else 404)
    t2 = _make_rich_texture(tex_size, seed=505)
    t3 = _make_rich_texture(tex_size, seed=606)

    if scene_type == "indoor":
        # Two room walls + floor corner
        quads = [
            np.array([[-2.0, -1.2, 0.0], [0.0, -1.2, 0.0], [0.0, 1.0, 0.0], [-2.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, -1.2, 0.0], [2.0, -1.2, 1.2], [2.0, 1.0, 1.2], [0.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[-2.2, 1.0, 2.2], [2.2, 1.0, 2.2], [2.2, 1.0, -0.5], [-2.2, 1.0, -0.5]], dtype=np.float64),
        ]
        texs = [t0, t1, t2]
        cam_dist = 3.2
    elif scene_type == "outdoor":
        # Wide building facade + courtyard ground + side structure
        quads = [
            np.array([[-2.5, -1.5, 0.5], [2.5, -1.5, 0.5], [2.5, 1.0, 0.5], [-2.5, 1.0, 0.5]], dtype=np.float64),
            np.array([[-2.8, 1.0, 2.5], [2.8, 1.0, 2.5], [2.8, 1.0, -2.5], [-2.8, 1.0, -2.5]], dtype=np.float64),
            np.array([[-0.6, -0.6, -0.8], [0.6, -0.6, -0.8], [0.6, 1.0, -0.8], [-0.6, 1.0, -0.8]], dtype=np.float64),
        ]
        texs = [t1, t3, t0]
        cam_dist = 3.6
    else:
        # Object orbit (textured multi-face cube + turntable floor)
        s = 0.75
        quads = [
            np.array([[-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s]], dtype=np.float64),
            np.array([[s, -s, -s], [s, -s, s], [s, s, s], [s, s, -s]], dtype=np.float64),
            np.array([[-s, -s, s], [-s, -s, -s], [-s, s, -s], [-s, s, s]], dtype=np.float64),
            np.array([[-2.4, s, 2.4], [2.4, s, 2.4], [2.4, s, -2.4], [-2.4, s, -2.4]], dtype=np.float64),
        ]
        texs = [t0, t1, t2, t3]
        cam_dist = 3.0

    src_pts = np.array([[0, 0], [tex_size - 1, 0], [tex_size - 1, tex_size - 1], [0, tex_size - 1]], dtype=np.float32)
    fx = fy = float(max(width, height))
    cx, cy = width * 0.5, height * 0.5
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 15.0, (width, height))

    for i in range(num_frames):
        theta = 0.75 * (i / max(1, num_frames - 1)) - 0.375
        cam_pos = np.array([cam_dist * math.sin(theta), -0.35 - 0.1 * math.sin(theta * 2.0), -cam_dist * math.cos(theta)])
        target = np.array([0.0, 0.1, 0.0])
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        true_up = np.cross(forward, right)
        R_cw = np.stack([right, true_up, forward], axis=0)
        t_cw = -R_cw @ cam_pos

        canvas = np.full((height, width, 3), 24, dtype=np.uint8)
        quad_order = sorted(
            range(len(quads)),
            key=lambda qi: -float(np.mean((R_cw @ quads[qi].T + t_cw.reshape(3, 1))[2, :])),
        )
        for qi in quad_order:
            q3d = quads[qi]
            cam_q = (R_cw @ q3d.T + t_cw.reshape(3, 1)).T
            if np.any(cam_q[:, 2] <= 0.2):
                continue
            proj = (K @ cam_q.T).T
            dst_pts = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(texs[qi], H, (width, height), flags=cv2.INTER_LINEAR)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.round(dst_pts).astype(np.int32), 255)
            canvas[mask > 0] = warped[mask > 0]

        writer.write(canvas)
    writer.release()


def run_benchmarks() -> None:
    scenarios = [
        ("object", 320, 12, 1500, 60, 40),
        ("indoor", 320, 12, 1500, 60, 40),
        ("outdoor", 320, 12, 1500, 60, 40),
    ]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name, res, kf, max_g, iters, final_iters in scenarios:
            vid = tmp_dir / f"{name}.mp4"
            generate_scene_video(vid, name, num_frames=36, width=res, height=int(res * 0.75))
            cfg = ReconstructionConfig(
                max_resolution=res,
                frame_interval=3,
                max_keyframes=kf,
                initial_batch_size=6,
                incremental_batch_size=3,
                max_gaussians=max_g,
                iterations_per_batch=iters,
                final_iterations=final_iters,
                blur_threshold=5.0,
                min_frame_diff=0.005,
                adaptive_early_stop=False,
            )
            pipe = ReconstructionPipeline(cfg)
            out = pipe.run(vid, tmp_dir / f"work_{name}", resume=False)
            row = {
                "scene": name,
                "resolution": f"{res}x{int(res*0.75)}",
                "keyframes": out.num_keyframes,
                "cameras": out.num_registered_cameras,
                "gaussians": out.num_gaussians,
                "sfm_backend": out.sfm_backend,
                "psnr_db": round(out.final_psnr, 2),
                "time_sec": round(out.total_time_sec, 2),
                "peak_rss_mb": round(out.peak_memory_mb, 1),
            }
            results.append(row)
            print(json.dumps(row))

    out_file = Path("benchmarks_results.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved benchmarks to {out_file}")


if __name__ == "__main__":
    run_benchmarks()


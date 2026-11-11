"""Isolated pycolmap subprocess worker (avoids libomp conflict with PyTorch on macOS).

This module MUST NOT import torch.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap


def _export_reconstruction(
    rec: pycolmap.Reconstruction,
    image_dir: Path,
    out_json: Path,
    out_npz: Path,
) -> None:
    """Serialize pycolmap Reconstruction poses & 3D points to JSON + NPZ."""
    poses = {}
    for img_id, img in rec.images.items():
        if not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t = np.asarray(cfw.translation, dtype=np.float64).reshape(3)
        poses[img.name] = {
            "image_name": img.name,
            "image_path": str(image_dir / img.name),
            "width": int(cam.width),
            "height": int(cam.height),
            "K": K.tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
        }

    pts = []
    cols = []
    pids = []
    for pid, p3d in rec.points3D.items():
        pts.append(p3d.xyz)
        cols.append(np.asarray(p3d.color, dtype=np.float32) / 255.0)
        pids.append(int(pid))

    points_arr = np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
    colors_arr = np.asarray(cols, dtype=np.float32) if cols else np.zeros((0, 3), dtype=np.float32)
    pids_arr = np.asarray(pids, dtype=np.int64)

    np.savez(out_npz, points=points_arr, colors=colors_arr, point_ids=pids_arr)

    mean_err = 0.0
    try:
        if rec.num_points3D() > 0:
            mean_err = float(rec.compute_mean_reprojection_error())
    except Exception:
        pass

    payload = {
        "num_reg_images": int(rec.num_reg_images()),
        "num_points3D": int(rec.num_points3D()),
        "mean_reprojection_error": mean_err,
        "poses": poses,
    }
    out_json.write_text(json.dumps(payload, indent=2))


def run_worker(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    image_dir = Path(args.image_dir)
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_names = args.images

    use_cuda = bool(getattr(args, "use_gpu", False) and getattr(pycolmap, "has_cuda", False))
    colmap_device = pycolmap.Device.cuda if use_cuda else pycolmap.Device.cpu

    # Configure extraction options (sensitive SIFT threshold for indoor/low-texture scenes)
    feo = pycolmap.FeatureExtractionOptions()
    feo.use_gpu = use_cuda
    feo.num_threads = args.threads
    feo.sift.max_num_features = args.max_features
    feo.sift.peak_threshold = 0.004

    # Detect whether batch images share uniform dimensions (required for CameraMode.SINGLE)
    from PIL import Image

    batch_dims: set[tuple[int, int]] = set()
    for name in image_names:
        try:
            with Image.open(image_dir / name) as im:
                batch_dims.add((int(im.width), int(im.height)))
        except Exception:
            pass

    uniform_dims = len(batch_dims) == 1
    camera_mode = pycolmap.CameraMode.SINGLE if uniform_dims else pycolmap.CameraMode.AUTO

    input_model_dir = workspace / "current_model"
    iro = pycolmap.ImageReaderOptions()
    if getattr(args, "fov_deg", 0.0) > 0.0:
        half_rad = math.radians(max(20.0, min(140.0, float(args.fov_deg)))) * 0.5
        iro.default_focal_length_factor = 0.5 / math.tan(half_rad)
    if uniform_dims and args.incremental and input_model_dir.exists() and (input_model_dir / "cameras.bin").exists():
        try:
            prev_rec = pycolmap.Reconstruction(input_model_dir)
            if prev_rec.cameras:
                cam_id = next(iter(prev_rec.cameras.keys()))
                cam = prev_rec.cameras[cam_id]
                if (int(cam.width), int(cam.height)) == next(iter(batch_dims)):
                    iro.existing_camera_id = cam_id
        except Exception:
            pass

    # Extract features for the requested image batch
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        image_names=image_names,
        camera_mode=camera_mode,
        reader_options=iro,
        extraction_options=feo,
        device=colmap_device,
    )

    # Configure matching options
    fmo = pycolmap.FeatureMatchingOptions()
    fmo.use_gpu = use_cuda
    fmo.num_threads = args.threads

    # Exhaustive matching for <= 120 total images; sequential overlap matching for large sets (up to 500)
    import sqlite3

    total_db_images = len(image_names)
    try:
        with sqlite3.connect(str(db_path)) as con:
            row = con.execute("SELECT COUNT(*) FROM images").fetchone()
            if row:
                total_db_images = int(row[0])
    except Exception:
        pass

    if total_db_images <= 120:
        pycolmap.match_exhaustive(
            database_path=db_path,
            matching_options=fmo,
            device=colmap_device,
        )
    else:
        spo = pycolmap.SequentialPairingOptions()
        spo.overlap = 20
        spo.quadratic_overlap = True
        pycolmap.match_sequential(
            database_path=db_path,
            matching_options=fmo,
            pairing_options=spo,
            device=colmap_device,
        )

    ipo = pycolmap.IncrementalPipelineOptions()
    ipo.ba_use_gpu = use_cuda
    ipo.ba_global_max_num_iterations = 25
    ipo.ba_local_max_num_iterations = 15
    ipo.min_model_size = 3
    ipo.mapper.init_min_num_inliers = 25
    ipo.mapper.init_min_tri_angle = 8.0
    ipo.mapper.abs_pose_min_num_inliers = 10
    ipo.mapper.abs_pose_min_inlier_ratio = 0.08

    batch_out_dir = workspace / f"batch_{args.batch_index}"
    if batch_out_dir.exists():
        shutil.rmtree(batch_out_dir)
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    input_path = ""
    if args.incremental and input_model_dir.exists() and (input_model_dir / "images.bin").exists():
        input_path = str(input_model_dir)
        # Keep existing registered frames & intrinsics fixed to preserve world coordinate system
        ipo.fix_existing_frames = True
        ipo.ba_refine_focal_length = False
        ipo.ba_refine_extra_params = False

    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=batch_out_dir,
        options=ipo,
        input_path=input_path,
    )

    if not maps:
        print("ERROR: pycolmap produced 0 reconstructions", file=sys.stderr)
        return 2

    # Select reconstruction with the most registered images
    best_rec = max(maps.values(), key=lambda r: r.num_reg_images())

    # Persist as current_model for future incremental batches
    if input_model_dir.exists():
        shutil.rmtree(input_model_dir)
    input_model_dir.mkdir(parents=True, exist_ok=True)
    best_rec.write(input_model_dir)

    _export_reconstruction(
        best_rec,
        image_dir=image_dir,
        out_json=Path(args.out_json),
        out_npz=Path(args.out_npz),
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only pycolmap SfM worker")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--fov-deg", type=float, default=0.0)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npz", required=True)
    args = parser.parse_args()
    sys.exit(run_worker(args))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Automated Monocular Video & Photo Preprocessing Pipeline for 3D Gaussian Splatting
Prepares raw unposed photos and single-camera video recordings into calibrated COLMAP datasets:
1. Video frame extraction via FFmpeg (configurable FPS)
2. Scales images (No Scaling, Half, or 1600k longest dimension, auto-converts HEIC/WebP)
3. Runs COLMAP database_creator, feature_extractor, matcher (Sequential/Exhaustive/Spatial)
4. Robust mapper with video-tuned triangulation & registration thresholds
5. Automatic selection and promotion of largest submodel (maximum registered cameras)
6. Packages into standard dataset.zip with images/ and sparse/0/
"""

import sys
import os
import shutil
import subprocess
import zipfile
import json
import argparse
import struct
from pathlib import Path

COLMAP_BIN = os.environ.get("COLMAP_BIN", "/Users/ritikkumarshahi/colmap_bin/bin/colmap")

def emit(event_type, **kwargs):
    payload = {"type": event_type, **kwargs}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

def extract_video_frames(video_path, output_dir, fps=5.0):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    emit("log", message=f"Extracting video frames from '{video_name}' at {fps} FPS using FFmpeg...")
    emit("progress", step="video_extract", percent=3, message=f"Extracting keyframes from video ({fps} FPS)...")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg")
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with code {res.returncode}: {res.stderr}")
    except Exception as e:
        raise RuntimeError(f"FFmpeg frame extraction failed: {str(e)}")

    frames = sorted([f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")])
    if not frames:
        raise RuntimeError(f"FFmpeg did not extract any frames from '{video_name}'")
    
    emit("log", message=f"Extracted {len(frames)} keyframes successfully from video '{video_name}'.")
    return len(frames)

def scale_images(input_images_dir, output_images_dir, scaling_option):
    os.makedirs(output_images_dir, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.heic', '.webp')
    files = sorted([
        f for f in os.listdir(input_images_dir)
        if f.lower().endswith(valid_exts)
    ])
    
    total = len(files)
    emit("log", message=f"Starting image preparation for {total} images (Scaling: {scaling_option})...")
    emit("progress", step="scaling", percent=5, message=f"Processing images (0/{total})")

    for idx, filename in enumerate(files):
        src = os.path.join(input_images_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        # Standardize non-JPG outputs to .jpg
        if ext in ['.jpeg', '.tif', '.tiff', '.bmp', '.heic', '.webp']:
            out_name = os.path.splitext(filename)[0] + ".jpg"
        else:
            out_name = filename
        dst = os.path.join(output_images_dir, out_name)

        # Convert HEIC or WebP to JPG directly if needed
        if ext in ['.heic', '.webp']:
            subprocess.run(["sips", "-s", "format", "jpeg", src, "--out", dst],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            src_for_scale = dst
        else:
            src_for_scale = src

        if scaling_option == "No Scaling":
            if src_for_scale != dst:
                shutil.copy2(src_for_scale, dst)
            elif not os.path.exists(dst):
                shutil.copy2(src, dst)
        elif scaling_option in ["1600k", "1600", "Optimal"]:
            # Resize longest dimension to 1600px with native macOS sips
            subprocess.run(["sips", "-Z", "1600", src_for_scale, "--out", dst],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not os.path.exists(dst):
                shutil.copy2(src_for_scale, dst)
        elif scaling_option in ["Half", "50%"]:
            try:
                out = subprocess.check_output(["sips", "-g", "pixelWidth", "-g", "pixelHeight", src_for_scale], text=True)
                w, h = 0, 0
                for line in out.splitlines():
                    if "pixelWidth:" in line:
                        w = int(line.split(":")[1].strip())
                    elif "pixelHeight:" in line:
                        h = int(line.split(":")[1].strip())
                if w > 0 and h > 0:
                    new_h, new_w = max(1, h // 2), max(1, w // 2)
                    subprocess.run(["sips", "--resampleHeightWidth", str(new_h), str(new_w), src_for_scale, "--out", dst],
                                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    shutil.copy2(src_for_scale, dst)
            except Exception:
                shutil.copy2(src_for_scale, dst)
        else:
            shutil.copy2(src_for_scale, dst)

        pct = int(5 + (idx + 1) / total * 15)
        emit("progress", step="scaling", percent=pct, message=f"Prepared image {idx+1}/{total}: {out_name}")

    emit("log", message=f"Image preparation complete: {len(os.listdir(output_images_dir))} images ready.")

def run_cmd(cmd, step_name, start_pct, end_pct):
    emit("log", message=f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in iter(process.stdout.readline, ''):
        line_clean = line.strip()
        if not line_clean:
            continue
        emit("log", message=line_clean)
        if "Processed file" in line_clean or "Matching block" in line_clean or "Registering image" in line_clean:
            emit("progress", step=step_name, percent=int((start_pct + end_pct) / 2), message=line_clean[:80])
            
    process.stdout.close()
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}: {' '.join(cmd)}")

def run_pipeline(workspace_dir, scaling="1600k", matching="Sequential", camera_model="SIMPLE_RADIAL", output_zip=None, video_path=None, fps=5.0):
    workspace = Path(workspace_dir).resolve()
    raw_images_dir = workspace / "raw_images"
    images_dir = workspace / "images"
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    
    if not os.path.exists(COLMAP_BIN):
        raise FileNotFoundError(f"COLMAP binary not found at: {COLMAP_BIN}")

    emit("log", message=f"Workspace initialized at: {workspace}")
    
    # 0. If video is provided, extract frames with FFmpeg
    if video_path and os.path.exists(video_path):
        extract_video_frames(video_path, str(raw_images_dir), fps)

    # 1. Scale images
    scale_images(str(raw_images_dir), str(images_dir), scaling)

    # 2. COLMAP: Database creator
    emit("progress", step="db", percent=22, message="Creating COLMAP database...")
    run_cmd([COLMAP_BIN, "database_creator", "--database_path", str(db_path)], "db", 22, 25)

    # 3. COLMAP: Feature extraction
    emit("progress", step="features", percent=26, message="Extracting SIFT GPU features from viewpoints...")
    feature_cmd = [
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", camera_model,
    ]
    # For video, sharing camera intrinsics can help; for general photos, single_camera 0 is safer
    if video_path:
        feature_cmd.extend(["--ImageReader.single_camera", "1"])
    run_cmd(feature_cmd, "features", 26, 50)

    # 4. COLMAP: Feature matching
    emit("progress", step="matching", percent=52, message=f"Matching features using {matching} matcher...")
    if matching.lower() == "exhaustive":
        matcher_cmd = [COLMAP_BIN, "exhaustive_matcher", "--database_path", str(db_path)]
    elif matching.lower() == "spatial":
        matcher_cmd = [COLMAP_BIN, "spatial_matcher", "--database_path", str(db_path)]
    else:  # Sequential (default & optimal for video / ordered photos)
        matcher_cmd = [
            COLMAP_BIN, "sequential_matcher",
            "--database_path", str(db_path),
            "--SequentialMatching.overlap", "15"
        ]
    run_cmd(matcher_cmd, "matching", 52, 70)

    # 5. COLMAP: Sparse Mapper (Robust handheld & video reconstruction)
    emit("progress", step="mapper", percent=72, message="Running SfM reconstruction & point-cloud triangulation...")
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_tri_angle", "4.0",
        "--Mapper.init_min_num_inliers", "25",
        "--Mapper.abs_pose_min_num_inliers", "12",
        "--Mapper.abs_pose_min_inlier_ratio", "0.12",
        "--Mapper.filter_min_tri_angle", "1.0",
        "--Mapper.min_model_size", "3",
        "--Mapper.max_reg_trials", "5"
    ]
    run_cmd(mapper_cmd, "mapper", 72, 88)

    # 6. Verify and select the BEST model (the one with the MAXIMUM registered cameras!)
    submodels = []
    for sub in sparse_dir.iterdir():
        if sub.is_dir() and sub.name != "sparse_bin":
            img_bin = sub / "images.bin"
            img_txt = sub / "images.txt"
            count = 0
            if img_bin.exists():
                try:
                    with open(img_bin, "rb") as f:
                        count = struct.unpack("<Q", f.read(8))[0]
                except Exception:
                    count = 0
            elif img_txt.exists():
                try:
                    with open(img_txt, "r") as f:
                        count = sum(1 for line in f if line.strip() and not line.startswith("#")) // 2
                except Exception:
                    count = 0
            submodels.append((count, sub))

    if not submodels:
        # Check if output is directly in sparse_dir
        if (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists():
            tmp_0 = workspace / "tmp_0"
            tmp_0.mkdir(parents=True, exist_ok=True)
            for f in sparse_dir.glob("*"):
                if f.is_file():
                    shutil.move(str(f), str(tmp_0 / f.name))
            sub0 = sparse_dir / "0"
            sub0.mkdir(parents=True, exist_ok=True)
            for f in tmp_0.glob("*"):
                shutil.move(str(f), str(sub0 / f.name))
            shutil.rmtree(tmp_0, ignore_errors=True)
            submodels = [(1, sub0)]
        else:
            raise RuntimeError("COLMAP could not find enough overlapping feature matches to reconstruct a 3D model.")

    # Sort descending by number of registered images
    submodels.sort(key=lambda x: x[0], reverse=True)
    best_count, best_dir = submodels[0]
    total_input_images = len(os.listdir(images_dir))
    emit("log", message=f"COLMAP generated {len(submodels)} submodel(s). Optimal submodel '{best_dir.name}' has {best_count} registered cameras (out of {total_input_images} frames).")

    # Ensure best model is located at sparse/0
    final_sub0 = sparse_dir / "0"
    if best_dir != final_sub0:
        temp_best = workspace / "temp_best_model"
        shutil.copytree(best_dir, temp_best)
        shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        final_sub0 = sparse_dir / "0"
        shutil.copytree(temp_best, final_sub0)
        shutil.rmtree(temp_best, ignore_errors=True)
    else:
        # Clean other submodels to avoid confusing downstream viewers
        for count, sub in submodels[1:]:
            if sub.exists():
                shutil.rmtree(sub, ignore_errors=True)

    # Convert to binary if text format was generated
    if (final_sub0 / "cameras.txt").exists() and not (final_sub0 / "cameras.bin").exists():
        emit("log", message="Converting COLMAP text format to binary .bin...")
        bin_dir = workspace / "sparse_bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        conv_cmd = [
            COLMAP_BIN, "model_converter",
            "--input_path", str(final_sub0),
            "--output_path", str(bin_dir),
            "--output_type", "BIN"
        ]
        run_cmd(conv_cmd, "convert", 88, 90)
        shutil.rmtree(final_sub0)
        shutil.move(str(bin_dir), str(final_sub0))

    expected_files = ["cameras.bin", "images.bin", "points3D.bin"]
    found = [f for f in expected_files if (final_sub0 / f).exists()]
    if len(found) < 3:
        emit("warning", message=f"Partial reconstruction: found {found} in {final_sub0}")

    # 7. Package ZIP archive for 3D Gaussian Splatting / Brush
    emit("progress", step="packaging", percent=92, message="Packaging COLMAP dataset ZIP archive...")
    if not output_zip:
        output_zip = workspace / "dataset.zip"
    else:
        output_zip = Path(output_zip).resolve()

    with zipfile.ZipFile(str(output_zip), 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Add images
        for root, _, files in os.walk(str(images_dir)):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, str(workspace))
                zf.write(full_path, rel_path)
        # Add sparse/0 files ONLY (ensures Brush loads the complete model)
        for root, _, files in os.walk(str(final_sub0)):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.join("sparse", "0", f)
                zf.write(full_path, rel_path)

    emit("progress", step="done", percent=100, message=f"Dataset complete with {best_count} registered camera views!")
    emit("done",
         zip_path=str(output_zip),
         zip_size=os.path.getsize(output_zip),
         images_count=total_input_images,
         registered_cameras=best_count,
         points_found=(final_sub0 / "points3D.bin").exists())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COLMAP Data Preparation Runner")
    parser.add_argument("--workspace", required=True, help="Path to workspace directory")
    parser.add_argument("--scaling", default="1600k", choices=["No Scaling", "1600k", "Half"], help="Image scaling")
    parser.add_argument("--matching", default="Sequential", choices=["Sequential", "Exhaustive", "Spatial"], help="Matching strategy")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL", help="COLMAP camera model")
    parser.add_argument("--output-zip", default=None, help="Path to output zip")
    parser.add_argument("--video", default=None, help="Path to input video file")
    parser.add_argument("--fps", type=float, default=5.0, help="Frame extraction rate in FPS")
    args = parser.parse_args()

    try:
        run_pipeline(args.workspace, args.scaling, args.matching, args.camera_model, args.output_zip, args.video, args.fps)
    except Exception as e:
        emit("error", message=str(e))
        sys.exit(1)

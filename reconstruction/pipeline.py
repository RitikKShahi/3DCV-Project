"""End-to-end CPU-only incremental 3D Gaussian Splatting reconstruction pipeline."""

import argparse
import json
import logging
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from reconstruction.config import ReconstructionConfig, resolve_torch_device
from reconstruction.export.ply_exporter import export_ply
from reconstruction.export.representations import generate_and_cache_representations
from reconstruction.export.splat_exporter import export_splat
from reconstruction.gaussian.model import GaussianModel
from reconstruction.incremental.batch_processor import create_frame_batches
from reconstruction.incremental.checkpoint import load_checkpoint, save_checkpoint
from reconstruction.incremental.scene_updater import IncrementalSceneUpdater
from reconstruction.preprocessing.frame_extractor import extract_frames, prepare_images_from_dir
from reconstruction.preprocessing.keyframe_selector import select_keyframes
from reconstruction.preprocessing.quality_filter import filter_frames
from reconstruction.sfm import SfMEngine, SfMBatchResult, SparsePointCloud

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """Structured progress update emitted by the reconstruction pipeline."""

    stage: str
    progress: float          # 0.0 to 1.0
    message: str
    batch_index: int = 0
    total_batches: int = 1
    num_gaussians: int = 0
    num_cameras: int = 0
    psnr: float = 0.0
    loss: float = 0.0
    preview_splat_path: Optional[str] = None
    preview_image_path: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Outputs and benchmarks produced by a full reconstruction run."""

    ply_path: Path
    splat_path: Path
    preview_dir: Path
    keyframes_dir: Path
    num_keyframes: int
    num_registered_cameras: int
    num_gaussians: int
    sfm_backend: str
    final_psnr: float
    final_loss: float
    total_time_sec: float
    peak_memory_mb: float
    batch_metrics: list[dict[str, Any]]
    compute_device: str = "cpu"


def _rescale_poses_to_train_res(sfm_res: SfMBatchResult, target_max_res: int) -> None:
    """Scale SfM pose dimensions and intrinsics down to training resolution."""
    for pose_dict in (sfm_res.all_poses, sfm_res.new_poses):
        for pose in pose_dict.values():
            cur_max = max(pose.width, pose.height)
            if cur_max <= target_max_res:
                continue
            scale = target_max_res / float(cur_max)
            new_w = max(2, int(round(pose.width * scale)))
            new_h = max(2, int(round(pose.height * scale)))
            new_w -= new_w % 2
            new_h -= new_h % 2
            sx = new_w / float(pose.width)
            sy = new_h / float(pose.height)
            pose.K[0, 0] *= sx
            pose.K[0, 2] *= sx
            pose.K[1, 1] *= sy
            pose.K[1, 2] *= sy
            pose.width = new_w
            pose.height = new_h


def _get_peak_rss_mb() -> float:
    """Return peak resident set size in MB (macOS reports bytes, Linux reports KB)."""
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


class ReconstructionPipeline:
    """Orchestrates Stages 1-6 on CPU or GPU (MPS/CUDA) with incremental updates & checkpointing."""

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = (config or ReconstructionConfig()).clamp_for_safety()
        torch.set_num_threads(self.config.cpu_threads)
        self.torch_device = resolve_torch_device(self.config.device)
        logger.info("Initialized ReconstructionPipeline with compute device: %s", self.torch_device)

    def run(
        self,
        video_path: Path,
        work_dir: Path,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        resume: bool = True,
    ) -> PipelineResult:
        """Execute full incremental reconstruction from input video to .ply and .splat."""
        t0 = time.perf_counter()
        video_path = Path(video_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        raw_frames_dir = work_dir / "frames_raw"
        keyframes_dir = work_dir / "keyframes"
        sfm_dir = work_dir / "sfm"
        preview_dir = work_dir / "previews"
        checkpoint_dir = work_dir / "checkpoint"
        export_dir = work_dir / "exports"

        def emit(ev: ProgressEvent) -> None:
            if progress_callback is not None:
                progress_callback(ev)

        # Check for existing checkpoint & keyframes to resume cleanly
        ckpt = load_checkpoint(checkpoint_dir) if (resume and self.config.checkpoint_enabled) else None
        existing_keyframes = sorted(keyframes_dir.glob("keyframe_*.jpg")) if keyframes_dir.exists() else []

        if ckpt is not None and existing_keyframes:
            keyframes = existing_keyframes
            batches = [[keyframes_dir / name for name in b] for b in ckpt["batches"]]
            start_batch = ckpt["completed_batch"] + 1
            model: GaussianModel = ckpt["model"].to(self.torch_device)
            metrics_history: list[dict[str, Any]] = ckpt["metrics_history"]
            sfm_engine = SfMEngine(
                workspace_dir=sfm_dir,
                image_dir=keyframes_dir,
                backend=ckpt["sfm_backend"],
                cpu_threads=self.config.cpu_threads,
                max_features=self.config.max_sift_features,
                min_inliers_pnp=self.config.min_inliers_pnp,
                camera_fov_deg=self.config.camera_fov_deg,
                use_gpu=(self.torch_device.type == "cuda"),
            )
            processed_so_far = [p for b in batches[:start_batch] for p in b]
            sfm_engine.restore_state(
                known_image_names=set(ckpt["poses"].keys()),
                backend=ckpt["sfm_backend"],
                batch_counter=start_batch,
                processed_frames=processed_so_far,
            )
            last_sfm_res = SfMBatchResult(
                all_poses=ckpt["poses"],
                new_poses={},
                all_points=SparsePointCloud(
                    points=model._xyz.detach().cpu().numpy(),
                    colors=torch.zeros((model.num_gaussians, 3)).numpy(),
                ),
                new_points=SparsePointCloud(
                    points=torch.zeros((0, 3)).numpy(),
                    colors=torch.zeros((0, 3)).numpy(),
                ),
                backend_used=ckpt["sfm_backend"],
            )
            emit(
                ProgressEvent(
                    stage="resumed",
                    progress=0.2 + 0.65 * (start_batch / max(1, len(batches))),
                    message=f"Resumed from checkpoint at batch {start_batch}/{len(batches)}",
                    batch_index=start_batch,
                    total_batches=len(batches),
                    num_gaussians=model.num_gaussians,
                    num_cameras=len(ckpt["poses"]),
                )
            )
        else:
            # Stage 1: Video or photo-set preprocessing (prepare at higher resolution for accurate SfM)
            extract_res = max(self.config.max_resolution, self.config.sfm_resolution)
            is_photo_dir = video_path.is_dir()
            if is_photo_dir:
                emit(ProgressEvent(stage="preprocessing", progress=0.02, message="Preparing uploaded photos..."))
                raw_frames = prepare_images_from_dir(
                    images_dir=video_path,
                    output_dir=raw_frames_dir,
                    max_resolution=extract_res,
                    max_images=max(500, self.config.max_keyframes),
                    progress_cb=lambda p, msg: emit(ProgressEvent(stage="preprocessing", progress=p, message=msg)),
                )
            else:
                emit(ProgressEvent(stage="preprocessing", progress=0.02, message="Extracting frames from video..."))
                raw_frames = extract_frames(
                    video_path=video_path,
                    output_dir=raw_frames_dir,
                    frame_interval=self.config.frame_interval,
                    max_resolution=extract_res,
                    progress_cb=lambda p, msg: emit(ProgressEvent(stage="preprocessing", progress=p, message=msg)),
                )

            emit(ProgressEvent(stage="preprocessing", progress=0.16, message="Filtering blurry & redundant frames..."))
            filtered, _ = filter_frames(
                raw_frames,
                blur_threshold=self.config.blur_threshold,
                min_frame_diff=(0.005 if is_photo_dir else self.config.min_frame_diff),
            )

            emit(ProgressEvent(stage="preprocessing", progress=0.20, message="Selecting informative keyframes..."))
            keyframes = select_keyframes(
                filtered,
                output_dir=keyframes_dir,
                max_keyframes=self.config.max_keyframes,
            )

            batches = create_frame_batches(
                keyframes,
                initial_batch_size=self.config.initial_batch_size,
                incremental_batch_size=self.config.incremental_batch_size,
            )
            start_batch = 0
            model = GaussianModel(sh_degree=self.config.sh_degree).to(self.torch_device)
            metrics_history = []
            sfm_engine = SfMEngine(
                workspace_dir=sfm_dir,
                image_dir=keyframes_dir,
                backend=self.config.sfm_backend,
                cpu_threads=self.config.cpu_threads,
                max_features=self.config.max_sift_features,
                min_inliers_pnp=self.config.min_inliers_pnp,
                camera_fov_deg=self.config.camera_fov_deg,
                use_gpu=(self.torch_device.type == "cuda"),
            )
            last_sfm_res = None

        updater = IncrementalSceneUpdater(model, self.config)
        total_batches = len(batches)
        batch_names_list = [[p.name for p in b] for b in batches]

        # Stages 2-5: Incremental SfM -> Gaussian update -> Preview -> Checkpoint
        for b_idx in range(start_batch, total_batches):
            batch_frames = batches[b_idx]
            base_prog = 0.20 + 0.65 * (b_idx / max(1, total_batches))
            span = 0.65 / max(1, total_batches)

            emit(
                ProgressEvent(
                    stage="sfm",
                    progress=base_prog + 0.1 * span,
                    message=f"Batch {b_idx + 1}/{total_batches}: Estimating camera poses ({len(batch_frames)} frames)...",
                    batch_index=b_idx + 1,
                    total_batches=total_batches,
                    num_gaussians=model.num_gaussians,
                )
            )
            sfm_res = sfm_engine.process_batch(batch_frames)
            _rescale_poses_to_train_res(sfm_res, self.config.max_resolution)
            last_sfm_res = sfm_res

            # Note: If OpenCV fallback kicked in on a later batch, reset model once if coordinate system changed
            if b_idx > 0 and sfm_res.backend_used == "opencv" and len(metrics_history) > 0 and metrics_history[-1].get("sfm_backend") == "pycolmap":
                logger.info("SfM switched to OpenCV fallback; reinitializing Gaussians in fallback coordinate frame")
                model = GaussianModel(sh_degree=self.config.sh_degree).to(self.torch_device)
                updater = IncrementalSceneUpdater(model, self.config)

            def _on_opt_step(step: int, total: int, ema_loss: float, cur_psnr: float) -> None:
                frac = step / max(1, total)
                emit(
                    ProgressEvent(
                        stage="optimizing",
                        progress=base_prog + (0.25 + 0.70 * frac) * span,
                        message=(
                            f"Batch {b_idx + 1}/{total_batches}: Optimizing {model.num_gaussians} Gaussians "
                            f"(iter {step}/{total}, loss={ema_loss:.4f}, PSNR={cur_psnr:.2f} dB)"
                        ),
                        batch_index=b_idx + 1,
                        total_batches=total_batches,
                        num_gaussians=model.num_gaussians,
                        num_cameras=len(sfm_res.all_poses),
                        psnr=cur_psnr,
                        loss=ema_loss,
                    )
                )

            b_metrics = updater.apply_sfm_batch(
                sfm_res=sfm_res,
                batch_index=b_idx,
                preview_dir=preview_dir,
                progress_cb=_on_opt_step,
            )
            metrics_history.append(b_metrics)

            if self.config.checkpoint_enabled:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    completed_batch=b_idx,
                    batches=batch_names_list,
                    model=model,
                    poses=sfm_res.all_poses,
                    sfm_backend=sfm_res.backend_used,
                    metrics_history=metrics_history,
                )

            emit(
                ProgressEvent(
                    stage="incremental_preview",
                    progress=base_prog + span,
                    message=(
                        f"Batch {b_idx + 1}/{total_batches} complete: "
                        f"{len(sfm_res.all_poses)} cameras, {model.num_gaussians} Gaussians, "
                        f"PSNR={b_metrics['mean_psnr']:.2f} dB"
                    ),
                    batch_index=b_idx + 1,
                    total_batches=total_batches,
                    num_gaussians=model.num_gaussians,
                    num_cameras=len(sfm_res.all_poses),
                    psnr=b_metrics["mean_psnr"],
                    loss=b_metrics["final_loss"],
                    preview_splat_path=b_metrics.get("preview_splat"),
                    preview_image_path=b_metrics.get("preview_image"),
                )
            )

        assert last_sfm_res is not None, "SfM produced no poses"

        # Final global refinement
        emit(
            ProgressEvent(
                stage="refining",
                progress=0.88,
                message=f"Final global refinement ({self.config.final_iterations} iters, {model.num_gaussians} Gaussians)...",
                batch_index=total_batches,
                total_batches=total_batches,
                num_gaussians=model.num_gaussians,
                num_cameras=len(last_sfm_res.all_poses),
            )
        )
        final_metrics = updater.final_refinement(
            last_sfm_res,
            progress_cb=lambda s, tot, loss, p: emit(
                ProgressEvent(
                    stage="refining",
                    progress=0.88 + 0.08 * (s / max(1, tot)),
                    message=f"Final refinement: iter {s}/{tot}, loss={loss:.4f}, PSNR={p:.2f} dB",
                    batch_index=total_batches,
                    total_batches=total_batches,
                    num_gaussians=model.num_gaussians,
                    num_cameras=len(last_sfm_res.all_poses),
                    psnr=p,
                    loss=loss,
                )
            ),
        )

        # Stage 6: Export final .ply and .splat
        emit(ProgressEvent(stage="exporting", progress=0.97, message="Exporting final .ply and .splat scenes..."))
        ply_path = export_ply(model, export_dir / "scene.ply")
        splat_path = export_splat(model, export_dir / "scene.splat")

        # Update latest_preview.splat and derived representations (Geometry, Mesh, Triangle Splats)
        (preview_dir / "latest_preview.splat").write_bytes(splat_path.read_bytes())
        try:
            generate_and_cache_representations(
                preview_dir=preview_dir,
                model=model,
                poses=last_sfm_res.all_poses,
                mesh_grid_res=56,
                export_obj_path=export_dir / "scene_mesh.obj",
            )
        except Exception as exc:
            logger.warning("Failed generating final derived representations: %s", exc)

        total_time = time.perf_counter() - t0
        peak_mem = _get_peak_rss_mb()

        final_psnr = float(final_metrics.get("mean_psnr") or (metrics_history[-1]["mean_psnr"] if metrics_history else 0.0))
        final_loss = float(final_metrics.get("final_loss") or (metrics_history[-1]["final_loss"] if metrics_history else 0.0))

        summary = {
            "num_keyframes": len(keyframes),
            "num_registered_cameras": len(last_sfm_res.all_poses),
            "num_gaussians": model.num_gaussians,
            "sfm_backend": last_sfm_res.backend_used,
            "compute_device": str(self.torch_device),
            "final_psnr": final_psnr,
            "final_loss": final_loss,
            "total_time_sec": total_time,
            "peak_memory_mb": peak_mem,
            "batch_metrics": metrics_history,
        }
        (export_dir / "benchmarks.json").write_text(json.dumps(summary, indent=2))

        emit(
            ProgressEvent(
                stage="completed",
                progress=1.0,
                message=f"Reconstruction complete ({self.torch_device})! {model.num_gaussians} Gaussians, PSNR={final_psnr:.2f} dB in {total_time:.1f}s",
                batch_index=total_batches,
                total_batches=total_batches,
                num_gaussians=model.num_gaussians,
                num_cameras=len(last_sfm_res.all_poses),
                psnr=final_psnr,
                loss=final_loss,
                preview_splat_path=str(splat_path),
                extra={"device": str(self.torch_device)},
            )
        )

        return PipelineResult(
            ply_path=ply_path,
            splat_path=splat_path,
            preview_dir=preview_dir,
            keyframes_dir=keyframes_dir,
            num_keyframes=len(keyframes),
            num_registered_cameras=len(last_sfm_res.all_poses),
            num_gaussians=model.num_gaussians,
            sfm_backend=last_sfm_res.backend_used,
            final_psnr=final_psnr,
            final_loss=final_loss,
            total_time_sec=total_time,
            peak_memory_mb=peak_mem,
            batch_metrics=metrics_history,
            compute_device=str(self.torch_device),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="CPU-only Incremental 3DGS Reconstruction CLI")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output_recon", help="Working/output directory")
    parser.add_argument("--max-res", type=int, default=320)
    parser.add_argument("--max-keyframes", type=int, default=16)
    parser.add_argument("--max-gaussians", type=int, default=3500)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--final-iters", type=int, default=100)
    parser.add_argument("--sfm", default="auto", choices=["auto", "pycolmap", "opencv"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    cfg = ReconstructionConfig(
        max_resolution=args.max_res,
        max_keyframes=args.max_keyframes,
        max_gaussians=args.max_gaussians,
        iterations_per_batch=args.iters,
        final_iterations=args.final_iters,
        sfm_backend=args.sfm,
        device=args.device,
    )
    pipeline = ReconstructionPipeline(cfg)
    res = pipeline.run(
        Path(args.video),
        Path(args.output),
        progress_callback=lambda e: print(f"[{e.progress*100:5.1f}%] ({e.stage}) {e.message}"),
    )
    print("\n=== DONE ===")
    print(f"PLY:   {res.ply_path}")
    print(f"SPLAT: {res.splat_path}")
    print(f"Stats: {res.num_gaussians} Gaussians, {res.num_registered_cameras} cameras, PSNR={res.final_psnr:.2f} dB, {res.total_time_sec:.1f}s, {res.peak_memory_mb:.1f} MB RSS")


if __name__ == "__main__":
    main()


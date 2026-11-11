"""Primary CPU-only SfM driver using pycolmap in an isolated subprocess."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from reconstruction.sfm.types import CameraPose, SfMBatchResult, SparsePointCloud

logger = logging.getLogger(__name__)


class PyColmapSfM:
    """Incremental SfM wrapper powered by CPU-only pycolmap."""

    def __init__(
        self,
        workspace_dir: Path,
        image_dir: Path,
        cpu_threads: int = 4,
        max_features: int = 4096,
        camera_fov_deg: float = 0.0,
        use_gpu: bool = False,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.image_dir = Path(image_dir)
        self.cpu_threads = cpu_threads
        self.max_features = max_features
        self.camera_fov_deg = camera_fov_deg
        self.use_gpu = use_gpu
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.known_images: set[str] = set()
        self.known_point_ids: set[int] = set()
        self.batch_counter = 0

    def restore_state(self, known_image_names: set[str], batch_counter: int) -> None:
        """Restore incremental tracking state when resuming from a checkpoint."""
        self.known_images = set(known_image_names)
        self.batch_counter = int(batch_counter)

    def process_batch(self, frame_paths: list[Path]) -> SfMBatchResult:
        """Run initial or incremental SfM on a batch of keyframes."""
        image_names = [Path(p).name for p in frame_paths]
        incremental = self.batch_counter > 0 and len(self.known_images) > 0

        out_json = self.workspace_dir / f"sfm_batch_{self.batch_counter}.json"
        out_npz = self.workspace_dir / f"sfm_batch_{self.batch_counter}.npz"

        cmd = [
            sys.executable, "-m", "reconstruction.sfm.pycolmap_worker",
            "--workspace", str(self.workspace_dir),
            "--image-dir", str(self.image_dir),
            "--batch-index", str(self.batch_counter),
            "--threads", str(self.cpu_threads),
            "--max-features", str(self.max_features),
            "--fov-deg", str(self.camera_fov_deg),
            "--out-json", str(out_json),
            "--out-npz", str(out_npz),
        ]
        if self.use_gpu:
            cmd.append("--use-gpu")
        if incremental:
            cmd.append("--incremental")
        cmd.extend(["--images", *image_names])

        logger.info("Running pycolmap worker (batch=%d, incremental=%s, images=%d)", self.batch_counter, incremental, len(image_names))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"pycolmap_worker failed (code {proc.returncode}): {proc.stderr.strip()}")

        payload = json.loads(out_json.read_text())
        data = np.load(out_npz)

        all_poses: dict[str, CameraPose] = {
            name: CameraPose.from_dict(pdict) for name, pdict in payload["poses"].items()
        }
        new_poses = {name: pose for name, pose in all_poses.items() if name not in self.known_images}
        self.known_images.update(all_poses.keys())

        points = data["points"].astype(np.float32)
        colors = data["colors"].astype(np.float32)
        pids = data["point_ids"].astype(np.int64).tolist()

        if self.known_point_ids:
            new_mask = np.array([pid not in self.known_point_ids for pid in pids], dtype=bool)
        else:
            new_mask = np.ones(len(pids), dtype=bool)

        new_points_arr = points[new_mask] if len(points) > 0 else points
        new_colors_arr = colors[new_mask] if len(colors) > 0 else colors
        new_pids = [pid for pid, m in zip(pids, new_mask) if m]
        self.known_point_ids.update(pids)

        self.batch_counter += 1

        return SfMBatchResult(
            all_poses=all_poses,
            new_poses=new_poses,
            all_points=SparsePointCloud(points=points, colors=colors, point_ids=pids),
            new_points=SparsePointCloud(points=new_points_arr, colors=new_colors_arr, point_ids=new_pids),
            backend_used="pycolmap",
            mean_reprojection_error=float(payload.get("mean_reprojection_error", 0.0)),
            workspace_dir=self.workspace_dir,
        )

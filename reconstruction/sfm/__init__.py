"""Stage 2: Unified CPU-only Structure-from-Motion engine (pycolmap primary + OpenCV fallback)."""

import logging
from pathlib import Path
from typing import Optional

from reconstruction.sfm.opencv_sfm import OpenCvSfM
from reconstruction.sfm.pycolmap_sfm import PyColmapSfM
from reconstruction.sfm.types import CameraPose, SfMBatchResult, SparsePointCloud

logger = logging.getLogger(__name__)


class SfMEngine:
    """Facade providing consistent incremental camera pose estimation on CPU."""

    def __init__(
        self,
        workspace_dir: Path,
        image_dir: Path,
        backend: str = "auto",
        cpu_threads: int = 4,
        max_features: int = 4096,
        min_inliers_pnp: int = 12,
        camera_fov_deg: float = 0.0,
        use_gpu: bool = False,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.image_dir = Path(image_dir)
        self.requested_backend = backend
        self.cpu_threads = cpu_threads
        self.max_features = max_features
        self.min_inliers_pnp = min_inliers_pnp
        self.camera_fov_deg = camera_fov_deg
        self.use_gpu = use_gpu

        self._pycolmap: Optional[PyColmapSfM] = None
        self._opencv: Optional[OpenCvSfM] = None
        self.active_backend: Optional[str] = None
        self._processed_frames: list[Path] = []

    def restore_state(
        self,
        known_image_names: set[str],
        backend: str,
        batch_counter: int,
        processed_frames: list[Path],
    ) -> None:
        """Restore SfM engine state when resuming an interrupted job."""
        self._processed_frames = list(processed_frames)
        self.active_backend = backend
        if backend == "pycolmap":
            self._pycolmap = PyColmapSfM(
                workspace_dir=self.workspace_dir / "colmap",
                image_dir=self.image_dir,
                cpu_threads=self.cpu_threads,
                max_features=self.max_features,
                camera_fov_deg=self.camera_fov_deg,
                use_gpu=self.use_gpu,
            )
            self._pycolmap.restore_state(known_image_names, batch_counter)

    def process_batch(self, batch_paths: list[Path]) -> SfMBatchResult:
        """Process initial or incremental batch of keyframes.

        If `pycolmap` fails during initialization (or mid-run), seamlessly falls back
        to the OpenCV + SciPy SfM engine across all accumulated frames.
        """
        self._processed_frames.extend(batch_paths)

        if self.requested_backend == "opencv":
            return self._run_opencv(batch_paths)

        if self.active_backend in (None, "pycolmap"):
            try:
                if self._pycolmap is None:
                    self._pycolmap = PyColmapSfM(
                        workspace_dir=self.workspace_dir / "colmap",
                        image_dir=self.image_dir,
                        cpu_threads=self.cpu_threads,
                        max_features=self.max_features,
                        camera_fov_deg=self.camera_fov_deg,
                        use_gpu=self.use_gpu,
                    )
                res = self._pycolmap.process_batch(batch_paths)
                self.active_backend = "pycolmap"
                return res
            except Exception as exc:
                logger.warning("pycolmap SfM failed (%s); falling back to OpenCV+SciPy SfM", exc)
                self.active_backend = "opencv"
                # Re-initialize OpenCV SfM over all frames processed so far if needed
                self._opencv = OpenCvSfM(
                    max_features=self.max_features,
                    min_inliers_pnp=self.min_inliers_pnp,
                    camera_fov_deg=self.camera_fov_deg,
                )
                return self._opencv.process_batch(self._processed_frames)

        return self._run_opencv(batch_paths)

    def _run_opencv(self, batch_paths: list[Path]) -> SfMBatchResult:
        if self._opencv is None:
            self._opencv = OpenCvSfM(
                max_features=self.max_features,
                min_inliers_pnp=self.min_inliers_pnp,
                camera_fov_deg=self.camera_fov_deg,
            )
        self.active_backend = "opencv"
        return self._opencv.process_batch(batch_paths)


__all__ = ["SfMEngine", "CameraPose", "SparsePointCloud", "SfMBatchResult"]

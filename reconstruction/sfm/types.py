"""Shared data structures for CPU-only Structure-from-Motion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CameraPose:
    """Pinhole camera intrinsics and world-to-camera extrinsics."""

    image_name: str
    image_path: str
    width: int
    height: int
    K: np.ndarray        # [3, 3] float64 intrinsics
    R: np.ndarray        # [3, 3] float64 world-to-camera rotation
    t: np.ndarray        # [3] float64 world-to-camera translation

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coordinates: -R^T @ t."""
        return -self.R.T @ self.t

    def to_dict(self) -> dict:
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "width": int(self.width),
            "height": int(self.height),
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraPose":
        return cls(
            image_name=data["image_name"],
            image_path=data["image_path"],
            width=int(data["width"]),
            height=int(data["height"]),
            K=np.asarray(data["K"], dtype=np.float64),
            R=np.asarray(data["R"], dtype=np.float64),
            t=np.asarray(data["t"], dtype=np.float64),
        )


@dataclass
class SparsePointCloud:
    """Sparse 3D points and RGB colors in [0, 1]."""

    points: np.ndarray   # [N, 3] float32
    colors: np.ndarray   # [N, 3] float32 in [0, 1]
    point_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.points.shape[0])


@dataclass
class SfMBatchResult:
    """Result of initial or incremental SfM batch."""

    all_poses: dict[str, CameraPose]
    new_poses: dict[str, CameraPose]
    all_points: SparsePointCloud
    new_points: SparsePointCloud
    backend_used: str
    mean_reprojection_error: float = 0.0
    workspace_dir: Optional[Path] = None


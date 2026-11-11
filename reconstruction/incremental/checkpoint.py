"""Checkpointing for resumable CPU-only incremental 3DGS reconstruction."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from reconstruction.gaussian.model import GaussianModel
from reconstruction.sfm.types import CameraPose

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: Path,
    completed_batch: int,
    batches: list[list[str]],
    model: GaussianModel,
    poses: dict[str, CameraPose],
    sfm_backend: str,
    metrics_history: list[dict[str, Any]],
) -> Path:
    """Persist full incremental state so interrupted jobs can resume cleanly."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch_path = checkpoint_dir / "gaussians.pt"
    tmp_torch = checkpoint_dir / "gaussians.pt.tmp"
    torch.save(model.export_checkpoint_dict(), tmp_torch)
    tmp_torch.replace(torch_path)

    state = {
        "completed_batch": int(completed_batch),
        "batches": batches,
        "sfm_backend": sfm_backend,
        "poses": {name: pose.to_dict() for name, pose in poses.items()},
        "metrics_history": metrics_history,
    }
    json_path = checkpoint_dir / "state.json"
    tmp_json = checkpoint_dir / "state.json.tmp"
    tmp_json.write_text(json.dumps(state, indent=2))
    tmp_json.replace(json_path)

    logger.info("Saved checkpoint after batch %d (%d Gaussians) to %s", completed_batch, model.num_gaussians, checkpoint_dir)
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: Path) -> Optional[dict[str, Any]]:
    """Load saved checkpoint if available and valid; otherwise return None."""
    checkpoint_dir = Path(checkpoint_dir)
    torch_path = checkpoint_dir / "gaussians.pt"
    json_path = checkpoint_dir / "state.json"
    if not torch_path.exists() or not json_path.exists():
        return None

    try:
        state = json.loads(json_path.read_text())
        g_data = torch.load(torch_path, map_location="cpu", weights_only=True)
        model = GaussianModel(sh_degree=int(g_data["sh_degree"]))
        model.load_checkpoint_dict(g_data)
        poses = {name: CameraPose.from_dict(p) for name, p in state["poses"].items()}
        return {
            "completed_batch": int(state["completed_batch"]),
            "batches": state["batches"],
            "sfm_backend": state["sfm_backend"],
            "model": model,
            "poses": poses,
            "metrics_history": state.get("metrics_history", []),
        }
    except Exception as exc:
        logger.warning("Failed to load checkpoint from %s (%s); starting fresh", checkpoint_dir, exc)
        return None


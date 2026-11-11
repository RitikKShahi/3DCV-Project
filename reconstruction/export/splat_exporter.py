"""Export GaussianModel to compact 32-byte .splat binary format for WebGL viewers."""

import logging
from pathlib import Path

import numpy as np
import torch

from reconstruction.gaussian.model import GaussianModel
from reconstruction.gaussian.sh_utils import sh_to_rgb

logger = logging.getLogger(__name__)


def export_splat(model: GaussianModel, output_path: Path) -> Path:
    """Write GaussianModel to antimatter15 / GaussianSplats3D `.splat` format.

    Layout per Gaussian (32 bytes, sorted by descending visual importance):
        - position: 3 x float32 (12 bytes)
        - scale:    3 x float32 (12 bytes, activated exp scale)
        - rgba:     4 x uint8   (4 bytes, 0..255)
        - rot:      4 x uint8   (4 bytes, (q * 128 + 128) clipped to 0..255)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        n = model.num_gaussians
        if n == 0:
            output_path.write_bytes(b"")
            return output_path

        pos = model._xyz.detach().cpu().numpy().astype(np.float32)
        scales = model.get_scales.detach().cpu().numpy().astype(np.float32)
        rgb = torch.clamp(sh_to_rgb(model._features_dc.detach()[:, 0, :]), 0.0, 1.0).cpu().numpy()
        alpha = model.get_opacity.detach().cpu().numpy().reshape(-1, 1)
        rot = model.get_rotations.detach().cpu().numpy().astype(np.float32)

    # Sort by descending scale_volume * opacity for progressive streaming/rendering
    importance = -np.prod(scales, axis=1) * alpha[:, 0]
    order = np.argsort(importance)

    pos = pos[order]
    scales = scales[order]
    rgb = rgb[order]
    alpha = alpha[order]
    rot = rot[order]

    rgba_u8 = np.clip(np.concatenate([rgb, alpha], axis=1) * 255.0, 0, 255).astype(np.uint8)
    rot_u8 = np.clip(rot * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # Pack structured 32-byte records cleanly without Python loops
    record_dtype = np.dtype(
        [
            ("pos", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rot", "u1", (4,)),
        ]
    )
    buf = np.empty(n, dtype=record_dtype)
    buf["pos"] = pos
    buf["scale"] = scales
    buf["rgba"] = rgba_u8
    buf["rot"] = rot_u8

    output_path.write_bytes(buf.tobytes())
    logger.info("Exported %d Gaussians to .splat (%d bytes): %s", n, output_path.stat().st_size, output_path)
    return output_path


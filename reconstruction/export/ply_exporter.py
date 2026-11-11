"""Export GaussianModel to standard 3DGS binary PLY format."""

import logging
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from reconstruction.gaussian.model import GaussianModel

logger = logging.getLogger(__name__)


def export_ply(model: GaussianModel, output_path: Path, pad_sh_to_degree3: bool = True) -> Path:
    """Write GaussianModel in standard GraphDeco-INRIA 3DGS binary PLY format.

    Attributes stored:
        - x, y, z (float32)
        - nx, ny, nz (float32 zeros)
        - f_dc_0..2 (float32)
        - f_rest_0..N (float32; padded to 45 channels when pad_sh_to_degree3=True for broad viewer support)
        - opacity (float32 raw logit)
        - scale_0..2 (float32 log-scale)
        - rot_0..3 (float32 normalized quaternion w,x,y,z)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        xyz = model._xyz.detach().cpu().numpy().astype(np.float32)
        n = len(xyz)
        normals = np.zeros((n, 3), dtype=np.float32)
        f_dc = model._features_dc.detach().cpu().numpy().reshape(n, 3).astype(np.float32)

        rest_raw = model._features_rest.detach().cpu().numpy()  # [N, K, 3]
        # Standard 3DGS PLY stores f_rest transposed: channel-major [N, 3*K]
        if rest_raw.shape[1] > 0:
            f_rest = rest_raw.transpose(0, 2, 1).reshape(n, -1).astype(np.float32)
        else:
            f_rest = np.zeros((n, 0), dtype=np.float32)

        if pad_sh_to_degree3 and f_rest.shape[1] < 45:
            padded = np.zeros((n, 45), dtype=np.float32)
            if f_rest.shape[1] > 0:
                padded[:, : f_rest.shape[1]] = f_rest
            f_rest = padded

        opacities = model._opacity.detach().cpu().numpy().reshape(n, 1).astype(np.float32)
        scales = model._scales.detach().cpu().numpy().reshape(n, 3).astype(np.float32)
        rots = model.get_rotations.detach().cpu().numpy().reshape(n, 4).astype(np.float32)

    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(3)]
    attrs += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    attrs += ["opacity"]
    attrs += [f"scale_{i}" for i in range(3)]
    attrs += [f"rot_{i}" for i in range(4)]

    dtype_full = [(a, "f4") for a in attrs]
    elements = np.empty(n, dtype=dtype_full)
    stacked = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scales, rots], axis=1)
    elements[:] = list(map(tuple, stacked))

    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=False).write(str(output_path))
    logger.info("Exported %d Gaussians to PLY: %s", n, output_path)
    return output_path


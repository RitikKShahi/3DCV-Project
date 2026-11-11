"""Spherical Harmonics utilities for 3D Gaussian Splatting."""

import torch

# Zero-order SH constant: 1 / (2 * sqrt(pi))
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to degree-0 SH coefficients."""
    return (rgb - 0.5) / C0


def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 SH coefficients to RGB in [0, 1] (unclamped)."""
    return sh_dc * C0 + 0.5


def eval_sh(
    degree: int,
    sh: torch.Tensor,
    dirs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate spherical harmonics up to `degree` (0..3).

    Args:
        degree: Active SH degree (0..3).
        sh: Tensor of shape [N, (deg_max+1)^2, 3].
        dirs: Normalized view directions [N, 3] (camera-to-point unit vectors).
              Required when degree >= 1.

    Returns:
        RGB colors of shape [N, 3] clamped to >= 0.
    """
    result = C0 * sh[:, 0, :]
    if degree >= 1 and dirs is not None and sh.shape[1] >= 4:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]

        if degree >= 2 and sh.shape[1] >= 9:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[:, 4]
                + C2[1] * yz * sh[:, 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                + C2[3] * xz * sh[:, 7]
                + C2[4] * (xx - yy) * sh[:, 8]
            )

        if degree >= 3 and sh.shape[1] >= 16:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C3[0] * y * (3 * xx - yy) * sh[:, 9]
                + C3[1] * xy * z * sh[:, 10]
                + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                + C3[5] * z * (xx - yy) * sh[:, 14]
                + C3[6] * x * (xx - 3 * yy) * sh[:, 15]
            )

    return torch.clamp(result + 0.5, 0.0, 1.0)


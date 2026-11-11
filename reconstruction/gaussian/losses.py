"""Photometric loss functions (L1 + SSIM) for 3DGS optimization on CPU."""

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between rendered and target images [H, W, 3]."""
    return torch.abs(pred - target).mean()


def _gaussian_window_1d(window_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute mean SSIM between two [H, W, 3] float32 images in [0, 1]."""
    # Convert [H, W, C] -> [1, C, H, W]
    img1 = pred.permute(2, 0, 1).unsqueeze(0)
    img2 = target.permute(2, 0, 1).unsqueeze(0)
    channels = img1.shape[1]

    # Separable 1D Gaussian kernel -> 2D window [C, 1, K, K]
    g1d = _gaussian_window_1d(window_size, sigma, img1.dtype, img1.device)
    g2d = (g1d.unsqueeze(1) @ g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = g2d.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean()


def photometric_loss(pred: torch.Tensor, target: torch.Tensor, lambda_dssim: float = 0.2) -> torch.Tensor:
    """Standard 3DGS combined photometric loss: (1 - lambda) * L1 + lambda * (1 - SSIM)."""
    l1 = l1_loss(pred, target)
    if lambda_dssim <= 0.0:
        return l1
    dssim = 1.0 - ssim(pred, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return 100.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())


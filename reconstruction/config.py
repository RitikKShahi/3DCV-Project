"""Configuration parameters for CPU-only 3DGS reconstruction."""

from dataclasses import dataclass, field
import os


@dataclass
class ReconstructionConfig:
    """All configurable parameters for preprocessing, SfM, 3DGS, and CPU tuning."""

    # Stage 1: Video preprocessing
    max_resolution: int = 384          # Max width/height in pixels for 3DGS training
    sfm_resolution: int = 720          # Higher resolution used for sharp SIFT SfM pose estimation
    frame_interval: int = 10           # Extract every N-th frame initially
    max_keyframes: int = 80            # CHANGED: 20 -> 80 (fewer, sharper, well-overlapping frames)
    blur_threshold: float = 25.0       # Minimum Laplacian variance to keep frame
    min_frame_diff: float = 0.025      # Minimum normalized difference to avoid duplicates

    # Stage 2: Camera pose estimation (SfM)
    sfm_backend: str = "auto"          # "auto", "pycolmap", or "opencv"
    max_sift_features: int = 6000      # SIFT keypoints per frame
    min_inliers_pnp: int = 12          # Minimum PnP inliers to register a frame
    camera_fov_deg: float = 0.0        # 0.0 = Auto-detect, or custom camera FOV in degrees (20-140)

    # Stage 3 & 4: Gaussian representation & optimization
    max_gaussians: int = 30000         # CHANGED: 5000 -> 30000
    sh_degree: int = 0                 # Spherical harmonics degree (0 = RGB only, 1-3 = view-dependent)
    iterations_per_batch: int = 400    # CHANGED: 200 -> 400
    final_iterations: int = 5000       # CHANGED: 150 -> 5000 (needs the raised clamp below)
    adaptive_early_stop: bool = False  # CHANGED: off. Per-view loss is noisy and was stopping runs early
    early_stop_patience: int = 500     # CHANGED: 60 -> 500 (only used if early stop is turned on)
    loss_ema_decay: float = 0.99       # NEW: smoother EMA for the early-stop check
    tile_size: int = 24                # Tile size in pixels for CPU sparse rasterization
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Optimization learning rates
    # CHANGED: position LRs are now multiplied by scene_extent in optimizer.py
    lr_position: float = 1.6e-4
    lr_position_final: float = 1.6e-6
    lr_feature_dc: float = 0.005       # CHANGED: 0.01 -> 0.005
    lr_feature_rest: float = 0.0005
    lr_opacity: float = 0.05
    lr_scale: float = 0.008            # CHANGED: 0.015 -> 0.008
    lr_rotation: float = 0.002         # CHANGED: 0.008 -> 0.002

    # Densification & pruning
    densify_interval: int = 100        # CHANGED: 40 -> 100 (each densify resets Adam state)
    densify_grad_threshold: float = 0.00025
    prune_opacity_threshold: float = 0.03
    prune_scale_percentile: float = 0.99
    screen_prune_frac: float = 0.05    # NEW: prune Gaussians whose screen radius > frac * max(W, H)
    screen_prune_start: float = 0.3    # NEW: only start screen-space pruning after this fraction of a run

    # Stage 5: Incremental reconstruction
    initial_batch_size: int = 20       # CHANGED: 8 -> 20
    incremental_batch_size: int = 20   # CHANGED: 4 -> 20
    replay_ratio: float = 0.5          # Fraction of replay views vs new views per step

    # CPU / GPU performance & resource controls
    device: str = "auto"               # "auto", "cpu", "mps" (Apple Silicon), or "cuda" (NVIDIA)
    cpu_threads: int = field(default_factory=lambda: min(8, os.cpu_count() or 4))  # CHANGED: 4 -> 8
    checkpoint_enabled: bool = True

    def clamp_for_safety(self) -> "ReconstructionConfig":
        """Clamp extreme values so CPU/GPU RAM is not exhausted."""
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            self.device = "auto"
        self.max_resolution = max(160, min(1280, int(self.max_resolution)))
        self.frame_interval = max(1, min(120, int(self.frame_interval)))
        self.max_keyframes = max(4, min(500, int(self.max_keyframes)))
        self.max_gaussians = max(500, min(50000, int(self.max_gaussians)))
        self.iterations_per_batch = max(20, min(3000, int(self.iterations_per_batch)))
        self.final_iterations = max(0, min(20000, int(self.final_iterations)))  # CHANGED: 2000 -> 20000
        self.initial_batch_size = max(3, min(100, min(self.max_keyframes, int(self.initial_batch_size))))
        self.incremental_batch_size = max(1, min(50, int(self.incremental_batch_size)))
        self.cpu_threads = max(1, min(os.cpu_count() or 8, int(self.cpu_threads)))
        self.sh_degree = 0 if self.sh_degree not in (0, 1, 2, 3) else self.sh_degree
        if self.camera_fov_deg > 0:
            self.camera_fov_deg = max(20.0, min(140.0, float(self.camera_fov_deg)))
        else:
            self.camera_fov_deg = 0.0
        return self


def _probe_mps_subprocess() -> tuple[bool, str]:
    """Verify MPS backward ops (matmul & conv2d) work in an isolated subprocess."""
    import platform
    import subprocess
    import sys
    import torch

    if os.environ.get("_3DGS_MPS_PROBE") == "1":
        return True, ""

    if not bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return False, "MPS backend is not available in this PyTorch installation."

    probe_code = (
        "import torch, torch.nn.functional as F\n"
        "dev = torch.device('mps')\n"
        "a = torch.randn(4, 3, device=dev, requires_grad=True)\n"
        "b = torch.randn(3, 3, device=dev, requires_grad=True)\n"
        "(a @ b.T).sum().backward()\n"
        "x = torch.randn(1, 3, 16, 16, device=dev, requires_grad=True)\n"
        "w = torch.randn(3, 1, 3, 3, device=dev)\n"
        "F.conv2d(x, w, padding=1, groups=3).sum().backward()\n"
        "torch.mps.synchronize()\n"
    )
    env = dict(os.environ)
    env["_3DGS_MPS_PROBE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode == 0:
            return True, "Apple Silicon MPS GPU verified"
        mac_ver = platform.mac_ver()[0] or platform.system()
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        hint = err[-1] if err else f"exit code {proc.returncode}"
        return (
            False,
            f"MPS GPU disabled: PyTorch {torch.__version__} Metal kernels aborted on macOS {mac_ver} ({hint}). "
            "Upgrade macOS to 15+ or use a compatible PyTorch build for MPS GPU.",
        )
    except Exception as exc:
        return False, f"MPS probe failed: {exc}"


_MPS_PROBE_CACHE: tuple[bool, str] | None = None


def probe_mps_support() -> tuple[bool, str]:
    """Return cached (usable, reason) for Apple Silicon MPS execution."""
    global _MPS_PROBE_CACHE
    if _MPS_PROBE_CACHE is None:
        _MPS_PROBE_CACHE = _probe_mps_subprocess()
    return _MPS_PROBE_CACHE


def resolve_torch_device(requested: str = "auto") -> "torch.device":
    """Resolve requested compute device ('auto', 'cpu', 'mps', 'cuda') to a safe torch.device."""
    import logging
    import torch

    log = logging.getLogger(__name__)
    req = (requested or "auto").lower()
    has_cuda = bool(torch.cuda.is_available())
    has_mps, mps_reason = probe_mps_support()

    if req == "cuda":
        if has_cuda:
            return torch.device("cuda")
        log.warning("CUDA requested but unavailable; falling back to %s", "mps" if has_mps else "cpu")
        return torch.device("mps" if has_mps else "cpu")

    if req == "mps":
        if has_mps:
            return torch.device("mps")
        log.warning("MPS requested but unusable (%s); falling back to cpu", mps_reason)
        return torch.device("cpu")

    if req == "cpu":
        return torch.device("cpu")

    # Auto: prefer verified CUDA, then verified Apple Silicon MPS, else CPU
    if has_cuda:
        return torch.device("cuda")
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")
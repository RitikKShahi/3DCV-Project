"""Backend configuration settings (SQLite dev / PostgreSQL prod, storage paths)."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "CPU-Only 3D Gaussian Splatting Platform"
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    database_url: str = f"sqlite+aiosqlite:///{Path(__file__).resolve().parent.parent / 'data' / 'platform.db'}"
    max_upload_mb: int = 2048
    max_concurrent_jobs: int = 1  # Keep 1 concurrent heavy CPU reconstruction by default to protect 8GB RAM
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = {"env_prefix": "GS_PLATFORM_"}


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)


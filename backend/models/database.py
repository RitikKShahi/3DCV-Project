"""Async SQLAlchemy ORM models for projects and reconstruction jobs."""

from datetime import datetime, timezone
from typing import Optional
import uuid

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool

from backend.config import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    video_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    video_path: Mapped[str] = mapped_column(Text, nullable=False)
    video_width: Mapped[int] = mapped_column(Integer, default=0)
    video_height: Mapped[int] = mapped_column(Integer, default=0)
    video_duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    jobs: Mapped[list["ReconstructionJob"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="ReconstructionJob.created_at.desc()"
    )


class ReconstructionJob(Base):
    __tablename__ = "reconstruction_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued")  # queued, running, completed, failed
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="Queued")

    config_json: Mapped[str] = mapped_column(Text, default="{}")
    work_dir: Mapped[str] = mapped_column(Text, nullable=False)

    batch_index: Mapped[int] = mapped_column(Integer, default=0)
    total_batches: Mapped[int] = mapped_column(Integer, default=1)
    num_keyframes: Mapped[int] = mapped_column(Integer, default=0)
    num_cameras: Mapped[int] = mapped_column(Integer, default=0)
    num_gaussians: Mapped[int] = mapped_column(Integer, default=0)
    psnr: Mapped[float] = mapped_column(Float, default=0.0)
    loss: Mapped[float] = mapped_column(Float, default=0.0)
    sfm_backend: Mapped[str] = mapped_column(String(32), default="")
    elapsed_sec: Mapped[float] = mapped_column(Float, default=0.0)
    peak_memory_mb: Mapped[float] = mapped_column(Float, default=0.0)

    ply_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    splat_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latest_preview_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_history_json: Mapped[str] = mapped_column(Text, default="[]")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="jobs")


_is_sqlite = settings.database_url.startswith("sqlite")
engine = create_async_engine(
    settings.database_url,
    echo=False,
    poolclass=NullPool if _is_sqlite else None,
    connect_args={"timeout": 30} if _is_sqlite else {},
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


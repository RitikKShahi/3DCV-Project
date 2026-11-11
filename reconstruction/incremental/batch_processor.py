"""Keyframe batch partitioning for incremental reconstruction."""

from pathlib import Path


def create_frame_batches(
    keyframes: list[Path],
    initial_batch_size: int = 6,
    incremental_batch_size: int = 4,
) -> list[list[Path]]:
    """Partition ordered keyframes into an initial batch followed by incremental batches.

    Ensures the initial batch has enough views for reliable SfM initialization
    and no trailing batch has fewer than 2 frames if avoidable.
    """
    if not keyframes:
        return []

    n = len(keyframes)
    init_n = min(n, max(3, initial_batch_size))
    batches: list[list[Path]] = [keyframes[:init_n]]

    idx = init_n
    step = max(1, incremental_batch_size)
    while idx < n:
        batches.append(keyframes[idx : idx + step])
        idx += step

    return batches


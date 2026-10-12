"""Source-timeline predecessors and disjoint chunks for stateless gas inference."""
from __future__ import annotations


def previous_frame_indices(frame_count: int, independent_reference_frames=()) -> tuple[int | None, ...]:
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("A positive native frame count is required")
    independent = set(independent_reference_frames)
    if any(not isinstance(index, int) or not 0 <= index < frame_count for index in independent):
        raise ValueError("Independent references must belong to the native timeline")
    result, previous = [], None
    for index in range(frame_count):
        result.append(previous)
        if index not in independent:
            previous = index
    return tuple(result)


def frame_chunks(frame_count: int, chunk_frames: int = 4) -> tuple[tuple[int, ...], ...]:
    if not isinstance(frame_count, int) or frame_count < 1 or not isinstance(chunk_frames, int) or chunk_frames < 1:
        raise ValueError("Positive frame and chunk sizes are required")
    return tuple(tuple(range(first, min(frame_count, first + chunk_frames)))
                 for first in range(0, frame_count, chunk_frames))


def assert_complete_chunk_frames(frame_count: int, chunks) -> None:
    received = [int(frame["frame_index"]) for chunk in chunks for frame in chunk["frames"]]
    if sorted(received) != list(range(frame_count)):
        raise ValueError("Chunk receipts overlap or omit native frames")


__all__ = ["previous_frame_indices", "frame_chunks", "assert_complete_chunk_frames"]

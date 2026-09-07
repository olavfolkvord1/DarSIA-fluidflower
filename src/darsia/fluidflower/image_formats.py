"""Shared image-file selection policy for FluidFlower source folders."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


PROCESSED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
CAMERA_RAW_SUFFIXES = frozenset(
    {
        ".3fr",
        ".arw",
        ".cr2",
        ".cr3",
        ".dcr",
        ".dng",
        ".erf",
        ".kdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".orf",
        ".pef",
        ".raf",
        ".raw",
        ".rw2",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
)
ALL_IMAGE_SUFFIXES = PROCESSED_IMAGE_SUFFIXES | CAMERA_RAW_SUFFIXES

IMAGE_FILE_POLICY_PROCESSED = "prefer_processed"
IMAGE_FILE_POLICY_RAW = "prefer_raw"
IMAGE_FILE_POLICY_ALL = "all"
DEFAULT_IMAGE_FILE_POLICY = IMAGE_FILE_POLICY_PROCESSED
IMAGE_FILE_POLICIES = {
    IMAGE_FILE_POLICY_PROCESSED,
    IMAGE_FILE_POLICY_RAW,
    IMAGE_FILE_POLICY_ALL,
}
IMAGE_FILE_POLICY_LABELS = {
    IMAGE_FILE_POLICY_PROCESSED: "Prefer JPEG/TIFF/PNG (recommended)",
    IMAGE_FILE_POLICY_RAW: "Prefer camera RAW",
    IMAGE_FILE_POLICY_ALL: "Use both processed and RAW files",
}


def normalize_image_file_policy(value: Any) -> str:
    policy = str(value or DEFAULT_IMAGE_FILE_POLICY).strip().casefold()
    if policy not in IMAGE_FILE_POLICIES:
        raise ValueError(f"Unsupported image-file policy: {value!r}")
    return policy


def image_family(path: str | Path) -> str:
    suffix = Path(path).suffix.casefold()
    if suffix in PROCESSED_IMAGE_SUFFIXES:
        return "processed"
    if suffix in CAMERA_RAW_SUFFIXES:
        return "raw"
    return "other"


def select_preferred_images(
    paths: Iterable[Path],
    policy: str = DEFAULT_IMAGE_FILE_POLICY,
) -> list[Path]:
    """Select one image family per folder, with fallback when unavailable."""

    normalized = normalize_image_file_policy(policy)
    candidates = [
        Path(path)
        for path in paths
        if Path(path).suffix.casefold() in ALL_IMAGE_SUFFIXES
    ]
    processed = [path for path in candidates if image_family(path) == "processed"]
    raw = [path for path in candidates if image_family(path) == "raw"]
    if normalized == IMAGE_FILE_POLICY_ALL:
        return candidates
    if normalized == IMAGE_FILE_POLICY_RAW:
        return raw or processed
    return processed or raw


def image_file_counts(paths: Iterable[Path]) -> dict[str, int]:
    counts = {"processed": 0, "raw": 0, "other": 0}
    for path in paths:
        counts[image_family(path)] += 1
    return counts


__all__ = [
    "ALL_IMAGE_SUFFIXES",
    "CAMERA_RAW_SUFFIXES",
    "DEFAULT_IMAGE_FILE_POLICY",
    "IMAGE_FILE_POLICIES",
    "IMAGE_FILE_POLICY_ALL",
    "IMAGE_FILE_POLICY_LABELS",
    "IMAGE_FILE_POLICY_PROCESSED",
    "IMAGE_FILE_POLICY_RAW",
    "PROCESSED_IMAGE_SUFFIXES",
    "image_family",
    "image_file_counts",
    "normalize_image_file_policy",
    "select_preferred_images",
]

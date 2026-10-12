"""Discover and triage overflow-scale camera sources for the AC campaign.

The inventory deliberately separates source existence, protocol coverage, manual
readability, and OCR configuration. A folder containing images is not treated as
a usable overflow data set until those independent checks have been completed.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import csv
import json
import os
import re
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

try:
    from darsia.fluidflower.tools.ac_production_campaign import FINAL_PRODUCTION_RUNS
except ImportError:  # pragma: no cover - direct execution from darsia.fluidflower.tools/
    from darsia.fluidflower.tools.ac_production_campaign import FINAL_PRODUCTION_RUNS


REPO_ROOT = series_root("ffac")
DEFAULT_RAW_ROOT = configured_tool_path('ffac', 'raw_inputs')
INVENTORY_SCHEMA = 1
TRIAGE_SCHEMA = 1
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
SOURCE_TOKENS = ("overflow", "weight", "scale", "vekt")
TIMESTAMP_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")
SAMPLE_HOURS = (-0.5, 0.0, 3.0, 6.0, 12.0, 24.0, 48.0)
TRIAGE_STATUSES = (
    "needs-review",
    "ready",
    "needs-configuration",
    "partially-readable",
    "invalid-weight-data",
    "unreadable",
    "no-images",
    "time-mismatch",
    "partial-coverage",
)


@dataclass(frozen=True)
class SourceFrame:
    time: datetime
    path: Path


def _run_key(run: str) -> int:
    return int(re.sub(r"\D", "", run) or 0)


def parse_source_timestamp(path: Path) -> datetime | None:
    match = TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    try:
        return datetime(*map(int, match.groups()))
    except ValueError:
        return None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def inventory_dir(output_root: Path) -> Path:
    return output_root / "_inventory"


def inventory_path(output_root: Path) -> Path:
    return inventory_dir(output_root) / "overflow_source_inventory.json"


def triage_path(output_root: Path) -> Path:
    return inventory_dir(output_root) / "overflow_run_triage.json"


def _load_json(path: Path, schema: int) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if payload.get("schema") == schema else {}


def load_triage(output_root: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(triage_path(output_root), TRIAGE_SCHEMA)
    return {
        str(run).lower(): dict(value)
        for run, value in (payload.get("runs") or {}).items()
        if isinstance(value, dict)
    }


def save_run_triage(
    output_root: Path,
    run: str,
    status: str,
    reason: str,
    reviewer: str,
) -> dict[str, Any]:
    run = run.lower()
    if status not in TRIAGE_STATUSES:
        raise ValueError(f"Unknown overflow source status: {status}")
    path = triage_path(output_root)
    payload = _load_json(path, TRIAGE_SCHEMA) or {"schema": TRIAGE_SCHEMA, "runs": {}}
    record = {
        "status": status,
        "reason": reason.strip(),
        "reviewer": reviewer.strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload.setdefault("runs", {})[run] = record
    payload["updated_at"] = record["updated_at"]
    _atomic_json(path, payload)
    return record


def _injection_window(run: str, protocol_root: Path) -> tuple[datetime, datetime]:
    path = protocol_root / run / "injection_protocol.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty injection protocol: {path}")
    starts = [datetime.fromisoformat(row["start"]) for row in rows]
    ends = [datetime.fromisoformat(row["end"]) for row in rows]
    return min(starts), max(ends)


def _find_run_root(raw_root: Path, run: str) -> Path | None:
    direct = raw_root / run.upper()
    if direct.is_dir():
        return direct
    try:
        candidates = [
            path
            for path in raw_root.iterdir()
            if path.is_dir() and path.name.lower() == run.lower()
        ]
    except OSError:
        return None
    return candidates[0] if candidates else None


def discover_source_roots(
    raw_root: Path, run: str, configured_root: Path | None = None
) -> list[Path]:
    roots: list[Path] = []
    if configured_root is not None and configured_root.is_dir():
        roots.append(configured_root)
    run_root = _find_run_root(raw_root, run)
    if run_root is None:
        return roots
    for current, directories, _files in os.walk(run_root):
        current_path = Path(current)
        depth = len(current_path.relative_to(run_root).parts)
        if depth >= 4:
            directories[:] = []
            continue
        matched: list[str] = []
        for directory in directories:
            if any(token in directory.lower() for token in SOURCE_TOKENS):
                roots.append(current_path / directory)
                matched.append(directory)
        directories[:] = [directory for directory in directories if directory not in matched]
    unique: dict[str, Path] = {}
    for root in roots:
        unique[str(root).lower()] = root
    return list(unique.values())


def scan_source_frames(
    root: Path, image_time_offset_h: float = 0.0
) -> tuple[list[SourceFrame], int, list[str]]:
    frames: list[SourceFrame] = []
    video_count = 0
    errors: list[str] = []
    time_offset = timedelta(hours=float(image_time_offset_h))

    def on_error(error: OSError) -> None:
        errors.append(str(error))

    try:
        walker = os.walk(root, onerror=on_error)
        for current, _directories, files in walker:
            current_path = Path(current)
            for filename in files:
                path = current_path / filename
                suffix = path.suffix.lower()
                if suffix in VIDEO_SUFFIXES:
                    video_count += 1
                    continue
                if suffix not in IMAGE_SUFFIXES:
                    continue
                timestamp = parse_source_timestamp(path)
                if timestamp is not None:
                    frames.append(SourceFrame(timestamp + time_offset, path))
    except OSError as exc:
        errors.append(str(exc))
    frames.sort(key=lambda frame: (frame.time, str(frame.path).lower()))
    return frames, video_count, errors


def representative_frames(
    frames: Sequence[SourceFrame], injection_start: datetime
) -> list[dict[str, Any]]:
    if not frames:
        return []
    times = [frame.time for frame in frames]
    selected: list[tuple[float, int]] = []
    selected_indices: set[int] = set()
    for hour in SAMPLE_HOURS:
        target = injection_start + timedelta(hours=hour)
        insertion = bisect_left(times, target)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(frames)]
        if not candidates:
            continue
        index = min(candidates, key=lambda item: abs((times[item] - target).total_seconds()))
        if index not in selected_indices:
            selected.append((hour, index))
            selected_indices.add(index)
    return [
        {
            "sample_index": sample_index,
            "target_h": target_h,
            "time_from_injection_h": (frames[index].time - injection_start).total_seconds()
            / 3600.0,
            "image_time": frames[index].time.isoformat(),
            "path": str(frames[index].path),
        }
        for sample_index, (target_h, index) in enumerate(selected)
    ]


def _write_contact_sheet(run: str, samples: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not samples:
        return
    tile_width, tile_height, label_height = 360, 270, 44
    columns = 2
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, sample in enumerate(samples):
        left = (index % columns) * tile_width
        top = (index // columns) * (tile_height + label_height)
        try:
            with Image.open(str(sample["path"])) as source:
                image = ImageOps.contain(source.convert("RGB"), (tile_width, tile_height))
            x = left + (tile_width - image.width) // 2
            y = top + (tile_height - image.height) // 2
            sheet.paste(image, (x, y))
        except (OSError, ValueError) as exc:
            draw.rectangle((left, top, left + tile_width, top + tile_height), fill="#eeeeee")
            draw.text((left + 8, top + 8), f"Could not open image: {exc}", fill="#8b1e1e")
        label = (
            f"{run.upper()}  {str(sample['image_time']).replace('T', ' ')}  "
            f"t={float(sample['time_from_injection_h']):.2f} h"
        )
        draw.text((left + 7, top + tile_height + 7), label, fill="#202124")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "JPEG", quality=90)


def _automatic_status(
    configured: bool,
    image_count: int,
    relevant_count: int,
    coverage_start_h: float | None,
    coverage_end_h: float | None,
) -> tuple[str, str]:
    if image_count == 0:
        return "no-images", "No timestamped overflow-scale images were found."
    if relevant_count == 0:
        return (
            "time-mismatch",
            "Images exist, but none fall between -0.5 and 48 h from protocol injection start.",
        )
    if (coverage_start_h is not None and coverage_start_h > 0.5) or (
        coverage_end_h is not None and coverage_end_h < 8.0
    ):
        return (
            "partial-coverage",
            "The source does not cover injection start through at least 8 h after injection start.",
        )
    return ("ready", "") if configured else ("needs-review", "Manual readability review is pending.")


def build_source_inventory(
    *,
    raw_root: Path,
    output_root: Path,
    runs: Iterable[str] = FINAL_PRODUCTION_RUNS,
    configured_roots: Mapping[str, Path] | None = None,
    image_time_offsets_h: Mapping[str, float] | None = None,
    protocol_root: Path = project_path('protocols'),
) -> dict[str, Any]:
    configured_roots = {str(key).lower(): Path(value) for key, value in (configured_roots or {}).items()}
    image_time_offsets_h = {
        str(key).lower(): float(value)
        for key, value in (image_time_offsets_h or {}).items()
    }
    existing_triage = load_triage(output_root)
    rows: dict[str, dict[str, Any]] = {}
    contact_dir = inventory_dir(output_root) / "contact_sheets"
    for run in sorted({str(value).lower() for value in runs}, key=_run_key):
        image_time_offset_h = image_time_offsets_h.get(run, 0.0)
        time_offset = timedelta(hours=image_time_offset_h)
        configured_root = configured_roots.get(run)
        roots = discover_source_roots(raw_root, run, configured_root)
        scans: list[tuple[Path, list[SourceFrame], int, list[str]]] = []
        for root in roots:
            frames, videos, errors = scan_source_frames(root, image_time_offset_h)
            scans.append((root, frames, videos, errors))
        if configured_root is not None:
            configured_scan = next(
                (item for item in scans if str(item[0]).lower() == str(configured_root).lower() and item[1]),
                None,
            )
        else:
            configured_scan = None
        selected = configured_scan or (max(scans, key=lambda item: len(item[1]), default=None))
        root, frames, video_count, errors = selected or (None, [], 0, [])
        protocol_error = None
        try:
            injection_start, injection_end = _injection_window(run, protocol_root)
        except (OSError, ValueError, KeyError) as exc:
            injection_start = injection_end = None
            protocol_error = str(exc)
        if injection_start is not None:
            lower = injection_start - timedelta(hours=0.5)
            upper = injection_start + timedelta(hours=48.0)
            relevant = [frame for frame in frames if lower <= frame.time <= upper]
            samples = representative_frames(relevant or frames, injection_start)
        else:
            relevant = []
            samples = []
        contact_sheet = contact_dir / f"{run}_overflow_source_samples.jpg"
        if samples:
            _write_contact_sheet(run, samples, contact_sheet)
        elif contact_sheet.exists():
            contact_sheet.unlink()
        configured = run in configured_roots
        coverage_start_h = (
            (frames[0].time - injection_start).total_seconds() / 3600.0
            if frames and injection_start
            else None
        )
        coverage_end_h = (
            (frames[-1].time - injection_start).total_seconds() / 3600.0
            if frames and injection_start
            else None
        )
        automatic_status, automatic_reason = _automatic_status(
            configured,
            len(frames),
            len(relevant),
            coverage_start_h,
            coverage_end_h,
        )
        manual = existing_triage.get(run, {})
        rows[run] = {
            "run": run,
            "configured": configured,
            "configured_root": str(configured_root) if configured_root is not None else None,
            "selected_root": str(root) if root is not None else None,
            "candidate_roots": [
                {
                    "path": str(candidate_root),
                    "image_count": len(candidate_frames),
                    "video_count": candidate_videos,
                    "scan_errors": candidate_errors,
                }
                for candidate_root, candidate_frames, candidate_videos, candidate_errors in scans
            ],
            "image_count": len(frames),
            "relevant_image_count": len(relevant),
            "video_count": video_count,
            "image_time_offset_h": image_time_offset_h,
            "raw_first_image_time": (
                (frames[0].time - time_offset).isoformat() if frames else None
            ),
            "raw_last_image_time": (
                (frames[-1].time - time_offset).isoformat() if frames else None
            ),
            "first_image_time": frames[0].time.isoformat() if frames else None,
            "last_image_time": frames[-1].time.isoformat() if frames else None,
            "injection_start": injection_start.isoformat() if injection_start else None,
            "injection_end": injection_end.isoformat() if injection_end else None,
            "coverage_start_h": coverage_start_h,
            "coverage_end_h": coverage_end_h,
            "samples": samples,
            "contact_sheet": str(contact_sheet) if samples else None,
            "automatic_status": automatic_status,
            "automatic_reason": automatic_reason,
            "triage_status": manual.get("status", automatic_status),
            "triage_reason": manual.get("reason", automatic_reason),
            "triage_reviewer": manual.get("reviewer", ""),
            "triage_updated_at": manual.get("updated_at"),
            "protocol_error": protocol_error,
            "scan_errors": errors,
        }
    payload = {
        "schema": INVENTORY_SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "run_count": len(rows),
        "runs": rows,
    }
    directory = inventory_dir(output_root)
    _atomic_json(inventory_path(output_root), payload)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "overflow_source_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "run",
            "triage_status",
            "configured",
            "image_count",
            "relevant_image_count",
            "video_count",
            "selected_root",
            "first_image_time",
            "last_image_time",
            "injection_start",
            "coverage_start_h",
            "coverage_end_h",
            "triage_reason",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())
    return payload


def load_source_catalog(
    *,
    output_root: Path,
    runs: Iterable[str] = FINAL_PRODUCTION_RUNS,
    configured_roots: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    configured_roots = {str(key).lower(): Path(value) for key, value in (configured_roots or {}).items()}
    payload = _load_json(inventory_path(output_root), INVENTORY_SCHEMA)
    saved = payload.get("runs") or {}
    manual = load_triage(output_root)
    catalog: dict[str, dict[str, Any]] = {}
    for run in sorted({str(value).lower() for value in runs} | set(configured_roots), key=_run_key):
        row = dict(saved.get(run) or {})
        configured = run in configured_roots
        row.update(
            {
                "run": run,
                "configured": configured,
                "configured_root": str(configured_roots[run]) if configured else None,
            }
        )
        row.setdefault("image_count", None)
        row.setdefault("relevant_image_count", None)
        row.setdefault("video_count", None)
        row.setdefault("samples", [])
        row.setdefault("automatic_status", "not-scanned")
        triage = manual.get(run, {})
        row["triage_status"] = triage.get(
            "status", row.get("triage_status", row["automatic_status"])
        )
        row["triage_reason"] = triage.get("reason", row.get("triage_reason", ""))
        row["triage_reviewer"] = triage.get("reviewer", row.get("triage_reviewer", ""))
        row["triage_updated_at"] = triage.get(
            "updated_at", row.get("triage_updated_at")
        )
        catalog[run] = row
    return catalog


def source_sample_path(catalog: Mapping[str, Mapping[str, Any]], run: str, sample: int) -> Path:
    row = catalog.get(run.lower())
    if row is None:
        raise ValueError(f"Unknown run: {run}")
    samples = list(row.get("samples") or [])
    if not 0 <= sample < len(samples):
        raise ValueError(f"Overflow source sample out of range: {sample}")
    path = Path(str(samples[sample]["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Overflow source image is unavailable: {path}")
    return path

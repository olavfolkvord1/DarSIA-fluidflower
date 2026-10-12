"""Review overflow-scale images with guided OCR and an auditable Excel log.

The OCR value is a proposal, never ground truth. One persistent row is created for
every relevant camera image. New runs establish whole-window trend anchors
before filling temporal gaps, and every decision remains linked to the exact
source image.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import bisect
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from datetime import date, datetime, time as datetime_time, timedelta
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage
from sklearn.ensemble import ExtraTreesClassifier

try:
    from darsia.fluidflower.tools.overflow_source_inventory import (
        DEFAULT_RAW_ROOT,
        FINAL_PRODUCTION_RUNS,
        TRIAGE_STATUSES,
        build_source_inventory,
        load_source_catalog,
        save_run_triage,
        source_sample_path,
    )
except ImportError:  # pragma: no cover - direct execution from darsia.fluidflower.tools/
    from darsia.fluidflower.tools.overflow_source_inventory import (
        DEFAULT_RAW_ROOT,
        FINAL_PRODUCTION_RUNS,
        TRIAGE_STATUSES,
        build_source_inventory,
        load_source_catalog,
        save_run_triage,
        source_sample_path,
    )


REPO_ROOT = series_root("ffac")
DEFAULT_CONFIG = project_path('config/overflow_scale_runs.json')
DEFAULT_OUTPUT = configured_tool_path('ffac', 'calibration_logs') / 'overflow_weight_review'
DEFAULT_SINGLE_SAMPLE_COUNT = 110
DEFAULT_SINGLE_MAX_HOURS = 15.0
DEFAULT_REVIEWER = os.environ.get("FFAC_DEFAULT_REVIEWER", "Olav Folkvord")
INVALID_WEIGHT_DATA_STATUS = "invalid-weight-data"
DEFAULT_RESET_REASON = (
    "Standardize manual overflow review to 110 images per run from the common "
    "pre-injection baseline through 15 h after injection start"
)
ARTIFACT_BUILDER = project_path('scripts/overflow_workbook_builder.mjs')
TIMESTAMP_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")
SCALE_READING_RE = re.compile(r"^([0-9xX]{1,4})[.,]([0-9xX])$")
REVIEWED_STATUSES = {"Verified", "Corrected", "Partial", "Unreadable"}
CONFIRMED_STATUSES = {"Verified", "Corrected"}
ALL_STATUSES = REVIEWED_STATUSES | {"Pending", "Not selected"}
IMAGE_MANIFEST_SCHEMA = 1
PREDICTION_CACHE_SCHEMA = 1
OCR_EVIDENCE_CACHE_SCHEMA = 1
AUTO_GEOMETRY_CACHE_SCHEMA = 2
PREDICTOR_NAME = "ffac-overflow-seven-segment-continuity"
PREDICTOR_VERSION = "3.4.0"
LEGACY_PREDICTOR_VERSION = "1.0.0"
OCR_EXTRACTOR_NAME = "ffac-overflow-seven-segment-green-difference"
OCR_EXTRACTOR_VERSION = "3.2.0"
SHARED_OCR_MODEL_NAME = "ffac-overflow-shared-cell-classifier"
SHARED_OCR_MODEL_VERSION = "1.0.0"
AUTO_GEOMETRY_DETECTOR_NAME = "ffac-overflow-display-grid"
AUTO_GEOMETRY_DETECTOR_VERSION = "1.1.0"
try:
    OCR_TRAINING_JOBS = int(os.environ.get("FFAC_OCR_TRAINING_JOBS", "-1"))
except ValueError:
    OCR_TRAINING_JOBS = -1
if OCR_TRAINING_JOBS == 0:
    OCR_TRAINING_JOBS = -1
ARTIFACT_BUILD_SLOTS = threading.Semaphore(2)
RUN_PAYLOAD_STAT_TTL_S = 30.0
INITIAL_ANCHOR_FRACTION = 0.10
INITIAL_ANCHOR_MINIMUM = 3
MONOTONIC_DROP_TOLERANCE_G = 0.3
PLATEAU_TOLERANCE_G = 0.3
PLATEAU_MAX_RATE_G_PER_H = 15.0
ACTIVE_FEED_MIN_RATE_G_PER_H = 50.0
PLATEAU_PRIOR_SOURCES = frozenset(
    {"verified-post-feed-plateau", "inferred-post-feed-plateau"}
)
LINEAR_TREND_PRIOR_SOURCES = frozenset(
    {
        "interpolated-linear-trend",
        "piecewise-knee-trend",
        "piecewise-outlier-resistant-trend",
    }
)
EXPORT_FIELDS = (
    "index",
    "image_id",
    "image_time",
    "time_from_injection_h",
    "image",
    "relative_path",
    "suggested_g",
    "confidence",
    "confidence_score",
    "expected_g",
    "prediction_model_id",
    "prediction_generated_at",
    "prediction_training_snapshot_id",
    "prediction_training_count",
    "prediction_partial_count",
    "prediction_reading_pattern",
    "prediction_prior_source",
    "prediction_prior_weight",
    "prediction_image_gate",
    "prediction_image_only_g",
    "prediction_bounded_image_g",
    "prediction_prior_changed_choice",
    "prediction_physical_bounds_changed_choice",
    "prediction_physical_min_g",
    "prediction_physical_max_g",
    "prediction_physical_bounds_source",
    "prediction_trend_knee_time",
    "prediction_trend_active_rate_g_per_h",
    "prediction_trend_post_rate_g_per_h",
    "prediction_trend_residual_scale_g",
    "prediction_trend_loss_ratio",
    "shared_ocr_model_id",
    "shared_ocr_training_count",
    "shared_ocr_training_cells",
    "shared_ocr_training_runs",
    "ocr_extractor_id",
    "prediction_error_g",
    "existing_value_g",
    "existing_source",
    "selected_for_review",
    "review_round_id",
    "review_position",
    "review_selection_phase",
    "weight_g",
    "reading_pattern",
    "status",
    "verified_at",
    "reviewer",
    "notes",
)

SEGMENT_BOXES = (
    (0.00, 0.18, 0.18, 0.82),
    (0.08, 0.48, 0.68, 1.00),
    (0.52, 0.92, 0.68, 1.00),
    (0.82, 1.00, 0.18, 0.82),
    (0.52, 0.92, 0.00, 0.32),
    (0.08, 0.48, 0.00, 0.32),
    (0.41, 0.59, 0.18, 0.82),
)
SEGMENT_PATTERNS = {
    "0": (1, 1, 1, 1, 1, 1, 0),
    "1": (0, 1, 1, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1),
    "3": (1, 1, 1, 1, 0, 0, 1),
    "4": (0, 1, 1, 0, 0, 1, 1),
    "5": (1, 0, 1, 1, 0, 1, 1),
    "6": (1, 0, 1, 1, 1, 1, 1),
    "7": (1, 1, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
}
OCR_SYMBOLS = tuple(SEGMENT_PATTERNS) + (" ",)


def _json_compatible(value: Any) -> Any:
    """Return strict-JSON data, replacing non-finite diagnostics with null."""
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _json_text(payload: Any, **kwargs: Any) -> str:
    return json.dumps(_json_compatible(payload), allow_nan=False, **kwargs)


def _has_nonfinite_json_number(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite_json_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite_json_number(item) for item in value)
    return False


def _repair_nonfinite_json(path: Path, payload: Any) -> bool:
    if not _has_nonfinite_json_number(payload):
        return False
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return True


@dataclass(frozen=True)
class Frame:
    time: datetime
    path: Path
    file_size: int | None = None


@dataclass(frozen=True)
class CellGeometry:
    x: tuple[int, ...]
    y: int
    width: int
    height: int
    horizontal_roll: int = 0


@dataclass(frozen=True)
class ScaleReading:
    canonical: str
    symbols: str
    weight_g: float | None

    @property
    def is_partial(self) -> bool:
        return self.weight_g is None


@dataclass(frozen=True)
class RunSpec:
    run: str
    image_root: Path
    gravity_feed_g_per_h: float
    gravity_start: datetime
    gravity_duration_h: float
    gravity_start_source: str
    cells: CellGeometry | None
    gravity_schedule_verified: bool = True
    configuration_source: str = "config"
    image_time_offset_h: float = 0.0
    injection_start: datetime | None = None
    injection_end: datetime | None = None
    image_roots: tuple[Path, ...] = ()
    cells_by_image_size: tuple[tuple[int, int, CellGeometry], ...] = ()
    screen_boxes_by_image_size: tuple[
        tuple[int, int, tuple[int, int, int, int]], ...
    ] = ()

    @property
    def gravity_end(self) -> datetime:
        return self.gravity_start + timedelta(hours=self.gravity_duration_h)

    @property
    def ocr_configured(self) -> bool:
        return self.cells is not None or bool(self.cells_by_image_size)

    @property
    def automatic_ocr(self) -> bool:
        return self.ocr_configured and self.configuration_source.startswith(
            "automatic-ocr:"
        )

    @property
    def ocr_cell_count(self) -> int:
        if self.cells is not None:
            return len(self.cells.x)
        if self.cells_by_image_size:
            return len(self.cells_by_image_size[0][2].x)
        return 0

    def cells_for_image_size(
        self, image_width: int, image_height: int
    ) -> CellGeometry | None:
        for width, height, geometry in self.cells_by_image_size:
            if width == image_width and height == image_height:
                return geometry
        # A configured fixed geometry remains valid for legacy single-layout runs.
        if not self.cells_by_image_size:
            return self.cells
        return None

    def screen_box_for_image_size(
        self, image_width: int, image_height: int
    ) -> tuple[int, int, int, int] | None:
        for width, height, screen_box in self.screen_boxes_by_image_size:
            if width == image_width and height == image_height:
                return screen_box
        return None


@dataclass(frozen=True)
class PiecewiseWeightTrend:
    origin: datetime
    knee_h: float
    intercept_g: float
    active_rate_g_per_h: float
    post_rate_g_per_h: float
    residual_scale_g: float
    loss_ratio: float
    support_count: int
    first_h: float
    last_h: float

    @property
    def knee_time(self) -> datetime:
        return self.origin + timedelta(hours=self.knee_h)

    def predict(self, timestamp: datetime) -> float:
        elapsed_h = (timestamp - self.origin).total_seconds() / 3600.0
        active_h = min(elapsed_h, self.knee_h)
        post_h = max(0.0, elapsed_h - self.knee_h)
        return max(
            0.0,
            self.intercept_g
            + self.active_rate_g_per_h * active_h
            + self.post_rate_g_per_h * post_h,
        )


@dataclass
class RunInventory:
    run: str
    frames: list[Frame]
    injection_start: datetime
    injection_end: datetime
    effective_cutoff: datetime
    cutoff_rule: str
    confirmed_count: int
    reviewed_count: int
    capacity: int
    existing_workbook: Path | None
    inventory_source: str
    prediction_model_id: str
    predicted_count: int
    missing_prediction_count: int
    eligible_image_ids: frozenset[str] | None = None


def parse_image_timestamp(path: Path) -> datetime | None:
    match = TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    try:
        return datetime(*map(int, match.groups()))
    except ValueError:
        return None


def discover_frames(root: Path, image_time_offset_h: float = 0.0) -> list[Frame]:
    frames: list[Frame] = []
    time_offset = timedelta(hours=float(image_time_offset_h))
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".bmp",
            ".jpg",
            ".jpeg",
            ".png",
        }:
            continue
        timestamp = parse_image_timestamp(path)
        if timestamp is not None:
            try:
                file_size = path.stat().st_size
            except OSError:
                file_size = None
            frames.append(Frame(timestamp + time_offset, path, file_size))
    frames.sort(key=lambda frame: (frame.time, str(frame.path).lower()))
    return frames


def _configured_image_roots(spec: RunSpec) -> tuple[Path, ...]:
    return spec.image_roots or (spec.image_root,)


def discover_spec_frames(spec: RunSpec) -> list[Frame]:
    """Discover only explicitly configured roots for one logical experiment."""

    by_path: dict[str, Frame] = {}
    for root in _configured_image_roots(spec):
        for frame in discover_frames(root, spec.image_time_offset_h):
            by_path[_normalise_path(frame.path)] = frame
    frames = list(by_path.values())
    frames.sort(key=lambda frame: (frame.time, str(frame.path).lower()))
    return frames


def _image_roots_payload(spec: RunSpec) -> list[str]:
    return [str(path) for path in _configured_image_roots(spec)]


def _image_roots_match(payload: dict[str, Any], spec: RunSpec) -> bool:
    stored = payload.get("image_roots")
    if stored is None:
        return len(_configured_image_roots(spec)) == 1
    return {_normalise_path(value) for value in stored if str(value).strip()} == {
        _normalise_path(value) for value in _configured_image_roots(spec)
    }


def _cell_configuration(spec: RunSpec) -> dict[str, Any] | None:
    if not spec.ocr_configured:
        return None
    default = (
        {
            "x": spec.cells.x,
            "y": spec.cells.y,
            "width": spec.cells.width,
            "height": spec.cells.height,
        }
        if spec.cells is not None
        else None
    )
    if not spec.cells_by_image_size:
        return default
    return {
        "default": default,
        "by_image_size": [
            {
                "image_size": [width, height],
                "x": geometry.x,
                "y": geometry.y,
                "width": geometry.width,
                "height": geometry.height,
                "reference_screen_box": spec.screen_box_for_image_size(width, height),
            }
            for width, height, geometry in spec.cells_by_image_size
        ],
    }


def _prediction_configuration_digest(spec: RunSpec) -> str:
    configuration = {
        "run": spec.run,
        "cells": _cell_configuration(spec),
        "gravity_feed_g_per_h": spec.gravity_feed_g_per_h,
        "gravity_start": spec.gravity_start.isoformat(),
        "gravity_duration_h": spec.gravity_duration_h,
        "image_time_offset_h": spec.image_time_offset_h,
        "ocr_extractor": (
            f"{OCR_EXTRACTOR_NAME}@{OCR_EXTRACTOR_VERSION}"
            if spec.ocr_configured
            else "manual-review-no-ocr"
        ),
    }
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]


def _legacy_prediction_configuration_digest(spec: RunSpec) -> str:
    configuration = {
        "cells": _cell_configuration(spec),
        "gravity_feed_g_per_h": spec.gravity_feed_g_per_h,
        "gravity_start": spec.gravity_start.isoformat(),
        "gravity_duration_h": spec.gravity_duration_h,
    }
    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]


def ocr_extractor_id(spec: RunSpec) -> str:
    if not spec.ocr_configured:
        return "manual-review-no-ocr"
    configuration = {"cells": _cell_configuration(spec)}
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{OCR_EXTRACTOR_NAME}@{OCR_EXTRACTOR_VERSION}+{digest}"


def prediction_training_snapshot(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze exact anchors and partial patterns for one predictor release."""
    anchors: list[dict[str, Any]] = []
    partial_patterns: list[dict[str, Any]] = []
    for record in records:
        weight = _float_or_none(record.get("weight_g"))
        status = str(record.get("status"))
        image_time = str(record.get("image_time") or "")
        if status == "Partial":
            try:
                partial = parse_scale_reading(record.get("reading_pattern"))
                timestamp = datetime.fromisoformat(image_time)
            except (TypeError, ValueError):
                continue
            if not partial.is_partial:
                continue
            partial_patterns.append(
                {
                    "image_id": str(record.get("image_id") or ""),
                    "image_time": timestamp.isoformat(),
                    "reading_pattern": partial.canonical,
                }
            )
            continue
        if status not in CONFIRMED_STATUSES or weight is None:
            continue
        try:
            timestamp = datetime.fromisoformat(image_time)
        except ValueError:
            continue
        anchors.append(
            {
                "image_id": str(record.get("image_id") or ""),
                "image_time": timestamp.isoformat(),
                "weight_g": weight,
            }
        )
    anchors.sort(key=lambda item: (item["image_time"], item["image_id"]))
    partial_patterns.sort(key=lambda item: (item["image_time"], item["image_id"]))
    frozen_inputs = {
        "anchors": anchors,
        "partial_patterns": partial_patterns,
    }
    digest = hashlib.sha256(
        json.dumps(frozen_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": digest,
        "count": len(anchors),
        "anchors": anchors,
        "partial_count": len(partial_patterns),
        "partial_patterns": partial_patterns,
    }


def prediction_model_id(
    spec: RunSpec,
    training_snapshot_id: str | None = None,
    shared_ocr_model_id: str | None = None,
) -> str:
    """Identify an immutable predictor release, including its frozen training data."""
    configuration_digest = _prediction_configuration_digest(spec)
    training_digest = str(
        training_snapshot_id or prediction_training_snapshot([])["id"]
    )[:12]
    model_id = (
        f"{PREDICTOR_NAME}@{PREDICTOR_VERSION}+{configuration_digest}"
        f".train-{training_digest}"
    )
    if shared_ocr_model_id:
        shared_digest = hashlib.sha256(shared_ocr_model_id.encode("utf-8")).hexdigest()[
            :12
        ]
        model_id += f".shared-{shared_digest}"
    return model_id


def _legacy_prediction_model_id(spec: RunSpec) -> str:
    return (
        f"{PREDICTOR_NAME}@{LEGACY_PREDICTOR_VERSION}"
        f"+{_legacy_prediction_configuration_digest(spec)}"
    )


def _active_prediction(record: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for prediction in reversed(record.get("predictions") or []):
        if str(prediction.get("model_id")) == model_id:
            return prediction
    return None


def _has_prediction(record: dict[str, Any], model_id: str) -> bool:
    return _active_prediction(record, model_id) is not None


def _legacy_prediction(
    record: dict[str, Any], model_id: str, generated_at: str | None
) -> dict[str, Any] | None:
    value = _float_or_none(record.get("suggested_g"))
    if value is None:
        return None
    return {
        "model_id": model_id,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "value_g": value,
        "expected_g": _float_or_none(record.get("expected_g")),
        "confidence": record.get("confidence"),
        "confidence_score": _float_or_none(record.get("confidence_score")),
        "fit": None,
        "margin": None,
        "image_cost": None,
        "migrated_from": "schema-2-unversioned",
    }


def _apply_prediction_aliases(
    record: dict[str, Any], prediction: dict[str, Any] | None
) -> None:
    record["suggested_g"] = prediction.get("value_g") if prediction else None
    record["expected_g"] = prediction.get("expected_g") if prediction else None
    record["confidence"] = prediction.get("confidence") if prediction else None
    record["confidence_score"] = (
        prediction.get("confidence_score") if prediction else None
    )
    record["prediction_model_id"] = prediction.get("model_id") if prediction else ""
    record["prediction_generated_at"] = (
        prediction.get("generated_at") if prediction else None
    )
    record["prediction_training_snapshot_id"] = (
        prediction.get("training_snapshot_id") if prediction else ""
    )
    record["prediction_training_count"] = (
        prediction.get("training_count") if prediction else None
    )
    record["prediction_partial_count"] = (
        prediction.get("partial_count") if prediction else None
    )
    record["prediction_reading_pattern"] = (
        prediction.get("reading_pattern") if prediction else None
    )
    record["prediction_prior_source"] = (
        prediction.get("prior_source") if prediction else None
    )
    record["prediction_prior_weight"] = (
        prediction.get("prior_weight") if prediction else None
    )
    record["prediction_image_gate"] = (
        prediction.get("image_gate") if prediction else None
    )
    record["prediction_image_only_g"] = (
        prediction.get("image_only_value_g") if prediction else None
    )
    record["prediction_bounded_image_g"] = (
        prediction.get("bounded_image_value_g") if prediction else None
    )
    record["prediction_prior_changed_choice"] = (
        prediction.get("prior_changed_choice") if prediction else None
    )
    record["prediction_physical_bounds_changed_choice"] = (
        prediction.get("physical_bounds_changed_choice") if prediction else None
    )
    record["prediction_physical_min_g"] = (
        prediction.get("physical_minimum_g") if prediction else None
    )
    record["prediction_physical_max_g"] = (
        prediction.get("physical_maximum_g") if prediction else None
    )
    record["prediction_physical_bounds_source"] = (
        prediction.get("physical_bounds_source") if prediction else None
    )
    record["prediction_trend_knee_time"] = (
        prediction.get("trend_knee_time") if prediction else None
    )
    record["prediction_trend_active_rate_g_per_h"] = (
        prediction.get("trend_active_rate_g_per_h") if prediction else None
    )
    record["prediction_trend_post_rate_g_per_h"] = (
        prediction.get("trend_post_rate_g_per_h") if prediction else None
    )
    record["prediction_trend_residual_scale_g"] = (
        prediction.get("trend_residual_scale_g") if prediction else None
    )
    record["prediction_trend_loss_ratio"] = (
        prediction.get("trend_loss_ratio") if prediction else None
    )
    record["shared_ocr_model_id"] = (
        prediction.get("shared_ocr_model_id") if prediction else ""
    )
    record["shared_ocr_training_count"] = (
        prediction.get("shared_ocr_training_count") if prediction else None
    )
    record["shared_ocr_training_cells"] = (
        prediction.get("shared_ocr_training_cells") if prediction else None
    )
    record["shared_ocr_training_runs"] = (
        prediction.get("shared_ocr_training_runs") if prediction else None
    )
    record["ocr_extractor_id"] = (
        prediction.get("ocr_extractor_id") if prediction else ""
    )


def _merge_predictions(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for group in groups:
        for prediction in group or []:
            model_id = str(prediction.get("model_id", "")).strip()
            if model_id:
                by_model[model_id] = copy.deepcopy(prediction)
    return sorted(
        by_model.values(),
        key=lambda item: (
            str(item.get("generated_at") or ""),
            str(item.get("model_id") or ""),
        ),
    )


def _manifest_records(
    frames: list[Frame], run: str, root: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame in frames:
        try:
            relative = str(frame.path.relative_to(root))
        except ValueError:
            relative = frame.path.name
        records.append(
            {
                "image_id": _frame_id(run, frame, root),
                "image_time": frame.time.isoformat(),
                "image": str(frame.path),
                "image_filename": frame.path.name,
                "relative_path": relative,
                "file_size": frame.file_size,
            }
        )
    return records


def _frames_from_records(records: list[dict[str, Any]]) -> list[Frame]:
    frames: list[Frame] = []
    for record in records:
        try:
            timestamp = datetime.fromisoformat(str(record["image_time"]))
            path = Path(str(record["image"]))
        except (KeyError, TypeError, ValueError):
            continue
        size = _float_or_none(record.get("file_size"))
        frames.append(Frame(timestamp, path, int(size) if size is not None else None))
    frames.sort(key=lambda frame: (frame.time, str(frame.path).lower()))
    return frames


def _write_image_manifest(path: Path, spec: RunSpec, frames: list[Frame]) -> None:
    payload = {
        "schema": IMAGE_MANIFEST_SCHEMA,
        "run": spec.run,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_root": str(spec.image_root),
        "image_roots": _image_roots_payload(spec),
        "image_time_offset_h": spec.image_time_offset_h,
        "records": _manifest_records(frames, spec.run, spec.image_root),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(_json_text(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_image_manifest(path: Path, spec: RunSpec) -> list[Frame]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if (
        payload.get("schema") != IMAGE_MANIFEST_SCHEMA
        or str(payload.get("run", "")).lower() != spec.run
        or _normalise_path(payload.get("image_root"))
        != _normalise_path(spec.image_root)
        or not _image_roots_match(payload, spec)
        or not math.isclose(
            float(payload.get("image_time_offset_h", 0.0)),
            spec.image_time_offset_h,
            abs_tol=1e-9,
        )
    ):
        return []
    return _frames_from_records(payload.get("records") or [])


def _load_prediction_cache(path: Path, spec: RunSpec) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        payload.get("schema") != PREDICTION_CACHE_SCHEMA
        or str(payload.get("run", "")).lower() != spec.run
        or _normalise_path(payload.get("image_root"))
        != _normalise_path(spec.image_root)
        or not _image_roots_match(payload, spec)
        or not math.isclose(
            float(payload.get("image_time_offset_h", 0.0)),
            spec.image_time_offset_h,
            abs_tol=1e-9,
        )
    ):
        return {}
    _repair_nonfinite_json(path, payload)
    return {
        str(record["image_id"]): record
        for record in payload.get("records") or []
        if record.get("image_id")
    }


def _write_prediction_cache(
    path: Path, spec: RunSpec, records_by_id: dict[str, dict[str, Any]]
) -> None:
    payload = {
        "schema": PREDICTION_CACHE_SCHEMA,
        "run": spec.run,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "image_root": str(spec.image_root),
        "image_roots": _image_roots_payload(spec),
        "image_time_offset_h": spec.image_time_offset_h,
        "predictor_name": PREDICTOR_NAME,
        "records": sorted(
            records_by_id.values(),
            key=lambda item: (str(item.get("image_time", "")), item["image_id"]),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(_json_text(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def _active_ocr_evidence(
    record: dict[str, Any] | None, extractor_id: str
) -> dict[str, Any] | None:
    if record is None:
        return None
    for evidence in reversed(record.get("evidence") or []):
        if str(evidence.get("extractor_id")) == extractor_id:
            return evidence
    return None


def _merge_ocr_evidence(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_extractor: dict[str, dict[str, Any]] = {}
    for group in groups:
        for evidence in group or []:
            extractor_id = str(evidence.get("extractor_id", "")).strip()
            if extractor_id:
                by_extractor[extractor_id] = copy.deepcopy(evidence)
    return sorted(
        by_extractor.values(),
        key=lambda item: (
            str(item.get("generated_at") or ""),
            str(item.get("extractor_id") or ""),
        ),
    )


def _load_ocr_evidence_cache(path: Path, spec: RunSpec) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        payload.get("schema") != OCR_EVIDENCE_CACHE_SCHEMA
        or str(payload.get("run", "")).lower() != spec.run
        or _normalise_path(payload.get("image_root"))
        != _normalise_path(spec.image_root)
        or not _image_roots_match(payload, spec)
        or not math.isclose(
            float(payload.get("image_time_offset_h", 0.0)),
            spec.image_time_offset_h,
            abs_tol=1e-9,
        )
    ):
        return {}
    records_by_id = {
        str(record["image_id"]): record
        for record in payload.get("records") or []
        if record.get("image_id")
    }
    dirty = False
    for image_id, record in list(records_by_id.items()):
        evidence = list(record.get("evidence") or [])
        valid_evidence = [
            item for item in evidence if not _has_nonfinite_json_number(item)
        ]
        if len(valid_evidence) == len(evidence):
            continue
        repaired = dict(record)
        repaired["evidence"] = valid_evidence
        records_by_id[image_id] = repaired
        dirty = True
    if dirty:
        _write_ocr_evidence_cache(path, spec, records_by_id)
    return records_by_id


def _write_ocr_evidence_cache(
    path: Path, spec: RunSpec, records_by_id: dict[str, dict[str, Any]]
) -> None:
    payload = {
        "schema": OCR_EVIDENCE_CACHE_SCHEMA,
        "run": spec.run,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "image_root": str(spec.image_root),
        "image_roots": _image_roots_payload(spec),
        "image_time_offset_h": spec.image_time_offset_h,
        "extractor_name": OCR_EXTRACTOR_NAME,
        "records": sorted(
            records_by_id.values(),
            key=lambda item: (str(item.get("image_time", "")), item["image_id"]),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(_json_text(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _image_time_offsets(payload: dict[str, Any]) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for run, value in (payload.get("overflow_image_time_offsets_h") or {}).items():
        offset = float(value)
        if not math.isfinite(offset):
            raise ValueError(f"Non-finite overflow image time offset for {run}")
        offsets[str(run).lower()] = offset
    return offsets


def load_image_time_offsets(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError(
            f"Unsupported overflow-scale config schema: {payload.get('schema')}"
        )
    return _image_time_offsets(payload)


def load_specs(path: Path) -> dict[str, RunSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError(
            f"Unsupported overflow-scale config schema: {payload.get('schema')}"
        )
    image_time_offsets = _image_time_offsets(payload)
    specs: dict[str, RunSpec] = {}
    for run, raw in payload["runs"].items():
        cells = raw.get("cells")
        specs[run.lower()] = RunSpec(
            run=run.lower(),
            image_root=Path(raw["image_root"]),
            gravity_feed_g_per_h=float(raw["gravity_feed_g_per_h"]),
            gravity_start=datetime.fromisoformat(raw["gravity_start"]),
            gravity_duration_h=float(raw["gravity_duration_h"]),
            gravity_start_source=str(raw.get("gravity_start_source", "")),
            cells=(
                CellGeometry(
                    x=tuple(int(value) for value in cells["x"]),
                    y=int(cells["y"]),
                    width=int(cells["width"]),
                    height=int(cells["height"]),
                )
                if cells is not None
                else None
            ),
            gravity_schedule_verified=bool(raw.get("gravity_schedule_verified", True)),
            configuration_source=str(raw.get("configuration_source", "config")),
            image_time_offset_h=image_time_offsets.get(run.lower(), 0.0),
            injection_start=(
                datetime.fromisoformat(str(raw["injection_start"]))
                if str(raw.get("injection_start") or "").strip()
                else None
            ),
            injection_end=(
                datetime.fromisoformat(str(raw["injection_end"]))
                if str(raw.get("injection_end") or "").strip()
                else None
            ),
            image_roots=tuple(
                Path(value)
                for value in raw.get("image_roots") or [raw["image_root"]]
                if str(value).strip()
            ),
        )
    return specs


def provisional_manual_spec(
    run: str,
    source: dict[str, Any],
    image_time_offset_h: float | None = None,
) -> RunSpec | None:
    """Build a non-physical review-only spec from a scanned camera source."""
    root = str(source.get("selected_root") or "").strip()
    injection_text = str(source.get("injection_start") or "").strip()
    if (
        not root
        or not injection_text
        or int(source.get("relevant_image_count") or 0) <= 0
    ):
        return None
    if str(source.get("triage_status") or "") in {
        "no-images",
        "time-mismatch",
        INVALID_WEIGHT_DATA_STATUS,
    }:
        return None
    try:
        injection_start = datetime.fromisoformat(injection_text)
    except ValueError:
        return None
    return RunSpec(
        run=run.lower(),
        image_root=Path(root),
        gravity_feed_g_per_h=0.0,
        gravity_start=injection_start - timedelta(minutes=30),
        gravity_duration_h=48.5,
        gravity_start_source=(
            "Provisional manual-review window. The physical gravity-feed schedule "
            "has not been verified."
        ),
        cells=None,
        gravity_schedule_verified=False,
        configuration_source="source-inventory-manual-review",
        image_time_offset_h=float(
            source.get("image_time_offset_h", 0.0)
            if image_time_offset_h is None
            else image_time_offset_h
        ),
    )


def injection_window(run: str) -> tuple[datetime, datetime]:
    path = project_path('protocols') / run / "injection_protocol.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty injection protocol: {path}")
    starts = [datetime.fromisoformat(row["start"]) for row in rows]
    ends = [datetime.fromisoformat(row["end"]) for row in rows]
    return min(starts), max(ends)


def resolved_injection_window(spec: RunSpec) -> tuple[datetime, datetime]:
    """Prefer project-supplied timing while retaining legacy AC compatibility."""

    if spec.injection_start is not None:
        end = spec.injection_end or spec.injection_start
        return spec.injection_start, max(spec.injection_start, end)
    return injection_window(spec.run)


def sampling_times(injection_start: datetime, end_h: float = 24.0) -> list[datetime]:
    """Legacy physical anchor schedule kept for diagnostics and compatibility."""
    anchor_hours = (-0.5, 0.0, 2.391, 3.0, 4.1, 6.0, 8.0, 16.0, min(24.0, end_h))
    return [
        injection_start + timedelta(hours=value)
        for value in sorted(set(anchor_hours))
        if value <= end_h
    ]


def parse_scale_reading(value: Any) -> ScaleReading:
    """Parse one exact or partially unreadable five-cell scale reading."""
    if value is None or isinstance(value, bool):
        raise ValueError("Scale reading is required")
    if isinstance(value, (int, float)):
        weight = float(value)
        if not math.isfinite(weight) or not 0 <= weight <= 9999.9:
            raise ValueError("Scale reading must be between 0.0 and 9999.9 g")
        weight = round(weight, 1)
        return ScaleReading(f"{weight:.1f}", reading_string(weight), weight)

    text = str(value).strip().replace(" ", "").lower()
    if re.fullmatch(r"[0-9]{1,4}", text):
        text += ".0"
    match = SCALE_READING_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            "Use 1-4 digits or x before the decimal separator and one digit or x after it"
        )
    integer, fraction = match.groups()
    integer = re.sub(r"^x+", "x", integer.lower())
    canonical = f"{integer}.{fraction}".lower()
    symbols = integer.lower().rjust(4, " ") + fraction.lower()
    if "x" in canonical:
        return ScaleReading(canonical, symbols, None)
    weight = float(canonical)
    return ScaleReading(f"{weight:.1f}", reading_string(weight), weight)


def _apply_review_edit(
    record: dict[str, Any],
    reading: Any,
    unreadable: bool,
    reviewer: str,
    notes: str | None = None,
) -> None:
    if unreadable:
        status = "Unreadable"
        weight_g = None
        reading_pattern = ""
    else:
        parsed = parse_scale_reading(reading)
        weight_g = parsed.weight_g
        reading_pattern = parsed.canonical
        if parsed.is_partial:
            status = "Partial"
        else:
            prediction = _float_or_none(record.get("suggested_g"))
            status = (
                "Verified"
                if prediction is None or abs(float(weight_g) - prediction) < 0.051
                else "Corrected"
            )
    prediction = _float_or_none(record.get("suggested_g"))
    record.update(
        {
            "weight_g": weight_g,
            "reading_pattern": reading_pattern,
            "status": status,
            "verified_at": datetime.now().isoformat(timespec="seconds"),
            "reviewer": reviewer,
            "prediction_error_g": (
                None
                if weight_g is None or prediction is None
                else float(weight_g) - prediction
            ),
        }
    )
    if notes is not None:
        record["notes"] = str(notes)


def _write_review_payload(
    state_path: Path, csv_path: Path, payload: dict[str, Any]
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(_json_text(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, state_path)
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload.get("records", []))
    os.replace(temporary_csv, csv_path)


def reading_pattern_matches(pattern: str, value_g: float) -> bool:
    reading = parse_scale_reading(pattern)
    candidate = float(value_g)
    if not math.isfinite(candidate) or not 0 <= candidate <= 9999.9:
        return False
    return _reading_pattern_regex(reading).fullmatch(f"{candidate:.1f}") is not None


def _reading_pattern_regex(reading: ScaleReading) -> re.Pattern[str]:
    """Allow a flexible leading x; every internal x is exactly one digit."""
    integer, fraction = reading.canonical.split(".")
    pieces: list[str] = []
    for match in re.finditer(r"x+|[0-9]+", integer):
        token = match.group(0)
        if token.startswith("x"):
            pieces.append(
                r"(?:[0-9]?|[1-9][0-9]{1,2})"
                if match.start() == 0
                else rf"[0-9]{{{len(token)}}}"
            )
        else:
            pieces.append(re.escape(token))
    fractional_piece = r"[0-9]" if fraction == "x" else re.escape(fraction)
    return re.compile(r"^" + "".join(pieces) + r"\." + fractional_piece + r"$")


def reading_string(value_g: float) -> str:
    value_g = min(9999.9, max(0.0, round(float(value_g), 1)))
    return f"{value_g:6.1f}".replace(".", "")[-5:]


def active_feed_mass(spec: RunSpec, start: datetime, end: datetime) -> float:
    if end <= start:
        return 0.0
    active_start = max(start, spec.gravity_start)
    active_end = min(end, spec.gravity_end)
    if active_end <= active_start:
        return 0.0
    return (
        spec.gravity_feed_g_per_h * (active_end - active_start).total_seconds() / 3600.0
    )


def _robust_huber_fit(
    matrix: np.ndarray, values: np.ndarray, iterations: int = 8
) -> tuple[np.ndarray, float, float]:
    """Small deterministic IRLS fit used by the overflow trend model."""
    coefficients = np.linalg.lstsq(matrix, values, rcond=None)[0]
    for _ in range(iterations):
        residuals = values - matrix @ coefficients
        center = float(np.median(residuals))
        scale = max(
            PLATEAU_TOLERANCE_G,
            1.4826 * float(np.median(np.abs(residuals - center))),
        )
        normalized = np.abs(residuals - center) / (1.5 * scale)
        weights = np.ones_like(normalized)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(
            matrix * root_weights[:, None], values * root_weights, rcond=None
        )[0]
        if float(np.max(np.abs(updated - coefficients))) < 1e-7:
            coefficients = updated
            break
        coefficients = updated

    residuals = values - matrix @ coefficients
    center = float(np.median(residuals))
    scale = max(
        PLATEAU_TOLERANCE_G,
        1.4826 * float(np.median(np.abs(residuals - center))),
    )
    cutoff = 1.5 * scale
    absolute = np.abs(residuals - center)
    loss = float(
        np.sum(
            np.where(
                absolute <= cutoff,
                0.5 * absolute**2,
                cutoff * (absolute - 0.5 * cutoff),
            )
        )
    )
    return coefficients, scale, loss


def fit_piecewise_weight_trend(
    anchors: list[tuple[datetime, float]],
) -> PiecewiseWeightTrend | None:
    """Fit a robust linear-growth/knee/post-feed trend to confirmed readings."""
    grouped: dict[datetime, list[float]] = {}
    for timestamp, value in anchors:
        grouped.setdefault(timestamp, []).append(float(value))
    ordered = sorted(
        (timestamp, float(np.median(values))) for timestamp, values in grouped.items()
    )
    if len(ordered) < 6:
        return None
    origin = ordered[0][0]
    elapsed_h = np.asarray(
        [(timestamp - origin).total_seconds() / 3600.0 for timestamp, _ in ordered],
        dtype=float,
    )
    values = np.asarray([value for _, value in ordered], dtype=float)
    duration_h = float(elapsed_h[-1] - elapsed_h[0])
    if duration_h < 2.0:
        return None

    linear_matrix = np.column_stack((np.ones_like(elapsed_h), elapsed_h))
    _linear_coefficients, _linear_scale, linear_loss = _robust_huber_fit(
        linear_matrix, values
    )
    best: tuple[float, float, np.ndarray, float] | None = None
    for knee_h in np.linspace(0.30 * duration_h, 0.95 * duration_h, 96):
        if int(np.count_nonzero(elapsed_h <= knee_h)) < 3:
            continue
        if int(np.count_nonzero(elapsed_h >= knee_h)) < 2:
            continue
        matrix = np.column_stack(
            (
                np.ones_like(elapsed_h),
                elapsed_h,
                np.maximum(0.0, elapsed_h - knee_h),
            )
        )
        coefficients, scale, loss = _robust_huber_fit(matrix, values)
        active_rate = float(coefficients[1])
        post_rate = float(coefficients[1] + coefficients[2])
        if not 50.0 <= active_rate <= 1000.0:
            continue
        if not -10.0 <= post_rate <= min(150.0, 0.45 * active_rate):
            continue
        if best is None or loss < best[0]:
            best = (loss, float(knee_h), coefficients, float(scale))
    if best is None:
        return None

    loss, knee_h, coefficients, scale = best
    loss_ratio = loss / max(linear_loss, 1e-9)
    if loss_ratio > 0.50:
        return None
    return PiecewiseWeightTrend(
        origin=origin,
        knee_h=knee_h,
        intercept_g=float(coefficients[0]),
        active_rate_g_per_h=float(coefficients[1]),
        post_rate_g_per_h=float(coefficients[1] + coefficients[2]),
        residual_scale_g=scale,
        loss_ratio=loss_ratio,
        support_count=len(ordered),
        first_h=float(elapsed_h[0]),
        last_h=float(elapsed_h[-1]),
    )


def temporal_plateau_source(
    spec: RunSpec,
    anchors: list[tuple[datetime, float]],
    timestamp: datetime,
) -> str | None:
    """Identify a supported post-feed plateau, not a momentary flat OCR reading."""
    ordered = sorted(anchors, key=lambda item: item[0])
    if len(ordered) < 2:
        return None

    # A confirmed scale reset starts a new monotonic segment. Never infer a
    # plateau by joining readings across such a reset.
    segment_start = 0
    for index in range(1, len(ordered)):
        if ordered[index][0] > timestamp:
            break
        if ordered[index][1] < ordered[index - 1][1] - 2.0:
            segment_start = index
    segment = ordered[segment_start:]
    if len(segment) < 2:
        return None

    pairs: list[tuple[datetime, datetime, float]] = []
    for (left_time, left_value), (right_time, right_value) in zip(segment, segment[1:]):
        duration_h = (right_time - left_time).total_seconds() / 3600.0
        if duration_h > 0:
            pairs.append(
                (left_time, right_time, (right_value - left_value) / duration_h)
            )
    if not pairs:
        return None

    local_index: int | None = None
    for index, (left_time, right_time, _slope) in enumerate(pairs):
        if left_time <= timestamp <= right_time:
            local_index = index
            break
    if local_index is None and timestamp >= pairs[-1][1]:
        local_index = len(pairs) - 1
    if local_index is None:
        return None

    if abs(pairs[local_index][2]) > PLATEAU_MAX_RATE_G_PER_H:
        return None
    flat_start = local_index
    flat_end = local_index
    while flat_start > 0 and abs(pairs[flat_start - 1][2]) <= PLATEAU_MAX_RATE_G_PER_H:
        flat_start -= 1
    while (
        flat_end + 1 < len(pairs)
        and abs(pairs[flat_end + 1][2]) <= PLATEAU_MAX_RATE_G_PER_H
    ):
        flat_end += 1

    plateau_start = pairs[flat_start][0]
    plateau_end = pairs[flat_end][1]
    plateau_duration_h = (plateau_end - plateau_start).total_seconds() / 3600.0
    # The first low-rate interval can still contain the curved transition from
    # active feed to a static display. Tight +/-0.3 g support starts only after
    # that complete interval has been confirmed.
    mature_plateau_start = pairs[flat_start][1]
    if timestamp < mature_plateau_start:
        return None
    prior_growth = any(
        slope >= ACTIVE_FEED_MIN_RATE_G_PER_H
        for _left, right, slope in pairs
        if right <= plateau_start
    )

    if spec.gravity_schedule_verified and timestamp >= spec.gravity_end:
        near_feed_end = any(
            abs((anchor_time - spec.gravity_end).total_seconds()) <= 2 * 3600
            for anchor_time, _value in segment
        )
        if near_feed_end:
            return "verified-post-feed-plateau"
    if prior_growth and plateau_duration_h >= 0.5:
        return "inferred-post-feed-plateau"
    return None


def prediction_window(
    spec: RunSpec,
    anchors: list[tuple[datetime, float]],
    timestamp: datetime,
    trend_model: PiecewiseWeightTrend | None = None,
) -> tuple[float, float, float, bool, str]:
    """Return a soft OCR prior without treating an unverified feed plan as truth."""
    anchors = sorted(anchors, key=lambda item: item[0])
    if trend_model is None:
        trend_model = fit_piecewise_weight_trend(anchors)
    plateau_source = temporal_plateau_source(spec, anchors, timestamp)
    anchor_times = [item[0] for item in anchors]
    position = bisect.bisect_right(anchor_times, timestamp)
    if spec.gravity_schedule_verified and not (0 < position < len(anchors)):
        anchor_index = position - 1
        if anchor_index < 0:
            expected = active_feed_mass(spec, spec.gravity_start, timestamp)
            feed_delta = expected
            anchored = False
        else:
            anchor_time, anchor_value = anchors[anchor_index]
            feed_delta = active_feed_mass(spec, anchor_time, timestamp)
            expected = max(0.0, anchor_value + feed_delta)
            anchored = True
        if anchored and plateau_source is not None:
            span = 2.0 * PLATEAU_TOLERANCE_G
            sigma = PLATEAU_TOLERANCE_G
            source = plateau_source
        elif anchored:
            span = max(20.0, abs(feed_delta) * 1.5 + 12.0)
            sigma = max(5.0, abs(feed_delta) * 0.5 + 4.0)
            source = "verified-reading-and-feed-plan"
        else:
            span = max(80.0, expected * 0.25 + 40.0)
            sigma = max(20.0, expected * 0.1 + 12.0)
            source = "verified-feed-plan"
        return expected, span, sigma, anchored, source

    if not anchors:
        return 5000.0, 5000.0, 2500.0, False, "image-only"

    if 0 < position < len(anchors):
        left_time, left_value = anchors[position - 1]
        right_time, right_value = anchors[position]
        duration = max(1.0, (right_time - left_time).total_seconds())
        fraction = (timestamp - left_time).total_seconds() / duration
        local_expected = left_value + fraction * (right_value - left_value)
        expected = local_expected
        if plateau_source is not None:
            return (
                max(0.0, expected),
                2.0 * PLATEAU_TOLERANCE_G,
                PLATEAU_TOLERANCE_G,
                True,
                plateau_source,
            )
        source = "interpolated-verified-readings"
        sigma = max(20.0, 0.35 * max(80.0, 0.30 * abs(right_value - left_value) + 40.0))
        if trend_model is not None:
            trend_expected = trend_model.predict(timestamp)
            knee_crossing = left_time < trend_model.knee_time < right_time
            elapsed_from_origin_h = (
                timestamp - trend_model.origin
            ).total_seconds() / 3600.0
            outlier_threshold = max(60.0, 8.0 * trend_model.residual_scale_g)
            model_supported = (
                trend_model.first_h <= elapsed_from_origin_h <= trend_model.last_h
            )
            left_is_outlier = (
                abs(left_value - trend_model.predict(left_time)) > outlier_threshold
            )
            right_is_outlier = (
                abs(right_value - trend_model.predict(right_time)) > outlier_threshold
            )
            if knee_crossing:
                expected = trend_expected
                sigma = max(5.0, min(20.0, 2.0 * trend_model.residual_scale_g))
                source = "piecewise-knee-trend"
            elif (
                model_supported
                and elapsed_from_origin_h >= 1.0
                and timestamp <= trend_model.knee_time - timedelta(hours=0.5)
                and left_is_outlier != right_is_outlier
            ):
                expected = trend_expected
                sigma = max(5.0, min(20.0, 2.0 * trend_model.residual_scale_g))
                source = "piecewise-outlier-resistant-trend"
            else:
                change_g = abs(right_value - left_value)
                sigma = max(
                    0.75,
                    min(
                        8.0,
                        0.25 * trend_model.residual_scale_g + 0.005 * change_g,
                    ),
                )
                source = "interpolated-linear-trend"
        span = max(20.0, 4.0 * sigma)
        return (
            max(0.0, expected),
            min(9999.9, span),
            sigma,
            True,
            source,
        )

    # Outside a bracket of confirmed readings, favour the recent monotonic
    # segment. This follows a changed feed rate or a scale reset much faster
    # than the previous median over the full experiment history.
    if position == len(anchors):
        segment_start = 0
        for index in range(1, len(anchors)):
            if anchors[index][1] < anchors[index - 1][1] - 2.0:
                segment_start = index
        recent = anchors[segment_start:][-4:]
        if len(recent) >= 3:
            pairwise_slopes: list[float] = []
            adjacent_slopes: list[float] = []
            for left_index, (left_time, left_value) in enumerate(recent[:-1]):
                for right_time, right_value in recent[left_index + 1 :]:
                    elapsed_h = (right_time - left_time).total_seconds() / 3600.0
                    if elapsed_h > 0:
                        pairwise_slopes.append((right_value - left_value) / elapsed_h)
            for (left_time, left_value), (right_time, right_value) in zip(
                recent, recent[1:]
            ):
                elapsed_h = (right_time - left_time).total_seconds() / 3600.0
                if elapsed_h > 0:
                    adjacent_slopes.append((right_value - left_value) / elapsed_h)

            if pairwise_slopes and adjacent_slopes:
                slope = min(1000.0, max(0.0, float(np.median(pairwise_slopes))))
                adjacent_median = float(np.median(adjacent_slopes))
                slope_mad = float(
                    np.median(np.abs(np.asarray(adjacent_slopes) - adjacent_median))
                )
                cadence_h = float(
                    np.median(
                        [
                            (right_time - left_time).total_seconds() / 3600.0
                            for (left_time, _), (right_time, _) in zip(
                                recent, recent[1:]
                            )
                        ]
                    )
                )
                anchor_time, anchor_value = recent[-1]
                elapsed_h = (timestamp - anchor_time).total_seconds() / 3600.0
                expected = max(0.0, anchor_value + slope * elapsed_h)
                if plateau_source is not None:
                    return (
                        expected,
                        2.0 * PLATEAU_TOLERANCE_G,
                        PLATEAU_TOLERANCE_G,
                        True,
                        plateau_source,
                    )
                stable_rate = slope_mad <= max(20.0, 0.5 * max(slope, 20.0))
                within_horizon = elapsed_h <= max(2.0, 4.0 * cadence_h) or slope <= 10.0
                if stable_rate and within_horizon:
                    uncertainty_rate = max(120.0, 0.35 * slope)
                    span = min(
                        9999.9,
                        max(500.0, abs(elapsed_h) * uncertainty_rate + 500.0),
                    )
                    return (
                        expected,
                        span,
                        max(60.0, 0.40 * span),
                        True,
                        "local-trend-verified-readings",
                    )

    slopes: list[float] = []
    for (left_time, left_value), (right_time, right_value) in zip(anchors, anchors[1:]):
        elapsed_h = (right_time - left_time).total_seconds() / 3600.0
        if elapsed_h > 0:
            slopes.append((right_value - left_value) / elapsed_h)
    slope = float(np.median(slopes)) if slopes else 0.0
    slope = min(1000.0, max(0.0, slope))
    anchor_time, anchor_value = anchors[0] if position == 0 else anchors[-1]
    elapsed_h = (timestamp - anchor_time).total_seconds() / 3600.0
    expected = max(0.0, anchor_value + slope * elapsed_h)
    if plateau_source is not None:
        return (
            expected,
            2.0 * PLATEAU_TOLERANCE_G,
            PLATEAU_TOLERANCE_G,
            True,
            plateau_source,
        )
    uncertainty_rate = 650.0 if len(anchors) == 1 else max(120.0, 0.35 * slope)
    span = min(
        9999.9,
        max(500.0, abs(elapsed_h) * uncertainty_rate + 500.0),
    )
    return (
        expected,
        span,
        max(60.0, 0.40 * span),
        True,
        "extrapolated-verified-readings",
    )


def prediction_prior_profile(
    prior_source: str,
    learned_costs_available: bool,
) -> tuple[float, float]:
    """Weight temporal evidence only when it is backed by confirmed readings."""
    if (
        prior_source in LINEAR_TREND_PRIOR_SOURCES
        or prior_source in PLATEAU_PRIOR_SOURCES
    ):
        return 2.0, 4.0 if learned_costs_available else 2.0
    strong_sources = {
        "verified-reading-and-feed-plan",
        "interpolated-verified-readings",
        "local-trend-verified-readings",
    }
    if prior_source in strong_sources:
        return 0.35, 2.0 if learned_costs_available else 0.75
    return 0.05, 0.75 if learned_costs_available else 0.25


def physical_candidate_bounds(
    anchors: list[tuple[datetime, float]],
    timestamp: datetime,
    tolerance_g: float = MONOTONIC_DROP_TOLERANCE_G,
    plateau_expected_g: float | None = None,
) -> tuple[float, float, str]:
    """Bound accumulated overflow weight within human-confirmed monotonic segments."""
    if plateau_expected_g is not None:
        expected = min(9999.9, max(0.0, float(plateau_expected_g)))
        return (
            max(0.0, expected - PLATEAU_TOLERANCE_G),
            min(9999.9, expected + PLATEAU_TOLERANCE_G),
            "supported-post-feed-plateau",
        )
    if not anchors:
        return 0.0, 9999.9, "no-confirmed-bound"
    ordered = sorted(anchors, key=lambda item: item[0])
    times = [item[0] for item in ordered]
    position = bisect.bisect_right(times, timestamp)
    left = ordered[position - 1] if position > 0 else None
    right = ordered[position] if position < len(ordered) else None
    tolerance = max(0.0, float(tolerance_g))
    if left is not None and right is not None:
        left_value = float(left[1])
        right_value = float(right[1])
        if right_value + tolerance >= left_value:
            return (
                max(0.0, left_value - tolerance),
                min(9999.9, right_value + tolerance),
                "between-monotonic-confirmed-readings",
            )
        return (
            max(0.0, min(left_value, right_value) - tolerance),
            min(9999.9, max(left_value, right_value) + tolerance),
            "between-confirmed-reset-readings",
        )
    if left is not None:
        return (
            max(0.0, float(left[1]) - tolerance),
            9999.9,
            "after-last-confirmed-reading",
        )
    return (
        0.0,
        min(9999.9, float(right[1]) + tolerance),
        "before-first-confirmed-reading",
    )


def evenly_spaced_indices(indices: list[int], count: int) -> list[int]:
    """Choose exactly count chronological entries, spread over all eligible entries."""
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    positions = [round(i * (len(indices) - 1) / (count - 1)) for i in range(count)]
    return [indices[position] for position in positions]


def initial_anchor_count(records: list[dict[str, Any]], requested: int) -> int:
    """Reserve the first part of a new run for whole-window trend anchors."""
    if requested <= 0 or any(
        str(record.get("status", "")) in CONFIRMED_STATUSES for record in records
    ):
        return 0
    return min(
        int(requested),
        max(INITIAL_ANCHOR_MINIMUM, math.ceil(INITIAL_ANCHOR_FRACTION * requested)),
    )


def select_unreviewed_indices(records: list[dict[str, Any]], count: int) -> list[int]:
    """Select trend anchors first, then fill the largest temporal gaps."""
    eligible_records = [
        record
        for record in records
        if bool(record.get("eligible_for_review", True))
        and str(record.get("status", "Not selected")) not in REVIEWED_STATUSES
    ]
    requested = min(max(0, int(count)), len(eligible_records))
    if requested == 0:
        return []

    def coordinate(record: dict[str, Any]) -> float:
        value = record.get("time_from_injection_h")
        return float(value) if value is not None else float(record["index"])

    reviewed_coordinates = sorted(
        coordinate(record)
        for record in records
        if bool(record.get("eligible_for_review", True))
        and str(record.get("status", "Not selected")) in REVIEWED_STATUSES
    )
    coordinates_by_index = {
        int(record["index"]): coordinate(record) for record in eligible_records
    }

    def fill_largest_gaps(
        available_indices: list[int], anchors: list[float], amount: int
    ) -> list[int]:
        if amount <= 0 or not available_indices:
            return []
        ordered = sorted(
            available_indices, key=lambda index: (coordinates_by_index[index], index)
        )
        if not anchors:
            return evenly_spaced_indices(ordered, min(amount, len(ordered)))
        candidate_indices = np.asarray(ordered, dtype=np.int64)
        candidate_coordinates = np.asarray(
            [coordinates_by_index[index] for index in ordered], dtype=float
        )
        minimum_distance = np.full(candidate_coordinates.shape, np.inf, dtype=float)
        for anchor in anchors:
            minimum_distance = np.minimum(
                minimum_distance, np.abs(candidate_coordinates - anchor)
            )
        available = np.ones(candidate_coordinates.shape, dtype=bool)
        selected: list[int] = []
        while np.any(available) and len(selected) < amount:
            position = int(np.argmax(np.where(available, minimum_distance, -np.inf)))
            selected_index = int(candidate_indices[position])
            selected_coordinate = float(candidate_coordinates[position])
            selected.append(selected_index)
            available[position] = False
            minimum_distance = np.minimum(
                minimum_distance,
                np.abs(candidate_coordinates - selected_coordinate),
            )
        return selected

    eligible_indices = sorted(
        coordinates_by_index, key=lambda index: (coordinates_by_index[index], index)
    )
    bootstrap_count = initial_anchor_count(records, requested)
    if bootstrap_count:
        bootstrap = evenly_spaced_indices(eligible_indices, bootstrap_count)
        bootstrap_set = set(bootstrap)
        remaining = [index for index in eligible_indices if index not in bootstrap_set]
        gap_fill = fill_largest_gaps(
            remaining,
            reviewed_coordinates + [coordinates_by_index[index] for index in bootstrap],
            requested - len(bootstrap),
        )
        return bootstrap + sorted(
            gap_fill, key=lambda index: (coordinates_by_index[index], index)
        )

    selected = fill_largest_gaps(eligible_indices, reviewed_coordinates, requested)
    return sorted(selected, key=lambda index: (coordinates_by_index[index], index))


def allocate_review_counts(
    runs: list[str],
    requested_count: int,
    capacities: dict[str, int],
    confirmed_counts: dict[str, int],
    strategy: str,
) -> dict[str, int]:
    """Allocate a campaign total either evenly or by balancing confirmed totals."""
    if strategy not in {"even", "balance"}:
        raise ValueError(f"Unknown allocation strategy: {strategy}")
    ordered = sorted(
        dict.fromkeys(runs), key=lambda value: int(re.sub(r"\D", "", value) or 0)
    )
    allocations = {run: 0 for run in ordered}
    remaining = min(
        max(0, int(requested_count)),
        sum(max(0, capacities.get(run, 0)) for run in ordered),
    )
    while remaining:
        available = [
            run for run in ordered if allocations[run] < max(0, capacities.get(run, 0))
        ]
        if not available:
            break
        if strategy == "balance":
            selected = min(
                available,
                key=lambda run: (
                    confirmed_counts.get(run, 0) + allocations[run],
                    allocations[run],
                    ordered.index(run),
                ),
            )
        else:
            selected = min(
                available, key=lambda run: (allocations[run], ordered.index(run))
            )
        allocations[selected] += 1
        remaining -= 1
    return allocations


def _symbol_costs(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    variance = float(np.var(values)) + 1e-6
    result: dict[str, float] = {}
    for symbol, pattern_values in SEGMENT_PATTERNS.items():
        pattern = np.asarray(pattern_values, dtype=float)
        design = np.column_stack((np.ones(7), pattern))
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = values - design @ coefficients
        error = float(np.mean(residual * residual) / variance)
        if coefficients[1] < 0:
            error += 2.0
        result[symbol] = error
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    result[" "] = min(4.0, 2.0 * spread / max(float(np.mean(values)), 5.0))
    return result


def _cell_image_feature(signal: np.ndarray, position: int) -> tuple[float, ...]:
    """Create a compact, contrast-normalised feature for one display cell."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        signal = np.zeros((16, 8), dtype=np.float32)
    signal = np.clip(
        np.nan_to_num(signal, nan=0.0, posinf=255.0, neginf=0.0),
        0.0,
        255.0,
    )
    low, high = np.percentile(signal, (5.0, 99.0))
    normalised = np.clip((signal - low) / max(float(high - low), 1.0), 0.0, 1.0)
    source = np.ascontiguousarray(np.rint(normalised * 255.0).astype(np.uint8))
    resized = (
        np.asarray(
            Image.fromarray(source).resize((8, 16), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    statistics = (
        float(np.mean(signal)) / 100.0,
        float(np.std(signal)) / 100.0,
        float(np.percentile(signal, 90.0)) / 100.0,
        float(np.mean(signal > 10.0)),
    )
    position_encoding = tuple(1.0 if index == position else 0.0 for index in range(5))
    return tuple(map(float, resized.ravel())) + statistics + position_encoding


def _valid_cell_features(value: Any, expected_cells: int = 5) -> bool:
    try:
        features = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        features.ndim == 2
        and features.shape == (expected_cells, 137)
        and bool(np.all(np.isfinite(features)))
        and bool(np.max(np.abs(features)) <= 10.0)
    )


def _valid_symbol_costs(value: Any, expected_cells: int = 5) -> bool:
    if not isinstance(value, list) or len(value) != expected_cells:
        return False
    try:
        costs = np.asarray(
            [[cell.get(symbol, 10.0) for symbol in OCR_SYMBOLS] for cell in value],
            dtype=float,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return costs.shape == (expected_cells, len(OCR_SYMBOLS)) and bool(
        np.all(np.isfinite(costs))
    )


@lru_cache(maxsize=4096)
def _frame_ocr_data(
    path_text: str,
    x_positions: tuple[int, ...],
    y: int,
    width: int,
    height: int,
    horizontal_roll: int = 0,
) -> tuple[
    tuple[tuple[tuple[str, float], ...], ...],
    tuple[tuple[float, ...], ...],
]:
    """Read only the display crop; full camera frames are never retained in RAM."""
    left = min(x_positions)
    top = y
    right = max(x_positions) + width
    bottom = y + height
    with Image.open(path_text) as image:
        image = image.convert("RGB")
        if horizontal_roll:
            full_rgb = np.asarray(image)
            full_rgb = np.roll(full_rgb, horizontal_roll, axis=1)
            rgb = full_rgb[top:bottom, left:right].astype(np.float32)
        else:
            crop = image.crop((left, top, right, bottom))
            rgb = np.asarray(crop, dtype=np.float32)
    green_signal = np.clip(rgb[..., 1] - rgb[..., 0], 0.0, None)
    result: list[tuple[tuple[str, float], ...]] = []
    features: list[tuple[float, ...]] = []
    for position, x in enumerate(x_positions):
        local_x = x - left
        values: list[float] = []
        for y0, y1, x0, x1 in SEGMENT_BOXES:
            row0 = int(y0 * height)
            row1 = max(int(y1 * height), row0 + 1)
            col0 = local_x + int(x0 * width)
            col1 = local_x + max(int(x1 * width), int(x0 * width) + 1)
            segment = green_signal[row0:row1, col0:col1]
            values.append(float(np.mean(segment)) if segment.size else 0.0)
        result.append(tuple(_symbol_costs(np.asarray(values)).items()))
        features.append(
            _cell_image_feature(
                green_signal[:height, local_x : local_x + width], position
            )
        )
    return tuple(result), tuple(features)


def _frame_costs(
    path_text: str,
    x_positions: tuple[int, ...],
    y: int,
    width: int,
    height: int,
    horizontal_roll: int = 0,
) -> tuple[tuple[tuple[str, float], ...], ...]:
    return _frame_ocr_data(path_text, x_positions, y, width, height, horizontal_roll)[0]


def frame_costs(frame: Frame, geometry: CellGeometry) -> list[dict[str, float]]:
    packed = _frame_costs(
        str(frame.path),
        geometry.x,
        geometry.y,
        geometry.width,
        geometry.height,
        geometry.horizontal_roll,
    )
    return [dict(items) for items in packed]


def frame_cell_features(frame: Frame, geometry: CellGeometry) -> list[list[float]]:
    features = _frame_ocr_data(
        str(frame.path),
        geometry.x,
        geometry.y,
        geometry.width,
        geometry.height,
        geometry.horizontal_roll,
    )[1]
    return [list(values) for values in features]


@lru_cache(maxsize=2048)
def _image_size(path_text: str) -> tuple[int, int]:
    with Image.open(path_text) as image:
        return image.size


def _frame_geometry(frame: Frame, spec: RunSpec) -> CellGeometry | None:
    if not spec.cells_by_image_size:
        return spec.cells
    width, height = _image_size(str(frame.path))
    geometry = spec.cells_for_image_size(width, height)
    reference_box = spec.screen_box_for_image_size(width, height)
    if geometry is None or reference_box is None:
        return geometry
    try:
        with Image.open(frame.path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    except OSError:
        return geometry
    detected_box = _automatic_screen_box(rgb)
    horizontal_roll = 0
    if detected_box is None:
        horizontal_roll = _horizontal_display_roll(rgb) or 0
        if horizontal_roll:
            detected_box = _automatic_screen_box(np.roll(rgb, horizontal_roll, axis=1))
    if detected_box is None:
        return geometry
    _reference_left, reference_top, reference_right, _reference_bottom = reference_box
    _detected_left, detected_top, detected_right, _detected_bottom = detected_box
    # At a fixed camera resolution the PE11 display does not change scale.
    # Reflections do change the detected dark box's left edge, so align using
    # the much more stable right edge and translate the cells without scaling.
    delta_x = detected_right - reference_right
    delta_y = detected_top - reference_top
    return CellGeometry(
        x=tuple(x + delta_x for x in geometry.x),
        y=geometry.y + delta_y,
        width=geometry.width,
        height=geometry.height,
        horizontal_roll=horizontal_roll,
    )


def _automatic_screen_box(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the dark LCD around the densest green seven-segment signal."""
    rgb = np.asarray(rgb, dtype=np.int16)
    height, width = rgb.shape[:2]
    search = np.zeros((height, width), dtype=bool)
    search[
        int(0.15 * height) : int(0.92 * height),
        int(0.10 * width) : int(0.90 * width),
    ] = True
    # The PE11 display varies from green to cyan with exposure. Both colours
    # remain distinct from the warm surroundings through their G-R contrast.
    green_score = rgb[..., 1] - rgb[..., 0]
    positive = green_score[(green_score > 0) & search]
    if positive.size == 0:
        return None
    green_threshold = max(12.0, float(np.percentile(positive, 98.0)))
    green = (green_score >= green_threshold) & (rgb[..., 1] >= 45) & search
    if not np.any(green):
        return None
    density = ndimage.gaussian_filter(
        np.where(green, green_score, 0.0).astype(float),
        sigma=(6.0, max(6.0, width / 60.0)),
    )
    center_y, center_x = np.unravel_index(int(np.argmax(density)), density.shape)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    total_green = max(1, int(np.count_nonzero(green)))
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for threshold in range(20, 86, 5):
        labels, count = ndimage.label((luminance < threshold) & search)
        component_sizes = np.bincount(labels.ravel())
        for label_index, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None or component_sizes[label_index] < 100:
                continue
            rows, columns = slices
            top, bottom = int(rows.start), int(rows.stop)
            left, right = int(columns.start), int(columns.stop)
            box_width, box_height = right - left, bottom - top
            aspect = box_width / max(1, box_height)
            if not (
                2.5 <= aspect <= 7.0
                and 0.09 * width <= box_width <= 0.34 * width
                and 0.03 * height <= box_height <= 0.15 * height
                and left - 0.10 * box_width <= center_x <= right + 0.10 * box_width
                and top - 0.50 * box_height <= center_y <= bottom + 0.50 * box_height
            ):
                continue
            green_fraction = float(np.count_nonzero(green[slices])) / total_green
            if green_fraction < 0.20:
                continue
            shape_cost = (
                abs(math.log(box_width / (0.19 * width)))
                + abs(math.log(box_height / (0.075 * height)))
                + 0.30 * abs(math.log(aspect / 4.2))
                - 0.70 * green_fraction
            )
            candidates.append((shape_cost, (left, top, right, bottom)))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    # Strong reflections can make the LCD housing too bright for the dark-box
    # search above. The illuminated symbols still provide a stable fallback.
    local = green.copy()
    local[
        : max(0, int(center_y - 0.10 * height)),
        :,
    ] = False
    local[
        min(height, int(center_y + 0.10 * height)) :,
        :,
    ] = False
    local[:, : max(0, int(center_x - 0.20 * width))] = False
    local[:, min(width, int(center_x + 0.20 * width)) :] = False
    ys, xs = np.where(local)
    if xs.size < 20:
        return None
    bright_left, bright_right = int(xs.min()), int(xs.max()) + 1
    bright_top, bright_bottom = int(ys.min()), int(ys.max()) + 1
    bright_width = bright_right - bright_left
    bright_height = bright_bottom - bright_top
    if bright_width < 0.04 * width or bright_height < 0.01 * height:
        return None
    left = max(0, round(bright_left - 0.16 * bright_width))
    right = min(width, round(bright_right + 0.36 * bright_width))
    top = max(0, round(bright_top - 0.15 * bright_height))
    bottom = min(height, round(bright_bottom + 0.65 * bright_height))
    return left, top, right, bottom


def _horizontal_display_roll(rgb: np.ndarray) -> int | None:
    """Detect a display clipped or split at an edge by a torn camera frame."""
    rgb = np.asarray(rgb, dtype=np.int16)
    height, width = rgb.shape[:2]
    if width < 80 or height < 40:
        return None
    y0, y1 = int(0.15 * height), int(0.92 * height)
    green_score = rgb[..., 1] - rgb[..., 0]
    band = green_score[y0:y1]
    positive = band[band > 0]
    if positive.size == 0:
        return None
    threshold = max(12.0, float(np.percentile(positive, 98.0)))
    green = (green_score >= threshold) & (rgb[..., 1] >= 45)
    green[:y0] = False
    green[y1:] = False
    edge_width = max(12, int(0.10 * width))
    left_y, left_x = np.where(green[:, :edge_width])
    right_y, right_x = np.where(green[:, width - edge_width :])
    left_hit = left_y.size >= 8
    right_hit = right_y.size >= 8
    if not left_hit and not right_hit:
        return None
    if (
        left_hit
        and right_hit
        and abs(float(np.median(left_y)) - float(np.median(right_y))) > 0.08 * height
    ):
        return None
    # Moving the source seam to the image centre joins split halves and moves a
    # one-sided clipped display away from the output boundary.
    horizontal_roll = width // 2 if left_hit else -(width // 2)
    rolled = np.roll(rgb, horizontal_roll, axis=1)
    if _automatic_screen_box(rolled) is None:
        return None
    return horizontal_roll


def _candidate_segment_values(
    integral: np.ndarray,
    candidates: list[tuple[tuple[int, ...], int, int, int]],
) -> np.ndarray:
    values = np.empty((len(candidates), 5, len(SEGMENT_BOXES)), dtype=float)
    for candidate_index, (x_positions, y, width, height) in enumerate(candidates):
        for cell_index, x in enumerate(x_positions):
            for segment_index, (y0, y1, x0, x1) in enumerate(SEGMENT_BOXES):
                row0 = y + int(y0 * height)
                row1 = y + max(int(y1 * height), int(y0 * height) + 1)
                col0 = x + int(x0 * width)
                col1 = x + max(int(x1 * width), int(x0 * width) + 1)
                area = max(1, (row1 - row0) * (col1 - col0))
                values[candidate_index, cell_index, segment_index] = (
                    integral[row1, col1]
                    - integral[row0, col1]
                    - integral[row1, col0]
                    + integral[row0, col0]
                ) / area
    return values


def _candidate_digit_costs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variance = np.var(values, axis=2) + 1e-6
    costs: list[np.ndarray] = []
    for pattern_values in SEGMENT_PATTERNS.values():
        pattern = np.asarray(pattern_values, dtype=float)
        design = np.column_stack((np.ones(7), pattern))
        pseudo_inverse = np.linalg.pinv(design)
        coefficients = np.einsum("...j,kj->...k", values, pseudo_inverse)
        predicted = np.einsum("...k,jk->...j", coefficients, design)
        error = np.mean((values - predicted) ** 2, axis=2) / variance
        error += np.where(coefficients[..., 1] < 0, 2.0, 0.0)
        costs.append(error)
    digit_cost = np.min(np.stack(costs, axis=2), axis=2)
    spread = np.ptp(values, axis=2)
    blank_cost = np.minimum(
        4.0,
        2.0 * spread / np.maximum(np.mean(values, axis=2), 5.0),
    )
    return digit_cost, blank_cost


def detect_automatic_cell_geometry(
    frames: list[Frame], sample_count: int = 7
) -> tuple[CellGeometry, dict[str, Any]] | None:
    """Estimate the five fixed digit cells from representative scale images."""
    if not frames:
        return None
    indices = evenly_spaced_indices(
        list(range(len(frames))), min(sample_count, len(frames))
    )
    samples: list[tuple[Frame, np.ndarray, tuple[int, int, int, int]]] = []
    for index in indices:
        frame = frames[index]
        try:
            with Image.open(frame.path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
        except OSError:
            continue
        screen_box = _automatic_screen_box(rgb)
        if screen_box is not None:
            samples.append((frame, rgb, screen_box))
    if len(samples) < min(3, len(indices)):
        return None

    dimensions: dict[tuple[int, int], int] = {}
    for _, rgb, _ in samples:
        dimensions[rgb.shape[:2]] = dimensions.get(rgb.shape[:2], 0) + 1
    selected_shape = max(dimensions, key=dimensions.get)
    samples = [sample for sample in samples if sample[1].shape[:2] == selected_shape]
    if len(samples) < 3:
        return None

    boxes = np.asarray([sample[2] for sample in samples], dtype=float)
    left, top, right, bottom = map(int, np.rint(np.median(boxes, axis=0)).astype(int))
    screen_width, screen_height = right - left, bottom - top
    box_spread = float(
        np.max(np.median(np.abs(boxes - np.median(boxes, axis=0)), axis=0))
    )
    if screen_width < 40 or screen_height < 12:
        return None

    image_height, image_width = selected_shape
    candidates: list[tuple[tuple[int, ...], int, int, int]] = []
    for height_ratio in (0.48, 0.56, 0.64, 0.72):
        cell_height = max(8, round(screen_height * height_ratio))
        for width_ratio in (0.48, 0.58, 0.68):
            cell_width = max(5, round(cell_height * width_ratio))
            for pitch_ratio in (0.58, 0.68, 0.78, 0.88):
                pitch = max(cell_width + 1, round(cell_height * pitch_ratio))
                for y_ratio in (-0.04, 0.04, 0.12, 0.20, 0.28):
                    y = round(top + screen_height * y_ratio)
                    if y < 0 or y + cell_height >= image_height:
                        continue
                    for right_ratio in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
                        rightmost_x = round(left + screen_width * right_ratio)
                        x_positions = tuple(
                            rightmost_x - offset * pitch for offset in (4, 3, 2, 1, 0)
                        )
                        if (
                            x_positions[0] < left - 5
                            or x_positions[-1] + cell_width > right + 5
                            or x_positions[0] < 0
                            or x_positions[-1] + cell_width >= image_width
                        ):
                            continue
                        candidates.append((x_positions, y, cell_width, cell_height))
    if not candidates:
        return None

    frame_scores: list[np.ndarray] = []
    green_right_edges: list[float] = []
    for _, rgb, _ in samples:
        green_signal = np.clip(rgb[..., 1] - rgb[..., 0], 0.0, None).astype(float)
        display_signal = green_signal[top:bottom, left:right]
        positive_display = display_signal[display_signal > 0]
        if positive_display.size:
            display_threshold = max(6.0, float(np.percentile(positive_display, 70.0)))
            _, active_columns = np.where(display_signal >= display_threshold)
            if active_columns.size:
                green_right_edges.append(
                    left + float(np.percentile(active_columns, 99.0))
                )
        integral = np.pad(
            green_signal.cumsum(axis=0).cumsum(axis=1),
            ((1, 0), (1, 0)),
        )
        values = _candidate_segment_values(integral, candidates)
        digit_cost, blank_cost = _candidate_digit_costs(values)
        spread = np.ptp(values, axis=2)
        mean = np.mean(values, axis=2)
        score = (
            0.20 * np.sum(np.minimum(digit_cost[:, :3], blank_cost[:, :3]), axis=1)
            + digit_cost[:, 3]
            + digit_cost[:, 4]
            + 0.12 * np.sum(np.maximum(0.0, 5.0 - spread[:, 3:]), axis=1)
            + 0.08 * np.sum(np.maximum(0.0, 2.0 - mean[:, 3:]), axis=1)
            + 0.60 * np.sum(blank_cost[:, 3:] < digit_cost[:, 3:], axis=1)
            - 0.18
            * np.sum(
                (digit_cost < blank_cost) & (spread > 4.0),
                axis=1,
            )
        )
        frame_scores.append(score)
    stacked_scores = np.stack(frame_scores)
    aggregate_score = np.median(stacked_scores, axis=0) + 0.10 * np.mean(
        stacked_scores, axis=0
    )
    green_right = (
        float(np.median(green_right_edges)) if green_right_edges else float(right)
    )
    expected_unit_gap = 0.05 * screen_width
    candidate_right_edges = np.asarray(
        [item[0][-1] + item[2] for item in candidates], dtype=float
    )
    aggregate_score += 0.12 * np.abs(
        (green_right - candidate_right_edges) - expected_unit_gap
    )
    best_index = int(np.argmin(aggregate_score))
    score = float(aggregate_score[best_index])
    if not math.isfinite(score) or score > 2.5:
        return None
    x_positions, y, cell_width, cell_height = candidates[best_index]
    geometry = CellGeometry(x_positions, y, cell_width, cell_height)
    diagnostics = {
        "detector": f"{AUTO_GEOMETRY_DETECTOR_NAME}@{AUTO_GEOMETRY_DETECTOR_VERSION}",
        "score": score,
        "sample_count": len(samples),
        "sample_images": [str(sample[0].path) for sample in samples],
        "screen_box": [left, top, right, bottom],
        "display_green_right_x": green_right,
        "screen_box_median_absolute_deviation_px": box_spread,
        "image_size": [image_width, image_height],
    }
    return geometry, diagnostics


def detect_automatic_cell_geometries(frames: list[Frame], sample_count: int = 31) -> (
    tuple[
        CellGeometry,
        tuple[tuple[int, int, CellGeometry], ...],
        tuple[tuple[int, int, tuple[int, int, int, int]], ...],
        dict[str, Any],
    ]
    | None
):
    """Detect independent OCR layouts when camera resolution changes mid-run."""
    if not frames:
        return None
    indices = evenly_spaced_indices(
        list(range(len(frames))), min(sample_count, len(frames))
    )
    frames_by_size: dict[tuple[int, int], list[Frame]] = {}
    for index in indices:
        frame = frames[index]
        try:
            image_size = _image_size(str(frame.path))
        except OSError:
            continue
        frames_by_size.setdefault(image_size, []).append(frame)

    detected_by_size: dict[
        tuple[int, int], tuple[CellGeometry, dict[str, Any], int]
    ] = {}
    for image_size, sampled_frames in frames_by_size.items():
        if len(sampled_frames) < 3:
            continue
        detected = detect_automatic_cell_geometry(
            sampled_frames, sample_count=min(7, len(sampled_frames))
        )
        if detected is not None:
            geometry, diagnostics = detected
            detected_by_size[image_size] = (
                geometry,
                diagnostics,
                len(sampled_frames),
            )
    if not detected_by_size:
        return None

    primary_size = max(
        detected_by_size,
        key=lambda image_size: detected_by_size[image_size][2],
    )
    mappings = tuple(
        (width, height, detected_by_size[(width, height)][0])
        for width, height in sorted(detected_by_size)
    )
    screen_boxes = tuple(
        (
            width,
            height,
            tuple(
                int(value)
                for value in detected_by_size[(width, height)][1]["screen_box"]
            ),
        )
        for width, height in sorted(detected_by_size)
    )
    diagnostics = {
        "detector": f"{AUTO_GEOMETRY_DETECTOR_NAME}@{AUTO_GEOMETRY_DETECTOR_VERSION}",
        "sampled_frame_count": sum(len(group) for group in frames_by_size.values()),
        "primary_image_size": list(primary_size),
        "geometries": [
            {
                "image_size": [width, height],
                "sample_hits": detected_by_size[(width, height)][2],
                **detected_by_size[(width, height)][1],
            }
            for width, height in sorted(detected_by_size)
        ],
    }
    return detected_by_size[primary_size][0], mappings, screen_boxes, diagnostics


def _auto_geometry_cache_path(output_dir: Path, run: str) -> Path:
    return output_dir / f"{run}_overflow_auto_ocr_geometry.json"


def _load_auto_geometry(path: Path, spec: RunSpec) -> (
    tuple[
        CellGeometry,
        tuple[tuple[int, int, CellGeometry], ...],
        tuple[tuple[int, int, tuple[int, int, int, int]], ...],
        dict[str, Any],
    ]
    | None
):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = payload["cells"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    schema = payload.get("schema")
    if (
        schema not in {1, AUTO_GEOMETRY_CACHE_SCHEMA}
        or str(payload.get("run", "")).lower() != spec.run
        or _normalise_path(payload.get("image_root"))
        != _normalise_path(spec.image_root)
        or not _image_roots_match(payload, spec)
        or (
            schema == AUTO_GEOMETRY_CACHE_SCHEMA
            and payload.get("detector_version") != AUTO_GEOMETRY_DETECTOR_VERSION
        )
    ):
        return None
    try:
        geometry = CellGeometry(
            tuple(int(value) for value in cells["x"]),
            int(cells["y"]),
            int(cells["width"]),
            int(cells["height"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    diagnostics = dict(payload.get("diagnostics") or {})
    if schema == 1:
        # Preserve the former fixed-layout configuration and its OCR cache ID.
        return geometry, (), (), diagnostics
    try:
        geometries = tuple(
            (
                int(item["image_size"][0]),
                int(item["image_size"][1]),
                CellGeometry(
                    tuple(int(value) for value in item["cells"]["x"]),
                    int(item["cells"]["y"]),
                    int(item["cells"]["width"]),
                    int(item["cells"]["height"]),
                ),
            )
            for item in payload["geometries"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not geometries:
        return None
    try:
        screen_boxes = tuple(
            (
                int(item["image_size"][0]),
                int(item["image_size"][1]),
                tuple(int(value) for value in item["screen_box"]),
            )
            for item in payload["geometries"]
            if item.get("screen_box") is not None
        )
    except (KeyError, TypeError, ValueError):
        screen_boxes = ()
    if not screen_boxes:
        try:
            screen_boxes = tuple(
                (
                    int(item["image_size"][0]),
                    int(item["image_size"][1]),
                    tuple(int(value) for value in item["screen_box"]),
                )
                for item in diagnostics.get("geometries", [])
                if item.get("screen_box") is not None
            )
        except (KeyError, TypeError, ValueError):
            screen_boxes = ()
    return geometry, geometries, screen_boxes, diagnostics


def _write_auto_geometry(
    path: Path,
    spec: RunSpec,
    geometry: CellGeometry,
    geometries: tuple[tuple[int, int, CellGeometry], ...],
    screen_boxes: tuple[tuple[int, int, tuple[int, int, int, int]], ...],
    diagnostics: dict[str, Any],
) -> None:
    screen_box_by_size = {
        (width, height): screen_box for width, height, screen_box in screen_boxes
    }
    payload = {
        "schema": AUTO_GEOMETRY_CACHE_SCHEMA,
        "run": spec.run,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_root": str(spec.image_root),
        "image_roots": _image_roots_payload(spec),
        "detector_name": AUTO_GEOMETRY_DETECTOR_NAME,
        "detector_version": AUTO_GEOMETRY_DETECTOR_VERSION,
        "cells": {
            "x": list(geometry.x),
            "y": geometry.y,
            "width": geometry.width,
            "height": geometry.height,
        },
        "geometries": [
            {
                "image_size": [width, height],
                "screen_box": list(screen_box_by_size[(width, height)]),
                "cells": {
                    "x": list(item.x),
                    "y": item.y,
                    "width": item.width,
                    "height": item.height,
                },
            }
            for width, height, item in geometries
        ],
        "diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(_json_text(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def resolve_automatic_ocr(
    spec: RunSpec, frames: list[Frame], output_dir: Path
) -> tuple[RunSpec, dict[str, Any] | None]:
    if spec.ocr_configured:
        return spec, None
    cache_path = _auto_geometry_cache_path(output_dir, spec.run)
    cached = _load_auto_geometry(cache_path, spec)
    if cached is None:
        detected = detect_automatic_cell_geometries(frames)
        if detected is None:
            return spec, None
        geometry, geometries, screen_boxes, diagnostics = detected
        _write_auto_geometry(
            cache_path,
            spec,
            geometry,
            geometries,
            screen_boxes,
            diagnostics,
        )
    else:
        geometry, geometries, screen_boxes, diagnostics = cached
    return (
        replace(
            spec,
            cells=geometry,
            cells_by_image_size=geometries,
            screen_boxes_by_image_size=screen_boxes,
            configuration_source=(f"automatic-ocr:{AUTO_GEOMETRY_DETECTOR_VERSION}"),
        ),
        diagnostics,
    )


def _training_symbols(record: dict[str, Any]) -> tuple[str | None, ...] | None:
    """Return human-known display symbols without promoting machine proposals."""
    status = str(record.get("status") or "")
    weight = _float_or_none(record.get("weight_g"))
    if status in CONFIRMED_STATUSES and weight is not None:
        return tuple(reading_string(weight))
    if status != "Partial":
        return None
    try:
        reading = parse_scale_reading(record.get("reading_pattern"))
    except (TypeError, ValueError):
        return None
    integer, fraction = reading.canonical.split(".")
    symbols: list[str | None] = [None] * 5
    if integer.startswith("x"):
        suffix = integer[1:]
        start = 4 - len(suffix)
        for offset, symbol in enumerate(suffix):
            if symbol.isdigit():
                symbols[start + offset] = symbol
    else:
        for position, symbol in enumerate(integer.rjust(4)):
            if symbol == " " or symbol.isdigit():
                symbols[position] = symbol
    if fraction.isdigit():
        symbols[4] = fraction
    return tuple(symbols)


def _frame_evidence(frame: Frame, spec: RunSpec) -> dict[str, Any]:
    geometry = _frame_geometry(frame, spec)
    if geometry is None:
        raise ValueError(f"OCR geometry is unavailable for {frame.path.name}")
    return {
        "extractor_id": ocr_extractor_id(spec),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "symbol_costs": frame_costs(frame, geometry),
        "cell_features": frame_cell_features(frame, geometry),
    }


@dataclass(frozen=True)
class SharedOCRPredictionModel:
    model_id: str
    training_readings: int
    training_cells: int
    training_runs: int
    classifier: ExtraTreesClassifier

    def symbol_costs(self, features: list[list[float]]) -> list[dict[str, float]]:
        probabilities = self.classifier.predict_proba(np.asarray(features, dtype=float))
        classes = [str(value) for value in self.classifier.classes_]
        result: list[dict[str, float]] = []
        for row in probabilities:
            by_symbol = {symbol: 1e-6 for symbol in OCR_SYMBOLS}
            by_symbol.update(
                {
                    symbol: max(1e-6, float(probability))
                    for symbol, probability in zip(classes, row)
                }
            )
            costs = {symbol: -math.log(by_symbol[symbol]) for symbol in OCR_SYMBOLS}
            minimum = min(costs.values())
            result.append({symbol: value - minimum for symbol, value in costs.items()})
        return result


class SharedOCRTrainer:
    """Learn display-cell appearance from reviewed images across all runs."""

    def __init__(self, output_root: Path, specs: dict[str, RunSpec]):
        self.output_root = output_root
        self.lock = threading.RLock()
        self.samples_by_run: dict[str, list[dict[str, Any]]] = {}
        self._model_cache: dict[tuple[str, str], SharedOCRPredictionModel | None] = {}
        self.snapshot = self._empty_snapshot()
        for run, spec in specs.items():
            self.refresh_run_from_disk(run, spec)

    @staticmethod
    def _empty_snapshot() -> dict[str, Any]:
        return {
            "id": hashlib.sha256(b"no-shared-ocr-training").hexdigest()[:12],
            "reading_count": 0,
            "cell_count": 0,
            "run_count": 0,
            "symbol_counts": {},
        }

    @property
    def model_id(self) -> str | None:
        with self.lock:
            if self.snapshot["cell_count"] < 25:
                return None
            return (
                f"{SHARED_OCR_MODEL_NAME}@{SHARED_OCR_MODEL_VERSION}"
                f".train-{self.snapshot['id']}"
            )

    @staticmethod
    def _collect_samples(
        run: str,
        spec: RunSpec,
        records: list[dict[str, Any]],
        evidence_cache: dict[str, dict[str, Any]],
        generate_missing: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        if not spec.ocr_configured:
            return [], False
        extractor_id = ocr_extractor_id(spec)
        samples: list[dict[str, Any]] = []
        dirty = False
        for record in records:
            labels = _training_symbols(record)
            if labels is None or not any(symbol is not None for symbol in labels):
                continue
            image_id = str(record.get("image_id") or "")
            if not image_id:
                continue
            cached = evidence_cache.get(image_id)
            evidence = _active_ocr_evidence(cached, extractor_id)
            features = evidence.get("cell_features") if evidence else None
            if not _valid_cell_features(features, spec.ocr_cell_count):
                if not generate_missing:
                    continue
                try:
                    frame = Frame(
                        datetime.fromisoformat(str(record["image_time"])),
                        Path(str(record["image"])),
                        None,
                    )
                    evidence = _frame_evidence(frame, spec)
                except (OSError, KeyError, TypeError, ValueError):
                    continue
                features = evidence["cell_features"]
                evidence_cache[image_id] = {
                    "image_id": image_id,
                    "image_time": record.get("image_time"),
                    "image": record.get("image"),
                    "relative_path": record.get("relative_path"),
                    "evidence": _merge_ocr_evidence(
                        list((cached or {}).get("evidence") or []), [evidence]
                    ),
                }
                dirty = True
            for position, symbol in enumerate(labels):
                if symbol is None:
                    continue
                samples.append(
                    {
                        "key": f"{run}:{image_id}:{position}",
                        "run": run,
                        "image_id": image_id,
                        "position": position,
                        "symbol": symbol,
                        "features": tuple(float(value) for value in features[position]),
                    }
                )
        return samples, dirty

    def _rebuild_snapshot(self) -> None:
        samples = [
            sample
            for run_samples in self.samples_by_run.values()
            for sample in run_samples
        ]
        if not samples:
            self.snapshot = self._empty_snapshot()
            self._model_cache.clear()
            return
        digest = hashlib.sha256()
        for sample in sorted(samples, key=lambda item: item["key"]):
            digest.update(sample["key"].encode("utf-8"))
            digest.update(sample["symbol"].encode("ascii"))
            digest.update(np.asarray(sample["features"], dtype=np.float32).tobytes())
        symbol_counts: dict[str, int] = {}
        for sample in samples:
            symbol = str(sample["symbol"])
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        self.snapshot = {
            "id": digest.hexdigest()[:12],
            "reading_count": len(
                {(sample["run"], sample["image_id"]) for sample in samples}
            ),
            "cell_count": len(samples),
            "run_count": len({sample["run"] for sample in samples}),
            "symbol_counts": symbol_counts,
        }
        self._model_cache.clear()

    def refresh_run_from_disk(self, run: str, spec: RunSpec) -> bool:
        state_path = self.output_root / run / f"{run}_overflow_scale_review.json"
        evidence_path = self.output_root / run / f"{run}_overflow_ocr_evidence.json"
        if not state_path.exists() or not spec.ocr_configured:
            return False
        try:
            records = json.loads(state_path.read_text(encoding="utf-8")).get(
                "records", []
            )
        except (OSError, json.JSONDecodeError):
            return False
        evidence_cache = _load_ocr_evidence_cache(evidence_path, spec)
        samples, dirty = self._collect_samples(
            run, spec, records, evidence_cache, generate_missing=True
        )
        if dirty:
            _write_ocr_evidence_cache(evidence_path, spec, evidence_cache)
        with self.lock:
            previous = self.samples_by_run.get(run, [])
            if previous == samples:
                return False
            self.samples_by_run[run] = samples
            self._rebuild_snapshot()
        return True

    def update_run(
        self,
        run: str,
        spec: RunSpec,
        records: list[dict[str, Any]],
        evidence_cache: dict[str, dict[str, Any]],
        generate_missing: bool = False,
    ) -> bool:
        samples, dirty = self._collect_samples(
            run, spec, records, evidence_cache, generate_missing=generate_missing
        )
        if dirty:
            _write_ocr_evidence_cache(
                self.output_root / run / f"{run}_overflow_ocr_evidence.json",
                spec,
                evidence_cache,
            )
        with self.lock:
            if self.samples_by_run.get(run, []) == samples:
                return False
            self.samples_by_run[run] = samples
            self._rebuild_snapshot()
        return True

    def model_for_run(self, run: str) -> SharedOCRPredictionModel | None:
        with self.lock:
            cache_key = (run, self.snapshot["id"])
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]
            samples = [
                sample
                for run_samples in self.samples_by_run.values()
                for sample in run_samples
            ]
            snapshot = dict(self.snapshot)
        if len(samples) < 25 or len({sample["symbol"] for sample in samples}) < 2:
            model = None
        else:
            features = np.asarray(
                [sample["features"] for sample in samples], dtype=float
            )
            labels = np.asarray([sample["symbol"] for sample in samples])
            sample_weight = np.asarray(
                [3.0 if sample["run"] == run else 1.0 for sample in samples],
                dtype=float,
            )
            classifier = ExtraTreesClassifier(
                n_estimators=192,
                min_samples_leaf=1,
                class_weight="balanced",
                max_features=0.7,
                random_state=20260810,
                n_jobs=OCR_TRAINING_JOBS,
            )
            classifier.fit(features, labels, sample_weight=sample_weight)
            model = SharedOCRPredictionModel(
                model_id=(
                    f"{SHARED_OCR_MODEL_NAME}@{SHARED_OCR_MODEL_VERSION}"
                    f".train-{snapshot['id']}"
                ),
                training_readings=int(snapshot["reading_count"]),
                training_cells=int(snapshot["cell_count"]),
                training_runs=int(snapshot["run_count"]),
                classifier=classifier,
            )
        with self.lock:
            self._model_cache[cache_key] = model
        return model


def _blend_symbol_costs(
    physical_costs: list[dict[str, float]],
    learned_costs: list[dict[str, float]] | None,
) -> list[dict[str, float]]:
    def finite_cost(value: Any, default: float = 10.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return float(np.nan_to_num(number, nan=default, posinf=default, neginf=0.0))

    blended: list[dict[str, float]] = []
    for index, physical in enumerate(physical_costs):
        physical_values = {
            symbol: finite_cost(physical.get(symbol, 10.0)) for symbol in OCR_SYMBOLS
        }
        if learned_costs is None:
            blended.append(physical_values)
            continue
        learned = learned_costs[index]
        physical_minimum = min(physical_values.values())
        blended.append(
            {
                symbol: (
                    0.15 * (physical_values[symbol] - physical_minimum)
                    + 0.85 * finite_cost(learned.get(symbol, 10.0))
                )
                for symbol in OCR_SYMBOLS
            }
        )
    return blended


def candidate_suggestion(
    costs: list[dict[str, float]],
    expected_g: float,
    span_g: float,
    sigma_g: float,
    reading_pattern: str | None = None,
    learned_costs: list[dict[str, float]] | None = None,
    minimum_g: float = 0.0,
    maximum_g: float = 9999.9,
    prior_weight: float = 0.05,
    image_gate: float | None = None,
) -> dict[str, float | str | bool | None]:
    low = max(0, int(math.floor((expected_g - span_g) * 10)))
    high = min(99999, int(math.ceil((expected_g + span_g) * 10)))
    prior_weight = max(0.0, float(prior_weight))
    if image_gate is None:
        image_gate = 0.75 if learned_costs is not None else 0.25
    image_gate = max(0.0, float(image_gate))
    costs = _blend_symbol_costs(costs, learned_costs)
    parsed_pattern = parse_scale_reading(reading_pattern) if reading_pattern else None
    pattern_regex = _reading_pattern_regex(parsed_pattern) if parsed_pattern else None

    def rank_range(start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value_deci_g = np.arange(start, stop + 1, dtype=np.int32)
        value_g = value_deci_g.astype(float) / 10.0
        if pattern_regex is not None:
            matches = np.fromiter(
                (
                    pattern_regex.fullmatch(f"{value:.1f}") is not None
                    for value in value_g
                ),
                dtype=bool,
                count=value_g.size,
            )
            value_deci_g = value_deci_g[matches]
            value_g = value_g[matches]
        if value_g.size == 0:
            return value_g, value_g.copy(), value_g.copy()

        grams = value_deci_g // 10
        symbols = np.full((value_deci_g.size, 5), 10, dtype=np.int8)
        symbols[:, 4] = value_deci_g % 10
        symbols[:, 3] = grams % 10
        symbols[grams >= 10, 2] = (grams[grams >= 10] // 10) % 10
        symbols[grams >= 100, 1] = (grams[grams >= 100] // 100) % 10
        symbols[grams >= 1000, 0] = (grams[grams >= 1000] // 1000) % 10
        cost_lookup = np.asarray(
            [
                [cell_costs.get(str(digit), 10.0) for digit in range(10)]
                + [cell_costs.get(" ", 10.0)]
                for cell_costs in costs
            ],
            dtype=float,
        )
        cost_lookup = np.nan_to_num(cost_lookup, nan=10.0, posinf=10.0, neginf=0.0)
        image_cost = np.zeros(value_g.size, dtype=float)
        for position in range(5):
            image_cost += cost_lookup[position, symbols[:, position]]
        prior_cost = prior_weight * ((value_g - expected_g) / max(sigma_g, 0.1)) ** 2
        return value_g, image_cost + prior_cost, image_cost

    values, total_costs, image_costs = rank_range(0, 99999)
    pattern_expanded_search = bool(
        parsed_pattern is not None
        and not np.any((values >= low / 10.0) & (values <= high / 10.0))
    )
    if values.size == 0:
        raise ValueError(
            "The partial reading pattern does not match a valid scale value"
        )
    image_only_index = int(np.lexsort((values, image_costs))[0])
    physical_minimum = min(9999.9, max(0.0, float(minimum_g)))
    physical_maximum = min(9999.9, max(physical_minimum, float(maximum_g)))
    admissible = (values >= physical_minimum) & (values <= physical_maximum)
    if not np.any(admissible):
        if parsed_pattern is None:
            raise ValueError("No OCR candidate is inside the physical weight bounds")
        admissible = np.ones(values.size, dtype=bool)
        physical_minimum, physical_maximum = 0.0, 9999.9
    admissible_indices = np.flatnonzero(admissible)
    bounded_order = np.lexsort(
        (values[admissible_indices], image_costs[admissible_indices])
    )
    bounded_image_index = int(admissible_indices[bounded_order[0]])
    image_floor = float(image_costs[bounded_image_index])
    gated_total_costs = np.where(
        admissible & (image_costs <= image_floor + image_gate),
        total_costs,
        np.inf,
    )
    order = np.lexsort((values, image_costs, gated_total_costs))
    best_index = int(order[0])
    alternatives = order[np.abs(values[order] - values[best_index]) >= 0.5]
    alternative_index = int(
        alternatives[0] if alternatives.size else order[min(1, order.size - 1)]
    )
    margin = max(
        0.0,
        float(gated_total_costs[alternative_index] - gated_total_costs[best_index]),
    )
    if not math.isfinite(margin):
        # No competing value passed the optical gate. The gate width is a
        # conservative finite lower bound and keeps API/state JSON standards-compliant.
        margin = image_gate
    fit = float(image_costs[best_index]) / 5.0
    confidence_score = math.exp(-max(0.0, fit - 0.15)) * (1.0 - math.exp(-2.5 * margin))
    if fit < 0.65 and margin >= 0.16:
        confidence = "high"
    elif fit < 1.0 and margin >= 0.06:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "value_g": float(values[best_index]),
        "image_only_value_g": float(values[image_only_index]),
        "bounded_image_value_g": float(values[bounded_image_index]),
        "prior_changed_choice": bool(best_index != bounded_image_index),
        "physical_bounds_changed_choice": bool(bounded_image_index != image_only_index),
        "physical_minimum_g": physical_minimum,
        "physical_maximum_g": physical_maximum,
        "prior_weight": prior_weight,
        "image_gate": image_gate,
        "cost": float(total_costs[best_index]),
        "image_cost": float(image_costs[best_index]),
        "fit": fit,
        "margin": margin,
        "confidence_score": confidence_score,
        "confidence": confidence,
        "reading_pattern": parsed_pattern.canonical if parsed_pattern else None,
        "pattern_applied": parsed_pattern is not None,
        "pattern_expanded_search": pattern_expanded_search,
    }


class ArtifactBridge:
    """Use the bundled artifact-tool runtime for safe Excel import/export."""

    def __init__(self) -> None:
        dependency_root = (
            Path.home()
            / ".cache"
            / "codex-runtimes"
            / "codex-primary-runtime"
            / "dependencies"
        )
        self.node = Path(
            os.environ.get(
                "FFAC_ARTIFACT_NODE", dependency_root / "node" / "bin" / "node.exe"
            )
        )
        self.modules = dependency_root / "node" / "node_modules"
        local_app = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        self.runtime = local_app / "ff_ac" / "artifact_runtime" / "overflow_workbook"
        self.helper = self.runtime / ARTIFACT_BUILDER.name
        self._prepare()

    def _prepare(self) -> None:
        if not self.node.exists():
            raise RuntimeError(f"Bundled Node.js runtime not found: {self.node}")
        if not self.modules.exists():
            raise RuntimeError(
                f"Bundled artifact-tool modules not found: {self.modules}"
            )
        self.runtime.mkdir(parents=True, exist_ok=True)
        junction = self.runtime / "node_modules"
        if not junction.exists():
            result = subprocess.run(
                [
                    "cmd.exe",
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(junction),
                    str(self.modules),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0 and not junction.exists():
                raise RuntimeError(
                    f"Could not create artifact-tool junction: {result.stderr or result.stdout}"
                )
        shutil.copy2(ARTIFACT_BUILDER, self.helper)

    def _run(self, arguments: list[str], timeout: int = 300) -> str:
        with ARTIFACT_BUILD_SLOTS:
            result = subprocess.run(
                [str(self.node), str(self.helper), *arguments],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "artifact-tool failed").strip()
            )
        return result.stdout

    def extract(self, workbook: Path) -> dict[str, Any]:
        temp_path = self.runtime / f"extract_{os.getpid()}_{threading.get_ident()}.json"
        try:
            self._run(["extract", str(workbook), str(temp_path)])
            return json.loads(temp_path.read_text(encoding="utf-8"))
        finally:
            temp_path.unlink(missing_ok=True)

    def build(
        self, state_path: Path, workbook_path: Path, preview_path: Path | None = None
    ) -> None:
        temporary = workbook_path.with_name(
            f".{workbook_path.stem}.{os.getpid()}.building.xlsx"
        )
        inspect_sidecar = Path(str(temporary) + ".inspect.ndjson")
        temporary_preview = None
        if preview_path is not None:
            temporary_preview = preview_path.with_name(
                f".{preview_path.stem}.building.png"
            )
        temporary.unlink(missing_ok=True)
        if temporary_preview is not None:
            temporary_preview.unlink(missing_ok=True)
        arguments = ["build", str(state_path), str(temporary)]
        if temporary_preview is not None:
            arguments.append(str(temporary_preview))
        try:
            self._run(arguments)
            workbook_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, workbook_path)
            if temporary_preview is not None and temporary_preview.exists():
                os.replace(temporary_preview, preview_path)
        finally:
            temporary.unlink(missing_ok=True)
            inspect_sidecar.unlink(missing_ok=True)
            if temporary_preview is not None:
                temporary_preview.unlink(missing_ok=True)


class WorkbookSync:
    def __init__(
        self,
        bridge: ArtifactBridge,
        state_path: Path,
        workbook_path: Path,
        preview_path: Path,
    ):
        self.bridge = bridge
        self.state_path = state_path
        self.workbook_path = workbook_path
        self.preview_path = preview_path
        self.lock = threading.Lock()
        self.requested_revision = 0
        self.completed_revision = 0
        self.requested_preview_revision = 0
        self.completed_preview_revision = 0
        self.last_request_at = 0.0
        self.debounce_seconds = 3.0
        self.state = "not built"
        self.error: str | None = None
        self.worker: threading.Thread | None = None

    def _build(self, include_preview: bool = False) -> None:
        with self.lock:
            self.state = "saving"
            self.error = None
        try:
            self.bridge.build(
                self.state_path,
                self.workbook_path,
                self.preview_path if include_preview else None,
            )
        except Exception as exc:
            with self.lock:
                self.state = "error"
                self.error = str(exc)
            raise
        else:
            with self.lock:
                self.state = "current"

    def sync_now(self, include_preview: bool = False) -> None:
        self._build(include_preview=include_preview)
        with self.lock:
            self.completed_revision = self.requested_revision
            if include_preview:
                self.completed_preview_revision = self.requested_revision

    def request(self, include_preview: bool = False) -> None:
        with self.lock:
            self.requested_revision += 1
            self.last_request_at = time.monotonic()
            if include_preview:
                self.requested_preview_revision = self.requested_revision
            if self.worker is not None and self.worker.is_alive():
                return
            self.worker = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker.start()

    def _worker_loop(self) -> None:
        while True:
            while True:
                with self.lock:
                    quiet_for = time.monotonic() - self.last_request_at
                remaining = self.debounce_seconds - quiet_for
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.5))
            with self.lock:
                target = self.requested_revision
                include_preview = (
                    self.requested_preview_revision > self.completed_preview_revision
                )
            try:
                self._build(include_preview=include_preview)
            except Exception:
                return
            with self.lock:
                self.completed_revision = target
                if include_preview:
                    self.completed_preview_revision = target
                if self.completed_revision >= self.requested_revision:
                    return

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "error": self.error,
                "path": str(self.workbook_path),
                "pending": self.requested_revision > self.completed_revision,
            }


def _normalise_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _normalise_path(value: Any) -> str:
    return os.path.normcase(os.path.normpath(str(value or "").strip()))


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().replace(" ", "").replace(",", ".")
        if not text:
            return None
        try:
            result = float(text)
        except ValueError:
            return None
    return result if math.isfinite(result) else None


def _datetime_from_cell(
    value: Any, base_date: date, previous: datetime | None
) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric >= 20_000:
            return datetime(1899, 12, 30) + timedelta(days=numeric)
        if 0 <= numeric < 3:
            day_offset = int(math.floor(numeric))
            seconds = round((numeric - day_offset) * 86_400)
            candidate = datetime.combine(base_date, datetime_time()) + timedelta(
                days=day_offset, seconds=seconds
            )
            if previous is not None:
                while candidate < previous - timedelta(hours=12):
                    candidate += timedelta(days=1)
            return candidate
        return None
    text = str(value).strip()
    if not text:
        return None
    clean = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(clean)
        return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        if pattern.startswith("%H"):
            candidate = datetime.combine(base_date, parsed.time())
            if previous is not None and candidate < previous - timedelta(hours=12):
                candidate += timedelta(days=1)
            return candidate
        return parsed
    return None


def _column(headers: list[str], aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        if alias in headers:
            return headers.index(alias)
    return None


def _cell(row: list[Any], index: int | None) -> Any:
    if index is None or index >= len(row):
        return None
    return row[index]


def _status_from_cells(status_value: Any, validated_value: Any) -> str | None:
    text = str(status_value or "").strip().lower()
    mapping = {
        "verified": "Verified",
        "validated": "Verified",
        "validert": "Verified",
        "confirmed": "Verified",
        "bekreftet": "Verified",
        "corrected": "Corrected",
        "korrigert": "Corrected",
        "partial": "Partial",
        "delvis": "Partial",
        "unreadable": "Unreadable",
        "uleselig": "Unreadable",
    }
    if text in mapping:
        return mapping[text]
    if validated_value is True or str(validated_value or "").strip().lower() in {
        "yes",
        "true",
        "1",
        "ja",
        "x",
    }:
        return "Verified"
    return None


def parse_imported_workbook(
    payload: dict[str, Any], workbook_path: Path, injection_start: datetime
) -> list[dict[str, Any]]:
    """Extract existing readings without modifying or trusting unrelated sheets."""
    imported: list[dict[str, Any]] = []
    for sheet in payload.get("sheets", []):
        if str(sheet.get("name", "")).strip().lower() in {
            "summary",
            "review rounds",
            "imported data",
            "predictions",
        }:
            continue
        rows = sheet.get("values") or []
        header_row = None
        header_index = None
        for index, row in enumerate(rows[:40]):
            normalised = [_normalise_header(value) for value in row]
            weight = _column(
                normalised,
                (
                    "finalweightg",
                    "confirmedcorrectedweightg",
                    "wread",
                    "weightg",
                    "weight",
                    "vektg",
                    "vekt",
                ),
            )
            status = _column(normalised, ("reviewstatus", "status", "validationstatus"))
            if weight is not None or (status is not None and "imageid" in normalised):
                header_row = normalised
                header_index = index
                break
        if header_row is None or header_index is None:
            continue
        weight_col = _column(
            header_row,
            (
                "finalweightg",
                "confirmedcorrectedweightg",
                "wread",
                "weightg",
                "weight",
                "vektg",
                "vekt",
            ),
        )
        existing_col = _column(header_row, ("existingvalueg",))
        reading_pattern_col = _column(
            header_row,
            (
                "readingpattern",
                "partialreading",
                "avlesningsmonster",
                "avlesningsmnster",
            ),
        )
        existing_source_col = _column(header_row, ("existingsource",))
        time_col = _column(
            header_row,
            (
                "imagetimestamp",
                "imagetime",
                "timestamp",
                "datetime",
                "tread",
                "time",
                "tid",
            ),
        )
        image_col = _column(
            header_row, ("imagepath", "sourceimage", "image", "path", "filbane")
        )
        image_id_col = _column(header_row, ("imageid",))
        status_col = _column(header_row, ("reviewstatus", "status", "validationstatus"))
        validated_col = _column(
            header_row, ("validated", "verified", "confirmed", "validert")
        )
        reviewed_col = _column(header_row, ("reviewedat", "verifiedat"))
        reviewer_col = _column(header_row, ("reviewer", "validatedby"))
        round_col = _column(header_row, ("reviewround",))
        previous_time: datetime | None = None
        for row_number, raw_row in enumerate(
            rows[header_index + 1 :], start=header_index + 2
        ):
            row = list(raw_row)
            weight = _float_or_none(_cell(row, weight_col))
            reading_pattern = ""
            raw_pattern = _cell(row, reading_pattern_col)
            if raw_pattern is not None and str(raw_pattern).strip():
                try:
                    parsed_pattern = parse_scale_reading(raw_pattern)
                except ValueError:
                    parsed_pattern = None
                if parsed_pattern is not None:
                    reading_pattern = parsed_pattern.canonical
                    if weight is None and not parsed_pattern.is_partial:
                        weight = parsed_pattern.weight_g
            used_existing_value = False
            if weight is None:
                weight = _float_or_none(_cell(row, existing_col))
                used_existing_value = weight is not None
            status = _status_from_cells(
                _cell(row, status_col), _cell(row, validated_col)
            )
            if status in {"Verified", "Corrected"} and weight is None:
                status = None
            if status == "Partial" and (
                not reading_pattern or "x" not in reading_pattern
            ):
                status = None
            if weight is None and status is None and not reading_pattern:
                continue
            source_time = _datetime_from_cell(
                _cell(row, time_col), injection_start.date(), previous_time
            )
            if source_time is not None:
                previous_time = source_time
            reviewed_at = _datetime_from_cell(
                _cell(row, reviewed_col), injection_start.date(), None
            )
            imported.append(
                {
                    "source_workbook": str(workbook_path),
                    "source_sheet": str(sheet.get("name", "")),
                    "source_row": row_number,
                    "source_time": source_time.isoformat() if source_time else None,
                    "source_image": str(_cell(row, image_col) or ""),
                    "source_image_id": str(_cell(row, image_id_col) or ""),
                    "weight_g": weight,
                    "reading_pattern": reading_pattern,
                    "status": status,
                    "reviewed_at": reviewed_at.isoformat() if reviewed_at else None,
                    "reviewer": str(_cell(row, reviewer_col) or ""),
                    "review_round_id": str(_cell(row, round_col) or ""),
                    "source_reference": (
                        str(_cell(row, existing_source_col) or "")
                        if used_existing_value
                        else ""
                    ),
                    "note": (
                        "Imported validated reading"
                        if status
                        else "Imported existing reading"
                    ),
                }
            )
    return imported


def _frame_id(run: str, frame: Frame, root: Path) -> str:
    try:
        relative = str(frame.path.relative_to(root))
    except ValueError:
        relative = frame.path.name
    digest = hashlib.sha1(
        f"{run}|{frame.time.isoformat()}|{relative}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{run}_{frame.time:%Y%m%d_%H%M%S}_{digest}"


def retain_reviewed_frames(
    run: str,
    root: Path,
    eligible_frames: list[Frame],
    saved_records: list[dict[str, Any]],
) -> tuple[list[Frame], frozenset[str]]:
    """Keep reviewed frames even when a later review window is shorter."""
    eligible_ids = frozenset(_frame_id(run, frame, root) for frame in eligible_frames)
    frames_by_id = {_frame_id(run, frame, root): frame for frame in eligible_frames}
    for record in saved_records:
        if str(record.get("status", "")) not in REVIEWED_STATUSES:
            continue
        try:
            frame = Frame(
                datetime.fromisoformat(str(record["image_time"])),
                Path(str(record["image"])),
                (
                    int(record["file_size"])
                    if record.get("file_size") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
        image_id = str(record.get("image_id") or _frame_id(run, frame, root))
        frames_by_id.setdefault(image_id, frame)
    retained = sorted(
        frames_by_id.values(), key=lambda frame: (frame.time, str(frame.path).lower())
    )
    return retained, eligible_ids


class RunSession:
    def __init__(
        self,
        spec: RunSpec,
        output_root: Path,
        bridge: ArtifactBridge,
        sample_count: int,
        max_hours: float,
        stop_after_feed: bool,
        feed_end_buffer_h: float,
        existing_workbook: Path | None,
        reviewer: str,
        campaign_id: str | None = None,
        inventory: RunInventory | None = None,
        progress: Callable[[float, str], None] | None = None,
        shared_ocr_trainer: SharedOCRTrainer | None = None,
        weight_data_status: str = "ready",
        weight_data_reason: str = "",
        weight_data_reviewer: str = "",
        weight_data_updated_at: str | None = None,
        resume_existing_round: bool = False,
    ):
        self.spec = spec
        self.shared_ocr_trainer = shared_ocr_trainer
        self.shared_ocr_model: SharedOCRPredictionModel | None = None
        self.prediction_model_id = ""
        self.training_snapshot: dict[str, Any] = prediction_training_snapshot([])
        self.ocr_extractor_id = ""
        self.auto_geometry_diagnostics: dict[str, Any] | None = None
        self.adaptive_refresh_count = 0
        if inventory is None:
            self.injection_start, self.injection_end = resolved_injection_window(spec)
        else:
            self.injection_start = inventory.injection_start
            self.injection_end = inventory.injection_end
        self.output_dir = output_root / spec.run
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / f"{spec.run}_overflow_scale_review.json"
        self.csv_path = self.output_dir / f"{spec.run}_overflow_scale_review.csv"
        self.workbook_path = self.output_dir / f"{spec.run}_overflow_scale_review.xlsx"
        self.preview_path = (
            self.output_dir / f"{spec.run}_overflow_scale_review_summary.png"
        )
        self.prediction_cache_path = (
            self.output_dir / f"{spec.run}_overflow_predictions.json"
        )
        self.prediction_cache = _load_prediction_cache(self.prediction_cache_path, spec)
        self.prediction_cache_dirty = False
        self.ocr_evidence_cache_path = (
            self.output_dir / f"{spec.run}_overflow_ocr_evidence.json"
        )
        self.ocr_evidence_cache = _load_ocr_evidence_cache(
            self.ocr_evidence_cache_path, spec
        )
        self.ocr_evidence_cache_dirty = False
        self.existing_workbook = existing_workbook
        self.reviewer = reviewer
        self.campaign_id = campaign_id
        self.sample_count = int(sample_count)
        self.max_hours = float(max_hours)
        self.stop_after_feed = bool(stop_after_feed)
        self.feed_end_buffer_h = float(feed_end_buffer_h)
        self.weight_data_status = str(weight_data_status or "needs-review")
        self.weight_data_reason = str(weight_data_reason or "")
        self.weight_data_reviewer = str(weight_data_reviewer or "")
        self.weight_data_updated_at = weight_data_updated_at
        self.lock = threading.RLock()
        self.review_rounds: list[dict[str, Any]] = []
        self.current_round_id = ""
        self.saved_round_id = ""
        self.resume_existing_round = bool(resume_existing_round)
        self.resumed_existing_round = False
        self.imported_records: list[dict[str, Any]] = []
        self._progress = progress or (lambda _value, _message: None)

        self._progress(0.03, "Finding camera images")
        if inventory is None:
            all_frames = discover_spec_frames(spec)
            if not all_frames:
                raise RuntimeError(
                    f"No timestamped scale images found under {spec.image_root}"
                )
            start = min(
                spec.gravity_start, self.injection_start - timedelta(minutes=30)
            )
            cutoffs = [self.injection_start + timedelta(hours=self.max_hours)]
            cutoff_rules = [f"injection start + {self.max_hours:g} h"]
            if self.stop_after_feed:
                cutoffs.append(
                    spec.gravity_end + timedelta(hours=self.feed_end_buffer_h)
                )
                cutoff_rules.append(f"gravity-feed end + {self.feed_end_buffer_h:g} h")
            self.effective_cutoff = min(cutoffs)
            self.cutoff_rule = "earliest of " + " and ".join(cutoff_rules)
            eligible_frames = [
                frame
                for frame in all_frames
                if start <= frame.time <= self.effective_cutoff
            ]
            saved_records: list[dict[str, Any]] = []
            if self.state_path.exists():
                try:
                    saved_records = list(
                        json.loads(self.state_path.read_text(encoding="utf-8")).get(
                            "records", []
                        )
                    )
                except (OSError, json.JSONDecodeError):
                    saved_records = []
            self.frames, self.review_eligible_image_ids = retain_reviewed_frames(
                self.spec.run,
                self.spec.image_root,
                eligible_frames,
                saved_records,
            )
        else:
            self.effective_cutoff = inventory.effective_cutoff
            self.cutoff_rule = inventory.cutoff_rule
            self.frames = list(inventory.frames)
            self.review_eligible_image_ids = (
                inventory.eligible_image_ids
                if inventory.eligible_image_ids is not None
                else frozenset(
                    _frame_id(self.spec.run, frame, self.spec.image_root)
                    for frame in self.frames
                )
            )
        if not self.frames:
            raise RuntimeError(
                "No camera images fall inside the selected overflow-review window"
            )
        if self.spec.automatic_ocr:
            cached_geometry = _load_auto_geometry(
                _auto_geometry_cache_path(self.output_dir, self.spec.run),
                self.spec,
            )
            if cached_geometry is not None:
                self.auto_geometry_diagnostics = cached_geometry[3]
        elif not self.spec.ocr_configured:
            self._progress(0.06, "Locating the scale display and digit cells")
            self.spec, self.auto_geometry_diagnostics = resolve_automatic_ocr(
                self.spec, self.frames, self.output_dir
            )
        self.ocr_extractor_id = ocr_extractor_id(self.spec)
        self.frame_times = [frame.time for frame in self.frames]
        self.records = self._build_records()
        for record in self.records:
            cached = self.prediction_cache.get(record["image_id"])
            if cached is not None:
                record["predictions"] = copy.deepcopy(cached.get("predictions") or [])
        self._load_previous_state()

        if existing_workbook is not None:
            if not existing_workbook.exists():
                raise FileNotFoundError(
                    f"Existing workbook not found: {existing_workbook}"
                )
            same_as_generated = _normalise_path(existing_workbook) == _normalise_path(
                self.workbook_path
            )
            if not (same_as_generated and self.state_path.exists()):
                self._progress(0.08, f"Reading {existing_workbook.name}")
                imported_payload = bridge.extract(existing_workbook)
                imported = parse_imported_workbook(
                    imported_payload, existing_workbook, self.injection_start
                )
                self._apply_imported(imported)

        self.training_snapshot = prediction_training_snapshot(self.records)
        if self.shared_ocr_trainer is not None:
            self.shared_ocr_trainer.update_run(
                self.spec.run,
                self.spec,
                self.records,
                self.ocr_evidence_cache,
                generate_missing=True,
            )
            self.shared_ocr_model = self.shared_ocr_trainer.model_for_run(self.spec.run)
        self.prediction_model_id = prediction_model_id(
            self.spec,
            self.training_snapshot["id"],
            self.shared_ocr_model.model_id if self.shared_ocr_model else None,
        )
        for record in self.records:
            _apply_prediction_aliases(
                record, _active_prediction(record, self.prediction_model_id)
            )
        self.resumed_existing_round = self._resume_round()
        if not self.resumed_existing_round:
            self._start_round()
        partial_indices = [
            int(record["index"])
            for record in self.records
            if record["status"] == "Partial" and record.get("reading_pattern")
        ]
        prediction_indices = sorted(
            set(self.reviewable_pending_indices()).union(partial_indices)
        )
        self._build_suggestions(prediction_indices)
        self.sync = WorkbookSync(
            bridge, self.state_path, self.workbook_path, self.preview_path
        )
        self._persist(request_excel=False, write_prediction_cache=True)
        self._progress(0.96, "Queueing review workbook")
        self.sync.request(include_preview=True)
        self._progress(1.0, "Ready")

    def _build_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, frame in enumerate(self.frames):
            image_id = _frame_id(self.spec.run, frame, self.spec.image_root)
            try:
                relative = str(frame.path.relative_to(self.spec.image_root))
            except ValueError:
                relative = frame.path.name
            records.append(
                {
                    "index": index,
                    "image_id": image_id,
                    "image_time": frame.time.isoformat(),
                    "time_from_injection_h": (
                        frame.time - self.injection_start
                    ).total_seconds()
                    / 3600.0,
                    "image": str(frame.path),
                    "image_filename": frame.path.name,
                    "relative_path": relative,
                    "file_size": frame.file_size,
                    "eligible_for_review": image_id in self.review_eligible_image_ids,
                    "predictions": [],
                    "suggested_g": None,
                    "expected_g": None,
                    "confidence": None,
                    "confidence_score": None,
                    "prediction_model_id": "",
                    "prediction_generated_at": None,
                    "prediction_training_snapshot_id": "",
                    "prediction_training_count": None,
                    "prediction_partial_count": None,
                    "prediction_reading_pattern": None,
                    "prediction_prior_source": None,
                    "prediction_prior_weight": None,
                    "prediction_image_gate": None,
                    "prediction_image_only_g": None,
                    "prediction_bounded_image_g": None,
                    "prediction_prior_changed_choice": None,
                    "prediction_physical_bounds_changed_choice": None,
                    "prediction_physical_min_g": None,
                    "prediction_physical_max_g": None,
                    "prediction_physical_bounds_source": None,
                    "prediction_trend_knee_time": None,
                    "prediction_trend_active_rate_g_per_h": None,
                    "prediction_trend_post_rate_g_per_h": None,
                    "prediction_trend_residual_scale_g": None,
                    "prediction_trend_loss_ratio": None,
                    "shared_ocr_model_id": "",
                    "shared_ocr_training_count": None,
                    "shared_ocr_training_cells": None,
                    "shared_ocr_training_runs": None,
                    "ocr_extractor_id": "",
                    "prediction_error_g": None,
                    "existing_value_g": None,
                    "existing_source": "",
                    "selected_for_review": False,
                    "review_round_id": "",
                    "review_position": None,
                    "review_selection_phase": "",
                    "weight_g": None,
                    "reading_pattern": "",
                    "status": "Not selected",
                    "verified_at": None,
                    "reviewer": "",
                    "notes": "",
                }
            )
        return records

    def _load_previous_state(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        _repair_nonfinite_json(self.state_path, payload)
        metadata = dict(payload.get("metadata") or {})
        previous_generated_at = str(metadata.get("updated_at") or "")
        self.saved_round_id = str(metadata.get("current_round_id") or "")
        self.adaptive_refresh_count = int(metadata.get("adaptive_refresh_count") or 0)
        self.review_rounds = list(payload.get("review_rounds", []))
        self.imported_records = list(payload.get("imported_records", []))
        saved_by_id = {
            str(record.get("image_id")): record
            for record in payload.get("records", [])
            if record.get("image_id")
        }
        saved_by_path = {
            _normalise_path(record.get("image")): record
            for record in payload.get("records", [])
            if record.get("image")
        }
        merge_fields = {
            "suggested_g",
            "expected_g",
            "confidence",
            "confidence_score",
            "existing_value_g",
            "existing_source",
            "selected_for_review",
            "review_round_id",
            "review_position",
            "review_selection_phase",
            "weight_g",
            "reading_pattern",
            "status",
            "verified_at",
            "reviewer",
            "notes",
            "prediction_error_g",
            "prediction_prior_source",
            "prediction_prior_weight",
            "prediction_image_gate",
            "prediction_image_only_g",
            "prediction_bounded_image_g",
            "prediction_prior_changed_choice",
            "prediction_physical_bounds_changed_choice",
            "prediction_physical_min_g",
            "prediction_physical_max_g",
            "prediction_physical_bounds_source",
            "prediction_trend_knee_time",
            "prediction_trend_active_rate_g_per_h",
            "prediction_trend_post_rate_g_per_h",
            "prediction_trend_residual_scale_g",
            "prediction_trend_loss_ratio",
            "shared_ocr_model_id",
            "shared_ocr_training_count",
            "shared_ocr_training_cells",
            "shared_ocr_training_runs",
        }
        for record in self.records:
            saved = saved_by_id.get(record["image_id"]) or saved_by_path.get(
                _normalise_path(record["image"])
            )
            if saved is None:
                continue
            for field in merge_fields:
                if field in saved:
                    record[field] = copy.deepcopy(saved[field])
            cached_predictions = list(record.get("predictions") or [])
            record["predictions"] = _merge_predictions(
                cached_predictions, list(saved.get("predictions") or [])
            )
            if not record.get("predictions"):
                migrated = _legacy_prediction(
                    saved,
                    _legacy_prediction_model_id(self.spec),
                    previous_generated_at or None,
                )
                if migrated is not None:
                    record["predictions"] = [migrated]
            if record["predictions"] != cached_predictions:
                self.prediction_cache_dirty = True
            saved_current = (
                self.resume_existing_round
                and bool(self.saved_round_id)
                and bool(saved.get("selected_for_review"))
                and str(saved.get("review_round_id") or "") == self.saved_round_id
            )
            if not saved_current:
                record["selected_for_review"] = False
                record["review_position"] = None
                record["review_selection_phase"] = ""
                if record["status"] == "Pending":
                    record["status"] = "Not selected"

    def _nearest_frame_index(self, timestamp: datetime) -> tuple[int, float]:
        position = bisect.bisect_left(self.frame_times, timestamp)
        candidates = [
            index for index in (position - 1, position) if 0 <= index < len(self.frames)
        ]
        selected = min(
            candidates,
            key=lambda index: abs(
                (self.frame_times[index] - timestamp).total_seconds()
            ),
        )
        return selected, abs((self.frame_times[selected] - timestamp).total_seconds())

    def _apply_imported(self, imported: list[dict[str, Any]]) -> None:
        by_id = {record["image_id"]: record for record in self.records}
        by_path = {_normalise_path(record["image"]): record for record in self.records}
        cadence = (
            np.median(np.diff([value.timestamp() for value in self.frame_times]))
            if len(self.frame_times) > 1
            else 60
        )
        max_delta = max(180.0, float(cadence) * 2.5)
        best_existing_delta: dict[str, float] = {}
        known_sources = {
            (
                _normalise_path(item.get("source_workbook")),
                str(item.get("source_sheet", "")),
                int(item.get("source_row") or 0),
            )
            for item in self.imported_records
        }
        for item in imported:
            record = by_id.get(str(item.get("source_image_id") or ""))
            delta = 0.0
            if record is None and item.get("source_image"):
                record = by_path.get(_normalise_path(item["source_image"]))
            if record is None and item.get("source_time"):
                timestamp = datetime.fromisoformat(str(item["source_time"]))
                frame_index, delta = self._nearest_frame_index(timestamp)
                if delta <= max_delta:
                    record = self.records[frame_index]
            copied = dict(item)
            if record is None:
                copied.update(
                    {
                        "matched_image_id": "",
                        "match_delta_s": None,
                        "note": item["note"] + "; no matching image",
                    }
                )
                source_key = (
                    _normalise_path(item.get("source_workbook")),
                    str(item.get("source_sheet", "")),
                    int(item.get("source_row") or 0),
                )
                if source_key not in known_sources:
                    self.imported_records.append(copied)
                    known_sources.add(source_key)
                continue
            copied.update(
                {"matched_image_id": record["image_id"], "match_delta_s": delta}
            )
            source_key = (
                _normalise_path(item.get("source_workbook")),
                str(item.get("source_sheet", "")),
                int(item.get("source_row") or 0),
            )
            if source_key not in known_sources:
                self.imported_records.append(copied)
                known_sources.add(source_key)
            weight = item.get("weight_g")
            previous_delta = best_existing_delta.get(record["image_id"], float("inf"))
            if weight is not None and delta <= previous_delta:
                record["existing_value_g"] = float(weight)
                record["existing_source"] = (
                    item.get("source_reference")
                    or f"{Path(item['source_workbook']).name} | {item['source_sheet']}!row {item['source_row']}"
                )
                best_existing_delta[record["image_id"]] = delta
            status = item.get("status")
            if status in REVIEWED_STATUSES:
                record["status"] = status
                record["weight_g"] = (
                    None if status in {"Unreadable", "Partial"} else weight
                )
                record["reading_pattern"] = (
                    str(item.get("reading_pattern") or "")
                    if status == "Partial"
                    else (f"{float(weight):.1f}" if weight is not None else "")
                )
                record["verified_at"] = item.get("reviewed_at")
                record["reviewer"] = item.get("reviewer") or "Imported"
                record["review_round_id"] = item.get("review_round_id") or "imported"

    def _build_suggestions(self, selected_indices: list[int]) -> None:
        """Decode selected images for the current versioned predictor release."""
        shared_model = getattr(self, "shared_ocr_model", None)
        anchors = [
            (datetime.fromisoformat(item["image_time"]), float(item["weight_g"]))
            for item in self.training_snapshot["anchors"]
        ]
        trend_model = fit_piecewise_weight_trend(anchors)
        total = len(selected_indices)
        generated_count = 0
        reused_count = 0
        evidence_generated_count = 0
        evidence_reused_count = 0
        for position, record_index in enumerate(selected_indices):
            frame = self.frames[record_index]
            record = self.records[record_index]
            expected, span, sigma, anchored, prior_source = prediction_window(
                self.spec, anchors, frame.time, trend_model=trend_model
            )
            display_expected = (
                expected if anchored or self.spec.gravity_schedule_verified else None
            )
            partial_reading = (
                str(record.get("reading_pattern"))
                if record.get("status") == "Partial"
                else None
            )
            if partial_reading:
                physical_minimum, physical_maximum, physical_bounds_source = (
                    0.0,
                    9999.9,
                    "human-partial-reading",
                )
            else:
                (
                    physical_minimum,
                    physical_maximum,
                    physical_bounds_source,
                ) = physical_candidate_bounds(
                    anchors,
                    frame.time,
                    plateau_expected_g=(
                        expected if prior_source in PLATEAU_PRIOR_SOURCES else None
                    ),
                )
            if not self.spec.ocr_configured:
                record.update(
                    {
                        "suggested_g": None,
                        "expected_g": display_expected,
                        "confidence": None,
                        "confidence_score": None,
                        "prediction_model_id": self.prediction_model_id,
                        "prediction_generated_at": None,
                        "prediction_training_snapshot_id": self.training_snapshot["id"],
                        "prediction_training_count": self.training_snapshot["count"],
                        "prediction_partial_count": self.training_snapshot[
                            "partial_count"
                        ],
                        "prediction_reading_pattern": None,
                        "prediction_prior_source": prior_source,
                        "prediction_prior_weight": None,
                        "prediction_image_gate": None,
                        "prediction_image_only_g": None,
                        "prediction_bounded_image_g": None,
                        "prediction_prior_changed_choice": None,
                        "prediction_physical_bounds_changed_choice": None,
                        "prediction_physical_min_g": physical_minimum,
                        "prediction_physical_max_g": physical_maximum,
                        "prediction_physical_bounds_source": physical_bounds_source,
                        "shared_ocr_model_id": "",
                        "shared_ocr_training_count": None,
                        "shared_ocr_training_runs": None,
                        "ocr_extractor_id": self.ocr_extractor_id,
                        "prediction_error_g": None,
                    }
                )
                if position % 10 == 0 or position == total - 1:
                    self._progress(
                        0.18 + 0.72 * (position + 1) / max(1, total),
                        f"Selected manual images {position + 1}/{total}",
                    )
                continue
            prediction = _active_prediction(record, self.prediction_model_id)
            if prediction is None:
                cached_evidence = self.ocr_evidence_cache.get(record["image_id"])
                evidence = _active_ocr_evidence(cached_evidence, self.ocr_extractor_id)
                costs = evidence.get("symbol_costs") if evidence else None
                features = evidence.get("cell_features") if evidence else None
                if not _valid_symbol_costs(
                    costs, self.spec.ocr_cell_count
                ) or not _valid_cell_features(features, self.spec.ocr_cell_count):
                    evidence = _frame_evidence(frame, self.spec)
                    costs = evidence["symbol_costs"]
                    features = evidence["cell_features"]
                    self.ocr_evidence_cache[record["image_id"]] = {
                        "image_id": record["image_id"],
                        "image_time": record["image_time"],
                        "image": record["image"],
                        "relative_path": record["relative_path"],
                        "evidence": _merge_ocr_evidence(
                            list((cached_evidence or {}).get("evidence") or []),
                            [evidence],
                        ),
                    }
                    self.ocr_evidence_cache_dirty = True
                    evidence_generated_count += 1
                else:
                    evidence_reused_count += 1
                learned_costs = (
                    shared_model.symbol_costs(features)
                    if shared_model is not None
                    else None
                )
                prior_weight, image_gate = prediction_prior_profile(
                    prior_source, learned_costs is not None
                )
                suggestion = candidate_suggestion(
                    costs,
                    expected,
                    span,
                    sigma,
                    partial_reading,
                    learned_costs=learned_costs,
                    minimum_g=physical_minimum,
                    maximum_g=physical_maximum,
                    prior_weight=prior_weight,
                    image_gate=image_gate,
                )
                prediction = {
                    "model_id": self.prediction_model_id,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "review_round_id": self.current_round_id,
                    "training_snapshot_id": self.training_snapshot["id"],
                    "training_count": self.training_snapshot["count"],
                    "partial_count": self.training_snapshot["partial_count"],
                    "ocr_extractor_id": self.ocr_extractor_id,
                    "shared_ocr_model_id": (
                        shared_model.model_id if shared_model is not None else ""
                    ),
                    "shared_ocr_training_count": (
                        shared_model.training_readings
                        if shared_model is not None
                        else 0
                    ),
                    "shared_ocr_training_cells": (
                        shared_model.training_cells if shared_model is not None else 0
                    ),
                    "shared_ocr_training_runs": (
                        shared_model.training_runs if shared_model is not None else 0
                    ),
                    "reading_pattern": suggestion["reading_pattern"],
                    "pattern_applied": suggestion["pattern_applied"],
                    "pattern_expanded_search": suggestion["pattern_expanded_search"],
                    "value_g": suggestion["value_g"],
                    "image_only_value_g": suggestion["image_only_value_g"],
                    "bounded_image_value_g": suggestion["bounded_image_value_g"],
                    "prior_changed_choice": suggestion["prior_changed_choice"],
                    "physical_bounds_changed_choice": suggestion[
                        "physical_bounds_changed_choice"
                    ],
                    "physical_minimum_g": suggestion["physical_minimum_g"],
                    "physical_maximum_g": suggestion["physical_maximum_g"],
                    "physical_bounds_source": physical_bounds_source,
                    "trend_knee_time": (
                        trend_model.knee_time.isoformat()
                        if trend_model is not None
                        else None
                    ),
                    "trend_active_rate_g_per_h": (
                        trend_model.active_rate_g_per_h
                        if trend_model is not None
                        else None
                    ),
                    "trend_post_rate_g_per_h": (
                        trend_model.post_rate_g_per_h
                        if trend_model is not None
                        else None
                    ),
                    "trend_residual_scale_g": (
                        trend_model.residual_scale_g
                        if trend_model is not None
                        else None
                    ),
                    "trend_loss_ratio": (
                        trend_model.loss_ratio if trend_model is not None else None
                    ),
                    "expected_g": display_expected,
                    "prior_source": prior_source,
                    "prior_weight": suggestion["prior_weight"],
                    "image_gate": suggestion["image_gate"],
                    "confidence": suggestion["confidence"],
                    "confidence_score": suggestion["confidence_score"],
                    "fit": suggestion["fit"],
                    "margin": suggestion["margin"],
                    "image_cost": suggestion["image_cost"],
                    "total_cost": suggestion["cost"],
                }
                record.setdefault("predictions", []).append(prediction)
                self.prediction_cache_dirty = True
                generated_count += 1
            else:
                reused_count += 1
            _apply_prediction_aliases(record, prediction)
            record["prediction_error_g"] = (
                float(record["weight_g"]) - float(prediction["value_g"])
                if record.get("weight_g") is not None
                else None
            )
            if position % 10 == 0 or position == total - 1:
                self._progress(
                    0.18 + 0.72 * (position + 1) / max(1, total),
                    (
                        f"Selected predictions {position + 1}/{total} "
                        f"({generated_count} new predictions, "
                        f"{evidence_generated_count} images opened)"
                    ),
                )
        self.predictions_generated = generated_count
        self.predictions_reused = reused_count
        self.ocr_evidence_generated = evidence_generated_count
        self.ocr_evidence_reused = evidence_reused_count

    def _start_round(self) -> None:
        bootstrap_count = initial_anchor_count(self.records, self.sample_count)
        for record in self.records:
            record["selected_for_review"] = False
            record["review_position"] = None
            record["review_selection_phase"] = ""
            if record["status"] == "Pending":
                record["status"] = "Not selected"
        selected = select_unreviewed_indices(self.records, self.sample_count)
        if not selected:
            raise RuntimeError("All relevant images have already been reviewed")
        sequence = len(self.review_rounds) + 1
        self.current_round_id = (
            self.campaign_id or f"{datetime.now():%Y%m%dT%H%M%S}_r{sequence:03d}"
        )
        for position, index in enumerate(selected):
            self.records[index].update(
                {
                    "selected_for_review": True,
                    "review_round_id": self.current_round_id,
                    "review_position": position,
                    "review_selection_phase": (
                        "bootstrap-anchor" if position < bootstrap_count else "gap-fill"
                    ),
                    "status": "Pending",
                }
            )
        release = self._predictor_release("round-start")
        self.review_rounds.append(
            {
                "id": self.current_round_id,
                "campaign_id": self.campaign_id or self.current_round_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "requested_count": self.sample_count,
                "selected_count": len(selected),
                "reviewer": self.reviewer,
                "source_workbook": str(self.existing_workbook or ""),
                "max_hours": self.max_hours,
                "stop_after_feed": self.stop_after_feed,
                "feed_end_buffer_h": self.feed_end_buffer_h,
                "prediction_model_id": self.prediction_model_id,
                "prediction_training_snapshot_id": self.training_snapshot["id"],
                "prediction_training_count": self.training_snapshot["count"],
                "prediction_partial_count": self.training_snapshot["partial_count"],
                "ocr_extractor_id": self.ocr_extractor_id,
                "shared_ocr_model_id": (
                    self.shared_ocr_model.model_id if self.shared_ocr_model else ""
                ),
                "shared_ocr_training_count": (
                    self.shared_ocr_model.training_readings
                    if self.shared_ocr_model
                    else 0
                ),
                "shared_ocr_training_runs": (
                    self.shared_ocr_model.training_runs if self.shared_ocr_model else 0
                ),
                "prediction_releases": [release],
            }
        )

    def _resume_round(self) -> bool:
        """Restore an unfinished round instead of silently replacing its queue."""
        if not self.resume_existing_round or not self.saved_round_id:
            return False
        selected = [
            record
            for record in self.records
            if record.get("selected_for_review")
            and str(record.get("review_round_id") or "") == self.saved_round_id
        ]
        if not selected or not any(
            record.get("status") == "Pending" for record in selected
        ):
            return False
        self.current_round_id = self.saved_round_id
        self.campaign_id = self.saved_round_id
        for review_round in reversed(self.review_rounds):
            if str(review_round.get("id") or "") != self.current_round_id:
                continue
            review_round.setdefault(
                "resumed_at", datetime.now().isoformat(timespec="seconds")
            )
            review_round["resume_count"] = (
                int(review_round.get("resume_count") or 0) + 1
            )
            break
        return True

    def _predictor_release(self, reason: str) -> dict[str, Any]:
        shared_model = getattr(self, "shared_ocr_model", None)
        return {
            "model_id": self.prediction_model_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "training_snapshot_id": self.training_snapshot["id"],
            "training_count": self.training_snapshot["count"],
            "partial_count": self.training_snapshot["partial_count"],
            "ocr_extractor_id": self.ocr_extractor_id,
            "shared_ocr_model_id": (shared_model.model_id if shared_model else ""),
            "shared_ocr_training_count": (
                shared_model.training_readings if shared_model else 0
            ),
            "shared_ocr_training_runs": (
                shared_model.training_runs if shared_model else 0
            ),
        }

    def _refresh_adaptive_predictions(
        self,
        prediction_indices: list[int] | None = None,
        refresh_shared_model: bool = True,
    ) -> bool:
        shared_trainer = getattr(self, "shared_ocr_trainer", None)
        if shared_trainer is not None and refresh_shared_model:
            shared_trainer.update_run(
                self.spec.run, self.spec, self.records, self.ocr_evidence_cache
            )
            self.shared_ocr_model = shared_trainer.model_for_run(self.spec.run)
        shared_model = getattr(self, "shared_ocr_model", None)
        snapshot = prediction_training_snapshot(self.records)
        model_id = prediction_model_id(
            self.spec,
            snapshot["id"],
            shared_model.model_id if shared_model else None,
        )
        if model_id == self.prediction_model_id:
            return False
        self.training_snapshot = snapshot
        self.prediction_model_id = model_id
        for index in self.current_indices():
            record = self.records[index]
            if (
                record.get("status") == "Pending"
                and _active_prediction(record, self.prediction_model_id) is None
            ):
                _apply_prediction_aliases(record, None)
        if prediction_indices is None:
            pending_indices = self.reviewable_pending_indices()
            partial_indices = [
                int(record["index"])
                for record in self.records
                if record["status"] == "Partial" and record.get("reading_pattern")
            ]
            prediction_indices = sorted(set(pending_indices).union(partial_indices))
        if prediction_indices:
            self._build_suggestions(sorted(set(prediction_indices)))
        self.adaptive_refresh_count += 1
        release = self._predictor_release("manual-review-update")
        for review_round in reversed(self.review_rounds):
            if review_round.get("id") != self.current_round_id:
                continue
            review_round.update(
                {
                    "prediction_model_id": self.prediction_model_id,
                    "prediction_training_snapshot_id": self.training_snapshot["id"],
                    "prediction_training_count": self.training_snapshot["count"],
                    "prediction_partial_count": self.training_snapshot["partial_count"],
                    "ocr_extractor_id": self.ocr_extractor_id,
                    "shared_ocr_model_id": (
                        shared_model.model_id if shared_model else ""
                    ),
                    "shared_ocr_training_count": (
                        shared_model.training_readings if shared_model else 0
                    ),
                    "shared_ocr_training_runs": (
                        shared_model.training_runs if shared_model else 0
                    ),
                }
            )
            review_round.setdefault("prediction_releases", []).append(release)
            break
        return True

    def ensure_shared_model_current(self) -> None:
        shared_trainer = getattr(self, "shared_ocr_trainer", None)
        if shared_trainer is None:
            return
        shared_model = getattr(self, "shared_ocr_model", None)
        current_shared_id = shared_trainer.model_id or ""
        loaded_shared_id = shared_model.model_id if shared_model else ""
        if current_shared_id == loaded_shared_id:
            return
        with self.lock:
            self._refresh_adaptive_predictions([])

    def ensure_prediction_current(self, index: int) -> bool:
        """Lazily decode one pending image against the newest reviewed context."""
        with self.lock:
            if not 0 <= index < len(self.records):
                raise ValueError(f"Image index out of range: {index}")
            record = self.records[index]
            if record.get("status") != "Pending":
                return False
            if (
                record.get("selected_for_review")
                and record.get("review_round_id") == self.current_round_id
            ):
                self._assert_review_phase(index)
            started = time.perf_counter()
            model_changed = self._refresh_adaptive_predictions(
                [], refresh_shared_model=False
            )
            prediction = _active_prediction(record, self.prediction_model_id)
            generated = prediction is None
            if generated:
                self._build_suggestions([index])
            elapsed = time.perf_counter() - started
            if model_changed or generated or elapsed >= 0.1:
                print(
                    f"[overflow-ui] prediction {self.spec.run}:{index} "
                    f"model_changed={model_changed} generated={generated} "
                    f"elapsed_s={elapsed:.3f}",
                    flush=True,
                )
            return generated

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": 11,
            "metadata": {
                "run": self.spec.run,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "gravity_feed_g_per_h": self.spec.gravity_feed_g_per_h,
                "gravity_start": self.spec.gravity_start.isoformat(),
                "gravity_end": self.spec.gravity_end.isoformat(),
                "gravity_start_source": self.spec.gravity_start_source,
                "gravity_schedule_verified": self.spec.gravity_schedule_verified,
                "weight_data_valid": (
                    self.weight_data_status != INVALID_WEIGHT_DATA_STATUS
                ),
                "weight_data_status": self.weight_data_status,
                "weight_data_reason": self.weight_data_reason,
                "weight_data_reviewer": self.weight_data_reviewer,
                "weight_data_updated_at": self.weight_data_updated_at,
                "configuration_source": self.spec.configuration_source,
                "image_time_offset_h": self.spec.image_time_offset_h,
                "review_mode": "ocr" if self.spec.ocr_configured else "manual",
                "automatic_ocr": self.spec.automatic_ocr,
                "automatic_ocr_geometry": self.auto_geometry_diagnostics,
                "injection_start": self.injection_start.isoformat(),
                "injection_end": self.injection_end.isoformat(),
                "image_root": str(self.spec.image_root),
                "existing_workbook": str(self.existing_workbook or ""),
                "sample_count": self.sample_count,
                "current_round_id": self.current_round_id,
                "effective_cutoff": self.effective_cutoff.isoformat(),
                "cutoff_rule": self.cutoff_rule,
                "prediction_model": {
                    "name": PREDICTOR_NAME,
                    "version": PREDICTOR_VERSION,
                    "model_id": self.prediction_model_id,
                    "training_snapshot_id": self.training_snapshot["id"],
                    "training_count": self.training_snapshot["count"],
                    "partial_count": self.training_snapshot["partial_count"],
                    "ocr_extractor_id": self.ocr_extractor_id,
                    "image_evidence_is_primary": True,
                    "shared_ocr_model_id": (
                        self.shared_ocr_model.model_id if self.shared_ocr_model else ""
                    ),
                    "shared_ocr_training_count": (
                        self.shared_ocr_model.training_readings
                        if self.shared_ocr_model
                        else 0
                    ),
                    "shared_ocr_training_cells": (
                        self.shared_ocr_model.training_cells
                        if self.shared_ocr_model
                        else 0
                    ),
                    "shared_ocr_training_runs": (
                        self.shared_ocr_model.training_runs
                        if self.shared_ocr_model
                        else 0
                    ),
                },
                "adaptive_refresh_count": self.adaptive_refresh_count,
                "predictions_generated_this_load": self.predictions_generated,
                "predictions_reused_this_load": self.predictions_reused,
                "ocr_evidence_generated_this_load": self.ocr_evidence_generated,
                "ocr_evidence_reused_this_load": self.ocr_evidence_reused,
            },
            "review_rounds": self.review_rounds,
            "records": self.records,
            "imported_records": self.imported_records,
        }

    def _persist(
        self,
        request_excel: bool = True,
        write_prediction_cache: bool = False,
    ) -> None:
        if self.prediction_cache_dirty and write_prediction_cache:
            for record in self.records:
                predictions = list(record.get("predictions") or [])
                if not predictions:
                    continue
                cached = self.prediction_cache.get(record["image_id"], {})
                self.prediction_cache[record["image_id"]] = {
                    "image_id": record["image_id"],
                    "image_time": record["image_time"],
                    "image": record["image"],
                    "relative_path": record["relative_path"],
                    "predictions": _merge_predictions(
                        list(cached.get("predictions") or []), predictions
                    ),
                }
            _write_prediction_cache(
                self.prediction_cache_path, self.spec, self.prediction_cache
            )
        if self.prediction_cache_dirty:
            # The review state is authoritative and already contains every
            # prediction. Rewriting the duplicate multi-megabyte cache after
            # each click only adds network I/O; it is refreshed at round setup.
            self.prediction_cache_dirty = False
        if self.ocr_evidence_cache_dirty:
            _write_ocr_evidence_cache(
                self.ocr_evidence_cache_path, self.spec, self.ocr_evidence_cache
            )
            self.ocr_evidence_cache_dirty = False
        payload = self._payload()
        _write_review_payload(self.state_path, self.csv_path, payload)
        if request_excel:
            self.sync.request()

    def current_indices(self) -> list[int]:
        indices = [
            record["index"]
            for record in self.records
            if record["review_round_id"] == self.current_round_id
            and record["selected_for_review"]
        ]
        return sorted(
            indices,
            key=lambda index: (
                (
                    self.records[index].get("review_position")
                    if self.records[index].get("review_position") is not None
                    else len(self.records)
                ),
                index,
            ),
        )

    def bootstrap_indices(self) -> list[int]:
        return [
            index
            for index in self.current_indices()
            if self.records[index].get("review_selection_phase") == "bootstrap-anchor"
        ]

    def pending_bootstrap_indices(self) -> list[int]:
        return [
            index
            for index in self.bootstrap_indices()
            if self.records[index].get("status") == "Pending"
        ]

    def reviewable_pending_indices(self) -> list[int]:
        """Expose gap-fill images only after every bootstrap image is reviewed."""
        bootstrap_pending = self.pending_bootstrap_indices()
        if bootstrap_pending:
            return bootstrap_pending
        return [
            index
            for index in self.current_indices()
            if self.records[index].get("status") == "Pending"
        ]

    def review_phase_indices(self) -> list[int]:
        return (
            self.bootstrap_indices()
            if self.pending_bootstrap_indices()
            else self.current_indices()
        )

    def _assert_review_phase(self, index: int) -> None:
        record = self.records[index]
        bootstrap_pending = self.pending_bootstrap_indices()
        if (
            bootstrap_pending
            and record.get("review_selection_phase") != "bootstrap-anchor"
        ):
            raise ValueError(
                "Complete the initial time-distributed anchor sweep before "
                "reviewing gap-fill images"
            )

    def save(self, index: int, reading: Any, unreadable: bool) -> dict[str, Any]:
        with self.lock:
            started = time.perf_counter()
            record = self.records[index]
            if (
                not record["selected_for_review"]
                or record["review_round_id"] != self.current_round_id
            ):
                raise ValueError(
                    "This image is not part of the current validation round"
                )
            self._assert_review_phase(index)
            self.ensure_prediction_current(index)
            if (
                self.spec.ocr_configured
                and not unreadable
                and _float_or_none(record.get("suggested_g")) is None
            ):
                raise RuntimeError("The selected image has no active prediction")
            _apply_review_edit(record, reading, unreadable, self.reviewer)
            reviewed_at = time.perf_counter()
            refresh_shared_model = record["status"] in {"Corrected", "Partial"}
            self._refresh_adaptive_predictions(
                [], refresh_shared_model=refresh_shared_model
            )
            model_at = time.perf_counter()
            self._persist()
            persisted_at = time.perf_counter()
            print(
                f"[overflow-ui] save {self.spec.run}:{index} "
                f"review_s={reviewed_at - started:.3f} "
                f"shared_refresh={refresh_shared_model} "
                f"model_s={model_at - reviewed_at:.3f} "
                f"persist_s={persisted_at - model_at:.3f} "
                f"total_s={persisted_at - started:.3f}",
                flush=True,
            )
            return dict(record)

    def edit(
        self,
        index: int,
        reading: Any,
        unreadable: bool,
        reviewer: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if not 0 <= index < len(self.records):
                raise ValueError(f"Image index out of range: {index}")
            record = self.records[index]
            if (
                record.get("status") == "Pending"
                and record.get("selected_for_review")
                and record.get("review_round_id") == self.current_round_id
            ):
                raise ValueError(
                    "This image belongs to an unfinished validation round; "
                    "resume that round instead of editing it as history"
                )
            _apply_review_edit(record, reading, unreadable, reviewer, notes)
            self._refresh_adaptive_predictions(
                [], refresh_shared_model=record["status"] in {"Corrected", "Partial"}
            )
            self._persist()
            return dict(record)

    def set_weight_data_status(
        self,
        status: str,
        reason: str,
        reviewer: str,
        updated_at: str,
    ) -> None:
        with self.lock:
            self.weight_data_status = str(status)
            self.weight_data_reason = str(reason)
            self.weight_data_reviewer = str(reviewer)
            self.weight_data_updated_at = str(updated_at)
            self._persist()

    def summary(self) -> dict[str, Any]:
        counts = {status: 0 for status in sorted(ALL_STATUSES)}
        for record in self.records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        indices = self.current_indices()
        current_counts = {status: 0 for status in sorted(ALL_STATUSES)}
        for index in indices:
            status = self.records[index]["status"]
            current_counts[status] = current_counts.get(status, 0) + 1
        pending = [
            index for index in indices if self.records[index]["status"] == "Pending"
        ]
        bootstrap = self.bootstrap_indices()
        bootstrap_pending = self.pending_bootstrap_indices()
        reviewable_pending = self.reviewable_pending_indices()
        return {
            "run": self.spec.run,
            "total_images": len(self.records),
            "counts": counts,
            "current_counts": current_counts,
            "current_round_id": self.current_round_id,
            "current_indices": indices,
            "current_total": len(indices),
            "current_done": len(indices) - len(pending),
            "next_pending": (
                reviewable_pending[0]
                if reviewable_pending
                else (indices[-1] if indices else 0)
            ),
            "pending_indices": pending,
            "reviewable_pending_indices": reviewable_pending,
            "bootstrap_total": len(bootstrap),
            "bootstrap_done": len(bootstrap) - len(bootstrap_pending),
            "bootstrap_pending": len(bootstrap_pending),
            "bootstrap_active": bool(bootstrap_pending),
            "gravity_feed_g_per_h": self.spec.gravity_feed_g_per_h,
            "gravity_start": self.spec.gravity_start.isoformat(),
            "gravity_end": self.spec.gravity_end.isoformat(),
            "gravity_start_source": self.spec.gravity_start_source,
            "gravity_schedule_verified": self.spec.gravity_schedule_verified,
            "image_time_offset_h": self.spec.image_time_offset_h,
            "weight_data_valid": (
                self.weight_data_status != INVALID_WEIGHT_DATA_STATUS
            ),
            "weight_data_status": self.weight_data_status,
            "weight_data_reason": self.weight_data_reason,
            "weight_data_reviewer": self.weight_data_reviewer,
            "weight_data_updated_at": self.weight_data_updated_at,
            "review_mode": "ocr" if self.spec.ocr_configured else "manual",
            "automatic_ocr": self.spec.automatic_ocr,
            "injection_start": self.injection_start.isoformat(),
            "injection_end": self.injection_end.isoformat(),
            "injection_end_h": (
                self.injection_end - self.injection_start
            ).total_seconds()
            / 3600.0,
            "gravity_end_h": (
                self.spec.gravity_end - self.injection_start
            ).total_seconds()
            / 3600.0,
            "effective_cutoff": self.effective_cutoff.isoformat(),
            "cutoff_rule": self.cutoff_rule,
            "existing_workbook": str(self.existing_workbook or ""),
            "workbook": self.sync.summary(),
            "prediction_model_id": self.prediction_model_id,
            "prediction_training_snapshot_id": self.training_snapshot["id"],
            "prediction_training_count": self.training_snapshot["count"],
            "prediction_partial_count": self.training_snapshot["partial_count"],
            "shared_ocr_model_id": (
                self.shared_ocr_model.model_id if self.shared_ocr_model else ""
            ),
            "shared_ocr_training_count": (
                self.shared_ocr_model.training_readings if self.shared_ocr_model else 0
            ),
            "shared_ocr_training_cells": (
                self.shared_ocr_model.training_cells if self.shared_ocr_model else 0
            ),
            "shared_ocr_training_runs": (
                self.shared_ocr_model.training_runs if self.shared_ocr_model else 0
            ),
            "ocr_extractor_id": self.ocr_extractor_id,
            "adaptive_refresh_count": self.adaptive_refresh_count,
            "predictions_generated_this_load": self.predictions_generated,
            "predictions_reused_this_load": self.predictions_reused,
            "ocr_evidence_generated_this_load": self.ocr_evidence_generated,
            "ocr_evidence_reused_this_load": self.ocr_evidence_reused,
            "confirmed_total": sum(
                record["status"] in CONFIRMED_STATUSES for record in self.records
            ),
            "reviewed_total": sum(
                record["status"] in REVIEWED_STATUSES for record in self.records
            ),
        }

    def series(self) -> list[dict[str, Any]]:
        return [
            {
                "index": record["index"],
                "h": record["time_from_injection_h"],
                "weight_g": record["weight_g"],
                "suggested_g": record["suggested_g"],
                "existing_value_g": record["existing_value_g"],
                "prediction_model_id": record["prediction_model_id"],
                "status": record["status"],
                "reading_pattern": record.get("reading_pattern", ""),
                "image_time": record["image_time"],
            }
            for record in self.records
        ]

    def restore(self, index: int, snapshot: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.records[index] = copy.deepcopy(snapshot)
            self._refresh_adaptive_predictions([])
            self._persist()
            return dict(self.records[index])


class ValidationCampaign:
    def __init__(
        self,
        campaign_id: str,
        sessions: dict[str, RunSession],
        selected_runs: list[str],
        requested_count: int,
        strategy: str,
        allocations: dict[str, int],
        inventories: dict[str, RunInventory],
    ):
        self.id = campaign_id
        self.sessions = sessions
        self.selected_runs = list(selected_runs)
        self.requested_count = requested_count
        self.strategy = strategy
        self.allocations = dict(allocations)
        self.inventories = inventories
        self.history: list[dict[str, Any]] = []
        self.invalid_runs: dict[str, dict[str, str]] = {}
        self.lock = threading.RLock()
        queue_with_positions: list[tuple[float, int, str, int]] = []
        for run in selected_runs:
            session = sessions.get(run)
            if session is None:
                continue
            indices = session.current_indices()
            denominator = max(1, len(indices) - 1)
            for position, index in enumerate(indices):
                run_number = int(re.sub(r"\D", "", run) or 0)
                queue_with_positions.append(
                    (position / denominator, run_number, run, index)
                )
        queue_with_positions.sort()
        self.queue = [
            {"run": run, "index": index} for _, _, run, index in queue_with_positions
        ]
        self.confirmed_before = {
            run: (
                sessions[run].summary()["confirmed_total"]
                if run in sessions
                else inventories[run].confirmed_count
            )
            for run in selected_runs
        }

    def save(
        self, run: str, index: int, reading: Any, unreadable: bool
    ) -> dict[str, Any]:
        with self.lock:
            if {"run": run, "index": index} not in self.queue:
                raise ValueError("This image is not part of the active campaign")
            if run in self.invalid_runs:
                raise ValueError(
                    f"{run.upper()} is marked as invalid overflow-weight data"
                )
            session = self.sessions[run]
            snapshot = copy.deepcopy(session.records[index])
            record = session.save(index, reading, unreadable)
            self.history.append(
                {
                    "run": run,
                    "index": index,
                    "snapshot": snapshot,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            return record

    def set_run_invalidity(
        self,
        run: str,
        invalid: bool,
        reason: str,
        reviewer: str,
        updated_at: str,
    ) -> None:
        run = run.lower()
        with self.lock:
            if run not in self.selected_runs:
                return
            if invalid:
                self.invalid_runs[run] = {
                    "reason": str(reason),
                    "reviewer": str(reviewer),
                    "updated_at": str(updated_at),
                }
            else:
                self.invalid_runs.pop(run, None)

    def undo(self) -> dict[str, Any]:
        with self.lock:
            if not self.history:
                raise ValueError("There is no entry to undo in this campaign")
            action = self.history.pop()
            record = self.sessions[action["run"]].restore(
                action["index"], action["snapshot"]
            )
            return {"run": action["run"], "index": action["index"], "record": record}

    def summary(self) -> dict[str, Any]:
        run_summaries = {
            run: session.summary() for run, session in self.sessions.items()
        }
        status_counts = {status: 0 for status in sorted(ALL_STATUSES)}
        status_counts["SkippedInvalid"] = 0
        for item in self.queue:
            status = self.sessions[item["run"]].records[item["index"]]["status"]
            if item["run"] in self.invalid_runs and status == "Pending":
                status = "SkippedInvalid"
            status_counts[status] = status_counts.get(status, 0) + 1
        reviewable_by_run = {
            run: set(
                session.reviewable_pending_indices()
                if hasattr(session, "reviewable_pending_indices")
                else [
                    index
                    for index in session.current_indices()
                    if session.records[index]["status"] == "Pending"
                ]
            )
            for run, session in self.sessions.items()
        }
        review_phase_by_run = {
            run: set(
                session.review_phase_indices()
                if hasattr(session, "review_phase_indices")
                else session.current_indices()
            )
            for run, session in self.sessions.items()
        }
        all_pending_queue = [
            item
            for item in self.queue
            if item["run"] not in self.invalid_runs
            and self.sessions[item["run"]].records[item["index"]]["status"] == "Pending"
        ]
        pending_queue = [
            item
            for item in all_pending_queue
            if item["index"] in reviewable_by_run[item["run"]]
        ]
        review_queue = [
            item
            for item in self.queue
            if item["run"] not in self.invalid_runs
            and item["index"] in review_phase_by_run[item["run"]]
        ]
        rows = []
        for run in self.selected_runs:
            session_summary = run_summaries.get(run)
            invalid_record = self.invalid_runs.get(run)
            skipped = (
                sum(
                    item["run"] == run
                    and self.sessions[run].records[item["index"]]["status"] == "Pending"
                    for item in self.queue
                )
                if invalid_record
                else 0
            )
            rows.append(
                {
                    "run": run,
                    "confirmed_before": self.confirmed_before[run],
                    "confirmed_now": (
                        session_summary["confirmed_total"]
                        if session_summary
                        else self.confirmed_before[run]
                    ),
                    "reviewed_before": self.inventories[run].reviewed_count,
                    "capacity": self.inventories[run].capacity,
                    "allocated": (
                        session_summary["current_total"] if session_summary else 0
                    ),
                    "done": session_summary["current_done"] if session_summary else 0,
                    "skipped": skipped,
                    "invalid": invalid_record is not None,
                    "invalid_reason": (
                        invalid_record.get("reason", "") if invalid_record else ""
                    ),
                    "invalid_reviewer": (
                        invalid_record.get("reviewer", "") if invalid_record else ""
                    ),
                    "invalid_updated_at": (
                        invalid_record.get("updated_at", "") if invalid_record else ""
                    ),
                }
            )
        skipped_total = status_counts.get("SkippedInvalid", 0)
        reviewed_done = len(self.queue) - len(all_pending_queue) - skipped_total
        bootstrap_pending = sum(
            int(summary.get("bootstrap_pending") or 0)
            for summary in run_summaries.values()
        )
        return {
            "id": self.id,
            "strategy": self.strategy,
            "requested_count": self.requested_count,
            "queue": self.queue,
            "review_queue": review_queue,
            "pending_queue": pending_queue,
            "all_pending_count": len(all_pending_queue),
            "locked_pending_count": len(all_pending_queue) - len(pending_queue),
            "bootstrap_pending": bootstrap_pending,
            "total": len(self.queue),
            "done": reviewed_done + skipped_total,
            "reviewed_done": reviewed_done,
            "skipped_invalid": skipped_total,
            "status_counts": status_counts,
            "can_undo": bool(self.history),
            "last_action": (
                {key: self.history[-1][key] for key in ("run", "index", "saved_at")}
                if self.history
                else None
            ),
            "runs": rows,
            "run_summaries": run_summaries,
        }


class ReviewApp:
    def __init__(
        self,
        specs: dict[str, RunSpec],
        output_root: Path,
        bridge: ArtifactBridge,
        raw_root: Path = DEFAULT_RAW_ROOT,
        production_runs: tuple[str, ...] | list[str] | None = None,
        image_time_offsets_h: dict[str, float] | None = None,
    ):
        self.specs = dict(specs)
        self.configured_runs = frozenset(specs)
        self.output_root = output_root
        self.bridge = bridge
        self.raw_root = raw_root
        self.image_time_offsets_h = {
            str(run).lower(): float(value)
            for run, value in (image_time_offsets_h or {}).items()
        }
        catalog_runs = production_runs if production_runs is not None else tuple(specs)
        self.production_runs = tuple(
            sorted(
                {str(run).lower() for run in catalog_runs} | set(specs),
                key=self._run_key,
            )
        )
        self.campaign: ValidationCampaign | None = None
        self.lock = threading.RLock()
        self.source_lock = threading.RLock()
        self.job: dict[str, Any] = {"state": "idle", "progress": 0.0, "message": ""}
        self.source_job: dict[str, Any] = {
            "state": "idle",
            "progress": 0.0,
            "message": "",
        }
        self.campaign_sequence = 0
        self.inventory_cache: dict[tuple[Any, ...], RunInventory] = {}
        self.overview_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        self.run_payload_cache: dict[
            str, tuple[tuple[int, int], float, dict[str, Any]]
        ] = {}
        self.explorer_syncs: dict[str, WorkbookSync] = {}
        self.source_catalog = load_source_catalog(
            output_root=self.output_root,
            runs=self.production_runs,
            configured_roots=self._configured_roots(),
        )
        self._sync_manual_specs()
        self.shared_ocr_trainer = SharedOCRTrainer(self.output_root, self.specs)
        self.edit_locks: dict[str, threading.RLock] = {
            run: threading.RLock() for run in self.specs
        }

    @staticmethod
    def _run_key(run: str) -> int:
        return int(re.sub(r"\D", "", run) or 0)

    def _configured_roots(self) -> dict[str, Path]:
        return {
            run: self.specs[run].image_root
            for run in self.configured_runs
            if run in self.specs
        }

    def _sync_manual_specs(self) -> None:
        for run, source in self.source_catalog.items():
            if run in self.configured_runs:
                continue
            manual_spec = provisional_manual_spec(
                run,
                source,
                self.image_time_offsets_h.get(run),
            )
            if manual_spec is None:
                self.specs.pop(run, None)
                continue
            cached_geometry = _load_auto_geometry(
                _auto_geometry_cache_path(self.output_root / run, run),
                manual_spec,
            )
            if cached_geometry is not None:
                manual_spec = replace(
                    manual_spec,
                    cells=cached_geometry[0],
                    cells_by_image_size=cached_geometry[1],
                    screen_boxes_by_image_size=cached_geometry[2],
                    configuration_source=(
                        f"automatic-ocr:{AUTO_GEOMETRY_DETECTOR_VERSION}"
                    ),
                )
            self.specs[run] = manual_spec
            if hasattr(self, "edit_locks"):
                self.edit_locks.setdefault(run, threading.RLock())

    def _reload_source_catalog(self) -> None:
        self.source_catalog = load_source_catalog(
            output_root=self.output_root,
            runs=self.production_runs,
            configured_roots=self._configured_roots(),
        )
        self._sync_manual_specs()
        self.overview_cache.clear()

    def source_detail(self, run: str) -> dict[str, Any]:
        run = run.lower()
        with self.source_lock:
            row = self.source_catalog.get(run)
            if row is None:
                raise ValueError(f"Unknown run: {run}")
            result = copy.deepcopy(row)
        for index, sample in enumerate(result.get("samples") or []):
            sample["url"] = f"/source/image?run={run}&sample={index}"
        result["triage_statuses"] = list(TRIAGE_STATUSES)
        return result

    def update_source_status(
        self, run: str, status: str, reason: str, reviewer: str
    ) -> dict[str, Any]:
        run = run.lower()
        if run not in self.source_catalog:
            raise ValueError(f"Unknown run: {run}")
        reason = reason.strip()
        reviewer = reviewer.strip() or DEFAULT_REVIEWER
        if status == INVALID_WEIGHT_DATA_STATUS and not reason:
            raise ValueError(
                "A reason is required when overflow-weight data are marked invalid"
            )
        with self.source_lock:
            triage = save_run_triage(self.output_root, run, status, reason, reviewer)
            self._reload_source_catalog()
        session = self._active_session(run)
        if session is not None:
            session.set_weight_data_status(
                status,
                reason,
                reviewer,
                str(triage["updated_at"]),
            )
        else:
            state_path, csv_path, _, _ = self._run_paths(run)
            if state_path.exists():
                with self.edit_locks.setdefault(run, threading.RLock()):
                    payload = json.loads(state_path.read_text(encoding="utf-8"))
                    metadata = payload.setdefault("metadata", {})
                    metadata.update(
                        {
                            "weight_data_valid": (status != INVALID_WEIGHT_DATA_STATUS),
                            "weight_data_status": status,
                            "weight_data_reason": reason,
                            "weight_data_reviewer": reviewer,
                            "weight_data_updated_at": triage["updated_at"],
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    _write_review_payload(state_path, csv_path, payload)
                    with self.lock:
                        self.run_payload_cache.pop(run, None)
                    self._sync_for_run(run).request(include_preview=True)
        if self.campaign is not None:
            self.campaign.set_run_invalidity(
                run,
                status == INVALID_WEIGHT_DATA_STATUS,
                reason,
                reviewer,
                str(triage["updated_at"]),
            )
        self.overview_cache.pop(run, None)
        result = self.source_detail(run)
        result["campaign"] = (
            self.campaign.summary() if self.campaign is not None else None
        )
        return result

    def source_image_path(self, run: str, sample: int) -> Path:
        with self.source_lock:
            return source_sample_path(self.source_catalog, run, sample)

    def start_source_refresh(self) -> None:
        with self.source_lock:
            if self.source_job.get("state") == "running":
                raise ValueError("Overflow source scan is already running")
            self.source_job = {
                "state": "running",
                "progress": 0.05,
                "message": "Scanning overflow camera sources on the raw-data volume",
            }

        def build() -> None:
            try:
                build_source_inventory(
                    raw_root=self.raw_root,
                    output_root=self.output_root,
                    runs=self.production_runs,
                    configured_roots=self._configured_roots(),
                    image_time_offsets_h=self.image_time_offsets_h,
                )
                with self.source_lock:
                    self._reload_source_catalog()
                    self.source_job = {
                        "state": "ready",
                        "progress": 1.0,
                        "message": "Overflow source inventory is current",
                    }
            except Exception as exc:
                with self.source_lock:
                    self.source_job = {
                        "state": "error",
                        "progress": 0.0,
                        "message": str(exc),
                    }

        threading.Thread(target=build, daemon=True).start()

    def source_job_summary(self) -> dict[str, Any]:
        with self.source_lock:
            return dict(self.source_job)

    def workbook_candidates(self, run: str) -> list[str]:
        spec = self.specs[run]
        generated = self.output_root / run / f"{run}_overflow_scale_review.xlsx"
        candidates: list[Path] = []
        if generated.exists():
            candidates.append(generated)
        parent = spec.image_root.parent
        for path in parent.glob("*.xlsx"):
            if path.name.startswith("~$"):
                continue
            name = path.name.lower()
            if any(token in name for token in ("overflow", "weight", "scale", "vekt")):
                candidates.append(path)
        unique: list[str] = []
        seen: set[str] = set()
        for path in candidates:
            key = _normalise_path(path)
            if key not in seen:
                seen.add(key)
                unique.append(str(path))
        return unique

    def _state_counts(self, run: str) -> tuple[int, int]:
        state_path = self.output_root / run / f"{run}_overflow_scale_review.json"
        if not state_path.exists():
            return 0, 0
        try:
            records = json.loads(state_path.read_text(encoding="utf-8")).get(
                "records", []
            )
        except (OSError, json.JSONDecodeError):
            return 0, 0
        confirmed = sum(
            str(record.get("status")) in CONFIRMED_STATUSES for record in records
        )
        reviewed = sum(
            str(record.get("status")) in REVIEWED_STATUSES for record in records
        )
        return confirmed, reviewed

    def _run_paths(self, run: str) -> tuple[Path, Path, Path, Path]:
        output = self.output_root / run
        return (
            output / f"{run}_overflow_scale_review.json",
            output / f"{run}_overflow_scale_review.csv",
            output / f"{run}_overflow_scale_review.xlsx",
            output / f"{run}_overflow_scale_review_summary.png",
        )

    def _active_session(self, run: str) -> RunSession | None:
        if self.campaign is None:
            return None
        return self.campaign.sessions.get(run)

    def _load_run_payload(self, run: str, *, mutable: bool = False) -> dict[str, Any]:
        run = run.lower()
        if run not in self.specs:
            raise ValueError(f"Unknown run: {run}")
        session = self._active_session(run)
        if session is not None:
            with session.lock:
                payload = session._payload()
                return copy.deepcopy(payload) if mutable else payload
        state_path, _, _, _ = self._run_paths(run)
        if not state_path.exists():
            raise ValueError(f"No overflow data have been created for {run.upper()}")
        with self.lock:
            cached = self.run_payload_cache.get(run)
            checked_at = time.monotonic()
            if cached is not None and checked_at - cached[1] < RUN_PAYLOAD_STAT_TTL_S:
                payload = cached[2]
            else:
                stat = state_path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if cached is not None and cached[0] == signature:
                    payload = cached[2]
                else:
                    payload = json.loads(state_path.read_text(encoding="utf-8"))
                    if _repair_nonfinite_json(state_path, payload):
                        stat = state_path.stat()
                        signature = (stat.st_mtime_ns, stat.st_size)
                self.run_payload_cache[run] = (signature, checked_at, payload)
        return copy.deepcopy(payload) if mutable else payload

    def _sync_for_run(self, run: str) -> WorkbookSync:
        session = self._active_session(run)
        if session is not None:
            return session.sync
        with self.lock:
            sync = self.explorer_syncs.get(run)
            if sync is None:
                state_path, _, workbook_path, preview_path = self._run_paths(run)
                sync = WorkbookSync(
                    self.bridge, state_path, workbook_path, preview_path
                )
                if workbook_path.exists():
                    sync.state = "current"
                self.explorer_syncs[run] = sync
            return sync

    @staticmethod
    def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
        counts = {status: 0 for status in sorted(ALL_STATUSES)}
        for record in records:
            status = str(record.get("status") or "Not selected")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _source_overview_fields(self, run: str) -> dict[str, Any]:
        source = self.source_catalog.get(run, {})
        image_count = source.get("image_count")
        relevant_count = source.get("relevant_image_count")
        spec = self.specs.get(run)
        source_status = source.get("triage_status", "not-scanned")
        weight_data_invalid = source_status == INVALID_WEIGHT_DATA_STATUS
        return {
            "configured": run in self.configured_runs,
            "review_available": spec is not None and not weight_data_invalid,
            "review_mode": (
                "ocr"
                if spec is not None and spec.ocr_configured
                else "manual" if spec is not None else "unavailable"
            ),
            "automatic_ocr": bool(spec is not None and spec.automatic_ocr),
            "gravity_schedule_verified": bool(
                spec is not None and spec.gravity_schedule_verified
            ),
            "source_status": source_status,
            "weight_data_invalid": weight_data_invalid,
            "source_reason": source.get("triage_reason", ""),
            "source_root": source.get("selected_root"),
            "source_images": image_count,
            "source_relevant_images": relevant_count,
            "source_first": source.get("first_image_time"),
            "source_last": source.get("last_image_time"),
            "source_coverage_start_h": source.get("coverage_start_h"),
            "source_coverage_end_h": source.get("coverage_end_h"),
            "source_image_time_offset_h": source.get("image_time_offset_h", 0.0),
            "source_sample_count": len(source.get("samples") or []),
            "source_scanned": image_count is not None,
        }

    def _overview_row(self, run: str) -> dict[str, Any]:
        state_path, _, workbook_path, _ = self._run_paths(run)
        if not state_path.exists():
            return {
                "run": run,
                "total": 0,
                "exact": 0,
                "partial": 0,
                "unreadable": 0,
                "reviewed": 0,
                "remaining": 0,
                "review_percent": 0.0,
                "updated_at": None,
                "has_data": False,
                "campaign_active": self._active_session(run) is not None,
                "unfinished_round": False,
                "unfinished_round_id": "",
                "unfinished_pending": 0,
                "unfinished_bootstrap_pending": 0,
                "workbook": str(workbook_path),
                **self._source_overview_fields(run),
            }
        mtime = state_path.stat().st_mtime_ns
        cached = self.overview_cache.get(run)
        if cached is not None and cached[0] == mtime:
            row = dict(cached[1])
            row["campaign_active"] = self._active_session(run) is not None
            return row
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        records = list(payload.get("records", []))
        metadata = dict(payload.get("metadata") or {})
        current_round_id = str(metadata.get("current_round_id") or "")
        current_records = [
            record
            for record in records
            if bool(record.get("selected_for_review"))
            and str(record.get("review_round_id") or "") == current_round_id
        ]
        unfinished = [
            record for record in current_records if record.get("status") == "Pending"
        ]
        unfinished_bootstrap = [
            record
            for record in unfinished
            if record.get("review_selection_phase") == "bootstrap-anchor"
        ]
        counts = self._counts(records)
        exact = sum(counts.get(status, 0) for status in CONFIRMED_STATUSES)
        partial = counts.get("Partial", 0)
        unreadable = counts.get("Unreadable", 0)
        reviewed = exact + partial + unreadable
        total = len(records)
        row = {
            "run": run,
            "total": total,
            "exact": exact,
            "partial": partial,
            "unreadable": unreadable,
            "reviewed": reviewed,
            "remaining": max(0, total - reviewed),
            "review_percent": 100.0 * reviewed / total if total else 0.0,
            "updated_at": metadata.get("updated_at"),
            "has_data": bool(records),
            "campaign_active": self._active_session(run) is not None,
            "unfinished_round": bool(unfinished),
            "unfinished_round_id": current_round_id if unfinished else "",
            "unfinished_pending": len(unfinished),
            "unfinished_bootstrap_pending": len(unfinished_bootstrap),
            "workbook": str(workbook_path),
            **self._source_overview_fields(run),
        }
        self.overview_cache[run] = (mtime, dict(row))
        return row

    def overview(self) -> dict[str, Any]:
        rows = [self._overview_row(run) for run in self.production_runs]
        return {
            "runs": rows,
            "totals": {
                "runs": len(rows),
                "configured": sum(row["configured"] for row in rows),
                "with_ocr": sum(row["review_mode"] == "ocr" for row in rows),
                "review_available": sum(row["review_available"] for row in rows),
                "with_source_images": sum(
                    (row["source_images"] or 0) > 0 for row in rows
                ),
                "source_ready": sum(row["source_status"] == "ready" for row in rows),
                "source_needs_review": sum(
                    row["source_status"] in {"needs-review", "not-scanned"}
                    for row in rows
                ),
                "invalid_weight_data": sum(row["weight_data_invalid"] for row in rows),
                "with_data": sum(row["has_data"] for row in rows),
                "images": sum(row["total"] for row in rows),
                "exact": sum(row["exact"] for row in rows),
                "partial": sum(row["partial"] for row in rows),
                "unreadable": sum(row["unreadable"] for row in rows),
                "reviewed": sum(row["reviewed"] for row in rows),
            },
            "campaign": self.campaign.summary() if self.campaign else None,
        }

    def explore_summary(self, run: str) -> dict[str, Any]:
        payload = self._load_run_payload(run)
        records = list(payload.get("records", []))
        counts = self._counts(records)
        metadata = dict(payload.get("metadata", {}))
        exact = sum(counts.get(status, 0) for status in CONFIRMED_STATUSES)
        reviewed = sum(counts.get(status, 0) for status in REVIEWED_STATUSES)
        injection_start = datetime.fromisoformat(metadata["injection_start"])
        injection_end = datetime.fromisoformat(metadata["injection_end"])
        gravity_end = datetime.fromisoformat(metadata["gravity_end"])
        return {
            "run": run,
            "total_images": len(records),
            "counts": counts,
            "exact": exact,
            "reviewed": reviewed,
            "injection_end_h": (injection_end - injection_start).total_seconds()
            / 3600.0,
            "gravity_end_h": (gravity_end - injection_start).total_seconds() / 3600.0,
            "gravity_feed_g_per_h": metadata.get("gravity_feed_g_per_h"),
            "gravity_start": metadata.get("gravity_start"),
            "gravity_end": metadata.get("gravity_end"),
            "gravity_schedule_verified": metadata.get(
                "gravity_schedule_verified", True
            ),
            "review_mode": metadata.get("review_mode", "ocr"),
            "updated_at": metadata.get("updated_at"),
            "workbook": self._sync_for_run(run).summary(),
        }

    def explore_series(self, run: str) -> list[dict[str, Any]]:
        payload = self._load_run_payload(run)
        return [
            {
                "index": record["index"],
                "h": record["time_from_injection_h"],
                "weight_g": record.get("weight_g"),
                "suggested_g": record.get("suggested_g"),
                "existing_value_g": record.get("existing_value_g"),
                "prediction_model_id": record.get("prediction_model_id"),
                "status": record.get("status"),
                "reading_pattern": record.get("reading_pattern", ""),
                "image_time": record.get("image_time"),
            }
            for record in payload.get("records", [])
        ]

    def explore_record(self, run: str, index: int) -> dict[str, Any]:
        payload = self._load_run_payload(run)
        records = list(payload.get("records", []))
        if not 0 <= index < len(records):
            raise ValueError(f"Image index out of range: {index}")
        record = records[index]
        if int(record.get("index", index)) != index:
            matches = [item for item in records if int(item.get("index", -1)) == index]
            if not matches:
                raise ValueError(f"Image index not found: {index}")
            record = matches[0]
        return dict(record)

    def edit_explore_record(
        self,
        run: str,
        index: int,
        reading: Any,
        unreadable: bool,
        reviewer: str,
        notes: str | None,
    ) -> dict[str, Any]:
        run = run.lower()
        reviewer = reviewer.strip() or DEFAULT_REVIEWER
        session = self._active_session(run)
        if session is not None:
            record = session.edit(index, reading, unreadable, reviewer, notes)
        else:
            with self.edit_locks[run]:
                state_path, csv_path, _, _ = self._run_paths(run)
                payload = self._load_run_payload(run, mutable=True)
                records = list(payload.get("records", []))
                if not 0 <= index < len(records):
                    raise ValueError(f"Image index out of range: {index}")
                record = records[index]
                current_round_id = str(
                    payload.get("metadata", {}).get("current_round_id") or ""
                )
                if (
                    record.get("status") == "Pending"
                    and record.get("selected_for_review")
                    and str(record.get("review_round_id") or "") == current_round_id
                ):
                    raise ValueError(
                        "This image belongs to an unfinished validation round; "
                        "resume that round instead of editing it as history"
                    )
                _apply_review_edit(record, reading, unreadable, reviewer, notes)
                payload.setdefault("metadata", {})[
                    "updated_at"
                ] = datetime.now().isoformat(timespec="seconds")
                _write_review_payload(state_path, csv_path, payload)
                with self.lock:
                    self.run_payload_cache.pop(run, None)
                self.shared_ocr_trainer.refresh_run_from_disk(run, self.specs[run])
                self._sync_for_run(run).request(include_preview=True)
                record = dict(record)
        self.overview_cache.pop(run, None)
        self.inventory_cache = {
            key: value for key, value in self.inventory_cache.items() if key[0] != run
        }
        return record

    def options(self) -> dict[str, Any]:
        runs = []
        for run in sorted(self.specs, key=self._run_key):
            confirmed, reviewed = self._state_counts(run)
            source_status = self.source_catalog.get(run, {}).get(
                "triage_status", "not-scanned"
            )
            runs.append(
                {
                    "run": run,
                    "image_root": str(self.specs[run].image_root),
                    "workbooks": self.workbook_candidates(run),
                    "confirmed": confirmed,
                    "reviewed": reviewed,
                    "source_status": source_status,
                    "source_reason": self.source_catalog.get(run, {}).get(
                        "triage_reason", ""
                    ),
                    "weight_data_invalid": (
                        source_status == INVALID_WEIGHT_DATA_STATUS
                    ),
                    "source_relevant_images": self.source_catalog.get(run, {}).get(
                        "relevant_image_count"
                    ),
                    "review_mode": (
                        "ocr" if self.specs[run].ocr_configured else "manual"
                    ),
                    "automatic_ocr": self.specs[run].automatic_ocr,
                    "gravity_schedule_verified": self.specs[
                        run
                    ].gravity_schedule_verified,
                    "image_time_offset_h": self.specs[run].image_time_offset_h,
                }
            )
        return {
            "runs": runs,
            "default_reviewer": DEFAULT_REVIEWER,
            "output_root": str(self.output_root),
            "shared_ocr": {
                "model_id": self.shared_ocr_trainer.model_id or "",
                **dict(self.shared_ocr_trainer.snapshot),
            },
        }

    def _selected_runs(self, payload: dict[str, Any]) -> list[str]:
        scope = str(payload.get("scope", "single"))
        if scope == "all":
            runs = [
                run
                for run in self.specs
                if self.source_catalog.get(run, {}).get("triage_status")
                != INVALID_WEIGHT_DATA_STATUS
            ]
        elif scope == "selected":
            runs = [str(run).lower() for run in payload.get("runs", [])]
        elif scope == "single":
            runs = [str(payload.get("run", "")).lower()]
        else:
            raise ValueError(f"Unknown campaign scope: {scope}")
        runs = sorted(dict.fromkeys(runs), key=self._run_key)
        unknown = [run for run in runs if run not in self.specs]
        if unknown:
            raise ValueError(f"Unknown runs: {', '.join(unknown)}")
        invalid = [
            run
            for run in runs
            if self.source_catalog.get(run, {}).get("triage_status")
            == INVALID_WEIGHT_DATA_STATUS
        ]
        if invalid:
            raise ValueError(
                "Invalid overflow-weight runs cannot start a review campaign: "
                + ", ".join(run.upper() for run in invalid)
            )
        if not runs:
            raise ValueError("Select at least one run")
        return runs

    def _source_workbook(
        self, run: str, payload: dict[str, Any], selected_runs: list[str]
    ) -> Path | None:
        mapping = payload.get("existing_workbooks") or {}
        if isinstance(mapping, dict) and run in mapping:
            mapped = str(mapping.get(run, "")).strip()
            return Path(mapped) if mapped else None
        if len(selected_runs) == 1 and "existing_workbook" in payload:
            manual = str(payload.get("existing_workbook", "")).strip()
            return Path(manual) if manual else None
        candidates = self.workbook_candidates(run)
        return Path(candidates[0]) if candidates else None

    def _inventory(
        self, run: str, payload: dict[str, Any], selected_runs: list[str]
    ) -> RunInventory:
        spec = self.specs[run]
        injection_start, injection_end = resolved_injection_window(spec)
        max_hours = float(payload.get("max_hours", DEFAULT_SINGLE_MAX_HOURS))
        feed_end_buffer_h = float(payload.get("feed_end_buffer_h", 1.0))
        stop_after_feed = bool(payload.get("stop_after_feed", False)) and bool(
            spec.gravity_schedule_verified
        )
        existing_workbook = self._source_workbook(run, payload, selected_runs)
        run_output = self.output_root / run
        state_path = run_output / f"{run}_overflow_scale_review.json"
        manifest_path = run_output / f"{run}_overflow_image_manifest.json"
        prediction_cache_path = run_output / f"{run}_overflow_predictions.json"
        state_mtime = state_path.stat().st_mtime_ns if state_path.exists() else 0
        manifest_mtime = (
            manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0
        )
        prediction_cache_mtime = (
            prediction_cache_path.stat().st_mtime_ns
            if prediction_cache_path.exists()
            else 0
        )
        workbook_mtime = (
            existing_workbook.stat().st_mtime_ns
            if existing_workbook is not None and existing_workbook.exists()
            else 0
        )
        cache_key = (
            run,
            max_hours,
            stop_after_feed,
            feed_end_buffer_h,
            _normalise_path(existing_workbook),
            workbook_mtime,
            state_mtime,
            manifest_mtime,
            prediction_cache_mtime,
        )
        cached = self.inventory_cache.get(cache_key)
        if cached is not None:
            return cached
        all_frames = _load_image_manifest(manifest_path, spec)
        inventory_source = "persistent manifest"
        if not all_frames:
            inventory_source = "new filesystem scan"
            all_frames = discover_spec_frames(spec)
            if all_frames:
                _write_image_manifest(manifest_path, spec, all_frames)
        if not all_frames:
            raise RuntimeError(
                f"No timestamped scale images found under {spec.image_root}"
            )
        start = min(spec.gravity_start, injection_start - timedelta(minutes=30))
        cutoffs = [injection_start + timedelta(hours=max_hours)]
        cutoff_rules = [f"injection start + {max_hours:g} h"]
        if stop_after_feed:
            cutoffs.append(spec.gravity_end + timedelta(hours=feed_end_buffer_h))
            cutoff_rules.append(f"gravity-feed end + {feed_end_buffer_h:g} h")
        effective_cutoff = min(cutoffs)
        eligible_frames = [
            frame for frame in all_frames if start <= frame.time <= effective_cutoff
        ]
        saved_by_id: dict[str, dict[str, Any]] = {}
        confirmed_count = 0
        reviewed_count = 0
        cached_predictions = _load_prediction_cache(prediction_cache_path, spec)
        prediction_cache_dirty = False
        saved_payload: dict[str, Any] = {}
        if state_path.exists():
            try:
                saved_payload = json.loads(state_path.read_text(encoding="utf-8"))
                saved = saved_payload.get("records", [])
            except (OSError, json.JSONDecodeError):
                saved = []
            for record in saved:
                status = str(record.get("status", ""))
                image_id = str(record.get("image_id", ""))
                if image_id:
                    saved_by_id[image_id] = record
                    state_predictions = list(record.get("predictions") or [])
                    if not state_predictions:
                        migrated = _legacy_prediction(
                            record,
                            _legacy_prediction_model_id(spec),
                            str(
                                saved_payload.get("metadata", {}).get("updated_at")
                                or ""
                            )
                            or None,
                        )
                        if migrated is not None:
                            state_predictions = [migrated]
                    if state_predictions:
                        cached_record = cached_predictions.get(image_id, {})
                        merged = _merge_predictions(
                            list(cached_record.get("predictions") or []),
                            state_predictions,
                        )
                        if merged != list(cached_record.get("predictions") or []):
                            cached_predictions[image_id] = {
                                "image_id": image_id,
                                "image_time": record.get("image_time"),
                                "image": record.get("image"),
                                "relative_path": record.get("relative_path"),
                                "predictions": merged,
                            }
                            prediction_cache_dirty = True
                confirmed_count += status in CONFIRMED_STATUSES
                reviewed_count += status in REVIEWED_STATUSES
        frames, eligible_image_ids = retain_reviewed_frames(
            run,
            spec.image_root,
            eligible_frames,
            list(saved_by_id.values()),
        )
        if prediction_cache_dirty:
            _write_prediction_cache(prediction_cache_path, spec, cached_predictions)
        training_snapshot = prediction_training_snapshot(list(saved_by_id.values()))
        model_id = prediction_model_id(
            spec,
            training_snapshot["id"],
            self.shared_ocr_trainer.model_id,
        )
        capacity = 0
        predicted_count = 0
        for frame in eligible_frames:
            image_id = _frame_id(run, frame, spec.image_root)
            saved_record = saved_by_id.get(image_id, {})
            if str(saved_record.get("status", "")) not in REVIEWED_STATUSES:
                capacity += 1
            cached_record = cached_predictions.get(image_id, {})
            prediction_record = dict(saved_record)
            prediction_record["predictions"] = _merge_predictions(
                list(cached_record.get("predictions") or []),
                list(saved_record.get("predictions") or []),
            )
            if _has_prediction(prediction_record, model_id):
                predicted_count += 1
        inventory = RunInventory(
            run=run,
            frames=frames,
            injection_start=injection_start,
            injection_end=injection_end,
            effective_cutoff=effective_cutoff,
            cutoff_rule="earliest of " + " and ".join(cutoff_rules),
            confirmed_count=confirmed_count,
            reviewed_count=reviewed_count,
            capacity=capacity,
            existing_workbook=existing_workbook,
            inventory_source=inventory_source,
            prediction_model_id=model_id,
            predicted_count=predicted_count,
            missing_prediction_count=len(eligible_frames) - predicted_count,
            eligible_image_ids=eligible_image_ids,
        )
        self.inventory_cache[cache_key] = inventory
        return inventory

    def _prepare_allocation(
        self,
        payload: dict[str, Any],
        progress: Callable[[float, str], None] | None = None,
    ) -> tuple[list[str], dict[str, RunInventory], dict[str, int], str, int]:
        selected_runs = self._selected_runs(payload)
        requested_count = int(payload.get("sample_count", DEFAULT_SINGLE_SAMPLE_COUNT))
        if requested_count <= 0:
            raise ValueError("Number of validation images must be positive")
        max_hours = float(payload.get("max_hours", DEFAULT_SINGLE_MAX_HOURS))
        if max_hours <= 0:
            raise ValueError("Maximum time must be positive")
        strategy = str(payload.get("strategy", "even"))
        inventories: dict[str, RunInventory] = {}
        for position, run in enumerate(selected_runs):
            if progress:
                progress(
                    0.02 + 0.08 * position / max(1, len(selected_runs)),
                    f"Inventory {run.upper()}",
                )
            inventories[run] = self._inventory(run, payload, selected_runs)
        allocations = allocate_review_counts(
            selected_runs,
            requested_count,
            {run: inventories[run].capacity for run in selected_runs},
            {run: inventories[run].confirmed_count for run in selected_runs},
            strategy,
        )
        return selected_runs, inventories, allocations, strategy, requested_count

    def preview_allocation(self, payload: dict[str, Any]) -> dict[str, Any]:
        selected_runs, inventories, allocations, strategy, requested = (
            self._prepare_allocation(payload)
        )
        return {
            "strategy": strategy,
            "requested_count": requested,
            "allocated_count": sum(allocations.values()),
            "runs": [
                {
                    "run": run,
                    "confirmed": inventories[run].confirmed_count,
                    "reviewed": inventories[run].reviewed_count,
                    "capacity": inventories[run].capacity,
                    "allocated": allocations[run],
                    "projected_confirmed": inventories[run].confirmed_count
                    + allocations[run],
                    "inventory_source": inventories[run].inventory_source,
                    "prediction_model_id": inventories[run].prediction_model_id,
                    "predicted": inventories[run].predicted_count,
                    "missing_predictions": inventories[run].missing_prediction_count,
                }
                for run in selected_runs
            ],
        }

    def start_setup(self, payload: dict[str, Any]) -> None:
        with self.lock:
            if self.job["state"] == "building":
                raise RuntimeError("A validation campaign is already being prepared")
            reviewer = str(payload.get("reviewer", "")).strip() or DEFAULT_REVIEWER
            self.campaign_sequence += 1
            campaign_id = (
                f"{datetime.now():%Y%m%dT%H%M%S}_c{self.campaign_sequence:03d}"
            )
            self.job = {
                "state": "building",
                "progress": 0.0,
                "message": "Starting",
                "campaign_id": campaign_id,
            }

        def progress(value: float, message: str) -> None:
            with self.lock:
                self.job.update(
                    {"progress": max(0.0, min(1.0, value)), "message": message}
                )

        def build() -> None:
            try:
                selected_runs, inventories, allocations, strategy, requested_count = (
                    self._prepare_allocation(payload, progress)
                )
                active_runs = [run for run in selected_runs if allocations[run] > 0]
                if not active_runs:
                    raise RuntimeError(
                        "No unreviewed images are available in the selected runs"
                    )
                sessions: dict[str, RunSession] = {}
                for position, run in enumerate(active_runs):

                    def run_progress(
                        value: float,
                        message: str,
                        *,
                        _position: int = position,
                        _run: str = run,
                    ) -> None:
                        progress(
                            0.1 + 0.86 * (_position + value) / len(active_runs),
                            f"{_run.upper()}: {message}",
                        )

                    inventory = inventories[run]
                    source = self.source_catalog.get(run, {})
                    sessions[run] = RunSession(
                        spec=self.specs[run],
                        output_root=self.output_root,
                        bridge=self.bridge,
                        sample_count=allocations[run],
                        max_hours=float(
                            payload.get("max_hours", DEFAULT_SINGLE_MAX_HOURS)
                        ),
                        stop_after_feed=(
                            bool(payload.get("stop_after_feed", False))
                            and self.specs[run].gravity_schedule_verified
                        ),
                        feed_end_buffer_h=float(payload.get("feed_end_buffer_h", 1.0)),
                        existing_workbook=inventory.existing_workbook,
                        reviewer=reviewer,
                        campaign_id=campaign_id,
                        inventory=inventory,
                        progress=run_progress,
                        shared_ocr_trainer=self.shared_ocr_trainer,
                        weight_data_status=str(
                            source.get("triage_status") or "needs-review"
                        ),
                        weight_data_reason=str(source.get("triage_reason") or ""),
                        weight_data_reviewer=str(source.get("triage_reviewer") or ""),
                        weight_data_updated_at=(
                            str(source["triage_updated_at"])
                            if source.get("triage_updated_at")
                            else None
                        ),
                    )
                    self.specs[run] = sessions[run].spec
                campaign = ValidationCampaign(
                    campaign_id=campaign_id,
                    sessions=sessions,
                    selected_runs=selected_runs,
                    requested_count=requested_count,
                    strategy=strategy,
                    allocations=allocations,
                    inventories=inventories,
                )
            except Exception as exc:
                with self.lock:
                    self.job = {
                        "state": "error",
                        "progress": 0.0,
                        "message": str(exc),
                        "campaign_id": campaign_id,
                    }
            else:
                with self.lock:
                    self.campaign = campaign
                    self.job = {
                        "state": "ready",
                        "progress": 1.0,
                        "message": "Ready",
                        "campaign_id": campaign_id,
                    }

        threading.Thread(target=build, daemon=True).start()

    def start_resume(self, run: str) -> None:
        run = run.lower()
        if run not in self.specs:
            raise ValueError(f"Unknown run: {run}")
        with self.lock:
            if self.job["state"] == "building":
                raise RuntimeError("A validation campaign is already being prepared")
            state = self._load_run_payload(run, mutable=True)
            metadata = dict(state.get("metadata") or {})
            round_id = str(metadata.get("current_round_id") or "")
            selected = [
                record
                for record in state.get("records", [])
                if record.get("selected_for_review")
                and str(record.get("review_round_id") or "") == round_id
            ]
            pending = [
                record for record in selected if record.get("status") == "Pending"
            ]
            if not round_id or not pending:
                raise ValueError(f"{run.upper()} has no unfinished validation round")
            review_round = next(
                (
                    item
                    for item in reversed(state.get("review_rounds", []))
                    if str(item.get("id") or "") == round_id
                ),
                {},
            )
            self.job = {
                "state": "building",
                "progress": 0.0,
                "message": f"Resuming {run.upper()}",
                "campaign_id": round_id,
            }

        max_hours = float(
            review_round.get("max_hours")
            or metadata.get("sample_window_h")
            or DEFAULT_SINGLE_MAX_HOURS
        )
        stop_after_feed = bool(review_round.get("stop_after_feed", False))
        feed_end_buffer_h = float(review_round.get("feed_end_buffer_h") or 1.0)
        reviewer = str(review_round.get("reviewer") or DEFAULT_REVIEWER)
        sample_count = int(
            review_round.get("requested_count")
            or metadata.get("sample_count")
            or len(selected)
        )
        setup_payload = {
            "scope": "single",
            "run": run,
            "sample_count": sample_count,
            "max_hours": max_hours,
            "stop_after_feed": stop_after_feed,
            "feed_end_buffer_h": feed_end_buffer_h,
            "existing_workbooks": {run: ""},
        }

        def progress(value: float, message: str) -> None:
            with self.lock:
                self.job.update(
                    {
                        "progress": max(0.0, min(1.0, value)),
                        "message": f"{run.upper()}: {message}",
                    }
                )

        def build() -> None:
            try:
                inventory = self._inventory(run, setup_payload, [run])
                source = self.source_catalog.get(run, {})
                session = RunSession(
                    spec=self.specs[run],
                    output_root=self.output_root,
                    bridge=self.bridge,
                    sample_count=sample_count,
                    max_hours=max_hours,
                    stop_after_feed=(
                        stop_after_feed and self.specs[run].gravity_schedule_verified
                    ),
                    feed_end_buffer_h=feed_end_buffer_h,
                    existing_workbook=None,
                    reviewer=reviewer,
                    campaign_id=round_id,
                    inventory=inventory,
                    progress=progress,
                    shared_ocr_trainer=self.shared_ocr_trainer,
                    weight_data_status=str(
                        source.get("triage_status") or "needs-review"
                    ),
                    weight_data_reason=str(source.get("triage_reason") or ""),
                    weight_data_reviewer=str(source.get("triage_reviewer") or ""),
                    weight_data_updated_at=(
                        str(source["triage_updated_at"])
                        if source.get("triage_updated_at")
                        else None
                    ),
                    resume_existing_round=True,
                )
                if not session.resumed_existing_round:
                    raise RuntimeError(
                        f"The unfinished round for {run.upper()} could not be restored"
                    )
                self.specs[run] = session.spec
                campaign = ValidationCampaign(
                    campaign_id=round_id,
                    sessions={run: session},
                    selected_runs=[run],
                    requested_count=sample_count,
                    strategy="resume",
                    allocations={run: len(session.current_indices())},
                    inventories={run: inventory},
                )
            except Exception as exc:
                with self.lock:
                    self.job = {
                        "state": "error",
                        "progress": 0.0,
                        "message": str(exc),
                        "campaign_id": round_id,
                    }
            else:
                with self.lock:
                    self.campaign = campaign
                    self.overview_cache.pop(run, None)
                    self.run_payload_cache.pop(run, None)
                    self.job = {
                        "state": "ready",
                        "progress": 1.0,
                        "message": "Ready",
                        "campaign_id": round_id,
                    }

        threading.Thread(target=build, daemon=True).start()

    def job_summary(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.job)

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return {"campaign": self.campaign.summary() if self.campaign else None}


def _display_crop_box(
    image_width: int, image_height: int, geometry: CellGeometry
) -> tuple[int, int, int, int]:
    """Include the complete LCD housing, not only the configured OCR cells."""
    left = max(0, min(geometry.x) - 6 * geometry.width)
    top = max(0, geometry.y - int(0.45 * geometry.height))
    right = min(image_width, max(geometry.x) + geometry.width * 3)
    bottom = min(image_height, geometry.y + int(1.45 * geometry.height))
    return left, top, right, bottom


def _automatic_display_crop(
    image: Image.Image,
) -> tuple[tuple[int, int, int, int], bool]:
    """Locate the scale display and report whether image evidence supports it."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    height, width = rgb.shape[:2]
    x0, x1 = int(0.15 * width), int(0.85 * width)
    y0, y1 = int(0.20 * height), int(0.90 * height)
    roi = rgb[y0:y1, x0:x1]
    # Include cyan digits as well as the greener status symbols.
    score = roi[..., 1] - roi[..., 0]
    positive = score[score > 0]
    if positive.size:
        threshold = max(18.0, float(np.percentile(positive, 98.5)))
        mask = (score >= threshold) & (roi[..., 1] >= 55)
    else:
        mask = np.zeros(score.shape, dtype=bool)

    ys, xs = np.where(mask)
    detected = bool(xs.size)
    if detected:
        weights = np.maximum(score[ys, xs], 1).astype(float)
        bins_x, bins_y = 16, 12
        histogram, _, _ = np.histogram2d(
            ys,
            xs,
            bins=(bins_y, bins_x),
            range=((0, roi.shape[0]), (0, roi.shape[1])),
            weights=weights,
        )
        padded = np.pad(histogram, 1)
        density = sum(
            padded[dy : dy + bins_y, dx : dx + bins_x]
            for dy in range(3)
            for dx in range(3)
        )
        bin_y, bin_x = np.unravel_index(int(np.argmax(density)), density.shape)
        center_x = x0 + (bin_x + 0.5) * roi.shape[1] / bins_x
        center_y = y0 + (bin_y + 0.5) * roi.shape[0] / bins_y
    else:
        center_x, center_y = 0.5 * width, 0.58 * height

    crop_width = min(width, max(220, int(0.42 * width)))
    crop_height = min(height, max(120, int(0.28 * height)))
    left = max(0, min(width - crop_width, int(center_x - crop_width / 2)))
    top = max(0, min(height - crop_height, int(center_y - crop_height / 2)))
    return (left, top, left + crop_width, top + crop_height), detected


def _display_review_image(image: Image.Image) -> tuple[Image.Image, bool]:
    """Return a readable LCD crop, repairing a horizontal frame tear if needed."""
    image = image.convert("RGB")
    rgb = np.asarray(image, dtype=np.int16)
    horizontal_roll = (
        _horizontal_display_roll(rgb) if _automatic_screen_box(rgb) is None else None
    )
    if horizontal_roll:
        image = Image.fromarray(np.roll(np.asarray(image), horizontal_roll, axis=1))
    crop_box, detected = _automatic_display_crop(image)
    if detected:
        return image.crop(crop_box), True
    return image, False


def _automatic_display_crop_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Locate the bright scale display for manual review."""
    return _automatic_display_crop(image)[0]


@lru_cache(maxsize=256)
def _jpeg_bytes_cached(
    image_path: str,
    cells: CellGeometry | None,
    cells_by_image_size: tuple[tuple[int, int, CellGeometry], ...],
    locate_display_per_image: bool,
    full: bool,
) -> bytes:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if full:
            image.thumbnail((1200, 850), Image.Resampling.LANCZOS)
        else:
            if locate_display_per_image:
                image, detected = _display_review_image(image)
                if detected:
                    scale = max(1, min(4, 900 // max(1, image.width)))
                else:
                    image.thumbnail((1200, 850), Image.Resampling.LANCZOS)
                    scale = 1
            else:
                geometry = next(
                    (
                        geometry
                        for width, height, geometry in cells_by_image_size
                        if width == image.width and height == image.height
                    ),
                    cells if not cells_by_image_size else None,
                )
                if geometry is not None:
                    image = image.crop(
                        _display_crop_box(image.width, image.height, geometry)
                    )
                    scale = max(4, min(8, 760 // max(1, image.width)))
                else:
                    crop_box, detected = _automatic_display_crop(image)
                    if detected:
                        image = image.crop(crop_box)
                        scale = max(1, min(4, 900 // max(1, image.width)))
                    else:
                        image.thumbnail((1200, 850), Image.Resampling.LANCZOS)
                        scale = 1
            if scale > 1:
                image = image.resize(
                    (image.width * scale, image.height * scale),
                    Image.Resampling.LANCZOS,
                )
            image = ImageEnhance.Contrast(
                ImageOps.autocontrast(image, cutoff=1)
            ).enhance(1.25)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        return buffer.getvalue()


def _jpeg_bytes_for_record(spec: RunSpec, record: dict[str, Any], full: bool) -> bytes:
    return _jpeg_bytes_cached(
        str(record["image"]),
        spec.cells,
        spec.cells_by_image_size,
        spec.automatic_ocr,
        full,
    )


def _jpeg_bytes(session: RunSession, index: int, full: bool) -> bytes:
    return _jpeg_bytes_for_record(session.spec, session.records[index], full)


def _jpeg_bytes_for_source(path: Path) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1280, 960), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
        return buffer.getvalue()


def make_handler(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FFACOverflowReview/5.20"

        def log_message(self, format_string: str, *args: Any) -> None:
            return

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = _json_text(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(
            self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST
        ) -> None:
            self._json({"error": message}, status)

        def _session_record(
            self,
            query: dict[str, list[str]],
            ensure_prediction: bool = False,
        ) -> tuple[RunSession, int]:
            run = query.get("run", [""])[0].lower()
            if app.campaign is None or run not in app.campaign.sessions:
                raise ValueError(f"Run is not loaded: {run}")
            session = app.campaign.sessions[run]
            index = int(query.get("index", ["0"])[0])
            if not 0 <= index < len(session.records):
                raise ValueError(f"Image index out of range: {index}")
            if ensure_prediction:
                session.ensure_prediction_current(index)
            return session, index

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK.value)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/options":
                    self._json(app.options())
                elif parsed.path == "/api/overview":
                    self._json(app.overview())
                elif parsed.path == "/api/source":
                    run = query.get("run", [""])[0].lower()
                    self._json(app.source_detail(run))
                elif parsed.path == "/api/source/job":
                    self._json(app.source_job_summary())
                elif parsed.path == "/api/job":
                    self._json(app.job_summary())
                elif parsed.path == "/api/summary":
                    self._json(app.summary())
                elif parsed.path == "/api/explore/summary":
                    run = query.get("run", [""])[0].lower()
                    self._json({"summary": app.explore_summary(run)})
                elif parsed.path == "/api/explore/record":
                    run = query.get("run", [""])[0].lower()
                    index = int(query.get("index", ["0"])[0])
                    self._json(
                        {
                            "record": app.explore_record(run, index),
                            "summary": app.explore_summary(run),
                        }
                    )
                elif parsed.path == "/api/explore/series":
                    run = query.get("run", [""])[0].lower()
                    self._json({"series": app.explore_series(run)})
                elif parsed.path == "/api/record":
                    session, index = self._session_record(query, ensure_prediction=True)
                    self._json(
                        {"record": session.records[index], "summary": session.summary()}
                    )
                elif parsed.path == "/api/series":
                    run = query.get("run", [""])[0].lower()
                    if app.campaign is None:
                        raise ValueError("No active campaign")
                    self._json({"series": app.campaign.sessions[run].series()})
                elif parsed.path == "/image":
                    session, index = self._session_record(query)
                    body = _jpeg_bytes(
                        session, index, query.get("full", ["0"])[0] == "1"
                    )
                    self.send_response(HTTPStatus.OK.value)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "private, max-age=3600")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/explore/image":
                    run = query.get("run", [""])[0].lower()
                    index = int(query.get("index", ["0"])[0])
                    record = app.explore_record(run, index)
                    body = _jpeg_bytes_for_record(
                        app.specs[run],
                        record,
                        query.get("full", ["0"])[0] == "1",
                    )
                    self.send_response(HTTPStatus.OK.value)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "private, max-age=3600")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/source/image":
                    run = query.get("run", [""])[0].lower()
                    sample = int(query.get("sample", ["0"])[0])
                    body = _jpeg_bytes_for_source(app.source_image_path(run, sample))
                    self.send_response(HTTPStatus.OK.value)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._error("Not found", HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._error(str(exc))

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = (
                    json.loads(self.rfile.read(length).decode("utf-8"))
                    if length
                    else {}
                )
                if parsed.path == "/api/preview":
                    self._json(app.preview_allocation(payload))
                elif parsed.path == "/api/source/refresh":
                    app.start_source_refresh()
                    self._json(app.source_job_summary(), HTTPStatus.ACCEPTED)
                elif parsed.path == "/api/source/status":
                    self._json(
                        app.update_source_status(
                            run=str(payload.get("run", "")),
                            status=str(payload.get("status", "")),
                            reason=str(payload.get("reason", "")),
                            reviewer=str(payload.get("reviewer", "")),
                        )
                    )
                elif parsed.path == "/api/setup":
                    app.start_setup(payload)
                    self._json(app.job_summary(), HTTPStatus.ACCEPTED)
                elif parsed.path == "/api/resume":
                    app.start_resume(str(payload.get("run", "")))
                    self._json(app.job_summary(), HTTPStatus.ACCEPTED)
                elif parsed.path == "/api/save":
                    if app.campaign is None:
                        raise ValueError("No active campaign")
                    run = str(payload["run"]).lower()
                    unreadable = bool(payload.get("unreadable", False))
                    reading = (
                        None
                        if unreadable
                        else payload.get("reading", payload.get("weight_g"))
                    )
                    record = app.campaign.save(
                        run, int(payload["index"]), reading, unreadable
                    )
                    self._json({"record": record, "campaign": app.campaign.summary()})
                elif parsed.path == "/api/explore/save":
                    run = str(payload["run"]).lower()
                    unreadable = bool(payload.get("unreadable", False))
                    reading = (
                        None
                        if unreadable
                        else payload.get("reading", payload.get("weight_g"))
                    )
                    record = app.edit_explore_record(
                        run=run,
                        index=int(payload["index"]),
                        reading=reading,
                        unreadable=unreadable,
                        reviewer=str(payload.get("reviewer", "")),
                        notes=(
                            None
                            if "notes" not in payload
                            else str(payload.get("notes", ""))
                        ),
                    )
                    self._json(
                        {
                            "record": record,
                            "summary": app.explore_summary(run),
                        }
                    )
                elif parsed.path == "/api/undo":
                    if app.campaign is None:
                        raise ValueError("No active campaign")
                    undone = app.campaign.undo()
                    self._json({"undone": undone, "campaign": app.campaign.summary()})
                else:
                    self._error("Not found", HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._error(str(exc))

    return Handler


_LEGACY_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FluidFlower Scale Image Reader</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#202124;background:#f3f5f5;letter-spacing:0}
*{box-sizing:border-box}body{margin:0;min-width:320px}button,input,select{font:inherit}
header{height:58px;background:#fff;border-bottom:1px solid #d4d9d7;display:flex;align-items:center;padding:0 18px;gap:16px}
h1{font-size:17px;margin:0}.spacer{flex:1}.secondary{height:34px;border:1px solid #bcc4c1;background:#fff;border-radius:4px;padding:0 12px;cursor:pointer}
#setup{max-width:860px;margin:30px auto;background:#fff;border:1px solid #d4d9d7;padding:24px}.setup-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.wide{grid-column:1/-1}
label{display:block;font-size:12px;color:#58605d;margin-bottom:5px}input,select{width:100%;height:40px;border:1px solid #aeb8b4;border-radius:4px;padding:0 10px;background:#fff}.check{display:flex;align-items:center;gap:8px;height:40px}.check input{width:18px;height:18px}.check label{margin:0}
.primary{height:42px;border:0;border-radius:4px;background:#176b52;color:#fff;padding:0 18px;font-weight:600;cursor:pointer}.primary:disabled{background:#8ca69d}.job{margin-top:18px}.progress{height:7px;background:#e2e6e4;margin-top:7px}.bar{height:100%;background:#176b52;width:0}.error{color:#a1261d;font-size:13px;min-height:20px;margin-top:8px}
#workspace{display:none;grid-template-columns:minmax(500px,1fr) minmax(340px,430px);min-height:calc(100vh - 58px)}.review{padding:22px 26px;border-right:1px solid #d4d9d7}.side{padding:22px;background:#fff}.meta{display:flex;justify-content:space-between;gap:20px;margin-bottom:14px;color:#606763;font-size:13px}.meta strong{color:#202124}
.display{height:340px;background:#171918;display:flex;align-items:center;justify-content:center;overflow:hidden;border-radius:4px}.display img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}.entry{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:end;margin-top:18px}.entry input{height:48px;font-size:25px;font-variant-numeric:tabular-nums}.entry .primary{height:48px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:#d8dddb;margin-top:18px;border:1px solid #d8dddb}.metric{background:#fff;padding:12px;min-width:0}.metric span{display:block;font-size:11px;color:#6c7370}.metric strong{display:block;font-size:17px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.confidence.high{color:#176b52}.confidence.medium{color:#9b5700}.confidence.low{color:#ad281f}.run-status tr.invalid-weight-data{background:#fff1f0;color:#92271f}
.nav{display:flex;gap:8px;margin-top:18px}.nav button{height:38px;border:1px solid #c4cbc8;background:#fff;border-radius:4px;padding:0 14px;cursor:pointer}.nav .unreadable{margin-left:auto;color:#98271f}.counts{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;font-size:12px;margin-bottom:20px}.counts div{border-bottom:1px solid #e3e7e5;padding:8px 0}.counts strong{float:right}.plot{width:100%;height:260px;border:1px solid #dfe4e2;background:#fff}h2{font-size:14px;margin:22px 0 8px}.source,.sync{font-size:12px;color:#606763;line-height:1.45;overflow-wrap:anywhere}.sync.ok{color:#176b52}.sync.bad{color:#a1261d}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);align-items:center;justify-content:center;z-index:20}#modal img{max-width:96vw;max-height:92vh}#modal.open{display:flex}
@media(max-width:900px){.setup-grid{grid-template-columns:1fr}.wide{grid-column:auto}#workspace{display:none;grid-template-columns:1fr}.review{border-right:0}.side{border-top:1px solid #d4d9d7}.display{height:260px}.metrics{grid-template-columns:repeat(2,1fr)}}
.source-sample{margin:0}@media(max-width:600px){#sourceModal{padding:0}.source-panel{width:100vw;max-height:100vh;border:0;padding:14px}.source-samples,.source-form{grid-template-columns:1fr}.source-form .wide{grid-column:auto}}
</style></head><body>
<header><h1>FF AC Overflow</h1><strong id="activeRun"></strong><div class="spacer"></div><button id="newRound" class="secondary" style="display:none">Ny valideringsrunde</button></header>
<section id="setup"><div class="setup-grid">
<div><label for="runSelect">Forsøk</label><select id="runSelect"></select></div>
<div><label for="sampleCount">Nye bilder som skal valideres</label><input id="sampleCount" type="number" min="1" value="110"></div>
<div class="wide"><label for="workbookSelect">Eksisterende Excel-fil</label><select id="workbookSelect"></select></div>
<div class="wide"><label for="workbookPath">Annen Excel-fil</label><input id="workbookPath" placeholder="Valgfri full filbane"></div>
<div><label for="maxHours">Maksimal tid etter injeksjonsstart (h)</label><input id="maxHours" type="number" min="0.1" step="0.5" value="15"></div>
<div><label for="bufferHours">Tid etter planlagt overflow-slutt (h)</label><input id="bufferHours" type="number" min="0" step="0.5" value="1"></div>
<div><label for="reviewer">Kontrollert av</label><input id="reviewer"></div>
<div class="check"><input id="stopAfterFeed" type="checkbox"><label for="stopAfterFeed">Stopp etter planlagt overflow-slutt</label></div>
<div><button id="start" class="primary">Start valideringsrunde</button></div>
</div><div class="job"><span id="jobText"></span><div class="progress"><div id="jobBar" class="bar"></div></div><div id="setupError" class="error"></div></div></section>
<main id="workspace"><section class="review">
<div class="meta"><div><strong id="phase">-</strong><br><span id="clock">-</span></div><div style="text-align:right"><strong id="position">-</strong><br><span id="imageName">-</span></div></div>
<div class="display"><img id="scaleImage" alt="Vektens display"></div>
<div class="entry"><div><label for="weight">Vekt (g)</label><input id="weight" inputmode="decimal" autocomplete="off"></div><button id="confirm" class="primary">Bekreft</button></div>
<div class="metrics"><div class="metric"><span>OCR-forslag</span><strong id="suggestion">-</strong></div><div class="metric"><span>Tidsprior (støtte)</span><strong id="expected">-</strong></div><div class="metric"><span>Eksisterende verdi</span><strong id="existing">-</strong></div><div class="metric"><span>Sikkerhet</span><strong id="confidence" class="confidence">-</strong></div><div class="metric"><span>Modellrelease</span><strong id="predictionModel">-</strong></div><div class="metric"><span>Modellgrunnlag</span><strong id="trainingCount">-</strong></div></div>
<div class="nav"><button id="previous" title="Forrige bilde">&larr;</button><button id="next" title="Neste bilde">&rarr;</button><button id="unreadable" class="unreadable">Uleselig bilde</button><button id="invalidateRun" class="invalidate-run" title="Stopp validering av hele forsøket">Ugyldig forsøk</button></div><div id="reviewError" class="error"></div>
</section><aside class="side"><div id="progressText"></div><div class="progress"><div id="reviewBar" class="bar"></div></div><div id="counts" class="counts"></div><canvas id="plot" class="plot"></canvas><h2>Arbeidsbok</h2><div id="sync" class="sync"></div><h2>Overflow</h2><div id="source" class="source"></div></aside></main>
<div id="modal"><img id="fullImage" alt="Komplett kamerabilde"></div>
<script>
let options=null,summary=null,run=null,index=0,current=null;
const $=id=>document.getElementById(id);const n=(v,d=1)=>v==null?'-':Number(v).toFixed(d);
async function api(url,settings){const r=await fetch(url,settings),p=await r.json();if(!r.ok||p.error)throw new Error(p.error||r.statusText);return p}
async function loadOptions(){options=await api('/api/options');$('runSelect').innerHTML='';options.runs.forEach(item=>{const o=document.createElement('option');o.value=item.run;o.textContent=item.run.toUpperCase();$('runSelect').appendChild(o)});$('reviewer').value=options.default_reviewer;updateWorkbooks()}
function selectedOption(){return options.runs.find(item=>item.run===$('runSelect').value)}
function updateWorkbooks(){const item=selectedOption();$('workbookSelect').innerHTML='<option value="">Ingen</option>';item.workbooks.forEach(path=>{const o=document.createElement('option');o.value=path;o.textContent=path;$('workbookSelect').appendChild(o)});$('workbookSelect').selectedIndex=0}
async function startRound(){try{$('setupError').textContent='';$('start').disabled=true;const body={run:$('runSelect').value,sample_count:Number($('sampleCount').value),existing_workbook:$('workbookPath').value.trim()||$('workbookSelect').value,max_hours:Number($('maxHours').value),feed_end_buffer_h:Number($('bufferHours').value),stop_after_feed:$('stopAfterFeed').checked,reviewer:$('reviewer').value};await api('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});pollJob()}catch(e){$('setupError').textContent=e.message;$('start').disabled=false}}
async function pollJob(){try{const job=await api('/api/job');$('jobText').textContent=job.message||'';$('jobBar').style.width=(100*(job.progress||0))+'%';if(job.state==='building'){setTimeout(pollJob,700);return}if(job.state==='error'){throw new Error(job.message)}if(job.state==='ready'){run=job.run;await openWorkspace();return}}catch(e){$('setupError').textContent=e.message;$('start').disabled=false}}
async function openWorkspace(){const all=await api('/api/summary');summary=all.runs.find(item=>item.run===run);index=summary.next_pending;$('setup').style.display='none';$('workspace').style.display='grid';$('newRound').style.display='block';$('activeRun').textContent=run.toUpperCase();$('start').disabled=false;await loadRecord()}
async function refreshSummary(){const all=await api('/api/summary');summary=all.runs.find(item=>item.run===run);renderSummary()}
function renderSummary(){const done=summary.current_done,total=summary.current_total,pct=total?100*done/total:0;$('progressText').textContent=`${done} av ${total} kontrollert i denne runden`;$('reviewBar').style.width=pct+'%';$('counts').innerHTML=`<div>Bekreftet<strong>${summary.current_counts.Verified||0}</strong></div><div>Korrigert<strong>${summary.current_counts.Corrected||0}</strong></div><div>Uleselig<strong>${summary.current_counts.Unreadable||0}</strong></div><div>Gjenstår<strong>${summary.current_counts.Pending||0}</strong></div><div>Alle bilder<strong>${summary.total_images}</strong></div><div>Tidligere kontrollert<strong>${(summary.counts.Verified||0)+(summary.counts.Corrected||0)+(summary.counts.Unreadable||0)-done}</strong></div>`;$('source').textContent=`${n(summary.gravity_feed_g_per_h,0)} g/h. ${summary.gravity_start.replace('T',' ')} til ${summary.gravity_end.replace('T',' ')}. Avgrensning: ${summary.effective_cutoff.replace('T',' ')} (${summary.cutoff_rule}).` ;const wb=summary.workbook;$('sync').textContent=(wb.state==='current'&&!wb.pending?'Oppdatert: ':'Lagrer: ')+wb.path+(wb.error?' | '+wb.error:'');$('sync').className='sync '+(wb.error?'bad':'ok')}
async function loadRecord(){try{$('reviewError').textContent='';current=await api(`/api/record?run=${run}&index=${index}`);summary=current.summary;const rec=current.record,indices=summary.current_indices,pos=Math.max(0,indices.indexOf(index));$('phase').textContent=`t = ${n(rec.time_from_injection_h,2)} h`;$('clock').textContent=rec.image_time.replace('T',' ');$('position').textContent=`${pos+1} / ${indices.length}`;$('imageName').textContent=rec.image_filename;$('suggestion').textContent=n(rec.suggested_g);$('expected').textContent=n(rec.expected_g);$('existing').textContent=n(rec.existing_value_g);$('existing').title=rec.existing_source||'';$('confidence').textContent=rec.confidence;$('confidence').className='confidence '+rec.confidence;$('weight').value=rec.weight_g!=null?n(rec.weight_g):n(rec.suggested_g);$('scaleImage').src=`/image?run=${run}&index=${index}&v=${Date.now()}`;$('fullImage').src=`/image?run=${run}&index=${index}&full=1&v=${Date.now()}`;renderSummary();drawPlot()}catch(e){$('reviewError').textContent=e.message}}
function move(delta){const ids=summary.current_indices,pos=Math.max(0,ids.indexOf(index)),next=Math.max(0,Math.min(ids.length-1,pos+delta));index=ids[next];loadRecord()}
async function save(unreadable=false){try{const body={run,index,unreadable};if(!unreadable){body.weight_g=Number($('weight').value.replace(',','.'));if(!Number.isFinite(body.weight_g))throw new Error('Ugyldig vekt')}const response=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});summary=response.summary;if(summary.pending_indices.length){const later=summary.pending_indices.find(id=>id>index);index=later==null?summary.pending_indices[0]:later}await loadRecord()}catch(e){$('reviewError').textContent=e.message}}
async function drawPlot(){const payload=await api(`/api/series?run=${run}`),c=$('plot'),ctx=c.getContext('2d'),rect=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=rect.width*dpr;c.height=rect.height*dpr;ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,pad=34,all=payload.series;if(!all.length)return;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#dfe3e1';ctx.strokeRect(pad,10,w-pad-10,h-pad-10);const xmin=Math.min(...all.map(x=>x.h)),xmax=Math.max(...all.map(x=>x.h)),values=all.flatMap(x=>[x.weight_g,x.existing_value_g].filter(v=>v!=null)),ymax=Math.max(1,...values)*1.05;function xy(p,v){return [pad+(p.h-xmin)/(xmax-xmin||1)*(w-pad-10),h-pad-v/ymax*(h-pad-20)]}ctx.strokeStyle='#aeb7b3';ctx.lineWidth=1;ctx.beginPath();all.forEach((p,i)=>{const [x,y]=xy(p,p.suggested_g||0);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();const verified=all.filter(p=>p.weight_g!=null);ctx.strokeStyle='#176b52';ctx.lineWidth=1.8;ctx.beginPath();verified.forEach((p,i)=>{const [x,y]=xy(p,p.weight_g);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();verified.forEach(p=>{const [x,y]=xy(p,p.weight_g);ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fillStyle='#176b52';ctx.fill()});ctx.fillStyle='#59615e';ctx.font='11px Segoe UI';ctx.fillText(`${n(xmin,1)} h`,pad,h-8);ctx.fillText(`${n(xmax,1)} h`,w-50,h-8);ctx.fillText(`${n(ymax,0)} g`,2,20)}
$('runSelect').onchange=updateWorkbooks;$('start').onclick=startRound;$('newRound').onclick=()=>{$('setup').style.display='block';$('workspace').style.display='none';$('newRound').style.display='none';$('runSelect').value=run;updateWorkbooks()};$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('confirm').onclick=()=>save(false);$('unreadable').onclick=()=>save(true);$('scaleImage').onclick=()=>$('modal').classList.add('open');$('modal').onclick=()=>$('modal').classList.remove('open');$('weight').onkeydown=e=>{if(e.key==='Enter')save(false)};document.addEventListener('keydown',e=>{if(e.target===$('weight'))return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});loadOptions();setInterval(()=>{if(run&&summary)refreshSummary()},3000);
</script></body></html>"""


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FluidFlower Scale Image Reader</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#202124;background:#f3f5f5;letter-spacing:0}
*{box-sizing:border-box}body{margin:0;min-width:320px}button,input,select{font:inherit}button{cursor:pointer}
header{height:58px;background:#fff;border-bottom:1px solid #d4d9d7;display:flex;align-items:center;padding:0 18px;gap:14px}
h1{font-size:17px;margin:0}.app-version{color:#737b78;font-size:11px;font-weight:600}.spacer{flex:1}.secondary,.icon-button{height:34px;border:1px solid #bcc4c1;background:#fff;border-radius:4px;padding:0 12px}.secondary.danger{border-color:#b94a43;color:#92271f;background:#fff1f0;font-weight:700}.secondary.danger:hover{background:#f9deda}.icon-button{width:38px;padding:0;font-size:20px}.icon-button:disabled{color:#aeb5b2;background:#f4f5f5;cursor:default}
#dashboard{max-width:1420px;margin:28px auto 48px;padding:0 20px}.dashboard-head{display:flex;align-items:flex-end;gap:20px;margin-bottom:18px}.dashboard-head h2{font-size:22px;margin:0 0 4px}.dashboard-head p{margin:0;color:#626a67;font-size:13px}.dashboard-actions{margin-left:auto;display:flex;gap:8px}.overview-stats{display:grid;grid-template-columns:repeat(7,1fr);border-top:1px solid #d5dad8;border-bottom:1px solid #d5dad8;background:#fff;margin-bottom:18px}.overview-stat{padding:15px 16px;border-right:1px solid #e0e4e2}.overview-stat:last-child{border-right:0}.overview-stat span{display:block;color:#69716e;font-size:11px}.overview-stat strong{display:block;font-size:21px;font-variant-numeric:tabular-nums;margin-top:2px}.overview-stat.invalid strong{color:#92271f}.campaign-banner{display:flex;align-items:center;gap:14px;background:#e8f1ed;border-left:4px solid #176b52;padding:11px 14px;margin-bottom:18px;font-size:13px}.campaign-banner .spacer{flex:1}.overview-table-wrap{background:#fff;border:1px solid #d4d9d7;overflow:auto}.overview-table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}.overview-table th{position:sticky;top:0;background:#edf2f0;text-align:right;color:#3f4945;font-size:11px;font-weight:600;padding:9px 12px;border-bottom:1px solid #ccd3d0;white-space:nowrap}.overview-table th:first-child,.overview-table td:first-child{text-align:left}.overview-table td{text-align:right;padding:10px 12px;border-bottom:1px solid #e4e8e6}.overview-table tbody tr[data-open]{cursor:pointer}.overview-table tbody tr[data-open]:hover,.overview-table tbody tr[data-open]:focus{background:#eef5f2;outline:none}.overview-table tbody tr.invalid-weight-data{background:#fff1f0}.overview-table tbody tr:last-child td{border-bottom:0}.overview-table .run-link{border:0;background:none;color:#125e48;font-weight:700;padding:0}.overview-table .muted{color:#909794}.coverage{display:flex;align-items:center;justify-content:flex-end;gap:8px;min-width:120px}.coverage-track{display:inline-block;width:62px;height:5px;background:#e1e5e3}.coverage-fill{display:block;height:100%;background:#176b52}.empty-state{padding:24px;color:#69716e;text-align:center}.source-status,.weight-status-action{height:28px;border:1px solid #b9c1be;background:#fff;border-radius:3px;padding:0 8px;font-size:11px;white-space:nowrap}.weight-status-action{border-color:#c6908c;color:#92271f;background:#fff}.weight-status-action:hover,.weight-status-action.invalid{background:#fff1f0;font-weight:700}.source-status.ready{border-color:#6d9e8d;color:#125e48;background:#edf6f2}.source-status.partially-readable{border-color:#c49b52;color:#7b520c;background:#fff8e9}.source-status.invalid-weight-data,.source-status.unreadable,.source-status.no-images,.source-status.time-mismatch{border-color:#c6908c;color:#92271f;background:#fff1f0;font-weight:700}.source-status.partial-coverage{border-color:#c49b52;color:#7b520c;background:#fff8e9}.source-status.needs-review,.source-status.needs-configuration,.source-status.not-scanned{color:#4f5955;background:#f4f6f5}.run-choice.invalid-weight-data{border-color:#c6908c;background:#fff1f0;color:#92271f}.invalid-badge{display:inline-block;margin-left:6px;color:#92271f;font-size:10px;font-weight:700}.nav .invalidate-run{border-color:#c6908c;color:#92271f;background:#fff1f0}
#setup{display:none;max-width:980px;margin:30px auto;background:#fff;border:1px solid #d4d9d7;padding:24px}.setup-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.wide{grid-column:1/-1}.review-mode-note{border-left:3px solid #c98219;background:#fff8e9;color:#60440f;padding:10px 12px;font-size:12px;line-height:1.45}
label{display:block;font-size:12px;color:#58605d;margin-bottom:5px}input,select{width:100%;height:40px;border:1px solid #aeb8b4;border-radius:4px;padding:0 10px;background:#fff}
.segmented{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;border:1px solid #aeb8b4;border-radius:4px;overflow:hidden}.segmented label{margin:0}.segmented input{position:absolute;opacity:0;pointer-events:none;width:1px;height:1px}.segmented span{display:flex;height:40px;align-items:center;justify-content:center;border-right:1px solid #cbd1cf;background:#fff;color:#3f4744;font-size:13px}.segmented label:last-child span{border-right:0}.segmented input:checked+span{background:#dceae5;color:#124d3b;font-weight:600}
.check{display:flex;align-items:center;gap:8px;height:40px}.check input{width:18px;height:18px}.check label{margin:0}.run-picker{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid #d5dad8}.run-choice{display:grid;grid-template-columns:22px 1fr auto;align-items:center;gap:7px;margin:0;padding:9px 10px;border-right:1px solid #e1e4e3;border-bottom:1px solid #e1e4e3}.run-choice input{width:17px;height:17px}.run-choice small{color:#69716e;font-variant-numeric:tabular-nums}
.primary{height:42px;border:0;border-radius:4px;background:#176b52;color:#fff;padding:0 18px;font-weight:600}.primary:disabled{background:#8ca69d}.actions{display:flex;gap:10px;align-items:center}.job{margin-top:18px}.progress{height:7px;background:#e2e6e4;margin-top:7px}.bar{height:100%;background:#176b52;width:0}.error{color:#a1261d;font-size:13px;min-height:20px;margin-top:8px}.error.success{color:#176b52}
.allocation{margin-top:4px;border:1px solid #d7dcda;overflow:auto}.data-table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}.data-table th{text-align:left;background:#edf2f0;color:#3f4945;font-weight:600}.data-table th,.data-table td{padding:7px 9px;border-bottom:1px solid #e2e6e4}.data-table td:not(:first-child),.data-table th:not(:first-child){text-align:right}
#workspace{display:none;grid-template-columns:minmax(520px,1fr) minmax(390px,470px);min-height:calc(100vh - 58px)}.review{padding:22px 26px;border-right:1px solid #d4d9d7}.side{padding:20px;background:#fff}.meta{display:flex;justify-content:space-between;gap:20px;margin-bottom:14px;color:#606763;font-size:13px}.meta strong{color:#202124}.run-badge{display:inline-block;background:#dceae5;color:#124d3b;padding:3px 7px;border-radius:3px;margin-right:7px;font-weight:700}
.display{height:340px;background:#171918;display:flex;align-items:center;justify-content:center;overflow:hidden;border-radius:4px}.display img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}.entry{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:end;margin-top:18px}.entry input{height:48px;font-size:25px;font-variant-numeric:tabular-nums}.entry .primary{height:48px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:#d8dddb;margin-top:18px;border:1px solid #d8dddb}.metric{background:#fff;padding:12px;min-width:0}.metric span{display:block;font-size:11px;color:#6c7370}.metric strong{display:block;font-size:17px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.metric .stored-value{color:#176b52;font-weight:700}.confidence.high{color:#176b52}.confidence.medium{color:#9b5700}.confidence.low{color:#ad281f}
.nav{display:flex;gap:8px;margin-top:18px}.nav button{height:38px;border:1px solid #c4cbc8;background:#fff;border-radius:4px;padding:0 14px}.nav .unreadable{margin-left:auto;color:#98271f}.counts{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;font-size:12px;margin-bottom:14px}.counts div{border-bottom:1px solid #e3e7e5;padding:7px 0}.counts strong{float:right}.plot-toolbar{display:flex;align-items:center;gap:10px;margin:7px 0}.plot-toggle{display:flex;align-items:center;gap:6px;margin:0;font-size:11px;color:#505956}.plot-toggle input{width:16px;height:16px}.plot-reset{height:28px;margin-left:auto;border:1px solid #c4cbc8;background:#fff;border-radius:4px;padding:0 9px;font-size:11px}.plot-wrap{position:relative}.plot{display:block;width:100%;height:300px;border:1px solid #dfe4e2;background:#fff;cursor:crosshair;touch-action:none}.plot-tooltip{position:absolute;z-index:2;pointer-events:none;max-width:210px;padding:5px 7px;background:#202523;color:#fff;border-radius:3px;font-size:11px;line-height:1.35;white-space:nowrap}.plot-hint{margin-top:5px;color:#757d79;font-size:10px}.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#5d6562;margin:7px 0}.swatch{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:4px}.run-status{max-height:180px;overflow:auto;border:1px solid #e0e4e2;margin:12px 0}.run-status .data-table th,.run-status .data-table td{padding:6px 7px}h2{font-size:14px;margin:18px 0 8px}.source,.sync{font-size:12px;color:#606763;line-height:1.45;overflow-wrap:anywhere}.sync.ok{color:#176b52}.sync.bad{color:#a1261d}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);align-items:center;justify-content:center;z-index:20}#modal img{max-width:96vw;max-height:92vh}#modal.open{display:flex}#sourceModal{display:none;position:fixed;inset:0;background:rgba(25,29,27,.72);z-index:25;align-items:center;justify-content:center;padding:24px}#sourceModal.open{display:flex}.source-panel{width:min(1040px,96vw);max-height:92vh;overflow:auto;background:#fff;border:1px solid #b9c1be;padding:20px}.source-panel-head{display:flex;align-items:flex-start;gap:16px;margin-bottom:12px}.source-panel-head h2{font-size:20px;margin:0 0 3px}.source-panel-head p{margin:0;color:#68706d;font-size:12px}.source-close{margin-left:auto}.source-meta{background:#f2f5f4;border-left:3px solid #66877b;padding:10px 12px;font-size:12px;line-height:1.5;overflow-wrap:anywhere}.source-samples{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:14px 0}.source-sample{border:1px solid #d5dad8;background:#171918;min-width:0}.source-sample img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain}.source-sample span{display:block;background:#fff;padding:7px 9px;font-size:11px;color:#4f5955}.source-form{display:grid;grid-template-columns:240px 1fr;gap:12px}.source-form textarea{width:100%;min-height:76px;border:1px solid #aeb8b4;border-radius:4px;padding:8px;font:inherit;resize:vertical}.source-form .wide{grid-column:1/-1}.source-form-actions{display:flex;justify-content:flex-end;gap:8px}.hidden{display:none!important}
@media(max-width:900px){#dashboard{margin:18px auto;padding:0 14px}.overview-stats{grid-template-columns:repeat(3,1fr)}.overview-stat:nth-child(3){border-right:0}.overview-stat:nth-child(n+4){border-top:1px solid #e0e4e2}.dashboard-head{align-items:flex-start}.dashboard-actions{margin-left:auto}#setup{margin:15px}.setup-grid{grid-template-columns:1fr}.wide{grid-column:auto}.run-picker{grid-template-columns:repeat(2,1fr)}#workspace{grid-template-columns:1fr}.review{border-right:0}.side{border-top:1px solid #d4d9d7}.display{height:260px}.metrics{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){header{padding:0 10px;gap:8px}h1{font-size:15px}#activeRun{font-size:12px}#newRound{width:38px;padding:0;font-size:0}#newRound::after{content:'+';font-size:22px}#home{width:38px;padding:0;font-size:0}#home::after{content:'\2302';font-size:20px}.dashboard-head{display:block}.dashboard-actions{margin-top:12px}.overview-stats{grid-template-columns:repeat(2,1fr)}.overview-stat{border-top:1px solid #e0e4e2}.overview-stat:nth-child(odd){border-right:1px solid #e0e4e2}.overview-stat:nth-child(even){border-right:0}.campaign-banner{align-items:flex-start;flex-wrap:wrap}.campaign-banner .spacer{display:none}#setup{margin:0;border-left:0;border-right:0;padding:16px}.segmented span{padding:0 4px;text-align:center;font-size:12px}.run-picker{grid-template-columns:1fr}.review,.side{padding:16px}}
</style></head><body>
<header><h1>Scale Image Reader</h1><span class="app-version">v5.20</span><strong id="activeRun"></strong><div class="spacer"></div><button id="invalidateRunHeader" class="secondary danger hidden" title="Stop the review and mark the entire run as unusable for scale readings">Mark run invalid</button><button id="home" class="secondary hidden" title="Return to the overview">Overview</button><button id="undo" class="icon-button hidden" title="Undo the latest entry" disabled>&#8630;</button><button id="newRound" class="secondary" title="Start a new review campaign">New review campaign</button></header>
<section id="dashboard"><div class="dashboard-head"><div><h2>Overflow readings</h2><p>Runs with time-aligned scale images can be reviewed manually; configured OCR is used when available.</p></div><div class="dashboard-actions"><button id="refreshSources" class="secondary">Scan image sources</button><button id="refreshOverview" class="secondary">Refresh</button><button id="dashboardNewRound" class="primary">New review</button></div></div>
<div id="campaignBanner" class="campaign-banner hidden"><strong>Active review campaign</strong><span id="campaignBannerText"></span><div class="spacer"></div><button id="resumeCampaign" class="secondary">Continue</button></div>
<div id="overviewStats" class="overview-stats"></div><div id="overviewTable" class="overview-table-wrap"><div class="empty-state">Loading overview...</div></div><div id="overviewError" class="error"></div></section>
<section id="setup"><div class="setup-grid">
<div class="wide"><label>Scope</label><div class="segmented"><label><input type="radio" name="scope" value="single" checked><span>One run</span></label><label><input type="radio" name="scope" value="selected"><span>Selected runs</span></label><label><input type="radio" name="scope" value="all"><span>All runs</span></label></div></div>
<div id="singleRun"><label for="runSelect">Run</label><select id="runSelect"></select></div>
<div><label for="sampleCount">Total number of new images</label><input id="sampleCount" type="number" min="1" value="110"></div>
<div id="runModeNote" class="wide review-mode-note hidden"></div>
<div id="selectedRuns" class="wide hidden"><label>Included runs</label><div id="runPicker" class="run-picker"></div></div>
<div id="strategyBlock" class="wide hidden"><label>Allocation</label><div class="segmented"><label><input type="radio" name="strategy" value="even" checked><span>Even new sample</span></label><label><input type="radio" name="strategy" value="balance"><span>Balance confirmed readings</span></label></div></div>
<div id="workbookBlock" class="wide"><label for="workbookSelect">Existing Excel workbook</label><select id="workbookSelect"></select></div>
<div id="workbookPathBlock" class="wide"><label for="workbookPath">Other Excel workbook</label><input id="workbookPath" placeholder="Optional full file path"></div>
<div><label for="maxHours">Maximum time after injection start (h)</label><input id="maxHours" type="number" min="0.1" step="0.5" value="15"></div>
<div><label for="bufferHours">Time after scheduled overflow end (h)</label><input id="bufferHours" type="number" min="0" step="0.5" value="1"></div>
<div><label for="reviewer">Reviewed by</label><input id="reviewer"></div>
<div class="check"><input id="stopAfterFeed" type="checkbox"><label for="stopAfterFeed">Stop after scheduled overflow end</label></div>
<div class="wide actions"><button id="preview" class="secondary">Calculate allocation</button><button id="start" class="primary">Start review campaign</button></div>
<div id="allocationPreview" class="wide allocation hidden"></div>
</div><div class="job"><span id="jobText"></span><div class="progress"><div id="jobBar" class="bar"></div></div><div id="setupError" class="error"></div></div></section>
<main id="workspace"><section class="review">
<div class="meta"><div><span id="runBadge" class="run-badge">-</span><strong id="phase">-</strong><br><span id="clock">-</span></div><div style="text-align:right"><strong id="position">-</strong><br><span id="imageName">-</span></div></div>
<div class="display"><img id="scaleImage" alt="Scale display"></div>
<div class="entry"><div><label for="weight">Reading (g)</label><input id="weight" inputmode="text" autocomplete="off" spellcheck="false" title="A leading x represents 0-3 digits; an internal x represents one digit"></div><button id="confirm" class="primary">Confirm</button></div>
<div class="metrics"><div class="metric"><span>Original OCR proposal</span><strong id="suggestion">-</strong></div><div class="metric"><span>Temporal prior (support)</span><strong id="expected">-</strong></div><div class="metric"><span>Saved manual value</span><strong id="existing">-</strong></div><div class="metric"><span>Confidence</span><strong id="confidence" class="confidence">-</strong></div><div class="metric"><span>Model release</span><strong id="predictionModel">-</strong></div><div class="metric"><span>Model basis</span><strong id="trainingCount">-</strong></div></div>
<div class="nav"><button id="previous" title="Previous image">&larr;</button><button id="next" title="Next image">&rarr;</button><button id="unreadable" class="unreadable">Unreadable image</button><button id="invalidateRun" class="invalidate-run" title="Stop reviewing the entire run">Invalid run</button></div><div id="reviewError" class="error"></div>
</section><aside class="side"><div id="progressText"></div><div class="progress"><div id="reviewBar" class="bar"></div></div><div id="counts" class="counts"></div><div id="runStatus" class="run-status"></div>
<div class="legend"><span><i class="swatch" style="background:#aeb7b3"></i>OCR</span><span><i class="swatch" style="background:#c98219"></i>Imported</span><span><i class="swatch" style="background:#176b52"></i>Confirmed</span><span><i class="swatch" style="background:#7755a6"></i>Uncertain (x)</span><span><i class="swatch" style="background:#b3261e"></i>Current image</span></div>
<div class="plot-toolbar"><label class="plot-toggle"><input id="showPartial" type="checkbox" checked>Show uncertain readings</label><button id="resetPlotZoom" class="plot-reset" title="Show the full time axis">Reset zoom</button></div>
<div id="plotWrap" class="plot-wrap"><canvas id="plot" class="plot"></canvas><div id="plotTooltip" class="plot-tooltip hidden"></div></div><div class="plot-hint">Scroll to zoom. Drag across a time range to zoom in. Click a point to open its image. Double-click to reset.</div>
<h2>Workbook</h2><div id="sync" class="sync"></div><h2>Overflow</h2><div id="source" class="source"></div></aside></main>
<div id="sourceModal"><section class="source-panel" role="dialog" aria-modal="true" aria-labelledby="sourceTitle"><div class="source-panel-head"><div><h2 id="sourceTitle">Image source</h2><p>Representative images are used only to assess the source. They are not automatically accepted as readings.</p></div><button id="sourceClose" class="secondary source-close">Close</button></div><div id="sourceMeta" class="source-meta"></div><div id="sourceSamples" class="source-samples"></div><div class="source-form"><div><label for="sourceStatus">Source status</label><select id="sourceStatus"></select></div><div><label for="sourceReviewer">Reviewed by</label><input id="sourceReviewer"></div><div class="wide"><label for="sourceReason">Reason / required action</label><textarea id="sourceReason"></textarea></div><div class="wide source-form-actions"><button id="sourceSave" class="primary">Save source status</button></div></div><div id="sourceError" class="error"></div></section></div>
<div id="modal"><img id="fullImage" alt="Full camera image"></div>
<script>
let uiMode='overview',options=null,overview=null,campaign=null,exploreSummary=null,run=null,index=0,current=null,plotSeries=[],plotGeometry=null,plotHits=[],plotDrag=null,sourceRun=null,fullImageUrl='',previousScope=null;
const plotViews={},plotSeriesByRun={},prefetchedImageUrls=new Set();
const $=id=>document.getElementById(id),n=(v,d=1)=>v==null?'-':Number(v).toFixed(d),key=item=>`${item.run}:${item.index}`;
async function api(url,settings){const response=await fetch(url,settings),payload=await response.json();if(!response.ok||payload.error)throw new Error(payload.error||response.statusText);return payload}
function scopeValue(){return document.querySelector('input[name="scope"]:checked').value}function strategyValue(){return document.querySelector('input[name="strategy"]:checked').value}
async function loadOptions(){options=await api('/api/options');$('runSelect').innerHTML='';$('runPicker').innerHTML='';options.runs.forEach(item=>{const mode=item.automatic_ocr?'auto-OCR':item.review_mode==='ocr'?'OCR':'manual',invalid=item.weight_data_invalid;const option=document.createElement('option');option.value=item.run;option.disabled=invalid;option.textContent=`${item.run.toUpperCase()} (${invalid?'invalid weight data':item.confirmed+' confirmed, '+mode})`;$('runSelect').appendChild(option);const label=document.createElement('label');label.className='run-choice'+(invalid?' invalid-weight-data':'');label.title=invalid?(item.source_reason||'Invalid weight data'):'';label.innerHTML=`<input type="checkbox" value="${item.run}" ${invalid?'disabled':'checked'}><strong>${item.run.toUpperCase()}</strong><small>${invalid?'Invalid':item.confirmed+' | '+mode}</small>`;$('runPicker').appendChild(label)});$('reviewer').value=options.default_reviewer;updateWorkbooks();updateScope()}
function dateLabel(value){return value?String(value).replace('T',' ').slice(0,19):'-'}
async function showSetup(selectedRun=null){uiMode='setup';$('dashboard').style.display='none';$('setup').style.display='block';$('workspace').style.display='none';$('activeRun').textContent='New review';$('invalidateRunHeader').classList.add('hidden');$('home').classList.remove('hidden');$('newRound').classList.add('hidden');$('undo').classList.add('hidden');$('setupError').textContent='';await loadOptions();if(selectedRun&&options.runs.some(item=>item.run===selectedRun)){$('runSelect').value=selectedRun;document.querySelector('input[name="scope"][value="single"]').checked=true;updateScope();updateWorkbooks()}}
async function loadOverview(){try{uiMode='overview';$('overviewError').textContent='';$('dashboard').style.display='block';$('setup').style.display='none';$('workspace').style.display='none';$('activeRun').textContent='';$('invalidateRunHeader').classList.add('hidden');$('home').classList.add('hidden');$('newRound').classList.remove('hidden');$('undo').classList.add('hidden');overview=await api('/api/overview');renderOverview()}catch(error){$('overviewError').textContent=error.message}}
const sourceStatusLabels={'not-scanned':'Not scanned','needs-review':'Needs review',ready:'Ready','needs-configuration':'Needs configuration','partially-readable':'Partially readable','invalid-weight-data':'Invalid weight data',unreadable:'Unreadable','no-images':'No images','time-mismatch':'Time mismatch','partial-coverage':'Partial coverage'};
function sourceStatusLabel(value){return sourceStatusLabels[value]||value||'-'}
function html(value){return String(value??'').replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[character])}
function renderOverview(){const totals=overview.totals;$('overviewStats').innerHTML=`<div class="overview-stat"><span>Image sources found</span><strong>${totals.with_source_images} / ${totals.runs}</strong></div><div class="overview-stat"><span>Available for review</span><strong>${totals.review_available}</strong></div><div class="overview-stat"><span>With OCR</span><strong>${totals.with_ocr??totals.configured}</strong></div><div class="overview-stat"><span>Exact readings</span><strong>${totals.exact}</strong></div><div class="overview-stat"><span>Uncertain readings</span><strong>${totals.partial}</strong></div><div class="overview-stat"><span>Total reviewed</span><strong>${totals.reviewed}</strong></div><div class="overview-stat invalid"><span>Invalid weight data</span><strong>${totals.invalid_weight_data||0}</strong></div>`;campaign=overview.campaign;const banner=$('campaignBanner');banner.classList.toggle('hidden',!campaign);if(campaign)$('campaignBannerText').textContent=campaign.skipped_invalid?`${campaign.reviewed_done} reviewed, ${campaign.skipped_invalid} skipped as invalid.`:`${campaign.done} of ${campaign.total} images reviewed across ${campaign.runs.length} runs.`;const rows=overview.runs.map(item=>{const coverage=item.total?item.review_percent:0,action=item.has_data?'explore':(item.review_available?'setup':''),open=action?`data-open="${action}" tabindex="0"`:'',runCell=action?`<button class="run-link" data-run="${item.run}">${item.run.toUpperCase()}</button>`:`<strong class="muted">${item.run.toUpperCase()}</strong>`,sourceCount=item.source_scanned?`${item.source_relevant_images??0} / ${item.source_images??0}`:'-',reason=item.source_reason||item.source_root||'',status=html(item.source_status),mode=item.automatic_ocr?'Auto-OCR':item.review_mode==='ocr'?'OCR':item.review_mode==='manual'?'Manual':'Not available',rowClass=item.weight_data_invalid?'class="invalid-weight-data"':'',weightAction=item.weight_data_invalid?`<button class="weight-status-action invalid" data-invalid-run="${item.run}" data-invalid-current="1" title="${html(reason)}">Invalid</button>`:`<button class="weight-status-action" data-invalid-run="${item.run}" title="Mark the entire run as unusable for scale readings">Mark invalid</button>`;return`<tr ${rowClass} ${open} data-run="${item.run}"><td>${runCell}</td><td><button class="source-status ${status}" data-source-run="${item.run}" title="${html(reason)}">${html(sourceStatusLabel(item.source_status))}</button></td><td>${weightAction}</td><td>${sourceCount}</td><td>${mode}</td><td>${item.exact}</td><td>${item.partial}</td><td>${item.unreadable}</td><td>${item.reviewed} / ${item.total}</td><td><div class="coverage"><span>${n(coverage,1)} %</span><i class="coverage-track"><i class="coverage-fill" style="width:${Math.max(0,Math.min(100,coverage))}%"></i></i></div></td><td>${dateLabel(item.updated_at)}</td></tr>`}).join('');$('overviewTable').innerHTML=rows?`<table class="overview-table"><thead><tr><th>Run</th><th>Source status</th><th>Weight data</th><th>Relevant / total</th><th>Reading mode</th><th>Exact</th><th>Uncertain</th><th>Unreadable</th><th>Reviewed / images</th><th>Coverage</th><th>Last updated</th></tr></thead><tbody>${rows}</tbody></table>`:'<div class="empty-state">No production runs found.</div>'}
async function openSource(selectedRun,statusOverride=null){try{sourceRun=selectedRun;$('sourceError').textContent='';const item=await api(`/api/source?run=${selectedRun}`),option=options?.runs.find(candidate=>candidate.run===selectedRun);$('sourceTitle').textContent=`${selectedRun.toUpperCase()} weight data`;$('sourceStatus').innerHTML=item.triage_statuses.map(status=>`<option value="${status}">${sourceStatusLabel(status)}</option>`).join('');$('sourceStatus').value=statusOverride||item.triage_status;$('sourceReason').value=statusOverride==='invalid-weight-data'?'':(item.triage_reason||'');$('sourceReviewer').value=item.triage_reviewer||options?.default_reviewer||'';const coverage=item.coverage_start_h==null?'-':`${n(item.coverage_start_h,1)} to ${n(item.coverage_end_h,1)} h`,mode=option?.automatic_ocr?'automatic OCR with adaptive proposals':option?.review_mode==='ocr'?'OCR with proposals':option?.review_mode==='manual'?'automatic capture during the first review':'not available',offset=Number(item.image_time_offset_h||0),clockNote=offset?` Camera time adjusted ${offset>0?'+':''}${n(offset,1)} h against the protocol.`:'';$('sourceMeta').innerHTML=`<strong>${html(item.selected_root||'No source found')}</strong><br>${item.relevant_image_count??0} relevant of ${item.image_count??0} timestamped images. Protocol coverage: ${coverage}.${clockNote} Reading mode: ${mode}. Select "Invalid weight data" when reflections or other persistent defects make the run unusable.`;$('sourceSamples').innerHTML=item.samples.length?item.samples.map(sample=>`<figure class="source-sample"><img src="${sample.url}&v=${Date.now()}" alt="Representative scale image"><span>${dateLabel(sample.image_time)} | t=${n(sample.time_from_injection_h,2)} h</span></figure>`).join(''):'<div class="empty-state">No representative images are available.</div>';$('sourceModal').classList.add('open');if(statusOverride)$('sourceReason').focus()}catch(error){$('overviewError').textContent=error.message}}
async function saveSourceStatus(){try{$('sourceError').textContent='';const status=$('sourceStatus').value,reason=$('sourceReason').value.trim(),wasCampaign=uiMode==='campaign';if(status==='invalid-weight-data'&&!reason)throw new Error('Explain why the weight data are invalid.');const result=await api('/api/source/status',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:sourceRun,status,reason,reviewer:$('sourceReviewer').value})});$('sourceModal').classList.remove('open');if(wasCampaign){campaign=result.campaign;Object.keys(plotSeriesByRun).forEach(key=>delete plotSeriesByRun[key]);const target=campaign?.pending_queue?.[0];if(target){run=target.run;index=target.index;await loadRecord()}else await loadOverview()}else await loadOverview()}catch(error){$('sourceError').textContent=error.message}}
async function pollSourceRefresh(){try{const job=await api('/api/source/job');$('overviewError').textContent=job.message||'';if(job.state==='running'){setTimeout(pollSourceRefresh,1000);return}$('refreshSources').disabled=false;if(job.state==='error')throw new Error(job.message);await loadOverview()}catch(error){$('refreshSources').disabled=false;$('overviewError').textContent=error.message}}
async function refreshSources(){try{$('refreshSources').disabled=true;$('overviewError').textContent='Scanning image sources in the background...';await api('/api/source/refresh',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});pollSourceRefresh()}catch(error){$('refreshSources').disabled=false;$('overviewError').textContent=error.message}}
async function openExplorer(selectedRun){try{uiMode='explore';run=selectedRun;current=null;plotSeries=[];$('overviewError').textContent='';const [summaryPayload,seriesPayload]=await Promise.all([api(`/api/explore/summary?run=${run}`),api(`/api/explore/series?run=${run}`)]);exploreSummary=summaryPayload.summary;plotSeries=seriesPayload.series;const pending=plotSeries.filter(point=>point.status==='Pending').sort((a,b)=>a.h-b.h),reviewed=plotSeries.filter(point=>['Verified','Corrected','Partial','Unreadable'].includes(point.status)).sort((a,b)=>a.h-b.h);index=(pending[0]||reviewed[0]||plotSeries[0]||{index:0}).index;$('dashboard').style.display='none';$('setup').style.display='none';$('workspace').style.display='grid';$('activeRun').textContent=run.toUpperCase();$('invalidateRunHeader').classList.remove('hidden');$('home').classList.remove('hidden');$('newRound').classList.remove('hidden');$('undo').classList.add('hidden');await loadRecord(true)}catch(error){$('overviewError').textContent=error.message;await loadOverview()}}
function selectedOption(){return options.runs.find(item=>item.run===$('runSelect').value)}
function updateWorkbooks(){const item=selectedOption();$('workbookSelect').innerHTML='<option value="">None</option>';if(!item)return;item.workbooks.forEach(path=>{const option=document.createElement('option');option.value=path;option.textContent=path;$('workbookSelect').appendChild(option)});$('workbookSelect').selectedIndex=0;const manual=item.review_mode==='manual',automatic=item.automatic_ocr;$('runModeNote').classList.toggle('hidden',!manual&&!automatic);$('runModeNote').textContent=manual?'The first round captures the scale display automatically. If the capture is not sufficiently reliable, the run continues as a manual review without OCR. The overflow schedule must still be verified before the data can be used as phase ground truth.':automatic?'Automatic OCR is active. New confirmations continuously produce improved, versioned proposals. The overflow schedule must still be verified before the data can be used as phase ground truth.':'';if(scopeValue()==='single'){$('stopAfterFeed').disabled=!item.gravity_schedule_verified;if(!item.gravity_schedule_verified)$('stopAfterFeed').checked=false}}
function updateScope(){const scope=scopeValue(),multi=scope!=='single';if(scope!==previousScope&&scope==='single'){$('sampleCount').value='110';$('maxHours').value='15';$('stopAfterFeed').checked=false}previousScope=scope;$('singleRun').classList.toggle('hidden',multi);$('selectedRuns').classList.toggle('hidden',scope!=='selected');$('strategyBlock').classList.toggle('hidden',!multi);$('workbookBlock').classList.toggle('hidden',multi);$('workbookPathBlock').classList.toggle('hidden',multi);$('allocationPreview').classList.add('hidden');if(multi){$('runModeNote').classList.add('hidden');$('stopAfterFeed').disabled=false}else updateWorkbooks()}
function selectedRuns(){return [...document.querySelectorAll('#runPicker input:checked')].map(input=>input.value)}
function requestBody(){const scope=scopeValue(),runs=scope==='selected'?selectedRuns():[];return{scope,run:$('runSelect').value,runs,strategy:scope==='single'?'even':strategyValue(),sample_count:Number($('sampleCount').value),existing_workbook:scope==='single'?($('workbookPath').value.trim()||$('workbookSelect').value):'',existing_workbooks:scope==='single'?undefined:Object.fromEntries(runs.map(run=>[run,''])),max_hours:Number($('maxHours').value),feed_end_buffer_h:Number($('bufferHours').value),stop_after_feed:$('stopAfterFeed').checked,reviewer:$('reviewer').value}}
function allocationTable(preview){const rows=preview.runs.map(item=>`<tr><td>${item.run.toUpperCase()}</td><td>${item.confirmed}</td><td>${item.capacity}</td><td>${item.inventory_source==='persistent manifest'?'Saved':'New scan'}</td><td><strong>${item.allocated}</strong></td><td>max ${item.allocated}</td><td>${item.projected_confirmed}</td></tr>`).join('');return`<table class="data-table"><thead><tr><th>Run</th><th>Confirmed</th><th>Available</th><th>Image inventory</th><th>Allocated</th><th>Images to read</th><th>After campaign</th></tr></thead><tbody>${rows}</tbody></table>`}
async function previewAllocation(){try{$('setupError').textContent='';$('preview').disabled=true;$('jobText').textContent='Calculating allocation';const result=await api('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requestBody())});$('allocationPreview').innerHTML=allocationTable(result);$('allocationPreview').classList.remove('hidden');$('jobText').textContent=`${result.allocated_count} of ${result.requested_count} images can be allocated`}catch(error){$('setupError').textContent=error.message}finally{$('preview').disabled=false}}
async function startCampaign(){try{$('setupError').textContent='';$('start').disabled=true;$('preview').disabled=true;await api('/api/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requestBody())});pollJob()}catch(error){$('setupError').textContent=error.message;$('start').disabled=false;$('preview').disabled=false}}
async function pollJob(){try{const job=await api('/api/job');$('jobText').textContent=job.message||'';$('jobBar').style.width=(100*(job.progress||0))+'%';if(job.state==='building'){setTimeout(pollJob,700);return}if(job.state==='error')throw new Error(job.message);if(job.state==='ready'){await openWorkspace()}}catch(error){$('setupError').textContent=error.message;$('start').disabled=false;$('preview').disabled=false}}
async function openWorkspace(){const state=await api('/api/summary');campaign=state.campaign;if(!campaign)throw new Error('No active review campaign');const target=campaign.pending_queue[0]||campaign.queue[0];if(!target)throw new Error('The review campaign contains no images');uiMode='campaign';run=target.run;index=target.index;$('dashboard').style.display='none';$('setup').style.display='none';$('workspace').style.display='grid';$('invalidateRunHeader').classList.remove('hidden');$('home').classList.remove('hidden');$('newRound').classList.remove('hidden');$('undo').classList.remove('hidden');$('start').disabled=false;$('preview').disabled=false;await loadRecord()}
async function refreshCampaign(){if(uiMode!=='campaign')return;const payload=await api('/api/summary');if(payload.campaign){campaign=payload.campaign;renderCampaign()}}
async function refreshExplorer(){if(uiMode!=='explore'||!run)return;const payload=await api(`/api/explore/summary?run=${run}`);exploreSummary=payload.summary;renderCampaign()}
function currentRunSummary(){return uiMode==='explore'?exploreSummary:campaign?.run_summaries?.[run]}
function statusLabel(status){return({Verified:'Confirmed',Corrected:'Corrected',Partial:'Uncertain',Unreadable:'Unreadable',Pending:'Pending','Not selected':'Not selected'})[status]||status||'-'}
function suggestionTitle(record){const parts=[`Raw image-only choice: ${n(record.prediction_image_only_g)} g.`];if(record.prediction_physical_bounds_changed_choice)parts.push(`Confirmed readings constrained the interval to ${n(record.prediction_physical_min_g)}-${n(record.prediction_physical_max_g)} g; the optical choice within the interval was ${n(record.prediction_bounded_image_g)} g.`);if(record.prediction_prior_changed_choice)parts.push('The temporal prior selected a nearby OCR alternative within the interval.');return parts.join(' ')}
function renderCampaign(){if(uiMode==='explore'){const summary=exploreSummary,total=summary.total_images,done=summary.reviewed,pct=total?100*done/total:0,counts=summary.counts;$('progressText').textContent=`${done} of ${total} images reviewed`;$('reviewBar').style.width=pct+'%';$('counts').innerHTML=`<div>Exact<strong>${summary.exact}</strong></div><div>Uncertain<strong>${counts.Partial||0}</strong></div><div>Unreadable<strong>${counts.Unreadable||0}</strong></div><div>Remaining<strong>${Math.max(0,total-done)}</strong></div>`;$('runStatus').innerHTML=`<table class="data-table"><thead><tr><th>Status</th><th>Count</th></tr></thead><tbody><tr><td>Confirmed</td><td>${counts.Verified||0}</td></tr><tr><td>Corrected</td><td>${counts.Corrected||0}</td></tr><tr><td>Uncertain</td><td>${counts.Partial||0}</td></tr><tr><td>Unreadable</td><td>${counts.Unreadable||0}</td></tr></tbody></table>`;const workbook=summary.workbook;$('sync').textContent=(workbook.state==='current'&&!workbook.pending?'Up to date: ':'Saving: ')+workbook.path+(workbook.error?' | '+workbook.error:'');$('sync').className='sync '+(workbook.error?'bad':'ok');$('source').textContent=`${n(summary.gravity_feed_g_per_h,0)} g/h. ${dateLabel(summary.gravity_start)} to ${dateLabel(summary.gravity_end)}.`;return}const total=campaign.total,done=campaign.done,pct=total?100*done/total:0;$('progressText').textContent=`${done} of ${total} reviewed in the campaign`;$('reviewBar').style.width=pct+'%';const counts=campaign.status_counts;$('counts').innerHTML=`<div>Confirmed<strong>${counts.Verified||0}</strong></div><div>Corrected<strong>${counts.Corrected||0}</strong></div><div>Partial<strong>${counts.Partial||0}</strong></div><div>Unreadable<strong>${counts.Unreadable||0}</strong></div><div>Remaining<strong>${counts.Pending||0}</strong></div>`;$('undo').disabled=!campaign.can_undo;const rows=campaign.runs.map(item=>`<tr><td>${item.run.toUpperCase()}</td><td>${item.confirmed_before}</td><td>${item.allocated}</td><td>${item.done}</td><td>${item.confirmed_now}</td></tr>`).join('');$('runStatus').innerHTML=`<table class="data-table"><thead><tr><th>Run</th><th>Before</th><th>Selected</th><th>Done</th><th>Now</th></tr></thead><tbody>${rows}</tbody></table>`;const summary=currentRunSummary();if(summary){const workbook=summary.workbook;$('sync').textContent=(workbook.state==='current'&&!workbook.pending?'Up to date: ':'Saving: ')+workbook.path+(workbook.error?' | '+workbook.error:'');$('sync').className='sync '+(workbook.error?'bad':'ok');$('source').textContent=`${n(summary.gravity_feed_g_per_h,0)} g/h. ${dateLabel(summary.gravity_start)} to ${dateLabel(summary.gravity_end)}. Cutoff: ${dateLabel(summary.effective_cutoff)}.`}}
const renderCampaignBase=renderCampaign;renderCampaign=function(){renderCampaignBase();if(uiMode==='campaign'&&campaign){if(campaign.skipped_invalid){$('progressText').textContent=`${campaign.reviewed_done} reviewed, ${campaign.skipped_invalid} skipped as invalid, ${campaign.pending_queue.length} remaining`;$('counts').insertAdjacentHTML('beforeend',`<div>Skipped<strong>${campaign.skipped_invalid}</strong></div>`)}const rows=$('runStatus').querySelectorAll('tbody tr');campaign.runs.forEach((item,position)=>{if(!item.invalid||!rows[position])return;rows[position].classList.add('invalid-weight-data');rows[position].title=item.invalid_reason||'Invalid weight data';rows[position].cells[0].insertAdjacentHTML('beforeend','<span class="invalid-badge">INVALID</span>')})}const summary=currentRunSummary();if(!summary)return;if(summary.gravity_schedule_verified===false){$('source').textContent='Manual image reading. The overflow schedule is not verified; the data cannot yet be used as phase ground truth.'}}
async function loadRecord(seriesReady=false){
  try{
    $('reviewError').textContent='';
    $('reviewError').classList.remove('success');
    const explorer=uiMode==='explore';
    const endpoint=explorer?`/api/explore/record?run=${run}&index=${index}`:`/api/record?run=${run}&index=${index}`;
    current=await api(endpoint);
    if(explorer)exploreSummary=current.summary;
    const record=current.record;
    const position=explorer?-1:campaign.queue.findIndex(item=>key(item)===`${run}:${index}`);
    const inCampaign=!explorer&&position>=0;
    $('activeRun').textContent=explorer?run.toUpperCase():(campaign.runs.length>1?`${campaign.runs.length} runs`:run.toUpperCase());
    $('runBadge').textContent=run.toUpperCase();
    $('invalidateRunHeader').textContent=`Mark ${run.toUpperCase()} invalid`;
    $('invalidateRunHeader').classList.remove('hidden');
    $('phase').textContent=`t = ${n(record.time_from_injection_h,2)} h | ${statusLabel(record.status)}`;
    $('clock').textContent=dateLabel(record.image_time);
    $('position').textContent=explorer?`Image ${index+1} / ${exploreSummary.total_images}`:(inCampaign?`${position+1} / ${campaign.queue.length}`:'Historical point');
    $('imageName').textContent=record.image_filename;
    $('suggestion').textContent=n(record.suggested_g);
    $('suggestion').title=suggestionTitle(record)+' This is the model\'s original proposal and does not change when a manual value is saved.';
    $('expected').textContent=n(record.expected_g);
    const stored=record.reading_pattern||(record.weight_g!=null?n(record.weight_g):n(record.existing_value_g));
    const storedDisplay=explorer?stored:n(record.existing_value_g);
    $('existing').textContent=storedDisplay;
    $('existing').classList.toggle('stored-value',storedDisplay!==''&&storedDisplay!=='-');
    $('existing').title=explorer?`${statusLabel(record.status)}${record.reviewer?' | '+record.reviewer:''}`:(record.existing_source||'');
    $('confidence').textContent=record.confidence||'-';
    $('confidence').className='confidence '+(record.confidence||'');
    $('predictionModel').textContent=record.prediction_model_id||'-';
    $('predictionModel').title=record.prediction_model_id||'';
    const basis=`${record.prediction_training_count??0} local anchors / ${record.shared_ocr_training_count??0} shared readings from ${record.shared_ocr_training_runs??0} runs`;
    $('trainingCount').textContent=basis;
    $('trainingCount').title=basis;
    const proposed=record.suggested_g==null?'':n(record.suggested_g);
    $('weight').value=record.reading_pattern||(record.weight_g!=null?n(record.weight_g):proposed);
    const explorerHasPending=explorer&&(exploreSummary?.counts?.Pending||0)>0;
    $('confirm').textContent=explorer?(explorerHasPending?'Save and continue':'Save change'):'Confirm';
    $('confirm').disabled=explorer?false:!inCampaign;
    $('unreadable').disabled=explorer?false:!inCampaign;
    const imageEndpoint=explorer?'/explore/image':'/image';
    const imageVersion=encodeURIComponent(`${record.image_id||record.image_time}:${record.ocr_extractor_id||''}`);
    $('scaleImage').src=`${imageEndpoint}?run=${run}&index=${index}&v=${imageVersion}`;
    fullImageUrl=`${imageEndpoint}?run=${run}&index=${index}&full=1&v=${imageVersion}`;
    $('fullImage').removeAttribute('src');
    renderCampaign();
    if(seriesReady)renderPlot();
    else if(plotSeriesByRun[run]){plotSeries=plotSeriesByRun[run];renderPlot()}
    else await drawPlot();
  }catch(error){
    $('reviewError').textContent=error.message;
  }
}
function move(delta){if(uiMode==='explore'){const candidates=plotSeries.filter(point=>['Verified','Corrected','Partial','Unreadable','Pending'].includes(point.status)).sort((a,b)=>a.h-b.h);if(!candidates.length)return;const position=candidates.findIndex(item=>item.index===index),next=Math.max(0,Math.min(candidates.length-1,(position<0?0:position)+delta));index=candidates[next].index;loadRecord(true);return}if(!campaign.queue.length)return;const position=campaign.queue.findIndex(item=>key(item)===`${run}:${index}`);let target;if(position>=0){const next=Math.max(0,Math.min(campaign.queue.length-1,position+delta));target=campaign.queue[next]}else{const currentH=current?.record?.time_from_injection_h??0,candidates=campaign.queue.filter(item=>item.run===run).map(item=>({item,h:plotSeries.find(point=>point.index===item.index)?.h})).filter(item=>item.h!=null).sort((a,b)=>a.h-b.h);target=(delta>0?candidates.find(item=>item.h>currentH):[...candidates].reverse().find(item=>item.h<currentH))?.item||(delta>0?candidates[0]?.item:candidates.at(-1)?.item)||campaign.queue[0]}run=target.run;index=target.index;loadRecord()}
function nextPending(){if(!campaign.pending_queue.length)return null;const currentPosition=campaign.queue.findIndex(item=>key(item)===`${run}:${index}`);return campaign.pending_queue.find(item=>campaign.queue.findIndex(candidate=>key(candidate)===key(item))>currentPosition)||campaign.pending_queue[0]}
async function save(unreadable=false){
  try{
    const body={run,index,unreadable};
    if(!unreadable){
      body.reading=$('weight').value.trim();
      if(!body.reading)throw new Error('Reading is required');
    }
    const savedRun=run;
    const savedIndex=index;
    const savedH=current?.record?.time_from_injection_h??-Infinity;
    if(uiMode==='explore'){
      body.reviewer=$('reviewer').value;
      const response=await api('/api/explore/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      exploreSummary=response.summary;
      const seriesPayload=await api(`/api/explore/series?run=${savedRun}`);
      plotSeries=seriesPayload.series;
      plotSeriesByRun[savedRun]=plotSeries;
      const pending=plotSeries.filter(point=>point.status==='Pending'&&point.index!==savedIndex).sort((a,b)=>a.h-b.h);
      const target=pending.find(point=>point.h>savedH)||pending[0];
      if(target)index=target.index;
      await loadRecord(true);
      const savedValue=response.record.reading_pattern||(response.record.weight_g!=null?n(response.record.weight_g):'');
      const advanceText=target?' Advanced to the next image.':' No more selected images remain.';
      $('reviewError').textContent=unreadable?`${savedRun.toUpperCase()}: The image was saved as unreadable.${advanceText}`:`${savedRun.toUpperCase()}: Saved manual value ${savedValue} g. The OCR proposal is retained separately.${advanceText}`;
      $('reviewError').classList.add('success');
      return;
    }
    const response=await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    campaign=response.campaign;
    Object.keys(plotSeriesByRun).forEach(key=>delete plotSeriesByRun[key]);
    const target=nextPending();
    if(target){run=target.run;index=target.index}
    await loadRecord();
    const savedValue=response.record.reading_pattern||(response.record.weight_g!=null?n(response.record.weight_g):'');
    $('reviewError').textContent=unreadable?`${savedRun.toUpperCase()}: The image was saved as unreadable.`:`${savedRun.toUpperCase()}: Saved manual value ${savedValue} g. The OCR proposal is retained separately.`;
    $('reviewError').classList.add('success');
  }catch(error){
    $('reviewError').classList.remove('success');
    $('reviewError').textContent=error.message;
  }
}
async function undoLast(){try{const response=await api('/api/undo',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});campaign=response.campaign;Object.keys(plotSeriesByRun).forEach(key=>delete plotSeriesByRun[key]);run=response.undone.run;index=response.undone.index;await loadRecord()}catch(error){$('reviewError').textContent=error.message}}
function renderPlot(){const all=plotSeries,canvas=$('plot'),context=canvas.getContext('2d'),rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=Math.max(1,rect.width*dpr);canvas.height=Math.max(1,rect.height*dpr);context.scale(dpr,dpr);const width=rect.width,height=rect.height,padL=48,padR=12,padT=16,padB=34;if(!all.length){context.clearRect(0,0,width,height);return}const fullMin=Math.min(...all.map(point=>point.h)),fullMax=Math.max(...all.map(point=>point.h)),fullSpan=fullMax-fullMin||1,saved=plotViews[run],xmin=Math.max(fullMin,Math.min(saved?.xmin??fullMin,fullMax)),xmax=Math.min(fullMax,Math.max(saved?.xmax??fullMax,xmin+fullSpan/500)),visible=all.filter(point=>point.h>=xmin&&point.h<=xmax),showPartial=$('showPartial').checked,partial=showPartial?visible.filter(point=>point.status==='Partial'&&point.reading_pattern):[],rawValues=visible.flatMap(point=>[point.suggested_g,point.existing_value_g,point.weight_g].filter(value=>value!=null)).concat(partial.map(point=>point.suggested_g??point.existing_value_g).filter(value=>value!=null)).sort((a,b)=>a-b);if(!rawValues.length)rawValues.push(0,1);const zoomed=xmax-xmin<fullSpan*.98,low=rawValues[Math.floor(.01*Math.max(0,rawValues.length-1))]??0,high=rawValues[Math.floor(.99*Math.max(0,rawValues.length-1))]??1,spread=Math.max(1,high-low),ymin=zoomed?Math.max(0,low-spread*.1):0,ymax=Math.max(ymin+1,high+spread*(zoomed?.1:.08)),x=value=>padL+(value-xmin)/(xmax-xmin||1)*(width-padL-padR),y=value=>padT+(1-(value-ymin)/(ymax-ymin))*(height-padT-padB);plotGeometry={run,fullMin,fullMax,xmin,xmax,ymin,ymax,width,height,padL,padR,padT,padB,x,y};plotHits=[];context.clearRect(0,0,width,height);context.strokeStyle='#e3e7e5';context.fillStyle='#5f6764';context.font='10px Segoe UI';for(let tick=0;tick<=4;tick++){const ratio=tick/4,yy=padT+ratio*(height-padT-padB),value=ymax-ratio*(ymax-ymin);context.beginPath();context.moveTo(padL,yy);context.lineTo(width-padR,yy);context.stroke();context.fillText(n(value,zoomed&&spread<20?1:0),2,yy+3)}for(let tick=0;tick<=4;tick++){const ratio=tick/4,value=xmin+ratio*(xmax-xmin),xx=x(value),label=`${n(value,xmax-xmin<2?2:1)} h`;context.fillStyle='#5f6764';context.fillText(label,Math.max(2,Math.min(width-42,xx-14)),height-9)}function line(points,valueKey,color,lineWidth,dash=[]){const usable=points.filter(point=>point[valueKey]!=null);if(!usable.length)return;context.save();context.beginPath();context.rect(padL,padT,width-padL-padR,height-padT-padB);context.clip();context.strokeStyle=color;context.lineWidth=lineWidth;context.setLineDash(dash);context.beginPath();usable.forEach((point,i)=>{const xx=x(point.h),yy=y(point[valueKey]);i?context.lineTo(xx,yy):context.moveTo(xx,yy)});context.stroke();context.restore()}line(visible,'suggested_g','#aeb7b3',1);line(visible,'existing_value_g','#c98219',1.2);const confirmed=visible.filter(point=>point.weight_g!=null);line(confirmed,'weight_g','#176b52',2);confirmed.forEach(point=>{const px=x(point.h),py=y(point.weight_g);context.beginPath();context.arc(px,py,3,0,Math.PI*2);context.fillStyle='#176b52';context.fill();plotHits.push({px,py,index:point.index,label:`${point.image_time.replace('T',' ')} | ${n(point.weight_g)} g | ${point.status}`})});partial.forEach(point=>{const value=point.suggested_g??point.existing_value_g;if(value==null)return;const px=x(point.h),py=y(value);context.save();context.translate(px,py);context.rotate(Math.PI/4);context.fillStyle='#fff';context.strokeStyle='#7755a6';context.lineWidth=2;context.fillRect(-4,-4,8,8);context.strokeRect(-4,-4,8,8);context.restore();context.fillStyle='#7755a6';context.fillText(point.reading_pattern,Math.min(width-padR-38,px+7),Math.max(padT+10,py-7));plotHits.push({px,py,index:point.index,label:`${point.image_time.replace('T',' ')} | ${point.reading_pattern} | estimate ${n(value)} g`})});function vertical(value,color,label,dash){if(value<xmin||value>xmax)return;context.save();context.strokeStyle=color;context.setLineDash(dash);context.beginPath();context.moveTo(x(value),padT);context.lineTo(x(value),height-padB);context.stroke();context.fillStyle=color;context.fillText(label,Math.min(width-55,x(value)+3),padT+10);context.restore()}const summary=currentRunSummary();vertical(0,'#4777a8','Start',[4,3]);vertical(summary.injection_end_h,'#4777a8','Injection end',[4,3]);vertical(summary.gravity_end_h,'#c98219','Overflow end',[4,3]);vertical(current.record.time_from_injection_h,'#b3261e','Current',[]);if(plotDrag&&plotDrag.run===run){const left=Math.min(plotDrag.startX,plotDrag.currentX),right=Math.max(plotDrag.startX,plotDrag.currentX);context.fillStyle='rgba(71,119,168,.16)';context.fillRect(left,padT,right-left,height-padT-padB);context.strokeStyle='#4777a8';context.strokeRect(left,padT,right-left,height-padT-padB)}}
async function drawPlot(){const requestedRun=run,endpoint=uiMode==='explore'?'/api/explore/series':'/api/series',payload=await api(`${endpoint}?run=${requestedRun}`);plotSeriesByRun[requestedRun]=payload.series;if(requestedRun!==run)return;plotSeries=payload.series;renderPlot()}
function plotEventPoint(event){const rect=$('plot').getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top}}
function nearestPlotHit(event,radius=11){const point=plotEventPoint(event);return plotHits.map(hit=>({hit,distance:Math.hypot(hit.px-point.x,hit.py-point.y)})).filter(item=>item.distance<=radius).sort((a,b)=>a.distance-b.distance)[0]?.hit}
function prefetchPendingImages(limit=4){if(uiMode!=='campaign'||!campaign?.pending_queue?.length)return;const currentKey=`${run}:${index}`,position=campaign.queue.findIndex(item=>key(item)===currentKey),queuePosition=item=>campaign.queue.findIndex(candidate=>key(candidate)===key(item)),ordered=campaign.pending_queue.filter(item=>key(item)!==currentKey).sort((left,right)=>{const leftPosition=queuePosition(left),rightPosition=queuePosition(right),leftDistance=(leftPosition-position+campaign.queue.length)%campaign.queue.length,rightDistance=(rightPosition-position+campaign.queue.length)%campaign.queue.length;return leftDistance-rightDistance});ordered.slice(0,limit).forEach(item=>{const url=`/image?run=${encodeURIComponent(item.run)}&index=${item.index}`;if(prefetchedImageUrls.has(url))return;prefetchedImageUrls.add(url);const image=new Image();image.decoding='async';image.src=url});if(prefetchedImageUrls.size>512)prefetchedImageUrls.clear()}
function resetPlotZoom(){delete plotViews[run];plotDrag=null;$('plotTooltip').classList.add('hidden');renderPlot()}
async function initialize(){await loadOptions();await loadOverview()}
function openOverviewRow(row){if(row.dataset.open==='explore')openExplorer(row.dataset.run);else if(row.dataset.open==='setup')showSetup(row.dataset.run)}document.querySelectorAll('input[name="scope"]').forEach(input=>input.onchange=updateScope);document.querySelectorAll('input[name="strategy"]').forEach(input=>input.onchange=()=>$('allocationPreview').classList.add('hidden'));$('runSelect').onchange=()=>{updateWorkbooks();$('allocationPreview').classList.add('hidden')};$('preview').onclick=previewAllocation;$('start').onclick=startCampaign;$('undo').onclick=undoLast;$('newRound').onclick=()=>showSetup();$('dashboardNewRound').onclick=()=>showSetup();$('home').onclick=loadOverview;$('refreshOverview').onclick=loadOverview;$('resumeCampaign').onclick=()=>openWorkspace().catch(error=>$('overviewError').textContent=error.message);$('overviewTable').addEventListener('click',event=>{const row=event.target.closest('tr[data-open]');if(row)openOverviewRow(row)});$('overviewTable').addEventListener('keydown',event=>{if(event.key!=='Enter'&&event.key!==' ')return;const row=event.target.closest('tr[data-open]');if(row){event.preventDefault();openOverviewRow(row)}});$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('confirm').onclick=()=>save(false);$('unreadable').onclick=()=>save(true);$('scaleImage').onclick=()=>{if(fullImageUrl)$('fullImage').src=fullImageUrl;$('modal').classList.add('open')};$('modal').onclick=()=>{$('modal').classList.remove('open');$('fullImage').removeAttribute('src')};$('weight').onkeydown=event=>{if(event.key==='Enter')save(false)};$('showPartial').onchange=renderPlot;$('resetPlotZoom').onclick=resetPlotZoom;
$('invalidateRun').onclick=()=>openSource(run,'invalid-weight-data');
$('invalidateRunHeader').onclick=()=>openSource(run,'invalid-weight-data');
$('refreshSources').onclick=refreshSources;$('sourceClose').onclick=()=>$('sourceModal').classList.remove('open');$('sourceSave').onclick=saveSourceStatus;$('sourceModal').onclick=event=>{if(event.target===$('sourceModal'))$('sourceModal').classList.remove('open')};$('overviewTable').addEventListener('click',event=>{const invalidButton=event.target.closest('[data-invalid-run]');if(invalidButton){event.preventDefault();event.stopPropagation();openSource(invalidButton.dataset.invalidRun,invalidButton.dataset.invalidCurrent==='1'?null:'invalid-weight-data');return}const button=event.target.closest('[data-source-run]');if(!button)return;event.preventDefault();event.stopPropagation();openSource(button.dataset.sourceRun)},true);
const renderOverviewWithoutResume=renderOverview;
renderOverview=function(){
  renderOverviewWithoutResume();
  overview.runs.forEach(item=>{
    const row=$('overviewTable').querySelector(`tr[data-run="${item.run}"]`);
    if(!row)return;
    const action=item.campaign_active?'campaign':item.unfinished_round?'resume':item.has_data?'explore':item.review_available?'setup':'';
    if(action){row.dataset.open=action;row.tabIndex=0}
    else{delete row.dataset.open;row.removeAttribute('tabindex')}
    if(item.unfinished_round){
      const note=document.createElement('small');
      note.className='resume-note';
      note.textContent=`Continue round: ${item.unfinished_pending} remaining${item.unfinished_bootstrap_pending?`, ${item.unfinished_bootstrap_pending} temporal anchors`:''}`;
      row.cells[4].append(document.createElement('br'),note);
    }
  });
};
async function resumeRound(selectedRun){
  try{
    $('overviewError').textContent=`Restoring unfinished round for ${selectedRun.toUpperCase()}...`;
    await api('/api/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run:selectedRun})});
    const poll=async()=>{const job=await api('/api/job');$('overviewError').textContent=job.message||'';if(job.state==='building'){setTimeout(poll,500);return}if(job.state==='error')throw new Error(job.message);await openWorkspace()};
    await poll();
  }catch(error){$('overviewError').textContent=error.message}
}
openOverviewRow=function(row){if(row.dataset.open==='campaign')openWorkspace().catch(error=>$('overviewError').textContent=error.message);else if(row.dataset.open==='resume')resumeRound(row.dataset.run);else if(row.dataset.open==='explore')openExplorer(row.dataset.run);else if(row.dataset.open==='setup')showSetup(row.dataset.run)};
const moveWithoutPhaseBarrier=move;
move=function(delta){
  if(uiMode==='explore')return moveWithoutPhaseBarrier(delta);
  const queue=campaign?.review_queue||campaign?.queue||[];
  if(!queue.length)return;
  const position=queue.findIndex(item=>key(item)===`${run}:${index}`);
  const next=Math.max(0,Math.min(queue.length-1,(position<0?0:position)+delta));
  const target=queue[next];
  run=target.run;index=target.index;loadRecord();
};
const plotCanvas=$('plot');
plotCanvas.addEventListener('wheel',event=>{if(!plotGeometry||plotGeometry.run!==run)return;event.preventDefault();const point=plotEventPoint(event),g=plotGeometry;if(point.x<g.padL||point.x>g.width-g.padR)return;const fullSpan=g.fullMax-g.fullMin||1,oldSpan=g.xmax-g.xmin,factor=event.deltaY>0?1.28:.78,newSpan=Math.min(fullSpan,Math.max(fullSpan/500,oldSpan*factor)),anchor=Math.max(0,Math.min(1,(point.x-g.padL)/(g.width-g.padL-g.padR))),focus=g.xmin+anchor*oldSpan;let left=focus-anchor*newSpan;left=Math.max(g.fullMin,Math.min(g.fullMax-newSpan,left));plotViews[run]={xmin:left,xmax:left+newSpan};renderPlot()},{passive:false});
plotCanvas.addEventListener('pointerdown',event=>{if(event.button!==0||!plotGeometry)return;const point=plotEventPoint(event),g=plotGeometry;if(point.x<g.padL||point.x>g.width-g.padR||point.y<g.padT||point.y>g.height-g.padB)return;plotDrag={run,startX:point.x,currentX:point.x};plotCanvas.setPointerCapture(event.pointerId)});
plotCanvas.addEventListener('pointermove',event=>{const point=plotEventPoint(event),tooltip=$('plotTooltip');if(plotDrag&&plotDrag.run===run){const g=plotGeometry;plotDrag.currentX=Math.max(g.padL,Math.min(g.width-g.padR,point.x));tooltip.classList.add('hidden');renderPlot();return}const hit=nearestPlotHit(event);if(!hit){tooltip.classList.add('hidden');return}const wrap=$('plotWrap').getBoundingClientRect();tooltip.textContent=hit.label;tooltip.style.left=`${Math.min(wrap.width-218,Math.max(4,point.x+10))}px`;tooltip.style.top=`${Math.max(4,point.y-30)}px`;tooltip.classList.remove('hidden')});
plotCanvas.addEventListener('pointerup',event=>{if(!plotDrag)return;const drag=plotDrag,g=plotGeometry,hit=nearestPlotHit(event),distance=Math.abs(drag.currentX-drag.startX);plotDrag=null;try{plotCanvas.releasePointerCapture(event.pointerId)}catch{}if(distance>=8){const left=Math.min(drag.startX,drag.currentX),right=Math.max(drag.startX,drag.currentX),span=g.xmax-g.xmin,xmin=g.xmin+(left-g.padL)/(g.width-g.padL-g.padR)*span,xmax=g.xmin+(right-g.padL)/(g.width-g.padL-g.padR)*span;if(xmax-xmin>=(g.fullMax-g.fullMin)/500)plotViews[run]={xmin,xmax};renderPlot();return}renderPlot();if(hit){index=hit.index;loadRecord()}});
plotCanvas.addEventListener('pointercancel',()=>{plotDrag=null;renderPlot()});plotCanvas.addEventListener('pointerleave',()=>{if(!plotDrag)$('plotTooltip').classList.add('hidden')});plotCanvas.ondblclick=resetPlotZoom;window.addEventListener('resize',()=>{if(plotSeries.length)renderPlot()});
document.addEventListener('keydown',event=>{if(event.target===$('weight'))return;if(event.key==='ArrowLeft'&&(uiMode==='campaign'||uiMode==='explore'))move(-1);else if(event.key==='ArrowRight'&&(uiMode==='campaign'||uiMode==='explore'))move(1);else if(uiMode==='campaign'&&(event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z')undoLast()});initialize();setInterval(()=>{if(uiMode==='campaign'&&campaign){prefetchPendingImages();refreshCampaign()}else if(uiMode==='explore')refreshExplorer()},3000);
</script></body></html>"""


def serve(args: argparse.Namespace) -> None:
    specs = load_specs(args.config)
    image_time_offsets_h = load_image_time_offsets(args.config)
    bridge = ArtifactBridge()
    app = ReviewApp(
        specs,
        args.output_dir,
        bridge,
        raw_root=args.raw_root,
        production_runs=(
            tuple(specs) if args.configured_runs_only else FINAL_PRODUCTION_RUNS
        ),
        image_time_offsets_h=image_time_offsets_h,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    host_for_url = "127.0.0.1" if args.host in {"", "0.0.0.0"} else args.host
    url = f"http://{host_for_url}:{server.server_port}/"
    print(f"Overflow review UI: {url}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def inventory(args: argparse.Namespace) -> None:
    specs = load_specs(args.config)
    selected = args.runs or list(specs)
    for run in selected:
        spec = specs[run.lower()]
        frames = discover_spec_frames(spec)
        injection_start, _ = resolved_injection_window(spec)
        print(
            f"{run.lower():4s} frames={len(frames):5d} "
            f"first={frames[0].time.isoformat(sep=' ')} last={frames[-1].time.isoformat(sep=' ')} "
            f"injection={injection_start.isoformat(sep=' ')} feed={spec.gravity_feed_g_per_h:.0f} g/h"
        )


def prepare_sources(args: argparse.Namespace) -> None:
    specs = load_specs(args.config)
    image_time_offsets_h = load_image_time_offsets(args.config)
    runs = [run.lower() for run in (args.runs or FINAL_PRODUCTION_RUNS)]
    payload = build_source_inventory(
        raw_root=args.raw_root,
        output_root=args.output_dir,
        runs=runs,
        configured_roots={run: spec.image_root for run, spec in specs.items()},
        image_time_offsets_h=image_time_offsets_h,
    )
    rows = list(payload["runs"].values())
    with_images = sum(int(row["image_count"] > 0) for row in rows)
    configured = sum(int(row["configured"]) for row in rows)
    ready = sum(int(row["triage_status"] == "ready") for row in rows)
    print(f"Overflow source inventory: {args.output_dir / '_inventory'}")
    print(
        f"Runs={len(rows)} with_images={with_images} "
        f"configured={configured} ready={ready}"
    )


def _review_state_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "records": 0,
            "reviewed": 0,
            "confirmed": 0,
            "partial": 0,
            "unreadable": 0,
            "minimum_time_from_injection_h": None,
            "maximum_time_from_injection_h": None,
            "reviewers": {},
        }
    try:
        records = list(json.loads(path.read_text(encoding="utf-8")).get("records", []))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}
    reviewed = [
        record
        for record in records
        if str(record.get("status", "")) in REVIEWED_STATUSES
    ]
    times = [
        value
        for record in reviewed
        if (value := _float_or_none(record.get("time_from_injection_h"))) is not None
    ]
    reviewers: dict[str, int] = {}
    for record in reviewed:
        reviewer = str(record.get("reviewer") or "").strip() or "(missing)"
        reviewers[reviewer] = reviewers.get(reviewer, 0) + 1
    return {
        "records": len(records),
        "reviewed": len(reviewed),
        "confirmed": sum(
            str(record.get("status", "")) in CONFIRMED_STATUSES for record in records
        ),
        "partial": sum(
            str(record.get("status", "")) == "Partial" for record in records
        ),
        "unreadable": sum(
            str(record.get("status", "")) == "Unreadable" for record in records
        ),
        "minimum_time_from_injection_h": min(times) if times else None,
        "maximum_time_from_injection_h": max(times) if times else None,
        "reviewers": reviewers,
    }


def _is_review_reset_file(path: Path, run: str) -> bool:
    name = path.name.lower()
    return (
        name.startswith(f"{run}_overflow_scale_review")
        or name.startswith(f".{run}_overflow_scale_review.")
        or name == f"{run}_overflow_predictions.json"
    )


def archive_reset_reviews(
    output_root: Path,
    runs: list[str],
    *,
    reason: str = DEFAULT_RESET_REASON,
    archive_name: str | None = None,
    sample_count: int = DEFAULT_SINGLE_SAMPLE_COUNT,
    max_hours: float = DEFAULT_SINGLE_MAX_HOURS,
    reviewer: str = DEFAULT_REVIEWER,
) -> dict[str, Any]:
    """Archive review-derived files so selected runs can be reviewed from scratch."""
    output_root = Path(output_root)
    normalized_runs = list(dict.fromkeys(str(run).strip().lower() for run in runs))
    invalid = [run for run in normalized_runs if not re.fullmatch(r"ac\d+", run)]
    if invalid:
        raise ValueError(f"Invalid run names: {', '.join(invalid)}")
    if not normalized_runs:
        raise ValueError("Select at least one run to reset")
    normalized_runs.sort(key=lambda run: int(run[2:]))
    if archive_name is None:
        archive_name = f"{datetime.now():%Y%m%dT%H%M%S}_standard_15h_110"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", archive_name):
        raise ValueError(
            "Archive name may contain only letters, numbers, dot, dash and underscore"
        )
    archive_root = output_root / "_review_archive" / archive_name
    if archive_root.exists():
        raise FileExistsError(f"Archive already exists: {archive_root}")
    archive_root.mkdir(parents=True)
    manifest_path = archive_root / "reset_manifest.json"
    manifest: dict[str, Any] = {
        "schema": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "action": "archive-reset-manual-overflow-review",
        "reason": str(reason),
        "output_root": str(output_root),
        "archive_root": str(archive_root),
        "replacement_standard": {
            "sample_count_per_run": int(sample_count),
            "maximum_time_after_injection_h": float(max_hours),
            "reviewer": str(reviewer),
            "stop_after_planned_overflow_end": False,
            "include_common_pre_injection_baseline": True,
        },
        "preserved_file_types": [
            "overflow_image_manifest",
            "overflow_ocr_evidence",
            "overflow_auto_ocr_geometry",
            "raw_source_images",
        ],
        "runs": [],
    }

    def write_manifest() -> None:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(_json_text(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)

    try:
        for run in normalized_runs:
            run_dir = output_root / run
            state_path = run_dir / f"{run}_overflow_scale_review.json"
            entry: dict[str, Any] = {
                "run": run,
                "previous_review": _review_state_audit(state_path),
                "moved_files": [],
            }
            manifest["runs"].append(entry)
            if not run_dir.exists():
                entry["status"] = "source-directory-missing"
                write_manifest()
                continue
            destination = archive_root / run
            candidates = sorted(
                path
                for path in run_dir.iterdir()
                if path.is_file() and _is_review_reset_file(path, run)
            )
            if candidates:
                destination.mkdir(parents=True)
            for source in candidates:
                target = destination / source.name
                stat = source.stat()
                shutil.move(str(source), str(target))
                entry["moved_files"].append(
                    {
                        "source": str(source),
                        "archive": str(target),
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(
                            timespec="seconds"
                        ),
                    }
                )
            entry["status"] = "reset" if candidates else "already-empty"
            write_manifest()
    except Exception as exc:
        manifest["error"] = str(exc)
        write_manifest()
        raise
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    write_manifest()
    return manifest


def archive_reset(args: argparse.Namespace) -> None:
    manifest = archive_reset_reviews(
        args.output_dir,
        args.runs,
        reason=args.reason,
        archive_name=args.archive_name,
        sample_count=args.sample_count,
        max_hours=args.max_hours,
        reviewer=args.reviewer,
    )
    print(f"Archived reset: {manifest['archive_root']}")
    for entry in manifest["runs"]:
        previous = entry["previous_review"]
        print(
            f"{entry['run']:4s} status={entry['status']} "
            f"confirmed={previous.get('confirmed', 0)} "
            f"reviewed={previous.get('reviewed', 0)} "
            f"files={len(entry['moved_files'])}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = result.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="Start the local review UI")
    serve_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    serve_parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--no-browser", action="store_true")
    serve_parser.add_argument(
        "--configured-runs-only",
        action="store_true",
        help="Expose only runs present in the supplied project adapter config",
    )
    serve_parser.set_defaults(func=serve)
    inventory_parser = subparsers.add_parser(
        "inventory", help="Show configured image coverage"
    )
    inventory_parser.add_argument("--runs", nargs="*")
    inventory_parser.set_defaults(func=inventory)
    prepare_parser = subparsers.add_parser(
        "prepare-sources",
        help="Scan production overflow sources and create representative contact sheets",
    )
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    prepare_parser.add_argument("--runs", nargs="*")
    prepare_parser.set_defaults(func=prepare_sources)
    reset_parser = subparsers.add_parser(
        "archive-reset",
        help="Archive review-derived files and reset selected runs for manual review",
    )
    reset_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    reset_parser.add_argument("--runs", nargs="+", required=True)
    reset_parser.add_argument("--reason", default=DEFAULT_RESET_REASON)
    reset_parser.add_argument("--archive-name")
    reset_parser.add_argument(
        "--sample-count", type=int, default=DEFAULT_SINGLE_SAMPLE_COUNT
    )
    reset_parser.add_argument(
        "--max-hours", type=float, default=DEFAULT_SINGLE_MAX_HOURS
    )
    reset_parser.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    reset_parser.set_defaults(func=archive_reset)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Blind schema-2 generic-versus-BTB transfer campaign for FF-AC and FF-UM.

The two frozen binary free-gas models are evaluated on corrected, native-resolution
images.  Target runs are never used for fitting.  Each completed run is published
atomically in the existing pairwise-review format, including unscored blind quiz
regions for future five-class training.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import csv
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import tomllib
from typing import Any, Iterator, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")
FFUM_ROOT = series_root("ffum")
FFUM_WORKFLOW_ROOT = FFUM_ROOT
FFUM_PROJECT_ENV = "PHASE_FFUM_PROJECT"

from darsia.fluidflower.tools import ac_overflow_first_colorchecker_screen as optical_control  # noqa: E402
from darsia.fluidflower.tools import auto_calibrate_color_to_mass as ffac_calibration  # noqa: E402
from darsia.fluidflower.tools import colorchecker_saturation_fallback as saturation_fallback  # noqa: E402

# Clipped grey patches of the chart (FF-UM RUN14, 2026-09-18): a flagged rebuild of the tone curve instead
# of a refused run. Installed here, before any refinement is configured, from outside the fingerprinted
# module; PHASE_COLORCHECKER_SATURATION_FALLBACK=0 keeps the frozen behaviour.
SATURATION_FALLBACK = saturation_fallback.install()
from darsia.fluidflower.tools import calibration_pairwise_review as review  # noqa: E402
from darsia.fluidflower.tools import calibration_review_movie as review_movie  # noqa: E402
from darsia.fluidflower.tools.dual_phase_signal import DualPhaseSignalModel  # noqa: E402
from darsia.fluidflower.tools.clearwater_phase_reference import (  # noqa: E402
    average_rgb,
    select_clear_reference_paths,
)
from darsia.fluidflower.tools.optical_phase_features import (  # noqa: E402
    OpticalFeatureContract,
    optical_feature_matrix,
    prepare_optical_feature_workspace,
)
from darsia.fluidflower.tools.temporal_gas_support import (  # noqa: E402
    CausalGasGateFrame,
    CausalGasGateSettings,
    apply_causal_gas_gate,
)


CAMPAIGN_ID = "schema2_phase_transfer_ab_20260829"
MODEL_CONTROL_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'dual_signal_exact_ab_darsia91f5bc9_20260829_5000'
MODEL_ROOT = MODEL_CONTROL_ROOT / "transferable_phase_models_v1"
MODEL_PATHS = {
    "generic": MODEL_ROOT / "none" / "uniform24" / "gas_signal.json",
    "btb_adapter": (
        MODEL_ROOT
        / "btb-ratiometric-v1"
        / "uniform24"
        / "gas_signal.json"
    ),
}
AQUEOUS_MODEL_PATH: Path | None = None
DEFAULT_OUTPUTS = {
    "ffac": configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}',
    "ffum": configured_tool_path('ffum', 'analysis_v2') / f'{CAMPAIGN_ID}',
}
RUNS = {
    "ffac": ("ac18", "ac52", "ac56"),
    "ffum": ("run10", "run15", "run16", "run18", "run20"),
}
TARGET_TIMES_H = {
    "ffac": (
        0.164,
        0.547,
        0.897,
        1.331,
        1.914,
        2.497,
        2.997,
        4.083,
        6.0,
        8.0,
        12.0,
        24.0,
        48.0,
    ),
    "ffum": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 48.0, 72.0, 96.0),
}
INJECTION_REVIEW_FRACTIONS: tuple[float, ...] = ()
BLIND_SEED = "schema2-generic-vs-btb-transfer-20260829"
PROBABILITY_LEVELS = (0.25, 0.50, 0.75)
PHASE_FILL_MIN_CO2_EVIDENCE = 0.50
PHASE_FILL_AQUEOUS_MAX_GAS_SHARE = 0.40
PHASE_FILL_GAS_MIN_GAS_SHARE = 0.60
PHASE_FILL_ALPHA = 255
PHASE_FILL_COLORS = {
    "aqueous": (49, 217, 160, PHASE_FILL_ALPHA),
    "mixed": (255, 191, 71, PHASE_FILL_ALPHA),
    "gas": (242, 76, 157, PHASE_FILL_ALPHA),
}
MODEL_ORDER = ("generic", "btb_adapter")
TRANSIENT_IO_RETRIES = 40
NEGATIVE_CONTROL_LIMITS = {
    "gas_probability_mean": 0.02,
    "gas_probability_p99": 0.20,
    "support_fraction_p025": 0.002,
    "support_fraction_p050": 0.0005,
    "support_fraction_p075": 0.0001,
}
AQUEOUS_NEGATIVE_CONTROL_LIMITS = {
    "gas_probability_mean": 0.02,
    "gas_probability_p99": 0.20,
    "support_fraction_p025": 0.002,
    "support_fraction_p050": 0.0005,
}
AQUEOUS_ENVELOPE_RADIUS_FRACTION = 0.005
AQUEOUS_NEGATIVE_CALIBRATION_QUANTILE = 0.999
AQUEOUS_NEGATIVE_CALIBRATION_TARGET = 0.05
MAX_AQUEOUS_LOGIT_SHIFT = 3.5
GAS_NEGATIVE_CALIBRATION_QUANTILE = 0.999
GAS_NEGATIVE_CALIBRATION_TARGET = 0.05
MAX_GAS_LOGIT_SHIFT = 4.0
HIERARCHICAL_CO2_NEGATIVE_FLOOR_MAX = 0.35
HIERARCHICAL_GAS_NEGATIVE_FLOOR_MAX = 0.75
CAUSAL_TOPOLOGY_SCALE = 0.20
CAUSAL_GAS_SETTINGS = CausalGasGateSettings(
    support_level=0.05,
    optical_onset=0.10,
    optical_width=0.15,
    history_radius_m=0.005,
    reentry_radius_m=0.015,
    source_radius_x_m=0.040,
    source_radius_y_m=0.030,
    optical_smoothing_sigma_m=0.0015,
    minimum_component_area_m2=1.0e-5,
    structure_scale=1.0,
    reject_seal_dominant_components=False,
    allow_nonseal_reentry=True,
)
CAUSAL_GAS_SETTINGS_BY_CANDIDATE: dict[str, CausalGasGateSettings] = {}
FFAC_RAW_ROOT = configured_tool_path('ffac', 'raw_inputs')
FFUM_CLEARWATER_REFERENCE_MANIFEST = os.environ.get(
    "FFUM_CLEARWATER_REFERENCE_MANIFEST",
    str(configured_tool_path('ffum', 'analysis_v2') / 'clearwater_references/manifest.json'),
)
LEGACY_NO_REFERENCE_CODE_HASHES = {
    "signal_model": "cf66f798a69f62d15c95614c9eac8ce58a3992bca4a21874422b195702894cf3",
    "optical_features": "34d173f21fdc1d777faaec71e1c7f9bbd2dcb2857c901e4e30851e51b544c3ca",
}
PHASE_QUIZ_NON_POROUS_FACIES = {
    "ffac": frozenset({0}),
    "ffum": frozenset({1}),
}
PHASE_QUIZ_DOMAIN_NAMES = {
    "ffac": "porous_medium_facies_nonzero",
    "ffum": "porous_medium_facies_2_to_9",
}
PHASE_QUIZ_DOMAIN_EXCLUSIONS = {
    "porous_medium_facies_nonzero": "overlying free-water layer",
    "porous_medium_facies_2_to_9": (
        "FF-UM non-porous facies 1, including the overlying free-water layer"
    ),
}
PHASE_QUIZ_HYPOTHESES = ("gas", "aqueous", "mixed", "water")
PHASE_QUIZ_CENTERS_PER_FACIES_FRAME = 4

# FF-UM's injection spreadsheet contains administrative source coordinates that do
# not coincide with the first visible CO2 in the corrected image plane.  The causal
# optical prior needs the latter.  These anchors were measured from the first
# 30-120 seconds of both injections in Run15 and independently checked in Run16.
# Protocol times and rates remain authoritative and are never replaced here.
FFUM_VISIBLE_SOURCE_ANCHORS_M = (
    {
        "x_m": 0.90,
        "y_m": 0.35,
        "uncertainty_m": 0.02,
        "coordinate_source": "ffum_visible_co2_onset_run15_run16_v1",
    },
    {
        "x_m": 1.88,
        "y_m": 0.58,
        "uncertainty_m": 0.02,
        "coordinate_source": "ffum_visible_co2_onset_run15_run16_v1",
    },
)


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    elapsed_h: float


@dataclass
class RunSource:
    dataset: str
    run: str
    baseline: np.ndarray
    frames: list[FrameInfo]
    rig: Any
    experiment: Any
    valid_mask: np.ndarray
    phase_quiz_mask: np.ndarray
    phase_quiz_domain: str
    correction_contract: Mapping[str, Any]
    phase_evaluation_mask: np.ndarray | None = None
    phase_evaluation_domain: str = "valid_analysis_image"
    clearwater: np.ndarray | None = None
    clearwater_contract: Mapping[str, Any] | None = None

    def read(self, path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
        try:
            image = self.rig.read_image(path, dtype=np.float32)
        except TypeError:
            image = self.rig.read_image(path)
        array = np.asarray(image.img, dtype=np.float32)[..., :3]
        return array, _correction_diagnostics(self.rig)


@dataclass(frozen=True)
class ReviewPipelineSettings:
    inference_workers: int
    render_workers: int
    max_buffered_frames: int


@dataclass(frozen=True)
class _PreparedReviewFrame:
    index: int
    frame_info: FrameInfo
    image: np.ndarray
    correction_rows: list[dict[str, Any]]
    read_seconds: float
    selection_status: str = "nearest_standardized_target"


@dataclass(frozen=True)
class _InferredReviewFrame:
    prepared: _PreparedReviewFrame
    predictions: dict[str, np.ndarray]
    aqueous_probabilities: dict[str, np.ndarray | None]
    inference_seconds: float


@dataclass(frozen=True)
class _CandidateReviewState:
    probability: np.ndarray
    aqueous_probability: np.ndarray | None
    record: dict[str, Any]


@dataclass(frozen=True)
class _RenderedReviewFrame:
    frame_source: dict[str, Any]
    registration: dict[str, Any]
    records: dict[str, dict[str, Any]]
    movie_frames: dict[str, np.ndarray]
    proposals: dict[str, list[dict[str, Any]]]


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value}")
    return value


def _review_pipeline_settings(
    dataset: str,
    frame_count: int,
) -> ReviewPipelineSettings:
    """Return conservative bounded concurrency for native-resolution review output."""

    if dataset not in {"ffac", "ffum"}:
        raise ValueError(f"Unsupported review dataset {dataset!r}")
    count = max(1, int(frame_count))
    default_workers = 2 if dataset == "ffum" else 3
    inference_workers = min(
        count,
        _positive_environment_integer(
            "DARSIA_PHASE_REVIEW_INFERENCE_WORKERS", default_workers
        ),
    )
    render_workers = min(
        count,
        _positive_environment_integer(
            "DARSIA_PHASE_REVIEW_RENDER_WORKERS", default_workers
        ),
    )
    max_buffered_frames = min(
        count,
        _positive_environment_integer(
            "DARSIA_PHASE_REVIEW_MAX_BUFFERED_FRAMES",
            max(inference_workers, render_workers),
        ),
    )
    max_buffered_frames = max(
        max_buffered_frames,
        inference_workers,
        render_workers,
    )
    return ReviewPipelineSettings(
        inference_workers=inference_workers,
        render_workers=render_workers,
        max_buffered_frames=max_buffered_frames,
    )


def _add_review_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inference-workers",
        type=int,
        help="Parallel native-frame optical inference workers.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        help="Parallel overlay, quiz, encoding, and write workers.",
    )
    parser.add_argument(
        "--max-buffered-frames",
        type=int,
        help="Bound corrected/inferred frames retained by the review pipeline.",
    )


def _apply_review_pipeline_arguments(args: argparse.Namespace) -> None:
    values = {
        "DARSIA_PHASE_REVIEW_INFERENCE_WORKERS": getattr(
            args, "inference_workers", None
        ),
        "DARSIA_PHASE_REVIEW_RENDER_WORKERS": getattr(args, "render_workers", None),
        "DARSIA_PHASE_REVIEW_MAX_BUFFERED_FRAMES": getattr(
            args, "max_buffered_frames", None
        ),
    }
    for name, value in values.items():
        if value is None:
            continue
        if int(value) < 1:
            raise SystemExit(f"{name} must be a positive integer")
        os.environ[name] = str(int(value))


class NegativeControlPreflightError(RuntimeError):
    def __init__(self, run: str, report_path: Path, report: Mapping[str, Any]):
        super().__init__(
            f"{run} failed the no-gas baseline/clear-water preflight; "
            f"see {report_path}"
        )
        self.run = str(run)
        self.report_path = Path(report_path)
        self.report = dict(report)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    for attempt in range(TRANSIENT_IO_RETRIES):
        try:
            temporary.write_bytes(content)
            os.replace(temporary, path)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt + 1 >= TRANSIENT_IO_RETRIES:
                raise
            time.sleep(min(3.0, 0.1 * (2**min(attempt, 5))))


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        json.dumps(payload, indent=2, allow_nan=False, default=str).encode("utf-8"),
    )


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    import io

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _write_jpeg(path: Path, image: np.ndarray, quality: int = 98) -> None:
    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not success:
        raise RuntimeError(f"Could not encode {path}")
    _atomic_bytes(path, encoded.tobytes())


def _write_png(path: Path, image: np.ndarray) -> None:
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[2] == 4:
        values = cv2.cvtColor(values, cv2.COLOR_RGBA2BGRA)
    elif values.ndim == 3 and values.shape[2] == 3:
        values = cv2.cvtColor(values, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".png", values, [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
    )
    if not success:
        raise RuntimeError(f"Could not encode {path}")
    _atomic_bytes(path, encoded.tobytes())


def _profile() -> Any:
    return next(profile for profile in optical_control.OPTICAL_PROFILES if profile.name == "uniform24")


def _correction_diagnostics(rig: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    registration = getattr(rig, "_phase_rig_registration", None)
    if registration is not None:
        rows.append({"type": "RigRegistrationCorrection", "stage": "shape",
                     "receipt": registration.last_receipt})
    for index, correction in enumerate(getattr(rig, "color_corrections", ()) or ()):
        if not getattr(correction, "active", False):
            continue
        row: dict[str, Any] = {
            "index": int(index),
            "type": type(correction).__name__,
            "flagged": bool(getattr(correction, "last_flagged", False)),
            "strategy": str(getattr(correction, "correction_strategy", "legacy")),
        }
        for attribute, key in (
            ("last_residual_before_correction", "residual_before_correction"),
            ("last_residual_before_refinement", "residual_before_refinement"),
            ("last_residual", "residual"),
            ("last_weighted_residual", "weighted_residual"),
            ("last_max_swatch_residual", "max_swatch_residual"),
        ):
            value = getattr(correction, attribute, None)
            if value is not None and np.isfinite(float(value)):
                row[key] = float(value)
        row["refinement_steps"] = int(
            getattr(correction, "last_refinement_steps_applied", 0) or 0
        )
        strategy_diagnostics = getattr(correction, "last_strategy_diagnostics", {}) or {}
        for key in ("algorithm_id", "algorithm_revision", "affine_accepted", "affine_bounds_ok",
                    "residual_affine_enabled", "residual_model",
                    "affine_relative_improvement", "affine_minimum_improvement", "affine_rejection_reason",
                    "tone_residual_score", "affine_residual_score"):
            value = strategy_diagnostics.get(key)
            if value is not None and not (isinstance(value, float) and not np.isfinite(value)):
                row[key] = value
        rows.append(row)
    return rows


def _valid_mask(rig: Any, shape: Sequence[int]) -> np.ndarray:
    for name in ("boolean_image_porosity", "image_porosity"):
        value = getattr(rig, name, None)
        if value is None:
            continue
        array = np.asarray(getattr(value, "img", value))
        if array.shape[:2] == tuple(shape[:2]):
            mask = np.isfinite(array) & (array > 0)
            if np.count_nonzero(mask) > 0.05 * mask.size:
                return np.asarray(mask, dtype=bool)
    return np.ones(tuple(shape[:2]), dtype=bool)


def _phase_quiz_domain(
    dataset: str,
    rig: Any,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return the physically relevant 2D domain for blind quiz regions."""

    mask = np.asarray(valid_mask, dtype=bool).copy()
    facies = getattr(rig, "facies", None)
    facies_array = None if facies is None else np.asarray(getattr(facies, "img", facies))
    if facies_array is not None and facies_array.shape[:2] == mask.shape:
        excluded = PHASE_QUIZ_NON_POROUS_FACIES.get(dataset)
        if excluded:
            # FF-AC uses 0 for free water. FF-UM uses facies 1 for "other"
            # with zero porosity/permeability; this includes its free-water
            # layer. Neither is a valid visible-phase training target.
            porous = np.isfinite(facies_array) & ~np.isin(
                facies_array,
                tuple(excluded),
            )
            restricted = mask & porous
            if np.count_nonzero(restricted) > 0.05 * restricted.size:
                return (
                    np.asarray(restricted, dtype=bool),
                    PHASE_QUIZ_DOMAIN_NAMES[dataset],
                )
    return mask, "valid_analysis_image"


def _phase_classification_domain(
    dataset: str,
    rig: Any,
    fallback_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return broad porous support for phase inference without optical seam holes.

    ``image_porosity`` is a continuous visibility/weight field and can contain
    zero-valued ribbons along facies interfaces. Those pixels may be unsuitable
    for quiz sampling or mass weighting, but they must not be hard-zeroed before
    optical phase inference. Prefer DarSIA's boolean domain and then remove only
    explicitly non-porous facies such as the overlying free-water layer.
    """

    shape = np.asarray(fallback_mask, dtype=bool).shape
    mask: np.ndarray | None = None
    for name in ("boolean_porosity", "boolean_image_porosity"):
        value = getattr(rig, name, None)
        if value is None:
            continue
        array = np.asarray(getattr(value, "img", value))
        if array.shape[:2] != shape:
            continue
        candidate = np.isfinite(array) & array.astype(bool)
        if np.count_nonzero(candidate) > 0.05 * candidate.size:
            mask = np.asarray(candidate, dtype=bool)
            break
    if mask is None:
        mask = np.ones(shape, dtype=bool)

    facies = getattr(rig, "facies", None)
    facies_array = None if facies is None else np.asarray(getattr(facies, "img", facies))
    excluded = PHASE_QUIZ_NON_POROUS_FACIES.get(dataset)
    if (
        facies_array is not None
        and facies_array.shape[:2] == shape
        and excluded
    ):
        porous = np.isfinite(facies_array) & ~np.isin(
            facies_array,
            tuple(excluded),
        )
        restricted = mask & porous
        if np.count_nonzero(restricted) > 0.05 * restricted.size:
            return (
                np.asarray(restricted, dtype=bool),
                f"{PHASE_QUIZ_DOMAIN_NAMES[dataset]}_broad_phase_inference",
            )
    return mask, "boolean_porosity_phase_inference"


def _select_frames(
    experiment: Any,
    target_times: Sequence[float],
    *,
    tolerance_seconds: float | None,
) -> list[FrameInfo]:
    selected: dict[str, FrameInfo] = {}
    for target in target_times:
        try:
            path = experiment.find_images_for_times(
                float(target), tol=tolerance_seconds
            )
        except (OSError, RuntimeError, ValueError):
            path = None
        if path is None:
            continue
        resolved = Path(path)
        acquired = experiment.get_datetime(resolved)
        elapsed = float(experiment.time_since_start(acquired))
        selected[str(resolved).casefold()] = FrameInfo(resolved, elapsed)
    frames = sorted(selected.values(), key=lambda item: item.elapsed_h)
    if len(frames) < 5:
        raise RuntimeError(
            f"Only {len(frames)} review frames resolved from {len(target_times)} targets"
        )
    return frames


def _review_target_times(
    experiment: Any,
    base_times: Sequence[float],
    *,
    injection_fractions: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Add protocol-derived injection history without changing legacy campaigns."""

    targets = {
        float(value)
        for value in base_times
        if np.isfinite(float(value)) and float(value) >= 0.0
    }
    fractions = tuple(
        float(value)
        for value in (
            INJECTION_REVIEW_FRACTIONS
            if injection_fractions is None
            else injection_fractions
        )
        if np.isfinite(float(value)) and 0.0 < float(value) < 1.0
    )
    if not fractions:
        return tuple(sorted(targets))
    try:
        stages = _injection_stages(experiment)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return tuple(sorted(targets))
    for stage in stages:
        start_h = float(stage["start_h"])
        end_h = float(stage["end_h"])
        if not (np.isfinite(start_h) and np.isfinite(end_h) and end_h > start_h):
            continue
        duration_h = end_h - start_h
        targets.update(start_h + fraction * duration_h for fraction in fractions)
    return tuple(sorted(targets))


def _read_corrected_clearwater(
    rig: Any,
    paths: Sequence[Path],
    baseline_shape: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    import darsia

    corrected: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        image = darsia.imread(
            path,
            transformations=rig.corrections,
            reference_date=rig.reference_date,
            name=path.name,
            dtype=np.float32,
        )
        array = np.asarray(image.img, dtype=np.float32)[..., :3]
        if array.shape != tuple(baseline_shape):
            raise RuntimeError(
                f"Clear-water image {path} has shape {array.shape}; "
                f"expected {tuple(baseline_shape)}"
            )
        corrected.append(array)
        rows.append(
            {
                "path": str(path),
                "source_sha256": _sha256(path),
                "corrected_array_sha256": _array_sha256(array),
                "color_corrections": _correction_diagnostics(rig),
            }
        )
    reference = average_rgb(corrected)
    from darsia.fluidflower.tools.phase_rig_registration import geometry_validity_contract
    geometry_validity = geometry_validity_contract(rig)
    return reference, {
        "available": True,
        "kind": "true-clear-water-soft-optical-reference",
        "aggregation": "mean of independently radiometrically corrected frames",
        "source_count": len(rows),
        "sources": rows,
        "corrected_reference_sha256": _array_sha256(reference),
        "direct_mask_or_saturation_use": False,
        **({"geometry_validity": geometry_validity} if geometry_validity is not None else {}),
    }


@contextmanager
def _open_ffac(run: str, *, require_clearwater: bool = False,
               registration_profile: Mapping[str, Any] | None = None,
               disable_registration: bool = False,
               common_colorchecker_target: tuple[Path, str] | None = None) -> Iterator[RunSource]:
    from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
    import darsia
    from darsia.fluidflower.tools.phase_rig_registration import (registration_profile_for_run,
                                               install_registration, registration_valid_mask)
    from darsia.fluidflower.series_paths import config_files

    profile = _profile()
    environment = optical_control._profile_environment(profile)
    # Phase correction needs the registered rig and experiment, not a legacy
    # colour-to-mass embedding overlay. Native series TOMLs own those inputs.
    config_paths = config_files("ffac", run)
    with optical_control.rebuilt._temporary_environment(environment):
        config = FluidFlowerConfig(
            config_paths, require_data=True, require_results=False
        )
        experiment = darsia.ProtocolledExperiment.init_from_config(config)
        if disable_registration and registration_profile is not None:
            raise ValueError("Cannot specify registration_profile when registration is disabled")
        registration_profile = (None if disable_registration else
                                registration_profile_for_run(run)
                                if registration_profile is None else dict(registration_profile))
        if registration_profile is None:
            rig = optical_control.rebuilt.RefinedColorPathRig.load(config.rig.path)
        else:
            rig = optical_control.rebuilt.Rig.load(config.rig.path)
            install_registration(rig, registration_profile, run=run)
            refinement = optical_control.rebuilt._configure_runtime_colorchecker_refinement(rig)
            rig._ffac_colorchecker_refinement_provenance = refinement
        common_target_receipt = None
        if common_colorchecker_target is not None:
            from darsia.fluidflower.tools.phase_two_arm_photometry import (
                install_common_target, load_target,
            )

            target_path, target_sha256 = common_colorchecker_target
            swatches, frozen_receipt = load_target(
                target_path, expected_sha256=target_sha256,
            )
            common_target_receipt = install_common_target(
                rig, swatches, target_receipt=frozen_receipt,
            )
        rig.load_experiment(experiment)
        baseline = np.asarray(rig.baseline.img, dtype=np.float32)[..., :3]
        clearwater = None
        clearwater_contract = None
        if require_clearwater:
            paths = select_clear_reference_paths(FFAC_RAW_ROOT, run)
            clearwater, clearwater_contract = _read_corrected_clearwater(
                rig,
                paths,
                baseline.shape,
            )
        frames = _select_frames(
            experiment,
            _review_target_times(experiment, TARGET_TIMES_H["ffac"]),
            tolerance_seconds=30.0 * 60.0,
        )
        valid_mask = _valid_mask(rig, baseline.shape)
        phase_quiz_mask, phase_quiz_domain = _phase_quiz_domain(
            "ffac", rig, valid_mask
        )
        phase_evaluation_mask, phase_evaluation_domain = (
            _phase_classification_domain("ffac", rig, valid_mask)
        )
        geometry_mask = registration_valid_mask(rig)
        if geometry_mask is not None:
            valid_mask &= geometry_mask
            phase_quiz_mask &= geometry_mask
            phase_evaluation_mask &= geometry_mask
            phase_evaluation_domain += ":registration-valid-domain"
        yield RunSource(
            dataset="ffac",
            run=run,
            baseline=baseline,
            frames=frames,
            rig=rig,
            experiment=experiment,
            valid_mask=valid_mask,
            phase_quiz_mask=phase_quiz_mask,
            phase_quiz_domain=phase_quiz_domain,
            phase_evaluation_mask=phase_evaluation_mask,
            phase_evaluation_domain=phase_evaluation_domain,
            correction_contract={
                "profile": profile.name,
                "environment": environment,
                "config_paths": [str(path) for path in config_paths],
                "source": "shared FF-AC uniform24 method with FF-AC rig data",
                **({"photometry_arm": "global-common-colorchecker-raw-rig",
                    "common_colorchecker_target": common_target_receipt}
                   if common_target_receipt is not None else {}),
                **({"rig_registration": rig._phase_rig_registration_provenance}
                   if registration_profile is not None else {}),
            },
            clearwater=clearwater,
            clearwater_contract=clearwater_contract,
        )


def _resolve_protocol(value: Any) -> Path | tuple[Path, str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        path = (FFUM_WORKFLOW_ROOT / Path(value[0])).resolve()
        return path, str(value[1])
    return (FFUM_WORKFLOW_ROOT / Path(value)).resolve()


def _ffum_serialized_rig_path(run: str) -> Path:
    number = int("".join(character for character in run if character.isdigit()))
    return (
        configured_tool_path('ffum', 'runtime_inputs') / 'Results'
        / f"Run{number}"
        / "results"
        / "setup"
        / "rig"
    )


def _ffum_project_root() -> Path | None:
    """FF-UM project folder in the DarSIA format, or None for the legacy V1 files.

    Set PHASE_FFUM_PROJECT to a folder holding config\\common.toml, config\\runs\\<run>.toml
    and the rigs built by DarSIA under <results>\\setup\\rig (e.g. Z:\\FF_UM\\DarSIA).
    """

    value = os.environ.get(FFUM_PROJECT_ENV, "").strip()
    return Path(value) if value else series_root("ffum")


def _deep_merge_toml(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested tables like DarSIA's configuration loader (later files win)."""

    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            _deep_merge_toml(base[key], value)
        else:
            base[key] = value
    return base


def _ffum_project_config(root: Path, run: str) -> tuple[dict[str, Any], list[Path]]:
    """common.toml + runs/<run>.toml of a DarSIA-format project, deep-merged."""

    paths = [root / "config" / "common.toml", root / "config" / "runs" / f"{run}.toml"]
    merged: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"FF-UM project configuration is missing: {path}")
        with path.open("rb") as stream:
            _deep_merge_toml(merged, tomllib.load(stream))
    return merged, paths


def _ffum_project_rig_path(config: Mapping[str, Any]) -> Path:
    results = Path(str(config["data"]["results"]))
    return results / "setup" / "rig"


def _ffum_rig_path(run: str) -> Path:
    """Rig folder of an FF-UM run: the project's rig when PHASE_FFUM_PROJECT is set."""

    root = _ffum_project_root()
    if root is None:
        return _ffum_serialized_rig_path(run)
    config, _ = _ffum_project_config(root, run)
    return _ffum_project_rig_path(config)


def _install_resize_correction_alias(darsia_module: Any) -> dict[str, Any]:
    """Let this darsia read rig files written by DarSIA-dev.

    DarSIA-dev persists its resize step as ``ResizeCorrection`` (a BaseCorrection wrapper
    around ``Resize`` with the same saved keys: shape, dsize, fx, fy, interpolation, dtype).
    ``darsia.read_correction`` evaluates the stored class name in its own module namespace,
    which in older darsia versions only knows ``Resize``. Give it the name; nothing else
    changes.
    """

    try:
        from darsia.corrections import readcorrection
    except ImportError:  # unexpected layout: let the loader report the real problem
        return {"installed": False, "reason": "darsia.corrections.readcorrection missing"}
    if hasattr(readcorrection, "ResizeCorrection"):
        return {"installed": False, "reason": "reader already knows ResizeCorrection"}
    target = getattr(darsia_module, "ResizeCorrection", None) or getattr(
        darsia_module, "Resize"
    )
    readcorrection.ResizeCorrection = target
    return {"installed": True, "alias": "ResizeCorrection -> " + target.__name__}


def _project_protocol_path(root: Path, value: Any) -> Path | tuple[Path, str] | None:
    """Protocol paths of a project configuration: absolute as given, relative to the project."""

    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        path = Path(str(value[0]))
        return (path if path.is_absolute() else root / path), str(value[1])
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _sanitize_blacklist_protocol(value, *, cache_root=None, image_paths=()):
    from darsia.fluidflower.tools.native_protocols import blacklist_protocol
    return blacklist_protocol(value, image_paths=image_paths, cache_root=cache_root)


def _single_blacklist_for_all_folders(blacklist_protocol, *, cache_root=None):
    """Native DarSIA preserves each photograph folder's blacklist scope."""
    return blacklist_protocol, {"mode": "per-folder" if isinstance(blacklist_protocol, Mapping) else "single"}


def _serialized_facies_map(
    dataset: str,
    run: str,
    expected_shape: Sequence[int],
) -> np.ndarray | None:
    """Load an existing FF-UM facies map without rebuilding optical context."""

    if dataset != "ffum":
        return None
    path = _ffum_rig_path(run) / "facies.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            facies = np.asarray(payload["array"])
    except (OSError, KeyError, ValueError):
        return None
    if facies.shape[:2] != tuple(int(value) for value in expected_shape[:2]):
        return None
    return np.asarray(facies, dtype=np.int16)


def _serialized_phase_quiz_domain(
    dataset: str,
    run: str,
    expected_shape: Sequence[int],
) -> tuple[np.ndarray, str] | None:
    """Load an existing rig facies map without rebuilding optical context."""

    facies = _serialized_facies_map(dataset, run, expected_shape)
    if facies is None:
        return None
    rig = type("SerializedFaciesRig", (), {"facies": facies})()
    return _phase_quiz_domain(
        dataset,
        rig,
        np.ones(facies.shape[:2], dtype=bool),
    )


def _sanitize_imaging_protocol(value, *, cache_root=None, image_paths=()):
    from darsia.fluidflower.tools.native_protocols import imaging_protocol
    resolved = _resolve_protocol(value)
    if resolved is None:
        raise ValueError("An imaging protocol is required")
    return imaging_protocol(resolved, image_paths=image_paths, cache_root=cache_root)


def _ffum_clearwater_paths(run: str) -> list[Path]:
    manifest_path = str(FFUM_CLEARWATER_REFERENCE_MANIFEST).strip()
    if not manifest_path:
        raise FileNotFoundError(
            "FF-UM has no configured true clear-water reference. Set "
            "FFUM_CLEARWATER_REFERENCE_MANIFEST to a JSON mapping from run name "
            "to raw clear-water image paths; synthetic or blue baselines are not accepted."
        )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    values = manifest.get(str(run).lower()) or manifest.get(str(run).upper())
    if not values:
        raise FileNotFoundError(
            f"No true clear-water references are declared for FF-UM {run} in "
            f"{manifest_path}"
        )
    paths = [Path(value).resolve() for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FF-UM clear-water files: {missing}")
    return paths


def _is_colorchecker_correction(correction: Any) -> bool:
    return all(
        hasattr(correction, name)
        for name in ("correct_array", "_restrict_to_roi", "roi", "colorchecker")
    )


def _configure_ffum_colorchecker_refinement(rig: Any) -> dict[str, Any]:
    """Install refinement at the actual input stage of FF-UM's colour pipeline.

    MuseumRig applies an illumination field before its ColorChecker correction.
    The generic FF-AC installer assumes that ColorChecker is the first colour
    operation, so passing the raw shape-corrected baseline gives it the wrong
    reference image.  Walk the pipeline and install against the intermediate
    image that each ColorChecker correction really receives.
    """

    original_shape_baseline = rig.shape_corrected_baseline
    original_baseline = rig.baseline
    original_corrections = list(getattr(rig, "color_corrections", None) or [])
    if not any(_is_colorchecker_correction(item) for item in original_corrections):
        raise RuntimeError("FF-UM rig has no compatible ColorChecker correction")

    stage = original_shape_baseline.copy()
    configured: list[dict[str, Any]] = []
    try:
        for pipeline_index, correction in enumerate(original_corrections):
            if not _is_colorchecker_correction(correction):
                stage = correction(stage)
                continue
            rig.shape_corrected_baseline = stage
            rig.color_corrections = [correction]
            details = ffac_calibration._configure_runtime_colorchecker_refinement(rig)
            if not details.get("active"):
                raise RuntimeError("FF-UM ColorChecker refinement was not activated")
            configured.append(
                {
                    "pipeline_index": int(pipeline_index),
                    "input_stage": "after preceding colour corrections",
                    **details,
                }
            )
            stage = rig.baseline
    except Exception:
        rig.baseline = original_baseline
        raise
    finally:
        rig.shape_corrected_baseline = original_shape_baseline
        rig.color_corrections = original_corrections

    rig.baseline = stage
    return {
        "active": True,
        "strategy": "staged-uniform24",
        "correction_count": len(configured),
        "baseline_rebuilt_with_complete_pipeline": True,
        "stage_order_preserved": True,
        "corrections": configured,
    }


@contextmanager
def _ffum_source_context(
    run: str,
    *,
    rig: Any,
    experiment: Any,
    rig_path: Path,
    baseline_path: Path,
    contract: Mapping[str, Any],
    require_clearwater: bool,
) -> Iterator[RunSource]:
    """Shared tail of the FF-UM openers: refinement, clear water, frames, masks, RunSource."""

    profile = _profile()
    environment = optical_control._profile_environment(profile)
    with optical_control.rebuilt._temporary_environment(environment):
        refinement = _configure_ffum_colorchecker_refinement(rig)
        baseline = np.asarray(rig.baseline.img, dtype=np.float32)[..., :3]
        clearwater = None
        clearwater_contract = None
        if require_clearwater:
            clearwater, clearwater_contract = _read_corrected_clearwater(
                rig,
                _ffum_clearwater_paths(run),
                baseline.shape,
            )
        frames = _select_frames(
            experiment,
            _review_target_times(experiment, TARGET_TIMES_H["ffum"]),
            tolerance_seconds=None,
        )
        valid_mask = _valid_mask(rig, baseline.shape)
        phase_quiz_mask, phase_quiz_domain = _phase_quiz_domain(
            "ffum", rig, valid_mask
        )
        phase_evaluation_mask, phase_evaluation_domain = (
            _phase_classification_domain("ffum", rig, valid_mask)
        )
        yield RunSource(
            dataset="ffum",
            run=run,
            baseline=baseline,
            frames=frames,
            rig=rig,
            experiment=experiment,
            valid_mask=valid_mask,
            phase_quiz_mask=phase_quiz_mask,
            phase_quiz_domain=phase_quiz_domain,
            phase_evaluation_mask=phase_evaluation_mask,
            phase_evaluation_domain=phase_evaluation_domain,
            correction_contract={
                "profile": profile.name,
                "environment": environment,
                **dict(contract),
                "legacy_rig": str(rig_path),
                "rig_path": str(rig_path),
                "raw_baseline": str(baseline_path),
                "refinement": refinement,
                "legacy_ffum_calibration_method_used": False,
            },
            clearwater=clearwater,
            clearwater_contract=clearwater_contract,
        )


@contextmanager
def _open_ffum_project(
    root: Path, run: str, *, require_clearwater: bool = False
) -> Iterator[RunSource]:
    """Open an FF-UM run from a DarSIA-format project folder (rig built by DarSIA)."""

    import darsia
    import ffum

    config, config_paths = _ffum_project_config(root, run)
    data = config["data"]
    protocols = config["protocols"]
    folders = [Path(str(item)) for item in (data.get("folders") or [])]
    if data.get("folder"):
        folders.insert(0, Path(str(data["folder"])))
    if not folders:
        raise ValueError(f"FF-UM project run {run} configures no photograph folder")
    baseline_name = str(data["baseline"])
    image_suffix = Path(baseline_name).suffix
    image_paths: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Photograph folder of FF-UM run {run} not found: {folder}")
        image_paths.extend(
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix == image_suffix
            )
        )
    if not image_paths:
        raise FileNotFoundError(
            f"No {image_suffix} images found for FF-UM run {run} in {folders}"
        )
    baseline_candidates = [folder / baseline_name for folder in folders]
    baseline_path = next(
        (candidate for candidate in baseline_candidates if candidate.is_file()),
        baseline_candidates[0],
    )
    imaging_value = protocols["imaging"]
    if isinstance(imaging_value, Mapping):
        imaging_protocol: Any = {}
        imaging_protocol_contract: Any = {}
        for folder, value in imaging_value.items():
            protocol, protocol_contract = _sanitize_imaging_protocol(
                _project_protocol_path(root, value), image_paths=[p for p in image_paths if p.parent == Path(str(folder))]
            )
            imaging_protocol[Path(str(folder))] = protocol
            imaging_protocol_contract[str(folder)] = protocol_contract
    else:
        imaging_protocol, imaging_protocol_contract = _sanitize_imaging_protocol(
            _project_protocol_path(root, imaging_value), image_paths=image_paths
        )
    blacklist_value = protocols.get("blacklist")
    if isinstance(blacklist_value, Mapping):
        blacklist_protocol: Any = {}
        blacklist_contract: Any = {}
        for folder, value in blacklist_value.items():
            protocol, protocol_contract = _sanitize_blacklist_protocol(
                _project_protocol_path(root, value), image_paths=[p for p in image_paths if p.parent == Path(str(folder))]
            )
            if protocol is not None:
                blacklist_protocol[Path(str(folder))] = protocol
            blacklist_contract[str(folder)] = protocol_contract
    else:
        blacklist_protocol, blacklist_contract = _sanitize_blacklist_protocol(
            _project_protocol_path(root, blacklist_value), image_paths=image_paths
        )
    # Native DarSIA keeps blacklist identities scoped to their photo folder.
    blacklist_protocol, blacklist_merge = _single_blacklist_for_all_folders(blacklist_protocol)
    if isinstance(imaging_protocol, Mapping) and blacklist_protocol is not None and not isinstance(blacklist_protocol, Mapping):
        blacklist_protocol = {folder: blacklist_protocol for folder in imaging_protocol}
    experiment = darsia.ProtocolledExperiment(
        data=image_paths,
        imaging_protocol=imaging_protocol,
        injection_protocol=_project_protocol_path(root, protocols.get("injection")),
        pressure_temperature_protocol=_project_protocol_path(
            root, protocols.get("pressure_temperature")
        ),
        blacklist_protocol=blacklist_protocol,
    )
    rig_path = _ffum_project_rig_path(config)
    if not (rig_path / "meta_data.json").is_file():
        raise FileNotFoundError(
            f"FF-UM run {run} has no rig under the project folder yet: {rig_path} "
            "(build it with DarSIA's setup: --config common.toml runs/<run>.toml --all)"
        )
    reader_shim = _install_resize_correction_alias(darsia)
    rig = ffum.MuseumRig.load(rig_path)
    rig.baseline_path = baseline_path
    rig.shape_corrected_baseline = darsia.imread(
        baseline_path, transformations=rig.shape_corrections
    ).img_as(np.float32)
    rig.load_experiment(experiment)
    contract = {
        "source": (
            "FF-UM rig built by DarSIA from the project configuration (shape and "
            "illumination corrections) with the shared FF-AC uniform24 ColorChecker "
            "implementation"
        ),
        "rig_source": "darsia-project",
        "project_root": str(root),
        "config_paths": [str(path) for path in config_paths],
        "imaging_protocol": imaging_protocol_contract,
        "blacklist_protocol": blacklist_contract,
        "blacklist_merge": blacklist_merge,
        "reader_shim": reader_shim,
    }
    with _ffum_source_context(
        run,
        rig=rig,
        experiment=experiment,
        rig_path=rig_path,
        baseline_path=baseline_path,
        contract=contract,
        require_clearwater=require_clearwater,
    ) as source:
        yield source


@contextmanager
def _open_ffum(run: str, *, require_clearwater: bool = False) -> Iterator[RunSource]:
    project_root = _ffum_project_root()
    if project_root is not None:
        with _open_ffum_project(
            project_root, run, require_clearwater=require_clearwater
        ) as source:
            yield source
        return

    import darsia
    import ffum

    config_path = FFUM_WORKFLOW_ROOT / "config" / f"{run}.toml"
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    data = config["data"]
    protocols = config["protocols"]
    folder = Path(data["folder"])
    baseline_name = str(data["baseline"])
    image_suffix = Path(baseline_name).suffix
    image_paths = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix == image_suffix
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No {image_suffix} images found for FF-UM run {run} in {folder}"
        )
    imaging_protocol, imaging_protocol_contract = _sanitize_imaging_protocol(
        protocols["imaging"], image_paths=image_paths
    )
    experiment = darsia.ProtocolledExperiment(
        data=image_paths,
        imaging_protocol=imaging_protocol,
        injection_protocol=_resolve_protocol(protocols.get("injection")),
        pressure_temperature_protocol=_resolve_protocol(
            protocols.get("pressure_temperature")
        ),
        blacklist_protocol=_sanitize_blacklist_protocol(_resolve_protocol(protocols.get("blacklist")), image_paths=image_paths)[0],
    )
    rig_path = _ffum_serialized_rig_path(run)
    rig = ffum.MuseumRig.load(rig_path)
    baseline_path = folder / baseline_name
    rig.baseline_path = baseline_path
    rig.shape_corrected_baseline = darsia.imread(
        baseline_path, transformations=rig.shape_corrections
    ).img_as(np.float32)
    rig.load_experiment(experiment)
    contract = {
        "source": (
            "FF-UM serialized rig shape/illumination data with the shared "
            "FF-AC uniform24 ColorChecker implementation"
        ),
        "rig_source": "legacy-v1",
        "imaging_protocol": imaging_protocol_contract,
    }
    with _ffum_source_context(
        run,
        rig=rig,
        experiment=experiment,
        rig_path=rig_path,
        baseline_path=baseline_path,
        contract=contract,
        require_clearwater=require_clearwater,
    ) as source:
        yield source


def _open_source(
    dataset: str,
    run: str,
    *,
    require_clearwater: bool = False,
) -> Any:
    return (
        _open_ffac(run, require_clearwater=require_clearwater)
        if dataset == "ffac"
        else _open_ffum(run, require_clearwater=require_clearwater)
    )


def _load_models() -> dict[str, DualPhaseSignalModel]:
    models = {name: DualPhaseSignalModel.load(path) for name, path in MODEL_PATHS.items()}
    if tuple(models) != tuple(MODEL_ORDER) or len(models) != 2:
        raise RuntimeError(
            f"Model paths/order mismatch: {tuple(models)} != {tuple(MODEL_ORDER)}"
        )
    if any(model.schema != 2 for model in models.values()):
        raise RuntimeError("The transfer campaign requires schema-2 models")
    contracts = {
        name: OpticalFeatureContract.from_dict(model.feature_contract or {})
        for name, model in models.items()
    }
    superset_name = max(MODEL_ORDER, key=lambda name: len(models[name].feature_names))
    superset = models[superset_name]
    superset_contract = contracts[superset_name]
    for name, model in models.items():
        if tuple(superset.feature_names[: len(model.feature_names)]) != tuple(
            model.feature_names
        ):
            raise RuntimeError(
                f"{name} features are not a prefix of the packed superset contract"
            )
        contract = contracts[name]
        if (
            contract.generic_version != superset_contract.generic_version
            or contract.input_color_space != superset_contract.input_color_space
            or contract.include_dynamic != superset_contract.include_dynamic
        ):
            raise RuntimeError(f"{name} does not share the packed optical contract")
        if not math.isclose(
            float(model.texture_sigma),
            float(superset.texture_sigma),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{name} uses a different texture sigma")
    for name, model in models.items():
        provenance = dict(model.training_provenance)
        if not bool(provenance.get("deployment_allowed", False)):
            raise RuntimeError(f"{name} model is not deployment validated")
        if set(provenance.get("runs") or ()) != {"ac60", "ac61"}:
            raise RuntimeError(f"{name} training runs differ from AC60/AC61")
        expected = dict(provenance.get("code_sha256") or {})
        current = {
            "signal_model": _sha256(project_path('scripts/dual_phase_signal.py')),
            "optical_features": _sha256(
                project_path('scripts/optical_phase_features.py')
            ),
            "colorchecker": _sha256(
                project_path('scripts/colorchecker_refinement.py')
            ),
        }
        if contracts[name].reference_adapter != "none":
            current["clearwater_reference"] = _sha256(
                project_path('scripts/clearwater_phase_reference.py')
            )
        mismatches = {
            key: {"expected": expected.get(key), "actual": value}
            for key, value in current.items()
            if expected.get(key) != value
        }
        if mismatches:
            legacy_extension_only = (
                contracts[name].reference_adapter == "none"
                and set(mismatches).issubset({"signal_model", "optical_features"})
                and all(
                    expected.get(key) == digest
                    for key, digest in LEGACY_NO_REFERENCE_CODE_HASHES.items()
                )
            )
            if not legacy_extension_only:
                raise RuntimeError(f"{name} model code mismatch: {mismatches}")
    return models


def _load_aqueous_model() -> DualPhaseSignalModel | None:
    if AQUEOUS_MODEL_PATH is None:
        return None
    model = DualPhaseSignalModel.load(AQUEOUS_MODEL_PATH)
    provenance = dict(model.training_provenance)
    if model.schema != 2:
        raise RuntimeError("The aqueous-envelope model must use schema 2")
    if provenance.get("target_kind") != "aqueous_co2":
        raise RuntimeError("The aqueous-envelope model has the wrong target kind")
    if not bool(provenance.get("deployment_allowed", False)):
        raise RuntimeError("The aqueous-envelope model is not deployment validated")
    if set(provenance.get("runs") or ()) != {"ac60", "ac61"}:
        raise RuntimeError("The aqueous-envelope training runs differ from AC60/AC61")
    return model


def _predict_model_collection(
    models: Mapping[str, DualPhaseSignalModel],
    image: np.ndarray,
    baseline: np.ndarray,
    previous: np.ndarray | None,
    clearwater: np.ndarray | None = None,
    *,
    chunk_rows: int = 48,
) -> dict[str, np.ndarray]:
    names = tuple(models)
    if not names:
        return {}
    superset_name = max(names, key=lambda name: len(models[name].feature_names))
    superset = models[superset_name]
    superset_contract = OpticalFeatureContract.from_dict(
        superset.feature_contract or {}
    )
    for name in names:
        model = models[name]
        if tuple(superset.feature_names[: len(model.feature_names)]) != tuple(
            model.feature_names
        ):
            raise ValueError(
                f"{name} features are not a prefix of the shared optical workspace"
            )
        contract = OpticalFeatureContract.from_dict(model.feature_contract or {})
        if (
            contract.generic_version != superset_contract.generic_version
            or contract.input_color_space != superset_contract.input_color_space
            or contract.include_dynamic != superset_contract.include_dynamic
            or not math.isclose(
                float(model.texture_sigma),
                float(superset.texture_sigma),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{name} cannot share the packed optical workspace")
    workspace = prepare_optical_feature_workspace(
        image,
        baseline,
        previous_rgb=previous,
        clearwater_rgb=clearwater,
        texture_sigma=superset.texture_sigma,
        contract=superset_contract,
    )
    outputs = {
        name: np.empty(image.shape[:2], dtype=np.float32) for name in names
    }
    for start in range(0, image.shape[0], max(1, int(chunk_rows))):
        stop = min(image.shape[0], start + max(1, int(chunk_rows)))
        features = optical_feature_matrix(workspace, np.s_[start:stop, :])
        for name in names:
            model = models[name]
            outputs[name][start:stop] = model.predict_features(
                features[:, : len(model.feature_names)]
            ).reshape(stop - start, image.shape[1])
    return outputs


def _predict_pair(
    models: Mapping[str, DualPhaseSignalModel],
    image: np.ndarray,
    baseline: np.ndarray,
    previous: np.ndarray | None,
    clearwater: np.ndarray | None = None,
    *,
    chunk_rows: int = 48,
) -> dict[str, np.ndarray]:
    if set(models) != set(MODEL_ORDER):
        raise ValueError("Phase-candidate models differ from MODEL_ORDER")
    return _predict_model_collection(
        models,
        image,
        baseline,
        previous,
        clearwater,
        chunk_rows=chunk_rows,
    )


def _predict_single_model(
    model: DualPhaseSignalModel,
    image: np.ndarray,
    baseline: np.ndarray,
    previous: np.ndarray | None,
    clearwater: np.ndarray | None,
    *,
    chunk_rows: int,
) -> np.ndarray:
    return model.predict_score(
        image,
        baseline,
        previous_rgb=previous,
        clearwater_rgb=clearwater,
        chunk_rows=chunk_rows,
    )


def _aqueous_envelope_support(aqueous_probability: np.ndarray) -> np.ndarray:
    """Return soft nearby aqueous support for free-gas evidence."""

    aqueous = np.clip(np.asarray(aqueous_probability, dtype=np.float32), 0.0, 1.0)
    radius = max(
        3,
        int(round(AQUEOUS_ENVELOPE_RADIUS_FRACTION * min(aqueous.shape))),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    nearby = cv2.dilate(aqueous, kernel)
    return np.clip((nearby - np.float32(0.10)) / np.float32(0.50), 0.0, 1.0)


def _apply_aqueous_envelope(
    gas_probability: np.ndarray,
    aqueous_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support = _aqueous_envelope_support(aqueous_probability)
    gate = np.float32(0.02) + np.float32(0.98) * support
    gated = np.asarray(gas_probability, dtype=np.float32) * gate
    return np.asarray(gated, dtype=np.float32), support


def _coordinate_axes(reference: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return physical pixel-center axes for either FluidFlower rig."""

    image = np.asarray(reference.img)
    height, width = image.shape[:2]
    rows = np.column_stack(
        (np.arange(height, dtype=float) + 0.5, np.zeros(height, dtype=float))
    )
    columns = np.column_stack(
        (np.zeros(width, dtype=float), np.arange(width, dtype=float) + 0.5)
    )
    coordinate_system = getattr(reference, "coordinatesystem", None)
    if coordinate_system is None:
        coordinate_system = getattr(reference, "coordinate_system", None)
    if coordinate_system is None:
        raise RuntimeError("The rig baseline has no physical coordinate system")
    y_m = np.asarray(coordinate_system.coordinate(rows), dtype=float)[:, 1]
    x_m = np.asarray(coordinate_system.coordinate(columns), dtype=float)[:, 0]
    return x_m, y_m


def _injection_overlay_points(
    source: RunSource,
    *,
    canvas_wh: Sequence[int],
    content_box: Sequence[int],
    stages: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Map protocol injector locations into the native review canvas."""

    canvas_width, canvas_height = (int(value) for value in canvas_wh)
    content_x, content_y, content_width, content_height = (
        int(value) for value in content_box
    )
    image_height, image_width = source.baseline.shape[:2]
    if (content_width, content_height) != (image_width, image_height):
        raise ValueError("Native review content box does not match the analysis image")
    x_axis_m, y_axis_m = _coordinate_axes(source.rig.baseline)
    stage_rows = tuple(stages or _source_injection_stages(source))
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for stage in stage_rows:
        x_m = float(stage["x_m"])
        y_m = float(stage["y_m"])
        column = int(np.argmin(np.abs(x_axis_m - x_m)))
        row = int(np.argmin(np.abs(y_axis_m - y_m)))
        key = (column, row)
        item = grouped.setdefault(
            key,
            {
                "x": (content_x + column + 0.5) / canvas_width,
                "y": (content_y + row + 0.5) / canvas_height,
                "analysis_xy_px": [column, row],
                "x_m": x_m,
                "y_m": y_m,
                "active_windows_h": [],
                "coordinate_source": str(
                    stage.get(
                        "coordinate_source",
                        "protocol_and_rig_coordinate_system",
                    )
                ),
            },
        )
        if "protocol_x_m" in stage and "protocol_y_m" in stage:
            item["protocol_x_m"] = float(stage["protocol_x_m"])
            item["protocol_y_m"] = float(stage["protocol_y_m"])
        if "uncertainty_m" in stage:
            item["uncertainty_m"] = float(stage["uncertainty_m"])
        item["active_windows_h"].append(
            [float(stage["start_h"]), float(stage["end_h"])]
        )
    points = list(grouped.values())
    points.sort(
        key=lambda item: min(window[0] for window in item["active_windows_h"])
    )
    for index, item in enumerate(points, start=1):
        item["label"] = f"Injector {index}"
    return points


def _injection_stages(experiment: Any) -> tuple[dict[str, Any], ...]:
    """Build dataset-independent causal source stages from DarSIA protocol data."""

    protocol = experiment.injection_protocol.df
    rate_column = next(
        (name for name in ("rate_kg_s", "rate_sccm") if name in protocol.columns),
        None,
    )
    if rate_column is None:
        raise RuntimeError("The injection protocol has no usable rate column")
    active = protocol.loc[np.asarray(protocol[rate_column], dtype=float) > 0.0]
    if active.empty:
        raise RuntimeError("The injection protocol contains no active injection")
    origin = experiment.experiment_start
    grouped: dict[tuple[float, float], list[Any]] = {}
    for _, row in active.iterrows():
        key = (float(row["location_x"]), float(row["location_y"]))
        grouped.setdefault(key, []).append(row)
    stages: list[dict[str, Any]] = []
    for (x_m, y_m), rows in grouped.items():
        start = min(row["start"] for row in rows)
        end = max(row["end"] for row in rows)
        rate_segments: list[dict[str, float]] = []
        injected_mass_kg = 0.0
        active_duration_s = 0.0
        peak_rate_kg_s = 0.0
        for row in sorted(rows, key=lambda item: item["start"]):
            duration_s = max(0.0, float((row["end"] - row["start"]).total_seconds()))
            if rate_column == "rate_kg_s":
                rate_kg_s = float(row[rate_column])
            else:
                density_kg_m3 = float(row.get("density kg/m3", 1.872))
                rate_kg_s = float(row[rate_column]) * density_kg_m3 * 1e-6 / 60.0
            rate_kg_s = max(0.0, rate_kg_s)
            segment = {
                "start_h": float(
                    (row["start"] - origin).total_seconds() / 3600.0
                ),
                "end_h": float((row["end"] - origin).total_seconds() / 3600.0),
                "rate_kg_s": rate_kg_s,
            }
            rate_segments.append(segment)
            injected_mass_kg += rate_kg_s * duration_s
            active_duration_s += duration_s
            peak_rate_kg_s = max(peak_rate_kg_s, rate_kg_s)
        stages.append(
            {
                "x_m": x_m,
                "y_m": y_m,
                "start_h": float((start - origin).total_seconds() / 3600.0),
                "end_h": float((end - origin).total_seconds() / 3600.0),
                "rate_segments": rate_segments,
                "peak_rate_kg_s": peak_rate_kg_s,
                "mean_rate_kg_s": (
                    injected_mass_kg / active_duration_s
                    if active_duration_s > 0.0
                    else 0.0
                ),
                "injected_mass_kg": injected_mass_kg,
                "rate_source": rate_column,
            }
        )
    return tuple(sorted(stages, key=lambda stage: stage["start_h"]))


def _source_injection_stages(source: RunSource | Any) -> tuple[dict[str, Any], ...]:
    """Return temporal protocol stages with dataset-specific optical anchors.

    FF-UM keeps the original spreadsheet coordinates as provenance, while the
    image-derived source anchors are used by causal topology and review overlays.
    Other datasets continue to use their protocol coordinates unchanged.
    """

    stages: list[dict[str, Any]] = [
        dict(stage) for stage in _injection_stages(source.experiment)
    ]
    if str(getattr(source, "dataset", "")).lower() != "ffum":
        return tuple(stages)
    if len(stages) != len(FFUM_VISIBLE_SOURCE_ANCHORS_M):
        raise RuntimeError(
            "FF-UM visible-source anchors require exactly two injection stages; "
            f"received {len(stages)}"
        )
    for stage, anchor in zip(stages, FFUM_VISIBLE_SOURCE_ANCHORS_M, strict=True):
        stage["protocol_x_m"] = float(stage["x_m"])
        stage["protocol_y_m"] = float(stage["y_m"])
        stage["x_m"] = float(anchor["x_m"])
        stage["y_m"] = float(anchor["y_m"])
        stage["uncertainty_m"] = float(anchor["uncertainty_m"])
        stage["coordinate_source"] = str(anchor["coordinate_source"])
    return tuple(stages)


@dataclass
class CausalTopologyContext:
    """Memory-bounded causal history shared by one run's candidate models."""

    valid_small: np.ndarray
    x_small_m: np.ndarray
    y_small_m: np.ndarray
    injection_stages: tuple[dict[str, Any], ...]
    elapsed_hours: list[float]
    score_history: dict[str, list[np.ndarray]]
    co2_history: dict[str, list[np.ndarray]]
    facies_small: np.ndarray | None
    permeability_small: np.ndarray | None
    full_shape: tuple[int, int]
    target_wh: tuple[int, int]


def _resample_axis(axis: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(axis, dtype=float)
    positions = np.linspace(0.0, len(source) - 1.0, int(size))
    return np.interp(positions, np.arange(len(source), dtype=float), source)


def _causal_topology_context(
    source: RunSource,
    candidate_names: Sequence[str],
) -> CausalTopologyContext:
    topology_mask = (
        np.asarray(source.phase_evaluation_mask, dtype=bool)
        if source.phase_evaluation_mask is not None
        else np.asarray(source.valid_mask, dtype=bool)
    )
    height, width = topology_mask.shape
    target_wh = (
        max(2, int(round(width * CAUSAL_TOPOLOGY_SCALE))),
        max(2, int(round(height * CAUSAL_TOPOLOGY_SCALE))),
    )
    valid_small = cv2.resize(
        topology_mask.astype(np.float32),
        target_wh,
        interpolation=cv2.INTER_AREA,
    ) >= 0.75
    x_m, y_m = _coordinate_axes(source.rig.baseline)
    raw_facies = getattr(source.rig, "facies", None)
    facies_array = (
        None
        if raw_facies is None
        else np.asarray(getattr(raw_facies, "img", raw_facies))
    )
    facies_small = (
        None
        if facies_array is None or facies_array.shape != source.valid_mask.shape
        else cv2.resize(
            facies_array.astype(np.int32),
            target_wh,
            interpolation=cv2.INTER_NEAREST,
        )
    )
    raw_permeability = getattr(source.rig, "permeability", None)
    permeability_array = (
        None
        if raw_permeability is None
        else np.asarray(getattr(raw_permeability, "img", raw_permeability))
    )
    permeability_small = (
        None
        if (
            permeability_array is None
            or permeability_array.shape != source.valid_mask.shape
        )
        else cv2.resize(
            permeability_array.astype(np.float32),
            target_wh,
            interpolation=cv2.INTER_AREA,
        )
    )
    return CausalTopologyContext(
        valid_small=valid_small,
        x_small_m=_resample_axis(x_m, target_wh[0]),
        y_small_m=_resample_axis(y_m, target_wh[1]),
        injection_stages=_source_injection_stages(source),
        elapsed_hours=[],
        score_history={name: [] for name in candidate_names},
        co2_history={name: [] for name in candidate_names},
        facies_small=facies_small,
        permeability_small=permeability_small,
        full_shape=(height, width),
        target_wh=target_wh,
    )


def _apply_causal_topology(
    probability: np.ndarray,
    co2_probability: np.ndarray,
    *,
    candidate_name: str,
    elapsed_h: float,
    context: CausalTopologyContext,
) -> tuple[np.ndarray, CausalGasGateFrame]:
    """Require current aqueous-supported evidence plus source/history continuity."""

    if not context.elapsed_hours or not math.isclose(
        context.elapsed_hours[-1], float(elapsed_h), rel_tol=0.0, abs_tol=1e-9
    ):
        context.elapsed_hours.append(float(elapsed_h))
    history = context.score_history[candidate_name]
    history.append(
        cv2.resize(
            np.asarray(probability, dtype=np.float32),
            context.target_wh,
            interpolation=cv2.INTER_AREA,
        )
    )
    co2_history = context.co2_history[candidate_name]
    co2_history.append(
        cv2.resize(
            np.asarray(co2_probability, dtype=np.float32),
            context.target_wh,
            interpolation=cv2.INTER_AREA,
        )
    )
    if len(history) != len(context.elapsed_hours):
        raise RuntimeError("Candidate causal histories are out of step")
    if len(co2_history) != len(context.elapsed_hours):
        raise RuntimeError("Candidate CO2 histories are out of step")
    candidate_settings = CAUSAL_GAS_SETTINGS_BY_CANDIDATE.get(candidate_name)
    settings = (
        candidate_settings
        if candidate_settings is not None
        else CAUSAL_GAS_SETTINGS
    )
    unit_saturation = [np.ones_like(score, dtype=np.float32) for score in history]
    frames = apply_causal_gas_gate(
        unit_saturation,
        history,
        context.valid_small,
        x_coordinates_m=context.x_small_m,
        y_coordinates_m=context.y_small_m,
        elapsed_hours=context.elapsed_hours,
        injection_stages=context.injection_stages,
        facies_labels=(
            context.facies_small
            if (
                candidate_settings is not None
                and candidate_settings.reject_seal_dominant_components
            )
            else None
        ),
        settings=settings,
        co2_scores=co2_history,
        permeability=(
            context.permeability_small if candidate_settings is not None else None
        ),
    )
    latest = frames[-1]
    gate = cv2.resize(
        latest.saturation,
        (context.full_shape[1], context.full_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    gated = np.asarray(probability, dtype=np.float32) * np.clip(gate, 0.0, 1.0)
    return np.asarray(gated, dtype=np.float32), latest


def _shift_probability(probability: np.ndarray, logit_shift: float) -> np.ndarray:
    values = np.clip(
        np.asarray(probability, dtype=np.float32),
        np.float32(1e-6),
        np.float32(1.0 - 1e-6),
    )
    logits = np.log(values / (np.float32(1.0) - values)) - np.float32(
        logit_shift
    )
    logits = np.clip(logits, -30.0, 30.0)
    return np.asarray(
        np.float32(1.0) / (np.float32(1.0) + np.exp(-logits)),
        dtype=np.float32,
    )


def _known_negative_calibration(
    controls: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
    *,
    quantile: float,
    target_probability: float,
    maximum_logit_shift: float,
    method: str,
) -> dict[str, Any]:
    quantiles = {
        kind: float(
            np.quantile(
                np.asarray(probability, dtype=np.float32)[valid_mask],
                quantile,
            )
        )
        for kind, probability in controls.items()
    }
    raw_quantile = max(quantiles.values(), default=0.0)
    clipped_raw = float(np.clip(raw_quantile, 1e-6, 1.0 - 1e-6))
    target = float(target_probability)
    raw_logit = math.log(clipped_raw / (1.0 - clipped_raw))
    target_logit = math.log(target / (1.0 - target))
    shift = max(0.0, raw_logit - target_logit)
    return {
        "method": method,
        "quantile": float(quantile),
        "target_probability": target,
        "raw_quantile_by_control": quantiles,
        "raw_quantile_max": raw_quantile,
        "logit_shift": float(shift),
        "maximum_allowed_logit_shift": float(maximum_logit_shift),
        "passed": bool(shift <= maximum_logit_shift),
    }


def _aqueous_negative_calibration(
    controls: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    return _known_negative_calibration(
        controls,
        valid_mask,
        quantile=AQUEOUS_NEGATIVE_CALIBRATION_QUANTILE,
        target_probability=AQUEOUS_NEGATIVE_CALIBRATION_TARGET,
        maximum_logit_shift=MAX_AQUEOUS_LOGIT_SHIFT,
        method="known-negative-co2-quantile-logit-shift-v2",
    )


def _gas_negative_calibration(
    controls: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    return _known_negative_calibration(
        controls,
        valid_mask,
        quantile=GAS_NEGATIVE_CALIBRATION_QUANTILE,
        target_probability=GAS_NEGATIVE_CALIBRATION_TARGET,
        maximum_logit_shift=MAX_GAS_LOGIT_SHIFT,
        method="known-negative-conditional-gas-quantile-logit-shift-v1",
    )


def _negative_floor_calibration(
    controls: Mapping[str, np.ndarray],
    valid_mask: np.ndarray,
    *,
    quantile: float,
    maximum_floor_probability: float,
    method: str,
) -> dict[str, Any]:
    quantiles = {
        kind: float(
            np.quantile(
                np.asarray(score, dtype=np.float32)[valid_mask],
                quantile,
            )
        )
        for kind, score in controls.items()
    }
    floor = max(quantiles.values(), default=0.0)
    return {
        "method": method,
        "quantile": float(quantile),
        "raw_quantile_by_control": quantiles,
        "negative_floor_probability": float(floor),
        "maximum_allowed_negative_floor_probability": float(
            maximum_floor_probability
        ),
        "passed": bool(floor <= maximum_floor_probability),
        "output_semantics": "normalized optical evidence, not probability",
    }


def _apply_negative_calibration(
    probability: np.ndarray,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    if "negative_floor_probability" in calibration:
        floor = np.float32(calibration["negative_floor_probability"])
        denominator = max(np.float32(1e-6), np.float32(1.0) - floor)
        return np.asarray(
            np.clip(
                (np.asarray(probability, dtype=np.float32) - floor) / denominator,
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )
    return _shift_probability(
        probability,
        float(calibration.get("logit_shift", 0.0)),
    )


def _model_digest(model: DualPhaseSignalModel) -> str:
    return str(model.to_dict()["model_sha256"])


def _aqueous_models_by_candidate(
    models: Mapping[str, DualPhaseSignalModel],
    aqueous_model: (
        DualPhaseSignalModel | Mapping[str, DualPhaseSignalModel] | None
    ),
) -> dict[str, DualPhaseSignalModel | None]:
    if aqueous_model is None:
        return {name: None for name in models}
    if isinstance(aqueous_model, DualPhaseSignalModel):
        return {name: aqueous_model for name in models}
    resolved = dict(aqueous_model)
    if set(resolved) != set(models):
        raise ValueError(
            "Per-candidate CO2-presence models must match gas candidates: "
            f"{sorted(resolved)} != {sorted(models)}"
        )
    if not all(isinstance(model, DualPhaseSignalModel) for model in resolved.values()):
        raise TypeError("Every per-candidate CO2-presence model must be loadable")
    return resolved


def _composite_candidate_digest(
    gas_model: DualPhaseSignalModel,
    aqueous_model: DualPhaseSignalModel | None,
) -> str:
    gas_digest = _model_digest(gas_model)
    if aqueous_model is None:
        return gas_digest
    payload = f"hierarchical-v1|{gas_digest}|{_model_digest(aqueous_model)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_asset_fingerprint(candidate_id: str, digest: str) -> str:
    """Return an asset identity that remains unique for equal model weights.

    Two review candidates can deliberately share the same learned model while
    differing in downstream physics. Their model/parameter digest is then
    identical, so it cannot by itself identify candidate-specific phase maps.
    """

    payload = f"review-asset-v2|{candidate_id}|{digest}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _blind_labels(run: str, candidate_ids: Mapping[str, str]) -> dict[str, str]:
    ordered = [candidate_ids[name] for name in MODEL_ORDER]
    fingerprint = "|".join((BLIND_SEED, run, *ordered))
    swap = hashlib.sha256(fingerprint.encode("utf-8")).digest()[0] & 1
    labels = ("B", "A") if swap else ("A", "B")
    return dict(zip(MODEL_ORDER, labels, strict=True))


def _candidate_jobs(
    dataset: str,
    runs: Sequence[str],
    models: Mapping[str, DualPhaseSignalModel],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for run in runs:
        for order, name in enumerate(MODEL_ORDER, start=1):
            digest = _model_digest(models[name])
            jobs.append(
                {
                    "job_id": f"{dataset}_{run}_{name}",
                    "run": run,
                    "role": name,
                    "family": "phase-transfer-schema2",
                    "variant": f"{order:02d}_{name}_schema2",
                    "grid_order": order,
                    "candidate_id": f"{dataset}__{run}__{name}__{digest[:16]}",
                    "quality_flags": {
                        "dataset": dataset,
                        "external_transfer": dataset == "ffum",
                        "target_used_for_fitting": False,
                        "five_class_model_active": False,
                    },
                }
            )
    return jobs


def _candidate_path(output_root: Path, job: Mapping[str, Any]) -> Path:
    return output_root / "postflight" / str(job["variant"]) / str(job["run"])


def _native_dimensions(shape: Sequence[int]) -> tuple[int, int]:
    return review._native_review_dimensions(shape)


def _probability_overlay(
    probability: np.ndarray,
    image_shape: Sequence[int],
    *,
    aqueous_probability: np.ndarray | None = None,
    width: int,
    height: int,
    preserve_native_resolution: bool,
    normalized_evidence: bool = False,
) -> np.ndarray:
    (
        layout_scale,
        _header,
        _margin,
        x0,
        y0,
        target_width,
        target_height,
    ) = review_movie._review_geometry(
        image_shape,
        width=width,
        height=height,
        preserve_native_resolution=preserve_native_resolution,
    )
    field = np.asarray(probability, dtype=np.float32)
    if field.shape != (target_height, target_width):
        field = cv2.resize(
            field, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
    local = np.zeros((target_height, target_width, 4), dtype=np.uint8)
    colors = (
        (255, 205, 70, 255),
        (245, 95, 55, 255),
        (210, 45, 160, 255),
    )
    styles: list[tuple[str, tuple[int, ...], int]] = []
    for index, (level, color) in enumerate(zip(PROBABILITY_LEVELS, colors, strict=True)):
        thickness = max(1, int(round((2 + (index == 2)) * layout_scale)))
        review_movie._draw_level(local, field, level, color, thickness)
        symbol = "E" if normalized_evidence else "P"
        styles.append((f"{symbol}(free gas) = {level:g}", color, thickness))
    if aqueous_probability is not None:
        aqueous_field = np.asarray(aqueous_probability, dtype=np.float32)
        if aqueous_field.shape != (target_height, target_width):
            aqueous_field = cv2.resize(
                aqueous_field,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        aqueous_color = (55, 225, 170, 255)
        aqueous_thickness = max(1, int(round(2 * layout_scale)))
        review_movie._draw_level(
            local,
            aqueous_field,
            0.50,
            aqueous_color,
            aqueous_thickness,
        )
        symbol = "E" if normalized_evidence else "P"
        styles.append(
            (f"{symbol}(aqueous CO2) = 0.5", aqueous_color, aqueous_thickness)
        )
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[y0 : y0 + target_height, x0 : x0 + target_width] = local
    legend_width = max(1, (width - int(round(48 * layout_scale))) // len(styles))
    for index, (label, color, thickness) in enumerate(styles):
        left = int(round(24 * layout_scale)) + index * legend_width
        y = int(round(103 * layout_scale))
        cv2.line(
            canvas,
            (left, y),
            (left + int(round(25 * layout_scale)), y),
            color,
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (left + int(round(34 * layout_scale)), int(round(109 * layout_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43 * layout_scale,
            (225, 230, 234, 255),
            max(1, int(round(layout_scale))),
            cv2.LINE_AA,
        )
    return canvas


def _filter_probability_overlay(
    overlay_rgba: np.ndarray,
    phase: str,
) -> np.ndarray:
    """Keep only gas or aqueous contour pixels from a transparent overlay."""

    if phase not in {"gas", "aqueous"}:
        raise ValueError(f"Unsupported contour phase: {phase!r}")
    values = np.asarray(overlay_rgba, dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("Probability overlays must be RGBA images")
    rgb = values[..., :3].astype(np.uint16)
    alpha = values[..., 3]
    total = np.sum(rgb, axis=2, dtype=np.uint16)
    red = rgb[..., 0]
    green = rgb[..., 1]
    if phase == "gas":
        keep = (alpha > 0) & (5 * red > 2 * total)
    else:
        keep = (alpha > 0) & (5 * green > 2 * total) & (4 * red < total)
    output = np.zeros_like(values)
    output[keep] = values[keep]
    return output


def _write_probability_overlay_assets(path: Path, overlay_rgba: np.ndarray) -> None:
    """Write combined and phase-specific native contour layers."""

    _write_png(path, overlay_rgba)
    for phase in ("gas", "aqueous"):
        _write_png(
            path.parent / phase / path.name,
            _filter_probability_overlay(overlay_rgba, phase),
        )


def _phase_fill_masks(
    gas_probability: np.ndarray,
    aqueous_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assign each visible-CO2 pixel to one hard display class."""

    gas = np.clip(np.asarray(gas_probability, dtype=np.float32), 0.0, 1.0)
    aqueous = np.clip(
        np.asarray(aqueous_probability, dtype=np.float32), 0.0, 1.0
    )
    if gas.shape != aqueous.shape:
        raise ValueError("Gas and aqueous fields must have matching shapes")
    co2_evidence = np.clip(gas + aqueous, 0.0, 1.0)
    active = co2_evidence >= PHASE_FILL_MIN_CO2_EVIDENCE
    gas_share = np.divide(
        gas,
        np.maximum(gas + aqueous, np.finfo(np.float32).eps),
        out=np.zeros_like(gas),
    )
    aqueous_mask = active & (gas_share < PHASE_FILL_AQUEOUS_MAX_GAS_SHARE)
    gas_mask = active & (gas_share >= PHASE_FILL_GAS_MIN_GAS_SHARE)
    return {
        "aqueous": aqueous_mask,
        "mixed": active & ~(aqueous_mask | gas_mask),
        "gas": gas_mask,
    }


def _phase_fill_overlay(
    gas_probability: np.ndarray,
    image_shape: Sequence[int],
    *,
    aqueous_probability: np.ndarray | None,
    width: int,
    height: int,
    preserve_native_resolution: bool,
) -> np.ndarray:
    """Render one mutually exclusive phase class per visible CO2 pixel."""

    (
        layout_scale,
        _header,
        _margin,
        x0,
        y0,
        target_width,
        target_height,
    ) = review_movie._review_geometry(
        image_shape,
        width=width,
        height=height,
        preserve_native_resolution=preserve_native_resolution,
    )
    gas = np.clip(np.asarray(gas_probability, dtype=np.float32), 0.0, 1.0)
    if gas.shape != (target_height, target_width):
        gas = cv2.resize(
            gas, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
    if aqueous_probability is None:
        aqueous = np.zeros_like(gas)
    else:
        aqueous = np.clip(
            np.asarray(aqueous_probability, dtype=np.float32), 0.0, 1.0
        )
        if aqueous.shape != (target_height, target_width):
            aqueous = cv2.resize(
                aqueous,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )

    masks = _phase_fill_masks(gas, aqueous)
    local = np.zeros((target_height, target_width, 4), dtype=np.uint8)
    for phase, color in PHASE_FILL_COLORS.items():
        local[masks[phase]] = color
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    canvas[y0 : y0 + target_height, x0 : x0 + target_width] = local

    legend = (
        ("Aqueous-dominant", PHASE_FILL_COLORS["aqueous"]),
        ("Mixed / uncertain", PHASE_FILL_COLORS["mixed"]),
        ("Free-gas-dominant", PHASE_FILL_COLORS["gas"]),
    )
    legend_width = max(1, (width - int(round(48 * layout_scale))) // len(legend))
    for index, (label, color) in enumerate(legend):
        left = int(round(24 * layout_scale)) + index * legend_width
        top = int(round(94 * layout_scale))
        size = max(8, int(round(16 * layout_scale)))
        cv2.rectangle(
            canvas,
            (left, top),
            (left + size, top + size),
            color,
            thickness=-1,
        )
        cv2.putText(
            canvas,
            label,
            (left + size + int(round(8 * layout_scale)), top + size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43 * layout_scale,
            (225, 230, 234, 255),
            max(1, int(round(layout_scale))),
            cv2.LINE_AA,
        )
    return canvas


def _filter_phase_fill_overlay(
    overlay_rgba: np.ndarray,
    phase: str,
) -> np.ndarray:
    """Keep one hard display class from a transparent phase-fill overlay."""

    normalized = str(phase or "all").casefold()
    if normalized not in PHASE_FILL_COLORS:
        raise ValueError(f"Unsupported phase-fill class: {phase!r}")
    values = np.asarray(overlay_rgba, dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("Phase-fill overlays must be RGBA images")
    expected = np.asarray(PHASE_FILL_COLORS[normalized][:3], dtype=np.uint8)
    keep = (values[..., 3] > 0) & np.all(values[..., :3] == expected, axis=2)
    output = np.zeros_like(values)
    output[keep] = values[keep]
    return output


def _write_phase_fill_overlay_assets(path: Path, overlay_rgba: np.ndarray) -> None:
    """Write combined and class-specific filled phase-map layers."""

    _write_png(path, overlay_rgba)
    for phase in PHASE_FILL_COLORS:
        _write_png(
            path.parent / phase / path.name,
            _filter_phase_fill_overlay(overlay_rgba, phase),
        )


def _backfill_probability_overlay_assets(
    review_root: Path,
    run: str,
) -> dict[str, int]:
    """Create missing gas/aqueous layers for an existing review package."""

    combined_paths = sorted(
        (review_root / "native_overlays" / run).glob("*/frame_*.png")
    )
    written = 0
    existing = 0
    unreadable = 0
    for path in combined_paths:
        overlay_bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if (
            overlay_bgra is None
            or overlay_bgra.ndim != 3
            or overlay_bgra.shape[2] != 4
        ):
            unreadable += 1
            continue
        overlay = cv2.cvtColor(overlay_bgra, cv2.COLOR_BGRA2RGBA)
        for phase in ("gas", "aqueous"):
            target = path.parent / phase / path.name
            if target.is_file():
                existing += 1
                continue
            _write_png(target, _filter_probability_overlay(overlay, phase))
            written += 1
    return {
        "combined_overlay_count": len(combined_paths),
        "phase_overlay_written": written,
        "phase_overlay_already_present": existing,
        "unreadable_overlay_count": unreadable,
    }


def _filled_contour_evidence(line_mask: np.ndarray) -> np.ndarray:
    """Convert sparse plotted contour lines into a bounded interior score."""

    lines = np.asarray(line_mask, dtype=np.uint8)
    if not np.any(lines):
        return np.zeros(lines.shape, dtype=np.float32)
    closed = cv2.morphologyEx(
        lines,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    contours, _hierarchy = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros(lines.shape, dtype=np.uint8)
    minimum_area = max(16.0, 0.00001 * float(lines.size))
    retained = [
        contour for contour in contours if cv2.contourArea(contour) >= minimum_area
    ]
    if retained:
        cv2.drawContours(filled, retained, -1, 1, thickness=-1)
    filled[lines > 0] = 1
    return cv2.GaussianBlur(
        filled.astype(np.float32),
        (0, 0),
        sigmaX=1.2,
    )


def _overlay_phase_evidence(
    path: Path,
    *,
    content_box: Sequence[int],
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Recover plotted gas/aqueous hypotheses from an existing RGBA overlay."""

    overlay_bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if overlay_bgra is None or overlay_bgra.ndim != 3 or overlay_bgra.shape[2] != 4:
        zeros = np.zeros((size[1], size[0]), dtype=np.float32)
        return zeros, None
    overlay = cv2.cvtColor(overlay_bgra, cv2.COLOR_BGRA2RGBA)
    content_x, content_y, content_width, content_height = map(int, content_box)
    local = overlay[
        content_y : content_y + content_height,
        content_x : content_x + content_width,
    ]
    if local.shape[:2] != (content_height, content_width):
        zeros = np.zeros((size[1], size[0]), dtype=np.float32)
        return zeros, None
    local = cv2.resize(local, size, interpolation=cv2.INTER_AREA)
    gas_lines = _filter_probability_overlay(local, "gas")[..., 3] > 0
    aqueous_lines = _filter_probability_overlay(local, "aqueous")[..., 3] > 0
    gas = _filled_contour_evidence(gas_lines)
    aqueous = (
        _filled_contour_evidence(aqueous_lines)
        if np.any(aqueous_lines)
        else None
    )
    return gas, aqueous


def _compose(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    alpha = np.asarray(overlay[..., 3:4], dtype=np.float32) / 255.0
    return np.clip(
        (1.0 - alpha) * np.asarray(base, dtype=np.float32)
        + alpha * np.asarray(overlay[..., :3], dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)


def _plain_frame(
    image: np.ndarray,
    *,
    run: str,
    index: int,
    count: int,
    elapsed_h: float,
    width: int,
    height: int,
    native: bool,
    normalized_evidence: bool = False,
) -> np.ndarray:
    zeros = np.zeros(image.shape[:2], dtype=np.float32)
    return review_movie.render_contour_frame(
        image=image,
        concentration_aq=zeros,
        saturation_g=zeros,
        run=run,
        variant="Blind schema-2 phase transfer",
        frame_index=index,
        frame_count=count,
        elapsed_h=elapsed_h,
        injected_mass_kg=0.0,
        detected_total_mass_kg=0.0,
        detected_gas_mass_kg=0.0,
        detected_aqueous_mass_kg=0.0,
        width=width,
        height=height,
        gas_fill_alpha=0.0,
        show_aqueous=False,
        show_gas=False,
        preserve_native_resolution=native,
        header_title=(
            f"{run.upper()} | frame {index + 1}/{count} | t={elapsed_h:.3f} h | "
            "radiometrically corrected native input"
        ),
        header_mass=(
            (
                "Contours show normalized free-gas and aqueous-CO2 optical "
                "evidence; no mass or five-class phase model is applied"
            )
            if normalized_evidence
            else (
                "Contours show free-gas and aqueous-CO2 probability; no mass or "
                "five-class phase model is applied"
            )
        ),
        status_text="",
    )


def _probability_record(
    elapsed_h: float,
    probability: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(probability, dtype=np.float32)[valid_mask]
    if values.size == 0:
        raise RuntimeError("The phase-transfer mask is empty")
    mean = float(np.mean(values, dtype=np.float64))
    gas_mask = np.asarray(probability >= 0.50, dtype=np.uint8)
    gas_mask[~valid_mask] = 0
    component_count, _labels, component_stats, _centroids = (
        cv2.connectedComponentsWithStats(gas_mask, connectivity=8)
    )
    component_areas = np.asarray(component_stats[1:, cv2.CC_STAT_AREA], dtype=np.int64)
    minimum_coherent_area = max(16, int(round(values.size * 1e-5)))
    isolated_area = int(np.sum(component_areas[component_areas < minimum_coherent_area]))
    return {
        "elapsed_h": float(elapsed_h),
        # Compatibility fields keep timeline navigation active in the existing UI.
        # They encode a unitless model score in grams, not physical mass, and are never used
        # by calibration or selection.
        "injected_mass_kg": 0.001,
        "detected_total_mass_kg": mean / 1000.0,
        "detected_gas_mass_kg": mean / 1000.0,
        "detected_aqueous_mass_kg": 0.0,
        "gas_probability_mean": mean,
        "gas_probability_p50": float(np.quantile(values, 0.50)),
        "gas_probability_p90": float(np.quantile(values, 0.90)),
        "gas_probability_p95": float(np.quantile(values, 0.95)),
        "gas_probability_p99": float(np.quantile(values, 0.99)),
        "support_fraction_p025": float(np.mean(values >= 0.25)),
        "support_fraction_p050": float(np.mean(values >= 0.50)),
        "support_fraction_p075": float(np.mean(values >= 0.75)),
        "gas_component_count_p050": int(max(0, component_count - 1)),
        "largest_gas_component_fraction_p050": (
            0.0
            if component_areas.size == 0
            else float(np.max(component_areas) / values.size)
        ),
        "isolated_gas_fraction_p050": float(isolated_area / values.size),
        "coherent_component_minimum_area_px": int(minimum_coherent_area),
    }


def _phase_evaluation_mask(
    source: RunSource,
    *,
    hierarchical_phase: bool,
) -> np.ndarray:
    mask = (
        source.phase_evaluation_mask
        if hierarchical_phase and source.phase_evaluation_mask is not None
        else source.phase_quiz_mask
        if hierarchical_phase
        else source.valid_mask
    )
    return np.asarray(mask, dtype=bool)


def _negative_control_input_contract(
    models: Mapping[str, DualPhaseSignalModel],
    source: RunSource,
    aqueous_model: (
        DualPhaseSignalModel | Mapping[str, DualPhaseSignalModel] | None
    ),
    *,
    hierarchical_phase: bool,
) -> dict[str, Any]:
    aqueous_models = _aqueous_models_by_candidate(models, aqueous_model)
    evaluation_mask = _phase_evaluation_mask(
        source,
        hierarchical_phase=hierarchical_phase,
    )
    return {
        "schema": 1,
        "dataset": source.dataset,
        "run": source.run,
        "hierarchical_phase": bool(hierarchical_phase),
        "baseline_array_sha256": _array_sha256(source.baseline),
        "clearwater_array_sha256": (
            None if source.clearwater is None else _array_sha256(source.clearwater)
        ),
        "evaluation_mask_sha256": _array_sha256(evaluation_mask),
        "evaluation_domain": (
            source.phase_evaluation_domain
            if hierarchical_phase
            else "valid_analysis_image"
        ),
        "gas_model_sha256_by_candidate": {
            name: _model_digest(model) for name, model in models.items()
        },
        "co2_model_sha256_by_candidate": {
            name: None if model is None else _model_digest(model)
            for name, model in aqueous_models.items()
        },
        "calibration_contract": {
            "co2": "known-negative-co2-evidence-floor-v2-porous-domain",
            "gas": (
                "known-negative-conditional-gas-evidence-floor-v3-baseline-only"
                if hierarchical_phase
                else "known-negative-conditional-gas-evidence-floor-v2-porous-domain"
            ),
            "quantile": AQUEOUS_NEGATIVE_CALIBRATION_QUANTILE,
            "co2_maximum_floor": HIERARCHICAL_CO2_NEGATIVE_FLOOR_MAX,
            "gas_maximum_floor": HIERARCHICAL_GAS_NEGATIVE_FLOOR_MAX,
        },
    }


def _negative_control_preflight_is_reusable(
    report: Mapping[str, Any],
    models: Mapping[str, DualPhaseSignalModel],
    source: RunSource,
    aqueous_model: (
        DualPhaseSignalModel | Mapping[str, DualPhaseSignalModel] | None
    ),
    *,
    hierarchical_phase: bool,
) -> bool:
    if not bool(report.get("passed")):
        return False
    expected = _negative_control_input_contract(
        models,
        source,
        aqueous_model,
        hierarchical_phase=hierarchical_phase,
    )
    return report.get("input_contract") == expected


def _negative_control_preflight(
    models: Mapping[str, DualPhaseSignalModel],
    source: RunSource,
    aqueous_model: (
        DualPhaseSignalModel | Mapping[str, DualPhaseSignalModel] | None
    ) = None,
    *,
    hierarchical_phase: bool = False,
) -> dict[str, Any]:
    """Require known no-gas observations to remain effectively gas-free."""

    controls: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        ("pre_injection_baseline_initial", source.baseline, None),
        ("pre_injection_baseline_static", source.baseline, source.baseline),
    ]
    if source.clearwater is not None:
        controls.append(("true_clearwater", source.clearwater, source.baseline))
    evaluation_mask = _phase_evaluation_mask(
        source,
        hierarchical_phase=hierarchical_phase,
    )
    aqueous_models = _aqueous_models_by_candidate(models, aqueous_model)
    raw_aqueous_controls: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in models
    }
    for name, model in aqueous_models.items():
        if model is None:
            continue
        for kind, current, previous in controls:
            raw_aqueous_controls[name][kind] = _predict_single_model(
                model,
                current,
                source.baseline,
                previous,
                source.clearwater,
                chunk_rows=32 if source.dataset == "ffum" else 64,
            )
    aqueous_calibrations = {
        name: (
            _negative_floor_calibration(
                {
                    kind: probability
                    for kind, probability in values.items()
                    if kind != "true_clearwater"
                },
                evaluation_mask,
                quantile=AQUEOUS_NEGATIVE_CALIBRATION_QUANTILE,
                maximum_floor_probability=HIERARCHICAL_CO2_NEGATIVE_FLOOR_MAX,
                method="known-negative-co2-evidence-floor-v2-porous-domain",
            )
            if hierarchical_phase
            else _aqueous_negative_calibration(values, evaluation_mask)
        )
        for name, values in raw_aqueous_controls.items()
    }
    raw_gas_controls: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in models
    }
    for kind, current, previous in controls:
        predictions = _predict_pair(
            models,
            current,
            source.baseline,
            previous,
            source.clearwater,
            chunk_rows=32 if source.dataset == "ffum" else 64,
        )
        for name, probability in predictions.items():
            raw_gas_controls[name][kind] = probability
    gas_calibrations = {
        name: (
            _negative_floor_calibration(
                {
                    kind: probability
                    for kind, probability in values.items()
                    if kind != "true_clearwater"
                },
                evaluation_mask,
                quantile=GAS_NEGATIVE_CALIBRATION_QUANTILE,
                maximum_floor_probability=HIERARCHICAL_GAS_NEGATIVE_FLOOR_MAX,
                method=(
                    "known-negative-conditional-gas-evidence-floor-v3-baseline-only"
                ),
            )
            if hierarchical_phase
            else {
                "method": "disabled-for-non-hierarchical-campaign",
                "quantile": GAS_NEGATIVE_CALIBRATION_QUANTILE,
                "target_probability": GAS_NEGATIVE_CALIBRATION_TARGET,
                "raw_quantile_by_control": {},
                "raw_quantile_max": 0.0,
                "logit_shift": 0.0,
                "maximum_allowed_logit_shift": MAX_GAS_LOGIT_SHIFT,
                "passed": True,
            }
        )
        for name, values in raw_gas_controls.items()
    }
    rows: list[dict[str, Any]] = []
    reported_aqueous_models: set[str] = set()
    for kind, current, previous in controls:
        for name in MODEL_ORDER:
            resolved_aqueous_model = aqueous_models[name]
            aqueous_calibration = aqueous_calibrations[name]
            aqueous_probability = (
                None
                if resolved_aqueous_model is None
                else _apply_negative_calibration(
                    raw_aqueous_controls[name][kind],
                    aqueous_calibration,
                )
            )
            if aqueous_probability is not None:
                aqueous_metrics = _probability_record(
                    0.0,
                    aqueous_probability,
                    evaluation_mask,
                )
                clearwater_is_optical_endpoint = bool(
                    hierarchical_phase and kind == "true_clearwater"
                )
                aqueous_failures = (
                    []
                    if clearwater_is_optical_endpoint
                    else [
                        f"{metric}={float(aqueous_metrics[metric]):.8g}>{float(limit):.8g}"
                        for metric, limit in AQUEOUS_NEGATIVE_CONTROL_LIMITS.items()
                        if float(aqueous_metrics[metric]) > float(limit)
                    ]
                )
                if (
                    not clearwater_is_optical_endpoint
                    and not bool(aqueous_calibration["passed"])
                ):
                    if "negative_floor_probability" in aqueous_calibration:
                        aqueous_failures.append(
                            "co2_negative_floor_probability="
                            f"{float(aqueous_calibration['negative_floor_probability']):.8g}>"
                            f"{HIERARCHICAL_CO2_NEGATIVE_FLOOR_MAX:.8g}"
                        )
                    else:
                        aqueous_failures.append(
                            "aqueous_logit_shift="
                            f"{float(aqueous_calibration['logit_shift']):.8g}>"
                            f"{MAX_AQUEOUS_LOGIT_SHIFT:.8g}"
                        )
                aqueous_digest = _model_digest(resolved_aqueous_model)
                report_key = f"{kind}:{aqueous_digest}"
                if report_key not in reported_aqueous_models:
                    reported_aqueous_models.add(report_key)
                    rows.append(
                        {
                            "control": kind,
                            "model": f"co2_presence:{name}",
                            "model_sha256": aqueous_digest,
                            "passed": not aqueous_failures,
                            "failures": aqueous_failures,
                            "metrics": aqueous_metrics,
                            "control_role": (
                                "optical_clearwater_endpoint_diagnostic_only"
                                if clearwater_is_optical_endpoint
                                else "known_no_visible_co2"
                            ),
                        }
                    )
            raw_probability = raw_gas_controls[name][kind]
            conditional_probability = (
                _apply_negative_calibration(
                    raw_probability,
                    gas_calibrations[name],
                )
                if hierarchical_phase
                else raw_probability
            )
            if aqueous_probability is None:
                probability = conditional_probability
                support = np.ones_like(conditional_probability, dtype=np.float32)
            elif hierarchical_phase:
                support = np.asarray(aqueous_probability, dtype=np.float32)
                probability = (
                    np.asarray(conditional_probability, dtype=np.float32) * support
                )
            else:
                probability, support = _apply_aqueous_envelope(
                    conditional_probability,
                    aqueous_probability,
                )
            probability = np.asarray(probability, dtype=np.float32)
            probability[~evaluation_mask] = 0.0
            conditional_probability = np.asarray(
                conditional_probability, dtype=np.float32
            )
            conditional_probability[~evaluation_mask] = 0.0
            metrics = _probability_record(0.0, probability, evaluation_mask)
            raw_metrics = _probability_record(
                0.0,
                raw_probability,
                evaluation_mask,
            )
            clearwater_is_optical_endpoint = bool(
                hierarchical_phase and kind == "true_clearwater"
            )
            failures = (
                []
                if clearwater_is_optical_endpoint
                else [
                    f"{metric}={float(metrics[metric]):.8g}>{float(limit):.8g}"
                    for metric, limit in NEGATIVE_CONTROL_LIMITS.items()
                    if float(metrics[metric]) > float(limit)
                ]
            )
            if (
                hierarchical_phase
                and not clearwater_is_optical_endpoint
                and not bool(gas_calibrations[name]["passed"])
            ):
                failures.append(
                    "conditional_gas_negative_floor_probability="
                    f"{float(gas_calibrations[name]['negative_floor_probability']):.8g}>"
                    f"{HIERARCHICAL_GAS_NEGATIVE_FLOOR_MAX:.8g}"
                )
            rows.append(
                {
                    "control": kind,
                    "model": name,
                    "model_sha256": _model_digest(models[name]),
                    "passed": not failures,
                    "failures": failures,
                    "metrics": metrics,
                    "raw_metrics_before_aqueous_envelope": raw_metrics,
                    "conditional_metrics_after_known_negative_calibration": (
                        _probability_record(
                            0.0,
                            conditional_probability,
                            evaluation_mask,
                        )
                    ),
                    "aqueous_support_fraction_p050": float(
                        np.mean(support[evaluation_mask] >= 0.50)
                    ),
                    "control_role": (
                        "optical_clearwater_endpoint_diagnostic_only"
                        if clearwater_is_optical_endpoint
                        else "known_no_visible_co2"
                    ),
                }
            )
    return {
        "schema": 1,
        "kind": "known-no-free-gas-preflight",
        "dataset": source.dataset,
        "run": source.run,
        "input_contract": _negative_control_input_contract(
            models,
            source,
            aqueous_model,
            hierarchical_phase=hierarchical_phase,
        ),
        "passed": bool(rows) and all(bool(row["passed"]) for row in rows),
        "interpretation": (
            "Pre-injection baselines are hard no-CO2 controls. In the hierarchical "
            "model, true clear water is an optical endpoint for displaced dyed "
            "water and is diagnostic rather than a negative gas label."
        ),
        "limits": dict(NEGATIVE_CONTROL_LIMITS),
        "aqueous_limits": dict(AQUEOUS_NEGATIVE_CONTROL_LIMITS),
        "aqueous_envelope": {
            "active": any(model is not None for model in aqueous_models.values()),
            "mode": (
                "pointwise-co2-presence-times-conditional-gas"
                if hierarchical_phase
                else "soft-nearby-aqueous-envelope"
            ),
            "radius_fraction_of_short_image_axis": AQUEOUS_ENVELOPE_RADIUS_FRACTION,
            "gate_floor": 0.02,
            "rule": (
                "Free-gas evidence requires nearby aqueous-CO2 evidence. The "
                "gate is soft to preserve uncertainty from a visible 2D slice."
            ),
        },
        "aqueous_negative_calibration": (
            next(iter(aqueous_calibrations.values()))
            if len(
                {
                    json.dumps(value, sort_keys=True)
                    for value in aqueous_calibrations.values()
                }
            )
            == 1
            else {"mode": "per-candidate", "by_candidate": aqueous_calibrations}
        ),
        "aqueous_negative_calibration_by_candidate": aqueous_calibrations,
        "conditional_gas_negative_calibration_by_candidate": gas_calibrations,
        "hierarchical_co2_calibration_contract": (
            {
                "intercept_controls": [
                    "pre_injection_baseline_initial",
                    "pre_injection_baseline_static",
                ],
                "true_clearwater_role": (
                    "diagnostic optical endpoint for displaced dyed water; excluded "
                    "from both CO2-presence and conditional-gas intercept calibration"
                ),
                "co2_negative_floor_contract": (
                    "Subtract the pre-injection 99.9th percentile score and "
                    "rescale remaining headroom to [0, 1] inside the porous-medium "
                    "evaluation domain."
                ),
                "conditional_gas_intercept_controls": [
                    "pre_injection_baseline_initial",
                    "pre_injection_baseline_static",
                ],
                "conditional_gas_negative_floor_contract": (
                    "Subtract the maximum pre-injection baseline 99.9th percentile "
                    "score and rescale remaining headroom to [0, 1] inside the "
                    "porous-medium evaluation domain. Clear water remains an "
                    "unshifted optical endpoint diagnostic."
                ),
                "output_semantics": "normalized optical evidence, not probability",
            }
            if hierarchical_phase
            else None
        ),
        "controls": rows,
        "created_utc": _utc_now(),
    }


def _write_probability_plot(
    path: Path,
    run: str,
    records: Sequence[Mapping[str, Any]],
    *,
    normalized_evidence: bool = False,
) -> None:
    elapsed = np.asarray([float(row["elapsed_h"]) for row in records])
    fig = plt.figure(figsize=(16, 9), dpi=100)
    axis = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("#f6f7f8")
    axis.set_facecolor("#ffffff")
    axis.plot(
        elapsed,
        [row["gas_probability_mean"] for row in records],
        color="#202124",
        linewidth=2.3,
        marker="o",
        label=("Mean gas evidence" if normalized_evidence else "Mean gas probability"),
    )
    axis.plot(
        elapsed,
        [row["gas_probability_p95"] for row in records],
        color="#9c3d95",
        linewidth=1.8,
        marker="o",
        label=("P95 gas evidence" if normalized_evidence else "P95 gas probability"),
    )
    for key, color, label in (
        ("support_fraction_p025", "#e0a321", "Area score(gas) >= 0.25"),
        ("support_fraction_p050", "#dd643f", "Area score(gas) >= 0.50"),
        ("support_fraction_p075", "#c7296f", "Area score(gas) >= 0.75"),
    ):
        axis.plot(
            elapsed,
            [row[key] for row in records],
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=3,
            label=label,
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Elapsed time (h)")
    axis.set_ylabel(
        "Normalized evidence / area fraction"
        if normalized_evidence
        else "Probability / area fraction"
    )
    axis.set_title(
        (
            f"{run.upper()} normalized optical free-gas evidence diagnostics"
            if normalized_evidence
            else f"{run.upper()} optical free-gas probability diagnostics"
        ),
        loc="left",
        fontsize=17,
        pad=18,
    )
    axis.text(
        0.995,
        1.025,
        "Target run was not used for model fitting; candidate identity is blinded",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#4d5358",
    )
    axis.grid(color="#d9dde1", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.84, bottom=0.18)
    fig.canvas.draw()
    image = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    _write_png(path, image)


def _candidate_asset_paths(
    review_root: Path,
    run: str,
    label: str,
    asset_fingerprint: str,
    index: int,
) -> tuple[Path, Path]:
    fingerprint = str(asset_fingerprint)[:16]
    overlay = (
        review_root
        / "native_overlays"
        / run
        / f"{label}_{fingerprint}"
        / f"frame_{index:03d}.png"
    )
    quiz = (
        review_root
        / "phase_quiz"
        / run
        / f"{label}_{fingerprint}_q{review.PHASE_QUIZ_PROPOSAL_SCHEMA}"
        / "proposals.json"
    )
    return overlay, quiz


def _candidate_phase_map_path(
    review_root: Path,
    run: str,
    label: str,
    asset_fingerprint: str,
    index: int,
) -> Path:
    fingerprint = str(asset_fingerprint)[:16]
    return (
        review_root
        / "native_phase_maps"
        / run
        / f"{label}_{fingerprint}"
        / f"frame_{index:03d}.png"
    )


def _sampling_facies_map(
    rig: Any,
    shape: Sequence[int],
    analysis_domain: np.ndarray,
) -> np.ndarray:
    """Return facies labels aligned with the phase-quiz analysis image."""

    expected = tuple(int(value) for value in shape[:2])
    facies = getattr(rig, "facies", None)
    values = None if facies is None else np.asarray(getattr(facies, "img", facies))
    if values is None or values.shape[:2] != expected:
        return np.where(np.asarray(analysis_domain, dtype=bool), 1, 0).astype(
            np.int16
        )
    result = np.asarray(values, dtype=np.int16).copy()
    result[~np.asarray(analysis_domain, dtype=bool)] = 0
    return result


def _normalized_optical_evidence(
    change: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Map corrected optical change to a robust zero-to-one evidence scale."""

    values = np.asarray(change, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.float32)
    if not np.any(mask):
        return output
    low, high = np.quantile(values[mask], (0.20, 0.95))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-8:
        output[mask] = np.float32(0.5)
        return output
    output = np.clip((values - np.float32(low)) / np.float32(high - low), 0.0, 1.0)
    output[~mask] = 0.0
    return np.asarray(output, dtype=np.float32)


def _phase_hypothesis_scores(
    gas_probability: np.ndarray,
    aqueous_probability: np.ndarray | None,
    optical_change: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    """Derive four blind sampling hypotheses from the available optical signals.

    These fields only stratify manual validation. They are not a five-class
    phase model and are never used as human labels.
    """

    gas = np.clip(np.asarray(gas_probability, dtype=np.float32), 0.0, 1.0)
    optical = _normalized_optical_evidence(optical_change, valid)
    if aqueous_probability is None:
        aqueous = optical * (np.float32(1.0) - gas)
        co2_presence = np.maximum(gas, optical)
        source = "gas_probability_plus_corrected_optical_change"
    else:
        aqueous = np.clip(
            np.asarray(aqueous_probability, dtype=np.float32), 0.0, 1.0
        )
        co2_presence = np.clip(gas + aqueous, 0.0, 1.0)
        source = "hierarchical_gas_and_aqueous_probabilities"

    scores = {
        "gas": gas * (np.float32(1.0) - np.float32(0.5) * aqueous),
        "aqueous": aqueous * (np.float32(1.0) - np.float32(0.5) * gas),
        "mixed": np.clip(
            np.float32(2.0) * np.minimum(gas, aqueous), 0.0, 1.0
        ),
        "water": np.clip(np.float32(1.0) - co2_presence, 0.0, 1.0),
    }
    mask = np.asarray(valid, dtype=bool)
    for field in scores.values():
        field[~mask] = 0.0
    scores["_source"] = source
    return scores


def _strategic_phase_centers(
    scores: Mapping[str, Any],
    facies: np.ndarray,
    interior_domain: np.ndarray,
    *,
    local_radius: int,
    count_per_combination: int = PHASE_QUIZ_CENTERS_PER_FACIES_FRAME,
) -> list[dict[str, Any]]:
    """Sample each predicted phase in every facies with spatial separation."""

    labels = np.asarray(facies, dtype=np.int16)
    interior = np.asarray(interior_domain, dtype=bool)
    phase_fields = [np.asarray(scores[name], dtype=np.float32) for name in PHASE_QUIZ_HYPOTHESES]
    stack = np.stack(phase_fields, axis=0)
    assignment = np.argmax(stack, axis=0)
    local_size = max(3, 2 * int(local_radius) + 1)
    suppression_radius = max(local_size * 2, int(round(0.035 * min(labels.shape))))
    rows: list[dict[str, Any]] = []
    for facies_id in sorted(int(value) for value in np.unique(labels[interior]) if int(value) > 0):
        facies_mask = labels == facies_id
        facies_fraction = cv2.boxFilter(
            facies_mask.astype(np.float32),
            ddepth=-1,
            ksize=(local_size, local_size),
            normalize=True,
        )
        facies_interior = interior & (facies_fraction >= np.float32(0.92))
        if not np.any(facies_interior):
            continue
        for phase_index, phase in enumerate(PHASE_QUIZ_HYPOTHESES):
            field = cv2.boxFilter(
                phase_fields[phase_index],
                ddepth=-1,
                ksize=(local_size, local_size),
                normalize=True,
            )
            # Sample the strongest evidence for every phase/facies combination.
            # Requiring the phase to be the argmax would suppress precisely the
            # ambiguous aqueous/mixed regions that are most useful to label.
            valid = facies_interior & np.isfinite(field) & (field > np.float32(1e-6))
            if not np.any(valid):
                continue
            work = np.where(valid, field, -np.inf).astype(np.float32, copy=True)
            for spatial_rank in range(max(0, int(count_per_combination))):
                flat = int(np.argmax(work))
                confidence = float(work.flat[flat])
                if not np.isfinite(confidence):
                    break
                row, column = np.unravel_index(flat, work.shape)
                rows.append(
                    {
                        "phase": phase,
                        "facies_id": facies_id,
                        "row": int(row),
                        "column": int(column),
                        "confidence": confidence,
                        "phase_is_dominant": bool(
                            assignment[int(row), int(column)] == phase_index
                        ),
                        "spatial_rank": int(spatial_rank),
                        "facies_fraction": float(facies_fraction[row, column]),
                    }
                )
                cv2.circle(
                    work,
                    (int(column), int(row)),
                    suppression_radius,
                    -np.inf,
                    -1,
                )
    return rows


def _proposal(
    *,
    run: str,
    label: str,
    candidate_id: str,
    digest: str,
    frame_index: int,
    elapsed_h: float,
    stratum: str,
    row: int,
    column: int,
    probability: np.ndarray,
    aqueous_probability: np.ndarray | None,
    change: np.ndarray,
    suggested_phase: str,
    facies_id: int,
    phase_confidence: float,
    phase_is_dominant: bool,
    phase_hypothesis_source: str,
    spatial_rank: int,
    facies_fraction: float,
    canvas_wh: Sequence[int],
    content_box: Sequence[int],
    analysis_domain: str,
    analysis_domain_margin_px: int,
) -> dict[str, Any]:
    image_height, image_width = probability.shape
    canvas_width, canvas_height = map(int, canvas_wh)
    content_x, content_y, _content_width, _content_height = map(int, content_box)
    half = max(20, int(round(0.012 * min(image_height, image_width))))
    left = max(0, min(image_width - 2 * half, int(column) - half))
    top = max(0, min(image_height - 2 * half, int(row) - half))
    right = min(image_width - 1, left + 2 * half)
    bottom = min(image_height - 1, top + 2 * half)
    points_px = [[left, top], [right, top], [right, bottom], [left, bottom]]
    points = [
        [(content_x + x) / canvas_width, (content_y + y) / canvas_height]
        for x, y in points_px
    ]
    region_probability = probability[top : bottom + 1, left : right + 1]
    region_aqueous = (
        None
        if aqueous_probability is None
        else aqueous_probability[top : bottom + 1, left : right + 1]
    )
    region_change = change[top : bottom + 1, left : right + 1]
    seed = (
        f"{run}|{candidate_id}|strategic-phase-v1|{frame_index}|{stratum}|"
        f"{facies_id}|{row}|{column}"
    )
    focus_left = max(0, left - 2 * half)
    focus_top = max(0, top - 2 * half)
    focus_right = min(image_width, right + 2 * half)
    focus_bottom = min(image_height, bottom + 2 * half)
    return {
        "proposal_id": "blind-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
        "run": run,
        "candidate": label,
        "candidate_id": candidate_id,
        "params_sha256": digest,
        "frame_index": int(frame_index),
        "elapsed_h": float(elapsed_h),
        "suggested_phase": suggested_phase,
        "suggestion_basis": (
            "blind phase-hypothesis sampling from gas, aqueous/optical-change, "
            "facies, time, and spatial position; the hypothesis is hidden until "
            "the human answer is recorded"
        ),
        "interpretation_scope": "visible_2d_slice_only",
        "analysis_domain": str(analysis_domain),
        "analysis_domain_margin_px": int(analysis_domain_margin_px),
        "coordinate_space": "normalized_native_review_canvas_16x9",
        "shape": "freehand",
        "points": points,
        "x": min(point[0] for point in points),
        "y": min(point[1] for point in points),
        "width": max(point[0] for point in points) - min(point[0] for point in points),
        "height": max(point[1] for point in points) - min(point[1] for point in points),
        "analysis_polygon_px": points_px,
        "analysis_bbox_xywh": [left, top, right - left, bottom - top],
        "focus_box": {
            "x": (content_x + focus_left) / canvas_width,
            "y": (content_y + focus_top) / canvas_height,
            "width": (focus_right - focus_left) / canvas_width,
            "height": (focus_bottom - focus_top) / canvas_height,
        },
        "model_stats": {
            "sampling_stratum": stratum,
            "phase_hypothesis": suggested_phase,
            "phase_hypothesis_source": phase_hypothesis_source,
            "phase_hypothesis_confidence": float(phase_confidence),
            "phase_hypothesis_is_dominant": bool(phase_is_dominant),
            "facies_id": int(facies_id),
            "facies_fraction": float(facies_fraction),
            "spatial_rank_within_phase_facies_frame": int(spatial_rank),
            "mean_gas_probability": float(np.mean(region_probability)),
            "median_gas_probability": float(np.median(region_probability)),
            "mean_aqueous_probability": (
                None
                if region_aqueous is None
                else float(np.mean(region_aqueous))
            ),
            "mean_rgb_change_norm": float(np.mean(region_change)),
            "normalized_center_xy": [
                float(column) / max(1, image_width - 1),
                float(row) / max(1, image_height - 1),
            ],
        },
        "priority": float(4.0 * phase_confidence + np.mean(region_change)),
    }


def _blind_proposals(
    *,
    run: str,
    label: str,
    candidate_id: str,
    digest: str,
    frame_index: int,
    frame_count: int,
    elapsed_h: float,
    image: np.ndarray,
    baseline: np.ndarray,
    probability: np.ndarray,
    aqueous_probability: np.ndarray | None = None,
    facies_map: np.ndarray | None = None,
    analysis_domain_mask: np.ndarray,
    analysis_domain_name: str = "physically_valid_analysis_domain",
    canvas_wh: Sequence[int],
    content_box: Sequence[int],
) -> list[dict[str, Any]]:
    del frame_count
    change = np.linalg.norm(
        np.asarray(image, dtype=np.float32) - np.asarray(baseline, dtype=np.float32),
        axis=2,
    )
    domain = np.asarray(analysis_domain_mask, dtype=bool)
    if domain.shape != probability.shape:
        raise ValueError(
            f"Quiz-domain shape {domain.shape} differs from probability shape "
            f"{probability.shape}"
        )
    if aqueous_probability is not None and aqueous_probability.shape != probability.shape:
        raise ValueError(
            "Aqueous-probability shape differs from gas-probability shape"
        )
    labels = (
        np.where(domain, 1, 0).astype(np.int16)
        if facies_map is None
        else np.asarray(facies_map, dtype=np.int16)
    )
    if labels.shape != probability.shape:
        raise ValueError(
            f"Facies-map shape {labels.shape} differs from probability shape "
            f"{probability.shape}"
        )

    scale = min(1.0, 1400.0 / max(probability.shape))
    small_width = max(1, int(round(probability.shape[1] * scale)))
    small_height = max(1, int(round(probability.shape[0] * scale)))
    size = (small_width, small_height)
    gas_small = cv2.resize(
        np.asarray(probability, dtype=np.float32), size, interpolation=cv2.INTER_AREA
    )
    aqueous_small = (
        None
        if aqueous_probability is None
        else cv2.resize(
            np.asarray(aqueous_probability, dtype=np.float32),
            size,
            interpolation=cv2.INTER_AREA,
        )
    )
    change_small = cv2.resize(
        np.asarray(change, dtype=np.float32), size, interpolation=cv2.INTER_AREA
    )
    domain_small = cv2.resize(
        domain.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ) > 0
    facies_small = cv2.resize(
        labels, size, interpolation=cv2.INTER_NEAREST
    ).astype(np.int16)

    half = max(20, int(round(0.012 * min(probability.shape))))
    local_radius = max(2, int(round(half * scale)))
    margin = max(
        review.PHASE_QUIZ_DOMAIN_MARGIN_MIN_PX,
        min(
            review.PHASE_QUIZ_DOMAIN_MARGIN_MAX_PX,
            int(
                round(
                    review.PHASE_QUIZ_DOMAIN_MARGIN_FRACTION
                    * min(probability.shape)
                )
            ),
        ),
    )
    small_domain_radius = max(2, int(round((half + margin) * scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * small_domain_radius + 1, 2 * small_domain_radius + 1),
    )
    interior_domain = cv2.erode(domain_small.astype(np.uint8), kernel) > 0
    valid = interior_domain & np.isfinite(gas_small) & np.isfinite(change_small)
    if aqueous_small is not None:
        valid &= np.isfinite(aqueous_small)
    if not np.any(valid):
        return []
    scores = _phase_hypothesis_scores(
        gas_small,
        aqueous_small,
        change_small,
        valid,
    )
    source_name = str(scores.pop("_source"))
    centers = _strategic_phase_centers(
        scores,
        facies_small,
        valid,
        local_radius=local_radius,
    )
    proposals: list[dict[str, Any]] = []
    for center in centers:
        row = int(round(int(center["row"]) / scale))
        column = int(round(int(center["column"]) / scale))
        phase = str(center["phase"])
        facies_id = int(center["facies_id"])
        stratum = f"predicted_{phase}_facies_{facies_id}"
        proposals.append(
            _proposal(
                run=run,
                label=label,
                candidate_id=candidate_id,
                digest=digest,
                frame_index=frame_index,
                elapsed_h=elapsed_h,
                stratum=stratum,
                row=int(row),
                column=int(column),
                probability=probability,
                aqueous_probability=aqueous_probability,
                change=change,
                suggested_phase=phase,
                facies_id=facies_id,
                phase_confidence=float(center["confidence"]),
                phase_is_dominant=bool(center["phase_is_dominant"]),
                phase_hypothesis_source=source_name,
                spatial_rank=int(center["spatial_rank"]),
                facies_fraction=float(center["facies_fraction"]),
                canvas_wh=canvas_wh,
                content_box=content_box,
                analysis_domain=analysis_domain_name,
                analysis_domain_margin_px=margin,
            )
        )
    return proposals


def _select_blind_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    max_total: int | None = 8,
) -> list[dict[str, Any]]:
    """Balance blind regions over phase, facies, space, and time."""

    strategic = [
        dict(item)
        for item in proposals
        if str(item.get("suggested_phase") or "") in PHASE_QUIZ_HYPOTHESES
        and (item.get("model_stats") or {}).get("facies_id") is not None
    ]
    if strategic:
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for proposal in strategic:
            stats = proposal.get("model_stats") or {}
            key = (
                str(proposal.get("suggested_phase") or ""),
                int(stats.get("facies_id")),
            )
            grouped.setdefault(key, []).append(proposal)

        frame_values = [int(item.get("frame_index") or 0) for item in strategic]
        frame_min = min(frame_values, default=0)
        frame_span = max(1, max(frame_values, default=frame_min) - frame_min)

        def center(item: Mapping[str, Any]) -> tuple[float, float, float]:
            stats = item.get("model_stats") or {}
            xy = stats.get("normalized_center_xy") or (0.5, 0.5)
            return (
                float(xy[0]),
                float(xy[1]),
                (int(item.get("frame_index") or 0) - frame_min) / frame_span,
            )

        def diverse_group_order(
            items: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            remaining = [dict(item) for item in items]
            if not remaining:
                return []
            priorities = [float(item.get("priority") or 0.0) for item in remaining]
            low = min(priorities)
            span = max(1e-9, max(priorities) - low)
            ordered = [
                remaining.pop(
                    max(
                        range(len(remaining)),
                        key=lambda index: float(
                            remaining[index].get("priority") or 0.0
                        ),
                    )
                )
            ]
            while remaining:
                chosen_centers = [center(item) for item in ordered]

                def score(item: Mapping[str, Any]) -> tuple[float, float, str]:
                    x, y, t = center(item)
                    distance = min(
                        math.sqrt(
                            (x - other_x) ** 2
                            + (y - other_y) ** 2
                            + 0.35 * (t - other_t) ** 2
                        )
                        for other_x, other_y, other_t in chosen_centers
                    )
                    priority = (float(item.get("priority") or 0.0) - low) / span
                    return (
                        distance + 0.10 * priority,
                        priority,
                        str(item.get("proposal_id") or ""),
                    )

                next_index = max(range(len(remaining)), key=lambda index: score(remaining[index]))
                ordered.append(remaining.pop(next_index))
            return ordered

        queues = {key: diverse_group_order(items) for key, items in grouped.items()}
        phase_counts = {phase: 0 for phase in PHASE_QUIZ_HYPOTHESES}
        facies_counts: dict[int, int] = {}
        combination_counts = {key: 0 for key in queues}
        frame_counts: dict[int, int] = {}
        selected: list[dict[str, Any]] = []
        limit = len(strategic) if max_total is None else max(0, int(max_total))
        phase_rank = {phase: index for index, phase in enumerate(PHASE_QUIZ_HYPOTHESES)}
        while len(selected) < limit:
            available = [key for key, queue in queues.items() if queue]
            if not available:
                break

            def group_score(key: tuple[str, int]) -> tuple[Any, ...]:
                phase, facies_id = key
                item = queues[key][0]
                frame = int(item.get("frame_index") or 0)
                return (
                    phase_counts.get(phase, 0),
                    facies_counts.get(facies_id, 0),
                    combination_counts.get(key, 0),
                    frame_counts.get(frame, 0),
                    -float(item.get("priority") or 0.0),
                    phase_rank.get(phase, 99),
                    facies_id,
                )

            key = min(available, key=group_score)
            item = queues[key].pop(0)
            selected.append(item)
            phase, facies_id = key
            frame = int(item.get("frame_index") or 0)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            facies_counts[facies_id] = facies_counts.get(facies_id, 0) + 1
            combination_counts[key] = combination_counts.get(key, 0) + 1
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
        return selected

    grouped: dict[int, list[dict[str, Any]]] = {}
    for proposal in proposals:
        grouped.setdefault(int(proposal["frame_index"]), []).append(dict(proposal))
    for group in grouped.values():
        group.sort(
            key=lambda item: (
                -float(item.get("priority") or 0.0),
                str((item.get("model_stats") or {}).get("sampling_stratum") or ""),
            )
        )

    selected: list[dict[str, Any]] = []
    ordered_frames = sorted(grouped)
    limit = len(proposals) if max_total is None else max(0, int(max_total))
    while len(selected) < limit:
        added = False
        for frame_index in ordered_frames:
            group = grouped[frame_index]
            if group and len(selected) < limit:
                selected.append(group.pop(0))
                added = True
        if not added:
            break
    return sorted(
        selected,
        key=lambda item: (
            int(item["frame_index"]),
            str((item.get("model_stats") or {}).get("sampling_stratum") or ""),
        ),
    )


def _proposal_inside_domain(
    proposal: Mapping[str, Any],
    domain: np.ndarray,
    *,
    minimum_fraction: float = 0.995,
) -> bool:
    bbox = proposal.get("analysis_bbox_xywh")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return False
    left, top, width, height = (int(round(float(value))) for value in bbox)
    right = min(domain.shape[1], left + max(1, width) + 1)
    bottom = min(domain.shape[0], top + max(1, height) + 1)
    left = max(0, left)
    top = max(0, top)
    if left >= right or top >= bottom:
        return False
    region = np.asarray(domain[top:bottom, left:right], dtype=bool)
    return bool(region.size and float(np.mean(region)) >= minimum_fraction)


def _domain_only_repair_proposals(
    *,
    run: str,
    label: str,
    candidate_id: str,
    digest: str,
    domain: np.ndarray,
    facies_map: np.ndarray | None = None,
    domain_name: str,
    native_manifest: Mapping[str, Any],
    max_total: int | None = review.PHASE_QUIZ_MAX_PROPOSALS_PER_CANDIDATE,
) -> list[dict[str, Any]]:
    """Create blind spatial controls when an old proposal pool is unsafe."""

    mask = np.asarray(domain, dtype=bool)
    height, width = mask.shape
    half = max(20, int(round(0.012 * min(height, width))))
    margin = max(
        review.PHASE_QUIZ_DOMAIN_MARGIN_MIN_PX,
        min(
            review.PHASE_QUIZ_DOMAIN_MARGIN_MAX_PX,
            int(
                round(
                    review.PHASE_QUIZ_DOMAIN_MARGIN_FRACTION
                    * min(height, width)
                )
            ),
        ),
    )
    radius = half + margin
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * radius + 1, 2 * radius + 1),
    )
    interior = cv2.erode(mask.astype(np.uint8), kernel) > 0
    interior[:radius, :] = False
    interior[-radius:, :] = False
    interior[:, :radius] = False
    interior[:, -radius:] = False
    if not np.any(interior):
        return []

    stride = max(1, int(round(min(height, width) / 420.0)))
    sampled = np.argwhere(interior[::stride, ::stride])
    if not sampled.size:
        return []
    sampled = sampled.astype(np.int32) * stride
    sampled[:, 0] = np.minimum(sampled[:, 0], height - 1)
    sampled[:, 1] = np.minimum(sampled[:, 1], width - 1)

    initial_targets = [
        (0.16, 0.24),
        (0.38, 0.24),
        (0.62, 0.24),
        (0.84, 0.24),
        (0.16, 0.72),
        (0.38, 0.72),
        (0.62, 0.72),
        (0.84, 0.72),
    ]
    reserve_targets = [
        (x_fraction, y_fraction)
        for y_fraction in (0.14, 0.39, 0.61, 0.86)
        for x_fraction in (0.08, 0.25, 0.42, 0.58, 0.75, 0.92)
    ]
    targets = initial_targets + reserve_targets
    limit = len(targets) if max_total is None else max(0, int(max_total))
    selected_centers: list[tuple[int, int]] = []
    used_indices: set[int] = set()
    for x_fraction, y_fraction in targets:
        if len(selected_centers) >= limit:
            break
        target_row = y_fraction * (height - 1)
        target_column = x_fraction * (width - 1)
        distances = (
            (sampled[:, 0] - target_row) ** 2
            + (sampled[:, 1] - target_column) ** 2
        ).astype(np.float64)
        if used_indices:
            distances[np.fromiter(used_indices, dtype=np.int64)] = np.inf
        index = int(np.argmin(distances))
        if not np.isfinite(distances[index]):
            break
        used_indices.add(index)
        selected_centers.append((int(sampled[index, 0]), int(sampled[index, 1])))

    canvas_wh = native_manifest.get("canvas_wh") or [width, height]
    content_box = native_manifest.get("content_box_xywh") or [0, 0, width, height]
    canvas_width, canvas_height = map(int, canvas_wh)
    content_x, content_y, content_width, content_height = map(int, content_box)
    if (content_width, content_height) != (width, height):
        raise ValueError("Native review manifest does not match phase-quiz domain")
    frame_sources = [
        dict(row)
        for row in (native_manifest.get("frame_sources") or [])
        if isinstance(row, Mapping)
    ]
    if not frame_sources:
        return []
    frame_slots = np.rint(
        np.linspace(0, len(frame_sources) - 1, len(selected_centers))
    ).astype(int)

    proposals: list[dict[str, Any]] = []
    for index, ((row, column), frame_slot) in enumerate(
        zip(selected_centers, frame_slots, strict=True)
    ):
        frame = frame_sources[int(frame_slot)]
        frame_index = int(frame.get("frame_index", frame_slot))
        elapsed_h = float(frame.get("elapsed_h") or 0.0)
        left, top = column - half, row - half
        right, bottom = column + half, row + half
        points_px = [[left, top], [right, top], [right, bottom], [left, bottom]]
        points = [
            [(content_x + x) / canvas_width, (content_y + y) / canvas_height]
            for x, y in points_px
        ]
        focus_left = max(0, left - 2 * half)
        focus_top = max(0, top - 2 * half)
        focus_right = min(width, right + 2 * half)
        focus_bottom = min(height, bottom + 2 * half)
        seed = f"{run}|{candidate_id}|domain-only|{frame_index}|{row}|{column}"
        proposals.append(
            {
                "proposal_id": "blind-"
                + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
                "run": run,
                "candidate": label,
                "candidate_id": candidate_id,
                "params_sha256": digest,
                "frame_index": frame_index,
                "elapsed_h": elapsed_h,
                "suggested_phase": "unscored",
                "suggestion_basis": (
                    "blind spatial-temporal sampling inside an eroded "
                    "porous-medium facies mask"
                ),
                "interpretation_scope": "visible_2d_slice_only",
                "analysis_domain": domain_name,
                "analysis_domain_margin_px": int(margin),
                "coordinate_space": "normalized_native_review_canvas_16x9",
                "shape": "freehand",
                "points": points,
                "x": min(point[0] for point in points),
                "y": min(point[1] for point in points),
                "width": max(point[0] for point in points)
                - min(point[0] for point in points),
                "height": max(point[1] for point in points)
                - min(point[1] for point in points),
                "analysis_polygon_px": points_px,
                "analysis_bbox_xywh": [left, top, 2 * half, 2 * half],
                "focus_box": {
                    "x": (content_x + focus_left) / canvas_width,
                    "y": (content_y + focus_top) / canvas_height,
                    "width": (focus_right - focus_left) / canvas_width,
                    "height": (focus_bottom - focus_top) / canvas_height,
                },
                "model_stats": {
                    "sampling_stratum": "domain_only_spatial_repair",
                    "spatial_rank": int(index),
                },
                "priority": float(limit - index),
            }
        )
    return proposals


def _optical_active_learning_proposals(
    *,
    run: str,
    label: str,
    candidate_id: str,
    digest: str,
    domain: np.ndarray,
    facies_map: np.ndarray | None = None,
    domain_name: str,
    native_manifest: Mapping[str, Any],
    review_root: Path,
    max_total: int | None = review.PHASE_QUIZ_MAX_PROPOSALS_PER_CANDIDATE,
) -> list[dict[str, Any]]:
    """Rebuild a blind phase/facies pool from corrected images and contours."""

    frame_sources = [
        dict(row)
        for row in (native_manifest.get("frame_sources") or [])
        if isinstance(row, Mapping)
    ]
    canvas_wh = native_manifest.get("canvas_wh") or []
    content_box = native_manifest.get("content_box_xywh") or []
    if not frame_sources or len(canvas_wh) != 2 or len(content_box) != 4:
        return []
    canvas_width, canvas_height = map(int, canvas_wh)
    content_x, content_y, content_width, content_height = map(int, content_box)
    mask = np.asarray(domain, dtype=bool)
    if mask.shape != (content_height, content_width):
        return []
    labels = (
        np.where(mask, 1, 0).astype(np.int16)
        if facies_map is None
        else np.asarray(facies_map, dtype=np.int16)
    )
    if labels.shape != mask.shape:
        return []

    reference_path = (
        review_root / "native_references" / run / "pre_injection.jpg"
    )
    reference_canvas = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference_canvas is None:
        return []
    reference = reference_canvas[
        content_y : content_y + content_height,
        content_x : content_x + content_width,
    ]
    if reference.shape[:2] != mask.shape:
        return []

    scale = min(1.0, 1400.0 / max(mask.shape))
    small_width = max(1, int(round(content_width * scale)))
    small_height = max(1, int(round(content_height * scale)))
    size = (small_width, small_height)
    reference_small = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
    domain_small = cv2.resize(
        mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
    ) > 0
    facies_small = cv2.resize(
        labels, size, interpolation=cv2.INTER_NEAREST
    ).astype(np.int16)
    half = max(20, int(round(0.012 * min(mask.shape))))
    margin = max(
        review.PHASE_QUIZ_DOMAIN_MARGIN_MIN_PX,
        min(
            review.PHASE_QUIZ_DOMAIN_MARGIN_MAX_PX,
            int(round(review.PHASE_QUIZ_DOMAIN_MARGIN_FRACTION * min(mask.shape))),
        ),
    )
    small_radius = max(2, int(round((half + margin) * scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * small_radius + 1, 2 * small_radius + 1),
    )
    interior = cv2.erode(domain_small.astype(np.uint8), kernel) > 0
    if not np.any(interior):
        return []

    def optical_arrays(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        total = np.sum(rgb, axis=2, keepdims=True)
        chroma = rgb / np.maximum(total, 1e-4)
        luminance = (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )
        return rgb, chroma, luminance

    reference_rgb, reference_chroma, reference_luminance = optical_arrays(
        reference_small
    )
    proposals: list[dict[str, Any]] = []
    local_radius = max(2, int(round(half * scale)))
    local_size = 2 * local_radius + 1

    for frame_slot, frame in enumerate(frame_sources):
        frame_index = int(frame.get("frame_index", frame_slot))
        frame_path = (
            review_root / "native_frames" / run / f"frame_{frame_index:03d}.jpg"
        )
        canvas = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if canvas is None:
            continue
        image = canvas[
            content_y : content_y + content_height,
            content_x : content_x + content_width,
        ]
        if image.shape[:2] != mask.shape:
            continue
        image_small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        rgb, chroma, luminance = optical_arrays(image_small)
        delta_rgb = rgb - reference_rgb
        delta_chroma = np.linalg.norm(chroma - reference_chroma, axis=2)
        delta_luminance = np.abs(luminance - reference_luminance)
        optical_change = delta_chroma + 0.20 * delta_luminance
        optical_change = cv2.GaussianBlur(optical_change, (0, 0), sigmaX=1.4)
        local_change = cv2.boxFilter(
            optical_change,
            ddepth=-1,
            ksize=(local_size, local_size),
            normalize=True,
        )
        valid_values = local_change[interior & np.isfinite(local_change)]
        if not valid_values.size:
            continue
        overlay_path = (
            review_root
            / "native_overlays"
            / run
            / f"{label}_{digest[:16]}"
            / f"frame_{frame_index:03d}.png"
        )
        gas_evidence, aqueous_evidence = _overlay_phase_evidence(
            overlay_path,
            content_box=content_box,
            size=size,
        )
        valid = interior & np.isfinite(local_change) & np.isfinite(gas_evidence)
        if aqueous_evidence is not None:
            valid &= np.isfinite(aqueous_evidence)
        scores = _phase_hypothesis_scores(
            gas_evidence,
            aqueous_evidence,
            local_change,
            valid,
        )
        source_name = "existing_contour_overlay_plus_" + str(scores.pop("_source"))
        centers = _strategic_phase_centers(
            scores,
            facies_small,
            valid,
            local_radius=local_radius,
        )

        sorted_values = np.sort(valid_values)
        elapsed_h = float(frame.get("elapsed_h") or 0.0)
        for center in centers:
                row_small = int(center["row"])
                column_small = int(center["column"])
                phase = str(center["phase"])
                facies_id = int(center["facies_id"])
                stratum = f"predicted_{phase}_facies_{facies_id}"
                row = int(round(row_small / scale))
                column = int(round(column_small / scale))
                left = max(0, min(content_width - 2 * half, column - half))
                top = max(0, min(content_height - 2 * half, row - half))
                right = min(content_width - 1, left + 2 * half)
                bottom = min(content_height - 1, top + 2 * half)
                points_px = [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ]
                points = [
                    [
                        (content_x + x) / canvas_width,
                        (content_y + y) / canvas_height,
                    ]
                    for x, y in points_px
                ]
                region_top = max(0, row_small - local_radius)
                region_bottom = min(small_height, row_small + local_radius + 1)
                region_left = max(0, column_small - local_radius)
                region_right = min(small_width, column_small + local_radius + 1)
                region = np.s_[region_top:region_bottom, region_left:region_right]
                mean_change = float(np.mean(local_change[region], dtype=np.float64))
                rank = float(
                    np.searchsorted(sorted_values, mean_change, side="right")
                    / max(1, sorted_values.size)
                )
                mean_delta_rgb = np.mean(
                    delta_rgb[region], axis=(0, 1), dtype=np.float64
                )
                mean_current_chroma = np.mean(
                    chroma[region], axis=(0, 1), dtype=np.float64
                )
                mean_reference_chroma = np.mean(
                    reference_chroma[region], axis=(0, 1), dtype=np.float64
                )
                seed = (
                    f"{run}|{candidate_id}|strategic-overlay-v1|{frame_index}|"
                    f"{phase}|{facies_id}|{row}|{column}"
                )
                focus = review._phase_quiz_focus_box(
                    left,
                    top,
                    right - left,
                    bottom - top,
                    image_width=content_width,
                    image_height=content_height,
                )
                focus_x, focus_y, focus_width, focus_height = focus
                proposals.append(
                    {
                        "proposal_id": "strategic-"
                        + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
                        "run": run,
                        "candidate": label,
                        "candidate_id": candidate_id,
                        "params_sha256": digest,
                        "frame_index": frame_index,
                        "elapsed_h": elapsed_h,
                        "suggested_phase": phase,
                        "suggestion_basis": (
                            "blind phase-hypothesis sampling reconstructed from the "
                            "candidate contour overlay, corrected optical change, "
                            "facies, time, and spatial position"
                        ),
                        "interpretation_scope": "visible_2d_slice_only",
                        "analysis_domain": domain_name,
                        "analysis_domain_margin_px": int(margin),
                        "coordinate_space": "normalized_native_review_canvas_16x9",
                        "shape": "freehand",
                        "points": points,
                        "x": min(point[0] for point in points),
                        "y": min(point[1] for point in points),
                        "width": max(point[0] for point in points)
                        - min(point[0] for point in points),
                        "height": max(point[1] for point in points)
                        - min(point[1] for point in points),
                        "analysis_polygon_px": points_px,
                        "analysis_bbox_xywh": [
                            left,
                            top,
                            right - left,
                            bottom - top,
                        ],
                        "focus_box": {
                            "x": (content_x + focus_x) / canvas_width,
                            "y": (content_y + focus_y) / canvas_height,
                            "width": focus_width / canvas_width,
                            "height": focus_height / canvas_height,
                        },
                        "model_stats": {
                            "sampling_stratum": stratum,
                            "phase_hypothesis": phase,
                            "phase_hypothesis_source": source_name,
                            "phase_hypothesis_confidence": float(
                                center["confidence"]
                            ),
                            "phase_hypothesis_is_dominant": bool(
                                center["phase_is_dominant"]
                            ),
                            "facies_id": facies_id,
                            "facies_fraction": float(center["facies_fraction"]),
                            "spatial_rank_within_phase_facies_frame": int(
                                center["spatial_rank"]
                            ),
                            "mean_gas_probability": float(
                                np.mean(gas_evidence[region], dtype=np.float64)
                            ),
                            "mean_aqueous_probability": (
                                None
                                if aqueous_evidence is None
                                else float(
                                    np.mean(
                                        aqueous_evidence[region],
                                        dtype=np.float64,
                                    )
                                )
                            ),
                            "mean_rgb_change_norm": mean_change,
                            "optical_change_quantile": rank,
                            "mean_delta_rgb": mean_delta_rgb.tolist(),
                            "mean_current_chromaticity": mean_current_chroma.tolist(),
                            "mean_reference_chromaticity": mean_reference_chroma.tolist(),
                            "normalized_center_xy": [
                                float(column) / max(1, content_width - 1),
                                float(row) / max(1, content_height - 1),
                            ],
                        },
                        "priority": float(
                            4.0 * float(center["confidence"])
                            + rank
                            + mean_change
                        ),
                    }
                )

    proposals.sort(
        key=lambda item: (
            -float(item.get("priority") or 0.0),
            int(item.get("frame_index") or 0),
            str(item.get("proposal_id") or ""),
        )
    )
    if max_total is None:
        return proposals
    return proposals[: max(0, int(max_total))]


def _select_optical_initial_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    max_total: int = review.PHASE_QUIZ_INITIAL_PROPOSALS_PER_CANDIDATE,
) -> list[dict[str, Any]]:
    """Select a CO2-enriched first batch with temporal coverage and one control."""

    if any(
        str(item.get("suggested_phase") or "") in PHASE_QUIZ_HYPOTHESES
        and (item.get("model_stats") or {}).get("facies_id") is not None
        for item in proposals
    ):
        return _select_blind_proposals(proposals, max_total=max_total)

    informative = [
        dict(item)
        for item in proposals
        if (item.get("model_stats") or {}).get("sampling_stratum")
        != "weak_change_negative_control"
    ]
    controls = [
        dict(item)
        for item in proposals
        if (item.get("model_stats") or {}).get("sampling_stratum")
        == "weak_change_negative_control"
    ]
    selected: list[dict[str, Any]] = []
    frame_indices = sorted({int(item.get("frame_index") or 0) for item in informative})
    target_count = max(0, min(max_total - (1 if controls else 0), len(frame_indices)))
    if target_count:
        target_frames = np.rint(
            np.linspace(frame_indices[0], frame_indices[-1], target_count)
        ).astype(int)
        used_ids: set[str] = set()
        for target in target_frames:
            candidates = [
                item
                for item in informative
                if str(item.get("proposal_id") or "") not in used_ids
            ]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda item: (
                    -abs(int(item.get("frame_index") or 0) - int(target)),
                    float(item.get("priority") or 0.0),
                ),
            )
            selected.append(chosen)
            used_ids.add(str(chosen.get("proposal_id") or ""))
    if controls and len(selected) < max_total:
        selected.append(max(controls, key=lambda item: float(item.get("priority") or 0.0)))
    selected_ids = {str(item.get("proposal_id") or "") for item in selected}
    remaining = sorted(
        (
            dict(item)
            for item in proposals
            if str(item.get("proposal_id") or "") not in selected_ids
        ),
        key=lambda item: -float(item.get("priority") or 0.0),
    )
    selected.extend(remaining[: max(0, max_total - len(selected))])
    return selected[:max_total]


def repair_phase_quiz_domains(
    dataset: str,
    *,
    output_root: Path | None = None,
    runs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write domain-filtered quiz assets without repeating model inference."""

    selected_runs = tuple(str(run).lower() for run in (runs or RUNS[dataset]))
    root = Path(output_root or DEFAULT_OUTPUTS[dataset]).resolve()
    review_root = root / "manual_pairwise_review"
    rows: list[dict[str, Any]] = []
    for run in selected_runs:
        source_paths = sorted(
            path
            for path in (review_root / "phase_quiz" / run).glob("*/proposals.json")
            if not path.parent.name.endswith(
                f"_q{review.PHASE_QUIZ_PROPOSAL_SCHEMA}"
            )
        )
        if not source_paths:
            # Some newer campaigns were published directly in the current
            # proposal schema and have no legacy folder. The current payload
            # still contains all candidate provenance needed for a safe repair.
            source_paths = sorted(
                (review_root / "phase_quiz" / run).glob(
                    f"*_q{review.PHASE_QUIZ_PROPOSAL_SCHEMA}/proposals.json"
                )
            )
        if not source_paths:
            rows.append({"run": run, "status": "missing_legacy_proposals"})
            continue
        native_manifest_path = review_root / "native_frames" / run / "manifest.json"
        native_manifest = (
            json.loads(native_manifest_path.read_text(encoding="utf-8-sig"))
            if native_manifest_path.is_file()
            else {}
        )
        content_box = native_manifest.get("content_box_xywh") or []
        expected_shape = native_manifest.get("analysis_image_shape_hw") or (
            [content_box[3], content_box[2]] if len(content_box) == 4 else []
        )
        serialized_facies = (
            _serialized_facies_map(dataset, run, expected_shape)
            if len(expected_shape) >= 2
            else None
        )
        if serialized_facies is not None:
            rig = type("SerializedFaciesRig", (), {"facies": serialized_facies})()
            domain, domain_name = _phase_quiz_domain(
                dataset,
                rig,
                np.ones(serialized_facies.shape, dtype=bool),
            )
            facies_map = _sampling_facies_map(rig, domain.shape, domain)
            if dataset == "ffum" and native_manifest:
                # FF-UM's protocol coordinates are not the optical source
                # locations. Rebuild these points so a quiz repair cannot
                # silently restore stale protocol-derived markers.
                with _open_source(dataset, run, require_clearwater=False) as source:
                    injection_points = _injection_overlay_points(
                        source,
                        canvas_wh=native_manifest.get("canvas_wh") or [],
                        content_box=content_box,
                    )
            else:
                injection_points = list(
                    native_manifest.get("injection_points") or []
                )
            domain_source = "serialized_rig_facies"
        else:
            with _open_source(dataset, run, require_clearwater=False) as source:
                domain = np.asarray(source.phase_quiz_mask, dtype=bool)
                domain_name = source.phase_quiz_domain
                facies_map = _sampling_facies_map(
                    source.rig,
                    source.baseline.shape,
                    domain,
                )
                injection_points = (
                    _injection_overlay_points(
                        source,
                        canvas_wh=native_manifest.get("canvas_wh") or [],
                        content_box=content_box,
                    )
                    if native_manifest
                    else []
                )
            domain_source = "rebuilt_run_context"
        if native_manifest:
            native_manifest["phase_quiz_domain"] = {
                "name": domain_name,
                "mask_fraction": float(np.mean(domain)),
                "quiz_boxes_require_full_containment": True,
                "excluded_from_sampling": PHASE_QUIZ_DOMAIN_EXCLUSIONS.get(
                    domain_name
                ),
                "source": domain_source,
            }
            native_manifest["injection_points"] = injection_points
            _atomic_json(native_manifest_path, native_manifest)
        candidate_rows: list[dict[str, Any]] = []
        for source_path in source_paths:
            payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
            original = [
                dict(item)
                for item in (payload.get("proposals") or [])
                if isinstance(item, Mapping)
            ]
            schema_suffix = f"_q{review.PHASE_QUIZ_PROPOSAL_SCHEMA}"
            if source_path.parent.name.endswith(schema_suffix):
                target = source_path
            else:
                target = (
                    source_path.parent.with_name(
                        source_path.parent.name + schema_suffix
                    )
                    / source_path.name
                )
            exemplar = original[0] if original else {}
            label = str(exemplar.get("candidate") or source_path.parent.name[:1])
            candidate_id = str(exemplar.get("candidate_id") or "")
            digest = str(exemplar.get("params_sha256") or "")
            optical = (
                _optical_active_learning_proposals(
                    run=run,
                    label=label,
                    candidate_id=candidate_id,
                    digest=digest,
                    domain=domain,
                    facies_map=facies_map,
                    domain_name=domain_name,
                    native_manifest=native_manifest,
                    review_root=review_root,
                )
                if native_manifest
                else []
            )
            initial = _select_optical_initial_proposals(optical)
            initial_ids = {
                str(item.get("proposal_id") or "") for item in initial
            }
            retained = initial + [
                item
                for item in optical
                if str(item.get("proposal_id") or "") not in initial_ids
            ]
            selection_mode = "strategic_phase_facies_time_position_pool"
            if not retained:
                retained = [
                    item for item in original if _proposal_inside_domain(item, domain)
                ]
                selection_mode = "filtered_source_proposals"
            if not retained and original and native_manifest:
                retained = _domain_only_repair_proposals(
                    run=run,
                    label=label,
                    candidate_id=candidate_id,
                    digest=digest,
                    domain=domain,
                    domain_name=domain_name,
                    native_manifest=native_manifest,
                )
                selection_mode = "domain_only_spatial_temporal_repair"

            # Keep safe prior proposal IDs so existing answers remain navigable,
            # but do not republish boxes outside the corrected porous domain.
            # Historical decisions remain intact in the review JSON.
            previous: list[dict[str, Any]] = []
            if target.is_file():
                previous_payload = json.loads(
                    target.read_text(encoding="utf-8-sig")
                )
                previous = [
                    dict(item)
                    for item in (previous_payload.get("proposals") or [])
                    if isinstance(item, Mapping)
                ]
            retained_ids = {
                str(item.get("proposal_id") or "") for item in retained
            }
            safe_previous = [
                item for item in previous if _proposal_inside_domain(item, domain)
            ]
            retained.extend(
                item
                for item in safe_previous
                if str(item.get("proposal_id") or "") not in retained_ids
            )
            replacement_count = len(optical)
            _atomic_json(
                target,
                {
                    **payload,
                    "schema": 4,
                    "proposal_schema": review.PHASE_QUIZ_PROPOSAL_SCHEMA,
                    "analysis_domain": domain_name,
                    "source_proposals": str(source_path),
                    "source_proposal_count": len(original),
                    "excluded_outside_domain": sum(
                        not _proposal_inside_domain(item, domain) for item in original
                    ),
                    "selection_mode": selection_mode,
                    "phase_hypothesis_stratification_active": bool(optical),
                    "sampling_dimensions": [
                        "phase_hypothesis",
                        "facies",
                        "time",
                        "position_within_facies",
                    ],
                    "proposal_pool_has_fixed_limit": False,
                    "adaptive_round_size": review.PHASE_QUIZ_ROUND_SIZE,
                    "initial_proposal_count": min(
                        review.PHASE_QUIZ_INITIAL_PROPOSALS_PER_CANDIDATE,
                        len(initial) if initial else len(retained),
                    ),
                    "available_proposal_count": len(retained),
                    "replacement_proposal_count": replacement_count,
                    "preserved_legacy_proposal_count": len(safe_previous),
                    "excluded_previous_outside_domain": (
                        len(previous) - len(safe_previous)
                    ),
                    "proposals": retained,
                },
            )
            candidate_rows.append(
                {
                    "source": str(source_path),
                    "target": str(target),
                    "before": len(original),
                    "after": len(retained),
                    "replacement_proposals": replacement_count,
                    "selection_mode": selection_mode,
                }
            )
        rows.append(
            {
                "run": run,
                "status": "complete",
                "analysis_domain": domain_name,
                "domain_source": domain_source,
                "contour_assets": _backfill_probability_overlay_assets(
                    review_root, run
                ),
                "candidates": candidate_rows,
            }
        )
    return {
        "schema": 1,
        "kind": "phase-quiz-domain-repair",
        "dataset": dataset,
        "runs": rows,
        "updated_utc": _utc_now(),
    }


def _injection_stages_match(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    tolerance_m: float = 1e-9,
) -> bool:
    if len(first) != len(second):
        return False
    return all(
        abs(float(left["x_m"]) - float(right["x_m"])) <= tolerance_m
        and abs(float(left["y_m"]) - float(right["y_m"])) <= tolerance_m
        for left, right in zip(first, second, strict=True)
    )


def repair_injection_anchors(
    dataset: str,
    *,
    output_root: Path | None = None,
    runs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Repair review markers while retaining candidate-generation provenance."""

    if dataset != "ffum":
        return {
            "schema": 1,
            "dataset": dataset,
            "status": "not_applicable",
            "reason": "Only FF-UM has image-derived source anchors.",
            "runs": [],
        }
    selected_runs = tuple(str(run).lower() for run in (runs or RUNS[dataset]))
    root = Path(output_root or DEFAULT_OUTPUTS[dataset]).resolve()
    review_root = root / "manual_pairwise_review"
    snapshot_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    snapshot_root = root / "injection_anchor_repair_snapshots" / snapshot_id
    rows: list[dict[str, Any]] = []
    for run in selected_runs:
        manifest_path = review_root / "native_frames" / run / "manifest.json"
        if not manifest_path.is_file():
            rows.append({"run": run, "status": "missing_manifest"})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        snapshot_path = snapshot_root / run / "manifest.json"
        _atomic_json(snapshot_path, manifest)
        generation_stages = [
            dict(stage)
            for stage in (
                (manifest.get("phase_topology") or {}).get("injection_stages")
                or []
            )
            if isinstance(stage, Mapping)
        ]
        with _open_source(dataset, run, require_clearwater=False) as source:
            corrected_stages = [
                dict(stage) for stage in _source_injection_stages(source)
            ]
            corrected_points = _injection_overlay_points(
                source,
                canvas_wh=manifest.get("canvas_wh") or [],
                content_box=manifest.get("content_box_xywh") or [],
                stages=corrected_stages,
            )
        generated_with_current = _injection_stages_match(
            generation_stages,
            corrected_stages,
        )
        manifest["injection_points"] = corrected_points
        manifest["injection_anchor_contract"] = {
            "schema": 1,
            "repaired_utc": _utc_now(),
            "coordinate_source": FFUM_VISIBLE_SOURCE_ANCHORS_M[0][
                "coordinate_source"
            ],
            "protocol_timing_and_rates_retained": True,
            "generation_injection_stages": generation_stages,
            "current_optical_injection_stages": corrected_stages,
            "candidate_outputs_generated_with_current_anchors": generated_with_current,
            "candidate_outputs_require_regeneration": not generated_with_current,
            "review_markers_use_current_anchors": True,
            "snapshot": str(snapshot_path),
        }
        _atomic_json(manifest_path, manifest)
        rows.append(
            {
                "run": run,
                "status": "repaired",
                "candidate_outputs_require_regeneration": not generated_with_current,
                "manifest": str(manifest_path),
                "snapshot": str(snapshot_path),
                "injection_points": corrected_points,
            }
        )
    return {
        "schema": 1,
        "dataset": dataset,
        "status": "complete",
        "snapshot_root": str(snapshot_root),
        "runs": rows,
    }


def _write_reference(
    review_root: Path,
    source: RunSource,
    *,
    width: int,
    height: int,
    normalized_evidence: bool = False,
) -> None:
    target = review_root / "native_references" / source.run / "pre_injection.jpg"
    frame = _plain_frame(
        source.baseline,
        run=source.run,
        index=0,
        count=1,
        elapsed_h=0.0,
        width=width,
        height=height,
        native=True,
        normalized_evidence=normalized_evidence,
    )
    _write_jpeg(target, frame)


def _infer_review_frame(
    prepared: _PreparedReviewFrame,
    previous: np.ndarray | None,
    baseline: np.ndarray,
    clearwater: np.ndarray | None,
    models: Mapping[str, DualPhaseSignalModel],
    aqueous_models: Mapping[str, DualPhaseSignalModel | None],
    aqueous_model_digests: Mapping[str, str | None],
    aqueous_calibrations: Mapping[str, Mapping[str, Any]],
    evaluation_mask: np.ndarray,
    *,
    dataset: str,
    hierarchical_phase: bool,
) -> _InferredReviewFrame:
    started = time.perf_counter()
    chunk_rows = 32 if dataset == "ffum" else 64
    packed_models: dict[str, DualPhaseSignalModel] = {}
    gas_score_keys: dict[str, str] = {}
    for name, model in models.items():
        key = f"gas::{_model_digest(model)}"
        packed_models.setdefault(key, model)
        gas_score_keys[name] = key
    aqueous_score_keys: dict[str, str | None] = {}
    for name, resolved_aqueous_model in aqueous_models.items():
        digest = aqueous_model_digests[name]
        if resolved_aqueous_model is None:
            aqueous_score_keys[name] = None
            continue
        if digest is None:
            raise RuntimeError(f"Missing aqueous-model digest for {name}")
        key = f"aqueous::{digest}"
        packed_models.setdefault(key, resolved_aqueous_model)
        aqueous_score_keys[name] = key

    packed_scores = _predict_model_collection(
        packed_models,
        prepared.image,
        baseline,
        previous,
        clearwater,
        chunk_rows=chunk_rows,
    )
    predictions = {name: packed_scores[gas_score_keys[name]] for name in models}
    aqueous_probabilities: dict[str, np.ndarray | None] = {}
    for name in aqueous_models:
        score_key = aqueous_score_keys[name]
        if score_key is None:
            aqueous_probabilities[name] = None
            continue
        calibrated = _apply_negative_calibration(
            packed_scores[score_key],
            aqueous_calibrations[name],
        )
        if hierarchical_phase:
            calibrated = np.asarray(calibrated, dtype=np.float32)
            calibrated[~evaluation_mask] = 0.0
        aqueous_probabilities[name] = calibrated
    return _InferredReviewFrame(
        prepared=prepared,
        predictions=predictions,
        aqueous_probabilities=aqueous_probabilities,
        inference_seconds=time.perf_counter() - started,
    )


def _resolve_review_phase_states(
    inferred: _InferredReviewFrame,
    conditional_gas_calibrations: Mapping[str, Mapping[str, Any]],
    evaluation_mask: np.ndarray,
    causal_context: Any,
    *,
    hierarchical_phase: bool,
) -> tuple[dict[str, _CandidateReviewState], float]:
    started = time.perf_counter()
    states: dict[str, _CandidateReviewState] = {}
    elapsed_h = inferred.prepared.frame_info.elapsed_h
    for name in MODEL_ORDER:
        aqueous_probability = inferred.aqueous_probabilities[name]
        raw_probability = inferred.predictions[name]
        conditional_probability = (
            _apply_negative_calibration(
                raw_probability,
                conditional_gas_calibrations[name],
            )
            if hierarchical_phase
            else raw_probability
        )
        if hierarchical_phase:
            conditional_probability = np.asarray(
                conditional_probability, dtype=np.float32
            )
            conditional_probability[~evaluation_mask] = 0.0
        if aqueous_probability is None:
            envelope_probability = conditional_probability
            aqueous_support = np.ones_like(
                conditional_probability, dtype=np.float32
            )
        elif hierarchical_phase:
            aqueous_support = np.asarray(aqueous_probability, dtype=np.float32)
            envelope_probability = (
                np.asarray(conditional_probability, dtype=np.float32)
                * aqueous_support
            )
        else:
            envelope_probability, aqueous_support = _apply_aqueous_envelope(
                conditional_probability,
                aqueous_probability,
            )
        probability, causal_frame = _apply_causal_topology(
            envelope_probability,
            aqueous_support,
            candidate_name=name,
            elapsed_h=elapsed_h,
            context=causal_context,
        )
        if hierarchical_phase:
            probability = np.asarray(probability, dtype=np.float32)
            probability[~evaluation_mask] = 0.0
        displayed_aqueous_probability = (
            aqueous_probability
            if not hierarchical_phase or aqueous_probability is None
            else np.clip(
                np.asarray(aqueous_probability, dtype=np.float32)
                - np.asarray(probability, dtype=np.float32),
                0.0,
                1.0,
            )
        )
        if hierarchical_phase and displayed_aqueous_probability is not None:
            displayed_aqueous_probability = np.asarray(
                displayed_aqueous_probability, dtype=np.float32
            )
            displayed_aqueous_probability[~evaluation_mask] = 0.0

        record = _probability_record(elapsed_h, probability, evaluation_mask)
        raw_record = _probability_record(
            elapsed_h, raw_probability, evaluation_mask
        )
        conditional_record = _probability_record(
            elapsed_h,
            conditional_probability,
            evaluation_mask,
        )
        envelope_record = _probability_record(
            elapsed_h,
            envelope_probability,
            evaluation_mask,
        )
        envelope_mean = float(envelope_record["gas_probability_mean"])
        causal_mean = float(record["gas_probability_mean"])
        record.update(
            {
                "raw_gas_probability_mean": raw_record[
                    "gas_probability_mean"
                ],
                "raw_support_fraction_p050": raw_record[
                    "support_fraction_p050"
                ],
                "conditional_gas_probability_mean": conditional_record[
                    "gas_probability_mean"
                ],
                "conditional_gas_support_fraction_p050": conditional_record[
                    "support_fraction_p050"
                ],
                "envelope_gas_probability_mean": envelope_record[
                    "gas_probability_mean"
                ],
                "envelope_support_fraction_p050": envelope_record[
                    "support_fraction_p050"
                ],
                "aqueous_probability_mean": (
                    0.0
                    if displayed_aqueous_probability is None
                    else float(
                        np.mean(
                            displayed_aqueous_probability[evaluation_mask],
                            dtype=np.float64,
                        )
                    )
                ),
                "co2_presence_probability_mean": (
                    0.0
                    if aqueous_probability is None
                    else float(
                        np.mean(
                            aqueous_probability[evaluation_mask],
                            dtype=np.float64,
                        )
                    )
                ),
                "aqueous_support_fraction_p050": float(
                    np.mean(aqueous_support[evaluation_mask] >= 0.50)
                ),
                "causal_component_count": causal_frame.component_count,
                "causal_retained_component_count": (
                    causal_frame.retained_component_count
                ),
                "causal_rejected_component_count": (
                    causal_frame.rejected_component_count
                ),
                "causal_source_supported_component_count": (
                    causal_frame.source_supported_component_count
                ),
                "causal_history_supported_component_count": (
                    causal_frame.history_supported_component_count
                ),
                "causal_reentry_supported_component_count": (
                    causal_frame.reentry_supported_component_count
                ),
                "causal_removed_gate_weight_fraction": (
                    causal_frame.removed_raw_gas_fraction
                ),
                "causal_removed_envelope_gas_fraction": (
                    max(0.0, 1.0 - causal_mean / envelope_mean)
                    if envelope_mean > 0.0
                    else 0.0
                ),
                "causal_plume_interior_gate_mean": (
                    causal_frame.plume_interior_gate_mean
                ),
                "raw_gas_plume_boundary_fraction": (
                    causal_frame.raw_gas_plume_boundary_fraction
                ),
                "retained_gas_plume_boundary_fraction": (
                    causal_frame.retained_gas_plume_boundary_fraction
                ),
                "causal_permeability_barrier_affinity_mean": (
                    causal_frame.permeability_barrier_affinity_mean
                ),
                "causal_downward_penalty_pixel_fraction": (
                    causal_frame.downward_penalty_pixel_fraction
                ),
                "causal_active_source_radius_scale": (
                    causal_frame.active_source_radius_scale
                ),
            }
        )
        states[name] = _CandidateReviewState(
            probability=probability,
            aqueous_probability=displayed_aqueous_probability,
            record=record,
        )
    return states, time.perf_counter() - started


def _render_review_frame(
    inferred: _InferredReviewFrame,
    states: Mapping[str, _CandidateReviewState],
    *,
    causal_seconds: float,
    run: str,
    review_root: Path,
    frame_count: int,
    baseline: np.ndarray,
    phase_quiz_mask: np.ndarray,
    phase_quiz_domain: str,
    quiz_facies: np.ndarray,
    blind_labels: Mapping[str, str],
    candidate_ids: Mapping[str, str],
    digests: Mapping[str, str],
    asset_fingerprints: Mapping[str, str],
    canvas_width: int,
    canvas_height: int,
    content_box: Sequence[int],
    hierarchical_phase: bool,
) -> _RenderedReviewFrame:
    started = time.perf_counter()
    prepared = inferred.prepared
    index = prepared.index
    frame_info = prepared.frame_info
    image = prepared.image
    plain_native = _plain_frame(
        image,
        run=run,
        index=index,
        count=frame_count,
        elapsed_h=frame_info.elapsed_h,
        width=canvas_width,
        height=canvas_height,
        native=True,
        normalized_evidence=hierarchical_phase,
    )
    native_path = review_root / "native_frames" / run / f"frame_{index:03d}.jpg"
    _write_jpeg(native_path, plain_native)
    plain_review = _plain_frame(
        image,
        run=run,
        index=index,
        count=frame_count,
        elapsed_h=frame_info.elapsed_h,
        width=review_movie.FRAME_WIDTH,
        height=review_movie.FRAME_HEIGHT,
        native=False,
        normalized_evidence=hierarchical_phase,
    )

    try:
        dx, dy, response, region = review._estimate_review_translation(
            baseline, image, header_height=0
        )
    except Exception:
        dx, dy, response, region = 0.0, 0.0, 0.0, "unavailable"
    registration = {
        "frame_index": index,
        "dx_px": float(dx),
        "dy_px": float(dy),
        "response": float(response),
        "region": region,
        "source": "native_corrected_images",
        "dx_canvas_fraction": float(dx) / canvas_width,
        "dy_canvas_fraction": float(dy) / canvas_height,
    }

    movie_frames: dict[str, np.ndarray] = {}
    proposals: dict[str, list[dict[str, Any]]] = {}
    records: dict[str, dict[str, Any]] = {}
    for name in MODEL_ORDER:
        state = states[name]
        records[name] = state.record
        overlay_native = _probability_overlay(
            state.probability,
            image.shape,
            aqueous_probability=state.aqueous_probability,
            width=canvas_width,
            height=canvas_height,
            preserve_native_resolution=True,
            normalized_evidence=hierarchical_phase,
        )
        overlay_path, _quiz_path = _candidate_asset_paths(
            review_root,
            run,
            blind_labels[name],
            asset_fingerprints[name],
            index,
        )
        _write_probability_overlay_assets(overlay_path, overlay_native)
        phase_map_native = _phase_fill_overlay(
            state.probability,
            image.shape,
            aqueous_probability=state.aqueous_probability,
            width=canvas_width,
            height=canvas_height,
            preserve_native_resolution=True,
        )
        phase_map_path = _candidate_phase_map_path(
            review_root,
            run,
            blind_labels[name],
            asset_fingerprints[name],
            index,
        )
        _write_phase_fill_overlay_assets(phase_map_path, phase_map_native)
        phase_map_review = _phase_fill_overlay(
            state.probability,
            image.shape,
            aqueous_probability=state.aqueous_probability,
            width=review_movie.FRAME_WIDTH,
            height=review_movie.FRAME_HEIGHT,
            preserve_native_resolution=False,
        )
        movie_frames[name] = _compose(plain_review, phase_map_review)
        proposals[name] = _blind_proposals(
            run=run,
            label=blind_labels[name],
            candidate_id=candidate_ids[name],
            digest=digests[name],
            frame_index=index,
            frame_count=frame_count,
            elapsed_h=frame_info.elapsed_h,
            image=image,
            baseline=baseline,
            probability=state.probability,
            aqueous_probability=state.aqueous_probability,
            facies_map=quiz_facies,
            analysis_domain_mask=phase_quiz_mask,
            analysis_domain_name=phase_quiz_domain,
            canvas_wh=(canvas_width, canvas_height),
            content_box=content_box,
        )

    render_seconds = time.perf_counter() - started
    frame_source = {
        "frame_index": index,
        "elapsed_h": float(frame_info.elapsed_h),
        "source_name": frame_info.path.name,
        "source_path": str(frame_info.path),
        "requested_name": frame_info.path.name,
        "selection_status": prepared.selection_status,
        "color_correction_active": bool(prepared.correction_rows),
        "color_correction_flagged": any(
            bool(row.get("flagged")) for row in prepared.correction_rows
        ),
        "color_corrections": prepared.correction_rows,
        "corrected_array_sha256": _array_sha256(image),
        "frame_seconds": (
            prepared.read_seconds
            + inferred.inference_seconds
            + causal_seconds
            + render_seconds
        ),
        "pipeline_stage_seconds": {
            "read_and_correct": prepared.read_seconds,
            "raw_inference": inferred.inference_seconds,
            "causal_resolution": causal_seconds,
            "render_and_write": render_seconds,
        },
    }
    return _RenderedReviewFrame(
        frame_source=frame_source,
        registration=registration,
        records=records,
        movie_frames=movie_frames,
        proposals=proposals,
    )


def _write_run(
    dataset: str,
    run: str,
    output_root: Path,
    models: Mapping[str, DualPhaseSignalModel],
    jobs_by_name: Mapping[str, Mapping[str, Any]],
    aqueous_model: (
        DualPhaseSignalModel | Mapping[str, DualPhaseSignalModel] | None
    ) = None,
    aqueous_model_paths: Mapping[str, Path] | None = None,
    *,
    hierarchical_phase: bool = False,
) -> dict[str, Any]:
    review_root = output_root / "manual_pairwise_review"
    aqueous_models = _aqueous_models_by_candidate(models, aqueous_model)
    digests = {
        name: (
            _composite_candidate_digest(models[name], aqueous_models[name])
            if isinstance(aqueous_model, Mapping)
            else _model_digest(models[name])
        )
        for name in MODEL_ORDER
    }
    candidate_ids = {
        name: str(jobs_by_name[name]["candidate_id"]) for name in MODEL_ORDER
    }
    asset_fingerprints = {
        name: _candidate_asset_fingerprint(candidate_ids[name], digests[name])
        for name in MODEL_ORDER
    }
    blind_labels = _blind_labels(run, candidate_ids)
    candidate_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_ORDER
    }
    candidate_movie_frames: dict[str, list[np.ndarray]] = {
        name: [] for name in MODEL_ORDER
    }
    candidate_proposals: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_ORDER
    }
    frame_sources: list[dict[str, Any]] = []
    registration: list[dict[str, Any]] = []
    started = time.perf_counter()
    requires_clearwater = any(
        OpticalFeatureContract.from_dict(model.feature_contract or {}).reference_adapter
        != "none"
        for model in models.values()
    )

    with _open_source(
        dataset,
        run,
        require_clearwater=requires_clearwater,
    ) as source:
        preflight_path = output_root / "preflight" / run / "negative_controls.json"
        preflight: dict[str, Any] | None = None
        preflight_reused = False
        if preflight_path.is_file():
            try:
                existing_preflight = json.loads(
                    preflight_path.read_text(encoding="utf-8-sig")
                )
                if isinstance(
                    existing_preflight, Mapping
                ) and _negative_control_preflight_is_reusable(
                    existing_preflight,
                    models,
                    source,
                    aqueous_model,
                    hierarchical_phase=hierarchical_phase,
                ):
                    preflight = dict(existing_preflight)
                    preflight_reused = True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                preflight = None
        if preflight is None:
            preflight = _negative_control_preflight(
                models,
                source,
                aqueous_model,
                hierarchical_phase=hierarchical_phase,
            )
            _atomic_json(preflight_path, preflight)
        if not bool(preflight["passed"]):
            raise NegativeControlPreflightError(run, preflight_path, preflight)
        aqueous_calibrations = dict(
            preflight["aqueous_negative_calibration_by_candidate"]
        )
        conditional_gas_calibrations = dict(
            preflight["conditional_gas_negative_calibration_by_candidate"]
        )
        evaluation_mask = _phase_evaluation_mask(
            source,
            hierarchical_phase=hierarchical_phase,
        )
        height_px, width_px = source.baseline.shape[:2]
        canvas_width, canvas_height = _native_dimensions(source.baseline.shape)
        content_box = review._native_content_box(
            source.baseline.shape, canvas_width, canvas_height
        )
        quiz_facies = _sampling_facies_map(
            source.rig,
            source.baseline.shape,
            source.phase_quiz_mask,
        )
        _write_reference(
            review_root,
            source,
            width=canvas_width,
            height=canvas_height,
            normalized_evidence=hierarchical_phase,
        )
        causal_context = _causal_topology_context(source, MODEL_ORDER)
        baseline_causal_context = _causal_topology_context(source, MODEL_ORDER)
        injection_points = _injection_overlay_points(
            source,
            canvas_wh=(canvas_width, canvas_height),
            content_box=content_box,
            stages=causal_context.injection_stages,
        )
        total_frame_count = len(source.frames) + 1
        pipeline_settings = _review_pipeline_settings(dataset, total_frame_count)
        aqueous_model_digests = {
            name: None if model is None else _model_digest(model)
            for name, model in aqueous_models.items()
        }
        pending_inference: deque[Future[_InferredReviewFrame]] = deque()
        pending_render: deque[Future[_RenderedReviewFrame]] = deque()
        previous: np.ndarray | None = None

        raw_baseline_path = getattr(source.rig, "baseline_path", None)
        baseline_path = (
            Path(raw_baseline_path)
            if raw_baseline_path not in (None, "")
            else Path(f"{run}_pre_injection_baseline")
        )
        baseline_prepared = _PreparedReviewFrame(
            index=0,
            frame_info=FrameInfo(
                path=baseline_path,
                elapsed_h=-1.0e-6,
            ),
            image=source.baseline,
            correction_rows=_correction_diagnostics(source.rig),
            read_seconds=0.0,
            selection_status="pre_injection_negative_control",
        )
        baseline_inferred = _infer_review_frame(
            baseline_prepared,
            None,
            source.baseline,
            source.clearwater,
            models,
            aqueous_models,
            aqueous_model_digests,
            aqueous_calibrations,
            evaluation_mask,
            dataset=dataset,
            hierarchical_phase=hierarchical_phase,
        )
        baseline_states, baseline_causal_seconds = _resolve_review_phase_states(
            baseline_inferred,
            conditional_gas_calibrations,
            evaluation_mask,
            baseline_causal_context,
            hierarchical_phase=hierarchical_phase,
        )
        baseline_rendered = _render_review_frame(
            baseline_inferred,
            baseline_states,
            causal_seconds=baseline_causal_seconds,
            run=run,
            review_root=review_root,
            frame_count=total_frame_count,
            baseline=source.baseline,
            phase_quiz_mask=source.phase_quiz_mask,
            phase_quiz_domain=source.phase_quiz_domain,
            quiz_facies=quiz_facies,
            blind_labels=blind_labels,
            candidate_ids=candidate_ids,
            digests=digests,
            asset_fingerprints=asset_fingerprints,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            content_box=content_box,
            hierarchical_phase=hierarchical_phase,
        )
        frame_sources.append(baseline_rendered.frame_source)
        registration.append(baseline_rendered.registration)
        for candidate_name in MODEL_ORDER:
            candidate_rows[candidate_name].append(
                baseline_rendered.records[candidate_name]
            )
            candidate_movie_frames[candidate_name].append(
                baseline_rendered.movie_frames[candidate_name]
            )
            candidate_proposals[candidate_name].extend(
                baseline_rendered.proposals[candidate_name]
            )

        def collect_oldest_render() -> None:
            rendered = pending_render.popleft().result()
            frame_sources.append(rendered.frame_source)
            registration.append(rendered.registration)
            for candidate_name in MODEL_ORDER:
                candidate_rows[candidate_name].append(
                    rendered.records[candidate_name]
                )
                candidate_movie_frames[candidate_name].append(
                    rendered.movie_frames[candidate_name]
                )
                candidate_proposals[candidate_name].extend(
                    rendered.proposals[candidate_name]
                )

        with (
            ThreadPoolExecutor(
                max_workers=pipeline_settings.inference_workers,
                thread_name_prefix=f"{run}-phase-inference",
            ) as inference_executor,
            ThreadPoolExecutor(
                max_workers=pipeline_settings.render_workers,
                thread_name_prefix=f"{run}-phase-render",
            ) as render_executor,
        ):

            def resolve_oldest_inference() -> None:
                inferred = pending_inference.popleft().result()
                states, causal_seconds = _resolve_review_phase_states(
                    inferred,
                    conditional_gas_calibrations,
                    evaluation_mask,
                    causal_context,
                    hierarchical_phase=hierarchical_phase,
                )
                pending_render.append(
                    render_executor.submit(
                        _render_review_frame,
                        inferred,
                        states,
                        causal_seconds=causal_seconds,
                        run=run,
                        review_root=review_root,
                        frame_count=total_frame_count,
                        baseline=source.baseline,
                        phase_quiz_mask=source.phase_quiz_mask,
                        phase_quiz_domain=source.phase_quiz_domain,
                        quiz_facies=quiz_facies,
                        blind_labels=blind_labels,
                        candidate_ids=candidate_ids,
                        digests=digests,
                        asset_fingerprints=asset_fingerprints,
                        canvas_width=canvas_width,
                        canvas_height=canvas_height,
                        content_box=content_box,
                        hierarchical_phase=hierarchical_phase,
                    )
                )
                if (
                    len(pending_render)
                    >= pipeline_settings.max_buffered_frames
                ):
                    collect_oldest_render()

            for index, frame_info in enumerate(source.frames, start=1):
                read_started = time.perf_counter()
                image, correction_rows = source.read(frame_info.path)
                if image.shape != source.baseline.shape:
                    raise RuntimeError(
                        f"{run} image shape {image.shape} differs from baseline "
                        f"{source.baseline.shape}"
                    )
                prepared = _PreparedReviewFrame(
                    index=index,
                    frame_info=frame_info,
                    image=image,
                    correction_rows=correction_rows,
                    read_seconds=time.perf_counter() - read_started,
                )
                pending_inference.append(
                    inference_executor.submit(
                        _infer_review_frame,
                        prepared,
                        previous,
                        source.baseline,
                        source.clearwater,
                        models,
                        aqueous_models,
                        aqueous_model_digests,
                        aqueous_calibrations,
                        evaluation_mask,
                        dataset=dataset,
                        hierarchical_phase=hierarchical_phase,
                    )
                )
                previous = image
                if (
                    len(pending_inference)
                    >= pipeline_settings.max_buffered_frames
                ):
                    resolve_oldest_inference()

            while pending_inference:
                resolve_oldest_inference()
            while pending_render:
                collect_oldest_render()

        native_manifest = {
            "schema": 1,
            "run": run,
            "dataset": dataset,
            "kind": "native_resolution_corrected_review_frames",
            "frame_count": total_frame_count,
            "canvas_wh": [canvas_width, canvas_height],
            "analysis_image_shape_hw": [height_px, width_px],
            "content_box_xywh": list(content_box),
            "full_analysis_resolution": True,
            "additional_spatial_resampling": False,
            "jpeg_quality": 98,
            "generation_pipeline": {
                "schema": 1,
                "architecture": "bounded-read-infer-causal-render-v1",
                "inference_workers": pipeline_settings.inference_workers,
                "render_workers": pipeline_settings.render_workers,
                "max_buffered_frames": pipeline_settings.max_buffered_frames,
                "read_and_color_correction": (
                    "chronological-single-owner-for-stateful-diagnostics"
                ),
                "raw_optical_inference": "parallel-across-frames",
                "shared_optical_workspace_across_phase_heads": True,
                "causal_gas_topology": "strictly-chronological",
                "overlay_quiz_render_and_write": "parallel-across-frames",
            },
            "base_image_shared_between_candidates": True,
            "contours_stored_as_transparent_overlays": True,
            "phase_specific_contour_overlays": ["gas", "aqueous"],
            "phase_maps_stored_as_transparent_overlays": True,
            "phase_map_rendering": "mutually_exclusive_phase_fill_v1",
            "phase_map_classes": ["aqueous", "mixed", "gas"],
            "phase_map_display_contract": {
                "minimum_co2_evidence": PHASE_FILL_MIN_CO2_EVIDENCE,
                "aqueous_maximum_gas_share": PHASE_FILL_AQUEOUS_MAX_GAS_SHARE,
                "gas_minimum_gas_share": PHASE_FILL_GAS_MIN_GAS_SHARE,
                "transparent_semantics": "no_visible_co2_predicted",
                "classification_semantics": "model_display_class_not_ground_truth",
            },
            "reference": {
                "kind": "radiometrically_corrected_pre_injection_baseline",
                "corrected_array_sha256": _array_sha256(source.baseline),
                "included_in_timeline": True,
                "timeline_frame_index": 0,
                "timeline_selection_status": "pre_injection_negative_control",
                "model_evaluated_with_same_phase_pipeline": True,
                "causal_history_isolated_from_experiment_frames": True,
            },
            "clearwater_reference": (
                None
                if source.clearwater_contract is None
                else dict(source.clearwater_contract)
            ),
            "negative_control_preflight": {
                "passed": True,
                "path": str(preflight_path),
                "sha256": _sha256(preflight_path),
                "reused_exact_input_contract": preflight_reused,
            },
            "phase_topology": {
                "aqueous_envelope_model_active": any(
                    model is not None for model in aqueous_models.values()
                ),
                "aqueous_model_by_candidate": {
                    name: (
                        None
                        if model is None
                        else {"model_sha256": _model_digest(model)}
                    )
                    for name, model in aqueous_models.items()
                },
                "gas_probability_is_post_envelope": any(
                    model is not None for model in aqueous_models.values()
                ),
                "hierarchical_phase_partition_active": bool(hierarchical_phase),
                "hierarchical_partition_rule": (
                    "P(gas)=P(CO2)*P(gas|CO2); P(aqueous)=max(P(CO2)-P(gas),0)"
                    if hierarchical_phase
                    else None
                ),
                "causal_source_history_gate_active": True,
                "causal_topology_scale": CAUSAL_TOPOLOGY_SCALE,
                "causal_settings": dict(CAUSAL_GAS_SETTINGS.__dict__),
                "causal_settings_by_candidate": {
                    name: dict(settings.__dict__)
                    for name, settings in CAUSAL_GAS_SETTINGS_BY_CANDIDATE.items()
                },
                "injection_stages": [
                    dict(stage) for stage in causal_context.injection_stages
                ],
                "gas_probability_is_post_causal_gate": True,
                "raw_gas_metrics_retained": True,
                "known_negative_score_normalization": {
                    "co2_presence_by_candidate": preflight[
                        "aqueous_negative_calibration_by_candidate"
                    ],
                    "conditional_gas_by_candidate": preflight[
                        "conditional_gas_negative_calibration_by_candidate"
                    ],
                    "applied_to_all_experiment_frames_without_refitting": True,
                },
                "pre_causal_envelope_metrics_retained": True,
                "visible_slice_caveat": (
                    "The rule is soft because connected gas may enter or leave "
                    "the camera-visible 2D slice within the 3D cell."
                ),
            },
            "injection_anchor_contract": {
                "schema": 1,
                "coordinate_source": str(
                    causal_context.injection_stages[0].get(
                        "coordinate_source",
                        "protocol_and_rig_coordinate_system",
                    )
                ),
                "protocol_timing_and_rates_retained": True,
                "current_optical_injection_stages": [
                    dict(stage) for stage in causal_context.injection_stages
                ],
                "candidate_outputs_generated_with_current_anchors": True,
                "candidate_outputs_require_regeneration": False,
                "review_markers_use_current_anchors": True,
            },
            "injection_points": injection_points,
            "phase_quiz_domain": {
                "name": source.phase_quiz_domain,
                "excluded_from_sampling": PHASE_QUIZ_DOMAIN_EXCLUSIONS.get(
                    source.phase_quiz_domain
                ),
            },
            "phase_evaluation_domain": {
                "name": source.phase_evaluation_domain,
                "mask_sha256": _array_sha256(evaluation_mask),
                "separate_from_quiz_and_mass_masks": True,
                "continuous_image_porosity_used_as_hard_mask": False,
                "purpose": "broad optical phase inference over the porous medium",
            },
            "candidate_asset_identity": {
                name: {
                    "candidate_id": candidate_ids[name],
                    "blind_label": blind_labels[name],
                    "model_digest": digests[name],
                    "asset_fingerprint": asset_fingerprints[name],
                }
                for name in MODEL_ORDER
            },
            "frame_sources": frame_sources,
            "reference_registration": registration,
            "correction_contract": dict(source.correction_contract),
            "created_utc": _utc_now(),
        }
        _atomic_json(
            review_root / "native_frames" / run / "manifest.json",
            native_manifest,
        )

    summaries: dict[str, Any] = {}
    for name in MODEL_ORDER:
        job = jobs_by_name[name]
        folder = _candidate_path(output_root, job)
        folder.mkdir(parents=True, exist_ok=True)
        records = candidate_rows[name]
        plot_path = folder / "calibration_mass_fit.png"
        movie_path = folder / "calibration_review.mp4"
        csv_path = folder / "calibration_review_per_frame.csv"
        _write_probability_plot(
            plot_path,
            run,
            records,
            normalized_evidence=hierarchical_phase,
        )
        movie_details = review_movie.write_calibration_movie(
            output=movie_path,
            plot_frame=cv2.cvtColor(
                cv2.imread(str(plot_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            ),
            contour_frames=candidate_movie_frames[name],
            gas_levels=PROBABILITY_LEVELS,
        )
        movie_details["params_sha256"] = digests[name]
        _atomic_csv(csv_path, records)
        initial_proposals = _select_blind_proposals(
            candidate_proposals[name],
            max_total=review.PHASE_QUIZ_INITIAL_PROPOSALS_PER_CANDIDATE,
        )
        proposal_pool = _select_blind_proposals(
            candidate_proposals[name],
            max_total=None,
        )
        initial_ids = {
            str(proposal.get("proposal_id") or "") for proposal in initial_proposals
        }
        proposals = initial_proposals + [
            proposal
            for proposal in proposal_pool
            if str(proposal.get("proposal_id") or "") not in initial_ids
        ]
        _overlay_path, quiz_path = _candidate_asset_paths(
            review_root,
            run,
            blind_labels[name],
            asset_fingerprints[name],
            0,
        )
        _atomic_json(
            quiz_path,
            {
                "schema": 2,
                "kind": "blind-visible-phase-sampling",
                "five_class_model_active": False,
                "model_output_used_for_stratification_only": True,
                "phase_hypothesis_stratification_active": True,
                "sampling_dimensions": [
                    "phase_hypothesis",
                    "facies",
                    "time",
                    "position_within_facies",
                ],
                "proposal_pool_has_fixed_limit": False,
                "analysis_domain": source.phase_quiz_domain,
                "initial_proposal_count": len(initial_proposals),
                "available_proposal_count": len(proposals),
                "proposals": proposals,
            },
        )
        summary = {
            "schema": 2,
            "kind": (
                "schema2-binary-phase-transfer-evidence-candidate"
                if hierarchical_phase
                else "schema2-binary-phase-transfer-candidate"
            ),
            "dataset": dataset,
            "run": run,
            "selection_candidate_id": candidate_ids[name],
            "normalized_mae": None,
            "review_plot_title": (
                f"{run.upper()} normalized optical phase-evidence diagnostics"
                if hierarchical_phase
                else f"{run.upper()} optical free-gas probability diagnostics"
            ),
            "calibration_review_movie": movie_details,
            "model": {
                "path": str(MODEL_PATHS[name]),
                "model_sha256": _model_digest(models[name]),
                "candidate_digest": digests[name],
                "review_asset_fingerprint": asset_fingerprints[name],
                "schema": 2,
                "training_runs": sorted(
                    set(models[name].training_provenance.get("runs") or [])
                ),
                "target_used_for_fitting": False,
            },
            "co2_presence_model": (
                None
                if aqueous_models[name] is None
                else {
                    "path": (
                        None
                        if aqueous_model_paths is None
                        else str(aqueous_model_paths[name])
                    ),
                    "model_sha256": _model_digest(aqueous_models[name]),
                    "target_kind": "co2_presence",
                }
            ),
            "hierarchical_phase_partition_active": bool(hierarchical_phase),
            "known_negative_score_normalization": {
                "co2_presence": preflight[
                    "aqueous_negative_calibration_by_candidate"
                ][name],
                "conditional_gas": preflight[
                    "conditional_gas_negative_calibration_by_candidate"
                ][name],
            },
            "five_class_model_active": False,
            "phase_quiz": {
                "mode": "blind-unscored",
                "proposal_count": len(initial_proposals),
                "proposal_pool_count": len(proposals),
                "labels_collected": [
                    "water",
                    "aqueous",
                    "gas",
                    "mixed",
                    "uncertain",
                ],
            },
            "frame_count": len(records),
            "created_utc": _utc_now(),
        }
        summaries[name] = summary
        # The summary is the final candidate completion marker.  A run becomes
        # visible only after both summaries and all native assets exist.
        _atomic_json(folder / "summary.json", summary)

    return {
        "run": run,
        "dataset": dataset,
        "status": "complete",
        "frame_count": len(frame_sources),
        "blind_labels": blind_labels,
        "seconds": time.perf_counter() - started,
        "completed_utc": _utc_now(),
        "summaries": summaries,
    }


def _campaign_manifest(
    dataset: str,
    output_root: Path,
    runs: Sequence[str],
    jobs: Sequence[Mapping[str, Any]],
    models: Mapping[str, DualPhaseSignalModel],
) -> dict[str, Any]:
    return {
        "schema": 2,
        "campaign_id": CAMPAIGN_ID,
        "dataset": dataset,
        "selection_id": f"{CAMPAIGN_ID}_{dataset}",
        "created_utc": _utc_now(),
        "output_root": str(output_root),
        "runs": list(runs),
        "jobs": list(jobs),
        "design": {
            "comparison": "schema2-matched-optical-model-pair",
            "optical_profile": "uniform24",
            "training_runs": ["ac60", "ac61"],
            "target_runs_disjoint_from_training": not bool(
                set(runs) & {"ac60", "ac61"}
            ),
            "target_used_for_refitting": False,
            "five_class_model_active": False,
            "quiz_regions": "blind-unscored-stratified-samples",
            "interpretation_scope": "visible-2d-slice-only",
            "shared_method_components": [
                "radiometric correction contract",
                "schema-2 optical feature extraction",
                "packed prefix-compatible paired-model inference",
                "candidate blinding",
                "native-resolution contour rendering",
                "quiz sampling and review storage",
                "known-negative baseline and clear-water preflight",
                "soft aqueous-envelope support for free-gas evidence",
                "protocol-seeded causal source/history/re-entry topology",
            ],
            "dataset_specific_inputs": {
                "ffac": ["rig data", "protocol", "raw images"],
                "ffum": ["rig data", "protocol", "raw images"],
            },
            "legacy_ffum_calibration_or_selection_used": False,
            "depth_contract": {
                "used_in_this_2d_phase_test": False,
                "reason": "Depth enters experiment-specific 3D mass conversion, not optical phase classification.",
                "future_mass_analysis": "Use each dataset's own depth map through the shared mass engine.",
            },
        },
        "models": {
            name: {
                "path": str(MODEL_PATHS[name]),
                "file_sha256": _sha256(MODEL_PATHS[name]),
                "model_sha256": _model_digest(models[name]),
                "feature_contract": models[name].feature_contract,
            }
            for name in MODEL_ORDER
        },
        "aqueous_envelope_model": (
            None
            if AQUEOUS_MODEL_PATH is None
            else {
                "path": str(AQUEOUS_MODEL_PATH),
                "file_sha256": _sha256(AQUEOUS_MODEL_PATH),
                "model_sha256": _model_digest(_load_aqueous_model()),
            }
        ),
        "review": {
            "blind_seed": BLIND_SEED,
            "incremental_publication": True,
            "output_dir": str(output_root / "manual_pairwise_review"),
        },
    }


def run_campaign(
    dataset: str,
    *,
    output_root: Path | None = None,
    runs: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    optical_control.install_runtime_hooks()
    models = _load_models()
    aqueous_model = _load_aqueous_model()
    selected_runs = tuple(str(run).lower() for run in (runs or RUNS[dataset]))
    if set(selected_runs) & {"ac60", "ac61"}:
        raise RuntimeError("Training runs AC60/AC61 cannot enter the transfer campaign")
    root = Path(output_root or DEFAULT_OUTPUTS[dataset]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs = _candidate_jobs(dataset, selected_runs, models)
    manifest = _campaign_manifest(dataset, root, selected_runs, jobs, models)
    _atomic_json(root / "postflight_manifest.json", manifest)
    _atomic_json(root / "campaign_manifest.json", manifest)
    jobs_by_run = {
        run: {
            str(job["role"]): job
            for job in jobs
            if str(job["run"]).lower() == run
        }
        for run in selected_runs
    }
    state_path = root / "campaign_state.json"
    completed: list[dict[str, Any]] = []
    _atomic_json(
        state_path,
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "dataset": dataset,
            "status": "running",
            "completed": [],
            "remaining": list(selected_runs),
            "started_utc": _utc_now(),
            "updated_utc": _utc_now(),
        },
    )
    for run in selected_runs:
        required = [
            _candidate_path(root, jobs_by_run[run][name]) / "summary.json"
            for name in MODEL_ORDER
        ]
        native_manifest = root / "manual_pairwise_review" / "native_frames" / run / "manifest.json"
        if not force and native_manifest.is_file() and all(path.is_file() for path in required):
            completed.append({"run": run, "dataset": dataset, "status": "reused"})
        else:
            try:
                completed.append(
                    _write_run(
                        dataset,
                        run,
                        root,
                        models,
                        jobs_by_run[run],
                        aqueous_model,
                    )
                )
            except NegativeControlPreflightError as exc:
                completed.append(
                    {
                        "run": run,
                        "dataset": dataset,
                        "status": "blocked_negative_control_preflight",
                        "report": str(exc.report_path),
                        "failures": [
                            {
                                "control": row["control"],
                                "model": row["model"],
                                "failures": row["failures"],
                            }
                            for row in exc.report.get("controls") or []
                            if not bool(row.get("passed"))
                        ],
                        "completed_utc": _utc_now(),
                    }
                )
        _atomic_json(
            state_path,
            {
                "schema": 1,
                "campaign_id": CAMPAIGN_ID,
                "dataset": dataset,
                "status": "running",
                "completed": completed,
                "remaining": [
                    candidate
                    for candidate in selected_runs
                    if candidate not in {row["run"] for row in completed}
                ],
                "updated_utc": _utc_now(),
            },
        )
    blocked = [
        row
        for row in completed
        if str(row.get("status", "")).startswith("blocked_")
    ]
    result = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "dataset": dataset,
        "status": "complete" if not blocked else "completed_with_blocked_runs",
        "completed": completed,
        "blocked": blocked,
        "remaining": [],
        "updated_utc": _utc_now(),
    }
    _atomic_json(state_path, result)
    _atomic_json(root / "campaign_terminal.json", result)
    return result


def run_negative_control_preflight(
    dataset: str,
    *,
    output_root: Path | None = None,
    runs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate target no-gas controls without rendering campaign frames."""

    optical_control.install_runtime_hooks()
    models = _load_models()
    aqueous_model = _load_aqueous_model()
    selected_runs = tuple(str(run).lower() for run in (runs or RUNS[dataset]))
    root = Path(output_root or DEFAULT_OUTPUTS[dataset]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    requires_clearwater = any(
        OpticalFeatureContract.from_dict(model.feature_contract or {}).reference_adapter
        != "none"
        for model in models.values()
    )
    reports: list[dict[str, Any]] = []
    for run in selected_runs:
        with _open_source(
            dataset,
            run,
            require_clearwater=requires_clearwater,
        ) as source:
            report = _negative_control_preflight(models, source, aqueous_model)
        report_path = root / "preflight" / run / "negative_controls.json"
        _atomic_json(report_path, report)
        reports.append(
            {
                "run": run,
                "passed": bool(report["passed"]),
                "report": str(report_path),
                "failed_controls": sum(
                    not bool(row["passed"])
                    for row in report.get("controls") or []
                ),
            }
        )
    result = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "dataset": dataset,
        "kind": "negative-control-preflight-only",
        "status": "passed" if all(row["passed"] for row in reports) else "failed",
        "runs": reports,
        "updated_utc": _utc_now(),
    }
    _atomic_json(root / "negative_control_preflight_summary.json", result)
    return result


def _status(dataset: str, output_root: Path | None = None) -> dict[str, Any]:
    root = Path(output_root or DEFAULT_OUTPUTS[dataset]).resolve()
    path = root / "campaign_state.json"
    if not path.is_file():
        return {"dataset": dataset, "status": "not_started", "root": str(root)}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "run",
            "preflight",
            "repair-quiz",
            "repair-injection-anchors",
            "status",
        ),
        nargs="?",
        default="run",
    )
    parser.add_argument("--dataset", choices=("ffac", "ffum", "all"), default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--force", action="store_true")
    _add_review_pipeline_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _apply_review_pipeline_arguments(args)
    datasets = ("ffac", "ffum") if args.dataset == "all" else (args.dataset,)
    if args.output_root is not None and len(datasets) != 1:
        raise SystemExit("--output-root requires one explicit dataset")
    results = []
    for dataset in datasets:
        if args.action == "status":
            results.append(_status(dataset, args.output_root))
        elif args.action == "repair-quiz":
            results.append(
                repair_phase_quiz_domains(
                    dataset,
                    output_root=args.output_root,
                    runs=args.runs,
                )
            )
        elif args.action == "repair-injection-anchors":
            results.append(
                repair_injection_anchors(
                    dataset,
                    output_root=args.output_root,
                    runs=args.runs,
                )
            )
        elif args.action == "preflight":
            results.append(
                run_negative_control_preflight(
                    dataset,
                    output_root=args.output_root,
                    runs=args.runs,
                )
            )
        else:
            results.append(
                run_campaign(
                    dataset,
                    output_root=args.output_root,
                    runs=args.runs,
                    force=args.force,
                )
            )
    print(json.dumps({"results": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

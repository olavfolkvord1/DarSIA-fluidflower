"""ff_ac-native auto-calibration objective (mass-balance), with the ff_um queue contract.

The science and orchestration are ff_um's: a per-label Optuna search over the
piecewise signal-model values that minimises the classic mass-balance error

    objective = sum_over_calibration_images | detected_total_mass - injected_mass |

but the *evaluation* is written against ff_ac's current DarSIA preset API
(``prepare_analysis_context`` + ``HeterogeneousColorToMassAnalysis``), which
loads experiment/rig/geometry/color-to-mass and resolves all the calibration
config internally. This module exposes exactly the names the ported
``distributed_auto_calibration_queue.py`` imports.
"""
from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path, resolve_storage_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import random
import copy
import csv
import hashlib
import json
import logging
import os
import pickle
import re
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from darsia.fluidflower.tools.ffac_titration_flash import TitrationFlash
except ModuleNotFoundError:
    from darsia.fluidflower.tools.ffac_titration_flash import TitrationFlash

try:
    from darsia.fluidflower.tools import colorchecker_refinement
except ModuleNotFoundError:
    from darsia.fluidflower.tools import colorchecker_refinement
try:
    from darsia.fluidflower.tools.aqueous_noise_gate import (
        AqueousNoiseGate,
        estimate_aqueous_noise_gate,
        normalise_aqueous_noise_gate_mode,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.aqueous_noise_gate import (
        AqueousNoiseGate,
        estimate_aqueous_noise_gate,
        normalise_aqueous_noise_gate_mode,
    )

try:
    from darsia.fluidflower.tools.observation_support import (
        SpatialAqueousSupportModel,
        btb_acidification_coordinate,
        estimate_spatial_aqueous_support,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.observation_support import (
        SpatialAqueousSupportModel,
        btb_acidification_coordinate,
        estimate_spatial_aqueous_support,
    )

try:
    from darsia.fluidflower.tools.color_path_evidence import (
        ColorPathEvidenceModel,
        estimate_color_path_evidence,
        normalise_color_path_evidence_mode,
        relative_rgb_for_color_analysis,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.color_path_evidence import (
        ColorPathEvidenceModel,
        estimate_color_path_evidence,
        normalise_color_path_evidence_mode,
        relative_rgb_for_color_analysis,
    )

try:
    from darsia.fluidflower.tools.expert_constraint_support import (
        call_flash_and_mass_with_constraints,
        constraint_settings_from_env,
        constraint_weights_for_calibration,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.expert_constraint_support import (
        call_flash_and_mass_with_constraints,
        constraint_settings_from_env,
        constraint_weights_for_calibration,
    )

try:
    from darsia.fluidflower.tools.overflow_phase_constraint import (
        OverflowPhaseConstraint,
        resolve_overflow_phase_constraint,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.overflow_phase_constraint import (
        OverflowPhaseConstraint,
        resolve_overflow_phase_constraint,
    )

try:
    from darsia.fluidflower.tools.dual_phase_signal import (
        DualPhaseSignalModel,
        darsia_runtime_fingerprints,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.dual_phase_signal import (
        DualPhaseSignalModel,
        darsia_runtime_fingerprints,
    )

try:
    from darsia.fluidflower.tools.temporal_gas_support import (
        CausalGasGateSettings,
        build_causal_optical_gate_weights,
        hybrid_shared_signal_weight,
        late_gas_fraction_envelope,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.temporal_gas_support import (
        CausalGasGateSettings,
        build_causal_optical_gate_weights,
        hybrid_shared_signal_weight,
        late_gas_fraction_envelope,
    )

try:
    from darsia.fluidflower.tools.temporal_aqueous_support import (
        CausalGateSettings,
        apply_causal_aqueous_gate,
        apply_hard_seal_contact_constraint,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.temporal_aqueous_support import (
        CausalGateSettings,
        apply_causal_aqueous_gate,
        apply_hard_seal_contact_constraint,
    )

try:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec
except ModuleNotFoundError:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec

logger = logging.getLogger(__name__)

_CORRECTED_IMAGE_CACHE_SCHEMA = 1
_PREPARED_CONTEXT_CACHE_SCHEMA = 1


def _corrected_image_cache_path(
    source_path: Path,
    *,
    quality_dtype: Optional[str],
) -> tuple[Path, Dict[str, Any]] | None:
    root = os.environ.get("FFAC_CORRECTED_IMAGE_CACHE_ROOT", "").strip()
    namespace = os.environ.get("FFAC_CORRECTED_IMAGE_CACHE_NAMESPACE", "").strip()
    if not root or not namespace:
        return None
    source = Path(source_path)
    stat = source.stat()
    identity: Dict[str, Any] = {
        "schema": _CORRECTED_IMAGE_CACHE_SCHEMA,
        "namespace": namespace,
        "source": str(source.resolve()).casefold(),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "quality_dtype": str(quality_dtype or "float64"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return Path(root) / namespace[:16] / digest[:2] / f"{digest}.pickle", identity


def _load_corrected_image_cache(
    source_path: Path,
    *,
    quality_dtype: Optional[str],
) -> tuple[Any, bool, Dict[str, Any]] | None:
    try:
        resolved = _corrected_image_cache_path(
            source_path,
            quality_dtype=quality_dtype,
        )
    except OSError:
        return None
    if resolved is None:
        return None
    cache_path, identity = resolved
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("identity") != identity:
            return None
        image = payload["image"]
        diagnostics = dict(payload["diagnostics"])
        diagnostics["corrected_image_cache"] = "hit"
        return image, bool(payload["flagged"]), diagnostics
    except (OSError, EOFError, KeyError, pickle.PickleError, ValueError, TypeError):
        cache_path.unlink(missing_ok=True)
        return None


def _store_corrected_image_cache(
    source_path: Path,
    *,
    quality_dtype: Optional[str],
    image: Any,
    flagged: bool,
    diagnostics: Mapping[str, Any],
) -> Path | None:
    try:
        resolved = _corrected_image_cache_path(
            source_path,
            quality_dtype=quality_dtype,
        )
    except OSError:
        return None
    if resolved is None:
        return None
    cache_path, identity = resolved
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    payload = {
        "identity": identity,
        "image": image,
        "flagged": bool(flagged),
        "diagnostics": dict(diagnostics),
    }
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
        return cache_path
    finally:
        temporary.unlink(missing_ok=True)


def _prepared_context_cache_path(
    context: Any,
) -> tuple[Path, Dict[str, Any]] | None:
    frozen = getattr(context, "_prepared_context_cache_resolved", None)
    if isinstance(frozen, tuple) and len(frozen) == 2:
        return frozen
    root = os.environ.get("FFAC_PREPARED_CONTEXT_CACHE_ROOT", "").strip()
    namespace = os.environ.get(
        "FFAC_CORRECTED_IMAGE_CACHE_NAMESPACE", ""
    ).strip()
    if not root or not namespace:
        return None

    frame_sources: list[dict[str, Any]] = []
    for record in getattr(context, "frame_selection_records", []) or []:
        raw_path = record.get("selected") if isinstance(record, dict) else None
        if not raw_path:
            continue
        path = Path(raw_path)
        row: Dict[str, Any] = {"path": str(path.resolve()).casefold()}
        try:
            stat = path.stat()
            row.update(
                {
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        except OSError:
            # The image has already been loaded into the resident context.  A
            # transient NAS metadata failure must not invalidate that context,
            # but the path remains part of the cache identity.
            row["stat_unavailable"] = True
        frame_sources.append(row)

    config_path = getattr(context, "run_config_path", None)
    config_identity: Dict[str, Any] | None = None
    if config_path is not None:
        path = Path(config_path)
        config_identity = {"path": str(path.resolve()).casefold()}
        try:
            stat = path.stat()
            config_identity.update(
                {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
            )
        except OSError:
            config_identity["stat_unavailable"] = True

    calibration_artifacts: list[dict[str, Any]] = []
    calibration_folder = getattr(context, "calibration_folder", None)
    if calibration_folder is not None:
        folder = Path(calibration_folder)
        if folder.is_dir():
            for path in sorted(
                (candidate for candidate in folder.rglob("*") if candidate.is_file()),
                key=lambda candidate: str(candidate).casefold(),
            ):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                calibration_artifacts.append(
                    {
                        "path": str(path.relative_to(folder)).replace("\\", "/"),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )

    identity: Dict[str, Any] = {
        "schema": _PREPARED_CONTEXT_CACHE_SCHEMA,
        "namespace": namespace,
        "run": str(getattr(context, "run", "")).lower(),
        "quality_dtype": os.environ.get(
            "FFAC_COLORCHECKER_WORK_DTYPE", "float64"
        ),
        "phase_separation": str(
            getattr(context, "phase_separation", "shared-signal")
        ),
        "signal_labels": [
            int(value) for value in getattr(context, "signal_labels", []) or []
        ],
        "color_path_evidence_mode": str(
            getattr(context, "color_path_evidence_mode", "off")
        ),
        "aqueous_noise_gate": getattr(
            context, "aqueous_noise_gate_provenance", None
        ),
        "spatial_observation_support": getattr(
            context, "spatial_observation_support_provenance", None
        ),
        "color_path_evidence": getattr(
            context, "color_path_evidence_provenance", None
        ),
        "color_correction": getattr(
            context, "color_correction_provenance", None
        ),
        "run_config": config_identity,
        "calibration_artifacts": calibration_artifacts,
        "frame_sources": frame_sources,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    resolved = Path(root) / namespace[:16] / f"{digest}.pickle", identity
    context._prepared_context_cache_resolved = resolved
    return resolved


def _load_prepared_context_cache(context: Any) -> Path | None:
    try:
        resolved = _prepared_context_cache_path(context)
    except (OSError, ValueError, TypeError):
        return None
    if resolved is None:
        return None
    cache_path, identity = resolved
    if not cache_path.is_file():
        return None
    expected_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    try:
        with cache_path.open("rb") as stream:
            payload = pickle.load(stream)
        if (
            not isinstance(payload, dict)
            or payload.get("identity_sha256") != expected_digest
        ):
            return None
        prepared = payload.get("prepared_colors")
        if not isinstance(prepared, list) or not prepared:
            return None
        context._prepared_colors = prepared
        context._spatial_observation_confidences = list(
            payload.get("spatial_observation_confidences", []) or []
        )
        context._color_path_gate_confidences = list(
            payload.get("color_path_gate_confidences", []) or []
        )
        for attribute in (
            "spatial_observation_support_provenance",
            "color_path_evidence_provenance",
            "aqueous_causal_gate_provenance",
        ):
            if attribute in payload:
                setattr(context, attribute, payload[attribute])
        return cache_path
    except (
        OSError,
        EOFError,
        KeyError,
        pickle.PickleError,
        ValueError,
        TypeError,
    ):
        cache_path.unlink(missing_ok=True)
        return None


def _store_prepared_context_cache(context: Any) -> Path | None:
    try:
        resolved = _prepared_context_cache_path(context)
    except (OSError, ValueError, TypeError):
        return None
    if resolved is None:
        return None
    cache_path, identity = resolved
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    payload = {
        "identity": identity,
        "identity_sha256": identity_digest,
        "prepared_colors": list(context._prepared_colors),
        "spatial_observation_confidences": list(
            getattr(context, "_spatial_observation_confidences", []) or []
        ),
        "color_path_gate_confidences": list(
            getattr(context, "_color_path_gate_confidences", []) or []
        ),
        "spatial_observation_support_provenance": getattr(
            context, "spatial_observation_support_provenance", None
        ),
        "color_path_evidence_provenance": getattr(
            context, "color_path_evidence_provenance", None
        ),
        "aqueous_causal_gate_provenance": getattr(
            context, "aqueous_causal_gate_provenance", None
        ),
    }
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
        return cache_path
    finally:
        temporary.unlink(missing_ok=True)

PENALTY_VALUE = 1e12
DEFAULT_LATE_GAS_START_H = 24.0
DEFAULT_LATE_GAS_MAX_FRACTION = 0.05
DEFAULT_LATE_GAS_PENALTY_WEIGHT = 0.0
DEFAULT_LATE_GAS_DECAY_HALF_LIFE_H = 0.0
DEFAULT_LATE_GAS_FLOOR_FRACTION = 0.0
DEFAULT_LATE_GAS_CONSTRAINT_MODE = "phase-closure"
DEFAULT_OVERFLOW_POST_COVERAGE_PENALTY_WEIGHT = 0.0
DEFAULT_OVERFLOW_POST_COVERAGE_SIGMA_MULTIPLIER = 2.0
DEFAULT_OVERFLOW_POST_COVERAGE_FRACTION_TOLERANCE = 0.02
DEFAULT_AQUEOUS_DIFFUSION_PENALTY_WEIGHT = 0.0
DEFAULT_AQUEOUS_DIFFUSION_START_H = 4.0
DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL = 0.03
DEFAULT_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H = 1.0
DEFAULT_POST_INJECTION_BALANCE_WEIGHT = 0.0
DEFAULT_POST_INJECTION_BALANCE_START_H = 4.083
DEFAULT_SPATIAL_OBJECTIVE_WEIGHT = 0.0
DEFAULT_SPATIAL_RECALL_WEIGHT = 1.0
DEFAULT_SPATIAL_SEAL_WEIGHT = 2.0
DEFAULT_SPATIAL_TRACE_LEVEL = 0.03
DEFAULT_SPATIAL_CORE_LEVEL = 0.05
DEFAULT_LAYER_GAIN_REGULARIZATION_WEIGHT = 0.0
DEFAULT_LAYER_GAIN_REGULARIZATION_SIGMA = 0.20
_RESTORATION_SPILL_LABEL = -(2**31)
DEFAULT_DEPTH_EDGE_BOUNDS_M = (0.0100, 0.0115)
DEFAULT_DEPTH_BULGE_BOUNDS_M = (0.0035, 0.0055)
DEFAULT_DEPTH_MIN_M = 0.0090
DEFAULT_DEPTH_MAX_M = 0.0180
DEFAULT_DEPTH_PRIOR_WEIGHT = 0.0
DEFAULT_DEPTH_EDGE_PRIOR_SIGMA_M = 0.0005
DEFAULT_DEPTH_BULGE_PRIOR_SIGMA_M = 0.0010
DEFAULT_HYBRID_HALF_LIFE_H = 2.66
DEFAULT_HYBRID_HALF_LIFE_BOUNDS_H = (1.5, 4.0)

# --- ff_ac signal model: 4 support points (value0 locked at 0, value1..3 free);
#     flash locked at ff_ac defaults (0, 0.75, 0.75, 1.0). Per-label expansion in
#     build_param_space replicates this across the active (non-ignored) labels. ---
PARAM_SPACE_TEMPLATE: List[Dict[str, Any]] = [
    {"name": "signal.label0.value0",
     "attr_path": ["signal_model", "model", 1, 0, "values", 0],
     "bounds": (0.0, 0.0), "type": "float"},
    *[{"name": f"signal.label0.value{i}",
       "attr_path": ["signal_model", "model", 1, 0, "values", i],
       "bounds": (0.0, 2.0), "type": "float"} for i in range(1, 4)],  # signal node bound (validated)
    {"name": "flash.min_value_aq", "attr_path": ["flash", "min_value_aq"],
     "bounds": (0.0, 0.0), "type": "float"},
    {"name": "flash.max_value_aq", "attr_path": ["flash", "max_value_aq"],
     "bounds": (0.75, 0.75), "type": "float"},
    {"name": "flash.min_value_g", "attr_path": ["flash", "min_value_g"],
     "bounds": (0.75, 0.75), "type": "float"},
    {"name": "flash.max_value_g", "attr_path": ["flash", "max_value_g"],
     "bounds": (1.0, 1.0), "type": "float"},  # ff_um value, LOCKED (static flash)
]

_SIGNAL_PARAM_RE = re.compile(r"^signal\.label(?P<label>-?\d+)\.value(?P<idx>\d+)$")
_SHARED_SIGNAL_PARAM_RE = re.compile(r"^signal\.shared\.value(?P<idx>\d+)$")
_FACIES_SIGNAL_PARAM_RE = re.compile(
    r"^signal\.facies(?P<facies>\d+)\.value(?P<idx>\d+)$"
)
_SIGNAL_GAIN_RE = re.compile(r"^signal\.label(?P<label>-?\d+)\.gain$")
_VALUE_RE = re.compile(r"(?:.*\.)?value(\d+)$", re.IGNORECASE)
_SHARED_SIGNAL_LABEL = -1
_FACIES_SIGNAL_LABEL_OFFSET = 1000
_RESIDUAL_GAS_SCORE_CLIP = 2.0


def depth_map_identity(geometry: Any) -> Dict[str, Any]:
    """Return a stable identity for the depth field used by the geometry."""

    depth = getattr(geometry, "depth", None)
    values = getattr(depth, "img", depth)
    array = np.asarray(values)
    if array.ndim == 0:
        finite = np.asarray([float(array)], dtype=np.float64)
        digest_array = finite
    else:
        finite = np.asarray(array[np.isfinite(array)], dtype=np.float64)
        digest_array = np.ascontiguousarray(array)
    if finite.size == 0:
        raise RuntimeError("Geometry depth contains no finite values.")
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(digest_array.tobytes()).hexdigest(),
        "minimum_m": float(np.min(finite)),
        "maximum_m": float(np.max(finite)),
        "mean_m": float(np.mean(finite)),
        "std_m": float(np.std(finite)),
        "varying": bool(array.ndim > 0 and np.ptp(finite) > 1e-6),
    }


def validate_depth_map(geometry: Any, run: str) -> Dict[str, Any]:
    """Validate and cache the depth identity requested by the campaign."""

    identity = depth_map_identity(geometry)
    require_varying = os.environ.get(
        "FFAC_REQUIRE_VARYING_DEPTH", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    expected_sha = os.environ.get(
        "FFAC_EXPECTED_DEPTH_SHA256", ""
    ).strip().lower()
    if require_varying and not identity["varying"]:
        raise RuntimeError(
            f"[{run}] varying depth map required, but geometry depth is constant: "
            f"{identity}"
        )
    if expected_sha and identity["sha256"].lower() != expected_sha:
        raise RuntimeError(
            f"[{run}] depth map SHA mismatch: {identity['sha256']} != "
            f"{expected_sha}"
        )
    setattr(geometry, "_ffac_depth_identity", identity)
    logger.info(
        "[%s] depth map shape=%s range=%.6f..%.6f m std=%.6f m sha256=%s",
        run,
        identity["shape"],
        identity["minimum_m"],
        identity["maximum_m"],
        identity["std_m"],
        identity["sha256"],
    )
    return identity


def normalise_depth_parameterization(mode: str | None) -> str:
    """Resolve the optional depth nuisance model used during calibration."""

    value = str(
        mode
        if mode is not None
        else os.environ.get("FFAC_DEPTH_PARAMETERIZATION", "fixed")
    ).strip().lower().replace("_", "-")
    aliases = {
        "": "fixed",
        "off": "fixed",
        "none": "fixed",
        "measured": "fixed",
        "map": "fixed",
        "on": "edge-bulge",
        "dome": "edge-bulge",
        "edge": "edge-bulge",
        "edge+bulge": "edge-bulge",
    }
    value = aliases.get(value, value)
    if value not in {"fixed", "edge-bulge"}:
        raise ValueError(
            "FFAC_DEPTH_PARAMETERIZATION must be fixed or edge-bulge; "
            f"got {mode!r}"
        )
    return value


def _depth_field(geometry: Any) -> np.ndarray:
    depth = getattr(geometry, "depth", None)
    values = getattr(depth, "img", depth)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return array.reshape(1, 1)
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2-D depth field, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise RuntimeError("Depth parameterization requires a finite depth field")
    if np.any(array <= 0.0):
        raise RuntimeError("Depth parameterization requires strictly positive depth")
    return array


def build_depth_uncertainty_model(
    geometry: Any,
    *,
    minimum_m: float | None = None,
    maximum_m: float | None = None,
) -> Dict[str, Any]:
    """Represent the measured map as an edge level plus a fixed dome shape.

    At the reference values, ``edge + bulge * shape`` reproduces the measured
    map exactly. Optuna may vary only the edge level and bulge amplitude; it
    cannot invent a new spatial pattern. A five-percent border strip defines
    the edge reference independently of the interior maximum.
    """

    measured = _depth_field(geometry)
    height, width = measured.shape
    strip = max(1, int(round(min(height, width) * 0.05)))
    border = np.concatenate(
        (
            measured[:strip, :].ravel(),
            measured[-strip:, :].ravel(),
            measured[:, :strip].ravel(),
            measured[:, -strip:].ravel(),
        )
    )
    edge_reference_m = float(np.median(border))
    peak_reference_m = float(np.max(measured))
    bulge_reference_m = peak_reference_m - edge_reference_m
    if bulge_reference_m <= 1e-6:
        raise RuntimeError(
            "Edge-bulge depth parameterization requires a varying depth map"
        )
    shape = (measured - edge_reference_m) / bulge_reference_m
    minimum_m = float(
        minimum_m
        if minimum_m is not None
        else os.environ.get("FFAC_DEPTH_MIN_M", DEFAULT_DEPTH_MIN_M)
    )
    maximum_m = float(
        maximum_m
        if maximum_m is not None
        else os.environ.get("FFAC_DEPTH_MAX_M", DEFAULT_DEPTH_MAX_M)
    )
    if not (0.0 < minimum_m < maximum_m):
        raise ValueError(
            f"Invalid physical depth limits {minimum_m}..{maximum_m} m"
        )
    return {
        "mode": "edge-bulge",
        "measured": measured,
        "shape": np.asarray(shape, dtype=np.float32),
        "edge_reference_m": edge_reference_m,
        "bulge_reference_m": bulge_reference_m,
        "minimum_m": minimum_m,
        "maximum_m": maximum_m,
        "border_fraction": 0.05,
        "shape_minimum": float(np.min(shape)),
        "shape_maximum": float(np.max(shape)),
    }


def depth_model_provenance(model: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not model:
        return {"mode": "fixed"}
    return {
        str(key): value
        for key, value in model.items()
        if key not in {"measured", "shape"}
    }


# =========================================================================
# Dataclasses (mirror the queue's expected shapes)
# =========================================================================
@dataclass
class Metrics:
    injected_full: float
    total_full: float
    gaseous_full: Optional[float] = None
    aqueous_full: Optional[float] = None
    aqueous_concentration_max: Optional[float] = None
    aqueous_concentration_peak_equivalent: Optional[float] = None
    aqueous_affected_pore_volume_m3: Optional[float] = None
    aqueous_effective_pore_volume_m3: Optional[float] = None
    aqueous_low_observation_mass_kg: Optional[float] = None
    co2_low_observation_mass_kg: Optional[float] = None
    aqueous_observation_recall_deficit: Optional[float] = None
    aqueous_unsupported_seal_mass_kg: Optional[float] = None
    co2_unsupported_seal_mass_kg: Optional[float] = None
    aqueous_low_observation_support_volume_m3: Optional[float] = None
    aqueous_predicted_support_volume_m3: Optional[float] = None


class IntegratedFrameResult(NamedTuple):
    """Integrated phase masses plus local aqueous-distribution moments."""

    total_mass: float
    gaseous_mass: float
    aqueous_mass: float
    aqueous_concentration_max: Optional[float] = None
    aqueous_concentration_peak_equivalent: Optional[float] = None
    aqueous_affected_pore_volume_m3: Optional[float] = None
    aqueous_effective_pore_volume_m3: Optional[float] = None
    aqueous_low_observation_mass_kg: Optional[float] = None
    co2_low_observation_mass_kg: Optional[float] = None
    aqueous_observation_recall_deficit: Optional[float] = None
    aqueous_unsupported_seal_mass_kg: Optional[float] = None
    co2_unsupported_seal_mass_kg: Optional[float] = None
    aqueous_low_observation_support_volume_m3: Optional[float] = None
    aqueous_predicted_support_volume_m3: Optional[float] = None


@dataclass
class EvalResult:
    objective: float
    feasible: bool
    metrics: Dict[str, Metrics]
    status: str
    params: Dict[str, Any]
    objective_terms: Dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationContext:
    run: str
    config: Any
    experiment: Any
    fluidflower: Any
    geometry: Any
    calibration: Any                       # HeterogeneousColorToMassAnalysis
    calibration_images: List[Path]
    reference_label: int
    signal_label: Optional[int]
    signal_labels: List[int]
    param_space: List[Dict[str, Any]] = field(default_factory=list)
    enforce_lower: bool = False
    per_label_params: bool = True
    use_facies: bool = True
    signal_parameterization: str = "per-label"
    signal_label_to_facies: Dict[int, int] = field(default_factory=dict)
    objective_integral: str = "off"
    late_gas_start_h: float = DEFAULT_LATE_GAS_START_H
    late_gas_max_fraction: float = DEFAULT_LATE_GAS_MAX_FRACTION
    late_gas_penalty_weight: float = DEFAULT_LATE_GAS_PENALTY_WEIGHT
    late_gas_decay_half_life_h: float = DEFAULT_LATE_GAS_DECAY_HALF_LIFE_H
    late_gas_floor_fraction: float = DEFAULT_LATE_GAS_FLOOR_FRACTION
    late_gas_constraint_mode: str = DEFAULT_LATE_GAS_CONSTRAINT_MODE
    overflow_phase_constraint: Optional[OverflowPhaseConstraint] = None
    overflow_phase_penalty_weight: float = 0.0
    overflow_phase_provenance: Optional[Dict[str, Any]] = None
    overflow_post_coverage_penalty_weight: float = (
        DEFAULT_OVERFLOW_POST_COVERAGE_PENALTY_WEIGHT
    )
    overflow_post_coverage_sigma_multiplier: float = (
        DEFAULT_OVERFLOW_POST_COVERAGE_SIGMA_MULTIPLIER
    )
    overflow_post_coverage_fraction_tolerance: float = (
        DEFAULT_OVERFLOW_POST_COVERAGE_FRACTION_TOLERANCE
    )
    aqueous_diffusion_penalty_weight: float = (
        DEFAULT_AQUEOUS_DIFFUSION_PENALTY_WEIGHT
    )
    aqueous_diffusion_start_h: float = DEFAULT_AQUEOUS_DIFFUSION_START_H
    aqueous_diffusion_support_level: float = (
        DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL
    )
    aqueous_diffusion_pair_min_dt_h: float = (
        DEFAULT_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H
    )
    post_injection_balance_weight: float = DEFAULT_POST_INJECTION_BALANCE_WEIGHT
    post_injection_balance_start_h: float = DEFAULT_POST_INJECTION_BALANCE_START_H
    depth_parameterization: str = "fixed"
    depth_model_provenance: Optional[Dict[str, Any]] = None
    depth_prior_weight: float = DEFAULT_DEPTH_PRIOR_WEIGHT
    depth_edge_prior_sigma_m: float = DEFAULT_DEPTH_EDGE_PRIOR_SIGMA_M
    depth_bulge_prior_sigma_m: float = DEFAULT_DEPTH_BULGE_PRIOR_SIGMA_M
    _depth_model: Optional[Dict[str, Any]] = None
    phase_separation: str = "shared-signal"
    label_weights: Optional[Dict[int, float]] = None
    calibration_folder: Optional[Path] = None
    # preloaded (corrected image, injected_mass) pairs - read once, reused per trial
    _loaded: List[Tuple[Any, float, float]] = field(default_factory=list)  # (img, injected, t_hours)
    # Quantized, parameter-independent optical residual used by residual-gas.
    _gas_scores: List[np.ndarray] = field(default_factory=list)
    _gas_score_scale: float = _RESIDUAL_GAS_SCORE_CLIP
    gas_score_provenance: Optional[Dict[str, Any]] = None
    hybrid_injection_end_h: float = 0.0
    _hybrid_gas_weights: List[np.ndarray] = field(default_factory=list)
    _hybrid_nonseal_mask: Optional[np.ndarray] = None
    hybrid_gas_provenance: Optional[Dict[str, Any]] = None
    # Fixed observation model estimated from corrected no-CO2 reference frames.
    # It gates the raw, parameter-independent color-path coordinate and is never
    # exposed to Optuna.
    aqueous_noise_gate: Optional[AqueousNoiseGate] = None
    aqueous_noise_gate_provenance: Optional[Dict[str, Any]] = None
    aqueous_causal_gate_provenance: Optional[Dict[str, Any]] = None
    run_config_path: Optional[Path] = None
    # Optional hard phase invariant. A value of one permits free gas; zero
    # identifies sealing-layer pixels where Sg must remain exactly zero.
    _hard_nonseal_gas_mask: Optional[np.ndarray] = None
    hard_seal_gas_provenance: Optional[Dict[str, Any]] = None
    # Color-path projection is independent of all Optuna parameters and can be
    # computed once per frame instead of once per trial.
    _prepared_colors: List[Any] = field(default_factory=list)
    _evaluation_backend: str = "legacy"
    _cuda_evaluator: Any = None
    _cuda_float32_evaluator: Any = None
    _opencl_evaluator: Any = None
    _opencl_float32_evaluator: Any = None
    _metal_evaluator: Any = None
    # Provenance for preprocessing choices that alter the objective surface.
    frame_selection_policy: str = "nearest-good"
    frame_selection_records: List[Dict[str, Any]] = field(default_factory=list)
    color_correction_provenance: Optional[Dict[str, Any]] = None
    evaluation_time_records: List[Dict[str, Any]] = field(default_factory=list)
    color_path_source: Optional[str] = None
    couple_aq_gas: bool = False
    # Independent, pixel-centred support from no-CO2 reference images. The
    # confidence is quantized once and reused by every trial.
    spatial_observation_support: Optional[SpatialAqueousSupportModel] = None
    spatial_observation_support_provenance: Optional[Dict[str, Any]] = None
    _spatial_observation_confidences: List[np.ndarray] = field(default_factory=list)
    color_path_evidence_mode: str = "off"
    color_path_evidence: Optional[ColorPathEvidenceModel] = None
    color_path_evidence_provenance: Optional[Dict[str, Any]] = None
    _color_path_gate_confidences: List[np.ndarray] = field(default_factory=list)
    spatial_objective_weight: float = DEFAULT_SPATIAL_OBJECTIVE_WEIGHT
    spatial_recall_weight: float = DEFAULT_SPATIAL_RECALL_WEIGHT
    spatial_seal_weight: float = DEFAULT_SPATIAL_SEAL_WEIGHT
    spatial_trace_level: float = DEFAULT_SPATIAL_TRACE_LEVEL
    spatial_core_level: float = DEFAULT_SPATIAL_CORE_LEVEL
    layer_gain_regularization_weight: float = (
        DEFAULT_LAYER_GAIN_REGULARIZATION_WEIGHT
    )
    layer_gain_regularization_sigma: float = (
        DEFAULT_LAYER_GAIN_REGULARIZATION_SIGMA
    )


# =========================================================================
# Param-name helpers (queue contract)
# =========================================================================
def _parse_signal_name(name: str) -> Optional[Tuple[int, int]]:
    m = _SIGNAL_PARAM_RE.match(name)
    if m:
        return int(m.group("label")), int(m.group("idx"))
    shared = _SHARED_SIGNAL_PARAM_RE.match(name)
    if shared:
        return _SHARED_SIGNAL_LABEL, int(shared.group("idx"))
    facies = _FACIES_SIGNAL_PARAM_RE.match(name)
    if facies:
        group = int(facies.group("facies"))
        return -(_FACIES_SIGNAL_LABEL_OFFSET + group), int(facies.group("idx"))
    return None


def _facies_from_signal_key(key: int) -> Optional[int]:
    if key <= -_FACIES_SIGNAL_LABEL_OFFSET:
        return -int(key) - _FACIES_SIGNAL_LABEL_OFFSET
    return None


def _normalise_signal_parameterization(mode: str | None) -> str:
    value = (
        mode
        or os.environ.get("FFAC_SIGNAL_PARAMETERIZATION")
        or "per-label"
    ).strip().lower().replace("_", "-")
    aliases = {
        "default": "per-label",
        "perlabel": "per-label",
        "shared": "shared-shape",
        "reduced": "shared-shape",
        "shared-gain": "shared-shape",
        "shared-shape-gain": "shared-shape",
        "hierarchical": "facies-shape-layer-gain",
        "layer-hierarchical": "facies-shape-layer-gain",
        "facies-layer": "facies-shape-layer-gain",
    }
    value = aliases.get(value, value)
    if value not in {
        "per-label",
        "shared-shape",
        "facies-shape-layer-gain",
    }:
        raise ValueError(
            f"Unknown signal parameterization {mode!r}; expected per-label, "
            "shared-shape, or facies-shape-layer-gain."
        )
    return value


def resolve_late_gas_constraint(
    start_h: float | str | None = None,
    max_fraction: float | str | None = None,
    penalty_weight: float | str | None = None,
) -> tuple[float, float, float]:
    """Resolve and validate the optional late free-gas objective settings."""

    raw_start = (
        start_h
        if start_h is not None
        else os.environ.get(
            "FFAC_LATE_GAS_START_H",
            str(DEFAULT_LATE_GAS_START_H),
        )
    )
    raw_fraction = (
        max_fraction
        if max_fraction is not None
        else os.environ.get(
            "FFAC_LATE_GAS_MAX_FRACTION",
            str(DEFAULT_LATE_GAS_MAX_FRACTION),
        )
    )
    raw_weight = (
        penalty_weight
        if penalty_weight is not None
        else os.environ.get(
            "FFAC_LATE_GAS_PENALTY_WEIGHT",
            str(DEFAULT_LATE_GAS_PENALTY_WEIGHT),
        )
    )
    try:
        resolved = (float(raw_start), float(raw_fraction), float(raw_weight))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Late-gas settings must be numeric: "
            f"start_h={raw_start!r}, max_fraction={raw_fraction!r}, "
            f"penalty_weight={raw_weight!r}"
        ) from exc
    resolved_start, resolved_fraction, resolved_weight = resolved
    if not all(np.isfinite(value) for value in resolved):
        raise ValueError(f"Late-gas settings must be finite, got {resolved}")
    if resolved_start < 0.0:
        raise ValueError("Late-gas start time must be non-negative")
    if not 0.0 <= resolved_fraction <= 1.0:
        raise ValueError("Late-gas maximum fraction must be within [0, 1]")
    if resolved_weight < 0.0:
        raise ValueError("Late-gas penalty weight must be non-negative")
    return resolved_start, resolved_fraction, resolved_weight


def resolve_late_gas_decay(
    half_life_h: float | str | None = None,
    floor_fraction: float | str | None = None,
) -> tuple[float, float]:
    """Resolve the optional smooth decay of the late-gas upper envelope."""

    raw_half_life = (
        half_life_h
        if half_life_h is not None
        else os.environ.get(
            "FFAC_LATE_GAS_DECAY_HALF_LIFE_H",
            str(DEFAULT_LATE_GAS_DECAY_HALF_LIFE_H),
        )
    )
    raw_floor = (
        floor_fraction
        if floor_fraction is not None
        else os.environ.get(
            "FFAC_LATE_GAS_FLOOR_FRACTION",
            str(DEFAULT_LATE_GAS_FLOOR_FRACTION),
        )
    )
    try:
        resolved = (float(raw_half_life), float(raw_floor))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Late-gas decay settings must be numeric: "
            f"half_life_h={raw_half_life!r}, floor_fraction={raw_floor!r}"
        ) from exc
    half_life, floor = resolved
    if not all(np.isfinite(value) for value in resolved):
        raise ValueError(f"Late-gas decay settings must be finite, got {resolved}")
    if half_life < 0.0:
        raise ValueError("Late-gas decay half-life cannot be negative")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("Late-gas floor fraction must be within [0, 1]")
    return half_life, floor


def resolve_late_gas_constraint_mode(mode: str | None = None) -> str:
    """Choose whether the late-gas cap also enforces aqueous mass closure."""

    value = (
        mode
        if mode is not None
        else os.environ.get(
            "FFAC_LATE_GAS_CONSTRAINT_MODE",
            DEFAULT_LATE_GAS_CONSTRAINT_MODE,
        )
    )
    normalized = str(value or DEFAULT_LATE_GAS_CONSTRAINT_MODE).strip().lower()
    normalized = normalized.replace("_", "-")
    aliases = {
        "legacy": "phase-closure",
        "phase": "phase-closure",
        "closure": "phase-closure",
        "gas": "gas-only",
        "gasonly": "gas-only",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"phase-closure", "gas-only"}:
        raise ValueError(
            "Late-gas constraint mode must be 'phase-closure' or 'gas-only', "
            f"got {value!r}"
        )
    return normalized


def resolve_overflow_post_coverage_objective(
    penalty_weight: float | str | None = None,
    sigma_multiplier: float | str | None = None,
    fraction_tolerance: float | str | None = None,
    *,
    overflow_phase_penalty_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Resolve the soft run-specific gas support after overflow coverage."""

    raw_weight = (
        penalty_weight
        if penalty_weight is not None
        else os.environ.get(
            "FFAC_OVERFLOW_POST_COVERAGE_PENALTY_WEIGHT",
            str(DEFAULT_OVERFLOW_POST_COVERAGE_PENALTY_WEIGHT),
        )
    )
    if str(raw_weight).strip().lower() == "inherit":
        raw_weight = overflow_phase_penalty_weight
    raw_sigma = (
        sigma_multiplier
        if sigma_multiplier is not None
        else os.environ.get(
            "FFAC_OVERFLOW_POST_COVERAGE_SIGMA_MULTIPLIER",
            str(DEFAULT_OVERFLOW_POST_COVERAGE_SIGMA_MULTIPLIER),
        )
    )
    raw_fraction = (
        fraction_tolerance
        if fraction_tolerance is not None
        else os.environ.get(
            "FFAC_OVERFLOW_POST_COVERAGE_FRACTION_TOLERANCE",
            str(DEFAULT_OVERFLOW_POST_COVERAGE_FRACTION_TOLERANCE),
        )
    )
    try:
        resolved = (float(raw_weight), float(raw_sigma), float(raw_fraction))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Overflow post-coverage settings must be numeric (or weight="
            f"'inherit'): weight={raw_weight!r}, sigma={raw_sigma!r}, "
            f"fraction_tolerance={raw_fraction!r}"
        ) from exc
    weight, sigma, fraction = resolved
    if not all(np.isfinite(value) for value in resolved):
        raise ValueError(
            f"Overflow post-coverage settings must be finite, got {resolved}"
        )
    if weight < 0.0 or sigma < 0.0 or not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "Overflow post-coverage weight and sigma must be non-negative, "
            "and fraction tolerance must be within [0, 1]."
        )
    return weight, sigma, fraction


def resolve_aqueous_diffusion_objective(
    penalty_weight: float | str | None = None,
    start_h: float | str | None = None,
    support_level: float | str | None = None,
    pair_min_dt_h: float | str | None = None,
) -> tuple[float, float, float, float]:
    """Resolve the optional post-injection aqueous redistribution objective."""

    raw_weight = (
        penalty_weight
        if penalty_weight is not None
        else os.environ.get(
            "FFAC_AQUEOUS_DIFFUSION_PENALTY_WEIGHT",
            str(DEFAULT_AQUEOUS_DIFFUSION_PENALTY_WEIGHT),
        )
    )
    raw_start = (
        start_h
        if start_h is not None
        else os.environ.get(
            "FFAC_AQUEOUS_DIFFUSION_START_H",
            str(DEFAULT_AQUEOUS_DIFFUSION_START_H),
        )
    )
    raw_support = (
        support_level
        if support_level is not None
        else os.environ.get(
            "FFAC_AQUEOUS_DIFFUSION_SUPPORT_LEVEL",
            str(DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL),
        )
    )
    raw_pair_min_dt = (
        pair_min_dt_h
        if pair_min_dt_h is not None
        else os.environ.get(
            "FFAC_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H",
            str(DEFAULT_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H),
        )
    )
    try:
        resolved = (
            float(raw_weight),
            float(raw_start),
            float(raw_support),
            float(raw_pair_min_dt),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Aqueous-diffusion objective settings must be numeric: "
            f"weight={raw_weight!r}, start_h={raw_start!r}, "
            f"support_level={raw_support!r}, pair_min_dt_h={raw_pair_min_dt!r}"
        ) from exc
    weight, start, support, minimum_pair_dt = resolved
    if not all(np.isfinite(value) for value in resolved):
        raise ValueError(
            f"Aqueous-diffusion objective settings must be finite, got {resolved}"
        )
    if weight < 0.0 or start < 0.0 or minimum_pair_dt < 0.0:
        raise ValueError(
            "Aqueous-diffusion weight, start time, and pair spacing must be "
            "non-negative"
        )
    if not 0.0 < support < 1.0:
        raise ValueError(
            "Aqueous-diffusion support level must be strictly between 0 and 1"
        )
    return weight, start, support, minimum_pair_dt


def resolve_extra_calibration_times(
    times_hours: Sequence[float] | str | None = None,
    tolerance_seconds: float | str | None = None,
) -> tuple[tuple[float, ...], float]:
    """Resolve optional campaign-local additions to the calibration schedule."""

    raw_times: Sequence[float] | str = (
        times_hours
        if times_hours is not None
        else os.environ.get("FFAC_EXTRA_CALIBRATION_TIMES_H", "")
    )
    if isinstance(raw_times, str):
        tokens = [token.strip() for token in re.split(r"[,;\s]+", raw_times)]
        tokens = [token for token in tokens if token]
        try:
            resolved_times = tuple(float(token) for token in tokens)
        except ValueError as exc:
            raise ValueError(
                "FFAC_EXTRA_CALIBRATION_TIMES_H must contain numeric hours, got "
                f"{raw_times!r}"
            ) from exc
    else:
        try:
            resolved_times = tuple(float(value) for value in raw_times)
        except (TypeError, ValueError) as exc:
            raise ValueError("Extra calibration times must be numeric") from exc

    raw_tolerance = (
        tolerance_seconds
        if tolerance_seconds is not None
        else os.environ.get(
            "FFAC_EXTRA_CALIBRATION_TIME_TOLERANCE_SECONDS",
            "900",
        )
    )
    try:
        resolved_tolerance = float(raw_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Extra calibration-time tolerance must be numeric, got "
            f"{raw_tolerance!r}"
        ) from exc
    if not all(np.isfinite(value) and value >= 0.0 for value in resolved_times):
        raise ValueError("Extra calibration times must be finite and non-negative")
    if not np.isfinite(resolved_tolerance) or resolved_tolerance < 0.0:
        raise ValueError(
            "Extra calibration-time tolerance must be finite and non-negative"
        )
    return tuple(dict.fromkeys(resolved_times)), resolved_tolerance


def resolve_depth_prior(
    weight: float | str | None = None,
    edge_sigma_m: float | str | None = None,
    bulge_sigma_m: float | str | None = None,
) -> tuple[float, float, float]:
    """Resolve a weak Gaussian prior around the measured depth-map geometry."""

    raw_weight = (
        weight
        if weight is not None
        else os.environ.get("FFAC_DEPTH_PRIOR_WEIGHT", str(DEFAULT_DEPTH_PRIOR_WEIGHT))
    )
    raw_edge_sigma = (
        edge_sigma_m
        if edge_sigma_m is not None
        else os.environ.get(
            "FFAC_DEPTH_EDGE_PRIOR_SIGMA_M",
            str(DEFAULT_DEPTH_EDGE_PRIOR_SIGMA_M),
        )
    )
    raw_bulge_sigma = (
        bulge_sigma_m
        if bulge_sigma_m is not None
        else os.environ.get(
            "FFAC_DEPTH_BULGE_PRIOR_SIGMA_M",
            str(DEFAULT_DEPTH_BULGE_PRIOR_SIGMA_M),
        )
    )
    try:
        resolved = (
            float(raw_weight),
            float(raw_edge_sigma),
            float(raw_bulge_sigma),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Depth-prior settings must be numeric") from exc
    if not all(np.isfinite(value) for value in resolved):
        raise ValueError(f"Depth-prior settings must be finite, got {resolved}")
    prior_weight, edge_sigma, bulge_sigma = resolved
    if prior_weight < 0.0:
        raise ValueError("Depth-prior weight cannot be negative")
    if edge_sigma <= 0.0 or bulge_sigma <= 0.0:
        raise ValueError("Depth-prior standard deviations must be positive")
    return prior_weight, edge_sigma, bulge_sigma


def _normalise_phase_separation(mode: str | None) -> str:
    value = (
        mode
        or os.environ.get("FFAC_PHASE_SEPARATION")
        or "shared-signal"
    ).strip().lower().replace("_", "-")
    aliases = {
        "off": "shared-signal",
        "none": "shared-signal",
        "default": "shared-signal",
        "shared": "shared-signal",
        "residual": "residual-gas",
        "optical-residual": "residual-gas",
        "separate-gas": "residual-gas",
        "clear": "clearwater-direct",
        "clearwater": "clearwater-direct",
        "wash": "clearwater-direct",
        "wash-direct": "clearwater-direct",
        "clear-gated": "clearwater-gated",
        "wash-gated": "clearwater-gated",
        "hybrid": "hybrid-causal",
        "causal-hybrid": "hybrid-causal",
        "hybrid-clearwater": "hybrid-causal",
        "dual": "supervised-dual",
        "dual-signal": "supervised-dual",
        "separate-signals": "supervised-dual",
        "supervised-gas": "supervised-dual",
    }
    value = aliases.get(value, value)
    if value not in {
        "shared-signal",
        "residual-gas",
        "clearwater-direct",
        "clearwater-gated",
        "hybrid-causal",
        "supervised-dual",
    }:
        raise ValueError(
            f"Unknown phase separation {mode!r}; expected shared-signal, "
            "residual-gas, clearwater-direct, clearwater-gated, "
            "hybrid-causal, or supervised-dual."
        )
    return value


def _value_entries_by_label(param_space: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    entries_by_label: Dict[int, List[Dict[str, Any]]] = {}
    for entry in param_space:
        parsed = _parse_signal_name(entry.get("name", ""))
        if not parsed:
            continue
        label, idx = parsed
        entries_by_label.setdefault(label, []).append(entry)
    for label, entries in entries_by_label.items():
        entries_by_label[label] = sorted(entries, key=lambda e: _parse_signal_name(e["name"])[1])
    return entries_by_label


def _monotonic_bounds(entries: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    lows = [float(e["bounds"][0]) for e in entries]
    highs = [float(e["bounds"][1]) for e in entries]
    lower: List[float] = []
    for i, low in enumerate(lows):
        lower.append(low if i == 0 else max(low, lower[i - 1]))
    upper: List[float] = [0.0] * len(entries)
    if entries:
        upper[-1] = highs[-1]
        for i in range(len(entries) - 2, -1, -1):
            upper[i] = min(highs[i], upper[i + 1])
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo > hi:
            raise ValueError("Infeasible monotonic bounds for {}: {} > {}".format(entries[i]["name"], lo, hi))
    return lower, upper


def _project_monotonic_values(values: Sequence[float], entries: Sequence[Dict[str, Any]]) -> List[float]:
    if not entries:
        return []
    if len(values) != len(entries):
        raise ValueError("Value count mismatch for monotonic projection.")
    lower, upper = _monotonic_bounds(entries)
    projected: List[float] = []
    prev = None
    for i, raw in enumerate(values):
        min_i = lower[i] if prev is None else max(lower[i], prev)
        max_i = upper[i]
        projected_val = min(max(float(raw), min_i), max_i)
        projected.append(projected_val)
        prev = projected_val
    return projected


def _parse_bool(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# =========================================================================
# Param space
# =========================================================================
def _match_bounds(name, override, default_override):
    if name in override:
        return override[name]
    if name in default_override:
        return default_override[name]
    # wildcard: signal.labelN.valueI is also matched by signal.label*.valueI
    # (this is the key form --param-ranges writes, applied across all labels).
    wild = re.sub(r"\.label-?\d+\.", ".label*.", name)
    if wild != name:
        if wild in override:
            return override[wild]
        if wild in default_override:
            return default_override[wild]
    facies_wild = re.sub(r"\.facies\d+\.", ".facies*.", name)
    if facies_wild != name:
        if facies_wild in override:
            return override[facies_wild]
        if facies_wild in default_override:
            return default_override[facies_wild]
    return None


def _rebuild_template_signal(template, n_free_values):
    """Regenerate signal.* as value0 (locked) + value1..value{n_free_values}, preserving the
    template free-value bounds and all non-signal (flash) entries. Lets the param space match
    the signal-model resolution (num_segments+1 points): a 7-point model optimises value1..6."""
    sig = [e for e in template if _parse_signal_name(e["name"])]
    other = [e for e in template if not _parse_signal_name(e["name"])]
    free_bounds = (0.0, 4.0)
    for e in sig:
        p = _parse_signal_name(e["name"])
        if p and p[1] >= 1:
            free_bounds = tuple(e["bounds"]); break
    new_sig = [{"name": "signal.label0.value0",
                "attr_path": ["signal_model", "model", 1, 0, "values", 0],
                "bounds": (0.0, 0.0), "type": "float"}]
    for i in range(1, int(n_free_values) + 1):
        new_sig.append({"name": f"signal.label0.value{i}",
                        "attr_path": ["signal_model", "model", 1, 0, "values", i],
                        "bounds": free_bounds, "type": "float"})
    return new_sig + other


def _rebuild_shared_shape_signal(
    template: Sequence[Dict[str, Any]],
    n_free_values: int,
    signal_labels: Sequence[int],
) -> List[Dict[str, Any]]:
    """Build an identifiable shared curve shape with one amplitude per facies.

    The shared shape starts at zero and ends at one. Intermediate values are
    monotone fractions in [0, 1]. Multiplying by each facies amplitude therefore
    keeps every effective signal value in the established [0, 2] range without
    a free shape/amplitude scale degeneracy.
    """

    other = [e for e in template if not _parse_signal_name(e["name"])]
    count = max(1, int(n_free_values))
    shared: List[Dict[str, Any]] = []
    for idx in range(count + 1):
        if idx == 0:
            bounds = (0.0, 0.0)
        elif idx == count:
            bounds = (1.0, 1.0)
        else:
            bounds = (0.0, 1.0)
        shared.append(
            {
                "name": f"signal.shared.value{idx}",
                "attr_path": ["signal_model", "shared_shape", "values", idx],
                "bounds": bounds,
                "type": "float",
            }
        )

    gains = [
        {
            "name": f"signal.label{int(label)}.gain",
            "attr_path": ["signal_model", "model", 1, int(label), "gain"],
            "bounds": (0.0, 2.0),
            "type": "float",
        }
        for label in sorted({int(label) for label in signal_labels})
    ]
    return [*shared, *gains, *copy.deepcopy(other)]


def _rebuild_hierarchical_signal(
    template: Sequence[Dict[str, Any]],
    n_free_values: int,
    signal_labels: Sequence[int],
    label_to_facies: Mapping[int, int],
) -> List[Dict[str, Any]]:
    """Build one curve shape per material and one amplitude per physical layer.

    Detailed layer colour paths remain run-specific. Layers made from the same
    sand share a monotone response shape, while the layer gain can absorb a
    bounded local illumination/sensitivity difference. The zero response is
    structural, so a layer cannot learn a non-zero CO2 signal at baseline.
    """

    other = [e for e in template if not _parse_signal_name(e["name"])]
    count = max(1, int(n_free_values))
    labels = sorted({int(label) for label in signal_labels})
    missing = [label for label in labels if label not in label_to_facies]
    if missing:
        raise ValueError(
            "Missing facies membership for detailed signal label(s): "
            + ", ".join(str(label) for label in missing)
        )
    facies_groups = sorted({int(label_to_facies[label]) for label in labels})
    shapes: List[Dict[str, Any]] = []
    for facies in facies_groups:
        for idx in range(count + 1):
            if idx == 0:
                bounds = (0.0, 0.0)
            elif idx == count:
                bounds = (1.0, 1.0)
            else:
                bounds = (0.0, 1.0)
            shapes.append(
                {
                    "name": f"signal.facies{facies}.value{idx}",
                    "attr_path": [
                        "signal_model",
                        "facies_shape",
                        facies,
                        "values",
                        idx,
                    ],
                    "bounds": bounds,
                    "type": "float",
                }
            )
    gains = [
        {
            "name": f"signal.label{label}.gain",
            "attr_path": ["signal_model", "model", 1, label, "gain"],
            "bounds": (0.5, 2.0),
            "type": "float",
        }
        for label in labels
    ]
    return [*shapes, *gains, *copy.deepcopy(other)]


def build_param_space(run, bounds_map, signal_label=None, signal_labels=None,
                      per_label_params=False, use_facies=True, n_free_values=None,
                      signal_parameterization="per-label",
                      label_to_facies=None):
    parameterization = _normalise_signal_parameterization(signal_parameterization)
    base = copy.deepcopy(PARAM_SPACE_TEMPLATE)
    if n_free_values is not None:
        base = _rebuild_template_signal(base, int(n_free_values))
    override = (bounds_map or {}).get(run, {})
    default_override = (bounds_map or {}).get("default", {})
    space: List[Dict[str, Any]] = []
    if parameterization == "shared-shape":
        labels = list(signal_labels or ([signal_label] if signal_label is not None else []))
        free_values = int(n_free_values) if n_free_values is not None else max(
            (_parse_signal_name(e["name"])[1] for e in base if _parse_signal_name(e["name"])),
            default=1,
        )
        space = _rebuild_shared_shape_signal(base, free_values, labels)
    elif parameterization == "facies-shape-layer-gain":
        labels = list(signal_labels or ([signal_label] if signal_label is not None else []))
        free_values = int(n_free_values) if n_free_values is not None else max(
            (_parse_signal_name(e["name"])[1] for e in base if _parse_signal_name(e["name"])),
            default=1,
        )
        space = _rebuild_hierarchical_signal(
            base,
            free_values,
            labels,
            label_to_facies or {},
        )
    elif per_label_params:
        sig = [e for e in base if _parse_signal_name(e["name"])]
        other = [e for e in base if not _parse_signal_name(e["name"])]
        labels = list(signal_labels or ([signal_label] if signal_label is not None else []))
        for label in labels:
            for e in sig:
                ne = copy.deepcopy(e)
                ne["name"] = e["name"].replace("signal.label0", f"signal.label{label}")
                ne["attr_path"] = list(e["attr_path"]); ne["attr_path"][3] = int(label)
                space.append(ne)
        space.extend(copy.deepcopy(other))
    else:
        space = base
        if signal_label is not None:
            for e in space:
                if _parse_signal_name(e["name"]):
                    e["attr_path"] = list(e["attr_path"]); e["attr_path"][3] = int(signal_label)
    for e in space:
        b = _match_bounds(e["name"], override, default_override)
        if b is not None:
            e["bounds"] = tuple(b)
    return space


def configure_flash_param_space(
    param_space: Sequence[Dict[str, Any]],
    run: str,
    bounds_map: Optional[Mapping[str, Mapping[str, Sequence[float]]]],
    flash: Any,
) -> List[Dict[str, Any]]:
    """Add structurally coupled phase parameters requested by the campaign."""

    space = [copy.deepcopy(entry) for entry in param_space]
    override = (bounds_map or {}).get(run, {})
    default_override = (bounds_map or {}).get("default", {})
    if os.environ.get("FFAC_PARAMETERIZE_GAS_WIDTH", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        space = [
            entry
            for entry in space
            if entry["name"] not in {"flash.min_value_g", "flash.max_value_g"}
        ]
        gas_width = {
            "name": "flash.gas_width",
            "attr_path": ["flash", "gas_width"],
            "bounds": (0.40, 0.85),
            "type": "float",
        }
        bounds = _match_bounds(gas_width["name"], override, default_override)
        if bounds is not None:
            gas_width["bounds"] = tuple(bounds)
        space.append(gas_width)
    if (
        isinstance(flash, TitrationFlash)
        and os.environ.get(
            "FFAC_OPTIMIZE_AQUEOUS_BLEND", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
    ):
        aqueous_blend = {
            "name": "flash.aqueous_blend",
            "attr_path": ["flash", "aqueous_blend"],
            "bounds": (0.0, 0.40),
            "type": "float",
        }
        bounds = _match_bounds(
            aqueous_blend["name"], override, default_override
        )
        if bounds is not None:
            aqueous_blend["bounds"] = tuple(bounds)
        space.append(aqueous_blend)
    return space


def configure_depth_param_space(
    param_space: Sequence[Dict[str, Any]],
    run: str,
    bounds_map: Optional[Mapping[str, Mapping[str, Sequence[float]]]],
    geometry: Any,
    mode: str | None = None,
    minimum_m: float | None = None,
    maximum_m: float | None = None,
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Append the physically constrained edge/bulge depth nuisance parameters."""

    resolved = normalise_depth_parameterization(mode)
    space = [copy.deepcopy(entry) for entry in param_space]
    if resolved == "fixed":
        return space, None

    model = build_depth_uncertainty_model(
        geometry,
        minimum_m=minimum_m,
        maximum_m=maximum_m,
    )
    override = (bounds_map or {}).get(run, {})
    default_override = (bounds_map or {}).get("default", {})
    entries = (
        {
            "name": "depth.edge_m",
            "attr_path": ["geometry", "depth", "edge_m"],
            "bounds": DEFAULT_DEPTH_EDGE_BOUNDS_M,
            "type": "float",
        },
        {
            "name": "depth.bulge_m",
            "attr_path": ["geometry", "depth", "bulge_m"],
            "bounds": DEFAULT_DEPTH_BULGE_BOUNDS_M,
            "type": "float",
        },
    )
    for entry in entries:
        bounds = _match_bounds(entry["name"], override, default_override)
        if bounds is not None:
            entry["bounds"] = tuple(bounds)
        low, high = (float(value) for value in entry["bounds"])
        if not (0.0 < low <= high):
            raise ValueError(
                f"Invalid bounds for {entry['name']}: {entry['bounds']}"
            )
        space.append(entry)

    logger.info(
        "[%s] edge-bulge depth nuisance ACTIVE: measured edge=%.6fm "
        "bulge=%.6fm bounds(edge=%s, bulge=%s) physical=%.6f..%.6fm",
        run,
        model["edge_reference_m"],
        model["bulge_reference_m"],
        entries[0]["bounds"],
        entries[1]["bounds"],
        model["minimum_m"],
        model["maximum_m"],
    )
    return space, model


def sample_params(
    param_space: Sequence[Dict[str, Any]],
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    random_source = rng or random
    samples: Dict[str, Any] = {}
    for _, entries in _value_entries_by_label(param_space).items():
        lower, upper = _monotonic_bounds(entries)
        prev = None
        for i, entry in enumerate(entries):
            lo, hi = entry["bounds"]
            if lo == hi:
                val = float(lo)
            else:
                min_i = lower[i] if prev is None else max(lower[i], prev)
                max_i = upper[i]
                val = min_i if max_i <= min_i else (
                    min_i + random_source.random() * (max_i - min_i)
                )
            samples[entry["name"]] = val
            prev = val
    for entry in param_space:
        if entry["name"] in samples:
            continue
        low, high = entry["bounds"]
        samples[entry["name"]] = (int(round(random_source.uniform(low, high)))
                                  if entry.get("type", "float") == "int"
                                  else random_source.uniform(low, high))
    return samples


def suggest_params_trial(trial, param_space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Cumulative-fraction monotone suggestion. For each label, value_i is built
    as value_{i-1} + f_i * (upper_i - value_{i-1}) with f_i in [0,1] (fixed bounds).
    The mapping f -> values is BIJECTIVE and monotone by construction, so Optuna
    optimises the monotone structure directly (no permutation symmetry / hidden
    sort, no clipping mismatch). Non-signal params (e.g. flash) are suggested
    directly. The returned values are exactly what is evaluated, logged and applied."""
    params: Dict[str, Any] = {}
    for _, entries in _value_entries_by_label(param_space).items():
        lower, upper = _monotonic_bounds(entries)
        prev = None
        for i, entry in enumerate(entries):
            lo, hi = entry["bounds"]
            if lo == hi:
                val = float(lo)
            else:
                min_i = lower[i] if prev is None else max(lower[i], prev)
                max_i = upper[i]
                if max_i <= min_i:
                    val = min_i
                else:
                    f = trial.suggest_float("cum_" + entry["name"], 0.0, 1.0)
                    val = min_i + f * (max_i - min_i)
            params[entry["name"]] = val
            prev = val
    for entry in param_space:
        if entry["name"] in params:
            continue
        low, high = entry["bounds"]
        if entry.get("type", "float") == "int":
            params[entry["name"]] = trial.suggest_int(entry["name"], int(low), int(high))
        else:
            params[entry["name"]] = trial.suggest_float(entry["name"], float(low), float(high))
    return params


# weighting stubs (not used by default; kept for queue contract) --------------
def compute_auto_label_weights(context) -> Dict[int, float]:
    return {}


def apply_label_weight_grouping(weights, context, grouping):
    return weights


def load_bounds_map(path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalise_static_light_mode(mode: str | None) -> str:
    value = (mode or os.environ.get("FFAC_STATIC_LIGHT_CORRECTION") or "off").strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "on": "blue-gain",
        "blue": "blue-gain",
        "rgb": "blue-gain",
        "static-rgb": "blue-gain",
        "intensity": "intensity",
        "luma": "intensity",
        "static-intensity": "intensity",
        "spatial": "blue-spatial",
        "spatial-gain": "blue-spatial",
        "spatial-rgb": "blue-spatial",
        "static-spatial": "blue-spatial",
        "blue-spatial-gain": "blue-spatial",
        "spatial-blue-gain": "blue-spatial",
        "spatial-intensity": "intensity-spatial",
        "intensity-spatial-gain": "intensity-spatial",
        "plume": "plume-excluded",
        "plume-excluded-rgb": "plume-excluded",
        "porous-reference": "plume-excluded",
    }
    return aliases.get(value, value)


def _smooth_static_gain_field(
    raw_gain: np.ndarray,
    mask: np.ndarray,
    fallback_gain: np.ndarray,
    sigma: float,
    gain_low: float,
    gain_high: float,
) -> np.ndarray:
    """Smooth sparse stable-region gains into a full low-resolution gain field."""

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return np.broadcast_to(fallback_gain.reshape(1, 1, 3), raw_gain.shape).astype(np.float32)

    field = np.empty_like(raw_gain, dtype=np.float32)
    base_weight = mask.astype(np.float32)
    for channel in range(3):
        values = raw_gain[..., channel].astype(np.float32)
        finite = np.isfinite(values)
        weights = base_weight * finite.astype(np.float32)
        if float(np.sum(weights)) <= 0:
            field[..., channel] = float(fallback_gain[channel])
            continue
        numerator = gaussian_filter(np.where(finite, values, 0.0) * weights, sigma=sigma, mode="nearest")
        denominator = gaussian_filter(weights, sigma=sigma, mode="nearest")
        smoothed = np.where(denominator > 1e-6, numerator / denominator, float(fallback_gain[channel]))
        field[..., channel] = np.clip(smoothed, gain_low, gain_high)
    return field


def _upsample_static_gain_field(field: np.ndarray, shape: tuple[int, int], stride: int) -> np.ndarray:
    """Nearest-neighbour upsampling from stride-sampled grid to image shape."""

    expanded = np.repeat(np.repeat(field, stride, axis=0), stride, axis=1)
    pad_y = max(0, shape[0] - expanded.shape[0])
    pad_x = max(0, shape[1] - expanded.shape[1])
    if pad_y or pad_x:
        expanded = np.pad(expanded, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
    return expanded[: shape[0], : shape[1], :]


def _default_calibration_log_root() -> str:
    env_root = os.environ.get("FFAC_CALIBRATION_LOG_ROOT")
    return str(resolve_storage_path(env_root) if env_root else
               configured_tool_path("ffac", "calibration_logs"))


def _apply_plume_excluded_light_correction(
    loaded: List[Tuple[Any, float, float]],
    *,
    reference_rgb: np.ndarray,
    valid_mask: np.ndarray,
    run: str,
    stride: int,
    min_samples: int,
    max_samples: int,
    gain_low: float,
    gain_high: float,
) -> None:
    """Estimate one RGB gain per frame from baseline-like porous pixels only."""

    arrays = [np.asarray(image.img, dtype=np.float32) for image, _m, _t in loaded]
    if not arrays or any(array.ndim != 3 or array.shape[2] < 3 for array in arrays):
        return
    shape = arrays[0].shape[:2]
    if any(array.shape[:2] != shape for array in arrays):
        raise ValueError("Static-light frames must share one spatial shape.")
    reference = np.asarray(reference_rgb, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if reference.shape[:2] != shape or valid.shape != shape:
        raise ValueError(
            "Plume-excluded light reference, mask, and frames must have equal shape."
        )

    sampled_reference = np.maximum(reference[::stride, ::stride, :3], 1e-6)
    sampled_valid = valid[::stride, ::stride]
    stack = np.stack(
        [array[::stride, ::stride, :3] for array in arrays], axis=0
    )
    finite = np.all(np.isfinite(stack), axis=(0, 3)) & np.all(
        np.isfinite(sampled_reference), axis=2
    )
    reference_luma = np.mean(sampled_reference, axis=2)
    usable_luma = reference_luma[finite & sampled_valid]
    if not usable_luma.size:
        return
    low_luma, high_luma = np.percentile(usable_luma, [5.0, 95.0])

    reference_chroma = sampled_reference / np.maximum(
        reference_luma[..., None], 1e-6
    )
    frame_luma = np.mean(stack, axis=3)
    frame_chroma = stack / np.maximum(frame_luma[..., None], 1e-6)
    chroma_distance = np.linalg.norm(
        frame_chroma - reference_chroma[None, ...], axis=3
    )
    median_chroma_distance = np.nanmedian(chroma_distance, axis=0)
    temporal_luma_std = np.nanstd(frame_luma, axis=0)
    candidate = (
        finite
        & sampled_valid
        & (reference_luma >= low_luma)
        & (reference_luma <= high_luma)
    )
    candidate_values = median_chroma_distance[candidate]
    temporal_values = temporal_luma_std[candidate]
    if not candidate_values.size:
        return

    mask = np.zeros_like(candidate)
    for chroma_quantile, temporal_quantile in ((35.0, 50.0), (50.0, 65.0), (70.0, 80.0)):
        chroma_limit = float(np.nanpercentile(candidate_values, chroma_quantile))
        temporal_limit = float(np.nanpercentile(temporal_values, temporal_quantile))
        mask = (
            candidate
            & (median_chroma_distance <= chroma_limit)
            & (temporal_luma_std <= temporal_limit)
        )
        if int(np.count_nonzero(mask)) >= min_samples:
            break
    if int(np.count_nonzero(mask)) < min_samples:
        logger.warning(
            "[%s] plume-excluded light correction skipped: only %d baseline-like "
            "porous samples found.",
            run,
            int(np.count_nonzero(mask)),
        )
        return

    ys, xs = np.nonzero(mask)
    if ys.size > max_samples:
        choose = np.linspace(0, ys.size - 1, max_samples, dtype=int)
        ys, xs = ys[choose], xs[choose]
    reference_values = sampled_reference[ys, xs, :]
    gains: list[np.ndarray] = []
    for index, (image, _injected, _time_h) in enumerate(loaded):
        current_values = np.maximum(stack[index, ys, xs, :], 1e-6)
        gain = np.nanmedian(reference_values / current_values, axis=0)
        if not np.all(np.isfinite(gain)):
            raise RuntimeError(f"[{run}] non-finite plume-excluded light gain")
        gain = np.clip(gain.astype(np.float32), gain_low, gain_high)
        corrected = np.asarray(image.img, dtype=np.float32).copy()
        corrected[..., :3] = np.clip(
            corrected[..., :3] * gain.reshape(1, 1, 3), 0.0, 1.0
        )
        image.img = corrected
        gains.append(gain)
    gain_array = np.stack(gains)
    logger.info(
        "[%s] static light correction mode=plume-excluded samples=%d stride=%d "
        "gain_min=%s gain_max=%s",
        run,
        int(len(ys)),
        stride,
        np.round(gain_array.min(axis=0), 4).tolist(),
        np.round(gain_array.max(axis=0), 4).tolist(),
    )


def _apply_static_light_correction(
    loaded: List[Tuple[Any, float, float]],
    *,
    mode: str | None,
    run: str,
    reference_rgb: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Apply a small per-frame gain from stable FluidFlower image regions."""

    correction = _normalise_static_light_mode(mode)
    if correction == "off" or len(loaded) < 2:
        return
    if correction not in {
        "blue-gain",
        "intensity",
        "blue-spatial",
        "intensity-spatial",
        "plume-excluded",
    }:
        raise ValueError(
            "Unknown static light correction mode "
            f"{correction!r}; expected off, blue-gain, intensity, blue-spatial, "
            "intensity-spatial, or plume-excluded."
        )

    stride = max(4, int(os.environ.get("FFAC_STATIC_LIGHT_STRIDE", "16")))
    min_samples = max(1000, int(os.environ.get("FFAC_STATIC_LIGHT_MIN_SAMPLES", "4000")))
    max_samples = max(min_samples, int(os.environ.get("FFAC_STATIC_LIGHT_MAX_SAMPLES", "50000")))
    clip_frac = float(os.environ.get("FFAC_STATIC_LIGHT_GAIN_CLIP", "0.15"))
    gain_low = max(0.01, 1.0 - abs(clip_frac))
    gain_high = 1.0 + abs(clip_frac)
    spatial_sigma = float(os.environ.get("FFAC_STATIC_LIGHT_SPATIAL_SIGMA", "6.0"))

    if correction == "plume-excluded":
        if reference_rgb is None or valid_mask is None:
            raise ValueError(
                "plume-excluded light correction requires a porous baseline "
                "reference and active-domain mask"
            )
        _apply_plume_excluded_light_correction(
            loaded,
            reference_rgb=np.asarray(reference_rgb),
            valid_mask=np.asarray(valid_mask),
            run=run,
            stride=stride,
            min_samples=min_samples,
            max_samples=max_samples,
            gain_low=gain_low,
            gain_high=gain_high,
        )
        return

    samples: List[np.ndarray] = []
    for img, _injected, _t_h in loaded:
        arr = np.asarray(img.img, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return
        samples.append(arr[::stride, ::stride, :3])
    stack = np.stack(samples, axis=0)

    finite = np.all(np.isfinite(stack), axis=(0, 3))
    mean_rgb = np.nanmean(stack, axis=0)
    std_rgb = np.nanstd(stack, axis=0)
    mean_luma = np.mean(mean_rgb, axis=2)
    std_luma = np.std(stack.mean(axis=3), axis=0)
    std_chroma = np.linalg.norm(std_rgb, axis=2)

    finite_luma = mean_luma[np.isfinite(mean_luma) & finite]
    if finite_luma.size == 0:
        return
    lo, hi = np.nanpercentile(finite_luma, [5.0, 95.0])
    stable_limit = np.nanpercentile(std_luma[finite], 35.0)
    chroma_stable_limit = np.nanpercentile(std_chroma[finite], 40.0)
    blueish = (mean_rgb[..., 2] > mean_rgb[..., 0] + 0.005) & (
        mean_rgb[..., 2] > mean_rgb[..., 1] + 0.002
    )
    base_mask = (
        finite
        & (mean_luma >= lo)
        & (mean_luma <= hi)
        & (std_luma <= stable_limit)
        & (std_chroma <= chroma_stable_limit)
    )
    mask = base_mask & blueish
    if int(np.count_nonzero(mask)) < min_samples:
        mask = base_mask
    if int(np.count_nonzero(mask)) < min_samples:
        logger.warning(
            "[%s] static light correction skipped: only %d stable samples found.",
            run,
            int(np.count_nonzero(mask)),
        )
        return

    ys, xs = np.nonzero(mask)
    if ys.size > max_samples:
        choose = np.linspace(0, ys.size - 1, max_samples, dtype=int)
        ys = ys[choose]
        xs = xs[choose]

    frame_values = stack[:, ys, xs, :]
    med_rgb = np.nanmedian(frame_values, axis=1)
    if not np.all(np.isfinite(med_rgb)):
        logger.warning("[%s] static light correction skipped: non-finite medians.", run)
        return

    reference_mode = (os.environ.get("FFAC_STATIC_LIGHT_REFERENCE") or "median").strip().lower()
    ref_rgb = med_rgb[0] if reference_mode == "first" else np.nanmedian(med_rgb, axis=0)
    ref_rgb = np.maximum(ref_rgb, 1e-6)

    spatial_reference = None
    if correction in {"blue-spatial", "intensity-spatial"}:
        spatial_reference = stack[0] if reference_mode == "first" else np.nanmedian(stack, axis=0)
        spatial_reference = np.maximum(spatial_reference.astype(np.float32), 1e-6)

    gains: List[np.ndarray] = []
    field_ranges: List[Tuple[float, float]] = []
    for idx, (img, _injected, _t_h) in enumerate(loaded):
        cur_rgb = np.maximum(med_rgb[idx], 1e-6)
        if correction == "intensity":
            ref_luma = float(np.mean(ref_rgb))
            cur_luma = float(np.mean(cur_rgb))
            gain = np.array([ref_luma / max(cur_luma, 1e-6)] * 3, dtype=np.float32)
        elif correction == "blue-gain":
            gain = (ref_rgb / cur_rgb).astype(np.float32)
        elif correction == "intensity-spatial":
            assert spatial_reference is not None
            cur_grid = np.maximum(stack[idx].astype(np.float32), 1e-6)
            ref_luma_grid = np.mean(spatial_reference, axis=2)
            cur_luma_grid = np.mean(cur_grid, axis=2)
            raw_scalar = ref_luma_grid / np.maximum(cur_luma_grid, 1e-6)
            raw_gain = np.repeat(raw_scalar[..., None], 3, axis=2)
            ref_luma = float(np.mean(ref_rgb))
            cur_luma = float(np.mean(cur_rgb))
            gain = np.array([ref_luma / max(cur_luma, 1e-6)] * 3, dtype=np.float32)
        else:
            assert spatial_reference is not None
            cur_grid = np.maximum(stack[idx].astype(np.float32), 1e-6)
            raw_gain = spatial_reference / cur_grid
            gain = (ref_rgb / cur_rgb).astype(np.float32)
        gain = np.clip(gain, gain_low, gain_high)
        arr = np.asarray(img.img, dtype=np.float32).copy()
        if correction in {"blue-spatial", "intensity-spatial"}:
            raw_gain = np.clip(raw_gain.astype(np.float32), gain_low, gain_high)
            field_small = _smooth_static_gain_field(
                raw_gain,
                mask,
                gain.astype(np.float32),
                spatial_sigma,
                gain_low,
                gain_high,
            )
            field = _upsample_static_gain_field(field_small, arr.shape[:2], stride)
            arr[..., :3] = np.clip(arr[..., :3] * field, 0.0, 1.0)
            field_ranges.append((float(np.nanmin(field)), float(np.nanmax(field))))
        else:
            arr[..., :3] = np.clip(arr[..., :3] * gain.reshape(1, 1, 3), 0.0, 1.0)
        img.img = arr
        gains.append(gain)

    gain_arr = np.stack(gains, axis=0)
    field_note = ""
    if field_ranges:
        field_note = (
            f" field_min={min(v[0] for v in field_ranges):.4f}"
            f" field_max={max(v[1] for v in field_ranges):.4f}"
            f" spatial_sigma={spatial_sigma:.2f}"
        )
    logger.info(
        "[%s] static light correction mode=%s samples=%d stride=%d "
        "gain_min=%s gain_max=%s%s",
        run,
        correction,
        int(ys.size),
        stride,
        np.round(gain_arr.min(axis=0), 4).tolist(),
        np.round(gain_arr.max(axis=0), 4).tolist(),
        field_note,
    )


_ANCHOR_BASELINE_MEDIAN_CACHE: Dict[Tuple[str, Tuple[int, ...], int], np.ndarray] = {}


def _porous_baseline_median(
    rgb: np.ndarray,
    labels: np.ndarray,
    active_labels: Sequence[int],
    *,
    stride: int = 8,
) -> np.ndarray:
    """Robust RGB median in active porous facies, sampled for low memory use."""

    arr = np.asarray(rgb)
    facies = np.asarray(labels)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected RGB baseline, got shape {arr.shape}")
    if facies.shape[:2] != arr.shape[:2]:
        raise ValueError(
            f"Baseline/facies shape mismatch: {arr.shape[:2]} vs {facies.shape[:2]}"
        )
    step = max(1, int(stride))
    sampled = np.asarray(arr[::step, ::step, :3], dtype=np.float32)
    sampled_labels = facies[::step, ::step]
    mask = np.isin(sampled_labels, np.asarray(active_labels, dtype=int))
    mask &= np.all(np.isfinite(sampled), axis=2)
    if int(np.count_nonzero(mask)) < 1000:
        raise ValueError(
            f"Only {int(np.count_nonzero(mask))} active baseline samples found"
        )
    return np.asarray(np.median(sampled[mask], axis=0), dtype=np.float32)


def _compute_anchor_photometric_gain(
    target_median: np.ndarray,
    anchor_median: np.ndarray,
    *,
    gain_low: float = 0.5,
    gain_high: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    target = np.maximum(np.asarray(target_median, dtype=np.float32), 1e-6)
    anchor = np.maximum(np.asarray(anchor_median, dtype=np.float32), 1e-6)
    raw = anchor / target
    return raw, np.clip(raw, float(gain_low), float(gain_high)).astype(np.float32)


def _apply_anchor_photometric_gain(image: Any, gain: np.ndarray) -> None:
    gain_arr = np.asarray(gain, dtype=np.float32).reshape(1, 1, 3)
    if np.allclose(gain_arr, 1.0, rtol=0.0, atol=1e-7):
        return
    source = np.asarray(image.img)
    arr = np.asarray(source, dtype=np.float32).copy()
    arr[..., :3] = np.clip(arr[..., :3] * gain_arr, 0.0, 1.0)
    image.img = arr.astype(source.dtype, copy=False)


def _setup_anchor_photometric_gain(
    *,
    run: str,
    config: Any,
    fluidflower: Any,
    active_labels: Sequence[int],
    master_light_context: bool,
) -> Optional[np.ndarray]:
    """Derive one global RGB gain from target and anchor pre-injection baselines."""

    anchor_run = os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_RUN", "").strip().lower()
    if anchor_run in {"", "0", "false", "no", "none", "off"}:
        return None
    if master_light_context:
        logger.info(
            "[%s] photometric anchor=%s deferred to workers (light master)",
            run,
            anchor_run,
        )
        return None

    try:
        results_dir = Path(getattr(getattr(config, "data", None), "results"))
        results_root = results_dir.parent
        target_rig = results_dir / "setup" / "rig"
        anchor_rig = results_root / anchor_run / "setup" / "rig"
        target_labels_path = target_rig / "facies.npz"
        anchor_labels_path = anchor_rig / "facies.npz"
        anchor_baseline_path = anchor_rig / "shape_corrected_baseline.npz"
        for required in (target_labels_path, anchor_labels_path, anchor_baseline_path):
            if not required.exists():
                raise FileNotFoundError(required)

        stride = max(1, int(os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_STRIDE", "8")))
        labels_tuple = tuple(sorted(int(label) for label in active_labels))
        target_labels = np.load(target_labels_path, allow_pickle=True)["array"]
        target_rgb = np.asarray(
            getattr(fluidflower, "shape_corrected_baseline").img,
            dtype=np.float32,
        )
        target_median = _porous_baseline_median(
            target_rgb,
            target_labels,
            labels_tuple,
            stride=stride,
        )

        cache_key = (str(anchor_baseline_path).lower(), labels_tuple, stride)
        anchor_median = _ANCHOR_BASELINE_MEDIAN_CACHE.get(cache_key)
        if anchor_median is None:
            anchor_rgb = _registration_baseline_array(anchor_baseline_path)
            anchor_labels = np.load(anchor_labels_path, allow_pickle=True)["array"]
            anchor_median = _porous_baseline_median(
                anchor_rgb,
                anchor_labels,
                labels_tuple,
                stride=stride,
            )
            _ANCHOR_BASELINE_MEDIAN_CACHE[cache_key] = anchor_median

        gain_low = float(os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_GAIN_MIN", "0.5"))
        gain_high = float(os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_GAIN_MAX", "2.0"))
        if not 0.0 < gain_low <= gain_high:
            raise ValueError(
                f"Invalid photometric anchor gain limits [{gain_low}, {gain_high}]"
            )
        raw_gain, gain = _compute_anchor_photometric_gain(
            target_median,
            anchor_median,
            gain_low=gain_low,
            gain_high=gain_high,
        )
        logger.info(
            "[%s] photometric anchor ACTIVE target=%s anchor=%s mode=baseline-diagonal "
            "target_rgb=%s anchor_rgb=%s raw_gain=%s gain=%s",
            run,
            run,
            anchor_run,
            np.round(target_median, 5).tolist(),
            np.round(anchor_median, 5).tolist(),
            np.round(raw_gain, 5).tolist(),
            np.round(gain, 5).tolist(),
        )
        return gain
    except Exception as exc:  # noqa: BLE001
        strict = os.environ.get("FFAC_PHOTOMETRIC_ANCHOR_STRICT", "").strip().lower()
        if strict in {"1", "true", "yes", "on"}:
            raise
        logger.warning("[%s] photometric anchor requested but skipped: %s", run, exc)
        return None


def _template_registration_target() -> str:
    value = os.environ.get("FFAC_TEMPLATE_REGISTRATION", "").strip().lower()
    if value in {"", "0", "false", "no", "none", "off"}:
        return ""
    return value


def _registration_baseline_array(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "array" in data:
        arr = data["array"]
    elif "img" in data:
        arr = data["img"]
    else:
        raise KeyError(f"{path} does not contain an image array")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3]


def _registration_gray(arr: np.ndarray, *, max_col: int, scale: float) -> np.ndarray:
    import cv2

    cropped = arr[:, : min(max_col, arr.shape[1]), :3]
    cropped = np.clip(cropped, 0.0, 1.0)
    gray = cv2.cvtColor((cropped * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray)


def _estimate_template_affine(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    mode: str,
    scale: float,
    max_col: int,
    max_features: int,
    keep_matches: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Estimate source->template registration from baseline images.

    The default mode is a partial affine transform, i.e. translation + rotation +
    uniform scale. This is intentionally less flexible than a homography so the
    mass integral is not changed by arbitrary perspective/shear deformation.
    """

    import cv2
    import math

    src_gray = _registration_gray(src, max_col=max_col, scale=scale)
    dst_gray = _registration_gray(dst, max_col=max_col, scale=scale)

    detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=8)
    kp_src, des_src = detector.detectAndCompute(src_gray, None)
    kp_dst, des_dst = detector.detectAndCompute(dst_gray, None)
    if des_src is None or des_dst is None or len(kp_src) < 10 or len(kp_dst) < 10:
        raise RuntimeError("too few registration features")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_src, des_dst), key=lambda m: m.distance)
    matches = matches[: max(10, keep_matches)]
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    mode = mode.strip().lower().replace("-", "_")
    if mode in {"similarity", "partial", "partial_affine", "affine"}:
        mat, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.995,
        )
        if mat is None:
            raise RuntimeError("partial affine registration failed")
        mat = mat.astype(np.float32)
        mat[:, 2] /= float(scale)
        a, b, tx = [float(x) for x in mat[0]]
        c, d, ty = [float(x) for x in mat[1]]
        reg_scale = math.sqrt(a * a + c * c)
        angle = math.degrees(math.atan2(c, a))
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        return mat, {
            "matches": float(len(matches)),
            "inliers": float(inlier_count),
            "dx_px": tx,
            "dy_px": ty,
            "scale": reg_scale,
            "angle_deg": angle,
        }
    if mode == "translation":
        # Use partial affine for robust matching, then keep only the translation.
        mat, stats = _estimate_template_affine(
            src,
            dst,
            mode="partial_affine",
            scale=scale,
            max_col=max_col,
            max_features=max_features,
            keep_matches=keep_matches,
        )
        out = np.array([[1.0, 0.0, mat[0, 2]], [0.0, 1.0, mat[1, 2]]], dtype=np.float32)
        stats["scale"] = 1.0
        stats["angle_deg"] = 0.0
        return out, stats
    if mode == "homography":
        hom, inliers = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        if hom is None:
            raise RuntimeError("homography registration failed")
        s = float(scale)
        to_small = np.diag([s, s, 1.0])
        to_full = np.diag([1.0 / s, 1.0 / s, 1.0])
        hom_full = (to_full @ hom @ to_small).astype(np.float32)
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        return hom_full, {
            "matches": float(len(matches)),
            "inliers": float(inlier_count),
            "dx_px": float(hom_full[0, 2]),
            "dy_px": float(hom_full[1, 2]),
            "scale": float(np.sqrt(abs(np.linalg.det(hom_full[:2, :2])))),
            "angle_deg": float("nan"),
        }
    raise ValueError(
        f"Unknown FFAC_TEMPLATE_REGISTRATION_MODE={mode!r}; expected partial_affine, translation, or homography."
    )


def _warp_registration_image(img: Any, transform: np.ndarray, *, mode: str) -> Any:
    import cv2

    arr = np.asarray(img.img)
    height, width = arr.shape[:2]
    if mode.strip().lower().replace("-", "_") == "homography":
        warped = cv2.warpPerspective(
            arr,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    else:
        warped = cv2.warpAffine(
            arr,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    img.img = warped.astype(arr.dtype, copy=False)
    return img


def _setup_template_registration(
    *,
    run: str,
    config: Any,
    fluidflower: Any,
    calibration: Any,
) -> Tuple[Optional[np.ndarray], str]:
    template_run = _template_registration_target()
    if not template_run or template_run == run.lower():
        return None, "off"

    mode = os.environ.get("FFAC_TEMPLATE_REGISTRATION_MODE", "partial_affine")
    scale = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_SCALE", "0.25"))
    max_col = int(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_COL", "4800"))
    max_features = int(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_FEATURES", "12000"))
    keep_matches = int(os.environ.get("FFAC_TEMPLATE_REGISTRATION_KEEP_MATCHES", "800"))
    min_inlier_frac = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MIN_INLIER_FRAC", "0.50"))
    max_abs_shift = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_SHIFT_PX", "500"))
    min_scale = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MIN_SCALE", "0.90"))
    max_scale = float(os.environ.get("FFAC_TEMPLATE_REGISTRATION_MAX_SCALE", "1.10"))

    try:
        results_dir = Path(getattr(getattr(config, "data", None), "results"))
        template_dir = results_dir.parent / template_run
        template_path = template_dir / "setup" / "rig" / "shape_corrected_baseline.npz"
        if not template_path.exists():
            raise FileNotFoundError(template_path)
        src = np.asarray(getattr(fluidflower, "shape_corrected_baseline").img, dtype=np.float32)[..., :3]
        dst = _registration_baseline_array(template_path)
        transform, stats = _estimate_template_affine(
            src,
            dst,
            mode=mode,
            scale=scale,
            max_col=max_col,
            max_features=max_features,
            keep_matches=keep_matches,
        )
        inlier_frac = stats["inliers"] / max(stats["matches"], 1.0)
        if inlier_frac < min_inlier_frac:
            raise RuntimeError(f"low inlier fraction {inlier_frac:.3f}")
        if abs(stats["dx_px"]) > max_abs_shift or abs(stats["dy_px"]) > max_abs_shift:
            raise RuntimeError(f"large shift dx={stats['dx_px']:.1f}, dy={stats['dy_px']:.1f}")
        if mode.strip().lower().replace("-", "_") != "homography":
            if not (min_scale <= stats["scale"] <= max_scale):
                raise RuntimeError(f"scale {stats['scale']:.5f} outside [{min_scale}, {max_scale}]")

        color_base = getattr(getattr(calibration, "color_analysis", None), "base", None)
        if color_base is not None:
            _warp_registration_image(color_base, transform, mode=mode)

        logger.info(
            "[%s] template registration ACTIVE target=%s mode=%s "
            "matches=%d inliers=%d frac=%.3f dx=%.1f dy=%.1f scale=%.5f angle=%.3f",
            run,
            template_run,
            mode,
            int(stats["matches"]),
            int(stats["inliers"]),
            inlier_frac,
            stats["dx_px"],
            stats["dy_px"],
            stats["scale"],
            stats["angle_deg"],
        )
        return transform, mode
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("FFAC_TEMPLATE_REGISTRATION_STRICT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise
        logger.warning("[%s] template registration requested but skipped: %s", run, exc)
        return None, "off"


def _path_arc_length(relative_colors: Sequence[np.ndarray]) -> float:
    colors = np.asarray(relative_colors, dtype=np.float64)
    if colors.ndim != 2 or len(colors) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(colors, axis=0), axis=1)))


def _resample_relative_path(color_path: Any, count: int) -> np.ndarray:
    colors = np.asarray(color_path.relative_colors, dtype=np.float64)
    source = np.asarray(color_path.relative_distances, dtype=np.float64)
    target = np.linspace(0.0, 1.0, int(count))
    return np.column_stack(
        [np.interp(target, source, colors[:, channel]) for channel in range(3)]
    )


def _resolve_color_path_source_folder(source_root: Path, run: str) -> Path:
    """Resolve a pristine per-run color-path folder without modifying results."""

    candidates = (
        source_root / run / "calibration" / "color_path_interpretation",
        source_root / run / "color_path_interpretation",
        source_root / "calibration" / "color_path_interpretation",
        source_root / "color_path_interpretation",
        source_root,
    )
    for candidate in candidates:
        if candidate.is_dir() and any(
            candidate.glob("color_path_interpretation_*.json")
        ):
            return candidate
    raise FileNotFoundError(
        f"No color_path_interpretation_*.json found for {run} below {source_root}"
    )


def _expand_run_path_template(raw_path: str, run: str) -> Path:
    """Expand the explicit ``{run}`` placeholder in campaign-owned paths."""

    expanded = str(raw_path).replace("{run}", str(run).lower())
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "Campaign path templates must resolve to an absolute path; "
            f"got {raw_path!r} -> {expanded!r}"
        )
    return path


def _restore_color_paths(
    *,
    run: str,
    calibration: Any,
    labels: Sequence[int],
    source_root: str | Path | None,
    strict: bool = False,
) -> Optional[Path]:
    """Restore the same pristine path basis before comparing anchor variants."""

    raw_root = str(
        source_root
        if source_root is not None
        else os.environ.get("FFAC_COLOR_PATH_SOURCE_ROOT", "")
    ).strip()
    if not raw_root:
        return None
    try:
        import darsia

        folder = _resolve_color_path_source_folder(Path(raw_root), run)
        mapping = calibration.color_path_interpretation
        executable = calibration.color_analysis.model[0]
        restored: list[int] = []
        for label in labels:
            source = darsia.ColorPathInterpolation.load(
                folder / f"color_path_interpretation_{int(label)}"
            )
            target = mapping[int(label)]
            target.color_path = copy.deepcopy(source.color_path)
            target.values = np.asarray(source.values, dtype=np.float64).copy()
            try:
                executable_target = executable[int(label)]
                executable_target.color_path = target.color_path
                executable_target.values = target.values.copy()
            except (KeyError, TypeError, AttributeError):
                pass
            restored.append(int(label))
        logger.info(
            "[%s] pristine color paths restored from %s for labels=%s",
            run,
            folder,
            restored,
        )
        return folder
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        logger.warning("[%s] pristine color-path restore skipped: %s", run, exc)
        return None


def _regularize_color_paths(
    *,
    run: str,
    config: Any,
    calibration: Any,
    calibration_folder: Path,
    labels: Sequence[int],
    anchor_run: str | None = None,
    anchor_weight: float | None = None,
    anchor_strict: bool | None = None,
    source_root: str | Path | None = None,
) -> None:
    """Blend noisy per-run path shapes toward a shared, independently observed path.

    The current run keeps its own baseline and path amplitude. Only the normalized
    RGB trajectory is shrunk toward the anchor, so colour correction remains
    run-specific while implausible path tortuosity is regularized.
    """

    anchor_run = (
        anchor_run
        if anchor_run is not None
        else os.environ.get("FFAC_COLOR_PATH_ANCHOR", "")
    ).strip().lower()
    if not anchor_run or anchor_run in {"off", "none"}:
        return
    weight = float(
        anchor_weight
        if anchor_weight is not None
        else os.environ.get("FFAC_COLOR_PATH_ANCHOR_WEIGHT", "0.75")
    )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            f"FFAC_COLOR_PATH_ANCHOR_WEIGHT must be within [0, 1], got {weight}"
        )
    if anchor_run == run.lower() or weight == 0.0:
        logger.info(
            "[%s] color-path anchor is the current run; no regularization needed.",
            run,
        )
        return

    try:
        import darsia

        results_dir = Path(getattr(getattr(config, "data", None), "results"))
        relative_folder = calibration_folder.relative_to(results_dir)
        raw_source_root = str(
            source_root
            if source_root is not None
            else os.environ.get("FFAC_COLOR_PATH_SOURCE_ROOT", "")
        ).strip()
        anchor_folder = (
            _resolve_color_path_source_folder(Path(raw_source_root), anchor_run)
            if raw_source_root
            else (
                results_dir.parent
                / anchor_run
                / relative_folder
                / "color_path_interpretation"
            )
        )
        changed: list[str] = []
        for label in labels:
            target_interpolation = calibration.color_path_interpretation[int(label)]
            anchor_path = (
                anchor_folder / f"color_path_interpretation_{int(label)}"
            )
            anchor_interpolation = darsia.ColorPathInterpolation.load(anchor_path)
            target_path = target_interpolation.color_path
            count = len(target_path.relative_colors)
            target_colors = _resample_relative_path(target_path, count)
            anchor_colors = _resample_relative_path(
                anchor_interpolation.color_path,
                count,
            )
            target_arc = _path_arc_length(target_colors)
            anchor_arc = _path_arc_length(anchor_colors)
            if target_arc <= 1e-12 or anchor_arc <= 1e-12:
                raise ValueError(
                    f"label {label}: zero-length target/anchor color path"
                )
            anchor_colors *= target_arc / anchor_arc
            blended = (1.0 - weight) * target_colors + weight * anchor_colors
            blended[0] = 0.0
            regularized_path = darsia.ColorPath(
                base_color=np.asarray(target_path.base_color, dtype=np.float64),
                relative_colors=[row for row in blended],
                mode=target_path.mode,
                name=f"{target_path.name}|anchor={anchor_run}|weight={weight:g}",
            )
            target_interpolation.color_path = regularized_path
            # HeterogeneousModel shallow-copies interpolation objects at setup.
            # Keep the executable copy synchronized with the public mapping.
            calibration.color_analysis.model[0][int(label)].color_path = (
                regularized_path
            )
            changed.append(
                f"{label}:arc={target_arc:.4g},anchor_scale={target_arc / anchor_arc:.4g}"
            )
        logger.info(
            "[%s] color-path regularization ACTIVE anchor=%s weight=%.3f [%s]",
            run,
            anchor_run,
            weight,
            "; ".join(changed),
        )
    except Exception as exc:  # noqa: BLE001
        strict = (
            bool(anchor_strict)
            if anchor_strict is not None
            else os.environ.get(
                "FFAC_COLOR_PATH_ANCHOR_STRICT", ""
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if strict:
            raise
        logger.warning(
            "[%s] color-path regularization requested but skipped: %s",
            run,
            exc,
        )


def _optical_residual_gas_score(
    calibration: Any,
    image: Any,
    labels: Sequence[int],
) -> np.ndarray:
    """Return a quantized gas observable independent of color-path position.

    Aqueous indicator changes should stay close to the calibrated RGB path.
    Gas bubbles and gas-water interfaces add reflection/scattering and local
    high-frequency structure, producing an off-path optical residual.
    """

    import cv2

    observed = np.asarray(image.img, dtype=np.float32)[..., :3]
    base = getattr(getattr(calibration, "color_analysis", None), "base", None)
    if base is not None:
        observed = observed - np.asarray(base.img, dtype=np.float32)[..., :3]
    label_image = np.asarray(calibration.labels.img)
    if label_image.shape != observed.shape[:2]:
        raise ValueError(
            "Residual-gas labels/image shape mismatch: "
            f"{label_image.shape} != {observed.shape[:2]}"
        )

    gray = np.mean(observed, axis=2, dtype=np.float32)
    sigma = max(
        0.1,
        float(os.environ.get("FFAC_RESIDUAL_GAS_TEXTURE_SIGMA", "2.0")),
    )
    texture_weight = max(
        0.0,
        float(os.environ.get("FFAC_RESIDUAL_GAS_TEXTURE_WEIGHT", "0.5")),
    )
    texture = np.abs(
        gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    )
    score = np.zeros(label_image.shape, dtype=np.float32)
    flat_observed = observed.reshape(-1, 3)
    flat_texture = texture.reshape(-1)
    flat_score = score.reshape(-1)
    chunk_size = max(
        10_000,
        int(os.environ.get("FFAC_RESIDUAL_GAS_CHUNK_PIXELS", "250000")),
    )

    for label in labels:
        interpolation = calibration.color_analysis.model[0][int(label)]
        color_path = interpolation.color_path
        positions = np.flatnonzero(label_image.reshape(-1) == int(label))
        if not positions.size:
            continue
        path_scale = max(
            1e-6,
            float(
                np.sum(
                    np.linalg.norm(
                        np.diff(
                            np.asarray(
                                color_path.relative_colors,
                                dtype=np.float32,
                            ),
                            axis=0,
                        ),
                        ord=1,
                        axis=1,
                    )
                )
            ),
        )
        for start in range(0, positions.size, chunk_size):
            index = positions[start : start + chunk_size]
            colors = flat_observed[index]
            parameters = color_path.fit(
                colors=colors,
                color_mode=interpolation.color_mode,
                mode="equidistant",
            )
            projected = color_path.interpret(
                parameters,
                color_mode=interpolation.color_mode,
                mode="equidistant",
            )
            off_path = np.linalg.norm(colors - projected, ord=1, axis=1)
            flat_score[index] = (
                off_path + texture_weight * flat_texture[index]
            ) / path_scale

    return np.rint(
        np.clip(score / _RESIDUAL_GAS_SCORE_CLIP, 0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _precompute_residual_gas_scores(
    calibration: Any,
    loaded: Sequence[Tuple[Any, float, float]],
    labels: Sequence[int],
    *,
    run: str,
) -> List[np.ndarray]:
    scores: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    active = np.isin(np.asarray(calibration.labels.img), np.asarray(labels))
    active_flat = active.reshape(-1)
    for image, _injected, _time_h in loaded:
        quantized = _optical_residual_gas_score(calibration, image, labels)
        scores.append(quantized)
        values = quantized.reshape(-1)[active_flat][::200]
        if values.size:
            samples.append(values)
    if samples:
        sampled = (
            np.concatenate(samples).astype(np.float32)
            * (_RESIDUAL_GAS_SCORE_CLIP / 255.0)
        )
        percentiles = np.percentile(sampled, [50.0, 90.0, 99.0, 99.9])
        logger.info(
            "[%s] residual-gas optical score ACTIVE; p50/p90/p99/p99.9=%s",
            run,
            np.round(percentiles, 4).tolist(),
        )
    return scores


def _precompute_supervised_dual_gas_scores(
    calibration: Any,
    loaded: Sequence[Tuple[Any, float, float]],
    valid_mask: np.ndarray,
    *,
    run: str,
) -> tuple[List[np.ndarray], Dict[str, Any]]:
    """Evaluate the reviewed optical gas model once per corrected frame."""

    raw_model_path = os.environ.get("FFAC_DUAL_SIGNAL_MODEL", "").strip()
    if not raw_model_path:
        raise ValueError(
            "supervised-dual phase separation requires FFAC_DUAL_SIGNAL_MODEL"
        )
    model_path = Path(raw_model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Dual-signal model not found: {model_path}")
    model = DualPhaseSignalModel.load(model_path)
    training = dict(model.training_provenance)
    deployment_allowed = bool(training.get("deployment_allowed", False))
    allow_preflight = os.environ.get(
        "FFAC_DUAL_SIGNAL_ALLOW_PREFLIGHT", "off"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not deployment_allowed and not allow_preflight:
        raise RuntimeError(
            "Dual-signal model is marked deployment_allowed=false. Rebuild it "
            "from the target campaign's corrected frames, or use "
            "FFAC_DUAL_SIGNAL_ALLOW_PREFLIGHT=on only for a labelled preflight."
        )

    runtime_validation: dict[str, Any] = {"status": "preflight-override"}
    if deployment_allowed:
        expected_environment = training.get("preprocessing_environment")
        if not isinstance(expected_environment, Mapping) or not expected_environment:
            raise RuntimeError(
                "Deployable dual-signal model lacks a preprocessing environment "
                "fingerprint"
            )
        environment_mismatches = {
            str(name): {
                "expected": str(expected),
                "actual": os.environ.get(str(name)),
            }
            for name, expected in expected_environment.items()
            if os.environ.get(str(name)) != str(expected)
        }
        expected_code = training.get("code_sha256")
        if not isinstance(expected_code, Mapping):
            raise RuntimeError("Deployable dual-signal model lacks code fingerprints")
        signal_module = sys.modules.get(DualPhaseSignalModel.__module__)
        signal_path = Path(getattr(signal_module, "__file__", ""))
        colorchecker_path = Path(colorchecker_refinement.__file__)
        actual_code = {
            "signal_model": (
                hashlib.sha256(signal_path.read_bytes()).hexdigest()
                if signal_path.is_file()
                else None
            ),
            "colorchecker": hashlib.sha256(
                colorchecker_path.read_bytes()
            ).hexdigest(),
            **darsia_runtime_fingerprints(),
        }
        if int(getattr(model, "schema", 1)) >= 2:
            optical_feature_path = signal_path.with_name("optical_phase_features.py")
            actual_code["optical_features"] = (
                hashlib.sha256(optical_feature_path.read_bytes()).hexdigest()
                if optical_feature_path.is_file()
                else None
            )
        code_mismatches = {
            name: {"expected": expected_code.get(name), "actual": actual}
            for name, actual in actual_code.items()
            if expected_code.get(name) != actual
        }
        if environment_mismatches or code_mismatches:
            raise RuntimeError(
                "Dual-signal preprocessing fingerprint mismatch: "
                + json.dumps(
                    {
                        "environment": environment_mismatches,
                        "code": code_mismatches,
                    },
                    sort_keys=True,
                )
            )
        runtime_validation = {
            "status": "exact-match",
            "environment_sha256": training.get(
                "preprocessing_environment_sha256"
            ),
            "code_sha256": actual_code,
        }

    base = getattr(getattr(calibration, "color_analysis", None), "base", None)
    if base is None:
        raise RuntimeError("supervised-dual requires the corrected analysis baseline")
    baseline = np.asarray(base.img, dtype=np.float32)[..., :3]
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != baseline.shape[:2]:
        raise ValueError(
            f"Dual-signal mask/baseline shape mismatch: {valid.shape} != "
            f"{baseline.shape[:2]}"
        )

    scores: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    previous_frame: np.ndarray | None = None
    for image, _injected, _time_h in loaded:
        current_frame = np.asarray(image.img, dtype=np.float32)[..., :3]
        probability = model.predict_score(
            current_frame,
            baseline,
            previous_rgb=previous_frame,
            chunk_rows=int(os.environ.get("FFAC_DUAL_SIGNAL_CHUNK_ROWS", "64")),
        )
        previous_frame = current_frame
        probability[~valid] = 0.0
        quantized = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(
            np.uint8
        )
        scores.append(quantized)
        sampled = probability[valid][::200]
        if sampled.size:
            samples.append(sampled)
    percentiles: list[float] = []
    if samples:
        percentiles = [
            float(value)
            for value in np.percentile(
                np.concatenate(samples), [50.0, 90.0, 99.0, 99.9]
            )
        ]
        logger.info(
            "[%s] supervised dual gas signal ACTIVE; p50/p90/p99/p99.9=%s",
            run,
            np.round(percentiles, 4).tolist(),
        )
    payload = model.to_dict()
    provenance = {
        "kind": "reviewed-independent-optical-gas-probability",
        "model_path": str(model_path),
        "model_file_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_sha256": str(payload["model_sha256"]),
        "model_algorithm": str(payload["algorithm"]),
        "score_scale": 1.0,
        "deployment_allowed": deployment_allowed,
        "preflight_override": bool(allow_preflight and not deployment_allowed),
        "probability_percentiles": percentiles,
        "training_provenance": training,
        "runtime_validation": runtime_validation,
        "coupling": "gas=Sg; aqueous pore fraction=(1-Sg)",
        "aqueous_arrival_gate": False,
    }
    return scores, provenance


def _precompute_clearwater_gas_scores(
    *,
    baseline_rgb: np.ndarray,
    clear_rgb: np.ndarray,
    loaded: Sequence[Tuple[Any, float, float]],
    valid_mask: np.ndarray,
    run: str,
) -> List[np.ndarray]:
    try:
        from darsia.fluidflower.tools.clearwater_phase_reference import clearwater_projection_score
    except ModuleNotFoundError:
        from darsia.fluidflower.tools.clearwater_phase_reference import clearwater_projection_score

    orthogonal_scale = max(
        1e-6,
        float(os.environ.get("FFAC_CLEARWATER_ORTHOGONAL_SCALE", "0.20")),
    )
    scores: list[np.ndarray] = []
    samples: list[np.ndarray] = []
    valid = np.asarray(valid_mask, dtype=bool)
    for image, _injected, _time_h in loaded:
        score, evidence, _orthogonal = clearwater_projection_score(
            baseline_rgb,
            clear_rgb,
            np.asarray(image.img, dtype=np.float32),
            valid,
            orthogonal_scale=orthogonal_scale,
        )
        quantized = np.rint(np.clip(score, 0.0, 1.0) * 255.0).astype(np.uint8)
        scores.append(quantized)
        values = score[evidence][::200]
        if values.size:
            samples.append(values)
    if samples:
        sampled = np.concatenate(samples).astype(np.float32)
        percentiles = np.percentile(sampled, [50.0, 75.0, 90.0, 99.0])
        logger.info(
            "[%s] clear-water gas score ACTIVE; p50/p75/p90/p99=%s",
            run,
            np.round(percentiles, 4).tolist(),
        )
    return scores


def _hybrid_coordinate_axes(reference: Any) -> tuple[np.ndarray, np.ndarray]:
    height, width = np.asarray(reference.img).shape[:2]
    rows = np.column_stack(
        (np.arange(height, dtype=float) + 0.5, np.zeros(height, dtype=float))
    )
    columns = np.column_stack(
        (np.zeros(width, dtype=float), np.arange(width, dtype=float) + 0.5)
    )
    y_m = np.asarray(reference.coordinatesystem.coordinate(rows))[:, 1]
    x_m = np.asarray(reference.coordinatesystem.coordinate(columns))[:, 0]
    return x_m, y_m


def _load_hybrid_injection_stages(
    config_path: Path,
    experiment_start: Any,
) -> tuple[dict[str, float], ...]:
    import tomllib
    from datetime import datetime

    with Path(config_path).open("rb") as stream:
        config = tomllib.load(stream)
    protocol_path = Path(config["protocols"]["injection"])
    if not protocol_path.is_absolute():
        protocol_path = SCRIPT_DIR.parent / protocol_path
    rows: list[dict[str, Any]] = []
    with protocol_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["rate_sccm"]) <= 0.0:
                continue
            rows.append(row)
    if not rows:
        return ()

    origin = experiment_start
    if not isinstance(origin, datetime):
        origin = datetime.fromisoformat(str(origin))
    if origin.tzinfo is not None:
        origin = origin.replace(tzinfo=None)
    locations: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["location_x"]), float(row["location_y"]))
        locations.setdefault(key, []).append(row)

    stages: list[dict[str, float]] = []
    for (x_m, y_m), location_rows in locations.items():
        starts = [datetime.fromisoformat(row["start"]) for row in location_rows]
        ends = [datetime.fromisoformat(row["end"]) for row in location_rows]
        stages.append(
            {
                "x_m": x_m,
                "y_m": y_m,
                "start_h": (min(starts) - origin).total_seconds() / 3600.0,
                "end_h": (max(ends) - origin).total_seconds() / 3600.0,
            }
        )
    return tuple(sorted(stages, key=lambda stage: stage["start_h"]))


def _precompute_hybrid_gas_support(
    *,
    gas_scores: Sequence[np.ndarray],
    loaded: Sequence[Tuple[Any, float, float]],
    valid_mask: np.ndarray,
    facies_labels: np.ndarray,
    config_path: Path,
    experiment_start: Any,
    run: str,
) -> tuple[list[np.ndarray], np.ndarray, float, dict[str, Any]]:
    if not loaded or len(gas_scores) != len(loaded):
        raise RuntimeError(
            f"[{run}] hybrid gas support requires one optical score per frame"
        )
    stages = _load_hybrid_injection_stages(config_path, experiment_start)
    if not stages:
        raise RuntimeError(f"[{run}] hybrid gas support found no injection stages")
    x_m, y_m = _hybrid_coordinate_axes(loaded[0][0])
    settings = CausalGasGateSettings(
        optical_onset=float(os.environ.get("FFAC_HYBRID_OPTICAL_ONSET", "0.23")),
        optical_width=float(os.environ.get("FFAC_HYBRID_OPTICAL_WIDTH", "0.08")),
        optical_smoothing_sigma_m=float(
            os.environ.get("FFAC_HYBRID_OPTICAL_SMOOTHING_SIGMA_M", "0.0015")
        ),
        minimum_component_area_m2=float(
            os.environ.get("FFAC_HYBRID_MIN_COMPONENT_AREA_M2", "1e-5")
        ),
        history_radius_m=float(
            os.environ.get("FFAC_HYBRID_HISTORY_RADIUS_M", "0.005")
        ),
        reentry_radius_m=float(
            os.environ.get("FFAC_HYBRID_REENTRY_RADIUS_M", "0.015")
        ),
    )
    score_fields = [
        np.asarray(score, dtype=np.float32) / np.float32(255.0)
        for score in gas_scores
    ]
    weights = build_causal_optical_gate_weights(
        score_fields,
        np.asarray(valid_mask, dtype=bool),
        x_coordinates_m=x_m,
        y_coordinates_m=y_m,
        elapsed_hours=[float(row[2]) for row in loaded],
        injection_stages=stages,
        facies_labels=np.asarray(facies_labels),
        settings=settings,
    )
    quantized = [
        np.rint(np.clip(weight, 0.0, 1.0) * 255.0).astype(np.uint8)
        for weight in weights
    ]
    nonseal = ~np.isin(np.asarray(facies_labels), np.asarray(settings.seal_facies_ids))
    injection_end_h = max(float(stage["end_h"]) for stage in stages)
    provenance = {
        "kind": "parameter-independent-causal-clearwater-hybrid",
        "injection_end_h": injection_end_h,
        "settings": {
            key: value
            for key, value in settings.__dict__.items()
        },
        "stages": [dict(stage) for stage in stages],
        "strict_nonseal_fraction": float(np.mean(nonseal)),
    }
    logger.info(
        "[%s] causal hybrid gas support ACTIVE; injection_end=%.3fh, "
        "mean optical gate=%.4f",
        run,
        injection_end_h,
        float(np.mean([np.mean(weight) for weight in weights])),
    )
    return quantized, nonseal.astype(np.uint8), injection_end_h, provenance


def _hard_nonseal_gas_support(
    facies_labels: np.ndarray,
    *,
    run: str,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Resolve the optional phase invariant forbidding free gas in seals."""

    enabled = os.environ.get("FFAC_HARD_SEAL_GAS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None, None
    raw_ids = os.environ.get("FFAC_HARD_SEAL_FACIES_IDS", "1")
    try:
        seal_ids = tuple(
            sorted({int(value.strip()) for value in raw_ids.split(",") if value.strip()})
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid FFAC_HARD_SEAL_FACIES_IDS={raw_ids!r}"
        ) from exc
    if not seal_ids:
        raise ValueError("At least one hard seal facies id is required")
    facies = np.asarray(facies_labels)
    nonseal = ~np.isin(facies, np.asarray(seal_ids, dtype=facies.dtype))
    provenance = {
        "active": True,
        "seal_facies_ids": list(seal_ids),
        "allowed_fraction": float(np.mean(nonseal)),
        "operation": "saturation_g *= nonseal_mask",
    }
    logger.info(
        "[%s] hard seal gas exclusion ACTIVE facies=%s allowed_fraction=%.4f",
        run,
        seal_ids,
        provenance["allowed_fraction"],
    )
    return nonseal.astype(np.uint8), provenance


def _phase_score_parameters(
    params: Mapping[str, Any],
    mode: str,
) -> tuple[float, float]:
    if mode == "residual-gas":
        onset = float(params.get("gas.residual_onset", 0.20))
        width = float(params.get("gas.residual_width", 0.40))
    elif mode == "supervised-dual":
        onset = float(params.get("gas.dual_onset", 0.50))
        width = float(params.get("gas.dual_width", 0.25))
    else:
        onset = float(params.get("gas.clear_onset", 0.20))
        width = float(params.get("gas.clear_width", 0.40))
    return onset, max(1e-6, width)


def write_history_csv(path: Path, history: Sequence[Dict[str, Any]]) -> None:  # type: ignore
    rows = list(history or [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("iter,objective\n", encoding="utf-8"); return
    keys = sorted({k for r in rows for k in r.keys()})
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================================================================
# Signal-value application (monotone) - mirrors the DarSIA UI / ff_um
# =========================================================================
def _apply_to_func(func, idx_vals, np_module) -> None:
    vals = [float(x) for x in list(getattr(func, "values"))]
    for i, x in idx_vals.items():
        if 0 <= i < len(vals):
            vals[i] = float(x)
    if np_module is not None and len(vals) > 1:
        vals = list(np_module.maximum.accumulate(np_module.asarray(vals, dtype=float)))
    else:
        for i in range(1, len(vals)):
            vals[i] = max(vals[i], vals[i - 1])
    func.update(values=vals)


def apply_params(calibration, params, labels=None, np_module=None) -> int:
    hetero = calibration.signal_model.model[1]
    per_label: Dict[int, Dict[int, float]] = {}
    gains: Dict[int, float] = {}
    global_idx: Dict[int, float] = {}
    for k, v in params.items():
        parsed = _parse_signal_name(k)
        if parsed:
            label, idx = parsed
            per_label.setdefault(label, {})[idx] = float(v)
            continue
        gain = _SIGNAL_GAIN_RE.match(k)
        if gain:
            gains[int(gain.group("label"))] = float(v)
            continue
        mv = _VALUE_RE.match(k)
        if mv and not k.lower().startswith("flash"):
            global_idx[int(mv.group(1))] = float(v)
    if labels is None:
        labels = list(hetero.keys()) if hasattr(hetero, "keys") else []
    shared_shape = per_label.get(_SHARED_SIGNAL_LABEL, {})
    facies_shapes = {
        facies: values
        for key, values in per_label.items()
        if (facies := _facies_from_signal_key(key)) is not None
    }
    label_to_facies = {
        int(label): int(facies)
        for label, facies in getattr(
            calibration,
            "_ffac_signal_label_to_facies",
            {},
        ).items()
    }
    updated = 0
    for lbl in labels:
        idx_vals = dict(global_idx)
        if shared_shape:
            amplitude = gains.get(int(lbl), 1.0)
            idx_vals.update(
                {idx: float(fraction) * amplitude for idx, fraction in shared_shape.items()}
            )
        facies = label_to_facies.get(int(lbl))
        facies_shape = facies_shapes.get(facies) if facies is not None else None
        if facies_shape:
            amplitude = gains.get(int(lbl), 1.0)
            idx_vals.update(
                {
                    idx: float(fraction) * amplitude
                    for idx, fraction in facies_shape.items()
                }
            )
        idx_vals.update(per_label.get(int(lbl), {}))
        if not idx_vals:
            continue
        try:
            func = hetero[lbl]
        except (KeyError, TypeError):
            continue
        _apply_to_func(func, idx_vals, np_module)
        updated += 1
    # Apply flash params (e.g. flash.max_value_g). SimpleFlash.__call__ reads the
    # attributes live, so a direct setattr is enough (and avoids update()'s
    # `x or self.x` bug that drops 0.0).
    flash = getattr(calibration, "flash", None)
    if flash is not None:
        for name, val in params.items():
            if name.startswith("flash."):
                attr = name.split(".", 1)[1]
                if hasattr(flash, attr):
                    setattr(flash, attr, float(val)); updated += 1
        # Physical coupling (opt-in via env FFAC_COUPLE_AQ_GAS): gas onset = aqueous
        # saturation, i.e. min_value_g := max_value_aq. The aq->gas transition is ONE
        # point (water saturates -> free gas begins), so min_value_g is DERIVED, not
        # optimised independently. This forbids a non-physical overlap zone where a
        # pixel is counted as both partially dissolved and partially gas below
        # saturation. Off by default, so the fleet and other runs are unaffected.
        if os.environ.get("FFAC_COUPLE_AQ_GAS", "").strip().lower() in ("1", "true", "yes", "on"):
            try:
                if hasattr(flash, "max_value_aq") and hasattr(flash, "min_value_g"):
                    setattr(flash, "min_value_g", float(getattr(flash, "max_value_aq")))
            except Exception:
                pass
        # Optional T/W parameterization. T is flash.max_value_aq and the gas
        # interval is [T, T + W]. This makes both phase continuity and positive
        # gas width structural rather than relying on correlated Optuna bounds.
        if "flash.gas_width" in params:
            transition = float(getattr(flash, "max_value_aq"))
            width = max(1e-6, float(params["flash.gas_width"]))
            setattr(flash, "min_value_g", transition)
            setattr(flash, "max_value_g", transition + width)
            updated += 2
    return updated


# =========================================================================
# Context building + evaluation (ff_ac-native)
# =========================================================================
def _infer_signal_label_to_facies(
    fluidflower: Any,
    signal_labels: Sequence[int],
    *,
    use_facies: bool,
) -> Dict[int, int]:
    """Map active analysis labels to material facies from the registered rig maps."""

    labels = sorted({int(label) for label in signal_labels})
    if use_facies:
        return {label: label for label in labels}

    detailed = np.asarray(getattr(fluidflower.labels, "img", fluidflower.labels))
    facies = np.asarray(getattr(fluidflower.facies, "img", fluidflower.facies))
    if detailed.shape != facies.shape:
        raise RuntimeError(
            "Detailed-label/facies shape mismatch: "
            f"{detailed.shape} != {facies.shape}"
        )
    mapping: Dict[int, int] = {}
    for label in labels:
        values = facies[detailed == label]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise RuntimeError(
                f"Detailed label {label} has no pixels with a finite facies value"
            )
        unique, counts = np.unique(values.astype(int), return_counts=True)
        winner = int(unique[int(np.argmax(counts))])
        purity = float(np.max(counts) / np.sum(counts))
        # The facies raster is derived independently and differs by a few edge
        # pixels after rasterisation. A dominant fraction below 95% would mean
        # that the layer-to-material relation itself is ambiguous.
        if purity < 0.95:
            raise RuntimeError(
                f"Detailed label {label} crosses facies boundaries "
                f"(dominant facies {winner}, purity={purity:.6f})"
            )
        mapping[label] = winner
    return mapping


def _configure_runtime_colorchecker_refinement(
    fluidflower: Any,
) -> Dict[str, Any]:
    """Apply opt-in ColorChecker refinement to loaded correction objects."""

    switch = os.environ.get("FFAC_COLORCHECKER_REFINEMENT", "off").strip().lower()
    if switch in {"", "0", "false", "no", "off"}:
        return {"active": False, "correction_count": 0}
    if switch not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "FFAC_COLORCHECKER_REFINEMENT must be on or off; "
            f"got {switch!r}"
        )
    orient_switch = os.environ.get(
        "FFAC_COLORCHECKER_AUTO_ORIENT", "on"
    ).strip().lower()
    if orient_switch not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(
            "FFAC_COLORCHECKER_AUTO_ORIENT must be on or off; "
            f"got {orient_switch!r}"
        )
    auto_orient = orient_switch in {"1", "true", "yes", "on"}
    strategy = os.environ.get(
        "FFAC_COLORCHECKER_STRATEGY", "neutral-tone-affine"
    ).strip().lower()
    if strategy != "neutral-tone-affine":
        raise ValueError(
            "FFAC_COLORCHECKER_STRATEGY must be neutral-tone-affine when "
            f"FFAC_COLORCHECKER_REFINEMENT is active; got {strategy!r}"
        )

    settings: Dict[str, Any] = {
        "residual_refinement_steps": int(
            os.environ.get("FFAC_COLORCHECKER_REFINEMENT_STEPS", "2")
        ),
        "residual_refinement_ridge": float(
            os.environ.get("FFAC_COLORCHECKER_REFINEMENT_RIDGE", "0.01")
        ),
        "residual_refinement_huber_delta": float(
            os.environ.get("FFAC_COLORCHECKER_REFINEMENT_HUBER_DELTA", "0.025")
        ),
        "residual_refinement_min_improvement": float(
            os.environ.get(
                "FFAC_COLORCHECKER_REFINEMENT_MIN_IMPROVEMENT", "0.005"
            )
        ),
        "residual_refinement_max_condition": float(
            os.environ.get("FFAC_COLORCHECKER_REFINEMENT_MAX_CONDITION", "4.0")
        ),
        "residual_refinement_diagonal_min": float(
            os.environ.get("FFAC_COLORCHECKER_REFINEMENT_DIAGONAL_MIN", "0.75")
        ),
        "residual_refinement_diagonal_max": float(
            os.environ.get("FFAC_COLORCHECKER_REFINEMENT_DIAGONAL_MAX", "1.35")
        ),
        "residual_refinement_off_diagonal_max_abs": float(
            os.environ.get(
                "FFAC_COLORCHECKER_REFINEMENT_OFF_DIAGONAL_MAX_ABS", "0.20"
            )
        ),
        "residual_refinement_translation_max_abs": float(
            os.environ.get(
                "FFAC_COLORCHECKER_REFINEMENT_TRANSLATION_MAX_ABS", "0.06"
            )
        ),
        "residual_warn_threshold": float(
            os.environ.get("FFAC_COLORCHECKER_RESIDUAL_WARN_THRESHOLD", "0.015")
        ),
        "btb_swatch_weight": float(
            os.environ.get("FFAC_COLORCHECKER_BTB_SWATCH_WEIGHT", "1.0")
        ),
        "max_swatch_loss_weight": float(
            os.environ.get("FFAC_COLORCHECKER_MAX_SWATCH_LOSS_WEIGHT", "0.0")
        ),
        "max_swatch_warn_threshold": float(
            os.environ.get(
                "FFAC_COLORCHECKER_MAX_SWATCH_WARN_THRESHOLD", "1.0"
            )
        ),
    }
    algorithm_revision = os.environ.get(
        "FFAC_COLORCHECKER_ALGORITHM_REVISION", "legacy-v2"
    ).strip().lower()
    if algorithm_revision not in colorchecker_refinement.ALGORITHM_REVISIONS:
        raise ValueError(f"Unsupported ColorChecker algorithm revision: {algorithm_revision!r}")
    if not 1 <= settings["residual_refinement_steps"] <= 4:
        raise ValueError("ColorChecker refinement steps must be between 1 and 4")
    numeric_values = [float(value) for value in settings.values()]
    if not all(np.isfinite(value) for value in numeric_values):
        raise ValueError("ColorChecker refinement settings must be finite")
    if settings["residual_refinement_ridge"] < 0.0:
        raise ValueError("ColorChecker refinement ridge must be non-negative")
    if settings["btb_swatch_weight"] <= 0.0:
        raise ValueError("ColorChecker BTB swatch weight must be positive")
    if settings["max_swatch_loss_weight"] < 0.0:
        raise ValueError("ColorChecker maximum-swatch loss weight must be non-negative")
    if settings["max_swatch_warn_threshold"] <= 0.0:
        raise ValueError("ColorChecker maximum-swatch threshold must be positive")
    if settings["residual_warn_threshold"] <= 0.0:
        raise ValueError("ColorChecker mean residual threshold must be positive")
    if settings["residual_refinement_diagonal_min"] <= 0.0:
        raise ValueError("ColorChecker refinement diagonal minimum must be positive")
    if (
        settings["residual_refinement_diagonal_max"]
        < settings["residual_refinement_diagonal_min"]
    ):
        raise ValueError("ColorChecker refinement diagonal bounds are reversed")

    configured: list[dict[str, Any]] = []
    shape_corrected_baseline = getattr(
        fluidflower, "shape_corrected_baseline", None
    )
    for index, correction in enumerate(
        getattr(fluidflower, "color_corrections", None) or []
    ):
        if not all(
            hasattr(correction, name)
            for name in ("correct_array", "_restrict_to_roi", "roi", "colorchecker")
        ):
            continue
        if shape_corrected_baseline is None:
            raise RuntimeError(
                "ColorChecker refinement requires the shape-corrected baseline"
            )
        details = colorchecker_refinement.install(
            correction,
            baseline=shape_corrected_baseline,
            auto_orientation=auto_orient,
            settings=settings,
            algorithm_revision=algorithm_revision,
        )
        configured.append({"index": int(index), **details})
    if not configured:
        raise RuntimeError(
            "ColorChecker refinement requested, but no compatible active "
            "ColorCorrection was loaded"
        )
    refined_baseline = fluidflower.shape_corrected_baseline.copy()
    for correction in fluidflower.color_corrections:
        refined_baseline = correction(refined_baseline)
    fluidflower.baseline = refined_baseline
    return {
        "active": True,
        "auto_orient": auto_orient,
        "strategy": "neutral-tone-only" if algorithm_revision == "neutral-tone-only-v4" else strategy,
        "correction_count": len(configured),
        "baseline_rebuilt_with_active_pipeline": True,
        "algorithm_revision": algorithm_revision,
        "corrections": configured,
    }


def build_context(run, config_dir, rig_cls, ref_config_path=None, use_facies=True,
                  bounds_map=None, enforce_lower=False, per_label_params=True,
                  use_label_weights=False, label_weights=None, quality_scale=1.0,
                  quality_dtype=None, objective_integral="off",
                  static_light_correction=None, signal_parameterization=None,
                  phase_separation=None,
                  color_path_anchor=None,
                  color_path_anchor_weight=None,
                  color_path_anchor_strict=None,
                  aqueous_noise_gate=None,
                  aqueous_noise_quantile_low=None,
                  aqueous_noise_quantile_high=None,
                  late_gas_start_h=None,
                  late_gas_max_fraction=None,
                  late_gas_penalty_weight=None,
                  late_gas_decay_half_life_h=None,
                  late_gas_floor_fraction=None,
                  late_gas_constraint_mode=None,
                   overflow_phase_targets=None,
                   overflow_phase_penalty_weight=None,
                   aqueous_diffusion_penalty_weight=None,
                   aqueous_diffusion_start_h=None,
                   aqueous_diffusion_support_level=None,
                   aqueous_diffusion_pair_min_dt_h=None,
                   post_injection_balance_weight=None,
                   post_injection_balance_start_h=None,
                   depth_parameterization=None,
                  depth_min_m=None,
                  depth_max_m=None,
                  depth_prior_weight=None,
                  depth_edge_prior_sigma_m=None,
                  depth_bulge_prior_sigma_m=None,
                  extra_calibration_times_hours=None,
                  extra_calibration_time_tolerance_seconds=None,
                  evaluation_times_hours=None,
                  evaluation_time_tolerance_seconds=600.0,
                  allow_partial_evaluation_times=False,
                  overflow_post_coverage_penalty_weight=None,
                  overflow_post_coverage_sigma_multiplier=None,
                  overflow_post_coverage_fraction_tolerance=None) -> CalibrationContext:
    import numpy as np
    signal_parameterization = _normalise_signal_parameterization(signal_parameterization)
    phase_separation = _normalise_phase_separation(phase_separation)
    aqueous_noise_gate_mode = normalise_aqueous_noise_gate_mode(
        aqueous_noise_gate
        if aqueous_noise_gate is not None
        else os.environ.get("FFAC_AQUEOUS_NOISE_GATE")
    )
    noise_quantile_low = float(
        aqueous_noise_quantile_low
        if aqueous_noise_quantile_low is not None
        else os.environ.get("FFAC_AQUEOUS_NOISE_QUANTILE_LOW", "0.99")
    )
    noise_quantile_high = float(
        aqueous_noise_quantile_high
        if aqueous_noise_quantile_high is not None
        else os.environ.get("FFAC_AQUEOUS_NOISE_QUANTILE_HIGH", "0.999")
    )
    spatial_support_enabled = os.environ.get(
        "FFAC_SPATIAL_OBSERVATION_SUPPORT", "off"
    ).strip().lower() in {"1", "true", "yes", "on"}
    color_path_evidence_mode = normalise_color_path_evidence_mode(
        os.environ.get("FFAC_COLOR_PATH_EVIDENCE", "off")
    )
    try:
        resolved_spatial_objective_weight = float(
            os.environ.get(
                "FFAC_SPATIAL_OBJECTIVE_WEIGHT",
                str(DEFAULT_SPATIAL_OBJECTIVE_WEIGHT),
            )
        )
        resolved_spatial_recall_weight = float(
            os.environ.get(
                "FFAC_SPATIAL_RECALL_WEIGHT",
                str(DEFAULT_SPATIAL_RECALL_WEIGHT),
            )
        )
        resolved_spatial_seal_weight = float(
            os.environ.get(
                "FFAC_SPATIAL_SEAL_WEIGHT",
                str(DEFAULT_SPATIAL_SEAL_WEIGHT),
            )
        )
        resolved_spatial_trace_level = float(
            os.environ.get(
                "FFAC_SPATIAL_TRACE_LEVEL",
                str(DEFAULT_SPATIAL_TRACE_LEVEL),
            )
        )
        resolved_spatial_core_level = float(
            os.environ.get(
                "FFAC_SPATIAL_CORE_LEVEL",
                str(DEFAULT_SPATIAL_CORE_LEVEL),
            )
        )
        resolved_gain_regularization_weight = float(
            os.environ.get(
                "FFAC_LAYER_GAIN_REGULARIZATION_WEIGHT",
                str(DEFAULT_LAYER_GAIN_REGULARIZATION_WEIGHT),
            )
        )
        resolved_gain_regularization_sigma = float(
            os.environ.get(
                "FFAC_LAYER_GAIN_REGULARIZATION_SIGMA",
                str(DEFAULT_LAYER_GAIN_REGULARIZATION_SIGMA),
            )
        )
        color_path_residual_low = float(
            os.environ.get("FFAC_COLOR_PATH_RESIDUAL_LOW", "0.15")
        )
        color_path_residual_high = float(
            os.environ.get("FFAC_COLOR_PATH_RESIDUAL_HIGH", "0.45")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Spatial objective settings must be numeric") from exc
    if (
        min(
            resolved_spatial_objective_weight,
            resolved_spatial_recall_weight,
            resolved_spatial_seal_weight,
            resolved_gain_regularization_weight,
        )
        < 0.0
        or not 0.0 < resolved_spatial_trace_level <= resolved_spatial_core_level <= 1.0
        or resolved_gain_regularization_sigma <= 0.0
        or not 0.0 <= color_path_residual_low < color_path_residual_high
    ):
        raise ValueError("Invalid spatial objective or layer-gain regularization settings")
    if resolved_spatial_objective_weight > 0.0 and not spatial_support_enabled:
        raise ValueError(
            "FFAC_SPATIAL_OBJECTIVE_WEIGHT requires "
            "FFAC_SPATIAL_OBSERVATION_SUPPORT=on"
        )
    (
        resolved_late_gas_start_h,
        resolved_late_gas_max_fraction,
        resolved_late_gas_penalty_weight,
    ) = resolve_late_gas_constraint(
        late_gas_start_h,
        late_gas_max_fraction,
        late_gas_penalty_weight,
    )
    (
        resolved_late_gas_decay_half_life_h,
        resolved_late_gas_floor_fraction,
    ) = resolve_late_gas_decay(
        late_gas_decay_half_life_h,
        late_gas_floor_fraction,
    )
    resolved_late_gas_constraint_mode = resolve_late_gas_constraint_mode(
        late_gas_constraint_mode
    )
    (
        resolved_aqueous_diffusion_penalty_weight,
        resolved_aqueous_diffusion_start_h,
        resolved_aqueous_diffusion_support_level,
        resolved_aqueous_diffusion_pair_min_dt_h,
    ) = resolve_aqueous_diffusion_objective(
        aqueous_diffusion_penalty_weight,
        aqueous_diffusion_start_h,
        aqueous_diffusion_support_level,
        aqueous_diffusion_pair_min_dt_h,
    )
    try:
        resolved_post_injection_balance_weight = float(
            post_injection_balance_weight
            if post_injection_balance_weight is not None
            else os.environ.get(
                "FFAC_POST_INJECTION_BALANCE_WEIGHT",
                str(DEFAULT_POST_INJECTION_BALANCE_WEIGHT),
            )
        )
        resolved_post_injection_balance_start_h = float(
            post_injection_balance_start_h
            if post_injection_balance_start_h is not None
            else os.environ.get(
                "FFAC_POST_INJECTION_BALANCE_START_H",
                str(DEFAULT_POST_INJECTION_BALANCE_START_H),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Post-injection balance settings must be numeric") from exc
    if (
        not np.isfinite(resolved_post_injection_balance_weight)
        or not np.isfinite(resolved_post_injection_balance_start_h)
        or resolved_post_injection_balance_weight < 0.0
        or resolved_post_injection_balance_start_h < 0.0
    ):
        raise ValueError(
            "Post-injection balance weight and start time must be finite and "
            "non-negative"
        )
    (
        resolved_depth_prior_weight,
        resolved_depth_edge_prior_sigma_m,
        resolved_depth_bulge_prior_sigma_m,
    ) = resolve_depth_prior(
        depth_prior_weight,
        depth_edge_prior_sigma_m,
        depth_bulge_prior_sigma_m,
    )
    (
        resolved_overflow_phase_constraint,
        resolved_overflow_phase_penalty_weight,
    ) = resolve_overflow_phase_constraint(
        run=run,
        path=overflow_phase_targets,
        penalty_weight=overflow_phase_penalty_weight,
    )
    (
        resolved_overflow_post_coverage_penalty_weight,
        resolved_overflow_post_coverage_sigma_multiplier,
        resolved_overflow_post_coverage_fraction_tolerance,
    ) = resolve_overflow_post_coverage_objective(
        overflow_post_coverage_penalty_weight,
        overflow_post_coverage_sigma_multiplier,
        overflow_post_coverage_fraction_tolerance,
        overflow_phase_penalty_weight=resolved_overflow_phase_penalty_weight,
    )
    if resolved_late_gas_penalty_weight > 0.0:
        logger.info(
            "[%s] late-gas objective ACTIVE: t>=%.3fh, gas<=%.3f of "
            "injected mass, penalty weight=%.3f, mode=%s",
            run,
            resolved_late_gas_start_h,
            resolved_late_gas_max_fraction,
            resolved_late_gas_penalty_weight,
            resolved_late_gas_constraint_mode,
        )
        if resolved_late_gas_decay_half_life_h > 0.0:
            logger.info(
                "[%s] late-gas envelope decays with half-life %.3fh to floor %.5f",
                run,
                resolved_late_gas_decay_half_life_h,
                resolved_late_gas_floor_fraction,
            )
    if resolved_overflow_phase_penalty_weight > 0.0:
        if resolved_overflow_phase_constraint is None:
            raise RuntimeError("Active overflow phase objective has no targets")
        logger.info(
            "[%s] overflow phase objective ACTIVE: %d targets, weight=%.3f, %s",
            run,
            len(resolved_overflow_phase_constraint.targets),
            resolved_overflow_phase_penalty_weight,
            resolved_overflow_phase_constraint.path,
        )
    if resolved_overflow_post_coverage_penalty_weight > 0.0:
        if resolved_overflow_phase_constraint is None:
            raise RuntimeError(
                "Overflow post-coverage support requires overflow phase targets"
            )
        logger.info(
            "[%s] overflow post-coverage support ACTIVE: weight=%.3f, "
            "sigma multiplier=%.3f, fraction tolerance=%.4f",
            run,
            resolved_overflow_post_coverage_penalty_weight,
            resolved_overflow_post_coverage_sigma_multiplier,
            resolved_overflow_post_coverage_fraction_tolerance,
        )
    if resolved_aqueous_diffusion_penalty_weight > 0.0:
        logger.info(
            "[%s] aqueous diffusion objective ACTIVE: t>=%.3fh, "
            "support>=%.4f, penalty weight=%.3f",
            run,
            resolved_aqueous_diffusion_start_h,
            resolved_aqueous_diffusion_support_level,
            resolved_aqueous_diffusion_penalty_weight,
        )
    from darsia.presets.workflows.analysis.analysis_context import (
        prepare_analysis_context,
        select_image_paths,
    )

    config_dir = Path(config_dir)
    config_path = config_dir / f"{run}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    common_path = Path(ref_config_path) if ref_config_path else config_dir.parent / "common.toml"
    config_paths = [common_path, config_path] if common_path.exists() else [config_path]

    if not use_facies:
        labels_overlay = config_dir.parent / "labels_basis.toml"
        if not labels_overlay.is_file():
            raise FileNotFoundError(
                "Detailed-layer calibration requested, but the labels-basis "
                f"overlay is missing: {labels_overlay}"
            )
        config_paths.append(labels_overlay)
        logger.info(
            "[%s] detailed layer basis ACTIVE -> overlay %s added",
            run,
            labels_overlay.name,
        )

    # Colour-correction toggle. The prep step writes a per-run stamp recording whether the
    # cached rig + embedding + seed were built WITH colour correction (built on corrected vs
    # uncorrected images - the two states are baked into the cache, see load_corrections).
    # build_context auto-follows the stamp so master and workers stay consistent with the
    # cache without any flag threading. Stamp: <config_dir>/.color_state/<run>.txt = on|off.
    _color_state = "off"
    try:
        _stamp = config_dir / ".color_state" / f"{run}.txt"
        if _stamp.exists():
            _color_state = _stamp.read_text(encoding="utf-8").strip().lower()
    except Exception:  # noqa: BLE001
        _color_state = "off"
    if _color_state == "on":
        _overlay = config_dir.parent / "coloron.toml"
        if _overlay.exists():
            config_paths = config_paths + [_overlay]
            logger.info("[%s] colour-correction ON (stamp) -> overlay %s added",
                        run, _overlay.name)
        else:
            logger.warning("[%s] colour stamp=on but %s missing; running WITHOUT colour",
                           run, _overlay)
    else:
        logger.info("[%s] colour-correction OFF (stamp=%s)", run, _color_state)

    # A colour path learned from images processed by a different optical
    # pipeline is not interchangeable with the active image correction. Pilot
    # campaigns can point at an isolated calibration embedding without
    # overwriting the established run-native artefacts in Results.
    _color_path_overlay_raw = os.environ.get(
        "FFAC_COLOR_PATH_CONFIG_OVERLAY", ""
    ).strip()
    if _color_path_overlay_raw:
        _color_path_overlay = _expand_run_path_template(
            _color_path_overlay_raw, run
        )
        if not _color_path_overlay.is_file():
            raise FileNotFoundError(
                "Colour-path config overlay not found: "
                f"{_color_path_overlay}"
            )
        config_paths.append(_color_path_overlay)
        logger.info(
            "[%s] isolated colour-path embedding ACTIVE -> overlay %s added",
            run,
            _color_path_overlay,
        )

    # Host-specific path overlays keep the scientific configuration identical
    # while translating storage paths (for example Windows UNC/Z: paths to a
    # mounted SMB volume on macOS). The overlay is deliberately applied last so
    # it can replace data/results and protocol mapping keys without duplicating
    # the full run configuration.
    _config_overlay_raw = os.environ.get("FFAC_CONFIG_OVERLAY", "").strip()
    if _config_overlay_raw:
        _config_overlay = _expand_run_path_template(_config_overlay_raw, run)
        if not _config_overlay.is_file():
            raise FileNotFoundError(
                f"Host config overlay not found: {_config_overlay}"
            )
        config_paths.append(_config_overlay)
        logger.info(
            "[%s] host config overlay ACTIVE -> %s added",
            run,
            _config_overlay,
        )

    ctx_raw = prepare_analysis_context(
        cls=rig_cls, path=config_paths, all=False, require_color_to_mass=True,
    )
    cta = ctx_raw.color_to_mass_analysis
    if cta is None:
        raise RuntimeError(f"[{run}] color_to_mass_analysis not initialised; seed/calibrate first.")
    fluidflower = ctx_raw.fluidflower
    experiment = ctx_raw.experiment
    color_correction_provenance = _configure_runtime_colorchecker_refinement(
        fluidflower
    )
    if color_correction_provenance["active"]:
        # prepare_analysis_context constructed the relative-colour analysis
        # before the opt-in runtime correction was installed. Keep its reference
        # and the thermodynamic geometry carrier aligned with the rebuilt rig
        # baseline; model/path parameters themselves remain unchanged.
        cta.color_analysis.base = fluidflower.baseline
        cta.co2_mass_analysis.baseline = fluidflower.baseline
        logger.info(
            "[%s] guarded ColorChecker residual refinement ACTIVE: %s",
            run,
            color_correction_provenance,
        )
    geometry = getattr(fluidflower, "geometry", None)
    config = getattr(ctx_raw, "config", None)
    validate_depth_map(geometry, run)

    # Sanitise the geometry integration weights: the porosity map carries NaN
    # OUTSIDE the sand domain (~26%% of pixels), and geometry.integrate weights by
    # voxel_volume(=area*depth*porosity) with a plain np.sum -> any NaN weight makes
    # the whole integral NaN. Those regions have no pore space, so set them to 0.
    for _attr in ("voxel_volume", "cached_voxel_volume"):
        _v = getattr(geometry, _attr, None)
        if isinstance(_v, np.ndarray):
            setattr(geometry, _attr,
                    np.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0))

    # active labels + reference + calibration folder (resolved ColorPathEmbedding).
    embedding = None
    try:
        embedding = config.calibration.mass.color
    except Exception:
        embedding = None
    if embedding is None:
        # fall back to the analysis mass embedding (same [color.path.*]) used to
        # build the color-to-mass pipeline.
        embedding = config.color.resolve(config.analysis.mass.color)
    basis_value = str(getattr(getattr(embedding, "basis", None), "value", getattr(embedding, "basis", ""))).lower()
    expected_basis = "facies" if use_facies else "labels"
    if basis_value != expected_basis:
        raise RuntimeError(
            f"[{run}] calibration basis mismatch: requested {expected_basis}, "
            f"resolved {basis_value or 'unknown'}"
        )
    ignore = set(getattr(embedding, "ignore_labels", []) or [])
    reference_label = int(getattr(embedding, "reference_label", 0))
    cp_folder = Path(embedding.color_paths_folder)
    calibration_folder = cp_folder.parent.parent / "color_to_mass" / cp_folder.name

    # --- Titration-anchored aqueous transfer (opt-in) ---
    # Replace the cached SimpleFlash's LINEAR c_aq(signal) ramp with the physically-derived
    # BTB/carbonate titration curve (TitrationFlash). Done here (not in the rig) so it needs
    # NO rig rebuild and applies in master + workers alike.
    #
    # TWO equivalent triggers (either turns it on):
    #   1. STAMP (recommended for the distributed queue): a per-run file
    #      <config_dir>/.titration_state/<run>.txt containing "on" (mirrors the colour stamp).
    #      Master + every worker call build_context and auto-follow the stamp, so there is NO
    #      env threading and no risk of mixed workers. Optional recipe on the same/next line:
    #      "on 1.25,0.726,34" (alk_mM,btb_mM,co2sat_mM).
    #   2. ENV (handy for the standalone script in one shell): FFAC_TITRATION_FLASH=on, recipe
    #      override FFAC_TITRATION_RECIPE="alk_mM,btb_mM,co2sat_mM".
    _titration_switch = os.environ.get("FFAC_TITRATION_FLASH", "").strip().lower()
    _titration_explicit = _titration_switch not in {"", "auto", "default"}
    if _titration_explicit and _titration_switch not in {
        "0", "1", "false", "true", "no", "yes", "off", "on"
    }:
        raise ValueError(
            "FFAC_TITRATION_FLASH must be on, off, or unset; got "
            f"{_titration_switch!r}"
        )
    _titr_on = _titration_switch in ("1", "true", "yes", "on")
    _titr_recipe = os.environ.get("FFAC_TITRATION_RECIPE", "").strip()
    try:
        _tstamp = config_dir / ".titration_state" / f"{run}.txt"
        if _tstamp.exists() and not _titration_explicit:
            _toks = _tstamp.read_text(encoding="utf-8").split()
            if _toks and _toks[0].strip().lower() in ("1", "true", "yes", "on"):
                _titr_on = True
                if len(_toks) > 1 and "," in _toks[1]:
                    _titr_recipe = _titr_recipe or _toks[1].strip()
    except Exception:  # noqa: BLE001
        pass
    if _titr_on:
        try:
            _flash = getattr(cta, "flash", None)
            if _flash is not None and not isinstance(_flash, TitrationFlash):
                _kw = {}
                if _titr_recipe:
                    _alk, _btb, _sat = (float(x) for x in _titr_recipe.split(","))
                    _kw = dict(alkalinity_M=_alk * 1e-3, btb_M=_btb * 1e-3, co2_sat_M=_sat * 1e-3)
                cta.flash = TitrationFlash.from_simple(_flash, **_kw)
                logger.info("[%s] TitrationFlash ACTIVE (aqueous branch = BTB titration curve) %s",
                            run, _kw or "(default recipe 1.25 mM alk, 0.726 mM BTB, 34 mM sat)")
        except Exception as _exc:  # noqa: BLE001
            logger.warning("[%s] TitrationFlash requested but injection failed: %s", run, _exc)
    elif _titration_explicit:
        try:
            import darsia

            _flash = getattr(cta, "flash", None)
            _flash_model = (
                str((_flash.to_dict() or {}).get("model", "simple")).lower()
                if _flash is not None and hasattr(_flash, "to_dict")
                else "simple"
            )
            if _flash is not None and _flash_model == "titration":
                cta.flash = darsia.SimpleFlash(
                    min_value_aq=float(_flash.min_value_aq),
                    max_value_aq=float(_flash.max_value_aq),
                    min_value_g=float(_flash.min_value_g),
                    max_value_g=float(_flash.max_value_g),
                    restoration=getattr(_flash, "restoration", None),
                )
            logger.info(
                "[%s] SimpleFlash explicitly ACTIVE; titration stamp ignored", run
            )
        except Exception as _exc:  # noqa: BLE001
            raise RuntimeError(
                f"[{run}] failed to activate explicit SimpleFlash"
            ) from _exc

    hetero = cta.signal_model.model[1]
    all_labels = [int(x) for x in (hetero.keys() if hasattr(hetero, "keys") else [])]
    if not all_labels:
        labels_img = getattr(getattr(fluidflower, "labels", None), "img", None)
        if labels_img is not None:
            all_labels = [int(x) for x in np.unique(labels_img) if x >= 0]
    # Facies label 0 is water, but detailed segmentation label 0 is a physical
    # F3 sand layer in FF-AC. The labels-basis overlay explicitly ignores the
    # water labels (1 and 2), so label 0 must remain active in detailed mode.
    signal_labels = [
        label
        for label in all_labels
        if label not in ignore and (not use_facies or label != 0)
    ]
    signal_label_to_facies = _infer_signal_label_to_facies(
        fluidflower,
        signal_labels,
        use_facies=use_facies,
    )
    setattr(
        cta,
        "_ffac_signal_label_to_facies",
        dict(signal_label_to_facies),
    )
    color_path_source_root = os.environ.get("FFAC_COLOR_PATH_SOURCE_ROOT", "")
    color_path_source_strict = os.environ.get(
        "FFAC_COLOR_PATH_SOURCE_STRICT", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    color_path_source = _restore_color_paths(
        run=run,
        calibration=cta,
        labels=signal_labels,
        source_root=color_path_source_root,
        strict=color_path_source_strict,
    )
    _regularize_color_paths(
        run=run,
        config=config,
        calibration=cta,
        calibration_folder=calibration_folder,
        labels=signal_labels,
        anchor_run=color_path_anchor,
        anchor_weight=color_path_anchor_weight,
        anchor_strict=color_path_anchor_strict,
        source_root=color_path_source_root,
    )

    light_master = os.environ.get("FFAC_MASTER_LIGHT_CONTEXT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    anchor_photometric_gain = _setup_anchor_photometric_gain(
        run=run,
        config=config,
        fluidflower=fluidflower,
        active_labels=signal_labels,
        master_light_context=light_master,
    )
    if anchor_photometric_gain is not None:
        color_base = getattr(getattr(cta, "color_analysis", None), "base", None)
        if color_base is not None:
            _apply_anchor_photometric_gain(color_base, anchor_photometric_gain)
    if light_master:
        template_registration, template_registration_mode = None, "off"
    else:
        template_registration, template_registration_mode = _setup_template_registration(
            run=run,
            config=config,
            fluidflower=fluidflower,
            calibration=cta,
        )

    # Calibration image selection belongs to [calibration.mass], independently
    # of the image schedule used by subsequent analyses. Historically this used
    # ctx_raw.image_paths ([analysis].data), which made additions to the declared
    # mass-calibration schedule ineffective.
    selected_image_paths = list(ctx_raw.image_paths)
    calibration_mass_config = getattr(
        getattr(config, "calibration", None),
        "mass",
        None,
    )
    if calibration_mass_config is not None:
        mass_data = getattr(calibration_mass_config, "data", None)
        if hasattr(mass_data, "get_times_with_uncertainty"):
            selected_image_paths = list(
                experiment.find_images_for_paths(
                    paths=list(getattr(mass_data, "image_paths", []) or [])
                )
            )
            for requested_time, tolerance_hours in (
                mass_data.get_times_with_uncertainty()
            ):
                path = experiment.find_images_for_times(
                    float(requested_time),
                    tol=float(tolerance_hours) * 3600.0,
                )
                if path is None:
                    logger.warning(
                        "[%s] no calibration image within %.1f min of %.3f h; "
                        "dropping this unavailable point",
                        run,
                        float(tolerance_hours) * 60.0,
                        float(requested_time),
                    )
                    continue
                path = Path(path)
                if path in selected_image_paths:
                    logger.warning(
                        "[%s] calibration time %.3f h resolves to duplicate %s; "
                        "keeping the image only once",
                        run,
                        float(requested_time),
                        path.name,
                    )
                    continue
                selected_image_paths.append(path)
            selected_image_paths.sort(key=experiment.get_datetime)
        else:
            selected_image_paths = select_image_paths(
                config,
                experiment,
                all=False,
                sub_config=calibration_mass_config,
                data_registry=config.data.registry,
            )
    (
        extra_calibration_times,
        extra_calibration_tolerance_seconds,
    ) = resolve_extra_calibration_times(
        extra_calibration_times_hours,
        extra_calibration_time_tolerance_seconds,
    )
    for requested_time in extra_calibration_times:
        path = experiment.find_images_for_times(
            requested_time,
            tol=extra_calibration_tolerance_seconds,
        )
        if path is None:
            raise ValueError(
                f"[{run}] no image within "
                f"{extra_calibration_tolerance_seconds:g} s of required extra "
                f"calibration time {requested_time:g} h"
            )
        path = Path(path)
        if path not in selected_image_paths:
            selected_image_paths.append(path)
            logger.info(
                "[%s] added campaign calibration frame %s near %.3f h",
                run,
                path.name,
                requested_time,
            )
    selected_image_paths.sort(key=experiment.get_datetime)
    if not selected_image_paths:
        raise ValueError(f"[{run}] no mass-calibration images were resolved")
    training_image_paths = list(selected_image_paths)
    evaluation_time_records: List[Dict[str, Any]] = []
    if evaluation_times_hours is not None:
        selected_image_paths = []
        selected_path_keys: set[str] = set()
        training_paths = {
            str(Path(path)).lower() for path in training_image_paths
        }
        for requested_time in evaluation_times_hours:
            path = experiment.find_images_for_times(
                float(requested_time),
                tol=float(evaluation_time_tolerance_seconds),
            )
            if path is None:
                if allow_partial_evaluation_times:
                    logger.warning(
                        "[%s] no image within %.0f s of holdout time %.3f h; "
                        "dropping this unavailable holdout point",
                        run,
                        float(evaluation_time_tolerance_seconds),
                        float(requested_time),
                    )
                    evaluation_time_records.append(
                        {
                            "requested_time_h": float(requested_time),
                            "selected": None,
                            "status": "unavailable-dropped",
                        }
                    )
                    continue
                raise ValueError(
                    f"[{run}] no image within {evaluation_time_tolerance_seconds:g} s "
                    f"of holdout time {float(requested_time):g} h"
                )
            path = Path(path)
            path_key = str(path).lower()
            if allow_partial_evaluation_times and path_key in selected_path_keys:
                logger.warning(
                    "[%s] holdout time %.3f h resolves to duplicate %s; dropping it",
                    run,
                    float(requested_time),
                    path.name,
                )
                evaluation_time_records.append(
                    {
                        "requested_time_h": float(requested_time),
                        "selected": str(path),
                        "status": "duplicate-dropped",
                    }
                )
                continue
            if allow_partial_evaluation_times and path_key in training_paths:
                logger.warning(
                    "[%s] holdout time %.3f h overlaps training frame %s; dropping it",
                    run,
                    float(requested_time),
                    path.name,
                )
                evaluation_time_records.append(
                    {
                        "requested_time_h": float(requested_time),
                        "selected": str(path),
                        "status": "training-overlap-dropped",
                    }
                )
                continue
            selected_image_paths.append(path)
            selected_path_keys.add(path_key)
            evaluation_time_records.append(
                {
                    "requested_time_h": float(requested_time),
                    "selected": str(path),
                    "status": "selected",
                }
            )
        if allow_partial_evaluation_times and not selected_image_paths:
            raise ValueError(f"[{run}] no usable non-training holdout images were resolved")
        if not allow_partial_evaluation_times and len(set(selected_image_paths)) != len(selected_image_paths):
            raise ValueError(f"[{run}] holdout times resolved to duplicate image paths")
        overlap = [
            path
            for path in selected_image_paths
            if str(Path(path)).lower() in training_paths
        ]
        if overlap and not allow_partial_evaluation_times:
            raise ValueError(
                f"[{run}] holdout selection overlaps calibration frame(s): "
                + ", ".join(path.name for path in overlap)
            )

    # preload corrected calibration/holdout images + param-independent injected mass
    loaded: List[Tuple[Any, float, float]] = []
    exp_start = getattr(experiment, "experiment_start", None)

    # Nearest-good-neighbour substitution. read_image applies the rig corrections incl.
    # ColorCorrection, which sets last_flagged when a frame's lighting/checker is too far
    # gone to recover. For calibration we need sparse, well-spread points, not these exact
    # frames, so a flagged frame is replaced by the nearest-in-time correctable neighbour
    # (frames are ~5 min apart and the CO2 state barely changes over a few minutes). If no
    # neighbour within the window is usable, the calibration point is dropped. When colour
    # correction is OFF, color_corrections is empty -> _flagged is always False -> this
    # behaves exactly as a plain read (no neighbour I/O, current behaviour unchanged).
    corrected_image_cache_stats: Dict[str, Any] = {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "load_seconds": 0.0,
        "build_seconds": 0.0,
    }

    def _read_and_flag(path):
        cache_started = time.perf_counter()
        cached = _load_corrected_image_cache(
            Path(path),
            quality_dtype=quality_dtype,
        )
        corrected_image_cache_stats["load_seconds"] += (
            time.perf_counter() - cache_started
        )
        if cached is not None:
            corrected_image_cache_stats["hits"] += 1
            return cached
        corrected_image_cache_stats["misses"] += 1
        build_started = time.perf_counter()
        try:
            try:
                im = fluidflower.read_image(
                    Path(path),
                    dtype=np.dtype(quality_dtype).type if quality_dtype else float,
                )
            except TypeError:
                # Compatibility with older DarSIA checkouts. The explicit cast
                # below still enforces the backend contract.
                im = fluidflower.read_image(Path(path))
        except (OSError, RuntimeError, ValueError) as exc:
            diagnostics = {
                "color_correction_active": bool(
                    getattr(fluidflower, "color_corrections", None)
                ),
                "color_correction_flagged": True,
                "color_corrections": [],
                "read_error_type": type(exc).__name__,
                "read_error": str(exc),
            }
            logger.warning(
                "[%s] calibration frame %s could not be corrected/read and is "
                "flagged: %s: %s",
                run,
                Path(path).name,
                type(exc).__name__,
                exc,
            )
            return None, True, diagnostics
        correction_rows: List[Dict[str, Any]] = []
        for index, correction in enumerate(
            getattr(fluidflower, "color_corrections", None) or []
        ):
            if not getattr(correction, "active", False):
                continue
            residual = getattr(correction, "last_residual", None)
            threshold = getattr(correction, "residual_warn_threshold", None)
            row: Dict[str, Any] = {
                "index": int(index),
                "type": type(correction).__name__,
                "flagged": bool(getattr(correction, "last_flagged", False)),
            }
            if residual is not None and np.isfinite(float(residual)):
                row["residual"] = float(residual)
            if threshold is not None and np.isfinite(float(threshold)):
                row["threshold"] = float(threshold)
            residual_before = getattr(
                correction, "last_residual_before_refinement", None
            )
            if residual_before is not None and np.isfinite(float(residual_before)):
                row["residual_before_refinement"] = float(residual_before)
            residual_before_correction = getattr(
                correction, "last_residual_before_correction", None
            )
            if residual_before_correction is not None and np.isfinite(
                float(residual_before_correction)
            ):
                row["residual_before_correction"] = float(
                    residual_before_correction
                )
            max_swatch_residual = getattr(
                correction, "last_max_swatch_residual", None
            )
            if max_swatch_residual is not None and np.isfinite(
                float(max_swatch_residual)
            ):
                row["max_swatch_residual"] = float(max_swatch_residual)
            max_swatch_threshold = getattr(
                correction, "max_swatch_warn_threshold", None
            )
            if max_swatch_threshold is not None and np.isfinite(
                float(max_swatch_threshold)
            ):
                row["max_swatch_threshold"] = float(max_swatch_threshold)
            for attribute, key in (
                ("last_weighted_residual", "weighted_residual"),
                ("last_p95_swatch_residual", "p95_swatch_residual"),
                ("last_residual_score", "residual_score"),
            ):
                value = getattr(correction, attribute, None)
                if value is not None and np.isfinite(float(value)):
                    row[key] = float(value)
            worst_swatch_name = getattr(
                correction, "last_worst_swatch_name", None
            )
            if worst_swatch_name:
                row["worst_swatch_name"] = str(worst_swatch_name)
            flag_reasons = getattr(correction, "last_flag_reasons", None)
            if isinstance(flag_reasons, (list, tuple)):
                row["flag_reasons"] = [str(reason) for reason in flag_reasons]
            row["refinement_steps_applied"] = int(
                getattr(correction, "last_refinement_steps_applied", 0) or 0
            )
            row["correction_strategy"] = str(
                getattr(correction, "correction_strategy", "legacy-affine")
            )
            strategy_diagnostics = getattr(
                correction, "last_strategy_diagnostics", None
            )
            if isinstance(strategy_diagnostics, dict):
                row["strategy_diagnostics"] = dict(strategy_diagnostics)
            refinement_diagnostics = getattr(
                correction, "last_refinement_diagnostics", None
            )
            if isinstance(refinement_diagnostics, list):
                row["refinement_diagnostics"] = [
                    dict(item)
                    for item in refinement_diagnostics
                    if isinstance(item, dict)
                ]
            swatch_residuals = np.asarray(
                getattr(correction, "last_swatch_residuals", []),
                dtype=np.float64,
            ).reshape(-1)
            if swatch_residuals.size and np.all(np.isfinite(swatch_residuals)):
                row["swatch_residuals"] = [
                    float(value) for value in swatch_residuals
                ]
            correction_rows.append(row)
        finite_residuals = [
            float(row["residual"])
            for row in correction_rows
            if "residual" in row
        ]
        finite_weighted_residuals = [
            float(row["weighted_residual"])
            for row in correction_rows
            if "weighted_residual" in row
        ]
        finite_max_swatch_residuals = [
            float(row["max_swatch_residual"])
            for row in correction_rows
            if "max_swatch_residual" in row
        ]
        if template_registration is not None:
            im = _warp_registration_image(
                im,
                template_registration,
                mode=template_registration_mode,
            )
        if quality_dtype:
            im.img = np.asarray(im.img, dtype=np.dtype(quality_dtype))
        flagged = any(bool(row["flagged"]) for row in correction_rows)
        diagnostics: Dict[str, Any] = {
            "color_correction_active": bool(correction_rows),
            "color_correction_flagged": bool(flagged),
            "color_corrections": correction_rows,
        }
        if finite_residuals:
            diagnostics["color_correction_residual"] = max(finite_residuals)
        if finite_weighted_residuals:
            diagnostics["color_correction_weighted_residual"] = max(
                finite_weighted_residuals
            )
        if finite_max_swatch_residuals:
            diagnostics["color_correction_max_swatch_residual"] = max(
                finite_max_swatch_residuals
            )
        diagnostics["corrected_image_cache"] = "miss"
        stored = _store_corrected_image_cache(
            Path(path),
            quality_dtype=quality_dtype,
            image=im,
            flagged=flagged,
            diagnostics=diagnostics,
        )
        if stored is not None:
            corrected_image_cache_stats["writes"] += 1
        corrected_image_cache_stats["build_seconds"] += (
            time.perf_counter() - build_started
        )
        return im, flagged, diagnostics

    frame_selection_policy = os.environ.get(
        "FFAC_FLAGGED_FRAME_POLICY", "nearest-good"
    ).strip().lower().replace("_", "-")
    if frame_selection_policy in {"keep", "exact"}:
        frame_selection_policy = "exact"
    elif frame_selection_policy in {"drop", "reject"}:
        frame_selection_policy = "drop"
    elif frame_selection_policy in {"nearest", "nearest-good", "substitute"}:
        frame_selection_policy = "nearest-good"
    else:
        raise ValueError(
            "FFAC_FLAGGED_FRAME_POLICY must be exact, drop, or nearest-good; got "
            f"{frame_selection_policy!r}"
        )
    frame_selection_window_frames = int(
        os.environ.get("FFAC_FLAGGED_FRAME_WINDOW_FRAMES", "6")
    )
    frame_selection_max_minutes = float(
        os.environ.get("FFAC_FLAGGED_FRAME_MAX_MINUTES", "30.0")
    )
    if frame_selection_window_frames < 1:
        raise ValueError("Flagged-frame search window must contain at least one frame")
    if (
        not np.isfinite(frame_selection_max_minutes)
        or frame_selection_max_minutes <= 0.0
    ):
        raise ValueError("Flagged-frame maximum time distance must be positive")
    frame_selection_records: List[Dict[str, Any]] = []

    def _quality_key(diagnostics: Dict[str, Any]) -> Tuple[float, float, float]:
        return tuple(
            float(diagnostics.get(key, float("inf")))
            for key in (
                "color_correction_max_swatch_residual",
                "color_correction_weighted_residual",
                "color_correction_residual",
            )
        )

    def _select_frame(path):
        im, flagged, diagnostics = _read_and_flag(path)
        if not flagged:
            frame_selection_records.append(
                {
                    "requested": str(path),
                    "selected": str(path),
                    "status": "exact",
                    **diagnostics,
                }
            )
            return im
        if frame_selection_policy == "exact":
            if im is None:
                raise RuntimeError(
                    f"[{run}] exact calibration frame {Path(path).name} could "
                    f"not be read/corrected: {diagnostics.get('read_error', 'unknown error')}"
                )
            logger.warning(
                "[%s] calibration frame %s flagged but retained by exact-frame policy",
                run,
                Path(path).name,
            )
            frame_selection_records.append(
                {
                    "requested": str(path),
                    "selected": str(path),
                    "status": "flagged-retained",
                    **diagnostics,
                }
            )
            return im
        if frame_selection_policy == "drop":
            frame_selection_records.append(
                {
                    "requested": str(path),
                    "selected": None,
                    "status": "flagged-dropped",
                    **diagnostics,
                }
            )
            return None
        folder = Path(path).parent
        siblings = sorted(folder.glob("*" + Path(path).suffix))
        names = [s.name for s in siblings]
        try:
            idx = names.index(Path(path).name)
        except ValueError:
            if im is None:
                frame_selection_records.append(
                    {
                        "requested": str(path),
                        "selected": None,
                        "status": "flagged-dropped-neighbour-index-missing",
                        **diagnostics,
                    }
                )
                return None
            frame_selection_records.append(
                {
                    "requested": str(path),
                    "selected": str(path),
                    "status": "flagged-retained-neighbour-index-missing",
                    **diagnostics,
                }
            )
            return im  # cannot locate neighbours; keep original (better than dropping)
        base_date = getattr(im, "date", None)
        for d in range(
            1, frame_selection_window_frames + 1
        ):  # expand outward -> nearest first
            cands = []
            for j in (idx - d, idx + d):
                if not (0 <= j < len(siblings)):
                    continue
                c_im, c_flag, c_diagnostics = _read_and_flag(siblings[j])
                if c_flag:
                    continue
                c_date = getattr(c_im, "date", None)
                if base_date is not None and c_date is not None:
                    dt_min = abs((c_date - base_date).total_seconds()) / 60.0
                    if dt_min > frame_selection_max_minutes:
                        continue
                else:
                    dt_min = float(d)
                cands.append(
                    (
                        dt_min,
                        _quality_key(c_diagnostics),
                        siblings[j].name,
                        c_im,
                        c_diagnostics,
                    )
                )
            if cands:
                cands.sort(key=lambda item: (item[0], item[1], item[2]))
                logger.warning(
                    "[%s] calibration frame %s flagged; substituting nearest good "
                    "neighbour %s (%.0f min away)", run, Path(path).name,
                    cands[0][2], cands[0][0])
                frame_selection_records.append(
                    {
                        "requested": str(path),
                        "selected": str(Path(path).parent / cands[0][2]),
                        "status": "flagged-substituted",
                        "delta_minutes": float(cands[0][0]),
                        "requested_diagnostics": diagnostics,
                        **cands[0][4],
                    }
                )
                return cands[0][3]
        frame_selection_records.append(
            {
                "requested": str(path),
                "selected": None,
                "status": "flagged-dropped",
                **diagnostics,
            }
        )
        return None  # no usable neighbour within window -> drop this calibration point

    if light_master:
        logger.info(
            "[%s] FFAC_MASTER_LIGHT_CONTEXT=on -> skipping calibration image preload",
            run,
        )
    else:
        for p in selected_image_paths:
            img = _select_frame(p)
            if img is None:
                logger.warning(
                    "[%s] calibration frame %s uncorrectable and no good neighbour within "
                    "window; dropping this calibration point.", run, Path(p).name)
                continue
            # Older DarSIA versions do not accept an output dtype in read_image;
            # keep this idempotent cast as a compatibility guard.
            if quality_dtype:
                try:
                    img.img = np.asarray(img.img, dtype=np.dtype(quality_dtype))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[%s] quality_dtype=%s cast failed: %s", run, quality_dtype, exc)
            injected = float(experiment.injection_protocol.injected_mass(date=img.date))
            try:
                t_h = (img.date - exp_start).total_seconds() / 3600.0
            except Exception:
                t_h = float(len(loaded))
            loaded.append((img, injected, t_h))
        if anchor_photometric_gain is not None:
            for image, _injected, _t_h in loaded:
                _apply_anchor_photometric_gain(image, anchor_photometric_gain)
        light_reference_rgb = np.asarray(
            cta.color_analysis.base.img, dtype=np.float32
        )[..., :3]
        light_valid_mask = np.isin(
            np.asarray(cta.labels.img), np.asarray(signal_labels, dtype=int)
        )
        _apply_static_light_correction(
            loaded,
            mode=static_light_correction,
            run=run,
            reference_rgb=light_reference_rgb,
            valid_mask=light_valid_mask,
        )
        logger.info("[%s] preloaded %d calibration image(s); active labels=%s",
                    run, len(loaded), signal_labels)

    noise_gate_model: Optional[AqueousNoiseGate] = None
    noise_gate_provenance: Dict[str, Any] = {
        "mode": aqueous_noise_gate_mode,
        "quantile_low": noise_quantile_low,
        "quantile_high": noise_quantile_high,
    }
    spatial_support_model: Optional[SpatialAqueousSupportModel] = None
    spatial_support_provenance: Dict[str, Any] = {
        "enabled": bool(spatial_support_enabled),
        "quantile_low": noise_quantile_low,
        "quantile_high": noise_quantile_high,
    }
    color_path_evidence_model: Optional[ColorPathEvidenceModel] = None
    color_path_evidence_provenance: Dict[str, Any] = {
        "mode": color_path_evidence_mode,
        "enabled": color_path_evidence_mode != "off",
        "quantile_low": noise_quantile_low,
        "quantile_high": noise_quantile_high,
        "residual_low": color_path_residual_low,
        "residual_high": color_path_residual_high,
    }
    reference_support_requested = (
        aqueous_noise_gate_mode != "off"
        or spatial_support_enabled
        or color_path_evidence_mode != "off"
    )
    if reference_support_requested and light_master:
        noise_gate_provenance["status"] = "deferred-to-full-worker-context"
        spatial_support_provenance["status"] = "deferred-to-full-worker-context"
        color_path_evidence_provenance["status"] = (
            "deferred-to-full-worker-context"
        )
        logger.info(
            "[%s] aqueous reference support deferred in light-master context", run
        )
    elif reference_support_requested:
        strict_noise_gate = os.environ.get(
            "FFAC_AQUEOUS_NOISE_GATE_STRICT", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        strict_reference_support = (
            strict_noise_gate
            or spatial_support_enabled
            or color_path_evidence_mode != "off"
        )
        try:
            registry = getattr(getattr(config, "data", None), "registry", None)
            if registry is None:
                raise RuntimeError("the data registry is unavailable")
            registry_keys = set(registry.keys())
            reference_key = (
                "aqueous_reference"
                if "aqueous_reference" in registry_keys
                else "baseline"
            )
            reference_data = registry.resolve(reference_key)
            reference_paths = list(
                dict.fromkeys(
                    Path(path)
                    for path in (
                        getattr(reference_data, "image_paths", None) or []
                    )
                )
            )
            if not reference_paths:
                raise RuntimeError("[data.path.baseline] resolved no images")

            reference_images: List[Any] = []
            used_reference_paths: List[Path] = []
            skipped_flagged_paths: List[str] = []
            for reference_path in reference_paths:
                reference_image, flagged, _diagnostics = _read_and_flag(reference_path)
                if flagged:
                    skipped_flagged_paths.append(str(reference_path))
                    continue
                reference_images.append(reference_image)
                used_reference_paths.append(reference_path)
            if strict_reference_support and len(reference_images) < 2:
                raise RuntimeError(
                    "strict mode requires at least two unflagged no-CO2 images; "
                    f"found {len(reference_images)}"
                )
            if not reference_images:
                raise RuntimeError("all configured no-CO2 images were flagged")
            if anchor_photometric_gain is not None:
                for reference_image in reference_images:
                    _apply_anchor_photometric_gain(
                        reference_image,
                        anchor_photometric_gain,
                    )
            reference_loaded = []
            for reference_image in reference_images:
                try:
                    reference_time_h = (
                        reference_image.date - exp_start
                    ).total_seconds() / 3600.0
                except Exception:
                    reference_time_h = 0.0
                reference_loaded.append((reference_image, 0.0, reference_time_h))
            _apply_static_light_correction(
                reference_loaded,
                mode=static_light_correction,
                run=f"{run}:noise-reference",
                reference_rgb=np.asarray(
                    cta.color_analysis.base.img, dtype=np.float32
                )[..., :3],
                valid_mask=np.isin(
                    np.asarray(cta.labels.img),
                    np.asarray(signal_labels, dtype=int),
                ),
            )
            prepared_references = [
                cta.call_color_interpretation(reference_image)
                for reference_image, _mass, _time_h in reference_loaded
            ]
            noise_stride = max(
                1,
                int(os.environ.get("FFAC_AQUEOUS_NOISE_STRIDE", "4")),
            )
            if aqueous_noise_gate_mode != "off":
                noise_gate_model = estimate_aqueous_noise_gate(
                    prepared_references,
                    np.asarray(cta.labels.img),
                    signal_labels,
                    mode=aqueous_noise_gate_mode,
                    source_paths=used_reference_paths,
                    quantile_low=noise_quantile_low,
                    quantile_high=noise_quantile_high,
                    spatial_stride=noise_stride,
                )
                setattr(cta, "aqueous_noise_gate", noise_gate_model)
                noise_gate_provenance = noise_gate_model.to_dict()
                noise_gate_provenance.update(
                    {
                        "status": "active",
                        "registry_key": reference_key,
                        "skipped_flagged_paths": skipped_flagged_paths,
                    }
                )
            if spatial_support_enabled:
                spatial_reference_coordinates = [
                    btb_acidification_coordinate(reference_image.img)
                    for reference_image, _mass, _time_h in reference_loaded
                ]
                spatial_support_model = estimate_spatial_aqueous_support(
                    spatial_reference_coordinates,
                    np.asarray(cta.labels.img),
                    signal_labels,
                    quantile_low=noise_quantile_low,
                    quantile_high=noise_quantile_high,
                )
                spatial_support_provenance = {
                    "status": "active",
                    "enabled": True,
                    "registry_key": reference_key,
                    "coordinate": (
                        "pixel-centred ratiometric BTB acidification "
                        "(mean(R,G)-B)/(mean(R,G)+B)"
                    ),
                    "independent_of_color_path": True,
                    "source_paths": [str(path) for path in used_reference_paths],
                    "skipped_flagged_paths": skipped_flagged_paths,
                    "drift_quantile": spatial_support_model.drift_quantile,
                    "thresholds": {
                        str(label): {
                            "low": threshold.low,
                            "high": threshold.high,
                            "sample_count": threshold.sample_count,
                        }
                        for label, threshold in spatial_support_model.thresholds.items()
                    },
                }
            if color_path_evidence_mode != "off":
                relative_rgb_references = [
                    relative_rgb_for_color_analysis(
                        cta.color_analysis,
                        reference_image,
                    )
                    for reference_image, _mass, _time_h in reference_loaded
                ]
                color_path_evidence_model = estimate_color_path_evidence(
                    relative_rgb_references,
                    np.asarray(cta.labels.img),
                    signal_labels,
                    quantile_low=noise_quantile_low,
                    quantile_high=noise_quantile_high,
                    spatial_stride=max(
                        1,
                        int(os.environ.get("FFAC_COLOR_PATH_EVIDENCE_STRIDE", "4")),
                    ),
                    residual_low=color_path_residual_low,
                    residual_high=color_path_residual_high,
                )
                color_path_evidence_provenance = {
                    **color_path_evidence_model.to_dict(),
                    "status": "active",
                    "mode": color_path_evidence_mode,
                    "parameter_independent": True,
                    "source_paths": [str(path) for path in used_reference_paths],
                    "skipped_flagged_paths": skipped_flagged_paths,
                }
            logger.info(
                "[%s] aqueous reference support ACTIVE references=%d "
                "global_gate=%s spatial=%s rgb_path=%s labels=%s",
                run,
                len(used_reference_paths),
                aqueous_noise_gate_mode,
                spatial_support_enabled,
                color_path_evidence_mode,
                sorted(int(label) for label in signal_labels),
            )
            del prepared_references, reference_images, reference_loaded
            if "relative_rgb_references" in locals():
                del relative_rgb_references
        except Exception as exc:
            noise_gate_provenance["status"] = "failed"
            noise_gate_provenance["error"] = str(exc)
            spatial_support_provenance["status"] = "failed"
            spatial_support_provenance["error"] = str(exc)
            color_path_evidence_provenance["status"] = "failed"
            color_path_evidence_provenance["error"] = str(exc)
            if strict_reference_support:
                raise RuntimeError(
                    f"[{run}] aqueous reference-support setup failed: {exc}"
                ) from exc
            logger.warning(
                "[%s] aqueous noise gate disabled after setup failure: %s",
                run,
                exc,
            )

    # ---- optional spatial downscale (workers only; master uses quality_scale=1.0) ----
    # Ported from ff_um's _apply_quality_to_rig. Use darsia.resize (which updates the
    # Image's num_voxels / coordinate system, unlike a raw cv2 resize of .img) on the
    # SOURCE arrays the pipeline derives from, to a single shared target shape:
    #   - the calibration images,
    #   - the relative-mode colour baseline (color_analysis.base) and facies labels,
    #   - the CO2 mass analysis baseline (co2_mass_analysis.baseline). Its height_map is a
    #     @property of baseline.num_voxels, and temperature/pressure/solubility/density all
    #     derive from it, so re-running setup_density_gaseous_co2() after coarsening the
    #     baseline rebuilds all mass arrays at the coarse resolution. (My earlier mistake
    #     was coarsening solubility/density directly with cv2 on .img, which they then
    #     recomputed at full res from the un-coarsened baseline -> shape mismatch.)
    # geometry.integrate() rescales the voxel volume to the data shape itself, so the
    # integral stays mass-conservative without rebuilding the geometry.
    if quality_scale and float(quality_scale) != 1.0 and loaded:
        import darsia as _darsia
        import cv2 as _cv2
        s = float(quality_scale)
        H0, W0 = loaded[0][0].img.shape[:2]
        target = (max(1, int(round(H0 * s))), max(1, int(round(W0 * s))))

        def _rs(im, nearest=False):
            if im is None or getattr(im, "img", None) is None:
                return im
            return _darsia.resize(
                im, shape=target,
                interpolation="inter_nearest" if nearest else "inter_area",
            )

        # Each coarsening step runs in ISOLATION. Previously everything sat in one outer
        # try, so the first step that raised aborted all SUBSEQUENT steps - but the in-place
        # mutations already applied left a HALF-coarsened rig (some arrays coarse, some full
        # res), which is exactly what produces "operands could not be broadcast together".
        # Isolating each step guarantees every array is attempted independently, and the
        # per-step warning + final shape report make any remaining mismatch visible.
        def _step(name, fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] quality_scale step '%s' FAILED: %s", run, name, exc)

        # 1) Calibration images.
        def _do_images():
            for i, (im, inj, th) in enumerate(loaded):
                loaded[i] = (_rs(im), inj, th)

        # 2) Per-analysis base / labels / mask (color_analysis + signal_model).
        def _do_analyses():
            for _ca_attr in ("color_analysis", "signal_model"):
                _ca = getattr(cta, _ca_attr, None)
                if _ca is None:
                    continue
                if getattr(_ca, "base", None) is not None:
                    _ca.base = _rs(_ca.base)                    # relative-mode colour baseline
                if getattr(_ca, "labels", None) is not None:
                    _ca.labels = _rs(_ca.labels, nearest=True)  # facies labels (integer ids)
                _m = getattr(_ca, "mask", None)
                if isinstance(_m, np.ndarray) and _m.ndim >= 2 and _m.shape[:2] == (H0, W0):
                    _ca.mask = _cv2.resize(_m.astype(np.uint8), (target[1], target[0]),
                                           interpolation=_cv2.INTER_NEAREST).astype(_m.dtype)

        # 3) HeterogeneousModel.masks - the ACTUAL per-facies masking (darsia.Masks built
        #    from full-res labels at setup). Rebuilt from each model's own coarse labels.
        def _do_masks():
            def _rebuild(_model, _seen):
                if _model is None or id(_model) in _seen:
                    return
                _seen.add(id(_model))
                _mk = getattr(_model, "masks", None)
                if isinstance(_mk, _darsia.Masks) and getattr(_mk, "labels", None) is not None:
                    if _mk.labels.img.shape[:2] != target:
                        _model.masks = _darsia.Masks(_rs(_mk.labels, nearest=True))
                for _sub in (getattr(_model, "models", None) or []):
                    _rebuild(_sub, _seen)
                _obj = getattr(_model, "obj", None)
                if isinstance(_obj, dict):
                    for _sub in _obj.values():
                        _rebuild(_sub, _seen)
            for _mattr in ("color_analysis", "signal_model"):
                _mca = getattr(cta, _mattr, None)
                if _mca is not None:
                    _rebuild(getattr(_mca, "model", None), set())

        # 4) CO2 mass analysis: coarsen baseline (height_map / temperature / pressure all
        #    derive from baseline.num_voxels) then rebuild density + solubility. As a
        #    guaranteed fallback, if setup_density did not yield the target shape, resize
        #    the density/solubility maps directly (they vary smoothly with height, so
        #    INTER_AREA is physically faithful).
        # 4a) Best-effort: coarsen the CO2 baseline and let setup_density rebuild the maps
        #     from the coarse height_map. May raise if the baseline is not a plain 2d Image
        #     (darsia.resize asserts space_dim==2) - that is fine, step 4b is the guarantee.
        def _do_mass_baseline():
            cma = getattr(cta, "co2_mass_analysis", None)
            if cma is None:
                return
            if getattr(cma, "baseline", None) is not None:
                cma.baseline = _rs(cma.baseline)
            if hasattr(cma, "setup_density_gaseous_co2"):
                cma.setup_density_gaseous_co2()

        # 4b) GUARANTEED: directly resize the density/solubility maps to the target shape.
        #     Independent of 4a, so even if the baseline resize raised, the mass-side maps
        #     are still coarsened and the multiply density*s_g / solubility*c_aq matches.
        #     They vary smoothly with height, so INTER_AREA is physically faithful.
        def _do_mass_force():
            cma = getattr(cta, "co2_mass_analysis", None)
            if cma is None:
                return
            for _attr in ("density_gaseous_co2", "solubility_co2"):
                _arr = getattr(cma, _attr, None)
                if isinstance(_arr, np.ndarray) and _arr.ndim >= 2 and _arr.shape[:2] != target:
                    setattr(cma, _attr,
                            _cv2.resize(_arr.astype(np.float64), (target[1], target[0]),
                                        interpolation=_cv2.INTER_AREA).astype(_arr.dtype))

        # 3b) Restoration objects (e.g. VolumeAveraging) hold their OWN full-res mask Image
        #     plus derived mean_pore_volume / zero_indices. ConcentrationAnalysis._restore_signal
        #     multiplies the coarse signal by restoration.mask.img -> the (2979,5472) mismatch
        #     the traceback pinpointed. Coarsen the mask, scale the REV (in voxels) with the
        #     resolution, and recompute the derived maps exactly as VolumeAveraging.__init__.
        def _do_restoration():
            for _ca_attr in ("color_analysis", "signal_model"):
                _ca = getattr(cta, _ca_attr, None)
                if _ca is None:
                    continue
                _rest = getattr(_ca, "restoration", None)
                if _rest is None:
                    continue
                for _r in (_rest if isinstance(_rest, (list, tuple)) else [_rest]):
                    _mask = getattr(_r, "mask", None)
                    if _mask is None or getattr(_mask, "img", None) is None:
                        continue
                    if _mask.img.shape[:2] != (H0, W0):
                        continue
                    if isinstance(getattr(_r, "rev_size", None), (int, float)):
                        _r.rev_size = max(1, int(round(_r.rev_size * s)))
                    _r.mask = _rs(_mask)
                    if hasattr(_r, "_heterogeneous_uniform_filter"):
                        _r.mean_pore_volume = _r._heterogeneous_uniform_filter(
                            _r.mask.astype(float).img)
                        _r.zero_indices = np.where(_r.mean_pore_volume < 1e-12)

        _step("images", _do_images)
        _step("analyses_base_labels_mask", _do_analyses)
        _step("restoration", _do_restoration)
        _step("hetero_masks", _do_masks)
        _step("co2_mass_baseline", _do_mass_baseline)
        _step("co2_mass_force", _do_mass_force)

        # 5) Diagnostics via print() (stdout is captured in the worker console, unlike the
        #    module logger). Shows the ACTUAL post-coarsening shapes - any surviving full-res
        #    array is then immediately visible.
        try:
            cma = getattr(cta, "co2_mass_analysis", None)
            _ish = loaded[0][0].img.shape[:2] if loaded else None
            _b = getattr(getattr(cta, "color_analysis", None), "base", None)
            _bsh = _b.img.shape[:2] if _b is not None and getattr(_b, "img", None) is not None else None
            _dsh = getattr(getattr(cma, "density_gaseous_co2", None), "shape", None)
            _ssh = getattr(getattr(cma, "solubility_co2", None), "shape", None)
            _r = getattr(getattr(cta, "color_analysis", None), "restoration", None)
            _r0 = _r[0] if isinstance(_r, (list, tuple)) and _r else _r
            _rsh = getattr(getattr(getattr(_r0, "mask", None), "img", None), "shape", None)
            print(f"[QSCALE {run}] scale={s:.3f} target={target} | img={_ish} base={_bsh} "
                  f"density={_dsh[:2] if _dsh else None} solubility={_ssh[:2] if _ssh else None} "
                  f"rest_mask={_rsh[:2] if _rsh else None}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[QSCALE {run}] shape-report failed: {exc}", flush=True)

    # auto-match the param space to the signal-model resolution (num_segments+1 points)
    n_free_values = None
    try:
        probe = signal_labels[0] if signal_labels else (
            list(hetero.keys())[0] if hasattr(hetero, "keys") and list(hetero.keys()) else None)
        if probe is not None:
            n_free_values = max(1, len(list(hetero[probe].values)) - 1)
    except Exception:
        n_free_values = None
    param_space = build_param_space(run, bounds_map, signal_labels=signal_labels,
                                    per_label_params=per_label_params, use_facies=use_facies,
                                    n_free_values=n_free_values,
                                    signal_parameterization=signal_parameterization,
                                    label_to_facies=signal_label_to_facies)
    param_space = configure_flash_param_space(
        param_space,
        run,
        bounds_map,
        getattr(cta, "flash", None),
    )
    depth_parameterization = normalise_depth_parameterization(
        depth_parameterization
    )
    param_space, depth_model = configure_depth_param_space(
        param_space,
        run,
        bounds_map,
        geometry,
        depth_parameterization,
        minimum_m=depth_min_m,
        maximum_m=depth_max_m,
    )
    if phase_separation == "supervised-dual":
        # The gas branch is derived exclusively from the reviewed optical
        # signal; Flash gas-width parameters would be inert in this mode.
        param_space = [
            entry
            for entry in param_space
            if entry.get("name") != "flash.gas_width"
        ]
        override = (bounds_map or {}).get(run, {})
        default_override = (bounds_map or {}).get("default", {})
        for entry in (
            {
                "name": "gas.dual_onset",
                "attr_path": ["phase_separation", "dual_onset"],
                "bounds": (0.05, 0.95),
                "type": "float",
            },
            {
                "name": "gas.dual_width",
                "attr_path": ["phase_separation", "dual_width"],
                "bounds": (0.02, 0.70),
                "type": "float",
            },
            {
                "name": "gas.dual_max_saturation",
                "attr_path": ["phase_separation", "dual_max_saturation"],
                "bounds": (0.02, 0.80),
                "type": "float",
            },
        ):
            bounds = _match_bounds(entry["name"], override, default_override)
            if bounds is not None:
                entry["bounds"] = tuple(bounds)
            param_space.append(entry)
    elif phase_separation == "residual-gas":
        override = (bounds_map or {}).get(run, {})
        default_override = (bounds_map or {}).get("default", {})
        for entry in (
            {
                "name": "gas.residual_onset",
                "attr_path": ["phase_separation", "residual_onset"],
                "bounds": (0.01, 0.80),
                "type": "float",
            },
            {
                "name": "gas.residual_width",
                "attr_path": ["phase_separation", "residual_width"],
                "bounds": (0.02, 1.50),
                "type": "float",
            },
        ):
            bounds = _match_bounds(entry["name"], override, default_override)
            if bounds is not None:
                entry["bounds"] = tuple(bounds)
            param_space.append(entry)
    elif phase_separation in {
        "clearwater-direct",
        "clearwater-gated",
        "hybrid-causal",
    }:
        if phase_separation == "clearwater-direct":
            param_space = [
                entry
                for entry in param_space
                if entry.get("name") != "flash.gas_width"
            ]
        override = (bounds_map or {}).get(run, {})
        default_override = (bounds_map or {}).get("default", {})
        entries = (
            (
                {
                    "name": "gas.hybrid_half_life_h",
                    "attr_path": ["phase_separation", "hybrid_half_life_h"],
                    "bounds": DEFAULT_HYBRID_HALF_LIFE_BOUNDS_H,
                    "type": "float",
                },
            )
            if phase_separation == "hybrid-causal"
            else (
                {
                    "name": "gas.clear_onset",
                    "attr_path": ["phase_separation", "clear_onset"],
                    "bounds": (0.05, 0.80),
                    "type": "float",
                },
                {
                    "name": "gas.clear_width",
                    "attr_path": ["phase_separation", "clear_width"],
                    "bounds": (0.05, 0.80),
                    "type": "float",
                },
            )
        )
        for entry in entries:
            bounds = _match_bounds(entry["name"], override, default_override)
            if bounds is not None:
                entry["bounds"] = tuple(bounds)
            param_space.append(entry)

    label_image = np.asarray(cta.labels.img)
    valid_mask = np.isin(label_image, np.asarray(signal_labels, dtype=int))
    hard_nonseal_gas_mask, hard_seal_gas_provenance = (
        _hard_nonseal_gas_support(
            np.asarray(fluidflower.facies.img),
            run=run,
        )
    )

    gas_scores: List[np.ndarray] = []
    gas_score_scale = _RESIDUAL_GAS_SCORE_CLIP
    gas_score_provenance: Optional[Dict[str, Any]] = None
    if phase_separation == "supervised-dual" and loaded:
        gas_scores, gas_score_provenance = (
            _precompute_supervised_dual_gas_scores(
                cta,
                loaded,
                valid_mask,
                run=run,
            )
        )
        gas_score_scale = 1.0
    elif phase_separation == "residual-gas" and loaded:
        gas_scores = _precompute_residual_gas_scores(
            cta,
            loaded,
            signal_labels,
            run=run,
        )
        gas_score_provenance = {
            "kind": "off-color-path-residual",
            "score_scale": gas_score_scale,
        }
    elif phase_separation in {
        "clearwater-direct",
        "clearwater-gated",
        "hybrid-causal",
    } and loaded:
        import cv2

        try:
            from darsia.fluidflower.tools.clearwater_phase_reference import (
                average_rgb,
                select_clear_reference_paths,
            )
        except ModuleNotFoundError:
            from darsia.fluidflower.tools.clearwater_phase_reference import (
                average_rgb,
                select_clear_reference_paths,
            )

        raw_root = Path(
            os.environ.get("FFAC_CLEARWATER_RAW_ROOT", str(configured_tool_path('ffac', 'raw_inputs')))
        )
        reference_paths = select_clear_reference_paths(raw_root, run)
        reference_images: list[np.ndarray] = []
        skipped_flagged: list[str] = []
        target_shape = tuple(np.asarray(loaded[0][0].img).shape[:2])
        for reference_path in reference_paths:
            try:
                reference_image, flagged, _diagnostics = _read_and_flag(reference_path)
            except ValueError as exc:
                if "No imaging protocol configured" not in str(exc):
                    raise
                # Flush/wash acquisitions sit outside the experiment timeline.
                # Apply the same rig corrections explicitly, then the same AC14
                # registration used for the calibration frames.
                import darsia

                reference_image = darsia.imread(
                    reference_path,
                    transformations=fluidflower.corrections,
                    reference_date=fluidflower.reference_date,
                    name=reference_path.name,
                )
                if template_registration is not None:
                    reference_image = _warp_registration_image(
                        reference_image,
                        template_registration,
                        mode=template_registration_mode,
                    )
                flagged = any(
                    getattr(correction, "active", False)
                    and getattr(correction, "last_flagged", False)
                    for correction in (
                        getattr(fluidflower, "color_corrections", None) or []
                    )
                )
            if flagged:
                skipped_flagged.append(str(reference_path))
                continue
            if anchor_photometric_gain is not None:
                _apply_anchor_photometric_gain(
                    reference_image,
                    anchor_photometric_gain,
                )
            reference_array = np.asarray(reference_image.img, dtype=np.float32)[
                ..., :3
            ]
            if reference_array.shape[:2] != target_shape:
                reference_array = cv2.resize(
                    reference_array,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            reference_images.append(reference_array)
        if not reference_images:
            raise RuntimeError(
                f"[{run}] all configured clear-water reference frames were flagged"
            )
        clear_reference = average_rgb(reference_images)
        baseline = np.asarray(cta.color_analysis.base.img, dtype=np.float32)[
            ..., :3
        ]
        gas_scores = _precompute_clearwater_gas_scores(
            baseline_rgb=baseline,
            clear_rgb=clear_reference,
            loaded=loaded,
            valid_mask=valid_mask,
            run=run,
        )
        gas_score_scale = 1.0
        gas_score_provenance = {
            "kind": "clear-water-rgb-projection",
            "score_scale": gas_score_scale,
            "raw_root": str(raw_root),
            "reference_paths": [str(path) for path in reference_paths],
            "skipped_flagged_paths": skipped_flagged,
            "orthogonal_scale": float(
                os.environ.get("FFAC_CLEARWATER_ORTHOGONAL_SCALE", "0.20")
            ),
        }

    hybrid_gas_weights: List[np.ndarray] = []
    hybrid_nonseal_mask: Optional[np.ndarray] = None
    hybrid_injection_end_h = 0.0
    hybrid_gas_provenance: Optional[Dict[str, Any]] = None
    if phase_separation == "hybrid-causal" and loaded:
        (
            hybrid_gas_weights,
            hybrid_nonseal_mask,
            hybrid_injection_end_h,
            hybrid_gas_provenance,
        ) = _precompute_hybrid_gas_support(
            gas_scores=gas_scores,
            loaded=loaded,
            valid_mask=valid_mask,
            facies_labels=np.asarray(fluidflower.facies.img),
            config_path=config_path,
            experiment_start=exp_start,
            run=run,
        )

    context = CalibrationContext(
        run=run, config=config, experiment=experiment, fluidflower=fluidflower,
        geometry=geometry, calibration=cta, calibration_images=selected_image_paths,
        reference_label=reference_label, signal_label=None, signal_labels=signal_labels,
        param_space=param_space, enforce_lower=enforce_lower,
        per_label_params=per_label_params,
        use_facies=use_facies,
        signal_parameterization=signal_parameterization,
        signal_label_to_facies=signal_label_to_facies,
        objective_integral=objective_integral,
        late_gas_start_h=resolved_late_gas_start_h,
        late_gas_max_fraction=resolved_late_gas_max_fraction,
        late_gas_penalty_weight=resolved_late_gas_penalty_weight,
        late_gas_decay_half_life_h=resolved_late_gas_decay_half_life_h,
        late_gas_floor_fraction=resolved_late_gas_floor_fraction,
        late_gas_constraint_mode=resolved_late_gas_constraint_mode,
        overflow_phase_constraint=resolved_overflow_phase_constraint,
        overflow_phase_penalty_weight=resolved_overflow_phase_penalty_weight,
        overflow_phase_provenance=(
            resolved_overflow_phase_constraint.to_dict()
            if resolved_overflow_phase_constraint is not None
            else None
        ),
        overflow_post_coverage_penalty_weight=(
            resolved_overflow_post_coverage_penalty_weight
        ),
        overflow_post_coverage_sigma_multiplier=(
            resolved_overflow_post_coverage_sigma_multiplier
        ),
        overflow_post_coverage_fraction_tolerance=(
            resolved_overflow_post_coverage_fraction_tolerance
        ),
        aqueous_diffusion_penalty_weight=(
            resolved_aqueous_diffusion_penalty_weight
        ),
        aqueous_diffusion_start_h=resolved_aqueous_diffusion_start_h,
        aqueous_diffusion_support_level=(
            resolved_aqueous_diffusion_support_level
        ),
        aqueous_diffusion_pair_min_dt_h=(
            resolved_aqueous_diffusion_pair_min_dt_h
        ),
        post_injection_balance_weight=(
            resolved_post_injection_balance_weight
        ),
        post_injection_balance_start_h=(
            resolved_post_injection_balance_start_h
        ),
        depth_parameterization=depth_parameterization,
        depth_model_provenance=depth_model_provenance(depth_model),
        depth_prior_weight=resolved_depth_prior_weight,
        depth_edge_prior_sigma_m=resolved_depth_edge_prior_sigma_m,
        depth_bulge_prior_sigma_m=resolved_depth_bulge_prior_sigma_m,
        _depth_model=depth_model,
        phase_separation=phase_separation,
        label_weights=label_weights, calibration_folder=calibration_folder, _loaded=loaded,
        _gas_scores=gas_scores,
        frame_selection_policy=frame_selection_policy,
        frame_selection_records=frame_selection_records,
        color_correction_provenance=color_correction_provenance,
        evaluation_time_records=evaluation_time_records,
        color_path_source=(str(color_path_source) if color_path_source else None),
        couple_aq_gas=os.environ.get("FFAC_COUPLE_AQ_GAS", "").strip().lower()
        in {"1", "true", "yes", "on"},
        _gas_score_scale=gas_score_scale,
        gas_score_provenance=gas_score_provenance,
        hybrid_injection_end_h=hybrid_injection_end_h,
        _hybrid_gas_weights=hybrid_gas_weights,
        _hybrid_nonseal_mask=hybrid_nonseal_mask,
        hybrid_gas_provenance=hybrid_gas_provenance,
        aqueous_noise_gate=noise_gate_model,
        aqueous_noise_gate_provenance=noise_gate_provenance,
        spatial_observation_support=spatial_support_model,
        spatial_observation_support_provenance=spatial_support_provenance,
        color_path_evidence_mode=color_path_evidence_mode,
        color_path_evidence=color_path_evidence_model,
        color_path_evidence_provenance=color_path_evidence_provenance,
        spatial_objective_weight=resolved_spatial_objective_weight,
        spatial_recall_weight=resolved_spatial_recall_weight,
        spatial_seal_weight=resolved_spatial_seal_weight,
        spatial_trace_level=resolved_spatial_trace_level,
        spatial_core_level=resolved_spatial_core_level,
        layer_gain_regularization_weight=resolved_gain_regularization_weight,
        layer_gain_regularization_sigma=resolved_gain_regularization_sigma,
        run_config_path=config_path,
        _hard_nonseal_gas_mask=hard_nonseal_gas_mask,
        hard_seal_gas_provenance=hard_seal_gas_provenance,
    )
    context.corrected_image_cache_stats = dict(corrected_image_cache_stats)
    return context


_EVAL_TB_PRINTED = False


def _normalise_evaluation_backend(mode: str | None) -> str:
    value = (mode or "prepared").strip().lower().replace("_", "-")
    aliases = {
        "cpu": "prepared",
        "fast": "prepared",
        "numpy": "prepared",
        "off": "legacy",
        "cuda32": "cuda-float32",
        "cuda-fp32": "cuda-float32",
        "ocl": "opencl",
        "ocl32": "opencl-float32",
        "opencl32": "opencl-float32",
        "opencl-fp32": "opencl-float32",
        "cpu64": "cpu-float64",
        "cpu-fp64": "cpu-float64",
        "metal": "mps-metal",
        "mps": "mps-metal",
        "torch-mps": "mps-metal",
        "torch-mps-metal": "mps-metal",
        "metal32": "metal-float32",
        "metal-fp32": "metal-float32",
    }
    value = aliases.get(value, value)
    if value not in {
        "legacy",
        "prepared",
        "cpu-float64",
        "cuda",
        "cuda-float32",
        "opencl",
        "opencl-float32",
        "mps-metal",
        "metal-float32",
    }:
        raise ValueError(
            f"Unknown evaluation backend {mode!r}; expected legacy, prepared, "
            "cpu-float64, cuda, cuda-float32, opencl, opencl-float32, "
            "mps-metal, or metal-float32."
        )
    return value


def depth_model_arrays(
    context: CalibrationContext,
    shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return measured depth and the fixed spatial shape at evaluator resolution."""

    target = (int(shape[0]), int(shape[1]))
    model = getattr(context, "_depth_model", None)
    if not model:
        # The accelerated fixed-depth path only needs a neutral scale factor:
        # weight / 1 * 1 exactly preserves the cached integration weights. This
        # also supports lightweight contexts that intentionally omit geometry.depth.
        neutral = np.ones(target, dtype=np.float64)
        return neutral, neutral
    measured = np.asarray(model["measured"], dtype=np.float64)
    if measured.shape != target:
        import cv2

        measured = cv2.resize(
            measured,
            (target[1], target[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    measured = np.asarray(measured, dtype=np.float64)
    spatial_shape = (
        measured - float(model["edge_reference_m"])
    ) / float(model["bulge_reference_m"])
    return measured, np.asarray(spatial_shape, dtype=np.float64)


def resolve_depth_parameters(
    context: CalibrationContext,
    params: Mapping[str, Any],
) -> tuple[float, float, float, float, bool]:
    """Resolve trial depth coefficients and physical clipping limits."""

    model = getattr(context, "_depth_model", None)
    if not model:
        return 0.0, 1.0, 0.0, float("inf"), False
    edge_m = float(params.get("depth.edge_m", model["edge_reference_m"]))
    bulge_m = float(params.get("depth.bulge_m", model["bulge_reference_m"]))
    minimum_m = float(model["minimum_m"])
    maximum_m = float(model["maximum_m"])
    return edge_m, bulge_m, minimum_m, maximum_m, True


def effective_depth_field(
    context: CalibrationContext,
    params: Mapping[str, Any],
    shape: Sequence[int],
) -> np.ndarray:
    measured, spatial_shape = depth_model_arrays(context, shape)
    edge_m, bulge_m, minimum_m, maximum_m, active = resolve_depth_parameters(
        context,
        params,
    )
    if not active:
        return measured
    return np.clip(
        edge_m + bulge_m * spatial_shape,
        minimum_m,
        maximum_m,
    )


def voxel_volume_with_depth_model(
    context: CalibrationContext,
    params: Mapping[str, Any],
    shape: Sequence[int],
) -> np.ndarray:
    """Return voxel volumes using the candidate-specific effective cell gap."""

    target = (int(shape[0]), int(shape[1]))
    context.geometry._prepare_cached_voxel_volume(list(target))
    voxel_volume = np.broadcast_to(
        np.asarray(context.geometry.cached_voxel_volume, dtype=np.float64),
        target,
    )
    if not getattr(context, "_depth_model", None):
        return voxel_volume
    measured, _spatial_shape = depth_model_arrays(context, target)
    effective = effective_depth_field(context, params, target)
    return voxel_volume * np.divide(
        effective,
        measured,
        out=np.ones_like(effective),
        where=measured > 0.0,
    )


def integrate_mass_with_depth_model(
    context: CalibrationContext,
    mass: Any,
    params: Mapping[str, Any],
) -> float:
    """Integrate a mass-density image with the trial-specific cell gap."""

    if not getattr(context, "_depth_model", None):
        return float(context.geometry.integrate(mass))
    values = np.asarray(getattr(mass, "img", mass), dtype=np.float64)
    if values.ndim != 2:
        raise RuntimeError(
            f"Depth-aware integration expects a 2-D scalar image, got {values.shape}"
        )
    adjusted_volume = voxel_volume_with_depth_model(context, params, values.shape)
    return float(np.sum(values * adjusted_volume, dtype=np.float64))


class _CudaIntegratedEvaluator:
    """Evaluate integrated masses while keeping prepared frame data in GPU memory."""

    def __init__(self, context: CalibrationContext) -> None:
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CUDA evaluation requires the optional 'cuda' dependency"
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA evaluation requested, but no CUDA device is available")
        if not context._prepared_colors:
            raise RuntimeError("CUDA evaluation requires prepared color frames")

        calibration = context.calibration
        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError("CUDA evaluation does not support flash restoration")

        shape = tuple(np.asarray(context._prepared_colors[0].img).shape[:2])
        self.restoration_spec = signal_restoration_spec(calibration, shape)
        context.geometry._prepare_cached_voxel_volume(list(shape))
        voxel_volume = np.asarray(context.geometry.cached_voxel_volume)
        density = np.asarray(calibration.co2_mass_analysis.density_gaseous_co2)
        solubility = np.asarray(calibration.co2_mass_analysis.solubility_co2)
        try:
            voxel_volume = np.broadcast_to(voxel_volume, shape)
            density = np.broadcast_to(density, shape)
            solubility = np.broadcast_to(solubility, shape)
        except ValueError as exc:
            raise RuntimeError(
                f"CUDA static-array shape mismatch for {context.run}: "
                f"shape={shape}, voxel={voxel_volume.shape}, density={density.shape}, "
                f"solubility={solubility.shape}"
            ) from exc

        labels = np.asarray(calibration.labels.img)
        if labels.shape != shape:
            raise RuntimeError(
                f"CUDA labels/frame shape mismatch for {context.run}: "
                f"{labels.shape} != {shape}"
            )

        self.cp = cp
        self.context = context
        self.flash = flash
        self.phase_separation = _normalise_phase_separation(
            getattr(context, "phase_separation", "shared-signal")
        )
        self.arrival_scale = max(
            1e-6,
            float(
                os.environ.get(
                    "FFAC_PHASE_GAS_ARRIVAL_SCALE",
                    os.environ.get("FFAC_RESIDUAL_GAS_ARRIVAL_SCALE", "0.02"),
                )
            ),
        )
        self.score_scale = (
            1.0
            if self.phase_separation == "hybrid-causal"
            else float(
                getattr(context, "_gas_score_scale", _RESIDUAL_GAS_SCORE_CLIP)
            )
        ) / 255.0
        spatial_confidences = list(
            getattr(context, "_spatial_observation_confidences", []) or []
        )
        self.spatial_objective_active = bool(spatial_confidences)
        if self.spatial_objective_active and len(spatial_confidences) != len(
            context._prepared_colors
        ):
            raise RuntimeError(
                f"CUDA spatial confidence count mismatch for {context.run}: "
                f"{len(spatial_confidences)} != {len(context._prepared_colors)}"
            )
        seal_mask = np.zeros(shape, dtype=bool)
        if self.spatial_objective_active:
            seal_ids = tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in os.environ.get(
                            "FFAC_SPATIAL_SEAL_FACIES_IDS", "1"
                        ).split(",")
                        if value.strip()
                    }
                )
            )
            facies = np.asarray(context.fluidflower.facies.img)
            if facies.shape != shape:
                raise RuntimeError(
                    f"CUDA facies/frame shape mismatch for {context.run}: "
                    f"{facies.shape} != {shape}"
                )
            seal_mask = np.isin(facies, np.asarray(seal_ids, dtype=facies.dtype))
        self.label_data: dict[int, dict[str, Any]] = {}
        active_signal_mask = np.zeros(shape, dtype=bool)

        def sliced_static_data(mask: np.ndarray) -> dict[str, Any]:
            return {
                "flat_indices": cp.asarray(
                    np.flatnonzero(mask.ravel()),
                    dtype=cp.int64,
                ),
                "weight_g_per_m": cp.asarray(
                    np.ascontiguousarray(weighted_g_per_m[mask])
                ),
                "weight_aq_per_m": cp.asarray(
                    np.ascontiguousarray(weighted_aq_per_m[mask])
                ),
                "pore_volume_per_m": cp.asarray(
                    np.ascontiguousarray(pore_volume_per_m[mask])
                ),
                "depth_shape": cp.asarray(
                    np.ascontiguousarray(spatial_depth_shape[mask])
                ),
                "gas_constraint": cp.asarray(
                    np.ascontiguousarray(
                        np.asarray(gas_constraint, dtype=np.float32)[mask]
                    )
                ),
                "aqueous_constraint": cp.asarray(
                    np.ascontiguousarray(
                        np.asarray(aqueous_constraint, dtype=np.float32)[mask]
                    )
                ),
                "seal_mask": cp.asarray(
                    np.ascontiguousarray(seal_mask[mask].astype(np.float32))
                ),
                "observation_confidences": (
                    [
                        cp.asarray(np.ascontiguousarray(frame[mask]))
                        for frame in spatial_confidences
                    ]
                    if self.spatial_objective_active
                    else []
                ),
            }

        weighted_g = voxel_volume * density
        weighted_aq = voxel_volume * solubility
        measured_depth, spatial_depth_shape = depth_model_arrays(context, shape)
        weighted_g_per_m = np.divide(
            weighted_g,
            measured_depth,
            out=np.zeros_like(weighted_g, dtype=np.float64),
            where=measured_depth > 0.0,
        )
        weighted_aq_per_m = np.divide(
            weighted_aq,
            measured_depth,
            out=np.zeros_like(weighted_aq, dtype=np.float64),
            where=measured_depth > 0.0,
        )
        pore_volume_per_m = np.divide(
            voxel_volume,
            measured_depth,
            out=np.zeros_like(voxel_volume, dtype=np.float64),
            where=measured_depth > 0.0,
        )
        reference_frame = context._prepared_colors[0]
        constraint_weights = constraint_weights_for_calibration(
            calibration,
            reference_frame,
        )
        self.constraint_mode = constraint_weights.mode
        self.constraint_expansion_m = constraint_weights.expansion_m
        gas_constraint = constraint_weights.gas
        aqueous_constraint = constraint_weights.aqueous
        hard_nonseal = getattr(context, "_hard_nonseal_gas_mask", None)
        if hard_nonseal is not None:
            hard_nonseal_array = np.asarray(hard_nonseal, dtype=np.float32)
            if hard_nonseal_array.shape != shape:
                raise RuntimeError(
                    f"CUDA hard non-seal mask shape mismatch for {context.run}: "
                    f"{hard_nonseal_array.shape} != {shape}"
                )
            gas_constraint = (
                np.asarray(gas_constraint, dtype=np.float32) * hard_nonseal_array
            )
        if self.phase_separation == "hybrid-causal":
            nonseal = np.asarray(context._hybrid_nonseal_mask, dtype=np.float32)
            if nonseal.shape != shape:
                raise RuntimeError(
                    f"CUDA hybrid non-seal mask shape mismatch for {context.run}: "
                    f"{nonseal.shape} != {shape}"
                )
            gas_constraint = np.asarray(gas_constraint, dtype=np.float32) * nonseal
        signal_models = calibration.signal_model.model[1]
        for label in context.signal_labels:
            mask = labels == int(label)
            if not np.any(mask):
                continue
            active_signal_mask |= mask
            model = signal_models[int(label)]
            supports = np.asarray(model.supports, dtype=np.float64)
            data = sliced_static_data(mask)
            data.update({
                "model": model,
                "supports": cp.asarray(supports),
                "signals": [
                    cp.asarray(
                        np.ascontiguousarray(
                            np.asarray(frame.img, dtype=np.float64)[mask]
                        )
                    )
                    for frame in context._prepared_colors
                ],
            })
            self.label_data[int(label)] = data

        self.restoration_filter = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        if self.restoration_spec is not None:
            try:
                import cupyx.scipy.ndimage as cpx_ndimage
            except ImportError as exc:
                raise RuntimeError(
                    "CUDA signal restoration requires cupyx.scipy.ndimage."
                ) from exc
            self.restoration_filter = cpx_ndimage.uniform_filter
            self.restoration_mask = cp.asarray(self.restoration_spec.mask)
            safe_mean = self.restoration_spec.mean_pore_volume.copy()
            safe_mean[self.restoration_spec.zero_mask] = 1.0
            self.restoration_mean = cp.asarray(safe_mean)
            self.restoration_zero = cp.asarray(self.restoration_spec.zero_mask)
            self.restoration_signal = cp.zeros(shape, dtype=cp.float64)
            self.restoration_scratch = cp.empty(shape, dtype=cp.float64)
            self.restoration_output = cp.empty(shape, dtype=cp.float64)
        self.restoration_spill_data = None
        restoration_spill_mask = ~active_signal_mask
        if self.restoration_spec is not None and np.any(restoration_spill_mask):
            self.restoration_spill_data = sliced_static_data(restoration_spill_mask)

        self.gas_scores: list[dict[int, Any]] = []
        if self.phase_separation != "shared-signal":
            source_scores = (
                context._hybrid_gas_weights
                if self.phase_separation == "hybrid-causal"
                else context._gas_scores
            )
            if len(source_scores) != len(context._prepared_colors):
                raise RuntimeError(
                    f"CUDA phase score count mismatch for {context.run}: "
                    f"{len(source_scores)} != {len(context._prepared_colors)}"
                )
            for score in source_scores:
                score_array = np.asarray(score)
                frame_scores = {
                    label: cp.asarray(
                        np.ascontiguousarray(score_array[labels == label])
                    )
                    for label in self.label_data
                }
                if self.restoration_spill_data is not None:
                    frame_scores[_RESTORATION_SPILL_LABEL] = cp.asarray(
                        np.ascontiguousarray(score_array[restoration_spill_mask])
                    )
                self.gas_scores.append(frame_scores)

        self.lut_y = (
            cp.asarray(np.asarray(flash._lut_y, dtype=np.float64))
            if hasattr(flash, "_lut_y")
            else None
        )
        self.lut_caq = (
            cp.asarray(np.asarray(flash._lut_caq, dtype=np.float64))
            if hasattr(flash, "_lut_caq")
            else None
        )
        self.frame_count = len(context._prepared_colors)
        loaded_rows = list(getattr(context, "_loaded", []))
        self.frame_times_h = (
            [float(row[2]) for row in loaded_rows]
            if len(loaded_rows) == self.frame_count
            else [0.0] * self.frame_count
        )
        self.hybrid_injection_end_h = float(
            getattr(context, "hybrid_injection_end_h", 0.0)
        )
        cp.cuda.get_current_stream().synchronize()
        free_bytes, total_bytes = cp.cuda.Device().mem_info
        logger.info(
            "[%s] CUDA evaluator ready on %s; frames=%d labels=%s "
            "constraint=%s/%.3fm restoration=%s vram_used_mb=%.0f "
            "vram_free_mb=%.0f",
            context.run,
            cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            self.frame_count,
            sorted(self.label_data),
            self.constraint_mode,
            self.constraint_expansion_m,
            (
                f"signal/rev{self.restoration_spec.rev_size}"
                if self.restoration_spec is not None
                else "off"
            ),
            (total_bytes - free_bytes) / (1024 * 1024),
            free_bytes / (1024 * 1024),
        )

    def evaluate(
        self,
        params: Dict[str, Any],
        *,
        include_distribution: bool = False,
        include_spatial: bool = False,
    ) -> list[Any]:
        cp = self.cp
        if include_spatial and not self.spatial_objective_active:
            raise RuntimeError(
                "CUDA spatial diagnostics requested without observation support"
            )
        values_by_label = {
            label: cp.asarray(
                np.asarray(data["model"].values, dtype=np.float64)
            )
            for label, data in self.label_data.items()
        }
        min_aq = float(self.flash.min_value_aq)
        max_aq = float(self.flash.max_value_aq)
        min_g = float(self.flash.min_value_g)
        max_g = float(self.flash.max_value_g)
        aq_denom = (max_aq - min_aq) or 1.0
        gas_denom = (max_g - min_g) or 1.0
        score_onset, score_width = _phase_score_parameters(
            params,
            self.phase_separation,
        )
        (
            depth_edge_m,
            depth_bulge_m,
            depth_minimum_m,
            depth_maximum_m,
            _depth_active,
        ) = resolve_depth_parameters(self.context, params)

        totals: list[Any] = []
        gaseous: list[Any] = []
        aqueous: list[Any] = []
        aqueous_maxima: list[Any] = []
        aqueous_peak_equivalents: list[Any] = []
        aqueous_support_volumes: list[Any] = []
        aqueous_c_volumes: list[Any] = []
        aqueous_low_observation_masses: list[Any] = []
        co2_low_observation_masses: list[Any] = []
        aqueous_observation_recall_deficits: list[Any] = []
        aqueous_unsupported_seal_masses: list[Any] = []
        co2_unsupported_seal_masses: list[Any] = []
        aqueous_low_observation_support_volumes: list[Any] = []
        aqueous_predicted_support_volumes: list[Any] = []
        diffusion_support_level = cp.float64(
            getattr(
                self.context,
                "aqueous_diffusion_support_level",
                DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
            )
        )
        for frame_index in range(self.frame_count):
            restored_signal_flat = None
            if self.restoration_spec is not None:
                self.restoration_signal.fill(0.0)
                signal_flat = self.restoration_signal.ravel()
                for label, data in self.label_data.items():
                    signal_flat[data["flat_indices"]] = cp.interp(
                        data["signals"][frame_index],
                        data["supports"],
                        values_by_label[label],
                    )
                cp.multiply(
                    self.restoration_signal,
                    self.restoration_mask,
                    out=self.restoration_scratch,
                )
                self.restoration_filter(
                    self.restoration_scratch,
                    size=self.restoration_spec.rev_size,
                    output=self.restoration_output,
                )
                cp.divide(
                    self.restoration_output,
                    self.restoration_mean,
                    out=self.restoration_output,
                )
                self.restoration_output[self.restoration_zero] = 0.0
                restored_signal_flat = self.restoration_output.ravel()
            frame_g = cp.asarray(0.0, dtype=cp.float64)
            frame_aq = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_cmax = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_c2_volume = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_c_volume = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_support_volume = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_low_observation_mass = cp.asarray(0.0, dtype=cp.float64)
            frame_co2_low_observation_mass = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_observation_missed = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_observation_weight = cp.asarray(0.0, dtype=cp.float64)
            frame_aq_unsupported_seal_mass = cp.asarray(0.0, dtype=cp.float64)
            frame_co2_unsupported_seal_mass = cp.asarray(
                0.0, dtype=cp.float64
            )
            frame_aq_low_observation_support_volume = cp.asarray(
                0.0, dtype=cp.float64
            )
            frame_aq_predicted_support_volume = cp.asarray(0.0, dtype=cp.float64)
            evaluation_rows = list(self.label_data.items())
            if self.restoration_spill_data is not None:
                evaluation_rows.append(
                    (_RESTORATION_SPILL_LABEL, self.restoration_spill_data)
                )
            for label, data in evaluation_rows:
                effective_depth = cp.clip(
                    cp.float64(depth_edge_m)
                    + cp.float64(depth_bulge_m) * data["depth_shape"],
                    cp.float64(depth_minimum_m),
                    cp.float64(depth_maximum_m),
                )
                signal = (
                    restored_signal_flat[data["flat_indices"]]
                    if restored_signal_flat is not None
                    else cp.interp(
                        data["signals"][frame_index],
                        data["supports"],
                        values_by_label[label],
                    )
                )
                y_norm = cp.clip((signal - min_aq) / aq_denom, 0.0, 1.0)
                if self.lut_y is not None and self.lut_caq is not None:
                    physical_concentration = cp.interp(
                        y_norm,
                        self.lut_y,
                        self.lut_caq,
                    )
                    aqueous_blend = cp.float64(
                        min(
                            1.0,
                            max(
                                0.0,
                                float(
                                    getattr(
                                        self.flash,
                                        "aqueous_blend",
                                        0.0,
                                    )
                                ),
                            ),
                        )
                    )
                    concentration_aq = (
                        (cp.float64(1.0) - aqueous_blend)
                        * physical_concentration
                        + aqueous_blend * y_norm
                    )
                else:
                    concentration_aq = y_norm
                concentration_aq *= data["aqueous_constraint"]

                if self.phase_separation != "shared-signal":
                    concentration_aq = concentration_aq.astype(cp.float32)
                    score = (
                        self.gas_scores[frame_index][label].astype(cp.float32)
                        * cp.float32(self.score_scale)
                    )
                    if self.phase_separation == "hybrid-causal":
                        shared_weight = hybrid_shared_signal_weight(
                            self.frame_times_h[frame_index],
                            injection_end_h=self.hybrid_injection_end_h,
                            half_life_h=float(
                                params.get(
                                    "gas.hybrid_half_life_h",
                                    DEFAULT_HYBRID_HALF_LIFE_H,
                                )
                            ),
                        )
                        hybrid_weight = (
                            cp.float32(shared_weight)
                            + cp.float32(1.0 - shared_weight) * score
                        )
                        flash_saturation = (
                            cp.clip(signal, min_g, max_g) - min_g
                        ) / gas_denom
                        saturation_g = flash_saturation * hybrid_weight
                    else:
                        score_gate = cp.clip(
                            (score - cp.float32(score_onset))
                            / cp.float32(score_width),
                            0.0,
                            1.0,
                        )
                        if self.phase_separation == "clearwater-gated":
                            flash_saturation = (
                                cp.clip(signal, min_g, max_g) - min_g
                            ) / gas_denom
                            saturation_g = flash_saturation * score_gate
                        elif self.phase_separation == "supervised-dual":
                            saturation_g = score_gate * cp.float32(
                                np.clip(
                                    float(
                                        params.get(
                                            "gas.dual_max_saturation", 0.50
                                        )
                                    ),
                                    0.0,
                                    1.0,
                                )
                            )
                        else:
                            saturation_g = score_gate * cp.clip(
                                concentration_aq / cp.float32(self.arrival_scale),
                                0.0,
                                1.0,
                            )
                    if self.phase_separation in {
                        "clearwater-direct",
                        "clearwater-gated",
                        "hybrid-causal",
                        "supervised-dual",
                    }:
                        saturation_g *= data["gas_constraint"]
                else:
                    saturation_g = (
                        cp.clip(signal, min_g, max_g) - min_g
                    ) / gas_denom
                    saturation_g *= data["gas_constraint"]

                gaseous_extensive_mass = (
                    data["weight_g_per_m"] * effective_depth * saturation_g
                )
                frame_g += cp.sum(gaseous_extensive_mass)
                aqueous_fraction = cp.clip(1.0 - saturation_g, 0.0, None)
                aqueous_extensive_mass = (
                    data["weight_aq_per_m"]
                    * effective_depth
                    * concentration_aq
                    * aqueous_fraction
                )
                frame_aq += cp.sum(aqueous_extensive_mass)
                if include_distribution or include_spatial:
                    aqueous_pore_volume = (
                        data["pore_volume_per_m"]
                        * effective_depth
                        * aqueous_fraction
                    )
                if include_distribution:
                    frame_aq_cmax = cp.maximum(
                        frame_aq_cmax,
                        cp.max(concentration_aq),
                    )
                    frame_aq_c_volume += cp.sum(
                        aqueous_pore_volume * concentration_aq
                    )
                    frame_aq_c2_volume += cp.sum(
                        aqueous_pore_volume * concentration_aq * concentration_aq
                    )
                    frame_aq_support_volume += cp.sum(
                        aqueous_pore_volume
                        * (concentration_aq >= diffusion_support_level)
                    )
                if include_spatial:
                    confidence = (
                        data["observation_confidences"][frame_index].astype(
                            cp.float32
                        )
                        / cp.float32(255.0)
                    )
                    unsupported = cp.clip(
                        cp.float32(1.0) - confidence, 0.0, 1.0
                    )
                    trace_level = cp.float32(self.context.spatial_trace_level)
                    core_level = cp.float32(self.context.spatial_core_level)
                    frame_aq_low_observation_mass += cp.sum(
                        aqueous_extensive_mass * unsupported
                    )
                    total_extensive_mass = (
                        aqueous_extensive_mass + gaseous_extensive_mass
                    )
                    frame_co2_low_observation_mass += cp.sum(
                        total_extensive_mass * unsupported
                    )
                    frame_aq_unsupported_seal_mass += cp.sum(
                        aqueous_extensive_mass * unsupported * data["seal_mask"]
                    )
                    frame_co2_unsupported_seal_mass += cp.sum(
                        total_extensive_mass * unsupported * data["seal_mask"]
                    )
                    predicted_support = concentration_aq >= trace_level
                    frame_aq_predicted_support_volume += cp.sum(
                        aqueous_pore_volume * predicted_support
                    )
                    frame_aq_low_observation_support_volume += cp.sum(
                        aqueous_pore_volume * predicted_support * unsupported
                    )
                    observation_weight = aqueous_pore_volume * confidence
                    frame_aq_observation_weight += cp.sum(observation_weight)
                    frame_aq_observation_missed += cp.sum(
                        observation_weight
                        * cp.clip(
                            (core_level - concentration_aq) / core_level,
                            0.0,
                            1.0,
                        )
                    )
            gaseous.append(frame_g)
            aqueous.append(frame_aq)
            totals.append(frame_g + frame_aq)
            if include_distribution:
                aqueous_maxima.append(frame_aq_cmax)
                aqueous_peak_equivalents.append(
                    cp.where(
                        frame_aq_c_volume > 0.0,
                        frame_aq_c2_volume / frame_aq_c_volume,
                        0.0,
                    )
                )
                aqueous_support_volumes.append(frame_aq_support_volume)
                aqueous_c_volumes.append(frame_aq_c_volume)
            if include_spatial:
                aqueous_low_observation_masses.append(
                    frame_aq_low_observation_mass
                )
                co2_low_observation_masses.append(
                    frame_co2_low_observation_mass
                )
                aqueous_observation_recall_deficits.append(
                    cp.where(
                        frame_aq_observation_weight > 0.0,
                        frame_aq_observation_missed / frame_aq_observation_weight,
                        0.0,
                    )
                )
                aqueous_unsupported_seal_masses.append(
                    frame_aq_unsupported_seal_mass
                )
                co2_unsupported_seal_masses.append(
                    frame_co2_unsupported_seal_mass
                )
                aqueous_low_observation_support_volumes.append(
                    frame_aq_low_observation_support_volume
                )
                aqueous_predicted_support_volumes.append(
                    frame_aq_predicted_support_volume
                )

        stacks = [*totals, *gaseous, *aqueous]
        if include_distribution:
            stacks.extend(
                [
                    *aqueous_maxima,
                    *aqueous_peak_equivalents,
                    *aqueous_support_volumes,
                    *aqueous_c_volumes,
                ]
            )
        if include_spatial:
            stacks.extend(
                [
                    *aqueous_low_observation_masses,
                    *co2_low_observation_masses,
                    *aqueous_observation_recall_deficits,
                    *aqueous_unsupported_seal_masses,
                    *co2_unsupported_seal_masses,
                    *aqueous_low_observation_support_volumes,
                    *aqueous_predicted_support_volumes,
                ]
            )
        values = cp.asnumpy(cp.stack(stacks))
        count = self.frame_count
        if not include_distribution and not include_spatial:
            return [
                (
                    float(values[index]),
                    float(values[count + index]),
                    float(values[2 * count + index]),
                )
                for index in range(count)
            ]
        distribution_offset = 3 * count
        spatial_offset = distribution_offset + (4 * count if include_distribution else 0)
        return [
            IntegratedFrameResult(
                total_mass=float(values[index]),
                gaseous_mass=float(values[count + index]),
                aqueous_mass=float(values[2 * count + index]),
                aqueous_concentration_max=(
                    float(values[distribution_offset + index])
                    if include_distribution
                    else None
                ),
                aqueous_concentration_peak_equivalent=float(
                    values[distribution_offset + count + index]
                ) if include_distribution else None,
                aqueous_affected_pore_volume_m3=(
                    float(values[distribution_offset + 2 * count + index])
                    if include_distribution
                    else None
                ),
                aqueous_effective_pore_volume_m3=(
                    float(
                        values[distribution_offset + 3 * count + index]
                        / values[distribution_offset + count + index]
                    )
                    if include_distribution
                    and values[distribution_offset + count + index] > 0.0
                    else (0.0 if include_distribution else None)
                ),
                aqueous_low_observation_mass_kg=(
                    float(values[spatial_offset + index])
                    if include_spatial
                    else None
                ),
                co2_low_observation_mass_kg=(
                    float(values[spatial_offset + count + index])
                    if include_spatial
                    else None
                ),
                aqueous_observation_recall_deficit=(
                    float(values[spatial_offset + 2 * count + index])
                    if include_spatial
                    else None
                ),
                aqueous_unsupported_seal_mass_kg=(
                    float(values[spatial_offset + 3 * count + index])
                    if include_spatial
                    else None
                ),
                co2_unsupported_seal_mass_kg=(
                    float(values[spatial_offset + 4 * count + index])
                    if include_spatial
                    else None
                ),
                aqueous_low_observation_support_volume_m3=(
                    float(values[spatial_offset + 5 * count + index])
                    if include_spatial
                    else None
                ),
                aqueous_predicted_support_volume_m3=(
                    float(values[spatial_offset + 6 * count + index])
                    if include_spatial
                    else None
                ),
            )
            for index in range(count)
        ]

    def close(self) -> None:
        self.label_data.clear()
        self.restoration_spill_data = None
        self.gas_scores.clear()
        self.lut_y = None
        self.lut_caq = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        self.restoration_filter = None
        self.cp.get_default_memory_pool().free_all_blocks()


def _apply_causal_aqueous_prepared_gate(
    context: CalibrationContext,
    prepared: Sequence[Any],
    confidences: Sequence[np.ndarray],
) -> None:
    """Apply one fixed causal support mask before repeated Optuna evaluation."""

    gate = getattr(context, "aqueous_noise_gate", None)
    if gate is None or getattr(gate, "mode", "off") != "baseline-causal":
        return
    if not prepared or len(prepared) != len(confidences):
        raise RuntimeError(
            f"[{context.run}] causal aqueous preparation has mismatched frames"
        )
    labels = np.asarray(context.calibration.labels.img)
    valid = np.isin(labels, np.asarray(context.signal_labels, dtype=int))
    facies = np.asarray(context.fluidflower.facies.img)
    if facies.shape != valid.shape:
        raise RuntimeError(
            f"[{context.run}] causal aqueous facies shape {facies.shape} "
            f"!= {valid.shape}"
        )
    x_m, y_m = _hybrid_coordinate_axes(prepared[0])
    config_path = getattr(context, "run_config_path", None)
    if config_path is None:
        raise RuntimeError(f"[{context.run}] causal aqueous gate has no run config")
    stages = _load_hybrid_injection_stages(
        Path(config_path), context.experiment.experiment_start
    )
    if not stages:
        raise RuntimeError(f"[{context.run}] causal aqueous gate found no injection stages")
    settings = CausalGateSettings(
        trace_level=float(os.environ.get("FFAC_CAUSAL_AQ_TRACE_CONFIDENCE", "0.25")),
        core_level=float(os.environ.get("FFAC_CAUSAL_AQ_CORE_CONFIDENCE", "0.75")),
        confidence_level=float(
            os.environ.get("FFAC_CAUSAL_AQ_OBSERVATION_CONFIDENCE", "0.50")
        ),
        minimum_observed_fraction=float(
            os.environ.get("FFAC_CAUSAL_AQ_MIN_OBSERVED_FRACTION", "0.10")
        ),
        history_radius_m=float(
            os.environ.get("FFAC_CAUSAL_AQ_HISTORY_RADIUS_M", "0.005")
        ),
        reentry_radius_m=float(
            os.environ.get("FFAC_CAUSAL_AQ_REENTRY_RADIUS_M", "0.015")
        ),
        minimum_reentry_observed_fraction=float(
            os.environ.get("FFAC_CAUSAL_AQ_REENTRY_OBSERVED_FRACTION", "0.75")
        ),
        minimum_reentry_core_fraction=float(
            os.environ.get("FFAC_CAUSAL_AQ_REENTRY_CORE_FRACTION", "0.10")
        ),
        structure_scale=float(
            os.environ.get("FFAC_CAUSAL_AQ_STRUCTURE_SCALE", "0.20")
        ),
        seal_facies_ids=tuple(
            sorted(
                {
                    int(value.strip())
                    for value in os.environ.get(
                        "FFAC_HARD_SEAL_FACIES_IDS", "1"
                    ).split(",")
                    if value.strip()
                }
            )
        ),
        allow_nonseal_reentry=True,
    )
    confidence_fields = [
        np.asarray(value, dtype=np.float32) for value in confidences
    ]
    causal = apply_causal_aqueous_gate(
        confidence_fields,
        confidence_fields,
        valid,
        x_coordinates_m=x_m,
        y_coordinates_m=y_m,
        elapsed_hours=[float(row[2]) for row in context._loaded],
        injection_stages=stages,
        facies_labels=facies,
        settings=settings,
    )
    hard_contact = apply_hard_seal_contact_constraint(
        [frame.concentration for frame in causal],
        [np.zeros(valid.shape, dtype=np.float32) for _frame in causal],
        valid,
        facies,
        seal_facies_ids=settings.seal_facies_ids,
        aqueous_trace_level=settings.trace_level,
        gas_support_level=0.05,
    )
    seal = np.isin(facies, np.asarray(settings.seal_facies_ids, dtype=facies.dtype))
    frame_rows: list[dict[str, Any]] = []
    for index, (image, causal_frame, seal_frame) in enumerate(
        zip(prepared, causal, hard_contact, strict=True)
    ):
        retained = np.asarray(causal_frame.retained_trace_mask, dtype=bool).copy()
        retained[seal] &= np.asarray(
            seal_frame.retained_seal_aqueous_mask, dtype=bool
        )[seal]
        values = np.asarray(image.img)
        scalar = values[..., 0] if values.ndim == 3 and values.shape[-1] == 1 else values
        if scalar.ndim != 2:
            raise RuntimeError(
                f"[{context.run}] causal aqueous coordinate is not scalar: {values.shape}"
            )
        filtered = np.array(scalar, copy=True)
        filtered[valid & ~retained] = 0.0
        image.img = filtered
        frame_rows.append(
            {
                "frame_index": index,
                "time_h": float(context._loaded[index][2]),
                "retained_valid_fraction": float(
                    np.count_nonzero(retained & valid) / max(1, np.count_nonzero(valid))
                ),
                "components": int(causal_frame.component_count),
                "retained_components": int(causal_frame.retained_component_count),
                "rejected_components": int(causal_frame.rejected_component_count),
                "removed_seal_components": int(
                    seal_frame.removed_seal_aqueous_component_count
                ),
            }
        )
    context.aqueous_causal_gate_provenance = {
        "active": True,
        "coordinate": "reference-normalized color-path confidence",
        "parameter_independent": True,
        "settings": {
            key: value for key, value in settings.__dict__.items()
        },
        "hard_current_frame_seal_contact": True,
        "mass_reassignment": False,
        "frames": frame_rows,
    }
    logger.info(
        "[%s] causal aqueous support ACTIVE mean_retained_fraction=%.4f "
        "rejected_components=%d",
        context.run,
        float(np.mean([row["retained_valid_fraction"] for row in frame_rows])),
        sum(int(row["rejected_components"]) for row in frame_rows),
    )


def prepare_evaluation_context(
    context: CalibrationContext,
    *,
    backend: str = "prepared",
    release_images: bool = True,
    initialize_backend: bool = True,
) -> CalibrationContext:
    """Prepare parameter-independent frame data for repeated trial evaluation.

    ``initialize_backend=False`` intentionally stops after the host-side arrays
    are complete.  A worker may then retain this context in RAM while another
    process owns the accelerator and initialise the requested backend only when
    an accelerator slot becomes available.
    """

    import gc
    import time

    call_started = time.perf_counter()
    timing: Dict[str, Any] = {
        "requested_backend": _normalise_evaluation_backend(backend),
        "initialize_backend": bool(initialize_backend),
    }
    requested = _normalise_evaluation_backend(backend)
    if requested == "legacy":
        context._evaluation_backend = "legacy"
        timing["total_seconds"] = time.perf_counter() - call_started
        context._prepare_evaluation_timing = timing
        return context
    cache_started = time.perf_counter()
    prepared_cache_path = None
    if not context._prepared_colors:
        prepared_cache_path = _load_prepared_context_cache(context)
    timing["prepared_context_cache"] = {
        "status": "hit" if prepared_cache_path is not None else "miss",
        "path": str(prepared_cache_path) if prepared_cache_path is not None else None,
        "load_seconds": time.perf_counter() - cache_started,
    }
    if not context._prepared_colors:
        started = time.perf_counter()
        prepared: list[Any] = []
        causal_confidences: list[np.ndarray] = []
        spatial_confidences: list[np.ndarray] = []
        spatial_offsets: list[dict[int, float]] = []
        color_path_gate_confidences: list[np.ndarray] = []
        color_path_frame_summaries: list[dict[str, float]] = []
        for image, _injected, _time_h in context._loaded:
            if image is None:
                raise RuntimeError(
                    f"[{context.run}] raw calibration images were released before preparation"
                )
            color_interpretation = context.calibration.call_color_interpretation(
                image
            )
            spatial_support = getattr(context, "spatial_observation_support", None)
            acidification_confidence: Optional[np.ndarray] = None
            if spatial_support is not None:
                confidence, offsets = spatial_support.confidence(
                    btb_acidification_coordinate(image.img),
                    context.calibration.labels.img,
                    remove_frame_offset=True,
                )
                acidification_confidence = np.asarray(confidence, dtype=np.float32)
                spatial_offsets.append(offsets)
            color_path_evidence = getattr(context, "color_path_evidence", None)
            if color_path_evidence is not None:
                evidence = color_path_evidence.evaluate(
                    relative_rgb_for_color_analysis(
                        context.calibration.color_analysis,
                        image,
                    ),
                    context.calibration.labels.img,
                    context.calibration.color_path_interpretation,
                    acidification_confidence=acidification_confidence,
                )
                gate_confidence = np.asarray(
                    evidence.path_support_confidence, dtype=np.float32
                )
                color_path_gate_confidences.append(
                    np.rint(np.clip(gate_confidence, 0.0, 1.0) * 255.0).astype(
                        np.uint8
                    )
                )
                if context.color_path_evidence_mode == "soft-gate":
                    gated_coordinate = context.calibration.color_analysis._restore_signal(
                        evidence.gated_coordinate
                    )
                    color_interpretation.img = np.asarray(
                        gated_coordinate, dtype=np.float32
                    )
                spatial_confidence = evidence.observation_confidence
                residual = np.asarray(
                    evidence.normalized_path_residual, dtype=np.float32
                )
                acid_rescue = (
                    np.maximum(
                        np.asarray(acidification_confidence, dtype=np.float32),
                        gate_confidence,
                    )
                    - gate_confidence
                    if acidification_confidence is not None
                    else np.zeros_like(gate_confidence)
                )
                color_path_frame_summaries.append(
                    {
                        "mean_baseline_change_confidence": float(
                            np.mean(evidence.baseline_change_confidence)
                        ),
                        "mean_path_fit_confidence": float(
                            np.mean(evidence.path_fit_confidence)
                        ),
                        "mean_path_support_confidence": float(
                            np.mean(gate_confidence)
                        ),
                        "mean_observation_confidence": float(
                            np.mean(evidence.observation_confidence)
                        ),
                        "fraction_above_residual_high": float(
                            np.mean(
                                residual
                                > float(color_path_evidence.residual_high)
                            )
                        ),
                        "mean_acidification_rescue": float(np.mean(acid_rescue)),
                    }
                )
            else:
                spatial_confidence = acidification_confidence
            if spatial_confidence is not None:
                spatial_confidences.append(
                    np.rint(
                        np.clip(spatial_confidence, 0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                )
            noise_gate = getattr(context, "aqueous_noise_gate", None)
            if noise_gate is not None:
                if getattr(noise_gate, "mode", "off") == "baseline-causal":
                    causal_confidences.append(
                        noise_gate.confidence_array(
                            color_interpretation.img,
                            context.calibration.labels.img,
                        )
                    )
                color_interpretation = noise_gate.apply_image(
                    color_interpretation,
                    context.calibration.labels.img,
                    copy=False,
                )
            prepared.append(color_interpretation)

        _apply_causal_aqueous_prepared_gate(
            context,
            prepared,
            causal_confidences,
        )
        if requested in {
            "cuda-float32",
            "opencl-float32",
            "mps-metal",
            "metal-float32",
        }:
            # Every qualified GPU backend consumes float32.  Persisting the
            # transient float64 interpretation doubled local cache size and
            # I/O without preserving any value that reached the evaluator.
            for prepared_image in prepared:
                prepared_image.img = np.asarray(
                    prepared_image.img, dtype=np.float32
                )
        timing["parameter_independent_frames_seconds"] = (
            time.perf_counter() - started
        )

        context._prepared_colors = prepared
        context._spatial_observation_confidences = spatial_confidences
        context._color_path_gate_confidences = color_path_gate_confidences
        if spatial_confidences:
            provenance = dict(context.spatial_observation_support_provenance or {})
            provenance.update(
                {
                    "confidence_dtype": "uint8",
                    "frame_count": len(spatial_confidences),
                    "frame_offsets": [
                        {str(label): float(value) for label, value in row.items()}
                        for row in spatial_offsets
                    ],
                }
            )
            context.spatial_observation_support_provenance = provenance
        if color_path_frame_summaries:
            provenance = dict(context.color_path_evidence_provenance or {})
            provenance.update(
                {
                    "gate_confidence_dtype": "uint8",
                    "frame_count": len(color_path_frame_summaries),
                    "frames": color_path_frame_summaries,
                }
            )
            context.color_path_evidence_provenance = provenance
        cache_write_started = time.perf_counter()
        stored_cache_path = _store_prepared_context_cache(context)
        timing["prepared_context_cache"].update(
            {
                "status": "written" if stored_cache_path is not None else "disabled",
                "path": (
                    str(stored_cache_path) if stored_cache_path is not None else None
                ),
                "write_seconds": time.perf_counter() - cache_write_started,
            }
        )
        context._evaluation_backend = "prepared"
        elapsed = time.perf_counter() - started
        logger.info(
            "[%s] prepared %d parameter-independent color frame(s) in %.2fs; "
            "released_raw=%s",
            context.run,
            len(prepared),
            elapsed,
            release_images,
        )
    else:
        context._evaluation_backend = "prepared"

    loaded_rows = list(getattr(context, "_loaded", []) or [])
    if release_images and any(
        image is not None for image, _injected, _time_h in loaded_rows
    ):
        release_started = time.perf_counter()
        context._loaded = [
            (None, injected, time_h)
            for _image, injected, time_h in loaded_rows
        ]
        gc.collect()
        timing["release_raw_images_seconds"] = (
            time.perf_counter() - release_started
        )

    if initialize_backend and requested == "cuda" and context._cuda_evaluator is None:
        backend_started = time.perf_counter()
        try:
            context._cuda_evaluator = _CudaIntegratedEvaluator(context)
        except Exception:
            try:
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            raise
        timing["backend_initialization_seconds"] = (
            time.perf_counter() - backend_started
        )
        context._prepared_colors.clear()
        gc.collect()
    if initialize_backend and requested == "cuda-float32" and getattr(
        context, "_cuda_float32_evaluator", None
    ) is None:
        try:
            from darsia.fluidflower.tools.cuda_float32_integrated_evaluator import (
                CUDAFloat32IntegratedEvaluator,
            )
        except ModuleNotFoundError:
            from darsia.fluidflower.tools.cuda_float32_integrated_evaluator import (
                CUDAFloat32IntegratedEvaluator,
            )

        backend_started = time.perf_counter()
        try:
            context._cuda_float32_evaluator = CUDAFloat32IntegratedEvaluator(
                context,
                residual_score_clip=_RESIDUAL_GAS_SCORE_CLIP,
            )
        except Exception:
            try:
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            raise
        timing["backend_initialization_seconds"] = (
            time.perf_counter() - backend_started
        )
        context._prepared_colors.clear()
        gc.collect()
    if initialize_backend and requested == "opencl" and context._opencl_evaluator is None:
        from darsia.fluidflower.tools.opencl_integrated_evaluator import OpenCLIntegratedEvaluator

        backend_started = time.perf_counter()
        try:
            context._opencl_evaluator = OpenCLIntegratedEvaluator(
                context,
                residual_score_clip=_RESIDUAL_GAS_SCORE_CLIP,
            )
        except Exception:
            gc.collect()
            raise
        timing["backend_initialization_seconds"] = (
            time.perf_counter() - backend_started
        )
        context._prepared_colors.clear()
        gc.collect()
    if initialize_backend and requested == "opencl-float32" and getattr(
        context, "_opencl_float32_evaluator", None
    ) is None:
        backend_started = time.perf_counter()
        try:
            from darsia.fluidflower.tools.opencl_float32_integrated_evaluator import (
                OpenCLFloat32IntegratedEvaluator,
            )
        except ModuleNotFoundError:
            from darsia.fluidflower.tools.opencl_float32_integrated_evaluator import (
                OpenCLFloat32IntegratedEvaluator,
            )

        try:
            context._opencl_float32_evaluator = OpenCLFloat32IntegratedEvaluator(
                context,
                residual_score_clip=_RESIDUAL_GAS_SCORE_CLIP,
            )
        except Exception:
            gc.collect()
            raise
        timing["backend_initialization_seconds"] = (
            time.perf_counter() - backend_started
        )
        context._prepared_colors.clear()
        gc.collect()
    if initialize_backend and requested in {"mps-metal", "metal-float32"} and getattr(
        context, "_metal_evaluator", None
    ) is None:
        backend_started = time.perf_counter()
        try:
            from darsia.fluidflower.tools.mps_metal_integrated_evaluator import (
                MPSMetalIntegratedEvaluator,
            )
        except ModuleNotFoundError:
            from darsia.fluidflower.tools.mps_metal_integrated_evaluator import MPSMetalIntegratedEvaluator

        try:
            context._metal_evaluator = MPSMetalIntegratedEvaluator(
                context,
                residual_score_clip=_RESIDUAL_GAS_SCORE_CLIP,
            )
        except Exception:
            gc.collect()
            try:
                import torch

                torch.mps.empty_cache()
            except Exception:
                pass
            raise
        timing["backend_initialization_seconds"] = (
            time.perf_counter() - backend_started
        )
        context._prepared_colors.clear()
        gc.collect()
    if initialize_backend:
        context._evaluation_backend = requested
        context._deferred_evaluation_backend = None
    else:
        context._evaluation_backend = "prepared"
        context._deferred_evaluation_backend = requested
        timing["backend_initialization_deferred"] = True
    timing["total_seconds"] = time.perf_counter() - call_started
    context._prepare_evaluation_timing = timing
    return context


def _mass_result_for_evaluation(
    context: CalibrationContext,
    image: Any,
    image_index: int,
    params: Dict[str, Any],
) -> Any:
    mode = _normalise_phase_separation(
        getattr(context, "phase_separation", "shared-signal")
    )

    def flash_and_mass(pH: Any) -> Any:
        if mode == "shared-signal":
            return call_flash_and_mass_with_constraints(context.calibration, pH)
        # Independent phase modes apply aqueous/gas expert weights below, after
        # replacing or gating the Flash gas field. Starting from a weighted
        # field here would square the soft gas mask on CPU only.
        concentration_aq, saturation_g = context.calibration.flash(pH)
        return context.calibration.co2_mass_analysis.mass_analysis(
            c_aq=concentration_aq,
            s_g=saturation_g,
        )

    if image_index < len(getattr(context, "_prepared_colors", [])):
        color_interpretation = context._prepared_colors[image_index]
        pH = context.calibration.call_pH_analysis(color_interpretation)
        result = flash_and_mass(pH)
    else:
        noise_gate = getattr(context, "aqueous_noise_gate", None)
        if (
            mode == "shared-signal"
            and noise_gate is None
            and getattr(context, "color_path_evidence", None) is None
            and constraint_settings_from_env().mode == "hard"
        ):
            result = context.calibration(image)
        else:
            color_interpretation = context.calibration.call_color_interpretation(image)
            color_path_evidence = getattr(context, "color_path_evidence", None)
            if color_path_evidence is not None:
                evidence = color_path_evidence.evaluate(
                    relative_rgb_for_color_analysis(
                        context.calibration.color_analysis,
                        image,
                    ),
                    context.calibration.labels.img,
                    context.calibration.color_path_interpretation,
                )
                if context.color_path_evidence_mode == "soft-gate":
                    color_interpretation.img = np.asarray(
                        context.calibration.color_analysis._restore_signal(
                            evidence.gated_coordinate
                        ),
                        dtype=np.float32,
                    )
            if noise_gate is not None:
                color_interpretation = noise_gate.apply_image(
                    color_interpretation,
                    context.calibration.labels.img,
                    copy=False,
                )
            pH = context.calibration.call_pH_analysis(color_interpretation)
            result = flash_and_mass(pH)
    hard_nonseal = getattr(context, "_hard_nonseal_gas_mask", None)
    if mode == "shared-signal" and hard_nonseal is None:
        return result
    if mode == "shared-signal":
        concentration_aq = np.asarray(result.concentration_aq.img, dtype=np.float32)
        saturation_g = np.asarray(result.saturation_g.img, dtype=np.float32)
        hard_nonseal_array = np.asarray(hard_nonseal, dtype=np.float32)
        if hard_nonseal_array.shape != saturation_g.shape:
            raise RuntimeError(
                f"Hard non-seal mask shape {hard_nonseal_array.shape} "
                f"!= saturation shape {saturation_g.shape}"
            )
        saturation_g = saturation_g * hard_nonseal_array
    else:
        constraint_weights = constraint_weights_for_calibration(
            context.calibration,
            pH,
        )
        concentration_aq = (
            np.asarray(result.concentration_aq.img, dtype=np.float32)
            * constraint_weights.aqueous
        )
        if mode == "hybrid-causal":
            if image_index >= len(context._hybrid_gas_weights):
                raise IndexError(
                    f"Missing hybrid gas weight {image_index} for {context.run}"
                )
            optical_weight = (
                np.asarray(
                    context._hybrid_gas_weights[image_index], dtype=np.float32
                )
                / np.float32(255.0)
            )
            nonseal = np.asarray(context._hybrid_nonseal_mask, dtype=np.float32)
            shared_weight = hybrid_shared_signal_weight(
                float(context._loaded[image_index][2]),
                injection_end_h=float(context.hybrid_injection_end_h),
                half_life_h=float(
                    params.get(
                        "gas.hybrid_half_life_h", DEFAULT_HYBRID_HALF_LIFE_H
                    )
                ),
            )
            hybrid_weight = shared_weight + (1.0 - shared_weight) * optical_weight
            saturation_g = (
                np.asarray(result.saturation_g.img, dtype=np.float32)
                * hybrid_weight
                * np.asarray(constraint_weights.gas, dtype=np.float32)
                * nonseal
            )
        else:
            if image_index >= len(context._gas_scores):
                raise IndexError(
                    f"Missing phase-separation score {image_index} for {context.run}"
                )
            score = (
                np.asarray(context._gas_scores[image_index], dtype=np.float32)
                * (
                    float(
                        getattr(
                            context,
                            "_gas_score_scale",
                            _RESIDUAL_GAS_SCORE_CLIP
                            if mode == "residual-gas"
                            else 1.0,
                        )
                    )
                    / 255.0
                )
            )
            onset, width = _phase_score_parameters(params, mode)
            score_gate = np.clip((score - onset) / width, 0.0, 1.0)

        if mode == "clearwater-gated":
            saturation_g = (
                np.asarray(result.saturation_g.img, dtype=np.float32) * score_gate
            )
        elif mode == "supervised-dual":
            # This is deliberately independent of the aqueous/BTB arrival
            # signal. The only coupling happens below through pore occupancy.
            saturation_g = score_gate * np.float32(
                np.clip(
                    float(params.get("gas.dual_max_saturation", 0.50)),
                    0.0,
                    1.0,
                )
            )
        elif mode != "hybrid-causal":
            # Independent scores can only produce gas where the indicator also
            # records CO2 arrival. This rejects fixed wash-like optical defects.
            arrival_scale = max(
                1e-6,
                float(
                    os.environ.get(
                        "FFAC_PHASE_GAS_ARRIVAL_SCALE",
                        os.environ.get(
                            "FFAC_RESIDUAL_GAS_ARRIVAL_SCALE", "0.02"
                        ),
                    )
                ),
            )
            saturation_g = score_gate * np.clip(
                concentration_aq / arrival_scale,
                0.0,
                1.0,
            )
        if mode.startswith("clearwater-") or mode == "supervised-dual":
            saturation_g *= constraint_weights.gas
        if hard_nonseal is not None:
            saturation_g *= np.asarray(hard_nonseal, dtype=np.float32)

    mass_analysis = context.calibration.co2_mass_analysis
    density = np.asarray(mass_analysis.density_gaseous_co2)
    solubility = np.asarray(mass_analysis.solubility_co2)
    mass_g = density * saturation_g
    mass_aq = solubility * concentration_aq * np.clip(
        1.0 - saturation_g,
        0.0,
        1.0,
    )
    result.saturation_g.img = saturation_g
    result.mass_g.img = mass_g
    result.mass_aq.img = mass_aq
    result.mass.img = mass_g + mass_aq
    return result


def summarize_spatial_objective(
    metrics: Mapping[str, Metrics],
    *,
    recall_weight: float,
    seal_weight: float,
) -> Dict[str, float]:
    """Reduce parameter-independent image support into mass-scaled diagnostics."""

    rows = list(metrics.values())
    required = (
        "aqueous_low_observation_mass_kg",
        "aqueous_observation_recall_deficit",
        "aqueous_unsupported_seal_mass_kg",
        "aqueous_low_observation_support_volume_m3",
        "aqueous_predicted_support_volume_m3",
    )
    complete = [
        row for row in rows
        if all(getattr(row, name) is not None for name in required)
    ]
    if not complete:
        return {
            "spatial_frames": 0.0,
            "spatial_low_observation_mass_kg": 0.0,
            "spatial_low_observation_co2_mass_kg": 0.0,
            "spatial_recall_equivalent_mass_kg": 0.0,
            "spatial_unsupported_seal_mass_kg": 0.0,
            "spatial_unsupported_seal_co2_mass_kg": 0.0,
            "spatial_low_observation_support_volume_m3": 0.0,
            "spatial_predicted_support_volume_m3": 0.0,
            "spatial_low_observation_support_fraction": 0.0,
            "spatial_raw_penalty_mass_kg": 0.0,
        }
    if len(complete) != len(rows):
        raise RuntimeError("Spatial diagnostics are missing for one or more frames")

    low_mass = sum(
        max(
            0.0,
            float(
                row.co2_low_observation_mass_kg
                if row.co2_low_observation_mass_kg is not None
                else row.aqueous_low_observation_mass_kg or 0.0
            ),
        )
        for row in complete
    )
    recall_equivalent_mass = sum(
        max(0.0, float(row.injected_full))
        * min(
            1.0,
            max(0.0, float(row.aqueous_observation_recall_deficit or 0.0)),
        )
        for row in complete
    )
    unsupported_seal_mass = sum(
        max(
            0.0,
            float(
                row.co2_unsupported_seal_mass_kg
                if row.co2_unsupported_seal_mass_kg is not None
                else row.aqueous_unsupported_seal_mass_kg or 0.0
            ),
        )
        for row in complete
    )
    low_support_volume = sum(
        max(
            0.0,
            float(row.aqueous_low_observation_support_volume_m3 or 0.0),
        )
        for row in complete
    )
    predicted_support_volume = sum(
        max(0.0, float(row.aqueous_predicted_support_volume_m3 or 0.0))
        for row in complete
    )
    low_support_fraction = (
        low_support_volume / predicted_support_volume
        if predicted_support_volume > 0.0
        else 0.0
    )
    raw_penalty = (
        low_mass
        + float(recall_weight) * recall_equivalent_mass
        + float(seal_weight) * unsupported_seal_mass
    )
    return {
        "spatial_frames": float(len(complete)),
        "spatial_low_observation_mass_kg": float(low_mass),
        "spatial_low_observation_co2_mass_kg": float(low_mass),
        "spatial_recall_equivalent_mass_kg": float(recall_equivalent_mass),
        "spatial_unsupported_seal_mass_kg": float(unsupported_seal_mass),
        "spatial_unsupported_seal_co2_mass_kg": float(unsupported_seal_mass),
        "spatial_low_observation_support_volume_m3": float(low_support_volume),
        "spatial_predicted_support_volume_m3": float(predicted_support_volume),
        "spatial_low_observation_support_fraction": float(low_support_fraction),
        "spatial_raw_penalty_mass_kg": float(raw_penalty),
    }


def summarize_layer_gain_regularization(
    params: Mapping[str, Any],
    label_to_facies: Mapping[int, int],
    *,
    sigma: float,
) -> Dict[str, float]:
    """Measure layer-gain freedom around each material's geometric mean.

    Relative deviations within a facies carry the full penalty. A weaker prior
    keeps the facies-level geometric mean near one without suppressing a real,
    run-wide photometric gain.
    """

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Layer-gain regularization sigma must be positive")
    grouped: Dict[int, List[float]] = {}
    for name, raw_value in params.items():
        match = _SIGNAL_GAIN_RE.match(str(name))
        if match is None:
            continue
        label = int(match.group("label"))
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid layer gain {name}={raw_value!r}")
        facies = int(label_to_facies.get(label, label))
        grouped.setdefault(facies, []).append(float(np.log(value)))

    if not grouped:
        return {
            "layer_gain_count": 0.0,
            "layer_gain_facies_count": 0.0,
            "layer_gain_within_facies_z2": 0.0,
            "layer_gain_center_z2": 0.0,
            "layer_gain_regularization_z2": 0.0,
            "layer_gain_min": 1.0,
            "layer_gain_max": 1.0,
        }

    within_residuals: List[float] = []
    center_residuals: List[float] = []
    all_log_gains: List[float] = []
    for values in grouped.values():
        center = float(np.mean(values))
        all_log_gains.extend(values)
        within_residuals.extend(
            ((value - center) / float(sigma)) ** 2 for value in values
        )
        # A two-sigma width makes the absolute gain prior deliberately weaker
        # than the same-material consistency prior.
        center_residuals.append((center / (2.0 * float(sigma))) ** 2)
    within_z2 = float(np.mean(within_residuals)) if within_residuals else 0.0
    center_z2 = float(np.mean(center_residuals)) if center_residuals else 0.0
    combined_z2 = within_z2 + 0.25 * center_z2
    gains = np.exp(np.asarray(all_log_gains, dtype=np.float64))
    return {
        "layer_gain_count": float(gains.size),
        "layer_gain_facies_count": float(len(grouped)),
        "layer_gain_within_facies_z2": within_z2,
        "layer_gain_center_z2": center_z2,
        "layer_gain_regularization_z2": float(combined_z2),
        "layer_gain_min": float(np.min(gains)),
        "layer_gain_max": float(np.max(gains)),
    }


def _integrated_evaluator_for_context(context: CalibrationContext) -> Any:
    """Return the accelerator selected by the context's backend contract."""

    requested = getattr(context, "_evaluation_backend", "legacy")
    if requested == "cuda-float32":
        return getattr(context, "_cuda_float32_evaluator", None)
    if requested == "opencl-float32":
        return getattr(context, "_opencl_float32_evaluator", None)
    if requested in {"mps-metal", "metal-float32"}:
        return getattr(context, "_metal_evaluator", None)
    if requested == "cuda":
        return getattr(context, "_cuda_evaluator", None)
    if requested == "opencl":
        return getattr(context, "_opencl_evaluator", None)
    evaluator = getattr(context, "_cuda_evaluator", None)
    if evaluator is None:
        evaluator = getattr(context, "_opencl_evaluator", None)
    if evaluator is None:
        evaluator = getattr(context, "_metal_evaluator", None)
    return evaluator


def _evaluate_run_impl(context: CalibrationContext, params: Dict[str, Any]) -> EvalResult:
    import numpy as np
    if not context._loaded:
        return EvalResult(
            objective=PENALTY_VALUE,
            feasible=False,
            metrics={},
            status="no-calibration-images",
            params=params,
        )
    apply_params(context.calibration, params, labels=context.signal_labels, np_module=np)
    total_err = 0.0
    feasible = True
    metrics: Dict[str, Metrics] = {}
    samples: List[Tuple[float, float, float]] = []  # (t_hours, injected, detected)
    phase_samples: List[
        Tuple[float, float, Optional[float], Optional[float]]
    ] = []
    integrated_masses: Optional[List[Any]] = None
    diffusion_samples: List[
        Tuple[float, float, float, float, float]
    ] = []  # time, cmax, peak-equivalent, threshold volume, effective volume
    aqueous_diffusion_weight = float(
        getattr(context, "aqueous_diffusion_penalty_weight", 0.0) or 0.0
    )
    aqueous_diffusion_active = aqueous_diffusion_weight > 0.0
    spatial_observation_active = bool(
        getattr(context, "_spatial_observation_confidences", [])
    )
    integrated_evaluator = _integrated_evaluator_for_context(context)
    if integrated_evaluator is not None:
        try:
            if spatial_observation_active and not bool(
                getattr(integrated_evaluator, "spatial_objective_active", False)
            ):
                raise RuntimeError(
                    "selected integrated backend does not support spatial diagnostics"
                )
            if spatial_observation_active:
                integrated_masses = integrated_evaluator.evaluate(
                    params,
                    include_distribution=aqueous_diffusion_active,
                    include_spatial=True,
                )
            elif aqueous_diffusion_active:
                integrated_masses = integrated_evaluator.evaluate(
                    params,
                    include_distribution=True,
                )
            else:
                integrated_masses = integrated_evaluator.evaluate(params)
        except Exception as exc:  # noqa: BLE001
            backend = getattr(context, "_evaluation_backend", "integrated")
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics={},
                status=f"eval-error:{backend}:{exc}",
                params=params,
            )
    for i, (img, injected, t_h) in enumerate(context._loaded):
        try:
            if integrated_masses is not None:
                integrated = integrated_masses[i]
                if isinstance(integrated, IntegratedFrameResult):
                    detected = integrated.total_mass
                    detected_g = integrated.gaseous_mass
                    detected_aq = integrated.aqueous_mass
                    aqueous_cmax = integrated.aqueous_concentration_max
                    aqueous_peak_equivalent = (
                        integrated.aqueous_concentration_peak_equivalent
                    )
                    aqueous_affected_volume = (
                        integrated.aqueous_affected_pore_volume_m3
                    )
                    aqueous_effective_volume = (
                        integrated.aqueous_effective_pore_volume_m3
                    )
                    aqueous_low_observation_mass = (
                        integrated.aqueous_low_observation_mass_kg
                    )
                    co2_low_observation_mass = (
                        integrated.co2_low_observation_mass_kg
                    )
                    aqueous_observation_recall_deficit = (
                        integrated.aqueous_observation_recall_deficit
                    )
                    aqueous_unsupported_seal_mass = (
                        integrated.aqueous_unsupported_seal_mass_kg
                    )
                    co2_unsupported_seal_mass = (
                        integrated.co2_unsupported_seal_mass_kg
                    )
                    aqueous_low_observation_support_volume = (
                        integrated.aqueous_low_observation_support_volume_m3
                    )
                    aqueous_predicted_support_volume = (
                        integrated.aqueous_predicted_support_volume_m3
                    )
                else:
                    detected, detected_g, detected_aq = integrated
                    aqueous_cmax = None
                    aqueous_peak_equivalent = None
                    aqueous_affected_volume = None
                    aqueous_effective_volume = None
                    aqueous_low_observation_mass = None
                    co2_low_observation_mass = None
                    aqueous_observation_recall_deficit = None
                    aqueous_unsupported_seal_mass = None
                    co2_unsupported_seal_mass = None
                    aqueous_low_observation_support_volume = None
                    aqueous_predicted_support_volume = None
            else:
                mass_result = _mass_result_for_evaluation(context, img, i, params)
                detected = integrate_mass_with_depth_model(
                    context,
                    mass_result.mass,
                    params,
                )
                detected_g = (
                    integrate_mass_with_depth_model(
                        context,
                        mass_result.mass_g,
                        params,
                    )
                    if getattr(mass_result, "mass_g", None) is not None
                    else None
                )
                aqueous_cmax = None
                aqueous_peak_equivalent = None
                aqueous_affected_volume = None
                aqueous_effective_volume = None
                aqueous_low_observation_mass = None
                co2_low_observation_mass = None
                aqueous_observation_recall_deficit = None
                aqueous_unsupported_seal_mass = None
                co2_unsupported_seal_mass = None
                aqueous_low_observation_support_volume = None
                aqueous_predicted_support_volume = None
                if aqueous_diffusion_active:
                    concentration_aq = np.asarray(
                        mass_result.concentration_aq.img, dtype=np.float64
                    )
                    saturation_g = np.asarray(
                        mass_result.saturation_g.img, dtype=np.float64
                    )
                    voxel_volume = voxel_volume_with_depth_model(
                        context, params, concentration_aq.shape
                    )
                    aqueous_pore_volume = voxel_volume * np.clip(
                        1.0 - saturation_g, 0.0, None
                    )
                    aqueous_cmax = float(np.max(concentration_aq))
                    aqueous_c_volume = float(
                        np.sum(
                            aqueous_pore_volume * concentration_aq,
                            dtype=np.float64,
                        )
                    )
                    aqueous_peak_equivalent = (
                        float(
                            np.sum(
                                aqueous_pore_volume
                                * concentration_aq
                                * concentration_aq,
                                dtype=np.float64,
                            )
                            / aqueous_c_volume
                        )
                        if aqueous_c_volume > 0.0
                        else 0.0
                    )
                    aqueous_c2_volume = float(
                        np.sum(
                            aqueous_pore_volume
                            * concentration_aq
                            * concentration_aq,
                            dtype=np.float64,
                        )
                    )
                    aqueous_effective_volume = (
                        float(aqueous_c_volume * aqueous_c_volume / aqueous_c2_volume)
                        if aqueous_c2_volume > 0.0
                        else 0.0
                    )
                    support_level = float(
                        getattr(
                            context,
                            "aqueous_diffusion_support_level",
                            DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
                        )
                    )
                    aqueous_affected_volume = float(
                        np.sum(
                            aqueous_pore_volume
                            * (concentration_aq >= support_level),
                            dtype=np.float64,
                        )
                    )
                detected_aq = (
                    integrate_mass_with_depth_model(
                        context,
                        mass_result.mass_aq,
                        params,
                    )
                    if getattr(mass_result, "mass_aq", None) is not None
                    else None
                )
                if spatial_observation_active:
                    concentration_aq = np.asarray(
                        mass_result.concentration_aq.img, dtype=np.float64
                    )
                    saturation_g = np.asarray(
                        mass_result.saturation_g.img, dtype=np.float64
                    )
                    voxel_volume = voxel_volume_with_depth_model(
                        context, params, concentration_aq.shape
                    )
                    confidence = np.asarray(
                        context._spatial_observation_confidences[i],
                        dtype=np.float64,
                    ) / 255.0
                    unsupported = np.clip(1.0 - confidence, 0.0, 1.0)
                    extensive_aqueous = (
                        np.asarray(mass_result.mass_aq.img, dtype=np.float64)
                        * voxel_volume
                    )
                    extensive_total = (
                        np.asarray(mass_result.mass.img, dtype=np.float64)
                        * voxel_volume
                    )
                    aqueous_pore_volume = voxel_volume * np.clip(
                        1.0 - saturation_g, 0.0, None
                    )
                    seal_ids = [
                        int(value.strip())
                        for value in os.environ.get(
                            "FFAC_SPATIAL_SEAL_FACIES_IDS", "1"
                        ).split(",")
                        if value.strip()
                    ]
                    seal = np.isin(
                        np.asarray(context.fluidflower.facies.img), seal_ids
                    )
                    predicted_support = (
                        concentration_aq >= context.spatial_trace_level
                    )
                    observation_weight = aqueous_pore_volume * confidence
                    observation_denominator = float(
                        np.sum(observation_weight, dtype=np.float64)
                    )
                    aqueous_low_observation_mass = float(
                        np.sum(extensive_aqueous * unsupported, dtype=np.float64)
                    )
                    co2_low_observation_mass = float(
                        np.sum(extensive_total * unsupported, dtype=np.float64)
                    )
                    aqueous_unsupported_seal_mass = float(
                        np.sum(
                            extensive_aqueous * unsupported * seal,
                            dtype=np.float64,
                        )
                    )
                    co2_unsupported_seal_mass = float(
                        np.sum(
                            extensive_total * unsupported * seal,
                            dtype=np.float64,
                        )
                    )
                    aqueous_low_observation_support_volume = float(
                        np.sum(
                            aqueous_pore_volume * predicted_support * unsupported,
                            dtype=np.float64,
                        )
                    )
                    aqueous_predicted_support_volume = float(
                        np.sum(
                            aqueous_pore_volume * predicted_support,
                            dtype=np.float64,
                        )
                    )
                    aqueous_observation_recall_deficit = (
                        float(
                            np.sum(
                                observation_weight
                                * np.clip(
                                    (
                                        context.spatial_core_level
                                        - concentration_aq
                                    )
                                    / context.spatial_core_level,
                                    0.0,
                                    1.0,
                                ),
                                dtype=np.float64,
                            )
                            / observation_denominator
                        )
                        if observation_denominator > 0.0
                        else 0.0
                    )
        except Exception as exc:  # noqa: BLE001
            # One-shot full traceback to stdout so the EXACT file:line of a shape/broadcast
            # mismatch is visible (the status string only carries the message). Also dumps
            # geometry weight shapes, the remaining un-coarsened suspect under quality_scale.
            global _EVAL_TB_PRINTED
            if not _EVAL_TB_PRINTED:
                _EVAL_TB_PRINTED = True
                import traceback as _tb
                _tbstr = _tb.format_exc()   # capture ORIGINAL traceback BEFORE any re-call
                try:
                    _gv = getattr(context.geometry, "voxel_volume", None)
                    _cv = getattr(context.geometry, "cached_voxel_volume", None)
                    _ish = getattr(getattr(img, "img", None), "shape", None)
                    print(f"[EVALTB {context.run}] {type(exc).__name__}: {exc}\n"
                          f"  img={_ish} geometry={type(context.geometry).__name__} "
                          f"voxel_volume={getattr(_gv,'shape',None)} "
                          f"cached={getattr(_cv,'shape',None)}", flush=True)
                except Exception as _e2:
                    print(f"[EVALTB {context.run}] shape probe failed: {_e2}", flush=True)
                print(_tbstr, flush=True)   # the traceback pinpoints the exact file:line
            return EvalResult(objective=PENALTY_VALUE, feasible=False, metrics={},
                              status=f"eval-error:{exc}", params=params)
        if not (np.isfinite(detected) and np.isfinite(injected)):
            # extreme signal values can make the flash/mass field non-finite;
            # penalise so Optuna avoids this region rather than crashing on NaN.
            return EvalResult(objective=PENALTY_VALUE, feasible=False, metrics=metrics,
                              status="non-finite-mass", params=params)
        total_err += abs(detected - injected)
        if context.enforce_lower and detected > injected:
            feasible = False
        # key by time-since-start in hours so the calibration viewer plots a real
        # time axis (0.17h .. 48h), not image indices.
        metrics[f"{t_h:.3f}h"] = Metrics(
            injected_full=injected,
            total_full=detected,
            gaseous_full=detected_g,
            aqueous_full=detected_aq,
            aqueous_concentration_max=aqueous_cmax,
            aqueous_concentration_peak_equivalent=aqueous_peak_equivalent,
            aqueous_affected_pore_volume_m3=aqueous_affected_volume,
            aqueous_effective_pore_volume_m3=aqueous_effective_volume,
            aqueous_low_observation_mass_kg=aqueous_low_observation_mass,
            co2_low_observation_mass_kg=co2_low_observation_mass,
            aqueous_observation_recall_deficit=(
                aqueous_observation_recall_deficit
            ),
            aqueous_unsupported_seal_mass_kg=aqueous_unsupported_seal_mass,
            co2_unsupported_seal_mass_kg=co2_unsupported_seal_mass,
            aqueous_low_observation_support_volume_m3=(
                aqueous_low_observation_support_volume
            ),
            aqueous_predicted_support_volume_m3=(
                aqueous_predicted_support_volume
            ),
        )
        samples.append((float(t_h), float(injected), float(detected)))
        phase_samples.append(
            (
                float(t_h),
                float(injected),
                float(detected_g) if detected_g is not None else None,
                float(detected_aq) if detected_aq is not None else None,
            )
        )
        if (
            aqueous_cmax is not None
            and aqueous_peak_equivalent is not None
            and aqueous_affected_volume is not None
            and aqueous_effective_volume is not None
        ):
            diffusion_samples.append(
                (
                    float(t_h),
                    float(aqueous_cmax),
                    float(aqueous_peak_equivalent),
                    float(aqueous_affected_volume),
                    float(aqueous_effective_volume),
                )
            )
    # Mass-fit modes may be combined, for example
    # --objective-integral window-balanced+drift:5.
    # Physics: after shut-in the cell is closed, so TRUE total mass is constant. But the BTB
    # indicator saturates at ~11 % of CO2 solubility (1.25 mM NaOH alkalinity + 0.73 mM BTB:
    # fully yellow at DIC ~3.9 mM vs ~34 mM saturation), so the dilute fringe of the dissolved
    # plume reads as saturated and detected mass tracks plume AREA, which keeps growing after
    # shut-in. Penalising the temporal total variation of detected mass over the post-injection
    # plateau rewards signal->c_aq maps that put their weight at the END of the colour path
    # (where the fringe contributes ~0), i.e. a FLAT detected-mass curve at the right level
    # rather than a drifting curve that is merely right on average. Metrics stay raw.
    mode = str(
        getattr(context, "objective_integral", "off") or "off"
    ).strip().lower()
    pointwise_mass_objective = float(total_err)
    window_balanced_mass_objective = 0.0
    window_balanced_unscaled_mass_objective = 0.0
    window_balanced_scale_factor = 1.0
    plateau_drift_mass = 0.0
    plateau_drift_penalty = 0.0
    use_window_balance = any(
        token in mode
        for token in ("window-balanced", "balanced-windows", "windows")
    )
    if use_window_balance:
        windows: dict[str, list[float]] = {
            "I1": [],
            "I2": [],
            "early": [],
            "late": [],
        }
        for time_h, injected, detected in samples:
            if time_h <= 0.92:
                key = "I1"
            elif time_h < 4.0:
                key = "I2"
            # Requested 4.1/8 h frames can resolve a few minutes to either side.
            # These tolerant cutoffs preserve their intended operational window.
            elif time_h <= 8.25:
                key = "early"
            else:
                key = "late"
            windows[key].append(abs(detected - injected))
        if any(not values for values in windows.values()):
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="missing-objective-window",
                params=params,
            )
        # Each operational window contributes one mean absolute error, regardless
        # of how many calibration frames happen to lie inside that window.
        window_balanced_unscaled_mass_objective = sum(
            float(np.mean(values)) for values in windows.values()
        )
        window_balanced_mass_objective = (
            window_balanced_unscaled_mass_objective
        )
        if "window-balanced-scaled" in mode:
            # Equalise the four operational windows while preserving the
            # point-wise L1 scale when every frame has the same error. This
            # keeps existing physics-penalty weights comparable across modes.
            window_balanced_scale_factor = len(samples) / len(windows)
            window_balanced_mass_objective *= window_balanced_scale_factor
        total_err = window_balanced_mass_objective
    if "drift" in mode:
        lam = 1.0
        drift_match = re.search(
            r"drift(?:[:=]([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?))?",
            mode,
        )
        if drift_match is not None and drift_match.group(1) is not None:
            try:
                lam = float(drift_match.group(1))
            except ValueError:
                lam = 1.0
        if lam < 0.0 or not np.isfinite(lam):
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="invalid-drift-weight",
                params=params,
            )
        pts = sorted(samples)  # chronological
        if pts:
            inj_max = max(p[1] for p in pts)
            plateau = [p for p in pts if p[1] >= inj_max * (1.0 - 1e-9)]
            if len(plateau) >= 2:
                plateau_drift_mass = sum(
                    abs(b[2] - a[2])
                    for a, b in zip(plateau, plateau[1:])
                )
                plateau_drift_penalty = lam * plateau_drift_mass
                total_err += plateau_drift_penalty
    mass_objective = float(total_err)

    # After injection, the closed-cell total mass should not drift between
    # calibration frames. This is separate from the late gas cap: aqueous mass
    # may increase, but only while gas mass decreases by the same amount.
    post_injection_balance_weight = float(
        getattr(
            context,
            "post_injection_balance_weight",
            DEFAULT_POST_INJECTION_BALANCE_WEIGHT,
        )
        or 0.0
    )
    post_injection_balance_start_h = float(
        getattr(
            context,
            "post_injection_balance_start_h",
            DEFAULT_POST_INJECTION_BALANCE_START_H,
        )
    )
    post_injection_total_drift_mass = 0.0
    post_injection_phase_direction_mass = 0.0
    post_injection_balance_pairs = 0
    if post_injection_balance_weight > 0.0:
        post_injection = sorted(
            sample for sample in phase_samples
            if sample[0] >= post_injection_balance_start_h
        )
        for previous, current in zip(post_injection, post_injection[1:]):
            if any(value is None for value in (*previous[2:], *current[2:])):
                return EvalResult(
                    objective=PENALTY_VALUE,
                    feasible=False,
                    metrics=metrics,
                    status="post-injection-phase-mass-unavailable",
                    params=params,
                    objective_terms={"mass_objective": mass_objective},
                )
            previous_total = float(previous[2]) + float(previous[3])
            current_total = float(current[2]) + float(current[3])
            post_injection_total_drift_mass += abs(
                current_total - previous_total
            )
            post_injection_phase_direction_mass += max(
                0.0, float(current[2]) - float(previous[2])
            )
            post_injection_phase_direction_mass += max(
                0.0, float(previous[3]) - float(current[3])
            )
            post_injection_balance_pairs += 1
        if post_injection_balance_pairs == 0:
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="missing-post-injection-balance-window",
                params=params,
                objective_terms={"mass_objective": mass_objective},
            )
    post_injection_balance_violation_mass = (
        post_injection_total_drift_mass
        + post_injection_phase_direction_mass
    )
    post_injection_balance_penalty = (
        post_injection_balance_weight
        * post_injection_balance_violation_mass
    )
    total_err += post_injection_balance_penalty

    # After injection, dissolution and dispersion should redistribute aqueous
    # CO2 from a concentrated core into a larger pore volume. The effective
    # volume (integral(c)^2 / integral(c^2)) is mass-normalized and therefore
    # does not mistake a larger total detected mass for stronger diffusion.
    # The threshold volume remains diagnostic but does not drive the objective.
    aqueous_diffusion_penalty = 0.0
    aqueous_diffusion_violation_mass = 0.0
    aqueous_diffusion_max_increase_relative = 0.0
    aqueous_diffusion_peak_increase_relative = 0.0
    aqueous_diffusion_volume_decrease_relative = 0.0
    aqueous_diffusion_effective_volume_decrease_relative = 0.0
    aqueous_diffusion_pairs = 0
    if aqueous_diffusion_weight > 0.0:
        start_h = float(
            getattr(
                context,
                "aqueous_diffusion_start_h",
                DEFAULT_AQUEOUS_DIFFUSION_START_H,
            )
        )
        if len(diffusion_samples) != len(phase_samples):
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="aqueous-diffusion-statistics-unavailable",
                params=params,
                objective_terms={"mass_objective": mass_objective},
            )
        post_injection = sorted(
            sample for sample in diffusion_samples if sample[0] >= start_h
        )
        if len(post_injection) < 2:
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="missing-aqueous-diffusion-window",
                params=params,
                objective_terms={"mass_objective": mass_objective},
            )
        injected_scale = max(
            (max(0.0, float(sample[1])) for sample in phase_samples),
            default=1.0,
        )
        for previous, current in zip(post_injection, post_injection[1:]):
            if float(current[0]) - float(previous[0]) < float(
                getattr(
                    context,
                    "aqueous_diffusion_pair_min_dt_h",
                    DEFAULT_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H,
                )
            ):
                continue
            previous_maximum = max(float(previous[1]), 1e-12)
            previous_peak = max(float(previous[2]), 1e-12)
            previous_threshold_volume = max(float(previous[3]), 1e-18)
            previous_effective_volume = max(float(previous[4]), 1e-18)
            maximum_increase = max(
                0.0,
                (float(current[1]) - float(previous[1])) / previous_maximum,
            )
            peak_increase = max(
                0.0,
                (float(current[2]) - float(previous[2])) / previous_peak,
            )
            threshold_volume_decrease = max(
                0.0,
                (float(previous[3]) - float(current[3]))
                / previous_threshold_volume,
            )
            effective_volume_decrease = max(
                0.0,
                (float(previous[4]) - float(current[4]))
                / previous_effective_volume,
            )
            aqueous_diffusion_max_increase_relative += maximum_increase
            aqueous_diffusion_peak_increase_relative += peak_increase
            aqueous_diffusion_volume_decrease_relative += (
                threshold_volume_decrease
            )
            aqueous_diffusion_effective_volume_decrease_relative += (
                effective_volume_decrease
            )
            aqueous_diffusion_pairs += 1
        aqueous_diffusion_violation_mass = injected_scale * (
            max(
                aqueous_diffusion_max_increase_relative,
                aqueous_diffusion_peak_increase_relative,
            )
            + max(
                aqueous_diffusion_effective_volume_decrease_relative,
                0.0,
            )
        )
        aqueous_diffusion_penalty = (
            aqueous_diffusion_weight * aqueous_diffusion_violation_mass
        )
        total_err += aqueous_diffusion_penalty

    # A late free-gas cap is independent of the mass-fitting mode above. The cap
    # is referenced to injected mass, rather than detected total mass, so an
    # optimizer cannot reduce the denominator by underdetecting the whole plume.
    late_gas_penalty = 0.0
    late_gas_excess_mass = 0.0
    late_aqueous_shortfall_mass = 0.0
    late_phase_violation_mass = 0.0
    late_gas_frames = 0
    late_gas_allowed_fractions: list[float] = []
    late_gas_weight = float(
        getattr(context, "late_gas_penalty_weight", 0.0) or 0.0
    )
    late_gas_constraint_mode = resolve_late_gas_constraint_mode(
        getattr(
            context,
            "late_gas_constraint_mode",
            DEFAULT_LATE_GAS_CONSTRAINT_MODE,
        )
    )
    if late_gas_weight > 0.0:
        late_gas_start_h = float(
            getattr(context, "late_gas_start_h", DEFAULT_LATE_GAS_START_H)
        )
        late_gas_max_fraction = float(
            getattr(
                context,
                "late_gas_max_fraction",
                DEFAULT_LATE_GAS_MAX_FRACTION,
            )
        )
        late_gas_decay_half_life_h = float(
            getattr(
                context,
                "late_gas_decay_half_life_h",
                DEFAULT_LATE_GAS_DECAY_HALF_LIFE_H,
            )
            or 0.0
        )
        late_gas_floor_fraction = float(
            getattr(
                context,
                "late_gas_floor_fraction",
                DEFAULT_LATE_GAS_FLOOR_FRACTION,
            )
            or 0.0
        )
        for time_h, injected, detected_g, detected_aq in phase_samples:
            if time_h < late_gas_start_h:
                continue
            late_gas_frames += 1
            if (
                detected_g is None
                or detected_aq is None
                or not np.isfinite(detected_g)
                or not np.isfinite(detected_aq)
            ):
                return EvalResult(
                    objective=PENALTY_VALUE,
                    feasible=False,
                    metrics=metrics,
                    status="late-phase-mass-unavailable",
                    params=params,
                    objective_terms={"mass_objective": mass_objective},
                )
            allowed_fraction = (
                late_gas_fraction_envelope(
                    time_h,
                    start_h=late_gas_start_h,
                    start_fraction=late_gas_max_fraction,
                    half_life_h=late_gas_decay_half_life_h,
                    floor_fraction=late_gas_floor_fraction,
                )
                if late_gas_decay_half_life_h > 0.0
                else late_gas_max_fraction
            )
            late_gas_allowed_fractions.append(float(allowed_fraction))
            allowed_gas_mass = allowed_fraction * max(injected, 0.0)
            required_aqueous_mass = (
                1.0 - allowed_fraction
            ) * max(injected, 0.0)
            gas_excess = max(0.0, detected_g - allowed_gas_mass)
            aqueous_shortfall = max(0.0, required_aqueous_mass - detected_aq)
            late_gas_excess_mass += gas_excess
            late_aqueous_shortfall_mass += aqueous_shortfall
            # Legacy phase-closure also requires all non-gaseous injected mass
            # to appear as aqueous CO2. New mass-conserving campaigns use the
            # independent post-injection balance and only cap free gas here.
            late_phase_violation_mass += (
                gas_excess
                if late_gas_constraint_mode == "gas-only"
                else max(gas_excess, aqueous_shortfall)
            )
        if late_gas_frames == 0:
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="missing-late-gas-window",
                params=params,
                objective_terms={"mass_objective": mass_objective},
            )
        late_gas_penalty = late_gas_weight * late_phase_violation_mass
        total_err += late_gas_penalty

    # The overflow scale independently constrains phase partition while the
    # image objective still determines spatial support. Each target carries a
    # deadband that represents scale, feed-rate, and balance-model uncertainty.
    overflow_phase_penalty = 0.0
    overflow_gas_violation_mass = 0.0
    overflow_aqueous_violation_mass = 0.0
    overflow_phase_violation_mass = 0.0
    overflow_gas_upper_bound_max_excess_mass = 0.0
    overflow_gas_upper_bound_max_excess_fraction = 0.0
    overflow_gas_hard_upper_bound_max_excess_mass = 0.0
    overflow_gas_hard_upper_bound_max_excess_fraction = 0.0
    overflow_gas_hard_upper_bound_frames = 0
    overflow_phase_frames = 0
    overflow_post_coverage_penalty = 0.0
    overflow_post_coverage_violation_mass = 0.0
    overflow_post_coverage_max_excess_fraction = 0.0
    overflow_post_coverage_frames = 0
    overflow_post_coverage_anchor_time_h = 0.0
    overflow_post_coverage_anchor_gas_fraction = 0.0
    overflow_constraint = getattr(context, "overflow_phase_constraint", None)
    overflow_weight = float(
        getattr(context, "overflow_phase_penalty_weight", 0.0) or 0.0
    )
    overflow_mode = "disabled"
    if overflow_weight > 0.0:
        if overflow_constraint is None:
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="missing-overflow-phase-targets",
                params=params,
                objective_terms={"mass_objective": mass_objective},
            )

        def interval_violation(value: float, lower: float, upper: float) -> float:
            return max(lower - value, value - upper, 0.0)

        overflow_mode = str(
            getattr(
                overflow_constraint,
                "constraint_mode",
                "two-sided-interval",
            )
        )
        for time_h, injected, detected_g, detected_aq in phase_samples:
            target = overflow_constraint.target_at(time_h)
            if target is None:
                continue
            overflow_phase_frames += 1
            if (
                detected_g is None
                or detected_aq is None
                or not np.isfinite(detected_g)
                or not np.isfinite(detected_aq)
            ):
                return EvalResult(
                    objective=PENALTY_VALUE,
                    feasible=False,
                    metrics=metrics,
                    status="overflow-phase-mass-unavailable",
                    params=params,
                    objective_terms={"mass_objective": mass_objective},
                )
            injected_tolerance = max(
                2e-6,
                0.02 * max(float(target.injected_mass_kg), 1e-12),
            )
            if abs(float(injected) - float(target.injected_mass_kg)) > injected_tolerance:
                return EvalResult(
                    objective=PENALTY_VALUE,
                    feasible=False,
                    metrics=metrics,
                    status="overflow-phase-time-alignment-mismatch",
                    params=params,
                    objective_terms={"mass_objective": mass_objective},
                )
            sigma = max(0.0, float(target.phase_mass_sigma_kg))
            gas_center = min(max(float(target.gas_mass_kg), 0.0), float(injected))
            gas_lower = max(0.0, gas_center - sigma)
            gas_upper = min(float(injected), gas_center + sigma)
            aqueous_center = max(0.0, float(injected) - gas_center)
            aqueous_lower = max(0.0, aqueous_center - sigma)
            aqueous_upper = min(float(injected), aqueous_center + sigma)
            if overflow_mode == "gas-upper-bound":
                # Overflow constrains the maximum free-gas inventory. It must
                # not force the colour model to invent gas when optical phase
                # evidence supports a lower value or rapid dissolution.
                gas_violation = max(0.0, float(detected_g) - gas_upper)
                aqueous_violation = 0.0
                overflow_gas_upper_bound_max_excess_mass = max(
                    overflow_gas_upper_bound_max_excess_mass,
                    gas_violation,
                )
                overflow_gas_upper_bound_max_excess_fraction = max(
                    overflow_gas_upper_bound_max_excess_fraction,
                    gas_violation / max(float(injected), 1e-12),
                )
                target_source = str(
                    getattr(target, "target_source", "") or ""
                ).strip().lower()
                # Only run-specific measured points are treated as absolute.
                # An unknown source is strict by default; explicitly imputed
                # cohort points remain a one-sided soft prior.
                if not target_source.startswith("imputed"):
                    overflow_gas_hard_upper_bound_frames += 1
                    overflow_gas_hard_upper_bound_max_excess_mass = max(
                        overflow_gas_hard_upper_bound_max_excess_mass,
                        gas_violation,
                    )
                    overflow_gas_hard_upper_bound_max_excess_fraction = max(
                        overflow_gas_hard_upper_bound_max_excess_fraction,
                        gas_violation / max(float(injected), 1e-12),
                    )
            else:
                gas_violation = interval_violation(
                    float(detected_g), gas_lower, gas_upper
                )
                aqueous_violation = interval_violation(
                    float(detected_aq), aqueous_lower, aqueous_upper
                )
            overflow_gas_violation_mass += gas_violation
            overflow_aqueous_violation_mass += aqueous_violation
            # Under exact mass conservation the two residuals are equivalent.
            # max() prevents counting the same phase-partition error twice and
            # still prevents a zero-mass solution from evading the constraint.
            overflow_phase_violation_mass += max(
                gas_violation,
                aqueous_violation,
            )
        if overflow_phase_frames == 0:
            return EvalResult(
                objective=PENALTY_VALUE,
                feasible=False,
                metrics=metrics,
                status="missing-overflow-phase-window",
                params=params,
                objective_terms={"mass_objective": mass_objective},
            )
        overflow_phase_penalty = (
            overflow_weight * overflow_phase_violation_mass
        )
        total_err += overflow_phase_penalty

    # Once the run-specific overflow record ends, retain its final upper bound
    # as a soft no-new-large-gas support. This does not prescribe a universal
    # decay curve: gas may persist or leave the visible slice, but a large new
    # inventory must overcome the last measured/imputed uncertainty plus a
    # configurable fraction of injected mass.
    overflow_post_weight = float(
        getattr(
            context,
            "overflow_post_coverage_penalty_weight",
            DEFAULT_OVERFLOW_POST_COVERAGE_PENALTY_WEIGHT,
        )
        or 0.0
    )
    if overflow_post_weight > 0.0 and overflow_constraint is not None:
        anchor = max(
            overflow_constraint.targets,
            key=lambda target: float(target.time_h),
        )
        overflow_post_coverage_anchor_time_h = float(anchor.time_h)
        overflow_post_coverage_anchor_gas_fraction = float(
            anchor.gas_mass_kg / max(anchor.injected_mass_kg, 1e-12)
        )
        sigma_multiplier = float(
            getattr(
                context,
                "overflow_post_coverage_sigma_multiplier",
                DEFAULT_OVERFLOW_POST_COVERAGE_SIGMA_MULTIPLIER,
            )
        )
        fraction_tolerance = float(
            getattr(
                context,
                "overflow_post_coverage_fraction_tolerance",
                DEFAULT_OVERFLOW_POST_COVERAGE_FRACTION_TOLERANCE,
            )
        )
        for time_h, injected, detected_g, _detected_aq in phase_samples:
            if time_h <= overflow_post_coverage_anchor_time_h + 1e-9:
                continue
            if detected_g is None or not np.isfinite(detected_g):
                return EvalResult(
                    objective=PENALTY_VALUE,
                    feasible=False,
                    metrics=metrics,
                    status="overflow-post-coverage-gas-unavailable",
                    params=params,
                    objective_terms={"mass_objective": mass_objective},
                )
            overflow_post_coverage_frames += 1
            allowed_gas_mass = min(
                max(float(injected), 0.0),
                float(anchor.gas_mass_kg)
                + sigma_multiplier * float(anchor.phase_mass_sigma_kg)
                + fraction_tolerance * max(float(injected), 0.0),
            )
            excess = max(0.0, float(detected_g) - allowed_gas_mass)
            overflow_post_coverage_violation_mass += excess
            overflow_post_coverage_max_excess_fraction = max(
                overflow_post_coverage_max_excess_fraction,
                excess / max(float(injected), 1e-12),
            )
        overflow_post_coverage_penalty = (
            overflow_post_weight * overflow_post_coverage_violation_mass
        )
        total_err += overflow_post_coverage_penalty

    spatial_summary = summarize_spatial_objective(
        metrics,
        recall_weight=float(getattr(context, "spatial_recall_weight", 0.0)),
        seal_weight=float(getattr(context, "spatial_seal_weight", 0.0)),
    )
    if spatial_observation_active and int(spatial_summary["spatial_frames"]) != len(
        metrics
    ):
        return EvalResult(
            objective=PENALTY_VALUE,
            feasible=False,
            metrics=metrics,
            status="spatial-diagnostics-unavailable",
            params=params,
            objective_terms={"mass_objective": mass_objective},
        )
    spatial_objective_weight = float(
        getattr(context, "spatial_objective_weight", 0.0) or 0.0
    )
    spatial_penalty = (
        spatial_objective_weight
        * float(spatial_summary["spatial_raw_penalty_mass_kg"])
    )
    total_err += spatial_penalty

    try:
        gain_summary = summarize_layer_gain_regularization(
            params,
            getattr(context, "signal_label_to_facies", {}),
            sigma=float(
                getattr(
                    context,
                    "layer_gain_regularization_sigma",
                    DEFAULT_LAYER_GAIN_REGULARIZATION_SIGMA,
                )
            ),
        )
    except ValueError as exc:
        return EvalResult(
            objective=PENALTY_VALUE,
            feasible=False,
            metrics=metrics,
            status=f"invalid-layer-gain:{exc}",
            params=params,
            objective_terms={"mass_objective": mass_objective},
        )
    layer_gain_weight = float(
        getattr(context, "layer_gain_regularization_weight", 0.0) or 0.0
    )
    if layer_gain_weight > 0.0 and gain_summary["layer_gain_count"] <= 0.0:
        return EvalResult(
            objective=PENALTY_VALUE,
            feasible=False,
            metrics=metrics,
            status="layer-gain-regularization-without-layer-gains",
            params=params,
            objective_terms={"mass_objective": mass_objective},
        )
    injected_scale = max(
        (max(0.0, float(sample[1])) for sample in phase_samples),
        default=1.0,
    )
    layer_gain_regularization_penalty = (
        layer_gain_weight
        * injected_scale
        * float(gain_summary["layer_gain_regularization_z2"])
    )
    total_err += layer_gain_regularization_penalty

    depth_prior_penalty = 0.0
    depth_prior_z2 = 0.0
    depth_prior_weight = float(
        getattr(context, "depth_prior_weight", DEFAULT_DEPTH_PRIOR_WEIGHT) or 0.0
    )
    depth_model = getattr(context, "_depth_model", None)
    if depth_prior_weight > 0.0 and depth_model:
        edge_sigma = float(context.depth_edge_prior_sigma_m)
        bulge_sigma = float(context.depth_bulge_prior_sigma_m)
        edge_m = float(params.get("depth.edge_m", depth_model["edge_reference_m"]))
        bulge_m = float(
            params.get("depth.bulge_m", depth_model["bulge_reference_m"])
        )
        depth_prior_z2 = (
            (edge_m - float(depth_model["edge_reference_m"])) / edge_sigma
        ) ** 2 + (
            (bulge_m - float(depth_model["bulge_reference_m"])) / bulge_sigma
        ) ** 2
        injected_scale = max(
            (max(0.0, float(sample[1])) for sample in phase_samples),
            default=1.0,
        )
        depth_prior_penalty = depth_prior_weight * injected_scale * depth_prior_z2
        total_err += depth_prior_penalty
    if not np.isfinite(total_err):
        return EvalResult(objective=PENALTY_VALUE, feasible=False, metrics=metrics,
                          status="non-finite-objective", params=params)
    return EvalResult(objective=float(total_err), feasible=feasible, metrics=metrics,
                      status="ok", params=params,
                      objective_terms={
                          "mass_objective": mass_objective,
                          "pointwise_mass_objective": pointwise_mass_objective,
                          "window_balanced_mass_objective": (
                              window_balanced_mass_objective
                          ),
                          "window_balanced_unscaled_mass_objective": (
                              window_balanced_unscaled_mass_objective
                          ),
                          "window_balanced_scale_factor": (
                              window_balanced_scale_factor
                          ),
                          "plateau_drift_mass_kg": plateau_drift_mass,
                          "plateau_drift_penalty": plateau_drift_penalty,
                          "post_injection_total_drift_mass_kg": (
                              post_injection_total_drift_mass
                          ),
                          "post_injection_phase_direction_mass_kg": (
                              post_injection_phase_direction_mass
                          ),
                          "post_injection_balance_violation_mass_kg": (
                              post_injection_balance_violation_mass
                          ),
                          "post_injection_balance_penalty": (
                              post_injection_balance_penalty
                          ),
                          "post_injection_balance_pairs": float(
                              post_injection_balance_pairs
                          ),
                          "aqueous_diffusion_violation_mass_kg": (
                              aqueous_diffusion_violation_mass
                          ),
                          "aqueous_diffusion_penalty": aqueous_diffusion_penalty,
                          "aqueous_diffusion_max_increase_relative": (
                              aqueous_diffusion_max_increase_relative
                          ),
                          "aqueous_diffusion_peak_increase_relative": (
                              aqueous_diffusion_peak_increase_relative
                          ),
                          "aqueous_diffusion_volume_decrease_relative": (
                              aqueous_diffusion_volume_decrease_relative
                          ),
                          "aqueous_diffusion_effective_volume_decrease_relative": (
                              aqueous_diffusion_effective_volume_decrease_relative
                          ),
                          "aqueous_diffusion_pairs": float(
                              aqueous_diffusion_pairs
                          ),
                          "late_gas_excess_mass_kg": late_gas_excess_mass,
                          "late_aqueous_shortfall_mass_kg": (
                              late_aqueous_shortfall_mass
                          ),
                          "late_phase_violation_mass_kg": (
                              late_phase_violation_mass
                          ),
                          "late_gas_penalty": late_gas_penalty,
                          "late_gas_frames": float(late_gas_frames),
                          "late_gas_allowed_fraction_min": (
                              min(late_gas_allowed_fractions)
                              if late_gas_allowed_fractions
                              else 0.0
                          ),
                          "late_gas_allowed_fraction_max": (
                              max(late_gas_allowed_fractions)
                              if late_gas_allowed_fractions
                              else 0.0
                          ),
                          "overflow_gas_violation_mass_kg": (
                              overflow_gas_violation_mass
                          ),
                          "overflow_aqueous_violation_mass_kg": (
                              overflow_aqueous_violation_mass
                          ),
                          "overflow_phase_violation_mass_kg": (
                              overflow_phase_violation_mass
                          ),
                          "overflow_gas_upper_bound_active": float(
                              overflow_mode == "gas-upper-bound"
                          ),
                          "overflow_gas_upper_bound_max_excess_mass_kg": (
                              overflow_gas_upper_bound_max_excess_mass
                          ),
                          "overflow_gas_upper_bound_max_excess_fraction": (
                              overflow_gas_upper_bound_max_excess_fraction
                          ),
                          "overflow_gas_hard_upper_bound_frames": float(
                              overflow_gas_hard_upper_bound_frames
                          ),
                          "overflow_gas_hard_upper_bound_max_excess_mass_kg": (
                              overflow_gas_hard_upper_bound_max_excess_mass
                          ),
                          "overflow_gas_hard_upper_bound_max_excess_fraction": (
                              overflow_gas_hard_upper_bound_max_excess_fraction
                          ),
                           "overflow_phase_penalty": overflow_phase_penalty,
                          "overflow_phase_frames": float(overflow_phase_frames),
                          "overflow_post_coverage_violation_mass_kg": (
                              overflow_post_coverage_violation_mass
                          ),
                          "overflow_post_coverage_max_excess_fraction": (
                              overflow_post_coverage_max_excess_fraction
                          ),
                          "overflow_post_coverage_frames": float(
                              overflow_post_coverage_frames
                          ),
                          "overflow_post_coverage_anchor_time_h": (
                              overflow_post_coverage_anchor_time_h
                          ),
                          "overflow_post_coverage_anchor_gas_fraction": (
                              overflow_post_coverage_anchor_gas_fraction
                          ),
                          "overflow_post_coverage_penalty": (
                              overflow_post_coverage_penalty
                          ),
                           **spatial_summary,
                           "spatial_penalty": float(spatial_penalty),
                           **gain_summary,
                           "layer_gain_regularization_penalty": float(
                               layer_gain_regularization_penalty
                           ),
                           "depth_prior_z2": float(depth_prior_z2),
                          "depth_prior_penalty": float(depth_prior_penalty),
                      })


_EVALUATION_THREAD_ENV = frozenset(
    {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_NUM_THREADS",
    }
)


@contextmanager
def _context_evaluation_environment(context: CalibrationContext):
    """Re-enter the exact FFAC environment used to construct ``context``."""

    requested = getattr(context, "_ffac_evaluation_environment", None)
    if requested is None:
        yield
        return

    requested = {str(name): str(value) for name, value in requested.items()}
    relevant = {
        name
        for name in set(os.environ) | set(requested)
        if name.startswith("FFAC_") or name in _EVALUATION_THREAD_ENV
    }
    previous = {name: os.environ.get(name) for name in relevant}
    try:
        for name in relevant:
            if name in requested:
                os.environ[name] = requested[name]
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def evaluate_run(context: CalibrationContext, params: Dict[str, Any]) -> EvalResult:
    """Evaluate one candidate under the context's immutable environment contract."""

    with _context_evaluation_environment(context):
        return _evaluate_run_impl(context, params)


def save_best_calibration(context: CalibrationContext, best_params, out_folder=None) -> None:
    import numpy as np
    if best_params is None:
        raise ValueError(f"[{context.run}] no best params to save")
    if context.calibration_folder is None:
        raise ValueError(f"[{context.run}] calibration folder is not configured")

    apply_params(context.calibration, best_params, labels=context.signal_labels, np_module=np)
    folder = Path(context.calibration_folder) / "signal_model"
    folder.mkdir(parents=True, exist_ok=True)
    hetero = context.calibration.signal_model.model[1]
    failures: List[str] = []
    for label in (hetero.keys() if hasattr(hetero, "keys") else []):
        try:
            hetero[label].save(folder / f"signal_model_{label}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"signal_model_{label}: {exc}")
    logger.info("[%s] saved optimised signal models to %s", context.run, folder)

    flash = getattr(context.calibration, "flash", None)
    if flash is None or not hasattr(flash, "save"):
        failures.append("flash: calibration has no saveable flash model")
    else:
        try:
            flash_path = Path(context.calibration_folder) / "flash" / "flash"
            flash.save(flash_path)
            logger.info(
                "[%s] saved optimised %s to %s",
                context.run,
                type(flash).__name__,
                flash_path.with_suffix(".json"),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"flash: {exc}")

    aqueous_noise_gate = getattr(context, "aqueous_noise_gate", None)
    if aqueous_noise_gate is not None:
        try:
            gate_path = (
                Path(context.calibration_folder)
                / "observation_model"
                / "aqueous_noise_gate.json"
            )
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate_path.write_text(
                json.dumps(
                    aqueous_noise_gate.to_dict(),
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            logger.info(
                "[%s] saved aqueous noise gate to %s", context.run, gate_path
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"aqueous_noise_gate: {exc}")

    if failures:
        raise RuntimeError(
            f"[{context.run}] calibration save failed: " + "; ".join(failures)
        )

    if out_folder:
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        (Path(out_folder) / "best_params.json").write_text(
            json.dumps(best_params, indent=2, default=str), encoding="utf-8"
        )


# =========================================================================
# Standalone per-run optimisation (no queue) - for smoke tests / single machine
# =========================================================================
def _diagnose_baseline(context) -> None:
    """Log per-image injected vs detected mass for the seeded values, to locate NaNs."""
    import numpy as np
    apply_params(context.calibration, {}, labels=context.signal_labels, np_module=np)
    for i, (img, injected, t_h) in enumerate(context._loaded):
        det = float("nan"); nan_frac = -1.0
        try:
            mass = _mass_result_for_evaluation(context, img, i, {}).mass
            arr = np.asarray(getattr(mass, "img", mass), dtype=float)
            nan_frac = float(np.mean(~np.isfinite(arr))) if arr.size else -1.0
            det = float(context.geometry.integrate(mass))
        except Exception as exc:  # noqa: BLE001
            logger.info("[diag %s] img%d ERROR %s", context.run, i, exc); continue
        source = (
            context._prepared_colors[i]
            if i < len(getattr(context, "_prepared_colors", []))
            else img
        )
        date = getattr(source, "date", None)
        logger.info("[diag %s] img%d date=%s injected=%s detected=%s mass_nan_frac=%.3f",
                    context.run, i, date, injected, det, nan_frac)


def optimize_per_run(context: CalibrationContext, max_iters: int, logs_dir: Path,
                     warmup_iters: int = 0):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    best = {"objective": float("inf"), "params": None}

    _diagnose_baseline(context)

    # Baseline = current (seeded) signal values (apply nothing). Anchors the search
    # and confirms the seeded calibration yields a finite objective.
    base = evaluate_run(context, {})
    logger.info("[%s] baseline (seeded) objective=%.6g feasible=%s status=%s",
                context.run, base.objective, base.feasible, base.status)
    if base.objective < best["objective"]:
        best["objective"] = base.objective; best["params"] = {}

    def _objective(trial):
        params = suggest_params_trial(trial, context.param_space)
        res = evaluate_run(context, params)
        obj = res.objective if (res.objective == res.objective) else PENALTY_VALUE  # NaN guard
        if obj < best["objective"]:
            best["objective"] = obj; best["params"] = params
        return obj

    study.optimize(_objective, n_trials=max(1, max_iters))
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_history_csv(logs_dir / f"auto_calibration_{context.run}.csv",
                      [{"iter": i, "objective": t.value} for i, t in enumerate(study.trials)])
    logger.info("[%s] best objective=%.6g over %d trials", context.run,
                best["objective"], len(study.trials))
    return best["params"], best["objective"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--mode", choices=["per-run"], default="per-run")
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--config-dir", type=str, default="config/run_ac")
    p.add_argument("--logs-dir", type=str, default=_default_calibration_log_root())
    p.add_argument("--ref-config", type=str, default=None)
    p.add_argument("--use-facies", action="store_true")
    p.add_argument("--per-label", action="store_true")
    p.add_argument("--use-last-best", action="store_true")
    p.add_argument("--enforce-lower", action="store_true")
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--warmup-levels", default=None)
    p.add_argument("--run-mode", default="serial")
    p.add_argument("--max-in-flight-per-run", type=int, default=0)
    p.add_argument("--objective-integral", default="off")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    from darsia.presets.workflows.rig import Rig
    config_dir = Path(args.config_dir); logs_dir = Path(args.logs_dir)
    for run in args.runs:
        ctx = build_context(run=run, config_dir=config_dir, rig_cls=Rig,
                            ref_config_path=args.ref_config,
                            use_facies=args.use_facies, per_label_params=args.per_label,
                            enforce_lower=args.enforce_lower,
                            objective_integral=args.objective_integral)
        params, obj = optimize_per_run(ctx, args.max_iters, logs_dir, args.warmup_iters)
        if params:
            save_best_calibration(ctx, params)
        print(f"DONE {run}: best objective={obj:.6g}")


if __name__ == "__main__":
    main()

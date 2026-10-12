"""Spatially validate and improve the final FF-AC calibration method.

Track A replays the complete feasible 5000-trial history with a fixed,
parameter-independent optical support model. Track B runs fresh local
ask/evaluate/tell Optuna searches whose objective includes the same spatial
evidence. Jobs are claimed dynamically so persistent CUDA processes can keep a
run context resident without binding the campaign to a fixed host assignment.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import optuna
from optuna.storages import JournalStorage


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")
WORKER_SCRIPT = Path(__file__).resolve()

from darsia.fluidflower.tools import ac_clearwater_phase_pilot as pilot  # noqa: E402
from darsia.fluidflower.tools import ac_final_method_selection as selection  # noqa: E402
from darsia.fluidflower.tools import ac_final_production_calibration as final_campaign  # noqa: E402
from darsia.fluidflower.tools import ac_final_production_selection as final_selection  # noqa: E402
from darsia.fluidflower.tools import auto_calibrate_color_to_mass as calibration  # noqa: E402
from darsia.fluidflower.tools import distributed_auto_calibration_queue as queue_support  # noqa: E402
from darsia.fluidflower.tools.calibration_backend_contract import (  # noqa: E402
    all_backend_names,
    backend_spec,
    lock_backend_contract,
)
from darsia.fluidflower.tools.optuna_objective_policy import (  # noqa: E402
    V2_CONSTRAINT_NAMES,
    is_physical_v2_policy,
    normalize_policy,
    score_search_objective,
    uses_late_gas_guardrail,
)
from darsia.fluidflower.tools.local_trial_cache import (  # noqa: E402
    BatchedJournalBackend,
    LocalJobCache,
)
from darsia.fluidflower.tools.optimized_tpe_sampler import (  # noqa: E402
    SAMPLER_CONTRACT,
    CachedGroupedTPESampler,
)
from darsia.fluidflower.tools.worker_resource_pool import (  # noqa: E402
    LocalSlotLease,
    LocalSlotPool,
)


CAMPAIGN_ID = "spatial_objective_screen_20260815_5000"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
PILOT_RUNS = ("ac17", "ac18", "ac20", "ac24", "ac56", "ac60")
ALL_RUNS = tuple(final_campaign.ALL_RUNS)
# Preserve the original six-run screen as the default. Production rollouts pass
# ``--runs all`` explicitly and record the resolved list in their manifest.
RUNS = PILOT_RUNS
MAX_ITERS = 5000
RANDOM_WARMUPS = 150
OPTUNA_SEED = 170815
REPLAY_LIMIT = 0  # zero means every feasible, de-duplicated history row
SPATIAL_OBJECTIVE_WEIGHT = 1.0
SPATIAL_RECALL_WEIGHT = 0.25
SPATIAL_SEAL_WEIGHT = 4.0
SPATIAL_TRACE_LEVEL = 0.02
SPATIAL_CORE_LEVEL = 0.02
GAIN_SIGMA = 0.20
LIMITED_TEMPORAL_COVERAGE_RUNS = ("ac23", "ac30")
TRANSIENT_IO_RETRIES = 180
OPTIMIZE_JOB_ORDER = "variant-major"
NETWORK_WINERRORS = frozenset((53, 54, 55, 59, 64, 67, 121, 1231, 1232))


@dataclass(frozen=True)
class SearchSpec:
    name: str
    variant: Any
    gain_weight: float
    rationale: str
    color_path_evidence: str = "off"
    objective_policy: str = "evaluation"
    spatial_objective_weight: float | None = None
    seed_group: str | None = None
    historical_seed_count: int = 1
    objective_integral: str = "off"
    additional_calibration_times_h: tuple[float, ...] = ()
    finalist_count: int = 4
    stage1_gain_fraction: float = 0.0
    seed_campaign_roots: tuple[str, ...] = ()
    validation_times_h: tuple[float, ...] = ()
    audit_times_h: tuple[float, ...] = ()
    preprocessing_cache_group: str | None = None
    environment_overrides: tuple[tuple[str, str], ...] = ()
    run_environment_overrides: tuple[
        tuple[str, tuple[tuple[str, str], ...]], ...
    ] = ()


SEARCH_SPECS = (
    SearchSpec(
        name="hierarchical_spatial",
        variant=replace(
            final_campaign.VARIANT,
            name="hierarchical_spatial",
        ),
        gain_weight=0.0,
        rationale="Current broad hierarchical model plus spatial evidence.",
    ),
    SearchSpec(
        name="hierarchical_spatial_gainreg",
        variant=replace(
            final_campaign.VARIANT,
            name="hierarchical_spatial_gainreg",
        ),
        gain_weight=0.10,
        rationale="Broad gains with a same-material consistency prior.",
    ),
    SearchSpec(
        name="hierarchical_narrow_spatial",
        variant=replace(
            final_campaign.VARIANT,
            name="hierarchical_narrow_spatial",
            bounds_file=(
                "config/"
                "bounds_seg6_phase_curve_titration_hierarchical_narrow.json"
            ),
        ),
        gain_weight=0.05,
        rationale="Physically narrower layer gains plus weak regularization.",
    ),
    SearchSpec(
        name="shared_narrow_spatial",
        variant=replace(
            final_campaign.VARIANT,
            name="shared_narrow_spatial",
            bounds_file=(
                "config/bounds_seg6_phase_curve_titration_shared_narrow.json"
            ),
            signal_parameterization="shared-shape",
        ),
        gain_weight=0.05,
        rationale="One response shape for all materials with narrow layer gains.",
    ),
)
SPECS_BY_NAME = {spec.name: spec for spec in SEARCH_SPECS}


def _uses_objective_v2(spec: SearchSpec) -> bool:
    return is_physical_v2_policy(spec.objective_policy)


def _v2_gamma(completed: int) -> int:
    """Use a broader elite set than Optuna's 25-trial default in 41D."""

    count = max(1, int(completed))
    return min(count, min(256, max(32, math.ceil(0.10 * count))))


def _trial_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    values = trial.user_attrs.get("constraint_values")
    if not isinstance(values, (list, tuple)) or len(values) != len(
        V2_CONSTRAINT_NAMES
    ):
        return tuple(1.0 for _name in V2_CONSTRAINT_NAMES)
    return tuple(float(value) for value in values)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(
    path: Path,
    payload: Any,
    *,
    attempts: int = TRANSIENT_IO_RETRIES,
) -> None:
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    resolved_attempts = max(1, int(attempts))
    for attempt in range(resolved_attempts):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            os.replace(temporary, path)
            return
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 >= resolved_attempts:
                raise
            time.sleep(min(5.0, 0.25 * (2**attempt)))


def _params_hash(params: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(params), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    line = json.dumps(dict(payload), separators=(",", ":"), default=str) + "\n"
    for attempt in range(TRANSIENT_IO_RETRIES):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(line)
            return
        except OSError:
            if attempt + 1 >= TRANSIENT_IO_RETRIES:
                raise
            time.sleep(min(5.0, 0.25 * (2**attempt)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _is_transient_network_error(exc: BaseException) -> bool:
    """Return whether an exception chain contains a Windows network failure."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and getattr(current, "winerror", None) in NETWORK_WINERRORS:
            return True
        if isinstance(current, FileNotFoundError):
            filename = str(getattr(current, "filename", "") or "")
            if filename.startswith("\\\\"):
                basename = filename.rsplit("\\", 1)[-1]
                if basename.startswith(".") and basename.endswith(".tmp"):
                    # Atomic publishers create this file themselves. ENOENT
                    # here means the SMB directory vanished between mkdir and
                    # open; it cannot represent a genuinely missing input.
                    return True
                # SMB can transiently return ENOENT for an existing file. Only
                # treat this as a network failure when the exact file is visible
                # again; genuinely incomplete campaign inputs must still fail.
                try:
                    with open(filename, "rb") as stream:
                        stream.read(1)
                    return True
                except OSError:
                    pass
        current = current.__cause__ or current.__context__
    return False


def _wait_for_campaign_storage(output_root: Path, poll_seconds: float = 10.0) -> None:
    """Wait indefinitely for the shared campaign manifest to become readable."""

    manifest = output_root / "campaign_manifest.json"
    while True:
        try:
            with manifest.open("rb") as stream:
                stream.read(1)
            return
        except OSError:
            time.sleep(max(0.1, poll_seconds))


def _read_status_json(
    path: Path,
    *,
    required: bool,
    attempts: int = 3,
) -> dict[str, Any]:
    """Read a small campaign status file without trusting one SMB snapshot."""

    last_error: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            if not path.is_file():
                if required:
                    raise FileNotFoundError(path)
                return {}
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object in {path}")
            return payload
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < max(1, attempts):
                time.sleep(0.1 * (attempt + 1))
    if required and last_error is not None:
        raise last_error
    return {}


def _metrics_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): asdict(value) if hasattr(value, "__dataclass_fields__") else vars(value)
        for key, value in metrics.items()
    }


def _objective_terms_subset(terms: Mapping[str, Any]) -> dict[str, float]:
    names = (
        "mass_objective",
        "post_injection_balance_penalty",
        "aqueous_diffusion_penalty",
        "late_gas_penalty",
        "overflow_phase_penalty",
        "overflow_post_coverage_penalty",
        "overflow_post_coverage_violation_mass_kg",
        "overflow_post_coverage_max_excess_fraction",
        "overflow_post_coverage_frames",
        "overflow_post_coverage_anchor_time_h",
        "overflow_post_coverage_anchor_gas_fraction",
        "overflow_gas_violation_mass_kg",
        "overflow_gas_upper_bound_active",
        "overflow_gas_upper_bound_max_excess_mass_kg",
        "overflow_gas_upper_bound_max_excess_fraction",
        "overflow_gas_hard_upper_bound_frames",
        "overflow_gas_hard_upper_bound_max_excess_mass_kg",
        "overflow_gas_hard_upper_bound_max_excess_fraction",
        "depth_prior_penalty",
        "spatial_low_observation_mass_kg",
        "spatial_low_observation_co2_mass_kg",
        "spatial_recall_equivalent_mass_kg",
        "spatial_unsupported_seal_mass_kg",
        "spatial_unsupported_seal_co2_mass_kg",
        "spatial_low_observation_support_fraction",
        "spatial_raw_penalty_mass_kg",
        "spatial_penalty",
        "layer_gain_within_facies_z2",
        "layer_gain_center_z2",
        "layer_gain_regularization_z2",
        "layer_gain_regularization_penalty",
        "layer_gain_min",
        "layer_gain_max",
    )
    return {
        name: float(terms[name])
        for name in names
        if name in terms and terms[name] is not None
    }


def _derived_seed(run: str, variant: str) -> int:
    digest = hashlib.sha256(
        f"{OPTUNA_SEED}:{run}:{variant}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _spatial_environment_values(run: str, spec: SearchSpec) -> dict[str, str]:
    environment = final_selection._campaign_environment_values(run)
    environment.update(
        {
            "FFAC_SPATIAL_OBSERVATION_SUPPORT": "on",
            "FFAC_SPATIAL_OBJECTIVE_WEIGHT": str(
                SPATIAL_OBJECTIVE_WEIGHT
                if spec.spatial_objective_weight is None
                else spec.spatial_objective_weight
            ),
            "FFAC_SPATIAL_RECALL_WEIGHT": str(SPATIAL_RECALL_WEIGHT),
            "FFAC_SPATIAL_SEAL_WEIGHT": str(SPATIAL_SEAL_WEIGHT),
            "FFAC_SPATIAL_TRACE_LEVEL": str(SPATIAL_TRACE_LEVEL),
            "FFAC_SPATIAL_CORE_LEVEL": str(SPATIAL_CORE_LEVEL),
            "FFAC_LAYER_GAIN_REGULARIZATION_WEIGHT": str(spec.gain_weight),
            "FFAC_LAYER_GAIN_REGULARIZATION_SIGMA": str(GAIN_SIGMA),
            "FFAC_AQUEOUS_NOISE_QUANTILE_LOW": "0.99",
            "FFAC_AQUEOUS_NOISE_QUANTILE_HIGH": "0.999",
            "FFAC_AQUEOUS_NOISE_GATE_STRICT": "on",
            "FFAC_COLOR_PATH_EVIDENCE": spec.color_path_evidence,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENCV_NUM_THREADS": "1",
        }
    )
    if run in LIMITED_TEMPORAL_COVERAGE_RUNS:
        # No image exists at or after the 24 h late-gas boundary. Preserve all
        # observable constraints and report this missing validation separately.
        environment["FFAC_LATE_GAS_PENALTY_WEIGHT"] = "0"
    if spec.additional_calibration_times_h:
        existing, _ = calibration.resolve_extra_calibration_times(
            environment.get("FFAC_EXTRA_CALIBRATION_TIMES_H", "")
        )
        combined: list[float] = []
        for value in (*existing, *spec.additional_calibration_times_h):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"Invalid additional calibration time {value!r} for "
                    f"{spec.name}."
                )
            if not any(abs(numeric - prior) <= 1e-9 for prior in combined):
                combined.append(numeric)
        environment["FFAC_EXTRA_CALIBRATION_TIMES_H"] = ",".join(
            f"{value:.9g}" for value in sorted(combined)
        )
    for name, value in spec.environment_overrides:
        key = str(name).strip()
        if not key.startswith("FFAC_"):
            raise ValueError(
                f"Campaign environment override must be an FFAC variable: {key!r}"
            )
        environment[key] = str(value).replace("{run}", str(run).lower())
    by_run = {
        str(raw_run).strip().lower(): tuple(overrides)
        for raw_run, overrides in spec.run_environment_overrides
    }
    for name, value in by_run.get(str(run).lower(), ()):
        key = str(name).strip()
        if not key.startswith("FFAC_"):
            raise ValueError(
                f"Run environment override must be an FFAC variable: {key!r}"
            )
        environment[key] = str(value).replace("{run}", str(run).lower())
    return environment


@contextmanager
def _spatial_environment(run: str, spec: SearchSpec) -> Iterator[None]:
    environment = _spatial_environment_values(run, spec)
    with pilot.campaign_environment(spec.variant, environment):
        yield


_PREPROCESSING_SOURCE_FINGERPRINT: str | None = None


def _preprocessing_source_fingerprint() -> str:
    """Fingerprint code that can alter corrected or interpreted image pixels."""

    global _PREPROCESSING_SOURCE_FINGERPRINT
    if _PREPROCESSING_SOURCE_FINGERPRINT is not None:
        return _PREPROCESSING_SOURCE_FINGERPRINT
    paths = (
        project_path('scripts/auto_calibrate_color_to_mass.py'),
        project_path('scripts/colorchecker_refinement.py'),
        project_path('external/darsia/src/darsia/image/imread.py'),
        project_path('external/darsia/src/darsia/presets/workflows/rig.py'),
        project_path('external/darsia/src/darsia/corrections/color/colorcorrection.py'),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    _PREPROCESSING_SOURCE_FINGERPRINT = digest.hexdigest()
    return _PREPROCESSING_SOURCE_FINGERPRINT


def _corrected_image_cache_root() -> Path:
    configured = os.environ.get("FFAC_CORRECTED_IMAGE_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured)
    local = Path(os.environ.get("LOCALAPPDATA", str(project_path('logs'))))
    return local / "ff_ac" / "corrected_image_cache" / CAMPAIGN_ID


def _context_environment_values(
    run: str,
    spec: SearchSpec,
    backend: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the exact immutable environment and cache identity for a context."""

    numerical_contract = backend_spec(backend)
    quality_dtype = (
        "float64" if numerical_contract.name == "cpu-float64" else "float32"
    )
    cache_group = spec.preprocessing_cache_group or spec.name
    cache_identity = {
        "campaign_id": CAMPAIGN_ID,
        "run": str(run).lower(),
        "cache_group": cache_group,
        "quality_dtype": quality_dtype,
        "source_sha256": _preprocessing_source_fingerprint(),
    }
    cache_namespace = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_root = _corrected_image_cache_root()
    environment = _spatial_environment_values(run, spec)
    environment.update(
        {
            "FFAC_CORRECTED_IMAGE_CACHE_ROOT": str(cache_root),
            "FFAC_CORRECTED_IMAGE_CACHE_NAMESPACE": cache_namespace,
            "FFAC_PREPARED_CONTEXT_CACHE_ROOT": str(
                cache_root / "prepared_contexts"
            ),
            "FFAC_COLORCHECKER_WORK_DTYPE": quality_dtype,
        }
    )
    return environment, {
        "quality_dtype": quality_dtype,
        "cache_group": cache_group,
        "cache_namespace": cache_namespace,
    }


def _local_worker_slot_root(output_root: Path, backend: str) -> Path:
    configured = os.environ.get("FFAC_WORKER_SLOT_ROOT", "").strip()
    base = Path(
        configured
        or Path(os.environ.get("LOCALAPPDATA", str(project_path('logs'))))
        / "ff_ac"
        / "worker_slots"
    )
    namespace = hashlib.sha256(
        str(Path(output_root).resolve()).casefold().encode("utf-8")
    ).hexdigest()[:12]
    return base / CAMPAIGN_ID / namespace / str(backend)


def _build_context(
    run: str,
    spec: SearchSpec,
    *,
    backend: str,
    times: Sequence[float] | None = None,
    initialize_backend: bool = True,
) -> Any:
    total_started = time.perf_counter()
    context = None
    try:
        evaluation_environment, cache_contract = _context_environment_values(
            run,
            spec,
            backend,
        )
        quality_dtype = cache_contract["quality_dtype"]
        cache_group = cache_contract["cache_group"]
        cache_namespace = cache_contract["cache_namespace"]
        with pilot.campaign_environment(spec.variant, evaluation_environment):
            build_started = time.perf_counter()
            context = selection._build_evaluation_context(
                run,
                spec.variant,
                times=times,
                allow_partial_evaluation_times=times is not None,
                quality_dtype=quality_dtype,
            )
            raw_context_seconds = time.perf_counter() - build_started
            context.objective_integral = str(spec.objective_integral)
            prepare_started = time.perf_counter()
            calibration.prepare_evaluation_context(
                context,
                backend=backend,
                release_images=True,
                initialize_backend=initialize_backend,
            )
            prepare_seconds = time.perf_counter() - prepare_started
            # Several evaluation settings are intentionally supplied through
            # FFAC_* variables. Keep that exact contract attached to the
            # resident context; evaluations happen after this temporary build
            # environment has exited and must not silently fall back to host
            # defaults (for example hard instead of soft expert constraints).
            context._ffac_evaluation_environment = dict(evaluation_environment)
            context._context_build_timing = {
                "raw_context_seconds": raw_context_seconds,
                "prepare_evaluation_seconds": prepare_seconds,
                "prepare_evaluation_timing": dict(
                    getattr(context, "_prepare_evaluation_timing", {}) or {}
                ),
                "total_seconds": time.perf_counter() - total_started,
                "quality_dtype": quality_dtype,
                "corrected_image_cache_namespace": cache_namespace,
                "corrected_image_cache_group": cache_group,
                "corrected_image_cache": dict(
                    getattr(context, "corrected_image_cache_stats", {}) or {}
                ),
                "backend_initialization_deferred": not initialize_backend,
            }
        if initialize_backend:
            _validate_context_backend(context, backend)
        return context
    except Exception:
        if context is not None:
            _close_context(context)
        raise


def _close_context(context: Any) -> None:
    selection._close_context(context)


def _validate_context_backend(context: Any, backend: str) -> None:
    evaluator_attributes = {
        "cuda": "_cuda_evaluator",
        "cuda-float32": "_cuda_float32_evaluator",
        "opencl": "_opencl_evaluator",
        "opencl-float32": "_opencl_float32_evaluator",
        "mps-metal": "_metal_evaluator",
        "metal-float32": "_metal_evaluator",
    }
    evaluator_attribute = evaluator_attributes.get(backend)
    if evaluator_attribute is None:
        return
    evaluator = getattr(context, evaluator_attribute, None)
    if evaluator is None or not bool(
        getattr(evaluator, "spatial_objective_active", False)
    ):
        raise RuntimeError(
            f"{backend} preflight failed: spatial diagnostics are not active"
        )


def _activate_context_backend(context: Any, backend: str) -> None:
    """Finish deferred accelerator setup without rebuilding host-side arrays."""

    started = time.perf_counter()
    # Accelerator constructors pack several FFAC settings into resident device
    # arrays.  Deferred activation must therefore re-enter the exact immutable
    # environment used for context construction, just like every evaluation.
    with calibration._context_evaluation_environment(context):
        calibration.prepare_evaluation_context(
            context,
            backend=backend,
            release_images=True,
            initialize_backend=True,
        )
    _validate_context_backend(context, backend)
    timing = dict(getattr(context, "_context_build_timing", {}) or {})
    timing.update(
        {
            "backend_activation_seconds": time.perf_counter() - started,
            "backend_activation_timing": dict(
                getattr(context, "_prepare_evaluation_timing", {}) or {}
            ),
            "backend_initialization_deferred": False,
        }
    )
    context._context_build_timing = timing


@dataclass
class _ContextResources:
    """Host-local context builder and accelerator-slot coordinator."""

    backend: str
    build_pool: LocalSlotPool | None = None
    backend_pool: LocalSlotPool | None = None

    @property
    def pipelined(self) -> bool:
        return self.backend_pool is not None

    def build(
        self,
        run: str,
        spec: SearchSpec,
        *,
        times: Sequence[float] | None = None,
    ) -> Any:
        build_lease: LocalSlotLease | None = None
        try:
            if self.build_pool is not None:
                build_lease = self.build_pool.acquire()
            context = _build_context(
                run,
                spec,
                backend=self.backend,
                times=times,
                initialize_backend=not self.pipelined,
            )
            timing = dict(getattr(context, "_context_build_timing", {}) or {})
            timing.update(
                {
                    "context_build_slot_wait_seconds": (
                        float(build_lease.wait_seconds)
                        if build_lease is not None
                        else 0.0
                    ),
                    "context_build_slot": (
                        int(build_lease.index) if build_lease is not None else None
                    ),
                    "ram_prefetch_ready": bool(self.pipelined),
                }
            )
            context._context_build_timing = timing
            context._ffac_prefetch_ready_monotonic = time.perf_counter()
            return context
        finally:
            if build_lease is not None:
                build_lease.release()

    def activate(self, context: Any) -> LocalSlotLease | None:
        if self.backend_pool is None:
            return None
        lease = self.backend_pool.acquire()
        try:
            ready_at = float(
                getattr(context, "_ffac_prefetch_ready_monotonic", time.perf_counter())
            )
            timing = dict(getattr(context, "_context_build_timing", {}) or {})
            timing.update(
                {
                    "backend_slot_wait_seconds": float(lease.wait_seconds),
                    "backend_slot": int(lease.index),
                    "ram_prefetch_ready_wait_seconds": max(
                        0.0, time.perf_counter() - ready_at
                    ),
                }
            )
            context._context_build_timing = timing
            _activate_context_backend(context, self.backend)
        except Exception:
            lease.release()
            raise
        return lease

    @staticmethod
    def release(lease: LocalSlotLease | None) -> None:
        if lease is not None:
            lease.release()


class _ValidationContextPrefetch:
    """Build disjoint validation contexts while the accelerator trains.

    Validation and audit deliberately remain separate. Temporal observation
    gates use earlier frames, so combining the two schedules would let audit
    images alter validation preprocessing.
    """

    def __init__(
        self,
        *,
        resources: _ContextResources,
        run: str,
        spec: SearchSpec,
    ) -> None:
        self.enabled = bool(resources.pipelined)
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future[Any]] = {}
        self._transferred: set[str] = set()
        self._started = time.perf_counter()
        self._provenance: dict[str, Any] = {
            "enabled": self.enabled,
            "contexts_disjoint": True,
            "reason": "preserve temporal validation/audit isolation",
            "started_utc": _utc_now(),
            "roles": {},
        }
        if not self.enabled:
            return

        validation_times = tuple(
            spec.validation_times_h or selection.HOLDOUT_TIMES_H
        )
        audit_times = tuple(spec.audit_times_h)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"ffac-validation-{run}",
        )
        self._submit(
            "validation",
            resources,
            run,
            spec,
            validation_times,
        )
        if audit_times:
            self._submit("audit", resources, run, spec, audit_times)

    def _submit(
        self,
        role: str,
        resources: _ContextResources,
        run: str,
        spec: SearchSpec,
        times_h: tuple[float, ...],
    ) -> None:
        if self._executor is None:
            return
        self._provenance["roles"][role] = {
            "requested_times_h": list(times_h),
            "submitted_seconds_after_training_start": (
                time.perf_counter() - self._started
            ),
        }
        self._futures[role] = self._executor.submit(
            resources.build,
            run,
            spec,
            times=times_h,
        )

    def take(self, role: str) -> Any | None:
        """Transfer ownership of one prefetched context to the caller."""

        future = self._futures.get(role)
        if future is None:
            return None
        self._transferred.add(role)
        wait_started = time.perf_counter()
        role_provenance = self._provenance["roles"][role]
        try:
            context = future.result()
        except Exception as exc:
            role_provenance.update(
                {
                    "status": "failed",
                    "wait_seconds_at_handoff": time.perf_counter() - wait_started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        role_provenance.update(
            {
                "status": "ready",
                "wait_seconds_at_handoff": time.perf_counter() - wait_started,
                "context_build_timing": dict(
                    getattr(context, "_context_build_timing", {}) or {}
                ),
            }
        )
        return context

    def snapshot(self) -> dict[str, Any]:
        return {
            **self._provenance,
            "elapsed_seconds": time.perf_counter() - self._started,
        }

    def close(self) -> None:
        """Close contexts whose ownership was never transferred."""

        if self._executor is None:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        for role, future in self._futures.items():
            if role in self._transferred or future.cancelled():
                continue
            try:
                context = future.result()
            except Exception:
                continue
            _close_context(context)
        self._executor = None


def _source_params(run: str) -> dict[str, Any]:
    candidates = (
        final_selection.DEFAULT_OUTPUT_ROOT / "selected" / f"final_full_scale_{run}.json",
        final_selection.DEFAULT_OUTPUT_ROOT
        / "provisional"
        / f"final_full_scale_{run}.json",
        final_selection._source_final_path(run),
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        params = payload.get("params")
        if isinstance(params, dict) and params:
            return dict(params)
    raise FileNotFoundError(f"No final source parameters found for {run}")


def _seed_params_from_campaign_roots(
    run: str,
    roots: Sequence[str],
) -> list[dict[str, Any]]:
    """Load selected and finalist parameters from newer campaign layouts."""

    rows: list[dict[str, Any]] = []
    for raw_root in roots:
        root = Path(raw_root)
        summaries = sorted(
            root.glob(f"track_b_optuna/*/{run}/optimization_summary.json")
        )
        for path in summaries:
            payload = _read_status_json(path, required=True)
            for candidate in (
                payload.get("selected"),
                *(payload.get("finalists") or []),
            ):
                params = candidate.get("params") if isinstance(candidate, Mapping) else None
                if isinstance(params, dict) and params:
                    rows.append(dict(params))
        for path in (
            root / "selected" / f"final_full_scale_{run}.json",
            root / "provisional" / f"final_full_scale_{run}.json",
        ):
            if not path.is_file():
                continue
            payload = _read_status_json(path, required=True)
            params = payload.get("params")
            if isinstance(params, dict) and params:
                rows.append(dict(params))
    return rows


def _historical_source_params(
    run: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Return the selected winner followed by ranked historical candidates."""

    if limit < 1:
        raise ValueError("historical seed count must be at least one")
    rows: list[dict[str, Any]] = [_source_params(run)]
    manifest_path = final_selection.DEFAULT_OUTPUT_ROOT / "selection_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        run_payload = manifest.get("runs", {}).get(run, {})
        for candidate in run_payload.get("replay_candidates", []):
            raw_path = Path(str(candidate.get("candidate_path", "")))
            paths = (
                raw_path,
                final_selection.DEFAULT_OUTPUT_ROOT
                / "replay_candidates"
                / run
                / raw_path.name,
            )
            path = next((item for item in paths if item.is_file()), None)
            if path is None:
                continue
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            params = payload.get("params")
            if isinstance(params, dict) and params:
                rows.append(dict(params))

    if len(rows) < limit:
        rows.extend(
            dict(candidate.params)
            for candidate in _collect_replay_candidates(run)
        )

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for params in rows:
        digest = _params_hash(params)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(params)
        if len(unique) >= limit:
            break
    return unique


def _collect_replay_candidates(run: str) -> list[Any]:
    candidates = final_selection.collect_candidates(run)
    if candidates:
        return candidates

    history = final_selection._history_path(run)
    control_hash = _params_hash(_source_params(run))
    by_hash: dict[str, Any] = {}
    with history.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("status", "")).strip().lower() != (
                "missing-late-gas-window"
            ):
                continue
            params = selection._parse_mapping(row.get("params"))
            metrics = selection._parse_mapping(row.get("metrics"))
            if not params or not metrics:
                continue
            try:
                diagnostics = selection.candidate_diagnostics(metrics)
                history_iter = int(row["iter"])
                objective = float(row["objective"])
            except (KeyError, TypeError, ValueError):
                continue
            digest = _params_hash(params)
            candidate = selection.Candidate(
                run=run,
                source="final-production-missing-late-gas-window",
                history_path=str(history),
                history_iter=history_iter,
                history_objective=objective,
                params=params,
                metrics=metrics,
                params_sha256=digest,
                diagnostics=diagnostics,
                guardrail_failures=selection.training_guardrail_failures(
                    diagnostics
                ),
                source_final=digest == control_hash,
            )
            previous = by_hash.get(digest)
            if previous is None or (
                diagnostics["selection_score"], history_iter
            ) < (
                previous.diagnostics["selection_score"],
                previous.history_iter,
            ):
                by_hash[digest] = candidate
    return sorted(
        by_hash.values(),
        key=lambda candidate: (
            candidate.diagnostics["selection_score"],
            candidate.history_iter,
        ),
    )


def _project_seed(
    raw_params: Mapping[str, Any], param_space: Sequence[Mapping[str, Any]]
) -> dict[str, float | int]:
    facies_values: dict[int, list[float]] = {}
    for name, value in raw_params.items():
        match = calibration._FACIES_SIGNAL_PARAM_RE.match(str(name))
        if match is None:
            continue
        index = int(match.group("idx"))
        facies_values.setdefault(index, []).append(float(value))

    projected: dict[str, float | int] = {}
    for entry in param_space:
        name = str(entry["name"])
        low, high = (float(value) for value in entry["bounds"])
        if name in raw_params:
            raw = float(raw_params[name])
        else:
            shared = calibration._SHARED_SIGNAL_PARAM_RE.match(name)
            if shared is not None:
                index = int(shared.group("idx"))
                values = facies_values.get(index, [])
                raw = float(sum(values) / len(values)) if values else 0.5 * (low + high)
            else:
                raw = 0.5 * (low + high)
        value = min(high, max(low, raw))
        projected[name] = int(round(value)) if entry.get("type") == "int" else value

    # Averaging facies shapes preserves monotonicity in exact arithmetic, but
    # projection also protects against legacy rows with tiny round-off reversals.
    for _label, entries in calibration._value_entries_by_label(param_space).items():
        previous: float | None = None
        for entry in entries:
            name = str(entry["name"])
            value = float(projected[name])
            if previous is not None:
                value = max(previous, value)
            value = min(float(entry["bounds"][1]), value)
            projected[name] = value
            previous = value
    return projected


def _optuna_fixed_params(
    params: Mapping[str, Any], param_space: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    fixed: dict[str, Any] = {}
    signal_names: set[str] = set()
    for _label, entries in calibration._value_entries_by_label(param_space).items():
        lower, upper = calibration._monotonic_bounds(entries)
        previous: float | None = None
        for index, entry in enumerate(entries):
            name = str(entry["name"])
            signal_names.add(name)
            low, high = (float(value) for value in entry["bounds"])
            value = float(params[name])
            if low == high:
                previous = value
                continue
            minimum = lower[index] if previous is None else max(lower[index], previous)
            maximum = upper[index]
            fraction = (
                0.0
                if maximum <= minimum
                else (value - minimum) / (maximum - minimum)
            )
            fixed[f"cum_{name}"] = min(1.0, max(0.0, fraction))
            previous = value
    for entry in param_space:
        name = str(entry["name"])
        if name not in signal_names:
            fixed[name] = params[name]
    return fixed


def _candidate_diagnostics(
    metrics: Mapping[str, Any],
    *,
    objective_policy: str,
) -> tuple[dict[str, float], list[str]]:
    diagnostics = selection.candidate_diagnostics(metrics)
    resolved_policy = normalize_policy(objective_policy)
    failures = (
        selection.training_guardrail_failures_v2(
            diagnostics,
            include_late_gas=uses_late_gas_guardrail(resolved_policy),
        )
        if is_physical_v2_policy(resolved_policy)
        else selection.training_guardrail_failures(diagnostics)
    )
    return diagnostics, failures


def _spatial_rank_score(
    diagnostics: Mapping[str, float],
    objective_terms: Mapping[str, float],
    metrics: Mapping[str, Any],
) -> float:
    maximum_injected = max(
        (float(value["injected_full"]) for value in metrics.values()),
        default=1.0,
    )
    denominator = max(maximum_injected * max(1, len(metrics)), 1e-12)
    normalized_spatial = float(
        objective_terms.get("spatial_raw_penalty_mass_kg", 0.0)
    ) / denominator
    return float(diagnostics["selection_score"]) + normalized_spatial


def _compact_result_row(
    *,
    phase: str,
    sequence: int,
    params: Mapping[str, Any],
    result: Any,
    elapsed_seconds: float,
    source: Mapping[str, Any] | None = None,
    objective_policy: str = "evaluation",
    execution_backend: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = _metrics_payload(result.metrics)
    diagnostics, failures = (
        _candidate_diagnostics(metrics, objective_policy=objective_policy)
        if metrics
        else ({}, ["missing_metrics"])
    )
    terms = _objective_terms_subset(result.objective_terms)
    rank_score = (
        _spatial_rank_score(diagnostics, terms, metrics)
        if diagnostics and metrics
        else math.inf
    )
    search_score = score_search_objective(
        policy=objective_policy,
        evaluation_objective=float(result.objective),
        status=str(result.status),
        feasible=bool(result.feasible),
        diagnostics=diagnostics,
        guardrail_failures=failures,
        objective_terms=terms,
        metrics=metrics,
    )
    compact = {
        "phase": phase,
        "sequence": int(sequence),
        "params_sha256": _params_hash(params),
        "params": dict(params),
        "objective": float(result.objective),
        "search_objective": float(search_score.value),
        "search_objective_policy": normalize_policy(objective_policy),
        "search_objective_terms": search_score.to_dict(),
        "feasible": bool(result.feasible),
        "status": str(result.status),
        "diagnostics": diagnostics,
        "guardrail_failures": failures,
        "objective_terms": terms,
        "spatial_rank_score": float(rank_score),
        "elapsed_seconds": float(elapsed_seconds),
        "source": dict(source or {}),
    }
    if execution_backend is not None:
        numerical_spec = backend_spec(execution_backend)
        compact["evaluation_backend"] = numerical_spec.name
        compact["execution_key"] = numerical_spec.execution_key
        compact["study_compatibility_key"] = (
            numerical_spec.study_compatibility_key
        )
    full = {**compact, "metrics": metrics}
    return compact, full


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    objective_terms = row.get("objective_terms") or {}
    upper_bound_active = bool(
        float(objective_terms.get("overflow_gas_upper_bound_active", 0.0) or 0.0)
        >= 0.5
    )
    upper_bound_excess = max(
        0.0,
        float(
            objective_terms.get(
                "overflow_gas_hard_upper_bound_max_excess_fraction",
                0.0,
            )
            or 0.0
        ),
    )
    overflow_key = (
        bool(upper_bound_active and upper_bound_excess > 1e-12),
        upper_bound_excess if upper_bound_active else 0.0,
    )
    policy = normalize_policy(str(row.get("search_objective_policy", "evaluation")))
    if policy == "aligned-spatial-v1":
        return (
            *overflow_key,
            float(row.get("search_objective", math.inf)),
            float(row.get("objective", math.inf)),
            str(row.get("params_sha256", "")),
        )
    return (
        *overflow_key,
        bool(row.get("guardrail_failures")),
        not bool(row.get("feasible")),
        str(row.get("status")) != "ok",
        float(row.get("spatial_rank_score", math.inf)),
        float(row.get("objective", math.inf)),
        str(row.get("params_sha256", "")),
    )


def _eligible_training_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return only physically evaluated rows that may become finalists."""

    def satisfies_overflow_upper_bound(row: Mapping[str, Any]) -> bool:
        terms = row.get("objective_terms") or {}
        if not bool(
            float(terms.get("overflow_gas_upper_bound_active", 0.0) or 0.0)
            >= 0.5
        ):
            return True
        return (
            max(
                0.0,
                float(
                    terms.get(
                        "overflow_gas_hard_upper_bound_max_excess_mass_kg",
                        0.0,
                    )
                    or 0.0
                ),
            )
            <= 1e-12
        )

    return sorted(
        (
            row
            for row in rows
            if row.get("status") == "ok" and bool(row.get("feasible"))
            and not row.get("guardrail_failures")
            and satisfies_overflow_upper_bound(row)
        ),
        key=_row_sort_key,
    )


def _select_holdout_finalist(
    finalists: Sequence[Mapping[str, Any]],
    holdout_by_hash: Mapping[str, Mapping[str, Any]],
    *,
    objective_policy: str = "aligned-spatial-v1",
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Choose a valid validation result using the policy's selection contract."""

    physical_v2 = is_physical_v2_policy(objective_policy)
    valid: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for candidate in finalists:
        candidate_hash = str(candidate["params_sha256"])
        holdout = holdout_by_hash.get(candidate_hash)
        if holdout is None:
            continue
        if holdout.get("status") == "ok" and bool(holdout.get("feasible")):
            pair = (candidate, holdout)
            if (
                not physical_v2
                and not holdout.get("holdout_guardrail_failures")
            ):
                return pair
            valid.append(pair)
    if physical_v2:
        passing = [
            pair
            for pair in valid
            if not pair[1].get("holdout_guardrail_failures")
        ]
        if not passing:
            raise RuntimeError(
                "No finalist passed the objective-v2 validation guardrails"
            )

        def robust_key(
            pair: tuple[Mapping[str, Any], Mapping[str, Any]],
        ) -> tuple[float, ...]:
            training, validation = pair
            training_terms = training.get("search_objective_terms") or {}
            validation_terms = validation.get("search_objective_terms") or {}
            training_predictive = float(
                training_terms.get("mass_tie_breaker", math.inf)
            ) + float(training_terms.get("normalized_spatial_penalty", math.inf))
            validation_predictive = float(
                validation_terms.get("mass_tie_breaker", math.inf)
            ) + float(validation_terms.get("normalized_spatial_penalty", math.inf))
            physics = float(
                training_terms.get("normalized_physical_penalty", 0.0)
            )
            gain = float(training_terms.get("gain_regularization_score", 0.0))
            robust_predictive = max(training_predictive, validation_predictive)
            generalization_gap = abs(training_predictive - validation_predictive)
            return (
                robust_predictive + 0.25 * generalization_gap + physics + gain,
                robust_predictive,
                generalization_gap,
                float(training.get("search_objective", math.inf)),
                str(training.get("params_sha256", "")),
            )

        return min(passing, key=robust_key)
    if valid:
        return valid[0]
    raise RuntimeError("No finalist produced a valid holdout evaluation")


def _write_progress(
    output_root: Path,
    job_id: str,
    *,
    completed: int,
    total: int,
    stage: str,
    started: float,
    context_build_timing: Mapping[str, Any] | None = None,
    trial_timing: Mapping[str, Any] | None = None,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = max(0, total - completed)
    payload: dict[str, Any] = {
            "schema": 1,
            "job_id": job_id,
            "stage": stage,
            "completed": completed,
            "total": total,
            "fraction": completed / total if total else 1.0,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "eta_seconds": remaining / rate if rate > 0.0 else None,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "updated_utc": _utc_now(),
        }
    if context_build_timing is not None:
        payload["context_build_timing"] = dict(context_build_timing)
    if trial_timing is not None:
        payload["trial_timing"] = dict(trial_timing)
    try:
        # Progress is observational, not scientific state.  A transient NAS
        # outage must not hold an accelerator idle for the full retry window;
        # durable local trials and bounded checkpoints remain authoritative.
        _atomic_json(
            output_root / "progress" / f"{job_id}.json",
            payload,
            attempts=1,
        )
    except OSError:
        pass


def _lock_job_backend(
    output_root: Path,
    *,
    job_id: str,
    backend: str,
    state_paths: Sequence[Path],
) -> dict[str, Any]:
    """Lock one campaign job to one backend before any evaluation."""

    contract_path = output_root / "backend_contracts" / f"{job_id}.json"
    if not contract_path.is_file():
        progress = _read_status_json(
            output_root / "progress" / f"{job_id}.json",
            required=False,
        )
        has_prior_state = int(progress.get("completed", 0) or 0) > 0 or any(
            path.is_file() and path.stat().st_size > 0 for path in state_paths
        )
        if has_prior_state:
            raise RuntimeError(
                f"{job_id} contains evaluations but has no backend contract. "
                "Archive/discard it and start a new study; backend adoption is "
                "intentionally not inferred."
            )
    required_extra = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_engine_contract": (
            "local-cache-validation-prefetch-cached-tpe-v4"
        ),
        "sampler_contract": SAMPLER_CONTRACT,
        "preprocessing_source_sha256": _preprocessing_source_fingerprint(),
    }
    contract = lock_backend_contract(
        contract_path,
        backend,
        scope="campaign-job",
        identity=job_id,
        extra=required_extra,
    )
    actual_extra = contract.get("extra") or {}
    for name, expected in required_extra.items():
        if str(actual_extra.get(name, "")) != str(expected):
            raise RuntimeError(
                f"Calibration engine contract mismatch for {job_id}: "
                f"{name}={actual_extra.get(name)!r}, expected {expected!r}. "
                "Start a new campaign instead of resuming incompatible state."
            )
    return contract


def _run_replay_job(
    output_root: Path,
    *,
    run: str,
    backend: str,
    replay_limit: int,
    resources: _ContextResources | None = None,
) -> dict[str, Any]:
    job_id = f"replay_{run}"
    folder = output_root / "track_a_replay" / run
    rows_path = folder / "candidate_replay.jsonl"
    backend_contract = _lock_job_backend(
        output_root,
        job_id=job_id,
        backend=backend,
        state_paths=(rows_path,),
    )
    existing = _read_jsonl(rows_path)
    completed_hashes = {str(row.get("params_sha256")) for row in existing}
    candidates = _collect_replay_candidates(run)
    if replay_limit > 0:
        candidates = candidates[:replay_limit]
    started = time.perf_counter()
    context = None
    backend_lease: LocalSlotLease | None = None
    full_by_hash: dict[str, dict[str, Any]] = {}
    resources = resources or _ContextResources(backend=backend)
    try:
        spec = SEARCH_SPECS[0]
        context = resources.build(run, spec)
        backend_lease = resources.activate(context)
        for index, candidate in enumerate(candidates):
            if candidate.params_sha256 in completed_hashes:
                continue
            trial_started = time.perf_counter()
            result = calibration.evaluate_run(context, dict(candidate.params))
            compact, full = _compact_result_row(
                phase="history_replay",
                sequence=candidate.history_iter,
                params=candidate.params,
                result=result,
                elapsed_seconds=time.perf_counter() - trial_started,
                source={
                    "history_path": candidate.history_path,
                    "history_iter": candidate.history_iter,
                    "history_objective": candidate.history_objective,
                    "source_final": candidate.source_final,
                },
                execution_backend=backend,
            )
            _append_jsonl(rows_path, compact)
            existing.append(compact)
            full_by_hash[compact["params_sha256"]] = full
            completed_hashes.add(compact["params_sha256"])
            if len(completed_hashes) % 25 == 0:
                _write_progress(
                    output_root,
                    job_id,
                    completed=len(completed_hashes),
                    total=len(candidates),
                    stage="history_replay",
                    started=started,
                )
        eligible = [
            row
            for row in existing
            if row.get("status") == "ok" and bool(row.get("feasible"))
        ]
        ranked = sorted(eligible or existing, key=_row_sort_key)
        if not ranked:
            raise RuntimeError(f"No replay result was produced for {run}")
        top_rows = ranked[:20]
        winner = top_rows[0]
        winner_full = full_by_hash.get(str(winner["params_sha256"]))
        if winner_full is None:
            trial_started = time.perf_counter()
            result = calibration.evaluate_run(context, dict(winner["params"]))
            _compact, winner_full = _compact_result_row(
                phase="history_replay_final",
                sequence=int(winner["sequence"]),
                params=winner["params"],
                result=result,
                elapsed_seconds=time.perf_counter() - trial_started,
                source=winner.get("source") or {},
                execution_backend=backend,
            )
        summary = {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "track": "A",
            "run": run,
            "status": "complete",
            "backend": backend,
            "backend_contract": backend_contract,
            "candidate_count": len(candidates),
            "evaluated_count": len(existing),
            "winner": winner_full,
            "top_candidates": top_rows,
            "spatial_support_provenance": (
                context.spatial_observation_support_provenance
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "completed_utc": _utc_now(),
        }
        _atomic_json(folder / "replay_summary.json", summary)
        _write_progress(
            output_root,
            job_id,
            completed=len(candidates),
            total=len(candidates),
            stage="complete",
            started=started,
        )
        return summary
    finally:
        try:
            if context is not None:
                _close_context(context)
        finally:
            resources.release(backend_lease)


def _study_for_job(
    *,
    job_id: str,
    run: str,
    spec: SearchSpec,
    warmups: Sequence[Mapping[str, Any]],
    historical_seed_hashes: Sequence[str],
    param_space: Sequence[Mapping[str, Any]],
    journal_backend: BatchedJournalBackend,
) -> optuna.Study:
    sampler_kwargs: dict[str, Any] = {
        "seed": _derived_seed(run, spec.seed_group or spec.name),
        "multivariate": True,
        "group": True,
    }
    if _uses_objective_v2(spec):
        sampler_kwargs.update(
            {
                "n_startup_trials": 64,
                "n_ei_candidates": 96,
                "gamma": _v2_gamma,
                "constraints_func": _trial_constraints,
            }
        )
    sampler = CachedGroupedTPESampler(**sampler_kwargs)
    storage = JournalStorage(journal_backend)
    study = optuna.create_study(
        direction="minimize",
        study_name=job_id,
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )
    if not bool(study.user_attrs.get("warmups_enqueued")):
        for params in warmups:
            study.enqueue_trial(_optuna_fixed_params(params, param_space))
        study.set_user_attr("warmups_enqueued", True)
        study.set_user_attr("warmup_count", len(warmups))
        study.set_user_attr(
            "historical_seed_hashes",
            list(historical_seed_hashes),
        )
        study.set_user_attr("campaign_id", CAMPAIGN_ID)
        study.set_user_attr(
            "objective_policy", normalize_policy(spec.objective_policy)
        )
        study.set_user_attr(
            "sampler_profile",
            "v2-broad-tpe-cached" if _uses_objective_v2(spec) else "legacy-tpe-cached",
        )
        study.set_user_attr("sampler_contract", SAMPLER_CONTRACT)
        study.set_user_attr(
            "storage_contract", "local-batched-journal-generation-v3-async"
        )
        journal_backend.checkpoint()
    expected = int(study.user_attrs.get("warmup_count", -1))
    if expected != len(warmups):
        raise RuntimeError(
            f"Persisted warmup contract changed for {job_id}: {expected} != {len(warmups)}"
        )
    expected_seeds = list(study.user_attrs.get("historical_seed_hashes", []))
    if expected_seeds != list(historical_seed_hashes):
        raise RuntimeError(
            "Persisted historical-seed contract changed for "
            f"{job_id}: {expected_seeds} != {list(historical_seed_hashes)}"
        )
    persisted_policy = study.user_attrs.get("objective_policy")
    if persisted_policy is not None and str(persisted_policy) != normalize_policy(
        spec.objective_policy
    ):
        raise RuntimeError(
            f"Persisted objective policy changed for {job_id}: "
            f"{persisted_policy} != {normalize_policy(spec.objective_policy)}"
        )
    persisted_sampler = study.user_attrs.get("sampler_contract")
    if persisted_sampler is not None and str(persisted_sampler) != SAMPLER_CONTRACT:
        raise RuntimeError(
            f"Persisted sampler contract changed for {job_id}: "
            f"{persisted_sampler} != {SAMPLER_CONTRACT}"
        )
    return study


def _warmup_params(
    context: Any,
    run: str,
    spec: SearchSpec,
) -> tuple[list[dict[str, Any]], int]:
    raw_seeds = [
        *_seed_params_from_campaign_roots(run, spec.seed_campaign_roots),
        *_historical_source_params(
            run,
            limit=int(spec.historical_seed_count),
        ),
    ][: int(spec.historical_seed_count)]
    seeds = queue_support._merge_unique_params(
        [_project_seed(params, context.param_space) for params in raw_seeds],
        (),
    )
    generated = queue_support._generate_warmup_params(
        context,
        warmup_iters=RANDOM_WARMUPS,
        warmup_levels=None,
        warmup_levels_by_idx=None,
        warmup_levels_default=None,
        warmup_high=None,
        warmup_mode="independent",
        rng=random.Random(_derived_seed(run, "warmup")),
    )
    return queue_support._merge_unique_params(seeds, generated), len(seeds)


def _suggest_search_params(
    trial: optuna.Trial,
    param_space: Sequence[Mapping[str, Any]],
    *,
    spec: SearchSpec,
    optuna_index: int,
    max_iters: int,
) -> tuple[dict[str, Any], str]:
    """First learn response shape with neutral gains, then refine jointly."""

    stage1_count = int(math.ceil(max_iters * max(0.0, spec.stage1_gain_fraction)))
    if stage1_count <= 0 or optuna_index >= stage1_count:
        return calibration.suggest_params_trial(trial, param_space), "joint"

    gain_entries = [
        entry for entry in param_space if str(entry["name"]).endswith(".gain")
    ]
    reduced_space = [entry for entry in param_space if entry not in gain_entries]
    params = calibration.suggest_params_trial(trial, reduced_space)
    for entry in gain_entries:
        low, high = (float(value) for value in entry["bounds"])
        params[str(entry["name"])] = min(high, max(low, 1.0))
    return params, "shape-neutral-gains"


def _completed_trials(study: optuna.Study) -> int:
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )


def _recover_missing_study_rows(
    *,
    study: optuna.Study,
    rows_path: Path,
    context: Any,
    objective_policy: str,
) -> int:
    """Re-evaluate completed journal trials missing from the shared JSONL."""

    existing_sequences = {
        int(row["sequence"])
        for row in _read_jsonl(rows_path)
        if row.get("sequence") is not None
    }
    recovered = 0
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.number in existing_sequences:
            continue
        trial_record = trial.user_attrs.get("ffac_trial_record")
        params = (
            trial_record.get("evaluated_params")
            if isinstance(trial_record, Mapping)
            else trial.user_attrs.get("evaluated_params")
        )
        if not isinstance(params, Mapping):
            raise RuntimeError(
                f"Completed trial {trial.number} has no recoverable parameters"
            )
        started = time.perf_counter()
        result = calibration.evaluate_run(context, dict(params))
        phase = str(
            trial_record.get("phase", "optuna")
            if isinstance(trial_record, Mapping)
            else trial.user_attrs.get("phase", "optuna")
        )
        warmup_source = (
            trial_record.get("warmup_source")
            if isinstance(trial_record, Mapping)
            else trial.user_attrs.get("warmup_source")
        )
        source: dict[str, Any] = {"resume_recovery": True}
        if warmup_source is not None:
            source["warmup_source"] = warmup_source
        compact, _full = _compact_result_row(
            phase=phase,
            sequence=trial.number,
            params=params,
            result=result,
            elapsed_seconds=time.perf_counter() - started,
            source=source,
            objective_policy=objective_policy,
            execution_backend=getattr(
                context, "_evaluation_backend", None
            ),
        )
        persisted_value = float(trial.value)
        recovered_value = float(compact["search_objective"])
        if not math.isclose(
            persisted_value,
            recovered_value,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                "Recovered trial is not reproducible: "
                f"trial={trial.number}, persisted={persisted_value}, "
                f"recovered={recovered_value}"
            )
        _append_jsonl(rows_path, compact)
        existing_sequences.add(trial.number)
        recovered += 1
    return recovered


def _run_holdout(
    output_root: Path,
    *,
    run: str,
    spec: SearchSpec,
    backend: str,
    finalists: Sequence[Mapping[str, Any]],
    times_h: Sequence[float] | None = None,
    artifact_name: str = "holdout",
    resources: _ContextResources | None = None,
    prepared_context: Any | None = None,
) -> list[dict[str, Any]]:
    context = prepared_context
    backend_lease: LocalSlotLease | None = None
    rows: list[dict[str, Any]] = []
    resources = resources or _ContextResources(backend=backend)
    try:
        requested_times = tuple(times_h or selection.HOLDOUT_TIMES_H)
        if context is None:
            context = resources.build(
                run,
                spec,
                times=requested_times,
            )
        backend_lease = resources.activate(context)
        resolved_times_h = [float(item[2]) for item in context._loaded]
        evaluation_time_records = list(context.evaluation_time_records)
        selection._disable_holdout_objective_terms(context)
        for index, candidate in enumerate(finalists):
            started = time.perf_counter()
            result = calibration.evaluate_run(context, dict(candidate["params"]))
            compact, full = _compact_result_row(
                phase=artifact_name,
                sequence=index,
                params=candidate["params"],
                result=result,
                elapsed_seconds=time.perf_counter() - started,
                source={
                    "training_params_sha256": candidate["params_sha256"],
                    f"requested_{artifact_name}_times_h": list(requested_times),
                    f"resolved_{artifact_name}_times_h": resolved_times_h,
                    "evaluation_time_records": evaluation_time_records,
                },
                objective_policy=spec.objective_policy,
                execution_backend=backend,
            )
            full["holdout_guardrail_failures"] = (
                (
                    selection.holdout_guardrail_failures_v2(
                        full["diagnostics"],
                        include_late_gas=uses_late_gas_guardrail(
                            spec.objective_policy
                        ),
                    )
                    if _uses_objective_v2(spec)
                    else selection.holdout_guardrail_failures(
                        full["diagnostics"]
                    )
                )
                if full["diagnostics"]
                else ["missing_metrics"]
            )
            rows.append(full)
        _atomic_json(
            output_root
            / "track_b_optuna"
            / spec.name
            / run
            / f"{artifact_name}.json",
            {
                "schema": 1,
                "role": artifact_name,
                "requested_times_h": list(requested_times),
                "times_h": resolved_times_h,
                "evaluation_time_records": evaluation_time_records,
                "context_build_timing": dict(
                    getattr(context, "_context_build_timing", {}) or {}
                ),
                "candidates": rows,
            },
        )
        return rows
    finally:
        try:
            if context is not None:
                _close_context(context)
        finally:
            resources.release(backend_lease)


def _accumulate_timing(
    totals: dict[str, float],
    sample: Mapping[str, float],
) -> None:
    for name, value in sample.items():
        totals[name] = totals.get(name, 0.0) + float(value)


def _timing_summary(
    totals: Mapping[str, float],
    count: int,
) -> dict[str, Any]:
    divisor = max(1, int(count))
    return {
        "trial_count": int(count),
        "total_seconds": {name: float(value) for name, value in totals.items()},
        "mean_seconds": {
            name: float(value) / divisor for name, value in totals.items()
        },
    }


def _run_optimize_job(
    output_root: Path,
    *,
    run: str,
    spec: SearchSpec,
    backend: str,
    max_iters: int,
    resources: _ContextResources | None = None,
) -> dict[str, Any]:
    # Keep one exact process-wide contract active while the main thread
    # evaluates trials and the background thread prepares validation images.
    # Nested environment guards then restore the same values and cannot race
    # each other back to host defaults.
    environment, _cache_contract = _context_environment_values(run, spec, backend)
    with pilot.campaign_environment(spec.variant, environment):
        return _run_optimize_job_impl(
            output_root,
            run=run,
            spec=spec,
            backend=backend,
            max_iters=max_iters,
            resources=resources,
        )


def _run_optimize_job_impl(
    output_root: Path,
    *,
    run: str,
    spec: SearchSpec,
    backend: str,
    max_iters: int,
    resources: _ContextResources | None = None,
) -> dict[str, Any]:
    job_id = f"optimize_{spec.name}_{run}"
    folder = output_root / "track_b_optuna" / spec.name / run
    canonical_rows_path = folder / "trials.jsonl"
    job_cache = LocalJobCache(
        campaign_id=CAMPAIGN_ID,
        output_root=output_root,
        job_id=job_id,
        canonical_folder=folder,
    )
    rows_path = job_cache.rows_path
    backend_contract = _lock_job_backend(
        output_root,
        job_id=job_id,
        backend=backend,
        state_paths=(
            canonical_rows_path,
            job_cache.remote_dir / "study_journal.batched.jsonl",
        ),
    )
    started = time.perf_counter()
    context = None
    backend_lease: LocalSlotLease | None = None
    journal_backend: BatchedJournalBackend | None = None
    storage_provenance: dict[str, Any] | None = None
    timing_totals: dict[str, float] = {}
    timed_trials = 0
    resources = resources or _ContextResources(backend=backend)
    validation_prefetch: _ValidationContextPrefetch | None = None
    try:
        context = resources.build(run, spec)
        backend_lease = resources.activate(context)
        context_build_timing = dict(
            getattr(context, "_context_build_timing", {}) or {}
        )
        validation_prefetch = _ValidationContextPrefetch(
            resources=resources,
            run=run,
            spec=spec,
        )
        warmups, historical_seed_count = _warmup_params(context, run, spec)
        historical_seed_hashes = [
            _params_hash(params)
            for params in warmups[:historical_seed_count]
        ]
        journal_backend = BatchedJournalBackend(job_cache.journal_path)
        study = _study_for_job(
            job_id=job_id,
            run=run,
            spec=spec,
            warmups=warmups,
            historical_seed_hashes=historical_seed_hashes,
            param_space=context.param_space,
            journal_backend=journal_backend,
        )
        recovered_trial_rows = _recover_missing_study_rows(
            study=study,
            rows_path=rows_path,
            context=context,
            objective_policy=spec.objective_policy,
        )
        target_total = len(warmups) + max_iters
        completed = _completed_trials(study)
        best_live = job_cache.load_best()
        while completed < target_total:
            cycle_started = time.perf_counter()
            section_started = cycle_started
            trial = study.ask()
            ask_seconds = time.perf_counter() - section_started
            phase = "warmup" if trial.number < len(warmups) else "optuna"
            warmup_source = (
                "historical-seed"
                if trial.number < historical_seed_count
                else "structured-random"
            ) if phase == "warmup" else None
            trial_started = time.perf_counter()
            section_started = trial_started
            if phase == "optuna":
                params, search_stage = _suggest_search_params(
                    trial,
                    context.param_space,
                    spec=spec,
                    optuna_index=trial.number - len(warmups),
                    max_iters=max_iters,
                )
            else:
                params = calibration.suggest_params_trial(
                    trial, context.param_space
                )
                search_stage = "warmup"
            suggest_seconds = time.perf_counter() - section_started
            section_started = time.perf_counter()
            result = calibration.evaluate_run(context, params)
            evaluate_seconds = time.perf_counter() - section_started
            section_started = time.perf_counter()
            compact, full = _compact_result_row(
                phase=phase,
                sequence=trial.number,
                params=params,
                result=result,
                elapsed_seconds=time.perf_counter() - trial_started,
                source={
                    **({"warmup_source": warmup_source} if warmup_source else {}),
                    "search_stage": search_stage,
                },
                objective_policy=spec.objective_policy,
                execution_backend=backend,
            )
            score_seconds = time.perf_counter() - section_started
            section_started = time.perf_counter()
            trial.set_user_attr(
                "ffac_trial_record",
                {
                    "phase": phase,
                    "warmup_source": warmup_source,
                    "search_stage": search_stage,
                    "evaluated_params": dict(params),
                    "params_sha256": compact["params_sha256"],
                    "spatial_rank_score": compact["spatial_rank_score"],
                    "search_objective_policy": compact["search_objective_policy"],
                    "search_objective_stage": compact["search_objective_terms"][
                        "stage"
                    ],
                },
            )
            if _uses_objective_v2(spec):
                constraints = compact["search_objective_terms"].get(
                    "constraint_values", []
                )
                if len(constraints) != len(V2_CONSTRAINT_NAMES):
                    constraints = [1.0] * len(V2_CONSTRAINT_NAMES)
                trial.set_user_attr("constraint_values", list(constraints))
            attrs_seconds = time.perf_counter() - section_started
            section_started = time.perf_counter()
            study.tell(trial, float(compact["search_objective"]))
            tell_seconds = time.perf_counter() - section_started
            trial_timing = {
                "ask": ask_seconds,
                "suggest": suggest_seconds,
                "evaluate": evaluate_seconds,
                "score": score_seconds,
                "trial_attributes": attrs_seconds,
                "tell_and_local_journal": tell_seconds,
            }
            compact["timing"] = dict(trial_timing)
            full["timing"] = dict(trial_timing)
            section_started = time.perf_counter()
            job_cache.append_row(compact)
            local_row_seconds = time.perf_counter() - section_started
            completed += 1
            section_started = time.perf_counter()
            if result.status == "ok" and result.feasible:
                if best_live is None or _row_sort_key(full) < _row_sort_key(best_live):
                    job_cache.write_best(full)
                    best_live = full
            best_seconds = time.perf_counter() - section_started
            section_started = time.perf_counter()
            checkpoint = job_cache.checkpoint(
                completed_trials=completed,
                journal=journal_backend,
            )
            checkpoint_seconds = time.perf_counter() - section_started
            trial_timing.update(
                {
                    "local_row": local_row_seconds,
                    "best_update": best_seconds,
                    "shared_checkpoint": checkpoint_seconds,
                    "cycle": time.perf_counter() - cycle_started,
                }
            )
            _accumulate_timing(timing_totals, trial_timing)
            timed_trials += 1
            if completed % 25 == 0 or completed == target_total:
                _write_progress(
                    output_root,
                    job_id,
                    completed=completed,
                    total=target_total,
                    stage=phase,
                    started=started,
                    context_build_timing=context_build_timing,
                    trial_timing=_timing_summary(timing_totals, timed_trials),
                )

        job_cache.checkpoint(
            completed_trials=completed,
            journal=journal_backend,
            force=True,
        )
        rows = job_cache.load_rows()
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            digest = str(row.get("params_sha256", ""))
            previous = unique.get(digest)
            if previous is None or _row_sort_key(row) < _row_sort_key(previous):
                unique[digest] = row
        ranked = _eligible_training_rows(list(unique.values()))
        if not ranked:
            status_counts: dict[str, int] = {}
            for row in unique.values():
                key = f"{row.get('status')}|feasible={bool(row.get('feasible'))}"
                status_counts[key] = status_counts.get(key, 0) + 1
            raise RuntimeError(
                f"No valid Optuna result was produced for {spec.name}/{run}; "
                f"trial_statuses={status_counts}"
            )
        finalists = ranked[: max(1, int(spec.finalist_count))]
        color_correction_provenance = getattr(
            context, "color_correction_provenance", None
        )
        try:
            _close_context(context)
            context = None
        finally:
            resources.release(backend_lease)
            backend_lease = None
        validation_artifact = (
            "validation" if _uses_objective_v2(spec) else "holdout"
        )
        validation_times = spec.validation_times_h or selection.HOLDOUT_TIMES_H
        _write_progress(
            output_root,
            job_id,
            completed=target_total,
            total=target_total,
            stage=validation_artifact,
            started=started,
            context_build_timing=context_build_timing,
            trial_timing=_timing_summary(timing_totals, timed_trials),
        )
        validation_context = None
        if validation_prefetch is not None:
            try:
                validation_context = validation_prefetch.take("validation")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[{_utc_now()}] {job_id} validation prefetch failed; "
                    f"falling back to a synchronous build: {exc}",
                    flush=True,
                )
        holdout = _run_holdout(
            output_root,
            run=run,
            spec=spec,
            backend=backend,
            finalists=finalists,
            times_h=validation_times,
            artifact_name=validation_artifact,
            resources=resources,
            prepared_context=validation_context,
        )
        holdout_by_hash = {
            row["source"]["training_params_sha256"]: row for row in holdout
        }
        selected, selected_holdout = _select_holdout_finalist(
            finalists,
            holdout_by_hash,
            objective_policy=spec.objective_policy,
        )
        audit_rows: list[dict[str, Any]] = []
        selected_audit: dict[str, Any] | None = None
        if spec.audit_times_h:
            _write_progress(
                output_root,
                job_id,
                completed=target_total,
                total=target_total,
                stage="audit",
                started=started,
                context_build_timing=context_build_timing,
                trial_timing=_timing_summary(timing_totals, timed_trials),
            )
            audit_context = None
            if validation_prefetch is not None:
                try:
                    audit_context = validation_prefetch.take("audit")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[{_utc_now()}] {job_id} audit prefetch failed; "
                        f"falling back to a synchronous build: {exc}",
                        flush=True,
                    )
            audit_rows = _run_holdout(
                output_root,
                run=run,
                spec=spec,
                backend=backend,
                finalists=[selected],
                times_h=spec.audit_times_h,
                artifact_name="audit",
                resources=resources,
                prepared_context=audit_context,
            )
            if audit_rows:
                selected_audit = audit_rows[0]
        storage_provenance = job_cache.publish_final(
            completed_trials=completed,
            journal=journal_backend,
        )
        summary = {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "track": "B",
            "run": run,
            "variant": spec.name,
            "status": "complete",
            "backend": backend,
            "backend_contract": backend_contract,
            "random_warmups": RANDOM_WARMUPS,
            "historical_seed_count_requested": spec.historical_seed_count,
            "historical_seed_count": historical_seed_count,
            "historical_seed_hashes": historical_seed_hashes,
            "total_warmups": len(warmups),
            "optuna_trials": max_iters,
            "total_completed_trials": completed,
            "recovered_trial_rows": recovered_trial_rows,
            "selected": selected,
            "selected_holdout": selected_holdout,
            "holdout_passed": not selected_holdout["holdout_guardrail_failures"],
            "validation_role": validation_artifact,
            "validation_times_h": list(validation_times),
            "selected_audit": selected_audit,
            "audit_times_h": list(spec.audit_times_h),
            "audit_passed": (
                not selected_audit["holdout_guardrail_failures"]
                if selected_audit is not None
                else None
            ),
            "finalists": finalists,
            "environment_overrides": {
                str(name): str(value)
                for name, value in spec.environment_overrides
            },
            "run_environment_overrides": {
                str(raw_run): {
                    str(name): str(value) for name, value in overrides
                }
                for raw_run, overrides in spec.run_environment_overrides
            },
            "color_correction_provenance": color_correction_provenance,
            "context_build_timing": context_build_timing,
            "validation_context_prefetch": (
                validation_prefetch.snapshot()
                if validation_prefetch is not None
                else {"enabled": False}
            ),
            "trial_timing": _timing_summary(timing_totals, timed_trials),
            "storage_provenance": storage_provenance,
            "sampler_contract": SAMPLER_CONTRACT,
            "rationale": spec.rationale,
            "elapsed_seconds": time.perf_counter() - started,
            "completed_utc": _utc_now(),
        }
        _atomic_json(folder / "optimization_summary.json", summary)
        _atomic_json(
            folder / f"final_full_scale_{run}.json",
            {
                "schema": 1,
                "run": run,
                "variant": spec.name,
                "objective_full_scale": selected["objective"],
                "objective_terms": selected["objective_terms"],
                "params": selected["params"],
                "selection": summary,
            },
        )
        _write_progress(
            output_root,
            job_id,
            completed=target_total,
            total=target_total,
            stage="complete",
            started=started,
            context_build_timing=context_build_timing,
            trial_timing=_timing_summary(timing_totals, timed_trials),
        )
        return summary
    finally:
        if validation_prefetch is not None:
            validation_prefetch.close()
        if journal_backend is not None:
            journal_backend.checkpoint()
        job_cache.close(wait=True)
        try:
            if context is not None:
                _close_context(context)
        finally:
            resources.release(backend_lease)


def _job_manifest(
    *,
    replay_limit: int,
    max_iters: int,
    runs: Sequence[str] = RUNS,
    tracks: Sequence[str] = ("replay", "optimize"),
) -> dict[str, Any]:
    selected_runs = tuple(str(run).lower() for run in runs)
    selected_tracks = tuple(dict.fromkeys(str(track) for track in tracks))
    unknown_tracks = set(selected_tracks) - {"replay", "optimize"}
    if unknown_tracks:
        raise ValueError(f"Unknown campaign tracks: {sorted(unknown_tracks)}")
    jobs: list[dict[str, Any]] = []
    if "replay" in selected_tracks:
        for run in selected_runs:
            jobs.append(
                {
                    "job_id": f"replay_{run}",
                    "track": "replay",
                    "run": run,
                    "replay_limit": replay_limit,
                }
            )
    if "optimize" in selected_tracks:
        if OPTIMIZE_JOB_ORDER == "run-major":
            groups: dict[str, list[SearchSpec]] = {}
            for spec in SEARCH_SPECS:
                key = spec.preprocessing_cache_group or spec.name
                groups.setdefault(key, []).append(spec)
            interleaved_specs = [
                group[index]
                for index in range(max(len(group) for group in groups.values()))
                for group in groups.values()
                if index < len(group)
            ]
            optimize_pairs = (
                (spec, run) for run in selected_runs for spec in interleaved_specs
            )
        elif OPTIMIZE_JOB_ORDER == "variant-major":
            optimize_pairs = (
                (spec, run) for spec in SEARCH_SPECS for run in selected_runs
            )
        else:
            raise ValueError(
                f"Unknown optimize job order {OPTIMIZE_JOB_ORDER!r}."
            )
        for spec, run in optimize_pairs:
            jobs.append(
                {
                    "job_id": f"optimize_{spec.name}_{run}",
                    "track": "optimize",
                    "run": run,
                    "variant": spec.name,
                    "max_iters": max_iters,
                }
            )
    return {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
        "runs": list(selected_runs),
        "tracks": list(selected_tracks),
        "optimize_job_order": OPTIMIZE_JOB_ORDER,
        "spatial_settings": {
            "objective_weight": SPATIAL_OBJECTIVE_WEIGHT,
            "recall_weight": SPATIAL_RECALL_WEIGHT,
            "seal_weight": SPATIAL_SEAL_WEIGHT,
            "trace_level": SPATIAL_TRACE_LEVEL,
            "core_level": SPATIAL_CORE_LEVEL,
            "gain_sigma": GAIN_SIGMA,
        },
        "limited_temporal_coverage_runs": list(LIMITED_TEMPORAL_COVERAGE_RUNS),
        "validation_context_policy": {
            "build_during_training": True,
            "retain_in_ram": True,
            "validation_audit_contexts_disjoint": True,
            "reason": "preserve temporal validation/audit isolation",
        },
        "variants": [
            {
                "name": spec.name,
                "bounds_file": spec.variant.bounds_file,
                "signal_parameterization": spec.variant.signal_parameterization,
                "gain_weight": spec.gain_weight,
                "color_path_evidence": spec.color_path_evidence,
                "objective_policy": normalize_policy(spec.objective_policy),
                "spatial_objective_weight": spec.spatial_objective_weight,
                "seed_group": spec.seed_group,
                "historical_seed_count": spec.historical_seed_count,
                "objective_integral": spec.objective_integral,
                "additional_calibration_times_h": list(
                    spec.additional_calibration_times_h
                ),
                "finalist_count": spec.finalist_count,
                "stage1_gain_fraction": spec.stage1_gain_fraction,
                "seed_campaign_roots": list(spec.seed_campaign_roots),
                "validation_times_h": list(spec.validation_times_h),
                "audit_times_h": list(spec.audit_times_h),
                "preprocessing_cache_group": spec.preprocessing_cache_group,
                "environment_overrides": {
                    str(name): str(value)
                    for name, value in spec.environment_overrides
                },
                "run_environment_overrides": {
                    str(raw_run): {
                        str(name): str(value) for name, value in overrides
                    }
                    for raw_run, overrides in spec.run_environment_overrides
                },
                "rationale": spec.rationale,
            }
            for spec in SEARCH_SPECS
        ],
        "jobs": jobs,
    }


def prepare_campaign(
    output_root: Path,
    *,
    replay_limit: int,
    max_iters: int,
    force: bool,
    runs: Sequence[str] = RUNS,
    tracks: Sequence[str] = ("replay", "optimize"),
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if force:
        _archive_existing_campaign_state(output_root)
    manifest_path = output_root / "campaign_manifest.json"
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        expected_runs = [str(run).lower() for run in runs]
        expected_tracks = list(dict.fromkeys(str(track) for track in tracks))
        actual_tracks = manifest.get("tracks") or sorted(
            {str(job["track"]) for job in manifest.get("jobs") or []}
        )
        if manifest.get("runs") != expected_runs or set(actual_tracks) != set(
            expected_tracks
        ):
            raise RuntimeError(
                "Existing campaign manifest does not match the requested runs/"
                "tracks. Use a new output root or --force."
            )
        return manifest
    errors: list[str] = []
    selected_runs = tuple(str(run).lower() for run in runs)
    for run in selected_runs:
        if not final_selection._history_path(run).is_file():
            errors.append(f"{run}: missing final history")
        try:
            _source_params(run)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{run}: {exc}")
    if "optimize" in tracks:
        for spec in SEARCH_SPECS:
            if not (REPO_ROOT / spec.variant.bounds_file).is_file():
                errors.append(f"{spec.name}: missing {spec.variant.bounds_file}")
            for root in spec.seed_campaign_roots:
                if not Path(root).is_dir():
                    errors.append(f"{spec.name}: missing seed campaign {root}")
    if errors:
        raise RuntimeError("Campaign preflight failed: " + "; ".join(errors))
    manifest = _job_manifest(
        replay_limit=replay_limit,
        max_iters=max_iters,
        runs=selected_runs,
        tracks=tracks,
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def _archive_existing_campaign_state(output_root: Path) -> Path | None:
    """Archive all mutable campaign state before a forced clean restart."""

    candidates = [
        output_root / "campaign_manifest.json",
        output_root / "claims",
        output_root / "done",
        output_root / "failed",
        output_root / "discarded",
        output_root / "backend_contracts",
        output_root / "progress",
        output_root / "track_a_replay",
        output_root / "track_b_optuna",
        *sorted(output_root.glob("launcher_*.json")),
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive = output_root / "archives" / f"forced_reset_{timestamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        os.replace(path, archive / path.name)
    return archive


def _claim_job(output_root: Path, job: Mapping[str, Any], worker_id: str) -> bool:
    if (output_root / "done" / f"{job['job_id']}.json").is_file():
        return False
    if (output_root / "failed" / f"{job['job_id']}.json").is_file():
        return False
    if (output_root / "discarded" / f"{job['job_id']}.json").is_file():
        return False
    claim = output_root / "claims" / str(job["job_id"])
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
    except FileExistsError:
        return False
    _atomic_json(
        claim / "lease.json",
        {
            "schema": 1,
            "job_id": job["job_id"],
            "worker_id": worker_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_utc": _utc_now(),
        },
    )
    return True


def _all_terminal(output_root: Path, jobs: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        (output_root / "done" / f"{job['job_id']}.json").is_file()
        or (output_root / "failed" / f"{job['job_id']}.json").is_file()
        or (output_root / "discarded" / f"{job['job_id']}.json").is_file()
        for job in jobs
    )


def worker_loop(
    output_root: Path,
    *,
    backend: str,
    prefer_track: str,
    once: bool,
    slot_root: Path | None = None,
    backend_slots: int = 0,
    context_build_slots: int = 1,
) -> dict[str, Any]:
    manifest = json.loads(
        (output_root / "campaign_manifest.json").read_text(encoding="utf-8-sig")
    )
    jobs = list(manifest["jobs"])
    worker_id = f"{socket.gethostname()}_{backend}_{prefer_track}_{os.getpid()}"
    processed: list[str] = []
    cache_affinity: tuple[str, str] | None = None
    backend_contract = backend_spec(backend)
    pipelined = (
        slot_root is not None
        and int(backend_slots) > 0
        and backend_contract.device_family != "cpu"
    )
    resources = _ContextResources(
        backend=backend,
        build_pool=(
            LocalSlotPool(
                slot_root,
                name="context-build",
                slots=max(1, int(context_build_slots)),
            )
            if pipelined
            else None
        ),
        backend_pool=(
            LocalSlotPool(
                slot_root,
                name="accelerator",
                slots=max(1, int(backend_slots)),
            )
            if pipelined
            else None
        ),
    )
    while True:
        ordered = [job for job in jobs if job["track"] == prefer_track]
        ordered.extend(job for job in jobs if job["track"] != prefer_track)
        if cache_affinity is not None:
            affinity_run, affinity_group = cache_affinity
            ordered.sort(
                key=lambda job: (
                    0
                    if job.get("track") == "optimize"
                    and str(job.get("run")) == affinity_run
                    and (
                        SPECS_BY_NAME[str(job["variant"])].preprocessing_cache_group
                        or str(job["variant"])
                    )
                    == affinity_group
                    else 1
                )
            )
        claimed = next(
            (
                job
                for job in ordered
                if _claim_job(output_root, job, worker_id)
            ),
            None,
        )
        if claimed is None:
            if _all_terminal(output_root, jobs) or once:
                break
            time.sleep(5.0)
            continue
        job_id = str(claimed["job_id"])
        if claimed.get("track") == "optimize":
            claimed_spec = SPECS_BY_NAME[str(claimed["variant"])]
            cache_affinity = (
                str(claimed["run"]),
                claimed_spec.preprocessing_cache_group or claimed_spec.name,
            )
        print(f"[{_utc_now()}] {worker_id} starting {job_id}", flush=True)
        try:
            if claimed["track"] == "replay":
                result = _run_replay_job(
                    output_root,
                    run=str(claimed["run"]),
                    backend=backend,
                    replay_limit=int(claimed.get("replay_limit", 0)),
                    resources=resources,
                )
            else:
                spec = SPECS_BY_NAME[str(claimed["variant"])]
                result = _run_optimize_job(
                    output_root,
                    run=str(claimed["run"]),
                    spec=spec,
                    backend=backend,
                    max_iters=int(claimed["max_iters"]),
                    resources=resources,
                )
            _atomic_json(
                output_root / "done" / f"{job_id}.json",
                {
                    "schema": 1,
                    "job": claimed,
                    "worker_id": worker_id,
                    "completed_utc": _utc_now(),
                    "result_path": str(
                        output_root
                        / (
                            "track_a_replay"
                            if claimed["track"] == "replay"
                            else "track_b_optuna"
                        )
                    ),
                    "status": result.get("status"),
                },
            )
            processed.append(job_id)
            print(f"[{_utc_now()}] {worker_id} completed {job_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            if _is_transient_network_error(exc):
                print(
                    f"[{_utc_now()}] {worker_id} PAUSED {job_id}: "
                    f"shared storage unavailable ({exc})",
                    flush=True,
                )
                traceback.print_exc()
                _wait_for_campaign_storage(output_root)
                claim_path = output_root / "claims" / job_id
                if claim_path.is_dir():
                    _archive_campaign_path(
                        claim_path,
                        output_root / "requeued" / "claims",
                    )
                print(
                    f"[{_utc_now()}] {worker_id} RESUMING {job_id}",
                    flush=True,
                )
                continue
            failure = {
                "schema": 1,
                "job": claimed,
                "worker_id": worker_id,
                "failed_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(output_root / "failed" / f"{job_id}.json", failure)
            print(f"[{_utc_now()}] {worker_id} FAILED {job_id}: {exc}", flush=True)
        if once:
            break
    if _all_terminal(output_root, jobs):
        cache_root = _corrected_image_cache_root()
        try:
            shutil.rmtree(cache_root)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(
                f"[{_utc_now()}] {worker_id} could not remove corrected-image "
                f"cache {cache_root}: {exc}",
                flush=True,
            )
    return {
        "worker_id": worker_id,
        "processed": processed,
        "pipelined": pipelined,
        "slot_root": str(slot_root) if slot_root is not None else None,
        "backend_slots": int(backend_slots),
        "context_build_slots": int(context_build_slots),
        "completed_utc": _utc_now(),
    }


def campaign_status(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "campaign_manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_prepared", "output_root": str(output_root)}
    manifest = _read_status_json(manifest_path, required=True)
    rows: list[dict[str, Any]] = []
    counts = {
        "pending": 0,
        "active": 0,
        "done": 0,
        "failed": 0,
        "discarded": 0,
    }
    for job in manifest["jobs"]:
        job_id = str(job["job_id"])
        if (output_root / "done" / f"{job_id}.json").is_file():
            state = "done"
        elif (output_root / "failed" / f"{job_id}.json").is_file():
            state = "failed"
        elif (output_root / "discarded" / f"{job_id}.json").is_file():
            state = "discarded"
        elif (output_root / "claims" / job_id).is_dir():
            state = "active"
        else:
            state = "pending"
        counts[state] += 1
        progress_path = output_root / "progress" / f"{job_id}.json"
        progress = _read_status_json(progress_path, required=False)
        rows.append(
            {
                "job_id": job_id,
                "track": job["track"],
                "run": job["run"],
                "variant": job.get("variant"),
                "state": state,
                "completed": progress.get("completed", 0),
                "total": progress.get("total"),
                "eta_seconds": progress.get("eta_seconds"),
            }
        )
    return {
        "schema": 1,
        "campaign_id": str(manifest.get("campaign_id", CAMPAIGN_ID)),
        "output_root": str(output_root),
        "counts": counts,
        "complete": (
            counts["done"] + counts["failed"] + counts["discarded"] == len(rows)
        ),
        "jobs": rows,
        "updated_utc": _utc_now(),
    }


def launch_workers(
    output_root: Path,
    *,
    workers: int,
    backend: str,
    prefetch_depth: int = -1,
    context_builders: int = 1,
) -> dict[str, Any]:
    local_logs = project_path('logs') / CAMPAIGN_ID / socket.gethostname()
    local_logs.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    processes: list[dict[str, Any]] = []
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    accelerator_slots = max(1, int(workers))
    accelerated = backend_spec(backend).device_family != "cpu"
    resolved_prefetch_depth = (
        (
            accelerator_slots
            if int(prefetch_depth) < 0
            else max(0, int(prefetch_depth))
        )
        if accelerated
        else 0
    )
    process_count = accelerator_slots + resolved_prefetch_depth
    slot_root = (
        _local_worker_slot_root(output_root, backend) if accelerated else None
    )
    for index in range(process_count):
        prefer = "replay" if index == 0 else "optimize"
        role = "active" if index < accelerator_slots else "prefetch"
        log_path = local_logs / f"worker_{index}_{role}_{prefer}.log"
        stream = log_path.open("a", encoding="utf-8", buffering=1)
        command = (
            str(python),
            str(WORKER_SCRIPT),
            "worker",
            "--output-root",
            str(output_root),
            "--backend",
            backend,
            "--prefer-track",
            prefer,
        )
        if slot_root is not None:
            command = (
                *command,
                "--slot-root",
                str(slot_root),
                "--backend-slots",
                str(accelerator_slots),
                "--context-builders",
                str(max(1, int(context_builders))),
            )
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=process_directory(),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )
        stream.close()
        processes.append(
            {
                "index": index,
                "pid": process.pid,
                "prefer_track": prefer,
                "role": role,
                "backend": backend,
                "log": str(log_path),
                "command": list(command),
            }
        )
    payload = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "host": socket.gethostname(),
        "started_utc": _utc_now(),
        "accelerator_slots": accelerator_slots if accelerated else 0,
        "worker_processes": process_count,
        "prefetch_depth": resolved_prefetch_depth,
        "context_builders": max(1, int(context_builders)),
        "slot_root": str(slot_root) if slot_root is not None else None,
        "processes": processes,
    }
    _atomic_json(local_logs / "launcher_processes.json", payload)
    _atomic_json(output_root / f"launcher_{socket.gethostname()}.json", payload)
    return payload


def stop_workers() -> dict[str, Any]:
    import psutil

    local_logs = project_path('logs') / CAMPAIGN_ID / socket.gethostname()
    manifest = local_logs / "launcher_processes.json"
    if not manifest.is_file():
        return {"status": "no_launcher_manifest", "path": str(manifest)}
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    stopped: list[int] = []
    missing: list[int] = []
    for row in payload.get("processes") or []:
        pid = int(row["pid"])
        if not psutil.pid_exists(pid):
            missing.append(pid)
            continue
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for process in reversed(children):
            process.terminate()
        parent.terminate()
        _gone, alive = psutil.wait_procs([*children, parent], timeout=5.0)
        for process in alive:
            process.kill()
        stopped.append(pid)
    return {"status": "stopped", "stopped": stopped, "already_missing": missing}


def _archive_campaign_path(path: Path, archive_dir: Path) -> Path:
    """Move a terminal marker or stale claim aside without losing provenance."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = archive_dir / f"{path.name}.{timestamp}"
    os.replace(path, destination)
    return destination


def discard_incomplete_jobs(
    output_root: Path,
    *,
    reason: str,
) -> dict[str, Any]:
    """Make every non-complete job terminal without deleting partial evidence."""

    manifest_path = output_root / "campaign_manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_prepared", "output_root": str(output_root)}
    manifest = _read_status_json(manifest_path, required=True)
    discarded_dir = output_root / "discarded"
    discarded_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    discarded: list[dict[str, Any]] = []
    already_discarded: list[str] = []

    for job in manifest.get("jobs") or []:
        job_id = str(job["job_id"])
        if (output_root / "done" / f"{job_id}.json").is_file():
            completed.append(job_id)
            continue
        marker = discarded_dir / f"{job_id}.json"
        if marker.is_file():
            already_discarded.append(job_id)
            continue

        archived: dict[str, str] = {}
        claim = output_root / "claims" / job_id
        if claim.is_dir():
            archived["claim"] = str(
                _archive_campaign_path(claim, discarded_dir / "claims")
            )
        failed = output_root / "failed" / f"{job_id}.json"
        if failed.is_file():
            archived["failure"] = str(
                _archive_campaign_path(failed, discarded_dir / "failures")
            )
        progress = _read_status_json(
            output_root / "progress" / f"{job_id}.json",
            required=False,
        )
        payload = {
            "schema": 1,
            "status": "discarded",
            "job": dict(job),
            "reason": str(reason),
            "discarded_utc": _utc_now(),
            "discarded_by": {
                "host": socket.gethostname(),
                "pid": os.getpid(),
            },
            "progress_at_discard": progress,
            "partial_artifacts_preserved": True,
            "archived_terminal_state": archived,
        }
        _atomic_json(marker, payload)
        discarded.append(
            {
                "job_id": job_id,
                "completed": int(progress.get("completed", 0) or 0),
                "total": progress.get("total"),
                "marker": str(marker),
            }
        )

    summary = {
        "schema": 1,
        "status": "discarded_incomplete",
        "campaign_id": str(manifest.get("campaign_id") or CAMPAIGN_ID),
        "output_root": str(output_root),
        "reason": str(reason),
        "completed_jobs_preserved": completed,
        "discarded_jobs": discarded,
        "already_discarded_jobs": already_discarded,
        "updated_utc": _utc_now(),
    }
    _atomic_json(output_root / "campaign_discarded.json", summary)
    return summary


def requeue_stale_jobs(output_root: Path) -> dict[str, Any]:
    """Requeue failed jobs and claims owned by dead workers on this host."""

    import psutil

    manifest_path = output_root / "campaign_manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_prepared", "output_root": str(output_root)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    host = socket.gethostname().casefold()
    requeued: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for job in manifest.get("jobs") or []:
        job_id = str(job["job_id"])
        if (output_root / "discarded" / f"{job_id}.json").is_file():
            skipped.append({"job_id": job_id, "reason": "discarded"})
            continue
        failed = output_root / "failed" / f"{job_id}.json"
        claim = output_root / "claims" / job_id
        if failed.is_file():
            archived_failure = _archive_campaign_path(
                failed,
                output_root / "requeued" / "failed",
            )
            archived_claim = None
            if claim.is_dir():
                archived_claim = _archive_campaign_path(
                    claim,
                    output_root / "requeued" / "claims",
                )
            requeued.append(
                {
                    "job_id": job_id,
                    "reason": "failed",
                    "archived_failure": str(archived_failure),
                    "archived_claim": str(archived_claim) if archived_claim else None,
                }
            )
            continue
        if not claim.is_dir():
            continue

        lease_path = claim / "lease.json"
        try:
            lease = json.loads(lease_path.read_text(encoding="utf-8-sig"))
            lease_host = str(lease.get("host", "")).casefold()
            lease_pid = int(lease.get("pid", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            skipped.append(
                {"job_id": job_id, "reason": f"unreadable-lease:{type(exc).__name__}"}
            )
            continue
        if lease_host != host:
            skipped.append({"job_id": job_id, "reason": "other-host"})
            continue

        process_alive = False
        if psutil.pid_exists(lease_pid):
            try:
                command = " ".join(psutil.Process(lease_pid).cmdline()).casefold()
                process_alive = (
                    WORKER_SCRIPT.name.casefold() in command and "worker" in command
                )
            except (psutil.Error, OSError):
                process_alive = False
        if process_alive:
            skipped.append({"job_id": job_id, "reason": "worker-alive"})
            continue

        archived_claim = _archive_campaign_path(
            claim,
            output_root / "requeued" / "claims",
        )
        requeued.append(
            {
                "job_id": job_id,
                "reason": "stale-local-claim",
                "archived_claim": str(archived_claim),
            }
        )

    return {
        "status": "requeued",
        "requeued": requeued,
        "skipped": skipped,
        "updated_utc": _utc_now(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "worker",
            "launch",
            "status",
            "stop",
            "discard-incomplete",
            "requeue-stale",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--backend",
        choices=all_backend_names(),
        default="cuda-float32",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=-1,
        help=(
            "Extra host-side contexts kept ready in RAM beyond active GPU slots. "
            "Defaults to one per GPU worker; use 0 to disable. Ignored for CPU "
            "backends."
        ),
    )
    parser.add_argument(
        "--context-builders",
        type=int,
        default=1,
        help="Maximum simultaneous cold/warm context builds on this host.",
    )
    parser.add_argument("--slot-root", type=Path)
    parser.add_argument("--backend-slots", type=int, default=0)
    parser.add_argument("--prefer-track", choices=("replay", "optimize"), default="optimize")
    parser.add_argument("--replay-limit", type=int, default=REPLAY_LIMIT)
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Run IDs, or the single value 'all'. Defaults to the six-run pilot.",
    )
    parser.add_argument(
        "--tracks",
        choices=("replay", "optimize", "both"),
        default="both",
        help="Prepare history replay, fresh Optuna searches, or both.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--discard-reason",
        default="superseded by explicit float32 GPU / float64 CPU backend contract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.runs and len(args.runs) == 1 and args.runs[0].casefold() == "all":
        runs = ALL_RUNS
    else:
        runs = tuple(str(run).lower() for run in (args.runs or RUNS))
    unknown_runs = sorted(set(runs) - set(ALL_RUNS))
    if unknown_runs:
        raise SystemExit(f"Unknown production run(s): {', '.join(unknown_runs)}")
    tracks = (
        ("replay", "optimize")
        if args.tracks == "both"
        else (args.tracks,)
    )
    if args.action == "prepare":
        result = prepare_campaign(
            args.output_root,
            replay_limit=max(0, args.replay_limit),
            max_iters=max(1, args.max_iters),
            force=args.force,
            runs=runs,
            tracks=tracks,
        )
    elif args.action == "launch":
        prepare_campaign(
            args.output_root,
            replay_limit=max(0, args.replay_limit),
            max_iters=max(1, args.max_iters),
            force=False,
            runs=runs,
            tracks=tracks,
        )
        result = launch_workers(
            args.output_root,
            workers=max(1, args.workers),
            backend=args.backend,
            prefetch_depth=args.prefetch_depth,
            context_builders=max(1, args.context_builders),
        )
    elif args.action == "worker":
        result = worker_loop(
            args.output_root,
            backend=args.backend,
            prefer_track=args.prefer_track,
            once=args.once,
            slot_root=args.slot_root,
            backend_slots=max(0, args.backend_slots),
            context_build_slots=max(1, args.context_builders),
        )
    elif args.action == "stop":
        result = stop_workers()
    elif args.action == "discard-incomplete":
        result = discard_incomplete_jobs(
            args.output_root,
            reason=args.discard_reason,
        )
    elif args.action == "requeue-stale":
        result = requeue_stale_jobs(args.output_root)
    else:
        result = campaign_status(args.output_root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

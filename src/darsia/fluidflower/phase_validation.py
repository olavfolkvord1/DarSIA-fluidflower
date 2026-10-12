"""Materialize model-bound phase-review assets from corrected source frames.

The source review campaign is treated as immutable evidence.  Generated phase
maps, active-learning proposals, and a compact compatibility manifest are
written to a model-specific cache that can be opened by the established
FluidFlower review application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import uuid
from dataclasses import replace

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_features import SPATIAL_CONTEXT_FRACTION
from darsia.fluidflower.phase_learning import (
    HierarchicalPhaseModel,
    propose_uncertain_phase_regions,
)
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.phase_rendering import (
    PHASE_DISPLAY_POLICY, two_phase_masks,
)
from darsia.fluidflower.phase_runtime import (
    load_physics_profile, load_run_context, physics_runtime_identity, phase_physics_sequence,
    bind_native_frame_context, bind_native_frame_reference,
)


PHASE_VALIDATION_CAMPAIGN_FORMAT = "darsia-phase-validation-campaign-v1"
PHASE_REVIEW_PROPOSAL_SCHEMA = 2
# Structured per-frame timing for callers that opt in with
# ``progress.accepts_frame_timing = True``; every other callback only sees text.
FRAME_TIMING_PREFIX = "PHASE_FRAME_TIMING "
# Output-only work of a finished frame (PNG encoding, review proposals, the production
# diagnostics archive) and the JPEG decode of the next frame run beside the ordered loop.
# Measured 2026-09-17: that work was 12-15 s of a 37 s loop period in review mode. The files
# are byte-identical to inline writing; set this variable to "0" to write inline as before.
BACKGROUND_OUTPUTS_ENV = "PHASE_BACKGROUND_OUTPUTS"
OUTPUT_EXECUTION_BACKGROUND = "background-thread-v1"
OUTPUT_EXECUTION_INLINE = "inline"
_PROPOSAL_FIELDS = ("untouched_water_probability", "aqueous_co2_probability", "free_co2_gas_probability")
# Review proposals are interpreter-bound. On a thread of this process they hold the interpreter
# lock and slow the physics (measured 2026-09-17 on real frames: +10-35 %), so they run in this
# many helper processes instead; "0" keeps them on the output thread. The proposals are identical.
PROPOSAL_PROCESSES_ENV = "PHASE_PROPOSAL_PROCESSES"
PHASE_FILL_COLORS = {
    "aqueous": (49, 217, 160, 255),
    "mixed": (255, 191, 71, 255),
    "gas": (242, 76, 157, 255),
    "unresolved": (30, 210, 255, 72),
}
_V5_METRIC_OUTPUT_GUARDS = frozenset({
    "bottom_gas_guard", "output_guards", "free_gas_cleanup",
    "region_edge_guard",
})


def _v5_physical_scale_warning(physics_profile: Mapping[str, Any] | None,
                               frame_bindings: Sequence[Mapping[str, Any]]) -> str | None:
    """Keep provisional metric geometry out of millimetre-based decisions.

    Optical review can proceed on independently checked native pixels while
    absolute rig/injector scale is still under audit. Position-dependent
    millimetre rules remain prohibited. A separate, explicitly nominal
    2.5 mm2 speckle filter is allowed only on V5 manual-review maps.
    """
    missing = [int(row.get("frame_index", index))
               for index, row in enumerate(frame_bindings)
               if row.get("physical_scale_qa_pass") is not True]
    if not missing:
        return None
    profile = physics_profile or {}
    requested = sorted(_V5_METRIC_OUTPUT_GUARDS & set(profile))
    nominal_review = profile.get("v5_nominal_review_gas_minimum")
    if requested == ["free_gas_cleanup"] and nominal_review == {
            "contract": "ffac-v5-nominal-review-gas-minimum-v1",
            "minimum_area_mm2": 2.5,
            "physical_scale_qa_pass": False,
            "mass_analysis_allowed": False,
            } and profile.get("free_gas_cleanup") == {
                "format": "native-free-gas-minimum-v1",
                "minimum_area_mm2": 2.5,
            }:
        return ("Physical rig/injector scale is unverified. The nominal 2.5 mm² "
                "free-gas speckle filter is for manual Studio review only; "
                "physical/mass interpretation is unavailable.")
    if requested:
        raise ValueError(
            "V5 physical scale is unverified for frames "
            f"{missing[:8]}; metric output guards require independent rig/"
            f"injector scale QA: {requested}")
    return ("Physical rig/injector scale is unverified; optical phase maps are "
            "for manual Studio review only and physical/mass interpretation "
            "is unavailable.")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object in {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(root: Path, value: object) -> Path:
    from .review_storage import resolve_review_path
    path = Path(str(value or ""))
    return resolve_review_path(path if path.is_absolute() else root / path)


def _model_identity(model_root: Path) -> tuple[str, str]:
    summary = _read_json(model_root / "summary.json")
    model_id = str(summary.get("model_id") or model_root.name).strip()
    digest = hashlib.sha256()
    for path in (
        model_root / "summary.json",
        model_root / "models" / "generic" / "co2_presence_signal.json",
        model_root / "models" / "generic" / "gas_within_co2_signal.json",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return model_id, digest.hexdigest()


def _source_jobs(source_root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(source_root / "postflight_manifest.json")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in manifest.get("jobs") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        run_id = str(row.get("run") or "").strip().casefold()
        if run_id:
            grouped.setdefault(run_id, []).append(row)
    selected: list[dict[str, Any]] = []
    for run_id, rows in sorted(grouped.items()):
        rows.sort(
            key=lambda row: (
                "physics" not in str(row.get("role") or "").casefold(),
                "reviewed" not in str(row.get("role") or "").casefold(),
                int(row.get("grid_order") or 999),
            )
        )
        selected.append({**rows[0], "run": run_id})
    if not selected:
        raise ValueError(f"No review candidates were found in {source_root}")
    return selected


def _candidate_paths(source_root: Path, job: Mapping[str, Any]) -> tuple[Path, Path]:
    run_id = str(job["run"]).casefold()
    variant = str(job.get("variant") or "")
    candidate = (
        _resolve(source_root, job["candidate_folder"])
        if job.get("candidate_folder")
        else (source_root / "postflight" / variant / run_id).resolve()
    )
    review_root = (
        _resolve(source_root, job["review_output_dir"])
        if job.get("review_output_dir")
        else (source_root / "manual_pairwise_review").resolve()
    )
    from .review_storage import resolve_review_path
    return resolve_review_path(candidate), resolve_review_path(review_root)


def _frame_timing_summary(run_id: str, frame_count: int, timings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Actual seconds of the ordered loop for the last frame and for the run so far.

    ``total_s`` spans loading the prepared photograph to the written phase map.
    Waiting (for a shared optical worker or CPU slot) is reported separately
    from work, and the optical worker's own seconds are not added because they
    overlap other frames.  Image preparation from the raw photograph happens
    before this loop and is reported per run by the series batch.
    """
    last = timings[-1]
    totals = sorted(float(row["total_s"]) for row in timings if row.get("total_s") is not None)
    median = totals[len(totals) // 2] if len(totals) % 2 else (totals[len(totals) // 2 - 1] + totals[len(totals) // 2]) / 2
    waited = float(last.get("optical_wait_s") or 0.0) + float(last.get("physics_slot_wait_s") or 0.0)
    physics = float(last.get("physics_s") or 0.0) - float(last.get("physics_slot_wait_s") or 0.0)
    decisions = [float(row["decision_s"]) for row in timings if row.get("decision_s") is not None]
    clock = last.get("optical_worker_clock") if isinstance(last.get("optical_worker_clock"), Mapping) else {}
    backend = clock.get("backend")
    done = len(timings)
    summary = {
        "run": run_id, "frame_index": int(last["frame_index"]), "frames_done": done, "frame_count": int(frame_count),
        "last_total_s": round(float(last["total_s"]), 2), "last_waiting_s": round(waited, 2),
        "last_physics_s": round(physics, 2), "last_writing_s": round(float(last.get("writing_s") or 0.0), 2),
        "last_optical_backend": backend, "last_optical_worker_s": last.get("optical_worker_total_s"),
        "median_total_s": round(median, 2), "elapsed_s": round(sum(totals), 1),
        "eta_s": round(median * max(0, int(frame_count) - done), 1),
    }
    optical = f"optical {backend} {float(summary['last_optical_worker_s']):.1f} s" if backend and summary[
        "last_optical_worker_s"] is not None else "optical"
    if last.get("decision_pipelined"):
        # physics_s is the chronological part; the final decision runs beside the next photograph.
        summary["last_decision_s"] = round(decisions[-1], 2) if decisions else None
        optical += ", final decision alongside" + (f" {decisions[-1]:.0f} s" if decisions else "")
    summary["text"] = (f"last image {summary['last_total_s']:.0f} s ({optical}, physics {physics:.0f} s, "
                       f"waited {waited:.0f} s) · median {median:.0f} s/image · "
                       f"{summary['elapsed_s'] / 60:.1f} min so far, about {summary['eta_s'] / 60:.0f} min left")
    return summary


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _select_native_current(prepared: np.ndarray | None, v5_input: Any | None,
                           frame_path: Path, frame_index: int,
                           frame_canvas: np.ndarray | None,
                           content_box: tuple[int, int, int, int]) -> np.ndarray:
    """Choose the exact prepared float frame or the source-bound V5 sidecar."""
    if prepared is not None:
        return prepared
    if v5_input is not None:
        return v5_input.load(frame_path, f"frame_{frame_index:03d}")
    if frame_canvas is None:
        raise ValueError("No native frame input is available")
    x, y, width, height = content_box
    return frame_canvas[y:y + height, x:x + width]


def predict_image_tiled(
    model: HierarchicalPhaseModel,
    image_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    *,
    previous_rgb: np.ndarray | None = None,
    clearwater_rgb: np.ndarray | None = None,
    tile_height: int = 384,
    texture_sigma: float = 2.0,
    radiometry_profile: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Apply one optical model in bounded-memory horizontal tiles."""

    current = np.asarray(image_rgb)
    baseline = np.asarray(baseline_rgb)
    if current.shape != baseline.shape or current.ndim != 3:
        raise ValueError("Current and baseline images must be matching RGB arrays")
    height, width = current.shape[:2]
    spatial_sigma = max(
        float(texture_sigma), SPATIAL_CONTEXT_FRACTION * float(min(height, width))
    )
    overlap = max(16, int(math.ceil(6.0 * spatial_sigma)))
    outputs: dict[str, np.ndarray] = {}
    for start in range(0, height, max(32, int(tile_height))):
        stop = min(height, start + max(32, int(tile_height)))
        extended_start = max(0, start - overlap)
        extended_stop = min(height, stop + overlap)
        selection = slice(extended_start, extended_stop)
        predicted = model.predict_image(
            current[selection],
            baseline[selection],
            previous_rgb=(
                None if previous_rgb is None else np.asarray(previous_rgb)[selection]
            ),
            clearwater_rgb=(
                None if clearwater_rgb is None else np.asarray(clearwater_rgb)[selection]
            ),
            texture_sigma=texture_sigma,
            spatial_context_sigma=spatial_sigma,
            **({"radiometry_profile": radiometry_profile} if radiometry_profile is not None else {}),
        )
        local = slice(start - extended_start, stop - extended_start)
        for name, values in predicted.items():
            output = outputs.setdefault(name, np.empty((height, width), dtype=np.float32))
            output[start:stop] = np.asarray(values[local], dtype=np.float32)
    return outputs


def _phase_masks(predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = two_phase_masks(predictions)
    # Keep the legacy empty layer path readable by older review clients.
    return {**masks, "mixed": np.zeros_like(masks["gas"])}


def _retain_v5_observation_only_scores(
    physical_predictions: Mapping[str, np.ndarray],
    optical_predictions: Mapping[str, np.ndarray],
    *,
    observable: np.ndarray,
    physics_supported: np.ndarray,
    certified_rig: np.ndarray | None = None,
    review_safe_support: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Keep native optical evidence where the physical context is unknown.

    The regional decision is valid only inside its source-bound facies and
    permeability support.  A missing physical context cannot erase an
    observed overflow label or invent a porous-sand mass interpretation.
    """
    observable = np.asarray(observable, dtype=bool)
    physical = np.asarray(physics_supported, dtype=bool)
    if (observable.shape != physical.shape or not observable.any()
            or np.any(physical & ~observable)):
        raise ValueError("V5 physical support must be a nonempty observation subset")
    # A photo-certified piece of the rig is not overflow.  Keep the uncertain
    # boundary eligible for observation so a source-backed post-injection
    # overflow signal is not erased by a hard sand-connectivity rule.
    if certified_rig is None:
        rig = np.zeros_like(observable)
    else:
        rig = np.asarray(certified_rig)
        if (rig.dtype != np.dtype("bool") or rig.shape != observable.shape
                or np.any(rig & physical)):
            raise ValueError("V5 certified rig conflicts with physical support")
    if review_safe_support is None:
        supported = observable
    else:
        safe = np.asarray(review_safe_support)
        if (safe.dtype != np.dtype("bool") or safe.shape != observable.shape
                or np.any(physical & ~safe)):
            raise ValueError("V5 review-safe source support conflicts with physical support")
        supported = observable & safe
    outside = supported & ~physical & ~rig
    scores: dict[str, np.ndarray] = {}
    for name in ("co2_presence_probability", "gas_probability_conditional_on_co2"):
        optical = np.asarray(optical_predictions[name], dtype=np.float32)
        decided = np.asarray(physical_predictions[name], dtype=np.float32)
        if (optical.shape != observable.shape or decided.shape != observable.shape
                or not np.isfinite(optical).all() or not np.isfinite(decided).all()
                or np.any((optical < 0) | (optical > 1))
                or np.any((decided < 0) | (decided > 1))):
            raise ValueError("V5 optical/physical scores need finite native unit grids")
        scores[name] = np.where(physical, decided,
                                np.where(outside, optical, 0)).astype(np.float32)
    co2 = scores["co2_presence_probability"]
    gas = scores["gas_probability_conditional_on_co2"]
    free = co2 * gas
    report = {
        "observation_only_pixels": int(outside.sum()),
        "observation_only_co2_pixels": int(np.count_nonzero(outside & (co2 >= .5))),
        "observation_only_gas_pixels": int(np.count_nonzero(outside & (co2 >= .5) & (gas >= .6))),
    }
    if review_safe_support is not None:
        report["unresolved_photo_or_feature_pixels"] = int(
            np.count_nonzero(observable & ~supported))
        report["review_safe_support_pixels"] = int(np.count_nonzero(supported))
    if certified_rig is not None:
        report["source_excluded_rig_pixels"] = int(np.count_nonzero(observable & rig))
    return {
        **physical_predictions, **scores,
        "untouched_water_probability": 1 - co2,
        "aqueous_co2_probability": co2 - free,
        "free_co2_gas_probability": free,
    }, report


def _write_phase_maps(
    root: Path,
    run_id: str,
    fingerprint: str,
    frame_index: int,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    predictions: Mapping[str, np.ndarray],
) -> None:
    _write_mask_maps(root, run_id, fingerprint, frame_index, canvas_shape, content_box,
                     _phase_masks(predictions))


def _save_phase_png(image: Image.Image, path: Path) -> dict[str, float]:
    """Encode locally, then publish one complete PNG; keep NAS writes separate."""
    import shutil
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scratch = os.environ.get('PHASE_SCRATCH_DIR') or None
    if scratch:
        Path(scratch).mkdir(parents=True, exist_ok=True)
    temporary = None
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 ** 2, dir=scratch) as encoded:
        image.save(encoded, format='PNG', compress_level=1)
        size = encoded.tell()
        encoded.seek(0)
        encoded_at = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                shutil.copyfileobj(encoded, stream, length=1024 ** 2)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(encode_s=encoded_at - started, write_s=time.perf_counter() - encoded_at,
                bytes_written=size)


def _write_mask_maps(
    root: Path,
    run_id: str,
    fingerprint: str,
    frame_index: int,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Encode the categorical layers of one frame from its boolean masks.

    The masks are new arrays that nothing else holds, so this may run after the
    ordered loop has moved on to the next frame.
    """
    canvas_height, canvas_width = canvas_shape
    x, y, width, height = content_box
    folder = root / "native_phase_maps" / run_id / f"A_{fingerprint[:16]}"
    metrics = dict(encode_s=0.0, write_s=0.0, bytes_written=0)
    def save(image, path):
        for key, value in _save_phase_png(image, path).items():
            metrics[key] += value
    combined = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    for phase, mask in masks.items():
        if mask.shape != (height, width):
            raise ValueError(
                f"Phase-map shape {mask.shape} differs from content box {(height, width)}"
            )
        local = np.zeros((height, width, 4), dtype=np.uint8)
        local[mask] = PHASE_FILL_COLORS[phase]
        combined[y : y + height, x : x + width][mask] = PHASE_FILL_COLORS[phase]
        if phase == "unresolved":
            # Review-only support tint lives in the combined map. It is not a
            # fourth CO2 phase or a scientific output layer.
            continue
        path = folder / phase / f"frame_{frame_index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save(Image.fromarray(_place_local(local, canvas_shape, x, y), "RGBA"), path)
    folder.mkdir(parents=True, exist_ok=True)
    save(Image.fromarray(combined, "RGBA"), folder / f"frame_{frame_index:03d}.png")
    return {'_phase_output_metrics': metrics}


def _place_local(
    local: np.ndarray, canvas_shape: tuple[int, int], x: int, y: int
) -> np.ndarray:
    canvas = np.zeros((*canvas_shape, 4), dtype=np.uint8)
    height, width = local.shape[:2]
    canvas[y : y + height, x : x + width] = local
    return canvas


def _canvas_proposals(
    predictions: Mapping[str, np.ndarray],
    *,
    run_id: str,
    frame_index: int,
    elapsed_h: float,
    model_id: str,
    candidate_id: str,
    fingerprint: str,
    canvas_shape: tuple[int, int],
    content_box: tuple[int, int, int, int],
    eligible_mask: np.ndarray | None = None,
    facies_labels: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    raw = propose_uncertain_phase_regions(
        predictions,
        run_id=run_id,
        frame_index=frame_index,
        elapsed_h=elapsed_h,
        model_id=model_id,
        count=16,
        eligible_mask=eligible_mask,
        facies_labels=facies_labels,
    )
    canvas_height, canvas_width = canvas_shape
    content_x, content_y, content_width, content_height = content_box
    output: list[dict[str, Any]] = []
    for row in raw:
        local_x = float(row["x"])
        local_y = float(row["y"])
        local_width = float(row["width"])
        local_height = float(row["height"])
        canvas_x = (content_x + local_x * content_width) / canvas_width
        canvas_y = (content_y + local_y * content_height) / canvas_height
        canvas_w = local_width * content_width / canvas_width
        canvas_h = local_height * content_height / canvas_height
        output.append(
            {
                **row,
                "candidate": "A",
                "candidate_id": candidate_id,
                "params_sha256": fingerprint,
                "analysis_domain": ("registered_porous_medium_2d" if eligible_mask is not None
                                    else "registered_visible_camera_plane_2d"),
                "coordinate_space": "normalized_native_review_canvas_16x9",
                "shape": "box",
                "x": canvas_x,
                "y": canvas_y,
                "width": canvas_w,
                "height": canvas_h,
                "points": [
                    [canvas_x, canvas_y],
                    [canvas_x + canvas_w, canvas_y],
                    [canvas_x + canvas_w, canvas_y + canvas_h],
                    [canvas_x, canvas_y + canvas_h],
                ],
                "focus_box": {
                    "x": max(0.0, canvas_x - canvas_w),
                    "y": max(0.0, canvas_y - canvas_h),
                    "width": min(1.0, 3.0 * canvas_w),
                    "height": min(1.0, 3.0 * canvas_h),
                },
            }
        )
    return output


def _write_diagnostic_evidence(root: Path, run_id: str, frame_index: int, physics: Any) -> None:
    """Keep the numerical archive without invoking diagnostic display rendering.

    The scientific sequence modules are fingerprinted by frozen models, so this
    output-only policy deliberately lives outside their runtime source files.
    """
    _write_diagnostic_flags(root, run_id, frame_index, _diagnostic_flags(physics))


def _diagnostic_flags(physics: Any, *, copy: bool = False) -> dict[str, np.ndarray]:
    return {name: (values.copy() if copy else values)
            for name, values in physics.last_diagnostics.items() if values.dtype == bool}


def _write_diagnostic_flags(root: Path, run_id: str, frame_index: int,
                            flags: Mapping[str, np.ndarray]) -> None:
    output = root / run_id
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"frame_{frame_index:03d}.npz").open("xb") as stream:
        np.savez_compressed(stream, **flags)


def _background_outputs_enabled() -> bool:
    return os.environ.get(BACKGROUND_OUTPUTS_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _proposal_inputs(predictions: Mapping[str, np.ndarray], *, copy: bool) -> Mapping[str, np.ndarray]:
    """The three fields proposals read, detached from the sequence when they run later."""
    if not copy:
        return predictions
    return {name: np.array(predictions[name], copy=True) for name in _PROPOSAL_FIELDS}


def _proposal_processes() -> int:
    try:
        requested = max(0, int(os.environ.get(PROPOSAL_PROCESSES_ENV, "2")))
    except ValueError:
        requested = 2
    # The resource planner already budgets helpers beside phase/GPU work.
    # Enforce a portable hardware guard here as well; measured allocations
    # above four are allowed instead of silently truncating their worker count.
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    try:
        import psutil
    except ImportError:
        pass
    else:
        try:
            available = min(available, psutil.cpu_count(logical=True) or available)
            affinity = getattr(psutil.Process(), "cpu_affinity", None)
            if affinity is not None:
                available = min(available, len(affinity()))
        except (OSError, AttributeError, psutil.Error):
            pass
    if os.environ.get("PHASE_CPU_LIMIT") is not None:
        try:
            available = min(available, max(1, int(os.environ["PHASE_CPU_LIMIT"])))
        except ValueError:
            pass
    return min(requested, max(0, int(available) - 1))


_PROPOSAL_GEOMETRY: dict[str, Any] = {}


def _proposal_worker_start(domain: np.ndarray | None, facies: np.ndarray | None) -> None:
    """Runs once in every helper process: the run's fixed geometry and one library thread."""
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    _PROPOSAL_GEOMETRY.update(domain=domain, facies=facies)
    # A killed run must not leave its helpers behind: they hold the pipe they wait on themselves.
    import multiprocessing
    import threading
    parent = multiprocessing.parent_process()
    if parent is not None:
        def leave_with_parent() -> None:
            from multiprocessing.connection import wait
            wait([parent.sentinel])
            os._exit(1)
        threading.Thread(target=leave_with_parent, name="phase-proposal-parent-watch", daemon=True).start()


def _proposal_worker_ready() -> bool:
    return "domain" in _PROPOSAL_GEOMETRY


def _proposal_worker(fields: Mapping[str, np.ndarray], options: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _canvas_proposals(fields, eligible_mask=_PROPOSAL_GEOMETRY["domain"],
                             facies_labels=_PROPOSAL_GEOMETRY["facies"], **options)


class _ProposalPool:
    """``_canvas_proposals`` of finished frames in helper processes, results in frame order.

    Same function, same arrays, same result; only the interpreter lock of the ordered loop's
    process is no longer involved. At most ``workers`` proposal frames are in flight, which bounds memory;
    a failed task stops the run when its result is collected.
    """

    def __init__(self, workers: int, domain: np.ndarray | None, facies: np.ndarray | None):
        import multiprocessing
        from .phase_proposal_transport import ProposalTransport
        self.workers = int(workers)
        self.transport = ProposalTransport(self.workers)
        self._tasks: deque[Future] = deque()
        self.tasks = 0
        self.backpressure_s = 0.0
        self.failure: str | None = None
        self._executor: ProcessPoolExecutor | None = None
        self._ready: Future | None = None
        try:                                   # helper processes start while the run is still being set up
            self._executor = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                                 initializer=_proposal_worker_start, initargs=(domain, facies))
            self._ready = self._executor.submit(_proposal_worker_ready)
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"

    def usable(self) -> bool:
        """False when helper processes cannot start here; proposals then stay on the output thread."""
        if self._ready is not None:
            try:
                if not self._ready.result(timeout=300):
                    self.failure = "helper process started without the run geometry"
            except Exception as error:
                self.failure = f"{type(error).__name__}: {error}"
            self._ready = None
        return self.failure is None

    def submit(self, fields: Mapping[str, np.ndarray], options: Mapping[str, Any]) -> Future:
        started = time.perf_counter()
        while self._tasks and self._tasks[0].done():
            self._tasks.popleft()
        while len(self._tasks) >= self.workers:
            self._tasks.popleft().result()
        self.backpressure_s += time.perf_counter() - started
        submitted = self.transport.submit(self._executor, _proposal_worker, fields, options)
        self._tasks.append(submitted.raw)
        self.tasks += 1
        return submitted.result

    def close(self) -> None:
        """After every result has been collected."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self.transport.close()

    def abandon(self) -> None:
        """The run failed: drop what is queued and do not wait for helpers that may be stuck."""
        executor, self._executor = self._executor, None
        if executor is None:
            self.transport.close()
            return
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        import time
        deadline = time.monotonic() + 2.0
        for process in processes:
            process.join(timeout=max(0., deadline-time.monotonic()))
        self.transport.close()

    def report(self) -> dict[str, Any]:
        return {"mode": "helper-processes-v1" if self.failure is None else "output-thread",
                "processes": self.workers, "tasks": self.tasks, "helper_processes_unavailable": self.failure,
                "loop_waited_for_proposals_s": self.backpressure_s, "identical_to_inline": True,
                "transport": self.transport.report()}


def _detached_diagnostics(physics: Any, diagnostics: Mapping[str, Any] | None = None) -> Callable[..., None] | None:
    """A writer for THIS frame's review diagnostics that may run later, or None.

    Only a sequence whose writer is ``PresenceSequence.write_diagnostics`` itself, or a wrapper
    that does nothing but forward to it, is detached: that writer reads ``last_diagnostics``
    and nothing else, so a copy taken now is all it needs. Any other writer (one of them first
    refreshes inherited state) stays inside the ordered loop, exactly as before.
    """
    from types import SimpleNamespace
    from darsia.fluidflower.phase_presence_context import PresenceSequence
    target = PresenceSequence.write_diagnostics
    function = getattr(type(physics), "write_diagnostics", None)
    code = getattr(function, "__code__", None)
    forwards_only = {"phase_presence_context", "PresenceSequence", "write_diagnostics"}
    if function is not target and (code is None or set(code.co_names) != forwards_only):
        return None
    # ``diagnostics`` are one photograph's own arrays (a decision made beside the loop): no copy.
    source = physics.last_diagnostics if diagnostics is None else diagnostics
    snapshot = SimpleNamespace(last_diagnostics={
        name: (values.copy() if diagnostics is None and isinstance(values, np.ndarray) else values)
        for name, values in source.items()})
    return lambda *args: target(snapshot, *args)


class _FrameOutputWriter:
    """Output-only work of finished frames, beside the ordered loop.

    Tasks write distinct frame files. Every task receives snapshots that a
    later frame cannot touch (new boolean masks, copied arrays), so the files
    equal inline writing byte for byte. At most ``pending`` tasks may wait,
    which bounds memory; a failed task stops the run at the next call. A second
    thread decodes the next frame's JPEG while the current frame is in physics.
    """

    def __init__(self, enabled: bool, pending: int | None = None, workers: int | None = None):
        import threading
        workers = int(os.environ.get("PHASE_OUTPUT_WORKERS", "1")) if workers is None else workers
        pending = int(os.environ.get("PHASE_OUTPUT_PENDING", "8")) if pending is None else pending
        if workers < 0 or pending < 1:
            raise ValueError("Output workers must be nonnegative and pending tasks positive")
        self.enabled = bool(enabled and workers)
        self.workers = workers if self.enabled else 0
        self._executor = ThreadPoolExecutor(workers, thread_name_prefix="phase-output") if self.enabled else None
        self._input = ThreadPoolExecutor(1, thread_name_prefix="phase-input") if self.enabled else None
        self._tasks: deque[Future] = deque()
        self._pending = max(1, int(pending))
        self.tasks = 0
        self.busy_s = 0.0
        self.backpressure_s = 0.0
        self.drain_s = 0.0
        self._timing_lock = threading.Lock()
        self._task_timings: dict[str, dict[str, float]] = {}

    def _timed(self, function, *args, **kwargs):
        started = time.perf_counter()
        result = None
        try:
            result = function(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - started
            name = getattr(function, "__name__", type(function).__name__)
            with self._timing_lock:
                self.busy_s += elapsed
                record = self._task_timings.setdefault(name, {"tasks": 0, "busy_s": 0.0})
                record["tasks"] += 1
                record["busy_s"] += elapsed
                if isinstance(result, Mapping) and '_phase_output_metrics' in result:
                    for key, value in result['_phase_output_metrics'].items():
                        record[key] = record.get(key, 0.0) + value

    def submit(self, function, *args, **kwargs) -> Future:
        self.tasks += 1
        if self._executor is None:
            future: Future = Future()
            future.set_result(self._timed(function, *args, **kwargs))
            return future
        started = time.perf_counter()
        while self._tasks and self._tasks[0].done():
            self._tasks.popleft().result()
        while len(self._tasks) >= self._pending:
            self._tasks.popleft().result()
        self.backpressure_s += time.perf_counter() - started
        future = self._executor.submit(self._timed, function, *args, **kwargs)
        self._tasks.append(future)
        return future

    def prefetch(self, function, *args) -> Future | None:
        return None if self._input is None else self._input.submit(function, *args)

    def drain(self) -> None:
        started = time.perf_counter()
        try:
            while self._tasks:
                self._tasks.popleft().result()
        finally:
            self.drain_s += time.perf_counter() - started

    def close(self) -> None:
        for executor in (self._executor, self._input):
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        self._executor = self._input = None

    def report(self) -> dict[str, Any]:
        return {"mode": OUTPUT_EXECUTION_BACKGROUND if self.enabled else OUTPUT_EXECUTION_INLINE,
                "tasks": self.tasks, "busy_s": self.busy_s,
                "workers": self.workers, "max_pending": self._pending,
                "task_timings": {k: dict(v) for k, v in self._task_timings.items()},
                "loop_waited_for_outputs_s": self.backpressure_s, "final_drain_s": self.drain_s,
                "byte_identical_to_inline": True}


def materialize_phase_validation_campaign(
    model_root: Path,
    source_campaign_root: Path,
    output_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
    predictor_factory: Callable | None = None,
    inference_provenance: Mapping[str, Any] | None = None,
    execution_cache_factory: Callable | None = None,
    reference_factory: Callable | None = None,
    output_mode: str = "review",
    decision_executor_factory: Callable | None = None,
    generate_review_proposals: bool = False,
    numeric_writer_factory: Callable | None = None,
) -> dict[str, Any]:
    """Build frozen phase maps; production omits optional Studio presentation.

    Whatever happens inside, the decision threads and proposal helper processes that a run
    started are released before this returns or raises.
    """
    started: list[Any] = []
    try:
        return _materialize_phase_validation_campaign(
            model_root, source_campaign_root, output_root, progress=progress,
            predictor_factory=predictor_factory, inference_provenance=inference_provenance,
            execution_cache_factory=execution_cache_factory, reference_factory=reference_factory,
            output_mode=output_mode, decision_executor_factory=decision_executor_factory,
            generate_review_proposals=generate_review_proposals,
            numeric_writer_factory=numeric_writer_factory, _started=started)
    finally:
        for resource in reversed(started):
            try:
                (getattr(resource, "abandon", None) or resource.close)()
            except Exception:                   # releasing must never hide the run's own error
                pass


def _materialize_phase_validation_campaign(
    model_root: Path,
    source_campaign_root: Path,
    output_root: Path,
    *,
    _started: list[Any] | None = None,
    progress: Callable[[str], None] | None = None,
    predictor_factory: Callable | None = None,
    inference_provenance: Mapping[str, Any] | None = None,
    execution_cache_factory: Callable | None = None,
    reference_factory: Callable | None = None,
    output_mode: str = "review",
    decision_executor_factory: Callable | None = None,
    generate_review_proposals: bool = False,
    numeric_writer_factory: Callable | None = None,
) -> dict[str, Any]:
    """Build frozen phase maps; production omits optional Studio presentation.

    Both modes retain final maps, physical reports and spatial evidence arrays.
    Review retains raw optical layers and diagnostic PNGs. Proposals are generated
    only when explicitly requested; ``proposals`` mode replays inference without
    rewriting phase maps or presentation assets for the on-demand quiz.
    """

    if output_mode not in ("review", "production", "proposals", "numerical"):
        raise ValueError("Phase output mode must be review, production, numerical or proposals")
    review_outputs = output_mode == "review"
    map_outputs = output_mode in ("review", "production")
    numeric_outputs = output_mode == "numerical" or numeric_writer_factory is not None
    if numeric_outputs and output_mode == "proposals":
        raise ValueError("Quiz replay cannot publish numeric production data")
    proposal_outputs = output_mode == "proposals" or (review_outputs and generate_review_proposals)
    proposals_status = "generated" if proposal_outputs else ("deferred" if review_outputs else "not-requested")
    model_root = Path(model_root).expanduser().resolve()
    source_root = Path(source_campaign_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    model_id, model_sha256 = _model_identity(model_root)
    physics_profile = load_physics_profile(model_root)
    physics_runtime = physics_runtime_identity(physics_profile)
    pipeline_sha256 = model_sha256 if physics_runtime is None else hashlib.sha256(
        f"{model_sha256}:{physics_runtime}".encode("ascii")).hexdigest()
    model = HierarchicalPhaseModel.load(model_root, adapter="generic")
    from darsia.fluidflower.phase_training.radiometry import model_profile, RunRadiometry, runtime_identity
    from darsia.fluidflower.phase_training.v5_linear_input import (
        RunV5LinearInput, model_v5_settings, runtime_identity as v5_runtime_identity,
    )
    profile = model_profile(model)
    v5_settings = model_v5_settings(model_root)
    if v5_settings is not None and output_mode != "review":
        raise ValueError("V5 moving-geometry maps are for manual Studio review only")
    v5_registered_runtime = None
    if v5_settings is not None:
        from . import phase_runtime as frame_physics_runtime
        from .tools import (
            ffac_v5_frame_context, ffac_v5_temporal_alignment,
            ffac_v5_track1_photo_context, ffac_v5_extended_reference,
            phase_v5_registered_frame,
            phase_frame_workers,
        )
        from .phase_training import v5_linear_input
        modules = (frame_physics_runtime, ffac_v5_frame_context,
                   ffac_v5_track1_photo_context,
                   ffac_v5_temporal_alignment, ffac_v5_extended_reference,
                   phase_v5_registered_frame, phase_frame_workers,
                   v5_linear_input)
        v5_registered_runtime = hashlib.sha256((sha256_file(Path(__file__)) + "".join(
            sha256_file(Path(module.__file__)) for module in modules)).encode()).hexdigest()
        pipeline_sha256 = hashlib.sha256(
            f"{pipeline_sha256}:{v5_registered_runtime}".encode("ascii")).hexdigest()
    if v5_settings is not None and profile is not None:
        raise ValueError("V5 float input cannot be combined with JPEG radiometry")
    if v5_settings is not None and reference_factory is not None:
        raise ValueError("V5 single-pass references require source-bound float endpoints")
    radiometry_runtime = (v5_runtime_identity() if v5_settings is not None else
                          None if profile is None else runtime_identity())
    source_manifest_sha256 = sha256_file(source_root / "postflight_manifest.json")
    binding_path = destination / "validation_binding.json"
    source_jobs = _source_jobs(source_root)
    from .phase_training.scope import validation_jobs
    source_jobs = validation_jobs(model_root, source_jobs)
    if predictor_factory is not None and not inference_provenance:
        raise ValueError("An external predictor requires explicit inference provenance")
    # Validate every physical input even on a cache hit. The profile records the
    # registered grid and protocol, not just a mutable path to a rig directory.
    if physics_profile is not None:
        for job in source_jobs:
            run = str(job["run"]).casefold()
            dataset = str(job.get("dataset") or ("ffac" if run.startswith("ac") else "ffum")).casefold()
            entry = physics_profile["runs"].get(f"{dataset}/{run}")
            if not entry or sha256_file(model_root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Missing or changed phase-physics context for {dataset}/{run}")
            from darsia.fluidflower.phase_training.radiometry import analysis_review_root
            _, root = _candidate_paths(source_root, job)
            root = analysis_review_root(Path(str(job.get("analysis_source_review_root") or root)), run)
            if sha256_file(root / "native_frames" / run / "manifest.json") != entry["native_manifest_sha256"]:
                raise ValueError(f"{dataset}/{run}: phase-physics context is from a different source revision")
    if binding_path.is_file():
        cached = _read_json(binding_path)
        if (
            cached.get("model_sha256") == model_sha256
            and cached.get("source_manifest_sha256") == source_manifest_sha256
            and (destination / "postflight_manifest.json").is_file()
            and cached.get("radiometry_runtime") == radiometry_runtime
            and cached.get("v5_registered_runtime") == v5_registered_runtime
            and cached.get("physics_runtime") == physics_runtime
            and cached.get("inference_provenance") == inference_provenance
            and cached.get("output_mode", "review") == output_mode
            and bool(cached.get("numeric_outputs")) == numeric_outputs
            and (not proposal_outputs or cached.get("proposals_status", "generated") == "generated")
        ):
            if v5_settings is not None:
                # External per-frame H receipts and their AC28 context are not
                # covered by the campaign manifest SHA.  A cached phase-map
                # folder is reusable only after the whole native geometry
                # chain is checked live against its immutable source revision.
                from .phase_training.radiometry import analysis_review_root
                from .tools.ffac_v5_frame_context import V5FrameGeometryCohort
                from .tools.ffac_v5_track1_photo_context import Track1FramePhotoCohort
                for job in source_jobs:
                    run = str(job["run"]).casefold()
                    _, root = _candidate_paths(source_root, job)
                    root = analysis_review_root(
                        Path(str(job.get("analysis_source_review_root") or root)), run)
                    native_path = root / "native_frames" / run / "manifest.json"
                    cohort_type = (Track1FramePhotoCohort
                                   if _read_json(native_path).get("v5_track1_photo_geometry")
                                   else V5FrameGeometryCohort)
                    cohort = cohort_type(native_path)
                    bindings = [cohort.frame(index)
                                for index in range(len(cohort.rows))]
                    _v5_physical_scale_warning(physics_profile, bindings)
                    folder = (destination / "review_assets" / "native_phase_maps" /
                              run / f"A_{pipeline_sha256[:16]}")
                    expected = [f"frame_{index:03d}.png"
                                for index in range(len(cohort.rows))]
                    for layer in (folder, folder / "aqueous", folder / "gas",
                                  folder / "mixed"):
                        if sorted(path.name for path in layer.glob("frame_*.png")) != expected:
                            raise ValueError(f"{run}: cached V5 native phase maps are incomplete")
            if numeric_writer_factory is not None:
                raise ValueError("Numeric consumer already materialized; replay its verified numeric archive instead")
            if numeric_outputs:
                from .phase_numeric_fields import load_numeric_phase_run
                for job in _read_json(destination / "postflight_manifest.json")["jobs"]:
                    for _ in load_numeric_phase_run(Path(job["numeric_phase_output_dir"])):
                        pass
            return cached
        raise ValueError("This validation folder belongs to another pipeline revision; use a new output folder")

    jobs: list[dict[str, Any]] = []
    selected_candidates: dict[str, str] = {}
    for source_job in source_jobs:
        run_id = str(source_job["run"]).casefold()
        dataset = str(source_job.get("dataset") or "").casefold()
        if not dataset:
            dataset = "ffac" if run_id.startswith("ac") else "ffum"
        candidate_root, review_root = _candidate_paths(source_root, source_job)
        review_root = Path(str(source_job.get("analysis_source_review_root") or review_root))
        from darsia.fluidflower.phase_training.radiometry import analysis_review_root
        review_root = analysis_review_root(review_root, run_id)
        native_manifest = _read_json(review_root / "native_frames" / run_id / "manifest.json")
        frame_count = int(native_manifest.get("frame_count") or 0)
        frame_geometry = None
        track1_photo_only = False
        temporal_reference_contract = None
        v5_scale_warning = None
        if v5_settings is not None:
            from .tools.ffac_v5_frame_context import (
                COHORT_CONTRACT, V5FrameGeometryCohort,
            )
            from .tools.ffac_v5_track1_photo_context import (
                COHORT_CONTRACT as TRACK1_COHORT_CONTRACT,
                Track1FramePhotoCohort,
            )
            if (native_manifest.get("v5_track1_photo_geometry") and
                    native_manifest.get("v5_frame_geometry")):
                raise ValueError(f"{run_id}: V5 source mixes photo-only and physical geometry")
            track1_photo_only = bool(native_manifest.get("v5_track1_photo_geometry"))
            marker = (native_manifest.get("v5_track1_photo_geometry") if track1_photo_only
                      else native_manifest.get("v5_frame_geometry")) or {}
            expected_contract = (TRACK1_COHORT_CONTRACT if track1_photo_only else COHORT_CONTRACT)
            if marker.get("contract") != expected_contract:
                raise ValueError(f"{run_id}: V5 source lacks certified per-frame photo geometry")
            cohort_type = Track1FramePhotoCohort if track1_photo_only else V5FrameGeometryCohort
            frame_geometry = cohort_type(review_root / "native_frames" / run_id / "manifest.json")
            temporal_reference_contract = (native_manifest.get("v5_temporal_reference") or {}).get(
                "contract")
            if temporal_reference_contract not in {
                    "ffac-v5-native-crop-source-supported-v1",
                    "ffac-v5-source-backed-extended-ring-v1"}:
                raise ValueError(f"{run_id}: V5 temporal source support contract is missing")
            if len(frame_geometry.rows) != frame_count:
                raise ValueError(f"{run_id}: V5 frame geometry timeline is incomplete")
            v5_scale_warning = _v5_physical_scale_warning(
                physics_profile, [frame_geometry.frame(index)
                                  for index in range(frame_count)])
            if numeric_outputs or proposal_outputs or reference_factory is not None:
                raise ValueError("V5 moving geometry currently supports review maps only")
        canvas_width, canvas_height = (
            int(value) for value in native_manifest.get("canvas_wh") or []
        )
        content_x, content_y, content_width, content_height = (
            int(value) for value in native_manifest.get("content_box_xywh") or []
        )
        physics = None
        context = None
        domain = None
        facies = None
        domain_record = None
        physics_report: list[dict[str, Any]] = []
        frame_timings: list[dict[str, Any]] = []
        if physics_profile is not None:
            context, context_entry = load_run_context(
                model_root, physics_profile, dataset, run_id,
                native_manifest_sha256=sha256_file(review_root / "native_frames" / run_id / "manifest.json"))
            if context.domain.shape != (content_height, content_width):
                raise ValueError(f"{run_id}: physics context does not match the corrected image grid")
            physics = phase_physics_sequence(context, physics_profile)
            if execution_cache_factory is not None:
                execution_cache_factory(context)
            domain, facies = context.domain, context.facies
            if frame_geometry is not None:
                provenance_key = ("v5_track1_photo_geometry" if track1_photo_only
                                  else "v5_frame_geometry")
                provenance = (context_entry.get("provenance") or {}).get(provenance_key) or {}
                if (provenance.get("contract") != expected_contract
                        or provenance.get("native_manifest_sha256") !=
                        frame_geometry.native_manifest_sha256
                        or provenance.get("original_source_manifest_sha256") !=
                        frame_geometry.source_manifest_sha256
                        or provenance.get("frame_receipt_count") != frame_count
                        or (track1_photo_only and
                            (provenance.get("validation_only") is not True or
                             provenance.get("mass_eligible") is not False))):
                    raise ValueError(f"{run_id}: V5 context lacks exact frame photo binding")
        elif native_manifest.get("domain", {}).get("path"):
            domain_path = _resolve(review_root, native_manifest["domain"]["path"])
            with np.load(domain_path, allow_pickle=False) as archive:
                domain = np.asarray(archive["domain"], dtype=bool)
                facies = np.asarray(archive["facies"])
            if domain.shape != (content_height, content_width) or facies.shape != domain.shape:
                raise ValueError(f"{run_id}: domain does not match the corrected image grid")
            domain_record = {"path": str(domain_path), "sha256": sha256_file(domain_path)}
        if numeric_outputs and domain is None:
            raise ValueError(f"{run_id}: numeric production requires an explicit registered porous domain")
        if frame_geometry is not None and physics is None:
            raise ValueError(f"{run_id}: V5 frame geometry requires source-bound phase physics")
        observation_domain = domain
        physical_sand_domain = domain
        reference_path = review_root / "native_references" / run_id / "pre_injection.jpg"
        if (physics_profile or {}).get("photo_sand_top_revision") is not None:
            from .phase_v5_source_alias import resolve_native_photo_path
            reference_path = resolve_native_photo_path(review_root, run_id, reference=True)
        baseline_canvas = _load_rgb(reference_path)
        baseline = baseline_canvas[
            content_y : content_y + content_height,
            content_x : content_x + content_width,
        ]
        radiometry = None
        v5_input = None
        display_root = review_root
        if v5_settings is not None:
            if (physics_profile or {}).get("photo_sand_top_revision") is not None:
                from .phase_v5_source_alias import RunV5AliasedLinearInput
                RunV5LinearInput = RunV5AliasedLinearInput
            v5_input = RunV5LinearInput(
                native_manifest_path=review_root / "native_frames" / run_id / "manifest.json",
                run_id=run_id, content_shape_hw=(content_height, content_width),
                unit_denominator=v5_settings["unit_denominator"],
                required_valid_mask=None if frame_geometry is not None else domain,
            )
            baseline = v5_input.load(reference_path, "pre_injection")
            if frame_geometry is not None:
                if temporal_reference_contract == "ffac-v5-source-backed-extended-ring-v1":
                    registered_baseline, baseline_origin, baseline_valid = (
                        v5_input.load_extended(reference_path, "pre_injection"))
                else:
                    registered_baseline, baseline_origin, baseline_valid = (
                        baseline, (0, 0), None)
        elif profile is not None:
            radiometry = RunRadiometry(
                profile, series_id=dataset, run_id=run_id, baseline_path=reference_path,
                content_box=(content_x, content_y, content_width, content_height),
                cache_root=destination / "radiometry_cache" / dataset / run_id,
                source_manifest=review_root / "native_frames" / run_id / "manifest.json",
            )
            baseline = radiometry.load(reference_path, "pre_injection")
            if map_outputs:
                display_root = destination / "review_assets"
                corrected_canvas = baseline_canvas.copy()
                corrected_canvas[content_y:content_y + content_height, content_x:content_x + content_width] = np.rint(baseline * 255).astype(np.uint8)
                for folder in ("native_references", "reference_frames"):
                    path = display_root / folder / run_id / "pre_injection.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(corrected_canvas).save(path)
        candidate_id = f"{model_id}__{dataset}__{run_id}"
        if physics is not None:
            candidate_id += f"__physics_{physics_runtime[:12]}"
        all_proposals: list[dict[str, Any]] = []
        # Failure cleanup joins output/input workers before a caller can retry.
        writer = _FrameOutputWriter(_background_outputs_enabled())
        if _started is not None:
            _started.append(writer)
        proposal_tasks: list[Future] = []
        proposal_pool = (_ProposalPool(_proposal_processes(), domain, facies)
                         if proposal_outputs and writer.enabled and _proposal_processes() > 0 else None)
        if proposal_pool is not None and _started is not None:
            _started.append(proposal_pool)

        def submit_proposals(fields: Mapping[str, np.ndarray], index: int, hours: float) -> Future:
            options = dict(run_id=run_id, frame_index=index, elapsed_h=hours, model_id=model_id,
                           candidate_id=candidate_id, fingerprint=pipeline_sha256,
                           canvas_shape=(canvas_height, canvas_width),
                           content_box=(content_x, content_y, content_width, content_height))
            if proposal_pool is not None and proposal_pool.usable():
                return proposal_pool.submit({name: fields[name] for name in _PROPOSAL_FIELDS}, options)
            return writer.submit(_canvas_proposals, fields, eligible_mask=domain, facies_labels=facies, **options)
        prefetched: tuple[Path, Future] | None = None
        previous: np.ndarray | None = None
        previous_extended: np.ndarray | None = None
        previous_origin = (0, 0)
        previous_valid: np.ndarray | None = None
        frame_sources = [
            dict(row)
            for row in native_manifest.get("frame_sources") or []
            if isinstance(row, Mapping)
        ]
        numeric_writer = None
        if numeric_outputs:
            from .phase_numeric_fields import NumericPhaseRunWriter, default_btb_acidification_coordinate
            factory = numeric_writer_factory or NumericPhaseRunWriter
            numeric_writer = factory(
                root=destination / "numeric_phase" / dataset / run_id,
                run_id=run_id, series_id=dataset, frame_count=frame_count,
                grid={"shape_rc": [content_height, content_width],
                      "content_box_xywh": [content_x, content_y, content_width, content_height],
                      **({"x_m": context.x_m.tolist(), "y_m": context.y_m.tolist()}
                         if context is not None else {})},
                domain=domain,
                baseline_rgb=baseline,
                identity={"model_id": model_id, "model_sha256": model_sha256,
                          "pipeline_sha256": pipeline_sha256, "physics_runtime": physics_runtime,
                          "native_manifest_sha256": sha256_file(review_root / "native_frames" / run_id / "manifest.json"),
                          "context": context_entry if context is not None else domain_record,
                          "radiometry_profile": profile, "radiometry_runtime": radiometry_runtime},
                acidification_coordinate=default_btb_acidification_coordinate)
        numeric_current: dict[int, np.ndarray] = {}

        def finished_fields(fields: Mapping[str, np.ndarray], index: int, hours: float,
                            current_rgb: np.ndarray | None, report: Mapping[str, Any]) -> dict[str, np.ndarray]:
            masks = _phase_masks(fields)
            if (physics_profile or {}).get("approved_g3_gas") is not None:
                # The approved gas partition owns the exact inherited water,
                # CO2 and unresolved categories, including unsupported pixels.
                masks = physics.exact_phase_masks()
            elif (frame_geometry is not None and
                    (physics_profile or {}).get("v5_nominal_review_gas_minimum") is not None):
                masks["unresolved"] = (native_observable &
                    ~(photo_sand & observation_domain))
            counts = {name: int(np.count_nonzero(mask if observation_domain is None else
                                                mask & observation_domain))
                      for name, mask in masks.items()
                      if name in ("aqueous", "mixed", "gas")}
            physics_report.append({"frame_index": index, **report, "phase_area_counts": counts,
                                   "porous_domain_pixels": (None if domain is None else
                                                            int(np.count_nonzero(domain))),
                                   **({"observation_domain_pixels": int(np.count_nonzero(observation_domain)),
                                       "native_content_pixels": int(np.count_nonzero(native_observable)),
                                       "feature_source_support_receipt_sha256": feature_receipt["receipt_sha256"],
                                       **({"photo_safe_sand_pixels": int(np.count_nonzero(photo_sand)),
                                           "photo_only_context": True} if track1_photo_only else
                                          {"physical_sand_pixels": int(np.count_nonzero(physical_sand_domain)),
                                           "physics_supported_pixels": int(np.count_nonzero(domain))})}
                                      if frame_geometry is not None else {})})
            if numeric_writer is not None:
                source = frame_sources[index] if index < len(frame_sources) else {}
                if source.get("elapsed_h") is None:
                    raise ValueError("Numeric production requires explicit acquisition times")
                numeric_writer.add_frame(frame_index=index,
                    frame_id=str(source.get("image_id") or source.get("frame_id") or f"frame_{index:03d}"),
                    time_hours=hours, predictions=fields, current_rgb=current_rgb,
                    source_reference=source)
            return masks
        independent_controls = source_job.get("independent_reference_frames") or []
        if independent_controls and (
            independent_controls != [0] or not frame_sources
            or float(frame_sources[0].get("elapsed_h", 0)) >= 0
            or native_manifest.get("timeline_revision", {}).get("kind") != "prepend-evaluated-baseline-v1"
        ):
            raise ValueError("Independent reference must be an explicitly prepended pre-injection control")
        cached_top_revision = (physics_profile or {}).get("photo_sand_top_revision") is not None
        predictor = None if predictor_factory is None or cached_top_revision else predictor_factory(model, baseline)
        if predictor is not None and hasattr(predictor, "close") and _started is not None:
            _started.append(predictor)
        if (frame_geometry is not None and predictor is not None
                and not getattr(predictor, "supports_registered_v5", False)):
            raise ValueError("V5 moving geometry requires registered optical frame inference")
        if predictor is not None and hasattr(predictor, "prepare_frames"):
            prepare_options = dict(review_root=review_root, run_id=run_id,
                frame_count=frame_count,
                content_box=(content_x, content_y, content_width, content_height),
                independent_controls=independent_controls,
                radiometry=v5_input if v5_input is not None else radiometry,
                radiometry_profile=profile)
            if frame_geometry is not None:
                prepare_options["frame_geometry"] = frame_geometry
            predictor.prepare_frames(**prepare_options)
        if physics is not None and hasattr(physics, "bind_reference"):
            physics.bind_reference(baseline)
        # Optional executor (scripts/phase_decision_pipeline.py): completion + final decision of a
        # photograph run beside the chronological part of the next one. Identical values; the
        # frozen call is kept whenever the executor does not recognise the sequence.
        decision_executor = None
        if physics is not None and decision_executor_factory is not None and frame_geometry is None:
            # The factory verifies its supported sequence and returns None for
            # stateful/unknown runtimes. Frame-local regional sequences have no
            # diagnostic writer, but can still use ordered asynchronous delivery.
            decision_executor = decision_executor_factory(physics, predictor)
        pipelined = decision_executor is not None
        if pipelined and _started is not None:
            _started.append(decision_executor)

        def deliver_decided(decided: Any) -> None:
            """Reports and maps of a photograph whose final decision ran beside the loop."""
            index, hours, frame_timing = decided.token
            delivery_started = time.perf_counter()
            geometry = ((canvas_height, canvas_width), (content_x, content_y, content_width, content_height))
            masks = finished_fields(decided.output, index, hours, numeric_current.pop(index, None), decided.report)
            if review_outputs:
                detached = _detached_diagnostics(physics, decided.diagnostics)
                if detached is not None:
                    writer.submit(detached, destination / "presence_diagnostics", run_id, index, *geometry)
            elif map_outputs and hasattr(physics, "write_diagnostics"):
                writer.submit(_write_diagnostic_flags, destination / "presence_diagnostics", run_id, index,
                              {name: values for name, values in decided.diagnostics.items()
                               if values.dtype == bool})
            if map_outputs:
                writer.submit(_write_mask_maps, destination / "review_assets", run_id, pipeline_sha256, index,
                              *geometry, masks)
            if proposal_outputs:
                proposal_tasks.append(submit_proposals(decided.output, index, hours))
            frame_timing.update(decided.seconds)
            frame_timing["writing_s"] = time.perf_counter() - delivery_started
        reference_observer = None if reference_factory is None else reference_factory(
            run_id=run_id, dataset=dataset, review_root=review_root,
            native_manifest=native_manifest, destination=destination, predictor=predictor,
            model=model, baseline=baseline, canvas_shape=(canvas_height, canvas_width),
            content_box=(content_x, content_y, content_width, content_height),
            pipeline_sha256=pipeline_sha256, model_sha256=model_sha256,
        )
        previous_index = None
        if reference_observer is not None and hasattr(reference_observer, "initial_reference_time"):
            # Baseline references always start from empty history, even when
            # their acquisition time is not recorded in the reviewed timeline.
            with predictor.cpu_slot() if hasattr(predictor, "cpu_slot") else nullcontext():
                reference_observer.through(reference_observer.initial_reference_time,
                    previous=None, previous_index=None, physics=physics)
        v5_cpu_predictor = None
        for frame_index in range(frame_count):
            frame_started = time.perf_counter()
            timing: dict[str, Any] = {"frame_index": frame_index, "started_unix": time.time()}
            frame_binding = None
            frame_context = None
            if frame_geometry is not None:
                geometry_started = time.perf_counter()
                observation_domain, feature_receipt = frame_geometry.feature_support(
                    frame_index, previous_index, v5_input)
                observation_domain = np.asarray(observation_domain, dtype=bool)
                if track1_photo_only:
                    photo_sand, photo_uncertain, frame_binding = frame_geometry.photo(frame_index)
                    native_observable = np.ones(photo_sand.shape, dtype=bool)
                    physical_sand_domain = photo_sand
                    physics_supported = photo_sand
                    facies_safe = photo_sand
                else:
                    from .tools.ffac_v5_frame_context import PHOTO_FIELDS, RASTER_FIELDS

                    frame_arrays, frame_binding = frame_geometry.context(
                        frame_index, fields=(*RASTER_FIELDS, *PHOTO_FIELDS))
                    native_observable = np.asarray(frame_arrays["observable_phase_domain"], dtype=bool)
                    physical_sand_domain = np.asarray(frame_arrays["physical_sand_domain"], dtype=bool)
                    physics_supported = np.asarray(frame_arrays["physics_supported_domain"], dtype=bool)
                    photo_sand = np.asarray(frame_arrays["photo_certified_sand_domain"], dtype=bool)
                    photo_uncertain = np.asarray(frame_arrays["photo_boundary_uncertain_domain"], dtype=bool)
                    facies_safe = np.asarray(frame_arrays["facies_safe_interior"], dtype=bool)
                if (observation_domain.shape != (content_height, content_width)
                        or native_observable.shape != observation_domain.shape
                        or physical_sand_domain.shape != observation_domain.shape
                        or physics_supported.shape != observation_domain.shape
                        or photo_sand.shape != observation_domain.shape
                        or photo_uncertain.shape != observation_domain.shape
                        or facies_safe.shape != observation_domain.shape
                        or not observation_domain.any() or not physics_supported.any()
                        or np.any(observation_domain & ~native_observable)
                        or np.any(physical_sand_domain & ~native_observable)
                        or np.any(physics_supported & ~physical_sand_domain)
                        or np.any(facies_safe & ~physical_sand_domain)
                        or np.any(photo_sand & photo_uncertain)):
                    raise ValueError(f"{run_id} frame {frame_index}: V5 mask hierarchy is invalid")
                # Physical context is never allowed to decide pixels that lack
                # source-supported current/B/previous optical features, nor
                # pixels excluded by the same photo/facies evidence used for
                # regional fitting. Other visible pixels remain optical-only
                # pending separate overflow/rig source QA.
                effective_physics = (physics_supported & facies_safe & photo_sand
                                     & observation_domain)
                if not effective_physics.any():
                    raise ValueError(f"{run_id} frame {frame_index}: V5 has no supported regional pixels")
                if track1_photo_only:
                    # This is an optical review mask, never a physical facies map.
                    frame_context = replace(
                        context, domain=effective_physics,
                        facies=np.ones(photo_sand.shape, dtype=np.uint8),
                        permeability=None, x_m_native=None, y_m_native=None,
                        pixel_area_m2=None)
                else:
                    frame_context = replace(
                        context, domain=effective_physics, facies=frame_arrays["facies"],
                        permeability=frame_arrays["permeability"],
                        x_m_native=frame_arrays["x_m_native"],
                        y_m_native=frame_arrays["y_m_native"],
                        pixel_area_m2=frame_arrays["pixel_area_m2"])
                domain, facies = frame_context.domain, frame_context.facies
                timing["frame_geometry_receipt_sha256"] = frame_binding["receipt_sha256"]
                timing["feature_source_support"] = feature_receipt
                timing["frame_geometry_s"] = time.perf_counter() - geometry_started
            if pipelined:
                for decided in decision_executor.ready():
                    deliver_decided(decided)
            reference_started = time.perf_counter()
            if progress is not None:
                progress(f"{run_id.upper()} frame {frame_index + 1}/{frame_count} · optical")
            if reference_observer is not None:
                if frame_index >= len(frame_sources) or frame_sources[frame_index].get("elapsed_h") is None:
                    raise ValueError("Single-pass references require explicit reviewed times")
                with predictor.cpu_slot() if hasattr(predictor, "cpu_slot") else nullcontext():
                    reference_observer.through(float(frame_sources[frame_index]["elapsed_h"]),
                        previous=previous, previous_index=previous_index, physics=physics)
            timing["reference_s"] = time.perf_counter() - reference_started
            frame_path = review_root / "native_frames" / run_id / f"frame_{frame_index:03d}.jpg"
            if (physics_profile or {}).get("photo_sand_top_revision") is not None:
                frame_path = resolve_native_photo_path(review_root, run_id, frame_index)
            raw_started = time.perf_counter()
            prepared = (predictor.prepared_frame(frame_index)
                        if predictor is not None and hasattr(predictor, "prepared_frame") else None)
            frame_canvas = None
            if (prepared is None and v5_input is None) or radiometry is not None:
                frame_canvas = (prefetched[1].result() if prefetched is not None and prefetched[0] == frame_path
                                else _load_rgb(frame_path))
            prefetched = None
            current = (
                v5_input.load(frame_path, f"frame_{frame_index:03d}",
                              required_valid_mask=observation_domain)
                if frame_geometry is not None and prepared is None else
                _select_native_current(
                    prepared, v5_input, frame_path, frame_index, frame_canvas,
                    (content_x, content_y, content_width, content_height)))
            timing["raw_load_s"] = time.perf_counter() - raw_started
            correction_started = time.perf_counter()
            if radiometry is not None:
                # ParallelFramePredictor has already applied this exact
                # radiometry profile while materializing its .npy inputs.
                # Reuse that array; otherwise a serial/GPU predictor still
                # follows the original direct correction path.
                current = (
                    prepared
                    if prepared is not None
                    else radiometry.load(frame_path, f"frame_{frame_index:03d}")
                )
                if map_outputs:
                    corrected_canvas = frame_canvas.copy()
                    corrected_canvas[content_y:content_y + content_height, content_x:content_x + content_width] = np.rint(current * 255).astype(np.uint8)
                    path = display_root / "native_frames" / run_id / f"frame_{frame_index:03d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(corrected_canvas).save(path)
            timing["radiometry_s"] = time.perf_counter() - correction_started
            optical_started = time.perf_counter()
            aligned_baseline = baseline
            if cached_top_revision:
                # The source-bound approved parent map supplies the entire
                # phase decision. Do not run heads or refilter gas components.
                predictions = {name: np.zeros(current.shape[:2], np.float32) for name in (
                    "co2_presence_probability", "gas_probability_conditional_on_co2",
                    "untouched_water_probability", "aqueous_co2_probability", "free_co2_gas_probability")}
                timing["optical_reused_approved_parent_map"] = True
            elif frame_geometry is not None:
                from .tools.ffac_v5_temporal_alignment import align_temporal_inputs
                from .tools.phase_v5_registered_frame import predict_registered_v5_frame
                h_current = frame_geometry.H_v5_B_to_frame(frame_index)
                h_previous = (None if previous_index is None else
                              frame_geometry.H_v5_B_to_frame(previous_index))
                if predictor is None:
                    predictions, aligned_baseline, alignment, v5_cpu_predictor = (
                        predict_registered_v5_frame(
                            model, baseline=registered_baseline, current=current,
                            previous=previous_extended,
                            h_baseline_to_current=h_current,
                            h_baseline_to_previous=h_previous,
                            required_valid_mask=observation_domain,
                            baseline_origin_xy=baseline_origin,
                            previous_origin_xy=previous_origin,
                            baseline_source_valid_mask=baseline_valid,
                            previous_source_valid_mask=previous_valid,
                            predictor=v5_cpu_predictor))
                else:
                    aligned_baseline, _, alignment = align_temporal_inputs(
                        baseline=registered_baseline, current=current,
                        previous=previous_extended,
                        h_baseline_to_current=h_current,
                        h_baseline_to_previous=h_previous,
                        required_valid_mask=observation_domain,
                        baseline_origin_xy=baseline_origin,
                        previous_origin_xy=previous_origin,
                        baseline_source_valid_mask=baseline_valid,
                        previous_source_valid_mask=previous_valid)
                    predictions = predictor(current, previous)
                    worker_alignment = (getattr(predictor, "last_receipt", None) or {}).get("v5_alignment") or {}
                    if (worker_alignment.get("frame_geometry_receipt_sha256") !=
                            frame_binding["receipt_sha256"] or
                            worker_alignment.get("feature_support_receipt_sha256") !=
                            feature_receipt["receipt_sha256"] or
                            worker_alignment.get("feature_support_mask_sha256") !=
                            feature_receipt["feature_support_mask_sha256"] or
                            worker_alignment.get("aligned_baseline_sha256") !=
                            hashlib.sha256(aligned_baseline.tobytes()).hexdigest()):
                        raise ValueError(f"{run_id} frame {frame_index}: worker V5 reference differs from physics")
                timing["v5_alignment"] = alignment
            else:
                predictions = predictor(current, previous) if predictor is not None else predict_image_tiled(
                    model, current, baseline, previous_rgb=previous,
                    radiometry_profile=profile)
            if frame_geometry is not None:
                predictions = {name: np.where(observation_domain, values, 0).astype(np.float32)
                               for name, values in predictions.items()}
            timing["optical_s"] = time.perf_counter() - optical_started
            # Parallel frame workers provide the internal optical breakdown;
            # retain it beside the end-to-end frame timing so a report answers
            # both "how long did this frame take?" and "where did it spend it?".
            worker_receipt = getattr(predictor, "last_receipt", None)
            if isinstance(worker_receipt, Mapping):
                worker_timing = worker_receipt.get("timing")
                if isinstance(worker_timing, Mapping):
                    timing["optical_worker"] = dict(worker_timing)
                timing["optical_worker_total_s"] = worker_receipt.get("seconds")
                consumer = worker_receipt.get("consumer")
                if isinstance(consumer, Mapping):
                    # optical_s above = waiting for the shared worker + loading its heads.
                    timing["optical_wait_s"] = consumer.get("wait_s")
                    timing["optical_heads_load_s"] = consumer.get("heads_load_s")
                timing["optical_worker_clock"] = {key: worker_receipt.get(key) for key in (
                    "backend", "reused", "pid", "submitted_unix", "dispatched_unix",
                    "started_unix", "finished_unix")}
            if frame_index + 1 < frame_count and prepared is None and v5_input is None:
                next_path = review_root / "native_frames" / run_id / f"frame_{frame_index + 1:03d}.jpg"
                next_canvas = writer.prefetch(_load_rgb, next_path)
                prefetched = None if next_canvas is None else (next_path, next_canvas)
            if pipelined:
                # Outside the CPU slot: an unfinished decision may need a slot of its own.
                capacity_started = time.perf_counter()
                decision_executor.wait_for_capacity()
                timing["decision_capacity_wait_s"] = time.perf_counter() - capacity_started
                for decided in decision_executor.ready():
                    deliver_decided(decided)
            physics_started = time.perf_counter()
            optical_for_observation = (
                {name: np.asarray(predictions[name], dtype=np.float32).copy()
                 for name in ("co2_presence_probability", "gas_probability_conditional_on_co2")}
                if frame_geometry is not None else None)
            with predictor.cpu_slot() if hasattr(predictor, "cpu_slot") else nullcontext():
                if progress is not None:
                    progress(f"{run_id.upper()} frame {frame_index + 1}/{frame_count} · physics/reference")
                if physics is None and domain is not None and frame_geometry is None:
                    predictions = {name: np.where(domain, values, 0).astype(np.float32)
                                   for name, values in predictions.items()}
                elapsed_h = (
                    float(frame_sources[frame_index]["elapsed_h"])
                    if frame_index < len(frame_sources) and frame_sources[frame_index].get("elapsed_h") is not None
                    else None
                )
                if physics is not None:
                    if elapsed_h is None:
                        raise ValueError(f"{run_id} frame {frame_index}: physical rules require an explicit elapsed time")
                    if (physics_profile or {}).get("approved_g3_gas") is not None:
                        physics.bind_cached_frame(run=run_id, frame_index=frame_index,
                            elapsed_h=elapsed_h, native_manifest_sha256=sha256_file(
                                review_root / "native_frames" / run_id / "manifest.json"))
                    if frame_geometry is not None:
                        bind_native_frame_context(physics, frame_context)
                        bind_native_frame_reference(physics, aligned_baseline)
                    if review_outputs and not cached_top_revision:
                        raw_masks = _phase_masks(predictions)
                        if frame_geometry is not None:
                            raw_masks = {name: mask & observation_domain
                                         for name, mask in raw_masks.items()}
                        writer.submit(
                            _write_mask_maps,
                            destination / "raw_optical_assets", run_id, model_sha256, frame_index,
                            (canvas_height, canvas_width), (content_x, content_y, content_width, content_height),
                            raw_masks)
                    if pipelined:
                        # Chronological part now; deliver_decided() receives the rest when it is ready.
                        if numeric_writer is not None:
                            numeric_current[frame_index] = current
                        decision_executor.submit(predictions, elapsed_h, current,
                                                 token=(frame_index, elapsed_h, timing))
                        timing["decision_pipelined"] = True
                    elif hasattr(physics, "apply_with_reference"):
                        predictions, report = physics.apply_with_reference(predictions, elapsed_h, current)
                    else:
                        predictions, report = physics.apply(predictions, elapsed_h)
                    if (frame_geometry is not None and
                            (physics_profile or {}).get("approved_g3_gas") is None):
                        predictions, observation_report = _retain_v5_observation_only_scores(
                            predictions, optical_for_observation,
                            observable=observation_domain,
                            physics_supported=domain,
                            review_safe_support=(photo_sand if
                                (physics_profile or {}).get(
                                    "v5_nominal_review_gas_minimum") is not None
                                else None))
                        report = {**report, **observation_report,
                                  **({"photo_safe_sand_pixels": int(np.count_nonzero(photo_sand)),
                                      "photo_only_context": True,
                                      "observation_only_policy": "track1-photo-safe-optical-only-v1"}
                                     if track1_photo_only else
                                     {"physical_sand_pixels": int(np.count_nonzero(physical_sand_domain)),
                                      "physics_supported_pixels": int(np.count_nonzero(domain)),
                                      "observation_only_policy": "source-optical-no-physical-claim-v1"})}
                    if map_outputs and not pipelined and hasattr(physics, "write_diagnostics"):
                        if review_outputs:
                            detached = _detached_diagnostics(physics) if writer.enabled else None
                            if detached is None:
                                physics.write_diagnostics(destination / "presence_diagnostics", run_id,
                                    frame_index, (canvas_height, canvas_width),
                                    (content_x, content_y, content_width, content_height))
                            else:
                                writer.submit(detached, destination / "presence_diagnostics", run_id,
                                    frame_index, (canvas_height, canvas_width),
                                    (content_x, content_y, content_width, content_height))
                        else:
                            writer.submit(_write_diagnostic_flags, destination / "presence_diagnostics", run_id,
                                          frame_index, _diagnostic_flags(physics, copy=writer.enabled))
            timing["physics_s"] = time.perf_counter() - physics_started
            slot_wait = getattr(predictor, "last_slot_wait_s", None)
            if slot_wait is not None:
                # physics_s includes waiting for a shared CPU slot; the work itself is
                # physics_s - physics_slot_wait_s.
                timing["physics_slot_wait_s"] = float(slot_wait)
            writing_started = time.perf_counter()
            if not pipelined:
                masks = finished_fields(predictions, frame_index, elapsed_h, current,
                                        report if physics is not None else {})
            if map_outputs and not pipelined:
                writer.submit(
                    _write_mask_maps,
                    destination / "review_assets",
                    run_id,
                    pipeline_sha256,
                    frame_index,
                    (canvas_height, canvas_width),
                    (content_x, content_y, content_width, content_height),
                    masks,
                )
            if progress is not None:
                progress(f"{run_id.upper()} frame {frame_index + 1}/{frame_count} · writing")
            if elapsed_h is None:
                elapsed_h = float(frame_index)
            if proposal_outputs and not pipelined:
                proposal_tasks.append(submit_proposals(
                    _proposal_inputs(predictions, copy=writer.enabled), frame_index, elapsed_h))
            # With background outputs this is snapshot + hand-over (+ any wait for the writer).
            if not pipelined:
                timing["writing_s"] = time.perf_counter() - writing_started
            timing["outputs_in_background"] = writer.enabled
            timing["total_s"] = time.perf_counter() - frame_started
            timing["finished_unix"] = time.time()
            frame_timings.append(timing)
            if progress is not None and getattr(progress, "accepts_frame_timing", False):
                progress(FRAME_TIMING_PREFIX + json.dumps(
                    _frame_timing_summary(run_id, frame_count, frame_timings)))
            if frame_index not in independent_controls:
                previous = current
                previous_index = frame_index
                if frame_geometry is not None:
                    if temporal_reference_contract == "ffac-v5-source-backed-extended-ring-v1":
                        previous_extended, previous_origin, previous_valid = (
                            v5_input.load_extended(frame_path, f"frame_{frame_index:03d}"))
                    else:
                        previous_extended, previous_origin, previous_valid = (
                            current, (0, 0), None)
        if reference_observer is not None:
            with predictor.cpu_slot() if hasattr(predictor, "cpu_slot") else nullcontext():
                reference_observer.through(float("inf"), previous=previous,
                    previous_index=previous_index, physics=physics)
        if pipelined:
            for decided in decision_executor.drain():
                deliver_decided(decided)
            decision_executor.close()
        if predictor is not None and hasattr(predictor, "close"):
            predictor.close()
            if _started is not None:
                _started.remove(predictor)
        writer.drain()
        for task in proposal_tasks:
            all_proposals.extend(task.result())
        if proposal_pool is not None:
            proposal_pool.close()
        writer.close()
        if numeric_writer is not None:
            numeric_writer.finalize()

        if output_mode != "proposals":
            _atomic_json(destination / "physics_reports" / f"{dataset}_{run_id}.json", {
                "physics_runtime": physics_runtime, "settings": (physics_profile or {}).get("settings", {}),
                "context": context_entry if context is not None else domain_record, "frames": physics_report,
                "phase_area_domain_pixels": (None if frame_geometry is not None else
                    content_height * content_width if domain is None else int(np.count_nonzero(domain))),
                "phase_area_denominator": {
                    "kind": ("frame-native-observation-domain" if frame_geometry is not None else
                             "image-crop" if domain is None else "porous-domain"),
                    **({"observation_policy": "current-baseline-previous-source-supported-eroded-8px-v1",
                        "outside_support": ("cyan-unresolved-not-predicted" if track1_photo_only else
                                            "transparent-not-predicted")}
                       if frame_geometry is not None else {}),
                    "pixels": (None if frame_geometry is not None else
                               content_height * content_width if domain is None else int(np.count_nonzero(domain))),
                    **({"per_frame_pixels": [int(row["observation_domain_pixels"])
                                             for row in physics_report]}
                       if frame_geometry is not None else {}),
                    "axis_label": ("Visible observation area [%]" if frame_geometry is not None else
                                   "Visible image crop area [%]" if domain is None else
                                   "Visible porous area [%]"),
                    "source": (context_entry if context is not None else (domain_record or {})).get("path"),
                },
                "frame_geometry_manifest_sha256": (None if frame_geometry is None else
                                                     frame_geometry.native_manifest_sha256),
                "timings": frame_timings,
                "output_execution": writer.report(),
                "proposal_execution": (proposal_pool.report() if proposal_pool is not None else
                    ({"mode": "output-thread" if writer.enabled else "inline"} if proposal_outputs else
                     {"mode": proposals_status, "tasks": 0, "processes": 0})),
                "decision_execution": (decision_executor.report() if pipelined
                                       else {"execution": "frozen-sequential-call"}),
                "raw_phase_map_output_dir": str(destination / "raw_optical_assets") if review_outputs else None,
                "output_mode": output_mode,
                "mass_constraints_applied": False,
                "probabilities_are_optical_scores_with_heuristic_physical_support": not track1_photo_only,
                "track1_photo_only_review": track1_photo_only,
                **({"physical_scale_qa_pass": v5_scale_warning is None,
                    "physical_scale_warning": v5_scale_warning}
                   if frame_geometry is not None else {}),
            })

        if radiometry is not None and map_outputs:
            _atomic_json(display_root / "native_frames" / run_id / "manifest.json", {
                **native_manifest,
                "radiometry_profile": profile,
                "analysis_source_review_root": str(review_root),
                "display_encoding": "8-bit lossless PNG; inference uses float32",
            })
            _atomic_json(display_root / "radiometry" / dataset / f"{run_id}.json", radiometry.report())
        if v5_input is not None:
            _atomic_json(destination / "v5_linear_input" / dataset / f"{run_id}.json",
                         v5_input.report())

        proposal_path = (
            destination
            / "review_assets"
            / "phase_quiz"
            / run_id
            / f"A_{pipeline_sha256[:16]}_q{PHASE_REVIEW_PROPOSAL_SCHEMA}"
            / "proposals.json"
        )
        if proposal_outputs:
            _atomic_json(
                proposal_path,
                {
                    "schema": PHASE_REVIEW_PROPOSAL_SCHEMA,
                    "candidate_id": candidate_id,
                    "model_id": model_id,
                    "initial_proposal_count": min(24, len(all_proposals)),
                    "proposal_count": len(all_proposals),
                    "proposals": all_proposals,
                },
            )
        selected_candidates[run_id] = candidate_id
        jobs.append(
            {
                **source_job,
                "run": run_id,
                "dataset": dataset,
                "role": "selected_model",
                "variant": "selected_model_validation",
                "grid_order": 1,
                "candidate_id": candidate_id,
                "candidate_folder": str(candidate_root),
                "review_output_dir": str(display_root),
                "analysis_source_review_root": str(review_root),
                "radiometry_profile_sha256": None if profile is None else profile["sha256"],
                "phase_map_output_dir": None if output_mode == "numerical" else str(destination / "review_assets"),
                "numeric_phase_output_dir": str(destination / "numeric_phase" / dataset / run_id) if numeric_outputs else None,
                "params_sha256": pipeline_sha256,
                "optical_model_sha256": model_sha256,
                "physics_runtime": physics_runtime,
                "frame_geometry_manifest_sha256": (None if frame_geometry is None else
                                                     frame_geometry.native_manifest_sha256),
                **({"physical_scale_qa_pass": v5_scale_warning is None,
                    "physical_scale_warning": v5_scale_warning}
                   if frame_geometry is not None else {}),
                "domain_mask": domain_record,
                "output_mode": output_mode,
                "proposals_status": proposals_status,
                "raw_phase_map_output_dir": str(destination / "raw_optical_assets") if physics is not None and review_outputs else None,
                "presence_diagnostics_dir": str(destination / "presence_diagnostics")
                    if physics is not None and hasattr(physics, "write_diagnostics") else None,
            }
        )

    campaign_id = f"phase-validation-{model_id}"
    manifest = {
        "schema": 1,
        "format": PHASE_VALIDATION_CAMPAIGN_FORMAT,
        "campaign_id": campaign_id,
        "phase_display_policy": PHASE_DISPLAY_POLICY,
        "output_mode": output_mode,
        "numeric_outputs": numeric_outputs,
        "proposals_status": proposals_status,
        "inference_provenance": inference_provenance,
        "created_utc": utc_now(),
        "source_campaign_root": str(source_root),
        "source_manifest_sha256": source_manifest_sha256,
        "model_id": model_id,
        "model_root": str(model_root),
        "model_sha256": model_sha256,
        "pipeline_sha256": pipeline_sha256,
        "physics_runtime": physics_runtime,
        "physics_profile": physics_profile,
        "models": {
            "selected_model": {
                "co2_path": str(
                    model_root / "models" / "generic" / "co2_presence_signal.json"
                ),
                "gas_path": str(
                    model_root / "models" / "generic" / "gas_within_co2_signal.json"
                ),
            }
        },
        "jobs": jobs,
    }
    _atomic_json(destination / "postflight_manifest.json", manifest)
    binding = {
        "output_mode": output_mode,
        "numeric_outputs": numeric_outputs,
        "proposals_status": proposals_status,
        "inference_provenance": inference_provenance,
        "radiometry_runtime": radiometry_runtime,
        "v5_registered_runtime": v5_registered_runtime,
        "radiometry_profile": profile,
        "pipeline_sha256": pipeline_sha256,
        "physics_runtime": physics_runtime,
        "physics_profile": physics_profile,
        "schema": 1,
        "format": PHASE_VALIDATION_CAMPAIGN_FORMAT,
        "campaign_id": campaign_id,
        "model_id": model_id,
        "model_root": str(model_root),
        "model_sha256": model_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "ffac_root": str(destination),
        "ffum_root": str(destination),
        "combined_root": str(destination),
        "selected_candidates": selected_candidates,
        "roles": ["selected_model"],
        "phase_display_policy": PHASE_DISPLAY_POLICY,
        "run_count": len(selected_candidates),
        "completed_utc": utc_now(),
    }
    _atomic_json(binding_path, binding)
    return binding


def _quiz_generation_owner() -> dict[str, Any]:
    record: dict[str, Any] = {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now()}
    try:
        import psutil
        record["process_created"] = psutil.Process().create_time()
    except (ImportError, OSError):
        pass
    return record


def _release_dead_quiz_lock(lock: Path) -> bool:
    """Recover a crashed local owner; never infer remote/unknown-owner death."""
    try:
        original = lock.read_bytes()
        record = json.loads(original)
        if record.get("host") != socket.gethostname() or not record.get("process_created"):
            return False
        import psutil
        try:
            owner = psutil.Process(int(record["pid"]))
            dead = owner.create_time() != float(record["process_created"])
        except psutil.NoSuchProcess:
            dead = True
        except (psutil.AccessDenied, OSError, ValueError):
            return False
        if dead and lock.read_bytes() == original:
            lock.unlink()
            return True
    except (ImportError, OSError, ValueError, TypeError, KeyError):
        pass
    return False


def ensure_review_proposals(
    campaign_root: Path,
    run_id: str,
    *,
    progress: Callable[[str], None] | None = None,
    predictor_factory: Callable | None = None,
    inference_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Prepare one selected quiz on demand without touching maps or human answers.

    The probability fields are deliberately not archived for every phase map.
    A missing quiz therefore replays the selected model and chronological phase
    decisions for this run. An existing quiz, including a completed empty quiz,
    is reused. Callers may supply the checked Studio GPU/CPU predictor; the
    portable default is the tiled CPU predictor. A progress callback may raise
    to cancel; no partial proposal file is then published.
    """
    root = Path(campaign_root).expanduser().resolve()
    run = str(run_id).strip().casefold()
    manifest = _read_json(root / "postflight_manifest.json")
    jobs = [job for job in manifest.get("jobs", []) if str(job.get("run", "")).casefold() == run]
    if len(jobs) != 1:
        raise ValueError(f"Expected one selected-model candidate for {run!r}")
    job = jobs[0]
    fingerprint = str(job.get("params_sha256") or manifest.get("pipeline_sha256") or "")
    if not fingerprint or any(character not in "0123456789abcdef" for character in fingerprint.casefold()):
        raise ValueError("The selected quiz has no valid phase-pipeline fingerprint")
    assets = _resolve(root, job.get("phase_map_output_dir") or "review_assets")
    target = assets / "phase_quiz" / run / f"A_{fingerprint[:16]}_q{PHASE_REVIEW_PROPOSAL_SCHEMA}" / "proposals.json"

    def existing() -> bool:
        if not target.is_file():
            return False
        payload = _read_json(target)
        if payload.get("candidate_id") != job.get("candidate_id"):
            raise ValueError("The saved quiz belongs to another selected-model candidate")
        if not isinstance(payload.get("proposals"), list):
            raise ValueError("The saved quiz is incomplete; its existing file has been preserved")
        return True

    if existing():
        return target
    # Combined campaigns point to their original per-run packages. Resolve the
    # package rather than inventing a source revision from an assembled index.
    package = assets.parent
    if package != root and (package / "validation_binding.json").is_file():
        return ensure_review_proposals(package, run, progress=progress,
            predictor_factory=predictor_factory, inference_provenance=inference_provenance)
    binding = _read_json(root / "validation_binding.json")
    model_root = _resolve(root, binding.get("model_root") or manifest.get("model_root"))
    source_root = _resolve(root, manifest.get("source_campaign_root"))
    if _model_identity(model_root) != (binding.get("model_id"), binding.get("model_sha256")):
        raise ValueError("The selected model changed; regenerate its review package before requesting a quiz")
    if sha256_file(source_root / "postflight_manifest.json") != binding.get("source_manifest_sha256"):
        raise ValueError("The quiz source campaign changed; its saved phase maps have been preserved")
    profile = load_physics_profile(model_root)
    if physics_runtime_identity(profile) != binding.get("physics_runtime"):
        raise ValueError("The selected phase runtime changed; regenerate its review package before requesting a quiz")
    from darsia.fluidflower.phase_training.radiometry import model_profile, runtime_identity
    from darsia.fluidflower.phase_training.v5_linear_input import (
        model_v5_settings, runtime_identity as v5_runtime_identity,
    )
    radiometry = model_profile(HierarchicalPhaseModel.load(model_root, adapter="generic"))
    v5_settings = model_v5_settings(model_root)
    expected_runtime = (v5_runtime_identity() if v5_settings is not None else
                        None if radiometry is None else runtime_identity())
    if expected_runtime != binding.get("radiometry_runtime"):
        raise ValueError("The selected radiometry runtime changed; regenerate its review package before requesting a quiz")
    source_jobs = [entry for entry in _source_jobs(source_root) if str(entry["run"]).casefold() == run]
    if len(source_jobs) != 1:
        raise ValueError(f"The original source for {run!r} is unavailable")
    source_job = dict(source_jobs[0])
    candidate, review = _candidate_paths(source_root, source_job)
    source_job.update(candidate_folder=str(candidate), review_output_dir=str(review))
    if source_job.get("analysis_source_review_root"):
        source_job["analysis_source_review_root"] = str(_resolve(source_root, source_job["analysis_source_review_root"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name("generation.lock")
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError as error:
            if attempt == 0 and _release_dead_quiz_lock(lock):
                continue
            raise RuntimeError(f"Quiz generation for {run.upper()} is locked by another or unverifiable owner ({lock})") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_quiz_generation_owner(), stream)
        if existing():
            return target
        started = time.perf_counter()
        if progress is not None:
            progress(f"{run.upper()} · preparing requested quiz; phase maps and saved answers are unchanged")
        with tempfile.TemporaryDirectory(prefix=".quiz-replay-", dir=target.parent) as temporary:
            scratch = Path(temporary)
            source = scratch / "source"
            _atomic_json(source / "postflight_manifest.json", {"jobs": [source_job]})
            replay = scratch / "replay"
            replay_binding = materialize_phase_validation_campaign(model_root, source, replay,
                output_mode="proposals", generate_review_proposals=True, progress=progress,
                predictor_factory=predictor_factory, inference_provenance=inference_provenance)
            if replay_binding["pipeline_sha256"] != fingerprint:
                raise ValueError("Quiz replay no longer matches the selected phase maps; no proposal file was replaced")
            proposed = replay / "review_assets/phase_quiz" / run / target.parent.name / "proposals.json"
            payload = _read_json(proposed)
            if payload.get("candidate_id") != job.get("candidate_id"):
                raise ValueError("Quiz replay candidate identity changed; no proposal file was replaced")
            # Cancellation is checked once more after the last worker completed,
            # before making a complete pool visible to the review application.
            if progress is not None:
                progress(f"{run.upper()} · saving {payload['proposal_count']} requested quiz proposals")
            if existing():
                return target
            _atomic_json(target.with_name("generation.json"), {
                "status": "generated", "trigger": "requested-quiz", "run": run,
                "model_id": binding["model_id"], "model_sha256": binding["model_sha256"],
                "pipeline_sha256": fingerprint, "source_manifest_sha256": binding["source_manifest_sha256"],
                "inference_provenance": inference_provenance or {"backend": "tiled-cpu"},
                "proposal_count": payload["proposal_count"], "seconds": time.perf_counter() - started,
                "phase_maps_rewritten": False, "completed_utc": utc_now(),
            })
            _atomic_json(target, payload)
        return target
    finally:
        lock.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-mode", choices=("review", "production", "numerical"), default="review",
                        help="Numerical streams final float32 fields for mass analysis without PNGs; production retains legacy final PNGs.")
    parser.add_argument("--generate-review-proposals", action="store_true",
                        help="Explicitly generate quiz proposals; ordinary map preparation defers them until the quiz is opened.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binding = materialize_phase_validation_campaign(
        args.model_root,
        args.source_campaign_root,
        args.output_root,
        output_mode=args.output_mode,
        generate_review_proposals=args.generate_review_proposals,
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(binding), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PHASE_VALIDATION_CAMPAIGN_FORMAT",
    "materialize_phase_validation_campaign",
    "ensure_review_proposals",
    "predict_image_tiled",
]

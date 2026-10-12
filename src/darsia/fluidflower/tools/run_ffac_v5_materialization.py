"""Bounded, resumable parallel FF_AC V5 full-native source materialization.

This launcher never creates an approval. It requires immutable source, code,
calibration and three independent renderer gates before scheduling 52 source
revisions/1045 frames. The approvals authorize image materialization only,
not phase-model publication, scientific acceptance, or mass analysis.

Example after EA has *actually passed* all gates::

    python -m darsia.fluidflower.tools.run_ffac_v5_materialization \
      preflight --spec approved_job.json
    python -m darsia.fluidflower.tools.run_ffac_v5_materialization \
      run --spec approved_job.json --workers 4

No scheduled task or background service is created by this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any
from uuid import uuid4

import psutil

from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
    aggregate_materialization, file_sha256, materialize_revision,
    source_inventory,
)
from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import source_manifests
from darsia.fluidflower.tools.ffac_v5_geometry_policy import (
    accepted_geometry_overrides, cohort_contract_from_rows,
)


CONTRACT = "ffac-v5-parallel-native-materialization-job-v1"
APPROVAL = "ffac-v5-full-native-validation-only-exception-v2"
AC45_EXCEPTION = "ffac-ea-v2-ac45-validation-only-source-exception-v1"
V4_CONTRACT = "ffac-v5-parallel-native-v4-validation-only-job-v1"
V4_APPROVAL = "ffac-v5-safe-interior-v4-validation-only-source-approval-v1"
V4_FRAME_AGGREGATE = "ffac-v5-validation-only-frame-geometry-52-v1"
V4_REQUIRED_FILES = (
    "training_request", "reference_state", "calibration_manifest",
    "source_binding_audit", "v4_baseline_lookup", "run_geometry_manifest",
    "frame_geometry_manifest", "geometry_params", "response",
    "photometry_module", "refgeom_module", "v4_native_core_module",
    "renderer_module", "renderer_helper_module", "chart_registration_module",
    "frame_context_module", "extended_ring_module", "temporal_alignment_module",
    "processor_module", "materializer_module",
)


def _verify_v4_approval(spec: dict[str, Any]) -> dict[str, Any]:
    """Require a new review-only approval; never inherit EA-v2/v3 gates."""
    approval_path = _checked_file(spec["approval_receipt"], "V4 approval")
    approval = _read(approval_path)
    if (approval.get("contract") != V4_APPROVAL
            or approval.get("status") != "approved-for-manual-validation-source-materialization"
            or approval.get("scope") != "studio-validation-only"
            or approval.get("renderer_candidate") != "EA"
            or approval.get("chemistry_alignment_mode") != "none-light-only"
            or approval.get("automatic_model_publication") is not False
            or approval.get("scientific_phase_or_mass_acceptance") is not False
            or approval.get("validation_only") is not True
            or approval.get("mass_eligible") is not False
            or approval.get("extended_pad_px") != spec.get("extended_pad_px")):
        raise ValueError("V4 source approval must explicitly limit the scientific scope")
    for name in V4_REQUIRED_FILES:
        if approval.get(f"{name}_sha256") != spec[name]["sha256"]:
            raise ValueError(f"V4 approval does not bind exact {name}")
    gates = spec.get("source_gates") or {}
    if set(gates) != {"baseline_B", "baseline_H", "yellow37", "tiny_positive"}:
        raise ValueError("V4 requires four independent, source-bound QA receipts")
    if approval.get("source_gates") != {
        name: entry["sha256"] for name, entry in gates.items()
    }:
        raise ValueError("V4 approval does not bind its source-only QA receipts")
    checked = {name: _read(_checked_file(entry, f"V4 {name} gate"))
               for name, entry in gates.items()}
    calibration_sha = spec["calibration_manifest"]["sha256"]
    for name in ("baseline_B", "baseline_H", "yellow37"):
        if checked[name].get("calibration_manifest_sha256") != calibration_sha:
            raise ValueError(f"V4 {name} QA belongs to another calibration")
    if (checked["baseline_B"].get("contract") !=
            "ffac-ea-v4-safe-interior-heldout-B-only-source-diagnostic-v1"
            or checked["baseline_B"].get("all_B_pass") is not True
            or checked["baseline_H"].get("contract") !=
            "ffac-ea-v4-safe-interior-independent-H-source-diagnostic-v1"
            or checked["baseline_H"].get("all_H_pass") is not True):
        raise ValueError("V4 baseline B/H independent source checks did not pass")
    yellow = checked["yellow37"]
    if (yellow.get("contract") !=
            "ffac-ea-v4-registered-temporal-yellow37-source-result-v1"
            or yellow.get("final_era_yellow_pass") is not False
            or yellow.get("checks", {}).get("log_G") is not False
            or yellow.get("rig_count") != 26 or yellow.get("new_count") != 11
            or approval.get("yellow37_status") !=
            "documented-failed-threshold-validation-only"):
        raise ValueError("V4 yellow-era failure was hidden or altered")
    tiny = checked["tiny_positive"]
    if (not str(tiny.get("contract") or "").startswith("ffac-ea-v4-")
            or tiny.get("corrected_positive_gate_pass") is not True):
        raise ValueError("V4 exact-label weak signal gate did not pass")
    return approval


def _verify_v4_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Preflight one immutable 52/1045 validation-only source campaign."""
    if (spec.get("renderer_candidate") != "EA"
            or spec.get("chemistry_alignment_mode") != "none-light-only"
            or spec.get("validation_only") is not True
            or spec.get("mass_eligible") is not False
            or spec.get("renderer_symbol") != "registered_edge_aware_raw_with_ring"
            or spec.get("renderer_import_module") !=
            "darsia.fluidflower.tools.ffac_raw_extended_reference"
            or spec.get("extended_pad_px") != 32):
        raise ValueError("V4 source job must use EA RAW in validation-only light mode")
    for name in V4_REQUIRED_FILES:
        _checked_file(spec[name], name)
    calibration = _read(Path(spec["calibration_manifest"]["path"]))
    if (calibration.get("contract") !=
            "ffac-ea-raw-ac28-safe-interior-calibration-v4"
            or calibration.get("convergence_pass") is not True
            or calibration.get("scientific_mass_eligible") is not False
            or calibration.get("absolute_facies_boundary_pass") is not False):
        raise ValueError("V4 calibration must retain its provisional geometry status")
    audit = _read(Path(spec["source_binding_audit"]["path"]))
    if (audit.get("contract") != "ffac-v5-revision-exact-reference-binding-52-v1"
            or audit.get("frozen_request_sha256") != spec["training_request"]["sha256"]):
        raise ValueError("V4 source binding belongs to another frozen request")
    from darsia.fluidflower.tools.ffac_v4_calibration_binding import V4BaselineLookup
    lookup = V4BaselineLookup(spec["v4_baseline_lookup"]["path"],
                              spec["source_binding_audit"]["path"])
    if lookup.lookup_sha256 != spec["v4_baseline_lookup"]["sha256"]:
        raise ValueError("V4 baseline lookup changed")
    revisions = source_manifests(Path(spec["training_request"]["path"]))
    if (len(revisions) != 52 or len({r["manifest_sha256"] for r in revisions}) != 52
            or len({r["run"] for r in revisions}) != 43
            or source_inventory(Path(spec["training_request"]["path"]))[
                "timeline_frames"] != 1045):
        raise ValueError("V4 request must bind exactly 52/1045 source revisions")
    expected = {row["manifest_sha256"] for row in revisions}
    if set(lookup.rows) != expected:
        raise ValueError("V4 baseline lookup differs from the frozen 52 revisions")
    for revision in revisions:
        if file_sha256(Path(revision["manifest_path"])) != revision["manifest_sha256"]:
            raise ValueError("V4 source manifest changed")
    geometry = _read(Path(spec["run_geometry_manifest"]["path"]))
    geometry_rows = geometry.get("rows") or []
    if (len(geometry_rows) != 52
            or {row.get("source_manifest_sha256") for row in geometry_rows} != expected
            or geometry.get("contract") != cohort_contract_from_rows(geometry_rows)):
        raise ValueError("V4 run geometry does not bind the exact frozen cohort")
    by_geometry = {row["source_manifest_sha256"]: row for row in geometry_rows}
    for revision in revisions:
        accepted_geometry_overrides(Path(revision["manifest_path"]),
                                    by_geometry[revision["manifest_sha256"]])
    frame_manifest = _read(Path(spec["frame_geometry_manifest"]["path"]))
    rows = frame_manifest.get("rows") or []
    by_frame = {row.get("source_manifest_sha256"): row for row in rows}
    if (frame_manifest.get("contract") != V4_FRAME_AGGREGATE
            or frame_manifest.get("validation_only") is not True
            or frame_manifest.get("mass_eligible") is not False
            or len(rows) != 52 or len(by_frame) != 52 or set(by_frame) != expected):
        raise ValueError("V4 frame geometry aggregate is incomplete or not review-only")
    from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
        _bound_frame_geometry_marker,
    )
    total = 0
    for revision in revisions:
        marker = _bound_frame_geometry_marker(Path(revision["manifest_path"]),
            by_frame[revision["manifest_sha256"]])
        total += len(marker["frames"])
    if total != 1045:
        raise ValueError("V4 frame geometry omitted source frames")
    _verify_v4_approval(spec)
    return revisions


def _verify_ac45_exception(spec: dict[str, Any], approval: dict[str, Any]) -> None:
    """Keep the failed .05 gate visible while allowing a review-only source set."""
    entry = spec["ac45_exception_receipt"]
    path = _checked_file(entry, "AC45 validation-only exception")
    if approval.get("ac45_exception_receipt_sha256") != entry["sha256"]:
        raise ValueError("Approval does not bind the documented AC45 exception")
    exception = _read(path)
    if (exception.get("contract") != AC45_EXCEPTION
            or exception.get("status") != "documented-failure-validation-only"
            or exception.get("source_only_0p05_gate_pass") is not False
            or exception.get("automatic_model_publication") is not False
            or exception.get("scientific_phase_or_mass_acceptance") is not False
            or exception.get("calibration_manifest_sha256") !=
            spec["calibration_manifest"]["sha256"]):
        raise ValueError("AC45 exception misstates its failed source-only gate")
    original_path = _checked_file(exception["unmasked_gate"], "AC45 unmasked gate")
    masked_path = _checked_file(exception["rescol_masked_gate"], "AC45 masked gate")
    original, masked = _read(original_path), _read(masked_path)
    holdout = [r for r in original["results"] if r["role"] == "whole_run_holdout"]
    failed = [r["run"] for r in holdout if r["pass_gate"] is not True]
    if (original.get("whole_run_holdout_pass") is not False
            or failed != ["ac45"] or len(holdout) != 4
            or masked.get("all_masked_B_H_sand_pass") is not False
            or masked.get("original_unmasked_gate_sha256") !=
            exception["unmasked_gate"]["sha256"]
            or masked.get("calibration_manifest_sha256") !=
            spec["calibration_manifest"]["sha256"]):
        raise ValueError("AC45 is not the sole documented failed holdout")
    ac45 = holdout[0]
    frames = masked["rows"][0]["frames"]
    observed = dict(unmasked_B=float(ac45["sand_block_median_abs_log"]),
        masked_B=float(frames["B"]["B_RESCol_masked"]["median_abs_log"]),
        unmasked_H=float(frames["H"]["unmasked"]["median_abs_log"]),
        masked_H=float(frames["H"]["B_RESCol_masked"]["median_abs_log"]))
    threshold = float(original["thresholds"]["sand_block_median_abs_log"])
    if (threshold != .05 or any(value <= threshold for value in observed.values())
            or exception.get("observed_median_abs_log") != observed
            or exception.get("frozen_threshold") != threshold):
        raise ValueError("AC45 exception changed the original frozen threshold/results")


def chemistry_pilot_indices(spec_path: Path) -> dict[str, list[int]]:
    """Freeze AC28, chemistry arms and AC22 control at fit/holdout times.

    The returned keys are exact native source-manifest hashes, never run names.
    It is a source-only materialization plan; it does not approve the optional
    chemistry transform or train a phase model.
    """
    plan = _read(spec_path)
    contract = plan.get("contract")
    if (contract not in {"ffac-v5-separate-chemistry-alignment-preregistered-v1",
                         "ffac-v5-separate-chemistry-alignment-preregistered-v2"}
            or plan.get("status") != "design-only-no-fit-no-production"):
        raise ValueError("Chemistry pilot lacks its frozen source-only plan")
    expected = ({"ac28", "ac20", "ac24", "ac27", "ac51"} if contract.endswith("v1")
                else {"ac28", "ac20", "ac21", "ac22", "ac24", "ac27", "ac51"})
    rows = [r for r in plan["source_rows"] if r["run"] in expected]
    if {r["run"] for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Chemistry pilot source rows are incomplete or ambiguous")
    selected = {}
    for row in rows:
        if file_sha256(Path(row["revision_manifest_path"])) != row["revision_manifest_sha256"]:
            raise ValueError("Preregistered chemistry source manifest changed")
        fit = set(int(x) for x in row["fit_frame_indices"].values())
        holdout = row["holdout_frame_indices"]
        test = set(int(holdout[key]) for key in ("0.5", "4", "8", "12"))
        late = set(int(x) for x in holdout["late_ge_48h_distinct"])
        indices = sorted(fit | test | late)
        if fit & (test | late) or max(indices) >= int(row["frame_count"]):
            raise ValueError("Chemistry pilot fit/holdout frame split is invalid")
        selected[row["revision_manifest_sha256"]] = indices
    return selected


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checked_file(entry: dict[str, Any], label: str) -> Path:
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{label} changed: {path}")
    return path


def _gate_pass(path: Path, expected_sha: str, candidate: str,
               expected_field: str, calibration_sha: str, gate_name: str) -> None:
    if file_sha256(path) != expected_sha:
        raise ValueError(f"V5 renderer gate receipt changed: {path}")
    value = _read(path)
    if (candidate == "EA" and gate_name == "late_yellow_weak" and
            value.get("contract") != "ffac-ea-v2-final-saturated-yellow-era-comparison-v1"):
        raise ValueError("EA yellow gate must be the frozen final-era comparison")
    if candidate == "EA":
        if value.get("candidate") not in (None, "EA-native-v5-fixedpoint-v2"):
            raise ValueError("EA source gate belongs to another renderer")
    elif value.get("primary", candidate) != candidate:
        raise ValueError("A source gate belongs to another renderer")
    if (value.get("calibration_manifest_sha256", calibration_sha) != calibration_sha
            or value.get(expected_field) is not True):
        raise ValueError(f"V5 renderer {candidate} did not pass {expected_field}: {path}")


def _verify_approval(spec: dict[str, Any]) -> dict[str, Any]:
    approval_path = _checked_file(spec["approval_receipt"], "full-native approval receipt")
    approval = _read(approval_path)
    candidate = spec["renderer_candidate"]
    if (approval.get("contract") != APPROVAL
            or approval.get("status") != "validation-only-ac45-exception"
            or approval.get("candidate") != candidate or candidate != "EA"
            or approval.get("scope") != "source-materialization-for-manual-studio-validation-only"
            or approval.get("chemistry_alignment_mode") != "none-light-only"
            or spec.get("chemistry_alignment_mode") != "none-light-only"
            or approval.get("automatic_model_publication") is not False
            or approval.get("scientific_phase_or_mass_acceptance") is not False):
        raise ValueError("No explicit, limited full-native materialization approval")
    expected = {
        "calibration_manifest_sha256": spec["calibration_manifest"]["sha256"],
        "source_binding_audit_sha256": spec["source_binding_audit"]["sha256"],
        "historical_calibration_manifest_sha256": spec["historical_calibration_manifest"]["sha256"],
        "historical_parity_receipt_sha256": spec["historical_parity_receipt"]["sha256"],
        "historical_chart_manifest_sha256": spec["historical_chart_manifest"]["sha256"],
        "chart_registration_module_sha256": spec["chart_registration_module"]["sha256"],
        "renderer_module_sha256": spec["renderer_module"]["sha256"],
        "renderer_helper_module_sha256": spec["renderer_helper_module"]["sha256"],
        "processor_module_sha256": spec["processor_module"]["sha256"],
        "materializer_module_sha256": spec["materializer_module"]["sha256"],
        "photometry_module_sha256": spec["photometry_module"]["sha256"],
        "geometry_params_sha256": spec["geometry_params"]["sha256"],
        "response_sha256": spec["response"]["sha256"],
        "training_request_sha256": spec["training_request"]["sha256"],
        "reference_state_sha256": spec["reference_state"]["sha256"],
        "run_geometry_manifest_sha256": spec["run_geometry_manifest"]["sha256"],
    }
    if any(approval.get(key) != value for key, value in expected.items()):
        raise ValueError("Full-native approval binds different code, source or calibration")
    _verify_ac45_exception(spec, approval)
    gate_fields = {"late_yellow_weak": "final_era_yellow_pass",
                   "tiny197": "corrected_positive_gate_pass"}
    for name, field in gate_fields.items():
        entry = approval.get("gates", {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Missing independent V5 {name} gate")
        _gate_pass(Path(entry["path"]), entry["sha256"], candidate, field,
                   spec["calibration_manifest"]["sha256"], name)
    calibration = _read(Path(spec["calibration_manifest"]["path"]))
    if (calibration.get("contract") !=
            "ffac-ea-raw-ac28-reference-geometry-calibration-v3"
            or calibration.get("convergence_pass") is not True
            or calibration.get("geometry_manifest_sha256") !=
            spec["run_geometry_manifest"]["sha256"]):
        raise ValueError("Selected V5 calibration is not converged on this run-native geometry")
    return approval


def verify_spec(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = _read(path)
    if spec.get("contract") not in (CONTRACT, V4_CONTRACT):
        raise ValueError("Unknown FF_AC V5 parallel materialization job contract")
    try:
        rawpy = importlib.import_module("rawpy")
    except ImportError as exc:
        raise RuntimeError("CPU LibRaw/rawpy is required for source-exact ARW materialization") from exc
    if rawpy.__version__ != "0.27.1":
        raise ValueError(f"Uncertified rawpy version: {rawpy.__version__}")
    if spec["contract"] == V4_CONTRACT:
        return spec, _verify_v4_spec(spec)
    for name in ("training_request", "reference_state", "calibration_manifest",
                 "run_geometry_manifest",
                 "source_binding_audit", "historical_calibration_manifest",
                 "historical_parity_receipt", "historical_chart_manifest",
                 "chart_registration_module", "geometry_params", "response",
                 "photometry_module", "renderer_module", "processor_module",
                 "renderer_helper_module", "materializer_module"):
        _checked_file(spec[name], name)
    if "pilot_chemistry_spec" in spec:
        path = _checked_file(spec["pilot_chemistry_spec"], "chemistry pilot spec")
        if spec.get("pilot_frame_indices_by_revision") != chemistry_pilot_indices(path):
            raise ValueError("Selected-frame pilot does not match frozen chemistry split")
    audit = _read(Path(spec["source_binding_audit"]["path"]))
    if audit.get("frozen_request_sha256") != spec["training_request"]["sha256"]:
        raise ValueError("Source binding audit belongs to another frozen request")
    if audit.get("contract") != "ffac-v5-revision-exact-reference-binding-52-v1":
        raise ValueError("Source binding audit is not revision-exact")
    states = _read(Path(spec["reference_state"]["path"]))["reference_state"]
    for row in audit["rows"]:
        if row["is_canonical"] and Path(row["camera_arw"]).name.casefold() != str(
                states[row["run"]]["file"]).casefold():
            raise ValueError("Canonical B differs from the SHA-bound Claude reference state")
    if spec.get("renderer_candidate") != "EA" and spec.get("renderer_candidate") != "A":
        raise ValueError("Renderer candidate must be A or EA")
    renderer_api = {
        "EA": ("darsia.fluidflower.tools.ffac_raw_native_ea",
               "registered_edge_aware_raw_linear"),
        "A": ("darsia.fluidflower.tools.ffac_raw_native",
              "registered_claude_raw_linear"),
    }
    if (spec.get("renderer_import_module"), spec.get("renderer_symbol")) != renderer_api[
            spec["renderer_candidate"]]:
        raise ValueError("Renderer candidate does not match its certified callable")
    _verify_approval(spec)
    revisions = source_manifests(Path(spec["training_request"]["path"]))
    if (len(revisions) != 52 or len({r["manifest_sha256"] for r in revisions}) != 52
            or len({r["run"] for r in revisions}) != 43):
        raise ValueError("Frozen request does not bind 52 revisions of 43 FF_AC runs")
    for revision in revisions:
        if file_sha256(Path(revision["manifest_path"])) != revision["manifest_sha256"]:
            raise ValueError("Source revision manifest changed before parallel batch")
    geometry_manifest = _read(Path(spec["run_geometry_manifest"]["path"]))
    geometry_rows = geometry_manifest.get("rows") or []
    indexed_geometry = {row["source_manifest_sha256"]: row for row in geometry_rows}
    if (geometry_manifest.get("contract") != cohort_contract_from_rows(geometry_rows)
            or len(geometry_rows) != 52 or len(indexed_geometry) != 52
            or set(indexed_geometry) != {row["manifest_sha256"] for row in revisions}):
        raise ValueError("Accepted run-native geometry does not cover exact 52 source revisions")
    for revision in revisions:
        accepted_geometry_overrides(Path(revision["manifest_path"]),
                                    indexed_geometry[revision["manifest_sha256"]])
    inventory = source_inventory(Path(spec["training_request"]["path"]))
    if inventory["timeline_frames"] != 1045:
        raise ValueError("Expected all 1045 source frames, with no short campaign")
    return spec, revisions


def _resources(output_root: Path, workers: int, spec: dict[str, Any]) -> dict[str, float]:
    control = spec.get("resource_limits") or {}
    reserve_gb = float(control.get("min_free_memory_gb", 24))
    peak_gb = float(control.get("estimated_peak_worker_gb", 3))
    min_disk_gb = float(control.get("min_free_output_gb", 50))
    if reserve_gb < 8 or peak_gb < 1 or min_disk_gb < 10:
        raise ValueError("V5 resource reserves below safe configured floor")
    memory_gb = psutil.virtual_memory().available / (1024**3)
    volume = output_root
    while not volume.exists() and volume != volume.parent:
        volume = volume.parent
    disk_gb = shutil.disk_usage(volume).free / (1024**3)
    return dict(available_memory_gb=memory_gb, free_output_disk_gb=disk_gb,
        memory_reserve_gb=reserve_gb, estimated_peak_worker_gb=peak_gb,
        disk_reserve_gb=min_disk_gb,
        permit_new_worker=memory_gb >= reserve_gb + workers * peak_gb
        and disk_gb >= min_disk_gb + 5)


def _processor(spec: dict[str, Any]):
    module = importlib.import_module(spec["renderer_import_module"])
    if Path(module.__file__).resolve() != Path(spec["renderer_module"]["path"]).resolve():
        raise ValueError("Imported RAW renderer differs from SHA-bound module file")
    renderer = getattr(module, spec["renderer_symbol"])
    helper = importlib.import_module(spec["renderer_helper_import_module"])
    if Path(helper.__file__).resolve() != Path(spec["renderer_helper_module"]["path"]).resolve():
        raise ValueError("Imported RAW geometry helper differs from SHA-bound module file")
    processor_module = importlib.import_module(spec["processor_import_module"])
    if (Path(processor_module.__file__).resolve() !=
            Path(spec["processor_module"]["path"]).resolve()):
        raise ValueError("Imported V5 processor differs from SHA-bound module file")
    cls = getattr(processor_module, spec["processor_class"])
    identity = dict(module_sha256=spec["renderer_module"]["sha256"],
        helper_module_sha256=spec["renderer_helper_module"]["sha256"],
        gate_receipt_path=spec["approval_receipt"]["path"],
        gate_receipt_sha256=spec["approval_receipt"]["sha256"],
        primary=spec["renderer_candidate"],
        full_native_approval_sha256=spec["approval_receipt"]["sha256"])
    return cls(calibration_manifest=Path(spec["calibration_manifest"]["path"]),
        source_binding_audit=Path(spec["source_binding_audit"]["path"]),
        historical_calibration_manifest=Path(spec["historical_calibration_manifest"]["path"]),
        historical_parity_receipt=Path(spec["historical_parity_receipt"]["path"]),
        historical_chart_manifest=Path(spec["historical_chart_manifest"]["path"]),
        geometry_params=Path(spec["geometry_params"]["path"]),
        response_path=Path(spec["response"]["path"]),
        photometry_module_path=Path(spec["photometry_module"]["path"]),
        renderer=renderer, renderer_identity=identity)


def _processor_v4(spec: dict[str, Any]):
    """Construct only the separately SHA-bound V4 frame processor."""
    renderer_module = importlib.import_module(spec["renderer_import_module"])
    processor_module = importlib.import_module(spec["processor_import_module"])
    if (Path(renderer_module.__file__).resolve() !=
            Path(spec["renderer_module"]["path"]).resolve()
            or Path(processor_module.__file__).resolve() !=
            Path(spec["processor_module"]["path"]).resolve()):
        raise ValueError("V4 runtime module differs from immutable job spec")
    renderer = getattr(renderer_module, spec["renderer_symbol"])
    cls = getattr(processor_module, spec["processor_class"])
    return cls(
        calibration_manifest=Path(spec["calibration_manifest"]["path"]),
        baseline_lookup=Path(spec["v4_baseline_lookup"]["path"]),
        source_binding_audit=Path(spec["source_binding_audit"]["path"]),
        frame_geometry_manifest=Path(spec["frame_geometry_manifest"]["path"]),
        run_geometry_manifest=Path(spec["run_geometry_manifest"]["path"]),
        geometry_params=Path(spec["geometry_params"]["path"]),
        response_path=Path(spec["response"]["path"]),
        renderer=renderer,
        renderer_identity=dict(
            module_sha256=spec["renderer_module"]["sha256"],
            helper_module_sha256=spec["renderer_helper_module"]["sha256"],
            core_module_sha256=spec["v4_native_core_module"]["sha256"],
            refgeom_module_sha256=spec["refgeom_module"]["sha256"],
            frame_context_module_sha256=spec["frame_context_module"]["sha256"],
            full_native_approval_path=spec["approval_receipt"]["path"],
            full_native_approval_sha256=spec["approval_receipt"]["sha256"],
            extended_pad_px=spec["extended_pad_px"],
            validation_only=True, mass_eligible=False),
    )


def _worker(spec_path: str, spec_sha: str, revision: dict[str, Any],
            selected_indices: set[int] | None) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if file_sha256(Path(spec_path)) != spec_sha:
        raise ValueError("Parallel job spec changed after dispatch")
    # The parent has verified all 52 revisions and the 1045-frame inventory.
    # Workers recheck only their immutable code/approval/calibration and their
    # own source; repeating the whole NAS scan per revision wastes I/O.
    spec = _read(Path(spec_path))
    v4 = spec.get("contract") == V4_CONTRACT
    sealed_names = (("approval_receipt", *V4_REQUIRED_FILES) if v4 else (
        "approval_receipt", "calibration_manifest", "source_binding_audit",
        "run_geometry_manifest", "historical_calibration_manifest",
        "historical_parity_receipt", "historical_chart_manifest",
        "chart_registration_module", "geometry_params", "response",
        "photometry_module", "renderer_module", "renderer_helper_module",
        "processor_module", "materializer_module"))
    for name in sealed_names:
        _checked_file(spec[name], name)
    if file_sha256(Path(revision["manifest_path"])) != revision["manifest_sha256"]:
        raise ValueError("Worker source revision changed after parent preflight")
    geometry_manifest = _read(Path(spec["run_geometry_manifest"]["path"]))
    geometry_rows = [row for row in geometry_manifest["rows"]
                     if row["source_manifest_sha256"] == revision["manifest_sha256"]]
    if len(geometry_rows) != 1:
        raise ValueError("Worker source has no unique run-native geometry row")
    frame_rows = []
    if v4:
        frame_manifest = _read(Path(spec["frame_geometry_manifest"]["path"]))
        frame_rows = [row for row in frame_manifest["rows"]
                      if row["source_manifest_sha256"] == revision["manifest_sha256"]]
        if len(frame_rows) != 1:
            raise ValueError("V4 worker source has no unique per-frame geometry row")
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    t0 = time.monotonic()
    start_io = psutil.Process().io_counters()
    processor = _processor_v4(spec) if v4 else _processor(spec)
    receipt = materialize_revision(revision,
        output_root=Path(spec["output_root"]), processor=processor,
        reference_state_path=Path(spec["reference_state"]["path"]),
        reference_binding_path=Path(spec["source_binding_audit"]["path"]),
        geometry_row=geometry_rows[0],
        frame_geometry_row=frame_rows[0] if v4 else None,
        frame_indices=selected_indices)
    io = psutil.Process().io_counters()
    mem = psutil.Process().memory_info()
    receipt_path = (Path(spec["output_root"]) / "revisions"
                    / revision["manifest_sha256"] / "materialization_receipt.json")
    pilot_frames = []
    if selected_indices is not None:
        frame_dir = (Path(spec["output_root"]) / "revisions" / revision["manifest_sha256"]
                     / "manual_pairwise_review" / "native_frames" / revision["run"])
        for frame in receipt["frames"]:
            frame_receipt = frame_dir / f"frame_{frame['frame_index']:03d}.v5.json"
            pilot_frames.append(dict(frame_index=frame["frame_index"],
                elapsed_h=frame["elapsed_h"], receipt_path=str(frame_receipt),
                receipt_sha256=file_sha256(frame_receipt),
                native_jpeg_sha256=frame["native_jpeg_sha256"],
                v5_raw_detail_jpeg_path=frame["v5_raw_detail_jpeg_path"],
                v5_raw_detail_jpeg_sha256=frame["v5_raw_detail_jpeg_sha256"],
                log_linear_rgb_sha256=frame["log_linear_rgb_sha256"],
                **({} if not v4 else dict(
                    v5_extended_ring_path=frame["v5_extended_ring_path"],
                    v5_extended_ring_sha256=frame["v5_extended_ring_sha256"]))))
    return dict(run=revision["run"], source_manifest_sha256=revision["manifest_sha256"],
        pilot_only=selected_indices is not None,
        pilot_frames=pilot_frames,
        revision_receipt_path=str(receipt_path) if selected_indices is None else None,
        revision_receipt_sha256=file_sha256(receipt_path) if selected_indices is None else None,
        frame_count=receipt["frame_count"], elapsed_s=time.monotonic()-t0,
        worker_peak_rss_bytes=getattr(mem, "peak_wset", mem.rss),
        os_read_bytes=max(0, io.read_bytes-start_io.read_bytes),
        os_write_bytes=max(0, io.write_bytes-start_io.write_bytes))


def _acquire_lock(path: Path, spec_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read(path)
        if (existing.get("host") != socket.gethostname()
                or psutil.pid_exists(int(existing.get("pid", -1)))):
            raise RuntimeError(f"Another V5 materialization runner holds {path}")
        preserved = path.with_name(f"runner.lock.stale.{int(time.time())}.json")
        os.replace(path, preserved)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(),
                       started_utc=datetime.now(timezone.utc).isoformat(),
                       spec_sha256=spec_sha), stream)
        stream.flush()
        os.fsync(stream.fileno())


def run(spec_path: Path, workers: int, *, pilot: bool = False) -> dict[str, Any]:
    if not (1 <= workers <= 8):
        raise ValueError("Start with at most 8 bounded CPU workers")
    spec_path = Path(spec_path)
    spec_sha = file_sha256(spec_path)
    spec, revisions = verify_spec(spec_path)
    nas_cap = int((spec.get("resource_limits") or {}).get("max_parallel_nas_revisions", 6))
    if not 1 <= nas_cap <= 8:
        raise ValueError("NAS revision concurrency must be between 1 and 8")
    inflight_limit = min(workers, nas_cap)
    pilot_by_revision = spec.get("pilot_frame_indices_by_revision") or {}
    if pilot:
        if "pilot_chemistry_spec" not in spec:
            raise ValueError("Selected-frame pilot requires SHA-bound preregistered chemistry plan")
        if not isinstance(pilot_by_revision, dict) or not pilot_by_revision:
            raise ValueError("Pilot frames must be frozen by source SHA in the immutable job spec")
        selected = []
        for revision in revisions:
            if revision["manifest_sha256"] not in pilot_by_revision:
                continue
            indices = pilot_by_revision[revision["manifest_sha256"]]
            if not isinstance(indices, list) or not indices or not all(
                isinstance(i, int) and i >= 0 for i in indices):
                raise ValueError("Invalid preregistered pilot frame indices")
            selected.append(revision)
        if not selected or set(pilot_by_revision) != {r["manifest_sha256"] for r in selected}:
            raise ValueError("Pilot refers to sources outside the frozen 52 revisions")
        revisions = selected
    output_root = Path(spec["output_root"])
    status = _resources(output_root, inflight_limit, spec)
    if not status["permit_new_worker"]:
        raise RuntimeError(f"Insufficient RAM/output disk for {inflight_limit} workers: {status}")
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / "runner.lock"
    _acquire_lock(lock, spec_sha)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    t0 = time.monotonic()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    try:
        job_path = output_root / "runner_job.json"
        job = dict(contract=CONTRACT, job_spec_path=str(spec_path), job_spec_sha256=spec_sha,
            frozen_request_sha256=spec["training_request"]["sha256"],
            candidate=spec["renderer_candidate"], output_root=str(output_root))
        if job_path.exists():
            if _read(job_path) != job:
                raise ValueError("Output root contains another immutable parallel V5 job")
        else:
            _write_atomic(job_path, job)
        iterator = iter(revisions)
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while True:
                while len(futures) < inflight_limit:
                    try:
                        revision = next(iterator)
                    except StopIteration:
                        break
                    pressure = _resources(output_root, len(futures)+1, spec)
                    if not pressure["permit_new_worker"]:
                        raise RuntimeError(f"Stopped scheduling at RAM/output pressure: {pressure}")
                    indices = set(pilot_by_revision[revision["manifest_sha256"]]) if pilot else None
                    future = pool.submit(_worker, str(spec_path), spec_sha, revision, indices)
                    futures[future] = revision
                if not futures:
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    revision = futures.pop(future)
                    try:
                        result = future.result()
                        completed.append(result)
                        print(f"V5 {len(completed)}/{len(revisions)} {result['run']} "
                              f"{result['frame_count']} frames in {result['elapsed_s']:.1f}s", flush=True)
                    except Exception as exc:
                        failed.append(dict(run=revision["run"],
                            source_manifest_sha256=revision["manifest_sha256"],
                            exception=repr(exc)))
                        # No automatic retry or replacement on a source/QA failure.
                        for queued in futures:
                            queued.cancel()
                    progress = dict(contract=CONTRACT, spec_sha256=spec_sha,
                        pilot_only=pilot,
                        candidate=spec["renderer_candidate"], completed=completed,
                        failed=failed, active=[r["manifest_sha256"] for r in futures.values()],
                        worker_limit=workers, nas_inflight_limit=inflight_limit,
                        elapsed_s=time.monotonic()-t0,
                        completed_frames=sum(r["frame_count"] for r in completed),
                        frame_throughput_per_hour=sum(r["frame_count"] for r in completed)
                        / max(time.monotonic()-t0, .001)*3600,
                        resources=_resources(output_root, 1, spec))
                    _write_atomic(output_root / "runner_progress.json", progress)
                if failed:
                    raise RuntimeError(f"V5 materialization failed; preserved partial receipts: {failed}")
        if pilot:
            pilot_receipt = dict(contract=CONTRACT, pilot_only=True,
                spec_sha256=spec_sha, completed=completed,
                elapsed_s=time.monotonic()-t0,
                completed_frames=sum(r["frame_count"] for r in completed),
                frame_throughput_per_hour=sum(r["frame_count"] for r in completed)
                / max(time.monotonic()-t0, .001)*3600)
            pilot_path = output_root / "pilot_receipt.json"
            if pilot_path.exists():
                prior = _read(pilot_path)
                if (prior.get("spec_sha256") != spec_sha
                        or {(r["source_manifest_sha256"],
                             tuple((f["frame_index"], f["receipt_sha256"])
                                   for f in r["pilot_frames"]))
                            for r in prior["completed"]}
                        != {(r["source_manifest_sha256"],
                             tuple((f["frame_index"], f["receipt_sha256"])
                                   for f in r["pilot_frames"]))
                            for r in completed}):
                    raise ValueError("Existing immutable pilot covers different source frames")
                pilot_receipt = prior
            else:
                _write_atomic(pilot_path, pilot_receipt)
            return dict(status="pilot-complete-no-training-manifests",
                pilot_receipt_path=str(pilot_path),
                pilot_receipt_sha256=file_sha256(pilot_path),
                frames=pilot_receipt["completed_frames"], elapsed_s=pilot_receipt["elapsed_s"])
        receipt = aggregate_materialization(Path(spec["training_request"]["path"]), output_root)
        return dict(status="complete", batch_receipt_path=str(output_root / "materialization_receipt.json"),
            batch_receipt_sha256=file_sha256(output_root / "materialization_receipt.json"),
            frames=receipt["timeline_frames"], revisions=receipt["source_revisions"],
            elapsed_s=time.monotonic()-t0)
    finally:
        # Preserve the lock record as an audit trail; a finished marker makes
        # a subsequent, explicitly resumed invocation distinguishable.
        if lock.exists():
            finished = lock.with_name(f"runner.lock.finished.{int(time.time())}.json")
            os.replace(lock, finished)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--spec", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--spec", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--pilot", action="store_true",
                            help="Only frozen selected frames; create no training manifests")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "status":
        path = args.output_root / "runner_progress.json"
        print(json.dumps(_read(path) if path.exists() else {"status": "not-started"}, indent=2))
    elif args.mode == "preflight":
        spec, revisions = verify_spec(args.spec)
        print(json.dumps(dict(status="ready-for-source-materialization-only",
            revisions=len(revisions), timeline_frames=1045,
            candidate=spec["renderer_candidate"],
            resources=_resources(Path(spec["output_root"]), 4, spec)), indent=2))
    else:
        print(json.dumps(run(args.spec, args.workers, pilot=args.pilot), indent=2))


if __name__ == "__main__":
    main()

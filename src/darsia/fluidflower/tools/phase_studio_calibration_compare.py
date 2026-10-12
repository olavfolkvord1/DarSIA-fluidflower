"""Compare individual and shared native calibrations with one optical pass.

Scientific arrays stay in a bounded, reusable local cache. Result maps and
auditable summaries belong to the selected series' NAS result directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from darsia.fluidflower import concentration_configuration as configuration
from darsia.fluidflower import concentration_inputs as inputs
from darsia.fluidflower import concentration_native as native
from darsia.fluidflower import concentration_signal_calibration as calibration
from darsia.fluidflower.series_paths import local_cache_root
from darsia.fluidflower.tools import phase_studio_native_mass as ordinary


VERSION = "native-concentration-calibration-comparison-v1"
CACHE_VERSION = "native-concentration-signal-cache-v1"
_WORKER_THREAD_LIMIT = None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def select_runs(request, catalog):
    """Account for every requested run, including explicitly blocked inputs."""
    if request.get("model_id") != catalog["model_id"]:
        raise ValueError("Phase model changed; reload the concentration workspace.")
    available = {job["run"]: job for job in catalog["jobs"]}
    requested = request.get("runs", list(available))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Select unique registered experiments.")
    if any(name not in available for name in requested):
        raise ValueError("A requested experiment is absent from the selected phase model.")
    exclusions = request.get("exclude_runs", {})
    if (not isinstance(exclusions, dict) or any(name not in requested for name in exclusions)
            or any(not isinstance(reason, str) or not reason.strip() for reason in exclusions.values())):
        raise ValueError("Explicit experiment exclusions need a requested experiment and an audited reason.")
    ready, blocked = [], []
    for name in requested:
        if name in exclusions:
            blocked.append(dict(run=name, reasons=[exclusions[name]], stage="explicit_measurement_exclusion"))
            continue
        job = dict(available[name])
        detail = configuration.describe_run(job)
        if not detail["ready"]:
            blocked.append(dict(run=name, reasons=detail["reasons"], stage="configuration"))
            continue
        job["concentration"] = detail
        ready.append(job)
    groups = {job["concentration"]["chemistry_group"] for job in ready}
    if (len(ready) < 2 or len(groups) != 1
            or groups & {"uncertainty", "provisional", "unconfirmed"}
            or any(not job["concentration"].get("shared_fit_eligible") for job in ready)):
        raise ValueError("Comparison requires at least two registered experiments with compatible chemistry.")
    return ready, blocked


def select_frames(request, inspected):
    """Record explicit measurement-coverage exclusions without renumbering photos."""
    if ("image_selection_receipt_path" in request
            or "image_selection_receipt_sha256" in request):
        return _select_frames_from_receipt(request, inspected)
    overrides = request.get("frame_indices_by_run", {})
    reasons = request.get("frame_exclusion_reasons", {})
    if not isinstance(overrides, dict) or any(run not in inspected for run in overrides):
        raise ValueError("Frame selection refers to an unregistered or blocked experiment.")
    exclusions = []
    for name, metadata in inspected.items():
        frames = metadata["frames"]
        indices = overrides.get(name, list(range(len(frames))))
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or index < 0 or index >= len(frames) for index in indices)
                or indices != sorted(set(indices))):
            raise ValueError("Frame indices must be unique, chronological and inside the registered timeline.")
        omitted = sorted(set(range(len(frames))) - set(indices))
        if omitted:
            reason = reasons.get(name)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Every excluded frame needs an explicit measurement-coverage reason.")
            exclusions.extend(dict(run=name, frame_index=index,
                                   elapsed_h=frames[index].get("elapsed_h"), reason=reason)
                              for index in omitted)
        metadata["selected_frame_indices"] = indices
    return exclusions


def _selection_manifest_runs(manifest):
    """Extract the complete input timelines, including a multi-run wrapper."""
    raw_runs = manifest.get("runs")
    if raw_runs is None:
        raw_runs = [manifest]
    elif isinstance(raw_runs, dict):
        raw_runs = [dict(value, run=key) for key, value in raw_runs.items()]
    if not isinstance(raw_runs, list):
        raise ValueError("Image-selection manifest has no runs list")
    result = {}
    for entry in raw_runs:
        if not isinstance(entry, dict):
            raise ValueError("Image-selection manifest contains an invalid run")
        name = str(entry.get("run_id") or entry.get("run") or "").casefold()
        rows = entry.get("frames", entry.get("frame_sources"))
        if not name or name in result or not isinstance(rows, list):
            raise ValueError("Image-selection manifest has invalid or repeated run timelines")
        result[name] = rows
    return result


def _same_frame_identity(source, native, index, *, require_time=True):
    """Compare source identity rather than corrected-photo or phase-map paths."""
    try:
        source_index = source.get("frame_index", index)
        source_time = next((source[key] for key in ("elapsed_h", "time_hours", "time_h")
                            if key in source), None)
        source_path = source.get("source_path") or source.get("path") or source.get("image_path")
        return (type(source_index) is int and source_index == native["frame_index"]
                and (source_time is None and not require_time
                     or source_time is not None and float(source_time) == float(native["elapsed_h"]))
                and isinstance(source_path, str) and source_path == native["source_path"])
    except (KeyError, StopIteration, TypeError, ValueError):
        return False


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _select_frames_from_receipt(request, inspected):
    """Apply a frozen CLI image plan only when it matches every native frame.

    The receipt may bind a single native manifest or a multi-run wrapper. A
    wrapper has no per-run manifest SHA, so its *entire* timeline is compared
    with each inspected native manifest; selected frames alone are insufficient.
    """
    from darsia.fluidflower.image_selection_plan import (
        RECEIPT_FORMAT, resolve_selection_files,
    )

    if "frame_indices_by_run" in request or "frame_exclusion_reasons" in request:
        raise ValueError("An image-selection receipt cannot be combined with manual frame overrides")
    path = request.get("image_selection_receipt_path")
    digest = request.get("image_selection_receipt_sha256")
    if not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Image-selection receipt requires its absolute path and SHA-256")
    receipt_path = Path(path)
    if not receipt_path.is_absolute() or _sha256_file(receipt_path) != digest:
        raise ValueError("Image-selection receipt changed after calibration was queued")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if (not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT
            or receipt.get("schema_version") != 1
            or receipt.get("purpose") != "calibration" or receipt.get("status") != "ready"):
        raise ValueError("A ready calibration image-selection receipt is required")
    source = receipt.get("manifest")
    plan = receipt.get("plan")
    if not isinstance(source, dict) or not isinstance(plan, dict):
        raise ValueError("Image-selection receipt is missing its manifest or plan binding")
    for binding in (source, plan):
        bound_path = binding.get("path")
        bound_sha = binding.get("sha256")
        if (binding.get("hash_basis") != "file" or not isinstance(bound_path, str)
                or not bound_path or not isinstance(bound_sha, str)
                or len(bound_sha) != 64 or _sha256_file(bound_path) != bound_sha):
            raise ValueError("Image-selection manifest or plan changed after selection")
    if type(receipt.get("check_files")) is not bool:
        raise ValueError("Image-selection receipt has no explicit source-file policy")
    recreated = resolve_selection_files(
        source["path"], plan["path"], "calibration",
        check_files=receipt["check_files"])
    if (receipt.get("selection_plan") != recreated["selection_plan"]
            or receipt["status"] != recreated["status"]
            or receipt.get("runs") != recreated["runs"]):
        raise ValueError("Image-selection receipt differs from its manifest and TOML plan")
    manifest = json.loads(Path(source["path"]).read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("Image-selection manifest must be a JSON object")
    timeline = _selection_manifest_runs(manifest)
    receipt_runs = receipt.get("runs")
    if not isinstance(receipt_runs, list):
        raise ValueError("Image-selection receipt has no run records")
    records = {}
    for item in receipt_runs:
        if not isinstance(item, dict):
            raise ValueError("Image-selection receipt has an invalid run record")
        name = str(item.get("run_id") or "").casefold()
        if not name or name in records or item.get("status") != "ready":
            raise ValueError("Image-selection receipt has invalid or repeated runs")
        records[name] = item
    requested = [str(name).casefold() for name in request.get("runs", inspected)]
    if (len(requested) != len(set(requested)) or not set(requested) <= set(records)
            or set(timeline) != set(records) or not set(inspected) <= set(requested)):
        raise ValueError("Image-selection receipt must cover every requested experiment timeline")
    exclusions = []
    for name, metadata in inspected.items():
        frames = metadata["frames"]
        raw = timeline[name]
        run = records[name]
        if (not isinstance(raw, list) or len(raw) != len(frames)
                or run.get("input_frame_count") != len(frames)
                or any(not isinstance(row, dict) or not _same_frame_identity(row, frames[index], index)
                       for index, row in enumerate(raw))):
            raise ValueError(f"Image-selection timeline differs from registered native frames for {name}")
        selected = run.get("selected")
        omitted = run.get("omitted")
        if not isinstance(selected, list) or not isinstance(omitted, list):
            raise ValueError(f"Image-selection receipt is incomplete for {name}")
        indices = []
        accounted = set()
        for group, rows in (("selected", selected), ("omitted", omitted)):
            for row in rows:
                index = row.get("input_index") if isinstance(row, dict) else None
                if (type(index) is not int or index < 0 or index >= len(frames)
                        or index in accounted or not _same_frame_identity(
                            row, frames[index], index, require_time=group == "selected")):
                    raise ValueError(f"Image-selection frame identity differs from native timeline for {name}")
                accounted.add(index)
                if group == "selected":
                    indices.append(index)
                else:
                    reason = row.get("reason")
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(f"Image-selection exclusion lacks a reason for {name} frame {index}")
                    exclusions.append(dict(run=name, frame_index=index,
                                           elapsed_h=frames[index].get("elapsed_h"), reason=reason))
        if (not indices or indices != sorted(set(indices))
                or accounted != set(range(len(frames)))
                or run.get("selected_frame_count") != len(indices)):
            raise ValueError(f"Image-selection receipt does not account for every native frame in {name}")
        metadata["selected_frame_indices"] = indices
    request["image_selection_unused_runs"] = sorted(set(records) - set(requested))
    return exclusions


def cache_identity(job, config, provenance, frame, state):
    """Pin optics, geometry, fixed phase support and measured state together."""
    return dict(format=CACHE_VERSION, candidate=job["candidate"],
                physical_binding=ordinary.physical_binding(config),
                native_provenance=provenance,
                source_reference=dict(frame.source_reference), state=state,
                frame_index=frame.frame_index, elapsed_h=frame.time_hours,
                nominal_injected_mass_g=frame.nominal_injected_mass_g)


def _cache_files(folder):
    return sorted(path for path in folder.iterdir() if path.is_file()
                  and path.name != "receipt.json" and not path.name.endswith(".tmp"))


def publish_cache_receipt(folder, identity, summary, empty):
    """Commit a cache frame only after every numerical file has been written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ordinary.write(folder / "summary.json", summary)
    files = {path.name: dict(bytes=path.stat().st_size, sha256=ordinary.sha(path))
             for path in _cache_files(folder)}
    ordinary.write(folder / "receipt.json", dict(format=CACHE_VERSION,
        identity=identity, identity_sha256=_digest(identity), files=files, empty=bool(empty)))


def read_cache_receipt(folder, identity):
    """Reject incomplete or changed caches; never reinterpret them as observations."""
    folder = Path(folder)
    path = folder / "receipt.json"
    if not path.is_file():
        return None
    receipt = ordinary.read(path)
    if (receipt.get("format") != CACHE_VERSION or receipt.get("identity") != identity
            or receipt.get("identity_sha256") != _digest(identity)):
        raise ValueError("Cached optical response has a different scientific identity.")
    for name, record in receipt["files"].items():
        target = folder / name
        if (target.parent != folder or not target.is_file()
                or target.stat().st_size != record["bytes"]
                or ordinary.sha(target) != record["sha256"]):
            raise ValueError("Cached optical response changed or is incomplete.")
    receipt["summary"] = ordinary.read(folder / "summary.json")
    return receipt


def mass_metrics(rows):
    """Report mass-reference error without claiming independent concentration truth."""
    selected = [row for row in rows if row.get("nominal_injected_mass_g") is not None
                and row["nominal_injected_mass_g"] > 0]
    if not selected:
        return dict(observations=0)
    actual = np.asarray([row["total_mass_g"] for row in selected], float)
    target = np.asarray([row["nominal_injected_mass_g"] for row in selected], float)
    error = actual - target
    lower = np.asarray([row.get("total_mass_lower_g", row["total_mass_g"]) for row in selected], float)
    upper = np.asarray([row.get("total_mass_upper_g", row["total_mass_g"]) for row in selected], float)
    interval_error = np.maximum(lower - target, 0) + np.minimum(upper - target, 0)
    dose = float(target.max())
    return dict(observations=len(selected), dose_g=dose,
                rmse_g=float(np.sqrt(np.mean(error ** 2))),
                normalized_rmse=float(np.sqrt(np.mean(error ** 2)) / dose),
                mean_error_g=float(error.mean()), max_absolute_error_g=float(np.abs(error).max()),
                last_mass_g=float(actual[-1]), last_reference_g=float(target[-1]),
                last_relative_error=float(error[-1] / target[-1]),
                interval_rmse_g=float(np.sqrt(np.mean(interval_error ** 2))),
                normalized_interval_rmse=float(np.sqrt(np.mean(interval_error ** 2)) / dose),
                reference_within_interval_fraction=float(np.mean((target >= lower) & (target <= upper))),
                point_mass_interpretation="quantified support only; unknown support is retained in bounds",
                reference="nominal injected mass; not independent concentration validation")


def _prepare_signal(analysis, config, frame, run):
    if not np.any(frame.phase_labels):
        return None, {}
    state = configuration.pressure_temperature_at(config, frame.time_hours)
    prepared = analysis.prepare_signal(frame.corrected_rgb, frame.phase_labels, run_id=run,
        time_hours=frame.time_hours, nominal_injected_mass_g=frame.nominal_injected_mass_g, **state)
    return prepared, state


def prepare_runs(jobs, inspected, output, update):
    """One native optical pass; reusable per-frame signal caches are committed last."""
    provenance, bindings, records, summaries = {}, {}, {}, {}
    timing = dict(optical_preparation_seconds=0., summary_seconds=0.,
                  cache_write_seconds=0., input_loading_seconds=0.,
                  cache_verification_seconds=0., cache_hits=0, cache_misses=0)
    cache_root = local_cache_root() / "native-concentration-signal"
    for job in jobs:
        name = job["run"]
        update("Preparing native optical signal", run=name)
        config = configuration.load_run_configuration(job)
        analysis = ordinary.create_analysis(config)
        provenance[name] = analysis.signal_provenance()
        bindings[name] = ordinary.physical_binding(config)
        records[name], summaries[name] = [], []
        indices = inspected[name].get("selected_frame_indices", list(range(len(inspected[name]["frames"]))))
        for position, index in enumerate(indices):
            update("Preparing native optical signal", run=name, frame=index + 1,
                   selected_position=position + 1, frames=len(indices), timing=timing)
            tick = time.perf_counter()
            frame = inputs.load_frame(inspected[name], index)
            timing["input_loading_seconds"] += time.perf_counter() - tick
            empty = not np.any(frame.phase_labels)
            state = {} if empty else configuration.pressure_temperature_at(config, frame.time_hours)
            identity = cache_identity(job, config, provenance[name], frame, state)
            folder = cache_root / name / _digest(identity)
            tick = time.perf_counter()
            cached = read_cache_receipt(folder, identity)
            timing["cache_verification_seconds"] += time.perf_counter() - tick
            if cached is None:
                tick = time.perf_counter()
                prepared, _ = _prepare_signal(analysis, config, frame, name)
                timing["optical_preparation_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                summary = (analysis.empty_signal_summary(frame.phase_labels, **ordinary.empty_arguments(frame, name))
                           if prepared is None else analysis.summarize_signal(prepared))
                timing["summary_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                if prepared is not None:
                    native.save_signal_frame(folder, prepared)
                publish_cache_receipt(folder, identity, summary, empty)
                timing["cache_write_seconds"] += time.perf_counter() - tick
                timing["cache_misses"] += 1
                del prepared
            else:
                summary = cached["summary"]
                timing["cache_hits"] += 1
            record = dict(frame_index=index, cache=str(folder), empty=empty,
                          identity=identity, source_reference=dict(frame.source_reference))
            records[name].append(record)
            summaries[name].append(summary)
            del frame
        analysis.verify_sources()
        ordinary.write(output / "preparation" / (name + ".json"),
            dict(run=name, native_provenance=provenance[name], physical_binding=bindings[name],
                 frames=records[name], summaries=summaries[name]))
        del analysis, config
    prepared = dict(format=VERSION, native_calibrations=provenance,
                    physical_bindings=bindings, frames=records, summaries=summaries, timing=timing)
    ordinary.write(output / "prepared.json", prepared)
    return prepared


def fit_variants(prepared, output, request, update):
    """All calibration and held-out fits consume small cached histograms only."""
    summaries, provenance = prepared["summaries"], prepared["native_calibrations"]
    common = dict(unknown_policy=request.get("unknown_policy", "bounded_interval"), fit_gas=False)
    fits = {"individual": {}}
    for name, frames in summaries.items():
        update("Fitting individual concentration response", run=name)
        fits["individual"][name] = calibration.fit_signal_summaries(
            {name: frames}, {name: provenance[name]}, scope="individual", **common)
        fits["individual"][name]["physical_bindings"] = {name: prepared["physical_bindings"][name]}
    update("Fitting shared concentration response")
    fits["shared"] = calibration.fit_signal_summaries(summaries, provenance, scope="shared", **common)
    fits["shared"]["physical_bindings"] = prepared["physical_bindings"]
    update("Testing chronological and held-out experiments")
    diagnostics = calibration.cross_validate_signal_summaries(summaries, provenance, **common)
    for scope in ("individual", "shared"):
        selected = fits[scope] if scope == "individual" else {"series": fits[scope]}
        for name, fit in selected.items():
            fit.update(dataset=request["dataset"], overflow_mode="off",
                       fitting_scope_description="Common monotone signal correction applied to each experiment's registered native optical calibration"
                       if scope == "shared" else "Individual monotone correction of the registered native optical calibration")
            ordinary.write(output / scope / name / "calibration.json", fit)
    ordinary.write(output / "fits.json", fits)
    ordinary.write(output / "cross_validation.json", diagnostics)
    return fits, diagnostics


def _check_binding(analysis, config, prepared, run):
    current = analysis.signal_provenance()
    if current != prepared["native_calibrations"][run]:
        raise ValueError("Native optical calibration or scientific implementation changed since preparation.")
    if ordinary.physical_binding(config) != prepared["physical_bindings"][run]:
        raise ValueError("Physical inputs, measured protocols or baseline changed since preparation.")
    return current


def render_dilution_curve(path, rows, title):
    """Show plume expansion and concentration separately, without mass-driven maps."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 6.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    times = [row["elapsed_h"] for row in rows]
    areas = [row.get("aqueous_support_area_m2", np.nan) for row in rows]
    means = [row.get("mean_aqueous_concentration_g_l") for row in rows]
    means = [np.nan if value is None else value for value in means]
    axes[0].plot(times, areas, color="teal", marker=".", label="Fixed aqueous phase support")
    partial = any(not row.get("quantification_complete", True) for row in rows)
    qualifier = " · calibrated support only" if partial else ""
    axes[1].plot(times, means, color="darkorange", marker=".", label="Pore-volume-weighted mean" + qualifier)
    axes[0].set(ylabel="Aqueous support area · m²", title=title + " · area and dilution")
    axes[1].set(xlabel="Time since injection start · h", ylabel="Mean dissolved concentration · g/L")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    fig.savefig(path, dpi=120)


def render_variants(jobs, inspected, prepared, fits, output, catalog, request, update, progress_path=None):
    """Evaluate both production fits while each full-resolution image is loaded."""
    from darsia.fluidflower.tools.phase_studio_mass_inputs import register_variant
    from darsia.fluidflower.tools.phase_studio_series_mass import export_csv
    delivered = []
    labels = dict(individual="Individual signal calibration", shared="Shared series signal calibration")
    labels.update(request.get("result_labels", {}))
    for job in jobs:
        name = job["run"]
        config = configuration.load_run_configuration(job)
        analysis = ordinary.create_analysis(config)
        provenance = _check_binding(analysis, config, prepared, name)
        by_scope = {"individual": fits["individual"][name], "shared": fits["shared"]}
        rows, folders = {}, {}
        for scope, fit in by_scope.items():
            folder = output / scope / name
            folder.mkdir(parents=True, exist_ok=True)
            existing = folder / "result.json"
            if existing.is_file():
                saved = ordinary.read(existing)
                if (saved.get("complete") is not True or saved.get("selected_inputs", {}).get("candidate") != job["candidate"]
                        or saved.get("calibration_identity") != _digest(fit)):
                    raise ValueError("A saved result belongs to another calibration; select a new output directory.")
                register_variant(existing, catalog["results_root"])
                delivered.append(dict(run=name, scope=scope, result=str(existing), resumed=True))
            else:
                rows[scope], folders[scope] = [], folder
        if not rows:
            del analysis, config
            continue
        source_bindings = {}
        for position, record in enumerate(prepared["frames"][name]):
            index = record["frame_index"]
            update("Rendering individual and shared concentration maps", run=name,
                   frame=index + 1, selected_position=position + 1,
                   frames=len(prepared["frames"][name]), completed_results=len(delivered))
            frame = inputs.load_frame(inspected[name], index)
            if dict(frame.source_reference) != record["source_reference"]:
                raise ValueError("A photograph or fixed phase map changed after signal preparation.")
            if read_cache_receipt(record["cache"], record["identity"]) is None:
                raise ValueError("Prepared signal cache is incomplete; prepare the inputs again.")
            signal = None if record["empty"] else native.load_signal_frame(record["cache"])
            if signal is not None and not np.array_equal(signal.phase_labels, frame.phase_labels):
                raise ValueError("Cached signal phase support differs from the selected phase map.")
            for scope, selected_rows in rows.items():
                fit = by_scope[scope]
                parameters = calibration.SignalCalibrationParameters(**fit["parameters"])
                evaluation = (analysis.empty_evaluation(frame.phase_labels, **ordinary.empty_arguments(frame, name))
                              if signal is None else analysis.evaluate_signal(signal, parameters))
                fields = evaluation.pop("fields")
                fields_path = folders[scope] / f"fields_{index:04d}.npz"
                np.savez_compressed(fields_path, **fields)
                row = dict(evaluation, frame_index=index, source_name=frame.source_reference.get("source_name"),
                    source_reference=dict(frame.source_reference), injected_mass_g=frame.nominal_injected_mass_g,
                    fields=str(fields_path), preview=str(folders[scope] / f"frame_{index:04d}.png"),
                    calibration_role="fit to nominal injected mass; unknown support retained as interval", gas_reference=None)
                predicted = fit.get("predictions", {}).get(name, [])
                if position < len(predicted):
                    summary = predicted[position]
                    error = abs(evaluation["total_mass_g"] - summary["total_mass_g"])
                    bound = summary.get("summary_mass_error_bound_g", 0.)
                    if error > bound + max(1e-5, abs(evaluation["total_mass_g"]) * 2e-5):
                        raise ValueError("Exact concentration mass exceeds its fitted histogram error bound.")
                    row["summary_mass_error_g"] = error
                    row["summary_mass_error_bound_g"] = bound
                ordinary.render_frame(row["preview"], frame.corrected_rgb, fields, row,
                                      name.upper() + " · " + labels[scope], evaluation["solubility_max_g_l"])
                selected_rows.append(row)
                del fields
            for key, digest in (("photo", "photo_sha256"), ("phase", "phase_sha256")):
                source_bindings[frame.source_reference[key]] = frame.source_reference[digest]
            del frame, signal
        analysis.verify_sources()
        for scope, selected_rows in rows.items():
            folder, fit = folders[scope], by_scope[scope]
            curve = folder / "mass_curve.png"
            ordinary.render_curve(curve, selected_rows, name.upper() + " · " + labels[scope], True)
            dilution = folder / "dilution_curve.png"
            render_dilution_curve(dilution, selected_rows, name.upper() + " · " + labels[scope])
            result = dict(format=native.VERSION, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
                result_label=labels[scope], calibration_identity=_digest(fit),
                selected_inputs=dict(candidate=job["candidate"], review_root=job["review_root"]),
                frames=selected_rows, calibration=fit, method=dict(settings=dict(optical="native_color_path",
                    chemistry="darsia_simple_flash", restoration="none", calibration="monotone_preflash_signal",
                    response="native_preflash_monotone"),
                    native_provenance=provenance), physical_provenance=config.get("provenance", {}),
                source_bindings=source_bindings, mass_curve=str(curve), dilution_curve=str(dilution),
                overflow_curve=None, overflow_mode="off",
                series_calibration=str(output / "shared" / "series" / "calibration.json") if scope == "shared"
                                   else str(folder / "calibration.json"),
                phase_inference_repeated=False, phase_labels_modified=False,
                requested_frame_indices=[record["frame_index"] for record in prepared["frames"][name]],
                available_review_frames=len(inspected[name]["frames"]),
                excluded_frame_reason=request.get("frame_exclusion_reasons", {}).get(name),
                cross_validation=str(output / "cross_validation.json"),
                limitations=inspected[name]["limitations"] + config.get("provenance", {}).get("warnings", []) + [
                    "Fitted to nominal injected mass; this is not independent local concentration validation.",
                    "Uncalibrated phase support remains unknown and is represented by physical mass bounds.",
                    "The shared correction uses each experiment's original native optical calibration."])
            ordinary.write(folder / "result.json", result)
            export_csv(result, folder / "mass_per_frame.csv")
            register_variant(folder / "result.json", catalog["results_root"])
            delivered.append(dict(run=name, scope=scope, result=str(folder / "result.json"),
                                  metrics=mass_metrics(selected_rows), resumed=False))
            ordinary.write(progress_path or output / "delivery_progress.json", dict(complete=False, experiments=delivered))
        del analysis, config
    return delivered


def _limit_worker_threads(count):
    """Bound inner numerical pools, leaving parallelism at experiment level."""
    global _WORKER_THREAD_LIMIT
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(count)
    try:
        from threadpoolctl import threadpool_limits
        _WORKER_THREAD_LIMIT = threadpool_limits(limits=count)
    except ImportError:
        pass
    try:
        import cv2
        cv2.setNumThreads(count)
    except ImportError:
        pass


def _child_update(output, stage, run):
    status_path = Path(output) / "worker_status" / stage / (run + ".json")
    def update(message, **extra):
        if (Path(output) / "cancel_workers.json").is_file():
            raise InterruptedError("Comparison workers cancelled after another worker failed.")
        ordinary.write(status_path, dict(stage=message, updated_utc=datetime.now(timezone.utc).isoformat(), **extra))
    return update


def _prepare_worker(job, metadata, output):
    """Spawn target returns a small manifest path, never a dense image array."""
    folder = Path(output) / "preparation_workers" / job["run"]
    update = _child_update(output, "prepare", job["run"])
    try:
        prepare_runs([job], {job["run"]: metadata}, folder, update)
        update("complete", run=job["run"])
        return str(folder / "prepared.json")
    except Exception:
        ordinary.write(Path(output) / "worker_status" / "prepare" / (job["run"] + ".json"),
                       dict(stage="failed", error=traceback.format_exc()))
        raise


def _render_worker(job, metadata, output, catalog, request):
    output = Path(output)
    update = _child_update(output, "render", job["run"])
    try:
        prepared = ordinary.read(output / "prepared.json")
        fits = ordinary.read(output / "fits.json")
        delivered = render_variants([job], {job["run"]: metadata}, prepared, fits, output, catalog,
            request, update, progress_path=output / "worker_status" / "render" / (job["run"] + ".delivery.json"))
        update("complete", run=job["run"])
        return delivered
    except Exception:
        ordinary.write(output / "worker_status" / "render" / (job["run"] + ".json"),
                       dict(stage="failed", error=traceback.format_exc()))
        raise


def _parallel_jobs(target, arguments, *, workers, threads, output, stage, update):
    results = []
    cancel = output / "cancel_workers.json"
    if cancel.is_file():
        cancel.unlink()
    with ProcessPoolExecutor(max_workers=workers, initializer=_limit_worker_threads,
                             initargs=(threads,)) as pool:
        pending = {pool.submit(target, *args): args[0]["run"] for args in arguments}
        try:
            while pending:
                done, _ = wait(pending, timeout=2., return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                    del pending[future]
                states = {}
                for run in pending.values():
                    path = output / "worker_status" / stage / (run + ".json")
                    if path.is_file():
                        try:
                            states[run] = ordinary.read(path)
                        except (OSError, ValueError):
                            pass
                update("Parallel experiment " + stage, workers=workers, threads_per_worker=threads,
                       completed_runs=len(results), remaining_runs=len(pending), worker_status=states)
        except BaseException:
            ordinary.write(cancel, dict(reason="One worker failed; stop the related comparison workers"))
            for future in pending:
                future.cancel()
            raise
    return results


def prepare_parallel(jobs, inspected, output, update, workers, threads):
    manifests = _parallel_jobs(_prepare_worker,
        [(job, inspected[job["run"]], str(output)) for job in jobs],
        workers=workers, threads=threads, output=output, stage="prepare", update=update)
    prepared = dict(format=VERSION, native_calibrations={}, physical_bindings={}, frames={}, summaries={},
                    timing={"task_times_overlap": True, "workers": workers, "threads_per_worker": threads})
    for path in manifests:
        child = ordinary.read(path)
        for key in ("native_calibrations", "physical_bindings", "frames", "summaries"):
            prepared[key].update(child[key])
        for key, value in child["timing"].items():
            prepared["timing"][key] = prepared["timing"].get(key, 0) + value
    # Stable requested ordering makes fitting and receipts reproducible across scheduling.
    for key in ("native_calibrations", "physical_bindings", "frames", "summaries"):
        prepared[key] = {job["run"]: prepared[key][job["run"]] for job in jobs}
    ordinary.write(output / "prepared.json", prepared)
    return prepared


def run(request_path):
    """Run or resume one explicit comparison; do not start phase inference."""
    start = time.perf_counter()
    request_path = Path(request_path)
    request = ordinary.read(request_path)
    status_path = request_path.with_name("status.json")
    def update(stage, **extra):
        ordinary.write(status_path, dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
                                        elapsed_seconds=time.perf_counter() - start, **extra))
    try:
        stage = request.get("stage", "all")
        if stage not in {"all", "prepare", "fit", "render"}:
            raise ValueError("Choose preparation, fitting, rendering or the complete comparison.")
        if request.get("overflow_mode", "off") != "off":
            raise ValueError("This comparison requires explicit zero overflow constraints; measured overflow is a separate experiment.")
        update("Checking registered concentration inputs")
        catalog = inputs.discover_model_inputs(request["model_root"])
        jobs, blocked = select_runs(request, catalog)
        request["dataset"] = catalog["dataset"]
        output = Path(request["output"]).resolve()
        if not output.is_relative_to(Path(catalog["results_root"]).resolve()):
            raise ValueError("Concentration comparison results must belong to the selected series on NAS.")
        resources = request.get("workers", 1), request.get("threads_per_worker", 1)
        if any(type(value) is not int or value < 1 for value in resources):
            raise ValueError("Worker and thread budgets must be positive integers.")
        workers = min(resources[0], len(jobs), os.cpu_count() or 1)
        threads = resources[1]
        scientific_request = {key: value for key, value in request.items()
                              if key not in {"stage", "resume", "workers", "threads_per_worker"}}
        identity_path = output / "request_identity.json"
        if output.exists():
            if not request.get("resume") or not identity_path.is_file() or ordinary.read(identity_path) != scientific_request:
                raise ValueError("Use a new job directory, or resume its unchanged scientific request.")
        else:
            output.mkdir(parents=True)
            ordinary.write(identity_path, scientific_request)
        inspected = {job["run"]: inputs.inspect_run(job) for job in jobs}
        excluded_frames = select_frames(request, inspected)
        ordinary.write(output / "selection.json", dict(requested_runs=request.get("runs", [job["run"] for job in catalog["jobs"]]),
            selected_runs=[job["run"] for job in jobs], blocked_runs=blocked,
            excluded_frames=excluded_frames,
            **({"image_selection_unused_runs": request["image_selection_unused_runs"]}
               if "image_selection_unused_runs" in request else {}),
            available_frames=sum(len(value["frames"]) for value in inspected.values()),
            frames=sum(len(value["selected_frame_indices"]) for value in inspected.values())))
        prepared_path = output / "prepared.json"
        tick = time.perf_counter()
        if stage in {"all", "prepare"} and not prepared_path.is_file():
            prepared = (prepare_parallel(jobs, inspected, output, update, workers, threads) if workers > 1
                        else prepare_runs(jobs, inspected, output, update))
        else:
            prepared = ordinary.read(prepared_path)
        preparation_seconds = time.perf_counter() - tick
        if stage == "prepare":
            update("preparation_complete", prepared=str(prepared_path))
            return prepared
        tick = time.perf_counter()
        if stage in {"all", "fit"} and not (output / "fits.json").is_file():
            fits, diagnostics = fit_variants(prepared, output, request, update)
        else:
            fits = ordinary.read(output / "fits.json")
            diagnostics = ordinary.read(output / "cross_validation.json")
        fitting_seconds = time.perf_counter() - tick
        if stage == "fit":
            update("fitting_complete", fits=str(output / "fits.json"), diagnostics=str(output / "cross_validation.json"))
            return fits
        tick = time.perf_counter()
        if workers > 1:
            groups = _parallel_jobs(_render_worker,
                [(job, inspected[job["run"]], str(output), catalog, request) for job in jobs],
                workers=workers, threads=threads, output=output, stage="render", update=update)
            delivered = [item for group in groups for item in group]
            delivered.sort(key=lambda item: ([job["run"] for job in jobs].index(item["run"]), item["scope"]))
        else:
            delivered = render_variants(jobs, inspected, prepared, fits, output, catalog, request, update)
        rendering_seconds = time.perf_counter() - tick
        timing = dict(total_seconds=time.perf_counter() - start, preparation_seconds=preparation_seconds,
                      fitting_and_cross_validation_seconds=fitting_seconds, rendering_seconds=rendering_seconds,
                      preparation=prepared["timing"], runs=len(jobs), variants=len(delivered),
                      images=sum(len(value["selected_frame_indices"]) for value in inspected.values()),
                      resumed=bool(request.get("resume")), workers=workers, threads_per_worker=threads,
                      total_seconds_scope="current invocation wall time; parallel task durations are not additive")
        ordinary.write(output / "timing.json", timing)
        delivery = dict(format=VERSION, complete=True, experiments=delivered, blocked_runs=blocked,
                        excluded_frames=excluded_frames,
                        cross_validation=str(output / "cross_validation.json"), timing=str(output / "timing.json"))
        ordinary.write(output / "delivery.json", delivery)
        update("complete", delivery=str(output / "delivery.json"), timing=timing)
        return delivery
    except Exception:
        update("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    run(parser.parse_args().request)

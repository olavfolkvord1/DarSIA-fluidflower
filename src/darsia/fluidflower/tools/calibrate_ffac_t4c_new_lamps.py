"""Count T4c placements on the fixed 8-run/2-frame cohort and calibrate delta.

No model is fitted, no human phase labels are opened, and no prediction maps
are saved. Exact source preparation and post-floor decisions are shared with
the T4c producer. Only Claude's unchanged frozen calibration script selects
the shift; the held-out acceptance rules are left unchanged.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

SPEC_SHA256 = "53aad00f20bec56092c160477210613e7849f50467303aa0efd5243d90337daf"
SUPERSEDED_SPEC_SHA256 = "9b60543d089bbfa0de6d6febf0bdaaf039969398a70d78627d975171ad91f3e3"
CALIBRATOR_SHA256 = "6d2d36db574965362cb6d183e48e6ac1218dc01eb67e1400cdf2071453597ef4"
FROZEN_TARGET_SHA256 = "b05b378730e461ae93da48d77bf3af094c25d3d3bb84ecf80ab1c0c06be916c5"
CONTRACT = "ffac-t4c-overflow-calibration-sweep-v1"
RUNS = ("ac52", "ac54", "ac56", "ac57", "ac58", "ac59", "ac60", "ac61")
FRAMES = (8, 9)
PLACEMENTS = ("P1", "P2", "P3")
NATIVE_SHAPE = (3572, 6304)
CPU_ENVIRONMENT = dict(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    if expected is not None and value != expected:
        raise ValueError(f"T4c calibration artifact changed: {path}")
    return dict(path=str(path), sha256=value)


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if getattr(value, "ndim", None) == 0:
        return value.item()
    if getattr(value, "ndim", None) is not None:
        raise ValueError("T4c sweep receipts must not contain scientific arrays or maps")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path, value = Path(path), _safe(value)
    if path.is_file():
        if read(path) != value:
            raise ValueError(f"Existing immutable T4c sweep receipt differs: {path}")
        return
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def delta_key(delta):
    value = float(delta)
    if not math.isfinite(value):
        raise ValueError("T4c delta must be finite")
    return format(value if value else 0., ".12g")


def validate_counts(counts):
    """Keep wrapper metadata outside Claude's exact bare counts layout."""
    if not isinstance(counts, dict) or set(counts) != set(PLACEMENTS):
        raise ValueError("T4c counts must contain exactly P1, P2 and P3")
    for placement, grid in counts.items():
        if not isinstance(grid, dict) or not grid:
            raise ValueError(f"T4c placement lacks grid counts: {placement}")
        numeric = [float(delta) for delta in grid]
        if len(set(numeric)) != len(numeric) or not all(math.isfinite(d) for d in numeric):
            raise ValueError("T4c grid has duplicate or nonfinite numeric deltas")
        for rows in grid.values():
            if not isinstance(rows, dict) or set(rows) != set(RUNS):
                raise ValueError("T4c counts lost the fixed 8-run overflow cohort")
            for frames in rows.values():
                if not isinstance(frames, dict) or set(frames) != {"8", "9"}:
                    raise ValueError("T4c counts require frames 8 and 9 only")
                if any(type(value) is not int or not 0 <= value <= math.prod(NATIVE_SHAPE) for value in frames.values()):
                    raise ValueError("T4c gas counts must be native-grid nonnegative integers")
    return counts


def refinement_grid(calibration, counts, *, maximum_step=.25):
    """Refine the returned full/half crossings without selecting another fit."""
    if not math.isfinite(maximum_step) or maximum_step <= 0:
        raise ValueError("T4c crossing step must be positive and finite")
    pending = {placement: set() for placement in PLACEMENTS}
    for placement in PLACEMENTS:
        result = calibration["variants"][placement]
        grid = sorted(float(value) for value in counts[placement])
        if len(grid) < 2:
            raise ValueError("T4c calibration requires at least two grid points")
        crossings = [result["delta_star"]]
        crossings += [half["delta"] for half in result["split_half"].values()]
        for crossing in crossings:
            if crossing is None:
                continue
            if not math.isfinite(float(crossing)) or not grid[0] <= crossing <= grid[-1]:
                raise ValueError("Frozen calibration returned a crossing outside the input grid")
            left = max(0, min(len(grid) - 2, bisect_left(grid, crossing) - 1))
            low, high = grid[left:left + 2]
            divisions = math.ceil((high - low) / maximum_step)
            if divisions > 1:
                pending[placement].update(delta_key(low + (high - low) * i / divisions)
                    for i in range(1, divisions))
        pending[placement].difference_update(counts[placement])
    return {placement: sorted(values, key=float) for placement, values in pending.items()}


def extension_grid(calibration, counts):
    """Extend missing full or half crossings once on both coarse-grid sides."""
    pending = {placement: [] for placement in PLACEMENTS}
    for placement in PLACEMENTS:
        result = calibration["variants"][placement]
        if result["delta_star"] is not None and all(h["delta"] is not None for h in result["split_half"].values()):
            continue
        low, high = min(map(float, counts[placement])), max(map(float, counts[placement]))
        pending[placement] = [delta_key(value) for value in range(math.floor(low) - 4, math.ceil(high) + 5)
                              if value < low or value > high]
    return pending


def calibration_bindings(script, x_root, project_root, parent):
    """Bind fixed calibration dependencies; never execute a selection scorer."""
    script = Path(script).resolve()
    bindings = dict(script=artifact(script, CALIBRATOR_SHA256), transitive_scripts={name: artifact(script.parent / name)
        for name in ("overflow_gas_compare.py", "overflow_candidate_check.py")})
    x_root, project_root = Path(x_root).resolve(), Path(project_root).resolve()
    expected_parent = x_root / "FF_AC/DarSIA/results/ffac-old-domain-seam-v2-20261005T121928Z/phase-model-20261005T121928Z-g3-old-domain-seam-corrected-ffac"
    if not Path(parent).samefile(expected_parent):
        raise ValueError("Frozen calibration x-root selects another base candidate")
    bindings.update(x_root=str(x_root), project_root=str(project_root),
        frozen_target=artifact(project_root / "ffac_overflow_20261007/results/t4b_overflow/t4b_overflow_result.json", FROZEN_TARGET_SHA256),
        base_area_qa=artifact(x_root / "FF_AC/DarSIA/results/ffac-g3-t2-floor-20261006T100045Z/phase-model-20261006T111405Z-g3-t2-floor-contact-ffac/production_qa/t2_floor_qa.json"),
        overflow_readings={run: artifact(x_root / f"Albus/Autokalibrering_log/overflow_weight_review/{run}/{run}_overflow_scale_review.json") for run in RUNS})
    return bindings


def verify_bindings(bindings):
    def visit(value):
        if isinstance(value, dict):
            if set(value) >= {"path", "sha256"}:
                artifact(value["path"], value["sha256"])
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(bindings)


def validate_config(config):
    """Recheck the compact sweep contract and the shared producer bindings."""
    from .run_ffac_t4c_floor_candidate import validate_config as validate_runtime
    if (config.get("contract") != CONTRACT or config.get("orchestrator_sha256") != sha(__file__)
            or config.get("spec", {}).get("sha256") != SPEC_SHA256
            or config.get("calibration_runs") != list(RUNS)
            or config.get("calibration_frames") != list(FRAMES)
            or config.get("cpu_only") is not True or config.get("threads_per_worker") != 1
            or config.get("coarse_grid") != list(range(-4, 5))
            or config.get("maximum_crossing_step") != .25
            or config.get("human_labels_read") is not False
            or config.get("training_performed") is not False
            or config.get("selection_tests_changed") is not False):
        raise ValueError("T4c sweep configuration changed")
    verify_bindings(config["calibration_bindings"])
    verify_bindings(config["requested_inputs"])
    validate_runtime(config["runtime"])
    if any(config["runtime"]["run_to_epoch"].get(run) != "new_lamps" for run in RUNS):
        raise ValueError("T4c overflow run is not in the declared new-lamp epoch")
    return config


def _worker_initializer():
    os.environ.update(CPU_ENVIRONMENT)
    from .refit_ffac_g3_downstream import below_normal_priority
    import cv2
    below_normal_priority()
    cv2.setNumThreads(1)


def _sweep_run(config_path, config_sha, stage_root, run, grids):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits
    from .run_ffac_t4c_floor_candidate import prepare_evaluation_frame, evaluate_prepared_placement
    artifact(config_path, config_sha)
    config, reports, baseline_cache = validate_config(read(config_path)), [], {}
    for index in FRAMES:
        path = Path(stage_root) / run / f"frame_{index:03d}.json"
        binding = dict(config_sha256=config_sha, run=run, frame_index=index, grids=grids)
        if path.is_file():
            report = read(path)
            if report.get("binding") != binding or report.get("complete") is not True:
                raise ValueError("T4c resumed frame counts belong to another sweep")
            reports.append(artifact(path))
            continue
        start, cpu_start = time.perf_counter(), time.process_time()
        with threadpool_limits(limits=1):
            state = prepare_evaluation_frame(config["runtime"], run, index, baseline_cache=baseline_cache)
            evaluations = {placement: {} for placement in PLACEMENTS}
            for placement in PLACEMENTS:
                for key in grids[placement]:
                    labels, gas_audit, masks, floor = evaluate_prepared_placement(state, placement, float(key))
                    if labels.dtype != np.uint8 or labels.shape != NATIVE_SHAPE:
                        raise ValueError("T4c calibration changed the native categorical grid")
                    evaluations[placement][key] = dict(gas_px=int(np.count_nonzero(labels == 2)),
                        gas_audit=_safe(gas_audit), floor_report=_safe(floor.report))
                    del labels, gas_audit, masks, floor
            source = _safe(state["source"])
            del state
        memory = psutil.Process().memory_info()
        report = dict(contract=CONTRACT, complete=True, binding=binding, source=source,
            evaluations=evaluations, wall_seconds=time.perf_counter() - start,
            cpu_seconds=time.process_time() - cpu_start, rss_bytes=memory.rss,
            peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss),
            threads=1, cpu_only=True, human_labels_read=False, training_performed=False,
            maps_written=False, counts_after_floor=True)
        write_new(path, report)
        reports.append(artifact(path))
    return dict(run=run, frame_receipts=reports)


def sweep_stage(config_path, stage_root, counts, pending, workers):
    """Prepare each source frame once, then discard every transient decision map."""
    import psutil
    from .distribute_ffac_t2_l1zone_candidate import resource_admission
    config_sha = sha(config_path)
    cpus = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), "cpu_affinity") else os.cpu_count() or 1
    admission = resource_admission(workers, len(RUNS), 12288, 2048,
        logical_cpus=cpus, available_bytes=psutil.virtual_memory().available)
    reports = {}
    with ProcessPoolExecutor(max_workers=admission["admitted_workers"],
            mp_context=multiprocessing.get_context("spawn"), initializer=_worker_initializer) as pool:
        futures = {pool.submit(_sweep_run, str(config_path), config_sha, str(stage_root), run, pending): run for run in RUNS}
        for future in as_completed(futures):
            result = future.result()
            reports[result["run"]] = result
            print(json.dumps(dict(event="t4c_calibration_run_complete", stage=str(stage_root), run=result["run"])), flush=True)
    combined = deepcopy(counts)
    for run in RUNS:
        for descriptor in reports[run]["frame_receipts"]:
            artifact(descriptor["path"], descriptor["sha256"])
            frame = read(descriptor["path"])
            for placement, grid in frame["evaluations"].items():
                for delta, value in grid.items():
                    frames = combined[placement].setdefault(delta, {}).setdefault(run, {})
                    key, count = str(frame["binding"]["frame_index"]), value["gas_px"]
                    if key in frames and frames[key] != count:
                        raise ValueError("Existing T4c count differs on the same bound delta/source")
                    frames[key] = count
    return validate_counts(combined), dict(admission=admission, runs=reports)


def run_frozen_calibrator(counts_path, output, bindings):
    verify_bindings(bindings)
    output = Path(output)
    if output.exists():
        result = read(output)
        if result.get("counts_sha256") != sha(counts_path) or result.get("script_sha256") != CALIBRATOR_SHA256:
            raise ValueError("Existing frozen calibration output belongs to other counts/code")
    else:
        subprocess.run([sys.executable, "-B", bindings["script"]["path"], "--counts", str(counts_path),
            "--out", str(output), "--x-root", bindings["x_root"], "--project-root", bindings["project_root"]],
            check=True, env=dict(os.environ, **CPU_ENVIRONMENT))
        result = read(output)
    if (result.get("contract") != "ffac-new-lamp-shift-calibration-v1" or set(result.get("variants", {})) != set(PLACEMENTS)
            or result.get("counts_sha256") != sha(counts_path) or result.get("script_sha256") != CALIBRATOR_SHA256
            or set(result.get("setup", {})) != set(RUNS)
            or any(row["frames"] != list(FRAMES) for row in result["setup"].values())):
        raise ValueError("Frozen calibration changed its fixed inputs, placements or bracketing frames")
    return result


def shift_receipt(placement, calibration, counts_path, script_output, bindings, cost, config_descriptor):
    result = calibration["variants"][placement]
    if result["delta_star"] is None:
        return None
    delta = round(float(result["delta_star"]), 2)
    receipt = dict(contract="ffac-t4c-overflow-calibrated-shift-v1", placement=placement,
        delta_star_unrounded=result["delta_star"], delta=delta, spec_sha256=SPEC_SHA256,
        script=bindings["script"], script_output=artifact(script_output), counts=artifact(counts_path),
        calibration_config=config_descriptor, target_rig_median_seff=calibration["target_rig_median_seff"],
        split_half=result["split_half"], split_half_pass=result["split_half_pass"], grid_deltas=result["deltas"],
        cost=cost, rounding_decimals=2, other_tuning_performed=False, acceptance_tests_changed=False,
        human_labels_read=False, scorer_run=False, training_performed=False)
    if placement in ("P1", "P2"):
        value = math.log(.7 / .3) - delta
        receipt["equivalent_pixel_threshold"] = 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))
        receipt["equivalent_threshold_scope"] = "combined optical-plus-fixed-neighbourhood logit" if placement == "P2" else "base optical logit"
    return receipt


def completed_receipt(path, config_descriptor):
    """Reuse a sealed sweep without changing timings, timestamps or outputs."""
    result = read(path)
    if (result.get("contract") != CONTRACT or result.get("complete") is not True
            or result.get("calibration_config") != config_descriptor
            or result.get("spec_sha256") != SPEC_SHA256
            or result.get("scorer_run") is not False or result.get("maps_written") is not False):
        raise ValueError("Existing completed T4c sweep receipt changed")
    verify_bindings(result)
    counts, calibration = read(result["counts"]["path"]), read(result["script_output"]["path"])
    validate_counts(counts)
    if (calibration.get("contract") != "ffac-new-lamp-shift-calibration-v1"
            or calibration.get("counts_sha256") != result["counts"]["sha256"]
            or calibration.get("script_sha256") != CALIBRATOR_SHA256):
        raise ValueError("Existing completed T4c calibration output changed")
    for placement in PLACEMENTS:
        star = calibration["variants"][placement]["delta_star"]
        if star is None:
            if placement not in result["dropped_placements_no_crossing"] or placement in result["placements"]:
                raise ValueError("Existing completed T4c missing-crossing outcome changed")
        else:
            receipt = read(result["placements"][placement]["path"])
            if (receipt.get("placement") != placement or receipt.get("delta_star_unrounded") != star
                    or receipt.get("delta") != round(float(star), 2)):
                raise ValueError("Existing completed T4c rounded shift changed")
    return result


def measured_cost(stages, orchestration_wall_seconds):
    frames = [read(frame["path"]) for stage in stages for run in stage["runs"].values()
              for frame in run["frame_receipts"]]
    return dict(orchestration_wall_seconds=orchestration_wall_seconds, stages=stages,
        worker_cpu_seconds=sum(frame["cpu_seconds"] for frame in frames),
        worker_frame_wall_seconds_sum=sum(frame["wall_seconds"] for frame in frames),
        worker_peak_working_set_bytes=max((frame["peak_working_set_bytes"] for frame in frames), default=0),
        evaluated_frame_stages=len(frames), placement_delta_evaluations=sum(
            len(grid) for frame in frames for grid in frame["evaluations"].values()),
        worker_timing_source="per-frame count receipts contain CPU/wall/peak working set", cpu_only=True)


def calibrate(args):
    os.environ.update(CPU_ENVIRONMENT)
    from .run_ffac_t4c_floor_candidate import prepare_calibration_config
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bindings = calibration_bindings(args.calibration_script, args.x_root, args.project_root, args.parent)
    requested_inputs = dict(parent=str(args.parent.resolve()), t4b=str(args.t4b.resolve()),
        spec=artifact(args.spec, SPEC_SHA256), facies_map=artifact(args.facies_map), zone_map=artifact(args.zone_map))
    config_path = output / "calibration_config.json"
    if config_path.is_file():
        config = read(config_path)
        if (config.get("orchestrator_sha256") != sha(__file__) or config.get("calibration_bindings") != bindings
                or config.get("workers_requested") != args.workers or config.get("requested_inputs") != requested_inputs):
            raise ValueError("Existing T4c sweep config differs from code, roots or owner resources")
    else:
        runtime = prepare_calibration_config(args.parent, args.t4b, args.spec, args.facies_map, args.zone_map)
        if any(runtime["run_to_epoch"].get(run) != "new_lamps" for run in RUNS):
            raise ValueError("T4c overflow run is not in the declared new-lamp epoch")
        config = dict(contract=CONTRACT, runtime=runtime, calibration_bindings=bindings, requested_inputs=requested_inputs,
            orchestrator_sha256=sha(__file__), spec=artifact(args.spec, SPEC_SHA256),
            superseded_spec_sha256_not_used=SUPERSEDED_SPEC_SHA256, workers_requested=args.workers,
            calibration_runs=list(RUNS), calibration_frames=list(FRAMES), cpu_only=True, threads_per_worker=1,
            coarse_grid=list(range(-4, 5)), maximum_crossing_step=.25, extension_policy="once four logits at each missing-crossing grid boundary",
            selection_tests_changed=False, human_labels_read=False, training_performed=False)
        write_new(config_path, config)
    validate_config(config)
    config_descriptor = artifact(config_path)
    final_path = output / "calibration_receipt.json"
    if final_path.is_file():
        return completed_receipt(final_path, config_descriptor)
    started = time.perf_counter()
    counts = {placement: {} for placement in PLACEMENTS}
    pending = {placement: [delta_key(d) for d in range(-4, 5)] for placement in PLACEMENTS}
    stages, extended, calibration = [], False, None
    for number in range(8):
        stage_root = output / f"stage_{number:03d}"
        counts, stage = sweep_stage(config_path, stage_root / "frame_counts", counts, pending, args.workers)
        counts_path, script_output = stage_root / "counts.json", stage_root / "script_output.json"
        write_new(counts_path, counts)
        calibration = run_frozen_calibrator(counts_path, script_output, bindings)
        stages.append(dict(counts=artifact(counts_path), script_output=artifact(script_output), **stage))
        pending = refinement_grid(calibration, counts)
        if not any(pending.values()) and not extended:
            pending = extension_grid(calibration, counts)
            extended = any(pending.values())
        if not any(pending.values()):
            break
    else:
        raise ValueError("T4c crossing refinement did not finish within the fixed stage bound")
    cost = measured_cost(stages, time.perf_counter() - started)
    existing_shifts = [read(output / f"calibrated_shift_{placement}.json") for placement in PLACEMENTS
                       if (output / f"calibrated_shift_{placement}.json").is_file()]
    if existing_shifts:
        previous = existing_shifts[0]["cost"]
        if any(row["cost"] != previous for row in existing_shifts) or previous["stages"] != stages:
            raise ValueError("Partially completed T4c shifts have another bound sweep cost")
        cost = previous
    placements, dropped = {}, []
    for placement in PLACEMENTS:
        receipt = shift_receipt(placement, calibration, counts_path, script_output, bindings, cost, config_descriptor)
        if receipt is None:
            dropped.append(placement)
        else:
            path = output / f"calibrated_shift_{placement}.json"
            write_new(path, receipt)
            placements[placement] = artifact(path)
    result = dict(contract=CONTRACT, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        spec_sha256=SPEC_SHA256, calibration_config=config_descriptor, placements=placements,
        dropped_placements_no_crossing=dropped, counts=artifact(counts_path), script_output=artifact(script_output),
        script=bindings["script"], cost=cost, calibration_only=True, selection_tests_changed=False,
        training_performed=False, human_labels_read=False, scorer_run=False, maps_written=False)
    write_new(final_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "t4b", "spec", "facies-map", "zone-map", "calibration-script", "x-root", "project-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("T4c calibration requires one to eight bounded single-thread CPU workers")
    result = calibrate(args)
    print(json.dumps(dict(complete=result["complete"], placements=result["placements"],
        dropped_placements=result["dropped_placements_no_crossing"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

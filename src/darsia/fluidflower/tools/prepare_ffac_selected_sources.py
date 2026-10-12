"""Plan and prepare selected camera sources with the frozen P1-final3 producers.

This controller prepares scientific RAW assets only. It does not infer phase
categories, train, alter source assets, or register a Studio candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from darsia.fluidflower.phase_ffac_selected_phase import asset, bound_json, frozen_parent_chain, SPEC_SHA256
from darsia.fluidflower.phase_ffac_selected_sources import SelectedRunSourcePreparer, selected_source_inventory
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, canonical_sha256, read_json, utc_now

CONTRACT = "ffac-selected-source-preparation-plan-v1"


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Existing selected-source metadata has another binding: " + str(path))
    else:
        atomic_json(path, value)
    return asset(path)


def copied_asset(source, destination):
    source_path = Path(source["path"])
    if sha256_file(source_path) != source["sha256"]:
        raise ValueError("Frozen source changed before portable copy")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != source["sha256"]:
            raise ValueError("Portable selected-source copy has another SHA")
    else:
        shutil.copy2(source_path, destination)
    return asset(destination)


def indexed_inventory(manifest, parent_rows, producer_rows):
    """Bind exact unscaled producer area for every original acquisition."""
    inventory = selected_source_inventory(manifest, parent_rows)
    producers = {}
    for index, row in enumerate(inventory["sources"]):
        row["work_index"] = index
        if row.get("studio_anchor"):
            key = f"{manifest['run']}/frame_{row['studio_frame_index']:03d}"
            if key not in producer_rows:
                raise ValueError("Original unscaled area producer is missing: " + key)
            producer = deepcopy(producer_rows[key])
            original = bound_json(producer)
            producer_sha = original.get("source_sha256") or original.get("binding", {}).get("source", {}).get("source_sha256")
            if producer_sha != row["source_sha256"]:
                raise ValueError("Original area receipt identifies another acquisition")
            producers[row["source_sha256"]] = producer
    inventory["source_previous"] = {row["source_sha256"]: row["previous_source_sha256"]
                                    for row in inventory["sources"]}
    return inventory, producers


def build_plan(selection_path, parent_path, spec_path, output):
    selection_asset = asset(selection_path)
    selection = bound_json(selection_asset)
    if not selection.get("complete") or selection.get("phase_inference"):
        raise ValueError("Selected-camera metadata must be complete and contain no inferred maps")
    spec_asset = asset(spec_path)
    if spec_asset["sha256"] != SPEC_SHA256:
        raise ValueError("Selected-source preparation requires the approved updated specification")
    chain = frozen_parent_chain(parent_path)
    print(json.dumps(dict(event="frozen_parent_verified", parent=str(parent_path))), flush=True)
    parent_seal = bound_json(chain["parent_seal"])
    seam_asset = asset(Path(chain["base_candidate"]) / "old_domain_seam_plan.json")
    seam = bound_json(seam_asset)
    print(json.dumps(dict(event="original_producer_plan_verified", plan=seam_asset)), flush=True)
    producer_rows = seam["source_producer_frames"]
    parent_rows = {run: [] for run in selection["runs"]}
    for row in parent_seal["frames"]:
        if row["run"] in parent_rows:
            parent_rows[row["run"]].append(row["receipt"])
    reference = bound_json(next(row["receipt"] for row in parent_seal["frames"]
                                if row["run"] == "ac28" and row["frame_index"] == 0))
    output = Path(output)
    prep = output / "source_preparation"
    immutable_json(prep / "verified_parent_chain.json", chain)
    top_source = reference["binding"]["source"]["rig_geometry"]["frame_photo_top_keep"]["curve_source"]
    top_asset = copied_asset(top_source, prep / "inputs" / "frozen_photo_top_curve.npy")
    top_copy = immutable_json(prep / "inputs" / "top_curve_copy.json",
        dict(contract="ffac-exact-portable-source-copy-v1", source=top_source, copied=top_asset,
             numerical_changes=False, bytes_identical=True))
    probability_asset = reference["binding"]["source"]["retained_source_cache"]["parent_probability_receipt"]
    probability = bound_json(probability_asset)
    t4c = bound_json(chain["t4c_plan"])
    light = t4c["evaluation_config"]
    barrier_frame = reference["binding"]["source"]["retained_source_cache"]["parent_frame_receipt"]
    barrier_root = Path(barrier_frame["path"]).parents[2] / "gas_barrier_rule_b"
    barrier_reference = dict(geometry_manifest=asset(barrier_root / "source_geometry_manifest.json"),
                             components=asset(barrier_root / "reference_components.npz"))
    physical_scale = reference["rig_domain_revision"]["parameters"]["physical_scale_revision"]
    files = {"controller": asset(__file__), "source_preparer": asset(Path(__file__).parents[1] / "phase_ffac_selected_sources.py"),
             "native_centre": asset(Path(__file__).parents[1] / "phase_ffac_selected_native_centre.py"),
             "source_geometry": asset(Path(__file__).parents[1] / "phase_ffac_camera_source_geometry.py")}
    binding = dict(contract=CONTRACT, approved_spec=spec_asset, selection=selection_asset,
        parent_chain=chain, source_producer_plan=seam_asset, code=files,
        frozen_photo_top_curve=top_asset, top_copy_receipt=top_copy,
        physical_scale_revision=physical_scale, frozen_barrier_reference=barrier_reference,
        history_policy="baseline plus selected acquisitions only; unselected anchors are probes",
        source_preparation_only=True, training=False, phase_inference=False)
    job_sha = canonical_sha256(binding)
    states = {}
    totals = dict(selected=0, anchors=0, sources=0, new_sources=0, baselines=0)
    for run in selection["runs"]:
        manifest_asset = selection["manifests"][run]
        manifest = bound_json(manifest_asset)
        inventory, producers = indexed_inventory(manifest, parent_rows[run], producer_rows)
        inventory.update(contract="ffac-selected-source-inventory-v1", job_sha256=job_sha,
                         source_manifest=manifest_asset, original_producer_receipts=producers)
        inventory_asset = immutable_json(prep / "inventories" / f"{run}.json", inventory)
        baseline = next(row for row in inventory["sources"] if row.get("baseline"))
        states[run] = dict(source_inventory=inventory_asset,
            parent_native_manifest=manifest["parent_native_manifest"],
            reference_parent_receipt=baseline["parent_receipt"],
            original_producer_receipts=producers, source_manifest=manifest_asset)
        print(json.dumps(dict(event="source_inventory_ready", run=run, sources=len(inventory["sources"]))), flush=True)
        totals["selected"] += len(inventory["selected_sources"])
        totals["anchors"] += len(inventory["studio_anchor_sources"])
        totals["sources"] += len(inventory["sources"])
        totals["new_sources"] += sum(not row.get("studio_anchor") for row in inventory["sources"])
        totals["baselines"] += 1
    plan = dict(contract=CONTRACT, binding=binding, job_sha256=job_sha, runs=states,
        parent_chain=chain, phase_grid=reference["baseline_extended_scientific"]["binding"]["phase_grid"],
        frozen_top_curve=top_asset, optical_candidate=probability["binding"]["frozen_candidate"],
        physical_scale_revision=physical_scale, frozen_barrier_reference=barrier_reference,
        heads=chain["heads"], light_epochs=dict(run_to_epoch=light["run_to_epoch"],
            receipt=light["binding"]["light_epoch_table"]), totals=totals,
        source_preparation_only=True, training=False, phase_inference=False, original_assets_changed=False)
    descriptor = immutable_json(prep / "plan.json", plan)
    return plan, descriptor


def selected_work(inventory, source_shas=None, pilot=False):
    rows = inventory["sources"]
    if pilot:
        if inventory["run"] != "ac28":
            return []
        wanted = {inventory["baseline_source_sha256"]}
        wanted.add(min((row for row in rows if row.get("selected")),
                       key=lambda row: abs(float(row["elapsed_h"]) - 2.125))["source_sha256"])
        wanted.add(next(row["source_sha256"] for row in rows
                        if row.get("studio_frame_index") == 8))
        rows = [row for row in rows if row["source_sha256"] in wanted]
    if source_shas:
        wanted = set(source_shas)
        rows = [row for row in rows if row["source_sha256"] in wanted]
        if {row["source_sha256"] for row in rows} != wanted:
            raise ValueError("Requested selected source is absent from this run")
    return rows


def prepare_run(root, plan_asset, run, source_shas=None, pilot=False):
    # Every subprocess is bounded to one library thread independently of host size.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    plan = bound_json(plan_asset)
    inventory = bound_json(plan["runs"][run]["source_inventory"])
    rows = selected_work(inventory, source_shas, pilot)
    started = time.perf_counter()
    result = dict(run=run, plan=plan_asset, requested_sources=len(rows), receipts=[], failures=[],
                  phase_inference=False, training=False, threads=1)
    with threadpool_limits(limits=1):
        preparer = SelectedRunSourcePreparer(root, plan, run)
        for row in rows:
            try:
                receipt = preparer.prepare(row)
                path = Path(root) / "selected_sources" / run / row["source_sha256"] / "receipt.json"
                result["receipts"].append(dict(source_sha256=row["source_sha256"],
                    elapsed_h=row["elapsed_h"], selected=row.get("selected", False),
                    studio_anchor=row.get("studio_anchor", False), receipt=asset(path),
                    reused=receipt["source_audit"]["reused_exact_scientific_source"]))
                print(json.dumps(dict(event="source_prepared", run=run, source=row["source_sha256"],
                                      elapsed_h=row["elapsed_h"], reused=result["receipts"][-1]["reused"])), flush=True)
            except Exception as exc:
                failure = dict(source_sha256=row["source_sha256"], elapsed_h=row["elapsed_h"],
                               error=str(exc), traceback=traceback.format_exc())
                result["failures"].append(failure)
                print(json.dumps(dict(event="source_failed", run=run, **failure)), flush=True)
                break
    result.update(complete=not result["failures"] and len(result["receipts"]) == len(rows),
                  seconds=time.perf_counter()-started)
    return result


def execute(root, plan_descriptor, runs, *, workers=1, source_shas=None, pilot=False, report=None):
    plan = bound_json(plan_descriptor)
    if any(run not in plan["runs"] for run in runs):
        raise ValueError("Requested run is outside the locked selected-image universe")
    if source_shas and len(runs) != 1:
        raise ValueError("Explicit source filter requires exactly one run")
    report = Path(report) if report else Path(root) / "source_preparation" / ("pilot_result.json" if pilot else "preparation_result.json")
    if report.exists():
        raise FileExistsError("Preparation reports are immutable; use a new report path")
    results = []
    if workers == 1:
        results = [prepare_run(root, plan_descriptor, run, source_shas, pilot) for run in runs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(prepare_run, root, plan_descriptor, run, source_shas, pilot): run for run in runs}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(dict(run=futures[future], complete=False, receipts=[], failures=[
                        dict(error=str(exc), traceback=traceback.format_exc())]))
    result = dict(contract="ffac-selected-source-preparation-result-v1", created_utc=utc_now(),
        plan=plan_descriptor, complete=all(row["complete"] for row in results), workers=workers,
        threads_per_worker=1, runs=results, phase_inference=False, training=False,
        original_assets_changed=False, source_count=sum(len(row["receipts"]) for row in results))
    immutable_json(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--source-sha", action="append")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.plan:
        descriptor = asset(args.plan)
        plan = bound_json(descriptor)
    else:
        if not all((args.selection, args.parent, args.spec)):
            parser.error("Planning requires --selection, --parent and --spec")
        plan, descriptor = build_plan(args.selection, args.parent, args.spec, args.output)
    print(json.dumps(dict(event="plan_ready", plan=descriptor, totals=plan["totals"])), flush=True)
    if args.execute:
        runs = args.runs or (["ac28"] if args.pilot else list(plan["runs"]))
        result = execute(args.output, descriptor, runs, workers=args.workers,
                         source_shas=args.source_sha, pilot=args.pilot, report=args.report)
        if not result["complete"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

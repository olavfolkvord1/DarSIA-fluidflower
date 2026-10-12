"""CPU Policy A inference admitted only by the all-run exact-convention proof.

This is a new operational producer. It inherits the frozen P1-final3 science,
replaces only the superseded coarse-QA stopping decision, and never reads human
annotations or a scorer. Unselected Studio anchors remain history probes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time

from darsia.fluidflower.phase_ffac_selected_policy_a_qa import (
    SPEC_SHA256, RUNS, EXPECTED_ANCHORS, EvidenceReader, asset,
    coarse_studio_gate, load_prediction, require_identity_gate,
)
from darsia.fluidflower.tools.refit_ffac_g3_downstream import (
    atomic_json, canonical_sha256, utc_now, worker_initializer,
)
from darsia.fluidflower.tools import run_ffac_selected_cpu_identity_exact as identity_driver

CONTRACT = "ffac-selected-policy-a-cpu-campaign-v1"
EXACT_HELPER = "darsia.fluidflower.phase_ffac_selected_cpu_identity_exact"
MODULES = tuple(dict.fromkeys(identity_driver.MODULES + (
    "darsia.fluidflower.phase_ffac_selected_policy_a_qa",
    "darsia.fluidflower.tools.prepare_ffac_selected_anchor_aliases",
    "darsia.fluidflower.phase_ffac_selected_source_alias_validation",
    "darsia.fluidflower.tools.report_ffac_selected_policy_a",
    "darsia.fluidflower.tools.export_ffac_selected_policy_a_labels",
    "darsia.fluidflower.tools.run_ffac_selected_policy_a_cpu",
    "darsia.fluidflower.tools.finish_ffac_selected_policy_a_cpu",
)))
_STATE = {}


def code_assets():
    return {name: {"sha256": asset(importlib.import_module(name).__file__)["sha256"]}
            for name in MODULES}


def require_both_identity(descriptor, reader, *, expected_runs=RUNS):
    proof = require_identity_gate(descriptor, reader, expected_runs=expected_runs)
    gate = reader.record(descriptor)
    if gate.get("stage") != "all39" or gate.get("mode") != "both":
        raise ValueError("Policy A requires the combined exact-convention all39 stage")
    identity_plan = reader.record(gate["identity_plan"])
    binding = identity_plan["phase_campaign_binding"]
    conventions = binding.get("exact_conventions", {})
    if (binding.get("spec_sha256") != SPEC_SHA256
            or conventions.get("both_previous_native_alignment_mask", "missing") is not None
            or conventions.get("decoder_and_feature_support_masks") != "unchanged"
            or conventions.get("extended_previous_alignment_mask") != "unchanged"
            or EXACT_HELPER not in binding.get("code", {})):
        raise ValueError("Identity proof does not bind both exact numerical conventions")
    compared = {}
    required = {"g3_pixel.npy", "regional_frozen.npy", "regional_refit.npy",
                "co2.packbits.npy", "extra_masks.packbits.npz"}
    for item in proof["compared_row_receipts"]:
        row = reader.record(item)
        if set(row["stored_probabilities"]["comparisons"]) != required:
            raise ValueError("Identity proof omits a literal stored probability layer")
        indices = compared.setdefault(row["run"], set())
        index = int(row["original_studio_frame_index"])
        if index in indices:
            raise ValueError("Identity proof duplicates a Studio frame")
        indices.add(index)
    if any(compared.get(run) != {0, 1} for run in expected_runs):
        raise ValueError("All-run proof must contain actual own-history frames 0 and 1")
    return dict(proof, mode="both", code=binding["code"],
                source_preparation=binding["source_preparation"])


def select_rows(inventory):
    """Unique acquisitions; probes never replace the selected predecessor."""
    rows = sorted((r for r in inventory["sources"]
                   if r.get("selected") or r.get("baseline") or r.get("studio_anchor")),
                  key=lambda r: (float(r["elapsed_h"]), r["source_sha256"]))
    seen, selected, baselines = set(), [], []
    for row in rows:
        digest = row["source_sha256"]
        if digest in seen:
            raise ValueError("A source acquisition occurs twice in the run inventory")
        seen.add(digest)
        if row.get("baseline"):
            baselines.append(digest)
        expected = None if row.get("baseline") else (
            selected[-1]["source_sha256"] if selected else None)
        if row.get("previous_source_sha256") != expected:
            raise ValueError("Policy A predecessor differs from chronological selected history")
        if row.get("selected") or row.get("baseline"):
            if selected and float(selected[-1]["elapsed_h"]) >= float(row["elapsed_h"]):
                raise ValueError("Selected acquisitions are not strictly ordered in camera time")
            selected.append(row)
    if baselines != [inventory["baseline_source_sha256"]]:
        raise ValueError("Run must have exactly one bound baseline acquisition")
    return rows


def verify_source_aliases(plan, source_descriptor, alias_descriptor, reader):
    """Recompose literal producer metadata; no scientific array is read."""
    from darsia.fluidflower.tools.prepare_ffac_selected_anchor_aliases import alias_payload
    proof = reader.record(alias_descriptor)
    if (proof.get("contract") != "ffac-selected-original-anchor-metadata-alias-inventory-v1"
            or proof.get("complete") is not True or proof.get("source_plan") != source_descriptor
            or proof.get("original_scientific_arrays_modified") is not False
            or proof.get("scientific_arrays_read_or_recomputed") is not False):
        raise ValueError("Original-source metadata alias proof is incomplete or belongs to another plan")
    module = importlib.import_module("darsia.fluidflower.tools.prepare_ffac_selected_anchor_aliases")
    if asset(module.__file__)["sha256"] != proof["tool"]["sha256"]:
        raise ValueError("Original-source metadata alias producer changed")
    inventories = {run: reader.record(state["source_inventory"])
                   for run, state in plan["runs"].items()}
    sources = {run: {r["source_sha256"]: r for r in state["sources"]}
               for run, state in inventories.items()}
    fields = ("native_scientific", "prepared_source", "full_extended_scientific",
              "baseline_extended_scientific", "geometry_frame", "rig_geometry")
    checked = set()
    for row in proof["aliases"]:
        run, digest = row["run"], row["source_sha256"]
        key = (run, digest)
        if key in checked:
            raise ValueError("Metadata alias proof duplicates an acquisition")
        checked.add(key)
        source, state = sources[run][digest], plan["runs"][run]
        if source["parent_receipt"] != row["original_parent_receipt"]:
            raise ValueError("Metadata alias substitutes the historical parent producer")
        parent, prepared = reader.record(row["original_parent_receipt"]), reader.record(row["original_prepared_receipt"])
        baseline = reader.record(state["reference_parent_receipt"])["baseline_extended_scientific"]
        expected = alias_payload(plan, run, state, source, parent, prepared, baseline,
            tool=proof["tool"], original_prepared_asset=row["original_prepared_receipt"])
        saved = reader.record(row["alias_receipt"])
        producer_fields = {name: saved[name] for name in fields}
        if (saved.get("complete") is not True or saved.get("binding") != expected["binding"]
                or saved.get("run") != run or saved.get("source_sha256") != digest
                or saved.get("elapsed_h") != source["elapsed_h"]
                or any(saved[name] != expected[name] for name in fields)
                or canonical_sha256(producer_fields) != row["scientific_fields_sha256"]
                or saved.get("source_audit", {}).get("scientific_arrays_generated") is not False):
            raise ValueError("Metadata alias changed original scientific descriptor fields")
    if len(checked) != proof.get("new_metadata_aliases"):
        raise ValueError("Metadata alias proof has an incorrect completed count")
    imported = 0
    for run, items in plan.get("source_imports", {}).items():
        for digest, descriptor in items.items():
            saved = reader.record(descriptor)
            source = sources[run][digest]
            if (saved.get("complete") is not True or saved.get("run") != run
                    or saved.get("source_sha256") != digest or saved.get("elapsed_h") != source["elapsed_h"]
                    or saved.get("binding", {}).get("source") != source
                    or saved["binding"].get("frozen_parent") != plan["parent_chain"]["parent_seal"]):
                raise ValueError("Imported scientific wrapper changed source or frozen-parent binding")
            imported += 1
    return dict(alias_proof=alias_descriptor, verified_aliases=len(checked),
                verified_imports=imported, scientific_arrays_read=False)


def bind(source_plan_path, identity_gate_path, destination, *, metadata_audit_path,
         code_audit_path, scientific_source_root=None, source_alias_audit_path=None):
    reader = EvidenceReader()
    source_descriptor, identity_descriptor = asset(source_plan_path), asset(identity_gate_path)
    source = reader.record(source_descriptor)
    if set(source.get("runs", {})) != set(RUNS):
        raise ValueError("Policy A requires the exact selected 39-run cohort")
    spec = source.get("binding", {}).get("approved_spec", {})
    if spec.get("sha256") != SPEC_SHA256:
        raise ValueError("Policy A source selection must bind the revised approved specification")
    reader.path(spec)
    identity = require_both_identity(identity_descriptor, reader)
    if identity["source_preparation"] != source_descriptor:
        raise ValueError("Own-history identity used another source-preparation plan")
    codes = code_assets()
    for name, expected in identity["code"].items():
        if codes.get(name) != expected:
            raise ValueError("Policy A numerical producer differs from the accepted identity: " + name)
    metadata_descriptor, audit_descriptor = asset(metadata_audit_path), asset(code_audit_path)
    metadata, audit = reader.record(metadata_descriptor), reader.record(audit_descriptor)
    if (metadata.get("scientific_pixels_read") is not False
            or metadata.get("model_executed") is not False
            or {r["run"] for r in metadata.get("rows", [])} != set(RUNS)
            or audit.get("read_only") is not True or audit.get("model_run") is not False):
        raise ValueError("Numerical metadata and code-consumer audits must cover all 39 sources")
    audited_helper = [r for r in audit["reviewed_modules"]
                      if Path(r["path"]).name == "phase_ffac_selected_cpu_identity_exact.py"]
    if len(audited_helper) != 1 or audited_helper[0]["sha256"] != codes[EXACT_HELPER]["sha256"]:
        raise ValueError("Numerical audit describes another exact-convention helper")
    alias_descriptor = asset(source_alias_audit_path or
        Path(source_plan_path).parent/"original_anchor_aliases"/"receipt.json")
    aliases = verify_source_aliases(source, source_descriptor, alias_descriptor, reader)
    counts = {"studio_anchors": 0, "selected_frames": 0, "baseline_frames": 0}
    for run, state in source["runs"].items():
        inventory = reader.record(state["source_inventory"])
        if inventory.get("run") != run:
            raise ValueError("Source inventory belongs to another run")
        for row in select_rows(inventory):
            counts["studio_anchors"] += int(bool(row.get("studio_anchor")))
            counts["selected_frames"] += int(bool(row.get("selected") and not row.get("baseline")))
            counts["baseline_frames"] += int(bool(row.get("baseline")))
    if counts["studio_anchors"] != EXPECTED_ANCHORS or counts["baseline_frames"] != len(RUNS):
        raise ValueError("Source plan does not cover the original Studio anchors and run baselines")
    binding = dict(contract=CONTRACT, spec_sha256=SPEC_SHA256,
        source_preparation=source_descriptor, identity_gate=identity_descriptor,
        identity_plan=identity["identity_plan"], code=codes, backend="cpu", internal_threads=1,
        exact_conventions_mode="both", numerical_metadata_audit=metadata_descriptor,
        numerical_code_consumer_audit=audit_descriptor,
        original_source_alias_audit=aliases,
        history_policy="baseline and selected acquisitions only; unselected Studio anchors are probes",
        coarse_studio_gate=dict(co2_relative_limit=.01, gas_relative_limit=.02,
            absolute_floor_cm2=.5, combination="max(relative_limit * parent_area, absolute_floor_cm2)"),
        old_strict_gate_decides_acceptance=False, shared_stop_required=True,
        expected_counts=counts, training=False, annotations_read=False, scorer_pending=True,
        round3_allowed=False, mass_computed=False, default_changed=False)
    plan = dict(source, phase_campaign_binding=binding,
                pipeline_sha256=canonical_sha256(binding), identity_gate=identity_descriptor,
                scientific_source_root=str(scientific_source_root or Path(source_plan_path).parent.parent))
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Policy A plan must be a new immutable file")
    atomic_json(destination, plan)
    return asset(destination)


def verify_code(plan):
    binding = plan["phase_campaign_binding"]
    if (binding.get("contract") != CONTRACT or binding.get("spec_sha256") != SPEC_SHA256
            or binding.get("backend") != "cpu" or binding.get("exact_conventions_mode") != "both"
            or canonical_sha256(binding) != plan["pipeline_sha256"]):
        raise ValueError("Policy A binding or CPU convention changed")
    for name, expected in binding["code"].items():
        if asset(importlib.import_module(name).__file__)["sha256"] != expected["sha256"]:
            raise ValueError("Policy A immutable producer changed: " + name)


def shared_stop(root):
    return Path(root)/"operational"/"policy_a_stop.json"


def write_stop(root, record):
    """First failure wins atomically on the shared NAS, across both hosts."""
    path = shared_stop(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(dict(contract=CONTRACT, stopped_utc=utc_now(), host=socket.gethostname(),
                           pid=os.getpid(), **record), stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        pass
    return path


def initialise(plan_descriptor, root):
    worker_initializer(1)
    reader = EvidenceReader()
    plan = reader.record(plan_descriptor)
    verify_code(plan)
    proof = require_both_identity(plan["identity_gate"], reader)
    if proof["identity_plan"] != plan["phase_campaign_binding"]["identity_plan"]:
        raise ValueError("Worker identity proof differs from its immutable plan")
    _STATE.update(plan=plan, plan_descriptor=plan_descriptor, root=Path(root), reader=reader)


def verify_ready_source(descriptor, source, plan, reader):
    """Current producer, declared byte-identical import, or literal alias only."""
    saved = reader.record(descriptor)
    binding = saved.get("binding", {})
    if (saved.get("contract") != "ffac-selected-frozen-scientific-source-v1"
            or saved.get("complete") is not True or saved.get("run") != source["run"]
            or saved.get("source_sha256") != source["source_sha256"]
            or saved.get("elapsed_h") != source["elapsed_h"]
            or binding.get("source") != source
            or binding.get("frozen_parent") != plan["parent_chain"]["parent_seal"]
            or binding.get("source_inventory") != plan["runs"][source["run"]]["source_inventory"]):
        raise ValueError("Ready scientific source changed its source/time/parent/inventory binding")
    imported = plan.get("source_imports", {}).get(source["run"], {}).get(source["source_sha256"])
    if imported is not None:
        if descriptor["sha256"] != imported["sha256"]:
            raise ValueError("Declared imported wrapper is not byte-identical to its producer")
        reader.path(imported)
    elif binding.get("job_sha256") != plan["job_sha256"]:
        raise ValueError("Scientific source is neither the current producer nor an exact declared import")
    if saved.get("source_audit", {}).get("metadata_only_original_source_alias"):
        alias_proof = reader.record(plan["phase_campaign_binding"]["original_source_alias_audit"]["alias_proof"])
        aliases = [r for r in alias_proof["aliases"] if r["run"] == source["run"]
                   and r["source_sha256"] == source["source_sha256"]]
        if len(aliases) != 1 or descriptor["sha256"] != aliases[0]["alias_receipt"]["sha256"]:
            raise ValueError("Original scientific metadata alias lacks its exact producer proof")
    return saved


def wait_ready_sources(root, plan, inventory, digests, reader, *, timeout=21600., poll=1.):
    """Wait for atomic source receipts while preparation runs in parallel.

    Only metadata is verified here. Scientific arrays are checked by the
    unchanged loader when used, avoiding a second full RAW SHA pass.
    """
    by_digest = {r["source_sha256"]: r for r in inventory["sources"]}
    required = list(dict.fromkeys(d for d in digests if d is not None))
    source_root = Path(plan.get("scientific_source_root", root))
    deadline, announced = time.monotonic()+timeout, -1e99
    while True:
        if shared_stop(root).exists():
            return False
        missing = []
        for digest in required:
            source = by_digest[digest]
            path = source_root/"selected_sources"/source["run"]/digest/"receipt.json"
            if not path.is_file():
                missing.append(digest); continue
            verify_ready_source(asset(path), source, plan, reader)
        if not missing:
            return True
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("Source preparation wait timed out for " + ", ".join(missing))
        if now-announced >= 60:
            print(json.dumps(dict(event="waiting_for_scientific_sources", missing=missing,
                remaining_seconds=max(0., deadline-now), host=socket.gethostname())), flush=True)
            announced = now
        time.sleep(min(poll, deadline-now))


def validate_resume(record, source, plan):
    proof = record.get("qa", {}).get("policy_a_cpu", {})
    coarse = record.get("qa", {}).get("policy_a_coarse_studio")
    if (proof.get("mode") != "both" or proof.get("identity_gate") != plan["identity_gate"]
            or proof.get("previous_source_sha256") != source.get("previous_source_sha256")
            or proof.get("exact_helper_sha256") != plan["phase_campaign_binding"]["code"][EXACT_HELPER]["sha256"]
            or proof.get("coarse_studio_gate_checked") != bool(source.get("studio_anchor"))
            or source.get("studio_anchor") and (not coarse or coarse.get("all_pass") is not True)):
        raise ValueError("Completed frame lacks identical source/history/CPU/coarse-QA evidence")


def decorate_result(result, source, plan, reader):
    exact = result["native_inference"].get("exact_conventions", {})
    if (exact.get("mode") != "both" or exact.get("previous_native_alignment_mask") != "absent"
            or exact.get("decoder_validity_mask") != "unchanged"
            or exact.get("feature_support_mask") != "unchanged"
            or exact.get("extended_previous_alignment_mask") != "unchanged"):
        raise ValueError("Inference did not use the accepted combined exact-convention helper")
    coarse = None
    if source.get("studio_anchor"):
        parent_descriptor = asset(Path(plan["parent_chain"]["parent"])/"rig_camera_frames"/
            source["run"]/f"frame_{int(source['studio_frame_index']):03d}.json")
        parent = reader.record(parent_descriptor)
        if (parent.get("run") != source["run"] or parent.get("source_sha256") != source["source_sha256"]
                or parent.get("elapsed_h") != source["elapsed_h"]):
            raise ValueError("Coarse comparison parent is another camera acquisition")
        before = load_prediction(parent["labels_map"], reader)
        coarse = dict(coarse_studio_gate(before, result["labels"], result["pixel_area_mm2"]),
            parent_receipt=parent_descriptor, run=source["run"],
            studio_frame_index=int(source["studio_frame_index"]),
            source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"])
    result["qa"]["policy_a_coarse_studio"] = coarse
    result["qa"]["superseded_strict_studio_replay_diagnostic"] = result["qa"].pop("strict_studio_replay", None)
    result["qa"]["policy_a_cpu"] = dict(mode="both", identity_gate=plan["identity_gate"],
        exact_helper_sha256=plan["phase_campaign_binding"]["code"][EXACT_HELPER]["sha256"],
        previous_source_sha256=source.get("previous_source_sha256"),
        coarse_studio_gate_checked=bool(source.get("studio_anchor")),
        exact_conventions=exact, old_strict_gate_decides_acceptance=False)
    return coarse


def run_one(run):
    from darsia.fluidflower.phase_ffac_selected_cpu_identity_exact import ExactSelectedPhaseEvaluator
    root, plan, reader = _STATE["root"], _STATE["plan"], _STATE["reader"]
    lock = root/"operational"/"run_claims"/(run+".json")
    lock.parent.mkdir(parents=True, exist_ok=True)
    claim = dict(plan=_STATE["plan_descriptor"], run=run, host=socket.gethostname(), pid=os.getpid())
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(claim, stream)
    receipts, started = [], time.perf_counter()
    try:
        if shared_stop(root).exists():
            return dict(run=run, complete=False, stopped="shared stop", receipts=receipts)
        inventory = reader.record(plan["runs"][run]["source_inventory"])
        timeout = _STATE.get("source_wait_seconds", 21600.)
        if not wait_ready_sources(root, plan, inventory, [inventory["baseline_source_sha256"]], reader,
                                  timeout=timeout):
            return dict(run=run, complete=False, stopped="shared stop awaiting baseline", receipts=receipts)
        evaluator = ExactSelectedPhaseEvaluator(root, plan, run, backend="cpu")
        for source in select_rows(evaluator.inventory):
            if shared_stop(root).exists():
                return dict(run=run, complete=False, stopped="shared stop", receipts=receipts)
            digest = source["source_sha256"]
            existing = evaluator.saved(digest)
            if existing is not None:
                saved = reader.record(existing)
                validate_resume(saved, source, plan)
                result_asset = existing
            else:
                if not wait_ready_sources(root, plan, inventory,
                        [digest, source.get("previous_source_sha256")], reader, timeout=timeout):
                    return dict(run=run, complete=False, stopped="shared stop awaiting source", receipts=receipts)
                result = evaluator.evaluate(digest)
                coarse = decorate_result(result, source, plan, reader)
                if coarse is not None and not coarse["all_pass"]:
                    write_stop(root, dict(reason="Policy A coarse Studio gate failed", run=run,
                        source_sha256=digest, coarse_gate=coarse, plan=_STATE["plan_descriptor"]))
                    result_asset = evaluator.save(result)
                    receipts.append(result_asset)
                    return dict(run=run, complete=False, stopped="coarse gate", receipts=receipts,
                                failed_gate=coarse, seconds=time.perf_counter()-started)
                if shared_stop(root).exists():
                    return dict(run=run, complete=False, stopped="shared stop after inference", receipts=receipts)
                result_asset = evaluator.save(result)
                saved = reader.record(result_asset)
            receipts.append(result_asset)
            print(json.dumps(dict(event="frame_complete", run=run, source_sha256=digest,
                elapsed_h=source["elapsed_h"], selected=source.get("selected"),
                studio_anchor=source.get("studio_anchor"), resumed=existing is not None,
                seconds=saved["qa"]["seconds"])), flush=True)
        return dict(run=run, complete=True, receipts=receipts, seconds=time.perf_counter()-started)
    except Exception as error:
        write_stop(root, dict(reason="Policy A operational failure", run=run, error=repr(error),
                             plan=_STATE["plan_descriptor"]))
        raise
    finally:
        # Remove only this process's claim; a killed process leaves evidence for
        # a deliberate operational recovery, rather than a blind duplicate run.
        if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) == claim:
            lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan"); parser.add_argument("--identity-gate")
    parser.add_argument("--metadata-audit"); parser.add_argument("--code-audit")
    parser.add_argument("--source-alias-audit")
    parser.add_argument("--bind-output"); parser.add_argument("--source-root")
    parser.add_argument("--plan"); parser.add_argument("--root", required=True)
    parser.add_argument("--runs", nargs="+"); parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-wait-seconds", type=float, default=21600.)
    parser.add_argument("--stage-receipt")
    args = parser.parse_args(argv)
    if args.bind_output:
        required = (args.source_plan, args.identity_gate, args.metadata_audit, args.code_audit)
        if not all(required):
            raise ValueError("Bind requires source plan, all39 identity and both numerical audits")
        print(json.dumps(bind(args.source_plan, args.identity_gate, args.bind_output,
            metadata_audit_path=args.metadata_audit, code_audit_path=args.code_audit,
            scientific_source_root=args.source_root, source_alias_audit_path=args.source_alias_audit)))
        return 0
    if not args.plan or not args.stage_receipt or args.workers < 1 or args.source_wait_seconds <= 0:
        raise ValueError("Inference requires an immutable plan, new stage receipt and bounded workers")
    descriptor = asset(args.plan)
    initialise(descriptor, args.root)
    _STATE["source_wait_seconds"] = args.source_wait_seconds
    if shared_stop(args.root).exists():
        raise ValueError("Shared stop exists; do not blindly resume the failed campaign")
    runs = args.runs or sorted(_STATE["plan"]["runs"])
    if len(set(runs)) != len(runs) or not runs or any(run not in RUNS for run in runs):
        raise ValueError("Explicit run shard escapes the immutable cohort")
    destination = Path(args.stage_receipt)
    if destination.exists():
        raise FileExistsError("Stage receipt must be a new file")
    started, results, errors = utc_now(), [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(runs)),
            mp_context=multiprocessing.get_context("spawn"), initializer=initialise,
            initargs=(descriptor, args.root)) as pool:
        tasks = {pool.submit(run_with_wait, run, args.source_wait_seconds): run for run in runs}
        for task in as_completed(tasks):
            try:
                results.append(task.result())
            except Exception as error:
                errors.append(dict(run=tasks[task], error=repr(error)))
                write_stop(args.root, dict(reason="Policy A worker failed", run=tasks[task],
                           error=repr(error), plan=descriptor))
            if shared_stop(args.root).exists():
                for pending in tasks:
                    pending.cancel()
    complete = (not errors and len(results) == len(runs) and all(row["complete"] for row in results)
                and not shared_stop(args.root).exists())
    atomic_json(destination, dict(contract=CONTRACT, complete=complete, plan=descriptor,
        pipeline_sha256=_STATE["plan"]["pipeline_sha256"], host=socket.gethostname(), pid=os.getpid(),
        started_utc=started, finished_utc=utc_now(), workers=min(args.workers, len(runs)),
        internal_threads=1, backend="cpu", mode="both", runs=runs, results=results, errors=errors,
        source_wait_seconds=args.source_wait_seconds,
        identity_gate=_STATE["plan"]["identity_gate"], shared_stop=shared_stop(args.root).exists(),
        training=False, annotations_read=False, scorer_pending=True, round3_allowed=False,
        mass_computed=False, default_changed=False))
    return 0 if complete else 2


def run_with_wait(run, timeout):
    _STATE["source_wait_seconds"] = timeout
    return run_one(run)


if __name__ == "__main__":
    raise SystemExit(main())

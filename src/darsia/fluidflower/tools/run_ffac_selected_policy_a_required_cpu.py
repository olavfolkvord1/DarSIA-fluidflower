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

from darsia.fluidflower.phase_ffac_selected_policy_a_required_qa import (
    SPEC_SHA256, RUNS, EXPECTED_ANCHORS, EvidenceReader, asset,
    coarse_studio_gate, load_prediction, require_identity_gate,
)
from darsia.fluidflower.tools.refit_ffac_g3_downstream import (
    atomic_json, canonical_sha256, utc_now, worker_initializer,
)
from darsia.fluidflower.tools import run_ffac_selected_required_identity as identity_driver

CONTRACT = "ffac-selected-policy-a-required-cpu-campaign-v2"
EXACT_HELPER = "darsia.fluidflower.phase_ffac_selected_policy_a_required_inference"
MODULES = tuple(dict.fromkeys(identity_driver.MODULES + (
    "darsia.fluidflower.phase_ffac_selected_policy_a_required_qa",
    "darsia.fluidflower.phase_ffac_selected_policy_a_required_inference",
    "darsia.fluidflower.tools.prepare_ffac_selected_required_anchor_aliases",
    "darsia.fluidflower.tools.prepare_ffac_selected_anchor_aliases",
    "darsia.fluidflower.phase_ffac_selected_source_alias_validation",
    "darsia.fluidflower.phase_ffac_common_mass_round2_backend",
    "darsia.fluidflower.tools.refit_ffac_g3_downstream",
    "darsia.fluidflower.tools.report_ffac_selected_policy_a_required",
    "darsia.fluidflower.tools.export_ffac_selected_policy_a_required_labels",
    "darsia.fluidflower.tools.run_ffac_selected_policy_a_required_cpu",
    "darsia.fluidflower.tools.finish_ffac_selected_policy_a_required_cpu",
)))
_STATE = {}


def code_assets():
    return {name: {"sha256": asset(importlib.import_module(name).__file__)["sha256"]}
            for name in MODULES}


def require_both_identity(descriptor, reader, *, expected_runs=RUNS):
    # Kept as an operational API name for existing completion callers. The
    # accepted convention is now independently bound for each historical stage.
    return require_identity_gate(descriptor, reader, expected_runs=expected_runs)


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
    """Current aliases are recomposed; imports retain byte-exact producer lineage."""
    from darsia.fluidflower.phase_ffac_selected_source_alias_validation import (
        AliasMetadataVerifier, verify_original_anchor_alias, verify_scientific_alias_fields)
    proof = reader.record(alias_descriptor)
    equivalent = asset(Path(alias_descriptor["path"]).parent/"tool_portable_equivalence.json")
    validation = asset(Path(alias_descriptor["path"]).parent/"validation.json")
    checked_validation = reader.record(validation)
    if (proof.get("complete") is not True or proof.get("source_plan") != source_descriptor
            or proof.get("scientific_arrays_read_or_recomputed") is not False
            or checked_validation.get("complete") is not True
            or checked_validation.get("source_plan") != source_descriptor
            or checked_validation.get("alias_inventory") != alias_descriptor
            or checked_validation.get("tool_equivalence") != equivalent
            or checked_validation.get("scientific_arrays_generated") is not False
            or checked_validation.get("old_artifacts_modified") is not False):
        raise ValueError("Required-source alias producer validation is incomplete")
    verifier = AliasMetadataVerifier()
    inventories = {run: reader.record(state["source_inventory"]) for run,state in plan["runs"].items()}
    sources = {run: {row["source_sha256"]:row for row in inv["sources"]} for run,inv in inventories.items()}
    checked=set()
    for row in proof["aliases"]:
        run,digest=row["run"],row["source_sha256"]
        if (run,digest) in checked:raise ValueError("Current alias acquisition duplicated")
        checked.add((run,digest))
        verify_original_anchor_alias(row["alias_receipt"],sources[run][digest],plan,run,
            alias_inventory=alias_descriptor,tool_equivalence=equivalent,verifier=verifier)
    imported_proof = reader.record(plan["source_import_proof"])
    if (imported_proof.get("complete") is not True
            or imported_proof.get("byte_exact_receipt_copies") is not True
            or imported_proof.get("producer_code_and_source_binding_preserved") is not True
            or imported_proof.get("arrays_copied_or_recomputed") is not False):
        raise ValueError("Scientific imports lack immutable producer equivalence")
    imports={}
    for row in imported_proof["imports"]:
        run,digest=row["run"],row["source_sha256"]
        if (run,digest) in imports:raise ValueError("Imported source duplicated")
        staged=row["staged_receipt"]
        saved=reader.record(staged)
        producer=reader.record(row["producer_receipt"])
        if (staged != plan["source_imports"][run][digest]
                or staged["sha256"] != row["producer_receipt"]["sha256"]
                or saved != producer or saved.get("binding") != row["producer_binding"]
                or saved["binding"].get("source") != sources[run][digest]
                or saved["binding"].get("frozen_parent") != plan["parent_chain"]["parent_seal"]):
            raise ValueError("Imported source changed its original producer binding")
        if saved.get("source_audit",{}).get("metadata_only_original_source_alias"):
            lineage=saved["metadata_alias_provenance"]
            if lineage["original_parent_receipt"] != sources[run][digest]["parent_receipt"]:
                raise ValueError("Imported alias substitutes original parent metadata")
            verify_scientific_alias_fields(saved,reader.record(lineage["original_parent_receipt"]),
                reader.record(lineage["original_prepared_receipt"]),
                reader.record(plan["runs"][run]["reference_parent_receipt"])["baseline_extended_scientific"])
        imports[(run,digest)]=row
    expected={(run,digest) for run,items in plan.get("source_imports",{}).items() for digest in items}
    if set(imports) != expected or len(checked) != proof["new_metadata_aliases"]:
        raise ValueError("Source alias/import validation omits an acquisition")
    return dict(alias_proof=alias_descriptor,tool_equivalence=equivalent,
        combined_validation=validation,import_proof=plan["source_import_proof"],
        verified_aliases=len(checked),verified_imports=len(imports),scientific_arrays_read=False)


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
            or audit_descriptor != identity["frozen_gas_audit"]
            or audit.get("required_helper_binding_pending") is not False):
        raise ValueError("Source metadata and final helper consumer audits must be exact")
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
        exact_conventions_mode="historical-stage-conventions", numerical_metadata_audit=metadata_descriptor,
        numerical_code_consumer_audit=audit_descriptor,
        original_source_alias_audit=aliases,
        frozen_gas_audit=identity["frozen_gas_audit"],
        inherited_stage_conventions=reader.record(identity["identity_plan"])["phase_campaign_binding"]["inherited_stage_conventions"],
        required_identity_stages=identity["required_identity_stages"],
        shared_stop_path=str(Path(scientific_source_root or Path(source_plan_path).parent.parent)/"operational"/"policy_a_stop.json"),
        readiness_policy="all chronological sources ready before heavy evaluator allocation",
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
            or binding.get("backend") != "cpu" or binding.get("exact_conventions_mode") != "historical-stage-conventions"
            or canonical_sha256(binding) != plan["pipeline_sha256"]):
        raise ValueError("Policy A binding or CPU convention changed")
    for name, expected in binding["code"].items():
        if asset(importlib.import_module(name).__file__)["sha256"] != expected["sha256"]:
            raise ValueError("Policy A immutable producer changed: " + name)


def shared_stop(root, plan=None):
    if plan is None and _STATE.get("root") == Path(root):
        plan = _STATE.get("plan")
    bound = (plan or {}).get("phase_campaign_binding", {}).get("shared_stop_path")
    return Path(bound) if bound else Path(root)/"operational"/"policy_a_stop.json"


def write_stop(root, record):
    """First failure wins atomically on the shared NAS, across both hosts."""
    path = shared_stop(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(dict(contract=CONTRACT, stopped_utc=utc_now(), host=socket.gethostname(),
                           pid=os.getpid(), stop=True, stop_new_work=True, production_allowed=False,
                           code_binding=_STATE.get("plan",{}).get("phase_campaign_binding",{}).get("code"),
                           source_binding=_STATE.get("plan",{}).get("phase_campaign_binding",{}).get("source_preparation"),
                           **record), stream, indent=2)
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
    """Verify current producer or exact imported producer before reading pixels."""
    saved = reader.record(descriptor)
    binding = saved.get("binding", {})
    if (saved.get("contract") != "ffac-selected-frozen-scientific-source-v1"
            or saved.get("complete") is not True or saved.get("run") != source["run"]
            or saved.get("source_sha256") != source["source_sha256"]
            or saved.get("elapsed_h") != source["elapsed_h"]
            or binding.get("source") != source
            or binding.get("frozen_parent") != plan["parent_chain"]["parent_seal"]
            or binding.get("source_inventory") != plan["runs"][source["run"]]["source_inventory"]):
        raise ValueError("Ready scientific source changed source/time/parent/inventory binding")
    evidence=plan["phase_campaign_binding"]["original_source_alias_audit"]
    imported=plan.get("source_imports",{}).get(source["run"],{}).get(source["source_sha256"])
    if imported is not None:
        proof=reader.record(evidence["import_proof"])
        matches=[row for row in proof["imports"] if row["run"]==source["run"]
                 and row["source_sha256"]==source["source_sha256"]]
        if (len(matches)!=1 or descriptor != imported
                or matches[0]["staged_receipt"] != imported
                or matches[0]["producer_receipt"]["sha256"] != imported["sha256"]
                or matches[0]["producer_binding"] != binding):
            raise ValueError("Declared imported wrapper is not its byte-exact producer")
        reader.path(matches[0]["producer_receipt"])
        # In particular, an imported metadata alias retains its old producer's
        # job and tool. It must never be relabelled as a new current-plan alias.
    else:
        if binding.get("job_sha256") != plan["job_sha256"]:
            raise ValueError("Scientific source is neither current nor explicitly imported")
        if saved.get("source_audit",{}).get("metadata_only_original_source_alias"):
            from darsia.fluidflower.phase_ffac_selected_source_alias_validation import (
                AliasMetadataVerifier, verify_original_anchor_alias)
            verifier = getattr(reader,"alias_verifier",None)
            if verifier is None:
                verifier=reader.alias_verifier=AliasMetadataVerifier()
            verify_original_anchor_alias(descriptor,source,plan,source["run"],
                alias_inventory=evidence["alias_proof"],tool_equivalence=evidence["tool_equivalence"],verifier=verifier)
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
    if (proof.get("mode") != "historical-stage-conventions" or proof.get("identity_gate") != plan["identity_gate"]
            or proof.get("previous_source_sha256") != source.get("previous_source_sha256")
            or proof.get("exact_helper_sha256") != plan["phase_campaign_binding"]["code"][EXACT_HELPER]["sha256"]
            or proof.get("coarse_studio_gate_checked") != bool(source.get("studio_anchor"))
            or source.get("studio_anchor") and (not coarse or coarse.get("all_pass") is not True)):
        raise ValueError("Completed frame lacks identical source/history/CPU/coarse-QA evidence")


def decorate_result(result, source, plan, reader):
    exact = result["native_inference"].get("historical_stage_conventions", {})
    if (exact.get("native_previous_alignment") != "historical absent explicit validity mask"
            or exact.get("inherited_native_baseline_alignment") != "historical absent explicit validity mask"
            or exact.get("g3_native_baseline_alignment") != "historical explicit source-valid mask"
            or exact.get("extended_previous_alignment") != "retain bound mask"
            or exact.get("decoder_and_feature_support_masks") != "unchanged"
            or exact.get("g3_probability",{}).get("fallback_probability_injected_into_inference") is not False
            or exact.get("independent_alignment_audit",{}).get("baseline_rgb_bit_identity",{}).get("bit_identical") is not True
            or exact.get("independent_alignment_audit",{}).get("previous_rgb_bit_identical") is not True):
        raise ValueError("Inference did not use independently verified historical stage conventions")
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
    result["qa"]["policy_a_cpu"] = dict(mode="historical-stage-conventions", identity_gate=plan["identity_gate"],
        exact_helper_sha256=plan["phase_campaign_binding"]["code"][EXACT_HELPER]["sha256"],
        previous_source_sha256=source.get("previous_source_sha256"),
        coarse_studio_gate_checked=bool(source.get("studio_anchor")),
        exact_conventions=exact, old_strict_gate_decides_acceptance=False)
    return coarse


def run_one(run):
    from darsia.fluidflower.phase_ffac_selected_policy_a_required_inference import RequiredSelectedPhaseEvaluator
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
        # Readiness waiting retains only metadata. No multi-gigabyte reference
        # context is allocated until every chronological source in this run is
        # ready, so independent source preparation can use the available RAM.
        if not wait_ready_sources(root, plan, inventory,
                                  [source["source_sha256"] for source in select_rows(inventory)], reader,
                                  timeout=timeout):
            return dict(run=run, complete=False, stopped="shared stop awaiting run sources", receipts=receipts)
        evaluator = RequiredSelectedPhaseEvaluator(root, plan, run, backend="cpu")
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
        internal_threads=1, backend="cpu", mode="historical-stage-conventions", runs=runs, results=results, errors=errors,
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

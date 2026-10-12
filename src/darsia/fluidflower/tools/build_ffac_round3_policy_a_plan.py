"""Prepare an immutable R3 runtime, then admit actual Policy A maps/Claude PASS.

The archive can be prepared before phase completion. ``seal`` cannot create
an accepted input plan until the actual full review and matching PASS exist.
This tool never reads scientific rasters, labels or sealed scorer code.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from darsia.fluidflower import phase_ffac_common_mass_inputs as source
from darsia.fluidflower import phase_ffac_round3_handoff_admission as admission
from darsia.fluidflower.tools import run_ffac_common_mass_round3_selected as runner
from darsia.fluidflower.tools import build_ffac_round3_selected_plan as base
from darsia.fluidflower.tools import ffac_round3_selected_adapter as adapter


def prepare_archive(parts_descriptor, destination):
    """Copy source bytes and original freeze bytes only; no accepted map plan."""
    reader = source.VerifiedAssetReader()
    parts = reader.json(parts_descriptor)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    package = Path(source.__file__).resolve().parents[1]
    copied = []
    for path in sorted(package.rglob("*.py")):
        target = destination / "runtime" / "src" / "darsia" / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        copied.append(source.asset(target))
    inventory = runner.code_inventory() | {
        "darsia.fluidflower.tools.ffac_round3_selected_adapter": source.asset(adapter.__file__)}
    rows = []
    for name, descriptor in inventory.items():
        original = Path(descriptor["path"]).resolve()
        target = destination / "runtime" / "src" / "darsia" / original.relative_to(package)
        saved = source.asset(target)
        if saved["sha256"] != descriptor["sha256"]:
            raise ValueError("Round 3 source changed during archival: " + name)
        rows.append(dict(module=name, original_source=descriptor, archived_source=saved,
            exact_source_equivalent=True))
    freezes = {}
    for name in ("r1_freeze", "r2_freeze"):
        descriptor = parts[name]
        target = destination / "original_freezes" / (name + ".json")
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(reader.path(descriptor).read_bytes())
        saved = source.asset(target)
        if saved["sha256"] != descriptor["sha256"]:
            raise ValueError("Original frozen calibration changed")
        freezes[name] = dict(original_source=descriptor, archived_source=saved, bytes_unchanged=True)
    producer = source.asset(__file__)
    producer_copy = source.asset(destination / "runtime" / "src" / "darsia"
        / Path(__file__).resolve().relative_to(package))
    if producer_copy["sha256"] != producer["sha256"]:
        raise ValueError("The handoff preparation bridge changed while archiving")
    value = dict(contract="ffac-round3-selected-loader-source-equivalence-v1", complete=True,
        code=rows, original_freezes=freezes, runtime_files=copied,
        preparation_code=dict(original_source=producer, archived_source=producer_copy,
            exact_source_equivalent=True),
        runtime_source_root=str(destination/"runtime"/"src"),
        saved_parts_plan=parts_descriptor, accepted_selected_plan_created=False,
        scientific_pixels_read=False, fits_started=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    source.write_new_json(destination/"archive.json", value)
    return source.asset(destination/"archive.json")


def seal_plan(*, phase_plan_descriptor, review_gate_descriptor, export_descriptor,
              claude_status_descriptor, parts_descriptor, archive_descriptor,
              candidate, output, reader=None):
    reader = reader or source.VerifiedAssetReader()
    review = reader.json(review_gate_descriptor)
    phase_plan = reader.json(phase_plan_descriptor)
    export = reader.json(export_descriptor)
    status = reader.json(claude_status_descriptor)
    scoring_root = Path(export_descriptor["path"]).parent
    check = admission.verify_label_pass(status, export_descriptor, scoring_root)
    result_descriptor = dict(path=check.get("result_path"), sha256=check.get("result_sha256"))
    reader.path(result_descriptor)
    if review.get("phase_plan") != phase_plan_descriptor:
        raise ValueError("Review completion belongs to another physical/source operation")
    gate = dict(contract=admission.CONTRACT, complete=True, all_39_complete=True,
        round3_allowed=True, claude_pass_verified=True, runs=list(sorted(runner.ALL_RUNS)),
        spec_sha256=runner.SPEC_SHA256, policy_a_review_gate=review_gate_descriptor,
        studio_export=export_descriptor, claude_status_snapshot=claude_status_descriptor,
        claude_result=result_descriptor, candidate_root=str(scoring_root),
        phase_candidate_root=str(candidate),
        phase_plan=phase_plan_descriptor, run_manifests=review.get("run_manifests"),
        production_automatically_allowed=False, legacy_strict_gate_claimed=False)
    admission.validate_policy_a_admission(gate, reader, runner.ALL_RUNS)
    entries = []
    identities = set()
    for row in review["entries"]:
        phase = reader.json(row["receipt"])
        science = reader.json(row["scientific_receipt"])
        binding = row["source_binding"]
        if (phase.get("complete") is not True or science.get("complete") is not True
                or phase.get("binding",{}).get("source") != binding
                or science.get("binding",{}).get("source") != binding
                or phase.get("source_sha256") != row["source_sha256"]
                or phase.get("elapsed_h") != row["elapsed_h"]
                or phase.get("frame_index") != row["frame_index"]
                or phase.get("full_extended_scientific") != science.get("full_extended_scientific")
                or phase.get("baseline_extended_scientific") != science.get("baseline_extended_scientific")
                or phase.get("bottom_mass_allowed") is not False):
            raise ValueError("Selected map/current/reference source binding changed")
        if len(row.get("target_hours",())) != 1:
            raise ValueError("The current R3 runner needs one target per unique acquisition; no duplicate mass evidence")
        entry = {name:row[name] for name in ("run","frame_index","elapsed_h","source_binding",
            "receipt","scientific_receipt","run_manifest")}
        entry["selected_target_h"] = float(row["target_hours"][0])
        identity = (row["run"],row["source_sha256"])
        if identity in identities:
            raise ValueError("Selected acquisition is repeated")
        identities.add(identity)
        manifest = reader.json(row["run_manifest"])
        adapter._target_match(entry, manifest)
        entries.append(entry)
    if len(entries) != 1564 or any(not any(row["run"]==run and row["selected_target_h"]==2.25
            for row in entries) for run in runner.ALL_RUNS):
        raise ValueError("Complete selected coverage and each explicit 2.25 h anchor are required")
    parts, adapter_descriptor, dependencies, folds, parameters = base.input_dependencies(
        parts_descriptor,archive_descriptor,reader)
    dependencies.update(policy_a_review=review_gate_descriptor, studio_export=export_descriptor,
        claude_result=result_descriptor, selected_phase_plan=phase_plan_descriptor)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve Claude's status bytes in our folder; never write to Claude.
    with (output/"claude_status_snapshot.json").open("xb") as stream:
        stream.write(reader.path(claude_status_descriptor).read_bytes())
    gate["claude_status_snapshot"] = source.asset(output/"claude_status_snapshot.json")
    gate["original_claude_status"] = claude_status_descriptor
    source.write_new_json(output/"accepted_round3_admission.json",gate)
    index = dict(contract=runner.INDEX_CONTRACT,complete=True,entries=entries,
        all_sources_are_selected=True,unique_acquisitions=len(identities),target_count=len(entries),
        repeated_acquisitions=[],phase_plan=phase_plan_descriptor,dense_scientific_arrays_read=False)
    source.write_new_json(output/"selected_metadata_index.json",index)
    plan = dict(contract=runner.PLAN_CONTRACT,spec=parts["spec"],spec_sha256=runner.SPEC_SHA256,
        calibration_runs=list(runner.engine.FIT_RUNS),heldout_runs=list(runner.engine.HELDOUT_RUNS),
        descriptive_runs=list(runner.engine.DESCRIPTIVE_RUNS),saved_parts_plan=parts_descriptor,
        selected_phase_gate=source.asset(output/"accepted_round3_admission.json"),
        selected_metadata_index=source.asset(output/"selected_metadata_index.json"),
        selected_adapter=dict(contract=runner.ADAPTER_CONTRACT,source=adapter_descriptor,function="load_selected"),
        additional_dependencies=dependencies,round2_loro_fits=folds,
        map_mass_rescaling=False,overflow_in_objective=False,production_automatically_allowed=False,
        selected_phase_candidate=str(candidate),scientific_source_root=phase_plan["scientific_source_root"])
    source.write_new_json(output/"selected_input_plan.json",plan)
    receipt = dict(complete=True,plan=source.asset(output/"selected_input_plan.json"),
        admission=plan["selected_phase_gate"],claude_status_snapshot=gate["claude_status_snapshot"],
        metadata_only=True,scientific_pixels_read=False,fits_started=False,
        production_automatically_allowed=False,original_round2_loro_parameters=parameters)
    source.write_new_json(output/"preparation_receipt.json",receipt)
    return source.asset(output/"preparation_receipt.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("archive","seal"))
    for name in ("phase-plan","review-gate","export-manifest","claude-status","source-archive","candidate"):
        parser.add_argument("--"+name)
    parser.add_argument("--saved-parts-plan",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.stage=="archive":
        result=prepare_archive(source.asset(args.saved_parts_plan),args.output)
    else:
        if not all((args.phase_plan,args.review_gate,args.export_manifest,args.claude_status,args.source_archive,args.candidate)):
            parser.error("seal requires completed Policy A review/export plus matching Claude status")
        result=seal_plan(phase_plan_descriptor=source.asset(args.phase_plan),
            review_gate_descriptor=source.asset(args.review_gate),export_descriptor=source.asset(args.export_manifest),
            claude_status_descriptor=source.asset(args.claude_status),parts_descriptor=source.asset(args.saved_parts_plan),
            archive_descriptor=source.asset(args.source_archive),candidate=args.candidate,output=args.output)
    print(json.dumps(result))


if __name__=="__main__":main()

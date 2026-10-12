"""Read-only Policy A completion, spatial report and Claude prediction export.

These artifacts are for review and sealed scoring. They never admit round3 or
claim that the prior strict 0.1% scientific gate passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from darsia.fluidflower.phase_ffac_selected_policy_a_qa import (
    EvidenceReader, RUNS, asset, verify_phase_source,
)
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, utc_now
from darsia.fluidflower.tools import run_ffac_selected_policy_a_cpu as producer

CONTRACT = "ffac-selected-policy-a-review-completion-v1"


def admission(plan_path, root, identity_gate_path):
    reader = EvidenceReader()
    descriptor = asset(plan_path)
    plan = reader.record(descriptor)
    producer.verify_code(plan)
    identity = asset(identity_gate_path)
    if identity != plan["identity_gate"]:
        raise ValueError("Review identity proof is not the bound all-run combined-convention proof")
    proof = producer.require_both_identity(identity, reader)
    if proof["identity_plan"] != plan["phase_campaign_binding"]["identity_plan"]:
        raise ValueError("Review refers to another accepted CPU numerical producer")
    if producer.shared_stop(root).exists():
        raise ValueError("Stopped Policy A campaign cannot be exported as a complete review candidate")
    return reader, descriptor, plan


def collect(plan_path, root, identity_gate_path, destination):
    reader, descriptor, plan = admission(plan_path, root, identity_gate_path)
    entries, missing = [], []
    counts = dict(selected_frames=0, baseline_frames=0, studio_anchors=0)
    runs = {}
    for run, state in sorted(plan["runs"].items()):
        inventory = reader.record(state["source_inventory"])
        run_entries, run_missing = [], []
        for source in producer.select_rows(inventory):
            path = Path(root)/"selected_phase_frames"/run/(source["source_sha256"]+".json")
            if not path.is_file():
                row = dict(run=run, source_sha256=source["source_sha256"],
                           work_index=source["work_index"], elapsed_h=source["elapsed_h"])
                missing.append(row); run_missing.append(row)
                continue
            phase_descriptor = asset(path)
            phase = reader.record(phase_descriptor)
            verify_phase_source(phase, source, plan["pipeline_sha256"])
            producer.validate_resume(phase, source, plan)
            counts["selected_frames"] += int(bool(source.get("selected") and not source.get("baseline")))
            counts["baseline_frames"] += int(bool(source.get("baseline")))
            counts["studio_anchors"] += int(bool(source.get("studio_anchor")))
            row = dict(run=run, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
                work_index=source["work_index"], previous_source_sha256=source.get("previous_source_sha256"),
                selected=bool(source.get("selected")), baseline=bool(source.get("baseline")),
                studio_anchor=bool(source.get("studio_anchor")),
                studio_frame_index=source.get("studio_frame_index"), phase_receipt=phase_descriptor,
                labels_map=phase["labels_map"], source_inventory=state["source_inventory"],
                coarse_studio_gate=phase["qa"]["policy_a_coarse_studio"], phase_area=phase["qa"]["phase_area"])
            entries.append(row); run_entries.append(row)
        runs[run] = dict(complete=not run_missing, frames=len(run_entries), missing=run_missing)
    expected = plan["phase_campaign_binding"]["expected_counts"]
    complete = not missing and counts == expected and set(runs) == set(RUNS)
    record = dict(contract=CONTRACT, complete=complete, created_utc=utc_now(),
        all_39_phase_complete=complete, plan=descriptor, identity_gate=plan["identity_gate"],
        pipeline_sha256=plan["pipeline_sha256"], runs=runs, counts=counts, expected_counts=expected,
        entries=entries, missing=missing, cpu_only=True, mode="both", coarse_area_gate_passed=complete,
        spatial_report_pending=True, claude_scorer_gate_pending=True, round3_allowed=False,
        mass_admission=False, annotations_read=False, scorer_executed=False,
        bottom_mass_allowed=False, default_changed=False, registered=False,
        old_strict_gate_decides_acceptance=False, code=asset(__file__))
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Review completion receipt must be a new file")
    atomic_json(destination, record)
    return asset(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect", "report", "export"), required=True)
    for name in ("plan", "root", "identity-gate", "destination"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--report"); parser.add_argument("--injector-reference")
    args = parser.parse_args(argv)
    if args.stage == "collect":
        result = collect(args.plan, args.root, args.identity_gate, args.destination)
    else:
        # Strong combined-convention admission also protects the unchanged v1
        # metadata/spatial-report and byte-copy export implementations.
        admission(args.plan, args.root, args.identity_gate)
        if args.stage == "report":
            from darsia.fluidflower.tools.report_ffac_selected_policy_a import build
            result = build(args.plan, args.root, args.identity_gate, args.destination,
                           injector_reference=args.injector_reference, backend="cpu")
        else:
            if not args.report:
                raise ValueError("Export requires the complete coarse spatial report")
            from darsia.fluidflower.tools.export_ffac_selected_policy_a_labels import build
            result = build(args.plan, args.root, args.identity_gate, args.report, args.destination)
    print(json.dumps(result))
    saved = EvidenceReader().record(result)
    return 0 if saved.get("complete") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())

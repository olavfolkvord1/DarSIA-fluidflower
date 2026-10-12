"""Export categorical predictions for Claude in exact P1-final3 Studio layout.

Only saved prediction PNGs and provenance certificates are read. No annotation
or sealed scorer is accessed, and this export does not admit round3 fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from darsia.fluidflower import phase_ffac_selected_policy_a_baseline_qa as qa
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, utc_now

CONTRACT = "ffac-selected-policy-a-required-claude-export-v1"


def accepted_report(descriptor, context, reader):
    report = reader.record(descriptor)
    if (report.get("contract") != "ffac-selected-policy-a-required-saved-report-v1"
            or report.get("complete") is not True or report.get("all_pass") is not True
            or report.get("studio_frames") != qa.EXPECTED_ANCHORS
            or report.get("plan") != context["plan_descriptor"]
            or report.get("identity_gate") != context["identity"]["identity_gate"]
            or report.get("pipeline_sha256") != context["plan"]["pipeline_sha256"]
            or report.get("spec_sha256") != qa.SPEC_SHA256):
        raise ValueError("Scorer export requires the complete source-bound Policy A coarse report")
    rows = {}
    for item in report["frame_receipts"]:
        row = reader.record(item)
        key = (row["run"], row["original_studio_frame_index"])
        if (key in rows or key not in context["anchors"] or row.get("complete") is not True
                or row.get("comparison", {}).get("all_pass") is not True
                or row.get("source_sha256") != context["anchors"][key]["source_sha256"]):
            raise ValueError("Policy A coarse report has a duplicate, failed or wrong-source frame")
        rows[key] = row
    if set(rows) != set(context["anchors"]):
        raise ValueError("Coarse report does not cover every original Studio source")
    return rows


def build(plan_path, root, identity_gate, report_path, destination):
    reader = qa.EvidenceReader()
    context = qa.policy_context(plan_path, root, qa.asset(identity_gate), reader)
    report_descriptor = qa.asset(report_path)
    comparisons = accepted_report(report_descriptor, context, reader)
    # All metadata admission precedes any output. No previous export is replaced.
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    exports = []
    for key in sorted(context["anchors"]):
        source, phase_descriptor, phase, parent_descriptor, parent, before, current, area = qa.anchor_evidence(context, key, reader)
        row = comparisons[key]
        if (row["phase_receipt"] != phase_descriptor or row["parent_receipt"] != parent_descriptor
                or row["current_labels"] != phase["labels_map"] or row["parent_labels"] != parent["labels_map"]
                or qa.coarse_studio_gate(before, current, area) != row["comparison"]):
            raise ValueError("Predicted map or coarse QA changed between report and scorer export")
        run, studio_index = key
        output = destination/"production_labels"/run/f"frame_{studio_index:03d}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        with reader.path(phase["labels_map"]).open("rb") as old, output.open("xb") as new:
            shutil.copyfileobj(old, new, length=8 << 20)
        exported = qa.asset(output)
        if exported["sha256"] != phase["labels_map"]["sha256"]:
            raise ValueError("Copied prediction PNG is not byte-identical to Policy A producer")
        exports.append(dict(run=run, original_studio_frame_index=studio_index,
            source_work_index=source["work_index"], source_sha256=source["source_sha256"],
            elapsed_h=source["elapsed_h"], original_studio_elapsed_h=parent["elapsed_h"],
            phase_receipt=phase_descriptor, parent_receipt=parent_descriptor,
            source_inventory=context["plan"]["runs"][run]["source_inventory"],
            exported_prediction=exported, current_prediction=phase["labels_map"],
            parent_prediction=parent["labels_map"], shape_hw=list(current.shape), encoding="uint8 L PNG: 0,1,2,255"))
    record = dict(contract=CONTRACT, complete=True, created_utc=utc_now(),
        spec_sha256=qa.SPEC_SHA256, plan=context["plan_descriptor"],
        pipeline_sha256=context["plan"]["pipeline_sha256"], identity_gate=context["identity"]["identity_gate"],
        coarse_report=report_descriptor, parent_seal=context["plan"]["parent_chain"]["parent_seal"],
        required_identity_stages=context["identity"].get("required_identity_stages"),
        frozen_gas_audit=context["identity"].get("frozen_gas_audit"),
        frozen_gas_diagnostics=context["identity"].get("frozen_gas_diagnostics"),
        frozen_gas_used_for_production=False,
        runs=list(qa.RUNS), frames=len(exports), predictions=exports,
        layout="production_labels/<run>/frame_NNN.png; original Studio frame NNN, camera SHA bound",
        source_join="exact acquisition SHA + run + original Studio index; never new work index",
        maps_changed=False, annotations_read=False, annotations_changed=False, scorer_read=False,
        scorer_executed=False, claude_scorer_gate_pending=True, round3_allowed=False,
        scorer_requirements=dict(regression="rounds 2-4, B-T2, B-T3", max_recall_drop_per_run=.03,
            max_false_gas_rate_rise_per_run=.01, baseline="locked P1 final3"),
        bottom_mass_allowed=False, default_changed=False,
        code=dict(export=qa.asset(__file__), qa=qa.asset(qa.__file__)))
    atomic_json(destination/"export_receipt.json", record)
    with (destination/"README.txt").open("x", encoding="utf-8") as stream:
        stream.write("Policy A predicted-map export for Claude\n\n"
            "All 39 runs / 782 original Studio acquisitions are copied byte-identically from the new producer, "
            "with P1 final3's production_labels layout and original Studio indices. "
            "The source camera SHA, current work index, original Studio index and producer SHA remain distinct.\n\n"
            "No human labels or sealed scorer were read or changed. Claude must score the sealed regression sets once. "
            "Recall may drop by at most 0.03 and false-gas rate may rise by at most 0.01 per run, versus P1 final3. "
            "This export is ready for scoring, not a scoring pass. Round3 is not authorized by this receipt. "
            "The bottom strip remains blocked for mass; no default or existing candidate is changed.\n")
    return qa.asset(destination/"export_receipt.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "root", "identity-gate", "report", "destination"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.plan, args.root, args.identity_gate, args.report, args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

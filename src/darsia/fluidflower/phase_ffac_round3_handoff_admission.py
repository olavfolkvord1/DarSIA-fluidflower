"""Read-only, source-bound Claude PASS admission for the selected R3 fit.

No label/scorer contents are read. A Policy A review seal remains a review
seal; this separate R3 admission records the human-authorized handoff rule.
"""
from pathlib import Path

CONTRACT = "ffac-round3-policy-a-claude-pass-admission-v1"
REVIEW_CONTRACT = "ffac-selected-policy-a-review-completion-v1"
EXPORT_CONTRACT = "ffac-selected-policy-a-required-claude-export-v1"
SELECTED_SPEC_SHA = "62bf2970077c9a270f85fe3f046e48811af70d5e1c1761cd729d6289782bef32"
ROUND3_SPEC_SHA = "cb7b3355efb27e21d53da868b4155c56fa2fb272c1641e9ab662d56227248e34"
LABEL_WRAPPER_SHA = "d4588bcfadcb8a3ed48210c027157d914839e8fbf01a54e0385132cee6d8bace"


def verify_label_pass(status, export_descriptor, candidate_root):
    check = status.get("label_check", {})
    if (check.get("state") != "PASS"
            or check.get("wrapper_sha256") != LABEL_WRAPPER_SHA
            or check.get("manifest_sha256") != export_descriptor["sha256"]
            or not check.get("candidate_root")
            or Path(check["candidate_root"]).resolve() != Path(candidate_root).resolve()):
        raise ValueError("Round 3 requires Claude PASS for this exact exported manifest and candidate")
    return check


def validate_policy_a_admission(gate, reader, expected_runs):
    """Check exact immutable textual evidence before any selected mass pixels."""
    runs = set(expected_runs)
    if (gate.get("contract") != CONTRACT or gate.get("complete") is not True
            or gate.get("round3_allowed") is not True or gate.get("claude_pass_verified") is not True
            or set(gate.get("runs", ())) != runs
            or gate.get("spec_sha256") != ROUND3_SPEC_SHA
            or gate.get("production_automatically_allowed") is not False
            or gate.get("legacy_strict_gate_claimed") is not False):
        raise ValueError("A complete, honest Policy A/Claude admission is required")
    review = reader.json(gate["policy_a_review_gate"])
    export = reader.json(gate["studio_export"])
    status = reader.json(gate["claude_status_snapshot"])
    check = verify_label_pass(status, gate["studio_export"], gate["candidate_root"])
    result = dict(path=check.get("result_path"), sha256=check.get("result_sha256"))
    saved_result = gate.get("claude_result", {})
    if (not result["path"] or not result["sha256"]
            or any(saved_result.get(key) != value for key, value in result.items())):
        raise ValueError("Claude PASS needs its exact saved result evidence")
    reader.path(result)
    if (review.get("contract") != REVIEW_CONTRACT or review.get("complete") is not True
            or review.get("all_39_complete") is not True or set(review.get("runs", ())) != runs
            or review.get("coarse_area_gate_passed") is not True
            or review.get("selected_frames") != 1564 or review.get("baseline_frames") != 39
            or review.get("studio_anchor_frames") != 782 or review.get("timeline_frames") != 1603
            or len(review.get("entries",())) != 1564 or len(review.get("baselines",())) != 39
            or len(review.get("studio_probes",())) != 43
            or review.get("legacy_strict_gate_claimed") is not False
            or review.get("run_manifests") != gate.get("run_manifests")
            or review.get("phase_plan") != gate.get("phase_plan")):
        raise ValueError("The source-bound full Policy A review coverage is incomplete")
    if (export.get("contract") != EXPORT_CONTRACT or export.get("complete") is not True
            or export.get("spec_sha256") != SELECTED_SPEC_SHA
            or set(export.get("runs", ())) != runs or export.get("frames") != 782
            or len(export.get("predictions", ())) != 782
            or export.get("pipeline_sha256") != review.get("pipeline_sha256")
            or export.get("plan") != review.get("phase_plan")
            or export.get("identity_gate") != review.get("identity_gate")
            or export.get("coarse_report") != review.get("coarse_spatial_report")
            or any(export.get(key) is not False for key in
                ("maps_changed", "annotations_read", "annotations_changed", "scorer_executed"))):
        raise ValueError("The scored export differs from the completed Policy A maps/QA")
    expected = {}
    for row in review["entries"]+review["baselines"]+review["studio_probes"]:
        if row.get("source_binding",{}).get("studio_anchor"):
            key = (row["run"], row["source_binding"]["studio_frame_index"])
            if key in expected:
                raise ValueError("The completed review repeats a Studio acquisition")
            expected[key] = row
    seen = set()
    for item in export["predictions"]:
        key = (item.get("run"), item.get("original_studio_frame_index"))
        row = expected.get(key)
        if (key in seen or row is None or item.get("source_sha256") != row["source_sha256"]
                or item.get("elapsed_h") != row["elapsed_h"]
                or item.get("source_work_index") != row["frame_index"]
                or item.get("phase_receipt") != row["receipt"]
                or item.get("shape_hw") != [3572,6304]):
            raise ValueError("The Claude export changes an original Studio source/index/grid")
        seen.add(key)
    if len(expected) != 782 or seen != set(expected):
        raise ValueError("Claude export does not cover every original Studio anchor")
    return gate


def require_selected_gate(gate, reader, expected_runs):
    if gate.get("contract") == CONTRACT:
        return validate_policy_a_admission(gate, reader, expected_runs)
    # Preserve the pre-existing strict admission API for old prepared plans.
    if (gate.get("all_39_complete") is not True or gate.get("studio_time_parity_passed") is not True
            or set(gate.get("runs", ())) != set(expected_runs)):
        raise ValueError("All 39 selected phase manifests must be sealed")
    return gate

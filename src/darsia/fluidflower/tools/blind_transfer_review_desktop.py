"""Launch the existing blind surface and durably mark its evidence as opt-in."""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from pathlib import Path
import sys


from darsia.fluidflower.tools.phase_transfer_review import EVIDENCE_ROLE, read, validate_review_binding


def session_policy(plan):
    if (plan.get("evidence_role") != EVIDENCE_ROLE or plan.get("include_in_training") is not False
            or plan.get("blind_surface") != "blind_region_labeling"):
        raise ValueError("Blind transfer policy must require explicit later training selection")
    return dict(evidence_role=EVIDENCE_ROLE, include_in_training=False,
        requires_explicit_training_selection=True,
        transfer_evaluation_root=plan["evaluation_root"],
        source_model_id=plan["source_model_id"], source_model_sha256=plan["source_model_sha256"],
        evaluation_model_sha256=plan["evaluation_model_sha256"], training_seen=plan["training_seen"],
        calibration_history=plan["calibration_history"], independent_holdout=False)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--open-correction", action="store_true")
    parser.add_argument("--blind-session-id")
    selected, remaining = parser.parse_known_args(argv)
    plan = read(selected.transfer_plan)
    policy = session_policy(plan)
    binding = read(Path(plan["evaluation_root"]) / "validation/validation_binding.json")
    validate_review_binding(plan, binding)
    from darsia.fluidflower.tools import calibration_review_desktop as desktop
    args = desktop.build_parser().parse_args(remaining)
    expected_surface = "open_correction" if selected.open_correction else "blind_regions"
    if not args.single_model or args.review_surface != expected_surface or args.target_model_id != plan["evaluation_model_id"]:
        raise ValueError("Transfer entry point requires the bound blind surface or an explicitly selected separate comparison")
    if selected.open_correction:
        from darsia.fluidflower.tools.phase_transfer_reveal import capture_blind_snapshot
        receipt = capture_blind_snapshot(plan, binding, blind_session_id=selected.blind_session_id,
            reviewer=args.reviewer, save_root=args.local_review_root)
        receipt_path = Path(receipt["comparison_seed"]).parent / "reveal.json"
        policy.update(blindness_status="open_comparison", frozen_blind_snapshot=str(receipt_path),
            post_reveal_edits_are_blind=False)
        original_workspace = desktop.local_reviewer_workspace
        def seed_comparison(workspace, save_root, reviewer_name):
            localized = original_workspace(workspace, save_root, reviewer_name)
            return desktop.replace(localized, seed_review_dir=Path(receipt["comparison_seed"]),
                title="Transfer comparison · frozen blind labels")
        desktop.local_reviewer_workspace = seed_comparison
    original = desktop.write_review_session_manifest
    def write_tagged(workspace, reviewer):
        path = original(workspace, reviewer)
        payload = read(path)
        payload.update(policy)
        desktop.atomic_json(path, payload)
        return path
    desktop.write_review_session_manifest = write_tagged
    return desktop.main(remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        # The app starts this companion through pythonw; a traceback alone
        # would leave a failed label handoff invisible to the reviewer.
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, "Cannot open transfer review", str(error))
        raise SystemExit(1) from error

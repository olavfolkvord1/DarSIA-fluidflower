"""Open a frozen transfer prediction directly; no blind labels are required."""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from pathlib import Path
import sys


from darsia.fluidflower.tools.phase_transfer_review import VISIBLE_EVIDENCE_ROLE, read, validate_review_binding


def session_policy(plan):
    if (plan.get("review_mode") != "visible" or plan.get("evidence_role") != VISIBLE_EVIDENCE_ROLE
            or plan.get("include_in_training") is not False or plan.get("predictions_visible") is not True):
        raise ValueError("Direct prediction review requires a visible test plan with separate evaluation evidence")
    return dict(evidence_role=VISIBLE_EVIDENCE_ROLE, include_in_training=False,
        requires_explicit_training_selection=True, transfer_evaluation_root=plan["evaluation_root"],
        source_model_id=plan["source_model_id"], source_model_sha256=plan["source_model_sha256"],
        evaluation_model_sha256=plan["evaluation_model_sha256"], training_seen=plan["training_seen"],
        calibration_history=plan["calibration_history"], independent_holdout=False,
        evaluation_review_mode="visible", blindness_status="not_blind",
        optical_training_status=plan.get("optical_training_status", "unverified"),
        calibration_evidence_status=plan.get("calibration_evidence_status", "unverified"))


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    selected, remaining = parser.parse_known_args(argv)
    plan = read(selected.transfer_plan)
    policy = session_policy(plan)
    binding = read(Path(plan["evaluation_root"]) / "validation/validation_binding.json")
    validate_review_binding(plan, binding)
    from darsia.fluidflower.tools import calibration_review_desktop as desktop
    args = desktop.build_parser().parse_args(remaining)
    if not args.single_model or args.review_surface != "open_correction" or args.target_model_id != plan["evaluation_model_id"]:
        raise ValueError("Direct transfer review must show the bound model's predictions")
    original = desktop.write_review_session_manifest
    def write_tagged(workspace, reviewer):
        path = original(workspace, reviewer)
        desktop.atomic_json(path, read(path) | policy)
        return path
    desktop.write_review_session_manifest = write_tagged
    return desktop.main(remaining)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, "Cannot open model test", str(error))
        raise SystemExit(1) from error

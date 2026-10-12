"""Freeze saved blind labels before showing predictions, then seed comparison."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

from darsia.fluidflower.phase_workflow import discover_phase_review_sessions
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.phase_transfer_review import EVIDENCE_ROLE, read, validate_review_binding, verify_hashes, write


def comparison_decisions(payload):
    """Render both manual labels and quiz answers as regions in the correction UI."""
    from darsia.fluidflower.tools import calibration_review_desktop as desktop
    result = copy.deepcopy(payload)
    for decision in (result.get("decisions") or {}).values():
        annotations = decision.setdefault("annotations", [])
        for annotation in annotations:
            annotation["original_blind_candidate"] = annotation.get("candidate", "A")
            annotation["candidate"] = "A"
            annotation["originated_in_blind_snapshot"] = True
            annotation["counts_as_blind_evidence"] = False
        ids = {item.get("id") for item in annotations}
        for answer in decision.get("phase_quiz_answers") or []:
            choice = str(answer.get("human_choice") or "not_assessable")
            identifier = "blind-quiz-" + hashlib.sha256(
                f"{answer.get('candidate', 'A')}:{answer['proposal_id']}".encode()).hexdigest()[:20]
            if identifier in ids:
                continue
            ids.add(identifier)
            category = desktop.PHASE_CHOICE_TO_ANNOTATION.get(choice, "other")
            annotations.append(dict(copy.deepcopy(answer), id=identifier, candidate="A",
                category=category, phase=desktop.ANNOTATION_PHASES.get(category, "unspecified"),
                source="blind_transfer_quiz_snapshot", blind_proposal_id=answer["proposal_id"],
                originated_in_blind_snapshot=True, counts_as_blind_evidence=False,
                **desktop.visible_phase_payload(choice)))
        # The original quiz records remain in the immutable blind snapshot.
        # This editable view has one representation of each region so later
        # explicit training selection cannot import both stale and edited labels.
        decision["phase_quiz_answers"] = []
        decision["phase_quiz_selected_ids"] = []
        decision["phase_quiz_rounds"] = []
        decision["phase_quiz_per_candidate_limit"] = 0
        decision["notes"] = (str(decision.get("notes") or "") +
            "\nComparison starts from the saved blind-label snapshot. Later edits are not blind.").strip()
    return result


def capture_blind_snapshot(plan, binding, *, blind_session_id, reviewer, save_root):
    """Capture the first reveal only; reopening never promotes later edits to blind labels."""
    from darsia.fluidflower.tools.calibration_review_desktop import atomic_json
    validate_review_binding(plan, binding)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(blind_session_id)):
        raise ValueError("A valid originating blind session is required")
    root = Path(plan["evaluation_root"]) / "blind_snapshots" / blind_session_id
    receipt_path = root / "reveal.json"
    if receipt_path.is_file():
        receipt = read(receipt_path)
        if (receipt["evaluation_model_id"] != plan["evaluation_model_id"]
                or receipt["evaluation_model_sha256"] != plan["evaluation_model_sha256"]
                or receipt["reviewer"].casefold() != reviewer.strip().casefold()):
            raise ValueError("The saved blind snapshot belongs to another evaluation or reviewer")
        verify_hashes(receipt["snapshot_sha256"])
        return receipt
    sessions = [row for row in discover_phase_review_sessions([Path(save_root)])
        if row.get("review_session_id") == blind_session_id
        and row.get("reviewer", "").strip().casefold() == reviewer.strip().casefold()]
    if len(sessions) != 1:
        raise ValueError("Save blind labels in the originating session before opening its comparison")
    session = sessions[0]
    if (session.get("evidence_role") != EVIDENCE_ROLE or session.get("include_in_training") is not False
            or session.get("review_surface") != "blind_regions"
            or session.get("target_model_id") != plan["evaluation_model_id"]
            or session.get("selected_candidates") != binding["selected_candidates"]):
        raise ValueError("The saved blind session identifies a different model or target")
    source = Path(session["decision_path"])
    content = source.read_bytes()
    payload = json.loads(content.decode("utf-8-sig"))
    decisions = payload.get("decisions") or {}
    if set(decisions) != {plan["run"]}:
        raise ValueError("Saved blind decisions must identify exactly the selected target")
    decision = decisions[plan["run"]]
    expected = binding["selected_candidates"][plan["run"]]
    if any((decision.get(side) or {}).get("candidate_id") != expected for side in ("candidate_a", "candidate_b")):
        raise ValueError("Saved blind decisions identify another candidate")
    manual = decision.get("annotations") or []
    quiz = decision.get("phase_quiz_answers") or []
    if not manual and not quiz:
        raise ValueError("Save at least one blind region label before opening comparison")
    if root.exists():
        raise ValueError("An incomplete blind snapshot exists; it must not be overwritten")
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / "original/pairwise_review_decisions.json"
    frozen.parent.mkdir()
    frozen.write_bytes(content)
    session_copy = root / "original/review_session.json"
    session_copy.write_bytes(Path(session["session_path"]).read_bytes())
    seed = root / "comparison_seed/pairwise_review_decisions.json"
    write(seed, comparison_decisions(payload))
    stamp = utc_now()
    receipt = dict(format="phase-blind-reveal-snapshot-v1", created_utc=stamp,
        evaluation_model_id=plan["evaluation_model_id"], evaluation_model_sha256=plan["evaluation_model_sha256"],
        source_model_id=plan["source_model_id"], source_model_sha256=plan["source_model_sha256"],
        blind_session_id=blind_session_id, reviewer=reviewer.strip(), run=plan["run"],
        source_decisions=str(source), source_decisions_sha256=hashlib.sha256(content).hexdigest(),
        frozen_decisions=str(frozen), comparison_seed=str(seed.parent),
        manual_regions=len(manual), quiz_answers=len(quiz),
        snapshot_sha256={str(path): sha256_file(path) for path in (frozen, session_copy, seed)},
        include_in_training=False, later_edits_are_blind=False)
    write(receipt_path, receipt)
    reveal_state = dict(blindness_status="revealed", first_prediction_reveal_utc=stamp,
        frozen_blind_snapshot=str(receipt_path), frozen_blind_snapshot_sha256=sha256_file(receipt_path),
        post_reveal_edits_are_blind=False)
    # This separate record cannot be lost to an already-running autosave writer.
    marker = source.parent / "blind_reveal.json"
    write(marker, reveal_state)
    manifest_path = Path(session["session_path"])
    current = read(manifest_path)
    current.update(reveal_state)
    atomic_json(manifest_path, current)
    return receipt

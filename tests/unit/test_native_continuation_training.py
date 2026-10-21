"""The native page explains eligibility and reports candidate completion honestly."""
import json
from types import SimpleNamespace

from darsia.fluidflower.native_continuation_training import NativeContinuationTraining
from darsia.fluidflower.phase_training.continuation import CONTRACT


def test_missing_selected_parent_has_a_visible_reason_and_registers_the_route():
    studio = SimpleNamespace(_selected_validation_model=lambda: None, _training_finished=lambda *args: None)
    steps = SimpleNamespace(studio=studio, training_routes={}, series_id="campaign")
    adapter = NativeContinuationTraining(steps)
    assert steps.training_routes["existing"] == adapter.start
    assert "Choose an indexed saved model" in adapter.refresh()
    assert not adapter.ready


def test_success_records_candidate_without_claiming_phase_maps_or_validation(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(dict(
        artifact_kind="phase-continuation-candidate", nonconverged_heads=["co2_presence"])), encoding="utf-8")
    events, status = [], []
    studio = SimpleNamespace(
        _training_finished=lambda *args: events.append("legacy-success"),
        training_request=dict(output_root=str(tmp_path), model_id="new-version", continuation=dict(contract=CONTRACT)),
        training_process=SimpleNamespace(deleteLater=lambda: events.append("released")),
        _read_output=lambda: None,
        stop_button=SimpleNamespace(setEnabled=lambda value: events.append(("stop-enabled", value))),
        _record_completed_model=lambda model, root: events.append((model, root)),
        training_completed=SimpleNamespace(emit=lambda value: events.append(("completed", value))),
        _update_action_states=lambda: events.append("refreshed"),
        status=SimpleNamespace(setText=status.append))
    adapter = NativeContinuationTraining(SimpleNamespace(studio=studio, training_routes={}, series_id="campaign"))
    adapter.finished(0, None)
    assert "legacy-success" not in events and studio.training_process is None
    assert ("new-version", tmp_path) in events
    assert "Prepare phase maps" in status[-1] and "independent validation" in status[-1]
    assert "convergence was not established" in status[-1]


def test_original_failures_and_other_training_contracts_keep_the_original_handler():
    handled = []
    studio = SimpleNamespace(training_request={}, _training_finished=lambda *args: handled.append(args))
    adapter = NativeContinuationTraining(SimpleNamespace(studio=studio, training_routes={}, series_id="campaign"))
    adapter.finished(0, "normal")
    adapter.finished(1, "failed")
    assert handled == [(0, "normal"), (1, "failed")]

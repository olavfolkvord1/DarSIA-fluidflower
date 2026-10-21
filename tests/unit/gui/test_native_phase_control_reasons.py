"""Native phase actions expose the exact selection and preserve launch gates."""
from types import SimpleNamespace as NS

import pytest
from darsia.fluidflower import native_phase_steps as ui
from darsia.fluidflower.native_darsia_app import NativeFluidFlowerMainWindow
from darsia.fluidflower.normalisation_step import NormalisationStep

from .test_native_phase_steps import FakeNormalisation, FakeStudio


@pytest.fixture
def page(qtbot):
    studio = FakeStudio()
    selection = NS(model=None, run="experiment-a")
    studio._selected_validation_model = lambda: selection.model
    studio._request_async_model_refresh = lambda **kw: studio.calls.append(("saved-metadata", kw))
    normalisation = FakeNormalisation()
    steps = ui.NativePhaseSteps(studio, normalisation=normalisation,
        run_provider=lambda: selection.run)
    steps.series_id = "campaign"
    studio._active_model_series = "campaign"
    qtbot.addWidget(studio)
    qtbot.addWidget(steps)
    return NS(studio=studio, steps=steps, selection=selection, normalisation=normalisation)


def selected(page, **changes):
    page.selection.model = dict(model_id="model-alpha", root="saved-model", **changes)
    page.steps.set_normalisation_ready(True)
    return page.selection.model


def test_missing_maps_name_the_model_series_and_run_and_offer_only_saved_metadata(page):
    selected(page)
    page.studio.open_model_correction.setEnabled(False)
    page.steps._legacy_availability["open_model_correction"] = False
    page.steps.refresh()
    text = page.steps.review_selection_notice.text()
    assert all(value in text for value in ("model-alpha", "campaign", "experiment-a", "no verified saved phase maps"))
    assert page.studio.open_model_correction.toolTip() == text
    page.steps.phase_review = lambda: None
    page.steps.refresh()
    assert page.steps.phase_review_button.toolTip() == text
    page.steps.review_route_button.click()
    assert page.studio.calls == [("saved-metadata", {"bindings_only": True})]
    assert page.studio.training_process is None
    assert not page.studio.open_model_correction.isEnabled()


def test_missing_model_route_opens_existing_basis_selector_without_selecting_a_model(page):
    page.steps.set_normalisation_ready(True)
    page.steps.phase_tabs.setCurrentWidget(page.steps.groups[3])
    page.steps.review_route_button.click()
    assert page.steps.phase_tabs.currentWidget() is page.steps.groups[2]
    assert page.selection.model is None
    assert page.studio.calls == []


def test_missing_normalisation_has_visible_reason_and_direct_route(page):
    page.steps.refresh()
    assert "Image normalisation" in page.steps.review_selection_notice.text()
    assert "Image normalisation" in page.steps.import_button.toolTip()
    page.steps.review_route_button.click()
    assert page.steps.section == "normalisation"
    assert not page.studio.open_model_correction.isEnabled()


@pytest.mark.parametrize("fields, text", [
    ({"validation_binding_loading": True}, "still being checked"),
    ({"validation_collection_error": "Saved source SHA differs"}, "Saved source SHA differs"),
])
def test_phase_window_never_opens_an_unverified_or_other_run_binding(page, fields, text):
    selected(page, **fields)
    host = NS(phase_model_workspace=NS(studio_page=NS(studio=page.studio, native_steps=page.steps), series_id="campaign"))
    NativeFluidFlowerMainWindow._open_existing_phase_review(host)
    assert text in page.studio.review_status.text()
    assert page.steps.phase_tabs.currentWidget() is page.steps.groups[3]
    assert not page.studio.calls


def test_phase_window_uses_existing_review_route_when_exact_selection_is_ready(page):
    selected(page, validation_binding={"selected_candidates": {"experiment-a": "saved-candidate"}})
    page.studio._active_model_series = "CAM_PAIGN"
    host = NS(phase_model_workspace=NS(studio_page=NS(studio=page.studio, native_steps=page.steps), series_id="campaign"))
    NativeFluidFlowerMainWindow._open_existing_phase_review(host)
    assert page.studio.calls == [("original-review", "open_correction")]
    assert page.studio.training_process is None


def test_run_change_refreshes_the_reason_without_rebinding_saved_maps(page):
    model = selected(page, validation_binding={"selected_candidates": {"experiment-a": "saved-candidate"}})
    assert not page.steps.review_launch_reason()
    page.selection.run = "experiment-b"
    host = NS(phase_model_workspace=NS(studio_page=NS(native_steps=page.steps)))
    NativeFluidFlowerMainWindow._refresh_phase_review_requirements(host)
    assert "experiment-b" in page.steps.review_selection_notice.text()
    assert model["validation_binding"]["selected_candidates"] == {"experiment-a": "saved-candidate"}
    assert page.studio.calls == []


def test_series_review_keeps_available_runs_when_the_hidden_run_selector_differs(page):
    selected(page, validation_binding={"selected_candidates": {"experiment-b": "saved-candidate"}})
    page.steps.refresh()
    assert "Available runs: experiment-b" in page.steps.review_selection_notice.text()
    assert "Current run selection: experiment-a" in page.steps.review_selection_notice.text()
    host = NS(phase_model_workspace=NS(studio_page=NS(studio=page.studio, native_steps=page.steps), series_id="campaign"))
    NativeFluidFlowerMainWindow._open_existing_phase_review(host)
    assert page.studio.calls == [("original-review", "open_correction")]


def test_phase_window_callback_error_is_visible_in_words(page, monkeypatch):
    errors = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, _title, text: errors.append(text))
    def fail():
        raise OSError("Saved phase-map folder is unavailable")
    page.steps.phase_review = fail
    page.steps.phase_review_button.setEnabled(True)
    page.steps.phase_review_button.clicked.connect(page.steps._open_phase_review)
    page.steps.phase_review_button.click()
    assert errors == ["Could not open phase review: Saved phase-map folder is unavailable"]
    assert page.studio.review_status.text() == errors[0]


def test_review_and_label_callback_error_is_visible_without_changing_the_gate(page, monkeypatch):
    selected(page, validation_binding={})
    errors = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, _title, text: errors.append(text))
    def fail(_mode):
        raise OSError("Review launcher could not read its saved manifest")
    page.steps._original_review = fail
    page.studio.open_model_correction.click()
    assert errors == ["Could not open phase review: Review launcher could not read its saved manifest"]
    assert page.steps.review_selection_notice.text() == errors[0]


def test_saved_map_recheck_error_is_visible_and_never_starts_a_job(page, monkeypatch):
    selected(page)
    errors = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda _parent, _title, text: errors.append(text))
    def fail(**_kwargs):
        raise OSError("Saved metadata location is unavailable")
    page.studio._request_async_model_refresh = fail
    page.steps.review_route_button.click()
    assert errors == ["Could not check saved phase maps: Saved metadata location is unavailable"]
    assert page.studio.review_status.text() == errors[0]
    assert page.studio.training_process is None


def test_training_and_saved_session_controls_explain_their_existing_gates(page):
    page.steps._legacy_availability["resume_session_button"] = False
    page.steps.refresh()
    assert "Image normalisation" in page.studio.resume_session_button.toolTip()
    assert "name the new model version" in page.studio.start_button.toolTip()
    page.steps.set_normalisation_ready(True)
    assert "Select an available saved review session" in page.studio.resume_session_button.toolTip()
    assert "compatible existing-model training route" in page.studio.start_button.toolTip()


def test_normalisation_disabled_controls_explain_source_and_override_requirements(qtbot):
    step = NormalisationStep(provider=lambda *_args: None)
    qtbot.addWidget(step)
    assert "Choose an experimental series" in step.override.toolTip()
    assert "Choose an experimental series" in step.save_button.toolTip()
    step.series_root, step.series_id = "unused", "campaign"
    step._update_controls()
    assert "Open saved v5 results" in step.reference.toolTip()
    step.snapshot, step.manifest_bytes = object(), b"saved"
    step._update_controls()
    assert "Enable Override" in step.reference.toolTip()

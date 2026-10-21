"""Attach explicit saved-model continuation to one native Phase model page."""
from __future__ import annotations

import json
from pathlib import Path

from .phase_training import continuation
from .phase_training.scope import requested_reviewers, series_id


class NativeContinuationTraining:
    """One page's continuation readiness, request launch and honest completion."""

    def __init__(self, steps):
        self.steps, self.studio = steps, steps.studio
        self.ready, self.reason = False, "Choose an indexed saved model as the training basis."
        self._cache_key = None
        self._parent_reviewers = None
        self._original_finished = getattr(self.studio, "_training_finished", None)
        steps.continuation_training = self
        steps.training_routes["existing"] = self.start
        self.studio._training_finished = self.finished
        for name, signal_name in (("annotators", "itemChanged"), ("bundles", "textChanged"),
                                  ("validation_model", "currentIndexChanged")):
            signal = getattr(getattr(self.studio, name, None), signal_name, None)
            if signal is not None:
                signal.connect(lambda *_args: steps.refresh())

    def refresh(self):
        """Return the model-specific reason; opening the page writes nothing."""
        try:
            getter = getattr(self.studio, "_selected_validation_model", None)
            entry = getter() if callable(getter) else None
            if not entry:
                raise ValueError("Choose an indexed saved model in Basis before continuing training.")
            root = Path(entry["root"])
            paths = [root / "training_request.json", *[root / "models/generic" / f"{head}_signal.json"
                                                       for head in continuation.HEADS]]
            key = (json.dumps(entry.get("library_source_identity"), sort_keys=True), self.steps.series_id,
                   tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths))
            if key != self._cache_key:
                _bound, parent, _previous, _heads = continuation.inspect_parent(entry)
                selected = {series_id(value) for value in parent["settings"].get("series") or ()}
                if selected != {series_id(self.steps.series_id)}:
                    raise ValueError("The selected model belongs to another series.")
                self._parent_reviewers = requested_reviewers(parent["settings"])
                self._cache_key = key
            selector = getattr(self.studio, "_selected_annotators", None)
            selected_reviewers = selector() if callable(selector) else None
            if selected_reviewers == []:
                raise ValueError("Tick the parent's annotators before continuing training.")
            if requested_reviewers({"reviewers": selected_reviewers}) != self._parent_reviewers:
                raise ValueError("The annotator selection differs from the parent; create a fresh training version to change reviewers.")
            if not callable(self._original_finished):
                raise ValueError("The training completion handler is unavailable.")
            actor = getattr(self.studio, "reviewer", None)
            evidence = getattr(self.studio, "bundles", None)
            if actor is None or not actor.text().strip():
                raise ValueError("Set the reviewer identity before continuing training.")
            if evidence is None or not evidence.toPlainText().strip():
                raise ValueError("Save and select new reviewed evidence before continuing training.")
            self.ready = True
            self.reason = ("Continue the selected model on CPU from its saved weights and scaling. "
                           "Earlier evidence and evaluation exclusions are retained; the new version requires independent validation.")
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._cache_key = None
            self.ready, self.reason = False, "Continuation unavailable: " + str(error)
        return self.reason

    def start(self):
        from .phase_workflow import parse_training_bundles
        self.refresh()
        if not self.ready:
            raise ValueError(self.reason)
        actor = self.studio.reviewer.text().strip()
        if not actor:
            raise ValueError("Set the reviewer identity before continuing training.")
        bundles = parse_training_bundles(self.studio.bundles.toPlainText())
        if not bundles:
            raise ValueError("Save and select new reviewed evidence before continuing training.")
        request = continuation.prepare_request(parent_entry=self.studio._selected_validation_model(),
            output_parent=self.studio._models_folder_for_series(self.steps.series_id),
            cache_parent=Path(self.studio.cache_parent.text()), bundles=bundles, actor=actor)
        self.studio._launch_training_job(request, continuation.build_command(request),
            Path(request["request_path"]).parent,
            "Continuing the saved model as a new CPU candidate; independent validation remains required.")

    def finished(self, exit_code, exit_status):
        request = dict(self.studio.training_request or {})
        if (int(exit_code) != 0
                or (request.get("continuation") or {}).get("contract") != continuation.CONTRACT):
            return self._original_finished(exit_code, exit_status) if callable(self._original_finished) else None
        self.studio._read_output()
        process, self.studio.training_process = self.studio.training_process, None
        self.studio.stop_button.setEnabled(False)
        if process is not None:
            process.deleteLater()
        try:
            root = Path(request["output_root"])
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            if summary.get("artifact_kind") != "phase-continuation-candidate":
                raise ValueError("The completed job has no continuation candidate summary")
            self.studio._record_completed_model(request["model_id"], root)
            self.studio.training_completed.emit(str(root))
            incomplete = summary.get("nonconverged_heads") or []
            message = "New candidate saved. Prepare phase maps for review and complete independent validation before using it."
            if incomplete:
                message += " Optimizer convergence was not established for: " + ", ".join(incomplete) + "."
            self.studio.status.setText(message)
        except (OSError, ValueError, KeyError) as error:
            self.studio.status.setText("Continuation files were retained; candidate inspection failed: " + str(error))
        self.studio._update_action_states()


__all__ = ["NativeContinuationTraining"]

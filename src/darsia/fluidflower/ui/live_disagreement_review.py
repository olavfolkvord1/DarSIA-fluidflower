"""Build and open ranked annotation checks from the active review window."""

from darsia.fluidflower.series_paths import tool_arguments, local_cache_root
from datetime import UTC, datetime
import json
from pathlib import Path
import uuid

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QPushButton

from darsia.fluidflower.phase_evidence_audit import load_queue, queue_directory
from darsia.fluidflower.phase_workflow import default_phase_review_roots, phase_python_executable
from .evidence_audit_library import EvidenceAuditLibrary


def review_model_root(workspace):
    """Resolve the source model of a normal or composite frozen review."""
    root = Path(workspace.combined_root)
    receipt = root / 'collection_receipt.json'
    if receipt.is_file():
        value = json.loads(receipt.read_text(encoding='utf-8-sig'))
        identity, path = value.get('source_model_id'), value.get('source_model_root')
    else:
        value = json.loads((root/'validation_binding.json').read_text(encoding='utf-8-sig'))
        identity, path = value.get('model_id'), value.get('model_root')
    if not path or identity != workspace.target_model_id:
        raise ValueError('The review binding does not identify the selected model')
    return Path(path), root


class LiveDisagreementReview:
    def __init__(self, window, api):
        self.window, self.api, self.process = window, api, None
        self.button = QPushButton('Check disagreements')
        self.button.setToolTip('Rank saved phase-map disagreements with current and inherited labels. '
            'Uses existing maps; does not train a model or change labels.')
        self.button.setEnabled(bool(window.workspace.single_model))
        self.button.clicked.connect(self.start)

    def status(self, text):
        self.window.statusBar().showMessage(text)
        self.button.setToolTip(text)

    def start(self):
        if self.process is not None:
            return
        w = self.window
        try:
            w._flush_save()
            if w.save_dirty:
                raise ValueError('Finish saving the current annotations before checking disagreements')
            previous = getattr(w, 'evidence_audit', None)
            if previous is not None and previous.pool_process is not None:
                raise ValueError('Wait until the current additional-region request finishes')
            model, source = review_model_root(w.workspace)
            self.model_id = w.workspace.target_model_id
            self.output = queue_directory(model)/(
                f'automatic-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.json')
            self.output.parent.mkdir(parents=True, exist_ok=True)
            process = QProcess(w)
            process.setProgram(phase_python_executable())
            process.setWorkingDirectory(str(local_cache_root()))
            process.setArguments(['-m', 'darsia.fluidflower.tools.build_phase_disagreement_review', '--model', str(model),
                '--review-root', str(source), '--output', str(self.output)])
            process.readyReadStandardOutput.connect(self.progress)
            process.finished.connect(self.finished)
            process.errorOccurred.connect(self.failed)
            self.process = process
            self.button.setEnabled(False)
            self.button.setText('Checking disagreements…')
            self.status('Checking saved labels against existing phase maps in the background…')
            process.start()
        except Exception as exc:
            self.status(f'Cannot check disagreements: {exc}')

    def progress(self):
        if self.process is not None:
            message = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace').strip()
            if message:
                self.status(message.splitlines()[-1][:500])

    def reset(self):
        self.button.setText('Check disagreements')
        self.button.setEnabled(bool(self.window.workspace.single_model))

    def failed(self, error):
        if self.process is not None and error == QProcess.ProcessError.FailedToStart:
            message = self.process.errorString()
            self.process.deleteLater()
            self.process = None
            self.reset()
            self.status(f'Could not start disagreement check: {message}')

    def finished(self, code, _status):
        process, self.process = self.process, None
        if process is None:
            return
        error = bytes(process.readAllStandardError()).decode('utf-8', errors='replace')
        process.deleteLater()
        self.reset()
        try:
            if code != 0:
                raise RuntimeError(error[-1600:] or f'Check exited with code {code}')
            queue = load_queue(self.output, verify_sources=True)
            if queue['model_id'] != self.model_id or self.window.workspace.target_model_id != self.model_id:
                raise ValueError('The selected model changed while the check was running; open its saved queue in Studio')
            w = self.window
            # Queues have their own progress namespace; replacing the dock does
            # not delete any decisions, deferred cases, or previous queue.
            previous = getattr(w, 'evidence_audit', None)
            if previous is not None:
                w.removeEventFilter(previous)
                w.removeDockWidget(previous)
                previous.deleteLater()
                w.evidence_audit = None
            from .evidence_audit import attach_evidence_audit
            attach_evidence_audit(w, self.output, self.api)
            coverage = EvidenceAuditLibrary.coverage_summary(queue)
            self.status(f"Disagreement check ready: {len(queue['entries'])} ranked areas. Labels are unchanged."
                        + (" " + coverage if coverage else ""))
        except Exception as exc:
            self.status(f'Could not open disagreement check: {exc}')


def attach_live_disagreement_action(window, api):
    controller = LiveDisagreementReview(window, api)
    window.live_disagreement_review = controller
    window.check_disagreements = controller.button
    return controller.button

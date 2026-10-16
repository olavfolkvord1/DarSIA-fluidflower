"""Native-only display adapter for existing saved workflow artefacts.

Navigation, numerical actions and the original settings panes are unchanged.
This adapter never creates a workspace, starts a workflow or discovers history.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from darsia.fluidflower.workflow_row_status import load_status_context, row_status
from darsia.fluidflower.series_paths import series_key
from darsia.presets.workflows.results_folder import open_in_file_explorer


STATE_LABELS = {
    "done": "Done",
    "missing": "Not done",
    "not_applicable": "Not applicable",
}


def status_text(state) -> str:
    """Keep what was found and the date visible; paths stay in the detail pane."""
    line = STATE_LABELS[state.state]
    if state.date:
        line += " · " + state.date
    return line + "\n" + state.summary


class NativeWorkflowStatus:
    """Attach passive saved-file status to the native window's existing rows."""

    def __init__(self, window):
        self.window = window
        self.states = {}
        self.rows = {}
        self.context = None
        self._refreshing = False
        self.selected_references = {}
        self.panel = QWidget()
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(0, 2, 0, 2)
        line = QHBoxLayout()
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        line.addWidget(self.label, 1)
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self._open_current)
        line.addWidget(self.open_button)
        self.start_button = QPushButton("Start")
        self.start_button.setToolTip(
            "Open this step's settings; no calculation starts."
        )
        self.start_button.clicked.connect(self._start_current)
        line.addWidget(self.start_button)
        layout.addLayout(line)
        self.missing_label = QLabel()
        self.missing_label.setTextFormat(Qt.TextFormat.PlainText)
        self.missing_label.setWordWrap(True)
        self.missing_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.missing_label)
        right = window.content_stack.parentWidget().layout()
        right.insertWidget(right.indexOf(window.content_stack), self.panel)
        self.panel.hide()
        workspace = window.phase_model_workspace
        original_studio = workspace.ensure_studio

        def ensure_studio(*args, **kwargs):
            page = original_studio(*args, **kwargs)
            self.bind_studio_page(page)
            return page

        ensure_studio.__wrapped__ = original_studio
        workspace.ensure_studio = ensure_studio
        original_mass = window.ensure_mass_analysis

        def ensure_mass(*args, **kwargs):
            page = original_mass(*args, **kwargs)
            self.bind_mass_workspace(page)
            return page

        ensure_mass.__wrapped__ = original_mass
        window.ensure_mass_analysis = ensure_mass
        for category in ("setup", "calibration", "analysis"):
            section = window.sidebar._sections.get(category)
            if section is None:
                continue
            for row in section._rows.values():
                logical = (row.action, row.checkbox_id)
                controls = QWidget(row)
                vertical = QVBoxLayout(controls)
                vertical.setContentsMargins(0, 0, 0, 0)
                vertical.setSpacing(1)
                row.layout().removeWidget(row._text_label)
                vertical.addWidget(row._text_label)
                text = QLabel()
                text.setTextFormat(Qt.TextFormat.PlainText)
                text.setWordWrap(True)
                text.setStyleSheet("font-size: 10px;")
                text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                vertical.addWidget(text)
                button = QPushButton("Start")
                button.setToolTip("Open this step's settings; no calculation starts.")
                button.clicked.connect(
                    lambda _checked=False, key=logical: self.activate(key)
                )
                vertical.addWidget(button)
                # Keep the original row label, selection and click behaviour.
                row.layout().insertWidget(1, controls, 1)
                self.rows[logical] = (row, text, button)

    def bind_studio_page(self, page) -> None:
        """Observe only an already requested Studio; never instantiate one."""
        if getattr(page, "_workflow_status_bound", False):
            return
        signal = getattr(page, "studio_loaded", None)
        if signal is None:
            return
        page._workflow_status_bound = True
        signal.connect(
            lambda: (
                self.bind_review_studio(page.studio)
                if getattr(page, "studio", None) is not None
                else None
            )
        )
        if getattr(page, "studio", None) is not None:
            self.bind_review_studio(page.studio)

    def bind_review_studio(self, studio) -> None:
        if getattr(studio, "_workflow_status_bound", False):
            return
        studio._workflow_status_bound = True
        studio.review_opened.connect(
            lambda binding: self._review_opened(studio, binding)
        )

    def _review_opened(self, studio, binding) -> None:
        workspace = self.window.phase_model_workspace
        if getattr(studio, "_active_model_series", None) != workspace.series_id:
            return
        selected = studio._selected_validation_model() or {}
        if selected.get("validation_binding") != binding:
            return
        root = binding.get("combined_root")
        if not isinstance(root, str) or not root.strip():
            return
        # This is the actual prepared collection manifest named by the already
        # selected binding, not its job-less receipt or a discovered history.
        self.set_selected_reference(
            "review", str(Path(root) / "postflight_manifest.json")
        )

    def bind_mass_workspace(self, page) -> None:
        saved = getattr(page, "saved_results", None)
        if saved is None or getattr(saved, "_workflow_status_bound", False):
            return
        saved._workflow_status_bound = True
        original = saved.load_result

        def load_result(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            current = getattr(saved, "result_path", None)
            if (
                current is not None
                and Path(current).absolute() == Path(path).absolute()
            ):
                self._result_loaded(saved, current)
            return result

        load_result.__wrapped__ = original
        saved.load_result = load_result

    def _result_loaded(self, saved, path) -> None:
        value = getattr(saved, "result", None) or {}
        candidate = value.get("selected_inputs", {}).get("candidate", {})
        series = self.window.phase_model_workspace.series_id
        try:
            if series_key(str(candidate.get("dataset") or "")) != series:
                return
        except ValueError:
            return
        wanted = str(Path(path).absolute()).casefold()
        entry = next(
            (
                entry
                for entry in saved.entries
                if str(Path(entry.get("path", "")).absolute()).casefold() == wanted
            ),
            None,
        )
        # A newly selected unregistered file must not inherit an earlier receipt.
        self.set_selected_reference(
            "result_receipt", str((entry or {}).get("receipt_path") or "")
        )

    def set_selected_reference(self, kind: str, path: str) -> None:
        """Called only by an existing explicit selection, never a history scan."""
        if kind not in {"selection_plan", "review", "result_receipt"}:
            raise ValueError(f"Unknown saved reference kind: {kind}")
        key = self._series_key()
        if not key:
            raise ValueError(
                "Open a registered series before selecting saved metadata."
            )
        self.selected_references.setdefault(key, {})[kind] = str(path or "")
        self.refresh()

    def _series_key(self):
        workspace = self.window.phase_model_workspace
        root = workspace.series_roots.get(workspace.series_id)
        return str(Path(root).absolute()).casefold() if root is not None else ""

    def refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            window = self.window
            workspace = window.phase_model_workspace
            root = workspace.series_roots.get(workspace.series_id)
            references = self.selected_references.get(self._series_key(), {})
            # The plan dialog has one documented default path. No latest/history
            # search and no pixel/NPZ read is needed to inspect that named plan.
            plan = references.get("selection_plan", "")
            if not plan and root is not None:
                plan = str(Path(root) / "config/image_selection_plan.toml")
            context = load_status_context(
                root,
                scope_context=window.scope_context,
                method_selection=window._saved_method,
                selection_plan_reference=plan,
                review_reference=references.get("review", ""),
                result_receipt_reference=references.get("result_receipt", ""),
            )
            self.context = context
            self.states = {}
            for logical, (row, text, button) in self.rows.items():
                state = row_status(context, *logical)
                self.states[logical] = state
                text.setText(status_text(state))
                details = self._details(state)
                text.setToolTip(details)
                button.setText("Open" if state.state == "done" else "Start")
                button.setVisible(state.state != "not_applicable")
                button.setEnabled(row.isEnabled())
                button.setToolTip(
                    details
                    if state.state == "done"
                    else "Open this step's settings; no calculation starts.\n" + details
                )
                row.set_completion_state("done" if state.state == "done" else "none")
            selected = window.sidebar.get_selected()
            state = self.states.get(selected)
            self.panel.setVisible(state is not None)
            if state is None:
                return
            self.label.setText(status_text(state))
            self.label.setToolTip(self._details(state))
            self.open_button.setVisible(state.state == "done" and bool(state.artifacts))
            self.start_button.setVisible(state.state == "missing")
            row = self.rows[selected][0]
            self.start_button.setEnabled(row.isEnabled())
            self.missing_label.setText(self._details(state))
            self.missing_label.setVisible(state.state == "missing")
        finally:
            self._refreshing = False

    @staticmethod
    def _details(state) -> str:
        lines = []
        if state.date_source:
            lines.append("Date source: " + state.date_source)
        lines.extend(f"Saved: {item.kind}: {item.path}" for item in state.artifacts)
        lines.extend("Missing: " + item for item in state.missing)
        if state.producer:
            lines.append("Produced by: " + state.producer)
        if state.next_step:
            lines.append("Next: " + state.next_step)
        return "\n".join(lines)

    def activate(self, logical) -> None:
        state = self.states.get(logical)
        if logical == ("image_selection", "plan") and state is not None and state.state == "done":
            self.window._select_row(logical)
            self.window.image_selection_workflow.open()
            return
        if state is not None and state.state == "done" and state.artifacts:
            try:
                open_in_file_explorer(Path(state.artifacts[0].path))
            except (OSError, RuntimeError) as error:
                self.window.print_log(f"Could not open saved output: {error}")
            return
        # An explicit Start is navigation only, including when output is absent.
        self.window._select_row(logical)

    def _open_current(self):
        self.activate(self.window.sidebar.get_selected())

    def _start_current(self):
        self.window._select_row(self.window.sidebar.get_selected())

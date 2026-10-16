"""Native-only Image selection row; no protocol or image read on navigation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from darsia.fluidflower.series_image_selection import read_series_image_selection
from darsia.fluidflower.image_selection_status import saved_selection_summary
from darsia.fluidflower.stored_image_selection import loaded_studio_stored_plan
from darsia.fluidflower.image_selection_widget import SeriesImageSelectionDialog


class NativeImageSelectionWorkflow:
    def __init__(
        self,
        window,
        *,
        dialog_factory=SeriesImageSelectionDialog,
        rule_interpretation="",
        stored_plan_provider=None,
    ):
        self.window, self.dialog_factory = window, dialog_factory
        self.stored_plan_provider = (
            stored_plan_provider
            if stored_plan_provider is not None
            else lambda metadata: loaded_studio_stored_plan(self.window, metadata)
        )
        # Kept for existing callers; the dialog names Q6 Default20 explicitly.
        self.rule_interpretation = rule_interpretation
        self.panel = QWidget()
        outer = QVBoxLayout(self.panel)
        title = QLabel("Image selection")
        outer.addWidget(title)
        self.status = QLabel("Choose a registered series on Start.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.note = QLabel(
            "Choose images per run using the saved imaging and injection protocols. Load, Preview and Save are explicit actions. No review or analysis is started here."
        )
        self.note.setWordWrap(True)
        outer.addWidget(self.note)
        self.edit_button = QPushButton("Edit image selection…")
        self.edit_button.clicked.connect(self.open)
        outer.addWidget(self.edit_button)
        outer.addStretch(1)
        self.dialog = None

    def refresh(self):
        root = self.window._method_series_root()
        self.edit_button.setEnabled(root is not None)
        if root is None:
            self.status.setText("Choose a registered series on Start.")
            return
        try:
            _table, raw = read_series_image_selection(root)
            import tomllib

            summary = saved_selection_summary(tomllib.loads(raw.decode("utf-8-sig")))
            self.status.setText(summary["summary"] + "\n" + summary["note"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.status.setText("Not done — " + str(error))

    def show(self):
        self.refresh()
        self.window.content_stack.setCurrentWidget(self.panel)

    def open(self, *_args):
        root = self.window._method_series_root()
        if root is None:
            self.window.print_log(
                "Choose a registered series on Start before editing image selection."
            )
            return
        try:
            dialog = self.dialog_factory(
                root,
                self.window,
                rule_interpretation=self.rule_interpretation,
                **(
                    {"stored_plan_provider": self.stored_plan_provider}
                    if self.stored_plan_provider is not None
                    else {}
                ),
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.warning(
                self.window, "Image selection is unavailable", str(error)
            )
            return
        self.dialog = dialog
        current_run = self.window.run_combo.currentText()
        index = dialog.run_combo.findData(current_run)
        if index >= 0:
            dialog.run_combo.setCurrentIndex(index)
        dialog.saved.connect(self._saved)
        dialog.exec()

    def _saved(self, _run):
        self.refresh()
        # Rebuild one bounded cached context using the actual saved manifest.
        # Numerical actions and other calibration/method widgets are untouched.
        self.window._refresh_scope_display()

"""Native Calibration context: one fixed line, details within the page scroll."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QLayout, QScrollArea, QVBoxLayout, QWidget

from darsia.fluidflower.series_method import METHOD_NAMES


class NativeCalibrationHeader:
    """Move existing controls, preserving their signals and settings behaviour."""

    def __init__(self, window):
        self.window = window
        self.right = window.content_stack.parentWidget().layout()
        self.widgets = [
            self.right.itemAt(i).widget()
            for i in range(self.right.count())
            if self.right.itemAt(i).widget() is not None
        ]
        self.line = QLabel()
        self.line.setObjectName("calibrationContextLine")
        self.line.setTextFormat(Qt.TextFormat.PlainText)
        self.line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.right.insertWidget(0, self.line)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("calibrationRowScroll")
        self.scroll.setWidgetResizable(True)
        self.body = QWidget()
        self.layout = QVBoxLayout(self.body)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.scroll.setWidget(self.body)
        self.right.addWidget(self.scroll, 1)
        self.active = False
        window.content_stack.currentChanged.connect(self.refresh)
        self.refresh()

    def refresh(self, *_args):
        w = self.window
        category = {"phase_model": "calibration"}.get(
            w._current_category, w._current_category
        )
        active = category == "calibration"
        if active != self.active:
            destination = self.layout if active else self.right
            for widget in self.widgets:
                hidden = widget.isHidden()
                source = widget.parentWidget().layout()
                source.removeWidget(widget)
                destination.addWidget(widget, 1 if widget is w.content_stack else 0)
                widget.setHidden(hidden)
            self.active = active
        self.line.setVisible(active)
        self.scroll.setVisible(active)
        if active:
            selection = w._saved_method
            method = METHOD_NAMES.get(selection.active, "Method not chosen")
            series = w.phase_model_workspace.series_selector.currentText()
            run = w.run_combo.currentText() or "No run loaded"
            self.line.setText(f"{series or 'No series selected'} · {run} · {method}")
            self.line.setToolTip(w.method_badge.text())
            # The original selector and per-row scope remain available in the
            # scrolling row, rather than becoming permanent header controls.
            w.method_badge.hide()
        w.setup_series_label.setVisible(not active)
        w.run_path.setVisible(not active and not w.run_combo.isHidden())

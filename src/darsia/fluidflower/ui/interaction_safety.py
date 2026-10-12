"""Application-wide guards against accidental Qt input changes."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QComboBox, QWidget


class ComboBoxWheelGuard(QObject):
    """Prevent passive page scrolling from changing editable controls."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._armed_spin_box: QAbstractSpinBox | None = None

    @staticmethod
    def _combo_box(widget: Any) -> QComboBox | None:
        current = widget if isinstance(widget, QWidget) else None
        while current is not None:
            if isinstance(current, QComboBox):
                return current
            current = current.parentWidget()
        return None

    @staticmethod
    def _spin_box(widget: Any) -> QAbstractSpinBox | None:
        current = widget if isinstance(widget, QWidget) else None
        while current is not None:
            if isinstance(current, QAbstractSpinBox):
                return current
            current = current.parentWidget()
        return None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress:
            self._armed_spin_box = self._spin_box(watched)
            return super().eventFilter(watched, event)
        if event.type() != QEvent.Type.Wheel:
            return super().eventFilter(watched, event)
        combo = self._combo_box(watched)
        if combo is not None:
            popup = combo.view()
            if (
                popup.isVisible()
                and isinstance(watched, QWidget)
                and (watched is popup or popup.isAncestorOf(watched))
            ):
                return super().eventFilter(watched, event)
            event.ignore()
            return True
        spin_box = self._spin_box(watched)
        if spin_box is not None and spin_box is not self._armed_spin_box:
            event.ignore()
            return True
        return super().eventFilter(watched, event)


def install_interaction_safety(app: QApplication) -> ComboBoxWheelGuard:
    """Install and retain the global input guard for one application instance."""

    existing = getattr(app, "_fluidflower_combo_wheel_guard", None)
    if isinstance(existing, ComboBoxWheelGuard):
        return existing
    guard = ComboBoxWheelGuard(app)
    app.installEventFilter(guard)
    app._fluidflower_combo_wheel_guard = guard  # type: ignore[attr-defined]
    return guard


__all__ = ["ComboBoxWheelGuard", "install_interaction_safety"]

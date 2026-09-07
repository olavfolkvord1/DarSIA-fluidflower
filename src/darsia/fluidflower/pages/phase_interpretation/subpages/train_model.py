"""Optional adapter for embedding the canonical phase-model studio page."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QWidget

from darsia.fluidflower.ui.model_training import SharedPhaseModelTrainingPage


def build(parent_page: Any) -> QWidget:
    del parent_page
    return SharedPhaseModelTrainingPage()


__all__ = ["build"]

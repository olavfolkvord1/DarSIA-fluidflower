"""Native-only lazy presentation adapter; standalone Studio is unchanged."""

from __future__ import annotations

import sys
import traceback

from darsia.fluidflower.pages.phase_interpretation.subpages.train_model import (
    PhaseModelStudioSubpage,
)


class NativePhaseModelStudioSubpage(PhaseModelStudioSubpage):
    def __init__(self, series_id="", project_root=None, *, step_options=None):
        super().__init__(series_id, project_root)
        self.step_options = dict(step_options or {})
        self.native_steps = None
        self.native_numbered_steps = True
        self._optical_basis_notice.hide()

    def _load_studio(self):
        """Keep the original lazy/close lifecycle, supplying a native layout."""
        self._load_pending = False
        if self.studio is not None:
            return
        self._load_failed = False
        self._retry_button.hide()
        self._placeholder.setText("Opening the phase-model steps…")
        try:
            from darsia.fluidflower.ui.model_training import (
                SharedPhaseModelTrainingPage,
            )
            from darsia.fluidflower.native_phase_steps import NativePhaseSteps

            studio = SharedPhaseModelTrainingPage()
            options = dict(self.step_options)
            before_attach = options.pop("before_attach", None)
            if before_attach is not None:
                before_attach()
            steps = NativePhaseSteps(studio, **options)
            steps.set_context(self.project_root, self.series_id)
        except Exception as error:
            self._load_failed = True
            if sys.stderr is not None:
                traceback.print_exc()
            self._placeholder.setText(
                f"Could not open phase-model steps: {type(error).__name__}: {error}"
            )
            self._retry_button.show()
            return
        self.studio, self.native_steps = studio, steps
        studio.setParent(self)
        studio.hide()
        self._layout.removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self._layout.removeWidget(self._retry_button)
        self._retry_button.deleteLater()
        self._layout.addWidget(steps)
        self._install_close_guard()
        self.studio_loaded.emit()

    def bind_context(self, series_id, project_root):
        self.series_id, self.project_root = str(series_id), project_root
        if self.native_steps is not None:
            self.native_steps.set_context(project_root, series_id)

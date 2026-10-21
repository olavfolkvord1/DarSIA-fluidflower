"""One explicit painting action: prepare, check, review, confirm, then adopt."""
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

from PySide6.QtCore import QObject, QTimer

from .painting_series_use_ui import matches_version


_VERSION_KEYS = ('folder', 'sha256', 'name', 'kind', 'run_id', 'files', 'proposal_method')
_BACKEND_MODULES = ('darsia.fluidflower.painted_adoption_table',
    'darsia.fluidflower.painted_adoption_geometry', 'darsia.fluidflower.raw_shape_basis_reader')


def automatic_backend_available():
    """Report installed capability only; every operation still runs real admission."""
    return all(find_spec(name) is not None for name in _BACKEND_MODULES)


def _version_identity(record):
    """Keep the literal persisted identity separate from mutable display labels."""
    return {key: deepcopy(record.get(key)) for key in _VERSION_KEYS}


class SeriesAdoptionController(QObject):
    """Chain existing GUI workers without granting authority from status metadata."""

    def __init__(self, view):
        """Retain no scientific inputs and start no I/O before the user's action."""
        super().__init__(view)
        self.view = view
        self.available = automatic_backend_available()
        self.context = self.preparation = self.dialog = None
        self.error = None
        self._suspended = False
        self._confirmation_pending = False
        self._abandoned_preparation = False
        self._confirm_timer = QTimer(self)
        self._confirm_timer.setSingleShot(True)
        self._confirm_timer.timeout.connect(self._confirm)

    def start(self, record, selection, _displayed_preparation=None):
        """Always run the real preparation; a displayed old record is no shortcut."""
        if not self.available:
            self.view.use_map_progress.setText(
                'The rigs need their RAW basis and the automatic adoption backend before the map can be adopted.')
            return False
        if (self.context is not None or self.view.busy()
                or getattr(self.view, "_adoption_thread", None) is not None):
            self.view.use_map_progress.setText('Wait for the current adoption operation to finish.')
            return False
        self._suspended = False
        self._abandoned_preparation = False
        if self.dialog is not None:
            old_dialog = self.dialog
            old_dialog.close()
            if self.view._adoption_dialog is old_dialog:
                self.view._adoption_dialog = None
            self.dialog = None
        self.preparation = self.error = None
        try:
            self.view._read_approvals(str(record['run_id']))
            approval = self.view._approval_for_version(record)
            if not approval:
                raise ValueError('Approve this exact saved version in step 4 before adoption.')
            frame = self.view.current_frame()
            if frame is None:
                raise ValueError('The selected saved map needs its current rig frame.')
            self.context = dict(project=self.view.project,
                project_root=str(self.view.project.root),
                version=_version_identity(record), approval=deepcopy(approval),
                frame=deepcopy(frame.to_dict()), selection=deepcopy(selection))
            if not self.view.prepare_adoption_of(record, selection.index_path):
                self.context = None
                self.view.use_map_progress.setText(self.view.versions_note.text())
                return False
            return True
        except Exception as error:
            self.context = None
            self.view.use_map_progress.setText(f'The adoption could not start: {error}')
            return False

    def prepared(self, result):
        """Keep the actual preparation until its original worker has retired."""
        if self._abandoned_preparation:
            self._abandoned_preparation = False
            return True
        if self.context is None:
            return False
        try:
            record = self.context['version']
            if (not matches_version(result, record)
                    or result['map_version'].get('approval') != self.context['approval']
                    or result.get('reference_run') != record['run_id']
                    or 'painting_facies' not in result):
                raise ValueError('The preparation does not bind this exact approved painting version.')
            bound = result['geometry_index']
            selected = self.context['selection']
            if Path(bound['path']) != Path(selected.index_path):
                raise ValueError('The preparation used another measured geometry index.')
            if getattr(selected, 'sha256', '') and (
                    bound.get('sha256') != selected.sha256
                    or bound.get('size_bytes') != selected.size_bytes):
                raise ValueError('The selected series geometry changed during preparation.')
            self.preparation = deepcopy(result)
            return True
        except (KeyError, TypeError, ValueError) as error:
            self.error = str(error)
            self.preparation = None
            return True

    def failed(self, message):
        """Retain a plain error and let the valid owned worker retire naturally."""
        if self.context is not None:
            self.error, self.preparation = str(message), None

    def _current(self):
        """Check current GUI context; scientific admission remains with the backend."""
        context = self.context
        if context is None or self._suspended:
            return False
        view = self.view
        try:
            view._read_approvals(str(context['version']['run_id']))
            selected = view._selected_version()
            frame = view.current_frame()
            return bool(view.project is context['project']
                and str(view.project.root) == context['project_root']
                and selected is not None and _version_identity(selected) == context['version']
                and view._approval_for_version(selected) == context['approval']
                and frame is not None and frame.to_dict() == context['frame']
                and view.run_preview.geometry_selection() == context['selection'])
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return False

    def preparation_retired(self):
        """Open only the automatic existing dialog after native worker retirement."""
        if self.context is None:
            return
        if self._suspended or self.error or self.preparation is None or not self._current():
            message = self.error or 'The selected version, approval or geometry changed; start adoption again.'
            self.view.use_map_progress.setText('The map was not adopted: ' + message)
            self.context = None
            return
        from .map_adoption_view import AdoptionDialog
        from .series_geometry import default_actor
        try:
            dialog = AdoptionDialog(self.view.project, self.preparation['folder'],
                actor=self.view._reviewer() or default_actor(), parent=self.view,
                automatic_only=True,
                adoption_scope_note=(self.preparation.get('gui_adoption_cohort') or {}).get('summary', ''))
            dialog.check_button.hide()
            dialog.save_button.hide()
            dialog.operation_retired.connect(self._dialog_retired)
            dialog.confirmation_decided.connect(self._confirmation_decided)
            dialog.finished.connect(self._dialog_closed)
            self.dialog = self.view._adoption_dialog = dialog
            dialog.show()
            if not dialog.check():
                raise ValueError(dialog.status.text())
            self.view.use_map_progress.setText('Checking the automatic facies table and the saved sources…')
        except Exception as error:
            self.view.use_map_progress.setText(f'The map was not adopted: {error}')
            self.context = None

    def _dialog_retired(self, operation):
        """Review only a fully retired check; report a save from its actual receipt."""
        if self.context is None or self._suspended:
            return
        dialog = self.dialog
        if operation == 'check':
            if dialog.candidate is None:
                self.view.use_map_progress.setText(dialog.status.text())
                self.context = None
                return
            if not self._current():
                dialog.status.setText('The selected version, approval or geometry changed. Nothing was adopted.')
                self.view.use_map_progress.setText(dialog.status.text())
                self.context = None
                return
            # Return once to the GUI so the checked table is painted before
            # the existing final human-confirmation question is shown.
            self._confirmation_pending = True
            self._confirm_timer.start(0)
        elif operation == 'save':
            if dialog.revision is None:
                self.view.use_map_progress.setText(dialog.status.text())
            else:
                self.view.use_map_progress.setText(
                    f"Adopted this map as geometry revision {dialog.revision['revision_id']}. "
                    'Apply and Build have not run.')
                self.view.series_use_status.invalidate()
            self.context = None
            self.view._refresh_versions()

    def _confirm(self):
        """Ask once, only for the still-current checked automatic candidate."""
        if not self._confirmation_pending:
            return
        self._confirmation_pending = False
        if not self._current() or self.dialog is None or not self.dialog.isVisible():
            self.view.use_map_progress.setText('The adoption context changed. Nothing was adopted.')
            self.context = None
            return
        self.dialog._save_clicked()

    def _confirmation_decided(self, accepted):
        """A declined final question leaves all Store revisions unchanged."""
        if accepted and not self._current():
            # The final question has a nested event loop. A queued context
            # change during that loop must not save the previous candidate.
            self.dialog.candidate = None
            self.dialog.status.setText('The selected version, approval or geometry changed. Nothing was adopted.')
            self.view.use_map_progress.setText(self.dialog.status.text())
            self.context = None
        elif not accepted:
            self.view.use_map_progress.setText('Adoption cancelled. The checked preparation is retained; nothing was adopted.')
            self.context = None

    def _dialog_closed(self, _result):
        """Ignore any pending confirmation after the user closes its result window."""
        self._confirmation_pending = False
        self._confirm_timer.stop()
        if self.dialog is not None and not self.dialog.busy():
            self.context = None

    def wait_for_workers(self):
        """Suppress continuation and naturally retire any owned dialog worker."""
        self._abandoned_preparation = self.context is not None and self.dialog is None
        self._suspended = True
        self._confirmation_pending = False
        self._confirm_timer.stop()
        if self.dialog is not None:
            self.dialog.wait_for_workers()
        self.context = None

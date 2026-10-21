"""Prepare typed Setup metadata from the current workspace and existing Jobs choices."""
from __future__ import annotations

import json
from pathlib import Path
import traceback
import weakref

from PySide6.QtCore import QMetaObject, QObject, Qt, Slot


def current_setup_job_context(workspace, *, include_identity=True):
    """Capture actual saved workspace and Jobs choices without inventing locations."""
    project = workspace.project
    root = workspace._loaded_root
    if project is None or root is None or workspace._loaded_series != workspace.series_id:
        raise ValueError("The selected series has no current loaded geometry workspace")
    current = Path(project.root) / "overrides" / "geometry_facies" / "current.json"
    with current.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("The adopted geometry revision exceeds the metadata limit")
    revision = json.loads(raw.decode("utf-8-sig"))
    if (revision.get("format") != "darsia-fluidflower-geometry-facies-v1"
            or revision.get("schema") != 1 or not revision.get("revision_id")
            or (revision.get("configuration") or {}).get("frame") != "rig-corrected"):
        raise ValueError("Setup needs the current saved corrected geometry revision")
    window = workspace.window()
    jobs = getattr(window, "jobs_workspace", None)
    if jobs is None:
        ensure = getattr(window, "ensure_jobs", None)
        jobs = ensure() if callable(ensure) else None
    if jobs is None:
        raise ValueError("The existing Jobs workspace is unavailable")
    if jobs.series_root is None or Path(jobs.series_root) != Path(root):
        raise ValueError("Jobs is bound to another loaded series")
    job_root = jobs.new_analysis.form.job_edit.text().strip()
    if not job_root:
        raise ValueError("Jobs has no fresh shared job folder recorded")
    from .pipeline.job_queue import name, require_unc
    require_unc(job_root)
    selected = jobs.panel.target_combo.currentData() or {}
    target = selected.get("machine")
    if not target:
        raise ValueError("Jobs has no selected worker target")
    if jobs.panel.backend_combo.currentData() != "cpu":
        raise ValueError("The selected Jobs backend must be CPU for Setup")
    if selected.get("local"):
        report = getattr(jobs.stores, "report", None) or {}
        if report.get("machine_id"):
            target = report["machine_id"]
    context = dict(project=project, series_root=Path(root), target_machine=name(target),
                   job_root=Path(job_root), adopted_revision_id=revision["revision_id"])
    if include_identity:
        from .analysis_preparation import job_identity
        identity = job_identity(Path(__file__).resolve().parents[3])
        context.update(commit=identity["commit"], environment_lock=identity["environment_lock"])
    stores_path = getattr(jobs.stores, "path", None)
    if stores_path is not None:
        context["stores_path"] = stores_path
    return context


def _prepare_current(project, series_root, **choices):
    """Call only the current-series metadata assembler after the explicit action."""
    from .series_setup_plan import prepare_current_series_setup
    return prepare_current_series_setup(project, series_root, **choices)


class SeriesSetupJobsController(QObject):
    """Bind the existing action to finite metadata preparation and original Jobs."""

    def __init__(self, workspace, view, *, context_provider=None, preparer=None):
        """Connect existing status signals while preserving the view's busy boundary."""
        super().__init__(view)
        self._workspace, self._view = weakref.ref(workspace), weakref.ref(view)
        self.context_provider = context_provider or current_setup_job_context
        self._default_context = context_provider is None
        self.preparer = preparer or _prepare_current
        self._thread = self._worker = None
        self._starting = False
        self._context = self._key = self._outcome = None
        self._wired_jobs = self._wired_dialog = None
        self._original_reason = view.apply_build_reason.text()
        self._original_busy = weakref.WeakMethod(view.busy)
        view.busy = self.busy
        view.apply_build_button.setText("Prepare Apply and Build job")
        view.apply_build_button.clicked.connect(self.prepare)
        view.busy_changed.connect(self.refresh)
        workspace.reference_changed.connect(self.refresh)
        status = getattr(view, "series_use_status", None)
        if status is not None:
            status.changed.connect(self.refresh)
        self.refresh()

    def busy(self):
        """Keep the original view busy while its metadata callback is retained."""
        original = self._original_busy()
        return self._starting or self._thread is not None or bool(original and original())

    def _jobs(self):
        """Use the native window's existing Jobs owner and call runner."""
        window = self._workspace().window()
        jobs = getattr(window, "jobs_workspace", None)
        if jobs is None and callable(getattr(window, "ensure_jobs", None)):
            jobs = window.ensure_jobs()
        if jobs is None:
            raise ValueError("The existing Jobs workspace is unavailable")
        return jobs

    def _wire_jobs_choices(self):
        """Observe each existing Jobs owner once, without creating another owner."""
        workspace = self._workspace()
        jobs = getattr(workspace.window(), "jobs_workspace", None) if workspace is not None else None
        if jobs is None or (self._wired_jobs is not None and self._wired_jobs() is jobs):
            return
        self._wired_jobs = weakref.ref(jobs)
        jobs.new_analysis.form.job_edit.textChanged.connect(self.refresh)
        jobs.panel.target_combo.currentIndexChanged.connect(self.refresh)
        jobs.panel.backend_combo.currentIndexChanged.connect(self.refresh)
        jobs.stores.check_button.clicked.connect(self._watch_store_check)
        jobs.stores.settings_button.clicked.connect(self._watch_store_settings)

    @Slot()
    def _watch_store_check(self):
        """Refresh after the existing availability writer publishes its report."""
        jobs = self._wired_jobs() if self._wired_jobs is not None else None
        task = getattr(jobs.stores, "_thread", None) if jobs is not None else None
        if task is not None:
            task.finished.connect(self.refresh, Qt.ConnectionType.QueuedConnection)
            if task.isFinished():
                QMetaObject.invokeMethod(self, "refresh", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _watch_store_settings(self):
        """Observe the existing settings dialog's actual saved-path signal once."""
        jobs = self._wired_jobs() if self._wired_jobs is not None else None
        dialog = getattr(jobs.stores, "_dialog", None) if jobs is not None else None
        if dialog is not None and (self._wired_dialog is None or self._wired_dialog() is not dialog):
            self._wired_dialog = weakref.ref(dialog)
            dialog.saved.connect(self.refresh)

    @Slot()
    def refresh(self, *_):
        """Offer preparation only for the currently captured genuine context."""
        workspace, view = self._workspace(), self._view()
        if workspace is None or view is None:
            return
        self._wire_jobs_choices()
        view.apply_build_button.setEnabled(False)
        if self._starting or self._thread is not None:
            return
        original = self._original_busy()
        if original and original():
            view.apply_build_reason.setText("Wait for the current geometry operation before preparing Setup metadata.")
            return
        try:
            context = (self.context_provider(workspace, include_identity=False)
                       if self._default_context else self.context_provider(workspace))
            self._wire_jobs_choices()
            if context is None:
                self._context = None
                view.apply_build_reason.setText(self._original_reason)
                return
            if (context["project"] is not workspace.project
                    or Path(context["series_root"]) != Path(workspace._loaded_root)):
                raise ValueError("The Setup context belongs to another loaded series")
            key = (str(context["project"].root), str(context["series_root"]),
                   str(context["job_root"]), context["target_machine"], context.get("commit"),
                   json.dumps(context.get("environment_lock"), sort_keys=True),
                   str(context.get("stores_path") or ""), context.get("adopted_revision_id"))
        except Exception as error:
            self._wire_jobs_choices()
            self._context = None
            view.apply_build_reason.setText(str(error))
            return
        if key != self._key:
            self._outcome = None
        self._context, self._key = dict(context), key
        view.apply_build_button.setEnabled(True)
        view.apply_build_reason.setText(self._outcome or
            "Prepare the current series' Setup job metadata, then start its workers in Jobs to run Apply and Build.")

    @Slot()
    def prepare(self):
        """Freeze existing choices and prepare metadata without starting a process."""
        self.refresh()
        view, workspace = self._view(), self._workspace()
        if view is None or workspace is None or not view.apply_build_button.isEnabled():
            return
        context = dict(self._context)
        try:
            if self._default_context:
                context = dict(self.context_provider(workspace))
            jobs = self._jobs()
            if jobs.series_root is None or Path(jobs.series_root) != Path(context["series_root"]):
                raise ValueError("Jobs is bound to another loaded series")
            series_id = jobs.series_id
        except Exception:
            self._failed(traceback.format_exc())
            return
        preparer = self.preparer

        def prepare_metadata():
            try:
                choices = {key: context[key] for key in
                           ("target_machine", "job_root", "commit", "environment_lock")}
                if "stores_path" in context:
                    choices["stores_path"] = context["stores_path"]
                registration = preparer(context["project"], context["series_root"], **choices)
                return dict(context=context, series_id=series_id, registration=registration)
            except Exception:
                return dict(error=traceback.format_exc())

        self._starting = True
        view.apply_build_button.setEnabled(False)
        view.apply_build_reason.setText("Preparing Setup job metadata… No worker has been started.")
        view.busy_changed.emit(True)
        self._thread, self._worker = jobs.panel._start_call(prepare_metadata, self._ready, self._closed)

    @Slot(object)
    def _ready(self, value):
        """Register the captured job and open Jobs only after preparation succeeds."""
        if value.get("error"):
            self._failed(value["error"])
            return
        try:
            context, registration = value["context"], value["registration"]
            if not isinstance(registration, dict) or registration.get("complete") is not True:
                raise ValueError("Setup metadata preparation did not report a complete job")
            root = Path(context["job_root"])
            if registration.get("root") is not None and Path(registration["root"]) != root:
                raise ValueError("Prepared Setup job belongs to another selected folder")
            if registration.get("job") is not None and Path(registration["job"]["path"]) != root / "job.json":
                raise ValueError("Prepared Setup job descriptor belongs to another selected folder")
            jobs = self._jobs()
            jobs.registry.register_job(context["series_root"], value["series_id"], root, activate=True)
            if jobs.series_root is not None and Path(jobs.series_root) == Path(context["series_root"]):
                jobs.refresh_registered()
            self._outcome = "Setup job metadata created. Start its workers in Jobs to run Apply and Build."
            if "selected_runs" in registration and "remaining_runs" in registration:
                selected, remaining = registration["selected_runs"], registration["remaining_runs"]
                self._outcome = (
                    f"Setup job metadata created for {len(selected)} selected run(s): "
                    + (", ".join(map(str, selected)) or "none")
                    + f". Remaining series runs ({len(remaining)}): "
                    + (", ".join(map(str, remaining)) or "none")
                    + ". Start its workers in Jobs to run Apply and Build.")
            self._view().apply_build_reason.setText(self._outcome)
            jobs.status_label.setText(self._outcome)
            self._workspace().window().show_jobs()
        except Exception:
            self._failed(traceback.format_exc())

    def _failed(self, text):
        """Keep the literal preparation error beside the action and its traceback in Jobs."""
        self._outcome = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), text)
        view = self._view()
        if view is not None:
            view.apply_build_reason.setText(self._outcome)
        workspace = self._workspace()
        window = workspace.window() if workspace is not None else None
        jobs = getattr(window, "jobs_workspace", None)
        if jobs is not None:
            jobs.panel._error(text)
        elif callable(getattr(window, "print_log", None)):
            window.print_log(text)

    @Slot()
    def _closed(self):
        """Retire the finite call before allowing workspace navigation again."""
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self._starting = False
        view = self._view()
        if view is not None:
            view.busy_changed.emit(False)
            self.refresh()

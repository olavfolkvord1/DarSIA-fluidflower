"""Read-only saved mass-calibration metadata for the native Calibration page.

Only explicit paths and an optional, already bound current result are read.
Nothing here fits a calibration, calculates a metric, changes parameters, starts
jobs, discovers history, opens images/NPZ, or writes any file.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

NOT_REPORTED = "Not reported"
NOT_IN_CALIBRATION = "Not in this calibration document"
MAX_JSON_BYTES = 1024 * 1024
MAX_CSV_BYTES = 256 * 1024
MAX_SIGNAL_FILES = 64
MAX_DIRECTORY_ENTRIES = 256
MAX_SIGNAL_ROWS = 128


@dataclass(frozen=True)
class SavedMassCalibrationData:
    # (parameter or metric name, literal saved value, scope/provenance)
    rows: tuple[tuple[str, str, str], ...]
    source_label: str
    notes: tuple[str, ...] = ()


def _text(value) -> str:
    """Format a saved scalar only; no derived quantity or guessed default."""
    if value is None or isinstance(value, (bool, Mapping, list, tuple)):
        return NOT_REPORTED
    if isinstance(value, float) and not math.isfinite(value):
        return NOT_REPORTED
    return str(value)


def _path(reference, series_root) -> Path | None:
    value = reference.get("path") if isinstance(reference, Mapping) else reference
    if value is None or not str(value).strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        if series_root is None:
            raise ValueError("A relative reference needs the selected series folder.")
        path = Path(series_root) / path
    return path


def _small_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Metadata exceeds the display limit ({limit} bytes): {path}")
    return raw


def _json(path: Path) -> dict:
    value = json.loads(_small_bytes(path, MAX_JSON_BYTES))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a saved JSON object: {path}")
    return value


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _error_row(values, source: str) -> tuple[str, str, str] | None:
    """Use known saved error fields verbatim; never derive an error from mass."""
    metrics = _mapping(values)
    for key in ("mean_run_nmae", "nmae", "nrmse", "rmse_g"):
        value = _text(metrics.get(key))
        if value != NOT_REPORTED:
            units = "g" if key == "rmse_g" else "saved fraction"
            return ("Saved mass-balance error", f"{key} = {value} ({units})", source)
    return None


def _bound_matching_result(reference, matching_result, notes) -> Mapping:
    """Caller supplies an already matched result, never an arbitrary selection.

    When the explicit reference also has a SHA and the result declares its SHA,
    reject disagreement. This does not claim to verify a file's content digest.
    """
    result = _mapping(matching_result)
    expected = _mapping(reference).get("sha256")
    found = (
        _mapping(result.get("bindings")).get("calibration_sha256")
        or _mapping(result.get("scalar_result")).get("calibration_sha256")
        or _mapping(result.get("calibration")).get("sha256")
        or result.get("calibration_sha256")
    )
    if expected and found and expected != found:
        notes.append(
            "The supplied result declares another calibration; its frame values are not shown."
        )
        return {}
    return result


def _phase_display(reference, series_root, matching_result) -> SavedMassCalibrationData:
    rows, notes, saved = [], [], {}
    try:
        path = _path(reference, series_root)
        if path is not None:
            saved = _json(path)
        else:
            notes.append("No phase-model mass calibration is referenced.")
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        path = None
        notes.append(str(error))
    source = (
        str(path)
        if path is not None
        else "No readable explicit phase calibration reference"
    )
    frozen_round4 = saved.get("contract") == "ffac-common-mass-round3-frozen-v1" and saved.get("frozen") is True
    absent = NOT_IN_CALIBRATION if frozen_round4 else NOT_REPORTED
    parameters = _mapping(saved.get("parameters"))
    if not parameters:
        parameters = _mapping(_mapping(saved.get("calibration")).get("parameters"))
    if not parameters:
        parameters = _mapping(_mapping(saved.get("scalar_result")).get("parameters"))
    for title, key in (("Gain", "gain"), ("Beta", "beta")):
        rows.append(
            (title, _text(parameters.get(key)) if key in parameters else absent, "Shared saved calibration; " + source)
        )

    result = _bound_matching_result(reference, matching_result, notes)
    # A full result may have many frames. Never silently choose its first frame.
    frame = _mapping(result.get("frame"))
    if not frame and "frames" not in result:
        frame = result
    scalar = _mapping(frame.get("scalar_result"))
    factor = frame.get("dilution_factor")
    if factor is None:
        factor = scalar.get("dilution_factor")
    if factor is None:
        factor = _mapping(frame.get("prediction")).get("dilution_factor")
    anchor = _mapping(frame.get("anchor"))
    if not anchor:
        anchor = _mapping(result.get("anchor"))
    anchor_fields = {
        key: anchor[key]
        for key in ("elapsed_h", "affected_pore_volume_m3")
        if _text(anchor.get(key)) != NOT_REPORTED
    }
    anchor_text = (
        json.dumps(anchor_fields, ensure_ascii=False) if anchor_fields else absent
    )
    frame_source = "Selected saved image"
    rows.extend(
        (
            ("Dilution factor", _text(factor) if factor is not None else absent, "Per image; " + frame_source),
            (
                "Anchor",
                anchor_text,
                "Per run: time [h], affected pore volume [m3]; " + frame_source,
            ),
        )
    )
    error = None
    for metrics, where in (
        (saved.get("metrics"), source),
        (_mapping(saved.get("calibration")).get("metrics"), source),
        (
            _mapping(result.get("calibration")).get("metrics"),
            "Saved result calibration",
        ),
        (result.get("metrics"), "Saved result"),
    ):
        error = _error_row(metrics, where)
        if error is not None:
            break
    rows.append(
        error
        or (
            "Saved mass-balance error",
            absent,
            "No error is reported with this saved calibration.",
        )
    )
    notes.append(
        "Gain and beta are shared by the calibration. View a saved result under Analysis for its per-image dilution factor and per-run anchor."
    )
    notes.append(
        "The fit error describes agreement with the calibration observations. A frozen calibration may have no error metric reported."
    )
    return SavedMassCalibrationData(tuple(rows), source, tuple(notes))


def _signal_rows(folder: Path, notes) -> list[tuple[str, str, str]]:
    signal_folder = folder / "signal_model"
    selected = []
    try:
        for count, path in enumerate(signal_folder.iterdir(), 1):
            if count > MAX_DIRECTORY_ENTRIES:
                raise ValueError(
                    "Signal-model folder exceeds the bounded display listing."
                )
            if (
                path.name.startswith("signal_model_")
                and path.suffix.lower() == ".csv"
                and path.is_file()
            ):
                selected.append(path)
                if len(selected) > MAX_SIGNAL_FILES:
                    raise ValueError("Too many signal CSV files for this display.")
    except (OSError, ValueError) as error:
        notes.append(str(error))
        return []
    rows = []
    for path in sorted(selected, key=lambda item: item.name):
        try:
            text = _small_bytes(path, MAX_CSV_BYTES).decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            if not {"supports", "values"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Signal CSV lacks supports/values: {path}")
            supports, values = [], []
            for index, row in enumerate(reader, 1):
                if index > MAX_SIGNAL_ROWS:
                    raise ValueError(
                        f"Signal CSV exceeds the display row limit: {path}"
                    )
                supports.append(str(row.get("supports") or ""))
                values.append(str(row.get("values") or ""))
            label = path.stem.removeprefix("signal_model_")
            rows.append(
                (f"Signal supports (facies {label})", json.dumps(supports), str(path))
            )
            rows.append(
                (f"Signal values (facies {label})", json.dumps(values), str(path))
            )
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(str(error))
    return rows


def _same_path(first: Path, second: Path) -> bool:
    # Absolute explicit paths only. No is_relative_to or filesystem traversal.
    return os.path.abspath(first).casefold() == os.path.abspath(second).casefold()


def _colour_display(
    reference, series_root, optuna_summary_reference
) -> SavedMassCalibrationData:
    rows, notes, flash = [], [], {}
    try:
        folder = _path(reference, series_root)
    except (ValueError, TypeError) as error:
        folder = None
        notes.append(str(error))
    source = (
        str(folder)
        if folder is not None
        else "No explicit colour-path calibration folder"
    )
    if folder is not None:
        rows.extend(_signal_rows(folder, notes))
        try:
            flash = _json(folder / "flash" / "flash.json")
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(str(error))
    if not rows:
        rows.append(("Signal values by facies", NOT_REPORTED, source))
    for key in ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g"):
        rows.append(
            ("Flash " + key, _text(flash.get(key)), source + "/flash/flash.json")
        )
    report_error = None
    if optuna_summary_reference:
        try:
            report_path = _path(optuna_summary_reference, series_root)
            report = _json(report_path)
            bound_folder = _path(report.get("calibration_folder"), series_root)
            if (
                folder is None
                or bound_folder is None
                or not _same_path(folder, bound_folder)
            ):
                raise ValueError(
                    "The explicit Optuna report is not bound to the selected colour calibration folder."
                )
            date = _text(report.get("finished"))
            saved = report.get("saved")
            notes.append(
                f"Explicit Optuna report: {report_path}; finished: {date}; saved: {saved!r}."
            )
            notes.append(
                "Its error and best values describe that report, not a new measurement of current manual slider values."
            )
            notes.append(
                "Report configuration: "
                + json.dumps(report.get("configuration"), ensure_ascii=False)
            )
            if saved is not True:
                notes.append(
                    "The report does not say its best values were adopted into the calibration."
                )
            percentage = _text(report.get("mean_error_percent_best"))
            if percentage != NOT_REPORTED:
                report_error = (
                    "Saved mass-balance error (Optuna report)",
                    percentage + " %",
                    f"Historical report at {date}; {report_path}; saved={saved!r}",
                )
            elif _text(report.get("objective_best")) != NOT_REPORTED:
                report_error = (
                    "Saved objective_best (Optuna report)",
                    _text(report["objective_best"]),
                    "Report objective: "
                    + str(
                        _mapping(report.get("settings")).get("objective", NOT_REPORTED)
                    ),
                )
            for label, values in _mapping(report.get("values_best")).items():
                rows.append(
                    (
                        f"Report best signal values (facies {label})",
                        json.dumps(values, ensure_ascii=False),
                        f"Optuna report only; {report_path}",
                    )
                )
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(str(error))
    else:
        notes.append(
            "No Optuna report is explicitly referenced; history folders are not searched."
        )
    rows.append(
        report_error
        or (
            "Saved mass-balance error",
            NOT_REPORTED,
            "No separate mass-balance error is reported with this calibration.",
        )
    )
    return SavedMassCalibrationData(tuple(rows), source, tuple(notes))


def read_mass_calibration_display(
    method: str,
    reference,
    *,
    series_root: str | Path | None = None,
    matching_result: Mapping | None = None,
    optuna_summary_reference=None,
) -> SavedMassCalibrationData:
    """Read bounded, explicitly referenced metadata only.

    ``reference`` is a calibration JSON path for phase_model, a calibration folder
    for colour_path, or an existing {path, sha256, ...} descriptor. Relative paths
    require series_root. The caller may supply a selected frame/full result only
    after matching it to this exact calibration. To select a frame from a full
    result, pass it explicitly as matching_result['frame']; no first-frame default
    is made. Optuna summaries must be explicitly supplied and bind the same folder.
    No library target, old global fit, latest report or result path is followed.
    """
    if method == "phase_model":
        return _phase_display(reference, series_root, matching_result)
    if method == "colour_path":
        return _colour_display(reference, series_root, optuna_summary_reference)
    return SavedMassCalibrationData(
        (("Saved calibration", NOT_REPORTED, "No method is chosen for this series."),),
        "Method: not chosen",
        (
            "Choose the method explicitly under Calibration. This display does not choose one.",
        ),
    )


class MassCalibrationDisplay(QWidget):
    """Reusable read-only table; both future calculation modes stay disabled."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Mass calibration — saved values and saved error"))
        self.source = QLabel()
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.source.setWordWrap(True)
        self.source.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.source)
        modes = QHBoxLayout()
        modes.addWidget(QLabel("Parameters:"))
        self.manual_button = QRadioButton("Manual")
        self.automatic_button = QRadioButton("Automatic (fitted to injected mass)")
        self.mode_buttons = QButtonGroup(self)
        for button in (self.manual_button, self.automatic_button):
            self.mode_buttons.addButton(button)
            button.setEnabled(False)
            modes.addWidget(button)
        layout.addLayout(modes)
        self.mode_notice = QLabel(
            "Manual and automatic calibration runs will be available on the Jobs page. This page shows saved values and reported errors."
        )
        self.mode_notice.setWordWrap(True)
        layout.addWidget(self.mode_notice)
        self.parameters = QTableWidget(0, 3)
        self.parameters.setHorizontalHeaderLabels(
            ("Parameter / metric", "Saved value", "Scope / source")
        )
        self.parameters.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parameters.horizontalHeader().setStretchLastSection(True)
        self.parameters.verticalHeader().hide()
        layout.addWidget(self.parameters, 1)
        self.notes = QLabel()
        self.notes.setTextFormat(Qt.TextFormat.PlainText)
        self.notes.setWordWrap(True)
        self.notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.notes)
        self.set_display(read_mass_calibration_display("", None))

    def set_display(self, display: SavedMassCalibrationData) -> None:
        """Render already loaded metadata; this method performs no I/O."""
        self.source.setText(display.source_label)
        self.notes.setText("\n".join(display.notes))
        self.parameters.setRowCount(len(display.rows))
        for index, values in enumerate(display.rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(str(value))
                self.parameters.setItem(index, column, item)
        self.parameters.resizeColumnsToContents()


OVERFLOW_MODES = ("off", "compare", "constrain", "hard")


def read_overflow_use(reference, series_root=None):
    """How the referenced phase-model calibration used overflow: (mode, explanation).

    mode is one of OVERFLOW_MODES as the calibration document records it, or
    None when nothing is referenced or recorded. A frozen calibration with
    overflow_in_objective = false is compare-only by its contract. Only the
    explicit reference is read.
    """
    try:
        path = _path(reference, series_root)
        if path is None:
            return None, "No phase-model mass calibration is referenced in series.toml."
        saved = _json(path)
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return None, f"The referenced calibration could not be read: {error}"
    if saved.get("overflow_in_objective") is False:
        return "compare", (
            "The referenced calibration keeps overflow out of the fit "
            "(overflow_in_objective = false): overflow is compared, not fitted."
        )
    mode = saved.get("overflow_mode") or _mapping(saved.get("calibration")).get("overflow_mode")
    if mode in OVERFLOW_MODES:
        return mode, f"The referenced calibration records overflow_mode = {mode}."
    return None, "The referenced calibration does not record how it used overflow."


class OverflowChoicePage(QWidget):
    """Free gas vs overflow (N141 spec 2.3): the series calibration's overflow use.

    Off and Compare only show what the referenced calibration does; a different
    choice needs a new calibration. Steer per run is not available yet.
    Constrain stays with the legacy comparison method, unchanged.
    """

    LABELS = {
        "off": "Off",
        "compare": "Compare only — shown in figures and tables, never fitted",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel("Free gas vs overflow")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        self.buttons = QButtonGroup(self)
        self.buttons.setExclusive(True)
        self.choice = {}
        for mode in ("off", "compare"):
            button = QRadioButton(self.LABELS[mode])
            button.setEnabled(False)
            button.setToolTip("Set by the referenced mass calibration; a different choice needs a new calibration.")
            self.buttons.addButton(button)
            self.choice[mode] = button
            layout.addWidget(button)
        self.steer = QRadioButton("Steer per run — Not available yet")
        self.steer.setEnabled(False)
        self.steer.setToolTip(
            "Would adjust each run's detected free gas towards that run's own overflow; not available yet."
        )
        self.buttons.addButton(self.steer)
        layout.addWidget(self.steer)
        self.detail = QLabel()
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        note = QLabel(
            "The Phase-map method never fits its calibration to overflow. The legacy "
            "comparison method keeps its own 'constrain' choice, unchanged."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        self.mode = None

    def set_use(self, mode, explanation) -> None:
        """Show the calibration's overflow use; nothing here writes a setting."""
        self.mode = mode if mode in OVERFLOW_MODES else None
        self.buttons.setExclusive(False)
        for button in (*self.choice.values(), self.steer):
            button.setChecked(False)
        if self.mode in self.choice:
            self.choice[self.mode].setChecked(True)
        self.buttons.setExclusive(True)
        extra = ""
        if self.mode in ("constrain", "hard"):
            extra = " That is the legacy comparison method's constraint, not one of these choices."
        self.detail.setText(explanation + extra)

"""Per-series method choice and its small native UI panel.

Only an explicit save changes ``series.toml``. References are opaque strings;
selecting a method does not load, fit, activate or alter a calibration.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
import os
import tomllib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import toml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


METHODS = ("colour_path", "phase_model")
# Display names of the methods (N141). The keys above and in series.toml stay.
METHOD_NAMES = {"colour_path": "Direct colour method", "phase_model": "Phase-map method"}
METHOD_HEADERS = {"colour_path": "Direct colour method (DarSIA original)", "phase_model": "Phase-map method"}
METHOD_DESCRIPTIONS = {
    "colour_path": "Colour → concentration → mass; phases from colour-signal thresholds (DarSIA original).",
    "phase_model": "Trained phase map first; concentration and mass are computed inside the phases.",
}


class ManifestChangedError(RuntimeError):
    """The manifest changed since this save's original bytes were read."""


@dataclass(frozen=True)
class MethodSelection:
    active: str = ""
    model_version: str = ""
    mass_calibration: str = ""
    colour_calibration: str = ""
    error: str = ""


def missing_method_references(selection: MethodSelection) -> tuple[str, ...]:
    """Required references still missing; no path lookups or inference."""
    required = (
        (
            ("model version", selection.model_version),
            ("mass calibration", selection.mass_calibration),
        )
        if selection.active == "phase_model"
        else (
            (("colour-path calibration", selection.colour_calibration),)
            if selection.active == "colour_path"
            else ()
        )
    )
    return tuple(
        label
        for label, value in required
        if not isinstance(value, str) or not value.strip()
    )


def _reference(table: Mapping, key: str) -> str:
    value = table.get(key)
    return value if isinstance(value, str) else ""


def _selection(document: Mapping) -> MethodSelection:
    method = document.get("method", {})
    if not isinstance(method, Mapping):
        return MethodSelection(error="The method section must be a TOML table.")
    phase = method.get("phase_model", {})
    colour = method.get("colour_path", {})
    if not isinstance(phase, Mapping) or not isinstance(colour, Mapping):
        return MethodSelection(
            error="Method calibration references must be TOML tables."
        )
    phase = phase if isinstance(phase, Mapping) else {}
    colour = colour if isinstance(colour, Mapping) else {}
    active = method.get("active", "")
    return MethodSelection(
        active=active if isinstance(active, str) and active in METHODS else "",
        model_version=_reference(phase, "model_version"),
        mass_calibration=_reference(phase, "mass_calibration"),
        colour_calibration=_reference(colour, "calibration"),
    )


def read_series_method(series_root: str | Path) -> MethodSelection:
    """Read the saved choice without discovery, calibration or writes.

    An absent/unknown choice is not chosen. An unreadable or malformed manifest
    also remains not chosen and carries its error for the UI to show.
    """
    try:
        raw = (Path(series_root) / "series.toml").read_bytes()
        return _selection(tomllib.loads(raw.decode("utf-8-sig")))
    except (OSError, UnicodeError, ValueError) as error:
        return MethodSelection(error=str(error))


def _same_values(first, second) -> bool:
    """Exact TOML semantics, including NaN and an offset's representation."""
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same_values(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, list):
        return len(first) == len(second) and all(
            _same_values(left, right) for left, right in zip(first, second)
        )
    if isinstance(first, float) and math.isnan(first):
        return math.isnan(second)
    if isinstance(first, (dt.datetime, dt.date, dt.time)):
        return first.isoformat() == second.isoformat()
    return first == second


def save_series_method(
    series_root: str | Path,
    selection: MethodSelection,
    *,
    expected_bytes: bytes | None = None,
) -> MethodSelection:
    """Atomically record one explicit method and its reference strings.

    Only the chosen method's reference fields are updated. The inactive method,
    unknown method fields and every unrelated TOML value are retained. Existing
    ``toml`` serialization may normalize comments/formatting; a parse-roundtrip
    must preserve all semantic values. Original bytes are checked immediately
    before replacement so a concurrent edit is rejected rather than overwritten.
    ``expected_bytes`` optionally also rejects changes since a caller's load.
    """
    if selection.active not in METHODS:
        raise ValueError("Choose colour_path or phase_model before saving.")
    for value in (
        selection.model_version,
        selection.mass_calibration,
        selection.colour_calibration,
    ):
        if not isinstance(value, str):
            raise ValueError("Calibration references must be strings.")
    missing = missing_method_references(selection)
    if missing:
        raise ValueError(
            "Provide " + " and ".join(missing) + " before saving this method."
        )
    path = Path(series_root) / "series.toml"
    original = path.read_bytes()
    if expected_bytes is not None and original != expected_bytes:
        raise ManifestChangedError(f"Manifest changed since it was loaded: {path}")
    document = tomllib.loads(original.decode("utf-8-sig"))
    updated = copy.deepcopy(document)
    method = updated.setdefault("method", {})
    if not isinstance(method, dict):
        raise ValueError("The method section must be a TOML table.")
    method["active"] = selection.active
    references = method.setdefault(selection.active, {})
    if not isinstance(references, dict):
        raise ValueError(f"method.{selection.active} must be a TOML table.")
    if selection.active == "phase_model":
        references.update(
            model_version=selection.model_version,
            mass_calibration=selection.mass_calibration,
        )
    else:
        references["calibration"] = selection.colour_calibration

    text = toml.dumps(updated)
    # Preserve the manifest's line ending convention and UTF-8 BOM.
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    raw = b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b""
    raw += text.encode("utf-8")
    parsed = tomllib.loads(raw.decode("utf-8-sig"))
    if not _same_values(updated, parsed):
        raise ValueError("Saving this choice would change other TOML values.")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
        if path.read_bytes() != original:
            raise ManifestChangedError(f"Manifest changed while saving: {path}")
        os.replace(temporary, path)
    finally:
        # This is only the temporary file owned by this explicit save.
        if temporary.exists():
            temporary.unlink()
    return _selection(parsed)


class SeriesMethodPanel(QWidget):
    """UI choice and references; saving belongs to the native window adapter."""

    use_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._available = False
        self._load_error = ""
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel("Method used for this series"))
        self.colour_path_button = QRadioButton(METHOD_NAMES["colour_path"])
        self.phase_model_button = QRadioButton(METHOD_NAMES["phase_model"])
        self.colour_path_button.setToolTip(METHOD_DESCRIPTIONS["colour_path"])
        self.phase_model_button.setToolTip(METHOD_DESCRIPTIONS["phase_model"])
        self.method_buttons = QButtonGroup(self)
        self.method_buttons.addButton(self.colour_path_button, 0)
        self.method_buttons.addButton(self.phase_model_button, 1)
        outer.addWidget(self.colour_path_button)
        colour_fields = QFormLayout()
        self.colour_calibration = QLineEdit()
        self.colour_calibration.setPlaceholderText("No calibration referenced")
        colour_fields.addRow("Calibration", self.colour_calibration)
        outer.addLayout(colour_fields)
        outer.addWidget(self.phase_model_button)
        phase_fields = QFormLayout()
        self.model_version = QLineEdit()
        self.mass_calibration = QLineEdit()
        self.model_version.setPlaceholderText("No model version referenced")
        self.mass_calibration.setPlaceholderText("No mass calibration referenced")
        phase_fields.addRow("Model version", self.model_version)
        phase_fields.addRow("Mass calibration", self.mass_calibration)
        outer.addLayout(phase_fields)
        self.status = QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.use_button = QPushButton("Use for this series")
        self.use_button.clicked.connect(self._request_use)
        outer.addWidget(self.use_button)
        self.colour_path_button.toggled.connect(self._update_controls)
        self.phase_model_button.toggled.connect(self._update_controls)
        for field in (
            self.model_version,
            self.mass_calibration,
            self.colour_calibration,
        ):
            field.textChanged.connect(self._update_controls)
        self.set_selection(MethodSelection(), available=False)

    def set_selection(
        self,
        saved: MethodSelection,
        *,
        phase_model_reference: str = "",
        mass_calibration_reference: str = "",
        colour_calibration_reference: str = "",
        available: bool = True,
    ) -> None:
        """Saved nonempty refs win; only supplied actual selections may seed."""
        self._available = available
        self._load_error = saved.error
        self.method_buttons.setExclusive(False)
        self.colour_path_button.setChecked(saved.active == "colour_path")
        self.phase_model_button.setChecked(saved.active == "phase_model")
        self.method_buttons.setExclusive(True)
        self.model_version.setText(saved.model_version or phase_model_reference)
        self.mass_calibration.setText(
            saved.mass_calibration or mass_calibration_reference
        )
        self.colour_calibration.setText(
            saved.colour_calibration or colour_calibration_reference
        )
        self._update_controls()

    def selection(self) -> MethodSelection:
        active = (
            "phase_model"
            if self.phase_model_button.isChecked()
            else "colour_path" if self.colour_path_button.isChecked() else ""
        )
        return MethodSelection(
            active=active,
            model_version=self.model_version.text(),
            mass_calibration=self.mass_calibration.text(),
            colour_calibration=self.colour_calibration.text(),
        )

    def _update_controls(self, *_args) -> None:
        available = self._available and not self._load_error
        self.colour_path_button.setEnabled(available)
        self.phase_model_button.setEnabled(available)
        self.colour_calibration.setEnabled(
            available and self.colour_path_button.isChecked()
        )
        for field in (self.model_version, self.mass_calibration):
            field.setEnabled(available and self.phase_model_button.isChecked())
        current = self.selection()
        missing = missing_method_references(current)
        self.use_button.setEnabled(
            available and current.active in METHODS and not missing
        )
        self.status.setText(
            self._load_error
            or (
                "Method reference missing: " + " and ".join(missing) + "."
                if self._available and current.active in METHODS and missing
                else (
                    "Save the method and its calibration references for this series."
                    if self._available
                    else "Choose a series on Start."
                )
            )
        )

    def _request_use(self) -> None:
        if self.use_button.isEnabled():
            self.use_requested.emit(self.selection())

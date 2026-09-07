"""Streaming construction of immutable numerical mass-analysis inputs."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darsia.fluidflower.mass_artifacts import (
    MASS_INPUT_DYNAMIC_PRODUCTS,
    MASS_INPUT_STATIC_PRODUCTS,
    write_field_frame,
    write_field_index,
    write_preview_index,
)
from darsia.fluidflower.mass_inputs import MASS_INPUT_MANIFEST_FORMAT
from darsia.fluidflower.project import sha256_file, utc_now

MASS_INPUT_MATERIALIZATION_FORMAT = (
    "darsia-fluidflower-mass-input-materialization-v1"
)

_QUANTITIES = {
    "optical_acidification": "baseline_relative_optical_acidification",
    "phase_co2_probability": "visible_co2_phase_probability",
    "phase_free_gas_probability": "visible_free_gas_phase_probability",
    "depth": "rig_depth",
    "porosity": "porosity",
    "permeability": "permeability",
}
_UNITS = {
    "optical_acidification": "1",
    "phase_co2_probability": "1",
    "phase_free_gas_probability": "1",
    "depth": "m",
    "porosity": "1",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except (FileNotFoundError, OSError):
            pass


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numerical") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _field(values: Any, *, label: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{label} shape {array.shape} differs from {shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numerical")
    output = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{label} contains non-finite values")
    return output


def _probability(values: Any, *, label: str, shape: tuple[int, int]) -> np.ndarray:
    array = _field(values, label=label, shape=shape)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < -1.0e-6 or maximum > 1.0 + 1.0e-6:
        raise ValueError(f"{label} lies outside [0, 1]: [{minimum}, {maximum}]")
    return np.clip(array, 0.0, 1.0)


def _preview_rgb(values: Any, maximum_dimension: int) -> np.ndarray:
    image = np.asarray(values)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Corrected preview requires an RGB image")
    image = np.asarray(image[..., :3], dtype=np.float32)
    if image.size and float(np.max(image)) > 1.5:
        image /= np.float32(255.0)
    image = np.clip(image, 0.0, 1.0)
    height, width = image.shape[:2]
    scale = min(1.0, float(maximum_dimension) / float(max(height, width)))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return np.asarray(np.rint(image * 255.0), dtype=np.uint8)


@dataclass(frozen=True)
class PreparedMassInputFrame:
    """One corrected frame and its synchronized experimental state."""

    frame_id: str
    time_hours: float
    injected_mass_g: float
    pressure_bar: float
    temperature_c: float
    pressure_gradient_bar_per_m: float
    temperature_gradient_c_per_m: float
    optical_acidification: np.ndarray
    phase_co2_probability: np.ndarray
    phase_free_gas_probability: np.ndarray
    corrected_rgb: np.ndarray | None = None
    source_reference: Mapping[str, Any] | None = None


class MassInputRunWriter:
    """Write one run incrementally without retaining prior image fields in RAM."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        phase_candidate_id: str,
        grid: Mapping[str, Any],
        depth: np.ndarray,
        porosity: np.ndarray,
        permeability: np.ndarray,
        permeability_unit: str,
        provenance: Mapping[str, Any] | None = None,
        preview_maximum_dimension: int = 1800,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_id = str(run_id).strip().casefold()
        self.phase_candidate_id = str(phase_candidate_id).strip()
        if not self.run_id or not self.phase_candidate_id:
            raise ValueError("Run and phase candidate identifiers are required")
        self.grid = deepcopy(dict(grid))
        shape_value = self.grid.get("shape_rc") or []
        if len(shape_value) != 2:
            raise ValueError("The registered grid requires shape_rc")
        self.shape = int(shape_value[0]), int(shape_value[1])
        if min(self.shape) <= 0:
            raise ValueError("The registered grid shape must be positive")
        self.preview_maximum_dimension = max(256, int(preview_maximum_dimension))
        self.provenance = deepcopy(dict(provenance or {}))
        self._dynamic_records: dict[str, list[dict[str, Any]]] = {
            product: [] for product in MASS_INPUT_DYNAMIC_PRODUCTS
        }
        self._preview_records: list[dict[str, Any]] = []
        self._times: list[float] = []
        self._injected_mass_g: list[float] = []
        self._state: dict[str, list[float]] = {
            "pressure_bar": [],
            "temperature_c": [],
            "pressure_gradient_bar_per_m": [],
            "temperature_gradient_c_per_m": [],
        }
        self._source_frames: list[dict[str, Any]] = []
        self._finalized = False
        self._static_references = self._write_static_fields(
            depth=depth,
            porosity=porosity,
            permeability=permeability,
            permeability_unit=permeability_unit,
        )

    def _write_static_fields(
        self,
        *,
        depth: np.ndarray,
        porosity: np.ndarray,
        permeability: np.ndarray,
        permeability_unit: str,
    ) -> dict[str, dict[str, Any]]:
        fields = {
            "depth": _field(depth, label="depth", shape=self.shape),
            "porosity": _field(porosity, label="porosity", shape=self.shape),
            "permeability": _field(
                permeability, label="permeability", shape=self.shape
            ),
        }
        if float(np.min(fields["depth"])) < 0.0:
            raise ValueError("Depth cannot be negative")
        porosity_min = float(np.min(fields["porosity"]))
        porosity_max = float(np.max(fields["porosity"]))
        if porosity_min < 0.0 or porosity_max > 1.0 + 1.0e-6:
            raise ValueError("Porosity must lie in [0, 1]")
        if float(np.min(fields["permeability"])) < 0.0:
            raise ValueError("Permeability cannot be negative")
        permeability_unit = str(permeability_unit).strip()
        if not permeability_unit:
            raise ValueError("Permeability unit is required")

        references: dict[str, dict[str, Any]] = {}
        for product in MASS_INPUT_STATIC_PRODUCTS:
            product_root = self.root / self.run_id / product
            record = write_field_frame(
                product_root / "static.npz",
                fields[product],
                run_id=self.run_id,
                product=product,
                frame_id="static-geometry",
                time_hours=0.0,
            )
            references[product] = write_field_index(
                product_root / "index.json",
                run_id=self.run_id,
                product=product,
                quantity=_QUANTITIES[product],
                unit=(
                    permeability_unit if product == "permeability" else _UNITS[product]
                ),
                frames=[record],
                grid=self.grid,
            )
        return references

    def add_frame(self, frame: PreparedMassInputFrame) -> None:
        if self._finalized:
            raise RuntimeError("Cannot add frames after finalization")
        frame_id = str(frame.frame_id).strip()
        if not frame_id:
            raise ValueError("Frame identifier is required")
        time_hours = _finite_number(frame.time_hours, "time_hours")
        if time_hours < 0.0:
            raise ValueError("Frame time cannot be negative")
        if self._times and time_hours <= self._times[-1]:
            raise ValueError("Frame times must be strictly increasing")

        co2 = _probability(
            frame.phase_co2_probability,
            label="phase_co2_probability",
            shape=self.shape,
        )
        gas = _probability(
            frame.phase_free_gas_probability,
            label="phase_free_gas_probability",
            shape=self.shape,
        )
        if float(np.max(gas - co2)) > 1.0e-5:
            raise ValueError("Free-gas probability cannot exceed total CO2 probability")
        fields = {
            "optical_acidification": _field(
                frame.optical_acidification,
                label="optical_acidification",
                shape=self.shape,
            ),
            "phase_co2_probability": co2,
            "phase_free_gas_probability": np.minimum(gas, co2),
        }
        for product, values in fields.items():
            product_root = self.root / self.run_id / product
            record = write_field_frame(
                product_root / f"{frame_id}.npz",
                values,
                run_id=self.run_id,
                product=product,
                frame_id=frame_id,
                time_hours=time_hours,
            )
            self._dynamic_records[product].append(record)

        if frame.corrected_rgb is not None:
            image = _preview_rgb(
                frame.corrected_rgb,
                maximum_dimension=self.preview_maximum_dimension,
            )
            preview_root = self.root / self.run_id / "preview_images"
            preview_root.mkdir(parents=True, exist_ok=True)
            preview_path = preview_root / f"{frame_id}.jpg"
            temporary = preview_root / f".{frame_id}.{uuid.uuid4().hex}.tmp.jpg"
            try:
                if not cv2.imwrite(
                    str(temporary),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 96],
                ):
                    raise OSError(f"Could not write preview image {temporary}")
                os.replace(temporary, preview_path)
            finally:
                try:
                    temporary.unlink()
                except (FileNotFoundError, OSError):
                    pass
            self._preview_records.append(
                {
                    "frame_id": frame_id,
                    "time_hours": time_hours,
                    "path": preview_path.name,
                    "size_bytes": preview_path.stat().st_size,
                    "sha256": sha256_file(preview_path),
                }
            )

        injected_mass_g = _finite_number(frame.injected_mass_g, "injected_mass_g")
        if injected_mass_g < 0.0:
            raise ValueError("Injected mass cannot be negative")
        state_values = {
            "pressure_bar": _finite_number(frame.pressure_bar, "pressure_bar"),
            "temperature_c": _finite_number(
                frame.temperature_c, "temperature_c"
            ),
            "pressure_gradient_bar_per_m": _finite_number(
                frame.pressure_gradient_bar_per_m,
                "pressure_gradient_bar_per_m",
            ),
            "temperature_gradient_c_per_m": _finite_number(
                frame.temperature_gradient_c_per_m,
                "temperature_gradient_c_per_m",
            ),
        }
        if state_values["pressure_bar"] <= 0.0:
            raise ValueError("Pressure must be positive")
        self._times.append(time_hours)
        self._injected_mass_g.append(injected_mass_g)
        for key, value in state_values.items():
            self._state[key].append(value)
        source = deepcopy(dict(frame.source_reference or {}))
        source.update({"frame_id": frame_id, "time_hours": time_hours})
        self._source_frames.append(source)

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("Run has already been finalized")
        if not self._times:
            raise ValueError(f"No mass-input frames were written for {self.run_id}")
        if self._preview_records and len(self._preview_records) != len(self._times):
            raise ValueError("Corrected previews must be supplied for every frame or none")

        references = deepcopy(self._static_references)
        for product in MASS_INPUT_DYNAMIC_PRODUCTS:
            product_root = self.root / self.run_id / product
            references[product] = write_field_index(
                product_root / "index.json",
                run_id=self.run_id,
                product=product,
                quantity=_QUANTITIES[product],
                unit=_UNITS[product],
                frames=self._dynamic_records[product],
                grid=self.grid,
            )
        preview_reference = None
        if self._preview_records:
            preview_reference = write_preview_index(
                self.root / self.run_id / "preview_images" / "index.json",
                run_id=self.run_id,
                frames=self._preview_records,
            )
        self._finalized = True
        row = {
            "run_id": self.run_id,
            "phase_candidate_id": self.phase_candidate_id,
            "target_times_hours": list(self._times),
            "injected_mass_g": list(self._injected_mass_g),
            "geometry": {
                "width_m": _finite_number(self.grid.get("width_m"), "width_m"),
                "height_m": _finite_number(self.grid.get("height_m"), "height_m"),
            },
            "thermodynamic_state": deepcopy(self._state),
            "products": references,
            "source_frames": deepcopy(self._source_frames),
            "provenance": deepcopy(self.provenance),
        }
        if preview_reference is not None:
            row["preview_images"] = preview_reference
        return row


def write_mass_input_manifest(
    path: Path,
    *,
    run_inputs: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Publish the final manifest only after every transitive product exists."""

    rows = [deepcopy(dict(row)) for row in run_inputs]
    if not rows:
        raise ValueError("At least one prepared run is required")
    run_ids = [str(row.get("run_id") or "") for row in rows]
    if any(not value for value in run_ids) or len(run_ids) != len(set(run_ids)):
        raise ValueError("Prepared run identifiers must be unique and non-empty")
    time_grids = [
        [float(value) for value in row.get("target_times_hours") or []]
        for row in rows
    ]
    if any(grid != time_grids[0] for grid in time_grids[1:]):
        raise ValueError("Every run in one manifest must share the same target times")
    payload = {
        "schema": 1,
        "format": MASS_INPUT_MANIFEST_FORMAT,
        "materialization_format": MASS_INPUT_MATERIALIZATION_FORMAT,
        "created_utc": utc_now(),
        "upstream": deepcopy(dict(upstream)),
        "run_inputs": rows,
        "provenance": deepcopy(dict(provenance or {})),
    }
    path = Path(path).expanduser().resolve()
    _atomic_json(path, payload)
    return path


__all__ = [
    "MASS_INPUT_MATERIALIZATION_FORMAT",
    "MassInputRunWriter",
    "PreparedMassInputFrame",
    "write_mass_input_manifest",
]

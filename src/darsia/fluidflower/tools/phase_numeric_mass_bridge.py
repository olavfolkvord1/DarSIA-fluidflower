"""Consume final numerical phase fields without image decoding or inference.

The caller must provide measured geometry, pressure/temperature, injection
history and their provenance. Indicator calibration remains a separate mass
analysis input; this bridge never turns a phase label into concentration.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from darsia.fluidflower.mass_input_materialization import (
    MassInputRunWriter, PreparedMassInputFrame, write_mass_input_manifest,
)
from darsia.fluidflower.phase_numeric_fields import NumericPhaseFrame, load_numeric_phase_run
from darsia.fluidflower.project import sha256_file

STATE_FIELDS = ("injected_mass_g", "pressure_bar", "temperature_c",
                "pressure_gradient_bar_per_m", "temperature_gradient_c_per_m")


class MassInputPhaseConsumer:
    """Callable for the live phase writer, or for its verified stored replay."""

    def __init__(self, writer: MassInputRunWriter, *, expected_identity: Mapping,
                 expected_grid: Mapping, frame_states: Mapping[str, Mapping],
                 skip_frame_ids=(), include_previews=False):
        self.writer = writer
        self.expected_identity = deepcopy(dict(expected_identity))
        self.expected_grid = deepcopy(dict(expected_grid))
        self.frame_states = deepcopy(dict(frame_states))
        self.skip_frame_ids = set(map(str, skip_frame_ids))
        self.include_previews = bool(include_previews)
        self.seen = set()
        self.last_index = -1
        self.last_time = -float("inf")
        self.finished = False
        if set(self.frame_states) & self.skip_frame_ids:
            raise ValueError("A mass frame cannot be both supplied and explicitly skipped")
        for field, value in self.expected_grid.items():
            if field in writer.grid and writer.grid[field] != value:
                raise ValueError(f"Mass geometry differs from numeric phase geometry: {field}")
        if tuple(self.expected_grid.get("shape_rc", [])) != tuple(writer.shape):
            raise ValueError("Mass and numeric phase grids must have the same native shape")

    def __call__(self, frame: NumericPhaseFrame) -> None:
        if self.finished:
            raise RuntimeError("Mass-input stream is finalized")
        if frame.run_id != self.writer.run_id or dict(frame.identity) != self.expected_identity:
            raise ValueError("Mass input requires the selected run/model/runtime/correction identity")
        if dict(frame.grid) != self.expected_grid or frame.domain.shape != self.writer.shape:
            raise ValueError("Mass input and numerical phase registration geometry differ")
        if frame.frame_index != self.last_index + 1 or frame.time_hours <= self.last_time or frame.frame_id in self.seen:
            raise ValueError("Mass-input phase frames must be unique and chronological")
        if frame.frame_id not in self.skip_frame_ids:
            state = self.frame_states.get(frame.frame_id)
            if not isinstance(state, Mapping):
                raise ValueError(f"Explicit thermodynamic/injection state is missing for {frame.frame_id}")
            if "time_hours" not in state or float(state["time_hours"]) != frame.time_hours:
                raise ValueError(f"Experimental and numerical timestamps differ for {frame.frame_id}")
            values = {}
            for name in STATE_FIELDS:
                if name not in state or not math.isfinite(float(state[name])):
                    raise ValueError(f"Explicit finite {name} is required for {frame.frame_id}")
                values[name] = float(state[name])
            if frame.time_hours < 0 or values["pressure_bar"] <= 0 or values["injected_mass_g"] < 0:
                raise ValueError("Mass input requires nonnegative time/injected mass and positive pressure")
            if self.include_previews and frame.corrected_rgb is None:
                raise ValueError("Stored numeric replay has no RGB preview; disable previews or supply the live frame")
            source = {**dict(frame.source_reference), "numeric_phase_identity": self.expected_identity,
                      "numeric_phase_frame_index": frame.frame_index,
                      "numeric_phase_geometry": self.expected_grid,
                      "phase_probability_semantics": "joint_total_co2_and_joint_free_gas",
                      "rendered_phase_maps_used_as_input": False}
            self.writer.add_frame(PreparedMassInputFrame(
                frame_id=frame.frame_id, time_hours=frame.time_hours, **values,
                optical_acidification=frame.optical_acidification,
                phase_co2_probability=frame.phase_co2_probability,
                phase_free_gas_probability=frame.phase_free_gas_probability,
                corrected_rgb=frame.corrected_rgb if self.include_previews else None,
                source_reference=source,
            ))
        self.seen.add(frame.frame_id)
        self.last_index, self.last_time = frame.frame_index, frame.time_hours

    def finalize(self):
        missing = (set(self.frame_states) | self.skip_frame_ids) - self.seen
        if missing:
            raise ValueError(f"Mass input is missing requested phase frames: {sorted(missing)}")
        result = self.writer.finalize()
        from darsia.fluidflower.phase_pixel_area import carry_mass_area_bindings
        result = carry_mass_area_bindings(result)
        self.finished = True
        return result


def _load_geometry(config_path: Path, specification: Mapping, name: str) -> np.ndarray:
    record = specification.get(name)
    if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
        raise ValueError(f"Explicit checksummed {name} field is required")
    path = Path(record["path"])
    if not path.is_absolute():
        path = config_path.parent / path
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Measured {name} field checksum differs")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as archive:
            if not record.get("key"):
                raise ValueError(f"An explicit array key is required for {name}")
            return np.asarray(archive[record["key"]], dtype=np.float32)
    return np.asarray(loaded, dtype=np.float32)


def materialize_numeric_mass_inputs(numeric_root: Path, output_root: Path, configuration: Path) -> Path:
    """Bridge one complete immutable run using an explicit physical metadata file."""
    configuration = Path(configuration).expanduser().resolve()
    config = json.loads(configuration.read_text(encoding="utf-8-sig"))
    for name in ("run_id", "phase_candidate_id", "numeric_identity", "numeric_grid", "mass_grid",
                 "geometry", "permeability_unit", "frame_states", "upstream", "provenance"):
        if name not in config:
            raise ValueError(f"Physical mass-input configuration requires {name}")
    numeric_root = Path(numeric_root).expanduser().resolve()
    # Obtain the first frame only after the reader has checked every stored
    # frame's checksum and the complete run index. No inference is invoked.
    stream = iter(load_numeric_phase_run(numeric_root, expected_identity=config["numeric_identity"],
                                        expected_grid=config["numeric_grid"]))
    first = next(stream)
    if first.run_id != str(config["run_id"]).casefold():
        raise ValueError("Physical metadata belongs to a different experiment")
    geometry = {name: _load_geometry(configuration, config["geometry"], name)
                for name in ("depth", "porosity", "permeability")}
    for name, values in geometry.items():
        if values.shape != first.domain.shape or not np.isfinite(values).all():
            raise ValueError(f"Measured {name} does not match the native registered domain")
    # The active porous domain is authoritative. Geometry outside it has zero
    # contribution; no depth/porosity/permeability value is estimated here.
    geometry = {name: np.where(first.domain, values, np.float32(0)).astype(np.float32, copy=False)
                for name, values in geometry.items()}
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    writer = MassInputRunWriter(output_root, run_id=config["run_id"],
        phase_candidate_id=config["phase_candidate_id"], grid=config["mass_grid"], **geometry,
        permeability_unit=config["permeability_unit"],
        provenance={**config["provenance"], "physical_configuration": str(configuration),
                    "physical_configuration_sha256": sha256_file(configuration),
                    "numeric_phase_identity": config["numeric_identity"],
                    "rendered_phase_maps_used_as_input": False})
    consumer = MassInputPhaseConsumer(writer, expected_identity=config["numeric_identity"],
        expected_grid=config["numeric_grid"], frame_states=config["frame_states"],
        skip_frame_ids=config.get("skip_frame_ids", ()))
    consumer(first)
    for frame in stream:
        consumer(frame)
    row = consumer.finalize()
    return write_mass_input_manifest(output_root / "mass_input_manifest.json", run_inputs=[row],
                                    upstream=config["upstream"], provenance=writer.provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path,
                        help="Explicit geometry, experiment state, identities and upstream revision JSON")
    args = parser.parse_args(argv)
    print(materialize_numeric_mass_inputs(args.numeric_root, args.output_root, args.configuration))


if __name__ == "__main__":
    main()

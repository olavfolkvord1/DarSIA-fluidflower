"""Lossless, ordered phase fields for subsequent concentration/mass analysis.

These are optical probabilities and an uncalibrated indicator coordinate, not
concentration or mass. A run is consumable only after its complete manifest is
published. PNG overlays are never numerical inputs.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from .phase_rendering import PHASE_DISPLAY_POLICY, two_phase_masks
from .project import sha256_file, utc_now

FORMAT = "darsia-phase-numeric-fields-v1"
FIELDS = ("phase_co2_probability", "phase_free_gas_probability", "optical_acidification", "phase_labels")


def default_btb_acidification_coordinate(image, *, epsilon: float = 1e-4) -> np.ndarray:
    """Exact portable counterpart of observation_support's established BTB coordinate."""
    array = np.asarray(getattr(image, "img", image), dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image, got {array.shape}.")
    red = np.maximum(array[..., 0], 0.0)
    green = np.maximum(array[..., 1], 0.0)
    blue = np.maximum(array[..., 2], 0.0)
    yellow = 0.5 * (red + green)
    return np.asarray((yellow - blue) / (yellow + blue + float(epsilon)), dtype=np.float32)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _array_sha(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((values.shape, values.dtype.str)).encode())
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _readonly(values):
    view = np.asarray(values).view()
    view.flags.writeable = False
    return view


def _field(values, shape, name, *, probability=False):
    array = np.asarray(values)
    if array.shape != shape or array.dtype != np.float32:
        raise ValueError(f"{name} must be native-grid float32 {shape}, got {array.shape}/{array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if probability and (array.min() < 0 or array.max() > 1):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _rgb(values, shape, name):
    values = np.asarray(values)
    if values.shape != (*shape, 3) or values.dtype not in (np.uint8, np.float32):
        raise ValueError(f"{name} must be native-grid RGB, uint8 or float32")
    if values.dtype == np.float32 and (not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
        raise ValueError(f"{name} float32 RGB must be finite [0, 1]")
    return values


def _identity(value):
    result = json.loads(_json(dict(value)))
    for field in ("model_id", "model_sha256", "pipeline_sha256", "native_manifest_sha256"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"Numeric phase identity requires {field}")
    for field in ("model_sha256", "pipeline_sha256", "native_manifest_sha256"):
        if len(result[field]) != 64 or any(c not in "0123456789abcdef" for c in result[field].lower()):
            raise ValueError(f"Numeric phase identity has an invalid {field}")
    return result


@dataclass(frozen=True)
class NumericPhaseFrame:
    run_id: str
    series_id: str
    frame_index: int
    frame_id: str
    time_hours: float
    phase_co2_probability: np.ndarray
    phase_free_gas_probability: np.ndarray
    optical_acidification: np.ndarray
    phase_labels: np.ndarray
    domain: np.ndarray
    grid: Mapping
    identity: Mapping
    source_reference: Mapping
    corrected_rgb: np.ndarray | None = None


class NumericPhaseRunWriter:
    """Archive each final decision once; optionally pass the same arrays onward."""

    def __init__(self, root: Path, *, run_id: str, series_id: str, frame_count: int,
                 grid: Mapping, domain: np.ndarray, baseline_rgb: np.ndarray,
                 identity: Mapping, acidification_coordinate: Callable = default_btb_acidification_coordinate,
                 consumer: Callable[[NumericPhaseFrame], None] | None = None):
        self.root = Path(root).expanduser().resolve()
        self.run_id, self.series_id = str(run_id).casefold(), str(series_id).casefold()
        if not self.run_id or not self.series_id or int(frame_count) < 1:
            raise ValueError("Explicit run, series and positive expected frame count are required")
        self.frame_count = int(frame_count)
        self.grid = json.loads(_json(dict(grid)))
        self.shape = tuple(self.grid.get("shape_rc", []))
        if len(self.shape) != 2 or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("grid.shape_rc must contain the native image height and width")
        self.domain = np.asarray(domain)
        if self.domain.dtype != np.bool_ or self.domain.shape != self.shape:
            raise ValueError("Domain must be the native-grid boolean mask")
        self.domain = self.domain.copy()
        self.identity = _identity(identity)
        baseline = _rgb(baseline_rgb, self.shape, "baseline_rgb")
        self.rgb_dtype = baseline.dtype
        self.acidification_coordinate = acidification_coordinate
        self.baseline_acid = _field(acidification_coordinate(baseline), self.shape, "baseline acidification")
        try:
            algorithm_source = inspect.getsource(acidification_coordinate).replace("\r\n", "\n")
        except (OSError, TypeError) as error:
            raise ValueError("The acidification function must have auditable Python source") from error
        self.consumer = consumer
        self.records = []
        self.finished = False
        self.failed = False
        self._frame_ids = set()
        self.metadata = dict(format=FORMAT, status="incomplete", run_id=self.run_id, series_id=self.series_id,
            expected_frame_count=self.frame_count, grid=self.grid, grid_sha256=_digest(self.grid),
            identity=self.identity, phase_display_policy=PHASE_DISPLAY_POLICY,
            numerical_semantics={"phase_co2_probability": "joint_total_visible_co2_probability",
                "phase_free_gas_probability": "joint_free_gas_probability_not_conditional",
                "optical_acidification": "current_btb_coordinate_minus_registered_baseline; uncalibrated",
                "phase_labels": {"0": "no_visible_co2", "1": "aqueous_co2", "2": "free_gas"}},
            baseline_array_sha256=_array_sha(baseline),
            corrected_rgb_dtype=baseline.dtype.str,
            acidification_algorithm=dict(function=acidification_coordinate.__module__ + "." + acidification_coordinate.__qualname__,
                                        source_sha256=hashlib.sha256(algorithm_source.encode()).hexdigest()),
            created_utc=utc_now(), rendered_phase_maps_used_as_input=False)
        self.root.mkdir(parents=True, exist_ok=False)
        with (self.root / "domain.npy").open("wb") as stream:
            np.save(stream, self.domain, allow_pickle=False)
        self.metadata["domain"] = dict(path="domain.npy", sha256=sha256_file(self.root / "domain.npy"),
                                       array_sha256=_array_sha(self.domain))
        self._publish("incomplete")

    def _publish(self, status):
        payload = {**self.metadata, "status": status, "frames": self.records}
        payload["content_sha256"] = _digest(payload)
        _atomic_json(self.root / "manifest.json", payload)

    def add_frame(self, *, frame_index: int, frame_id: str, time_hours: float,
                  predictions: Mapping, current_rgb: np.ndarray, source_reference: Mapping | None = None) -> NumericPhaseFrame:
        if self.finished or self.failed:
            raise RuntimeError("Numeric phase writer is finalized or failed")
        index = int(frame_index)
        identifier, hours = str(frame_id).strip(), float(time_hours)
        if index != len(self.records) or index >= self.frame_count:
            raise ValueError("Numeric phase frames must arrive once in chronological index order")
        if not identifier or identifier in self._frame_ids or not math.isfinite(hours):
            raise ValueError("Frame identifiers must be unique and time must be finite")
        if self.records and hours <= self.records[-1]["time_hours"]:
            raise ValueError("Numeric phase frame times must be strictly increasing")
        current = _rgb(current_rgb, self.shape, "current_rgb")
        if current.dtype != self.rgb_dtype:
            raise ValueError("Current and baseline RGB encodings must match")
        p = _field(predictions["co2_presence_probability"], self.shape, "co2", probability=True)
        conditional = _field(predictions["gas_probability_conditional_on_co2"], self.shape, "conditional gas", probability=True)
        # This multiplication is essential: mass analysis consumes JOINT gas,
        # whereas the optical head is conditioned on visible CO2 presence.
        if "free_co2_gas_probability" in predictions:
            gas = _field(predictions["free_co2_gas_probability"], self.shape, "joint free gas", probability=True)
            # Reuse the final result already in memory. Bounded strips verify
            # its semantics without duplicating a full native probability map.
            for start in range(0, self.shape[0], 256):
                stop = start + 256
                if not np.array_equal(gas[start:stop], p[start:stop] * conditional[start:stop]):
                    raise ValueError("Final free-gas field is not the joint CO2 × conditional-gas probability")
        else:
            gas = np.multiply(p, conditional, dtype=np.float32)
        if np.any(p[~self.domain] != 0) or np.any(gas[~self.domain] != 0):
            raise ValueError("Final CO2 probabilities extend outside the registered domain")
        masks = two_phase_masks(predictions)
        labels = np.zeros(self.shape, np.uint8)
        labels[masks["aqueous"]] = 1
        labels[masks["gas"]] = 2
        del masks
        acid = _field(self.acidification_coordinate(current), self.shape, "current acidification")
        acid = np.subtract(acid, self.baseline_acid, dtype=np.float32)
        source = json.loads(_json(dict(source_reference or {})))
        source["corrected_array_sha256"] = _array_sha(current)
        source["baseline_array_sha256"] = self.metadata["baseline_array_sha256"]
        arrays = dict(phase_co2_probability=p, phase_free_gas_probability=gas,
                      optical_acidification=acid, phase_labels=labels)
        filename = f"frame-{index:06d}.npz"
        target = self.root / filename
        temporary = self.root / ("." + filename + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            os.replace(temporary, target)
            record = dict(frame_index=index, frame_id=identifier, time_hours=hours, path=filename,
                          sha256=sha256_file(target), source_reference=source)
            frame = NumericPhaseFrame(self.run_id, self.series_id, index, identifier, hours,
                *[_readonly(arrays[field]) for field in FIELDS], _readonly(self.domain),
                self.grid, self.identity, source, _readonly(current))
            if self.consumer is not None:
                self.consumer(frame)
            self.records.append(record)
            self._frame_ids.add(identifier)
            self._publish("incomplete")
            return frame
        except BaseException:
            self.failed = True
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def finalize(self) -> Path:
        if self.finished or self.failed or len(self.records) != self.frame_count:
            raise ValueError("Cannot publish an incomplete, failed or finalized numeric phase run")
        self._publish("complete")
        self.finished = True
        return self.root / "manifest.json"


def load_numeric_phase_run(root: Path, *, expected_identity: Mapping | None = None,
                           expected_grid: Mapping | None = None) -> Iterator[NumericPhaseFrame]:
    """Validate the whole archive before delivering any frame to mass analysis."""
    root = Path(root).expanduser().resolve()
    if root.is_file():
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("content_sha256", None)
    if digest != _digest(manifest) or manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Numeric phase manifest is corrupt, incomplete or incompatible")
    if manifest.get("phase_display_policy") != PHASE_DISPLAY_POLICY:
        raise ValueError("Numeric phase rendering policy is incompatible")
    identity, grid = _identity(manifest["identity"]), manifest["grid"]
    if _digest(grid) != manifest.get("grid_sha256"):
        raise ValueError("Numeric phase geometry checksum differs")
    if expected_identity is not None and dict(expected_identity) != identity:
        raise ValueError("Numeric phase model/runtime/correction identity differs")
    if expected_grid is not None and dict(expected_grid) != grid:
        raise ValueError("Numeric phase registered geometry differs")
    shape = tuple(grid.get("shape_rc", []))
    domain_record = manifest["domain"]
    if domain_record.get("path") != "domain.npy" or sha256_file(root / "domain.npy") != domain_record["sha256"]:
        raise ValueError("Numeric phase domain checksum differs")
    domain = np.load(root / "domain.npy", allow_pickle=False)
    if domain.dtype != np.bool_ or domain.shape != shape or _array_sha(domain) != domain_record["array_sha256"]:
        raise ValueError("Numeric phase domain differs from its registered grid")
    rows = manifest.get("frames", [])
    if not rows or len(rows) != manifest.get("expected_frame_count"):
        raise ValueError("Numeric phase frame sequence is incomplete")
    seen, previous = set(), -float("inf")
    for index, row in enumerate(rows):
        if row.get("frame_index") != index or row.get("path") != f"frame-{index:06d}.npz":
            raise ValueError("Numeric phase sequence contains reordered or incompatible frame paths")
        identifier, hours = row.get("frame_id"), float(row["time_hours"])
        if not identifier or identifier in seen or not math.isfinite(hours) or hours <= previous:
            raise ValueError("Numeric phase identifiers/times are invalid or nonchronological")
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Numeric phase frame checksum differs: {identifier}")
        seen.add(identifier)
        previous = hours
    for row in rows:
        with np.load(root / row["path"], allow_pickle=False) as archive:
            if set(archive.files) != set(FIELDS):
                raise ValueError("Numeric phase fields are incomplete or incompatible")
            arrays = {field: archive[field] for field in FIELDS}
        p = _field(arrays[FIELDS[0]], shape, FIELDS[0], probability=True)
        gas = _field(arrays[FIELDS[1]], shape, FIELDS[1], probability=True)
        _field(arrays[FIELDS[2]], shape, FIELDS[2])
        labels = arrays["phase_labels"]
        if labels.dtype != np.uint8 or labels.shape != shape or not np.isin(labels, (0, 1, 2)).all():
            raise ValueError("Invalid numeric phase labels")
        if np.any(gas > p) or np.any(p[~domain]) or np.any(labels[~domain]):
            raise ValueError("Numeric phase gas/support/domain contract differs")
        if not np.array_equal(labels > 0, p >= np.float32(.5)):
            raise ValueError("Numeric phase labels differ from optical support")
        yield NumericPhaseFrame(manifest["run_id"], manifest["series_id"], row["frame_index"],
            row["frame_id"], row["time_hours"], *[_readonly(arrays[field]) for field in FIELDS],
            _readonly(domain), grid, identity, row.get("source_reference", {}), None)

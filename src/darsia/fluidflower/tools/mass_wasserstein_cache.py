"""Mass-preserving coarse fields shared by mass and Wasserstein analysis."""

from __future__ import annotations

import hashlib
import json
import os
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any

import darsia
import numpy as np


CACHE_SCHEMA = 2


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def geometry_weight_array(geometry: Any) -> np.ndarray:
    """Return porosity times depth for both legacy and image-valued geometry."""

    raw_weight = getattr(geometry.weight, "img", geometry.weight)
    return np.nan_to_num(
        np.asarray(raw_weight),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def geometry_weight_sha256(geometry: Any) -> str:
    return _array_sha256(geometry_weight_array(geometry))


def factor_token(factor: float) -> str:
    value = float(factor)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Invalid Wasserstein cache factor {factor!r}.")
    return "f" + format(value, ".8g").replace(".", "p")


def cache_root(mass_folder: str | Path, factor: float) -> Path:
    folder = Path(mass_folder)
    return folder.parent / "mass_wasserstein" / factor_token(factor)


def build_mass_preserving_coarse(
    mass: darsia.Image,
    geometry: Any,
    factor: float,
    *,
    output_dtype: str | np.dtype[Any] = np.float64,
) -> darsia.Image:
    """Apply geometry weight once, resize, and preserve the original integral."""

    factor = float(factor)
    if not 0.0 < factor <= 1.0:
        raise ValueError(f"Invalid Wasserstein cache factor {factor!r}.")
    weight = geometry_weight_array(geometry)
    if tuple(weight.shape) != tuple(mass.img.shape[:2]):
        raise ValueError(
            "Geometry weight and mass field shapes differ: "
            f"{weight.shape} != {mass.img.shape[:2]}."
        )

    weighted = darsia.ScalarImage(
        np.multiply(
            np.asarray(mass.img, dtype=np.float64),
            np.asarray(weight, dtype=np.float64),
        ),
        **mass.metadata(),
    )
    coarse = darsia.resize(
        weighted,
        fx=factor,
        fy=factor,
        interpolation="inter_area",
    )
    original_integral = float(weighted.integral())
    coarse_integral = float(coarse.integral())
    if abs(coarse_integral) > np.finfo(float).eps:
        coarse *= original_integral / coarse_integral
    coarse.img = np.asarray(coarse.img, dtype=np.dtype(output_dtype))
    return coarse


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_image_atomic(image: darsia.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    )
    try:
        image.save(temporary, verbose=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class MassWassersteinCacheWriter:
    """Write one interruption-safe cache manifest for a mass-analysis run."""

    def __init__(
        self,
        mass_folder: str | Path,
        geometry: Any,
        factor: float,
        *,
        output_dtype: str = "float64",
        documented_missing_seconds: list[float] | None = None,
        manifest_name: str = "manifest.json",
    ) -> None:
        self.mass_folder = Path(mass_folder)
        self.geometry = geometry
        self.factor = float(factor)
        self.output_dtype = np.dtype(output_dtype)
        if self.output_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("Wasserstein cache dtype must be float32 or float64.")
        self.root = cache_root(self.mass_folder, self.factor)
        self.data_folder = self.root / "npz"
        if Path(manifest_name).name != manifest_name:
            raise ValueError("Cache manifest_name must be a plain file name.")
        self.manifest_path = self.root / manifest_name
        self.weight_sha256 = geometry_weight_sha256(geometry)
        normalized_missing: set[float] = set()
        for raw_seconds in documented_missing_seconds or []:
            seconds = _finite_float(raw_seconds)
            if seconds is None:
                raise ValueError(
                    "Documented missing cache times must be finite seconds."
                )
            normalized_missing.add(round(seconds, 6))
        self.documented_missing_seconds = sorted(normalized_missing)
        self.entries: dict[str, dict[str, Any]] = {}
        self.seen_stems: set[str] = set()
        self.complete = False
        self._load_compatible_manifest()
        _write_json_atomic(self.manifest_path, self._manifest())

    def _load_compatible_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema": CACHE_SCHEMA,
            "resize_factor": self.factor,
            "geometry_weight_sha256": self.weight_sha256,
            "output_dtype": self.output_dtype.name,
            "documented_missing_requested_seconds": (
                self.documented_missing_seconds
            ),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"Incompatible existing Wasserstein cache at {self.root}: "
                    f"{key}={payload.get(key)!r}, expected {value!r}."
                )
        for entry in payload.get("entries", []):
            stem = str(entry.get("source_stem") or "")
            if stem:
                self.entries[stem] = dict(entry)

    def _manifest(self) -> dict[str, Any]:
        return {
            "schema": CACHE_SCHEMA,
            "resize_factor": self.factor,
            "interpolation": "inter_area",
            "geometry_weight_applied": True,
            "integral_preserved": True,
            "geometry_weight_sha256": self.weight_sha256,
            "output_dtype": self.output_dtype.name,
            "documented_missing_requested_seconds": (
                self.documented_missing_seconds
            ),
            "complete": self.complete,
            "mass_folder": str(self.mass_folder),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": sorted(
                self.entries.values(),
                key=lambda entry: (
                    float(entry.get("time_seconds") or 0.0),
                    str(entry.get("source_stem") or ""),
                ),
            ),
        }

    def write(
        self,
        mass: darsia.Image,
        *,
        source_stem: str,
        source_image: str | Path | None = None,
    ) -> Path:
        coarse = build_mass_preserving_coarse(
            mass,
            self.geometry,
            self.factor,
            output_dtype=self.output_dtype,
        )
        destination = self.data_folder / f"{source_stem}.npz"
        _save_image_atomic(coarse, destination)
        relative_path = destination.relative_to(self.root)
        self.entries[source_stem] = {
            "source_stem": source_stem,
            "source_image": None if source_image is None else str(source_image),
            "time_seconds": _finite_float(getattr(mass, "time", None)),
            "path": relative_path.as_posix(),
            "shape": list(coarse.img.shape),
            "dimensions": [float(value) for value in coarse.dimensions],
            "integral": float(coarse.integral()),
        }
        self.seen_stems.add(source_stem)
        _write_json_atomic(self.manifest_path, self._manifest())
        return destination

    def close(self, *, complete: bool = True) -> None:
        if self.seen_stems:
            self.entries = {
                stem: entry
                for stem, entry in self.entries.items()
                if stem in self.seen_stems
            }
        self.complete = bool(complete)
        _write_json_atomic(self.manifest_path, self._manifest())


def merge_cache_shards(
    mass_folder: str | Path,
    factor: float,
    *,
    expected_entries: int | None = None,
) -> Path:
    """Validate and merge completed per-worker cache manifests."""

    root = cache_root(mass_folder, factor)
    shard_paths = sorted(root.glob("manifest.shard-*.json"))
    if not shard_paths:
        raise RuntimeError(f"No Wasserstein cache shard manifests found in {root}.")

    contracts: list[dict[str, Any]] = []
    entries: dict[str, dict[str, Any]] = {}
    common_keys = (
        "schema",
        "resize_factor",
        "interpolation",
        "geometry_weight_applied",
        "integral_preserved",
        "geometry_weight_sha256",
        "output_dtype",
        "documented_missing_requested_seconds",
    )
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            raise RuntimeError(f"Wasserstein cache shard is incomplete: {path}.")
        if contracts:
            for key in common_keys:
                if payload.get(key) != contracts[0].get(key):
                    raise RuntimeError(
                        f"Wasserstein cache shards disagree on {key}: {path}."
                    )
        contracts.append(payload)
        for entry in payload.get("entries", []):
            stem = str(entry.get("source_stem") or "")
            relative_path = str(entry.get("path") or "")
            if not stem or not relative_path:
                raise RuntimeError(f"Invalid cache entry in {path}.")
            if stem in entries:
                raise RuntimeError(
                    f"Duplicate source stem {stem!r} across cache shards."
                )
            destination = root / Path(relative_path)
            if not destination.is_file():
                raise RuntimeError(
                    f"Cache shard references a missing field: {destination}."
                )
            entries[stem] = dict(entry)

    if expected_entries is not None and len(entries) != int(expected_entries):
        raise RuntimeError(
            "Merged Wasserstein cache has "
            f"{len(entries)} entries, expected {int(expected_entries)}."
        )

    merged = dict(contracts[0])
    merged["complete"] = True
    merged["updated_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    merged["entries"] = sorted(
        entries.values(),
        key=lambda entry: (
            float(entry.get("time_seconds") or 0.0),
            str(entry.get("source_stem") or ""),
        ),
    )
    merged["shard_manifests"] = [path.name for path in shard_paths]
    destination = root / "manifest.json"
    _write_json_atomic(destination, merged)
    return destination


def load_cache_contract(
    mass_folder: str | Path,
    factor: float,
    *,
    expected_geometry_weight_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Return a validated complete cache manifest, or ``None`` if absent."""

    root = cache_root(mass_folder, factor)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema": CACHE_SCHEMA,
        "resize_factor": float(factor),
        "interpolation": "inter_area",
        "geometry_weight_applied": True,
        "integral_preserved": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Invalid Wasserstein cache contract in {manifest_path}: "
                f"{key}={payload.get(key)!r}, expected {expected!r}."
            )
    if payload.get("complete") is not True:
        raise RuntimeError(
            f"Wasserstein cache is incomplete: {manifest_path}."
        )
    if (
        expected_geometry_weight_sha256 is not None
        and payload.get("geometry_weight_sha256")
        != expected_geometry_weight_sha256
    ):
        raise RuntimeError(
            "Wasserstein cache geometry weight differs from the active geometry: "
            f"{manifest_path}."
        )
    missing = payload.get("documented_missing_requested_seconds", [])
    if not isinstance(missing, list) or any(
        _finite_float(value) is None for value in missing
    ):
        raise RuntimeError(
            "Invalid documented-missing schedule in Wasserstein cache: "
            f"{manifest_path}."
        )
    return payload


def cache_documents_missing_time(
    contract: dict[str, Any] | None,
    *,
    time_hours: float,
) -> bool:
    if contract is None:
        return False
    target = round(float(time_hours) * 3600.0, 6)
    return any(
        round(float(seconds), 6) == target
        for seconds in contract.get(
            "documented_missing_requested_seconds",
            [],
        )
    )


def load_cache_index(
    mass_folder: str | Path,
    factor: float,
    *,
    expected_geometry_weight_sha256: str | None = None,
) -> list[tuple[float, Path]] | None:
    """Return a validated cache index, or ``None`` when no cache exists."""

    root = cache_root(mass_folder, factor)
    payload = load_cache_contract(
        mass_folder,
        factor,
        expected_geometry_weight_sha256=expected_geometry_weight_sha256,
    )
    if payload is None:
        return None

    by_time: dict[float, Path] = {}
    for entry in payload.get("entries", []):
        seconds = _finite_float(entry.get("time_seconds"))
        relative_path = entry.get("path")
        if seconds is None or not relative_path:
            continue
        path = root / Path(str(relative_path))
        if not path.exists():
            raise RuntimeError(
                f"Wasserstein cache manifest references a missing file: {path}."
            )
        key = round(seconds, 6)
        previous = by_time.get(key)
        if previous is not None and previous != path:
            raise RuntimeError(
                "Ambiguous Wasserstein cache fields at the same saved time: "
                f"{previous} and {path}."
            )
        by_time[key] = path
    return sorted((seconds, path) for seconds, path in by_time.items())


def load_nearest_cached_mass(
    index: list[tuple[float, Path]],
    *,
    time_hours: float,
    tolerance_hours: float,
) -> darsia.Image | None:
    path = find_nearest_cached_path(
        index,
        time_hours=time_hours,
        tolerance_hours=tolerance_hours,
    )
    return None if path is None else darsia.imread(path)


def find_nearest_cached_path(
    index: list[tuple[float, Path]],
    *,
    time_hours: float,
    tolerance_hours: float,
) -> Path | None:
    """Resolve a cache entry without loading its numerical field."""

    if not index:
        return None
    target = float(time_hours) * 3600.0
    seconds = [entry[0] for entry in index]
    position = bisect_left(seconds, target)
    candidates = [
        candidate
        for candidate in (position - 1, position)
        if 0 <= candidate < len(index)
    ]
    selected = min(
        candidates,
        key=lambda candidate: (abs(seconds[candidate] - target), candidate),
    )
    if abs(seconds[selected] - target) > float(tolerance_hours) * 3600.0:
        return None
    return index[selected][1]

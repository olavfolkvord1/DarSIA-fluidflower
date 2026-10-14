"""Geometry-and-facies work for registered series folders.

A registered series keeps one geometry-and-facies project package inside its
series folder (``<series>/geometry_facies`` unless ``[paths] geometry_facies``
in ``series.toml`` names another location). The package uses the revision,
example and model stores of the geometry-and-facies page, so the native
application can host that page for a series that has no earlier ``series.json``
project.

A saved revision can be exported as inputs for the native DarSIA Setup.
Native Setup derives labels from a colored image of the raw baseline
photograph (eight pure colors, regions separated by connectivity) and applies
the rig corrections to those labels afterwards. The export therefore writes
such a colored image, the labels native Setup will derive from it, a facies
property table and the ``[labeling]`` and ``[facies]`` sections that connect
them. Revisions drawn on an already corrected rig grid are not exported.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import json
import os
import shutil
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from darsia.fluidflower.project import FluidFlowerProject, utc_now
from darsia.fluidflower.storage_paths import resolve_storage_path
from darsia.fluidflower.templates import RIG_TYPES

GEOMETRY_PACKAGE_KEY = "geometry_facies"
GEOMETRY_PACKAGE_DEFAULT = "geometry_facies"
NATIVE_SERIES_TOOL_KEY = "native_series"
NATIVE_SETUP_EXPORT_FORMAT = "darsia-fluidflower-native-setup-export-v1"

# Colors that darsia.label_image recognizes by default, in its label order.
# White (label 0) marks unlabeled pixels and is never used for a region.
NATIVE_LABEL_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (0, 0, 0)),
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("magenta", (255, 0, 255)),
    ("yellow", (255, 255, 0)),
)
# darsia.label_image stores consecutive labels as uint8.
NATIVE_MAX_REGIONS = 255
NATIVE_DEFAULT_RTOL = 0.001


# ---------------------------------------------------------------------------
# Series configuration


@dataclass(frozen=True)
class SeriesRunInputs:
    """Inputs of one run that the geometry workspace can start from."""

    run_id: str
    config_path: Path
    baseline_image: Path | None = None
    image_folders: tuple[Path, ...] = ()
    injection_protocol: Path | None = None
    rig_folder: Path | None = None
    width_m: float | None = None
    height_m: float | None = None
    issues: tuple[str, ...] = ()
    crop: dict[str, Any] | None = None


def _read_toml(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Merge TOML tables like the native loader: later files win."""

    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _storage_path(value: Any, base: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = resolve_storage_path(text)
    return path if path.is_absolute() else Path(base) / path


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0.0 else None


def _is_file(path: Path | None) -> bool:
    try:
        return path is not None and path.is_file()
    except OSError:
        return False


def read_series_manifest(series_root: Path) -> dict[str, Any]:
    path = Path(series_root) / "series.toml"
    manifest = _read_toml(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported series manifest: {path}")
    return manifest


def geometry_package_root(series_root: Path) -> Path:
    """Folder of the series' geometry-and-facies project package."""

    root = Path(series_root)
    try:
        manifest = read_series_manifest(root)
    except (OSError, ValueError):
        manifest = {}
    raw = (manifest.get("paths") or {}).get(GEOMETRY_PACKAGE_KEY)
    return _storage_path(raw or GEOMETRY_PACKAGE_DEFAULT, root) or (
        root / GEOMETRY_PACKAGE_DEFAULT
    )


def series_run_inputs(
    series_root: Path,
    *,
    check_files: bool = True,
    runs: Sequence[str] | None = None,
) -> list[SeriesRunInputs]:
    """List the runs of a series with their baseline image and rig folder.

    Paths follow the native loader: ``[data] baseline`` is looked up in the
    image folders, and the rig folder defaults to ``<results>/setup/rig``.
    Relative paths are read relative to the series folder. Without
    ``check_files`` no image or rig file is accessed (the first image folder is
    assumed); ``runs`` limits the list to the given run IDs.
    """

    root = Path(series_root)
    manifest = read_series_manifest(root)
    paths = manifest.get("paths") or {}
    common_path = _storage_path(paths.get("common_config", "config/common.toml"), root)
    common: dict[str, Any] = {}
    common_issue = ""
    if _is_file(common_path):
        try:
            common = _read_toml(common_path)
        except (OSError, ValueError) as error:
            common_issue = f"Could not read {common_path.name}: {error}"
    run_table = manifest.get("runs") or {}
    config_files = run_table.get("config_files") or {}
    run_folder = str(paths.get("run_configs") or "config/runs")
    wanted = None if runs is None else {str(run) for run in runs}
    rows: list[SeriesRunInputs] = []
    for raw_id in run_table.get("ids") or []:
        run_id = str(raw_id)
        if wanted is not None and run_id not in wanted:
            continue
        issues = [common_issue] if common_issue else []
        config_path = _storage_path(
            config_files.get(run_id) or f"{run_folder}/{run_id}.toml", root
        )
        try:
            config = _merge(common, _read_toml(config_path))
        except (OSError, ValueError) as error:
            issues.append(f"Could not read {config_path.name}: {error}")
            rows.append(SeriesRunInputs(run_id, config_path, issues=tuple(issues)))
            continue
        data = config.get("data") or {}
        folders: list[Path] = []
        for value in [data.get("folder"), *(data.get("folders") or [])]:
            folder = _storage_path(value, root)
            if folder is not None and folder not in folders:
                folders.append(folder)
        baseline: Path | None = None
        raw_baseline = str(data.get("baseline") or "").strip()
        if raw_baseline:
            candidate = resolve_storage_path(raw_baseline)
            if candidate.is_absolute():
                baseline = candidate
            elif folders:
                candidates = [folder / candidate for folder in folders]
                existing = (
                    [path for path in candidates if _is_file(path)]
                    if check_files
                    else []
                )
                baseline = existing[0] if existing else candidates[0]
        if baseline is None:
            issues.append("No baseline image is configured in [data].")
        elif check_files and not _is_file(baseline):
            issues.append(f"Baseline image is unavailable: {baseline}")
        protocols = config.get("protocols") or {}
        protocol = _storage_path(protocols.get("injection"), root)
        rig = config.get("rig") or {}
        results = _storage_path(data.get("results"), root)
        rig_folder = _storage_path(rig.get("path"), root) or (
            results / "setup" / "rig" if results is not None else None
        )
        if rig_folder is not None and check_files:
            if not _is_file(rig_folder / "baseline.npz"):
                rig_folder = None
        crop = ((config.get("corrections") or {}).get("curvature") or {}).get("crop")
        rows.append(
            SeriesRunInputs(
                run_id=run_id,
                config_path=config_path,
                baseline_image=baseline,
                image_folders=tuple(folders),
                injection_protocol=protocol,
                rig_folder=rig_folder,
                width_m=_float(rig.get("width")),
                height_m=_float(rig.get("height")),
                issues=tuple(issues),
                crop=dict(crop) if isinstance(crop, Mapping) else None,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Rig frame: raw photograph to rig coordinates

CROP_CORNERS = ("top_left", "bottom_left", "bottom_right", "top_right")
FRAME_FROM_CONFIGURATION = "series configuration"
FRAME_MARKED = "marked in geometry workspace"
# Unit-square position of each crop corner as (u, v): u to the right, v down.
_CORNER_UNIT = {
    "top_left": (0.0, 0.0),
    "bottom_left": (0.0, 1.0),
    "bottom_right": (1.0, 1.0),
    "top_right": (1.0, 0.0),
}


def _homography(
    source: Sequence[Sequence[float]], target: Sequence[Sequence[float]]
) -> np.ndarray:
    """3x3 projective map taking four (x, y) source points to the targets."""

    rows = []
    rhs = []
    for (x, y), (u, v) in zip(source, target):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.extend([u, v])
    solution = np.linalg.solve(
        np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)
    )
    return np.append(solution, 1.0).reshape(3, 3)


@dataclass
class RigFrame:
    """Inner corners of the rig in a raw photograph and the rig's inner size.

    Corners are (row, col) pixels, as in DarSIA's ``[corrections.curvature.crop]``.
    The crop maps them onto a rectangle; rig coordinates are then x from the
    left and y above the bottom edge of that rectangle, in metres.
    """

    corners: dict[str, tuple[float, float]]
    width_m: float | None = None
    height_m: float | None = None
    run_id: str = ""
    source: str = FRAME_MARKED
    image: str = ""
    image_size: tuple[int, int] | None = None

    def complete(self) -> bool:
        return all(name in self.corners for name in CROP_CORNERS)

    def has_size(self) -> bool:
        return bool(self.width_m and self.height_m)

    def _forward(self) -> np.ndarray:
        if not self.complete():
            raise ValueError("The rig frame needs all four corners")
        source = [
            (self.corners[name][1], self.corners[name][0]) for name in CROP_CORNERS
        ]
        return _homography(source, [_CORNER_UNIT[name] for name in CROP_CORNERS])

    def pixel_to_unit(self, row: float, col: float) -> tuple[float, float]:
        u, v, w = self._forward() @ np.array([col, row, 1.0])
        return float(u / w), float(v / w)

    def unit_to_pixel(self, u: float, v: float) -> tuple[float, float]:
        col, row, w = np.linalg.inv(self._forward()) @ np.array([u, v, 1.0])
        return float(row / w), float(col / w)

    def pixel_to_rig(self, row: float, col: float) -> tuple[float, float]:
        if not self.has_size():
            raise ValueError("Enter the rig's inner width and height first")
        u, v = self.pixel_to_unit(row, col)
        return u * float(self.width_m), (1.0 - v) * float(self.height_m)

    def rig_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        if not self.has_size():
            raise ValueError("Enter the rig's inner width and height first")
        return self.unit_to_pixel(
            float(x_m) / float(self.width_m), 1.0 - float(y_m) / float(self.height_m)
        )

    def native_crop(self) -> dict[str, Any]:
        """The frame as DarSIA's ``[corrections.curvature.crop]`` table."""

        table: dict[str, Any] = {
            name: [int(round(self.corners[name][0])), int(round(self.corners[name][1]))]
            for name in CROP_CORNERS
        }
        if self.has_size():
            table.update(width=float(self.width_m), height=float(self.height_m))
        table["in_meters"] = True
        return table

    def to_dict(self) -> dict[str, Any]:
        return {
            "corners": {
                name: [float(v) for v in self.corners[name]] for name in self.corners
            },
            "width_m": self.width_m,
            "height_m": self.height_m,
            "run_id": self.run_id,
            "source": self.source,
            "image": self.image,
            "image_size": list(self.image_size) if self.image_size else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RigFrame":
        corners = {
            str(name): (float(value[0]), float(value[1]))
            for name, value in dict(payload.get("corners") or {}).items()
            if name in CROP_CORNERS and value is not None and len(value) == 2
        }
        size = payload.get("image_size")
        return cls(
            corners=corners,
            width_m=_float(payload.get("width_m")),
            height_m=_float(payload.get("height_m")),
            run_id=str(payload.get("run_id") or ""),
            source=str(payload.get("source") or FRAME_MARKED),
            image=str(payload.get("image") or ""),
            image_size=(int(size[0]), int(size[1])) if size else None,
        )


def rig_frame_from_inputs(inputs: SeriesRunInputs) -> RigFrame | None:
    """The rig frame of a run's ``[corrections.curvature.crop]``, if configured."""

    crop = inputs.crop or {}
    corners: dict[str, tuple[float, float]] = {}
    for name in CROP_CORNERS:
        value = crop.get(name)
        if (
            isinstance(value, Sequence)
            and not isinstance(value, str)
            and len(value) == 2
        ):
            corners[name] = (float(value[0]), float(value[1]))
    if len(corners) != len(CROP_CORNERS):
        return None
    if crop.get("in_meters") is False:
        width = height = None
    else:
        width = _float(crop.get("width")) or inputs.width_m
        height = _float(crop.get("height")) or inputs.height_m
    return RigFrame(
        corners=corners,
        width_m=width,
        height_m=height,
        run_id=inputs.run_id,
        source=FRAME_FROM_CONFIGURATION,
        image=str(inputs.baseline_image or ""),
    )


# ---------------------------------------------------------------------------
# Project package


def default_actor() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - user lookup is best effort only
        return ""


def load_series_geometry_project(series_root: Path) -> FluidFlowerProject | None:
    root = geometry_package_root(series_root)
    if not _is_file(root / "series.json"):
        return None
    return FluidFlowerProject.load(root)


def native_series_link(project: FluidFlowerProject) -> dict[str, Any]:
    tools = project.manifest.get("tools") or {}
    link = tools.get(NATIVE_SERIES_TOOL_KEY) if isinstance(tools, Mapping) else None
    return dict(link) if isinstance(link, Mapping) else {}


def create_series_geometry_project(
    *,
    series_id: str,
    series_name: str,
    series_root: Path,
    rig_type: str = "custom",
    reference: SeriesRunInputs | None = None,
    created_by: str = "",
) -> FluidFlowerProject:
    """Create the geometry-and-facies package of a registered series.

    The reference run's injection protocol is registered so injection points
    can be read from it. No image is copied or modified.
    """

    if rig_type not in RIG_TYPES:
        raise ValueError(f"Unknown FluidFlower rig type: {rig_type!r}")
    root = geometry_package_root(series_root)
    if _is_file(root / "series.json"):
        raise FileExistsError(f"A geometry workspace already exists in {root}")
    protocol = (
        reference.injection_protocol
        if reference is not None and _is_file(reference.injection_protocol)
        else None
    )
    project = FluidFlowerProject.create(
        root,
        name=f"{series_name} · geometry and facies",
        series_id=series_id,
        template_id="custom",
        rig_type=rig_type,
        protocol_path=protocol,
        created_by=created_by,
    )
    project.update_metadata(
        "tools",
        {
            NATIVE_SERIES_TOOL_KEY: {
                "package_role": "series_geometry_facies",
                "series_id": str(series_id),
                "series_name": str(series_name),
                "series_root": str(series_root),
                "reference_run": reference.run_id if reference is not None else "",
                "linked_utc": utc_now(),
            }
        },
        updated_by=created_by,
        reason="Geometry and facies workspace for a registered series",
    )
    return project


def set_reference_run(
    project: FluidFlowerProject, run_id: str, *, actor: str = ""
) -> bool:
    """Remember the run whose raw baseline the geometry is drawn on."""

    link = native_series_link(project)
    if str(link.get("reference_run") or "") == str(run_id):
        return False
    link["reference_run"] = str(run_id)
    project.update_metadata(
        "tools",
        {NATIVE_SERIES_TOOL_KEY: link},
        updated_by=actor,
        reason=f"Reference run set to {run_id}",
    )
    return True


def _update_link(
    project: FluidFlowerProject, values: Mapping[str, Any], *, actor: str, reason: str
) -> None:
    link = native_series_link(project)
    link.update(values)
    project.update_metadata(
        "tools", {NATIVE_SERIES_TOOL_KEY: link}, updated_by=actor, reason=reason
    )


def saved_rig_frames(project: FluidFlowerProject) -> dict[str, RigFrame]:
    raw = native_series_link(project).get("rig_frames") or {}
    frames: dict[str, RigFrame] = {}
    for run_id, payload in dict(raw).items():
        if isinstance(payload, Mapping):
            frames[str(run_id)] = RigFrame.from_dict(payload)
    return frames


def save_rig_frame(
    project: FluidFlowerProject, frame: RigFrame, *, actor: str = ""
) -> None:
    """Store a rig frame marked in the workspace for one reference run."""

    if not frame.run_id:
        raise ValueError("A rig frame belongs to a run")
    frames = dict(native_series_link(project).get("rig_frames") or {})
    payload = frame.to_dict()
    payload["updated_utc"] = utc_now()
    frames[frame.run_id] = payload
    _update_link(
        project,
        {"rig_frames": frames},
        actor=actor,
        reason=f"Rig frame of run {frame.run_id} updated",
    )


def remove_rig_frame(
    project: FluidFlowerProject, run_id: str, *, actor: str = ""
) -> None:
    frames = dict(native_series_link(project).get("rig_frames") or {})
    if frames.pop(str(run_id), None) is not None:
        _update_link(
            project,
            {"rig_frames": frames},
            actor=actor,
            reason=f"Rig frame of run {run_id} removed",
        )


def _point(row: Mapping[str, Any]) -> dict[str, Any]:
    pixel = row.get("pixel")
    return {
        "location_id": str(row.get("location_id") or "").strip(),
        "name": str(row.get("name") or "").strip(),
        "x_m": _number_or_none(row.get("x_m")),
        "y_m": _number_or_none(row.get("y_m")),
        "pixel": (
            {
                "run_id": str(pixel.get("run_id") or ""),
                "row": float(pixel["row"]),
                "col": float(pixel["col"]),
            }
            if isinstance(pixel, Mapping) and pixel.get("row") is not None
            else None
        ),
        "source": str(row.get("source") or ""),
    }


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def saved_injection_points(project: FluidFlowerProject) -> list[dict[str, Any]]:
    raw = native_series_link(project).get("injection_points") or []
    return [_point(row) for row in raw if isinstance(row, Mapping)]


def save_injection_points(
    project: FluidFlowerProject, points: Sequence[Mapping[str, Any]], *, actor: str = ""
) -> list[dict[str, Any]]:
    """Store injection points (rig coordinates and the picked pixel)."""

    rows = [_point(row) for row in points]
    ids = [row["location_id"] for row in rows if row["location_id"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Injection point IDs must be unique")
    _update_link(
        project,
        {"injection_points": rows},
        actor=actor,
        reason="Injection points updated",
    )
    return rows


def protocol_injection_points(path: Path) -> list[dict[str, Any]]:
    """Distinct (location_x, location_y) pairs of a native injection protocol."""

    points: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            x = _number_or_none(row.get("location_x"))
            y = _number_or_none(row.get("location_y"))
            if x is None or y is None or (x, y) in seen:
                continue
            seen.add((x, y))
            points.append(
                {
                    "location_id": f"ip{len(points) + 1}",
                    "name": "",
                    "x_m": x,
                    "y_m": y,
                    "pixel": None,
                    "source": "injection protocol",
                }
            )
    return points


# ---------------------------------------------------------------------------
# Native Setup export


@dataclass
class NativeSetupExport:
    revision_id: str
    destination: Path
    colored_image: Path
    native_labels: Path
    facies_properties: Path
    setup_toml: Path
    toml_text: str
    region_count: int
    facies_count: int
    rtol: float
    grid: str
    warnings: list[str] = field(default_factory=list)
    reused: bool = False


_TABLE_LIMIT = 50_000_000


def _squeeze_labels(array: np.ndarray) -> np.ndarray:
    labels = np.squeeze(np.asarray(array))
    if labels.ndim != 2:
        raise ValueError(f"Label map must be two-dimensional, not {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        rounded = np.rint(labels)
        if not np.array_equal(rounded, labels):
            raise ValueError("Label map must contain integer label IDs")
        labels = rounded
    low, high = int(labels.min()), int(labels.max())
    small = np.iinfo(np.int32).min <= low and high <= np.iinfo(np.int32).max
    return labels.astype(np.int32 if small else np.int64, copy=False)


def _present_values(array: np.ndarray) -> np.ndarray:
    low, high = int(array.min()), int(array.max())
    if high - low + 1 <= _TABLE_LIMIT:
        shifted = array - low if low else array
        counts = np.bincount(shifted.ravel(), minlength=high - low + 1)
        return np.flatnonzero(counts) + low
    return np.unique(array)


def _remap(array: np.ndarray, mapping: Mapping[int, int], dtype: Any) -> np.ndarray:
    """Replace every value by mapping[value]; all present values must be mapped."""

    low, high = int(array.min()), int(array.max())
    if high - low + 1 <= _TABLE_LIMIT:
        table = np.zeros(high - low + 1, dtype=dtype)
        for key, value in mapping.items():
            if low <= int(key) <= high:
                table[int(key) - low] = value
        return table[array - low] if low else table[array]
    values, inverse = np.unique(array, return_inverse=True)
    table = np.asarray([mapping[int(value)] for value in values], dtype=dtype)
    return table[inverse.reshape(array.shape)]


def _neighbor_pairs(dense: np.ndarray, count: int) -> np.ndarray:
    """Pairs of node numbers (0..count-1) that touch, including diagonally."""

    keys: list[np.ndarray] = []
    for left, right in (
        (dense[:, :-1], dense[:, 1:]),
        (dense[:-1, :], dense[1:, :]),
        (dense[:-1, :-1], dense[1:, 1:]),
        (dense[:-1, 1:], dense[1:, :-1]),
    ):
        mask = left != right
        if np.any(mask):
            a = left[mask].astype(np.int64)
            b = right[mask].astype(np.int64)
            keys.append(np.unique(np.minimum(a, b) * count + np.maximum(a, b)))
    if not keys:
        return np.zeros((0, 2), dtype=np.int64)
    merged = np.unique(np.concatenate(keys))
    return np.stack((merged // count, merged % count), axis=1)


def _greedy_colors(count: int, pairs: np.ndarray, colors: int) -> list[int] | None:
    neighbors: list[set[int]] = [set() for _ in range(count)]
    for a, b in pairs.tolist():
        neighbors[a].add(b)
        neighbors[b].add(a)
    order = sorted(range(count), key=lambda node: (-len(neighbors[node]), node))
    result = [-1] * count
    for node in order:
        used = {result[other] for other in neighbors[node] if result[other] >= 0}
        free = next((index for index in range(colors) if index not in used), None)
        if free is None:
            return None
        result[node] = free
    return result


def native_color_index(labels: np.ndarray) -> np.ndarray:
    """Color each region so that touching regions never share a color.

    Returns color numbers 1-7 (the darsia.label_image labels of black, red,
    green, blue, cyan, magenta and yellow). Native Setup then recovers one
    label per connected region.
    """

    from skimage.measure import label as connected

    labels = _squeeze_labels(labels)
    values = _present_values(labels)
    dense = _remap(
        labels, {int(value): index for index, value in enumerate(values)}, np.int32
    )
    colors = _greedy_colors(
        values.size, _neighbor_pairs(dense, values.size), len(NATIVE_LABEL_COLORS)
    )
    if colors is not None:
        return np.asarray([color + 1 for color in colors], dtype=np.uint8)[dense]
    # Labels with several parts can need more colors; color the parts.
    components = connected(dense + 1, background=0, connectivity=2)
    count = int(components.max()) + 1
    colors = _greedy_colors(
        count, _neighbor_pairs(components, count), len(NATIVE_LABEL_COLORS)
    )
    if colors is None:
        raise ValueError(
            "The regions cannot be told apart with the seven colors native Setup "
            "recognizes. Merge small neighboring regions and save a new revision."
        )
    return np.asarray([color + 1 for color in colors], dtype=np.uint8)[components]


def native_labels_from_color_index(
    color_index: np.ndarray,
    *,
    significance: float | None = NATIVE_DEFAULT_RTOL,
    ensure_connectivity: bool = True,
) -> np.ndarray:
    """Labels that ``darsia.label_image`` derives from a pure-color image.

    This mirrors darsia.label_image with its default color map for images in
    which every pixel has one of the map colors (color number 1-7, 0 for
    white): connected components, removal of insignificant regions,
    consecutive relabelling and a one-pixel expansion into removed pixels.
    """

    import skimage.measure
    import skimage.segmentation

    color = np.asarray(color_index)
    unlabeled = color == 0
    if ensure_connectivity:
        labeled = skimage.measure.label(color, background=0, connectivity=2)
    else:
        labeled = color.astype(np.int64)
    labeled[unlabeled] = 0
    del unlabeled
    if significance is not None:
        counts = np.bincount(labeled.ravel())
        small = np.flatnonzero((counts > 0) & (counts < significance * labeled.size))
        small = small[small != 0]
        if small.size:
            labeled[np.isin(labeled, small)] = 0
    present = _present_values(labeled)
    present = present[present != 0]
    if present.size > NATIVE_MAX_REGIONS:
        raise ValueError(
            f"Native Setup supports at most {NATIVE_MAX_REGIONS} regions; "
            f"this label map has {present.size}."
        )
    result = _remap(
        labeled,
        {int(value): index for index, value in enumerate(present, start=1)},
        np.uint8,
    )
    del labeled
    holes = result == 0
    if np.any(holes):
        expanded = skimage.segmentation.expand_labels(result)
        result[holes] = expanded[holes]
    return result.astype(np.min_scalar_type(int(result.max(initial=0))))


def _facies_by_label(
    present: Sequence[int], regions: Sequence[Mapping[str, Any]], grouping_mode: str
) -> dict[int, int]:
    """Analysis facies of every label, as in the canonical facies map."""

    by_label: dict[int, int] = {}
    for row in regions:
        label_id = int(row["label_id"])
        by_label[label_id] = (
            label_id if grouping_mode == "per_layer" else int(row["facies_id"])
        )
    missing = [int(value) for value in present if int(value) not in by_label]
    if missing:
        raise ValueError(f"Labels without a layer definition: {missing[:10]}")
    return {int(value): by_label[int(value)] for value in present}


def _sha256(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(Path(path))


def _toml_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _comment(value: Any) -> str:
    return " ".join(str(value).split())


def _image_size(path: Path) -> tuple[int, int, int | None]:
    """Width, height and EXIF orientation from the image header."""

    from PIL import Image

    with Image.open(path) as image:
        orientation = None
        try:
            orientation = image.getexif().get(0x0112)
        except Exception:  # noqa: BLE001 - orientation is informational
            orientation = None
        return int(image.width), int(image.height), orientation


def photo_size(path: Path) -> tuple[int, int]:
    """Width and height of a photograph, from its header."""

    width, height, _orientation = _image_size(Path(path))
    return width, height


def _grid_check(
    config: Mapping[str, Any],
    shape: tuple[int, int],
    reference: SeriesRunInputs | None,
) -> tuple[str, list[str]]:
    """Describe the label grid; raise if native Setup cannot use it."""

    warnings: list[str] = []
    rig_root = Path(str(config.get("source_rig_path") or "")).expanduser()
    corrected = False
    if str(config.get("source_rig_path") or "").strip():
        try:
            corrected = any(
                next(rig_root.glob(pattern), None) is not None
                for pattern in (
                    "shape_correction_*.npz",
                    "color_correction_*.npz",
                    "correction_*.npz",
                )
            )
        except OSError:
            corrected = False
    if corrected:
        raise ValueError(
            "This revision uses labels on the corrected grid of an imported DarSIA "
            "rig. Native Setup expects labels on the raw baseline photograph and "
            "corrects them itself. Draw the geometry on a run's raw baseline image "
            "to export it."
        )
    height, width = shape
    if reference is None or reference.baseline_image is None:
        warnings.append(
            "No reference run baseline was available to check the label grid."
        )
        return f"{width} x {height} px (unchecked)", warnings
    try:
        image_width, image_height, orientation = _image_size(reference.baseline_image)
    except (OSError, ValueError) as error:
        warnings.append(
            f"Could not read the raw baseline of run {reference.run_id}: {error}"
        )
        return f"{width} x {height} px (unchecked)", warnings
    if orientation not in (None, 1):
        warnings.append(
            f"The raw baseline of run {reference.run_id} has EXIF orientation "
            f"{orientation}; check that the label map has the same orientation."
        )
    if (image_width, image_height) == (width, height):
        return (
            f"raw baseline of run {reference.run_id} "
            f"({reference.baseline_image.name}, {width} x {height} px)"
        ), warnings
    if abs(width / height - image_width / image_height) <= 0.005 * (
        image_width / image_height
    ):
        warnings.append(
            f"The label map has {width} x {height} px and the raw baseline of run "
            f"{reference.run_id} has {image_width} x {image_height} px; native Setup "
            "resizes the labels to the baseline."
        )
        return (
            f"{width} x {height} px, same aspect as the raw baseline of run "
            f"{reference.run_id}"
        ), warnings
    raise ValueError(
        f"The label map ({width} x {height} px) does not match the raw baseline of "
        f"run {reference.run_id} ({image_width} x {image_height} px). Native Setup "
        "expects labels drawn on the raw baseline photograph."
    )


def _resolved_points(
    points: Sequence[Mapping[str, Any]], frame: RigFrame | None
) -> list[dict[str, Any]]:
    """Injection points with rig coordinates, computed from the pixel if needed."""

    resolved = []
    for raw in points:
        point = _point(raw)
        pixel = point.get("pixel")
        if (
            (point["x_m"] is None or point["y_m"] is None)
            and pixel
            and frame is not None
            and frame.complete()
            and frame.has_size()
            and pixel.get("run_id") in ("", frame.run_id)
        ):
            point["x_m"], point["y_m"] = frame.pixel_to_rig(pixel["row"], pixel["col"])
        if point["x_m"] is not None:
            point["x_m"] = round(float(point["x_m"]), 4)
        if point["y_m"] is not None:
            point["y_m"] = round(float(point["y_m"]), 4)
        resolved.append(point)
    return resolved


def _frame_and_point_lines(
    frame: RigFrame | None, points: Sequence[Mapping[str, Any]], destination: Path
) -> list[str]:
    """Commented TOML for the rig frame and injection points.

    These sections often exist already or differ between runs, so they are
    given as comments to copy where they belong.
    """

    lines: list[str] = []
    if frame is not None and frame.complete():
        lines += [
            "",
            f"# Rig frame of run {frame.run_id} ({frame.source}). The crop applies to",
            "# runs photographed from the same camera position; add it to their run",
            "# files. [rig] width and height are the rig's inner size in metres.",
        ]
        if frame.has_size():
            lines += [
                "# [rig]",
                "# dim = 2",
                f"# width = {float(frame.width_m)!r}",
                f"# height = {float(frame.height_m)!r}",
            ]
        lines.append("# [corrections.curvature.crop]")
        for key, value in frame.native_crop().items():
            text = (
                "true"
                if value is True
                else (
                    f"[{value[0]}, {value[1]}]"
                    if isinstance(value, list)
                    else repr(value)
                )
            )
            lines.append(f"# {key} = {text}")
    if points:
        lines += [
            "",
            "# Injection points in rig coordinates (x from the left, y above the",
            "# bottom edge, metres) for location_x and location_y in the injection",
            f"# protocol; also in {_comment(destination / 'injection_points.csv')}.",
        ]
        for point in points:
            x, y = point.get("x_m"), point.get("y_m")
            where = (
                f"location_x = {x!r}, location_y = {y!r}"
                if x is not None and y is not None
                else "no rig coordinates (define the rig frame)"
            )
            label = point.get("location_id") or "point"
            name = f" ({_comment(point['name'])})" if point.get("name") else ""
            lines.append(f"# {label}{name}: {where}")
    return lines


def export_native_setup(
    project: FluidFlowerProject,
    *,
    reference: SeriesRunInputs | None = None,
    actor: str = "",
    replace: bool = False,
) -> NativeSetupExport:
    """Export the current geometry revision as native Setup inputs.

    The export folder is ``<package>/exports/native_setup/<revision id>``. An
    existing export of the same revision is reused unless ``replace`` is set.
    """

    from PIL import Image

    from darsia.fluidflower.geometry_facies import (
        GEOMETRY_REVISION_FORMAT,
        GeometryRevisionStore,
        _load_array,
    )

    current = GeometryRevisionStore(project).current()
    if current.get("format") != GEOMETRY_REVISION_FORMAT:
        raise ValueError("Save a validated geometry revision before exporting it.")
    revision_id = str(current["revision_id"])
    config = dict(current.get("configuration") or {})
    artifacts = dict(current.get("artifacts") or {})
    export_root = project.root / "exports" / "native_setup"
    destination = export_root / revision_id
    manifest_path = destination / "export.json"
    reference_run = reference.run_id if reference is not None else ""
    frame = saved_rig_frames(project).get(reference_run) if reference_run else None
    if frame is None and reference is not None:
        frame = rig_frame_from_inputs(reference)
    points = _resolved_points(saved_injection_points(project), frame)
    inputs_digest = hashlib.sha256(
        json.dumps(
            {
                "reference_run": reference_run,
                "frame": frame.to_dict() if frame is not None else None,
                "points": points,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    previous: dict[str, Any] = {}
    if _is_file(manifest_path) and not replace:
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    if previous and str(previous.get("inputs_digest") or "") == inputs_digest:
        manifest = previous
        return NativeSetupExport(
            revision_id=revision_id,
            destination=destination,
            colored_image=destination / "colored_labels.png",
            native_labels=destination / "native_labels.npz",
            facies_properties=destination / "facies_properties.csv",
            setup_toml=destination / "native_setup.toml",
            toml_text=(destination / "native_setup.toml").read_text(encoding="utf-8"),
            region_count=int(manifest.get("region_count") or 0),
            facies_count=int(manifest.get("facies_count") or 0),
            rtol=float(manifest.get("rtol") or NATIVE_DEFAULT_RTOL),
            grid=str(manifest.get("grid") or ""),
            warnings=list(manifest.get("warnings") or []),
            reused=True,
        )

    labels_path = Path(str(config.get("labels_path") or ""))
    if not _is_file(labels_path):
        raise FileNotFoundError(
            f"Label map of the revision is unavailable: {labels_path}"
        )
    labels = _squeeze_labels(_load_array(labels_path))
    grid, warnings = _grid_check(config, labels.shape, reference)
    present = [int(value) for value in _present_values(labels)]
    facies_by_label = _facies_by_label(
        present,
        [dict(row) for row in config.get("regions") or []],
        str(config.get("grouping_mode") or "shared_facies"),
    )
    facies_ids = sorted(set(facies_by_label.values()))
    position = {facies_id: index for index, facies_id in enumerate(facies_ids)}
    facies_dense = _remap(
        labels,
        {label: position[facies_id] for label, facies_id in facies_by_label.items()},
        np.int32,
    )

    color_index = native_color_index(labels)
    del labels
    from skimage.measure import label as connected

    components = connected(color_index, background=0, connectivity=2)
    areas = np.bincount(components.ravel())[1:]
    del components
    region_count = int(np.count_nonzero(areas))
    if region_count > NATIVE_MAX_REGIONS:
        raise ValueError(
            f"Native Setup supports at most {NATIVE_MAX_REGIONS} connected regions; "
            f"this revision has {region_count}. Merge small islands "
            "(Boundary edits) and save a new revision."
        )
    smallest = int(areas[areas > 0].min()) if region_count else 0
    rtol = NATIVE_DEFAULT_RTOL
    if smallest and smallest < rtol * color_index.size:
        # Keep every region: native Setup removes regions below rtol * area.
        rtol = float(f"{0.5 * smallest / color_index.size:.3g}")
    native = native_labels_from_color_index(color_index, significance=rtol)
    native_count = int(native.max(initial=0))
    if native_count != region_count:
        raise RuntimeError(
            f"Native label check failed: {native_count} labels for {region_count} regions"
        )

    # Facies of each native label. Every region lies inside one label, so the
    # majority is the only facies; anything else is reported.
    table = np.zeros((native_count + 1, len(facies_ids)), dtype=np.int64)
    for index in range(len(facies_ids)):
        table[:, index] = np.bincount(
            native[facies_dense == index], minlength=native_count + 1
        )
    del facies_dense
    mapping: dict[int, list[int]] = {}
    for native_label in range(1, native_count + 1):
        row = table[native_label]
        total = int(row.sum())
        if total == 0:
            continue
        best = int(np.argmax(row))
        if row[best] < total:
            warnings.append(
                f"Native label {native_label} covers {total - int(row[best])} pixels "
                "of another facies."
            )
        mapping.setdefault(int(facies_ids[best]), []).append(native_label)

    properties_source = Path(str(artifacts.get("facies_properties_path") or ""))
    if not _is_file(properties_source):
        raise FileNotFoundError(
            f"Facies properties of the revision are unavailable: {properties_source}"
        )
    with properties_source.open(encoding="utf-8", newline="") as stream:
        properties = list(csv.DictReader(stream))
    names = {int(row["id"]): str(row.get("name") or "") for row in properties}
    missing_props = sorted(set(mapping) - set(names))
    if missing_props:
        raise ValueError(f"Facies without properties: {missing_props}")
    overrides = sorted(str(run) for run in (artifacts.get("run_geometry") or {}))
    if overrides:
        warnings.append(
            "Run-specific geometry overrides are not exported: " + ", ".join(overrides)
        )
    width_m = _float(config.get("width_m"))
    height_m = _float(config.get("height_m"))

    export_root.mkdir(parents=True, exist_ok=True)
    staging = export_root / f".{revision_id}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        palette = np.asarray(
            [(255, 255, 255)] + [rgb for _name, rgb in NATIVE_LABEL_COLORS],
            dtype=np.uint8,
        )
        rgb = palette[color_index]
        Image.fromarray(rgb).save(staging / "colored_labels.png")
        preview_scale = min(1.0, 1600.0 / max(rgb.shape[:2]))
        if preview_scale < 1.0:
            size = (
                max(1, int(round(rgb.shape[1] * preview_scale))),
                max(1, int(round(rgb.shape[0] * preview_scale))),
            )
            Image.fromarray(rgb).resize(size, Image.NEAREST).save(
                staging / "colored_labels_preview.png"
            )
        del rgb
        metadata = {"space_dim": 2, "indexing": "ij"}
        if width_m and height_m:
            metadata.update(
                {"dimensions": [height_m, width_m], "origin": [0.0, height_m]}
            )
        np.savez_compressed(
            staging / "native_labels.npz", array=native, metadata=metadata
        )
        shutil.copy2(properties_source, staging / "facies_properties.csv")

        colored_final = destination / "colored_labels.png"
        props_final = destination / "facies_properties.csv"
        lines = [
            "# Native Setup inputs exported from a geometry-and-facies revision.",
            f"# Revision {revision_id} ({_comment(current.get('recorded_utc'))}, "
            f"{_comment(current.get('recorded_by') or 'unknown reviewer')}).",
            f"# Label grid: {_comment(grid)}.",
            "# Native Setup derives labels from colored_image (Segment colored image)",
            "# and corrects them with the rig corrections. Add these sections to",
            "# config/common.toml for all runs, or to one run file.",
            "",
            "[labeling]",
            f"colored_image = {_toml_string(colored_final)}",
            f"rtol = {rtol!r}",
            "ensure_connectivity = true",
            "",
            "[facies]",
            f"props = {_toml_string(props_final)}",
        ]
        for facies_id in sorted(set(names) | set(mapping)):
            name = _comment(names.get(facies_id, ""))
            lines.append("")
            lines.append(f"[facies.{facies_id}]" + (f" # {name}" if name else ""))
            lines.append(
                "labels = ["
                + ", ".join(str(value) for value in mapping.get(facies_id, []))
                + "]"
            )
        lines.extend(_frame_and_point_lines(frame, points, destination))
        toml_text = "\n".join(lines) + "\n"
        tomllib.loads(toml_text)
        (staging / "native_setup.toml").write_text(toml_text, encoding="utf-8")
        if points:
            with (staging / "injection_points.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "location_id",
                        "name",
                        "location_x",
                        "location_y",
                        "pixel_run",
                        "pixel_row",
                        "pixel_col",
                        "source",
                    ],
                )
                writer.writeheader()
                for point in points:
                    pixel = point.get("pixel") or {}
                    writer.writerow(
                        {
                            "location_id": point["location_id"],
                            "name": point["name"],
                            "location_x": point["x_m"],
                            "location_y": point["y_m"],
                            "pixel_run": pixel.get("run_id", ""),
                            "pixel_row": pixel.get("row", ""),
                            "pixel_col": pixel.get("col", ""),
                            "source": point["source"],
                        }
                    )
        manifest = {
            "schema": 1,
            "format": NATIVE_SETUP_EXPORT_FORMAT,
            "revision_id": revision_id,
            "created_utc": utc_now(),
            "created_by": str(actor or ""),
            "grid": grid,
            "reference_run": reference_run,
            "reference_baseline": (
                str(reference.baseline_image)
                if reference is not None and reference.baseline_image is not None
                else ""
            ),
            "inputs_digest": inputs_digest,
            "rig_frame": frame.to_dict() if frame is not None else None,
            "injection_points": points,
            "labels_source": str(labels_path),
            "label_shape": [int(value) for value in color_index.shape],
            "region_count": region_count,
            "facies_count": len(mapping),
            "rtol": rtol,
            "facies_to_native_labels": {
                str(key): value for key, value in sorted(mapping.items())
            },
            "warnings": warnings,
            "files": {
                name: {"sha256": _sha256(staging / name)}
                for name in (
                    "colored_labels.png",
                    "native_labels.npz",
                    "facies_properties.csv",
                    "native_setup.toml",
                )
            },
        }
        (staging / "export.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if destination.exists():
            retired = export_root / f".{revision_id}.{uuid.uuid4().hex}.old"
            os.replace(destination, retired)
            os.replace(staging, destination)
            shutil.rmtree(retired, ignore_errors=True)
        else:
            os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    project._audit(
        "native_setup_exported",
        actor,
        {
            "revision_id": revision_id,
            "destination": str(destination),
            "region_count": region_count,
            "facies_count": len(mapping),
        },
    )
    return NativeSetupExport(
        revision_id=revision_id,
        destination=destination,
        colored_image=destination / "colored_labels.png",
        native_labels=destination / "native_labels.npz",
        facies_properties=destination / "facies_properties.csv",
        setup_toml=destination / "native_setup.toml",
        toml_text=toml_text,
        region_count=region_count,
        facies_count=len(mapping),
        rtol=rtol,
        grid=grid,
        warnings=warnings,
    )


__all__ = [
    "CROP_CORNERS",
    "FRAME_FROM_CONFIGURATION",
    "FRAME_MARKED",
    "RigFrame",
    "protocol_injection_points",
    "remove_rig_frame",
    "rig_frame_from_inputs",
    "save_injection_points",
    "save_rig_frame",
    "saved_injection_points",
    "saved_rig_frames",
    "GEOMETRY_PACKAGE_DEFAULT",
    "GEOMETRY_PACKAGE_KEY",
    "NATIVE_LABEL_COLORS",
    "NATIVE_SETUP_EXPORT_FORMAT",
    "NativeSetupExport",
    "SeriesRunInputs",
    "create_series_geometry_project",
    "default_actor",
    "export_native_setup",
    "geometry_package_root",
    "load_series_geometry_project",
    "native_color_index",
    "native_labels_from_color_index",
    "native_series_link",
    "photo_size",
    "read_series_manifest",
    "series_run_inputs",
    "set_reference_run",
]

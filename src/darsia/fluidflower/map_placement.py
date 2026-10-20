"""Place an approved painted map in every run of a series, for review only.

The map is painted on the reference run's photograph. Each run has its own
measured geometry (format ``darsia-measured-run-geometry-v1``): a homography
from metric rig millimetres to the run's camera grid. A map pixel goes to a
run through the rig: ``inverse(H_target_grid_to_mm) @ H_reference_grid_to_mm``,
which is :func:`geometry_raster_sources.carry_label_reference` with the two
metric-to-grid matrices (nearest neighbour, edges replicated, uncovered
pixels flagged).

A placement writes only a new review folder: per run the placed layer map
and an overlay of its boundaries on the run's own reference photograph, and
one ``index.json`` with each run's offset and checks. It changes no map,
session, geometry revision, run configuration or result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

GEOMETRY_FORMAT = "darsia-measured-run-geometry-v1"
PLACEMENT_FORMAT = "darsia-fluidflower-map-placement-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RunGeometry:
    """One run's measured geometry: metric rig millimetres to its camera grid."""

    run_id: str
    metric_to_grid: np.ndarray
    reference_image: dict[str, Any]
    shape_hw: tuple[int, int] | None
    source: str


def _read_json(path: Path) -> dict[str, Any]:
    """A bounded JSON object read from a file."""

    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{path} is larger than {MAX_JSON_BYTES} bytes")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return value


def _matrix(value: Any, label: str) -> np.ndarray:
    """A finite, invertible 3 x 3 float64 matrix."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 3 x 3 matrix")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(f"{label} is not invertible")
    return matrix


def read_run_geometry(index_path: Path | str) -> dict[str, RunGeometry]:
    """Every run's measured geometry from a geometry index.

    ``index.json`` has ``format`` GEOMETRY_FORMAT and ``runs``: a mapping of
    run id to ``{"reference_model": "<relative path>"}`` (or a list of such
    entries with ``run_id``). Each reference model holds
    ``H_metric_mm_to_camera_u`` (or ``H_camera_u_to_metric_mm``, inverted
    here), its ``reference_image`` binding (path, sha256, size_bytes) and
    optionally ``shape_hw`` there. Run ids are compared case-insensitively.
    """

    index_path = Path(index_path)
    index = _read_json(index_path)
    if index.get("format") != GEOMETRY_FORMAT:
        raise ValueError(f"{index_path} is not a {GEOMETRY_FORMAT} index")
    runs = index.get("runs")
    if isinstance(runs, Mapping):
        entries = [dict(value, run_id=key) for key, value in runs.items()]
    elif isinstance(runs, list):
        entries = [dict(value) for value in runs]
    else:
        raise ValueError(f"{index_path} lists no runs")
    result: dict[str, RunGeometry] = {}
    for entry in entries:
        run = str(entry.get("run_id") or "").strip().casefold()
        relative = entry.get("reference_model")
        if not run or not isinstance(relative, str) or not relative:
            raise ValueError(f"{index_path}: each run needs run_id and reference_model")
        model_path = (index_path.parent / relative).resolve()
        model = _read_json(model_path)
        if model.get("format", GEOMETRY_FORMAT) != GEOMETRY_FORMAT:
            raise ValueError(f"{model_path} is not a {GEOMETRY_FORMAT} model")
        if "H_metric_mm_to_camera_u" in model:
            metric_to_grid = _matrix(model["H_metric_mm_to_camera_u"], f"{run} H_metric_mm_to_camera_u")
        elif "H_camera_u_to_metric_mm" in model:
            metric_to_grid = np.linalg.inv(
                _matrix(model["H_camera_u_to_metric_mm"], f"{run} H_camera_u_to_metric_mm")
            )
        else:
            raise ValueError(f"{model_path} has no metric-to-camera homography")
        image = dict(model.get("reference_image") or {})
        shape = image.get("shape_hw")
        result[run] = RunGeometry(
            run_id=run,
            metric_to_grid=metric_to_grid,
            reference_image=image,
            shape_hw=(int(shape[0]), int(shape[1])) if shape else None,
            source=str(model_path),
        )
    if not result:
        raise ValueError(f"{index_path} lists no runs")
    return result


def place_layer_map(
    layers: np.ndarray,
    reference: RunGeometry,
    target: RunGeometry,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """A reference run's layer map (uint8 layer ids) placed in another run.

    Returns the placed map on ``shape_hw`` and the pixels the map does not
    cover there (filled by edge replication, see carry_label_reference).
    The reference run itself gets its own map back unchanged.
    """

    from darsia.fluidflower.geometry_raster_sources import carry_label_reference

    layers = np.ascontiguousarray(layers, dtype=np.uint8)
    if reference.run_id == target.run_id and tuple(shape_hw) == layers.shape:
        return layers.copy(), np.zeros(layers.shape, dtype=bool)
    carried = carry_label_reference(
        layers,
        # The carrier takes explicit nested lists of numbers.
        reference_matrix=reference.metric_to_grid.tolist(),
        target_matrix=target.metric_to_grid.tolist(),
        target_shape_hw=[int(shape_hw[0]), int(shape_hw[1])],
        reference_filled=np.zeros(layers.shape, dtype=bool),
        source_fill_status="NOT_RECORDED",
    )
    return np.asarray(carried["labels"], dtype=np.uint8), np.asarray(
        carried["geometry_filled"], dtype=bool
    )


def boundary_overlay(
    photo: np.ndarray, layers: np.ndarray, colors: Mapping[int, tuple[int, int, int]]
) -> np.ndarray:
    """The photograph (uint8 RGB) with each layer's boundary drawn in its colour.

    A pixel is on a boundary when its 4-neighbour below or to the right has
    another layer; both sides of the boundary are drawn, two pixels wide.
    """

    if photo.shape[:2] != layers.shape:
        raise ValueError("The photograph and the layer map differ in size")
    overlay = np.ascontiguousarray(photo[..., :3], dtype=np.uint8).copy()
    edge = np.zeros(layers.shape, dtype=bool)
    down = layers[1:, :] != layers[:-1, :]
    right = layers[:, 1:] != layers[:, :-1]
    edge[1:, :] |= down
    edge[:-1, :] |= down
    edge[:, 1:] |= right
    edge[:, :-1] |= right
    for layer_id in np.unique(layers[edge]):
        color = colors.get(int(layer_id), (255, 255, 255))
        overlay[edge & (layers == layer_id)] = color
    return overlay


def map_offset(
    layers: np.ndarray, reference: RunGeometry, target: RunGeometry, steps: int = 17
) -> dict[str, float]:
    """How far the map moves from the reference run's grid to a run's grid.

    The median and the largest displacement in pixels over ``steps`` x
    ``steps`` points spanning the map's inside (its non-zero layers; the
    whole grid when there are none).
    """

    rows = np.flatnonzero(layers.any(axis=1))
    columns = np.flatnonzero(layers.any(axis=0))
    if rows.size and columns.size:
        top, bottom, left, right = rows[0], rows[-1], columns[0], columns[-1]
    else:
        top, bottom, left, right = 0, layers.shape[0] - 1, 0, layers.shape[1] - 1
    x, y = np.meshgrid(np.linspace(left, right, steps), np.linspace(top, bottom, steps))
    points = np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    moved = target.metric_to_grid @ np.linalg.inv(reference.metric_to_grid) @ points
    moved = moved[:2] / moved[2]
    distance = np.hypot(moved[0] - points[0], moved[1] - points[1])
    return {
        "median_px": round(float(np.median(distance)), 2),
        "max_px": round(float(distance.max()), 2),
    }


def run_checks(
    layers: np.ndarray, uncovered: np.ndarray, roles: Mapping[int, str], known: set[int]
) -> dict[str, Any]:
    """Counts for the review: pixels per layer, pixels inside the map (any
    layer), free-water pixels, pixels the map does not cover, and layer ids
    that are not in the map's layer table."""

    counts = np.bincount(layers.ravel(), minlength=256)
    present = {int(value): int(counts[value]) for value in np.flatnonzero(counts)}
    water = sum(count for value, count in present.items() if roles.get(value) == "free_water")
    unknown = sorted(value for value in present if value and value not in known)
    return {
        "pixels_per_layer": present,
        "inside_pixels": int(layers.size - present.get(0, 0)),
        "free_water_pixels": int(water),
        "uncovered_pixels": int(np.count_nonzero(uncovered)),
        "outside_pixels": int(present.get(0, 0)),
        "unknown_layer_ids": unknown,
    }


def review_order(runs: Sequence[str], reference_run: str, first: Sequence[str] = ()) -> list[str]:
    """The reference run first (the control), then ``first`` in its order,
    then the other runs in series order."""

    wanted = [str(reference_run).casefold(), *(str(run).casefold() for run in first)]
    ordered = [run for run in dict.fromkeys(wanted) if run in runs]
    return ordered + [run for run in runs if run not in ordered]


def place_map_in_runs(
    destination_root: Path | str,
    layers: np.ndarray,
    *,
    reference_run: str,
    geometry_index: Path | str,
    runs: Sequence[str],
    photo_loader: Callable[[str, RunGeometry], np.ndarray],
    colors: Mapping[int, tuple[int, int, int]],
    roles: Mapping[int, str],
    version: Mapping[str, Any],
    first_runs: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Place the map in every run and write a new review folder.

    ``layers`` is the map's layer ids on the reference run's grid;
    ``photo_loader(run, geometry)`` returns that run's reference photograph
    as uint8 RGB on its grid; ``version`` names the approved map version
    (name, sha256, approval); ``first_runs`` are shown right after the
    reference run. The folder ``<destination_root>/<UTC>-<id>`` gets
    ``<run>/layers.npz`` (exact), ``<run>/overlay.jpg`` (for viewing) and
    ``index.json`` (format PLACEMENT_FORMAT, with the runs in review order and
    each run's offset and checks). Runs without a geometry are listed as
    such. Returns the index.
    """

    from PIL import Image

    from darsia.fluidflower.project import sha256_file

    index_path = Path(geometry_index)
    geometries = read_run_geometry(index_path)
    reference_key = str(reference_run).casefold()
    if reference_key not in geometries:
        raise ValueError(f"The geometry index has no model for the reference run {reference_run}")
    reference = geometries[reference_key]
    layers = np.ascontiguousarray(layers, dtype=np.uint8)
    if reference.shape_hw is not None and layers.shape != reference.shape_hw:
        raise ValueError(
            f"The map ({layers.shape[1]} x {layers.shape[0]} px) is not on the reference "
            f"run's grid ({reference.shape_hw[1]} x {reference.shape_hw[0]} px)"
        )
    keys = [str(run).casefold() for run in runs]
    order = review_order(keys, reference_key, first_runs)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = Path(destination_root) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    known = set(colors)
    rows = []
    for number, run in enumerate(order, 1):
        if progress is not None:
            progress(f"Placing the map in {run.upper()} ({number} of {len(order)})…")
        target = geometries.get(run)
        if target is None:
            rows.append({"run_id": run, "status": "no geometry"})
            continue
        photo = np.asarray(photo_loader(run, target))
        shape = target.shape_hw or tuple(photo.shape[:2])
        if tuple(photo.shape[:2]) != tuple(shape):
            raise ValueError(f"{run}: the photograph does not have the geometry's grid size")
        placed, uncovered = place_layer_map(layers, reference, target, shape)
        run_folder = folder / run
        run_folder.mkdir()
        np.savez_compressed(run_folder / "layers.npz", layers=placed, uncovered=uncovered)
        # A full-resolution JPEG without chroma subsampling: the boundaries
        # stay sharp, at a fraction of a PNG's size. The exact map is the npz.
        Image.fromarray(boundary_overlay(photo, placed, colors)).save(
            run_folder / "overlay.jpg", quality=92, subsampling=0
        )
        del photo
        rows.append(
            {
                "run_id": run,
                "status": "placed",
                "layers": f"{run}/layers.npz",
                "overlay": f"{run}/overlay.jpg",
                "shape_hw": [int(shape[0]), int(shape[1])],
                "geometry_model": target.source,
                "offset_px": map_offset(layers, reference, target),
                "checks": run_checks(placed, uncovered, roles, known),
            }
        )
    index = {
        "format": PLACEMENT_FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_run": reference_key,
        "map_version": dict(version),
        "geometry_index": {
            "path": str(index_path.resolve()),
            "sha256": sha256_file(index_path),
        },
        "runs": rows,
    }
    temporary = folder / f".index.json.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, folder / "index.json")
    return dict(index, folder=str(folder))


def read_placement(folder: Path | str) -> dict[str, Any]:
    """A placement review folder's index, with ``folder`` added."""

    folder = Path(folder)
    index = _read_json(folder / "index.json")
    if index.get("format") != PLACEMENT_FORMAT:
        raise ValueError(f"{folder} is not a map placement folder")
    return dict(index, folder=str(folder))


__all__ = [
    "GEOMETRY_FORMAT",
    "PLACEMENT_FORMAT",
    "RunGeometry",
    "boundary_overlay",
    "map_offset",
    "place_layer_map",
    "place_map_in_runs",
    "read_placement",
    "read_run_geometry",
    "review_order",
    "run_checks",
]

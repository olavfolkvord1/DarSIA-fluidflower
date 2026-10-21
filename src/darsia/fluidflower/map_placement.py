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
one ``index.json`` with each run's shift, layer offsets and checks. The
layer offsets are the existing read-only measurement of
:mod:`geometry_offset` (the strongest colour edge within +/-25 mm of each
boundary; median, 10th and 90th percentiles). The reference run measures
its own map on its own photograph, so each offset is also given as the
change from the reference run's median for the same pair of layers: the
part that the placement adds. The checks are the round-5
review's: layer ids missing from the map's table, free water enclosed in the
medium, and pixels the map does not cover that need filling. A placement
changes no map, session, geometry revision, run configuration or result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
# The overlays have at most two labelled line types.
BOUNDARY_COLOR = (255, 255, 0)
GAP_COLOR = (255, 0, 255)
OVERLAY_LEGEND = (
    ("Boundaries of the placed map", BOUNDARY_COLOR),
    ("Outline of the pixels the map does not cover", GAP_COLOR),
)
# The layer offsets are measured on the run's grid reduced to about this
# many millimetres per pixel.
OFFSET_SPACING_MM = 0.5


@dataclass(frozen=True)
class RunGeometry:
    """One run's measured geometry: metric rig millimetres to its camera grid.

    ``model`` is the whole reference model as read, for display.
    """

    run_id: str
    metric_to_grid: np.ndarray
    reference_image: dict[str, Any]
    shape_hw: tuple[int, int] | None
    source: str
    model: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


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


def _model_path(index_path: Path, run: str, binding: Any) -> Path:
    """The reference model file a run's index entry names.

    The binding is a path, or ``{"path", "sha256", "size_bytes"}``; a
    relative path is taken from the index's folder. A bound size or SHA-256
    that no longer matches is refused.
    """

    from darsia.fluidflower.project import sha256_file

    row = dict(binding) if isinstance(binding, Mapping) else {"path": binding}
    text = row.get("path")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{index_path}: run {run} names no reference model")
    path = Path(text)
    path = path if path.is_absolute() else (index_path.parent / path).resolve()
    if "size_bytes" in row and path.stat().st_size != int(row["size_bytes"]):
        raise ValueError(f"{path} is not the reference model the index bound (size)")
    if row.get("sha256") and sha256_file(path) != row["sha256"]:
        raise ValueError(f"{path} is not the reference model the index bound (SHA-256)")
    return path


def read_run_geometry(index_path: Path | str) -> dict[str, RunGeometry]:
    """Every run's measured geometry from a geometry index.

    ``index.json`` has ``format`` GEOMETRY_FORMAT and ``runs``: a mapping of
    run id to ``{"reference_model": <binding>}`` (or a list of such entries
    with ``run_id``), where the binding is ``{"path", "sha256",
    "size_bytes"}`` or a path. Each reference model holds
    ``H_metric_mm_to_camera_u`` (or ``H_camera_u_to_metric_mm``, inverted
    here), ``shape_hw`` of its grid and its ``reference_image`` binding
    (path, sha256, size_bytes). Run ids are compared case-insensitively.
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
        run = str(entry.get("run_id") or entry.get("run") or "").strip().casefold()
        if not run:
            raise ValueError(f"{index_path}: each run needs its run id")
        model_path = _model_path(index_path, run, entry.get("reference_model"))
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
        shape = model.get("shape_hw") or image.get("shape_hw")
        result[run] = RunGeometry(
            run_id=run,
            metric_to_grid=metric_to_grid,
            reference_image=image,
            shape_hw=(int(shape[0]), int(shape[1])) if shape else None,
            source=str(model_path),
            model=model,
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


def _edges(values: np.ndarray) -> np.ndarray:
    """Pixels whose 4-neighbour below or to the right differs, on both sides."""

    edge = np.zeros(values.shape, dtype=bool)
    down = values[1:, :] != values[:-1, :]
    right = values[:, 1:] != values[:, :-1]
    edge[1:, :] |= down
    edge[:-1, :] |= down
    edge[:, 1:] |= right
    edge[:, :-1] |= right
    return edge


def boundary_overlay(
    photo: np.ndarray, layers: np.ndarray, uncovered: np.ndarray | None = None
) -> np.ndarray:
    """The photograph (uint8 RGB) with the two line types of OVERLAY_LEGEND:
    the placed map's boundaries and the outline of the pixels it does not
    cover. Each line is two pixels wide, one on each side of the change."""

    if photo.shape[:2] != layers.shape:
        raise ValueError("The photograph and the layer map differ in size")
    overlay = np.ascontiguousarray(photo[..., :3], dtype=np.uint8).copy()
    overlay[_edges(layers)] = BOUNDARY_COLOR
    if uncovered is not None:
        overlay[_edges(np.asarray(uncovered, dtype=bool))] = GAP_COLOR
    return overlay


def grid_spacing_mm(geometry: RunGeometry, shape_hw: tuple[int, int]) -> tuple[float, float]:
    """Millimetres per pixel down a column and along a row at the grid's centre."""

    to_mm = np.linalg.inv(geometry.metric_to_grid)
    row, col = (shape_hw[0] - 1) / 2.0, (shape_hw[1] - 1) / 2.0
    points = np.array([[col, col + 1.0, col], [row, row, row + 1.0], [1.0, 1.0, 1.0]])
    mm = to_mm @ points
    mm = mm[:2] / mm[2]
    along = float(np.hypot(*(mm[:, 1] - mm[:, 0])))
    down = float(np.hypot(*(mm[:, 2] - mm[:, 0])))
    return down, along


def layer_offsets(
    photo: np.ndarray,
    layers: np.ndarray,
    geometry: RunGeometry,
    roles: Mapping[int, str],
    *,
    spacing_mm: float = OFFSET_SPACING_MM,
) -> dict[str, Any]:
    """How far each layer boundary lies from the edge seen in the photograph.

    The existing read-only method, geometry_offset.measure_arrays: along each
    boundary's normal within +/-25 mm, the strongest colour edge, and per pair
    of layers the median and the 10th and 90th percentiles in millimetres.
    It runs on the run's grid reduced to about ``spacing_mm`` per pixel
    (block means of the photograph, block centres of the map), with the
    spacing of the grid's centre from the run's geometry. Descriptive only.
    """

    import cv2

    from darsia.fluidflower.geometry_offset import measure_arrays

    down, along = grid_spacing_mm(geometry, layers.shape)
    factor = max(1, int(round(spacing_mm / min(down, along))))
    height, width = layers.shape[0] // factor, layers.shape[1] // factor
    start = factor // 2
    reduced = np.ascontiguousarray(
        layers[start : height * factor : factor, start : width * factor : factor]
    )
    image = np.ascontiguousarray(photo[: height * factor, : width * factor, :3])
    if factor > 1:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    report, _drawings = measure_arrays(
        image,
        reduced,
        {
            "space_dim": 2,
            "indexing": "ij",
            "dimensions": [height * factor * down / 1000.0, width * factor * along / 1000.0],
        },
        [
            {"label_id": int(layer), "analysis_role": str(role), "enabled": True}
            for layer, role in roles.items()
        ],
    )

    def number(value: Any) -> float | None:
        return None if value is None else round(float(value), 2)

    return {
        "method": "geometry_offset.measure_arrays",
        "band_mm": float(report["parameters"]["band_mm"]),
        "spacing_mm": [round(float(value), 4) for value in report["spacing_mm"]],
        "sign_convention": report["sign_convention"],
        "boundaries": [
            {
                "labels": list(row["labels"]),
                "median_mm": number(row["median_mm"]),
                "p10_mm": number(row["p10_mm"]),
                "p90_mm": number(row["p90_mm"]),
                "measured_sample_count": int(row["measured_sample_count"]),
                "sample_count": int(row["sample_count"]),
                "status": row["status"],
            }
            for row in report["boundaries"]
            if row["measurement_eligible"]
        ],
    }


def add_changes_from_reference(
    offsets: Mapping[str, Any] | None, reference: Mapping[str, Any] | None
) -> None:
    """Give each boundary of ``offsets`` its ``change_from_reference_mm``: its
    median minus the reference run's median for the same pair of layers, or
    None when either is missing."""

    if not offsets:
        return
    medians = {
        tuple(row["labels"]): row["median_mm"]
        for row in (reference or {}).get("boundaries") or []
    }
    for row in offsets.get("boundaries") or []:
        base = medians.get(tuple(row["labels"]))
        value = row.get("median_mm")
        row["change_from_reference_mm"] = (
            None if base is None or value is None else round(float(value) - float(base), 2)
        )


def water_pockets(layers: np.ndarray, roles: Mapping[int, str]) -> int:
    """Free-water pixels outside the largest connected free-water region:
    water enclosed in the medium, which the review looks for."""

    from scipy import ndimage as ndi

    water = np.isin(layers, [layer for layer, role in roles.items() if role == "free_water"])
    labelled, count = ndi.label(water)
    if count <= 1:
        return 0
    sizes = np.bincount(labelled.ravel())[1:]
    return int(sizes.sum() - sizes.max())


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
    layer), free-water pixels, free water enclosed in the medium, pixels the
    map does not cover, those of them inside the map (gaps to fill, nearest
    sand in the row, then in the column) and layer ids that are not in the
    map's layer table."""

    counts = np.bincount(layers.ravel(), minlength=256)
    present = {int(value): int(counts[value]) for value in np.flatnonzero(counts)}
    water = sum(count for value, count in present.items() if roles.get(value) == "free_water")
    unknown = sorted(value for value in present if value and value not in known)
    return {
        "pixels_per_layer": present,
        "inside_pixels": int(layers.size - present.get(0, 0)),
        "free_water_pixels": int(water),
        "water_pocket_pixels": water_pockets(layers, roles),
        "uncovered_pixels": int(np.count_nonzero(uncovered)),
        "gap_pixels": int(np.count_nonzero(uncovered & (layers != 0))),
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
    roles: Mapping[int, str],
    version: Mapping[str, Any],
    names: Mapping[int, str] | None = None,
    facies: Mapping[int, str] | None = None,
    first_runs: Sequence[str] = (),
    measure_offsets: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Place the map in every run and write a new review folder.

    ``layers`` is the map's layer ids on the reference run's grid; ``roles``,
    ``names`` and ``facies`` give each layer id's role, name and facies
    name; ``photo_loader(run,
    geometry)`` returns that run's reference photograph as uint8 RGB on its
    grid; ``version`` names the approved map version (name, sha256,
    approval); ``first_runs`` are shown right after the reference run. The
    folder ``<destination_root>/<UTC>-<id>`` gets ``<run>/layers.npz``
    (exact), ``<run>/overlay.jpg`` (for viewing) and ``index.json`` (format
    PLACEMENT_FORMAT, with the runs in review order and each run's shift,
    layer offsets and checks). Runs without a geometry are listed as such.
    Returns the index.
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
    known = {int(layer) for layer in roles}
    rows = []
    reference_offsets: dict[str, Any] | None = None
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
        Image.fromarray(boundary_overlay(photo, placed, uncovered)).save(
            run_folder / "overlay.jpg", quality=92, subsampling=0
        )
        offsets = None
        if measure_offsets:
            if progress is not None:
                progress(f"Measuring the layer offsets in {run.upper()} ({number} of {len(order)})…")
            offsets = layer_offsets(photo, placed, target, roles)
            if run == reference_key:
                reference_offsets = offsets
            add_changes_from_reference(offsets, reference_offsets)
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
                "layer_offsets": offsets,
                "checks": run_checks(placed, uncovered, roles, known),
            }
        )
    index = {
        "format": PLACEMENT_FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_run": reference_key,
        "map_version": dict(version),
        "layers": {
            str(int(layer)): {
                "name": str((names or {}).get(layer, "")),
                "role": str(role),
                "facies": str((facies or {}).get(layer, "")),
            }
            for layer, role in sorted(roles.items())
        },
        "overlay_legend": [
            {"label": label, "rgb": list(color)} for label, color in OVERLAY_LEGEND
        ],
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
    "BOUNDARY_COLOR",
    "GAP_COLOR",
    "OVERLAY_LEGEND",
    "add_changes_from_reference",
    "boundary_overlay",
    "grid_spacing_mm",
    "layer_offsets",
    "map_offset",
    "place_layer_map",
    "place_map_in_runs",
    "read_placement",
    "read_run_geometry",
    "review_order",
    "run_checks",
    "water_pockets",
]

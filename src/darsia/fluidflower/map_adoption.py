"""The provenance and the preparation of a painted map's adoption.

A placement review folder (see :mod:`map_placement`) holds, per run, the
approved map placed on the run's camera grid. Before such a placement can
be adopted as a geometry revision, its provenance is bound: the approved
version and its approval, the placement index, the geometry index, and each
placed run's layer map, all by SHA-256. :func:`adoption_blockers` names what
an adoption still lacks.

:func:`prepare_adoption` carries the approved map to each run's
rig-corrected native label grid, the grid that Build run model and the
jobs read, through the run's measured geometry and its baseline native
grid: ``H_referenceU_to_targetNative = H_targetU_to_targetNative @
H_target_mm_to_U @ H_referenceU_to_mm``, in the existing nearest-neighbour
categorical carrier, with the fill and support flags kept. It writes a new
preparation folder; the facies ids of the regions wait for the facies table
of the map, and none is invented.

:func:`adoption_candidate` turns a preparation and the facies table of the
map (each label's facies id in the series' full facies table) into a
geometry revision candidate on the current corrected workspace revision: a
canonical edit of its imported parent whose every run carries a
``painted-map-placement-v1`` registration, replayed exactly by the Store.
Saving it is a separate, explicit step.

Nothing here writes a revision, a map or a run configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

PROVENANCE_FORMAT = "darsia-fluidflower-painted-placement-provenance-v1"
PREPARATION_FORMAT = "darsia-fluidflower-painted-map-adoption-preparation-v1"
FACIES_IDS_PENDING = "waits for the facies table of the map"


def _bound(path: Path) -> dict[str, Any]:
    """A file bound by its resolved path, SHA-256 and size."""

    from darsia.fluidflower.geometry_raster_sources import descriptor

    return descriptor(path)


def placement_provenance(folder: Path | str) -> dict[str, Any]:
    """The provenance record of a placement review folder.

    Every placed run's ``layers.npz`` is bound by SHA-256 and size, as is the
    folder's ``index.json``; a placed run whose layer map is missing is
    refused. Runs the placement could not place are listed as incomplete.
    """

    from darsia.fluidflower.map_placement import read_placement

    folder = Path(folder)
    placement = read_placement(folder)
    runs, incomplete = [], []
    for row in placement.get("runs") or []:
        run = str(row.get("run_id") or "")
        if row.get("status") != "placed":
            incomplete.append({"run_id": run, "status": str(row.get("status") or "")})
            continue
        layers = folder / str(row["layers"])
        if not layers.is_file():
            raise FileNotFoundError(f"{run}: the placed layer map is missing: {layers}")
        runs.append(
            {
                "run_id": run,
                "layers": _bound(layers),
                "shape_hw": list(row.get("shape_hw") or []),
                "geometry_model": row.get("geometry_model"),
                "offset_px": row.get("offset_px"),
            }
        )
    version = dict(placement.get("map_version") or {})
    return {
        "format": PROVENANCE_FORMAT,
        "placement": {"folder": str(folder), "index": _bound(folder / "index.json")},
        "reference_run": placement.get("reference_run"),
        "map_version": version,
        "approval": version.get("approval"),
        "geometry_index": placement.get("geometry_index"),
        "layers": placement.get("layers"),
        "runs": runs,
        "incomplete_runs": incomplete,
    }


def map_facies(provenance: Mapping[str, Any]) -> list[str]:
    """The facies names of the placed map's layers, from its placement index."""

    return sorted(
        {
            str(row.get("facies") or "")
            for row in (provenance.get("layers") or {}).values()
            if row.get("facies")
        }
    )


def adoption_blockers(
    provenance: Mapping[str, Any],
    *,
    transports: Mapping[str, Any] | None = None,
    facies_ids: Mapping[str, int] | None = None,
    facies: Iterable[str] = (),
) -> list[str]:
    """What an adoption of this placement still lacks, one line each; empty
    when nothing does.

    ``transports`` maps a run id to its transport from the camera grid to
    the run's label grid; ``facies_ids`` maps each facies name of the map to
    its id in the series' facies table; ``facies`` names the map's facies.
    """

    blockers = []
    approval = provenance.get("approval")
    version = dict(provenance.get("map_version") or {})
    if not approval or approval.get("sha256") != version.get("sha256"):
        blockers.append("The placed map version has no approval of its own SHA-256.")
    incomplete = [row["run_id"].upper() for row in provenance.get("incomplete_runs") or []]
    if incomplete:
        blockers.append(f"Not placed: {', '.join(incomplete)}.")
    placed = [row["run_id"] for row in provenance.get("runs") or []]
    missing = [run.upper() for run in placed if run not in (transports or {})]
    if missing:
        blockers.append(
            "No transport from the camera grid to the run's label grid: "
            f"{', '.join(missing)}."
        )
    names = sorted(set(facies))
    without = [name for name in names if name not in (facies_ids or {})]
    if without:
        blockers.append(f"No facies id for: {', '.join(without)}.")
    return blockers


def native_transport(geometry: Any) -> tuple[np.ndarray, tuple[int, int]]:
    """A run's homography from its baseline camera grid to its rig-corrected
    native grid, and the native grid's shape, from the model's
    ``baseline_native_grid``."""

    from darsia.fluidflower.map_placement import _matrix

    grid = dict(geometry.model.get("baseline_native_grid") or {})
    value = grid.get("H_baseline_camera_u_to_rig_corrected_native")
    shape = grid.get("shape_hw")
    if value is None or not shape:
        raise ValueError(f"{geometry.run_id}: the geometry has no baseline native grid")
    matrix = _matrix(value, f"{geometry.run_id} H_baseline_camera_u_to_rig_corrected_native")
    return matrix, (int(shape[0]), int(shape[1]))


def native_labels(
    layers: np.ndarray, reference: Any, target: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    """The reference run's layer map on another run's rig-corrected native grid.

    Returns the layer ids, the pixels the map does not reach (filled from its
    edge), the pixels that come from inside the map (its non-zero layers)
    and the composed matrix, reference camera grid to target native grid.
    """

    from darsia.fluidflower.geometry_raster_sources import carry_label_reference

    to_native, shape = native_transport(target)
    layers = np.ascontiguousarray(layers, dtype=np.uint8)
    target_matrix = to_native @ target.metric_to_grid
    carried = carry_label_reference(
        layers,
        # The carrier takes explicit nested lists of numbers.
        reference_matrix=reference.metric_to_grid.tolist(),
        target_matrix=target_matrix.tolist(),
        target_shape_hw=[shape[0], shape[1]],
        reference_filled=np.zeros(layers.shape, dtype=bool),
        source_fill_status="KNOWN",
        reference_support=layers != 0,
    )
    composition = target_matrix @ np.linalg.inv(reference.metric_to_grid)
    return (
        np.asarray(carried["labels"], dtype=np.uint8),
        np.asarray(carried["geometry_filled"], dtype=bool),
        np.asarray(carried["support"], dtype=bool),
        composition.tolist(),
    )


def prepare_adoption(
    destination_root: Path | str,
    layers: np.ndarray,
    *,
    reference_run: str,
    geometry_index: Path | str,
    runs: Sequence[str],
    roles: Mapping[int, str],
    version: Mapping[str, Any],
    names: Mapping[int, str] | None = None,
    facies: Mapping[int, str] | None = None,
    facies_ids: Mapping[str, int] | None = None,
    current_labels: Callable[[str], Mapping[str, Any] | None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write a new adoption preparation folder for every run; adopt nothing.

    ``layers`` is the approved map's layer ids on the reference run's camera
    grid. ``current_labels(run)`` gives the run's current corrected label
    field as ``{"metadata", "shape_hw", "source"}``; the prepared map takes
    its DarSIA metadata, and a run whose current field has another shape is
    refused. Per run, ``<run>/labels.npz`` holds ``array`` (the layer ids on
    the native grid, int32), ``metadata``, ``filled`` and ``support``.
    ``index.json`` (PREPARATION_FORMAT) binds the approved version, the
    geometry index and each run's composition and output by SHA-256, lists
    the regions (layer id, name, role, facies, facies id) and what an
    adoption still lacks. Facies without an id in ``facies_ids`` stay
    without one.
    """

    from darsia.fluidflower.map_placement import read_run_geometry
    from darsia.fluidflower.project import sha256_file

    index_path = Path(geometry_index)
    geometries = read_run_geometry(index_path)
    reference_key = str(reference_run).casefold()
    if reference_key not in geometries:
        raise ValueError(f"The geometry index has no model for the reference run {reference_run}")
    reference = geometries[reference_key]
    layers = np.ascontiguousarray(layers, dtype=np.uint8)
    if reference.shape_hw is not None and layers.shape != reference.shape_hw:
        raise ValueError("The map is not on the reference run's grid")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = Path(destination_root) / f"{stamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    # The map itself and its approval, bound for the Store's replay.
    np.savez_compressed(folder / "source_map.npz", array=layers)
    with (folder / "approval.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(version.get("approval") or {}), stream, indent=2, ensure_ascii=False)
    reference_bound = _bound(Path(reference.source))
    keys = [str(run).casefold() for run in runs]
    rows = []
    for number, run in enumerate(keys, 1):
        if progress is not None:
            progress(f"Preparing {run.upper()} on its label grid ({number} of {len(keys)})…")
        target = geometries.get(run)
        if target is None:
            rows.append({"run_id": run, "status": "no geometry"})
            continue
        try:
            labels, filled, support, composition = native_labels(layers, reference, target)
        except ValueError as error:
            rows.append({"run_id": run, "status": "no native grid", "error": str(error)})
            continue
        current = dict(current_labels(run) or {}) if current_labels is not None else {}
        if current.get("shape_hw") and tuple(current["shape_hw"]) != labels.shape:
            rows.append(
                {
                    "run_id": run,
                    "status": "grid differs",
                    "error": f"current label field {list(current['shape_hw'])}, "
                    f"native grid {list(labels.shape)}",
                }
            )
            continue
        run_folder = folder / run
        run_folder.mkdir()
        path = run_folder / "labels.npz"
        np.savez_compressed(
            path,
            array=labels.astype(np.int32),
            metadata=dict(current.get("metadata") or {}),
            filled=filled,
            support=support,
        )
        rows.append(
            {
                "run_id": run,
                "status": "prepared",
                "labels": {
                    "path": f"{run}/labels.npz",
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                },
                "shape_hw": list(labels.shape),
                "label_values": [int(value) for value in np.unique(labels)],
                "reference_model": reference_bound,
                "target_model": _bound(Path(target.source)),
                "composition_reference_u_to_native": composition,
                "filled_pixels": int(np.count_nonzero(filled)),
                "outside_support_pixels": int(np.count_nonzero(~support)),
                "current_labels": current.get("source"),
            }
        )
    regions = [
        {
            "layer_id": int(layer),
            "name": str((names or {}).get(layer, "")),
            "role": str(role),
            "facies": str((facies or {}).get(layer, "")),
            "facies_id": (facies_ids or {}).get(str((facies or {}).get(layer, ""))),
        }
        for layer, role in sorted(roles.items())
    ]
    blockers = []
    not_prepared = [row["run_id"].upper() for row in rows if row["status"] != "prepared"]
    if not_prepared:
        blockers.append(f"Not prepared: {', '.join(not_prepared)}.")
    without = sorted({row["facies"] for row in regions if row["facies"] and row["facies_id"] is None})
    if without:
        blockers.append(f"No facies id for: {', '.join(without)} ({FACIES_IDS_PENDING}).")
    index = {
        "format": PREPARATION_FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_run": reference_key,
        "map_version": dict(version),
        "geometry_index": {"path": str(index_path.resolve()), "sha256": sha256_file(index_path)},
        "source_map": {"path": "source_map.npz", **{
            key: value for key, value in _bound(folder / "source_map.npz").items() if key != "path"
        }},
        "approval": dict(version.get("approval") or {}),
        "composition": "H_targetU_to_targetNative @ H_target_mm_to_U @ H_referenceU_to_mm, "
        "nearest neighbour; filled and support flags kept",
        "regions": regions,
        "runs": rows,
        "blockers": blockers,
        "adopted": False,
    }
    temporary = folder / f".index.json.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, folder / "index.json")
    return dict(index, folder=str(folder))


def read_preparation(folder: Path | str) -> dict[str, Any]:
    """An adoption preparation folder's index, with ``folder`` added."""

    folder = Path(folder)
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    if index.get("format") != PREPARATION_FORMAT:
        raise ValueError(f"{folder} is not an adoption preparation folder")
    return dict(index, folder=str(folder))


def facies_table(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The rows of a configuration's full facies properties table (id, name,
    porosity, permeability), in id order."""

    from darsia.fluidflower.geometry_facies import _load_properties

    path = str(config.get("facies_properties_source_path") or "")
    if not path:
        raise ValueError("The geometry revision names no facies properties table")
    return [row for _key, row in sorted(_load_properties(Path(path)).items())]


def label_values(preparation: Mapping[str, Any]) -> list[int]:
    """Every label value in the prepared runs' maps (0 is outside the map)."""

    return sorted(
        {
            int(value)
            for row in preparation.get("runs") or []
            if row.get("status") == "prepared"
            for value in row.get("label_values") or []
        }
    )


def adoption_candidate(
    current: Mapping[str, Any],
    folder: Path | str,
    facies_ids: Mapping[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A geometry revision candidate adopting a prepared painted map.

    ``current`` is the current corrected workspace revision, a canonical
    edit of its imported parent; ``facies_ids`` maps each label value of the
    prepared maps to a facies id of the series' full facies table (the
    facies table of the map). The candidate keeps the parent, the frame and
    the photographs; its labels are the display run's prepared map, its
    regions take each facies' name, porosity and permeability from the
    table and the role of its layer (excluded outside the map), and every
    prepared run of the frame binding gets a ``painted-map-placement-v1``
    registration. Returns the candidate and a summary; nothing is saved.
    """

    from darsia.fluidflower import geometry_raster_sources as sources

    folder = Path(folder)
    preparation = read_preparation(folder)
    config = deepcopy(dict(current.get("configuration") or {}))
    if config.get("source_mode") != sources.CANONICAL_EDIT:
        raise ValueError(
            "The adoption builds on a corrected workspace revision that is a canonical edit; "
            "the current revision is not."
        )
    parent = config["canonical_import_parent"]
    binding = dict((config.get("frame_binding") or {}).get("runs") or {})
    display = str(config["frame_binding"]["display_reference_run"])
    prepared = {
        row["run_id"]: row for row in preparation.get("runs") or [] if row.get("status") == "prepared"
    }
    if display not in prepared:
        raise ValueError(f"The display reference run {display.upper()} is not prepared")
    values = label_values(preparation)
    missing = [value for value in values if value not in facies_ids]
    if missing:
        raise ValueError(
            f"No facies id for label(s) {', '.join(str(value) for value in missing)} "
            f"({FACIES_IDS_PENDING})"
        )
    table = {int(row["facies_id"]): row for row in facies_table(config)}
    layers = {int(row["layer_id"]): row for row in preparation.get("regions") or []}
    regions = []
    for value in values:
        facies_id = int(facies_ids[value])
        if facies_id not in table:
            raise ValueError(f"Facies id {facies_id} is not in the series' facies table")
        row = table[facies_id]
        regions.append(
            {
                "label_id": value,
                "facies_id": facies_id,
                "name": row["name"],
                "analysis_role": str((layers.get(value) or {}).get("role") or "excluded"),
                "porosity": row["porosity"],
                "permeability": row["permeability"],
                "enabled": True,
            }
        )
    measurement = _bound(folder / "index.json")
    authority = _bound(folder / "approval.json")
    source_map = _bound(folder / "source_map.npz")
    rows, not_prepared = [], []
    for run, row in binding.items():
        entry = prepared.get(run)
        if entry is None:
            not_prepared.append(run)
            continue
        labels = folder / entry["labels"]["path"]
        photograph = (row.get("fields") or {}).get("shape_corrected_baseline") or {}
        rows.append(
            {
                "run_id": run,
                "enabled": True,
                "geometry_choice": "run_override",
                "acquisition_state": "unknown",
                "labels_path": str(labels.resolve()),
                "source_image_path": str(photograph.get("path") or ""),
                "registration": {
                    "kind": sources.PAINTED,
                    "parent_revision_id": parent["revision_id"],
                    "status": "painted",
                    "authority": authority,
                    "measurement": measurement,
                    "labels": _bound(labels),
                    "source_image": {
                        key: photograph.get(key) for key in ("path", "sha256", "size_bytes")
                    },
                    "source_map": source_map,
                    "reference_model": entry["reference_model"],
                    "target_model": entry["target_model"],
                },
            }
        )
    config["run_geometry"] = dict(config.get("run_geometry") or {}, runs=rows)
    config["labels_path"] = str((folder / prepared[display]["labels"]["path"]).resolve())
    config["regions"] = regions
    config["provenance"] = {
        "origin": "approved painted map placed in every run",
        "parent_revision_id": parent["revision_id"],
        "previous_revision_id": current.get("revision_id"),
        "preparation": measurement,
        "map_version": {
            key: (preparation.get("map_version") or {}).get(key) for key in ("name", "sha256")
        },
        "registered_geometry_authorized": False,
    }
    summary = {
        "runs": [row["run_id"] for row in rows],
        "not_prepared": not_prepared,
        "regions": regions,
        "display_reference_run": display,
    }
    return config, summary


__all__ = [
    "FACIES_IDS_PENDING",
    "PREPARATION_FORMAT",
    "PROVENANCE_FORMAT",
    "adoption_blockers",
    "adoption_candidate",
    "facies_table",
    "label_values",
    "map_facies",
    "native_labels",
    "native_transport",
    "placement_provenance",
    "prepare_adoption",
    "read_preparation",
]

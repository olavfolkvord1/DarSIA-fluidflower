"""The provenance of a painted map placed in all runs, for its adoption.

A placement review folder (see :mod:`map_placement`) holds, per run, the
approved map placed on the run's camera grid. Before such a placement can
be adopted as a geometry revision, its provenance is bound: the approved
version and its approval, the placement index, the geometry index, and each
placed run's layer map, all by SHA-256. :func:`adoption_blockers` names what
an adoption still lacks: runs without a placement, runs without a transport
from their camera grid to their label grid, and facies without an id.

Nothing here writes a revision, a map or a run configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROVENANCE_FORMAT = "darsia-fluidflower-painted-placement-provenance-v1"


def _bound(path: Path) -> dict[str, Any]:
    """A file bound by its path, SHA-256 and size."""

    from darsia.fluidflower.project import sha256_file

    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


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


__all__ = ["PROVENANCE_FORMAT", "adoption_blockers", "map_facies", "placement_provenance"]

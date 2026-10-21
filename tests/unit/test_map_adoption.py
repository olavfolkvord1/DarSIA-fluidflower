"""The provenance of a placed painted map, and what its adoption still lacks.

A synthetic placement in a temporary folder (the geometry index form of the
placement tests); nothing is adopted.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import map_adoption as ma
from darsia.fluidflower import map_placement as mp

from .test_map_placement import REFERENCE_H, SHIFTED_H, _index, _layers


def _placement(tmp_path, version):
    index = _index(tmp_path / "geometry", {"run1": REFERENCE_H, "run2": SHIFTED_H})
    return mp.place_map_in_runs(
        tmp_path / "reviews",
        _layers(),
        reference_run="run2",
        geometry_index=index,
        runs=["run1", "run2", "run3"],
        photo_loader=lambda run, geometry: np.full(geometry.shape_hw + (3,), 50, dtype=np.uint8),
        roles={1: "free_water", 2: "porous_medium", 3: "porous_medium"},
        facies={1: "water", 2: "C", 3: "ESF"},
        version=version,
        measure_offsets=False,
    )


def test_the_provenance_binds_every_placed_run_by_sha256(tmp_path):
    version = {"name": "Map A", "sha256": "a" * 64, "approval": {"sha256": "a" * 64}}
    result = _placement(tmp_path, version)
    record = ma.placement_provenance(result["folder"])
    assert record["format"] == ma.PROVENANCE_FORMAT
    assert record["reference_run"] == "run2" and record["approval"] == {"sha256": "a" * 64}
    index = record["placement"]["index"]
    with open(index["path"], "rb") as stream:
        assert index["sha256"] == hashlib.sha256(stream.read()).hexdigest()
    assert [row["run_id"] for row in record["runs"]] == ["run2", "run1"]
    for row in record["runs"]:
        with open(row["layers"]["path"], "rb") as stream:
            assert row["layers"]["sha256"] == hashlib.sha256(stream.read()).hexdigest()
    assert record["incomplete_runs"] == [{"run_id": "run3", "status": "no geometry"}]
    # A placed run whose layer map went missing is refused.
    (Path(result["folder"]) / "run1" / "layers.npz").unlink()
    with pytest.raises(FileNotFoundError, match="run1: the placed layer map is missing"):
        ma.placement_provenance(result["folder"])


def test_the_blockers_name_what_an_adoption_still_lacks(tmp_path):
    record = ma.placement_provenance(
        _placement(tmp_path, {"name": "Map A", "sha256": "a" * 64, "approval": {"sha256": "a" * 64}})["folder"]
    )
    assert ma.map_facies(record) == ["C", "ESF", "water"]
    assert ma.adoption_blockers(record, facies=["ESF", "C"]) == [
        "Not placed: RUN3.",
        "No transport from the camera grid to the run's label grid: RUN2, RUN1.",
        "No facies id for: C, ESF.",
    ]
    assert ma.adoption_blockers(
        record, transports={"run1": {}, "run2": {}}, facies_ids={"ESF": 1, "C": 2}, facies=["ESF", "C"]
    ) == ["Not placed: RUN3."]
    record["approval"] = {"sha256": "b" * 64}
    assert ma.adoption_blockers(record)[0] == (
        "The placed map version has no approval of its own SHA-256."
    )

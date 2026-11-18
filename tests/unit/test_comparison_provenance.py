"""Compared images and actual acquisition times survive the distance CSV."""

import csv

import pytest

from darsia.presets.workflows.analysis.domains import write_rows
from darsia.presets.workflows.comparison.mass_wasserstein import (
    ComparedFrame,
    WassersteinDistance,
    distance_rows,
)


def test_actual_frame_times_survive_csv_and_missing_frames_stay_empty(tmp_path):
    frames = {
        "a": ComparedFrame("first/image.ARW", "2023-01-01T13:00:00+01:00", 1.98),
        "b": ComparedFrame("second/image.ARW", "2023-02-01T14:00:00", 2.03),
        "c": None,
    }
    distances = [
        WassersteinDistance(2.0, "a", "b", "sand", "not_converged", 0.125),
        WassersteinDistance(2.0, "b", "c", "sand", "missing"),
    ]
    rows = distance_rows(distances, frames)
    path = tmp_path / "distances.csv"
    write_rows(path, rows, list(rows[0]))
    with path.open(encoding="utf-8", newline="") as stream:
        first, missing = list(csv.DictReader(stream))
    assert first["case_id_1"] == "first/image.ARW"
    assert first["case_id_2"] == "second/image.ARW"
    assert first["acquired_at_1"].endswith("+01:00")
    assert first["acquired_at_2"] == "2023-02-01T14:00:00"
    assert float(first["elapsed_1_h"]) == 1.98
    assert float(first["elapsed_2_h"]) == 2.03
    assert float(first["offset_1_s"]) == pytest.approx(-72)
    assert float(first["offset_2_s"]) == pytest.approx(108)
    assert first["status"] == "not_converged"
    assert float(first["distance_kg_m"]) == 0.125
    assert missing["case_id_1"] == "second/image.ARW"
    assert all(
        missing[key] == ""
        for key in ("case_id_2", "acquired_at_2", "elapsed_2_h", "offset_2_s")
    )


def test_a_computed_distance_requires_both_image_identities():
    distance = WassersteinDistance(2.0, "a", "b", "sand", "success", 0.0)
    with pytest.raises(ValueError, match="no selected image"):
        distance_rows([distance], {"a": None, "b": None})
    with pytest.raises(KeyError):
        distance_rows([distance], {})

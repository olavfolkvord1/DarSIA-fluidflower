"""Layer painting tab: placing the approved map in all runs, for review.

"Place in all runs…" in the Versions box takes the selected approved version,
places it in every run through each run's measured geometry and opens the
review window: the runs in review order with their offsets and checks, and
each run's overlay. It writes a new review folder only. A synthetic 80 x 60
photograph, a synthetic geometry index and a project in a temporary folder;
no real series is used.
"""

import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import map_placement as mp
from darsia.fluidflower import series_layer_view as view_module
from darsia.fluidflower.map_placement_view import MapPlacementDialog

from .test_painting_corrections import (  # noqa: F401
    _proposal,
    _show,
    view,
)

SHAPE = [60, 80]
# Metric millimetres to grid pixels; the second run sees the rig 3 px further right.
REFERENCE_H = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]]
SHIFTED_H = [[0.1, 0.0, 3.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]]


def _index(folder):
    folder.mkdir(parents=True)
    runs = {}
    for run, matrix in (("ac52", REFERENCE_H), ("ac53", SHIFTED_H)):
        (folder / run).mkdir()
        model = {
            "format": mp.GEOMETRY_FORMAT,
            "run_id": run,
            "H_metric_mm_to_camera_u": matrix,
            "reference_image": {"path": f"{run}.arw"},
            "shape_hw": SHAPE,
        }
        path = folder / run / "reference_model.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        runs[run] = {"run": run, "reference_model": {"path": str(path)}}
    path = folder / "index.json"
    path.write_text(json.dumps({"format": mp.GEOMETRY_FORMAT, "runs": runs}), encoding="utf-8")
    return path


def test_the_typical_offset_change_weighs_pairs_by_their_samples():
    from darsia.fluidflower.map_placement_view import offset_summary

    def row(*pairs):
        return {
            "layer_offsets": {
                "boundaries": [
                    {"change_from_reference_mm": change, "measured_sample_count": count}
                    for change, count in pairs
                ]
            }
        }

    # A short unsteady pair moves neither the typical change nor the count.
    assert offset_summary(row((0.3, 1900), (-0.7, 1700), (-25.4, 60))) == (0.3, 0)
    assert offset_summary(row((0.3, 1900), (6.0, 1700), (-25.4, 210))) == (0.3, 2)
    assert offset_summary(row((None, 100))) == (None, 0)
    assert offset_summary({}) == (None, 0)


def _files(folder):
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_an_approved_version_is_placed_in_all_runs_for_review(view, qtbot, tmp_path, monkeypatch):
    assert view.place_version_button.text() == "Place in all runs…"
    _show(view, _proposal())
    record = view.save_version("Map A")
    view.versions_table.selectRow(0)
    # Only an approved version can be placed.
    assert not view.place_version_button.isEnabled()
    assert not view.place_version(record, tmp_path / "none.json", runs=["ac52"])
    assert "Approve the version" in view.versions_note.text()
    view.approve_version(record)
    view.versions_table.selectRow(0)
    assert view.place_version_button.isEnabled()
    painting_before = _files(lp.painting_folder(view.project))
    loaded = []

    def photo(run, geometry):
        loaded.append(run)
        return np.full(tuple(geometry.shape_hw) + (3,), 90, dtype=np.uint8)

    index = _index(tmp_path / "geometry")
    assert view.place_version(
        record, index, runs=["ac52", "ac53", "ac54"], first_runs=["AC54"], photo_loader=photo
    )
    qtbot.waitUntil(lambda: view.last_placement is not None and not view.busy(), timeout=30000)
    result = view.last_placement
    folder = lp.painting_folder(view.project).parent / "map_placement"
    assert [path.parent for path in folder.glob("*/index.json")] == [Path(result["folder"])]
    assert [row["run_id"] for row in result["runs"]] == ["ac52", "ac54", "ac53"]
    assert [row["status"] for row in result["runs"]] == ["placed", "no geometry", "placed"]
    assert loaded == ["ac52", "ac53"]
    assert result["map_version"]["sha256"] == record["sha256"]
    assert result["map_version"]["approval"]["sha256"] == record["sha256"]
    with np.load(f"{result['folder']}/ac52/layers.npz") as saved:
        reference = saved["layers"]
    with np.load(f"{result['folder']}/ac53/layers.npz") as saved:
        shifted = saved["layers"]
    # Fine sand left, coarse sand right; in AC53 everything is 3 px further right.
    assert reference[30, 10] == 2 and reference[30, 60] == 3
    assert np.array_equal(shifted[:, 3:], reference[:, :-3])
    assert "Placed in 2 of 3 runs" in view.versions_note.text()
    # The review window lists the runs with their offsets and shows the overlays.
    dialog = view._placement_dialog
    assert isinstance(dialog, MapPlacementDialog) and dialog.isVisible()
    cells = [
        [dialog.table.item(row, column).text() for column in range(dialog.table.columnCount())]
        for row in range(dialog.table.rowCount())
    ]
    assert [row[:3] for row in cells] == [
        ["AC52", "placed", "0.0 / 0.0"],
        ["AC54", "no geometry", ""],
        ["AC53", "placed", "3.0 / 3.0"],
    ]
    # A uniform photograph has no edge to measure; the 3 columns the map does
    # not reach lie inside the painted area, so they are gaps to fill.
    assert cells[2][3:7] == ["none measured", "0", str(3 * 60), "0"]
    assert dialog.legend.text().count("■") == 2
    assert "layer offsets" in dialog.details.toPlainText()
    assert not dialog.view.item.pixmap().isNull()
    assert dialog.view.item.pixmap().size().width() == 80
    assert not dialog.show_run(1) and "no geometry" in dialog.status.text()
    assert "not measured" in dialog.details.toPlainText()
    assert dialog.show_run(2) and "AC53" in dialog.status.text()
    # Nothing in the painting folder changed: no map, session, version or approval.
    assert _files(lp.painting_folder(view.project)) == painting_before
    dialog.close()
    # The review opens again from its folder, without placing anew.
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        lambda *args, **kwargs: (str(Path(result["folder"]) / "index.json"), ""),
    )
    view.open_placement_button.click()
    again = view._placement_dialog
    assert again is not dialog and again.isVisible() and again.table.rowCount() == 3
    again.close()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[1]))
    assert view.open_placement(tmp_path) is None
    assert warnings == ["Could not open the placement review"]


def test_the_button_asks_for_the_geometry_and_the_runs_to_review_first(
    view, monkeypatch, tmp_path
):
    _show(view, _proposal())
    record = view.save_version("Map A")
    view.approve_version(record)
    view.versions_table.selectRow(0)
    settings = view_module.QSettings(str(tmp_path / "settings.ini"), view_module.QSettings.Format.IniFormat)
    monkeypatch.setattr(view_module, "QSettings", lambda *args: settings)
    index = _index(tmp_path / "geometry")
    monkeypatch.setattr(view, "_series_runs", lambda: ["ac52", "ac53"])
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(index), "")
    )
    asked, started = [], []

    def ask(_parent, title, label, *args, **kwargs):
        asked.append((title, label, kwargs.get("text")))
        return "ac53", False

    monkeypatch.setattr(QInputDialog, "getText", ask)
    monkeypatch.setattr(view, "place_version", lambda *args, **kwargs: started.append(kwargs))
    view.place_version_button.click()
    title, label, text = asked[0]
    assert title == "Place in all runs" and "in all 2 runs" in label and record["sha256"][:12] in label
    assert "changes no map, geometry revision or run configuration" in label
    assert started == [] and text == ""
    # The geometry index is remembered; the runs to review first after an accepted answer.
    assert settings.value("placement/geometry_index") == str(index)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: (" ac53 , ac60 ", True))
    view.place_version_button.click()
    assert started == [{"runs": ["ac52", "ac53"], "first_runs": ["ac53", "ac60"]}]
    assert settings.value("placement/first_runs") == "ac53 , ac60"

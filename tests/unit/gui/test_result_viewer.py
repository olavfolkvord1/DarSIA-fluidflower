"""Unit tests for the one viewer of stored results."""

import gc
import threading
import weakref
from time import monotonic

import numpy as np
import pytest
from matplotlib.colors import to_rgb
from PySide6.QtCore import QLocale, Qt, QTimer

from darsia.gui import result_viewer
from darsia.gui.result_viewer import (
    OTHER,
    Category,
    Field,
    ResultViewer,
    Source,
    _class_colours,
)
from darsia.utils import results
from darsia.utils.locations import Frame, Locations

FRAMES = (Frame("first", "a.jpg"), Frame("first", "b.jpg"), Frame("second", "a.jpg"))
"""A run's frames, one photo name in two sessions."""
HOURS = (0.5, 1.0, 2.0)
FIELDS = (
    Field("mass", "Total CO2 mass", "g/L"),
    Field("saturation_g", "Gas saturation", limit=1.0),
)


def _maps(folder):
    """The mass maps' reader of this test."""
    with np.load(folder / "maps.npz") as archive:
        return {key: archive[key] for key in archive.files}


def _project(folder, runs=("run",), skip=(), factor=1.0):
    """A project whose runs' frames have stored mass maps, the mass a frame's index
    and its run's number apart, times ``factor``."""
    sessions = {"first": "photos/first", "second": "photos/second"}
    locations = Locations(folder, "results", {run: sessions for run in runs})
    for number, run in enumerate(runs, 1):
        for index, frame in enumerate(FRAMES):
            mass = np.full((4, 6), 0.1234 * (index + 1) * number * factor, np.float32)
            mass[0, 0] = np.nan
            saturation = np.zeros((4, 6), np.float32)
            saturation[1:3, 2:4] = 0.25

            def write(folder, mass=mass, saturation=saturation):
                np.savez(folder / "maps.npz", mass=mass, saturation_g=saturation)

            if (run, index) not in skip:
                location = locations.result("mass-maps", run=run, frame=frame)
                token = f"{number}{index}" * 16
                results.publish(location, write, {}, token, root=locations.results)
    return locations


def _runs(runs=("run",)):
    """Each run's frames with their times."""
    return {run: list(zip(FRAMES, HOURS)) for run in runs}


def _shown(qtbot, viewer):
    """Wait until the viewer has drawn its frame."""
    qtbot.waitUntil(lambda: not viewer.busy, timeout=20000)


def _stored(locations, index, run="run"):
    """The stored mass of a run's frame."""
    return _maps(locations.result("mass-maps", run=run, frame=FRAMES[index]))["mass"]


def test_a_read_in_progress_never_blocks_the_window_thread(qtbot, tmp_path):
    """While a stored result is read, the window thread runs and shows the read
    with its elapsed time; then the latest frame asked for is read and shown, and a
    frame passed over meanwhile is not read."""
    locations = _project(tmp_path)
    reading, release, read = threading.Event(), threading.Event(), []

    def slow(folder):
        reading.set()
        release.wait(10)
        read.append(folder.parent.name)
        return _maps(folder)

    viewer = ResultViewer(locations, [Source("mass-maps", "Mass maps", slow, FIELDS)])
    qtbot.addWidget(viewer)
    try:
        started = monotonic()
        viewer.set_runs(_runs())
        assert reading.wait(5)
        viewer.show_frame(1)
        viewer.show_frame(2)
        assert monotonic() - started < 2
        assert viewer.busy and not viewer.busy_bar.isHidden()
        seen = []
        QTimer.singleShot(300, lambda: seen.append(viewer.status.text()))
        qtbot.waitUntil(lambda: bool(seen), timeout=5000)
        assert viewer.busy and seen[0].startswith("Reading run, frame 3 of 3… ")
    finally:
        release.set()
    _shown(qtbot, viewer)
    np.testing.assert_array_equal(viewer.panels[0].values, _stored(locations, 2))
    assert read == ["first", "second"] and viewer.busy_bar.isHidden()


def test_a_new_run_shows_nothing_of_the_last_until_it_is_read(qtbot, tmp_path):
    """While another run is read, no panel holds the last run's frame, its colour
    scale cannot be changed, and no scale of the last run's values is kept."""
    locations = _project(tmp_path, runs=("one", "two"))
    release = threading.Event()
    release.set()

    def gated(folder):
        release.wait(10)
        return _maps(folder)

    viewer = ResultViewer(locations, [Source("mass-maps", "Mass maps", gated, FIELDS)])
    qtbot.addWidget(viewer)
    viewer.set_runs(_runs(("one", "two")))
    _shown(qtbot, viewer)
    panel = viewer.panels[0]
    assert panel.values is not None and panel.fit_button.isEnabled()
    release.clear()
    try:
        viewer.select_run("two")
        assert panel.values is None and panel.pixels is None
        assert not panel.fit_button.isEnabled() and not panel.scale_edit.isEnabled()
        panel.scale_set.emit(panel.number, 9.0)
    finally:
        release.set()
    _shown(qtbot, viewer)
    assert panel.scale_edit.text() == "0.25" and panel.fit_button.isEnabled()


def test_a_frame_not_stored_is_told_from_a_share_that_gives_no_answer(qtbot, tmp_path):
    """A frame without a stored result is not stored; one whose read gets no answer
    from the share is not available now, never not stored, and is read again when
    chosen again."""
    locations = _project(tmp_path, skip={("run", 0)})
    outages = [results.Unavailable("No answer from the share")]

    def share(folder):
        if outages:
            raise outages.pop()
        return _maps(folder)

    viewer = ResultViewer(locations, [Source("mass-maps", "Mass maps", share, FIELDS)])
    qtbot.addWidget(viewer)
    viewer.set_runs(_runs())
    _shown(qtbot, viewer)
    assert viewer.panels[0].message.endswith("first/a.jpg, is not stored.")
    viewer.show_frame(1)
    _shown(qtbot, viewer)
    message = viewer.panels[0].message
    assert "is not available now" in message and "not stored" not in message
    viewer.show_frame(1)
    _shown(qtbot, viewer)
    assert viewer.panels[0].values is not None


def test_the_colour_scale_is_kept_per_run_and_field(qtbot, tmp_path):
    """A field starts at its physical limit or fitted to the first frame shown, and
    a fitted or typed scale stays for the run's frames, not for another run. A
    scale with decimals is typed with a point also where the system writes a
    comma."""
    locations = _project(tmp_path, runs=("one", "two"))
    system = QLocale()
    QLocale.setDefault(
        QLocale(QLocale.Language.NorwegianBokmal, QLocale.Country.Norway)
    )
    try:
        viewer = ResultViewer(
            locations,
            [Source("mass-maps", "Mass maps", _maps, FIELDS)],
            contents=("mass-maps/mass", "mass-maps/saturation_g"),
            layout="side_by_side",
        )
        qtbot.addWidget(viewer)
        viewer.set_runs(_runs(("one", "two")))
        _shown(qtbot, viewer)
        mass, saturation = viewer.panels[:2]
        assert (mass.scale_edit.text(), saturation.scale_edit.text()) == ("0.13", "1")
        viewer.show_frame(2)
        _shown(qtbot, viewer)
        assert mass.scale_edit.text() == "0.13"
        mass.fit()
        _shown(qtbot, viewer)
        assert mass.scale_edit.text() == "0.38"
        for typed, kept in (("0", "0.38"), ("0.5", "0.5")):
            mass.scale_edit.clear()
            qtbot.keyClicks(mass.scale_edit, typed)
            qtbot.keyClick(mass.scale_edit, Qt.Key.Key_Return)
            _shown(qtbot, viewer)
            assert mass.scale_edit.text() == kept
    finally:
        QLocale.setDefault(system)
    viewer.show_frame(0)
    _shown(qtbot, viewer)
    assert mass.scale_edit.text() == "0.5"
    viewer.select_run("two")
    _shown(qtbot, viewer)
    assert mass.scale_edit.text() == "0.25"


def test_other_results_are_read_from_their_own_root(qtbot, tmp_path):
    """Results of another root with the same runs, e.g. of another Setup variant,
    are read anew, with new colour scales; a refused switch changes nothing, and new
    runs keep the run and frame shown."""
    first = _project(tmp_path / "first")
    second = _project(tmp_path / "second", factor=10)
    maps = Source("mass-maps", "Mass maps", _maps, FIELDS)
    viewer = ResultViewer(first, [maps])
    qtbot.addWidget(viewer)
    viewer.set_runs(_runs())
    viewer.show_frame(2)
    _shown(qtbot, viewer)
    with pytest.raises(ValueError):
        viewer.set_results([maps], contents=("mass-maps/area",), locations=second)
    viewer.set_runs(_runs())
    _shown(qtbot, viewer)
    assert viewer.locations is first and viewer.slider.value() == 2
    np.testing.assert_array_equal(viewer.panels[0].values, _stored(first, 2))
    viewer.set_results([maps], locations=second)
    _shown(qtbot, viewer)
    np.testing.assert_array_equal(viewer.panels[0].values, _stored(second, 2))
    assert viewer.panels[0].scale_edit.text() == "3.8"


def test_the_slider_and_play_step_through_the_frames(qtbot, tmp_path):
    """The slider's frame is read once it rests; Play starts again from the first
    frame, shows every frame, stops at the last, and stops when the viewer is
    hidden."""
    locations = _project(tmp_path)
    read = []

    def counted(folder):
        read.append((folder.parent.name, folder.name))
        return _maps(folder)

    viewer = ResultViewer(
        locations, [Source("mass-maps", "Mass maps", counted, FIELDS)]
    )
    qtbot.addWidget(viewer)
    viewer.set_runs(_runs())
    _shown(qtbot, viewer)
    viewer.slider.setValue(1)
    viewer.slider.setValue(2)
    assert not viewer.busy
    qtbot.waitUntil(lambda: ("second", "a.jpg") in read and not viewer.busy)
    assert ("first", "b.jpg") not in read
    viewer.play_button.click()
    assert viewer.slider.value() == 0 and viewer.play_button.text() == "Pause"
    qtbot.waitUntil(lambda: not viewer.play_button.isChecked(), timeout=20000)
    assert viewer.slider.value() == 2 and viewer.play_button.text() == "Play"
    assert ("first", "b.jpg") in read
    viewer.show()
    viewer.play_button.click()
    viewer.hide()
    assert not viewer.play_button.isChecked()


def test_a_viewer_closed_during_a_read_shows_the_frame_when_shown_again(
    qtbot, tmp_path
):
    """A frame still being read when the viewer closes is shown when it opens
    again."""
    locations = _project(tmp_path)
    release = threading.Event()

    def gated(folder):
        release.wait(10)
        return _maps(folder)

    viewer = ResultViewer(locations, [Source("mass-maps", "Mass maps", gated, FIELDS)])
    qtbot.addWidget(viewer)
    viewer.show()
    try:
        viewer.set_runs(_runs())
        viewer.close()
    finally:
        release.set()
    qtbot.wait(300)
    viewer.show()
    _shown(qtbot, viewer)
    np.testing.assert_array_equal(viewer.panels[0].values, _stored(locations, 0))
    assert viewer.status.text() == ""


def test_large_fields_are_shown_at_a_stride_and_few_frames_are_kept(
    qtbot, tmp_path, monkeypatch
):
    """A field longer than the display shows every n-th cell, and only the last
    frames read stay in memory."""
    monkeypatch.setattr(result_viewer, "DISPLAY_PIXELS", 3)
    monkeypatch.setattr(result_viewer, "CACHED", 2)
    locations = _project(tmp_path)
    read = []

    def counted(folder):
        read.append(folder)
        return _maps(folder)

    viewer = ResultViewer(
        locations, [Source("mass-maps", "Mass maps", counted, FIELDS)]
    )
    qtbot.addWidget(viewer)
    viewer.set_runs(_runs())
    _shown(qtbot, viewer)
    np.testing.assert_array_equal(
        viewer.panels[0].values, _stored(locations, 0)[::2, ::2]
    )
    for index in (1, 2, 0):
        viewer.show_frame(index)
        _shown(qtbot, viewer)
    assert len(read) == 4


def test_categorical_fields_and_photographs_have_no_colour_scale(qtbot, tmp_path):
    """A categorical field is drawn in its classes' colours, a code it does not
    declare as «Other label», and a photograph as it is; neither has a colour
    scale."""
    locations = _project(tmp_path)
    codes = np.array([[0, 1, 7], [1, 1, 0]], np.uint8)
    classes = (Category(0, "Water", "#a9cfee"), Category(1, "Gas", "#dc398b"))

    def labelled(folder):
        photo = np.linspace(0, 2, 2 * 3 * 3).reshape(2, 3, 3)
        return {**_maps(folder), "labels": codes, "photo": photo}

    fields = (
        Field("labels", "Phase map", classes=classes),
        Field("photo", "Photograph", photo=True),
    )
    viewer = ResultViewer(
        locations,
        [Source("mass-maps", "Mass maps", labelled, fields)],
        layout="side_by_side",
    )
    qtbot.addWidget(viewer)
    viewer.set_runs(_runs())
    _shown(qtbot, viewer)
    for panel in viewer.panels[:2]:
        assert panel.pixels is not None and panel.values is None
        assert panel.scale_row.isHidden()
    rgb, legend = _class_colours(codes, classes)
    assert [patch.get_label() for patch in legend] == ["Water", "Gas", OTHER[0]]
    np.testing.assert_allclose(rgb[0, 1], to_rgb("#dc398b"))
    np.testing.assert_allclose(rgb[0, 2], to_rgb(OTHER[1]))


def test_a_viewer_without_parent_is_freed_at_once(qtbot, tmp_path):
    """No connection keeps a viewer alive: one without parent is gone as soon as it
    is no longer referenced, without the garbage collector, which could otherwise
    free it later in another thread."""
    locations = _project(tmp_path)
    viewer = ResultViewer(
        locations, [Source("mass-maps", "Mass maps", _maps, FIELDS)], layout="pyramid"
    )
    viewer.set_runs(_runs())
    _shown(qtbot, viewer)
    gc.collect()
    gc.disable()
    try:
        freed = weakref.ref(viewer)
        del viewer
        assert freed() is None
    finally:
        gc.enable()


def test_the_viewer_fits_a_laptop_screen(qtbot, tmp_path):
    """Shown at 1,000 × 640 pixels with three panels of continuous fields and a
    message, every wrapped line of text has its height and every map at least 150
    pixels."""
    locations = _project(tmp_path)
    viewer = ResultViewer(
        locations,
        [Source("mass-maps", "Mass maps", _maps, FIELDS)],
        contents=("mass-maps/mass", "mass-maps/saturation_g", "mass-maps/mass"),
        layout="pyramid",
    )
    qtbot.addWidget(viewer)
    viewer.resize(1000, 640)
    viewer.show()
    viewer.set_runs(_runs())
    _shown(qtbot, viewer)
    viewer.status.setText("The frame could not be shown: " + "no answer. " * 6)
    labels = [viewer.frame_label, viewer.status]
    labels += [panel.notice for panel in viewer.panels]

    def laid_out():
        return all(
            label.height() >= label.heightForWidth(label.width())
            for label in labels
            if label.isVisible()
        )

    qtbot.waitUntil(laid_out)
    assert viewer.size().width() == 1000 and viewer.size().height() == 640
    assert all(panel.canvas.height() >= 150 for panel in viewer.panels)

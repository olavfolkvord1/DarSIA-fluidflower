"""Unit tests for the locations of a project's results and photos."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from darsia.experiment.experiment import ProtocolledExperiment
from darsia.utils import results
from darsia.utils.locations import Frame, Locations, normal_path

SESSIONS = {"run": {"first": "photos/first", "second": "photos/second"}}
"""One run whose photos come from two sessions, relative to the project."""
NAME = "DSC00001.ARW"


def _project(folder):
    """A project whose run has a photo of one name in each of its two sessions."""
    for session in SESSIONS["run"]:
        photos = folder / "photos" / session
        photos.mkdir(parents=True)
        (photos / NAME).write_text(session)
    return folder


def _writer(text):
    """A method's own writer of one text file."""
    return lambda folder: (folder / "image.txt").write_text(text)


def _reader(folder):
    """The method's own reader of its text file."""
    return (folder / "image.txt").read_text()


def test_a_moved_project_still_resolves(tmp_path, monkeypatch):
    """A result published in the project and the photo it was made from are found
    where the project has moved to."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    project, frame = _project(tmp_path / "project"), Frame("first", NAME)
    location = Locations(project, "results", SESSIONS).result(
        "corrected-image", run="run", frame=frame
    )
    results.publish(
        location, _writer("corrected"), {}, "1" * 32, root=project / "results"
    )

    moved = Locations(project.rename(tmp_path / "moved"), "results", SESSIONS)
    location = moved.result("corrected-image", run="run", frame=frame)
    assert results.read(location, _reader) == "corrected"
    assert moved.photo("run", frame).read_text() == "first"


def test_one_photo_name_in_two_sessions(tmp_path):
    """One photo name in two sessions of a run gives each session's photo and two
    results."""
    locations = Locations(_project(tmp_path), "results", SESSIONS)
    frames = [Frame(session, NAME) for session in SESSIONS["run"]]
    assert [locations.photo("run", frame).read_text() for frame in frames] == [
        "first",
        "second",
    ]
    for token, frame in enumerate(frames):
        location = locations.result("phase-maps", run="run", frame=frame)
        results.publish(
            location,
            _writer(frame.session),
            {},
            str(token) * 32,
            root=locations.results,
        )
    assert [
        results.read(locations.result("phase-maps", run="run", frame=frame), _reader)
        for frame in frames
    ] == ["first", "second"]


def _link(link: Path, target: Path) -> None:
    """A folder that reaches another, as a mapped drive reaches its share: a
    junction on Windows, a symbolic link elsewhere."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def test_a_session_outside_the_project_has_its_normal_path():
    """A session folder outside the project, e.g. ``../raw/<run>`` of a
    project on a share, is in its normal form, without ``..``."""
    project = Path("//server/share/series/project")  # a share, as a task sees it
    folder = normal_path("../raw/run/first", project)
    assert folder == Path("//server/share/series/raw/run/first")
    assert ".." not in folder.parts
    assert normal_path(project / ".." / "raw") == folder.parents[1]


def test_a_project_through_a_mapped_root_has_its_resolved_paths(tmp_path):
    """A project reached through a mapped root (here a link) has the locations of
    its resolved folder, so a job made through one and its tasks, run in the other,
    give the same paths; a photo outside the project is found."""
    share = tmp_path / "share"
    _project(share / "raw")
    (share / "project").mkdir()
    _link(tmp_path / "mapped", share)
    sessions = {"run": {s: f"../raw/photos/{s}" for s in SESSIONS["run"]}}
    mapped = Locations(tmp_path / "mapped" / "project", "results", sessions)
    resolved = Locations(share / "project", "results", sessions)
    frame = Frame("second", NAME)
    assert mapped.project == resolved.project == share / "project"
    assert mapped.sessions == resolved.sessions
    assert mapped.photo("run", frame) == share / "raw" / "photos" / "second" / NAME
    assert mapped.photo("run", frame).read_text() == "second"
    phase_maps = resolved.results / "phase-maps" / "run"
    assert mapped.result("phase-maps", run="run") == phase_maps


def test_a_photo_outside_the_project_finds_its_imaging_protocol(tmp_path):
    """A photo in a session folder outside the project is found by the imaging
    protocol of its folder, given by its locations or by a path with ``..``."""
    start = datetime(2026, 1, 1)
    folder = tmp_path / "raw" / "run" / "first"
    folder.mkdir(parents=True)
    (folder / NAME).write_text("first")
    taken = start + timedelta(hours=2)
    (tmp_path / "imaging.csv").write_text(f"path,datetime\n{NAME},{taken}\n")
    (tmp_path / "injection.csv").write_text(
        "id,location_x,location_y,start,end,rate_kg/s\n"
        f"1,0.0,0.0,{start},{start + timedelta(hours=1)},0.0\n"
    )
    (tmp_path / "pressure.csv").write_text(
        "datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,"
        f"temperature_gradient_celsius\n{start},1.013,20.0,0.0,0.0\n"
    )
    experiment = ProtocolledExperiment(
        data=[folder / NAME],
        imaging_protocol={folder: tmp_path / "imaging.csv"},
        injection_protocol=tmp_path / "injection.csv",
        pressure_temperature_protocol=tmp_path / "pressure.csv",
    )
    project = tmp_path / "project"
    sessions = {"run": {"first": "../raw/run/first"}}
    photo = Locations(project, "results", sessions).photo("run", Frame("first", NAME))
    assert experiment.get_datetime(photo) == pd.Timestamp(taken)
    outside = project / ".." / "raw" / "run" / "first" / NAME
    assert experiment.get_datetime(outside) == pd.Timestamp(taken)

"""Unit tests for the locations of a project's results and photos."""

from darsia.utils import results
from darsia.utils.locations import Frame, Locations

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

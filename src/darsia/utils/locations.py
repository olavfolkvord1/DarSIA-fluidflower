"""Where a project's results and photos are, relative to the project folder.

The results folder and the folders of the photos are given relative to the project
folder, so that every location follows the project when it moves; an absolute
folder stays where it is. A run's photos come from one or more sessions, each with
its own folder, e.g. the photo folders of its days, and one photo name can occur in
two sessions of a run. A frame is therefore known by its session and its name, and
both its photo and its results are found by both.

"""

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple


class Frame(NamedTuple):
    """One photo of a run, by the session it was taken in and its file name."""

    session: str
    """The session: one photo folder of the run."""
    name: str
    """The photo's file name in the session's folder."""


class Locations:
    """The locations of a project's results and photos.

    Parameters
    ----------
    project : Path
        The project folder, where it is now.
    results : Path
        The results folder, relative to the project (the Setup's ``[data]
        results``).
    sessions : Mapping[str, Mapping[str, Path]]
        Each run's session folders, by run and session, relative to the project.

    """

    def __init__(
        self,
        project: Path,
        results: Path,
        sessions: Mapping[str, Mapping[str, Path]],
    ) -> None:
        self.project = Path(project)
        """The project folder."""
        self.results = self.project / results
        """The results folder."""
        self.sessions = {
            run: {session: self.project / folder for session, folder in folders.items()}
            for run, folders in sessions.items()
        }
        """Each run's session folders, by run and session."""

    def result(
        self,
        kind: str,
        *,
        run: str | None = None,
        frame: Frame | None = None,
        part: str | None = None,
    ) -> Path:
        """The folder of one result (:func:`darsia.utils.results.publish`).

        It is ``<results>/<kind>[/<run>][/<session>/<name> | /<part>]``: a result of
        one frame is stored by the frame's session and name, a result in parts by
        its part, of a run or of none, a result of a run by its run, and a single
        result, e.g. a series' phase model, by its kind alone.

        Parameters
        ----------
        kind : str
            The kind of result, the name of the step that makes it.
        run : str, optional
            The run of the result.
        frame : Frame, optional
            The frame of the result, of the run.
        part : str, optional
            The part of the result, e.g. a fold or a comparison time.

        Returns
        -------
        Path
            The result's folder.

        Raises
        ------
        ValueError
            If a frame is given without its run, or with a part.

        """
        if frame is not None and (run is None or part is not None):
            raise ValueError("A frame's result needs its run and has no part.")
        location = self.results / kind
        if run is not None:
            location = location / run
        if frame is not None:
            location = location / frame.session / frame.name
        if part is not None:
            location = location / part
        return location

    def photo(self, run: str, frame: Frame) -> Path:
        """The file of a frame's photo, in its session's folder of its run.

        Parameters
        ----------
        run : str
            The run of the frame.
        frame : Frame
            The frame.

        Returns
        -------
        Path
            The photo's file.

        Raises
        ------
        KeyError
            If the run has no such session.

        """
        try:
            folder = self.sessions[run][frame.session]
        except KeyError:
            raise KeyError(f"Run {run!r} has no session {frame.session!r}.") from None
        return folder / frame.name

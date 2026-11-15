"""Films written as MP4, frame by frame, and decoded once as a check.

A film is written as its frames come, so no film is held in memory. The written
file is decoded once: its reported frame count and rate and every frame's size
must be those written (the frames' content is not compared).

"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np


class MovieWriter:
    """An MP4 film of frames of one size, written as they come."""

    def __init__(self, path: Path, size, fps: float, codec: str, lock=None):
        """Start the film at ``path``: ``size`` (width, height) pixels, ``fps``
        frames per second, written with ``codec`` (a four-character code).

        ``lock``, a context manager, is held while the encoder opens, and only
        then: encoders opened at the same time can crash a process on Windows.

        """
        self.path, self.size, self.fps = Path(path), tuple(size), float(fps)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        with lock or nullcontext():
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
            opened = self.writer.isOpened()
        if not opened:
            raise RuntimeError(f"No {codec} video writer for {self.path.name}")
        self.count = 0

    def write(self, frame: np.ndarray) -> None:
        """Add one BGR frame of the film's size."""
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("A film frame has another size than its film")
        self.writer.write(np.ascontiguousarray(frame))
        self.count += 1

    def close(self) -> dict:
        """Finish the film and decode it once.

        Returns
        -------
        dict
            The decoded frame count, the rate and the frame size.

        Raises
        ------
        ValueError
            Where the film decodes to another frame count, rate or frame size than
            written.

        """
        self.writer.release()
        video = cv2.VideoCapture(str(self.path))
        try:
            reported = video.get(cv2.CAP_PROP_FRAME_COUNT), video.get(cv2.CAP_PROP_FPS)
            decoded = 0
            while True:
                ok, image = video.read()
                if not ok:
                    break
                if image.shape[:2] != (self.size[1], self.size[0]):
                    raise ValueError(f"{self.path.name} decodes to another frame size")
                decoded += 1
        finally:
            video.release()
        if reported != (self.count, self.fps) or decoded != self.count:
            raise ValueError(
                f"{self.path.name}: {self.count} frames at {self.fps} per second "
                f"written, {decoded} decoded, {reported[0]:g} at {reported[1]:g} "
                "reported"
            )
        return dict(decoded_frames=decoded, fps=self.fps, size=list(self.size))

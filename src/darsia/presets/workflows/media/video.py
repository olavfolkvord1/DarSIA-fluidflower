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

try:
    from imageio_ffmpeg import write_frames
except ImportError:  # Optional dependency for H.264 encoding.
    write_frames = None


class MovieWriter:
    """An MP4 film of frames of one size, written as they come."""

    def __init__(self, path: Path, size, fps: float, codec: str, lock=None, *, crf=17):
        """Start the film at ``path``: ``size`` (width, height) pixels, ``fps``
        frames per second, using libx264 with ``crf`` or an OpenCV four-character codec.

        ``lock``, a context manager, is held while the encoder opens, and only
        then: encoders opened at the same time can crash a process on Windows.

        """
        self.path, self.size, self.fps = Path(path), tuple(size), float(fps)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.codec = codec
        with lock or nullcontext():
            if codec == "libx264":
                self.writer = self._open_h264(crf)
            else:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.writer = cv2.VideoWriter(
                    str(self.path), fourcc, self.fps, self.size
                )
                if not self.writer.isOpened():
                    raise RuntimeError(f"No {codec} video writer for {self.path.name}")
        self.count = 0

    def _open_h264(self, crf):
        if type(crf) is not int or not 0 <= crf <= 51:
            raise ValueError("H.264 CRF must be an integer from 0 to 51")
        if write_frames is None:
            raise RuntimeError("Install DarSIA's media extra for libx264 video output.")
        encoder = write_frames(
            str(self.path),
            self.size,
            fps=self.fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            quality=None,
            macro_block_size=1,
            output_params=[
                "-crf",
                str(crf),
                "-preset",
                "fast",
                "-threads",
                "1",
                "-movflags",
                "+faststart",
            ],
        )
        encoder.send(None)
        return encoder

    def write(self, frame: np.ndarray) -> None:
        """Add one BGR frame of the film's size."""
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("A film frame has another size than its film")
        contiguous = np.ascontiguousarray(frame)
        if self.codec == "libx264":
            self.writer.send(contiguous)
        else:
            self.writer.write(contiguous)
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
        if self.codec == "libx264":
            self.writer.close()
        else:
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

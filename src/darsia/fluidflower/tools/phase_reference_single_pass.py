"""Evaluate off-timeline injection references in one causal pass.

The reference result is recorded while the reviewed sequence is in memory.
The publisher verifies its identity before making it visible in Studio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import hashlib
import json

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Endpoint:
    id: str
    elapsed_h: float
    photo: Path
    sha256: str


class SinglePassReferences:
    """Evaluate references before the first reviewed frame at/after their time.

    A reference does not advance canonical physical or predictor history.
    ``predict_reference`` must preserve the predictor's ordinary frame counter
    and its previous-feature cache. The caller supplies the *previous reviewed*
    image, not the image most recently evaluated by this observer.
    """

    def __init__(self, endpoints: list[Endpoint], *, run: str, destination: Path,
                 canvas_shape: tuple[int, int], content_box: tuple[int, int, int, int],
                 pipeline_sha256: str, model_sha256: str,
                 predict_reference: Callable[[np.ndarray, np.ndarray | None, str], Mapping[str, np.ndarray]],
                 output_mode: str = "review"):
        if output_mode not in ("review", "production"):
            raise ValueError("Reference output mode must be review or production")
        self.output_mode = output_mode
        self.remaining = sorted(endpoints, key=lambda item: (item.elapsed_h, item.id))
        self.run = run
        self.destination = Path(destination)
        self.canvas_shape = canvas_shape
        self.content_box = content_box
        self.pipeline_sha256 = pipeline_sha256
        self.model_sha256 = model_sha256
        self.predict_reference = predict_reference
        self.completed: list[dict] = []

    def through(self, next_reviewed_time: float, *, previous: np.ndarray | None,
                previous_index: int | None, physics) -> None:
        from darsia.fluidflower.phase_validation import _write_phase_maps, _write_diagnostic_evidence
        from darsia.fluidflower.tools.concentration_movie_compute import fork_sequence, history_digest

        x, y, width, height = self.content_box
        while self.remaining and self.remaining[0].elapsed_h <= next_reviewed_time:
            endpoint = self.remaining.pop(0)
            if sha256(endpoint.photo) != endpoint.sha256:
                raise ValueError(f"Reference photograph changed: {endpoint.photo}")
            with Image.open(endpoint.photo) as image:
                if image.size != (self.canvas_shape[1], self.canvas_shape[0]):
                    raise ValueError("Reference photograph has another native geometry")
                current = np.asarray(image.convert("RGB"))[y:y+height, x:x+width].copy()
            before = None if physics is None else history_digest(physics)
            raw = self.predict_reference(current, previous, endpoint.id)
            disposable = None if physics is None else fork_sequence(physics)
            if disposable is None:
                final, report = raw, None
            elif hasattr(disposable, "apply_with_reference"):
                final, report = disposable.apply_with_reference(raw, endpoint.elapsed_h, current)
            else:
                final, report = disposable.apply(raw, endpoint.elapsed_h)
            if physics is not None and history_digest(physics) != before:
                raise ValueError("Reference modified canonical physical history")
            folder = self.destination / "single_pass_references" / self.run / endpoint.id
            if self.output_mode == "review":
                _write_phase_maps(folder / "raw", self.run, self.model_sha256, 0,
                                  self.canvas_shape, self.content_box, raw)
            _write_phase_maps(folder / "final", self.run, self.pipeline_sha256, 0,
                              self.canvas_shape, self.content_box, final)
            if disposable is not None and hasattr(disposable, "write_diagnostics"):
                if self.output_mode == "review":
                    disposable.write_diagnostics(folder / "diagnostics", self.run, 0,
                                                 self.canvas_shape, self.content_box)
                else:
                    _write_diagnostic_evidence(folder / "diagnostics", self.run, 0, disposable)
            record = dict(id=endpoint.id, elapsed_h=endpoint.elapsed_h,
                          method_sha256=sha256(Path(__file__)),
                          photo={"path": str(endpoint.photo), "sha256": endpoint.sha256},
                          previous_reviewed_frame_index=previous_index,
                          canonical_history_unchanged=True,
                          canonical_history_sha256=before,
                          physical_report=report,
                          output_mode=self.output_mode,
                          raw_root=str(folder / "raw") if self.output_mode == "review" else None,
                          final_root=str(folder / "final"),
                          diagnostics_root=str(folder / "diagnostics"))
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "single_pass.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8")
            self.completed.append(record)

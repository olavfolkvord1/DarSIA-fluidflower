"""Read-only series defaults for painting display, placement and analysis boxes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path, PureWindowsPath
import re

FORMAT = "darsia-series-display-frame-models-v1"
SCOPE = "display-placement-boxes"
MAX_INDEX_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class FrameModelSelection:
    index_path: str
    source: str
    project_root: str
    project_id: str
    series_root: str
    series_id: str
    sha256: str = ""
    size_bytes: int = 0


def capture_frame_model_selection(project, *, override_path=""):
    """Capture only in-memory metadata; USER choice is an explicit override.

    A series default remains a series choice. It is never written back as a
    user override, and it grants no adoption or scientific geometry authority.
    """
    manifest = project.manifest
    tools = manifest.get("tools") or {}
    link = tools.get("native_series") or {}
    if not isinstance(link, Mapping):
        raise ValueError("The series link is unavailable. Reopen the series and retry.")
    context = dict(project_root=str(project.root),
        project_id=str(getattr(project, "project_id", manifest.get("project_id", ""))),
        series_root=str(link.get("series_root") or ""),
        series_id=str(link.get("series_id") or manifest.get("series_id") or ""))
    if override_path:
        path = str(override_path)
        if not Path(path).is_absolute():
            raise ValueError("The frame-model override needs an absolute index path. Use Frame models to change it.")
        return FrameModelSelection(path, "user override", **context)
    binding = link.get("display_frame_models")
    if binding is None:
        raise ValueError("The series has no measured frame models. Use Frame models to set an override.")
    if (not isinstance(binding, Mapping) or binding.get("format") != FORMAT
            or binding.get("scope") != SCOPE or binding.get("adopted") is not False):
        raise ValueError("The series frame-model binding is invalid. Retry after its display binding is corrected.")
    descriptor = binding.get("geometry_index")
    if not isinstance(descriptor, Mapping):
        raise ValueError("The series frame-model index binding is missing. Retry after its display binding is corrected.")
    relative, digest, size = (descriptor.get(key) for key in ("path", "sha256", "size_bytes"))
    if (not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute()
            or PureWindowsPath(relative).drive or relative.startswith(("/", "\\"))
            or ".." in PureWindowsPath(relative).parts
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int or not 0 < size <= MAX_INDEX_BYTES
            or not all(context.values())):
        raise ValueError("The series frame-model path, SHA, size or identity is invalid. Retry after its display binding is corrected.")
    return FrameModelSelection(str(Path(project.root) / relative), "series display binding",
        sha256=digest, size_bytes=size, **context)


def verify_frame_model_selection(selection):
    """Metadata-worker check of the immutable captured series descriptor."""
    if selection is None:
        return
    if not isinstance(selection, FrameModelSelection):
        raise ValueError("The captured frame-model selection is invalid. Reload the series and retry.")
    if not selection.sha256:
        return
    path = Path(selection.index_path)
    before = path.stat()
    with path.open("rb") as stream:
        raw = stream.read(MAX_INDEX_BYTES + 1)
    after = path.stat()
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (version(before) != version(after) or len(raw) != selection.size_bytes
            or hashlib.sha256(raw).hexdigest() != selection.sha256):
        raise ValueError("The series frame-model index changed. Retry after its display binding is corrected.")

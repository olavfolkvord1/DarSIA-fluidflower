"""RAW-derived lossless photographic review roots (N106/N108, package D2b).

A review root holds, for one run, PNG display assets of the reference and of
each frame. They are rendered only from the run's bound records:

- the camera RAW, through the shared camera-U renderer, for the ``original``
  photograph;
- the already computed native log-linear camera RGB, through the configured
  display helpers, for the photographic modes.

No JPEG is read or written, nothing scientific is recomputed, and an existing
root is never changed: every file is created exclusively and the manifest is
written last. Layout::

    <root>/native_frames/<run>/manifest.json
    <root>/native_frames/<run>/frames/<case_id>/<mode>.png
    <root>/native_frames/<run>/reference/<mode>.png

All assets lie on the run's phase canvas (``phase_shape_hw``). The native
content sits at ``content_box_xywh``, given by the native record's
``native_origin_xy``. Asset descriptors are relative to the run folder, so a
root can be moved as a whole.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .phase_saved_assets import VerifiedAssetReader, asset

RAW_REVIEW_FORMAT = "darsia-raw-photographic-review-v1"
RAW_SOURCE_CONTRACT = "darsia-raw-rig-source-v1"
ORIGINAL_MODE = "original"
PHOTO_SUPPORT_ASSET = "photo_support"

_READER = VerifiedAssetReader()


def _descriptor(value, name):
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
            or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64):
        raise ValueError(f"{name} needs an exact path and SHA-256")
    return {key: value[key] for key in ("path", "sha256", "size_bytes") if key in value}


def raw_sources(full: Mapping, audit_descriptor: Mapping, *, reader=None) -> dict:
    """Validate one RAW-mode native record and its producer audit.

    ``full`` is a native record's ``full`` section (native centre, origin and
    phase canvas). The audit must come from the RAW rig source and declare the
    camera RAW, its profile and the native-to-U transform. Returns the parts
    a display writer needs; nothing is decoded here.
    """
    reader = reader or _READER
    audit = reader.json(_descriptor(audit_descriptor, "RAW producer audit"))
    renderer = audit.get("renderer") or {}
    camera = (audit.get("sources") or {}).get("camera_arw") or {}
    identity = audit.get("source_identity") or {}
    centre = full.get("native_centre") or {}
    if (audit.get("contract") != RAW_SOURCE_CONTRACT or audit.get("jpeg_inputs") is not False
            or identity.get("kind") != "raw-acquisition"
            or identity.get("raw_sha256") != camera.get("sha256")
            or renderer.get("source_kind") != "raw"):
        raise ValueError("Native record is not a RAW-acquisition source")
    if full.get("complete") is not True or full.get("source_sha256") != camera.get("sha256"):
        raise ValueError("Native record and its RAW audit belong to different acquisitions")
    shape_hw = tuple(map(int, centre.get("native_shape_hw") or ()))
    phase_hw = tuple(map(int, full.get("phase_shape_hw") or ()))
    origin = tuple(map(int, full.get("native_origin_xy") or ()))
    homography = np.asarray(renderer.get("native_to_camera_u_h"), dtype=np.float64)
    if (len(shape_hw) != 2 or len(phase_hw) != 2 or len(origin) != 2
            or tuple(map(int, renderer.get("native_shape_hw") or ())) != shape_hw
            or homography.shape != (3, 3) or not np.isfinite(homography).all()
            or abs(np.linalg.det(homography)) < 1e-12):
        raise ValueError("RAW native record has incomplete native geometry")
    content = canvas_content_box(origin, phase_hw, shape_hw)
    return dict(
        acquisition=_descriptor(camera, "Camera RAW"),
        profile_id=str(renderer.get("profile_id") or renderer.get("profile") or ""),
        native_to_camera_u_h=homography,
        camera_u_wh=tuple(map(int, renderer.get("camera_u_wh") or ())),
        native_shape_hw=shape_hw,
        phase_shape_hw=phase_hw,
        native_origin_xy=origin,
        content_box_xywh=content,
        log_linear_rgb=_descriptor(centre.get("log_linear_rgb"), "Native log-linear RGB"),
        valid_mask=_descriptor(centre.get("valid_mask"), "Native valid mask"),
        audit=_descriptor(audit_descriptor, "RAW producer audit"),
    )


def canvas_content_box(native_origin_xy, phase_shape_hw, native_shape_hw) -> tuple[int, int, int, int]:
    """Where the native grid sits on the phase canvas, as (x, y, width, height).

    ``native_origin_xy`` is the canvas origin in native coordinates, so the
    native grid starts at its negative. The box must lie inside the canvas.
    """
    x, y = -int(native_origin_xy[0]), -int(native_origin_xy[1])
    height, width = map(int, native_shape_hw)
    canvas_height, canvas_width = map(int, phase_shape_hw)
    if x < 0 or y < 0 or x + width > canvas_width or y + height > canvas_height:
        raise ValueError("Native grid does not lie inside its phase canvas")
    return x, y, width, height


def render_original(sources: Mapping, *, profile: Mapping, fill_rgb8, reader=None) -> np.ndarray:
    """The camera RAW on the phase canvas, as HWC uint8 RGB.

    The shared renderer puts the RAW on the camera grid U. A pull-warp through
    the saved native-to-U transform then places it on the canvas, as the
    camera original was placed before. ``profile`` is the bound native
    configuration descriptor that holds the camera profiles.
    """
    import cv2

    from .native import raw_photograph

    reader = reader or _READER
    acquisition = sources["acquisition"]
    camera_u = raw_photograph.load_camera_u_rgb8(
        reader.path(acquisition), profile_path=profile["path"], profile_sha256=profile["sha256"],
        profile_id=sources["profile_id"], expected_sha256=acquisition["sha256"])
    if sources["camera_u_wh"] and (camera_u.shape[1], camera_u.shape[0]) != tuple(sources["camera_u_wh"]):
        raise ValueError("Camera-U render differs from the size its audit declares")
    x, y, _width, _height = sources["content_box_xywh"]
    canvas_to_native = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]])
    canvas_to_u = sources["native_to_camera_u_h"] @ canvas_to_native
    canvas_height, canvas_width = sources["phase_shape_hw"]
    return cv2.warpPerspective(
        np.ascontiguousarray(camera_u), canvas_to_u, (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
        borderValue=_rgb_triplet(fill_rgb8))


def _rgb_triplet(value) -> tuple[int, int, int]:
    """A configured fill colour as three ints; a single grey level means all three."""
    values = np.atleast_1d(np.asarray(value, dtype=np.int64)).tolist()
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError("A fill colour is one grey level or three RGB values")
    return tuple(int(v) for v in values)


def render_display_assets(sources: Mapping, photo_support: np.ndarray, *, parameters: Mapping,
                          modes: Sequence[str], reader=None) -> dict[str, np.ndarray]:
    """The configured photographic modes on the phase canvas, as HWC uint8 RGB.

    The native log-linear camera RGB is an already computed product; it is
    only passed through ``native.display_modes.render_display_modes`` with the
    configuration's ``display_modes`` section. ``photo_support`` is the
    native-shape boolean support that the highlight treatment protects.
    Outside the native grid the canvas has the configuration's unfilled colour.
    """
    from .native.display_modes import render_display_modes

    reader = reader or _READER
    log_rgb = np.asarray(reader.array(sources["log_linear_rgb"]), dtype=np.float32)
    if log_rgb.shape != (3, *sources["native_shape_hw"]):
        raise ValueError("Native log-linear RGB differs from its declared native grid")
    support = np.asarray(photo_support)
    if support.dtype != np.bool_ or support.shape != tuple(sources["native_shape_hw"]):
        raise ValueError("Photo support must be a native-shape boolean mask")
    rendered, _metadata = render_display_modes(
        log_rgb, support, tile_rows=int(parameters["tile_rows"]), parameters=parameters,
        selected_modes=tuple(modes))
    return {name: place_on_canvas(np.rint(np.float32(255) * image).astype(np.uint8), sources,
                                  fill_rgb8=parameters["unfilled_rgb8"])
            for name, image in rendered.items()}


def place_on_canvas(native_image: np.ndarray, sources: Mapping, *, fill_rgb8) -> np.ndarray:
    """Put a native-grid image (HWC RGB or HW mask) on the phase canvas."""
    x, y, width, height = sources["content_box_xywh"]
    canvas_height, canvas_width = sources["phase_shape_hw"]
    image = np.asarray(native_image)
    if image.shape[:2] != (height, width):
        raise ValueError("Image differs from its native grid")
    canvas = np.empty((canvas_height, canvas_width, *image.shape[2:]), dtype=image.dtype)
    canvas[...] = fill_rgb8
    canvas[y:y + height, x:x + width] = image
    return canvas


def _write_png(path: Path, image: np.ndarray) -> None:
    """Create one lossless PNG exclusively; an existing file is never replaced."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())


def _relative_asset(run_folder: Path, path: Path) -> dict:
    described = asset(path)
    described["path"] = PurePosixPath(path.relative_to(run_folder)).as_posix()
    return described


def _render_entry(run_folder: Path, folder: Path, sources: Mapping, photo_support, *,
                  profile, parameters, modes, reader) -> dict:
    assets = {}
    fill = parameters["unfilled_rgb8"]
    for mode in modes:
        if mode == ORIGINAL_MODE:
            image = render_original(sources, profile=profile, fill_rgb8=fill, reader=reader)
            _write_png(folder / f"{mode}.png", image)
            assets[mode] = _relative_asset(run_folder, folder / f"{mode}.png")
    display = [mode for mode in modes if mode != ORIGINAL_MODE]
    if display:
        for mode, image in render_display_assets(sources, photo_support, parameters=parameters,
                                                 modes=display, reader=reader).items():
            _write_png(folder / f"{mode}.png", image)
            assets[mode] = _relative_asset(run_folder, folder / f"{mode}.png")
    support = place_on_canvas(np.where(np.asarray(photo_support), np.uint8(255), np.uint8(0)),
                              sources, fill_rgb8=0)
    _write_png(folder / f"{PHOTO_SUPPORT_ASSET}.png", support)
    return dict(assets=assets,
                photo_support=_relative_asset(run_folder, folder / f"{PHOTO_SUPPORT_ASSET}.png"))


def write_raw_review_root(review_root: str | Path, *, run: str, native_configuration: Mapping,
                          raw_binding: Mapping, reference: Mapping, frames: Sequence[Mapping],
                          modes: Sequence[str] | None = None, reader=None) -> Path:
    """Write one fresh RAW photographic review root for a run; return its manifest.

    ``reference`` holds the bound ``acquisition_manifest``, its
    ``record_index`` and a native-shape boolean ``photo_support``. Each frame
    holds ``case_id``, ``frame_index``, ``elapsed_h``, the bound
    ``native_record``, ``native_geometry`` and ``rig_geometry`` descriptors
    and its ``photo_support``. ``modes`` defaults to the original plus every
    configured display mode. The run folder must not exist yet.
    """
    reader = reader or _READER
    run = str(run).casefold()
    configuration = _descriptor(native_configuration, "Native configuration")
    binding = _descriptor(raw_binding, "RAW source binding")
    parameters = reader.json(configuration)["display_modes"]
    modes = tuple(modes) if modes is not None else (ORIGINAL_MODE, *parameters["modes"])
    unknown = [mode for mode in modes if mode != ORIGINAL_MODE and mode not in parameters["modes"]]
    if unknown or not modes:
        raise ValueError(f"Unknown RAW review modes: {unknown}")
    run_folder = Path(review_root) / "native_frames" / run
    run_folder.mkdir(parents=True, exist_ok=False)  # One fresh root; never mutate one.

    acquisition_manifest = _descriptor(reference["acquisition_manifest"], "Reference acquisition manifest")
    record_index = int(reference["record_index"])
    record = reader.json(acquisition_manifest)["records"][record_index]
    native_row = reader.json(_descriptor(record["native_row"], "Reference native row"))
    reference_sources = raw_sources(record["full"], native_row["raw_producer_audit"], reader=reader)
    if str(record["source"].get("run") or "").casefold() != run:
        raise ValueError("Reference acquisition belongs to another run")
    reference_entry = dict(
        acquisition_manifest=acquisition_manifest, record_index=record_index,
        acquisition=reference_sources["acquisition"], audit=reference_sources["audit"],
        profile_id=reference_sources["profile_id"],
        **_render_entry(run_folder, run_folder / "reference", reference_sources,
                        reference["photo_support"], profile=configuration,
                        parameters=parameters, modes=modes, reader=reader))
    geometry = {key: reference_sources[key] for key in
                ("phase_shape_hw", "native_origin_xy", "native_shape_hw", "content_box_xywh")}

    entries = []
    for index, frame in enumerate(frames):
        record_descriptor = _descriptor(frame["native_record"], "Native record")
        native = reader.json(record_descriptor)
        if native.get("complete") is not True or native.get("processor_kind") != "raw_rig":
            raise ValueError("Frame native record is not a complete RAW rig record")
        sources = raw_sources(native["full"], native["audit"], reader=reader)
        audit = reader.json(sources["audit"])
        declared = (audit.get("processor") or {}).get("source_binding") or {}
        if declared.get("sha256") != binding["sha256"]:
            raise ValueError("Frame was produced under another RAW source binding")
        if any(sources[key] != geometry[key] for key in geometry):
            raise ValueError("Frames of one review root must share one phase canvas")
        if int(frame["frame_index"]) != index:
            raise ValueError("Frame indices must count 0, 1, 2, ... in timeline order")
        case_id = str(frame["case_id"])
        if PurePosixPath(case_id).name != case_id or case_id in {"", ".", ".."}:
            raise ValueError("Case id must be a plain folder name")
        entries.append(dict(
            case_id=case_id, frame_index=index, elapsed_h=float(frame["elapsed_h"]),
            native_record=record_descriptor,
            native_geometry=_descriptor(frame["native_geometry"], "Native geometry"),
            rig_geometry=_descriptor(frame["rig_geometry"], "Rig geometry"),
            acquisition=sources["acquisition"], audit=sources["audit"], profile_id=sources["profile_id"],
            **_render_entry(run_folder, run_folder / "frames" / case_id, sources,
                            frame["photo_support"], profile=configuration,
                            parameters=parameters, modes=modes, reader=reader)))

    canvas_height, canvas_width = geometry["phase_shape_hw"]
    manifest = dict(
        format=RAW_REVIEW_FORMAT, run=run, source_kind="raw",
        native_configuration=configuration, raw_binding=binding,
        modes=list(modes), phase_shape_hw=list(geometry["phase_shape_hw"]),
        native_origin_xy=list(geometry["native_origin_xy"]),
        native_shape_hw=list(geometry["native_shape_hw"]),
        content_box_xywh=list(geometry["content_box_xywh"]), canvas_wh=[canvas_width, canvas_height],
        reference=reference_entry, frames=entries, frame_count=len(entries))
    path = run_folder / "manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=1)
    return path


_MANIFESTS: dict = {}


def read_raw_review_manifest(review_root: str | Path, run: str) -> dict | None:
    """The run's RAW review manifest, or None when the root holds another format.

    A missing manifest or one of another format returns None, so a caller can
    try its historical branches. A RAW manifest with a broken structure raises.
    An unchanged file (same size and modification time) is parsed once.
    """
    path = Path(review_root) / "native_frames" / str(run).casefold() / "manifest.json"
    try:
        stamp = path.stat()
    except FileNotFoundError:
        return None
    key = (str(path.resolve()).casefold(), stamp.st_size, stamp.st_mtime_ns)
    if key in _MANIFESTS:
        return dict(_MANIFESTS[key]) if _MANIFESTS[key] is not None else None
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(_MANIFESTS) > 64:
        _MANIFESTS.pop(next(iter(_MANIFESTS)))
    if not isinstance(manifest, Mapping) or manifest.get("format") != RAW_REVIEW_FORMAT:
        _MANIFESTS[key] = None
        return None
    manifest = validate_raw_review_manifest(manifest, run)
    _MANIFESTS[key] = manifest
    return dict(manifest)


def validate_raw_review_manifest(manifest: Mapping, run: str) -> dict:
    """Check a parsed RAW review manifest of ``run`` and return a copy.

    Shared by the path reader and by consumers that read a manifest through
    its own SHA-bound descriptor. Any other format or a broken structure raises.
    """
    frames = manifest.get("frames") if isinstance(manifest, Mapping) else None
    if (not isinstance(manifest, Mapping) or manifest.get("format") != RAW_REVIEW_FORMAT
            or manifest.get("source_kind") != "raw"
            or str(manifest.get("run") or "") != str(run).casefold()
            or not isinstance(frames, list) or len(frames) != manifest.get("frame_count")
            or any(not isinstance(frame, Mapping) or frame.get("frame_index") != index
                   for index, frame in enumerate(frames))
            or not isinstance(manifest.get("reference"), Mapping)):
        raise ValueError("RAW review manifest is incomplete or out of order")
    for name in ("native_configuration", "raw_binding"):
        _descriptor(manifest.get(name), name)
    return dict(manifest)


def raw_review_record(manifest: Mapping, frame_index: int | None = None, *, reference: bool = False) -> Mapping:
    """The reference entry or one frame entry of a RAW review manifest."""
    if reference:
        return manifest["reference"]
    frames = manifest["frames"]
    if type(frame_index) is not int or not 0 <= frame_index < len(frames):
        raise ValueError("RAW review frame is outside the run's timeline")
    return frames[frame_index]


def raw_modes_available(manifest: Mapping, modes: Sequence[str], *, reference: bool = False) -> bool:
    """Whether every frame (or the reference) has each of ``modes``; no file I/O."""
    rows = [manifest.get("reference") or {}] if reference else list(manifest.get("frames") or [])
    return bool(rows) and all(isinstance(row.get("assets"), Mapping)
                              and all(mode in row["assets"] for mode in modes) for row in rows)


def raw_asset_descriptor(run_folder: str | Path, descriptor: Mapping) -> dict:
    """The absolute descriptor of an asset recorded relative to a RAW review run folder.

    SHA and size are kept. A path that would leave the run folder (absolute,
    with a drive or with ``..``) raises.
    """
    relative = PureWindowsPath(_descriptor(descriptor, "RAW review asset")["path"])
    if relative.drive or relative.root or ".." in relative.parts or not relative.parts:
        raise ValueError("RAW review asset escapes its run folder")
    return dict(descriptor, path=str(Path(run_folder).joinpath(*relative.parts)))


def _resolve(review_root, run, descriptor, reader) -> Path:
    run_folder = (Path(review_root) / "native_frames" / str(run).casefold()).resolve()
    return reader.path(raw_asset_descriptor(run_folder, descriptor))


def resolve_raw_photograph(review_root, run, frame_index: int | None = None, *, mode: str = "natural",
                           reference: bool = False, reader=None) -> Path:
    """The SHA-verified PNG of one mode of a frame or of the reference."""
    manifest = read_raw_review_manifest(review_root, run)
    if manifest is None:
        raise ValueError("The review root holds no RAW photographic review for this run")
    record = raw_review_record(manifest, frame_index, reference=reference)
    assets = record.get("assets") or {}
    if mode not in assets:
        raise ValueError(f"RAW review has no {mode!r} photograph")
    return _resolve(review_root, run, assets[mode], reader or _READER)


def load_raw_photo_support(review_root, run, frame_index: int | None = None, *, reference: bool = False,
                           canvas: bool = True, reader=None) -> np.ndarray:
    """The photo support mask of a frame or the reference, on the canvas or the native grid."""
    from PIL import Image

    manifest = read_raw_review_manifest(review_root, run)
    if manifest is None:
        raise ValueError("The review root holds no RAW photographic review for this run")
    record = raw_review_record(manifest, frame_index, reference=reference)
    with Image.open(_resolve(review_root, run, record["photo_support"], reader or _READER)) as image:
        mask = np.asarray(image) > 0
    if list(mask.shape) != list(manifest["phase_shape_hw"]):
        raise ValueError("Photo support belongs to another canvas")
    if canvas:
        return mask
    x, y, width, height = manifest["content_box_xywh"]
    return mask[y:y + height, x:x + width].copy()


__all__ = [
    "ORIGINAL_MODE", "PHOTO_SUPPORT_ASSET", "RAW_REVIEW_FORMAT", "RAW_SOURCE_CONTRACT",
    "canvas_content_box", "load_raw_photo_support", "place_on_canvas", "raw_asset_descriptor",
    "raw_modes_available", "raw_review_record", "raw_sources", "read_raw_review_manifest",
    "render_display_assets", "render_original", "resolve_raw_photograph",
    "validate_raw_review_manifest", "write_raw_review_root",
]

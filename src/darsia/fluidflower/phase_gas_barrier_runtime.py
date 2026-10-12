"""Source-bound Rule B on saved top-surface G3 labels, without re-inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .phase_photo_sand_top_runtime import PhotoSandTopSequence
from .project import sha256_file

VERSION = "ffac-gas-barrier-rule-b-v1"
PROFILE_KEY = "gas_barrier_rule_b"
SPEC_SHA256 = "41fa22e6bbf91b10086eaf9eb5a198039d4453c1687370338221ca010322a3f8"
OPERATION = "gas-to-aqueous-after-approved-g3-top-surface"


def bound_path(asset, *, root=None):
    if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
        raise ValueError("Barrier Rule B requires an exact artifact binding")
    path = Path(asset["path"])
    path = (Path(root) / path).resolve() if root is not None else path.resolve()
    if sha256_file(path) != asset["sha256"]:
        raise ValueError("Barrier Rule B bound artifact changed")
    return path


def read_parent_decision(labels_asset, rgba_asset, content_box):
    """Native proof is authoritative: top exclusion255 has transparent RGBA."""
    from .tools.train_ffac_gas_flatfield_variants import _read_frozen_map
    with Image.open(bound_path(labels_asset)) as image:
        labels = np.asarray(image).copy()
    x, y, width, height = map(int, content_box)
    if (labels.dtype != np.uint8 or labels.shape != (height, width)
            or not np.isin(labels, (0, 1, 2, 255)).all()):
        raise ValueError("Barrier Rule B parent proof is not a native categorical grid")
    rendered, rgba = _read_frozen_map(bound_path(rgba_asset), content_box)
    union = np.isin(labels, (1, 2))
    if (not np.array_equal(union, np.isin(rendered, (1, 2)))
            or not np.array_equal(labels[union], rendered[union])
            or np.any(rendered[labels == 0] != 0)
            or np.any(~np.isin(rendered[labels == 255], (0, 255)))):
        raise ValueError("Barrier Rule B parent native proof differs from its Studio map")
    return labels, rgba


def read_components(asset, shape):
    with np.load(bound_path(asset), allow_pickle=False) as archive:
        values = {name: archive[name] for name in ("open_components", "barrier_mask", "reference_domain")}
    components, barrier, domain = (values[key] for key in ("open_components", "barrier_mask", "reference_domain"))
    if (components.shape != tuple(shape) or not np.issubdtype(components.dtype, np.integer)
            or np.any(components < 0) or barrier.dtype != bool or domain.dtype != bool
            or barrier.shape != components.shape or domain.shape != components.shape
            or np.any((components > 0) & (barrier | ~domain))):
        raise ValueError("Barrier Rule B component bundle differs from its native domain")
    return values


def load_barrier_profile(root, profile):
    marker = profile[PROFILE_KEY]
    if (not profile.get("_photo_sand_top_runtime") or not isinstance(marker, dict)
            or set(marker) != {"format", "manifest", "spec_sha256", "operation", "barrier_derivation"}
            or marker["format"] != VERSION or marker["spec_sha256"] != SPEC_SHA256
            or marker["operation"] != OPERATION):
        raise ValueError("Barrier Rule B requires the explicit inherited G3 top-surface profile")
    candidate = Path(root).resolve()
    manifest_path = bound_path(marker["manifest"], root=candidate)
    derivation_path = bound_path(marker["barrier_derivation"], root=candidate)
    if not manifest_path.is_relative_to(candidate) or not derivation_path.is_relative_to(candidate):
        raise ValueError("Barrier Rule B profile artifacts escaped their candidate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (manifest.get("contract") != VERSION or manifest.get("spec_sha256") != SPEC_SHA256
            or manifest.get("barrier_derivation_sha256") != marker["barrier_derivation"]["sha256"]
            or not isinstance(manifest.get("frames"), list) or not manifest["frames"]):
        raise ValueError("Barrier Rule B profile has no exact frame/derivation binding")
    frames = {}
    for row in manifest["frames"]:
        run, index = str(row.get("run", "")).casefold(), row.get("frame_index")
        if (type(index) is not int or index < 0 or (run, index) in frames
                or "ffac/" + run not in profile["runs"]
                or not np.isfinite(float(row.get("elapsed_h", np.nan)))):
            raise ValueError("Barrier Rule B frame inventory differs from the inherited source")
        frames[run, index] = row
    return dict(profile, _gas_barrier_runtime=dict(root=str(candidate), frames=frames))


def load_barrier_frame(row, *, run, frame_index, elapsed_h, native_manifest_sha256):
    if (row["run"] != str(run).casefold() or row["frame_index"] != frame_index
            or float(row["elapsed_h"]) != float(elapsed_h)
            or row["native_manifest_sha256"] != native_manifest_sha256):
        raise ValueError("Barrier Rule B frame differs from requested source/time")
    native_path = bound_path(dict(path=row["native_manifest_path"], sha256=native_manifest_sha256))
    native = json.loads(native_path.read_text(encoding="utf-8-sig"))
    record = native["frame_sources"][frame_index]
    if (native["run"] != row["run"] or float(record["elapsed_h"]) != float(elapsed_h)
            or list(native["content_box_xywh"]) != list(row["content_box_xywh"])):
        raise ValueError("Barrier Rule B native metadata differs from its frame binding")
    labels, _rgba = read_parent_decision(row["parent_labels_map"], row["parent_phase_map"], row["content_box_xywh"])
    parent_receipt = json.loads(bound_path(row["parent_frame_receipt"]).read_text(encoding="utf-8-sig"))
    area = float(row["pixel_area_mm2"])
    if (parent_receipt.get("complete") is not True or not np.isfinite(area) or area <= 0
            or area != float(parent_receipt["qa"]["pixel_area_mm2"])):
        raise ValueError("Barrier Rule B nominal pixel scale differs from its sealed parent")
    arrays = read_components(row["components"], labels.shape)
    with np.load(bound_path(row["top_surface_mask"]), allow_pickle=False) as masks:
        keep, photo = masks["top_keep"], masks["photo_sand_revised"]
    photo_path = bound_path(row["revised_photo_mask"])
    if (keep.dtype != bool or photo.dtype != bool or keep.shape != labels.shape or photo.shape != labels.shape
            or not np.array_equal(np.load(photo_path, allow_pickle=False), photo)
            or Path(record["v5_photo_sand_mask_path"]).resolve() != photo_path
            or record["v5_photo_sand_mask_sha256"] != row["revised_photo_mask"]["sha256"]
            or np.any(labels[~keep] != 255)):
        raise ValueError("Barrier Rule B changed the inherited top-surface/photo domain")
    bound_path(row["registration_receipt"])
    return dict(parent_labels=labels, open_components=arrays["open_components"],
        barrier_mask=arrays["barrier_mask"], source_component_ids=row["source_component_ids"],
        top_keep=keep, photo_sand=photo, pixel_area_mm2=area,
        elapsed_h=float(elapsed_h), provenance=dict(components_sha256=row["components"]["sha256"],
        barrier_derivation_sha256=row["barrier_derivation_sha256"],native_manifest_sha256=native_manifest_sha256,
        parent_labels_map_sha256=row["parent_labels_map"]["sha256"],run=run,frame_index=frame_index))


def barrier_runtime_identity(inherited_identity, marker):
    from . import phase_gas_barrier_rule as core
    value = dict(inherited_identity=inherited_identity, marker=marker,
                 code_sha256=sha256_file(Path(__file__)), core_sha256=sha256_file(Path(core.__file__)),
                 numpy=np.__version__)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class GasBarrierSequence(PhotoSandTopSequence):
    """Read final parent maps and apply only the approved gas->aqueous rule."""
    def __init__(self, context, profile):
        super().__init__(context, profile)
        if not profile.get("_gas_barrier_runtime"):
            raise ValueError("Barrier Rule B runtime requires a validated frame inventory")
        self.barrier_runtime = profile["_gas_barrier_runtime"]

    def bind_frame(self, **payload):
        if self.frame_payload is not None:
            raise ValueError("Barrier Rule B frame binding was not consumed")
        if not np.isfinite(float(payload["elapsed_h"])):
            raise ValueError("Barrier Rule B requires an acquisition time")
        self.frame_payload = payload

    def bind_cached_frame(self, *, run, frame_index, elapsed_h, native_manifest_sha256):
        key = str(run).casefold(), frame_index
        if key not in self.barrier_runtime["frames"]:
            raise ValueError("Barrier Rule B has no exact binding for this frame")
        self.bind_frame(**load_barrier_frame(self.barrier_runtime["frames"][key],run=run,
            frame_index=frame_index,elapsed_h=elapsed_h,native_manifest_sha256=native_manifest_sha256))

    def apply_with_reference(self, predictions, elapsed_h, current):
        from .phase_gas_barrier_rule import apply_barrier_rule_b
        payload = self.frame_payload
        if (payload is None or float(elapsed_h) != payload["elapsed_h"]
                or np.asarray(current).shape[:2] != payload["parent_labels"].shape):
            raise ValueError("Barrier Rule B requires the exact saved parent frame/time")
        labels, audit = apply_barrier_rule_b(payload["parent_labels"],payload["open_components"],
            payload["source_component_ids"],pixel_area_mm2=payload["pixel_area_mm2"],barrier_mask=payload["barrier_mask"])
        self.last_labels = labels
        self.last_photo_sand = payload["photo_sand"].copy()
        self.last_top_keep = payload["top_keep"].copy()
        domain = np.asarray(self.context.domain)
        if domain.dtype != bool or domain.shape != labels.shape:
            raise ValueError("Barrier Rule B phase domain differs from its native context")
        self.last_phase_domain = domain & self.last_top_keep
        self.last_diagnostics = dict(approved_g3_gas=labels == 2,
                                     approved_g3_unresolved=(labels == 255) & self.last_photo_sand)
        self.frame_payload = None
        co2 = np.isin(labels, (1, 2)).astype(np.float32)
        gas = (labels == 2).astype(np.float32)
        result = dict(co2_presence_probability=co2,gas_probability_conditional_on_co2=gas,
            untouched_water_probability=1-co2,aqueous_co2_probability=co2*(1-gas),free_co2_gas_probability=co2*gas)
        return result, dict(gas_barrier_rule_b=audit,frame_binding=payload.get("provenance",{}),
            elapsed_h=float(elapsed_h),classification_scores_not_concentration=True,
            old_observation_retention_applied=False)

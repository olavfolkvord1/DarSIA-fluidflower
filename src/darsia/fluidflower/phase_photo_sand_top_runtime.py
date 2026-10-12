"""Explicit, source-bound top-surface clipping of already approved phase maps.

The inherited G3 decision is never inferred or filtered again. Only the
photographic top exclusion may change categorical labels; the old runtime
and its production-code identity remain unchanged without this opt-in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .phase_g3_production import ApprovedG3GasSequence, _mask_identity
from .project import sha256_file


VERSION = "ffac-photo-sand-top-revision-v1"
PROFILE_KEY = "photo_sand_top_revision"


def _read_bound(entry, label, *, root=None):
    if (not isinstance(entry, dict) or set(entry) != {"path", "sha256"}
            or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64):
        raise ValueError(f"Top revision {label} lacks an exact artifact binding")
    path = Path(entry["path"])
    path = (Path(root) / path).resolve() if root is not None else path.resolve()
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Top revision {label} changed")
    return path


def load_top_revision_profile(root, profile):
    """Validate the new revision without changing the inherited G3 profile."""
    marker = profile[PROFILE_KEY]
    if ("approved_g3_gas" not in profile or not profile.get("_approved_g3_runtime")
            or not isinstance(marker, dict) or set(marker) != {
                "format", "manifest", "operation", "outside_phase_label", "outside_phase_display"}
            or marker["format"] != VERSION
            or marker["operation"] != "clip-only-after-approved-g3"
            or marker["outside_phase_label"] != 255
            or marker["outside_phase_display"] != "transparent"):
        raise ValueError("Top revision requires the explicit approved clip-only profile")
    candidate = Path(root).resolve()
    manifest_path = _read_bound(marker["manifest"], "manifest", root=candidate)
    if not manifest_path.is_relative_to(candidate):
        raise ValueError("Top revision manifest escaped its candidate")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    frames = manifest.get("frames")
    if (manifest.get("contract", manifest.get("format")) != VERSION
            or not isinstance(frames, list) or not frames):
        raise ValueError("Top revision manifest has no source-bound frames")
    rows = {}
    for row in frames:
        run, index = str(row.get("run", "")).casefold(), row.get("frame_index")
        if (not run or not isinstance(index, int) or index < 0 or (run, index) in rows
                or f"ffac/{run}" not in profile["runs"]
                or not np.isfinite(float(row.get("elapsed_h", np.nan)))):
            raise ValueError("Top revision frame inventory is ambiguous or belongs to another cohort")
        rows[run, index] = row
    return dict(profile, _photo_sand_top_runtime={"root": str(candidate), "frames": rows,
                                                 "manifest_sha256": marker["manifest"]["sha256"]})


def load_top_revision_frame(row, *, root, run, frame_index, elapsed_h,
                             native_manifest_sha256):
    """Verify the source and native masks before reading the approved map."""
    if (str(row.get("run", "")).casefold() != str(run).casefold()
            or row.get("frame_index") != frame_index
            or float(row.get("elapsed_h", np.nan)) != float(elapsed_h)
            or row.get("shadow_native_manifest_sha256") != native_manifest_sha256):
        raise ValueError("Top revision binding differs from the shadow source/frame/time")
    original_path = _read_bound({"path": row["native_manifest_path"],
                                  "sha256": row["native_manifest_sha256"]}, "original native source")
    original = json.loads(original_path.read_text(encoding="utf-8-sig"))
    frames = original.get("frame_sources") or []
    if (str(original.get("run", "")).casefold() != str(run).casefold()
            or frame_index >= len(frames)
            or float(frames[frame_index].get("elapsed_h", np.nan)) != float(elapsed_h)
            or list(original.get("content_box_xywh") or []) != list(row["content_box_xywh"])):
        raise ValueError("Top revision original frame and native geometry differ")
    shadow_path = _read_bound({"path": row["shadow_native_manifest_path"],
                                "sha256": native_manifest_sha256}, "shadow native source", root=root)
    shadow = json.loads(shadow_path.read_text(encoding="utf-8-sig"))
    from .phase_v5_source_alias import resolve_original_native_source
    _original_root, alias_path, _alias_source = resolve_original_native_source(
        shadow_path.parent.parent.parent, run, shadow)
    if alias_path != original_path:
        raise ValueError("Top revision shadow source aliases another original source")
    mask_path = _read_bound(row["mask"], "native mask bundle", root=root)
    registration_path = _read_bound(row["registration_receipt"], "registration receipt", root=root)
    parent_path = _read_bound(row["parent_phase_map"], "approved parent phase map", root=root)
    from .tools.train_ffac_gas_flatfield_variants import _read_frozen_map
    labels, _rgba = _read_frozen_map(parent_path, row["content_box_xywh"])
    with np.load(mask_path, allow_pickle=False) as arrays:
        keep = arrays["top_keep"]
        old_photo = arrays["photo_sand_original"]
        revised_photo = arrays["photo_sand_revised"]
    old_row = frames[frame_index]
    old_photo_path = _read_bound({"path": old_row["v5_photo_sand_mask_path"],
                                   "sha256": old_row["v5_photo_sand_mask_sha256"]}, "original photo-sand")
    new_photo_path = _read_bound(row["revised_photo_mask"], "revised photo-sand", root=root)
    shadow_row = shadow["frame_sources"][frame_index]
    if (Path(shadow_row["v5_photo_sand_mask_path"]).resolve() != new_photo_path
            or shadow_row["v5_photo_sand_mask_sha256"] != row["revised_photo_mask"]["sha256"]
            or not np.array_equal(np.load(old_photo_path, allow_pickle=False), old_photo)
            or not np.array_equal(np.load(new_photo_path, allow_pickle=False), revised_photo)):
        raise ValueError("Top revision mask bundle differs from source-bound Studio photo masks")
    if tuple(labels.shape) != tuple(original["analysis_image_shape_hw"]):
        raise ValueError("Top revision map differs from the original native analysis grid")
    # This also verifies strict upper-prefix removal and the exact photo intersection.
    clip_approved_top_domain(labels, keep, old_photo, revised_photo)
    return dict(approved_labels=labels, top_keep=keep,
                photo_sand_original=old_photo, photo_sand_revised=revised_photo,
                elapsed_h=float(elapsed_h), provenance=dict(
                    original_native_manifest_sha256=row["native_manifest_sha256"],
                    shadow_native_manifest_sha256=native_manifest_sha256,
                    mask_sha256=row["mask"]["sha256"],
                    registration_receipt_sha256=sha256_file(registration_path),
                    parent_phase_map_sha256=row["parent_phase_map"]["sha256"],
                    frame_index=frame_index, run=str(run).casefold()))


def clip_approved_top_domain(approved_labels, top_keep, photo_sand_original,
                             photo_sand_revised):
    """Clip only a column's upper prefix; retained gas components stay exact."""
    labels = np.asarray(approved_labels)
    keep = np.asarray(top_keep)
    old_photo = np.asarray(photo_sand_original)
    photo = np.asarray(photo_sand_revised)
    if (labels.dtype != np.uint8 or labels.ndim != 2 or not labels.size
            or not np.isin(labels, (0, 1, 2, 255)).all()):
        raise ValueError("Top revision requires native categorical uint8 phase labels")
    if any(mask.dtype != bool or mask.shape != labels.shape
           for mask in (keep, old_photo, photo)):
        raise ValueError("Top revision masks must be native boolean grids")
    if (not keep[-1].all() or np.any(keep[:-1] & ~keep[1:])):
        raise ValueError("Top revision may remove only an upper prefix in each column")
    if not np.array_equal(photo, old_photo & keep):
        raise ValueError("Revised photo-sand differs from the declared top-only intersection")
    clipped = labels.copy()
    clipped[~keep] = 255
    if (not np.array_equal(clipped[keep], labels[keep])
            or not np.all(clipped[~keep] == 255)):
        raise AssertionError("Top clipping changed a retained phase decision")
    audit = dict(format=VERSION, operation="clip-only-after-approved-g3",
                 inside_top_phase_labels_exact=True,
                 inside_top_co2_union_exact=True, inside_top_gas_labels_exact=True,
                 outside_top_category=255, outside_top_display="transparent",
                 top_keep_mask_sha256=_mask_identity(keep),
                 photo_sand_original_mask_sha256=_mask_identity(old_photo),
                 photo_sand_revised_mask_sha256=_mask_identity(photo),
                 top_excluded_pixels=int(np.count_nonzero(~keep)),
                 removed_aqueous_pixels=int(np.count_nonzero((labels == 1) & ~keep)),
                 removed_gas_pixels=int(np.count_nonzero((labels == 2) & ~keep)),
                 removed_co2_pixels=int(np.count_nonzero(np.isin(labels, (1, 2)) & ~keep)),
                 revised_photo_sand_pixels=int(np.count_nonzero(photo)),
                 gas_rethresholded=False, gas_components_refiltered=False,
                 inference_rerun=False, labels_retrained=False, mass_eligible=False)
    return clipped, audit


def top_revision_runtime_identity(inherited_identity, marker):
    value = dict(inherited_identity=inherited_identity, revision=marker,
                 code_sha256=sha256_file(Path(__file__)), numpy=np.__version__)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class PhotoSandTopSequence(ApprovedG3GasSequence):
    """Read an approved parent decision and clip its top; never re-infer gas."""
    def __init__(self, context, profile):
        super().__init__(context, profile)
        runtime = profile.get("_photo_sand_top_runtime")
        if not runtime:
            raise ValueError("Top revision runtime requires a validated explicit profile")
        self.top_runtime = runtime
        self.last_photo_sand = None
        self.last_top_keep = None
        self.last_phase_domain = None

    def bind_frame(self, *, approved_labels, top_keep, photo_sand_original,
                   photo_sand_revised, elapsed_h, provenance=None):
        if self.frame_payload is not None:
            raise ValueError("A top revision frame binding was not consumed")
        if not np.isfinite(float(elapsed_h)):
            raise ValueError("Top revision requires an explicit acquisition time")
        self.frame_payload = dict(approved_labels=np.asarray(approved_labels),
                                  top_keep=np.asarray(top_keep),
                                  photo_sand_original=np.asarray(photo_sand_original),
                                  photo_sand_revised=np.asarray(photo_sand_revised),
                                  elapsed_h=float(elapsed_h), provenance=dict(provenance or {}))

    def bind_cached_frame(self, *, run, frame_index, elapsed_h, native_manifest_sha256):
        key = str(run).casefold(), frame_index
        if key not in self.top_runtime["frames"]:
            raise ValueError("Top revision has no binding for this source frame")
        payload = load_top_revision_frame(self.top_runtime["frames"][key],
            root=self.top_runtime["root"], run=run, frame_index=frame_index,
            elapsed_h=elapsed_h, native_manifest_sha256=native_manifest_sha256)
        self.bind_frame(**payload)

    def apply_with_reference(self, predictions, elapsed_h, current):
        if self.frame_payload is None:
            raise ValueError("Top revision requires a source-bound approved-map frame binding")
        payload = self.frame_payload
        if (float(elapsed_h) != payload["elapsed_h"]
                or np.asarray(current).shape[:2] != payload["approved_labels"].shape):
            raise ValueError("Top revision frame binding differs from requested image/time")
        labels, report = clip_approved_top_domain(payload["approved_labels"], payload["top_keep"],
            payload["photo_sand_original"], payload["photo_sand_revised"])
        old_domain = np.asarray(self.context.domain)
        if old_domain.dtype != bool or old_domain.shape != labels.shape:
            raise ValueError("Top revision phase domain differs from its native context")
        self.last_phase_domain = old_domain & payload["top_keep"]
        report.update(inherited_phase_domain_pixels=int(np.count_nonzero(old_domain)),
                      revised_phase_domain_pixels=int(np.count_nonzero(self.last_phase_domain)),
                      revised_phase_domain_mask_sha256=_mask_identity(self.last_phase_domain))
        self.frame_payload = None
        self.last_labels = labels
        self.last_photo_sand = payload["photo_sand_revised"].copy()
        self.last_top_keep = payload["top_keep"].copy()
        self.last_diagnostics = {"approved_g3_gas": labels == 2,
                                 "approved_g3_unresolved": (labels == 255) & self.last_photo_sand}
        co2 = np.isin(labels, (1, 2)).astype(np.float32)
        gas = (labels == 2).astype(np.float32)
        result = dict(co2_presence_probability=co2, gas_probability_conditional_on_co2=gas,
                      untouched_water_probability=1-co2, aqueous_co2_probability=co2*(1-gas),
                      free_co2_gas_probability=co2*gas)
        return result, dict(photo_sand_top_revision=report, elapsed_h=float(elapsed_h),
                            frame_binding=payload["provenance"],
                            classification_scores_not_concentration=True,
                            old_observation_retention_applied=False)

    def exact_phase_masks(self):
        masks = super().exact_phase_masks()
        # Categorical 255 outside the phase domain is excluded, not cyan evidence.
        masks["unresolved"] &= self.last_photo_sand
        return masks

    def exact_phase_domain(self):
        if self.last_phase_domain is None:
            raise ValueError("Top revision has no completed phase domain")
        return self.last_phase_domain.copy()


__all__ = ["VERSION", "PROFILE_KEY", "clip_approved_top_domain", "load_top_revision_profile",
           "load_top_revision_frame", "top_revision_runtime_identity", "PhotoSandTopSequence"]


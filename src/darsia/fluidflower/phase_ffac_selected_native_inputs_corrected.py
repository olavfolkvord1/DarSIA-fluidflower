"""Unbound corrective input adapter for review after the strict AC18 stop.

This module does not run inference or admit a campaign. It preserves the
historical masked native colour/support representation independently of the
unmasked scientific rig sidecars. The stopped campaign never imports it.
Every future use needs a new bound plan, snapshot, and the unchanged strict
all-Studio replay gate. No historical receipt or numerical asset is rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .phase_ffac_rig_camera_runtime import bound_path
from .phase_retained_source_inference import MASK_NAMES

CONTRACT = "ffac-selected-historical-native-input-correction-proposal-v1"


@dataclass(frozen=True)
class MaskedNativeInput:
    log: np.ndarray
    valid: np.ndarray
    provenance: dict


@dataclass(frozen=True)
class TemporalNativeInput:
    log: np.ndarray
    valid: np.ndarray
    origin_xy: tuple[int, int]
    provenance: dict


def _source_array(path, digest):
    return np.load(bound_path(dict(path=str(path), sha256=digest)),
                   allow_pickle=False, mmap_mode="r")


def _validate(log, valid):
    if (log.dtype != np.float32 or log.ndim != 3 or log.shape[0] != 3
            or valid.dtype != bool or valid.shape != log.shape[1:]
            or not np.array_equal(valid, np.isfinite(log).all(axis=0))):
        raise ValueError("Historical native input requires exact float32 finite/support agreement")


def original_native_input(row, *, scientific_native_log=None):
    """Read the original masked source instead of the rig's unmasked centre.

    The optional scientific input is checked only on original valid support;
    its additional pixels are never promoted to native model support.
    """
    required = ("v5_log_linear_rgb_path", "v5_log_linear_rgb_sha256",
                "v5_log_linear_valid_mask_path", "v5_log_linear_valid_mask_sha256")
    if any(not row.get(key) for key in required):
        raise ValueError("Original native source has no exact masked RAW descriptors")
    log = _source_array(row[required[0]], row[required[1]])
    valid = _source_array(row[required[2]], row[required[3]])
    _validate(log, valid)
    if scientific_native_log is not None:
        current = np.asarray(scientific_native_log)
        if (current.dtype != np.float32 or current.shape != log.shape
                or not np.array_equal(current[:, valid].view(np.uint32), log[:, valid].view(np.uint32))):
            raise ValueError("Rig RAW differs from original native source on historical valid pixels")
    return MaskedNativeInput(log, valid, dict(contract=CONTRACT,
        kind="exact_original_masked_native", source_sha256=row.get("source_sha256"),
        log_linear_rgb=dict(path=row[required[0]],sha256=row[required[1]]),
        valid_mask=dict(path=row[required[2]],sha256=row[required[3]]),
        support_promoted=False, scientific_sidecar_changed=False))


def new_native_input(log, valid, photo_sand, *, source_binding, photo_mask_binding):
    """Mask a new source transiently using its separately bound photo geometry.

    The source/frozen photometry and geometric photo-domain certificates must
    be supplied by the future driver. Predicted phase classes cannot provide
    this mask. This operation never changes the original full rig float array.
    """
    current, supplied = np.asarray(log), np.asarray(valid)
    _validate(current, supplied)
    photo = np.asarray(photo_sand)
    if (photo.dtype != bool or photo.shape != supplied.shape
            or not isinstance(source_binding, dict) or not source_binding.get("source_sha256")
            or not isinstance(photo_mask_binding, dict) or not photo_mask_binding.get("sha256")
            or photo_mask_binding.get("phase_labels_used") is not False):
        raise ValueError("New native input needs a source-bound geometric photo mask, independent of phase labels")
    supported = supplied & photo
    masked = current.copy()
    masked[:, ~supported] = np.nan
    masked.setflags(write=False); supported.setflags(write=False)
    return MaskedNativeInput(masked,supported,dict(contract=CONTRACT,
        kind="new_source_historical_masked_representation", source=source_binding,
        photo_mask=photo_mask_binding,support_promoted=False,
        scientific_sidecar_changed=False,model_or_photometry_fit=False))


def temporal_policy(reference_row):
    """A missing historical ring means zero padding, never a default of 64."""
    keys = ("v5_extended_ring_path", "v5_extended_ring_sha256",
            "v5_extended_pad_px", "v5_extended_origin_xy")
    present = [reference_row.get(key) is not None for key in keys]
    if not any(present):
        return dict(contract=CONTRACT,pad_px=0,origin_xy=[0,0],
                    source="historical reference has no declared temporal ring")
    if not all(present):
        raise ValueError("Historical temporal ring metadata is partial")
    pad=reference_row["v5_extended_pad_px"]
    origin=tuple(reference_row["v5_extended_origin_xy"])
    if type(pad) is not int or pad <= 0 or origin != (-pad,-pad):
        raise ValueError("Historical temporal ring has invalid native origin/padding")
    # The policy itself is source bound before any new-source extension.
    ring=bound_path(dict(path=reference_row[keys[0]],sha256=reference_row[keys[1]]))
    return dict(contract=CONTRACT,pad_px=pad,origin_xy=list(origin),
                source="exact bound historical extended-reference convention",
                ring=dict(path=str(ring),sha256=reference_row[keys[1]]))


def original_temporal_input(native, row):
    """Preserve each old acquisition's own explicitly declared ring or core."""
    if not isinstance(native,MaskedNativeInput):
        raise TypeError("Original temporal input requires a masked native source")
    policy=temporal_policy(row)
    if policy["pad_px"] == 0:
        return TemporalNativeInput(native.log,native.valid,(0,0),
            dict(contract=CONTRACT,kind="exact_original_native_core",policy=policy))
    from .tools.ffac_v5_extended_reference import assemble_extended_log
    log=assemble_extended_log(native.log,Path(policy["ring"]["path"]),
        ring_sha256=policy["ring"]["sha256"],expected_pad_px=policy["pad_px"])
    valid=np.isfinite(log).all(0)
    return TemporalNativeInput(log,valid,tuple(policy["origin_xy"]),
        dict(contract=CONTRACT,kind="exact_original_temporal_ring",policy=policy))


def new_temporal_input(native, full_rig_log, *, native_box_xywh, frozen_policy):
    """Use an explicitly inherited ring convention for a genuinely new image.

    When a bound ring is permitted, preserve the masked native centre inside
    the transient ring crop. Larger unmasked rig support cannot replace it.
    """
    if not isinstance(native,MaskedNativeInput) or frozen_policy.get("contract") != CONTRACT:
        raise ValueError("New temporal input requires an explicit historical input policy")
    pad=frozen_policy.get("pad_px")
    if type(pad) is not int or pad < 0 or tuple(frozen_policy.get("origin_xy",())) != (-pad,-pad):
        raise ValueError("New temporal padding must equal the bound reference convention")
    if pad == 0:
        return TemporalNativeInput(native.log,native.valid,(0,0),
            dict(contract=CONTRACT,kind="new_source_no_ring",policy=frozen_policy))
    if not frozen_policy.get("ring",{}).get("sha256"):
        raise ValueError("A new temporal ring needs the original bound ring-policy evidence")
    full=np.asarray(full_rig_log)
    x,y,w,h=map(int,native_box_xywh)
    if (full.dtype != np.float32 or full.ndim != 3 or full.shape[0] != 3
            or native.log.shape != (3,h,w) or min(x-pad,y-pad)<0
            or x+w+pad>full.shape[2] or y+h+pad>full.shape[1]):
        raise ValueError("Bound temporal ring escapes the exact full rig source")
    log=full[:,y-pad:y+h+pad,x-pad:x+w+pad].copy()
    log[:,pad:pad+h,pad:pad+w]=native.log
    valid=np.isfinite(log).all(0)
    log.setflags(write=False);valid.setflags(write=False)
    return TemporalNativeInput(log,valid,(-pad,-pad),
        dict(contract=CONTRACT,kind="new_source_own_ring_masked_native_centre",policy=frozen_policy,
             rig_scientific_input_changed=False,masked_core_preserved=True))


def original_photo_sand(probability_receipt):
    """Read the exact historical producer's photo mask for an anchor replay."""
    metadata=probability_receipt.get("metadata",{})
    if metadata.get("masks_order") != list(MASK_NAMES) or metadata.get("bitorder") != "little":
        raise ValueError("Original photo domain lacks its frozen packed-mask contract")
    shape=tuple(metadata["native_shape_hw"])
    descriptor=probability_receipt["files"]["extra_masks.packbits.npz"]
    with np.load(bound_path(descriptor),allow_pickle=False) as archive:
        rows=archive["masks"]
        if rows.shape != (len(MASK_NAMES),(int(np.prod(shape))+7)//8):
            raise ValueError("Original photo mask does not share the frozen native grid")
        mask=np.unpackbits(rows[0],bitorder="little",count=int(np.prod(shape))).reshape(shape).astype(bool)
    mask.setflags(write=False)
    return mask,dict(contract=CONTRACT,original_probability_masks=descriptor,
                     array_key="photo_sand_domain",phase_labels_used=False)


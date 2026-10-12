"""Reuse immutable P1-final photographs with final-2 domain-only receipts.

No RAW pixels are read or rendered. The JPEG is the exact parent's asset;
only presentation support masks and candidate-bound receipts are new.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np

from . import phase_rig_display_prerender as saved
from .phase_ffac_p1_final2_review import _bound, _read, _require, p1_final2_saved_review
from .project import sha256_file
from .review_storage import resolve_review_path
from .tools.prerender_ffac_p1_final_display import _json, _write

EQUIVALENCE_CONTRACT = "ffac-p1-final2-photograph-domain-equivalence-v1"
_DOMAIN_KEYS = {"classes", "rig_domain", "below_sand_floor", "domain_revision", "sand_floor_revision"}
_DOMAIN_ACCOUNTING_KEYS = {"domain_area_cm2", "area_cm2_by_class", "excluded_below_sand_floor_area_cm2", "domain_revision"}
_POSTPROCESS_KEYS = {"below_sand_floor_masks", "floor_connectivity_masks", "floor_contact_fill_v2",
                    "floor_contact_fill_v2_masks", "gas_envelope_masks", "previous_labels", "removed_display_layers",
                    "p1_final2_revision", "p1_final2", "domain_revision", "pooled_floor_curve"}


def _invariants(value):
    result = {key: deepcopy(item) for key, item in value.items() if key not in _POSTPROCESS_KEYS}
    rig = result.get("rig_domain_revision")
    if isinstance(rig, dict):
        result["rig_domain_revision"] = {key: item for key, item in rig.items() if key not in _DOMAIN_KEYS}
        parameters = result["rig_domain_revision"].get("parameters")
        if isinstance(parameters, dict):
            parameters = {key: item for key, item in parameters.items() if key not in _DOMAIN_ACCOUNTING_KEYS}
            if parameters:
                result["rig_domain_revision"]["parameters"] = parameters
            else:
                result["rig_domain_revision"].pop("parameters", None)
    return result


def require_photo_equivalence(parent, current):
    """Allow declared domain provenance only; never another RAW/H/time/area."""
    keys = ("renderer_contract", "renderer_code", "mode", "reference", "source_sha256",
            "content_box_xywh", "analysis_image_shape_hw")
    _require(all(parent.get(key) == current.get(key) for key in keys),
             "Final2 photograph changed its source, rendering transform or display grid")
    for key in ("source_record", "frame_binding", "reference_record"):
        before, after = parent.get(key), current.get(key)
        _require((before is None and after is None)
                 or (isinstance(before, dict) and isinstance(after, dict) and _invariants(before) == _invariants(after)),
                 "Final2 photograph changed scientific RAW, registration, time or pixel area")


def _decode_masks(asset):
    path = _bound(asset)
    with np.load(path, allow_pickle=False) as arrays:
        shape = tuple(int(v) for v in arrays["shape_hw"])
        return {name: np.unpackbits(arrays[name], bitorder="little")[:np.prod(shape)].reshape(shape).astype(bool)
                for name in ("support", "missing", "rig_domain")}


def rebind_record(review_root, run, frame_index, *, reference, parent_rows, output):
    from .phase_rig_raw_display import _native_frame_assets_cached, _verified_display_path
    from .phase_pixel_area import load_rig_classes
    key = "reference" if reference else f"frame_{frame_index:03d}"
    state, record, binding = _native_frame_assets_cached(review_root, run, frame_index, reference=reference)
    fields, classes_asset = load_rig_classes(record["rig_domain_revision"], base_dir=state["native_path"].parent)
    domain = fields["rig_domain"]
    raw_support = _decode_masks(parent_rows[key]["raw_detail"]["masks"])["support"]
    extended_domain = domain.copy()
    x, y, width, height = state["native"]["content_box_xywh"]
    extended_domain[y:y + height, x:x + width] = False
    result = {}
    for mode in ("change", "raw_detail"):
        parent_row = parent_rows[key][mode]
        old = saved._read_verified(parent_row["receipt"])
        _require(old.get("contract") == saved.CONTRACT and old.get("display") == parent_row["display"]
                 and old.get("masks") == parent_row["masks"]
                 and parent_row["binding_sha256"] == saved._token(old["source_binding"]),
                 "Parent photograph receipt differs from its exact display manifest")
        expected = saved.source_binding(state, record, binding, mode=mode, reference=reference)
        require_photo_equivalence(old["source_binding"], expected)
        receipt_path = Path(output) / run / key / (mode + ".receipt.json")
        if receipt_path.is_file():
            existing = _read(receipt_path)
            _require(existing.get("source_binding") == expected
                     and existing.get("parent_display_receipt") == parent_row["receipt"]
                     and existing.get("display") == parent_row["display"],
                     "Existing final2 photograph receipt binds another source")
            _verified_display_path(existing["display"])
            _verified_display_path(existing["masks"])
            result[mode] = dict(binding_sha256=saved._token(expected), display=existing["display"], masks=existing["masks"],
                                receipt=dict(path=str(receipt_path), sha256=sha256_file(receipt_path)))
            continue
        old_masks = _decode_masks(parent_row["masks"])
        _require(domain.dtype == bool and domain.shape == old_masks["rig_domain"].shape
                 and not np.any(domain & ~old_masks["rig_domain"]),
                 "Final2 photograph reuse requires only a tightened parent rig domain")
        _verified_display_path(parent_row["display"])
        masks = dict(rig_domain=domain, support=domain & old_masks["support"], missing=domain & old_masks["missing"])
        packed = BytesIO()
        np.savez_compressed(packed, shape_hw=np.array(domain.shape, np.int32),
                            **{name: np.packbits(value.reshape(-1), bitorder="little") for name, value in masks.items()})
        mask_asset = _write(receipt_path.with_name(mode + ".masks.npz"), packed.getvalue())
        audit = dict(old["live_audit"], rig_classes=classes_asset,
                     source_supported_rig_pixels=int((domain & raw_support).sum()),
                     mode_supported_rig_pixels=int(masks["support"].sum()), missing_rig_pixels=int(masks["missing"].sum()),
                     outside_native_rig_pixels=int(extended_domain.sum()),
                     outside_native_source_supported_pixels=int((extended_domain & raw_support).sum()),
                     outside_native_mode_supported_pixels=int((extended_domain & masks["support"]).sum()),
                     outside_native_missing_pixels=int((extended_domain & masks["missing"]).sum()),
                     parent_photograph_reused=True, domain_only_presentation_revision=True)
        receipt = dict(old, source_binding=expected, display=parent_row["display"], masks=mask_asset, live_audit=audit,
                       parent_display_receipt=parent_row["receipt"],
                       photograph_equivalence=dict(contract=EQUIVALENCE_CONTRACT, jpeg_unchanged=True,
                           raw_source_registration_time_pixel_area_unchanged=True,
                           domain_subset_of_parent=True, excluded_parent_domain_pixels=int((old_masks["rig_domain"] & ~domain).sum()),
                           parent_rendered_pixels_sha256=old.get("rendered_pixels_sha256"),
                           parent_render_statistics_inherited=True),
                       display_only=True, maps_changed=False, scientific_sidecars_changed=False,
                       labels_changed=False, sessions_changed=False)
        result[mode] = dict(binding_sha256=saved._token(expected), display=parent_row["display"], masks=mask_asset,
                            receipt=_json(receipt_path, receipt))
    return key, result


def import_parent_displays(candidate_root, *, output_root=None, workers=12, resume=True, publish=True):
    """Import all 907 reference/frame pairs and publish an external pointer."""
    from .phase_ffac_p1_final2_review import PLAN, SEAL
    root = resolve_review_path(candidate_root).resolve()
    checked = p1_final2_saved_review(root)
    _require(checked["placement"] == "P1-final2", "Photograph import needs fully admitted final2")
    plan = _read(root / PLAN)
    parent = resolve_review_path(plan["binding"]["parent"]).resolve()
    pointer = _read(saved.pointer_path(parent))
    _require(pointer.get("contract") == saved.CONTRACT and pointer.get("display_only") is True
             and resolve_review_path(pointer["candidate_root"]).resolve() == parent,
             "Final2 photograph reuse lacks the complete P1-final display pointer")
    parent_manifest = saved._read_verified(pointer["manifest"])
    _require(parent_manifest.get("candidate_id") == parent.name
             and parent_manifest.get("frame_and_reference_count") == 907 and len(parent_manifest.get("runs") or {}) == 43,
             "Final2 photograph reuse needs all parent references and frames")
    output = Path(output_root) if output_root is not None else root.parent / "p1_final2_display_derivatives"
    _require(resume or not output.exists(), "Refusing to overwrite an existing display-import batch")
    started = time.perf_counter()
    def run_import(run):
        parent_run = saved._read_verified(parent_manifest["runs"][run])
        records = {}
        for key in parent_run["records"]:
            reference = key == "reference"
            index = None if reference else int(key.removeprefix("frame_"))
            actual_key, row = rebind_record(root / "source_alias", run, index, reference=reference,
                                           parent_rows=parent_run["records"], output=output)
            records[actual_key] = row
        _require(len(records) == parent_run["frame_count"] + 1, "Parent photograph timeline is incomplete")
        return _json(output / run / "manifest.json", dict(contract=saved.CONTRACT, run=run,
                      frame_count=parent_run["frame_count"], records=records))
    manifests = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="final2-display-import") as pool:
        jobs = {pool.submit(run_import, run): run for run in sorted(parent_manifest["runs"])}
        for future in as_completed(jobs):
            manifests[jobs[future]] = future.result()
    manifest = dict(contract=saved.CONTRACT, candidate_id=root.name, candidate_root=str(root),
                    candidate_seal=dict(path=str(root / SEAL), sha256=sha256_file(root / SEAL)),
                    parent_display_manifest=pointer["manifest"], runs=manifests, frame_and_reference_count=907,
                    photograph_reuse_contract=EQUIVALENCE_CONTRACT, jpeg_assets_copied=0, raw_assets_read=0,
                    display_only=True, maps_changed=False, labels_changed=False, sessions_changed=False)
    asset = _json(output / "manifest.json", manifest)
    if publish:
        _json(saved.pointer_path(root), dict(contract=saved.CONTRACT, candidate_root=str(root), manifest=asset,
                                            display_only=True, default_candidate_changed=False))
    print(json.dumps(dict(status="complete", imported_pairs=907, workers=workers,
                          wall_seconds=time.perf_counter()-started, manifest=asset)), flush=True)
    return Path(asset["path"])


__all__ = ["import_parent_displays", "rebind_record", "require_photo_equivalence", "EQUIVALENCE_CONTRACT"]

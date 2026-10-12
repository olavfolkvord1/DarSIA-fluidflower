"""Reuse P1-final2's exact photographs and support masks for P1-final3.

Only new candidate-bound receipts are written. No RAW pixels are read, no
photograph is rendered, and no inherited display asset is rewritten.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import json
import time

from . import phase_rig_display_prerender as saved
from .phase_ffac_p1_final3_review import _bound, _read, _require, p1_final3_saved_review, PLAN, SEAL
from .project import sha256_file
from .review_storage import resolve_review_path
from .tools.prerender_ffac_p1_final_display import _json

EQUIVALENCE_CONTRACT = "ffac-p1-final3-photograph-label-only-equivalence-v1"
_POSTPROCESS_KEYS = {"below_sand_floor_masks", "floor_connectivity_masks", "floor_contact_fill_v2",
                    "floor_contact_fill_v2_masks", "gas_envelope_masks", "previous_labels", "removed_display_layers",
                    "fill_reason_masks", "contact_fill_b_reverted_masks", "reverted_previous_labels", "p1_final2_revision", "p1_final3_revision",
                    "p1_final2", "p1_final3"}


def _invariants(value):
    return {key: deepcopy(item) for key, item in value.items() if key not in _POSTPROCESS_KEYS}


def require_photo_equivalence(parent, current):
    """Require identical photographs, including domain, scale and source grid."""
    before, after = deepcopy(parent), deepcopy(current)
    for key in ("source_record", "frame_binding", "reference_record"):
        left, right = before.get(key), after.get(key)
        _require((left is None and right is None)
                 or (isinstance(left, dict) and isinstance(right, dict) and _invariants(left) == _invariants(right)),
                 "Final3 photograph changed RAW, registration, domain, time or pixel area")
        if isinstance(left, dict):
            before[key], after[key] = _invariants(left), _invariants(right)
    _require(saved._compatible_binding(after, before),
             "Final3 photograph changed its source, transform, display grid or unaudited renderer")


def _available(asset):
    """Keep the exact immutable descriptor; pixels are SHA-verified on first UI use."""
    _require(isinstance(asset, dict) and isinstance(asset.get("sha256"), str)
             and len(asset["sha256"]) == 64, "Parent photograph has no SHA-bound asset")
    path = resolve_review_path(asset["path"])
    _require(path.is_file() and path.stat().st_size > 0, "Parent photograph asset is absent")
    if "size_bytes" in asset:
        _require(path.stat().st_size == asset["size_bytes"], "Parent photograph asset changed its size")


def rebind_record(review_root, run, frame_index, *, reference, parent_rows, output):
    from .phase_rig_raw_display import _native_frame_assets_cached
    key = "reference" if reference else f"frame_{frame_index:03d}"
    state, record, binding = _native_frame_assets_cached(review_root, run, frame_index, reference=reference)
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
        _available(parent_row["display"])
        _available(parent_row["masks"])
        receipt_path = Path(output) / run / key / (mode + ".receipt.json")
        receipt = dict(old, source_binding=expected, display=parent_row["display"], masks=parent_row["masks"],
                       parent_display_receipt=parent_row["receipt"],
                       photograph_equivalence=dict(contract=EQUIVALENCE_CONTRACT, jpeg_unchanged=True,
                           masks_unchanged=True, raw_source_registration_domain_time_pixel_area_unchanged=True,
                           parent_rendered_pixels_sha256=old.get("rendered_pixels_sha256"),
                           parent_render_statistics_inherited=True),
                       live_audit=dict(old["live_audit"], parent_photograph_reused=True,
                                       label_only_presentation_revision=True),
                       display_only=True, maps_changed=False, scientific_sidecars_changed=False,
                       labels_changed=False, sessions_changed=False)
        asset = _json(receipt_path, receipt)
        result[mode] = dict(binding_sha256=saved._token(expected), display=parent_row["display"],
                            masks=parent_row["masks"], receipt=asset)
    return key, result


def import_parent_displays(candidate_root, *, output_root=None, workers=12, resume=True, publish=True):
    root = resolve_review_path(candidate_root).resolve()
    _require(p1_final3_saved_review(root)["placement"] == "P1-final3",
             "Photograph import needs fully admitted final3")
    parent = resolve_review_path(_read(root / PLAN)["binding"]["parent"]).resolve()
    pointer = _read(saved.pointer_path(parent))
    _require(pointer.get("contract") == saved.CONTRACT and pointer.get("display_only") is True
             and resolve_review_path(pointer["candidate_root"]).resolve() == parent,
             "Final3 photograph reuse lacks the complete P1-final2 display pointer")
    manifest = saved._read_verified(pointer["manifest"])
    _require(manifest.get("candidate_id") == parent.name and manifest.get("frame_and_reference_count") == 907
             and len(manifest.get("runs") or {}) == 43,
             "Final3 photograph reuse needs all parent references and frames")
    output = Path(output_root) if output_root is not None else root.parent / "p1_final3_display_derivatives"
    _require(resume or not output.exists(), "Refusing to overwrite an existing display-import batch")
    started = time.perf_counter()
    def run_import(run):
        parent_run = saved._read_verified(manifest["runs"][run])
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
    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="final3-display-import") as pool:
        jobs = {pool.submit(run_import, run): run for run in sorted(manifest["runs"])}
        for future in as_completed(jobs):
            manifests[jobs[future]] = future.result()
    value = dict(contract=saved.CONTRACT, candidate_id=root.name, candidate_root=str(root),
                 candidate_seal=dict(path=str(root / SEAL), sha256=sha256_file(root / SEAL)),
                 parent_display_manifest=pointer["manifest"], runs=manifests, frame_and_reference_count=907,
                 photograph_reuse_contract=EQUIVALENCE_CONTRACT, jpeg_assets_copied=0, raw_assets_read=0,
                 display_only=True, maps_changed=False, labels_changed=False, sessions_changed=False)
    asset = _json(output / "manifest.json", value)
    if publish:
        _json(saved.pointer_path(root), dict(contract=saved.CONTRACT, candidate_root=str(root), manifest=asset,
                                            display_only=True, default_candidate_changed=False))
    print(json.dumps(dict(status="complete", imported_pairs=907, workers=workers,
                          wall_seconds=time.perf_counter()-started, manifest=asset)), flush=True)
    return Path(asset["path"])


__all__ = ["import_parent_displays", "rebind_record", "require_photo_equivalence", "EQUIVALENCE_CONTRACT"]

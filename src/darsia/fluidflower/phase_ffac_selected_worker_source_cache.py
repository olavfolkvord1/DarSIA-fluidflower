"""Execution-only, exact-descriptor SHA reuse for one selected-image worker.

This module enumerates metadata, not scientific pixels. It does not admit a
source, change an inference operator, or cache a result. The caller supplies
already admitted, committed scientific receipts. The existing source-hash
context performs both first-use verification and an independent full-SHA exit
seal; a completion receipt is unavailable until that seal succeeds.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .phase_source_hash_cache import ImmutableSourceHashCache, immutable_source_hash_cache

CONTRACT = "ffac-selected-worker-bound-source-sha-cache-v1"
_SHA = re.compile(r"^[0-9a-fA-F]{64}$")

# These are input roles from the persisted selected/native schemas. Unknown
# fields never create cache eligibility, even when they resemble descriptors.
_ROLE_FIELDS = {
    "source_inventory": "metadata", "source_manifest": "metadata",
    "parent_native_manifest": "metadata", "reference_parent_receipt": "metadata",
    "parent_receipt": "metadata", "original_parent_receipt": "metadata",
    "original_prepared_receipt": "metadata", "original_producer_receipts": "metadata",
    "prepared_source": "metadata", "receipt": "metadata",
    "source_geometry_receipt": "geometry", "geometry_frame": "geometry",
    "rig_geometry": "geometry", "geometry": "geometry", "registered_context": "geometry",
    "context": "geometry", "area_map": "geometry", "area_parameters": "geometry",
    "pixel_area": "geometry", "pixel_area_mm2": "geometry", "pixel_area_m2": "geometry",
    "physical_scale_revision": "geometry", "frozen_top_curve": "geometry",
    "fixed_floor_curve": "geometry", "fixed_zones": "geometry",
    "frozen_barrier_reference": "geometry", "reference_components": "geometry",
    "barrier_components": "geometry", "open_components": "geometry",
    "barrier_mask": "geometry", "photo_sand_mask": "geometry",
    "frame_photo_top_keep": "geometry",
    "rig_domain": "geometry", "native_domain": "geometry", "facies": "geometry",
    "depth": "geometry", "porosity": "geometry", "physical_reference": "geometry",
    "native_scientific": "scientific", "full_extended_scientific": "scientific",
    "baseline_extended_scientific": "scientific", "native_centre": "scientific",
    "ring_windows": "scientific", "parent_cache": "scientific",
    "log_linear_rgb": "scientific", "valid_mask": "scientific",
    "reference": "reference", "baseline": "reference", "clear_water_reference": "reference",
    "source_photo": "source", "raw_source": "source", "source": "source",
    "camera_photo": "source", "model": "model", "models": "model", "heads": "model",
    "co2": "model", "g3": "model", "regional": "model", "t4b": "model",
    "pixel_model": "model", "regional_model": "model", "light_epochs": "model",
    "profile": "model", "postflight_manifest": "metadata",
    "parent_probability_receipt": "metadata", "parent_seal": "metadata",
    "final3_plan": "metadata", "final2_plan": "metadata", "final1_plan": "metadata",
    "t4c_plan": "metadata",
}
_CONTAINERS = {"binding", "sources", "frame_sources", "configured_baseline_members",
    "reference_frames", "independent_reference_frames", "selected_sources",
    "studio_anchor_sources", "parent_source", "retained_source_cache", "extension_audit",
    "correction_contract", "area", "classes", "assets", "epochs", "native_references"}
_DENIED_FIELDS = {"code", "implementation", "implementation_files", "module", "modules",
    "runtime", "code_binding", "code_assets", "labels", "labels_map", "label_map",
    "phase_labels", "production_labels", "maps", "phase_maps", "annotations", "sessions",
    "scorer", "scorers", "sealed_scorer", "training_labels", "validation_labels",
    "photographs", "display_renderings", "display_preparations", "review_outputs",
    "selected_phase_frames", "operator_masks", "extra_masks", "g3_pixel",
    "regional_frozen", "regional_refit", "co2_packbits", "frozen_gas"}
_DENIED_SEGMENTS = {"selected_phase_frames", "production_labels", "display_renderings",
    "display_preparations", "review_outputs", "annotations", "sessions", "scorer",
    "scorers", "labels", "training", "training_labels", "code", "code_snapshot", "src", "tests"}
_DENIED_SUFFIXES = {".py", ".pyc", ".pyo", ".exe", ".dll", ".zip", ".whl", ".mp4"}
_NATIVE_PREFIXES = ("v5_", "native_", "source_", "camera_", "raw_", "baseline_",
    "reference_", "geometry_", "context_", "rig_", "valid_", "log_", "facies_",
    "depth_", "porosity_", "physical_", "optical_")
# Expand the actual source preparation/geometry closure, not older candidate
# plans, seals, acquisition inventories, or model-training provenance. Those
# bound metadata files may be cached when read, but their unused dependencies
# do not belong to this worker's current/reference/predecessor closure.
_EXPAND_FIELDS = {"prepared_source", "receipt", "original_prepared_receipt",
    "source_geometry_receipt", "geometry_frame", "rig_geometry", "postflight_manifest"}
_MAX_JSON_DOCUMENTS = 128
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _token(value: str) -> str:
    return str(value).casefold().replace("-", "_")


def _forbidden_field(value: str) -> bool:
    token = _token(value)
    return (token in _DENIED_FIELDS or "scorer" in token or "annotation" in token
        or "label" in token or "training" in token or "implementation" in token
        or token.startswith("code_") or token.endswith(("_code", "_module")))


def _read_metadata_bytes(path: Path) -> bytes:
    # Hash and parse the same byte string. A separate hash pass followed by
    # read_text could inspect different metadata if the file changed between
    # reads. This function never opens scientific raster assets.
    return path.read_bytes()


@dataclass(frozen=True)
class WorkerSourceAllowlist:
    run: str
    assets: Mapping[Path, str]
    excluded_roots: tuple[Path, ...]
    enumeration: Mapping[str, Any]


@dataclass
class WorkerSourceCache:
    allowlist: WorkerSourceAllowlist
    cache: ImmutableSourceHashCache

    def receipt(self) -> dict:
        if not self.cache.sealed:
            raise ValueError("Worker source cache receipt requires the actual full-SHA exit seal")
        return dict(contract=CONTRACT, run=self.allowlist.run,
            enumeration=dict(self.allowlist.enumeration),
            source_verification_cache=self.cache.receipt(),
            scientific_arrays_read_for_enumeration=False,
            numerical_operators_changed=False, output_sha_cache_allowed=False,
            actual_full_sha_exit_seal_completed=True)


def build_worker_source_allowlist(plan: Mapping[str, Any], run: str, *,
        scientific_receipts: Sequence[Mapping[str, Any]],
        excluded_roots: Sequence[str | Path] = ()) -> WorkerSourceAllowlist:
    """Enumerate existing, bound inputs for an already admitted worker prefix.

    No files are discovered by directory scanning. A missing unused referenced
    asset is excluded; a missing required inventory/committed receipt fails.
    File pixels are never opened here, and only SHA-verified JSON is expanded.
    This is intentionally narrower than source admission: unknown fields retain
    normal full hashing rather than gaining cache eligibility.
    """
    run = str(run).casefold()
    state = plan.get("runs", {}).get(run)
    if not isinstance(state, Mapping):
        raise ValueError("Worker source cache requires an exact plan run")
    roots = [Path(p).expanduser().resolve() for p in excluded_roots]
    operation = plan.get("execution_operation", {})
    output_root = operation.get("phase_output_root") if isinstance(operation, Mapping) else None
    if output_root:
        roots.append(Path(output_root).expanduser().resolve())
    roots = tuple(dict.fromkeys(roots))
    assets: dict[Path, str] = {}
    keys: dict[str, Path] = {}
    documents: dict[tuple[Path, str], Any] = {}
    roles: dict[Path, set[str]] = {}
    skipped = {"missing": 0, "output_or_code": 0, "unbound_or_unknown": 0,
        "other_run_rows": 0, "bounded_json_expansion": 0}
    json_bytes = 0

    def descriptor_path(descriptor, *, required=False):
        if (not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str)
                or not isinstance(descriptor.get("sha256"), str) or not _SHA.fullmatch(descriptor["sha256"])):
            if required:
                raise ValueError("Required source metadata needs an exact path/SHA descriptor")
            skipped["unbound_or_unknown"] += 1
            return None
        path = Path(descriptor["path"]).expanduser()
        # Persisted descriptors are absolute. Relative names cannot be guessed
        # relative to a model, receipt, CWD, or a different host.
        if not path.is_absolute():
            if required:
                raise ValueError("Required source metadata path must be absolute")
            skipped["unbound_or_unknown"] += 1
            return None
        path = path.resolve()
        if (path.suffix.casefold() in _DENIED_SUFFIXES
                or any(_token(part) in _DENIED_SEGMENTS
                       or _token(part).startswith(("labels_", "label_", "scorer_", "sealed_labels", "sealed_scorer"))
                       or _token(part).endswith(("_labels", "_scorer")) for part in path.parts)
                or "label" in _token(path.stem) or "scorer" in _token(path.stem)
                or any(path.is_relative_to(root) for root in roots)):
            if required:
                raise ValueError("Required source metadata lies in an excluded output/code namespace")
            skipped["output_or_code"] += 1
            return None
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            skipped["missing"] += 1
            return None
        return path

    def add(descriptor, role, *, required=False, expand=False, follow=True):
        nonlocal json_bytes
        path = descriptor_path(descriptor, required=required)
        if path is None:
            return None
        digest = descriptor["sha256"].lower()
        key = os.path.normcase(str(path))
        prior = keys.get(key)
        if prior is not None and assets[prior] != digest:
            raise ValueError("Conflicting bound source SHA for " + str(path))
        keys[key] = path
        assets[path] = digest
        roles.setdefault(path, set()).add(role)
        if path.suffix.casefold() != ".json" or not expand:
            return None
        document_key = (path, digest)
        if document_key in documents:
            return documents[document_key]
        before = path.stat()
        if (len(documents) >= _MAX_JSON_DOCUMENTS
                or json_bytes + before.st_size > _MAX_JSON_BYTES):
            if required:
                raise ValueError("Required worker metadata exceeds the bounded JSON enumeration")
            skipped["bounded_json_expansion"] += 1
            return None
        content = _read_metadata_bytes(path)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Referenced source metadata SHA mismatch: " + str(path))
        payload = json.loads(content.decode("utf-8-sig"))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Referenced source metadata changed during enumeration: " + str(path))
        if not isinstance(payload, (dict, list)):
            raise ValueError("Source metadata must be a JSON object or array: " + str(path))
        documents[document_key] = payload
        json_bytes += before.st_size
        if follow:
            walk(payload, role)
        return payload

    def walk(value, role=None, *, field=None):
        if isinstance(value, list):
            for item in value:
                walk(item, role)
            return
        if not isinstance(value, Mapping):
            return
        if isinstance(value.get("run"), str) and value["run"].casefold() != run:
            skipped["other_run_rows"] += 1
            return
        if "path" in value and "sha256" in value:
            if role is not None:
                add(value, role, expand=field in _EXPAND_FIELDS)
            else:
                skipped["unbound_or_unknown"] += 1
            return
        for name, child in value.items():
            token = _token(name)
            if _forbidden_field(token):
                continue
            if token == "runs" and isinstance(child, Mapping):
                if run in child:
                    walk(child[run], role)
                continue
            # Exact explicit native alias pairs; never derive a filename or SHA.
            if token.endswith("_path"):
                prefix = token[:-5]
                digest = value.get(name[:-5] + "_sha256")
                if (prefix.startswith(_NATIVE_PREFIXES) and not _forbidden_field(prefix)
                        and isinstance(child, str) and isinstance(digest, str)):
                    add({"path": child, "sha256": digest}, "native_alias")
                continue
            child_role = _ROLE_FIELDS.get(token)
            if child_role is not None:
                walk(child, child_role, field=token)
            elif token in _CONTAINERS:
                walk(child, role)
            elif role == "model" and isinstance(child, Mapping):
                # Head/epoch names inside an explicitly frozen model group.
                walk(child, role)
            elif token in ("original_producer_receipts",):
                walk(child, "metadata")

    inventory = add(state.get("source_inventory"), "metadata", required=True,
        expand=True, follow=False)
    if not isinstance(inventory, Mapping) or inventory.get("run") != run:
        raise ValueError("Worker inventory belongs to another run")
    source_rows = {row["source_sha256"]: row for row in inventory.get("sources", [])}
    if len(source_rows) != len(inventory.get("sources", [])):
        raise ValueError("Worker inventory repeats a source acquisition")
    admitted, wrappers = [], []
    for descriptor in scientific_receipts:
        payload = add(descriptor, "metadata", required=True, expand=True, follow=False)
        if (not isinstance(payload, Mapping) or payload.get("complete") is not True
                or payload.get("contract") != "ffac-selected-frozen-scientific-source-v1"
                or payload.get("run") != run or payload.get("source_sha256") not in source_rows
                or payload.get("binding", {}).get("source") != source_rows[payload.get("source_sha256")]):
            raise ValueError("Cache source receipt is not the exact committed inventory acquisition")
        admitted.append(payload["source_sha256"])
        wrappers.append(payload)
    if len(admitted) != len(set(admitted)):
        raise ValueError("Worker cache input receipt repeats an acquisition")
    active_rows = [source_rows[digest] for digest in admitted]
    for row in active_rows:
        walk(row, "source")
    for wrapper in wrappers:
        walk(wrapper, "metadata")
    if "parent_native_manifest" in state:
        native = add(state["parent_native_manifest"], "metadata", expand=True, follow=False)
        if isinstance(native, Mapping):
            if native.get("run", run).casefold() != run:
                raise ValueError("Parent native manifest belongs to another run")
            walk(native.get("reference", {}), "reference")
            active_indices = {row.get("parent_source", {}).get("parent_source_frame_index")
                for row in active_rows if isinstance(row.get("parent_source"), Mapping)}
            for index, frame in enumerate(native.get("frame_sources", [])):
                if (frame.get("source_sha256") in admitted or index in active_indices):
                    walk(frame, "scientific")
    for key in ("reference_parent_receipt", "source_manifest"):
        if key in state:
            add(state[key], "metadata")
    for key in ("frozen_top_curve", "physical_scale_revision", "frozen_barrier_reference", "heads", "light_epochs"):
        if key in plan:
            walk(plan[key], _ROLE_FIELDS[key])
    walk(plan.get("parent_chain", {}), "metadata")
    conventions = plan.get("phase_campaign_binding", {}).get("inherited_stage_conventions", {}).get(run)
    if conventions is not None:
        walk(conventions, "geometry")
    serial = [{"path": str(path), "sha256": digest} for path, digest in
              sorted(assets.items(), key=lambda item: os.path.normcase(str(item[0])))]
    identity = hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    enumeration = dict(contract=CONTRACT, run=run, exact_asset_allowlist_sha256=identity,
        allowed_input_files=len(assets), allowed_input_size_bytes=sum(p.stat().st_size for p in assets),
        verified_json_documents=len(documents), verified_json_read_bytes=json_bytes,
        json_expansion_limit_documents=_MAX_JSON_DOCUMENTS,
        json_expansion_limit_bytes=_MAX_JSON_BYTES,
        source_closure="supplied committed current/reference/predecessor acquisitions only",
        admitted_source_sha256=admitted,
        input_roles={name: sum(name in kinds for kinds in roles.values())
                     for name in sorted({n for kinds in roles.values() for n in kinds})},
        excluded_assets=skipped, directory_discovery=False,
        scientific_arrays_read=False, unbound_paths_allowed=False,
        output_or_code_cache_allowed=False)
    return WorkerSourceAllowlist(run, assets, roots, enumeration)


@contextmanager
def worker_source_hash_cache(plan, run, *, scientific_receipts, excluded_roots=()):
    """Use the existing full-verification/exit-seal scope; no opt-out exists."""
    allowlist = build_worker_source_allowlist(plan, run,
        scientific_receipts=scientific_receipts, excluded_roots=excluded_roots)
    with immutable_source_hash_cache(allowlist.assets, excluded_roots=allowlist.excluded_roots) as cache:
        worker = WorkerSourceCache(allowlist, cache)
        yield worker


__all__ = ["CONTRACT", "WorkerSourceAllowlist", "WorkerSourceCache",
           "build_worker_source_allowlist", "worker_source_hash_cache"]

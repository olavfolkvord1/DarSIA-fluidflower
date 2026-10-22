"""Assemble existing phase-only DATA from exact original acquisition declarations.

Explicit original RAW-preparation reference/history specs and complete selected
sources are caller inputs. Replacement geometry/area/static/transform DATA comes
from its genuine producers. This assembler reads metadata only, writes nothing,
and neither adapts a grid nor grants scientific eligibility.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..mass.build_native_bridge import BuildNativeBridge
from ..mass.n1_pipeline import DataReader
from ..mass.native_optical_source import _bound, _same
from .build_raw_phase import FORMAT, _ROLES, _specification


@dataclass(frozen=True)
class BuildPhaseReplacement:
    """Explicit source-bound products; no inferred geometry or default transforms."""
    raw_acquisition: Mapping
    geometry: Mapping
    area: Mapping
    static: Mapping
    current_h: Sequence
    reference_to_current_h: Sequence


def assemble_build_phase_inputs(
    bridge: BuildNativeBridge, old_run_configuration: Mapping,
    reference_spec: Mapping, initial_previous_spec: Mapping,
    selected_sources: Sequence[Mapping], replacements: Sequence[BuildPhaseReplacement],
    *, phase_configuration: Mapping, temporal_pad_px: int, reader=None,
) -> dict:
    """Join literal old roles to supplied current products in the original format.

    The selected cohort must be the complete old cohort, in its existing order.
    Reference/previous specs are original RAW-preparation acquisition declarations;
    this function does not discover or synthesize them. Grid compatibility remains
    the original collector's responsibility. Historical tasks, mass inputs and
    positional caches never become current inputs.
    """
    if not isinstance(bridge, BuildNativeBridge):
        raise TypeError("The original completed-Build native bridge is required")
    reader = DataReader() if reader is None else reader
    pins = []

    def document(binding):
        bound = deepcopy(_bound(dict(binding)))
        pins.append(bound)
        return reader.json(bound)

    old = document(old_run_configuration)
    phase_binding = deepcopy(_bound(dict(phase_configuration)))
    before_phase = document(old["phase"]["configuration"])
    after_phase = document(phase_binding)
    native_binding = deepcopy(bridge.context.provenance["native_configuration"])
    before_native = document(old["native"]["configuration"])
    after_native = document(native_binding)
    if (old["run"] != bridge.context.run
            or not _same(before_phase["chain_assets"]["floor"], after_phase["chain_assets"]["floor"])
            or not _same(before_phase["parameters"]["geometry"]["floor_source_origin_xy"],
                         after_phase["parameters"]["geometry"]["floor_source_origin_xy"])):
        raise ValueError("The original run/floor source or its source origin changed")
    if not _same(before_native["support"], after_native["support"]):
        raise ValueError("Original photographic support configuration changed")
    frames = old["frames"]
    if (not frames or len(selected_sources) != len(frames)
            or any(not _same(source, spec["source"]) for source, spec in zip(selected_sources, frames))):
        raise ValueError("The complete original selected source cohort/order must remain exact")
    reference, previous = old["phase"]["reference"], old["phase"]["initial_previous"]
    if (reference_spec["source"]["source_sha256"] != reference["source_sha256"]
            or reference["source_sha256"] != reference["full"]["source_sha256"]
            or not _same(initial_previous_spec["source"], previous["source"])
            or previous["source"]["source_sha256"] != previous["full"]["source_sha256"]):
        raise ValueError("Original reference/history acquisition declarations differ")

    def identity(source):
        raw = source["source_binding"]["raw_acquisition"]
        if raw["sha256"] != source["source_sha256"]:
            raise ValueError("The declared RAW identity differs from its acquisition source SHA")
        return raw["path"], raw["sha256"]

    originals = [reference_spec, initial_previous_spec, *frames]
    identities = {identity(spec["source"]) for spec in originals}
    products = {}
    for item in replacements:
        if not isinstance(item, BuildPhaseReplacement):
            raise TypeError("Explicit typed Build phase replacements are required")
        raw = _bound(dict(item.raw_acquisition))
        key = raw["path"], raw["sha256"]
        if key in products:
            raise ValueError("Each original RAW acquisition needs exactly one replacement record")
        products[key] = item
    if set(products) != identities:
        raise ValueError("Replacement RAW path/SHA coverage differs from the original acquisitions")
    baseline = products[identity(reference_spec["source"])].raw_acquisition
    if not _same(dict(baseline), bridge.context.provenance["raw_acquisition"]):
        raise ValueError("The original reference differs from the completed Build baseline RAW")

    def assemble(spec, *, selected):
        item = products[identity(spec["source"])]
        roles = {key: deepcopy(spec["phase_roles"][key]) for key in _ROLES}
        roles.update(area=deepcopy(dict(item.area)), static=deepcopy(dict(item.static)),
                     current_h=deepcopy(item.current_h), reference_to_current_h=deepcopy(item.reference_to_current_h),
                     retained_source_cache=None, encoding_contract=None)
        result = {key: deepcopy(spec[key]) for key in ("case_id", "frame_index", "elapsed_h", "source")}
        result.update(raw_acquisition=deepcopy(dict(item.raw_acquisition)),
                      geometry=deepcopy(dict(item.geometry)), phase_roles=roles)
        for key in ("provenance", "context_acceptance"):
            if key in spec:
                result[key] = deepcopy(spec[key])
        if "raw_acquisition" in spec and any(
                spec["raw_acquisition"][key] != result["raw_acquisition"][key] for key in ("path", "sha256")):
            raise ValueError("Replacement RAW path/SHA differs from its original acquisition")
        _specification(result, old["run"], old["selection_index_field"], selected=selected)
        return result

    native = deepcopy(old["native"])
    native["configuration"] = native_binding
    result = dict(format=FORMAT, run=old["run"], native=native,
                  phase=dict(configuration=phase_binding), history_policy=deepcopy(old["history_policy"]),
                  selection_index_field=old["selection_index_field"], temporal_pad_px=temporal_pad_px,
                  reference_spec=assemble(reference_spec, selected=False),
                  initial_previous_spec=assemble(initial_previous_spec, selected=False),
                  frames=[assemble(spec, selected=True) for spec in frames])
    for key in ("data_aliases", "provenance"):
        if key in old:
            result[key] = deepcopy(old[key])
    for binding in pins:
        reader.path(binding)
    return result

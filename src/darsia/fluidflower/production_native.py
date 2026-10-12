"""Four locked native FF_AC diagnostics with call-local IO and SESSION.

This operator never initializes a campaign, fits, publishes or resumes R3.
The selected runner's load/summary code objects are used unchanged. Reference
modules must come from the sealed original runtime in a separate process;
candidate modules are the reviewed IO-wired committed worktree. Their source
provenance is explicit and is never represented as an old source hash.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import importlib
from pathlib import Path
import re
from types import FunctionType

from .production_io import normalize_locator

FORMAT = "darsia-production-native-four-case-v1"
SELECTION_SHA256 = "6350f5dada834147cefb1ec7c35cd2771a25dc4e8476f62ad9d38a2640167822"
FREEZE_SHA256 = "38f8d821eabf92d25ceb8a6ce76f69d7c2054fb888fbab9cac6bdb03e2f5f00c"
LOCKED_CASES = {
    "ac17_f6": ("ac17", 6, "b0b8c316562e2fa574ca2be9c26263868f4933a380a7663b7dcd9aae10c8c2e5"),
    "ac19_f34": ("ac19", 34, "f32f53af4d3cdf952b3cbfa72500a86528d41b9d934b7c5a939f6f2026a94291"),
    "ac52_f43": ("ac52", 43, "3a53f1a18d7ad69ac1badd05e96c54235520fdf96f15beb0c97d7abb1b43516a"),
    "ac19_f40": ("ac19", 40, "8b9b2862f1322f2b678154b7d7cc62bca2a7674a78c90288c2dbe5d46e234ff0"),
}


def _clone(function, **globals_overrides):
    """Retain the exact code object and give this call its own globals table."""
    result = FunctionType(function.__code__, dict(function.__globals__, **globals_overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    return result


class _Proxy:
    def __init__(self, original, **overrides):
        self.original, self.overrides = original, overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.original, name)


def _modules():
    base = "darsia.fluidflower."
    return dict(
        runner=importlib.import_module(base + "tools.run_ffac_common_mass_round3_selected"),
        adapter=importlib.import_module(base + "tools.ffac_round3_selected_adapter"),
        source=importlib.import_module(base + "phase_ffac_common_mass_inputs"),
        geometry=importlib.import_module(base + "phase_ffac_common_mass_round3_inputs"),
        engine=importlib.import_module(base + "phase_ffac_common_mass_round3"))


def _execution_identity(reader):
    from .production_job import code_identity
    return code_identity(reader.worktree)


def _configuration(config):
    if (not isinstance(config, Mapping) or config.get("format") != FORMAT
            or config.get("backend") != "cpu"
            or config.get("production_permission_granted") is not False
            or config.get("execution_mode") not in {"original_reference", "candidate_io"}
            or config.get("purpose") not in {"file_handling_acceptance", "profile"}):
        raise ValueError("Only explicit CPU native diagnostics are admitted")
    for key in ("campaign", "profile_selection", "actual_R3_freeze"):
        _descriptor(config[key])
    if (config["profile_selection"]["sha256"] != SELECTION_SHA256
            or config["actual_R3_freeze"]["sha256"] != FREEZE_SHA256):
        raise ValueError("The original fixed selection and actual B1 freeze are required")
    return config


def _descriptor(value):
    if (not isinstance(value, Mapping) or set(value) not in
            ({"path", "sha256"}, {"path", "sha256", "size_bytes"})
            or not isinstance(value["path"], (str, Path)) or not str(value["path"])
            or not isinstance(value["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or ("size_bytes" in value and
                (type(value["size_bytes"]) is not int or value["size_bytes"] < 0))):
        raise ValueError("An exact logical input descriptor is required")
    return dict(value)


def _cases(selection):
    rows = selection.get("cases", ())
    if len(rows) != 4 or {row.get("case_id") for row in rows} != set(LOCKED_CASES):
        raise ValueError("Exactly the four predeclared native cases are required")
    anchors, result = {}, {}
    for row in rows:
        entry, anchor = row["entry"], row["anchor_entry"]
        identity = (entry["run"], entry["frame_index"], entry["source_binding"]["source_sha256"])
        if (identity != LOCKED_CASES[row["case_id"]] or type(entry["frame_index"]) is not int
                or anchor["run"] != entry["run"] or float(anchor["selected_target_h"]) != 2.25):
            raise ValueError("A selected acquisition or native anchor differs from the locked selection")
        if anchor["run"] in anchors and anchors[anchor["run"]] != anchor:
            raise ValueError("A run has conflicting frozen dilution anchors")
        anchors[anchor["run"]] = anchor
        result[row["case_id"]] = row
    if len(anchors) != 3:
        raise ValueError("The locked cases require exactly three original run anchors")
    return result


def _candidate_reader(config, reader):
    if config["execution_mode"] == "original_reference":
        if callable(getattr(reader, "resolve_path", None)) or getattr(reader, "asset_reader", None) is not None:
            raise ValueError("Reference requires the original reader and sealed original modules")
        return None
    if (not callable(getattr(reader, "source_path", None))
            or not re.fullmatch(r"[0-9a-f]{40}", getattr(reader, "code_commit", ""))
            or getattr(reader, "execution_identity", {}).get("clean_tree") is not True
            or not getattr(reader, "source_bindings", None)
            or getattr(reader, "source_qualification_state", None) not in
                {"exact_CPU_parity_bootstrap", "exact_CPU_parity_verified"}):
        raise ValueError("Candidate requires the committed manifest reader and explicit source qualification")
    if (config["purpose"] == "profile"
            and reader.source_qualification_state != "exact_CPU_parity_verified"):
        raise ValueError("Profiling requires actual exact CPU parity first")
    if _execution_identity(reader) != reader.execution_identity:
        raise ValueError("Candidate execution tree changed since manifest reader construction")
    return reader


def _source_gate(campaign, plan, reader, asset_reader, source, modules):
    """Verify original evidence AND honest current execution provenance."""
    current = {}
    for name, bound in campaign["code_binding"].items():
        module = next((module for module in modules.values() if module.__name__ == name), None)
        module = module or importlib.import_module(name)
        reader.path(bound)
        if asset_reader is None:
            if source.sha256(module.__file__) != bound["sha256"]:
                raise ValueError("Reference is not the sealed original campaign implementation: " + name)
        else:
            asset_reader.source_path(module.__file__, bound["sha256"])
        current[name] = source.asset(module.__file__)
    adapter = modules["adapter"]
    bound = plan["selected_adapter"]["source"]
    reader.path(bound)
    if plan["selected_adapter"]["function"] != "load_selected":
        raise ValueError("The selected native loader differs")
    if asset_reader is None:
        if source.sha256(adapter.__file__) != bound["sha256"]:
            raise ValueError("Reference selected adapter is not the sealed original source")
    else:
        asset_reader.source_path(adapter.__file__, bound["sha256"])
    current[adapter.__name__] = source.asset(adapter.__file__)
    return current


@dataclass
class NativeContext:
    configuration: Mapping
    modules: Mapping
    reader: object
    asset_reader: object
    session: dict
    cases: Mapping
    provenance: dict
    load_entry: object
    summary_from_loaded: object
    captured: dict
    boundary_guard: object = None

    def roots(self):
        """Explicit live roots for BoundaryObserver; no numerical reads here."""
        return dict(session=self.session, last_native_returns=self.captured)


@dataclass
class NativeResult:
    case_id: str
    loaded: dict
    row: dict
    geometry: object
    own_colour: dict
    static: dict
    provenance: dict

    def scientific_products(self):
        """Reference existing numerical buffers; never deep-copy/recalculate."""
        def record(value, excluded=()):
            if not is_dataclass(value):
                raise TypeError("Native structured products must retain their dataclass schema")
            return {field.name: getattr(value, field.name) for field in fields(value)
                    if field.name not in excluded}
        response = self.loaded["response"]
        # Analysis/flash objects remain available in loaded. Operational source
        # receipts are intentionally outside numerical product comparison.
        native = {key: response[key] for key in
                  ("native_signal", "quantifiable", "linear_relative", "joint_signal", "swi")}
        native.update(frame=record(response["frame"], ("provenance",)),
                      prepared=record(response["prepared"]))
        summary = {key: value for key, value in self.row.items() if key not in
                   {"source_binding", "geometry_source_binding", "manifest_receipt", "run_manifest"}}
        def reduction_product(value):
            # Only the original histogram/pairs reduction clock is operational.
            # All weights, codes, backend facts and audit flags remain products;
            # the complete original audit, including time, remains in self.row.
            if "reduction_audit" not in value:
                return value
            return {**value, "reduction_audit": {
                key: item for key, item in value["reduction_audit"].items()
                if key != "wall_seconds"}}
        for key in ("main_aqueous_histogram", "paired_estimates"):
            if key in summary:
                summary[key] = reduction_product(summary[key])
        if "excluded_parts" in summary:
            parts = {}
            for name, original in summary["excluded_parts"].items():
                part = dict(original)
                if "own_colour_histogram" in part:
                    part["own_colour_histogram"] = reduction_product(part["own_colour_histogram"])
                if "by_facies" in part:
                    part["by_facies"] = {label: {
                        **values, "own_colour_histogram": reduction_product(values["own_colour_histogram"])}
                        if "own_colour_histogram" in values else values
                        for label, values in part["by_facies"].items()}
                parts[name] = part
            summary["excluded_parts"] = parts
        return dict(native=native, geometry=record(self.geometry, ("source_binding",)),
                    own_colour=self.own_colour, photo_support=self.loaded["photo_support"],
                    summary=summary,
                    static=self.static)


def prepare_native_context(configuration, reader, *, modules=None, boundary_guard=None):
    """Actual native setup, called only by an explicitly authorized job.

    No permission is inferred here: the production job/live controller owns
    admission and resource checks. ``boundary_guard`` is called before native
    setup/load/summary; exceptions propagate and are never retried.
    """
    config = _configuration(configuration)
    asset_reader = _candidate_reader(config, reader)
    modules = modules or _modules()
    runner, source, geometry, engine = (modules[key] for key in ("runner", "source", "geometry", "engine"))
    if boundary_guard is not None:
        boundary_guard()
    campaign = reader.json(config["campaign"])
    if (campaign.get("contract") != runner.CAMPAIGN_CONTRACT
            or campaign.get("source_equivalence_exact") is not True
            or campaign.get("production_automatically_allowed") is not False):
        raise ValueError("The original source-equivalent selected campaign is required")
    # Original full plan/39-run/PASS/Studio/index checks; no compact admission
    # surrogate and no scientific decoder are used by this validation.
    validated = runner.validate_selected_plan(campaign["plan"], reader)
    plan, parts = validated["plan"], validated["parts"]
    provenance = _source_gate(campaign, plan, reader, asset_reader, source, modules)
    cases = _cases(reader.json(config["profile_selection"]))
    originals = {(row["run"], row["frame_index"]): row for row in validated["entries"]}
    for case in cases.values():
        for entry in (case["entry"], case["anchor_entry"]):
            if originals.get((entry["run"], entry["frame_index"])) != entry:
                raise ValueError("Selected case/anchor is not the exact original indexed entry")
    if boundary_guard is not None:
        boundary_guard()
    kwargs = {} if asset_reader is None else dict(asset_reader=asset_reader)
    engine.verify_round3_freeze(config["actual_R3_freeze"]["path"], **kwargs)
    import numpy as np
    coordinate = reader.json(parts["physical_coordinate_provenance"])
    with np.load(reader.path(parts["physical_reference"]), allow_pickle=False) as archive:
        matrix = geometry.physical_homography(coordinate, archive["x_m"].shape)
        validation = geometry.validate_reference_coordinates(matrix, archive["x_m"], archive["y_m"])
    points, phi, swi = geometry.read_measured_properties(
        reader.path(parts["depth_measurements"]), reader.path(parts["facies_properties"]))
    captured = {}
    def freeze(path):
        return engine.verify_round3_freeze(path, **kwargs)
    def estimate(*args, **kw):
        value = engine.build_estimate_geometry(*args, **kw)
        captured["geometry"] = value
        return value
    def own(*args, **kw):
        value = engine.prepare_own_colour_response(*args, **kw, **kwargs)
        captured["own_colour"] = value
        return value
    local_engine = _Proxy(engine, verify_round3_freeze=freeze,
                          build_estimate_geometry=estimate, prepare_own_colour_response=own)
    adapter = modules["adapter"]
    loader = adapter.load_selected
    if asset_reader is not None:
        # registered_clearwater has a direct read_bound_json call. Retain its
        # original code and numeric operations, replacing only that IO lookup.
        clearwater = _clone(source.registered_clearwater, read_bound_json=reader.json)
        adapter_source = _Proxy(source, registered_clearwater=clearwater)
        selected = _clone(loader, source=adapter_source)
        def loader(**kw):
            return selected(**kw, asset_reader=asset_reader)
    session = dict(validated, campaign=campaign, campaign_descriptor=config["campaign"],
                   loader=loader, matrix=matrix, coordinate_validation=validation,
                   depth_points=points, porosity=phi, swi=swi,
                   floor_curve=np.load(reader.path(parts["pooled_floor_curve"]), allow_pickle=False))
    provenance = dict(execution_mode=config["execution_mode"], executed_sources=provenance,
                      execution_commit=None if asset_reader is None else asset_reader.code_commit,
                      frozen_source_bindings=[] if asset_reader is None else asset_reader.source_bindings,
                      source_qualification_state="sealed_original_reference" if asset_reader is None
                      else asset_reader.source_qualification_state,
                      production_permission_granted=False, scientific_eligibility=False,
                      original_runner_code_objects_used=True, global_modules_modified=False)
    context = NativeContext(config, modules, reader, asset_reader, session, cases, provenance,
                            _clone(runner._load_entry, SESSION=session, engine=local_engine),
                            _clone(runner._summary_from_loaded, SESSION=session, engine=local_engine), captured,
                            boundary_guard)
    return context


def load_case(context, case_id):
    """Original full native response and R3 summary, once each, CPU only."""
    if case_id not in context.cases:
        raise ValueError("Case is outside the locked four-case diagnostic")
    if context.boundary_guard is not None:
        context.boundary_guard()
    context.captured.clear()
    entry = context.cases[case_id]["entry"]
    loaded = context.load_entry(entry, scope="frozen_evaluation",
                               frozen_calibration=context.configuration["actual_R3_freeze"]["path"])
    if context.boundary_guard is not None:
        context.boundary_guard()
    row = context.summary_from_loaded(loaded, entry, "frozen_evaluation", "cpu")
    if context.boundary_guard is not None:
        context.boundary_guard()
    static = {key: context.session[key] for key in ("matrix", "depth_points", "porosity", "swi", "floor_curve")}
    result = NativeResult(case_id, loaded, row, context.captured["geometry"],
                          context.captured["own_colour"], static, dict(context.provenance))
    context.captured["result"] = result
    return result


def enumerate_native_closure(configuration, metadata_reader, *, modules=None, size_resolver=None,
                             include_anchors=False):
    """Flat byte bindings actually required by native guards and these cases.

    JSON metadata may be parsed; scientific arrays/maps/opaque evidence are
    never decoded or opened by this enumeration. Missing sizes use stat only
    (or an explicit metadata size resolver). Original full plan admission is
    replayed with a collecting path reader; all B1/R1 freeze members are kept.
    Anchors are optional because load_case does not predict diluted fields.
    """
    config = _configuration(configuration)
    if type(include_anchors) is not bool:
        raise ValueError("Anchor payload inclusion must be an explicit boolean")
    modules = modules or _modules()
    assets = {}
    size_resolver = size_resolver or (lambda descriptor: Path(descriptor["path"]).stat().st_size)
    def add(descriptor, role):
        value = _descriptor({key: descriptor[key] for key in ("path", "sha256", "size_bytes") if key in descriptor})
        if "size_bytes" not in value:
            value["size_bytes"] = size_resolver(value)
        value = _descriptor(value)
        key = normalize_locator(value["path"])
        if key in assets:
            if (value["sha256"], value["size_bytes"]) != (assets[key]["descriptor"]["sha256"], assets[key]["descriptor"]["size_bytes"]):
                raise ValueError("Native closure has contradictory bytes for one locator")
            assets[key]["roles"].add(role)
        else:
            assets[key] = dict(descriptor=value, roles={role})
        return value
    def document(descriptor, role):
        add(descriptor, role)
        return metadata_reader.json(descriptor)
    class CollectingReader:
        def path(self, descriptor):
            add(descriptor, "original_plan_metadata_guard")
            return Path(descriptor["path"])
        def json(self, descriptor):
            return document(descriptor, "original_plan_metadata")
        def array(self, descriptor):
            raise RuntimeError("Native closure enumeration cannot decode scientific arrays")
    campaign = document(config["campaign"], "campaign")
    validated = modules["runner"].validate_selected_plan(campaign["plan"], CollectingReader())
    plan, parts = validated["plan"], validated["parts"]
    for descriptor in campaign["code_binding"].values():
        add(descriptor, "original_campaign_code")
    selection = document(config["profile_selection"], "locked_selection")
    cases = _cases(selection)
    def freeze_members(descriptor, role):
        value = document(descriptor, role)
        for path, item in value["calibration_files"].items():
            item = dict(item) if isinstance(item, Mapping) else dict(path=path, sha256=item)
            item.setdefault("path", path)
            if not Path(item["path"]).is_absolute():
                item["path"] = str(Path(descriptor["path"]).parent / item["path"])
            add(item, role + "_guard_member")
        return value
    freeze_members(config["actual_R3_freeze"], "actual_B1")
    r1 = freeze_members(parts["r1_freeze"], "original_R1_models_and_guards")
    for path, sha in r1["native_implementation_files"].items():
        add(dict(path=path, sha256=sha), "original_R1_implementation_evidence")
    r2 = document(parts["r2_input_plan"], "R2_input_plan")
    base = document(r2["base_input_plan"], "base_input_plan")
    for key in ("recipes", "atmosphere"):
        add(base[key], key)
    for key in ("dose_inventory", "light_epochs"):
        add(r2[key], key)
    domains = document(base["original_domain_reference"], "original_bottom_membership")
    entries = [case["entry"] for case in cases.values()]
    if include_anchors:
        anchors = {case["anchor_entry"]["run"]: case["anchor_entry"] for case in cases.values()}
        entries.extend(anchors.values())
    for entry in entries:
        add(entry["run_manifest"], "selected_camera_manifest")
        # Camera originals are evidence bindings, not alternate decoder input.
        binding = entry["source_binding"]
        for path, sha in (("source_path", "source_sha256"), ("raw_path", "raw_sha256")):
            if path in binding:
                add(dict(path=binding[path], sha256=binding[sha]), "camera_source_evidence")
        phase = document(entry["receipt"], "selected_phase")
        scientific = document(entry["scientific_receipt"], "selected_scientific")
        add(phase["labels_map"], "fixed_current_labels")
        for key in ("classes", "area_map"):
            add(phase["rig_domain_revision"][key], "selected_" + key)
        for key in ("full_extended_scientific", "baseline_extended_scientific"):
            for descriptor in modules["source"]._scientific_assets(phase[key]):
                add(descriptor, key)
        geo = phase["full_extended_scientific"]["binding"]["geometry"]
        add(dict(path=geo["native_manifest_path"], sha256=geo["native_manifest_sha256"]), "native_photometry_manifest")
        add(scientific["rig_geometry"]["frame_photo_top_keep"], "photo_top_keep")
        indexes = {0}
        binding = phase["binding"]["source"]
        if binding.get("studio_anchor"):
            parent = binding["parent_source"]
            indexes.add(parent["frame_index"] if parent["source_changed"] else parent["parent_source_frame_index"])
        for index in indexes:
            row = modules["adapter"]._old_domain_row(domains, entry["run"], index)
            add(row["parent_labels_map"], "original_bottom_membership_labels")
        transport = document(r2["clearwater_transports"][entry["run"]], "native_clearwater_transport")
        wash = document(dict(path=transport["clearwater_receipt"], sha256=transport["clearwater_receipt_sha256"]), "native_clearwater_receipt")
        for path, sha in (("rgb_path", "rgb_sha256"), ("valid_mask_path", "valid_mask_sha256")):
            add(dict(path=wash[path], sha256=wash[sha]), "native_clearwater_array")
    return dict(format="darsia-production-native-flat-closure-v1", complete=True,
                scientific_arrays_read=False, opaque_evidence_parsed=False,
                case_ids=list(LOCKED_CASES), includes_anchor_payloads=bool(include_anchors),
                assets=[dict(descriptor=row["descriptor"], roles=sorted(row["roles"]))
                        for _, row in sorted(assets.items())], production_permission_granted=False)

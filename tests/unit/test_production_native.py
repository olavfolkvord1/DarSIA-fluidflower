"""Tiny synthetic buffers/metadata and original function AST; no FF_AC data."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "src/darsia/fluidflower"
PACKAGE = ModuleType("production_native_unit_package")
PACKAGE.__path__ = [str(SOURCES)]
sys.modules[PACKAGE.__name__] = PACKAGE
spec = importlib.util.spec_from_file_location(PACKAGE.__name__ + ".production_native", SOURCES / "production_native.py")
operator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = operator
spec.loader.exec_module(operator)


def descriptor(path, sha="a" * 64, size=1):
    return dict(path=str(path), sha256=sha, size_bytes=size)


class Reader:
    def __init__(self, documents):
        self.documents = documents
        self.raw_baselines, self.physical_references = {}, {}
        self.paths, self.array_calls = [], []

    def path(self, bound):
        self.paths.append(dict(bound))
        return Path(bound["path"])

    def json(self, bound):
        return self.documents[bound["path"]]

    def array(self, bound):
        self.array_calls.append(bound)
        raise AssertionError("No array decoder in this reader fixture")


class Candidate(Reader):
    code_commit = "0" * 40
    execution_identity = dict(clean_tree=True)
    source_bindings = [dict(current_source=descriptor("current"), original_source=descriptor("old"))]
    source_qualification_state = "exact_CPU_parity_bootstrap"

    def source_path(self, current, sha):
        self.paths.append(dict(path=str(current), sha256=sha))
        return Path(current)

    def resolve_path(self, path, sha=None):
        return Path(path)


@dataclass
class Frame:
    run_id: str
    frame_index: int
    elapsed_h: float
    domain: np.ndarray
    facies: np.ndarray
    current_rgb: np.ndarray
    provenance: dict


@dataclass
class Geometry:
    x_m: np.ndarray
    y_m: np.ndarray
    source_binding: dict


@dataclass
class Prepared:
    aqueous_relative: np.ndarray
    gas_saturation: np.ndarray


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(operator, "_execution_identity", lambda reader: dict(reader.execution_identity))
    calls = []
    selection = {"cases": []}
    for case_id, (run, index, sha) in operator.LOCKED_CASES.items():
        entry = dict(run=run, frame_index=index, elapsed_h=float(index), selected_target_h=float(index),
                     source_binding=dict(source_sha256=sha), receipt=descriptor(tmp_path / (case_id + ".json")),
                     run_manifest=descriptor(tmp_path / (run + ".json")))
        anchor = dict(entry, frame_index=2, elapsed_h=2.24, selected_target_h=2.25,
                      source_binding=dict(source_sha256="b" * 64), receipt=descriptor(tmp_path / (run + "_anchor.json")))
        selection["cases"].append(dict(case_id=case_id, entry=entry, anchor_entry=anchor))
    physical = tmp_path / "physical.npz"
    np.savez(physical, x_m=np.zeros((2, 3)), y_m=np.zeros((2, 3)))
    floor = tmp_path / "floor.npy"
    np.save(floor, np.zeros(3))
    source = ModuleType("fake_native.source")
    source.__file__ = str(tmp_path / "source.py")
    Path(source.__file__).write_bytes(b"# exact original\n")
    source.sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    source.asset = lambda path: descriptor(path, source.sha256(path), Path(path).stat().st_size)
    adapter = ModuleType("fake_native.adapter")
    adapter.__file__ = source.__file__
    engine = ModuleType("fake_native.engine")
    engine.__file__ = source.__file__
    engine.FIT_RUNS = ["ac17", "ac19", "ac52"]
    def freeze(path, **kwargs):
        calls.append(("freeze", path, kwargs))
        return {}, "f" * 64
    engine.verify_round3_freeze = freeze
    def estimate(x, y, facies, points, **kwargs):
        calls.append(("estimate", kwargs))
        return Geometry(x, y, kwargs["source_binding"])
    engine.build_estimate_geometry = estimate
    def own(response, folder, geo, **kwargs):
        calls.append(("own", kwargs))
        return dict(aqueous_relative=np.ones((2, 3), np.float32), available=np.ones((2, 3), bool))
    engine.prepare_own_colour_response = own
    def summarize(response, geo, **kwargs):
        calls.append(("summary", kwargs))
        return dict(excluded_parts={}, native_statistic=np.float64(3),
                    source_binding=kwargs["source_binding"], geometry_source_binding=geo.source_binding)
    engine.summarize_round3_response = summarize
    geometry = ModuleType("fake_native.geometry")
    geometry.physical_homography = lambda *args: np.eye(3)
    geometry.validate_reference_coordinates = lambda *args: dict(passed=True)
    geometry.read_measured_properties = lambda *args: (np.zeros((66, 3)), {0: .4}, {0: .1})
    geometry.frame_coordinates = lambda *args: (np.zeros((2, 3)), np.ones((2, 3)))
    def selected(**kwargs):
        calls.append(("selected", kwargs))
        entry = kwargs["entry"]
        frame = Frame(entry["run"], entry["frame_index"], entry["elapsed_h"],
                      np.ones((2, 3), bool), np.zeros((2, 3), int), np.ones((2, 3, 3), np.float32),
                      dict(current_source="operational"))
        prepared = Prepared(np.ones((2, 3), np.float32), np.zeros((2, 3), np.float32))
        response = dict(frame=frame, prepared=prepared, native_signal=prepared.aqueous_relative,
                        quantifiable=frame.domain, linear_relative=prepared.aqueous_relative,
                        joint_signal=prepared.aqueous_relative, swi=np.full((2, 3), .1))
        return dict(response=response, receipt=dict(rig_domain_revision={}, full_extended_scientific=dict(binding=dict(geometry={}))),
                    source_binding=entry["source_binding"], photo_support=frame.domain, path_folder="original-models")
    adapter.load_selected = selected
    # The original runner code is compiled unchanged against tiny fake callees.
    runner = ModuleType("fake_native.runner")
    runner.__file__ = source.__file__
    runner.CAMPAIGN_CONTRACT = "ffac-common-mass-round3-selected-campaign-v1"
    runner.SESSION = {"must_remain": True}
    tree = ast.parse((SOURCES / "tools/run_ffac_common_mass_round3_selected.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_load_entry", "_summary_from_loaded"}]
    runner.__dict__.update(np=np, engine=engine, geometry_inputs=geometry, native=SimpleNamespace())
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCES / "tools/run_ffac_common_mass_round3_selected.py"), "exec"), runner.__dict__)
    parts = {name: descriptor(tmp_path / name) for name in ("depth_measurements", "facies_properties", "physical_coordinate_provenance")}
    parts.update(physical_reference=descriptor(physical), pooled_floor_curve=descriptor(floor))
    plan = dict(selected_adapter=dict(source=source.asset(adapter.__file__), function="load_selected"))
    entries = [row["entry"] for row in selection["cases"]] + list({row["anchor_entry"]["run"]: row["anchor_entry"] for row in selection["cases"]}.values())
    runner.validate_selected_plan = lambda bound, reader: dict(plan=plan, parts=parts, entries=entries, reader=reader)
    config = dict(format=operator.FORMAT, execution_mode="original_reference", purpose="file_handling_acceptance",
                  backend="cpu", production_permission_granted=False, campaign=descriptor("campaign"),
                  profile_selection=descriptor("selection", operator.SELECTION_SHA256),
                  actual_R3_freeze=descriptor("original_B1", operator.FREEZE_SHA256))
    campaign = dict(contract=runner.CAMPAIGN_CONTRACT, source_equivalence_exact=True,
                    production_automatically_allowed=False, code_binding={source.__name__: source.asset(source.__file__)},
                    plan=descriptor("plan"))
    documents = {"campaign": campaign, "selection": selection, parts["physical_coordinate_provenance"]["path"]: {}}
    return SimpleNamespace(config=config, modules=dict(runner=runner, source=source, geometry=geometry,
                          engine=engine, adapter=adapter), reader=Reader(documents), calls=calls,
                          selection=selection, parts=parts, campaign=campaign)


def test_exact_original_code_objects_call_once_and_local_session(fixture):
    f = fixture
    context = operator.prepare_native_context(f.config, f.reader, modules=f.modules)
    assert context.load_entry.__code__ is f.modules["runner"]._load_entry.__code__
    assert context.summary_from_loaded.__code__ is f.modules["runner"]._summary_from_loaded.__code__
    result = operator.load_case(context, "ac17_f6")
    assert [name for name, *rest in f.calls].count("selected") == 1
    assert [name for name, *rest in f.calls].count("summary") == 1
    assert f.modules["runner"].SESSION == {"must_remain": True}
    products = result.scientific_products()
    assert products["native"]["frame"]["current_rgb"] is result.loaded["response"]["frame"].current_rgb
    assert products["own_colour"]["aqueous_relative"] is result.own_colour["aqueous_relative"]
    assert "provenance" not in products["native"]["frame"]
    assert products["summary"]["native_statistic"].tobytes() == np.float64(3).tobytes()


def test_guard_failure_never_calls_native_or_retries(fixture):
    def stop(): raise RuntimeError("authority denied")
    with pytest.raises(RuntimeError, match="authority"):
        operator.prepare_native_context(fixture.config, fixture.reader, modules=fixture.modules, boundary_guard=stop)
    assert fixture.calls == []


def test_candidate_forwards_reader_without_global_mutation(fixture):
    f = fixture
    # This source function represents the original direct JSON IO boundary.
    namespace = {"read_bound_json": lambda bound: pytest.fail("original locator opened")}
    exec("def registered_clearwater(bound):\n    return read_bound_json(bound)\n", namespace)
    clearwater = namespace["registered_clearwater"]
    f.modules["source"].registered_clearwater = clearwater
    config = dict(f.config, execution_mode="candidate_io")
    reader = Candidate(f.reader.documents)
    context = operator.prepare_native_context(config, reader, modules=f.modules)
    result = operator.load_case(context, "ac19_f34")
    own_kwargs = next(row[1] for row in f.calls if row[0] == "own")
    selected_kwargs = next(row[1] for row in f.calls if row[0] == "selected")
    assert own_kwargs["asset_reader"] is reader
    assert selected_kwargs["asset_reader"] is reader
    assert f.modules["source"].registered_clearwater is clearwater
    assert f.modules["runner"].SESSION == {"must_remain": True}
    assert result.provenance["execution_commit"] == reader.code_commit
    assert result.provenance["scientific_eligibility"] is False


@pytest.mark.parametrize("kind", ["backend", "permission", "selection", "freeze", "mode"])
def test_scope_fails_before_native_setup(fixture, kind):
    config = deepcopy(fixture.config)
    if kind == "backend": config["backend"] = "auto"
    if kind == "permission": config["production_permission_granted"] = 0
    if kind in {"selection", "freeze"}: config["profile_selection" if kind == "selection" else "actual_R3_freeze"]["sha256"] = "f" * 64
    if kind == "mode": config["execution_mode"] = "implicit"
    with pytest.raises(ValueError): operator.prepare_native_context(config, fixture.reader, modules=fixture.modules)
    assert fixture.calls == []


@pytest.mark.parametrize("kind", ["source", "frame", "anchor", "different_anchor", "missing"])
def test_locked_selection_rejects_changed_identity(fixture, kind):
    value = deepcopy(fixture.selection)
    if kind == "source": value["cases"][0]["entry"]["source_binding"]["source_sha256"] = "f" * 64
    if kind == "frame": value["cases"][0]["entry"]["frame_index"] = 6.0
    if kind == "anchor": value["cases"][0]["anchor_entry"]["selected_target_h"] = 3
    if kind == "different_anchor": value["cases"][3]["anchor_entry"]["elapsed_h"] = 2.3
    if kind == "missing": value["cases"].pop()
    with pytest.raises(ValueError): operator._cases(value)


def test_indexed_entry_is_not_replaced_by_similar_copy(fixture):
    fixture.selection["cases"][0]["entry"] = dict(fixture.selection["cases"][0]["entry"], extra="not_original")
    with pytest.raises(ValueError, match="indexed entry"):
        operator.prepare_native_context(fixture.config, fixture.reader, modules=fixture.modules)


def test_wrong_original_source_stops_reference(fixture):
    fixture.campaign["code_binding"]["fake_native.source"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="sealed original"):
        operator.prepare_native_context(fixture.config, fixture.reader, modules=fixture.modules)


def test_reference_cannot_hide_manifest_source_substitution(fixture):
    with pytest.raises(ValueError, match="original reader"):
        operator.prepare_native_context(fixture.config, Candidate(fixture.reader.documents), modules=fixture.modules)


def test_profile_cannot_bootstrap_without_actual_parity(fixture):
    config = dict(fixture.config, execution_mode="candidate_io", purpose="profile")
    with pytest.raises(ValueError, match="actual exact CPU parity"):
        operator.prepare_native_context(config, Candidate(fixture.reader.documents), modules=fixture.modules)


def test_unknown_case_stops_before_load(fixture):
    context = operator.prepare_native_context(fixture.config, fixture.reader, modules=fixture.modules)
    before = len(fixture.calls)
    with pytest.raises(ValueError, match="outside"):
        operator.load_case(context, "ac17_f7")
    assert len(fixture.calls) == before


def test_candidate_reader_requires_source_qualification(fixture):
    config = dict(fixture.config, execution_mode="candidate_io")
    with pytest.raises(ValueError, match="source qualification"):
        operator.prepare_native_context(config, fixture.reader, modules=fixture.modules)


def test_changed_execution_tree_stops_before_any_native_call(fixture, monkeypatch):
    monkeypatch.setattr(operator, "_execution_identity", lambda reader: dict(clean_tree=False))
    config = dict(fixture.config, execution_mode="candidate_io")
    with pytest.raises(ValueError, match="execution tree changed"):
        operator.prepare_native_context(config, Candidate(fixture.reader.documents), modules=fixture.modules)
    assert fixture.calls == []


def test_clearwater_clone_uses_bound_io_and_exact_code():
    namespace = {"read_bound_json": lambda bound: pytest.fail("direct JSON IO"), "calls": []}
    exec("def wash(bound):\n    calls.append(bound)\n    return read_bound_json(bound)\n", namespace)
    original = namespace["wash"]
    reader = Reader({"metadata": {"exact": 1}})
    cloned = operator._clone(original, read_bound_json=reader.json)
    assert cloned.__code__ is original.__code__
    assert cloned(descriptor("metadata")) == {"exact": 1}
    assert len(namespace["calls"]) == 1
    assert original.__globals__["read_bound_json"] is namespace["read_bound_json"]


@pytest.fixture
def closure(fixture):
    f = fixture
    docs = f.reader.documents
    def asset(name): return descriptor("C:/synthetic/" + name)
    def document(name, value):
        desc = asset(name)
        docs[desc["path"]] = value
        return desc
    domain_rows = []
    def raw(name):
        return dict(native_centre=dict(log_linear_rgb=asset(name + "/rgb.npy"), valid_mask=asset(name + "/valid.npy")),
                    ring_windows=[dict(log_linear_rgb=asset(name + "/ring.npy"), canvas_xyxy=[0, 0, 1, 1])])
    def scientific_assets(value):
        rows = [value["native_centre"][key] for key in ("log_linear_rgb", "valid_mask")]
        return rows + [window["log_linear_rgb"] for window in value["ring_windows"]]
    f.modules["source"]._scientific_assets = scientific_assets
    all_entries = [row["entry"] for row in f.selection["cases"]]
    all_entries += list({row["anchor_entry"]["run"]: row["anchor_entry"] for row in f.selection["cases"]}.values())
    for entry in all_entries:
        name = entry["run"] + "/" + str(entry["frame_index"])
        binding = dict(entry["source_binding"], source_path="C:/synthetic/" + name + ".JPG", raw_path="C:/synthetic/" + name + ".ARW", raw_sha256="c" * 64,
                       studio_anchor=True, parent_source=dict(source_changed=False, parent_source_frame_index=entry["frame_index"]))
        entry["source_binding"] = binding
        current, baseline = raw(name), raw(entry["run"] + "/baseline")
        current["binding"] = dict(geometry=dict(native_manifest_path="C:/synthetic/" + entry["run"] + "_native.json", native_manifest_sha256="a" * 64))
        docs[entry["receipt"]["path"]] = dict(labels_map=asset(name + "/labels.png"), binding=dict(source=binding),
              rig_domain_revision=dict(classes=asset(name + "/classes.npz"), area_map=asset(name + "/area.npz")),
              full_extended_scientific=current, baseline_extended_scientific=baseline)
        entry["scientific_receipt"] = document(name + "/scientific.json", dict(rig_geometry=dict(frame_photo_top_keep=asset(name + "/top.npz"))))
        for index in (0, entry["frame_index"]):
            row = dict(run=entry["run"], frame_index=index, parent_labels_map=asset(entry["run"] + "/old_" + str(index) + ".png"))
            if row not in domain_rows: domain_rows.append(row)
    anchors = {entry["run"]: entry for entry in all_entries if entry["frame_index"] == 2}
    for case in f.selection["cases"]:
        case["anchor_entry"] = anchors[case["entry"]["run"]]
    # Fake original metadata validator still records every binding it checks.
    f.modules["runner"].validate_selected_plan = lambda bound, reader: (
        reader.path(asset("original-index-receipt.json")) and
        dict(plan={}, parts=f.parts, entries=all_entries, reader=reader))
    f.modules["adapter"]._old_domain_row = lambda manifest, run, index: next(row for row in manifest["frames"] if row["run"] == run and row["frame_index"] == index)
    f.parts["r1_freeze"] = document("r1.json", dict(calibration_files={"C:/synthetic/models/flash.json": "a" * 64},
                                                        native_implementation_files={"C:/synthetic/original/flash.py": "d" * 64}))
    docs[f.config["actual_R3_freeze"]["path"]] = dict(calibration_files={
        "C:/synthetic/B1/member_" + str(i) + ".gz": "e" * 64 for i in range(2308)})
    base = document("base.json", dict(recipes=asset("recipes.json"), atmosphere=asset("atmosphere.json"),
                                        original_domain_reference=document("domain.json", dict(frames=domain_rows))))
    transports = {}
    for run in ("ac17", "ac19", "ac52"):
        wash = document(run + "/wash.json", dict(rgb_path="C:/synthetic/" + run + "/wash.npy", rgb_sha256="a" * 64,
                                                 valid_mask_path="C:/synthetic/" + run + "/wash-valid.npy", valid_mask_sha256="a" * 64))
        transports[run] = document(run + "/transport.json", dict(clearwater_receipt=wash["path"], clearwater_receipt_sha256=wash["sha256"]))
    f.parts["r2_input_plan"] = document("r2-plan.json", dict(base_input_plan=base, dose_inventory=asset("dose.json"),
                             light_epochs=asset("epochs.json"), clearwater_transports=transports))
    return f


def test_closure_preserves_all_2308_guard_members_and_exact_four_payloads(closure):
    f = closure
    value = operator.enumerate_native_closure(f.config, f.reader, modules=f.modules, size_resolver=lambda bound: 1)
    rows = value["assets"]
    assert sum("actual_B1_guard_member" in row["roles"] for row in rows) == 2308
    assert sum("original_R1_implementation_evidence" in row["roles"] for row in rows) == 1
    assert sum("selected_phase" in row["roles"] for row in rows) == 4
    assert sum("native_clearwater_array" in row["roles"] for row in rows) == 6
    assert value["scientific_arrays_read"] is False
    assert value["opaque_evidence_parsed"] is False
    assert f.reader.array_calls == []
    assert not any("old21" in row["descriptor"]["path"] for row in rows)


def test_anchor_payloads_are_explicit_and_deduplicated(closure):
    value = operator.enumerate_native_closure(closure.config, closure.reader, modules=closure.modules,
                                               size_resolver=lambda bound: 1, include_anchors=True)
    assert sum("selected_phase" in row["roles"] for row in value["assets"]) == 7
    with pytest.raises(ValueError, match="explicit boolean"):
        operator.enumerate_native_closure(closure.config, closure.reader, modules=closure.modules, include_anchors=1)


def test_conflicting_member_never_merges_with_scientific_asset(closure):
    phase = closure.reader.documents[closure.selection["cases"][0]["entry"]["receipt"]["path"]]
    path = phase["labels_map"]["path"]
    closure.reader.documents[closure.config["actual_R3_freeze"]["path"]]["calibration_files"][path] = "f" * 64
    with pytest.raises(ValueError, match="contradictory bytes"):
        operator.enumerate_native_closure(closure.config, closure.reader, modules=closure.modules, size_resolver=lambda bound: 1)

"""Tiny synthetic N1 products against unchanged native calculation ASTs."""
import ast
from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SOURCES = Path(__file__).resolve().parents[2] / "src/darsia/fluidflower"
spec = importlib.util.spec_from_file_location("retained_products_unit", SOURCES / "production_retained.py")
retained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)


def definitions(relative, names, namespace):
    tree = ast.parse((SOURCES / relative).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCES / relative), "exec"), namespace)
    return namespace


@pytest.fixture
def synthetic(monkeypatch):
    shape = (3, 4)
    phase = np.array([[1, 2, 1, 0], [2, 1, 1, 2], [1, 0, 2, 255]], np.uint8)
    domain = phase != 255
    known = domain.copy()
    known[1, 2] = False
    bottom = np.zeros(shape, bool)
    bottom[0, 0] = True
    eligible = domain & ~bottom
    frame = SimpleNamespace(domain=domain, phase_labels=phase, known_physical_domain=known,
        mass_eligible_mask=eligible, depth_m=np.full(shape, .008), porosity=np.full(shape, .41),
        pixel_area_m2=np.full(shape, 1.e-6), excluded_masks={"bottom_strip": bottom},
        run_id="synthetic", frame_index=1, elapsed_h=9., nominal_injected_mass_g=1.,
        recipe={}, provenance={})
    value = np.arange(12).reshape(shape)
    physics = SimpleNamespace(solubility_co2=1.83+value*.031,
                              density_gaseous_co2=1.93+value*.065)
    parameters = dict(candidate="N1", gain=24.253536351793787, beta=.6448739280294578)
    response = dict(frame=frame, physics=physics, quantifiable=known.copy(),
        prepared=SimpleNamespace(aqueous_relative=np.asarray((value+.51)*.003, np.float32)),
        linear_relative=np.zeros(shape, np.float32), swi=np.full(shape, .231),
        frozen_parameters=parameters)
    dilution = definitions("concentration_joint.py", {"DilutionAnchor", "_anchor", "dilution_factor"},
                           dict(np=np, dataclass=dataclass, Mapping=Mapping, __name__=__name__))
    volume = np.where(known, frame.pixel_area_m2*frame.depth_m*frame.porosity, 0.)
    geometry_tree = ast.parse((SOURCES.parent / "measure/integration.py").read_text(encoding="utf-8"))
    geometry_class = next(node for node in geometry_tree.body if isinstance(node, ast.ClassDef) and node.name == "Geometry")
    integration = next(node for node in geometry_class.body if isinstance(node, ast.FunctionDef) and node.name == "integrate")
    geometry_ns = dict(np=np, darsia=SimpleNamespace(ExtensiveImage=type("Extensive", (), {})), __name__=__name__)
    annotation = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[annotation, integration], type_ignores=[])),
                 "unchanged native geometry integration", "exec"), geometry_ns)
    geometry = SimpleNamespace(space_dim=2, cached_voxel_volume=volume,
        _prepare_cached_voxel_volume=lambda shape: None)
    geometry.integrate = lambda array: geometry_ns["integrate"](geometry, array)
    def scalar(array, elapsed_h):
        return SimpleNamespace(img=np.asarray(array, np.float32), name="synthetic", date=None, time=elapsed_h)
    response["analysis"] = SimpleNamespace(geometry=geometry, _scalar=scalar)
    mass_tree = ast.parse((SOURCES.parent / "multiphase/mass_analysis.py").read_text(encoding="utf-8"))
    mass_class = next(node for node in mass_tree.body if isinstance(node, ast.ClassDef) and node.name == "CO2MassAnalysis")
    mass_method = next(node for node in mass_class.body if isinstance(node, ast.FunctionDef) and node.name == "mass_analysis")
    mass_ns = dict(np=np, darsia=SimpleNamespace(SimpleMassAnalysisResults=SimpleNamespace,
        full_like=lambda image, values: SimpleNamespace(img=np.full_like(image.img, values))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[annotation, mass_method], type_ignores=[])),
                 "unchanged native mass analysis", "exec"), mass_ns)
    physics.mass_analysis = lambda a, g: mass_ns["mass_analysis"](physics, a, g)
    namespace = definitions("phase_ffac_common_mass_round2.py", {
        "_physical_partition", "_native_fraction", "interval_residual", "evaluate_round2_frame"},
        dict(np=np, MAX_DEPTH_M=.0155, MAX_POROSITY=.45, CONTRACT="original-R2",
             _validate_parameters=lambda value: None, dilution_factor=dilution["dilution_factor"]))
    modules = {"engine": SimpleNamespace(r2=SimpleNamespace(dilution_factor=dilution["dilution_factor"]))}
    choice = dict(verifier="round3", asset=dict(path="exact synthetic freeze", sha256="a"*64, size_bytes=1))
    calls = []
    def selected(*args):
        calls.append(args)
        return dict(parameters), "a"*64
    monkeypatch.setattr(retained, "_selected_n1", selected)
    anchor = dict(elapsed_h=2.247, affected_pore_volume_m3=1.e-8)
    box1 = domain & (np.indices(shape)[0] < 1)
    domains = dict(full_rig=domain, box1=box1, box2=domain & ~box1)
    return SimpleNamespace(response=response, modules=modules, choice=choice, anchor=anchor,
        domains=domains, parameters=parameters, old=namespace["evaluate_round2_frame"], calls=calls)


def evaluate(f, **kwargs):
    return retained.evaluate_main(f.response, f.anchor, modules=f.modules,
        selected_calibration=f.choice, domain_masks=f.domains, **kwargs)


def test_native_fields_casts_and_domain_masses_are_bit_identical(synthetic):
    f = synthetic
    old = f.old(f.response, f.parameters, f.anchor, fields=True)
    result = evaluate(f, kept_fields=retained.FIELDS)
    for name in retained.FIELDS:
        first, second = old["fields"][name], result["fields"][name]
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes(order="C") == second.tobytes(order="C")
    for name in ("total_mass_g", "aqueous_mass_g", "gas_mass_g", "dilution_factor", "affected_pore_volume_m3"):
        assert np.float64(old[name]).tobytes() == np.float64(result[name]).tobytes()
    for name, domain in f.domains.items():
        for mass, field in (("total_mass_g", "mass_density_g_l"),
                            ("aqueous_mass_g", "aqueous_mass_density_g_l"),
                            ("gas_mass_g", "gas_mass_density_g_l")):
            reference = float(f.response["analysis"].geometry.integrate(np.where(
                old["fields"]["main_mass_eligible_mask"] & domain, old["fields"][field], 0.)))*1000
            assert np.float64(reference).tobytes() == np.float64(result["domains"][name][mass]).tobytes()
    for name, part in result["excluded_mass"].items():
        for mass in ("total_mass_g", "aqueous_mass_g", "gas_mass_g"):
            assert np.float64(old["excluded_mass"][name][mass]).tobytes() == np.float64(part[mass]).tobytes()
    assert result["scalar_reference"] == "native_final_calibration_fields"
    assert len(f.calls) == 1 and result["scientific_eligibility"] is False


def test_only_declared_maps_are_materialized_and_no_inputs_modified(synthetic):
    f = synthetic
    original = {name: array.tobytes() for name, array in vars(f.response["frame"]).items()
                if isinstance(array, np.ndarray)}
    prepared = f.response["prepared"].aqueous_relative.tobytes()
    result = evaluate(f, kept_fields={"c_aq_g_l", "s_g", "phase_labels"})
    assert set(result["fields"]) == {"c_aq_g_l", "s_g", "phase_labels"}
    assert not ({"E_lo_g", "E_hi_g", "U_rest_g", "paired_estimates", "excluded_parts"} & set(result))
    assert original == {name: array.tobytes() for name, array in vars(f.response["frame"]).items()
                        if isinstance(array, np.ndarray)}
    assert prepared == f.response["prepared"].aqueous_relative.tobytes()


@pytest.mark.parametrize("kind", ["field", "domain_missing", "domain_dtype", "domain_shape", "full_domain", "calibration"])
def test_wrong_products_domains_or_freeze_fail_closed(synthetic, kind):
    kwargs = {}
    if kind == "field": kwargs["kept_fields"] = {"inferred_new_product"}
    if kind == "domain_missing": synthetic.domains.pop("box1")
    if kind == "domain_dtype": synthetic.domains["box1"] = synthetic.domains["box1"].astype(np.uint8)
    if kind == "domain_shape": synthetic.domains["box1"] = np.zeros((1, 1), bool)
    if kind == "full_domain": synthetic.domains["full_rig"] = np.zeros((3, 4), bool)
    if kind == "calibration": synthetic.response["frozen_parameters"] = {**synthetic.parameters, "beta": .2}
    with pytest.raises(ValueError): evaluate(synthetic, **kwargs)


def test_retained_path_has_no_secondary_or_summary_scientific_calls():
    source = (SOURCES / "production_retained.py").read_text(encoding="utf-8")
    compile(source, str(SOURCES / "production_retained.py"), "exec")
    tree = ast.parse(source)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)}
    assert not calls & {"_joint_signal", "SimpleFlash", "summarize_round3_response", "_native_histogram",
                        "_pairs", "distances_for_part", "build_estimate_geometry", "prepare_own_colour_response"}


def test_prepare_runs_original_optics_once_with_frozen_r1_guard_and_measured_swi(tmp_path, monkeypatch):
    parameters = dict(candidate="N1", gain=24.253536351793787, beta=.6448739280294578)
    calls = []
    def verify(path, **kwargs):
        calls.append(("r1_verify", path, kwargs))
        return dict(path_folder=str(tmp_path), cross_host_source_equivalence={"exact": True},
                    production_io_source_provenance={"separate_execution": True}), "r1-sha"
    namespace = {"calls": calls}
    exec("def native_prepare(frame, folder, **kwargs):\n"
         "    calls.append(('native', frame, folder, kwargs))\n"
         "    verify_frozen_calibration(kwargs['frozen_calibration_path'], asset_reader=kwargs['asset_reader'])\n"
         "    return {'frame': frame, 'unchanged_optics': True}\n", namespace)
    original = namespace["native_prepare"]
    r1 = SimpleNamespace(prepare_native_response=original, normalize_run=lambda run: run)
    r2 = SimpleNamespace(verify_original_round1_freeze=verify, HELDOUT_RUNS=("ac44",))
    frame = SimpleNamespace(run_id="ac44", domain=np.ones((2, 2), bool), facies=np.array([[1, 2], [1, 2]]))
    reader = object()
    monkeypatch.setattr(retained, "_selected_n1", lambda *args: (dict(parameters), "selected-sha"))
    result = retained.prepare_n1_response(frame, tmp_path, {"r1_freeze": {"path": "frozen-r1"}},
        {"1": .23, "2": .31}, "frozen_evaluation", "original-access-B1",
        modules={"engine": SimpleNamespace(r1=r1, r2=r2)}, selected_calibration={"explicit": True}, asset_reader=reader)
    assert [call[0] for call in calls].count("native") == 1
    assert calls[1][3] == dict(flash_kind="titration", frozen_calibration_path="frozen-r1", asset_reader=reader)
    assert "verify_frozen_calibration" not in original.__globals__
    assert result["frozen_parameters"] == parameters and result["selected_calibration_sha256"] == "selected-sha"
    assert result["swi"].dtype == np.float64 and np.array_equal(result["swi"], [[.23, .31], [.23, .31]])
    assert not {"joint_signal", "linear_relative", "linear_flash"} & set(result)


def test_missing_freeze_scope_stops_before_any_native_prepare(monkeypatch):
    monkeypatch.setattr(retained, "_selected_n1", lambda *args: pytest.fail("freeze/preparation was entered"))
    for scope, freeze in (("calibration", "B1"), ("frozen_evaluation", None)):
        with pytest.raises(ValueError):
            retained.prepare_n1_response(None, None, None, None, scope, freeze,
                                         modules={}, selected_calibration={})


def test_run_case_preserves_guard_failures_and_does_not_retry(synthetic, monkeypatch):
    calls = []
    monkeypatch.setattr(retained, "load_response", lambda *args, **kwargs: calls.append("load") or {"response": synthetic.response})
    context = SimpleNamespace(modules=synthetic.modules, asset_reader=None,
        boundary_guard=lambda: (_ for _ in ()).throw(RuntimeError("authority stopped")))
    with pytest.raises(RuntimeError, match="authority"):
        retained.run_case(context, "synthetic", selected_calibration=synthetic.choice,
            anchor=synthetic.anchor, domain_masks=synthetic.domains, kept_fields=())
    assert calls == ["load"] and synthetic.calls == []


def test_main_export_zeroes_excluded_only_and_preserves_raw_native_maps(synthetic):
    result = evaluate(synthetic, kept_fields=retained.FIELDS)
    before = {name: value.tobytes() for name, value in result["fields"].items()}
    exported = retained.main_products(result)
    assert set(exported["products"]) == {"concentration_aqueous", "saturation_free_gas", "phase_labels",
                                         "mass_total", "mass_aqueous", "mass_free_gas"}
    assert exported["products"]["concentration_aqueous"] is result["fields"]["c_aq_g_l"]
    mask = result["fields"]["main_mass_eligible_mask"]
    for name, native in (("mass_total", "mass_density_g_l"), ("mass_aqueous", "aqueous_mass_density_g_l"),
                         ("mass_free_gas", "gas_mass_density_g_l")):
        projected = exported["products"][name]
        raw = result["fields"][native]
        assert projected.dtype == np.float32
        assert np.all(projected[~mask] == 0) and projected[mask].tobytes() == raw[mask].tobytes()
        assert projected.tobytes() == np.where(mask, raw, 0.).tobytes()
    assert result["fields"]["mass_density_g_l"][0, 0] > 0  # Real excluded mass stays in the raw view.
    assert before == {name: value.tobytes() for name, value in result["fields"].items()}
    assert exported["units"]["mass_total"] == "g/L"
    assert exported["required_transport_weight_units"] == "L pore volume per pixel"
    assert exported["scientific_eligibility"] is False


def test_separate_total_cast_is_not_sum_of_cast_components(synthetic):
    # An exact half-ULP witness: gas rounds to 1, water to 2**-24, while
    # their float64 sum rounds to the next float32 above 1.
    synthetic.response["swi"][0, 1] = .5
    synthetic.response["physics"].density_gaseous_co2[0, 1] = 2. + 2.**-23
    synthetic.response["physics"].solubility_co2[0, 1] = 2.**-23
    result = evaluate(synthetic, kept_fields=retained.FIELDS)
    fields = result["fields"]
    physics = synthetic.response["physics"]
    aqueous, gas = fields["c_aq_relative"], fields["s_g"].copy()
    gas[~np.isfinite(gas)] = 0
    raw_g = physics.density_gaseous_co2*gas
    raw_aq = physics.solubility_co2*aqueous*np.clip(1-gas, 0, None)
    assert fields["mass_density_g_l"].tobytes() == (raw_g+raw_aq).astype(np.float32).tobytes()
    # This fixture witnesses the rounding distinction, not just equal zeros.
    assert fields["mass_density_g_l"].tobytes() != (
        fields["aqueous_mass_density_g_l"]+fields["gas_mass_density_g_l"]).tobytes()


@pytest.mark.parametrize("kind", ["actual_B1", "zero_gain", "negative_gain", "nonfinite_gain",
    "negative_beta", "gas_parameter", "other_candidate", "sha", "size", "returned_sha", "verifier"])
def test_selected_n1_uses_verified_freeze_parameters_not_b0_constant(tmp_path, monkeypatch, kind):
    # Exercise the actual selector, not the evaluator fixture's fake selector.
    # Only native freeze verification is mocked; real local SHA/size checks and
    # the original parameter-validation function still run.
    io_spec = importlib.util.spec_from_file_location("selected_n1_unit_io", SOURCES / "production_io.py")
    io_module = importlib.util.module_from_spec(io_spec)
    io_spec.loader.exec_module(io_module)
    package = ModuleType("darsia")
    package.__path__ = []
    fluidflower = ModuleType("darsia.fluidflower")
    fluidflower.__path__ = []
    monkeypatch.setitem(sys.modules, "darsia", package)
    monkeypatch.setitem(sys.modules, "darsia.fluidflower", fluidflower)
    monkeypatch.setitem(sys.modules, "darsia.fluidflower.production_io", io_module)
    parameters = dict(candidate="N1", gain=23.882294164545023, beta=.6143584288103799)
    if kind == "zero_gain": parameters["gain"] = 0.
    if kind == "negative_gain": parameters["gain"] = -1.
    if kind == "nonfinite_gain": parameters["gain"] = float("inf")
    if kind == "negative_beta": parameters["beta"] = -1.
    if kind == "gas_parameter": parameters["gas_scale"] = 1.
    if kind == "other_candidate": parameters["candidate"] = "N2"
    path = tmp_path / "synthetic_freeze.json"
    path.write_text(json.dumps({"parameters": parameters}), encoding="utf-8")
    io = io_module.NasIO()
    bound = io.asset(path)
    calls = []
    def verify(frozen_path, **kwargs):
        calls.append((frozen_path, kwargs))
        # Same Python JSON numeric values as the original native freeze loader.
        document = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
        return document, "f"*64 if kind == "returned_sha" else io.asset(frozen_path)["sha256"]
    validate = definitions("phase_ffac_common_mass_round2.py", {"_validate_parameters"},
        dict(np=np, CANDIDATES=("N1", "N2", "N3", "J"),
             candidate_availability=lambda: {key: {"available": True} for key in ("N1", "N2", "N3", "J")}))["_validate_parameters"]
    engine = SimpleNamespace(GAIN=24.253536351793787, verify_round3_freeze=verify,
        r2=SimpleNamespace(verify_round2_freeze=lambda *a, **k: pytest.fail("other calibration evaluated"),
                           _validate_parameters=validate))
    choice = dict(verifier="round3", asset=bound)
    if kind == "sha": bound["sha256"] = "f"*64
    if kind == "size": bound["size_bytes"] += 1
    if kind == "verifier": choice["verifier"] = "unregistered"
    reader = SimpleNamespace(path=io.verify)
    if kind == "actual_B1":
        selected, digest = retained._selected_n1({"engine": engine}, choice, reader)
        assert selected == parameters and selected["gain"] != engine.GAIN
        assert digest == bound["sha256"] and calls == [(str(path), {"asset_reader": reader})]
    else:
        with pytest.raises(ValueError): retained._selected_n1({"engine": engine}, choice, reader)
        if kind in {"sha", "size", "verifier"}: assert calls == []

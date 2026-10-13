"""UNQUALIFIED draft: one frozen N1 calibration and explicit retained products.

This is a separate production proposal, not an R3 replacement. B1 is the current
performance/parity test choice, not an assertion about Olav's final selection.
It grants no
admission, production permission, parity result or performance claim. The
caller must bind/review this source, compare actual retained products and use
the existing native input, source, phase, freeze and live-control guards.
"""
from __future__ import annotations

from types import FunctionType


FIELDS = frozenset({"c_aq_g_l", "c_aq_relative", "s_g", "phase_labels",
                    "unquantified_phase_mask", "main_mass_eligible_mask",
                    "mass_density_g_l", "aqueous_mass_density_g_l",
                    "gas_mass_density_g_l"})


def _clone(function, **overrides):
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def _selected_n1(modules, selected_calibration, asset_reader):
    engine = modules["engine"]
    if set(selected_calibration) != {"verifier", "asset"}:
        raise ValueError("One explicit selected frozen calibration binding is required")
    verifiers = {"round3": engine.verify_round3_freeze, "round2": engine.r2.verify_round2_freeze}
    if selected_calibration["verifier"] not in verifiers:
        raise ValueError("Unknown native frozen-calibration verifier")
    bound = selected_calibration["asset"]
    from darsia.fluidflower.production_io import NasIO
    if asset_reader is None:
        NasIO().verify(bound)
    else:
        asset_reader.path(bound)
    kwargs = {} if asset_reader is None else {"asset_reader": asset_reader}
    freeze, digest = verifiers[selected_calibration["verifier"]](bound["path"], **kwargs)
    if digest != bound["sha256"]:
        raise ValueError("Selected frozen calibration SHA differs from its explicit binding")
    parameters = dict(freeze["parameters"])
    engine.r2._validate_parameters(parameters)
    if parameters.get("candidate") != "N1":
        raise ValueError("This production proposal only implements the explicit frozen native N1 choice")
    return parameters, digest


def prepare_n1_response(frame, path_folder, parts, swi, scope, frozen_calibration,
                        *, modules, selected_calibration, asset_reader=None):
    """Retain native pooled optics/titration; omit unused J/linear sensitivity."""
    import numpy as np
    if scope != "frozen_evaluation" or frozen_calibration is None:
        raise ValueError("Production requires the actual chosen freeze before pixels")
    parameters, digest = _selected_n1(modules, selected_calibration, asset_reader)
    r1, r2 = modules["engine"].r1, modules["engine"].r2
    kwargs = {} if asset_reader is None else {"asset_reader": asset_reader}
    original, original_digest = r2.verify_original_round1_freeze(parts["r1_freeze"]["path"], **kwargs)
    # Same pooled-folder equality and source checks as native R2 preparation.
    from pathlib import Path
    if Path(original["path_folder"]).resolve() != Path(path_folder).resolve():
        raise ValueError("Production must reuse the unchanged round-1 pooled colour paths")
    prepare = _clone(r1.prepare_native_response,
                     verify_frozen_calibration=r2.verify_original_round1_freeze)
    response = prepare(frame, path_folder, flash_kind="titration",
                       frozen_calibration_path=parts["r1_freeze"]["path"]
                       if r1.normalize_run(frame.run_id) in r2.HELDOUT_RUNS else None,
                       **kwargs)
    measured_swi = np.full(frame.domain.shape, np.nan, np.float64)
    for label, value in swi.items():
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Swi must be a measured sand property in [0,1]")
        measured_swi[frame.facies == int(label)] = value
    response.update(swi=measured_swi, frozen_parameters=parameters,
                    round1_freeze_sha256=original_digest,
                    selected_calibration_sha256=digest,
                    cross_host_source_equivalence=original["cross_host_source_equivalence"])
    if asset_reader is not None:
        response["production_io_source_provenance"] = original["production_io_source_provenance"]
    return response


def load_response(context, case_id, *, selected_calibration):
    """Run unchanged selected load/admission code, with explicit N1 preparation.

    The original loader still produces its descriptive floor metadata. No R3
    summary, extended estimate geometry, own-colour, column or E path is called.
    Removing further loader-only work requires a separate reviewed source diff.
    """
    if case_id not in context.cases:
        raise ValueError("Case is outside the locked native diagnostic")
    if context.boundary_guard is not None:
        context.boundary_guard()
    adapter, source = context.modules["adapter"], context.modules["source"]
    asset_reader = context.asset_reader
    def prepare(*args, **kwargs):
        return prepare_n1_response(*args, modules=context.modules,
                                   selected_calibration=selected_calibration, **kwargs)
    namespace = {"_prepare_response": prepare}
    if asset_reader is not None:
        from darsia.fluidflower.production_native import _Proxy
        clearwater = _clone(source.registered_clearwater, read_bound_json=context.reader.json)
        namespace["source"] = _Proxy(source, registered_clearwater=clearwater)
    selected = _clone(adapter.load_selected, **namespace)
    def loader(**kwargs):
        return selected(**kwargs, **({} if asset_reader is None else {"asset_reader": asset_reader}))
    session = dict(context.session, loader=loader)
    load = _clone(context.load_entry, SESSION=session)
    result = load(context.cases[case_id]["entry"], scope="frozen_evaluation",
                  frozen_calibration=context.configuration["actual_R3_freeze"]["path"])
    if context.boundary_guard is not None:
        context.boundary_guard()
    return result


def _main_partition(response):
    """Compatibility wrapper for the canonical native N1 volume partition."""
    from .n1_arithmetic import main_partition
    return main_partition(response)


def evaluate_main(response, anchor, *, modules, selected_calibration, domain_masks,
                  kept_fields=(), asset_reader=None):
    """One selected calibration, exact old operation order; no E or second choice.

    The reference is the original final-calibrated native field integration.
    R3's quantized histogram/dot scalars are a separate informative comparison,
    never recomputed here and never substituted for the retained field products.
    Arrays used for native integration keep their historical dtype and full-grid
    zero-mask/axis order. No chunk reduction, new precision or backend is used.
    """
    import numpy as np
    if set(kept_fields) - FIELDS or set(domain_masks) != {"full_rig", "box1", "box2"}:
        raise ValueError("Supported retained fields and all three frozen domain masks are required")
    parameters, digest = _selected_n1(modules, selected_calibration, asset_reader)
    if response.get("frozen_parameters") != parameters:
        raise ValueError("Response differs from the one frozen production calibration")
    engine, frame = modules["engine"], response["frame"]
    for name, mask in domain_masks.items():
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != frame.domain.shape:
            raise ValueError("Frozen domain mask differs from the native full-resolution grid")
        if name == "full_rig" and not np.array_equal(mask, frame.domain):
            raise ValueError("Full-rig domain must be the exact original frozen phase domain")
    analysis, physics = response["analysis"], response["physics"]
    partition = _main_partition(response)
    factor = engine.r2.dilution_factor(frame.elapsed_h, partition["affected_pore_volume_m3"],
                                      anchor, parameters.get("beta", 0.))
    # Literal N1 branch of the native R2 map calculation, with the unused
    # second/linear response omitted rather than changed.
    fraction = (np.clip(response["prepared"].aqueous_relative.astype(np.float64)
                        * parameters["gain"], 0, 1)*factor).astype(np.float32)
    aqueous = np.where(partition["aq_known"], fraction, 0).astype(np.float32)
    gas = np.zeros(frame.domain.shape, np.float32)
    gas[partition["gas_known"]] = (1-response["swi"][partition["gas_known"]]).astype(np.float32)
    aqueous[partition["gas_known"]] = 1.
    # CO2MassAnalysis.mass_analysis -> full_like casts each field separately to
    # aqueous.dtype. In particular total is cast before its original integration.
    raw_gas = physics.density_gaseous_co2 * gas
    raw_aqueous = physics.solubility_co2 * aqueous * np.clip(1-gas, 0, None)
    mass_gas = np.asarray(raw_gas, dtype=aqueous.dtype)
    mass_aqueous = np.asarray(raw_aqueous, dtype=aqueous.dtype)
    eligible = frame.mass_eligible_mask & frame.known_physical_domain
    def integrate(value, mask):
        return float(analysis.geometry.integrate(np.where(mask, value, 0.)))*1000
    total_mass_field = np.asarray(raw_gas + raw_aqueous, dtype=aqueous.dtype)
    del raw_gas, raw_aqueous
    def masses(mask):
        return dict(total_mass_g=integrate(total_mass_field, mask),
                    aqueous_mass_g=integrate(mass_aqueous, mask),
                    gas_mass_g=integrate(mass_gas, mask))
    domains = {}
    for name, mask in domain_masks.items():
        domains[name] = masses(eligible & mask)
    excluded, covered = {}, np.zeros(frame.domain.shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        covered |= mask
        excluded[name] = masses(mask)
    excluded["other_not_mass_eligible"] = masses(frame.domain & ~frame.mass_eligible_mask & ~covered)
    fields = {}
    if "c_aq_g_l" in kept_fields or "unquantified_phase_mask" in kept_fields or "s_g" in kept_fields:
        unknown = frame.domain & np.isin(frame.phase_labels, (1, 2)) & ~(
            partition["aq_known"] | partition["gas_known"])
        if "c_aq_g_l" in kept_fields:
            concentration = (aqueous*physics.solubility_co2).astype(np.float32)
            concentration[unknown] = np.nan
            fields["c_aq_g_l"] = concentration
        if "unquantified_phase_mask" in kept_fields: fields["unquantified_phase_mask"] = unknown
        if "s_g" in kept_fields:
            gas[unknown & (frame.phase_labels == 2)] = np.nan
            fields["s_g"] = gas
    if "c_aq_relative" in kept_fields: fields["c_aq_relative"] = aqueous
    if "phase_labels" in kept_fields: fields["phase_labels"] = frame.phase_labels.copy()
    if "main_mass_eligible_mask" in kept_fields: fields["main_mass_eligible_mask"] = eligible
    if "aqueous_mass_density_g_l" in kept_fields: fields["aqueous_mass_density_g_l"] = mass_aqueous
    if "gas_mass_density_g_l" in kept_fields: fields["gas_mass_density_g_l"] = mass_gas
    if "mass_density_g_l" in kept_fields:
        fields["mass_density_g_l"] = total_mass_field
    return dict(contract="darsia-production-retained-main-draft-v1", parameters=parameters,
        calibration_sha256=digest, scalar_reference="native_final_calibration_fields", run=frame.run_id,
        frame_index=frame.frame_index, elapsed_h=frame.elapsed_h, dilution_factor=factor,
        **domains["full_rig"], domains=domains, excluded_mass=excluded,
        affected_pore_volume_m3=partition["affected_pore_volume_m3"], fields=fields,
        scientific_eligibility=False, production_permission_granted=False)


def run_case(context, case_id, *, selected_calibration, anchor, domain_masks, kept_fields):
    """One guarded diagnostic load and retained-product calculation, never R3."""
    loaded = load_response(context, case_id, selected_calibration=selected_calibration)
    if context.boundary_guard is not None:
        context.boundary_guard()
    result = evaluate_main(loaded["response"], anchor, modules=context.modules,
        selected_calibration=selected_calibration, domain_masks=domain_masks,
        kept_fields=kept_fields, asset_reader=context.asset_reader)
    if context.boundary_guard is not None:
        context.boundary_guard()
    return result


def main_products(result):
    """Explicit §3/§6 export view; keep raw native fields unchanged for parity.

    Native mass images are intensive g/L values, not per-cell grams. Their
    original geometry integration times 1000 gives grams. Transport must use
    the separately recorded native pore-volume weight in litres per pixel;
    this projection performs no rescaling or conservative-grid conversion.
    """
    import numpy as np
    fields = result["fields"]
    required = {"c_aq_g_l", "s_g", "phase_labels", "main_mass_eligible_mask",
                "mass_density_g_l", "aqueous_mass_density_g_l", "gas_mass_density_g_l"}
    if not required.issubset(fields):
        raise ValueError("Every retained native §3 product and the main mask must be present")
    mask = fields["main_mass_eligible_mask"]
    if not isinstance(mask, np.ndarray) or mask.dtype != bool:
        raise ValueError("Main product mask must retain the original boolean grid")
    products = dict(concentration_aqueous=fields["c_aq_g_l"], saturation_free_gas=fields["s_g"],
                    phase_labels=fields["phase_labels"])
    for name, native in (("mass_total", "mass_density_g_l"),
                         ("mass_aqueous", "aqueous_mass_density_g_l"),
                         ("mass_free_gas", "gas_mass_density_g_l")):
        value = fields[native]
        if value.dtype != np.float32 or value.shape != mask.shape:
            raise ValueError("Retained mass map must keep native float32 dtype and exact grid")
        products[name] = np.where(mask, value, 0.)
    return dict(products=products, units={"concentration_aqueous": "g/L", "saturation_free_gas": "1",
        "mass_total": "g/L", "mass_aqueous": "g/L", "mass_free_gas": "g/L", "phase_labels": "category"},
        required_transport_weight_units="L pore volume per pixel",
        native_mass_scalar_units="g", main_mask_applied=True,
        raw_native_fields_retained=True, scientific_eligibility=False, production_permission_granted=False)

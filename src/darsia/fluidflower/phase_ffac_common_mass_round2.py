"""Pre-registered second common FF_AC calibration on immutable native inputs.

DarSIA supplies the colour paths, flashes, thermodynamics and mass integration.
Only the aqueous label-1 response and post-injection dilution are fitted.
Reduced, source-bound frame histograms let every LORO fold reuse the same RAW
interpretation. Unknown geometry enters a finite capacity bound, never a map.
"""
from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
from types import FunctionType
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

import darsia
from . import phase_ffac_common_mass as r1
from .concentration_joint import DilutionAnchor, JointConcentrationParameters, concentration_field, dilution_factor
from .tools import phase_mass_joint_optics as joint_optics
from .phase_ffac_common_mass_round2_backend import reduce_weighted_codes

CONTRACT = "ffac-common-mass-round2-v1"
FREEZE_CONTRACT = "ffac-common-mass-round2-frozen-v1"
FIT_RUNS = tuple(run for run in r1.CALIBRATION_RUNS if run != "ac38")
HELDOUT_RUNS = r1.HELDOUT_RUNS
DESCRIPTIVE_RUNS = ("ac38",)
CANDIDATES = ("N1", "N2", "N3", "J")
SUPPORTED_CANDIDATES = CANDIDATES
N3_STATUS = "included: existing native TitrationFlash aqueous_blend"
INTERVALS = 4096
INJECTION_TARGETS = (.5, .75, 1., 1.25, 1.75, 2., 2.25)
POST_TARGETS = (2.75, 3., 4., 5., 6., 8., 10., 12., 24.)
ANCHOR_TARGET = 2.247
MAX_DEPTH_M = .0155
MAX_POROSITY = .45
ROUND1_FREEZE_SHA256 = "ce4b402784f6d45662b807831a1c54e791d1438d0442fe84ffd79034f269c986"
_VERIFIED_SOURCE_FILES = {}
_VERIFIED_ORIGINAL_FREEZES = {}


def candidate_availability():
    native_blend = "aqueous_blend" in inspect.signature(darsia.TitrationFlash).parameters
    return {name: {"available": name != "N3" or native_blend,
        "reason": "existing native TitrationFlash aqueous_blend" if name == "N3" and native_blend
        else "native blend unavailable; no substitute model added" if name == "N3" else "specified existing response"}
        for name in CANDIDATES}


def verify_round2_freeze(path, *, asset_reader=None):
    logical_path = Path(path)
    path = logical_path if asset_reader is None else asset_reader.resolve_path(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("contract") != FREEZE_CONTRACT or value.get("frozen") is not True:
        raise ValueError("Held-out scientific computation requires the round-2 chosen calibration freeze")
    if (set(value.get("calibration_runs", ())) != set(FIT_RUNS)
            or set(value.get("heldout_runs", ())) != set(HELDOUT_RUNS)
            or set(value.get("descriptive_runs", ())) != {"ac38"}):
        raise ValueError("Round-2 freeze has a different pre-registered cohort")
    if value.get("parameters", {}).get("candidate") not in CANDIDATES or not value.get("frozen_utc"):
        raise ValueError("Missing frozen chosen response")
    for name, item in value.get("calibration_files", {}).items():
        file = item.get("path", name) if isinstance(item, dict) else name
        expected = item.get("sha256") if isinstance(item, dict) else item
        file = Path(file)
        if not file.is_absolute():
            file = logical_path.parent / file
        if asset_reader is not None:
            file = asset_reader.resolve_path(file, expected)
        if r1.sha256(file) != expected:
            raise ValueError(f"Frozen calibration file changed: {file}")
    if not value.get("calibration_files"):
        raise ValueError("A round-2 freeze must bind its actual calibration files")
    return value, r1.sha256(path)


def _verify_file(path,expected, *, asset_reader=None, source_file=False):
    path = (Path(path) if asset_reader is None else
        asset_reader.source_path(path, expected) if source_file else
        asset_reader.resolve_path(path, expected))
    stat = path.stat()
    key = (str(path.resolve()),stat.st_size,stat.st_mtime_ns,expected)
    if key not in _VERIFIED_SOURCE_FILES:
        if r1.sha256(path) != expected:
            raise ValueError(f"Frozen source file changed: {path}")
        _VERIFIED_SOURCE_FILES[key] = True


def verify_original_round1_freeze(path, *, expected_sha256=ROUND1_FREEZE_SHA256, asset_reader=None):
    """Verify immutable R1 bytes at their equivalent current-host locations.

    The original record contains its producer's absolute implementation paths.
    Scientific files keep their original NAS paths. Implementation equivalence
    is established by exact source hashes of the currently imported classes,
    without rewriting the record, creating old folders or monkeypatching R1.
    """
    path = Path(path) if asset_reader is None else asset_reader.resolve_path(path, expected_sha256)
    stat = path.stat()
    key = (str(path.resolve()),stat.st_size,stat.st_mtime_ns,expected_sha256,
        None if asset_reader is None else id(asset_reader))
    cached = _VERIFIED_ORIGINAL_FREEZES.get(key)
    if cached is not None:
        record,actual,components = cached
        # Reuse parsed receipt and verified SHA, but still invalidate whenever
        # any bound component's path/size/mtime changes during the session.
        for file,expected,source_file in components:
            _verify_file(file,expected,asset_reader=asset_reader,source_file=source_file)
        return record,actual
    actual = r1.sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Round-1 pooled colour-path freeze SHA differs from the approved source")
    freeze = json.loads(path.read_text(encoding="utf-8-sig"))
    if freeze.get("contract") != r1.CONTRACT or freeze.get("frozen") is not True:
        raise ValueError("Original colour paths require the published round-1 freeze")
    r1.validate_cohort(freeze["calibration_runs"],freeze["heldout_runs"])
    if freeze.get("map_mass_rescaling") is not False or freeze.get("overflow_in_objective") is not False:
        raise ValueError("Original freeze violates the fixed no-rescaling/compare-only contract")
    for file,expected in freeze["calibration_files"].items():
        _verify_file(file,expected,asset_reader=asset_reader)
    _verify_file(r1.__file__,freeze["common_engine_sha256"],asset_reader=asset_reader,source_file=True)
    objects = (darsia.ColorPath,darsia.ColorPathInterpolation,darsia.ConcentrationAnalysis,
        darsia.PWTransformation,darsia.TitrationFlash,darsia.SimpleFlash,darsia.CO2MassAnalysis,
        darsia.ExtrudedPorousGeometry,r1.NativeConcentrationAnalysis)
    current_by_hash = {}
    components = [(file,expected,False) for file,expected in freeze["calibration_files"].items()]+[(r1.__file__,freeze["common_engine_sha256"],True)]
    for obj in objects:
        source = str(Path(inspect.getfile(obj)).resolve())
        expected = freeze["native_implementations"].get(obj.__name__)
        if expected is None:
            raise ValueError("Original native implementation inventory is incomplete")
        _verify_file(source,expected,asset_reader=asset_reader,source_file=True)
        components.append((source,expected,True))
        current_by_hash.setdefault(expected,[]).append(source if asset_reader is None
            else str(asset_reader.source_path(source, expected)))
    equivalence = []
    for original,expected in freeze["native_implementation_files"].items():
        sources = current_by_hash.get(expected)
        if not sources:
            raise ValueError("Frozen native implementation has no byte-identical currently imported source")
        equivalence.append(dict(original_path=original,current_equivalent_paths=sorted(set(sources)),
            sha256=expected,exact_source_bytes=True))
    record = dict(freeze,cross_host_source_equivalence=dict(original_freeze_sha256=actual,
        original_freeze_rewritten=False,original_native_bytecode_changed=False,
        common_engine_current_path=str(Path(r1.__file__).resolve()),
        common_engine_sha256=freeze["common_engine_sha256"],native_implementation_files=equivalence))
    if asset_reader is not None:
        record["cross_host_source_equivalence"]["common_engine_current_path"] = str(
            asset_reader.source_path(r1.__file__, freeze["common_engine_sha256"]))
        record["production_io_source_provenance"] = dict(
            execution_commit=asset_reader.code_commit,
            current_common_engine=dict(path=str(Path(r1.__file__).resolve()), sha256=r1.sha256(r1.__file__)),
            original_calibration_source_sha256=freeze["common_engine_sha256"],
            frozen_evidence_path=str(asset_reader.source_path(r1.__file__, freeze["common_engine_sha256"])),
            source_bindings=asset_reader.source_bindings,
            qualification_state=asset_reader.source_qualification_state,
            frozen_evidence_is_not_current_executed_source=True)
    _VERIFIED_ORIGINAL_FREEZES[key] = (record,actual,tuple(components))
    return record,actual


def _scientific_joint_image(rgb, shape=None):
    """Validation adapter only: the existing J camera-linear math stays exact.

    Scientific camera RGB may exceed the photographic unit interval. No clipping,
    gamma, response gain, or per-image rescaling is introduced by this adapter.
    The original reference's <.995 photometric-reference gate stays unchanged.
    """
    value = np.asarray(rgb)
    if (value.dtype != np.float32 or value.ndim != 3 or value.shape[-1] != 3
            or (shape is not None and value.shape != shape) or not np.isfinite(value).all()
            or np.any(value < 0)):
        raise ValueError("J requires finite nonnegative scientific float32 HWC camera RGB")
    return value


def _joint_signal(frame, support):
    namespace = dict(joint_optics.__dict__, _image=_scientific_joint_image)
    build = FunctionType(joint_optics.build_joint_reference.__code__, namespace,
        joint_optics.build_joint_reference.__name__, joint_optics.build_joint_reference.__defaults__)
    build.__kwdefaults__ = joint_optics.build_joint_reference.__kwdefaults__
    evaluate = FunctionType(joint_optics.joint_optical_signals.__code__, namespace,
        joint_optics.joint_optical_signals.__name__, joint_optics.joint_optical_signals.__defaults__)
    # Zero is an invalid container outside measured support, not learned colour.
    baseline = np.where(support[..., None], frame.baseline_rgb, 0).astype(np.float32)
    clear = np.where(support[..., None], frame.clear_water_rgb, 0).astype(np.float32)
    current = np.where(support[..., None], frame.current_rgb, baseline).astype(np.float32)
    reference = build(baseline, clear, frame.facies, domain_mask=support, rgb_encoding="camera_linear")
    signal = evaluate(current, reference)["joint_chromatic_od"]
    return signal, {**reference.diagnostics,
        "scientific_validation_adapter": "finite positive camera-linear float; no unit-range clipping",
        "original_model_source_sha256": r1.sha256(joint_optics.__file__),
        "original_functions_used": ["build_joint_reference", "joint_optical_signals"],
        "original_reference_photometric_gate_unchanged": True}


def prepare_round2_response(frame, path_folder, *, swi_by_facies,
        r1_frozen_calibration_path, frozen_calibration=None, asset_reader=None):
    """Interpret one full-resolution frame once, before any candidate fitting."""
    run = r1.normalize_run(frame.run_id)
    freeze = None
    if run not in FIT_RUNS:
        if frozen_calibration is None:
            raise ValueError("Final held-out runs and AC38 stay untouched until the chosen R2 freeze")
        freeze, freeze_sha = (verify_round2_freeze(frozen_calibration) if asset_reader is None
            else verify_round2_freeze(frozen_calibration, asset_reader=asset_reader))
    original, original_sha = verify_original_round1_freeze(r1_frozen_calibration_path,
        **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    if Path(original["path_folder"]).resolve() != Path(path_folder).resolve():
        raise ValueError("Round 2 must reuse the unchanged round-1 pooled colour paths")
    native_namespace = dict(r1.__dict__,verify_frozen_calibration=verify_original_round1_freeze)
    native_prepare = FunctionType(r1.prepare_native_response.__code__,native_namespace,
        r1.prepare_native_response.__name__,r1.prepare_native_response.__defaults__)
    native_prepare.__kwdefaults__ = r1.prepare_native_response.__kwdefaults__
    response = native_prepare(frame, path_folder, flash_kind="titration",
        frozen_calibration_path=r1_frozen_calibration_path if run in HELDOUT_RUNS else None,
        **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    analysis = response["analysis"]
    signal = analysis._scalar(response["native_signal"], frame.elapsed_h)
    linear_flash = darsia.SimpleFlash(**response["metadata"]["flash_bounds"])
    linear, _ = linear_flash(signal)
    swi = np.full(frame.domain.shape, np.nan, np.float64)
    for label, value in swi_by_facies.items():
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Swi must be a measured sand property in [0,1]")
        swi[frame.facies == int(label)] = value
    joint_signal, joint_metadata = _joint_signal(frame,
        frame.known_physical_domain & frame.supported_mask & frame.clear_water_supported_mask)
    response.update(linear_relative=np.asarray(linear.img, np.float32), linear_flash=linear_flash,
        joint_signal=joint_signal, joint_metadata=joint_metadata, swi=swi,
        round1_freeze_sha256=original_sha, frozen_parameters=None if freeze is None else freeze["parameters"],
        cross_host_source_equivalence=original["cross_host_source_equivalence"],
        round2_freeze_sha256=None if freeze is None else freeze_sha)
    if asset_reader is not None:
        response["production_io_source_provenance"] = original["production_io_source_provenance"]
    return response


def _physical_partition(response, *, max_depth_m=MAX_DEPTH_M, max_porosity=MAX_POROSITY):
    if max_depth_m != MAX_DEPTH_M or max_porosity != MAX_POROSITY:
        raise ValueError("Capacity-only limits are fixed by the measured rig/facies maxima")
    frame, physics = response["frame"], response["physics"]
    shape = frame.domain.shape
    positive = frame.domain & np.isin(frame.phase_labels, (1, 2))
    known = frame.known_physical_domain
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), shape)
    depth = np.broadcast_to(np.asarray(frame.depth_m, np.float64), shape)
    phi = np.broadcast_to(np.asarray(frame.porosity, np.float64), shape)
    volume = np.where(known, area*depth*phi, 0.)
    aq_known = known & response["quantifiable"] & (frame.phase_labels == 1)
    gas_known = known & np.isfinite(response["swi"]) & (frame.phase_labels == 2)
    eligible = frame.mass_eligible_mask & frame.domain
    main_aq, main_gas = eligible & aq_known, eligible & gas_known
    missing = positive & ~(main_aq | main_gas)
    # These maxima provide a capacity only. The measured-volume array stays zero
    # on missing geometry and is never replaced by the maximum-volume array.
    bound_volume = np.where(known, volume, area*max_depth_m*max_porosity)
    sol = np.asarray(physics.solubility_co2, np.float64)
    density = np.asarray(physics.density_gaseous_co2, np.float64)
    if (not np.isfinite(sol[positive]).all() or not np.isfinite(density[positive]).all()
            or np.any(sol[positive] < 0) or np.any(density[positive] < 0)):
        raise ValueError("Native thermodynamic capacity must be finite across the locked phase domain")
    maximum = np.where(frame.phase_labels == 2, np.maximum(sol, density), sol)
    parts, covered = {}, np.zeros(shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        selection = missing & mask
        covered |= mask
        parts[name] = {"capacity_upper_g": float(np.sum(bound_volume[selection]*maximum[selection]))*1000,
            "phase_pixels": int(selection.sum()), "unknown_geometry_phase_pixels": int((selection & ~known).sum())}
    masks = {"other_not_mass_eligible": missing & ~eligible & ~covered,
        "unquantified_main": missing & eligible}
    for name, selection in masks.items():
        parts[name] = {"capacity_upper_g": float(np.sum(bound_volume[selection]*maximum[selection]))*1000,
            "phase_pixels": int(selection.sum()), "unknown_geometry_phase_pixels": int((selection & ~known).sum())}
    upper = sum(item["capacity_upper_g"] for item in parts.values())
    expected = float(np.sum(bound_volume[missing]*maximum[missing]))*1000
    if not np.isclose(upper, expected, rtol=1e-12, atol=1e-12):
        raise ValueError("Missing capacity masks overlap or omit positive phase support")
    return dict(volume=volume, bound_volume=bound_volume, main_aq=main_aq, main_gas=main_gas,
        aq_known=aq_known, gas_known=gas_known, missing=missing, solubility=sol, density=density,
        parts=parts, missing_capacity_upper_g=upper,
        affected_pore_volume_m3=float(np.sum(volume[positive])),
        affected_unknown_geometry_pixels=int((positive & ~known).sum()),
        affected_unknown_geometry_area_cm2=float(np.sum(area[positive & ~known]))*1e4)


def _histogram_pair(response, partition, backend, mask=None):
    mask = partition["main_aq"] if mask is None else mask
    first = np.clip(response["prepared"].aqueous_relative[mask], 0, 1)
    second = np.clip(response["linear_relative"][mask], 0, 1)
    code = np.rint(first*INTERVALS).astype(np.int64)*(INTERVALS+1) + np.rint(second*INTERVALS).astype(np.int64)
    weights = partition["volume"][mask]*partition["solubility"][mask]*1000
    unique, sums, audit = reduce_weighted_codes(code, weights, backend=backend)
    count_codes, counts, count_audit = reduce_weighted_codes(code, np.ones(len(code),np.float64), backend=backend)
    if not np.array_equal(unique,count_codes):
        raise ValueError("Native weighted/counted histogram geometry differs")
    audit = dict(audit, pixel_count_backend_audit=count_audit)
    return dict(native_titration_response=(unique//(INTERVALS+1)/INTERVALS).tolist(),
        native_linear_response=(unique%(INTERVALS+1)/INTERVALS).tolist(),
        native_aqueous_capacity_g=sums.tolist(),native_aqueous_pixel_counts=counts.astype(np.int64).tolist()), audit


def summarize_round2_frame(response, *, no_outflow_evidence, light_epoch,
        max_depth_m=MAX_DEPTH_M, max_porosity=MAX_POROSITY, backend="auto",descriptive_sensitivity=False,
        postcheck_sensitivity=None):
    """Reduce a frame without fitting mass or using final-held-out information."""
    frame = response["frame"]
    postcheck_allowed = False
    if postcheck_sensitivity is not None:
        if not isinstance(postcheck_sensitivity,dict) or not {"path","sha256"} <= set(postcheck_sensitivity):
            raise ValueError("Held-out sensitivity needs a source-bound final-check receipt")
        _verify_file(postcheck_sensitivity["path"],postcheck_sensitivity["sha256"])
        check = json.loads(Path(postcheck_sensitivity["path"]).read_text(encoding="utf-8-sig"))
        postcheck_allowed = (check.get("full_production_allowed") is True and check.get("final") is True
            and check.get("freeze",{}).get("sha256") == response.get("round2_freeze_sha256")
            and response.get("round2_freeze_sha256") is not None)
        if not postcheck_allowed:
            raise ValueError("Final-check evidence does not authorize frozen held-out sensitivities")
    if frame.run_id not in FIT_RUNS and not (descriptive_sensitivity and frame.run_id == "ac38" and response.get("round2_freeze_sha256")) and not (frame.run_id in HELDOUT_RUNS and postcheck_allowed):
        raise ValueError("Only the 26 fit runs may enter cached selection summaries")
    partition = _physical_partition(response, max_depth_m=max_depth_m, max_porosity=max_porosity)
    native_hist, native_audit = _histogram_pair(response, partition, backend)
    mask = partition["main_aq"]
    # Signed J signals are retained without a fixed upper tail truncation.
    codes = np.rint(response["joint_signal"][mask].astype(np.float64)*INTERVALS).astype(np.int64)
    weights = partition["volume"][mask]*partition["solubility"][mask]*1000
    unique, sums, joint_audit = reduce_weighted_codes(codes, weights, backend=backend)
    count_codes, counts, count_audit = reduce_weighted_codes(codes,np.ones(len(codes),np.float64),backend=backend)
    if not np.array_equal(unique,count_codes):
        raise ValueError("J weighted/counted histogram geometry differs")
    joint_audit = dict(joint_audit,pixel_count_backend_audit=count_audit)
    gas = partition["main_gas"]
    swi = response["swi"][gas]
    gas_mass = float(np.sum(partition["volume"][gas]*partition["density"][gas]*(1-swi)))*1000
    gas_water_mass = float(np.sum(partition["volume"][gas]*partition["solubility"][gas]*swi))*1000
    l1 = np.zeros(frame.domain.shape,bool) if frame.l1_neighbourhood_mask is None else gas & frame.l1_neighbourhood_mask
    l1_swi = response["swi"][l1]
    l1_aq = l1 & response["quantifiable"]
    l1_native, l1_native_audit = _histogram_pair(response,partition,backend,mask=l1_aq)
    l1_codes = np.rint(response["joint_signal"][l1_aq].astype(np.float64)*INTERVALS).astype(np.int64)
    l1_weights = partition["volume"][l1_aq]*partition["solubility"][l1_aq]*1000
    l1_unique, l1_sums, l1_joint_audit = reduce_weighted_codes(l1_codes,l1_weights,backend=backend)
    l1_unknown = l1 & ~response["quantifiable"]
    l1_missing = (np.zeros(frame.domain.shape,bool) if frame.l1_neighbourhood_mask is None
        else partition["missing"] & frame.l1_neighbourhood_mask & (frame.phase_labels == 2))
    l1_cache = dict(gas_mass_g=float(np.sum(partition["volume"][l1]*partition["density"][l1]*(1-l1_swi)))*1000,
        gas_water_mass_g=float(np.sum(partition["volume"][l1]*partition["solubility"][l1]*l1_swi))*1000,
        aqueous_equilibrium_mass_g=float(np.sum(partition["volume"][l1]*partition["solubility"][l1]))*1000,
        phase_pixels=int(l1.sum()), source_bound=frame.provenance.get("l1_position_source_bound") is True,
        verified=frame.provenance.get("l1_position_verified") is True,
        status=frame.provenance.get("l1_position_status"),source_sha256=frame.provenance.get("l1_position_source_sha256"),
        **l1_native,joint_signal=(l1_unique/INTERVALS).tolist(),joint_aqueous_capacity_g=l1_sums.tolist(),
        newly_unquantified_aqueous_capacity_g=float(np.sum(partition["volume"][l1_unknown]*partition["solubility"][l1_unknown]))*1000,
        missing_gas_to_aqueous_capacity_delta_g=float(np.sum(partition["bound_volume"][l1_missing]*(partition["solubility"][l1_missing]-np.maximum(partition["solubility"][l1_missing],partition["density"][l1_missing]))))*1000,
        convention="diagnostic reclassification to chosen aqueous response plus dilution; original labels unchanged",
        saturated_undiluted_replacement_is_separate_upper_assumption=True,
        histogram_backend_audit={"native":l1_native_audit,"joint":l1_joint_audit})
    dose = frame.nominal_injected_mass_g
    return dict(contract=CONTRACT, run=frame.run_id, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        nominal_injected_mass_g=dose, injected_mass_g=dose, light_epoch=light_epoch,
        no_outflow_evidence=dict(no_outflow_evidence), **native_hist,
        joint_signal=(unique/INTERVALS).tolist(), joint_aqueous_capacity_g=sums.tolist(),
        joint_aqueous_pixel_counts=counts.astype(np.int64).tolist(),l1_sensitivity=l1_cache,
        gas_mass_g=gas_mass, gas_water_mass_g=gas_water_mass,
        gas_pore_volume_m3=float(np.sum(partition["volume"][gas]*(1-swi))),
        gas_capacity_sg1_g=float(np.sum(partition["volume"][gas]*partition["density"][gas]))*1000,
        gas_water_capacity_sg0_g=float(np.sum(partition["volume"][gas]*partition["solubility"][gas]))*1000,
        affected_pore_volume_m3=partition["affected_pore_volume_m3"],
        affected_volume_includes_excluded_known_physical=True, affected_volume_imputes_missing_geometry=False,
        affected_unknown_geometry_pixels=partition["affected_unknown_geometry_pixels"],
        affected_unknown_geometry_area_cm2=partition["affected_unknown_geometry_area_cm2"],
        missing_capacity_upper_g=partition["missing_capacity_upper_g"],
        mass_balance_missing_capacity_upper_g=partition["missing_capacity_upper_g"],
        missing_capacity_parts=partition["parts"], capacity_only_max_depth_m=max_depth_m,
        capacity_only_max_porosity=max_porosity, main_geometry_imputed=False,
        unknown_phase_pixels=int(partition["missing"].sum()), aqueous_support_pixels=int(mask.sum()),
        gas_support_pixels=int(gas.sum()), optical_path_not_required_for_fixed_physical_gas=True,
        native_flash_bounds=dict(response["metadata"]["flash_bounds"]),
        recipe=dict(frame.recipe), source_binding=dict(frame.provenance),
        round1_freeze_sha256=response["round1_freeze_sha256"], joint_reference=response["joint_metadata"],
        cross_host_source_equivalence=response.get("cross_host_source_equivalence"),
        histogram_intervals=INTERVALS, histogram_backend_audit={"native": native_audit, "joint": joint_audit},
        no_map_mass_rescaling=True, overflow_in_objective=False)


def _validate_parameters(parameters):
    name = parameters.get("candidate")
    if name not in CANDIDATES or not candidate_availability()[name]["available"]:
        raise ValueError("Unavailable or unknown pre-registered candidate")
    if not np.isfinite(parameters.get("beta", 0.)) or parameters.get("beta", 0.) < 0:
        raise ValueError("A shared nonnegative dilution exponent is required")
    if name == "J":
        JointConcentrationParameters(endpoint=parameters["endpoint"], amplitude=parameters["amplitude"],
            gamma=parameters["gamma"], beta=parameters.get("beta", 0.))
    elif not np.isfinite(parameters.get("gain", 0)) or parameters.get("gain", 0) <= 0:
        raise ValueError("A common positive aqueous gain is required")
    if name == "N3" and (not np.isfinite(parameters.get("blend", -1)) or not 0 <= parameters["blend"] <= 1):
        raise ValueError("Native aqueous blend must lie in [0,1]")
    if any(key in parameters for key in ("gas_scale", "gas_saturation", "gas_gain")):
        raise ValueError("The gas branch has no fitted parameter")


def _native_fraction(first, second, parameters):
    name = parameters["candidate"]
    response = first if name == "N1" else second if name == "N2" else (1-parameters["blend"])*first+parameters["blend"]*second
    return np.clip(response*parameters["gain"], 0, 1)


def _joint_fraction(signal, parameters):
    # Exactly the existing concentration_joint power formula, evaluated in
    # float64 during fitting; full-resolution maps call its actual function.
    return parameters["amplitude"]*np.clip(signal/parameters["endpoint"], 0, 1)**parameters["gamma"]


def interval_residual(mass, dose, upper):
    if not np.isfinite([mass, dose, upper]).all() or min(mass, upper) < 0 or dose <= 0:
        raise ValueError("A two-sided residual requires finite physical bounds and a positive dose")
    return float((mass-dose)/dose if mass > dose else (mass+upper-dose)/dose if mass+upper < dose else 0.)


def evaluate_round2_summary(summary, parameters, anchor, *, gas_sg1=False,
        port_as_aqueous=False,allow_provisional_l1_sensitivity=False):
    _validate_parameters(parameters)
    factor = dilution_factor(summary["elapsed_h"], summary["affected_pore_volume_m3"], anchor, parameters.get("beta", 0.))
    if parameters["candidate"] == "J":
        fractions = _joint_fraction(np.asarray(summary["joint_signal"], np.float64), parameters)
        weights = np.asarray(summary["joint_aqueous_capacity_g"], np.float64)
        pixel_counts = np.asarray(summary.get("joint_aqueous_pixel_counts", np.zeros(len(weights))),np.float64)
    else:
        fractions = _native_fraction(np.asarray(summary["native_titration_response"], np.float64),
            np.asarray(summary["native_linear_response"], np.float64), parameters)
        weights = np.asarray(summary["native_aqueous_capacity_g"], np.float64)
        pixel_counts = np.asarray(summary.get("native_aqueous_pixel_counts", np.zeros(len(weights))),np.float64)
    nongas = float(np.sum(weights*fractions))*factor
    if parameters["candidate"] == "J":
        signal = np.asarray(summary["joint_signal"], np.float64)
        lower = _joint_fraction(signal-.5/INTERVALS, parameters)
        upper = _joint_fraction(signal+.5/INTERVALS, parameters)
        quantization_bound = max(float(np.sum(weights*(fractions-lower))), float(np.sum(weights*(upper-fractions))))*factor
    else:
        first = np.asarray(summary["native_titration_response"],np.float64)
        second = np.asarray(summary["native_linear_response"],np.float64)
        lower = _native_fraction(np.clip(first-.5/INTERVALS,0,1),np.clip(second-.5/INTERVALS,0,1),parameters)
        upper = _native_fraction(np.clip(first+.5/INTERVALS,0,1),np.clip(second+.5/INTERVALS,0,1),parameters)
        quantization_bound = max(float(np.sum(weights*(fractions-lower))),float(np.sum(weights*(upper-fractions))))*factor
    gas_mass = float(summary["gas_capacity_sg1_g"] if gas_sg1 else summary["gas_mass_g"])
    residual_water = 0. if gas_sg1 else float(summary["gas_water_mass_g"])
    provisional = False
    l1_capacity_adjustment = 0.
    if port_as_aqueous and summary["elapsed_h"] >= 4:
        l1 = summary.get("l1_sensitivity",{})
        provisional = (allow_provisional_l1_sensitivity and l1.get("source_bound") is True
            and l1.get("status") == "protocol_proxy_unverified" and bool(l1.get("source_sha256")))
        if not (l1.get("verified") is True or provisional):
            raise ValueError("Cached L1 sensitivity needs its source-bound injector/protocol mask")
        if gas_sg1:
            raise ValueError("L1 and s_g=1 are distinct pre-declared sensitivities")
        gas_mass -= float(l1["gas_mass_g"])
        residual_water -= float(l1["gas_water_mass_g"])
        if parameters["candidate"] == "J":
            moved_fraction = _joint_fraction(np.asarray(l1["joint_signal"],np.float64),parameters)
            moved_weight = np.asarray(l1["joint_aqueous_capacity_g"],np.float64)
        else:
            moved_fraction = _native_fraction(np.asarray(l1["native_titration_response"],np.float64),
                np.asarray(l1["native_linear_response"],np.float64),parameters)
            moved_weight = np.asarray(l1["native_aqueous_capacity_g"],np.float64)
        nongas += float(np.sum(moved_fraction*moved_weight))*factor
        l1_capacity_adjustment = float(l1["newly_unquantified_aqueous_capacity_g"])+float(l1["missing_gas_to_aqueous_capacity_delta_g"])
    total = nongas+gas_mass+residual_water
    dose = summary.get("injected_mass_g", summary.get("nominal_injected_mass_g"))
    upper = max(0.,float(summary["missing_capacity_upper_g"])+l1_capacity_adjustment)
    residual = interval_residual(total, dose, upper) if dose is not None and dose > 0 else None
    return {key: summary[key] for key in ("run", "frame_index", "elapsed_h", "light_epoch", "source_binding", "missing_capacity_parts") if key in summary} | dict(
        contract=CONTRACT, parameters=dict(parameters), total_mass_g=total, aqueous_mass_g=nongas+residual_water,
        aqueous_nongas_mass_g=nongas, gas_mass_g=gas_mass, gas_water_mass_g=residual_water,
        gas_mass_share=gas_mass/total if total > 0 else 0., injected_mass_g=dose,
        nominal_injected_mass_g=dose, missing_capacity_upper_g=upper, mass_balance_missing_capacity_upper_g=upper,
        ratio=total/dose if dose is not None and dose > 0 else None,
        delta=None if residual is None else abs(residual), signed_interval_residual=residual,
        U_over_d=upper/dose if dose is not None and dose > 0 else None, dilution_factor=factor,
        summary_quantization_bound_g=quantization_bound,
        aqueous_clip_capacity_share=float(np.sum(weights[fractions == 1])/np.sum(weights)) if np.sum(weights) > 0 else 0.,
        aqueous_clip_pixel_share=float(np.sum(pixel_counts[fractions == 1])/np.sum(pixel_counts)) if np.sum(pixel_counts) > 0 else 0.,
        provisional_l1_location=provisional,
        l1_capacity_adjustment_g=l1_capacity_adjustment,
        no_map_mass_rescaling=True, gas_branch_fitted=False)


def round2_summary_precision_bound(summary,parameters,anchor):
    """Guaranteed monotone within-bin optical mass error before float roundoff."""
    return evaluate_round2_summary(summary,parameters,anchor)["summary_quantization_bound_g"]


def _nearest(rows, target):
    return min(rows, key=lambda row: (abs(float(row["elapsed_h"])-target), float(row["elapsed_h"]), row["frame_index"]))


def anchors_by_run(rows):
    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)
    anchors = {}
    for run, values in by_run.items():
        row = _nearest(values, ANCHOR_TARGET)
        if row["affected_pore_volume_m3"] <= 0:
            raise ValueError(f"No observed positive CO2 pore-volume anchor for {run}")
        anchors[run] = {key: row[key] for key in ("elapsed_h", "affected_pore_volume_m3", "frame_index", "source_binding") if key in row}
    return anchors


def anchor_from_response(response):
    """Measured end-frame volume for a frozen evaluation, never a fitted value."""
    frame = response["frame"]
    partition = _physical_partition(response)
    if partition["affected_pore_volume_m3"] <= 0:
        raise ValueError("The end-of-injection image has no measured affected pore volume")
    return dict(elapsed_h=frame.elapsed_h, affected_pore_volume_m3=partition["affected_pore_volume_m3"],
        frame_index=frame.frame_index, source_binding=dict(frame.provenance),
        affected_volume_includes_excluded_known_physical=True,
        affected_volume_imputes_missing_geometry=False,
        unknown_geometry_phase_pixels=partition["affected_unknown_geometry_pixels"])


def select_fit_frames(rows, *, excluded_run=None, include_quarter_hour=False,include_ac38=False):
    """Bind target times to existing Studio frames; retain actual top contact."""
    by_run, seen = {}, set()
    expected_runs = (*FIT_RUNS,"ac38") if include_ac38 else FIT_RUNS
    for row in rows:
        if row.get("contract") != CONTRACT or row["run"] not in expected_runs:
            raise ValueError("Selection may read only source-bound R2 summaries from the 26 fit runs")
        key = row["run"], row["frame_index"]
        if key in seen:
            raise ValueError("Duplicate cached frame")
        seen.add(key)
        by_run.setdefault(row["run"], []).append(row)
    if set(by_run) != set(expected_runs):
        raise ValueError("The LORO/final fit needs the complete pre-registered 26-run cache")
    if excluded_run is not None and excluded_run not in FIT_RUNS:
        raise ValueError("A LORO fold may hold out only one of its fit runs")
    stages = {"stage1": [], "stage2": []}
    decisions = []
    for run, values in sorted(by_run.items()):
        if run == excluded_run:
            continue
        ordered = sorted(values, key=lambda item: (item["elapsed_h"], item["frame_index"]))
        for stage, start, end in (("stage1", .247 if include_quarter_hour else .497, 2.247), ("stage2", 2.747, 24.)):
            left, right = _nearest(ordered, start), _nearest(ordered, end)
            selected_rows = [row for row in ordered if left["elapsed_h"] <= row["elapsed_h"] <= right["elapsed_h"]]
            for row in selected_rows:
                dose = row.get("injected_mass_g", row.get("nominal_injected_mass_g"))
                source = row.get("no_outflow_evidence", {})
                reasons = []
                if dose is None or dose <= 0:
                    reasons.append("nonpositive_dose")
                if source.get("no_co2_left_rig") is not True or not source.get("source"):
                    reasons.append("no_loss_proxy_not_passed")
                if dose is not None and dose > 0 and row["missing_capacity_upper_g"] > .05*dose:
                    reasons.append("U_exceeds_5_percent_dose")
                if stage == "stage2" and row["elapsed_h"] <= ANCHOR_TARGET:
                    reasons.append("post_injection_target_missing")
                decisions.append(dict(run=run, stage=stage, target_window_h=[start, end], frame_index=row["frame_index"],
                    elapsed_h=row["elapsed_h"], selected=not reasons, omitted_reasons=reasons,
                    injected_mass_g=dose, missing_capacity_upper_g=row["missing_capacity_upper_g"], no_outflow_evidence=source))
                if not reasons:
                    stages[stage].append(row)
    return stages, decisions


def _weights(rows):
    counts = {}
    for row in rows:
        counts[row["run"]] = counts.get(row["run"], 0)+1
    return np.array([1/np.sqrt(len(counts)*counts[row["run"]]) for row in rows], np.float64)


def _decode_stage1(candidate, vector):
    if candidate == "J":
        return dict(candidate=candidate, endpoint=float(np.exp(vector[0])),
            amplitude=float(vector[1]), gamma=float(np.exp(vector[2])), beta=0.)
    result = dict(candidate=candidate, gain=float(np.exp(vector[0])), beta=0.)
    if candidate == "N3":
        result["blend"] = float(vector[1])
    return result


def _optimize_multistart(function, starts, bounds):
    attempts = []
    for start in starts:
        result = least_squares(function, start, bounds=bounds, max_nfev=600, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        attempts.append(result)
    finite = [result for result in attempts if result.success and np.isfinite(result.cost)]
    if not finite:
        raise ValueError("Common response optimizer did not produce a finite successful fit")
    minimum = min(result.cost for result in finite)
    # Deterministic equivalent-cost preference; no final-held-out score is used.
    chosen = next(result for result in finite if result.cost <= minimum+max(1e-12, abs(minimum)*1e-8))
    singular = np.linalg.svd(chosen.jac, compute_uv=False)
    return chosen, dict(multistart=[dict(initial=np.asarray(start).tolist(), solution=result.x.tolist(),
        cost=float(result.cost), success=bool(result.success), nfev=int(result.nfev)) for start, result in zip(starts, attempts)],
        jacobian_singular_values=singular.tolist(), jacobian_rank=int(np.linalg.matrix_rank(chosen.jac)),
        equivalent_cost_choice="first successful start within max(1e-12, minimum cost ×1e-8)")


def fit_round2_candidate(rows, candidate, *, excluded_run=None, include_quarter_hour=False,
        gas_sg1=False,no_dilution=False,include_ac38=False):
    """Fit shared aqueous parameters, then beta only, from cached fit-run data."""
    if candidate not in CANDIDATES or not candidate_availability()[candidate]["available"]:
        raise ValueError("Candidate is unavailable")
    stages, decisions = select_fit_frames(rows, excluded_run=excluded_run, include_quarter_hour=include_quarter_hour,include_ac38=include_ac38)
    anchors = anchors_by_run(rows)
    first, second = stages["stage1"], stages["stage2"]
    if not first:
        raise ValueError("No eligible two-sided injection observations")
    weights = _weights(first)
    def residual1(vector):
        parameters = _decode_stage1(candidate, vector)
        return weights*np.array([evaluate_round2_summary(row, parameters, anchors[row["run"]],gas_sg1=gas_sg1)["signed_interval_residual"] for row in first])
    if candidate == "J":
        starts = [np.array([np.log(endpoint), amplitude, np.log(gamma)])
            for endpoint, amplitude, gamma in ((1., 1., 1.), (.5, .8, .5), (2., 1., 2.), (4., .5, 1.))]
        bounds = ([-700., np.finfo(float).eps, -700.], [700., 1., 700.])
    elif candidate == "N3":
        starts = [np.array([np.log(gain), blend]) for gain, blend in ((1., 0.), (20., 0.), (1., 1.), (5., .5))]
        bounds = ([-700., 0.], [700., 1.])
    else:
        starts = [np.array([np.log(gain)]) for gain in (1., 20.)]
        bounds = ([-700.], [700.])
    result1, audit1 = _optimize_multistart(residual1, starts, bounds)
    parameters = _decode_stage1(candidate, result1.x)
    if second and not no_dilution:
        second_weights = _weights(second)
        def residual2(vector):
            value = dict(parameters, beta=float(vector[0]))
            return second_weights*np.array([evaluate_round2_summary(row, value, anchors[row["run"]],gas_sg1=gas_sg1)["signed_interval_residual"] for row in second])
        result2, audit2 = _optimize_multistart(residual2, [np.array([0.]), np.array([1.]), np.array([3.])], ([0.], [np.inf]))
        parameters["beta"] = float(result2.x[0])
    else:
        # An unfitted beta is explicit, never silently reported as calibrated.
        audit2 = {"assessable": False, "reason": "pre-declared no-dilution sensitivity" if no_dilution else "no eligible post-injection frames", "beta_fitted": False}
    return dict(contract=CONTRACT, candidate=candidate, parameters=parameters, anchors_by_run=anchors,
        calibration_runs=[run for run in ((*FIT_RUNS,"ac38") if include_ac38 else FIT_RUNS) if run != excluded_run], heldout_fit_run=excluded_run,
        stages={name: {"frames": len(values), "runs": sorted({row["run"] for row in values})} for name, values in stages.items()},
        frame_decisions=decisions, optimizer_audit={"stage1": audit1, "stage2": audit2},
        beta_fitted=bool(second) and not no_dilution, equal_run_weight=True, equal_frame_weight_within_run=True,
        sensitivity_policy={"gas_sg1":gas_sg1,"no_dilution":no_dilution,"include_quarter_hour":include_quarter_hour,"include_ac38":include_ac38},
        gas_branch_fitted=False, overflow_in_objective=False, final_heldout_used=False,
        native_paths_refitted=False, map_mass_rescaling=False)


def evaluate_round2_frame(response, parameters, anchor, *, fields=True, gas_sg1=False,
        port_as_aqueous=False, allow_provisional_l1_sensitivity=False,allow_frozen_sensitivity=False):
    """Use native CO2 physics on the locked full-resolution phase support."""
    _validate_parameters(parameters)
    if response.get("frozen_parameters") is not None and dict(parameters) != response["frozen_parameters"] and not (
            allow_frozen_sensitivity and parameters["candidate"] == response["frozen_parameters"]["candidate"]):
        raise ValueError("Only the frozen chosen response may be computed on final-held-out frames")
    frame, analysis, physics = response["frame"], response["analysis"], response["physics"]
    partition = _physical_partition(response)
    factor = dilution_factor(frame.elapsed_h, partition["affected_pore_volume_m3"], anchor, parameters.get("beta", 0.))
    if parameters["candidate"] == "J":
        p = JointConcentrationParameters(endpoint=parameters["endpoint"], amplitude=parameters["amplitude"], gamma=parameters["gamma"], beta=parameters.get("beta", 0.))
        fraction = concentration_field(response["joint_signal"], p, factor, np.ones(frame.domain.shape, np.float32))
    else:
        fraction = (_native_fraction(response["prepared"].aqueous_relative.astype(np.float64),
            response["linear_relative"].astype(np.float64), parameters)*factor).astype(np.float32)
    aq_known, gas_known = partition["aq_known"].copy(), partition["gas_known"].copy()
    if port_as_aqueous and frame.elapsed_h >= 4:
        verified = frame.provenance.get("l1_position_verified") is True
        provisional = (allow_provisional_l1_sensitivity and frame.provenance.get("l1_position_source_bound") is True
            and frame.provenance.get("l1_position_status") == "protocol_proxy_unverified")
        if frame.l1_neighbourhood_mask is None or not (verified or provisional):
            raise ValueError("L1 sensitivity requires a bound injector/protocol location")
        moved = gas_known & frame.l1_neighbourhood_mask
        gas_known &= ~moved
        aq_known |= moved & response["quantifiable"]
    aqueous = np.where(aq_known, fraction, 0).astype(np.float32)
    gas = np.zeros(frame.domain.shape, np.float32)
    gas[gas_known] = 1. if gas_sg1 else (1-response["swi"][gas_known]).astype(np.float32)
    # Exactly 1 denotes local solubility. This branch does not use gain, blend,
    # endpoint, amplitude, gamma, beta or the neighbouring aqueous plume.
    aqueous[gas_known] = 1.
    native = physics.mass_analysis(analysis._scalar(aqueous, frame.elapsed_h), analysis._scalar(gas, frame.elapsed_h))
    eligible = frame.mass_eligible_mask & frame.known_physical_domain
    def integrate(value, mask):
        return float(analysis.geometry.integrate(np.where(mask, value, 0.)))*1000
    aqueous_mass = integrate(native.mass_aq.img, eligible)
    gas_mass = integrate(native.mass_g.img, eligible)
    total = integrate(native.mass.img, eligible)
    residual_water = integrate(native.mass_aq.img, eligible & gas_known)
    dose, upper = frame.nominal_injected_mass_g, partition["missing_capacity_upper_g"]
    if port_as_aqueous and frame.elapsed_h >= 4:
        newly_unknown = moved & ~response["quantifiable"] & frame.mass_eligible_mask
        old_missing = partition["missing"] & frame.l1_neighbourhood_mask & (frame.phase_labels == 2)
        upper += float(np.sum(partition["volume"][newly_unknown]*partition["solubility"][newly_unknown]))*1000
        upper += float(np.sum(partition["bound_volume"][old_missing]*(partition["solubility"][old_missing]-np.maximum(partition["solubility"][old_missing],partition["density"][old_missing]))))*1000
        upper = max(0.,upper)
    residual = interval_residual(total, dose, upper) if dose is not None and dose > 0 else None
    excluded = {}
    covered = np.zeros(frame.domain.shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        covered |= mask
        excluded[name] = dict(aqueous_mass_g=integrate(native.mass_aq.img, mask), gas_mass_g=integrate(native.mass_g.img, mask),
            total_mass_g=integrate(native.mass.img, mask), main_total_includes_this=False,
            **partition["parts"][name])
    other = frame.domain & ~frame.mass_eligible_mask & ~covered
    excluded["other_not_mass_eligible"] = dict(aqueous_mass_g=integrate(native.mass_aq.img, other), gas_mass_g=integrate(native.mass_g.img, other),
        total_mass_g=integrate(native.mass.img, other), main_total_includes_this=False, **partition["parts"]["other_not_mass_eligible"])
    unknown = frame.domain & np.isin(frame.phase_labels, (1, 2)) & ~(aq_known | gas_known)
    out = dict(contract=CONTRACT, run=frame.run_id, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        parameters=dict(parameters), total_mass_g=total, aqueous_mass_g=aqueous_mass, gas_mass_g=gas_mass,
        aqueous_nongas_mass_g=aqueous_mass-residual_water, gas_water_mass_g=residual_water,
        gas_mass_share=gas_mass/total if total > 0 else 0., injected_mass_g=dose, nominal_injected_mass_g=dose,
        missing_capacity_upper_g=upper, mass_balance_missing_capacity_upper_g=upper,
        missing_capacity_parts=partition["parts"], excluded_mass=excluded,
        ratio=total/dose if dose is not None and dose > 0 else None,
        delta=None if residual is None else abs(residual), signed_interval_residual=residual,
        U_over_d=upper/dose if dose is not None and dose > 0 else None,
        affected_pore_volume_m3=partition["affected_pore_volume_m3"], dilution_factor=factor,
        aqueous_clip_pixel_share=float(np.mean(fraction[partition["main_aq"]] >= factor)) if partition["main_aq"].any() else 0.,
        aqueous_clip_share_before_dilution=True, unknown_phase_pixels=int(unknown.sum()),
        gas_branch_fitted=False, gas_rule="fixed 1-Swi; residual water at undiluted local solubility",
        optical_path_not_required_for_fixed_physical_gas=True, phase_labels_modified=False,
        main_geometry_imputed=False, capacity_only_max_depth_m=MAX_DEPTH_M, capacity_only_max_porosity=MAX_POROSITY,
        no_map_mass_rescaling=True, bottom_mass_allowed=False, overflow_in_objective=False,
        source_binding=dict(frame.provenance), recipe=dict(frame.recipe),
        gas_density_pore_volume_mean_g_l=float(np.sum(partition["volume"][partition["main_gas"]]*partition["density"][partition["main_gas"]])/np.sum(partition["volume"][partition["main_gas"]])) if partition["main_gas"].any() else None,
        overflow_gas_density_g_l=float(np.median(physics.density_gaseous_co2[0])))
    if fields:
        concentration = (aqueous*physics.solubility_co2).astype(np.float32)
        concentration[unknown] = np.nan
        gas[unknown & (frame.phase_labels == 2)] = np.nan
        out["fields"] = dict(c_aq_g_l=concentration, c_aq_relative=aqueous, s_g=gas,
            phase_labels=frame.phase_labels.copy(), unquantified_phase_mask=unknown,
            main_mass_eligible_mask=eligible, mass_density_g_l=np.asarray(native.mass.img, np.float32),
            aqueous_mass_density_g_l=np.asarray(native.mass_aq.img, np.float32), gas_mass_density_g_l=np.asarray(native.mass_g.img, np.float32))
    return out


def score_loro(predictions):
    """Literal saved-table scorer; production maintains an independent check."""
    grouped = {}
    for row in predictions:
        if row["run"] not in FIT_RUNS:
            raise ValueError("Final held-out/descriptive runs cannot enter candidate selection")
        grouped.setdefault(row["run"], []).append(row)
    if set(grouped) != set(FIT_RUNS):
        raise ValueError("Scores require all 26 left-out runs")
    injection, groups = {}, {}
    for target in (1., 2.25):
        rows = [_nearest(values, target) for values in grouped.values()]
        injection[str(target)] = float(np.median([abs(row["ratio"]-1) for row in rows]))
        for epoch in ("rig_light", "new_lamps"):
            selected = [row["ratio"] for row in rows if row["light_epoch"] == epoch]
            groups[f"{epoch}:{target}"] = float(np.median(selected)) if selected else None
    post, reasons = {}, {}
    for target in (4., 8., 12., 24.):
        rows = [_nearest(values, target) for values in grouped.values()]
        selected = [row for row in rows if row["injected_mass_g"] > 0 and row["missing_capacity_upper_g"] <= .05*row["injected_mass_g"]]
        post[str(target)] = float(np.median([row["delta"] for row in selected])) if len(selected) >= 13 else None
        reasons[str(target)] = dict(eligible_runs=len(selected), minimum_runs=13, assessable=len(selected) >= 13)
    e_inj = max(injection.values())
    available = [value for value in post.values() if value is not None]
    e_post = max(available) if available else None
    passes = e_inj <= .10 and all(value is not None and .90 <= value <= 1.10 for value in groups.values()) and (e_post is None or e_post <= .10)
    return dict(E_inj=e_inj, E_post=e_post, S=max(e_inj, e_post) if e_post is not None else e_inj,
        injection_errors=injection, group_median_ratios=groups, post_errors=post, post_assessability=reasons, passes=passes)


def select_candidate(scores):
    native = [(name, row) for name, row in scores.items() if name in ("N1", "N2", "N3") and row.get("passes")]
    selected = None
    if native:
        minimum = min(row["S"] for _, row in native)
        close = [(name, row) for name, row in native if row["S"] <= minimum+.005]
        selected = min(close, key=lambda item: ({"N1": 2, "N2": 2, "N3": 3}[item[0]], item[0] != "N1", item[0]))[0]
    joint = scores.get("J", {})
    if joint.get("passes") and (selected is None or joint["S"] < scores[selected]["S"]-.03):
        return dict(stop=True, reason="J requires Olav's choice before freeze", native_candidate=selected, joint_candidate="J")
    if selected is None:
        return dict(stop=True, reason="No candidate passes the pre-registered gates", native_candidate=None)
    return dict(stop=False, candidate=selected, reason="pre-registered native score/tie rule")


def fit_loro_candidates(rows, candidates=CANDIDATES):
    results = {}
    for candidate in candidates:
        if not candidate_availability()[candidate]["available"]:
            results[candidate] = dict(available=False, reason=candidate_availability()[candidate]["reason"])
            continue
        predictions, fits = [], []
        for run in FIT_RUNS:
            fit = fit_round2_candidate(rows, candidate, excluded_run=run)
            fits.append(fit)
            predictions.extend(evaluate_round2_summary(row, fit["parameters"], fit["anchors_by_run"][run]) for row in rows if row["run"] == run)
        results[candidate] = dict(available=True, predictions=predictions, fits=fits, score=score_loro(predictions))
    return results

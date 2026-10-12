"""Fixed, separate estimates for excluded FF_AC mass in round three.

The round-two main prediction is retained. Measured depth interpolation and
column-wise sand properties apply only to the separately reported estimates.
Correlated colour/column statistics retain the min/max operation per pixel
after dilution; independent sums of the two estimates would be incorrect.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from . import phase_ffac_common_mass as r1
from . import phase_ffac_common_mass_round2 as r2
from .concentration_joint import dilution_factor
from .phase_ffac_common_mass_round2_backend import reduce_weighted_codes

CONTRACT = "ffac-common-mass-round3-late-balance-v1"
FREEZE_CONTRACT = "ffac-common-mass-round3-frozen-v1"
ROUND2_FREEZE_SHA256 = "809165f719d45be20255f2da29353379ed85404c9fb82b426e67dd759b8e4431"
GAIN = 24.253536351793787
B0_BETA = 0.6448739280294578
MAX_COLUMN_DISTANCE_M = .020
INTERVALS = 32768
FIT_RUNS, HELDOUT_RUNS = r2.FIT_RUNS, r2.HELDOUT_RUNS
DESCRIPTIVE_RUNS = ("ac38",)


@dataclass(frozen=True)
class EstimateGeometry:
    x_m: np.ndarray
    y_m: np.ndarray
    depth_m: np.ndarray
    porosity: np.ndarray
    swi: np.ndarray
    facies_for_geometry: np.ndarray
    valid: np.ndarray
    depth_nearest_outside_hull: np.ndarray
    porosity_from_column: np.ndarray
    source_binding: dict


def _coordinates(x_m, y_m):
    x, y = np.asarray(x_m, np.float64), np.asarray(y_m, np.float64)
    if x.ndim != 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite source-bound physical x/y grids are required")
    if np.any(np.diff(y, axis=0) >= 0):
        raise ValueError("Physical height must decrease down each image column; no automatic axis flip")
    return x, y


def _column_neighbours(mask):
    rows = np.arange(mask.shape[0], dtype=np.int32)[:, None]
    previous = np.maximum.accumulate(np.where(mask, rows, -1), axis=0)
    following = np.minimum.accumulate(np.where(mask, rows, mask.shape[0])[::-1], axis=0)[::-1]
    return previous, following


def build_estimate_geometry(x_m, y_m, facies, depth_points, *, porosity_by_facies,
        swi_by_facies, source_binding=None, chunk_pixels=250000):
    """Linear measured depth, nearest outside hull, nearest sand in one column.

    Missing sand in a column remains explicitly undefined. No neighbouring
    column, maximum depth or maximum porosity is substituted into an estimate.
    """
    x, y = _coordinates(x_m, y_m)
    labels = np.asarray(facies)
    points = np.asarray(depth_points, np.float64)
    if (labels.shape != x.shape or not np.issubdtype(labels.dtype, np.integer)
            or points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3
            or not np.isfinite(points).all() or np.any(points[:, 2] <= 0)):
        raise ValueError("Registered facies and measured x/y/depth triples are required")
    linear, nearest = LinearNDInterpolator(points[:, :2], points[:, 2]), cKDTree(points[:, :2])
    depth, outside = np.empty(x.size, np.float64), np.zeros(x.size, bool)
    for start in range(0, x.size, chunk_pixels):
        stop = min(x.size, start + chunk_pixels)
        query = np.column_stack((x.ravel()[start:stop], y.ravel()[start:stop]))
        values = np.asarray(linear(query), np.float64)
        bad = ~np.isfinite(values)
        if bad.any():
            values[bad] = points[nearest.query(query[bad])[1], 2]
        depth[start:stop], outside[start:stop] = values, bad
    phi, swi = np.full(x.shape, np.nan), np.full(x.shape, np.nan)
    for label, value in porosity_by_facies.items():
        if int(label) == 0:
            continue
        if not np.isfinite(value) or not 0 < float(value) <= 1:
            raise ValueError("Measured sand porosity must be positive and no greater than one")
        phi[labels == int(label)] = float(value)
    for label, value in swi_by_facies.items():
        if int(label) == 0:
            continue
        if not np.isfinite(value) or not 0 <= float(value) <= 1:
            raise ValueError("Measured sand Swi must lie in [0,1]")
        swi[labels == int(label)] = float(value)
    sand = np.isfinite(phi)
    previous, following = _column_neighbours(sand)
    cols = np.arange(x.shape[1])[None, :]
    prev_y = y[np.maximum(previous, 0), cols]
    next_y = y[np.minimum(following, x.shape[0]-1), cols]
    prev_dist = np.where(previous >= 0, np.abs(prev_y-y), np.inf)
    next_dist = np.where(following < x.shape[0], np.abs(next_y-y), np.inf)
    chosen = np.where(prev_dist <= next_dist, previous, following)
    available = (chosen >= 0) & (chosen < x.shape[0])
    donor = np.clip(chosen, 0, x.shape[0]-1)
    filled = ~sand & available
    phi[filled] = phi[donor, cols][filled]
    swi[filled] = swi[donor, cols][filled]
    geometry_facies = labels.copy()
    geometry_facies[filled] = labels[donor, cols][filled]
    depth = depth.reshape(x.shape)
    valid = np.isfinite(phi) & np.isfinite(depth) & (depth > 0)
    return EstimateGeometry(x, y, depth, phi, swi, geometry_facies, valid,
        outside.reshape(x.shape), filled, dict(source_binding or {}))


def nearest_main_above(main_positive, y_m, *, max_distance_m=MAX_COLUMN_DISTANCE_M):
    """Indices of the nearest strictly higher main CO2 pixel in the same column."""
    mask, y = np.asarray(main_positive, bool), np.asarray(y_m, np.float64)
    if mask.shape != y.shape or y.ndim != 2 or np.any(np.diff(y, axis=0) >= 0):
        raise ValueError("Main mask must share a downward-decreasing physical-height grid")
    if max_distance_m != MAX_COLUMN_DISTANCE_M:
        raise ValueError("The column estimate distance is fixed at 20 mm")
    previous, _ = _column_neighbours(mask)
    previous = np.vstack((np.full((1, mask.shape[1]), -1, np.int32), previous[:-1]))
    cols = np.arange(mask.shape[1])[None, :]
    distance = y[np.maximum(previous, 0), cols]-y
    available = (previous >= 0) & (distance > 0) & (distance <= max_distance_m + 1e-12)
    return previous, available, np.where(available, distance, np.nan)


def prepare_own_colour_response(response, path_folder, geometry, *, photo_support):
    """Reuse native optics/flash without putting estimation geometry into M.

    The scientific RAW, registered start and measured clear-water divisor are
    unchanged. The original flash object is used, including its recipe and
    solubility reference. The main analysis and arrays are never modified.
    """
    frame = response["frame"]
    shape = frame.domain.shape
    photo = np.asarray(photo_support, bool)
    if photo.shape != shape or geometry.valid.shape != shape:
        raise ValueError("Photo support and estimate geometry must share the locked grid")
    clear = frame.clear_water_rgb
    observed = (frame.domain & photo & frame.clear_water_supported_mask
        & np.isfinite(clear).all(axis=2) & (clear > 0).all(axis=2) & geometry.valid)
    current, baseline = np.zeros_like(frame.current_rgb), np.zeros_like(frame.baseline_rgb)
    np.divide(frame.current_rgb, clear, out=current, where=observed[..., None])
    np.divide(frame.baseline_rgb, clear, out=baseline, where=observed[..., None])
    if not np.isfinite(current).all() or not np.isfinite(baseline).all():
        raise ValueError("Measured-colour normalization produced invalid values")
    analysis = r1.UnclippedNativeConcentrationAnalysis(path_folder, baseline_rgb=baseline,
        labels=frame.facies, domain=geometry.valid & frame.domain,
        depth_m=np.where(geometry.valid, geometry.depth_m, 0),
        porosity=np.where(geometry.valid, geometry.porosity, 0), dimensions_m=frame.dimensions_m)
    interpretation = analysis.pipeline.call_color_interpretation(analysis._rgb(current))
    signal = analysis.pipeline.call_pH_analysis(interpretation)
    aqueous, _ = response["flash"](signal)
    available = observed & ~np.isin(frame.facies, response["metadata"]["ignore_labels"])
    return dict(aqueous_relative=np.asarray(aqueous.img, np.float32), available=available,
        audit=dict(native_optics_used=True, original_flash_object_used=True,
            main_geometry_modified=False, colour_paths_modified=False,
            main_analysis_modified=False, measured_colour_pixels=int(available.sum()),
            photo_support_required=True, undefined_facies_paths_borrowed=False))


def _parts(frame, missing):
    parts, covered = {}, np.zeros(missing.shape, bool)
    for name, mask in sorted((frame.excluded_masks or {}).items()):
        selection = missing & mask & ~covered
        parts[name] = selection
        covered |= mask
    parts["other_not_mass_eligible"] = missing & ~frame.mass_eligible_mask & ~covered
    parts["unquantified_facies"] = missing & frame.mass_eligible_mask & ~covered
    return parts


def _weighted_median(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return None
    value, weight = values[valid], weights[valid]
    order = np.argsort(value, kind="stable")
    return float(value[order[min(np.searchsorted(np.cumsum(weight[order]), weight.sum()/2), len(order)-1)]])


def _native_histogram(relative, capacity, mask, backend):
    codes = np.rint(np.clip(relative[mask],0,1)*INTERVALS).astype(np.int64)
    codes,weights,audit = reduce_weighted_codes(codes,capacity[mask],backend=backend)
    return dict(native_relative=(codes/INTERVALS).tolist(),capacity_g=weights.tolist(),
        quantization_half_width=.5/INTERVALS,reduction_audit=audit)


def _pairs(own, column, solubility_ratio, source_gas, own_available, column_available, capacity, mask, backend):
    # E1 is diluted aqueous. E2 is either diluted aqueous or undiluted water in
    # a source gas pixel. Recipient gas is aggregated separately, never fitted.
    a1, a2, ratio, weights = own[mask], column[mask],solubility_ratio[mask],capacity[mask]
    scale = max(1., float(np.max(ratio, initial=0)))
    q1 = np.rint(np.clip(a1, 0, 1)*INTERVALS).astype(np.int64)
    q2 = np.rint(np.clip(a2,0,1)*INTERVALS).astype(np.int64)
    qr = np.rint(ratio/scale*INTERVALS).astype(np.int64)
    kinds = (own_available[mask].astype(np.int64)
        + 2*column_available[mask].astype(np.int64) + 4*source_gas[mask].astype(np.int64))
    codes = ((q1*(INTERVALS+1)+q2)*(INTERVALS+1)+qr)*8+kinds
    codes, sums, audit = reduce_weighted_codes(codes, weights, backend=backend)
    kind, pair = codes % 8, codes // 8
    qr,pair = pair%(INTERVALS+1),pair//(INTERVALS+1)
    return dict(own_native_relative=(pair//(INTERVALS+1)/INTERVALS).tolist(),
        column_native_relative=(pair%(INTERVALS+1)/INTERVALS).tolist(),
        solubility_ratio=(qr/INTERVALS*scale).tolist(),
        kind=kind.tolist(), aqueous_capacity_g=sums.tolist(),
        quantization_half_width_own=.5/INTERVALS,
        quantization_half_width_column=.5*scale/INTERVALS,
        aqueous_capacity_total_g=float(np.sum(weights)),
        ratio_scale=scale,
        reduction_audit=audit)


def summarize_round3_response(response, geometry, *, own_colour, photo_support,
        no_outflow_evidence, light_epoch, source_binding=None, distances_m=None,
        backend="auto", scope="calibration"):
    """Extract native main and correlated excluded estimates once per frame."""
    frame, physics = response["frame"], response["physics"]
    run = r1.normalize_run(frame.run_id)
    if scope == "calibration" and run not in FIT_RUNS:
        raise ValueError("Only the unchanged 26 fit runs may enter round-three extraction")
    if scope not in ("calibration", "saved_baseline_diagnostic", "frozen_evaluation"):
        raise ValueError("Extraction requires an explicit source-bound evaluation scope")
    partition = r2._physical_partition(response)
    parameters = dict(candidate="N1", gain=GAIN, beta=0.)
    anchor = dict(elapsed_h=r2.ANCHOR_TARGET, affected_pore_volume_m3=max(partition["affected_pore_volume_m3"], 1e-20))
    main = r2.evaluate_round2_frame(response, parameters, anchor, fields=True,
        allow_frozen_sensitivity=True)
    shape = frame.domain.shape
    area = np.broadcast_to(np.asarray(frame.pixel_area_m2, np.float64), shape)
    sol, density = partition["solubility"], partition["density"]
    volume = np.where(geometry.valid, area*geometry.depth_m*geometry.porosity, 0.)
    capacity = volume*sol*1000
    missing = partition["missing"]
    is_gas = frame.phase_labels == 2
    gas_available = missing & is_gas & geometry.valid & np.isfinite(geometry.swi)
    fixed_gas = volume*density*(1-geometry.swi)*1000
    fixed_water = volume*sol*geometry.swi*1000
    gas_constant = np.where(gas_available, fixed_gas+fixed_water, 0.)
    own_fraction = np.asarray(own_colour["aqueous_relative"],np.float64)
    own_available = missing & ~is_gas & own_colour["available"] & np.asarray(photo_support, bool) & geometry.valid
    main_positive = partition["main_aq"] | partition["main_gas"]
    donor, column_available, column_distance = nearest_main_above(main_positive, geometry.y_m)
    cols = np.arange(shape[1])[None, :]
    donor = np.maximum(donor, 0)
    source_gas = partition["main_gas"][donor, cols]
    source_relative = response["prepared"].aqueous_relative[donor,cols].astype(np.float64)
    source_sol = sol[donor,cols]
    solubility_ratio = np.divide(source_sol,sol,out=np.zeros(shape),where=sol>0)
    column_fraction = source_relative
    column_available &= missing & ~is_gas & geometry.valid & np.isfinite(source_relative) & np.isfinite(solubility_ratio)
    source_gas &= column_available
    pair_mask = missing & ~is_gas & (own_available | column_available)
    own_fraction = np.where(own_available, own_fraction, 0.)
    column_fraction = np.where(column_available, column_fraction, 0.)
    neither = missing & ~(gas_available | own_available | column_available)
    undefined = neither & ~geometry.valid
    urest = volume* np.where(is_gas, np.maximum(sol, density), sol)*1000
    parts = _parts(frame, missing)
    part_records = {}
    distance_arrays = {name: np.broadcast_to(np.asarray(value, np.float64), shape)
        for name, value in (distances_m or {}).items()}
    for name, mask in parts.items():
        labels = {}
        for label in np.unique(frame.facies[mask]):
            selection = mask & (frame.facies == label)
            labels[str(int(label))] = dict(co2_area_cm2=float(area[selection].sum()*10000),
                without_main_geometry_area_cm2=float(area[selection & ~frame.known_physical_domain].sum()*10000),
                estimate_geometry_undefined_area_cm2=float(area[selection & ~geometry.valid].sum()*10000),
                capacity_g=None if (selection & ~geometry.valid).any() else float(urest[selection].sum()),
                own_colour_histogram=_native_histogram(own_fraction,capacity,selection & own_available,backend),
                fixed_gas_mass_g=float(fixed_gas[selection & gas_available].sum()),
                fixed_gas_water_mass_g=float(fixed_water[selection & gas_available].sum()))
        part_records[name] = dict(by_facies=labels, co2_area_cm2=float(area[mask].sum()*10000),
            without_main_geometry_area_cm2=float(area[mask & ~frame.known_physical_domain].sum()*10000),
            geometry_extended_area_cm2=float(area[mask & geometry.valid & ~frame.known_physical_domain].sum()*10000),
            depth_nearest_outside_hull_area_cm2=float(area[mask & geometry.depth_nearest_outside_hull].sum()*10000),
            porosity_column_extended_area_cm2=float(area[mask & geometry.porosity_from_column].sum()*10000),
            capacity_g=None if (mask & ~geometry.valid).any() else float(urest[mask].sum()),
            own_colour_histogram=_native_histogram(own_fraction,capacity,mask & own_available,backend),
            fixed_gas_mass_g=float(fixed_gas[mask & gas_available].sum()),
            fixed_gas_water_mass_g=float(fixed_water[mask & gas_available].sum()),
            urest_g=None if (mask & undefined).any() else float(urest[mask & neither].sum()),
            median_distances_mm={key: None if (value := _weighted_median(array[mask],area[mask])) is None
                else value*1000 for key,array in distance_arrays.items()})
    row = dict(contract=CONTRACT, run=run, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
        light_epoch=light_epoch, injected_mass_g=frame.nominal_injected_mass_g,
        source_binding=dict(source_binding or frame.provenance), geometry_source_binding=geometry.source_binding,
        no_outflow_evidence=dict(no_outflow_evidence), scope=scope,
        main_aqueous_histogram=_native_histogram(response["prepared"].aqueous_relative,
            partition["volume"]*sol*1000,partition["main_aq"],backend),
        baseline_undiluted_main_mass_g=main["total_mass_g"],main_gas_mass_g=main["gas_mass_g"],
        main_gas_water_mass_g=main["gas_water_mass_g"], affected_pore_volume_m3=partition["affected_pore_volume_m3"],
        estimate_fixed_gas_mass_g=float(fixed_gas[gas_available].sum()),
        estimate_fixed_gas_water_mass_g=float(fixed_water[gas_available].sum()),
        estimate_gas_constant_g=float(gas_constant[gas_available].sum()),
        urest_g=None if undefined.any() else float(urest[neither].sum()),
        estimate_geometry_undefined_co2_pixels=int(undefined.sum()),
        estimate_geometry_extended_area_cm2=float(area[missing & geometry.valid & ~frame.known_physical_domain].sum()*10000),
        excluded_parts=part_records, own_colour_audit=own_colour["audit"],
        own_estimate_pixels=int(own_available.sum()), column_estimate_pixels=int(column_available.sum()),
        both_estimates_pixels=int((own_available & column_available).sum()),
        fixed_gas_estimate_pixels=int(gas_available.sum()), neither_estimate_pixels=int(neither.sum()),
        column_distance_median_mm=None if not column_available.any() else float(np.nanmedian(column_distance[column_available])*1000),
        main_geometry_modified=False, main_labels_modified=False, gas_branch_fitted=False,
        paired_estimates=_pairs(own_fraction,column_fraction,solubility_ratio,source_gas,
            own_available,column_available,capacity,pair_mask,backend))
    return row


def _authorize_prediction(row,parameters,frozen_calibration):
    if row["run"] not in (*HELDOUT_RUNS,*DESCRIPTIVE_RUNS):
        return
    if parameters==dict(candidate="N1",gain=GAIN,beta=B0_BETA):
        return
    if frozen_calibration is None:
        raise ValueError("Freeze B1 before evaluating any final-held-out or descriptive image with it")
    freeze,_ = verify_round3_freeze(frozen_calibration)
    if parameters != freeze["parameters"]:
        raise ValueError("Held-out evaluation differs from the frozen round-three calibration")


def evaluate_round3_summary(row, parameters, anchor, *, frozen_calibration=None):
    """Evaluate the literal per-pixel estimate interval at one shared beta."""
    parameters = dict(candidate="N1",gain=GAIN,beta=float(parameters)) if np.isscalar(parameters) else dict(parameters)
    r2._validate_parameters(parameters)
    if parameters["candidate"] != "N1" or row.get("contract") != CONTRACT:
        raise ValueError("Round-three evaluation uses the unchanged native N1 response")
    _authorize_prediction(row,parameters,frozen_calibration)
    gain,beta = parameters["gain"],parameters["beta"]
    factor = dilution_factor(row["elapsed_h"],row["affected_pore_volume_m3"],anchor,beta)
    paired = row["paired_estimates"]
    own,column,ratio,kind,weight = (np.asarray(paired[key],np.float64 if key != "kind" else np.int64)
        for key in ("own_native_relative","column_native_relative","solubility_ratio","kind","aqueous_capacity_g"))
    e1 = np.clip(own*gain,0,1)*factor
    e2 = ratio*np.where((kind & 4) != 0,1.,np.clip(column*gain,0,1)*factor)
    has1,has2 = (kind & 1) != 0,(kind & 2) != 0
    both = has1 & has2
    single = np.where(has1,e1,e2)
    lo = float(np.sum(weight*np.where(both,np.minimum(e1,e2),single)))+row["estimate_gas_constant_g"]
    hi = float(np.sum(weight*np.where(both,np.maximum(e1,e2),single)))+row["estimate_gas_constant_g"]
    def own_mass(hist):
        return float(np.dot(np.asarray(hist["capacity_g"],np.float64),
            np.clip(np.asarray(hist["native_relative"],np.float64)*gain,0,1)))*factor
    main_aq = own_mass(row["main_aqueous_histogram"])
    main = main_aq+row["main_gas_mass_g"]+row["main_gas_water_mass_g"]
    urest = row["urest_g"]
    width = None if urest is None else hi-lo+urest
    dose = row["injected_mass_g"]
    residual = None if width is None or dose is None or dose <= 0 else r2.interval_residual(main+lo,dose,width)
    def evaluated_part(part):
        own_aq = own_mass(part["own_colour_histogram"])
        gas_total = part["fixed_gas_mass_g"]+part["fixed_gas_water_mass_g"]
        return dict(part,own_colour_aqueous_mass_g=own_aq,gas_rule_total_mass_g=gas_total,
            own_colour_mass_g=own_aq+gas_total)
    parts = {name: dict(evaluated_part(part),by_facies={label:evaluated_part(value)
        for label,value in part["by_facies"].items()})
        for name,part in row.get("excluded_parts",{}).items()}
    return dict(contract=CONTRACT,run=row["run"],frame_index=row["frame_index"],elapsed_h=row["elapsed_h"],
        light_epoch=row.get("light_epoch"),source_binding=row.get("source_binding"),parameters=parameters,
        main_mass_g=main,total_mass_g=main,main_aqueous_nongas_mass_g=main_aq,
        aqueous_mass_g=main_aq+row["main_gas_water_mass_g"],gas_mass_g=row["main_gas_mass_g"],
        gas_water_mass_g=row["main_gas_water_mass_g"],gas_share=0. if main == 0 else row["main_gas_mass_g"]/main,
        E_lo_g=lo,E_hi_g=hi,U_rest_g=urest,W_g=width,
        interval_lower_g=main+lo,interval_upper_g=None if urest is None else main+hi+urest,
        injected_mass_g=dose,W_over_d=None if width is None or dose is None or dose <= 0 else width/dose,
        signed_interval_residual=residual,delta=None if residual is None else abs(residual),
        dilution_factor=factor,estimate_gas_mass_g=row["estimate_fixed_gas_mass_g"],
        estimate_gas_water_mass_g=row["estimate_fixed_gas_water_mass_g"],
        estimate_geometry_extended_area_cm2=row["estimate_geometry_extended_area_cm2"],excluded_parts=parts,
        summary_quantization_bound_g=(paired["aqueous_capacity_total_g"]*.5/INTERVALS*
            (gain*(1+paired["ratio_scale"])+paired["ratio_scale"])
            +sum(row["main_aqueous_histogram"]["capacity_g"])*gain*.5/INTERVALS),
        main_geometry_modified=False,
        main_labels_modified=False,map_mass_rescaling=False,overflow_in_objective=False)


def prepare_round3_beta_summary(row, gain):
    """Compile an exact evaluation of the cached statistics for beta fitting.

    Most pairs have two diluted aqueous estimates and reduce to two sums. A
    pair with undiluted gas-water above it has one crossing. Sorted crossings
    and prefix sums preserve every per-pixel min/max while making a beta
    evaluation logarithmic, rather than scanning the large excluded band.
    """
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("One positive common native gain is required")
    paired = row["paired_estimates"]
    own,column,ratio,kind,weight = (np.asarray(paired[key],np.float64 if key != "kind" else np.int64)
        for key in ("own_native_relative","column_native_relative","solubility_ratio","kind","aqueous_capacity_g"))
    first = np.clip(own*gain,0,1)
    second = ratio*np.clip(column*gain,0,1)
    has1,has2,gas2 = (kind&1)!=0,(kind&2)!=0,(kind&4)!=0
    both_aq = has1 & has2 & ~gas2
    only_aq = (has1 & ~has2) | (~has1 & has2 & ~gas2)
    single = np.where(has1,first,second)
    lo_aq = float(np.sum(weight[both_aq]*np.minimum(first[both_aq],second[both_aq]))+np.sum(weight[only_aq]*single[only_aq]))
    hi_aq = float(np.sum(weight[both_aq]*np.maximum(first[both_aq],second[both_aq]))+np.sum(weight[only_aq]*single[only_aq]))
    only_gas = ~has1 & has2 & gas2
    constant = row["estimate_gas_constant_g"]+float(np.sum(weight[only_gas]*ratio[only_gas]))
    cross = has1 & has2 & gas2
    a,c = first[cross]*weight[cross],ratio[cross]*weight[cross]
    zero = a == 0
    constant_hi = float(c[zero].sum())
    a,c = a[~zero],c[~zero]
    cut = c/a
    order = np.argsort(cut,kind="stable")
    cut,a,c = cut[order],a[order],c[order]
    hist = row["main_aqueous_histogram"]
    main_coefficient = float(np.dot(np.asarray(hist["capacity_g"],np.float64),
        np.clip(np.asarray(hist["native_relative"],np.float64)*gain,0,1)))
    return dict(row=row,gain=float(gain),main_coefficient_g=main_coefficient,
        lo_aqueous_coefficient_g=lo_aq,hi_aqueous_coefficient_g=hi_aq,
        constant_g=constant,constant_hi_g=constant_hi,crossings=cut,
        prefix_a=np.r_[0.,np.cumsum(a)],prefix_c=np.r_[0.,np.cumsum(c)],
        total_a=float(a.sum()),total_c=float(c.sum()))


def evaluate_round3_compiled(compiled, beta, anchor, *, frozen_calibration=None):
    row = compiled["row"]
    _authorize_prediction(row,dict(candidate="N1",gain=compiled["gain"],beta=float(beta)),frozen_calibration)
    factor = dilution_factor(row["elapsed_h"],row["affected_pore_volume_m3"],anchor,beta)
    index = int(np.searchsorted(compiled["crossings"],factor,side="right"))
    a,c = compiled["prefix_a"][index],compiled["prefix_c"][index]
    cross_lo = c+(compiled["total_a"]-a)*factor
    cross_hi = a*factor+compiled["total_c"]-c
    lo = compiled["constant_g"]+compiled["lo_aqueous_coefficient_g"]*factor+cross_lo
    hi = compiled["constant_g"]+compiled["hi_aqueous_coefficient_g"]*factor+cross_hi+compiled["constant_hi_g"]
    main = compiled["main_coefficient_g"]*factor+row["main_gas_mass_g"]+row["main_gas_water_mass_g"]
    upper = row["urest_g"]
    width = None if upper is None else hi-lo+upper
    dose = row["injected_mass_g"]
    residual = None if width is None or dose is None or dose <= 0 else r2.interval_residual(main+lo,dose,width)
    return dict(main_mass_g=main,total_mass_g=main,E_lo_g=lo,E_hi_g=hi,U_rest_g=upper,
        W_g=width,W_over_d=None if width is None or dose is None or dose<=0 else width/dose,
        signed_interval_residual=residual,delta=None if residual is None else abs(residual),
        dilution_factor=factor,gas_mass_g=row["main_gas_mass_g"],gas_water_mass_g=row["main_gas_water_mass_g"])


def verify_round3_freeze(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if (value.get("contract") != FREEZE_CONTRACT or value.get("frozen") is not True
            or set(value.get("calibration_runs",())) != set(FIT_RUNS)
            or set(value.get("heldout_runs",())) != set(HELDOUT_RUNS)
            or value.get("round2_freeze_sha256") != ROUND2_FREEZE_SHA256
            or value.get("parameters",{}).get("candidate") != "N1"):
        raise ValueError("B1 held-out computation requires the exact round-three frozen calibration")
    r2._validate_parameters(value["parameters"])
    if value.get("map_mass_rescaling") is not False or value.get("overflow_in_objective") is not False:
        raise ValueError("Frozen calibration changes the fixed scientific contract")
    for name,item in value.get("calibration_files",{}).items():
        file,expected = (item.get("path",name),item.get("sha256")) if isinstance(item,dict) else (name,item)
        file = Path(file)
        if not file.is_absolute():
            file = path.parent/file
        r2._verify_file(file,expected)
    if not value.get("calibration_files") or not value.get("frozen_utc"):
        raise ValueError("Freeze lacks actual source files or publication time")
    return value,r1.sha256(path)

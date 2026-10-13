"""Selected native N1 fields and original float64 Geometry integration."""
from __future__ import annotations
import numpy as np
from .concentration_joint import dilution_factor

FIELDS = {"c_aq_g_l", "c_aq_relative", "s_g", "phase_labels", "unquantified_phase_mask", "main_mass_eligible_mask", "mass_density_g_l", "aqueous_mass_density_g_l", "gas_mass_density_g_l"}

def validate_parameters(parameters):
    """Require the frozen positive N1 gain and nonnegative dilution beta."""
    if (parameters.get("candidate") != "N1" or not np.isfinite(parameters.get("gain", 0))
            or parameters.get("gain",0) <= 0 or not np.isfinite(parameters.get("beta",0))
            or parameters.get("beta",0) < 0 or any(k in parameters for k in ("gas_scale","gas_saturation","gas_gain"))):
        raise ValueError("Frozen N1 gain/beta and an unfitted gas branch are required")

def main_partition(response):
    """Original measured main support/volume, without discarded bound products."""
    import numpy as np
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
    sol = np.asarray(physics.solubility_co2, np.float64)
    density = np.asarray(physics.density_gaseous_co2, np.float64)
    if (not np.isfinite(sol[positive]).all() or not np.isfinite(density[positive]).all()
            or np.any(sol[positive] < 0) or np.any(density[positive] < 0)):
        raise ValueError("Native thermodynamic capacity must be finite across the locked phase domain")
    return dict(volume=volume, aq_known=aq_known, gas_known=gas_known,
                affected_pore_volume_m3=float(np.sum(volume[positive])))

def evaluate_main(response, anchor, *, parameters, pixel_math, calibration_sha256, domain_masks, kept_fields=FIELDS):
    """Evaluate original typed N1 fields and native geometry reductions."""
    math = pixel_math
    validate_parameters(parameters)
    if response.get("precision") != math.precision or set(domain_masks) != {"full_rig", "box1", "box2"}:
        raise ValueError("Exact working precision and all three native domains are required")
    if set(kept_fields)-FIELDS:
        raise ValueError("Unknown retained field")
    digest = calibration_sha256
    frame = response["frame"]
    for name, mask in domain_masks.items():
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != frame.domain.shape:
            raise ValueError("Frozen domain mask differs from the native full-resolution grid")
        if name == "full_rig" and not np.array_equal(mask, frame.domain):
            raise ValueError("Full-rig domain must be the exact original frozen phase domain")
    analysis, physics = response["analysis"], response["physics"]
    partition = main_partition(response)  # original finite/positive physics guards
    factor = dilution_factor(frame.elapsed_h, partition["affected_pore_volume_m3"],
                                      anchor, parameters.get("beta", 0.))
    maps = math.selected_n1_fields(response["prepared"].aqueous_relative,
        gain=parameters["gain"], factor=factor, swi=response["swi"],
        aqueous_known=partition["aq_known"], gas_known=partition["gas_known"],
        solubility=physics.solubility_co2, gas_density=physics.density_gaseous_co2)
    eligible = frame.mass_eligible_mask & frame.known_physical_domain
    def integrate(value, mask):
        # Original Geometry.integrate itself remains unchanged; its weights
        # are genuinely float64 before multiply and repeated axis-zero sums.
        """Use original float64 native geometry weights and axis order."""
        analysis.geometry._prepare_cached_voxel_volume(list(value.shape[:analysis.geometry.space_dim]))
        if np.asarray(analysis.geometry.cached_voxel_volume).dtype != np.float64:
            raise TypeError("Geometry weights must be float64 before reduction")
        return float(analysis.geometry.integrate(np.where(mask, value, 0.)))*1000
    def masses(mask):
        """Report separate total, aqueous and gas native integrals."""
        return dict(total_mass_g=integrate(maps["mass_total"], mask),
                    aqueous_mass_g=integrate(maps["mass_aqueous"], mask),
                    gas_mass_g=integrate(maps["mass_gas"], mask))
    domains = {name: masses(eligible & mask) for name, mask in domain_masks.items()}
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
            concentration = maps["concentration"]
            concentration[unknown] = np.nan
            fields["c_aq_g_l"] = concentration
        if "unquantified_phase_mask" in kept_fields:
            fields["unquantified_phase_mask"] = unknown
        if "s_g" in kept_fields:
            gas = maps["gas"]
            gas[unknown & (frame.phase_labels == 2)] = np.nan
            fields["s_g"] = gas
    if "c_aq_relative" in kept_fields: fields["c_aq_relative"] = maps["aqueous"]
    if "phase_labels" in kept_fields: fields["phase_labels"] = frame.phase_labels.copy()
    if "main_mass_eligible_mask" in kept_fields: fields["main_mass_eligible_mask"] = eligible
    for name, value in (("mass_density_g_l", "mass_total"),
                        ("aqueous_mass_density_g_l", "mass_aqueous"),
                        ("gas_mass_density_g_l", "mass_gas")):
        if name in kept_fields: fields[name] = maps[value]
    return dict(contract="darsia-n1-fields-v1", parameters=parameters,
        calibration_sha256=digest, scalar_reference="native_final_calibration_fields", run=frame.run_id,
        frame_index=frame.frame_index, elapsed_h=frame.elapsed_h, dilution_factor=factor,
        **domains["full_rig"], domains=domains, excluded_mass=excluded,
        affected_pore_volume_m3=partition["affected_pore_volume_m3"], fields=fields,
        precision=math.precision, pixel_backend=math.facts()["backend"],
        pixel_backend_facts=dict(math.facts()),
        input_storage_dtype="float32-log-RGB/frozen-clearwater", working_pixel_dtype=math.precision,
        parameter_dtype="float64", reduction_dtype="float64",
        scientific_eligibility=False, production_permission_granted=False)

def main_products(result, pixel_math):
    """Original retained export/mask/units, with truthful chosen mass dtype."""
    import numpy as np
    if result.get("precision") != pixel_math.precision:
        raise ValueError("Products differ from the declared working precision")
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
        if value.dtype != pixel_math.dtype or value.shape != mask.shape:
            raise ValueError("Retained mass map must keep chosen precision and exact grid")
        products[name] = np.where(mask, value, 0.)
    return dict(products=products, units={"concentration_aqueous": "g/L", "saturation_free_gas": "1",
        "mass_total": "g/L", "mass_aqueous": "g/L", "mass_free_gas": "g/L", "phase_labels": "category"},
        required_transport_weight_units="L pore volume per pixel", native_mass_scalar_units="g",
        main_mask_applied=True, raw_native_fields_retained=True, precision=pixel_math.precision,
        pixel_backend=result["pixel_backend"], scientific_eligibility=False, production_permission_granted=False)

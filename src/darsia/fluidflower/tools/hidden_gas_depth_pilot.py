"""Evaluate a declared hidden-depth gas scenario on saved concentration fields.

This is a diagnostic sidecar. It never changes the observed phase map, refits
optics, or registers a production concentration result. In particular, an
overflow target is compared *after* the scenario is generated, not used to
choose a path or occupancy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower.hidden_gas_depth import (
    HiddenGasDepthSettings,
    HiddenGasDepthTracker,
)


FORMAT = "fluidflower-hidden-depth-gas-diagnostic-v1"


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _coarse_extensive(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Area-pool a volume field while preserving its total integral."""
    field = np.asarray(field, np.float64)
    resized = cv2.resize(field, target_hw[::-1], interpolation=cv2.INTER_AREA)
    original = float(field.sum(dtype=np.float64))
    resized_sum = float(resized.sum(dtype=np.float64))
    if original > 0 and resized_sum > 0:
        resized *= original / resized_sum
    return resized.astype(np.float32)


def _coarse_intensive(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(np.asarray(field, np.float32), target_hw[::-1],
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def _fractions(gas_volume: np.ndarray, aqueous_volume: np.ndarray,
               pore_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gas = np.divide(gas_volume, pore_volume, out=np.zeros_like(pore_volume), where=pore_volume > 0)
    aqueous = np.divide(aqueous_volume, pore_volume, out=np.zeros_like(pore_volume), where=pore_volume > 0)
    if np.any(gas + aqueous > 1.001):
        raise ValueError("Area-pooled gas and aqueous volumes exceed pore capacity")
    gas = np.clip(gas, 0, 1)
    return gas, np.clip(aqueous, 0, 1 - gas)


def _prepared_for(result_path: Path, result: dict) -> tuple[Path, dict]:
    if len(result_path.parents) < 4:
        raise ValueError("Expected saved concentration result under a series job")
    run = result_path.parent.name
    candidates = [parent / "preparation" / run / "prepared.json"
                  for parent in result_path.parents[1:5]]
    photograph_arrays = result["frames"][0].get("photograph_arrays")
    if photograph_arrays:
        candidates.insert(0, Path(photograph_arrays).parent.parent / "prepared.json")
    expected = result.get("prepared_manifest_sha256")
    prepared_path = next((path for path in candidates if path.is_file()
                          and (not expected or _sha(path) == expected)), None)
    if prepared_path is None:
        raise FileNotFoundError("No prepared input manifest matching the saved result hash: "
                                + ", ".join(map(str, candidates)))
    return prepared_path, json.loads(prepared_path.read_text(encoding="utf-8"))


def _preview(path: Path, data: dict, hidden: np.ndarray, observed: np.ndarray) -> None:
    rgb = data.get("photograph_rgb")
    if rgb is None:
        concentration = data["concentration_g_l"]
        level = np.clip(concentration / max(float(np.nanpercentile(concentration, 99)), 1e-6), 0, 1)
        base = np.stack((level * .15, level * .55, level), axis=-1)
    else:
        base = np.asarray(rgb)
    base = np.clip(base, 0, 1)
    base = (base * 255).astype(np.uint8)
    base = cv2.resize(base, hidden.shape[::-1], interpolation=cv2.INTER_AREA)
    observed_panel = base.copy()
    inferred_panel = base.copy()
    observed_panel[observed] = (255, 80, 180)
    inferred_panel[observed] = (255, 80, 180)
    inferred_panel[hidden] = (0, 235, 255)
    composite = np.concatenate((observed_panel, inferred_panel), axis=1)
    cv2.putText(composite, "Visible gas", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(composite, "Visible + hidden-gas hypothesis", (base.shape[1] + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write preview: {path}")


def run(result_path: str | Path, output: str | Path, *, width: int = 768,
        settings: HiddenGasDepthSettings | None = None,
        fixed_visible_saturation: float | None = None) -> dict:
    """Stream one frozen run into a causal scenario and immutable NAS sidecar."""
    result_path = Path(result_path).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("complete") is not True or not result.get("frames"):
        raise ValueError("A complete saved concentration result is required")
    prepared_path, prepared = _prepared_for(result_path, result)
    dimensions = prepared["dimensions_m"]
    native_hw = tuple(int(v) for v in prepared["shape_hw"])
    if isinstance(dimensions, dict):
        height_m, width_m = float(dimensions["height"]), float(dimensions["width"])
    else:
        height_m, width_m = map(float, dimensions)
    if not (0 < width <= native_hw[1]):
        raise ValueError("Analysis width must be positive and no greater than the saved grid")
    target_hw = (max(1, round(native_hw[0] * width / native_hw[1])), width)
    settings = settings or HiddenGasDepthSettings(causal_visible_saturation_enabled=True)
    if fixed_visible_saturation is not None and not 0 <= fixed_visible_saturation <= 1:
        raise ValueError("Fixed visible gas saturation must be in [0, 1]")
    facies = None
    if settings.persistent_halo_enabled and settings.halo_require_same_facies:
        references = Path(prepared["references"])
        if _sha(references) != prepared["references_sha256"]:
            raise ValueError("Registered facies reference changed")
        with np.load(references, allow_pickle=False) as source:
            if "facies" not in source or source["facies"].shape != native_hw:
                raise ValueError("Registered facies grid is missing or misaligned")
            facies = cv2.resize(source["facies"], target_hw[::-1],
                                interpolation=cv2.INTER_NEAREST).astype(np.int16)
    tracker = HiddenGasDepthTracker((height_m / target_hw[0], width_m / target_hw[1]), settings)
    output.mkdir(parents=True, exist_ok=True)
    frames_out = []
    preview_choice: tuple[float, Path, dict, np.ndarray, np.ndarray,
                          str | None, str | None] | None = None
    for frame in result["frames"]:
        fields_path = Path(frame["fields"])
        with np.load(fields_path, allow_pickle=False) as source:
            required = ("pore_volume_m3", "gas_volume_m3", "aqueous_volume_m3",
                        "gas_density_g_l", "concentration_g_l", "solubility_g_l")
            if any(key not in source for key in required):
                raise ValueError(f"Missing physical field in {fields_path}")
            native = {key: source[key] for key in required}
            preview_data = {key: source[key] for key in ("photograph_rgb", "concentration_g_l")
                            if key in source}
        if native["pore_volume_m3"].shape != native_hw:
            raise ValueError("Saved field differs from prepared grid")
        pore = _coarse_extensive(native["pore_volume_m3"], target_hw)
        gas = _coarse_extensive(native["gas_volume_m3"], target_hw)
        aqueous = _coarse_extensive(native["aqueous_volume_m3"], target_hw)
        gas_fraction, aqueous_fraction = _fractions(gas, aqueous, pore)
        density = _coarse_intensive(native["gas_density_g_l"], target_hw)
        concentration = _coarse_intensive(native["concentration_g_l"], target_hw)
        solubility = _coarse_intensive(native["solubility_g_l"], target_hw)
        dose = float(frame["nominal_injected_mass_g"])
        # The source deliberately stores no gas density before injection when
        # no physical state is needed. At later frames a corridor needs a
        # density at every candidate cell, including those labelled aqueous.
        if np.all(density > 0):
            scenario_density = density
        elif dose == 0 and not np.any(gas_fraction):
            scenario_density = None
        else:
            raise ValueError(f"Gas density is incomplete for scenario: {fields_path}")
        observed = tracker.update(visible_gas_pore_fraction=gas_fraction,
            aqueous_pore_fraction=aqueous_fraction, pore_volume_m3=pore,
            elapsed_h=float(frame["elapsed_h"]), injected_mass_g=dose,
            visible_gas_saturation=fixed_visible_saturation,
            gas_density_g_l=scenario_density, facies_labels=facies)
        hidden = observed.hidden_occupancy_scenario.astype(np.float64)
        density_factor = pore.astype(np.float64) * density.astype(np.float64) * 1000
        base_aqueous = float(np.sum(aqueous_fraction * pore * concentration * 1000,
                                    dtype=np.float64) + np.sum(
            gas_fraction * (1 - observed.visible_gas_saturation) * pore * solubility * 1000,
            dtype=np.float64))
        base_gas = float(np.sum(observed.observed_gas_occupancy * density_factor,
                                dtype=np.float64))
        # A scenario cannot spend more CO2 than has entered the rig. Hidden gas
        # displaces already counted dissolved CO2, so cap its NET mass change,
        # not just the extra gas mass (which the core tracker also bounds).
        hidden_net_mass = float(np.sum(hidden * pore * (density - concentration) * 1000,
                                       dtype=np.float64))
        mass_headroom = dose - base_gas - base_aqueous
        conservation_scale = (min(1., max(0., mass_headroom) / hidden_net_mass)
                              if hidden_net_mass > 0 else 1.)
        hidden *= conservation_scale
        scenario_gas = float(np.sum(observed.total_gas_occupancy_scenario * density_factor,
                                    dtype=np.float64))
        if conservation_scale != 1:
            scenario_gas = base_gas + float(np.sum(hidden * density_factor, dtype=np.float64))
        hidden_gas = float(np.sum(hidden * density_factor, dtype=np.float64))
        possible_hidden = observed.candidate_mask | observed.halo_candidate_mask
        max_hidden = float(np.sum(possible_hidden * aqueous_fraction * density_factor,
                                  dtype=np.float64))
        scenario_aqueous = base_aqueous - float(np.sum(
            hidden * pore * concentration * 1000, dtype=np.float64))
        gas_reference = frame.get("gas_reference") or None
        lower = (max(0., float(gas_reference["mass_g"]) - float(gas_reference["sigma_g"]))
                 if gas_reference else None)
        row = dict(frame_index=int(frame["frame_index"]), elapsed_h=float(frame["elapsed_h"]),
            nominal_injected_mass_g=dose, source_fields=str(fields_path),
            source_fields_sha256=_sha(fields_path),
            observed_gas_mass_g=float(frame["gas_mass_g"]),
            observed_aqueous_mass_g=float(frame["aqueous_mass_g"]),
            scenario_visible_gas_mass_g=observed.visible_gas_mass_g,
            visible_mass_over_injected_g=observed.visible_mass_over_injected_g,
            scenario_base_total_mass_g=base_gas + base_aqueous,
            base_total_mass_over_injected_g=max(0., -mass_headroom),
            hidden_mass_conservation_scale=conservation_scale,
            scenario_gas_mass_g=scenario_gas, hidden_gas_mass_g=hidden_gas,
            hidden_corridor_gas_mass_g=float(np.sum(
                observed.corridor_hidden_occupancy_scenario * conservation_scale * density_factor,
                dtype=np.float64)),
            hidden_halo_gas_mass_g=float(np.sum(
                observed.halo_hidden_occupancy_scenario * conservation_scale * density_factor,
                dtype=np.float64)),
            maximum_corridor_hidden_gas_mass_g=max_hidden,
            scenario_aqueous_mass_g=scenario_aqueous,
            scenario_total_mass_g=scenario_gas + scenario_aqueous,
            overflow_lower_g=lower,
            overflow_shortfall_after_scenario_g=(max(0., lower - scenario_gas) if lower is not None else None),
            overflow_shortfall_after_max_corridor_g=(max(0., lower - scenario_gas + hidden_gas - max_hidden)
                                                      if lower is not None else None),
            eligible_components=observed.eligible_component_count,
            ignored_components=observed.ignored_component_count,
            candidate_bridges=len(observed.bridges),
            candidate_cells=int(np.count_nonzero(possible_hidden)),
            halo_candidate_cells=int(np.count_nonzero(observed.halo_candidate_mask)),
            visible_gas_cells=int(np.count_nonzero(observed.visible_gas_mask)),
            full_depth_visible_gas_cells=int(np.count_nonzero(
                observed.visible_gas_mask & (observed.visible_gas_saturation == 1))),
            unsupported_visible_cells=int(np.count_nonzero(observed.unsupported_visible_gas_mask)),
            maximum_hidden_capacity_m3=observed.maximum_hidden_capacity_m3,
            hidden_gas_volume_scenario_m3=float(np.sum(hidden * pore, dtype=np.float64)),
            bridges=[asdict(bridge) for bridge in observed.bridges])
        frames_out.append(row)
        np.savez_compressed(output / f"hypothesis_{row['frame_index']:04d}.npz",
            visible_gas_mask=observed.visible_gas_mask,
            unsupported_visible_gas_mask=observed.unsupported_visible_gas_mask,
            candidate_mask=observed.candidate_mask,
            halo_candidate_mask=observed.halo_candidate_mask,
            halo_raw_support_score=observed.halo_raw_support_score,
            halo_support_score=observed.halo_support_score,
            visible_gas_saturation=observed.visible_gas_saturation,
            visible_gas_consecutive_observations=observed.visible_gas_consecutive_observations,
            hidden_occupancy_scenario=hidden.astype(np.float32),
            total_gas_occupancy_scenario=(observed.observed_gas_occupancy + hidden).astype(np.float32),
            persistence_h=observed.persistence_h)
        priority = (float(lower - float(frame["gas_mass_g"])) if lower is not None
                    else max_hidden)
        if preview_choice is None or priority > preview_choice[0]:
            preview_choice = (priority, output / "comparison.png", preview_data,
                              (hidden > 0).copy(), observed.visible_gas_mask.copy(),
                              frame.get("photograph_arrays"),
                              frame.get("photograph_arrays_sha256"))
    if preview_choice is not None:
        _, path, data, hidden_mask, visible_mask, photo_arrays, photo_digest = preview_choice
        if "photograph_rgb" not in data and photo_arrays:
            source = Path(photo_arrays)
            if source.is_file() and (photo_digest is None or _sha(source) == photo_digest):
                with np.load(source, allow_pickle=False) as image_source:
                    if "rgb" in image_source:
                        data["photograph_rgb"] = image_source["rgb"]
        _preview(path, data, hidden_mask, visible_mask)
    report = dict(format=FORMAT, complete=True, source_result=str(result_path),
        source_result_sha256=_sha(result_path), prepared_input=str(prepared_path),
        prepared_input_sha256=_sha(prepared_path),
        model_id=prepared.get("model_id"), dataset=prepared.get("dataset"),
        run=prepared.get("run"), scope=result.get("calibration", {}).get("scope"),
        phase_labels_modified=False, phase_inference_repeated=False,
        optical_calibration_refitted=False, overflow_used_to_construct_scenario=False,
        pilot_resolution_hw=list(target_hw), dimensions_m=dimensions,
        visible_saturation_policy=(
            f"fixed_{fixed_visible_saturation:.6g}" if fixed_visible_saturation is not None
            else "persistent_full_depth" if settings.causal_visible_full_depth_enabled
            else "causal_persistence" if settings.causal_visible_saturation_enabled
            else "unit_saturation"),
        fixed_visible_saturation_override=fixed_visible_saturation,
        settings=asdict(settings), frames=frames_out,
        limitations=[
            "A candidate corridor is a hypothesis, not observed 3D connectivity.",
            "The tested corridor only bridges nearby visible components; it does not search hidden pockets over the full plume.",
            "Visible-gas saturation and hidden-depth fraction are declared scenarios, not fitted measurements.",
            "The optional same-facies restriction is coarse; capillary entry pressure is not modeled.",
            "Overflow is an indirect gas constraint and is compared only after constructing the scenario.",
            "A saved calibration already exceeding cumulative injected mass cannot gain hidden gas under the mass-conservation cap.",
            "No held-out concentration or 3D measurement validates the scenario.",
        ])
    _json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--hidden-depth-fraction", type=float, default=.25)
    parser.add_argument("--max-gap-mm", type=float, default=20.)
    parser.add_argument("--fixed-visible-saturation", type=float)
    parser.add_argument("--visible-saturation-mode",
                        choices=("causal_persistence", "persistent_full_depth"),
                        default="causal_persistence")
    parser.add_argument("--visible-saturation-min", type=float, default=.6)
    parser.add_argument("--full-depth-required-observations", type=int, default=3)
    parser.add_argument("--full-depth-min-persistence-h", type=float, default=.5)
    parser.add_argument("--full-depth-max-observation-gap-h", type=float, default=1.)
    parser.add_argument("--full-depth-min-visible-gas-fraction", type=float, default=.5)
    parser.add_argument("--persistent-halo", action="store_true")
    parser.add_argument("--halo-only", action="store_true")
    parser.add_argument("--visible-only", action="store_true",
                        help="Evaluate visible-gas saturation without hidden corridor or halo")
    parser.add_argument("--halo-max-distance-mm", type=float, default=5.)
    parser.add_argument("--halo-depth-fraction", type=float, default=.25)
    parser.add_argument("--halo-support-exponent", type=float, default=1.)
    parser.add_argument("--halo-require-same-facies", action="store_true")
    args = parser.parse_args()
    if args.visible_only and (args.persistent_halo or args.halo_only):
        parser.error("--visible-only cannot be combined with hidden-gas halo options")
    settings = HiddenGasDepthSettings(
        causal_visible_saturation_enabled=(args.visible_saturation_mode == "causal_persistence"),
        causal_visible_full_depth_enabled=(args.visible_saturation_mode == "persistent_full_depth"),
        visible_saturation_min=args.visible_saturation_min,
        full_depth_required_observations=args.full_depth_required_observations,
        full_depth_min_persistence_h=args.full_depth_min_persistence_h,
        full_depth_max_observation_gap_h=args.full_depth_max_observation_gap_h,
        full_depth_min_visible_gas_fraction=args.full_depth_min_visible_gas_fraction,
        hidden_depth_fraction=args.hidden_depth_fraction,
        max_gap_m=args.max_gap_mm / 1000.,
        corridor_enabled=not (args.halo_only or args.visible_only),
        persistent_halo_enabled=args.persistent_halo or args.halo_only,
        halo_max_distance_m=args.halo_max_distance_mm / 1000.,
        halo_max_depth_fraction=args.halo_depth_fraction,
        halo_support_exponent=args.halo_support_exponent,
        halo_require_same_facies=args.halo_require_same_facies)
    result = run(args.result, args.output, width=args.width, settings=settings,
                 fixed_visible_saturation=args.fixed_visible_saturation)
    print(json.dumps(dict(output=str(args.output), dataset=result["dataset"],
                          run=result["run"], frames=len(result["frames"]))))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Blind optical-only versus generic physics-aware phase review."""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.dual_phase_signal import DualPhaseSignalModel  # noqa: E402
from darsia.fluidflower.tools import schema2_hierarchical_blind_campaign as base  # noqa: E402


_BASE_MANIFEST = base._manifest

CAMPAIGN_ID = "schema5_crossseries_reviewed_physics_blind_20260831"
V5_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'crossseries_phase_model_v5_reviewed_physics_20260831/models/generic'
MODEL_ORDER = ("reviewed_v5_optical", "reviewed_v5_physics")
GAS_MODEL_PATHS = {
    name: V5_ROOT / "gas_within_co2_signal.json" for name in MODEL_ORDER
}
CO2_MODEL_PATHS = {
    name: V5_ROOT / "co2_presence_signal.json" for name in MODEL_ORDER
}
RUNS = {"ffac": ("ac32", "ac61"), "ffum": ("run18", "run20")}
OUTPUTS = {
    "ffac": configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}',
    "ffum": configured_tool_path('ffum', 'analysis_v2') / f'{CAMPAIGN_ID}',
}
BLIND_SEED = "crossseries-reviewed-v5-optical-vs-physics-20260831"
TRAINING_RUNS = ("ac18", "ac52", "ac56", "run10", "run15", "run16")
INJECTION_REVIEW_FRACTIONS = (0.125, 0.375, 0.625, 0.875)

PHYSICS_SETTINGS = replace(
    base.transfer.CAUSAL_GAS_SETTINGS,
    use_plume_interior_prior=True,
    co2_support_level=0.20,
    co2_interior_scale_m=0.005,
    co2_boundary_floor=0.15,
    use_permeability_structure_prior=True,
    permeability_interface_scale_m=0.006,
    use_buoyancy_prior=True,
    post_injection_downward_tolerance_m=0.005,
    post_injection_downward_floor=0.15,
    # Facies identifiers are rig-specific. The transferable structural prior
    # therefore consumes the physical permeability field instead.
    reject_seal_dominant_components=False,
)


def _configure_base() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.MODEL_ROOT = V5_ROOT.parent
    base.MODEL_ORDER = MODEL_ORDER
    base.GAS_MODEL_PATHS = dict(GAS_MODEL_PATHS)
    base.CO2_MODEL_PATHS = dict(CO2_MODEL_PATHS)
    base.RUNS = dict(RUNS)
    base.OUTPUTS = dict(OUTPUTS)
    base.BLIND_SEED = BLIND_SEED
    base.transfer.INJECTION_REVIEW_FRACTIONS = INJECTION_REVIEW_FRACTIONS
    base.transfer.CAUSAL_GAS_SETTINGS_BY_CANDIDATE = {
        "reviewed_v5_physics": PHYSICS_SETTINGS,
    }


def _load_model_pairs() -> tuple[
    dict[str, DualPhaseSignalModel],
    dict[str, DualPhaseSignalModel],
]:
    gas = {
        name: DualPhaseSignalModel.load(GAS_MODEL_PATHS[name])
        for name in MODEL_ORDER
    }
    co2 = {
        name: DualPhaseSignalModel.load(CO2_MODEL_PATHS[name])
        for name in MODEL_ORDER
    }
    for name in MODEL_ORDER:
        if "gas_within_co2" not in str(
            gas[name].training_provenance.get("target_kind") or ""
        ):
            raise RuntimeError(f"{name} is not a conditional gas model")
        if "co2_presence" not in str(
            co2[name].training_provenance.get("target_kind") or ""
        ):
            raise RuntimeError(f"{name} is not a CO2-presence model")
        if not bool(gas[name].training_provenance.get("deployment_allowed")):
            raise RuntimeError(f"{name} gas model did not pass internal validation")
        if not bool(co2[name].training_provenance.get("deployment_allowed")):
            raise RuntimeError(f"{name} CO2 model did not pass internal validation")
    return gas, co2


def _manifest(
    dataset: str,
    root: Path,
    runs: Sequence[str],
    jobs: Sequence[Mapping[str, Any]],
    gas: Mapping[str, DualPhaseSignalModel],
    co2: Mapping[str, DualPhaseSignalModel],
) -> dict[str, Any]:
    payload = _BASE_MANIFEST(dataset, root, runs, jobs, gas, co2)
    payload["design"].update(
        {
            "comparison": (
                "same-reviewed-v5-optical-model-without-vs-with-generic-physics"
            ),
            "training_runs": list(TRAINING_RUNS),
            "target_runs_disjoint_from_training": not bool(
                set(runs) & set(TRAINING_RUNS)
            ),
            "target_used_for_refitting": False,
            "identical_learned_model_between_candidates": True,
            "physics_candidate": "reviewed_v5_physics",
            "physics_contract": {
                "free_gas_density_order": "rho_free_CO2_gas << rho_liquid_water",
                "directional_consequence": (
                    "after injection, newly descending free-gas support is penalized; "
                    "descending optical change remains available to aqueous CO2"
                ),
                "gas_inside_CO2_plume_prior": True,
                "relative_permeability_and_interface_geometry": True,
                "curved_interface_diagnostics": True,
                "active_injector_exception": True,
                "visible_2d_slice_reentry_exception": True,
                "hard_exclusion": False,
            },
            "physics_settings": dict(PHYSICS_SETTINGS.__dict__),
            "review_frame_selection": {
                "base_times_h": list(base.transfer.TARGET_TIMES_H[dataset]),
                "additional_injection_fractions_per_stage": list(
                    INJECTION_REVIEW_FRACTIONS
                ),
                "additional_injection_images_requested": (
                    len(INJECTION_REVIEW_FRACTIONS) * 2
                ),
                "pre_injection_reference_in_timeline": True,
                "pre_injection_reference_model_evaluated": True,
                "separate_aligned_reference_check_available": True,
            },
            "deployment_allowed": False,
            "deployment_reason": "blind review required",
        }
    )
    return payload


def _install_overrides() -> None:
    _configure_base()
    base._load_model_pairs = _load_model_pairs
    base._manifest = _manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("run", "preflight", "status"), nargs="?", default="run"
    )
    parser.add_argument("--dataset", choices=("ffac", "ffum", "all"), default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--force", action="store_true")
    base.transfer._add_review_pipeline_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    base.transfer._apply_review_pipeline_arguments(args)
    _install_overrides()
    datasets = ("ffac", "ffum") if args.dataset == "all" else (args.dataset,)
    results: dict[str, Any] = {}
    for dataset in datasets:
        if args.action == "status":
            results[dataset] = base._status(dataset, args.output_root)
        elif args.action == "preflight":
            results[dataset] = base.run_preflight(
                dataset, output_root=args.output_root, runs=args.runs
            )
        else:
            results[dataset] = base.run_campaign(
                dataset,
                output_root=args.output_root,
                runs=args.runs,
                force=args.force,
            )
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

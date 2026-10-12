"""Restore the separately prepared P1 reference role without changing operators.

The selected source wrapper still describes the current acquisition. Its
baseline_extended_scientific describes the separately prepared pre-injection
reference. The two can identify the same camera photograph while containing
different arrays, so source identity does not make their roles interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .phase_ffac_selected_inference_corrected import SelectedPhaseEvaluator as HistoricalSelectedPhaseEvaluator
from .phase_ffac_selected_inference import _source_path, _require_source
from .phase_ffac_selected_native_inputs_corrected import original_native_input, original_temporal_input, temporal_policy
from .phase_ffac_selected_phase import asset, bound_json, read_parent_envelopes
from .phase_ffac_rig_camera_runtime import bound_path
from .tools import run_ffac_rig_camera_inference as rig

CONTRACT = "ffac-selected-separate-pre-injection-baseline-role-v1"


class BaselineRoleInputError(ValueError):
    """A source-bound constructor failure; never an assertion waiver."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True)
class BaselineRoleInputs:
    native: object
    temporal: object
    full: np.ndarray
    full_valid: np.ndarray
    provenance: dict


def load_baseline_role_inputs(wrapper, native_reference, run):
    """Load the reference package, retaining every historical pixel check.

    This function never loads the wrapper's current native/full assets and
    never rewrites any source, valid mask, ring, or temporal convention.
    """
    reference = wrapper.get("baseline_extended_scientific")
    evidence = dict(contract=CONTRACT, run=run,
        current_source_sha256=wrapper.get("source_sha256"),
        reference_source_sha256=native_reference.get("source_sha256"),
        current_native=wrapper.get("native_scientific"),
        current_full=wrapper.get("full_extended_scientific"),
        reference_full=reference,
        historical_reference_log=dict(path=native_reference.get("v5_log_linear_rgb_path"),
            sha256=native_reference.get("v5_log_linear_rgb_sha256")),
        source_role="separately prepared pre-injection reference",
        historical_valid_pixel_assertion_waived=False)
    try:
        if not isinstance(reference, dict):
            raise ValueError("Selected source has no separately prepared scientific reference")
        if reference.get("source_sha256") != native_reference.get("source_sha256"):
            raise ValueError("Scientific reference identifies another pre-injection acquisition")
        geometry = reference.get("binding", {}).get("geometry", {})
        if geometry.get("run") != run or geometry.get("reference") is not True:
            raise ValueError("Scientific baseline is not the run's separately prepared reference role")
        scientific_native = np.load(bound_path(reference["native_centre"]["log_linear_rgb"]),
            allow_pickle=False, mmap_mode="r")
        native = original_native_input(native_reference, scientific_native_log=scientific_native)
        temporal = original_temporal_input(native, native_reference)
        full, full_valid = rig.compose_extended_scientific(reference)
    except Exception as error:
        evidence.update(failed=True, cause_type=type(error).__name__, cause=str(error))
        raise BaselineRoleInputError("Separately prepared baseline input failed: " + str(error), evidence) from error
    return BaselineRoleInputs(native, temporal, full, full_valid,
        dict(evidence, failed=False, current_acquisition_changed=False,
            previous_acquisition_changed=False, scientific_assets_changed=False,
            native_temporal_policy=temporal_policy(native_reference)))


class ReferenceSelectedPhaseEvaluator(HistoricalSelectedPhaseEvaluator):
    """The original evaluate/save methods with the correct constructor roles."""
    def __init__(self, root, plan, run, *, backend="cpu"):
        from .phase_ffac_selected_native_centre import NativeCentreEvaluator
        self.root, self.plan, self.run, self.backend = Path(root), plan, run, backend
        self.source_root = Path(plan.get("scientific_source_root", root))
        self.state = plan["runs"][run]
        self.inventory = bound_json(self.state["source_inventory"])
        self.sources = {row["source_sha256"]: row for row in self.inventory["sources"]}
        self.chain = plan["parent_chain"]
        self.native = bound_json(self.state["parent_native_manifest"])
        heads = self.chain["heads"]
        self.centre = NativeCentreEvaluator(plan["optical_candidate"], bound_path(heads["g3"]),
            bound_path(heads["regional"]), threads=1)
        self.envelopes = read_parent_envelopes(self.chain, run)
        self.floor_curve = np.load(bound_path(self.chain["fixed_floor_curve"]), allow_pickle=False)
        with np.load(bound_path(self.chain["fixed_zones"]), allow_pickle=False) as archive:
            self.zones = archive["zones"].copy()
        self.baseline_source = self.sources[self.inventory["baseline_source_sha256"]]
        # Preserve the current frame-0 wrapper. evaluate() continues loading
        # this acquisition as current/previous using the unchanged methods.
        self.baseline_receipt = bound_json(asset(_source_path(self.source_root, run,
            self.baseline_source["source_sha256"])))
        _require_source(self.baseline_receipt, self.baseline_source, run)
        reference = load_baseline_role_inputs(self.baseline_receipt, self.native["reference"], run)
        self.baseline_native, self.baseline_native_valid = reference.native.log, reference.native.valid
        self.baseline_full, self.baseline_valid = reference.full, reference.full_valid
        self.baseline_temporal, self.baseline_temporal_valid = reference.temporal.log, reference.temporal.valid
        self.baseline_origin = reference.temporal.origin_xy
        self.baseline_role_proof = reference.provenance
        self.baseline_temporal_proof = dict(reference.temporal.provenance,
            native_input=reference.native.provenance, baseline_role=reference.provenance)
        self.native_temporal_policy = temporal_policy(self.native["reference"])
        self._previous_cache = None


__all__ = ["ReferenceSelectedPhaseEvaluator", "load_baseline_role_inputs",
    "BaselineRoleInputs", "BaselineRoleInputError", "CONTRACT"]

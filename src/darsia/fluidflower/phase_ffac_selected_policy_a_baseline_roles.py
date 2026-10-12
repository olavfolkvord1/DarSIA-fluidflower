"""Policy A uses the same corrected reference constructor as CPU identity."""
from .phase_ffac_selected_baseline_roles import ReferenceSelectedPhaseEvaluator
from .phase_ffac_selected_policy_a_required_inference import RequiredConventionCentre, STAGE_CONVENTION_SOURCE_SHA256
from .phase_ffac_selected_required_identity import require_frozen_gas_audit
from .phase_ffac_selected_phase import bound_json
from .phase_ffac_rig_camera_runtime import bound_path


class RequiredSelectedPhaseEvaluator(ReferenceSelectedPhaseEvaluator):
    def __init__(self, root, plan, run, *, backend="cpu"):
        if backend != "cpu":
            raise ValueError("Policy A baseline-role inference is CPU only")
        super().__init__(root, plan, run, backend=backend)
        binding = plan["phase_campaign_binding"]
        convention = binding["inherited_stage_conventions"][run]
        for key in ("postflight_manifest", "context"):
            bound_path(convention[key])
        audit = bound_json(binding["frozen_gas_audit"])
        require_frozen_gas_audit(audit)
        if (audit.get("required_helper_binding_pending") is not False
                or binding["code"]["darsia.fluidflower.phase_ffac_selected_required_identity"]["sha256"] != STAGE_CONVENTION_SOURCE_SHA256):
            raise ValueError("Policy A lacks the exact independently bound historical stage helper")
        baseline_probability = self.baseline_source["parent_source"]["retained_source_cache"]["parent_probability_receipt"]
        bound_json(baseline_probability)
        self.centre = RequiredConventionCentre(self.centre, convention, baseline_probability)


__all__ = ["RequiredSelectedPhaseEvaluator"]

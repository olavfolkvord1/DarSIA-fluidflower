"""Required CPU replay with P1's separate pre-injection reference role.

Capture/comparison and the independently bound stage conventions are inherited
literally. Only constructor source roles change; frozen_gas remains diagnostic.
"""
from .phase_ffac_selected_baseline_roles import ReferenceSelectedPhaseEvaluator
from .phase_ffac_selected_required_identity import (
    RequiredIdentityEvaluator as HistoricalRequiredIdentityEvaluator,
    RequiredCapturedCentre, StageConventionCentre, require_frozen_gas_audit,
)
from .phase_ffac_selected_phase import bound_json
from .phase_ffac_rig_camera_runtime import bound_path


class RequiredIdentityEvaluator(HistoricalRequiredIdentityEvaluator):
    def __init__(self, root, plan, run):
        ReferenceSelectedPhaseEvaluator.__init__(self, root, plan, run, backend="cpu")
        convention = plan["phase_campaign_binding"]["inherited_stage_conventions"][run]
        for key in ("postflight_manifest", "context"):
            bound_path(convention[key])
        audit_descriptor = plan["phase_campaign_binding"]["frozen_gas_audit"]
        require_frozen_gas_audit(bound_json(audit_descriptor))
        self.frozen_gas_audit = audit_descriptor
        self.centre = RequiredCapturedCentre(StageConventionCentre(self.centre, convention))


__all__ = ["RequiredIdentityEvaluator"]

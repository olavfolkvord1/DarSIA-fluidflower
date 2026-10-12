"""Display-only selected photographs using the separately prepared reference.

Photographic transfers and current-frame loading are inherited unchanged. Only
the reference role is explicit; this module never evaluates a phase model.
"""
from pathlib import Path

from .phase_ffac_selected_display import SelectedDisplayRenderer
from .phase_ffac_selected_baseline_roles import load_baseline_role_inputs
from .phase_ffac_selected_phase import asset, bound_json
from .tools.ffac_track1_change_flatfield import prepare_flatfield_reference


class BaselineRoleSelectedDisplayRenderer(SelectedDisplayRenderer):
    def __init__(self, root, plan, run, *, backend="cpu"):
        self.root,self.plan,self.run,self.backend=Path(root),plan,run,backend
        self.source_root=Path(plan["scientific_source_root"])
        self.inventory=bound_json(plan["runs"][run]["source_inventory"])
        self.sources={row["source_sha256"]:row for row in self.inventory["sources"]}
        self.baseline_sha=self.inventory["baseline_source_sha256"]
        self.baseline_asset=asset(self.source_root/"selected_sources"/run/self.baseline_sha/"receipt.json")
        wrapper=bound_json(self.baseline_asset)
        native=bound_json(plan["runs"][run]["parent_native_manifest"])
        reference=load_baseline_role_inputs(wrapper,native["reference"],run)
        self.baseline=prepare_flatfield_reference(reference.full,source_valid_mask=reference.full_valid)
        self.origin=tuple(wrapper["rig_geometry"]["native_origin_xy"])
        self.baseline_reference_scientific=wrapper["baseline_extended_scientific"]
        self.baseline_role_proof=reference.provenance

    def binding(self, source, phase_asset, scientific_asset):
        return dict(super().binding(source,phase_asset,scientific_asset),
            baseline_reference_scientific=self.baseline_reference_scientific,
            baseline_reference_role="separately prepared historical pre-injection reference",
            baseline_role_proof=self.baseline_role_proof,baseline_role_display_code=asset(__file__),
            current_frame0_is_not_substituted=True)


__all__=["BaselineRoleSelectedDisplayRenderer"]

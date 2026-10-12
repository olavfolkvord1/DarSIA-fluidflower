"""Dispatch opt-in physics revisions without changing legacy runtime hashes."""
from . import phase_physics, phase_temporal, phase_plume, phase_plume_envelope, phase_presence_context
from . import phase_reference_support
from . import phase_reviewed_reference
from . import phase_facies_reference
from . import phase_native_reference


def load_physics_profile(root):
    from pathlib import Path
    if (Path(root) / "phase_selected_policy_a.json").exists():
        if any((Path(root) / name).exists() for name in ("phase_rig_camera.json", "phase_optical_regions.json")):
            raise ValueError("Selected Policy A review requires its own source-bound profile")
        from .phase_ffac_selected_policy_a_review import load_profile
        return load_profile(root)
    if (Path(root) / "phase_rig_camera.json").exists():
        if (Path(root) / "phase_optical_regions.json").exists():
            raise ValueError("Rig/camera revision requires its own master profile, not a stacked legacy optical profile")
        from .phase_ffac_rig_camera_runtime import load_profile
        return load_profile(root)
    from . import phase_optical_regions_runtime as optical_regions
    regional = optical_regions.load_profile(root)
    if regional is not None:
        from pathlib import Path
        track1_scope = regional.get("v5_track1_review_scope")
        if track1_scope is not None:
            if (track1_scope != {
                    "contract": "ffac-v5-track1-photo-only-review-context-v1",
                    "facies_ids_used": False,
                    "physical_scale_qa_pass": False,
                    "absolute_position_rules_enabled": False,
                    "mass_analysis_allowed": False,
                } or any(key in regional for key in (
                    "bottom_gas_guard", "region_edge_guard", "output_guards",
                    "aqueous_temporal_support"))):
                raise ValueError("Track1 photo-only review cannot use physical or temporal guards")
        nominal_review = regional.get("v5_nominal_review_gas_minimum")
        if nominal_review is not None:
            from dataclasses import asdict
            from .phase_free_gas_output_guard import Settings as GasSettings
            expected = {
                "contract": "ffac-v5-nominal-review-gas-minimum-v1",
                "minimum_area_mm2": 2.5,
                "physical_scale_qa_pass": False,
                "mass_analysis_allowed": False,
            }
            if (nominal_review != expected
                    or regional.get("free_gas_cleanup") != asdict(GasSettings())
                    or any(key in regional for key in (
                        "bottom_gas_guard", "output_guards", "region_edge_guard"))):
                raise ValueError("Invalid V5 nominal review-only gas minimum")
        if "output_guards" in regional:
            from .phase_region_output_guards import Settings
            Settings(**regional["output_guards"])
        if "free_gas_cleanup" in regional:
            from .phase_free_gas_output_guard import Settings
            Settings(**regional["free_gas_cleanup"])
        if "bottom_gas_guard" in regional:
            from .phase_bottom_gas_guard import Settings
            Settings(**regional["bottom_gas_guard"])
        if "region_edge_guard" in regional:
            from .phase_region_edge_guard import Settings
            Settings(**regional["region_edge_guard"])
        if "aqueous_temporal_support" in regional:
            runtime = _aqueous_temporal_runtime(regional["aqueous_temporal_support"])
            runtime.Settings(**regional["aqueous_temporal_support"])
        if "regional_gas" in regional:
            from .phase_regional_gas import Settings
            Settings(**regional["regional_gas"])
        if "approved_g3_gas" in regional:
            from .phase_g3_production import load_approved_profile
            regional = load_approved_profile(root, regional)
        if "photo_sand_top_revision" in regional:
            from .phase_photo_sand_top_runtime import load_top_revision_profile
            regional = load_top_revision_profile(root, regional)
        if "gas_barrier_rule_b" in regional:
            from .phase_gas_barrier_runtime import load_barrier_profile
            regional = load_barrier_profile(root, regional)
        if any((Path(root) / name).exists() for name in (
                "phase_native_reference.json", "phase_facies_reference.json", "phase_reviewed_reference.json",
                "phase_reference_support.json", "phase_presence_context.json", "phase_physics.json",
                "phase_plume_support.json", "phase_aqueous_context_completion.json")):
            raise ValueError("The optical region model cannot silently stack legacy context completion")
        return regional
    from .phase_decision_versions import attach_profile as attach_decision
    from .phase_aqueous_context_completion import attach_profile
    from .phase_local_aqueous_support import attach_profile as attach_local
    return attach_decision(root, attach_local(root, attach_profile(root, _load_base_physics_profile(root))))


def _load_base_physics_profile(root):
    native = phase_native_reference.load_profile(root)
    if native is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_facies_reference.json", "phase_reviewed_reference.json", "phase_reference_support.json",
                "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Native reference contains its inherited support profiles")
        return native
    facies = phase_facies_reference.load_profile(root)
    if facies is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_reviewed_reference.json", "phase_reference_support.json", "phase_presence_context.json",
                "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Facies reference contains its inherited support profiles")
        return facies
    reviewed = phase_reviewed_reference.load_profile(root)
    if reviewed is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_reference_support.json", "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Reviewed reference contains its inherited support profiles")
        return reviewed
    reference = phase_reference_support.load_profile(root)
    if reference is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in (
                "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Reference support already contains its inherited presence and gas profiles")
        return reference
    presence = phase_presence_context.load_profile(root)
    if presence is not None:
        from pathlib import Path
        if any((Path(root) / name).exists() for name in ("phase_physics.json", "phase_plume_support.json")):
            raise ValueError("Presence profile already contains its inherited gas profile")
        return presence
    legacy = phase_physics.load_physics_profile(root)
    plume = phase_plume_envelope.load_profile(root)
    if plume is None:
        plume = phase_plume.load_profile(root)
    if legacy is not None and plume is not None:
        raise ValueError("Choose one phase-support profile; legacy and plume rules cannot be stacked")
    return plume if plume is not None else legacy


def _new(profile):
    return profile is not None and profile.get("format") in (phase_plume.VERSION, phase_plume_envelope.VERSION)


def _aqueous_temporal_runtime(specification):
    if not isinstance(specification, dict):
        raise ValueError("Aqueous temporal-support profile must be an object")
    from . import phase_aqueous_temporal_support as first

    if specification.get("format") == first.VERSION:
        return first
    from . import phase_aqueous_source_history as second

    if specification.get("format") == second.VERSION:
        return second
    raise ValueError("Unknown aqueous temporal-support format")


def _boundary_runtime(profile):
    from . import phase_aqueous_boundary_runtime
    specification = profile.get("boundary_refinement")
    if isinstance(specification, dict) and specification.get("format") == "native-aqueous-boundary-v3":
        from . import phase_aqueous_boundary_v3_runtime
        return phase_aqueous_boundary_v3_runtime
    if isinstance(specification, dict) and specification.get("format") == "signal-backed-aqueous-continuity-v3":
        from . import phase_aqueous_continuity
        return phase_aqueous_continuity
    return phase_aqueous_boundary_runtime


def physics_runtime_identity(profile):
    if profile is not None and profile.get("format") == "ffac-selected-policy-a-review-physics-v1":
        from .phase_ffac_selected_policy_a_review import runtime_identity
        return runtime_identity(profile)
    if profile is not None and profile.get("format") == "ffac-rig-camera-physics-v1":
        from .phase_ffac_rig_camera_runtime import runtime_identity
        return runtime_identity(profile)
    if profile is not None and profile.get("format") == "baseline-relative-optical-regions-v1":
        from . import phase_optical_regions_runtime as optical_regions
        identity = optical_regions.runtime_identity(profile)
        if "approved_g3_gas" in profile:
            from .phase_g3_production import approved_runtime_identity
            identity = approved_runtime_identity(identity, profile["approved_g3_gas"])
            if "photo_sand_top_revision" in profile:
                from .phase_photo_sand_top_runtime import top_revision_runtime_identity
                identity = top_revision_runtime_identity(identity, profile["photo_sand_top_revision"])
            if "gas_barrier_rule_b" in profile:
                from .phase_gas_barrier_runtime import barrier_runtime_identity
                identity = barrier_runtime_identity(identity, profile["gas_barrier_rule_b"])
            return identity
        if "regional_gas" in profile:
            from .phase_regional_gas import runtime_identity
            identity = runtime_identity(identity, profile["regional_gas"])
        if "output_guards" in profile:
            from .phase_region_output_guards import runtime_identity
            identity = runtime_identity(identity, profile["output_guards"])
        if "free_gas_cleanup" in profile:
            from .phase_free_gas_output_guard import runtime_identity
            identity = runtime_identity(identity, profile["free_gas_cleanup"])
        if "bottom_gas_guard" in profile:
            from .phase_bottom_gas_guard import runtime_identity
            identity = runtime_identity(identity, profile["bottom_gas_guard"])
        if "region_edge_guard" in profile:
            from .phase_region_edge_guard import runtime_identity
            identity = runtime_identity(identity, profile["region_edge_guard"])
        if "aqueous_temporal_support" in profile:
            runtime = _aqueous_temporal_runtime(profile["aqueous_temporal_support"])
            identity = runtime.runtime_identity(identity, profile["aqueous_temporal_support"])
        return identity
    if profile is not None and "unified_phase_decision" in profile:
        from . import phase_decision_versions as decision
        inherited,specification=decision.split_profile(profile)
        return decision.runtime_identity(physics_runtime_identity(inherited),specification)
    if profile is not None and "local_aqueous_support" in profile:
        from . import phase_local_aqueous_support as local
        inherited, specification = local.split_profile(profile)
        return local.runtime_identity(physics_runtime_identity(inherited), specification)
    if profile is not None and "aqueous_context_completion" in profile:
        from . import phase_aqueous_context_completion as completion
        inherited, specification = completion.split_profile(profile)
        return completion.runtime_identity(physics_runtime_identity(inherited), specification)
    if profile is not None and "boundary_refinement" in profile:
        runtime = _boundary_runtime(profile)
        inherited, settings = runtime.split_profile(profile)
        return runtime.runtime_identity(physics_runtime_identity(inherited), settings)
    if profile is not None and profile.get("format") == phase_native_reference.VERSION:
        return phase_native_reference.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_facies_reference.VERSION:
        return phase_facies_reference.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_reviewed_reference.VERSION:
        return phase_reviewed_reference.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_reference_support.VERSION:
        return phase_reference_support.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_presence_context.VERSION:
        return phase_presence_context.runtime_identity(profile)
    if profile is not None and profile.get("format") == phase_plume_envelope.VERSION:
        return phase_plume_envelope.runtime_identity(profile)
    return phase_plume.runtime_identity(profile) if _new(profile) else phase_temporal.physics_runtime_identity(profile)


def load_run_context(root, profile, dataset, run, *, native_manifest_sha256):
    if profile is not None and profile.get("format") == "ffac-selected-policy-a-review-physics-v1":
        from .phase_ffac_selected_policy_a_review import require_dedicated_inference
        require_dedicated_inference()
    if profile is not None and profile.get("format") == "ffac-rig-camera-physics-v1":
        from .phase_ffac_rig_camera_runtime import require_dedicated_inference
        require_dedicated_inference()
    if profile is not None and profile.get("format") == "baseline-relative-optical-regions-v1":
        return phase_plume.load_context(root, profile, dataset, run,
                                       native_manifest_sha256=native_manifest_sha256)
    if profile is not None and "unified_phase_decision" in profile:
        from .phase_decision_runtime import split_profile
        profile,_=split_profile(profile)
    if profile is not None and "local_aqueous_support" in profile:
        from .phase_local_aqueous_support import split_profile
        profile, _ = split_profile(profile)
    if profile is not None and "aqueous_context_completion" in profile:
        from .phase_aqueous_context_completion import split_profile
        profile, _ = split_profile(profile)
    if profile is not None and "boundary_refinement" in profile:
        profile, _ = _boundary_runtime(profile).split_profile(profile)
    if profile is not None and profile.get("format") in (phase_presence_context.VERSION, phase_reference_support.VERSION, phase_reviewed_reference.VERSION, phase_facies_reference.VERSION, phase_native_reference.VERSION):
        return phase_plume.load_context(root, profile, dataset, run, native_manifest_sha256=native_manifest_sha256)
    loader = phase_plume.load_context if _new(profile) else phase_physics.load_run_context
    return loader(root, profile, dataset, run, native_manifest_sha256=native_manifest_sha256)


def phase_physics_sequence(context, profile):
    if profile is not None and profile.get("format") == "ffac-selected-policy-a-review-physics-v1":
        from .phase_ffac_selected_policy_a_review import require_dedicated_inference
        require_dedicated_inference()
    if profile is not None and profile.get("format") == "ffac-rig-camera-physics-v1":
        from .phase_ffac_rig_camera_runtime import require_dedicated_inference
        require_dedicated_inference()
    if profile is not None and profile.get("format") == "baseline-relative-optical-regions-v1":
        from . import phase_optical_regions_runtime as optical_regions
        if "approved_g3_gas" in profile:
            if "gas_barrier_rule_b" in profile:
                from .phase_gas_barrier_runtime import GasBarrierSequence
                return GasBarrierSequence(context, profile)
            if "photo_sand_top_revision" in profile:
                from .phase_photo_sand_top_runtime import PhotoSandTopSequence
                return PhotoSandTopSequence(context, profile)
            from .phase_g3_production import ApprovedG3GasSequence
            return ApprovedG3GasSequence(context, profile)
        sequence = optical_regions.OpticalRegionSequence(context, profile)
        if "regional_gas" in profile:
            from .phase_regional_gas import RegionalGasSequence
            sequence = RegionalGasSequence(context, profile)
        if "output_guards" in profile:
            from .phase_region_output_guards import GuardedRegionSequence
            sequence = GuardedRegionSequence(sequence, context, profile["output_guards"])
        if "free_gas_cleanup" in profile:
            from .phase_free_gas_output_guard import FreeGasSequence
            sequence = FreeGasSequence(sequence, context, profile["free_gas_cleanup"])
        if "bottom_gas_guard" in profile:
            from .phase_bottom_gas_guard import BottomGasSequence
            sequence = BottomGasSequence(sequence, context, profile["bottom_gas_guard"])
        if "region_edge_guard" in profile:
            from .phase_region_edge_guard import RegionEdgeSequence
            sequence = RegionEdgeSequence(sequence, context, profile["region_edge_guard"])
        if "aqueous_temporal_support" in profile:
            runtime = _aqueous_temporal_runtime(profile["aqueous_temporal_support"])
            cls = getattr(runtime, "AqueousTemporalSequence", None)
            if cls is None:
                cls = runtime.AqueousSourceHistorySequence
            sequence = cls(sequence, context, profile["aqueous_temporal_support"])
        return sequence
    if profile is not None and "unified_phase_decision" in profile:
        from . import phase_decision_versions as decision, phase_aqueous_context_completion as completion
        inherited,specification=decision.split_profile(profile)
        conservative,checkpoint=completion.split_profile(inherited)
        from . import phase_regional_runtime as regional
        if specification.get("format") == regional.VERSION:
            return regional.create_sequence(context, conservative, specification, checkpoint)
        # B does not wrap the final map. Its fitted network is an evidence
        # provider evaluated inside the one decision entry point.
        return decision.DecisionSequence(context,phase_physics_sequence(context,conservative),specification,checkpoint)
    if profile is not None and "local_aqueous_support" in profile:
        from . import phase_local_aqueous_support as local
        inherited, specification = local.split_profile(profile)
        return local.LocalAqueousSequence(phase_physics_sequence(context, inherited), specification)
    if profile is not None and "aqueous_context_completion" in profile:
        from . import phase_aqueous_context_completion as completion
        inherited, specification = completion.split_profile(profile)
        return completion.AqueousCompletionSequence(phase_physics_sequence(context, inherited), specification)
    if profile is not None and "boundary_refinement" in profile:
        runtime = _boundary_runtime(profile)
        inherited, settings = runtime.split_profile(profile)
        return runtime.AqueousBoundarySequence(phase_physics_sequence(context, inherited), settings)
    if profile is not None and profile.get("format") == phase_native_reference.VERSION:
        return phase_native_reference.NativeReferenceSequence(context, profile)
    if profile is not None and profile.get("format") == phase_facies_reference.VERSION:
        return phase_facies_reference.FaciesReferenceSequence(context, profile)
    if profile is not None and profile.get("format") == phase_reviewed_reference.VERSION:
        return phase_reviewed_reference.ReviewedReferenceSequence(context, profile)
    if profile is not None and profile.get("format") == phase_reference_support.VERSION:
        return phase_reference_support.ReferenceSequence(context, profile)
    if profile is not None and profile.get("format") == phase_presence_context.VERSION:
        return phase_presence_context.PresenceSequence(context, profile)
    if _new(profile):
        sequence = (phase_plume_envelope.LocalFloorSequence
                    if profile.get("format") == phase_plume_envelope.VERSION else phase_plume.PlumeSequence)
        return sequence(context, phase_plume.PlumeSettings(**profile.get("settings", {})))
    return phase_temporal.phase_physics_sequence(context, profile)


def _native_regional_chain(sequence):
    from .phase_optical_regions_runtime import OpticalRegionSequence
    from .phase_region_output_guards import GuardedRegionSequence
    from .phase_free_gas_output_guard import FreeGasSequence
    from .phase_bottom_gas_guard import BottomGasSequence
    from .phase_region_edge_guard import RegionEdgeSequence
    supported_wrappers = (GuardedRegionSequence, FreeGasSequence,
                          BottomGasSequence, RegionEdgeSequence)
    nodes = []
    current = sequence
    while isinstance(current, supported_wrappers):
        nodes.append(current)
        current = current.inherited
    if not isinstance(current, OpticalRegionSequence):
        raise ValueError("Per-frame native geometry is unsupported for this phase sequence")
    nodes.append(current)
    return nodes


def bind_native_frame_context(sequence, context):
    """Move the frame-local regional chain to one verified native image geometry.

    A run-level context is insufficient when DarSIA's chart-based drift moves
    the sand within a run.  Keep the optical reference and fitted weights, but
    replace every spatial mask/coordinate field before this frame is decided.
    Unknown or stateful sequence types fail closed rather than silently using
    the previous frame's porous domain.
    """
    from .phase_bottom_gas_guard import BottomGasSequence, _prepare_geometry
    if not isinstance(context, phase_plume.PlumeContext):
        raise TypeError("Native frame context must be a validated PlumeContext")
    nodes = _native_regional_chain(sequence)
    old = nodes[0].context
    if (context.domain.shape != old.domain.shape
            or context.injection_stages != old.injection_stages):
        raise ValueError("Native frame changed the grid or injection protocol")
    prepared = _prepare_geometry(context) if any(
        isinstance(node, BottomGasSequence) for node in nodes) else None
    for node in nodes:
        node.context = context
        if isinstance(node, BottomGasSequence):
            node._prepared_geometry = prepared


def bind_native_frame_reference(sequence, baseline):
    """Bind the same source-aligned reference used by the optical frame head."""
    optical = _native_regional_chain(sequence)[-1]
    optical.bind_native_frame_reference(baseline)

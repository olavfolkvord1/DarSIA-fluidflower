"""One registered phase decision from independent evidence providers.

No provider writes the final labels. Learned completion is a proposal, reference
change is regional evidence, and causal anchors must come from observations
independent of completion. This is an explicit new method; legacy maps and
their runtime identities are not modified. Scores are not concentrations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

VERSION = "unified-phase-decision-v1"


@dataclass(frozen=True)
class Settings:
    working_spacing_m: float = .001
    reference_smoothing_m: float = .002
    minimum_chromatic_change: float = .006
    noise_multiplier: float = 3.
    minimum_direction_cosine: float = .5
    reference_search_m: float = .025
    maximum_reference_evidence_weight: float = 3.
    minimum_controls: int = 64
    minimum_direction_samples: int = 12
    independent_observation_score: float = 3.
    minimum_observation_area_m2: float = 2e-6
    contour_smoothing_m: float = .003
    contour_outward_allowance_m: float = .001
    contour_band_m: float = .006
    contour_reference_floor: float = .25
    secure_core_distance_m: float = .006

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite decision setting required: {name}")
        if self.minimum_direction_cosine > 1:
            raise ValueError("Direction cosine exceeds one")
        if self.contour_band_m < self.contour_outward_allowance_m:
            raise ValueError("Contour allowance exceeds its physical uncertainty band")
        for key in ("minimum_controls", "minimum_direction_samples"):
            if int(getattr(self, key)) != getattr(self, key):
                raise ValueError(f"Integer sample count required: {key}")


@dataclass(frozen=True)
class Evidence:
    """Native registered observations; completion must never supply anchors.

    Conservative labels are an evidence provider's proposal, not protected
    final labels. Native gas comes from the existing gas evidence provider.
    The proposal adapter must document its independent causal history.
    """
    current: np.ndarray
    baseline: np.ndarray
    conservative_labels: np.ndarray
    completion_probability: np.ndarray
    anchors: np.ndarray
    domain: np.ndarray
    facies: np.ndarray
    spacing_m: tuple[float, float]
    elapsed_h: float
    pre_injection: bool
    provenance: dict = field(default_factory=dict)
    independent_spatial_observation: np.ndarray | None = None


@dataclass(frozen=True)
class PhaseResult:
    labels: np.ndarray
    diagnostics: dict[str, np.ndarray]
    report: dict


def identity(settings=Settings()):
    return hashlib.sha256(json.dumps(dict(format=VERSION, settings=asdict(settings),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()


def _check(e):
    dom = np.asarray(e.domain)
    shape = dom.shape
    if dom.dtype != bool or dom.ndim != 2 or min(shape) < 2 or not dom.any():
        raise ValueError("Nonempty registered physical domain required")
    if not np.isfinite(e.elapsed_h):
        raise ValueError("Finite experiment time required")
    for rgb in (e.current, e.baseline):
        if np.asarray(rgb).shape != (*shape, 3) or np.asarray(rgb).dtype != np.uint8:
            raise ValueError("Registered native uint8 current and baseline images required")
    for a in (e.conservative_labels, e.anchors, e.facies):
        if np.asarray(a).shape != shape or not np.isfinite(a).all():
            raise ValueError("All evidence must use the same native physical grid")
    if not np.isin(e.conservative_labels, (0, 1, 2)).all() or np.asarray(e.anchors).dtype != bool:
        raise ValueError("Invalid conservative phase labels or independent anchor mask")
    if np.any(e.anchors & ~dom):
        raise ValueError("Causal anchors cannot lie outside the physical domain")
    if e.independent_spatial_observation is not None:
        observed=np.asarray(e.independent_spatial_observation)
        if observed.shape!=shape or observed.dtype!=bool:
            raise ValueError("Independent observations must use the registered native boolean grid")
    if len(e.spacing_m) != 2 or not np.isfinite(e.spacing_m).all() or min(e.spacing_m) <= 0:
        raise ValueError("Positive registered pixel spacing required")
    q = np.asarray(e.completion_probability)
    if q.ndim != 2 or not np.isfinite(q).all() or q.min() < 0 or q.max() > 1:
        raise ValueError("Finite unit-interval completion scores required")
    return shape


def anchored(mask, anchors):
    """Multiple sources allowed; no implicit diagonal crossing or size exemption."""
    components, n = ndi.label(mask)
    accepted = np.zeros(n+1, bool)
    accepted[np.unique(components[anchors & mask])] = True
    accepted[0] = False
    return accepted[components]


def joint_reach(proposals, reference, anchors, domain, spacing, maximum_reference_path_m,
                *, reference_strength=None, maximum_evidence_weight=1.):
    """One path budget through reference evidence, with model-supported spans.

    Crossing a reference-backed gap can connect a legitimate weak model region.
    The gap consumes a cumulative evidence-weighted path budget; encountering a
    new proposal does not reset it. Strong regional reference change reduces
    cost up to a fixed cap. This is not a physical transport-distance model.
    A path cannot cross a no-evidence pixel.
    """
    starts=anchors & proposals & domain
    if not starts.any():return np.zeros_like(domain)
    edge=starts & ~ndi.binary_erosion(starts)
    strength=(np.ones_like(domain,dtype=np.float32) if reference_strength is None
              else np.clip(np.asarray(reference_strength,np.float32),1.,maximum_evidence_weight))
    costs=np.where(domain & (proposals|reference),np.where(proposals,0.,1./strength),np.inf)
    distance,_=MCP_Geometric(costs,fully_connected=False,sampling=spacing).find_costs(np.argwhere(edge))
    return domain & (distance<=maximum_reference_path_m)


def reference_evidence(current, baseline, conservative, domain, facies, spacing, settings, *, occupied=None):
    """Preserve the reviewed local-search optical method as an evidence provider.

    Inputs are already area-pooled RGB. Regional chromatic change is smoothed
    within sand types exactly as the existing local search, so this architecture
    change does not silently replace its successful weak-signal response.
    """
    sigma = tuple(settings.reference_smoothing_m/s for s in spacing)
    score = np.zeros(domain.shape, np.float32)
    available = np.zeros_like(domain)
    records = []
    occupied = conservative if occupied is None else occupied
    near = ndi.distance_transform_edt(~occupied, sampling=spacing) <= settings.reference_search_m
    chroma = [rgb/np.maximum(rgb.sum(-1,keepdims=True),.08) for rgb in (current,baseline)]
    delta = chroma[0]-chroma[1]
    for label in np.unique(facies[domain]):
        mask = domain & (facies == label)
        weight = ndi.gaussian_filter(mask.astype(np.float32), sigma)
        smooth = np.stack([ndi.gaussian_filter(delta[...,c]*mask,sigma)/np.maximum(weight,1e-6)
                           for c in range(3)],axis=-1)
        controls = mask & ~near
        core = mask & conservative
        row = dict(facies=int(label), controls=int(controls.sum()), teaching_pixels=int(core.sum()))
        if controls.sum() < settings.minimum_controls or core.sum() < settings.minimum_direction_samples:
            records.append({**row, "status":"insufficient_independent_controls_or_direction"}); continue
        drift = np.median(smooth[controls],axis=0)
        residual = smooth-drift
        direction = np.median(residual[core],axis=0)
        length = float(np.linalg.norm(direction))
        if length < settings.minimum_chromatic_change:
            records.append({**row,"status":"no_measurable_direction"}); continue
        direction /= length
        projection = residual @ direction
        samples = projection[controls]
        noise = 1.4826*float(np.median(np.abs(samples-np.median(samples))))
        threshold = max(settings.minimum_chromatic_change,settings.noise_multiplier*noise)
        cosine = projection/np.maximum(np.linalg.norm(residual,axis=-1),1e-8)
        usable = mask & (cosine >= settings.minimum_direction_cosine)
        score[usable] = np.maximum(projection[usable],0)/threshold
        available |= mask
        records.append({**row,"status":"available","noise":noise,"threshold":threshold,
                        "drift":drift.tolist(),"direction":direction.tolist()})
    return score, available, records


def regularized_field(proposal, secure, domain, spacing, settings, *, expansion_support=None):
    """Fit a smooth implicit contour inside an explicit uncertainty band.

    A signed-distance field is regularized at a physical scale before a single
    final native-grid decision. The old pixel fringe is NOT unioned back in.
    Separate supported components keep a watershed separator, so smoothing
    cannot invent a bridge across a visible water channel.
    """
    inside = ndi.distance_transform_edt(proposal,sampling=spacing)
    outside = ndi.distance_transform_edt(~proposal,sampling=spacing)
    signed = inside-outside
    smooth = ndi.gaussian_filter(signed,tuple(settings.contour_smoothing_m/s for s in spacing))
    smooth += settings.contour_outward_allowance_m
    # Confidence applies to interior regions, never to every old boundary pixel.
    core = secure & (inside >= settings.secure_core_distance_m)
    smooth[core] = np.maximum(smooth[core],signed[core])
    allowed = domain & (outside <= settings.contour_band_m)
    if expansion_support is not None:
        # Curvature cannot fill optically unchanged water between two nearby
        # plumes. The weaker evidence floor only governs new contour expansion;
        # it never deletes an existing supported proposal. Missing reference
        # controls are distinguished from an available near-zero measurement.
        allowed &= proposal | expansion_support
    components,n = ndi.label(proposal)
    # A real, separately observed small object must not vanish because the
    # regional curvature scale is larger than the object itself. Its contour
    # still comes from a smooth implicit field, with a scale bounded by its
    # inradius. Unsupported model islands have already been excluded upstream.
    radii=ndi.maximum(inside,components,np.arange(1,n+1))
    for label,(region,radius) in enumerate(zip(ndi.find_objects(components),radii),start=1):
        if region is None or radius>=2*settings.contour_smoothing_m:continue
        sigma_m=max(min(spacing)*.5,min(settings.contour_smoothing_m,radius*.5))
        padding=int(np.ceil((4*sigma_m+settings.contour_band_m)/min(spacing)))
        box=tuple(slice(max(0,s.start-padding),min(limit,s.stop+padding)) for s,limit in zip(region,domain.shape))
        local=components[box]==label
        distance=ndi.distance_transform_edt(~local,sampling=spacing)
        near=distance<=settings.contour_band_m
        own=near & ((components[box]==0)|local)
        fitted=ndi.gaussian_filter(signed[box],tuple(sigma_m/s for s in spacing))+settings.contour_outward_allowance_m
        smooth[box][own]=np.maximum(smooth[box][own],fitted[own])
    if n > 1:
        nearest = ndi.distance_transform_edt(~proposal,return_distances=False,return_indices=True)
        owner = components[tuple(nearest)]
        separator = np.zeros_like(domain)
        separator[1:] |= (owner[1:] != owner[:-1]) & (owner[1:] > 0) & (owner[:-1] > 0)
        separator[:,1:] |= (owner[:,1:] != owner[:,:-1]) & (owner[:,1:] > 0) & (owner[:,:-1] > 0)
        allowed &= ~(separator & ~proposal)
    smooth[~allowed] = np.minimum(smooth[~allowed],-max(spacing))
    return smooth.astype(np.float32), core


MINIMUM_FREE_GAS_AREA_M2 = 2.5e-6
"""Smallest free-gas component that may be reported as a separate body of gas."""


def withdraw_small_free_gas(labels, dy, dx, minimum_area_m2=MINIMUM_FREE_GAS_AREA_M2):
    """Reclassify free-gas components below the minimum area as aqueous CO2.

    The free-gas head marks individual pixels: measured on 2026-09-19 the median free-gas
    component is under two native pixels. A speck that small is not a body of gas, but it is
    still CO2, so it becomes aqueous and never water -- the CO2 decision cannot move. Area is
    the only test here; a cluster of bright sand grains larger than the threshold is not
    addressed, and needs evidence over time rather than size.
    """
    if minimum_area_m2 <= 0:
        return labels
    cell = abs(float(dy)) * abs(float(dx))
    if not np.isfinite(cell) or cell <= 0:
        return labels
    pieces, count = ndi.label(labels == 2)
    if not count:
        return labels
    sizes = np.bincount(pieces.ravel()) * cell
    small = sizes < float(minimum_area_m2)
    small[0] = False
    if not small.any():
        return labels
    withdrawn = labels.copy()
    withdrawn[small[pieces]] = 1
    return withdrawn


def decide(evidence: Evidence, settings=Settings()) -> PhaseResult:
    shape = _check(evidence); e = evidence
    empty = np.zeros(shape,np.uint8)
    base_report = dict(format=VERSION, method_identity=identity(settings),settings=asdict(settings),
                       elapsed_h=e.elapsed_h,provenance=e.provenance,
                       history_updates=False,probabilities_calibrated=False)
    if e.pre_injection:
        empty.setflags(write=False)
        return PhaseResult(empty,{},dict(base_report,reason="pre_injection",co2_pixels=0))
    h,w=shape;dy,dx=e.spacing_m
    target=(max(2,round(w*dx/settings.working_spacing_m)),max(2,round(h*dy/settings.working_spacing_m)))
    step=(h*dy/target[1],w*dx/target[0])
    area=lambda a: cv2.resize(np.asarray(a,np.float32),target,interpolation=cv2.INTER_AREA)
    # Include partial boundary cells while solving the coarse field. Applying
    # a near-full occupancy threshold here erodes the physical rig border.
    # The exact native domain remains authoritative for the final labels.
    dom=area(e.domain)>0
    facies=cv2.resize(np.asarray(e.facies,np.float32),target,interpolation=cv2.INTER_NEAREST)
    conservative=area((e.conservative_labels>0)&e.domain)
    aq=area((e.conservative_labels==1)&e.domain)
    q=np.asarray(e.completion_probability)
    if q.shape==shape:
        completion=area((q>=.5)&(e.conservative_labels==0))>.01
    elif q.shape==dom.shape:
        native_q=cv2.resize(q.astype(np.float32),shape[::-1],interpolation=cv2.INTER_LINEAR)
        completion=area((native_q>=.5)&(e.conservative_labels==0))>.01
    else:raise ValueError("Completion scores use neither the native nor declared working grid")
    anchor=area(e.anchors)>.05
    # Providers propose. Independent anchors, never completion itself, own
    # whether a component is admissible. A missed weak fringe may join it.
    candidate=(conservative>.05)|completion
    supported=anchored(candidate & dom,anchor)
    gas=area((e.conservative_labels==2)&e.domain)
    reference,available,records=reference_evidence(area(e.current)/255.,area(e.baseline)/255.,
        supported & (gas<=.2),dom,facies,step,settings,occupied=supported)
    observed=np.zeros_like(dom)
    if e.independent_spatial_observation is not None:
        # Agreement with a separately computed optical observation and a strong
        # regional indicator change can establish a real separate object even
        # when its transit is not visible. B alone never supplies this evidence.
        observed=(area(e.independent_spatial_observation)>.05)&candidate&dom&(reference>=settings.independent_observation_score)
        labels,n=ndi.label(observed)
        sizes=np.bincount(labels.ravel())*step[0]*step[1]
        keep=sizes>=settings.minimum_observation_area_m2;keep[0]=False
        observed=keep[labels]
        anchor |= observed
    reference_candidate=(reference>=1)&dom
    proposal=joint_reach(candidate,reference_candidate,anchor,dom,step,settings.reference_search_m,
        reference_strength=reference,maximum_evidence_weight=settings.maximum_reference_evidence_weight)
    rejected=completion&~proposal&dom
    if not proposal.any():
        empty.setflags(write=False)
        return PhaseResult(empty,{"rejected_completion":rejected},dict(base_report,reason="no_causally_supported_region",
                           co2_pixels=0,rejected_completion_cells=int(rejected.sum()),facies=records))
    contour_evidence=(~available)|(reference>=settings.contour_reference_floor)
    signed,core=regularized_field(proposal,supported,dom,step,settings,
                                 expansion_support=contour_evidence)
    native_signed=cv2.resize(signed,shape[::-1],interpolation=cv2.INTER_LINEAR)
    native_mask=(native_signed>=0)&e.domain
    native_anchor=cv2.resize(anchor.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    native_mask=anchored(native_mask,native_anchor)
    labels=np.where(native_mask,1,0).astype(np.uint8)
    # Gas is a separate evidence provider, restricted by the same final CO2
    # decision. No aqueous completion score may turn a sand grain into gas.
    labels[native_mask & (e.conservative_labels==2)]=2
    # A gas claim too small to be a body of gas is withdrawn to aqueous, not to water.
    labels=withdraw_small_free_gas(labels,dy,dx)
    labels.setflags(write=False)
    return PhaseResult(labels,dict(reference_score=reference,reference_available=available,
        supported_proposal=supported,regional_proposal=proposal,rejected_completion=rejected,
        independent_observation=observed,
        contour_expansion_support=contour_evidence,
        secure_core=core,contour_field_m=signed),dict(base_report,reason="decided",working_shape=list(dom.shape),
        co2_pixels=int(native_mask.sum()),aqueous_pixels=int((labels==1).sum()),gas_pixels=int((labels==2).sum()),
        removed_conservative_pixels=int(np.count_nonzero((e.conservative_labels>0)&~native_mask)),
        added_conservative_pixels=int(np.count_nonzero((e.conservative_labels==0)&native_mask)),
        rejected_completion_cells=int(rejected.sum()),facies=records))

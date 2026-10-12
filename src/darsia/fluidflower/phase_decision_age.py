"""Unified phase decision with a causal local-age prior for weak outer fronts.

The age is time since first observed conservative CO2, not a concentration or a
known molecular contact time. The length/time settings are empirical inference
priors, not a fitted diffusion coefficient. Mature regions recover v1 exactly.
Neither completed nor final phase maps can age their own future extensions.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from . import phase_decision as base

VERSION = 'unified-phase-decision-local-age-v2'


@dataclass(frozen=True)
class Settings(base.Settings):
    mature_local_age_h: float = 6.
    young_reference_threshold: float = 2.
    young_weak_fringe_m: float = .006
    strong_reference_threshold: float = 3.
    young_contour_reference_floor: float = 1.
    young_outward_allowance_m: float = .0005

    def __post_init__(self):
        super().__post_init__()
        if self.young_reference_threshold < 1 or self.strong_reference_threshold < self.young_reference_threshold:
            raise ValueError('Young and strong reference thresholds must preserve their order')
        if self.young_contour_reference_floor < self.contour_reference_floor:
            raise ValueError('A young contour cannot require less evidence than a mature one')
        if self.young_outward_allowance_m > self.contour_outward_allowance_m:
            raise ValueError('A young contour cannot have a larger outward allowance')


@dataclass(frozen=True)
class Evidence(base.Evidence):
    first_observed_h: np.ndarray | None = None


def grid(shape, spacing, settings):
    h,w=shape;dy,dx=spacing
    target=(max(2,round(w*dx/settings.working_spacing_m)),max(2,round(h*dy/settings.working_spacing_m)))
    return target,(h*dy/target[1],w*dx/target[0])


class ExposureHistory:
    """Small chronological observation state, copied with disposable references."""
    def __init__(self, shape, spacing, settings):
        self.target,self.spacing=grid(shape,spacing,settings)
        self.first_observed_h=np.full(self.target[::-1],np.nan,np.float32)
        self.previous_time=None

    def observe(self, conservative, domain, time, *, pre_injection=False):
        time=float(time)
        if not np.isfinite(time) or (self.previous_time is not None and time<=self.previous_time):
            raise ValueError('Exposure observations require strictly increasing time')
        if not pre_injection:
            observed=cv2.resize(((conservative>0)&domain).astype(np.float32),self.target,
                                interpolation=cv2.INTER_AREA)>.05
            unseen=observed & ~np.isfinite(self.first_observed_h)
            self.first_observed_h[unseen]=time
        self.previous_time=time
        return self.first_observed_h


def identity(settings=Settings()):
    return hashlib.sha256(json.dumps(dict(format=VERSION,settings=asdict(settings),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        common_source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()


def local_age(first, observed, elapsed, spacing, settings):
    if first is None or np.asarray(first).shape!=observed.shape:
        raise ValueError('Local-age decisions require chronological exposure on the working grid')
    first=np.asarray(first)
    if np.isinf(first).any() or np.any(first[np.isfinite(first)]>elapsed+1e-5):
        raise ValueError('Exposure history contains invalid or future observations')
    roots=observed & np.isfinite(first)
    if not roots.any():return np.zeros(observed.shape,np.float32)
    # Distance is across the nearby observed front, never from an injector or
    # gas phase. Remote historical regions cannot age a newly observed front.
    nearest=ndi.distance_transform_edt(~roots,return_distances=False,return_indices=True)
    age=np.maximum(elapsed-first[tuple(nearest)],0)
    return ndi.gaussian_filter(np.minimum(age,settings.mature_local_age_h),
        tuple(settings.reference_smoothing_m/s for s in spacing)).astype(np.float32)


def decide(evidence, settings=Settings()):
    shape=base._check(evidence);e=evidence
    if e.pre_injection:
        labels=np.zeros(shape,np.uint8);labels.setflags(write=False)
        return base.PhaseResult(labels,{},dict(format=VERSION,reason='pre_injection',co2_pixels=0,
            method_identity=identity(settings),elapsed_h=e.elapsed_h,history_updates=False,settings=asdict(settings)))
    target,step=grid(shape,e.spacing_m,settings)
    area=lambda a:cv2.resize(np.asarray(a,np.float32),target,interpolation=cv2.INTER_AREA)
    dom=area(e.domain)>0
    facies=cv2.resize(np.asarray(e.facies,np.float32),target,interpolation=cv2.INTER_NEAREST)
    conservative=area((e.conservative_labels>0)&e.domain)
    q=np.asarray(e.completion_probability)
    if q.shape!=shape:
        if q.shape!=dom.shape:raise ValueError('Completion uses neither the native nor working grid')
        q=cv2.resize(q.astype(np.float32),shape[::-1],interpolation=cv2.INTER_LINEAR)
    completion=area((q>=.5)&(e.conservative_labels==0))>.01
    anchor=area(e.anchors)>.05
    candidate=(conservative>.05)|completion
    supported=base.anchored(candidate&dom,anchor)
    gas=area((e.conservative_labels==2)&e.domain)
    reference,available,records=base.reference_evidence(area(e.current)/255.,area(e.baseline)/255.,
        supported&(gas<=.2),dom,facies,step,settings,occupied=supported)
    observed=np.zeros_like(dom)
    if e.independent_spatial_observation is not None:
        observed=(area(e.independent_spatial_observation)>.05)&candidate&dom&(reference>=settings.independent_observation_score)
        components,n=ndi.label(observed)
        sizes=np.bincount(components.ravel())*step[0]*step[1]
        keep=sizes>=settings.minimum_observation_area_m2;keep[0]=False
        observed=keep[components];anchor|=observed
    age=local_age(e.first_observed_h,conservative>.05,e.elapsed_h,step,settings)
    maturity=np.sqrt(np.clip(age/settings.mature_local_age_h,0,1))
    threshold=settings.young_reference_threshold-(settings.young_reference_threshold-1)*maturity
    width=settings.young_weak_fringe_m+(settings.reference_search_m*settings.maximum_reference_evidence_weight-settings.young_weak_fringe_m)*maturity
    distance=ndi.distance_transform_edt(~supported,sampling=step)
    # Strong observed reference changes remain admissible; the age prior limits
    # only weak extrapolation beyond existing model-supported CO2.
    reference_candidate=(reference>=threshold)&((distance<=width)|(reference>=settings.strong_reference_threshold))&dom
    proposal=base.joint_reach(candidate,reference_candidate,anchor,dom,step,settings.reference_search_m,
        reference_strength=reference,maximum_evidence_weight=settings.maximum_reference_evidence_weight)
    rejected=completion&~proposal&dom
    report=dict(format=VERSION,method_identity=identity(settings),settings=asdict(settings),elapsed_h=e.elapsed_h,
        provenance=e.provenance,history_updates=False,probabilities_calibrated=False,facies=records,
        local_age_basis='First conservative observation; sparse-image estimate, not measured contact time',
        age_parameters_are_empirical=True)
    if not proposal.any():
        labels=np.zeros(shape,np.uint8);labels.setflags(write=False)
        return base.PhaseResult(labels,{'rejected_completion':rejected},dict(report,reason='no_causally_supported_region',co2_pixels=0))
    floor=settings.young_contour_reference_floor-(settings.young_contour_reference_floor-settings.contour_reference_floor)*maturity
    contour_evidence=(~available)|(reference>=floor)
    # The normal contour solver still owns one smooth implicit field. The age
    # prior adjusts its outward allowance before the single final label decision.
    signed,secure=base.regularized_field(proposal,supported,dom,step,settings,expansion_support=contour_evidence)
    delta=(settings.contour_outward_allowance_m-settings.young_outward_allowance_m)*(1-maturity)
    signed-=delta
    native_signed=cv2.resize(signed,shape[::-1],interpolation=cv2.INTER_LINEAR)
    native_anchor=cv2.resize(anchor.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    mask=base.anchored((native_signed>=0)&e.domain,native_anchor)
    labels=np.where(mask,1,0).astype(np.uint8)
    labels[mask&(e.conservative_labels==2)]=2;labels.setflags(write=False)
    report.update(reason='decided',working_shape=list(dom.shape),co2_pixels=int(mask.sum()),
        aqueous_pixels=int((labels==1).sum()),gas_pixels=int((labels==2).sum()),
        removed_conservative_pixels=int(((e.conservative_labels>0)&~mask).sum()),
        added_conservative_pixels=int(((e.conservative_labels==0)&mask).sum()))
    return base.PhaseResult(labels,dict(reference_score=reference,reference_available=available,
        supported_proposal=supported,regional_proposal=proposal,rejected_completion=rejected,
        independent_observation=observed,contour_expansion_support=contour_evidence,secure_core=secure,
        contour_field_m=signed,local_age_h=age,local_maturity=maturity,reference_threshold=threshold,
        weak_fringe_width_m=width),report)

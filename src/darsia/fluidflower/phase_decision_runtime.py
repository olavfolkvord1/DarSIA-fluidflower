"""Versioned single-owner phase runtime; legacy providers supply evidence only.

The conservative provider retains its independent observation history because
the frozen completion weights were fitted against those inputs. Neither the
completion proposal nor this decision is fed into that history. References use
disposable chronological replays, just like ordinary observations.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import phase_decision as core
from . import phase_aqueous_context_completion as completion

PROFILE_FILE = "phase_decision.json"
PROFILE_KEY = "unified_phase_decision"
VERSION = "unified-phase-decision-runtime-v1"


def attach_profile(root, inherited):
    path=Path(root)/PROFILE_FILE
    if not path.is_file():return inherited
    spec=json.loads(path.read_text(encoding="utf-8"))
    if set(spec)!={"format","settings"} or spec['format']!=VERSION:
        raise ValueError("Unknown unified decision profile")
    core.Settings(**spec['settings'])
    if inherited is None or completion.PROFILE_KEY not in inherited:
        raise ValueError("Unified decision requires explicit conservative and learned evidence providers")
    if "local_aqueous_support" in inherited:
        raise ValueError("Local reference search is internal to unified decision; it cannot be stacked")
    return {**inherited,PROFILE_KEY:spec}


def split_profile(profile):
    return {k:v for k,v in profile.items() if k!=PROFILE_KEY},profile[PROFILE_KEY]


def runtime_identity(inherited_identity,specification):
    return hashlib.sha256(json.dumps(dict(format=VERSION,provider_identity=inherited_identity,
        specification=specification,decision_identity=core.identity(core.Settings(**specification['settings'])),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()


class DecisionSequence:
    def __init__(self,context,provider,specification,checkpoint):
        self.context,self.provider,self.specification=context,provider,specification
        self.settings=core.Settings(**specification['settings'])
        self.checkpoint=checkpoint
        self.baseline=None
        self._predictor=None
        self.last_diagnostics={}
        self.last_result=None
        self.previous_time=None

    def __getattr__(self,name):
        return getattr(object.__getattribute__(self,'provider'),name)

    def bind_reference(self,baseline_rgb):
        if self.baseline is not None:raise ValueError("Decision reference is immutable")
        if hasattr(self.provider,'bind_reference'):self.provider.bind_reference(baseline_rgb)
        self.baseline=np.array(baseline_rgb,copy=True)
        self.baseline.setflags(write=False)

    def apply(self,predictions,elapsed_h):
        raise ValueError("Unified decision requires registered current and baseline RGB")

    def apply_with_reference(self,predictions,elapsed_h,current_rgb):
        if self.baseline is None:raise ValueError("Bind the decision reference first")
        time=float(elapsed_h)
        if not np.isfinite(time) or (self.previous_time is not None and time<=self.previous_time):
            raise ValueError("Decision observations require strictly increasing times; fork reference histories")
        if hasattr(self.provider,'apply_with_reference'):
            evidence,report=self.provider.apply_with_reference(predictions,time,current_rgb)
        else:evidence,report=self.provider.apply(predictions,time)
        before=completion.delivered_labels(evidence,self.context.domain)
        spacing=(abs(float(self.context.y_m[1]-self.context.y_m[0])),abs(float(self.context.x_m[1]-self.context.x_m[0])))
        negative=bool(report.get('pre_injection_negative_control'))
        q=np.zeros_like(self.context.domain,dtype=np.float32)
        if not negative:
            x,_=completion.prepare_inputs(current_rgb,self.baseline,predictions['co2_presence_probability'],
                                          before,self.context.domain,spacing)
            if self._predictor is None:self._predictor=completion._Predictor(self.checkpoint)
            q=cv2.resize(self._predictor(x),self.context.domain.shape[::-1],interpolation=cv2.INTER_LINEAR)
        result=core.decide(core.Evidence(current_rgb,self.baseline,before,q,
            (before>0)&self.context.domain,self.context.domain,self.context.facies,spacing,time,negative,
            dict(provider='frozen-causal-observations-before-completion',completion_checkpoint_sha256=self.checkpoint['checkpoint_sha256'],
                 completion_and_final_result_do_not_update_provider_history=True),
            independent_spatial_observation=self.provider.last_diagnostics.get('spatial_only_presence')),self.settings)
        self.previous_time=time;self.last_result=result
        labels=result.labels
        # Categorical decisions are explicit. These 0/1 fields satisfy the
        # existing phase renderer contract; they are NOT calibrated confidence.
        co2=(labels>0).astype(np.float32);gas=(labels==2).astype(np.float32)
        output={**predictions,'co2_presence_probability':co2,'gas_probability_conditional_on_co2':gas,
                'aqueous_co2_probability':(labels==1).astype(np.float32),
                'free_co2_gas_probability':gas,'untouched_water_probability':1-co2}
        self.last_diagnostics=dict(added_aqueous=(labels==1)&(before==0),
            suppressed_presence=(before>0)&(labels==0),unresolved=np.zeros_like(self.context.domain),
            decision_before_co2=before>0,decision_before_gas=before==2)
        return output,{**report,PROFILE_KEY:result.report}

    def write_diagnostics(self,*args,**kwargs):
        from .phase_presence_context import PresenceSequence
        return PresenceSequence.write_diagnostics(self,*args,**kwargs)

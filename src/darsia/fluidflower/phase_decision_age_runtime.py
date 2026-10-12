"""Chronological runtime for the local-age unified phase decision."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from . import phase_decision_age as core
from . import phase_decision_runtime as legacy
from . import phase_aqueous_context_completion as completion

PROFILE_FILE=legacy.PROFILE_FILE
PROFILE_KEY=legacy.PROFILE_KEY
VERSION='unified-phase-decision-local-age-runtime-v2'
split_profile=legacy.split_profile


def attach_profile(root,inherited):
    path=Path(root)/PROFILE_FILE
    if not path.is_file():return inherited
    spec=json.loads(path.read_text(encoding='utf-8'))
    if set(spec)!={'format','settings'} or spec['format']!=VERSION:
        raise ValueError('Unknown local-age decision profile')
    core.Settings(**spec['settings'])
    if inherited is None or completion.PROFILE_KEY not in inherited:
        raise ValueError('Local-age decision requires frozen conservative and completion providers')
    if 'local_aqueous_support' in inherited:
        raise ValueError('Reference search is internal to the unified decision; it cannot be stacked')
    return {**inherited,PROFILE_KEY:spec}


def runtime_identity(inherited_identity,specification):
    return hashlib.sha256(json.dumps(dict(format=VERSION,provider_identity=inherited_identity,
        specification=specification,decision_identity=core.identity(core.Settings(**specification['settings'])),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_adapter_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()


class DecisionSequence(legacy.DecisionSequence):
    def __init__(self,context,provider,specification,checkpoint):
        self.context,self.provider,self.specification=context,provider,specification
        self.settings=core.Settings(**specification['settings']);self.checkpoint=checkpoint
        self.baseline=None;self._predictor=None;self.last_diagnostics={};self.last_result=None;self.previous_time=None
        self.spacing=(abs(float(context.y_m[1]-context.y_m[0])),abs(float(context.x_m[1]-context.x_m[0])))
        self.exposure_history=core.ExposureHistory(context.domain.shape,self.spacing,self.settings)

    def apply_with_reference(self,predictions,elapsed_h,current_rgb):
        if self.baseline is None:raise ValueError('Bind the decision reference first')
        time=float(elapsed_h)
        if not np.isfinite(time) or (self.previous_time is not None and time<=self.previous_time):
            raise ValueError('Decision observations require strictly increasing times; fork reference histories')
        if hasattr(self.provider,'apply_with_reference'):
            evidence,report=self.provider.apply_with_reference(predictions,time,current_rgb)
        else:evidence,report=self.provider.apply(predictions,time)
        before=completion.delivered_labels(evidence,self.context.domain)
        negative=bool(report.get('pre_injection_negative_control'))
        q=np.zeros_like(self.context.domain,dtype=np.float32)
        if not negative:
            x,_=completion.prepare_inputs(current_rgb,self.baseline,predictions['co2_presence_probability'],before,self.context.domain,self.spacing)
            if self._predictor is None:self._predictor=completion._Predictor(self.checkpoint)
            q=cv2.resize(self._predictor(x),self.context.domain.shape[::-1],interpolation=cv2.INTER_LINEAR)
        first=self.exposure_history.observe(before,self.context.domain,time,pre_injection=negative)
        result=core.decide(core.Evidence(current_rgb,self.baseline,before,q,(before>0)&self.context.domain,
            self.context.domain,self.context.facies,self.spacing,time,negative,
            dict(provider='frozen-causal-observations-before-completion',completion_checkpoint_sha256=self.checkpoint['checkpoint_sha256'],
                 completion_and_final_result_do_not_update_provider_history=True),
            independent_spatial_observation=self.provider.last_diagnostics.get('spatial_only_presence'),first_observed_h=first),self.settings)
        self.previous_time=time;self.last_result=result
        labels=result.labels;co2=(labels>0).astype(np.float32);gas=(labels==2).astype(np.float32)
        output={**predictions,'co2_presence_probability':co2,'gas_probability_conditional_on_co2':gas,
            'aqueous_co2_probability':(labels==1).astype(np.float32),'free_co2_gas_probability':gas,'untouched_water_probability':1-co2}
        self.last_diagnostics=dict(added_aqueous=(labels==1)&(before==0),suppressed_presence=(before>0)&(labels==0),
            unresolved=np.zeros_like(self.context.domain),decision_before_co2=before>0,decision_before_gas=before==2)
        return output,{**report,PROFILE_KEY:result.report}

"""Joint two-head refinement with truthful regional aqueous likelihoods.

The existing pure and mixed-region objectives remain active. Aqueous presence
is no longer used as an all-pixel gas-negative bag. Coupled aqueous absence
permits water OR gas, and coupled presence selects the same pixels in both
heads. This module performs training only; inherited OOF metrics do not measure
the returned model unless an orchestrator runs fresh fold-local refinement.
"""
from dataclasses import replace
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from . import logistic as base
from . import coupled_aqueous as coupled

POLICY = 'joint-hierarchical-aqueous-mil-v1'
HEADS = ('co2_presence', 'gas_within_co2')


def _pure_weights(owners, regions, head, runs):
    weights=np.zeros(owners.size,np.float64);labels=np.zeros(owners.size,np.float64)
    pending=[];totals={}
    for index,region in enumerate(regions):
        target=base.pixel_target(region.phase_class,head)
        if target is None or region.supervision_weight<=0 or (runs is not None and region.run not in runs):continue
        rows=np.flatnonzero(owners==index)
        if not rows.size:continue
        key=(region.run,int(target));mass=float(region.supervision_weight)
        pending.append((rows,int(target),mass,key));totals[key]=totals.get(key,0.)+mass
    for rows,target,mass,key in pending:
        weights[rows]=mass/totals[key]/rows.size;labels[rows]=target
    if weights.sum():weights/=weights.sum()
    return labels,weights


def _pack(bags, target, inverse, *, positive_top_fraction, selections=None):
    selected=[b for b in bags if int(b[1])==target]
    if not selected:return None
    counts=np.array([len(b[0]) for b in selected],np.int64)
    width=int(counts.max());indices=np.zeros((len(selected),width),np.int64)
    valid=np.arange(width)[None,:]<counts[:,None]
    for index,bag in enumerate(selected):indices[index,:counts[index]]=inverse[bag[0]]
    chosen=np.array([base._bag_selected_count(len(b[0]),target,b[3],positive_top_fraction,selections)
                     for b in selected],np.int64)
    total=sum(float(b[2]) for b in bags)
    return dict(indices=indices,valid=valid,counts=counts,chosen=chosen,
        weights=np.array([float(b[2])/total for b in selected]),target=int(target))


def _device_pack(pack,xp,dtype):
    if pack is None:return None
    return {k:(xp.asarray(v,dtype=dtype if k=='weights' else None) if isinstance(v,np.ndarray) else v)
            for k,v in pack.items()}


def _scatter(residual,indices,values,xp):
    # Duplicate row references in overlapping regional constraints add weight.
    residual += xp.bincount(indices.ravel(),weights=values.ravel(),minlength=residual.size)


def _selected(pack, values, *, smallest, xp):
    valid=pack['valid']
    if bool(np.all(pack['chosen_host']==pack['counts_host'])):
        return xp.where(valid,1.,0.)
    padded=xp.where(valid,values,np.inf if smallest else -np.inf)
    if xp is np:
        order=xp.argsort(padded,axis=1,kind='stable')
    else:order=xp.argsort(padded,axis=1)
    if not smallest:order=order[:,::-1]
    selected=xp.zeros_like(values)
    selection=xp.arange(values.shape[1])[None,:]<pack['chosen'][:,None]
    xp.put_along_axis(selected,order,selection.astype(values.dtype),axis=1)
    return selected


class JointAqueousObjective:
    """Reusable value/gradient with either NumPy or CuPy device evaluation."""
    def __init__(self,matrix,owners,regions,*,seed_fits,regularization,runs=None,
            weak_strength=.35,aqueous_absence_strength=1.,aqueous_presence_strength=None,
            positive_top_fraction=.1,backend='cpu'):
        if backend not in ('cpu','cuda'):raise ValueError('Joint backend must be cpu or cuda')
        if set(seed_fits)!=set(HEADS) or set(regularization)!=set(HEADS):
            raise ValueError('Joint fit requires both hierarchical heads and regularizations')
        strengths=(weak_strength,aqueous_absence_strength,
            weak_strength if aqueous_presence_strength is None else aqueous_presence_strength)
        if any(not np.isfinite(v) or v<0 for v in strengths) or not 0<positive_top_fraction<=1:
            raise ValueError('Invalid joint supervision strengths or MIL fraction')
        self.weak_strength,self.absence_strength,self.presence_strength=map(float,strengths)
        self.backend=backend;self.seed_fits=seed_fits;self.regularization=regularization
        self.matrix=np.asarray(matrix,dtype=np.float32);owners=np.asarray(owners,np.int32)
        if self.matrix.ndim!=2 or owners.shape!=(self.matrix.shape[0],):
            raise ValueError('Joint feature rows and region ownership do not agree')
        if owners.size and (owners.min()<0 or owners.max()>=len(regions)):
            raise ValueError('Joint row ownership refers to an unknown region')
        self.features=self.matrix.shape[1];self.stride=self.features+1
        for head in HEADS:
            fit=seed_fits[head]
            if any(np.asarray(getattr(fit,k)).shape!=(self.features,) for k in ('mean','scale','coefficients')):
                raise ValueError('Both seed heads must use the exact feature contract')
            if not np.isfinite(np.r_[fit.mean,fit.scale,fit.coefficients,fit.intercept]).all() or np.min(fit.scale)<=0:
                raise ValueError('Invalid seed normalization or coefficients')
            if not np.isfinite(regularization[head]) or regularization[head]<0:
                raise ValueError('Invalid per-head regularization')
        self.coupled_bags=coupled.bags(owners,regions,runs=runs)
        active=np.zeros(owners.size,bool);plans={}
        for head in HEADS:
            # Only the invalid gas-negative interpretation is removed. The
            # necessary CO2-positive condition remains active in that head.
            view=[replace(r,phase_class=base.PHASE_IGNORED)
                  if head=='gas_within_co2' and r.phase_class==base.PHASE_AQUEOUS_PRESENT_REGION else r
                  for r in regions]
            labels,weights=_pure_weights(owners,view,head,runs)
            bags=base._weak_region_bags(owners,view,head=head,runs=runs)
            selection=base._dominance_bag_selection(view,bags,head=head)
            plans[head]=(labels,weights,bags,selection)
            active|=weights>0
            for rows,*_ in bags:active[rows]=True
        for bag in self.coupled_bags:active[bag.rows]=True
        self.active_rows=np.flatnonzero(active)
        if not self.active_rows.size:raise ValueError('No active supervised feature rows')
        inverse=np.full(owners.size,-1,np.int64);inverse[self.active_rows]=np.arange(self.active_rows.size)
        self.host={};self.feature_range={}
        selected=np.asarray(self.matrix[self.active_rows],np.float64)
        if not np.isfinite(selected).all():raise ValueError('Active joint features must be finite')
        for head in HEADS:
            fit=seed_fits[head];labels,weights,bags,selection=plans[head]
            x=(selected-fit.mean)/fit.scale
            if not np.isfinite(x).all():raise ValueError('Seed scaling produces nonfinite active features')
            sample=x[np.linspace(0,len(x)-1,min(len(x),10000),dtype=int)]
            self.feature_range[head]=dict(maximum_absolute_standardized_feature=float(np.max(np.abs(x))),
                sampled_q999_absolute_standardized_feature=float(np.quantile(np.abs(sample),.999)),
                normalization_reused_from_seed=True)
            self.host[head]=dict(x=x,labels=labels[active],weights=weights[active],
                bags=[_pack(bags,target,inverse,positive_top_fraction=positive_top_fraction,selections=selection) for target in (0,1)])
        # Normalize absence and presence separately, then apply their explicit
        # strengths. Each group is confidence-balanced within each run.
        self.coupled_packs=[]
        for target in (0,1):
            group=[(b.rows,b.target,b.weight,b.region_index) for b in self.coupled_bags if b.target==target]
            self.coupled_packs.append(_pack(group,target,inverse,positive_top_fraction=positive_top_fraction))
        self.initial=np.concatenate([np.r_[seed_fits[h].coefficients,seed_fits[h].intercept] for h in HEADS]).astype(np.float64)
        self.device=None
        if backend=='cuda':
            import cupy as cp
            self.xp=cp
            self.device=self._convert(cp,np.float32)
        else:self.xp=np
        self.cpu=self._convert(np,np.float64)
        self.metadata=dict(policy=POLICY,objective_scope='joint-two-head-training-only',
            active_feature_rows=int(active.sum()),feature_count=self.features,
            aqueous_absence_regions=sum(b.target==0 for b in self.coupled_bags),
            aqueous_presence_regions=sum(b.target==1 for b in self.coupled_bags),
            coupled_regions=[dict(region_id=b.region_id,region_index=b.region_index,run=b.run,target=b.target,
                sampled_rows=len(b.rows),run_normalized_weight=b.weight) for b in self.coupled_bags],
            weak_strength=self.weak_strength,aqueous_absence_strength=self.absence_strength,
            aqueous_presence_strength=self.presence_strength,positive_top_fraction=positive_top_fraction,
            coupled_weight_normalization='separate target groups; equal run mass; confidence-weighted regions; pixel-count invariant',
            aqueous_absence_quantifier='every sampled pixel permits water OR gas',
            aqueous_presence_quantifier='best joint aqueous likelihood rows, same rows for both heads',
            invalid_aqueous_present_gas_negative_bags_excluded=True,
            seed_normalization_reused=True,feature_range=self.feature_range,
            new_oof_validation=False,inherited_oof_metrics_not_reused=True,
            stable_likelihood='unclipped-logit logaddexp; consistent analytical gradient')

    def _convert(self,xp,dtype):
        def pack(p):
            if p is None:return None
            out=_device_pack(p,xp,dtype)
            out['chosen_host']=p['chosen'];out['counts_host']=p['counts']
            return out
        return dict(heads={h:dict(x=xp.asarray(v['x'],dtype=dtype),labels=xp.asarray(v['labels'],dtype=dtype),
            weights=xp.asarray(v['weights'],dtype=dtype),bags=[pack(b) for b in v['bags']]) for h,v in self.host.items()},
            coupled=[pack(p) for p in self.coupled_packs])

    def _evaluate(self,parameters,*,xp,data):
        theta=xp.asarray(parameters,dtype=np.float64 if xp is np else np.float32)
        logits=[];residuals=[];coefficients=[];loss=xp.asarray(0.,dtype=np.float64)
        for index,head in enumerate(HEADS):
            info=data['heads'][head];start=index*self.stride;coef=theta[start:start+self.features];bias=theta[start+self.features]
            z=info['x']@coef+bias;p=xp.exp(-xp.logaddexp(0.,-z))
            residual=info['weights']*(p-info['labels'])
            loss+=xp.sum(info['weights']*(xp.logaddexp(0.,z)-info['labels']*z),dtype=np.float64)
            loss+=.5*self.regularization[head]*xp.sum(coef*coef,dtype=np.float64)
            for bag in info['bags']:
                if bag is None:continue
                values=z[bag['indices']];target=bag['target']
                chosen=_selected(bag,values,smallest=not target,xp=xp)
                weights=self.weak_strength*bag['weights'][:,None]/bag['chosen'][:,None]*chosen
                value=xp.logaddexp(0.,-values if target else values)
                probability=xp.exp(-xp.logaddexp(0.,-values))
                loss+=xp.sum(weights*value,dtype=np.float64)
                _scatter(residual,bag['indices'],weights*(probability-target),xp)
            coefficients.append(coef);logits.append(z);residuals.append(residual)
        for bag,strength in zip(data['coupled'],(self.absence_strength,self.presence_strength)):
            if bag is None or strength==0:continue
            rows=bag['indices'];values,gc,gg=coupled.likelihood_terms(logits[0][rows],logits[1][rows],target=bag['target'],xp=xp)
            chosen=_selected(bag,values,smallest=True,xp=xp)
            weights=strength*bag['weights'][:,None]/bag['chosen'][:,None]*chosen
            loss+=xp.sum(weights*values,dtype=np.float64)
            _scatter(residuals[0],rows,weights*gc,xp);_scatter(residuals[1],rows,weights*gg,xp)
        gradients=[]
        for index,head in enumerate(HEADS):
            grad=data['heads'][head]['x'].T@residuals[index]+self.regularization[head]*coefficients[index]
            gradients.append(xp.concatenate((grad,xp.sum(residuals[index],dtype=np.float64).reshape(1))))
        gradient=xp.concatenate(gradients)
        if xp is np:return float(loss),np.asarray(gradient,np.float64)
        return float(loss.item()),xp.asnumpy(gradient).astype(np.float64)

    def evaluate(self,parameters):
        values=np.asarray(parameters,np.float64)
        if values.shape!=self.initial.shape or not np.isfinite(values).all():
            raise ValueError('Joint parameter vector must be finite and match both heads')
        return self._evaluate(values,xp=self.xp,data=self.cpu if self.xp is np else self.device)

    def cpu_reference(self,parameters):
        return self._evaluate(parameters,xp=np,data=self.cpu)

    def close(self):
        self.device=None
        if self.backend=='cuda':
            self.xp.cuda.Device().synchronize()
            self.xp.get_default_memory_pool().free_all_blocks()


def fit_joint_aqueous_mil(matrix,region_indices,regions,*,seed_fits:Mapping,
        regularization:Mapping,runs=None,weak_strength=.35,aqueous_absence_strength=1.,
        aqueous_presence_strength=None,positive_top_fraction=.1,backend='cpu',
        max_iter=250,tolerance=1e-7):
    """Return two refined LogisticFits plus explicit training-only diagnostics."""
    if max_iter<1 or tolerance<=0:raise ValueError('Invalid joint optimizer stopping settings')
    objective=JointAqueousObjective(matrix,region_indices,regions,seed_fits=seed_fits,
        regularization=regularization,runs=runs,weak_strength=weak_strength,
        aqueous_absence_strength=aqueous_absence_strength,aqueous_presence_strength=aqueous_presence_strength,
        positive_top_fraction=positive_top_fraction,backend=backend)
    try:
        initial_loss,initial_gradient=objective.evaluate(objective.initial)
        initial_reference_loss,initial_reference_gradient=objective.cpu_reference(objective.initial)
        result=minimize(objective.evaluate,objective.initial,jac=True,method='L-BFGS-B',
            options=dict(maxiter=int(max_iter),ftol=0.,gtol=float(tolerance),maxls=40,maxcor=20))
        first_optimizer=dict(backend=backend,iterations=int(result.nit),success=bool(result.success),message=str(result.message))
        final_loss,gradient=objective.evaluate(result.x)
        reference_loss,reference_gradient=objective.cpu_reference(result.x)
        polished=False
        final_backend=backend
        # Float32 objective noise must not turn an ftol/line-search stop into a
        # convergence claim. Use the same objective in CPU64 for a final solve.
        if backend=='cuda' and (not result.success or np.max(np.abs(reference_gradient))>tolerance):
            previous_iterations=int(result.nit);previous_evaluations=int(result.nfev)
            result=minimize(objective.cpu_reference,result.x,jac=True,method='L-BFGS-B',
                options=dict(maxiter=int(max_iter),ftol=0.,gtol=float(tolerance),maxls=60,maxcor=20))
            polished=True;final_backend='cpu'
            final_loss,gradient=objective.evaluate(result.x)
            reference_loss,reference_gradient=objective.cpu_reference(result.x)
        else:previous_iterations=previous_evaluations=0
        converged=bool(result.success and np.max(np.abs(reference_gradient))<=tolerance)
        output_loss=reference_loss if final_backend=='cpu' else final_loss
        if not np.isfinite(final_loss) or not np.isfinite(gradient).all():
            raise ValueError('Joint optimization returned nonfinite parameters or objective')
        fits={}
        for index,head in enumerate(HEADS):
            start=index*objective.stride;seed=seed_fits[head]
            fits[head]=base.LogisticFit(mean=np.array(seed.mean,copy=True),scale=np.array(seed.scale,copy=True),
                coefficients=np.asarray(result.x[start:start+objective.features],np.float64),
                intercept=float(result.x[start+objective.features]),backend=final_backend,
                precision='float64' if final_backend=='cpu' else 'float32',iterations=previous_iterations+int(result.nit),
                converged=converged,objective=float(output_loss),regularization=float(regularization[head]))
        report=dict(objective.metadata,backend=final_backend,requested_backend=backend,
            precision='float64' if final_backend=='cpu' else 'float32',
            optimizer='scipy-L-BFGS-B',converged=converged,iterations=previous_iterations+int(result.nit),
            function_evaluations=previous_evaluations+int(result.nfev),optimizer_message=str(result.message),
            first_optimizer=first_optimizer,cpu64_polish_performed=polished,
            convergence_criterion='SciPy success AND full CPU64 gradient infinity norm <= requested tolerance',
            gradient_tolerance=float(tolerance),objective=float(output_loss),initial_objective=float(initial_reference_loss),
            maximum_absolute_gradient=float(np.max(np.abs(reference_gradient))),
            initial_maximum_absolute_gradient=float(np.max(np.abs(initial_gradient))),
            numerical_validation=dict(cpu_float64_objective=reference_loss,backend_objective=final_loss,
                objective_absolute_difference=abs(final_loss-reference_loss),
                gradient_maximum_absolute_difference=float(np.max(np.abs(gradient-reference_gradient))),
                seed_objective_absolute_difference=abs(initial_loss-initial_reference_loss),
                seed_gradient_maximum_absolute_difference=float(np.max(np.abs(initial_gradient-initial_reference_gradient)))),
            changed_heads=[h for h in HEADS if not np.array_equal(fits[h].coefficients,seed_fits[h].coefficients)
                           or fits[h].intercept!=seed_fits[h].intercept])
        return fits,report
    finally:objective.close()

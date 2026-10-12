"""Coupled likelihoods for regional aqueous evidence in two optical heads.

These primitives do not change training dispatch. For p=P(CO2) and
q=P(gas|CO2), aqueous probability is p*(1-q). No-aqueous means the union
of water and gas, not a hard target for either separate head. Log-space
evaluation remains differentiable when a wrong prediction is saturated.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .logistic import PHASE_AQUEOUS_ABSENT_REGION, PHASE_AQUEOUS_PRESENT_REGION

POLICY = 'coupled-aqueous-region-likelihood-v1'


def likelihood_terms(co2_logits, gas_logits, *, target, xp=np):
    """Return per-pixel loss and derivatives with respect to both logits.

    ``target=0`` is certified aqueous absence: every sampled pixel contributes.
    ``target=1`` requires a caller to select the *same* pixels in both heads for
    regional MIL. A positive region never implies absence of gas elsewhere.
    Pass numpy or cupy explicitly as ``xp``; no device transfer is performed.
    """
    if target not in (0, 1):
        raise ValueError('Coupled aqueous target must be zero or one')
    zc, zg = xp.asarray(co2_logits), xp.asarray(gas_logits)
    if zc.shape != zg.shape:
        raise ValueError('Both aqueous logits must describe identical pixels')
    # log(p), log(1-p), log(q), log(1-q), without clipped probabilities.
    lp = -xp.logaddexp(0., -zc)
    lnp = -xp.logaddexp(0., zc)
    lq = -xp.logaddexp(0., -zg)
    lnq = -xp.logaddexp(0., zg)
    if target == 1:
        return -lp-lnq, -xp.exp(lnp), xp.exp(lq)
    log_allowed = xp.logaddexp(lnp, lp+lq)
    # Stable positive magnitudes avoid cancellation such as p-gas_responsibility.
    grad_co2 = xp.exp(lp+lnp+lnq-log_allowed)
    grad_gas = -xp.exp(lp+lq+lnq-log_allowed)
    return -log_allowed, grad_co2, grad_gas


@dataclass(frozen=True)
class CoupledBag:
    rows: np.ndarray
    target: int
    weight: float
    region_index: int
    region_id: str
    run: str


def bags(region_indices, regions: Sequence, *, runs=None):
    """Select only active, sampled training regions and balance run/target mass.

    No missing direct label is turned into water or gas. Zero-weight, unsampled
    and held-out regions cannot enter the coupled objective or its provenance.
    """
    owners = np.asarray(region_indices)
    pending=[];totals={}
    targets={PHASE_AQUEOUS_ABSENT_REGION:0, PHASE_AQUEOUS_PRESENT_REGION:1}
    for index, region in enumerate(regions):
        target=targets.get(region.phase_class)
        if target is None or region.supervision_weight<=0 or (runs is not None and region.run not in runs):
            continue
        rows=np.flatnonzero(owners==index)
        if not rows.size:continue
        key=(region.run,target)
        weight=float(region.supervision_weight)
        totals[key]=totals.get(key,0.)+weight
        pending.append((rows,target,weight,index,region.region_id,region.run))
    return [CoupledBag(rows,target,weight/totals[(run,target)],index,identifier,run)
            for rows,target,weight,index,identifier,run in pending]


def objective_gradient(co2_logits, gas_logits, bags, *, positive_top_fraction=.1):
    """CPU reference: balanced region likelihood plus two logit gradients.

    Absence uses every sampled pixel. Positive MIL selects pixels by joint
    aqueous likelihood, so the two heads cannot satisfy different pixels.
    Region size does not increase a region's total optimization weight.
    """
    zc,zg=np.asarray(co2_logits,np.float64),np.asarray(gas_logits,np.float64)
    if zc.ndim!=1 or zc.shape!=zg.shape or not np.isfinite(zc).all() or not np.isfinite(zg).all():
        raise ValueError('Coupled objective requires aligned finite logit vectors')
    if not 0<positive_top_fraction<=1:
        raise ValueError('Positive MIL fraction must be in (0, 1]')
    gc=np.zeros_like(zc);gg=np.zeros_like(zg);loss=0.
    total=sum(b.weight for b in bags)
    if not bags:return 0.,gc,gg
    if total<=0:raise ValueError('Coupled bags need positive total weight')
    for bag in bags:
        if bag.rows.size==0 or bag.rows.min()<0 or bag.rows.max()>=zc.size:
            raise ValueError('Invalid coupled bag row selection')
        values,dc,dg=likelihood_terms(zc[bag.rows],zg[bag.rows],target=bag.target)
        selected=np.arange(bag.rows.size)
        if bag.target==1:
            count=max(1,round(positive_top_fraction*bag.rows.size))
            selected=np.argsort(values,kind='stable')[:count]
        weight=bag.weight/total/selected.size
        rows=bag.rows[selected]
        np.add.at(gc,rows,weight*dc[selected]);np.add.at(gg,rows,weight*dg[selected])
        loss+=weight*float(values[selected].sum())
    return float(loss),gc,gg

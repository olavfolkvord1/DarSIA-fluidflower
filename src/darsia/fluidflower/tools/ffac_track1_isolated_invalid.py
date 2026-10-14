"""Compatibility entry points for the configured local invalid-pixel repair."""
from __future__ import annotations
from darsia.fluidflower.native import support as _support
from darsia.fluidflower.native.config import require_parameters

def __getattr__(name):
    if name not in ('POLICY', 'CONTRACT', 'MAX_AREA_PX'): raise AttributeError(name)
    p=require_parameters('isolated_invalid')
    values={'POLICY':p['policy'],'CONTRACT':p['contract_metadata']['contract'],'MAX_AREA_PX':p['max_area_px']}
    if name in values:return values[name]
    raise AttributeError(name)

def validate_policy(policy: str) -> str:
    return _support.validate_invalid_policy(policy,parameters=require_parameters('isolated_invalid'))
def policy_contract() -> dict:
    return _support.invalid_policy_contract(parameters=require_parameters('isolated_invalid'))
def fill_isolated_invalid(log_rgb,safe_filled):
    return _support.fill_isolated_invalid(log_rgb,safe_filled,parameters=require_parameters('isolated_invalid'))

__all__=['POLICY','CONTRACT','MAX_AREA_PX','validate_policy','policy_contract','fill_isolated_invalid']

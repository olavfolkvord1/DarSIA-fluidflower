"""Compatibility entry points for explicitly configured photographic display."""
from __future__ import annotations
import numpy as np
from darsia.fluidflower.native import display_modes as _display
from darsia.fluidflower.native.config import require_parameters

def __getattr__(name):
    if name not in ('CONTRACT', 'DISPLAY_MODES', 'MATRIX'): raise AttributeError(name)
    p = require_parameters('photographic_display')
    if name == 'CONTRACT': return p['contract_metadata']['contract']
    if name == 'DISPLAY_MODES': return tuple(p['modes'])
    if name == 'MATRIX':
        matrix = np.array(p['matrix'], dtype=np.float64); matrix.setflags(write=False); return matrix
    raise AttributeError(name)

def validate_display_mode(mode: str) -> str:
    return _display.validate_display_mode(mode, parameters=require_parameters('photographic_display'))

def display_contract() -> dict:
    return _display.photographic_contract(parameters=require_parameters('photographic_display'))

def photographic_display(log_camera_rgb, *, tile_rows=128):
    return _display.photographic_display(log_camera_rgb,tile_rows=tile_rows,parameters=require_parameters('photographic_display'))

__all__ = ['CONTRACT', 'MATRIX', 'DISPLAY_MODES', 'display_contract', 'photographic_display', 'validate_display_mode']

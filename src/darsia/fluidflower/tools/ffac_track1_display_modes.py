"""Compatibility entry points for configured display-only colour transfers."""
from __future__ import annotations
import numpy as np
from darsia.fluidflower.native import display_modes as _display
from darsia.fluidflower.native.config import require_parameters

def __getattr__(name):
    if name not in ('CONTRACT', 'DISPLAY_MODE', 'DISPLAY_MODES', 'GAINS', 'Y_REFERENCE', 'CLAUDE_CONTRACT_SHA256', 'CLAUDE_CONTRACT_PATH', '_MATRIX_SHA256', 'M1', 'A', 'M_NATURAL', 'LUMINANCE_WEIGHTS'): raise AttributeError(name)
    p = require_parameters('display_modes'); c = p['contract_metadata']
    keys = {'CONTRACT':c['contract'], 'DISPLAY_MODE':c['display_mode'], 'DISPLAY_MODES':tuple(p['modes']),
            'GAINS':p['gains'], 'Y_REFERENCE':c['sand_reference_luminance'],
            'CLAUDE_CONTRACT_SHA256':c['matrix_source_json_sha256'],
            'CLAUDE_CONTRACT_PATH':c['matrix_source_json_path'], '_MATRIX_SHA256':c['matrix_definition_sha256']}
    if name in keys: return keys[name]
    if name in ('M1','A','M_NATURAL','LUMINANCE_WEIGHTS'):
        value = c[name] if name in ('M1','A') else p['matrix'] if name=='M_NATURAL' else p['luminance_weights']
        matrix = np.array(value,dtype=np.float64);matrix.setflags(write=False);return matrix
    raise AttributeError(name)

def display_contract() -> dict:
    return _display.display_contract(parameters=require_parameters('display_modes'))
display_modes_contract = display_contract

def _fill_small_invalid(linear,valid):
    return _display._fill_small_invalid(linear,valid,parameters=require_parameters('display_modes'))
def _neutralise_highlights(linear,photo_sand,valid,raw_saturated_mask):
    return _display._neutralise_highlights(linear,photo_sand,valid,raw_saturated_mask,parameters=require_parameters('display_modes'))
def _srgb_encode(linear):
    return _display._srgb_encode(linear,parameters=require_parameters('display_modes'))
def _median_chroma_rgb8(rgb8):
    return _display._median_chroma_rgb8(rgb8,parameters=require_parameters('display_modes'))
def render_display_modes(log_camera_rgb,photo_sand,*,raw_saturated_mask=None,tile_rows=128):
    return _display.render_display_modes(log_camera_rgb,photo_sand,raw_saturated_mask=raw_saturated_mask,
        tile_rows=tile_rows,parameters=require_parameters('display_modes'))

__all__=['CONTRACT','DISPLAY_MODE','DISPLAY_MODES','CLAUDE_CONTRACT_SHA256','M1','A','M_NATURAL','Y_REFERENCE','GAINS','display_contract','display_modes_contract','render_display_modes']

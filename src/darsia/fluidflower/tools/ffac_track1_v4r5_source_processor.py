"""Compatibility names for the configured native source processor."""
from darsia.fluidflower.native.config import require_configuration
from darsia.fluidflower.native import source as _core

class Track1V4R5SourceProcessor(_core.SourceProcessor):
    """Retain the historical constructor while requiring explicit configuration."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, parameters=require_configuration(), **kwargs)

def _validate_output_policy(*args, **kwargs):
    return _core._validate_output_policy(*args, parameters=require_configuration(), **kwargs)

def _output_display_contract(*args, **kwargs):
    return _core.output_display_contract(*args, parameters=require_configuration(), **kwargs)

def _source_outputs(*args, **kwargs):
    return _core.source_outputs(*args, parameters=require_configuration(), **kwargs)

def _read(*args, **kwargs):
    return _core._read(*args, parameters=require_configuration(), **kwargs)

def _h(*args, **kwargs):
    return _core.source_homography(*args, parameters=require_configuration(), **kwargs)

def _chart_xy_baseline_native(*args, **kwargs):
    return _core.chart_baseline_coordinates(*args, parameters=require_configuration(), **kwargs)

def __getattr__(name):
    keys={'CONTRACT':'processor_contract','FILLED_DISPLAY_CONTRACT':'filled_processor_contract','ISOLATED_DISPLAY_CONTRACT':'isolated_processor_contract'}
    if name in keys: return require_configuration()['source'][keys[name]]
    raise AttributeError(name)

__all__=['Track1V4R5SourceProcessor','CONTRACT']

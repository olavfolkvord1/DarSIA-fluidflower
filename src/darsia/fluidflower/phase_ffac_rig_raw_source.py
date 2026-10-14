"""Compatibility names for configured source-bound native radiometry."""
from .native.config import require_configuration
from .native import source as _core

class RigRawSourceProcessor(_core.RigSourceProcessor):
    """Retain the original entry point without duplicate correction equations."""
    def __init__(self, native_manifest_path):
        super().__init__(native_manifest_path, parameters=require_configuration())

def read_bound(*args, **kwargs):
    return _core.read_bound(*args, parameters=require_configuration(), **kwargs)

def verify_calibration_baselines(*args, **kwargs):
    return _core.verify_calibration_baselines(*args, parameters=require_configuration(), **kwargs)

def retain_parent_scientific_pixels(*args, **kwargs):
    return _core.retain_parent_scientific_pixels(*args, parameters=require_configuration(), **kwargs)

def __getattr__(name):
    keys={'CONTRACT':'rig_contract','SHAPE_HW':'native_shape_hw','BOX_XYWH':'content_box_xywh'}
    if name in keys:
        value=require_configuration()['source'][keys[name]]
        return tuple(value) if isinstance(value,list) else value
    raise AttributeError(name)

__all__=['RigRawSourceProcessor','retain_parent_scientific_pixels','CONTRACT']

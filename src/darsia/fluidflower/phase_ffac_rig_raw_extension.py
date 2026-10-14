"""Compatibility entry points for frozen peripheral RAW correction."""
from .native.config import require_configuration
from .native import extension as _core

def frozen_light_from_receipt(*args, **kwargs):
    return _core.frozen_light_from_receipt(*args, parameters=require_configuration(), **kwargs)

def peripheral_rectangles(mask, native_hw=None, origin_xy=None):
    configuration=require_configuration()
    return _core.peripheral_rectangles(mask, native_hw=tuple(configuration['extension']['native_shape_hw']) if native_hw is None else native_hw, origin_xy=tuple(configuration['extension']['native_origin_xy']) if origin_xy is None else origin_xy, parameters=configuration)

def sample_sensor_window(demosaiced,renderer,rectangle,*,origin_xy=None,float_planes=None):
    configuration=require_configuration()
    return _core.sample_sensor_window(demosaiced,renderer,rectangle,origin_xy=tuple(configuration['extension']['native_origin_xy']) if origin_xy is None else origin_xy,float_planes=float_planes,parameters=configuration)

def extend_prepared_source(*args, **kwargs):
    return _core.extend_prepared_source(*args, parameters=require_configuration(), **kwargs)

def __getattr__(name):
    keys={'CONTRACT':'contract','NATIVE_HW':'native_shape_hw','CANVAS_HW':'minimum_canvas_hw','ORIGIN_XY':'native_origin_xy'}
    if name in keys:
        value=require_configuration()['extension'][keys[name]]
        return tuple(value) if isinstance(value,list) else value
    raise AttributeError(name)

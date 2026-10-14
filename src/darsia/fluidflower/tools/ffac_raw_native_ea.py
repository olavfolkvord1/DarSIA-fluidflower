"""Compatibility entry point for the configured edge-aware sensor renderer."""
import numpy as np

from darsia.fluidflower.native import registration as _core
from darsia.fluidflower.native.config import require_parameters
from .ffac_raw_geometry import _legacy_scope


def __getattr__(name):
    if name == "_CHANNEL_ORDER_TEST":
        return np.array(require_parameters("registration")["edge_aware"]["order_fixture"],
                        dtype=np.uint16)
    raise AttributeError(name)


def _check_opencv_cfa_order():
    return _core.check_demosaic_order(parameters=require_parameters("registration"))


def registered_edge_aware_raw_linear(arw_path, camera_jpeg_path, review_native_jpeg_path, *,
                                      expected_source_hashes, content_box):
    parameters = require_parameters("registration")
    source_id, profile_id = _legacy_scope(arw_path, parameters)
    return _core.registered_edge_aware_sensor_linear(arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box,
        parameters=parameters, source_id=source_id, profile_id=profile_id)

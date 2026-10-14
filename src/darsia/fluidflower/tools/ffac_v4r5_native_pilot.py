"""Compatibility facade; numerical implementation is native.photometry.

Frozen parameters are obtained from the explicitly configured native DATA.
The generic core never loads this compatibility module.
"""
from __future__ import annotations
from ..native import photometry as _core
from ..native.config import require_parameters
def native_area_log8(linear_native_rgb):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.native_area_log(linear_native_rgb, parameters=require_parameters('photometry'))

def frame_light_from_native_grid(frame_area_log8, fit, grid, h_ac28_to_frame_native, frame_chart_xy8, frame_chart_usable):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.frame_light_from_native_grid(frame_area_log8, fit, grid, h_ac28_to_frame_native, frame_chart_xy8, frame_chart_usable, parameters=require_parameters('photometry'))

def correct_frame_native_v4r5(linear_native_rgb, fit, virtual_reference, grid, h_ac28_to_frame_native, source_eligible_mask, *, exact_baseline_source, frame_chart_xy8, frame_chart_usable, old_field_ref8, expected_area_log8=None, tile_rows=128):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.correct_frame_native(linear_native_rgb, fit, virtual_reference, grid, h_ac28_to_frame_native, source_eligible_mask, exact_baseline_source=exact_baseline_source, frame_chart_xy8=frame_chart_xy8, frame_chart_usable=frame_chart_usable, old_field_ref8=old_field_ref8, expected_area_log8=expected_area_log8, tile_rows=tile_rows, parameters=require_parameters('photometry'))

"""Compatibility facade; numerical implementation is native.photometry.

Frozen parameters are obtained from the explicitly configured native DATA.
The generic core never loads this compatibility module.
"""
from __future__ import annotations
from ..native import photometry as _core
from ..native.config import require_parameters
def validate_homography(h_ref_to_run):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.validate_homography(h_ref_to_run, parameters=require_parameters('photometry'))

def _map_coordinates(h_ref_to_run, *, source_shape, destination_shape, destination_y0=0, destination_y1=None, run_to_reference=False):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._map_coordinates(h_ref_to_run, source_shape=source_shape, destination_shape=destination_shape, destination_y0=destination_y0, destination_y1=destination_y1, run_to_reference=run_to_reference, parameters=require_parameters('photometry'))

def remap_to_reference(run_array, h_ref_to_run, *, categorical=False, destination_shape=None):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.remap_to_reference(run_array, h_ref_to_run, categorical=categorical, destination_shape=destination_shape, parameters=require_parameters('photometry'))

def chart_xy_to_reference(chart_xy_run8, h_ref_to_run):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.chart_xy_to_reference(chart_xy_run8, h_ref_to_run, parameters=require_parameters('photometry'))

def _remap_reference_tile(ref_array, h_ref_to_run, destination_shape, y0, y1, *, coordinates=None, categorical=False):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._remap_reference_tile(ref_array, h_ref_to_run, destination_shape, y0, y1, coordinates=coordinates, categorical=categorical, parameters=require_parameters('photometry'))

def _native_exact_sand_field(fit_ref, reference, facies_tile, mx_ref8, my_ref8):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._native_exact_sand_field(fit_ref, reference, facies_tile, mx_ref8, my_ref8, parameters=require_parameters('photometry'))

def estimate_frame_light_refgeom(fit_ref, frame_log_run8, h_ref_to_run, *, frame_chart_xy_run8, frame_chart_usable):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.estimate_frame_light_reference_geometry(fit_ref, frame_log_run8, h_ref_to_run, frame_chart_xy_run8=frame_chart_xy_run8, frame_chart_usable=frame_chart_usable, parameters=require_parameters('photometry'))

def apply_refgeom_v5(log_rgb_run, fit_ref, reference, h_ref_to_run, *, facies_run_native=None, light_field_ref8=None, tile_rows=256):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.apply_reference_geometry(log_rgb_run, fit_ref, reference, h_ref_to_run, facies_run_native=facies_run_native, light_field_ref8=light_field_ref8, tile_rows=tile_rows, parameters=require_parameters('photometry'))

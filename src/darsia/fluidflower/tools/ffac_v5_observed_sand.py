"""Compatibility facade; numerical implementation is native.photometry.

Frozen parameters are obtained from the explicitly configured native DATA.
The generic core never loads this compatibility module.
"""
from __future__ import annotations
from ..native import photometry as _core
from ..native.config import require_parameters
class ReferenceGrid(_core.ReferenceGrid):
    """Legacy constructor forms with explicit configured default geometry."""
    def __init__(self, shape_hw=None, origin_cell_xy=None, original_shape_hw=None):
        p = require_parameters("photometry")
        super().__init__(tuple(p["expanded_shape_hw"]) if shape_hw is None else shape_hw,
                         tuple(p["expanded_origin_cell_xy"]) if origin_cell_xy is None else origin_cell_xy,
                         tuple(p["original_shape_hw"]) if original_shape_hw is None else original_shape_hw,
                         parameters=p)

def require_green_donor_only_receipt(receipt):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.require_green_donor_only_receipt(receipt, parameters=require_parameters('photometry'))

def _rgb(array, grid):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._observed_rgb(array, grid, parameters=require_parameters('photometry'))

def _fit_field(green, fac, keep, grid):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._fit_observed_field(green, fac, keep, grid, parameters=require_parameters('photometry'))

def fit_baseline_grid(*, run, logB, facies, chart_xy, chart_log_rgb, ref_chart_log_rgb, chart_usable, matrix, rig_consensus, grid, flat=None):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.fit_observed_baseline(run=run, logB=logB, facies=facies, chart_xy=chart_xy, chart_log_rgb=chart_log_rgb, ref_chart_log_rgb=ref_chart_log_rgb, chart_usable=chart_usable, matrix=matrix, rig_consensus=rig_consensus, grid=grid, flat=flat, parameters=require_parameters('photometry'))

def baseline_field(fit, reference, grid):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.observed_baseline_field(fit, reference, grid, parameters=require_parameters('photometry'))

def apply_grid(log_rgb, fit, reference, grid, *, light_field=None, source_eligible_mask=None):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.apply_grid(log_rgb, fit, reference, grid, light_field=light_field, source_eligible_mask=source_eligible_mask, parameters=require_parameters('photometry'))

def apply_native(log_rgb_run, fit, reference, grid, h_ref_to_run, *, source_eligible_mask, light_field_ref8=None, destination_origin_xy=(0, 0), tile_rows=128):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.apply_native(log_rgb_run, fit, reference, grid, h_ref_to_run, source_eligible_mask=source_eligible_mask, light_field_ref8=light_field_ref8, destination_origin_xy=destination_origin_xy, tile_rows=tile_rows, parameters=require_parameters('photometry'))

def estimate_frame_light_expanded_diagnostic(fit, frame_log_ref8, grid, *, frame_chart_xy_ref8, frame_chart_usable):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.estimate_frame_light_expanded(fit, frame_log_ref8, grid, frame_chart_xy_ref8=frame_chart_xy_ref8, frame_chart_usable=frame_chart_usable, parameters=require_parameters('photometry'))

def make_local_flat(corrected_new, consensus, reference, common_new_support, grid):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.make_local_flat(corrected_new, consensus, reference, common_new_support, grid, parameters=require_parameters('photometry'))

def save_baseline(path, fit, grid, provenance):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.save_baseline(path, fit, grid, provenance, parameters=require_parameters('photometry'))

def load_baseline(path, *, verify=True):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.load_baseline(path, verify=verify, parameters=require_parameters('photometry'))

"""Compatibility facade; numerical implementation is native.photometry.

Frozen parameters are obtained from the explicitly configured native DATA.
The generic core never loads this compatibility module.
"""
from __future__ import annotations
from ..native import photometry as _core
from ..native.config import require_parameters
def apply_old_field_native(corrected_native_log_rgb, field_ref8, grid, h_ref_to_run, *, destination_origin_xy=(0, 0), tile_rows=128):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.apply_old_field_native(corrected_native_log_rgb, field_ref8, grid, h_ref_to_run, destination_origin_xy=destination_origin_xy, tile_rows=tile_rows, parameters=require_parameters('photometry'))

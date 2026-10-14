"""Compatibility facade; numerical implementation is native.photometry.

Frozen parameters are obtained from the explicitly configured native DATA.
The generic core never loads this compatibility module.
"""
from __future__ import annotations
from ..native import photometry as _core
from ..native.config import require_parameters
Baseline = _core.Baseline

def __getattr__(name):
    if name not in {"GRID", "NATIVE", "GB_SAT", "LIGHT_WEIGHTS"}:
        raise AttributeError(name)
    p = require_parameters("photometry")
    values = {"GRID": lambda: tuple(p["grid_shape_hw"]), "NATIVE": lambda: tuple(p["native_shape_hw"]), "GB_SAT": lambda: p["green_blue_saturation"], "LIGHT_WEIGHTS": lambda: __import__("numpy").array(p["light_weights"], dtype="float64")}
    return values[name]()

def _log_rgb(a, shape=None):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._log_rgb(a, shape, parameters=require_parameters('photometry'))

def _basis(h=372, w=684):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.polynomial_basis(h, w, parameters=require_parameters('photometry'))

def _same_fac(f):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._same_fac(f, parameters=require_parameters('photometry'))

def _fit_field(g, fac, keep):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._fit_grid_field(g, fac, keep, parameters=require_parameters('photometry'))

def _colour(log_rgb, inv):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._colour(log_rgb, inv, parameters=require_parameters('photometry'))

def _chart_plane(C, CR, usable, M):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._chart_plane(C, CR, usable, M, parameters=require_parameters('photometry'))

def fit_baseline_grid(*, run, logB, facies, chart_xy, chart_log_rgb, ref_chart_log_rgb, chart_usable, matrix, rig_consensus, flat=None):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.fit_baseline_grid(run=run, logB=logB, facies=facies, chart_xy=chart_xy, chart_log_rgb=chart_log_rgb, ref_chart_log_rgb=ref_chart_log_rgb, chart_usable=chart_usable, matrix=matrix, rig_consensus=rig_consensus, flat=flat, parameters=require_parameters('photometry'))

def baseline_field(fit, ref):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.baseline_field(fit, ref, parameters=require_parameters('photometry'))

def _resize(a, shape):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._resize(a, shape, parameters=require_parameters('photometry'))

def _native_sand_field_tile(fit, ref, facies_tile, lo, width):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core._native_sand_field_tile(fit, ref, facies_tile, lo, width, parameters=require_parameters('photometry'))

def apply_v5(log_rgb, fit, ref, *, facies_native=None, light_field8=None, tile_rows=384):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.apply_registered(log_rgb, fit, ref, facies_native=facies_native, light_field8=light_field8, tile_rows=tile_rows, parameters=require_parameters('photometry'))

def render_ac28_response(corrected_log_linear, response):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.render_response(corrected_log_linear, response, parameters=require_parameters('photometry'))

def load_ac28_response(path):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.load_response(path, parameters=require_parameters('photometry'))

def save_baseline(path, fit, provenance):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.save_grid_baseline(path, fit, provenance, parameters=require_parameters('photometry'))

def load_baseline(path, *, verify=True):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.load_grid_baseline(path, verify=verify, parameters=require_parameters('photometry'))

from ..native import registration as _registration
def registered_raw_grid_from_arw(arw_path, review_to_camera_h, profile, geometry_params):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _registration.registered_sensor_grid_log(arw_path, review_to_camera_h, profile, geometry_params, parameters=require_parameters('registration'))

def sample_chart_log(log8, xy8):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.sample_chart_log(log8, xy8, parameters=require_parameters('photometry'))

def _light_basis(x, y, degree):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.light_basis(x, y, degree, parameters=require_parameters('photometry'))

def estimate_frame_light_grid(fit, frame_log8, *, frame_chart_xy8=None, frame_chart_usable=None, minimum_weak_sand_cells=200):
    """Delegate to the canonical numerical implementation with configured parameters."""
    return _core.estimate_frame_light_grid(fit, frame_log8, frame_chart_xy8=frame_chart_xy8, frame_chart_usable=frame_chart_usable, minimum_weak_sand_cells=minimum_weak_sand_cells, parameters=require_parameters('photometry'))

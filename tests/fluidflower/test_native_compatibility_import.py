"""Import metadata must not request numerical calibration parameters."""

import subprocess
import sys
from pathlib import Path


def test_photometry_function_import_without_numerical_configuration():
    """A legacy saved-model reader can import functions without evaluating them."""
    source = Path(__file__).resolve().parents[2] / "src"
    code = f"""
import os, sys
sys.path.insert(0, {str(source)!r})
os.environ.pop('DARSIA_NATIVE_CONFIGURATION', None)
os.environ.pop('DARSIA_NATIVE_CONFIGURATION_SHA256', None)
from darsia.fluidflower.tools.ffac_v5_photometry import render_ac28_response
from darsia.fluidflower.tools import ffac_v5_photometry as facade
assert callable(render_ac28_response)
try:
    facade.__path__
except AttributeError:
    pass
else:
    raise AssertionError('Unknown module attribute must be rejected')
try:
    facade.GRID
except ValueError as error:
    assert str(error) == 'An explicit native parameter document is required'
else:
    raise AssertionError('Numerical constants still require explicit DATA')
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True)

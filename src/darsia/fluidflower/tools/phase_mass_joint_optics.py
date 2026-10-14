"""Compatibility alias for the shared mass optical signals."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("darsia.fluidflower.mass.phase_mass_joint_optics")

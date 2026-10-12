"""Compatibility alias for DarSIA's serialized optical phase model."""

from __future__ import annotations

import sys

from darsia.fluidflower.phase_training import model as _implementation


sys.modules[__name__] = _implementation

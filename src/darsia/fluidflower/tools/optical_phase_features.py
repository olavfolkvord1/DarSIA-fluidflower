"""Compatibility alias for DarSIA's authoritative optical feature module."""

from __future__ import annotations

import sys

from darsia.fluidflower import phase_features as _implementation


sys.modules[__name__] = _implementation

"""Overflow circulating its water: the displaced water is the absolute weight.

A pump takes water from a container on the scale through the free water and
back. The water the injected CO2 displaces from the porous layers runs over to
the container, so the weight rises with the displaced water and falls as the
gas dissolves. The displaced water is the weight above the reference, the mean
weight of the baseline readings before the injection, at one gram per mL.
Water evaporating lowers the weight directly.

Every setting is a field of :class:`CirculationSettings`.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CirculationSettings:
    """Settings of a circulating overflow; the ``[overflow.circulation]`` table."""

    minimum_readings: int
    """Fewest readings a run needs."""

    def displaced_ml(
        self, times_h: np.ndarray, weights_g: np.ndarray, baseline: np.ndarray
    ) -> tuple[np.ndarray | None, tuple[str, ...]]:
        """Displaced water [mL] at each reading: the weight above the baseline's mean.

        Parameters
        ----------
        times_h : array
            Times of the readings [h since the injection start], increasing.
        weights_g : array
            Scale weights [g].
        baseline : array of bool
            The baseline readings, before the injection and within the
            baseline window.

        Returns
        -------
        tuple
            The volume at each reading, and no reasons: the readings are usable;
            without a baseline reading, no volume and that reason.

        """
        if not baseline.any():
            return None, ("no baseline reading in the window",)
        return weights_g - weights_g[baseline].mean(), ()

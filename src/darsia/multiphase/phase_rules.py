"""Phase rules: phase maps of a run from its phase probabilities.

A phase map gives each pixel of a frame one category: water, dissolved CO2, free
gas or unknown. The labels of a frame follow from its CO2-presence and
gas-within-CO2 probabilities, the CO2 threshold of the settings and the gas
threshold of the phase model; CO2 is only where the phase model found it. The
rules then correct the labels with temporal knowledge:

1. time average: free gas is decided by the gas probability averaged with the
   run's neighbouring frames; small gas bodies of the frame itself, high enough
   above a started injection port, are kept.

Beside each phase map, a rule map records which rule last changed each pixel's
label (:class:`PhaseRule`).

Distances are physical, taken from the grid of the sand map; the area of a
component is the sum of its pixels' areas, which the caller gives. Every
setting is a field of :class:`PhaseRuleSettings`.

"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

import cv2
import numpy as np
from scipy import ndimage

import darsia


class PhaseLabel(IntEnum):
    """Categories of a phase map (uint8)."""

    WATER = 0
    DISSOLVED = 1
    GAS = 2
    UNKNOWN = 255


class PhaseRule(IntEnum):
    """Which rule last changed a pixel's label, in a rule map (uint8).

    The values are those our production stored, so that its rule maps read the
    same.

    """

    UNCHANGED = 0
    TIME_AVERAGE_LOST = 7
    """Free gas the time average made dissolved CO2."""
    TIME_AVERAGE_GAINED = 8
    """Dissolved CO2 the time average made free gas."""


@dataclass(frozen=True)
class PhaseRuleSettings:
    """Settings of the phase labels and rules; one configuration table."""

    co2_threshold: float
    """Smallest CO2-presence probability of CO2."""
    smallest_gas_patch_mm2: float
    """Free gas in 4-connected patches below this area is dissolved CO2."""
    neighbour_window_h: float
    """Time [h] over which a neighbouring frame's weight falls linearly to zero."""
    neighbours_per_side: int
    """Number of nearest earlier and later frames entering the time average."""
    small_bubble_max_mm2: float
    """Averaged gas bodies below this area are dissolved; the frame's own gas
    pieces below it are its small bubbles."""
    small_bubble_min_height_mm: float
    """A small bubble is kept only if it lies wholly higher than this above the
    lowest started port."""


def _co2(labels: np.ndarray) -> np.ndarray:
    """Pixels labelled dissolved CO2 or free gas."""
    return (labels == PhaseLabel.DISSOLVED) | (labels == PhaseLabel.GAS)


def _summed_areas(components: np.ndarray, count: int, pixel_area: float | np.ndarray):
    """Area of each component (index 0: background): its pixels' areas summed."""
    if np.ndim(pixel_area) == 0:
        return np.bincount(components.ravel(), minlength=count) * pixel_area
    return np.bincount(components.ravel(), weights=pixel_area.ravel(), minlength=count)


def _areas(mask: np.ndarray, connectivity: int, pixel_area: float | np.ndarray):
    """Components of a mask, their count and their areas (index 0: background)."""
    count, components = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=connectivity
    )
    return components, _summed_areas(components, count, pixel_area)


def phase_labels(
    co2: np.ndarray,
    gas: np.ndarray,
    domain: np.ndarray,
    pixel_area_mm2: float | np.ndarray,
    settings: PhaseRuleSettings,
    *,
    gas_threshold: float,
) -> np.ndarray:
    """Phase labels of one frame from its two phase probabilities.

    A domain pixel is CO2 where its CO2-presence probability reaches the CO2
    threshold, and free gas where its gas-within-CO2 probability also reaches
    the gas threshold; elsewhere it is water, including pixels without
    probabilities (NaN). Free gas in 4-connected patches smaller than
    ``smallest_gas_patch_mm2`` is dissolved CO2. Pixels outside the domain are
    unknown.

    Parameters
    ----------
    co2, gas : array
        CO2-presence and gas-within-CO2 probabilities of shape (rows, cols).
    domain : array
        Pixels whose phase is decided.
    pixel_area_mm2 : float or array
        Area of each pixel: one value for a uniform grid, or an array of shape
        (rows, cols), e.g. measured pixel areas. The area of a patch is the sum
        of its pixels' areas.
    settings : PhaseRuleSettings
        The CO2 threshold and the smallest gas patch.
    gas_threshold : float
        Gas threshold of the phase model that gave the probabilities
        (``PhaseModel.gas_threshold``).

    Returns
    -------
    array
        Labels (uint8) of :class:`PhaseLabel`.

    """
    present = domain & (co2 >= settings.co2_threshold)
    labels = np.where(domain, PhaseLabel.WATER, PhaseLabel.UNKNOWN).astype(np.uint8)
    labels[present] = PhaseLabel.DISSOLVED
    patches, area = _areas(present & (gas >= gas_threshold), 4, pixel_area_mm2)
    kept = area >= settings.smallest_gas_patch_mm2
    kept[0] = False
    labels[kept[patches]] = PhaseLabel.GAS
    return labels


def _time_average(labels, gas, neighbours, domain, chart, settings, gas_threshold):
    """Decide free gas by the time-averaged gas probability; return the raw gas.

    The gas probability of the frame (weight one) and of each neighbour (weight
    falling linearly from one to zero over the window), each taken where its own
    probabilities give CO2 and zero elsewhere, is averaged. Within the frame's
    CO2 the average decides free gas, except that gas the smallest-patch rule
    removed is not restored, nor is gas on the chart. The raw gas is the frame's
    own gas decision before the smallest-patch rule.
    """
    dtype = gas.dtype.type
    scored = _co2(labels)
    total = np.where(scored, gas, 0)
    weight_sum = 1.0
    for (other_co2, other_gas), gap_h in neighbours:
        weight = max(0.0, 1.0 - gap_h / settings.neighbour_window_h)
        if weight > 0:
            other = domain & (other_co2 >= settings.co2_threshold)
            total += dtype(weight) * np.where(other, other_gas, 0)
            weight_sum += weight
    averaged = scored & (total / dtype(weight_sum) >= gas_threshold)
    raw = scored & (gas >= gas_threshold)
    gained = (labels == PhaseLabel.DISSOLVED) & averaged & ~raw
    if chart is not None:
        gained &= ~chart
    labels[(labels == PhaseLabel.GAS) & ~averaged] = PhaseLabel.DISSOLVED
    labels[gained] = PhaseLabel.GAS
    return raw


def _small_bubbles(labels, raw, lowest_port_y, y, pixel_area, settings):
    """Keep the frame's own small gas bubbles beside the averaged gas bodies.

    Averaged gas bodies below ``small_bubble_max_mm2`` become dissolved CO2. An
    8-connected piece of the frame's own gas below that area, touching no large
    averaged body and lying wholly higher than ``small_bubble_min_height_mm``
    above the lowest started port, is free gas where it is CO2.
    """
    bodies, area = _areas(labels == PhaseLabel.GAS, 8, pixel_area)
    large = area >= settings.small_bubble_max_mm2
    large[0] = False
    labels[(bodies > 0) & ~large[bodies]] = PhaseLabel.DISSOLVED
    if lowest_port_y is None:
        return
    near = cv2.dilate(large[bodies].astype(np.uint8), np.ones((3, 3), np.uint8))
    pieces, area = _areas(raw, 8, pixel_area)
    small = area < settings.small_bubble_max_mm2
    small[0] = False
    small[np.unique(pieces[(near > 0) & (pieces > 0)])] = False
    bottom = [box[0].stop - 1 for box in ndimage.find_objects(pieces)]
    lowest = np.concatenate(([np.inf], y[bottom]))
    small &= lowest > lowest_port_y + settings.small_bubble_min_height_mm / 1000
    labels[small[pieces] & _co2(labels)] = PhaseLabel.GAS


def _time_average_rules(before, labels):
    """The rule map of the time average: free gas lost and gained."""
    rules = np.zeros(labels.shape, np.uint8)
    lost = (before == PhaseLabel.GAS) & (labels == PhaseLabel.DISSOLVED)
    rules[lost] = PhaseRule.TIME_AVERAGE_LOST
    gained = (before == PhaseLabel.DISSOLVED) & (labels == PhaseLabel.GAS)
    rules[gained] = PhaseRule.TIME_AVERAGE_GAINED
    return rules


def phase_maps(
    probabilities: Sequence[tuple[np.ndarray, np.ndarray]],
    dates: Sequence[datetime],
    sand_map: darsia.Image,
    domain: np.ndarray,
    pixel_area_mm2: float | np.ndarray,
    protocol: darsia.InjectionProtocol,
    settings: PhaseRuleSettings,
    chart: np.ndarray | None = None,
    *,
    gas_threshold: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Phase maps of a run's frames, by its phase probabilities and the rules.

    Each frame's labels (:func:`phase_labels`) are corrected by the rules listed
    in the module description. A frame's time average uses its nearest
    ``neighbours_per_side`` earlier and later frames, read from
    ``probabilities`` by index. Each frame's rule map records which rule last
    changed each pixel's label.

    Parameters
    ----------
    probabilities : sequence of (array, array)
        CO2-presence and gas-within-CO2 probabilities of each frame, in time
        order, on the grid of the sand map.
    dates : sequence of datetime
        Acquisition time of each frame.
    sand_map : darsia.Image
        Facies ids of the run's grid, with its physical dimensions and origin.
    domain : array
        Pixels whose phase is decided.
    pixel_area_mm2 : float or array
        Area of each pixel, as for :func:`phase_labels`.
    protocol : darsia.InjectionProtocol
        Injection timetable; each injecting location is a port.
    settings : PhaseRuleSettings
        Settings of the labels and rules.
    chart : array, optional
        The colour checker's pixels.
    gas_threshold : float
        Gas threshold of the phase model that gave the probabilities, as for
        :func:`phase_labels`.

    Yields
    ------
    tuple of arrays
        The phase map (uint8) and the rule map (uint8, :class:`PhaseRule`) of
        each frame, in order; the rule map is zero where no rule changed the
        label.

    """
    _, y = darsia.pixel_centres(sand_map)

    injecting = protocol.df[protocol.df["rate_kg_s"] > 0]
    first_start = dict(injecting.groupby(["location_x", "location_y"])["start"].min())

    for index, ((co2, gas), date) in enumerate(zip(probabilities, dates)):
        started = [port for port, start in first_start.items() if start <= date]
        earlier = [j for j in range(index - 1, -1, -1) if dates[j] < date]
        later = [j for j in range(index + 1, len(dates)) if dates[j] > date]
        neighbours = [
            (probabilities[j], abs((dates[j] - date).total_seconds()) / 3600)
            for j in earlier[: settings.neighbours_per_side]
            + later[: settings.neighbours_per_side]
        ]
        labels = phase_labels(
            co2, gas, domain, pixel_area_mm2, settings, gas_threshold=gas_threshold
        )
        before = labels.copy()
        raw = _time_average(
            labels, gas, neighbours, domain, chart, settings, gas_threshold
        )
        lowest = min((port[1] for port in started), default=None)
        _small_bubbles(labels, raw, lowest, y, pixel_area_mm2, settings)
        rules = _time_average_rules(before, labels)
        yield labels, rules

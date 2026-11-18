"""Active injector corridors carrying inferred gas through optically clear water."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GasBridge:
    """One current-frame hidden stretch and its smooth transverse profile."""

    hidden: np.ndarray
    meeting: tuple[int, int]
    profile: np.ndarray
    transition: np.ndarray
    length_mm: float


def hidden_bridges(zone, co2, known_water, porous, pitch_mm, settings):
    """Derive hidden stretches only from this frame's active, ordered links.

    The first currently observed CO2 cell is the meeting point. History may
    guide the route, but cannot provide the meeting saturation of this frame.
    UNKNOWN, seals and nonporous cells never become inferred support.
    """
    bridges = []
    radius = settings.link_width_mm / 2
    central = -np.expm1(-radius / settings.fade_length_mm)
    for link in zone.links.values():
        points = link.path_coordinates
        seen = co2[points[:, 0], points[:, 1]]
        indices = np.flatnonzero(seen)
        if not len(indices) or indices[0] == 0:
            continue
        first = int(indices[0])
        meeting = tuple(int(value) for value in points[first])
        leading = np.zeros(co2.shape, bool)
        leading[points[: first + 1, 0], points[: first + 1, 1]] = True
        distance = ndimage.distance_transform_edt(~leading, sampling=pitch_mm)
        corridor = (distance < radius) & porous & zone.contour
        hidden = corridor & known_water
        if not hidden.any():
            continue
        # A copied centreline fraction tapers to zero at both corridor sides.
        # Normalising at the centre preserves the visible meeting fraction.
        profile = (
            -np.expm1(-np.maximum(0, radius - distance) / settings.fade_length_mm)
            / central
        )
        profile[~corridor] = 0
        rr = (np.arange(co2.shape[0]) - meeting[0]) * pitch_mm[0]
        cc = (np.arange(co2.shape[1]) - meeting[1]) * pitch_mm[1]
        t = np.clip(np.hypot(rr[:, None], cc[None, :]) / settings.fade_length_mm, 0, 1)
        transition = (1 - t) ** 2 * (1 + 2 * t) * profile
        transition[~co2] = 0
        length = np.linalg.norm(
            np.diff(points[: first + 1], axis=0) * pitch_mm, axis=1
        ).sum()
        bridges.append(
            GasBridge(
                hidden,
                meeting,
                profile.astype(np.float32),
                transition.astype(np.float32),
                float(length),
            )
        )
    return bridges


def bridge_scores(base, candidate, gas, bridges):
    """Carry the meeting coefficient into the same scalar volume solve.

    For unsaturated pixels the fraction is coefficient/T. Copying the meeting
    coefficient therefore copies its saturation for every T, including the
    common cap. A compact cubic transition in the existing fade length joins
    the visible score to the tapered corridor. Distinct bridges must not share
    allocated hidden cells or alter one another's meeting coefficient. Such
    interference needs a scientific rule and is refused; exactly identical
    fields are deduplicated without changing their meaning.
    """
    hidden = np.zeros(base.shape, bool)
    all_hidden = np.zeros(base.shape, bool)
    for bridge in bridges:
        all_hidden |= bridge.hidden
    carried = np.zeros_like(base)
    scale = np.ones_like(base)
    fixed = np.zeros_like(base)
    total_blend = np.zeros_like(base)
    total_target = np.zeros_like(base)
    prepared = []
    for bridge in bridges:
        point = bridge.meeting
        if not (candidate[point] or gas[point]):
            continue
        coefficient = base[point]
        scored = bool(np.isfinite(coefficient) and coefficient > 0)
        if not scored and not gas[point]:
            continue
        mask = bridge.hidden & candidate
        duplicate = False
        for other, other_mask, _, other_scored in prepared:
            if point == other.meeting and all(
                np.array_equal(getattr(bridge, name), getattr(other, name))
                for name in ("hidden", "profile", "transition")
            ):
                duplicate = True
                break
            if (mask & other_mask).any():
                raise ValueError(
                    "Active gas bridges have incompatible overlapping hidden "
                    f"support at meetings {other.meeting} and {point}; "
                    "no shared bridge field has been defined."
                )
            if (
                scored
                and candidate[other.meeting]
                and bridge.transition[other.meeting] > 0
            ) or (other_scored and candidate[point] and other.transition[point] > 0):
                raise ValueError(
                    "An active gas bridge's transition changes another bridge's "
                    f"meeting fraction at {other.meeting} or {point}; "
                    "no shared bridge field has been defined."
                )
        if not duplicate:
            prepared.append((bridge, mask, coefficient, scored))
    for bridge, mask, coefficient, scored in prepared:
        point = bridge.meeting
        if gas[point]:
            # Model GAS is fixed at its cap even when its score is zero. This
            # establishes volume, not a made-up probability for nearby pixels.
            np.maximum(fixed, np.where(mask, bridge.profile, 0), out=fixed)
        if not scored:
            continue
        replace = mask & (~hidden | (coefficient > carried))
        same = mask & hidden & (coefficient == carried)
        carried[replace] = coefficient
        scale[replace] = bridge.profile[replace]
        scale[same] = np.maximum(scale[same], bridge.profile[same])
        hidden |= mask
        blend = bridge.transition * candidate
        total_blend += blend
        total_target += blend * coefficient * bridge.profile
    mixed = base.copy()
    used = total_blend > 0
    alpha = np.minimum(total_blend[used], 1)
    mixed[used] = (1 - alpha) * base[used] + alpha * total_target[used] / total_blend[
        used
    ]
    mixed[all_hidden] = carried[all_hidden]
    return mixed, fixed, scale

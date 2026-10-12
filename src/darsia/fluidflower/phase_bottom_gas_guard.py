"""Opt-in gas-only guard for shallow components at the registered lower boundary.

The native domain, rather than the display canvas, defines the lower edge.
Only a four-connected gas component near that edge and confined to a shallow
physical band is reclassified as aqueous CO2. Interior-connected gas and gas
near a recently active injector are retained. This is a conservative display
classification guard, not a change to total CO2 support or concentrations.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .phase_rendering import two_phase_masks
from .project import sha256_file

VERSION = "native-bottom-gas-boundary-v1"
PROFILE_KEY = "bottom_gas_guard"


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    maximum_depth_mm: float = 8.0
    boundary_contact_mm: float = 2.5
    activation_after_last_injection_h: float = 1.0
    source_protection_radius_mm: float = 65.0
    source_protection_after_end_h: float = 3.0

    def __post_init__(self):
        if self.format != VERSION:
            raise ValueError("Invalid bottom-gas guard format")
        for name in ("maximum_depth_mm", "boundary_contact_mm",
                     "activation_after_last_injection_h", "source_protection_radius_mm",
                     "source_protection_after_end_h"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid bottom-gas guard {name}")
        if (self.maximum_depth_mm > 25
                or self.boundary_contact_mm > self.maximum_depth_mm
                or self.source_protection_radius_mm > 150
                or self.activation_after_last_injection_h > 24
                or self.source_protection_after_end_h > 24):
            raise ValueError("Bottom-gas guard settings exceed conservative bounds")


def runtime_identity(inherited, specification):
    return hashlib.sha256(json.dumps(dict(inherited=inherited,
        settings=asdict(Settings(**specification)), code_sha256=sha256_file(Path(__file__)),
        opencv=cv2.__version__), sort_keys=True).encode()).hexdigest()


def _geometry(context, shape):
    domain = np.asarray(context.domain, bool)
    x = np.asarray(context.x_m, float)
    y = np.asarray(context.y_m, float)
    if (domain.shape != shape or x.shape != (shape[1],) or y.shape != (shape[0],)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) < 0)
            or not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-4)
            or not np.allclose(np.diff(y), y[1] - y[0], rtol=1e-4)):
        raise ValueError("Bottom-gas guard requires registered domain and uniform x/y coordinates")
    x_native = getattr(context, "x_m_native", None)
    y_native = getattr(context, "y_m_native", None)
    if (x_native is None) != (y_native is None):
        raise ValueError("Bottom-gas guard requires both exact native coordinate fields")
    if x_native is not None:
        # Preserve float32 geometry fields; converting a full native grid to
        # float64 for every frame would dominate this guard's runtime.
        x_native = np.asarray(x_native)
        y_native = np.asarray(y_native)
        if (x_native.shape != shape or y_native.shape != shape
                or not np.isfinite(x_native).all() or not np.isfinite(y_native).all()
                or not (np.diff(x_native, axis=1)[domain[:, 1:]] > 0).all()
                or not (np.diff(y_native, axis=0)[domain[1:, :]] < 0).all()):
            raise ValueError("Bottom-gas guard requires valid exact native physical coordinates")
    return domain, x, y, x_native, y_native


def _prepare_geometry(context):
    domain, x, y, x_native, y_native = _geometry(context, context.domain.shape)
    valid_columns = np.flatnonzero(domain.any(axis=0))
    floor = domain.shape[0] - 1 - domain[::-1].argmax(axis=0)
    minimum_dy_mm = None
    if y_native is not None:
        minimum_dy_mm = float(np.min(-np.diff(y_native, axis=0)[domain[1:, :]])) * 1e3
        if not np.isfinite(minimum_dy_mm) or minimum_dy_mm <= 0:
            raise ValueError("Bottom-gas guard requires positive exact native vertical spacing")
    return domain, x, y, x_native, y_native, valid_columns, floor, minimum_dy_mm


def _near_recent_source(box, x, y, stages, elapsed_h, settings,
                        x_native=None, y_native=None):
    rows, cols = box
    left, right = x[cols.start], x[cols.stop - 1]
    bottom, top = y[rows.stop - 1], y[rows.start]
    radius = settings.source_protection_radius_mm * 1e-3
    for stage in stages:
        start, end = float(stage["start_h"]), float(stage["end_h"])
        if not start <= elapsed_h <= end + settings.source_protection_after_end_h:
            continue
        source_x, source_y = float(stage["x_m"]), float(stage["y_m"])
        if x_native is not None:
            # A native rectangle is skewed in physical space by the camera
            # homography. Search the whole rectangle, conservatively keeping
            # ambiguous gas near a recently active source.
            distance2 = ((x_native[rows, cols] - source_x) ** 2
                         + (y_native[rows, cols] - source_y) ** 2)
            near = bool(np.any(distance2 <= radius ** 2))
        else:
            dx = max(left - source_x, 0.0, source_x - right)
            dy = max(bottom - source_y, 0.0, source_y - top)
            near = bool(np.hypot(dx, dy) <= radius)
        if near:
            return True
    return False


def _exact_geometry_audit(x_native):
    return {"physical_coordinate_basis": "exact-native-2d"} if x_native is not None else {}


def clean_labels(labels, context, elapsed_h, settings=Settings(), *, _prepared=None):
    labels = np.asarray(labels)
    if labels.ndim != 2 or min(labels.shape) < 2 or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("Bottom-gas guard requires categorical phase labels")
    prepared = _prepare_geometry(context) if _prepared is None else _prepared
    domain, x, y, x_native, y_native, valid_columns, floor, minimum_dy_mm = prepared
    if labels.shape != domain.shape:
        raise ValueError("Bottom-gas guard requires the registered domain grid")
    if ((labels == 2) & ~domain).any():
        raise ValueError("Bottom-gas guard requires gas within the registered domain")
    elapsed_h = float(elapsed_h)
    if not np.isfinite(elapsed_h):
        raise ValueError("Bottom-gas guard requires finite elapsed time")
    stages = tuple(context.injection_stages)
    if not stages:
        raise ValueError("Bottom-gas guard requires registered injection stages")
    for stage in stages:
        values = np.asarray([stage["x_m"], stage["y_m"], stage["start_h"],
                             stage["end_h"]], float)
        if not np.isfinite(values).all() or values[3] <= values[2]:
            raise ValueError("Bottom-gas guard requires valid injection stages")
    latest_end = max(float(stage["end_h"]) for stage in stages)
    activation_elapsed_h = latest_end + settings.activation_after_last_injection_h
    if elapsed_h <= activation_elapsed_h:
        return labels.astype(np.uint8, copy=True), dict(
            format=VERSION, active=False, activation_elapsed_h=activation_elapsed_h,
            maximum_depth_mm=settings.maximum_depth_mm,
            boundary_contact_mm=settings.boundary_contact_mm,
            activation_after_last_injection_h=settings.activation_after_last_injection_h,
            source_protection_radius_mm=settings.source_protection_radius_mm,
            source_protection_after_end_h=settings.source_protection_after_end_h,
            connectivity=4, bottom_touching_components=0,
            shallow_bottom_components=0, interior_connected_components=0,
            source_protected_components=0, reclassified_components=0,
            reclassified_gas_pixels=0, co2_support_unchanged=True,
            **_exact_geometry_audit(x_native))

    _, components, stats, _ = cv2.connectedComponentsWithStats(
        (labels == 2).astype(np.uint8), connectivity=4)
    # Probe just the registered floor band, without allocating a full-frame
    # distance map. This includes small gaps left by registration or rasterization.
    near_boundary = np.zeros(len(stats), bool)
    dy_mm = abs(float(y[1] - y[0])) * 1e3
    if y_native is not None:
        contact_offsets = int(np.ceil(settings.boundary_contact_mm / minimum_dy_mm)) + 1
    else:
        contact_offsets = int(settings.boundary_contact_mm / dy_mm) + 1
    for offset in range(contact_offsets):
        rows = floor[valid_columns] - offset
        columns = valid_columns[rows >= 0]
        rows = rows[rows >= 0]
        inside = domain[rows, columns]
        if y_native is not None:
            inside &= ((y_native[rows, columns] - y_native[floor[columns], columns])
                       * 1e3 <= settings.boundary_contact_mm)
        near_boundary[components[rows[inside], columns[inside]]] = True
    near_boundary[0] = False
    touching = np.flatnonzero(near_boundary)
    remove = np.zeros(len(stats), bool)
    shallow_count = source_protected = interior_connected = 0
    for component in touching:
        left = int(stats[component, cv2.CC_STAT_LEFT])
        top = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        # Max floor within the component's span deliberately overestimates
        # reach into the interior on a sloping boundary, preserving ambiguity.
        deepest_floor = int(floor[left:left + width].max())
        if y_native is not None:
            columns = np.arange(left, left + width)
            depth_mm = float(np.max(y_native[top, columns]
                                    - y_native[floor[columns], columns])) * 1e3
        else:
            depth_mm = (deepest_floor - top) * dy_mm
        if depth_mm > settings.maximum_depth_mm:
            interior_connected += 1
            continue
        shallow_count += 1
        box = (slice(top, top + height), slice(left, left + width))
        if _near_recent_source(box, x, y, stages, elapsed_h, settings,
                               x_native, y_native):
            source_protected += 1
            continue
        remove[component] = True

    changed = remove[components]
    updated = labels.astype(np.uint8, copy=True)
    updated[changed] = 1
    report = dict(format=VERSION, active=True,
        activation_elapsed_h=activation_elapsed_h,
        maximum_depth_mm=settings.maximum_depth_mm,
        boundary_contact_mm=settings.boundary_contact_mm,
        activation_after_last_injection_h=settings.activation_after_last_injection_h,
        source_protection_radius_mm=settings.source_protection_radius_mm,
        source_protection_after_end_h=settings.source_protection_after_end_h,
        connectivity=4, bottom_touching_components=int(len(touching)),
        shallow_bottom_components=int(shallow_count),
        interior_connected_components=int(interior_connected),
        source_protected_components=int(source_protected),
        reclassified_components=int(remove.sum()),
        reclassified_gas_pixels=int(changed.sum()), co2_support_unchanged=True,
        **_exact_geometry_audit(x_native))
    return updated, report


class BottomGasSequence:
    def __init__(self, inherited, context, specification):
        self.inherited, self.context = inherited, context
        self.settings = Settings(**specification)
        self.last_diagnostics = {}
        self._prepared_geometry = _prepare_geometry(context)

    def bind_reference(self, baseline):
        self.inherited.bind_reference(baseline)

    def apply_with_reference(self, predictions, elapsed_h, current):
        result, report = self.inherited.apply_with_reference(predictions, elapsed_h, current)
        masks = two_phase_masks(result)
        labels = np.where(masks["gas"], 2, masks["aqueous"]).astype(np.uint8)
        cleaned, audit = clean_labels(labels, self.context, elapsed_h, self.settings,
                                      _prepared=self._prepared_geometry)
        changed = (labels == 2) & (cleaned == 1)
        updated = dict(result)
        gas = np.where(changed, 0., result["gas_probability_conditional_on_co2"]).astype(np.float32)
        updated["gas_probability_conditional_on_co2"] = gas
        co2 = result["co2_presence_probability"]
        for key, values in (("aqueous_co2_probability", co2 * (1 - gas)),
                            ("free_co2_gas_probability", co2 * gas)):
            if key in result:
                updated[key] = np.where(changed, values, result[key]).astype(np.float32)
        self.last_diagnostics = {**self.inherited.last_diagnostics, "bottom_gas_reclassified": changed}
        return updated, {**report, PROFILE_KEY: audit}

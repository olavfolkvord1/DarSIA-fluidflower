"""Read-only Policy A gates and spatial diagnostics for categorical predictions.

This module does not call a model, read human annotations, or interpret mass.
The CPU float64 area sums define the acceptance decision. Distances use the
approved projective metric chain, rather than a nominal constant pixel size.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

SPEC_SHA256 = "62bf2970077c9a270f85fe3f046e48811af70d5e1c1761cd729d6289782bef32"
CONTRACT = "ffac-selected-policy-a-required-coarse-spatial-qa-v2"
IDENTITY_CONTRACT = "ffac-selected-required-cpu-identity-v2"
# Explicit original 43-run cohort minus the four chemical exclusions.
RUNS = ("ac17", "ac18", "ac19", "ac21", "ac22", "ac23", "ac25", "ac26", "ac28", "ac29", "ac30", "ac31", "ac32", "ac33", "ac34", "ac35", "ac38", "ac39", "ac40", "ac41", "ac42", "ac43", "ac44", "ac45", "ac46", "ac47", "ac48", "ac49", "ac50", "ac52", "ac53", "ac54", "ac55", "ac56", "ac57", "ac58", "ac59", "ac60", "ac61")
EXPECTED_ANCHORS = 782


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def asset(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path))


class EvidenceReader:
    """Verify each immutable asset once while its path/size/mtime is unchanged."""
    def __init__(self):
        self.verified = {}
        self.records = {}
        self.arrays = {}

    def path(self, descriptor):
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("sha256"), str):
            raise ValueError("A source-bound asset descriptor is required")
        path = Path(descriptor["path"])
        stat = path.stat()
        key = (str(path.resolve()).casefold(), stat.st_size, stat.st_mtime_ns)
        expected = descriptor["sha256"]
        if self.verified.get(key) != expected:
            if digest(path) != expected:
                raise ValueError("Policy A evidence SHA changed: " + str(path))
            self.verified[key] = expected
        return path

    def record(self, descriptor):
        path = self.path(descriptor)
        key = (str(path.resolve()), descriptor["sha256"])
        if key not in self.records:
            self.records[key] = json.loads(path.read_text(encoding="utf-8-sig"))
        return self.records[key]

    def array(self, descriptor, key=None, *, cache=False):
        path = self.path(descriptor)
        key = descriptor.get("array_key", key)
        identity = (str(path.resolve()), descriptor["sha256"], key)
        if cache and identity in self.arrays:
            return self.arrays[identity]
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if key is None:
                    raise ValueError("NPZ evidence requires an explicit array key")
                value = archive[key].copy()
        else:
            value = np.load(path, allow_pickle=False)
        if cache:
            value.flags.writeable = False
            self.arrays[identity] = value
        return value


def require_identity_gate(descriptor, reader, *, expected_runs=RUNS):
    """Accept the real composite producer seal without relabelling old rows."""
    from .phase_ffac_selected_composite_identity_admission import require_composite_identity_gate
    return require_composite_identity_gate(descriptor, reader, expected_runs=expected_runs)


def _grids(parent, current, area):
    a, b, area = np.asarray(parent), np.asarray(current), np.asarray(area)
    if (a.ndim != 2 or a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != b.shape
            or area.shape != a.shape or not np.isfinite(area).all() or np.any(area < 0)
            or not np.isin(a, (0, 1, 2, 255)).all() or not np.isin(b, (0, 1, 2, 255)).all()):
        raise ValueError("Policy A QA requires exact categorical and measured-area grids")
    return a, b, area


def coarse_studio_gate(parent, current, pixel_area_mm2):
    a, b, area = _grids(parent, current, pixel_area_mm2)
    result = dict(contract=CONTRACT, categorical_changed_pixels=int(np.count_nonzero(a != b)),
                  support_changed_pixels=int(np.count_nonzero((a == 255) != (b == 255))),
                  acceptance_backend="CPU float64 measured pixel-area sums")
    measure = lambda mask: float(np.sum(area[mask], dtype=np.float64) / 100.)
    for phase, fraction, old, new in (("co2", .01, np.isin(a, (1, 2)), np.isin(b, (1, 2))),
                                       ("gas", .02, a == 2, b == 2)):
        changed, added, removed = old ^ new, new & ~old, old & ~new
        parent_area, xor_area = measure(old), measure(changed)
        limit = max(fraction * parent_area, .5)
        result[phase] = dict(parent_cm2=parent_area, current_cm2=measure(new), xor_cm2=xor_area,
            relative_xor=None if parent_area == 0 else xor_area/parent_area,
            added_cm2=measure(added), removed_cm2=measure(removed), changed_pixels=int(changed.sum()),
            added_pixels=int(added.sum()), removed_pixels=int(removed.sum()),
            relative_limit=fraction, absolute_floor_cm2=.5, acceptance_limit_cm2=limit,
            pass_gate=bool(xor_area <= limit))
    result["all_pass"] = result["co2"]["pass_gate"] and result["gas"]["pass_gate"]
    return result


def projected_points(matrix, xy):
    matrix, xy = np.asarray(matrix, np.float64), np.asarray(xy, np.float64)
    if matrix.shape != (3, 3) or xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(matrix).all():
        raise ValueError("A finite source-bound homography and pixel coordinates are required")
    transformed = np.column_stack((xy, np.ones(len(xy)))) @ matrix.T
    valid = np.isfinite(transformed).all(axis=1) & (np.abs(transformed[:, 2]) >= 1e-12)
    result = np.full((len(xy), 2), np.nan, np.float64)
    result[valid] = transformed[valid, :2]/transformed[valid, 2, None]
    return result


def approved_canvas_to_mm(revision):
    """Same approved metric chain as the existing source-bound L1 diagnostic."""
    approval = revision["physical_scale_revision"]
    if approval.get("physical_scale_approved") is not True:
        return None
    parameters = revision["parameters"]
    unit = np.sqrt(float(parameters["metric_pixel_area_mm2"])) * float(approval["linear_factor"])
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("Invalid approved metric scale")
    origin = np.eye(3); origin[:2, 2] = revision["native_origin_xy"]
    scale = np.diag([unit, unit, 1.])
    return scale @ np.asarray(parameters["H_frame_native_to_metric_pixel"], np.float64) @ origin


def metric_injectors(reference, revision):
    """Measured metric scale, explicitly unverified nominal injector locations."""
    approval = revision["physical_scale_revision"]
    unit = np.sqrt(float(revision["parameters"]["metric_pixel_area_mm2"])) * float(approval["linear_factor"])
    return {name: np.asarray(reference[name][:2], np.float64)*unit for name in ("L1", "L2")}


def _distance_summary(values):
    values = np.asarray(values, np.float64)
    known = values[np.isfinite(values)]
    return dict(known_pixels=int(len(known)), unknown_pixels=int(len(values)-len(known)),
                min_mm=None if not len(known) else float(known.min()),
                median_mm=None if not len(known) else float(np.median(known)),
                p95_mm=None if not len(known) else float(np.percentile(known, 95)),
                max_mm=None if not len(known) else float(known.max()))


def change_spatial_details(parent, current, area, *, zones=None, canvas_to_mm=None,
                           injectors_mm=None, geometry_known=None, zone_names=None):
    """Every changed pixel is retained; absent geometry produces NaN, never zero.

    Plume-edge distance is to the exact physical centres of parent CO2 boundary
    pixels (8-neighbour erosion). It is not a signed or subpixel distance.
    """
    a, b, area = _grids(parent, current, area)
    y, x = np.nonzero(a != b)
    xy = np.column_stack((x, y))
    if geometry_known is not None and np.asarray(geometry_known).shape != a.shape:
        raise ValueError("Known-geometry mask differs from phase grid")
    if zones is not None and np.asarray(zones).shape != a.shape:
        raise ValueError("Zone grid differs from phase grid")
    points = np.full((len(xy), 2), np.nan) if canvas_to_mm is None else projected_points(canvas_to_mm, xy)
    if geometry_known is not None:
        points[~np.asarray(geometry_known, bool)[y, x]] = np.nan
    known = np.isfinite(points).all(axis=1)
    distances = {"L1": np.full(len(xy), np.nan), "L2": np.full(len(xy), np.nan),
                 "parent_plume_edge": np.full(len(xy), np.nan)}
    for name in ("L1", "L2"):
        target = None if injectors_mm is None else injectors_mm.get(name)
        if target is not None and np.asarray(target).shape == (2,) and np.isfinite(target).all():
            distances[name][known] = np.linalg.norm(points[known]-target, axis=1)
    plume = np.isin(a, (1, 2))
    edge = plume & ~binary_erosion(plume, structure=np.ones((3, 3), bool), border_value=0)
    if geometry_known is not None:
        edge &= np.asarray(geometry_known, bool)
    ey, ex = np.nonzero(edge)
    if canvas_to_mm is not None and len(ex) and known.any():
        boundary = projected_points(canvas_to_mm, np.column_stack((ex, ey)))
        boundary = boundary[np.isfinite(boundary).all(axis=1)]
        if len(boundary):
            distances["parent_plume_edge"][known] = cKDTree(boundary).query(points[known], workers=1)[0]
    z = np.full(len(x), -1, np.int16) if zones is None else np.asarray(zones)[y, x].astype(np.int16)
    details = dict(x=x.astype(np.int32), y=y.astype(np.int32), parent=a[y, x], current=b[y, x],
                   pixel_area_mm2=area[y, x].astype(np.float64), zone=z, metric_x_mm=points[:, 0],
                   metric_y_mm=points[:, 1], geometry_known=known)
    details.update({name.lower()+"_distance_mm": values for name, values in distances.items()})
    by_zone = []
    for code in np.unique(z):
        mask = z == code
        old_co2, new_co2 = np.isin(a[y, x], (1, 2)), np.isin(b[y, x], (1, 2))
        by_zone.append(dict(zone=int(code), name=(zone_names or {}).get(int(code), "unknown" if code < 0 else f"reference zone {code}"),
            changed_pixels=int(mask.sum()), changed_cm2=float(area[y[mask], x[mask]].sum(dtype=np.float64)/100),
            co2_xor_cm2=float(details["pixel_area_mm2"][mask & (old_co2 ^ new_co2)].sum(dtype=np.float64)/100),
            gas_xor_cm2=float(details["pixel_area_mm2"][mask & ((a[y, x] == 2) ^ (b[y, x] == 2))].sum(dtype=np.float64)/100),
            distances={name: _distance_summary(values[mask]) for name, values in distances.items()}))
    transitions = {f"{old}->{new}": int(np.count_nonzero((a[y, x] == old) & (b[y, x] == new)))
                   for old, new in zip(*np.unique(np.column_stack((a[y, x], b[y, x])), axis=0).T)} if len(x) else {}
    return dict(changed_pixels=len(x), geometry_known_pixels=int(known.sum()),
        geometry_unknown_pixels=int((~known).sum()), zones=by_zone, transitions=transitions,
        distances={name: _distance_summary(values) for name, values in distances.items()},
        plume_edge_definition="parent CO2 union boundary pixel centres; 8-neighbour erosion; exact projective metric",
        injector_position_status="unknown" if injectors_mm is None else "protocol_proxy_unverified"), details


def source_studio_index(source):
    if source.get("studio_anchor") is not True or type(source.get("studio_frame_index")) is not int:
        raise ValueError("Scorer export requires an original Studio acquisition")
    if source.get("studio_source_sha256", source["source_sha256"]) != source["source_sha256"]:
        raise ValueError("Studio frame index belongs to another camera acquisition")
    return source["studio_frame_index"]


def verify_phase_source(saved, source, pipeline_sha256):
    if (saved.get("complete") is not True or saved.get("binding", {}).get("source") != source
            or saved.get("binding", {}).get("pipeline_sha256") != pipeline_sha256
            or saved.get("run") != source["run"] or saved.get("source_sha256") != source["source_sha256"]
            or saved.get("elapsed_h") != source["elapsed_h"]
            or saved.get("frame_index") != source["work_index"]):
        raise ValueError("Selected phase receipt has another camera source/time/work identity")


def policy_context(plan_path, root, identity_descriptor, reader):
    """Metadata admission shared by the independent report and scorer export."""
    plan_descriptor = asset(plan_path)
    plan = reader.record(plan_descriptor)
    spec = plan.get("binding", {}).get("approved_spec")
    if not spec or spec.get("sha256") != SPEC_SHA256:
        raise ValueError("Policy A report requires the revised approved specification")
    reader.path(spec)
    if set(plan.get("runs", {})) != set(RUNS):
        raise ValueError("Policy A report requires the exact 39-run cohort")
    bound_identity = plan.get("identity_gate", plan.get("binding", {}).get("identity_gate"))
    if bound_identity != identity_descriptor:
        raise ValueError("CPU identity gate is not bound into the Policy A phase plan")
    identity = require_identity_gate(identity_descriptor, reader)
    chain = plan["parent_chain"]
    parent_plan = reader.record(chain["final3_plan"])
    seal = reader.record(chain["parent_seal"])
    if (seal.get("complete") is not True or seal.get("contract") != "ffac-g3-t4c-p1-final3-candidate-v1"
            or seal.get("source_plan_sha256") != chain["final3_plan"]["sha256"]):
        raise ValueError("Policy A source parent is not the sealed P1 final3")
    expected = {(row["run"], int(row["frame_index"])): row for row in parent_plan["source_inventory"]
                if row["run"] in RUNS}
    if len(expected) != EXPECTED_ANCHORS:
        raise ValueError("Parent Studio source inventory is not the original 782 frames")
    inventories = {}
    anchors = {}
    for run, state in sorted(plan["runs"].items()):
        inventory = reader.record(state["source_inventory"])
        if inventory.get("run") != run:
            raise ValueError("Source inventory belongs to another run")
        inventories[run] = inventory
        for source in inventory["sources"]:
            if not source.get("studio_anchor"):
                continue
            index = source_studio_index(source)
            key = (run, index)
            old = expected.get(key)
            if (key in anchors or old is None or old["source_sha256"] != source["source_sha256"]
                    or source["run"] != run):
                raise ValueError("Studio source identity differs from P1 final3")
            anchors[key] = source
    if set(anchors) != set(expected):
        raise ValueError("Policy A source inventory does not cover every old Studio frame")
    return dict(plan=plan, plan_descriptor=plan_descriptor, root=Path(root),
                identity=identity, inventories=inventories, anchors=anchors, expected=expected)


def load_prediction(descriptor, reader):
    from PIL import Image
    with Image.open(reader.path(descriptor)) as image:
        if image.mode != "L":
            raise ValueError("Scorer predictions must preserve P1 final3 grayscale category encoding")
        value = np.asarray(image).copy()
    if value.dtype != np.uint8 or value.ndim != 2 or not np.isin(value, (0, 1, 2, 255)).all():
        raise ValueError("Invalid predicted categorical PNG")
    return value


def anchor_evidence(context, key, reader):
    """Resolve by acquisition SHA; never use the new work index as Studio index."""
    source = context["anchors"][key]
    run, index = key
    phase_descriptor = asset(context["root"]/"selected_phase_frames"/run/(source["source_sha256"]+".json"))
    phase = reader.record(phase_descriptor)
    verify_phase_source(phase, source, context["plan"]["pipeline_sha256"])
    parent_descriptor = asset(Path(context["plan"]["parent_chain"]["parent"])/"rig_camera_frames"/run/f"frame_{index:03d}.json")
    parent = reader.record(parent_descriptor)
    if (parent.get("complete") is not True or parent.get("run") != run
            or parent.get("frame_index") != index or parent.get("source_sha256") != source["source_sha256"]):
        raise ValueError("P1 final3 prediction is bound to a different Studio acquisition")
    current = load_prediction(phase["labels_map"], reader)
    before = load_prediction(parent["labels_map"], reader)
    area = reader.array(phase["rig_domain_revision"]["area_map"], "pixel_area_mm2")
    return source, phase_descriptor, phase, parent_descriptor, parent, before, current, area


__all__ = ["SPEC_SHA256", "CONTRACT", "RUNS", "EXPECTED_ANCHORS", "EvidenceReader", "asset",
           "require_identity_gate", "coarse_studio_gate", "change_spatial_details", "approved_canvas_to_mm",
           "metric_injectors", "source_studio_index", "verify_phase_source", "projected_points"]

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

SPEC_SHA256 = "46060237a9b158d3fcddcd96dc1c4ff12b0b3b57611c0ff9523117d22030204b"
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
    """Verify all actual own histories and every nondiagnostic identity product.

    The all39 stage contains frames 0--2. The full-runs predecessor supplies
    the remaining AC18/AC52 history. Reused frame receipts must be byte-exact.
    No gate flag can substitute for those independently bound row receipts.
    """
    gate = reader.record(descriptor)
    flags = ("complete", "all_39_complete", "cpu_only", "required_identity_passed",
             "computed_inherited_identity_required", "frozen_gas_diagnostic_only")
    if (gate.get("contract") != IDENTITY_CONTRACT or gate.get("stage") != "all39"
            or any(gate.get(key) is not True for key in flags)):
        raise ValueError("All-run required CPU own-history identity must pass before Policy A")
    plan = reader.record(gate["identity_plan"])
    binding = plan["phase_campaign_binding"]
    helper = "darsia.fluidflower.phase_ffac_selected_required_identity"
    if (set(plan["runs"]) != set(expected_runs) or binding.get("spec_sha256") != SPEC_SHA256
            or binding.get("stage_conventions_independently_bound") is not True
            or binding.get("fresh_inherited_map_required") is not True
            or binding.get("archived_inherited_labels_used_as_inference_input") is not False):
        raise ValueError("Required identity has another cohort/spec or inherited-map convention")
    audit_descriptor = binding["frozen_gas_audit"]
    audit = reader.record(audit_descriptor)
    if (audit.get("contract") != "ffac-selected-required-identity-code-consumer-audit-v1"
            or audit.get("complete") is not True
            or audit.get("required_helper_binding_pending") is not False
            or audit.get("frozen_gas_no_production_reads") is not True
            or audit.get("legacy_reconstruction_unreachable") is not True
            or audit.get("frozen_gas_compare_and_report_only") is not True
            or audit.get("required_identity_helper_sha256") != binding.get("code", {}).get(helper, {}).get("sha256")
            or audit.get("approved_spec", {}).get("sha256") != SPEC_SHA256):
        raise ValueError("Required identity does not bind the final no-reader/helper code audit")
    inventory = reader.record(binding["identity_inventory"])
    if (inventory.get("complete") is not True or set(inventory["runs"]) != set(expected_runs)
            or inventory.get("every_sequence_starts_from_baseline") is not True
            or inventory.get("only_original_parent_history") is not True
            or inventory.get("selected_history_used") is not False
            or inventory.get("inherited_map_must_be_computed_not_loaded") is not True):
        raise ValueError("Required own-history source inventory is incomplete")
    expected = {}
    for run, state in inventory["runs"].items():
        frames = state["frames"]
        indices = [row["studio_frame_index"] for row in frames]
        if (indices != list(range(len(frames))) or len(frames) < 3
                or run not in ("ac18", "ac52") and len(frames) != 3
                or state.get("start_from_baseline") is not True):
            raise ValueError("Required inventory omits an original own-history frame")
        expected.update({(run, row["studio_frame_index"]): row for row in frames})
    if set(expected_runs) == set(RUNS) and len(expected) != 153:
        raise ValueError("Required inventory must retain all 153 original identity frames")
    folder = Path(descriptor["path"]).parent
    stages = {"all39": descriptor,
              "full-runs": asset(folder/"full-runs_stage.json"),
              "ac19": asset(folder/"ac19_stage.json")}
    checked, compared, diagnostic = {}, {}, []
    probability_names = {"g3_pixel.npy", "regional_frozen.npy", "regional_refit.npy", "co2.packbits.npy", "extra_masks.packbits.npz"}
    mask_names = {"photo_sand_domain", "feature_support", "regional_native_domain", "flatfield_valid",
                  "feature_eligible", "optical_presence_ge_half", "frozen_gas", "frozen_unresolved"}
    for stage_name, stage_descriptor in stages.items():
        stage = reader.record(stage_descriptor)
        if (stage.get("contract") != IDENTITY_CONTRACT or stage.get("stage") != stage_name
                or stage.get("complete") is not True or stage.get("required_identity_passed") is not True
                or stage.get("cpu_only") is not True or stage.get("identity_plan") != gate["identity_plan"]):
            raise ValueError("Required own-history predecessor stage failed or changed")
        expected_stage = {run: set(range(3)) for run in expected_runs} if stage_name == "all39" else (
            {run: set(range(len(inventory["runs"][run]["frames"]))) for run in ("ac18", "ac52")}
            if stage_name == "full-runs" else {"ac19": {0, 1}})
        actual_stage = {}
        for result in stage.get("results", []):
            run = result["run"]
            if run in actual_stage or result.get("complete") is not True:
                raise ValueError("Duplicate or incomplete required identity run")
            actual_stage[run] = set()
            for item in result.get("receipts", []):
                row = reader.record(item)
                index = row["original_studio_frame_index"]
                key = (run, index)
                if key not in expected or index in actual_stage[run]:
                    raise ValueError("Duplicate or unexpected required identity frame")
                actual_stage[run].add(index)
                source = expected[key]
                if (row.get("contract") != IDENTITY_CONTRACT or row.get("run") != run
                        or any(row.get(flag) is not True for flag in ("complete", "cpu_only", "all_pass"))
                        or row.get("identity_plan") != gate["identity_plan"]
                        or row.get("source_sha256") != source["source_sha256"]
                        or row.get("elapsed_h") != source["elapsed_h"]
                        or row.get("previous_source_sha256") != source["original_previous_source_sha256"]
                        or row.get("labels", {}).get("bit_identical") is not True):
                    raise ValueError("Required identity row has a changed source/history or label failure")
                products = row.get("stored_probabilities", {})
                comparisons = products.get("comparisons", {})
                if (products.get("all_bit_identical") is not True or set(comparisons) != probability_names
                        or any(comparisons[name].get("bit_identical") is not True
                               for name in probability_names-{"extra_masks.packbits.npz"})
                        or products.get("frozen_gas_audit") != audit_descriptor
                        or products.get("receipt") != source["original_probability_receipt"]):
                    raise ValueError("A required stored probability product is missing or nonidentical")
                masks = products.get("extra_masks", {})
                mask_checks = masks.get("comparisons", {})
                if (set(mask_checks) != mask_names or masks.get("frozen_gas_required") is not False
                        or masks.get("all_required_bit_identical") is not True
                        or any(mask_checks[name].get("required") is not True or mask_checks[name].get("bit_identical") is not True
                               for name in mask_names-{"frozen_gas"})):
                    raise ValueError("A required nondiagnostic extra mask was suppressed or differs")
                frozen_gas = mask_checks["frozen_gas"]
                if (frozen_gas.get("required") is not False
                        or type(frozen_gas.get("changed_pixels")) is not int
                        or frozen_gas["changed_pixels"] < 0 or row.get("frozen_gas") != frozen_gas):
                    raise ValueError("Diagnostic frozen_gas counts must be preserved per frame")
                inherited = products.get("inherited_map", {})
                inherited_checks = inherited.get("comparisons", {})
                if (inherited.get("computed_from_scientific_images") is not True
                        or inherited.get("archived_labels_used_as_inference_input") is not False
                        or inherited.get("all_bit_identical") is not True
                        or set(inherited_checks) != {"co2_union", "water", "unresolved"}
                        or any(value.get("bit_identical") is not True for value in inherited_checks.values())
                        or inherited.get("expected_original_stage_map") != source["original_inherited_map"]
                        or not row.get("generated_inherited_map", {}).get("sha256")):
                    raise ValueError("Fresh inherited CO2/water/unresolved identity was not proven")
                if key in checked and checked[key] != item:
                    raise ValueError("Reused own-history receipt is not byte-identical")
                if key not in checked:
                    diagnostic.append(dict(run=run, original_studio_frame_index=index, **frozen_gas))
                checked[key] = item
                compared[key] = row
        if actual_stage != expected_stage:
            raise ValueError("Required stage omits complete AC18/AC52 histories or all39 frame 2")
    if set(checked) != set(expected):
        raise ValueError("Required identity comparisons do not cover every original own-history frame")
    return dict(identity_gate=descriptor, identity_plan=gate["identity_plan"],
        required_identity_stages=stages, runs=list(expected_runs),
        compared_row_receipts=[checked[key] for key in sorted(checked)],
        frozen_gas_diagnostics=diagnostic, frozen_gas_audit=audit_descriptor,
        verified=True, mode="historical-stage-conventions", code=binding["code"],
        source_preparation=binding["source_preparation"])


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

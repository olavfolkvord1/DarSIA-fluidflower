"""Section 5f photographs, rig edges and Studio views from saved outputs.

No inference, scientific array, label, session or model is changed. Individual
crops retain full pixel resolution; the contact sheet is a display thumbnail.
The figure receipt binds every photograph, saved map and class-mask input.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

from darsia.fluidflower.phase_ffac_rig_geometry import transform_points
from darsia.fluidflower.phase_pixel_area import rig_phase_shape
from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
from darsia.fluidflower.tools.phase_view_overlay import display_phase_overlay

DEFAULT_RUNS = ("ac28", "ac47", "ac56")
DEFAULT_TIMES = (4., 24.)
PLAN_FILE = "rig_camera_inference_plan.json"
CONTRACT = "ffac-rig-camera-section5f-figures-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _plan_path(root, *, allow_probe=False):
    production = root / PLAN_FILE
    if production.is_file():
        return production
    probe = root / "rig_camera_probe_plan.json"
    if allow_probe and probe.is_file():
        return probe
    return production


def asset(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_bound(value, *, base="."):
    path = Path(value["path"])
    path = path if path.is_absolute() else Path(base) / path
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != value["sha256"]:
        raise ValueError("Figure input SHA changed: " + str(path))
    return path, data


def select_target_sources(inventory, *, runs=DEFAULT_RUNS, target_times=DEFAULT_TIMES):
    """Choose from the complete frozen timeline, even before maps are ready."""
    result = []
    for run in runs:
        rows = [row for row in inventory if row["run"] == run]
        if not rows or len({r["frame_index"] for r in rows}) != len(rows):
            raise ValueError("Figure run has an absent or ambiguous frozen timeline: " + run)
        if not np.isfinite([r["elapsed_h"] for r in rows]).all():
            raise ValueError("Figure timeline contains nonfinite camera times")
        for target in target_times:
            if not np.isfinite(target):
                raise ValueError("Figure target must be a finite camera time")
            row = min(rows, key=lambda r: (abs(float(r["elapsed_h"]) - target), r["frame_index"]))
            result.append(dict(run=run, frame_index=row["frame_index"], elapsed_h=row["elapsed_h"],
                               source_sha256=row["source_sha256"], target_h=float(target),
                               camera_offset_h=float(row["elapsed_h"] - target)))
    return result


def _photo_asset(root, source, inference):
    """Final source contract, or exact unchanged-parent photograph for a pilot."""
    run, index = source["run"], source["frame_index"]
    native_path = root / "source_alias/native_frames" / run / "manifest.json"
    if native_path.is_file():
        run_receipt_path = root / "rig_camera_revision/source_runs" / (run + ".json")
        run_receipt = read_json(run_receipt_path)
        _, native_bytes = read_bound(run_receipt["native_manifest"])
        native = json.loads(native_bytes)
        row = native["frame_sources"][index]
        _, input_bytes = read_bound(run_receipt["input_manifest"])
        input_row = json.loads(input_bytes)["frames"][index]
        photo = input_row["photographs"]["natural"]
        if (row["source_sha256"] != source["source_sha256"] or row["frame_index"] != index
                or float(row["elapsed_h"]) != float(source["elapsed_h"])
                or input_row["source_sha256"] != source["source_sha256"]
                or row["v5_display_modes"]["natural"]["jpeg_sha256"] != photo["sha256"]):
            raise ValueError("Figure photograph belongs to another saved source/frame")
        return photo, dict(method="final-source-photograph", native_manifest=run_receipt["native_manifest"],
                           input_manifest=run_receipt["input_manifest"], source_run_receipt=asset(run_receipt_path))
    if source.get("source_changed"):
        raise FileNotFoundError("New-source Natural photograph has not yet been materialized")
    tasks = {t["task_id"]: t for t in inference["upstream_plan"]["source_inventory"]["scientific_tasks"]}
    task = tasks[source["scientific_task_id"]]
    _, native_bytes = read_bound(task["parent_native_manifest"])
    native = json.loads(native_bytes)
    row = native["frame_sources"][source["parent_source_frame_index"]]
    if row["source_sha256"] != source["source_sha256"]:
        raise ValueError("Pilot photograph is not the exact retained parent acquisition")
    declaration = row["v5_display_modes"]["natural"]
    return dict(path=declaration["jpeg_path"], sha256=declaration["jpeg_sha256"]), dict(
        method="exact-retained-parent-Natural-photograph", parent_native_manifest=task["parent_native_manifest"],
        parent_source_frame_index=source["parent_source_frame_index"])


def _l1_reference(root, inference):
    parent = Path(inference["upstream_plan"]["binding"]["parent"])
    profile_path = parent / "phase_optical_regions.json"
    profile = read_json(profile_path)
    rule = profile["gas_barrier_rule_b"]
    manifest_path, manifest_data = read_bound(rule["manifest"], base=parent)
    manifest = json.loads(manifest_data)
    derivation_asset = rule["barrier_derivation"]
    if derivation_asset["sha256"] != manifest["barrier_derivation_sha256"]:
        raise ValueError("The protocol/reference-grid L1 point differs from the frozen Rule B derivation")
    derivation_path, derivation_data = read_bound(derivation_asset, base=parent)
    derivation = json.loads(derivation_data)
    injectors = derivation["reference_topology"]["injectors"]
    matches = [p for p in injectors if p["name"] == "L1"]
    if len(matches) != 1:
        raise ValueError("A source-bound protocol/reference-grid L1 point is required for its display crop")
    return np.asarray(matches[0]["xy_px"], float), dict(
        **asset(derivation_path), parent_rule_manifest=asset(manifest_path), parent_profile=asset(profile_path))


def crop_boxes(rig_domain, l1_canvas_xy):
    """Use the actual saved domain and bound injector, never predicted gas."""
    domain = np.asarray(rig_domain)
    l1 = np.asarray(l1_canvas_xy, float)
    if domain.dtype != bool or domain.ndim != 2 or not domain.any() or l1.shape != (2,) or not np.isfinite(l1).all():
        raise ValueError("Figure crop needs an actual boolean rig domain and bound L1 coordinate")
    ys, xs = np.nonzero(domain)
    xmin, xmax, ymin, ymax = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    height, width = domain.shape
    def box(left, top, right, bottom):
        return [max(0, int(np.floor(left))), max(0, int(np.floor(top))),
                min(width, int(np.ceil(right))), min(height, int(np.ceil(bottom)))]
    def bottom_near(x):
        column = max(0, min(width-1, int(round(x))))
        occupied = np.flatnonzero(domain[:, column])
        return int(occupied[-1]) if len(occupied) else ymax
    left_bottom, right_bottom, l1_bottom = bottom_near(xmin+24), bottom_near(xmax-24), bottom_near(l1[0])
    return dict(lower_left=box(xmin-128, left_bottom-512, xmin+640, left_bottom+128),
                lower_right=box(xmax-640, right_bottom-512, xmax+128, right_bottom+128),
                bottom_at_L1=box(l1[0]-512, l1_bottom-512, l1[0]+512, l1_bottom+128),
                left_wall=box(xmin-128, ymin-64, xmin+512, ymax+64),
                right_wall=box(xmax-512, ymin-64, xmax+128, ymax+64))


def _rgb_qimage(image):
    from PySide6.QtGui import QImage
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    return np.frombuffer(rgb.constBits(), np.uint8).reshape(rgb.height(), rgb.bytesPerLine())[:, :rgb.width()*3].reshape(rgb.height(), rgb.width(), 3).copy()


def render_saved_views(photo, overlay, rig_domain, filled):
    """Exact Studio phase filtering, opacity50% and default filled hatching."""
    from darsia.fluidflower.tools.phase_model_comparison_view import compose
    filtered = display_phase_overlay(overlay, "all", photo_sand=rig_domain)
    filtered._filled_phase_mask = filled
    studio = _rgb_qimage(compose(photo, filtered, .5, enabled=True, filled=True))
    edge_view = _rgb_qimage(photo)
    boundary = rig_domain & ~binary_erosion(rig_domain, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], bool), border_value=0)
    # This two-pixel display stroke does not alter or erode the saved domain.
    edge_view[binary_dilation(boundary, structure=np.ones((3,3), bool))] = (75, 245, 105)
    return edge_view, studio


def _rendering_code():
    from darsia.fluidflower import phase_filled_display, phase_rig_display_canvas
    from darsia.fluidflower.tools import phase_model_comparison_view, phase_view_overlay
    return {module.__name__: asset(module.__file__) for module in (
        phase_model_comparison_view, phase_view_overlay, phase_filled_display, phase_rig_display_canvas)}


def _contact_sheet(title, image_pairs):
    cell_w, cell_h, header_h = 620, 400, 90
    sheet = Image.new("RGB", (cell_w*2+40, header_h + len(image_pairs)*(cell_h+35)), "#202326")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), title, fill="white")
    draw.text((20, 34), "Natural photograph + actual rig-domain edge", fill="#4bf569")
    draw.text((cell_w+30, 34), "Studio: phases50%, filled-pixel hatching ON", fill="white")
    draw.text((20, 56), "Validation only; relative AC28 scale remains provisional. Full-resolution crops accompany this sheet.", fill="#c0c6cb")
    for index, (name, left, right) in enumerate(image_pairs):
        y = header_h + index*(cell_h+35)
        for side, image in enumerate((left, right)):
            view = image.copy(); view.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = 20 + side*(cell_w+10)
            draw.text((x, y), name.replace("_", " "), fill="white")
            sheet.paste(view, (x+(cell_w-view.width)//2, y+24))
    return sheet


def write_frame_figures(candidate, run, frame_index, output, *, target_h=None):
    """Render one completed receipt, including a pilot without a sealed run."""
    from PySide6.QtGui import QImage
    root, output = Path(candidate), Path(output)
    if output.exists():
        raise FileExistsError("Figures must use a new output directory")
    inference_path = _plan_path(root, allow_probe=True)
    inference = read_json(inference_path)
    matches = [s for s in inference["source_inventory"] if s["run"] == run and s["frame_index"] == frame_index]
    if len(matches) != 1:
        raise ValueError("Figure frame lacks one exact frozen inventory entry")
    source = matches[0]
    receipt_path = root / "rig_camera_frames" / run / f"frame_{frame_index:03d}.json"
    receipt = read_json(receipt_path)
    if (receipt.get("complete") is not True or receipt["binding"]["source"] != source
            or receipt["binding"]["pipeline_sha256"] != inference["pipeline_sha256"]):
        raise ValueError("Figure map receipt differs from its immutable source/pipeline")
    revision = receipt["rig_domain_revision"]
    shape = rig_phase_shape(revision)
    photo_asset, photo_binding = _photo_asset(root, source, inference)
    _, photo_data = read_bound(photo_asset, base=root)
    _, overlay_data = read_bound(receipt["maps"]["combined"], base=root)
    _, classes_data = read_bound(revision["classes"], base=root)
    with np.load(io.BytesIO(classes_data), allow_pickle=False) as archive:
        domain = archive["rig_domain"]
        observed, unobserved = archive["filled_observed"], archive["filled_unobserved"]
    if any(value.dtype != bool or value.shape != shape for value in (domain, observed, unobserved)):
        raise ValueError("Figure support flags differ from their declared complete phase grid")
    if np.any(observed & unobserved) or np.any((observed | unobserved) & ~domain):
        raise ValueError("Figure filled flags overlap or lie outside the saved rig domain")
    photo = QImage.fromData(photo_data)
    photo = pad_rig_photograph(photo, revision)
    overlay = QImage.fromData(overlay_data)
    if overlay.isNull() or (overlay.height(), overlay.width()) != shape:
        raise ValueError("Figure map is not on its declared complete phase grid")
    edge, studio = render_saved_views(photo, overlay, domain, observed | unobserved)
    l1, injector_binding = _l1_reference(root, inference)
    geometry = source["rig_geometry"]
    H = geometry["area_parameters"]["H_ac28_to_frame_native"]
    l1_native = transform_points(l1[None], H)[0]
    origin = np.asarray(revision["native_origin_xy"], int)
    l1_canvas = l1_native - origin
    boxes = crop_boxes(domain, l1_canvas)
    output.mkdir(parents=True)
    pairs, files = [], []
    for name, box in boxes.items():
        photo_crop = Image.fromarray(edge).crop(tuple(box))
        studio_crop = Image.fromarray(studio).crop(tuple(box))
        for kind, image in (("photo_edge", photo_crop), ("studio", studio_crop)):
            path = output / f"{name}.{kind}.png"; image.save(path)
            files.append(dict(region=name, rendering=kind, crop_box_canvas_xyxy=box,
                              full_resolution=True, figure=asset(path)))
        pairs.append((name, photo_crop, studio_crop))
    elapsed = float(source["elapsed_h"])
    title = f"{run.upper()} | native frame{frame_index} | camera t={elapsed:.6f} h"
    if target_h is not None:
        title += f" | nearest requested{float(target_h):g} h (offset{elapsed-float(target_h):+.6f} h)"
    sheet_path = output / "contact_sheet.png"; _contact_sheet(title, pairs).save(sheet_path)
    result = dict(contract=CONTRACT, candidate=str(root), run=run, frame_index=frame_index,
        elapsed_h=elapsed, target_h=target_h, source_sha256=source["source_sha256"],
        pipeline_sha256=inference["pipeline_sha256"], phase_shape_hw=list(shape),
        photograph_canvas_wh=revision["canvas_wh"], native_origin_xy=revision["native_origin_xy"],
        inference_plan=asset(inference_path), frame_receipt=asset(receipt_path),
        photograph=photo_asset, photographic_binding=photo_binding, combined_map=receipt["maps"]["combined"],
        support_classes=revision["classes"], injector_reference=injector_binding,
        l1_native_xy=l1_native.tolist(), l1_canvas_xy=l1_canvas.tolist(),
        contact_sheet=asset(sheet_path), contact_sheet_is_thumbnail=True, crops=files,
        rendering=dict(photo_mode="natural", phase_layer="all", opacity=.5, filled_hatching=True,
                       actual_saved_domain_outline=True, display_stroke_dilation_px=1),
        code=asset(__file__), rendering_code=_rendering_code(),
        scientific_float_arrays_read=False, labels_or_models_changed=False,
        report_only_not_a_release_gate=True, partial_input_probe=bool(inference["binding"].get("probe_only"))
            if "binding" in inference else False,
        scientific_acceptance=False, mass_eligible=False,
        warning="Validation only. Relative area uses assigned AC28 scale; actual rig width remains pending.")
    (output / "figure_receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_rig_camera_figures(candidate, output, *, runs=DEFAULT_RUNS, target_times=DEFAULT_TIMES):
    root, output = Path(candidate), Path(output)
    if output.exists():
        raise FileExistsError("Figures must use a new output directory")
    inference = read_json(root / PLAN_FILE)
    selected = select_target_sources(inference["source_inventory"], runs=runs, target_times=target_times)
    # Do not create a misleading partial report from nearest completed maps.
    for row in selected:
        read_json(root / "rig_camera_frames" / row["run"] / f"frame_{row['frame_index']:03d}.json")
    output.mkdir(parents=True)
    figures = []
    for row in selected:
        folder = output / f"{row['run']}_target{row['target_h']:g}h_frame{row['frame_index']:03d}"
        write_frame_figures(root, row["run"], row["frame_index"], folder, target_h=row["target_h"])
        figures.append(dict(**row, receipt=asset(folder / "figure_receipt.json")))
    result = dict(contract=CONTRACT, candidate=str(root), inference_plan=asset(root / PLAN_FILE),
                  runs=list(runs), targets_h=list(target_times), figures=figures,
                  report_only_not_a_release_gate=True, labels_or_models_changed=False)
    (output / "section5f_figures.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--times", nargs="+", type=float, default=list(DEFAULT_TIMES))
    parser.add_argument("--frame", help="Pilot display-only figure RUN:zero-based-index")
    args = parser.parse_args(argv)
    if args.frame:
        run, index = args.frame.split(":")
        result = write_frame_figures(args.candidate, run.casefold(), int(index), args.output)
        print(json.dumps(dict(output=args.output, run=result["run"], frame_index=result["frame_index"])))
    else:
        result = write_rig_camera_figures(args.candidate, args.output, runs=args.runs, target_times=args.times)
        print(json.dumps(dict(output=args.output, figures=len(result["figures"]))))


if __name__ == "__main__":
    main()

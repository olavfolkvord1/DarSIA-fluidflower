"""New, source-bound R3 §1 report on the unchanged R2 saved frames.

This stage cannot fit B1 or start 39-run mass production. It evaluates the
separate excluded estimates at B0 and compares the exact main mass with R2.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gc
import gzip
import importlib
import json
import os
from pathlib import Path
import time
import uuid

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from .. import phase_ffac_common_mass_inputs as source
from .. import phase_ffac_common_mass_round2 as r2
from .. import phase_ffac_common_mass_round2_production as saved
from .. import phase_ffac_common_mass_round3 as engine
from .. import phase_ffac_common_mass_round3_inputs as geometry_inputs
from . import run_ffac_common_mass_round2 as adapter

CONTRACT = "ffac-round3-saved-excluded-parts-report-v1"
SESSION = {}


def code_inventory():
    names = ("phase_ffac_common_mass", "phase_ffac_common_mass_inputs",
        "phase_ffac_common_mass_round2", "phase_ffac_common_mass_round2_inputs",
        "phase_ffac_common_mass_round2_production", "phase_ffac_common_mass_round2_backend",
        "phase_ffac_common_mass_round3", "phase_ffac_common_mass_round3_inputs",
        "concentration_joint", "concentration_native", "tools.run_ffac_common_mass_round2")
    return {"darsia.fluidflower."+name: source.asset(importlib.import_module(
        "darsia.fluidflower."+name).__file__) for name in names}


def init_worker(plan_descriptor, code_binding):
    import cv2
    cv2.setNumThreads(1)
    for name, descriptor in code_binding.items():
        module = importlib.import_module(name)
        if source.sha256(module.__file__) != descriptor["sha256"]:
            raise ValueError("Immutable §1 worker implementation differs: "+name)
    plan = source.read_bound_json(plan_descriptor)
    if source.sha256(plan["r2_freeze"]["path"]) != engine.ROUND2_FREEZE_SHA256:
        raise ValueError("The approved B0 source is not the frozen R2 calibration")
    adapter.init_worker(plan["r2_input_plan"]["path"], plan["r1_freeze"]["path"], "cpu")
    reader = adapter._SESSION["reader"]
    provenance = reader.json(plan["physical_coordinate_provenance"])
    with np.load(reader.path(plan["physical_reference"]), allow_pickle=False) as archive:
        x, y = archive["x_m"], archive["y_m"]
        matrix = geometry_inputs.physical_homography(provenance, x.shape)
        validation = geometry_inputs.validate_reference_coordinates(matrix, x, y)
    points, phi, swi = geometry_inputs.read_measured_properties(
        reader.path(plan["depth_measurements"]), reader.path(plan["facies_properties"]))
    floor = np.load(reader.path(plan["pooled_floor_curve"]), allow_pickle=False)
    SESSION.update(plan=plan, plan_descriptor=plan_descriptor, code_binding=code_binding,
        matrix=matrix, coordinate_validation=validation, depth_points=points,
        porosity=phi, swi=swi, floor_curve=floor)


def commit_json(folder, name, record):
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        raise FileExistsError("Do not overwrite an R3 report frame: "+str(folder))
    temporary = folder.with_name("."+folder.name+"."+uuid.uuid4().hex+".partial")
    temporary.mkdir()
    with gzip.open(temporary / "native_statistics.json.gz", "wt", encoding="utf-8", compresslevel=3) as stream:
        json.dump(record.pop("native_statistics"), stream, allow_nan=False, separators=(",", ":"))
    record["native_statistics"] = source.asset(temporary / "native_statistics.json.gz")
    record["native_statistics"]["path"] = str(folder / "native_statistics.json.gz")
    source.write_new_json(temporary / name, record)
    temporary.rename(folder)
    return record


def render_parts(frame, parts, candidate_root, output):
    from PIL import Image, ImageDraw
    from .report_ffac_p1_final2 import parent_photo
    rgb, photo = parent_photo(candidate_root, frame.run_id, frame.frame_index)
    if rgb.shape[:2] != frame.domain.shape:
        raise ValueError("Report photograph and excluded masks use different canvases")
    colours = dict(bottom_strip=(255, 180, 15), below_photo_filled=(180, 110, 255),
        other_not_mass_eligible=(230, 50, 130), unquantified_facies=(30, 200, 240))
    marked = rgb.copy()
    for part, mask in parts.items():
        colour = np.asarray(colours.get(part, (255, 255, 255)), np.float32)
        marked[mask] = np.rint(marked[mask]*.45+colour*.55).astype(np.uint8)
    panels = [Image.fromarray(v) for v in (rgb, marked)]
    for image in panels:
        image.thumbnail((1400, 850), Image.Resampling.LANCZOS)
    width, height = max(v.width for v in panels), max(v.height for v in panels)
    figure = Image.new("RGB", (width*2, height+85), "#171a20")
    draw = ImageDraw.Draw(figure)
    draw.text((12, 10), f"{frame.run_id.upper()} f{frame.frame_index} t={frame.elapsed_h:.4f} h - frozen R2 photo / CO2 outside main M", fill="white")
    draw.text((12, 32), "Yellow: blocked bottom; purple: below photo; pink: other ineligible; cyan: unquantified facies", fill="white")
    draw.text((12, 54), "Estimates reported separately. Main mass, phase labels and original files unchanged.", fill="#bccdde")
    for n, image in enumerate(panels):
        figure.paste(image, (n*width, 85))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    figure.save(output)
    return dict(figure=source.asset(output), photo=photo)


def report_run(task):
    run, job, campaign_sha, frame_indexes, backend = task
    if run not in set(r2.HELDOUT_RUNS) | {"ac38"}:
        raise ValueError("Initial §1 report is the twelve held-out runs and AC38 only")
    plan = SESSION["plan"]
    rows = adapter.rows_for_run(run)
    anchor_metadata = saved.nearest(rows, 2.25)
    frame, response = adapter.prepare(anchor_metadata, plan["r2_freeze"]["path"])
    anchor = r2.anchor_from_response(response)
    del frame, response
    gc.collect()
    r2_frozen = source.read_bound_json(plan["r2_freeze"])
    parameters = r2_frozen["parameters"]
    reader = adapter._SESSION["reader"]
    candidate = adapter._SESSION["index"]["candidate_root"]
    started = time.perf_counter(); records = []
    nearest24 = saved.nearest(rows, 24.)["frame_index"]
    for metadata in rows:
        index = metadata["frame_index"]
        if frame_indexes and index not in frame_indexes:
            continue
        folder = Path(job) / "saved_parts/frames" / run / f"frame_{index:03d}"
        receipt_path = folder / "receipt.json"
        if receipt_path.is_file():
            record = saved.read_json(receipt_path)
            if record.get("complete") is not True or record.get("campaign_sha256") != campaign_sha:
                raise ValueError("Resume encountered a partial or differently bound report frame")
            records.append(dict(run=run, frame_index=index, cached=True)); continue
        tick = time.perf_counter()
        frame, response = adapter.prepare(metadata, plan["r2_freeze"]["path"])
        receipt = reader.json(metadata["receipt"])
        revision = receipt["rig_domain_revision"]
        raw_geometry = receipt["full_extended_scientific"]["binding"]["geometry"]
        xm, ym = geometry_inputs.frame_coordinates(SESSION["matrix"], raw_geometry, revision)
        geometry = engine.build_estimate_geometry(xm, ym, frame.facies, SESSION["depth_points"],
            porosity_by_facies=SESSION["porosity"], swi_by_facies=SESSION["swi"],
            source_binding={k: plan[k] for k in ("physical_reference", "physical_coordinate_provenance",
                "depth_measurements", "facies_properties")})
        del xm, ym
        photo = geometry_inputs.measured_photo_support(frame, receipt, reader)
        own = engine.prepare_own_colour_response(response, adapter._SESSION["path_folder"],
            geometry, photo_support=photo)
        row = engine.summarize_round3_response(response, geometry, own_colour=own,
            photo_support=photo, no_outflow_evidence=adapter.no_outflow_proxy(frame, metadata),
            light_epoch=frame.provenance.get("light_epoch", saved.light_group(run)),
            source_binding=metadata, backend=backend, scope="saved_baseline_diagnostic")
        prediction = engine.evaluate_round3_summary(row, parameters, anchor)
        exact = r2.evaluate_round2_frame(response, parameters, anchor, fields=False)
        old_path = Path(plan["r2_job"]) / "production" / run / f"frame_{index:03d}" / "receipt.json"
        old = reader.json(source.asset(old_path))
        errors = {key: abs(exact[key]-old[key]) for key in ("total_mass_g", "gas_mass_g", "gas_water_mass_g")}
        if any(value > 1e-10 for value in errors.values()):
            raise ValueError("R3 report changed the frozen exact main mass: "+str(errors))
        # Histogram quantisation is an explicitly bounded approximation for
        # fitting. The §1 report uses the exact unchanged R2 main prediction.
        prediction["summary_main_mass_g"] = prediction["main_mass_g"]
        prediction["main_mass_g"] = exact["total_mass_g"]
        prediction["total_mass_g"] = exact["total_mass_g"]
        prediction["gas_share"] = (0. if exact["total_mass_g"] == 0 else
                                   exact["gas_mass_g"] / exact["total_mass_g"])
        prediction["interval_lower_g"] = exact["total_mass_g"]+prediction["E_lo_g"]
        prediction["interval_upper_g"] = (None if prediction["U_rest_g"] is None else
            exact["total_mass_g"]+prediction["E_hi_g"]+prediction["U_rest_g"])
        prediction["summary_delta"] = prediction["delta"]
        dose = prediction["injected_mass_g"]
        residual = (None if prediction["W_g"] is None or dose is None or dose <= 0 else
                    r2.interval_residual(prediction["interval_lower_g"], dose, prediction["W_g"]))
        prediction["signed_interval_residual"] = residual
        prediction["delta"] = None if residual is None else abs(residual)
        partition = r2._physical_partition(response)
        parts = engine._parts(frame, partition["missing"])
        from .build_ffac_p1_final2_candidate import transported_floor
        floor_row = dict(rig_domain_revision=revision,
            H_ac28_to_frame_native=raw_geometry["H_ac28_to_frame_native"])
        floor = transported_floor(SESSION["floor_curve"], floor_row)
        for name, mask in parts.items():
            prediction["excluded_parts"][name].pop("own_colour_histogram", None)
            for facies in prediction["excluded_parts"][name]["by_facies"].values():
                facies.pop("own_colour_histogram", None)
            prediction["excluded_parts"][name]["distances"] = geometry_inputs.distances_for_part(mask,
                revision=revision, geometry=raw_geometry, floor_curve_canvas_y=floor, sand_domain=frame.domain)
            original_name = "unquantified_main" if name == "unquantified_facies" else name
            prediction["excluded_parts"][name]["round2_capacity_bound"] = partition["parts"].get(original_name)
        figure = None
        if run in ("ac19", "ac47", "ac56") and index == nearest24:
            figure = render_parts(frame, parts, candidate, Path(job) / "saved_parts/figures" / (run+"_nearest_24h.png"))
        proof = dict(contract=CONTRACT, complete=True, run=run, frame_index=index,
            campaign_sha256=campaign_sha, source_binding=metadata, source_binding_sha256=saved.digest(metadata),
            original_r2_frame=source.asset(old_path), prediction=prediction,
            main_mass_exact_replay_errors_g=errors, native_statistics=row,
            coordinate_validation=SESSION["coordinate_validation"], figure=figure,
            created_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-tick,
            full_resolution=True, main_mass_changed=False, phase_labels_changed=False,
            training=False, fitting=False, full39_mass_production_started=False)
        commit_json(folder, "receipt.json", proof)
        records.append(dict(run=run, frame_index=index, cached=False))
        print(json.dumps(dict(run=run, frame_index=index, committed=True, seconds=proof["wall_seconds"])), flush=True)
        del frame, response, geometry, own, row, prediction, exact, photo, partition, parts
        gc.collect()
    return dict(run=run, frames=len(records), cached=sum(v["cached"] for v in records),
                seconds=time.perf_counter()-started)


def new_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for v in rows for k in v)))
        writer.writeheader(); writer.writerows(rows)


def summarize(job, campaign):
    receipts = [saved.read_json(path) for path in sorted((Path(job) / "saved_parts/frames").glob("*/frame_*/receipt.json"))]
    plan = source.read_bound_json(campaign["plan"])
    input_plan = source.read_bound_json(plan["r2_input_plan"])
    index = source.read_bound_json(input_plan["source_metadata"])
    expected = {(row["run"], row["frame_index"]) for row in index["frames"]
                if row["run"] in set(r2.HELDOUT_RUNS) | {"ac38"}}
    observed = {(row["run"], row["frame_index"]) for row in receipts}
    if observed != expected or len(observed) != len(receipts):
        raise ValueError(f"§1 report requires all saved frames: {len(observed)}/{len(expected)}")
    by_frame, by_part, by_facies, distances = [], [], [], []
    for receipt in receipts:
        if receipt["campaign_sha256"] != source.sha256(Path(job) / "saved_parts/campaign.json"):
            raise ValueError("Report seal has another code/input binding")
        value = receipt["prediction"]
        identity = {k: value[k] for k in ("run", "frame_index", "elapsed_h")}
        by_frame.append(dict(identity, **{k: value[k] for k in ("main_mass_g", "gas_mass_g", "gas_water_mass_g", "gas_share", "E_lo_g", "E_hi_g", "U_rest_g", "W_g", "injected_mass_g", "interval_lower_g", "interval_upper_g", "delta", "dilution_factor")}))
        for name, part in value["excluded_parts"].items():
            columns = {k: v for k, v in part.items() if np.isscalar(v) or v is None}
            by_part.append(dict(identity, part=name, **columns))
            for label, facies in part["by_facies"].items():
                by_facies.append(dict(identity, part=name, facies=int(label),
                    **{k: v for k, v in facies.items() if np.isscalar(v) or v is None}))
            for boundary, stats in part["distances"].items():
                distances.append(dict(identity, part=name, boundary=boundary, **stats))
    folder = Path(job) / "saved_parts/report"
    folder.mkdir(exist_ok=False)
    for name, rows in (("mass_interval_by_frame.csv", by_frame), ("excluded_parts_by_frame.csv", by_part),
                       ("excluded_parts_by_facies.csv", by_facies), ("distances_to_floor_top_walls.csv", distances)):
        new_csv(folder / name, rows)
    record = dict(contract=CONTRACT, complete=True, source_frame_count=len(receipts),
        runs=sorted({r["run"] for r in receipts}), main_mass_unchanged=True,
        baseline_parameters_only=True, fitting_started=False, full39_mass_production_started=False,
        selected_images_phase_gate_still_required=True,
        undefined_estimation_geometry_frames=sum(r["prediction"]["U_rest_g"] is None for r in receipts),
        maximum_exact_main_replay_error_g=max((max(r["main_mass_exact_replay_errors_g"].values()) for r in receipts), default=0),
        campaign=source.asset(Path(job) / "saved_parts/campaign.json"),
        figures=[r["figure"] for r in receipts if r.get("figure")],
        created_utc=datetime.now(timezone.utc).isoformat())
    source.write_new_json(folder / "report.json", record)
    with (folder / "README.txt").open("x", encoding="utf-8") as stream:
        stream.write("R3 section 1, unchanged round-2 saved frames.\n"
            "Main M was replayed exactly and checked against the frozen saved R2 results.\n"
            "Excluded-part own-colour estimates and measured/column-extended geometry are separate from M.\n"
            "An undefined same-column sand property remains unknown; it is not silently replaced.\n"
            "No B1 fit or 39-run mass production was started. Selected-image phase QA is still required.\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--backend", choices=("cpu", "auto", "gpu"), default="auto")
    parser.add_argument("--stage", choices=("extract", "report"), default="extract")
    args = parser.parse_args()
    folder = args.job / "saved_parts"; folder.mkdir(parents=True, exist_ok=True)
    campaign_path = folder / "campaign.json"
    proposed = dict(contract=CONTRACT, plan=source.asset(args.plan), code=code_inventory(),
                    driver=source.asset(__file__), fitting=False, full39_mass_production=False)
    if campaign_path.is_file():
        old = saved.read_json(campaign_path)
        if old != proposed:
            raise ValueError("The report campaign is immutable; new code needs a new report folder")
    else:
        source.write_new_json(campaign_path, proposed)
    if args.stage == "report":
        print(json.dumps(summarize(args.job, proposed)), flush=True); return
    tasks = [(run, str(args.job), source.sha256(campaign_path), args.frames, args.backend)
             for run in sorted(args.runs or (*r2.HELDOUT_RUNS, "ac38"))]
    started = time.perf_counter(); completed, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
            initargs=(proposed["plan"], proposed["code"])) as pool:
        futures = {pool.submit(report_run, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                value = future.result(); completed.append(value)
                print(json.dumps(dict(completed=value)), flush=True)
            except Exception as error:
                value = dict(run=futures[future][0], error=repr(error)); failures.append(value)
                print(json.dumps(dict(failure=value)), flush=True)
    record = dict(contract=CONTRACT, workers=args.workers, threads_per_worker=1,
        backend=args.backend, completed=completed, failures=failures, seconds=time.perf_counter()-started,
        created_utc=datetime.now(timezone.utc).isoformat())
    source.write_new_json(folder / ("execution_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json"), record)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

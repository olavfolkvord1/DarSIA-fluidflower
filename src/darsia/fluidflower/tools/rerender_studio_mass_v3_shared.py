"""Rebuild Studio concentration movies with one calibration per experiment series.

The verified v2 phase maps, registered RGB, corrected optical signals and
DarSIA fields are reused byte-for-byte.  FF_AC, FF_BM and FF_UM each receive
one shared optical response, dilution law and gas saturation.  Whole runs are
held out in cross-validation.  No per-run concentration fit remains.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import cv2
import numpy as np

from darsia_extensions import studio_mass as core


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render_frame(path, rgb, c, av, gv, pv, row, title, limit):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(13, 4.4), layout="constrained"); FigureCanvasAgg(fig)
    photo, field = fig.subplots(1, 2)
    photo.imshow(rgb); photo.set_title("Corrected photograph"); photo.axis("off")
    image = field.imshow(np.ma.array(c, mask=av <= 0), vmin=0, vmax=limit, cmap="viridis")
    field.set_facecolor("#e6edf2")
    gas = np.zeros((*gv.shape, 4)); gas[..., :3] = [.25, .29, .34]
    gas[..., 3] = np.divide(gv, pv, out=np.zeros_like(gv), where=pv > 0)
    field.imshow(gas); field.set_title("Estimated dissolved CO2 · g/L"); field.axis("off")
    fig.colorbar(image, ax=field, fraction=.035, label="g/L · fixed linear scale")
    fig.suptitle(
        f"{title} · {float(row['elapsed_h']):.3f} h\n"
        f"Injected: {row['injected_mass_g']:.3f} g · Estimated: {row['total_mass_g']:.3f} g"
    )
    fig.savefig(path, dpi=120)


def render_mass_curve(path, rows, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 4.5), layout="constrained"); FigureCanvasAgg(fig)
    ax = fig.subplots(); times = [row["elapsed_h"] for row in rows]
    for key, label, color in (
        ("injected_mass_g", "Injected reference", "black"),
        ("total_mass_g", "Estimated total", "#175aad"),
        ("aqueous_mass_g", "Dissolved", "#169879"),
        ("gas_mass_g", "Free gas", "#dc398b"),
    ):
        ax.plot(times, [row[key] for row in rows], label=label, color=color)
    ax.fill_between(times, [row["gas_mass_min_g"] for row in rows],
                    [row["gas_mass_max_g"] for row in rows], color="#dc398b", alpha=.14,
                    label="Free-gas partition interval")
    held = [row for row in rows if row["calibration_role"] == "held-out time"]
    ax.scatter([row["elapsed_h"] for row in held], [row["total_mass_g"] for row in held],
               marker="o", facecolors="white", edgecolors="#175aad", zorder=5,
               label="Held-out time")
    ax.set(xlabel="Time since injection start · h", ylabel="CO2 mass · g",
           title=title + " · concentration and gas-partition fit")
    ax.grid(alpha=.2); ax.legend(); fig.savefig(path, dpi=120)


def source_run_folder(result_path):
    result_path = Path(result_path).resolve()
    if result_path.name != "result.json" or result_path.parent.parent.name != "results":
        raise ValueError(f"Unexpected source result layout: {result_path}")
    return result_path.parents[2]


def summarize_result(result_path, *, signal_max=4., histogram_bins=1024):
    """Create a bounded-memory sufficient-statistics record for one v2 run."""
    result = read(result_path)
    run = result["selected_inputs"]["candidate"]["run"]
    bins = np.linspace(0., float(signal_max), int(histogram_bins) + 1)
    weights, volumes, gas, residual = [], [], [], []
    for row in result["frames"]:
        with np.load(row["fields"], allow_pickle=False) as frame:
            signal = np.asarray(frame["signal"])
            av = np.asarray(frame["aqueous_volume_m3"])
            gv = np.asarray(frame["gas_volume_m3"])
            solubility = np.asarray(frame["solubility_g_l"])
            density = np.asarray(frame["gas_density_g_l"])
            positive = signal > 0
            weights.append(np.histogram(
                np.clip(signal[positive], bins[0], bins[-1]), bins,
                weights=(av * solubility * 1000)[positive],
            )[0])
            volumes.append(float(np.sum(av + gv, dtype=np.float64)))
            gas.append(float(np.sum(gv * density, dtype=np.float64) * 1000))
            residual.append(float(np.sum(gv * solubility, dtype=np.float64) * 1000))
    times = np.asarray([row["elapsed_h"] for row in result["frames"]], float)
    injected = np.asarray([row["injected_mass_g"] for row in result["frames"]], float)
    volumes = np.asarray(volumes)
    end = np.flatnonzero(injected >= .9999 * injected.max())
    if not len(end):
        raise ValueError(f"{run}: injection reference has no end point")
    selected = np.flatnonzero((injected > 0) & (volumes > 0))
    if len(selected) < 3:
        raise ValueError(f"{run}: fewer than three positive-mass frames")
    return {
        "run": run, "times": times, "injected_g": injected,
        "positive_indices": selected, "volumes_m3": volumes,
        "anchor_index": int(end[0]),
        "histogram_centers": (bins[:-1] + bins[1:]) * .5,
        "aqueous_histogram_weights": np.asarray(weights),
        "gas_full_mass_g": np.asarray(gas),
        "residual_saturated_mass_g": np.asarray(residual),
        "signal_max": float(signal_max), "histogram_bins": int(histogram_bins),
    }


def prepare(source_delivery, destination, workers=6):
    """Fit shared series models and write an executable render manifest."""
    source_delivery, destination = Path(source_delivery).resolve(), Path(destination).resolve()
    if destination == source_delivery.parent:
        raise ValueError("The v3 destination must differ from the v2 source")
    destination.mkdir(parents=True, exist_ok=False)
    delivery = read(source_delivery)
    groups = {}
    sources = {}
    methods = {}
    for record in delivery["experiments"]:
        result_path = Path(record["result"].replace("file:///", ""))
        result = read(result_path)
        dataset = result["selected_inputs"]["candidate"]["dataset"]
        groups.setdefault(dataset, []).append(summarize_result(result_path))
        sources[(dataset, result["selected_inputs"]["candidate"]["run"])] = result_path
        methods.setdefault(dataset, core.Method(**{
            key: result["method"]["settings"][key] for key in asdict(core.Method())
        }))
    calibration_paths = {}
    for dataset, summaries in sorted(groups.items()):
        fit = core.fit_series(summaries, methods[dataset], core.MassPolicy(), cross_validate=True)
        fit["dataset"] = dataset
        fit["source_delivery"] = str(source_delivery)
        path = destination / "series_calibrations" / f"{dataset}.json"
        write(path, fit); calibration_paths[dataset] = path
    jobs = []
    for (dataset, run), source_result in sorted(sources.items()):
        jobs.append({
            "dataset": dataset, "run": run, "source_result": str(source_result),
            "series_calibration": str(calibration_paths[dataset]),
            "actual_root": str(destination), "final_root": str(destination),
        })
    manifest = destination / "render_manifest.json"
    write(manifest, {
        "format": "studio-mass-v3-shared-render-manifest-v1",
        "source_delivery": str(source_delivery), "destination": str(destination),
        "workers": int(workers), "jobs": jobs,
    })
    return manifest


def rerender(job):
    started = time.perf_counter()
    source_result_path = Path(job["source_result"]).resolve()
    source_folder = source_result_path.parent
    source = read(source_result_path)
    calibration_path = Path(job["series_calibration"]).resolve()
    shared_fit = read(calibration_path)
    dataset = source["selected_inputs"]["candidate"]["dataset"]
    run = source["selected_inputs"]["candidate"]["run"]
    if (dataset, run) != (job["dataset"], job["run"]):
        raise ValueError("Job and source result identify different experiments")
    if source.get("format") != "studio-concentration-mass-v2" or not source.get("complete"):
        raise ValueError("A complete v2 Studio result is required")
    if shared_fit.get("format") != "shared-series-mass-fit-v1" or dataset != shared_fit.get("dataset"):
        raise ValueError("The shared calibration does not match the experiment series")
    if run not in shared_fit.get("runs", []):
        raise ValueError("Run missing from its shared series calibration")
    actual_run = Path(job["actual_root"]).resolve() / run
    final_run = Path(job["final_root"]).resolve() / run
    if actual_run == source_run_folder(source_result_path) or final_run == source_run_folder(source_result_path):
        raise ValueError("The v3 destination must not equal the v2 source")
    completion = actual_run / "completion.json"
    source_binding = sha(source_result_path)
    calibration_binding = sha(calibration_path)
    if completion.exists():
        cached = read(completion)
        if (cached.get("source_result_sha256") == source_binding
                and cached.get("series_calibration_sha256") == calibration_binding
                and cached.get("complete") is True):
            return cached
        raise ValueError(f"Stale partial destination exists: {actual_run}")
    actual_run.mkdir(parents=True, exist_ok=False)
    try:
        settings = source["method"]["settings"]
        method = core.Method(**{key: settings[key] for key in asdict(core.Method())})
        policy = core.MassPolicy(**shared_fit["mass_policy"])
        source_files = [Path(__file__).resolve(), Path(core.__file__).resolve()]
        source_hashes = {str(path): sha(path) for path in source_files}
        method_provenance = core.provenance(method)
        identity = core.fingerprint({
            "format": "studio-concentration-mass-v3-shared-rerender-v1",
            "source_result": str(source_result_path),
            "source_result_sha256": source_binding,
            "series_calibration": str(calibration_path),
            "series_calibration_sha256": calibration_binding,
            "method": method_provenance,
            "mass_policy": asdict(policy),
            "rerender_source_sha256": source_hashes,
        })
        actual_result_folder = actual_run / "results" / identity[:20]
        final_result_folder = final_run / "results" / identity[:20]
        actual_result_folder.mkdir(parents=True)
        frame_count = len(source["frames"])
        times = np.asarray([row["elapsed_h"] for row in source["frames"]], float)
        injected = np.asarray([row["injected_mass_g"] for row in source["frames"]], float)
        parameters = shared_fit["parameters"]
        volumes, solubility_max = [], 0.
        for row in source["frames"]:
            with np.load(row["fields"], allow_pickle=False) as frame:
                volumes.append(float(np.sum(
                    frame["aqueous_volume_m3"] + frame["gas_volume_m3"], dtype=np.float64)))
                solubility_max = max(solubility_max, float(np.max(frame["solubility_g_l"])))
        end = np.flatnonzero(injected >= .9999 * injected.max())
        if not len(end):
            raise ValueError("Injection reference has no end point")
        anchor = int(end[0])
        dilution = core.dilution_factors(volumes, times, anchor, parameters["beta"])
        limit = float(np.ceil(solubility_max * 10) / 10)
        title = dataset.upper().replace("FF", "FF_", 1) + " · " + run.upper()
        rows = []
        for index, old_row in enumerate(source["frames"]):
            with np.load(old_row["fields"], allow_pickle=False) as frame:
                signal = np.asarray(frame["signal"], np.float32)
                raw_signal = np.asarray(frame.get("signal_before_facies_correction", signal), np.float32)
                av = np.asarray(frame["aqueous_volume_m3"], np.float32)
                gv = np.asarray(frame["gas_volume_m3"], np.float32)
                pv = np.asarray(frame["pore_volume_m3"], np.float32)
                solubility = np.asarray(frame["solubility_g_l"], np.float32)
                density = np.asarray(frame["gas_density_g_l"], np.float32)
                rgb = np.asarray(frame["photograph_rgb"], np.float32)
            concentration = core.concentration(signal, solubility, parameters, dilution[index], method)
            totals = core.totals(
                concentration, av, gv, solubility, density,
                parameters["gas_saturation"], policy.residual_liquid_concentration_fraction)
            interval = core.total_mass_interval(
                concentration, av, gv, solubility, density, policy)
            actual_preview = actual_result_folder / f"frame_{index:03d}.png"
            actual_fields = actual_result_folder / f"frame_{index:03d}.npz"
            final_preview = final_result_folder / actual_preview.name
            final_fields = final_result_folder / actual_fields.name
            record = {**old_row, **totals, **interval,
                      "dilution_factor": float(dilution[index]),
                      "calibration_role": "shared series fit",
                      "preview": str(final_preview), "fields": str(final_fields)}
            np.savez_compressed(
                actual_fields,
                concentration_g_l=np.where(av > 0, concentration, 0),
                signal=signal, signal_before_facies_correction=raw_signal,
                aqueous_volume_m3=av, gas_volume_m3=gv,
                pore_volume_m3=pv, solubility_g_l=solubility,
                gas_density_g_l=density, photograph_rgb=rgb,
            )
            render_frame(actual_preview, rgb, concentration, av, gv, pv, record, title, limit)
            rows.append(record)
        actual_curve = actual_result_folder / "mass_curve.png"
        final_curve = final_result_folder / actual_curve.name
        render_mass_curve(actual_curve, rows, title)
        snapshot = actual_result_folder / "result_source"; snapshot.mkdir()
        final_snapshot = final_result_folder / "result_source"
        manifest = {}
        for index, path in enumerate(source_files):
            target = snapshot / f"{index:02d}_{path.name}"
            shutil.copyfile(path, target)
            if sha(target) != source_hashes[str(path)]:
                raise ValueError("Result source changed while snapshotting")
            manifest[str(path)] = source_hashes[str(path)]
        write(snapshot / "manifest.json", manifest)
        prepared = read(source_folder / "inputs.json")
        prepared["rerender"] = {
            "format": "studio-v3-shared-intermediate-reuse-v1",
            "source_result": str(source_result_path),
            "source_result_sha256": source_binding,
            "series_calibration": str(calibration_path),
            "series_calibration_sha256": calibration_binding,
            "reused": "registered RGB, fixed phase volumes, DarSIA fields and corrected optical signal",
            "recomputed": "shared-series concentration response, gas partition, fields and renderings",
        }
        write(actual_result_folder / "inputs.json", prepared)
        result = {
            "format": "studio-concentration-mass-v3-shared", "identity": identity,
            "method": method_provenance, "host_sha256": source_hashes[str(Path(__file__).resolve())],
            "calibration": shared_fit, "frames": rows, "mass_policy": asdict(policy),
            "optical_noise_correction": source["optical_noise_correction"],
            "result_source_manifest": str(final_snapshot / "manifest.json"),
            "physical_arrays": source.get("physical_arrays", {}),
            "selected_inputs": source["selected_inputs"], "color_limit_g_l": limit,
            "weak_optical_reference_fraction": source.get("weak_optical_reference_fraction"),
            "source_result": str(source_result_path), "source_result_sha256": source_binding,
            "series_calibration": str(calibration_path),
            "series_calibration_sha256": calibration_binding,
            "mass_curve": str(final_curve), "complete": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        actual_result = actual_result_folder / "result.json"
        write(actual_result, result)
        source_request = read(source_run_folder(source_result_path) / "request.json")
        source_request["mass_policy"] = asdict(policy)
        source_request["results_root"] = str(final_run / "results")
        write(actual_run / "request.json", source_request)
        write(actual_run / "status.json", {"stage": "complete", "result": str(final_result_folder / "result.json")})
        completion_record = {
            "complete": True, "dataset": dataset, "run": run, "frames": frame_count,
            "source_result": str(source_result_path), "source_result_sha256": source_binding,
            "series_calibration": str(calibration_path),
            "series_calibration_sha256": calibration_binding,
            "result": str(final_result_folder / "result.json"),
            "seconds": time.perf_counter() - started,
        }
        write(completion, completion_record)
        return completion_record
    except Exception:
        write(actual_run / "failure.json", {"error": traceback.format_exc()})
        raise


def run_campaign(manifest_path):
    manifest_path = Path(manifest_path); manifest = read(manifest_path)
    jobs = manifest["jobs"]; workers = int(manifest.get("workers", 1))
    status_path = manifest_path.with_name("status.json")
    complete, failed = [], []
    write(status_path, {"stage": "running", "total": len(jobs), "complete": complete, "failed": failed})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rerender, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result(); complete.append(row)
            except Exception:
                failed.append({"run": job["run"], "error": traceback.format_exc()})
            write(status_path, {"stage": "running" if len(complete)+len(failed) < len(jobs) else "complete",
                                "total": len(jobs), "complete": complete, "failed": failed})
    if failed:
        raise RuntimeError(f"{len(failed)} rerender jobs failed")
    return complete


def validate_result(result_path):
    result = read(result_path); policy = core.MassPolicy(**result["mass_policy"])
    maximum_error = 0.
    for row in result["frames"]:
        with np.load(row["fields"], allow_pickle=False) as fields:
            if not all(np.isfinite(fields[name]).all() for name in fields.files):
                raise ValueError("Nonfinite saved field")
            c, av, gv, pv = [fields[name] for name in (
                "concentration_g_l", "aqueous_volume_m3", "gas_volume_m3", "pore_volume_m3")]
            if any((value < 0).any() for value in (c, av, gv, pv)) or np.any(av+gv > pv*1.0001+1e-12):
                raise ValueError("Invalid phase pore volumes")
            if np.any(c[av <= 0] != 0) or np.any(c > fields["solubility_g_l"]*1.00001):
                raise ValueError("Concentration violates phase support or solubility")
            totals = core.totals(c, av, gv, fields["solubility_g_l"], fields["gas_density_g_l"],
                                 result["calibration"]["parameters"]["gas_saturation"],
                                 policy.residual_liquid_concentration_fraction)
            interval = core.total_mass_interval(c, av, gv, fields["solubility_g_l"],
                                                fields["gas_density_g_l"], policy)
            for key, value in {**totals, **interval}.items():
                maximum_error = max(maximum_error, abs(value-row[key]))
                if not np.isclose(value, row[key], rtol=2e-6, atol=1e-7):
                    raise ValueError("Saved fields do not reproduce mass")
        image = cv2.imread(row["preview"])
        if image is None or image.shape[:2] != (528, 1560):
            raise ValueError("Concentration preview missing or incomplete")
    times = np.asarray([row["elapsed_h"] for row in result["frames"]])
    injected = np.asarray([row["injected_mass_g"] for row in result["frames"]])
    total = np.asarray([row["total_mass_g"] for row in result["frames"]])
    run = result["selected_inputs"]["candidate"]["run"]
    held_run = next((row for row in result["calibration"].get("leave_one_run_out", [])
                     if row["run"] == run), None)
    return {
        "frames": len(times), "time_range_h": [float(times[0]), float(times[-1])],
        "calibration_scope": "one shared parameter set per experiment series",
        "whole_run_held_out": held_run is not None,
        "whole_run_held_out_rmse_g": None if held_run is None else held_run["rmse_g"],
        "whole_run_held_out_rmse_percent_of_dose": (
            None if held_run is None else held_run["rmse_percent_of_dose"]),
        "in_sample_rmse_g": float(np.sqrt(np.mean((total-injected)**2))),
        "in_sample_rmse_percent_of_dose": float(
            100*np.sqrt(np.mean((total-injected)**2))/injected.max()),
        "maximum_mass_reintegration_error_g": maximum_error,
        "parameters_at_bounds": result["calibration"]["parameters_at_bounds"],
        "gas_saturation_bounds": [policy.gas_saturation_min, policy.gas_saturation_max],
        "fields_finite": True, "phase_support_checked": True, "chronology_checked": True,
    }


def encode(result_path, target):
    # Video support is optional for numerical results and CSV/PNG exports.
    import imageio_ffmpeg

    result_path, target = Path(result_path), Path(target)
    count = len(read(result_path)["frames"])
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", "2/3", "-i", str(result_path.parent / "frame_%03d.png"),
        "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-threads", "2", "-movflags", "+faststart", str(target),
    ], check=True)
    video = cv2.VideoCapture(str(target))
    try:
        frames, fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT)), video.get(cv2.CAP_PROP_FPS)
        if frames != count*36 or fps != 24:
            raise ValueError(f"Unexpected movie length: {frames}/{fps}")
        for index in (0, frames-1):
            video.set(cv2.CAP_PROP_POS_FRAMES, index); ok, image = video.read()
            if not ok or image.shape[:2] != (528, 1560):
                raise ValueError("Movie did not decode")
    finally:
        video.release()
    return {"path": str(target), "sha256": sha(target), "frames": frames, "fps": fps,
            "duration_s": count*1.5, "playback": "one selected image per 1.5 seconds"}


def publish(root):
    from darsia.fluidflower.tools.phase_studio_mass_cache import register_result, find_result
    root = Path(root).resolve(); records = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and (path/"completion.json").exists()):
        completion = read(folder/"completion.json"); result_path = Path(completion["result"])
        request = read(folder/"request.json"); result = read(result_path)
        audit = validate_result(result_path)
        movie = encode(result_path, folder/(folder.name+"_concentration_mass_v3_shared.mp4"))
        index = register_result(result_path, request)
        if find_result(request, request["method"]) != result_path:
            raise ValueError("Studio cannot retrieve the published v2 result")
        with (folder/"mass_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
            keys = ["frame_index", "source_name", "elapsed_h", "injected_mass_g", "aqueous_mass_g",
                    "gas_mass_g", "gas_mass_min_g", "gas_mass_max_g", "total_mass_g",
                    "total_mass_min_g", "total_mass_max_g", "dilution_factor", "calibration_role"]
            writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
            writer.writerows({key: row.get(key) for key in keys} for row in result["frames"])
        candidate = request["candidate"]
        record = {
            "title": candidate["dataset"].upper().replace("FF", "FF_", 1)+" · "+candidate["run"].upper(),
            "model": candidate["candidate_id"], "method": result["method"],
            "mass_policy": result["mass_policy"], "optical_noise_correction": result["optical_noise_correction"],
            "series_calibration": result["calibration"],
            "audit": audit, "result": result_path.as_uri(), "curve": Path(result["mass_curve"]).as_uri(),
            "video": Path(movie["path"]).as_uri(), "movie": movie, "studio_index": str(index),
            "frames": [{"image": Path(row["preview"]).as_uri(), "time": row["elapsed_h"],
                        "injected": row["injected_mass_g"], "aqueous": row["aqueous_mass_g"],
                        "gas": row["gas_mass_g"], "gas_min": row["gas_mass_min_g"],
                        "gas_max": row["gas_mass_max_g"], "total": row["total_mass_g"],
                        "total_min": row["total_mass_min_g"], "total_max": row["total_mass_max_g"],
                        "role": row["calibration_role"]} for row in result["frames"]],
        }
        write(folder/"delivery_record.json", record); records.append(record)
    if not records:
        raise ValueError("No completed v3 runs")
    data = json.dumps(records).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FluidFlower · shared-series mass v3</title><style>body{font:16px system-ui;background:#f2f5f8;color:#172b40;margin:24px auto;max-width:1600px;padding:0 20px}button,select{font:inherit;padding:8px;margin:0 8px 10px 0}input{width:100%}img,video{max-width:100%;display:block}details{margin:18px 0}pre{white-space:pre-wrap}a{color:#165e93}</style>
<h1>FluidFlower · shared-series concentration and mass validation v3</h1><p id="scope"></p><p>Every experiment in FF_AC, FF_BM or FF_UM uses the same optical response, dilution law and gas saturation as the other experiments in that series. Whole-run leave-one-out checks replace per-run retuning. Fixed v2 phase maps, registered optics and FF_AC facies/grain correction are reused; older results are unchanged.</p>
<select id="run"></select><button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button><a id="download">MP4 video</a><input id="frame" type="range" min="0" step="1"><p id="status"></p><img id="image"><details open><summary>Mass versus injected reference</summary><img id="curve"></details><details><summary>Method and validation</summary><pre id="method"></pre><a id="result">Numerical result and provenance</a> · <a href="delivery.json">Delivery checks</a></details>
<script>const runs=DATA,el=id=>document.getElementById(id),sel=el('run'),slider=el('frame');let timer=null;el('scope').textContent=`${runs.length} experiments · ${runs.reduce((n,r)=>n+r.frames.length,0)} images`;runs.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.title;sel.append(o)});function stop(){if(timer)clearInterval(timer);timer=null;el('play').textContent='Play'}function show(){let r=runs[+sel.value],f=r.frames[+slider.value];el('image').src=f.image;el('status').textContent=`${r.title} · image ${+slider.value+1}/${r.frames.length} · ${f.time.toFixed(3)} h | Injected ${f.injected.toFixed(3)} g · Dissolved ${f.aqueous.toFixed(3)} g · Free gas ${f.gas.toFixed(3)} g [${f.gas_min.toFixed(3)}, ${f.gas_max.toFixed(3)}] · Total ${f.total.toFixed(3)} g`;}function choose(){stop();let r=runs[+sel.value];slider.max=r.frames.length-1;slider.value=0;el('curve').src=r.curve;el('download').href=r.video;el('result').href=r.result;el('method').textContent=JSON.stringify({model:r.model,method:r.method,mass_policy:r.mass_policy,shared_series_calibration:r.series_calibration,optical_correction:r.optical_noise_correction,audit:r.audit},null,2);show()}sel.onchange=choose;slider.oninput=()=>{stop();show()};el('prev').onclick=()=>{stop();slider.stepDown();show()};el('next').onclick=()=>{stop();slider.stepUp();show()};el('play').onclick=()=>{if(timer){stop();return}if(+slider.value===+slider.max)slider.value=0;el('play').textContent='Pause';timer=setInterval(()=>{if(+slider.value>=+slider.max){stop();return}slider.stepUp();show()},1500)};choose()</script></html>'''
    (root/"report.html").write_text(page.replace("DATA", data), encoding="utf-8")
    write(root/"delivery.json", {"format": "studio-mass-v3-shared-delivery-v1", "complete": True,
          "experiments": records, "frames": sum(row["audit"]["frames"] for row in records),
          "source_v2_unchanged": True, "production_approved": False})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if not args.destination:
            parser.error("--prepare requires --destination")
        print(prepare(args.prepare, args.destination, args.workers))
    elif args.manifest:
        run_campaign(args.manifest)
    elif args.publish:
        publish(args.publish)
    else:
        parser.error("Choose --prepare, --manifest or --publish")

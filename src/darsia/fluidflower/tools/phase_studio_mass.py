"""Host adapter: analyze the exact registered photographs/maps selected in Studio.

No optical phase inference is repeated. Numerical result generation lives in
darsia_extensions.studio_mass. All outputs are new, method/phase-bound artifacts.
"""
from __future__ import annotations

from darsia.fluidflower.series_paths import local_cache_root
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import traceback

import cv2
import numpy as np

from darsia_extensions import studio_mass as core
from darsia.fluidflower.tools.phase_mass_overnight_data import read, write, sha, crop_rgb, decode_labels, small, frame_state


def candidate_from_dict(values):
    from darsia.fluidflower.tools.calibration_pairwise_review import Candidate
    paths = {"mass_plot", "contour_movie", "per_frame_csv", "summary_path", "review_output_dir",
             "phase_map_output_dir", "raw_phase_map_output_dir", "presence_diagnostics_dir"}
    return Candidate(**{k: (Path(v) if v is not None and k in paths else v) for k, v in values.items()})


def inventory(request):
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    candidate = candidate_from_dict(request["candidate"])
    root, run = Path(request["review_root"]), candidate.run
    native_path = review._native_manifest_path(root, run)
    native = read(native_path)
    if native["run"].casefold() != run.casefold():
        raise ValueError("Review and mass experiment differ")
    rows = []
    for i, row in enumerate(native["frame_sources"]):
        photo = review._native_frame_path(root, run, i)
        phase = review._native_phase_map_path(root, candidate, "A", i)
        if not phase.is_file():
            raise ValueError("Mass calculation requires native categorical maps, not display contours")
        if request.get("local_baseline_search"):
            from darsia.fluidflower.tools.phase_local_support_preview import resolve
            phase = resolve(root, run, candidate, i)
        rows.append({**row, "frame_index": i, "photo": str(photo), "photo_sha256": sha(photo),
                     "phase": str(phase), "phase_sha256": sha(phase)})
    baseline = review._native_reference_path(root, run)
    selected_index = int(request.get("frame_index", 0))
    if request.get("reference_mode"):
        from darsia.fluidflower.tools import phase_injection_references as refs
        from darsia.fluidflower.tools.phase_reference_maps import resolve_reference_overlay
        reference = request.get("reference")
        row = dict(reference if reference is not None else native["reference"])
        if reference is None and row.get("elapsed_h") is None:
            from darsia.fluidflower.tools.phase_reference_maps import build_plan
            reference_plan = build_plan(None, root, run, candidate)["reference"]
            row.update(elapsed_h=reference_plan["phase_time_h"],time_basis=reference_plan["time_basis"])
        photo = refs.resolve_reference_asset(reference, root, run) if reference is not None else baseline
        matches = [i for i, item in enumerate(rows) if item.get("source_name") == row.get("source_name")
                   and abs(float(item["elapsed_h"])-float(row["elapsed_h"])) < 1e-7]
        if matches:
            selected_index = matches[0]
        else:
            phase = resolve_reference_overlay(reference, root, run, candidate, "A", "all")
            if request.get("local_baseline_search"):
                from darsia.fluidflower.tools.phase_local_support_preview import resolve
                phase = resolve(root, run, candidate, 0, reference_mode=True, reference=reference)
            rows.append({**row, "frame_index": None, "selection_status": "exact" if reference is not None else "pre_injection_negative_control",
                         "photo": str(photo), "photo_sha256": sha(photo), "phase": str(phase), "phase_sha256": sha(phase)})
            rows.sort(key=lambda item: float(item["elapsed_h"]))
            selected_index = next(i for i, item in enumerate(rows) if item["photo"] == str(photo))
    return native, {"candidate": candidate.provenance(), "native_manifest": str(native_path),
        "native_manifest_sha256": sha(native_path), "baseline": str(baseline),
        "baseline_sha256": sha(baseline), "frames": rows, "selected_frame_index": selected_index,
        "local_baseline_search": bool(request.get("local_baseline_search"))}


def prepare_ac_um(request, native, inputs, folder, update):
    if any(row.get("rig_domain_revision") for row in native.get("frame_sources", ())):
        raise ValueError("This historical fixed-grid calibration cannot quantify the new rig-domain revision. Use source-bound native/joint inputs with verified physical coverage; mass remains provisional and incomplete outside that coverage.")
    from darsia.fluidflower.tools.prepare_phase_injection_references import matching_source, _parity
    from darsia.fluidflower.tools.concentration_movie_input_cache import MovieInputCache
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
    from darsia.fluidflower.tools.materialize_phase_mass_inputs import _registered_grid
    from darsia.fluidflower.mass_materialization import _mass_analysis
    dataset, run = request["candidate"]["dataset"], request["candidate"]["run"]
    width = int(request.get("analysis_width", 1024))
    box = native["content_box_xywh"]
    baseline = small(crop_rgb(inputs["baseline"], box), width)
    update("Loading the registered rig and independent clear-water reference")
    with matching_source(dataset, run, native, folder) as (source, binding):
        parity = _parity(source, native, Path(request["review_root"]), run,
                         MovieInputCache(folder/"raw_parity", run))
        paths = (transfer.select_clear_reference_paths(transfer.FFAC_RAW_ROOT, run) if dataset == "ffac"
                 else transfer._ffum_clearwater_paths(run))
        clear, clear_contract = transfer._read_corrected_clearwater(source.rig, paths, source.baseline.shape)
        clipping_fraction = float(np.mean((clear < 0) | (clear > 1)))
        # Match the Studio uint8 display/correction contract, with explicit
        # accounting of out-of-range corrected wash values before encoding.
        clear = np.clip(clear, 0, 1)
        grid = _registered_grid(transfer, source)
        facies_native = np.asarray(source.rig.facies.img)
        domain = ~np.isin(facies_native, [0] if dataset == "ffac" else [1])
        depth = np.asarray(getattr(source.rig.geometry.depth, "img", source.rig.geometry.depth))
        porosity = np.asarray(getattr(source.rig.porosity, "img", source.rig.porosity))
        depth, porosity = np.broadcast_to(depth, domain.shape), np.broadcast_to(porosity, domain.shape)
        refs = dict(baseline=baseline, clearwater=small(clear, width),
                    facies=small(facies_native.astype(np.float32), width, cv2.INTER_NEAREST).astype(np.int16))
        rows, images, aqueous, gas, solubilities, densities = [], [], [], [], [], []
        for i, item in enumerate(inputs["frames"]):
            update(f"Reading fixed phase map and optical image {i+1}/{len(inputs['frames'])}")
            labels = decode_labels(item["phase"], box)
            pv, av, gv = core.native_phase_volumes(depth, porosity, labels, domain,
                grid["height_m"], grid["width_m"], baseline.shape[:2])
            current = small(crop_rgb(item["photo"], box), width)
            # A baseline may be separate from the timeline. No synthetic phase
            # or current image is inserted into the calibration set.
            state = frame_state(source, item)
            physics = _mass_analysis(baseline.shape[:2], grid["width_m"], grid["height_m"],
                state["pressure_bar"], state["temperature_c"], state["pressure_gradient_bar_per_m"],
                state["temperature_gradient_c_per_m"])
            images.append(current); aqueous.append(av); gas.append(gv)
            solubilities.append(np.asarray(physics.solubility_co2, np.float32))
            densities.append(np.asarray(physics.density_gaseous_co2, np.float32))
            rows.append({**item, **state})
        refs["pore_volume"] = pv
        arrays = dict(rgb=np.asarray(images), av=np.asarray(aqueous), gv=np.asarray(gas),
                      solubility=np.asarray(solubilities), density=np.asarray(densities), **refs)
        metadata = dict(grid=grid, geometry_binding=binding, source_parity=parity,
                        clearwater_clipped_fraction=clipping_fraction,
                        clearwater_contract=clear_contract, correction_contract=source.correction_contract,
                        reference_inputs=[{"path": str(p), "sha256": sha(p)} for p in paths],
                        frames=rows, color_direction=[.5, .5, -1.])
    return arrays, metadata


def prepare_bm(request, native, inputs, folder, update):
    # Every run needs its own registered wash binding. Retain the verified C1
    # legacy binding for existing Studio requests; never borrow it for a run.
    import csv
    from datetime import timedelta
    import pandas as pd
    from darsia.fluidflower.tools.ffbm_movie_concentration import gas_density
    from darsia.fluidflower.tools.prepare_bm_phase_target import PROTOCOL, protocol_stages
    movie = local_cache_root() / 'phase_diagnostics/concentration_movies_ffbm_c1_20260914'
    run = request["candidate"]["run"]
    binding_path = request.get('bm_reference')
    baseline_native = crop_rgb(inputs['baseline'], native['content_box_xywh'])
    if binding_path:
        binding_path = Path(binding_path); physical = read(binding_path)
        if (physical.get('format') != 'bm-studio-mass-reference-v1' or physical['run'] != run
                or physical['native_manifest_sha256'] != inputs['native_manifest_sha256']
                or physical['baseline_source_sha256'] != native['frame_sources'][0]['source_sha256']):
            raise ValueError('BM mass reference identifies another experiment or baseline')
        if any(sha(path) != digest for path,digest in physical['source_files'].items()):
            raise ValueError('BM mass reference source changed')
        geometry = physical['geometry']; difference = np.zeros(1)
        physical_files = [binding_path, *map(Path,physical['source_files'])]
    elif run == 'c1':
        physical = read(movie/'physical_references.json')
        geometry = read(movie/'preflight.json')['physical_geometry']
        with np.load(movie/'physical_references.npz') as reference:
        # Studio's sources are TIFFs. The dense movie's JPG baseline is a
        # different optical observation and must not calibrate these images.
            difference = abs(small(baseline_native, reference['baseline'].shape[1])-reference['baseline'])
        binding_path = movie/'physical_references.json'
        physical_files = [binding_path,movie/'physical_references.npz', Path(physical['clearwater_path']),
            Path(geometry['depth_source']), Path(geometry['labels_source']), Path(physical['pressure_source'])]
    else:
        raise ValueError(f'{run}: a registered clear-water/physical reference binding must be prepared before mass analysis')
    if np.quantile(difference, .99) > .025:
        raise ValueError("BM optical reference differs from the selected Studio baseline")
    width = int(request.get('analysis_width', 1024))
    baseline = small(baseline_native, width)
    depth = np.load(geometry['depth_source']); facies = np.load(geometry['labels_source'])
    domain = facies != 0
    clear = small(crop_rgb(physical['clearwater_path'], native['content_box_xywh']), width)
    with PROTOCOL.open(encoding='utf-8-sig', newline='') as stream:
        origin, stages = protocol_stages(list(csv.DictReader(stream)), run)
    first = native['frame_sources'][0]
    image_origin = datetime.strptime(first['source_name'][:17], '%y%m%d_time%H%M%S')-timedelta(hours=first['elapsed_h'])
    if abs((origin-image_origin).total_seconds()) > 1:
        raise ValueError('BM pressure/injection references and Studio timeline have different origins')
    table = pd.read_excel(physical['pressure_source'])
    dates = pd.to_datetime(table['Dato'].astype(str)+' '+table['Tid'].astype(str), format='%Y-%m-%d %H:%M')
    pressure_times = (dates-origin).dt.total_seconds().to_numpy()/3600
    bars = (table['Lufttrykk'].to_numpy()+3.125)*.001
    if np.any(np.diff(pressure_times)<=0) or not np.isfinite(bars).all():
        raise ValueError('BM pressure observations are not finite and chronological')
    times = [row['elapsed_h'] for row in inputs['frames']]
    if min(times)<pressure_times[0] or max(times)>pressure_times[-1]:
        raise ValueError('Pressure observations do not span the selected BM timeline')
    images, avs, gvs, densities, rows = [], [], [], [], []
    for i, item in enumerate(inputs["frames"]):
        update(f"Reading fixed BM phase map {i+1}/{len(inputs['frames'])}")
        labels = decode_labels(item["phase"], native["content_box_xywh"])
        photo = small(crop_rgb(item['photo'], native['content_box_xywh']), width)
        pv, av, gv = core.native_phase_volumes(depth, geometry['porosity'], labels, domain,
            geometry['height_m'], geometry['width_m'], baseline.shape[:2])
        pressure = float(np.interp(item['elapsed_h'], pressure_times, bars))
        injected = 0.
        for stage in stages:
            segments = stage.get("rate_segments")
            if not segments: raise ValueError("BM reference requires explicit injection-rate segments")
            for segment in segments:
                injected += max(0., min(float(item["elapsed_h"]), segment["end_h"])-segment["start_h"])*3600*segment["rate_kg_s"]*1000
        images.append(photo); avs.append(av); gvs.append(gv)
        densities.append(gas_density(pressure, baseline.shape[:2]))
        rows.append({**item, 'pressure_bar': pressure, 'temperature_c': 23., 'injected_mass_g': injected})
    return dict(rgb=np.asarray(images), av=np.asarray(avs), gv=np.asarray(gvs), density=np.asarray(densities),
        solubility=np.full_like(densities, physical['solubility_g_l']), baseline=baseline, clearwater=clear,
        facies=small(facies.astype(np.float32), width, cv2.INTER_NEAREST).astype(np.int16), pore_volume=pv), dict(frames=rows,
        grid={"width_m": 2.8, "height_m": 1.5}, color_direction=[1., -1., 0.],
        physical_reference=physical, physical_reference_sha256=sha(binding_path),
        correction_contract=native['correction_contract'],
        source_parity={'passed':True,'binding':'exact native baseline source' if request.get('bm_reference') else 'verified C1 paired baseline',
            'baseline_difference_q99':float(np.quantile(difference,.99))},
        source_files={str(p): sha(p) for p in [*physical_files, PROTOCOL]},
        baseline_difference_q99=float(np.quantile(difference, .99)))


def render_frame(path, rgb, c, av, gv, pv, row, title, limit):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(13, 4.4), layout="constrained"); FigureCanvasAgg(fig)
    ax, field = fig.subplots(1, 2)
    ax.imshow(rgb); ax.set_title("Corrected photograph"); ax.axis("off")
    masked = np.ma.array(c, mask=av <= 0)
    im = field.imshow(masked, vmin=0, vmax=limit, cmap="viridis")
    field.set_facecolor("#e6edf2")
    # Dominant gas only: isolated native gas grains must not erase an entire
    # pooled aqueous cell. Phase pore fractions remain in the saved arrays.
    gas = np.zeros((*gv.shape, 4)); gas[..., :3] = [.25, .29, .34]
    gas[..., 3] = np.divide(gv, pv, out=np.zeros_like(gv), where=pv > 0)
    field.imshow(gas); field.set_title("Estimated dissolved CO₂ · g/L"); field.axis("off")
    fig.colorbar(im, ax=field, fraction=.035, label="g/L · fixed linear scale")
    fig.suptitle(f"{title} · {float(row['elapsed_h']):.3f} h\nInjected: {row['injected_mass_g']:.3f} g · Estimated: {row['total_mass_g']:.3f} g")
    fig.savefig(path, dpi=120)


def run(request_path):
    request_path = Path(request_path); request = read(request_path)
    status_path = request_path.with_name("status.json")
    def update(stage, **values):
        write(status_path, {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **values})
    try:
        cv2.setNumThreads(2)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            method = core.Method(**request["method"])
            mass_policy = core.MassPolicy(**request.get("mass_policy", {}))
            method_provenance = core.provenance(method)
            host_sha = sha(__file__)
            import darsia
            source_files = [Path(__file__), Path(core.__file__), Path(inspect.getfile(darsia.TVD)),
                Path(inspect.getfile(darsia.SimpleFlash)), Path(inspect.getfile(darsia.ExtrudedPorousGeometry))]
            source_files += [Path(__file__).with_name(name+'.py') for name in
                ('prepare_phase_injection_references', 'schema2_phase_transfer_campaign',
                 'phase_mass_overnight_data', 'compute_ffbm_c1_concentration_movie', 'ffbm_movie_concentration',
                 'prepare_bm_mass_reference')]
            source_bindings = {str(p): sha(p) for p in source_files}
            update("Checking selected photograph and phase-map identities")
            native, inputs = inventory(request)
            scientific_inputs = {k: v for k, v in inputs.items() if k != "selected_frame_index"}
            request_identity = core.fingerprint({"inputs": scientific_inputs, "method": method_provenance,
                                         "mass_policy": asdict(mass_policy),
                                         "host_sha256": host_sha,
                                         "analysis_width": request.get("analysis_width", 1024)})
            # Re-read physical references before reusing a numerical result.
            # Model/image identity alone cannot validate changed rig geometry,
            # wash images, pressure records or injection protocols.
            preparation_folder = request_path.parent/'preparation'
            preparation_folder.mkdir(parents=True, exist_ok=True)
            prepare = prepare_bm if request["candidate"]["dataset"] == "ffbm" else prepare_ac_um
            arrays, prepared = prepare(request, native, inputs, preparation_folder, update)
            array_bindings = {name: dict(shape=list(values.shape), dtype=str(values.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(values).data).hexdigest())
                for name, values in arrays.items()}
            identity = core.fingerprint(dict(request=request_identity, arrays=array_bindings,
                frame_physics=[{key: row.get(key) for key in
                    ('elapsed_h', 'injected_mass_g', 'pressure_bar', 'temperature_c')}
                    for row in prepared['frames']], source_bindings=source_bindings))
            folder = Path(request.get("results_root", request_path.parent))/identity[:20]
            result_path = folder/"result.json"
            if result_path.exists():
                update("complete", result=str(result_path)); return read(result_path)
            folder.mkdir(parents=True, exist_ok=True)
            write(folder/"inputs.json", {"selected_inputs": inputs, "prepared": prepared})
            for name, values in arrays.items(): np.save(folder/(name+".npy"), values)
            snapshot = folder/'result_source'; snapshot.mkdir(exist_ok=True)
            for i, path in enumerate(source_files):
                target = snapshot/f'{i:02d}_{path.name}'
                target.write_bytes(path.read_bytes())
                if sha(target) != source_bindings[str(path)]: raise ValueError('Result source changed while snapshotting')
            write(snapshot/'manifest.json', source_bindings)
            optical = core.ReferenceSignal(arrays["baseline"], arrays["clearwater"], arrays["facies"],
                                           arrays["pore_volume"] > 0, method, direction=prepared["color_direction"])
            signals = []
            for i, rgb in enumerate(arrays["rgb"]):
                update(f"Calculating optical signal {i+1}/{len(arrays['rgb'])}")
                signals.append(optical(rgb))
            raw_signals = np.asarray(signals)
            optical_noise_correction = {"format": "none", "reason": "series does not use FF_AC facies controls"}
            if request["candidate"]["dataset"] == "ffac":
                update("Correcting FF_AC facies offsets and grain-scale optical noise")
                signals, optical_noise_correction = core.correct_facies_optical_noise(
                    raw_signals, arrays["av"], arrays["gv"], arrays["facies"], arrays["pore_volume"])
            else:
                signals = raw_signals
            times = np.array([r["elapsed_h"] for r in prepared["frames"]])
            injected = np.array([r["injected_mass_g"] for r in prepared["frames"]])
            update("Fitting one concentration curve against the injection protocol")
            fit = core.fit(signals, arrays["av"], arrays["gv"], arrays["solubility"],
                           arrays["density"], times, injected, method, mass_policy)
            p = fit["parameters"]
            volumes = np.sum(arrays["av"]+arrays["gv"], axis=(1, 2), dtype=np.float64)
            dilution = core.dilution_factors(volumes, times, fit["anchor_index"], p["beta"])
            rows = []; limit = float(np.ceil(arrays["solubility"].max()*10)/10)
            title = request["candidate"]["dataset"].upper()+" · "+request["candidate"]["run"].upper()
            for i, row in enumerate(prepared["frames"]):
                update(f"Saving concentration and mass {i+1}/{len(times)}")
                c = core.concentration(signals[i], arrays["solubility"][i], p, dilution[i], method)
                totals = core.totals(
                    c, arrays["av"][i], arrays["gv"][i], arrays["solubility"][i],
                    arrays["density"][i], p["gas_saturation"],
                    mass_policy.residual_liquid_concentration_fraction)
                mass_interval = core.total_mass_interval(
                    c, arrays["av"][i], arrays["gv"][i], arrays["solubility"][i],
                    arrays["density"][i], mass_policy)
                preview = folder/f"frame_{i:03d}.png"; archive = folder/f"frame_{i:03d}.npz"
                record = {**row, **totals, **mass_interval, "dilution_factor": float(dilution[i]),
                    "calibration_role": "held-out time" if i in fit["held_out_indices"] else "fit",
                    "preview": str(preview), "fields": str(archive)}
                np.savez_compressed(archive, concentration_g_l=np.where(arrays["av"][i]>0, c, 0),
                    signal=signals[i], signal_before_facies_correction=raw_signals[i],
                    aqueous_volume_m3=arrays["av"][i], gas_volume_m3=arrays["gv"][i],
                    pore_volume_m3=arrays["pore_volume"], solubility_g_l=arrays["solubility"][i],
                    gas_density_g_l=arrays["density"][i], photograph_rgb=arrays["rgb"][i])
                render_frame(preview, arrays["rgb"][i], c, arrays["av"][i], arrays["gv"][i],
                             arrays["pore_volume"], record, title, limit)
                rows.append(record)
            # Reject stale outputs if annotations/model publication replaced a
            # selected map while this calculation was running.
            _, after = inventory(request)
            if after != inputs: raise ValueError("Selected phase maps or photographs changed during analysis")
            if any(sha(path) != digest for path, digest in source_bindings.items()) or core.provenance(method) != method_provenance:
                raise ValueError("Result-generating code changed during analysis; rerun with the new revision")
            result = {"format": "studio-concentration-mass-v2", "identity": identity,
                "method": method_provenance, "host_sha256": host_sha, "calibration": fit, "frames": rows,
                "mass_policy": asdict(mass_policy),
                "optical_noise_correction": optical_noise_correction,
                "result_source_manifest": str(snapshot/'manifest.json'), "physical_arrays": array_bindings,
                "selected_inputs": inputs, "color_limit_g_l": limit,
                "weak_optical_reference_fraction": float(optical.reference_weak.sum()/optical.domain.sum()),
                "complete": True, "created_utc": datetime.now(timezone.utc).isoformat()}
            render_mass_curve(folder/"mass_curve.png", rows, title)
            result["mass_curve"] = str(folder/"mass_curve.png")
            write(result_path, result); update("complete", result=str(result_path)); return result
    except Exception:
        update("failed", error=traceback.format_exc()); raise


def render_mass_curve(path, rows, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 4.5), layout="constrained"); FigureCanvasAgg(fig)
    ax = fig.subplots(); time = [r["elapsed_h"] for r in rows]
    for key, name, color in [("injected_mass_g", "Injected reference", "black"),
                              ("total_mass_g", "Estimated total", "#175aad"),
                              ("aqueous_mass_g", "Dissolved", "#169879"),
                              ("gas_mass_g", "Free gas", "#dc398b")]:
        ax.plot(time, [r[key] for r in rows], label=name, color=color)
    held = [r for r in rows if r["calibration_role"] == "held-out time"]
    ax.scatter([r["elapsed_h"] for r in held], [r["total_mass_g"] for r in held],
               marker="o", facecolors="white", edgecolors="#175aad", zorder=5, label="Held-out time")
    ax.set(xlabel="Time since injection start · h", ylabel="CO₂ mass · g",
           title=title+" · one fitted concentration curve")
    ax.grid(alpha=.2); ax.legend(); fig.savefig(path, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    run(parser.parse_args().request)

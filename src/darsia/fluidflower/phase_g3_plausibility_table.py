"""Label-free, source-bound gas-area table for FF_AC downstream spec section 4c.

Only exported probabilities, categorical reference maps, source manifests and
per-frame geometric receipts are read. No annotation loader, fitting, scoring,
figures or final-map production is part of this module.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .project import sha256_file
from .phase_g3_probability_layers import (
    load_exported_layers, validate_export_receipt, _read_frozen_map,
)

CONTRACT = "ffac-g3-label-free-plausibility-table-v1"
SPEC_SHA256 = "9d773b61c370dfa2e7f1116c4b858ea143ef8545581847810c52a96bf19f1bc9"
G3_SHA256 = "90f1d4608506273974e4cc758573f4c6467a823243abdaa02ef214b7286c9692"
CO2_SHA256 = "8c3727110be01dbef4849fad6849e50bcc8e28bdd1ee22a1c24c0ab8027c64e3"
RUNS = tuple(f"ac{n}" for n in range(17, 62) if n not in (36, 37))
OUTSIDE_COHORT = frozenset(("ac20", "ac24", "ac27", "ac29", "ac44", "ac51"))
THRESHOLDS = (.5, .6, .7)
REGIONAL_MODES = ("frozen", "refit", "off")
AREAS_MM2 = (0., 1., 2.5)
FIELDS = ("run", "cohort_status", "group", "frame_index", "elapsed_h",
          "configuration", "regional", "pixel_threshold", "minimum_area_mm2",
          "co2_px", "gas_px", "gas_share_of_co2", "gas_px_ESF", "gas_px_C",
          "reference_method", "storage_layout", "g3_quantization", "regional_quantization")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_metadata(run):
    run = run.casefold()
    if run not in RUNS:
        raise ValueError("Run is outside the fixed current 43-run inventory")
    n = int(run[2:])
    return dict(run=run, cohort_status="outside" if run in OUTSIDE_COHORT else "cohort",
                group="early" if n <= 35 else "mid" if n <= 49 else "late")


def transport_facies(facies, domain, homography, shape_hw):
    """Carry AC28-native categorical facies into this frame, never the reverse."""
    facies, domain = np.asarray(facies), np.asarray(domain, bool)
    h = np.asarray(homography, dtype=np.float64)
    if facies.ndim != 2 or domain.shape != facies.shape or facies.dtype != np.uint8:
        raise ValueError("AC28 reference needs a uint8 facies map and matching domain")
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("Invalid per-frame AC28-to-frame homography")
    height, width = map(int, shape_hw)
    if height < 1 or width < 1:
        raise ValueError("Invalid native shape")
    reference = np.where(domain, facies, 0).astype(np.uint8)
    return cv2.warpPerspective(reference, h, (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _mask_counts(gas, facies):
    return dict(gas_px=int(np.count_nonzero(gas)),
                gas_px_ESF=int(np.count_nonzero(gas & (facies == 1))),
                gas_px_C=int(np.count_nonzero(gas & (facies == 2))))


def configuration_counts(arrays, pixel_area_mm2, facies):
    """All 27 literal configurations, with nine CCL passes instead of 27.

    Comparing probabilities happens on the native grid after the export loader's
    documented float32 resize. NaNs never pass either probability threshold.
    Connected-component areas use the unchanged nominal per-frame pixel area.
    """
    co2 = np.asarray(arrays["co2"], bool)
    facies = np.asarray(facies)
    if facies.shape != co2.shape:
        raise ValueError("Transported facies and probability grids differ")
    if not np.isfinite(pixel_area_mm2) or pixel_area_mm2 <= 0:
        raise ValueError("Component filtering needs positive nominal pixel area")
    g3 = np.asarray(arrays["g3_pixel"])
    if g3.shape != co2.shape:
        raise ValueError("G3 probability grid differs from frozen CO2 grid")
    seals = (facies == 1, facies == 2)
    result = []
    for threshold in THRESHOLDS:
        pixel_gas = co2 & (g3 >= threshold)
        for mode in REGIONAL_MODES:
            gas = pixel_gas.copy()
            if mode != "off":
                probability = np.asarray(arrays["regional_" + mode])
                if probability.shape != co2.shape:
                    raise ValueError("Regional probability was not resized to native grid")
                gas &= probability >= .6
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                gas.astype(np.uint8), connectivity=4)
            sizes = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
            in_seals = [np.bincount(components[seal], minlength=count) for seal in seals]
            for minimum in AREAS_MM2:
                keep = sizes * float(pixel_area_mm2) >= minimum
                keep[0] = False
                result.append(dict(configuration=f"{mode}_t{threshold:g}_A{minimum:g}",
                                   regional=mode, pixel_threshold=threshold,
                                   minimum_area_mm2=minimum,
                                   gas_px=int(sizes[keep].sum()),
                                   gas_px_ESF=int(in_seals[0][keep].sum()),
                                   gas_px_C=int(in_seals[1][keep].sum()),
                                   reference_method="literal_section3_exported_probabilities"))
    return result


def frame_rows(arrays, metadata, facies, *, current_g3_labels=None):
    """Return counts only. References are visibly distinguished from the grid."""
    co2 = np.asarray(arrays["co2"], bool)
    nco2 = int(np.count_nonzero(co2))
    base = dict(**run_metadata(metadata["run"]), frame_index=int(metadata["frame_index"]),
                elapsed_h=float(metadata["elapsed_h"]), co2_px=nco2,
                storage_layout=metadata["storage_layout"],
                g3_quantization=("float16_native" if metadata["storage_layout"] == "native-float16"
                                 else "round_even_254p_uint8_255invalid_decode_div254"),
                regional_quantization="float16_then_float32_native_resize_if_needed")
    rows = configuration_counts(arrays, metadata["pixel_area_mm2"], facies)
    g0 = np.asarray(arrays["frozen_gas"], bool)
    if g0.shape != co2.shape or np.any(g0 & ~co2):
        raise ValueError("G0 reference gas lies outside frozen CO2 union")
    if current_g3_labels is not None:
        labels = np.asarray(current_g3_labels)
        if labels.shape != co2.shape or not np.array_equal(np.isin(labels, (1, 2)), co2):
            raise ValueError("Saved G3 reference changes the frozen CO2 union")
        reference_counts = _mask_counts(labels == 2, facies)
        method = "exact_saved_G3_map"
    else:
        reference_counts = {k: next(r for r in rows if r["configuration"] == "frozen_t0.6_A2.5")[k]
                            for k in ("gas_px", "gas_px_ESF", "gas_px_C")}
        method = "literal_export_reconstruction"
    for name, counts, reference_method in (("reference_G3_today", reference_counts, method),
                                          ("reference_G0", _mask_counts(g0, facies), "exact_frozen_G0_mask")):
        rows.append(dict(configuration=name, regional="frozen", pixel_threshold="",
                         minimum_area_mm2="", reference_method=reference_method,
                         **counts))
    return [dict(base, **row, gas_share_of_co2=(row["gas_px"] / nco2 if nco2 else None))
            for row in rows]


def _geometry_binding(native, metadata):
    index = int(metadata["frame_index"])
    frames = native["v5_track1_photo_geometry"]["frames"]
    matches = [f for f in frames if int(f["frame_index"]) == index]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous exact per-frame photo geometry")
    entry = matches[0]
    path = Path(entry["receipt_path"]).resolve()
    expected = metadata["domain_binding"]["frame_geometry_receipt_sha256"]
    if entry["receipt_sha256"] != expected or sha256_file(path) != expected:
        raise ValueError("Geometry SHA differs from the exported frame's binding")
    geometry = _read(path)
    if (str(geometry["run"]).casefold() != metadata["run"] or
            int(geometry["frame_index"]) != index or
            list(geometry["native_shape_hw"]) != list(metadata["native_shape_hw"])):
        raise ValueError("Geometry belongs to another run, frame or native grid")
    original_sha = native["v5_track1_photo_geometry"].get("original_source_manifest_sha256")
    if original_sha and geometry.get("source_manifest_sha256") != original_sha:
        raise ValueError("Geometry binds another historical source version")
    matrix = geometry["H_ac28_to_frame_native"]
    h = np.asarray(matrix, np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("Invalid registered AC28-to-frame geometry")
    return dict(path=str(path), sha256=expected, H_ac28_to_frame_native=matrix)


def build_request(export_root, spec_path, facies_path, *, facies_metadata_path,
                  facies_properties_path, expected_spec_sha256=SPEC_SHA256):
    """Freeze a table request only after the complete, non-provisional 864 export."""
    root, spec, facies = map(lambda p: Path(p).resolve(), (export_root, spec_path, facies_path))
    if sha256_file(spec) != expected_spec_sha256:
        raise ValueError("Section 4c specification SHA changed")
    job_path = root / "job_request.json"
    job = _read(job_path)
    inventory = job["binding"]["source_inventory"]
    if canonical_sha256(inventory) != job["binding"]["source_inventory_sha256"]:
        raise ValueError("Export source inventory SHA changed")
    keys = [(f["run"].casefold(), int(f["frame_index"])) for f in inventory]
    if len(keys) != 864 or len(set(keys)) != 864 or set(r for r, i in keys) != set(RUNS):
        raise ValueError("Table requires all 864 exact frames in all 43 current runs")
    with Path(facies_properties_path).open(encoding="utf-8-sig", newline="") as handle:
        properties = {int(row["id"]): row["name"] for row in csv.DictReader(handle)}
    if properties.get(1) != "ESF" or properties.get(2) != "C":
        raise ValueError("Barrier facies IDs 1/2 are not bound to ESF/C")
    facies_metadata = _read(facies_metadata_path)
    if facies_metadata.get("sha256", facies_metadata.get("output_sha256")) != sha256_file(facies):
        raise ValueError("AC28 facies file SHA differs from its provenance receipt")
    with np.load(facies, allow_pickle=False) as archive:
        reference, domain = archive["facies"], archive["domain"]
        if reference.dtype != np.uint8 or reference.ndim != 2 or domain.shape != reference.shape:
            raise ValueError("Invalid AC28-native facies/domain archive")
        reference_shape = list(reference.shape)
    native_cache, frames, model_bindings = {}, [], set()
    for item in inventory:
        run, index = item["run"].casefold(), int(item["frame_index"])
        path = root / "layers" / run / f"frame_{index:03d}" / "receipt.json"
        if not path.is_file():
            raise ValueError(f"Full probability export is incomplete at {run} frame {index}")
        receipt = _read(path)
        b, m = receipt["binding"], receipt["metadata"]
        if (receipt.get("contract") != "ffac-g3-downstream-probability-layers-v1" or
                receipt.get("complete") is not True or b.get("provisional") is not False):
            raise ValueError("Full probability export includes incomplete or provisional frames")
        if (b["run"] != run or int(b["frame_index"]) != index or m["run"] != run or
                int(m["frame_index"]) != index or b["g3_model_sha256"] != G3_SHA256 or
                m["co2_head_sha256"] != CO2_SHA256 or
                b["native_manifest_sha256"] != item["native_manifest_sha256"] or
                list(m["native_shape_hw"]) != item["native_shape_hw"] or
                abs(float(m["elapsed_h"]) - float(item["elapsed_h"])) > 1e-10 or
                Path(m["frozen_map_path"]).resolve() != Path(item["frozen_map_path"]).resolve()):
            raise ValueError("Export frame differs from frozen source/model inventory")
        model_bindings.add((b["g3_model_sha256"], b["regional_refit_sha256"],
                            m["regional_frozen_sha256"], m["co2_head_sha256"]))
        native_path = Path(m["native_manifest_path"]).resolve()
        if native_path != Path(item["native_manifest_path"]).resolve():
            raise ValueError("Export native source path differs from inventory")
        key = str(native_path)
        if key not in native_cache:
            if sha256_file(native_path) != item["native_manifest_sha256"]:
                raise ValueError("Native source manifest changed")
            native_cache[key] = _read(native_path)
        geometry = _geometry_binding(native_cache[key], m)
        reference_map = m.get("current_g3_map_path")
        if reference_map and not Path(reference_map).is_file():
            raise ValueError("Declared current G3 saved map is missing")
        frames.append(dict(run=run, frame_index=index, export_receipt_path=str(path),
                           export_receipt_sha256=sha256_file(path), geometry=geometry,
                           content_box_xywh=native_cache[key]["content_box_xywh"],
                           current_g3_map_path=reference_map,
                           current_g3_map_sha256=sha256_file(reference_map) if reference_map else None))
    if len(model_bindings) != 1:
        raise ValueError("Frames mix different frozen or refitted model heads")
    return dict(contract=CONTRACT, label_free=True, cpu_only=True, final_maps_created=False,
                spec_path=str(spec), spec_sha256=expected_spec_sha256,
                implementation_code_sha256=sha256_file(Path(__file__)),
                export_loader_code_sha256=sha256_file(Path(load_exported_layers.__code__.co_filename)),
                reference_map_reader_code_sha256=sha256_file(Path(_read_frozen_map.__code__.co_filename)),
                runtime_versions=dict(numpy=np.__version__, opencv=cv2.__version__),
                export_root=str(root), export_job_request_sha256=sha256_file(job_path),
                source_inventory_sha256=job["binding"]["source_inventory_sha256"],
                facies_path=str(facies), facies_sha256=sha256_file(facies),
                facies_metadata_path=str(Path(facies_metadata_path).resolve()),
                facies_metadata_sha256=sha256_file(facies_metadata_path),
                facies_properties_path=str(Path(facies_properties_path).resolve()),
                facies_properties_sha256=sha256_file(facies_properties_path),
                reference_shape_hw=reference_shape,
                model_sha256=dict(zip(("G3", "regional_refit", "regional_frozen", "CO2"),
                                      next(iter(model_bindings)))),
                barrier_ids={"ESF": 1, "C": 2}, barrier_rule="literal transported facies 1/2; exclude fault9 and boundary zones10/11",
                geometry_rule="per-frame source-bound H_ac28_to_frame_native, nearest-neighbour, border0",
                facies_certification="provisional boundary placement; secondary physical check, not mass eligibility",
                outside_cohort=sorted(OUTSIDE_COHORT),
                group_rule="early AC17–35; mid AC38–49; late AC50–61",
                quantization_note="27 configurations use exported float16 or decoded uint8 probabilities. Regional float16 is resized as float32 before threshold. Saved G3 maps are exact when available; otherwise the literal section3 frozen/.6/2.5 configuration is reconstructed from exported probabilities. The literal reference is quantized and does not reproduce additional legacy presence/eligibility restoration operators of historical saved maps; reference methods are not interchangeable.",
                frames=sorted(frames, key=lambda f: (int(f["run"][2:]), f["frame_index"])))


_REFERENCE = None


def _worker_init(facies_path, facies_sha256, threads):
    global _REFERENCE
    cv2.setNumThreads(threads)
    if sha256_file(facies_path) != facies_sha256:
        raise ValueError("Frozen AC28 facies changed before worker start")
    with np.load(facies_path, allow_pickle=False) as archive:
        _REFERENCE = (archive["facies"], archive["domain"])


def _process_frame(frame, request_sha256, output):
    path = Path(frame["export_receipt_path"])
    if sha256_file(path) != frame["export_receipt_sha256"]:
        raise ValueError("Frozen probability receipt changed")
    geo = frame["geometry"]
    if sha256_file(geo["path"]) != geo["sha256"]:
        raise ValueError("Frozen geometry receipt changed")
    reference = frame["current_g3_map_path"]
    if reference and sha256_file(reference) != frame["current_g3_map_sha256"]:
        raise ValueError("Frozen G3 reference map changed")
    folder = Path(output) / "frames" / frame["run"] / f"frame_{frame['frame_index']:03d}"
    rows_path, result_path = folder / "rows.json", folder / "receipt.json"
    binding = dict(request_sha256=request_sha256, frame=frame)
    if result_path.is_file():
        result = _read(result_path)
        if result.get("complete") is not True or result.get("binding") != binding:
            raise ValueError("Existing table frame binds different inputs")
        if not rows_path.is_file() or sha256_file(rows_path) != result["rows_sha256"]:
            raise ValueError("Completed table frame rows changed")
        if len(_read(rows_path)) != 29:
            raise ValueError("Completed table frame has wrong row count")
        validate_export_receipt(path)
        return dict(run=frame["run"], frame_index=frame["frame_index"],
                    receipt_path=str(result_path), receipt_sha256=sha256_file(result_path), reused=True)
    started = time.monotonic()
    arrays, receipt = load_exported_layers(path)
    facies = transport_facies(*_REFERENCE, geo["H_ac28_to_frame_native"],
                              receipt["metadata"]["native_shape_hw"])
    current_labels = None
    if reference:
        current_labels, rgba = _read_frozen_map(Path(reference), frame["content_box_xywh"])
        del rgba
    rows = frame_rows(arrays, receipt["metadata"], facies, current_g3_labels=current_labels)
    _atomic_json(rows_path, rows)
    result = dict(contract=CONTRACT, complete=True, binding=binding,
                  rows_path=str(rows_path), rows_sha256=sha256_file(rows_path), rows_count=len(rows),
                  source_files=receipt["files"], storage_layout=receipt["metadata"]["storage_layout"],
                  seconds=time.monotonic() - started, label_files_read=0,
                  final_maps_created=False, reference_G3_method=rows[-2]["reference_method"])
    _atomic_json(result_path, result)
    return dict(run=frame["run"], frame_index=frame["frame_index"],
                receipt_path=str(result_path), receipt_sha256=sha256_file(result_path), reused=False)


def build_plausibility_table(export_root, spec_path, facies_path, output, *,
                             facies_metadata_path, facies_properties_path,
                             workers=8, threads=1):
    """Parallel counts with immutable request, per-frame resume and one final CSV."""
    if not 1 <= workers <= 32 or not 1 <= threads <= 4:
        raise ValueError("Invalid bounded CPU allocation")
    output, export_root = Path(output).resolve(), Path(export_root).resolve()
    if output.is_relative_to(export_root):
        raise ValueError("Table output must be detached from the immutable export job")
    request = build_request(export_root, spec_path, facies_path,
                            facies_metadata_path=facies_metadata_path,
                            facies_properties_path=facies_properties_path)
    request_sha = canonical_sha256(request)
    request_path = output / "request.json"
    if output.exists() and any(output.iterdir()) and not request_path.is_file():
        raise ValueError("Table output contains unrelated existing files")
    if request_path.exists():
        if _read(request_path) != request:
            raise ValueError("Table request differs from existing immutable request")
    else:
        _atomic_json(request_path, request)
    started, completed, records = time.monotonic(), 0, []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(str(facies_path), request["facies_sha256"], threads)) as pool:
        futures = [pool.submit(_process_frame, f, request_sha, str(output)) for f in request["frames"]]
        try:
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                seconds = time.monotonic() - started
                print(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                       stage="label_free_table_frame_complete", frames=completed,
                                       total=864, run=record["run"], frame_index=record["frame_index"],
                                       reused=record["reused"], seconds=round(seconds, 2),
                                       eta_seconds=round(seconds/completed*(864-completed), 1))), flush=True)
        except BaseException:
            for pending in futures:
                pending.cancel()
            raise
    records.sort(key=lambda r: (int(r["run"][2:]), r["frame_index"]))
    csv_path = output / "plausibility_table.csv"
    temporary = csv_path.with_name(csv_path.name + "." + uuid.uuid4().hex + ".tmp")
    row_count = 0
    methods = {}
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for record in records:
                if sha256_file(record["receipt_path"]) != record["receipt_sha256"]:
                    raise ValueError("Table frame receipt changed before CSV assembly")
                frame_receipt = _read(record["receipt_path"])
                if sha256_file(frame_receipt["rows_path"]) != frame_receipt["rows_sha256"]:
                    raise ValueError("Table rows changed before CSV assembly")
                rows = _read(frame_receipt["rows_path"])
                for row in rows:
                    writer.writerow(row)
                    row_count += 1
                method = rows[-2]["reference_method"]
                methods[method] = methods.get(method, 0) + 1
        if row_count != 864*29:
            raise ValueError("Incomplete 29-by-864 plausibility table")
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(contract=CONTRACT, complete=True, request_sha256=request_sha,
                  request_file_sha256=sha256_file(request_path),
                  csv_path=str(csv_path), csv_sha256=sha256_file(csv_path),
                  row_count=row_count, frame_count=864, run_count=43,
                  configurations_count=27, reference_rows_per_frame=2,
                  reference_G3_methods=methods, frame_receipts=records,
                  label_files_read=0, final_maps_created=False, figures_created=False,
                  configuration_selected=False, mass_eligible=False,
                  cpu_workers=workers, threads_per_worker=threads,
                  elapsed_seconds=time.monotonic()-started,
                  code_sha256=sha256_file(Path(__file__)),
                  export_loader_code_sha256=sha256_file(Path(load_exported_layers.__code__.co_filename)))
    _atomic_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--facies", required=True)
    parser.add_argument("--facies-metadata", required=True)
    parser.add_argument("--facies-properties", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    result = build_plausibility_table(args.export_root, args.spec, args.facies, args.output,
                                      facies_metadata_path=args.facies_metadata,
                                      facies_properties_path=args.facies_properties,
                                      workers=args.workers, threads=args.threads)
    print(json.dumps({k: result[k] for k in ("complete", "csv_path", "csv_sha256",
                                           "row_count", "frame_count", "run_count")}), flush=True)


if __name__ == "__main__":
    main()

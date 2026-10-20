"""A synthetic series folder and its finished production job, in the job layer's formats.

The series, its camera folder, the producer's local results and the shared
job folder are written below one temporary folder. Machine and person names
are fictional. Arrays are tiny; nothing here runs an analysis.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np

COMMIT = "c" * 40
TOKEN = "a" * 32
PRODUCER = "example-producer"
SERIES_ID = "SYN"
LAYER_SHAPE = (4, 5)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, data: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def as_json(value) -> bytes:
    return (json.dumps(value, indent=1) + "\n").encode("utf-8")


def ref(alias: str, relative: str, data: bytes, role: str) -> dict:
    return {"root_alias": alias, "relative_path": relative, "sha256": sha(data), "bytes": len(data),
            "role": role}


def tiff(payload: bytes, *, artist: str | None = None) -> bytes:
    """A minimal little-endian TIFF with an optional Artist tag, then a payload."""
    entries, data = [], b""
    if artist is not None:
        text = artist.encode("ascii") + b"\x00"
        entries.append((0x013B, 2, len(text), text))
    ifd_size = 2 + 12 * len(entries) + 4
    offset = 8 + ifd_size
    body = struct.pack("<H", len(entries))
    for tag, kind, count, value in entries:
        if len(value) <= 4:
            body += struct.pack("<HHI", tag, kind, count) + value.ljust(4, b"\x00")
        else:
            body += struct.pack("<HHII", tag, kind, count, offset + len(data))
            data += value
    body += struct.pack("<I", 0)
    return b"II*\x00" + struct.pack("<I", 8) + body + data + payload


def layer_arrays(run_index: int, frame: int) -> dict:
    """A deterministic stored coarse layer of one frame."""
    rows, columns = LAYER_SHAPE
    base = np.arange(rows * columns, dtype=np.float32).reshape(LAYER_SHAPE) / 100.0
    gas = (base * (frame + 1) * (run_index + 1) / 10.0).astype(np.float32)
    dissolved = np.clip(0.5 - base, 0.0, 1.0).astype(np.float32)
    water = (1.0 - dissolved).astype(np.float32)
    return {
        "cell_mass_total_g": (gas + dissolved).astype(np.float32),
        "cell_mass_aqueous_g": dissolved,
        "cell_mass_gas_g": gas,
        "concentration_aqueous": (dissolved * 2).astype(np.float32),
        "saturation_free_gas": (gas / 10).astype(np.float32),
        "phase_water_area_fraction": water,
        "phase_dissolved_area_fraction": dissolved,
        "phase_gas_area_fraction": np.zeros(LAYER_SHAPE, np.float32),
        "phase_excluded_area_fraction": np.zeros(LAYER_SHAPE, np.float32),
        "mask_main_mass_eligible_area_fraction": np.ones(LAYER_SHAPE, np.float32),
        "static_pixel_area_m2_sum": np.full(LAYER_SHAPE, 1e-6, np.float32),
        "static_depth_m_mean": np.full(LAYER_SHAPE, 0.011, np.float32),
        "static_facies_majority": np.ones(LAYER_SHAPE, np.int32),
    }


def npz_bytes(arrays: dict) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def make_series(root: Path, *, runs=("r1", "r2"), frames: int = 3, resolution_mm: float = 1.0,
                tiers: dict | None = None, reviewer: str | None = None, extra: dict | None = None,
                extra_camera: dict | None = None, artist: str | None = None,
                series_toml_extra: str = "") -> SimpleNamespace:
    """Write a series, its camera folder, a finished job and its producer results below ``root``.

    ``extra`` adds series files {relative path: bytes} that every run reads;
    ``extra_camera`` adds reference images {path below the camera folder: bytes}
    that the first run reads. Returns the folders and, per run, the frames'
    raw names and SHA-256.
    """
    root = Path(root)
    series, camera, results, job = root / "series", root / "camera", root / "producer", root / "jobs" / "job-1"
    manifest = write(series / "series.toml", (
        'schema_version = 1\nseries_id = "SYN"\nname = "Synthetic series"\n'
        '[roots]\nseries = "."\nraw = "../camera"\n' + series_toml_extra).encode("utf-8"))
    plan = write(series / "plan" / "execution.toml",
                 b'format = "darsia-series-v1"\nadopted_for_execution = true\njob_id = "job-1"\n')
    calibration = write(series / "calibration" / "frozen.json", as_json({"gain": 2.0}))
    native = write(series / "config" / "native.json", as_json({"profile": "fixture"}))
    protocol = write(series / "protocols" / "injection.csv", b"time_h,rate\n0,1\n1,0\n")
    shared = {"plan/execution.toml": (plan, "configuration", "run.execution_series"),
              "series.toml": (manifest, "configuration", "run.series_manifest"),
              "calibration/frozen.json": (calibration, "calibration", "mass.calibration"),
              "config/native.json": (native, "configuration", "native.configuration"),
              "protocols/injection.csv": (protocol, "protocol", "mass.protocol")}
    if reviewer is not None:
        record = write(series / "config" / "phase_model_studio_reviewer.json",
                       as_json({"format": "darsia-phase-model-studio-reviewer-v1", "reviewer_name": reviewer}))
        shared["config/phase_model_studio_reviewer.json"] = (record, "configuration", "phase.reviewer")
    for relative, data in (extra or {}).items():
        shared[relative] = (write(series / relative, data), "configuration", "extra." + relative)
    lock = b"# Python 3.13. Exact environment.\nnumpy==2.4.4\n"
    job_document = {"format": "darsia-alias-job-v1", "commit": COMMIT, "precision": "float32",
                    "calibration_sha256": sha(calibration),
                    "environment_lock": {"path": "requirements-worker.lock", "sha256": sha(lock)},
                    "expected_runs": list(runs), "series": ref("series", "plan/execution.toml", plan, "configuration"),
                    "series_manifest": ref("series", "series.toml", manifest, "configuration"),
                    "series_id": SERIES_ID, "storage_profile": "full"}
    write(job / "job.json", as_json(job_document))
    frame_names = {}
    for run_index, run in enumerate(runs):
        baseline = write(camera / run / "BASE_0000.ARW", tiff(f"{run} baseline".encode(), artist=artist))
        raws = [write(camera / run / f"IMG_{index + 1:04d}.ARW", tiff(f"{run} frame {index}".encode() * 50))
                for index in range(frames)]
        cases = [f"{run}_f{index + 1}" for index in range(frames)]
        frame_names[run] = [{"name": f"IMG_{index + 1:04d}.ARW", "sha256": sha(data)} for index, data in enumerate(raws)]
        specs, previous = [], sha(baseline)
        for index, (case, data) in enumerate(zip(cases, raws)):
            specs.append({"case_id": case, "frame_index": index + 1, "elapsed_h": 0.5 * (index + 1),
                          "source": {"run": run, "selected": True, "frame_index": index + 1,
                                     "elapsed_h": 0.5 * (index + 1), "source_sha256": sha(data),
                                     "previous_source_sha256": previous}})
            previous = sha(data)
        config = write(series / "config" / "runs" / f"{run}.json", as_json({
            "format": "darsia-native-phase-n1-run-v1", "run": run, "history_policy": "camera chronology",
            "selection_index_field": "frame_index", "frames": specs,
            "phase": {"initial_previous": {"source": {"source_sha256": sha(baseline)}}},
            "native": {}, "mass": {}, "table_fields": ["case_id", "total_mass_g"]}))
        inputs = [{"reference": ref("series", f"config/runs/{run}.json", config, "configuration"),
                   "uses": ["run.configuration"]}]
        inputs += [{"reference": ref("series", relative, data, role), "uses": [use]}
                   for relative, (data, role, use) in shared.items()]
        inputs.append({"reference": ref("raw", f"{run}/BASE_0000.ARW", baseline, "image"),
                       "uses": ["phase.reference.raw_acquisition"]})
        inputs += [{"reference": ref("raw", f"{run}/IMG_{index + 1:04d}.ARW", data, "image"),
                    "uses": [f"frame.{case}.native.acquisition"]}
                   for index, (case, data) in enumerate(zip(cases, raws))]
        if run_index == 0:
            inputs += [{"reference": ref("raw", relative, write(camera / relative, data), "image"),
                        "uses": ["native.extra_reference"]} for relative, data in (extra_camera or {}).items()]
        task = {"format": "darsia-alias-task-v1", "run": run, "pipeline": "native-phase-n1", "backend": "cpu",
                "threads": 2, "commit": COMMIT, "precision": "float32", "job_id": "job-1",
                "ordered_case_ids": cases,
                "configuration": ref("series", f"config/runs/{run}.json", config, "configuration"),
                "input_catalog": {"format": "darsia-role-input-catalog-v1", "complete": True, "run": run,
                                  "inputs": inputs}}
        write(job / "results" / run / "task.json", as_json(task))
        attempt = results / "job-1" / run / ".attempts" / TOKEN
        lease = job / "leases" / f"{run}.{TOKEN}" / "artifacts"
        products = []

        def product(product_id, data, kind, *, shared_copy):
            relative = product_id.removeprefix(f"attempts/{TOKEN}/")
            write(attempt / relative, data)
            row = {"product_id": product_id, "machine_id": PRODUCER, "local_path": str(attempt / relative),
                   "sha256": sha(data), "bytes": len(data), "kind": kind,
                   "identity_basis": "original_producer_descriptor"}
            if shared_copy:
                write(lease / "metadata" / product_id, data)
                row["metadata_copy"] = {"relative_path": f"metadata/{product_id}", "sha256": sha(data),
                                        "bytes": len(data)}
            products.append(row)

        table = "case_id,total_mass_g,aqueous_mass_g,gas_mass_g\n"
        for index, case in enumerate(cases):
            arrays = layer_arrays(run_index, index)
            masses = {name: float(arrays[f"cell_mass_{key}_g"].sum(dtype=np.float64))
                      for key, name in (("total", "total_mass_g"), ("aqueous", "aqueous_mass_g"),
                                        ("gas", "gas_mass_g"))}
            table += f"{case},{masses['total_mass_g']!r},{masses['aqueous_mass_g']!r},{masses['gas_mass_g']!r}\n"
            result = {"case_id": case, "frame_index": index + 1, "elapsed_h": 0.5 * (index + 1), **masses,
                      "bindings": {"source_sha256": sha(raws[index])}, "backend": "cpu", "precision": "float32",
                      "compute_seconds": 2.0}
            product(f"frames/{case}/result.json", as_json(result), "frame_scalar", shared_copy=True)
            receipt = {"format": "darsia-storage-frame-v1", "case_id": case,
                       "attempt_directory": str(attempt),
                       "coarse": {"grid": {"format": "darsia-metric-storage-grid-v1", "shape": list(LAYER_SHAPE),
                                           "extent_m": [0.0, 0.005, 0.0, 0.004], "resolution_mm": resolution_mm,
                                           "axes": "row=y increasing, column=x increasing",
                                           "storage_precision": "float32"},
                                  "arrays": {}}}
            product(f"attempts/{TOKEN}/preparation/{case}/storage_stored_frame.npz", npz_bytes(arrays),
                    "intermediate", shared_copy=False)
            product(f"attempts/{TOKEN}/preparation/{case}/storage_completed.json", as_json(receipt),
                    "operational_receipt", shared_copy=True)
        product("mass_table.csv", table.encode("utf-8"), "scalar_table", shared_copy=True)
        index = as_json({"format": "darsia-products-index-v1", "run": run, "attempt_id": TOKEN,
                         "publishing_machine_id": PRODUCER, "products": products})
        write(lease / "products_index.json", index)
        completed = as_json({"complete": True, "run": run, "pipeline": "native-phase-n1", "backend": "cpu",
                             "precision": "float32", "case_count": frames, "ordered_case_ids": cases,
                             "wall_seconds": 6.0 * frames, "newly_computed_frames": frames,
                             "actual_worker": {"machine": PRODUCER + "-cpu", "pid": 1}})
        write(job / "results" / run / "completed.json", completed)
        write(job / "done" / f"{run}.json", as_json({
            "format": "darsia-alias-done-v1", "run": run, "attempt_id": TOKEN,
            "completed": {"relative_path": f"results/{run}/completed.json", "sha256": sha(completed),
                          "bytes": len(completed)},
            "products_index": {"relative_path": f"leases/{run}.{TOKEN}/artifacts/products_index.json",
                               "sha256": sha(index), "bytes": len(index)}}))
    if tiers is None:
        tiers = {"S50": {"runs": {run: [row["name"] for row in frame_names[run]] for run in runs}}}
    write(series / "image_selection" / "published_tiers.json",
          as_json({"format": "darsia-published-tiers-v1", "tiers": tiers}))
    return SimpleNamespace(root=root, series=series, camera=camera, results=results, job=job, runs=list(runs),
                           frames=frame_names, lock=lock)

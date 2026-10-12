"""Run the three prepared full fits sequentially, reserving the GPU for the user.

This durable local worker writes progress and per-series logs. It never replaces
an existing model or declares candidates scientifically accepted.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constrain_resources():
    # Set these before importing any numerical packages, including via DarSIA.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONUNBUFFERED"] = "1"
    import psutil
    process = psutil.Process()
    eligible = process.cpu_affinity()
    selected = eligible[-min(4, len(eligible)):]
    process.cpu_affinity(selected)
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(10)
    return {"backend": "cpu", "gpu_disabled": True, "blas_threads": 2,
            "cpu_affinity": process.cpu_affinity(), "priority": "below-normal",
            "stage_workers": 1, "extraction_workers": 1, "opencv_threads": 2}


def verify_request(series, entry):
    from darsia.fluidflower.phase_features import GENERIC_FEATURE_VERSION
    request = read(Path(entry["root"]) / "training_request.json")
    settings = request["settings"]
    expected = {"backend": "cpu", "stage_workers": 1, "extraction_workers": 1,
                "opencv_threads": 2, "series": [series]}
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Unexpected resource/scope setting {series}/{key}: {settings.get(key)}")
    if GENERIC_FEATURE_VERSION != "baseline-aware-bright-grain-v4":
        raise ValueError("The new pre-injection bright-grain features are not installed")
    if settings.get("fit_heads") != ["co2_presence", "gas_within_co2"]:
        raise ValueError("Both optical heads must be freshly fitted")
    if settings.get("inherited_heads"):
        raise ValueError("This iteration must not inherit any fitted optical head")
    for bundle in request["bundles"]:
        for role in ("decisions", "combined_manifest"):
            record = bundle[role]
            if digest(record["snapshot_path"]) != record["sha256"]:
                raise ValueError("Frozen training evidence changed")
    from darsia.fluidflower.phase_workflow import build_phase_training_command
    # Rebuild from the checked immutable request instead of executing arbitrary
    # command strings from review or manifest content.
    command = build_phase_training_command(request)
    command[0] = sys.executable
    return request, command


def verify_backend_report(report, allowed_backends=("cpu",)):
    backend = report["final_fit"]["backend"]
    if backend not in allowed_backends:
        raise ValueError(f"Unexpected fitting backend: {backend}")
    numerical = report.get("numerical_validation", {})
    if not numerical.get("passed"):
        raise ValueError("Numerical validation did not pass")
    if backend == "cpu" and numerical.get("cuda_checked") is not False:
        raise ValueError("CPU-only numerical validation has an unexpected CUDA claim")
    if backend == "cuda" and not report.get("cpu64_cuda32_parity", {}).get("passed"):
        raise ValueError("GPU fit did not pass CPU/CUDA numerical parity")
    if not report["final_fit"].get("converged"):
        raise ValueError("Final fit did not converge")
    return backend


def verify_fit(series, entry, request, *, allowed_backends=("cpu",)):
    import numpy as np
    from darsia.fluidflower.phase_features import GENERIC_FEATURE_VERSION
    from darsia.fluidflower.phase_learning import HierarchicalPhaseModel, inspect_phase_model_bundle
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_training import trainer as t
    root = Path(entry["root"])
    summary = read(root / "summary.json")
    preflight = read(root / "evidence_preflight.json")
    inventory = summary["inventory"]
    if inventory["regions"] != preflight["cumulative_regions"]:
        raise ValueError("Fitted region count differs from the frozen full dataset")
    if inventory["datasets"] != [series] or inventory["series_scope"] != series:
        raise ValueError("Fitted model series differs from the prepared request")
    verified, fit_backends = {}, {}
    with np.load(Path(entry["cache"]) / "result_staging/training_samples.npz", allow_pickle=False) as samples:
        names = list(samples["feature_names"])
        rows = samples["features"]
        stride = max(1, len(rows) // 10000)
        subset = rows[::stride][:10000]
    for adapter, record in summary["models"].items():
        if set(record["heads"]) != {"co2_presence", "gas_within_co2"}:
            raise ValueError("The fitted model is missing an optical head")
        model = HierarchicalPhaseModel.load(root, adapter=adapter)
        if model.feature_contract.generic_version != GENERIC_FEATURE_VERSION:
            raise ValueError("Model did not train with baseline-aware grain features")
        selected = subset[:, [names.index(name) for name in model.feature_contract.feature_names]]
        scores = model.predict_features(selected)
        if any(not np.all(np.isfinite(values)) for values in scores.values()):
            raise ValueError("Reloaded model produced nonfinite probabilities")
        partition = sum(scores[name] for name in (
            "untouched_water_probability", "aqueous_co2_probability", "free_co2_gas_probability"))
        if not np.allclose(partition, 1., atol=1e-6):
            raise ValueError("Reloaded phase probabilities do not form a partition")
        for head, report in record["heads"].items():
            payload = read(root / "models" / adapter / f"{head}_signal.json")
            provenance = payload["training_provenance"]
            if provenance["campaign_id"] != request["model_id"]:
                raise ValueError("Fitted head has a different campaign identity")
            fit_backends[f"{adapter}/{head}"] = verify_backend_report(report, allowed_backends)
        verified[adapter] = {"features": len(model.feature_contract.feature_names),
                             "reloaded_prediction_rows": len(selected),
                             "heads": list(record["heads"])}
    if physics_runtime_identity(load_physics_profile(root)) != physics_runtime_identity(load_physics_profile(Path(entry["parent"]))):
        raise ValueError("Inherited physics changed during training")
    inspect_phase_model_bundle(root)
    receipt = {"completed_utc": utc(), "series": series, "model_id": request["model_id"],
               "regions": inventory["regions"], "runs": summary["runs"], "adapters": verified,
               "model_sha256": _model_identity(root)[1], "training_seconds": summary["seconds"],
               "gpu_used": "cuda" in fit_backends.values(), "fit_backends": fit_backends,
               "cpu_numerics_verified": True,
               "native_validation_pending": True, "human_review_required": True,
               "publication_performed": False}
    t._write_json(root / "baseline_grain_fit_verification.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=Path, required=True)
    args = parser.parse_args()
    resources = constrain_resources()
    iteration = args.iteration.resolve()
    entries = read(iteration)
    if set(entries) != {"ffac", "ffum", "ffbm"}:
        raise ValueError("Exactly the three requested series are required")
    root = iteration.parent
    status_path = root / "training_status.json"
    if status_path.exists():
        raise FileExistsError("This iteration already has a worker record; inspect it before resuming")
    repo = process_directory()
    source_root = package_root()
    source_paths = [Path(__file__), source_root / "phase_features.py",
                    *sorted((source_root / "phase_training").glob("*.py"))]
    frozen_code = {}
    for path in source_paths:
        relative = path.relative_to(repo)
        target = root / "training_code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        frozen_code[str(path)] = digest(target)
    write(root / "training_code_sha256.json", frozen_code)
    checked = {series: verify_request(series, entries[series]) for series in entries}
    status = {"started_utc": utc(), "pid": os.getpid(), "status": "running",
              "resources": resources, "series": {name: {"status": "queued"} for name in entries}}
    write(status_path, status)
    # UM first gives the earliest result on the user's concrete grain examples.
    for series in ("ffum", "ffac", "ffbm"):
        try:
            for path, expected in frozen_code.items():
                if digest(path) != expected:
                    raise ValueError(f"Training source changed after freezing: {path}")
            request, command = checked[series]
            log_path = root / f"{series}_training.log"
            state = {"status": "running", "started_utc": utc(), "log": str(log_path)}
            status["series"][series] = state
            write(status_path, status)
            with log_path.open("w", encoding="utf-8") as log:
                flags = ((subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                         if os.name == "nt" else 0)
                child = subprocess.Popen(command, cwd=repo, env=os.environ.copy(),
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                state["pid"] = child.pid
                write(status_path, status)
                returncode = child.wait()
            state["returncode"] = returncode
            if returncode:
                raise RuntimeError(f"Training exited with code {returncode}; see {log_path}")
            state["status"] = "verifying"
            write(status_path, status)
            state["verification"] = verify_fit(series, entries[series], request)
            state["status"] = "trained_and_checked"
        except Exception as error:
            status["series"][series].update(status="failed", error=f"{type(error).__name__}: {error}")
        status["series"][series]["completed_utc"] = utc()
        write(status_path, status)
    status["status"] = ("trained_and_checked" if all(item["status"] == "trained_and_checked"
                          for item in status["series"].values()) else "needs_attention")
    status["completed_utc"] = utc()
    write(status_path, status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()

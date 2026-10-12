"""Exact, frozen-candidate phase maps for Studio reference photographs.

The GUI-facing resolver only validates identities and starts an isolated worker.
Workers replay strictly earlier reviewed images, then evaluate the reference on
a disposable physical history. They never update a model or its validation.
"""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import load_snapshot_module, canonical_metadata_paths

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import OrderedDict, deque
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Mapping

FORMAT = "exact-reference-phase-map-v1"
POLICY = "strictly-earlier-reviewed-history-disposable-reference-v1"
LAYERS = {"all", "gas", "aqueous", "raw", "context_added", "context_removed", "context_uncertain"}
# This source revision has the same prediction/replay implementation. Accept its
# receipts only when every computational input still matches; cache/UI changes
# must not force users to repeat minutes of verified native inference.
COMPATIBLE_METHODS = {
    # The pre-package implementation: only imports and verified storage aliases moved.
    "0ee6d15d30ba38ec51657c166805f160b9b64a3deb32c644a9ac217bc7e64943",
    "3d985c4c32fc6b63ab153b99fce01d15456ccd87afbfc0e6746a29b351f8e208",
    "b5982f21c4ae272950b1347679b385cd0c2ee893ebb8225efc96f76230c5f996",
    "28c6e24a3cedceb7ed04941d9fdb85d1ba3affa0f2d6912e23d234c1fca1ca4c",
    "1701cdee6a8fe42a9a039aa6fbe28c0622edf814e6fa7cb67ab33140b225f482",
    "76ba451889c8031c0f4bffec25c2ad0a6b748971b390997a9508615686f58901",
    "d74a32106a7f4f99ba2f2cf01641639174ac3bf7c7303227929387ba52ab6ec3",
    "d1118b2ae90443aa62153c2ca2ecc860ca35dc3af05b7b60b5baae4d8bdb2014",
}
WINDOWS_SHARED_CACHE = local_cache_root() / 'phase-reference-maps'
_HASH_CACHE = OrderedDict()
_HASH_LOCK = threading.RLock()
_RESOLVE_LOCK = threading.RLock()


def _read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    path = resolve_review_path(path)
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    # Validate the file revision on every use, but read unchanged NAS images only
    # once per process, including when the selected phase layer changes.
    with _HASH_LOCK:
        cached = _HASH_CACHE.get(path)
        if cached and cached[0] == signature:
            _HASH_CACHE.move_to_end(path)
            return cached[1]
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"Reference asset changed while hashing: {path}")
    result = h.hexdigest()
    with _HASH_LOCK:
        _HASH_CACHE[path] = (signature, result)
        while len(_HASH_CACHE) > 4096:
            _HASH_CACHE.popitem(last=False)
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cache_root():
    override = os.environ.get("FFAC_PHASE_REFERENCE_MAP_CACHE")
    if override:
        return Path(override)
    if os.name == "nt":
        # AppData is privately redirected when launched from a packaged app.
        # Studio and preparation workers must see the same receipts and maps.
        return WINDOWS_SHARED_CACHE
    from darsia.fluidflower.execution import default_execution_cache_root
    return default_execution_cache_root() / "phase-reference-maps"


def _receipt_cache_roots():
    current = cache_root()
    roots = [current]
    if os.name == "nt" and current == WINDOWS_SHARED_CACHE:
        local = Path.home() / "AppData" / "Local"
        roots.append(local / "DarSIA/FluidFlower/execution/phase-reference-maps")
        roots.extend((local / "Packages").glob(
            "*/LocalCache/Local/DarSIA/FluidFlower/execution/phase-reference-maps"))
    return roots


def _import_cached_layers(layers, folder):
    """Move verified display assets out of a private cache without changing pixels."""
    result = {}
    for layer, item in layers.items():
        if layer not in LAYERS:
            raise ValueError("Unknown cached reference layer")
        source = Path(item["path"])
        if _sha(source) != item["sha256"]:
            raise ValueError("Legacy reference map differs from its receipt")
        target = folder / "imported_layers" / (layer + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if _sha(temporary) != item["sha256"]:
                raise ValueError("Reference map changed while importing")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        result[layer] = {**item, **_asset(target)}
    return result


def _frame(root, run, index):
    # This is the original predictor's input, not a display thumbnail.
    return Path(root) / "native_frames" / run / f"frame_{index:03d}.jpg"


def _asset(path):
    path = resolve_review_path(path)
    return {"path": str(path), "sha256": _sha(path)}


def _strictly_earlier(frames, elapsed):
    previous = float("-inf")
    result = []
    for index, row in enumerate(frames):
        value = float(row["elapsed_h"])
        if not previous < value:
            raise ValueError("Reviewed reference history must have strictly increasing explicit times")
        previous = value
        if value < elapsed:
            result.append({**row, "frame_index": index})
    return result


def _original_layers(job, index):
    run, fingerprint = job["run"], job["params_sha256"]
    folder = Path(job["phase_map_output_dir"]) / "native_phase_maps" / run / f"A_{fingerprint[:16]}"
    name = f"frame_{index:03d}.png"
    paths = {"all": folder / name, "gas": folder / "gas" / name, "aqueous": folder / "aqueous" / name}
    if job.get("raw_phase_map_output_dir"):
        paths["raw"] = Path(job["raw_phase_map_output_dir"]) / "native_phase_maps" / run / f"A_{job['optical_model_sha256'][:16]}" / name
    if job.get("presence_diagnostics_dir"):
        for layer, folder_name in (("context_added", "added_aqueous"), ("context_removed", "suppressed_presence"), ("context_uncertain", "unresolved")):
            paths[layer] = Path(job["presence_diagnostics_dir"]) / run / folder_name / name
    return {key: _asset(value) for key, value in paths.items() if value.is_file()}


def _display_layers(layers):
    from darsia.fluidflower.phase_rendering import two_phase_overlay_path
    result = {}
    for layer, source in layers.items():
        if layer in {"all", "gas", "aqueous", "raw"}:
            phase = "all" if layer == "raw" else layer
            display = two_phase_overlay_path(Path(source["path"]), phase, cache_root=cache_root() / "display")
            result[layer] = {**_asset(display), "source": source}
        else:
            result[layer] = source
    return result


def build_plan(reference, review_root, run, candidate):
    """Bind one immutable request to the actual candidate's validation package."""
    from darsia.fluidflower.tools.phase_injection_references import resolve_reference_asset
    from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_training.radiometry import analysis_review_root

    candidate = dict(candidate) if isinstance(candidate, Mapping) else candidate.provenance()
    run = run.casefold()
    if candidate.get("run", run).casefold() != run:
        raise ValueError("Reference candidate belongs to a different experiment")
    validation = Path(candidate.get("phase_map_output_dir") or "").parent
    binding_path, manifest_path = validation / "validation_binding.json", validation / "postflight_manifest.json"
    binding, campaign = _read(binding_path), _read(manifest_path)
    if binding.get("selected_candidates", {}).get(run) != candidate.get("candidate_id") or binding.get("pipeline_sha256") != candidate.get("params_sha256"):
        raise ValueError("Reference candidate does not match its frozen validation binding")
    matches = [j for j in campaign["jobs"] if j.get("run") == run and j.get("candidate_id") == candidate["candidate_id"]]
    if len(matches) != 1:
        raise ValueError("Reference candidate has no unique original validation job")
    job = matches[0]
    model_root = resolve_review_path(binding["model_root"])
    if _model_identity(model_root) != (binding["model_id"], binding["model_sha256"]):
        raise ValueError("Frozen reference model identity changed")
    if candidate.get("optical_model_sha256") and candidate["optical_model_sha256"] != binding["model_sha256"]:
        raise ValueError("Reference candidate optical model differs")
    profile = load_physics_profile(model_root)
    if physics_runtime_identity(profile) != binding.get("physics_runtime"):
        raise ValueError("Frozen reference physical runtime changed")
    if binding.get("radiometry_profile") is not None:
        raise ValueError("Exact reference maps for a separate model radiometry profile are not yet available")
    native_root = resolve_review_path(analysis_review_root(resolve_review_path(job.get("analysis_source_review_root") or job["review_output_dir"]), run))
    native_path = native_root / "native_frames" / run / "manifest.json"
    native, native_sha = _read(native_path), _sha(native_path)
    supplied_native = Path(review_root) / "native_frames" / run / "manifest.json"
    if _sha(supplied_native) != native_sha:
        raise ValueError("Displayed reference native geometry differs from candidate's actual input")
    frames = native["frame_sources"]
    baseline = _asset(native_root / "native_references" / run / "pre_injection.jpg")
    exact_index = None
    if reference is None:
        ref_metadata = native.get("reference") or {}
        index = ref_metadata.get("timeline_frame_index")
        elapsed = ref_metadata.get("elapsed_h")
        if index is not None:
            index = int(index)
            if not float(frames[index]["elapsed_h"]) < 0:
                raise ValueError("Baseline timeline index is not a pre-injection image")
            elapsed = frames[index]["elapsed_h"]
            if _sha(_frame(native_root, run, index)) == baseline["sha256"]:
                exact_index = index
        ref = {"id": "baseline", "label": "Before injection", "source_name": ref_metadata.get("source_name", "Pre-injection baseline"),
               "elapsed_h": elapsed, "phase_time_h": float(elapsed) if elapsed is not None else -1e-6,
               "time_basis": "native reference metadata" if elapsed is not None else "pre-injection control; acquisition time not recorded",
               "photo": baseline, "native_manifest_sha256": native_sha}
        replay = []
    else:
        reference = dict(reference)
        if reference.get("native_manifest_sha256") != native_sha:
            raise ValueError("Selected reference native identity is stale")
        photo = _asset(resolve_reference_asset(reference, Path(review_root), run))
        elapsed = float(reference["elapsed_h"])
        index = reference.get("frame_index")
        if index is not None:
            index = int(index)
            original = frames[index]
            if abs(float(original["elapsed_h"]) - elapsed) > 1e-9 or original.get("source_name", "").casefold() != reference["source_name"].casefold():
                raise ValueError("Reference frame index does not identify the selected acquisition")
            if _sha(_frame(native_root, run, index)) == photo["sha256"]:
                exact_index = index
        ref = {**reference, "photo": photo, "phase_time_h": elapsed}
        replay = _strictly_earlier(frames, elapsed)
    history = [{"frame_index": row["frame_index"], "source_name": row.get("source_name"), "elapsed_h": float(row["elapsed_h"]),
                "photo": _asset(_frame(native_root, run, row["frame_index"]))} for row in replay]
    # Baselines are evaluated independently even if their recorded time follows
    # another pre-injection acquisition in a historical timeline.
    plan = {"format": FORMAT, "policy": POLICY, "method_sha256": _sha(__file__), "run": run, "dataset": job["dataset"],
            "candidate_id": candidate["candidate_id"], "pipeline_sha256": binding["pipeline_sha256"],
            "model_root": str(model_root), "model_id": binding["model_id"], "model_sha256": binding["model_sha256"],
            "physics_runtime": binding.get("physics_runtime"), "binding": _asset(binding_path), "campaign": _asset(manifest_path),
            "native_manifest": _asset(native_path), "native_root": str(native_root), "baseline": baseline,
            "canvas_wh": native["canvas_wh"], "content_box_xywh": native["content_box_xywh"],
            "reference": ref, "replay": history, "job": job, "inference_provenance": binding.get("inference_provenance"),
            "exact_reviewed_frame_index": exact_index}
    if exact_index is not None:
        plan["reviewed_layers"] = _original_layers(job, exact_index)
    if profile is not None and "unified_phase_decision" in profile:
        plan["decision_cache_adapter_sha256"] = _sha(tool_path("phase_decision_reference_cache.py"))
        plan["decision_evidence_reader_sha256"] = _sha(tool_path("materialize_phase_decision_revision.py"))
    plan["key"] = _digest(plan)
    return plan


def _check_assets(plan):
    assets = [plan["binding"], plan["campaign"], plan["native_manifest"], plan["baseline"], plan["reference"]["photo"]]
    assets += [row["photo"] for row in plan["replay"]]
    for item in assets:
        if _sha(item["path"]) != item["sha256"]:
            raise ValueError(f"Reference input changed: {item['path']}")


def _computation_inputs(plan):
    result = {key: value for key, value in plan.items() if key not in {"key", "method_sha256"}}
    # Materialization changes UI readiness/path metadata after the first click.
    # The hash-verified pixels and acquisition identity are the model inputs.
    # Two injectors can end in the same acquisition. Their button IDs differ,
    # but identical pixels, time and preceding history have the same phase map.
    result["reference"] = {key: plan["reference"].get(key) for key in (
        "source_name", "elapsed_h", "phase_time_h", "native_manifest_sha256", "photo")}
    return result


def _reuse_compatible_receipt(plan):
    expected = _computation_inputs(plan)
    paths = (path for root in _receipt_cache_roots() for path in root.glob("*/receipt.json"))
    for receipt_path in paths:
        try:
            receipt = _read(receipt_path)
            if receipt.get("method_sha256") not in COMPATIBLE_METHODS | {plan.get("method_sha256")}:
                continue
            original = {key: receipt[key] for key in plan if key != "key"}
            if _digest(original) != receipt.get("key") or canonical_metadata_paths(_computation_inputs(original)) != canonical_metadata_paths(expected):
                continue
            for item in receipt["layers"].values():
                if _sha(item["path"]) != item["sha256"]:
                    raise ValueError("Compatible reference map differs from its receipt")
            migrated = {**receipt, **plan, "reused_receipt": _asset(receipt_path)}
            if receipt_path.parent.parent.resolve() != cache_root().resolve():
                migrated["layers"] = _import_cached_layers(receipt["layers"], cache_root() / plan["key"])
            _write(cache_root() / plan["key"] / "receipt.json", migrated)
            return migrated
        except (OSError, KeyError, ValueError):
            # An unrelated incomplete/old cache entry cannot block this request.
            continue
    return None


def _cached(plan, layer):
    receipt_path = cache_root() / plan["key"] / "receipt.json"
    if receipt_path.is_file():
        receipt = _read(receipt_path)
    else:
        receipt = _reuse_compatible_receipt(plan) if "reference" in plan else None
        if receipt is None:
            return None
    if receipt.get("key") != plan["key"]:
        raise ValueError("Reference map receipt identity differs")
    item = receipt.get("layers", {}).get(layer)
    if item is None:
        raise ValueError(f"The exact reference has no {layer} phase layer")
    if _sha(item["path"]) != item["sha256"]:
        raise ValueError("Reference map asset differs from its receipt")
    return Path(item["path"])


@contextmanager
def _worker_lock(path):
    # Advisory OS lock automatically releases if a worker exits unexpectedly.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            deadline = time.monotonic() + 3600
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Another exact reference worker is still running")
                    time.sleep(.5)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def resolve_reference_overlay(reference, review_root, run, candidate, side, layer, *, cached_only=False,
                              progress=None):
    """Resolve a hash-verified map; all native inference is in a child process."""
    if side not in {"A", "B"} or layer not in LAYERS:
        raise ValueError("Unknown candidate side or exact reference phase layer")
    # Background warming may check prepared photos/maps; it never prepares a
    # source image or starts a native inference/replay on its own.
    if cached_only and reference is not None and reference.get("needs_materialization"):
        return None
    with _RESOLVE_LOCK:
        plan = build_plan(reference, Path(review_root), run, candidate)
        ready = _cached(plan, layer)
        if ready:
            return ready
        if plan.get("reviewed_layers", {}).get("all"):
            # Hash-verified exact timeline maps need no model imports/child boot.
            compute_request(plan)
            return _cached(plan, layer)
        if cached_only:
            return None
        return _resolve_fresh(plan, layer, progress=progress)


def _resolve_fresh(plan, layer, *, progress=None):
    folder = cache_root() / plan["key"]
    request = folder / "request.json"
    _write(request, plan)
    environment = dict(os.environ)
    environment["FFAC_PHASE_REFERENCE_MAP_CACHE"] = str(cache_root().resolve())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "2"
    repo = process_directory()
    interpreter = Path(sys.executable)
    command = [str(interpreter if interpreter.is_file() else Path(sys.executable)),
               "-m", "darsia.fluidflower.tools.phase_reference_maps", "--request", str(request)]
    if progress is None:
        result = subprocess.run(command, cwd=repo, env=environment, capture_output=True,
                                text=True, timeout=3600,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError("Exact reference phase-map preparation failed: "
                               + (result.stderr or result.stdout)[-4000:])
        result_path = _cached(plan, layer)
        if result_path is None:
            raise RuntimeError("Reference worker completed without a bound map receipt")
        return result_path
    replay_positions = {int(row["frame_index"]): index + 1
                        for index, row in enumerate(plan.get("replay") or [])}
    replay_prefix = f"{plan.get('run')} reference {plan.get('reference', {}).get('id')}: replay "
    label = plan.get("reference", {}).get("label") or plan.get("reference", {}).get("id")
    output_tail = deque(maxlen=256)
    reader_errors = []
    with subprocess.Popen(command, cwd=repo, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as worker:
        def read_output():
            try:
                for line in worker.stdout:
                    output_tail.append(line)
                    if progress is not None and line.startswith(replay_prefix):
                        frame = int(line[len(replay_prefix):].split(" ", 1)[0])
                        if frame in replay_positions:
                            progress(f"{plan['run'].upper()}: {label} — replay "
                                     f"{replay_positions[frame]}/{len(replay_positions)}")
            except Exception as error:
                reader_errors.append(error)
                try:
                    worker.kill()
                except OSError:
                    pass

        reader = threading.Thread(target=read_output, name="phase-reference-progress", daemon=True)
        reader.start()
        try:
            returncode = worker.wait(timeout=3600)
        except subprocess.TimeoutExpired as error:
            worker.kill()
            worker.wait()
            reader.join()
            raise subprocess.TimeoutExpired(command, error.timeout,
                                            output="".join(output_tail)[-4000:]) from error
        reader.join()
    if reader_errors:
        raise reader_errors[0]
    if returncode:
        raise RuntimeError("Exact reference phase-map preparation failed: " + "".join(output_tail)[-4000:])
    result_path = _cached(plan, layer)
    if result_path is None:
        raise RuntimeError("Reference worker completed without a bound map receipt")
    return result_path


def _load_frozen_backend(validation, expected):
    """Use the recorded implementation, independent of later source edits."""
    snapshot = Path(validation) / "code_snapshot" / "phase_inference_benchmark_backends.py"
    source = snapshot if snapshot.is_file() else tool_path("phase_inference_benchmark_backends.py")
    if _sha(source) != expected:
        raise ValueError("Frozen CUDA reference backend snapshot differs or is unavailable")
    return load_snapshot_module("reference_frozen_cuda_backend_" + expected, source)


@contextmanager
def _backend_import(backend):
    """Bind the wrapper's absolute import to the verified backend in this worker."""
    import darsia.fluidflower.tools as scripts
    name = "darsia.fluidflower.tools.phase_inference_benchmark_backends"
    absent = object()
    previous_module = sys.modules.get(name, absent)
    previous_attribute = getattr(scripts, "phase_inference_benchmark_backends", absent)
    sys.modules[name] = backend
    scripts.phase_inference_benchmark_backends = backend
    try:
        yield
    finally:
        if previous_module is absent:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module
        if previous_attribute is absent:
            delattr(scripts, "phase_inference_benchmark_backends")
        else:
            scripts.phase_inference_benchmark_backends = previous_attribute


def _predictor(plan, model, baseline):
    provenance = plan.get("inference_provenance") or {}
    backend = provenance.get("backend")
    if backend in {None, "cpu-tiles", "cpu-reference-tiled", "cpu-reference-tiles", "cpu-reference-strips"}:
        from darsia.fluidflower.phase_validation import predict_image_tiled
        return lambda current, previous, index, previous_index: predict_image_tiled(model, current, baseline, previous_rgb=previous), None, {"backend": "cpu-reference-tiled"}
    if backend not in {"cuda-float32-cached-features-checked", "cuda-float32-checked",
                       "cuda-float32-cached-features", "cuda-float32"}:
        raise ValueError(f"Unsupported frozen reference backend: {backend}")
    validation = Path(plan["binding"]["path"]).parent
    wrapper_info = provenance.get("execution_wrapper")
    checked_validation = validation
    transfer_path = (Path(plan["model_root"]) / "transfer_review.json"
                     if wrapper_info is None else None)
    if transfer_path is not None and transfer_path.is_file():
        # Preserve the previously successful transfer execution path. Older
        # packages inherit their source model's recorded transient wrapper.
        parent = Path(_read(transfer_path)["source_model"])
        parent_binding = parent / "validation/validation_binding.json"
        if parent_binding.is_file():
            wrapper_info = (_read(parent_binding).get("inference_provenance") or {}).get("execution_wrapper")
            if wrapper_info is not None:
                validation = parent / "validation"
    if wrapper_info is None:
        # Standard Studio retraining uses this materializer directly, without
        # the transient wrapper. Keep its frozen implementation and explicit
        # production/audit policy when preparing an optional reference later.
        checked_source = checked_validation / "code_snapshot" / "materialize_phase_iteration.py"
        checked_sha = provenance.get("checked_cuda_sha256") or provenance.get("materializer_sha256")
        if checked_source.is_file() and _sha(checked_source) == checked_sha:
            backends = _load_frozen_backend(checked_validation, provenance["backend_sha256"])
            with _backend_import(backends):
                module = load_snapshot_module("reference_frozen_checked_cuda_" + checked_sha, checked_source)
                options = {}
                if "verify_numerics" in provenance:
                    import inspect
                    if "verify_numerics" in inspect.signature(module.CheckedCuda).parameters:
                        options["verify_numerics"] = bool(provenance["verify_numerics"])
                    elif provenance["verify_numerics"] is False:
                        raise ValueError("Frozen CUDA materializer cannot honor production verification policy")
                predictor = module.CheckedCuda(model, baseline, {"runs": []}, **options)
            def predict_checked(current, previous, index, previous_index):
                with _backend_import(backends):
                    return predictor(current, previous)
            return predict_checked, predictor, {"backend": backend,
                "materializer": _asset(checked_source), "backend_sha256": provenance["backend_sha256"],
                "backend_source": _asset(backends.__file__),
                "verify_numerics": provenance.get("verify_numerics", True)}
        raise ValueError("Frozen checked CUDA materializer snapshot differs or is unavailable")
    wrapper = validation / wrapper_info["wrapper_snapshot"]
    if _sha(wrapper) != wrapper_info["wrapper_sha256"]:
        raise ValueError("Frozen transient CUDA wrapper changed")
    backends = _load_frozen_backend(validation, provenance["backend_sha256"])
    with _backend_import(backends):
        module = load_snapshot_module("reference_frozen_cuda_wrapper", wrapper)
        predictor = module.TransientCudaFeaturePredictor(model, baseline)
    def predict(current, previous, index, previous_index):
        import numpy as np
        with _backend_import(backends):
            result = predictor.predict(current, previous, current_key=index, previous_key=previous_index)
        return backends.partition(np.stack([result["co2_presence_probability"], result["gas_probability_conditional_on_co2"]], axis=-1))
    return predict, predictor, {"backend": backend, "wrapper": _asset(wrapper),
        "backend_sha256": provenance["backend_sha256"], "backend_source": _asset(backends.__file__)}


def _apply(sequence, prediction, elapsed, current):
    if sequence is None:
        return prediction, None
    if hasattr(sequence, "apply_with_reference"):
        return sequence.apply_with_reference(prediction, elapsed, current)
    return sequence.apply(prediction, elapsed)


def _assert_phase_parity(prediction, path, box):
    import numpy as np
    from PIL import Image
    from darsia.fluidflower.phase_validation import _phase_masks, PHASE_FILL_COLORS
    x, y, w, h = box
    with Image.open(path) as im:
        original = np.asarray(im.convert("RGBA"))[y:y+h, x:x+w].copy()
    masks = _phase_masks(prediction)
    differences = {}
    for phase in ("gas", "aqueous"):
        expected = np.all(original == PHASE_FILL_COLORS[phase], axis=-1)
        differences[phase] = int(np.count_nonzero(expected != masks[phase]))
    if any(differences.values()):
        raise ValueError(f"Original reviewed history phase replay differs: {differences}; {path}")
    return differences


def compute_request(plan):
    """Worker entry point. Publish a receipt only after complete validation."""
    if plan.get("format") != FORMAT or plan.get("policy") != POLICY or plan.get("method_sha256") != _sha(__file__):
        raise ValueError("Reference worker method does not match its request")
    if _digest({k: v for k, v in plan.items() if k != "key"}) != plan["key"]:
        raise ValueError("Reference request identity changed")
    folder = cache_root() / plan["key"]
    with _worker_lock(folder / "worker.lock"):
        _check_assets(plan)
        if (folder / "receipt.json").is_file():
            return _read(folder / "receipt.json")
        compatible = _reuse_compatible_receipt(plan)
        if compatible is not None:
            return compatible
        if plan.get("reviewed_layers", {}).get("all"):
            layers = plan["reviewed_layers"]
            for asset in layers.values():
                if _sha(asset["path"]) != asset["sha256"]:
                    raise ValueError("Existing exact reviewed map changed")
            receipt = {**plan, "mode": "exact-reviewed-image-reuse", "layers": _display_layers(layers), "replay_validation": "original validation maps; no fresh inference"}
            _write(folder / "receipt.json", receipt)
            return receipt
        # A transfer validation may have evaluated this off-timeline endpoint
        # while the canonical reviewed sequence was already in memory. Accept
        # it only for the same photographed acquisition and causal predecessor.
        single_pass = (Path(plan["binding"]["path"]).parent / "single_pass_references"
                       / plan["run"] / plan["reference"]["id"] / "single_pass.json")
        if single_pass.is_file():
            record = _read(single_pass)
            method = (plan.get("inference_provenance") or {}).get("single_pass_reference_sha256")
            snapshot = Path(plan["binding"]["path"]).parent / "code_snapshot" / "phase_reference_single_pass.py"
            if (method is None or record.get("method_sha256") != method
                or not snapshot.is_file() or _sha(snapshot) != method):
                raise ValueError("Single-pass reference method differs from frozen validation")
            excluded = set(plan["job"].get("independent_reference_frames") or [])
            prior = [row["frame_index"] for row in plan["replay"]
                     if row["frame_index"] not in excluded]
            expected_previous = prior[-1] if prior else None
            if (record.get("id") != plan["reference"]["id"]
                or record.get("elapsed_h") != plan["reference"]["phase_time_h"]
                or record.get("photo", {}).get("sha256") != plan["reference"]["photo"]["sha256"]
                or record.get("previous_reviewed_frame_index") != expected_previous
                or record.get("canonical_history_unchanged") is not True):
                raise ValueError("Single-pass reference does not match the exact causal request")
            generated = {**plan["job"],
                         "phase_map_output_dir": record["final_root"],
                         "raw_phase_map_output_dir": record["raw_root"],
                         "presence_diagnostics_dir": record["diagnostics_root"]}
            layers = _original_layers(generated, 0)
            _check_assets(plan)
            receipt = {**plan, "mode": "single-pass-exact-reference",
                       "layers": _display_layers(layers),
                       "replay_validation": "causal state from the same reviewed validation pass",
                       "single_pass_provenance": _asset(single_pass),
                       "previous_reviewed_frame_index": expected_previous,
                       "canonical_history_unchanged": True,
                       "canonical_history_sha256": record["canonical_history_sha256"],
                       "physical_report": record.get("physical_report")}
            _write(folder / "receipt.json", receipt)
            return receipt
        if "decision_cache_adapter_sha256" in plan:
            from darsia.fluidflower.tools.phase_decision_reference_cache import derive
            if _sha(tool_path("phase_decision_reference_cache.py")) != plan["decision_cache_adapter_sha256"]:
                raise ValueError("Reference evidence adapter changed")
            if _sha(tool_path("materialize_phase_decision_revision.py")) != plan["decision_evidence_reader_sha256"]:
                raise ValueError("Reference evidence reader changed")
            derived = derive(plan, folder)
            if derived is not None:
                return derived
        import numpy as np
        from PIL import Image
        from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
        from darsia.fluidflower.phase_runtime import load_physics_profile, load_run_context, physics_runtime_identity, phase_physics_sequence
        from darsia.fluidflower.phase_validation import _model_identity, _write_phase_maps
        from darsia.fluidflower.tools.concentration_movie_compute import fork_sequence, history_digest

        model_root = Path(plan["model_root"])
        if _model_identity(model_root) != (plan["model_id"], plan["model_sha256"]):
            raise ValueError("Frozen model changed before reference inference")
        profile = load_physics_profile(model_root)
        if physics_runtime_identity(profile) != plan["physics_runtime"]:
            raise ValueError("Frozen physics changed before reference inference")
        model = HierarchicalPhaseModel.load(model_root, adapter="generic")
        x, y, w, h = plan["content_box_xywh"]
        canvas_shape = (plan["canvas_wh"][1], plan["canvas_wh"][0])
        def rgb(asset):
            if _sha(asset["path"]) != asset["sha256"]:
                raise ValueError("Exact reference photograph changed before decode")
            with Image.open(asset["path"]) as im:
                if im.size != tuple(plan["canvas_wh"]):
                    raise ValueError("Reference image canvas differs from its native geometry")
                return np.asarray(im.convert("RGB"))[y:y+h, x:x+w].copy()
        baseline = rgb(plan["baseline"])
        sequence, context_entry = None, None
        if profile is not None:
            context, context_entry = load_run_context(model_root, profile, plan["dataset"], plan["run"], native_manifest_sha256=plan["native_manifest"]["sha256"])
            if context.domain.shape != (h, w):
                raise ValueError("Reference physics context has different geometry")
            sequence = phase_physics_sequence(context, profile)
            if hasattr(sequence, "bind_reference"):
                sequence.bind_reference(baseline)
        elif plan["job"].get("domain_mask"):
            raise ValueError("Standalone domain-masked reference predictions are not supported")
        predict, predictor, backend = _predictor(plan, model, baseline)
        previous, previous_index = None, None
        checks = []
        try:
            for row in plan["replay"]:
                index = row["frame_index"]
                print(f"{plan['run']} reference {plan['reference']['id']}: replay {index} at {row['elapsed_h']:.9g} h", flush=True)
                current = rgb(row["photo"])
                predictions = predict(current, previous, ("reviewed", index), previous_index)
                predictions, _ = _apply(sequence, predictions, row["elapsed_h"], current)
                original = _original_layers(plan["job"], index).get("all")
                if original is None:
                    raise ValueError("Original phase map needed for causal replay parity is missing")
                differences = _assert_phase_parity(predictions, original["path"], plan["content_box_xywh"])
                checks.append({"frame_index": index, "elapsed_h": row["elapsed_h"], "differences": differences, "original_map": original})
                if index not in (plan["job"].get("independent_reference_frames") or []):
                    previous, previous_index = current, ("reviewed", index)
                del predictions
            current = rgb(plan["reference"]["photo"])
            raw = predict(current, previous, ("reference", plan["reference"]["id"]), previous_index)
            before = None if sequence is None else history_digest(sequence)
            disposable = None if sequence is None else fork_sequence(sequence)
            predictions, report = _apply(disposable, raw, plan["reference"]["phase_time_h"], current)
            if sequence is not None and history_digest(sequence) != before:
                raise ValueError("Disposable reference prediction changed canonical reviewed history")
            review_outputs = plan["job"].get("output_mode", "review") == "review"
            _write_phase_maps(folder / "final", plan["run"], plan["pipeline_sha256"], 0, canvas_shape, (x, y, w, h), predictions)
            if review_outputs:
                _write_phase_maps(folder / "raw", plan["run"], plan["model_sha256"], 0, canvas_shape, (x, y, w, h), raw)
            if disposable is not None and hasattr(disposable, "write_diagnostics"):
                if review_outputs:
                    disposable.write_diagnostics(folder / "diagnostics", plan["run"], 0, canvas_shape, (x, y, w, h))
                else:
                    from darsia.fluidflower.phase_validation import _write_diagnostic_evidence
                    _write_diagnostic_evidence(folder / "diagnostics", plan["run"], 0, disposable)
            generated_job = {**plan["job"], "phase_map_output_dir": str(folder / "final"),
                "raw_phase_map_output_dir": str(folder / "raw") if review_outputs else None,
                "presence_diagnostics_dir": str(folder / "diagnostics")}
            layers = _original_layers(generated_job, 0)
            _check_assets(plan)
            receipt = {**plan, "mode": "fresh-exact-reference", "layers": _display_layers(layers), "replay_validation": checks,
                       "previous_reviewed_frame_index": None if previous_index is None else previous_index[1],
                       "canonical_history_unchanged": True, "canonical_history_sha256": before,
                       "physical_report": report, "context_entry": context_entry, "execution": backend}
            _write(folder / "receipt.json", receipt)
            return receipt
        finally:
            if predictor is not None:
                predictor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    receipt = compute_request(_read(args.request))
    print(json.dumps({"reference": receipt["reference"]["id"], "key": receipt["key"], "mode": receipt["mode"], "receipt": str(cache_root() / receipt["key"] / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()

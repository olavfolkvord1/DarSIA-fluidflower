"""Prepare a new review candidate from immutable corrected evidence.

Numerical CPU-reference checks are available through explicit audit mode. Normal
production uses the same optical calculation without duplicate reference work.
"""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from darsia.fluidflower.phase_physics import freeze_run_context
from darsia.fluidflower.phase_runtime import load_physics_profile, load_run_context
from darsia.fluidflower.phase_support import CEILING_VERSION
from darsia.fluidflower.phase_validation import (
    _atomic_json, _candidate_paths, _source_jobs, predict_image_tiled,
    materialize_phase_validation_campaign,
)
from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor, partition


def plot_visible_phase_area(path, times, counts, domain_pixels, run):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 5))
    values = [np.array([row[k] for row in counts]) * 100 / domain_pixels
              for k in ("aqueous", "gas")]
    ax.stackplot(times, *values, colors=["#31d9a0", "#f24c9d"], labels=["Aqueous", "Free gas"])
    ax.set(xlabel="Time since first injection [h]", ylabel="Visible porous area [%]",
           title=f"{run.upper()} predicted 2D phase area - not concentration or mass")
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def freeze_ceiling(model, parent, source):
    target = model / "phase_physics.json"
    if target.exists():
        raise FileExistsError(target)
    old = load_physics_profile(parent)
    profile = copy.deepcopy(old)
    profile["runs"] = {}
    for job in _source_jobs(source):
        run, dataset = job["run"], job["dataset"]
        entry = old["runs"][f"{dataset}/{run}"]
        context, _ = load_run_context(parent, old, dataset, run,
                                     native_manifest_sha256=entry["native_manifest_sha256"])
        _, review = _candidate_paths(source, job)
        review = analysis_review_root(Path(job.get("analysis_source_review_root") or review), run)
        native = review / "native_frames" / run / "manifest.json"
        info = json.loads(native.read_text())
        if tuple(info["content_box_xywh"][2:][::-1]) != context.domain.shape:
            raise ValueError("Inherited geometry does not match current image grid")
        profile["runs"][f"{dataset}/{run}"] = freeze_run_context(
            model, dataset, run, context, native_manifest=native,
            provenance={"parent_model": str(parent), "parent_context": entry,
                        "change": "same registered grid and protocol; updated source timeline identity"})
    profile["post_injection_ceiling"] = {"format": CEILING_VERSION, "gas_score_cap": .49}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2)
    from darsia.fluidflower import phase_physics, phase_support
    snapshot = model / "physics_context" / "code_snapshot"
    snapshot.mkdir()
    for module in (phase_physics, phase_support):
        path = Path(module.__file__)
        shutil.copy2(path, snapshot / path.name)


PARITY_REFERENCE_THREADS_ENV = "PHASE_PARITY_REFERENCE_THREADS"


def parity_reference_threads():
    """Threads for the CPU reference of a parity check; 1 = the single-threaded strips as before."""
    value = os.environ.get(PARITY_REFERENCE_THREADS_ENV, "").strip()
    if value:
        return max(1, int(value))
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def cpu_parity_reference(model, current, baseline, previous):
    """The values of ``predict_image_tiled`` for the CUDA parity check, and how they were obtained.

    Measured 2026-09-17: the strips take 28-35 s on one thread for a 37 MP frame, three times per
    experiment, while the shared GPU worker serves nothing else (about 7 of the first minutes of a
    four-experiment batch). Exact tiles run the same model code on several threads. They are used
    only if they agree bit for bit with full-width reference strips of this very frame; any
    difference or error falls back to the strips, so the check never compares against other values.
    """
    started = time.perf_counter()
    threads, reason = parity_reference_threads(), "one thread requested"
    if threads > 1:
        try:
            from darsia.fluidflower.tools.phase_optical_exact_tiles import ExactTilePredictor, verify_against_reference_strip
            predictor = ExactTilePredictor(model, baseline, threads=threads)
            previous_threads = cv2.getNumThreads()
            try:
                cv2.setNumThreads(1)
                with threadpool_limits(limits=1):
                    outputs, evaluated, _ = predictor.predict(current, previous)
                    height = predictor.shape[0]
                    strips = [verify_against_reference_strip(predictor, current, previous, outputs, evaluated, row=row)
                              for row in sorted({height // 2, max(0, height - 64)})]
            finally:
                cv2.setNumThreads(previous_threads)
            if all(strip["exact"] and strip["compared_pixels"] > 0 for strip in strips):
                return outputs, dict(method="exact-tiles", threads=threads, tile=predictor.tile, halo=predictor.halo,
                    strips_compared_bit_for_bit=[strip["rows"] for strip in strips],
                    seconds=time.perf_counter() - started)
            reason = "exact tiles differed from a reference strip"
        except Exception as error:  # the strips below are the reference in every case
            reason = f"{type(error).__name__}: {error}"
    reference = predict_image_tiled(model, current, baseline, previous_rgb=previous)
    return reference, dict(method="strips", threads=1, fallback_reason=reason, seconds=time.perf_counter() - started)


class CheckedCuda:
    def __init__(self, model, baseline, report, support_audit=None, parity_parent=None,
                 *, optical_cache_root=None, optical_cache_max_bytes=None, verify_numerics=False):
        from darsia.fluidflower.tools.phase_optical_cache import create_cache
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor as expected_gpu
        self.model, self.baseline, self.report = model, baseline, report
        self.support_audit = support_audit
        self.parity_parent = parity_parent
        self.verify_numerics = bool(verify_numerics)
        self._gpu = None
        self.frame = 0
        self._parity_cache = {}
        self.cache, reason = ((None, "Nonstandard predictor") if CudaFeaturePredictor is not expected_gpu else
            create_cache(model, baseline, "cuda", root=optical_cache_root, max_bytes=optical_cache_max_bytes))
        self.run_report = self._new_report()
        if reason:
            self.run_report["optical_cache_unavailable"] = reason
        report["runs"].append(self.run_report)

    def _new_report(self):
        report = {"shape_hw": list(self.baseline.shape[:2]), "frames": [], "parity": [],
                  "backend": "cuda-float32-checked" if self.verify_numerics else "cuda-float32",
                  "verify_numerics": self.verify_numerics}
        if self.cache is not None:
            report["optical_cache"] = self.cache.stats
        return report

    @property
    def gpu(self):
        # A wholly cached timeline performs no GPU feature calculation or upload.
        if self._gpu is None:
            self._gpu = CudaFeaturePredictor(self.model, self.baseline)
        return self._gpu

    @gpu.setter
    def gpu(self, value):
        self._gpu = value

    def __call__(self, current, previous):
        from darsia.fluidflower.tools.phase_optical_cache import array_digest, json_digest, verified_parity
        start = time.perf_counter()
        key = self.cache.key(current, previous) if self.cache is not None else None
        loaded = self.cache.load(key) if self.cache is not None else None
        cached_check = None
        if loaded is not None:
            prediction, receipt = loaded
            cached_check = verified_parity(receipt)
            # A skipped frame has no GPU feature cache. An ensuing miss must
            # reconstruct its actual predecessor, never reuse an older frame.
            if self._gpu is not None:
                self._gpu.previous = self._gpu.previous_key = None
            cuda_seconds = 0.0
        else:
            gpu_started = time.perf_counter()
            prediction = self.gpu.predict(current, previous, current_key=self.frame, previous_key=self.frame-1)
            cuda_seconds = time.perf_counter()-gpu_started
        self.run_report["frames"].append(dict(frame=self.frame, cuda_seconds=cuda_seconds,
            optical_cache_hit=loaded is not None, optical_cache_key=key,
            optical_seconds=time.perf_counter()-start))
        if self.verify_numerics and self.parity_parent is not None:
            root, job = self.parity_parent
            from darsia.fluidflower.tools.verify_local_floor_candidate import masks, read
            meta = read(Path(job['review_output_dir']) / 'native_frames' / job['run'] / 'manifest.json')
            old_c, old_g = masks(root, job, self.frame, meta['content_box_xywh'], raw=True)
            c = prediction['co2_presence_probability'] >= .5
            g = c & (prediction['gas_probability_conditional_on_co2'] >= .6)
            if not np.array_equal(c, old_c) or not np.array_equal(g, old_g):
                raise ValueError(f'Frozen optical-label replay differs: {job["run"]}/{self.frame}')
            # Matching old threshold masks is useful replay evidence, but does
            # not prove numerical parity for changed pixels/runtime/environment.
            self.run_report['parity'].append(dict(frame=self.frame,
                raw_optical_labels_exact=True, parent=str(root), inherited_cpu_parity=False))
        check = cached_check
        if self.verify_numerics and self.frame in (0, 1, 10):
            parity_key = json_digest([array_digest(array) for array in
                (current, previous, *[prediction[name] for name in sorted(prediction)])])
            cached = cached_check or self._parity_cache.get(parity_key)
            if cached is not None:
                check = dict(cached, frame=self.frame, reused_exact_inputs_and_outputs=True,
                    persistent_optical_receipt=loaded is not None,
                    cpu_reference=dict(cached['cpu_reference'], seconds=0.0,
                        original_check_seconds=cached['cpu_reference'].get('original_check_seconds',
                            cached['cpu_reference']['seconds'])))
            else:
                reference, how = cpu_parity_reference(self.model, current, self.baseline, previous)
                check = {"frame": self.frame, "heads": {}, "cpu_reference": how}
                for name, values in prediction.items():
                    error = np.abs(reference[name] - values)
                    threshold = .6 if name.startswith("gas_") else .5
                    disagreement = float(np.mean((values >= threshold) != (reference[name] >= threshold)))
                    q99 = np.quantile(error, .99)
                    check["heads"][name] = {"q99_abs_error": float(q99),
                        "max_abs_error": float(error.max()), "disagreement": disagreement}
                    if not np.isfinite(error).all() or q99 > 1e-3 or disagreement > 1e-4:
                        if self.cache is not None:
                            self.cache.invalidate(key)
                        raise ValueError(f"CPU/CUDA parity failed: {check}")
            self.run_report["parity"].append(check)
            if len(self._parity_cache) >= 8:
                self._parity_cache.pop(next(iter(self._parity_cache)))
            self._parity_cache[parity_key] = check
        if self.cache is not None and (loaded is None or (cached_check is None and check is not None)):
            self.cache.store(key, prediction, check)
        result = partition(np.stack([prediction["co2_presence_probability"],
                                     prediction["gas_probability_conditional_on_co2"]], axis=-1))
        if self.support_audit is not None:
            self.support_audit.observe(result, self.frame)
        self.frame += 1
        return result

    def predict_reference(self, current, previous, report):
        """Reuse baseline/storage without advancing reviewed-frame history."""
        gpu = self._gpu
        saved = (self.frame, self.run_report,
                 None if gpu is None else gpu.previous, None if gpu is None else gpu.previous_key,
                 self.support_audit, self.parity_parent)
        self.frame = 0
        self.run_report = self._new_report()
        report.setdefault('runs', []).append(self.run_report)
        self.support_audit = self.parity_parent = None
        if gpu is not None:
            gpu.previous_key = None
        try:
            return self(current, previous)
        finally:
            (self.frame, self.run_report, previous_features, previous_key,
             self.support_audit, self.parity_parent) = saved
            if self._gpu is not None:
                self._gpu.previous, self._gpu.previous_key = previous_features, previous_key

    def close(self):
        if self._gpu is not None:
            import cupy as cp
            self._gpu = None
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()


def relocate(value, old, new):
    if isinstance(value, dict):
        return {k: relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old, new) for v in value]
    if isinstance(value, str) and (value == str(old) or value.startswith(str(old) + "\\")):
        return str(new) + value[len(str(old)):]
    return value


def verify_optical_parity_parent(model, source, parent, backend):
    """Reuse parity only for byte-identical weights and the same source revision."""
    if sha256_file(source / 'postflight_manifest.json') != json.loads(
            (parent / 'validation/validation_binding.json').read_text())['source_manifest_sha256']:
        raise ValueError('Optical parity source revision differs')
    for relative in [p.relative_to(model) for p in (model / 'models').rglob('*') if p.is_file()]:
        if not (parent / relative).is_file() or sha256_file(model / relative) != sha256_file(parent / relative):
            raise ValueError(f'Optical parity weights differ: {relative}')
    for name in ('phase_decision_policy.json',):
        if sha256_file(model / name) != sha256_file(parent / name):
            raise ValueError('Optical parity decision policy differs')
    binding = json.loads((parent / 'validation/validation_binding.json').read_text())
    provenance = binding['inference_provenance']
    if provenance['backend_sha256'] != sha256_file(backend) or provenance.get('cpu_reference_frames') != [0, 1, 10]:
        raise ValueError('Optical parity backend or CPU checks differ')
    previous = json.loads((parent / 'validation/inference_report.json').read_text())
    jobs = _source_jobs(parent / 'validation')
    if len(previous['runs']) != len(jobs) or any(
            {r['frame'] for r in run['parity']} != {0, 1, 10} for run in previous['runs']):
        raise ValueError('Parent CPU parity is incomplete')
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source", "work"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--physics-parent", type=Path)
    parser.add_argument("--audit-front-memory", action="store_true")
    parser.add_argument("--reuse-optical-parity", type=Path)
    parser.add_argument("--verify-numerics", action="store_true",
                        help="Audit CUDA outputs against CPU references; adds duplicate inference work")
    args = parser.parse_args()
    if args.reuse_optical_parity and not args.verify_numerics:
        parser.error("--reuse-optical-parity requires --verify-numerics")
    model, source, work = args.model.resolve(), args.source.resolve(), args.work.resolve()
    published = model / "validation"
    if published.exists() or work.exists():
        raise FileExistsError("Use a new candidate and empty staging directory")
    if args.physics_parent:
        freeze_ceiling(model, args.physics_parent.resolve(), source)
    model_hashes = {str(p.relative_to(model)): sha256_file(p) for p in
                    [model / "summary.json", *sorted((model / "models/generic").glob("*.json"))]}
    report = {"created_utc": utc_now(), "runs": [], "model_hashes": model_hashes,
              "training_seen_review": True, "independent_validation": False}
    code = Path(__file__).resolve()
    backend = code.with_name("phase_inference_benchmark_backends.py")
    import cupy as cp
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
    provenance = {"backend": "cuda-float32-cached-features-checked" if args.verify_numerics
                  else "cuda-float32-cached-features", "device": device,
                  "materializer_sha256": sha256_file(code), "backend_sha256": sha256_file(backend),
                  "verify_numerics": args.verify_numerics,
                  "cpu_reference_frames": [0, 1, 10] if args.verify_numerics else []}
    if args.verify_numerics:
        provenance.update(q99_tolerance=.001, label_disagreement_tolerance=.0001)
    parity_jobs = None
    if args.reuse_optical_parity:
        parity_jobs = iter(verify_optical_parity_parent(model, source, args.reuse_optical_parity, backend))
        provenance.update(cpu_reference_frames=[], inherited_cpu_reference_frames=[0, 1, 10],
            cpu_parity_parent=str(args.reuse_optical_parity), raw_optical_labels_checked_every_frame=True)
    cv2.setNumThreads(4)
    support_audit = None
    if args.audit_front_memory:
        from darsia.fluidflower.tools.phase_support_training_audit import SupportAudit
        source_jobs = _source_jobs(source)
        support_audit = SupportAudit(model, source_jobs, work / "temporal_evaluation")
        pending_jobs = iter(source_jobs)

    def predictor_factory(selected_model, baseline):
        if support_audit is not None:
            support_audit.begin(next(pending_jobs), baseline)
        parent_job = (args.reuse_optical_parity, next(parity_jobs)) if parity_jobs is not None else None
        return CheckedCuda(selected_model, baseline, report, support_audit, parity_parent=parent_job,
                           verify_numerics=args.verify_numerics)

    with threadpool_limits(limits=4):
        materialize_phase_validation_campaign(model, source, work,
            progress=lambda s: print(s, flush=True), inference_provenance=provenance,
            predictor_factory=predictor_factory)
    if support_audit is not None:
        support_audit.finish()
    manifest = json.loads((work / "postflight_manifest.json").read_text())
    # Replace compatibility media with actual new phase areas, never old mass fits.
    for job in manifest["jobs"]:
        run = job["run"]
        native = json.loads((Path(job["review_output_dir"]) / "native_frames" / run / "manifest.json").read_text())
        candidate = work / "postflight" / run
        candidate.mkdir(parents=True)
        counts = []
        movie = cv2.VideoWriter(str(candidate / "calibration_review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2., (1280, 720))
        if not movie.isOpened():
            raise RuntimeError("Cannot open review movie writer")
        for frame in range(native["frame_count"]):
            path = work / "review_assets/native_phase_maps" / run / f"A_{job['params_sha256'][:16]}" / f"frame_{frame:03d}.png"
            with Image.open(path) as layer:
                rgba = np.asarray(layer)
                counts.append({"aqueous": int(((rgba[..., 0] == 49) & (rgba[..., 3] > 0)).sum()),
                               "gas": int(((rgba[..., 0] == 242) & (rgba[..., 3] > 0)).sum()), "mixed": 0})
                if support_audit is not None:
                    checked = support_audit.report["runs"][run][frame]
                    if (counts[-1]["gas"] != checked["old_gas_pixels"] or
                            counts[-1]["gas"] + counts[-1]["aqueous"] != checked["co2_pixels"]):
                        raise ValueError("Paired audit does not match the materialized base support")
                with Image.open(Path(job["review_output_dir"]) / "native_frames" / run / f"frame_{frame:03d}.jpg") as photo:
                    image = Image.alpha_composite(photo.convert("RGBA"), layer).convert("RGB")
                    image.thumbnail((1280, 720))
                    canvas = Image.new("RGB", (1280, 720))
                    canvas.paste(image, ((1280-image.width)//2, (720-image.height)//2))
                    movie.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        movie.release()
        times = [row["elapsed_h"] for row in native["frame_sources"]]
        x, y, w, h = native["content_box_xywh"]
        physics = load_physics_profile(model)
        if physics:
            entry = physics["runs"][f"{job['dataset']}/{run}"]
            context, _ = load_run_context(model, physics, job["dataset"], run,
                native_manifest_sha256=entry["native_manifest_sha256"])
            domain_pixels = int(context.domain.sum())
        elif native.get("domain", {}).get("path"):
            with np.load(resolve_review_path(native["domain"]["path"]), allow_pickle=False) as archive:
                domain_pixels = int(archive["domain"].sum())
        else:
            domain_pixels = w*h
        plot_visible_phase_area(candidate / "calibration_mass_fit.png", times, counts, domain_pixels, run)
        with (candidate / "calibration_review_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["elapsed_h", "aqueous_pixels", "gas_pixels", "mixed_pixels"])
            writer.writeheader()
            for t, values in zip(times, counts):
                writer.writerow({"elapsed_h": t, **{k+"_pixels": v for k,v in values.items()}})
        _atomic_json(candidate / "summary.json", {"selection_candidate_id": job["candidate_id"],
            "review_plot_title": f"{run.upper()} visible 2D phase areas; no mass analysis",
            "calibration_review_movie": {"params_sha256": job["params_sha256"], "fps": 2,
                "calibration_frame_count": native["frame_count"], "plot_seconds": 0, "image_seconds_per_calibration_frame": .5}})
        job["candidate_folder"] = str(candidate)
        job["quality_flags"] = {"dataset": job["dataset"], "training_seen_review": True,
                                "independent_holdout": False, "human_review_required": True}
    _atomic_json(work / "postflight_manifest.json", manifest)
    report["completed_utc"] = utc_now()
    _atomic_json(work / "inference_report.json", report)
    snapshot = work / "code_snapshot"
    snapshot.mkdir()
    for path in (code, backend):
        shutil.copy2(path, snapshot / path.name)
    for path in work.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        _atomic_json(path, relocate(value, work, published))
    for name, digest in model_hashes.items():
        if sha256_file(model / name) != digest:
            raise ValueError("Model changed during materialization")
    # Some SMB servers retain the Hidden attribute when a dot-prefixed staging
    # directory is renamed. Keep the final validation folder visible in Explorer.
    pending = model / f"validation-publishing-{uuid.uuid4().hex}"
    shutil.copytree(work, pending)
    for path in work.rglob("*"):
        if path.is_file() and sha256_file(path) != sha256_file(pending / path.relative_to(work)):
            raise ValueError(f"Publication checksum mismatch: {path}")
    pending.rename(published)
    print(f"PUBLISHED_AND_VERIFIED {published}", flush=True)


if __name__ == "__main__":
    main()

"""Audit alternative physical support while review predictions are already in RAM."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np

from darsia.fluidflower import phase_temporal
from darsia.fluidflower.phase_physics import load_physics_profile, load_run_context
from darsia.fluidflower.phase_training.review_regions import annotation_mask
from darsia.fluidflower.phase_validation import _model_identity, _atomic_json
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.evaluate_phase_temporal import read, review_regions


class SupportAudit:
    def __init__(self, model, source_jobs, output):
        self.model, self.jobs, self.output = model, source_jobs, output
        self.output.mkdir(parents=True, exist_ok=False)
        self.old_profile = load_physics_profile(model)
        if self.old_profile is None:
            raise ValueError("Paired support requires registered physical contexts")
        self.new_profile = {**self.old_profile, "aqueous_front_memory": {
            "format": phase_temporal.FRONT_VERSION,
            **asdict(phase_temporal.AqueousFrontSettings())}}
        self.regions = review_regions(model, source_jobs)
        snapshot = output / "code_snapshot"
        snapshot.mkdir()
        code = [Path(__file__), Path(phase_temporal.__file__)]
        for path in code:
            shutil.copy2(path, snapshot / path.name)
        protected = [model / "summary.json", model / "phase_physics.json",
                     *sorted((model / "models").rglob("*.json"))]
        self.report = {
            "parent": str(model), "parent_identity": _model_identity(model)[1],
            "created_utc": utc_now(), "runs": {}, "regions": [],
            "new_front_settings": self.new_profile["aqueous_front_memory"],
            "protected_sources": {str(p): sha256_file(p) for p in protected},
            "code_sha256": {p.name: sha256_file(p) for p in code},
            "selected_runs": [job["run"] for job in source_jobs],
            "independent_reference_frames": {}, "weights_retrained": False,
            "scope": "Paired support on the same newly trained optical weights and training evidence; not independent accuracy",
            "inference_reused_from_materialization": True,
        }
        self.active = None

    def begin(self, job, baseline):
        run, dataset = job["run"], job["dataset"]
        root = Path(job["analysis_source_review_root"])
        native_path = root / "native_frames" / run / "manifest.json"
        self.native = read(native_path)
        self.context, _ = load_run_context(self.model, self.old_profile, dataset, run,
                                          native_manifest_sha256=sha256_file(native_path))
        if baseline.shape[:2] != self.context.domain.shape:
            raise ValueError("Audit grid differs from the GPU input grid")
        self.old = phase_temporal.phase_physics_sequence(self.context, self.old_profile)
        self.new = phase_temporal.phase_physics_sequence(self.context, self.new_profile)
        self.active = run
        self.report["runs"][run] = []
        self.report["independent_reference_frames"][run] = job.get("independent_reference_frames") or []

    def observe(self, predictions, index):
        run = self.active
        elapsed = float(self.native["frame_sources"][index]["elapsed_h"])
        old, _ = self.old.apply(predictions, elapsed)
        new, info = self.new.apply(predictions, elapsed)
        if not np.array_equal(old["co2_presence_probability"], new["co2_presence_probability"]):
            raise ValueError("Temporal support changed total CO2 presence")
        co2 = old["co2_presence_probability"] >= .5
        old_gas = co2 & (old["gas_probability_conditional_on_co2"] >= .6)
        new_gas = co2 & (new["gas_probability_conditional_on_co2"] >= .6)
        if np.any(new_gas & ~old_gas):
            raise ValueError("Temporal support introduced gas")
        self.report["runs"][run].append({
            "frame": index, "elapsed_h": elapsed, "old_gas_pixels": int(old_gas.sum()),
            "new_gas_pixels": int(new_gas.sum()), "co2_pixels": int(co2.sum()),
            "gas_added_pixels": 0, "physics": info})
        for region in self.regions[(run, index)]:
            mask = annotation_mask(region.payload, canvas_wh=self.native["canvas_wh"],
                                   content_box=self.native["content_box_xywh"])
            if mask.any():
                self.report["regions"].append({
                    "region_id": region.region_id, "run": run, "frame": index,
                    "elapsed_h": elapsed, "class": region.phase_class,
                    "old_gas_fraction": float(old_gas[mask].mean()),
                    "new_gas_fraction": float(new_gas[mask].mean()),
                    "co2_fraction": float(co2[mask].mean())})
        _atomic_json(self.output / "report.partial.json", self.report)

    def finish(self):
        for job in self.jobs:
            run = job["run"]
            native = read(Path(job["analysis_source_review_root"]) / "native_frames" / run / "manifest.json")
            if len(self.report["runs"].get(run, [])) != native["frame_count"]:
                raise ValueError(f"Incomplete paired support evaluation for {run}")
        grouped = defaultdict(list)
        for row in self.report["regions"]:
            grouped[f"{row['run']}/{row['class']}"].append(row)
        self.report["region_summary"] = {key: {
            "regions": len(rows),
            "old_gas_fraction": float(np.mean([r["old_gas_fraction"] for r in rows])),
            "new_gas_fraction": float(np.mean([r["new_gas_fraction"] for r in rows])),
            "regions_losing_all_detected_gas": sum(r["old_gas_fraction"] > 0 and r["new_gas_fraction"] == 0 for r in rows),
        } for key, rows in grouped.items()}
        for path, digest in self.report["protected_sources"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"Model changed during paired evaluation: {path}")
        self.report["completed_utc"] = utc_now()
        _atomic_json(self.output / "report.json", self.report)

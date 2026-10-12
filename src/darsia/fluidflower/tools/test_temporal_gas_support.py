"""Read-only FF_AC pilot: replay a recent-gas rule on saved phase maps.

The input model and Studio binding are untouched. Results are written to an
explicit experiment folder, normally the series' DarSIA/results directory.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.phase_temporal_gas_support import (
    Settings, BuoyancySettings, apply_labels, apply_buoyancy_floor,
    prepare_buoyancy_floor,
)
from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS
from darsia.fluidflower.project import sha256_file, utc_now

COLORS = {1: PHASE_FILL_COLORS["aqueous"], 2: PHASE_FILL_COLORS["gas"]}
GAS_POSITIVE = {"gas", "gas_present_region", "mixed_gas_aqueous"}
GAS_NEGATIVE = {"aqueous", "water", "co2_absent_region",
                "current_experiment_co2_artifact", "gas_absent_region"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_labels(path, content_box, shape):
    with Image.open(path) as image:
        image = image.crop(tuple((content_box[0], content_box[1],
                                  content_box[0] + content_box[2],
                                  content_box[1] + content_box[3])))
        rgba = np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST).convert("RGBA"))
    labels = np.zeros(shape, np.uint8)
    known = rgba[..., 3] == 0
    for label, color in COLORS.items():
        hit = np.all(rgba == color, axis=-1)
        labels[hit] = label
        known |= hit
    if not known.all():
        raise ValueError(f"Unknown categorical phase-map color: {path}")
    return labels


def annotation_mask(payload, canvas_wh, box, shape):
    points = payload.get("points") or []
    if not points:
        x, y = float(payload["x"]), float(payload["y"])
        w, h = float(payload["width"]), float(payload["height"])
        points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    sx, sy = shape[1] / box[2], shape[0] / box[3]
    poly = np.array([[(float(x)*canvas_wh[0] - box[0])*sx,
                      (float(y)*canvas_wh[1] - box[1])*sy] for x, y in points])
    poly[:, 0] = np.clip(poly[:, 0], 0, shape[1]-1)
    poly[:, 1] = np.clip(poly[:, 1], 0, shape[0]-1)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
    return mask.astype(bool)


def render(path, original, candidate, *, run, frame, elapsed_h, mode):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    palette = np.array([[6, 31, 84], [54, 204, 154], [246, 92, 151]], np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
    for ax, labels, title in zip(axes, (original, candidate),
                                 ("Current optical phase model", f"{mode} pilot candidate")):
        ax.imshow(palette[labels], interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"{run.upper()} · frame {frame+1} · {elapsed_h:.2f} h  |  green aqueous, pink free gas")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=["ac17", "ac18", "ac31", "ac47", "ac60", "ac61"])
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--maximum-component-area-mm2", type=float, default=25.)
    parser.add_argument("--mode", choices=("small_recent", "buoyancy"), default="small_recent")
    args = parser.parse_args()
    if args.width < 512 or args.width > 4096:
        raise ValueError("Choose a review resolution between 512 and 4096 pixels")
    model = args.model.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    settings = Settings(maximum_component_area_mm2=args.maximum_component_area_mm2)
    profile = read(model / "phase_optical_regions.json")
    manifest = read(model / "validation/postflight_manifest.json")
    evidence = read(model / "evidence.json")
    by_run = defaultdict(lambda: defaultdict(list))
    for item in evidence["regions"]:
        if item["run"].casefold() in args.runs and item["phase_class"] in GAS_POSITIVE | GAS_NEGATIVE:
            by_run[item["run"]][int(item["frame_index"])].append(item)
    runs = []
    for job in manifest["jobs"]:
        run = job["run"]
        if run not in args.runs:
            continue
        source = Path(job.get("analysis_source_review_root") or job["review_output_dir"])
        meta_path = source / "native_frames" / run / "manifest.json"
        meta = read(meta_path)
        box = list(map(int, meta["content_box_xywh"]))
        canvas = list(map(int, meta["canvas_wh"]))
        shape = (round(box[3] * args.width / box[2]), args.width)
        entry = profile["runs"][f"ffac/{run}"]
        with np.load(model / entry["path"], allow_pickle=False) as context:
            spacing_mm = (abs(float(context["y_m"][1]-context["y_m"][0]))*1000*box[3]/shape[0],
                          abs(float(context["x_m"][1]-context["x_m"][0]))*1000*box[2]/shape[1])
            x_m = np.linspace(float(context["x_m"][0]), float(context["x_m"][-1]), shape[1])
            y_m = np.linspace(float(context["y_m"][0]), float(context["y_m"][-1]), shape[0])
            facies = cv2.resize(context["facies"].astype(np.float32), shape[::-1],
                                interpolation=cv2.INTER_NEAREST)
        stages = entry["injection_stages"]
        end_snapshots = [None] * len(stages)
        buoyancy_prepared = None
        injection_end = max(float(stage["end_h"]) for stage in entry["injection_stages"])
        csv_path = Path(job["per_frame_csv"])
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            times = [float(row["elapsed_h"]) for row in csv.DictReader(stream)]
        previous, previous_h = np.zeros(shape, np.uint8), -1e9
        frames, observations = [], []
        for frame, elapsed_h in enumerate(times):
            path = (Path(job["phase_map_output_dir"]) / "native_phase_maps" / run /
                    ("A_"+job["params_sha256"][:16]) / f"frame_{frame:03d}.png")
            labels = load_labels(path, box, shape)
            for i, stage in enumerate(stages):
                if (end_snapshots[i] is None and
                    float(stage["end_h"]) <= elapsed_h <=
                    float(stage["end_h"]) + BuoyancySettings().end_snapshot_max_delay_h):
                    end_snapshots[i] = labels == 2
                    buoyancy_prepared = None
            if args.mode == "buoyancy":
                if (buoyancy_prepared is None and
                    elapsed_h >= max(float(s["end_h"]) for s in stages) +
                    BuoyancySettings().post_injection_grace_h):
                    buoyancy_prepared = prepare_buoyancy_floor(
                        x_m, y_m, facies, stages, end_snapshots)
                candidate, audit = apply_buoyancy_floor(labels, elapsed_h=elapsed_h,
                    x_m=x_m, y_m=y_m, facies=facies, stages=stages,
                    end_snapshots=end_snapshots, prepared=buoyancy_prepared,
                    previous_h=previous_h if frame else None)
            elif frame:
                candidate, audit = apply_labels(labels, previous, elapsed_h, previous_h,
                    injection_end, spacing_mm, settings)
            else:
                candidate, audit = labels.copy(), dict(skipped="first_frame", reclassified_gas_pixels=0,
                                                       removed_components=0)
            current_gas, new_gas = labels == 2, candidate == 2
            frames.append({**audit, "frame": frame+1, "elapsed_h": elapsed_h,
                "gas_pixels": int(current_gas.sum()), "candidate_gas_pixels": int(new_gas.sum())})
            for item in by_run[run][frame]:
                mask = annotation_mask(item["payload"], canvas, box, shape)
                n = int(mask.sum())
                if not n:
                    continue
                before, after = int((mask & current_gas).sum()), int((mask & new_gas).sum())
                observations.append(dict(region_id=item["region_id"], frame=frame+1,
                    phase_class=item["phase_class"], box_pixels=n,
                    original_gas_pixels=before, candidate_gas_pixels=after,
                    positive_gas_lost=bool(item["phase_class"] in GAS_POSITIVE and before and not after),
                    negative_gas_reduced=bool(item["phase_class"] in GAS_NEGATIVE and after < before)))
            if (run, frame) in {("ac47", 13), ("ac31", 9), ("ac18", 20), ("ac60", 12)}:
                render(output / f"{run}_frame_{frame+1:02d}.png", labels, candidate,
                    run=run, frame=frame, elapsed_h=elapsed_h, mode=args.mode)
            previous, previous_h = candidate, elapsed_h
        runs.append(dict(run=run, frames=frames, observations=observations,
            total_reclassified_gas_pixels=sum(row["reclassified_gas_pixels"] for row in frames),
            positive_gas_regions_lost=sum(row["positive_gas_lost"] for row in observations),
            negative_gas_regions_improved=sum(row["negative_gas_reduced"] for row in observations)))
        print(f"{run}: {len(frames)} frames; removed {runs[-1]['total_reclassified_gas_pixels']} gas pixels; "
              f"positive boxes lost {runs[-1]['positive_gas_regions_lost']}", flush=True)
    report = dict(format="saved-ffac-recent-gas-pilot-v1", created_utc=utc_now(), mode=args.mode,
        source_model=str(model), source_profile_sha256=sha256_file(model / "phase_optical_regions.json"),
        source_evidence_sha256=sha256_file(model / "evidence.json"),
        settings=(settings.__dict__ if args.mode == "small_recent" else BuoyancySettings().__dict__),
        downsampled_review_width=args.width, training_or_model_modified=False,
        claims="Training-seen labels; no independent accuracy or full-resolution production claim",
        runs=runs)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

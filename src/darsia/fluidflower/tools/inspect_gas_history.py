"""Compare saved visible gas with its injection-end footprint on one experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from darsia.fluidflower.tools.test_temporal_gas_support import load_labels
from darsia.fluidflower.phase_temporal_gas_support import (
    BuoyancySettings, apply_buoyancy_floor, prepare_buoyancy_floor,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--frame", type=int, required=True, help="One-based target frame")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=2048)
    args = parser.parse_args()
    model = args.model.resolve()
    jobs = read(model / "validation/postflight_manifest.json")["jobs"]
    job = next(j for j in jobs if j["run"] == args.run)
    meta = read(Path(job.get("analysis_source_review_root") or job["review_output_dir"])
                / "native_frames" / args.run / "manifest.json")
    box = list(map(int, meta["content_box_xywh"]))
    shape = (round(box[3] * args.width / box[2]), args.width)
    profile = read(model / "phase_optical_regions.json")
    entry = profile["runs"][f"{job['dataset']}/{args.run}"]
    stages = entry["injection_stages"]
    with np.load(model / entry["path"], allow_pickle=False) as context:
        x = cv2.resize(context["x_m"].astype(np.float32)[None, :], (shape[1], 1))[0]
        y = cv2.resize(context["y_m"].astype(np.float32)[:, None], (1, shape[0]))[:, 0]
        facies = cv2.resize(context["facies"].astype(np.float32), shape[::-1],
                            interpolation=cv2.INTER_NEAREST)
    with Path(job["per_frame_csv"]).open(newline="", encoding="utf-8-sig") as stream:
        times = [float(row["elapsed_h"]) for row in csv.DictReader(stream)]
    current_index = args.frame - 1
    if not 0 <= current_index < len(times):
        raise ValueError("Target frame outside saved timeline")

    def labels(index):
        root = Path(job["phase_map_output_dir"]) / "native_phase_maps" / args.run
        path = root / ("A_" + job["params_sha256"][:16]) / f"frame_{index:03d}.png"
        return load_labels(path, box, shape)

    settings = BuoyancySettings()
    snapshot_indices = []
    for stage in stages:
        end_h = float(stage["end_h"])
        matches = [i for i, t in enumerate(times[:current_index+1])
                   if end_h <= t <= end_h + settings.end_snapshot_max_delay_h]
        snapshot_indices.append(matches[0] if matches else None)
    if snapshot_indices[0] is None:
        raise ValueError("No observed image close to injection-one end")
    snapshots = [None if index is None else labels(index) == 2
                 for index in snapshot_indices]
    at_end, current = labels(snapshot_indices[0]), labels(current_index)
    gas_current = current == 2
    prepared = prepare_buoyancy_floor(x, y, facies, stages, snapshots)
    candidate, audit = apply_buoyancy_floor(
        current, elapsed_h=times[current_index], x_m=x, y_m=y,
        facies=facies, stages=stages, end_snapshots=snapshots,
        settings=settings, prepared=prepared,
        previous_h=times[current_index-1] if current_index else None,
    )
    removed = gas_current & (candidate == 1)
    facies_edge = np.zeros(shape, bool)
    facies_edge[1:, :] |= facies[1:, :] != facies[:-1, :]
    facies_edge[:, 1:] |= facies[:, 1:] != facies[:, :-1]
    counts = dict(current_gas_pixels=int(gas_current.sum()),
        candidate_gas_pixels=int((candidate == 2).sum()),
        gas_reclassified_as_aqueous=int(removed.sum()),
        candidate_audit=audit,
        current_frame=args.frame, current_h=times[current_index],
        injection_end_snapshot_frames=[None if i is None else i+1 for i in snapshot_indices],
        region="full registered image; no injector ownership division")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")

    source = Path(job.get("analysis_source_review_root") or job["review_output_dir"])
    photo_path = source / "native_frames" / args.run / f"frame_{current_index:03d}.jpg"
    with Image.open(photo_path) as image:
        photo = np.asarray(image.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
                           .resize(shape[::-1], Image.Resampling.BILINEAR).convert("RGB"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    palette = np.array([[7, 29, 79], [52, 204, 156], [248, 83, 150]], np.uint8)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    for ax, rgb, title in zip(axes.ravel(),
        (photo, palette[at_end], palette[current], np.where(removed[..., None],
            np.array([255, 218, 0], np.uint8), palette[current])),
        (f"Corrected photo · {times[current_index]:.2f} h",
         f"Free gas at injection-1 end · {times[snapshot_indices[0]]:.2f} h",
         f"Current phase map · {times[current_index]:.2f} h",
         "Yellow: gas reclassified by two-source pilot (no vertical zone cut)")):
        ax.imshow(rgb)
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(args.output / "comparison.png", dpi=130)
    plt.close(fig)
    # Native coordinates remain aligned: the same crop is used for both images.
    x0 = max(0, int((float(stages[0]["x_m"]) - .22 - x[0]) / (x[1] - x[0])))
    x1 = min(shape[1], int((float(stages[0]["x_m"]) + .12 - x[0]) / (x[1] - x[0])))
    y0 = max(0, int((y[0] - .14) / (y[0] - y[1])))
    region_slice = np.s_[y0:shape[0], x0:x1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    zooms = (photo[region_slice], palette[current[region_slice]],
             np.where(removed[region_slice][..., None], np.array([255, 218, 0], np.uint8),
                      palette[current[region_slice]]))
    for ax, rgb, title in zip(axes, zooms,
                              ("Corrected photo", "Current phase map", "Candidate reclassification in yellow")):
        ax.imshow(rgb)
        ax.contour(facies_edge[region_slice], levels=[.5], colors=["white"], linewidths=.4, alpha=.6)
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(args.output / "first_injector_zoom.png", dpi=140)
    plt.close(fig)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

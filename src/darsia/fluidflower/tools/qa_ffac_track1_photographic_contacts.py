"""Read-only photographic-display QA and current43 contact sheets.

Only run after the new materialization aggregate is sealed. This checks
rendering and the source chart region, not scientific colour accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import ffac_v5_observed_sand as observed
from darsia.fluidflower.tools.ffac_track1_photographic_display import display_contract


CONTRACT = "ffac-track1-photographic-display-contact-qa-v1"


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_new_json(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if _read(path) != value:
            raise FileExistsError(f"QA report already contains different results: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _revisions(batch_root: str | Path) -> tuple[dict, list[dict]]:
    receipt_path = Path(batch_root) / "materialization_receipt.json"
    aggregate = _read(receipt_path)
    rows = aggregate.get("revisions") or []
    if (aggregate.get("source_revisions") != 43 or len(rows) != 43
            or len({row["run"] for row in rows}) != 43
            or aggregate.get("timeline_frames") != 864):
        raise ValueError("Display QA requires sealed current43/864 materialization")
    revisions = []
    for row in sorted(rows, key=lambda item: int(item["run"][2:])):
        path = Path(row["receipt_path"])
        if sha256_file(path) != row["receipt_sha256"]:
            raise ValueError(f"Materialization revision changed: {path}")
        revision = _read(path)
        manifest_path = Path(row["materialized_manifest_path"])
        if sha256_file(manifest_path) != row["materialized_manifest_sha256"]:
            raise ValueError(f"Native manifest changed: {manifest_path}")
        manifest = _read(manifest_path)
        frames = revision["frames"]
        if (revision["run"] != row["run"] or len(frames) != row["frame_count"]
                or len(manifest["frame_sources"]) != len(frames)):
            raise ValueError("Revision frame order/count differs from sealed aggregate")
        revisions.append(dict(row=row, revision=revision, manifest=manifest))
    return aggregate, revisions


def _chart_centres_native(record: dict, manifest: dict, fit_cache: dict) -> np.ndarray:
    certificate = record["certificate"]
    chart = certificate.get("chart_certificate")
    if chart is not None and "frame_chart_xy8" in chart:
        xy = np.asarray(chart["frame_chart_xy8"], np.float64) * 8 + 3.5
    else:
        processor = record["processor"]
        calpath = Path(processor["calibration_manifest_path"])
        key = (str(calpath), record["run"])
        if key not in fit_cache:
            fit, grid = observed.load_baseline(calpath.parent / f"{record['run']}_baseline.npz")
            fit_cache[key] = (np.asarray(fit.xy, np.float64), grid)
        chart_xy8, grid = fit_cache[key]
        ref_xy = (chart_xy8 + np.asarray(grid.origin_cell_xy)) * 8 + 3.5
        transport_row = manifest["v5_track1_photo_geometry"]["frames"][record["frame_index"]]
        transport_path = Path(transport_row["receipt_path"])
        if sha256_file(transport_path) != transport_row["receipt_sha256"]:
            raise ValueError("Chart source-frame transport receipt changed")
        transport = _read(transport_path)
        h = np.asarray(transport["H_ac28_to_frame_native"], np.float64)
        xy = cv2.perspectiveTransform(ref_xy[None], h)[0]
    if xy.shape != (24, 2) or not np.isfinite(xy).all():
        raise ValueError("Chart visibility check requires 24 finite native centres")
    return xy


def _chart_samples(path: Path, centres: np.ndarray, content_box: list[int]) -> dict:
    left, top, width, height = content_box
    samples = []
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        for x, y in centres:
            px, py = int(round(x)), int(round(y))
            if not (4 <= px < width - 4 and 4 <= py < height - 4):
                samples.append(dict(inside_native=False))
                continue
            patch = np.asarray(image.crop((left + px - 4, top + py - 4,
                                           left + px + 5, top + py + 5)))
            samples.append(dict(inside_native=True,
                                median_rgb8=np.median(patch, axis=(0, 1)).tolist(),
                                near_white_fraction=float(np.mean(np.all(patch >= 250, axis=2)))))
    inside = [value for value in samples if value["inside_native"]]
    nonwhite = sum(value["near_white_fraction"] < .95 for value in inside)
    # A naturally white field may clip. The original blank domain made all
    # chart fields white; this proxy asks whether a varied chart region exists.
    return dict(samples=samples, centres_inside_native=len(inside),
                nonblank_nonwhite_patch_count=nonwhite,
                chart_region_rendered_proxy_pass=len(inside) == 24 and nonwhite >= 12,
                scientific_chart_colour_accuracy_evaluated=False,
                source_chart_occlusion_not_corrected=True)


def inspect_display(new_batch_root: str | Path, output_dir: str | Path) -> dict:
    """Inspect all 864 sealed display pairs and their fixed-transfer receipts."""
    aggregate, revisions = _revisions(new_batch_root)
    expected_transfer = display_contract()
    results = []
    fit_cache = {}
    for item in revisions:
        manifest, revision = item["manifest"], item["revision"]
        content_box = revision["content_box_xywh"]
        total_pixels = content_box[2] * content_box[3]
        for record in revision["frames"]:
            index = record["frame_index"]
            row = manifest["frame_sources"][index]
            main_path, detail_path = Path(row["frame_path"]), Path(record["v5_raw_detail_jpeg_path"])
            main_sha, detail_sha = sha256_file(main_path), sha256_file(detail_path)
            certificate = record["certificate"]
            transfer_pass = (record["processor"].get("display_mode") == "photographic_v1"
                             and record["processor"].get("display_transfer") == expected_transfer
                             and certificate.get("photographic_display") == expected_transfer)
            chart = _chart_samples(main_path, _chart_centres_native(record, manifest, fit_cache),
                                   content_box)
            results.append(dict(
                run=record["run"], frame_index=index, elapsed_h=record["elapsed_h"],
                main_jpeg_sha256=main_sha, raw_detail_sha256=detail_sha,
                receipted_jpeg_hashes_match=(main_sha == record["native_jpeg_sha256"]
                                           and detail_sha == record["v5_raw_detail_jpeg_sha256"]),
                main_and_raw_detail_differ=main_sha != detail_sha,
                fixed_photographic_transfer_pass=transfer_pass,
                full_frame_finite_pixel_count=certificate.get("photographic_full_frame_finite_pixels"),
                full_frame_pixel_count=total_pixels,
                full_frame_finite_fraction=(certificate["photographic_full_frame_finite_pixels"] / total_pixels),
                chart_visibility=chart))
    if len(results) != 864:
        raise ValueError("Display QA did not inspect exactly 864 source frames")
    passed = all(row["receipted_jpeg_hashes_match"] and row["main_and_raw_detail_differ"]
                 and row["fixed_photographic_transfer_pass"]
                 and row["chart_visibility"]["chart_region_rendered_proxy_pass"] for row in results)
    report = dict(contract=CONTRACT, batch_root=str(Path(new_batch_root)),
                  batch_receipt_sha256=sha256_file(Path(new_batch_root) / "materialization_receipt.json"),
                  source_revisions=43, frame_count=864, photographic_transfer=expected_transfer,
                  display_rendering_qa_pass=passed,
                  minimum_full_frame_finite_fraction=min(row["full_frame_finite_fraction"] for row in results),
                  all_main_and_raw_detail_differ=all(row["main_and_raw_detail_differ"] for row in results),
                  scientific_colour_accuracy_or_phase_acceptance_claimed=False,
                  chart_visibility_is_rendered_region_proxy=True, frames=results)
    _write_new_json(Path(output_dir) / "qa_display.json", report)
    return report


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _contact(selection: list[dict], destination: Path, title: str) -> str:
    columns, tile_w, tile_h, caption_h, heading_h = 6, 550, 305, 42, 45
    rows = (len(selection) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, heading_h + rows * (tile_h + caption_h)), "#151b20")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, font=_font(24), fill="white")
    for position, selected in enumerate(selection):
        x = (position % columns) * tile_w
        y = heading_h + (position // columns) * (tile_h + caption_h)
        box = selected["content_box_xywh"]
        left, top, width, height = box
        with Image.open(selected["frame_path"]) as opened:
            image = opened.crop((left, top, left + width, top + height)).convert("RGB")
            image.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
            sheet.paste(image, (x + (tile_w - image.width) // 2, y + (tile_h - image.height) // 2))
        caption = f"{selected['run'].upper()} | f{selected['frame_index']:03d} | {selected['elapsed_h']:.3f} h"
        draw.text((x + 8, y + tile_h + 6), caption, font=_font(18), fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Contact sheet exists: {destination}")
    sheet.save(destination, "JPEG", quality=95, subsampling=0)
    return sha256_file(destination)


def make_contacts(new_batch_root: str | Path, output_dir: str | Path) -> dict:
    """Create all43 start and nearest-24h sheets, retaining the chart and rig."""
    _, revisions = _revisions(new_batch_root)
    start, near24 = [], []
    for item in revisions:
        rows = item["manifest"]["frame_sources"]
        for selected, destination in ((rows[0], start),
                                      (min(rows, key=lambda row: abs(float(row["elapsed_h"]) - 24)), near24)):
            destination.append(dict(run=item["revision"]["run"],
                                    frame_index=selected["frame_index"], elapsed_h=float(selected["elapsed_h"]),
                                    frame_path=selected["frame_path"],
                                    content_box_xywh=item["revision"]["content_box_xywh"]))
    output_dir = Path(output_dir)
    paths = [output_dir / "contacts_start.jpg", output_dir / "contacts_24h.jpg"]
    hashes = [_contact(start, paths[0], "FF_AC all43 | frame 0 | fixed photographic display"),
              _contact(near24, paths[1], "FF_AC all43 | nearest 24 h | fixed photographic display")]
    report = dict(contract=CONTRACT, batch_root=str(Path(new_batch_root)),
                  start_contact_path=str(paths[0]), start_contact_sha256=hashes[0],
                  nearest24h_contact_path=str(paths[1]), nearest24h_contact_sha256=hashes[1],
                  start_frames=start, nearest24h_frames=near24,
                  display_only_artifacts_not_scientific_resampling=True)
    _write_new_json(output_dir / "contacts_selection.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contacts", action="store_true")
    arguments = parser.parse_args()
    report = inspect_display(arguments.batch_root, arguments.output_dir)
    if arguments.contacts:
        make_contacts(arguments.batch_root, arguments.output_dir)
    print(json.dumps(dict(frame_count=report["frame_count"],
                          display_rendering_qa_pass=report["display_rendering_qa_pass"])))


if __name__ == "__main__":
    main()

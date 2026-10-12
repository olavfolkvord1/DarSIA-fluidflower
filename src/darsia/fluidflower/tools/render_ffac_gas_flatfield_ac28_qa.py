"""Render AC28-only natural photographs with the isolated gas variants.

This presentation command reads only the AC28 inference receipt, the frozen
AC28 package and its already-rendered natural JPEGs.  It never opens a test
run, fits a model, transforms photographic brightness, or computes mass.
Native overlays use ``frameNNN.png``; the inference runner's categorical
``frame_NNN.png`` and numerical archives remain separate source artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS
from darsia.fluidflower.project import sha256_file

CONTRACT = "ffac-gas-flatfield-ac28-natural-overlay-qa-v1"
VARIANTS = ("G0", "G0prime", "G1", "G2", "G3")
NAMES = {
    "G0": "G0 (frozen reference)",
    "G0prime": "G0′",
    "G1": "G1",
    "G2": "G2",
    "G3": "G3",
}
PHASE_TINTS = {
    1: ((49, 217, 160), 0.20),
    2: ((242, 76, 157), 0.36),
    255: ((30, 210, 255), 0.10),
}
ADDED_GAS = (255, 155, 35)
REMOVED_GAS = (45, 160, 255)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _bound(path, digest, label):
    source = Path(path)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or sha256_file(source) != digest
    ):
        raise ValueError(f"AC28 QA {label} changed after source binding: {source}")
    return source


def _font(size):
    for name in ("arial.ttf", "segoeui.ttf"):
        path = Path(r"C:\Windows\Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _phase_labels(path, native):
    with Image.open(path) as opened:
        if opened.mode != "RGBA" or opened.size != tuple(native["canvas_wh"]):
            raise ValueError("AC28 phase map must have its exact native RGBA canvas")
        rgba = np.asarray(opened)
    x, y, width, height = map(int, native["content_box_xywh"])
    local = rgba[y : y + height, x : x + width]
    labels = np.full((height, width), 255, np.uint8)
    known = local[..., 3] == 0
    labels[known] = 0
    for phase, code in (("aqueous", 1), ("gas", 2), ("unresolved", 255)):
        match = np.all(local == np.asarray(PHASE_FILL_COLORS[phase], np.uint8), axis=-1)
        labels[match] = code
        known |= match
    if not known.all():
        raise ValueError("AC28 categorical map has unknown phase colours")
    return labels


def _geometry(native):
    canvas = tuple(map(int, native["canvas_wh"]))
    box = tuple(map(int, native["content_box_xywh"]))
    x, y, width, height = box
    if (
        native.get("run") != "ac28"
        or tuple(native["analysis_image_shape_hw"]) != (height, width)
        or min(canvas) < 1
        or min(width, height) < 1
        or x < 0
        or y < 0
        or x + width > canvas[0]
        or y + height > canvas[1]
    ):
        raise ValueError("AC28 QA source has inconsistent native geometry")
    return canvas, box


def _selected_frames(frames):
    indices = [int(row["frame_index"]) for row in frames]
    if not frames or indices != list(range(len(frames))):
        raise ValueError("AC28 QA receipt does not cover an ordered complete timeline")
    near = min(indices, key=lambda index: abs(float(frames[index]["elapsed_h"]) - 2.25))
    return sorted({0, near, indices[-1]})


def _outline(co2):
    return cv2.morphologyEx(
        co2.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def _tint(image, selected, colour, opacity):
    if selected.any():
        values = image[selected].astype(np.float32)
        image[selected] = np.rint(
            values * (1 - opacity) + np.asarray(colour) * opacity
        ).astype(np.uint8)


def render_overlay(photo, labels, co2, box, *, reference_labels=None):
    """Preserve the existing photo outside explicit optical overlay pixels."""
    rgb = np.asarray(photo)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise TypeError("AC28 photographic display must be existing uint8 RGB")
    x, y, width, height = map(int, box)
    if labels.shape != (height, width) or labels.dtype != np.uint8:
        raise TypeError("AC28 phase labels must be native HW uint8")
    if co2.dtype != np.bool_ or co2.shape != labels.shape:
        raise TypeError("AC28 frozen CO2 support must be native HW bool")
    if not np.isin(labels, (0, 1, 2, 255)).all() or not np.array_equal(
        np.isin(labels, (1, 2)), co2
    ):
        raise ValueError("AC28 variant changed the frozen CO2/water boundary")
    result = rgb.copy()
    local = result[y : y + height, x : x + width]
    if reference_labels is None:
        for code, (colour, opacity) in PHASE_TINTS.items():
            _tint(local, labels == code, colour, opacity)
    else:
        if reference_labels.shape != labels.shape or not np.array_equal(
            np.isin(reference_labels, (1, 2)), co2
        ):
            raise ValueError("AC28 difference reference changed CO2 support")
        _tint(local, (labels == 2) & (reference_labels == 1), ADDED_GAS, 0.75)
        _tint(local, (labels == 1) & (reference_labels == 2), REMOVED_GAS, 0.75)
    local[_outline(co2)] = (255, 255, 255)
    return result


def _annotated(rgb, title):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _font(max(22, image.width // 180))
    text = title + " | CO₂ boundary unchanged | optical QA; no mass"
    text_box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((8, 8, text_box[2] + 32, text_box[3] + 28), fill=(12, 16, 20))
    draw.text((20, 16), text, font=font, fill=(245, 245, 245))
    return image


def _save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as previous:
            if previous.size != image.size or not np.array_equal(
                np.asarray(previous.convert("RGB")), np.asarray(image.convert("RGB"))
            ):
                raise FileExistsError(f"Existing AC28 QA image differs: {path}")
    else:
        with path.open("xb") as stream:
            image.save(stream, format="PNG", compress_level=3)
    return {"path": str(path), "sha256": sha256_file(path), "size_wh": list(image.size)}


def _thumbnail(image, width=800):
    return image.resize(
        (width, round(image.height * width / image.width)), Image.Resampling.LANCZOS
    )


def _contacts(rows):
    cell_width = 800
    cell_height = rows[0]["natural"].height
    band = 44
    top, bottom, gap = 116, 132, 12
    sheet = Image.new(
        "RGB",
        (
            6 * cell_width + 7 * gap,
            top + len(rows) * 2 * (cell_height + band + gap) + bottom,
        ),
        (17, 22, 29),
    )
    draw = ImageDraw.Draw(sheet)
    font, small = _font(25), _font(21)
    draw.text(
        (gap, 12),
        "AC28 | Existing natural display | Gas-head variants G0′–G3",
        font=font,
        fill="white",
    )
    draw.text(
        (gap, 48),
        "All CO₂/water boundaries are identical. G0 is the frozen reference. Optical QA only; no mass interpretation.",
        font=small,
        fill=(225, 228, 234),
    )
    for group, row in enumerate(rows):
        for difference in (False, True):
            line = group * 2 + int(difference)
            y = top + line * (cell_height + band + gap)
            title = f"frame {row['frame_index']} | {row['elapsed_h']:.3f} h"
            draw.text(
                (gap, y + 7),
                "Δ gas vs G0′" if difference else title,
                font=small,
                fill="white",
            )
            sheet.paste(row["natural"], (gap, y + band))
            for column, variant in enumerate(VARIANTS, start=1):
                x = gap + column * (cell_width + gap)
                draw.text(
                    (x, y + 7),
                    NAMES[variant] + (" | difference" if difference else ""),
                    font=small,
                    fill="white",
                )
                sheet.paste(
                    row["differences" if difference else "overlays"][variant],
                    (x, y + band),
                )
    legend_y = sheet.height - bottom + 8
    legend = [
        ((49, 217, 160), "Dissolved / aqueous"),
        ((242, 76, 157), "Free gas"),
        ((30, 210, 255), "Unresolved"),
        ((255, 255, 255), "Shared CO₂ boundary"),
        (ADDED_GAS, "Added gas vs G0′"),
        (REMOVED_GAS, "Removed gas vs G0′"),
    ]
    x = gap
    for colour, label in legend:
        draw.rectangle((x, legend_y, x + 30, legend_y + 24), fill=colour)
        draw.text((x + 42, legend_y - 1), label, font=small, fill="white")
        x += 800
    draw.text(
        (gap, legend_y + 44),
        "Natural JPEG pixels are unchanged beneath fixed transparent overlays. Difference colours show gas/dissolved switches only.",
        font=small,
        fill=(225, 228, 234),
    )
    draw.text(
        (gap, legend_y + 77),
        "Only AC28 is displayed. Held-out runs remain sealed for Claude until the labels are locked.",
        font=small,
        fill=(225, 228, 234),
    )
    return sheet


def render_ac28_qa(frozen_candidate, experiment):
    """Render the three AC28 QA frames after the restricted inference finishes."""
    started = time.perf_counter()
    candidate, experiment = Path(frozen_candidate).resolve(), Path(experiment).resolve()
    if experiment.is_relative_to(candidate) or candidate.is_relative_to(experiment):
        raise ValueError("AC28 QA outputs must be separate from the frozen candidate")
    root = experiment / "qa_ac28/ac28"
    inference_path = root / "inference_receipt.json"
    inference_sha = sha256_file(inference_path)
    inference = _read(inference_path)
    if (
        inference.get("run") != "ac28"
        or Path(inference["frozen_candidate"]).resolve() != candidate
    ):
        raise ValueError("AC28 QA received another run or frozen candidate")
    selected = _selected_frames(inference["frames"])
    postflight_path = (
        candidate / "validation/run_packages/ac28/postflight_manifest.json"
    )
    postflight = _read(postflight_path)
    jobs = [job for job in postflight["jobs"] if job.get("run") == "ac28"]
    if len(jobs) != 1:
        raise ValueError(
            "Frozen AC28 postflight manifest lacks a unique native source job"
        )
    job = jobs[0]
    expected_native = (
        Path(job["analysis_source_review_root"]) / "native_frames/ac28/manifest.json"
    )
    if Path(inference["native_manifest_path"]).resolve() != expected_native.resolve():
        raise ValueError(
            "AC28 natural source differs from the original postflight manifest"
        )
    native_path = _bound(
        inference["native_manifest_path"],
        inference["native_manifest_sha256"],
        "native manifest",
    )
    native = _read(native_path)
    canvas, box = _geometry(native)
    frozen_folder = (
        Path(job["phase_map_output_dir"])
        / "native_phase_maps/ac28"
        / ("A_" + job["params_sha256"][:16])
    )
    previous_receipt = root / "qa_render_receipt.json"
    if previous_receipt.exists():
        prior = _read(previous_receipt)
        if prior.get("inference_receipt_sha256") != inference_sha or prior.get(
            "renderer_code_sha256"
        ) != sha256_file(Path(__file__)):
            raise ValueError(
                "Existing AC28 QA rendering belongs to another input or renderer"
            )
        for output in prior["outputs"]:
            if not Path(output["path"]).resolve().is_relative_to(root):
                raise ValueError("Saved AC28 QA output points outside its allowed run")
            _bound(output["path"], output["sha256"], "saved overlay")
        return prior
    output_rows, contact_rows, outputs = [], [], []
    for index in selected:
        row = inference["frames"][index]
        source = native["frame_sources"][index]
        if int(source["frame_index"]) != index or float(source["elapsed_h"]) != float(
            row["elapsed_h"]
        ):
            raise ValueError("AC28 QA frame time differs from its native source")
        natural = (source.get("v5_display_modes") or {}).get("natural") or {}
        expected_photo = native_path.parent / f"frame_{index:03d}.jpg"
        if Path(natural["jpeg_path"]).resolve() != expected_photo.resolve():
            raise ValueError(
                "AC28 natural photograph is outside its exact original native path"
            )
        natural_path = _bound(
            natural["jpeg_path"], natural["jpeg_sha256"], "natural display photograph"
        )
        with Image.open(natural_path) as image:
            if image.size != canvas:
                raise ValueError("AC28 natural display changed its native canvas")
            photo = np.asarray(image.convert("RGB"))
        if (
            Path(row["frozen_map_path"]).resolve()
            != (frozen_folder / f"frame_{index:03d}.png").resolve()
        ):
            raise ValueError(
                "AC28 G0 reference differs from its original postflight map"
            )
        frozen_path = _bound(
            row["frozen_map_path"], row["frozen_map_sha256"], "frozen G0 phase map"
        )
        labels = {"G0": _phase_labels(frozen_path, native)}
        inputs = {
            "G0": {"map_path": str(frozen_path), "map_sha256": row["frozen_map_sha256"]}
        }
        for variant in VARIANTS[1:]:
            entry = row["variants"][variant]
            expected_map = root / variant / f"frame_{index:03d}.png"
            if Path(entry["map_path"]).resolve() != expected_map.resolve():
                raise ValueError(
                    "AC28 QA variant map points outside its fixed allowed run"
                )
            map_path = _bound(
                entry["map_path"], entry["map_sha256"], f"{variant} phase map"
            )
            labels[variant] = _phase_labels(map_path, native)
            inputs[variant] = {
                "map_path": str(map_path),
                "map_sha256": entry["map_sha256"],
            }
        co2 = np.isin(labels["G0"], (1, 2))
        if any(
            not np.array_equal(np.isin(value, (1, 2)), co2) for value in labels.values()
        ):
            raise ValueError("AC28 variant changed the frozen CO2/water boundary")
        natural_output = _save_png(
            root / "natural" / f"frame{index:03d}.png", Image.fromarray(photo)
        )
        outputs.append(natural_output)
        contact = dict(
            frame_index=index,
            elapsed_h=float(row["elapsed_h"]),
            natural=_thumbnail(Image.fromarray(photo)),
            overlays={},
            differences={},
        )
        artifacts = {}
        for variant in VARIANTS:
            title = f"AC28 frame {index:03d} | {float(row['elapsed_h']):.3f} h | {NAMES[variant]}"
            overlay = render_overlay(photo, labels[variant], co2, box)
            difference = render_overlay(
                photo, labels[variant], co2, box, reference_labels=labels["G0prime"]
            )
            overlay_artifact = _save_png(
                root / variant / f"frame{index:03d}.png", _annotated(overlay, title)
            )
            difference_artifact = _save_png(
                root / variant / f"frame{index:03d}.difference_G0prime.png",
                _annotated(difference, title + " | gas difference vs G0′"),
            )
            outputs.extend((overlay_artifact, difference_artifact))
            contact["overlays"][variant] = _thumbnail(Image.fromarray(overlay))
            contact["differences"][variant] = _thumbnail(Image.fromarray(difference))
            artifacts[variant] = dict(
                overlay=overlay_artifact, difference=difference_artifact
            )
            del overlay, difference
        contact_rows.append(contact)
        output_rows.append(
            dict(
                frame_index=index,
                elapsed_h=float(row["elapsed_h"]),
                natural_source_path=str(natural_path),
                natural_source_sha256=natural["jpeg_sha256"],
                co2_union_identical_all_variants=True,
                inputs=inputs,
                outputs=artifacts,
            )
        )
        print(f"AC28 QA rendering frame {index} complete", flush=True)
    contacts = _save_png(
        root / "ac28_natural_gas_variants_contactsheet.png", _contacts(contact_rows)
    )
    outputs.append(contacts)
    receipt = dict(
        contract=CONTRACT,
        created_utc=datetime.now(timezone.utc).isoformat(),
        run="ac28",
        inference_receipt_sha256=inference_sha,
        renderer_code_sha256=sha256_file(Path(__file__)),
        original_postflight_path=str(postflight_path),
        original_postflight_sha256=sha256_file(postflight_path),
        native_manifest_sha256=inference["native_manifest_sha256"],
        photographic_transform="existing-natural-JPEG; no brightness or colour transform",
        co2_union_identical_all_variants=True,
        mass_interpretation=False,
        held_out_files_read=False,
        frames=output_rows,
        contactsheet=contacts,
        outputs=outputs,
        wall_seconds=time.perf_counter() - started,
    )
    with previous_receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-candidate", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render_ac28_qa(args.frozen_candidate, args.experiment)
    print(
        json.dumps(
            {
                "run": "ac28",
                "contactsheet": receipt["contactsheet"]["path"],
                "frames": [row["frame_index"] for row in receipt["frames"]],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

"""Render every computed acquisition as a paired photograph/concentration movie."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODES = {"overview": 1.6, "detail": .6}
SIZE = (3840, 1440)
BG = (15, 25, 40)
FG = (238, 243, 248)
MUTED = (164, 183, 201)
WATER = (232, 237, 241)
GAS = (67, 76, 89)
OVER = (249, 115, 22)
LUT = np.rint(matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
TEXT = {
    "nb": dict(before="Før injeksjon", photo="Fotografi · samme bildekorrigering som analysen",
        concentration="Innløst CO₂ · konsentrasjonsestimat", outside="Utenfor innløst fase",
        gas="Fri CO₂ · også blandede piksler", scale="Fast lineær skala · samme skala gjennom hele filmen",
        time="Forsøkstid (timer) · ett fotografi per opptak · opptaksintervallene varierer",
        image="bilde", injected="Injisert", estimated="Estimert CO₂ (fri + innløst)", over="Over"),
    "en": dict(before="Before injection", photo="Photograph · same image correction as analysis",
        concentration="Dissolved CO₂ · concentration estimate", outside="Outside dissolved phase",
        gas="Free CO₂ · including mixed pixels", scale="Fixed linear scale · unchanged throughout the movie",
        time="Experiment time (hours) · one photograph per acquisition · acquisition intervals vary",
        image="image", injected="Injected", estimated="Estimated CO₂ (free + dissolved)", over="Above"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value, *, replace_attempts=10, retry_delay=.05):
    """Atomically publish JSON despite short Windows reader sharing conflicts."""
    if replace_attempts < 1 or retry_delay < 0:
        raise ValueError("Invalid JSON replacement retry policy")
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        for attempt in range(replace_attempts):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == replace_attempts - 1:
                    raise
                time.sleep(min(retry_delay * 2 ** attempt, .5))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Cleanup must not mask the publication error or remove the
                # previously published JSON. A locked unique temp is harmless.
                pass


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def font(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/" + ("segoeuib.ttf" if bold else "segoeui.ttf"), size)


def elapsed_text(hours, language="nb"):
    if hours < 0:
        return TEXT[language]["before"]
    seconds = round(hours * 3600)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def schedule(rows, fps=25, target_seconds=90):
    if not rows or fps <= 0:
        raise ValueError("A movie needs input frames and a positive frame rate")
    times = [float(row["elapsed_h"]) for row in rows]
    if not all(np.isfinite(times)) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Acquisition times must be finite and strictly increasing")
    repeat = max(1, round(fps * target_seconds / len(rows)))
    durations = [repeat] * len(rows)
    durations[0] = max(repeat, round(2 * fps))
    durations[-1] = max(repeat, round(3 * fps))
    starts = np.r_[0, np.cumsum(durations)[:-1]].tolist()
    return durations, starts


def colors(concentration, aqueous, gas, limit):
    c = np.asarray(concentration, np.float32)
    aq = np.asarray(aqueous) > 0
    gas_present = np.asarray(gas) > 0
    if c.ndim != 2 or c.shape != aq.shape or c.shape != gas_present.shape:
        raise ValueError("Concentration and phase grids differ")
    if not np.isfinite(c[aq]).all() or (c[aq] < -1e-7).any():
        raise ValueError("Aqueous concentration is negative or nonfinite")
    safe = np.where(aq, c, 0)
    out = LUT[np.rint(np.clip(safe / limit, 0, 1) * 255).astype(np.uint8)].copy()
    out[~aq] = WATER
    out[aq & (safe > limit)] = OVER
    # Gas takes display priority in mixed area-reduced cells. Do not erase
    # gas merely because the same reduced cell also contains aqueous pores.
    out[gas_present] = GAS
    return out


def decimal_text(value, language="nb"):
    return f"{value:.2f}".replace(".", ",") if language == "nb" else f"{value:.2f}"


def frame_labels(record, index, total, language="nb"):
    text = TEXT[language]
    position = f"{elapsed_text(float(record['elapsed_h']), language)}   ·   {text['image']} {index + 1:,} / {total:,}"
    if language == "nb":
        position = position.replace(",", " ")
    mass = record.get("injected_mass_g")
    detected = record.get("total_mass_g", record.get("total_g"))
    amounts = f"{text['injected']}: {decimal_text(mass, language)} g" if mass is not None else ""
    if detected is not None:
        amounts += f"    {text['estimated']}: {decimal_text(detected, language)} g"
    return position, amounts


def make_background(dataset, run, limit, language="nb"):
    text = TEXT[language]
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    name = {"ffum": "FF_UM", "ffac": "FF_AC"}.get(dataset, dataset.upper())
    draw.text((65, 26), f"{name} · {run.upper()}", fill=FG, font=font(51, True))
    draw.text((65, 96), text["photo"], fill=FG, font=font(34))
    draw.text((1900, 96), text["concentration"], fill=FG, font=font(34))
    # Fixed positions and aspect ratio for every frame, including the baseline.
    bar_x, bar_y, bar_w, bar_h = 3640, 183, 39, 846
    bar = np.repeat(LUT[::-1, None], bar_w, axis=1)
    canvas.paste(Image.fromarray(bar).resize((bar_w, bar_h)), (bar_x, bar_y))
    draw.text((3627, 137), "g/L", fill=FG, font=font(27))
    ticks = np.linspace(0, limit, 9 if limit > 1 else 7)
    for tick in ticks:
        y = bar_y + round((1 - tick / limit) * bar_h)
        draw.line((bar_x + bar_w, y, bar_x + bar_w + 10, y), fill=FG, width=2)
        draw.text((bar_x + bar_w + 17, y - 18), decimal_text(tick, language), fill=FG, font=font(27))
    draw.rectangle((3255, 1155, 3281, 1181), fill=OVER)
    draw.text((3292, 1148), f"{text['over']} {decimal_text(limit, language)} g/L", fill=FG, font=font(25))
    draw.rectangle((65, 1155, 91, 1181), fill=WATER)
    draw.text((105, 1148), text["outside"], fill=MUTED, font=font(27))
    draw.rectangle((475, 1155, 501, 1181), fill=GAS)
    draw.text((515, 1148), text["gas"], fill=MUTED, font=font(27))
    draw.text((1900, 1148), text["scale"], fill=MUTED, font=font(27))
    x0, x1, y = 75, 3760, 1327
    draw.line((x0, y, x1, y), fill=(69, 89, 109), width=5)
    for t in range(0, 73, 12):
        x = round(x0 + (x1 - x0) * t / 72)
        draw.line((x, y - 8, x, y + 8), fill=MUTED, width=2)
        draw.text((x - 15 if t else x, y + 13), str(t), fill=MUTED, font=font(24))
    draw.text((65, 1390), text["time"], fill=MUTED, font=font(25))
    return canvas


def put_panel(canvas, rgb, box, resample=Image.Resampling.LANCZOS):
    x, y, width, height = box
    image = ImageOps.contain(Image.fromarray(rgb), (width, height), resample)
    canvas.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))


def compose(background, photograph, concentration, aqueous, gas, record, index, total, limit, language="nb"):
    canvas = background.copy()
    put_panel(canvas, photograph, (65, 159, 1690, 922))
    put_panel(canvas, colors(concentration, aqueous, gas, limit), (1900, 159, 1690, 922), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    t = float(record["elapsed_h"])
    position, amounts = frame_labels(record, index, total, language)
    draw.text((65, 1227), position, fill=FG, font=font(43, True))
    draw.text((1900, 1227), amounts, fill=FG, font=font(31))
    x = round(75 + (3760 - 75) * np.clip(t / 72, 0, 1))
    draw.line((75, 1327, x, 1327), fill=(87, 208, 195), width=5)
    draw.ellipse((x - 9, 1318, x + 9, 1336), fill=(87, 208, 195))
    return np.asarray(canvas)


class Encoder:
    def __init__(self, output, fps):
        self.output = Path(output)
        self.temporary = self.output.with_name(self.output.stem + ".partial.mp4")
        if self.output.exists() or self.temporary.exists():
            raise FileExistsError(self.output)
        self.log = self.output.with_suffix(".ffmpeg.log").open("wb")
        args = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{SIZE[0]}x{SIZE[1]}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "19", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(self.temporary)]
        self.command = args
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def add(self, frame, repeat):
        data = frame.tobytes()
        for _ in range(repeat):
            self.process.stdin.write(data)

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError(f"Movie encoding failed ({code}): {self.output}")
        self.temporary.replace(self.output)

    def abort(self):
        self.process.terminate()
        self.process.wait()
        self.log.close()


def await_field(path, record_path, timeout, job_root):
    start = time.monotonic()
    while not path.is_file() or not record_path.is_file():
        if (job_root / "compute_failure.json").is_file():
            raise RuntimeError(f"Field computation failed: {job_root / 'compute_failure.json'}")
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Still waiting for computed field {path}")
        time.sleep(.5)


def validate_video(path, expected_frames, fps):
    capture = cv2.VideoCapture(str(path))
    try:
        count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        if count != expected_frames or abs(actual_fps - fps) > 1e-5:
            raise ValueError(f"Incorrect encoded length: {count}, {actual_fps}")
        inspected = sorted(set([0, count // 4, count // 2, 3 * count // 4, count - 1]))
        for index in inspected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame.shape[:2] != SIZE[::-1]:
                raise ValueError(f"Undecodable movie frame {index}: {path}")
        return {"passed": True, "decoded_frame_indices": inspected, "frame_count": count, "fps": actual_fps}
    finally:
        capture.release()


def render(manifest_path, output_dir, *, fps=25, target_seconds=90, modes=("overview", "detail"), wait_seconds=3600, language="nb"):
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = read(manifest_path)
    rows = manifest["frames"]
    dataset, run = manifest["dataset"], manifest["run"]
    durations, starts = schedule(rows, fps, target_seconds)
    output_dir.mkdir(parents=True, exist_ok=True)
    backgrounds = {mode: make_background(dataset, run, MODES[mode], language) for mode in modes}
    encoders = {}
    times = np.asarray([r["elapsed_h"] for r in rows], float)
    snapshot_indices = sorted(set(int(np.argmin(abs(times - t))) for t in (0, .5, 6, 24, 48, 72)))
    metadata = []
    posters = {}
    started = time.monotonic()
    try:
        for mode in modes:
            encoders[mode] = Encoder(output_dir / f"{run}_{mode}.mp4", fps)
        for index, row in enumerate(rows):
            fields = Path(row["fields_path"])
            record_path = Path(row.get("record_path") or fields.with_suffix(".json"))
            await_field(fields, record_path, wait_seconds, manifest_path.parent)
            record = read(record_path)
            if abs(float(record["elapsed_h"]) - times[index]) > 1e-8:
                raise ValueError("Field acquisition time differs from planned movie frame")
            digest = sha(fields)
            if record.get("fields_sha256") and digest != record["fields_sha256"]:
                raise ValueError("Computed movie field changed")
            with np.load(fields, allow_pickle=False) as z:
                photo, c = z["photograph_rgb"], z["concentration_g_l"]
                aq, gas = z["aqueous_fraction"], z["gas_fraction"]
                if photo.dtype != np.uint8 or photo.shape[:2] != c.shape or photo.shape[-1] != 3:
                    raise ValueError("Photograph and field grids differ")
                for mode, encoder in encoders.items():
                    frame = compose(backgrounds[mode], photo, c, aq, gas, record, index, len(rows), MODES[mode], language)
                    encoder.add(frame, durations[index])
                    if index in snapshot_indices:
                        image_path = output_dir / f"{run}_{mode}_{index:05d}.png"
                        Image.fromarray(frame).save(image_path)
                        if index == snapshot_indices[-2 if len(snapshot_indices) > 1 else -1]:
                            posters[mode] = str(image_path)
            metadata.append({"source_index": index, "elapsed_h": times[index],
                "video_start_seconds": starts[index] / fps, "video_end_seconds": (starts[index] + durations[index]) / fps,
                "source_path": row["source_path"], "fields_path": str(fields), "fields_sha256": digest})
            if index % 25 == 0 or index == len(rows) - 1:
                status = {"stage": "rendering", "run": run, "completed": index + 1, "total": len(rows),
                    "elapsed_seconds": time.monotonic() - started}
                write(output_dir / "render_progress.json", status)
                print(json.dumps(status), flush=True)
        for encoder in encoders.values():
            encoder.close()
    except BaseException:
        for encoder in encoders.values():
            if encoder.process.poll() is None:
                encoder.abort()
        raise
    results = []
    for mode, encoder in encoders.items():
        verification = validate_video(encoder.output, sum(durations), fps)
        result = {"format": "concentration-side-by-side-movie-v1", "dataset": dataset, "run": run, "mode": mode,
            "output": str(encoder.output), "sha256": sha(encoder.output), "language": language, "fps": fps, "frame_count": sum(durations),
            "duration_seconds": sum(durations) / fps, "source_frame_count": len(rows),
            "source_start_h": float(times[0]), "source_end_h": float(times[-1]), "linear_limits_g_l": [0, MODES[mode]],
            "poster": posters[mode], "frames": metadata, "computation_manifest": str(manifest_path),
            "method_manifest": str(manifest_path.parent / "method.json"), "size": list(SIZE),
            "temporal_interpolation": False, "framewise_color_autoscaling": False, "verification": verification,
            "gas_display_policy": "gray takes priority wherever gas_fraction > 0, including cells also containing aqueous pores",
            "field_display_resampling": "nearest; no interpolation between concentration, gas and background colors",
            "over_range_color_rgb": list(OVER), "encoder_command": encoder.command}
        write(encoder.output.with_suffix(".json"), result)
        results.append(result)
    write(output_dir / "render_progress.json", {"stage": "complete", "run": run, "completed": len(rows), "total": len(rows)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--target-seconds", type=float, default=90)
    parser.add_argument("--wait-seconds", type=float, default=3600)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument("--language", choices=tuple(TEXT), default="nb")
    args = parser.parse_args()
    for result in render(args.manifest, args.output_dir, fps=args.fps, target_seconds=args.target_seconds, wait_seconds=args.wait_seconds, modes=args.modes, language=args.language):
        print(json.dumps({key: result[key] for key in ("output", "source_frame_count", "duration_seconds", "verification")}), flush=True)

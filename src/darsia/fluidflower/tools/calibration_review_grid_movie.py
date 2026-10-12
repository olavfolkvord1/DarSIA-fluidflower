"""Create a ranked grid movie from calibration review movies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MovieEntry:
    rank: int
    variant: str
    objective: float
    path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Postflight root containing <variant>/<run>/summary.json.",
    )
    parser.add_argument("--run", default="ac60")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument(
        "--sort-by",
        choices=("objective", "grid-order", "variant"),
        default="objective",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _discover(
    input_root: Path,
    run: str,
    *,
    sort_by: str = "objective",
) -> list[MovieEntry]:
    candidates: list[tuple[str, float, float | None, Path]] = []
    for summary_path in input_root.glob(f"*/{run}/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        movie = summary_path.parent / "calibration_review.mp4"
        if not movie.is_file():
            continue
        objective = float(summary["objective_full_scale"])
        variant = str(summary.get("variant") or summary_path.parents[1].name)
        grid_order = (
            float(summary["grid_order"])
            if summary.get("grid_order") is not None
            else None
        )
        candidates.append((variant, objective, grid_order, movie))

    if not candidates:
        raise FileNotFoundError(
            f"No calibration review movies found below {input_root} for {run}."
        )
    if sort_by == "grid-order":
        missing = [variant for variant, _objective, order, _movie in candidates if order is None]
        if missing:
            raise ValueError(f"Missing grid_order for variants: {missing}")
        candidates.sort(key=lambda item: (float(item[2]), item[0]))
    elif sort_by == "variant":
        candidates.sort(key=lambda item: item[0])
    else:
        candidates.sort(key=lambda item: (item[1], item[0]))
    return [
        MovieEntry(rank=index, variant=variant, objective=objective, path=movie)
        for index, (variant, objective, _order, movie) in enumerate(
            candidates, start=1
        )
    ]


def _ranking_color(rank: int, count: int) -> tuple[int, int, int]:
    fraction = 0.0 if count <= 1 else (rank - 1) / (count - 1)
    if fraction <= 0.5:
        local = fraction / 0.5
        return (55, 185, int(round(65 + 155 * local)))
    local = (fraction - 0.5) / 0.5
    return (55, int(round(220 - 165 * local)), 220)


def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, int(round(frame.shape[1] * scale)))
    resized_height = max(1, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def _draw_header(
    tile: np.ndarray,
    entry: MovieEntry,
    count: int,
    header_height: int,
) -> None:
    color = _ranking_color(entry.rank, count)
    tile[:header_height] = (28, 30, 32)
    cv2.rectangle(tile, (0, 0), (9, header_height - 1), color, thickness=-1)
    label = (
        f"#{entry.rank:02d}  {entry.variant}  |  "
        f"objective {1000.0 * entry.objective:.4f} g"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 1
    max_width = tile.shape[1] - 32
    while cv2.getTextSize(label, font, font_scale, thickness)[0][0] > max_width:
        font_scale -= 0.025
        if font_scale <= 0.35:
            break
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    baseline_y = (header_height + text_size[1]) // 2
    cv2.putText(
        tile,
        label,
        (22, baseline_y),
        font,
        font_scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def _write_ranking(entries: list[MovieEntry], output: Path) -> Path:
    ranking_path = output.with_name(f"{output.stem}_ranking.csv")
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["rank", "variant", "objective_full_scale", "objective_g", "movie"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "rank": entry.rank,
                    "variant": entry.variant,
                    "objective_full_scale": entry.objective,
                    "objective_g": 1000.0 * entry.objective,
                    "movie": entry.path,
                }
            )
    return ranking_path


def _progress(message: str) -> None:
    """Keep detached renderers alive if their original stdout disappears."""

    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        pass


def create_grid_movie(
    entries: list[MovieEntry],
    output: Path,
    columns: int,
    output_width: int,
    output_height: int,
) -> tuple[int, float]:
    rows = math.ceil(len(entries) / columns)
    if rows * columns != len(entries):
        raise ValueError(
            f"The {len(entries)} movies do not fill a complete {columns}-column grid."
        )
    if output_width % columns or output_height % rows:
        raise ValueError("Output dimensions must be divisible by the grid dimensions.")

    captures = [cv2.VideoCapture(str(entry.path)) for entry in entries]
    try:
        if not all(capture.isOpened() for capture in captures):
            failed = [
                str(entry.path)
                for entry, capture in zip(entries, captures, strict=True)
                if not capture.isOpened()
            ]
            raise RuntimeError(f"Unable to open input movies: {failed}")

        frame_counts = [
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures
        ]
        frame_rates = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
        if len(set(frame_counts)) != 1:
            raise ValueError(f"Input frame counts differ: {frame_counts}")
        if max(frame_rates) - min(frame_rates) > 1e-6:
            raise ValueError(f"Input frame rates differ: {frame_rates}")

        frame_count = frame_counts[0]
        fps = frame_rates[0]
        tile_width = output_width // columns
        tile_height = output_height // rows
        header_height = max(44, round(tile_height * 0.095))

        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f"{output.stem}.partial{output.suffix}")
        writer = cv2.VideoWriter(
            str(partial),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create video writer for {partial}.")

        try:
            for frame_index in range(frame_count):
                grid = np.full(
                    (output_height, output_width, 3), 22, dtype=np.uint8
                )
                for index, (entry, capture) in enumerate(
                    zip(entries, captures, strict=True)
                ):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Failed to read frame {frame_index} from {entry.path}."
                        )
                    tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
                    tile[header_height:] = _fit_frame(
                        frame,
                        tile_width,
                        tile_height - header_height,
                    )
                    _draw_header(tile, entry, len(entries), header_height)
                    row, column = divmod(index, columns)
                    y0 = row * tile_height
                    x0 = column * tile_width
                    grid[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
                writer.write(grid)
                if (frame_index + 1) % 20 == 0 or frame_index + 1 == frame_count:
                    _progress(f"frames {frame_index + 1}/{frame_count}")
        finally:
            writer.release()

        partial.replace(output)
        return frame_count, fps
    finally:
        for capture in captures:
            capture.release()


def main() -> None:
    args = _parse_args()
    entries = _discover(args.input_root, args.run, sort_by=args.sort_by)
    output = args.output or args.input_root / f"{args.run}_ranked_variant_grid.mp4"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    print("Ranking:")
    for entry in entries:
        print(
            f"  {entry.rank:02d}. {entry.variant}: "
            f"{1000.0 * entry.objective:.6f} g"
        )
    frame_count, fps = create_grid_movie(
        entries,
        output,
        columns=args.columns,
        output_width=args.width,
        output_height=args.height,
    )
    ranking = _write_ranking(entries, output)
    print(f"Movie: {output}")
    print(f"Ranking: {ranking}")
    print(f"Frames: {frame_count}; fps: {fps:.3f}")


if __name__ == "__main__":
    main()

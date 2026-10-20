"""The record's README.md, written from its manifest (English, no names)."""

from __future__ import annotations

from .layout import CHECKSUMS, EXPECTED_ARCHIVE, MANIFEST, README, SERIES_ARCHIVE


def _gb(value) -> str:
    """Bytes as decimal GB, or "not recorded"."""
    return "not recorded" if value is None else f"{value / 1e9:.2f} GB"


def _link(value) -> str:
    """A link, or the note that it comes at upload."""
    return value if value else "to be added at upload"


def _seconds(value) -> str:
    """Seconds with one decimal, or "not recorded"."""
    return "not recorded" if value is None else f"{value:.1f}"


def render_readme(manifest: dict) -> str:
    """The README text of a record, from its manifest only."""
    identity = manifest["identity"]
    record = identity["record"]
    runs = manifest["runs"]
    frames = sum(len(run["frames"]) for run in runs)
    code = manifest["code"]
    machine = manifest.get("machine") or {}
    rule = manifest["reproducibility"]["other_machines"]
    lines = [
        f"# {identity['title']}",
        "",
        f"Series `{identity['series_id']}`, record **{record}** "
        f"({'reproduction: raw images as acquired' if record == 'R' else 'simulator products'}), "
        f"tier **{identity['tier']}**: {len(runs)} runs, {frames} frames.",
        "",
        "## Links",
        "",
        f"- Other record of this series: {_link(identity['links'].get('other_record'))}",
        f"- DarSIA code: {_link(identity['links'].get('code'))}",
        f"- Article: {_link(identity['links'].get('article'))}",
        "- Licence and citation: see the record's metadata.",
        "",
        "## Files",
        "",
        f"- `{MANIFEST}`: what this record holds, with every SHA-256 (format `{manifest['format']}`).",
        f"- `{README}`: this file.",
        f"- `{CHECKSUMS}`: SHA-256 of every file (check with `sha256sum --check {CHECKSUMS}`).",
        f"- `{SERIES_ARCHIVE}`: the series folder (configuration, protocols, geometry, calibration, "
        "phase model, execution plan, path bindings and neutral provenance records).",
    ]
    if record == "R":
        lines += [
            f"- `{EXPECTED_ARCHIVE}`: the expected results (masses per frame and the stored coarse layer).",
            "- `raw_<run>.zip`: the run's raw images exactly as acquired (stored, not recompressed).",
        ]
    else:
        lines += ["- `simulator_<run>.zip`: the run's coarse products, static fields and time series."]
    lines += [
        "",
        f"Size: {_gb(manifest['size'].get('bytes'))} in {manifest['size']['files']} files.",
        "",
        "## Unpacking",
        "",
        "Download every file into one empty folder, then unpack "
        + (f"`{SERIES_ARCHIVE}` and `{EXPECTED_ARCHIVE}`" if record == "R" else f"`{SERIES_ARCHIVE}`")
        + " there. The folder is then a DarSIA series folder. "
        + ("The raw archives may stay packed; the reproduction launcher unpacks one run at a time "
           "into its results folder." if record == "R" else ""),
        "",
    ]
    if record == "R":
        lines += [
            "## Reproducing the results",
            "",
            f"1. Install DarSIA at commit `{code['published_commit']}` (a Git checkout of that commit) "
            f"with Python {manifest['environment'].get('python') or '3'} and the pinned packages of "
            f"`{(manifest['environment'].get('lock') or {}).get('path', 'requirements-worker.lock')}`.",
            "2. Run `darsia-reproduce` and choose the unpacked folder and an empty results folder, or run "
            "`darsia-reproduce --dataset <folder> --results <folder>` "
            "(options: `--runs`, `--quick`, `--workers`, `--threads`, `--backend cpu|cuda`).",
            "3. The launcher writes `report.html`, `report.json` and figures to the results folder.",
            "",
            "Alternatively, add the unpacked folder in DarSIA under Start > Add existing series folder. "
            "It opens read-only; results go to a location outside the dataset.",
            "",
            f"Measured on the reference machine: {_seconds(machine.get('seconds_per_frame'))} s per frame "
            f"with {machine.get('threads')} thread(s) per worker, backend {machine.get('backend')}, "
            f"precision {machine.get('precision')}; peak memory per worker {_gb(machine.get('peak_memory_bytes'))}.",
            "",
            "## Reproducibility",
            "",
            "On the reference machine the results are bit-identical. On other machines the following "
            "limits apply per frame:",
            "",
            f"- labels: the changed CO2 area is at most max({rule['labels']['co2_relative'] * 100:g} % of the "
            f"reference CO2 area, {rule['labels']['absolute_cm2']:g} cm²); the changed gas area at most "
            f"max({rule['labels']['gas_relative'] * 100:g} % of the reference gas area, "
            f"{rule['labels']['absolute_cm2']:g} cm²);",
            f"- masses and areas: |a − b| ≤ {rule['masses']['relative']:g}·|a| + "
            f"{rule['masses']['absolute']:g} (g or cm²);",
            f"- maps: per cell {rule['maps']['relative_to_frame_maximum']:g} × the frame's largest value, "
            f"with at most {rule['maps']['allowed_fraction_above'] * 100:g} % of the cells above; cells whose "
            "labels differ count under the label limit instead.",
            "",
            f"Quick check (the first frames of one run): {'valid' if manifest['quick_check_valid'] else 'not valid'} "
            "for this record.",
            "",
        ]
        variation = manifest.get("known_variation") or []
        if variation:
            lines += ["Known variation between machines:", ""]
            lines += [f"- {entry.get('summary', '')}" for entry in variation]
            lines += [""]
    layer = (manifest.get("expected") or manifest.get("simulator") or {}).get("layer") or {}
    units = manifest.get("units") or {}
    lines += [
        "## Units and conventions",
        "",
        f"- Masses in {units.get('masses', 'g')}, areas in {units.get('areas', 'cm2')}, times in hours "
        "since the injection start.",
        f"- Stored grid: {layer.get('resolution_mm', 'n/a')} mm cells, {layer.get('shape', 'n/a')} "
        "(rows, columns); rows run upward in y, columns along x; the origin is the rig's metric origin.",
        "- Labels: " + ", ".join(f"{key} = {value}" for key, value in (units.get("label_legend") or {}).items())
        + ".",
        "",
    ]
    return "\n".join(lines)

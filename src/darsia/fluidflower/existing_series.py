"""Reference-only import helpers for established FluidFlower series.

The import records existing files without copying or modifying source data.  It
also keeps proposed baseline and clear-water references explicitly unapproved.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import uuid

from darsia.fluidflower.phase_learning import (
    PhaseLearningPlanStore,
    phase_learning_plan_from_bundle,
    validate_phase_learning_plan,
)
from darsia.fluidflower.project import FluidFlowerProject, utc_now
from darsia.fluidflower.source_discovery import IMAGE_SUFFIXES


EXISTING_SERIES_IMPORT_FORMAT = "darsia-fluidflower-existing-series-import-v1"
_FFAC_RUN = re.compile(r"^AC(?P<number>\d+)$", re.IGNORECASE)
_FFUM_RUN = re.compile(r"^Run(?P<number>\d+)$", re.IGNORECASE)
_FOLDER_TIMESTAMP = re.compile(r"(?<!\d)(?P<date>\d{6})_(?P<time>\d{6})(?!\d)")


@dataclass(frozen=True)
class ExistingSeriesImportResult:
    """Published project and a concise summary of its reference coverage."""

    project: FluidFlowerProject
    report: dict[str, Any]


def _shallow_images(
    folder: Path,
    inventory: dict[str, list[Path]] | None = None,
) -> list[Path]:
    if not folder.is_dir():
        return []
    key = os.path.normcase(str(folder))
    if inventory is not None and key in inventory:
        return inventory[key]
    with os.scandir(folder) as entries:
        images = sorted(
            (
                Path(entry.path)
                for entry in entries
                if entry.is_file() and Path(entry.name).suffix.casefold() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
    if inventory is not None:
        inventory[key] = images
    return images


def _folder_timestamp(path: Path) -> datetime | None:
    match = _FOLDER_TIMESTAMP.search(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(
            f"{match.group('date')}_{match.group('time')}", "%y%m%d_%H%M%S"
        )
    except ValueError:
        return None


def _phase_folders(
    run_root: Path,
    token: str,
    inventory: dict[str, list[Path]],
) -> list[Path]:
    candidates = [
        folder.resolve()
        for folder in run_root.iterdir()
        if folder.is_dir() and token in folder.name.casefold()
    ]
    populated = [(folder, _shallow_images(folder, inventory)) for folder in candidates]
    populated = [(folder, images) for folder, images in populated if images]

    # Some runs contain two names for the same acquisition start. Keep the
    # directory with the most frames instead of counting duplicate acquisitions.
    by_timestamp: dict[datetime | str, tuple[Path, list[Path]]] = {}
    for folder, images in populated:
        key: datetime | str = _folder_timestamp(folder) or folder.name.casefold()
        previous = by_timestamp.get(key)
        if previous is None or len(images) > len(previous[1]):
            by_timestamp[key] = (folder, images)
    return sorted(
        (folder for folder, _ in by_timestamp.values()),
        key=lambda path: (_folder_timestamp(path) or datetime.max, path.name.casefold()),
    )


def _reference_folder(
    folders: Sequence[Path],
    *,
    injection_start: datetime | None,
    latest: bool,
) -> Path | None:
    if not folders:
        return None
    before_injection = [
        folder
        for folder in folders
        if injection_start is None
        or _folder_timestamp(folder) is None
        or _folder_timestamp(folder) <= injection_start
    ]
    if not before_injection:
        return None
    dated = sorted(
        before_injection,
        key=lambda path: (_folder_timestamp(path) or datetime.min, path.name.casefold()),
    )
    return dated[-1] if latest else dated[0]


def _first_or_last_image(
    folder: Path | None,
    *,
    last: bool = False,
    inventory: dict[str, list[Path]] | None = None,
) -> str:
    images = _shallow_images(folder, inventory) if folder is not None else []
    if not images:
        return ""
    return str(images[-1] if last else images[0])


def _protocol_workbook(run_root: Path) -> str:
    workbooks = sorted(
        path.resolve()
        for path in run_root.iterdir()
        if path.is_file()
        and path.suffix.casefold() in {".xls", ".xlsx"}
        and not path.name.startswith("~$")
    )
    preferred = [path for path in workbooks if "protocol" in path.name.casefold()]
    selected = preferred or workbooks
    return str(selected[0]) if selected else ""


def discover_ffac_series(
    raw_root: Path,
    *,
    overflow_review_root: Path | None = None,
    normalized_protocol_root: Path | None = None,
    minimum_run_number: int = 17,
) -> list[dict[str, Any]]:
    """Discover FF_AC production runs without traversing unrelated image folders."""

    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    review_root = (
        Path(overflow_review_root).expanduser().resolve()
        if overflow_review_root is not None
        else None
    )
    protocol_root = (
        Path(normalized_protocol_root).expanduser().resolve()
        if normalized_protocol_root is not None
        else None
    )
    runs: list[tuple[int, Path]] = []
    for folder in root.iterdir():
        match = _FFAC_RUN.fullmatch(folder.name)
        if folder.is_dir() and match is not None:
            number = int(match.group("number"))
            if number >= int(minimum_run_number):
                runs.append((number, folder.resolve()))

    rows: list[dict[str, Any]] = []
    inventory: dict[str, list[Path]] = {}
    for number, run_root in sorted(runs):
        run_id = f"ac{number}"
        injection_folders = _phase_folders(run_root, "injection", inventory)
        resting_folders = _phase_folders(run_root, "rest", inventory)
        injection_start = min(
            (
                timestamp
                for timestamp in map(_folder_timestamp, injection_folders)
                if timestamp is not None
            ),
            default=None,
        )
        flush_folders = _phase_folders(run_root, "flush", inventory)
        wash_folders = _phase_folders(run_root, "wash", inventory)
        baseline_folder = _reference_folder(
            flush_folders, injection_start=injection_start, latest=True
        )
        clearwater_folder = _reference_folder(
            wash_folders, injection_start=injection_start, latest=False
        ) or _reference_folder(
            flush_folders, injection_start=injection_start, latest=False
        )
        experiment_folders = [*injection_folders, *resting_folders]
        experiment_images = [
            image
            for folder in experiment_folders
            for image in _shallow_images(folder, inventory)
        ]
        overflow_folders = sorted(
            (
                folder.resolve()
                for folder in run_root.iterdir()
                if folder.is_dir() and "overflow" in folder.name.casefold()
            ),
            key=lambda path: path.name.casefold(),
        )
        review = (
            review_root / run_id / f"{run_id}_overflow_scale_review.json"
            if review_root is not None
            else None
        )
        normalized = protocol_root / run_id if protocol_root is not None else None
        rows.append(
            {
                "run_id": run_id,
                "experiment_folder": str(run_root),
                "experiment_folders": [str(path) for path in experiment_folders],
                "injection_folders": [str(path) for path in injection_folders],
                "resting_folders": [str(path) for path in resting_folders],
                "image_count": len(experiment_images),
                "first_image": str(experiment_images[0]) if experiment_images else "",
                "last_image": str(experiment_images[-1]) if experiment_images else "",
                "baseline_candidate": _first_or_last_image(
                    baseline_folder, last=True, inventory=inventory
                ),
                "clearwater_candidate": _first_or_last_image(
                    clearwater_folder, inventory=inventory
                ),
                "raw_protocol_workbook": _protocol_workbook(run_root),
                "normalized_protocol_folder": (
                    str(normalized) if normalized is not None and normalized.is_dir() else ""
                ),
                "overflow_image_folders": [str(path) for path in overflow_folders],
                "overflow_review": str(review) if review is not None and review.is_file() else "",
                "candidate_status": "requires_user_confirmation",
                "mapping_status": (
                    "ready_for_review"
                    if injection_folders and resting_folders
                    else "incomplete_source_mapping"
                ),
            }
        )
    return rows


def discover_ffum_series(
    experiment_root: Path,
    *,
    clearwater_root: Path | None = None,
    configuration_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Discover flat FF_UM run folders and optional clear-water references."""

    root = Path(experiment_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    clear_root = (
        Path(clearwater_root).expanduser().resolve()
        if clearwater_root is not None
        else None
    )
    config_root = (
        Path(configuration_root).expanduser().resolve()
        if configuration_root is not None
        else None
    )
    runs: list[tuple[int, Path]] = []
    for folder in root.iterdir():
        match = _FFUM_RUN.fullmatch(folder.name)
        if folder.is_dir() and match is not None:
            runs.append((int(match.group("number")), folder.resolve()))

    rows: list[dict[str, Any]] = []
    inventory: dict[str, list[Path]] = {}
    for number, run_root in sorted(runs):
        run_id = f"run{number}"
        images = _shallow_images(run_root, inventory)
        clearwater_folder = clear_root / run_id if clear_root is not None else None
        config = config_root / f"{run_id}.toml" if config_root is not None else None
        rows.append(
            {
                "run_id": run_id,
                "experiment_folder": str(run_root),
                "experiment_folders": [str(run_root)],
                "image_count": len(images),
                "first_image": str(images[0]) if images else "",
                "last_image": str(images[-1]) if images else "",
                "baseline_candidate": str(images[0]) if images else "",
                "clearwater_candidate": _first_or_last_image(
                    clearwater_folder, inventory=inventory
                ),
                "run_configuration": (
                    str(config) if config is not None and config.is_file() else ""
                ),
                "candidate_status": "requires_user_confirmation",
                "mapping_status": "ready_for_review" if images else "no_images_found",
            }
        )
    return rows


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "first_image",
        "last_image",
        "baseline_candidate",
        "clearwater_candidate",
        "raw_protocol_workbook",
        "overflow_review",
        "run_configuration",
    )
    return {
        "run_count": len(rows),
        "experiment_image_count": sum(int(row.get("image_count") or 0) for row in rows),
        "reference_counts": {
            field: sum(bool(str(row.get(field) or "")) for row in rows) for field in fields
        },
        "incomplete_runs": [
            str(row.get("run_id") or "")
            for row in rows
            if str(row.get("mapping_status") or "")
            not in {"ready_for_review"}
        ],
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def create_existing_series_project(
    root: Path,
    *,
    name: str,
    series_id: str,
    template_id: str,
    rig_type: str,
    runs: Sequence[Mapping[str, Any]],
    protocol_path: Path | None,
    source_paths: Mapping[str, Path | None],
    protocol_organization: str,
    overflow_mode: str = "none",
    overflow_path: Path | None = None,
    shared_phase_model: Path | None = None,
    created_by: str = "",
    notes: str = "",
) -> ExistingSeriesImportResult:
    """Create one project whose registered external sources remain read-only."""

    destination = Path(root).expanduser().resolve()
    project = FluidFlowerProject.create(
        destination,
        name=name,
        series_id=series_id,
        template_id=template_id,
        rig_type=rig_type,
        protocol_path=protocol_path,
        image_sources=source_paths,
        overflow_mode=overflow_mode,
        overflow_path=overflow_path,
        created_by=created_by,
    )
    mappings = [dict(row) for row in runs]
    project.set_runs(
        [
            {
                "run_id": str(row.get("run_id") or ""),
                "label": str(row.get("run_id") or "").upper(),
                "indicator_state": "BTB",
                "source_mapping": dict(row),
            }
            for row in mappings
        ],
        updated_by=created_by,
        reason="Imported established series by read-only source reference.",
    )
    project.update_metadata(
        "protocol",
        {"organization": protocol_organization},
        updated_by=created_by,
        reason="Record the established protocol layout.",
    )
    project.update_metadata(
        "chemistry",
        {
            "indicator_system": "BTB",
            "notes": notes,
        },
        updated_by=created_by,
        reason="Record the imported series chemistry boundary.",
    )
    report = {
        "schema": 1,
        "format": EXISTING_SERIES_IMPORT_FORMAT,
        "created_utc": utc_now(),
        "created_by": created_by,
        "series_id": series_id,
        "project_root": ".",
        "mode": "reference_only",
        "source_files_modified": False,
        "existing_results_are_reference_only": True,
        "reference_candidates_approved": False,
        "coverage": _coverage(mappings),
    }
    source_mapping_index = dict(
        (project.manifest.get("tools") or {}).get("source_mapping") or {}
    )
    source_mapping_index.update(
        {
            "structure": "established_series_adapter",
            "baseline_and_clearwater_candidates_approved": False,
        }
    )
    project.update_metadata(
        "tools",
        {
            "source_mapping": source_mapping_index,
            "existing_series_import": report,
            "shared_phase_model": (
                {
                    "path": str(Path(shared_phase_model).expanduser().resolve()),
                    "scope": "visible_registered_camera_plane_2d_only",
                    "mass_conversion_is_series_specific": True,
                }
                if shared_phase_model is not None
                else {}
            ),
        },
        updated_by=created_by,
        reason="Register source mapping and prior work without promoting legacy outputs.",
    )

    if shared_phase_model is not None:
        model_root = Path(shared_phase_model).expanduser().resolve()
        evidence = sorted((model_root / "input_snapshot").glob("*.json"))
        plan = phase_learning_plan_from_bundle(model_root, annotation_files=evidence)
        validation = validate_phase_learning_plan(plan)
        PhaseLearningPlanStore(project).save_revision(
            plan,
            actor=created_by,
            reason="Register the reviewed shared FF_AC/FF_UM optical phase model.",
            validation=validation,
        )

    report_path = project.root / "existing_series_import.json"
    readme_path = project.root / "README_IMPORT.md"
    _atomic_text(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(
        readme_path,
        "\n".join(
            (
                f"# {name}",
                "",
                "This project imports an established FluidFlower series by reference.",
                "Original images, protocols, annotations, and results are read-only and",
                "were not copied, moved, renamed, or modified during import.",
                "",
                "Baseline and clear-water paths are proposals that must be reviewed in",
                "the application before image correction or calibration is approved.",
                "Legacy calibration and analysis outputs remain provenance references;",
                "they are not treated as results produced by this project.",
                "",
            )
        ),
    )
    return ExistingSeriesImportResult(project=project, report=report)


__all__ = [
    "EXISTING_SERIES_IMPORT_FORMAT",
    "ExistingSeriesImportResult",
    "create_existing_series_project",
    "discover_ffac_series",
    "discover_ffum_series",
]

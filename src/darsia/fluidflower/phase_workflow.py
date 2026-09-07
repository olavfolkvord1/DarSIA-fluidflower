"""Bridges for the mature phase-review UI and cross-series model trainer.

The Qt FluidFlower application owns configuration and provenance.  Until the
large review canvas is moved into the library package, this module starts the
versioned companion programs without a shell and with explicit input/output
paths.  Source evidence is never modified.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from darsia.fluidflower.project import sha256_file

PHASE_TRAINING_REQUEST_FORMAT = "darsia-fluidflower-phase-training-request-v1"
PHASE_REVIEW_SESSION_FORMAT = "fluidflower-local-review-session-v1"


def phase_python_executable(*, gui: bool = False) -> str:
    """Return the active environment's interpreter for companion processes.

    Repository GUI launchers can bootstrap site-packages into a base CPython
    process. In that case ``sys.executable`` points outside the virtual
    environment even though DarSIA itself is running correctly. Companion
    review and training processes must therefore prefer ``VIRTUAL_ENV``.
    """

    configured = str(os.environ.get("DARSIA_FLUIDFLOWER_PYTHON") or "").strip()
    if configured:
        executable = Path(configured).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(
                f"Configured DarSIA Python executable is unavailable: {executable}"
            )
        return str(executable)

    virtual_environment = str(os.environ.get("VIRTUAL_ENV") or "").strip()
    if virtual_environment:
        root = Path(virtual_environment).expanduser().resolve()
        if os.name == "nt":
            names = ("pythonw.exe", "python.exe") if gui else ("python.exe",)
            candidates = tuple(root / "Scripts" / name for name in names)
        else:
            candidates = (root / "bin" / "python",)
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    current = Path(sys.executable).resolve()
    if gui and os.name == "nt":
        windowless = current.with_name("pythonw.exe")
        if windowless.is_file():
            return str(windowless)
    return str(current)


def default_phase_review_roots() -> tuple[Path, ...]:
    """Return local review roots without probing campaign/NAS paths."""

    candidates: list[Path] = []
    configured = str(os.environ.get("DARSIA_FLUIDFLOWER_REVIEW_ROOT") or "").strip()
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path.home() / "Documents" / "FluidFlower_Review_Local")
    if sys.platform == "darwin":
        candidates.append(
            Path.home()
            / "Library"
            / "Application Support"
            / "FluidFlower Review"
            / "reviews"
        )
    else:
        candidates.append(
            Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
            / "FluidFlower Review"
            / "reviews"
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        identity = str(resolved).casefold()
        if identity not in seen:
            unique.append(resolved)
            seen.add(identity)
    return tuple(unique)


def _review_counts(decision_path: Path) -> dict[str, int]:
    counts = {
        "reviewed_experiments": 0,
        "manual_regions": 0,
        "blind_region_labels": 0,
        "frame_verdicts": 0,
    }
    try:
        payload = json.loads(decision_path.read_text(encoding="utf-8-sig"))
        decisions = payload.get("decisions") or {}
    except (OSError, TypeError, ValueError):
        return counts
    if not isinstance(decisions, Mapping):
        return counts
    rows = [row for row in decisions.values() if isinstance(row, Mapping)]
    counts["reviewed_experiments"] = len(rows)
    counts["manual_regions"] = sum(len(row.get("annotations") or []) for row in rows)
    counts["blind_region_labels"] = sum(
        len(row.get("phase_quiz_answers") or []) for row in rows
    )
    counts["frame_verdicts"] = sum(len(row.get("frame_choices") or []) for row in rows)
    return counts


def _review_selected_candidates(decision_path: Path) -> dict[str, str]:
    """Recover an unambiguous single-model binding from legacy decisions."""

    try:
        payload = json.loads(decision_path.read_text(encoding="utf-8-sig"))
        decisions = payload.get("decisions") or {}
    except (OSError, TypeError, ValueError):
        return {}
    if not isinstance(decisions, Mapping):
        return {}
    selected: dict[str, str] = {}
    for run, raw in decisions.items():
        if not isinstance(raw, Mapping):
            continue
        candidate_a = str((raw.get("candidate_a") or {}).get("candidate_id") or "")
        candidate_b = str((raw.get("candidate_b") or {}).get("candidate_id") or "")
        comparison_mode = str(raw.get("comparison_mode") or "")
        if candidate_a and (
            candidate_a == candidate_b or comparison_mode == "single_candidate"
        ):
            selected[str(run).casefold()] = candidate_a
    return selected


def _local_review_root(session: Mapping[str, Any], decision_dir: Path) -> Path:
    """Recover the reviewer-independent save root from a saved session."""

    explicit = str(session.get("local_review_root") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    expected = (
        str(session.get("reviewer_id") or "").casefold(),
        str(session.get("campaign_id") or "").casefold(),
        str(session.get("workspace") or "").casefold(),
        "manual_pairwise_review",
    )
    parts = tuple(part.casefold() for part in decision_dir.parts[-4:])
    if all(expected) and parts == expected:
        return decision_dir.parents[3]
    return decision_dir.parent


def discover_phase_review_sessions(
    roots: Iterable[Path] | None = None,
) -> list[dict[str, Any]]:
    """Discover resumable local review sessions without reading image assets."""

    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_root in roots or default_phase_review_roots():
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in root.rglob("review_session.json"):
            identity = str(path.resolve()).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, TypeError, ValueError):
                continue
            if (
                not isinstance(payload, Mapping)
                or payload.get("format") != PHASE_REVIEW_SESSION_FORMAT
            ):
                continue
            decision_dir = (
                Path(str(payload.get("decision_dir") or path.parent))
                .expanduser()
                .resolve()
            )
            decision_path = decision_dir / "pairwise_review_decisions.json"
            modified_timestamp = float(path.stat().st_mtime)
            if decision_path.is_file():
                modified_timestamp = max(
                    modified_timestamp,
                    float(decision_path.stat().st_mtime),
                )
            sessions.append(
                {
                    **dict(payload),
                    "session_path": str(path.resolve()),
                    "decision_path": str(decision_path),
                    "local_review_root": str(_local_review_root(payload, decision_dir)),
                    "modified_timestamp": modified_timestamp,
                    "selected_candidates": dict(
                        payload.get("selected_candidates")
                        or _review_selected_candidates(decision_path)
                    ),
                    **_review_counts(decision_path),
                }
            )
    return sorted(
        sessions,
        key=lambda row: (
            float(row.get("modified_timestamp") or 0.0),
            str(row.get("session_path") or ""),
        ),
        reverse=True,
    )


def _find_companion(name: str, explicit: Path | None = None) -> Path:
    if explicit is not None and str(explicit):
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    environment = os.environ.get(
        "DARSIA_FLUIDFLOWER_REVIEW_SCRIPT"
        if name == "calibration_review_desktop.py"
        else "DARSIA_FLUIDFLOWER_PHASE_TRAINER"
    )
    if environment:
        path = Path(environment).expanduser().resolve()
        if path.is_file():
            return path
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"The FluidFlower companion program {name!r} could not be located"
    )


def _phase_trainer_prefix(program: Path | None = None) -> list[str]:
    """Resolve an explicit legacy trainer or the installed DarSIA module."""

    if program is not None and str(program):
        return [
            phase_python_executable(),
            str(_find_companion("phase-trainer", program)),
        ]
    environment = os.environ.get("DARSIA_FLUIDFLOWER_PHASE_TRAINER")
    if environment:
        path = Path(environment).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [phase_python_executable(), str(path)]
    return [
        phase_python_executable(),
        "-m",
        "darsia.fluidflower.phase_training.trainer",
    ]


def review_campaign_root(value: Path) -> Path:
    """Resolve a prepared review root containing a postflight manifest."""

    path = Path(value).expanduser().resolve()
    root = path.parent if path.is_file() else path
    manifest = root / "postflight_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Prepared review manifest is unavailable: {manifest}")
    return root


def build_phase_review_command(
    *,
    mode: str,
    reviewer: str,
    save_root: Path,
    cache_root: Path,
    campaign_root: Path | None = None,
    ffac_root: Path | None = None,
    ffum_root: Path | None = None,
    combined_root: Path | None = None,
    selected_candidates: Mapping[str, str] | None = None,
    target_model_id: str = "",
    review_session_id: str = "",
    program: Path | None = None,
    use_gpu: bool = True,
) -> list[str]:
    """Build a shell-free command for one review purpose."""

    purpose = str(mode).strip().casefold()
    if purpose not in {
        "active_learning",
        "blind_region_labeling",
        "model_validation",
        "open_correction",
    }:
        raise ValueError(f"Unsupported phase-review mode: {mode!r}")
    actor = str(reviewer).strip()
    if not actor:
        raise ValueError("Reviewer name is required")
    fallback = campaign_root
    if fallback is None and any(
        value is None for value in (ffac_root, ffum_root, combined_root)
    ):
        raise ValueError(
            "Choose either one campaign root or explicit FF_AC, FF_UM, and "
            "combined review roots"
        )
    resolved_ffac = review_campaign_root(ffac_root or fallback)  # type: ignore[arg-type]
    resolved_ffum = review_campaign_root(ffum_root or fallback)  # type: ignore[arg-type]
    resolved_combined = review_campaign_root(  # type: ignore[arg-type]
        combined_root or fallback
    )
    executable = _find_companion("calibration_review_desktop.py", program)
    workspace = (
        "training"
        if purpose in {"active_learning", "blind_region_labeling"}
        else "current_comparison"
    )
    review_surface = {
        "blind_region_labeling": "blind_regions",
        "open_correction": "open_correction",
    }.get(purpose, "default")
    command = [
        phase_python_executable(gui=True),
        str(executable),
        "--workspace",
        workspace,
        "--ffac-root",
        str(resolved_ffac),
        "--ffum-root",
        str(resolved_ffum),
        "--combined-root",
        str(resolved_combined),
        "--reviewer",
        actor,
        "--local-review-root",
        str(Path(save_root).expanduser().resolve()),
        "--cache-root",
        str(Path(cache_root).expanduser().resolve()),
        "--no-sync-manifest",
        "--online",
        "--single-model",
    ]
    if str(target_model_id).strip():
        command.extend(("--target-model-id", str(target_model_id).strip()))
    if str(review_session_id).strip():
        command.extend(("--review-session-id", str(review_session_id).strip()))
    if review_surface != "default":
        command.extend(("--review-surface", review_surface))
    if purpose == "model_validation":
        command.append("--approval-only")
    for run_id, candidate_id in sorted((selected_candidates or {}).items()):
        if str(run_id).strip() and str(candidate_id).strip():
            command.extend(
                (
                    "--single-candidate",
                    f"{str(run_id).strip()}={str(candidate_id).strip()}",
                )
            )
    if not use_gpu:
        command.append("--no-gpu")
    return command


def build_phase_review_resume_command(
    session: Mapping[str, Any],
    *,
    cache_root: Path,
    program: Path | None = None,
    use_gpu: bool = True,
    read_only: bool = False,
) -> list[str]:
    """Resume an existing session with its original workspace and save path."""

    workspace = str(session.get("workspace") or "").strip().casefold()
    if workspace not in {"current_comparison", "training", "validation", "comparison"}:
        raise ValueError(f"Unsupported saved review workspace: {workspace!r}")
    reviewer = str(session.get("reviewer") or "").strip()
    if not reviewer:
        raise ValueError("The saved review session has no reviewer identity")
    ffac_root = review_campaign_root(Path(str(session.get("ffac_root") or "")))
    ffum_root = review_campaign_root(Path(str(session.get("ffum_root") or "")))
    combined_root = review_campaign_root(Path(str(session.get("combined_root") or "")))
    save_root = Path(str(session.get("local_review_root") or "")).expanduser().resolve()
    executable = _find_companion("calibration_review_desktop.py", program)
    command = [
        phase_python_executable(gui=True),
        str(executable),
        "--workspace",
        workspace,
        "--ffac-root",
        str(ffac_root),
        "--ffum-root",
        str(ffum_root),
        "--combined-root",
        str(combined_root),
        "--reviewer",
        reviewer,
        "--local-review-root",
        str(save_root),
        "--cache-root",
        str(Path(cache_root).expanduser().resolve()),
        "--no-sync-manifest",
        "--online",
    ]
    if bool(session.get("single_model")):
        command.append("--single-model")
    target_model_id = str(session.get("target_model_id") or "").strip()
    if target_model_id:
        command.extend(("--target-model-id", target_model_id))
    review_session_id = str(session.get("review_session_id") or "").strip()
    if review_session_id:
        command.extend(("--review-session-id", review_session_id))
    for run_id, candidate_id in sorted(
        dict(session.get("selected_candidates") or {}).items()
    ):
        if str(run_id).strip() and str(candidate_id).strip():
            command.extend(
                ("--single-candidate", f"{str(run_id).strip()}={str(candidate_id).strip()}")
            )
    review_surface = str(session.get("review_surface") or "default").strip()
    if review_surface in {"blind_regions", "open_correction"}:
        command.extend(("--review-surface", review_surface))
    if not use_gpu:
        command.append("--no-gpu")
    if read_only:
        command.append("--read-only")
    return command


def lock_phase_review_sessions(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Lock live review sessions once their evidence is frozen for one model."""

    model_id = str(request.get("model_id") or "").strip()
    request_path = Path(str(request.get("request_path") or "")).expanduser()
    if not model_id:
        raise ValueError("Training request has no model ID")
    locked: list[dict[str, Any]] = []
    for bundle in request.get("bundles") or []:
        if not isinstance(bundle, Mapping):
            continue
        decision_record = bundle.get("decisions") or {}
        if not isinstance(decision_record, Mapping):
            continue
        decision_path = Path(
            str(decision_record.get("source_path") or "")
        ).expanduser()
        session_path = decision_path.parent / "review_session.json"
        if decision_path.name != "pairwise_review_decisions.json" or not session_path.is_file():
            continue
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8-sig"))
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(payload, Mapping) or payload.get("format") != PHASE_REVIEW_SESSION_FORMAT:
            continue
        mutable = dict(payload)
        locks = [
            dict(row)
            for row in mutable.get("training_locks") or []
            if isinstance(row, Mapping)
        ]
        if not any(str(row.get("model_id") or "") == model_id for row in locks):
            current_sha = sha256_file(decision_path) if decision_path.is_file() else ""
            frozen_sha = str(decision_record.get("sha256") or "")
            locks.append(
                {
                    "model_id": model_id,
                    "training_request": str(request_path.resolve()) if request_path else "",
                    "locked_utc": datetime.now(UTC).isoformat(),
                    "source_decisions_sha256": current_sha,
                    "frozen_decisions_sha256": frozen_sha,
                    "source_matches_frozen_revision": bool(
                        current_sha and frozen_sha and current_sha == frozen_sha
                    ),
                }
            )
        mutable["editing_locked"] = True
        mutable["training_locks"] = locks
        mutable["updated_utc"] = datetime.now(UTC).isoformat()
        temporary = session_path.with_name(f".{session_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(mutable, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, session_path)
        locked.append(
            {
                "session_path": str(session_path.resolve()),
                "decision_path": str(decision_path.resolve()),
                "model_id": model_id,
            }
        )
    return locked


def parse_training_bundles(value: str | Sequence[str]) -> list[tuple[Path, Path]]:
    """Parse `decisions | combined manifest` rows from the training UI."""

    rows = value.splitlines() if isinstance(value, str) else list(value)
    bundles: list[tuple[Path, Path]] = []
    for line_number, raw in enumerate(rows, start=1):
        text = str(raw).strip()
        if not text:
            continue
        parts = [part.strip() for part in text.replace("\t", "|").split("|")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Training row {line_number} must be DECISIONS_JSON | COMBINED_MANIFEST_JSON"
            )
        decisions, manifest = (Path(part).expanduser().resolve() for part in parts)
        if not decisions.is_file():
            raise FileNotFoundError(decisions)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        bundles.append((decisions, manifest))
    if not bundles:
        raise ValueError("At least one frozen review bundle is required")
    return bundles


def prepare_phase_training_request(
    *,
    output_parent: Path,
    cache_parent: Path,
    bundles: Sequence[tuple[Path, Path]],
    actor: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze training inputs into a new immutable model-version directory."""

    settings = dict(settings)
    from darsia.fluidflower.phase_training.radiometry import resolve_profile
    radiometry_value = str(settings.get("radiometry_profile") or "legacy")
    reference_image = Path(radiometry_value[6:]) if radiometry_value.startswith("image:") else None
    if reference_image is not None and not reference_image.is_file():
        raise FileNotFoundError("Choose an existing common ColorChecker reference image")
    radiometry_profile = None if reference_image is not None else resolve_profile(radiometry_value)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_id = f"phase-model-{timestamp}-{uuid.uuid4().hex[:8]}"
    output_root = Path(output_parent).expanduser().resolve() / model_id
    cache_root = Path(cache_parent).expanduser().resolve() / model_id
    output_root.mkdir(parents=True, exist_ok=False)
    snapshot_root = output_root / "input_snapshot"
    snapshot_root.mkdir()
    if reference_image is not None:
        destination = snapshot_root / ("common_color_reference" + reference_image.suffix)
        shutil.copy2(reference_image, destination)
        if sha256_file(destination) != sha256_file(reference_image):
            raise OSError("Color-reference snapshot verification failed")
        settings["radiometry_profile"] = "image:" + str(destination)
        settings["radiometry_reference_source"] = {
            "path": str(reference_image), "sha256": sha256_file(destination)
        }
    if radiometry_profile is not None:
        from darsia_extensions.phase_radiometry import save_profile
        profile_path = snapshot_root / "radiometry_profile.json"
        save_profile(profile_path, radiometry_profile)
        settings["radiometry_profile"] = str(profile_path)
    frozen: list[dict[str, Any]] = []
    frozen_pairs: list[tuple[Path, Path]] = []
    for index, (decisions_source, manifest_source) in enumerate(bundles, start=1):
        pair: list[Path] = []
        records: dict[str, Any] = {"bundle_index": index}
        for role, source in (
            ("decisions", Path(decisions_source).resolve()),
            ("combined_manifest", Path(manifest_source).resolve()),
        ):
            suffix = source.suffix or ".json"
            destination = snapshot_root / f"{index:02d}_{role}{suffix}"
            shutil.copy2(source, destination)
            source_sha = sha256_file(source)
            destination_sha = sha256_file(destination)
            if source_sha != destination_sha:
                raise OSError(f"Snapshot hash mismatch for {source}")
            records[role] = {
                "source_path": str(source),
                "snapshot_path": str(destination),
                "size_bytes": int(destination.stat().st_size),
                "sha256": destination_sha,
            }
            pair.append(destination)
        frozen.append(records)
        frozen_pairs.append((pair[0], pair[1]))
    request = {
        "schema": 1,
        "format": PHASE_TRAINING_REQUEST_FORMAT,
        "model_id": model_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "created_by": str(actor).strip(),
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "settings": dict(settings),
        "bundles": frozen,
    }
    request_path = output_root / "training_request.json"
    temporary = request_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, request_path)
    request["request_path"] = str(request_path)
    request["frozen_pairs"] = frozen_pairs
    request["locked_review_sessions"] = lock_phase_review_sessions(request)
    persisted = {
        key: value
        for key, value in request.items()
        if key not in {"request_path", "frozen_pairs"}
    }
    temporary = request_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(persisted, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, request_path)
    return request


def build_phase_training_command(
    request: Mapping[str, Any],
    *,
    program: Path | None = None,
) -> list[str]:
    """Build the current cross-series hierarchical training command."""

    settings = dict(request.get("settings") or {})
    command = [
        *_phase_trainer_prefix(program),
        "--campaign-id",
        str(request["model_id"]),
        "--output-root",
        str(request["output_root"]),
        "--cache-root",
        str(request["cache_root"]),
        "--fit-mode",
        str(settings.get("fit_mode") or "mil"),
        "--heads",
        "co2_presence",
        "gas_within_co2",
        "--backend",
        str(settings.get("backend") or "auto"),
        "--stage-workers",
        str(max(1, int(settings.get("stage_workers") or 4))),
        "--extraction-workers",
        str(max(1, int(settings.get("extraction_workers") or 4))),
        "--opencv-threads",
        str(max(1, int(settings.get("opencv_threads") or 2))),
        "--pixels-per-region",
        str(max(64, int(settings.get("pixels_per_region") or 1024))),
        "--mil-max-iters",
        str(max(10, int(settings.get("mil_max_iters") or 160))),
        "--resume-completed-heads",
    ]
    if bool(settings.get("change_evidence_ablation", False)):
        command.append("--change-evidence-ablation")
    if settings.get("radiometry_profile"):
        command.extend(("--radiometry-profile", str(settings["radiometry_profile"])))
    if settings.get("series"):
        command.extend(("--series", *settings["series"]))
    for decisions, manifest in request.get("frozen_pairs") or []:
        command.extend(("--bundle", str(decisions), str(manifest)))
    return command


__all__ = [
    "PHASE_REVIEW_SESSION_FORMAT",
    "PHASE_TRAINING_REQUEST_FORMAT",
    "build_phase_review_command",
    "build_phase_review_resume_command",
    "build_phase_training_command",
    "default_phase_review_roots",
    "discover_phase_review_sessions",
    "lock_phase_review_sessions",
    "parse_training_bundles",
    "phase_python_executable",
    "prepare_phase_training_request",
    "review_campaign_root",
]

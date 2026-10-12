"""Isolated, read-only offscreen QA of the real DarSIA-dev Studio viewer.

The probe opens only AC28 in a new Qt process, using the selected candidate's
immutable validation jobs.  It never attaches to an existing user window or
creates annotations.  Its JPEGs are presentation assets, not numerical input.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from PIL import Image

from darsia.fluidflower.project import sha256_file, utc_now


FORMAT = "ffac-track1-display-mask-rerun-studio-qa-v1"
PROBE_TIMEOUT_SECONDS = 180
RUN = "ac28"
_PATH_FIELDS = (
    "candidate_folder", "review_output_dir", "analysis_source_review_root",
    "phase_map_output_dir", "raw_phase_map_output_dir", "presence_diagnostics_dir",
    "phase_quiz_proposals_path", "summary", "mass_plot", "contour_movie", "per_frame_csv",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _prepare_selection(candidate: Path, output_dir: Path) -> tuple[Path, str, str]:
    """Copy only AC28 selection metadata; all input paths remain source-bound."""
    from darsia.fluidflower.phase_validation import _model_identity

    model_id, model_sha = _model_identity(candidate)
    validation = candidate / "validation"
    original = validation / "postflight_manifest.json"
    source = _read(original)
    jobs = [dict(job) for job in source.get("jobs", [])
            if str(job.get("run") or "").casefold() == RUN]
    if len(jobs) != 1 or not jobs[0].get("optical_model_sha256"):
        raise ValueError("Studio QA needs exactly one native AC28 candidate job")
    job = jobs[0]
    for key in _PATH_FIELDS:
        if job.get(key):
            path = Path(job[key])
            job[key] = str(path.resolve() if path.is_absolute()
                           else (validation / path).resolve())
    if not job.get("candidate_folder"):
        job["candidate_folder"] = str(validation / "postflight" / job["variant"] / RUN)
    root = output_dir / "studio_probe" / "selection"
    payload = dict(source, jobs=jobs)
    _write_new(root / "postflight_manifest.json", payload)
    _write_new(root / "qa_source_binding.json", {
        "candidate": str(candidate), "model_id": model_id, "model_sha256": model_sha,
        "source_manifest": str(original), "source_manifest_sha256": sha256_file(original),
        "run": RUN, "selection_only": True, "original_job_unchanged": True,
    })
    return root, model_id, model_sha


def _asset_check(main: Path, detail: Path, manifest: dict[str, Any],
                 index: int) -> dict[str, Any]:
    record = manifest["frame_sources"][index]
    main_sha, detail_sha = sha256_file(main), sha256_file(detail)
    with Image.open(main) as image:
        main_wh = list(image.size)
    with Image.open(detail) as image:
        detail_wh = list(image.size)
    expected_wh = list(manifest["canvas_wh"])
    checks = {
        "main_sha_matches_manifest": main_sha == record.get("native_jpeg_sha256"),
        "detail_sha_matches_manifest": detail_sha == record.get("v5_raw_detail_jpeg_sha256"),
        "main_detail_have_distinct_sha": main_sha != detail_sha,
        "main_native_canvas": main_wh == expected_wh,
        "detail_native_canvas": detail_wh == expected_wh,
    }
    return {"main_path": str(main), "main_sha256": main_sha,
            "raw_detail_path": str(detail), "raw_detail_sha256": detail_sha,
            "main_wh": main_wh, "raw_detail_wh": detail_wh,
            "expected_canvas_wh": expected_wh, "checks": checks,
            "passed": all(checks.values())}


def _qt_image_sha(image: Any, qt: Any) -> str:
    image = image.convertToFormat(qt.QImage.Format.Format_RGBA8888)
    digest = hashlib.sha256()
    digest.update(f"{image.width()}x{image.height()}|RGBA8888".encode("ascii"))
    digest.update(bytes(image.constBits()))
    return digest.hexdigest()


def _probe(candidate: Path, output_dir: Path) -> dict[str, Any]:
    # Import Qt only inside the isolated process. Existing Studio processes and
    # their QApplication/annotation state cannot be reached by this probe.
    from darsia.fluidflower.tools import calibration_review_desktop as desktop

    selection, model_id, model_sha = _prepare_selection(candidate, output_dir)
    args = desktop.build_parser().parse_args([
        "--workspace", "validation", "--online", "--no-sync-manifest", "--read-only",
        "--single-model", "--target-model-id", model_id,
        "--validation-ffac-root", str(candidate / "validation"),
        "--validation-ffum-root", str(selection / "unused_ffum"),
        "--validation-combined-root", str(selection), "--validation-ffac-runs", RUN,
        "--reviewer", "Read-only rerun QA", "--no-gpu",
        "--cache-root", str(output_dir / "studio_probe" / "cache"),
        "--local-review-root", str(output_dir / "studio_probe" / "decisions"),
    ])
    args.offline_bundle = None  # --online is normally applied by main().
    workspace = replace(desktop.review_workspaces(args)["validation"],
                        decision_dir=output_dir / "studio_probe" / "decisions")
    app = desktop.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    model, store = desktop.build_workspace_model(args, workspace, reviewer_name=args.reviewer)
    if not store.is_read_only or set(model.pairs) != {RUN}:
        raise ValueError("Studio QA must expose AC28 only and prohibit evidence writes")
    window = desktop.workspace_window_class(workspace)(
        model, store, args.cache_root, workspace,
        reviewer_name=args.reviewer, use_opengl=False)
    errors: list[str] = []
    window.image_cache.image_ready.connect(
        lambda _key, _image, error: errors.append(error) if error else None)
    window.resize(1600, 1000)
    window.show()
    app.processEvents()
    if window.current_run != RUN or not window.current_pair:
        raise ValueError("The real Studio viewer did not select AC28")
    window.frame_index = min(14, int(window.current_pair["candidate_a"]["frame_count"]) - 1)
    if window.frame_index < 0:
        raise ValueError("The selected Studio candidate has no frames")
    window.focus_after_load = False
    window.contours.setChecked(True)
    window._request_frame()
    deadline = time.monotonic() + 100

    def wait_rendered() -> None:
        while time.monotonic() < deadline:
            app.processEvents()
            base = window.viewer.base_item.pixmap()
            overlay = window.viewer.overlay_item.pixmap()
            if (not base.isNull() and not overlay.isNull()
                    and window.displayed_base_key == window.base_key):
                return
            if errors:
                raise RuntimeError("Studio image worker: " + "; ".join(errors))
            time.sleep(0.025)
        raise TimeoutError("The actual Studio scene did not render its full-resolution frame and phase map")

    wait_rendered()
    root = Path(window.current_review_root)
    manifest_path = root / "native_frames" / RUN / "manifest.json"
    manifest = _read(manifest_path)
    marker = manifest.get("v5_track1_photo_geometry") or {}
    if marker.get("enclosed_hole_policy") != "fill_enclosed_v1":
        raise ValueError("Studio loaded an old conservative-mask source instead of the rerun")
    main_asset = window._frame_asset(window.frame_index)
    main_path = main_asset.resolve()
    main_render_sha = _qt_image_sha(window.viewer.base_item.pixmap().toImage(), desktop)
    main_expected_sha = _qt_image_sha(desktop.QImage(str(main_path)), desktop)
    main_overlay_sha = _qt_image_sha(window.viewer.overlay_item.pixmap().toImage(), desktop)
    overlay_path = window._overlay_asset("A", window.frame_index).resolve()
    main_image_wh = [window.viewer.base_item.pixmap().width(), window.viewer.base_item.pixmap().height()]
    main_screenshot = output_dir / "studio_main.png"
    if main_screenshot.exists() or not window.grab().save(str(main_screenshot)):
        raise RuntimeError("Could not exclusively save the main Studio screenshot")
    mode_assets = []
    v2_modes = manifest["frame_sources"][window.frame_index].get("v5_display_modes") or {}
    if v2_modes:
        if not window.brightness.isEnabled() or window.display_mode != "natural":
            raise ValueError("The V2 Studio brightness control did not start in natural mode")
        for mode in ("natural", "lift1", "lift2"):
            if window.display_mode != mode:
                raise ValueError("The Studio brightness control cycled out of order")
            path = window._frame_asset(window.frame_index).resolve()
            actual = _qt_image_sha(window.viewer.base_item.pixmap().toImage(), desktop)
            expected = _qt_image_sha(desktop.QImage(str(path)), desktop)
            declaration = v2_modes.get(mode) or {}
            mode_assets.append({
                "mode": mode, "path": str(path), "sha256": sha256_file(path),
                "asset_matches_manifest": (bool(declaration.get("jpeg_path")) and
                    path == Path(declaration["jpeg_path"]).resolve() and
                    sha256_file(path) == declaration.get("jpeg_sha256")),
                "scene_matches_asset": actual == expected,
                "overlay_unchanged": _qt_image_sha(
                    window.viewer.overlay_item.pixmap().toImage(), desktop) == main_overlay_sha,
            })
            if mode != "lift2":
                window.brightness.click()
                wait_rendered()
        window.brightness.click()
        wait_rendered()
        if (window.display_mode != "natural" or
                _qt_image_sha(window.viewer.base_item.pixmap().toImage(), desktop) != main_render_sha):
            raise ValueError("The fixed brightness cycle did not return to the natural frame")
    if not window.raw_detail.isEnabled():
        raise ValueError("The real Studio RAW-detail toggle is unavailable")
    window.raw_detail.setChecked(True)
    wait_rendered()
    detail_path = window._frame_asset(window.frame_index).resolve()
    detail_render_sha = _qt_image_sha(window.viewer.base_item.pixmap().toImage(), desktop)
    detail_expected_sha = _qt_image_sha(desktop.QImage(str(detail_path)), desktop)
    detail_overlay_sha = _qt_image_sha(window.viewer.overlay_item.pixmap().toImage(), desktop)
    detail_screenshot = output_dir / "studio_raw_detail.png"
    if detail_screenshot.exists() or not window.grab().save(str(detail_screenshot)):
        raise RuntimeError("Could not exclusively save the RAW-detail Studio screenshot")
    assets = _asset_check(main_path, detail_path, manifest, window.frame_index)
    decision_files = sorted(str(path) for path in workspace.output_dir.rglob("*") if path.is_file())
    checks = {
        "real_darsia_dev_validation_window": isinstance(window, desktop.CalibrationReviewWindow),
        "offscreen_qt_platform": app.platformName().casefold() == "offscreen",
        "selected_target_model_id": workspace.target_model_id == model_id,
        "loaded_candidate_matches_selected_job": window.current_pair_object.candidate_a.candidate_id
            == _read(selection / "postflight_manifest.json")["jobs"][0]["candidate_id"],
        "ac28_only": set(model.pairs) == {RUN} and window.run_list.count() == 1,
        "read_only_decisions": store.is_read_only and not decision_files,
        "main_scene_matches_native_asset": main_render_sha == main_expected_sha,
        "main_scene_native_canvas": main_image_wh == manifest["canvas_wh"],
        "raw_detail_scene_matches_native_asset": detail_render_sha == detail_expected_sha,
        "raw_detail_toggle_changes_scene": main_render_sha != detail_render_sha,
        "phase_overlay_unchanged_by_raw_detail_toggle": main_overlay_sha == detail_overlay_sha,
        "native_presentation_assets_bound": assets["passed"],
        "new_filled_mask_source_loaded": marker.get("enclosed_hole_policy") == "fill_enclosed_v1",
    }
    if v2_modes:
        checks["v2_three_fixed_mode_assets"] = (
            [row["mode"] for row in mode_assets] == ["natural", "lift1", "lift2"]
            and all(row["asset_matches_manifest"] and row["scene_matches_asset"]
                    and row["overlay_unchanged"] for row in mode_assets)
            and not window.brightness.isEnabled())
    report = {
        "format": FORMAT, "created_utc": utc_now(), "candidate": str(candidate),
        "model_id": model_id, "model_sha256": model_sha, "run": RUN,
        "loaded_candidate_id": window.current_pair_object.candidate_a.candidate_id,
        "frame_index": window.frame_index, "desktop_module": str(Path(desktop.__file__).resolve()),
        "desktop_module_sha256": sha256_file(Path(desktop.__file__)),
        "studio_qa_module_sha256": sha256_file(Path(__file__)),
        "window_class": type(window).__name__, "qt_platform": app.platformName(),
        "offscreen_process": True, "existing_user_windows_touched": False,
        "native_manifest_path": str(manifest_path), "native_manifest_sha256": sha256_file(manifest_path),
        "assets": assets, "main_scene_rgba_sha256": main_render_sha,
        "display_mode_assets": mode_assets,
        "display_overlay_path": str(overlay_path), "display_overlay_sha256": sha256_file(overlay_path),
        "raw_detail_scene_rgba_sha256": detail_render_sha, "phase_overlay_rgba_sha256": main_overlay_sha,
        "main_screenshot": str(main_screenshot), "main_screenshot_sha256": sha256_file(main_screenshot),
        "raw_detail_screenshot": str(detail_screenshot),
        "raw_detail_screenshot_sha256": sha256_file(detail_screenshot),
        "checks": checks, "decision_files": decision_files,
        "passed": all(checks.values()), "status": "passed" if all(checks.values()) else "failed",
    }
    window.hide()
    # Prefetched photos must finish before Qt destroys its worker objects. This
    # waits only for this isolated process, never for a user's Studio instance.
    window.image_cache.pool.waitForDone(15000)
    return report


def inspect_studio(candidate: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run real offscreen Studio QA and exclusively create qa_studio.json."""
    candidate, output_dir = Path(candidate).resolve(), Path(output_dir).resolve()
    report_path = output_dir / "qa_studio.json"
    probe_path = output_dir / "studio_probe_result.json"
    if report_path.exists() or probe_path.exists() or (output_dir / "studio_probe").exists():
        raise FileExistsError("Studio QA output already exists; use a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_OPENGL="software",
               DARSIA_PHASE_DISPLAY_CACHE=str(output_dir / "studio_probe" / "phase_display_cache"))
    command = [sys.executable, "-m", __name__ if __name__ != "__main__" else
               "darsia.fluidflower.tools.qa_ffac_track1_rerun_studio",
               "--probe", "--candidate", str(candidate), "--output-dir", str(output_dir)]
    started = utc_now()
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True,
                                timeout=PROBE_TIMEOUT_SECONDS, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode not in (0, 1) or not probe_path.is_file():
            raise RuntimeError(f"Offscreen Studio probe exited {result.returncode}: "
                               + (result.stderr or result.stdout)[-4000:])
        report = _read(probe_path)
        if (report.get("format") != FORMAT or report.get("candidate") != str(candidate)
                or report.get("run") != RUN or not report.get("offscreen_process")):
            raise ValueError("Offscreen Studio probe returned an invalid candidate-bound receipt")
        if bool(report.get("passed")) != (result.returncode == 0):
            raise ValueError("Offscreen Studio exit code disagrees with its acceptance result")
        report.update(probe_exit_code=result.returncode, started_utc=started)
    except Exception as exc:  # QA failure must be recorded, not mistaken for a successful render.
        report = {"format": FORMAT, "candidate": str(candidate), "created_utc": utc_now(),
                  "started_utc": started, "run": RUN, "passed": False, "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}", "offscreen_process": True,
                  "existing_user_windows_touched": False}
    _write_new(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe:
        report = _probe(args.candidate.resolve(), args.output_dir.resolve())
        _write_new(args.output_dir / "studio_probe_result.json", report)
    else:
        report = inspect_studio(args.candidate, args.output_dir)
    print(json.dumps({"passed": report["passed"], "status": report["status"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

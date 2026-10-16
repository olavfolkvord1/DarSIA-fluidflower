"""Native UI binding of saved photometry to the unchanged training producer.

Only JSON metadata is read. Scientific arrays are checked for declared SHA and
file presence here; their bytes are verified by the original trainer before use.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from PySide6.QtWidgets import QLabel

from darsia.fluidflower.phase_training.chemistry_fit_scope import EXCLUDED_RUNS
from darsia.fluidflower.phase_training.log_linear_input import (
    EXACT_VALID_MASK_POLICY,
    MODEL_FORMAT,
    SOURCE_FORMAT,
    V5_UNIT_DENOMINATOR,
)
from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.phase_workflow import parse_training_bundles
from darsia.fluidflower.review_storage import resolve_review_path


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _read(path, expected=None):
    path = Path(path)
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError(
            "Training metadata is too large for this UI binding: " + str(path)
        )
    raw = path.read_bytes()
    if expected is not None and (not _digest(expected) or _sha(raw) != expected):
        raise ValueError("Saved training metadata SHA changed: " + str(path))

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate training metadata key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite training metadata: " + value)

    return raw, json.loads(
        raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=nonfinite
    )


def _fragment(raw, key):
    text = raw.decode("utf-8-sig")
    matches = list(re.finditer(r'"' + key + r'"\s*:\s*', text))
    if len(matches) != 1:
        raise ValueError("One unambiguous saved " + key + " block is required")
    start = matches[0].end()
    _value, length = json.JSONDecoder().raw_decode(text[start:])
    return text[start : start + length].encode("utf-8")


def _series(value):
    return str(value or "").casefold().replace("_", "").replace("-", "")


@dataclass(frozen=True)
class SavedTrainingPhotometry:
    model_id: str
    request_path: Path
    request_sha256: str
    request_model_id: str
    settings: dict
    fragments: dict
    reference_paths: tuple[Path, ...]

    @property
    def label(self):
        return f"Photometry: v5, from request {self.request_sha256} of model {self.model_id}"

    def require_request(self, request):
        """The real producer's persisted blocks must retain their exact bytes."""
        _read(self.request_path, self.request_sha256)
        raw, saved = _read(request["request_path"])
        for key, expected in self.fragments.items():
            if (
                _fragment(raw, key) != expected
                or saved["settings"].get(key) != self.settings[key]
            ):
                raise ValueError(
                    "The original request producer did not preserve "
                    + key
                    + " byte-identically"
                )
        if saved["settings"].get("radiometry_profile") not in (None, "", "legacy"):
            raise ValueError("The V5 request unexpectedly carries legacy radiometry")


def read_saved_training_photometry(entry, series_id):
    """Resolve the selected indexed model's explicit request; no path discovery."""
    if not isinstance(entry, dict) or not entry.get("root"):
        raise ValueError("Choose an indexed saved model as the photometry source")
    identity = entry.get("library_source_identity") or {}
    expected = (identity.get("metadata_sha256") or {}).get("training_request.json")
    if not _digest(expected) or identity.get("model_id") != entry.get("model_id"):
        raise ValueError("The selected model lacks its indexed training-request SHA")
    selected = _series(series_id)
    covered = entry.get("datasets") or [entry.get("series_scope")]
    if isinstance(covered, str):
        covered = [covered]
    if not selected or selected not in {_series(item) for item in covered}:
        raise ValueError("The saved photometry model belongs to another series")
    path = Path(entry["root"]) / "training_request.json"
    raw, document = _read(path, expected)
    settings = document.get("settings") or {}
    photometry, chemistry = settings.get("v5_raw_photometry"), settings.get(
        "v5_chemistry_fit_scope"
    )
    if (
        not isinstance(photometry, dict)
        or photometry.get("format") != MODEL_FORMAT
        or photometry.get("full_native_resolution") is not True
        or photometry.get("unit_denominator") != V5_UNIT_DENOMINATOR
        or photometry.get("display_jpeg_is_training_input") is not False
    ):
        raise ValueError(
            "The indexed request has no complete recorded V5 input contract"
        )
    if not isinstance(chemistry, dict) or chemistry.get("excluded_runs") != list(
        EXCLUDED_RUNS
    ):
        raise ValueError(
            "The indexed V5 request lacks its original chemistry-fit scope"
        )
    if settings.get("radiometry_profile") not in (None, "", "legacy"):
        raise ValueError(
            "The saved V5 source request also carries incompatible legacy radiometry"
        )
    references = []
    for key, value in photometry.items():
        if key.endswith("_path"):
            expected_sha = photometry.get(key[:-5] + "_sha256")
            if not isinstance(value, str) or not value:
                raise ValueError("Missing saved V5 reference path: " + key)
            if not _digest(expected_sha):
                raise ValueError("Missing saved V5 reference SHA: " + key)
            _read(value, expected_sha)
            references.append(Path(value))
    if not references:
        raise ValueError("The saved V5 request has no SHA-bound source metadata")
    copied = {
        key: deepcopy(settings[key])
        for key in ("v5_raw_photometry", "v5_chemistry_fit_scope")
    }
    if "v5_review_fit_partition" in settings:
        copied["v5_review_fit_partition"] = deepcopy(
            settings["v5_review_fit_partition"]
        )
    return SavedTrainingPhotometry(
        str(entry["model_id"]),
        path,
        expected,
        str(document.get("model_id") or ""),
        copied,
        {key: _fragment(raw, key) for key in copied},
        tuple(references),
    )


def require_saved_sidecars(bundles, binding):
    """Check exactly the selected bundles' declared V5 files, never their pixels."""
    if not bundles:
        raise ValueError("Choose saved reviewed evidence before training")
    checked, payloads = set(), set()
    for decisions_path, combined_path in bundles:
        _raw, decisions = _read(decisions_path)
        _raw, combined = _read(combined_path)
        jobs = combined.get("jobs") or []
        for run, decision in (decisions.get("decisions") or {}).items():
            if not any(
                decision.get(key)
                for key in ("annotations", "phase_quiz_answers", "frame_verdicts")
            ):
                continue
            rows = [
                job
                for job in jobs
                if str(job.get("run", "")).casefold() == str(run).casefold()
            ]
            if not rows:
                raise ValueError(
                    "Saved reviewed evidence has no original source job for " + str(run)
                )
            job = rows[0]
            root = resolve_review_path(
                job.get("analysis_source_review_root") or job["review_output_dir"]
            )
            root = analysis_review_root(root, str(run).casefold())
            path = root / "native_frames" / str(run).casefold() / "manifest.json"
            if path in checked:
                continue
            expected = job.get("native_manifest_sha256") or (
                job.get("v5_raw_photometry") or {}
            ).get("materialized_manifest_sha256")
            if not _digest(expected):
                raise ValueError("The reviewed V5 source lacks its native-manifest SHA")
            _raw, native = _read(path, expected)
            if (native.get("v5_photometry") or {}).get("format") != SOURCE_FORMAT:
                raise ValueError(
                    "Selected evidence uses historical JPEG input instead of saved V5 sidecars"
                )
            if (
                native.get("v5_track1_photo_geometry")
                and "v5_review_fit_partition" not in binding.settings
            ):
                raise ValueError(
                    "Track1 evidence needs its original saved V5 review-fit partition; this request does not contain it"
                )
            records = [
                native.get("reference") or {},
                *(native.get("frame_sources") or []),
            ]
            if len(records) < 2:
                raise ValueError("Selected V5 source has no recorded frame timeline")
            for record in records:
                keys = ["v5_log_linear_rgb_path"]
                if native.get("v5_float_support_policy") == EXACT_VALID_MASK_POLICY:
                    keys.append("v5_log_linear_valid_mask_path")
                if record.get("v5_extended_ring_path") is not None:
                    keys.append("v5_extended_ring_path")
                for key in keys:
                    value, digest = record.get(key), record.get(key[:-5] + "_sha256")
                    if not isinstance(value, str) or not value or not _digest(digest):
                        raise ValueError(
                            "Missing saved sidecar path/SHA: "
                            + key
                            + " in "
                            + str(path)
                        )
                    if not Path(value).is_file():
                        raise FileNotFoundError("Saved V5 sidecar is missing: " + value)
                    payloads.add(value)
            checked.add(path)
    if not checked:
        raise ValueError(
            "No reviewed V5 run source is available in the selected evidence"
        )
    return {
        "native_manifests": len(checked),
        "declared_existing_sidecars": len(payloads),
        "payload_sha_verified_here": False,
        "payload_sha_validation": "Original trainer before use",
    }


class NativeTrainingPhotometry:
    """One native instance's settings hook, source label and fresh-route gate."""

    def __init__(self, steps):
        self.steps, self.studio = steps, steps.studio
        self.binding, self.reason, self._cache_key = (
            None,
            "Choose saved V5 photometry",
            None,
        )
        self.label = QLabel(self.reason)
        self.label.setWordWrap(True)
        steps.layouts[4].addWidget(self.label)
        self.studio._native_training_photometry = self
        for name, signal_name in (
            ("validation_model", "currentIndexChanged"),
            ("bundles", "textChanged"),
            ("reviewer", "textChanged"),
            ("training_series", "textChanged"),
            ("training_scope", "currentIndexChanged"),
            ("radiometry_mode", "currentIndexChanged"),
        ):
            control = getattr(self.studio, name, None)
            signal = getattr(control, signal_name, None)
            if signal is not None:
                signal.connect(self.changed)
        profile = getattr(self.studio, "radiometry_profile_path", None)
        profile_changed = getattr(profile, "changed", None)
        if profile_changed is not None:
            profile_changed.connect(self.changed)
        self._attach_effect_summary()

    def _attach_effect_summary(self):
        original = getattr(self.studio, "_update_training_effect_summary", None)
        if not callable(original):
            return

        def summary(*args, **kwargs):
            result = original(*args, **kwargs)
            self._apply_effect_summary()
            return result

        self.studio._update_training_effect_summary = summary
        # Original signal slots stay connected. Add a final presentation-only
        # label update, preserving their backend and concurrency wording.
        for name in (
            "radiometry_mode",
            "radiometry_profile_path",
            "backend",
            "stage_workers",
            "extraction_workers",
            "pixels",
            "mil_iterations",
            "change_ablation",
        ):
            control = getattr(self.studio, name, None)
            for signal_name in (
                "changed",
                "currentIndexChanged",
                "stateChanged",
                "valueChanged",
            ):
                signal = getattr(control, signal_name, None)
                if signal is not None:
                    signal.connect(self._apply_effect_summary)
                    break
        self._apply_effect_summary()

    def _apply_effect_summary(self, *_args):
        label = getattr(self.studio, "training_effect_summary", None)
        if label is None:
            return
        _old, marker, remainder = label.text().partition(" Current effect: ")
        prefix = (
            "Manual new-model training uses saved V5 photometry."
            if self.binding is not None
            else "Manual new-model training: V5 photometry unavailable."
        )
        label.setText(prefix + marker + remainder)

    def fresh_ui_ready(self):
        """Original UI eligibility, excluding only the unused legacy profile.

        The source/evidence admission remains separate and force-revalidated by
        Train. Standalone and non-fresh callbacks retain their original gate.
        """
        studio = self.studio
        required = ("reviewer", "bundles", "training_series", "training_scope")
        if self.binding is None or any(not hasattr(studio, name) for name in required):
            return False
        ready = bool(studio.reviewer.text().strip()) and bool(
            studio.bundles.toPlainText().strip()
        )
        if studio.training_scope.currentData() == "single-series":
            ready = (
                ready
                and len(
                    [
                        value
                        for value in studio.training_series.text().split(",")
                        if value.strip()
                    ]
                )
                == 1
            )
        return (
            ready
            and getattr(studio, "training_process", None) is None
            and not getattr(studio, "_series_batch_active", False)
        )

    def changed(self, *_args):
        self._cache_key = None
        self.steps.refresh()

    def refresh(self, *, force=False):
        try:
            entry = self.studio._selected_validation_model()
            bundles = parse_training_bundles(self.studio.bundles.toPlainText())
            paths = [Path(path) for pair in bundles for path in pair]
            request = Path(entry["root"]) / "training_request.json" if entry else None
            if request is not None:
                paths.append(request)
            key = (
                self.steps.series_id,
                json.dumps(
                    entry.get("library_source_identity") if entry else None,
                    sort_keys=True,
                ),
                tuple(
                    (str(path), path.stat().st_size, path.stat().st_mtime_ns)
                    for path in paths
                ),
            )
            if not force and self._cache_key == key:
                self._apply_effect_summary()
                return self.binding is not None
            binding = read_saved_training_photometry(entry, self.steps.series_id)
            require_saved_sidecars(bundles, binding)
            self.binding, self._cache_key = binding, key
            self.reason = (
                binding.label
                + "\nSaved sidecar path/SHA metadata and presence checked; the original trainer verifies payload SHA before use."
            )
            self.steps.training_routes["scratch"] = self.train
        except Exception as error:
            self.binding, self._cache_key = None, None
            self.reason = "Fresh training is unavailable: " + str(error)
            self.steps.training_routes.pop("scratch", None)
        self.label.setText(self.reason)
        self._apply_effect_summary()
        return self.binding is not None

    def prepare_settings(self, settings, bundles):
        if not self.refresh(force=True) or self.binding is None:
            raise ValueError(self.reason)
        try:
            require_saved_sidecars(bundles, self.binding)
        except Exception as error:
            self._disable(error)
            raise
        result = deepcopy(settings)
        result.pop("radiometry_profile", None)
        result.update(deepcopy(self.binding.settings))
        return result

    def require_request(self, request):
        if self.binding is None:
            raise ValueError(self.reason)
        try:
            self.binding.require_request(request)
        except Exception as error:
            self._disable(error)
            raise

    def _disable(self, error):
        self.binding = None
        self.reason = "Fresh training is unavailable: " + str(error)
        self.steps.training_routes.pop("scratch", None)
        self.label.setText(self.reason)
        self._apply_effect_summary()

    def train(self):
        if not self.refresh(force=True):
            self.steps.refresh()
            raise ValueError(self.reason)
        try:
            return self.steps._original_start()
        finally:
            self.steps.refresh()

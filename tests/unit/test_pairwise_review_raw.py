"""Studio pairwise review on a RAW photographic review root (package D2b).

The photographs of a RAW root are its SHA-verified natural PNGs; the frame
rows are its frames, named by their camera RAW file. Historical roots keep
their old paths.
"""
import json

import pytest

from darsia.fluidflower.tools import calibration_pairwise_review as pairwise
from .test_raw_photographic_review import _write, inputs  # noqa: F401


def test_raw_root_photographs_and_rows(tmp_path, inputs):
    manifest = json.loads(_write(tmp_path, inputs).read_text(encoding="utf-8"))
    root = tmp_path / "review"
    folder = (root / "native_frames/ac52").resolve()
    assert pairwise._native_frame_path(root, "ac52", 1).resolve() == folder / "frames/ac52_f2/natural.png"
    assert pairwise._native_reference_path(root, "ac52").resolve() == folder / "reference/natural.png"
    reference, ready = pairwise._native_photograph_availability(root, "ac52", manifest, 2)
    assert ready and reference.resolve() == folder / "reference/natural.png"
    with pytest.raises(ValueError, match="frame count"):
        pairwise._native_photograph_availability(root, "ac52", manifest, 3)
    (folder / "frames/ac52_f1/natural.png").unlink()
    assert not pairwise._native_photograph_availability(root, "ac52", manifest, 2)[1]
    rows = pairwise._raw_review_rows(manifest)
    assert [row["source_name"] for row in rows] == ["ac52_f1.ARW", "ac52_f2.ARW"]
    assert rows[1]["case_id"] == "ac52_f2" and "source_name" not in manifest["frames"][1]


def test_historical_roots_keep_their_photograph_paths(tmp_path):
    assert pairwise._raw_review_photograph(tmp_path, "ac52") is None
    folder = tmp_path / "native_frames/ac52"
    assert pairwise._native_frame_path(tmp_path, "ac52", 3) == folder.resolve() / "frame_003.jpg"
    assert pairwise._native_reference_path(tmp_path, "ac52") == (
        tmp_path.resolve() / "native_references/ac52/pre_injection.jpg")


def test_every_new_raw_consumer_function_has_a_docstring():
    from darsia.fluidflower import phase_photographic_assets, phase_saved_presentation, storage_original_photo
    from darsia.fluidflower.phase_training import raw_frame_context
    from darsia.fluidflower.tools import phase_studio_photometry_preview_cache

    cohort = raw_frame_context.RawFrameGeometryCohort
    items = [
        phase_photographic_assets._raw_review, phase_photographic_assets._raw_rows,
        phase_photographic_assets._raw_modes, storage_original_photo.publish_raw_original_png,
        storage_original_photo._publish, phase_saved_presentation._timeline_rows,
        phase_saved_presentation._presentation_photo, phase_studio_photometry_preview_cache.raw_review_key_info,
        pairwise._raw_review_photograph, pairwise._raw_review_rows, cohort,
        *[getattr(cohort, name) for name in ("frame", "context", "H_v5_B_to_frame", "H_previous_to_frame",
                                             "feature_support", "_valid", "_frame_valid")]]
    for item in items:
        assert item.__doc__ and item.__doc__.strip(), item

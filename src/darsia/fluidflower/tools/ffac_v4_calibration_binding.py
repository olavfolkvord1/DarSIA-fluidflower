"""Exact 52-revision optical baseline lookup for V4 FF_AC diagnostics.

The V4 fit is keyed by the original source revision, not merely an AC run
number.  This validates the separately fitted historical baselines before a
frame processor can use them.  It never approves the still-pending temporal
geometry, phase model, or scientific mass calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from darsia.fluidflower.project import sha256_file


LOOKUP_CONTRACT = "ffac-v4-safe-interior-exact-52-source-B-optical-lookup-v1"
SOURCE_CONTRACT = "ffac-v5-revision-exact-reference-binding-52-v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _checked(path: Path, digest: str, what: str) -> None:
    if (not isinstance(digest, str) or len(digest) != 64
            or not path.is_file() or sha256_file(path) != digest):
        raise ValueError(f"V4 {what} changed or lacks exact SHA: {path}")


class V4BaselineLookup:
    """Read-only source/B fit lookup with no run-name fallback."""

    def __init__(self, lookup_path: str | Path,
                 source_binding_path: str | Path):
        self.lookup_path = Path(lookup_path)
        self.source_binding_path = Path(source_binding_path)
        self.lookup_sha256 = sha256_file(self.lookup_path)
        self.source_binding_sha256 = sha256_file(self.source_binding_path)
        lookup = _read(self.lookup_path)
        source = _read(self.source_binding_path)
        if (lookup.get("contract") != LOOKUP_CONTRACT
                or source.get("contract") != SOURCE_CONTRACT
                or lookup.get("source_52_binding_sha256") != self.source_binding_sha256
                or lookup.get("source_revision_count") != 52
                or lookup.get("canonical_count") != 43
                or lookup.get("historical_count") != 9
                or lookup.get("all_B_source_binding_pass") is not True
                or lookup.get("temporal_frame_geometry_pass") is not False
                or lookup.get("physical_facies_boundary_pass") is not False
                or lookup.get("scientific_mass_eligible") is not False):
            raise ValueError("V4 B lookup has another source or acceptance scope")
        source_rows = {r["old_manifest_sha256"]: r for r in source.get("rows") or []}
        rows = {r["source_manifest_sha256"]: r for r in lookup.get("rows") or []}
        if len(source_rows) != 52 or len(rows) != 52 or set(rows) != set(source_rows):
            raise ValueError("V4 B lookup does not cover 52 distinct frozen revisions")
        self.rows: dict[str, dict[str, Any]] = {}
        self.source_rows = source_rows
        # The diagnostic author stored AC28-H certificates relative to the
        # FF Albus project, while B certificates are absolute. Resolve that
        # spelling once, then live-hash the bound bytes.
        project_root = self.lookup_path.resolve().parents[2]
        for digest, row in rows.items():
            bound = source_rows[digest]
            if (row.get("run") != bound.get("run")
                    or row.get("original_B_ARW_sha256") != bound.get("camera_arw_sha256")
                    or row.get("original_B_JPEG_sha256") != bound.get("camera_jpeg_sha256")
                    or row.get("original_B_review_sha256") !=
                    bound.get("original_review_jpeg_sha256")):
                raise ValueError(f"V4 B calibration has another source acquisition: {digest}")
            baseline = Path(row["baseline_path"])
            _checked(baseline, row["baseline_sha256"], "baseline fit")
            _checked(baseline.with_suffix(".json"), row["baseline_sidecar_sha256"],
                     "baseline provenance")
            _checked(Path(row["baseline_registration_certificate_path"]),
                     row["baseline_registration_certificate_sha256"],
                     "B review/camera registration")
            h_path = Path(row["AC28_to_revision_H_certificate_path"])
            if not h_path.is_absolute():
                h_path = project_root / h_path
            _checked(h_path, row["AC28_to_revision_H_certificate_sha256"],
                     "AC28-to-revision B geometry")
            self.rows[digest] = dict(row,
                AC28_to_revision_H_certificate_path=str(h_path))

    def binding(self, source_manifest_sha256: str, run: str) -> dict[str, Any]:
        row = self.rows.get(source_manifest_sha256)
        if row is None or row["run"] != run:
            raise ValueError("V4 has no exact revision-keyed B fit")
        source = self.source_rows[source_manifest_sha256]
        _checked(Path(source["old_manifest_path"]), source_manifest_sha256,
                 "original source manifest")
        for key, digest_key, label in (
            ("camera_arw", "camera_arw_sha256", "original B ARW"),
            ("camera_jpeg", "camera_jpeg_sha256", "original B camera JPEG"),
            ("original_review_jpeg", "original_review_jpeg_sha256", "original B native JPEG"),
        ):
            _checked(Path(source[key]), source[digest_key], label)
        return dict(row, source_manifest_path=source["old_manifest_path"],
                    source_manifest_sha256=source_manifest_sha256,
                    original_B_ARW_path=source["camera_arw"],
                    original_B_JPEG_path=source["camera_jpeg"],
                    original_B_review_path=source["original_review_jpeg"],
                    lookup_sha256=self.lookup_sha256,
                    source_binding_sha256=self.source_binding_sha256,
                    validation_only=True, mass_eligible=False)


__all__ = ["V4BaselineLookup", "LOOKUP_CONTRACT"]

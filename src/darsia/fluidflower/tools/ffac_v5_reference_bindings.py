"""Source-exact B acquisitions for all frozen FF_AC V5 native revisions.

The current Claude B is bound to each canonical run.  Historical review
revisions instead keep their own acquisition and native reference.  In
particular AC56, AC60 and AC61 must never inherit the current run's B fit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import source_manifests
from darsia.fluidflower.tools.materialize_ffac_v5_sources import file_sha256, _read


CONTRACT = "ffac-v5-revision-exact-reference-binding-52-v1"


def _checked(path: Path, expected: str, description: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{description} SHA differs: {path}")
    return actual


def _same_stem_arw(camera_jpeg: Path) -> Path:
    if camera_jpeg.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"B acquisition is not an original JPEG: {camera_jpeg}")
    arw = camera_jpeg.with_suffix(".ARW")
    if not arw.is_file() or arw.stem.casefold() != camera_jpeg.stem.casefold():
        raise FileNotFoundError(f"No exact same-stem B ARW: {camera_jpeg}")
    return arw


def _historical_reference(revision: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    run = revision["run"]
    root = Path(revision["review_root"])
    availability_path = root / "native_references" / run / "reference_availability.json"
    if availability_path.is_file():
        availability_sha = file_sha256(availability_path)
        availability = _read(availability_path)
        baseline = availability.get("baseline") or {}
        if (availability.get("run") != run
                or availability.get("materialized_manifest_sha256") != revision["manifest_sha256"]
                or baseline.get("available") is not True):
            raise ValueError(f"Historical B availability belongs to another source: {run}")
        camera_jpeg = Path(baseline["raw_path"])
        review_jpeg = Path(baseline["materialized_native_path"])
        camera_jpeg_sha = _checked(camera_jpeg, baseline["raw_sha256"], "historical B JPEG")
        review_sha = _checked(review_jpeg, baseline["materialized_native_sha256"], "historical B native")
        evidence = dict(kind="frozen-reference-availability",
                        path=str(availability_path), sha256=availability_sha)
    else:
        # Older native sources can bind B directly to a timeline acquisition.
        reference = manifest.get("reference") or {}
        if reference.get("included_in_timeline") is not True:
            raise ValueError(f"Historical {run} has no source-bound B reference")
        index = int(reference["timeline_frame_index"])
        frame = manifest["frame_sources"][index]
        camera_jpeg = Path(frame["source_path"])
        review_jpeg = root / "native_frames" / run / f"frame_{index:03d}.jpg"
        if (frame["frame_index"] != index
                or Path(reference["source_path"]) != camera_jpeg
                or (reference.get("source_sha256") is not None
                    and reference["source_sha256"] != frame["source_sha256"])):
            raise ValueError(f"Historical {run} timeline B is inconsistent")
        camera_jpeg_sha = _checked(camera_jpeg, frame["source_sha256"], "historical timeline B JPEG")
        review_sha = _checked(review_jpeg, frame["native_jpeg_sha256"], "historical timeline B native")
        evidence = dict(kind="frozen-timeline-reference", index=index,
                        source_manifest_sha256=revision["manifest_sha256"])
    arw = _same_stem_arw(camera_jpeg)
    return dict(camera_arw=str(arw), camera_arw_sha256=file_sha256(arw),
                camera_jpeg=str(camera_jpeg), camera_jpeg_sha256=camera_jpeg_sha,
                original_review_jpeg=str(review_jpeg), original_review_jpeg_sha256=review_sha,
                evidence=evidence)


def build_reference_bindings(
    *, request_path: Path, baseline_certificate_root: Path,
    reference_state_path: Path,
) -> dict[str, Any]:
    """Audit 43 canonical B triples and nine historical revision triples.

    A returned row is usable only with the *same* native source-manifest SHA.
    This audit deliberately does not claim that historical labels or facies are
    training eligible; it only seals the exact acquisition and registration B.
    """
    request_path = Path(request_path)
    reference_state_path = Path(reference_state_path)
    reference_state = _read(reference_state_path)["reference_state"]
    revisions = source_manifests(request_path)
    if len(revisions) != 52 or len({r["run"] for r in revisions}) != 43:
        raise ValueError("Expected the complete frozen 52-revision/43-run cohort")
    certificates: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for run in {r["run"] for r in revisions}:
        path = Path(baseline_certificate_root) / f"{run}_B_certificate.json"
        digest = file_sha256(path)
        cert = _read(path)
        if cert.get("run", "").casefold() != run:
            raise ValueError(f"Canonical B certificate belongs to another run: {run}")
        _checked(Path(cert["native_manifest_path"]), cert["native_manifest_sha256"],
                 "canonical B native manifest")
        certificates[run] = path, digest, cert
    rows: list[dict[str, Any]] = []
    canonical_counts = {run: 0 for run in certificates}
    for revision in revisions:
        run = revision["run"]
        digest = revision["manifest_sha256"]
        _checked(Path(revision["manifest_path"]), digest, "frozen native source manifest")
        manifest = _read(Path(revision["manifest_path"]))
        cert_path, cert_sha, cert = certificates[run]
        source_parent = (manifest.get("two_arm_photometry") or {}).get("source_manifest_sha256")
        canonical = digest == cert["native_manifest_sha256"] or source_parent == cert["native_manifest_sha256"]
        if canonical:
            canonical_counts[run] += 1
            inputs = cert["sources"]
            if Path(cert["source_arw_path"]).name.casefold() != str(
                    reference_state[run]["file"]).casefold():
                raise ValueError(f"Canonical B is not Claude's stated B: {run}")
            triple = dict(camera_arw=cert["source_arw_path"],
                          camera_arw_sha256=inputs["arw_sha256"],
                          camera_jpeg=cert["source_camera_jpeg_path"],
                          camera_jpeg_sha256=inputs["camera_jpeg_sha256"],
                          original_review_jpeg=cert["source_review_path"],
                          original_review_jpeg_sha256=inputs["review_native_jpeg_sha256"],
                          evidence=dict(kind="current-run-source-B-certificate",
                                        path=str(cert_path), sha256=cert_sha,
                                        source_native_manifest_sha256=cert["native_manifest_sha256"]))
        else:
            triple = _historical_reference(revision, manifest)
        _checked(Path(triple["camera_arw"]), triple["camera_arw_sha256"], "B ARW")
        _checked(Path(triple["camera_jpeg"]), triple["camera_jpeg_sha256"], "B camera JPEG")
        _checked(Path(triple["original_review_jpeg"]), triple["original_review_jpeg_sha256"],
                 "B native review JPEG")
        if Path(triple["camera_arw"]).stem.casefold() != Path(triple["camera_jpeg"]).stem.casefold():
            raise ValueError(f"Unpaired B camera files for {run}")
        rows.append(dict(run=run, old_manifest_sha256=digest,
                         old_manifest_path=revision["manifest_path"],
                         is_canonical=canonical, calibration_key=run if canonical else digest,
                         **triple))
    if any(count != 1 for count in canonical_counts.values()):
        raise ValueError(f"Canonical B source is not unique per run: {canonical_counts}")
    rows.sort(key=lambda r: (r["run"], r["old_manifest_sha256"]))
    return dict(contract=CONTRACT, frozen_request_path=str(request_path),
                frozen_request_sha256=file_sha256(request_path),
                reference_state_path=str(reference_state_path),
                reference_state_sha256=file_sha256(reference_state_path),
                baseline_certificate_root=str(baseline_certificate_root),
                source_revisions=52, canonical_runs=43, historical_revisions=9,
                rows=rows)

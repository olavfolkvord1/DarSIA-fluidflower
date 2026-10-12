"""Full-native source-bound FF_AC V5 processor for immutable review assets.

The renderer is injected so its certified implementation and SHA are fixed by
the batch job. This module binds the 43 frozen B fits, per-frame light field,
facies grid, response, original JPEG/ARW and native review geometry. It does
not inspect labels or phase predictions and does not publish a model.
"""

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from darsia.fluidflower.tools.ffac_v5_photometry import (
    apply_v5, estimate_frame_light_grid, load_ac28_response, load_baseline,
    registered_raw_grid_from_arw, render_ac28_response,
)
from darsia.fluidflower.tools.ffac_v5_chart_registration import register_frame_chart_xy8
from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
    CorrectedFrame, file_sha256,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value: dict[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _native_registration_gate(geometry_cert: dict[str, Any]) -> dict[str, Any]:
    """Pre-batch source-only gate on sand geometry, separate from label QA.

    The ColorChecker sits outside the porous sand and its residual is reported,
    but it cannot veto otherwise source-aligned sand/phase coordinates.
    """
    areas = geometry_cert.get("raw_review_direct_residual") or {}
    checked = {}
    for name in ("center", "right_lower_sand"):
        stats = areas.get(name) or {}
        n = int(stats.get("n") or 0)
        p95 = float(stats.get("p95_px", float("inf")))
        if n < 100 or not np.isfinite(p95) or p95 >= 1.0:
            raise ValueError(f"Native RAW/review {name} registration failed: n={n}, p95={p95}")
        checked[name] = dict(n=n, p95_px=p95)
    chart = areas.get("chart") or {}
    return dict(contract="ffac-v5-sand-registration-gate-v1",
        center_and_right_lower_sand_p95_limit_px=1.0,
        minimum_correspondences_per_sand_region=100,
        checked=checked, chart_p95_px=chart.get("p95_px"),
        chart_is_observation_only=True, passed=True)


def _ea_native_light_grid(linear: np.ndarray) -> np.ndarray:
    """Exact log-area signal used to fit the EA fixed-point baselines."""
    if linear.shape != (3, 2979, 5472) or linear.dtype != np.float32:
        raise ValueError("EA light grid requires the full-native camera-linear renderer")
    native_area = np.mean(linear[:, :2976, :5472].reshape(
        3, 372, 8, 684, 8), axis=(2, 4), dtype=np.float64)
    if not np.isfinite(native_area).all() or np.any(native_area <= 0):
        raise ValueError("EA full-native 8x8 area grid is not finite and positive")
    return np.log(native_area).astype(np.float32)


def _repair_isolated_porous_holes(log_linear: np.ndarray,
                                  facies: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill only sparse nonfinite pixels from finite same-facies 3x3 neighbours.

    The limits were frozen after localizing 1,225 original pilot holes: every
    hole had a finite neighbour at one pixel, components were at most four
    pixels, and the largest frame had 116 holes. Valid pixels never change.
    """
    if log_linear.ndim != 3 or log_linear.shape[0] != 3 or log_linear.shape[1:] != facies.shape:
        raise ValueError("Sparse-hole repair requires three channels on the native facies grid")
    height, width = facies.shape
    valid = np.isfinite(log_linear).all(axis=0)
    porous = facies > 0
    bad = porous & ~valid
    count = int(np.count_nonzero(bad))
    mask_sha = sha256(np.packbits(bad, bitorder="little").tobytes()).hexdigest()
    receipt = dict(contract="ffac-v5-sparse-porous-hole-repair-v1",
        original_invalid_porous_mask_sha256=mask_sha,
        original_invalid_porous_pixels=count,
        maximum_allowed_pixels_per_frame=256,
        maximum_allowed_component_pixels=4,
        same_facies_neighbour_radius_px=1,
        minimum_finite_same_facies_neighbours=2,
        repaired_pixels=[])
    if count == 0:
        receipt["passed"] = True
        return log_linear, receipt
    if count > 256:
        raise ValueError(f"Too many invalid porous pixels for sparse repair: {count}")
    n, _labels, stats, _centres = cv2.connectedComponentsWithStats(
        np.asarray(bad, np.uint8), connectivity=8)
    if n > 1 and int(np.max(stats[1:, cv2.CC_STAT_AREA])) > 4:
        raise ValueError("Invalid porous component exceeds four native pixels")
    source = log_linear
    repaired = np.array(log_linear, copy=True)
    ys, xs = np.nonzero(bad)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, int(y)-1), min(height, int(y)+2)
        x0, x1 = max(0, int(x)-1), min(width, int(x)+2)
        neighbours = (valid[y0:y1, x0:x1] &
            (facies[y0:y1, x0:x1] == facies[y, x]))
        n_neighbours = int(np.count_nonzero(neighbours))
        if n_neighbours < 2:
            raise ValueError("Sparse hole has fewer than two finite same-facies neighbours")
        replacement = np.median(source[:, y0:y1, x0:x1][:, neighbours], axis=1)
        if not np.isfinite(replacement).all():
            raise ValueError("Sparse repair produced nonfinite camera-linear signal")
        repaired[:, y, x] = replacement
        receipt["repaired_pixels"].append(dict(x=int(x), y=int(y),
            facies=int(facies[y, x]), finite_same_facies_neighbours=n_neighbours,
            log_rgb=[float(value) for value in replacement]))
    if not np.isfinite(repaired[:, porous]).all():
        raise ValueError("Sparse repair left invalid pixels in porous domain")
    receipt["passed"] = True
    return repaired, receipt


class FFACV5SourceProcessor:
    """RAW→native V5 correction with a separable, certified native renderer."""

    def __init__(
        self, *, calibration_manifest: Path, source_binding_audit: Path,
        historical_calibration_manifest: Path | None = None,
        historical_parity_receipt: Path | None = None,
        historical_chart_manifest: Path | None = None,
        geometry_params: Path, response_path: Path,
        photometry_module_path: Path, renderer: Callable[..., tuple[np.ndarray, dict]],
        renderer_identity: dict[str, Any],
    ) -> None:
        self.calibration_manifest_path = Path(calibration_manifest)
        self.source_binding_path = Path(source_binding_audit)
        self.historical_calibration_path = (Path(historical_calibration_manifest)
                                            if historical_calibration_manifest is not None else None)
        self.historical_parity_path = (Path(historical_parity_receipt)
                                       if historical_parity_receipt is not None else None)
        self.historical_chart_path = (Path(historical_chart_manifest)
                                      if historical_chart_manifest is not None else None)
        self.geometry_params_path = Path(geometry_params)
        self.response_path = Path(response_path)
        self.photometry_module_path = Path(photometry_module_path)
        if Path(inspect.getfile(apply_v5)).resolve() != self.photometry_module_path.resolve():
            raise ValueError("Imported V5 photometry code differs from SHA-bound module")
        self.renderer = renderer
        self.renderer_identity = dict(renderer_identity)
        if not self.renderer_identity.get("module_sha256") or not self.renderer_identity.get("gate_receipt_sha256"):
            raise ValueError("Native RAW renderer lacks immutable module/gate provenance")
        gate = Path(self.renderer_identity["gate_receipt_path"])
        if file_sha256(gate) != self.renderer_identity["gate_receipt_sha256"]:
            raise ValueError("Native RAW renderer gate receipt changed")
        gate_value = _read(gate)
        candidate = self.renderer_identity.get("primary")
        if gate_value.get("contract") == "ffac-v5-full-native-validation-only-exception-v2":
            if (gate_value.get("status") != "validation-only-ac45-exception"
                    or gate_value.get("candidate") != candidate or candidate != "EA"
                    or gate_value.get("scope") !=
                    "source-materialization-for-manual-studio-validation-only"
                    or gate_value.get("automatic_model_publication") is not False
                    or gate_value.get("scientific_phase_or_mass_acceptance") is not False):
                raise ValueError("Full-native materialization approval does not bind this renderer")
        else:
            raise ValueError("V5 source processor requires the documented EA validation-only contract")
        self.calibration_manifest = _read(self.calibration_manifest_path)
        self.source_binding = _read(self.source_binding_path)
        if self.source_binding.get("contract") != "ffac-v5-revision-exact-reference-binding-52-v1":
            raise ValueError("V5 B binding must be the corrected 52-revision source-exact audit")
        if (self.historical_calibration_path is None or self.historical_parity_path is None
                or self.historical_chart_path is None):
            raise ValueError("Historical B fits, chart geometry and grid parity are required")
        self.historical_calibration = _read(self.historical_calibration_path)
        self.historical_parity = _read(self.historical_parity_path)
        self.historical_chart = _read(self.historical_chart_path)
        if (self.historical_calibration.get("contract") !=
                "ffac-ea-v2-historical-B-source-specific-calibration-v1"
                or self.historical_parity.get("contract") !=
                "ffac-v5-historical-B-EA-fullnative-grid-parity-v1"
                or self.historical_parity.get("exact_source_binding_sha256") !=
                file_sha256(self.source_binding_path)
                or self.historical_parity.get("current_calibration_manifest_sha256") !=
                file_sha256(self.calibration_manifest_path)
                or self.historical_chart.get("contract") !=
                "ffac-historical-B-chart-source-feasibility-v1"):
            raise ValueError("Historical B calibration or grid parity has another source/code basis")
        self.geometry = _read(self.geometry_params_path)
        self.response = load_ac28_response(self.response_path)
        self.identity_value = dict(
            contract="ffac-v5-full-native-source-processor-v1",
            calibration_manifest_sha256=file_sha256(self.calibration_manifest_path),
            source_binding_audit_sha256=file_sha256(self.source_binding_path),
            historical_calibration_manifest_sha256=file_sha256(self.historical_calibration_path),
            historical_parity_receipt_sha256=file_sha256(self.historical_parity_path),
            historical_chart_manifest_sha256=file_sha256(self.historical_chart_path),
            chart_registration_module_sha256=file_sha256(Path(inspect.getfile(register_frame_chart_xy8))),
            geometry_params_sha256=file_sha256(self.geometry_params_path),
            response_sha256=file_sha256(self.response_path),
            photometry_module_sha256=file_sha256(self.photometry_module_path),
            renderer=self.renderer_identity,
            trainer_feature_input="sha-bound-float32-log-camera-linear-sidecar",
            unit_denominator=65535,
            display_jpeg_is_training_input=False,
            corrected_signal_sidecar="native-float32-log-camera-linear")
        calibration_contract = self.calibration_manifest.get("contract")
        if candidate == "EA":
            if (calibration_contract != "ffac-ea-native-v5-donor-fixedpoint-v2"
                    or self.calibration_manifest.get("convergence_pass") is not True
                    or self.calibration_manifest.get("ea_fit_array_key") != "quality_area_log_rgb"
                    or self.calibration_manifest.get("port_module_sha256") !=
                    self.identity_value["photometry_module_sha256"]):
                raise ValueError("EA calibration is not converged or uses a different native basis")
        elif candidate == "A":
            if (calibration_contract != "ffac-v5-source-bound-calibration43-v1"
                    or self.calibration_manifest.get("module_sha256") !=
                    self.identity_value["photometry_module_sha256"]
                    or self.calibration_manifest.get("response_sha256") !=
                    self.identity_value["response_sha256"]):
                raise ValueError("A diagnostic numerical calibration/code/response mismatch")
        else:
            raise ValueError("Unknown full-native renderer candidate")
        self.rows = {r["run"]: r for r in self.calibration_manifest["results"]}
        if len(self.rows) != 43:
            raise ValueError("V5 calibration does not contain 43 unique FF_AC runs")
        self.binding_by_manifest = {r["old_manifest_sha256"]: r
                                    for r in self.source_binding["rows"]}
        self.manifest_path_by_sha = {r["old_manifest_sha256"]: Path(r["old_manifest_path"])
                                     for r in self.source_binding["rows"]}
        if len(self.binding_by_manifest) != 52:
            raise ValueError("V5 source audit does not contain 52 unique revisions")
        historical_rows = {r["source_manifest_sha256"]: r
                           for r in self.historical_calibration["revision_results"]}
        if (len(historical_rows) != 3
                or set(historical_rows) != {
                    "f18e54ed93b1711be5bb43269523bb078a69740285b94915b611b6eb3e8cedde",
                    "1a167b591ae74a90a0b6d37a7daad8a5cbf83ce40939c910cd12b34dcc81df25",
                    "e6e25869e9757dd347b57fb18874d386fb92b00de80c138df9c08045ee95443f"}):
            raise ValueError("Three changed historical B acquisitions are not separately fitted")
        parity_rows = {r["source_manifest_sha256"]: r for r in self.historical_parity["rows"]}
        if len(parity_rows) != 9:
            raise ValueError("Historical EA B grid parity does not cover nine revisions")
        self.historical_parity_rows = parity_rows
        self.fit_rows_by_manifest: dict[str, dict[str, Any]] = {}
        chart_rows = {r["source_manifest_sha256"]: r
                      for r in self.historical_chart["rows"]}
        if set(chart_rows) != set(historical_rows):
            raise ValueError("Historical B chart certificates do not cover the three refits")
        self.historical_chart_rows = chart_rows
        for digest, binding in self.binding_by_manifest.items():
            run = binding["run"]
            if binding["is_canonical"]:
                row = self.rows[run]
            elif digest in historical_rows:
                row = historical_rows[digest]
            else:
                row = self.rows[run]
                parity = parity_rows[digest]
                metric = parity["metrics"]
                if (parity["run"] != run
                        or parity["B_arw_sha256"] != binding["camera_arw_sha256"]
                        or parity["B_review_sha256"] != binding["original_review_jpeg_sha256"]
                        or parity["current_fit_sha256"] != row["baseline_sha256"]
                        or metric["max_absolute_log"] > 1e-5):
                    raise ValueError(f"Historical {run} B grid is not certified equal to current fit")
            self.fit_rows_by_manifest[digest] = row
        self._fits: dict[str, Any] = {}
        self._facies: dict[str, np.ndarray] = {}
        self._domain_provenance: dict[str, dict[str, Any]] = {}

    def _baseline_geometry(self, run: str, source_manifest_sha256: str,
                           binding: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        if source_manifest_sha256 in self.historical_chart_rows:
            row = self.historical_chart_rows[source_manifest_sha256]
            path = Path(row["chart_certificate_path"])
            expected_sha = row["chart_certificate_sha256"]
        else:
            current = next((r for r in self.binding_by_manifest.values()
                            if r["run"] == run and r["is_canonical"]), None)
            if current is None:
                raise ValueError(f"No canonical source-bound B geometry for {run}")
            path = Path(current["evidence"]["path"])
            expected_sha = current["evidence"]["sha256"]
        if file_sha256(path) != expected_sha:
            raise ValueError("V5 B registration certificate changed")
        cert = _read(path)
        sources = cert["sources"]
        native_exact = (sources["review_native_jpeg_sha256"] ==
                        binding["original_review_jpeg_sha256"])
        reused_identical_grid = (not binding["is_canonical"]
            and source_manifest_sha256 not in self.historical_chart_rows
            and self.historical_parity_rows[source_manifest_sha256]["metrics"][
                "max_absolute_log"] <= 1e-5)
        if (sources["arw_sha256"] != binding["camera_arw_sha256"]
                or sources["camera_jpeg_sha256"] != binding["camera_jpeg_sha256"]
                or not (native_exact or reused_identical_grid)):
            raise ValueError("V5 B chart registration belongs to another acquisition")
        return np.asarray(cert["review_to_camera_h"], np.float64), dict(
            path=str(path), sha256=expected_sha,
            historical_native_grid_reused_by_exact_EA_parity=reused_identical_grid)

    def identity(self) -> dict[str, Any]:
        return dict(self.identity_value)

    def _fit(self, run: str, source_manifest_sha256: str | None = None):
        row = (self.rows[run] if source_manifest_sha256 is None else
               self.fit_rows_by_manifest[source_manifest_sha256])
        key = str(row["baseline_path"])
        if key not in self._fits:
            path = Path(row["baseline_path"])
            if file_sha256(path) != row["baseline_sha256"] or file_sha256(path.with_suffix(".json")) != row["sidecar_sha256"]:
                raise ValueError(f"V5 B calibration changed for {run}")
            fit = load_baseline(path)
            if fit.run != run:
                raise ValueError("V5 B calibration run mismatch")
            sidecar = _read(path.with_suffix(".json"))
            fit_source = sidecar.get("source", {})
            if "historical_EA_grid_sha256" in row:
                if (fit_source.get("reference_ac28_baseline_sha256") !=
                        self.rows["ac28"]["baseline_sha256"]
                        or fit_source.get("port_module_sha256") !=
                        self.identity_value["photometry_module_sha256"]):
                    raise ValueError("Historical B fit uses a different AC28/photometry basis")
            elif fit_source.get("response_sha256") != self.identity_value["response_sha256"]:
                raise ValueError("V5 fit and full-native response differ")
            if self.renderer_identity["primary"] == "EA":
                if "ea_source_binding_path" in row:
                    binding_path = Path(row["ea_source_binding_path"])
                    if file_sha256(binding_path) != row["ea_source_binding_sha256"]:
                        raise ValueError("EA calibration source-array binding changed")
                    binding = _read(binding_path)
                    if (binding.get("source_array_key") != "quality_area_log_rgb"
                            or binding.get("source_arw_sha256") != sidecar["source"]["source_arw_sha256"]
                            or binding.get("fitted_baseline_sha256") != row["baseline_sha256"]):
                        raise ValueError("EA calibration was fitted on a different RAW basis")
                elif (sidecar["source"].get("source_manifest_sha256") != source_manifest_sha256
                      or sidecar["source"].get("historical_EA_grid_sha256") !=
                      row["historical_EA_grid_sha256"]):
                    raise ValueError("Historical EA B fit belongs to another source revision/grid")
            self._fits[key] = fit
        return self._fits[key]

    def _source(self, run: str, source_manifest_sha256: str,
                camera_arw: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        binding = self.binding_by_manifest.get(source_manifest_sha256)
        if binding is None or binding["run"] != run:
            raise ValueError("Frame belongs to an unfrozen V5 source revision")
        manifest_path = self.manifest_path_by_sha[source_manifest_sha256]
        if file_sha256(manifest_path) != source_manifest_sha256:
            raise ValueError("Frozen V5 source manifest changed")
        manifest = _read(manifest_path)
        fit_row = self.fit_rows_by_manifest[source_manifest_sha256]
        calibration = _read(Path(fit_row["baseline_path"]).with_suffix(".json"))
        calibration_source = calibration["source"]
        facies_sha = calibration_source["facies_sha256"]
        facies_path = Path(calibration_source["source_facies_path"])
        if facies_sha not in self._facies:
            if file_sha256(facies_path) != facies_sha:
                raise ValueError("Frozen native facies/geometry domain changed")
            with np.load(facies_path, allow_pickle=False) as z:
                facies = np.asarray(z["facies"], np.int16)
            if facies.shape != (2979, 5472):
                raise ValueError("Facies grid has a different native geometry")
            self._facies[facies_sha] = facies
        if source_manifest_sha256 not in self._domain_provenance:
            domain = manifest.get("domain")
            explicit_domain_binding = isinstance(domain, dict)
            relation = "same-revision-npz"
            lineage_manifest_sha = source_manifest_sha256
            if not isinstance(domain, dict):
                siblings = []
                for row in self.source_binding["rows"]:
                    if row["run"] != run or row["old_manifest_sha256"] == source_manifest_sha256:
                        continue
                    sibling = _read(Path(row["old_manifest_path"]))
                    if isinstance(sibling.get("domain"), dict):
                        if (sibling.get("content_box_xywh") != manifest.get("content_box_xywh")
                                or sibling.get("analysis_image_shape_hw") != manifest.get("analysis_image_shape_hw")):
                            raise ValueError("Historical facies sibling has different native geometry")
                        siblings.append((row["old_manifest_sha256"], sibling["domain"]))
                if not siblings:
                    raise ValueError("Legacy native source lacks a bound same-run facies sibling")
                siblings.sort(key=lambda item: item[0])
                lineage_manifest_sha, domain = siblings[0]
                relation = "same-run-frozen-sibling-array-identity"
            domain_path = Path(domain["path"])
            domain_sha = domain["sha256"]
            if file_sha256(domain_path) != domain_sha:
                raise ValueError("Frozen historical facies NPZ changed")
            with np.load(facies_path, allow_pickle=False) as current, np.load(domain_path, allow_pickle=False) as historical:
                if (not np.array_equal(current["facies"], historical["facies"])
                        or not np.array_equal(current["domain"], historical["domain"])):
                    raise ValueError("V5 calibration facies/domain differs from historical source")
            if domain_sha != facies_sha and relation == "same-revision-npz":
                relation = "same-revision-different-npz-array-identity"
            self._domain_provenance[source_manifest_sha256] = dict(
                relation=relation, lineage_manifest_sha256=lineage_manifest_sha,
                historical_domain_npz_path=str(domain_path), historical_domain_npz_sha256=domain_sha,
                calibration_domain_npz_path=str(facies_path), calibration_domain_npz_sha256=facies_sha,
                facies_and_domain_arrays_identical=True,
                historical_manifest_explicit_domain_binding=explicit_domain_binding,
                independent_historical_source_facies_sha_proven=explicit_domain_binding)
        if (calibration_source["facies_sha256"] != facies_sha or
                calibration_source["source_arw_sha256"] != binding["camera_arw_sha256"]):
            raise ValueError("V5 calibration does not bind this source's B ARW/facies")
        if (binding["is_canonical"] or "historical_EA_grid_sha256" in fit_row):
            if (calibration_source.get("source_review_sha256") !=
                    binding["original_review_jpeg_sha256"]
                    or calibration_source.get("source_camera_jpeg_sha256") !=
                    binding["camera_jpeg_sha256"]):
                raise ValueError("V5 B fit uses a different source-native registration triple")
        if Path(camera_arw).suffix.lower() != ".arw":
            raise ValueError("V5 frame is not a camera RAW acquisition")
        return self._facies[facies_sha], binding, self._domain_provenance[source_manifest_sha256]

    def _correct_signal(
        self, *, run: str, frame_id: str, camera_jpeg: Path,
        camera_arw: Path, review_jpeg: Path, source_manifest_sha256: str,
        content_shape_hw: tuple[int, int], content_box_xywh: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if content_shape_hw != (2979, 5472) or content_box_xywh != (416, 496, 5472, 2979):
            raise ValueError("Source is not the certified full-native review grid")
        facies, binding, domain_provenance = self._source(run, source_manifest_sha256, camera_arw)
        input_shas = dict(arw_sha256=file_sha256(camera_arw),
                          camera_jpeg_sha256=file_sha256(camera_jpeg),
                          review_native_jpeg_sha256=file_sha256(review_jpeg))
        linear, geometry_cert = self.renderer(
            camera_arw, camera_jpeg, review_jpeg,
            expected_source_hashes=input_shas, content_box=content_box_xywh)
        if linear.shape != (3, 2979, 5472) or linear.dtype != np.float32:
            raise ValueError("RAW renderer returned a non-native image")
        if geometry_cert.get("sources") != input_shas or geometry_cert.get("profile") not in ("early", "late"):
            raise ValueError("RAW renderer certificate does not bind the source triple")
        native_registration_gate = _native_registration_gate(geometry_cert)
        fit, ref = self._fit(run, source_manifest_sha256), self._fit("ac28")
        if (input_shas["arw_sha256"] == binding["camera_arw_sha256"]
                and input_shas["camera_jpeg_sha256"] == binding["camera_jpeg_sha256"]
                and input_shas["review_native_jpeg_sha256"] ==
                binding["original_review_jpeg_sha256"]):
            light = None
            light_qa = dict(mode="exact-frozen-B", zero_field=True)
            chart_registration = None
        else:
            if self.renderer_identity["primary"] == "EA":
                # Fixed-point EA fits used log of 8x8 area mean on the exact
                # full-native EA renderer output, not Gaussian-CFA samples.
                grid = _ea_native_light_grid(linear)
            else:
                grid = registered_raw_grid_from_arw(camera_arw,
                    geometry_cert["review_to_camera_h"], geometry_cert["profile"], self.geometry)
            baseline_h, baseline_h_source = self._baseline_geometry(
                run, source_manifest_sha256, binding)
            frame_xy, frame_usable, chart_registration = register_frame_chart_xy8(
                baseline_camera_jpeg=Path(binding["camera_jpeg"]),
                baseline_camera_sha256=binding["camera_jpeg_sha256"],
                frame_camera_jpeg=camera_jpeg,
                frame_camera_sha256=input_shas["camera_jpeg_sha256"],
                baseline_chart_xy8=fit.xy,
                baseline_chart_log_rgb=fit.CB,
                baseline_review_to_camera_h=baseline_h,
                frame_review_to_camera_h=geometry_cert["review_to_camera_h"],
                frame_native_linear_rgb=linear)
            chart_registration["baseline_registration_source"] = baseline_h_source
            light, light_qa = estimate_frame_light_grid(
                fit, grid, frame_chart_xy8=frame_xy,
                frame_chart_usable=frame_usable)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_linear = np.where(linear > 2, np.log(linear), np.nan).astype(np.float32)
        corrected = apply_v5(log_linear, fit, ref,
                             facies_native=facies, light_field8=light)
        corrected = np.asarray(corrected, np.float32)
        corrected, porous_hole_repair = _repair_isolated_porous_holes(corrected, facies)
        cert = dict(geometry=geometry_cert, native_registration_gate=native_registration_gate,
            per_frame_light=light_qa,
            frame_chart_registration=chart_registration,
            porous_hole_repair=porous_hole_repair,
            calibration_run=run,
            calibration_sha256=self.fit_rows_by_manifest[source_manifest_sha256]["baseline_sha256"],
            reference_calibration_sha256=self.rows["ac28"]["baseline_sha256"],
            facies_sha256=_read(Path(self.fit_rows_by_manifest[source_manifest_sha256]["baseline_path"]).with_suffix(".json"))["source"]["facies_sha256"],
            historical_facies_lineage=domain_provenance,
            source_manifest_sha256=source_manifest_sha256,
            frame_id=frame_id, source_binding_audit_sha256=self.identity_value["source_binding_audit_sha256"])
        return corrected, cert

    def correct(
        self, *, run: str, frame_id: str, camera_jpeg: Path,
        camera_arw: Path, review_jpeg: Path, source_manifest_sha256: str,
        content_shape_hw: tuple[int, int],
        content_box_xywh: tuple[int, int, int, int],
    ) -> CorrectedFrame:
        corrected, details = self._correct_signal(
            run=run, frame_id=frame_id, camera_jpeg=camera_jpeg,
            camera_arw=camera_arw, review_jpeg=review_jpeg,
            source_manifest_sha256=source_manifest_sha256,
            content_shape_hw=content_shape_hw, content_box_xywh=content_box_xywh)
        rgb = render_ac28_response(corrected, self.response)
        cert = dict(camera_arw_sha256=file_sha256(camera_arw),
            camera_jpeg_sha256=file_sha256(camera_jpeg),
            original_review_jpeg_sha256=file_sha256(review_jpeg),
            native_shape_hw=list(content_shape_hw), details=details)
        cert["transform_sha256"] = _digest(cert)
        return CorrectedFrame(rgb=rgb, log_linear_rgb=corrected, certificate=cert)

    def correct_composite_baseline(
        self, *, run: str, frame_id: str, members: list[dict[str, Any]],
        review_jpeg: Path, source_manifest_sha256: str,
        content_shape_hw: tuple[int, int],
        content_box_xywh: tuple[int, int, int, int],
        historical_prepared_sha256: str,
    ) -> CorrectedFrame:
        if run != "ac61" or frame_id != "frame_000" or len(members) != 3:
            raise ValueError("Only the frozen AC61 three-RAW control may be composited")
        signals, children = [], []
        for index, member in enumerate(members):
            corrected, child = self._correct_signal(
                run=run, frame_id=f"frame_000_member_{index}",
                camera_jpeg=Path(member["camera_jpeg"]),
                camera_arw=Path(member["camera_arw"]),
                review_jpeg=review_jpeg,
                source_manifest_sha256=source_manifest_sha256,
                content_shape_hw=content_shape_hw,
                content_box_xywh=content_box_xywh)
            signals.append(corrected)
            children.append(child)
        median = np.nanmedian(np.stack(signals, axis=0), axis=0).astype(np.float32)
        rgb = render_ac28_response(median, self.response)
        cert = dict(source_kind="new-v5-median-of-three-corrected-camera-raws",
            camera_members=[dict(camera_jpeg_sha256=m["camera_jpeg_sha256"],
                                 camera_arw_sha256=m["camera_arw_sha256"])
                            for m in members],
            historical_prepared_sha256=historical_prepared_sha256,
            exact_historical_composite_reconstruction=False,
            original_review_jpeg_sha256=file_sha256(review_jpeg),
            native_shape_hw=list(content_shape_hw), details=children)
        cert["transform_sha256"] = _digest(cert)
        return CorrectedFrame(rgb=rgb, log_linear_rgb=median, certificate=cert)

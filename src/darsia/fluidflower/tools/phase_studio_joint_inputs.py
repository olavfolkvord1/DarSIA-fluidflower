"""Stream fixed Studio phases into the two-reference concentration method.

The legacy route retains registered photograph photometry. An explicit v5
route reads the phase model's SHA-bound float32 scientific sidecars and needs
a separately sealed clearwater reference in that same camera-linear domain.
It never silently combines v5 RAW frames with JPEG-domain references.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower import concentration_configuration as configuration
from darsia.fluidflower import concentration_inputs as inputs
from darsia.fluidflower.tools.phase_mass_joint_optics import (
    build_joint_reference, joint_optical_signals,
)

FORMAT = "studio-joint-reference-inputs-v1"
V5_CLEARWATER_FORMAT = "ffac-v5-normalized-clearwater-reference-v1"
V5_CLEARWATER_PROCESSING_FORMAT = "ffac-v5-normalized-clearwater-processing-v1"
V5_RGB_ENCODING = "float32-camera-linear-unit-rgb"


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _archive(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return _sha(path)


def _resize(array, shape, interpolation=cv2.INTER_AREA):
    return cv2.resize(np.asarray(array), tuple(shape[::-1]), interpolation=interpolation)


def _resize_scientific_rgb(rgb, valid, shape):
    """Area-pool float RAW without mixing invalid transport zeros into sand."""
    source = np.asarray(rgb)
    support = np.asarray(valid)
    if (source.dtype != np.dtype("float32") or source.ndim != 3 or source.shape[-1] != 3
            or support.shape != source.shape[:2] or support.dtype != np.dtype("bool")):
        raise ValueError("Scientific pooling needs native float32 RGB and exact bool support")
    weight = _resize(support.astype(np.float32), shape)
    weighted = _resize(np.where(support[..., None], source, 0), shape)
    result = np.zeros((*shape, 3), np.float32)
    np.divide(weighted, weight[..., None], out=result, where=weight[..., None] > 0)
    return result


def _dimensions(config):
    dimensions = config["dimensions_m"]
    return float(dimensions["height"]), float(dimensions["width"])


def _v5_scientific_reader(inspected, domain):
    """Bind exactly the scientific float revision used for phase inference."""
    from darsia.fluidflower.phase_training.v5_linear_input import RunV5LinearInput

    if inspected["dataset"] != "ffac":
        raise ValueError("The v5 scientific float route is FF_AC-specific")
    manifest_path = inputs._path(inspected["native_manifest"])
    if _sha(manifest_path) != inspected["native_manifest_sha256"]:
        raise ValueError("Scientific native manifest changed")
    reader = RunV5LinearInput(
        native_manifest_path=manifest_path, run_id=inspected["run"],
        content_shape_hw=tuple(inspected["shape_rc"]),
        required_valid_mask=np.asarray(domain, dtype=bool),
    )
    if reader.manifest_sha256 != inspected["native_manifest_sha256"]:
        raise ValueError("Scientific reader differs from selected phase revision")
    return reader


def _v5_clearwater(reference, inspected, reader, domain):
    """Require a SHA-sealed RAW clearwater asset in the selected v5 domain.

    This function does not create the asset. A separate correction step must
    register and normalize its source before joint preparation can opt in.
    """
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("V5 joint input needs a sealed scientific clearwater receipt")
    receipt_path = inputs._path(reference["path"])
    if _sha(receipt_path) != reference["sha256"]:
        raise ValueError("Scientific clearwater receipt SHA-256 mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(Path(inspected["native_manifest"]).read_text(encoding="utf-8-sig"))
    photometry_sha = hashlib.sha256(json.dumps(
        manifest.get("v5_photometry"), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if (receipt.get("format") != V5_CLEARWATER_FORMAT
            or receipt.get("dataset") != "ffac"
            or receipt.get("run") != inspected["run"]
            or receipt.get("native_manifest_sha256") != reader.manifest_sha256
            or receipt.get("v5_photometry_sha256") != photometry_sha
            or receipt.get("encoding") != V5_RGB_ENCODING
            or receipt.get("shape_hw") != list(inspected["shape_rc"])):
        raise ValueError("Scientific clearwater does not match selected v5 radiometry/grid")
    for key in ("raw_source_sha256", "correction_receipt_sha256", "rgb_sha256", "valid_mask_sha256"):
        value = receipt.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Scientific clearwater lacks a valid {key}")
    correction_path = inputs._path(receipt["correction_receipt_path"], receipt_path.parent)
    if _sha(correction_path) != receipt["correction_receipt_sha256"]:
        raise ValueError("Scientific clearwater correction receipt changed")
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    if (correction.get("format") != V5_CLEARWATER_PROCESSING_FORMAT
            or correction.get("dataset") != "ffac"
            or correction.get("run") != inspected["run"]
            or correction.get("native_manifest_sha256") != reader.manifest_sha256
            or correction.get("v5_photometry_sha256") != photometry_sha
            or correction.get("raw_source_sha256") != receipt["raw_source_sha256"]
            or correction.get("rgb_sha256") != receipt["rgb_sha256"]
            or correction.get("valid_mask_sha256") != receipt["valid_mask_sha256"]):
        raise ValueError("Scientific clearwater correction differs from its sealed output")
    raw_path = inputs._path(correction["raw_source_path"], correction_path.parent)
    if _sha(raw_path) != receipt["raw_source_sha256"]:
        raise ValueError("Scientific clearwater RAW source changed")
    for path_key, sha_key in (
        ("camera_jpeg_path", "camera_jpeg_sha256"),
        ("baseline_review_jpeg_path", "baseline_review_jpeg_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
        ("calibration_manifest_path", "calibration_manifest_sha256"),
        ("photo_aggregate_path", "photo_aggregate_sha256"),
    ):
        if not correction.get(path_key) or not correction.get(sha_key):
            raise ValueError(f"Scientific clearwater lacks {path_key} source binding")
        path = inputs._path(correction[path_key], correction_path.parent)
        if _sha(path) != correction[sha_key]:
            raise ValueError(f"Scientific clearwater {path_key} changed")
    if (correction["source_manifest_sha256"] !=
            (manifest.get("v5_photometry") or {}).get("source_manifest_sha256")
            or correction["calibration_manifest_sha256"] !=
            (manifest.get("v5_photometry") or {}).get("processor", {}).get("calibration_manifest_sha256")
            or correction["photo_aggregate_sha256"] !=
            (manifest.get("v5_photometry") or {}).get("processor", {}).get("photo_source_aggregate_sha256")):
        raise ValueError("Scientific clearwater belongs to another Track1 photometry fit")
    rgb_path = inputs._path(receipt["rgb_path"], receipt_path.parent)
    valid_path = inputs._path(receipt["valid_mask_path"], receipt_path.parent)
    if _sha(rgb_path) != receipt["rgb_sha256"] or _sha(valid_path) != receipt["valid_mask_sha256"]:
        raise ValueError("Scientific clearwater array or support mask changed")
    rgb = np.load(rgb_path, allow_pickle=False, mmap_mode="r")
    valid = np.load(valid_path, allow_pickle=False, mmap_mode="r")
    shape = tuple(inspected["shape_rc"])
    if rgb.shape != (*shape, 3) or rgb.dtype != np.dtype("float32"):
        raise ValueError("Scientific clearwater must be full-native float32 RGB")
    if valid.shape != shape or valid.dtype != np.dtype("bool") or np.any(domain & ~valid):
        raise ValueError("Scientific clearwater lacks exact porous-domain support")
    if (not np.isfinite(rgb[valid]).all() or np.any(rgb[valid] < 0)
            or np.any(rgb[valid] > 1)):
        raise ValueError("Scientific clearwater has invalid camera-linear values")
    clear = np.array(rgb, dtype=np.float32, copy=True)
    clear[~valid] = 0
    return clear, np.asarray(valid), dict(receipt_path=str(receipt_path), receipt_sha256=reference["sha256"],
        rgb_path=str(rgb_path), rgb_sha256=receipt["rgb_sha256"],
        valid_mask_sha256=receipt["valid_mask_sha256"],
        raw_source_path=str(raw_path), raw_source_sha256=receipt["raw_source_sha256"],
        correction_receipt_path=str(correction_path),
        correction_receipt_sha256=receipt["correction_receipt_sha256"],
        v5_photometry_sha256=photometry_sha, encoding=V5_RGB_ENCODING)


def _validate_config(job, inspected, config):
    if job["run"] != inspected["run"] or job["dataset"] != inspected["dataset"]:
        raise ValueError("Selected experiment and inspected inputs differ")
    if any(inspected["identity"].get(key) != job["candidate"].get(key)
           for key in inputs.IDENTITY_KEYS):
        raise ValueError("Selected phase candidate differs from inspected inputs")
    shape = tuple(inspected["shape_rc"])
    if np.shape(config["baseline_rgb"]) != (*shape, 3):
        raise ValueError("Baseline and fixed phase grids differ")
    for key in ("labels", "domain", "depth_m", "porosity"):
        if np.shape(config[key]) != shape:
            raise ValueError(f"Registered {key} and fixed phase grids differ")
    provenance = config["provenance"]
    if provenance.get("native_manifest_sha256") != inspected["native_manifest_sha256"]:
        raise ValueError("Physical configuration belongs to another photograph manifest")
    expected_baseline = provenance.get("selected_phase_baseline", {})
    if expected_baseline.get("sha256") != _sha(inspected["baseline"]):
        raise ValueError("Registered optical baseline changed")
    height, width = _dimensions(config)
    dimensions = inspected["geometry_metadata"].get("dimensions")
    if dimensions is not None and not np.allclose(dimensions, (height, width), rtol=0, atol=1e-10):
        raise ValueError("Physical dimensions differ from fixed phase grid")
    domain = np.asarray(config["domain"], bool)
    if not domain.any():
        raise ValueError("Empty registered porous domain")
    for key in ("depth_m", "porosity"):
        value = np.asarray(config[key])
        if not np.isfinite(value).all() or np.any(value[domain] <= 0):
            raise ValueError(f"Invalid registered {key}")
    if np.any(np.asarray(config["porosity"]) > 1):
        raise ValueError("Porosity exceeds one")


def _clearwater_paths(job):
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer

    assumption = job.get("clearwater_reference_assumption")
    if assumption is not None:
        # The joint driver validates the donor/run relation before dispatch.
        # Repeat source verification here so direct preparation cannot silently
        # accept another photograph or changed bytes.
        from darsia.fluidflower.tools.phase_studio_joint_calibration_compare import (
            validate_clearwater_assumptions,
        )

        run = str(job["run"]).lower()
        donor = assumption.get("donor_run") if isinstance(assumption, dict) else None
        validate_clearwater_assumptions(
            {"clearwater_reference_assumptions": {run: assumption}},
            {run: job, donor: {}}, [run],
        )
        paths = [Path(source["path"]).resolve() for source in assumption["sources"]]
        for path, source in zip(paths, assumption["sources"], strict=True):
            if not path.is_file() or _sha(path).lower() != source["sha256"].lower():
                raise ValueError("Borrowed clearwater reference changed: " + str(path))
        return paths
    if job["dataset"] == "ffum":
        return transfer._ffum_clearwater_paths(job["run"])
    if job["dataset"] == "ffac":
        from darsia.fluidflower.series_paths import series_manifest
        raw_root = series_manifest("ffac").get("legacy", {}).get("raw_inputs")
        if not raw_root:
            raise ValueError("FF_AC series manifest has no raw clear-water source root")
        return transfer.select_clear_reference_paths(Path(raw_root), job["run"])
    raise ValueError("No registered clear-water reference for this series")


def _final_dose(inspected):
    stages = inspected.get("injection_stages") or []
    ends = [float(segment["end_h"]) for stage in stages
            for segment in stage.get("rate_segments", [])]
    if not ends:
        raise ValueError("A complete injection schedule is required")
    mass = inputs.nominal_injected_mass_g(stages, max(ends))
    if mass is None or not np.isfinite(mass) or mass <= 0:
        raise ValueError("A positive verified final injection dose is required")
    return float(mass)


def _retain_recorded_optical_rig(source, native):
    """Retained reviews bind an optical rig, independently of new project setup.

    The current project's experiment supplies acquisition identities only. Its
    rebuilt optical rig must not replace the serialized rig named by an older
    review. This changes this worker's objects, never the project or its files.
    """
    frozen = native.get("correction_contract") or {}
    declared_legacy = frozen.get("rig_source") == "legacy-v1"
    older_explicit_legacy = (
        frozen.get("rig_source") is None and bool(frozen.get("legacy_rig"))
        and frozen.get("source") == "FF-UM serialized rig shape/illumination data with the shared FF-AC uniform24 ColorChecker implementation"
    )
    if not (declared_legacy or older_explicit_legacy):
        return dict(policy="current_project_matches_recorded_rig_kind", replaced=False)
    import darsia
    import ffum
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
    from darsia.fluidflower.tools.phase_geometry_contract import require_matching_geometry
    from darsia.fluidflower.review_storage import resolve_review_path

    recorded_path = frozen.get("rig_path") or frozen.get("legacy_rig")
    if not recorded_path:
        raise ValueError("Retained legacy review does not identify its optical rig")
    path = resolve_review_path(Path(recorded_path))
    if frozen.get("rig_path") and frozen.get("legacy_rig") and path != resolve_review_path(Path(frozen["legacy_rig"])):
        raise ValueError("Retained review contains conflicting optical rig paths")
    correction_paths = sorted(path.glob("correction_*.npz"))
    if not correction_paths or not (path / "meta_data.json").is_file():
        raise ValueError("Retained review's recorded serialized optical rig is unavailable")
    files = [path / "meta_data.json", *correction_paths]
    bindings = [dict(path=str(p), sha256=_sha(p)) for p in files]
    reference = native["reference"]
    baseline_path = resolve_review_path(Path(reference["source_path"]))
    if not reference.get("source_sha256") or _sha(baseline_path) != reference["source_sha256"]:
        raise ValueError("Recorded pre-injection optical source changed")
    previous = source.correction_contract
    expected_shape = source.baseline.shape
    # Release the project rig before loading the recorded rig's large fields.
    source.rig = None
    rig = ffum.MuseumRig.load(path)
    rig.baseline_path = baseline_path
    rig.shape_corrected_baseline = darsia.imread(
        baseline_path, transformations=rig.shape_corrections).img_as(np.float32)
    rig.load_experiment(source.experiment)
    refinement = transfer._configure_ffum_colorchecker_refinement(rig)
    baseline = np.asarray(rig.baseline.img, np.float32)[..., :3]
    if baseline.shape != expected_shape:
        raise ValueError("Recorded optical rig geometry differs from retained photographs")
    actual = dict(previous, source="Recorded serialized FF-UM optical rig with current project acquisition protocols",
        rig_source="legacy-v1", rig_path=str(path), legacy_rig=str(path), raw_baseline=str(baseline_path),
        refinement=refinement, legacy_ffum_calibration_method_used=False)
    geometry = require_matching_geometry(native, actual, context="retained joint-reference optical rig")
    if any(_sha(item["path"]) != item["sha256"] for item in bindings):
        raise ValueError("Recorded optical correction changed while loading")
    source.rig, source.baseline, source.correction_contract = rig, baseline, actual
    return dict(policy="optical rig explicitly recorded by retained review; project protocols supply acquisition identities",
        replaced=True, original_manifest_rig_path=recorded_path, selected_rig_path=str(path),
        original_rig_path_field="rig_path" if frozen.get("rig_path") else "legacy_rig",
        selected_optical_artifacts=bindings, source_baseline_sha256=reference["source_sha256"],
        previous_project_rig_path=previous.get("rig_path"), previous_project_refinement=previous.get("refinement"),
        geometry_binding=geometry, validation="selected bytes hashed before/after loading; unchanged independent photograph parity must also pass")


def _prepare_clearwater(job, inspected, output, paths):
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
    from darsia.fluidflower.tools.concentration_movie_input_cache import MovieInputCache
    from darsia.fluidflower.tools.prepare_phase_injection_references import matching_source, _parity

    native = json.loads(Path(inspected["native_manifest"]).read_text(encoding="utf-8-sig"))
    with matching_source(job["dataset"], job["run"], native, output) as (source, binding):
        rig_selection = _retain_recorded_optical_rig(source, native)
        parity = _parity(source, native, Path(job["review_root"]), job["run"],
                         MovieInputCache(output / "reference_parity", job["run"]))
        if not parity.get("passed"):
            raise ValueError("Regenerated optical source does not match fixed Studio photographs")
        clear, contract = transfer._read_corrected_clearwater(source.rig, paths, source.baseline.shape)
        if clear.shape != (*inspected["shape_rc"], 3) or not np.isfinite(clear).all():
            raise ValueError("Corrected clearwater is not finite RGB on the registered grid")
        clipped_fraction = float(np.mean((clear < 0) | (clear > 1)))
        # This matches the existing Studio adapter's documented optical input
        # convention: clip the corrected wash before analysis-grid pooling.
        return np.clip(clear, 0, 1).astype(np.float32), dict(
            correction_contract=source.correction_contract, geometry_binding=binding,
            optical_rig_selection=rig_selection,
            source_parity=parity, clearwater_contract=contract,
            clearwater_clipped_fraction=clipped_fraction,
            clearwater_encoding="corrected normalized float RGB clipped to [0,1]; no JPEG re-encoding",
        )


class _PoreVolumes:
    """Compute native DarSIA extensive pore volume once, then pool each mask."""
    def __init__(self, config, shape):
        import darsia
        self.domain = np.asarray(config["domain"], bool)
        self.shape = tuple(shape)
        height, width = _dimensions(config)
        geometry = darsia.ExtrudedPorousGeometry(
            porosity=np.where(self.domain, config["porosity"], 0), depth=config["depth_m"],
            space_dim=2, num_voxels=self.domain.shape, dimensions=[height, width])
        domain = darsia.ScalarImage(self.domain.astype(np.float32), dimensions=[height, width])
        self.native = np.asarray(geometry.make_extensive(domain).img, np.float32)
        self.nominal_native = self.native.copy()
        self.fixed_pixel_area_m2 = height * width / self.domain.size
        self.pv = self._pool(self.native)

    def _pool(self, native):
        pooled = _resize(native, self.shape) * np.float32(native.size / np.prod(self.shape))
        if not np.isclose(native.sum(dtype=np.float64), pooled.sum(dtype=np.float64),
                          rtol=2e-5, atol=1e-12):
            raise ValueError("Native DarSIA phase-volume pooling failed conservation")
        return np.asarray(pooled, np.float32)

    def phases(self, labels, pixel_area_m2=None):
        phase = np.asarray(labels)
        if phase.shape != self.domain.shape or not np.isin(phase, [0, 1, 2]).all():
            raise ValueError("Invalid fixed categorical phase map")
        if np.any(phase[~self.domain]):
            raise ValueError("Fixed phase support extends outside the registered porous domain")
        if pixel_area_m2 is not None:
            from darsia.fluidflower.phase_pixel_area import validate_pixel_area
            area = validate_pixel_area(pixel_area_m2, self.domain.shape)
            native = self.nominal_native * (area / self.fixed_pixel_area_m2)
        else:
            native = self.nominal_native
        return tuple(self._pool(native * (phase == label)) for label in (1, 2))


def _physics(config, time_hours, phase, shape):
    # An exact empty negative control needs no invented/extrapolated P/T state.
    if not np.any(phase):
        zero = np.zeros(shape, np.float32)
        return zero, zero.copy(), None, True
    state = configuration.pressure_temperature_at(config, time_hours)
    from darsia.fluidflower.mass_materialization import _mass_analysis
    height, width = _dimensions(config)
    physical = _mass_analysis(shape, width, height, **state)
    solubility = np.broadcast_to(np.asarray(physical.solubility_co2, np.float32), shape).copy()
    density = np.broadcast_to(np.asarray(physical.density_gaseous_co2, np.float32), shape).copy()
    if not np.isfinite(solubility).all() or not np.isfinite(density).all() or np.min(solubility) <= 0 or np.min(density) <= 0:
        raise ValueError("Native DarSIA returned invalid thermodynamic properties")
    return solubility, density, state, False


def prepare_run(job, inspected, config, output, frame_indices, analysis_width=1536,
                update=None, *, scientific_reference=None):
    """Prepare a bounded frame selection; no fitting, phase changes or UI launch.

    Config must come from ``load_run_configuration(..., purpose='joint_reference')``.
    Registered PT rows (including the corrected RUN7 clock) remain unchanged;
    any bounded physical assumption is explicit in the binding and each frame.
    The returned manifest contains per-frame NPZ paths, not full movie arrays.
    ``scientific_reference`` opts FF_AC v5 into the phase model's SHA-bound
    float32 sidecars. It names a sealed, source-bound normalized RAW
    clearwater receipt. Without that receipt, a v5 source is blocked rather
    than fitted using the display JPEG and legacy corrected clearwater.
    """
    _validate_config(job, inspected, config)
    if scientific_reference is None and inspected.get("native_manifest"):
        native_path = inputs._path(inspected["native_manifest"])
        if _sha(native_path) != inspected["native_manifest_sha256"]:
            raise ValueError("Selected native manifest changed")
        native = json.loads(native_path.read_text(encoding="utf-8-sig"))
        if (native.get("v5_photometry") or {}).get("format") == (
                "ffac-v5-native-source-materialization-v1"):
            raise ValueError("V5 scientific float source requires a sealed normalized "
                             "float clearwater reference; JPEG-domain joint fit is blocked")
    if not isinstance(analysis_width, int) or not 1 <= analysis_width <= inspected["shape_rc"][1]:
        raise ValueError("Analysis width must be a positive non-upsampling integer")
    indices = list(frame_indices)
    if not indices or indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < len(inspected["frames"]) for i in indices):
        raise ValueError("Frame indices must be unique increasing valid indices")
    final_dose = _final_dose(inspected)
    callback = update or (lambda message: None)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scientific = scientific_reference is not None
    reader = _v5_scientific_reader(inspected, config["domain"]) if scientific else None
    if scientific:
        if not isinstance(scientific_reference, dict) or set(scientific_reference) != {"path", "sha256"}:
            raise ValueError("V5 joint input needs a sealed scientific clearwater receipt")
        reference_inputs = [dict(path=str(inputs._path(scientific_reference["path"])),
                                 sha256=scientific_reference["sha256"])]
        selected = [reader.frames[f"frame_{index:03d}"] for index in indices]
        scientific_binding = dict(
            encoding=V5_RGB_ENCODING, unit_denominator=reader.unit_denominator,
            decoder_runtime=reader.report()["runtime"],
            native_manifest_sha256=reader.manifest_sha256,
            invalid_policy=reader.invalid_policy,
            baseline_sidecar_sha256=reader.reference.get("v5_log_linear_rgb_sha256"),
            baseline_valid_mask_sha256=reader.reference.get("v5_log_linear_valid_mask_sha256"),
            frames=[dict(frame_index=index,
                sidecar_sha256=row.get("v5_log_linear_rgb_sha256"),
                valid_mask_sha256=row.get("v5_log_linear_valid_mask_sha256"))
                for index, row in zip(indices, selected)],
            clearwater_receipt=reference_inputs[0])
    else:
        paths = _clearwater_paths(job)
        reference_inputs = [dict(path=str(p), sha256=_sha(p)) for p in paths]
        scientific_binding = None
    binding = dict(format=FORMAT, run=job["run"], dataset=job["dataset"],
        model_id=job["model_id"], identity=inspected["identity"],
        native_manifest_sha256=inspected["native_manifest_sha256"],
        source_manifest_sha256=inspected["source_manifest_sha256"],
        physical_profile_sha256=config["provenance"]["scientific_profile_sha256"],
        physical_artifact_record_sha256=config["provenance"]["artifact_record_sha256"],
        physical_configuration_purpose=config["provenance"].get("purpose", "native"),
        pressure_temperature_assumption=config.get("pressure_temperature_assumption") or {},
        injection_stages=inspected["injection_stages"], expected_final_injected_mass_g=final_dose,
        reference_inputs=reference_inputs, scientific_float_input=scientific_binding,
        frame_indices=indices, analysis_width=analysis_width,
        input_code_sha256=_sha(__file__), joint_optics_code_sha256=_sha(__import__(build_joint_reference.__module__, fromlist=["__file__"]).__file__))
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "prepared.json"
    if manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if cached.get("binding_sha256") != key:
            raise ValueError("Existing preparation has another immutable input/method binding")
        artifacts = [(cached["references"], cached["references_sha256"])]
        artifacts.extend((f["arrays"], f["arrays_sha256"]) for f in cached["frames"])
        if cached.get("complete") and all(Path(p).is_file() and _sha(p) == s for p, s in artifacts):
            if scientific:
                # Cache reuse must recheck the scientific source, not just the
                # small receipt and already prepared NPZ files.
                for frame_id, photo in [("pre_injection", inspected["baseline"]), *[
                        (f"frame_{index:03d}", inspected["frames"][index]["photo"])
                        for index in indices]]:
                    _record, _photo_sha, sidecar, sidecar_sha, mask, mask_sha = (
                        reader._bound_source(inputs._path(photo), frame_id))
                    if _sha(sidecar) != sidecar_sha or (mask is not None and _sha(mask) != mask_sha):
                        raise ValueError("Cached scientific float source changed")
                _v5_clearwater(scientific_reference, inspected, reader, config["domain"])
            callback("Reusing verified prepared joint-reference inputs")
            return cached
        raise ValueError("Existing prepared joint inputs are incomplete or changed")
    shape = (round(inspected["shape_rc"][0] * analysis_width / inspected["shape_rc"][1]), analysis_width)
    if shape[0] < 1:
        raise ValueError("Analysis grid has zero height")
    callback("Verifying this run's scientific float photometry" if scientific else
             "Verifying current photometry and correcting this run's clearwater reference")
    if scientific:
        baseline_native = reader.load(inputs._path(inspected["baseline"]), "pre_injection")
        baseline_valid, _ = reader.load_validity(
            inputs._path(inspected["baseline"]), "pre_injection")
        clear, clear_valid, optical_provenance = _v5_clearwater(
            scientific_reference, inspected, reader, config["domain"])
        optical_provenance["baseline_scientific_float"] = reader.receipts[-1]
    else:
        baseline_native = config["baseline_rgb"]
        clear, optical_provenance = _prepare_clearwater(job, inspected, output, paths)
    if any(_sha(item["path"]) != item["sha256"] for item in reference_inputs):
        raise ValueError("Clearwater source changed during preparation")
    baseline = (_resize_scientific_rgb(baseline_native, baseline_valid, shape) if scientific
                else _resize(baseline_native, shape).astype(np.float32))
    clearwater = (_resize_scientific_rgb(clear, clear_valid, shape) if scientific
                  else _resize(clear, shape).astype(np.float32))
    del clear
    facies = _resize(np.asarray(config["labels"], np.float32), shape, cv2.INTER_NEAREST).astype(np.int16)
    volumes = _PoreVolumes(config, shape)
    reference = build_joint_reference(baseline, clearwater, facies,
        domain_mask=volumes.pv > 0,
        rgb_encoding="camera_linear" if scientific else "srgb")
    if not reference.usable:
        raise ValueError("This run's baseline/clearwater pair has no usable optical span")
    reference_path = output / "references.npz"
    reference_sha = _archive(reference_path, baseline=baseline, clearwater=clearwater, facies=facies, pv=volumes.pv,
        pore_volume=volumes.pv, denominator=reference.denominators["joint_chromatic_od"],
        reliability=reference.reliability["joint_chromatic_od"])
    (output / "frames").mkdir(exist_ok=True)
    rows = []
    for ordinal, index in enumerate(indices):
        callback(f"Preparing fixed phase/photo {ordinal + 1}/{len(indices)} for {job['run'].upper()}")
        frame = inputs.load_frame(inspected, index)
        if scientific:
            rgb_native = reader.load(inputs._path(inspected["frames"][index]["photo"]),
                                     f"frame_{index:03d}")
            frame_valid, _ = reader.load_validity(
                inputs._path(inspected["frames"][index]["photo"]), f"frame_{index:03d}")
            frame = replace(frame, corrected_rgb=rgb_native,
                source_reference={**frame.source_reference,
                    "effective_rgb_encoding": V5_RGB_ENCODING,
                    "scientific_float_source": reader.receipts[-1]})
        target = frame.nominal_injected_mass_g
        if target is None or not np.isfinite(target) or target < 0:
            raise ValueError("Verified nominal injected mass is required")
        # Some AC runs retain a small phase signal before this run's injection.
        # Preserve that observation rather than silently erasing it. Zero-dose
        # frames are excluded from the mass fit in fit_joint_summaries.
        zero_dose_phase_support_pixels = (int(np.count_nonzero(frame.phase_labels))
                                          if target == 0 else 0)
        av, gv = volumes.phases(frame.phase_labels, getattr(frame, "pixel_area_m2", None))
        rgb = (_resize_scientific_rgb(frame.corrected_rgb, frame_valid, shape) if scientific
               else _resize(frame.corrected_rgb, shape).astype(np.float32))
        signal = joint_optical_signals(rgb, reference)["joint_chromatic_od"]
        solubility, density, state, physical_not_required = _physics(config, frame.time_hours, frame.phase_labels, shape)
        physical_state_provenance = (dict(assumed=False, mode="not_required_empty_phase")
            if physical_not_required else configuration.pressure_temperature_provenance_at(config, frame.time_hours))
        path = output / "frames" / f"frame_{index:04d}.npz"
        digest = _archive(path, rgb=rgb, signal=signal, av=av, gv=gv, solubility=solubility, density=density)
        row = dict(frame_index=index, elapsed_h=frame.time_hours, source_name=frame.source_reference.get("source_name"),
            injected_mass_g=float(target), nominal_injected_mass_g=float(target), state=state,
            expected_final_injected_mass_g=final_dose,
            physical_not_required=physical_not_required, arrays=str(path), arrays_sha256=digest,
            zero_dose_phase_support_pixels=zero_dose_phase_support_pixels,
            physical_state_provenance=physical_state_provenance,
            aqueous_support_area_m2=float(np.sum(np.broadcast_to(
                (frame.pixel_area_m2 if getattr(frame, "pixel_area_m2", None) is not None else
                 np.prod(_dimensions(config)) / frame.phase_labels.size), frame.phase_labels.shape),
                where=frame.phase_labels == 1, dtype=np.float64)),
            aqueous_pore_volume_m3=float(av.sum(dtype=np.float64)), gas_pore_volume_m3=float(gv.sum(dtype=np.float64)),
            affected_pore_volume_m3=float(av.sum(dtype=np.float64) + gv.sum(dtype=np.float64)),
            source_reference=dict(frame.source_reference))
        if frame.source_reference.get("rig_domain_coverage"):
            coverage = frame.source_reference["rig_domain_coverage"]
            row.update(rig_domain_coverage=coverage,
                       quantification_complete=coverage["outside_native_grid_area_cm2"] == 0,
                       mass_eligible=False)
        rows.append(row)
        _write(path.with_suffix(".json"), row)
        del frame, av, gv, rgb, signal, solubility, density
    manifest = dict(format=FORMAT, complete=True, binding=binding, binding_sha256=key,
        run=job["run"], dataset=job["dataset"], model_id=job["model_id"],
        prepared_path=str(manifest_path), references=str(reference_path), references_sha256=reference_sha,
        analysis_width=analysis_width, shape_hw=list(shape), dimensions_m=config["dimensions_m"],
        frames=rows, physical_binding=config["provenance"], optical_binding=optical_provenance,
        provenance=optical_provenance, expected_final_injected_mass_g=final_dose,
        source_bindings={item["path"]: item["sha256"] for item in reference_inputs},
        reference_diagnostics=reference.diagnostics, reference_inputs=reference_inputs,
        current_correction_contract=inspected["correction_contract"],
        photometric_policy=("Exact source-bound v5 float32 camera-linear baseline/frames and sealed normalized float32 clearwater; display JPEGs are identity checks only"
            if scientific else "Current fixed Studio photographs and matching newly corrected per-run clearwater; historical September14 neutral-tone-only-v4 photographs are not substituted"),
        fixed_phase_support=True, phase_inference_repeated=False, calibration_label_exclusions_applied=False,
        zero_dose_phase_support=[dict(frame_index=row["frame_index"], elapsed_h=row["elapsed_h"],
                                      pixels=row["zero_dose_phase_support_pixels"])
                                 for row in rows if row["zero_dose_phase_support_pixels"]],
        method="joint_chromatic_od", concentration_fitted=False,
        precision="float32 arrays; float64 reductions; native DarSIA extensive geometry and thermodynamics",
        units=dict(signal="dimensionless signed optical response", volumes="m3", solubility="g/L", density="g/L", injected_mass="g"))
    _write(manifest_path, manifest)
    return manifest

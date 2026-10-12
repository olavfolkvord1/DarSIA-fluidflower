"""Approved G3 gas partition with an immutable inherited CO2 union.

The production decision is the literal exported-probability rule.  It does not
reuse the legacy optical-presence gate, source-eligibility gate or old gas
fallback.  Only labels 1/2 may change; inherited water and unresolved labels
stay byte-identical.  Diagnostic settings expose the same predeclared grid,
while the production artifact adapter accepts the approved setting only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Any

import cv2
import numpy as np

from .project import sha256_file


VERSION = "ffac-approved-g3-frozen-co2-partition-v1"
PROFILE_KEY = "approved_g3_gas"
G3_SHA256 = "90f1d4608506273974e4cc758573f4c6467a823243abdaa02ef214b7286c9692"
CO2_SHA256 = "8c3727110be01dbef4849fad6849e50bcc8e28bdd1ee22a1c24c0ab8027c64e3"
REFIT_SHA256 = "d33b3eea47392268d3ddebe2d9e9c695bfc640eb6e252d83670a3eaf3e7428be"


@dataclass(frozen=True)
class ApprovedG3Settings:
    format: str = VERSION
    pixel_threshold: float = .7
    regional_mode: str = "refit"
    regional_threshold: float = .6
    minimum_area_mm2: float = 2.5
    connectivity: int = 4

    def __post_init__(self):
        if (self.format != VERSION or self.pixel_threshold not in (.5, .6, .7)
                or self.regional_mode not in ("frozen", "refit", "off")
                or self.regional_threshold != .6 or self.minimum_area_mm2 not in (0., 1., 2.5)
                or self.connectivity != 4):
            raise ValueError("G3 decision settings differ from the prespecified configuration grid")


APPROVED_SETTINGS = ApprovedG3Settings()


def configuration_identity(settings=APPROVED_SETTINGS):
    return hashlib.sha256(json.dumps(asdict(settings), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _mask_identity(mask):
    shape = np.asarray(mask.shape, dtype="<i8")
    return hashlib.sha256(shape.tobytes() + np.packbits(np.asarray(mask, bool).reshape(-1),
                                                      bitorder="little").tobytes()).hexdigest()


def _check_scores(values, name, *, shape=None, allow_nan=False):
    array = np.asarray(values)
    if (array.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))
            or array.ndim != 2 or not array.size or (shape is not None and array.shape != shape)):
        raise ValueError(f"{name} requires floating-point probabilities on its declared grid")
    finite = np.isfinite(array)
    if np.isinf(array).any() or (not allow_nan and not finite.all()):
        raise ValueError(f"{name} has unsupported non-finite probabilities")
    if ((array[finite] < 0) | (array[finite] > 1)).any():
        raise ValueError(f"{name} probabilities are outside [0,1]")
    return array


def apply_g3_configuration(frozen_labels, g3_pixel, regional_probability,
                           pixel_area_mm2, settings=APPROVED_SETTINGS):
    """Apply thresholds and native four-connected minimum inside frozen CO2.

    Regional continuous probabilities are converted to float32 and resized
    with OpenCV INTER_LINEAR *before* thresholding, as in ``confirm_gas``.
    G3 NaNs are invalid evidence and fail its threshold; they never substitute
    the old gas head or restore a frozen gas pixel inside the CO2 union.
    """
    if not isinstance(settings, ApprovedG3Settings):
        raise TypeError("G3 decisions require an immutable settings object")
    labels = np.asarray(frozen_labels)
    if labels.dtype != np.uint8 or labels.ndim != 2 or not labels.size or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Frozen labels must be a native uint8 grid with values 0,1,2,255")
    g3 = _check_scores(g3_pixel, "G3 pixel", shape=labels.shape, allow_nan=True)
    area = float(pixel_area_mm2)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("G3 gas minimum requires a positive nominal pixel area")
    co2 = np.isin(labels, (1, 2))
    pixel_gas = co2 & (g3 >= settings.pixel_threshold)
    if settings.regional_mode == "off":
        regional_gas = pixel_gas.copy()
        source_shape = None
    else:
        values = _check_scores(regional_probability, "Regional gas")
        source_shape = list(values.shape)
        scores = cv2.resize(np.asarray(values, np.float32), labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
        regional_gas = pixel_gas & (scores >= settings.regional_threshold)
    gas = regional_gas.copy()
    withdrawn_components = 0
    if settings.minimum_area_mm2:
        _, components, statistics, _ = cv2.connectedComponentsWithStats(
            regional_gas.astype(np.uint8), connectivity=settings.connectivity)
        small = statistics[:, cv2.CC_STAT_AREA] * area < settings.minimum_area_mm2
        small[0] = False
        withdrawn_components = int(np.count_nonzero(small))
        gas[small[components]] = False
    result = labels.copy()
    result[co2] = np.where(gas[co2], 2, 1).astype(np.uint8)
    if not np.array_equal(np.isin(result, (1, 2)), co2):
        raise AssertionError("Approved G3 decision changed the immutable CO2 union")
    if not np.array_equal(result[~co2], labels[~co2]):
        raise AssertionError("Approved G3 decision changed frozen water or unresolved pixels")
    gas_count = int(np.count_nonzero(gas))
    audit = dict(format=VERSION, settings=asdict(settings),
                 configuration_sha256=configuration_identity(settings),
                 co2_union_mask_sha256=_mask_identity(co2), co2_support_exact=True,
                 outside_co2_labels_exact=True, co2_pixels=int(np.count_nonzero(co2)),
                 gas_pixels_before_regional=int(np.count_nonzero(pixel_gas)),
                 gas_pixels_before_minimum=int(np.count_nonzero(regional_gas)),
                 gas_pixels=gas_count, gas_area_mm2=gas_count * area,
                 pixel_area_mm2=area, withdrawn_gas_components=withdrawn_components,
                 withdrawn_gas_pixels=int(np.count_nonzero(regional_gas & ~gas)),
                 invalid_g3_pixels_inside_co2=int(np.count_nonzero(co2 & ~np.isfinite(g3))),
                 regional_source_shape_hw=source_shape,
                 regional_resize="float32 cv2.INTER_LINEAR to native grid before threshold",
                 minimum_policy="4-connected component area strictly smaller than minimum is aqueous",
                 optical_presence_gate_used=False, feature_eligibility_gate_used=False,
                 legacy_gas_fallback_used=False, mass_eligible=False)
    return result, audit


def apply_exported_g3(arrays: Mapping[str, Any], pixel_area_mm2,
                      settings=APPROVED_SETTINGS):
    """The same production decision for decoded probability-export arrays."""
    co2 = np.asarray(arrays["co2"])
    unresolved = np.asarray(arrays["frozen_unresolved"])
    if (co2.dtype != bool or unresolved.dtype != bool or co2.shape != unresolved.shape
            or co2.ndim != 2 or np.any(co2 & unresolved)):
        raise ValueError("Exported frozen CO2 and unresolved masks must be disjoint native bool grids")
    frozen = np.zeros(co2.shape, np.uint8)
    frozen[co2] = 1
    frozen[unresolved] = 255
    if "frozen_labels" in arrays:
        supplied = np.asarray(arrays["frozen_labels"])
        if (supplied.dtype != np.uint8 or supplied.shape != co2.shape
                or not np.array_equal(np.isin(supplied, (1, 2)), co2)
                or not np.array_equal(supplied == 255, unresolved)):
            raise ValueError("Supplied frozen labels differ from exported union and unresolved masks")
        frozen = supplied
    probability = None if settings.regional_mode == "off" else arrays["regional_" + settings.regional_mode]
    return apply_g3_configuration(frozen, arrays["g3_pixel"], probability, pixel_area_mm2, settings)


class ApprovedG3GasPath:
    """Artifact-bound CPU gas adapter for future RAW or cached-frame callers."""
    def __init__(self, g3_model_path, regional_model_path, co2_model_path,
                 settings=APPROVED_SETTINGS):
        if settings != APPROVED_SETTINGS:
            raise ValueError("The production adapter requires Olav's approved refit/.7/2.5 setting")
        paths = {"g3": Path(g3_model_path).resolve(), "regional": Path(regional_model_path).resolve(),
                 "co2": Path(co2_model_path).resolve()}
        expected = {"g3": G3_SHA256, "regional": REFIT_SHA256, "co2": CO2_SHA256}
        actual = {name: sha256_file(path) for name, path in paths.items()}
        if actual != expected:
            raise ValueError("Production G3, refit regional or frozen CO2 artifact changed")
        self.model = json.loads(paths["g3"].read_text(encoding="utf-8-sig"))
        self.regional_model = json.loads(paths["regional"].read_text(encoding="utf-8-sig"))
        if (self.model.get("variant") != "G3" or len(self.model.get("feature_names") or ()) != 4
                or not self.model["fit"].get("converged")):
            raise ValueError("Approved G3 artifact is not a converged four-feature gas head")
        self.settings = settings
        self.binding = dict(format=VERSION, settings=asdict(settings),
                            configuration_sha256=configuration_identity(settings),
                            artifacts={name: dict(path=str(paths[name]), sha256=actual[name]) for name in paths},
                            production_code_sha256=sha256_file(Path(__file__)),
                            generic_gas_bundle_replaced=False, co2_head_retrained=False)

    def pixel_probability(self, flatfield_features, valid_mask=None):
        from .tools.train_ffac_gas_flatfield_variants import gas_probability
        features = np.asarray(flatfield_features)
        if features.dtype != np.float32 or features.ndim < 2 or features.shape[-1] != 4:
            raise ValueError("G3 scientific features require float32 with four flat-field columns")
        valid = np.isfinite(features).all(-1)
        if valid_mask is not None:
            mask = np.asarray(valid_mask)
            if mask.dtype != bool or mask.shape != valid.shape:
                raise ValueError("G3 feature validity mask differs from its feature grid")
            valid &= mask
        safe = np.where(valid[..., None], features, 0)
        score = gas_probability(self.model, np.empty((*features.shape[:-1], 0), np.float32), safe)
        score[~valid] = np.nan
        return score

    def regional_probability(self, prepared):
        from .phase_optical_regions import probability
        return probability(prepared["features"], self.regional_model["heads"]["gas_within_co2"])

    def apply(self, frozen_labels, g3_pixel, regional_probability, pixel_area_mm2):
        result, audit = apply_g3_configuration(frozen_labels, g3_pixel, regional_probability,
                                               pixel_area_mm2, self.settings)
        return result, dict(audit, artifact_binding=self.binding)


def load_approved_profile(root, profile):
    """Validate an explicit opt-in without replacing the legacy generic head."""
    root = Path(root).resolve()
    marker = profile[PROFILE_KEY]
    required = {"format", "settings", "g3", "regional", "co2", "frame_bindings_directory"}
    if (not isinstance(marker, dict) or set(marker) != required or marker["format"] != VERSION
            or ApprovedG3Settings(**marker["settings"]) != APPROVED_SETTINGS
            or marker["frame_bindings_directory"] != "approved_g3_frame_bindings"):
        raise ValueError("Invalid approved G3 production profile")
    if any(name in profile for name in ("output_guards", "bottom_gas_guard", "region_edge_guard",
                                       "aqueous_temporal_support")):
        raise ValueError("Approved G3 cannot stack stages that change the frozen CO2 partition")
    paths = {}
    for name, expected in (("g3", G3_SHA256), ("regional", REFIT_SHA256), ("co2", CO2_SHA256)):
        entry = marker[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("Approved G3 artifact binding is incomplete")
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or entry["sha256"] != expected or sha256_file(path) != expected:
            raise ValueError("Approved G3 bound artifact changed or escaped its candidate")
        paths[name] = str(path)
    if marker["regional"] != profile["model"]:
        raise ValueError("Approved G3 regional artifact differs from the ordinary runtime model")
    if marker["co2"]["path"] != "models/generic/co2_presence_signal.json":
        raise ValueError("Approved G3 must retain the ordinary frozen CO2 bundle")
    runtime = dict(root=str(root), artifact_paths=paths,
                   frame_bindings_directory=str(root / marker["frame_bindings_directory"]))
    return dict(profile, _approved_g3_runtime=runtime)


def approved_runtime_identity(inherited_identity, marker):
    value = dict(inherited_identity=inherited_identity, marker=marker,
                 code_sha256=sha256_file(Path(__file__)), numpy=np.__version__, opencv=cv2.__version__)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _bound_json(entry, label):
    path = Path(entry["path"]).resolve()
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Approved G3 {label} changed")
    return json.loads(path.read_text(encoding="utf-8-sig")), path


def load_bound_probability_frame(binding_path, *, run, frame_index, elapsed_h,
                                 native_manifest_sha256):
    """Read a copied or original immutable export without mutating its receipt."""
    from .tools.train_ffac_gas_flatfield_variants import _read_frozen_map
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8-sig"))
    if (binding.get("format") != VERSION or str(binding.get("run")).casefold() != str(run).casefold()
            or binding.get("frame_index") != frame_index
            or float(binding.get("elapsed_h", np.nan)) != float(elapsed_h)
            or binding.get("native_manifest_sha256") != native_manifest_sha256):
        raise ValueError("Approved G3 frame binding differs from the current source/frame/time")
    receipt, receipt_path = _bound_json(binding["probability_receipt"], "probability receipt")
    metadata, source = receipt["metadata"], receipt["binding"]
    if (receipt.get("contract") != "ffac-g3-downstream-probability-layers-v1"
            or receipt.get("complete") is not True or source.get("provisional") is not False
            or source.get("g3_model_sha256") != G3_SHA256
            or source.get("regional_refit_sha256") != REFIT_SHA256
            or source.get("native_manifest_sha256") != native_manifest_sha256
            or metadata.get("co2_head_sha256") != CO2_SHA256
            or str(metadata.get("run")).casefold() != str(run).casefold()
            or metadata.get("frame_index") != frame_index
            or float(metadata.get("elapsed_h", np.nan)) != float(elapsed_h)):
        raise ValueError("Approved G3 probability receipt identifies other scientific inputs")
    folder = Path(binding.get("probability_folder") or receipt_path.parent).resolve()
    required = ("g3_pixel.npy", "regional_refit.npy", "co2.packbits.npy", "extra_masks.packbits.npz")
    files = {}
    for name in required:
        path = folder / name
        if receipt["files"][name]["sha256"] != sha256_file(path):
            raise ValueError("Approved G3 probability artifact changed")
        files[name] = path
    shape = tuple(metadata["native_shape_hw"])
    count = int(np.prod(shape))
    raw_g3 = np.load(files["g3_pixel.npy"], allow_pickle=False)
    if raw_g3.dtype == np.uint8:
        g3 = raw_g3.astype(np.float32) / 254
        g3[raw_g3 == 255] = np.nan
    else:
        g3 = raw_g3.astype(np.float32)
    regional = np.load(files["regional_refit.npy"], allow_pickle=False).astype(np.float32)
    co2 = np.unpackbits(np.load(files["co2.packbits.npy"], allow_pickle=False), bitorder="little",
                        count=count).reshape(shape).astype(bool)
    with np.load(files["extra_masks.packbits.npz"], allow_pickle=False) as archive:
        unresolved_row = archive["masks"][metadata["masks_order"].index("frozen_unresolved")]
    unresolved = np.unpackbits(unresolved_row, bitorder="little", count=count).reshape(shape).astype(bool)
    frozen_entry = binding["frozen_map"]
    frozen_path = Path(frozen_entry["path"]).resolve()
    if sha256_file(frozen_path) != frozen_entry["sha256"]:
        raise ValueError("Approved G3 inherited frozen map changed")
    frozen, _rgba = _read_frozen_map(frozen_path, binding["content_box_xywh"])
    if (frozen.shape != shape or not np.array_equal(np.isin(frozen, (1, 2)), co2)
            or not np.array_equal(frozen == 255, unresolved)
            or source.get("frozen_map_sha256") != frozen_entry["sha256"]):
        raise ValueError("Approved G3 inherited frozen CO2/unresolved partition differs")
    return dict(frozen_labels=frozen, g3_pixel=g3, regional_probability=regional,
                pixel_area_mm2=float(metadata["pixel_area_mm2"]),
                provenance=dict(frame_binding_sha256=sha256_file(Path(binding_path)),
                                probability_receipt_sha256=binding["probability_receipt"]["sha256"],
                                native_manifest_sha256=native_manifest_sha256,
                                frame_index=frame_index, elapsed_h=float(elapsed_h)))


from .phase_optical_regions_runtime import OpticalRegionSequence


class ApprovedG3GasSequence(OpticalRegionSequence):
    """Fail-closed ordinary-runtime adapter; exact-type fused paths reject it."""
    def __init__(self, context, profile):
        super().__init__(context, profile)
        runtime = profile.get("_approved_g3_runtime")
        if not runtime:
            raise ValueError("Approved G3 runtime requires a validated explicit profile")
        paths = runtime["artifact_paths"]
        self.gas_path = ApprovedG3GasPath(paths["g3"], paths["regional"], paths["co2"])
        self.frame_bindings_directory = Path(runtime["frame_bindings_directory"])
        self.frame_payload = None
        self.last_labels = None

    def bind_frame(self, *, frozen_labels, g3_pixel, regional_probability,
                   pixel_area_mm2, elapsed_h, provenance=None):
        if self.frame_payload is not None:
            raise ValueError("An approved G3 frame binding was not consumed")
        if not np.isfinite(float(elapsed_h)):
            raise ValueError("Approved G3 frame requires an explicit acquisition time")
        self.frame_payload = dict(frozen_labels=np.asarray(frozen_labels),
                                  g3_pixel=np.asarray(g3_pixel), regional_probability=np.asarray(regional_probability),
                                  pixel_area_mm2=float(pixel_area_mm2), elapsed_h=float(elapsed_h),
                                  provenance=dict(provenance or {}))

    def bind_cached_frame(self, *, run, frame_index, elapsed_h, native_manifest_sha256):
        path = self.frame_bindings_directory / str(run).casefold() / f"frame_{frame_index:03d}.json"
        frame = load_bound_probability_frame(path, run=run, frame_index=frame_index,
                                             elapsed_h=elapsed_h,
                                             native_manifest_sha256=native_manifest_sha256)
        self.bind_frame(**frame, elapsed_h=elapsed_h)

    def apply_with_reference(self, predictions, elapsed_h, current):
        if self.frame_payload is None:
            raise ValueError("Approved G3 requires an exact frozen-CO2/probability frame binding")
        payload = self.frame_payload
        if (float(elapsed_h) != payload["elapsed_h"]
                or np.asarray(current).shape[:2] != payload["frozen_labels"].shape):
            raise ValueError("Approved G3 frame binding differs from the requested image/time")
        labels, report = self.gas_path.apply(payload["frozen_labels"], payload["g3_pixel"],
                                           payload["regional_probability"], payload["pixel_area_mm2"])
        self.frame_payload = None
        self.last_labels = labels
        self.last_diagnostics = {"approved_g3_gas": labels == 2,
                                 "approved_g3_unresolved": labels == 255}
        co2 = np.isin(labels, (1, 2)).astype(np.float32)
        gas = (labels == 2).astype(np.float32)
        result = dict(co2_presence_probability=co2, gas_probability_conditional_on_co2=gas,
                      untouched_water_probability=1-co2, aqueous_co2_probability=co2*(1-gas),
                      free_co2_gas_probability=co2*gas)
        return result, dict(report, elapsed_h=float(elapsed_h), frame_binding=payload["provenance"],
                            classification_scores_not_concentration=True,
                            old_observation_retention_applied=False)

    def exact_phase_masks(self):
        if self.last_labels is None:
            raise ValueError("Approved G3 has no completed frame decision")
        return {"aqueous": self.last_labels == 1, "gas": self.last_labels == 2,
                "mixed": np.zeros_like(self.last_labels, bool), "unresolved": self.last_labels == 255}


__all__ = ["ApprovedG3Settings", "APPROVED_SETTINGS", "ApprovedG3GasPath", "apply_g3_configuration",
           "apply_exported_g3", "configuration_identity", "VERSION", "PROFILE_KEY",
           "ApprovedG3GasSequence", "load_approved_profile", "load_bound_probability_frame"]

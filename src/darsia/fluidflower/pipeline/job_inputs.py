"""Input catalogues from named native/phase/N1 consumers; metadata/stat only."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import ntpath
from pathlib import Path
import re
import stat
from ..store_map import input_reference, validate_input_reference

CATALOG_FORMAT = "darsia-role-input-catalog-v1"
MAX_METADATA_BYTES = 128 * 1024 * 1024


def _key(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _json_bytes(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate metadata key: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite metadata constant: " + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Input metadata must be a JSON object")
    return value


class InputCatalog:
    """Read-only role preparation, never an execution or SHA-check bypass.

    Original metadata bytes are SHA-verified. Payload SHA is the original
    descriptor's value; actual full payload verification remains J3 pending.
    """
    def __init__(self, roots, aliases=(), *, metadata_bytes=None, file_stat=None):
        self.roots, self.aliases = dict(roots), {}
        for row in aliases:
            logical, physical = row["logical"], row["physical"]
            if logical["sha256"] != physical["sha256"]:
                raise ValueError("Input alias changes source SHA")
            if "size_bytes" in logical and logical["size_bytes"] != physical.get("size_bytes"):
                raise ValueError("Input alias changes source size")
            key = _key(logical["path"])
            if key in self.aliases and self.aliases[key] != physical:
                raise ValueError("Conflicting source input aliases")
            self.aliases[key] = dict(physical)
        self.metadata_bytes = metadata_bytes or self._read_metadata
        self.file_stat = file_stat or (lambda path: Path(path).stat())
        self.entries, self.documents, self.sizes = {}, {}, {}
        self.consumers = {}
        self.source_setups = set()

    @staticmethod
    def _read_metadata(path):
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("Explicit input metadata exceeds 128 MiB: " + str(path))
        return raw

    def physical(self, descriptor):
        if (not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str)
                or re.fullmatch(r"[0-9a-f]{64}", descriptor.get("sha256", "")) is None):
            raise ValueError("Consumed input needs its original path and SHA")
        result = self.aliases.get(_key(descriptor["path"]), descriptor)
        if result["sha256"] != descriptor["sha256"]:
            raise ValueError("Input locator and source identity differ")
        return result

    def add(self, descriptor, role, use):
        actual = self.physical(descriptor)
        # Root admission precedes any stat/open. Passive provenance is not visited.
        reference = input_reference(actual["path"], self.roots,
            sha256=descriptor["sha256"], size_bytes=0, role=role)
        identity = (_key(actual["path"]), descriptor["sha256"])
        if identity not in self.sizes:
            info = self.file_stat(actual["path"])
            if not stat.S_ISREG(info.st_mode) or type(info.st_size) is not int or info.st_size < 0:
                raise ValueError("Consumed input is not a regular file: " + str(actual["path"]))
            self.sizes[identity] = info.st_size
        size = self.sizes[identity]
        declared = descriptor.get("size_bytes", actual.get("size_bytes"))
        if declared is not None and (type(declared) is not int or declared != size):
            raise ValueError("Consumed input size changed: " + str(actual["path"]))
        reference["bytes"] = size  # Actual stat, never guessed from provenance.
        key = (reference["root_alias"], reference["relative_path"], reference["sha256"], role)
        if not isinstance(use, str) or not use:
            raise ValueError("Consumed input needs a named original consumer")
        previous = self.consumers.get(use)
        if previous is not None and previous != reference:
            raise ValueError("Ambiguous input consumer maps to different references: " + use)
        self.consumers[use] = deepcopy(reference)
        row = self.entries.setdefault(key, {"reference": reference, "uses": []})
        if use not in row["uses"]:
            row["uses"].append(use)
        return deepcopy(reference)

    def doc(self, descriptor, role, use):
        reference = self.add(descriptor, role, use)
        actual = self.physical(descriptor)
        key = (_key(actual["path"]), descriptor["sha256"])
        if key not in self.documents:
            raw = self.metadata_bytes(actual["path"])
            if (not isinstance(raw, bytes) or len(raw) > MAX_METADATA_BYTES
                    or len(raw) != reference["bytes"]
                    or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]):
                raise ValueError("Consumed metadata SHA/size changed: " + str(actual["path"]))
            self.documents[key] = _json_bytes(raw)
        return deepcopy(self.documents[key])

    def pair(self, value, path_key, sha_key, role, use):
        return self.add({"path": value[path_key], "sha256": value[sha_key]}, role, use)

    def pair_doc(self, value, path_key, sha_key, role, use):
        return self.doc({"path": value[path_key], "sha256": value[sha_key]}, role, use)

    def full(self, value, use):
        centre = value["native_centre"]
        self.add(centre["log_linear_rgb"], "image", use + ".centre.log")
        if "valid_mask" in centre:
            self.add(centre["valid_mask"], "rig_model", use + ".centre.valid")
        for index, window in enumerate(value["ring_windows"]):
            self.add(window["log_linear_rgb"], "image", use + ".ring." + str(index))

    def original_native_row(self, row, parameters, use):
        keys = parameters["fields"]["native"]
        if len(keys) != 4:
            raise ValueError("Unsupported native descriptor-field schema")
        self.pair(row, keys[0], keys[1], "image", use + ".masked_raw")
        self.pair(row, keys[2], keys[3], "rig_model", use + ".masked_valid")
        temporal = parameters["fields"]["temporal"]
        if len(temporal) < 2:
            raise ValueError("Unsupported temporal descriptor-field schema")
        if all(row.get(key) is not None for key in temporal):
            self.pair(row, temporal[0], temporal[1], "image", use + ".temporal_ring")

    def original_acquisition(self, value, parameters, use):
        row = self.doc(value["native_row"], "configuration", use + ".native_row")
        self.full(value["full"], use + ".full")
        self.original_native_row(row, parameters, use)

    def mass(self, config, frames, *, standalone=False):
        self.doc(config["calibration"], "calibration", "mass.calibration")
        self.doc(config["physical_correction"], "rig_model", "mass.physical_correction")
        shared = config["shared"]
        for key in ("physical_reference", "floor_curve"):
            self.add(shared[key], "rig_model", "mass." + key)
        if isinstance(shared["depth_points"], dict):
            self.add(shared["depth_points"], "rig_model", "mass.depth_points")
        if ("reference_matrix" not in shared and isinstance(shared["physical_coordinate_provenance"], dict)
                and "path" in shared["physical_coordinate_provenance"]):
            self.doc(shared["physical_coordinate_provenance"], "rig_model", "mass.coordinate_provenance")
        for key in ("rgb", "valid"):
            self.add(shared["clearwater"][key], "image" if key == "rgb" else "rig_model", "mass.clearwater." + key)
        optics = shared["optics"]
        self.doc(optics["metadata"], "model", "mass.optics.metadata")
        for group in ("color_paths", "signals"):
            for label, bound in optics[group].items():
                self.add(bound, "model", "mass.optics." + group + "." + str(label))
        self.add(optics["flash"], "model", "mass.optics.flash")
        if "path" in config["boxes"]:
            self.add(config["boxes"], "rig_model", "mass.boxes")
        for frame in frames:
            use = "frame." + frame["case_id"] + ".mass"
            self.add(frame["photo_top"], "rig_model", use + ".photo_top")
            self.add(frame["original_domain"]["parent_labels_map"], "rig_model",
                     use + ".original_domain.parent_labels_map")
            if standalone:
                self.full(frame["current"], use + ".current")
                self.full(frame["baseline"], use + ".baseline")
                for key in ("labels", "classes", "area"):
                    self.add(frame[key], "model" if key in {"labels", "classes"} else "rig_model", use + "." + key)

    def statistics(self, config, frames, *, standalone=False):
        """Original summary-source consumers, omitting only fitted mass calibration."""
        self.doc(config["physical_correction"], "rig_model", "mass.physical_correction")
        shared = config["shared"]
        for key in ("physical_reference", "floor_curve"):
            self.add(shared[key], "rig_model", "mass." + key)
        if isinstance(shared["depth_points"], dict):
            self.add(shared["depth_points"], "rig_model", "mass.depth_points")
        if ("reference_matrix" not in shared and isinstance(shared["physical_coordinate_provenance"], dict)
                and "path" in shared["physical_coordinate_provenance"]):
            self.doc(shared["physical_coordinate_provenance"], "rig_model", "mass.coordinate_provenance")
        for key in ("rgb", "valid"):
            self.add(shared["clearwater"][key], "image" if key == "rgb" else "rig_model", "mass.clearwater." + key)
        optics = shared["optics"]
        self.doc(optics["metadata"], "model", "mass.optics.metadata")
        for group in ("color_paths", "signals"):
            for label, bound in optics[group].items():
                self.add(bound, "model", "mass.optics." + group + "." + str(label))
        self.add(optics["flash"], "model", "mass.optics.flash")
        if "path" in config["boxes"]:
            self.add(config["boxes"], "rig_model", "mass.boxes")
        for frame in frames:
            use = "frame." + frame["case_id"] + ".mass"
            self.add(frame["photo_top"], "rig_model", use + ".photo_top")
            self.add(frame["original_domain"]["parent_labels_map"], "rig_model",
                     use + ".original_domain.parent_labels_map")
            if standalone:
                self.full(frame["current"], use + ".current")
                self.full(frame["baseline"], use + ".baseline")
                for key in ("labels", "classes", "area"):
                    self.add(frame[key], "model" if key in {"labels", "classes"} else "rig_model", use + "." + key)

    def phase(self, config):
        phase = self.doc(config["phase"]["configuration"], "configuration", "phase.configuration")
        p = phase["parameters"]
        native = self.doc(config["native"]["configuration"], "configuration", "native.configuration")
        for key in ("reference", "initial_previous"):
            self.original_acquisition(config["phase"][key], p["inputs"], "phase." + key)
        bundles = phase["model_bundles"]
        for role in ("native", "edge"):
            item = bundles[role]
            for key in item["requirements"]["expected_sha256"]:
                self.doc(item["documents"][key], "model", "phase.model." + role + "." + key)
        for key in ("pixel_model", "regional_model"):
            self.doc(bundles["placement"][key], "model", "phase.placement." + key)
        if bundles["placement"].get("projected_models") is not None:
            for key in ("pixel", "regional"):
                self.doc(bundles["placement"]["projected_models"][key], "model", "phase.placement.projected." + key)
        chain = phase["chain_assets"]
        envelopes = self.doc(chain["envelopes"], "model", "phase.envelopes")
        for key in ("envelope1", "envelope2"):
            self.add(envelopes[key], "rig_model", "phase." + key)
        for key in ("floor", "zones"):
            self.add(chain[key], "rig_model", "phase." + key)
        return phase, native

    def retained(self, binding, parameters, use):
        parent = self.doc(binding["parent_native_manifest"], "configuration", use + ".parent_native_manifest")
        receipt = self.doc(binding["parent_frame_receipt"], "model", use + ".parent_frame_receipt")
        exported = self.doc(binding["parent_probability_receipt"], "model", use + ".parent_probability_receipt")
        self.doc({"path": exported["metadata"]["native_manifest_path"], "sha256": exported["binding"]["native_manifest_sha256"]},
                 "configuration", use + ".original_probability_native_manifest")
        frame = parent["frame_sources"][binding["frame_index"]]
        for name in parameters["parent_array_names"]:
            self.pair(frame, name + "_path", name + "_sha256", "image", use + ".parent." + name)
        self.add(exported["files"][parameters["mask_file"]], "rig_model", use + ".masks")
        self.add(receipt["labels_map"], "model", use + ".labels_proof")

    def _cohort(self, source_row, parameters, use):
        source = self.pair_doc(source_row, "source_manifest_path", "source_manifest_sha256", "configuration", use + ".source_manifest")
        symbols = parameters["source"]["symbols"]
        for index, row in enumerate(source_row["frames"]):
            tag = use + ".cohort." + str(index)
            receipt = self.pair_doc(row, "receipt_path", "receipt_sha256", "configuration", tag + ".receipt")
            for key in ("native_review_jpeg", "camera_jpeg", "v5_B_native_jpeg", "v5_B_arw", "v5_B_camera_jpeg"):
                self.pair(receipt, key + "_path", key + "_sha256", "image", tag + "." + key)
            self.pair(receipt, "v5_B_registration_certificate_path", "v5_B_registration_certificate_sha256", "rig_model", tag + ".B_registration")
            temporal = self.pair_doc(receipt, "temporal_geometry_receipt_path", "temporal_geometry_receipt_sha256", "rig_model", tag + ".temporal")
            if temporal["frames"][index].get("temporal_registration_pass") is not True:
                self.pair_doc(receipt, symbols["temporal_exception_path_key"], symbols["temporal_exception_sha_key"], "rig_model", tag + ".temporal_exception")
            self.pair_doc(receipt, "photo_sand_reference_receipt_path", "photo_sand_reference_receipt_sha256", "rig_model", tag + ".photo_acceptance")
            self.pair(receipt, "photo_sand_reference_mask_path", "photo_sand_reference_mask_sha256", "rig_model", tag + ".photo_mask")
            if source["run"] == symbols["prepared_source_id"] and index == 0:
                original = source["frame_sources"][0]
                self.add({"path": original["source_path"], "sha256": original.get("source_sha256") or original["source_raw_sha256"]}, "image", tag + ".prepared_source")

    def _source_setup(self, source_row, calibration_bound, aggregate_bound, parameters, use):
        identity = (source_row["run"], source_row["source_manifest_sha256"], calibration_bound["sha256"], aggregate_bound["sha256"])
        if identity in self.source_setups:
            return
        calibration = self.doc(calibration_bound, "calibration", use + ".calibration")
        self.doc(aggregate_bound, "configuration", use + ".photo_aggregate")
        self._cohort(source_row, parameters, use)
        calibration_path = Path(self.physical(calibration_bound)["path"])
        for source_id in (source_row["run"], "virtual_reference"):
            rows = [row for row in calibration["results"] if row.get("run") == source_id]
            if len(rows) != 1:
                raise ValueError("Exactly one original baseline binding required: " + source_id)
            row = rows[0]
            stem = calibration_path.parent / (source_id + "_baseline")
            self.add({"path": str(stem.with_suffix(".npz")), "sha256": row["baseline_sha256"]}, "calibration", use + ".baseline." + source_id)
            self.doc({"path": str(stem.with_suffix(".json")), "sha256": row["sidecar_sha256"]}, "calibration", use + ".baseline_sidecar." + source_id)
        for row in calibration.get("old_B_local_residual_fields", []):
            if row.get("run") == source_row["run"]:
                path = self.aliases.get(_key(row["field_path"]), {}).get("path", row["field_path"])
                if not Path(path).is_absolute():
                    path = str(calibration_path.parents[3] / path)
                self.add({"path": path, "sha256": row["field_sha256"]}, "calibration", use + ".old_B_residual")
                break
        self.source_setups.add(identity)

    def native_task(self, task, geometry, parameters, use):
        processor = task["processor"]
        if task["kind"] == "rig":
            manifest = self.doc(processor["native_manifest"], "configuration", use + ".native_manifest")
            binding = manifest["v5_photometry"]
            self.pair_doc(binding, "source_manifest", "source_manifest_sha256", "configuration", use + ".source_manifest")
            source = binding["processor"]
            calibration = {"path": source["calibration_manifest_path"], "sha256": source["calibration_manifest_sha256"]}
            aggregate = {"path": source["photo_source_aggregate_path"], "sha256": source["photo_source_aggregate_sha256"]}
            photos = self.doc(aggregate, "configuration", use + ".photo_aggregate")
            matches = [row for row in photos["rows"] if row["run"] == manifest["run"] and row["source_manifest_sha256"] == binding["source_manifest_sha256"]]
            if len(matches) != 1:
                raise ValueError("Exactly one original photo cohort is required")
            self._source_setup(matches[0], calibration, aggregate, parameters, use)
            parent = task.get("parent_frame_index")
            original = manifest["reference"] if task.get("parent_reference") else (manifest["frame_sources"][parent] if parent is not None else None)
            if original is not None:
                for key in ("v5_log_linear_rgb", "v5_log_linear_valid_mask"):
                    self.pair(original, key + "_path", key + "_sha256", "image", use + ".parent." + key)
                if task.get("parent_reference"):
                    self.pair(original, "v5_baseline_receipt_path", "v5_baseline_receipt_sha256", "configuration", use + ".parent.baseline_receipt")
            for key in ("camera_jpeg", "camera_arw", "native_review_jpeg"):
                self.pair(geometry, key + "_path", key + "_sha256", "image", use + "." + key)
            exact_b = (geometry["camera_arw_sha256"] == geometry["v5_B_arw_sha256"]
                       and geometry["camera_jpeg_sha256"] == geometry["v5_B_camera_jpeg_sha256"])
            if not exact_b:
                self.pair(geometry, "v5_B_registration_certificate_path", "v5_B_registration_certificate_sha256", "rig_model", use + ".B_registration")
                self.pair(geometry, "v5_B_camera_jpeg_path", "v5_B_camera_jpeg_sha256", "image", use + ".B_camera")
        elif task["kind"] == "source":
            source_row = self.doc(processor["source_row"], "configuration", use + ".source_row")
            self._source_setup(source_row, processor["calibration_manifest"], processor["photo_aggregate"], parameters, use)
            for key in ("camera_jpeg", "camera_arw", "review_jpeg"):
                self.add(task["arguments"][key], "image", use + "." + key)
        else:
            raise ValueError("Unsupported native processor kind: " + str(task["kind"]))

    def collect(self, config_descriptor, config):
        self.add(config_descriptor, "configuration", "run.configuration")
        if config.get("format") == "darsia-n1-run-v1":
            self.mass(config, config["frames"], standalone=True)
        elif config.get("format") == "darsia-native-phase-n1-run-v1":
            phase, native = self.phase(config)
            specs = [dict(frame["mass_spec"], case_id=frame["case_id"]) for frame in config["frames"]]
            self.mass(config["mass"]["template"], specs)
            for frame in config["frames"]:
                use = "frame." + frame["case_id"]
                task = frame["native_task"]
                geometry = self.doc(task["geometry"], "rig_model", use + ".native_geometry")
                rig = self.doc(frame["geometry"], "rig_model", use + ".rig_geometry")
                self.add(rig["frame_photo_top_keep"], "rig_model", use + ".photo_top")
                self.native_task(task, geometry, native, use + ".native")
                roles = frame["phase_roles"]
                if roles.get("native_row") is not None:
                    row = self.doc(roles["native_row"], "configuration", use + ".phase.native_row")
                    self.original_native_row(row, phase["parameters"]["inputs"], use + ".phase")
                    probability = self.doc(roles["probability_receipt"], "model", use + ".phase.probability_receipt")
                    self.add(probability["files"][phase["parameters"]["inputs"]["mask_file"]], "rig_model", use + ".phase.photo_mask")
                else:
                    self.add(roles["photo_sand_reference"], "rig_model", use + ".phase.photo_sand_reference")
                if roles["area"]["role"] == "measured_parent":
                    for key in ("historical", "measured"):
                        self.add(roles["area"][key], "rig_model", use + ".area." + key)
                self.add(roles["static"]["components"], "rig_model", use + ".static_components")
                if roles.get("retained_source_cache") is not None:
                    self.retained(roles["retained_source_cache"], phase["parameters"]["retained"], use + ".retained")
        elif config.get("format") == "darsia-native-phase-statistics-run-v1":
            from .staged_data import validate_phase1_run
            validate_phase1_run(config)
            phase, native = self.phase(config)
            specs = [dict(frame["mass_spec"], case_id=frame["case_id"]) for frame in config["frames"]]
            self.statistics(config["statistics"], specs)
            for frame in config["frames"]:
                use = "frame." + frame["case_id"]
                task = frame["native_task"]
                geometry = self.doc(task["geometry"], "rig_model", use + ".native_geometry")
                rig = self.doc(frame["geometry"], "rig_model", use + ".rig_geometry")
                self.add(rig["frame_photo_top_keep"], "rig_model", use + ".photo_top")
                self.native_task(task, geometry, native, use + ".native")
                roles = frame["phase_roles"]
                if roles.get("native_row") is not None:
                    row = self.doc(roles["native_row"], "configuration", use + ".phase.native_row")
                    self.original_native_row(row, phase["parameters"]["inputs"], use + ".phase")
                    probability = self.doc(roles["probability_receipt"], "model", use + ".phase.probability_receipt")
                    self.add(probability["files"][phase["parameters"]["inputs"]["mask_file"]], "rig_model", use + ".phase.photo_mask")
                else:
                    self.add(roles["photo_sand_reference"], "rig_model", use + ".phase.photo_sand_reference")
                if roles["area"]["role"] == "measured_parent":
                    for key in ("historical", "measured"):
                        self.add(roles["area"][key], "rig_model", use + ".area." + key)
                self.add(roles["static"]["components"], "rig_model", use + ".static_components")
                if roles.get("retained_source_cache") is not None:
                    self.retained(roles["retained_source_cache"], phase["parameters"]["retained"], use + ".retained")
            for case, request in config.get("photographs", {}).items():
                for field in ("native_geometry", "geometry_certificate", "current_geometry"):
                    self.doc(request[field], "rig_model", "frame." + case + ".photo." + field)
                self.add(request["saved_native_review"], "image", "frame." + case + ".photo.saved_native_review")
        else:
            raise ValueError("Unsupported role catalogue run format: " + str(config.get("format")))
        return {"format": CATALOG_FORMAT, "complete": True, "run": config["run"],
                "configuration": self.add(config_descriptor, "configuration", "run.configuration"),
                "inputs": [self.entries[key] for key in sorted(self.entries)],
                "payload_hashes_verified": False, "numerical_calls_started": False,
                "runtime_support": "J3 pending; no alias task may be claimed yet"}


def validate_catalog(value):
    if value.get("format") != CATALOG_FORMAT or value.get("complete") is not True:
        raise ValueError("Complete explicit role input catalogue required")
    validate_input_reference(value["configuration"])
    if not isinstance(value.get("inputs"), list) or not value["inputs"]:
        raise ValueError("Role input catalogue is empty")
    seen, consumers = set(), {}
    for row in value["inputs"]:
        validate_input_reference(row["reference"])
        identity = tuple((key,row["reference"][key]) for key in sorted(row["reference"]))
        if identity in seen:
            raise ValueError("Duplicate portable input catalogue reference")
        seen.add(identity)
        if not isinstance(row.get("uses"), list) or not row["uses"] or any(not isinstance(use, str) or not use for use in row["uses"]):
            raise ValueError("Input catalogue needs named original consumers")
        if len(row["uses"]) != len(set(row["uses"])):
            raise ValueError("Duplicate input catalogue consumer")
        for use in row["uses"]:
            if use in consumers and consumers[use] != row["reference"]:
                raise ValueError("Ambiguous input consumer maps to different references: " + use)
            consumers[use] = row["reference"]
    if value["configuration"] not in [row["reference"] for row in value["inputs"]]:
        raise ValueError("Catalogue omits its bound configuration input")
    return value

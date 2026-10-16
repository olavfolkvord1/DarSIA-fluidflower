"""Explicit per-run export/apply of already corrected workspace rasters.

No segmentation, registration, interpolation or rig calculation is implemented
here. Original raster archives and the selected run's original property table
are copied verbatim. Only four configuration leaves are applied explicitly.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tomllib
import uuid
import numpy as np

from darsia.fluidflower.geometry_facies import GEOMETRY_REVISION_FORMAT, GeometryRevisionStore, _load_array
from darsia.fluidflower.geometry_workspace_frame import adopted_content_digest, frame_kind, selected_fields
from darsia.fluidflower.project import utc_now
from darsia.fluidflower.series_geometry import (
    NATIVE_SETUP_EXPORT_FORMAT, NATIVE_LABEL_COLORS, native_color_index,
    native_series_link, _storage_path,
)
from darsia.fluidflower.setup_geometry_apply import ReadBinding, _exclusive
from darsia.fluidflower.setup_rig import _require_exact_grid

FORMAT = NATIVE_SETUP_EXPORT_FORMAT
LIMIT = 1024 * 1024
NAMES = {"labels": "native_labels.npz", "facies": "native_facies.npz"}
EXPORT_FILES = {*NAMES.values(), "facies_properties.csv", "native_setup.toml", "colored_labels.png"}


def _small(path):
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError(f"Corrected setup metadata exceeds 1 MiB: {path}")
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError(f"Corrected setup metadata exceeds 1 MiB: {path}")
    return data


def _descriptor(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during corrected setup verification: {path}")
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def _verified(row):
    actual = _descriptor(row["path"])
    if actual["sha256"] != row.get("sha256") or actual["size_bytes"] != row.get("size_bytes"):
        raise ValueError(f"Corrected setup source identity mismatch: {actual['path']}")
    return actual


def _same(first, second):
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def _identity(row):
    return {key: row[key] for key in ("path", "sha256", "size_bytes")}


def _replacements(destination):
    return {
        "labeling": {"labels": str(destination / NAMES["labels"]), "frame": "rig-corrected"},
        "facies": {"path": str(destination / NAMES["facies"]), "props": str(destination / "facies_properties.csv")},
    }


def _run_name(run):
    if not isinstance(run, str) or not run or run in (".", "..") or Path(run).name != run or any(c in run for c in "\\/:\x00"):
        raise ValueError("Corrected setup requires one explicit safe run ID.")
    return run


def _require_unedited_adoption(current, config):
    provenance = config.get("provenance") or {}
    if provenance.get("adopted_content_sha256") != adopted_content_digest(config):
        raise ValueError("This corrected revision was edited or lacks its adoption identity; export of edited/registered geometry is not qualified.")
    display = config["frame_binding"]["display_reference_run"]
    fields = selected_fields(config, display)
    for key, field in (("baseline_path", "shape_corrected_baseline"), ("labels_path", "labels"), ("facies_source_path", "facies"), ("depth_path", "depth")):
        actual = _descriptor(config[key])
        if actual["sha256"] != fields[field]["sha256"] or actual["size_bytes"] != fields[field]["size_bytes"]:
            raise ValueError("Saved corrected revision fields differ from the original adopted maps.")
    derived = (current.get("artifacts") or {}).get("facies_path")
    if not derived:
        raise ValueError("The genuine Store-derived facies artifact is missing.")
    if not np.array_equal(_load_array(Path(derived)), _load_array(Path(fields["facies"]["path"]))):
        raise ValueError("Store-derived facies differ from the original adopted facies; no corrected export was written.")


@dataclass(frozen=True)
class CorrectedSetupExport:
    revision_id: str
    run_id: str
    destination: Path
    toml_text: str
    current_json: Path
    current_bytes: bytes
    receipt: dict
    reused: bool = False


class CorrectedApplyError(RuntimeError):
    def __init__(self, message, receipt):
        self.receipt = Path(receipt)
        super().__init__(f"{message}. Application receipt and preserved originals: {receipt}")


def _canonical_location(package_root, config, revision, run):
    display = _run_name(config["frame_binding"]["display_reference_run"])
    runs = config["frame_binding"]["runs"]
    if run not in runs:
        raise ValueError("No corrected export binding exists for this selected run.")
    planned = {_run_name(key): f"runs/{key}/export.json" for key in sorted(runs) if key != display}
    root = package_root / "exports" / "native_setup" / revision
    return root, root if run == display else root / "runs" / run, display, planned


def _configuration_bundle(series, run, configuration):
    """Bind exact declared config paths, never a basename or path substring."""
    manifest_path = series / "series.toml"
    manifest_bytes = _small(manifest_path)
    manifest = tomllib.loads(manifest_bytes.decode("utf-8-sig"))
    table, paths = manifest.get("runs") or {}, manifest.get("paths") or {}
    ids = table.get("ids") or []
    if run not in ids or len(ids) != len(set(ids)):
        raise ValueError("The corrected export run is not uniquely registered in this series.")
    bundles = {}
    for key in ids:
        _run_name(key)
        values = (
            (table.get("config_files") or {}).get(key) or f"{paths.get('run_configs', 'config/runs')}/{key}.toml",
            (table.get("merged_files") or {}).get(key) or f"{paths.get('merged_configs', 'config/merged')}/{key}.toml",
        )
        own = []
        for value in values:
            item = _storage_path(value, series)
            if item is None or not item.resolve().is_relative_to(series.resolve()):
                raise ValueError("A corrected configuration bundle escapes the selected series.")
            if not any(_same(item, other) for other in own):
                own.append(str(item))
        bundles[key] = own
    selected = bundles[run]
    if not any(_same(configuration["path"], item) for item in selected):
        raise ValueError("The bound original configuration is not this run's declared configuration.")
    if any(_same(item, other) for key, rows in bundles.items() if key != run for item in selected for other in rows):
        raise ValueError("The selected corrected configuration bundle is ambiguous between runs.")
    common = _storage_path(paths.get("common_config", "config/common.toml"), series)
    if common is not None and any(_same(common, item) for rows in bundles.values() for item in rows):
        raise ValueError("The common configuration is ambiguous with a selected-run bundle.")
    common_binding = _descriptor(common) if common is not None and common.is_file() else None
    return dict(series_root=str(series), series_manifest=_descriptor(manifest_path),
                configuration_bundles=bundles, common_configuration=common_binding)


def _verify_export_files(receipt, destination):
    if set(receipt.get("files") or {}) != EXPORT_FILES:
        raise ValueError("The corrected export must contain exactly its five canonical named files.")
    for name, row in receipt["files"].items():
        if not _same(row["path"], destination / name):
            raise ValueError("A corrected export file belongs to another destination.")
        _verified(row)
    binding = receipt["binding"]
    for name, source in ((NAMES["labels"], binding["sources"]["labels"]),
                         (NAMES["facies"], binding["sources"]["facies"]),
                         ("facies_properties.csv", binding["properties"])):
        row = receipt["files"][name]
        if row["sha256"] != source["sha256"] or row["size_bytes"] != source["size_bytes"]:
            raise ValueError("The corrected export byte copies do not match their revision sources.")


def _apply_completion(destination, receipt):
    """Read one immutable successful application, refusing any partial claim."""
    complete = destination / "apply-complete.json"
    claim = destination / "apply-claim.json"
    if not complete.exists():
        if claim.exists():
            raise ValueError("An earlier corrected Apply is incomplete; all partial artifacts were preserved.")
        return None
    value = json.loads(_small(complete))
    if (value.get("format") != "darsia-corrected-setup-apply-complete-v1"
            or value.get("complete") is not True
            or value.get("run_id") != receipt["run_id"]
            or value.get("revision_id") != receipt["revision_id"]
            or value.get("export") != _descriptor(destination / "export.json")
            or value.get("input_binding") != receipt["binding"]):
        raise ValueError("An existing corrected Apply completion has a different identity; it was preserved.")
    claim_value = json.loads(_small(claim))
    if (claim_value.get("format") != "darsia-corrected-setup-apply-claim-v1"
            or claim_value.get("run_id") != receipt["run_id"]
            or claim_value.get("revision_id") != receipt["revision_id"]
            or claim_value.get("export") != value["export"]
            or claim_value.get("input_binding") != value["input_binding"]
            or not _same(Path(value["attempt_receipt"]["path"]).parent, claim_value.get("attempt_directory", ""))
            or Path(value["attempt_receipt"]["path"]).name != "receipt.json"
            or not Path(value["attempt_receipt"]["path"]).resolve().is_relative_to(destination.resolve())):
        raise ValueError("The corrected Apply completion does not match its preserved exclusive claim.")
    attempt = _verified(value["attempt_receipt"])
    record = json.loads(_small(attempt["path"]))
    if (record.get("state") != "complete" or record.get("run_id") != receipt["run_id"]
            or record.get("revision_id") != receipt["revision_id"]):
        raise ValueError("The preserved corrected Apply record is not complete for this run.")
    files = value.get("configurations")
    if not isinstance(files, list) or not files or len(files) != len(record.get("files") or []):
        raise ValueError("The corrected Apply completion has no exact configuration bundle.")
    for index, row in enumerate(files):
        old = record["files"][index]
        if (old.get("state") != "applied" or not _same(old["path"], row["after"]["path"])
                or old.get("before_sha256") != row["before"]["sha256"]
                or old.get("after_sha256") != row["after"]["sha256"]
                or not _same(row["before"]["path"], Path(attempt["path"]).parent / f"{index:02d}-before.toml")
                or not any(_same(row["after"]["path"], p) for p in receipt["binding"]["configuration_bundles"][receipt["run_id"]])):
            raise ValueError("The corrected Apply completion does not bind its actual before/after files.")
        _verified(row["before"])
        _verified(row["after"])
    return value


def _original_configuration_bytes(configuration, destination, receipt=None):
    """Keep the source SHA; only a proved prior Apply may supply its backup."""
    before = _small(configuration["path"])
    if hashlib.sha256(before).hexdigest() == configuration["sha256"] and len(before) == configuration["size_bytes"]:
        return before
    if receipt is not None:
        completed = _apply_completion(destination, receipt)
        for row in (completed or {}).get("configurations", []):
            if (_same(row["after"]["path"], configuration["path"])
                    and row["before"]["sha256"] == configuration["sha256"]
                    and row["before"]["size_bytes"] == configuration["size_bytes"]):
                return _small(row["before"]["path"])
    raise ValueError(f"Corrected setup source identity mismatch: {configuration['path']}")


def _verify_immutable_export_inputs(receipt):
    """Source data that Apply never modifies; configuration targets are separate."""
    data = receipt["binding"]
    for item in (*data["sources"].values(), data["properties"], data["configured_depth_source"], data["series_manifest"]):
        _verified(item)
    if data.get("common_configuration") is not None:
        _verified(data["common_configuration"])


def _verify_reusable_export(receipt, destination, *, revision, run, display, planned, current_sha, binding=None):
    if (receipt.get("schema") != 1 or receipt.get("format") != FORMAT
            or receipt.get("revision_id") != revision or receipt.get("run_id") != run
            or receipt.get("frame") != "rig-corrected" or receipt.get("reference_run") != display
            or receipt.get("current_sha256") != current_sha
            or (binding is not None and receipt.get("binding") != binding)
            or (run == display and receipt.get("runs") != planned)):
        raise ValueError("An existing corrected export has a different identity; it was preserved.")
    _verify_export_files(receipt, destination)
    _verify_immutable_export_inputs(receipt)
    data = receipt["binding"]
    _original_configuration_bytes(data["configuration"], destination, receipt)


def _verify_reference_export(root, *, revision, display, planned, current_sha, child=None):
    """One shared root receipt/files/linkage check for child export/Apply/Build."""
    path = root / "export.json"
    if not path.is_file():
        raise ValueError("Export the display-reference run before a canonical selected-run child.")
    root_receipt = json.loads(_small(path))
    _verify_reusable_export(root_receipt, root, revision=revision, run=display,
                            display=display, planned=planned, current_sha=current_sha)
    root_binding = _descriptor(path)
    if child is not None and child.get("root_export") != root_binding:
        raise ValueError("The selected child is bound to another reference export; it was preserved.")
    return root_binding


def require_corrected_export_for_build(paths):
    """Admit only this exact declared selected-run bundle and canonical export."""
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    labeling = _get_section_from_toml(paths, "labeling")
    facies = _get_section_from_toml(paths, "facies")
    labels_path = Path(labeling["labels"])
    destination = labels_path.parent
    receipt = json.loads(_small(destination / "export.json"))
    run, revision, display = _run_name(receipt.get("run_id")), receipt.get("revision_id"), _run_name(receipt.get("reference_run"))
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{32}", revision) is None:
        raise ValueError("Corrected Build requires the genuine revision UUID.")
    root = destination if run == display else destination.parent.parent
    expected = root if run == display else root / "runs" / run
    if (not _same(destination, expected) or root.name != revision
            or root.parent.name != "native_setup" or root.parent.parent.name != "exports"):
        raise ValueError("Corrected Build requires this run's canonical export location.")
    for name, value in ((NAMES["labels"], labeling["labels"]), (NAMES["facies"], facies["path"]), ("facies_properties.csv", facies["props"])):
        if not _same(value, destination / name):
            raise ValueError("Corrected Build inputs belong to different run exports.")
    current_path = root.parent.parent.parent / "overrides/geometry_facies/current.json"
    current_bytes = _small(current_path)
    current = json.loads(current_bytes)
    config = current["configuration"]
    actual_root, selected, recorded_display, planned = _canonical_location(root.parent.parent.parent, config, revision, run)
    if current.get("revision_id") != revision or not _same(selected, destination) or recorded_display != display:
        raise ValueError("Corrected Build export belongs to another current revision or selected run.")
    _require_unedited_adoption(current, config)
    _verify_reusable_export(receipt, destination, revision=revision, run=run, display=display,
                            planned=planned, current_sha=hashlib.sha256(current_bytes).hexdigest())
    binding = receipt["binding"]
    saved = config["frame_binding"]["runs"][run]
    if (binding["configuration"] != _identity(saved["configuration"])
            or binding["properties"] != _identity(saved["facies_properties"])
            or binding["configured_depth_source"] != _identity(saved["configured_depth_map"])
            or binding["replacements"] != _replacements(destination)
            or any(binding["sources"][name] != _identity(saved["fields"][name])
                   for name in ("shape_corrected_baseline", "labels", "facies", "depth"))):
        raise ValueError("Corrected Build export inputs differ from this run's current revision binding.")
    if run != display:
        _verify_reference_export(actual_root, revision=revision, display=display,
                                 planned=planned, current_sha=hashlib.sha256(current_bytes).hexdigest(), child=receipt)
    series = Path(binding["series_root"])
    declared = _configuration_bundle(series, run, binding["configuration"])
    if any(binding.get(key) != value for key, value in declared.items()):
        raise ValueError("Corrected Build configuration bundle changed since export.")
    actual_paths = [paths] if isinstance(paths, Path) else paths
    selected_paths = binding["configuration_bundles"][run]
    common = binding.get("common_configuration")
    allowed = selected_paths + ([common["path"]] if common is not None else [])
    if (not actual_paths or not any(_same(p, q) for p in actual_paths for q in selected_paths)
            or any(not any(_same(p, q) for q in allowed) for p in actual_paths)):
        raise ValueError("Corrected export belongs to a different or ambiguous selected-run configuration bundle.")
    # Actual selected config bytes must either be their original bound source,
    # or the exact output of the immutable successful Apply record.
    completion = _apply_completion(destination, receipt)
    if completion is None:
        raise ValueError("Apply this run's bound corrected export before Build run model.")
    for path in actual_paths:
        if common is not None and _same(path, common["path"]):
            _verified(common)
        elif not any(_same(path, row["after"]["path"]) for row in completion["configurations"]):
            raise ValueError("The selected run configuration is not an actual completed Apply target.")
    return receipt


def export_corrected_setup(project, *, reference, actor=""):
    """Explicit export only: full identity checks and byte copies, no Build."""
    if reference is None:
        raise ValueError("Choose the run whose corrected maps are to be exported.")
    run = _run_name(reference.run_id)
    current_path = GeometryRevisionStore(project).current_path
    before = _small(current_path)
    current = json.loads(before)
    if current.get("format") != GEOMETRY_REVISION_FORMAT or current.get("schema") != 1:
        raise ValueError("A saved original-schema workspace revision is required.")
    config = current["configuration"]
    if frame_kind(config) != "rig-corrected":
        raise ValueError("This export is only for an explicit corrected-frame revision.")
    revision = current.get("revision_id")
    if not isinstance(revision, str) or len(revision) != 32 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Corrected setup requires the original Store UUID.")
    fields = selected_fields(config, run)
    _require_unedited_adoption(current, config)
    export_root, destination, display, planned = _canonical_location(project.root, config, revision, run)
    current_sha = hashlib.sha256(before).hexdigest()
    existing = json.loads(_small(destination / "export.json")) if destination.exists() else None
    root_binding = None
    if run != display:
        # A child never manufactures or mutates the ordinary reference export.
        root_binding = _verify_reference_export(export_root, revision=revision, display=display,
                                                planned=planned, current_sha=current_sha, child=existing)
    row = config["frame_binding"]["runs"][run]
    configuration = _identity(row["configuration"])
    original = tomllib.loads(_original_configuration_bytes(configuration, destination, existing).decode("utf-8-sig"))
    depth_pointer = (original.get("depth") or {}).get("depth_map")
    series = Path(native_series_link(project)["series_root"])
    configured_depth_path = _storage_path(depth_pointer, series)
    if configured_depth_path is None or not _same(
        configured_depth_path.resolve(), Path(row["configured_depth_map"]["path"]).resolve()
    ):
        raise ValueError("The original configured depth pointer changed.")
    # This is intentionally the selected run's FULL original CSV, not the
    # revision store's reduced/canonical derived property table.
    props_path = _storage_path((original.get("facies") or {}).get("props"), series)
    if props_path is None:
        raise ValueError("The selected run has no original facies property table.")
    _small(props_path)
    properties_binding = row.get("facies_properties")
    if not isinstance(properties_binding, dict) or not _same(properties_binding.get("path", ""), props_path):
        raise ValueError("The saved revision must bind this run's original facies property table and SHA.")
    properties = _verified(properties_binding)
    configured_depth_source = _verified(row["configured_depth_map"])
    sources = {name: _verified(fields[name]) for name in ("shape_corrected_baseline", "labels", "facies", "depth")}
    import darsia
    baseline = darsia.imread(sources["shape_corrected_baseline"]["path"])
    for name in ("labels", "facies", "depth"):
        _require_exact_grid(Path(sources[name]["path"]), baseline, name)
    replacements = _replacements(destination)
    text = "\n".join(
        f"[{section}]\n" + "\n".join(f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in values.items())
        for section, values in replacements.items()
    ) + "\n"
    binding = dict(configuration=configuration, sources=sources, properties=properties,
                   configured_depth_source=configured_depth_source,
                   configured_depth_pointer=depth_pointer, replacements=replacements,
                   **_configuration_bundle(series, run, configuration))
    receipt_path = destination / "export.json"
    if existing is not None:
        receipt = existing
        _verify_reusable_export(receipt, destination, revision=revision, run=run,
                                display=display, planned=planned, current_sha=current_sha, binding=binding)
        if receipt.get("root_export") != root_binding:
            raise ValueError("The selected child is bound to another reference export; it was preserved.")
        if _small(current_path) != before:
            raise RuntimeError("The workspace revision changed during corrected export.")
        return CorrectedSetupExport(revision, run, destination, text, current_path, before, receipt, True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{run}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    # Failed staging is intentionally preserved. No existing export is removed.
    for name, output_name in NAMES.items():
        shutil.copyfile(sources[name]["path"], staging / output_name)
        copied = _descriptor(staging / output_name)
        if copied["sha256"] != sources[name]["sha256"]:
            raise RuntimeError("Corrected raster copy changed bytes.")
    shutil.copyfile(properties["path"], staging / "facies_properties.csv")
    if _descriptor(staging / "facies_properties.csv")["sha256"] != properties["sha256"]:
        raise RuntimeError("Original facies property table copy changed bytes.")
    _exclusive(staging / "native_setup.toml", text.encode("utf-8"))
    # Exactly the existing native exporter colour renderer; presentation only.
    from PIL import Image
    labels = _load_array(Path(sources["labels"]["path"]))
    color_index = native_color_index(labels)
    palette = np.asarray([(255, 255, 255)] + [rgb for _name, rgb in NATIVE_LABEL_COLORS], dtype=np.uint8)
    Image.fromarray(palette[color_index]).save(staging / "colored_labels.png")
    files = {}
    for name in sorted(EXPORT_FILES):
        files[name] = {**_descriptor(staging / name), "path": str(destination / name)}
    receipt = dict(schema=1, format=FORMAT, revision_id=revision, run_id=run, frame="rig-corrected",
                   reference_run=display, source_reference_run=config["frame_binding"]["source_reference_run"],
                   root_export=root_binding,
                   created_utc=utc_now(), created_by=str(actor), binding=binding, files=files,
                   current_sha256=hashlib.sha256(before).hexdigest(), lossless_rebuilt_gate="NOT RUN",
                   fields_are_original_byte_copies=True, actual_Build_run=False)
    if run == display:
        receipt.update(runs=planned, runs_are_planned_paths_only=True, child_completion_not_claimed=True)
    _exclusive(staging / "export.json", (json.dumps(receipt, ensure_ascii=False, indent=2)+"\n").encode())
    if _small(current_path) != before:
        raise RuntimeError("The workspace revision changed during corrected export.")
    selected_fields(config, run)
    _original_configuration_bytes(configuration, destination, existing)
    os.rename(staging, destination)
    return CorrectedSetupExport(revision, run, destination, text, current_path, before, receipt)


def corrected_config_bytes(before: bytes, replacements: dict) -> bytes:
    """Change only four physical lines/insertions; preserve all other bytes."""
    original = tomllib.loads(before.decode("utf-8-sig"))
    if set(replacements) != {"labeling", "facies"} or set(replacements["labeling"]) != {"labels", "frame"} or set(replacements["facies"]) != {"path", "props"} or replacements["labeling"]["frame"] != "rig-corrected":
        raise ValueError("Only the four explicit corrected setup leaves may be applied.")
    expected = deepcopy(original)
    bom = b"\xef\xbb\xbf" if before.startswith(b"\xef\xbb\xbf") else b""
    text = before[len(bom):].decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    for section, updates in replacements.items():
        expected.setdefault(section, {}).update(updates)
        headers = [(i, re.match(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$", line.rstrip("\r\n"))) for i, line in enumerate(lines)]
        starts = [i for i, m in headers if m and m.group(1).strip() == section]
        if len(starts) > 1:
            raise ValueError("Duplicate setup table.")
        if not starts:
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += newline
            lines += [newline, f"[{section}]{newline}"]
            start = len(lines)-1
        else:
            start = starts[0]
        end = next((i for i in range(start+1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        for key, value in updates.items():
            found = [i for i in range(start+1, end) if re.match(rf"^\s*{re.escape(key)}\s*=", lines[i])]
            replacement = f"{key} = {json.dumps(value, ensure_ascii=False)}{newline}"
            if found:
                if len(found) != 1:
                    raise ValueError("Duplicate setup key.")
                lines[found[0]] = replacement
            else:
                if end and not lines[end-1].endswith(("\n", "\r")):
                    lines[end-1] += newline
                lines.insert(end, replacement)
                end += 1
    after = bom + "".join(lines).encode("utf-8")
    if tomllib.loads(after.decode("utf-8-sig")) != expected:
        raise ValueError("Corrected setup text patch changed unrelated configuration values.")
    return after


def apply_corrected_selected_run(project, export: CorrectedSetupExport):
    """Explicit selected-run CAS only; no common config/crop/registration/Build."""
    root = Path(native_series_link(project)["series_root"])
    manifest_path = root / "series.toml"
    manifest_bytes = _small(manifest_path)
    manifest = tomllib.loads(manifest_bytes.decode("utf-8-sig"))
    if export.run_id not in (manifest.get("runs") or {}).get("ids", []):
        raise ValueError("The corrected export run is not in the selected series.")
    if _small(export.current_json) != export.current_bytes:
        raise RuntimeError("The current workspace revision changed; no setup was applied.")
    receipt_path = export.destination / "export.json"
    actual_receipt = json.loads(_small(receipt_path))
    if actual_receipt != export.receipt:
        raise RuntimeError("The corrected export receipt changed; no setup was applied.")
    current = json.loads(export.current_bytes)
    config = current["configuration"]
    _export_root, expected_destination, display, planned = _canonical_location(
        project.root, config, export.revision_id, export.run_id
    )
    if not _same(export.destination, expected_destination):
        raise ValueError("The corrected export belongs to another selected-run destination.")
    _verify_reusable_export(actual_receipt, export.destination, revision=export.revision_id,
                            run=export.run_id, display=display, planned=planned,
                            current_sha=hashlib.sha256(export.current_bytes).hexdigest())
    if export.run_id != display:
        _verify_reference_export(_export_root, revision=export.revision_id, display=display,
                                 planned=planned, current_sha=hashlib.sha256(export.current_bytes).hexdigest(), child=actual_receipt)
    saved = json.loads(export.current_bytes)["configuration"]["frame_binding"]["runs"][export.run_id]
    binding = export.receipt["binding"]
    if binding["configuration"] != _identity(saved["configuration"]) or binding["properties"] != _identity(saved["facies_properties"]) or binding["configured_depth_source"] != _identity(saved["configured_depth_map"]) or binding["replacements"] != _replacements(export.destination):
        raise ValueError("The corrected export input identities or target paths are not bound to this revision.")
    if any(binding["sources"][name] != _identity(saved["fields"][name]) for name in ("shape_corrected_baseline", "labels", "facies", "depth")):
        raise ValueError("The corrected export field identities differ from this revision.")
    if set(export.receipt["files"]) != EXPORT_FILES:
        raise ValueError("The corrected export must contain exactly its five canonical named files.")
    for name, source in ((NAMES["labels"], saved["fields"]["labels"]), (NAMES["facies"], saved["fields"]["facies"]), ("facies_properties.csv", saved["facies_properties"])):
        row = export.receipt["files"][name]
        if not _same(row["path"], export.destination / name) or row["sha256"] != source["sha256"] or row["size_bytes"] != source["size_bytes"]:
            raise ValueError("The corrected export byte copies do not match their revision sources.")
    for row in export.receipt["files"].values():
        _verified(row)
    _verified(export.receipt["binding"]["configured_depth_source"])
    configuration = export.receipt["binding"]["configuration"]
    _original_configuration_bytes(configuration, export.destination, export.receipt)
    for row in (*export.receipt["binding"]["sources"].values(), binding["properties"], binding["series_manifest"]):
        _verified(row)
    if binding.get("common_configuration") is not None:
        _verified(binding["common_configuration"])
    expected_bundle = _configuration_bundle(root, export.run_id, configuration)
    if any(binding.get(key) != value for key, value in expected_bundle.items()):
        raise ValueError("The selected run's declared configuration bundle changed.")
    completed = _apply_completion(export.destination, export.receipt)
    if completed is not None:
        # Pure revalidation/reuse: no new UUID, timestamp, write or replacement.
        return Path(completed["attempt_receipt"]["path"])
    targets = [Path(configuration["path"])]
    paths = manifest.get("paths") or {}
    table = manifest.get("runs") or {}
    partial = _storage_path((table.get("config_files") or {}).get(export.run_id) or f"{paths.get('run_configs','config/runs')}/{export.run_id}.toml", root)
    merged = _storage_path((table.get("merged_files") or {}).get(export.run_id) or f"{paths.get('merged_configs','config/merged')}/{export.run_id}.toml", root)
    declared = [p for p in (partial, merged) if p is not None]
    if not any(_same(configuration["path"], p) for p in declared):
        raise ValueError("The bound original configuration is not this run's declared configuration.")
    if partial is not None and not any(_same(partial, p) for p in targets):
        targets.append(partial)
    if merged is not None and merged.is_file() and not any(_same(merged, p) for p in targets):
        targets.append(merged)
    for path in targets:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Corrected setup target is outside the selected series.")
    bindings = [ReadBinding(manifest_path, manifest_bytes), ReadBinding(export.current_json, export.current_bytes), ReadBinding(receipt_path, _small(receipt_path))]
    changes = []
    for path in targets:
        before = _small(path)
        bindings.append(ReadBinding(path, before))
        changes.append((path, before, corrected_config_bytes(before, export.receipt["binding"]["replacements"])))
    for item in bindings:
        item.require_unchanged()
    attempt = export.destination / ("apply-" + uuid.uuid4().hex)
    attempt.mkdir()
    _exclusive(export.destination / "apply-claim.json", (json.dumps(dict(
        format="darsia-corrected-setup-apply-claim-v1", run_id=export.run_id,
        revision_id=export.revision_id, attempt_directory=str(attempt),
        export=_descriptor(receipt_path), input_binding=binding,
    ), ensure_ascii=False, indent=2)+"\n").encode())
    files, staged = [], []
    receipt = dict(format="darsia-corrected-setup-apply-v1", run_id=export.run_id,
                   revision_id=export.revision_id, created_utc=utc_now(), state="prepared", files=files,
                   crop_changed=False, depth_pointer_changed=False, registration_run=False, Build_run=False)
    try:
        for i, (path, before, after) in enumerate(changes):
            _exclusive(attempt / f"{i:02d}-before.toml", before)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            _exclusive(temporary, after)
            staged.append((path, temporary))
            files.append(dict(path=str(path), before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(), state="staged"))
        applied = set()
        for i, (path, temporary) in enumerate(staged):
            for item in bindings:
                if str(item.path) not in applied:
                    item.require_unchanged()
            os.replace(temporary, path)
            applied.add(str(path))
            files[i]["state"] = "applied"
        _verify_export_files(export.receipt, export.destination)
        _verify_immutable_export_inputs(export.receipt)
        if export.run_id != display:
            _verify_reference_export(_export_root, revision=export.revision_id, display=display,
                                     planned=planned, current_sha=hashlib.sha256(export.current_bytes).hexdigest(), child=actual_receipt)
        for item in bindings:
            if not any(_same(item.path, path) for path, _before, _after in changes):
                item.require_unchanged()
        # Recheck the entire planned result, including targets already replaced.
        # An observed external edit is a failure, never the new expected output.
        for index, (path, before, after) in enumerate(changes):
            if _small(path) != after:
                raise RuntimeError("Corrected Apply target differs from its expected-after bytes: " + str(path))
            if _small(attempt / f"{index:02d}-before.toml") != before:
                raise RuntimeError("A preserved corrected Apply before-copy changed.")
        receipt["state"] = "complete"
    except Exception as error:
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        _exclusive(attempt / "receipt.json", (json.dumps(receipt, indent=2)+"\n").encode())
        raise CorrectedApplyError(str(error), attempt / "receipt.json") from error
    _exclusive(attempt / "receipt.json", (json.dumps(receipt, indent=2)+"\n").encode())
    complete = dict(format="darsia-corrected-setup-apply-complete-v1", complete=True,
                    run_id=export.run_id, revision_id=export.revision_id,
                    export=_descriptor(receipt_path), input_binding=binding,
                    attempt_receipt=_descriptor(attempt / "receipt.json"),
                    configurations=[dict(before=dict(path=str(attempt / f"{index:02d}-before.toml"),
                                                     sha256=hashlib.sha256(before).hexdigest(), size_bytes=len(before)),
                                         after=dict(path=str(path), sha256=hashlib.sha256(after).hexdigest(), size_bytes=len(after)))
                                    for index, (path, before, after) in enumerate(changes)])
    _exclusive(export.destination / "apply-complete.json", (json.dumps(complete, ensure_ascii=False, indent=2)+"\n").encode())
    return attempt / "receipt.json"

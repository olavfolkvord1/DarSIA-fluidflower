"""Explicit lossless imported raster sources for geometry workspace revisions.

This is a workspace representation, not a map correction. The ordinary painted
and raw region-derived paths do not call these functions. An imported revision
keeps both categorical rasters and the full original CSV independently; edits
require a deliberately created canonical draft and retain their parent identity.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

IMPORTED = "imported-rasters"
CANONICAL_EDIT = "canonical-from-import"
FORMAT = "darsia-geometry-imported-rasters-v1"
KEYS = {"labels", "facies", "facies_properties"}
PATH_KEYS = {"labels": "labels_path", "facies": "facies_source_path",
             "facies_properties": "facies_properties_source_path"}
SILENT_EDIT = (
    "Imported rasters are independently authoritative. Save cannot silently "
    "derive facies from labels or edit their sources/properties. Create an "
    "explicit canonical edit draft first; it will report the difference to "
    "the imported parent. Registered geometry still requires Olav's decision."
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def content_digest(config):
    """Permit only the Store's existing depth materialization metadata changes."""
    value = deepcopy(dict(config))
    for key in ("provenance", "depth_path", "depth_artifact_path", "depth_source_reference"):
        value.pop(key, None)
    value.get("imported_rasters", {}).pop("configuration_sha256", None)
    return _digest(value)


def descriptor(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("An imported geometry source changed while being read: " + str(path))
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def _verify(binding, path=None):
    if (not isinstance(binding, dict) or not Path(str(binding.get("path", ""))).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))) is None
            or type(binding.get("size_bytes")) is not int or binding["size_bytes"] < 0):
        raise ValueError("Imported geometry requires explicit path/SHA-256/size bindings.")
    actual = descriptor(path or binding["path"])
    if any(actual[key] != binding[key] for key in ("sha256", "size_bytes")):
        raise ValueError("An imported geometry source changed: " + str(path or binding["path"]))
    return actual


def bind_imported_rasters(config, *, sources, diagnosis):
    """Opt in explicitly during preparation; this does not read or write maps."""
    if set(sources) != KEYS or not isinstance(diagnosis, dict):
        raise ValueError("Bind both original rasters, the FULL CSV, and the diagnosis.")
    candidate = deepcopy(dict(config))
    candidate["source_mode"] = IMPORTED
    candidate["imported_rasters"] = {
        "format": FORMAT, "sources": deepcopy(sources), "diagnosis": deepcopy(diagnosis),
        "independent_facies_raster": True, "full_properties_table": True,
        "edit_policy": "explicit canonical draft only; no silent recanonicalisation",
    }
    candidate["imported_rasters"]["configuration_sha256"] = content_digest(candidate)
    return candidate


def _full_csv(path):
    # Preserve arbitrary original columns and all IDs. Never regenerate a CSV
    # from only those facies which happen to be a region's modal assignment.
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        names = {name.strip().casefold(): name for name in columns}
        identity = names.get("id") or names.get("facies_id")
        if identity is None:
            raise ValueError("The FULL original facies CSV has no id column.")
        rows = list(reader)
    ids = []
    for row in rows:
        text = str(row.get(identity, "")).strip()
        try:
            value = float(text)
            if not np.isfinite(value) or not value.is_integer():
                raise ValueError
            ids.append(int(value))
        except (ValueError, OverflowError):
            raise ValueError("The FULL original facies CSV has a noninteger ID.") from None
    if len(ids) != len(set(ids)):
        raise ValueError("The FULL original facies CSV has duplicate IDs.")
    return columns, rows, ids


def _integer_map(path):
    from .geometry_facies import _load_array
    value = np.asarray(_load_array(Path(path)))
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("Imported labels and facies must be two-dimensional integer rasters.")
    return value


def imported_payload(config):
    """Verify immutable independently authoritative data, including full CSV IDs."""
    contract = config.get("imported_rasters") or {}
    if (config.get("source_mode") != IMPORTED or contract.get("format") != FORMAT
            or contract.get("independent_facies_raster") is not True
            or contract.get("full_properties_table") is not True
            or set(contract.get("sources") or {}) != KEYS
            or contract.get("configuration_sha256") != content_digest(config)):
        raise ValueError(SILENT_EDIT)
    sources = contract["sources"]
    actual = {name: _verify(sources[name], config.get(PATH_KEYS[name])) for name in KEYS}
    _verify(contract["diagnosis"])
    if any(Path(actual[name]["path"]).suffix.casefold() != ".npz"
           or Path(sources[name]["path"]).suffix.casefold() != ".npz"
           for name in ("labels", "facies")):
        raise ValueError("Imported raster authority requires original NPZ archives; PNG/TIFF bytes cannot be named NPZ.")
    if Path(actual["facies_properties"]["path"]).suffix.casefold() != ".csv":
        raise ValueError("Imported raster mode requires the full original CSV, without conversion.")
    labels = _integer_map(actual["labels"]["path"])
    facies = _integer_map(actual["facies"]["path"])
    if labels.shape != facies.shape:
        raise ValueError("Imported labels and facies grids differ; no resampling is allowed.")
    columns, rows, ids = _full_csv(actual["facies_properties"]["path"])
    missing = sorted(set(map(int, np.unique(facies))) - set(ids))
    if missing:
        raise ValueError("The FULL original CSV omits raster facies IDs: " + repr(missing))
    from .geometry_facies import _effective_run_geometry_choice
    run_geometry = config.get("run_geometry") or {}
    if any(_effective_run_geometry_choice(row) == "run_override" for row in run_geometry.get("runs") or []):
        raise ValueError("Imported raster revisions cannot silently add edited run overrides.")
    return dict(labels=labels, facies=facies, columns=columns, rows=rows, ids=ids,
                actual=actual, diagnosis=deepcopy(contract["diagnosis"]))


def difference_to_canonical(config, facies):
    from .geometry_facies import _canonical_facies, _layer_boundary_mask
    canonical, _ = _canonical_facies(config)
    if canonical.shape != facies.shape:
        raise ValueError("A canonical comparison changed the imported grid.")
    changed = canonical != facies
    labels = _integer_map(config["labels_path"])
    boundary = _layer_boundary_mask(labels, set(map(int, np.unique(labels))))
    count = int(np.count_nonzero(changed))
    total = int(changed.size)
    return dict(pixel_count=total, different_pixels=count,
                different_fraction=count / total,
                different_pixels_on_label_boundary=int(np.count_nonzero(changed & boundary)),
                difference_rule="array values; no tolerance, splitting or interpolation",
                notice=("imported rasters; facies raster independent of labels; "
                        f"{count:,} pixels ({100 * count / total:.2f} %) differ from a label-derived facies map"))


def validate_source_mode(config, report):
    """Keep ordinary validation; don't show a label-derived pore-volume as truth."""
    if config.get("source_mode") not in {IMPORTED, CANONICAL_EDIT}:
        return report
    value = deepcopy(report)
    try:
        payload = (imported_payload(config) if config["source_mode"] == IMPORTED
                   else canonical_parent_payload(config))
        if config["source_mode"] == IMPORTED:
            detail = difference_to_canonical(config, payload["facies"])
            value["summary"].update(source_mode=IMPORTED, source_mode_notice=detail["notice"],
                                    imported_raster_difference=detail, facies_count=len(payload["ids"]),
                                    full_properties_ids=payload["ids"],
                                    estimated_pore_volume_m3=None, active_area_fraction=None,
                                    physical_summary_note="Independent raster properties; use the actual rebuilt-rig gate.")
        else:
            value["summary"].update(source_mode=CANONICAL_EDIT,
                                    source_mode_notice="Explicit canonical edit draft; differences to imported parent will be recorded.")
    except (ValueError, OSError, KeyError) as error:
        value["issues"].append(dict(severity="error", code="invalid_geometry_source_mode", message=str(error)))
        value["summary"]["error_count"] += 1
    return value


def prepare_canonical_edit(project, *, actor, reason):
    """Explicit opt-in only. No save, correction, labels, or registration occurs."""
    from .geometry_facies import GeometryRevisionStore
    if not str(actor).strip() or not str(reason).strip():
        raise ValueError("A canonical edit draft needs an explicit actor and reason.")
    store = GeometryRevisionStore(project)
    revision = store.current()
    config = deepcopy(revision.get("configuration") or {})
    imported_payload(config)
    revision_path = store.artifact_root / str(revision.get("revision_id")) / "revision.json"
    record = _verify(descriptor(revision_path))
    if json.loads(revision_path.read_bytes()) != revision:
        raise ValueError("The imported parent revision record differs from current.")
    parent = dict(revision=record, revision_id=revision["revision_id"],
                  imported_configuration=deepcopy(config), actor=str(actor).strip(), reason=str(reason).strip())
    config["source_mode"] = CANONICAL_EDIT
    config.pop("imported_rasters")
    config["canonical_import_parent"] = parent
    config["provenance"] = {"origin": "explicit canonical edit draft from imported rasters",
                            "parent_revision_id": revision["revision_id"],
                            "registered_geometry_authorized": False}
    return config


def canonical_parent_payload(config, *, run_id=None):
    parent = config.get("canonical_import_parent") or {}
    record = _verify(parent.get("revision"))
    revision = json.loads(Path(record["path"]).read_bytes())
    old = parent.get("imported_configuration") or {}
    if (config.get("source_mode") != CANONICAL_EDIT or not parent.get("actor") or not parent.get("reason")
            or revision.get("revision_id") != parent.get("revision_id")
            or revision.get("configuration") != old):
        raise ValueError("A canonical edit lacks its actual imported parent and explicit opt-in.")
    payload = imported_payload(old)
    if (str(config.get("facies_properties_source_path")) != str(old.get("facies_properties_source_path"))
            or config.get("grouping_mode") != "shared_facies"):
        raise ValueError("A canonical edit must keep the FULL original CSV and facies IDs.")
    labels = _integer_map(config["labels_path"])
    if labels.shape != payload["labels"].shape:
        raise ValueError("A canonical edit cannot silently change the imported grid.")
    _validate_registered_runs(config, payload, parent, run_id=run_id)
    from .geometry_facies import _load_properties
    props = _load_properties(Path(payload["actual"]["facies_properties"]["path"]))
    for row in config.get("regions") or []:
        actual = props.get(int(row["facies_id"]))
        if actual is None or any(row.get(key) != actual[key] for key in ("porosity", "permeability")):
            raise ValueError("A canonical edit cannot silently change the original physical CSV properties.")
    return payload



def integer_label_translation(labels, *, row_shift, column_shift, fill_label):
    """Translate categorical pixel values; explicit fill, no wrap/interpolation.

    This is a geometry-representation operation only. It never selects a shift,
    a fill category, an epoch, or a scientific acceptance threshold.
    """
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("A label translation needs a two-dimensional integer raster.")
    if any(type(value) is not int for value in (row_shift, column_shift, fill_label)):
        raise ValueError("Label translation offsets and fill must be explicit integers.")
    if fill_label not in set(map(int, np.unique(values))):
        raise ValueError("The explicit fill must be an existing parent label category.")
    height, width = values.shape
    if abs(row_shift) >= height or abs(column_shift) >= width:
        raise ValueError("A label translation must retain overlapping parent pixels.")
    source_r = slice(max(0, -row_shift), min(height, height - row_shift))
    source_c = slice(max(0, -column_shift), min(width, width - column_shift))
    target_r = slice(max(0, row_shift), min(height, height + row_shift))
    target_c = slice(max(0, column_shift), min(width, width + column_shift))
    translated = np.full_like(values, fill_label)
    translated[target_r, target_c] = values[source_r, source_c]
    retained = np.zeros(values.shape, dtype=bool)
    retained[source_r, source_c] = True
    ids, counts = np.unique(values[~retained], return_counts=True)
    metrics = dict(row_shift=row_shift, column_shift=column_shift, fill_label=fill_label,
                   source_shape=list(values.shape), source_dtype=values.dtype.str,
                   clipped_source_pixel_count=int(np.count_nonzero(~retained)),
                   clipped_source_label_counts={str(int(k)): int(v) for k, v in zip(ids, counts)},
                   introduced_fill_pixel_count=int(np.count_nonzero(~retained)),
                   interpolation=False, wrapping=False)
    return translated, metrics


def integer_label_edge_replication(labels, *, row_shift, column_shift, pixel_size_mm):
    """Explicit Q43 categorical shift and nearest covered-edge replication.

    Filled strips are flagged by exact half-open rectangles in revision metrics.
    These flags mean extrapolated geometry, not camera or physical-domain validity.
    Existing values, including 255, retain their categorical meaning.
    """
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("A label translation needs a two-dimensional integer raster.")
    if any(type(value) is not int for value in (row_shift, column_shift)):
        raise ValueError("Label translation offsets must be explicit integers.")
    registration_pixel_shift(dx_mm=0., dy_mm=0., pixel_size_mm=pixel_size_mm)
    height, width = values.shape
    if abs(row_shift) >= height or abs(column_shift) >= width:
        raise ValueError("A label translation must retain overlapping parent pixels.")
    rows = np.clip(np.arange(height) - row_shift, 0, height - 1)
    columns = np.clip(np.arange(width) - column_shift, 0, width - 1)
    translated = values[rows[:, None], columns[None, :]]
    source_r = slice(max(0, -row_shift), min(height, height - row_shift))
    source_c = slice(max(0, -column_shift), min(width, width - column_shift))
    retained = np.zeros(values.shape, dtype=bool)
    retained[source_r, source_c] = True
    ids, counts = np.unique(values[~retained], return_counts=True)
    r0, r1 = max(0, row_shift), min(height, height + row_shift)
    c0, c1 = max(0, column_shift), min(width, width + column_shift)
    rectangles = {"top": [0, r0, 0, width], "bottom": [r1, height, 0, width],
                  "left": [0, height, 0, c0], "right": [0, height, c1, width]}
    pixel_area_mm2 = float(pixel_size_mm) ** 2
    edges = {}
    for edge, bounds in rectangles.items():
        ar0, ar1, ac0, ac1 = bounds
        count = (ar1 - ar0) * (ac1 - ac0)
        edges[edge] = dict(bounds_rc=bounds, pixel_count=count,
                           area_mm2=count * pixel_area_mm2)
    union_count = height * width - (r1 - r0) * (c1 - c0)
    geometry_filled = dict(format="integer-translation-extrapolated-strips-v1",
        shape=list(values.shape), bounds="half-open [row_start,row_stop,column_start,column_stop]",
        pixel_count=union_count, area_mm2=union_count * pixel_area_mm2,
        pixel_size_mm=float(pixel_size_mm), edges=edges,
        corner_overlap_pixel_count=abs(row_shift) * abs(column_shift),
        edge_counts_include_corner_overlap=True,
        flag_rule="geometry_filled is true in the union of the four edge rectangles",
        meaning="extrapolated geometry; not camera or physical-domain validity")
    metrics = dict(row_shift=row_shift, column_shift=column_shift,
        fill_mode="edge-replication", source_shape=list(values.shape),
        source_dtype=values.dtype.str,
        clipped_source_pixel_count=int(np.count_nonzero(~retained)),
        clipped_source_label_counts={str(int(k)): int(v) for k,v in zip(ids,counts)},
        introduced_fill_pixel_count=union_count, geometry_filled=geometry_filled,
        interpolation=False, wrapping=False)
    return translated, metrics


def registration_pixel_shift(*, dx_mm, dy_mm, pixel_size_mm):
    """Explicit metric-to-integer display-grid conversion, with residuals."""
    values = (dx_mm, dy_mm, pixel_size_mm)
    if (any(not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_)) for value in values)
            or not all(np.isfinite(value) for value in values) or pixel_size_mm <= 0):
        raise ValueError("Registration displacement and explicit pixel size must be finite.")
    row, column = int(round(dy_mm / pixel_size_mm)), int(round(dx_mm / pixel_size_mm))
    return dict(row_shift=row, column_shift=column, pixel_size_mm=float(pixel_size_mm),
                candidate_dx_mm=float(dx_mm), candidate_dy_mm=float(dy_mm),
                represented_dx_mm=column * pixel_size_mm, represented_dy_mm=row * pixel_size_mm,
                residual_dx_mm=dx_mm - column * pixel_size_mm,
                residual_dy_mm=dy_mm - row * pixel_size_mm,
                sign_convention="positive dx right; positive dy down", interpolation=False)


def _validate_registered_runs(config, payload, parent, *, run_id=None):
    """Admit explicit original-Store run rows, not unbound arbitrary overrides."""
    from .geometry_facies import _darsia_metadata, _effective_run_geometry_choice
    rows = (config.get("run_geometry") or {}).get("runs") or []
    overrides = [row for row in rows if _effective_run_geometry_choice(row) == "run_override"]
    if overrides:
        old = parent["imported_configuration"]
        if any(config.get(key) != old.get(key) for key in
               ("frame", "width_m", "height_m", "frame_binding", "baseline_path")):
            raise ValueError("Registered runs cannot change their imported frame or photograph bindings.")
    if run_id is not None:
        key = str(run_id).strip().casefold()
        rows = [row for row in rows if str(row.get("run_id") or "").strip().casefold() == key]
        if len(rows) != 1 or _effective_run_geometry_choice(rows[0]) != "run_override":
            raise ValueError("The selected run has no explicit registered override.")
    seen = set()
    for row in rows:
        if _effective_run_geometry_choice(row) != "run_override":
            continue
        run = str(row.get("run_id") or "").strip().casefold()
        if not run or run in seen or not all(c.isalnum() or c in "-_" for c in run):
            raise ValueError("Registered run identifiers must be explicit and unique.")
        seen.add(run)
        record = row.get("registration") or {}
        if (record.get("parent_revision_id") != parent["revision_id"]
                or record.get("parent_labels") != payload["actual"]["labels"]
                or record.get("status") not in {"registered", "provisional - Olav to review"}):
            raise ValueError("A registered run needs its actual imported parent and explicit status.")
        _verify(record.get("authority"))
        _verify(record.get("measurement"))
        actual = _verify(record.get("labels"), row.get("labels_path"))
        if (Path(actual["path"]).suffix.casefold() != ".npz"
                or str(row.get("facies_path") or "").strip()):
            raise ValueError("Registered run facies must be derived canonically from its NPZ labels.")
        target = _integer_map(actual["path"])
        metric = record.get("candidate") or {}
        converted = registration_pixel_shift(dx_mm=metric.get("candidate_dx_mm"),
            dy_mm=metric.get("candidate_dy_mm"), pixel_size_mm=metric.get("pixel_size_mm"))
        if (metric != converted or any(record.get(key) != converted[key]
                                      for key in ("row_shift", "column_shift"))):
            raise ValueError("Registered candidate metadata disagrees with its integer pixel shift.")
        fill_mode = record.get("fill_mode")
        if fill_mode == "edge-replication":
            if "fill_label" in record:
                raise ValueError("Edge replication cannot substitute a constant fill label.")
            expected, metrics = integer_label_edge_replication(payload["labels"],
                row_shift=record.get("row_shift"), column_shift=record.get("column_shift"),
                pixel_size_mm=converted["pixel_size_mm"])
        elif fill_mode is None:
            expected, metrics = integer_label_translation(payload["labels"], row_shift=record.get("row_shift"),
                column_shift=record.get("column_shift"), fill_label=record.get("fill_label"))
        else:
            raise ValueError("An unknown registered geometry fill mode is forbidden.")
        if record.get("metrics") != metrics:
            raise ValueError("Registered translation metrics disagree with the actual parent pixels.")
        if target.dtype != expected.dtype or not np.array_equal(target, expected):
            raise ValueError("Registered labels differ from the explicit integer parent translation.")
        if _digest(_darsia_metadata(Path(actual["path"]))) != _digest(
                _darsia_metadata(Path(payload["actual"]["labels"]["path"]))):
            raise ValueError("Registered labels changed the imported grid metadata.")
        source = _verify(record.get("source_image"), row.get("source_image_path"))
        binding = config.get("frame_binding") or {}
        if binding:
            fields = ((binding.get("runs") or {}).get(run) or {}).get("fields") or {}
            declared = fields.get("shape_corrected_baseline") or {}
            if any(source.get(key) != declared.get(key) for key in ("path", "sha256", "size_bytes")):
                raise ValueError("Registered run photograph is not its existing bound corrected baseline.")


def registered_run_sources(project_root, current, run_id):
    """Read one genuine saved Store run; no new map, cache, or fallback.

    Store admission still validates all run rows by default. This explicit
    selected-run read verifies only the selected translation and its sources.
    """
    from .geometry_facies import GEOMETRY_REVISION_FORMAT, _canonical_facies, _darsia_metadata, resolve_run_geometry_artifacts
    root = Path(project_root).resolve()
    current_path = root / "overrides" / "geometry_facies" / "current.json"
    if current_path.stat().st_size > 1024 * 1024:
        raise ValueError("Registered current metadata exceeds 1 MiB.")
    current_file = descriptor(current_path)
    saved = json.loads(current_path.read_bytes())
    revision_id = current.get("revision_id")
    if (saved != current or current.get("schema") != 1
            or current.get("format") != GEOMETRY_REVISION_FORMAT
            or re.fullmatch(r"[0-9a-f]{32}", str(revision_id)) is None):
        raise ValueError("Registered geometry needs its genuine current Store UUID and record.")
    revision_path = root / "results" / "geometry" / revision_id / "revision.json"
    if revision_path.stat().st_size > 1024 * 1024:
        raise ValueError("Registered revision metadata exceeds 1 MiB.")
    revision_file = descriptor(revision_path)
    if json.loads(revision_path.read_bytes()) != current:
        raise ValueError("Registered current differs from the actual immutable Store revision.")
    config = current["configuration"]
    payload = canonical_parent_payload(config, run_id=run_id)
    run = str(run_id).strip().casefold()
    row = next(row for row in config["run_geometry"]["runs"]
               if str(row.get("run_id") or "").strip().casefold() == run)
    artifacts = resolve_run_geometry_artifacts(current, run)
    if artifacts.get("scope") != "run_override" or artifacts.get("run_id") != run:
        raise ValueError("No genuine Store run artifact exists; series fallback is forbidden.")
    destination = root / "results" / "geometry" / revision_id / "runs" / run
    files = {}
    for name, filename in (("labels", "labels.npz"), ("facies", "facies.npz"),
                           ("facies_properties", "facies_properties.csv")):
        path = Path(str(artifacts.get(name + "_path") or "")).resolve()
        if path != destination / filename:
            raise ValueError("Registered artifacts belong to another Store UUID or run.")
        files[name] = descriptor(path)
    labels = _integer_map(files["labels"]["path"])
    expected = _integer_map(row["registration"]["labels"]["path"])
    if labels.dtype != np.dtype(np.int32) or not np.array_equal(labels, expected):
        raise ValueError("Stored registered labels differ numerically from the admitted translation.")
    if _digest(_darsia_metadata(Path(files["labels"]["path"]))) != _digest(
            _darsia_metadata(Path(row["registration"]["labels"]["path"]))):
        raise ValueError("Stored registered labels changed their grid metadata.")
    run_config = deepcopy(config)
    run_config["labels_path"] = files["labels"]["path"]
    expected_facies, _ = _canonical_facies(run_config)
    facies = _integer_map(files["facies"]["path"])
    if facies.dtype != np.dtype(np.int32) or not np.array_equal(facies, expected_facies):
        raise ValueError("Stored registered facies are not the exact canonical label mapping.")
    if _digest(_darsia_metadata(Path(files["facies"]["path"]))) != _digest(
            _darsia_metadata(Path(files["labels"]["path"]))):
        raise ValueError("Stored registered facies changed their grid metadata.")
    if Path(files["facies_properties"]["path"]).read_bytes() != Path(payload["actual"]["facies_properties"]["path"]).read_bytes():
        raise ValueError("Stored registered run lost the FULL original properties CSV.")
    guarded = [current_file, revision_file, config["canonical_import_parent"]["revision"],
               payload["diagnosis"], *payload["actual"].values(),
               *[row["registration"][key] for key in ("authority", "measurement", "labels", "source_image")]]
    result = dict(revision_id=revision_id, run_id=run, current=current_file,
        revision=revision_file, files=files, labels_shape=list(labels.shape),
        registration=deepcopy(row["registration"]), guarded_sources=guarded,
        label_serialization="Original Store int32; exact numeric equality to parent-typed translation",
        provenance="Fresh observed Store product SHA bound to genuine UUID/record and exact arrays; not historical source SHA")
    verify_registered_sources(result)
    return result


def verify_registered_sources(binding):
    """Recheck the exact already-admitted files without decoding them again."""
    for row in [*binding["guarded_sources"], *binding["files"].values()]:
        _verify(row)

def require_source_mode_transition(store, config):
    """The simple editor cannot remove the mode and bypass its authority."""
    mode = config.get("source_mode")
    if mode not in {None, "region-derived", IMPORTED, CANONICAL_EDIT}:
        raise ValueError("Unknown geometry source mode.")
    if mode == IMPORTED:
        imported_payload(config)
    elif mode == CANONICAL_EDIT:
        canonical_parent_payload(config)
    if store.current_path.is_file():
        current = store.current()
    else:
        # Ordinary first Save previously did not construct a project-derived
        # default configuration. Existing history still carries authority even
        # if current.json is absent; never bypass an imported historical parent.
        history = store.revisions()
        current = history[-1] if history else {}
    old = current.get("configuration") or {}
    if old.get("source_mode") in {IMPORTED, CANONICAL_EDIT} and mode not in {IMPORTED, CANONICAL_EDIT}:
        raise ValueError(SILENT_EDIT)
    if old.get("source_mode") == IMPORTED and mode != IMPORTED:
        parent = config.get("canonical_import_parent") or {}
        if mode != CANONICAL_EDIT or parent.get("revision_id") != current.get("revision_id"):
            raise ValueError(SILENT_EDIT)


def materialize_source_mode(config, staging, destination):
    """The Store's own artifacts; imported arrays and CSV are exact byte copies."""
    from .geometry_facies import _canonical_facies, _darsia_metadata
    mode = config["source_mode"]
    payload = imported_payload(config) if mode == IMPORTED else canonical_parent_payload(config)
    copied = {}
    for name in ("labels", "facies", "facies_properties"):
        target = Path(staging) / ("facies_properties.csv" if name == "facies_properties" else name + ".npz")
        if mode == CANONICAL_EDIT and name == "facies":
            facies, _ = _canonical_facies(config)
            np.savez_compressed(target, array=facies, metadata=_darsia_metadata(Path(config["labels_path"])))
        else:
            source = (config["labels_path"] if mode == CANONICAL_EDIT and name == "labels"
                      else payload["actual"][name]["path"])
            expected = descriptor(source)
            with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            _verify(expected, target)
            _verify(expected)
        row = descriptor(target)
        row["path"] = str(Path(destination) / target.name)
        copied[name] = row
    detail = difference_to_canonical(config, payload["facies"])
    if mode == CANONICAL_EDIT:
        labels = _integer_map(config["labels_path"])
        detail = {**detail, "labels_changed_from_parent": int(np.count_nonzero(labels != payload["labels"])),
                  "parent_revision": deepcopy(config["canonical_import_parent"]["revision"]),
                  "known_change": "Explicit canonical derivation from edited labels and the full original CSV; not an adopted-v1 identity gate."}
    return dict(source_mode=mode, labels_path=copied["labels"]["path"],
                source_mode_files=copied, source_mode_difference=detail,
                source_mode_diagnosis=payload["diagnosis"]), payload["rows"]

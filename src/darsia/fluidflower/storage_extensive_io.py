"""Optional, commit-anchored N14 I/O with explicitly synthetic physical inputs.

This module admits no physical producer. Existing job DATA cannot select it.
The native/N1 writers retain their original operands and publication formats.
"""
from copy import deepcopy
from pathlib import Path, PureWindowsPath
import re

import numpy as np

from . import storage_extensive as extensive
from . import storage_outputs as output

INPUT_FORMAT = "darsia-extensive-io-input-v1"
FRAME_FORMAT = "darsia-extensive-io-frame-v1"
QUALIFICATION = "explicit-synthetic-inputs"
ARCHIVE = "storage_extensive_frame.npz"
RECEIPT = "storage_extensive_frame.json"
PHYSICAL_KEYS = {
    "depth_m", "porosity", "pixel_pore_volume_m3", "facies", "physical_mask",
    "physical_properties_valid_mask", "mass_support_mask", "hidden_mask",
    "coverage_masks", "intensive_validity_masks", "saturation_valid_mask",
    "native_extensive_totals", "channels", "provenance",
}


def validate_inputs(value):
    if (not isinstance(value, dict)
            or set(value) != {"format", "qualification", "physical_producer_admitted", "physical_inputs"}
            or value["format"] != INPUT_FORMAT or value["qualification"] != QUALIFICATION
            or value["physical_producer_admitted"] is not False
            or not isinstance(value["physical_inputs"], dict)
            or set(value["physical_inputs"]) != PHYSICAL_KEYS):
        raise ValueError("Explicit closed synthetic-only extensive inputs required")
    return value["physical_inputs"]


def prepare_payload(value, grid, *, labels, saturation, masses, totals):
    physical = validate_inputs(value)
    if grid is None:
        raise ValueError("Extensive I/O requires an explicitly requested coarse grid")
    return extensive.aggregate_extensive(grid, **physical, phase_labels=labels,
        saturation_free_gas=saturation, native_pixel_masses_g=masses,
        native_mass_totals_g=totals)


def frame_identity(document):
    keys = {"format", "job_id", "run", "case_id", "configuration", "code_commit",
            "calibration_sha256", "precision", "source", "request"}
    if document["format"] == "darsia-n1-storage-frame-v1":
        keys |= {"source_binding", "manifest_sha256"}
    elif document["format"] != output.FRAME_FORMAT:
        raise ValueError("Only an existing native or genuine N1 storage identity is admitted")
    return deepcopy({key: document[key] for key in keys})


def declared_files(section):
    """Admit exactly two optional files; this never reads their payloads."""
    if section is None:
        return {}
    if (not isinstance(section, dict)
            or set(section) != {"format", "qualification", "physical_producer_admitted", "receipt", "archive"}
            or section["format"] != FRAME_FORMAT or section["qualification"] != QUALIFICATION
            or section["physical_producer_admitted"] is not False):
        raise ValueError("Extensive I/O section is not the closed synthetic-only format")
    result = {}
    for key, name, kind in (("receipt", RECEIPT, "operational_receipt"), ("archive", ARCHIVE, "map")):
        row = section[key]
        if (not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes", "kind"}
                or not isinstance(row["path"], str) or PureWindowsPath(row["path"]).name != name
                or type(row["size_bytes"]) is not int or row["size_bytes"] < 1
                or not isinstance(row["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or row["kind"] != kind):
            raise ValueError("Extensive I/O descriptor/name/kind differs")
        result[name] = row
    if PureWindowsPath(section["receipt"]["path"]).parent != PureWindowsPath(section["archive"]["path"]).parent:
        raise ValueError("Extensive I/O files belong to different frames")
    return result


def write_payload(temporary, final, product, document):
    """Add only new immutable files to the original caller-owned frame partial."""
    temporary, final = Path(temporary), Path(final)
    if any((temporary / name).exists() or (final / name).exists() for name in (ARCHIVE, RECEIPT)):
        raise FileExistsError("Extensive outputs must be new immutable frame files")
    archive = temporary / ARCHIVE
    output._write_npz(archive, product["arrays"])
    metadata = product["metadata"]
    # Reuse the strict pure reader; no second reduction or producer is run.
    extensive.read_extensive_archive(archive.read_bytes(), metadata,
        expected_input_identities=metadata["input_identities"],
        native_mass_totals_g=metadata["native_mass_totals_g"],
        native_extensive_totals=metadata["native_extensive_totals"],
        expected_channels=metadata["channels"], expected_provenance=metadata["provenance"],
        expected_grid_metadata=metadata["grid"])
    bound_archive = dict(output._asset(archive), path=str(final / ARCHIVE), kind="map")
    envelope = dict(format=FRAME_FORMAT, qualification=QUALIFICATION,
        physical_producer_admitted=False, frame_identity=frame_identity(document),
        archive=bound_archive, api_metadata=metadata)
    receipt = temporary / RECEIPT
    output._write_json(receipt, envelope)
    section = dict(format=FRAME_FORMAT, qualification=QUALIFICATION,
        physical_producer_admitted=False, archive=bound_archive,
        receipt=dict(output._asset(receipt), path=str(final / RECEIPT), kind="operational_receipt"))
    declared_files(section)
    return section


def _read_payload(stored, resolve_asset):
    from . import storage_saved_products as saved
    document = stored["storage_document"]
    section = document.get("extensive")
    files = declared_files(section)
    if not files:
        return dict(state="not_requested", physical_binding_admitted=False,
                    producer_identity_admitted=False)
    loaded = []
    envelope = saved._json(saved._read_bytes(files[RECEIPT], resolve_asset, loaded))
    if (not isinstance(envelope, dict)
            or set(envelope) != {"format", "qualification", "physical_producer_admitted", "frame_identity", "archive", "api_metadata"}
            or envelope["format"] != FRAME_FORMAT or envelope["qualification"] != QUALIFICATION
            or envelope["physical_producer_admitted"] is not False
            or not extensive._exact(envelope["frame_identity"], frame_identity(document))
            or not extensive._exact(envelope["archive"], files[ARCHIVE])):
        raise ValueError("Extensive I/O original frame/source/descriptor identity differs")
    metadata = envelope["api_metadata"]
    if (not isinstance(metadata, dict) or not {"input_identities", "native_mass_totals_g",
            "native_extensive_totals", "channels", "provenance", "grid"} <= set(metadata)):
        raise ValueError("Extensive I/O API metadata is incomplete")
    totals = {name: stored["native_scalar_reference"][key] for name, key in output.TOTALS.items()}
    result = extensive.read_extensive_archive(
        saved._read_bytes(files[ARCHIVE], resolve_asset, loaded), metadata,
        expected_input_identities=metadata["input_identities"], native_mass_totals_g=totals,
        native_extensive_totals=metadata["native_extensive_totals"],
        expected_channels=metadata["channels"], expected_provenance=metadata["provenance"],
        expected_grid_metadata=stored["grid"])
    for component in output.TOTALS:
        key = "cell_mass_" + component + "_g"
        actual, original = result["arrays"][key], stored["arrays"][key]
        if actual.dtype != original.dtype or not np.array_equal(actual, original):
            raise ValueError("Extensive " + component + " cell masses differ from committed coarse arrays")
    return dict(result, state="synthetic_io_only", loaded_assets=loaded,
                original_frame_identity=deepcopy(envelope["frame_identity"]))


def read_committed_payload(record, *, producer_kind, resolve_asset, manifest=None, manifest_asset=None):
    """First enforce the existing positive producer/commit/source reader."""
    if producer_kind == "native-phase-n1":
        from .storage_saved_products import read_stored_frame
        stored = read_stored_frame(record, resolve_asset=resolve_asset)
    elif producer_kind == "n1-mass-only":
        from .storage_n1_saved_products import read_n1_stored_frame
        stored = read_n1_stored_frame(record, manifest=manifest, manifest_asset=manifest_asset,
                                      resolve_asset=resolve_asset)
    else:
        raise ValueError("An explicit original native or genuine N1 producer kind is required")
    return _read_payload(stored, resolve_asset)


def read_indexed_extensive_run(index, *, producer_kind, current_machine_id, collections=(), source_paths=None):
    """One committed frame at a time; existing resolver/collect rules unchanged."""
    from . import storage_saved_products as saved
    from .storage_indexed_products import indexed_asset_resolver
    from .pipeline import products_publication as publication
    resolve = indexed_asset_resolver(index, current_machine_id=current_machine_id,
                                    collections=collections, source_paths=source_paths)
    completion = saved._json(publication.read_published_metadata(index))
    if producer_kind == "native-phase-n1":
        run = saved.read_stored_run(completion, resolve_asset=resolve, stream=True)
    elif producer_kind == "n1-mass-only":
        from .storage_n1_saved_products import read_n1_stored_run
        run = read_n1_stored_run(completion, resolve_asset=resolve, stream=True)
    else:
        raise ValueError("An explicit original native or genuine N1 producer kind is required")
    frames = (_read_payload(frame, resolve) for frame in run["frames"])
    return dict(state="synthetic_io_only", producer_kind=producer_kind,
                ordered_case_ids=list(run["ordered_case_ids"]), frames=frames,
                physical_binding_admitted=False, producer_identity_admitted=False)

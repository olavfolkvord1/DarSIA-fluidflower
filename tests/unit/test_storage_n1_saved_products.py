"""Tiny real N1 evaluate→publish→saved-reader tests; no native/phase replay.

Phase-one inputs are synthetic 3x3 saved artifacts written by the actual commit,
BEFORE and optional original-photo writers. They are not a Main/queue/RunIO or
campaign qualification. No candidate function is replaced with fake success.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower.mass import n1_pipeline as n1
from darsia.fluidflower import storage_n1_saved_products as saved_n1
from darsia.fluidflower import storage_saved_products as native_reader
from darsia.fluidflower import storage_n1_outputs as output
from darsia.fluidflower.pipeline import phase_statistics as stage
from darsia.fluidflower.pipeline import staged_data as staged
from darsia.fluidflower.pipeline.series import bind_mass_frame

_fixture_path = Path(__file__).parents[1] / "fixtures" / "n1_phase2_fixture.py"
if hashlib.sha256(_fixture_path.read_bytes()).hexdigest() != "1d47a4af55ba923fc29ef9b8d0f2bd3999e9821b548d7e79c4bdc2b1c8981979":
    raise RuntimeError("Frozen validation-only tiny saved-model/phase-one metadata fixture changed")
_fixture_spec = importlib.util.spec_from_file_location("ffac_test_fixture_n1_phase2", _fixture_path)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
tiny_config = _fixture_module.tiny_config
phase1_data = _fixture_module.phase1_data
COMMIT = "6dfca622f5ed8ebe5c3ea6474025dcf5bf28023d"
GEOMETRY = "f" * 64


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return n1._write_json(path, value)


def _copy(bound, destination):
    destination.write_bytes(Path(bound["path"]).read_bytes())
    return dict(bound, **n1.asset(destination))


def _photo_inputs(root, spec):
    """Only a synthetic saved geometry-only JPEG is decoded by the producer."""
    root.mkdir()
    path = root / "saved_native_review.jpg"
    Image.fromarray(np.arange(27, dtype=np.uint8).reshape(3, 3, 3) * 7).save(path, quality=95, subsampling=0)
    review = n1.asset(path)
    camera = dict(path=str(root / "camera_NEVER_READ.jpg"), sha256=spec["source"]["source_sha256"])
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    cert = dict(run=spec["source"]["run"], frame_index=17, camera_jpeg_path=camera["path"], camera_jpeg_sha256=camera["sha256"],
        native_review_jpeg_path=review["path"], native_review_jpeg_sha256=review["sha256"],
        native_shape_hw=[3, 3], content_box_xywh=[0, 0, 3, 3], H_ac28_to_frame_native=matrix,
        binding=dict(geometry_module_sha256=GEOMETRY, input_sources=dict(camera_jpeg=camera),
                     content_box_xywh=[0, 0, 3, 3], native_shape_hw=[3, 3], original_shape_configs=[], original_shape_corrections=[]))
    certificate = _json(root / "certificate.json", cert)
    native = _json(root / "native_geometry.json", dict(cert, geometry_certificate_path=certificate["path"],
                                                     geometry_certificate_sha256=certificate["sha256"]))
    current = _json(root / "current_geometry.json", dict(run=spec["source"]["run"], frame_index=spec["frame_index"],
        source_sha256=camera["sha256"], source_photo=camera,
        source_geometry_receipt={k: certificate[k] for k in ("path", "sha256")},
        source_content_box_xywh=[0, 0, 3, 3], source_native_shape_hw=[3, 3], native_origin_xy=[0, 0],
        phase_shape_hw=[3, 3], photograph_canvas_wh=[3, 3], H_ac28_to_frame_native=matrix))
    return dict(native_geometry=native, geometry_certificate=certificate, current_geometry=current,
                saved_native_review=review, geometry_producer_sha256=GEOMETRY)


def _inputs(tmp_path, tiny_config, phase1_data, *, before=True, photo=False, precision="float32"):
    """Seal real tiny saved fields as synthetic phase-one input commits."""
    config, _ = tiny_config
    config = deepcopy(config)
    config["precision"] = precision
    config["geometry_fields"] = dict(source_to_native="H_ac28_to_frame_native",
        original_domain_to_native="H_ac28_to_frame_native", carried_homography_provenance="original_H_ac28_to_native")
    config["shared"]["reference_matrix"] = [[.001, 0., .0005], [0., -.001, .0025], [0., 0., 1.]]
    config["shared"]["depth_points"] = [[0., .001, .01], [.003, .001, .01], [0., .003, .01], [.003, .003, .01]]
    config["summary"]["anchor_target_h"] = config["anchor"]["elapsed_h"]
    data = deepcopy(phase1_data)
    data["precision"] = precision
    data.update(run=config["run"], frames=[], ordered_case_ids=[])
    data["statistics"] = {key: deepcopy(config[key]) for key in ("shared", "geometry_fields", "physical_correction",
        "recipe", "dimensions_m", "boxes", "summary", "bindings", "contracts")}
    data["statistics"]["format"] = staged.STATISTICS_FORMAT
    data["phase"]["reference"]["full"] = deepcopy(config["frames"][0]["baseline"])
    data["phase"]["reference"]["source_sha256"] = "b" * 64
    data["phase"]["initial_previous"]["full"] = deepcopy(data["phase"]["reference"]["full"])
    data["photographs"] = {}
    original_spec = deepcopy(phase1_data["frames"][0])
    records, native_products, original_specs = [], [], []
    previous = "b" * 64
    for number, mass in enumerate(config["frames"]):
        spec = deepcopy(original_spec)
        source_sha = hashlib.sha256(("synthetic-acquisition-" + mass["case_id"]).encode()).hexdigest()
        mass["source_binding"] = dict(source_sha256=source_sha)
        spec.update(case_id=mass["case_id"], frame_index=mass["frame_index"], elapsed_h=mass["elapsed_h"])
        spec["source"].update(run=config["run"], work_index=mass["frame_index"], elapsed_h=mass["elapsed_h"],
                              source_sha256=source_sha, previous_source_sha256=previous,
                              source_binding=deepcopy(mass["source_binding"]))
        spec["mass_spec"] = {key: deepcopy(mass[key]) for key in ("atmosphere", "inner_polygon_native", "original_domain",
                            "photo_top", "nominal_injected_mass_g", "selected_target_h", "light_epoch")}
        spec["mass_spec"]["source_receipt"] = original_spec["mass_spec"]["source_receipt"]
        data["frames"].append(spec)
        data["ordered_case_ids"].append(spec["case_id"])
        original_specs.append(spec)
        root = tmp_path / "phase1" / spec["case_id"]
        root.mkdir(parents=True)
        full = deepcopy(mass["current"])
        full.update(complete=True, source_sha256=source_sha)
        centre = full["native_centre"]
        centre["log_linear_rgb"] = _copy(centre["log_linear_rgb"], root / "native_log.npy")
        centre["valid_mask"] = _copy(centre["valid_mask"], root / "native_valid.npy")
        native = dict(complete=True, source_sha256=source_sha, processor_kind="rig", full=full,
                      audit=_json(root / "native_audit.json", dict(synthetic_unit=True)))
        native_receipt = _json(root / "native.json", native)
        history = _json(root / "history_input.json", dict(complete=True, source=spec["source"], full=full,
            temporal_log=centre["log_linear_rgb"], temporal_valid=centre["valid_mask"]))
        phase = dict(complete=True, source_sha256=source_sha,
            labels=_copy(mass["labels"], root / "phase_labels.png"),
            classes=_copy(mass["classes"], root / "phase_classes.npz"),
            area=dict(_copy(mass["area"], root / "phase_area.npz"), array_key="pixel_area_mm2"),
            qa=_json(root / "phase_qa.json", dict(synthetic_unit=True)),
            operator_masks=_json(root / "phase_masks.json", dict(synthetic_unit=True)), history_input=history)
        capture = (dict(g3_pixel=np.linspace(0., 1., 9, dtype=np.float32).reshape(3, 3),
                        flatfield_valid=np.ones((3, 3), bool), source_box_xywh=[0, 0, 3, 3], probability_encoding=None)
                   if before else {})
        before_evidence = stage._save_before(root, spec, capture, [3, 3])
        request = _photo_inputs(tmp_path / ("photo_" + spec["case_id"]), spec) if photo else None
        if request:
            data["photographs"][spec["case_id"]] = request
        photograph = stage._save_original_photo(root, dict(run=config["run"], precision=precision),
                                               spec, request, n1.DataReader())
        statistics = stage._save_statistics(root, dict(statistics=dict(original_synthetic_count=9), sandmap={}))
        native_products.append((root, native, native_receipt, phase, before_evidence, photograph, statistics))
        previous = source_sha
    staged.validate_phase1_run(data)
    configuration = _json(tmp_path / "phase1_configuration.json", data)
    for spec, (root, native, native_receipt, phase, before_evidence, photograph, statistics) in zip(original_specs, native_products):
        commit = stage._commit_phase1(root, spec, task=dict(run=config["run"], precision=precision),
            configuration=configuration, native=dict(record=native, record_asset=native_receipt), phase=phase,
            statistics=statistics, before=before_evidence, photograph=photograph)
        records.append((commit, json.loads(Path(commit["path"]).read_bytes())))
    completion = _json(tmp_path / "phase1_completed.json", dict(format=staged.COMPLETION_FORMAT, complete=True,
        run=config["run"], configuration=configuration, precision=precision, code_commit="1" * 40,
        ordered_case_ids=data["ordered_case_ids"], frames=[bound for bound, _ in records], history_complete=True))
    config["frames"] = [bind_mass_frame(spec, record["native"], record["phase"], baseline=data["phase"]["reference"]["full"])
                         for spec, (_, record) in zip(original_specs, records)]
    config["phase1"] = dict(configuration=configuration, completion=completion, code_commit="1" * 40,
        frames={record["case_id"]: dict(commit=bound, statistics=record["statistics"], before_rules=record["before_rules"],
                                       photograph=record["photograph"]) for bound, record in records})
    config["storage"] = dict(format="darsia-storage-output-request-v1", resolution_mm=2,
                             gas_score_encoding="uint8-la-q255-v1", extent_m=[0., .004, 0., .004])
    return config


def _run(tmp_path, config, *, precision="float32"):
    bound = _json(tmp_path / "n1_configuration.json", config)
    completion = n1.run_n1(dict(run=config["run"], configuration=bound, commit=COMMIT, job_id="tiny-n1"),
                           tmp_path / "mass", "cpu", precision)
    return completion


def _run_with_extensive(tmp_path, config, monkeypatch, *, precision="float32"):
    """Pass the optional keyword at its direct API seam, never alter evaluation.

    Production callers intentionally cannot activate synthetic physical data.
    This test-only forwarding shim calls the real writer once per original case.
    """
    from .test_storage_saved_products import _synthetic_extensive_inputs
    original, calls = output.write_n1_frame, []
    def forward(*args, **kwargs):
        frame = kwargs["capture"]["frame"]
        labels = frame.phase_labels
        area = np.broadcast_to(np.asarray(frame.pixel_area_m2), labels.shape)
        saturation = kwargs["evaluated"]["export"]["products"]["saturation_free_gas"]
        calls.append(kwargs["spec"]["case_id"])
        return original(*args, **kwargs,
            extensive_inputs=_synthetic_extensive_inputs(labels, area, saturation))
    with monkeypatch.context() as context:
        context.setattr(output, "write_n1_frame", forward)
        completed = _run(tmp_path, config, precision=precision)
    assert calls == [spec["case_id"] for spec in config["frames"]]
    return completed


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_optional_extensive_real_N1_commit_and_reader_keep_genuine_producer(tiny_config, phase1_data, tmp_path, monkeypatch, precision):
    from darsia.fluidflower import storage_extensive_io as payload
    config = _inputs(tmp_path, tiny_config, phase1_data, precision=precision)
    completed = _run_with_extensive(tmp_path, config, monkeypatch, precision=precision)
    manifest = json.loads(Path(completed["manifest"]["path"]).read_bytes())
    for record in completed["records"]:
        def resolve(bound):
            assert Path(bound["path"]).name not in {"products.npz", "native_geometry.npz"}
            return Path(bound["path"])
        stored = saved_n1.read_n1_stored_frame(record, manifest=manifest,
            manifest_asset=completed["manifest"], resolve_asset=resolve)
        value = payload.read_committed_payload(record, producer_kind="n1-mass-only",
            manifest=manifest, manifest_asset=completed["manifest"], resolve_asset=resolve)
        assert value["state"] == "synthetic_io_only"
        assert value["physical_binding_admitted"] is value["producer_identity_admitted"] is False
        assert value["metadata"]["precision"] == precision
        assert value["original_frame_identity"]["format"] == output.FRAME_FORMAT
        assert value["original_frame_identity"]["source_binding"] == manifest["frames"][record["frame_id"]]["source_binding"]
        for name, total in stored["mass_totals_g"].items():
            array = value["arrays"]["cell_mass_" + name + "_g"]
            assert array.dtype == np.dtype(precision)
            assert abs(array.sum(dtype=np.float64) - total) <= 1e-6 * abs(total)


def test_optional_extensive_genuine_N1_index_collect_remains_saved_only(tiny_config, phase1_data, tmp_path, monkeypatch):
    from darsia.fluidflower import storage_extensive_io as payload
    from darsia.fluidflower.pipeline import products_publication as publication
    config = _inputs(tmp_path, tiny_config, phase1_data)
    completion = _run_with_extensive(tmp_path, config, monkeypatch)
    index = _index(tmp_path, completion)
    _root, _doc, rows = publication._read_index(index)
    wanted = [key for key, row in rows.items() if key.startswith("frames/") and row["kind"] == "map"
              and Path(row["local_path"]).name.startswith("storage_")]
    collection = publication.collect_products(index, wanted, tmp_path / "collected_extensive",
        current_machine_id="Main", file_writers_closed=True)
    original_read = Path.read_bytes
    def guard(path):
        assert not str(path).startswith(str(tmp_path / "mass")), "Producer payload fallback"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guard)
    run = payload.read_indexed_extensive_run(index, producer_kind="n1-mass-only",
        current_machine_id="Olav", collections=[collection])
    assert not isinstance(run["frames"], list)
    assert len(list(run["frames"])) == 2


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("photo", [False, True])
def test_actual_N1_writer_then_positive_reader_keeps_saved_phase1_evidence(tiny_config, phase1_data, tmp_path, before, photo):
    config = _inputs(tmp_path, tiny_config, phase1_data, before=before, photo=photo)
    completion = _run(tmp_path, config)
    assert "pipeline" not in completion
    stored = saved_n1.read_n1_stored_run(completion, resolve_asset=lambda b: Path(b["path"]), load_before=True)
    assert stored["producer_kind"] == "n1-mass-only" and len(stored["frames"]) == 2
    for item, spec in zip(stored["frames"], config["frames"]):
        assert item["source"]["source_binding"] == spec["source_binding"]
        assert item["before_rules"]["state"] == ("complete" if before else "pending")
        assert (item["before_native"] is not None) == before
        assert item["photograph"]["state"] == ("complete" if photo else "pending")
        assert item["grid"]["shape"] == [2, 2]
        assert all(not a.flags.writeable for a in item["arrays"].values())
        for component, total in item["mass_totals_g"].items():
            assert abs(total - item["native_scalar_reference"][output.output.TOTALS[component]]) <= 1e-6 * abs(total)
    with pytest.raises(ValueError, match="native-phase"):
        native_reader.read_stored_run(completion, resolve_asset=lambda b: Path(b["path"]))


def test_default_native_request_has_original_result_and_no_storage_files(tiny_config, tmp_path):
    config, _ = tiny_config
    config = deepcopy(config)
    config["geometry_fields"] = dict(source_to_native="H_ac28_to_frame_native",
        original_domain_to_native="H_ac28_to_frame_native", carried_homography_provenance="original_H_ac28_to_native")
    config["storage"] = dict(format="darsia-storage-output-request-v1", resolution_mm="native", gas_score_encoding="uint8-la-q255-v1")
    completion = _run(tmp_path, config)
    manifest = json.loads(Path(completion["manifest"]["path"]).read_bytes())
    assert "storage" not in manifest
    assert all(not any(name.startswith("storage_") for name in r["files"]) for r in completion["records"])


def test_optional_storage_keeps_all_original_native_arrays_and_scalar_values_exact(tiny_config, phase1_data, tmp_path):
    config = _inputs(tmp_path, tiny_config, phase1_data)
    before_dir, coarse_dir = tmp_path / "default", tmp_path / "coarse"
    plain = deepcopy(config)
    plain.pop("storage")
    before = _run(before_dir, plain)
    after = _run(coarse_dir, config)
    for old, new in zip(before["records"], after["records"]):
        original = json.loads((Path(old["nas_directory"]) / "result.json").read_bytes())
        candidate = json.loads((Path(new["nas_directory"]) / "result.json").read_bytes())
        for key in ("scalar_result", "total_mass_g", "aqueous_mass_g", "gas_mass_g", "bindings", "prediction"):
            assert original[key] == candidate[key]
        for name in ("products.npz", "native_geometry.npz"):
            with np.load(Path(old["nas_directory"]) / name, allow_pickle=False) as a, np.load(Path(new["nas_directory"]) / name, allow_pickle=False) as b:
                assert a.files == b.files
                for key in a.files:
                    assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                    assert a[key].tobytes() == b[key].tobytes()


@pytest.mark.parametrize("bad", ["core_commit", "source", "labels", "before", "unexplained_case"])
def test_actual_input_binding_mismatch_stops_before_optional_frame_write(tiny_config, phase1_data, tmp_path, bad):
    config = _inputs(tmp_path, tiny_config, phase1_data)
    case = config["frames"][0]["case_id"]
    if bad == "core_commit": config["phase1"]["frames"][case]["commit"]["sha256"] = "e" * 64
    elif bad == "source": config["frames"][0]["source_binding"]["source_sha256"] = "e" * 64
    elif bad == "labels": config["frames"][0]["labels"]["sha256"] = "e" * 64
    elif bad == "before": config["phase1"]["frames"][case]["before_rules"]["state"] = "pending"
    else: config["phase1"]["frames"]["unexpected"] = deepcopy(config["phase1"]["frames"][case])
    with pytest.raises(ValueError): _run(tmp_path, config)
    assert not list((tmp_path / "mass").glob("frames/*/storage_completed.json"))


@pytest.mark.parametrize("bad", ["pipeline", "format", "code", "configuration", "order", "storage_kind"])
def test_positive_reader_rejects_other_manifest_or_completion_kind(tiny_config, phase1_data, tmp_path, bad):
    config = _inputs(tmp_path, tiny_config, phase1_data)
    completion = _run(tmp_path, config)
    manifest = json.loads(Path(completion["manifest"]["path"]).read_bytes())
    if bad == "pipeline": completion["pipeline"] = "native-phase-n1"
    elif bad == "format": manifest["format"] = "darsia-series-run-manifest-v1"
    elif bad == "code": manifest["code"]["commit"] = "2" * 40
    elif bad == "configuration": completion["configuration"] = dict(completion["configuration"], sha256="d" * 64)
    elif bad == "order": completion["ordered_case_ids"].reverse()
    else: manifest["storage"]["producer_kind"] = "native-phase-n1"
    if bad in {"format", "code", "storage_kind"}:
        altered = _json(tmp_path / (bad + "_manifest.json"), manifest)
        completion["manifest"] = altered
    with pytest.raises(ValueError): saved_n1.read_n1_stored_run(completion, resolve_asset=lambda b: Path(b["path"]))


def test_stream_releases_first_coarse_arrays_before_reading_second(tiny_config, phase1_data, tmp_path):
    import weakref
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data))
    stream = saved_n1.read_n1_stored_run(completion, resolve_asset=lambda b: Path(b["path"]), stream=True)["frames"]
    first = next(stream)
    refs = [weakref.ref(a) for a in first["arrays"].values()]
    del first
    second = next(stream)
    assert all(ref() is None for ref in refs)
    assert second["case_id"] == completion["ordered_case_ids"][1]


def test_export_no_selection_is_IO_free(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_n1_saved_export as export
    def forbidden(*args, **kwargs): raise AssertionError("No-op attempted indexed IO")
    monkeypatch.setattr(export, "indexed_asset_resolver", forbidden)
    assert export.export_indexed_n1_storage({}, tmp_path / "none", current_machine_id="host")["state"] == "not_requested"
    assert not (tmp_path / "none").exists()


def _index(tmp_path, completion):
    """Original publication API over the actual closed N1 producer inventory."""
    from darsia.fluidflower.pipeline import products_publication as publication
    rows, metadata = [], []
    for record in completion["records"]:
        for leaf, entry in record["files"].items():
            identifier = "frames/" + record["frame_id"] + "/" + leaf
            kind = entry.get("kind", "operational_receipt" if leaf.endswith(".json") else "map")
            rows.append(publication.product(identifier, machine_id="Main", local_path=str(Path(record["nas_directory"]) / leaf),
                sha256=entry["sha256"], bytes=entry["size_bytes"], kind=kind))
            if kind == "operational_receipt": metadata.append(identifier)
    for producer in completion["frame_producers"]:
        for key in ("frame_registration", "publication"):
            entry = producer[key]
            identifier = "receipts/" + producer["case_id"] + "/" + key + ".json"
            rows.append(publication.product(identifier, machine_id="Main", local_path=entry["path"],
                sha256=entry["sha256"], bytes=entry["size_bytes"], kind="operational_receipt"))
            metadata.append(identifier)
    manifest = completion["manifest"]
    rows.append(publication.product("manifest.json", machine_id="Main", local_path=manifest["path"],
        sha256=manifest["sha256"], bytes=manifest["size_bytes"], kind="manifest"))
    metadata.append("manifest.json")
    bound = _json(tmp_path / "mass_completed.json", completion)
    job = _json(tmp_path / "synthetic_publication_identity.json", dict(synthetic_unit=True))
    return publication.publish_run_metadata(tmp_path / "shared", job=job, run=completion["run"],
        attempt_id="a" * 32, machine_id="Main", completed=bound, products=rows, metadata_product_ids=metadata,
        file_writers_closed=True, all_final_products_listed=True)


def test_actual_writer_publication_collect_and_foreign_saved_reader_need_no_native_inputs(tiny_config, phase1_data, tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline import products_publication as publication
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data, photo=True))
    index = _index(tmp_path, completion)
    _, _, rows = publication._read_index(index)
    wanted = [key for key, row in rows.items() if "/storage_" in key and row["kind"] == "map"]
    collected = publication.collect_products(index, wanted, tmp_path / "collected", current_machine_id="Olav",
        source_paths={key: row["local_path"] for key, row in rows.items() if key in wanted}, file_writers_closed=True)
    def forbidden(*args, **kwargs): raise AssertionError("Saved reader attempted native evaluation")
    monkeypatch.setattr(n1.N1Pipeline, "evaluate", forbidden)
    stored = saved_n1.read_indexed_n1_stored_run(index, current_machine_id="Olav", collections=[collected], load_before=True)
    assert len(stored["frames"]) == 2
    assert all("/storage_" in str(bound["path"]).replace("\\", "/") or Path(bound["path"]).name in {"manifest.json", "result.json", "commit.json"}
               for item in stored["frames"] for bound in item["loaded_assets"])


@pytest.mark.parametrize("photo,before,field,ready", [(False, False, "after_rules_coarse", False),
    (True, False, "after_rules_coarse", True), (True, False, "before_rules_coarse", False),
    (True, True, "before_rules_coarse", True)])
def test_N1_saved_export_keeps_photo_and_BEFORE_blockers_independent(tiny_config, phase1_data, tmp_path, photo, before, field, ready):
    from darsia.fluidflower import storage_n1_saved_export as export
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data, photo=photo, before=before))
    index = _index(tmp_path, completion)
    value = export.export_indexed_n1_storage(index, tmp_path / "export", current_machine_id="Main", table=True,
        film_field=field, fps=2., geometry_producer_sha256=GEOMETRY)
    assert value["table"] and Path(value["table"]["table"]["path"]).is_file()
    assert value["film_frames"]["sequence"]["original_left_frame_sequence_ready"] is ready
    assert value["film_frames"]["sequence"]["movie_encoded"] is False
    assert len(value["film_frames"]["sequence"]["frames"]) == (2 if ready else 0)


@pytest.mark.parametrize("field", ["concentration", "after_rules_coarse", "before_rules_native_pixels"])
def test_positive_N1_figures_use_saved_field_and_explicit_scale(tiny_config, phase1_data, tmp_path, field):
    from darsia.fluidflower import storage_n1_saved_export as export
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data))
    value = export.export_indexed_n1_storage(_index(tmp_path, completion), tmp_path / "figures_export", current_machine_id="Main",
        figure_field=field, concentration_limit_g_l=1. if field == "concentration" else None)
    assert value["complete"] and len(value["figures"]) == 2
    assert all(Path(item["preview"]["path"]).is_file() for item in value["figures"])


def test_unavailable_saved_photo_payload_blocks_only_film_not_mass_table(tiny_config, phase1_data, tmp_path):
    from darsia.fluidflower import storage_n1_saved_export as export
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data, photo=True))
    index = _index(tmp_path, completion)
    for record in completion["records"]:
        Path(record["nas_directory"]).joinpath("storage_original_photo.png").rename(
            Path(record["nas_directory"]).joinpath("preserved_unavailable_photo.png"))
    value = export.export_indexed_n1_storage(index, tmp_path / "missing_photo_export", current_machine_id="Main",
        table=True, film_field="after_rules_coarse", fps=2., geometry_producer_sha256=GEOMETRY)
    assert value["table"] and value["state"] == "not_film_ready"
    assert all(row["reason"] == "explicit_photo_collect_required" for row in value["film_frames"]["sequence"]["blocked"].values())


def _reseal_synthetic_metadata(completion, *, alter_manifest=None, alter_frame=None):
    """Deliberately malformed tiny metadata, with every outer byte hash valid.

    This is a unit-fixture corruption oracle. It does not claim a new producer
    or edit any campaign output. Scientific arrays and saved photo/BEFORE bytes
    stay exact; only the test's own metadata is rewritten and re-anchored.
    """
    def rewrite(path, value):
        path = Path(path)
        path.write_bytes((json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                    separators=(",", ":")) + "\n").encode())
        return n1.asset(path)
    value = deepcopy(completion)
    manifest = json.loads(Path(value["manifest"]["path"]).read_bytes())
    if alter_manifest: alter_manifest(manifest)
    value["manifest"] = rewrite(value["manifest"]["path"], manifest)
    for number, record in enumerate(value["records"]):
        directory = Path(record["nas_directory"])
        result = json.loads((directory / "result.json").read_bytes())
        commit = json.loads((directory / "commit.json").read_bytes())
        stored = json.loads((directory / "storage_completed.json").read_bytes())
        if alter_frame and number == 0: alter_frame(result, commit)
        result_bound = rewrite(directory / "result.json", result)
        stored["result"] = result_bound
        stored["manifest_sha256"] = value["manifest"]["sha256"]
        rewrite(directory / "storage_completed.json", stored)
        files = {}
        for name, old in commit["files"].items():
            fresh = n1.asset(directory / name)
            files[name] = {key: fresh[key] for key in ("sha256", "size_bytes")}
            if "kind" in old: files[name]["kind"] = old["kind"]
        commit["files"] = files
        commit["manifest_sha256"] = value["manifest"]["sha256"]
        commit_bound = rewrite(directory / "commit.json", commit)
        record["manifest_sha256"] = value["manifest"]["sha256"]
        record["files"] = dict(files, **{"commit.json": {key: commit_bound[key] for key in ("sha256", "size_bytes")}})
        # The requested malformed consistency check, not an incidental SHA
        # failure, must be the only remaining reason for a reader refusal.
        for name, bound in record["files"].items():
            actual = n1.asset(directory / name)
            assert all(bound[key] == actual[key] for key in ("sha256", "size_bytes"))
        assert stored["result"] == n1.asset(directory / "result.json")
    return value


def test_byte_reanchored_synthetic_metadata_control_is_still_readable(tiny_config, phase1_data, tmp_path):
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data))
    reanchored = _reseal_synthetic_metadata(completion)
    value = saved_n1.read_n1_stored_run(reanchored, resolve_asset=lambda b: Path(b["path"]))
    assert len(value["frames"]) == 2 and value["producer_kind"] == "n1-mass-only"


def _replace_synthetic_storage_json(path, value):
    """Alter existing test-owned metadata without changing the immutable producer."""
    assert path.is_file()
    path.write_bytes((json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                separators=(",", ":")) + "\n").encode())
    return n1.asset(path)


def _reseal_n1_storage_record(record, document, *, extra_files=()):
    """Commit a deliberately altered tiny storage document with valid outer hashes."""
    directory = Path(record["nas_directory"])
    _replace_synthetic_storage_json(directory / "storage_completed.json", document)
    commit = json.loads((directory / "commit.json").read_bytes())
    for name in extra_files:
        commit["files"][name] = dict(kind="operational_receipt")
    for name, old in list(commit["files"].items()):
        row = n1.asset(directory / name)
        commit["files"][name] = {key: row[key] for key in ("sha256", "size_bytes")}
        if "kind" in old:
            commit["files"][name]["kind"] = old["kind"]
    bound = _replace_synthetic_storage_json(directory / "commit.json", commit)
    record["files"] = dict(commit["files"], **{"commit.json": {key: bound[key] for key in ("sha256", "size_bytes")}})
    for name, row in record["files"].items():
        actual = n1.asset(directory / name)
        assert all(row[key] == actual[key] for key in ("sha256", "size_bytes"))


@pytest.mark.parametrize("omitted", ["both", "archive", "receipt", "unknown"])
def test_committed_N1_storage_cannot_be_hidden_from_document_inventory(tiny_config, phase1_data, tmp_path, monkeypatch, omitted):
    from darsia.fluidflower import storage_extensive_io as payload
    config = _inputs(tmp_path, tiny_config, phase1_data)
    completed = _run_with_extensive(tmp_path, config, monkeypatch)
    record = completed["records"][0]
    directory = Path(record["nas_directory"])
    document = json.loads((directory / "storage_completed.json").read_bytes())
    if omitted == "both":
        document.pop("extensive")
        document["files"] = [row for row in document["files"] if Path(row["path"]).name not in {payload.ARCHIVE, payload.RECEIPT}]
    elif omitted == "unknown":
        _json(directory / "storage_undeclared.json", dict(synthetic_unexplained=True))
    else:
        document.pop("extensive")
        # Remove both declarations so the old document-only set was valid,
        # while retaining exactly the selected extensive file in the commit.
        document["files"] = [row for row in document["files"] if Path(row["path"]).name not in {payload.ARCHIVE, payload.RECEIPT}]
        other = payload.RECEIPT if omitted == "archive" else payload.ARCHIVE
        commit = json.loads((directory / "commit.json").read_bytes())
        del commit["files"][other]
        _replace_synthetic_storage_json(directory / "commit.json", commit)
    _reseal_n1_storage_record(record, document, extra_files=("storage_undeclared.json",) if omitted == "unknown" else ())
    manifest = json.loads(Path(completed["manifest"]["path"]).read_bytes())
    reads = []
    def resolve(bound):
        reads.append(Path(bound["path"]).name)
        return Path(bound["path"])
    with pytest.raises(ValueError, match="N1 committed storage inventory differs from its declared flat files"):
        saved_n1.read_n1_stored_frame(record, manifest=manifest, manifest_asset=completed["manifest"], resolve_asset=resolve)
    assert not any(name.endswith(".npz") for name in reads)


def test_resealed_extensive_mass_permutation_cannot_replace_genuine_N1_coarse_map(tiny_config, phase1_data, tmp_path, monkeypatch):
    from .test_storage_saved_products import _reseal_extensive_array_values
    from darsia.fluidflower import storage_extensive_io as payload
    completed = _run_with_extensive(tmp_path, _inputs(tmp_path, tiny_config, phase1_data), monkeypatch)
    record = completed["records"][0]
    document = json.loads((Path(record["nas_directory"]) / "storage_completed.json").read_bytes())
    def permute(arrays):
        key = "cell_mass_total_g"
        before = arrays[key].copy()
        valid = np.flatnonzero(arrays["pore_volume_m3"].ravel() > 0)
        first = int(valid[np.argmin(before.ravel()[valid])])
        second = int(valid[np.argmax(before.ravel()[valid])])
        assert before.flat[first] != before.flat[second]
        arrays[key].flat[first], arrays[key].flat[second] = before.flat[second], before.flat[first]
        assert arrays[key].sum(dtype=np.float64) == before.sum(dtype=np.float64)
    updated = _reseal_extensive_array_values(document["extensive"], permute)
    document["files"] = [updated.get(Path(row["path"]).name, row) for row in document["files"]]
    _reseal_n1_storage_record(record, document)
    manifest = json.loads(Path(completed["manifest"]["path"]).read_bytes())
    with pytest.raises(ValueError, match="Extensive total cell masses differ from committed coarse arrays"):
        payload.read_committed_payload(record, producer_kind="n1-mass-only", manifest=manifest,
            manifest_asset=completed["manifest"], resolve_asset=lambda bound: Path(bound["path"]))


@pytest.mark.parametrize("bad", ["algorithm", "digest"])
def test_manifest_original_commit_ASCII_identity_is_consistent_beyond_outer_SHA(tiny_config, phase1_data, tmp_path, bad):
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data))
    def change(manifest):
        if bad == "algorithm": manifest["code"]["identity_algorithm"] = "different-algorithm"
        else: manifest["code_sha256"] = "d" * 64
    malformed = _reseal_synthetic_metadata(completion, alter_manifest=change)
    with pytest.raises(ValueError, match="Genuine N1 manifest code identity differs"):
        saved_n1.read_n1_stored_run(malformed, resolve_asset=lambda b: Path(b["path"]))


@pytest.mark.parametrize("field", ["completion", "manifest", "commit_settings", "commit_products", "result"])
def test_backend_consistency_rejects_each_original_metadata_field_beyond_outer_SHA(tiny_config, phase1_data, tmp_path, field):
    completion = _run(tmp_path, _inputs(tmp_path, tiny_config, phase1_data))
    def manifest_change(value):
        if field == "manifest": value["settings"]["backend"] = "cuda"
    def frame_change(result, commit):
        if field == "result": result["backend"] = "cuda"
        elif field == "commit_settings": commit["settings"]["backend"] = "cuda"
        elif field == "commit_products": commit["products"]["backend"] = "cuda"
    malformed = _reseal_synthetic_metadata(completion, alter_manifest=manifest_change, alter_frame=frame_change)
    if field == "completion": malformed["backend"] = "cuda"
    message = ("N1 saved completion backend differs from its manifest" if field in {"completion", "manifest"}
               else "N1 saved backend differs across manifest/commit/result")
    with pytest.raises(ValueError, match=message):
        saved_n1.read_n1_stored_run(malformed, resolve_asset=lambda b: Path(b["path"]))

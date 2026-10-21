"""Compile complete registered run inputs into a fresh, inert analysis plan."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess

from .analysis_plan import phase1_configuration, write_phase1_plan, write_phase2_plan
from .jobs_metadata import JobSnapshotReader
from .pipeline.job_queue import descriptor


def job_identity(repository):
    """Bind the current tracked source and lock; keep unrelated user files."""
    repository = Path(repository)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], text=True)
    untracked = subprocess.check_output(["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard", "--", "src"], text=True)
    if dirty.strip() or untracked.strip():
        raise ValueError("Commit the analysis source before preparing a plan")
    relative = "requirements-worker.lock"
    lock = subprocess.check_output(["git", "-C", str(repository), "show", head + ":" + relative])
    if (repository / relative).read_bytes().replace(b"\r\n", b"\n") != lock.replace(b"\r\n", b"\n"):
        raise ValueError("The current environment lock differs from the tracked lock")
    return dict(commit=head, precision="float32", created_utc=datetime.now(timezone.utc).isoformat(),
                environment_lock=dict(path=relative, sha256=hashlib.sha256(lock).hexdigest()))


def prepare_phase1_inputs(inputs, folder, *, series_root, geometry_index, pressure_index,
                          gas_probability, job_id, job, threads=1):
    """Read only supplied bound roles; refuse every missing run before writing.

    The input manifest binds each complete run configuration and the registered
    series manifest. It is produced by source preparation, not guessed from GUI
    strings. Geometry and GFI indices are explicit selected inputs.
    """
    from .pipeline.alias_job import _manifest
    from .pipeline.job_inputs import InputCatalog
    reader = JobSnapshotReader()
    read = lambda bound: reader.read(Path(bound["path"]), bound)
    document, geometry, pressure = read(inputs), read(geometry_index), read(pressure_index)
    manifest = descriptor(Path(series_root) / "series.toml")
    if (document.get("format") != "darsia-analysis-plan-inputs-v1"
            or document.get("series_manifest") != manifest):
        raise ValueError("Choose the source-preparation inputs for this exact registered series")
    _, registered, roots = _manifest(manifest)
    rows = document.get("runs")
    if not isinstance(rows, list) or not rows or len({row["run"] for row in rows}) != len(rows):
        raise ValueError("Source preparation must bind the complete distinct run order")
    if (geometry.get("format") != "darsia-measured-run-geometry-v1"
            or geometry.get("reference_models_complete") is not True):
        raise ValueError("The selected measured geometry index is incomplete")
    configurations, resources, errors = [], {}, []
    for row in rows:
        run = row["run"]
        try:
            original = InputCatalog(roots).doc(row["configuration"], "configuration", "analysis_inputs." + run)
            if original["run"] != run:
                raise ValueError("Run input belongs to another run")
            frame_index = read(geometry["runs"][run]["frame_data"])
            if (frame_index.get("run") != run or frame_index.get("complete_for_this_run") is not True
                    or set(frame_index["frames"]) != set(original["ordered_case_ids"])):
                raise ValueError("Measured frame geometry does not cover the complete selected order")
            frames = {case: read(value["frame_model"]) for case, value in frame_index["frames"].items()}
            atmosphere = read(pressure["runs"][registered["series_id"]][run]["atmosphere_series"])
            if atmosphere.get("provisional_gap_default") is True:
                raise ValueError("Provisional pressure gaps await an explicit decision")
            calibration = read(original["mass"]["calibration"])
            method = {key: calibration[key] for key in ("round4_spec_sha256", "amendment_sha256")}
            candidate = phase1_configuration(original, precision=job["precision"],
                atmosphere=atmosphere, frame_data=frames, gas_probability=gas_probability,
                method_bindings=method, role_catalog=InputCatalog(roots, original.get("data_aliases", ())))
            provenance = candidate.setdefault("provenance", {})
            if "analysis_plan_inputs" in provenance:
                raise ValueError("The input run already binds another plan-preparation record")
            provenance["analysis_plan_inputs"] = dict(source_manifest=inputs, original_configuration=row["configuration"],
                geometry_index=geometry_index, pressure_index=pressure_index,
                original_method_source=original["mass"]["calibration"])
            configurations.append(candidate)
            resources[run] = dict(backend="cpu", threads=threads)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(run + ": " + str(error))
    if errors:
        raise ValueError("Analysis inputs are incomplete; no plan was written:\n" + "\n".join(errors))
    if descriptor(Path(series_root) / "series.toml") != manifest:
        raise ValueError("The registered series changed during preparation")
    return write_phase1_plan(folder, configurations, job_id=job_id, job=job, resources=resources)


def prepare_phase2_inputs(inputs, folder, *, series_root, fitted_calibration, job_id, job, threads=1):
    """Use explicit genuine completion/anchor/storage roles from source preparation."""
    from .mass.n1_pipeline import DataReader
    reader = DataReader(unc_only=False)
    document = reader.json(inputs)
    manifest = descriptor(Path(series_root) / "series.toml")
    if (document.get("format") != "darsia-analysis-phase2-inputs-v1"
            or document.get("series_manifest") != manifest):
        raise ValueError("Choose the saved phase-one completion inputs")
    rows = document.get("runs")
    if not isinstance(rows, list) or not rows or len({row["run"] for row in rows}) != len(rows):
        raise ValueError("Explicit distinct phase-one completion order required")
    sources = []
    for row in rows:
        source = {key: row[key] for key in (
            "completion_bound", "configuration_bound", "expected_phase1_commit", "fitted_anchor")}
        source["fitted_calibration"] = fitted_calibration
        for key in ("storage_data", "pixel_domain_data", "storage_profile"):
            if key in row:
                source[key] = row[key]
        configuration = reader.json(source["configuration_bound"])
        if configuration.get("run") != row["run"]:
            raise ValueError("Phase-one completion input belongs to another run")
        sources.append(source)
    if descriptor(Path(series_root) / "series.toml") != manifest:
        raise ValueError("The registered series changed during preparation")
    selected_job = dict(job, calibration_sha256=fitted_calibration["sha256"])
    return write_phase2_plan(folder, sources, job_id=job_id, job=selected_job,
        resources={row["run"]: dict(backend="cpu", threads=threads) for row in rows}, reader=reader)

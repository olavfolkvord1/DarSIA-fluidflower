"""Bind frozen phase rules to an explicitly prepared experiment context.

Only the context registries change. No settings, calibration, optical weights,
or source files are fitted, inferred, or modified by this helper.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np

from darsia.fluidflower.phase_plume import PlumeContext, freeze_context
from darsia.fluidflower.phase_runtime import load_physics_profile, load_run_context, physics_runtime_identity
from darsia.fluidflower.phase_training.scope import series_id
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path


FORMAT = "phase-transfer-target-context-v1"
FACIES_IDENTITY = "same-series-categorical-labels"
PROFILE_FILES = ("phase_native_reference.json", "phase_facies_reference.json", "phase_reviewed_reference.json",
    "phase_reference_support.json", "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json",
    "phase_optical_regions.json")
SUPPORT_KEYS = ("reference_support", "presence_support", "gas_support")


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def resolve(path, origin):
    result = Path(path)
    return resolve_review_path(result if result.is_absolute() else origin / result)


def _same_historical_localcache_manifest(source: Path, bound: Path) -> bool:
    """Recognize the old Codex MSIX LocalCache spelling of one source path.

    This is intentionally limited to phase-transfer source manifests. It does
    not make arbitrary same-content files interchangeable as provenance.
    """
    for sandbox, ordinary in ((source, bound), (bound, source)):
        parts = sandbox.parts
        lower = tuple(part.casefold() for part in parts)
        for index in range(len(parts) - 6):
            if lower[index:index + 3] != ("appdata", "local", "packages"):
                continue
            if not lower[index + 3].startswith("openai.codex_"):
                continue
            if lower[index + 4:index + 6] != ("localcache", "local"):
                continue
            suffix = lower[index + 6:]
            if (len(suffix) != 6 or suffix[:4] != ("darsia", "fluidflower", "execution", "phase_transfer_targets")
                    or suffix[-1] != "postflight_manifest.json"):
                continue
            expected = Path(*parts[:index + 2], *parts[index + 6:])
            if os.path.normcase(str(expected)) == os.path.normcase(str(ordinary)):
                return True
    return False


def prepared_target_physics(model_root, target, *, expected_source_manifest_sha256=None):
    """Validate registered target inputs without creating files or running inference.

    ``target.job.phase_transfer_context`` must contain the descriptor. Its
    provenance.target_binding repeats dataset/run/native SHA/source manifest,
    making the provenance explicitly belong to the selected experiment.
    """
    root = resolve_review_path(model_root)
    source_manifest = resolve_review_path(target["source_manifest"])
    descriptor = copy.deepcopy(target.get("job", {}).get("phase_transfer_context"))
    if not isinstance(descriptor, dict) or descriptor.get("format") != FORMAT:
        raise ValueError("Target needs an explicit phase-transfer target-context descriptor")
    dataset, run = str(target["dataset"]), str(target["run"])
    if series_id(dataset) != dataset or dataset not in {"ffac", "ffum", "ffbm"} or not run or run != run.casefold():
        raise ValueError("Target context requires canonical dataset/run identifiers")
    key = f"{dataset}/{run}"
    native_path = resolve_review_path(target["native_manifest"])
    native_hash = sha256_file(native_path)
    expected_binding = dict(dataset=dataset, run=run, native_manifest_sha256=native_hash)
    if native_hash != target["native_manifest_sha256"] or any(descriptor.get(k) != v for k, v in expected_binding.items()):
        raise ValueError("Target context identifies another dataset, run or native source revision")
    registered = [job for job in read(source_manifest).get("jobs", []) if isinstance(job, dict)
                  and str(job.get("run") or "").casefold() == run
                  and job.get("phase_transfer_context") == descriptor]
    if not registered or any(target.get("job", {}).get(k) != v for k, v in (('dataset', dataset), ('run', run))):
        raise ValueError("Prepared descriptor is not registered on the selected source experiment")
    if descriptor.get("facies_identity") != FACIES_IDENTITY:
        raise ValueError("Target facies must explicitly use the same-series categorical labels")
    provenance = descriptor.get("provenance")
    binding = provenance.get("target_binding") if isinstance(provenance, dict) else None
    if not isinstance(binding, dict) or any(binding.get(k) != v for k, v in expected_binding.items()):
        raise ValueError("Target context provenance must bind its own dataset, run and native revision")
    bound_value = Path(binding["source_manifest"]) if binding.get("source_manifest") else None
    bound_manifest = (resolve_review_path(bound_value if bound_value.is_absolute()
                      else source_manifest.parent / bound_value, context=source_manifest.parent)
                      if bound_value is not None else None)
    if bound_manifest is None:
        raise ValueError("Target context provenance identifies another source manifest")
    if (expected_source_manifest_sha256 is not None
            and sha256_file(source_manifest) != expected_source_manifest_sha256):
        raise ValueError("Target context source manifest changed from its frozen revision")
    if bound_manifest.resolve() != source_manifest.resolve():
        if not _same_historical_localcache_manifest(source_manifest, bound_manifest):
            raise ValueError("Target context provenance identifies another source manifest")
        try:
            bound_manifest.stat()
        except FileNotFoundError:
            # Native Windows processes cannot see the unvirtualized spelling
            # written by the old packaged Codex app. Only a separately frozen
            # source digest permits this known structural alias.
            if expected_source_manifest_sha256 is None:
                raise ValueError("Target context provenance identifies another source manifest") from None
        except OSError as error:
            raise ValueError("Target context provenance identifies another source manifest") from error
        else:
            if (not bound_manifest.is_file()
                    or sha256_file(bound_manifest) != sha256_file(source_manifest)):
                raise ValueError("Target context provenance identifies another source manifest")
    context_path = resolve(descriptor["path"], source_manifest.parent)
    if context_path.is_relative_to(root):
        raise ValueError("Prepared target context must come from its own inputs, not another run in the source model")
    if sha256_file(context_path) != descriptor.get("sha256"):
        raise ValueError("Prepared target context checksum changed")
    profile = load_physics_profile(root)
    profile_paths = [root / name for name in PROFILE_FILES if (root / name).is_file()]
    if profile is None or len(profile_paths) != 1:
        raise ValueError("Prepared-target mode requires one frozen source physics profile")
    if {str(k).split("/")[0] for k in profile["runs"]} != {dataset}:
        raise ValueError("Source physics and prepared categorical context must belong to the same series")
    with np.load(context_path, allow_pickle=False) as archive:
        context = PlumeContext(domain=archive["domain"], facies=archive["facies"],
            x_m=archive["x_m"], y_m=archive["y_m"],
            injection_stages=tuple(copy.deepcopy(descriptor["injection_stages"])),
            permeability=archive["permeability"] if "permeability" in archive else None)
    if not np.issubdtype(context.facies.dtype, np.integer):
        raise ValueError("Facies must contain categorical integer labels, not continuous fields")
    native = read(native_path)
    box = native["content_box_xywh"]
    canvas = native["canvas_wh"]
    if (len(box) != 4 or len(canvas) != 2 or any(type(value) is not int for value in (*box, *canvas))
            or min(box[:2]) < 0 or min(box[2:]) < 2 or box[0] + box[2] > canvas[0]
            or box[1] + box[3] > canvas[1] or context.domain.shape != (box[3], box[2])):
        raise ValueError("Prepared target context differs from the native crop geometry")
    # Legacy physics additionally requires permeability and an explicit rate
    # protocol. Validate its existing contract before writing an evaluation.
    from darsia.fluidflower.phase_physics import PHASE_PHYSICS_VERSION, PhasePhysicsContext
    if profile["format"] == PHASE_PHYSICS_VERSION:
        if context.permeability is None:
            raise ValueError("This frozen physics model requires target permeability")
        PhasePhysicsContext(context.domain, context.facies, context.permeability,
            context.x_m, context.y_m, context.injection_stages)
    source_files = [source_manifest, native_path, context_path, profile_paths[0]]
    source_files += extension_files(root)
    return dict(profile=profile, profile_path=profile_paths[0], source_runtime=physics_runtime_identity(profile),
        context_path=context_path, context=context, provenance=copy.deepcopy(provenance), key=key,
        source_files=source_files, source_sha256={str(path): sha256_file(path) for path in source_files},
        descriptor=descriptor)


def extension_files(root):
    """Result extensions are separate bundle files, never inline base rules."""
    root = Path(root)
    files = [root/name for name in ('phase_aqueous_context_completion.json',
        'phase_local_aqueous_support.json', 'phase_decision.json') if (root/name).is_file()]
    completion = root/'phase_aqueous_context_completion.json'
    if completion.is_file():
        checkpoint = (root/read(completion)['checkpoint']).resolve()
        if not checkpoint.is_relative_to(root.resolve()):
            raise ValueError('Completion checkpoint escaped source model')
        files.append(checkpoint)
    regions = root / 'phase_optical_regions.json'
    if regions.is_file():
        model = read(regions)['model']
        checkpoint = (root / model['path']).resolve()
        if not checkpoint.is_relative_to(root.resolve()) or sha256_file(checkpoint) != model['sha256']:
            raise ValueError('Regional optical checkpoint changed or escaped source model')
        files.append(checkpoint)
    return files


def _replace_registries(profile, key, entry):
    result = copy.deepcopy(profile)
    def visit(value):
        if not isinstance(value, dict):
            raise ValueError("Invalid nested frozen physics profile")
        if "format" in value and isinstance(value.get("runs"), dict):
            value["runs"] = {key: copy.deepcopy(entry)}
        for name in SUPPORT_KEYS:
            if name in value:
                visit(value[name])
    visit(result)
    return result


def freeze_target_physics(evaluation_root, model_root, target, prepared):
    """Write a new context and profile while retaining every calibrated value."""
    evaluation, original = Path(evaluation_root).resolve(), Path(model_root).resolve()
    if evaluation == original or evaluation.is_relative_to(original) or original.is_relative_to(evaluation):
        raise ValueError("Evaluation must be separate from the frozen source model")
    for name, digest in prepared["source_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Prepared physical input changed: {name}")
    # Revalidate/reload immutable inputs so mutated in-memory arrays cannot be
    # mistaken for the registered NPZ, nor can a different target be substituted.
    checked = prepared_target_physics(original, target)
    if (checked["key"] != prepared["key"] or checked["source_sha256"] != prepared["source_sha256"]
            or checked["descriptor"] != prepared["descriptor"] or checked["source_runtime"] != prepared["source_runtime"]):
        raise ValueError("Prepared target identity changed before freezing")
    if any((evaluation / name).exists() for name in PROFILE_FILES):
        raise FileExistsError("Evaluation already contains a physical profile")
    entry = freeze_context(evaluation, target["dataset"], target["run"], checked["context"],
        native_manifest=Path(target["native_manifest"]), provenance=checked["provenance"])
    # load_physics_profile attaches completion/decision sidecars in memory.
    # Persisting that expanded object in the base JSON would either lose their
    # portable checkpoint or attach the extensions twice on the next load.
    profile = _replace_registries(read(checked['profile_path']), checked["key"], entry)
    path = evaluation / checked["profile_path"].name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2, allow_nan=False)
    for source in extension_files(original):
        destination = evaluation/source.relative_to(original)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != checked['source_sha256'][str(source)]:
            raise ValueError('Copied result extension differs from source model')
    loaded = load_physics_profile(evaluation)
    expected = _replace_registries(checked['profile'], checked['key'], entry)
    if 'aqueous_context_completion' in expected:
        expected['aqueous_context_completion']['_checkpoint_path'] = str(
            (evaluation/expected['aqueous_context_completion']['checkpoint']).resolve())
    if loaded != expected or physics_runtime_identity(loaded) != physics_runtime_identity(expected):
        raise ValueError("Frozen target physics failed profile validation")
    context, _ = load_run_context(evaluation, loaded, target["dataset"], target["run"],
        native_manifest_sha256=target["native_manifest_sha256"])
    for name in ("domain", "facies", "x_m", "y_m", "permeability"):
        before, after = getattr(checked["context"], name), getattr(context, name)
        if (before is None) != (after is None) or (before is not None and not np.array_equal(before, after, equal_nan=True)):
            raise ValueError("Freezing changed target physical arrays")
    for name, digest in checked["source_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Original physical input changed: {name}")
    return loaded

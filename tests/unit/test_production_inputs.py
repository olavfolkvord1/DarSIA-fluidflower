"""Local synthetic bytes/metadata and ASTs; never load native images/arrays."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "src/darsia/fluidflower"
PACKAGE = ModuleType("production_inputs_unit_package")
PACKAGE.__path__ = [str(SOURCES)]
sys.modules[PACKAGE.__name__] = PACKAGE


def module(name):
    spec = importlib.util.spec_from_file_location(PACKAGE.__name__ + "." + name, SOURCES / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


pio = module("production_io")
inputs = module("production_inputs")


def extract(path, names, namespace):
    tree = ast.parse((SOURCES / path).read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(SOURCES / path), "exec"), namespace)
    return namespace


@pytest.fixture
def staged(tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    code = dict(commit="0" * 40, clean_tree=True, submodules=[])
    monkeypatch.setattr(inputs, "_code_identity", lambda root: deepcopy(code))
    manifest = dict(format="darsia-production-job-v1", code=code, inputs={},
                    settings=dict(backend="cpu", precision="float64"),
                    purpose="file_handling_acceptance", production_permission_granted=False)
    local = {}
    def add(role, logical, data):
        path = cache / role
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        descriptor = dict(path=logical, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        manifest["inputs"][role] = descriptor
        local[role] = {**descriptor, "path": str(path)}
        return descriptor
    def reader(**kwargs):
        return inputs.ManifestAssetReader(manifest, local, cache_root=cache, worktree=worktree, **kwargs)
    return SimpleNamespace(worktree=worktree, cache=cache, manifest=manifest, local=local, add=add, reader=reader)


def test_local_only_compact_and_logical_provenance(staged, monkeypatch):
    original = staged.add("json", r"\\NAS\share\folder\entry.json", b'{"exact_epoch":"1791789968.2188113"}')
    reader = staged.reader()
    compact = {k: original[k] for k in ("path", "sha256")}
    assert reader.json(compact)["exact_epoch"] == "1791789968.2188113"
    assert reader.path(compact) == staged.cache / "json"
    assert reader.logical_descriptor(compact) == original
    assert compact == {k: original[k] for k in ("path", "sha256")}
    # No fallback: even if an original path happens to exist, it is not opened.
    unregistered = staged.worktree / "unregistered.json"
    unregistered.write_text("{}")
    with pytest.raises(ValueError, match="absent"):
        reader.resolve_path(unregistered)


def test_explicit_alias_preserves_old_metadata_and_never_opens_mutable_source(staged, monkeypatch):
    old = staged.worktree / "old_absolute.json"
    old.write_bytes(b"wrong mutable bytes")
    canonical = staged.add("archived", str(staged.worktree / "sealed_archive" / "source.json"),
                           b'{"original_locator":"unchanged","exact_epoch":"1791789968.2188113"}')
    reader = staged.reader(logical_aliases={str(old): canonical})
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path == old: pytest.fail("reader touched mutable original")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    request = dict(path=str(old), sha256=canonical["sha256"])
    assert reader.path(request) == reader.resolve_path(old, canonical["sha256"]) == staged.cache / "archived"
    assert reader.json(request)["original_locator"] == "unchanged"
    assert reader.logical_descriptor(request) == {**canonical, "path": str(old)}
    assert reader.logical_aliases == {str(old): canonical}
    assert request == dict(path=str(old), sha256=canonical["sha256"])


@pytest.mark.parametrize("kind", ["unbound", "sha", "size", "compact", "canonical_conflict", "alias_conflict", "relative", "chain"])
def test_alias_binding_failures(staged, kind):
    first = staged.add("first", r"C:\sealed\first.py", b"frozen first")
    second = staged.add("second", r"C:\sealed\second.py", b"frozen second")
    alias = r"C:\old\original.py"
    target = dict(first)
    aliases = {alias: target}
    if kind == "unbound": target["path"] = r"C:\undeclared\same_SHA.py"
    if kind == "sha": target["sha256"] = "f" * 64
    if kind == "size": target["size_bytes"] += 1
    if kind == "compact": target.pop("size_bytes")
    if kind == "canonical_conflict": aliases = {first["path"]: second}
    if kind == "alias_conflict": aliases[r"c:/OLD/original.py"] = second
    if kind == "relative": aliases = {"old/relative.py": first}
    if kind == "chain": aliases[r"C:\another\original.py"] = {**first, "path": alias}
    with pytest.raises(ValueError): staged.reader(logical_aliases=aliases)


def test_alias_sha_and_cache_tamper_remain_fail_closed(staged):
    canonical = staged.add("archive", r"C:\archive\source.py", b"exact source bytes")
    alias = r"\\NAS\share\old\source.py"
    reader = staged.reader(logical_aliases={alias: canonical})
    assert reader.resolve_path(r"//nas/SHARE/old/../old/source.py", canonical["sha256"]) == staged.cache / "archive"
    with pytest.raises(ValueError): reader.resolve_path(alias, "f" * 64)
    reader.path(dict(path=alias, sha256=canonical["sha256"]))
    (staged.cache / "archive").write_bytes(b"corrupt cached source")
    with pytest.raises(ValueError): reader.resolve_path(alias, canonical["sha256"])


def test_locator_alias_cannot_substitute_an_execution_source_without_review(staged):
    canonical = staged.add("archive", r"C:\archive\source.py", b"original source bytes")
    current = staged.worktree / "current_execution.py"
    current.write_bytes(b"changed current IO wiring")
    reader = staged.reader(logical_aliases={str(current): canonical})
    assert reader.resolve_path(current, canonical["sha256"]) == staged.cache / "archive"
    with pytest.raises(ValueError, match="separate reviewed source_provenance"):
        reader.source_path(current, canonical["sha256"])


def test_opaque_content_remains_denied_through_alias_and_canonical(staged, monkeypatch):
    canonical = staged.add("opaque", r"C:\archive\opaque.json", b'{"never_parse":true}')
    alias = r"C:\old\result.json"
    request = {**canonical, "path": alias}
    reader = staged.reader(logical_aliases={alias: canonical}, opaque_descriptors=[request])
    monkeypatch.setattr(reader.io, "read_json", lambda *a, **k: pytest.fail("opaque was parsed"))
    for bound in (request, canonical):
        assert reader.path(bound) == staged.cache / "opaque"
        with pytest.raises(ValueError, match="Opaque"): reader.json(bound)
        with pytest.raises(ValueError, match="Opaque"): reader.array(bound)


@pytest.mark.parametrize("mutation", ["sha", "size", "bool_size", "unknown"])
def test_requests_fail_closed(staged, mutation):
    bound = staged.add("a", r"\\NAS\share\file", b"tiny")
    reader = staged.reader()
    wrong = dict(bound)
    if mutation == "sha": wrong["sha256"] = "f" * 64
    if mutation == "size": wrong["size_bytes"] += 1
    if mutation == "bool_size": wrong["size_bytes"] = True
    if mutation == "unknown": wrong["path"] += "unknown"
    with pytest.raises(ValueError): reader.path(wrong)


def test_normalized_locator_and_changed_cached_bytes(staged):
    bound = staged.add("a", r"\\NAS\share\folder\file", b"tiny")
    reader = staged.reader()
    assert reader.resolve_path(r"//nas/SHARE/folder/../folder/file", bound["sha256"]) == staged.cache / "a"
    reader.path(bound)
    (staged.cache / "a").write_bytes(b"corrupt")
    with pytest.raises(ValueError): reader.path(bound)


def test_all_inputs_staged_and_confined(staged):
    bound = staged.add("a", "original", b"a")
    staged.local.clear()
    with pytest.raises(ValueError, match="staged"): staged.reader()
    staged.local["a"] = {**bound, "path": str(staged.worktree / "elsewhere")}
    with pytest.raises(ValueError, match="escaped"): staged.reader()


def test_startup_git_identity_is_mandatory(staged, monkeypatch):
    staged.add("a", "original", b"a")
    monkeypatch.setattr(inputs, "_code_identity", lambda root: dict(staged.manifest["code"], commit="f" * 40))
    with pytest.raises(ValueError, match="Git identity"): staged.reader()
    staged.manifest["code"]["clean_tree"] = False
    with pytest.raises(ValueError, match="code commit"): staged.reader()


def test_opaque_hash_only_before_decoder(staged, monkeypatch):
    bound = staged.add("opaque", "result.json", b'{"must_not_parse":true}')
    reader = staged.reader(opaque_descriptors=[bound])
    assert reader.path(bound).is_file()
    monkeypatch.setattr(reader.io, "read_json", lambda *a, **k: pytest.fail("opaque parsed"))
    with pytest.raises(ValueError, match="Opaque"): reader.json(bound)
    with pytest.raises(ValueError, match="Opaque"): reader.array(bound)


def source_review(staged, purpose="file_handling_acceptance"):
    source = staged.worktree / "implementation.py"
    source.write_bytes(b"# current IO wiring\n")
    frozen = staged.add("frozen_source", "frozen/original.py", b"# frozen evidence\n")
    mappings = {str(source): frozen}
    bindings = [dict(current_source=pio.NasIO().asset(source), original_source=frozen)]
    staged.manifest["purpose"] = purpose
    staged.manifest["scientific_input_roles"] = list(staged.manifest["inputs"])
    staged.manifest["assurance_input_roles"] = []
    proof = dict(contract="darsia-production-io-source-review-v1", complete=True,
                 io_only_diff_passed=True, purpose_scope="exact_CPU_parity_bootstrap",
                 execution_commit=staged.manifest["code"]["commit"], source_bindings=bindings,
                 input_bindings=deepcopy(staged.manifest["inputs"]),
                 reference_settings=dict(backend="cpu", precision="float64"))
    def save(value, name="review.json"):
        path = staged.worktree / name
        pio.NasIO().write_json(path, value)
        descriptor = staged.add(name, str(path), path.read_bytes())
        staged.manifest["assurance_input_roles"].append(name)
        return descriptor
    return source, frozen, mappings, proof, save


def test_source_substitution_requires_bound_review(staged):
    source, frozen, mapping, proof, save = source_review(staged)
    with pytest.raises(ValueError): staged.reader(source_provenance=mapping)
    reader = staged.reader(source_provenance=mapping, source_qualification=dict(source_review=save(proof)))
    assert reader.source_path(source, frozen["sha256"]) == staged.cache / "frozen_source"
    assert reader.source_qualification_state == "exact_CPU_parity_bootstrap"
    assert reader.source_bindings[0]["current_source"]["sha256"] != frozen["sha256"]
    source.write_bytes(b"# changed runtime\n")
    with pytest.raises(ValueError): reader.source_path(source, frozen["sha256"])


def test_assurance_alias_cannot_enter_array_inputs(staged):
    _, _, mapping, proof, save = source_review(staged)
    review = save(proof)
    alias = r"C:\old\source_review.json"
    reader = staged.reader(source_provenance=mapping, source_qualification=dict(source_review=review),
                           logical_aliases={alias: review})
    with pytest.raises(ValueError, match="Assurance metadata"):
        reader.array({**review, "path": alias})


@pytest.mark.parametrize("kind", ["missing_review", "unknown_source", "wrong_evidence", "no_io_review", "different_inputs", "not_cpu", "scope", "prod_permission"])
def test_invalid_source_qualification(staged, kind):
    source, frozen, mapping, proof, save = source_review(staged)
    if kind == "unknown_source":
        other = staged.worktree / "unknown.py"
        other.write_bytes(b"unknown")
        mapping = {str(other): frozen}
    if kind == "wrong_evidence": frozen["sha256"] = "f" * 64
    if kind == "no_io_review": proof["io_only_diff_passed"] = False
    if kind == "different_inputs": proof["input_bindings"] = {}
    if kind == "not_cpu": staged.manifest["settings"]["backend"] = "cuda"
    if kind == "scope": proof["purpose_scope"] = "profile"
    if kind == "prod_permission": staged.manifest["production_permission_granted"] = True
    descriptor = None if kind == "missing_review" else dict(source_review=save(proof))
    with pytest.raises(ValueError): staged.reader(source_provenance=mapping, source_qualification=descriptor)


def test_profile_requires_actual_exact_cpu_parity(staged):
    source, frozen, mapping, proof, save = source_review(staged, "profile")
    with pytest.raises(ValueError): staged.reader(source_provenance=mapping, source_qualification=dict(source_review=save(proof)))
    parity = {**proof, "contract": "darsia-production-exact-cpu-parity-v1", "passed": True}
    parity_desc = save(parity, "parity.json")
    reader = staged.reader(source_provenance=mapping, source_qualification=dict(
        source_review=save(proof, "review2.json"), exact_cpu_parity_result=parity_desc))
    assert reader.source_qualification_state == "exact_CPU_parity_verified"
    assert reader.source_path(source, frozen["sha256"]) == staged.cache / "frozen_source"


@pytest.mark.parametrize("changed", ["passed", "complete", "source_bindings", "input_bindings", "execution_commit", "reference_settings"])
def test_parity_must_match_exact_source_input_precision(staged, changed):
    _, _, mapping, proof, save = source_review(staged, "profile")
    parity = {**proof, "contract": "darsia-production-exact-cpu-parity-v1", "passed": True}
    parity[changed] = False if changed in {"passed", "complete"} else {}
    parity_desc = save(parity, "parity.json")
    with pytest.raises(ValueError): staged.reader(source_provenance=mapping, source_qualification=dict(
        source_review=save(proof), exact_cpu_parity_result=parity_desc))


def test_verified_reader_delegates_without_opening_original(staged):
    bound = staged.add("a", r"\\NAS\share\a.json", b'{"synthetic":true}')
    reader = staged.reader()
    namespace = extract("phase_ffac_common_mass_inputs.py", {"VerifiedAssetReader"}, {"Path": Path})
    wrapped = namespace["VerifiedAssetReader"](asset_reader=reader)
    assert wrapped.path(bound) == staged.cache / "a"
    assert wrapped.json(bound) == {"synthetic": True}


def test_original_round3_content_guards_use_local_relative_binding(staged):
    calibration = staged.add("parameters", r"\\NAS\share\freeze\parameters.json", b'{"frozen":true}')
    value = dict(contract="B1", frozen=True, calibration_runs=["fit"], heldout_runs=["test"],
                 round2_freeze_sha256="B0", parameters=dict(candidate="N1"),
                 map_mass_rescaling=False, overflow_in_objective=False,
                 calibration_files={"parameters.json": calibration["sha256"]}, frozen_utc="exact")
    freeze = staged.add("freeze", r"\\NAS\share\freeze\freeze.json", json.dumps(value).encode())
    reader = staged.reader()
    calls = []
    r1 = SimpleNamespace(sha256=lambda path: pio.NasIO().asset(path)["sha256"])
    ns2 = extract("phase_ffac_common_mass_round2.py", {"_verify_file"}, dict(Path=Path, r1=r1, _VERIFIED_SOURCE_FILES={}))
    r2 = SimpleNamespace(_verify_file=ns2["_verify_file"], _validate_parameters=lambda p: calls.append(p))
    ns = extract("phase_ffac_common_mass_round3.py", {"verify_round3_freeze"},
                 dict(Path=Path, json=json, r1=r1, r2=r2, FREEZE_CONTRACT="B1", FIT_RUNS=["fit"], HELDOUT_RUNS=["test"], ROUND2_FREEZE_SHA256="B0"))
    result, digest = ns["verify_round3_freeze"](freeze["path"], asset_reader=reader)
    assert result == value and digest == freeze["sha256"] and calls == [value["parameters"]]
    (staged.cache / "parameters").write_bytes(b"changed")
    with pytest.raises(ValueError): ns["verify_round3_freeze"](freeze["path"], asset_reader=reader)


def test_native_model_constructor_opens_only_manifest_cache(staged):
    # This executes the original IO-hook constructor with entirely fake scalar/
    # array-like objects. No numpy, image decoding or native calculation occurs.
    class Check:
        def __init__(self, value=True): self.value = value
        def all(self): return self.value
        def any(self): return self.value
    class Fake:
        shape = (1, 1)
        ndim = 2
        dtype = "integer"
        def any(self): return True
        def copy(self): return Fake()
        def __getitem__(self, item): return self
        def __le__(self, other): return Check(False)
        def __gt__(self, other): return Check(False)
        def __and__(self, other): return Fake()
    class NP:
        float32 = "float32"
        integer = "integer"
        @staticmethod
        def asarray(*args): return Fake()
        @staticmethod
        def broadcast_to(*args): return Fake()
        @staticmethod
        def isfinite(*args): return Check()
        @staticmethod
        def unique(*args): return [1]
        @staticmethod
        def issubdtype(*args): return True
        @staticmethod
        def where(*args): return Fake()
        @staticmethod
        def isin(*args): return Fake()
        @staticmethod
        def prod(*args): return 1
        @staticmethod
        def any(value): return value.any()
    folder = Path(r"\\NAS\share\models")
    metadata = dict(contract="R1", basis="facies", label_ids=[1], ignore_labels=[])
    logical_files = {
        "metadata.json": json.dumps(metadata).encode(),
        "color_path_interpretation/color_path_interpretation_1.json": b'{"colors":"unchanged"}',
        "signal_model/signal_model_1.csv": b"supports,values\n0,0\n1,1\n",
        "flash/flash.json": b'{"bounds":"unchanged"}',
    }
    for relative, data in logical_files.items(): staged.add(relative, str(folder / relative), data)
    reader = staged.reader()
    opened = []
    def loader(suffix):
        def load(path):
            file = path.with_suffix(suffix)
            opened.append(file)
            assert file.resolve().is_relative_to(staged.cache)
            return file.read_bytes()
        return SimpleNamespace(load=load)
    darsia = SimpleNamespace(ExtrudedPorousGeometry=lambda **k: SimpleNamespace(voxel_volume=Fake()),
                            ScalarImage=lambda *a, **k: object(), ColorMode=SimpleNamespace(RELATIVE="relative"),
                            ColorPathInterpolation=loader(".json"), PWTransformation=loader(".csv"), SimpleFlash=loader(".json"))
    ns = extract("phase_ffac_common_mass.py", {"UnclippedNativeConcentrationAnalysis"},
                 dict(Path=Path, json=json, np=NP, darsia=darsia, NativeConcentrationAnalysis=object,
                      ColorEmbeddingBasis=lambda v: SimpleNamespace(value=v), CONTRACT="R1",
                      sha256=lambda p: pio.NasIO().asset(p)["sha256"],
                      HeterogeneousColorToMassAnalysis=lambda **k: SimpleNamespace(**k)))
    cls = ns["UnclippedNativeConcentrationAnalysis"]
    cls._rgb = lambda self, value: object()  # Fake-only inherited image creation.
    analysis = cls(folder, baseline_rgb=None, labels=None, domain=None, depth_m=None,
                   porosity=None, dimensions_m=(1, 1), asset_reader=reader)
    assert len(opened) == 3
    assert set(analysis.source_hashes) == {str(folder / name) for name in logical_files}
    assert analysis.folder == folder.resolve()  # Original provenance stays logical.
    assert analysis.source_hashes == {str(folder / name): hashlib.sha256(data).hexdigest()
                                      for name, data in logical_files.items()}
    missing = staged.reader()
    missing._bindings.pop(pio.normalize_locator(folder / "flash/flash.json"))
    with pytest.raises(ValueError, match="absent"):
        cls(folder, baseline_rgb=None, labels=None, domain=None, depth_m=None,
            porosity=None, dimensions_m=(1, 1), asset_reader=missing)


def test_round1_frozen_source_evidence_is_not_execution_source(staged):
    current, original_source, mapping, review, save = source_review(staged)
    model = staged.add("model", r"\\NAS\share\models\metadata.json", b'{"unchanged":true}')
    native_source = staged.worktree / "native_unchanged.py"
    native_source.write_bytes(b"# unchanged native implementation\n")
    native_bound = staged.add("native", str(native_source), native_source.read_bytes())
    names = ("ColorPath", "ColorPathInterpolation", "ConcentrationAnalysis", "PWTransformation",
             "TitrationFlash", "SimpleFlash", "CO2MassAnalysis", "ExtrudedPorousGeometry", "NativeConcentrationAnalysis")
    objects = {name: type(name, (), {}) for name in names}
    value = dict(contract="R1", frozen=True, calibration_runs=["fit"], heldout_runs=["test"],
                 map_mass_rescaling=False, overflow_in_objective=False,
                 calibration_files={model["path"]: model["sha256"]}, common_engine_sha256=original_source["sha256"],
                 native_implementations={name: native_bound["sha256"] for name in names},
                 native_implementation_files={"producer/native.py": native_bound["sha256"]})
    freeze = staged.add("r1_freeze", r"\\NAS\share\freeze.json", json.dumps(value).encode())
    staged.manifest["scientific_input_roles"] = list(staged.manifest["inputs"])
    review["input_bindings"] = deepcopy(staged.manifest["inputs"])
    reader = staged.reader(source_provenance=mapping, source_qualification=dict(source_review=save(review)))
    cohorts = []
    r1 = SimpleNamespace(__file__=str(current), CONTRACT="R1", NativeConcentrationAnalysis=objects[names[-1]],
                         sha256=lambda p: pio.NasIO().asset(p)["sha256"],
                         validate_cohort=lambda *c: cohorts.append(c))
    ns = extract("phase_ffac_common_mass_round2.py", {"_verify_file", "verify_original_round1_freeze"},
                 dict(Path=Path, json=json, r1=r1, darsia=SimpleNamespace(**objects),
                      inspect=SimpleNamespace(getfile=lambda obj: str(native_source)),
                      ROUND1_FREEZE_SHA256=freeze["sha256"], _VERIFIED_SOURCE_FILES={}, _VERIFIED_ORIGINAL_FREEZES={}))
    record, digest = ns["verify_original_round1_freeze"](freeze["path"], asset_reader=reader)
    provenance = record["production_io_source_provenance"]
    assert digest == freeze["sha256"] and cohorts == [(["fit"], ["test"])]
    assert provenance["current_common_engine"]["sha256"] == pio.NasIO().asset(current)["sha256"]
    assert provenance["original_calibration_source_sha256"] == original_source["sha256"]
    assert provenance["current_common_engine"]["sha256"] != provenance["original_calibration_source_sha256"]
    assert provenance["frozen_evidence_is_not_current_executed_source"] is True
    assert record["cross_host_source_equivalence"]["common_engine_current_path"] == str(staged.cache / "frozen_source")
    (staged.cache / "frozen_source").write_bytes(b"changed evidence")
    with pytest.raises(ValueError): ns["verify_original_round1_freeze"](freeze["path"], asset_reader=reader)


def test_default_freeze_api_matches_original_without_reader(tmp_path):
    calibration = tmp_path / "parameters.json"
    calibration.write_bytes(b"tiny frozen calibration")
    value = dict(contract="B1", frozen=True, calibration_runs=["fit"], heldout_runs=["test"],
                 round2_freeze_sha256="B0", parameters=dict(candidate="N1"),
                 map_mass_rescaling=False, overflow_in_objective=False,
                 calibration_files={str(calibration): pio.NasIO().asset(calibration)["sha256"]}, frozen_utc="exact")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(value))
    r1 = SimpleNamespace(sha256=lambda p: pio.NasIO().asset(p)["sha256"])
    r2 = SimpleNamespace(_validate_parameters=lambda p: None,
                         _verify_file=lambda p, sha: pio.NasIO().verify(dict(path=str(p), sha256=sha)))
    namespace = dict(Path=Path, json=json, r1=r1, r2=r2, FREEZE_CONTRACT="B1",
                     FIT_RUNS=["fit"], HELDOUT_RUNS=["test"], ROUND2_FREEZE_SHA256="B0")
    old = subprocess.run(["git", "show", "5d2f2b94:src/darsia/fluidflower/phase_ffac_common_mass_round3.py"], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    node = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == "verify_round3_freeze")
    before = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "original metadata verifier", "exec"), before)
    after = extract("phase_ffac_common_mass_round3.py", {"verify_round3_freeze"}, dict(namespace))
    assert before["verify_round3_freeze"](freeze) == after["verify_round3_freeze"](freeze)
    value["map_mass_rescaling"] = True
    freeze.write_text(json.dumps(value))
    for ns in (before, after):
        with pytest.raises(ValueError, match="scientific contract"): ns["verify_round3_freeze"](freeze)


def test_native_function_scope_and_compile_only():
    allowed = {
        "phase_ffac_common_mass_inputs.py": {"VerifiedAssetReader"},
        "phase_ffac_common_mass.py": {"UnclippedNativeConcentrationAnalysis", "prepare_native_response", "verify_frozen_calibration"},
        "phase_ffac_common_mass_round2.py": {"verify_round2_freeze", "_verify_file", "verify_original_round1_freeze", "prepare_round2_response"},
        "phase_ffac_common_mass_round3.py": {"prepare_own_colour_response", "_authorize_prediction", "evaluate_round3_summary", "evaluate_round3_compiled", "verify_round3_freeze"},
        "tools/ffac_round3_selected_adapter.py": {"_admit", "_prepare_response", "load_selected"},
    }
    for relative, changed in allowed.items():
        path = SOURCES / relative
        old = subprocess.run(["git", "show", "5d2f2b94:src/darsia/fluidflower/" + relative], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
        current = path.read_text(encoding="utf-8")
        before, after = ast.parse(old), ast.parse(current)
        compile(current, str(path), "exec")  # No numerical imports or module execution.
        first = {n.name: ast.dump(n, include_attributes=False) for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        second = {n.name: ast.dump(n, include_attributes=False) for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert first.keys() == second.keys()
        assert {name for name in first if first[name] != second[name]} == changed
        # Neither constants nor imports change in any frozen scientific module.
        assert [ast.dump(n) for n in before.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))] == [ast.dump(n) for n in after.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        if relative == "phase_ffac_common_mass.py":
            first_cls = next(n for n in before.body if isinstance(n, ast.ClassDef) and n.name == "UnclippedNativeConcentrationAnalysis")
            second_cls = next(n for n in after.body if isinstance(n, ast.ClassDef) and n.name == first_cls.name)
            assert ast.dump(first_cls.body[-1]) == ast.dump(second_cls.body[-1])  # _rgb untouched.


def test_b1_prediction_math_and_admission_conditions_unchanged():
    relative = "phase_ffac_common_mass_round3.py"
    old = subprocess.run(["git", "show", "5d2f2b94:src/darsia/fluidflower/" + relative], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    before = {n.name: n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: n for n in ast.parse((SOURCES / relative).read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
    for name in ("evaluate_round3_summary", "evaluate_round3_compiled"):
        def without_verifier(function):
            return [ast.dump(n) for n in function.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_authorize_prediction")]
        assert without_verifier(before[name]) == without_verifier(after[name])
    for name in ("verify_round3_freeze", "_authorize_prediction"):
        assert [ast.dump(n) for n in ast.walk(before[name]) if isinstance(n, ast.Raise)] == [ast.dump(n) for n in ast.walk(after[name]) if isinstance(n, ast.Raise)]


def test_own_colour_optional_resolver_forwarding_and_unchanged_math():
    class Fake:
        shape = (1, 1)
        def __and__(self, other): return Fake()
        def __gt__(self, other): return Fake()
        def __invert__(self): return Fake()
        def __getitem__(self, item): return Fake()
        def all(self, *args, **kwargs): return True
        def sum(self): return 0
    class NP:
        float32 = "float32"
        @staticmethod
        def asarray(*args): return Fake()
        @staticmethod
        def isfinite(*args): return Fake()
        @staticmethod
        def zeros_like(*args): return Fake()
        @staticmethod
        def divide(*args, **kwargs): return None
        @staticmethod
        def where(*args): return Fake()
        @staticmethod
        def isin(*args): return Fake()
    calls = []
    analysis = SimpleNamespace(_rgb=lambda rgb: rgb,
        pipeline=SimpleNamespace(call_color_interpretation=lambda rgb: rgb, call_pH_analysis=lambda image: image))
    def constructor(folder, **kwargs):
        calls.append((folder, kwargs))
        return analysis
    ns = extract("phase_ffac_common_mass_round3.py", {"prepare_own_colour_response"},
                 dict(np=NP, r1=SimpleNamespace(UnclippedNativeConcentrationAnalysis=constructor)))
    frame = SimpleNamespace(domain=Fake(), clear_water_rgb=Fake(), clear_water_supported_mask=Fake(),
                            current_rgb=Fake(), baseline_rgb=Fake(), facies=Fake(), dimensions_m=(.49, .90))
    geometry = SimpleNamespace(valid=Fake(), depth_m=Fake(), porosity=Fake())
    response = dict(frame=frame, flash=lambda signal: (SimpleNamespace(img=signal), None), metadata=dict(ignore_labels=[]))
    default = ns["prepare_own_colour_response"](response, "logical/frozen/models", geometry, photo_support=Fake())
    reader = object()
    forwarded = ns["prepare_own_colour_response"](response, "logical/frozen/models", geometry,
                                                  photo_support=Fake(), asset_reader=reader)
    assert "asset_reader" not in calls[0][1]
    assert calls[1][1]["asset_reader"] is reader
    assert calls[0][0] == calls[1][0] == "logical/frozen/models"
    assert default["audit"] == forwarded["audit"]
    old = subprocess.run(["git", "show", "5d2f2b94:src/darsia/fluidflower/phase_ffac_common_mass_round3.py"], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    before = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == "prepare_own_colour_response")
    after = next(n for n in ast.parse((SOURCES / "phase_ffac_common_mass_round3.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == before.name)
    after.args.kwonlyargs.pop()
    after.args.kw_defaults.pop()
    call = next(n.value for n in after.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "analysis")
    assert call.keywords[-1].arg is None
    call.keywords.pop()
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)


def test_r1_and_selected_math_statements_are_literal_ast_identical():
    scopes = {
        "phase_ffac_common_mass.py": {"prepare_native_response": {"freeze", "metadata", "metadata_path", "analysis"}},
        "tools/ffac_round3_selected_adapter.py": {"load_selected": {"reader", "asset_reader", "receipt", "original", "response"}},
    }
    for relative, functions in scopes.items():
        old = subprocess.run(["git", "show", "5d2f2b94:src/darsia/fluidflower/" + relative], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
        before = {n.name: n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in ast.parse((SOURCES / relative).read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        for name, allowed in functions.items():
            def math_statements(function):
                result = []
                for statement in function.body:
                    # Keep all numerical statements, removing only known IO
                    # assignments and the verified-reader setup/held-out gate.
                    if isinstance(statement, ast.Assign):
                        targets = {n.id for target in statement.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
                        if targets & allowed: continue
                    if isinstance(statement, ast.If):
                        if name == "prepare_native_response" and ast.unparse(statement.test) == "run in HELDOUT_RUNS": continue
                        if name == "load_selected" and "asset_reader" in ast.unparse(statement.test): continue
                    result.append(ast.dump(statement))
                return result
            assert math_statements(before[name]) == math_statements(after[name])

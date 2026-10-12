"""Portable launcher tests: synthetic files/transport, no SSH or analysis."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(stem):
    name = "standalone_production_launch_test_" + stem
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/darsia/fluidflower" / (stem + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = load("production_launch")
io_module = load("production_io")


@pytest.fixture
def endpoint(tmp_path):
    hosts = tmp_path / "known hosts"
    hosts.write_text("10.10.10.146 ssh-ed25519 " + m.OLAV_HOST_KEY + "\n")
    return m.Endpoint(known_hosts=str(hosts), ssh="ssh synthetic", scp="scp synthetic")


class FakeTransport(m.RemoteLauncher):
    def __init__(self, endpoint):
        super().__init__(endpoint=endpoint)
        self.files = {}
        self.calls = []

    def _python(self, code, *args, timeout=60):
        self.calls.append((code, args))
        if code == m._INSPECT:
            if args[0] not in self.files:
                return "ABSENT"
            raw = self.files[args[0]]
            return hashlib.sha256(raw).hexdigest() + " " + str(len(raw))
        if code == m._MKDIR:
            return ""
        if code == m._PUBLISH:
            source, dest, sha, size = args
            raw = self.files[source]
            assert hashlib.sha256(raw).hexdigest() == sha and len(raw) == size
            if dest in self.files:
                if self.files[dest] != raw:
                    raise FileExistsError("Immutable conflict")
            else:
                self.files[dest] = self.files.pop(source)
            return ""
        raise AssertionError("No remote execution in this fixture")

    def _run(self, argv, timeout=60):
        self.calls.append((argv, timeout))
        if argv[0] != self.endpoint.scp:
            raise AssertionError("No real SSH")
        source, dest = argv[-2:]
        prefix = self.endpoint.target + ":"
        if dest.startswith(prefix):
            remote = dest[len(prefix):]
            assert remote not in self.files
            self.files[remote] = Path(source).read_bytes()
        else:
            Path(dest).write_bytes(self.files[source[len(prefix):]])
        return ""


@pytest.fixture
def transport(endpoint):
    return FakeTransport(endpoint)


def file_asset(tmp_path, name="script with space.py", content=b"# Own synthetic metadata script\n"):
    path = tmp_path / name
    path.write_bytes(content)
    return m._asset(path)


def packet():
    prefix = "C:/FFAC_cache/production/worktrees/" + "a" * 40
    return dict(purpose="profile", worktree=prefix,
        manifest=dict(path="C:/FFAC_cache/production/manifests/own.json", sha256="b" * 64, size_bytes=123),
        sources={name: dict(path=prefix + "/src/darsia/fluidflower/" + name,
                           sha256="c" * 64, size_bytes=100) for name in ("production_launch.py", "production_io.py", "production_job.py")})


def test_short_arguments_literal_spaces_and_apostrophe(endpoint):
    args = [endpoint.python, "-I", "C:/FFAC_cache/Own Folder/O'Neil.py", "--manifest", "C:/FFAC_cache/Own Folder/input.json"]
    argv = m.ssh_argv(endpoint, args)
    tokens = shlex.split(argv[-1])
    assert tokens[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert tokens[4] == "& " + " ".join(m._ps_quote(arg) for arg in args)
    assert "O''Neil.py" in tokens[4]
    assert "StrictHostKeyChecking=yes" in argv
    assert "HostKeyAlgorithms=ssh-ed25519" in argv
    assert "EncodedCommand" not in " ".join(argv)
    assert len(argv[-1]) <= m.MAX_COMMAND


def test_long_command_and_unreviewed_endpoint_rejected(endpoint):
    with pytest.raises(ValueError):
        m.ssh_argv(endpoint, ["x" * 4096])
    with pytest.raises(ValueError):
        m.ssh_argv(m.Endpoint(target="other@host"), ["python"])


@pytest.mark.parametrize("value", ["relative.py", "D:/outside.py", "C:/Users/Claude/file.py", "C:/FFAC_cache/../x.py", "C:/FFAC_cache/a%PATH%.py", "C:/FFAC_cache/x\n.py"])
def test_remote_locator_confined(value):
    with pytest.raises(ValueError):
        m._remote_path(value)


def test_existing_reviewed_host_key_required(endpoint, tmp_path):
    m._known_host(endpoint)
    Path(endpoint.known_hosts).write_text("10.10.10.146 ssh-ed25519 wrong\n")
    with pytest.raises(ValueError):
        m._known_host(endpoint)


@pytest.mark.parametrize("change", [dict(host_key="other"), dict(python_sha256="f" * 64), dict(python_size_bytes=True), dict(python_size_bytes=1)])
def test_endpoint_cannot_replace_host_or_interpreter_identity(endpoint, change):
    from dataclasses import replace
    with pytest.raises(ValueError):
        m._known_host(replace(endpoint, **change))


def test_stage_once_and_reuse_never_scp_final(transport, tmp_path):
    asset = file_asset(tmp_path)
    dest = "C:/FFAC_cache/Own Folder/script.py"
    first = transport.stage_once(asset, dest)
    assert first["reused"] is False
    calls = [c for c in transport.calls if isinstance(c[0], list)]
    assert len(calls) == 1 and calls[0][0][-1].endswith(".partial")
    assert transport.files[dest] == Path(asset["path"]).read_bytes()
    second = transport.stage_once(asset, dest)
    assert second["reused"] is True
    assert len([c for c in transport.calls if isinstance(c[0], list)]) == 1


def test_remote_conflict_preserved_before_scp(transport, tmp_path):
    asset = file_asset(tmp_path)
    dest = "C:/FFAC_cache/Own Folder/script.py"
    transport.files[dest] = b"previous unrelated bytes"
    with pytest.raises(FileExistsError):
        transport.stage_once(asset, dest)
    assert transport.files[dest] == b"previous unrelated bytes"
    assert not any(isinstance(c[0], list) for c in transport.calls)


@pytest.mark.parametrize("delta", [dict(sha256="f" * 64), dict(size_bytes=True), dict(size_bytes=999)])
def test_wrong_script_bytes_or_integer_size_fail_before_remote(transport, tmp_path, delta):
    asset = file_asset(tmp_path)
    with pytest.raises(ValueError):
        transport.stage_once({**asset, **delta}, "C:/FFAC_cache/script.py")
    assert transport.calls == []


def test_remote_python_and_script_sha_fence_in_every_helper_command(endpoint):
    launcher = m.RemoteLauncher(endpoint=endpoint)
    calls = []
    with mock.patch.object(launcher, "_run", side_effect=lambda argv, **kw: calls.append(argv) or ""):
        launcher._python(m._INSPECT, "C:/FFAC_cache/own.json")
        argv = launcher._command(packet(), "remote-self-test", "C:/FFAC_cache/diagnostics/own.json")
    for argv in [calls[0], argv]:
        assert endpoint.python_sha256 in argv[-1]
        assert str(endpoint.python_size_bytes) in argv[-1]
        assert len(argv[-1]) <= m.MAX_COMMAND
    assert "c" * 64 in argv[-1]


def test_fetch_copied_file_verified_and_reused_without_overwrite(transport, tmp_path):
    raw = b"OWN synthetic returned file; no arrays"
    remote = "C:/FFAC_cache/results/own.json"
    transport.files[remote] = raw
    descriptor = dict(path=remote, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    target = tmp_path / "returned receipt.json"
    io = io_module.NasIO()
    result = transport.fetch_file(descriptor, target, io=io)
    assert target.read_bytes() == raw and result["sha256"] == descriptor["sha256"]
    before = len(transport.calls)
    transport.fetch_file(descriptor, target, io=io)
    assert len(transport.calls) == before + 1  # Only remote SHA metadata check.
    target.write_bytes(b"conflicting local output")
    with pytest.raises(ValueError):
        transport.fetch_file(descriptor, target, io=io)
    assert target.read_bytes() == b"conflicting local output"


def test_wrong_remote_result_binding_rejects_before_download(transport, tmp_path):
    remote = "C:/FFAC_cache/results/own.json"
    transport.files[remote] = b"other"
    with pytest.raises(ValueError):
        transport.fetch_file(dict(path=remote, sha256="f" * 64, size_bytes=5), tmp_path / "result.json", io=io_module.NasIO())
    assert not any(isinstance(c[0], list) for c in transport.calls)


def test_result_size_bool_is_not_integer_byte_binding(transport, tmp_path):
    with pytest.raises(ValueError):
        transport.fetch_file(dict(path="C:/FFAC_cache/own.json", sha256="f" * 64, size_bytes=True), tmp_path / "own.json", io=io_module.NasIO())
    assert transport.calls == []


@pytest.mark.parametrize("guard", [None, False, lambda: False])
def test_no_root_guard_no_start(endpoint, guard):
    launcher = m.RemoteLauncher(endpoint=endpoint)
    with mock.patch.object(launcher, "_run") as run:
        with pytest.raises(PermissionError):
            launcher.launch(packet(), {}, "C:/FFAC_cache/results/own.json", before_launch=guard)
        run.assert_not_called()


def test_root_guard_once_and_no_blind_job_retry(endpoint):
    launcher = m.RemoteLauncher(endpoint=endpoint)
    guard = mock.Mock(return_value=True)
    entry = dict(path=packet()["worktree"] + "/tools/own.py", sha256="d" * 64, size_bytes=22)
    with mock.patch.object(launcher, "_run", side_effect=RuntimeError("Connection ended")) as run:
        with pytest.raises(RuntimeError):
            launcher.launch(packet(), entry, "C:/FFAC_cache/results/own.json", before_launch=guard)
        assert run.call_count == guard.call_count == 1
        assert run.call_args.kwargs["timeout"] is None


def test_absent_completion_metadata_is_not_returned_as_success(endpoint):
    launcher = m.RemoteLauncher(endpoint=endpoint)
    entry = dict(path=packet()["worktree"] + "/tools/own.py", sha256="d" * 64, size_bytes=22)
    with mock.patch.object(launcher, "_run") as run, mock.patch.object(launcher, "remote_asset", return_value=None):
        with pytest.raises(RuntimeError):
            launcher.launch(packet(), entry, "C:/FFAC_cache/results/own.json", before_launch=lambda: True)
        assert run.call_count == 1


def test_guard_exception_never_becomes_io_retry(endpoint):
    launcher = m.RemoteLauncher(endpoint=endpoint)
    def stopped():
        raise RuntimeError("Root STOP")
    with mock.patch.object(launcher, "_run") as run:
        with pytest.raises(RuntimeError):
            launcher.launch(packet(), {}, "C:/FFAC_cache/results/own.json", before_launch=stopped)
        run.assert_not_called()


def test_production_flags_and_truthy_root_boolean_cannot_grant_go(endpoint):
    p = packet()
    p.update(purpose="production", production_go=True, production_permission_granted=True)
    launcher = m.RemoteLauncher(endpoint=endpoint)
    with mock.patch.object(launcher, "_run") as run:
        with pytest.raises(PermissionError):
            launcher.self_test(p, "C:/FFAC_cache/results/own.json")
        with pytest.raises(PermissionError):
            launcher.launch(p, {}, "C:/FFAC_cache/results/own.json", before_launch=lambda: True)
        run.assert_not_called()


def test_command_and_output_caps_without_network(endpoint):
    response = SimpleNamespace(stdout=b"x" * (m.MAX_STDOUT + 1), stderr=b"")
    launcher = m.RemoteLauncher(endpoint=endpoint, runner=mock.Mock(return_value=response))
    with pytest.raises(ValueError):
        launcher._run(["synthetic only"])
    assert launcher.runner.call_args.kwargs["shell"] is False


def test_bootstrap_sha_before_source_execution_and_no_overwrite(tmp_path):
    executable = m._asset(sys.executable)
    source = tmp_path / "owned script.py"
    marker = tmp_path / "must not execute"
    source.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('OWN')\n")
    argv = [sys.executable, "-I", "-B", "-c", m._EXECUTE,
            str(executable["size_bytes"]), executable["sha256"], str(source), "f" * 64, str(source.stat().st_size)]
    result = subprocess.run(argv, capture_output=True, timeout=10)
    assert result.returncode != 0 and not marker.exists()
    before = tmp_path / "owned partial"
    after = tmp_path / "existing final"
    before.write_bytes(b"OWN new");after.write_bytes(b"OWN existing")
    expected = m._asset(before)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", m._PUBLISH,
        str(executable["size_bytes"]), executable["sha256"], str(before), str(after),
        expected["sha256"], str(expected["size_bytes"])], capture_output=True, timeout=10)
    assert result.returncode != 0 and before.read_bytes() == b"OWN new" and after.read_bytes() == b"OWN existing"


def test_short_bash_powershell_argument_roundtrip_with_spaces(endpoint, tmp_path):
    if os.name != "nt":
        pytest.skip("Windows Bash/PowerShell roundtrip; portable quoting checks run elsewhere")
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not bash.is_file():
        pytest.skip("Git Bash is absent; portable quoting checks still run")
    executable = sys.executable
    values = ["C:/FFAC_cache/a space/file.json", "O'Neil", "--literal", "a b", "$HOME;`echo unsafe`", "a&b"]
    code = "import json,sys;print(json.dumps(sys.argv[1:]))"
    command = m.ssh_argv(endpoint, [executable, "-I", "-B", "-c", code, *values])[-1]
    result = subprocess.run([str(bash), "-c", command], check=True, capture_output=True, timeout=10)
    assert json.loads(result.stdout) == values


@pytest.mark.parametrize("exists", [True, False])
def test_actual_hash_bootstrap_through_bash_powershell(endpoint, tmp_path, exists):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if os.name != "nt" or not bash.is_file():
        pytest.skip("Windows Git Bash/PowerShell bootstrap")
    executable = m._asset(sys.executable)
    own = file_asset(tmp_path, "O'Neil own metadata.json", b"OWN metadata only")
    if not exists:
        Path(own["path"]).unlink()
    args = [sys.executable, "-I", "-B", "-c", m._INSPECT,
            str(executable["size_bytes"]), executable["sha256"], own["path"]]
    command = m.ssh_argv(endpoint, args)[-1]
    result = subprocess.run([str(bash), "-c", command], check=True, capture_output=True, timeout=10)
    expected = own["sha256"] + " " + str(own["size_bytes"]) if exists else "ABSENT"
    assert result.stdout.decode().strip() == expected


@pytest.mark.parametrize("destination_state", ["absent", "same", "conflicting"])
def test_actual_publication_bootstrap_through_bash_powershell(endpoint, tmp_path, destination_state):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if os.name != "nt" or not bash.is_file():
        pytest.skip("Windows Git Bash/PowerShell bootstrap")
    executable = m._asset(sys.executable)
    source = file_asset(tmp_path, "O'Neil owned partial.json", b"OWN exact metadata")
    target = tmp_path / "O'Neil immutable final.json"
    if destination_state != "absent":
        target.write_bytes(b"OWN exact metadata" if destination_state == "same" else b"OWN conflict")
    args = [sys.executable, "-I", "-B", "-c", m._PUBLISH,
            str(executable["size_bytes"]), executable["sha256"], source["path"], str(target),
            source["sha256"], str(source["size_bytes"])]
    result = subprocess.run([str(bash), "-c", m.ssh_argv(endpoint, args)[-1]], capture_output=True, timeout=10)
    if destination_state == "conflicting":
        assert result.returncode != 0 and target.read_bytes() == b"OWN conflict"
        assert Path(source["path"]).read_bytes() == b"OWN exact metadata"
    else:
        assert result.returncode == 0 and target.read_bytes() == b"OWN exact metadata"
        assert Path(source["path"]).exists() is (destination_state == "same")


def test_small_bootstrap_code_avoids_powershell_native_quote_stripping():
    for code in (m._PIN, m._INSPECT, m._MKDIR, m._PUBLISH, m._EXECUTE):
        assert '"' not in code and "\n" not in code and "\r" not in code


def test_remote_metadata_only_commit_environment_and_precision_receipt(tmp_path):
    io = io_module.NasIO()
    code = dict(commit="a" * 40, clean_tree=True, submodules=[])
    env = dict(python_version="OWN synthetic", pip_freeze="", pip_freeze_sha256=hashlib.sha256(b"").hexdigest())
    settings = dict(host="Olav", precision="float64", backend="cpu", threads=1, workers=1, logical_cpus=24)
    manifest = dict(format="darsia-production-job-v1", purpose="profile", code=code, environment=env, settings=settings)
    path = tmp_path / "own manifest.json"
    path.write_text(json.dumps(manifest))
    asset = io.asset(path)
    args = SimpleNamespace(mode="remote-self-test", worktree=str(tmp_path), io_sha256="c" * 64, io_size=1,
                           job_sha256="d" * 64, job_size=1, manifest=asset["path"], manifest_sha256=asset["sha256"],
                           manifest_size=asset["size_bytes"], purpose="profile", result=str(tmp_path / "own result.json"))
    job = SimpleNamespace(validate_manifest=lambda d: None, code_identity=lambda p: code, environment_lock=lambda: env)
    with mock.patch.object(m, "_helpers", return_value=(io_module, job)), mock.patch.object(m, "_remote_path", side_effect=lambda p: p), \
            mock.patch.object(m.socket, "gethostname", return_value="OLAV"), mock.patch.object(m.subprocess, "Popen") as child:
        record = m._remote(args)
    child.assert_not_called()
    assert record["complete"] and record["scientific_arrays_read"] is False and record["scientific_process_started"] is False
    assert record["settings"]["precision"] == "float64" and record["code"] == code and record["environment"] == env
    assert json.loads(Path(args.result).read_text())["manifest"] == asset


@pytest.mark.parametrize("fault", ["code", "environment", "host", "purpose"])
def test_remote_metadata_mismatches_stop_before_receipt_or_child(tmp_path, fault):
    io = io_module.NasIO();code = dict(commit="a" * 40, clean_tree=True, submodules=[]);env = {"python_version": "OWN"}
    document = dict(purpose="profile", code=code, environment=env, settings=dict(host="Olav", precision="float64"))
    path = tmp_path / "own manifest.json";path.write_text(json.dumps(document));asset = io.asset(path)
    args = SimpleNamespace(mode="remote-self-test", worktree=str(tmp_path), io_sha256="c" * 64, io_size=1,
        job_sha256="d" * 64, job_size=1, manifest=asset["path"], manifest_sha256=asset["sha256"], manifest_size=asset["size_bytes"],
        purpose="equivalence" if fault == "purpose" else "profile", result=str(tmp_path / "never result.json"))
    job = SimpleNamespace(validate_manifest=lambda d: None, code_identity=lambda p: {**code, "commit": "b" * 40} if fault == "code" else code,
                          environment_lock=lambda: {"python_version": "DIFFERENT"} if fault == "environment" else env)
    with mock.patch.object(m, "_helpers", return_value=(io_module, job)), mock.patch.object(m, "_remote_path", side_effect=lambda p: p), \
            mock.patch.object(m.socket, "gethostname", return_value="OTHER" if fault == "host" else "OLAV"), mock.patch.object(m.subprocess, "Popen") as child:
        with pytest.raises((ValueError, PermissionError)):
            m._remote(args)
    child.assert_not_called();assert not Path(args.result).exists()


def test_pinned_helper_mismatch_stops_before_any_module_execution(tmp_path):
    folder = tmp_path / "src/darsia/fluidflower";folder.mkdir(parents=True)
    marker = tmp_path / "source must not execute"
    for name in ("production_io.py", "production_job.py"):
        (folder / name).write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('OWN')\n")
    bindings = {name: {k: v for k, v in m._asset(folder / name).items() if k != "path"}
                for name in ("production_io.py", "production_job.py")}
    bindings["production_job.py"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        m._helpers(tmp_path, bindings)
    assert not marker.exists()


def test_remote_single_claim_child_wait_and_preimport_thread_environment(tmp_path):
    io = io_module.NasIO();code = dict(commit="a" * 40, clean_tree=True, submodules=[]);env = {"python_version": "OWN"}
    settings = dict(host="Olav", precision="float64", backend="cpu", threads=4)
    document = dict(purpose="profile", code=code, environment=env, settings=settings)
    manifest = tmp_path / "own manifest.json";manifest.write_text(json.dumps(document));asset = io.asset(manifest)
    entry = tmp_path / "own entry.py";entry.write_text("# SYNTHETIC, never executed\n");entry_asset = io.asset(entry)
    args = SimpleNamespace(mode="remote-run", worktree=str(tmp_path), io_sha256="c" * 64, io_size=1,
        job_sha256="d" * 64, job_size=1, manifest=asset["path"], manifest_sha256=asset["sha256"], manifest_size=asset["size_bytes"],
        purpose="profile", result=str(tmp_path / "own result.json"), entrypoint=str(entry),
        entrypoint_sha256=entry_asset["sha256"], entrypoint_size=entry_asset["size_bytes"])
    job = SimpleNamespace(validate_manifest=lambda d: None, code_identity=lambda p: code, environment_lock=lambda: env,
                          worker_identity=lambda pid: dict(pid=pid, process_create_time="1.25"))
    child = mock.Mock(pid=7777);child.wait.return_value=0
    with mock.patch.object(m, "_helpers", return_value=(io_module, job)), mock.patch.object(m, "_remote_path", side_effect=lambda p: p), \
            mock.patch.object(m.socket, "gethostname", return_value="OLAV"), mock.patch.object(m.subprocess, "Popen", return_value=child) as start:
        result = m._remote(args)
        assert start.call_count == 1 and child.wait.call_count == 1
        assert all(start.call_args.kwargs["env"][k] == "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
        assert result["child"]["process_create_time"] == "1.25" and result["exit_code"] == 0
        with pytest.raises(FileExistsError):
            m._remote(args)
        assert start.call_count == 1  # Preserved start record blocks duplicate launch.


def test_no_encoded_or_scientific_imports_in_launcher_module():
    import ast
    source = (ROOT / "src/darsia/fluidflower/production_launch.py").read_text()
    imports = {alias.name.split('.')[0] for node in ast.parse(source).body if isinstance(node, ast.Import) for alias in node.names}
    assert not imports & {"numpy", "darsia", "cv2", "torch", "pyopencl", "cupy"}
    assert "EncodedCommand" not in source
    assert not any(isinstance(node, ast.Call) and any(k.arg == "shell" and isinstance(k.value, ast.Constant)
                   and k.value.value is True for k in node.keywords) for node in ast.walk(ast.parse(source)))

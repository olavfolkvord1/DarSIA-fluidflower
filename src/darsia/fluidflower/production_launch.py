"""One short-argument, content-bound SSH launcher for production preparation.

Importing this module performs no IO. Production is deliberately unavailable:
manifest flags and command-line switches cannot supply Olav's separate GO.
Remote worktrees must already exist at their reviewed commit; this module does
not install environments, create checkouts, use scheduled tasks or kill jobs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import ntpath
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import types
import uuid

PREPARATION_PURPOSES = frozenset({"file_handling_acceptance", "profile", "equivalence"})
MAX_COMMAND = 4096
MAX_METADATA = 2 * 1024**2
MAX_STDOUT = 65536
OLAV_HOST_KEY = "AAAAC3NzaC1lZDI1NTE5AAAAIL5f/yOUfOgaT2rtsKBGNfck2Vo6oO2VWLGs/SXuGA4F"


@dataclass(frozen=True)
class Endpoint:
    target: str = "olav_@10.10.10.146"
    python: str = "C:/FFAC_cache/production_second_host_env_20261011T160500Z/venv/Scripts/python.exe"
    python_sha256: str = "a059f15cc572e1f4259b647fc0cb8f7bad2c08510d7176b12ff85827c562091a"
    python_size_bytes: int = 241152
    ssh: str = "C:/Windows/System32/OpenSSH/ssh.exe"
    scp: str = "C:/Windows/System32/OpenSSH/scp.exe"
    known_hosts: str = str(Path.home() / ".ssh/known_hosts")
    host_key: str = OLAV_HOST_KEY


OLAV = Endpoint()


def _asset(path):
    p = Path(path)
    before = p.stat()
    if p.is_symlink() or not p.is_file():
        raise ValueError("Bound ordinary file required")
    with p.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Bound file changed")
    return dict(path=str(p), sha256=digest, size_bytes=after.st_size)


def _verify(bound):
    if (type(bound) is not dict or set(bound) != {"path", "sha256", "size_bytes"}
            or not isinstance(bound.get("path"), str)
            or not (Path(bound["path"]).is_absolute() or ntpath.isabs(bound["path"]))
            or not re.fullmatch(r"[0-9a-f]{64}", str(bound.get("sha256", "")))
            or type(bound.get("size_bytes")) is not int or bound["size_bytes"] < 0):
        raise ValueError("Full SHA/integer-size binding required")
    actual = _asset(bound["path"])
    if any(actual[k] != bound[k] for k in ("sha256", "size_bytes")):
        raise ValueError("Bound content differs")
    return Path(bound["path"])


def _remote_path(value):
    if (type(value) is not str or not ntpath.isabs(value) or not re.match(r"^[A-Za-z]:[\\/]", value)
            or any(c in value for c in '\r\n\0"%!<>|&^')
            or ".." in value.replace("\\", "/").split("/")
            or not ntpath.normcase(ntpath.normpath(value)).startswith("c:\\ffac_cache\\")):
        raise ValueError("Absolute dedicated C:/FFAC_cache locator required")
    return value.replace("\\", "/")


def _ps_quote(value):
    if not isinstance(value, str) or any(c in value for c in "\0\r\n"):
        raise ValueError("One literal argument required")
    return "'" + value.replace("'", "''") + "'"


def ssh_argv(endpoint, arguments):
    """Explicit PowerShell through the reviewed Bash shell, with literal args."""
    if endpoint.target != "olav_@10.10.10.146":
        raise ValueError("Only the existing reviewed Olav endpoint is implemented")
    literal = "& " + " ".join(_ps_quote(v) for v in arguments)
    command = "powershell.exe -NoProfile -NonInteractive -Command " + shlex.quote(literal)
    if len(command) > MAX_COMMAND:
        raise ValueError("Remote command exceeds the short-command limit")
    return [endpoint.ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=yes", "-o", "HostKeyAlgorithms=ssh-ed25519",
            "-o", "UserKnownHostsFile=" + endpoint.known_hosts, endpoint.target, command]


def _known_host(endpoint):
    if (endpoint.target != OLAV.target or endpoint.host_key != OLAV_HOST_KEY
            or endpoint.python_sha256 != OLAV.python_sha256
            or type(endpoint.python_size_bytes) is not int
            or endpoint.python_size_bytes != OLAV.python_size_bytes):
        raise ValueError("One reviewed server key and interpreter binding required")
    _remote_path(endpoint.python)
    p = Path(endpoint.known_hosts)
    with p.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise ValueError("Known-host inventory exceeds its bound")
    rows = [line.split() for line in raw.decode("utf-8").splitlines() if line and not line.startswith("#")]
    if not any(len(row) == 3 and "10.10.10.146" in row[0].split(",")
               and row[1:] == ["ssh-ed25519", endpoint.host_key] for row in rows):
        raise ValueError("Existing reviewed server key is absent; no TOFU or key replacement")


# Every small bootstrap verifies the actual interpreter before touching files.
_PIN = "import hashlib,sys,os;from pathlib import Path;p=Path(sys.executable);assert p.stat().st_size==int(sys.argv[1]) and hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
_INSPECT = _PIN + "p=Path(sys.argv[3]);e=p.exists();assert not e or (p.is_file() and not p.is_symlink());a=p.stat() if e else None;f=p.open('rb') if e else None;h=hashlib.file_digest(f,'sha256').hexdigest() if e else None;f.close() if e else None;b=p.stat() if e else None;assert not e or (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns);print(h+' '+str(b.st_size) if e else 'ABSENT')"
_MKDIR = _PIN + "Path(sys.argv[3]).parent.mkdir(parents=True,exist_ok=True)"
_PUBLISH = _PIN + "s,d=map(Path,sys.argv[3:5]);h,n=sys.argv[5],int(sys.argv[6]);f=lambda p:p.is_file() and not p.is_symlink() and p.stat().st_size==n and hashlib.sha256(p.read_bytes()).hexdigest()==h;assert f(s);e=d.exists();assert not e or f(d);os.rename(s,d) if not e and os.name=='nt' else (os.link(s,d),s.unlink()) if not e else None;assert e or (not s.exists() and f(d))"
_EXECUTE = _PIN + "import runpy;p=Path(sys.argv[3]);assert p.stat().st_size==int(sys.argv[5]) and hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[4];sys.argv=[str(p),*sys.argv[6:]];runpy.run_path(str(p),run_name='__main__')"


def _helpers(worktree, source_bindings=None):
    """Load only the two stdlib orchestration modules, bypassing DarSIA imports."""
    folder = Path(worktree) / "src/darsia/fluidflower"
    if source_bindings is not None:
        for name in ("production_io.py", "production_job.py"):
            _verify(dict(source_bindings[name], path=str(folder / name)))
    name = "_darsia_production_launch_helpers"
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    modules = {}
    for stem in ("production_io", "production_job"):
        spec = importlib.util.spec_from_file_location(name + "." + stem, folder / (stem + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if source_bindings is not None:
            _verify(dict(source_bindings[stem + ".py"], path=str(folder / (stem + ".py"))))
        modules[stem] = module
    return modules["production_io"], modules["production_job"]


class RemoteLauncher:
    def __init__(self, *, endpoint=OLAV, runner=subprocess.run):
        self.endpoint, self.runner = endpoint, runner

    def _run(self, argv, *, timeout=60):
        _known_host(self.endpoint)
        result = self.runner(argv, check=True, capture_output=True, timeout=timeout, shell=False)
        stdout, stderr = result.stdout or b"", result.stderr or b""
        if isinstance(stdout, str):
            stdout = stdout.encode("utf-8")
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8")
        if len(stdout) > MAX_STDOUT or len(stderr) > MAX_STDOUT:
            raise ValueError("Remote command output exceeded its metadata bound")
        return stdout.decode("utf-8-sig").strip()

    def _python(self, code, *arguments, timeout=60):
        return self._run(ssh_argv(self.endpoint, [self.endpoint.python, "-I", "-B", "-c", code,
            str(self.endpoint.python_size_bytes), self.endpoint.python_sha256, *map(str, arguments)]), timeout=timeout)

    def remote_asset(self, path):
        path = _remote_path(path)
        value = self._python(_INSPECT, path)
        if value == "ABSENT":
            return None
        if not re.fullmatch(r"[0-9a-f]{64} [0-9]+", value):
            raise ValueError("Remote file hash/size response differs")
        sha, size = value.split()
        return dict(path=path, sha256=sha, size_bytes=int(size))

    def stage_once(self, descriptor, remote_path):
        """Reuse verified content; upload only to a fresh temporary, never final."""
        source = _verify(descriptor)
        if source.suffix.lower() not in {".py", ".json"} or descriptor["size_bytes"] > MAX_METADATA:
            raise ValueError("Only bounded scripts/job metadata are staged")
        remote_path = _remote_path(remote_path)
        current = self.remote_asset(remote_path)
        if current is not None:
            if any(current[k] != descriptor[k] for k in ("sha256", "size_bytes")):
                raise FileExistsError("Remote immutable content conflict")
            return dict(current, reused=True)
        temporary = remote_path + "." + uuid.uuid4().hex + ".partial"
        self._python(_MKDIR, temporary)
        options = ["-B", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "StrictHostKeyChecking=yes", "-o", "HostKeyAlgorithms=ssh-ed25519",
                   "-o", "UserKnownHostsFile=" + self.endpoint.known_hosts]
        self._run([self.endpoint.scp, *options, str(source), self.endpoint.target + ":" + temporary])
        _verify(descriptor)
        self._python(_PUBLISH, temporary, remote_path, descriptor["sha256"], descriptor["size_bytes"])
        actual = self.remote_asset(remote_path)
        if actual is None or any(actual[k] != descriptor[k] for k in ("sha256", "size_bytes")):
            raise ValueError("Remote staged content verification failed")
        return dict(actual, reused=False)

    def fetch_file(self, descriptor, destination, *, io):
        """Return a file with its recorded SHA; no JSON/stdout result transport."""
        if (type(descriptor) is not dict or set(descriptor) != {"path", "sha256", "size_bytes"}
                or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("sha256", "")))
                or type(descriptor.get("size_bytes")) is not int or descriptor["size_bytes"] < 0):
            raise ValueError("Full recorded result SHA/integer-size descriptor required")
        remote = _remote_path(descriptor["path"])
        current = self.remote_asset(remote)
        if current is None or any(current[k] != descriptor[k] for k in ("sha256", "size_bytes")):
            raise ValueError("Remote result content differs from its recorded binding")
        destination = Path(destination)
        if destination.exists():
            io.verify(dict(descriptor, path=str(destination)))
            return io.asset(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".download.partial")
        options = ["-B", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
                   "-o", "HostKeyAlgorithms=ssh-ed25519", "-o", "UserKnownHostsFile=" + self.endpoint.known_hosts]
        self._run([self.endpoint.scp, *options, self.endpoint.target + ":" + remote, str(temporary)])
        io.verify(dict(descriptor, path=str(temporary)))
        io.publish_file(temporary, destination, descriptor)
        return io.asset(destination)

    def prepare(self, manifest, *, local_worktree, remote_worktree):
        io_module, job_module = _helpers(local_worktree)
        io = io_module.NasIO()
        path = io.verify(manifest)
        document = io.read_json(path, expected=manifest, max_bytes=MAX_METADATA)
        job_module.validate_manifest(document)
        if document["purpose"] not in PREPARATION_PURPOSES:
            raise PermissionError("This launcher has no production GO channel")
        if document["settings"]["host"].casefold() != "olav":
            raise ValueError("Manifest host differs from the reviewed endpoint")
        if job_module.code_identity(local_worktree) != document["code"]:
            raise ValueError("Local code differs from the clean manifest commit")
        remote_worktree = _remote_path(remote_worktree)
        sources = {}
        for name in ("production_io.py", "production_job.py", "production_launch.py"):
            local = Path(local_worktree) / "src/darsia/fluidflower" / name
            sources[name] = self.stage_once(io.asset(local), remote_worktree + "/src/darsia/fluidflower/" + name)
        if not re.fullmatch(r"[0-9a-f]{32}", str(document.get("job_id", ""))):
            raise ValueError("One immutable job ID required")
        manifest_root = ntpath.dirname(remote_worktree).replace("\\", "/") + "/job_manifests"
        remote_manifest = self.stage_once(manifest, manifest_root + "/" + document["job_id"] + ".json")
        return dict(manifest=remote_manifest, local_manifest=manifest, worktree=remote_worktree,
                    sources=sources, purpose=document["purpose"], code=document["code"], settings=document["settings"])

    def _command(self, packet, mode, result_path, *, entrypoint=None):
        source = packet["sources"]["production_launch.py"]
        args = [mode, "--manifest", packet["manifest"]["path"], "--manifest-sha256", packet["manifest"]["sha256"],
                "--manifest-size", str(packet["manifest"]["size_bytes"]), "--worktree", packet["worktree"],
                "--purpose", packet["purpose"], "--result", _remote_path(result_path)]
        for name, option in (("production_io.py", "io"), ("production_job.py", "job")):
            args.extend(["--" + option + "-sha256", packet["sources"][name]["sha256"],
                         "--" + option + "-size", str(packet["sources"][name]["size_bytes"])])
        if entrypoint is not None:
            args.extend(["--entrypoint", entrypoint["path"], "--entrypoint-sha256", entrypoint["sha256"],
                         "--entrypoint-size", str(entrypoint["size_bytes"])])
        return ssh_argv(self.endpoint, [self.endpoint.python, "-I", "-B", "-c", _EXECUTE,
            str(self.endpoint.python_size_bytes), self.endpoint.python_sha256, source["path"], source["sha256"],
            str(source["size_bytes"]), *args])

    def self_test(self, packet, result_path):
        """Root-only metadata check: no arrays, native imports or scientific child."""
        if packet["purpose"] not in PREPARATION_PURPOSES:
            raise PermissionError("No production permission from packet data")
        self._run(self._command(packet, "remote-self-test", result_path))
        actual = self.remote_asset(result_path)
        if actual is None:
            raise ValueError("Actual metadata receipt was not written")
        return actual

    def launch(self, packet, entrypoint, result_path, *, before_launch):
        """One actual start, never retried; Root supplies the fresh authority check."""
        if packet["purpose"] not in PREPARATION_PURPOSES:
            raise PermissionError("Production requires a future separate trusted Root GO channel")
        if not callable(before_launch) or before_launch() is not True:
            raise PermissionError("Fresh Root authorization is required")
        argv = self._command(packet, "remote-run", result_path, entrypoint=entrypoint)
        # No timeout/kill/retry for an actual job. Its own reviewed lifecycle owns retirement.
        self._run(argv, timeout=None)
        actual = self.remote_asset(result_path)
        if actual is None:
            raise RuntimeError("Remote completion metadata is absent; never retry the job blindly")
        return actual


def _remote(args):
    bindings = {"production_io.py": dict(sha256=args.io_sha256, size_bytes=args.io_size),
                "production_job.py": dict(sha256=args.job_sha256, size_bytes=args.job_size)}
    io_module, job = _helpers(args.worktree, bindings)
    io = io_module.NasIO()
    descriptor = dict(path=args.manifest, sha256=args.manifest_sha256, size_bytes=args.manifest_size)
    manifest = io.read_json(io.verify(descriptor), expected=descriptor, max_bytes=MAX_METADATA)
    job.validate_manifest(manifest)
    if args.purpose not in PREPARATION_PURPOSES or manifest["purpose"] != args.purpose:
        raise PermissionError("Production cannot be authorized by remote arguments or manifest")
    if socket.gethostname().casefold() != manifest["settings"]["host"].casefold():
        raise ValueError("Actual remote host differs")
    if job.code_identity(args.worktree) != manifest["code"] or job.environment_lock() != manifest["environment"]:
        raise ValueError("Actual remote code/environment differs from manifest")
    result_path = _remote_path(args.result)
    record = dict(format="darsia-production-remote-receipt-v1", complete=True, mode=args.mode,
                  manifest=descriptor, code=manifest["code"], settings=manifest["settings"],
                  environment=manifest["environment"], host=socket.gethostname(),
                  scientific_arrays_read=False, scientific_process_started=False, production_permission_granted=False)
    if args.mode == "remote-run":
        entry = dict(path=args.entrypoint, sha256=args.entrypoint_sha256, size_bytes=args.entrypoint_size)
        _remote_path(entry["path"])
        io.verify(entry)
        if not Path(entry["path"]).resolve().is_relative_to(Path(args.worktree).resolve()):
            raise ValueError("Entrypoint must be part of the clean reviewed worktree")
        argv = [sys.executable, "-I", "-B", entry["path"], "--manifest", descriptor["path"],
                "--manifest-sha256", descriptor["sha256"], "--manifest-size-bytes", str(descriptor["size_bytes"]),
                "--purpose", args.purpose, "--worktree", args.worktree]
        # An exclusive start record denies duplicate invocation before Popen.
        io.write_json(result_path + ".start.json", dict(manifest=descriptor, entrypoint=entry, argv=argv))
        out, err = Path(result_path + ".stdout.log"), Path(result_path + ".stderr.log")
        with out.open("xb") as stdout, err.open("xb") as stderr:
            env = dict(os.environ)
            thread_environment = {name: str(manifest["settings"]["threads"]) for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
            env.update(thread_environment)
            flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, cwd=args.worktree, env=env, creationflags=flags)
            record["thread_environment"] = thread_environment
            record["child"] = job.worker_identity(child.pid)
            io.write_json(result_path + ".child.json", dict(manifest=descriptor, child=record["child"]))
            record["exit_code"] = child.wait()
        record.update(scientific_arrays_read=None, scientific_process_started=True,
                      complete=record["exit_code"] == 0, entrypoint=entry,
                      logs={"stdout": io.asset(out), "stderr": io.asset(err)})
    io.write_json(result_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("remote-self-test", "remote-run"))
    for name in ("manifest", "manifest-sha256", "worktree", "purpose", "result", "io-sha256", "job-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("manifest-size", "io-size", "job-size"):
        parser.add_argument("--" + name, required=True, type=int)
    parser.add_argument("--entrypoint");parser.add_argument("--entrypoint-sha256")
    parser.add_argument("--entrypoint-size", type=int)
    args = parser.parse_args()
    if args.mode == "remote-run" and any(getattr(args, k) is None for k in ("entrypoint", "entrypoint_sha256", "entrypoint_size")):
        parser.error("Actual entrypoint binding required")
    _remote(args)


if __name__ == "__main__":
    main()

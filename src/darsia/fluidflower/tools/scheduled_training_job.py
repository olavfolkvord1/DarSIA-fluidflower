"""Run a frozen, long Windows job independently of a Codex terminal.

Prepare a local JSON configuration first, then register and start the task as
separate, explicit steps. The scheduled action invokes a local snapshot of this
runner through pythonw.exe. The actual program is passed to Popen as an argv
list, with no shell. A user must be logged on: InteractiveToken is intentional
because FF_AC reads source files over SMB.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SCHEMA = 1
TASK_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_file(value: str, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{field} must be an existing absolute file: {value}")
    return path


def _absolute_dir(value: str, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{field} must be an existing absolute directory: {value}")
    return path


def _task_name(value: str) -> str:
    import re

    if not re.fullmatch(r"\\[A-Za-z0-9_.-]+(?:\\[A-Za-z0-9_.-]+)*", value):
        raise ValueError("task_name must be a full Scheduler path such as \\FFAC\\Candidate001")
    return value


def _local_log_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.drive or str(path).startswith("\\\\"):
        raise ValueError("log_dir must be a local absolute drive path")
    return path


def _local_lock_file(value: str) -> Path:
    path = _local_log_dir(value)
    if path.exists() and not path.is_file():
        raise ValueError("lock_file must be a local file path")
    return path


def validate_config(config: dict) -> dict:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    name = _task_name(config["task_name"])
    executable = _absolute_file(config["executable"], "executable")
    cwd = _absolute_dir(config["cwd"], "cwd")
    runner = _absolute_file(config["runner_pythonw"], "runner_pythonw")
    if runner.name.lower() != "pythonw.exe":
        raise ValueError("runner_pythonw must name pythonw.exe (hidden window)")
    log_dir = _local_log_dir(config["log_dir"])
    argv = config["argv"]
    if not isinstance(argv, list) or not all(isinstance(s, str) and "\x00" not in s for s in argv):
        raise ValueError("argv must be a list of strings without NUL")
    extra_env = config.get("environment", {})
    if not isinstance(extra_env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and k and "=" not in k and "\x00" not in k + v
        for k, v in extra_env.items()
    ):
        raise ValueError("environment must map variable names to strings")
    required = config.get("required_paths", [])
    absent = config.get("absent_paths", [])
    if not isinstance(required, list) or not isinstance(absent, list):
        raise ValueError("required_paths and absent_paths must be lists")
    if not all(isinstance(p, str) and Path(p).is_absolute() for p in required + absent):
        raise ValueError("required_paths and absent_paths must contain absolute paths")
    required_sha256 = config.get("required_sha256", {})
    if not isinstance(required_sha256, dict) or not all(
        isinstance(path, str) and Path(path).is_absolute()
        and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        for path, digest in required_sha256.items()
    ):
        raise ValueError("required_sha256 must map absolute file paths to lowercase SHA-256")
    lock_file = config.get("lock_file")
    if lock_file is not None:
        lock_file = str(_local_lock_file(lock_file))
    return dict(schema=SCHEMA, task_name=name, executable=str(executable), argv=argv,
                cwd=str(cwd), runner_pythonw=str(runner), log_dir=str(log_dir),
                environment=extra_env, required_paths=required, absent_paths=absent,
                required_sha256=required_sha256, lock_file=lock_file)


def _task_xml(spec: dict, runner: Path, spec_path: Path) -> bytes:
    def node(parent, name, value=None, **attrs):
        child = ET.SubElement(parent, f"{{{TASK_NS}}}{name}", attrs)
        if value is not None:
            child.text = str(value)
        return child

    ET.register_namespace("", TASK_NS)
    task = ET.Element(f"{{{TASK_NS}}}Task", {"version": "1.2"})
    registration = node(task, "RegistrationInfo")
    node(registration, "Description", "Frozen local process with heartbeat and exit receipt")
    node(task, "Triggers")
    principals = node(task, "Principals")
    principal = node(principals, "Principal", id="Author")
    account = os.environ.get("USERDOMAIN", ".") + "\\" + getpass.getuser()
    node(principal, "UserId", account)
    node(principal, "LogonType", "InteractiveToken")
    node(principal, "RunLevel", "LeastPrivilege")
    settings = node(task, "Settings")
    for key, value in (
        ("MultipleInstancesPolicy", "IgnoreNew"),
        ("DisallowStartIfOnBatteries", "false"),
        ("StopIfGoingOnBatteries", "false"),
        ("AllowHardTerminate", "true"),
        ("StartWhenAvailable", "true"),
        ("ExecutionTimeLimit", "PT0S"),
        ("Hidden", "true"),
        ("Enabled", "true"),
    ):
        node(settings, key, value)
    actions = node(task, "Actions", Context="Author")
    action = node(actions, "Exec")
    node(action, "Command", spec["runner_pythonw"])
    node(action, "Arguments", subprocess.list2cmdline(["-I", str(runner), "run", "--spec", str(spec_path)]))
    node(action, "WorkingDirectory", spec["cwd"])
    return ET.tostring(task, encoding="utf-16", xml_declaration=True)


def prepare(config_path: Path) -> Path:
    spec = validate_config(read_json(config_path))
    log_dir = Path(spec["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=False)
    spec_path = log_dir / "job_spec.json"
    runner = log_dir / "scheduled_training_job.py"
    shutil.copy2(Path(__file__).resolve(), runner)
    atomic_json(spec_path, spec)
    (log_dir / "task.xml").write_bytes(_task_xml(spec, runner, spec_path))
    atomic_json(log_dir / "preparation.json", dict(prepared_utc=utc_now(),
        task_name=spec["task_name"], spec_sha256=sha256(spec_path), runner_sha256=sha256(runner),
        task_xml_sha256=sha256(log_dir / "task.xml"), registered=False, started=False))
    return log_dir


def _prepared(log_dir: Path) -> tuple[dict, dict]:
    receipt = read_json(log_dir / "preparation.json")
    spec_path = log_dir / "job_spec.json"
    runner = log_dir / "scheduled_training_job.py"
    if (sha256(spec_path) != receipt["spec_sha256"]
            or sha256(runner) != receipt["runner_sha256"]
            or sha256(log_dir / "task.xml") != receipt["task_xml_sha256"]):
        raise ValueError("Prepared task files changed")
    spec = validate_config(read_json(spec_path))
    if Path(spec["log_dir"]).resolve() != log_dir.resolve():
        raise ValueError("Prepared log_dir does not match")
    return spec, receipt


def scheduler(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks.exe", *args], capture_output=True, text=True, shell=False)


def register(log_dir: Path) -> dict:
    spec, prep = _prepared(log_dir)
    result = scheduler("/Create", "/TN", spec["task_name"], "/XML", str(log_dir / "task.xml"))
    if result.returncode:
        raise RuntimeError(f"Task registration failed ({result.returncode}): {result.stdout} {result.stderr}")
    prep.update(registered=True, registered_utc=utc_now())
    atomic_json(log_dir / "preparation.json", prep)
    return dict(task_name=spec["task_name"], registered=True, detail=result.stdout.strip())


def start(log_dir: Path) -> dict:
    spec, prep = _prepared(log_dir)
    if not prep["registered"]:
        raise ValueError("Register this prepared task first")
    if prep["started"] or (log_dir / "heartbeat.json").exists() or (log_dir / "exit_receipt.json").exists():
        raise ValueError("This frozen task has already been started; prepare a new log directory")
    _check_paths(spec)
    result = scheduler("/Run", "/TN", spec["task_name"])
    if result.returncode:
        raise RuntimeError(f"Task start failed ({result.returncode}): {result.stdout} {result.stderr}")
    prep.update(started=True, start_requested_utc=utc_now())
    atomic_json(log_dir / "preparation.json", prep)
    return dict(task_name=spec["task_name"], start_requested=True, detail=result.stdout.strip())


def _check_paths(spec: dict) -> None:
    for path in spec["required_paths"]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path missing: {path}")
    for path in spec["absent_paths"]:
        if Path(path).exists():
            raise FileExistsError(f"Path must be absent at launch: {path}")
    for raw_path, expected in spec.get("required_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Required SHA-256 source missing: {path}")
        if sha256(path) != expected:
            raise ValueError(f"Required SHA-256 source changed: {path}")


@contextmanager
def _exclusive_lock(path: Path | None):
    """Hold a process-lifetime lock shared by FF_UM and FF_BM workers.

    The kernel releases it if the launcher exits unexpectedly. This guard only
    serializes jobs which opt into this launcher, not unrelated manual fits.
    """
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another scheduled phase job holds {path}") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Another scheduled phase job holds {path}") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _terminate_tree(child: subprocess.Popen) -> int:
    """End the coordinator and subprocesses it created; preserve its exit code."""
    if child.poll() is not None:
        return child.returncode
    if os.name == "nt":
        # The training coordinator starts further Python processes. Ending only
        # its direct PID can leave a fitting or phase-map child running.
        subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                       capture_output=True, text=True, shell=False)
    if child.poll() is None:
        child.terminate()
    try:
        return child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def run(spec_path: Path) -> int:
    log_dir = spec_path.parent
    started = utc_now()
    child = None
    stdout = stderr = None
    state = "launcher_failed"
    error = None
    child_code = None
    cancelled = False
    try:
        spec, _ = _prepared(log_dir)
        _check_paths(spec)
        with _exclusive_lock(Path(spec["lock_file"]) if spec.get("lock_file") else None):
            lock = log_dir / "worker.lock"
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(pid=os.getpid(), started_utc=started)))
            env = os.environ.copy()
            env.update(spec["environment"])
            env["PYTHONUNBUFFERED"] = "1"
            stdout = (log_dir / "stdout.log").open("xb")
            stderr = (log_dir / "stderr.log").open("xb")
            child = subprocess.Popen([spec["executable"], *spec["argv"]], cwd=spec["cwd"],
                                     env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            while True:
                child_code = child.poll()
                atomic_json(log_dir / "heartbeat.json", dict(state="running" if child_code is None else "exited",
                    updated_utc=utc_now(), worker_pid=os.getpid(), child_pid=child.pid,
                    task_name=spec["task_name"], spec_sha256=sha256(spec_path)))
                if child_code is not None:
                    break
                if (log_dir / "cancel.request.json").exists():
                    cancelled = True
                    child_code = _terminate_tree(child)
                    break
                time.sleep(5)
            state = "cancelled" if cancelled else ("succeeded" if child_code == 0 else "failed")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if child is not None and child.poll() is None:
            child_code = _terminate_tree(child)
        (log_dir / "launcher_error.log").write_text(error + "\n", encoding="utf-8")
    finally:
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
        atomic_json(log_dir / "exit_receipt.json", dict(state=state, started_utc=started,
            completed_utc=utc_now(), worker_pid=os.getpid(), child_pid=child.pid if child else None,
            exit_code=child_code, cancelled=cancelled, error=error,
            task_name=read_json(spec_path).get("task_name") if spec_path.is_file() else None,
            spec_sha256=sha256(spec_path) if spec_path.is_file() else None))
    if state == "succeeded":
        return 0
    if state == "failed" and child_code is not None and 0 < child_code <= 255:
        return child_code
    return 1


def status(log_dir: Path) -> dict:
    spec, prep = _prepared(log_dir)
    result = scheduler("/Query", "/TN", spec["task_name"], "/FO", "LIST", "/V")
    return dict(preparation=prep,
        heartbeat=read_json(log_dir / "heartbeat.json") if (log_dir / "heartbeat.json").is_file() else None,
        exit_receipt=read_json(log_dir / "exit_receipt.json") if (log_dir / "exit_receipt.json").is_file() else None,
        scheduler_query_exit_code=result.returncode,
        scheduler_query=result.stdout.strip() if result.returncode == 0 else result.stderr.strip())


def cancel(log_dir: Path) -> dict:
    spec, prep = _prepared(log_dir)
    if not prep["started"]:
        raise ValueError("Task was never started")
    if (log_dir / "exit_receipt.json").exists():
        return dict(already_finished=True, exit_receipt=read_json(log_dir / "exit_receipt.json"))
    atomic_json(log_dir / "cancel.request.json", dict(requested_utc=utc_now(), task_name=spec["task_name"]))
    return dict(cancellation_requested=True, task_name=spec["task_name"])


def remove(log_dir: Path) -> dict:
    spec, _ = _prepared(log_dir)
    if (log_dir / "heartbeat.json").exists() and not (log_dir / "exit_receipt.json").exists():
        raise ValueError("Task may still be running; cancel and wait for its exit receipt first")
    result = scheduler("/Delete", "/TN", spec["task_name"], "/F")
    if result.returncode:
        raise RuntimeError(f"Task deletion failed ({result.returncode}): {result.stdout} {result.stderr}")
    return dict(removed=True, task_name=spec["task_name"], detail=result.stdout.strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument("--config", type=Path, required=True)
    commands.add_parser("run").add_argument("--spec", type=Path, required=True)
    for name in ("register", "start", "status", "cancel", "remove"):
        commands.add_parser(name).add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.spec)
    if args.command == "prepare":
        value = dict(log_dir=str(prepare(args.config)), prepared=True)
    else:
        value = globals()[args.command](args.log_dir)
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

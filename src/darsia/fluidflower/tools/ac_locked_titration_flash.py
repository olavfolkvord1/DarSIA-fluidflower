"""Run the physically locked TitrationFlash diagnostic campaign.

The aqueous chemistry is the shared BTB/carbonate titration transfer. Flash is
fixed globally to the FF_UM convention ``(0, 1, 1, 1.5)``, so aqueous
saturation and free-gas onset are one common signal coordinate without an
optimisable overlap. Run-native color paths remain free to map each experiment
onto that common signal axis.

The six diagnostic runs are split evenly between CUDA and OpenCL. Each master
owns exactly one run and keeps its GPU context resident for all trials.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    import psutil
except Exception:  # Native process diagnostics are optional in bridged sessions.
    psutil = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.ac_production_campaign import EXPECTED_RUNTIME_DEPTH_SHA256  # noqa: E402
from darsia.fluidflower.tools.verify_ac_depth_maps import verify_depth_maps  # noqa: E402


CAMPAIGN_ID = "locked_titration_ffum_20260804_3200"
VARIANT = "titration_locked_ffum_runpath"
RUN_ASSIGNMENTS = {
    "moderskipet": ("ac20", "ac24", "ac26"),
    "olav": ("ac60", "ac32", "ac27"),
}
RUNS = tuple(run for role in RUN_ASSIGNMENTS.values() for run in role)
MAX_ITERS = 3200
WARMUP_ITERS = 150
OPTUNA_SEED = 17
BOUNDS_FILE = "config/bounds_seg6_titration_locked_ffum.json"
LOCKED_FLASH = {
    "flash.min_value_aq": 0.0,
    "flash.max_value_aq": 1.0,
    "flash.min_value_g": 1.0,
    "flash.max_value_g": 1.5,
}
RESULTS_ROOT = configured_tool_path('ffac', 'previous_results')
MEASUREMENTS_PATH = project_path('data/depth_measurements.csv')
COLOR_PATH_SOURCE_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'final5_20260729/pre_final5_calibration_backup'
FACTORIAL_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'flash_anchor_factorial_20260804_3200_pairedseeds'
SEED_VARIANTS = (
    "titration_coupled_runpath",
    "titration_uncoupled_runpath",
)
DEFAULT_LOGS_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
DEFAULT_QUEUE_ROOT = rf"\\Moderskipet\Darsia_Queue\Kalibrering_AC_{CAMPAIGN_ID}"


@dataclass(frozen=True)
class ProcessSpec:
    run: str
    queue: str
    command: tuple[str, ...]
    stdout: str
    env: dict[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    return completed.stdout.strip()


def _source_folder(run: str) -> Path:
    return (
        COLOR_PATH_SOURCE_ROOT
        / run
        / "calibration"
        / "color_path_interpretation"
    )


def _locked_params(params: dict) -> dict:
    locked = dict(params)
    locked.update(LOCKED_FLASH)
    return locked


def _build_seed_bundle(logs_root: Path) -> Path:
    params_by_run: dict[str, list[dict]] = {}
    sources: dict[str, list[dict[str, object]]] = {}
    for run in RUNS:
        run_candidates: list[dict] = []
        run_sources: list[dict[str, object]] = []
        seen: set[str] = set()
        for source_variant in SEED_VARIANTS:
            source = FACTORIAL_ROOT / source_variant / f"final_full_scale_{run}.json"
            if not source.is_file():
                raise FileNotFoundError(f"Factorial seed is unavailable: {source}")
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            raw_params = payload.get("params")
            if not isinstance(raw_params, dict) or not raw_params:
                raise RuntimeError(f"Factorial winner has no params mapping: {source}")
            params = _locked_params(raw_params)
            params_hash = hashlib.sha256(
                json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if params_hash not in seen:
                run_candidates.append(
                    {
                        "params": params,
                        "source": str(source),
                        "source_family": source_variant,
                    }
                )
                seen.add(params_hash)
            run_sources.append(
                {
                    "family": source_variant,
                    "path": str(source),
                    "sha256": _sha256(source),
                    "locked_params_sha256": params_hash,
                    "source_objective_full_scale": float(
                        payload["objective_full_scale"]
                    ),
                }
            )
        params_by_run[run] = run_candidates
        sources[run] = run_sources

    output = logs_root / "locked_seed_params.json"
    _atomic_json(
        output,
        {
            "metadata": {
                "created_at_utc": _utc_now(),
                "purpose": (
                    "Seed from both completed Titration runpath cells after "
                    "replacing every Flash coordinate with the global FF_UM values."
                ),
                "locked_flash": LOCKED_FLASH,
                "sources": sources,
            },
            "params_by_run": params_by_run,
        },
    )
    return output


def validate_plan() -> dict:
    assigned = [run for runs in RUN_ASSIGNMENTS.values() for run in runs]
    if len(assigned) != 6 or len(set(assigned)) != 6:
        raise RuntimeError(f"Expected six unique diagnostic runs, got {assigned}")
    if set(RUN_ASSIGNMENTS) != {"moderskipet", "olav"}:
        raise RuntimeError(f"Unexpected host assignments: {RUN_ASSIGNMENTS}")
    if any(len(runs) != 3 for runs in RUN_ASSIGNMENTS.values()):
        raise RuntimeError(f"GPU assignments are not balanced: {RUN_ASSIGNMENTS}")
    return {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "variant": VARIANT,
        "runs": list(RUNS),
        "run_assignments": {
            host: list(runs) for host, runs in RUN_ASSIGNMENTS.items()
        },
        "calibrations": len(RUNS),
        "optuna_trials": len(RUNS) * MAX_ITERS,
        "random_warmups": len(RUNS) * WARMUP_ITERS,
        "max_iters_per_calibration": MAX_ITERS,
        "random_warmups_per_calibration": WARMUP_ITERS,
        "optuna_seed": OPTUNA_SEED,
        "locked_flash": LOCKED_FLASH,
        "max_concurrent_gpu_contexts_per_host": 3,
        "max_active_runs_per_context": 1,
        "max_in_flight_per_run": 1,
        "fixed_settings": {
            "flash_model": "TitrationFlash",
            "phase_transition": "max_value_aq == min_value_g",
            "color_paths": "run-native",
            "template_registration": "ac14_template",
            "template_registration_mode": "partial_affine",
            "objective": "point-wise L1",
            "frame_policy": "exact",
            "static_light_correction": "off",
            "bounds_file": BOUNDS_FILE,
            "depth_sha256": EXPECTED_RUNTIME_DEPTH_SHA256,
            "color_path_source_root": str(COLOR_PATH_SOURCE_ROOT),
        },
    }


def _campaign_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "FFAC_REQUIRE_VARYING_DEPTH": "on",
            "FFAC_EXPECTED_DEPTH_SHA256": EXPECTED_RUNTIME_DEPTH_SHA256,
            "FFAC_TITRATION_FLASH": "on",
            "FFAC_COUPLE_AQ_GAS": "on",
            "FFAC_TEMPLATE_REGISTRATION": "ac14_template",
            "FFAC_TEMPLATE_REGISTRATION_MODE": "partial_affine",
            "FFAC_TEMPLATE_REGISTRATION_STRICT": "on",
            "FFAC_STATIC_LIGHT_CORRECTION": "off",
            "FFAC_COLOR_PATH_SOURCE_ROOT": str(COLOR_PATH_SOURCE_ROOT),
            "FFAC_COLOR_PATH_SOURCE_STRICT": "on",
            "FFAC_FLAGGED_FRAME_POLICY": "exact",
            "FFAC_COLOR_PATH_ANCHOR": "off",
            "FFAC_COLOR_PATH_ANCHOR_WEIGHT": "0.75",
            "FFAC_COLOR_PATH_ANCHOR_STRICT": "on",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENCV_NUM_THREADS": "1",
        }
    )
    for name in (
        "FFAC_MASTER_LIGHT_CONTEXT",
        "FFAC_STATIC_LIGHT_REFERENCE",
        "FFAC_STATIC_LIGHT_SPATIAL_SIGMA",
        "FFAC_PHOTOMETRIC_ANCHOR",
        "FFAC_PHOTOMETRIC_ANCHOR_MODE",
        "FFAC_SIGNAL_PARAMETERIZATION",
        "FFAC_PHASE_SEPARATION",
    ):
        env.pop(name, None)
    return env


def _master_command(
    *,
    run: str,
    backend: str,
    queue: str,
    logs_root: Path,
    seed_path: Path,
) -> tuple[str, ...]:
    python = Path(sys.executable)
    logs_dir = logs_root / VARIANT / run
    local_optuna = (
        Path(os.environ.get("LOCALAPPDATA", str(project_path('logs'))))
        / "ff_ac"
        / "optuna"
        / CAMPAIGN_ID
        / run
    )
    return (
        str(python),
        "scripts/distributed_auto_calibration_queue.py",
        "master",
        "--queue",
        queue,
        "--no-clear-queue",
        "--runs",
        run,
        "--config-dir",
        "config_seg6/run_ac",
        "--logs-dir",
        str(logs_dir),
        "--exact-logs-dir",
        "--use-facies",
        "true",
        "--per-label",
        "true",
        "--objective-integral",
        "off",
        "--bounds-file",
        BOUNDS_FILE,
        "--no-save-calibration",
        "--seed-params-file",
        str(seed_path),
        "--optuna-seed",
        str(OPTUNA_SEED),
        "--optuna-storage-dir",
        str(local_optuna),
        "--max-iters",
        str(MAX_ITERS),
        "--warmup-iters",
        str(WARMUP_ITERS),
        "--run-mode",
        "parallel",
        "--max-active-runs",
        "1",
        "--max-in-flight-per-run",
        "1",
        "--control-dir",
        f"{queue}\\control",
        "--sanity-every",
        "0",
        "--sanity-scale",
        "1.00",
        "--quality-dtype",
        "float32",
        "--local-eval-backend",
        backend,
        "--local-run",
        run,
        "--local-owner-id",
        f"{socket.gethostname()}_{VARIANT}_{run}",
        "--max-retries",
        "4",
        "--master-mem-log-seconds",
        "120",
    )


def run_preflight(logs_root: Path, role: str) -> tuple[dict, Path]:
    plan = validate_plan()
    python = Path(sys.executable)
    if not python.is_file():
        raise FileNotFoundError(f"Python unavailable: {python}")
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"Results root unavailable: {RESULTS_ROOT}")
    bounds_path = REPO_ROOT / BOUNDS_FILE
    if not bounds_path.is_file():
        raise FileNotFoundError(f"Locked Flash bounds unavailable: {bounds_path}")
    for run in RUN_ASSIGNMENTS[role]:
        files = sorted(_source_folder(run).glob("color_path_interpretation_*.json"))
        if not files:
            raise FileNotFoundError(f"Run-native color paths unavailable for {run}")
    seed_path = _build_seed_bundle(logs_root)
    depth = verify_depth_maps(
        list(RUN_ASSIGNMENTS[role]),
        results_root=RESULTS_ROOT,
        measurements_path=MEASUREMENTS_PATH,
    )
    payload = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "checked_at_utc": _utc_now(),
        "host": socket.gethostname().upper(),
        "role": role,
        "backend": "cuda" if role == "moderskipet" else "opencl",
        "ok": True,
        "ff_ac_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "ff_ac_dirty": bool(_git_value(REPO_ROOT, "status", "--porcelain")),
        "darsia_commit": _git_value(project_path('external/darsia'), "rev-parse", "HEAD"),
        "depth": depth,
        "seed_file": str(seed_path),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (
                SCRIPT_DIR / "auto_calibrate_color_to_mass.py",
                SCRIPT_DIR / "distributed_auto_calibration_queue.py",
                SCRIPT_DIR / "ffac_titration_flash.py",
                REPO_ROOT / BOUNDS_FILE,
                Path(__file__).resolve(),
            )
        },
        "plan": plan,
    }
    _atomic_json(logs_root / "campaign_plan.json", plan)
    _atomic_json(logs_root / f"preflight_{socket.gethostname().upper()}.json", payload)
    return payload, seed_path


def build_specs(
    *, role: str, logs_root: Path, queue_root: str, seed_path: Path
) -> tuple[ProcessSpec, ...]:
    backend = "cuda" if role == "moderskipet" else "opencl"
    local_logs = project_path('logs') / CAMPAIGN_ID / role
    env = _campaign_env()
    return tuple(
        ProcessSpec(
            run=run,
            queue=f"{queue_root}_{run}",
            command=_master_command(
                run=run,
                backend=backend,
                queue=f"{queue_root}_{run}",
                logs_root=logs_root,
                seed_path=seed_path,
            ),
            stdout=str(local_logs / f"{run}.log"),
            env=env,
        )
        for run in RUN_ASSIGNMENTS[role]
    )


def _queue_complete(queue: str) -> bool:
    try:
        return (Path(queue) / "master_complete.json").is_file()
    except OSError:
        return False


def _queue_status(queue: str) -> dict:
    root = Path(queue)

    def count(directory: str) -> int:
        try:
            return sum(path.is_file() for path in (root / directory).iterdir())
        except OSError:
            return 0

    return {
        "queue": queue,
        "exists": root.exists(),
        "pending": count("pending"),
        "in_progress": count("in_progress"),
        "done": count("done"),
        "failed": count("failed"),
        "master_complete": _queue_complete(queue),
    }


def _active_processes() -> list[dict]:
    if psutil is None:
        return []
    marker = CAMPAIGN_ID.lower()
    excluded = {os.getpid()}
    parent = psutil.Process(os.getpid()).parent()
    while parent is not None:
        excluded.add(parent.pid)
        parent = parent.parent()
    rows = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        folded = command.lower()
        process_name = str(process.info.get("name") or "").lower()
        is_campaign_process = (
            process_name in {"python.exe", "pythonw.exe", "python"}
            and (
                "ac_locked_titration_flash.py" in folded
                or (
                    "distributed_auto_calibration_queue.py" in folded
                    and " master " in f" {folded} "
                )
            )
        )
        if process.pid in excluded or marker not in folded or not is_campaign_process:
            continue
        rows.append(
            {"pid": process.pid, "name": process.info.get("name"), "command": command}
        )
    return rows


def _spawn(spec: ProcessSpec) -> subprocess.Popen:
    output = Path(spec.stdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("ab")
    try:
        return subprocess.Popen(
            list(spec.command),
            cwd=process_directory(),
            env=spec.env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        stream.close()


def launch(args: argparse.Namespace) -> int:
    role = args.role.lower()
    logs_root = Path(args.logs_root)
    preflight, seed_path = run_preflight(logs_root, role)
    specs = build_specs(
        role=role,
        logs_root=logs_root,
        queue_root=args.queue_root,
        seed_path=seed_path,
    )
    active = _active_processes()
    if active:
        raise RuntimeError(
            "Locked Titration campaign processes already active: "
            + ", ".join(str(row["pid"]) for row in active)
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preflight": preflight,
                    "processes": [
                        {
                            **asdict(spec),
                            "env": {
                                key: value
                                for key, value in spec.env.items()
                                if key.startswith("FFAC_")
                            },
                        }
                        for spec in specs
                    ],
                },
                indent=2,
            )
        )
        return 0

    pending = list(specs)
    running: dict[str, dict] = {}
    finished: dict[str, dict] = {}
    manifest_path = logs_root / f"launch_manifest_{socket.gethostname().upper()}.json"
    status_path = logs_root / f"supervisor_{socket.gethostname().upper()}.json"

    def start(spec: ProcessSpec, restarts: int = 0) -> None:
        process = _spawn(spec)
        running[spec.run] = {
            "spec": spec,
            "process": process,
            "restarts": restarts,
        }
        print(f"START {spec.run} pid={process.pid}", flush=True)

    while pending and len(running) < args.max_parallel:
        start(pending.pop(0))
        time.sleep(args.gpu_stagger_seconds)
    _atomic_json(
        manifest_path,
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "started_at_utc": _utc_now(),
            "host": socket.gethostname().upper(),
            "role": role,
            "processes": [asdict(spec) for spec in specs],
        },
    )

    while running or pending:
        for run, item in list(running.items()):
            process: subprocess.Popen = item["process"]
            exit_code = process.poll()
            if exit_code is None:
                continue
            spec: ProcessSpec = item["spec"]
            complete = _queue_complete(spec.queue)
            if not complete and item["restarts"] < args.max_restarts:
                restarts = item["restarts"] + 1
                print(
                    f"RESTART {run} exit={exit_code} count={restarts}",
                    flush=True,
                )
                time.sleep(10)
                start(spec, restarts=restarts)
                continue
            finished[run] = {
                "run": run,
                "exit_code": exit_code,
                "queue_complete": complete,
                "restarts": item["restarts"],
            }
            del running[run]
        while pending and len(running) < args.max_parallel:
            start(pending.pop(0))
            time.sleep(args.gpu_stagger_seconds)
        _atomic_json(
            status_path,
            {
                "schema": 1,
                "campaign_id": CAMPAIGN_ID,
                "updated_at_utc": _utc_now(),
                "host": socket.gethostname().upper(),
                "pending": [spec.run for spec in pending],
                "running": [
                    {
                        "run": run,
                        "pid": item["process"].pid,
                        "restarts": item["restarts"],
                        "queue": item["spec"].queue,
                    }
                    for run, item in running.items()
                ],
                "finished": list(finished.values()),
            },
        )
        if running or pending:
            time.sleep(args.supervisor_poll_seconds)
    failed = [row for row in finished.values() if not row["queue_complete"]]
    return 2 if failed else 0


def campaign_status(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "schema": 1,
                "campaign_id": CAMPAIGN_ID,
                "checked_at_utc": _utc_now(),
                "plan": validate_plan(),
                "active_processes": _active_processes(),
                "queues": [
                    _queue_status(f"{args.queue_root}_{run}") for run in RUNS
                ],
            },
            indent=2,
        )
    )
    return 0


def stop_campaign(args: argparse.Namespace) -> int:
    if psutil is None:
        raise RuntimeError(
            "psutil is unavailable in this session; stop the campaign from an "
            "ordinary desktop terminal or by process ID."
        )
    active = _active_processes()
    processes = []
    for row in active:
        try:
            processes.append(psutil.Process(int(row["pid"])))
        except psutil.NoSuchProcess:
            pass
    for process in sorted(processes, key=lambda item: len(item.parents())):
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    _, alive = psutil.wait_procs(processes, timeout=args.stop_timeout)
    for process in alive:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    remaining = _active_processes()
    print(
        json.dumps(
            {"stopped": len(active) - len(remaining), "remaining": remaining},
            indent=2,
        )
    )
    return 1 if remaining else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "stop", "plan"))
    parser.add_argument(
        "--role",
        choices=("moderskipet", "olav"),
        default=socket.gethostname().lower(),
    )
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT))
    parser.add_argument("--queue-root", default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--gpu-stagger-seconds", type=float, default=20.0)
    parser.add_argument("--supervisor-poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--stop-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "launch":
        raise SystemExit(launch(args))
    if args.action == "status":
        raise SystemExit(campaign_status(args))
    if args.action == "stop":
        raise SystemExit(stop_campaign(args))
    print(json.dumps(validate_plan(), indent=2))


if __name__ == "__main__":
    main()

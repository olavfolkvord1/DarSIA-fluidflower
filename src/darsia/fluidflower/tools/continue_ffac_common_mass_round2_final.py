"""One-shot operational continuation after disjoint frozen final-map shards.

Configuration is supplied at runtime. This controller never fits, freezes,
edits arrays, modifies a scientific module, or retries a failed stage.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


CONTRACT = "ffac-common-mass-round2-final-continuation-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def asset(path):
    return dict(path=str(path), sha256=sha(path), size_bytes=Path(path).stat().st_size)


def bound_json(record, base):
    path = Path(record["path"])
    if not path.is_absolute():
        path = Path(base) / path
    if sha(path) != record["sha256"]:
        raise ValueError(f"Source SHA changed: {path}")
    return read(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def process_state(pid):
    """Observe a local PID without starting or terminating any process."""
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False, None
    code = ctypes.c_ulong()
    ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code)); kernel.CloseHandle(handle)
    return bool(ok and code.value == 259), int(code.value) if ok else None


def exact_inventory(job, metadata, expected_runs, freeze_sha):
    """Check every categorical-source/time/area/freeze receipt, not RAW data."""
    expected = {(row["run"], int(row["frame_index"])): row for row in metadata["frames"]
        if row["run"] in set(expected_runs)}
    if len(expected) != sum(row["run"] in set(expected_runs) for row in metadata["frames"]):
        raise ValueError("Duplicate source frame in the bound metadata")
    actual = {}; sources = []
    for path in sorted((Path(job) / "production").glob("*/frame_*/receipt.json")):
        row = read(path); key = row.get("run"), row.get("frame_index")
        if key in actual or key not in expected:
            raise ValueError(f"Duplicate or unexpected final-map receipt: {path}")
        source = expected[key]
        if row.get("complete") is not True or row.get("calibration_freeze_sha256") != freeze_sha:
            raise ValueError(f"Incomplete or differently frozen map: {path}")
        for field, value in (("source_sha256", source["source_sha256"]),
                ("phase_sha256", source["labels"]["sha256"]),
                ("pixel_area_sha256", source["area_map"]["sha256"]),
                ("elapsed_h", source["elapsed_h"])):
            if row.get(field) != value:
                raise ValueError(f"Final map changed {field}: {path}")
        if row.get("bottom_mass_allowed") is not False or row.get("no_map_mass_rescaling") is not True:
            raise ValueError(f"Final map lost the locked mass-support policy: {path}")
        for record in row.get("assets", {}).values():
            target = path.parent / record["path"]
            if not target.is_file():
                raise ValueError(f"Committed receipt lacks an asset: {target}")
        actual[key] = row; sources.append(asset(path))
    if set(actual) != set(expected):
        raise ValueError(f"Incomplete final-map inventory: {len(actual)}/{len(expected)}")
    return expected, sources


def successful_shard_receipts(job, assignments, expected):
    """A null shell exit is acceptable only with full immutable completion."""
    matches = {}; wanted_counts = {run: sum(key[0] == run for key in expected) for run in {key[0] for key in expected}}
    for path in sorted((Path(job) / "stage_receipts").glob("final-maps_*.json")):
        record = read(path)
        if record.get("contract") != "ffac-common-mass-round2-campaign-v1" or record.get("stage") != "final-maps" or record.get("failures"):
            continue
        completed = record.get("completed", [])
        if len({row.get("run") for row in completed}) != len(completed):
            continue
        for host, runs in assignments.items():
            if {row.get("run") for row in completed} != set(runs):
                continue
            if all(row.get("frame_count") == wanted_counts[row["run"]] for row in completed):
                matches[host] = asset(path)
    if set(matches) != set(assignments):
        raise ValueError("Both disjoint run shards require complete successful final-maps stage receipts")
    return matches


class Controller:
    def __init__(self, config, *, observe=process_state, sleep=time.sleep, launch=None):
        self.config = config; self.job = Path(config["job"]); self.ops = Path(config["ops"])
        self.observe = observe; self.sleep = sleep; self.launch = launch
        self.check_attempted = False
        self.ops.mkdir(parents=True, exist_ok=True); (self.ops / "logs").mkdir(exist_ok=True)

    def status(self, stage, **details):
        row = dict(contract=CONTRACT, stage=stage, utc=utc(), pid=os.getpid(), **details)
        temporary = self.ops / f".status_{uuid.uuid4().hex}.partial"
        temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
        temporary.replace(self.ops / "controller_status.json")
        with (self.ops / "controller_events.ndjson").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def execute(self, name, command):
        out_path = self.ops / "logs" / (name + ".out.log")
        err_path = self.ops / "logs" / (name + ".err.log")
        if out_path.exists() or err_path.exists():
            raise FileExistsError(f"Stage already launched; no retry: {name}")
        self.status(name, command=command)
        if self.launch is not None:
            code = self.launch(name, command)
        else:
            with out_path.open("x", encoding="utf-8") as out, err_path.open("x", encoding="utf-8") as err:
                flags = 0x08000000 | 0x00004000 if os.name == "nt" else 0
                child = subprocess.Popen(command, stdout=out, stderr=err, creationflags=flags)
                self.status(name, child_pid=child.pid, command=command)
                code = child.wait()
        self.status(name + "_finished", exit_code=code)
        return code

    def stage(self, name, workers=1):
        if name not in {"check", "produce"}:
            raise ValueError("This continuation may only check or produce the frozen calibration")
        return self.execute(name, [sys.executable, "-m", "darsia.fluidflower.tools.run_ffac_common_mass_round2",
            *self.config["core_arguments"], "--stage", name, "--workers", str(workers), "--backend", self.config.get("backend", "auto")])

    def report(self, register=False):
        tool = self.config["report_tool"]
        if sha(tool["path"]) != tool["sha256"]:
            raise ValueError("The absolute runtime report companion changed")
        command = [sys.executable, tool["path"], "--job", str(self.job),
            "--workflow-binding", self.config["workflow_binding"],
            "--destination", str(self.ops / ("report_complete" if register else "report_stop"))]
        if self.config.get("overflow_inputs"):
            command += ["--overflow-inputs", self.config["overflow_inputs"]]
        if register:
            command.append("--register")
        return self.execute("report_registered" if register else "report_stop", command)

    def stop(self, reason, **details):
        production_started = details.pop("full_production_started", False)
        self.status("specified_stop", reason=reason, automatic_retry=False, **details)
        report_error = None
        if self.check_attempted:
            try:
                if self.report() != 0:
                    report_error = "Saved report companion returned a non-zero exit"
            except Exception as exc:
                report_error = repr(exc)
        write_new(self.ops / "specified_stop.json", dict(contract=CONTRACT, utc=utc(), reason=reason,
            automatic_retry=False, full_production_started=production_started, frozen_calibration_changed=False,
            report_error=report_error, report_attempted=self.check_attempted, **details))
        return 0 if report_error is None else 1

    def wait_shards(self):
        local_path = Path(self.config["main_process_json"])
        remote_path = Path(self.config["remote_done_json"])
        while not local_path.is_file():
            self.status("waiting_for_main_process_record"); self.sleep(5)
        local = read(local_path); last_notice = 0.
        if self.config.get("verify_process_records"):
            if local.get("stage") != "final-maps" or set(local.get("runs", [])) != set(self.config["assignments"]["main"]):
                raise ValueError("Main process record differs from its final-map assignment")
        while True:
            alive, code = self.observe(local["pid"])
            main_done_path = Path(self.config["main_done_json"]) if self.config.get("main_done_json") else None
            main_done = read(main_done_path) if main_done_path and main_done_path.is_file() else None
            remote = read(remote_path) if remote_path.is_file() else None
            if code not in (None, 0, 259):
                raise RuntimeError(f"Main final-map shard exited {code}; no automatic retry")
            if remote is not None and remote.get("exit_code") not in (None, 0):
                raise RuntimeError(f"Olav final-map shard exited {remote.get('exit_code')}; no automatic retry")
            if main_done is not None and main_done.get("exit_code") not in (None, 0):
                raise RuntimeError(f"Main final-map shard reported exit {main_done.get('exit_code')}; no automatic retry")
            if main_done is not None and main_done.get("pid") != local["pid"]:
                raise ValueError("Main completion PID differs from the monitored process")
            if remote is not None and self.config.get("remote_process_json"):
                process_path = Path(self.config["remote_process_json"])
                process = read(process_path)
                if process.get("stage") != "final-maps" or set(process.get("runs", [])) != set(self.config["assignments"]["olav"]) or process.get("pid") != remote.get("pid"):
                    raise ValueError("Remote process/completion record differs from its final-map assignment")
            if not alive and remote is not None:
                evidence = dict(main_process=asset(local_path), main_observed_exit_code=code,
                    remote_done=asset(remote_path), remote_reported_exit_code=remote.get("exit_code"))
                if main_done is not None:
                    evidence["main_done"] = asset(main_done_path)
                return evidence
            if time.monotonic() - last_notice >= 60:
                self.status("waiting_for_disjoint_final_map_shards", main_running=alive, remote_done=remote is not None)
                last_notice = time.monotonic()
            self.sleep(5)

    def run(self):
        campaign_path = self.job / "campaign_binding.json"
        if sha(campaign_path) != self.config["campaign_binding_sha256"]:
            raise ValueError("The immutable campaign binding changed")
        freeze = self.job / "calibration/freeze/frozen_calibration.json"
        if sha(freeze) != self.config["frozen_sha256"]:
            raise ValueError("The already frozen calibration changed")
        campaign = read(campaign_path)
        plan = bound_json(campaign["input_plan"], self.job)
        metadata = bound_json(plan["source_metadata"], Path(campaign["input_plan"]["path"]).parent)
        expected_runs = set(plan["split"]["held_out_runs"]) | set(plan["split"]["descriptive_runs"])
        assignments = self.config["assignments"]
        main = set(assignments["main"]); remote = set(assignments["olav"])
        if self.config.get("assignment_asset"):
            assignment = bound_json(self.config["assignment_asset"], self.ops)
            if (assignment.get("frozen_sha256") != self.config["frozen_sha256"]
                    or set(assignment["main"]["runs"]) != main or set(assignment["olav"]["runs"]) != remote):
                raise ValueError("The source-bound operational assignment changed")
        if main & remote or main | remote != expected_runs:
            raise ValueError("Final-map shards must be disjoint and cover precisely the 12 held-out runs plus AC38")
        if "--stage" in self.config["core_arguments"] or "--runs" in self.config["core_arguments"]:
            raise ValueError("Runtime core arguments must not introduce a fit or run subset")
        evidence = self.wait_shards()
        expected, receipts = exact_inventory(self.job, metadata, expected_runs, self.config["frozen_sha256"])
        if len(expected) != self.config["expected_final_frames"]:
            raise ValueError("Final-map cohort frame count differs from the predeclared inventory")
        stages = successful_shard_receipts(self.job, assignments, expected)
        write_new(self.ops / "final_map_completion.json", dict(contract=CONTRACT, complete=True,
            frozen_sha256=self.config["frozen_sha256"], source_bound_frames=len(expected),
            frame_receipts=receipts, shard_stage_receipts=stages, shell_evidence=evidence,
            null_shell_exit_accepted_only_with_exact_completion=True))
        self.check_attempted = True
        code = self.stage("check", 1)
        check_path = self.job / "final_check.json"
        if not check_path.is_file():
            return self.stop("check_did_not_produce_a_frozen_result", check_exit_code=code)
        checked = read(check_path)
        if checked.get("freeze", {}).get("sha256") != self.config["frozen_sha256"]:
            return self.stop("final_check_frozen_binding_changed", check_exit_code=code)
        if code not in (0, 2):
            return self.stop("final_check_process_error", check_exit_code=code, final_check=asset(check_path))
        if code == 2 or checked.get("full_production_allowed") is not True or checked.get("outcome") != "pass":
            return self.stop("specified_final_check_stop", check_exit_code=code,
                check_outcome=checked.get("outcome"), final_check=asset(check_path))
        self.status("frozen_final_check_passed", final_check=asset(check_path))
        code = self.stage("produce", int(self.config.get("production_workers", 16)))
        if code != 0:
            return self.stop("full_production_process_error", produce_exit_code=code, full_production_started=True)
        all_runs = set(plan["split"]["production_runs"])
        expected, receipts = exact_inventory(self.job, metadata, all_runs, self.config["frozen_sha256"])
        if len(expected) != self.config["expected_production_frames"]:
            raise ValueError("Full-production frame count differs from the sealed inventory")
        code = self.report(register=True)
        if code != 0:
            self.status("report_registration_error", exit_code=code, automatic_retry=False)
            return code
        write_new(self.ops / "complete.json", dict(contract=CONTRACT, complete=True, utc=utc(),
            frozen_sha256=self.config["frozen_sha256"], frame_count=len(expected), run_count=len(all_runs),
            frame_receipts=receipts, default_changed=False, scientific_parameters_changed=False))
        self.status("complete", frames=len(expected), runs=len(all_runs), default_changed=False)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = read(args.config)
    controller = Controller(config)
    write_new(Path(config["ops"]) / "controller_started.json", dict(contract=CONTRACT, pid=os.getpid(),
        started_utc=utc(), controller=asset(__file__), configuration=asset(args.config), automatic_retry=False))
    try:
        # Import only the immutable snapshot's code verifier, not a frame loader.
        verifier = importlib.import_module("darsia.fluidflower.tools.run_ffac_common_mass_round2")
        verifier.verify_worker_code(Path(config["job"]) / "campaign_binding.json")
        return controller.run()
    except Exception as exc:
        controller.status("operational_error", error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False)
        if not (controller.ops / "specified_stop.json").is_file():
            controller.stop("operational_error", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

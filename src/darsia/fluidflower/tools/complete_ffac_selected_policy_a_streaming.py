"""Guarded continuation of real phase completion to review-only Studio."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from darsia.fluidflower.tools.complete_ffac_selected_policy_a import (
    asset, bound, now, phase_completed, write_new,
)


def display_completed(record, plan_descriptor, runs, expected=1603, *, stage="prepare"):
    if (record.get("contract") != "ffac-selected-display-preparation-operation-v1"
            or record.get("plan") != plan_descriptor or record.get("stage") != stage
            or record.get("complete") is not True or record.get("stopped") is not False
            or record.get("all_requested_displays_complete") is not True
            or set(record.get("runs", [])) != set(runs)
            or len(record.get("runs", [])) != len(runs)
            or record.get("counts", {}).get("total") != expected
            or record.get("counts", {}).get("completed") != expected
            or record.get("counts", {}).get("missing_sources") != 0
            or any(record.get(key) is not False for key in (
                "model_inference", "scientific_inputs_changed", "maps_changed", "labels_changed"))):
        raise ValueError("Actual complete source-bound photographic coverage is required")
    return True


class Controller:
    def __init__(self, config_path):
        self.descriptor = asset(config_path)
        self.config = bound(self.descriptor)
        self.plan = bound(self.config["phase_plan"])
        self.ops = Path(self.config["operations"])
        if self.ops.exists():
            raise FileExistsError("Completion requires a new operational namespace")
        if socket.gethostname().casefold() != self.config["approved_main_host"].casefold():
            raise ValueError("Completion is restricted to Main")
        self.stop = Path(self.plan["execution_operation"]["shared_stop_path"])
        if self.stop.exists():
            raise ValueError("Shared stop exists; no completion startup")
        self.records = []
        self.ops.mkdir(parents=True)
        write_new(self.ops/"process.json", dict(contract="ffac-streaming-review-controller-process-v1",
            pid=os.getpid(), host=socket.gethostname(), started_utc=now(), config=self.descriptor,
            phase_plan=self.config["phase_plan"], phase_inference=False, scorer_pending=True, round3_allowed=False))

    def wait_file(self, path):
        deadline = time.monotonic()+self.config.get("wait_hours", 72)*3600
        while not Path(path).is_file():
            if self.stop.exists():
                raise ValueError("Shared stop; downstream work is blocked")
            if time.monotonic() > deadline:
                raise TimeoutError("Waiting for " + str(path))
            time.sleep(self.config.get("poll_seconds", 30))
        if self.stop.exists():
            raise ValueError("Shared stop; downstream publication is blocked")
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))

    def call(self, name, module, arguments, runtime):
        if self.stop.exists():
            raise ValueError("Shared stop before " + name)
        folder = self.ops/name
        folder.mkdir()
        environment = dict(os.environ, PYTHONPATH=runtime)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = "1"
        command = [self.config["python"], "-m", module, *map(str, arguments)]
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
        with (folder/"stdout.log").open("wb") as out, (folder/"stderr.log").open("wb") as err:
            child = subprocess.Popen(command, env=environment, stdout=out, stderr=err, creationflags=flags)
            write_new(folder/"process.json", dict(pid=child.pid, started_utc=now(), command=command,
                runtime_src=runtime, phase_plan=self.config["phase_plan"], phase_inference=False))
            while child.poll() is None:
                time.sleep(10)
            row = dict(stage=name, exit_code=child.returncode, finished_utc=now())
            write_new(folder/"done.json", row)
            self.records.append(row)
            if child.returncode != 0 or self.stop.exists():
                raise RuntimeError("Stage stopped without retry: " + name)

    def execute(self):
        config = self.config
        root = config["candidate"]
        plan = config["phase_plan"]["path"]
        identity = self.plan["identity_gate"]["path"]
        phase_completed(self.wait_file(config["phase_supervisor_done"]), config["phase_plan"], self.plan["runs"])
        producer, review = config["producer_runtime_src"], config["review_runtime_src"]
        for runtime, manifest in ((producer, config["producer_runtime_manifest"]), (review, config["review_runtime_manifest"])):
            declaration = bound(manifest)
            if declaration.get("runtime_src") != runtime or declaration.get("complete") is not True:
                raise ValueError("Completion runtime differs from its immutable manifest")
        completion = Path(root)/"policy_a_completion_v1.json"
        report = Path(root)/"policy_a_saved_report_v1"
        export = Path(root)/"claude_prediction_export_v1"
        finisher = "darsia.fluidflower.tools.finish_ffac_selected_policy_a_streaming"
        common = ["--plan", plan, "--root", root, "--identity-gate", identity]
        self.call("collect", finisher, [*common, "--stage", "collect", "--destination", completion], producer)
        self.call("spatial_report", finisher, [*common, "--stage", "report", "--destination", report], producer)
        self.call("claude_export", finisher, [*common, "--stage", "export", "--report", report/"report.json", "--destination", export], producer)
        display_completed(self.wait_file(config["photo_preparation_report"]), config["phase_plan"], self.plan["runs"])
        display_report = self.ops/"display_finalization.json"
        self.call("display_finalize", "darsia.fluidflower.tools.render_ffac_selected_display_preparations",
            ["--root", root, "--plan", plan, "--workers", config["finalization_workers"],
             "--stage", "finalize", "--batch-size", "64", "--backend", "cpu", "--stop-path", self.stop,
             "--report", display_report], review)
        display_completed(bound(asset(display_report)), config["phase_plan"], self.plan["runs"], stage="finalize")
        gate = Path(root)/"selected_policy_a_review_completion_v1.json"
        self.call("review_seal", "darsia.fluidflower.tools.seal_ffac_selected_policy_a_streaming_review",
            ["--plan", plan, "--candidate", root, "--completion", completion,
             "--report", report/"report.json", "--destination", gate], review)
        self.call("studio_package", "darsia.fluidflower.tools.build_ffac_selected_policy_a_streaming_review",
            ["--candidate", root, "--phase-plan", plan, "--accepted-gate", gate], review)
        self.call("studio_register", "darsia.fluidflower.tools.register_ffac_selected_policy_a_studio_candidate",
            ["--candidate", root, "--results", config["results_root"]], review)
        return dict(complete=True, phase_completion=asset(completion), report=asset(report/"report.json"),
            claude_export=asset(export/"export_receipt.json"), review_seal=asset(gate),
            registered_nondefault=True, claude_scorer_gate_pending=True, round3_allowed=False,
            mass_admission=False, bottom_mass_allowed=False, scorer_executed=False, default_changed=False)

    def run(self):
        try:
            result = self.execute()
        except Exception as error:
            result = dict(complete=False, error=repr(error), registered_nondefault=False,
                claude_scorer_gate_pending=True, round3_allowed=False, scorer_executed=False)
        result.update(contract="ffac-streaming-complete-to-review-controller-v1", finished_utc=now(),
            config=self.descriptor, phase_plan=self.config["phase_plan"], stages=self.records, code=asset(__file__))
        write_new(self.ops/"controller_done.json", result)
        print(json.dumps(result), flush=True)
        return 0 if result["complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return Controller(parser.parse_args(argv).config).run()


if __name__ == "__main__":
    raise SystemExit(main())

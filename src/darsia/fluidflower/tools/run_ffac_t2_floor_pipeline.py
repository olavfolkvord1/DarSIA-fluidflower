"""Detached sequential stages for one immutable G3-T2/floor candidate job."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .refit_ffac_g3_downstream import JobLease, atomic_json, read_json, sha256_file, utc_now, below_normal_priority


def write_telemetry(path, value):
    """Only live telemetry is replaced; source/model receipts stay immutable."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(__import__('json').dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    config = read_json(args.config)
    root = Path(config["job_root"])
    digest = sha256_file(args.config)
    exit_path = root / "exit_receipt.json"
    if exit_path.exists() and read_json(exit_path).get("status") == "succeeded":
        raise ValueError("This pipeline is already complete")
    def progress(stage, **extra):
        value = dict(contract="ffac-g3-t2-floor-pipeline-v1", config_sha256=digest, pid=os.getpid(),
            stage=stage, updated_utc=utc_now(), complete=False, **extra)
        write_telemetry(root / "heartbeat.json", value)
        print(__import__('json').dumps(value), flush=True)
    below_normal_priority()
    with JobLease(root, digest):
        try:
            pixel_job = Path(config["pixel_job"])
            fit_receipt = pixel_job / "work/pixel_fit/fit_receipt.json"
            while not fit_receipt.is_file():
                completed = pixel_job / "exit_receipt.json"
                if completed.is_file() and read_json(completed).get("status") not in ("succeeded", "completed"):
                    raise RuntimeError("Pixel job failed; no automatic retry: " + str(completed))
                progress("waiting_for_pixel_fit", pixel_job=str(pixel_job))
                time.sleep(20)
            fit = read_json(fit_receipt)
            if fit.get("converged") is not True or sha256_file(fit["model_path"]) != fit["model_sha256"]:
                raise ValueError("Pixel fit is unconverged or changed")
            progress("pixel_complete", model_sha256=fit["model_sha256"])
            for stage in config["stages"]:
                receipt = Path(stage["receipt"])
                if receipt.is_file():
                    if not args.resume:
                        raise ValueError("Existing stage receipt requires exact --resume: " + stage["name"])
                    progress(stage["name"] + "_import", receipt=str(receipt))
                    continue
                command = [config["python"], "-I", config["bootstrap"], stage["module"], *stage["args"]]
                stdout_path = root / (stage["name"] + ".stdout.log")
                stderr_path = root / (stage["name"] + ".stderr.log")
                # Each new invocation is logged; failed partial stage data stay in place.
                mode = "ab" if args.resume else "xb"
                with stdout_path.open(mode) as stdout, stderr_path.open(mode) as stderr:
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                        creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0)
                    while process.poll() is None:
                        progress(stage["name"], child_pid=process.pid, stdout=str(stdout_path), stderr=str(stderr_path))
                        time.sleep(20)
                if process.returncode or not receipt.is_file():
                    raise RuntimeError(f"{stage['name']} failed (exit {process.returncode}); inspect {stderr_path}; no automatic retry")
                progress(stage["name"] + "_complete", receipt=str(receipt), receipt_sha256=sha256_file(receipt))
            atomic_json(exit_path, dict(status="succeeded", complete=True, finished_utc=utc_now(), config_sha256=digest,
                candidate=config["candidate"], automatic_activation=False, mass_eligible=False, bottom_mass_allowed=False))
        except BaseException as error:
            atomic_json(exit_path, dict(status="failed", complete=False, failed_utc=utc_now(), config_sha256=digest,
                error=repr(error), automatic_retry=False, mass_eligible=False))
            raise
    return 0


if __name__ == "__main__":
    main()

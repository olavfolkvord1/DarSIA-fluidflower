"""Write and SHA-copy the stopped campaign's saved-input diagnosis only."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def asset(path):
    p = Path(path)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def read(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Saved metadata SHA changed: {path}")
    return json.loads(raw)


def write_new(path, text):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--local", required=True)
    args = parser.parse_args()
    job = Path(args.job)
    folder = job / "diagnostics/identity_baseline_array_stop_v1"
    comparison = read(folder / "comparison.json", "ec125cd7d8ddc01c153fe151ad8568c1507742c847537b3f03916c6a199e0dc3")
    metadata_path = job / "diagnostics/identity_baseline_representation_metadata_stop_v1/receipt.json"
    metadata = read(metadata_path, "fdd5f668c89df84c96a5d241921c9c81893327a2c0da6e84c02ad65231ac02ba")
    by_run = {row["run"]: row for row in metadata["runs"]}
    table = []
    for row in comparison["rows"]:
        q = row["comparison"]
        m = by_run[row["run"]]
        if not (m["same_acquisition"] and m["reference_native_cache_matches_historical_reference"]):
            raise ValueError("Source-role evidence inconsistent")
        first = q["first_difference"]
        table.append(dict(run=row["run"], historical_valid_pixels=q["original_valid_pixels"],
            changed_pixels=q["changed_pixels"], changed_components=q["changed_components"],
            nonfinite_current_components=q["current_nonfinite_changed_components"],
            first_channel=first["channel"], first_y=first["y"], first_x=first["x"],
            original_value=first["original_value"], current_value=first["current_value"],
            max_finite_absolute_log_delta=q["max_absolute_log_delta"],
            source_sha256=row["baseline_source_sha256"],
            frame0_scientific_path=row["current_log"]["path"], frame0_scientific_sha256=row["current_log"]["sha256"],
            original_reference_path=row["original_log"]["path"], original_reference_sha256=row["original_log"]["sha256"],
            frame0_parent_cache_sha256=m["current_frame0"]["parent_cache"]["log_linear_rgb"]["sha256"],
            reference_parent_cache_sha256=m["independently_prepared_reference"]["parent_cache"]["log_linear_rgb"]["sha256"],
            frame0_exact_B=m["current_frame0"]["exact_B"],
            reference_exact_B=m["independently_prepared_reference"]["exact_B"]))
    with (folder / "baseline_differences.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    lines = [
        "REQUIRED CPU IDENTITY: STOPPED BEFORE 11 BASELINE CONSTRUCTORS COULD INFER",
        "",
        "120 of 153 distinct required Studio frames were saved and pass all required comparisons.",
        "AC19 through frame 1 and both full 21-frame AC18/AC52 histories pass.",
        "The other 33 required frames have no result: 11 runs could not initialize the baseline.",
        "No required stored-array comparison failed among the 120 completed frames.",
        "The frozen producer did not persist Future-to-run association; the first scheduler failure's run remains unknown.",
        "",
        "READ-ONLY INPUT DIAGNOSIS",
        "All 11 missing runs independently fail the exact constructor assertion on existing arrays.",
        "The constructor loads the selected frame-0 native log as baseline_native and compares it against",
        "the separately prepared original native_references/pre_injection log on that reference's valid support.",
        "These are two historical representations of the same camera acquisition, with different bound log SHAs.",
        "The wrapper already carries the independently prepared reference in baseline_extended_scientific.",
        "The metadata audit shows that reference's native parent cache matches the original reference exactly.",
        "Thus the current constructor compares different source roles; shared acquisition identity is insufficient for bit equality.",
        "",
        "Both compared arrays have identical float32 CHW shape 3 x 2979 x 5472.",
        "Every original bool validity mask exactly matches finite original reference support.",
        "AC28 has 10 differing valid pixels / 29 components; 18 current components are nonfinite.",
        "Its first difference is channel 0, y=169, x=4054: 3.1267099380493164 -> 3.1075382232666016.",
        "Its maximum finite absolute log delta is 0.42114198207855225.",
        "The other ten runs differ at every historical-valid pixel, with finite log differences beyond representation/encoding roundoff.",
        "Metadata also separates exact_B/reference preparation from the current-frame branch in ten runs.",
        "AC28 has true exact_B in both roles but distinct near-identity versus literal-identity registration matrices.",
        "These are observed input/convention differences; no model or counterfactual execution isolated each operator's contribution.",
        "",
        "BOUNDARIES",
        "This diagnosis reads arrays only. It neither constructs an evaluator nor runs inference, changes a convention, or retries.",
        "Full NAS array SHAs were not re-hashed: cited asset SHAs are exact producer-bound descriptors; small metadata SHAs were verified.",
        "All previous jobs, snapshots, arrays, maps, labels and diagnostics remain unchanged.",
        "Policy A, selected-frame inference, registration, scoring and mass work remain blocked.",
        "The separate Olav probe failed its host equivalence and is not part of the main-machine acceptance evidence.",
        "",
        "PER-RUN READ-ONLY COMPARISON (pixels / components / max finite absolute log delta):",
    ]
    lines.extend(f"{row['run'].upper()}: {row['changed_pixels']:,} / {row['changed_components']:,} / {row['max_finite_absolute_log_delta']:.10g}" for row in table)
    write_new(folder / "SUMMARY.txt", "\n".join(lines) + "\n")
    proof = dict(contract="ffac-required-identity-baseline-diagnosis-report-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),code=asset(__file__),
        input_comparison=asset(folder / "comparison.json"), representation_metadata=asset(metadata_path),
        completed_required_frames=120,total_required_frames=153,required_array_comparison_failures=0,
        missing_runs=list(by_run),first_failed_scheduler_run=None,
        model_executed=False,arrays_modified=False,full_array_sha_recomputed=False,
        outputs={name:asset(folder / name) for name in ("comparison.json","baseline_differences.csv","SUMMARY.txt")})
    write_new(folder / "report_receipt.json", json.dumps(proof, indent=2) + "\n")
    local = Path(args.local)
    local.mkdir(parents=True, exist_ok=False)
    copies = []
    for name in ("comparison.json","baseline_differences.csv","SUMMARY.txt","report_receipt.json"):
        source = folder / name; target = local / name
        shutil.copyfile(source, target)
        a,b = asset(source),asset(target)
        if a["sha256"] != b["sha256"]: raise ValueError("Copy SHA failed")
        copies.append(dict(source=a,target=b))
    write_new(local / "copy_verification.json", json.dumps(dict(copies=copies,all_byte_identical=True), indent=2)+"\n")
    print(json.dumps(dict(local=str(local),receipt=asset(folder / "report_receipt.json"),copies_verified=len(copies))))


if __name__ == "__main__": main()

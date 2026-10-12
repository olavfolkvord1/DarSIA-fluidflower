"""Stage exact dependencies and inspect headers while scientific jobs are held.

No image payload is decoded or interpreted. Existing files may be reused only
when their exact producer SHA matches; they are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bound_json(descriptor):
    if sha(descriptor["path"]) != descriptor["sha256"]:
        raise ValueError("Changed readiness metadata: " + descriptor["path"])
    return json.loads(Path(descriptor["path"]).read_text(encoding="utf-8-sig"))


def stage(item):
    target = Path(item["canonical_path"])
    expected = item["source"]["sha256"]
    if target.is_file():
        if sha(target) != expected:
            raise ValueError("Existing dependency must not be overwritten: " + str(target))
        copied = False
    else:
        source = Path(item["shared_copy"]["path"])
        if sha(source) != expected:
            raise ValueError("Shared dependency does not match its producer")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("." + target.name + ".readiness-" + str(os.getpid()))
        with source.open("rb") as stream, temporary.open("xb") as output:
            shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
        if sha(temporary) != expected:
            raise ValueError("Transferred dependency differs")
        os.rename(temporary, target)
        copied = True
    return dict(path=str(target), sha256=expected, copied=copied,
                size_bytes=target.stat().st_size)


def header(descriptor, expected_kind):
    from numpy.lib import format as npformat
    path = Path(descriptor["path"])
    with path.open("rb") as stream:
        version = npformat.read_magic(stream)
        shape, fortran, dtype = npformat._read_array_header(stream, version)
    expected_shape = (3, 2979, 5472) if expected_kind == "log" else (2979, 5472)
    expected_dtype = "float32" if expected_kind == "log" else "bool"
    if shape != expected_shape or str(dtype) != expected_dtype or fortran:
        raise ValueError("Original native array header differs: " + str(path))
    return dict(path=str(path), declared_sha256=descriptor["sha256"],
                size_bytes=path.stat().st_size, shape=list(shape), dtype=str(dtype),
                science_payload_read=False, SHA_verified_by_scientific_session_loader=True)


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config.get("readiness_only") is not True or config.get("scientific_worker_dispatch_allowed") is not False:
        raise ValueError("This process is restricted to staging and metadata headers")
    started = time.perf_counter()
    dependencies = bound_json(config["dependencies"])
    native = bound_json(config["native_closure"])
    staged = [stage(item) for item in dependencies["dependencies"]]
    arrays, probability_assets = {}, {}
    for run in config["runs"]:
        row = native["runs"][run]
        records = [row["baseline"]] + [a["native_input"] for a in row["anchors"] if a["native_input"]]
        for record in records:
            for key, kind in [("log_linear_rgb", "log"), ("valid_mask", "mask"), ("photographed_mask", "mask")]:
                descriptor = record[key]
                if descriptor["path"] not in arrays:
                    arrays[descriptor["path"]] = header(descriptor, kind)
            if record.get("temporal_ring") and not Path(record["temporal_ring"]["path"]).is_file():
                raise FileNotFoundError("Declared historical temporal ring is unavailable")
        for anchor in row["anchors"]:
            if anchor["probability_input"]:
                item = anchor["probability_input"]["extra_masks"]
                if not Path(item["path"]).is_file():
                    raise FileNotFoundError("Original exported photo mask is unavailable")
                probability_assets[item["path"]] = dict(path=item["path"], sha256=item["sha256"])
    result = dict(contract="ffac-selected-corrected-remote-readiness-v1", complete=True,
        dependencies=config["dependencies"], native_closure=config["native_closure"], runs=config["runs"],
        assets=staged, copied_count=sum(item["copied"] for item in staged),
        copied_bytes=sum(item["size_bytes"] for item in staged if item["copied"]),
        checked_native_headers=list(arrays.values()), checked_header_count=len(arrays),
        original_exported_photo_masks=list(probability_assets.values()),
        seconds=time.perf_counter()-started, science_payload_read=False,
        scientific_jobs_started=False, existing_scientific_stop_preserved=config["preserved_stop"],
        old_files_changed=False)
    path = Path(config["result_path"])
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"assets", "checked_native_headers", "original_exported_photo_masks"}}, indent=2))


if __name__ == "__main__":
    main()

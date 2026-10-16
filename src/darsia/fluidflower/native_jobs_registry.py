"""Local UI associations for explicitly selected UNC jobs/adopted plans.

No NAS discovery, worker, producer, numerical code or series/config write.
Only this UI-owned local registry is saved after an explicit user action.
"""

from copy import deepcopy
import hashlib
import json
import ntpath
import os
from pathlib import Path
import tomllib
import uuid

from .pipeline.job_queue import descriptor, name, read_json, require_unc


def _path(value):
    return ntpath.normcase(ntpath.normpath(str(value)))


def _asset(path):
    path = Path(path)
    before = descriptor(path)
    raw = path.read_bytes()
    if (
        len(raw) != before["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != before["sha256"]
    ):
        raise ValueError("Selected metadata changed while read")
    return raw, before


class JobsUIRegistry:
    def __init__(self, root):
        self.root = Path(root)

    def _file(self, series_root, series_id):
        identity = json.dumps(
            [_path(series_root), str(series_id)], ensure_ascii=False
        ).encode()
        return self.root / (hashlib.sha256(identity).hexdigest() + ".json")

    def context(self, series_root, series_id):
        path = self._file(series_root, series_id)
        if not path.exists():
            return dict(
                format="darsia-native-jobs-registry-v1",
                series_root=str(series_root),
                series_id=str(series_id),
                jobs=[],
                plans=[],
                active_job=None,
            )
        doc = read_json(path)
        if (
            doc.get("format") != "darsia-native-jobs-registry-v1"
            or _path(doc.get("series_root", "")) != _path(series_root)
            or doc.get("series_id") != series_id
        ):
            raise ValueError("Local Jobs registration belongs to another series")
        if not isinstance(doc.get("jobs"), list) or not isinstance(
            doc.get("plans"), list
        ):
            raise ValueError("Malformed local Jobs registry")
        return deepcopy(doc)

    def _save(self, series_root, series_id, mutation):
        path = self._file(series_root, series_id)
        before = path.read_bytes() if path.exists() else None
        doc = self.context(series_root, series_id)
        mutation(doc)
        data = (
            json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            "." + path.name + "." + uuid.uuid4().hex + ".partial"
        )
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        current = path.read_bytes() if path.exists() else None
        if current != before:
            raise RuntimeError(
                "Local Jobs registry changed; registration not replaced. Pending bytes: "
                + str(temporary)
            )
        os.replace(temporary, path)
        return self.context(series_root, series_id)

    def register_job(
        self, series_root, series_id, job_root, *, label="", activate=True
    ):
        require_unc(str(job_root))
        root = Path(job_root)
        raw, bound = _asset(root / "job.json")
        doc = read_json(root / "job.json")
        if descriptor(root / "job.json") != bound:
            raise ValueError("Selected job metadata changed while parsed")
        runs = doc.get("expected_runs")
        if (
            not isinstance(runs, list)
            or not runs
            or len({name(run) for run in runs}) != len(runs)
        ):
            raise ValueError(
                "An existing job with exact distinct expected_runs is required"
            )
        if doc.get("precision") not in {"float32", "float64"}:
            raise ValueError("Existing job precision is not explicit")
        row = dict(
            root=str(root),
            name=str(label).strip() or root.name,
            series_label="UI association with " + str(series_id),
            job=bound,
        )

        def change(state):
            prior = next(
                (item for item in state["jobs"] if _path(item["root"]) == _path(root)),
                None,
            )
            if prior is not None and prior["job"] != bound:
                raise ValueError("The registered immutable job metadata changed")
            if prior is None:
                state["jobs"].append(row)
            if activate:
                state["active_job"] = str(root)

        return self._save(series_root, series_id, change)

    def register_plan(self, series_root, series_id, path):
        require_unc(str(path))
        raw, bound = _asset(path)
        doc = tomllib.loads(raw.decode("utf-8-sig"))
        if (
            doc.get("format") != "darsia-series-v1"
            or doc.get("adopted_for_execution") is not True
        ):
            raise ValueError(
                "Choose an already adopted darsia-series-v1 execution plan"
            )
        if doc.get("job", {}).get("precision") != "float32":
            raise ValueError("New production execution plans require float32")
        row = dict(bound, label=str(doc.get("job_id") or Path(path).stem))

        def change(state):
            prior = next(
                (item for item in state["plans"] if _path(item["path"]) == _path(path)),
                None,
            )
            if (
                prior is not None
                and {key: prior[key] for key in ("path", "sha256", "size_bytes")}
                != bound
            ):
                raise ValueError("The registered immutable execution plan changed")
            if prior is None:
                state["plans"].append(row)

        return self._save(series_root, series_id, change)

    def set_active(self, series_root, series_id, job_root):
        def change(state):
            if not any(_path(row["root"]) == _path(job_root) for row in state["jobs"]):
                raise ValueError("Select an explicitly registered job")
            state["active_job"] = str(job_root)

        return self._save(series_root, series_id, change)

    def jobs(self, series_root, series_id):
        state = self.context(series_root, series_id)
        result = []
        for row in state["jobs"]:
            require_unc(row["root"])
            if descriptor(Path(row["root"]) / "job.json") != row["job"]:
                raise ValueError("Registered job metadata SHA changed: " + row["root"])
            result.append(
                dict(
                    row,
                    active=_path(row["root"]) == _path(state.get("active_job") or ""),
                )
            )
        return result

    def plans(self, series_root, series_id):
        result = self.context(series_root, series_id)["plans"]
        for row in result:
            require_unc(row["path"])
            if descriptor(Path(row["path"])) != {
                key: row[key] for key in ("path", "sha256", "size_bytes")
            }:
                raise ValueError(
                    "Registered execution plan SHA changed: " + row["path"]
                )
        return result

    def known_jobs(self):
        """All explicit local registrations, without discovering NAS folders.

        The consumer revalidates each job's own immutable descriptor/requests.
        A malformed local record is an error, never evidence of a dead worker.
        """
        result = {}
        for path in sorted(self.root.glob("*.json")):
            raw, bound = _asset(path)
            doc = json.loads(raw)
            if (
                doc.get("format") != "darsia-native-jobs-registry-v1"
                or not isinstance(doc.get("series_root"), str)
                or not isinstance(doc.get("series_id"), str)
                or self._file(doc["series_root"], doc["series_id"]) != path
                or not isinstance(doc.get("jobs"), list)
            ):
                raise ValueError("Malformed local Jobs registration: " + str(path))
            for row in doc["jobs"]:
                require_unc(row["root"])
                if not isinstance(row.get("job"), dict):
                    raise ValueError("Registered job lacks its metadata binding")
                key = _path(row["root"])
                if key in result and result[key]["job"] != row["job"]:
                    raise ValueError(
                        "Conflicting local registrations for job: " + row["root"]
                    )
                result[key] = deepcopy(row)
            if descriptor(path) != bound:
                raise ValueError(
                    "Local Jobs registration changed while read: " + str(path)
                )
        return list(result.values())

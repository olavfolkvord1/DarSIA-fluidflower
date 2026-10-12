"""Manifest-bound local inputs for the native production IO hooks.

Logical descriptors are immutable provenance. Only the file locator used for
an open changes: it must name a verified file inside the staged local cache.
This module does not discover dependencies, read the original locator, or grant
scientific access. The job still checks its Git commit, environment and guards.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import ntpath
from pathlib import Path
import re

from .production_io import NasIO, IntegrityError, is_network_path, normalize_locator


def _code_identity(worktree):
    from .production_job import code_identity
    return code_identity(worktree)


def _environment_lock():
    from .production_job import environment_lock
    return environment_lock()


def _require_workers_retired(workers):
    from .production_job import require_workers_retired
    return require_workers_retired(workers)


class ManifestAssetReader:
    """Use ``ProductionJob.stage_inputs()`` without rewriting native records.

    ``source_provenance`` explicitly maps a currently imported source locator to
    a frozen source descriptor in the same flat input manifest. It verifies the
    old evidence bytes; the manifest's Git commit identifies the new IO wiring.
    No source substitution is inferred merely from a matching hash.
    """

    def __init__(self, manifest, staged_inputs, *, cache_root, worktree,
                 source_provenance=None, source_qualification=None,
                 logical_aliases=None, io=None, opaque_descriptors=(),
                 manifest_descriptor=None, restart_descriptor=None):
        self.io = io or NasIO()
        if (manifest.get("format") != "darsia-production-job-v1"
                or manifest.get("code", {}).get("clean_tree") is not True
                or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("code", {}).get("commit", ""))):
            raise ValueError("Inputs require the production manifest and code commit")
        execution = _code_identity(worktree)
        self.restart_record = None
        self._restart_review = None
        if restart_descriptor is not None:
            self._restart_identity(manifest, manifest_descriptor, restart_descriptor, execution)
        elif execution != manifest["code"]:
            raise ValueError("Input reader runtime differs from the manifest clean Git identity")
        inputs = manifest.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs or set(inputs) != set(staged_inputs):
            raise ValueError("Every flat manifest input must be staged exactly once")
        self.code_commit = execution["commit"]
        self.execution_identity = deepcopy(execution)
        self.worktree = Path(worktree).resolve()
        self.cache_root = Path(cache_root).resolve()
        if is_network_path(self.cache_root):
            raise ValueError("Scientific input cache must be local")
        self._bindings = {}
        self._local = {}
        self._verified = {}
        self.verified = self._verified
        self.raw_baselines = {}
        self.physical_references = {}
        self.json_documents = {}
        for role, logical in inputs.items():
            logical = self._full(logical)
            local = self._full(staged_inputs[role])
            if (logical["sha256"], logical["size_bytes"]) != (local["sha256"], local["size_bytes"]):
                raise IntegrityError("Staged bytes differ from the flat manifest")
            path = Path(local["path"]).resolve()
            if is_network_path(path) or not path.is_relative_to(self.cache_root) or path == self.cache_root:
                raise ValueError("Staged input escaped the local cache")
            key = normalize_locator(logical["path"])
            if key in self._bindings and self._bindings[key] != logical:
                # Equal normalized locators cannot bind contradictory bytes.
                previous = self._bindings[key]
                if (previous["sha256"], previous["size_bytes"]) != (logical["sha256"], logical["size_bytes"]):
                    raise IntegrityError("Logical locator has conflicting input bindings")
            else:
                self._bindings[key] = logical
                self._local[key] = {**local, "path": str(path)}
        # Aliases locate already-declared bytes; they cannot add assets, infer a
        # match by SHA, chain through other aliases, or replace canonical entries.
        canonical = dict(self._bindings)
        self.logical_aliases = {}
        self._alias_targets = {}
        for alias, descriptor in (logical_aliases or {}).items():
            if (not isinstance(alias, str) or not alias
                    or not (Path(alias).is_absolute() or ntpath.isabs(alias))):
                raise ValueError("Logical aliases need an explicit absolute original locator")
            target = self._full(descriptor)
            target_key = normalize_locator(target["path"])
            declared = canonical.get(target_key)
            if declared is None:
                raise ValueError("Logical alias target is absent from the flat manifest")
            if (target["sha256"], target["size_bytes"]) != (declared["sha256"], declared["size_bytes"]):
                raise IntegrityError("Logical alias target differs from its manifest binding")
            key = normalize_locator(alias)
            if key in canonical and key != target_key:
                raise ValueError("Logical alias conflicts with a canonical input locator")
            if key in self._alias_targets and self._alias_targets[key] != target_key:
                raise ValueError("Logical aliases have conflicting normalized locators")
            if key != target_key:
                self._alias_targets[key] = target_key
                self._bindings[key] = {**declared, "path": alias}
                self._local[key] = self._local[target_key]
            self.logical_aliases[alias] = deepcopy(declared)
        self._sources = {}
        self.source_bindings = []
        self.source_qualification_state = "no_source_substitution"
        self._opaque = set()
        self._opaque_contents = set()
        for descriptor in opaque_descriptors:
            bound = self.logical_descriptor(descriptor)
            self._opaque.add((normalize_locator(bound["path"]), bound["sha256"]))
            self._opaque_contents.add((bound["sha256"], bound["size_bytes"]))
        for current, frozen in (source_provenance or {}).items():
            frozen = self.logical_descriptor(frozen)
            current_path = Path(current).resolve()
            if is_network_path(current_path) or not current_path.is_relative_to(self.worktree):
                raise ValueError("Executed source must stay inside the committed worktree")
            key = normalize_locator(current_path)
            if key in self._sources:
                raise ValueError("Executed source has duplicate provenance mappings")
            self._sources[key] = frozen
            self.source_bindings.append(dict(current_source=self.io.asset(current_path), original_source=frozen))
        if self._sources:
            self._qualify_sources(manifest, source_qualification)

    def _restart_identity(self, manifest, manifest_descriptor, restart_descriptor, execution):
        """Bind an explicit code fix to the same original immutable job."""
        original = self._full(manifest_descriptor)
        restart_bound = self._full(restart_descriptor)
        if self.io.read_json(original, max_bytes=original["size_bytes"]) != manifest:
            raise IntegrityError("Restart original manifest differs from the supplied manifest")
        restart = self.io.read_json(restart_bound, max_bytes=restart_bound["size_bytes"])
        if (not isinstance(restart, Mapping)
                or restart.get("format") != "darsia-production-restart-v1"
                or restart.get("manifest_sha256") != original["sha256"]
                or restart.get("initial_commit") != manifest["code"]["commit"]
                or not isinstance(restart.get("fix_commit"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", restart["fix_commit"])
                or set(execution) != set(manifest["code"])
                or execution.get("clean_tree") is not True
                or execution.get("commit") != restart["fix_commit"]
                or execution.get("submodules") != manifest["code"].get("submodules")
                or _environment_lock() != manifest.get("environment")):
            raise ValueError("Restart requires the exact manifest and clean fix runtime/environment")
        workers = restart.get("workers_retired")
        if not isinstance(workers, list) or restart.get("partial_folders_read_or_modified") is not False:
            raise ValueError("Restart needs the actual retired-worker union and untouched partials")
        _require_workers_retired(workers)
        if "original_source_review" in restart or "fix_source_review" in restart:
            old_review = self._full(restart.get("original_source_review"))
            fix_review = self._full(restart.get("fix_source_review"))
            assurance = manifest.get("assurance_input_roles", [])
            if not any(self._same_descriptor(old_review, manifest["inputs"].get(role))
                       for role in assurance):
                raise ValueError("Restart must preserve its original manifest source-review assurance")
            self._restart_review = fix_review
        self.restart_record = deepcopy(restart)
        self.restart_descriptor = deepcopy(restart_bound)
        self.manifest_descriptor = deepcopy(original)

    def _qualify_sources(self, manifest, qualification):
        """Source review permits parity bootstrap; actual parity gates profiling."""
        if qualification is None and self._restart_review is not None:
            qualification = dict(source_review=self._restart_review)
        if (not isinstance(qualification, Mapping)
                or not {"source_review"}.issubset(qualification)
                or set(qualification) - {"source_review", "exact_cpu_parity_result"}):
            raise ValueError("Source qualification needs explicit review/parity descriptors")
        review_bound = self._full(qualification["source_review"])
        if self._restart_review is not None:
            if not self._same_descriptor(review_bound, self._restart_review):
                raise ValueError("Fix source review must match the SHA-bound restart assurance")
            review = self.io.read_json(review_bound, max_bytes=review_bound["size_bytes"])
        else:
            if self.restart_record is not None:
                raise ValueError("Source substitutions after a code fix need explicit restart assurance")
            review = self.json(review_bound)
        roles = manifest.get("scientific_input_roles")
        assurance = manifest.get("assurance_input_roles")
        if (not isinstance(roles, list) or not roles or len(set(roles)) != len(roles)
                or not isinstance(assurance, list) or len(set(assurance)) != len(assurance)
                or set(roles) & set(assurance)
                or set(roles) | set(assurance) != set(manifest["inputs"])):
            raise ValueError("Source qualification needs the explicit scientific input roles")
        scientific_inputs = {role: manifest["inputs"][role] for role in roles}
        self._assurance_locators = {normalize_locator(manifest["inputs"][role]["path"]) for role in assurance}
        for descriptor in qualification.values():
            if self._canonical_key(descriptor["path"]) in {
                    normalize_locator(bound["path"]) for bound in scientific_inputs.values()}:
                raise ValueError("Assurance records must not enter their own scientific input table")
        settings = dict(backend="cpu", precision=manifest["settings"]["precision"])
        def bindings_match(value):
            return (value.get("execution_commit") == self.code_commit
                    and self._same_input_bindings(value.get("input_bindings"), scientific_inputs)
                    and self._same_source_bindings(value.get("source_bindings"), self.source_bindings)
                    and value.get("reference_settings") == settings)
        if (review.get("contract") != "darsia-production-io-source-review-v1"
                or review.get("complete") is not True or review.get("io_only_diff_passed") is not True
                or not bindings_match(review)):
            raise ValueError("Source substitutions require the exact Root IO-only source review")
        for row in self.source_bindings:
            if self._canonical_key(row["original_source"]["path"]) in self._assurance_locators:
                raise ValueError("Frozen calibration source cannot be an assurance input")
            self.io.verify(row["current_source"])
            self.path(row["original_source"])
        if manifest.get("purpose") == "file_handling_acceptance":
            if (review.get("purpose_scope") != "exact_CPU_parity_bootstrap"
                    or manifest["settings"].get("backend") != "cpu"
                    or manifest.get("production_permission_granted") is not False):
                raise ValueError("Unqualified source substitutions are limited to CPU parity bootstrap")
            self.source_qualification_state = "exact_CPU_parity_bootstrap"
        else:
            parity = self.json(self._full(qualification.get("exact_cpu_parity_result")))
            if (manifest.get("purpose") not in {"profile", "equivalence", "production"}
                    or parity.get("contract") != "darsia-production-exact-cpu-parity-v1"
                    or parity.get("complete") is not True or parity.get("passed") is not True
                    or not bindings_match(parity)):
                raise ValueError("Profile/equivalence/production needs actual exact CPU parity")
            self.source_qualification_state = "exact_CPU_parity_verified"
        self.source_qualification = deepcopy(qualification)

    def _canonical_key(self, locator):
        key = normalize_locator(locator)
        return self._alias_targets.get(key, key)

    @staticmethod
    def _full(descriptor):
        if (not isinstance(descriptor, Mapping)
                or set(descriptor) != {"path", "sha256", "size_bytes"}
                or not isinstance(descriptor["path"], str) or not descriptor["path"]
                or not isinstance(descriptor["sha256"], str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", descriptor["sha256"])
                or type(descriptor["size_bytes"]) is not int or descriptor["size_bytes"] < 0):
            raise ValueError("Manifest/cache descriptors require path, SHA and integer size")
        return dict(descriptor, sha256=descriptor["sha256"].lower())

    @classmethod
    def _same_descriptor(cls, left, right):
        try:
            left, right = cls._full(left), cls._full(right)
        except ValueError:
            return False
        return (normalize_locator(left["path"]) == normalize_locator(right["path"])
                and (left["sha256"], left["size_bytes"]) == (right["sha256"], right["size_bytes"]))

    @classmethod
    def _same_input_bindings(cls, left, right):
        return (isinstance(left, Mapping) and set(left) == set(right)
                and all(cls._same_descriptor(left[role], right[role]) for role in right))

    @classmethod
    def _same_source_bindings(cls, left, right):
        return (isinstance(left, list) and len(left) == len(right)
                and all(isinstance(a, Mapping) and set(a) == {"current_source", "original_source"}
                        and all(cls._same_descriptor(a[key], b[key]) for key in b)
                        for a, b in zip(left, right)))

    def logical_descriptor(self, descriptor):
        """Return declared provenance; compact requests cannot invent a size."""
        if (not isinstance(descriptor, Mapping) or not {"path", "sha256"}.issubset(descriptor)
                or not isinstance(descriptor["path"], (str, Path))
                or not isinstance(descriptor["sha256"], str)):
            raise ValueError("Input request needs its logical locator and SHA")
        key = normalize_locator(descriptor["path"])
        logical = self._bindings.get(key)
        if logical is None:
            raise ValueError(f"Input is absent from the flat manifest: {descriptor['path']}")
        if descriptor["sha256"].lower() != logical["sha256"]:
            raise IntegrityError("Input request SHA differs from the flat manifest")
        if "size_bytes" in descriptor and (type(descriptor["size_bytes"]) is not int
                or descriptor["size_bytes"] != logical["size_bytes"]):
            raise IntegrityError("Input request size differs from the flat manifest")
        return deepcopy(logical)

    def path(self, descriptor):
        logical = self.logical_descriptor(descriptor)
        local = self._local[normalize_locator(logical["path"])]
        path = Path(local["path"])
        # Recheck confinement after construction, including changed symlinks.
        resolved = path.resolve()
        if is_network_path(resolved) or not resolved.is_relative_to(self.cache_root):
            raise IntegrityError("Input cache locator no longer stays local")
        stamp = path.stat()
        key = (str(resolved), stamp.st_size, stamp.st_mtime_ns, logical["sha256"])
        if key not in self._verified:
            self.io.verify(local)
            after = path.stat()
            if (stamp.st_size, stamp.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise IntegrityError("Local input changed during verification")
            self._verified[key] = True
        return path

    def resolve_path(self, logical_path, expected_sha256=None):
        bound = self._bindings.get(normalize_locator(logical_path))
        if bound is None:
            raise ValueError(f"Input is absent from the flat manifest: {logical_path}")
        return self.path({"path": str(logical_path),
                          "sha256": bound["sha256"] if expected_sha256 is None else expected_sha256})

    def source_path(self, current_source, expected_sha256):
        bound = self._sources.get(normalize_locator(current_source))
        if bound is None:
            if normalize_locator(current_source) in self._alias_targets:
                raise ValueError("Execution source aliases require separate reviewed source_provenance")
            return self.resolve_path(current_source, expected_sha256)
        current = next(row["current_source"] for row in self.source_bindings
                       if normalize_locator(row["current_source"]["path"]) == normalize_locator(current_source))
        self.io.verify(current)
        if bound["sha256"] != expected_sha256:
            raise IntegrityError("Frozen implementation provenance SHA differs")
        return self.path(bound)

    def sha256(self, logical_path):
        path = self.resolve_path(logical_path)
        return self.io.asset(path)["sha256"]

    def json(self, descriptor):
        bound = self.logical_descriptor(descriptor)
        if (bound["sha256"], bound["size_bytes"]) in self._opaque_contents:
            raise ValueError("Opaque evidence may be hashed but never parsed")
        path = self.path(descriptor)
        # Read through the same local verified descriptor, never the NAS path.
        key = (str(path.resolve()), bound["sha256"])
        if key not in self.json_documents:
            # The sealed selected metadata index is larger than the generic
            # transport default. Its exact SHA and integer size are already
            # verified through the flat manifest, so use that bound read cap.
            value = self.io.read_json({**bound, "path": str(path)}, max_bytes=bound["size_bytes"])
            if not isinstance(value, dict):
                raise ValueError("Expected source-bound JSON object")
            self.json_documents[key] = value
        return self.json_documents[key]

    def array(self, descriptor):
        path = self.path(descriptor)
        bound = self.logical_descriptor(descriptor)
        if (bound["sha256"], bound["size_bytes"]) in self._opaque_contents:
            raise ValueError("Opaque evidence may not be decoded")
        if self._canonical_key(bound["path"]) in getattr(self, "_assurance_locators", set()):
            raise ValueError("Assurance metadata may not enter scientific arrays")
        import numpy as np
        return np.load(path, mmap_mode="r", allow_pickle=False)

"""Named control inputs for the explicit series Setup operation.

This is a control catalogue, not a complete scientific input closure. The
original corrected-geometry proof readers run after capacity/startup/claim
admission and bind the sources they actually validate into the operation's
receipt. No map, image or RAW acquisition is decoded here. Original physical
locators are required; this route does not grant general input relocation.
"""
from __future__ import annotations

from copy import deepcopy
import json
import ntpath
from pathlib import Path
import re

from .series_setup_protocol import (
    APPLY_COMPLETE_FORMAT, validate_setup_operation, validate_setup_plan,
)


SCOPE = "series-setup-controls-only"
_IDENTITY = ("path", "sha256", "size_bytes")
_REQUEST_FIELDS = {"format", "run_id", "configuration_files", "series_context",
                   "baseline", "profile", "operation_overrides", "protocol_inputs",
                   "output", "shape_basis_certificate"}


def _key(path):
    """Compare current-host spellings without relocating recorded history."""
    return ntpath.normcase(ntpath.normpath(str(path)))


def _same(left, right):
    """Keep bool/int and int/float differences in authored JSON metadata."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


class _Controls:
    """Use only the original catalogue's admitted named consumers."""

    def __init__(self, catalog):
        self.catalog = catalog

    def source(self, bound, role, use, *, metadata=False):
        """Reject relocation before stat/open; metadata uses the strict reader."""
        actual = self.catalog.physical(bound)
        if _key(actual["path"]) != _key(bound["path"]):
            raise ValueError("Setup controls require their original physical locators")
        operation = self.catalog.doc if metadata else self.catalog.add
        return operation(bound, role, "setup." + use)

    def named(self, path, role, use, *, metadata=False):
        """Bind one finite canonical control path, never a recursive discovery.

        Preparation checks containment before hashing. Runtime replay uses its
        already verified named reference and must not recapture a changed SHA.
        The temporary zero digest is only a containment check, never evidence.
        """
        name = "setup." + use
        binding = getattr(self.catalog, "binding", None)
        if binding is not None:
            row = binding.verified.for_use(name)
            if _key(row.path) != _key(path):
                raise ValueError("Setup control replay changed its original physical locator")
            bound = dict(path=str(path), sha256=row.reference["sha256"],
                         size_bytes=row.reference["bytes"])
        else:
            from ..store_map import input_reference
            from ..corrected_setup import _descriptor
            input_reference(str(path), self.catalog.roots, sha256="0" * 64,
                            size_bytes=0, role=role)
            bound = _descriptor(path)
        return bound, self.source(bound, role, use, metadata=metadata)


def _plan(controls, config):
    """Join the immutable plan to the literal selected project and revision."""
    plan = validate_setup_plan(controls.source(
        config["setup_plan"], "configuration", "plan", metadata=True))
    if any(not _same(plan[key], config[key]) for key in
           ("project_root", "project_manifest", "adopted_revision", "series_manifest")):
        raise ValueError("The Setup plan belongs to another project or adopted revision")
    runs, plans = plan["runs"], plan["build_plans"]
    if config["operation"] == "apply":
        if (not _same(config["runs"], runs) or not _same(config["build_plans"], plans)
                or any(not _same(config[key], plan[key]) for key in
                       ("native_parameters", "target_machine", "metadata_output", "output_scope"))):
            raise ValueError("Apply differs from its exact immutable run templates")
        selected = plan
    else:
        matches = [row for row in plans if row["run_id"] == config["run"]]
        if len(matches) != 1:
            raise ValueError("Build names a run outside its immutable Setup plan")
        selected = matches[0]
        for key in ("native_parameters", "target_machine", "metadata_output", "output_scope"):
            if not _same(config[key], selected.get(key)):
                raise ValueError("Build changed its original plan field: " + key)
    if not _same(config["native_parameters"], selected.get("native_parameters")):
        raise ValueError("Setup native parameters differ from the selected original run")
    return plan, selected


def _completed_controls(controls, config, plan, revision):
    """Catalogue completed Apply metadata without replaying scientific proofs."""
    completed = controls.source(config["coordinator_completion"], "configuration",
                                "coordinator_completion", metadata=True)
    expected = {"format", "complete", "setup_plan", "adopted_revision", "project_manifest",
                "series_manifest", "revision_id", "current_sha256", "runs", "Build_run",
                "optical_calibration_run"}
    if (set(completed) != expected or completed["format"] != APPLY_COMPLETE_FORMAT
            or completed["complete"] is not True or completed["Build_run"] is not False
            or completed["optical_calibration_run"] is not False
            or any(not _same(completed[key], config[key]) for key in
                   ("setup_plan", "adopted_revision", "project_manifest", "series_manifest"))
            or completed["revision_id"] != revision.get("revision_id")
            or completed["current_sha256"] != config["adopted_revision"]["sha256"]
            or _key(config["coordinator_completion"]["path"])
                != _key(Path(plan["control_root"]) / "apply-complete.json")):
        raise ValueError("Build lacks its exact complete coordinator Apply controls")
    rows = completed["runs"]
    if (not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) !=
            {"run_id", "export", "apply_completion", "configuration_files"} for row in rows)
            or [row["run_id"] for row in rows] != plan["runs"]):
        raise ValueError("Coordinator Apply rows changed the original ordered run scope")
    selected = next(row for row in rows if row["run_id"] == config["run"])
    if any(not _same(selected[key], config[key]) for key in
           ("apply_completion", "configuration_files")):
        raise ValueError("Build is not bound to its actual selected Apply postimages")
    exported = controls.source(selected["export"], "configuration", "export", metadata=True)
    applied = controls.source(config["apply_completion"], "configuration", "apply_completion", metadata=True)
    if (exported.get("run_id") != config["run"]
            or exported.get("revision_id") != revision.get("revision_id")
            or exported.get("current_sha256") != completed["current_sha256"]
            or applied.get("format") != "darsia-corrected-setup-apply-complete-v1"
            or applied.get("complete") is not True or applied.get("run_id") != config["run"]
            or applied.get("revision_id") != revision.get("revision_id")
            or not _same(applied.get("export"), selected["export"])):
        raise ValueError("Build's actual export and per-run Apply controls disagree")
    destination = Path(selected["export"]["path"]).parent
    if _key(config["apply_completion"]["path"]) != _key(destination / "apply-complete.json"):
        raise ValueError("The selected Apply completion belongs to another export")
    controls.named(destination / "apply-claim.json", "configuration", "apply_claim", metadata=True)
    controls.source(applied["attempt_receipt"], "configuration", "apply_attempt", metadata=True)
    for index, row in enumerate(applied["configurations"]):
        for side in ("before", "after"):
            controls.source(row[side], "configuration", "applied." + str(index) + "." + side)
    if exported.get("root_export") is not None:
        controls.source(exported["root_export"], "configuration", "reference_export", metadata=True)
    return selected


def _raw_request(controls, config, selected):
    """Admit every original request pin before its unchanged verifying reader."""
    from ..rig_raw_build import REQUEST_FORMAT, read_build_request
    request = controls.source(config["raw_build_request"], "configuration", "raw_request", metadata=True)
    if (set(request) != _REQUEST_FIELDS or request["format"] != REQUEST_FORMAT
            or request["run_id"] != config["run"]
            or not _same(request["configuration_files"], config["configuration_files"])
            or not _same({key: value for key, value in request.items() if key != "configuration_files"},
                         selected["request_template"])):
        raise ValueError("The full RAW request differs from its actual run, Apply postimages or original template")
    for index, row in enumerate(config["configuration_files"]):
        controls.source(row, "configuration", "raw.configuration." + str(index))
    controls.source(request["baseline"], "image", "raw.baseline")
    controls.source(request["profile"], "rig_model", "raw.profile", metadata=True)
    if not _same(request["series_context"]["manifest"], config["series_manifest"]):
        raise ValueError("The full RAW request belongs to another original series manifest")
    controls.source(request["series_context"]["manifest"], "configuration", "raw.series_manifest")
    for index, row in enumerate(request["protocol_inputs"]):
        controls.source(row["source"], "protocol", "raw.protocol." + str(index) + "." + row["role"])
    controls.source(request["shape_basis_certificate"], "rig_model", "raw.shape_basis", metadata=True)
    verified, _pins = read_build_request(config["raw_build_request"], artifact_kind="raw_rig_baseline")
    if not _same(verified, request):
        raise ValueError("The original RAW request changed after its named controls were read")
    return verified


def catalogue_setup_inputs(catalog, config):
    """Bind finite Setup controls; leave actual scientific proof replay explicit.

    The enclosing catalogue retains its existing schema. Callers must record
    the returned ``scope`` and must not describe this as all scientific input
    admission. Apply target TOMLs and future Build template paths are not read
    or frozen here: original Apply owns their CAS and retained-before proof.
    """
    validate_setup_operation(config)
    controls = _Controls(catalog)
    controls.source(config["project_manifest"], "configuration", "project_manifest", metadata=True)
    revision = controls.source(config["adopted_revision"], "rig_model", "adopted_revision", metadata=True)
    controls.source(config["series_manifest"], "configuration", "series_manifest")
    controls.source(config["native_parameters"], "configuration", "native_parameters", metadata=True)
    plan, selected = _plan(controls, config)
    if (revision.get("format") != "darsia-fluidflower-geometry-facies-v1"
            or revision.get("schema") != 1
            or re.fullmatch(r"[0-9a-f]{32}", str(revision.get("revision_id"))) is None):
        raise ValueError("Setup requires its actual saved adopted revision controls")
    request = None
    if config["operation"] == "build":
        _completed_controls(controls, config, plan, revision)
        request = _raw_request(controls, config, selected)
    return dict(scope=SCOPE, plan=deepcopy(plan), revision=deepcopy(revision),
                raw_request=deepcopy(request))

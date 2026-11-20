"""Exact provenance for explicitly retained preallocation model frames."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import cv2

import darsia
from darsia.jobs.step import plain, run_tables


class _NoDoc(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def _closure(path, roots, *, prefix=False):
    tree = ast.parse(path.read_text(encoding="utf8"))
    definitions = {}
    imports = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                key = alias.asname or alias.name.split(".")[0]
                single = (
                    ast.Import(names=[alias])
                    if isinstance(node, ast.Import)
                    else ast.ImportFrom(
                        module=node.module, names=[alias], level=node.level
                    )
                )
                imports[key] = single
    if prefix:
        node = definitions["_phase_maps"]
        stop = next(
            (
                i
                for i, statement in enumerate(node.body)
                if isinstance(statement, ast.With)
            ),
            None,
        )
        if stop is None:
            raise ValueError("The phase workflow has no identifiable model prepass.")
        node.body = node.body[:stop]
    pending = list(roots)
    seen = {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        node = definitions.get(name, imports.get(name))
        if node is None:
            if name in roots:
                raise ValueError(f"Missing model source identity: {name}.")
            continue
        seen[name] = node
        pending.extend(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
    body = ast.Module(body=[seen[key] for key in sorted(seen)], type_ignores=[])
    return hashlib.sha256(
        ast.dump(_NoDoc().visit(body), include_attributes=False).encode()
    ).hexdigest()


def model_source_identity(root: Path | None = None) -> dict[str, str]:
    """Conservative code identity with only allocation-only code excluded.

    All other package modules are pinned. In the two mixed modules, the model
    generator and workflow's prefix/recursive local dependencies are pinned by
    AST, including imported bindings. This allows corrections to the later gas
    allocation body/settings without accepting changed model or rule methods.
    """
    root = Path(darsia.__file__).resolve().parent if root is None else Path(root)
    gas = "multiphase/gas_allocation.py"
    workflow = "presets/workflows/calibration/steps.py"
    excluded = {
        "multiphase/gas_zone.py",
        "multiphase/gas_zone_cache.py",
        "multiphase/gas_bridge.py",
        "multiphase/gas_parameters.py",
    }
    found = {}
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix()
        if name in excluded:
            continue
        if name == gas:
            found[name] = _closure(path, {"gas_model_frames"})
        elif name == workflow:
            found[name] = _closure(path, {"_phase_maps"}, prefix=True)
        else:
            found[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if gas not in found or workflow not in found:
        raise ValueError("Cannot establish the complete model source identity.")
    return found


def _photo_identity(path):
    if path is None:
        return None
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("A previous acquisition changed while identifying it.")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest}


def model_cache_identity(ctx, frames, dates, previous_photos):
    """Bind actual model inputs, acquisition order, settings and compute source.

    ``previous_photos`` must come from the same previous-acquisition resolver as
    the model reader. These photos are not corrected-image result inputs, so
    their content is pinned separately. No allocation/overflow identity enters
    this model-only contract. This function never selects or adopts a cache.
    """
    frames, dates, previous_photos = tuple(frames), tuple(dates), tuple(previous_photos)
    if not (len(frames) == len(dates) == len(previous_photos)):
        raise ValueError("Model provenance must cover every selected acquisition.")
    inputs = [
        i
        for i in ctx.task.inputs
        if i.kind in {"rig", "phase-model", "corrected-image"}
    ]
    if (
        sum(i.kind == "rig" for i in inputs) != 1
        or sum(i.kind == "phase-model" for i in inputs) != 1
    ):
        raise ValueError("A model cache requires one accepted rig and phase model.")
    corrected = [i for i in inputs if i.kind == "corrected-image"]
    if len(corrected) != len(frames) or {i.frame for i in corrected} != set(frames):
        raise ValueError("Model provenance lacks the selected corrected-image inputs.")
    tokens = {i.key: ctx.token_of(i) for i in inputs}
    if any(not isinstance(token, str) or not token for token in tokens.values()):
        raise ValueError("Model cache inputs must have accepted tokens.")
    own = run_tables(ctx.tables, ctx.task.run)
    settings = {
        "phase_rules": plain(vars(ctx.settings["phase_rules"])),
        "material": plain(vars(ctx.settings["material"])),
        "data": own.get("data", {}),
        "protocols": own.get("protocols", {}),
        "configured_files": dict(ctx._files),
        "previous_photos": [_photo_identity(path) for path in previous_photos],
    }
    # The run's expert knowledge limits the model frames' probabilities; only a
    # run that names it carries it, so that every other run's identity stays.
    for section in ("expert_knowledge", "roi"):
        if section in own:
            settings[section] = own[section]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn")
    }
    versions["opencv"] = cv2.__version__
    versions["python"] = platform.python_version()
    return json.loads(
        json.dumps(
            {
                "frame_keys": ["/".join(frame) for frame in frames],
                "dates": [date.isoformat() for date in dates],
                "input_tokens": tokens,
                "model": plain(vars(ctx.settings["phase_model"])),
                "settings": plain(settings),
                "backend": {
                    "environment": ctx._job["environment"],
                    "device": ctx.device,
                    "versions": versions,
                },
                "source": model_source_identity(),
            },
            allow_nan=False,
        )
    )

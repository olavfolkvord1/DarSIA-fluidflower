"""Locked P1-final3 rules and source-exact QA for selected camera images.

This adapter adds no fit or rule. Native-centre and rig-edge inference are
supplied by the frozen producers; their provenance is mandatory. Existing
envelopes are read, never rebuilt from the denser selected-image timeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .project import sha256_file

CONTRACT = "ffac-selected-p1-final3-phase-v1"
SPEC_SHA256 = "83ab221ee3eeee00396e406c2c586e5a38372ac410837d7f3b99c312be4931a5"
CO2_REPLAY_LIMIT = .001
GAS_REPLAY_LIMIT = .01


def bound_json(asset):
    path = Path(asset["path"])
    if sha256_file(path) != asset["sha256"]:
        raise ValueError("Selected-image source SHA changed: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def asset(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def frozen_parent_chain(parent):
    """Resolve the actual executed ancestor plans without reading image arrays."""
    parent = Path(parent)
    final3 = bound_json(asset(parent / "p1_final3_plan.json"))
    seal = bound_json(asset(parent / "rig_camera_candidate_receipt.json"))
    if (seal.get("complete") is not True or seal.get("contract") != "ffac-g3-t4c-p1-final3-candidate-v1"
            or seal.get("source_plan_sha256") != sha256_file(parent / "p1_final3_plan.json")):
        raise ValueError("Selected images require the sealed P1-final3 parent")
    final2_root = Path(final3["binding"]["parent"])
    final2 = bound_json(asset(final2_root / "p1_final2_plan.json"))
    final1_root = Path(final2["binding"]["parent"])
    final1 = bound_json(asset(final1_root / "p1_final_plan.json"))
    t4c_root = Path(final1["binding"]["parent"])
    t4c = bound_json(asset(t4c_root / "t4c_floor_plan.json"))
    if t4c["placement"] != "P1" or final3["placement"] != "P1-final3":
        raise ValueError("Selected images cannot change the chosen P1 placement")
    for item in [*final3["binding"]["heads"].values()]:
        if "path" in item:
            bound_json(item)
    return dict(parent=str(parent), parent_seal=asset(parent / "rig_camera_candidate_receipt.json"),
        final3_plan=asset(parent / "p1_final3_plan.json"), final2_root=str(final2_root),
        final2_plan=asset(final2_root / "p1_final2_plan.json"), final1_root=str(final1_root),
        final1_plan=asset(final1_root / "p1_final_plan.json"), t4c_root=str(t4c_root),
        t4c_plan=asset(t4c_root / "t4c_floor_plan.json"), delta=float(t4c["delta"]),
        placement="P1", heads=final3["binding"]["heads"],
        base_candidate=t4c["evaluation_config"]["parent"],
        fixed_floor_curve=final2["binding"]["inputs"]["pooled_floor_curve"],
        fixed_zones=final2["binding"]["inputs"]["zones"],
        training=False, parent_changed=False)


def read_parent_envelopes(chain, run):
    """Read exact E1/E2 and actual U1/U2 camera times from final2 receipts."""
    path = Path(chain["final2_root"]) / "envelopes" / run / "receipt.json"
    source = asset(path)
    receipt = bound_json(source)
    if receipt.get("run") != run or receipt.get("contract") != "ffac-g3-t4c-p1-final2-candidate-v1":
        raise ValueError("Gas envelope belongs to another run or revision")
    arrays = []
    for name in ("envelope1", "envelope2"):
        item = receipt[name]
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("Frozen parent gas envelope changed")
        shape = tuple(item["shape_hw"])
        packed = np.load(item["path"], allow_pickle=False)
        arrays.append(np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool))
    b = receipt["binding"]
    return dict(env1=arrays[0], env2=arrays[1], u1_elapsed_h=float(b["u1_elapsed_h"]),
        u2_elapsed_h=float(b["u2_elapsed_h"]), source_receipt=source,
        source_envelopes={name: receipt[name] for name in ("envelope1", "envelope2")},
        source_u1_index=int(b["u1_frame_index"]), source_u2_index=int(b["u2_frame_index"]),
        rebuilt=False, transported=False)


def envelope_ordinal(elapsed_h, u1_elapsed_h, u2_elapsed_h):
    """Encode literal 'after parent U' semantics for the old integer API."""
    if not all(np.isfinite([elapsed_h, u1_elapsed_h, u2_elapsed_h])) or u2_elapsed_h < u1_elapsed_h:
        raise ValueError("Gas-envelope camera times must be finite and ordered")
    return int(elapsed_h > u1_elapsed_h) + int(elapsed_h > u2_elapsed_h)


@dataclass(frozen=True)
class SelectedFrameRuleInputs:
    base_labels: np.ndarray
    modelled: np.ndarray
    filled_support: np.ndarray
    top_keep: np.ndarray
    open_components: np.ndarray
    barrier_mask: np.ndarray
    source_component_ids: tuple[int, ...]
    frozen_base_rule_b: dict
    pixel_area_mm2: np.ndarray
    corrected_domain: np.ndarray
    below_floor: np.ndarray
    l1_zone: np.ndarray
    elapsed_h: float
    run: str
    source_sha256: str
    native_inference_provenance: dict
    edge_inference_provenance: dict


def apply_locked_postprocessing(inputs, shift_context, delta, envelopes, *, backend="auto"):
    """Apply frozen P1 shift/B/top/floor Aâ€“D, then drop contact-fill reason b.

    Original T4c's temporary floor fill is omitted: P1-final restored every
    such pixel to its pre-fill category. No history or envelope is refitted.
    """
    from .phase_gas_t4c_placements import evaluate_placement
    from .phase_ffac_p1_final import apply_p1_final
    from .phase_ffac_p1_final2 import apply_final_round
    from .phase_ffac_p1_final3 import remove_contact_fill_b
    for name in ("native_inference_provenance", "edge_inference_provenance"):
        proof = getattr(inputs, name)
        if not isinstance(proof, dict) or proof.get("inference_performed") is not True:
            raise ValueError("Selected phase production needs numerical " + name)
    if len(inputs.source_sha256) != 64:
        raise ValueError("Selected phase production needs exact camera acquisition SHA")
    p1, p1_audit, _ = evaluate_placement(shift_context, "P1", float(delta), inputs.base_labels,
        inputs.modelled, inputs.pixel_area_mm2, filled_support=inputs.filled_support,
        top_keep=inputs.top_keep, open_components=inputs.open_components,
        barrier_mask=inputs.barrier_mask, frozen_base_rule_b=inputs.frozen_base_rule_b,
        source_component_ids=list(inputs.source_component_ids))
    top = apply_p1_final(p1, p1, np.zeros(p1.shape, bool), inputs.pixel_area_mm2, backend=backend)
    ordinal = envelope_ordinal(inputs.elapsed_h, envelopes["u1_elapsed_h"], envelopes["u2_elapsed_h"])
    # The frozen final2 contact rule accepts every resolved rig category,
    # including declared nearest fill. T4c's gas evaluation still uses the
    # modelled population separately above.
    floor_support = (inputs.modelled | inputs.filled_support) & inputs.corrected_domain
    final2 = apply_final_round(top.labels, inputs.below_floor, floor_support,
        None, inputs.l1_zone, envelopes["env1"], envelopes["env2"], float(inputs.elapsed_h),
        ordinal, 0, 1, inputs.pixel_area_mm2, backend=backend,
        corrected_domain_mask=inputs.corrected_domain)
    if not final2.qa["all_pass"]:
        raise ValueError("Selected frame failed the unchanged floor Aâ€“D QA")
    labels = remove_contact_fill_b(final2.labels, final2.fill_reason_a, final2.fill_reason_b, final2.step_masks["C"])
    # No carryover mask is needed: every reason-b pixel would be undone. The
    # untouched reason-a short contact rule remains the native frozen rule.
    if final2.fill_reason_b.any():
        raise AssertionError("An absent carryover mask cannot create contact reason b")
    qa = dict(contract=CONTRACT, run=inputs.run, elapsed_h=float(inputs.elapsed_h),
        source_sha256=inputs.source_sha256, p1=p1_audit, top=top.qa, floor=final2.qa,
        temporary_t4c_floor_fill="omitted because P1-final restores it exactly",
        contact_reason_b="absent; P1-final3 removes every such pixel", gas_envelopes={
            key: value for key, value in envelopes.items() if key not in ("env1", "env2")},
        envelope_ordinal=ordinal, parent_u1_u2_camera_times_preserved=True,
        native_inference=inputs.native_inference_provenance,
        edge_inference=inputs.edge_inference_provenance, training=False, mass_computed=False,
        bottom_mass_allowed=False, automatic_activation=False)
    return labels, qa, dict(removed_top=top.removed_top, **final2.step_masks,
                           contact_reason_a=final2.fill_reason_a, contact_reason_b=final2.fill_reason_b)


def replay_difference(parent_labels, replay_labels, pixel_area_mm2, *, run, frame_index,
                      source_sha256, elapsed_h):
    """Literal predeclared replay gate; zero parent area allows zero changes only."""
    a, b, area = map(np.asarray, (parent_labels, replay_labels, pixel_area_mm2))
    if (a.dtype != np.uint8 or b.dtype != np.uint8 or a.ndim != 2 or a.shape != b.shape
            or area.shape != a.shape or not np.isfinite(area).all() or np.any(area < 0)
            or not np.isin(a, (0, 1, 2, 255)).all() or not np.isin(b, (0, 1, 2, 255)).all()):
        raise ValueError("Replay QA requires exact categorical and measured-area grids")
    result = dict(run=run, frame_index=int(frame_index), source_sha256=source_sha256,
                  elapsed_h=float(elapsed_h), categorical_changed_pixels=int((a != b).sum()))
    passes = []
    for name, before, after, limit in (("co2", np.isin(a, (1, 2)), np.isin(b, (1, 2)), CO2_REPLAY_LIMIT),
                                      ("gas", a == 2, b == 2, GAS_REPLAY_LIMIT)):
        changed = before != after
        measure = lambda mask: float(area[mask].sum(dtype=np.float64) / 100)
        old_area, changed_area = measure(before), measure(changed)
        relative = changed_area / old_area if old_area > 0 else (0. if not changed.any() else None)
        passed = (not changed.any() if old_area == 0 else relative <= limit)
        result[name] = dict(parent_pixels=int(before.sum()), replay_pixels=int(after.sum()),
            changed_pixels=int(changed.sum()), added_pixels=int((after & ~before).sum()),
            removed_pixels=int((before & ~after).sum()), parent_cm2=old_area, replay_cm2=measure(after),
            changed_cm2=changed_area, added_cm2=measure(after & ~before), removed_cm2=measure(before & ~after),
            changed_area_fraction=relative, zero_parent_area=old_area == 0,
            acceptance_limit=limit, pass_gate=passed)
        passes.append(passed)
    result.update(all_pass=all(passes), bit_identical=bool(np.array_equal(a, b)),
                  support_changed_pixels=int(((a == 255) != (b == 255)).sum()))
    return result


def chronological_history(rows):
    """Source identities, not frame indices, own policy-A predecessors."""
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise ValueError("Policy A must contain each selected acquisition exactly once")
    runs = {row["run"] for row in rows}
    if len(runs) != 1:
        raise ValueError("A chronological history belongs to exactly one run")
    ordered = sorted(rows, key=lambda row: (float(row["elapsed_h"]), row["camera_datetime"], row["source_sha256"]))
    if any(float(a["elapsed_h"]) >= float(b["elapsed_h"]) for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Distinct selected acquisitions require strictly increasing camera times")
    result = []
    previous = None
    for row in ordered:
        result.append(dict(row, history_policy="Olav policy A, 18 September 2026",
                           previous_source_sha256=None if previous is None else previous["source_sha256"]))
        previous = row
    return result


def area_time_series(rows, studio_rows):
    """Flag literal selected-image jumps against surrounding Studio area steps."""
    original = sorted(studio_rows, key=lambda row: row["elapsed_h"])
    ordered = sorted(rows, key=lambda row: row["elapsed_h"])
    result = []
    for index, row in enumerate(ordered):
        report = dict(row)
        for phase in ("co2", "gas"):
            change = None if index == 0 else abs(row[phase + "_cm2"] - ordered[index-1][phase + "_cm2"])
            surrounding = [abs(b[phase + "_cm2"] - a[phase + "_cm2"])
                for a, b in zip(original, original[1:])
                if index > 0 and a["elapsed_h"] <= row["elapsed_h"] and b["elapsed_h"] >= ordered[index-1]["elapsed_h"]]
            benchmark = max(surrounding) if surrounding else None
            report[phase + "_step_cm2"] = change
            report[phase + "_surrounding_studio_step_cm2"] = benchmark
            report[phase + "_step_flag"] = bool(change is not None and benchmark is not None and change > benchmark)
        result.append(report)
    return result

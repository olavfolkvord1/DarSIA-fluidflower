"""Optional lossless copies of the probabilities used by frozen decisions.

This module is an output side channel. It never predicts, resizes, reconstructs
scores from labels, or supplies values to a scientific decision.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

REQUEST_FORMAT = "darsia-decision-probability-storage-request-v1"
FORMAT = "darsia-decision-probabilities-v1"
ENCODING = "float32-exact-decision-values"
ARRAYS = ("pixel_probability", "regional_probability", "branch_support")
CODES = {"unscored": 0, "centre": 1, "edge": 2, "placement": 3}


def parse_request(config):
    if "gas_probability_storage" not in config:
        return None
    request = config["gas_probability_storage"]
    if (type(request) is not dict or set(request) != {"format", "encoding"}
            or request != dict(format=REQUEST_FORMAT, encoding=ENCODING)):
        raise ValueError("Explicit exact-float32 decision probability request required")
    return deepcopy(request)


def same_metadata(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, allow_nan=False, separators=(",", ":"))


def identity(array):
    value = np.asarray(array)
    digest = hashlib.sha256()
    for chunk in np.nditer(value, flags=["external_loop", "buffered", "zerosize_ok"],
                          op_flags=["readonly"], order="C", buffersize=65536):
        digest.update(chunk.tobytes(order="C"))
    return dict(dtype=value.dtype.str, shape=list(value.shape), sha256=digest.hexdigest())


def transition(capture, name, before, after):
    """Count actual ordered transitions, including overlaps across operators."""
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape:
        raise ValueError("Decision transition grids differ")
    transitions = {}
    for old in np.unique(a):
        for new in np.unique(b):
            count = int(np.count_nonzero((a == old) & (b == new)))
            if count:
                transitions[str(int(old)) + "->" + str(int(new))] = count
    capture.setdefault("transitions", []).append(dict(
        operator=name, before=identity(a), after=identity(b),
        changed_pixels=int(np.count_nonzero(a != b)), transitions=transitions))


def pair(capture, pixel, regional, support, co2, selected, *, branch,
         pixel_threshold, regional_threshold):
    """Copy the actual native arrays immediately after their original use."""
    p, r = np.asarray(pixel), np.asarray(regional)
    valid, union, chosen = (np.asarray(v) for v in (support, co2, selected))
    if (p.dtype != np.float32 or r.dtype != np.float32 or p.ndim != 2
            or r.shape != p.shape or any(v.dtype != bool or v.shape != p.shape
                                      for v in (valid, union, chosen))):
        raise ValueError("Decision-used native float32 pair and bool supports required")
    valid = valid & np.isfinite(p) & np.isfinite(r)
    reproduced = union & valid & (p >= pixel_threshold) & (r >= regional_threshold)
    if not np.array_equal(reproduced, chosen):
        raise ValueError("Captured pair does not reproduce the actual pre-minimum decision")
    capture.update(pixel=p.copy(), regional=r.copy(), support=valid.copy(), branch=branch,
                   decision=dict(pixel=identity(p), regional=identity(r), support=identity(valid),
                                 frozen_co2=identity(union), selected=identity(chosen),
                                 selected_pixels=int(chosen.sum()),
                                 pixel_threshold=float(pixel_threshold),
                                 regional_threshold=float(regional_threshold),
                                 reproduced_pre_minimum_exact=True))


def capture_centre(capture, values):
    if values["source_shape"] is None:
        raise ValueError("Probability pair storage requires an actually used regional head")
    g3, scores = values["g3"], values["scores"]
    settings = values["settings"]
    pair(capture, g3, scores, np.isfinite(g3) & np.isfinite(scores), values["co2"],
         values["regional_gas"], branch="centre", pixel_threshold=settings.pixel_threshold,
         regional_threshold=settings.regional_threshold)
    transition(capture, "centre-minimum", values["regional_gas"], values["gas"])


def capture_centre_selection(capture, values):
    pixel, native, parameters = values["pixel"], values["native"], values["parameters"]
    pt, rt = (parameters["centre_selection"][key] for key in ("pixel_threshold", "regional_threshold"))
    pair(capture, pixel, native, np.isfinite(pixel) & np.isfinite(native), np.ones(pixel.shape, bool),
         (pixel >= pt) & (native >= rt), branch="centre", pixel_threshold=pt, regional_threshold=rt)


def require_pair_mode(placement, parameters):
    if placement == parameters["placement"]["placement_modes"]["constrained_neighbourhood"]:
        raise ValueError("P2 has no decision-used regional probability pair; optional pair storage is unavailable")


def capture_placement(capture, values):
    support, masks, params = values["support"], values["threshold_masks"], values["parameters"]
    pair(capture, values["pixel"], values["regional_score"], support,
         np.isin(values["labels"], (1, 2)), masks["pre_minimum_gas"] & support,
         branch="placement", pixel_threshold=params["literal_values"]["replace_gas_from_scores_float_0_7"],
         regional_threshold=params["literal_values"]["replace_gas_from_scores_float_0_6"])
    capture["placement_mode"] = values["placement"]
    capture["light_epoch_id"] = values["context"].light_epoch_id
    transition(capture, "placement-minimum-including-declared-fill",
               masks["pre_minimum_gas"], masks["post_minimum_gas"])
    transition(capture, "placement-threshold-and-minimum", values["labels"], values["result"])
    state(capture, "placement-minimum-labels", values["result"])


def state(capture, name, labels):
    value = np.asarray(labels)
    if "last_labels" in capture:
        transition(capture, name, capture["last_labels"], value)
    capture["last_labels"] = value.copy()


def finish(capture, labels, classes, source_box_xywh):
    """Compose actual branch populations, then apply only the storage mask."""
    labels = np.asarray(labels)
    shape = labels.shape
    p, r = (np.full(shape, np.nan, np.float32) for _ in range(2))
    codes = np.zeros(shape, np.uint8)

    def assign(part, mask, code, target=None):
        if not part or "pixel" not in part:
            return
        dest = (slice(None), slice(None)) if target is None else target
        keep = np.asarray(mask) & part["support"]
        p[dest][keep] = part["pixel"][keep]
        r[dest][keep] = part["regional"][keep]
        codes[dest][keep] = code

    placement = capture.get("placement", {})
    if placement.get("unsupported"):
        raise ValueError(placement["unsupported"])
    if "pixel" in placement:
        assign(placement, placement["support"], CODES["placement"])
    else:
        edge = capture["edge"]
        assign(edge, edge["support"], CODES["edge"])
        if "inherited_selection" in capture.get("composition", {}):
            composition = capture["composition"]
            centre = capture["centre"].get("seam_pair", capture["centre"])
            x, y, width, height = source_box_xywh
            target = (slice(y, y + height), slice(x, x + width))
            # The original seam chooses inherited scores on old_domain & ~ring;
            # it chooses the fresh edge pair everywhere else.
            selected = composition["inherited_selection"][target]
            p[target][selected] = np.nan
            r[target][selected] = np.nan
            codes[target][selected] = CODES["unscored"]
            assign(centre, selected, CODES["centre"], target)
    scored = (codes != CODES["unscored"]) & np.asarray(classes["modelled"], bool)
    for name in ("filled_observed", "filled_unobserved"):
        scored &= ~np.asarray(classes[name], bool)
    codes[~scored] = CODES["unscored"]
    p[~scored], r[~scored] = np.nan, np.nan
    before_mask = dict(pixel=identity(p), regional=identity(r))
    outside = scored & ~np.isin(labels, (1, 2))
    p[outside], r[outside] = np.float32(0), np.float32(0)
    arrays = dict(pixel_probability=p, regional_probability=r, branch_support=codes)
    validate_arrays(arrays, labels=labels)
    decisions, transitions = {}, []
    for name in ("edge", "centre", "composition", "placement", "postprocessing"):
        part = capture.get(name, {})
        if "decision" in part:
            decisions[name] = deepcopy(part["decision"])
        if "encoding" in part:
            decisions[name + "_encoding"] = deepcopy(part["encoding"])
        if "placement_mode" in part:
            decisions[name + "_mode"] = dict(placement=part["placement_mode"], light_epoch_id=part["light_epoch_id"])
        if "seam_pair" in part:
            seam = part["seam_pair"]
            if identity(seam["actual_returned_selection"]) != seam["decision"]["selected"]:
                raise ValueError("Captured seam pair differs from the original returned decision")
            decisions[name + "_seam"] = deepcopy(seam["decision"])
        transitions.extend(deepcopy(part.get("transitions", [])))
    capture["product"] = dict(arrays=arrays, evidence=dict(
        decisions=decisions, ordered_transitions=transitions,
        pre_storage_mask_pair=before_mask, scored_outside_final_co2_zeroed=int(outside.sum()),
        branch_counts={name: int(np.count_nonzero(codes == code)) for name, code in CODES.items()},
        unscored_policy="NaN takes precedence over outside-CO2 zero; no invented score",
        score_feedback=False, score_reconstruction=False, quantization=False))


def validate_arrays(arrays, *, labels=None):
    if type(arrays) is not dict or set(arrays) != set(ARRAYS):
        raise ValueError("Exactly two probabilities and one branch/support array required")
    p, r, codes = (np.asarray(arrays[name]) for name in ARRAYS)
    if (p.dtype != np.float32 or r.dtype != np.float32 or codes.dtype != np.uint8
            or p.ndim != 2 or not p.size or r.shape != p.shape or codes.shape != p.shape
            or not np.isin(codes, tuple(CODES.values())).all()):
        raise ValueError("Exact float32/uint8 probability storage grid required")
    scored = codes != CODES["unscored"]
    if (not np.isnan(p[~scored]).all() or not np.isnan(r[~scored]).all()
            or not np.isfinite(p[scored]).all() or not np.isfinite(r[scored]).all()
            or np.any((p[scored] < 0) | (p[scored] > 1))
            or np.any((r[scored] < 0) | (r[scored] > 1))):
        raise ValueError("Probability values and explicit unscored support disagree")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != p.shape:
            raise ValueError("Probability labels/grid differ")
        outside = scored & ~np.isin(labels, (1, 2))
        if np.any(p[outside] != 0) or np.any(r[outside] != 0):
            raise ValueError("Scored probabilities outside final CO2 must be exactly zero")
    return arrays


def _head_bindings(models):
    result = {name: deepcopy(models[name]["documents"]) for name in ("native", "edge")}
    placement = models["placement"]
    result["placement"] = {name: deepcopy(placement[name])
                           for name in ("pixel_model", "regional_model")}
    if placement.get("projected_models") is not None:
        result["placement"]["projected_models"] = deepcopy(placement["projected_models"])
    validate_heads(result)
    return result


def validate_heads(heads):
    if type(heads) is not dict or set(heads) != {"native", "edge", "placement"}:
        raise ValueError("Original centre, edge and placement head bindings required")
    def visit(value):
        if type(value) is not dict or not value:
            raise ValueError("Exact original head descriptors required")
        if "path" in value:
            if (not isinstance(value["path"], str) or not value["path"]
                    or re.fullmatch("[0-9a-f]{64}", str(value.get("sha256"))) is None):
                raise ValueError("Head source path/SHA required; no redumped model identity")
        else:
            for child in value.values():
                visit(child)
    visit(heads)


def write_product(stage, spec, capture, *, model_bindings, phase_configuration, labels, label_descriptor):
    from ..mass.n1_pipeline import _write_json, _write_npz
    product = capture["product"]
    arrays = validate_arrays(product["arrays"], labels=labels)
    archive = _write_npz(Path(stage) / "phase_probabilities.npz", arrays)
    logical = sum(value.nbytes for value in arrays.values())
    retained = {}
    def count_arrays(value):
        if isinstance(value, np.ndarray):
            retained[id(value)] = value.nbytes
        elif isinstance(value, dict):
            for child in value.values():
                count_arrays(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                count_arrays(child)
    count_arrays(capture)
    receipt = dict(format=FORMAT, complete=True, source=deepcopy(spec["source"]),
        case_id=spec["case_id"], encoding=ENCODING, branch_codes=CODES,
        phase_configuration=deepcopy(phase_configuration), head_bindings=_head_bindings(model_bindings),
        labels=deepcopy(label_descriptor), final_labels=identity(labels),
        decision_sources={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("infer.py", "rules.py", "domain.py", "probability_storage.py")},
        arrays={name: identity(value) for name, value in arrays.items()}, archive=archive,
        evidence=deepcopy(product["evidence"]), logical_bytes=logical,
        captured_retained_array_bytes=sum(retained.values()),
        memory_accounting_scope="unique retained capture arrays only; not RSS, transient peak, or admission",
        compressed_bytes=archive["size_bytes"], projected_frames=3090,
        projected_3090_compressed_bytes=archive["size_bytes"] * 3090,
        projection_basis="this actual frame only; not an admitted job budget or measured maximum",
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    saved = _write_json(Path(stage) / "phase_probabilities_receipt.json", receipt)
    return dict(archive=archive, receipt=saved)


def validate_receipt(receipt, roles, *, source, case_id, shape=None):
    required = {"format", "complete", "source", "case_id", "encoding", "branch_codes",
                "phase_configuration", "head_bindings", "labels", "final_labels", "decision_sources",
                "arrays", "archive", "evidence",
                "logical_bytes", "captured_retained_array_bytes", "memory_accounting_scope",
                "compressed_bytes", "projected_frames",
                "projected_3090_compressed_bytes", "projection_basis", "producer_sha256"}
    if (type(receipt) is not dict or set(receipt) != required or receipt["format"] != FORMAT
            or receipt["complete"] is not True or not same_metadata(receipt["source"], source)
            or receipt["case_id"] != case_id or receipt["encoding"] != ENCODING
            or not same_metadata(receipt["branch_codes"], CODES) or not same_metadata(receipt["archive"], roles["archive"])
            or set(receipt["arrays"]) != set(ARRAYS)):
        raise ValueError("Probability receipt/source/archive contract differs")
    validate_heads(receipt["head_bindings"])
    for descriptor in (receipt["labels"], receipt["phase_configuration"]):
        if (type(descriptor) is not dict or not isinstance(descriptor.get("path"), str)
                or re.fullmatch("[0-9a-f]{64}", str(descriptor.get("sha256"))) is None):
            raise ValueError("Original phase configuration and final label descriptors required")
    if (type(receipt["decision_sources"]) is not dict
            or set(receipt["decision_sources"]) != {"infer.py", "rules.py", "domain.py", "probability_storage.py"}
            or any(re.fullmatch("[0-9a-f]{64}", str(v)) is None for v in receipt["decision_sources"].values())):
        raise ValueError("Actual decision/capture source identities required")
    for name, row in receipt["arrays"].items():
        if (type(row) is not dict or set(row) != {"dtype", "shape", "sha256"}
                or row["dtype"] != ("|u1" if name == "branch_support" else "<f4")
                or len(row["shape"]) != 2 or any(type(v) is not int or v <= 0 for v in row["shape"])
                or re.fullmatch("[0-9a-f]{64}", str(row["sha256"])) is None
                or shape is not None and row["shape"] != list(shape)):
            raise ValueError("Probability array dtype/grid/identity differs")
    dimensions = receipt["arrays"][ARRAYS[0]]["shape"]
    if (any(row["shape"] != dimensions for row in receipt["arrays"].values())
            or type(receipt["logical_bytes"]) is not int
            or receipt["logical_bytes"] != dimensions[0] * dimensions[1] * 9
            or type(receipt["captured_retained_array_bytes"]) is not int
            or receipt["captured_retained_array_bytes"] < receipt["logical_bytes"]
            or receipt["memory_accounting_scope"] != "unique retained capture arrays only; not RSS, transient peak, or admission"
            or type(receipt["compressed_bytes"]) is not int or receipt["compressed_bytes"] <= 0
            or receipt["compressed_bytes"] != roles["archive"]["size_bytes"]
            or type(receipt["projected_frames"]) is not int or receipt["projected_frames"] != 3090
            or receipt["projection_basis"] != "this actual frame only; not an admitted job budget or measured maximum"
            or receipt["projected_3090_compressed_bytes"] != receipt["compressed_bytes"] * 3090
            or re.fullmatch("[0-9a-f]{64}", str(receipt["producer_sha256"])) is None):
        raise ValueError("Probability byte accounting differs")
    final = receipt["final_labels"]
    if (type(final) is not dict or set(final) != {"dtype", "shape", "sha256"}
            or final["dtype"] != "|u1" or final["shape"] != dimensions
            or re.fullmatch("[0-9a-f]{64}", str(final["sha256"])) is None):
        raise ValueError("Actual final label array identity differs")
    evidence = receipt["evidence"]
    if (type(evidence) is not dict
            or set(evidence) != {"decisions", "ordered_transitions", "pre_storage_mask_pair",
                                 "scored_outside_final_co2_zeroed", "branch_counts", "unscored_policy",
                                 "score_feedback", "score_reconstruction", "quantization"}
            or type(evidence["branch_counts"]) is not dict or set(evidence["branch_counts"]) != set(CODES)
            or any(type(v) is not int or v < 0 for v in evidence["branch_counts"].values())
            or sum(evidence["branch_counts"].values()) != dimensions[0] * dimensions[1]
            or type(evidence["ordered_transitions"]) is not list or not isinstance(evidence["decisions"], dict)
            or type(evidence["scored_outside_final_co2_zeroed"]) is not int
            or not 0 <= evidence["scored_outside_final_co2_zeroed"] <= dimensions[0] * dimensions[1]):
        raise ValueError("Explicit actual branch/support and ordered transition evidence required")
    if any(evidence.get(key) is not False for key in ("score_feedback", "score_reconstruction", "quantization")):
        raise ValueError("Probability evidence must retain no-feedback/no-reconstruction contract")
    return receipt


def read_product(roles, *, reader, source, case_id, labels=None):
    """Use the existing exact reader (including its indexed resolver), no fallback."""
    from ..mass.n1_pipeline import DataReader
    if type(reader) is not DataReader:
        raise TypeError("The existing exact DataReader is required")
    if type(roles) is not dict or set(roles) != {"archive", "receipt"}:
        raise ValueError("Explicit probability archive and receipt roles required")
    receipt = validate_receipt(reader.json(roles["receipt"]), roles, source=source,
                               case_id=case_id, shape=None if labels is None else labels.shape)
    if labels is None:
        from PIL import Image
        label_path = reader.path(receipt["labels"])
        with Image.open(label_path) as image:
            labels = np.asarray(image).copy()
        reader.path(receipt["labels"])
    if identity(labels) != receipt["final_labels"]:
        raise ValueError("Probability storage mask is not the actual bound final label product")
    path = reader.path(roles["archive"])
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAYS):
            raise ValueError("Undeclared probability archive members")
        arrays = {name: archive[name].copy() for name in ARRAYS}
    reader.path(roles["archive"])
    validate_arrays(arrays, labels=labels)
    if {name: identity(value) for name, value in arrays.items()} != receipt["arrays"]:
        raise ValueError("Stored probabilities differ from decision-used byte identities")
    counts = {name: int(np.count_nonzero(arrays["branch_support"] == code)) for name, code in CODES.items()}
    if counts != receipt["evidence"]["branch_counts"]:
        raise ValueError("Actual stored branch counts differ from their receipt")
    return dict(arrays=arrays, receipt=receipt)

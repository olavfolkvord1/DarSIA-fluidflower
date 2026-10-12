"""Pure saved-label revision: remove contact-fill reason (b), nothing else."""
from __future__ import annotations

import numpy as np

CONTRACT = "ffac-g3-t4c-p1-final3-candidate-v1"
SPEC_SHA256 = "f70dc1bcc6de5b3118a1983547bcca0aa41f057ab829b217c2068ea2edce0a77"
RULE = {
    "name": "drop_contact_fill_reason_b_v1",
    "operation": "P1-final2 label 1 to 0 exactly on contact fill reason (b)",
    "contact_fill_a_maximum_run_mm": 1.5,
    "training": False,
    "inference": False,
    "backend": "cpu",
}


def remove_contact_fill_b(labels, reason_a, reason_b, parent_contact_mask):
    """Validate the parent's fixed reason partition before applying one change."""
    labels = np.asarray(labels)
    masks = [np.asarray(value) for value in (reason_a, reason_b, parent_contact_mask)]
    if labels.dtype != np.uint8 or labels.ndim != 2:
        raise ValueError("Production labels must be a uint8 image")
    if any(mask.dtype != np.bool_ or mask.shape != labels.shape for mask in masks):
        raise ValueError("Contact flags must be Boolean masks on the exact label grid")
    a, b, contact = masks
    if np.any(a & b) or not np.array_equal(contact, a | b):
        raise ValueError("Parent contact mask differs from its disjoint reason partition")
    if np.any(labels[b] != 1):
        raise ValueError("Every contact-fill reason (b) pixel must be parent label 1")
    if np.any(~np.isin(labels, (0, 1, 2, 255))):
        raise ValueError("Unexpected saved production category")
    result = labels.copy()
    result[b] = 0
    return result

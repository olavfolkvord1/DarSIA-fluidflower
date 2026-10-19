"""TESTONLY: R3, the RAW39 reference, is admitted only by its exact release entry.

Like N43, R3 is pinned in REFERENCE_RELEASES: its two answer records with
their deciders and its accepted plan's SHA-256 (N134, N135, N136). Any other
authority set, record or plan is refused. No actual namespace or plan is read.
"""
from copy import deepcopy

import pytest

from darsia_cache import first_qualification as first
from darsia_cache.policy import CachePolicyError
from .test_n43_reference_administration import administration_TESTONLY

R3_PLAN = "3ea1caa8ff432f5c17d2904f7f43080e23aeac039a447d8667a1586749e0426e"
R3_GO = ("N134-OLAV-r3-inputs-into-a-new-ff-ac-folder-reference-build-once-fresh-cache-released-"
         "after-the-code-change")
R3_PLAN_RECORD = "N136-CLAUDE-r3-plan-3ea1caa8-entitlement-65728995330B-pinned-for-the-exact-code-entry"


def r3_TESTONLY(answers=None, authority_ids=("N134", "N136"), plan=R3_PLAN):
    state, value, _read = administration_TESTONLY()
    value.update(authority_ids=list(authority_ids),
                 accepted_plan=dict(path="C:/TESTONLY/R3_N135_accepted_plan.json", sha256=plan, size_bytes=1))
    answers = answers if answers is not None else [
        dict(id=R3_GO, decided_by="olav"), dict(id=R3_PLAN_RECORD, decided_by="claude")]

    def read(binding):
        return deepcopy(answers if binding == value["answers_source"] else {"TESTONLY": "read exact plan"})

    return state, value, read


def test_r3_is_admitted_by_its_exact_release_entry():
    state, value, read = r3_TESTONLY()
    actual = first.checked_reference_administration(value, state, read=read)
    assert actual["authority_ids"] == ["N134", "N136"]
    assert actual["aggregate_entitlement_bytes"] == first.REFERENCE_ENTITLEMENT_BYTES


@pytest.mark.parametrize("kwargs", [
    dict(plan="3ea1caa8" + "0" * 56),  # Only the short prefix matches.
    dict(authority_ids=("N134",)),
    dict(authority_ids=("N134", "N137")),
    dict(authority_ids=("N136", "N134")),  # Order is part of the entry.
    dict(answers=[dict(id=R3_GO, decided_by="olav")]),
    dict(answers=[dict(id=R3_GO, decided_by="claude"), dict(id=R3_PLAN_RECORD, decided_by="claude")]),
    dict(answers=[dict(id=R3_GO, decided_by="olav"), dict(id=R3_PLAN_RECORD, decided_by="olav")]),
    dict(answers=[]),
])
def test_r3_refuses_any_other_plan_authority_set_or_record(kwargs):
    state, value, read = r3_TESTONLY(**kwargs)
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


def test_r3_keeps_the_n43_entitlement():
    state, value, read = r3_TESTONLY()
    value["aggregate_entitlement_bytes"] = first.REFERENCE_ENTITLEMENT_BYTES + 1
    with pytest.raises(CachePolicyError):
        first.checked_reference_administration(value, state, read=read)


def test_released_reference_names_only_exact_entries():
    assert first.released_reference(["N43", "Q109", "Q106"]) is first.REFERENCE_RELEASES[("N43", "Q109", "Q106")]
    assert first.released_reference(["N134", "N136"])["plan_sha256"] == R3_PLAN
    assert first.released_reference(["N134"]) is None
    assert first.released_reference(("N134", "N136")) is None  # A list, as stored in JSON.
    assert first.released_reference([["N134"], "N136"]) is None
    assert first.released_reference(None) is None

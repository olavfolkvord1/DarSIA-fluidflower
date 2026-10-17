"""Concrete operational adapters; no numerical or GUI import.

No automatic cleanup or fabricated child retirement. The display writer returns
its original scope so an actual consumer-lifecycle owner can later close it.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from .bootstrap import (current_import_lease, _owner, select_entry_role,
                        RECEIPT_ENV, POLICY_ENV, EXPECTED_STORES_ENV)
from .control import OwnedScratch
from .policy import CachePolicyError
from .windows import WindowsNamespace


def bind_actual_worker_owner(owner, *, slot=None, generation=None, role=None):
    """Attach the existing semantic worker identity to its early native owner.

    Call from the existing non-GUI worker boundary; the early registration has
    already happened before spawn imports. This is not a retirement callback.
    """
    lease=current_import_lease()
    actual,_=_owner()
    if (not isinstance(owner,dict) or owner.get("pid")!=actual["pid"]
            or owner.get("process_create_time")!=actual["process_create_time"]):
        raise CachePolicyError("Semantic worker PID/epoch differs from current native owner")
    identity=hashlib.sha256(json.dumps(owner,sort_keys=True).encode()).hexdigest()
    return lease._record(lease.reservation["token"]+".worker-"+identity+".json",
        dict(owner=owner,cache_native_owner=actual,slot=slot,generation=generation,role=role,
             reservation=lease.reservation,role_identifier=lease.role_identifier,
             entry_kind=lease.entry_kind,retirement=False))


def require_same_startup_stores(stores):
    """Supplement original startup validation; never replace its Git/lock checks."""
    lease=current_import_lease()
    if lease.entry_kind!="worker" or lease.purpose!="jobs":
        raise CachePolicyError("Worker startup requires its configured worker/jobs role")
    if any(stores.source.get(k)!=lease.source[k] for k in ("path","sha256","size_bytes")):
        raise CachePolicyError("Worker startup stores differs from pre-import cache binding")
    return lease


def record_claim_cache_requirement(claim, *, applicable_profile, maximum_bytes, reservation, owner):
    """Record an already reserved task supplement; not a permissive adapter.

    The existing ClaimCacheBudget must perform admission/reservation first. The
    caller supplies that original binding, not a self-reported coverage boolean.
    """
    lease=current_import_lease()
    if type(maximum_bytes) is not int or maximum_bytes<0 or not isinstance(applicable_profile,dict):
        raise CachePolicyError("Original applicable cache profile/maximum required")
    if reservation.get("token")!=claim.token:
        raise CachePolicyError("Original claim and cache reservation differ")
    with lease.ledger.locked():
        lease.ledger.require_reservation(reservation,owner,maximum_bytes)
    return lease._record(lease.reservation["token"]+".claim-"+claim.token+".json",
        dict(original_claim_token=claim.token,import_reservation=lease.reservation,
             task_cache_reservation=reservation,task_owner=owner,
             applicable_profile=applicable_profile,maximum_bytes=maximum_bytes,
             per_claim_admission_must_be_proven_separately=True))


def optional_import_lease():
    """Only the original missing-lease case means unmanaged ordinary import."""
    try:
        return current_import_lease()
    except CachePolicyError:
        return None


def independent_worker_environment(stores, *, machine, series_id,
                                   explicit_stores, parent_environment):
    """Select original stores/worker role; child creates its independent lease.

    No scope allocation here, no command rewriting, and no GUI retirement claim.
    The only removed environment variable is the parent service receipt.
    """
    if stores.machine_id.casefold()!=str(machine).casefold():
        raise CachePolicyError("Selected worker stores belongs to another machine")
    if series_id is not None:
        stores.roots(series_id)  # Original series alias validation, no discovery.
    select_entry_role(stores.document,entry_kind="worker")
    source={key:stores.source[key] for key in ("path","sha256","size_bytes")}
    result=dict(parent_environment)
    if not explicit_stores and POLICY_ENV in result:
        if Path(result[POLICY_ENV]).resolve()!=Path(source["path"]).resolve():
            raise CachePolicyError("Inherited cache policy differs from the original default worker stores")
    result.pop(RECEIPT_ENV,None)
    result[EXPECTED_STORES_ENV]=json.dumps(source,sort_keys=True,separators=(",",":"))
    return result


def cooperative_child_environment(*, parent_environment):
    """Inherit the exact admitted jobs family; workspace writers stay unqualified."""
    lease=current_import_lease()
    if lease.purpose!="jobs":
        raise CachePolicyError("Legacy worker needs an explicitly configured jobs service role")
    result=dict(parent_environment)
    if RECEIPT_ENV not in result or json.loads(result[RECEIPT_ENV])!=lease.receipt:
        raise CachePolicyError("Child environment lacks the exact original service receipt")
    return result


def record_cache_observation(kind, details):
    """Record only an actual existing observation; never close a reservation."""
    allowed={"worker_spawn_intent","worker_spawn_observed","worker_children_retired",
             "legacy_spawn_intent","legacy_spawn_observed"}
    if kind not in allowed or not isinstance(details,dict):
        raise CachePolicyError("An exact supported operational observation is required")
    lease=current_import_lease();actual,_=_owner()
    from .service_lifecycle import ServiceLifecycle
    ServiceLifecycle(lease).record_launch(kind,details)  # Every allowed launch shares the seal transaction.
    return lease._record(lease.reservation["token"]+".observation-"+uuid.uuid4().hex+".json",
        dict(kind=kind,details=details,observer=actual,reservation=lease.reservation,
             service_or_family_retirement_claimed=False))


def bind_legacy_worker_identity(worker_id):
    lease=current_import_lease()
    if lease.purpose!="jobs":
        raise CachePolicyError("Legacy worker lacks its admitted jobs service role")
    actual,_=_owner();actual["worker_id"]=worker_id
    return bind_actual_worker_owner(actual,role="legacy_worker")


_DISPLAY_ALLOCATIONS=[]
_DISPLAY_REUSE={}


def reusable_display_png(*, logical_gate):
    """Reuse only this process's own closed immutable allocation for this lease.

    The key retains the original source signature, phase and display policy.
    No legacy directory, arbitrary existing PNG or previous lease is adopted.
    """
    lease=current_import_lease()
    result=_DISPLAY_REUSE.get((lease.reservation["token"],logical_gate))
    if result is None:return None
    allocation=result["allocation"];frozen=result["owned_frozen"]
    if (result["immutable_writer_closed"] is not True or allocation.outcome["outcome"]!="success"
            or not allocation.tree.closed or frozen["unresolved_creates"]
            or frozen["format"]!="darsia-windows-owned-tree-v2"
            or frozen["scope"]!=allocation.root.name
            or set(frozen["files"])!={"display.png","provenance.json"}
            or allocation.policy!=lease.policy or allocation.ledger.admission!=lease.admission):
        raise CachePolicyError("Display reuse requires the original closed owned allocation")
    namespace=WindowsNamespace(allocation.root,frozen["root"])
    try:
        if set(namespace.names())!=set(frozen["files"]):
            raise CachePolicyError("Display namespace changed; no reuse or legacy adoption")
        for name,expected in frozen["files"].items():
            handle,identity=namespace.api.child(namespace.handle,name,directory=False)
            try:
                if identity.document()!=expected:
                    raise CachePolicyError("Owned display immutable file version changed; no reuse")
            finally:namespace.api.close(handle)
    finally:namespace.close()
    return Path(result["path"])


def managed_display_png(png_bytes, provenance, *, logical_gate):
    """Keep local immutable allocation receipts until real process-reader closure.

    There is no function-return, widget or atexit cleanup. Exact future reader
    retirement and external writer closure still need qualification.
    """
    lease=current_import_lease();key=(lease.reservation["token"],logical_gate)
    existing=reusable_display_png(logical_gate=logical_gate)
    if existing is not None:
        result=_DISPLAY_REUSE[key]
        metadata=(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
        if (hashlib.sha256(png_bytes).hexdigest()!=result["png_sha256"]
                or hashlib.sha256(metadata).hexdigest()!=result["provenance_sha256"]):
            raise CachePolicyError("Repeated display key has different encoded content/provenance")
        return existing
    result=write_display_png(png_bytes,provenance,logical_gate=logical_gate)
    _DISPLAY_ALLOCATIONS.append(result)
    _DISPLAY_REUSE[key]=result
    return Path(result["path"])


def write_display_png(png_bytes, provenance, *, logical_gate):
    """Encode before this call; reserve exact lengths and write immutable bytes.

    No Pillow import or pixel/mask changes. Native arrays/products are not an
    input contract. No publication, success cleanup or reader retirement is
    claimed merely by returning a local display path.
    """
    if not isinstance(png_bytes,bytes) or not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CachePolicyError("Already encoded display PNG bytes required")
    if not isinstance(provenance,dict) or provenance.get("model_scores_changed") is not False:
        raise CachePolicyError("Explicit unchanged display provenance required")
    metadata=(json.dumps(provenance,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    lease=current_import_lease();owner,_=_owner();token=uuid.uuid4().hex
    caps={"display.png":len(png_bytes),"provenance.json":len(metadata)}
    with lease.ledger.locked():
        reservation=lease.ledger.reserve(token,owner,sum(caps.values()))
        allocation=OwnedScratch(lease.policy,"display",token,owner,logical_gate,caps,
            namespace_admission=lease.admission,ledger=lease.ledger,reservation=reservation)
    try:
        path=allocation.write_bytes("display.png",png_bytes)
        allocation.write_bytes("provenance.json",metadata)
        publication=dict(format="closed-local-display-png-v1",path=str(path),
            sha256=hashlib.sha256(png_bytes).hexdigest(),size_bytes=len(png_bytes),
            NAS_publication_claimed=False)
        allocation.finish("success",utc=datetime.now(timezone.utc).isoformat(),publication=publication)
        return dict(path=path,allocation=allocation,reservation=reservation,
            immutable_writer_closed=True,
            owned_frozen=allocation.tree.frozen(),
            png_sha256=hashlib.sha256(png_bytes).hexdigest(),
            provenance_sha256=hashlib.sha256(metadata).hexdigest(),reader_retirement_pending=True,
            created_utc=datetime.now(timezone.utc).isoformat())
    except BaseException:
        # Original create/write evidence and reservation remain. Do not unlink.
        allocation.close_handles()
        raise


def seal_current_service(*, logical_gate, outcome, utc, traceback_text=None, publication=None):
    """Explicit parent intent only; never called automatically from stopped/atexit."""
    from .service_lifecycle import ServiceLifecycle
    return ServiceLifecycle(current_import_lease()).seal(logical_gate=logical_gate,
        outcome=outcome,utc=utc,traceback_text=traceback_text,publication=publication)


def service_retention_eligibility(original_lease, **proofs):
    """External observer supplies original accepted lease/ledger and real proofs.

    No lease reconstruction, provider, root or quota is guessed by this API.
    Result is never authority to delete or to close the service reservation.
    """
    from .service_lifecycle import ServiceLifecycle
    return ServiceLifecycle(original_lease).retention_eligibility(**proofs)

"""Separate 127-frame selected metadata/context extension; no original edits.

Metadata preparation enumerates declarations, never decodes/hashes scientific
payloads or qualifies them. Root stages/verifies the flat bytes and owns actual
context preparation/run. The original four-case configuration/case validators
are called on the preserved base and never modified or presented as a 127 proof.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import re

SELECTION_FORMAT = "darsia-production-precision127-selection-v1"
CONFIGURATION_FORMAT = "darsia-production-precision127-native-v1"
COUNTS = {"ac18":44, "ac52":39, "ac19":44}
RUN_ORDER = tuple(COUNTS)
LIMITS_SHA256 = "36be522e1323c6bf376f0868e3ab52a21090185d6908322d7c48ed450df2ed96"


def _clone(function, **overrides):
    result=FunctionType(function.__code__, dict(function.__globals__, **overrides),
                        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def _descriptor(bound):
    if (not isinstance(bound,Mapping) or set(bound) not in
        ({"path","sha256"},{"path","sha256","size_bytes"})
        or not isinstance(bound["path"],str) or not bound["path"]
        or re.fullmatch("[0-9a-f]{64}",bound.get("sha256","")) is None
        or ("size_bytes" in bound and (type(bound["size_bytes"]) is not int or bound["size_bytes"]<0))):
        raise ValueError("Exact frozen descriptor required")
    return dict(bound)


def _indexed_cases(entries):
    """Exact full indexed rows; no nearest-frame replacement/selection/fit."""
    import math
    result={}
    for run in RUN_ORDER:
        selected=[entry for entry in entries if entry.get("run")==run]
        if len(selected)!=COUNTS[run]:
            raise ValueError(f"Locked {run} must contain exactly {COUNTS[run]} original indexed entries")
        if any(type(entry.get("frame_index")) is not int or entry["frame_index"]<0
               or not math.isfinite(entry["elapsed_h"]) for entry in selected):
            raise ValueError("Native integer frame identity/finite camera time required")
        selected.sort(key=lambda row:(row["elapsed_h"],row["frame_index"]))
        if any(a["elapsed_h"]>=b["elapsed_h"] for a,b in zip(selected,selected[1:])):
            raise ValueError("Whole-run camera time must be strictly chronological")
        anchors=[row for row in selected if float(row["selected_target_h"])==2.25]
        if len(anchors)!=1:
            raise ValueError("Each full run requires its one original source-bound 2.25 h anchor")
        for entry in selected:
            case_id=f"{run}_f{entry['frame_index']}"
            if case_id in result:
                raise ValueError("Whole-run native source frame repeated")
            result[case_id]={"case_id":case_id,"entry":deepcopy(entry),"anchor_entry":deepcopy(anchors[0])}
    return result


def build_selection(campaign_descriptor, metadata_reader, *, native_module, limits_descriptor):
    """Original all-39/PASS/index validation runs before deriving the exact subset."""
    limits=_descriptor(limits_descriptor)
    if limits["sha256"]!=LIMITS_SHA256:
        raise ValueError("Exact approved equivalence addendum8v2 required")
    metadata_reader.path(limits)
    campaign_descriptor=_descriptor(campaign_descriptor)
    campaign=metadata_reader.json(campaign_descriptor)
    runner=native_module._modules()["runner"]
    if (campaign.get("contract")!=runner.CAMPAIGN_CONTRACT
        or campaign.get("source_equivalence_exact") is not True
        or campaign.get("production_automatically_allowed") is not False):
        raise ValueError("Original source-bound selected campaign required")
    validated=runner.validate_selected_plan(campaign["plan"],metadata_reader)
    cases=_indexed_cases(validated["entries"])
    return dict(format=SELECTION_FORMAT,complete=True,
        limits=limits,source_campaign=campaign_descriptor,source_selected_plan=campaign["plan"],
        source_selected_metadata_index=validated["plan"]["selected_metadata_index"],
        case_count=127,run_order=list(RUN_ORDER),run_counts=dict(COUNTS),
        cases=list(cases.values()),selection_method="exact_original_index_subset_chronological_whole_runs",
        original_all_39_metadata_guards_replayed=True,scientific_arrays_read=False,
        scientific_payload_bytes_verified=False,scientific_admission_granted=False,
        production_permission_granted=False)


def validate_selection(selection, *, indexed_entries):
    expected=_indexed_cases(indexed_entries)
    if (selection.get("format")!=SELECTION_FORMAT or selection.get("complete") is not True
        or selection.get("limits",{}).get("sha256")!=LIMITS_SHA256
        or selection.get("case_count")!=127 or selection.get("run_order")!=list(RUN_ORDER)
        or selection.get("run_counts")!=COUNTS
        or selection.get("cases")!=list(expected.values())
        or selection.get("scientific_arrays_read") is not False
        or selection.get("scientific_payload_bytes_verified") is not False
        or selection.get("scientific_admission_granted") is not False
        or selection.get("production_permission_granted") is not False):
        raise ValueError("127 selection is not the exact unchanged chronological original index subset")
    return expected


def validate_configuration(configuration, original_four_configuration, selection, *, native_module,
                           indexed_entries):
    """Old four-case guard stays strict; only explicit scope/config selection differs."""
    original=native_module._configuration(original_four_configuration)
    if (configuration.get("format")!=CONFIGURATION_FORMAT
        or configuration.get("limits",{}).get("sha256")!=LIMITS_SHA256):
        raise ValueError("Explicit locked127 CPU precision test configuration required")
    _descriptor(configuration["profile_selection"])
    _descriptor(configuration["original_four_configuration"])
    _descriptor(configuration["limits"])
    # All original mode, purpose, campaign, B1 and permission values are literal.
    stripped={k:v for k,v in configuration.items() if k not in
              {"format","profile_selection","original_four_configuration","limits"}}
    original_stripped={k:v for k,v in original.items() if k not in {"format","profile_selection"}}
    if stripped!=original_stripped:
        raise ValueError("Whole127 extension changed an original native guard/setting/source binding")
    if (selection["source_campaign"]!=original["campaign"]
        or selection["limits"]!=configuration["limits"]):
        raise ValueError("Selection parent campaign/limit binding differs")
    return validate_selection(selection,indexed_entries=indexed_entries)


def _configuration_and_cases(configuration, reader, native_module):
    original=reader.json(_descriptor(configuration["original_four_configuration"]))
    native_module._configuration(original)
    old_selection=reader.json(original["profile_selection"])
    native_module._cases(old_selection)  # genuine old4 validation, not reused as127 proof
    campaign=reader.json(original["campaign"])
    modules=native_module._modules()
    validated=modules["runner"].validate_selected_plan(campaign["plan"],reader)
    selection=reader.json(configuration["profile_selection"])
    if (selection.get("source_selected_plan")!=campaign["plan"]
        or selection.get("source_selected_metadata_index")!=validated["plan"]["selected_metadata_index"]):
        raise ValueError("Selection source plan/index descriptor differs")
    cases=validate_configuration(configuration,original,selection,native_module=native_module,
                                  indexed_entries=validated["entries"])
    return cases,modules


def prepare_precision_context(configuration, reader, *, native_module=None, boundary_guard=None):
    """Root-only numerical SETUP entrypoint; never called by metadata preparation.

    The unchanged original setup may read physical coordinate/floor assets.
    Every original candidate/source/freeze/index/live check remains in its body.
    This entrypoint grants no Root admission or scientific qualification.
    """
    if native_module is None:
        from darsia.fluidflower import production_native as native_module
    cases,modules=_configuration_and_cases(configuration,reader,native_module)
    def config(value):
        if value!=configuration: raise ValueError("Prepared configuration changed")
        return value
    def selected(value):
        # _configuration_and_cases already compares the full indexed entry list.
        if value!=reader.json(configuration["profile_selection"]):
            raise ValueError("Whole127 selection bytes/content changed")
        return cases
    prepare=_clone(native_module.prepare_native_context,_configuration=config,_cases=selected)
    context=prepare(configuration,reader,modules=modules,boundary_guard=boundary_guard)
    context.provenance.update(whole127_extension=True,whole127_selection=configuration["profile_selection"],
        original_four_configuration=configuration["original_four_configuration"],
        original_four_proof_claimed_for_127=False,scientific_eligibility=False,production_permission_granted=False)
    return context


def enumerate_precision_closure(configuration, metadata_reader, *, native_module=None,
                                size_resolver, include_anchors=True):
    """Flat exact source declarations grouped into shared and per-run working sets."""
    if include_anchors is not True:
        raise ValueError("Whole-run precision requires the original dilution anchors")
    if native_module is None:
        from darsia.fluidflower import production_native as native_module
    cases,modules=_configuration_and_cases(configuration,metadata_reader,native_module)
    all_scopes={}
    for run in RUN_ORDER:
        scope={key:value for key,value in cases.items() if value["entry"]["run"]==run}
        def config(value):
            if value!=configuration: raise ValueError("Enumerated configuration changed")
            return value
        def selected(value):
            if value!=metadata_reader.json(configuration["profile_selection"]):
                raise ValueError("Whole127 selection changed")
            return scope
        collector=_clone(native_module.enumerate_native_closure,_configuration=config,_cases=selected,
                         LOCKED_CASES=scope)
        all_scopes[run]=collector(configuration,metadata_reader,modules=modules,
                                 size_resolver=size_resolver,include_anchors=True)
    # Preserve normalized path + SHA + size identity, never deduplicate by SHA alone.
    from darsia.fluidflower.production_io import normalize_locator
    union={}
    for run,closure in all_scopes.items():
        if closure.get("scientific_arrays_read") is not False or closure.get("opaque_evidence_parsed") is not False:
            raise ValueError("A metadata closure attempted a scientific/opaque decode")
        for row in closure["assets"]:
            bound=_descriptor(row["descriptor"])
            key=normalize_locator(bound["path"])
            if key in union and (union[key]["descriptor"]["sha256"],union[key]["descriptor"]["size_bytes"])!=(bound["sha256"],bound["size_bytes"]):
                raise ValueError("One declared locator has contradictory frozen bytes")
            current=union.setdefault(key,dict(descriptor=bound,roles=set(),working_runs=set()))
            current["roles"].update(row["roles"]); current["working_runs"].add(run)
    # Explicit metadata supporting this extension is bound too; no self-binding.
    for name in ("original_four_configuration","limits"):
        bound=_descriptor(configuration[name]); bound.setdefault("size_bytes",size_resolver(bound))
        key=normalize_locator(bound["path"])
        current=union.setdefault(key,dict(descriptor=bound,roles=set(),working_runs=set()))
        if (current["descriptor"]["sha256"],current["descriptor"]["size_bytes"])!=(bound["sha256"],bound["size_bytes"]):
            raise ValueError("Extension metadata has contradictory frozen bytes")
        current["roles"].add(name);current["working_runs"].update(RUN_ORDER)
    assets=[]
    for key,row in sorted(union.items()):
        runs=[run for run in RUN_ORDER if run in row["working_runs"]]
        assets.append(dict(descriptor=row["descriptor"],roles=sorted(row["roles"]),
                           working_runs=runs,staging_group="shared" if len(runs)==3 else runs[0]))
    return dict(format="darsia-production-precision127-flat-closure-v1",complete=True,
        selection=configuration["profile_selection"],run_counts=dict(COUNTS),run_order=list(RUN_ORDER),
        case_ids=list(cases),includes_anchor_payloads=True,assets=assets,
        shared_roles=[index for index,row in enumerate(assets) if row["staging_group"]=="shared"],
        per_run_roles={run:[index for index,row in enumerate(assets) if run in row["working_runs"]]
                       for run in RUN_ORDER},
        scientific_arrays_read=False,scientific_payload_bytes_verified=False,
        opaque_evidence_parsed=False,scientific_admission_granted=False,production_permission_granted=False,
        worker_ownership_policy=dict(unit="whole_AC_run",max_workers=3,
            chronological_within_run=True,one_worker_per_run=True,
            static_host_assignment_before_start=True,common_task_queue=False,nas_work_locks=False,
            assignment_rule="Root longest-first using actual measured images/hour",
            automatic_host_assignment=False,host_assignment=None))


class MetadataReader:
    """Verified JSON only; declarations/stat for scientific paths, never payload IO.

    Explicit old aliases locate existing bound assets; never infer aliases by
    SHA. Additional JSON must have a frozen producer descriptor and is verified
    by the common NasIO before parsing. Scientific byte verification is deferred
    to Root's immutable staging/coordinator, not falsely claimed by path().
    """
    OPAQUE_SHA="4491b8ea239853d441ece789a371e92f6516874b6ea62308eeafb057981d938c"

    def __init__(self,manifest,staged_inputs,logical_aliases,*,io):
        from darsia.fluidflower.production_io import normalize_locator
        self.io,self.norm=io,normalize_locator
        self.bound,self.local,self.aliases={}, {}, {}
        self.documents,self.metadata_verified={}, {}
        if set(manifest["inputs"])!=set(staged_inputs):
            raise ValueError("Old flat manifest/stage roles differ")
        for role,bound in manifest["inputs"].items():
            bound=_descriptor(bound);local=_descriptor(staged_inputs[role])
            if (bound["sha256"],bound["size_bytes"])!=(local["sha256"],local["size_bytes"]):
                raise ValueError("Declared stage differs from immutable flat input bytes")
            key=self.norm(bound["path"])
            if key in self.bound and self.bound[key]!=bound:
                if (self.bound[key]["sha256"],self.bound[key]["size_bytes"])!=(bound["sha256"],bound["size_bytes"]):
                    raise ValueError("Conflicting old flat locator")
            self.bound[key],self.local[key]=bound,local
        for locator,target in logical_aliases.items():
            target=_descriptor(target);target_key=self.norm(target["path"])
            if target_key not in self.bound or (self.bound[target_key]["sha256"],self.bound[target_key]["size_bytes"])!=(target["sha256"],target["size_bytes"]):
                raise ValueError("Explicit alias target must bind the same normalized locator/SHA/size in old flat manifest")
            alias_key=self.norm(locator)
            if alias_key in self.bound and alias_key!=target_key:
                raise ValueError("Alias conflicts with canonical flat locator")
            self.aliases[alias_key]=target_key

    def register(self,bound):
        bound=_descriptor(bound);key=self.norm(bound["path"])
        if key in self.bound and self.bound[key]!=bound:
            raise ValueError("New metadata descriptor conflicts with frozen bytes")
        self.bound[key]=bound
        self.local[key]=bound

    def _identity(self,descriptor):
        # Preserve scientific array_key/role metadata outside the byte descriptor.
        value=_descriptor({key:descriptor[key] for key in ("path","sha256","size_bytes") if key in descriptor})
        key=self.norm(value["path"]);key=self.aliases.get(key,key)
        existing=self.bound.get(key)
        if existing is not None:
            if value["sha256"]!=existing["sha256"] or ("size_bytes" in value and value["size_bytes"]!=existing["size_bytes"]):
                raise ValueError("Requested frozen SHA/size differs from explicit flat/alias binding")
            return dict(path=value["path"],sha256=existing["sha256"],size_bytes=existing["size_bytes"]),self.local[key],existing
        path=Path(value["path"])
        actual_size=self.io.retry.run("metadata_descriptor_stat",path,lambda:path.stat().st_size)  # no scientific bytes read
        if "size_bytes" in value and value["size_bytes"]!=actual_size:
            raise ValueError("Declared additional frozen asset size differs")
        return {**value,"size_bytes":actual_size},{**value,"size_bytes":actual_size},{**value,"size_bytes":actual_size}

    def size(self,bound): return self._identity(bound)[0]["size_bytes"]

    def path(self,bound): return Path(self._identity(bound)[1]["path"])

    def array(self,bound):
        raise RuntimeError("Whole127 metadata cannot decode any scientific array")

    def json(self,descriptor):
        if (descriptor.get("sha256")==self.OPAQUE_SHA
            or Path(str(descriptor.get("path",""))).name.casefold().startswith("result_4491")):
            raise ValueError("Opaque saved PASS is hash/path evidence only, never JSON")
        logical,local,storage=self._identity(descriptor)
        if Path(local["path"]).suffix.casefold()!=".json":
            raise ValueError("Only frozen declared JSON metadata can be parsed")
        key=(self.norm(logical["path"]),logical["sha256"],logical["size_bytes"])
        if key not in self.documents:
            value=self.io.read_json(local,max_bytes=local["size_bytes"])
            if not isinstance(value,dict): raise ValueError("Native metadata JSON must be an object")
            self.documents[key]=value
            self.metadata_verified[key]=dict(logical_descriptor=logical,storage_descriptor=storage,
                                              read_descriptor=local,verified=True)
        return self.documents[key]


def prepare_metadata_artifacts(arguments_descriptor,output_root,*,native_module=None,io=None,
                               limits_descriptor):
    """Authorized metadata action only; no context setup/array/staging/admission."""
    from darsia.fluidflower.production_io import NasIO
    io=io or NasIO()
    arguments_descriptor=_descriptor(arguments_descriptor)
    io.verify(arguments_descriptor)
    arguments=io.read_json(arguments_descriptor,max_bytes=arguments_descriptor["size_bytes"])
    manifest_descriptor=_descriptor(arguments["manifest"])
    io.verify(manifest_descriptor)
    manifest=io.read_json(manifest_descriptor,max_bytes=manifest_descriptor["size_bytes"])
    verified_stage=_descriptor(arguments["verified_stage"])
    io.verify(verified_stage)  # own metadata only, never rehash its payloads
    if native_module is None:
        from darsia.fluidflower import production_native as native_module
    output=Path(output_root).resolve()
    if output.exists(): raise FileExistsError("Fresh immutable metadata namespace required")
    old=native_module._configuration(arguments["configuration"])
    reader=MetadataReader(manifest,arguments["staged_inputs"],arguments.get("logical_aliases",{}),io=io)
    limits_descriptor=_descriptor(limits_descriptor);io.verify(limits_descriptor)
    output.mkdir(parents=True,exist_ok=False)
    old_path=output/"original_four_configuration.json"
    io.write_json(old_path,old);old_descriptor=io.asset(old_path);reader.register(old_descriptor)
    selection=build_selection(old["campaign"],reader,native_module=native_module,limits_descriptor=limits_descriptor)
    selection_path=output/"precision127_selection.json"
    io.write_json(selection_path,selection);selection_descriptor=io.asset(selection_path);reader.register(selection_descriptor)
    configuration={**old,"format":CONFIGURATION_FORMAT,"profile_selection":selection_descriptor,
                   "original_four_configuration":old_descriptor,"limits":limits_descriptor}
    configuration_path=output/"precision127_native_configuration.json"
    io.write_json(configuration_path,configuration)
    closure=enumerate_precision_closure(configuration,reader,native_module=native_module,size_resolver=reader.size)
    for row in closure["assets"]:
        logical,local,storage=reader._identity(row["descriptor"])
        row.update(staging_source=storage,existing_local_stage=local if storage!=local else None,
                   scientific_bytes_verified_by_this_preparation=False)
    closure_path=output/"precision127_flat_closure.json"
    io.write_json(closure_path,closure)
    preparation=dict(format="darsia-production-precision127-metadata-preparation-v1",complete=True,
        source_arguments=arguments_descriptor,prior_manifest=manifest_descriptor,prior_verified_stage=verified_stage,
        old_four_configuration=old_descriptor,selection=selection_descriptor,
        native_configuration=io.asset(configuration_path),flat_closure=io.asset(closure_path),
        run_counts=dict(COUNTS),case_count=127,metadata_json_verified=list(reader.metadata_verified.values()),
        worker_ownership_policy=closure["worker_ownership_policy"],
        source_provenance=arguments["source_provenance"],logical_aliases=arguments.get("logical_aliases",{}),
        source_qualification=arguments["source_qualification"],
        selected_calibration=arguments["selected_calibration"],calibration_test_choice_only=True,
        arrays_read=False,scientific_payload_bytes_verified=False,native_setup_called=False,
        live_admission_granted=False,scientific_eligibility=False,production_permission_granted=False)
    path=output/"precision127_metadata_preparation.json"
    io.write_json(path,preparation)
    return io.asset(path)

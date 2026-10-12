"""Metadata-only bridge from accepted selected phase receipts to the R3 runner.

Inspection never admits an unfinished phase campaign. Sealing writes a new
plan only after the real all-39/time-parity gate and all selected producer
receipts exist. It reads JSON/source metadata, never dense scientific pixels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from darsia.fluidflower import phase_ffac_common_mass_inputs as source
from darsia.fluidflower import phase_ffac_common_mass_round3 as engine
from darsia.fluidflower.tools import run_ffac_common_mass_round3_selected as runner
from darsia.fluidflower.tools import ffac_round3_selected_adapter as adapter

CONTRACT = "ffac-round3-selected-metadata-bridge-v1"
PHASE_COMPLETION_CONTRACT = "ffac-selected-p1-final3-completion-v1"
STRICT_REPLAY_CONTRACT = "ffac-selected-phase-campaign-v1"
ALL_RUNS = set(engine.FIT_RUNS) | set(engine.HELDOUT_RUNS) | set(engine.DESCRIPTIVE_RUNS)
METADATA_SUFFIXES = {".json", ".toml", ".csv", ".md", ".py"}


def _asset(path):
    """Only source code or textual metadata may be read by this tool."""
    path = Path(path)
    if path.suffix.casefold() not in METADATA_SUFFIXES:
        raise ValueError("Metadata bridge must not read a dense scientific asset")
    return source.asset(path)


def _json(descriptor, reader):
    if Path(descriptor["path"]).suffix.casefold() != ".json":
        raise ValueError("Expected a source-bound JSON metadata receipt")
    return reader.json(descriptor)


def _metadata_dependencies(value, dependencies, prefix, reader):
    """Bind nested textual receipts; dense arrays remain bound in their producers."""
    if isinstance(value, dict):
        if value.get("path") and value.get("sha256"):
            if Path(value["path"]).suffix.casefold() in METADATA_SUFFIXES:
                reader.path(value)
                dependencies[prefix] = value
            return
        for name, item in value.items():
            _metadata_dependencies(item, dependencies, prefix+"."+str(name), reader)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _metadata_dependencies(item, dependencies, prefix+"."+str(index), reader)


def source_archive(archive_descriptor, reader):
    archive = _json(archive_descriptor, reader)
    if archive.get("contract") != "ffac-round3-selected-loader-source-equivalence-v1" or archive.get("complete") is not True:
        raise ValueError("The completed NAS source-equivalence archive is required")
    expected = runner.code_inventory() | {"darsia.fluidflower.tools.ffac_round3_selected_adapter":source.asset(adapter.__file__)}
    rows = {row["module"]:row for row in archive["code"]}
    if set(rows) != set(expected):
        raise ValueError("The source archive does not contain the exact stable 13 modules")
    dependencies = {"loader_source_equivalence":archive_descriptor}
    for name, descriptor in expected.items():
        row = rows[name]
        copy = row["archived_source"]
        if row.get("exact_source_equivalent") is not True or copy["sha256"] != descriptor["sha256"]:
            raise ValueError("An imported source module differs from the immutable archive: "+name)
        reader.path(copy)
        dependencies["archived_code."+name] = copy
    for name in ("r1_freeze", "r2_freeze"):
        row = archive["original_freezes"][name]
        if row.get("bytes_unchanged") is not True or row["archived_source"]["sha256"] != row["original_source"]["sha256"]:
            raise ValueError("An original frozen calibration was changed while archiving")
        reader.path(row["archived_source"])
        dependencies["archived_"+name] = row["archived_source"]
    return rows["darsia.fluidflower.tools.ffac_round3_selected_adapter"]["archived_source"], dependencies


def input_dependencies(parts_descriptor, archive_descriptor, reader):
    parts = _json(parts_descriptor, reader)
    if parts["spec"]["sha256"] != runner.SPEC_SHA256 or parts["r2_freeze"]["sha256"] != engine.ROUND2_FREEZE_SHA256:
        raise ValueError("The saved input plan changes the approved R3 spec or B0")
    adapter_descriptor, dependencies = source_archive(archive_descriptor, reader)
    # Coordinates/depth/physical arrays are NOT loaded. Their exact descriptors
    # remain in saved_parts_plan and are verified by the actual scientific reader.
    _metadata_dependencies(parts, dependencies, "saved_parts", reader)
    inputs = _json(parts["r2_input_plan"], reader)
    _metadata_dependencies(inputs, dependencies, "r2_inputs", reader)
    base = _json(inputs["base_input_plan"], reader)
    for name in ("recipes", "atmosphere", "original_domain_reference"):
        descriptor = base[name]
        _json(descriptor, reader)
        dependencies["base_inputs."+name] = descriptor
    dependencies["base_inputs"] = inputs["base_input_plan"]
    for run in sorted(ALL_RUNS):
        descriptor = inputs["clearwater_transports"][run]
        transport = _json(descriptor, reader)
        if transport.get("run") != run or transport.get("contract") != "ffac-common-native-clearwater-transport-v1":
            raise ValueError("Clearwater transport belongs to a different run")
        clear = dict(path=transport["clearwater_receipt"], sha256=transport["clearwater_receipt_sha256"])
        receipt = _json(clear, reader)
        if receipt.get("run") != run or receipt.get("encoding") != "float32-camera-linear-unit-rgb":
            raise ValueError("Clearwater receipt has a different source or radiometry")
        dependencies["clearwater_transport."+run] = descriptor
        dependencies["clearwater_receipt."+run] = clear
    folds, parameters = {}, {}
    for run in engine.FIT_RUNS:
        descriptor = _asset(Path(parts["r2_job"])/"work"/"loro"/"N1"/run/"receipt.json")
        record = _json(descriptor, reader)
        parameters[run] = runner._original_fold_parameters(record, run)
        folds[run] = descriptor
    # R2's preserved input record names its original producer's code paths.
    # Bind their byte-identical NAS copies without rewriting the old record or
    # requiring a remote worker to create the old user's repository folders.
    archived = {value["sha256"]:value for name,value in dependencies.items() if name.startswith("archived_code.")}
    for name, descriptor in list(dependencies.items()):
        if Path(descriptor["path"]).suffix.casefold() == ".py" and descriptor["sha256"] in archived:
            dependencies[name] = archived[descriptor["sha256"]]
    return parts, adapter_descriptor, dependencies, folds, parameters


def inspect_selection(phase_plan_descriptor, candidate, scientific_root, reader):
    """Inspect accepted camera targets and available producer JSON only."""
    phase_plan = _json(phase_plan_descriptor, reader)
    if (set(phase_plan["runs"]) != ALL_RUNS or phase_plan.get("phase_campaign_binding",{}).get("bounded_pilot")
            or not phase_plan.get("pipeline_sha256")):
        raise ValueError("A real all-39 frozen selected phase plan is required")
    candidate, scientific_root = Path(candidate), Path(scientific_root)
    if phase_plan.get("scientific_source_root") and Path(phase_plan["scientific_source_root"]).resolve() != scientific_root.resolve():
        raise ValueError("Scientific root differs from the frozen phase plan")
    manifests, entries, missing, reuse, baseline = {}, [], [], [], {}
    for run in sorted(ALL_RUNS):
        state = phase_plan["runs"][run]
        descriptor = state["source_manifest"]
        manifest = _json(descriptor, reader)
        if manifest.get("run") != run or manifest.get("contract") != "ffac-selected-camera-images-v1":
            raise ValueError("Selected run manifest has a different identity")
        manifests[run] = descriptor
        baseline[run] = manifest["baseline"]["source_sha256"]
        inventory = _json(state["source_inventory"], reader)
        if inventory.get("run") != run:
            raise ValueError("Selected source inventory has a different run")
        by_sha = {row["source_sha256"]:row for row in inventory["sources"]}
        sources = {row["source_sha256"]:row for row in manifest["frame_sources"]}
        targets = [row for row in manifest["target_matches"] if row.get("selected") is True]
        counts = Counter(row["source_sha256"] for row in targets)
        reused = {key:value for key,value in counts.items() if value>1}
        if reused:
            reuse.append(dict(run=run,source_targets=reused))
        for target in sorted(targets,key=lambda row:float(row["target_hour"])):
            digest = target["source_sha256"]
            selected, acquisition = sources[digest], by_sha[digest]
            if (target["selection_index"] != selected["selection_index"] or acquisition.get("selected") is not True
                    or float(target["elapsed_h"]) != float(selected["elapsed_h"])
                    or float(acquisition["elapsed_h"]) != float(selected["elapsed_h"])):
                raise ValueError("Selected target does not bind the accepted acquisition/time")
            phase_path = candidate/"selected_phase_frames"/run/(digest+".json")
            science_path = scientific_root/"selected_sources"/run/digest/"receipt.json"
            absent = [str(path) for path in (phase_path,science_path) if not path.is_file()]
            if absent:
                missing.append(dict(run=run,target_hour=target["target_hour"],source_sha256=digest,paths=absent))
                continue
            phase_descriptor, science_descriptor = _asset(phase_path), _asset(science_path)
            phase, science = _json(phase_descriptor,reader), _json(science_descriptor,reader)
            if (phase.get("contract") != adapter.PHASE_CONTRACT or phase.get("complete") is not True
                    or phase.get("binding",{}).get("source") != acquisition
                    or phase["binding"].get("pipeline_sha256") != phase_plan["pipeline_sha256"]
                    or phase["run"] != run or phase["source_sha256"] != digest
                    or phase["elapsed_h"] != selected["elapsed_h"]
                    or phase["frame_index"] != acquisition["work_index"]):
                raise ValueError("Selected phase receipt changes source/time/pipeline")
            if (science.get("contract") != adapter.SOURCE_CONTRACT or science.get("complete") is not True
                    or science.get("binding",{}).get("source") != acquisition
                    or science["run"] != run or science["source_sha256"] != digest
                    or science["elapsed_h"] != selected["elapsed_h"]
                    or phase["full_extended_scientific"] != science["full_extended_scientific"]
                    or phase["baseline_extended_scientific"] != science["baseline_extended_scientific"]):
                raise ValueError("Selected phase and scientific producer receipts differ")
            if phase["baseline_extended_scientific"]["source_sha256"] != baseline[run]:
                raise ValueError("Selected source has a different pre-injection baseline")
            entries.append(dict(run=run,frame_index=int(acquisition["work_index"]),
                elapsed_h=float(acquisition["elapsed_h"]),selected_target_h=float(target["target_hour"]),
                source_binding=acquisition,receipt=phase_descriptor,scientific_receipt=science_descriptor,
                run_manifest=descriptor))
    anchor_missing = [run for run in ALL_RUNS if not any(entry["run"]==run and entry["selected_target_h"]==2.25 for entry in entries)]
    return dict(phase_plan=phase_plan,run_manifests=manifests,entries=entries,
        missing=missing,reused_acquisitions=reuse,missing_anchors=sorted(anchor_missing),
        unique_acquisitions=len({(row["run"],row["source_binding"]["source_sha256"]) for row in entries}),
        accepted=False,dense_scientific_arrays_read=False,heldout_pixel_computation=False)


def verify_completion_gate(gate, phase_plan_descriptor, inspection, reader):
    """Verify the producer's real two seals and area report, without reading pixels."""
    pipeline = inspection["phase_plan"]["pipeline_sha256"]
    if (gate.get("contract") != PHASE_COMPLETION_CONTRACT or gate.get("complete") is not True
            or gate.get("phase_plan") != phase_plan_descriptor
            or gate.get("pipeline_sha256") != pipeline):
        raise ValueError("The final producer completion seal must bind the exact phase plan")
    strict_descriptor = gate["strict_replay_gate"]
    complete_descriptor = gate["selected_completion_seal"]
    report_descriptor = gate["area_time_report"]
    strict, complete, report = (_json(d, reader) for d in
        (strict_descriptor, complete_descriptor, report_descriptor))
    if (strict.get("contract") != STRICT_REPLAY_CONTRACT or strict.get("complete") is not True
            or strict.get("all_pass") is not True or strict.get("pipeline_sha256") != pipeline
            or strict.get("plan") != phase_plan_descriptor or strict.get("runs") != len(ALL_RUNS)
            or strict.get("missing") != [] or strict.get("failed") != []
            or strict.get("replayed_frames") != len(strict.get("rows",()))
            or gate.get("original_studio_replays") != strict.get("replayed_frames")
            or any(row.get("all_pass") is not True for row in strict.get("rows",()))):
        raise ValueError("The source-bound complete strict Studio replay has not passed")
    if (complete.get("contract") != PHASE_COMPLETION_CONTRACT or complete.get("complete") is not True
            or complete.get("pipeline_sha256") != pipeline or complete.get("plan") != phase_plan_descriptor
            or complete.get("missing") != [] or complete.get("run_manifests") != inspection["run_manifests"]
            or gate.get("selected_frames") != len(complete.get("entries",()))
            or complete.get("selected_expected") != len(complete.get("entries",()))
            or len(complete.get("entries",())) != inspection["unique_acquisitions"]
            or gate.get("baseline_frames") != len(ALL_RUNS)
            or len(complete.get("baselines",())) != len(ALL_RUNS)):
        raise ValueError("The selected/baseline completion inventory is incomplete or belongs to other sources")
    if (report.get("pipeline_sha256") != pipeline or report.get("all_studio_replays_pass") is not True
            or report.get("production_complete") is not True):
        raise ValueError("The completed source-bound area/time report is required")
    lookup = {(row["run"], row["source_sha256"]):row for row in complete["entries"]}
    if len(lookup) != len(complete["entries"]):
        raise ValueError("The completion inventory repeats a selected acquisition")
    for row in inspection["entries"]:
        key = (row["run"], row["source_binding"]["source_sha256"])
        sealed = lookup.get(key)
        if (sealed is None or any(sealed.get(name) != row[name] for name in
                ("frame_index", "elapsed_h", "source_binding", "receipt", "scientific_receipt", "run_manifest"))
                or sealed.get("target_hours") != [row["selected_target_h"]]):
            raise ValueError("The completion seal changes selected source/time/target or producer receipt")
    baseline_runs = set()
    expected_replays = 0
    for run, state in inspection["phase_plan"]["runs"].items():
        inventory = _json(state["source_inventory"], reader)
        expected_replays += sum(bool(row.get("studio_anchor")) for row in inventory["sources"])
    if strict["replayed_frames"] != expected_replays:
        raise ValueError("The strict replay seal omits original Studio acquisitions")
    for row in complete["baselines"]:
        run = row["run"]
        if run not in ALL_RUNS or run in baseline_runs:
            raise ValueError("The completion seal repeats or changes a run baseline")
        baseline_runs.add(run)
        manifest = _json(inspection["run_manifests"][run], reader)
        phase, science = (_json(row[name], reader) for name in ("receipt", "scientific_receipt"))
        binding = row["source_binding"]
        if (row.get("run_manifest") != inspection["run_manifests"][run]
                or row.get("source_sha256") != manifest["baseline"]["source_sha256"]
                or binding.get("baseline") is not True or row.get("target_hours") != []
                or any(record.get("complete") is not True or record.get("binding",{}).get("source") != binding
                    or record.get("run") != run or record.get("elapsed_h") != row["elapsed_h"]
                    or record.get("source_sha256") != row["source_sha256"] for record in (phase, science))
                or phase.get("contract") != adapter.PHASE_CONTRACT or science.get("contract") != adapter.SOURCE_CONTRACT
                or phase["binding"].get("pipeline_sha256") != pipeline
                or phase["full_extended_scientific"] != science["full_extended_scientific"]
                or phase["baseline_extended_scientific"] != science["baseline_extended_scientific"]):
            raise ValueError("A completed baseline differs from its manifest or scientific producer")
    dependencies = dict(selected_strict_replay=strict_descriptor,
        selected_completion_inventory=complete_descriptor, selected_area_time_report=report_descriptor)
    # This checks the textual producer receipts named by the strict replay. It
    # never loads maps or scientific arrays, and repeated assets use reader's cache.
    _metadata_dependencies(strict["rows"], dependencies, "strict_replay_producers", reader)
    _metadata_dependencies(complete["baselines"], dependencies, "baseline_producers", reader)
    return dependencies


def seal_plan(*, phase_plan_descriptor, accepted_gate_descriptor, parts_descriptor,
              archive_descriptor, candidate, scientific_root, output, reader=None):
    reader = reader or source.VerifiedAssetReader()
    # The accepted gate is checked before creating a destination or any plan.
    gate = _json(accepted_gate_descriptor,reader)
    if (gate.get("contract") != PHASE_COMPLETION_CONTRACT or gate.get("complete") is not True
            or gate.get("all_39_complete") is not True or gate.get("studio_time_parity_passed") is not True
            or set(gate.get("runs",())) != ALL_RUNS or set(gate.get("run_manifests",{})) != ALL_RUNS):
        raise ValueError("An actual accepted all-39 phase/time-parity gate is required before sealing")
    inspection = inspect_selection(phase_plan_descriptor,candidate,scientific_root,reader)
    if inspection["missing"] or inspection["missing_anchors"]:
        raise ValueError("Selected producers or explicit 2.25 h anchors are incomplete")
    if inspection["reused_acquisitions"]:
        raise ValueError("Current selected runner needs one unique acquisition per target; report reuse, never duplicate evidence")
    if gate["run_manifests"] != inspection["run_manifests"]:
        raise ValueError("Accepted gate and frozen phase plan use different camera manifests")
    if gate.get("pipeline_sha256") and gate["pipeline_sha256"] != inspection["phase_plan"]["pipeline_sha256"]:
        raise ValueError("Accepted gate belongs to another phase pipeline")
    completion_dependencies = verify_completion_gate(gate,phase_plan_descriptor,inspection,reader)
    parts, adapter_descriptor, dependencies, folds, parameters = input_dependencies(parts_descriptor,archive_descriptor,reader)
    dependencies.update(completion_dependencies)
    dependencies["selected_phase_plan"] = phase_plan_descriptor
    output = Path(output)
    output.mkdir(exist_ok=False,parents=True)
    # The operational producer is archived too; no current user's repository
    # path is required by a future remote worker or the calibration freeze.
    producer_bytes = Path(__file__).read_bytes()
    with (output/"metadata_bridge.py").open("xb") as stream:
        stream.write(producer_bytes)
    dependencies["metadata_bridge"] = _asset(output/"metadata_bridge.py")
    index = dict(contract=runner.INDEX_CONTRACT,complete=True,entries=inspection["entries"],
        all_sources_are_selected=True,historical_source_substitution=False,
        unique_acquisitions=inspection["unique_acquisitions"],target_count=len(inspection["entries"]),
        repeated_acquisitions=inspection["reused_acquisitions"],phase_plan=phase_plan_descriptor,
        dense_scientific_arrays_read=False)
    source.write_new_json(output/"selected_metadata_index.json",index)
    plan = dict(contract=runner.PLAN_CONTRACT,spec=parts["spec"],spec_sha256=runner.SPEC_SHA256,
        calibration_runs=list(engine.FIT_RUNS),heldout_runs=list(engine.HELDOUT_RUNS),descriptive_runs=list(engine.DESCRIPTIVE_RUNS),
        saved_parts_plan=parts_descriptor,selected_phase_gate=accepted_gate_descriptor,
        selected_metadata_index=_asset(output/"selected_metadata_index.json"),
        selected_adapter=dict(contract=runner.ADAPTER_CONTRACT,source=adapter_descriptor,function="load_selected"),
        additional_dependencies=dependencies,round2_loro_fits=folds,
        map_mass_rescaling=False,overflow_in_objective=False,
        production_automatically_allowed=False,selected_phase_candidate=str(candidate),scientific_source_root=str(scientific_root))
    source.write_new_json(output/"selected_input_plan.json",plan)
    receipt = dict(contract=CONTRACT,complete=True,accepted=True,
        created_utc=datetime.now(timezone.utc).isoformat(),plan=_asset(output/"selected_input_plan.json"),
        source_index=plan["selected_metadata_index"],accepted_phase_gate=accepted_gate_descriptor,
        phase_plan=phase_plan_descriptor,run_count=39,selected_target_count=len(index["entries"]),
        distinct_source_count=index["unique_acquisitions"],explicit_2_25_anchor_each_run=True,
        original_round2_loro_parameters=parameters,round2_loro_fits=folds,
        source_equivalence_archive=archive_descriptor,metadata_only=True,dense_scientific_arrays_read=False,
        heldout_pixel_computation=False,fits_started=False,original_assets_changed=False,
        metadata_bridge_archived_source=dependencies["metadata_bridge"],
        metadata_bridge_source_sha256=hashlib.sha256(producer_bytes).hexdigest(),
        historical_code_path_equivalence="Preserved R2 local implementation descriptors bind their exact archived NAS source SHA; original records are not rewritten.")
    source.write_new_json(output/"metadata_bridge_receipt.json",receipt)
    return _asset(output/"metadata_bridge_receipt.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-plan",required=True)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--scientific-root",required=True)
    parser.add_argument("--inspect-output",help="Optional NEW unaccepted metadata diagnostic JSON")
    parser.add_argument("--accepted-gate")
    parser.add_argument("--saved-parts-plan")
    parser.add_argument("--source-archive")
    parser.add_argument("--seal-output",help="NEW folder; requires the real accepted all-39 gate")
    args=parser.parse_args(argv)
    phase=_asset(args.phase_plan)
    if args.seal_output:
        if not all((args.accepted_gate,args.saved_parts_plan,args.source_archive)):
            raise ValueError("Sealing requires actual accepted gate, saved source inputs and immutable archive")
        result=seal_plan(phase_plan_descriptor=phase,accepted_gate_descriptor=_asset(args.accepted_gate),
            parts_descriptor=_asset(args.saved_parts_plan),archive_descriptor=_asset(args.source_archive),
            candidate=args.candidate,scientific_root=args.scientific_root,output=args.seal_output)
    else:
        result=inspect_selection(phase,args.candidate,args.scientific_root,source.VerifiedAssetReader())
        report={name:value for name,value in result.items() if name not in ("phase_plan","entries")}
        report.update(contract=CONTRACT,complete=False,accepted=False,available_targets=len(result["entries"]),
            pending_all39_gate=True,phase_plan=phase,created_utc=datetime.now(timezone.utc).isoformat())
        if args.inspect_output:
            source.write_new_json(args.inspect_output,report)
            result=_asset(args.inspect_output)
        else:
            result=report
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    main()

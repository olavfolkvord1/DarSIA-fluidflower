"""Honest metadata seal for complete Policy A review, never mass admission.

Selected camera acquisitions and original Studio probes remain distinct. The
coarse Policy A result is not represented as the superseded strict replay.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime,timezone

from . import phase_ffac_common_mass_inputs as assets
from .phase_ffac_selected_policy_a_baseline_qa import RUNS, SPEC_SHA256
from .tools import run_ffac_selected_policy_a_baseline_inventory as producer

CONTRACT='ffac-selected-policy-a-review-completion-v1'


class MetadataReader:
    """Cache parsed small receipts with the reader's exact session SHA checks."""
    def __init__(self,reader=None):
        self.reader=reader or assets.VerifiedAssetReader()
        self.documents={}
    def path(self,item):
        return self.reader.path(item)
    def record(self,item):
        path=self.path(item);stamp=path.stat()
        key=(str(path.resolve()),item['sha256'],stamp.st_size,stamp.st_mtime_ns)
        if key not in self.documents:
            self.documents[key]=self.reader.json(item)
        return self.documents[key]


def _same_coarse_result(saved, produced):
    """Compare the numerical gate, excluding only producer provenance."""
    if saved == produced:
        return True
    provenance = {'parent_receipt','run','studio_frame_index','source_sha256','elapsed_h'}
    return saved == {key:value for key,value in produced.items() if key not in provenance}


def inspect_completed_review(plan_descriptor,root,completion_descriptor,report_descriptor,*,reader=None):
    reader=MetadataReader(reader)
    plan=reader.record(plan_descriptor)
    producer.verify_code(plan)
    runs=tuple(sorted(plan['runs']))
    if set(runs)!=set(RUNS):
        raise ValueError('Policy A review requires the exact selected cohort')
    identity=producer.require_both_identity(plan['identity_gate'],reader,expected_runs=runs)
    if identity['identity_plan']!=plan['phase_campaign_binding']['identity_plan']:
        raise ValueError('Review identity belongs to another source/code producer')
    if producer.shared_stop(root,plan).exists():
        raise ValueError('A stopped Policy A campaign cannot publish Studio readiness')
    completed=reader.record(completion_descriptor)
    report=reader.record(report_descriptor)
    expected=plan['phase_campaign_binding']['expected_counts']
    if (completed.get('contract')!='ffac-selected-policy-a-required-review-completion-v1'
            or any(completed.get(key) is not True for key in ('complete','all_39_phase_complete','coarse_area_gate_passed'))
            or completed.get('plan')!=plan_descriptor or completed.get('identity_gate')!=plan['identity_gate']
            or completed.get('pipeline_sha256')!=plan['pipeline_sha256']
            or completed.get('counts')!=expected or completed.get('missing')
            or set(completed.get('runs',{}))!=set(runs)
            or completed.get('round3_allowed') is not False or completed.get('mass_admission') is not False
            or completed.get('claude_scorer_gate_pending') is not True or completed.get('scorer_executed') is not False
            or completed.get('bottom_mass_allowed') is not False or completed.get('default_changed') is not False
            or completed.get('old_strict_gate_decides_acceptance') is not False):
        raise ValueError('All selected maps and every coarse Studio anchor must actually be complete')
    if (report.get('contract')!='ffac-selected-policy-a-required-saved-report-v1'
            or report.get('complete') is not True or report.get('all_pass') is not True
            or report.get('plan')!=plan_descriptor or report.get('identity_gate')!=plan['identity_gate']
            or report.get('pipeline_sha256')!=plan['pipeline_sha256']
            or report.get('spec_sha256')!=SPEC_SHA256 or report.get('studio_frames')!=expected['studio_anchors']
            or report.get('selected_timeline_complete') is not True or report.get('missing')
            or report.get('round3_allowed') is not False or report.get('claude_scorer_gate_pending') is not True
            or any(report.get(key) is not False for key in ('parameters_changed','maps_changed','annotations_read','mass_computed','bottom_mass_allowed','default_changed'))):
        raise ValueError('Studio publication needs the actual full source-bound coarse/spatial report')
    actual={(row['run'],row['source_sha256']):row for row in completed['entries']}
    if len(actual)!=len(completed['entries']):
        raise ValueError('Review completion repeats an acquisition')
    report_rows={}
    for item in report['frame_receipts']:
        row=reader.record(item);key=(row['run'],row['original_studio_frame_index'])
        if key in report_rows or row.get('complete') is not True or row.get('comparison',{}).get('all_pass') is not True:
            raise ValueError('Coarse report has a duplicate or failed original Studio frame')
        report_rows[key]=row
    selected,baselines,probes=[],[],[]
    count=dict(selected_frames=0,baseline_frames=0,studio_anchors=0)
    declared=set();anchors=set();manifests={}
    scientific_root=Path(plan['scientific_source_root'])
    for run in runs:
        state=plan['runs'][run]
        inventory=reader.record(state['source_inventory'])
        manifests[run]=state['source_manifest']
        manifest=reader.record(manifests[run])
        targets={}
        for match in manifest['target_matches']:
            if match.get('selected'):
                targets.setdefault((match['selection_index'],match['source_sha256']),[]).append(match['target_hour'])
        for source in producer.select_rows(inventory):
            key=(run,source['source_sha256']);declared.add(key)
            if key not in actual:
                raise ValueError('Completed review omits a selected acquisition or Studio probe')
            entry=actual[key]
            phase=reader.record(entry['phase_receipt'])
            if (phase.get('complete') is not True or phase.get('binding',{}).get('source')!=source
                    or phase['binding'].get('pipeline_sha256')!=plan['pipeline_sha256']
                    or phase.get('run')!=run or phase.get('source_sha256')!=source['source_sha256']
                    or phase.get('frame_index')!=source['work_index'] or phase.get('elapsed_h')!=source['elapsed_h']
                    or entry.get('work_index')!=source['work_index']
                    or entry.get('elapsed_h')!=source['elapsed_h'] or entry.get('labels_map')!=phase['labels_map']):
                raise ValueError('Review entry changed its source, camera time, work index or producer')
            producer.validate_resume(phase,source,plan)
            scientific_descriptor=assets.asset(scientific_root/'selected_sources'/run/source['source_sha256']/'receipt.json')
            science=producer.verify_ready_source(scientific_descriptor,source,plan,reader)
            if (phase['full_extended_scientific']!=science['full_extended_scientific']
                    or phase['baseline_extended_scientific']!=science['baseline_extended_scientific']):
                raise ValueError('Review phase/source wrapper uses a different current or reference role')
            if source.get('studio_anchor'):
                anchor=(run,source['studio_frame_index']);anchors.add(anchor)
                comparison=report_rows.get(anchor)
                if (comparison is None or comparison.get('source_sha256')!=source['source_sha256']
                        or comparison.get('phase_receipt')!=entry['phase_receipt']
                        or comparison.get('current_labels')!=phase['labels_map']
                        or not _same_coarse_result(comparison.get('comparison'),phase['qa']['policy_a_coarse_studio'])):
                    raise ValueError('Saved coarse report belongs to another anchor prediction or result')
                count['studio_anchors']+=1
            row=dict(run=run,source_sha256=source['source_sha256'],elapsed_h=source['elapsed_h'],
                frame_index=source['work_index'],source_binding=source,
                receipt=entry['phase_receipt'],scientific_receipt=scientific_descriptor,
                run_manifest=manifests[run],selection_index=source.get('selection_index'),
                target_hours=targets.get((source.get('selection_index'),source['source_sha256']),[]))
            if source.get('baseline'):
                baselines.append(row);count['baseline_frames']+=1
            elif source.get('selected'):
                selected.append(row);count['selected_frames']+=1
            else:
                probes.append(row)
    if (set(actual)!=declared or set(report_rows)!=anchors or count!=expected
            or len(baselines)!=len(runs)):
        raise ValueError('Policy A selected/baseline/Studio coverage is incomplete')
    return dict(contract=CONTRACT,complete=True,all_runs_complete=True,all_39_complete=len(runs)==39,
        created_utc=datetime.now(timezone.utc).isoformat(),spec_sha256=SPEC_SHA256,
        phase_plan=plan_descriptor,pipeline_sha256=plan['pipeline_sha256'],identity_gate=plan['identity_gate'],
        source_completion=completion_descriptor,coarse_spatial_report=report_descriptor,
        runs=list(runs),run_manifests=manifests,selected_frames=len(selected),baseline_frames=len(baselines),
        studio_anchor_frames=len(anchors),timeline_frames=len(selected)+len(baselines),
        old_only_studio_probes=len(probes),entries=selected,baselines=baselines,studio_probes=probes,
        coarse_area_gate_passed=True,claude_scorer_gate_pending=True,scorer_executed=False,
        round3_allowed=False,mass_admission=False,bottom_mass_allowed=False,default_changed=False,
        old_strict_gate_decides_acceptance=False,legacy_strict_gate_claimed=False)


def verify_gate(gate_descriptor,plan_descriptor,root,*,reader=None):
    raw=reader or assets.VerifiedAssetReader()
    gate=raw.json(gate_descriptor)
    expected=inspect_completed_review(plan_descriptor,root,gate['source_completion'],gate['coarse_spatial_report'],reader=raw)
    # Creation time belongs to the producer receipt; every evidence field must
    # still be exactly reproducible from actual immutable completion artifacts.
    expected['created_utc']=gate['created_utc']
    if gate!=expected:
        raise ValueError('Policy A review completion seal changed its source evidence or claims')
    return gate


__all__=['CONTRACT','inspect_completed_review','verify_gate']

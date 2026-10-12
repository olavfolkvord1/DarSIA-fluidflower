"""Freeze a new result-chain revision while preserving fitted evidence and weights.

Preparation is not publication or retraining. The original causal provider is
retained as an evidence adapter for the frozen completion input contract. Only
the new decision owns the final output. All old bundles remain reproducible.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import copy
import shutil

from darsia.fluidflower import phase_decision as core, phase_decision_runtime as runtime, phase_runtime
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.materialize_phase_iteration import relocate
from darsia.fluidflower.tools.publish_fitted_phase_candidate import fitted_files, verify_fitted
from darsia.fluidflower.tools.phase_mass_overnight_data import read, write


def verify(root):
    root=Path(root).resolve();record=read(root/'revision_provenance.json')
    parent=Path(record['parent']);profile=phase_runtime.load_physics_profile(root)
    inherited,spec=runtime.split_profile(profile)
    expected=phase_runtime.load_physics_profile(parent)
    # attach_profile has already verified the local checkpoint path and bytes.
    # This one resolved path is expected to differ after a portable bundle copy.
    expected['aqueous_context_completion']['_checkpoint_path']=inherited['aqueous_context_completion']['_checkpoint_path']
    if inherited!=expected:
        raise ValueError('Frozen evidence provider changed')
    if phase_runtime.physics_runtime_identity(profile)!=record['physics_runtime']:
        raise ValueError('Prepared decision runtime changed')
    if list(_model_identity(parent))!=record['parent_identity']:
        raise ValueError('Fitted parent identity changed')
    for relative,digest in record['parent_files'].items():
        if sha256_file(parent/relative)!=digest:raise ValueError(f'Protected parent changed: {relative}')
    for relative,digest in record['frozen_copy_files'].items():
        if sha256_file(root/relative)!=digest:raise ValueError(f'Frozen evidence changed: {relative}')
    summary=read(root/'summary.json');request=read(root/'training_request.json')
    if summary['models']!=read(parent/'summary.json')['models'] or request['training_performed'] is not False:
        raise ValueError('Decision revision must not claim a new fit')
    if summary['model_id']!=root.name or request['model_id']!=root.name:
        raise ValueError('Decision revision identity mismatch')
    return dict(model_id=root.name,model_sha256=_model_identity(root)[1],physics_runtime=record['physics_runtime'],
                training_performed=False,parent=str(parent),settings=spec['settings'])


def prepare(parent,output,*,settings=None,local_age=False):
    from darsia.fluidflower import phase_decision_versions
    from darsia.fluidflower import phase_decision_age_runtime
    runtime=phase_decision_versions.implementation(dict(format=phase_decision_age_runtime.VERSION if local_age else 'unified-phase-decision-runtime-v1'))
    core=runtime.core
    parent,output=Path(parent).resolve(),Path(output).resolve()
    if parent==output or parent.is_relative_to(output) or output.is_relative_to(parent):
        raise ValueError('Use a separate new revision folder')
    if output.exists():raise FileExistsError(output)
    if not output.name.endswith('-decision'):raise ValueError('New revision ID must end in -decision')
    # Existing nonconvergence is recorded, never relabelled as a new successful fit.
    fitted=verify_fitted(parent,allow_nonconverged=True)
    inherited=phase_runtime.load_physics_profile(parent)
    if runtime.PROFILE_KEY in inherited or 'local_aqueous_support' in inherited:
        raise ValueError('Select the frozen B parent; do not stack result revisions')
    spec=dict(format=runtime.VERSION,settings=asdict(core.Settings(**(settings or {}))))
    profile={**inherited,runtime.PROFILE_KEY:spec}
    identity=phase_runtime.physics_runtime_identity(profile)
    files=fitted_files(parent);parent_hashes={p.relative_to(parent).as_posix():sha256_file(p) for p in files}
    output.mkdir(parents=True);provenance=output/'parent_provenance';provenance.mkdir()
    frozen={}
    for path in files:
        relative=path.relative_to(parent)
        if relative.as_posix() in {'summary.json','training_request.json','physics_inheritance.json'}:
            shutil.copy2(path,provenance/path.name);continue
        target=(provenance/relative if path.name.endswith('_fit_verification.json') else output/relative)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        digest=parent_hashes[relative.as_posix()]
        if sha256_file(target)!=digest:raise OSError(f'Frozen copy mismatch: {relative}')
        frozen[target.relative_to(output).as_posix()]=digest
    request=relocate(read(parent/'training_request.json'),parent,output)
    request.update(model_id=output.name,created_utc=utc_now(),output_root=str(output),
        request_path=str(output/'training_request.json'),artifact_kind='phase-support-revision',
        training_performed=False,optical_training_performed=False,inherited_training_request=str(parent/'training_request.json'))
    request['settings'].update(iteration='Unified phase decision v2 · Local front age' if local_age else 'Unified phase decision v1',parent_model=str(parent),
        phase_support_extension=core.VERSION,model_initialization='Frozen optical and completion weights; one final decision')
    summary=copy.deepcopy(read(parent/'summary.json'))
    summary.update(model_id=output.name,campaign_id=output.name,created_utc=utc_now(),
        artifact_kind='phase-support-revision',weights_retrained=False,optical_training_performed=False,
        inherited_optical_model_id=parent.name,evaluation_parent=dict(model_root=str(parent),model_identity=_model_identity(parent)[1]),
        evaluation_scope='Same fitted evidence; unified final phase decision. Development checks are not independent validation.',
        phase_support_extension=core.VERSION,human_validation=None,deployment=None)
    write(output/runtime.PROFILE_FILE,spec);write(output/'training_request.json',request)
    write(output/'physics_inheritance.json',dict(parent_model=str(parent),
        parent_physics_runtime=phase_runtime.physics_runtime_identity(inherited),physics_runtime=identity,
        inherited_settings_unchanged=True,calibration_unchanged=True,physical_contexts_unchanged=True,
        settings_unchanged=False,change='One final decision from frozen causal, completion and reference evidence'))
    snapshot=output/'decision_code_snapshot';snapshot.mkdir()
    for source in (Path(core.__file__),Path(runtime.__file__),Path(phase_runtime.__file__),Path(__file__)):
        shutil.copy2(source,snapshot/source.name)
    if local_age:
        for source in (Path(core.base.__file__),Path(runtime.legacy.__file__),Path(phase_decision_versions.__file__)):
            shutil.copy2(source,snapshot/source.name)
    write(output/'revision_provenance.json',dict(format=core.VERSION,created_utc=utc_now(),parent=str(parent),
        parent_identity=list(_model_identity(parent)),parent_model_id=parent.name,
        parent_model_sha256=_model_identity(parent)[1],physics_runtime=identity,parent_files=parent_hashes,
        frozen_copy_files=frozen,inherited_fit=fitted,training_performed=False,optical_training_performed=False,
        source_reviews_changed=False,independent_validation=False,
        migration_limit='Existing causal map/history is an evidence adapter for frozen B inputs; gas evidence is inherited.',
        delivery_status='prepared; native sequences, exact references and result-chain checks required'))
    write(output/'summary.json',summary)
    return verify(output)


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();print(json.dumps(prepare(args.parent,args.output)),flush=True)

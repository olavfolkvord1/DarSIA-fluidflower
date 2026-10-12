"""Reuse checked causal evidence for an exact reference with a new final decision.

The original reference was evaluated on a disposable, strictly earlier history.
Only that immutable evidence is reused. The current core decides the new labels.
Missing evidence returns None so the normal exact live replay remains available.
"""
from pathlib import Path
from darsia.fluidflower.tools.phase_mass_overnight_data import read,write,sha


def derive(plan,folder):
    from darsia.fluidflower.tools import phase_reference_maps as refs,calibration_pairwise_review as review
    from darsia.fluidflower.tools.train_phase_model_further import native_source
    from darsia.fluidflower.tools.prepare_phase_decision_revision import verify
    from darsia.fluidflower.tools.materialize_phase_decision_revision import cached_decision,rgb,fields,exposure_history
    from darsia.fluidflower import phase_runtime,phase_decision as core
    from darsia.fluidflower.phase_validation import _write_phase_maps
    import numpy as np
    root=Path(plan['model_root'])
    if not (root/'phase_decision.json').is_file():return None
    record=root/'revision_provenance.json'
    if not record.is_file():return None  # newly fitted bundles use normal live inference
    parent=Path(read(record)['parent'])
    if not (parent/'validation/validation_binding.json').is_file():return None  # portable bundle on another host
    checked=verify(root);parent=Path(checked['parent']);local_parent=parent
    # The same frozen fit may have been reviewed from its published NAS copy.
    # Resolve that already prepared parent by model/runtime and image identity,
    # never by a similar folder name or by an unverified ready flag.
    from darsia.fluidflower.phase_validation import _model_identity
    expected_identity=_model_identity(parent)
    expected_runtime=phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(parent))
    for cache_root in refs._receipt_cache_roots():
        for path in cache_root.glob('*/receipt.json'):
            candidate_receipt=read(path)
            if (candidate_receipt.get('model_id'),candidate_receipt.get('model_sha256'))!=expected_identity:
                continue
            if candidate_receipt.get('run')!=plan['run'] or candidate_receipt.get('reference',{}).get('photo',{}).get('sha256')!=plan['reference']['photo']['sha256']:
                continue
            published=Path(candidate_receipt['model_root'])
            if not (published/'validation/validation_binding.json').is_file():continue
            if _model_identity(published)!=expected_identity or phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(published))!=expected_runtime:
                raise ValueError('Published parent does not match the frozen numerical evidence')
            parent=published
            break
        if parent!=local_parent:break
    validation=parent/'validation';job=next(j for j in read(validation/'postflight_manifest.json')['jobs'] if j['run']==plan['run'])
    source,native_path,_=native_source(job);candidate=review._candidate_from_job(validation,job)
    reference=None if plan['reference']['id']=='baseline' else plan['reference']
    original_plan=refs.build_plan(reference,source,plan['run'],candidate)
    for key in ('native_manifest','baseline'):
        if original_plan[key]['sha256']!=plan[key]['sha256']:raise ValueError('Inherited reference evidence uses another registration or baseline')
    if original_plan['reference']['photo']['sha256']!=plan['reference']['photo']['sha256']:
        raise ValueError('Inherited reference evidence uses another photograph')
    if [(r['elapsed_h'],r['photo']['sha256']) for r in original_plan['replay']]!=[(r['elapsed_h'],r['photo']['sha256']) for r in plan['replay']]:
        raise ValueError('Inherited reference evidence uses another causal timeline')
    if refs._cached(original_plan,'all') is None:return None
    parent_receipt=refs.cache_root()/original_plan['key']/'receipt.json';receipt=read(parent_receipt)
    original_item=receipt['layers']['all'];original=Path(original_item.get('source',original_item)['path'])
    if not original.is_file():return None
    if receipt['mode']=='exact-reviewed-image-reuse':
        index=original_plan['exact_reviewed_frame_index']
        diagnostics=parent/'validation/presence_diagnostics'/plan['run']/f'frame_{index:03d}.npz'
        if not diagnostics.is_file():
            diagnostics=local_parent/'validation/presence_diagnostics'/plan['run']/f'frame_{index:03d}.npz'
    elif receipt['mode']=='fresh-exact-reference':
        if receipt.get('canonical_history_unchanged') is not True:raise ValueError('Original reference changed canonical history')
        final=next((p for p in original.parents if p.name=='final'),None)
        if final is None:return None
        diagnostics=final.parent/'diagnostics'/plan['run']/'frame_000.npz'
    else:return None
    if not diagnostics.is_file():return None
    profile=phase_runtime.load_physics_profile(root)
    if phase_runtime.physics_runtime_identity(profile)!=plan['physics_runtime']:raise ValueError('Reference runtime changed')
    context,_=phase_runtime.load_run_context(root,profile,plan['dataset'],plan['run'],native_manifest_sha256=sha(native_path))
    from darsia.fluidflower.phase_decision_versions import implementation
    settings=implementation(profile['unified_phase_decision']).core.Settings(**profile['unified_phase_decision']['settings'])
    history=exposure_history(context,settings);history_assets=[]
    if history is not None:
        for frame in plan['replay']:
            if frame['elapsed_h']>=plan['reference']['phase_time_h']:
                raise ValueError('Reference exposure history contains the target or a future frame')
            path=local_parent/'validation/presence_diagnostics'/plan['run']/f"frame_{frame['frame_index']:03d}.npz"
            if not path.is_file():return None
            asset=refs._asset(path)
            with np.load(path) as z:observed=z['completion_before_co2']
            history.observe(observed,context.domain,frame['elapsed_h'],pre_injection=frame['elapsed_h']<0)
            history_assets.append(dict(elapsed_h=frame['elapsed_h'],diagnostics=asset))
    box=plan['content_box_xywh'];baseline=rgb(plan['baseline']['path'],box);current=rgb(plan['reference']['photo']['path'],box)
    evidence=dict(parent_receipt=refs._asset(parent_receipt),diagnostics=refs._asset(diagnostics),exposure_history=history_assets,
                  original_phase=refs._asset(original),canonical_history_reused_without_mutation=True)
    result=cached_decision(current,baseline,context,plan['reference']['phase_time_h'],diagnostics,original,box,
        settings,evidence,history=history)
    _write_phase_maps(folder/'final',plan['run'],plan['pipeline_sha256'],0,plan['canvas_wh'][::-1],box,fields(result.labels))
    generated={**plan['job'],'phase_map_output_dir':str(folder/'final'),'presence_diagnostics_dir':None,'raw_phase_map_output_dir':None}
    layers=refs._display_layers(refs._original_layers(generated,0))
    if 'raw' in receipt['layers']:layers['raw']=receipt['layers']['raw']
    # Verify the renderer does not change the shared numeric result.
    from darsia.fluidflower.tools.phase_mass_overnight_data import decode_labels
    if not np.array_equal(decode_labels(layers['all']['source']['path'],box),result.labels):
        raise ValueError('Exact reference rendering changed numerical phase labels')
    refs._check_assets(plan)
    if sha(diagnostics)!=evidence['diagnostics']['sha256']:raise ValueError('Reference evidence changed while computing')
    new={**plan,'mode':'unified-decision-from-verified-reference-evidence','layers':layers,
         'inherited_reference_evidence':evidence,'canonical_history_unchanged':True,
         'replay_validation':'Same immutable causal evidence; final shared decision recomputed; no fresh optical inference',
         'physical_report':result.report}
    write(folder/'receipt.json',new);return new

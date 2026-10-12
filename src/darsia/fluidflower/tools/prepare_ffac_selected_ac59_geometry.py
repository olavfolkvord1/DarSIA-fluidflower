"""Bind and check AC59's own archived shape rig; no frame inference."""
from pathlib import Path
import argparse,json,tomllib

from darsia.fluidflower.phase_ffac_selected_phase import asset,bound_json
from darsia.fluidflower.phase_ffac_selected_ac59_source_geometry import CONTRACT,AC59OwnPersistedSourceGeometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--project-run-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    plan_asset=asset(args.plan);plan=bound_json(plan_asset)
    native_asset=plan['runs']['ac59']['parent_native_manifest'];native=bound_json(native_asset)
    marker=native['v5_track1_photo_geometry']
    original_asset=dict(path=marker['original_source_manifest_path'],sha256=marker['original_source_manifest_sha256'])
    original=bound_json(original_asset)
    config_asset=asset(args.project_run_config)
    config=tomllib.loads(args.project_run_config.read_text(encoding='utf-8-sig'))
    rig=Path(config['rig']['path']);shapes=sorted(rig.glob('shape_correction_*.npz'))
    binding=dict(contract=CONTRACT,run='ac59',source_plan=plan_asset,parent_native_manifest=native_asset,
        original_source_manifest=original_asset,original_shape_configs=[asset(p) for p in original['correction_contract']['config_paths']],
        project_run_config=config_asset,own_rig_path=str(rig),shape_corrections=[asset(p) for p in shapes],
        historical_path_was_implicit=True,source_manifest_changed=False,borrowed_other_run_rig=False,
        new_photometry_fit=False,model_or_labels_used=False,
        recovery_adapter=asset(Path(__import__('darsia.fluidflower.phase_ffac_selected_ac59_source_geometry',fromlist=['x']).__file__)))
    target=args.output/'own_rig_binding.json'
    with target.open('x',encoding='utf-8') as f:json.dump(binding,f,indent=2)
    descriptor=asset(target)
    geometry=AC59OwnPersistedSourceGeometry(native_asset['path'],descriptor)
    result=dict(contract='ffac-selected-ac59-own-shape-parity-v1',complete=True,source_binding=descriptor,
        baseline_geometry_parity=geometry.baseline_geometry_parity,
        original_baseline_array_changed=False,new_source_pixels_prepared=False,
        phase_inference=False,labels_read=False,old_files_overwritten=False)
    with (args.output/'parity.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

"""Bind each BM run to its own wash, registered to its own baseline.

Uses the C1 whole-rig registration method and published BM physical properties.
Wash photographs are optical references, never known-concentration standards.
"""
from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.series_paths import local_cache_root, series_path
import argparse
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from darsia.fluidflower.tools.phase_mass_overnight_data import read,write,sha
from darsia.fluidflower.tools.prepare_bm_phase_target import ORIGINAL

RAW=configured_tool_path('ffbm', 'raw_inputs')
WASH={
 'c1':'FF_BM_cycle_1/C1_images/C1_preparations/211027_c0.1_clean_flush/DSC06341.JPG',
 'c2':'FF_BM_cycle_2/C2_images/C2_preparations/211129_170000_c2_clean_fush_15min/DSC04600.JPG',
 'c3':'FF_BM_cycle_3/C3_images/211209_112100/DSC01118.JPG',
 'c4':'FF_BM_cycle_4/C4_images/211219_164500_c4_flush/DSC06867.JPG',
 'c5':'FF_BM_cycle_5/C5_images/211230_130000_cleaning_5min/DSC12799.JPG',
}


def register(baseline,clear):
    if baseline.shape!=clear.shape: raise ValueError('Wash and baseline use different raw geometry')
    width=2000;scale=width/baseline.shape[1];size=(width,round(baseline.shape[0]*scale))
    images=[cv2.resize(a,size,interpolation=cv2.INTER_AREA) for a in (baseline,clear)]
    clahe=cv2.createCLAHE(clipLimit=3.,tileGridSize=(16,12))
    gray=[clahe.apply(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)) for a in images]
    mask=np.zeros(gray[0].shape,np.uint8);mask[int(.20*size[1]):int(.95*size[1]),int(.04*size[0]):int(.96*size[0])]=255
    cv2.setRNGSeed(20260914)
    detector=cv2.SIFT_create(nfeatures=20000,contrastThreshold=.01)
    kb,db=detector.detectAndCompute(gray[0],mask);kc,dc=detector.detectAndCompute(gray[1],mask)
    good=[a for a,b in cv2.BFMatcher().knnMatch(dc,db,k=2) if a.distance<.7*b.distance]
    if len(good)<40:raise ValueError('Too few independent whole-rig wash features')
    src=np.array([kc[m.queryIdx].pt for m in good],np.float32)
    dst=np.array([kb[m.trainIdx].pt for m in good],np.float32);train=np.arange(len(src))%4!=0
    matrix,inlier=cv2.estimateAffinePartial2D(src[train],dst[train],method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=5000,confidence=.999)
    if matrix is None:raise ValueError('Whole-rig wash transform not supported')
    residual=np.linalg.norm(np.column_stack([src,np.ones(len(src))])@matrix.T-dst,axis=1)[~train]
    report=dict(method='SIFT CLAHE, ratio0.7, similarity RANSAC; every fourth match held out',
        matched_features=len(good),held_median_residual_native_pixels=float(np.median(residual)/scale),
        held_p90_residual_native_pixels=float(np.percentile(residual,90)/scale),held_inlier_fraction=float(np.mean(residual<=2.)))
    if report['held_inlier_fraction']<.85 or report['held_median_residual_native_pixels']>3.:
        raise ValueError('Whole-rig wash registration fails the C1 support guard: '+str(report))
    matrix[:,2]/=scale;report['native_transform']=matrix.tolist()
    return matrix,report


def prepare(native_path,output):
    import darsia
    from darsia.fluidflower.image_correction import _reference_colorchecker
    from darsia.fluidflower.phase_training.radiometry import core
    from darsia.fluidflower.tools import prepare_bm_injection_references as bm
    from threadpoolctl import threadpool_limits
    native_path,output=Path(native_path),Path(output);output.mkdir(parents=True,exist_ok=True)
    native=read(native_path);run=native['run'];path=RAW/WASH[run]
    context=bm._context(native)
    with Image.open(io.BytesIO(context['baseline_bytes'])) as im:baseline=np.asarray(im.convert('RGB'))
    data,source_sha,_=bm._source_bytes(path)
    with Image.open(io.BytesIO(data)) as im:clear=np.asarray(im.convert('RGB'))
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        matrix,probe=register(baseline,clear)
        write(output/'wash_registration.json',probe)
        registered=cv2.warpAffine(clear,matrix,(clear.shape[1],clear.shape[0])).astype(np.float32)/255
        cv2.setRNGSeed(0)
        swatches,roi=_reference_colorchecker(registered,corner='upper_left',roi_rc=np.asarray(context['roi_rc']))
        corrected,receipt=core().prepare_image(registered,context['profile'],observed_swatches=swatches,
            series_id='ffbm',run_id=run,frame_id='registered_wash_reference',source={'path':str(path),'sha256':source_sha},
            sampling={'roi_rc':roi.tolist(),'basis':'Whole-rig supported similarity registration then frozen baseline chart ROI'})
        config=context['config'];config['curvature']['use_cache']=False
        curvature=darsia.CurvatureCorrection(config=config['curvature'])
        curvature.cache=np.load(io.BytesIO(context['grid_bytes']),allow_pickle=True).item()
        corrected=curvature.correct_array(corrected)
    target=output/'clearwater_registered.jpg'
    Image.fromarray(np.rint(np.clip(corrected,0,1)*255).astype(np.uint8)).save(target,quality=98,subsampling=0)
    cfg=read(ORIGINAL/'phase_segmentation/config.json')
    geometry=dict(width_m=cfg['physical_asset']['dimensions']['width'],height_m=cfg['physical_asset']['dimensions']['height'],
        porosity=cfg['physical_asset']['parameters']['porosity'],depth_source=str(ORIGINAL/'cache/depth.npy'),
        labels_source=str(ORIGINAL/'cache/labels_coarse.npy'),property_policy='Published shared benchmark sand pack; measured depth and uniform porosity')
    pressure=ORIGINAL/'measurements/Florida_2021-11-24_2022-02-02_1669032742.xlsx'
    binding=dict(format='bm-studio-mass-reference-v1',run=run,native_manifest_sha256=sha(native_path),
        baseline_source_sha256=context['baseline_sha'],clearwater_path=str(target),clearwater_sha256=sha(target),
        clearwater_source_path=str(path),clearwater_source_sha256=source_sha,registration=probe,
        pressure_source=str(pressure),pressure_sha256=sha(pressure),geometry=geometry,solubility_g_l=1.8,
        solubility_source='Published BM mass_analysis.py: dissolution_limit=1.8 kg/m3',radiometry=receipt,
        correction_contract=native['correction_contract'],
        optical_reference_limit='Selected wash, not a known pH or concentration. Residual staining can bias optical normalization.',
        source_files={str(p):sha(p) for p in [path,target,pressure,Path(geometry['depth_source']),Path(geometry['labels_source'])]})
    write(output/'reference.json',binding)
    return output/'reference.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();print(prepare(args.native,args.output))

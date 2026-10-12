"""Read-only final-round A–D tables, locked floor-box checks and figures.

Requires a complete new candidate. No inference, model fit, label update or
independent sealed-scorer execution is performed by this reporting tool.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage

from .. import phase_ffac_p1_final2 as rules
from ..phase_rig_display_prerender import pointer_path
from . import report_ffac_p1_final as saved

CONTRACT="ffac-p1-final2-final-round-report-v1"
ANNOTATIONS_SHA256="e54392b651d466949232c210a3ccd4fac9fe55e932147bffedf1167377a73c1f"
MASK_FIELDS=dict(A="below_sand_floor_masks",B="floor_connectivity_masks",C="floor_contact_fill_v2",D="gas_envelope_masks")
COLOURS=dict(A=(40,215,255),B=(255,190,20),C=(190,110,255),D=(255,70,165))


def csv_new(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def restoration_accounting(row,parent_row):
    """Keep stage operations distinct from net final-label changes."""
    qa=row['qa'];count=int(qa['unchanged_stage_restoration_pixels'])
    stage_sum=sum(float(value) for value in qa['step_area_cm2'].values())
    result=dict(unchanged_stage_restoration_pixels=count,unchanged_stage_restoration_cm2=0.,
        stage_operation_area_sum_cm2=stage_sum,net_final_changed_area_cm2=stage_sum,
        net_step_area_cm2=dict(qa['step_area_cm2']))
    if not count:return result
    assert_frame_parent_identity(row,parent_row)
    before=np.asarray(Image.open(saved.bound(parent_row['labels_map'])))
    after=np.asarray(Image.open(saved.bound(row['labels_map'])))
    masks={step:saved.unpack_mask(row[field],after.shape) for step,field in MASK_FIELDS.items()}
    flagged=np.logical_or.reduce(list(masks.values()));restored=flagged & (before==after)
    if int(restored.sum())!=count:raise ValueError('Saved unchanged-stage restoration count changed')
    _,area=saved.arrays(row['rig_domain_revision'])
    cm2=lambda mask:float(np.sum(area,where=mask,dtype=np.float64))/100.
    result['unchanged_stage_restoration_cm2']=cm2(restored)
    result['net_final_changed_area_cm2']=cm2(before!=after)
    result['net_step_area_cm2']={step:cm2(mask & ~restored) for step,mask in masks.items()}
    return result


def aggregate_identities(qa_rows):
    keys=('changed_unflagged_pixels','unchanged_stage_restoration_pixels','gas_changed_outside_a_d_pixels',
          'co2_union_changed_outside_a_b_c_pixels','domain_changed_outside_a_pixels')
    result={key:sum(int(row[key]) for row in qa_rows) for key in keys}
    for field in ('mask_overlaps_pixels','mask_overlaps_on_changed_pixels'):
        result[field]={key:sum(int(row[field][key]) for row in qa_rows) for key in qa_rows[0][field]}
    result['final_changed_pixels_exactly_one_flag']=(result['changed_unflagged_pixels']==0 and
        not any(result['mask_overlaps_on_changed_pixels'].values()))
    result['labels_elsewhere_bit_identical']=all(row['labels_outside_masks_identical'] for row in qa_rows)
    result['source_time_binding_unchanged_frames']=len(qa_rows)
    result['stage_overlap_is_not_a_final_changed_pixel_failure']=True
    return result


def frame_table(row,restorations=None):
    qa=row["qa"]
    if qa.get("all_pass") is not True:
        raise ValueError("A failed frame cannot become a successful report row")
    return dict(run=row["run"].upper(),frame_index=int(row["frame_index"]),elapsed_h=float(row["elapsed_h"]),
        source_sha256=row["source_sha256"],light_epoch_id=row["light_epoch_id"],
        domain_removed_cm2=float(qa["domain_removed_cm2"]),
        labelled_pixels_removed_below_sand_floor_cm2=float(qa["step_area_cm2"]["A"]),
        co2_removed_below_sand_floor_cm2=float(qa["step_previous_aqueous_cm2"]["A"]+qa["step_previous_gas_cm2"]["A"]),
        gas_removed_below_sand_floor_cm2=float(qa["step_previous_gas_cm2"]["A"]),
        floor_connectivity_removed_cm2=float(qa["step_area_cm2"]["B"]),floor_connectivity_removed_groups=int(qa["floor_connectivity_removed_groups"]),
        removed_group_distance_median_mm=qa["floor_connectivity_distance_median_mm"],removed_group_distance_max_mm=qa["floor_connectivity_distance_max_mm"],
        removed_group_distance_status=qa["floor_connectivity_distance_status"],
        floor_contact_fill_a_cm2=float(qa["fill_reason_a_cm2"]),floor_contact_fill_b_cm2=float(qa["fill_reason_b_cm2"]),
        gas_envelope_removed_cm2=float(qa["step_area_cm2"]["D"]),carryover_mask_known=bool(qa["carryover_mask_known"]),
        filled_under_gas_pixels=int(qa["filled_under_gas_pixels"]),preinjection_fill_pixels=int(qa["preinjection_fill_pixels"]),
        **(restorations or {}),
        backend=qa["backend"],cpu_seconds=qa["cpu_seconds"],gpu_seconds=qa["gpu_seconds"],wall_seconds=qa["wall_seconds"],all_pass=True)


def nearest_frame(rows,time_h):
    if not rows:raise ValueError("No run frames to select")
    return min(rows,key=lambda row:(abs(float(row["elapsed_h"])-time_h),int(row["frame_index"])))


def gas_epoch_summary(rows):
    runs=defaultdict(list)
    for row in rows:runs[row["run"]].append(row)
    summary=[]
    for target in (1.,2.25,4.):
        for epoch in ("rig_light","new_lamps"):
            selected=[nearest_frame(group,target) for group in runs.values() if group[0]["light_epoch_id"]==epoch]
            amounts=[row["gas_envelope_removed_cm2"] for row in selected]
            summary.append(dict(target_h=target,light_epoch_id=epoch,run_count=len(selected),
                median_removed_gas_cm2=float(np.median(amounts)) if amounts else None,
                p90_removed_gas_cm2=float(np.percentile(amounts,90)) if amounts else None,
                source_selection="closest exact camera-time frame; earliest frame index resolves ties",
                selected=[{k:row[k] for k in ("run","frame_index","elapsed_h","source_sha256","gas_envelope_removed_cm2")} for row in selected]))
    return summary


def remaining_detached_floor_mask(labels,corrected_domain):
    """Reuse the exact B connectivity primitives on final saved labels."""
    domain=np.asarray(corrected_domain,bool)
    if domain.shape!=labels.shape:raise ValueError("Detached floor check needs exact corrected rig-domain mask")
    has=domain.any(axis=0)
    if not has.any():return np.zeros(labels.shape,bool)
    floor=np.where(has,labels.shape[0]-1-domain[::-1].argmax(axis=0),-1)
    acting=has & (np.abs(floor-np.median(floor[has]))<=rules.COLUMN_GATE_PIXELS)
    heights=floor[None,:]-np.arange(labels.shape[0])[:,None]
    co2=((labels==1)|(labels==2))&domain
    dilated=rules._dilate(co2,1.5/rules.MM_PER_PIXEL,np,ndimage)
    groups,count=rules._label(dilated,np,ndimage)
    touched=co2 & acting[None,:] & (heights>=0) & (heights<=rules.FLOOR_BAND_PIXELS)
    deeper=co2 & (heights>rules.FLOOR_BAND_PIXELS)
    def present(mask):return np.bincount(groups[mask],minlength=count+1)>0
    detached=present(touched)&~present(deeper);detached[0]=False
    return co2&detached[groups]


def floor_box_bounds(annotation,canvas_wh,phase_shape):
    if annotation["coordinate_space"]!="normalized_native_review_canvas_16x9" or annotation["shape"]!="rectangle":
        raise ValueError("Locked floor box uses another coordinate space or shape")
    W,H=canvas_wh
    points=np.array([[float(annotation["x"])*W,float(annotation["y"])*H],
                     [(float(annotation["x"])+float(annotation["width"]))*W,(float(annotation["y"])+float(annotation["height"]))*H]])
    bounds=np.rint(points).astype(int)
    x0,y0=bounds[0];x1,y1=bounds[1]
    x0=max(0,min(phase_shape[1]-1,x0));x1=max(0,min(phase_shape[1]-1,x1))
    y0=max(0,min(phase_shape[0]-1,y0));y1=max(0,min(phase_shape[0]-1,y1))
    # NumPy rounds the source coordinates, but report JSON must contain native
    # Python integers. The conversion is exact and does not alter the bounds.
    return tuple(int(value) for value in (x0,y0,x1+1,y1+1))


def assert_frame_parent_identity(row,parent_row):
    """A report never pairs frames by index alone."""
    keys=("run","frame_index","elapsed_h","source_sha256","light_epoch_id")
    if any(row[key]!=parent_row[key] for key in keys):
        raise ValueError("Final report source/time/epoch differs from its P1-final parent")
    if row["binding"]["source"]!=parent_row["binding"]["source"]:
        raise ValueError("Final report complete source binding differs from parent")


def floor_boxes_for_frame(annotations,labels,area,canvas_wh,source,corrected_domain):
    if not annotations:return []
    floor=labels.shape[0]-1-corrected_domain[::-1].argmax(axis=0)
    has=corrected_domain.any(axis=0)
    selected=[]
    for annotation in annotations:
        if abs(float(annotation["elapsed_h"])-float(source["elapsed_h"]))>1e-8:
            raise ValueError("Locked carryover box camera time differs from exact saved frame")
        x0,y0,x1,y1=floor_box_bounds(annotation,canvas_wh,labels.shape)
        columns=has[x0:x1]
        distance=np.abs(floor[x0:x1][columns]-(y1-1))*rules.MM_PER_PIXEL
        nearest=float(distance.min()) if len(distance) else None
        if nearest is not None and nearest<=15.:
            selected.append((annotation,(x0,y0,x1,y1),nearest))
    if not selected:return []
    detached=remaining_detached_floor_mask(labels,corrected_domain)
    result=[]
    for annotation,(x0,y0,x1,y1),distance in selected:
        local=detached[y0:y1,x0:x1]
        result.append(dict(run=source["run"].upper(),frame_index=source["frame_index"],elapsed_h=source["elapsed_h"],
            source_sha256=source["source_sha256"],annotation_id=annotation["annotation_id"],bounds_xyxy=[x0,y0,x1,y1],
            nearest_box_bottom_to_floor_mm=distance,detached_co2_pixels=int(local.sum()),
            detached_co2_area_cm2=float(np.sum(area[y0:y1,x0:x1],where=local,dtype=np.float64))/100.,
            passes=not local.any(),source="locked pre-registered floor carryover annotations; report only",independent_sealed_scorer_run=False))
    return result


def outline(mask):
    edge=mask & ~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    return cv2.dilate(edge.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)


def parent_photo(parent,run,index):
    pointer=saved.read(pointer_path(parent))
    manifest=saved.read(saved.bound(pointer["manifest"]))
    local=saved.read(saved.bound(manifest["runs"][run]))
    item=local["records"][f"frame_{index:03d}"]["raw_detail"]
    return np.asarray(Image.open(saved.bound(item["display"])).convert("RGB")),item


def render_pair(parent,final,run,index,output,*,close_floor=False,annotations=()):
    started,cpu=time.perf_counter(),time.process_time()
    before_row=saved.read(saved.frame_path(parent,run,index));after_path=saved.frame_path(final,run,index);after_row=saved.read(after_path)
    assert_frame_parent_identity(after_row,before_row)
    before=np.asarray(Image.open(saved.bound(before_row["labels_map"])));after=np.asarray(Image.open(saved.bound(after_row["labels_map"])))
    rgb,photo_asset=parent_photo(parent,run,index)
    if rgb.shape[:2]!=after.shape:raise ValueError("Reused pre-rendered photograph has another phase canvas")
    masks={step:saved.unpack_mask(after_row[field],after.shape) for step,field in MASK_FIELDS.items()}
    old=saved._phase_overlay(rgb,before);new=saved._phase_overlay(rgb,after);marked=rgb.copy()
    for step,mask in masks.items():marked[outline(mask)]=COLOURS[step]
    crop=[0,0,after.shape[1],after.shape[0]]
    if close_floor:
        # Fixed displayed crop centred on the protocol L1 x proxy. No label or
        # rule uses this crop or derives an injector from the photograph.
        domain,_=saved.arrays(after_row["rig_domain_revision"]);occupied=domain.any(axis=0);bottom=after.shape[0]-1-domain[::-1].argmax(axis=0)
        y=int(np.median(bottom[occupied]));x=int(after.shape[1]*.5)
        crop=[max(0,x-1700),max(0,y-700),min(after.shape[1],x+1700),min(after.shape[0],y+60)]
    x0,y0,x1,y1=crop
    panels=[Image.fromarray(image[y0:y1,x0:x1]) for image in (old,new,marked)]
    box_ids=[]
    if annotations:
        _,native=saved.native_manifest(parent,run)
        for annotation in annotations:
            if int(annotation['frame_index'])!=index or annotation['run'].casefold()!=run.casefold():
                raise ValueError('Figure annotation has another exact source frame')
            bx0,by0,bx1,by1=floor_box_bounds(annotation,native['canvas_wh'],after.shape)
            for image in panels:
                ImageDraw.Draw(image).rectangle((bx0-x0,by0-y0,bx1-1-x0,by1-1-y0),outline=(255,255,255),width=5)
            box_ids.append(annotation['annotation_id'])
    for image in panels:image.thumbnail((1000,640),Image.Resampling.LANCZOS)
    width=max(image.width for image in panels);height=max(image.height for image in panels)
    sheet=Image.new("RGB",(3*width,height+95),"#171a20");draw=ImageDraw.Draw(sheet)
    for n,(image,title) in enumerate(zip(panels,("P1-final","P1-final2","Exact A-D flags; photograph only"))):
        draw.text((n*width+10,8),f"{run.upper()} f{index} t={after_row['elapsed_h']:.3f}h  {title}",fill="white");sheet.paste(image,(n*width,90))
    draw.text((10,32),"Cyan=A below sand floor; yellow=B disconnected floor CO2; purple=C contact fill; pink=D gas envelope",fill="white")
    caption=(f"White boxes: {len(box_ids)} locked no-gas annotations; displayed only, not scored." if box_ids else
             "Frozen parent RAW-detail photograph reused; bottom blocked for mass. Outlines are display-only.")
    draw.text((10,54),caption,fill="#b6c7d8")
    path=Path(output)/f"{run.upper()}_frame_{index:03d}_{'floor' if close_floor else 'gas'}_comparison.png";sheet.save(path)
    return dict(run=run.upper(),frame_index=index,elapsed_h=after_row["elapsed_h"],source_sha256=after_row["source_sha256"],
        figure=saved.artifact(path),photo_asset=photo_asset,parent_receipt=saved.artifact(saved.frame_path(parent,run,index)),final_receipt=saved.artifact(after_path),
        display_crop_xyxy=crop,close_floor=close_floor,displayed_annotation_ids=box_ids,
        independent_sealed_scorer_run=False,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu)


def montage(output,rows):
    cellw,cellh,columns=900,255,4
    sheet=Image.new("RGB",(columns*cellw,70+math.ceil(len(rows)/columns)*cellh),"#171a20");draw=ImageDraw.Draw(sheet)
    draw.text((15,10),"Every run nearest 12h: left P1-final, middle final2, right removed gas (pink) and other fixed flags",fill="white")
    draw.text((15,34),"Exact parent pre-rendered RAW-detail photograph reused; no map or label changed by this report.",fill="#c7d4df")
    for n,row in enumerate(rows):
        image=Image.open(row["figure"]["path"]).convert("RGB");image.thumbnail((cellw-8,cellh-8),Image.Resampling.LANCZOS)
        sheet.paste(image,((n%columns)*cellw+4,70+(n//columns)*cellh+4))
    path=Path(output)/"gas_removed_envelope_all_43_nearest_12h.png";sheet.save(path);return saved.artifact(path)


def floor_gate_figures(a_report,gate):
    """Follow the measured gate's import proof; never invent a local path."""
    a_report=Path(a_report);manifest=a_report.parent/'floor_measurement_figures.json'
    if manifest.is_file():return saved.read(manifest)
    proof=gate.get('measurement_import',{})
    if (proof.get('contract')!='ffac-byte-equivalent-floor-gate-import-v1' or
        proof.get('all_measurement_values_and_gate_unchanged') is not True or
        proof.get('curve_bytes_sha_unchanged') is not True):
        raise ValueError('Imported A gate needs exact original measurement proof')
    original=saved.bound(proof['original_gate']);old_gate=saved.read(original)
    if old_gate['gate']!=gate['gate'] or old_gate['per_run']!=gate['per_run']:
        raise ValueError('Imported A gate measurement or acceptance values changed')
    result=saved.read(original.parent/'floor_measurement_figures.json')
    assets=[]
    for asset in result['figures']:
        imported=saved.artifact(a_report.parent/Path(asset['path']).name)
        if imported['sha256']!=asset['sha256']:raise ValueError('Imported A figure changed')
        assets.append(imported)
    residual=saved.artifact(a_report.parent/Path(result['residual_table']['path']).name)
    if residual['sha256']!=result['residual_table']['sha256']:raise ValueError('Imported A residual table changed')
    return dict(contract=result['contract'],gate_report=saved.artifact(a_report),figures=assets,
                residual_table=residual,measurement_import=proof,scientific_assets_changed=False)


def run_report(parent,candidate,a_report,annotations,output,workers=6):
    start,cpu=time.perf_counter(),time.process_time();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    parent,candidate=Path(parent).resolve(),Path(candidate).resolve()
    seal_asset=saved.artifact(candidate/"rig_camera_candidate_receipt.json");seal=saved.read(seal_asset["path"])
    if seal.get("contract")!=rules.CONTRACT or seal.get("complete") is not True or (seal.get("run_count"),seal.get("frame_count"))!=(43,864) or seal.get("bottom_mass_allowed") is not False:
        raise ValueError("Report needs complete sealed 43/864 final2 candidate with bottom mass blocked")
    A=saved.read(a_report)
    if A["gate"]["passes"] is not True or A["candidate"]!=saved.artifact(parent/"rig_camera_candidate_receipt.json"):
        raise ValueError("Source-bound A gate does not authorize this parent")
    annotation_asset=saved.artifact(annotations)
    if annotation_asset["sha256"]!=ANNOTATIONS_SHA256:raise ValueError("Locked carryover annotation CSV changed")
    with Path(annotations).open(encoding="utf-8-sig",newline="") as f:
        all_annotations=list(csv.DictReader(f))
        carryover=[row for row in all_annotations if row["category"]=="incomplete_wash_indicator_carryover"]
    annotation_groups=defaultdict(list)
    for row in carryover:annotation_groups[(row["run"].casefold(),int(row["frame_index"]))].append(row)
    tables=[];group_rows=[];boxed=[];figures=[];by_run=defaultdict(list);qa_rows=[]
    for item in seal["frames"]:
        path=saved.bound(item["receipt"],base=candidate);row=saved.read(path)
        if row.get("complete") is not True or (row["run"].casefold(),int(row["frame_index"]))!=(item["run"].casefold(),int(item["frame_index"])):
            raise ValueError("Sealed frame receipt identity changed")
        parent_row=saved.read(saved.frame_path(parent,row['run'],row['frame_index']))
        assert_frame_parent_identity(row,parent_row)
        if Path(saved.bound(row["labels_map"])).resolve()!=(candidate/"production_labels"/row["run"].casefold()/f"frame_{row['frame_index']:03d}.png").resolve():
            raise ValueError("Regression export is not the mandated production_labels layout")
        table=frame_table(row,restoration_accounting(row,parent_row));tables.append(table);by_run[table["run"]].append(table);qa_rows.append(row['qa'])
        for n,distance in enumerate(row["qa"]["floor_connectivity_removed_group_min_distance_to_kept_co2_mm"]):
            group_rows.append(dict(run=table["run"],frame_index=table["frame_index"],elapsed_h=table["elapsed_h"],source_sha256=table["source_sha256"],group_diagnostic_ordinal=n+1,minimum_distance_to_kept_co2_mm=distance))
        boxes=annotation_groups.get((row["run"].casefold(),int(row["frame_index"])),[])
        if boxes:
            _,manifest=saved.native_manifest(parent,row["run"].casefold())
            labels=np.asarray(Image.open(saved.bound(row["labels_map"])))
            corrected_domain,area=saved.arrays(row["rig_domain_revision"])
            boxed.extend(floor_boxes_for_frame(boxes,labels,area,manifest["canvas_wh"],row,corrected_domain))
    if len(tables)!=864 or len(by_run)!=43 or len({(r['run'],r['frame_index']) for r in tables})!=864:
        raise ValueError("Duplicate or incomplete source frame inventory")
    tables.sort(key=lambda r:(r["run"],r["frame_index"]))
    epoch_counts=Counter(group[0]["light_epoch_id"] for group in by_run.values())
    if epoch_counts!={"rig_light":31,"new_lamps":12}:raise ValueError("Light epoch inventory changed")
    residual_by_run={row["run"]:row for row in A["per_run"]};run_rows=[]
    for run,group in sorted(by_run.items()):
        group.sort(key=lambda r:r["frame_index"])
        values=[row["domain_removed_cm2"] for row in group]
        run_rows.append(dict(run=run,frame_count=len(group),domain_removed_first_frame_cm2=values[0],domain_removed_last_frame_cm2=values[-1],domain_removed_median_cm2=float(np.median(values)),
            domain_removed_min_cm2=min(values),domain_removed_max_cm2=max(values),co2_removed_sum_across_frames_cm2=sum(row["co2_removed_below_sand_floor_cm2"] for row in group),
            time_sum_is_one_physical_area=False,reference_median_absolute_residual_mm=residual_by_run[run]["reference"]["median_absolute_residual_mm"],last_median_absolute_residual_mm=residual_by_run[run]["last"]["median_absolute_residual_mm"],
            carryover_mask_known=all(row["carryover_mask_known"] for row in group)))
    for name,rows in (("A_B_C_D_by_run_frame.csv",tables),("A_domain_removed_and_residuals_by_run.csv",run_rows),("B_removed_groups_distance_to_kept_CO2.csv",group_rows),("B_locked_floor_carryover_boxes.csv",boxed)):
        csv_new(output/name,rows)
    restoration_rows=[row for row in tables if row['unchanged_stage_restoration_pixels']]
    csv_new(output/'unchanged_stage_restorations_by_frame.csv',restoration_rows)
    epochs=gas_epoch_summary(tables);csv_new(output/"D_removed_gas_medians_by_light_epoch.csv",[{k:v for k,v in row.items() if k!="selected"} for row in epochs])
    selections={(run.casefold(),nearest_frame(group,12.)["frame_index"],False) for run,group in by_run.items()}
    selections.update({("ac17",5,True),("ac17",17,True),("ac23",10,True),("ac52",6,True),("ac52",14,True)})
    selections.update({("ac52",6,False),("ac52",14,False)})
    selections.update((run.casefold(),nearest_frame(by_run[run],target)["frame_index"],False) for run,target in (("AC22",6.),("AC22",24.),("AC28",24.)))
    with ProcessPoolExecutor(max_workers=workers,initializer=saved._init_worker) as pool:
        pending=[pool.submit(render_pair,parent,candidate,run,index,output,close_floor=close,
            annotations=([a for a in all_annotations if a['run'].casefold()==run and int(a['frame_index'])==index and
                          a['category']=='confirmed_no_free_gas_in_selected_area'] if run=='ac52' and not close else ()))
            for run,index,close in sorted(selections)]
        for future in as_completed(pending):
            value=future.result();figures.append(value);print(f"Report figure {value['run']} f{value['frame_index']}",flush=True)
    twelve_keys={(run,nearest_frame(group,12.)["frame_index"]) for run,group in by_run.items()}
    twelve=sorted([row for row in figures if not row["close_floor"] and (row["run"],row["frame_index"]) in twelve_keys],key=lambda row:row["run"])
    image=montage(output,twelve)
    code_dir=output/'code';code_dir.mkdir()
    code_assets=[]
    for path in (Path(__file__),Path(rules.__file__)):
        copy=code_dir/path.name;shutil.copyfile(path,copy);code_assets.append(saved.artifact(copy))
    a_figures=floor_gate_figures(a_report,A)
    for asset in a_figures['figures']:saved.bound(asset)
    identities=aggregate_identities(qa_rows)
    report=dict(contract=CONTRACT,complete=True,spec_sha256=rules.SPEC_SHA256,candidate=seal_asset,parent=saved.artifact(parent/"rig_camera_candidate_receipt.json"),
        code=code_assets,production_qa=seal['qa'],identities=identities,
        unchanged_stage_restorations=restoration_rows,
        area_accounting='Step areas count stage operations. Net changed areas count each final changed pixel once; restored pixels are listed separately and are unchanged from P1-final.',
        A_gate=saved.artifact(a_report),A_gate_counts=A["gate"],A_figures=a_figures,run_count=43,frame_count=864,per_run=run_rows,per_frame=tables,
        B_floor_carryover_boxes=dict(source=annotation_asset,locked_carryover_boxes=len(carryover),selected_floor_boxes=len(boxed),passing_boxes=sum(row["passes"] for row in boxed),
            failing_boxes=[row for row in boxed if not row["passes"]],per_box=boxed),D_epoch_medians=epochs,
        C_runs_without_carryover_mask=[row["run"] for row in run_rows if not row["carryover_mask_known"]],
        C_filled_under_gas_pixels=sum(row["filled_under_gas_pixels"] for row in tables),C_preinjection_fill_pixels=sum(row["preinjection_fill_pixels"] for row in tables),
        regression_export=dict(layout="production_labels/<run>/frame_NNN.png",source_bound_frames=864,independent_sealed_scorer_run=False),
        figures=figures,twelve_hour_montage=image,
        figure_annotation_policy='Only locked no-gas boxes from the exact frame are outlined; no interpolation or scoring. AC52 frame 6 has none in the snapshot.',
        cost=dict(report_wall_seconds=time.perf_counter()-start,report_cpu_seconds=time.process_time()-cpu+sum(row["cpu_seconds"] for row in figures),report_gpu_seconds=0.,
            report_backend="cpu-read-only-saved-tables-and-display-files",production_backend_counts=dict(Counter(row["backend"] for row in tables)),
            production_cpu_seconds=sum(row["cpu_seconds"] for row in tables),production_gpu_seconds=sum(row["gpu_seconds"] for row in tables)),
        scientific_assets_changed=False,labels_changed=False,bottom_mass_allowed=False,mass_eligible=False,independent_regression_pending=True)
    saved.write_new(output/"p1_final2_report.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("parent","candidate","a-report","annotations","output"):parser.add_argument("--"+field,required=True)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args(argv);run_report(args.parent,args.candidate,args.a_report,args.annotations,args.output,args.workers)


if __name__=="__main__":main()

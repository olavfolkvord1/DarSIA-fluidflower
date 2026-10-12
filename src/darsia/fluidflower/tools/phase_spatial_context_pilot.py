"""Isolated, fixed-budget regional context corrector; no production promotion.

Whole RUN18 is excluded from corrective training. The frozen optical model and
delivered prior previously saw RUN18: this is not independent generalization.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import process_directory, series_root, series_path, package_root, local_cache_root
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO = series_root("ffac")
None
from darsia.fluidflower.tools.phase_spatial_pilot_evidence import CAMPAIGN,load_frame,read,sha,write
from darsia.fluidflower.tools.phase_coupled_constraint_comparison import annotation_window
from darsia.fluidflower.tools.phase_training_review_comparison import fractions
from darsia.fluidflower.tools.compare_ac_um_review_update_20260914 import constraints
from darsia.fluidflower.tools.phase_model_comparison_unet import _build_model
from darsia.fluidflower.phase_training.logistic import ReviewedRegion,regional_presence_target

ROOT=CAMPAIGN/'context_corrector'
PATCH=128
SPACING=.001
SEED=20260914
BASE_LOGIT=float(np.log(4.))


def protocol(manifest):
    data=read(manifest)
    train=[f for f in data['frames'] if f['run']!='run18' and f['regions']]
    held=[f for f in data['frames'] if f['run']=='run18']
    return dict(format='regional-context-corrector-pilot-v1',seed=SEED,
        evidence_manifest=str(manifest),evidence_manifest_sha256=sha(manifest),
        train_frames=[f['frame_id'] for f in train],held_frames=[f['frame_id'] for f in held],
        train_region_records=sum(len(f['regions']) for f in train),held_region_records=sum(len(f['regions']) for f in held),
        unlabelled_excluded=[f['frame_id'] for f in data['frames'] if not f['regions']],
        train_whole_runs=sorted({f['whole_run_group'] for f in train}),held_whole_runs=['ffum/run18'],
        independent_labeled_accuracy=False,held_run_labels_used_for_fit_or_selection=False,
        inputs=['current RGB','pre-injection RGB','raw CO2 probability','delivered CO2 fraction','delivered gas fraction'],
        channel_count=9,working_spacing_m=SPACING,patch_shape=[PATCH,PATCH],
        supervised_region_bags='Complete regions partitioned over non-overlapping physical patches. Weak positive loss aggregates across the whole bag, never each partial patch.',
        loss_semantics={'pure':'Joint water, aqueous or gas likelihood on all labelled pixels',
            'positive_region':'Negative log of area-weighted best 10% joint phase probability; no target phase area is imposed',
            'negative_region':'Negative log of one minus joint phase probability on every annotated pixel',
            'mixed_aqueous_water':'Gas absence over whole mask; separate aqueous/water MIL presence',
            'mixed_gas_aqueous':'CO2 presence throughout; separate aqueous/gas MIL presence',
            'unlabelled_context':'Only weak prior tether and guide-weighted neighbour consistency; never treated as water truth'},
        fixed_training=dict(epochs=8,learning_rate=.001,base_channels=8,prior_weight=.02,spatial_weight=.002,
            device='cuda',threads=4,maximum_fit_seconds=480,throughput_gate_steps=32),
        normalization='Per-region loss; equal mean weight per training run and class stratum, without held-run tuning.',
        limitations=['Frozen optical and delivered prior already saw held RUN18; this is conditional correction transfer, not independent architecture generalization.',
            'No wash-reference input and no newly learned or replayed temporal history.',
            'Supervision uses area-resampled annotation masks on a 1mm grid; native evaluation retains exact original polygons. Submillimetre controls share a contextual score.',
            'No dense boundary truth. Regional presence and shorter contours cannot certify pixel-exact fronts.',
            'Output is evaluated on labelled native crops, not a new full native chronological materialization.',
            'AC44 has no labels and is excluded from fit and quantitative holdout evaluation.'])


def _tiles(start,stop,limit):
    if stop-start<=PATCH:
        return [max(0,min(limit-PATCH,(start+stop-PATCH)//2))]
    first=max(0,min(start,limit-PATCH))
    return list(range(first,stop,PATCH))


def make_bag(entry,record,shape):
    region=ReviewedRegion(**record)
    bounds,mask=annotation_window(region.payload,canvas_wh=entry['canvas_wh'],content_box=entry['content_box_xywh'])
    x0,y0,x1,y1=bounds;nh,nw=entry['content_box_xywh'][3],entry['content_box_xywh'][2]
    h,w=shape;sx=w/nw;sy=h/nh
    ax=max(0,int(np.floor(x0*sx)));ay=max(0,int(np.floor(y0*sy)))
    bx=min(w,max(ax+1,int(np.ceil(x1*sx))));by=min(h,max(ay+1,int(np.ceil(y1*sy))))
    weights=cv2.resize(mask.astype(np.float32),(bx-ax,by-ay),interpolation=cv2.INTER_AREA)
    bag=[];covered=0.
    for yy in _tiles(ay,by,h):
        for xx in _tiles(ax,bx,w):
            tile=np.zeros((PATCH,PATCH),np.float32)
            iy0=max(ay,yy);iy1=min(by,yy+PATCH);ix0=max(ax,xx);ix1=min(bx,xx+PATCH)
            tile[iy0-yy:iy1-yy,ix0-xx:ix1-xx]=weights[iy0-ay:iy1-ay,ix0-ax:ix1-ax]
            if tile.sum()>0:bag.append((yy,xx,tile));covered+=float(tile.sum())
    if not np.isclose(covered,weights.sum(),rtol=1e-5):raise ValueError('Incomplete or duplicated regional bag')
    return dict(frame_id=entry['frame_id'],record=record,bounds=bounds,tiles=bag,mask_mass=covered)


def prepare(manifest,plan):
    data=read(manifest);directory=ROOT/'prepared';directory.mkdir(parents=True,exist_ok=True)
    report=[]
    for entry in data['frames']:
        if entry['frame_id'] not in plan['train_frames']+plan['held_frames']:continue
        destination=directory/(entry['frame_id']+'.npz')
        if destination.exists():raise FileExistsError(destination)
        f=load_frame(data,entry['frame_id'])
        if f['raw_co2'] is None:raise ValueError('Frozen raw optical cache is missing')
        nh,nw=f['domain'].shape;dy,dx=f['spacing_m'];shape=(round(nh*dy/SPACING),round(nw*dx/SPACING));target=shape[::-1]
        rgb=np.concatenate([cv2.resize(f[k],target,interpolation=cv2.INTER_AREA) for k in ('current_rgb','baseline_rgb')],axis=-1)
        raw=cv2.resize(f['raw_co2'],target,interpolation=cv2.INTER_AREA)
        co2=cv2.resize((f['saved_labels']>0).astype(np.float32),target,interpolation=cv2.INTER_AREA)
        gas=cv2.resize((f['saved_labels']==2).astype(np.float32),target,interpolation=cv2.INTER_AREA)
        domain=cv2.resize(f['domain'].astype(np.float32),target,interpolation=cv2.INTER_AREA)
        np.savez_compressed(destination,rgb=rgb,raw=raw.astype(np.float16),co2=co2.astype(np.float16),
            gas=gas.astype(np.float16),domain=domain.astype(np.float16))
        bags=[make_bag(entry,r,shape) for r in entry['regions']]
        report.append(dict(frame_id=entry['frame_id'],path=str(destination),sha256=sha(destination),
            working_shape=list(shape),native_shape=[nh,nw],regions=len(bags),
            complete_region_bags=len(bags),tiles=sum(len(b['tiles']) for b in bags),
            maximum_bag_tiles=max([len(b['tiles']) for b in bags] or [0]),excluded_incomplete_regions=0))
        print(json.dumps(dict(prepared=entry['frame_id'],regions=len(bags),tiles=report[-1]['tiles'])),flush=True)
        del f,bags
    write(ROOT/'preparation.json',dict(frames=report,all_region_bags_complete=True))
    return report


def _weighted_mean(values,weights):return (values*weights).sum()/weights.sum().clamp_min(1e-7)


def _mil(probability,weights):
    """Weighted best decile; target probability, never a 10% dense phase mask."""
    import torch
    selected=weights>0;p=probability[selected];w=weights[selected]
    order=torch.argsort(p,descending=True);p=p[order];w=w[order]
    mass=w.sum()*.1
    used=torch.clamp(mass-(torch.cumsum(w,0)-w),min=0)
    used=torch.minimum(used,w)
    return -torch.log((p*used).sum()/mass.clamp_min(1e-7)+1e-7)


def supervision_loss(probability,weights,phase_class):
    import torch
    c,g=probability[:,0],probability[:,1]
    values=dict(co2=c,water=1-c,aqueous=c*(1-g),gas=c*g)
    def pos(name):return _mil(values[name],weights)
    def neg(name):return _weighted_mean(-torch.log((1-values[name]).clamp_min(1e-7)),weights)
    def dense(name):return _weighted_mean(-torch.log(values[name].clamp_min(1e-7)),weights)
    if phase_class in ('water','aqueous','gas'):return dense(phase_class)
    if phase_class=='aqueous_present_region':return .35*pos('aqueous')
    if phase_class=='aqueous_absent_region':return neg('aqueous')
    if phase_class=='gas_present_region':return .35*pos('gas')
    if phase_class=='gas_absent_region':return neg('gas')
    if phase_class=='co2_absent_region':return neg('co2')
    if phase_class=='mixed_aqueous_water':return neg('gas')+.35*(pos('aqueous')+pos('water'))
    if phase_class=='mixed_gas_aqueous':return dense('co2')+.35*(pos('aqueous')+pos('gas'))
    raise ValueError('Unsupported region semantics: '+phase_class)


def _network(torch):
    model=_build_model(torch,base_channels=8)
    model.encoder1.layers[0]=torch.nn.Conv2d(9,8,3,padding=1)
    torch.nn.init.zeros_(model.heads.weight);torch.nn.init.zeros_(model.heads.bias)
    return model


def _base(torch,x):
    c=.2+.6*x[:,7];gas=.2+.6*x[:,8]
    return torch.logit(torch.stack((c,gas),dim=1).clamp(.01,.99))


def _regularization(torch,probability,x,domain):
    base=torch.sigmoid(_base(torch,x))
    tether=_weighted_mean((probability-base).square().mean(1),domain)
    change=x[:,:3]-x[:,3:6]
    losses=[]
    for axis in (2,3):
        a=[slice(None)]*4;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        guide=(change[tuple(a)]-change[tuple(b)]).square().mean(1)
        weight=torch.exp(-guide/.0025)
        da=[slice(None)]*3;db=da.copy();da[axis-1]=slice(1,None);db[axis-1]=slice(None,-1)
        weight=weight*domain[tuple(da)]*domain[tuple(db)]
        losses.append(_weighted_mean((probability[tuple(a)]-probability[tuple(b)]).abs().mean(1),weight))
    return tether,sum(losses)/2


def load_prepared(entry,device):
    import torch
    with np.load(ROOT/'prepared'/(entry['frame_id']+'.npz')) as a:
        x=np.concatenate((a['rgb'].astype(np.float32).transpose(2,0,1)/255,
            a['raw'][None].astype(np.float32),a['co2'][None].astype(np.float32),a['gas'][None].astype(np.float32)),axis=0)
        domain=a['domain'].astype(np.float32)
    return torch.from_numpy(x).to(device),torch.from_numpy(domain).to(device)


def _bag_arrays(torch,item,arrays):
    x,domain=arrays[item['frame_id']];patches=[];domains=[];masks=[]
    for yy,xx,mask in item['tiles']:
        piece=x[:,yy:yy+PATCH,xx:xx+PATCH];d=domain[yy:yy+PATCH,xx:xx+PATCH]
        ph=PATCH-piece.shape[1];pw=PATCH-piece.shape[2]
        patches.append(torch.nn.functional.pad(piece,(0,pw,0,ph),mode='replicate'))
        domains.append(torch.nn.functional.pad(d,(0,pw,0,ph)))
        masks.append(torch.from_numpy(mask).to(x.device))
    tx=torch.stack(patches);td=torch.stack(domains);tm=torch.stack(masks)*td
    return tx,td,tm


def fit(manifest,plan):
    import torch
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=read(manifest);entries=[f for f in data['frames'] if f['frame_id'] in plan['train_frames']]
    arrays={f['frame_id']:load_prepared(f,'cuda') for f in entries};bags=[]
    for e in entries:
        shape=tuple(arrays[e['frame_id']][0].shape[-2:])
        bags.extend(make_bag(e,r,shape) for r in e['regions'])
    strata=Counter((b['record']['run'],b['record']['phase_class']) for b in bags)
    classes=Counter(k[0] for k in strata);runs=len(classes)
    for b in bags:
        key=(b['record']['run'],b['record']['phase_class'])
        b['weight']=len(bags)/(runs*classes[key[0]]*strata[key])
    model=_network(torch).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(SEED);history=[];start=time.perf_counter();steps=0;completed=0
    initial_loss=None
    for epoch in range(plan['fixed_training']['epochs']):
        model.train();values=[];component=[]
        for index in rng.permutation(len(bags)):
            item=bags[int(index)];x,domain,mask=_bag_arrays(torch,item,arrays)
            if mask.sum().item()==0:continue
            optimizer.zero_grad(set_to_none=True)
            p=torch.sigmoid(_base(torch,x)+model(x))
            supervised=supervision_loss(p,mask,item['record']['phase_class'])*item['weight']
            tether,spatial=_regularization(torch,p,x,domain)
            loss=supervised+.02*tether+.002*spatial
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            values.append(float(loss.detach()));component.append(float(supervised.detach()));steps+=1
            if steps==32:
                elapsed=time.perf_counter()-start;expected=elapsed/32*len(bags)*plan['fixed_training']['epochs']
                write(ROOT/'throughput_gate.json',dict(steps=32,seconds=elapsed,projected_fit_seconds=expected,
                    maximum_fit_seconds=480,passed=expected<=480))
                print(json.dumps(dict(throughput_projected_seconds=expected)),flush=True)
                if expected>480:raise RuntimeError('Throughput gate rejected a misleading truncated neural fit; protocol and timing only')
            if time.perf_counter()-start>480:raise RuntimeError('Fixed fit exceeded resource limit; do not interpret incomplete candidate')
        completed+=1
        row=dict(epoch=epoch+1,steps=steps,mean_training_objective=float(np.mean(values)),
            mean_supervision=float(np.mean(component)),seconds=time.perf_counter()-start)
        history.append(row);print(json.dumps(row),flush=True)
    checkpoint=ROOT/'context_corrector.pt'
    torch.save(dict(state_dict=model.state_dict(),protocol=plan),checkpoint)
    report=dict(complete=completed==plan['fixed_training']['epochs'],steps=steps,epochs_completed=completed,
        training_seconds=time.perf_counter()-start,training_history=history,checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),parameters=sum(p.numel() for p in model.parameters()),
        unique_training_regions=len(bags),all_training_bags_complete=True,held_labels_used=False,
        optimization_convergence_certified=False,checkpoint_selection='Final precommitted epoch, no held-run selection')
    write(ROOT/'fit_report.json',report)
    del arrays,optimizer
    torch.cuda.empty_cache()
    return model,report


def predict_working(model,x):
    import torch
    _,h,w=x.shape;output=np.zeros((2,h,w),np.float32);mass=np.zeros((h,w),np.float32)
    starts=lambda size:sorted(set(list(range(0,max(size-PATCH,0)+1,PATCH-32))+[max(0,size-PATCH)]))
    locations=[(y,z) for y in starts(h) for z in starts(w)]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(locations),8):
            group=locations[start:start+8]
            batch=torch.stack([x[:,y:y+PATCH,z:z+PATCH] for y,z in group])
            values=model(batch).float().cpu().numpy()
            for (y,z),v in zip(group,values):
                output[:,y:y+PATCH,z:z+PATCH]+=v;mass[y:y+PATCH,z:z+PATCH]+=1
    return output/np.maximum(mass,1e-7)


def _native_residual(residual,bounds,native_shape):
    x0,y0,x1,y1=bounds;h,w=native_shape;wh,ww=residual.shape[-2:]
    xs=(np.arange(x0,x1,dtype=np.float32)+.5)*ww/w-.5
    ys=(np.arange(y0,y1,dtype=np.float32)+.5)*wh/h-.5
    mx,my=np.meshgrid(xs,ys)
    return np.stack([cv2.remap(v,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE) for v in residual])


def _native_prediction(saved,residual):
    base=np.stack((np.where(saved>0,BASE_LOGIT,-BASE_LOGIT),np.where(saved==2,BASE_LOGIT,-BASE_LOGIT)))
    p=1/(1+np.exp(-np.clip(base+residual,-30,30)))
    labels=(p[0]>=.5).astype(np.uint8);labels[(p[0]>=.5)&(p[1]>=.6)]=2
    return labels,p


def evaluate(manifest,plan,model):
    from PIL import Image,ImageDraw
    data=read(manifest);result=[];crop_root=ROOT/'crops';crop_root.mkdir(exist_ok=True)
    for entry in data['frames']:
        if entry['frame_id'] not in plan['held_frames']:continue
        f=load_frame(data,entry['frame_id']);x,_=load_prepared(entry,'cuda')
        residual=predict_working(model,x);path=ROOT/(entry['frame_id']+'_working_residual.npy')
        np.save(path,residual.astype(np.float16))
        rows=[]
        for record in entry['regions']:
            region=ReviewedRegion(**record)
            bounds,mask=annotation_window(region.payload,canvas_wh=entry['canvas_wh'],content_box=entry['content_box_xywh'])
            x0,y0,x1,y1=bounds;window=np.s_[y0:y1,x0:x1];mask &= f['domain'][window]
            if not mask.any():continue
            old=f['saved_labels'][window];new,p=_native_prediction(old,_native_residual(residual,bounds,f['domain'].shape))
            a,b=fractions(old,mask),fractions(new,mask)
            phases={}
            for phase,key in [('co2_presence','co2'),('aqueous','aqueous_evidence'),('gas','gas_evidence')]:
                target=regional_presence_target(region.phase_class,phase);aa=a[key+'_pixels'];bb=b[key+'_pixels']
                phases[phase]=dict(target=target,old_pixels=aa,new_pixels=bb,change_pixels=bb-aa,
                    change_pp=100*(b[key+'_fraction']-a[key+'_fraction']),
                    positive_newly_empty=target==1 and aa>0 and bb==0,
                    gained_any=target==1 and aa==0 and bb>0,
                    gained_single_pixel=target==1 and aa==0 and bb==1,
                    positive_coverage_reduced=target==1 and bb<aa,
                    negative_coverage_increased=target==0 and bb>aa)
            rows.append(dict(region_id=region.region_id,phase_class=region.phase_class,bounds=list(bounds),before=a,after=b,
                before_satisfied=constraints(a,region)['satisfied'],after_satisfied=constraints(b,region)['satisfied'],
                phases=phases,gas_label_changes=int(((old==2)!=(new==2))[mask].sum())))
        positive=[r for r in rows if any(p['target']==1 for p in r['phases'].values())]
        negative=[r for r in rows if any(p['target']==0 for p in r['phases'].values())]
        picks=[]
        if positive:
            picks.append(('largest_positive_gain',max(positive,key=lambda r:max(p['change_pixels'] if p['target']==1 else -1e12 for p in r['phases'].values()))))
            picks.append(('largest_positive_loss',min(positive,key=lambda r:min(p['change_pixels'] if p['target']==1 else 1e12 for p in r['phases'].values()))))
        if negative:
            picks.append(('largest_negative_worsening',max(negative,key=lambda r:max(p['change_pixels'] if p['target']==0 else -1e12 for p in r['phases'].values()))))
        crop_paths=[]
        for name,row in picks:
            x0,y0,x1,y1=row['bounds'];pad=40;h,w=f['domain'].shape
            bounds=[max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad)];xx0,yy0,xx1,yy1=bounds
            window=np.s_[yy0:yy1,xx0:xx1];photo=f['current_rgb'][window];old=f['saved_labels'][window]
            new,_=_native_prediction(old,_native_residual(residual,bounds,f['domain'].shape))
            panels=[]
            for title,labels in [('Photo',None),('Delivered',old),('Context corrector',new)]:
                pixels=photo.copy()
                if labels is not None:
                    for label,color in ((1,(45,190,235)),(2,(255,167,40))):
                        selected=labels==label;pixels[selected]=np.rint(.5*pixels[selected]+.5*np.array(color)).astype(np.uint8)
                im=Image.fromarray(pixels);im.thumbnail((550,400));panel=Image.new('RGB',(im.width,im.height+30),'white');panel.paste(im,(0,30))
                draw=ImageDraw.Draw(panel);draw.text((5,6),title,fill='black')
                scale=im.width/photo.shape[1]
                draw.rectangle(((x0-xx0)*scale,(y0-yy0)*scale+30,(x1-xx0)*scale,(y1-yy0)*scale+30),outline=(255,0,100),width=2)
                panels.append(panel)
            canvas=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white');offset=0
            for panel in panels:canvas.paste(panel,(offset,0));offset+=panel.width
            destination=crop_root/f"{entry['frame_id']}_{name}.png";canvas.save(destination)
            crop_paths.append(dict(kind=name,region_id=row['region_id'],path=str(destination)))
        counts={key:sum(any(p[key] for p in r['phases'].values()) for r in rows) for key in ('positive_newly_empty','gained_any','gained_single_pixel','positive_coverage_reduced','negative_coverage_increased')}
        frame_report=dict(frame_id=entry['frame_id'],evaluation_role='held_corrector_run_with_training_seen_upstream_prior',
            evaluated_regions=len(rows),strict_before=sum(r['before_satisfied'] for r in rows),strict_after=sum(r['after_satisfied'] for r in rows),
            improved=sum(not r['before_satisfied'] and r['after_satisfied'] for r in rows),
            worsened=sum(r['before_satisfied'] and not r['after_satisfied'] for r in rows),counters=counts,
            working_residual_path=str(path),working_residual_sha256=sha(path),crops=crop_paths,regions=rows)
        result.append(frame_report);print(json.dumps({k:v for k,v in frame_report.items() if k not in ('regions','crops')}),flush=True)
        write(ROOT/'evaluation.json',dict(complete=len(result)==len(plan['held_frames']),frames=result,
            independent_labeled_accuracy=False,no_held_checkpoint_selection=True))
        del f,x,residual
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=CAMPAIGN/'evidence_manifest.json')
    parser.add_argument('--action',choices=['prepare','train','all'],default='all')
    args=parser.parse_args();cv2.setNumThreads(4);ROOT.mkdir(parents=True,exist_ok=True)
    plan=protocol(args.manifest)
    if (ROOT/'protocol.json').exists():
        if read(ROOT/'protocol.json')!=plan:raise ValueError('The frozen neural protocol changed')
    else:write(ROOT/'protocol.json',plan)
    if args.action in ('prepare','all'):prepare(args.manifest,plan)
    if args.action in ('train','all'):
        model,fit_report=fit(args.manifest,plan);evaluate(args.manifest,plan,model)


if __name__=='__main__':main()

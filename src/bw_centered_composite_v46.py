"""Observed X plus one centered line, or two with one vertical, for v46.

Positive ink composition, not errorbar erasure. Pure-line fits are independently
optimized as a null model; native non-X detections are never replaced.
"""
from dataclasses import dataclass, asdict
import time
import cv2
import numpy as np
import bw_observed_raster_v46 as E

VERSION = "bw-centered-x-composite-v1"
MODES = ("one_or_two",)

@dataclass(frozen=True)
class Config:
    angles:tuple=tuple(range(0,180,15))  # direction angle: 0 horizontal, 90 vertical
    widths:tuple=(.8,1.6)
    amplitudes_relative_to_swatch_peak:tuple=(.4,.8,1.2)
    gains:tuple=(.85,1.,1.15)
    complexity_per_line:float=.02
    maximum_loss:float=.48
    minimum_identity_margin:float=.035
    minimum_line_improvement:float=.06
    minimum_visible_ink:float=.70
    chunk:int=512


def stroke(size,center,angle,width):
    y,x=np.indices((size,size),dtype=np.float32)
    rad=np.deg2rad(angle)
    distance=-(x-center[0])*np.sin(rad)+(y-center[1])*np.cos(rad)
    return np.clip(.5+(width/2-np.abs(distance))/.8,0,1).astype(np.float32)


def line_styles(size,center,peak,cfg=Config()):
    lines=[np.zeros((size,size),np.float32)];pars=[[]]
    singles=[]
    for angle in cfg.angles:
        for width in cfg.widths:
            for relative in cfg.amplitudes_relative_to_swatch_peak:
                amp=float(np.clip(peak*relative,0,1))
                raster=amp*stroke(size,center,angle,width)
                record=dict(angle_degrees=int(angle),width_px=width,amplitude=amp,
                            center=list(map(float,center)),offset_px=0.)
                lines.append(raster);pars.append([record]);singles.append((raster,record))
    vertical=[p for p in singles if p[1]['angle_degrees']==90]
    other=[p for p in singles if p[1]['angle_degrees']!=90]
    for v,pv in vertical:
        for other_line,po in other:
            lines.append(1-(1-v)*(1-other_line));pars.append([pv,po])
    return np.asarray(lines,np.float32),pars


def model_bank(rec,fields,scale,phase,cfg=Config()):
    t,w,r=E.raster_variant(fields['target'],fields['target_weight'],rec['crop_center'],scale,phase)
    center=(r+phase[0],r+phase[1])
    # Inherit the observed swatch confidence, including its unresolved legend
    # connector row. No plot-derived uncertainty mask or new erased pixels.
    yy,xx=np.where(w>1.e-5);weight=w[yy,xx];target=t[yy,xx]
    energy=max(float((weight*target*target).sum()),1.e-7)
    lines,styles=line_styles(len(t),center,float(t.max()),cfg)
    flatlines=lines[:,yy,xx]
    counts=np.array([len(p) for p in styles],np.int8)
    composite=[];parameters=[];divisors=[]
    for g in cfg.gains:
        body=np.clip(g*target,0,1)
        composite.append(1-(1-body[None,:])*(1-flatlines))
        parameters.extend([dict(gain=g,lines=p,line_count=len(p),style_index=i) for i,p in enumerate(styles)])
        divisors.extend([g*g*energy]*len(styles))
    composite=np.concatenate(composite).astype(np.float32)
    counts_all=np.tile(counts,len(cfg.gains))
    return dict(target=t,weight_image=w,radius=r,yy=yy,xx=xx,weight=weight,energy=energy,
        models=composite,weighted_models=(composite*weight).T.copy(),
        model_norms=(composite**2*weight).sum(axis=1),divisors=np.asarray(divisors,np.float32),
        counts=counts_all,parameters=parameters,lines=flatlines,line_images=lines,
        line_weighted=(flatlines*weight).T.copy(),line_norms=(flatlines**2*weight).sum(axis=1),
        line_counts=counts,styles=styles,center=center)


def eligible(counts,mode):
    if mode=='one_line':return counts==1
    if mode=='two_lines':return counts==2
    if mode=='one_or_two':return counts>=1
    if mode=='best_composite':return np.ones(len(counts),bool)
    raise ValueError(mode)


def scores_for_patches(z,bank,cfg=Config()):
    z2=(z*z*bank['weight']).sum(axis=1)
    sse=np.maximum(z2[:,None]-2*(z@bank['weighted_models'])+bank['model_norms'][None,:],0)
    loss=sse/bank['divisors'][None,:]+cfg.complexity_per_line*bank['counts'][None,:]
    return loss


def score_series(ink,allowed,rec,fields,scale,cfg=Config()):
    halo=cv2.dilate(allowed.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)
    near=cv2.dilate((ink>.04).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&halo
    ys,xs=np.where(near)
    maps={mode:dict(score=np.full(ink.shape,np.inf,np.float32),dx=np.zeros_like(ink),dy=np.zeros_like(ink),
                    model_index=np.full(ink.shape,-1,np.int32)) for mode in MODES}
    for dx in (0.,.5):
        for dy in (0.,.5):
            bank=model_bank(rec,fields,scale,(dx,dy),cfg);r=bank['radius'];pad=np.pad(ink,r)
            choices={m:np.flatnonzero(eligible(bank['counts'],m)) for m in MODES}
            for j in range(0,len(xs),cfg.chunk):
                xx=xs[j:j+cfg.chunk];yy=ys[j:j+cfg.chunk]
                z=pad[yy[:,None]+bank['yy'],xx[:,None]+bank['xx']]
                values=scores_for_patches(z,bank,cfg)
                for mode in MODES:
                    ids=choices[mode];k=np.argmin(values[:,ids],axis=1);model=ids[k]
                    score=values[np.arange(len(xx)),model];m=maps[mode]
                    better=score<m['score'][yy,xx]
                    a=xx[better];b=yy[better];m['score'][b,a]=score[better]
                    m['dx'][b,a]=dx;m['dy'][b,a]=dy;m['model_index'][b,a]=model[better]
    return maps,len(xs)


def inspect_point(ink,rec,fields,p,mode,cfg=Config(),return_images=False):
    bank=model_bank(rec,fields,p['scale'],(p['x']-p['ix'],p['y']-p['iy']),cfg)
    r=bank['radius'];yy,xx=bank['yy'],bank['xx'];pad=np.pad(ink,r)
    z=pad[p['iy']+yy,p['ix']+xx];ids=np.flatnonzero(eligible(bank['counts'],mode))
    losses=scores_for_patches(z[None,:],bank,cfg)[0]
    best=int(ids[np.argmin(losses[ids])]);parameters=bank['parameters'][best]
    divisor=float(bank['divisors'][best]);penalty=cfg.complexity_per_line*parameters['line_count']
    raw=float(losses[best]-penalty)
    null_sse=np.maximum(float((z*z*bank['weight']).sum())-2*(z@bank['line_weighted'])+bank['line_norms'],0)
    null_losses=null_sse/divisor+cfg.complexity_per_line*bank['line_counts']
    # Give the line-only control the SAME allowable line-count family.
    null_ids=np.flatnonzero(eligible(bank['line_counts'],mode))
    null_best=int(null_ids[np.argmin(null_losses[null_ids])]);null=float(null_losses[null_best])
    target=bank['target'];weight=bank['weight_image'];g=parameters['gain']
    patch=pad[p['iy']:p['iy']+len(target),p['ix']:p['ix']+len(target)]
    core=(target>max(.03,.35*target.max()))&(weight>.2)
    cw=weight*target*target*core
    visible=float((cw*np.clip(patch/np.maximum(g*target,1.e-6),0,1)).sum()/max(float(cw.sum()),1.e-7))
    gain=null-float(losses[best])
    record=dict(passed=bool(gain>cfg.minimum_line_improvement and visible>=cfg.minimum_visible_ink),
       loss=float(losses[best]),raw_loss=raw,complexity_penalty=penalty,parameters=parameters,
       line_only_loss=null,line_only_parameters=bank['styles'][null_best],line_improvement=gain,
       visible_base_ink=visible,model_index=best,energy=bank['energy'])
    if not return_images:return record
    line=bank['line_images'][parameters['style_index']]
    composite=1-(1-np.clip(g*target,0,1))*(1-line)
    return record,dict(patch=patch,target=target,weight=weight,line=line,composite=composite,
            line_only=bank['line_images'][null_best],residual=weight*(patch-composite)**2/divisor)


@dataclass(frozen=True)
class RecoveryLimits:
    """Bound the optional rescue cost; the native detector is never disabled."""
    maximum_pixels: int = 4_000_000
    maximum_swatch_count: int = 10
    maximum_swatch_diameter: float = 32.
    maximum_centres: int = 40_000
    maximum_weighted_samples: int = 12_000_000


def allowed_centres(shape, plot, exclusions):
    allowed = np.zeros(shape, bool)
    x0, y0, x1, y1 = map(int, plot)
    allowed[y0:y1, x0:x1] = True
    for x0, y0, x1, y1 in exclusions:
        allowed[max(0,int(y0)):max(0,int(y1)), max(0,int(x0)):max(0,int(x1))] = False
    return allowed


def inside(x, y, plot, exclusions):
    return (plot[0] <= x < plot[2] and plot[1] <= y < plot[3]
            and not any(a <= x < c and b <= y < d for a,b,c,d in exclusions))


def calibrate_observed(ink, allowed, templates, cfg=E.Config()):
    """One image-only scale per source raster, using the same bounded anchor loss.

    Native idealized glyph sizes cannot be copied onto a different observed
    raster. Re-measure the observed model itself, never use reference points.
    Missing anchors contribute loss 1, not a free win for undersized models.
    """
    result = {}
    for key, (rec, fields) in templates.items():
        trials = []
        for scale in cfg.scales:
            loss, px, py = E.evaluate_scale(ink, rec, fields, scale, cfg)
            anchors = E.local_candidates(loss, px, py, rec['diameter']*scale, allowed, .85)
            anchors = anchors[:cfg.top_anchors]
            value = float(np.mean([p['loss'] for p in anchors]+[1.]*(cfg.top_anchors-len(anchors))))
            trials.append(dict(scale=scale, anchor_loss=value, anchors=anchors))
        best = min(trials, key=lambda r:r['anchor_loss'])
        result[key] = dict(scale=best['scale'], trials=trials,
                           status='measured' if best['anchors'] else 'no_anchors')
    return result


def compete_nearby(proposals, x_ids, rejected):
    """Other swatches participate in NMS even though only X will be emitted."""
    selected = []
    for p in sorted(proposals,key=lambda q:q['loss']):
        competitor = next((q for q in selected if np.hypot(p['x']-q['x'],p['y']-q['y'])
                           < .55*min(p['diameter'],q['diameter'])),None)
        if competitor is not None:
            if p['series_id'] in x_ids:
                rejected.append(dict(p,reason='nearby_better_swatch_proposal',
                    rival_series=competitor['series_id'],rival_x=competitor['x'],rival_y=competitor['y']))
            continue
        selected.append(p)
    return selected


def recover(image, plot, exclusions, native_templates, kept, series_indices,
            cfg=Config(), limits=RecoveryLimits(), log_fn=print):
    """Return new X points and an auditable report, without mutating inputs.

    All usable legend rasters receive the same composite family for fair
    identity competition. Only a native X-labelled swatch can emit a rescue.
    Other series are rivals, not new detections. Native active points win any
    spatial conflict. Colour-group adapters must not call this grayscale arm.
    """
    start = time.perf_counter()
    report = dict(version=VERSION, status='not_applicable', added=0,
                  x_ids=[t.key for t in native_templates if t.name=='x_marker'],
                  reference_points_read=False, source_pixels_modified=False,
                  config=asdict(cfg), limits=asdict(limits), rejected=[], accepted=[],
                  scale_policy='image_measured_observed_raster_per_swatch',
                  scope='BW X rescue only; native active points take precedence')
    def finish(status):
        report.update(status=status, seconds=time.perf_counter()-start)
        return [], report
    if not report['x_ids']:
        return finish('no_x_swatch')
    if image.shape[0]*image.shape[1] > limits.maximum_pixels or len(native_templates)>limits.maximum_swatch_count:
        return finish('resource_limit_image_or_swatch_count')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # This gate also protects an accidental direct call on a colour source.
    chroma = image.max(axis=2).astype(float)-image.min(axis=2)
    if np.count_nonzero(chroma>24) > .01*chroma.size:
        return finish('colour_source_not_supported')
    paper = float(np.percentile(gray,95))
    ink = np.clip((paper-gray)/255.,0,1)
    allowed = allowed_centres(gray.shape,plot,exclusions)
    templates = {}; extraction = {}
    for t in native_templates:
        a,b,c,d = map(int,t.swatch_box)
        crop = gray[b:d,a:c]
        if crop.ndim!=2 or min(crop.shape)<4:
            extraction[t.key] = dict(status='insufficient_source_resolution')
            report['extraction'] = extraction
            return finish('unresolved_or_oversized_rival')
        rec, fields = E.extract(crop)
        extraction[t.key] = rec
        if fields is None or rec['diameter']>limits.maximum_swatch_diameter:
            # Omitting a rival would artificially improve the X identity margin.
            report['extraction'] = extraction
            return finish('unresolved_or_oversized_rival')
        templates[t.key] = (rec,fields)
    report['extraction'] = extraction
    halo = cv2.dilate(allowed.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)
    near = cv2.dilate((ink>.04).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&halo
    n = int(near.sum()); report['candidate_centres'] = n
    if not n:
        return finish('no_plot_ink')
    if n>limits.maximum_centres:
        return finish('resource_limit_centres')
    calibration = calibrate_observed(ink,allowed,templates)
    report['calibration'] = calibration
    scales = {k:v['scale'] for k,v in calibration.items()}
    # Estimate dot-product work before launching the dense bank. Four phases
    # and the fixed line bank are common factors in this bound.
    samples = sum(n*np.count_nonzero(E.raster_variant(f['target'],f['target_weight'],
                rec['crop_center'],scales[k],(0,0))[1]>1.e-5)
                  for k,(rec,f) in templates.items())
    report['weighted_samples'] = int(samples)
    if samples>limits.maximum_weighted_samples:
        return finish('resource_limit_composite_work')
    maps = {}
    for key,(rec,fields) in templates.items():
        log_fn(f'[v46 centered X] scoring observed rival {key}, shared scale={scales[key]:g}')
        maps[key] = score_series(ink,allowed,rec,fields,scales[key],cfg)[0]['one_or_two']
    proposals = []
    # Keep the experiment's neighbourhood competition as well as its
    # same-centre margin. Otherwise an X a few pixels off a better triangle
    # centre could survive merely because only X proposals were emitted.
    for key in templates:
        rec,fields = templates[key]; m = maps[key]; diameter = rec['diameter']*scales[key]
        for p in E.local_candidates(m['score'],m['dx'],m['dy'],diameter,allowed,cfg.maximum_loss):
            if not inside(p['x'],p['y'],plot,exclusions):
                continue  # Fractional phases also obey exclusive ROI bounds.
            p.update(series_id=key,scale=scales[key],diameter=diameter)
            rivals = [(float(v['score'][p['iy'],p['ix']]),k) for k,v in maps.items() if k!=key]
            rival_loss,rival = min(rivals) if rivals else (float('inf'),None)
            p.update(rival=rival,rival_loss=rival_loss if np.isfinite(rival_loss) else None,
                     identity_margin=rival_loss-p['loss'] if np.isfinite(rival_loss) else None)
            if rival_loss-p['loss'] < cfg.minimum_identity_margin:
                if key in report['x_ids']:
                    report['rejected'].append(dict(p,reason='ambiguous_swatch_identity'))
                continue
            proposals.append(p)
    selected = compete_nearby(proposals,report['x_ids'],report['rejected'])
    proposals = []
    for p in selected:
        key = p['series_id']
        if key in report['x_ids'] and calibration[key]['status']!='no_anchors':
            rec,fields = templates[key]
            v = inspect_point(ink,rec,fields,p,'one_or_two',cfg)
            p['verification'] = v
            if not v['passed']:
                report['rejected'].append(dict(p,reason='line_only_or_insufficient_base_ink'))
                continue
            proposals.append(p)
    from bw_suppressed_v46 import encode_marker_mask
    by_key = {t.key:t for t in native_templates}
    additions = []
    for p in sorted(proposals,key=lambda q:q['loss']):
        conflict = next((q for q in [*kept,*additions]
            if np.hypot(p['x']-q['cx'],p['y']-q['cy']) < .55*min(p['diameter'],
                    float(q.get('effective_diameter',p['diameter'])))),None)
        if conflict is not None:
            report['rejected'].append(dict(p,reason='native_active_or_added_point_precedence',
                                           conflict_point_id=conflict.get('point_id')))
            continue
        key = p['series_id']; t = by_key[key]; rec,f = templates[key]
        # Export the observed marker body only. Added long lines must never
        # become the Step-5 marker footprint. Use centre-aligned phase zero.
        target,weight,r = E.raster_variant(f['target'],f['target_weight'],rec['crop_center'],p['scale'],(0,0))
        mask = (target>max(.03,.35*target.max()))&(weight>.2)
        point_id = f'CX{len(additions)+1:03d}'
        existing_ids = {q.get('point_id') for q in [*kept,*additions]}
        while point_id in existing_ids:
            point_id += '_x'
        q = dict(class_name='x_marker',shape_hint='x_marker',template=key,swatch_id=t.swatch_id,
            class_idx=series_indices[key],shape_idx=10,cx=p['x'],cy=p['y'],
            confidence=p['verification']['visible_base_ink'],confidence_kind='observed_base_ink_coverage',
            source=VERSION,point_id=point_id,original_detection=True,
            marker_mask=encode_marker_mask(mask),marker_offset_x=0.,marker_offset_y=0.,
            marker_scale=p['diameter']/float(t.diameter),marker_aspect=1.,source_diameter=float(t.diameter),
            observed_raster_scale=p['scale'],observed_raster_diameter=rec['diameter'],
            effective_diameter=p['diameter'],centered_composite=p)
        additions.append(q); report['accepted'].append(p)
    report.update(status='completed',added=len(additions),seconds=time.perf_counter()-start)
    return additions,report

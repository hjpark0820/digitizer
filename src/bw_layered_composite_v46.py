"""Observed marker + zero/one/two centered lines with per-line z order.

Runtime matching core. No shape catalog, point truth or dataset inputs.
Source legend-line ambiguity is local to pixels; scoring and localization
weights are distinct. Opaque white interiors come from enclosed SOURCE holes,
not an ideal circle/triangle or a hole fabricated in the target plot.
"""
from dataclasses import dataclass, asdict
import time
import cv2,numpy as np
from scipy.ndimage import binary_fill_holes
import bw_observed_raster_v46 as R
import bw_centered_composite_v46 as X

VERSION='bw-layered-centered-lines-v1'
MODES=('marker_only','front_only','layered')

class ResourceLimit(RuntimeError):
    """Optional recovery cannot complete within its declared budget."""



@dataclass(frozen=True)
class Config:
    angles:tuple=tuple(range(0,180,15))
    widths:tuple=(.8,1.6)
    amplitudes:tuple=(.4,.8,1.2)
    gains:tuple=(.85,1.,1.15)
    complexity_per_line:float=.02
    boundary_weight:float=.35
    maximum_loss:float=.65
    minimum_line_improvement:float=.06
    minimum_independent_fraction:float=.20
    minimum_independent_visible:float=.70
    minimum_identity_margin:float=.035
    chunk:int=96
    maximum_centres:int=12000


def prepare_source(rec,fields):
    """Condition observed ink on a measured legend connector, without erasing
    identifiable marker ink. Exactly unidentifiable line/marker pixels retain
    an uncertainty flag and small weight, NOT a claim that they are white.
    """
    o=fields['observed'].astype(np.float32);h,w=o.shape
    side=max(1,int(.20*w));flanks=np.r_[np.arange(side),np.arange(w-side,w)]
    profile=np.quantile(o[:,flanks],.75,axis=1).astype(np.float32)
    nuisance=fields['line_uncertainty'][:,0]
    profile=np.where(nuisance>.01,profile,0.)
    line=np.broadcast_to(profile[:,None],o.shape).copy()
    x0,y0,x1,y1=rec['bbox'];yy,xx=np.indices(o.shape)
    body=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    dx=np.maximum.reduce([x0-xx,xx-(x1-1),np.zeros_like(xx)])
    dy=np.maximum.reduce([y0-yy,yy-(y1-1),np.zeros_like(yy)])
    spatial=np.clip(1-np.maximum(dx,dy)/2.,0,1).astype(np.float32)
    # A source line weaker than the observation leaves recoverable marker
    # contrast. A darker source-line prediction than observed paper indicates
    # the marker's opaque interior could have hidden the source line.
    front=line>.025
    hidden=front&(o<line-.08)&body
    uncertain=front&body&((np.abs(o-line)<.08)|(line>.90))&~hidden
    marker=np.where(front&~hidden,np.clip((o-line)/np.maximum(1-line,.08),0,1),o)
    marker[front&~body]=0.
    confidence=spatial*np.where(uncertain,.06,1.)
    peak=max(float(o[body].max()),.05)
    # Enclosed low-ink islands of the original source glyph determine whether
    # a white opaque interior is supported. Never hole-fill target observations.
    core=(o>.28*peak)&body
    enclosed=binary_fill_holes(np.pad(core,1))[1:-1,1:-1]
    holes=enclosed&~core&body
    white_holes=holes&(o<.45*peak)
    opaque=np.maximum(np.clip(marker/peak,0,1),white_holes.astype(np.float32))
    # The source line can divide an open interior into multiple observed holes.
    # Keep those actual islands; do not invent the missing bridge through it.
    a,b,c,d=rec['crop']
    prepared=dict(marker=marker[b:d,a:c].astype(np.float32),weight=confidence[b:d,a:c].astype(np.float32),
                  cover=opaque[b:d,a:c].astype(np.float32),spatial=spatial[b:d,a:c].copy(),
                  uncertain=uncertain[b:d,a:c].copy(),source_line=line[b:d,a:c].copy(),
                  source_observed=o[b:d,a:c].copy(),source_gray=fields['source_gray'].copy(),
                  holes=white_holes[b:d,a:c].copy(),localization_weight=fields['target_weight'].copy())
    meta=dict(source_open=bool(white_holes.any()),source_hole_pixels=int(white_holes.sum()),
              uncertain_body_pixels=int(uncertain.sum()),source_line_profile=profile.tolist(),
              source_pixels_modified=False,shape_class=None,
              note='Conditional source marker estimate; unresolved pixels are NOT observed white.')
    return prepared,meta


def variants(rec,fields,scale,phase):
    marker,weight,r=R.raster_variant(fields['marker'],fields['weight'],rec['crop_center'],scale,phase)
    cover,_,r2=R.raster_variant(fields['cover'],np.ones_like(fields['cover']),rec['crop_center'],scale,phase)
    assert r==r2
    return marker,weight,np.clip(cover,0,1),r


def compose(marker,cover,lines,orders):
    """Gray ink alpha composition; 'behind' is attenuated by marker coverage."""
    behind=np.zeros_like(marker);front=np.zeros_like(marker)
    for line,order in zip(lines,orders):
        if order=='behind':behind=1-(1-behind)*(1-line)
        elif order=='front':front=1-(1-front)*(1-line)
        else:raise ValueError('Unknown drawing order')
    base=np.clip(marker+(1-cover)*behind,0,1)
    return 1-(1-base)*(1-front)


def feature_matrix(images,weight,marker,cfg=Config()):
    """One symmetric gray+signed-edge objective for models AND center search."""
    single=images.ndim==2
    ims=images[None] if single else images
    energy=max(float(np.sum(weight*marker**2)),1.e-7)
    ew0=np.minimum(weight[1:],weight[:-1]);ew1=np.minimum(weight[:,1:],weight[:,:-1])
    edge_energy=max(float(np.sum(ew0*np.diff(marker,axis=0)**2)+np.sum(ew1*np.diff(marker,axis=1)**2)),1.e-7)
    f0=(ims*np.sqrt(weight/energy)).reshape(len(ims),-1)
    f1=(np.diff(ims,axis=1)*np.sqrt(cfg.boundary_weight*ew0/edge_energy)).reshape(len(ims),-1)
    f2=(np.diff(ims,axis=2)*np.sqrt(cfg.boundary_weight*ew1/edge_energy)).reshape(len(ims),-1)
    out=np.concatenate((f0,f1,f2),axis=1).astype(np.float32)
    return out[0] if single else out


def bank(rec,fields,scale,phase,allow_back,cfg=Config()):
    marker,weight,cover,r=variants(rec,fields,scale,phase)
    center=(r+phase[0],r+phase[1]);size=len(marker);peak=float(marker.max())
    singles=[]
    for angle in cfg.angles:
        for width in cfg.widths:
            for rel in cfg.amplitudes:
                line=np.clip(peak*rel,0,1)*X.stroke(size,center,angle,width)
                singles.append((line,dict(angle=int(angle),width=float(width),amplitude=float(np.clip(peak*rel,0,1)))))
    families=[([],[])] + [([line],[p]) for line,p in singles]
    vertical=[q for q in singles if q[1]['angle']==90]
    other=[q for q in singles if q[1]['angle']!=90]
    families += [([a,b],[pa,pb]) for a,pa in vertical for b,pb in other]
    models=[];parameters=[];full_lines=[];nulls=[];null_params=[]
    for lines,pars in families:
        full=np.zeros_like(marker)
        for line in lines:full=1-(1-full)*(1-line)
        nulls.append(full);null_params.append(pars)
        if not lines:orders=[()]
        elif not allow_back:orders=[tuple('front' for _ in lines)]
        elif len(lines)==1:orders=[('front',),('behind',)]
        else:orders=[('front','front'),('front','behind'),('behind','front'),('behind','behind')]
        for order in orders:
            for gain in cfg.gains:
                body=np.clip(gain*marker,0,1)
                # Gray foreground ink cannot exceed alpha coverage.
                matte=np.maximum(cover,body)
                models.append(compose(body,matte,lines,order));full_lines.append(full)
                parameters.append(dict(gain=gain,line_count=len(lines),lines=[dict(p,order=o) for p,o in zip(pars,order)],
                                       has_behind='behind' in order))
    models=np.asarray(models,np.float32);nulls=np.asarray(nulls,np.float32)
    features=feature_matrix(models,weight,marker,cfg);nf=feature_matrix(nulls,weight,marker,cfg)
    return dict(marker=marker,weight=weight,cover=cover,radius=r,center=center,models=models,
                features=features,norms=(features*features).sum(axis=1),parameters=parameters,
                full_lines=np.asarray(full_lines,np.float32),nulls=nulls,null_features=nf,
                null_norms=(nf*nf).sum(axis=1),null_parameters=null_params,
                gains=np.array([p['gain'] for p in parameters],np.float32),
                counts=np.array([p['line_count'] for p in parameters],np.float32),
                null_counts=np.array([len(p) for p in null_params],np.float32))


def candidates_for_image(ink,plot,legend,templates,cfg=Config(),exclusions=None):
    """Union of all source glyph proposals, expanded before composite scoring.
    No reference or old acceptance gate is used. This is a bounded candidate
    experiment, not an exhaustive scan of every background pixel.
    """
    allowed=np.zeros(ink.shape,bool);a,b,c,d=plot;allowed[b:d,a:c]=True
    for a,b,c,d in (exclusions if exclusions is not None else [legend]):
        allowed[max(0,b):max(0,d),max(0,a):max(0,c)]=False
    seed=np.zeros(ink.shape,np.uint8)
    for key,(rec,fields) in templates.items():
        for scale in R.Config().scales:
            loss,px,py=R.evaluate_scale(ink,rec,fields,scale,R.Config())
            pp=R.local_candidates(loss,px,py,rec['diameter']*scale,allowed,.85,cap=250)
            for p in pp:seed[p['iy'],p['ix']]=1
    mask=cv2.dilate(seed,np.ones((3,3),np.uint8)).astype(bool)&allowed
    ys,xs=np.where(mask)
    if len(xs)>cfg.maximum_centres:raise ResourceLimit('candidate_centres')
    return xs,ys,allowed,mask


def score_bank(ink,xs,ys,b,mode,cfg=Config()):
    pars=b['parameters']
    if mode=='marker_only':ids=np.array([i for i,p in enumerate(pars) if p['line_count']==0])
    elif mode=='front_only':ids=np.array([i for i,p in enumerate(pars) if not p['has_behind']])
    else:ids=np.arange(len(pars))
    null_ids=np.flatnonzero(b['null_counts']==0) if mode=='marker_only' else np.arange(len(b['null_counts']))
    r=b['radius'];pad=np.pad(ink,r);size=len(b['marker']);dy,dx=np.indices((size,size))
    outputs=[]
    for start in range(0,len(xs),cfg.chunk):
        xx,yy=xs[start:start+cfg.chunk],ys[start:start+cfg.chunk]
        z=pad[yy[:,None,None]+dy,xx[:,None,None]+dx]
        f=feature_matrix(z,b['weight'],b['marker'],cfg);z2=(f*f).sum(axis=1)
        sse=np.maximum(z2[:,None]-2*f@b['features'][ids].T+b['norms'][ids],0)
        losses=sse/b['gains'][ids]**2+cfg.complexity_per_line*b['counts'][ids]
        jj=np.argmin(losses,axis=1);chosen=ids[jj];best=losses[np.arange(len(xx)),jj]
        ns=np.maximum(z2[:,None]-2*f@b['null_features'][null_ids].T+b['null_norms'][null_ids],0)
        nl=ns/b['gains'][chosen,None]**2+cfg.complexity_per_line*b['null_counts'][null_ids]
        ni=np.argmin(nl,axis=1);null_index=null_ids[ni];improvement=nl[np.arange(len(xx)),ni]-best
        # Lines cannot supply the positive marker evidence. Use evidence
        # outside ALL proposed full lines, regardless of their z order.
        lines=b['full_lines'][chosen];target=b['marker'][None];gain=b['gains'][chosen,None,None]
        mass=b['weight']*target**2
        independent=mass*(1-lines)**2
        independent_fraction=independent.sum(axis=(1,2))/max(float(mass.sum()),1.e-9)
        remaining=np.clip((z-lines)/np.maximum(1-lines,.02),0,1)
        vis=(independent*np.clip(remaining/np.maximum(gain*target,1.e-6),0,1)).sum(axis=(1,2))/np.maximum(independent.sum(axis=(1,2)),1.e-9)
        good=(best<=cfg.maximum_loss)&(improvement>cfg.minimum_line_improvement)&(independent_fraction>=cfg.minimum_independent_fraction)&(vis>=cfg.minimum_independent_visible)
        for k in range(len(xx)):
            outputs.append(dict(ix=int(xx[k]),iy=int(yy[k]),loss=float(best[k]),passed=bool(good[k]),
                                model_index=int(chosen[k]),null_index=int(null_index[k]),line_improvement=float(improvement[k]),
                                independent_fraction=float(independent_fraction[k]),independent_visible=float(vis[k]),
                                parameters=pars[chosen[k]],null_parameters=b['null_parameters'][null_index[k]]))
    return outputs


def apply(image,plot,legend,templates,scales,cfg=Config(),progress=print,*,
          modes=MODES,exclusions=None,deadline=None,maximum_work=250_000_000_000,
          maximum_bank_bytes=256_000_000):
    start=time.perf_counter();gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink=np.clip((np.percentile(gray,95)-gray)/255,0,1)
    prepared={};meta={}
    for key,(rec,fields) in templates.items():prepared[key],meta[key]=prepare_source(rec,fields)
    xs,ys,allowed,coverage=candidates_for_image(ink,plot,legend,templates,cfg,exclusions)
    progress(f'Composite candidate centres: {len(xs)}',flush=True)
    all_modes={m:dict(series={},points=[],suppressed=[]) for m in modes}
    work=0
    for key,(rec,f) in templates.items():
        n=variants(rec,prepared[key],scales[key],(0.,0.))[0].size
        single=len(cfg.angles)*len(cfg.widths)*len(cfg.amplitudes)
        vert=sum(a==90 for a in cfg.angles)*len(cfg.widths)*len(cfg.amplitudes)
        pairs=vert*(single-vert)
        count=len(cfg.gains)*(1+single*(2 if meta[key]["source_open"] else 1)+pairs*(4 if meta[key]["source_open"] else 1))
        # Dense model, feature, full-line banks plus construction overhead.
        if count*n*32>maximum_bank_bytes:raise ResourceLimit("model_bank_memory")
        work+=len(xs)*count*n*3*4*len(modes)
    if work>maximum_work:raise ResourceLimit("composite_work")
    bank_counts={}
    for key,(rec,fields) in templates.items():
        scale=scales[key];series_start=time.perf_counter()
        best={m:{} for m in modes};raw_best={m:{} for m in modes}
        for dx,dy in ((0.,0.),(.5,0.),(0.,.5),(.5,.5)):
            if deadline is not None and time.perf_counter()>deadline:raise ResourceLimit("elapsed_time")
            b=bank(rec,prepared[key],scale,(dx,dy),meta[key]['source_open'],cfg)
            bank_counts[key]=len(b['parameters'])
            for mode in modes:
                for p in score_bank(ink,xs,ys,b,mode,cfg):
                    p.update(series_id=key,x=p['ix']+dx,y=p['iy']+dy,scale=scale,diameter=rec['diameter']*scale,phase=[dx,dy])
                    if not X.inside(p['x'],p['y'],plot,exclusions if exclusions is not None else [legend]):continue
                    index=(p['ix'],p['iy'])
                    if index not in raw_best[mode] or p['loss']<raw_best[mode][index]['loss']:raw_best[mode][index]=p
                    if p['passed'] and (index not in best[mode] or p['loss']<best[mode][index]['loss']):best[mode][index]=p
            progress(f'{key} phase {dx},{dy}: {time.perf_counter()-series_start:.2f}s',flush=True)
        for mode in modes:
            points=sorted(best[mode].values(),key=lambda p:p['loss']);selected=[]
            for p in points:
                if all(np.hypot(p['x']-q['x'],p['y']-q['y'])>.55*p['diameter'] for q in selected):selected.append(p)
            all_modes[mode]['series'][key]=dict(scale=scale,candidates=selected,raw_best=list(raw_best[mode].values()),
                                                passed_centres=len(points))
        progress(f'{key} completed',flush=True)
    for mode,arm in all_modes.items():
        candidates=[p for s in arm['series'].values() for p in s['candidates']]
        for p in sorted(candidates,key=lambda p:p['loss']):
            rivals=[q for q in candidates if q['series_id']!=p['series_id'] and np.hypot(p['x']-q['x'],p['y']-q['y'])<.4*min(p['diameter'],q['diameter'])]
            rival=min(rivals,key=lambda q:q['loss']) if rivals else None
            if rival and rival['loss']-p['loss']<cfg.minimum_identity_margin:
                arm['suppressed'].append(dict(p,reason='ambiguous_marker_identity',rival=rival['series_id']));continue
            if any(np.hypot(p['x']-q['x'],p['y']-q['y'])<.55*min(p['diameter'],q['diameter']) for q in arm['points']):
                arm['suppressed'].append(dict(p,reason='same_location_competition'));continue
            arm['points'].append(p)
    return dict(version=VERSION,modes=all_modes,source_meta=meta,config=asdict(cfg),scales=scales,
                bank_counts=bank_counts,candidate_centres=len(xs),seconds=time.perf_counter()-start,
                scale_policy='Caller-supplied per-observed-swatch scales',
                operational_defaults_modified=False,reference_points_read=False,step5=False),prepared,coverage


@dataclass(frozen=True)
class RecoveryLimits:
    maximum_pixels:int=1_000_000
    maximum_swatch_count:int=10
    maximum_swatch_diameter:float=24.
    maximum_seconds:float=60.
    maximum_work:int=250_000_000_000
    maximum_bank_bytes:int=256_000_000


def recover(image,plot,exclusions,native_templates,kept,series_indices,
            cfg=Config(),limits=RecoveryLimits(),log_fn=print,*,legend_reports=None):
    """Add observed-body rescues to native BW points; never replace native points.

    Unlike the X-only stage this allows all non-X marker swatches to emit.
    All usable source swatches still compete. Native X recovery stays separate.
    No manually supplied scale, shape-generated pixels or saved experiment is
    read. A missing scale/rival abstains; the original detector still returns.
    """
    start=time.perf_counter()
    report=dict(version=VERSION,status='not_applicable',added=0,seconds=0.,
                accepted=[],rejected=[],config=asdict(cfg),limits=asdict(limits),
                reference_points_read=False,source_pixels_modified=False,
                scale_policy='verified_observed_boundary_same_site_scale_profiles',
                native_points_preserved=True,scope='BW-only additive non-X recovery')
    def finish(status):
        report.update(status=status,seconds=time.perf_counter()-start)
        return [],report
    eligible={t.key for t in native_templates if t.name!='x_marker'}
    if not eligible:return finish('no_non_x_swatch')
    if image.shape[0]*image.shape[1]>limits.maximum_pixels or len(native_templates)>limits.maximum_swatch_count:
        return finish('resource_limit_image_or_swatch_count')
    chroma=image.max(axis=2).astype(float)-image.min(axis=2)
    if np.count_nonzero(chroma>24)>.01*chroma.size:return finish('colour_source_not_supported')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    templates={};extraction={}
    legend_by_id={r['swatch_id']:r for r in (legend_reports or [])}
    for t in native_templates:
        a,b,c,d=map(int,t.swatch_box)
        crop=gray[max(0,b):max(0,d),max(0,a):max(0,c)]
        if crop.ndim!=2 or min(crop.shape)<4:
            report['extraction']=extraction
            return finish('unresolved_or_oversized_rival')
        rec,f=R.extract(crop,connected_line=legend_by_id.get(t.key,{}).get('connected_line'));extraction[t.key]=rec
        if f is None or rec['diameter']>limits.maximum_swatch_diameter:
            report['extraction']=extraction
            return finish('unresolved_or_oversized_rival')
        templates[t.key]=(rec,f)
    report['extraction']=extraction
    from bw_observed_scale_v46 import apply as calibrate
    # The scale module reads only this image, ROI and observed source rasters.
    calibration,_=calibrate(image,plot,(0,0,0,0),templates,exclusions=exclusions)
    report['calibration']=calibration['series']
    scales={k:s['scale'] for k,s in calibration['series'].items()}
    if any(s is None for s in scales.values()):return finish('insufficient_verified_scale_anchors')
    report['scales']=scales
    try:
        result,prepared,_=apply(image,plot,(0,0,0,0),templates,scales,cfg,
            progress=lambda message,**kw:log_fn('[v46 layered] '+message),
            modes=('layered',),exclusions=exclusions,deadline=start+limits.maximum_seconds,
            maximum_work=limits.maximum_work,maximum_bank_bytes=limits.maximum_bank_bytes)
    except ResourceLimit as exc:
        report['limit_reason']=str(exc)
        return finish('resource_limit')
    arm=result['modes']['layered']
    report.update(candidate_centres=result['candidate_centres'],bank_counts=result['bank_counts'],
                  source_meta=result['source_meta'],
                  candidates_by_series={k:dict(scale=s['scale'],passed_centres=s['passed_centres'],
                                             candidates=s['candidates']) for k,s in arm['series'].items()})
    report['rejected'].extend(arm['suppressed'])
    additions=[];by_key={t.key:t for t in native_templates}
    from bw_suppressed_v46 import encode_marker_mask
    classes=['filled_circle','open_circle','filled_square','open_square',
             'open_triangle','open_inv_triangle','filled_triangle','filled_inv_triangle',
             'open_rhombus','filled_rhombus','x_marker','plus_marker','unknown_marker']
    existing_ids={q.get('point_id') for q in kept}
    for p in arm['points']:
        key=p['series_id']
        if key not in eligible:continue
        conflict=next((q for q in [*kept,*additions] if np.hypot(p['x']-q['cx'],p['y']-q['cy'])
                    <.55*min(p['diameter'],float(q.get('effective_diameter',p['diameter'])))),None)
        if conflict is not None:
            report['rejected'].append(dict(p,reason='native_active_or_added_point_precedence',
                                          conflict_point_id=conflict.get('point_id')))
            continue
        t=by_key[key];rec=templates[key][0]
        target,weight,cover,r=variants(rec,prepared[key],p['scale'],(0.,0.))
        # Only the observed marker ink goes to Step 5, never the long lines
        # or opaque white-interior coverage mask.
        mask=(target>max(.03,.35*target.max()))&(weight>.2)
        if not mask.any():
            report['rejected'].append(dict(p,reason='empty_marker_footprint'));continue
        point_id=f'LC{len(additions)+1:03d}'
        while point_id in existing_ids:point_id+='_r'
        existing_ids.add(point_id)
        q=dict(class_name=t.name,shape_hint=t.shape_hint or t.name,template=key,swatch_id=t.swatch_id,
            class_idx=series_indices[key],shape_idx=classes.index(t.name),cx=p['x'],cy=p['y'],
            confidence=p['independent_visible'],confidence_kind='observed_independent_ink_coverage',
            source=VERSION,point_id=point_id,original_detection=True,
            marker_mask=encode_marker_mask(mask),marker_offset_x=0.,marker_offset_y=0.,
            marker_scale=p['diameter']/float(t.diameter),marker_aspect=1.,source_diameter=float(t.diameter),
            observed_raster_scale=p['scale'],observed_raster_diameter=rec['diameter'],
            effective_diameter=p['diameter'],layered_composite=p)
        additions.append(q);report['accepted'].append(p)
    report.update(status='completed',added=len(additions),seconds=time.perf_counter()-start)
    return additions,report

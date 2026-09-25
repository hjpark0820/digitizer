"""Image-only, per-legend-identity size calibration before colour grid voting.

Native colour membership locates potential anchors across 33 scales. Complete
window evidence rejects line/occlusion dominated anchors. Independent x sites
then choose ONE working scale; candidate-specific scaling is not enabled.
No labels, saved detections, manual centres or expected point counts are inputs.
"""
from copy import deepcopy
from time import perf_counter
import math

import cv2
import numpy as np
from color_hollow_scale_v46 import CONFIG as HOLLOW_CONFIG, is_hollow, outline_evidence, anchor_passes

VERSION = 'colour_shared_symbol_scale_v6_hollow_geometry'
SCALES = tuple(round(.50+.025*i, 3) for i in range(33))
CONFIG = dict(minimum_anchors=3, maximum_anchors=7, maximum_candidates=64,
              minimum_correlation=.55, minimum_window_score=.35,
              maximum_missing=.25, maximum_extra=.40, minimum_visible=.85,
              consensus_fraction=.70, scale_agreement=.075, minimum_gain=.01)


def scaled_template(template, scale):
    """One centre-preserving transform for ALL template raster fields.

    Original legend boxes, source centre and colour model remain source facts.
    Composition support/uncertainty and hollow interiors use the same transform.
    This does not alter brightness or fit a separate horizontal/vertical size.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Symbol scale must be positive and finite')
    result = deepcopy(template)
    if abs(scale-1.) < 1e-12:
        return result
    h,w = template['soft'].shape
    cx,cy = template.get('center',[(w-1)/2,(h-1)/2])
    rx,ry = math.ceil(max(cx,w-1-cx)*scale),math.ceil(max(cy,h-1-cy)*scale)
    size=(max(3,2*rx+1),max(3,2*ry+1))
    center=[(size[0]-1)/2,(size[1]-1)/2]
    matrix=np.float32([[scale,0,center[0]-scale*cx],[0,scale,center[1]-scale*cy]])
    for key,value in template.items():
        if not isinstance(value,np.ndarray) or value.ndim<2 or value.shape[:2]!=(h,w):
            continue
        boolean=value.dtype==bool
        border=tuple(float(v) for v in template['model']['paper_bgr']) if key=='raw_bgr' else 0
        # Frozen JSON templates may restore raster integers as int64, which
        # OpenCV cannot warp. Converting those values to float32 is lossless
        # for 8-bit image pixels and does not mutate the source template.
        raster=value.astype(np.uint8) if boolean else value
        if raster.dtype in (np.dtype('int64'),np.dtype('uint64'),np.dtype('int32')):
            raster=raster.astype(np.float32)
        transformed=cv2.warpAffine(raster,matrix,size,
            flags=cv2.INTER_NEAREST if boolean else cv2.INTER_LINEAR,borderValue=border)
        result[key]=transformed>0 if boolean else transformed
    if 'hole_core' in result:
        result['hole_core'] &= result['face'] & ~result['boundary_uncertain'] & (result['soft']<.12)
    result['diameter']=float(template['diameter'])*scale
    result['center']=center
    if 'center_convention' in result:
        result['center_convention']['template_geometric_center']=center.copy()
        result['center_convention']['working_symbol_scale']=float(scale)
    return result


def _match_map(observed, template, allowed):
    """Weighted colour-strength correlation, including required paper holes."""
    soft=np.asarray(template['soft'],np.float32)
    support=soft>.04
    output=np.full(observed.shape,-1,np.float32)
    if support.sum()<5:return output
    yy,xx=np.nonzero(support)
    a,b=max(0,int(xx.min())-3),max(0,int(yy.min())-3)
    c,d=min(soft.shape[1],int(xx.max())+4),min(soft.shape[0],int(yy.max())+4)
    model=soft[b:d,a:c]
    if model.shape[0]>observed.shape[0] or model.shape[1]>observed.shape[1]:return output
    weight=np.ones(model.shape,np.float32)
    nuisance=np.asarray(template.get('nuisance',np.zeros_like(soft)),bool)
    connector=np.asarray(template.get('central_connector',np.zeros_like(soft)),bool)
    weight[(nuisance|connector)[b:d,a:c]]=.15
    model=cv2.GaussianBlur(model,(3,3),.45)
    total=float(weight.sum());mean=float((model*weight).sum()/total)
    centered=weight*(model-mean);variance=float((weight*(model-mean)**2).sum())
    if variance<1e-6:return output
    one=cv2.matchTemplate(observed,weight,cv2.TM_CCORR)
    two=cv2.matchTemplate(observed**2,weight,cv2.TM_CCORR)
    numerator=cv2.matchTemplate(observed,centered,cv2.TM_CCORR)
    score=np.clip(numerator/np.sqrt(np.maximum(two-one*one/total,1e-6)*variance),-1,1)
    coverage=cv2.matchTemplate(allowed.astype(np.float32),np.ones(model.shape,np.float32),cv2.TM_CCORR)
    score[coverage<model.size-.1]=-1
    cx,cy=template['center'];ox,oy=round(cx-a),round(cy-b)
    output[oy:oy+score.shape[0],ox:ox+score.shape[1]]=score
    return output


def _window(evidence,index,template,x,y):
    from color_marker_window_v2 import verify_window
    from color_blend_uncertainty_v46 import window_kwargs
    guide=None if 'guide_membership' not in evidence else tuple(evidence[k][index] for k in ('guide_membership','guide_lower','guide_upper'))
    return verify_window(evidence['membership'][index],evidence['other'][index],template,x,y,
        ignore_mask=evidence.get('ignore_mask'),guide_model=guide,**window_kwargs(evidence,index))


def _clean_anchor(window, correlation, outline=None):
    clean=(correlation>=CONFIG['minimum_correlation'] and not window['line_only_reject']
           and window['score']>=CONFIG['minimum_window_score']
           and window['missing']<=CONFIG['maximum_missing'] and window['extra']<=CONFIG['maximum_extra']
           and window['observed_fraction']>=CONFIG['minimum_visible'] and window['ignored_marker_fraction']<=.05)
    return bool(clean or (outline is not None and anchor_passes(window,correlation,outline)))


def calibrate(evidence, *, _hollow_geometry=False, _indices=None):
    """Return JSON diagnostics; anchors are calibration evidence, not points."""
    started=perf_counter();reports={}
    valid=np.asarray(evidence['valid'],bool)&(np.asarray(evidence.get('ignore_mask',0))<.1)
    for index,template in enumerate(evidence['templates']):
        if _indices is not None and index not in _indices:
            continue
        tick=perf_counter();diameter=float(template['diameter'])
        own=evidence['membership'][index]
        observed=np.maximum(own-evidence['guide_upper'][index],0) if 'guide_upper' in evidence else own
        observed=cv2.GaussianBlur(observed.astype(np.float32),(3,3),.45)
        variants=[scaled_template(template,s) for s in SCALES]
        if _hollow_geometry:
            from color_hollow_geometry_scale_v46 import render_scaled
            variants=[render_scaled(template,v,s) for v,s in zip(variants,SCALES)]

        def inspect_window(variant, x, y):
            # Integer centres are a large fraction of a small hollow rim.
            # Refine within half a pixel, then retain THAT centre for all
            # subsequent outline checks and shared-size validation.
            offsets=(-.5,0.,.5) if _hollow_geometry and variant['diameter']<12 else (0.,)
            choices=[(_window(evidence,index,variant,x+dx,y+dy),x+dx,y+dy)
                     for dx in offsets for dy in offsets]
            return max(choices,key=lambda q:q[0]['score'])
        retain=observed.size*len(SCALES)<=16_000_000
        stack=[];best=np.full(observed.shape,-1,np.float32)
        for variant in variants:
            plane=_match_map(observed,variant,valid)
            np.maximum(best,plane,out=best)
            if retain:stack.append(plane)
        local=cv2.dilate(best,np.ones((max(3,round(.5*diameter))|1,)*2,np.uint8))
        yy,xx=np.nonzero((best>=CONFIG['minimum_correlation'])&(best>=local-1e-6))
        candidates=[]
        for pos in sorted(zip(xx,yy),key=lambda xy:-float(best[xy[1],xy[0]])):
            if any(np.hypot(pos[0]-q[0],pos[1]-q[1])<.85*diameter for q in candidates):continue
            candidates.append((int(pos[0]),int(pos[1])))
            if len(candidates)>=CONFIG['maximum_candidates']:break
        profiles=np.full((len(candidates),len(SCALES)),-1,np.float32)
        centres=np.zeros((len(candidates),len(SCALES),2),int)
        for j,variant in enumerate(variants):
            if not candidates:break
            plane=stack[j] if retain else _match_map(observed,variant,valid)
            for i,(x,y) in enumerate(candidates):
                a,b=max(0,x-2),max(0,y-2);cut=plane[b:y+3,a:x+3]
                cy,cx=np.unravel_index(np.argmax(cut),cut.shape)
                profiles[i,j]=float(cut[cy,cx]);centres[i,j]=[a+cx,b+cy]
        anchors=[];rejected=[];anchor_indices=[]
        hollow=is_hollow(template)
        for i,(x,y) in enumerate(candidates):
            j=int(np.argmax(profiles[i]));sx,sy=map(int,centres[i,j])
            window,sx,sy=inspect_window(variants[j],sx,sy)
            outline=outline_evidence(evidence,index,variants[j],sx,sy) if hollow else None
            hollow_clean=anchor_passes(window,float(profiles[i,j]),outline) if hollow else False
            clean=_clean_anchor(window,float(profiles[i,j]),outline)
            if not clean or not np.all(profiles[i]>-.99):
                rejected.append(dict(x=x,y=y,scale=SCALES[j],window={k:window[k] for k in
                    ('score','missing','extra','observed_fraction','line_only_reject')},
                    correlation=float(profiles[i,j]),hollow_outline=outline,reason='not_clean_full_marker_anchor'))
                continue
            if any(abs(sx-a['x'])<.85*diameter for a in anchors):continue
            anchors.append(dict(x=sx,y=sy,best_scale=SCALES[j],quality=float(profiles[i,j]),profile=profiles[i].tolist(),
                hollow_outline=outline,hollow_anchor=hollow_clean,
                x_px=sx/evidence['scale_x']+evidence['source_center_offset'][0],
                y_px=sy/evidence['scale_y']+evidence['source_center_offset'][1]))
            anchor_indices.append(i)
            if len(anchors)>=CONFIG['maximum_anchors']:break
        scale=1.;gain=0.;consensus=0.;proposed=1.;status='insufficient_clean_anchors_fallback_1x'
        regret=[]
        sparse=(hollow and len(anchors)==HOLLOW_CONFIG['sparse_minimum_anchors']
                and all(a['hollow_anchor'] for a in anchors))
        validation=[];recovery=dict(attempted=False,reason='initial_consensus_or_insufficient_evidence')
        if len(anchors)>=CONFIG['minimum_anchors'] or sparse:
            curves=np.array([a['profile'] for a in anchors]);loss=np.mean(curves.max(axis=1,keepdims=True)-curves,axis=0)
            j=int(np.argmin(loss));proposed=SCALES[j];regret=loss.tolist()
            gain=float(loss[SCALES.index(1.)]-loss[j])
            consensus=float(np.mean([abs(a['best_scale']-proposed)<=CONFIG['scale_agreement']+1e-9 for a in anchors]))
            # With only two locations, both must agree and independently pass
            # at the shared size, not merely at their own best sizes.
            if sparse:
                for i in anchor_indices:
                    sx,sy=map(int,centres[i,j]);shared_window=_window(evidence,index,variants[j],sx,sy)
                    shared_outline=outline_evidence(evidence,index,variants[j],sx,sy)
                    validation.append(dict(x=sx,y=sy,score=shared_window['score'],
                        accepted=anchor_passes(shared_window,float(profiles[i,j]),shared_outline),
                        hollow_outline=shared_outline))
            conflict=consensus<CONFIG['consensus_fraction'] or (sparse and
                (np.ptp([a['best_scale'] for a in anchors])>HOLLOW_CONFIG['sparse_scale_agreement']+1e-9
                 or not all(v['accepted'] for v in validation)))
            minimum_gain=HOLLOW_CONFIG['sparse_minimum_gain'] if sparse else CONFIG['minimum_gain']
            if conflict:status='conflicting_sizes_fallback_1x'
            elif proposed!=1. and gain<minimum_gain:status='native_size_equivalent_keep_1x'
            else:scale=proposed;status='shared_hollow_two_anchor_consensus' if sparse else 'shared_consensus'
            if conflict and not sparse:
                from color_scale_recovery_v46 import recover
                def inspect(site,j):
                    i=anchor_indices[site];sx,sy=map(int,centres[i,j])
                    window=_window(evidence,index,variants[j],sx,sy)
                    outline=outline_evidence(evidence,index,variants[j],sx,sy) if hollow else None
                    return dict(x=sx,y=sy,correlation=float(profiles[i,j]),
                        clean=_clean_anchor(window,float(profiles[i,j]),outline),
                        **{k:window[k] for k in ('score','missing','extra','observed_fraction',
                                                 'ignored_marker_fraction','line_only_reject')})
                recovery=recover(SCALES,curves,anchors,j,inspect)
                if recovery['applied']:
                    scale=recovery['scale'];status=recovery['status']
        if _hollow_geometry and status=='shared_consensus' and scale!=1.:
            j=SCALES.index(scale)
            native_j=SCALES.index(1.)
            validation=[]
            for i in anchor_indices:
                sx,sy=map(int,centres[i,j]);win,sx,sy=inspect_window(variants[j],sx,sy)
                nx,ny=map(int,centres[i,native_j]);native,_,_=inspect_window(variants[native_j],nx,ny)
                validation.append(dict(x=sx,y=sy,score=win['score'],native_score=native['score'],
                    accepted=_clean_anchor(win,float(profiles[i,j])),
                    missing=win['missing'],extra=win['extra']))
            good=[v for v in validation if v['accepted']]
            if len(good)<CONFIG['minimum_anchors'] or np.median([v['score']-v['native_score'] for v in good])<.05:
                scale=1.;status='hollow_geometry_shared_validation_failed'
            else:
                status='shared_hollow_geometry_consensus'
        from color_split_body_v46 import review as review_split_body
        split_body=review_split_body(evidence,index,template,dict(scale=scale))
        if split_body['applied']:
            scale=split_body['scale'];status='shared_occluded_whole_body_consensus'
        reports[str(template['id'])]=dict(scale=float(scale),proposed_scale=float(proposed),status=status,
            anchor_count=len(anchors),anchors=anchors,rejected_anchors=rejected,consensus_fraction=consensus,
            gain_over_native=gain,mean_regret=regret,legend_diameter=diameter,working_diameter=diameter*scale,
            hollow_template=hollow,sparse_hollow_policy=sparse,shared_anchor_validation=validation,
            scale_recovery=recovery,
            search_boundary=proposed in (SCALES[0],SCALES[-1]),split_body=split_body,seconds=perf_counter()-tick)
        if _hollow_geometry:
            from color_hollow_geometry_scale_v46 import POLICY
            reports[str(template['id'])]['render_policy']=POLICY
        elif status=='insufficient_clean_anchors_fallback_1x':
            from color_hollow_geometry_scale_v46 import eligible
            if eligible(template):
                # No clean raster anchors is distinct from conflicting sizes.
                # Refit geometry/pen-width only for independently supported
                # hollow legends, keeping every existing clean-window gate.
                original=reports[str(template['id'])]
                retry=calibrate(evidence,_hollow_geometry=True,_indices=(index,))['series'][str(template['id'])]
                if retry['status']=='shared_hollow_geometry_consensus':
                    retry['raster_attempt']=original
                    reports[str(template['id'])]=retry
                else:
                    original['hollow_geometry_attempt']=retry
                reports[str(template['id'])]['seconds']=perf_counter()-tick
    return dict(version=VERSION,policy='shared_symbol',scales=list(SCALES),config=CONFIG.copy(),
        hollow_config=HOLLOW_CONFIG.copy(),series=reports,
        geometry_scope='one isotropic scale per legend identity for grid, window and tentative scoring',
        coordinates_are_calibration_anchors_not_detections=True,seconds=perf_counter()-started)


def apply_shared_scale(evidence):
    if 'symbol_scale_calibration' in evidence:
        raise ValueError('Colour symbol calibration already applied; refusing double scaling')
    report=calibrate(evidence)
    templates=[]
    for template in evidence['templates']:
        decision=report['series'][str(template['id'])]
        transformed=scaled_template(template,decision['scale'])
        from color_hollow_geometry_scale_v46 import POLICY,render_scaled
        if decision.get('render_policy')==POLICY:
            transformed=render_scaled(template,transformed,decision['scale'])
        transformed.update(symbol_scale=decision['scale'],legend_diameter=float(template['diameter']),
            symbol_scale_status=decision['status'],symbol_scale_version=VERSION)
        templates.append(transformed)
    evidence['templates']=templates
    evidence['symbol_scale_calibration']=report
    return report

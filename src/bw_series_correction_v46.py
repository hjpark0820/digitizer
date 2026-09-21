"""GUI/CLI BW dispatcher: connector SSIM only with repeated source evidence.

Restricted actions use frozen marker rasters, not distances to fitted curves.
Templates and coordinates retain their source-image units. No legend or full
plot re-detection is performed when resuming a correction session.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import cv2
import numpy as np

import bw_step5_v46 as ssim
from bw_suppressed_v46 import encode_marker_mask, decode_marker_mask
from bw_series_structure_v46 import ink_image, classify_connection

VERSION = 'bw_series_correction_v46_v1'


def template_evidence(templates, scales, reports):
    """Save fixed, shared-scale legend geometry, including zero-point series."""
    by_id = {r['swatch_id']:r for r in reports}
    result = {}
    for t in templates:
        scale = float(scales[t.key])
        mask = cv2.resize(t.mask.astype('uint8'), None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        yy, xx = np.nonzero(mask)
        if not len(xx):
            continue
        mask = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
        result[t.key] = dict(swatch_id=t.key, class_name=t.name,
            source_diameter=float(t.diameter), effective_diameter=float(t.diameter*scale),
            marker_scale=scale, marker_mask=encode_marker_mask(mask),
            marker_offset_x=0., marker_offset_y=0.,
            fill_evidence=deepcopy(by_id[t.key].get('fill_evidence', {})),
            provenance='saved_pure_legend_at_shared_symbol_scale')
    return result


def _models(points, saved):
    result = deepcopy(saved or {})
    for p in sorted(points, key=lambda p:-float(p.get('confidence',0.))):
        if p['swatch_id'] not in result:
            result[p['swatch_id']] = deepcopy(p)
            result[p['swatch_id']]['provenance'] = 'legacy_saved_marker_raster'
    for sid, model in result.items():
        if sid != model['swatch_id']:
            raise ValueError('Saved marker evidence identity mismatch')
        mask = decode_marker_mask(model['marker_mask']).astype('uint8')
        yy, xx = np.nonzero(mask)
        if not len(xx) or max(mask.shape)>1024:
            raise ValueError('Invalid saved BW marker evidence')
        # Crop padding, preserving the original anchor relative to the raster.
        ax = mask.shape[1]//2+float(model.get('marker_offset_x',0.))-xx.min()
        ay = mask.shape[0]//2+float(model.get('marker_offset_y',0.))-yy.min()
        mask = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
        model.update(mask=mask, anchor=[ax,ay])
    return result


def _measure(ink, scope, model, x, y):
    """Bidirectional body check; thin lines/bars can explain EXTRA ink only.

    Missing expected ink is always counted. Hollow interiors are separately
    tested after removing thin nuisance strokes. Close rival shapes abstain.
    """
    mask = model['mask']; h,w = mask.shape
    pad = max(3, int(round(.15*max(h,w))))
    left = int(round(x-model['anchor'][0]))-pad
    top = int(round(y-model['anchor'][1]))-pad
    right,bottom = left+w+2*pad,top+h+2*pad
    if left<0 or top<0 or right>ink.shape[1] or bottom>ink.shape[0]:
        return dict(score=0., accepted=False, reason='outside_image')
    observed = ink[top:bottom,left:right]
    valid = scope[top:bottom,left:right]
    expected = np.pad(mask,pad)
    if np.mean(valid[expected>0])<.98:
        return dict(score=0., accepted=False, reason='outside_roi_or_legend')
    near = cv2.distanceTransform(1-observed,cv2.DIST_L2,5)
    recall = float(np.mean(near[expected>0]<=1.))
    hull = np.zeros_like(expected)
    yy,xx=np.nonzero(expected)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.column_stack((xx,yy)).astype('int32')),1)
    core = cv2.erode(hull,np.ones((3,3),np.uint8))
    # A thin T or connector should not explain the full marker silhouette.
    k=max(3,int(round(.16*min(h,w)))|1)
    body=cv2.morphologyEx(observed,cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    expanded=cv2.dilate(hull,np.ones((3,3),np.uint8))
    extra=float(np.sum(body*(1-expanded))/max(1,hull.sum()))
    hole=(core>0)&(expected==0)
    hollow=model['class_name'].startswith('open_')
    hole_clear=float(np.mean(body[hole]==0)) if hole.sum()>=4 else 0.
    hollow_ok=not hollow or (hole.sum()>=4 and hole_clear>=.70)
    # Filled bodies cannot be inferred from a T-shaped stroke alone.
    body_recall=float(np.mean(cv2.dilate(body,np.ones((3,3),np.uint8))[core>0])) if core.any() else 0.
    filled_ok=hollow or body_recall>=.82
    score=recall-.65*min(1.,extra)-(.6*(1-hole_clear) if hollow else .4*(1-body_recall))
    known=model['class_name'] not in ('unknown_marker','suppressed')
    fill=model.get('fill_evidence',{})
    tone_known=(not fill or fill.get('style') in ('solid','open','hollow') or
                bool(fill.get('opaque_body_supported')) or hollow)
    return dict(score=float(score),recall=recall,body_recall=body_recall,
        unexplained_body=extra,hole_clear=hole_clear,accepted=bool(known and tone_known and
        recall>=.92 and extra<=.18 and hollow_ok and filled_ok),
        reason='source_body_and_hole_test',unknown_fill_preserved=not tone_known)


def _references(ink, scope, plot, diameter):
    """Unassigned source fragments: trend/x support only, never marker targets.

    No interpolation across crossings or missing columns is labelled observed.
    Long vertical error bars and border axes are excluded in an auxiliary map.
    """
    a,b,c,d=plot; raw=(ink[b:d,a:c]*scope[b:d,a:c]).copy()
    vertical=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((max(9,round(1.4*diameter)),1),np.uint8))
    horizontal=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,max(20,round(.65*(c-a)))),np.uint8))
    distance=cv2.distanceTransform(raw,cv2.DIST_L2,5)
    ridges=(distance>=cv2.dilate(distance,np.ones((3,3),np.uint8))-1e-5)&(distance>.5)
    linewidth=max(1.,float(2*np.percentile(distance[ridges],40)-1)) if ridges.any() else 1.
    k=max(5,round(.3*diameter)|1,round(2.5*linewidth)|1)
    bodies=cv2.morphologyEx(raw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    nuisance=cv2.dilate(np.maximum.reduce([vertical,horizontal,bodies]),np.ones((3,3),np.uint8))
    thin=raw*(1-nuisance)
    # Border rails are not curve-support evidence, even when broken by ticks.
    border=max(3,round(2*linewidth))
    thin[:border]=0;thin[-border:]=0;thin[:,:border]=0;thin[:,-border:]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(thin,8)
    fragments=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if w<max(12,2*diameter) or area<15:
            continue
        ys,xs=np.where(labels==i)
        path=[[float(xx+a),float(np.median(ys[xs==xx])+b)] for xx in np.unique(xs)
              if np.ptp(ys[xs==xx])<=max(5,3*linewidth)]
        if len(path)<max(12,2*diameter):continue
        fragments.append(dict(path=path,x_support=[path[0][0],path[-1][0]],
            series_id=None,role='unassigned_observed_trend_fragment',not_a_marker_target=True))
    return fragments


def _recover(ink,scope,active,pool,models,plot,limited):
    """At observed x slots search independent y peaks, not another series' y."""
    added=[];audit=[]
    if not active:
        return pool,added,audit
    d=float(np.median([max(m['mask'].shape) for m in models.values()]))
    xs=ssim.clustered_grid(active,d)
    a,b,c,e=plot
    for sid,m in models.items():
        if sid not in limited or m['class_name']=='unknown_marker':
            continue
        mask=m['mask'];h,w=mask.shape; ax,ay=m['anchor']
        own=[p for p in active if p['swatch_id']==sid]
        for x in xs:
            if any(abs(p['cx']-x)<.45*d for p in own):
                continue
            # Extent is image/ROI constrained, not fitted-path predicted y.
            lo=max(a,int(x-ax-.35*d));hi=min(c,int(x-ax+w+.35*d)+1)
            if hi-lo<w or e-b<h:
                continue
            response=cv2.matchTemplate(ink[b:e,lo:hi].astype('float32'),mask.astype('float32'),cv2.TM_CCORR_NORMED)
            for _ in range(12):
                _,v,_,pos=cv2.minMaxLoc(response)
                if v<.55:break
                px,py=pos;cx,cy=lo+px+ax,b+py+ay
                response[max(0,py-h//2):py+h//2+1,:]=0
                if any(p['swatch_id']==sid and np.hypot(p['cx']-cx,p['cy']-cy)<.35*d for p in active+pool+added):continue
                test=_measure(ink,scope,m,cx,cy)
                audit.append(dict(swatch_id=sid,cx=cx,cy=cy,proposal_score=v,verification=test))
                if test['score']<.65 or test.get('recall',0)<.78:continue
                p={k:deepcopy(v) for k,v in m.items() if k not in ('mask','anchor')}
                p.update(cx=float(cx),cy=float(cy),original_detection=False,confidence=test['score'],
                    point_id=f'{sid}_slot_{len(pool)+len(added):05}',candidate_id=f'{sid}_slot_{len(pool)+len(added):05}',
                    source='bw_image_slot_recovery_v46',state='suppressed',
                    suppression_reason='independent_image_peak_at_observed_x; not confirmed')
                added.append(p)
    return pool+added,added,audit


def run_correction(image_bgr,out_dir,init_points=None,init_suppressed=None,plot_area=None,legend_box=None,
                   d_est=None,max_iter=5,init_state=None,log_fn=print,return_diag_imgs=True,no_legend=False,
                   marker_evidence=None,connection_alignment=True,structural_edits=True):
    image=np.asarray(image_bgr)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 BGR image')
    plot=ssim._box(plot_area,image.shape,True);legend=ssim._box(legend_box,image.shape)
    budget=int(max_iter)
    if isinstance(max_iter,bool) or budget!=max_iter or budget<1:
        raise ValueError('max_iter must be a positive integer')
    diameter=float(d_est or (init_state or {}).get('diameter',15.))
    if not np.isfinite(diameter) or diameter<=0:
        raise ValueError('d_est must be positive and finite')
    active,pool,_=ssim._normalise(init_points or [],init_suppressed or [],plot,legend,diameter)
    binding=ssim._binding(image,plot,legend)
    if no_legend:
        if legend:raise ValueError('No-legend state has a legend box')
        binding['explicit_no_legend']=True
    previous=deepcopy(init_state or {})
    if previous and previous.get('binding')!=binding:
        raise ValueError('Saved BW state belongs to another image or ROI')
    if previous and previous.get('version') not in (VERSION,ssim.VERSION):
        raise ValueError('Unsupported BW correction state')
    state=(previous if previous.get('version')==VERSION else
        dict(version=VERSION,binding=binding,diameter=diameter,total_iterations=0,
             migration_from=previous.get('version'),initial_points=deepcopy(active)))
    models=_models(active+pool,marker_evidence or state.get('marker_evidence'))
    # Existing source-model identity may not silently change in a saved session.
    for p in active+pool:
        model=models[p['swatch_id']]
        if model['class_name']!=p['class_name'] or abs(float(model.get('source_diameter',diameter))-
                float(p.get('source_diameter',diameter)))>1e-6:
            raise ValueError('Saved marker evidence differs from the point model')
    state['marker_evidence']={sid:{k:v for k,v in m.items() if k not in ('mask','anchor')} for sid,m in models.items()}
    ink,scope=ink_image(image,plot,legend)
    if 'structure' not in state:
        state['structure']={sid:classify_connection([p for p in active if p['swatch_id']==sid],ink,
            max(m['mask'].shape),scope) for sid,m in models.items()}
        reference_diameter=float(np.median([max(m['mask'].shape) for m in models.values()])) if models else diameter
        state['reference_fragments']=_references(ink,scope,plot,reference_diameter)
    connected={sid for sid,r in state['structure'].items() if r['label']=='connected_supported'}
    limited=set(models)-connected
    log_fn('[v46 BW Step5] series-aware routes: '+str({sid:r['label'] for sid,r in state['structure'].items()}))
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    dest=Path(tempfile.mkdtemp(prefix='bw_series_',dir=str(out_dir)))
    initial=deepcopy(active);initial_pool=deepcopy(pool)
    pool,new_candidates,recovery=_recover(ink,scope,active,pool,models,plot,limited)
    active,pool,_=ssim._normalise(active,pool,plot,legend,diameter)
    ssim.save(dest/'candidate_recovery.json',dict(added=new_candidates,reviews=recovery))
    ssim.save(dest/'structure.json',dict(series=state['structure'],reference_fragments=state['reference_fragments'],
        references_used_for_marker_y=False,reference_ownership='unassigned; no endpoint extrapolation'))
    cache={}
    def tests(p):
        loc=(p['cx'],p['cy'])
        if loc not in cache:
            cache[loc]={sid:_measure(ink,scope,m,*loc) for sid,m in models.items()}
        return cache[loc]
    # Keep a joint iteration budget; at most one accepted action per iteration.
    trace=[];diag=[];history=[];engine_state=state.get('connected_state')
    connected_stopped=not connected
    for number in range(1,budget+1):
        before=deepcopy(active);before_pool=deepcopy(pool);trials=[]
        for p in pool:
            sid=p['swatch_id']
            if sid not in limited:continue
            own=tests(p)[sid];rivals=[t['score'] for k,t in tests(p).items() if k!=sid]
            margin=own['score']-max(rivals) if rivals else 1.
            d=max(models[sid]['mask'].shape)
            occupants=[q for q in active if q['swatch_id']==sid and abs(q['cx']-p['cx'])<=.45*d]
            shared=any(abs(q['cx']-p['cx'])<=.45*d for q in state['initial_points'])
            other_body=any(q['swatch_id']!=sid and np.hypot(q['cx']-p['cx'],q['cy']-p['cy'])<.35*d for q in active)
            reason=('failed_marker_pixels' if not own['accepted'] else 'ambiguous_shape' if margin<.06 else
                    'own_x_slot_occupied' if occupants else 'same_body_other_series' if other_body else
                    'no_observed_x_slot' if not shared else '')
            trials.append(dict(action='ACTIVATE',candidate_id=p['candidate_id'],point=p,
                admissible=not reason,reason=reason,image_score=own['score'],shape_margin=margin,
                gain=own['score']-.70,verification=own))
        # Only nearly coincident SAME-series duplicates can be removed here.
        # No cross-series reassignments or deletion for better curve fit.
        for i,p in enumerate(active):
            sid=p['swatch_id']
            if sid not in limited:continue
            d=max(models[sid]['mask'].shape)
            for q in active[i+1:]:
                if q['swatch_id']!=sid or np.hypot(q['cx']-p['cx'],q['cy']-p['cy'])>=.3*d:continue
                first,second=tests(p)[sid],tests(q)[sid]
                keep,drop,kt,dt=(p,q,first,second) if first['score']>=second['score'] else (q,p,second,first)
                gain=kt['score']-dt['score']
                trials.append(dict(action='REMOVE_DUPLICATE',point=drop,retained=keep,
                    admissible=kt['accepted'] and gain>=.10,gain=gain,
                    reason='same_series_same_body_only',verification=kt))
        eligible=[t for t in trials if t['admissible'] and t['gain']>0]
        chosen=max(eligible,key=lambda t:t['gain']) if eligible else None
        action='NONE';connected_row=None
        if chosen:
            p=deepcopy(chosen['point']);action=chosen['action']
            if action=='ACTIVATE':active.append(p);pool=[q for q in pool if ssim.key(q)!=ssim.key(p)]
            else:
                active=[q for q in active if ssim.key(q)!=ssim.key(p)]
                p.update(candidate_id=p.get('candidate_id',p['point_id']),state='suppressed',suppression_reason='same_body_duplicate')
                pool.append(p)
        elif not connected_stopped:
            # Freeze nonconnected/unknown series in the legacy SSIM engine.
            r=ssim.run_correction(image,dest/'connected',init_points=active,init_suppressed=pool,
                plot_area=plot,legend_box=legend,d_est=diameter,max_iter=1,init_state=engine_state,
                no_legend=no_legend,connection_alignment=connection_alignment,structural_edits=structural_edits,
                editable_series=connected,return_diag_imgs=return_diag_imgs,log_fn=log_fn)
            engine_state=r['runtime_state'];active=r['P_current'];pool=r['S_current']
            connected_row=r['trace'][-1];action=connected_row['action'];diag+=r['diag_imgs']
            if r.get('artifact_dir'):
                import json
                child=Path(r['artifact_dir'])/'trace.json'
                child_report=json.loads(child.read_text(encoding='utf-8'))
                child_report['parent_dispatcher']=VERSION
                ssim.save(child,child_report)
            connected_stopped=action=='NONE'
        row=dict(iteration=state['total_iterations']+number,run_iteration=number,action=action,
            points_before=before,points_after=deepcopy(active),suppressed_before=before_pool,
            suppressed_after=deepcopy(pool),added=ssim.difference(active,before),removed=ssim.difference(before,active),
            trials=trials,selected=chosen,connected_action=connected_row,
            stop_reason='no_admissible_image_candidate' if action=='NONE' else '',
            metric='restricted_marker_evidence' if chosen or connected_row is None else '1-SSIM_connected_only')
        trace.append(row);history.append((row['iteration'],action,len(active),None,action!='NONE'))
        log_fn(f'[v46 BW Step5] iter={row["iteration"]} {action}, P={len(active)}, S={len(pool)}')
        if action=='NONE':break
    state.update(P_current=deepcopy(active),S_current=deepcopy(pool),connected_state=engine_state,
                 total_iterations=state['total_iterations']+len(trace))
    report=dict(version=VERSION,series='all_bw_series',metric='structure_routed',initial_points=initial,
        initial_suppressed=initial_pool,P_current=active,S_current=pool,iterations=trace,
        structure=state['structure'],max_iterations=budget,previous_iteration_count=state['total_iterations']-len(trace),
        warning='Source-supported hypotheses, not validated ground truth; no fitted-path y objective.')
    ssim.save(dest/'trace.json',report);ssim.save(dest/'runtime_state.json',state)
    return dict(P_current=active,S_current=pool,runtime_state=state,mode_xs=ssim.clustered_grid(active,diameter),
        history=history,trace=trace,diag_imgs=diag,steps=diag,status='completed',artifact_dir=str(dest),
        backend='bw_series_correction_v46',metric='structure_routed',initial_score=None,final_score=None)

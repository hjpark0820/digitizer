"""GUI/CLI BW dispatcher: connected SSIM first for source-supported polylines.

Restricted actions use frozen marker rasters, not distances to fitted curves.
Templates and coordinates retain their source-image units. No legend or full
plot re-detection is performed when resuming a correction session.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import tempfile
import cv2
import numpy as np

import bw_step5_v46 as ssim
from bw_suppressed_v46 import encode_marker_mask, decode_marker_mask
from bw_series_structure_v46 import ink_image, classify_connection

VERSION = 'bw_series_correction_v46_v1'
CONNECTED_FIRST_POLICY = 'source-supported-active-series-connected-first-v2'
CONNECTED_IDENTITY_POLICY = 'all-legend-connected-identity-v1'


def _overlap_budget_lane(point):
    """Additional filled hypotheses cannot consume the existing replay lane.

    Source AND the explicit new hypothesis reason identify the new work.
    A generic overlap flag, candidate ID, or legacy image-slot source alone
    remains in the original lane. Routing grants no acceptance authority.
    """
    hypothesis=point.get('overlap_hypothesis') or {}
    if not isinstance(hypothesis,dict):
        return 'native'
    if (point.get('source')=='filled-boundary-foreground-retention-v1' and
            point.get('admission_reason')=='supported_filled_boundary_needs_foreground' and
            hypothesis.get('reason')=='supported_filled_boundary_needs_foreground'):
        return 'filled_boundary'
    if (point.get('source')=='bw_image_slot_recovery_v46' and
            hypothesis.get('reason')=='filled_image_peak_needs_foreground' and
            hypothesis.get('requires_fresh_scene_validation') is True):
        return 'filled_boundary'
    return 'native'


class _OverlapReplayBudget:
    """Two independent bounded caches; both execute the same strict verifier."""
    def __init__(self):
        self.reports={lane:dict(tested=0,seconds=0.,maximum_candidates=120,
            maximum_seconds=35.,budget_exhausted=False,accepted=0,reasons={})
            for lane in ('native','filled_boundary')}
        self.cache={}

    def measure(self,point,binding,verifier):
        lane=_overlap_budget_lane(point)
        report=self.reports[lane]
        key=(lane,binding)
        if key not in self.cache:
            if (report['tested']>=report['maximum_candidates'] or
                    report['seconds']>=report['maximum_seconds']):
                self.cache[key]=dict(valid=False,reason='overlap_recheck_budget_exhausted')
                report['budget_exhausted']=True
            else:
                started=perf_counter()
                self.cache[key]=verifier()
                report['seconds']+=perf_counter()-started
                report['tested']+=1
                value=self.cache[key]
                report['accepted']+=int(value.get('valid',False))
                reason=value['reason']
                report['reasons'][reason]=report['reasons'].get(reason,0)+1
        return self.cache[key]


def _connected_identity_check(sid, measurements):
    """Veto a connected identity only when an accepted legend rival wins.

    Connector eligibility does not restrict which legend symbols may compete.
    This reuses the restricted-local pixel tests and their 0.06 identity gap;
    it does not require a partly hidden marker to pass the isolated body test.
    Replayed contextual evidence has its own identity contract, not the same
    numerical scale as the isolated tests, and is therefore preserved.
    """
    own = measurements[sid]
    check = dict(policy=CONNECTED_IDENTITY_POLICY, valid=True, own_series=sid,
        own_score=float(own['score']), minimum_advantage=.06,
        measurements=deepcopy(measurements))
    if own.get('partial_visibility', {}).get('valid'):
        check['reason'] = 'contextual_identity_preserved'
        return check
    # The axis lane observes original pixels outside the plot, but it may
    # never win by explaining the axis itself as a marker. Its explicit
    # independent-support rejection applies even when every rival also fails.
    if own.get('axis_contact') and not own.get('accepted'):
        check.update(valid=False, reason='axis_contact_marker_not_supported')
        return check
    rivals = [(other, value) for other, value in measurements.items()
              if other != sid and value.get('accepted')]
    if not rivals:
        check['reason'] = 'no_accepted_legend_rival'
        return check
    rival_sid, rival = max(rivals, key=lambda item:(float(item[1]['score']), item[0]))
    advantage = float(rival['score']) - float(own['score'])
    check.update(rival_series=rival_sid, rival_score=float(rival['score']),
                 rival_advantage=advantage, valid=advantage < .06,
                 reason='accepted_legend_rival_wins' if advantage >= .06 else
                        'no_decisive_legend_rival')
    return check


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
        # Save measured legend gray separately from the legacy binary mask.
        # Local overlap replay must not invent a perfect named-shape glyph.
        from bw_partial_visibility_v46 import _source_template_raster
        source = _source_template_raster(t, scale)
        if source is not None:
            result[t.key]['overlap_source'] = dict(version='saved-legend-gray-overlap-v1',
                ink=source[0].tolist(), reliability=source[2].tolist(),
                marker_scale=scale, class_name=t.name,
                source_diameter=float(t.diameter), anchor='aligned_center')
    return result


def _models(points, saved):
    result = deepcopy(saved or {})
    for p in sorted(points, key=lambda p:('pending_identity' in p,-float(p.get('confidence',0.)))):
        if p['swatch_id'] not in result:
            result[p['swatch_id']] = deepcopy(p)
            result[p['swatch_id']]['provenance'] = ('pending_only_marker_raster' if 'pending_identity' in p
                                                    else 'legacy_saved_marker_raster')
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


def _measure(ink, scope, model, x, y, *, context_evidence=None):
    """Bidirectional body check; thin lines/bars can explain EXTRA ink only.

    Missing expected ink is always counted. Hollow interiors are separately
    tested after removing thin nuisance strokes. Close rival shapes abstain.
    """
    # This evidence is revalidated against the source pixels, exact target
    # geometry and current strict neighbour roster by tests(), not a flag
    # copied from a saved result. Do not apply the isolated-hole veto a second
    # time after a complete contextual explanation has passed.
    if context_evidence is not None and context_evidence.get('valid'):
        return dict(score=float(context_evidence['score']), accepted=True,
            reason='source_bound_partial_visibility',
            partial_visibility=context_evidence,
            identity_margin=float(context_evidence.get('identity_margin', 0.)))
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


def _recover(ink,scope,active,pool,models,plot,limited, *, context_verifier=None):
    """At observed x slots search independent y peaks, not another series' y."""
    added=[];audit=[]
    observed_active=[p for p in active if 'pending_identity' not in p]
    if not observed_active:
        return pool,added,audit
    pending_pool=[p for p in pool if 'pending_identity' in p]
    pool=[p for p in pool if 'pending_identity' not in p]
    d=float(np.median([max(m['mask'].shape) for m in models.values()]))
    xs=ssim.clustered_grid(observed_active,d)
    a,b,c,e=plot
    for sid,m in models.items():
        if (sid not in limited or m['class_name']=='unknown_marker' or
                m.get('provenance')=='pending_only_marker_raster'):
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
                p={k:deepcopy(v) for k,v in m.items() if k not in ('mask','anchor','overlap_source')}
                p.update(cx=float(cx),cy=float(cy),original_detection=False,confidence=test['score'],
                    point_id=f'{sid}_slot_{len(pool)+len(added):05}',candidate_id=f'{sid}_slot_{len(pool)+len(added):05}',
                    source='bw_image_slot_recovery_v46',state='suppressed',
                    suppression_reason='independent_image_peak_at_observed_x; not confirmed')
                review=dict(swatch_id=sid,cx=cx,cy=cy,proposal_score=v,verification=test)
                audit.append(review)
                isolated_pass=test['score']>=.65 and test.get('recall',0)>=.78
                if not isolated_pass:
                    # A white-faced foreground marker can remove real body
                    # ink. Test that complete scene before vetoing an image
                    # peak for missing fill; the peak is never direct active
                    # authority and every source/visibility/rival gate stays.
                    neighbours=[q for q in active if q['swatch_id']!=sid and
                        not any(k in q for k in ('pending_identity','partial_visibility_certificate',
                            'partial_visibility_hypothesis','overlap_hypothesis','overlap_recheck','seed_bootstrap'))
                        and np.hypot(q['cx']-cx,q['cy']-cy)<.45*(max(m['mask'].shape)+
                            max(models[q['swatch_id']]['mask'].shape))]
                    if context_verifier is None or not m['class_name'].startswith('filled_') or not neighbours:
                        continue
                    # Mark only this NEW isolated-fail contextual probe before
                    # calling the budgeted verifier. Ordinary legacy slots
                    # retain their original lane and acceptance contracts.
                    p['overlap_hypothesis']=dict(reason='filled_image_peak_needs_foreground',
                        active_authority=False,requires_fresh_scene_validation=True)
                    context=context_verifier(p,active)
                    review['contextual_verification']=context
                    if not context.get('valid'):
                        continue
                    p['confidence']=float(context['score'])
                added.append(p)
    active_keys={ssim.key(p) for p in active}
    # A generated native slot is still a hypothesis, not an accepted marker.
    # Preserve both evidence contracts until one route actually activates the
    # typed location. Pending alternatives must not block native generation,
    # but native generation must not silently erase pending source checks.
    return pool+added+[p for p in pending_pool if ssim.key(p) not in active_keys],added,audit


def run_correction(image_bgr,out_dir,init_points=None,init_suppressed=None,plot_area=None,legend_box=None,
                   d_est=None,max_iter=5,init_state=None,log_fn=print,return_diag_imgs=True,no_legend=False,
                   marker_evidence=None,connection_alignment=True,structural_edits=True,init_untyped=None,
                   allow_untyped_activation=False,seed_series_bootstrap=None):
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
    restored_pool=ssim.restore_pending_archive(init_points or [],init_suppressed or [],
                                              (init_state or {}).get('pending_identity_archive',[]))
    active,pool,_=ssim._normalise(init_points or [],restored_pool,plot,legend,diameter)
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
    # Experimental opt-in. Omitted options preserve the saved seed policy;
    # connected-first routing applies independently of that extra phase.
    from bw_seed_route_v46 import POLICY as SEED_POLICY
    saved_seed=state.get('seed_series_bootstrap')
    if seed_series_bootstrap is not None and not isinstance(seed_series_bootstrap,bool):
        raise ValueError('seed_series_bootstrap must be bool or None')
    seed_enabled=(bool(saved_seed and saved_seed.get('enabled')) if seed_series_bootstrap is None
                  else seed_series_bootstrap)
    seed_policy=dict(policy=SEED_POLICY,enabled=seed_enabled)
    if saved_seed is not None and saved_seed!=seed_policy:
        raise ValueError('Cannot change saved seed bootstrap policy; start from detection')
    if saved_seed is None and seed_enabled and previous and previous.get('total_iterations',0):
        raise ValueError('Seed bootstrap requires a fresh detection correction state')
    if seed_enabled or saved_seed is not None:
        state['seed_series_bootstrap']=seed_policy
    archived_pending=ssim.pending_archive(state.get('pending_identity_archive',[])+active+pool)
    if archived_pending:
        state['pending_identity_archive']=archived_pending
    models=_models(active+pool,marker_evidence or state.get('marker_evidence'))
    # Existing source-model identity may not silently change in a saved session.
    for p in active+pool:
        model=models[p['swatch_id']]
        if model['class_name']!=p['class_name'] or abs(float(model.get('source_diameter',diameter))-
                float(p.get('source_diameter',diameter)))>1e-6:
            raise ValueError('Saved marker evidence differs from the point model')
    state['marker_evidence']={sid:{k:v for k,v in m.items() if k not in ('mask','anchor')} for sid,m in models.items()}
    pending_validation=(ssim.make_pending_validation(image,models,plot,legend)
                        if archived_pending else None)
    from bw_unknown_candidates_v46 import validate_regions
    untyped=validate_regions(image,
        init_untyped if init_untyped is not None else state.get('U_current',[]),
        plot,[legend] if legend else [])
    initial_untyped=deepcopy(untyped)
    untyped_reviews=[];untyped_cache={};untyped_seconds=0.
    ink,scope=ink_image(image,plot,legend)
    from bw_axis_contact_v46 import AxisContactVerifier, POLICY as AXIS_POLICY
    axis_verifier=AxisContactVerifier(image,plot,legend,models)
    state['axis_contact_policy']=dict(policy=AXIS_POLICY,
        center_domain='plot', observation_domain='bounded_original_source',
        valid_replayed_context='preserved; isolated_axis_verifier_not_applied')
    if 'structure' not in state:
        state['structure']={sid:classify_connection([p for p in active if p['swatch_id']==sid and
                                                   'pending_identity' not in p],ink,
            max(m['mask'].shape),scope) for sid,m in models.items()}
        reference_diameter=float(np.median([max(m['mask'].shape) for m in models.values()])) if models else diameter
        state['reference_fragments']=_references(ink,scope,plot,reference_diameter)
    connected={sid for sid,r in state['structure'].items() if r['label']=='connected_supported'}
    limited=set(models)-connected
    # Straight connector support belongs to each series, not the whole plot.
    # Current active points alone do not establish that a curved/uncertain
    # series can be rendered or edited with straight segments.
    routing_policy=dict(policy=CONNECTED_FIRST_POLICY,minimum_active_points=1,
        required_structure_label='connected_supported',
        source_supported_series=sorted(connected),
        order=['connected','restricted_local','seed_add_only','untyped'],
        local_fallback='original_limited_series_and_retained_overlap_hypotheses',
        structure_role='frozen_initial_per_series_straight_connector_eligibility')
    state['connected_first_policy']=deepcopy(routing_policy)
    state['connected_identity_policy']=dict(policy=CONNECTED_IDENTITY_POLICY,
        competing_series=sorted(models),minimum_advantage=.06,
        role='veto_before_connected_scoring; does_not_expand_connector_eligibility')
    seed_routes=None
    if seed_enabled:
        from bw_seed_route_v46 import SeedRoutes
        seed_routes=SeedRoutes(ink,scope,models,state['initial_points'],state['structure'])
    log_fn('[v46 BW Step5] connected first only for source-supported straight series with active points; initial evidence: '+
           str({sid:r['label'] for sid,r in state['structure'].items()}))
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    dest=Path(tempfile.mkdtemp(prefix='bw_series_',dir=str(out_dir)))
    initial=deepcopy(active);initial_pool=deepcopy(pool)
    ssim.save(dest/'structure.json',dict(series=state['structure'],reference_fragments=state['reference_fragments'],
        references_used_for_marker_y=False,reference_ownership='unassigned; no endpoint extrapolation'))
    cache={}
    replay_budget=_OverlapReplayBudget()
    overlap_rechecks=replay_budget.reports['native']
    filled_boundary_rechecks=replay_budget.reports['filled_boundary']
    def overlap_binding(roster):
        return (ssim.rendering_key(roster),tuple(sorted((str(q.get('point_id')),
            str(q.get('class_name')),float(q.get('marker_scale',1.)),
            float(q.get('marker_aspect',1.)),float(q.get('source_diameter',0.)),
            tuple(name in q for name in ('pending_identity','partial_visibility_certificate',
                'partial_visibility_hypothesis','overlap_hypothesis','overlap_recheck')))
            for q in roster)))
    def overlap_validation(point, roster):
        # Exact rendered geometry, not a saved `valid` flag or coordinate-only
        # key. Changed masks, offsets, identities, or neighbours require replay.
        binding=(overlap_binding([point]),overlap_binding(roster))
        from bw_overlap_recheck_v46 import evaluate
        return replay_budget.measure(point,binding,
            lambda:evaluate(image,point,roster,models,plot,legend))
    def overlap_roster_checks(roster):
        return [dict(point_id=q.get('point_id'),**overlap_validation(q,roster))
                for q in roster if 'overlap_recheck' in q]
    pool,new_candidates,recovery=_recover(ink,scope,active,pool,models,plot,limited,
                                         context_verifier=overlap_validation)
    active,pool,_=ssim._normalise(active,pool,plot,legend,diameter)
    ssim.save(dest/'candidate_recovery.json',dict(added=new_candidates,reviews=recovery))
    def tests(p, roster=None):
        roster=active if roster is None else roster
        loc=(p['cx'],p['cy'])
        # Context may change during correction. A coordinate-only cache would
        # reuse an overlap explanation after its supporting neighbour moved.
        context_key=(loc,ssim.setkey(roster),p.get('point_id'),_overlap_budget_lane(p))
        if pending_validation is not None:
            import json
            context_key=json.dumps(ssim.plain(dict(point=p,active=roster)),sort_keys=True,allow_nan=False)
        if context_key not in cache:
            contextual=None
            if p.get('partial_visibility_certificate') and 'pending_identity' not in p:
                from bw_partial_visibility_v46 import validate_certificate
                contextual=validate_certificate(image,p,roster)
            overlap_context = None
            if (p.get('overlap_hypothesis') or p.get('admission_reason') == 'verified_identity_ambiguous'):
                overlap_context = overlap_validation(p,roster)
                if overlap_context.get('valid'):
                    contextual = overlap_context
            pending_unverified=('partial_visibility_hypothesis' in p and
                                not (contextual is not None and contextual.get('valid')))
            # Valid, source-replayed overlap evidence already explains other
            # markers as well as lines. Do not replace that richer contract
            # with the axis-only nuisance model. All other axis candidates
            # compete on one common original-source window and fixed axes.
            axis_measurements=(None if (contextual or {}).get('valid') else
                               axis_verifier.measurements(*loc))
            cache[context_key]={sid:(dict(score=0.,accepted=False,
                reason='partial_visibility_hypothesis_not_verified',partial_visibility=contextual or {})
                if sid==p['swatch_id'] and pending_unverified else
                axis_measurements[sid] if axis_measurements is not None else
                _measure(ink,scope,m,*loc,
                    context_evidence=contextual if sid==p['swatch_id'] else None))
                for sid,m in models.items()}
            own = cache[context_key][p['swatch_id']]
            if overlap_context is not None:
                own['overlap_recheck'] = overlap_context
            if p.get('overlap_hypothesis') and not (overlap_context or {}).get('valid'):
                own.update(accepted=False,reason='overlap_hypothesis_not_verified')
            if 'pending_identity' in p:
                proposed=(roster if ssim.key(p) in set(ssim.setkey(roster)) else roster+[p])
                own=cache[context_key][p['swatch_id']]
                check=ssim.pending_roster_checks([q for q in proposed],pending_validation)
                failed=[q for q in check if not q.get('valid')]
                own['pending_identity_checks']=check
                if failed:
                    own.update(accepted=False,reason='pending_identity_not_verified')
        return cache[context_key]
    def connected_identity_validation(point, proposed_roster):
        # Replay against the proposed neighbours: a REPLACE/REASSIGN must not
        # keep using a removed marker as an explanation of the new candidate.
        neighbours=[q for q in proposed_roster if ssim.key(q)!=ssim.key(point)]
        return _connected_identity_check(point['swatch_id'],tests(point,neighbours))
    # Keep a joint iteration budget; at most one accepted action per iteration.
    trace=[];diag=[];history=[];engine_state=state.get('connected_state')
    for number in range(1,budget+1):
        before=deepcopy(active);before_pool=deepcopy(pool);before_untyped=deepcopy(untyped);trials=[]
        active_counts={sid:sum(p['swatch_id']==sid for p in active) for sid in models}
        priority_series={sid for sid,count in active_counts.items() if count>=1 and sid in connected}
        action='NONE';connected_row=None;legacy_connected_row=None;seed_phase=False;route_attempts=[]
        if priority_series:
            pending_options={'pending_validation':pending_validation} if pending_validation is not None else {}
            if any('overlap_recheck' in p for p in active):
                pending_options['overlap_validation']=overlap_validation
            r=ssim.run_correction(image,dest/'connected',init_points=active,init_suppressed=pool,
                plot_area=plot,legend_box=legend,d_est=diameter,max_iter=1,init_state=engine_state,
                no_legend=no_legend,connection_alignment=connection_alignment,structural_edits=structural_edits,
                editable_series=connected,call_editable_series=priority_series,
                marker_identity_validation=connected_identity_validation,
                return_diag_imgs=return_diag_imgs,log_fn=log_fn,**pending_options)
            engine_state=r['runtime_state'];active=r['P_current'];pool=r['S_current']
            connected_row=r['trace'][-1];action=connected_row['action'];diag+=r['diag_imgs']
            route_attempts.append(dict(route='connected',series=sorted(priority_series),action=action,
                score_before=connected_row.get('score_before'),score_after=connected_row.get('score_after')))
            if r.get('artifact_dir'):
                import json
                child=Path(r['artifact_dir'])/'trace.json'
                child_report=json.loads(child.read_text(encoding='utf-8'))
                child_report['parent_dispatcher']=VERSION
                ssim.save(child,child_report)
        else:
            route_attempts.append(dict(route='connected',series=[],action='SKIPPED',
                reason='no_active_source_supported_straight_series'))
        local_attempted=action=='NONE'
        for p in pool if local_attempted else ():
            sid=p['swatch_id']
            if sid not in limited and not p.get('overlap_hypothesis'):continue
            own=tests(p)[sid];rivals=[t['score'] for k,t in tests(p).items() if k!=sid]
            contextual=own.get('partial_visibility',{})
            margin=(own['identity_margin'] if contextual.get('valid') else
                    own['score']-max(rivals) if rivals else 1.)
            d=max(models[sid]['mask'].shape)
            pair_neighbors=set(contextual.get('explained_neighbor_ids',[])) if (
                contextual.get('valid') and contextual.get('explained_occlusion_validated')) else set()
            occupants=[q for q in active if q['swatch_id']==sid and q.get('point_id') not in pair_neighbors
                       and abs(q['cx']-p['cx'])<=.45*d]
            # A source-backed marker need not share x with an initial active.
            # Observed x slots guide _recover() proposals only; they are not
            # evidence against an existing suppressed candidate elsewhere.
            # Keep pixel, identity, occupancy and scene-context checks below.
            explained_neighbors=set(contextual.get('neighbor_ids',[])) if contextual.get('valid') else set()
            other_body=any(q['swatch_id']!=sid and q.get('point_id') not in explained_neighbors
                and np.hypot(q['cx']-p['cx'],q['cy']-p['cy'])<.35*d for q in active)
            identity_supported=bool(contextual.get('valid')) or margin>=.06
            reason=('pending_identity_not_verified' if own.get('reason')=='pending_identity_not_verified' else
                    'overlap_hypothesis_not_verified' if own.get('reason')=='overlap_hypothesis_not_verified' else
                    'partial_visibility_hypothesis_not_verified'
                    if own.get('reason')=='partial_visibility_hypothesis_not_verified' else
                    'failed_marker_pixels' if not own['accepted'] else 'ambiguous_shape' if not identity_supported else
                    'own_x_slot_occupied' if occupants else 'same_body_other_series' if other_body else '')
            pending_checks=ssim.pending_roster_checks(active+[p],pending_validation)
            if any(not check.get('valid') for check in pending_checks):
                reason='pending_identity_not_verified'
            overlap_checks=overlap_roster_checks(active+[p])
            if any(not check.get('valid') for check in overlap_checks):
                reason='retained_overlap_context_not_verified'
            trials.append(dict(action='ACTIVATE',candidate_id=p['candidate_id'],point=p,
                admissible=not reason,reason=reason,image_score=own['score'],shape_margin=margin,
                gain=(float(contextual['scene_improvement'])-float(contextual['minimum_scene_improvement'])
                      if contextual.get('valid') else own['score']-.70),verification=own))
            if pending_checks:trials[-1]['pending_identity_checks']=pending_checks
            if overlap_checks:trials[-1]['overlap_context_checks']=overlap_checks
        # Only nearly coincident SAME-series duplicates can be removed here.
        # No cross-series reassignments or deletion for better curve fit.
        for i,p in enumerate(active if local_attempted else ()):
            sid=p['swatch_id']
            if sid not in limited:continue
            d=max(models[sid]['mask'].shape)
            for q in active[i+1:]:
                if q['swatch_id']!=sid or np.hypot(q['cx']-p['cx'],q['cy']-p['cy'])>=.3*d:continue
                first,second=tests(p)[sid],tests(q)[sid]
                # Near centers alone cannot erase a source-verified two-marker
                # scene. This exemption binds only the freshly checked donor.
                first_context=first.get('partial_visibility',{})
                second_context=second.get('partial_visibility',{})
                distinct=any(ctx.get('valid') and ctx.get('explained_occlusion_validated') and
                             other.get('point_id') in ctx.get('explained_neighbor_ids',[])
                             for ctx,other in ((first_context,q),(second_context,p)))
                keep,drop,kt,dt=(p,q,first,second) if first['score']>=second['score'] else (q,p,second,first)
                gain=kt['score']-dt['score']
                pending_checks=ssim.pending_roster_checks(
                    [a for a in active if ssim.key(a)!=ssim.key(drop)],pending_validation)
                pending_safe=not any(not check.get('valid') for check in pending_checks)
                overlap_checks=overlap_roster_checks([a for a in active if ssim.key(a)!=ssim.key(drop)])
                overlap_safe=all(check.get('valid') for check in overlap_checks)
                pending_pair='pending_identity' in keep or 'pending_identity' in drop
                trials.append(dict(action='REMOVE_DUPLICATE',point=drop,retained=keep,
                    admissible=not distinct and kt['accepted'] and gain>=.10 and pending_safe and not pending_pair and overlap_safe,gain=gain,
                    reason=('pending_identity_activation_only' if pending_pair else
                            'pending_identity_not_verified' if not pending_safe else
                            'retained_overlap_context_not_verified' if not overlap_safe else
                            'certified_distinct_occluded_markers' if distinct else 'same_series_same_body_only'),
                    verification=kt))
                if pending_checks:trials[-1]['pending_identity_checks']=pending_checks
                if overlap_checks:trials[-1]['overlap_context_checks']=overlap_checks
        eligible=[t for t in trials if t['admissible'] and t['gain']>0]
        if local_attempted and not eligible and seed_routes is not None:
            pair_trials=seed_routes.pairs(active,pool,tests)
            for trial in pair_trials:
                if trial.get('admissible'):
                    proposed=active+trial['points']
                    pending_checks=ssim.pending_roster_checks(proposed,pending_validation)
                    overlap_checks=overlap_roster_checks(proposed)
                    if any(not r.get('valid') for r in pending_checks+overlap_checks):
                        trial.update(admissible=False,reason='seed_pair_roster_context_not_verified')
                    trial['pending_identity_checks']=pending_checks
                    trial['overlap_context_checks']=overlap_checks
            trials.extend(pair_trials)
            eligible=[t for t in pair_trials if t.get('admissible') and t['gain']>0]
        chosen=max(eligible,key=lambda t:t['gain']) if eligible else None
        if local_attempted:
            route_attempts.append(dict(route='restricted_local',series=sorted(limited),
                action=chosen['action'] if chosen else 'NONE',trial_count=len(trials)))
        if chosen:
            action=chosen['action']
            if action=='ACTIVATE_PAIR_INIT':
                additions=deepcopy(chosen['points'])
                for point in additions:
                    point.update(original_detection=False,seed_bootstrap=dict(policy=SEED_POLICY,
                        source='independent_pair',validated_at_iteration=state['total_iterations']+number))
                keys={ssim.key(p) for p in additions}
                active,pool,_=ssim._normalise(active+additions,[p for p in pool if ssim.key(p) not in keys],plot,legend,diameter)
                p=None
            else:
                p=deepcopy(chosen['point'])
            if action=='ACTIVATE' and chosen.get('verification',{}).get('overlap_recheck',{}).get('valid'):
                p['overlap_recheck'] = deepcopy(chosen['verification']['overlap_recheck'])
            if action=='ACTIVATE':active.append(p);pool=[q for q in pool if ssim.key(q)!=ssim.key(p)]
            elif action!='ACTIVATE_PAIR_INIT':
                active=[q for q in active if ssim.key(q)!=ssim.key(p)]
                p.update(candidate_id=p.get('candidate_id',p['point_id']),state='suppressed',suppression_reason='same_body_duplicate')
                pool.append(p)
        # Only after the connected-first/local sequence has no action may a
        # refreshed route propose additions. Reuse its reference and immutable
        # original route: the new phase has its own freshly computed baseline.
        if action=='NONE' and seed_routes is not None:
            refreshed=seed_routes.refresh(active,tests)
            if refreshed:
                legacy_connected_row=connected_row
                seed_before_keys=set(ssim.setkey(active))
                options={'pending_validation':pending_validation} if pending_validation is not None else {}
                if any('overlap_recheck' in p for p in active):
                    options['overlap_validation']=overlap_validation
                r=ssim.run_correction(image,dest/'seed_connected',init_points=active,init_suppressed=pool,
                    plot_area=plot,legend_box=legend,d_est=diameter,max_iter=1,init_state=engine_state,
                    no_legend=no_legend,connection_alignment=connection_alignment,structural_edits=structural_edits,
                    editable_series=connected,addition_only_series=refreshed,
                    addition_validation=lambda p,roster:seed_routes.validate_addition(p,roster,tests),
                    overlap_candidates=False,return_diag_imgs=return_diag_imgs,log_fn=log_fn,**options)
                engine_state=r['runtime_state'];active=r['P_current'];pool=r['S_current']
                connected_row=r['trace'][-1];action=connected_row['action'];diag+=r['diag_imgs'];seed_phase=True
                route_attempts.append(dict(route='seed_add_only',series=sorted(refreshed),action=action,
                    score_before=connected_row.get('score_before'),score_after=connected_row.get('score_after')))
                # Keep add-only recoveries distinguishable from native anchors.
                # A newly added point cannot become a foreground donor merely
                # because the connected engine promoted its saved hypothesis.
                for point in active:
                    if ssim.key(point) not in seed_before_keys:
                        point.update(original_detection=False,seed_bootstrap=dict(policy=SEED_POLICY,
                            source='refreshed_route_addition',validated_at_iteration=state['total_iterations']+number))
                engine_state['P_current']=deepcopy(active)
                connected_row['points_after']=deepcopy(active)
                connected_row['added']=ssim.difference(active,before)
        # Keep the existing typed correction sequence unchanged. Only when
        # that sequence has no admissible action do unresolved source regions
        # compete as bounded local scenes. A scene is one atomic Step-5 action;
        # it never deletes/relabels an active point or fabricates a source mask.
        if action=='NONE' and untyped:
            from bw_untyped_resolver_v46 import resolve_regions
            context_key=(tuple(u['region_id'] for u in untyped),ssim.setkey(active))
            if context_key not in untyped_cache:
                untyped_started=perf_counter()
                untyped_cache[context_key]=resolve_regions(image,untyped,models,active,plot,legend)
                untyped_seconds+=perf_counter()-untyped_started
            groups,reviews=untyped_cache[context_key]
            untyped_reviews.append(dict(iteration=state['total_iterations']+number,reviews=reviews))
            eligible_groups=[]
            for group in groups:
                pending_checks=(ssim.pending_roster_checks(active+group['points'],pending_validation)
                                if allow_untyped_activation else [])
                pending_safe=all(check.get('valid') for check in pending_checks)
                overlap_checks=(overlap_roster_checks(active+group['points']) if allow_untyped_activation else [])
                overlap_safe=all(check.get('valid') for check in overlap_checks)
                if pending_safe and overlap_safe:eligible_groups.append(group)
                trials.append(dict(action='RESOLVE_UNTYPED',admissible=bool(allow_untyped_activation and pending_safe and overlap_safe),
                    reason=('untyped_promotion_experimental_only' if not allow_untyped_activation else
                            'retained_overlap_context_not_verified' if not overlap_safe else
                            'pending_identity_not_verified' if not pending_safe else ''),
                    region_id=group['region_id'],points=group['points'],gain=group['gain'],
                    evidence=group['evidence']))
                if pending_checks:trials[-1]['pending_identity_checks']=pending_checks
                if overlap_checks:trials[-1]['overlap_context_checks']=overlap_checks
            # A local two-marker fit can still explain a square plus its
            # error-bar cap as a square/triangle pair. Keep that proposal and
            # the source region inspectable, but do not enable this unvalidated
            # promotion path in the GUI/saved-session default.
            if eligible_groups and allow_untyped_activation:
                group=max(eligible_groups,key=lambda g:(g['gain'],g['region_id']))
                additions=deepcopy(group['points'])
                if group['region_id'] not in {u['region_id'] for u in untyped}:
                    raise ValueError('Untyped resolver returned an unknown source region')
                if not additions or any(ssim.key(p) in set(ssim.setkey(active)) for p in additions):
                    raise ValueError('Untyped resolver returned an empty or already active scene')
                for point in additions:
                    model=models.get(point.get('swatch_id'))
                    if model is None or any(point.get(field)!=model.get(field) for field in
                            ('class_name','marker_mask','source_diameter')):
                        raise ValueError('Untyped resolver changed the saved marker model')
                    if any(float(point.get(field,0.))!=float(model.get(field,0.)) for field in
                            ('marker_offset_x','marker_offset_y')):
                        raise ValueError('Untyped resolver changed the saved marker anchor')
                new_keys=set(ssim.setkey(additions))
                new_pool=[p for p in pool if ssim.key(p) not in new_keys]
                checked,checked_pool,_=ssim._normalise(active+additions,new_pool,plot,legend,diameter)
                # Assignment is committed only after every proposed point and
                # its series-bound geometry validate successfully.
                active,pool=checked,checked_pool
                untyped=[u for u in untyped if u['region_id']!=group['region_id']]
                chosen=dict(action='RESOLVE_UNTYPED',region_id=group['region_id'],
                    points=additions,gain=group['gain'],evidence=group['evidence'])
                action='RESOLVE_UNTYPED'
            route_attempts.append(dict(route='untyped',action=action,activation_enabled=bool(allow_untyped_activation)))
        pool=ssim.restore_pending_archive(active,pool,archived_pending)
        row=dict(iteration=state['total_iterations']+number,run_iteration=number,action=action,
            points_before=before,points_after=deepcopy(active),suppressed_before=before_pool,
            suppressed_after=deepcopy(pool),added=ssim.difference(active,before),removed=ssim.difference(before,active),
            untyped_before=before_untyped,untyped_after=deepcopy(untyped),
            trials=trials,selected=chosen,connected_action=connected_row,
            connected_first_policy=CONNECTED_FIRST_POLICY,active_point_counts=active_counts,
            connected_priority_series=sorted(priority_series),route_attempts=route_attempts,
            stop_reason='no_admissible_image_candidate' if action=='NONE' else '',
            metric=('independent_pair_seed' if action=='ACTIVATE_PAIR_INIT' else
                    '1-SSIM_add_only_refreshed_route' if seed_phase else
                    'common_local_scene' if action=='RESOLVE_UNTYPED' else
                    'restricted_marker_evidence' if chosen or connected_row is None else '1-SSIM_connected_only'))
        if seed_routes is not None:
            row.update(seed_add_only_phase=seed_phase,legacy_connected_action=legacy_connected_row)
        trace.append(row);history.append((row['iteration'],action,len(active),None,action!='NONE'))
        log_fn(f'[v46 BW Step5] iter={row["iteration"]} {action}, P={len(active)}, S={len(pool)}, U={len(untyped)}')
        if action=='NONE':break
    state.update(P_current=deepcopy(active),S_current=deepcopy(pool),U_current=deepcopy(untyped),connected_state=engine_state,
                 total_iterations=state['total_iterations']+len(trace))
    ssim.save(dest/'untyped_resolution.json',dict(initial_count=len(initial_untyped),remaining_count=len(untyped),
        seconds=untyped_seconds,iterations=untyped_reviews,
        activation_enabled=bool(allow_untyped_activation),
        policy='source_bound_local_scenes_after_typed_correction; no_active_deletion_or_reassignment'))
    ssim.save(dest/'overlap_rechecks.json',overlap_rechecks)
    ssim.save(dest/'filled_boundary_rechecks.json',filled_boundary_rechecks)
    ssim.save(dest/'axis_contact.json',axis_verifier.report)
    if seed_routes is not None:
        seed_report=dict(policy=seed_policy,initial_zero_series=sorted(seed_routes.empty),
            initial_uncertain_series=sorted(seed_routes.uncertain),route_reviews=seed_routes.reviews,
            pair_reviews=seed_routes.pair_reviews,
            original_structure_unchanged=True,extra_routes_addition_only=True)
        ssim.save(dest/'seed_bootstrap.json',seed_report)
    report=dict(version=VERSION,series='all_bw_series',metric='structure_routed',initial_points=initial,
        initial_suppressed=initial_pool,initial_untyped=initial_untyped,
        P_current=active,S_current=pool,U_current=untyped,iterations=trace,
        untyped_resolution_seconds=untyped_seconds,
        overlap_rechecks=overlap_rechecks,
        filled_boundary_rechecks=filled_boundary_rechecks,
        axis_contact=axis_verifier.report,
        connected_first_policy=routing_policy,
        structure=state['structure'],max_iterations=budget,previous_iteration_count=state['total_iterations']-len(trace),
        warning='Source-supported hypotheses, not validated ground truth; no fitted-path y objective.')
    ssim.save(dest/'trace.json',report);ssim.save(dest/'runtime_state.json',state)
    return dict(P_current=active,S_current=pool,U_current=untyped,runtime_state=state,mode_xs=ssim.clustered_grid(active,diameter),
        history=history,trace=trace,diag_imgs=diag,steps=diag,status='completed',artifact_dir=str(dest),
        backend='bw_series_correction_v46',metric='structure_routed',initial_score=None,final_score=None)

"""Joint BW connector geometry with frozen colour marker identities.

Only colour-group sessions with repeated neutral connector evidence opt in.
Original RGB, legend IDs, and scale are retained. No detector, calibration,
reference labels, or per-series black path assignment is run here.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import cv2
import numpy as np
import bw_observed_raster_v46 as R
import bw_layered_composite_v46 as L
from bw_raster_identity_v46 import Config
from bw_raster_identity_v46 import features
from bw_centered_composite_v46 import stroke
from color_group_relative_identity_v46 import Competition
from color_source_observation_v46 import ColourObservation
from color_marker_evidence import _swatch_model
from color_palette_identity_v46 import paper_mixture_evidence
from bw_step5_v46 import key, prepare_reference, save

VERSION = 'colour_markers_joint_neutral_connectors_v1'


def routing(image, state, active):
    """Repeated neutral ink between measured coloured markers, not filename."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    chroma=np.ptp(image.astype(np.float32),axis=2)
    neutral=((gray<190)&(chroma<24)).astype(np.uint8)
    distance=cv2.distanceTransform(1-neutral,cv2.DIST_L2,5)
    a,b,c,d=state['plot_box']
    rows=[]; supported=[]
    for sid,m in state['series_models'].items():
        if np.ptp(m.get('rgb',[0,0,0]))<24:
            continue
        pts=sorted([p for p in active if p['swatch_id']==sid],key=lambda p:p['cx'])
        diameter=float(m.get('effective_diameter',m['source_diameter']))
        pairs=[]
        for p,q in zip(pts,pts[1:]):
            v=np.array([q['cx']-p['cx'],q['cy']-p['cy']]);length=np.linalg.norm(v)
            if v[0]<3*diameter or length<4*diameter:
                continue  # No vertical error bars or marker-body observations.
            xy=np.array([p['cx'],p['cy']])+np.linspace(diameter/length,1-diameter/length,41)[:,None]*v
            x,y=np.rint(xy).astype(int).T
            good=(x>a+2)&(x<c-2)&(y>b+2)&(y<d-2)
            if state['legend_box']:
                x0,y0,x1,y1=state['legend_box'];good &= ~((x>=x0)&(x<x1)&(y>=y0)&(y<y1))
            if good.sum()<20:continue
            fraction=float(np.mean(distance[y[good],x[good]]<=1.5))
            pairs.append(dict(a=[p['cx'],p['cy']],b=[q['cx'],q['cy']],neutral_support=fraction))
        n=sum(p['neutral_support']>=.70 for p in pairs)
        if n>=2 and n>=.6*len(pairs):supported.append(sid)
        rows.append(dict(series=sid,pairs=pairs,supported_pairs=n))
    return dict(version=VERSION,enabled=len(supported)>=2,supported_series=supported,
        reason='repeated_neutral_connectors' if len(supported)>=2 else 'neutral_connectors_not_established',
        series=rows,reference_points_read=False)


class FrozenCompetition(Competition):
    """Reuse common-patch raster/line scoring, without recalibrating markers."""
    def __init__(self,image,models,observation):
        self.cfg=replace(Config(),minimum_visible=.55,maximum_centres=2000)
        self.ready=False;self.banks={};self.cache={};self.observation=observation
        self.ink=(255-cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32))/255
        self.templates={};self.prepared={};self.source_meta={};self.colour_fits={};self.scales={}
        self.report=dict(version=VERSION,status='unresolved_saved_models',config=asdict(self.cfg))
        if image.shape[0]*image.shape[1]>1_000_000 or not models or len(models)>10:
            self.report['status']='resource_limit_preserved';return
        for sid,m in models.items():
            a,b,c,d=map(int,m['legend_box'])
            rec,f=R.extract(cv2.cvtColor(image[b:d,a:c],cv2.COLOR_BGR2GRAY))
            if f is None or rec['diameter']>self.cfg.maximum_diameter:return
            scale=float(m.get('effective_diameter',m['source_diameter']))/rec['diameter']
            if not .4<=scale<=2.:return
            self.templates[sid]=(rec,f);self.scales[sid]=scale
            self.prepared[sid],self.source_meta[sid]=L.prepare_source(rec,f)
            self.colour_fits[sid]=paper_mixture_evidence(image,_swatch_model(image,[a,b,c,d],m.get('rgb')))['fit']
        self.radius=max(L.variants(r,self.prepared[s],self.scales[s],(0.,0.))[3] for s,(r,_) in self.templates.items())
        self.ready=True
        self.report.update(status='ready',scales=self.scales,common_radius=self.radius)


def source_line_null(comp,center,segments,diameter):
    """Fit measured long connectors outside the body, score the held-out body.

    Unlike centered marker+errorbar hypotheses, a line-only null may contain
    two parallel connectors. Both must extend past the body on BOTH sides.
    Source segments, not target marker labels, set their directions/offsets.
    """
    comp._bank();r=comp.radius;size=2*r+1
    xy=np.asarray(center,float);yy,xx=np.indices((size,size),dtype=float)
    outer=np.hypot(xx-r,yy-r)>=.60*diameter
    patch=cv2.getRectSubPix(comp.ink,(size,size),tuple(map(float,xy)))
    valid=1-cv2.getRectSubPix(comp.observation.other,(size,size),tuple(map(float,xy)))
    weight=outer*valid
    candidates=[]
    for index,s in enumerate(segments):
        a=np.asarray(s[:2]);v=np.asarray(s[2:])-a;length=np.linalg.norm(v)
        if length<2.5*diameter:continue
        t=v/length;projection=float((xy-a)@t);normal=np.array([-t[1],t[0]])
        offset=float((a-xy)@normal)
        if projection<.6*diameter or length-projection<.6*diameter or abs(offset)>.6*diameter:continue
        angle=float(np.degrees(np.arctan2(t[1],t[0])))
        best=None
        for shift in (-.5,0.,.5):
            for width in (.8,1.2,1.6,2.):
                line=stroke(size,(r+normal[0]*(offset+shift),r+normal[1]*(offset+shift)),angle,width)
                norm=float(np.sum(weight*line*line))
                if norm<3:continue
                amplitude=float(np.clip(np.sum(weight*line*patch)/norm,0,1))
                rendered=line*amplitude
                loss=float(np.sum(weight*(patch-rendered)**2))
                if best is None or loss<best[0]:best=(loss,rendered,dict(segment=index,angle=angle,width=width,offset=offset+shift,amplitude=amplitude))
        if best is not None and best[2]['amplitude']>.15:candidates.append(best)
    from bw_local_line_null_v46 import fit as fit_local_lines
    local=fit_local_lines(comp,center,segments,diameter)
    if not candidates:return local
    candidates=sorted(candidates,key=lambda q:q[0])[:6]
    variants=[(q[1],[q[2]]) for q in candidates]
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            # Distinct observed strokes, not duplicate edges of one connector.
            if abs(a[2]['angle']-b[2]['angle'])<5 and abs(a[2]['offset']-b[2]['offset'])<.8:continue
            variants.append((1-(1-a[1])*(1-b[1]),[a[2],b[2]]))
    # Select on the outer region only; the marker body is held out of this fit.
    rendered,pars=min(variants,key=lambda q:float(np.sum(weight*(patch-q[0])**2)))
    f=features(patch,comp.weight,comp.cfg.boundary_weight)[0]
    ff=features(rendered,comp.weight,comp.cfg.boundary_weight)[0]
    conf=np.r_[valid.ravel(),np.minimum(valid[1:],valid[:-1]).ravel(),np.minimum(valid[:,1:],valid[:,:-1]).ravel()]
    loss=float(np.sum(conf*(f-ff)**2)/max(float(np.sum(conf*f*f)),1.e-6)+len(pars)*comp.cfg.line_cost)
    result=dict(loss=loss,lines=pars,fit_region='outside_marker_body',source_segments_only=True)
    return local if local is not None and local['loss']<loss else result


class MarkerEvidence:
    def __init__(self,image,state,points,segments=(),*,duplicate_policy='x_slot'):
        from color_group_runtime_v46 import unpack
        self.cache={};self.reports=[]
        if duplicate_policy not in ('x_slot','body'):raise ValueError('Unknown marker duplicate policy')
        self.duplicate_policy=duplicate_policy
        self.original={key(p):p for p in state['original_points']}
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        contrast=255-gray.astype(np.float32)
        chroma=np.ptp(image.astype(np.float32),axis=2)
        # Neutral markers may use neutral ink, including black connectors;
        # line-only alternatives must then explain those connectors explicitly.
        neutral_own=np.clip(contrast/24,0,1)*np.exp(-.5*(chroma/18)**2)
        other=np.clip((chroma-18)/24,0,1)*np.clip(contrast/24,0,1)
        paper=np.clip((16-contrast)/10,0,1)
        for group in state['groups']:
            names=group['names'];models={s:state['series_models'][s] for s in names}
            ownpoints=[p for p in points if p['swatch_id'] in names]
            if not ownpoints:continue
            roles=group.get('role_maps')
            if roles:
                obs=ColourObservation(image,*[unpack(roles[k],gray.shape).astype(np.float32)/255 for k in ('own','other','paper')])
            elif all(np.ptp(m.get('rgb',[0,0,0]))<24 for m in models.values()):
                obs=ColourObservation(image,neutral_own,other,paper)
            else:
                self.reports.append(dict(group=group['id'],status='missing_colour_roles_preserved'))
                continue
            comp=FrozenCompetition(image,models,obs)
            self.reports.append(dict(group=group['id'],**comp.report))
            if not comp.ready:continue
            centers=list(dict.fromkeys((float(p['cx'])+dx,float(p['cy'])+dy) for p in ownpoints for dx,dy in [(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)]))
            if len(centers)>comp.cfg.maximum_centres:
                self.reports[-1]['status']='marker_budget_preserved';continue
            lookup={}
            for i in range(0,len(centers),32):lookup.update(zip(centers[i:i+32],comp.score(centers[i:i+32])))
            neutral_group=all(np.ptp(m.get('rgb',[0,0,0]))<24 for m in models.values())
            for p in ownpoints:
                sid=p['swatch_id'];xy=(float(p['cx']),float(p['cy']))
                candidates=[lookup[(xy[0]+dx,xy[1]+dy)] for dx,dy in [(0,0),(-.5,0),(.5,0),(0,-.5),(0,.5)]]
                own=[deepcopy(q['scores'][sid]) for q in candidates]
                nulls=[]
                if neutral_group:
                    for q,s in zip(candidates,own):
                        null=source_line_null(comp,q['center'],segments,float(models[sid].get('effective_diameter',models[sid]['source_diameter'])))
                        nulls.append(null)
                        if null:
                            s['line_improvement']=min(s['line_only_loss'],null['loss'])-s.get('geometry_loss',s['loss'])
                            s['passed']=s['passed'] and s['line_improvement']>=comp.cfg.minimum_improvement
                exact=lookup[xy]
                m=state['series_models'][sid];rad=max(1,int(m.get('effective_diameter',m['source_diameter'])/3))
                occ=cv2.getRectSubPix(obs.other,(2*rad+1,2*rad+1),xy)
                supported=any(s['passed'] or (s['line_improvement']>.04 and s['independent_visible']>.40 and s.get('geometry_loss',s['loss'])<.70) for s in own)
                # Prior reviewed positive evidence protects unchanged detections,
                # never promotes an old suppressed hypothesis automatically.
                original=self.original.get(key(p),{})
                prior=original.get('relative_identity',{})
                prior_good=bool(prior.get('winner') and prior.get('scores',{}).get(prior['winner'],{}).get('passed')
                    and np.linalg.norm(np.array(prior.get('center',xy))-xy)<.01)
                # Neutral ink includes other series' black connectors. Require
                # a strictly better line-only model at EVERY subpixel probe,
                # not merely absence of a decisive marker win. Near ties are
                # uncertainty (e.g. a genuine tiny marker on its own line).
                line_only=(all(s['line_improvement']<-.02 for s in own) if neutral_group else
                    all(s['line_improvement']<.02 or s['independent_visible']<.25 for s in own))
                from marker_confidence_v46 import assess
                confidence=(assess(own,occluded_fraction=float(occ.mean()),
                    minimum_improvement=comp.cfg.minimum_improvement,
                    maximum_loss=comp.cfg.maximum_loss,
                    minimum_visible=comp.cfg.minimum_visible,
                    minimum_independent_fraction=comp.cfg.minimum_independent_fraction)
                    if neutral_group else None)
                self.cache[key(p)]=dict(known=True,supported=bool(supported or prior_good),
                    prior_detection_supported=prior_good,occluded_fraction=float(occ.mean()),
                    removable=bool(line_only and not supported and not prior_good and occ.mean()<.35),
                    addable=bool(exact['winner']==sid and own[0]['passed']),
                    exact=exact,source_line_nulls=nulls,marker_confidence=confidence,
                    neighbor_best_improvement=max(s['line_improvement'] for s in own))

    def measure(self,p):
        return self.cache.get(key(p),dict(known=False,supported=False,removable=False,addable=False))

    def update_confidence(self,points):
        from marker_confidence_v46 import update_point
        return [update_point(p,self.measure(p).get('marker_confidence')) for p in points]

    def guard(self,trial,before):
        checks=[]
        for p in trial['removed']:
            e=self.measure(p);checks.append(dict(operation='remove',point=key(p),evidence=e))
            if p.get('manual_edit') or not e['removable']:
                return dict(admissible=False,reason='preserve_observed_or_uncertain_marker',checks=checks)
        for p in trial['added']:
            e=self.measure(p);checks.append(dict(operation='add',point=key(p),evidence=e))
            if not e['addable']:
                return dict(admissible=False,reason='missing_independent_colour_marker_identity',checks=checks)
            # Suppressed points may predate the raster recovery; do not add a
            # second sample beside an existing marker at a slightly shifted x.
            diameter=float(p.get('effective_diameter',p['source_diameter']))
            if any(q['swatch_id']==p['swatch_id'] and key(q)!=key(p) and
                   (np.hypot(q['cx']-p['cx'],q['cy']-p['cy']) if getattr(self,'duplicate_policy','x_slot')=='body'
                    else abs(q['cx']-p['cx']))<.65*diameter for q in trial['points']):
                return dict(admissible=False,reason='observed_series_x_slot_occupied',checks=checks)
        return dict(admissible=True,reason='independent_marker_evidence',checks=checks)


def filter_reference(segments,diameter):
    kept=[];rejected=[]
    for s in segments:
        dx,dy=s[2]-s[0],s[3]-s[1];length=float(np.hypot(dx,dy))
        reason='vertical_errorbar_or_stem' if abs(dy)>4*abs(dx) else 'short_body_fragment' if length<.65*diameter else ''
        (rejected if reason else kept).append(dict(segment=s,reason=reason) if reason else s)
    return kept,rejected


def review_neutral_detections(image,state,points,pool):
    """Defer line-explained neutral detections, retaining every hypothesis.

    Only charts with independently established neutral connectors opt in.
    Chromatic identity, measured scale, and marker positions are unchanged.
    No post-hoc reference labels or curve-smoothness heuristic are consulted.
    """
    import time
    start=time.perf_counter()
    route=routing(image,state,points)
    report=dict(version='neutral_initial_body_evidence_v1',route=route,
        status='not_applicable',before=len(points),after=len(points),deferred=[])
    neutral=[p for p in points if np.ptp(state['series_models'][p['swatch_id']].get('rgb',[0,0,0]))<24]
    if not route['enabled'] or not neutral:return points,pool,report
    if image.shape[0]*image.shape[1]>1_000_000 or len(neutral)>100:
        report['status']='resource_limit_preserved'
        return points,pool,report
    diameter=float(np.median([g['diameter'] for g in state['groups']]))
    _,reference=prepare_reference(image,state['plot_box'],state['legend_box'],points,diameter,recover_dashes=True)
    segments,_=filter_reference(reference['reference_segments'],diameter)
    evidence=MarkerEvidence(image,state,neutral,segments)
    kept=[];suppressed=deepcopy(pool);present={key(p) for p in suppressed}
    rows=[]
    neutral_keys={key(p) for p in neutral}
    for p in points:
        if key(p) not in neutral_keys:
            kept.append(p);continue
        e=evidence.measure(p)
        q=evidence.update_confidence([p])[0]
        q['neutral_body_evidence']=e
        rows.append(dict(point=list(key(p)),evidence=e))
        if not e['removable'] or q.get('manual_edit'):
            kept.append(q);continue
        q.update(candidate_id='LINE_'+q['point_id'],state='suppressed',
            original_detection=False,suppression_reason='independent_connectors_explain_body',
            prior_detector_confidence=q.get('detector_confidence',q.get('confidence')))
        if key(q) not in present:suppressed.append(q);present.add(key(q))
        report['deferred'].append(dict(point=list(key(q)),prior_confidence=q['prior_detector_confidence']))
    report.update(status='completed',after=len(kept),marker_models=evidence.reports,
        measurements=rows,reference_points_read=False,existing_coordinates_changed=False,
        seconds=time.perf_counter()-start)
    return kept,suppressed,report


def run(image,state,active,pool,iterations,folder,route):
    from group_element_correction_v46 import run as run_elements
    runtime=deepcopy(state.get('joint_neutral_runtime'))
    diameter=float(np.median([g['diameter'] for g in state['groups']]))
    if runtime is not None and runtime.get('version')!=VERSION:
        raise ValueError('Incompatible joint neutral Step-5 state')
    if runtime is None:
        _,ref=prepare_reference(image,state['plot_box'],state['legend_box'],state['original_points'],diameter,recover_dashes=True)
        segments,rejected=filter_reference(ref['reference_segments'],diameter)
        runtime=dict(version=VERSION,total_iterations=0,reference_segments=segments,
            reference_rejected=rejected,raw_reference=ref,route=route)
    evidence=MarkerEvidence(image,state,active+pool,runtime['reference_segments'])
    # The same measured body confidence travels to displayed/saved points and
    # the removal prior. No binary zero overwrite merely because a guard passed.
    active=evidence.update_confidence(active)
    pool=evidence.update_confidence(pool)
    originals=evidence.update_confidence(state['original_points'])
    result=run_elements(active,pool,runtime['reference_segments'],diameter,
        state['plot_box'],state['legend_box'],max_iter=iterations,original_points=originals,
        action_guard=evidence.guard)
    previous=runtime['total_iterations']
    for row in result['trace']:row['iteration']+=previous
    runtime['total_iterations']+=len(result['trace'])
    dest=folder/'group_step5'/'joint_neutral'/f'run_{state["total_correction_runs"]+1:03d}'
    dest.mkdir(parents=True,exist_ok=True)
    trace=dict(result,iterations=result['trace'],series='all_colour_series_joint_neutral',
        metric=VERSION,previous_iteration_count=previous,
        stop_reason='unavailable_reference_preserved' if result.get('reason') else '',
        marker_models=evidence.reports,reference_policy='original_gray_joint_unlabelled',
        colour_ids_preserved=True,detector_executed=False,
        confidence_policy='marker_vs_line_confidence_v1',
        confidence_scope='measurable neutral groups; chromatic group scores unchanged',
        marker_evidence=[dict(point=list(k),**v) for k,v in evidence.cache.items()])
    save(dest/'trace.json',trace)
    state.update(joint_neutral_runtime=runtime,P_current=result['final'],S_current=result['suppressed'],
        original_points=originals,
        total_correction_runs=state['total_correction_runs']+1)
    print(f'[v46 joint neutral Step5] {[r["action"] for r in result["trace"]]}; {len(active)} -> {len(result["final"])}',flush=True)
    return trace

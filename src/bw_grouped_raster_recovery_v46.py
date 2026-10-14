"""Bounded body-grouped recovery and evidence-backed identity replacement.

Reuse the final native raster competitor, its measured scales and common loss.
No independent second definition of a winning symbol. Budget exhaustion keeps
completed batches; it never silently drops the whole recovery stage.
"""
from dataclasses import dataclass,asdict
import math,time
import numpy as np
import bw_observed_raster_v46 as R
import bw_layered_composite_v46 as L
from bw_suppressed_v46 import encode_marker_mask

VERSION='body-grouped-observed-raster-recovery-v1'


@dataclass(frozen=True)
class Config:
    proposals_per_series: int = 250
    centres_per_body: int = 3
    maximum_bodies: int = 1800
    batch_centres: int = 64
    maximum_seconds: float = 60.
    minimum_replace_margin: float = .10
    maximum_replace_loss: float = .35
    minimum_replace_visible: float = .80
    minimum_replace_improvement: float = .10


def group_proposals(proposals,cfg=Config()):
    """No transitive chaining along curves; group to a fixed body anchor."""
    groups=[];buckets={};cell=8.
    for p in sorted(proposals,key=lambda p:(p['loss'],p['x'],p['y'])):
        x,y=p['x'],p['y'];bx,by=int(x//cell),int(y//cell)
        candidates=[i for xx in range(bx-1,bx+2) for yy in range(by-1,by+2) for i in buckets.get((xx,yy),[])]
        close=[i for i in candidates if math.hypot(x-groups[i]['x'],y-groups[i]['y'])<=
               max(1.5,.35*min(p['diameter'],groups[i]['diameter']))]
        if close:
            g=min((groups[i] for i in close),key=lambda g:math.hypot(x-g['x'],y-g['y']))
            g['proposal_count']+=1
            if (len(g['centres'])<cfg.centres_per_body and
                    all(math.hypot(x-q[0],y-q[1])>=.45 for q in g['centres'])):
                g['centres'].append((x,y))
        elif len(groups)<cfg.maximum_bodies:
            buckets.setdefault((bx,by),[]).append(len(groups))
            groups.append(dict(x=x,y=y,diameter=p['diameter'],loss=p['loss'],centres=[(x,y)],proposal_count=1))
    return groups


def replacement_allowed(comparison,old_key,new_key,cfg=Config()):
    if comparison['winner']!=new_key or old_key not in comparison['scores']:return False
    new=comparison['scores'][new_key];old=comparison['scores'][old_key]
    return bool(new['passed'] and old['loss']-new['loss']>=cfg.minimum_replace_margin and
                new['loss']<=cfg.maximum_replace_loss and new['independent_visible']>=cfg.minimum_replace_visible
                and new['line_improvement']>=cfg.minimum_replace_improvement)


def recover(competition,plot,exclusions,native_templates,kept,records,series_indices,
            cfg=Config(),log_fn=print,minimum_visible=None,source_tone_guard=None,
            window_guard=None):
    start=time.perf_counter()
    report=dict(version=VERSION,status='not_applicable',added=0,seconds=0.,accepted=[],rejected=[],
                replacements=[],replaced_point_ids=[],config=asdict(cfg),reference_points_read=False,
                source_pixels_modified=False,native_points_preserved=True,scoring=competition.report['version'])
    if not competition.ready:
        report['status']='raster_competitor_unavailable';return [],report
    shared_source_mode = competition.report.get('source_model_policy') == 'shared_native_window'
    ink=competition.ink;allowed=np.zeros(ink.shape,bool);a,b,c,d=plot;allowed[b:d,a:c]=True
    for a,b,c,d in exclusions:allowed[max(0,b):max(0,d),max(0,a):max(0,c)]=False
    proposals=[];clipped=[]
    for key,(rec,fields) in competition.templates.items():
        scale=competition.scales[key]
        if scale is None:continue  # Rival hypotheses do not fabricate active sizes.
        loss,px,py=R.evaluate_scale(ink,rec,fields,scale,R.Config())
        pp=R.local_candidates(loss,px,py,rec['diameter']*scale,allowed,.85,cap=cfg.proposals_per_series)
        proposals.extend(dict(x=p['x'],y=p['y'],loss=p['loss'],diameter=rec['diameter']*scale,series_id=key) for p in pp)
        if len(pp)==cfg.proposals_per_series:clipped.append(key)
    for p in records:
        x,y=float(p['aligned_x']),float(p['aligned_y'])
        if not (0<=round(x)<allowed.shape[1] and 0<=round(y)<allowed.shape[0] and allowed[round(y),round(x)]):continue
        proposals.append(dict(x=x,y=y,loss=max(0.,1.-float(p.get('score',0.))),diameter=p.get('effective_diameter',10.)))
    if shared_source_mode:
        # Native windows observe integer-centered source pixels. Do not let a
        # half-pixel interpolated recovery score bypass that same observation.
        # Both freshly proposed and saved-record centers use this convention.
        proposals = [dict(p,x=float(round(p['x'])),y=float(round(p['y']))) for p in proposals]
        proposals = [p for p in proposals if
            0 <= p['x'] < allowed.shape[1] and 0 <= p['y'] < allowed.shape[0]
            and allowed[int(p['y']),int(p['x'])]]
    groups=group_proposals(proposals,cfg)
    centers=list(dict.fromkeys(q for g in groups for q in g['centres']))
    report.update(proposals=len(proposals),body_groups=len(groups),candidate_centres=len(centers),
                  proposal_cap_series=clipped,body_cap_reached=len(groups)==cfg.maximum_bodies,
                  candidates_by_series={},grouping=groups)
    scored={};batches=0
    for start_index in range(0,len(centers),cfg.batch_centres):
        if time.perf_counter()-start>cfg.maximum_seconds:break
        batch=centers[start_index:start_index+cfg.batch_centres]
        scorer=getattr(competition,'score_cached',competition.score)
        for center,result in zip(batch,scorer(batch)):scored[center]=result
        batches+=1
    # Within a body, compare all rival identities at each same center. Then
    # select the best supported localization. Near ties stay diagnostic-only.
    proposals=[]
    for i,g in enumerate(groups):
        choices=[scored[c] for c in g['centres'] if c in scored]
        good=[q for q in choices if q['winner'] is not None and
              (minimum_visible is None or q['scores'][q['winner']]['independent_visible']>=minimum_visible)]
        if not good:
            if choices:report['rejected'].append(dict(body_id=i,reason='unresolved_body',comparisons=choices))
            continue
        best=min(good,key=lambda q:q['scores'][q['winner']]['loss'])
        key=best['winner'];s=best['scores'][key];x,y=best['center']
        if competition.scales[key] is None:
            report['rejected'].append(dict(body_id=i,reason='unresolved_winner_scale',comparison=best));continue
        selected_scale=float(s['parameters'].get('raster_scale',competition.scales[key]))
        proposals.append(dict(series_id=key,x=x,y=y,loss=s['loss'],diameter=competition.templates[key][0]['diameter']*selected_scale,
                              selected_raster_scale=selected_scale,
                              body_id=i,comparison=best,**{k:s[k] for k in ('independent_visible','line_improvement','parameters')}))
    additions=[];by_key={t.key:t for t in native_templates};removed=set()
    classes=['filled_circle','open_circle','filled_square','open_square','open_triangle','open_inv_triangle',
             'filled_triangle','filled_inv_triangle','open_rhombus','filled_rhombus','x_marker','plus_marker','unknown_marker']
    existing_ids={p.get('point_id') for p in kept}
    for p in sorted(proposals,key=lambda p:p['loss']):
        key=p['series_id'];t=by_key[key]
        if t.name=='x_marker':continue  # Preserve dedicated X recovery.
        if source_tone_guard is not None:
            # Another series' detected marker may legitimately cover part of
            # this hole. Its measured footprint is excluded, never re-scored.
            occupied=[q for q in [*kept,*additions] if q.get('point_id') not in removed and
                      q.get('swatch_id')!=key and q.get('marker_mask') and
                      math.hypot(q['cx']-p['x'],q['cy']-p['y'])<.5*(p['diameter']+q.get('effective_diameter',p['diameter']))+1]
            tone=source_tone_guard(key,p['x'],p['y'],p['diameter'],occupied)
            p['source_open_tone']=tone
            if tone.get('decision')=='conflict':
                report['rejected'].append(dict(p,reason='observed_source_hole_conflict'));continue
        from bw_raster_context_v46 import evidence,native_body_supported,selected_body_fill
        bank=competition.banks[key]
        variant=bank.get('variants',{}).get(p['selected_raster_scale'])
        if variant is None:
            report['rejected'].append(dict(p,reason='selected_raster_variant_unavailable'));continue
        marker,cover,marker_weight=variant
        context=evidence(ink,(p['x'],p['y']),marker,cover,p['parameters'],competition.weight)
        p['context_evidence']=context
        # New exterior explanations cannot erase a source-body contradiction.
        # An old nearby verdict is not itself a veto: remeasure the exact
        # selected raster scale/center, using the existing fill thresholds.
        contradictions=[q for q in records if q.get('template')==key and
            q.get('geometry_first',{}).get('decision')=='conflict' and
            q.get('geometry_first',{}).get('reason')=='fill_distribution_mismatch' and
            q.get('geometry_first',{}).get('style') in ('solid','open') and
            math.hypot(q.get('aligned_x',q.get('x',math.inf))-p['x'],q.get('aligned_y',q.get('y',math.inf))-p['y'])<=1.]
        if context['passed'] and contradictions:
            previous=evidence(ink,(p['x'],p['y']),marker,cover,p['parameters'],competition.weight,
                              finite_exterior=False)
            p['previous_centered_context']=previous
            if not previous['historical_context_passed']:
                native=min(contradictions,key=lambda q:math.hypot(
                    q.get('aligned_x',q.get('x',math.inf))-p['x'],q.get('aligned_y',q.get('y',math.inf))-p['y']))
                tone=selected_body_fill(ink,(p['x'],p['y']),marker,cover,marker_weight,
                                       p['parameters'],native['geometry_first']['style'])
                p['selected_body_fill']=dict(tone,requesting_candidate_index=native.get('candidate_index'),
                    native_scale_not_reused=True)
                if tone['decision']=='conflict':
                    report['rejected'].append(dict(p,reason='finite_context_cannot_override_body_fill'));continue
        native_body=native_body_supported(records,key,(p['x'],p['y']),p['diameter'])
        if native_body:p['native_body_evidence']=native_body
        if not context['passed'] and native_body is None:
            report['rejected'].append(dict(p,reason='unsupported_composite_context'));continue
        conflicts=[q for q in [*kept,*additions] if q.get('point_id') not in removed and
                   math.hypot(q['cx']-p['x'],q['cy']-p['y'])<.55*min(p['diameter'],q.get('effective_diameter',p['diameter']))]
        replacement=None
        if conflicts:
            if len(conflicts)!=1 or conflicts[0] in additions:
                report['rejected'].append(dict(p,reason='multiple_or_recovery_body_conflict'));continue
            q=conflicts[0];old_key=q['swatch_id']
            if old_key==key:
                report['rejected'].append(dict(p,reason='same_swatch_existing_body'));continue
            # Different centers may be real overlapping markers. Never replace
            # merely because two footprints touch or a new score is higher.
            same=math.hypot(q['cx']-p['x'],q['cy']-p['y'])<=.25*min(p['diameter'],q.get('effective_diameter',p['diameter']))
            at_old=competition.score([(q['cx'],q['cy'])])[0] if same else None
            if at_old is None or not replacement_allowed(p['comparison'],old_key,key,cfg) or not replacement_allowed(at_old,old_key,key,cfg):
                report['rejected'].append(dict(p,reason='replacement_not_independently_supported'));continue
            replacement=dict(old_point=dict(q),new_series_id=key,comparison_at_old_center=at_old)
        guard = None
        if shared_source_mode:
            # The raster comparator no longer constitutes an independent
            # acceptance route. A recovery/replacement must pass the same
            # physical window constraints before any old point is retired.
            guard = (dict(decision='rejected',reason='shared_window_guard_unavailable')
                     if window_guard is None else
                     window_guard(key,(p['x'],p['y']),p['comparison']))
            p['shared_window_guard'] = guard
            if guard.get('decision') != 'verified':
                report['rejected'].append(dict(p,reason='shared_window_recovery_guard'))
                continue
        rec=competition.templates[key][0]
        target,weight,cover,r=(competition._variant(key,p['selected_raster_scale']) if shared_source_mode else
            L.variants(rec,competition.prepared[key],p['selected_raster_scale'],(0.,0.)))
        mask=(target>max(.03,.35*target.max()))&(weight>.2)
        if not mask.any():continue
        point_id=f'GR{len(additions)+1:03d}'
        while point_id in existing_ids:point_id+='_r'
        existing_ids.add(point_id)
        q=dict(class_name=t.name,shape_hint=t.shape_hint or t.name,template=key,swatch_id=t.swatch_id,
               class_idx=series_indices[key],shape_idx=classes.index(t.name),cx=p['x'],cy=p['y'],
               confidence=p['independent_visible'],source=VERSION,point_id=point_id,original_detection=True,
               marker_mask=encode_marker_mask(mask),marker_offset_x=0.,marker_offset_y=0.,
               marker_scale=p['diameter']/float(getattr(t,'diameter',rec['diameter'])),marker_aspect=1.,
               source_diameter=float(getattr(t,'diameter',rec['diameter'])),
               observed_raster_scale=p['selected_raster_scale'],observed_raster_diameter=rec['diameter'],
               effective_diameter=p['diameter'],raster_identity=p['comparison'],layered_composite=p)
        if guard is not None:
            q['shared_window_guard'] = guard
            q['joint_window'] = dict(guard.get('joint_window') or guard,recovery_guard_verified=True)
        if replacement:
            removed.add(replacement['old_point']['point_id']);replacement['new_point_id']=point_id
            report['replacements'].append(replacement)
        additions.append(q);report['accepted'].append(p)
    report.update(status='completed' if len(scored)==len(centers) else 'partial_budget',
                  scored_centres=len(scored),completed_batches=batches,
                  added=len(additions),replaced_point_ids=sorted(removed),native_points_preserved=not removed,
                  seconds=time.perf_counter()-start)
    return additions,report

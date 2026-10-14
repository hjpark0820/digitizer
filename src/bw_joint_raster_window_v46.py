"""One native-BW full-window localization and observed-raster decision.

The grid supplies locations, never an authoritative identity. Every tested
center uses the same observed patch for all legend rasters and the line null.
No legacy recall/fill verdict is required before this comparison, and no
second raster filter or native-identity exemption follows it.
"""
from dataclasses import dataclass, asdict
import math
import time
import cv2
import numpy as np
from bw_raster_context_v46 import evidence as context_evidence

VERSION = 'unified-observed-raster-window-v1'


@dataclass(frozen=True)
class Config:
    maximum_centres: int = 6000
    batch_centres: int = 64
    maximum_seconds: float = 60.
    maximum_shift: int = 4
    shift_fraction: float = .16
    shift_cost: float = .004


class ResourceLimit(RuntimeError):
    """Atomic fallback to legacy windows; never return a partial active set."""


def _inside(center, plot, exclusions):
    x,y = center
    return (plot[0] <= x < plot[2] and plot[1] <= y < plot[3] and
            not any(a <= x < c and b <= y < d for a,b,c,d in exclusions))


def verify_candidates(competition, image, proposals, templates, plot, exclusions,
                      *, minimum_recall=None, cfg=Config()):
    """Return records and measured masks from one coherent window decision.

    Budget exhaustion raises before the caller mutates its candidate records.
    Scores cached during an incomplete attempt remain valid for fallback.
    Unknown-scale rivals compete, but cannot supply an active calibrated size.
    """
    if not competition.ready:
        raise ValueError('Joint windows require a ready observed-raster comparator')
    if cfg.maximum_centres < 1 or cfg.batch_centres < 1 or cfg.maximum_seconds <= 0:
        raise ValueError('Joint window budgets must be positive')
    started=time.perf_counter()
    by_key={t.key:t for t in templates}
    neighborhoods=[]
    for p in proposals:
        key=p['template'];rec=competition.templates[key][0]
        scales=competition.scale_hypotheses[key]
        diameter=rec['diameter']*max(scales)
        shift=min(cfg.maximum_shift,max(2,round(cfg.shift_fraction*diameter)))
        x,y=float(p['x']),float(p['y'])
        if not math.isfinite(x+y):
            raise ValueError('Candidate centers must be finite')
        neighborhoods.append([(x+dx,y+dy) for dy in range(-shift,shift+1)
            for dx in range(-shift,shift+1) if _inside((x+dx,y+dy),plot,exclusions)])
    centers=list(dict.fromkeys(c for row in neighborhoods for c in row))
    if len(centers)>cfg.maximum_centres:
        raise ResourceLimit(f'{len(centers)} joint centers exceed budget {cfg.maximum_centres}')
    scored={}
    for i in range(0,len(centers),cfg.batch_centres):
        if time.perf_counter()-started > cfg.maximum_seconds:
            raise ResourceLimit('Joint window time budget exceeded')
        batch=centers[i:i+cfg.batch_centres]
        scored.update(zip(batch,competition.score_cached(batch)))
    if time.perf_counter()-started > cfg.maximum_seconds:
        raise ResourceLimit('Joint window time budget exceeded')
    records=[];verified=[];measured={};stats=[]
    for index,(p,centers_for_seed) in enumerate(zip(proposals,neighborhoods)):
        if time.perf_counter()-started > cfg.maximum_seconds:
            raise ResourceLimit('Joint window time budget exceeded during export')
        key=p['template'];template=by_key[key]
        record=dict(p,candidate_index=index,swatch_id=template.swatch_id,
            class_name=template.name,shape_hint=template.shape_hint or template.name,
            selected=False,matching_profile=VERSION,geometry_fixed=True,
            window_scale_reference='observed_legend_raster',scale_policy='observed_shared_symbol',
            aspect_ratio=1.,window_decision_source=VERSION)
        if not centers_for_seed:
            record.update(aligned_x=float(p['x']),aligned_y=float(p['y']),decision='rejected',
                          exclusion_reason='aligned_centre_outside_plot')
            records.append(record);continue
        # The proposal supplies a location, not an identity lock. Localize the
        # best all-legend explanation, comparing rivals at EACH shared center.
        # Do not choose a worse fit solely because it passes an identity gate.
        q=min((scored[c] for c in centers_for_seed),key=lambda q:(
            min(s['loss'] for s in q['scores'].values())+cfg.shift_cost*math.hypot(q['center'][0]-p['x'],q['center'][1]-p['y']),
            math.hypot(q['center'][0]-p['x'],q['center'][1]-p['y'])))
        # A decisive winner may relabel the seed. On an unresolved identity,
        # retain each proposal's typed alternative instead of collapsing all
        # plausible rivals into whichever key happens to have the lowest loss.
        key=q['winner'] if q['winner'] is not None else p['template']
        template=by_key[key]
        record.update(proposal_template=p['template'],template=key,swatch_id=template.swatch_id,
                      class_name=template.name,shape_hint=template.shape_hint or template.name)
        s=q['scores'][key];x,y=map(float,q['center']);pars=s['parameters']
        marker,cover,reliability=competition.banks[key]['variants'][pars['raster_scale']]
        diameter=competition.templates[key][0]['diameter']*pars['raster_scale']
        mask=(marker>max(.03,.35*float(marker.max()))) & (reliability>.2)
        context=context_evidence(competition.ink,(x,y),marker,cover,pars,
            competition.weight)
        patch=cv2.getRectSubPix(competition.ink,(marker.shape[1],marker.shape[0]),(x,y))
        expected=np.clip(marker*pars['gain'],0,1)
        support=np.clip(patch/np.maximum(expected,1.e-6),0,1)
        weight=np.where(mask,reliability*marker,0.)
        recall=float(np.sum(support*weight)/max(float(weight.sum()),1.e-8))
        core=mask & (expected>=.95) & (reliability>=.85)
        boundary=mask & ~core
        core_recall=float(support[core].mean()) if core.any() else None
        boundary_recall=float(support[boundary].mean()) if boundary.any() else None
        # Exterior line support is already mandatory in Competition.score().
        # A differently weighted residual must not become a second body veto.
        supported=bool(s['passed'] and mask.any())
        measured_scale=competition.scales[key] is not None
        active=supported and measured_scale and q['winner']==key
        reason=(None if active else 'identity_ambiguous' if supported and measured_scale
                else 'unresolved_raster_scale' if supported else 'raster_marker_not_supported')
        record.update(aligned_x=x,aligned_y=y,window_scale=diameter/template.diameter,
            observed_raster_scale=pars['raster_scale'],observed_raster_diameter=competition.templates[key][0]['diameter'],
            effective_diameter=diameter,scale_status='measured' if measured_scale else 'unresolved',
            decision='verified' if supported and measured_scale else 'ambiguous' if supported else 'rejected',
            required_recall=recall,weighted_required_recall=recall,strict_core_recall=core_recall,
            boundary_recall=boundary_recall,contour_support=None,minimum_sector_recall=None,
            shape_support=None,missing_fraction=1.-recall,extra_fraction=context.get('extra_ink_fraction'),
            visible_fraction=s['independent_visible'],occluded_fraction=None,contradiction_fraction=None,
            selection_rank=1.-s['loss'],base_selection_rank=1.-s['loss'],
            raster_identity=q,joint_window=dict(version=VERSION,context=context,context_diagnostic_only=True,
                center_count=len(centers_for_seed),legacy_window_used=False,second_identity_filter=False,
                localization_loss=s['loss']+cfg.shift_cost*math.hypot(x-p['x'],y-p['y']),
                marker_support_passed=supported,identity_resolved=active),
            ink_evidence=dict(version=VERSION,direct_required_recall=recall,
                tolerant_required_recall=recall,borrowed_boundary_credit=0.,
                strict_core_pixels=int(core.sum()),measured_strict_core_recall=core_recall,
                strict_core_status='measured' if core.any() else 'unmeasurable',
                confidence_kind='direct_required_ink_not_probability'))
        if reason:record['exclusion_reason']=reason
        if minimum_recall is not None and recall<minimum_recall:
            active=False;record.update(exclusion_reason='confidence_floor',excluded_by_confidence_floor=True)
        # Export only measured source ink. Neither nuisance lines nor an opaque
        # white interior may be exported as positive marker-mask pixels.
        measured[index]=dict(mask=mask,marker_offset_x=0.,marker_offset_y=0.)
        yy,xx=np.nonzero(mask)
        if len(xx)>=3:
            hull=cv2.convexHull(np.column_stack((xx,yy)).astype(np.float32)).reshape(-1,2)
            hull+=np.array([x-marker.shape[1]//2,y-marker.shape[0]//2],np.float32)
            record['footprint_polygon']=hull.tolist()
        records.append(record)
        if active:verified.append(record)
        stats.append(dict(swatch_id=key,x=float(p['x']),y=float(p['y']),backend='cpu',
                          used_cuda=False,joint_window=record['joint_window']))
    report=dict(version=VERSION,status='completed',config=asdict(cfg),candidates=len(proposals),
        centres=len(scored),retained=len(verified),seconds=time.perf_counter()-started,
        single_decision=True,legacy_window_used=False,second_identity_filter=False,
        scoring=competition.report['version'],reference_points_read=False,source_pixels_modified=False)
    return dict(records=records,verified=verified,measured_geometry=measured,window_stats=stats,report=report)

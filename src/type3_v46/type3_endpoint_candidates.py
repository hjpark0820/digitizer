"""Test whole-path endpoints using source pixels, not endpoint necessity alone.

The prior is a line-only chart connecting measurements. A cropped curve or a
retained path fragment does not establish a terminal measurement. All coordinates
in output are original-image pixels; masks/stems supplied here are plot-local.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import numpy as np

POLICY = 'type3_source_endpoint_v1'
ENDPOINT_PRIOR_POLICY = 'global_endpoint_strong_prior_v1'


@dataclass(frozen=True)
class EndpointConfig:
    force_suppressed_endpoints_strong: bool = True
    search_widths: float = 8.
    fit_widths: float = 12.
    probe_widths: float = 8.
    source_corridor_widths: float = 1.25
    minimum_columns_widths: float = 2.
    maximum_outward_own: float = .15
    maximum_outward_other: float = .20
    minimum_outward_paper: float = .65


def apply_endpoint_strong_prior(candidate):
    """User-directed Step5 admission, not a new source-evidence measurement.

    Only the two whole-path endpoint candidate records use this helper. Active,
    rejected and internal candidates are left unchanged. In particular we do
    not rewrite failed fit/gates or the measured strong_suppressed flag.
    """
    result=deepcopy(candidate)
    evidence=result.get('evidence') or {}
    if (result.get('kind')!='endpoint_measurement_hypothesis'
            or result.get('side') not in ('left','right')
            or result.get('state')!='suppressed' or evidence.get('policy')!=POLICY):
        return result
    evidence.update(step5_admission_policy=ENDPOINT_PRIOR_POLICY,
                    endpoint_scope='whole_path_outer',endpoint_side=result['side'],
                    endpoint_state='suppressed',treat_as_strong_for_step5=True,
                    strong_basis='user_directed_endpoint_prior_not_new_image_evidence')
    result.update(evidence=evidence,step5_eligible=True)
    return result


def endpoint_prior_admission(point):
    """Recognize an explicitly admitted suppressed endpoint, never an interior point."""
    evidence=point.get('endpoint_evidence') or {}
    return bool(point.get('status') not in ('accepted','active','rejected','unresolved')
        and evidence.get('policy')==POLICY
        and evidence.get('step5_admission_policy')==ENDPOINT_PRIOR_POLICY
        and evidence.get('endpoint_scope')=='whole_path_outer'
        and evidence.get('endpoint_state')=='suppressed'
        and evidence.get('endpoint_side') in ('left','right')
        and point.get('endpoint_side')==evidence.get('endpoint_side')
        and evidence.get('treat_as_strong_for_step5') is True)


def _fit_source(curve, soft, raw_xy, x, seed_y, inward, width, exclusion, cfg):
    """Fit original own-color column centers, excluding stem/cap footprints."""
    start = max(2, int(np.ceil(exclusion)))
    reach = start + max(16, int(np.ceil(cfg.fit_widths*width)))
    offsets = np.arange(start, reach+1)
    columns = np.rint(x+inward*offsets).astype(int)
    xs, ys, weights = [], [], []
    h, w = curve.shape
    for col in columns:
        if not 0 <= col < w:
            continue
        predicted = float(np.interp(col, raw_xy[:, 0], raw_xy[:, 1]))
        # No unrestricted same-column search: nearby unrelated curves must not
        # establish endpoint evidence just because their color is compatible.
        radius = max(3., cfg.source_corridor_widths*width)
        a, b = max(0, int(np.floor(predicted-radius))), min(h, int(np.ceil(predicted+radius))+1)
        hits = np.flatnonzero(curve[a:b, col])+a
        if not len(hits):
            continue
        # Select the nearest contiguous stroke, never average distinct y runs.
        chunks = np.split(hits, np.flatnonzero(np.diff(hits)>1)+1)
        hits = min(chunks, key=lambda q:abs(float(np.mean(q))-predicted))
        mass = np.maximum(.01, soft[hits, col])
        xs.append(float(col)); ys.append(float(np.average(hits, weights=mass)))
        weights.append(float(min(3., mass.sum())))
    minimum = max(6, int(np.ceil(cfg.minimum_columns_widths*width)))
    if len(xs) < 4:
        return dict(available=False, source_columns=len(xs), required_columns=minimum,
                    points=[[a,b] for a,b in zip(xs,ys)])
    xs, ys, weights = np.asarray(xs), np.asarray(ys), np.asarray(weights)
    slope, intercept = np.polyfit(xs-x, ys, 1, w=np.sqrt(weights))
    residual = ys-(slope*(xs-x)+intercept)
    rms = float(np.sqrt(np.average(residual**2, weights=weights)))
    near = np.argsort(abs(xs-x))[:max(4, int(np.ceil(len(xs)*.6)))]
    short_y = float(np.polyfit(xs[near]-x, ys[near], 1, w=np.sqrt(weights[near]))[1])
    instability = abs(short_y-intercept)
    stable = bool(len(xs)>=minimum and rms<=max(1., .65*width)
                  and instability<=max(1., .75*width))
    return dict(available=True, stable=stable, source_columns=len(xs), required_columns=minimum,
                slope=float(slope), y=float(intercept), rms=rms, intercept_instability=float(instability),
                nearest_source_gap=float(abs(xs-x).min()), points=np.column_stack((xs,ys)).tolist(),
                source='inward_same_series_original_pixels_not_path_coordinates')


def _probe(own, other, paper, valid, text, grid, x, y, slope, inward, width, exclusion, cfg):
    start = max(2, int(np.ceil(exclusion)))
    distances = np.arange(start, start+max(12, int(np.ceil(cfg.probe_widths*width))))
    rows = []
    h, w = own.shape
    radius = max(1, int(np.ceil(.5*width)))
    for dist in distances:
        xx = int(round(x-inward*dist)); yy = y-slope*inward*dist
        if not (0 <= xx < w and radius <= yy < h-radius):
            rows.append(dict(in_scope=False)); continue
        a, b = int(round(yy))-radius, int(round(yy))+radius+1
        scope = valid[a:b, xx] & ~text[a:b, xx] & (grid[a:b, xx]<.5)
        rows.append(dict(in_scope=bool(scope.all()), own=bool((own[a:b,xx]&scope).any()),
                         other=bool((other[a:b,xx]&scope).any()),
                         paper=float(paper[a:b,xx].mean()), x=xx, y=float(yy)))
    inside = [r for r in rows if r['in_scope']]
    return dict(in_scope_fraction=len(inside)/max(1,len(rows)),
                own_fraction=float(np.mean([r['own'] for r in inside])) if inside else 0.,
                other_fraction=float(np.mean([r['other'] for r in inside])) if inside else 0.,
                paper_fraction=float(np.mean([r['paper'] for r in inside])) if inside else 0.,
                samples=rows)


def _at(mask, x, y, radius=1):
    xx, yy = int(round(x)), int(round(y))
    if not (0 <= xx < mask.shape[1] and 0 <= yy < mask.shape[0]):
        return False
    return bool(mask[max(0,yy-radius):yy+radius+1,max(0,xx-radius):xx+radius+1].any())


def test_path_endpoints(record, *, image_bgr, plot_box, own, soft, valid, grid,
                        text_mask, other_ink, line_width, stems=None, config=None):
    """Return two tested global endpoints, never every disconnected-run edge.

    Active requires stable original inward ink, an observed endpoint/bar, and
    verified outward termination. Plausible but continuing/occluded/cropped
    candidates are suppressed. No source arm or obvious axis/text is rejected.
    Source strong_suppressed records independent image evidence. By default the
    user's endpoint prior ALSO admits every suppressed outer endpoint to Step5,
    without changing its source evidence or activating it automatically.
    """
    cfg = config or EndpointConfig()
    width = float(line_width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('line_width must be positive finite')
    width = max(1., width)
    x0,y0,x1,y1 = map(int,plot_box)
    shape = (y1-y0,x1-x0)
    for label, a in [('own',own),('soft',soft),('valid',valid),('grid',grid),
                     ('text_mask',text_mask),('other_ink',other_ink)]:
        if np.asarray(a).shape != shape or not np.isfinite(a).all():
            raise ValueError(label+' must be finite and plot-local')
    own,valid,text,other = [np.asarray(a,bool) for a in (own,valid,text_mask,other_ink)]
    soft,grid = np.asarray(soft,float),np.asarray(grid,float)
    clean = valid & ~text & (grid<.5)
    original = np.asarray(image_bgr)[y0:y1,x0:x1]
    paper = original.min(axis=2)>=245
    ink = own & clean & ~paper
    raw = np.asarray(record['raw_path_source'],float).reshape(-1,2)-[x0,y0]
    keep = np.asarray(record['reference_allowed'],bool)
    if keep.shape != (len(raw),) or not np.isfinite(raw).all():
        raise ValueError('path/reference_allowed mismatch')
    indices = np.flatnonzero(keep)
    if not len(indices):
        return []
    candidates=[]
    for side,index,inward in [('left',indices[0],1),('right',indices[-1],-1)]:
        seed_x,seed_y = raw[index]
        variants=[dict(x=float(seed_x), y=float(seed_y), stem=None, offset=0.)]
        for stem in stems or []:
            if (abs(stem['x']-seed_x)<=cfg.search_widths*width and
                    stem['y0']-2*width<=seed_y<=stem['y1']+2*width):
                variants.append(dict(x=float(stem['x']),y=float(seed_y),stem=deepcopy(stem),
                                     offset=float(abs(stem['x']-seed_x))))
        tested=[]
        nearby_bar_ambiguity=any(v['stem'] is not None and v['stem'].get('caps') for v in variants)
        for v in variants:
            stem=v['stem']; caps=[] if stem is None else stem.get('caps',[])
            # Cap extent sets how far inward we must look to fit the curve,
            # and how far outward we must look before testing continuation.
            cap_extent=max([abs(float(c['x0'])-v['x']) for c in caps]+
                           [abs(float(c['x1'])-v['x']) for c in caps]+[.75*width])
            exclusion=cap_extent+max(1.,.5*width)
            fit=_fit_source(ink,soft,raw,v['x'],seed_y,inward,width,exclusion,cfg)
            if not fit['available']:
                tested.append(dict(**v,fit=fit,rank=-1000.-v['offset'])); continue
            y=fit['y']
            # A cap fragment is not an error-bar center. The fitted source
            # curve must cross the vertical core with ink extent on both sides.
            # One-sided/hidden cores remain review hypotheses, not active data.
            margin=max(1.,.5*width)
            in_stem=bool(stem is not None and stem['y0']+margin<=y<=stem['y1']-margin)
            stem_width=float(stem.get('width',stem['x1']-stem['x0']+1)) if stem else 0.
            stem_height=float(stem['y1']-stem['y0']+1) if stem else 0.
            narrow_core=bool(stem and stem_width<=max(2.,2*width) and
                             stem_height>=max(4.,1.5*width) and stem_height>=1.3*stem_width)
            cap_supported=bool(in_stem and caps and narrow_core and stem.get('source_support',0)>=.5)
            # A physical error bar is original ink; its midpoint is never used.
            bar=dict(present=cap_supported, stem=stem, cap_count=len(caps),
                     x_from='observed_vertical_stem' if stem else 'retained_path_tip',
                     y_from='inward_source_curve_fit', midpoint_used=False,
                     center_spanning_curve=in_stem, center_margin_px=margin,
                     narrow_vertical_core=narrow_core,stem_width_px=stem_width,stem_height_px=stem_height)
            probe=_probe(ink,other,paper,valid,text,grid,v['x'],y,fit['slope'],
                         inward,width,exclusion,cfg)
            center_ink=_at(ink,v['x'],y,max(1,int(np.ceil(width/2))))
            border=min(v['x'],shape[1]-1-v['x'],y,shape[0]-1-y)<max(1.,width)
            bad_text=_at(text,v['x'],y,1)
            grid_at=_at(grid>=.5,v['x'],y,0)
            # Extreme endpoint y movement is not justified by a far-away fit.
            localization=abs(y-seed_y)<=max(3.,3*width)
            terminal=(probe['in_scope_fraction']>=.85 and probe['own_fraction']<=cfg.maximum_outward_own
                      and probe['other_fraction']<=cfg.maximum_outward_other
                      and probe['paper_fraction']>=cfg.minimum_outward_paper)
            stable=fit.get('stable',False)
            gates=dict(source_arm_stable=stable, localized_to_seed=localization,
                       endpoint_original_ink=bool(center_ink or cap_supported),
                       outward_termination=terminal, not_crop_boundary=not border,
                       not_text=not bad_text, not_grid=not grid_at,
                       not_unresolved_cap_edge=bool(cap_supported or not nearby_bar_ambiguity))
            # The payload adapter operates before JSON serialization. Normalize
            # numpy scalar comparisons now, so saved and in-memory gates agree.
            gates={key:bool(value) for key,value in gates.items()}
            active=all(gates.values())
            rank=(100. if active else 0.)+(15. if cap_supported else 0.)+(10. if stable else 0.)
            rank-=fit['rms']+fit['intercept_instability']+v['offset']/max(1.,width)
            # Prefer a physical stem over an observed cap edge even if the raw
            # edge has a slightly easier termination probe.
            tested.append(dict(**v,fit=fit,bar=bar,probe=probe,center_y=float(y),gates=gates,
                               rank=rank,cap_extent=float(cap_extent)))
        usable=[v for v in tested if v['fit']['available']]
        bars=[v for v in usable if v['bar']['present'] and v['gates']['localized_to_seed']
              and v['gates']['source_arm_stable']]
        chosen=max(bars or usable or tested,key=lambda v:v['rank'])
        fit=chosen['fit']; y=chosen.get('center_y',float(seed_y)); x=chosen['x']
        gates=chosen.get('gates',{})
        if not fit['available']:
            state,reason='rejected','no_original_inward_curve_arm'
        elif not gates['not_text'] or not gates['not_grid']:
            state,reason='rejected','text_or_grid_confounded_endpoint'
        elif not gates['not_crop_boundary']:
            state,reason='suppressed','crop_or_axis_boundary_not_proven_curve_end'
        elif all(gates.values()):
            state,reason='active','observed_terminal_bar_and_inward_curve' if chosen['bar']['present'] else 'observed_stroke_termination'
        elif not gates['not_unresolved_cap_edge']:
            state,reason='suppressed','cap_or_stem_localization_unresolved'
        elif chosen['probe']['own_fraction']>cfg.maximum_outward_own:
            state,reason='suppressed','source_curve_continues_outward_path_was_truncated'
        elif chosen['probe']['other_fraction']>cfg.maximum_outward_other:
            state,reason='suppressed','outer_region_occluded_end_position_uncertain'
        elif not gates['source_arm_stable'] or not gates['localized_to_seed']:
            state,reason='suppressed','endpoint_source_fit_or_location_unstable'
        else:
            state,reason='suppressed','outward_termination_not_verified'
        strong=(state=='suppressed' and fit.get('stable',False) and chosen.get('bar',{}).get('present',False)
                and gates.get('localized_to_seed',False) and gates.get('not_crop_boundary',False)
                and gates.get('not_text',False) and gates.get('not_grid',False)
                and chosen.get('probe',{}).get('own_fraction',1)<=cfg.maximum_outward_own)
        proposed_center=[float(x+x0),float(y+y0)]
        if fit.get('available') and not gates.get('localized_to_seed',False):
            # Preserve a review hypothesis at the seed rather than exporting
            # an arbitrarily distant extrapolation as a candidate location.
            x,y=float(seed_x),float(seed_y)
            state,reason='suppressed','endpoint_source_fit_or_location_unstable'
            strong=False
        c=dict(id=f"{record['series_id']}_END_{side.upper()}",series_id=record['series_id'],side=side,
               seed_x_px=float(seed_x+x0),seed_y_px=float(seed_y+y0),x_px=float(x+x0),y_px=float(y+y0),
               state=state,reason=reason,step5_eligible=bool(strong),
               kind='endpoint_measurement_hypothesis',marker_glyph_detected=False,
               evidence=dict(policy=POLICY,gates=gates,fit=fit,bar=chosen.get('bar'),
                             outward_probe=chosen.get('probe'),strong_suppressed=bool(strong),
                             geometry_coordinates='plot-local',plot_offset=[x0,y0],
                             raw_endpoint_index=int(index),variants_tested=len(tested),
                             proposed_unstable_center_source=proposed_center,
                             nearby_bar_ambiguity=bool(nearby_bar_ambiguity),
                             variant_diagnostics=[dict(x=v['x'], offset=v['offset'],rank=v['rank'],
                                 fit=v['fit'],gates=v.get('gates'),bar=v.get('bar')) for v in tested],
                             endpoint_prior='line_only_measurement_polyline',
                             config=asdict(cfg),probability_claimed=False))
        candidates.append(apply_endpoint_strong_prior(c) if cfg.force_suppressed_endpoints_strong else c)
    return candidates


def endpoint_structural_admission(point):
    """Recheck endpoint evidence for an explicitly suppressed pool entry."""
    if point.get('status') in ('rejected','unresolved'):
        return False,'rejected_endpoint_cannot_enter_step5'
    if endpoint_prior_admission(point):
        return True,'global_endpoint_treated_as_strong_by_user_prior'
    e=point.get('endpoint_evidence') or {}
    gates=e.get('gates') or {};fit=e.get('fit') or {};probe=e.get('outward_probe') or {}
    required=('source_arm_stable','localized_to_seed','endpoint_original_ink',
              'not_crop_boundary','not_text','not_grid','not_unresolved_cap_edge')
    okay=(e.get('policy')==POLICY and e.get('strong_suppressed') is True
          and all(gates.get(k) is True for k in required)
          and fit.get('stable') is True and (e.get('bar') or {}).get('present') is True
          and (e.get('bar') or {}).get('center_spanning_curve') is True
          and probe.get('own_fraction',1)<=EndpointConfig().maximum_outward_own)
    return bool(okay),('strong_terminal_structure_with_uncertain_visibility' if okay
                       else 'endpoint_has_no_independent_strong_terminal_structure')

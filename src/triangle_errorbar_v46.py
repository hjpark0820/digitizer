"""v46 filled-triangle / finite error-bar-cap comparison.

Ported from the reviewed ASN100 experiment. Colour marker runtime enables this
only for supported filled-triangle families. Filenames, expected point counts
and audit coordinates never enter inference. Upright templates use reflection.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

VERSION = 'v46_triangle_errorbar_confidence_weighted_nuisance_v4'


@dataclass(frozen=True)
class Config:
    # These initial dimensionless settings are experimental, not calibrated.
    minimum_shape_score: float = .60
    maximum_shape_gap: float = .08
    minimum_interior_fill: float = .85
    minimum_visible_wing_fraction: float = .30
    minimum_t_confidence: float = .55
    maximum_penalty_strength: float = .95
    shape_ink_allowance: float = .12
    shape_missing_tolerance: float = .25
    shape_missing_span: float = .50
    shape_maximum_penalty: float = .97
    require_net_body_evidence: bool = True
    gate_unconfirmed_t: bool = True


def triangle_prior(report, config=Config()):
    """Do not convert an uncertain legend classification into a certain label."""
    e = report.get('shape_evidence', {})
    name = report.get('shape_hint', 'unknown_marker')
    scores = e.get('shape_scores', {})
    fill = e.get('continuous_fill_evidence', {})
    filled = (e.get('independent_fill', 0) >= config.minimum_interior_fill and
              not e.get('strong_hollow_evidence', False) and
              fill.get('mean_membership', 0) >= .72)
    if not filled:
        return dict(applicable=False, reason='filled_interior_not_supported', legend_label=name)
    for shape, orientation in [('triangle', 'up'), ('inv_triangle', 'down')]:
        if name == 'filled_' + shape:
            return dict(applicable=True, strength='confirmed', orientation=orientation,
                        legend_label=name, shape_score=scores.get(shape, 1.))
    best = max(scores.values(), default=0.)
    shape = max(('triangle', 'inv_triangle'), key=lambda k: scores.get(k, 0.))
    score = scores.get(shape, 0.)
    if (name == 'unknown_marker' and score >= config.minimum_shape_score and
            best-score <= config.maximum_shape_gap and
            e.get('polygon_consensus', {}).get(shape, 0) >= 2):
        return dict(applicable=True, strength='tentative', orientation='up' if shape=='triangle' else 'down',
                    legend_label=name, shape_score=score,
                    reason='near_best_triangle_with_polygon_and_continuous_fill_evidence')
    return dict(applicable=False, reason='triangle_family_not_supported', legend_label=name)


def _rectangle(xx, yy, left, right, top, bottom, sigma):
    """Analytic Gaussian pixel-footprint model shared by H0 and H1."""
    horizontal = ndtr((xx-left)/sigma)-ndtr((xx-right)/sigma)
    vertical = ndtr((yy-top)/sigma)-ndtr((yy-bottom)/sigma)
    return horizontal*vertical


def _render(params, xx, yy, bottom):
    cx, cap_y, half_length, cap_width, stem_width, amplitude, sigma = params
    cap = _rectangle(xx, yy, cx-half_length, cx+half_length,
                     cap_y-cap_width/2, cap_y+cap_width/2, sigma)
    stem = _rectangle(xx, yy, cx-stem_width/2, cx+stem_width/2,
                      cap_y, bottom, sigma)
    return (amplitude*np.maximum(cap, stem)).astype(np.float32)


def independent_shape_check(fields, diameter, config=Config(), debug=False, nuisance=None):
    """Test required visible ink without depending on a fitted T explanation.

    Inputs are in canonical down-triangle orientation. The *observed legend*
    defines all required ink. Geometric weights emphasize its shoulders below
    the wide top row and outside the central vertical stroke; no ideal triangle
    is painted. Other-color occlusion/ignored pixels are untestable, not positive
    evidence. A small intensity allowance protects uncertain antialiased edges.
    An optional frozen nuisance model selects a third, discriminating subset:
    required glyph ink exceeding that model. Its absence is checked in the
    image even when T identity is uncertain. Body/shoulder required-ink regions
    are independent of T confidence; nuisance_excess uses the admitted model.
    """
    marker=np.clip(np.asarray(fields['template'],np.float32),0.,1.)
    observed=np.clip(np.asarray(fields['observed'],np.float32),0.,1.)
    valid=np.clip(np.asarray(fields['available'],np.float32),0.,1.)
    visible=valid*(1-np.clip(np.asarray(fields['other'],np.float32),0.,1.))
    required=np.maximum(marker-config.shape_ink_allowance,0.)
    ys,xs=np.nonzero(required>0)
    result=dict(version='visible_observed_triangle_ink_v1',applied=False,factor=1.,
        reason='insufficient_visible_shape_evidence',depends_on_t_confidence=False,
        other_ink_is_positive_evidence=False,source_pixels_changed=False)
    if not len(xs):return result
    yy,xx=np.mgrid[:marker.shape[0],:marker.shape[1]].astype(np.float32)
    cx=(float(xs.min())+float(xs.max()))/2
    width=max(3.,float(xs.max()-xs.min()+1));height=max(3.,float(ys.max()-ys.min()+1))
    side=np.clip((np.abs(xx-cx)/width-.12)/.12,0.,1.)
    below_top=np.clip(((yy-float(ys.min()))/height-.12)/.16,0.,1.)
    expectations=dict(body=required,shoulders=required*side*below_top)
    if nuisance is not None:
        expectations['nuisance_excess']=np.maximum(required-np.asarray(nuisance,np.float32),0.)
    regions={};debug_maps={}
    deficit=np.maximum(required-observed,0.)*fields.get('missing_weight', 1.)
    for name,expected in expectations.items():
        total=float(expected.sum());visible_mass=float((expected*visible).sum())
        visible_fraction=visible_mass/max(total,1e-6)
        if name=='nuisance_excess':
            missing_map=np.minimum(deficit,expected)*visible
        else:
            missing_map=deficit*np.divide(expected,required,out=np.zeros_like(required),where=required>0)*visible
        missing=float(missing_map.sum())/max(visible_mass,1e-6)
        proportion=np.divide(expected,required,out=np.zeros_like(required),where=required>0)
        positive=float((np.minimum(observed,required)*proportion*visible).sum())/max(visible_mass,1e-6)
        enough=(visible_mass>=max(1.,.02*diameter*diameter) and
                visible_fraction>=config.minimum_visible_wing_fraction)
        mismatch=float(np.clip((missing-config.shape_missing_tolerance)/config.shape_missing_span,0.,1.))
        factor=1-config.shape_maximum_penalty*mismatch*mismatch if enough else 1.
        regions[name]=dict(expected_mass=total,visible_mass=visible_mass,
            visible_fraction=visible_fraction,missing_fraction=missing,
            positive_fraction=positive,net_support=float(np.clip(positive-missing,0.,1.)),
            enough_visible_evidence=bool(enough),factor=float(factor))
        if debug:
            debug_maps[name+'_required']=expected*visible
            debug_maps[name+'_missing']=missing_map
    factor=min(r['factor'] for r in regions.values())
    applied=any(r['enough_visible_evidence'] for r in regions.values())
    result.update(applied=applied,factor=float(factor),regions=regions,
        reason=('unoccluded_required_triangle_ink_missing' if factor<.98 else
                'visible_triangle_ink_consistent' if applied else 'insufficient_visible_shape_evidence'))
    if debug:result['maps']=debug_maps
    return result


def compare_cap(window, template, prior, config=Config(), debug=False):
    """Return only a conservative score reduction; never add positive evidence.

    H0 is a frozen finite cap + stem + existing line/guide model. H1 adds the
    same observed legend glyph. The cap does not erase source pixels or excuse
    missing glyph wings. An independent required-ink check also runs when the
    T identity is uncertain. Colour occlusion only excuses missing target ink.
    """
    old_score = float(window['score'])
    result = dict(applied=False, prior=prior, original_score=old_score,
                  score=old_score, factor=1., decision='unchanged')
    if not prior.get('applicable'):
        result['reason'] = prior.get('reason', 'not_triangle'); return result
    maps = window['maps']
    flip = prior['orientation'] == 'up'
    fields = {k: np.flipud(v).copy() if flip else np.asarray(v).copy() for k,v in maps.items()}
    observed, other, valid = (fields[k] for k in ('observed','other','available'))
    missing_weight = fields.get('missing_weight', 1.)
    guide, guide_high, line, marker = (fields[k] for k in ('guide','guide_upper','line','template'))
    n = observed.shape[0]; center = (n-1)/2
    yy,xx = np.mgrid[:n,:n].astype(np.float32); xx-=center; yy-=center
    d = max(3.,float(template['diameter']))
    # The bounded cap model cannot expand its vertical stroke into a glyph.
    # A smooth transition accounts for image blur, not arbitrary template scaling.
    bounds = ([-.20*d,-.60*d,.25*d,.07*d,.06*d,.30,.35],
              [ .20*d, .12*d,.85*d,.35*d,.35*d,1.0,max(.6,.16*d)])
    search = ((abs(xx)<=1.1*d)&(yy>=-.85*d)&(yy<=1.65*d)).astype(np.float32)*valid
    visible = 1-other
    fixed = np.maximum(guide,line)
    # Fit the same visible pixels for every candidate; no manual cap location.
    # The template's centre/size bounds the search but does not supply observed ink.
    def residual(p):
        rendered = np.maximum(_render(p,xx,yy,1.85*d),fixed)
        deficit = np.maximum(rendered-observed,0)*visible*missing_weight
        surplus = np.maximum(observed-rendered,0)
        return ((deficit-surplus)*search).ravel()
    choices=[]
    for y_start in (-.40*d,-.18*d):
        initial=[0,y_start,.50*d,.17*d,.16*d,.90,min(.8,.11*d)]
        fit=least_squares(residual,initial,bounds=bounds,loss='soft_l1',f_scale=.18,
                          max_nfev=75,diff_step=1e-3)
        choices.append(fit)
    fit=min(choices,key=lambda f:f.cost); params=fit.x
    t=_render(params,xx,yy,1.85*d)
    cx,cap_y,half_length,cap_width,stem_width,amplitude,sigma=map(float,params)
    # Independent evidence for a cap and a coherent attached stem is mandatory.
    cap_region=(abs(yy-cap_y)<=max(1.,.65*cap_width)) & (abs(xx-cx)<=half_length) & (abs(xx-cx)>=max(1.,stem_width))
    stem_region=(abs(xx-cx)<=max(.65,stem_width/2)) & (yy>=cap_y+.4*d) & (yy<=cap_y+1.45*d)
    def support(region):
        expected=t*region*valid
        mass=float(expected.sum())
        matched=float((np.minimum(np.maximum(observed-guide_high,0),t)*region*valid).sum())
        return matched/max(mass,1e-6),mass
    cap_support,cap_mass=support(cap_region)
    stem_support,stem_mass=support(stem_region)
    confidence=min(cap_support,stem_support)*min(1.,cap_mass/max(1.,.025*d*d))*min(1.,stem_mass/max(1.,.04*d*d))
    t_admitted = confidence >= config.minimum_t_confidence or not config.gate_unconfirmed_t
    # An uncertain T is not a certain nuisance, but discarding a partly
    # supported cap entirely can resurrect real error bars. Use its measured
    # cap/stem confidence as a bounded soft amplitude, never as marker credit.
    # This weight is a heuristic, not a calibrated probability.
    t_weight=1. if t_admitted else float(np.clip(confidence,0.,1.))
    h0=np.maximum(fixed,t*t_weight);h1=np.maximum(h0,marker)
    # Restrict the decision to the glyph body, including its blur halo.
    domain=(np.hypot(xx,yy)<=.95*d).astype(np.float32)*valid
    mass=max(float((marker*valid).sum()),1.)
    def loss(model):
        return float(((np.maximum(model-observed,0)*visible*missing_weight+
            np.maximum(observed-np.maximum(model,guide_high),0))*domain).sum())/mass
    loss0,loss1=loss(h0),loss(h1);gain=loss0-loss1
    # The discriminating wings are exactly what the frozen nuisance model lacks.
    wings=np.maximum(marker-h0,0)*domain
    wing_mass=float(wings.sum())
    seen=wings*visible
    visible_mass=float(seen.sum())
    visible_fraction=visible_mass/max(wing_mass,1e-6)
    residual_ink=np.maximum(observed-np.maximum(h0,guide_high),0)
    matched=np.minimum(residual_ink,wings)*visible
    wing_support=float(matched.sum())/max(visible_mass,1e-6)
    wing_missing=float((np.minimum(np.maximum(marker-observed,0),wings)*visible*missing_weight).sum())/max(visible_mass,1e-6)
    # Separate sides are diagnostics, not a rigid two-visible-sides requirement.
    sides={}
    for name,region in [('left',xx<cx),('right',xx>=cx)]:
        sm=float((seen*region).sum())
        sides[name]=dict(visible_mass=sm,support=float((matched*region).sum())/max(sm,1e-6))
    result.update(applied=True,orientation=prior['orientation'],fit_success=bool(fit.success),
        t_confidence=float(confidence),cap_support=cap_support,stem_support=stem_support,
        t_admitted_as_nuisance=bool(t_admitted),
        t_nuisance_weight=t_weight,
        parameters=dict(cx=cx,cap_y=cap_y,half_length=half_length,cap_width=cap_width,
            stem_width=stem_width,amplitude=amplitude,blur_sigma=sigma),
        loss_errorbar=loss0,loss_errorbar_plus_triangle=loss1,marker_gain=gain,
        wing_mass=wing_mass,visible_wing_fraction=visible_fraction,
        wing_support=wing_support,wing_missing=wing_missing,wing_sides=sides)
    enough=(wing_mass>=max(1.,.025*d*d) and visible_fraction>=config.minimum_visible_wing_fraction)
    if confidence<config.minimum_t_confidence:
        result['reason']='cap_and_stem_not_independently_supported'
    elif not enough:
        result['decision']='ambiguous';result['reason']='too_little_visible_discriminating_ink'
    else:
        # A smooth, bounded penalty, not a new binary per-pixel test. A good
        # triangle adds ink where T alone fails and therefore keeps factor ~1.
        mismatch=float(np.clip((wing_missing-.10)/.55,0,1))
        cap_strength=float(np.clip((confidence-.45)/.45,0,1))
        lack_of_gain=float(np.clip((.15-gain)/.30,0,1))
        penalty=config.maximum_penalty_strength*cap_strength*mismatch*max(.35,lack_of_gain)
        result['factor']=1-penalty
        result['score']=old_score*(1-penalty)
        result['decision']='score_reduced' if penalty>.02 else 'triangle_supported'
        result['reason']='visible_triangle_wings_missing' if penalty>.02 else 'independent_triangle_wings_observed'
    # Nuisance identity and marker existence are different questions. Failure
    # to prove a T must not erase directly observed missing-triangle evidence.
    shape=independent_shape_check(fields,d,config,debug,nuisance=h0)
    shape_maps=shape.pop('maps',{})
    result.update(cap_only_score=result['score'],cap_only_factor=result['factor'],
        cap_only_reason=result['reason'],cap_only_decision=result['decision'],
        independent_shape=shape)
    if shape['factor']<result['factor']:
        result.update(factor=shape['factor'],score=old_score*shape['factor'])
        if shape['factor']<.98:
            result.update(decision='score_reduced',reason=shape['reason'])
    # When furniture masking hides most of a glyph, the leftover central
    # axis/tick stroke is especially ambiguous. Require net shoulder evidence
    # in that case. Do not impose a hard outer-shoulder ceiling on complete
    # glyphs: raster registration affects their rim (ordinary shape penalties
    # and the stable filled-core check still apply to every complete glyph).
    shoulders=shape.get('regions',{}).get('shoulders',{})
    available_body=float((marker*valid).sum())/max(float(marker.sum()),1e-6)
    shoulder_ceiling=(shoulders.get('net_support',1.) if
        available_body<.60 and shoulders.get('enough_visible_evidence') else 1.)
    result['available_body_fraction']=available_body
    result['shoulder_score_ceiling']=shoulder_ceiling
    if shoulder_ceiling<result['score']:
        result.update(score=shoulder_ceiling,factor=shoulder_ceiling/max(old_score,1e-8),
            decision='score_reduced',reason='insufficient_observed_triangle_shoulders')
    # A relative penalty alone lets a high-scoring T survive: .8 reduced by
    # half still exceeds the production cutoff .25. Bound the final score by
    # *net* discriminating body evidence, independently of the old score.
    # T/line/guide ink earns no support. Missing unoccluded glyph ink is a debit;
    # other-colour occlusion is neither credit nor debit. A good partially
    # occluded triangle can qualify through one visible wing. The two terms
    # share the same visible-wing denominator, so net=1 means entirely positive
    # evidence and net<=0 means contradiction outweighs positive body evidence.
    # Insufficient visible discriminating area abstains; it is not proof of a T.
    net_body=float(np.clip(wing_support-wing_missing,0.,1.))
    body_applied=bool(config.require_net_body_evidence and enough and
                      visible_mass>=max(1.,.025*d*d))
    result['body_evidence']=dict(version='net_discriminating_body_v1',applied=body_applied,
        positive_fraction=wing_support,missing_fraction=wing_missing,
        net_support=net_body,score_ceiling=net_body if body_applied else None,
        visible_mass=visible_mass,visible_fraction=visible_fraction,
        depends_on_t_confidence=bool(config.gate_unconfirmed_t),other_ink_is_positive_evidence=False,
        reason='sufficient_discriminating_area' if body_applied else 'abstain_insufficient_visible_body')
    result['pre_body_score']=result['score']
    if body_applied and net_body<result['score']:
        result.update(score=net_body,factor=net_body/max(old_score,1e-8),
                      decision='score_reduced',reason='insufficient_net_triangle_body_evidence')
    if debug:
        debug_maps=dict(T=t,H0=h0,H1=h1,wings=wings,wing_missing=np.minimum(np.maximum(marker-observed,0),wings)*visible*missing_weight,
                        independent_wing_ink=matched,fit_domain=search)
        debug_maps.update({'independent_'+k:v for k,v in shape_maps.items()})
        result['maps']={k:np.flipud(v).copy() if flip else v for k,v in debug_maps.items()}
    return result


def apply_guards(evidence, candidates, priors, threshold):
    """Reduce qualifying window scores in-place before centre competition.

    Below-threshold candidates cannot become active because this guard never
    raises a score, so their expensive fit is unnecessary. Historical raw
    window diagnostics remain nested separately from the final score.
    """
    from time import perf_counter
    from color_marker_window_v2 import verify_window
    from color_blend_uncertainty_v46 import window_kwargs
    started = perf_counter()
    indices = {str(t['id']): i for i, t in enumerate(evidence['templates'])}
    checked = demoted = shape_checked = shape_reduced = body_reduced = 0
    for point in candidates:
        prior = priors.get(point['series_id'], {})
        if point['score'] < threshold or not prior.get('applicable'):
            continue
        si = indices[point['series_id']]
        window = verify_window(evidence['membership'][si], evidence['other'][si],
            evidence['templates'][si], point['x'], point['y'], debug=True,
            ignore_mask=evidence.get('ignore_mask'),
            **window_kwargs(evidence, si),
            guide_model=None if 'guide_membership' not in evidence else (
                evidence['guide_membership'][si], evidence['guide_lower'][si], evidence['guide_upper'][si]))
        guard = compare_cap(window, evidence['templates'][si], prior)
        guard['below_final_threshold'] = bool(guard['score'] < threshold)
        guard['accepted_at_final_threshold'] = bool(guard['score'] >= threshold)
        point.update(original_score=float(point['score']), score=guard['score'], triangle_cap=guard)
        checked += 1
        demoted += guard['score'] < threshold
        shape_checked += bool(guard.get('independent_shape',{}).get('applied'))
        shape_reduced += guard.get('pre_body_score',guard['score']) < guard.get('cap_only_score',guard['score'])-1e-8
        body_reduced += guard['score'] < guard.get('pre_body_score',guard['score'])-1e-8
    return dict(version=VERSION, enabled=True, status='completed', priors=priors,
        checked_candidates=checked, below_threshold_candidates=int(demoted),
        final_threshold=float(threshold), seconds=perf_counter()-started,
        independent_shape_checked=int(shape_checked),independent_shape_reduced=int(shape_reduced),
        net_body_reduced=int(body_reduced),
        score_policy='only_reduce_existing_score; cap/required-ink penalties plus net discriminating body ceiling; no hard errorbar veto',
        weak_policy='diagnostic suppressed only; not strong Step-5 evidence')


def weak_suppressed(before, candidates, plot, legend):
    """Keep unique formerly active demotions, without promoting duplicate IDs."""
    old_ids = {p['candidate_id'] for p in before}
    result = []
    for point in candidates:
        guard = point.get('triangle_cap', {})
        if point['candidate_id'] not in old_ids or not guard.get('below_final_threshold', False):
            continue
        x, y = point['x_px'], point['y_px']
        if not (plot[0] <= x < plot[2] and plot[1] <= y < plot[3]):
            continue
        if legend is not None and legend[0] <= x < legend[2] and legend[1] <= y < legend[3]:
            continue
        result.append(dict(point, source=VERSION, origin='triangle_errorbar_competition',
            state='suppressed', evidence_tier='weak', evidence_supported=False,
            strong_evidence=False, existence='unknown', auto_promote=False,
            triangle_cap_review_only=True, step5_suppressed_eligible=False,
            suppression_reason=guard.get('reason','visible_triangle_wings_missing')))
    return result


def protect_weak_candidates(tentative, weak_points):
    """Do not relabel the same cap as strong via a second proposal source.

    Match only same-series near-identical source centres, not whole x columns.
    Unrelated series and nearby real markers retain their own evidence.
    """
    count = 0
    for point in tentative.get('candidates', []):
        x, y = point.get('x_px', point.get('x')), point.get('y_px', point.get('y'))
        match = next((q for q in weak_points if q['series_id'] == point.get('series_id') and
            math.hypot(x-q['x_px'], y-q['y_px']) <= max(1., .16*q['diameter_source'])), None)
        if match is not None:
            point.update(triangle_cap_review_only=True, evidence_supported=False,
                evidence_tier='weak', state='diagnostic_only', step5_suppressed_eligible=False,
                triangle_cap_candidate_id=match['candidate_id'],
                evidence_status='triangle_errorbar_competition_weak')
            count += 1
    tentative['triangle_errorbar_weak_suppressed'] = weak_points
    tentative['triangle_errorbar_guarded_proposals'] = count
    return count


def draw_review(image, points, weak_points, destination):
    """Source-aligned diagnostic only; cyan rings active, magenta X weak."""
    from pathlib import Path
    canvas = image.copy()
    for point in points:
        xy = (round(point['x_px']), round(point['y_px']))
        radius = max(3, round(.5*point.get('diameter_source', point.get('diameter', 8))))
        cv2.circle(canvas, xy, radius, (220,140,0), 1, cv2.LINE_AA)
    for point in weak_points:
        xy = (round(point['x_px']), round(point['y_px']))
        cv2.drawMarker(canvas, xy, (160,0,180), cv2.MARKER_TILTED_CROSS, 9, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(Path(destination)/'triangle_errorbar_review_v46.png'), canvas):
        raise OSError('Could not save triangle/error-bar review overlay')

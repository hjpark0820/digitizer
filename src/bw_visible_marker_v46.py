"""Optional visible-first rescue at the *detection* raster filter.

Only already window-verified, raster-abstained bodies enter. The measured
legend raster, not its named glyph, defines ink, paper and outward transitions.
Additional ink outside that body is not a marker contradiction. Missing own
ink, dark observed holes, absent outward contrast, line nulls and every legend
rival still compete. This is not an occluded-marker recovery algorithm.
"""
from dataclasses import dataclass, asdict
from copy import copy
from itertools import product
import time

import cv2
import numpy as np
import bw_layered_composite_v46 as L

VERSION = 'visible-first-detection-observed-body-v2-preserved-physical-scales'


@dataclass(frozen=True)
class Config:
    maximum_centres: int = 120
    maximum_line_families: int = 8
    maximum_body_loss: float = .22
    minimum_required_ink: float = .92
    maximum_white_contradiction: float = .18
    minimum_boundary_support: float = .78
    minimum_boundary_visible: float = .65
    minimum_sector_support: float = .45
    minimum_independent_visible: float = .85
    minimum_independent_fraction: float = .30
    minimum_null_improvement: float = .10
    minimum_identity_margin: float = .035
    offsets: tuple = (-.5, 0., .5)


def _shift(a, dx, dy):
    return cv2.warpAffine(a.astype(np.float32), np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                          (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _sample(a, x, y):
    return cv2.remap(a.astype(np.float32), x.astype(np.float32)[None], y.astype(np.float32)[None],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]


def _outline(marker, cover, weight):
    """Measured outer silhouette and soft inward/outward brightness samples.

    The silhouette is used for coordinates, not a binary plot observation.
    A one-pixel exact transition is never required. Source-line uncertainty
    removes source ambiguity only; it cannot erase a target's white ink gap.
    """
    peak = float(marker.max())
    hull = cover >= .45
    edge = hull & ~(cv2.erode(hull.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    ys, xs = np.nonzero(edge)
    if peak < .08 or len(xs) < 8:
        return None
    soft = cv2.GaussianBlur(cover.astype(np.float32), (0, 0), .7)
    gy, gx = np.gradient(soft)
    nx, ny = -gx[ys, xs], -gy[ys, xs]
    norm = np.hypot(nx, ny)
    nx, ny = nx / np.maximum(norm, 1.e-6), ny / np.maximum(norm, 1.e-6)
    # The inward point sits on the observed stroke; the outer point measures
    # its paper-side transition, including antialiasing over a 2-pixel span.
    ix, iy = xs - .4 * nx, ys - .4 * ny
    ox, oy = xs + 1.6 * nx, ys + 1.6 * ny
    inside, outside = _sample(marker, ix, iy), _sample(marker, ox, oy)
    contrast = inside - outside
    reliable = ((norm > .025) & (contrast >= .15 * peak)
                & (_sample(weight, ix, iy) >= .2) & (_sample(weight, ox, oy) >= .2))
    cy, cx = (np.array(marker.shape) - 1) / 2
    sectors = (np.mod(np.arctan2(ys - cy, xs - cx), 2 * np.pi) * 8 / (2 * np.pi)).astype(int)
    represented = [s for s in range(8) if np.any(reliable & (sectors == s))]
    if len(represented) < 6 or reliable.sum() < .5 * len(xs):
        return None
    return dict(ix=ix[reliable], iy=iy[reliable], ox=ox[reliable], oy=oy[reliable],
                contrast=contrast[reliable], sectors=sectors[reliable],
                source_coverage=float(reliable.mean()), represented=represented)


class VisibleCompetition:
    def __init__(self, competition, cfg=Config()):
        self.base, self.cfg = competition, cfg
        competition._bank()
        self.variants = {key: bank['variants'] for key, bank in competition.banks.items()}
        self.cache = {}
        self.phase_cache = {}

    def _phase(self, dx, dy):
        if (dx, dy) in self.phase_cache:
            return self.phase_cache[dx, dy]
        b = self.base
        domain = _shift(b.context_body.astype(np.float32), dx, dy) > .01
        # The fixed canvas already includes a one-pixel exterior band. Keep
        # its shared weight/energy fixed across all half-pixel hypotheses too.
        weight = b.weight
        yy, xx = np.indices(domain.shape)
        center = (b.radius+dx, b.radius+dy)
        strokes, families = {}, []
        for lines in b.family_lines:
            ids=[]
            for line in lines:
                ident=id(line);ids.append(ident)
                if ident in strokes:
                    continue
                a=_shift(line, dx, dy)
                ys,xs=np.nonzero(a>.01)
                axis=xx-center[0] if np.ptp(xs)>=np.ptp(ys) else yy-center[1]
                exterior=[a*(~domain)*(axis*sign>0) for sign in (-1,1)]
                strokes[ident]=(a,exterior)
            full=np.zeros_like(weight)
            for ident in ids:
                full=1-(1-full)*(1-strokes[ident][0])
            families.append((ids,full))
        variants={}
        for key, rows in self.variants.items():
            variants[key]=[]
            for scale,(m,c,w) in rows.items():
                marker,cover,reliability=(_shift(a,dx,dy) for a in (m,c,w))
                outline=_outline(marker,cover,reliability)
                if outline is not None:
                    variants[key].append((scale,marker,cover,reliability,outline))
        data=dict(domain=domain,weight=weight,strokes=strokes,families=families,variants=variants)
        self.phase_cache[dx,dy]=data
        return data

    def _lines(self, patch, dx, dy):
        """Select nuisance hypotheses using only independently outside ink.

        Every identity receives the same set. The no-line null is always
        present; selecting a line never awards marker confidence by itself.
        """
        data = self._phase(dx,dy)
        domain=data['domain'];supported={}
        for ident,(a,weights) in data['strokes'].items():
            present=np.minimum(patch/np.maximum(a,.02),1.)
            supported[ident]=max(float(np.sum(w*present)/max(float(w.sum()),1.e-6)) for w in weights)>=.45
        valid, rows = [], []
        for ids,full in data['families']:
            moved=[data['strokes'][ident][0] for ident in ids]
            ok=all(supported[ident] for ident in ids)
            rows.append((moved, full))
            if ok:
                valid.append((float(np.sum(np.square(patch - full) * (~domain))), len(rows)-1))
        selected = [0] + [idx for _, idx in sorted(valid) if idx != 0][:self.cfg.maximum_line_families-1]
        # H1 refinement is budgeted, but the best independently supported
        # line-only null must not disappear merely because it was not one of
        # those selected H1 nuisance combinations.
        return [rows[i] for i in selected],np.asarray([rows[i][1] for _,i in valid])

    def score(self, center):
        center = tuple(map(float, center))
        if center in self.cache:
            return self.cache[center]
        started = time.perf_counter()
        b, cfg = self.base, self.cfg
        size = 2 * b.radius + 1
        h,w=b.ink.shape
        x,y=center
        box=[x-b.radius,y-b.radius,x+b.radius+1,y+b.radius+1]
        intersects=lambda a: box[0]<a[2] and box[2]>a[0] and box[1]<a[3] and box[3]>a[1]
        if (box[0]<0 or box[1]<0 or box[2]>w or box[3]>h
                or any(intersects(a) for a in getattr(b,'visible_exclusions',()))):
            q=dict(version=VERSION,scope='detection_full_window_verified_raster_abstention_only',
                   center=list(center),winner=None,reason='source_window_outside_image_or_protected_region',
                   margin=None,scores={},supported_scores={},seconds=time.perf_counter()-started)
            self.cache[center]=q
            return q
        patch = cv2.getRectSubPix(b.ink, (size, size), center)
        best, supported = {}, {}
        for dx, dy in product(cfg.offsets, repeat=2):
            data=self._phase(dx,dy)
            lines,all_nulls = self._lines(patch, dx, dy)
            common_energy=max(float(np.sum(data['weight']*patch*patch)),1.e-6)
            for key, variants in data['variants'].items():
                for scale,marker,cover,reliability,outline in variants:
                    row,passed = self._measure(patch, marker, cover, reliability, outline, lines,
                                        bool(b.source_meta[key]['source_open']),data['weight'],common_energy,all_nulls)
                    for item in (row,passed):
                        if item is not None:
                            item.update(center=[center[0]+dx, center[1]+dy], offset=[dx, dy], scale=float(scale))
                    if key not in best or row['loss'] < best[key]['loss']:
                        best[key] = row
                    if passed is not None and (key not in supported or passed['loss'] < supported[key]['loss']):
                        supported[key] = passed
        # An unresolved source rival is never silently discarded.
        unobservable = sorted(set(self.variants) - set(best))
        winner = min(supported, key=lambda k:supported[k]['loss']) if supported else None
        margin = None
        reason = 'no_directly_visible_marker'
        if winner is not None:
            rivals = [v['loss'] for k, v in best.items() if k != winner]
            margin = min(rivals) - supported[winner]['loss'] if rivals else 1.
            reason = 'direct_visible_marker_verified'
            if unobservable:
                winner, reason = None, 'unobservable_legend_rival'
            elif margin < cfg.minimum_identity_margin:
                winner, reason = None, 'direct_visible_identity_ambiguous'
        elif unobservable:
            reason = 'unobservable_legend_rival'
        q = dict(version=VERSION, scope='detection_full_window_verified_raster_abstention_only',
                 center=list(center), common_radius=int(b.radius), winner=winner, reason=reason, margin=margin,
                 scores=best, supported_scores=supported, unobservable_rivals=unobservable,
                 seconds=time.perf_counter()-started, source='prepared_observed_legend_rasters',
                 shared_localisations=True, shared_nuisance_hypotheses=True,
                 shared_identity_domain_and_denominator=True,
                 residual_policy='common_family_roi; negative residual always; positive residual inside proposed body only',
                 model_specific_exterior_relief_requires_signed_gray_contour_and_sector_guards=True,
                 outside_positive_residual_ignored=True, required_ink_and_hole_tests_preserved=True)
        self.cache[center] = q
        return q

    def _measure(self, patch, marker, cover, reliability, edge, lines, is_open, common_weight, common_energy, all_nulls):
        cfg = self.cfg
        peak = max(float(marker.max()), .05)
        domain = (cover >= .30) & (reliability >= .2)
        weight = reliability * domain
        mass = weight * marker * marker
        energy = max(float(mass.sum()), 1.e-6)
        hole = domain & (marker < .15 * peak)
        required = (marker >= .20 * peak) & (reliability >= .2)
        required_weight = marker * reliability * required
        # Identical body observation, reliability and denominator for every
        # H1 and every valid line-only H0. A marker cannot win by selecting a
        # weak paper-only H0 while a different observed line explains it.
        null_losses=np.sum(np.square(all_nulls-patch)*weight,axis=(1,2))/energy
        null_loss=float(null_losses.min())
        e = edge
        observed_contrast = _sample(patch, e['ix'], e['iy']) - _sample(patch, e['ox'], e['oy'])
        raw_support = np.clip(observed_contrast / np.maximum(e['contrast'], .03), 0, 1)
        observed_inside=_sample(patch,e['ix'],e['iy'])
        expected_inside=_sample(marker,e['ix'],e['iy'])
        hypotheses = []
        for family_index, (strokes, null) in enumerate(lines):
            orders = list(product(('front', 'behind'), repeat=len(strokes))) if is_open else [tuple('front' for _ in strokes)]
            for order in orders:
                for gain in self.base.cfg.gains:
                    body = np.clip(marker * gain, 0, 1)
                    rendered = L.compose(body, np.maximum(cover, body), strokes, order)
                    error = np.square(patch - rendered)
                    body_loss = float(np.sum(error * weight) / energy)
                    independent = mass * np.square(1 - null)
                    fraction = float(independent.sum() / energy)
                    remain = np.clip((patch-null)/np.maximum(1-null, .02), 0, 1)
                    visibility = float(np.sum(independent*np.clip(remain/np.maximum(body, .02), 0, 1))
                                       / max(float(independent.sum()), 1.e-6))
                    recall = float(np.sum(required_weight * np.minimum(patch/np.maximum(body, .02), 1))
                                   / max(float(required_weight.sum()), 1.e-6))
                    contradiction = (float(np.mean(np.clip((patch-rendered)/peak, 0, 1)[hole]))
                                     if hole.any() else 0.)
                    # A continued line may obscure a narrow contour segment,
                    # but unobserved/white segments never become occlusion.
                    line_edge = np.maximum(_sample(null, e['ix'], e['iy']), _sample(null, e['ox'], e['oy']))
                    observed_line_ink=observed_inside >= .75*np.maximum(expected_inside,.03)
                    independent_edge = (line_edge < .35 * peak) | ~observed_line_ink
                    coverage = float(independent_edge.mean())
                    boundary = float(raw_support[independent_edge].mean()) if independent_edge.any() else 0.
                    sectors = [float(raw_support[(e['sectors']==s) & independent_edge].mean())
                               if np.any((e['sectors']==s) & independent_edge) else None for s in e['represented']]
                    measured = [v for v in sectors if v is not None]
                    sector_ok = len(measured) >= 6 and min(measured, default=0.) >= cfg.minimum_sector_support
                    residual=patch-rendered
                    # All identities use ONE observation, source energy and
                    # uncertainty weight. Only excess ink OUTSIDE a proposed
                    # body is exempt. Different proposed exteriors are honest
                    # nuisance hypotheses, not different normalization; each
                    # must first prove its own signed outward contour.
                    identity_error=np.square(np.minimum(residual,0)) + np.square(np.maximum(residual,0))*domain
                    identity_pixel_loss=float(np.sum(identity_error*common_weight)/common_energy)
                    loss = identity_pixel_loss + .35*(1-boundary) + .20*(1-recall) + len(strokes)*self.base.cfg.line_cost
                    gates = dict(body_loss=body_loss <= cfg.maximum_body_loss,
                                 required_ink=recall >= cfg.minimum_required_ink,
                                 hole_paper=contradiction <= cfg.maximum_white_contradiction,
                                 outward_boundary=boundary >= cfg.minimum_boundary_support,
                                 visible_boundary=coverage >= cfg.minimum_boundary_visible,
                                 boundary_sectors=sector_ok,
                                 independent_ink=fraction >= cfg.minimum_independent_fraction and visibility >= cfg.minimum_independent_visible,
                                 line_null=(null_loss-body_loss) >= cfg.minimum_null_improvement)
                    hypotheses.append(dict(loss=float(loss), accepted=bool(all(gates.values())),
                        gates=gates, body_loss=body_loss, required_ink=recall, white_contradiction=contradiction,
                        identity_pixel_loss=identity_pixel_loss,identity_energy=common_energy,
                        boundary_support=boundary, boundary_visible=coverage, boundary_sectors=sectors,
                        independent_fraction=fraction, independent_visible=visibility,
                        null_improvement=null_loss-body_loss, gain=float(gain), line_family=family_index,
                        best_line_only_loss=null_loss,line_only_hypotheses=int(len(all_nulls)),
                        line_count=len(strokes), line_orders=list(order), source_boundary_coverage=e['source_coverage']))
        # Best failed rivals remain visible to identity competition too.
        best = min(hypotheses, key=lambda q:q['loss'])
        accepted=[q for q in hypotheses if q['accepted']]
        return best,min(accepted,key=lambda q:q['loss']) if accepted else None


def prior_scale_comparisons(competition, records, sites, cfg):
    """Reuse all already admitted physical scales, without fitting new sizes.

    An earlier same-body comparison may have admitted a measured native size
    for EVERY rival. Resetting the visible-first retry to global calibration
    alone would silently discard its strongest rival. Preserve both sets at
    each centre, grouping equal dictionaries so each expanded bank is built
    once. At most one expanded family is resident at a time.
    """
    competition._bank()
    base={key:set(map(float,bank['variants'])) for key,bank in competition.banks.items()}
    def signature(scales):
        return tuple((key,tuple(sorted(values))) for key,values in sorted(scales.items()))
    base_signature=signature(base)
    groups={};comparisons={};prior_counts={}
    by_center={center:[] for center in sites}
    for p in records:
        center=(float(p['aligned_x']),float(p['aligned_y']))
        if center in by_center:
            by_center[center].append(p)
    for center,rows in by_center.items():
        scales={key:set(values) for key,values in base.items()}
        contracts=[];invalid=False
        for p in rows:
            for field in ('raster_identity','raster_own_centre_identity'):
                physical=p.get(field,{}).get('physical_body_hypothesis')
                if not physical:
                    continue
                previous=physical.get('scale_hypotheses',{})
                if set(previous)!=set(base):
                    invalid=True;break
                try:
                    current={key:tuple(map(float,values)) for key,values in previous.items()}
                except (TypeError,ValueError):
                    invalid=True;break
                if any(not values or not np.isfinite(values).all() or min(values)<=0 for values in current.values()):
                    invalid=True;break
                contracts.append(signature(current))
                for key,values in current.items():
                    scales[key].update(values)
            if invalid:
                break
        if invalid:
            comparisons[center]=dict(version=VERSION,center=list(center),winner=None,margin=None,
                scope='detection_full_window_verified_raster_abstention_only',seconds=0.,
                reason='invalid_prior_physical_scale_contract',scores={},supported_scores={})
            continue
        sig=signature(scales)
        groups.setdefault(sig,[]).append(center)
        prior_counts[center]=len(set(contracts))
    summaries=[]
    for sig,centres in sorted(groups.items()):
        scales={key:values for key,values in sig}
        expanded=sig!=base_signature
        if expanded:
            bank=copy(competition)
            bank.banks={};bank.cache={};bank.phase_competitors={}
            bank.report=dict(competition.report)
            bank.scale_hypotheses=scales
            bank.radius=max(bank._variant(key,scale)[3] for key,values in sig for scale in values)
        else:
            bank=competition
        evaluator=VisibleCompetition(bank,cfg)
        additions={key:sorted(set(values)-base[key]) for key,values in sig}
        for center in centres:
            q=evaluator.score(center)
            q.update(scale_hypotheses_used={key:list(values) for key,values in sig},
                calibrated_scale_hypotheses_preserved=True,prior_physical_scales_preserved=True,
                prior_physical_contract_count=prior_counts[center],previously_admitted_extra_scales=additions,
                new_scale_search_performed=False)
            comparisons[center]=q
        summaries.append(dict(centres=len(centres),expanded_from_base=expanded,
            scale_hypotheses={key:list(values) for key,values in sig},common_radius=int(bank.radius)))
        # Releasing the previous family before building another bounds raster
        # banks and cached phase geometry independently of candidate count.
        del evaluator,bank
    return comparisons,summaries


def rescue_records(competition, records, kept, cfg=Config()):
    """Append only decisive new bodies; never reopen a retained body/identity."""
    from bw_marker_selection import duplicate_reason
    started = time.perf_counter()
    baseline = list(kept)
    retained = {id(p) for p in kept}
    eligible = []
    for p in records:
        if id(p) in retained or p.get('decision') != 'verified':
            continue
        q = p.get('raster_identity', {})
        if q.get('winner') is not None:
            continue
        if any(duplicate_reason(p, old) for old in baseline):
            p['visible_first_detection'] = dict(version=VERSION, accepted=False, reason='existing_body_retained')
            continue
        eligible.append(p)
    sites = sorted({(float(p['aligned_x']), float(p['aligned_y'])) for p in eligible})
    allowed = set(sites[:cfg.maximum_centres])
    comparisons,families=prior_scale_comparisons(competition,records,sorted(allowed),cfg) if allowed else ({},[])
    promoted = 0
    for p in eligible:
        center = (float(p['aligned_x']), float(p['aligned_y']))
        if center not in allowed:
            p['visible_first_detection'] = dict(version=VERSION, accepted=False, reason='explicit_candidate_budget')
            continue
        q = comparisons[center]
        p['visible_first_detection'] = q
        if q['winner'] != p['template']:
            continue
        row = q['supported_scores'][p['template']]
        dx, dy = row['offset']
        moved = dict(p, aligned_x=row['center'][0], aligned_y=row['center'][1])
        if p.get('footprint_polygon'):
            moved['footprint_polygon']=[[float(x)+dx,float(y)+dy] for x,y in p['footprint_polygon']]
        bounds=getattr(competition,'visible_plot_bounds',None)
        if bounds is not None and not (bounds[0]<=row['center'][0]<bounds[2] and bounds[1]<=row['center'][1]<bounds[3]):
            p['visible_first_detection_retention']='aligned_center_outside_plot'
            continue
        if any(duplicate_reason(moved, old) for old in baseline):
            p['visible_first_detection_retention'] = 'aligned_body_already_retained'
            continue
        p['pre_visible_first_center'] = list(center)
        p['aligned_x'], p['aligned_y'] = row['center']
        if p.get('footprint_polygon'):
            p['footprint_polygon'] = moved['footprint_polygon']
        p['selection_rank'] = 1.-row['loss']
        p.pop('exclusion_reason', None)
        p.pop('raster_deferred_reason', None)
        p['pre_contour_raster_retained'] = True
        p['visible_first_detection_retention'] = 'direct_visible_source_marker'
        kept.append(p)
        promoted += 1
    competition.report['visible_first_detection'] = dict(version=VERSION, enabled=True,
        scope='detection_full_window_verified_raster_abstention_only', config=asdict(cfg),
        eligible_records=len(eligible), eligible_centres=len(sites), tested_centres=len(allowed),
        budget_exhausted=len(sites)>cfg.maximum_centres, promoted_records=promoted,
        baseline_retained_records=len(baseline), seconds=time.perf_counter()-started,
        physical_scale_families=families,expanded_scale_families=sum(f['expanded_from_base'] for f in families),
        comparisons=[comparisons[center] for center in sorted(comparisons)])
    return kept

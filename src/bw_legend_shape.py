"""Evidence-only legend shape hints; prototype pixels never enter matching.

An attached line can alter polygon vertex counts and hollow-marker fill. Use
the envelope of observed pixels outside that nuisance band to measure shape,
and independently measure interior ink away from the line. Low confidence
returns unknown_marker instead of silently defaulting to a circle.
"""
from __future__ import annotations

import cv2
import numpy as np


def _prototypes(size=80):
    y, x = np.mgrid[:size, :size].astype(np.float32)
    x = (x + .5) / size
    y = (y + .5) / size
    return {
        'square': np.ones((size, size), bool),
        'circle': (2*x-1)**2 + (2*y-1)**2 <= 1.,
        'rhombus': np.abs(2*x-1) + np.abs(2*y-1) <= 1.,
        'triangle': np.abs(2*x-1) <= y,
        'inv_triangle': np.abs(2*x-1) <= 1-y,
    }


def _iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def _off_line_shape_scores(envelope, nuisance):
    """Fit shape measurement widths using visible rows, not a line's width.

    Missing centre rows can hide a diamond's widest point. Normalizing the
    remaining width as if it were complete would turn that diamond into a
    circle. Instead compare a bounded range of envelopes on the *observed*
    rows; nuisance pixels contribute neither a reward nor a mismatch.
    """
    # Leave room for the widest hypothesis. Otherwise its part outside the
    # tight crop is never penalized and a clipped wide ellipse mimics a square.
    pad = max(3, int(np.ceil(.30*envelope.shape[1])))
    envelope = np.pad(envelope, ((0,0),(pad,pad)))
    nuisance = np.pad(nuisance, ((0,0),(pad,pad)), mode='edge')
    ys, xs = np.nonzero(envelope)
    left, right, top, bottom = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    width, height = right-left, bottom-top
    scale = max(1., 80./height)
    sh, sw = round(envelope.shape[0]*scale), round(envelope.shape[1]*scale)
    target = cv2.resize(envelope, (sw,sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid = ~cv2.resize(nuisance.astype(np.uint8), (sw,sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    yy, xx = np.mgrid[:sh,:sw].astype(np.float32)
    xx = (xx+.5)/scale
    yy = ((yy+.5)/scale-top)/height
    scores = {kind: 0. for kind in _prototypes()}
    factors = np.linspace(1.,1.45,10) if nuisance.any() else (1.,)
    for factor in factors:
        horizontal = np.abs((xx-(left+right)/2)/(width*factor/2))
        vertical = np.abs(2*yy-1)
        candidates = {'square': (horizontal<=1) & (vertical<=1),
                      'circle': horizontal**2+vertical**2<=1,
                      'rhombus': horizontal+vertical<=1,
                      'triangle': (horizontal<=yy) & (yy>=0) & (yy<=1),
                      'inv_triangle': (horizontal<=1-yy) & (yy>=0) & (yy<=1)}
        for kind, proto in candidates.items():
            scores[kind] = max(scores[kind], _iou(target & valid, proto & valid))
    return scores


def classify_observed_shape(mask, nuisance=None):
    """Return (shape_hint, off-line fill fraction, auditable evidence).

    The convex envelope and ideal silhouettes are measurement regions only.
    Neither the caller's mask nor its nuisance map is modified or returned as
    reconstructed template ink. Shape scores are not calibrated probabilities.
    """
    observed = np.asarray(mask, dtype=bool)
    if observed.ndim != 2 or not observed.any():
        raise ValueError('Empty legend marker')
    line = np.zeros_like(observed) if nuisance is None else np.asarray(nuisance, dtype=bool)
    if line.shape != observed.shape:
        raise ValueError('Shape nuisance map must match the observed mask')
    ys, xs = np.nonzero(observed)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    m = observed[y0:y1, x0:x1].copy()
    line = line[y0:y1, x0:x1].copy()
    evidence = m & ~line
    ey, ex = np.nonzero(evidence)
    report = {'method': 'off_line_envelope_and_independent_fill',
              'observed_pixels': int(m.sum()), 'off_line_pixels': int(evidence.sum()),
              'nuisance_pixels': int((m & line).sum()),
              'template_pixels_reconstructed': False, 'shape_scores': {},
              'shape_confidence': 0., 'status': 'insufficient_evidence'}
    if len(ex) < 5 or len(np.unique(ey)) < 3 or len(np.unique(ex)) < 3:
        return 'unknown_marker', float(m.mean()), report
    hull = cv2.convexHull(np.column_stack((ex, ey)).astype(np.int32))
    envelope = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(envelope, hull, 1)
    # Padding is essential when a square fills its entire crop: otherwise a
    # distance transform has no outside background from which to measure.
    distance = cv2.distanceTransform(np.pad(envelope, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    diameter = max(int(np.ptp(ex))+1, int(np.ptp(ey))+1)
    inner = distance >= max(1.5, .15*diameter)
    independent = inner & ~line
    # Prefer the deepest sufficiently supported off-line interior. Thick open
    # borders must not count as a solid fill merely because the glyph is small.
    deepest = float(distance[~line].max()) if (~line).any() else 0.
    for fraction in (.45, .60, .75):
        candidate = (distance >= max(1.5, fraction*deepest)) & ~line
        if candidate.sum() >= max(6, int(np.ceil(.12*inner.sum()))):
            independent = candidate
    fill = float(m[independent].mean()) if independent.any() else float(m[evidence].mean())
    count = int(independent.sum())
    enough_fill = count >= max(3, int(np.ceil(.08 * inner.sum())))
    report.update(independent_interior_pixels=count, independent_fill=fill,
                  sufficient_fill_evidence=bool(enough_fill))

    # Topology supplements average fill: a thick hollow border can dominate
    # every off-line sample while a clearly enclosed white region still exists.
    # Several substantial internal holes, however, indicate a patterned glyph,
    # not an ordinary filled circle inferred from its outer envelope alone.
    contours,hierarchy=cv2.findContours(m.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    outer_index=max(range(len(contours)),key=lambda i:cv2.contourArea(contours[i]))
    outer=contours[outer_index]
    outer_hull_area=max(1.,cv2.contourArea(cv2.convexHull(outer)))
    solidity=float(cv2.contourArea(outer)/outer_hull_area)
    hole_fractions=sorted([float(cv2.contourArea(c)/outer_hull_area)
                          for i,c in enumerate(contours) if hierarchy[0,i,3]==outer_index],reverse=True)
    substantial_holes=[value for value in hole_fractions if value>=.012]
    patterned=len(substantial_holes)>=3 and sum(substantial_holes)>=.08
    hollow=(bool(hole_fractions) and (hole_fractions[0]>=.16 or sum(hole_fractions[:2])>=.22))
    report.update(outer_solidity=solidity, enclosed_white_fractions=hole_fractions,
                  patterned_internal_evidence=bool(patterned), strong_hollow_evidence=bool(hollow))

    # Test clearly non-convex crosses before the convex silhouette comparison.
    # A nuisance-bearing crossing cannot alone prove a genuine cross marker.
    occupancy = float((m & ~line).sum() / max(1, (envelope.astype(bool) & ~line).sum()))
    report['convex_occupancy'] = occupancy
    cross_scores = {}
    if not line.any() and occupancy < .77 and fill > .5:
        norm = cv2.resize(m.astype(np.uint8), (80,80), interpolation=cv2.INTER_NEAREST).astype(bool)
        yy, xx = np.mgrid[:80, :80].astype(np.float32)
        xx = (xx+.5)/80-.5
        yy = (yy+.5)/80-.5
        for kind in ('plus_marker', 'x_marker'):
            values = []
            for half_width in (.07, .10, .14, .18, .22):
                proto = ((np.abs(xx)<half_width) | (np.abs(yy)<half_width)) if kind=='plus_marker' else (
                    (np.abs(xx-yy)<half_width*1.42) | (np.abs(xx+yy)<half_width*1.42))
                values.append(_iou(norm, proto))
            cross_scores[kind] = max(values)
        ranked_cross = sorted(cross_scores, key=cross_scores.get, reverse=True)
        winner = ranked_cross[0]
        if cross_scores[winner] >= .78 and cross_scores[winner]-cross_scores[ranked_cross[1]] >= .10:
            report.update(status='classified', shape_scores=cross_scores,
                          shape_confidence=cross_scores[winner], shape_hint=winner)
            return winner, fill, report
    report['cross_scores'] = cross_scores

    scores = _off_line_shape_scores(envelope, line)
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, runner_up = ranked[:2]
    margin = scores[best] - scores[runner_up]
    # A raster-rounded square and circle can have similar areas. Require
    # agreement across several contour simplifications to use straight-corner
    # evidence as a tie-breaker; unlike the old classifier, other vertex counts
    # never imply "circle" and one polygon cannot decide the class by itself.
    votes = []
    hx, hy, hw, hh = cv2.boundingRect(hull)
    for epsilon in (.020,.030,.040,.050,.060):
        polygon = cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True).reshape(-1,2)
        if len(polygon)==4:
            corners = np.array([[hx,hy],[hx+hw-1,hy],[hx,hy+hh-1],[hx+hw-1,hy+hh-1]])
            near = np.linalg.norm(polygon[:,None,:]-corners[None,:,:],axis=2).min(axis=1).mean()
            votes.append('square' if near<.23*max(hw,hh) else 'rhombus')
        elif len(polygon)==3:
            mean_y=float(polygon[:,1].mean())
            votes.append('triangle' if mean_y>hy+(hh-1)/2 else 'inv_triangle')
    consensus={kind:votes.count(kind) for kind in set(votes)}
    corroborated = next((kind for kind,count in consensus.items()
                        if scores[best]-scores[kind]<=.04 and
                        ((count>=3 and scores[kind]>=.80) or
                         (count>=4 and scores[kind]>=.74 and
                          scores[kind]-max(value for key,value in scores.items() if key!=kind)>=.07))),None)
    if corroborated:
        best=corroborated
        runner_up=max((key for key in scores if key!=best),key=scores.get)
        margin=scores[best]-scores[runner_up]
    report.update(shape_scores=scores, best_shape=best, runner_up=runner_up,
                  shape_margin=margin, shape_confidence=float(scores[best]),
                  polygon_consensus=consensus, corroborated_shape=corroborated)
    # Strongly asymmetric/fragmentary marks must not become ideal circles or
    # squares simply because those shapes were the least bad alternatives.
    shape_ok = ((scores[best] >= .80 and margin >= .035) or
                (scores[best] >= .74 and margin >= .10) or bool(corroborated))
    if not shape_ok:
        report.update(status='uncertain_shape', shape_hint='unknown_marker')
        return 'unknown_marker', fill, report
    if patterned or (not line.any() and fill>=.65 and solidity<.84):
        report.update(status='patterned_or_nonstandard', shape_hint='unknown_marker')
        return 'unknown_marker', fill, report
    if not hollow and (not enough_fill or .38 < fill < .65):
        report.update(status='uncertain_fill', shape_hint='unknown_marker')
        return 'unknown_marker', fill, report
    prefix = 'open' if hollow else 'filled' if fill >= .65 else 'open'
    name = prefix + '_' + best
    report.update(status='classified', shape_hint=name)
    return name, fill, report

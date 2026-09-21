"""Image-only comparison of a connector and a connector plus observed glyph.

Connector parameters are fitted exclusively outside the complete marker body.
The fitted connector is then frozen for both reconstruction hypotheses. Colour
occlusion only excuses missing target ink; it never supplies positive support.
No reference positions, identities, nominal shape names or learned weights are
used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    context_radius_fraction: float = 1.85
    body_margin_pixels: float = 1.25
    maximum_line_seeds: int = 7
    maximum_lines: int = 2
    minimum_line_confidence: float = .52
    confident_line_threshold: float = .72
    line_only_gain_ceiling: float = .055
    line_only_support_ceiling: float = .075
    score_threshold: float = .5
    interior_void_guard: bool = True
    physical_paper_guard: bool = True


def _configuration(config):
    if config is None:
        return WindowConfig()
    if isinstance(config, dict):
        valid = {field.name for field in fields(WindowConfig)}
        return WindowConfig(**{key: value for key, value in config.items() if key in valid})
    return config


def _crop(array, x, y, radius):
    """Resample on an odd canvas whose centre is exactly (x, y), not rounded."""
    side = 2*radius+1
    transform = np.float32([[1, 0, radius-x], [0, 1, radius-y]])
    return cv2.warpAffine(np.asarray(array, np.float32), transform, (side, side),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _template_map(template, key, radius, default=None, nearest=False):
    value = template.get(key, default)
    if value is None:
        value = np.zeros_like(template['soft'])
    value = np.asarray(value, np.float32)
    center = template.get('center', [(value.shape[1]-1)/2, (value.shape[0]-1)/2])
    transform = np.float32([[1, 0, radius-float(center[0])],
                            [0, 1, radius-float(center[1])]])
    return cv2.warpAffine(value, transform, (2*radius+1, 2*radius+1),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)


def _enclosed_holes(ink):
    """Four-connected enclosed paper, excluding every exterior AA pixel."""
    paper = (ink < .15).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    holes = np.zeros_like(paper, bool)
    height, width = paper.shape
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if x > 0 and y > 0 and x+w < width and y+h < height and area >= 2:
            holes[labels == index] = True
    return holes


def _line_seeds(observed, external, diameter, maximum):
    binary = ((observed >= .24) & external).astype(np.uint8)*255
    if np.count_nonzero(binary) < max(6, .8*diameter):
        return []
    segments = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=max(4, round(.35*diameter)),
        minLineLength=max(4, .55*diameter), maxLineGap=max(1, .20*diameter))
    if segments is None:
        return []
    center = (observed.shape[0]-1)/2
    result = []
    ordered = sorted(np.asarray(segments).reshape(-1, 4),
                     key=lambda p: -float((p[2]-p[0])**2+(p[3]-p[1])**2))
    for x0, y0, x1, y1 in ordered:
        angle = math.atan2(float(y1-y0), float(x1-x0)) % math.pi
        normal = np.array([-math.sin(angle), math.cos(angle)])
        offset = float(np.dot(normal, [(x0+x1)/2-center, (y0+y1)/2-center]))
        if any(abs(math.sin(angle-a)) < .12 and abs(offset-b) < max(2., .25*diameter)
               for a, b in result):
            continue
        result.append((angle, offset))
        if len(result) >= maximum:
            break
    return result


def _cross_sections(observed, other, valid, angle, offset, inner, outer):
    """Measure actual external stroke profiles, without a diameter width cap."""
    along = np.r_[-np.linspace(inner+1, outer-1, 7), np.linspace(inner+1, outer-1, 7)]
    across = np.arange(-outer, outer+.25, .5, dtype=np.float32)
    vx, vy = math.cos(angle), math.sin(angle)
    nx, ny = -vy, vx
    center = (observed.shape[0]-1)/2
    mx = (center+along[:, None]*vx+across[None, :]*nx).astype(np.float32)
    my = (center+along[:, None]*vy+across[None, :]*ny).astype(np.float32)
    sampled = cv2.remap(observed, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    covered = cv2.remap(other, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    available = cv2.remap(valid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    records = []
    for index, profile in enumerate(sampled):
        binary = (profile >= .22) & (available[index] > .95)
        edges = np.diff(np.r_[False, binary, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        choices = []
        for start, end in zip(starts, ends):
            # A run must have observed paper on either flank. A cropped or
            # all-ink profile cannot supply a measured connector width.
            if (start == 0 or end == len(profile) or end-start < 2 or
                    available[index, start-1] <= .95 or available[index, end] <= .95):
                continue
            midpoint = float((across[start]+across[end-1])/2)
            distance = max(0., abs(midpoint-offset)-.25*(end-start))
            choices.append((distance, start, end))
        if not choices:
            continue
        distance, start, end = min(choices)
        if distance > max(2., .18*inner):
            continue
        if float(covered[index, start:end].mean()) > .25:
            continue
        lo, hi = max(0, start-2), min(len(profile), end+2)
        strength = float(np.percentile(profile[start:end], 85))
        mass = float(profile[lo:hi].sum())
        if strength < .35 or mass <= 0:
            continue
        midpoint = float(np.dot(profile[lo:hi], across[lo:hi])/mass)
        width = .5*mass/strength
        records.append((float(along[index]), midpoint, width, strength))
    return np.asarray(records, np.float32).reshape(-1, 4)


def _fit_line(observed, other, valid, seed, inner, outer, diameter):
    angle, offset = seed
    records = _cross_sections(observed, other, valid, angle, offset, inner, outer)
    if len(records) < 4:
        return None
    # External centroids refine the Hough orientation. No interior pixels
    # participate in this regression or subsequent width measurements.
    design = np.column_stack((records[:, 0], np.ones(len(records))))
    slope, shift = np.linalg.lstsq(design, records[:, 1], rcond=None)[0]
    residual = np.abs(records[:, 1]-design@np.array([slope, shift]))
    keep = residual <= max(.8, float(np.median(residual))*2.5)
    if keep.sum() >= 4:
        slope, shift = np.linalg.lstsq(design[keep], records[keep, 1], rcond=None)[0]
    if abs(float(slope)) > .25:
        return None
    angle = angle+math.atan(float(slope))
    offset = float(shift)/math.sqrt(1+float(slope)**2)
    records = _cross_sections(observed, other, valid, angle, offset, inner, outer)
    if len(records) < 4:
        return None
    width = float(np.median(records[:, 2]))
    amplitude = float(np.median(records[:, 3]))
    offset = float(np.median(records[:, 1]))
    if abs(offset) > inner+.5*width:
        return None
    error = float(np.median(np.abs(records[:, 1]-offset)))
    spread = float(np.median(np.abs(records[:, 2]-width)))/max(width, 1.)
    negative = int(np.count_nonzero(records[:, 0] < 0))
    positive = int(np.count_nonzero(records[:, 0] > 0))
    both = min(negative, positive) >= 3
    side = 0 if both else (1 if positive >= negative else -1)
    side_count = len(records) if both else max(negative, positive)
    completeness = min(1., side_count/(12. if both else 6.))
    confidence = completeness*math.exp(-error/max(1., .25*width)-1.8*spread)
    if not both:
        confidence *= .9
    yy, xx = np.mgrid[:observed.shape[0], :observed.shape[1]].astype(np.float32)
    xx -= (observed.shape[1]-1)/2; yy -= (observed.shape[0]-1)/2
    longitudinal = xx*math.cos(angle)+yy*math.sin(angle)
    perpendicular = -xx*math.sin(angle)+yy*math.cos(angle)-offset
    rendered = amplitude*np.clip(.5*width+.5-np.abs(perpendicular), 0., 1.)
    if side:
        # An external ray fixes its direction and width, but cannot reveal
        # where it ends underneath the candidate. Extrapolate across the
        # whole body instead of inventing a cap at the candidate centre.
        rendered *= np.clip(side*longitudinal+inner+.5, 0., 1.)
    return dict(map=rendered.astype(np.float32), angle=float(angle), offset=offset,
                width=width, amplitude=amplitude, confidence=float(confidence),
                side=side, cross_sections=len(records))


def _connector(observed, other, valid, external, inner, outer, diameter, config):
    seeds = _line_seeds(observed, external, diameter, config.maximum_line_seeds)
    fits = [_fit_line(observed, other, valid, seed, inner, outer, diameter) for seed in seeds]
    fits = [fit for fit in fits if fit is not None and fit['confidence'] >= config.minimum_line_confidence]
    baseline = float((observed*external).sum())
    line = np.zeros_like(observed)
    accepted = []
    for _ in range(min(2, max(0, config.maximum_lines))):
        best = None
        old_error = float((np.abs(observed-line)*external).sum())
        for fit in fits:
            if any(fit is old for old in accepted):
                continue
            if any(abs(math.sin(fit['angle']-old['angle'])) < .22 for old in accepted):
                continue
            combined = np.maximum(line, fit['map'])
            error = float((np.abs(observed-combined)*external).sum())
            improvement = old_error-error
            if best is None or improvement > best[0]:
                best = (improvement, fit, combined)
        if best is None or best[0] < max(3., .10*baseline):
            break
        _, fit, line = best
        accepted.append(fit)
    # Local external precision measures extrapolation credibility, while
    # unrelated external ink need not reduce the fitted line's confidence.
    if accepted:
        domain = external & (line > .08)
        fit_error = float((np.abs(observed-line)*domain).sum())/max(float((line*domain).sum()), 1.)
        confidence = min(fit['confidence'] for fit in accepted)*math.exp(-1.6*fit_error)
    else:
        confidence = 0.
    return line, accepted, float(confidence)


def verify_window(own, other, template, x, y, *, config=None, debug=False, ignore_mask=None,
                  guide_model=None, blend_uncertainty=None, raw_bgr=None, filled_prior=None,
                  native_pixel_area=1.,target_confidence=None):
    """Return comparable continuous H0/H1 losses and a transparent 0..1 score.

    ``own`` and ``other`` are complete processed-plot float membership maps;
    ``x,y`` use their pixel-centre coordinates. ``marker_support`` and
    ``off_line_support`` use a fixed observed-template mass denominator and
    therefore cannot increase when the known-other occlusion map increases.
    Confident line-only explanations reject a candidate. Filled-template raw
    paper contradictions can additionally cap or subtract its score; foreign
    ink never supplies positive target support. Acceptance uses the configured
    score threshold (the production selector may use a different cutoff).
    """
    config = _configuration(config)
    own = np.asarray(own, np.float32)
    other = np.asarray(other, np.float32)
    if own.ndim != 2 or other.shape != own.shape:
        raise ValueError('own and other must be same-sized 2D processed-plot arrays')
    if raw_bgr is not None and np.shape(raw_bgr)!=(*own.shape,3):
        raise ValueError('Raw BGR plot pixels must match membership geometry')
    if target_confidence is not None and np.shape(target_confidence)!=own.shape:
        raise ValueError('Target colour confidence must match membership geometry')
    ignored = None if ignore_mask is None else np.asarray(ignore_mask, np.float32)
    if ignored is not None and (ignored.shape != own.shape or not np.isfinite(ignored).all()
                                or np.any((ignored < 0) | (ignored > 1))):
        raise ValueError('ignore_mask must be a same-sized finite 0..1 array')
    if not np.isfinite([x, y]).all():
        raise ValueError('Candidate coordinates must be finite')
    if blend_uncertainty is not None:
        blend_uncertainty = np.asarray(blend_uncertainty, np.float32)
        if (blend_uncertainty.shape != own.shape or not np.isfinite(blend_uncertainty).all()
                or np.any((blend_uncertainty < 0) | (blend_uncertainty > 1))):
            raise ValueError('blend_uncertainty must be a same-sized finite 0..1 array')
    if guide_model is not None:
        if len(guide_model)!=3 or any(np.shape(v)!=own.shape for v in guide_model):
            raise ValueError('guide_model needs same-sized middle, lower and upper maps')
    diameter = max(3., float(template['diameter']))
    source = np.asarray(template['soft'], np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError('template soft must be a finite 2D array')
    center = template.get('center', [(source.shape[1]-1)/2, (source.shape[0]-1)/2])
    sy, sx = np.nonzero(source > .12)
    body_radius = max(.5*diameter, float(np.hypot(sx-center[0], sy-center[1]).max()) if len(sx) else 0.)
    inner = body_radius+config.body_margin_pixels
    outer = max(config.context_radius_fraction*diameter, inner+6.)
    radius = int(math.ceil(outer+2))
    # Crop first to avoid allocating two full image arrays per candidate.
    # Remove ignored ink before subpixel interpolation (masked neighbours
    # must not leak back into an observed pixel).
    if ignored is None:
        observed = np.clip(_crop(own, x, y, radius), 0., 1.)
        occluder = np.clip(_crop(other, x, y, radius), 0., 1.)
        available = 1.
    else:
        left, top = max(0, math.floor(x-radius)-1), max(0, math.floor(y-radius)-1)
        right, bottom = min(own.shape[1], math.ceil(x+radius)+2), min(own.shape[0], math.ceil(y+radius)+2)
        if right <= left or bottom <= top:
            observed = np.zeros((2*radius+1,2*radius+1),np.float32)
            occluder = observed.copy()
            available = 1.
        else:
            region = np.s_[top:bottom, left:right]
            local_available = 1-ignored[region]
            observed = np.clip(_crop(own[region]*local_available, x-left, y-top, radius), 0., 1.)
            occluder = np.clip(_crop(other[region]*local_available, x-left, y-top, radius), 0., 1.)
            available = 1-np.clip(_crop(ignored[region], x-left, y-top, radius), 0., 1.)
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    # Compute boundary coverage locally; allocating a full-plot all-ones
    # image for every candidate is unnecessarily expensive.
    source_x, source_y = xx+float(x), yy+float(y)
    valid = (np.clip(source_x+1., 0., 1.)*np.clip(own.shape[1]-source_x, 0., 1.)*
             np.clip(source_y+1., 0., 1.)*np.clip(own.shape[0]-source_y, 0., 1.)).astype(np.float32)
    valid *= available
    blend = (np.zeros_like(observed) if blend_uncertainty is None else
             np.clip(_crop(blend_uncertainty, x, y, radius), 0, 1))
    # Retain at least 35% of the deficit. Uncertainty never earns support,
    # changes the occluder map, or shrinks the positive-evidence denominator.
    missing_weight = 1-.65*blend
    guide,guide_low,guide_high=(np.zeros_like(observed) for _ in range(3))
    if guide_model is not None:
        guide,guide_low,guide_high=(np.clip(_crop(v,x,y,radius),0,1)*available for v in guide_model)
    expected = np.clip(_template_map(template, 'soft', radius), 0., 1.)
    uncertain = _template_map(template, 'uncertain', radius, nearest=True) > .5
    rim = _template_map(template, 'observed_rim', radius, nearest=True) > .5
    hole = (_template_map(template, 'hole_core', radius, nearest=True) > .5
            if 'hole_core' in template else _enclosed_holes(expected))
    # An observed legend connector may bisect an otherwise hollow face.
    # Remove only explicitly uncertain interior connector pixels, never rim
    # pixels, when evaluating a connector-independent hollow composition.
    marker = expected.copy()
    if hole.any():
        marker[uncertain & ~rim] = 0.
    face = (_template_map(template, 'face', radius, nearest=True) > .5
            if 'face' in template else ((marker > .15) | hole))
    halo = (_template_map(template, 'boundary_uncertain', radius, nearest=True) > .5
            if 'boundary_uncertain' in template else
            (cv2.dilate((expected > .12).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            & ~(expected > .12) & ~hole)
    radial = np.hypot(xx, yy)
    external = (radial >= inner) & (radial <= outer) & (valid > .95)
    # Dot geometry/intensity is frozen from distant clean repeats. Fit only
    # the remaining connector/stem evidence outside the marker body.
    line, fits, confidence = _connector(np.maximum(observed-guide,0), occluder, valid, external, inner, outer, diameter, config)
    # Both hypotheses are scored on exactly the same pixels and denominator.
    # A half-weight exterior AA allowance never weakens true hole constraints.
    domain = ((radial <= inner)*valid).astype(np.float32)
    domain[halo & ~hole] *= .5
    visible = 1.-occluder
    mass = max(float(marker.sum()), 1.)
    physical_maps={}
    physical=dict(applied=False,reason='disabled' if not config.physical_paper_guard else 'raw_pixels_unavailable')
    # Freeze connector fitting and positive colour evidence. Only negative
    # evidence changes: uncertain coloured ink is not a physical white hole.
    if config.physical_paper_guard and raw_bgr is not None:
        from color_window_physical_v46 import evaluate
        # Do not let ignored raw pixels leak into a fractional-centre sample.
        # Work on a small ROI and normalize by observed interpolation weight.
        rl,rt=max(0,math.floor(x-radius)-1),max(0,math.floor(y-radius)-1)
        rr,rb=min(own.shape[1],math.ceil(x+radius)+2),min(own.shape[0],math.ceil(y+radius)+2)
        raw_window=np.zeros((*marker.shape,3),np.float32)
        if rr>rl and rb>rt:
            region=np.s_[rt:rb,rl:rr]
            raw_tile=np.asarray(raw_bgr[region],np.float32)
            if not np.isfinite(raw_tile).all() or np.any((raw_tile<0)|(raw_tile>255)):
                raise ValueError('Raw window pixels must be finite BGR 0..255')
            weights=np.ones((rb-rt,rr-rl),np.float32) if ignored is None else 1-ignored[region]
            sampled_weight=_crop(weights,x-rl,y-rt,radius)
            sampled=_crop(raw_tile*weights[...,None],x-rl,y-rt,radius)
            np.divide(sampled,sampled_weight[...,None],out=raw_window,where=sampled_weight[...,None]>1e-6)
            raw_window=np.clip(raw_window,0,255)
        physical_maps,physical=evaluate(raw_window,template,marker,observed,occluder,valid,
            prior=filled_prior,native_pixel_area=native_pixel_area,
            target_confidence=None if target_confidence is None else _crop(target_confidence,x,y,radius),
            legacy_missing_weight=missing_weight)
        if physical['applied']:
            occluder=physical_maps['other'];visible=1.-occluder
            missing_weight=physical_maps['missing_weight']

    def loss(model):
        # The same nuisance uncertainty is used for both explanations. It
        # cannot explain missing glyph ink where the guide has no support.
        low=np.maximum(model,guide_low)
        high=np.maximum(model,guide_high)
        missing_ink = np.maximum(low-observed, 0.)*visible*missing_weight
        extra_ink = np.maximum(observed-high, 0.)
        return float(((missing_ink+extra_ink)*domain).sum())/mass

    h0 = line
    compositions = [('connector_through_marker', np.maximum(line, marker))]
    if hole.any():
        compositions.append(('paper_face_hides_connector', np.maximum(line*(~face), marker)))
        if np.any(expected != marker):
            compositions.append(('observed_legend_composition', np.maximum(line, expected)))
    losses = [loss(model) for _, model in compositions]
    index = int(np.argmin(losses))
    composition, h1 = compositions[index]
    loss_h0, loss_h1 = loss(h0), losses[index]
    gain = loss_h0-loss_h1
    independent=np.maximum(observed-guide_high,0)
    support = float(np.minimum(independent, marker).sum())/mass
    missing = float((np.maximum(marker-observed, 0.)*visible*valid*missing_weight).sum())/mass
    off_line = float(np.minimum(np.maximum(observed-np.maximum(line,guide_high), 0.), marker).sum())/mass
    h0=np.maximum(h0,guide)
    h1=np.maximum(h1,guide)
    extra = float((np.maximum(observed-h1, 0.)*domain).sum())/mass
    hole_extra = float((np.maximum(observed-h1, 0.)*hole*valid).sum())/mass
    observed_fraction = float((marker*visible*valid).sum())/mass
    # Gain and off-line support must both exist. One visible lobe can provide
    # them; no opposite-side or quadrant-count requirement is imposed.
    positive = min(1., off_line/.20)*min(1., max(0., gain)/.20)
    quality = positive*(.55+.45*min(1., support))*math.exp(-1.5*missing-.8*extra-.8*hole_extra)
    line_only = bool(confidence >= config.confident_line_threshold and
                     gain <= config.line_only_gain_ceiling and
                     off_line <= config.line_only_support_ceiling)
    if line_only:
        quality = 0.
        reason = 'externally_supported_line_only'
    elif support < .03:
        reason = 'no_positive_marker_ink'
    elif gain <= 0:
        reason = 'marker_does_not_improve_reconstruction'
    elif quality >= config.score_threshold:
        reason = 'observed_marker_improves_reconstruction'
    elif missing > .25:
        reason = 'unexplained_missing_marker_ink'
    else:
        reason = 'insufficient_independent_marker_evidence'
    quality = float(np.clip(quality, 0., 1.))
    pre_paper_score=quality
    if physical.get('applied'):
        quality=max(0.,quality-physical['paper_penalty'])
        if physical['paper_penalty']>0 and quality<config.score_threshold:
            reason='raw_white_paper_contradiction'
    pre_interior_score = quality
    interior_maps = {}
    interior = dict(applied=False, score_ceiling=1., reason='disabled')
    if config.interior_void_guard:
        from marker_interior_void_v46 import inspect_interior
        interior = inspect_interior(marker, observed, occluder, valid, diameter,
            hole=hole, uncertain=uncertain, debug=debug, missing_weight=missing_weight,
            paper_probability=physical_maps.get('paper'),native_pixel_area=native_pixel_area)
        interior_maps = interior.pop('maps', {})
        if interior['score_ceiling'] < quality:
            quality = interior['score_ceiling']
            reason = 'unexplained_connected_interior_void'
    result = dict(score=quality, quality_score=quality, accepted=quality >= config.score_threshold,
        reconstruction_gain=float(gain), loss_h0=float(loss_h0), loss_h1=float(loss_h1),
        marker_support=float(support), missing=float(missing), extra=float(extra),
        blend_uncertain_fraction=float((marker*blend*valid).sum())/mass,
        missing_before_blend_weight=float((np.maximum(marker-observed, 0.)*visible*valid).sum())/mass,
        off_line_support=float(off_line), line_confidence=confidence,
        line_only_reject=line_only, reason=reason, diagnostic_reason=reason,
        hole_extra=float(hole_extra), observed_fraction=float(observed_fraction),
        ignored_marker_fraction=float((marker*(1-available)).sum())/mass,
        guide_explained_marker_fraction=float(np.minimum(guide_high,marker).sum())/mass,
        line_width=float(max((fit['width'] for fit in fits), default=0.)),
        score_before_interior_guard=pre_interior_score, interior_void=interior,
        score_before_paper_penalty=pre_paper_score,physical_paper=physical,
        line_count=len(fits), composition=composition,
        connector_parameters=[{key: value for key, value in fit.items() if key != 'map'} for fit in fits])
    if debug:
        result['maps'] = dict(observed=observed, other=occluder, expected=expected,
            template=marker, line=line, guide=guide, guide_upper=guide_high, independent=independent,
            H0=h0, H1=h1, external_fit_mask=external.astype(np.float32),
            hole=hole.astype(np.float32), face=face.astype(np.float32),
            missing=np.maximum(marker-observed, 0.)*visible*valid*missing_weight,
            missing_weight=missing_weight, blend_uncertainty=blend,
            available=valid,
            extra=np.maximum(observed-h1, 0.)*domain,
            off_line=np.minimum(np.maximum(observed-np.maximum(line,guide_high), 0.), marker))
        result['maps'].update({'interior_'+k: v for k, v in interior_maps.items()})
        result['maps'].update({'physical_'+k:v for k,v in physical_maps.items()})
        result['crop_origin'] = [float(x-radius), float(y-radius)]
        result['crop_center'] = [radius, radius]
    return result

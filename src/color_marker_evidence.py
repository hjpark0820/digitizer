"""Image-only colour and legend evidence for the generic hybrid marker detector.

No reference points, dose schedules, antibody names, shape labels or pretrained
detector are used here.  Caller-supplied boxes are structural inputs, expressed
in full-source, half-open ``(left, top, right, bottom)`` pixel coordinates.
Membership is continuous observed ink strength, not a hard colour assignment.
An independent unknown class prevents arbitrary ink from becoming a marker.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from color_membership_v46 import chromatic_evidence, reference_contrast

VERSION = "image_only_palette_and_legend_evidence_v46_shared_chromatic"


def _box(value, shape):
    if value is None or len(value) != 4:
        raise ValueError("A four-coordinate half-open source box is required")
    h, w = shape[:2]
    x0, y0 = np.floor(value[:2]).astype(int)
    x1, y1 = np.ceil(value[2:]).astype(int)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty source box: {value}")
    return [int(x0), int(y0), int(x1), int(y1)]


def _paper(image):
    flat = image.reshape(-1, 3).astype(np.float32)
    bright = flat.mean(axis=1)
    selected = flat[bright >= np.percentile(bright, 90)]
    return np.percentile(selected, 75, axis=0).astype(np.float32)


def _estimate_model(patch, rgb=None, paper_bgr=None):
    paper = _paper(patch) if paper_bgr is None else np.asarray(paper_bgr, np.float32)
    flat = patch.reshape(-1, 3).astype(np.float32)
    contrast = np.linalg.norm(paper-flat, axis=1)
    ink = contrast >= max(12., float(np.percentile(contrast, 65)))
    if not ink.any():
        raise ValueError("Swatch has no distinguishable ink")
    if rgb is None:
        values = flat[ink]
        chroma = np.ptp(values, axis=1)
        # Coloured legend letters are harmless, black adjacent letters must
        # not determine the colour of a genuinely chromatic graphical key.
        colourful = chroma >= 30
        if colourful.sum() >= max(3, .15*len(values)):
            values = values[colourful]
            strength = np.ptp(values, axis=1)
        else:
            strength = np.linalg.norm(paper-values, axis=1)
        core = values[strength >= np.percentile(strength, 65)]
        bgr = np.median(core, axis=0).astype(np.float32)
    else:
        bgr = np.asarray(rgb, np.float32)[::-1]
    direction = paper-bgr
    length = float(np.linalg.norm(direction))
    if length < 12:
        raise ValueError("Legend colour is indistinguishable from its paper")
    delta = paper-flat
    alpha = (delta@direction)/(length*length)
    residual = np.linalg.norm(delta-alpha[:, None]*direction, axis=1)
    selected = (alpha >= .4) & (alpha <= 1.25) & ink
    sigma = float(np.percentile(residual[selected], 80)) if selected.any() else 6.
    sigma = float(np.clip(sigma+3., 5., 16.))
    return dict(bgr=bgr, paper_bgr=paper, residual_sigma=sigma,
                achromatic=bool(np.ptp(bgr) < 24), contrast=length,
                colour_reference_contrast=reference_contrast(flat, paper, bgr),
                membership_policy='chromatic_hue_contrast_v46_neutral_ray')


def _swatch_model(image, box, rgb=None):
    a, b, c, d = box
    margin = max(4, int(math.ceil(.5*(d-b))))
    x0, y0, x1, y1 = _box([a-margin, b-margin, c+margin, d+margin], image.shape)
    # A tightly annotated 6-pixel-high line box may contain NO actual paper.
    # Estimate paper in a wider source neighborhood, ink only in the key.
    return _estimate_model(image[b:d, a:c], rgb,
                           paper_bgr=_paper(image[y0:y1, x0:x1]))


def _membership(image, model):
    if not model['achromatic']:
        return chromatic_evidence(image, model['paper_bgr'], model['bgr'],
            model.get('colour_reference_contrast', model.get('contrast',
                np.linalg.norm(np.asarray(model['paper_bgr'])-np.asarray(model['bgr'])))))
    pixels = np.asarray(image, np.float32)
    paper = np.asarray(model['paper_bgr'], np.float32)
    direction = paper-np.asarray(model['bgr'], np.float32)
    norm = float(np.linalg.norm(direction))
    delta = paper-pixels
    magnitude = np.linalg.norm(delta, axis=-1)
    alpha = (delta@direction)/max(norm*norm, 1.)
    residual = np.linalg.norm(delta-alpha[..., None]*direction, axis=-1)
    # JPEG direction drift is a relative colour error, with a small absolute
    # noise floor; pale pixels do not become strong evidence by hue alone.
    tolerance = model['residual_sigma']+.065*magnitude
    fit = np.exp(-.5*(residual/np.maximum(tolerance, 1.))**2)
    cosine = np.clip((delta@direction)/np.maximum(magnitude*norm, 1.), -1, 1)
    direction_fit = np.exp(-.5*((1.-cosine)/.025)**2)
    fit *= direction_fit
    # A gray key cannot claim arbitrarily dark black ink. Slight JPEG
    # darkening is allowed, gross overshoot is an independent colour error.
    fit *= np.exp(-.5*(np.maximum(alpha-1.35, 0)/.30)**2)
    if model['achromatic']:
        fit *= np.exp(-.5*(np.ptp(pixels, axis=-1)/18.)**2)
    strength = np.clip(alpha, 0., 1.)
    fit *= np.clip((magnitude-3.)/10., 0., 1.)
    return (strength*fit).astype(np.float32), fit.astype(np.float32)


def _tight(mask):
    yy, xx = np.nonzero(mask)
    if len(xx) < 3:
        raise ValueError("Fewer than three source ink pixels in the swatch")
    return int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)


def _marker_body(support):
    """Cut connector tails at a measured bulge; retain all central source ink."""
    x0, y0, x1, y1 = _tight(support)
    width, height = x1-x0, y1-y0
    rows = np.zeros(support.shape[0], bool)
    if width < 1.45*height:
        return (x0, y0, x1, y1), rows, None
    spans = np.zeros(support.shape[1], np.float32)
    for x in range(x0, x1):
        yy = np.flatnonzero(support[:, x])
        spans[x] = np.ptp(yy)+1 if len(yy) else 0
    tail_width = max(1, int(round(.20*width)))
    tails = np.r_[np.arange(x0, x0+tail_width), np.arange(x1-tail_width, x1)]
    positive = spans[tails][spans[tails] > 0]
    thickness = float(np.percentile(positive, 60)) if len(positive) else 1.
    tall = np.flatnonzero(spans > max(thickness+1., .45*height))
    if height <= max(3., 1.5*thickness) or not len(tall):
        raise ValueError("Legend key is line-only; no two-dimensional marker bulge")
    runs = np.split(tall, np.flatnonzero(np.diff(tall)>1)+1)
    center = .5*(x0+x1-1)
    run = max(runs, key=lambda r: float(np.sum(spans[r]-thickness)) /
              (1+.02*abs(float(np.mean(r))-center)))
    a, b = int(run[0]), int(run[-1])+1
    while a > x0 and spans[a-1] > thickness+.75:
        a -= 1
    while b < x1 and spans[b] > thickness+.75:
        b += 1
    # The horizontal tip of a diamond can be indistinguishable from its line.
    # Include one uncertain shoulder pixel, without fabricating filled ink.
    a, b = max(x0, a-1), min(x1, b+1)
    if b-a > 1.9*height:
        raise ValueError("Legend component has no compact marker body")
    rows = support[:, tails].mean(axis=1) >= .25
    return (a, y0, b, y1), rows, 'horizontal'


def build_series_template(image_bgr, spec: dict[str, Any], *, _body_source_box=None,
                          _source_model=None):
    """Extract a complete observed graphical body, never an idealized symbol.

    ``spec`` requires id and swatch_box, with optional label, color_rgb and
    legend_box. Shape names in annotations are deliberately ignored.
    """
    image = np.asarray(image_bgr, np.uint8)
    box = _box(spec.get('swatch_box', spec.get('swatch')), image.shape)
    sx0, sy0, sx1, sy1 = box
    seed = image[sy0:sy1, sx0:sx1]
    model = _source_model if _source_model is not None else _swatch_model(image, box, spec.get('color_rgb'))
    margin = max(2, min(6, int(round(.12*(sy1-sy0)))))
    x0, y0, x1, y1 = _box([sx0-margin, sy0-margin, sx1+margin, sy1+margin], image.shape)
    raw = image[y0:y1, x0:x1]
    soft, fit = _membership(raw, model)
    support = soft >= .28
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    seed_domain = np.zeros(support.shape, bool)
    seed_domain[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = True
    components = []
    for k in range(1, n):
        observed = labels == k
        overlap = int((observed & seed_domain).sum())
        if overlap >= 3:
            components.append((overlap, int(stats[k, cv2.CC_STAT_AREA]), k))
    if not components:
        raise ValueError("No graphical ink intersects the supplied swatch box")
    component = labels == max(components)[2]
    # Keep weak observed edges connected to the main graphical component.
    near = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    support = (soft >= .22) & near
    if _body_source_box is None:
        body, line_rows, direction = _marker_body(support)
    else:
        ba, bb, bc, bd = _box(_body_source_box, image.shape)
        body = (ba-x0, bb-y0, bc-x0, bd-y0)
        line_rows = np.zeros(support.shape[0], bool)
        direction = 'plot_observed_body'
    a, b, c, d = body
    if min(c-a, d-b) < 3:
        raise ValueError("Marker is too small for two-dimensional shape evidence")
    diameter = float(max(c-a, d-b))
    center = np.array([(a+c-1)/2, (b+d-1)/2], np.float32)
    radius = int(math.ceil(diameter/2))+2
    side = 2*radius+1
    transform = np.array([[1., 0., radius-center[0]], [0., 1., radius-center[1]]], np.float32)
    warp = lambda value, flag=cv2.INTER_LINEAR: cv2.warpAffine(value, transform, (side, side), flags=flag)
    body_domain = np.zeros(support.shape, np.float32)
    body_domain[b:d, a:c] = 1.
    body_domain *= near
    template_soft = warp(soft*body_domain)
    template_core = template_soft >= .60
    template_support = template_soft >= .18
    envelope = np.zeros((side, side), np.uint8)
    yy, xx = np.nonzero(template_support)
    if len(xx) >= 3:
        cv2.fillConvexPoly(envelope, cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)), 1)
    envelope = cv2.dilate(envelope, np.ones((3, 3), np.uint8)).astype(bool)
    raw_soft = warp(soft)
    nuisance = (raw_soft >= .15) & ~template_support
    central = warp(np.broadcast_to(line_rows[:, None], support.shape).astype(np.float32)) > .5
    distance = cv2.distanceTransform(template_support.astype(np.uint8), cv2.DIST_L2, 5)
    weight = template_soft*np.where(distance >= 1.5, 1., .65)
    # Hollow markers retain their source connector pixels, but those pixels
    # cannot independently establish interior fill. No rim pixel is erased.
    interior = cv2.erode(envelope.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    hollow_fraction = float((interior & ~template_support).sum()/max(1, interior.sum()))
    uncertain = central & interior if hollow_fraction > .20 else np.zeros_like(central)
    weight[uncertain] *= .25
    paper_tuple = tuple(float(v) for v in model['paper_bgr'])
    raw_template = cv2.warpAffine(raw, transform, (side, side), flags=cv2.INTER_LINEAR,
                                  borderValue=paper_tuple)
    return dict(id=str(spec.get('id', spec.get('label', 'series'))),
                label=str(spec.get('label', spec.get('id', 'series'))),
                rgb=[int(round(v)) for v in model['bgr'][::-1]], model=model,
                diameter=diameter, soft=template_soft.astype(np.float32),
                core=template_core, envelope=envelope, weight=weight.astype(np.float32),
                nuisance=nuisance, central_connector=central, uncertain=uncertain,
                raw_bgr=raw_template, raw_soft=raw_soft, center=[radius, radius],
                source_center=[float(center[0]+x0), float(center[1]+y0)],
                swatch_box=box, marker_box=[x0+a, y0+b, x0+c, y0+d],
                legend_box=spec.get('legend_box'), source_crop_box=[x0, y0, x1, y1],
                connector_direction=direction, hollow_fraction=hollow_fraction,
                evidence_version=VERSION,
                provenance=dict(kind='legend_observed', confidence=1., supporting_count=1,
                                source_bbox=[x0+a, y0+b, x0+c, y0+d]))


def _mine_plot_template(image, plot_box, spec, exclusion_specs):
    """Mine an observed medoid only when a line key has repeated plot bodies.

    Opening and enclosed paper regions propose geometry only. The chosen
    prototype always retains the original pixels, not the opened/filled mask.
    Three independently located, compact and shape-consistent bodies must
    agree; a bare line, isolated cap, or one imagined marker is insufficient.
    """
    sx0, sy0, sx1, sy1 = _box(spec.get('swatch_box', spec.get('swatch')), image.shape)
    model = _swatch_model(image, [sx0, sy0, sx1, sy1], spec.get('color_rgb'))
    seed, _ = _membership(image[sy0:sy1, sx0:sx1], model)
    seed_support = seed >= .30
    widths = seed_support.sum(axis=0)
    widths = widths[widths > 0]
    line_width = max(1., float(np.median(widths))) if len(widths) else 1.
    x0, y0, x1, y1 = plot_box
    soft, _ = _membership(image[y0:y1, x0:x1], model)
    rival_strength = np.zeros_like(soft)
    for item in exclusion_specs:
        if item is spec or str(item.get('id')) == str(spec.get('id')):
            continue
        try:
            rival_box = _box(item.get('swatch_box', item.get('swatch')), image.shape)
            rival_model = _swatch_model(image, rival_box, item.get('color_rgb'))
        except ValueError:
            continue
        if np.linalg.norm(model['bgr']-rival_model['bgr']) < 35.:
            continue
        rival_strength = np.maximum(rival_strength, _membership(image[y0:y1, x0:x1], rival_model)[0])
    valid = np.ones(soft.shape, np.uint8)
    for item in exclusion_specs:
        value = item.get('legend_box', item.get('swatch_box', item.get('swatch')))
        if value is None:
            continue
        a, b, c, d = value
        a, c = max(0, int(a)-x0), min(x1-x0, int(c)-x0)
        b, d = max(0, int(b)-y0), min(y1-y0, int(d)-y0)
        if a < c and b < d:
            valid[b:d, a:c] = 0
    binary = ((soft >= .22) & (soft >= rival_strength+.05) & (valid > 0)).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ridges = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    if ridges.any():
        # Legend strokes may be heavier than the data-curve strokes. The
        # lower ridge-width quartile measures ordinary plot strokes, while
        # marker interiors occupy the upper tail and do not set the kernel.
        plot_line_width = max(1., float(2*np.percentile(distance[ridges], 25)-.5))
        line_width = min(line_width, plot_line_width)
    radius = max(1, int(math.ceil(.70*line_width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    proposals = []

    def add_components(mask, kind, expansion=1):
        n, _, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        for k in range(1, n):
            a, b, w, h, area = [int(v) for v in stats[k]]
            diameter = max(w, h)
            if min(w, h) < max(4., 1.7*line_width) or diameter > max(18., 12*line_width):
                continue
            if max(w, h)/max(1, min(w, h)) > 1.55 or area/(w*h) < .35:
                continue
            a, b = max(0, a-expansion), max(0, b-expansion)
            c, d = min(soft.shape[1], a+w+2*expansion), min(soft.shape[0], b+h+2*expansion)
            if not valid[b:d, a:c].all() or min(a, b) < 1 or c >= soft.shape[1]-1 or d >= soft.shape[0]-1:
                continue
            observed = soft[b:d, a:c].copy()
            density = float(observed.mean())
            if density < (.12 if kind == 'closed_hollow_body' else .22):
                continue
            if kind == 'thickness_bulge':
                # A thin curve crossing an error-bar stem survives a small
                # round opening too. It is not a marker-sized ink interior.
                if float(distance[b:d, a:c].max()) < max(2.5, 1.5*line_width):
                    continue
                yy, xx = np.nonzero(observed >= .35)
                if len(xx) < 6:
                    continue
                covariance = np.cov(np.column_stack((xx, yy)).T)
                eigenvalues = np.linalg.eigvalsh(covariance)
                if eigenvalues[0]/max(1., eigenvalues[-1]) < .20:
                    continue
                # Repeated error-cap/stem crossings are recurrent compact
                # plus signs too. Require independent ink away from BOTH
                # central stroke bands. Ambiguous plus-shaped real markers
                # consequently need an actual legend glyph, not this fallback.
                gy, gx = np.mgrid[:d-b, :c-a]
                dx, dy = gx-(c-a-1)/2., gy-(d-b-1)/2.
                half_band = max(1., math.ceil(.12*max(c-a, d-b)), math.ceil(.6*line_width))
                independent = (np.abs(dx) > half_band) & (np.abs(dy) > half_band)
                quadrant_masses = [float(observed[independent & (dx*xs > 0) & (dy*ys > 0)].sum())
                                   for xs, ys in [(1,1), (1,-1), (-1,1), (-1,-1)]]
                if sum(v >= 1.5 for v in quadrant_masses) < 3:
                    continue
            cx, cy = .5*(a+c-1), .5*(b+d-1)
            if any((cx-p['center'][0])**2+(cy-p['center'][1])**2 < (.5*diameter)**2 for p in proposals):
                continue
            descriptor = cv2.resize(observed, (24, 24), interpolation=cv2.INTER_AREA)
            # This normalization is only for recurrence/shape agreement.
            # The returned template retains the unnormalised source strength.
            descriptor = np.clip(descriptor/max(.25, float(np.percentile(descriptor, 90))), 0., 1.)
            proposals.append(dict(box=[a, b, c, d], center=[cx, cy], diameter=max(c-a, d-b),
                                  density=density, kind=kind, descriptor=descriptor))

    add_components(opened, 'thickness_bulge', expansion=max(1, min(2, int(math.ceil(.4*line_width)))))
    # Closed paper islands recover hollow markers that do not survive erosion.
    # Holes split by an actual connector are grouped geometrically; their
    # interiors remain white in the ultimately selected source template.
    # Eight-connected one-pixel rims require four-connected paper topology;
    # otherwise paper leaks diagonally through a perfectly visible circle.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(1-binary, connectivity=4)
    holes = np.zeros_like(binary)
    source_paper = np.linalg.norm(image[y0:y1, x0:x1].astype(np.float32)-
                                   model['paper_bgr'], axis=-1) < 32.
    for k in range(1, n):
        a, b, w, h, area = [int(v) for v in stats[k]]
        if min(a, b) <= 0 or a+w >= soft.shape[1] or b+h >= soft.shape[0]:
            continue
        # Low colour confidence is NOT evidence of a hollow white interior.
        # A hollow candidate must actually contain source-paper pixels.
        observed_paper_fraction = float(source_paper[labels == k].mean())
        if observed_paper_fraction < .70:
            continue
        if 3 <= area <= max(100., 80*line_width*line_width) and max(w, h) <= max(18., 12*line_width):
            holes[labels == k] = 1
    hole_radius = max(1, int(math.ceil(.70*line_width)))
    grouped_holes = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                               (2*hole_radius+1, 2*hole_radius+1)))
    add_components(grouped_holes, 'closed_hollow_body')
    if len(proposals) < 3:
        raise ValueError(f"Line-only key: only {len(proposals)} independent compact plot bodies; need 3")
    # Symmetric weighted-overlap compares the entire observed body, including
    # holes. No nominal circle/square and no expected sampling x-values enter.
    similarities = np.zeros((len(proposals), len(proposals)), np.float32)
    for i, p in enumerate(proposals):
        for j, q in enumerate(proposals):
            ratio = p['diameter']/max(1., q['diameter'])
            if .80 <= ratio <= 1.25:
                a, b = p['descriptor'], q['descriptor']
                similarities[i, j] = 2*float(np.minimum(a, b).sum())/max(1., float(a.sum()+b.sum()))
    support = similarities >= .78
    eligible = []
    for i, p in enumerate(proposals):
        neighbors = np.flatnonzero(support[i])
        independent = []
        for j in neighbors:
            q = proposals[int(j)]
            if all(np.linalg.norm(np.subtract(q['center'], proposals[k]['center'])) >=
                   1.5*max(q['diameter'], proposals[k]['diameter']) for k in independent):
                independent.append(int(j))
        if len(independent) >= 3:
            eligible.append((len(independent), float(similarities[i, independent].mean()),
                             float(p['density']), i, independent))
    if not eligible:
        raise ValueError("Line-only key: compact plot bodies lack three independent shape-consistent examples")
    _, quality, _, index, supporters = max(eligible)
    chosen = proposals[index]
    a, b, c, d = chosen['box']
    source_box = [x0+a, y0+b, x0+c, y0+d]
    mined_spec = dict(spec, swatch_box=source_box)
    result = build_series_template(image, mined_spec, _body_source_box=source_box, _source_model=model)
    result['swatch_box'] = [sx0, sy0, sx1, sy1]
    result['provenance'] = dict(kind='plot_mined', confidence=float(quality),
        supporting_count=len(supporters), source_bbox=source_box,
        palette_source_bbox=[sx0, sy0, sx1, sy1], measured_line_width_px=float(line_width),
        proposal_kind=chosen['kind'], candidate_count=len(proposals),
        supporting_source_boxes=[[x0+proposals[j]['box'][0], y0+proposals[j]['box'][1],
                                  x0+proposals[j]['box'][2], y0+proposals[j]['box'][3]] for j in supporters],
        limitations='Image-only recurrence, not marker ground truth; full-window verifier still required')
    return result


def colour_evidence(crop_bgr, templates):
    """Return same-colour strength, known rivals, paper, and unmodelled ink.

    Different shapes with indistinguishable swatch colours retain their own
    template identities but are not counted as one another's occluders.
    """
    if not templates:
        raise ValueError("At least one valid swatch template is required")
    crop = np.asarray(crop_bgr, np.uint8)
    pairs = [_membership(crop, t['model']) for t in templates]
    membership = np.stack([p[0] for p in pairs])
    confidence = np.stack([p[1] for p in pairs])
    paper_bgr = _paper(crop)
    contrast = np.linalg.norm(paper_bgr-crop.astype(np.float32), axis=-1)
    all_ink = np.clip((contrast-4.)/35., 0., 1.).astype(np.float32)
    paper = np.exp(-.5*(contrast/12.)**2).astype(np.float32)
    directions = np.array([t['model']['paper_bgr']-t['model']['bgr'] for t in templates])
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.)
    cosines = np.clip(directions@directions.T, -1, 1)
    rgb = np.array([t['rgb'] for t in templates], float)
    distances = np.linalg.norm(rgb[:, None]-rgb[None, :], axis=-1)
    equivalent = (cosines >= math.cos(math.radians(9))) & (distances < 85.)
    other = np.zeros_like(membership)
    for i in range(len(templates)):
        rivals = np.flatnonzero(~equivalent[i])
        if len(rivals):
            rival = np.max(membership[rivals]*confidence[rivals], axis=0)
            # Ambiguous blended pixels cannot excuse absent required ink.
            other[i] = rival*np.clip((rival-membership[i]-.08)/.35, 0., 1.)
    unknown = all_ink*(1.-np.max(confidence, axis=0))
    return dict(membership=membership, colour_confidence=confidence,
                other=other, paper=paper, unknown=unknown.astype(np.float32),
                all_ink=all_ink, equivalent_colours=equivalent,
                palette_rgb=rgb.astype(np.uint8), paper_bgr=paper_bgr)


def prepare_evidence(image_bgr, plot_box, series_specs, max_side=1400):
    """Prepare detector arrays and an explicit source-coordinate transform.

    Invalid legend keys are reported in ``template_errors``; they are never
    replaced with synthetic circles or silently assigned a neighbouring key.
    Coordinates map back as ``source = plot_origin + crop_coordinate / scale``.
    The scale factors use pixel dimensions; centre mapping includes OpenCV's
    half-pixel resize offset, exposed in ``source_center_offset``.
    """
    image = np.asarray(image_bgr, np.uint8)
    box = _box(plot_box, image.shape)
    x0, y0, x1, y1 = box
    native = image[y0:y1, x0:x1].copy()
    templates, errors = [], []
    for spec in series_specs:
        try:
            templates.append(build_series_template(image, spec))
        except ValueError as exc:
            reason = str(exc)
            if 'line-only' in reason.lower():
                try:
                    templates.append(_mine_plot_template(image, box, spec, series_specs))
                    continue
                except ValueError as mining_error:
                    reason += '; ' + str(mining_error)
            errors.append(dict(id=str(spec.get('id', 'series')), reason=reason))
    if not templates:
        raise ValueError(f"No usable legend templates: {errors}")
    factor = min(1., float(max_side)/max(native.shape[:2])) if max_side else 1.
    w = max(1, int(round(native.shape[1]*factor)))
    h = max(1, int(round(native.shape[0]*factor)))
    scale_x, scale_y = w/native.shape[1], h/native.shape[0]
    crop = cv2.resize(native, (w, h), interpolation=cv2.INTER_AREA) if factor < 1 else native
    if factor < 1:
        for t in templates:
            original_shape = t['soft'].shape
            size = (max(3, int(round(original_shape[1]*scale_x))) | 1,
                    max(3, int(round(original_shape[0]*scale_y))) | 1)
            # Apply the exact plot scale on a centred odd canvas. Simply
            # resizing to the rounded canvas dimensions would silently apply
            # a different marker scale, particularly for small glyphs.
            source_cx, source_cy = (original_shape[1]-1)/2., (original_shape[0]-1)/2.
            matrix = np.array([[scale_x, 0., (size[0]-1)/2.-scale_x*source_cx],
                               [0., scale_y, (size[1]-1)/2.-scale_y*source_cy]], np.float32)
            resample = lambda value, flag=cv2.INTER_LINEAR: cv2.warpAffine(
                value, matrix, size, flags=flag)
            # Re-sampling retains source evidence; no scale search is performed.
            for key in ('soft', 'weight', 'raw_soft'):
                t[key] = resample(t[key])
            for key in ('core', 'envelope', 'nuisance', 'central_connector', 'uncertain'):
                t[key] = resample(t[key].astype(np.uint8), cv2.INTER_NEAREST)>0
            t['raw_bgr'] = cv2.warpAffine(t['raw_bgr'], matrix, size,
                flags=cv2.INTER_LINEAR, borderValue=tuple(float(v) for v in t['model']['paper_bgr']))
            t['source_diameter'] = t['diameter']
            t['diameter'] *= .5*(scale_x+scale_y)
            t['center'] = [(size[0]-1)/2., (size[1]-1)/2.]
    result = colour_evidence(crop, templates)
    valid = np.ones((h, w), bool)
    excluded = []
    for spec in series_specs:
        value = spec.get('legend_box')
        if value is None:
            value = spec.get('swatch_box', spec.get('swatch'))
        if value is None:
            continue
        a, b, c, d = [float(v) for v in value]
        left, top = max(0, int(np.floor((a-x0)*scale_x))), max(0, int(np.floor((b-y0)*scale_y)))
        right, bottom = min(w, int(np.ceil((c-x0)*scale_x))), min(h, int(np.ceil((d-y0)*scale_y)))
        if right > left and bottom > top:
            valid[top:bottom, left:right] = False
            excluded.append([left, top, right, bottom])
    for key in ('membership', 'colour_confidence', 'other'):
        result[key][:, ~valid] = 0
    result['all_ink'][~valid] = 0
    result['unknown'][~valid] = 0
    result.update(crop_bgr=crop, templates=templates, template_errors=errors,
                  valid=valid, excluded_boxes=excluded, plot_box_source=box,
                  scale=float(factor), scale_x=float(scale_x), scale_y=float(scale_y),
                  source_center_offset=[x0+.5/scale_x-.5, y0+.5/scale_y-.5],
                  native_shape=list(native.shape[:2]), version=VERSION)
    return result

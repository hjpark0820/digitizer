"""Repeated occluded-triangle body calibration, before marker grid proposals.

Compare one whole glyph with two apparent fragments on the SAME observed crop.
Foreign ink only excuses missing ink; it never supplies positive evidence.
No series labels, expected counts, saved detections or reference points enter.
"""
import cv2
import numpy as np

VERSION = 'occluded_triangle_shared_body_v1'
CONFIG = dict(minimum_sites=3, minimum_scale_ratio=1.45,
              minimum_gain=.025, complexity_cost=.10, maximum_sites=12,
              minimum_foreign_fraction=.12, maximum_white_fraction=.10,
              minimum_direct_image_wins=2)


def _stamp(template, x, y, shape):
    cx, cy = template['center']
    return cv2.warpAffine(np.asarray(template['soft'], np.float32),
        np.float32([[1, 0, x-cx], [0, 1, y-cy]]), (shape[1], shape[0]))


def _loss(model, own, residual, other, line, paper, valid, count):
    mass = max(float(np.square(residual).sum()), 5.)
    unexplained = float((np.maximum(residual-model, 0)**2*valid).sum())/mass
    occlusion = np.clip(other/.30, 0, 1)
    missing = float((np.maximum(model-own, 0)**2*(1-occlusion)**2*(1-line)**2*valid).sum())/mass
    white = float((model**2*paper**2*valid).sum())/max(float((model**2*valid).sum()), 1.)
    paper_cost = float((model**2*paper**2*valid).sum())/mass
    return dict(loss=unexplained+.25*missing+2*paper_cost+CONFIG['complexity_cost']*count,
                unexplained=unexplained, missing=missing, white_fraction=white,
                foreign_fraction=float((model*other).sum())/max(float(model.sum()), 1.),
                observed_mass=float((model*residual).sum()))


def review(evidence, index, template, decision):
    """Propose a single shared scale and image-fitted centers, or leave unchanged."""
    from color_shared_scale_v46 import scaled_template, _match_map, SCALES
    result = dict(version=VERSION, applied=False, reason='not_applicable', sites=[])
    hint = template.get('legend_shape_hint', '')
    if hint not in ('filled_inv_triangle', 'filled_triangle'):
        return result
    old_scale = decision['scale']
    scales = [s for s in SCALES if s >= CONFIG['minimum_scale_ratio']*old_scale]
    if not scales:
        return result
    d = float(template['diameter'])
    own = np.asarray(evidence['membership'][index], np.float32)
    other = np.asarray(evidence['other'][index], np.float32)
    valid = np.asarray(evidence['valid'], bool) & (evidence.get('ignore_mask', 0) < .1)
    # Local straight connector evidence is a nuisance, not erased source ink.
    line = cv2.morphologyEx(own, cv2.MORPH_OPEN, np.ones((1, max(9, round(1.5*d))), np.uint8))
    residual = np.maximum(own-line, 0)*valid
    paper = np.clip((np.min(evidence['crop_bgr'], axis=2).astype(np.float32)-220)/30, 0, 1)
    small = scaled_template(template, old_scale)
    if np.count_nonzero(small['soft'] > .5) < 3:
        result['reason'] = 'insufficient_template_body'
        return result
    plane = _match_map(cv2.GaussianBlur(own, (3, 3), .45), small, valid)
    maxima = cv2.dilate(plane, np.ones((5, 5), np.uint8))
    yy, xx = np.nonzero((plane > .55) & (plane >= maxima-1e-6))
    peaks = []
    for x, y in sorted(zip(xx, yy), key=lambda p: -plane[p[1], p[0]]):
        if any(np.hypot(x-a, y-b) < .30*d for a, b in peaks):
            continue
        peaks.append((int(x), int(y)))
        if len(peaks) >= 100:
            break
    pairs = []
    for i, (x, y) in enumerate(peaks):
        for a, b in peaks[i+1:]:
            if .30*d <= abs(a-x) <= .80*d and abs(b-y) <= max(2, .18*d):
                pairs.append((float(plane[y, x]+plane[b, a]), (x, y), (a, b)))
    sites = []
    direction = 1 if hint == 'filled_inv_triangle' else -1
    variants = [scaled_template(template, s) for s in scales]
    if any(np.count_nonzero(v['soft'] > .5) < 3 for v in variants):
        result['reason'] = 'insufficient_template_body'
        return result
    support_y = np.nonzero(small['soft'] > .5)[0]
    edge = (support_y.min() if direction == 1 else support_y.max())-small['center'][1]
    for quality, p, q in sorted(pairs, reverse=True):
        mx, my = np.mean([p, q], axis=0)
        if any(np.hypot(mx-s['seed'][0], my-s['seed'][1]) < 1.35*d for s in sites):
            continue
        r = int(np.ceil(1.3*d))
        x0, y0 = int(round(mx))-r, int(round(my))-r
        x1, y1 = x0+2*r+1, y0+2*r+1
        if x0 < 0 or y0 < 0 or x1 > own.shape[1] or y1 > own.shape[0]:
            continue
        fields = [a[y0:y1, x0:x1].copy() for a in (own, residual, other, line, paper, valid)]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        # Compare the same complete-body neighborhood, not unrelated distant
        # connectors or neighboring observations inside a larger context crop.
        area = (abs(xx-mx) <= .75*d) & (abs(yy-my-direction*.18*d) <= .70*d)
        fields[1] *= area
        fields[-1] &= area
        shape = fields[0].shape
        two = np.maximum(_stamp(small, p[0]-x0, p[1]-y0, shape),
                         _stamp(small, q[0]-x0, q[1]-y0, shape))
        baseline = _loss(two, *fields, 2)
        fits = []
        for scale, variant in zip(scales, variants):
            ys = np.nonzero(variant['soft'] > .5)[0]
            new_edge = (ys.min() if direction == 1 else ys.max())-variant['center'][1]
            cy = my+edge-new_edge
            options = []
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    cx, fy = round(mx)+dx, round(cy)+dy
                    model = _stamp(variant, cx-x0, fy-y0, shape)
                    metrics = _loss(model, *fields, 1)
                    options.append(dict(x=float(cx), y=float(fy), scale=scale, **metrics))
            fits.append(min(options, key=lambda f: f['loss']))
        best = min(fits, key=lambda f: f['loss'])
        sites.append(dict(seed=[float(mx), float(my)], fragments=[list(p), list(q)],
                          two=baseline, fits=fits, best=best))
        if len(sites) >= CONFIG['maximum_sites']:
            break
    result.update(reason='insufficient_repeated_whole_body_evidence', sites=sites,
                  original_scale=old_scale, scales=scales)
    if len(sites) < CONFIG['minimum_sites']:
        return result
    losses = np.array([[f['loss'] for f in s['fits']] for s in sites])
    j = int(np.argmin(np.mean(losses, axis=0)))
    scale = scales[j]
    accepted = []; direct_wins = 0
    for site in sites:
        fit = site['fits'][j]
        fit['gain_over_two'] = site['two']['loss']-fit['loss']
        fit['image_gain_over_two'] = fit['gain_over_two']-CONFIG['complexity_cost']
        # A simplicity preference must not turn visually indistinguishable
        # one/two-body explanations into an asserted global size change.
        direct_wins += (fit['image_gain_over_two'] >= CONFIG['minimum_gain']
            and fit['white_fraction'] <= CONFIG['maximum_white_fraction']
            and fit['foreign_fraction'] >= CONFIG['minimum_foreign_fraction']
            and fit['observed_mass'] >= max(8, .025*d*d))
        fit['accepted'] = (fit['gain_over_two'] >= CONFIG['minimum_gain']
            and fit['foreign_fraction'] >= CONFIG['minimum_foreign_fraction']
            and fit['white_fraction'] <= CONFIG['maximum_white_fraction']
            and fit['unexplained'] < .25 and fit['observed_mass'] >= max(8, .025*d*d))
        site['shared_fit'] = fit
        if fit['accepted']:
            accepted.append(fit)
    result.update(proposed_scale=scale, accepted_sites=len(accepted),direct_image_wins=int(direct_wins))
    if (len(accepted) >= CONFIG['minimum_sites'] and len(accepted)/len(sites) >= .7
            and direct_wins >= CONFIG['minimum_direct_image_wins']):
        result.update(applied=True, reason='repeated_one_body_better_than_two_fragments',
                      scale=scale, centers=accepted)
    return result

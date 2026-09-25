"""Bounded plot-ink adaptation followed by exclusive residual assignment.

Only observed chromatic pixels inside the plot may gain colour ownership.
Repeated compact ink cores calibrate a secondary colour prototype; they are
not accepted markers. Original strong ownership is immutable. Residual ink
must independently fit a unique palette group AND attach to existing support.
Neither phase paints shapes, fills white paper, or decides same-colour symbols.
"""
import cv2
import numpy as np

from color_membership_v46 import chromatic_evidence

VERSION = 'repeated_ink_and_exclusive_residual_v1'
POLICIES = ('off', 'calibrate', 'calibrate_complete')


def _metric(image, model, ink=None):
    """Distance to bounded ink/paper mixtures plus a chromatic-direction cap."""
    p = np.asarray(image, np.float32)
    paper = np.asarray(model['paper_bgr'], np.float32)
    ink = np.asarray(model['bgr'] if ink is None else ink, np.float32)
    delta, direction = paper-p, paper-ink
    contrast = np.linalg.norm(delta, axis=-1)
    alpha = np.clip((delta@direction)/max(float(direction@direction), 1.), 0., 1.08)
    residual = np.linalg.norm(delta-alpha[..., None]*direction, axis=-1)
    hue, key = p-p.min(axis=-1, keepdims=True), ink-ink.min()
    cosine = (hue@key)/np.maximum(np.linalg.norm(hue, axis=-1)*np.linalg.norm(key), 1.)
    angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    # Absolute noise floor plus a bounded relative colour drift allowance.
    cost = residual/(12.+.10*contrast)
    return np.where(angle <= 35., cost, np.inf).astype(np.float32)


def _competition(image, models, variants):
    shape = image.shape[:2]
    best = np.full(shape, np.inf, np.float32)
    second = best.copy()
    winner = np.full(shape, -1, np.int16)
    for i, model in enumerate(models):
        if model['achromatic']:
            continue
        score = _metric(image, model)
        if i in variants:
            score = np.minimum(score, _metric(image, model, variants[i]))
        improve = score < best
        second = np.where(improve, best, np.minimum(second, score))
        winner = np.where(improve, i, winner)
        best = np.minimum(best, score)
    # Avoid inf-inf warnings, and abstain on exact or near ties, independent
    # of palette order. A lone plausible colour still needs absolute evidence.
    margin = np.zeros(shape, np.float32)
    finite = np.isfinite(best)
    margin[finite] = second[finite]-best[finite]
    return winner, (best <= 1.25) & (margin >= .30), best


def _distance(mask):
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def recover(image, models, strengths, fits, valid, diameters, *, policy='calibrate_complete'):
    """Return memberships, fits, JSON diagnostics, and phase provenance masks.

    Models must represent DISTINCT colour groups. Same-colour shapes belong
    to one group upstream; this module deliberately abstains on colour ties.
    All coordinates in diagnostic anchors are native source coordinates.
    """
    if policy not in POLICIES:
        raise ValueError('Unknown palette recovery policy')
    pixels = np.asarray(image)
    m, f = np.asarray(strengths, np.float32), np.asarray(fits, np.float32)
    valid = np.asarray(valid, bool)
    expected = (len(models), *pixels.shape[:2])
    if (m.shape != expected or f.shape != expected or valid.shape != pixels.shape[:2]
            or len(diameters) != len(models) or not np.isfinite(m).all()
            or not np.isfinite(f).all() or np.any((m < 0) | (m > 1))
            or np.any((f < 0) | (f > 1))):
        raise ValueError('Palette recovery inputs must be finite, aligned probabilities')
    report = dict(version=VERSION, policy=policy, groups=[],
                  preserved_strong_ownership=True, exclusive_group_assignment=True,
                  white_and_neutral_excluded=True, generated_pixels=False,
                  marker_acceptance='unchanged downstream geometry verification')
    provenance = np.zeros(pixels.shape[:2], np.uint8)
    if policy == 'off' or not valid.any() or not len(models):
        report['status'] = 'disabled' if policy == 'off' else 'empty'
        return m.copy(), f.copy(), report, provenance
    for model, diameter in zip(models, diameters):
        for key in ('bgr', 'paper_bgr'):
            value = np.asarray(model[key])
            if value.shape != (3,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 255)):
                raise ValueError('Invalid palette colour')
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError('Invalid marker diameter')
    yy, xx = np.where(valid)
    y0,y1,x0,x1 = int(yy.min()),int(yy.max()+1),int(xx.min()),int(xx.max()+1)
    s = np.s_[y0:y1,x0:x1]
    # Runtime guard: keep the already-computed evidence on very large inputs.
    if (y1-y0)*(x1-x0)*len(models) > 16_000_000:
        report['status'] = 'recovery_budget_exceeded'
        return m.copy(), f.copy(), report, provenance
    image = pixels[s].astype(np.float32)
    region = valid[s]
    own_m, own_f = m[:,y0:y1,x0:x1].copy(), f[:,y0:y1,x0:x1].copy()
    chroma = np.ptp(image, axis=-1)
    contrast = np.linalg.norm(255.-image, axis=-1)
    chromatic = region & (chroma >= 18.) & (contrast >= 24.)
    frozen = own_f.max(axis=0) >= .35
    unresolved = chromatic & ~frozen
    initial_unknown = int(unresolved.sum())
    winner, unique, _ = _competition(image, models, {})
    variants = {}
    sources = []
    # Frozen pre-adaptation evidence only: no self-training through iteration.
    for i, model in enumerate(models):
        diameter = float(diameters[i])
        entry = dict(group_index=i, status='no_repeated_compact_cores', anchors=[],
                     calibration_pixels=0, residual_pixels=0)
        sources.append(entry)
        if model['achromatic']:
            entry['status'] = 'achromatic_unchanged'
            continue
        selected = unresolved & unique & (winner == i) & (chroma >= 30.)
        # Thin connector fragments cannot define the variant colour.
        thickness = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, 5)
        cores = thickness >= 1.4
        count, labels, stats, centres = cv2.connectedComponentsWithStats(cores.astype(np.uint8), 8)
        seeds = region & (own_f[i] >= .35) & (own_m[i] >= .10)
        distance = _distance(seeds)
        anchors=[]
        for label in range(1,count):
            left,top,width,height,area = stats[label]
            if area < max(3, .015*diameter**2) or min(width,height) < 2 or max(width,height) > 2*diameter:
                continue
            mask = labels[top:top+height,left:left+width] == label
            if distance[top:top+height,left:left+width][mask].min() > 2*diameter:
                continue
            colour = np.median(image[top:top+height,left:left+width][mask],axis=0)
            anchors.append((centres[label],colour,int(area)))
        if len(anchors) < 3:
            entry['compact_core_count'] = len(anchors)
            continue
        # Find a colour-consistent cluster; component area cannot dominate it.
        colours = np.stack([a[1] for a in anchors])
        distances = np.linalg.norm(colours[:,None]-colours[None],axis=-1)
        cluster = distances[np.argmin(np.median(distances,axis=1))] <= 32.
        chosen = [a for a,keep in zip(anchors,cluster) if keep]
        separated=[]
        for a in sorted(chosen,key=lambda a:(a[0][0],a[0][1])):
            if all(np.linalg.norm(a[0]-b[0]) >= max(4.,diameter) for b in separated):
                separated.append(a)
        if len(separated) < 3:
            entry['status'] = 'insufficient_independent_colour_consensus'
            continue
        variant = np.median([a[1] for a in separated],axis=0).astype(np.float32)
        variants[i] = variant
        entry.update(status='calibrated_secondary_prototype', variant_bgr=variant.tolist(),
            legend_bgr=np.asarray(model['bgr']).tolist(),
            anchors=[dict(x=float(a[0][0]+x0),y=float(a[0][1]+y0),bgr=a[1].tolist(),pixels=a[2]) for a in separated])
    # Re-evaluate all colours together; a new prototype never wins by order.
    winner, unique, _ = _competition(image, models, variants)
    for i, ink in variants.items():
        model = models[i]
        reference = max(float(model.get('colour_reference_contrast',model.get('contrast',100.))),8.)
        strength, fit = chromatic_evidence(image,model['paper_bgr'],ink,reference)
        seeds = region & (own_f[i] >= .35) & (own_m[i] >= .10)
        near = _distance(seeds) <= max(3.,float(diameters[i]))
        possible = unresolved & unique & (winner == i) & (fit >= .42) & near
        count,labels,stats,_ = cv2.connectedComponentsWithStats(possible.astype(np.uint8),8)
        good = np.flatnonzero(stats[:,cv2.CC_STAT_AREA] >= 3)
        good = good[good != 0]
        take = np.isin(labels,good) if len(good) else np.zeros(region.shape,bool)
        own_f[i,take] = np.maximum(own_f[i,take],.85*fit[take])
        own_m[i,take] = np.maximum(own_m[i,take],.85*strength[take])
        provenance[s][take] = 1
        sources[i]['calibration_pixels'] = int(take.sum())
    if policy == 'calibrate_complete':
        # One non-iterative pass. Accepted pixels cannot recursively grow a
        # coloured background or a remote noise cluster into marker support.
        residual = chromatic & (own_f.max(axis=0) < .35)
        seed_maps = [region & (own_f[i] >= .35) & (own_m[i] >= .10)
                     for i in range(len(models))]
        local_support = np.stack([cv2.boxFilter(seed.astype(np.float32),-1,(7,7),normalize=False)
                                  for seed in seed_maps])
        additions=[]
        for i, model in enumerate(models):
            take=np.zeros(region.shape,bool)
            if not model['achromatic']:
                seeds = seed_maps[i]
                near = _distance(seeds) <= max(2.,min(4.,.30*diameters[i]))
                support = local_support[i]
                rival = np.max(np.delete(local_support,i,axis=0),axis=0) if len(models)>1 else np.zeros(region.shape)
                # A pale segment can fit a dark ink/paper mixture better than
                # its actual key. Do not move that segment to a minority local
                # colour: both RGB identity AND curve-neighbour support must
                # agree. Conflict remains unknown, never resolves by order.
                local_agreement = (support >= 3) & (support >= 2*rival) & (support >= rival+3)
                possible = residual & unique & (winner == i) & near & local_agreement
                count,labels,stats,_ = cv2.connectedComponentsWithStats(possible.astype(np.uint8),8)
                good = np.flatnonzero(stats[:,cv2.CC_STAT_AREA] >= 2)
                good = good[good != 0]
                take = np.isin(labels,good) if len(good) else take
            additions.append(take)
        for i,take in enumerate(additions):
            if not take.any():continue
            model=models[i]
            reference=max(float(model.get('colour_reference_contrast',model.get('contrast',100.))),8.)
            observed=np.clip(np.linalg.norm(np.maximum(np.asarray(model['paper_bgr'])-image,0),axis=-1)/reference,0,1)
            own_f[i,take]=.60
            own_m[i,take]=.60*observed[take]
            provenance[s][take]=2
            sources[i]['residual_pixels']=int(take.sum())
    result_m, result_f = m.copy(), f.copy()
    result_m[:,y0:y1,x0:x1]=own_m
    result_f[:,y0:y1,x0:x1]=own_f
    report.update(status='completed',groups=sources,
        chromatic_unassigned_before=initial_unknown,
        chromatic_unassigned_after=int((chromatic & (own_f.max(axis=0)<.35)).sum()),
        calibration_pixels=int((provenance==1).sum()),residual_pixels=int((provenance==2).sum()),
        abstention='Rival ties, isolated pixels, neutral ink and paper remain unassigned')
    return result_m,result_f,report,provenance

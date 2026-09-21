"""Conservative repeated-dot guide hypotheses, without deleting source pixels.

A dense, long, repeated horizontal row is ambiguous with a flat data series.
Only activate when at least three larger round body cores provide independent
alternative marker evidence. Observed ink direction protects differently
coloured marker pixels at crossings; colour never identifies the data series.
"""
import math

import cv2
import numpy as np


def core_shape(distance, record):
    """Roundness of the thick interior, not of attached lines/error bars."""
    x, y, r = record['x'], record['y'], record['radius']
    extent = math.ceil(2*r)
    a, b = max(0, x-extent), max(0, y-extent)
    core = distance[b:y+extent+1, a:x+extent+1] >= .60*r
    _, labels = cv2.connectedComponents(core.astype(np.uint8), 8)
    yy, xx = np.nonzero(labels == labels[y-b, x-a])
    if len(xx) < 3:
        return dict(core_aspect=None, round_body=False)
    eigen = np.linalg.eigvalsh(np.cov(np.array([xx, yy])))
    aspect = float(np.sqrt((eigen[-1]+.25)/(eigen[0]+.25)))
    return dict(core_aspect=aspect, round_body=aspect <= 1.60)


def prepare(crop, own, valid, raw_size, paper):
    """Return source-pixel ignore mask and filtered radius/proposal evidence."""
    binary = ((own > .5) & valid).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    records = [dict(q, **core_shape(distance, q)) for q in raw_size['candidates']]
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    component_radii = np.zeros(count, np.float32)
    np.maximum.at(component_radii, labels.ravel(), distance.ravel())
    height, width = binary.shape
    components = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if (area >= 8 and min(w,h) >= 3 and max(w,h) <= max(12, .08*min(height,width))
                and .55 <= w/h <= 1.8):
            components.append(dict(label=i, x=float(centers[i,0]), y=float(centers[i,1]),
                                   width=w, height=h, area=area))
    ignore = np.zeros(binary.shape, bool)
    proposals = []
    for anchor in components:
        near = [q for q in components if abs(q['y']-anchor['y']) <= max(1.5,.22*anchor['height'])
                and .60 <= q['width']/anchor['width'] <= 1.65
                and .60 <= q['height']/anchor['height'] <= 1.65]
        if len(near) < 10:
            continue
        near.sort(key=lambda q:q['x'])
        y = float(np.median([q['y'] for q in near]))
        h = float(np.median([q['height'] for q in near]))
        w = float(np.median([q['width'] for q in near]))
        xs = np.array([q['x'] for q in near])
        gaps = np.diff(xs)
        period = float(np.median(gaps[gaps <= np.percentile(gaps,60)]))
        multiples = np.maximum(1,np.rint(gaps/max(period,1)))
        regular = float(np.mean((np.abs(gaps/period-multiples) <= .22)&(multiples<=4)))
        if xs[-1]-xs[0] < .65*width or not 1.15 <= period/w <= 3.5 or regular < .80:
            continue
        if any(abs(y-q['y']) <= .5*h for q in proposals):
            continue
        component_ids = [q['label'] for q in near]
        row_pixels = np.isin(labels, component_ids)
        # Small guide dots need not have passed the detector's marker-size
        # seed threshold. Measure their own components independently.
        sample_radii = component_radii[component_ids]
        guide_radius = float(np.median(sample_radii))
        alternatives = [q for q in records if q['round_body'] and
                        q['radius'] >= 1.60*guide_radius and not row_pixels[q['y'],q['x']]]
        if len(alternatives) < 3:
            continue  # Do not erase a flat row just because its spacing is regular.
        # Estimate appearance from observed interior pixels, not a hardcoded hue.
        core = row_pixels & (distance >= .50*guide_radius)
        delta = np.asarray(paper,np.float32)-crop.astype(np.float32)
        unit = delta/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-6)
        direction = np.median(unit[core],axis=0)
        direction /= max(float(np.linalg.norm(direction)),1e-6)
        matches = np.sum(unit*direction,axis=-1) >= .975
        band = np.zeros_like(ignore)
        lo,hi = max(0,math.floor(y-.65*h)),min(height,math.ceil(y+.65*h)+1)
        band[lo:hi,:] = True
        # Only observed matching ink is ignored; never a whole horizontal row.
        ignored = band & matches & binary.astype(bool)
        # Large thick cores are protected even if guide and markers share ink.
        yy,xx = np.mgrid[:height,:width]
        for q in alternatives:
            ignored[(xx-q['x'])**2+(yy-q['y'])**2 <= (1.25*q['radius'])**2] = False
        ignore |= ignored
        proposals.append(dict(y=y, height=h, component_count=len(near),
            x_span=[float(xs[0]),float(xs[-1])],period=period,regular_gap_fraction=regular,
            guide_radius=guide_radius,alternative_round_bodies=len(alternatives),
            ink_direction=direction.tolist(),ignored_pixels=int(ignored.sum())))
    audit = dict(method='dense_repeated_guides_with_larger_round_body_support_v1',
        groups=proposals,ignored_pixels=int(ignore.sum()),source_pixels_deleted=False,
        radius_measured_on_original_ink=True,colour_used_only_for_guide_pixel_protection=True,
        limitations=['Dense flat series without distinct body evidence remain ambiguous.',
                    'This profile handles horizontal repeated guides, not arbitrary annotations.'])
    if not proposals:
        return ignore, None, audit
    candidates, excluded = [], []
    for q in records:
        r=q['radius'];e=math.ceil(r);x,y=q['x'],q['y']
        a,b=max(0,x-e),max(0,y-e);c,d=min(width,x+e+1),min(height,y+e+1)
        yy,xx=np.mgrid[b:d,a:c];disk=(xx-x)**2+(yy-y)**2<=r*r
        fraction=float(ignore[b:d,a:c][disk].mean())
        q['ignored_disk_fraction']=fraction
        reason=('guide_fragment' if fraction>.35 else
                'elongated_thick_core' if not q['round_body'] else None)
        if reason:
            excluded.append(dict(q,exclusion_reason=reason))
        else:
            candidates.append(q)
    audit.update(excluded_size_votes=excluded,remaining_size_votes=candidates)
    if len(candidates)<3:
        return ignore, dict(status='unresolved',reason='fewer_than_three_non_guide_round_bodies'),audit
    best=max(([q for q in candidates if .8<=q['radius']/p['radius']<=1.25] for p in candidates),
             key=lambda group:(len(group),np.median([q['radius'] for q in group])))
    if len(best)<3:
        return ignore,dict(status='unresolved',reason='no_recurrent_non_guide_body_size'),audit
    radius=float(np.median([q['radius'] for q in best]))
    size=dict(raw_size,method='guide_excluded_round_core_recurrent_radius',radius=radius,
        supporting_count=len(best),supporters=best,candidates=candidates,
        original_radius=raw_size['radius'],status='recurrent_radius_supported')
    return ignore,size,audit

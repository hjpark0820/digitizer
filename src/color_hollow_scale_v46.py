"""Conservative outline evidence for calibrating hollow marker size.

Outer radius measures size; radial ink-band width measures stroke thickness.
Neither black/foreign ink nor missing sectors count as target-colour support.
This only admits size anchors; the detector's final window cutoff is unchanged.
"""
import cv2
import numpy as np


CONFIG = dict(rays=64, radial_step=.5, ink_threshold=.25,
              minimum_ray_fraction=.75, minimum_quadrant_fraction=.50,
              minimum_width_ratio=.45, maximum_width_ratio=1.65,
              maximum_hole_ink=.15, minimum_correlation=.80,
              minimum_window_score=.40, maximum_missing=.45, maximum_extra=.25,
              sparse_minimum_anchors=2, sparse_scale_agreement=.05,
              sparse_minimum_gain=.035)


def is_hollow(template):
    hole = np.asarray(template.get('hole_core', []), bool)
    face = np.asarray(template.get('face', []), bool)
    return bool(hole.size and face.shape == hole.shape
                and hole.sum() >= max(4, .04*face.sum()))


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def outline_evidence(evidence, index, template, x, y):
    """Compare independent ring sectors without requiring equal ink thickness.

    A bounded width range avoids mistaking filled glyphs, straight lines, T
    caps or a small surviving arc for a complete calibration marker. Rays with
    legend connectors through the hole cannot establish the hollow outline.
    """
    if not is_hollow(template):
        return dict(eligible=False, accepted=False, reason='not_hollow')
    diameter=float(template['diameter'])
    radius=np.arange(0, .8*diameter+3, CONFIG['radial_step'], dtype=np.float32)
    angle=(np.arange(CONFIG['rays'], dtype=np.float32)+.5)*(2*np.pi/CONFIG['rays'])
    dx=np.cos(angle)[:, None]*radius
    dy=np.sin(angle)[:, None]*radius

    def sample(field, cx, cy):
        return cv2.remap(np.asarray(field, np.float32), dx+float(cx), dy+float(cy),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    expected=sample(template['soft'], *template['center'])
    own=sample(evidence['membership'][index], x, y)
    if 'guide_upper' in evidence:
        own=np.maximum(own-sample(evidence['guide_upper'][index], x, y), 0)
    available=sample(evidence['valid'], x, y)
    if 'ignore_mask' in evidence:
        available*=1-sample(evidence['ignore_mask'], x, y)
    own*=available
    tolerance=min(3., max(1.25, .05*diameter))
    records=[]
    for k in range(CONFIG['rays']):
        bands=_runs(expected[k] >= CONFIG['ink_threshold'])
        # Do not use rays through a connector or an unresolved multiple band.
        if len(bands)!=1:
            continue
        a,b=bands[0]
        if radius[a] < max(1.5, .12*diameter) or b-a<2:
            continue
        inner,outer=radius[a],radius[b-1]
        target_mid=(inner+outer)/2
        observed_bands=_runs(own[k] >= CONFIG['ink_threshold'])
        possible=[(p,q) for p,q in observed_bands
                  if radius[p] <= outer+tolerance and radius[q-1] >= inner-tolerance]
        hole=(radius>=.12*diameter)&(radius<inner-1.)
        hole_ink=float(np.mean(own[k,hole])) if hole.any() else 1.
        rec=dict(ray=k, quadrant=k//(CONFIG['rays']//4), supported=False,
                 expected_outer=float(outer), expected_width=float(outer-inner+.5),
                 hole_ink=hole_ink)
        if possible:
            p,q=min(possible,key=lambda band:abs((radius[band[0]]+radius[band[1]-1])/2-target_mid))
            observed_outer=float(radius[q-1]);width=float(radius[q-1]-radius[p]+.5)
            error=abs(observed_outer-float(outer));ratio=width/rec['expected_width']
            rec.update(observed_outer=observed_outer, observed_width=width,
                       outer_error=error, width_ratio=ratio,
                       supported=bool(error<=tolerance and
                           CONFIG['minimum_width_ratio']<=ratio<=CONFIG['maximum_width_ratio']
                           and hole_ink<=CONFIG['maximum_hole_ink']
                           and np.mean(available[k,a:b])>=.9))
        records.append(rec)
    fraction=float(np.mean([r['supported'] for r in records])) if records else 0.
    quadrants=[]
    for q in range(4):
        group=[r['supported'] for r in records if r['quadrant']==q]
        quadrants.append(float(np.mean(group)) if len(group)>=4 else 0.)
    good=[r for r in records if r['supported']]
    accepted=(len(records)>=CONFIG['rays']//2 and fraction>=CONFIG['minimum_ray_fraction']
              and min(quadrants)>=CONFIG['minimum_quadrant_fraction'])
    return dict(eligible=True, accepted=bool(accepted),
                reason='distributed_hollow_outline' if accepted else 'insufficient_hollow_outline',
                ray_fraction=fraction, quadrant_fractions=quadrants, usable_rays=len(records),
                outer_tolerance_px=tolerance,
                median_outer_error=float(np.median([r['outer_error'] for r in good])) if good else None,
                median_width_ratio=float(np.median([r['width_ratio'] for r in good])) if good else None,
                rays=records)


def anchor_passes(window, correlation, outline):
    return bool(outline.get('accepted') and correlation>=CONFIG['minimum_correlation']
                and not window['line_only_reject'] and window['score']>=CONFIG['minimum_window_score']
                and window['missing']<=CONFIG['maximum_missing'] and window['extra']<=CONFIG['maximum_extra']
                and window['observed_fraction']>=.85 and window['ignored_marker_fraction']<=.05)

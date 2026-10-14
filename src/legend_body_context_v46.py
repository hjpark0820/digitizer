"""Source-only fallback for small legend bodies attached to a connector.

The normal legend extractor remains the first choice.  This fallback measures
the connector from *both* flanks, then locates a compact body above and below
that band.  Frame bands and neighbouring legend rows restrict geometry, not
pixel values.  No idealized marker or replacement image is constructed.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def _runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []


def recovery_candidates(image, box, peers=(), *, bounds=None, frame_report=None):
    """Return auditable body-geometry proposals and a failure diagnostic.

    Candidate dictionaries contain ``crop_box``, ``glyph_box`` and global
    ``line_rows``.  A consumer must preserve original grayscale pixels in the
    glyph rectangle, treat the line as nuisance, and retain its usual shape,
    fill and clipping checks.  This function does not classify a marker.
    """
    height, width = image.shape[:2]
    a, b, c, d = map(int, box)
    if not (0 <= a < c <= width and 0 <= b < d <= height):
        raise ValueError(f'Invalid legend swatch: {box}')
    e, f, g, h = (0, 0, width, height) if bounds is None else map(int, bounds)
    if not (0 <= e < g <= width and 0 <= f < h <= height):
        raise ValueError(f'Invalid legend context bounds: {bounds}')
    budget = max(2, min(6, int(math.ceil(.35 * (d-b)))))
    x0, x1 = max(a, e), min(c, g)
    y0, y1 = max(0, b-budget, f), min(height, d+budget, h)
    cx, cy = (a+c-1)/2., (b+d-1)/2.
    previous, following = None, None
    # Unlike connected-component growth, row ownership cannot cross a weak
    # antialias bridge joining two stacked handles.
    for peer in peers:
        pa, pb, pc, pd = map(int, peer)
        if (pa, pb, pc, pd) == (a, b, c, d) or min(c, pc) <= max(a, pa):
            continue
        py = (pb+pd-1)/2.
        if py < cy:
            previous = max(previous, py) if previous is not None else py
            y0 = max(y0, int(math.floor((py+cy)/2.))+1)
        elif py > cy:
            following = min(following, py) if following is not None else py
            y1 = min(y1, int(math.floor((py+cy)/2.))+1)
    # A pixel row containing a low-contrast minimum can be shared by two
    # antialiased keys. The arbitrary half-pixel row ownership cut should not
    # omit that boundary fringe. Require a measured valley before admitting
    # one row across the geometric midpoint; never grow into a dark neighbour.
    row_terminations = {}
    full_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for edge, neighbour in (('top',previous),('bottom',following)):
        if neighbour is None: continue
        mid = (cy+neighbour)/2.
        low = max(f,b-budget,int(math.floor(mid))-1)
        high = min(h-1,d+budget-1,int(math.ceil(mid))+1)
        inner_left = max(x0,int(math.floor(cx-max(2.,.55*(d-b)))))
        inner_right = min(x1,int(math.ceil(cx+max(2.,.55*(d-b))))+1)
        own_low,own_high=max(f,b),min(h,d)
        if low>high or inner_right<=inner_left or own_low>=own_high: continue
        own = full_gray[own_low:own_high,inner_left:inner_right].astype(float)
        own_peak = float(np.max(np.mean(255.-own,axis=1)))
        rows=np.arange(low,high+1)
        mass=(255.-full_gray[low:high+1,inner_left:inner_right].astype(float)).mean(axis=1)
        index=int(np.argmin(mass));row=int(rows[index]);minimum=float(mass[index])
        if own_peak < 24. or minimum > .45*own_peak: continue
        # Explicit local trough: support on the body-facing side must be
        # stronger. This excludes a cut through a uniformly pale stroke.
        own_rows=[row+offset if edge=='top' else row-offset for offset in (1,2,3)]
        own_rows=[q for q in own_rows if f<=q<h and
                  ((q<=cy) if edge=='top' else (q>=cy))]
        if not own_rows:continue
        inward=float(np.max((255.-full_gray[own_rows,inner_left:inner_right].astype(float)).mean(axis=1)))
        if inward < minimum+max(3.,.05*own_peak):continue
        if edge=='top':y0=row
        else:y1=row+1
        row_terminations[edge]=dict(policy='between_key_ink_valley_v1',row=row,
            profile_x=[inner_left,inner_right],mean_darkness=minimum,
            own_peak_darkness=own_peak,inward_darkness=inward,
            relative_depth=minimum/own_peak,neighbour_center_y=neighbour,
            source_pixels_modified=False)
    diag = dict(policy='source_context_bilateral_body_v1', input_box=[a,b,c,d],
                search_box=[x0,y0,x1,y1], source_pixels_modified=False,
                vertical_budget_px=budget, frame_context=frame_report,
                row_terminations=row_terminations)
    if x1-x0 < 7 or y1-y0 < 5 or not (x0 <= cx < x1 and y0 <= cy < y1):
        return [], dict(diag, reason='insufficient_context')
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    paper = max(235., float(np.percentile(gray, 95)))
    values = gray[gray < min(235., paper-12)]
    if len(values) < 12:
        return [], dict(diag, reason='insufficient_body_contrast')
    core = max(32., float(np.percentile(values, 10)))
    if paper-core < 24:
        return [], dict(diag, reason='insufficient_body_contrast')
    ink = np.clip((paper-gray.astype(np.float32))/(paper-core), 0, 1)
    xx = np.arange(x0, x1)
    # Centre comes from the supplied graphical key, never from target points.
    # A small range of flank starts handles keys whose supplied vertical crop
    # is truncated, but every fit must have independent bilateral evidence.
    proposals = []
    rejected = []
    for radius_fraction in (.55, .70, .85):
        radius = max(2., radius_fraction*(d-b))
        left = xx < cx-radius
        right = xx > cx+radius
        if left.sum() < 2 or right.sum() < 2:
            continue
        # Source keys can contain substantial white padding. Requiring two
        # supporting columns per flank ignores that padding without allowing
        # a lone label/frame fragment to define the connector by itself.
        lp = np.sort(ink[:, left], axis=1)[:, -2:].mean(axis=1)
        rp = np.sort(ink[:, right], axis=1)[:, -2:].mean(axis=1)
        profile = np.minimum(lp, rp)
        allowed = (np.arange(y0,y1) >= b-1) & (np.arange(y0,y1) < d+1)
        peaks = np.where(allowed, profile, -1.)
        anchor = int(np.argmax(peaks))
        peak = float(profile[anchor])
        if peak < .30:
            rejected.append('no_bilateral_connector')
            continue
        # Keep the measured antialiased line band, not all vaguely dark rows.
        threshold = max(.12, .18*peak)
        lo, hi = anchor, anchor+1
        while lo > 0 and profile[lo-1] >= threshold: lo -= 1
        while hi < len(profile) and profile[hi] >= threshold: hi += 1
        if hi-lo > max(3, .45*(y1-y0)):
            rejected.append('connector_is_not_thin')
            continue
        strong = ink >= .30
        off = strong.copy()
        off[lo:hi] = False
        # Disregard a tail/letter only for geometry. Original grayscale stays
        # untouched in any final glyph rectangle chosen below.
        upper = np.sum(off[:lo], axis=0)
        lower = np.sum(off[hi:], axis=0)
        body_columns = (upper+lower) >= 2
        runs = _runs(np.flatnonzero(body_columns))
        choices = []
        for run in runs:
            if len(run) < 3:
                continue
            l, r = int(run[0]), int(run[-1])+1
            off_body = off[:,l:r]
            yy = np.flatnonzero(off_body.any(axis=1))
            if not len(yy): continue
            top, bottom = int(yy.min()), int(yy.max())+1
            bh, bw = bottom-top, r-l
            if bh < 4 or not (.35*bh <= bw <= 1.65*bh): continue
            if hi-lo > max(3, .40*bh): continue
            if int(upper[l:r].sum()) < 3 or int(lower[l:r].sum()) < 3: continue
            centre_x = x0+(l+r-1)/2.
            centre_y = y0+(top+bottom-1)/2.
            if not (a <= centre_x < c and b <= centre_y < d): continue
            if abs(centre_x-cx) > .22*(c-a): continue
            # Thin lines may extend to a border; the measured body may not
            # merely be an entire frame-bounded handle union.
            if l < 1 or r > x1-x0-1: continue
            # Require at least one independently measured thin flank column
            # on both sides of this actual body (not just the initial guess).
            actual_left = ink[lo:hi,:l]
            actual_right = ink[lo:hi,r:]
            if actual_left.size == 0 or actual_right.size == 0: continue
            if min(float(np.max(actual_left.mean(axis=0))),
                   float(np.max(actual_right.mean(axis=0)))) < .20: continue
            score = 20*peak + .10*float(off_body.sum()) - .5*abs(centre_x-cx)
            choices.append((score, l, r, top, bottom))
        if not choices:
            rejected.append('no_independent_compact_body')
            continue
        score,l,r,top,bottom = max(choices)
        # Attach only immediately adjoining pale source pixels. Extension is
        # limited to the row-owned region, so JPEG bridges cannot claim rows.
        seed = np.zeros_like(strong)
        seed[top:bottom,l:r] = strong[top:bottom,l:r]
        seed[lo:hi,:] = False
        pale = (ink >= .09) & cv2.dilate(seed.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        pale[lo:hi,:] = False
        yy, bx = np.nonzero(pale)
        if len(bx):
            l = max(l-1, int(bx.min()))
            r = min(r+1, int(bx.max())+1)
            top = min(top, int(yy.min()))
            bottom = max(bottom, int(yy.max())+1)
        glyph = (x0+l,y0+top,x0+r,y0+bottom)
        # Some bodies legitimately touch a measured frame. Report that
        # occlusion explicitly; a downstream classifier may remain unknown.
        frame_contact = [side for side,hit in (
            ('left', e > 0 and glyph[0] <= e),
            ('right', g < width and glyph[2] >= g),
            ('top', f > 0 and glyph[1] <= f),
            ('bottom', h < height and glyph[3] >= h)) if hit]
        proposal = dict(crop_box=[x0,y0,x1,y1], glyph_box=list(glyph),
            line_rows=list(range(y0+lo,y0+hi)), connected_line=True,
            score=score, frame_contact=frame_contact,
            row_terminations=row_terminations,
            line_analysis=dict(status='separated',reason='source_context_bilateral_body',
                support='contrast_normalized_original_gray',
                connector_rows=[y0+lo,y0+hi], bilateral_strength=peak,
                independent_upper_pixels=int(upper[max(0,l):r].sum()),
                independent_lower_pixels=int(lower[max(0,l):r].sum()),
                width=glyph[2]-glyph[0],height=glyph[3]-glyph[1]),
            source_pixels_modified=False)
        if not any(q['glyph_box']==proposal['glyph_box'] and
                   q['line_rows']==proposal['line_rows'] for q in proposals):
            proposals.append(proposal)
    proposals.sort(key=lambda q:-q['score'])
    return proposals, dict(diag, reason='measured_bilateral_body' if proposals else
                          'no_valid_body_hypothesis', rejections=sorted(set(rejected)),
                          candidate_count=len(proposals))

"""Measure long boundary strokes in legend context, never in a marker crop.

The returned interior only limits crop geometry. No source pixels are erased,
and a small closed square/ring is not sufficient evidence for a legend frame.
"""
from __future__ import annotations

import cv2
import numpy as np


def interior_bounds(image, legend, swatches=()):
    """Conservative half-open limits from straight strokes near legend edges."""
    h, w = image.shape[:2]
    if legend is None:
        return (0, 0, w, h), dict(policy='long_legend_boundary_v1', bands=[])
    a, b, c, d = map(int, legend)
    if not (0 <= a < c <= w and 0 <= b < d <= h):
        raise ValueError(f'Invalid legend bounds: {legend}')
    rows = [tuple(map(int, q)) for q in swatches]
    typical_h = float(np.median([q[3]-q[1] for q in rows])) if rows else 8.
    typical_w = float(np.median([q[2]-q[0] for q in rows])) if rows else 16.
    pad = max(3, min(6, int(np.ceil(.035*min(c-a, d-b)))))
    x0, y0, x1, y1 = max(0, a-pad), max(0, b-pad), min(w, c+pad), min(h, d+pad)
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    bounds = [0, 0, w, h]; bands = []
    # A stroke must persist over most of the entire legend, be much longer
    # than a handle/body and remain supported in each third of its length.
    def evidence(values, axis, minimum):
        if values.shape[axis] < minimum:
            return np.zeros(values.shape[1-axis], bool)
        strong = values < 235
        # Blurred letters can occupy >80% of a text row without forming a
        # border. A straight stroke also has a stable tone along its length.
        tone_range = np.percentile(values, 90, axis=axis)-np.percentile(values, 10, axis=axis)
        return (strong.mean(axis=axis) >= .80) & (tone_range <= 70) & np.logical_and.reduce([
            part.mean(axis=axis) >= .68 for part in np.array_split(strong, 3, axis=axis)])
    v = gray[b-y0:d-y0, :]
    u = gray[:, a-x0:c-x0]
    vertical = evidence(v, 0, max(20, 3*typical_h))
    horizontal = evidence(u, 1, max(35, 2.5*typical_w))
    for side, profile, values, axis, origin, edge in (
        ('left', vertical, v, 0, x0, a),
        ('right', vertical, v, 0, x0, c-1),
        ('top', horizontal, u, 1, y0, b),
        ('bottom', horizontal, u, 1, y0, d-1),
    ):
        positions = np.flatnonzero(profile)
        near = positions[np.abs(positions+origin-edge) <= pad]
        if side == 'left': near = near[near+origin <= a+.15*(c-a)]
        elif side == 'right': near = near[near+origin >= c-.15*(c-a)]
        elif side == 'top': near = near[near+origin <= b+.15*(d-b)]
        else: near = near[near+origin >= d-.15*(d-b)]
        if not len(near):
            continue
        best = int(near[np.argmin(np.abs(near+origin-edge))])
        start, stop = best, best+1
        while start > 0 and profile[start-1]: start -= 1
        while stop < len(profile) and profile[stop]: stop += 1
        # A diffuse block of text is not a narrow border. One measured pale
        # column/row may join the straight core but arbitrary padding may not.
        if stop-start > max(3, .08*min(c-a, d-b)):
            continue
        pale = (values < 248).mean(axis=axis)
        if start > 0 and pale[start-1] >= .75: start -= 1
        if stop < len(profile) and pale[stop] >= .75: stop += 1
        lo, hi = start+origin, stop+origin
        if side == 'left': bounds[0] = max(bounds[0], hi)
        elif side == 'right': bounds[2] = min(bounds[2], lo)
        elif side == 'top': bounds[1] = max(bounds[1], hi)
        else: bounds[3] = min(bounds[3], lo)
        bands.append(dict(side=side, start=lo, stop=hi, legend_edge=edge,
                          support_fraction=float((values.take(best, axis=1-axis) < 235).mean())))
    return tuple(bounds), dict(policy='long_legend_boundary_v1', legend=list(legend),
                               interior=list(bounds), bands=bands)


def clamp_swatch(box, bounds):
    """Do not retain a fragment when a putative boundary crosses its centre."""
    a, b, c, d = map(int, box)
    e, f, g, h = max(a, bounds[0]), max(b, bounds[1]), min(c, bounds[2]), min(d, bounds[3])
    if g-e < max(3, .60*(c-a)) or h-f < max(3, .60*(d-b)):
        return None
    if not (e <= (a+c)/2 < g and f <= (b+d)/2 < h):
        return None
    return e, f, g, h


def recovery_candidates(image, box, bounds, peers=()):
    """Retry within verified border limits, retaining source x identity.

    Directed ink completion is preferred. The existing two-pixel crop retry
    is also tested with independent top/bottom padding, always clamped before
    extraction so an expanded crop cannot reintroduce the verified frame.
    """
    from legend_crop_v46 import completion_candidates
    base = clamp_swatch(box, bounds)
    if base is None:
        return [], dict(reason='frame_crosses_body')
    a, b, c, d = base
    # Only keys next to a measured border need this alternate path.
    near = any(0 <= distance <= 2 for distance in (
        a-bounds[0] if bounds[0] else float('inf'),
        b-bounds[1] if bounds[1] else float('inf'),
        bounds[2]-c if bounds[2] < image.shape[1] else float('inf'),
        bounds[3]-d if bounds[3] < image.shape[0] else float('inf')))
    if not near:
        return [], dict(reason='no_border_contact')
    measured, diagnostic = completion_candidates(image, base, peers, bounds=bounds)
    candidates = [base] + measured
    for up, down in sorted(((u, v) for u in range(3) for v in range(3)), key=lambda q:(sum(q), q)):
        expanded = (a, max(bounds[1], b-up), c, min(bounds[3], d+down))
        if expanded not in candidates: candidates.append(expanded)
    return candidates, dict(reason='frame_bounded_remeasurement', completion=diagnostic)

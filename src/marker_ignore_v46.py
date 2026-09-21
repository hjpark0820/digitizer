"""Observed neutral furniture pixels, separate from allowed marker centres.

No source pixels are edited. Long thin boundary strokes identify axes; short
outward strokes identify ticks. Dense periodic small dots identify guides only
when their size is distinctly below the legend marker size. Colour at a crossing
is protected. A gray/black marker crossing an axis retains its off-axis evidence.
"""
from __future__ import annotations

import cv2
import numpy as np


def _runs(flags):
    edges = np.diff(np.r_[False, flags, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def build_ignore_mask(crop_bgr, *, valid=None, marker_diameter=None, include_guides=True):
    """Return a crop-local bool mask and JSON-safe diagnostics.

    ``valid`` describes excluded legends, not an axis-centre veto. Furniture
    classification is deliberately conservative: interior solid horizontal
    curves and vertical error bars are not classified as axes.
    """
    image = np.asarray(crop_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('crop_bgr must be an HxWx3 image')
    h, w = image.shape[:2]
    allowed = np.ones((h, w), bool) if valid is None else np.asarray(valid, bool)
    if allowed.shape != (h, w):
        raise ValueError('valid must match the crop shape')
    channels = image.astype(np.int16)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    neutral = np.ptp(channels, axis=2) <= 30
    core = neutral & (gray < 190) & allowed
    fringe = neutral & (gray < 225) & allowed
    axes = np.zeros((h, w), bool)
    ticks = axes.copy()
    guides = axes.copy()
    records = []
    axis_bands = []
    for orientation, binary in [('horizontal', core), ('vertical', core.T)]:
        length = binary.shape[1]
        # Opening cannot mistake a sparse row of markers for a solid axis.
        opened = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((1, max(9, round(.55*length))), np.uint8))
        flags = (opened.sum(axis=1) >= .55*length) & (binary.sum(axis=1) >= .65*length)
        for lo, hi in _runs(flags):
            dimension = binary.shape[0]
            center = .5*(lo+hi-1)
            if hi-lo > max(4, round(.018*min(h, w))):
                continue
            if min(center, dimension-1-center) > .15*dimension:
                continue
            # Extend only into adjacent rows that still have long-stroke support.
            soft = fringe if orientation == 'horizontal' else fringe.T
            start, stop = int(lo), int(hi)
            if start > 0 and soft[start-1].mean() >= .65:
                start -= 1
            if stop < dimension and soft[stop].mean() >= .65:
                stop += 1
            target = axes if orientation == 'horizontal' else axes.T
            target[start:stop] |= soft[start:stop]
            axis_bands.append((orientation, start, stop))
            records.append(dict(kind='axis', orientation=orientation,
                                band=[start, stop], center=float(center)))
            # Ticks point OUT of the plot. Do not mask inward error-bar stems.
            outward = -1 if center < dimension/2 else 1
            reach = max(3, min(12, round(.04*min(h, w))))
            begin = max(0, start-reach) if outward < 0 else stop
            end = start if outward < 0 else min(dimension, stop+reach)
            if end <= begin:
                continue
            strip = soft[begin:end]
            # Measure the dark tick core: JPEG/AA halos can make a 3px tick
            # look 7px wide or connect it to unrelated faint neighbouring ink.
            components, labels, stats, _ = cv2.connectedComponentsWithStats(binary[begin:end].astype(np.uint8), 8)
            tick_target = ticks if orientation == 'horizontal' else ticks.T
            for label in range(1, components):
                x, y, sw, sh, area = stats[label]
                touches = y+sh == end-begin if outward < 0 else y == 0
                if touches and sw <= max(3, hi-lo+1) and sh >= 2:
                    seed = (labels == label).astype(np.uint8)
                    halo = cv2.dilate(seed, np.ones((3,3),np.uint8)) > 0
                    tick_target[begin:end] |= halo & strip

    # Restrict guide length tests to the interior span delimited by axes.
    left, right = 0, w
    for orientation, lo, hi in axis_bands:
        if orientation == 'vertical':
            if lo < w/2:
                left = hi
            else:
                right = lo
    span = right-left
    diameter = float(marker_diameter or 0)
    seen = []
    if diameter >= 4 and span >= 30:
        for y in range(h):
            if any(abs(y-old) <= max(2, round(.25*diameter)) for old in seen):
                continue
            pulses = [(a+left, b+left) for a, b in _runs(core[y, left:right])]
            small = [(a, b) for a, b in pulses if b-a <= max(2, .50*diameter)]
            if len(small) < 10:
                continue
            widths = np.array([b-a for a, b in small], float)
            centers = np.array([.5*(a+b-1) for a, b in small])
            gaps = np.diff(centers)
            period = float(np.median(gaps[gaps <= np.percentile(gaps, 60)]))
            multiples = np.maximum(1, np.rint(gaps/max(period, 1)))
            regular = float(np.mean((np.abs(gaps/period-multiples) <= .20) & (multiples <= 4)))
            if (centers[-1]-centers[0] < .70*span or regular < .85 or
                    not 1.6 <= period/max(float(np.median(widths)), 1) <= 7):
                continue
            # Measure repeated dot thickness at its x positions; crossing
            # markers/error bars must not set the guide-band thickness.
            cy = np.rint(centers).astype(int)
            support = np.mean(core[:, cy], axis=1)
            bands = [(a, b) for a, b in _runs(support >= .55) if a <= y < b]
            if not bands:
                continue
            lo, hi = bands[0]
            if hi-lo > max(2, .40*diameter):
                continue
            lo, hi = max(0, lo-1), min(h, hi+1)
            guides[lo:hi, left:right] |= fringe[lo:hi, left:right]
            seen.append(y)
            records.append(dict(kind='periodic_guide', band=[int(lo), int(hi)],
                                period=period, pulses=len(small), regularity=regular,
                                x_span=[int(left),int(right)]))
    mask = axes | ticks | (guides if include_guides else False)
    return mask, dict(version='observed_furniture_ignore_v1',
        ignored_pixels=int(mask.sum()), axis_pixels=int(axes.sum()),
        tick_pixels=int(ticks.sum()), guide_pixels=int(guides.sum()) if include_guides else 0,
        guide_policy='legacy_band' if include_guides else 'separate_model', structures=records,
        source_pixels_deleted=False, whole_rows_deleted=False, center_veto=False,
        policy='Ignored pixels supply neither positive ink nor missing/extra-ink penalties; '
               'on-axis centres remain allowed and require visible independent marker ink.')

"""Shared achromatic error-bar evidence, independent of series colors.

Input is a native plot-local BGR crop and geometric valid mask. Output ``own``
is suitable for errorbar_stems.propose_stems(), but only proposes shared x
structures: gray lines do not establish a series identity or measurement y.

Long horizontal ink is suppressed only in this STRUCTURE evidence. The caller
must keep original per-series color evidence for fitting the surrounding curve.
An isolated short cap is never removed just because it is horizontal. A darker
cap superimposed on a faint grid is retained by its observed excess darkness;
same-gray coincident cap arms cannot be distinguished and remain unresolved.
No original image pixels, production modules, or color masks are modified.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = 'shared_neutral_stem_evidence_v1'


def _runs(indices):
    indices = np.asarray(indices, int)
    return [] if not len(indices) else np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)


def _paper_model(image, valid):
    pixels = image[valid].astype(np.float32)
    if not len(pixels):
        return np.full(3, 255., np.float32), 0.
    level = pixels.mean(axis=1)
    bright = pixels[level >= np.percentile(level, 80)]
    paper = np.median(bright, axis=0)
    noise = float(1.4826*np.median(np.abs(bright-paper)))
    return paper.astype(np.float32), noise


def _horizontal_support(raw_neutral, valid, line_width):
    """Read long rows with small gaps, without painting any gap as evidence."""
    height, width = raw_neutral.shape
    closing_width = 2*max(1, int(math.ceil(line_width)))+1
    closed = cv2.morphologyEx(raw_neutral.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((1, closing_width), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    closed &= valid
    candidate = np.zeros_like(raw_neutral)
    row_records = []
    for y in range(height):
        usable = np.flatnonzero(valid[y])
        if not len(usable):
            continue
        native_span = int(usable[-1]-usable[0]+1)
        minimum_span = max(24, int(math.ceil(16*line_width)), int(math.ceil(.45*native_span)))
        for run in _runs(np.flatnonzero(closed[y])):
            a, b = int(run[0]), int(run[-1])+1
            if b-a < minimum_span:
                continue
            occupancy = float(raw_neutral[y, a:b].sum()/max(1, valid[y, a:b].sum()))
            if occupancy < .65:
                continue
            candidate[y, a:b] = raw_neutral[y, a:b]
            row_records.append(dict(y=y, x0=a, x1=b, span=b-a, observed_fraction=occupancy,
                                    minimum_span=minimum_span))
    # Thick text/filled objects are not thin grid-line evidence.
    bands, rejected = [], []
    max_thickness = max(3, int(math.ceil(3*line_width)))
    for group in _runs(np.flatnonzero(candidate.any(axis=1))):
        a, b = int(group[0]), int(group[-1])+1
        if b-a > max_thickness:
            candidate[a:b] = False
            rejected.append(dict(y0=a, y1=b, reason='horizontal_object_too_thick'))
        else:
            bands.append(dict(y0=a, y1=b, thickness=b-a))
    return candidate, bands, row_records, rejected, closing_width


def build_neutral_evidence(crop_bgr, valid, line_width=1.):
    """Build source-only shared neutral masks in native plot-local coordinates.

    Returns ``own``, ``raw_neutral``, ``grid_mask`` (actually removed pixels),
    ``long_horizontal_mask`` (observed broad-row candidates),
    ``preserved_dark_detail``, ``preserved_vertical_bridge``, and diagnostics.

    All masks are bool[H,W]. All output ink masks are subsets of actual observed
    neutral ink and valid pixels. No new pixel is added across a gap. Boxes in
    diagnostics are half-open; this module does not return stem coordinates.
    ``line_width`` is the measured shared neutral stroke width, not a color's
    marker diameter. The default1px permits thin error bars on small figures.
    """
    image = np.asarray(crop_bgr)
    valid = np.asarray(valid)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise TypeError('crop_bgr must be a uint8 HxWx3 native BGR image')
    if valid.dtype != bool or valid.ndim != 2 or valid.shape != image.shape[:2]:
        raise ValueError('valid must be a same-size boolean geometric mask')
    line_width = float(line_width)
    if not math.isfinite(line_width) or line_width <= 0:
        raise ValueError('line_width must be finite and positive')
    lw = max(1., line_width)
    paper, noise = _paper_model(image, valid)
    delta = paper-image.astype(np.float32)
    darkness = np.maximum(0., delta.mean(axis=2))
    channel_residual = np.max(np.abs(delta-delta.mean(axis=2, keepdims=True)), axis=2)
    minimum_contrast = max(6., 3.*noise)
    neutral_base_tolerance = max(6., 3.*noise)
    neutral_tolerance = neutral_base_tolerance+.14*darkness
    raw = valid & (darkness >= minimum_contrast) & (channel_residual <= neutral_tolerance)
    if raw.size and raw.any():
        horizontal, bands, row_records, thick_rejected, close_width = _horizontal_support(raw, valid, lw)
    else:
        horizontal, bands, row_records, thick_rejected, close_width = np.zeros_like(raw), [], [], [], 0
    detail, bridge = np.zeros_like(raw), np.zeros_like(raw)
    row_models = []
    for y in np.flatnonzero(horizontal.any(axis=1)):
        values = darkness[y, horizontal[y]]
        baseline = float(np.median(values))
        sigma = float(1.4826*np.median(np.abs(values-baseline)))
        tolerance = max(10., 4.*sigma, .25*baseline)
        detail[y] = horizontal[y] & (darkness[y] > baseline+tolerance)
        row_models.append(dict(y=int(y), baseline_darkness=baseline,
                               robust_sigma=sigma, excess_darkness_required=tolerance,
                               darker_detail_pixels=int(detail[y].sum())))
    # Preserve observed neutral ink at a real vertical crossing, using ink on
    # both sides OUTSIDE the complete grid-row band. Merely opening with a 3px
    # vertical kernel can preserve the entire antialiased 3px-high grid itself.
    probe = max(2, int(math.ceil(2.*lw)))
    horizontal_drift = max(0, int(math.floor(.5*lw)))
    for band in bands:
        y0, y1 = band['y0'], band['y1']
        upper = raw[max(0, y0-probe):y0].any(axis=0)
        lower = raw[y1:min(raw.shape[0], y1+probe)].any(axis=0)
        if horizontal_drift:
            kernel = np.ones((1, 2*horizontal_drift+1), np.uint8)
            upper = cv2.dilate(upper[None].astype(np.uint8), kernel)[0].astype(bool)
            lower = cv2.dilate(lower[None].astype(np.uint8), kernel)[0].astype(bool)
        columns = upper & lower
        bridge[y0:y1] = horizontal[y0:y1] & columns[None, :]
        band['vertical_crossing_columns'] = int(columns.sum())
    removed = horizontal & ~detail & ~bridge
    own = raw & ~removed
    # Explicit invariants: no morphology-created pixel enters the evidence.
    detail &= raw
    bridge &= raw
    return dict(own=own, raw_neutral=raw, grid_mask=removed,
                long_horizontal_mask=horizontal,
                preserved_dark_detail=detail, preserved_vertical_bridge=bridge,
                diagnostics=dict(version=VERSION, coordinate_system='native plot-local pixels; diagnostic boxes half-open',
                    color_order='BGR', identity='shared achromatic structure only; no series assignment',
                    input_line_width=line_width, effective_line_width=lw,
                    paper_bgr=[float(v) for v in paper], paper_noise_sigma=noise,
                    minimum_contrast=minimum_contrast, neutral_base_tolerance=neutral_base_tolerance,
                    neutral_darkness_fraction=.14, horizontal_closing_width=close_width,
                    valid_pixels=int(valid.sum()), raw_neutral_pixels=int(raw.sum()),
                    long_horizontal_pixels=int(horizontal.sum()), removed_grid_pixels=int(removed.sum()),
                    preserved_darker_detail_pixels=int(detail.sum()),
                    preserved_vertical_crossing_pixels=int(bridge.sum()), own_pixels=int(own.sum()),
                    horizontal_bands=bands, horizontal_rows=row_records,
                    row_darkness_models=row_models, rejected_thick_horizontal_objects=thick_rejected,
                    semantics='Long-horizontal candidates are non-cap structure, not a guaranteed semantic grid classifier.',
                    caller_contract='Use own for shared vertical-x proposals only. Fit series y on unchanged original per-series color evidence.',
                    limitations=['Neutral evidence includes gray data curves and neutral text if not excluded by valid.',
                                 'Same-gray cap arms exactly coincident with a grid cannot be separated from the raster; not reconstructed.',
                                 'A darker detail can be another curve, not necessarily an error-bar cap; later stem/cap and curve tests are required.',
                                 'Both-sided crossing restoration preserves source pixels only, not occluded or missing stem pixels.',
                                 'A colored connector can split a tiny neutral bar into fragments below the stem detector minimum; colored intersection pixels are not borrowed or fabricated.',
                                 'No marker centers, manual coordinates, observation schedule or reference points are used.']))

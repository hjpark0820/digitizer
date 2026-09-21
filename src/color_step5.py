"""
color_step5.py -- v46 path-shape correction and legacy colour compatibility.

Validated v46 payloads dispatch to color_path_correction_v46, without SSIM.
The per-colour SSIM implementation documented below is the legacy fallback for
non-v46 callers only; it is not the current v46 correction algorithm.

Approach A (per-colour decomposition)
------------------------------------
The Step-5 correction is colour-agnostic: it needs a reference image, a point
list, a renderer and a segment detector -- all of which are classical CV.  So a
colour plot is handled by turning it into N independent black-and-white problems,
one per curve colour:

    for each curve c:
        I0_c = white canvas with ONLY curve c's ink kept (from its colour mask)
        P_c  = that curve's detected points
        run_correction(I0_c, init_points=P_c)      # unchanged Step-5 code

Because each curve is isolated, overlapping curves and colour bleed between
neighbouring hues cannot confuse the objective: a curve is only ever scored
against its own ink.

The caller supplies the per-curve ink mask (the colour pipeline already builds
one via `_curve_ink_mask`, which matches pixels against the clean legend swatch
colour rather than the contaminated assign map).  If no mask is given, a simple
swatch-colour distance mask is used as a fallback.

Typical use from the colour pipeline
------------------------------------
    from color_step5 import correct_colour_curves
    curves = [{'name': 'color06', 'swatch_rgb': (234,158,79),
               'points': [(x, y), ...],            # plot-space pixels
               'ink_mask': mask_bool_or_None}, ...]
    out = correct_colour_curves(img_bgr, curves, plot_area, legend_box)
    # out[i]['points'] -> corrected points for curve i
"""

from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
import sys
import tempfile

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Two proposals closer than this are the same data point (px).  Used to merge
# template-matched markers with supplied points and to drop suppressed
# proposals that duplicate an active point.
PT_MERGE_TOL = 12.0


# ---------------------------------------------------------------------------
# Step-5 module loader (file name starts with a digit -> import by path)
# ---------------------------------------------------------------------------
def _load_step5():
    path = os.path.join(_HERE, '5_correction_color.py')
    spec = importlib.util.spec_from_file_location('correction_color', path)
    mod = importlib.util.module_from_spec(spec)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Per-curve reference image
# ---------------------------------------------------------------------------
def swatch_ink_mask(img_bgr: np.ndarray, swatch_rgb, tol: float = 40.0,
                    plot_area=None, white_thresh: int = 244,
                    blend_max: float = 0.70) -> np.ndarray:
    """Fallback mask: pixels belonging to this curve, ANTI-ALIASING INCLUDED.

    A hard "distance to the swatch colour" test throws away the anti-aliased
    edge of every stroke -- measured at ~72% of a thin curve's pixels -- which
    leaves a dotted reference that the solid renderer cannot match, so the
    correction "improves" SSIM by deleting points.  Anti-aliased pixels are
    blends of the swatch colour toward the white page, so we accept any colour
    lying close to the segment  swatch -> white  instead.

    The colour pipeline's own `_curve_ink_mask` may be passed in instead.
    """
    px = img_bgr[:, :, ::-1].astype(np.float64)          # -> RGB
    s_ = np.asarray(swatch_rgb, dtype=np.float64)
    w_ = np.array([255.0, 255.0, 255.0])
    d = w_ - s_
    denom = float(d @ d) or 1.0
    t = np.clip(((px - s_) @ d) / denom, 0.0, 1.0)        # position along the blend
    proj = s_ + t[..., None] * d
    dist = np.linalg.norm(px - proj, axis=-1)
    nonwhite = px.min(axis=-1) < white_thresh
    # Cap the blend fraction: near the white end of the tube every colour
    # converges, so without this the mask swallows light grey text/JPEG noise.
    mask = (dist <= tol) & (t <= blend_max) & nonwhite
    if plot_area is not None:
        x0, y0, x1, y1 = [int(v) for v in plot_area]
        keep = np.zeros(mask.shape, dtype=bool)
        keep[y0:y1 + 1, x0:x1 + 1] = True
        mask &= keep
    return mask


def build_reference(img_bgr: np.ndarray, ink_mask: np.ndarray,
                    binary: bool = True) -> np.ndarray:
    """White canvas that keeps ONLY this curve's ink.

    binary=True paints the curve pure black, matching the renderer (which also
    draws pure black); copying the original greyscale instead leaves an
    anti-aliasing gradient that the renderer cannot reproduce, inflating the
    residual everywhere along the curve.
    """
    h, w = img_bgr.shape[:2]
    out = np.full((h, w), 255, dtype=np.uint8)
    if binary:
        out[ink_mask] = 0
    else:
        out[ink_mask] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[ink_mask]
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def estimate_stroke(ink_mask: np.ndarray) -> tuple:
    """(line_width, marker_radius) in px, measured from the curve's own ink.

    The renderer draws a fixed 1-px polyline and r=4 markers; if the real curve
    is thicker every segment carries a constant residual, which pushes the greedy
    step toward deleting points instead of fixing them.  A distance transform of
    the ink gives both numbers directly: along a plain stroke the transform
    plateaus at half the line width, and it peaks inside the markers.
    """
    m8 = (np.asarray(ink_mask, dtype=np.uint8) * 255)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    vals = dt[ink_mask]
    if vals.size == 0:
        return 1, 4
    # Line width from a LOW percentile: the median is pulled up by the markers
    # (measured on a 2px line with r=4 markers: median -> 3px, p25 -> 2px). An
    # over-thick render inks every segment more than the reference does, which
    # makes deleting points look like an improvement.
    lw = int(round(2 * float(np.percentile(vals, 25))))
    r  = int(round(float(np.percentile(vals, 98))))
    return max(1, min(lw, 6)), max(2, min(r, 10))


def marker_candidates(ink_mask: np.ndarray, active_pts, marker_r: int,
                      min_sep: float = None) -> list:
    """Suppressed-point pool for a colour curve.

    The B&W pipeline gets this pool for free: the ViT proposes many markers and
    the NMS files the losers under "suppressed", which the correction can later
    ACTIVATE.  The colour pipeline keeps no such pool, so without one the
    ACTIVATE action is dead and a marker the walk skipped can never be restored.

    Markers are locally THICKER than the connecting line, so the distance
    transform of the curve's own ink peaks at marker centres.  Every peak that is
    not already an active point becomes a candidate.
    """
    m8 = (np.asarray(ink_mask, dtype=np.uint8) * 255)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    thresh = max(1.5, float(marker_r) * 0.60)        # thicker than a plain stroke
    thick = (dt >= thresh).astype(np.uint8)
    if thick.sum() == 0:
        return []
    n, _lab, _st, cent = cv2.connectedComponentsWithStats(thick, 8)
    sep = float(min_sep if min_sep is not None else max(3.0, marker_r * 1.2))
    out = []
    for i in range(1, n):
        cx, cy = float(cent[i][0]), float(cent[i][1])
        if any((cx - x) ** 2 + (cy - y) ** 2 <= sep * sep for (x, y) in active_pts):
            continue                                  # already represented
        out.append({'cx': cx, 'cy': cy,
                    'class_name': 'suppressed', 'class_idx': -1})
    return out


def _load_segment_detector():
    path = os.path.join(_HERE, '3_segment_detection_v2.py')
    spec = importlib.util.spec_from_file_location('segment_detector_c', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_dashed_mask(ink_mask: np.ndarray, plot_area=None) -> tuple:
    """(is_dashed, zero_fraction) using the pipeline's own test.

    run_A4_walk classifies a curve as dashed when more than 15% of the columns
    it spans carry no ink (`zero_frac > 0.15`); WIN_W=25 is sized for exactly
    those gaps.  The same test is applied here so segment extraction agrees with
    the walk about which curves are dashed.
    """
    m = np.asarray(ink_mask, bool)
    cols = m.any(0)
    xs = np.nonzero(cols)[0]
    if xs.size < 2:
        return False, 0.0
    span = cols[xs.min():xs.max() + 1]
    zero_frac = float((~span).sum()) / max(span.size, 1)
    return zero_frac > 0.15, zero_frac


def bridge_dashes(ink_mask: np.ndarray, gap: int = None) -> np.ndarray:
    """Close the gaps of a dashed stroke so it reads as one line.

    The segment detector needs continuous ink: a dashed curve breaks into
    fragments whose x-span falls under `refine`'s min_span, so every piece is
    pruned and the curve ends up with no segments at all.  A morphological
    closing along the stroke bridges the gaps without thickening the line
    (closing = dilate then erode, so the original width is restored).
    """
    m = (np.asarray(ink_mask, bool) * 255).astype(np.uint8)
    if gap is None:
        # widest dash gap actually present, capped at the walk's WIN_W (25px)
        cols = np.asarray(ink_mask, bool).any(0)
        xs = np.nonzero(cols)[0]
        gap = 5
        if xs.size >= 2:
            run = 0; longest = 0
            for v in cols[xs.min():xs.max() + 1]:
                run = 0 if v else run + 1
                longest = max(longest, run)
            gap = int(min(max(longest + 2, 3), 25))
    # Close along several orientations and union the results: a single
    # horizontal kernel only bridges horizontal gaps, so the dashes of a steep
    # or vertical stroke stay separated.  Each kernel is thin, so this bridges
    # far less aggressively than one large disc would.
    g = max(3, int(gap))
    out = np.zeros_like(m, bool)
    for ang in (0, 45, 90, 135):
        k = np.zeros((g, g), np.uint8)
        c = g // 2
        if ang == 0:
            k[c, :] = 1
        elif ang == 90:
            k[:, c] = 1
        elif ang == 45:
            for i in range(g):
                k[g - 1 - i, i] = 1
        else:
            for i in range(g):
                k[i, i] = 1
        out |= cv2.morphologyEx(m, cv2.MORPH_CLOSE, k) > 0
    return out


def extract_segments(bin_img: np.ndarray, prep_info=None) -> list:
    """Run the segment detector on a binary (white background) image."""
    try:
        seg = _load_segment_detector()
        if bin_img.ndim == 2:
            bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        return list(seg.detect_debug(bin_img, prep_info=prep_info)['segments'])
    except Exception as e:
        print(f"  [color-step5] segment extraction failed: {e}")
        return []


def _segments_near_mask(segments, ink_mask: np.ndarray, tol: int = 2,
                        frac: float = 0.90) -> list:
    """Keep only segments whose body lies on this curve's ink.

    Segments taken from the whole plot (read as greyscale) describe every curve;
    without this filter a curve would be offered candidate locations that belong
    to its neighbours.  The test is deliberately strict (2px tolerance, 90% of
    the segment on the ink): at 4px/60% the black curve admitted 39 foreign
    segments totalling 15000px of span on a 652px-wide plot -- segments that
    merely passed nearby -- which after grid cutting became 294 bogus candidates.
    """
    if not segments:
        return []
    m = np.asarray(ink_mask, np.uint8)
    if tol > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        m = cv2.dilate(m, k)
    H, W = m.shape[:2]
    out = []
    for sgc in segments:
        x0, y0, x1, y1 = [int(round(v)) for v in sgc[:4]]
        n = max(2, int(max(abs(x1 - x0), abs(y1 - y0))))
        hit = 0
        for t in range(n + 1):
            x = int(round(x0 + (x1 - x0) * t / n)); y = int(round(y0 + (y1 - y0) * t / n))
            if 0 <= x < W and 0 <= y < H and m[y, x]:
                hit += 1
        if hit >= frac * (n + 1):         # essentially ON this curve
            out.append(sgc)
    return out


def _segments_off_stems(segments, stems_mask: np.ndarray, frac: float = 0.5) -> list:
    """Drop segments that run along an error-bar stem.

    A same-colour error bar is tall vertical ink, so the segment detector reports
    it as a perfectly good segment.  Offered as an l* candidate it invites points
    ON the error bar.  `confirmed_stems` marks those pixels with the pipeline's
    own geometric test, so any segment lying mostly on them is removed.
    """
    if not segments or stems_mask is None or not np.any(stems_mask):
        return list(segments)
    H, W = stems_mask.shape[:2]
    out = []
    for sg in segments:
        x0, y0, x1, y1 = [float(v) for v in sg[:4]]
        n = max(2, int(max(abs(x1 - x0), abs(y1 - y0))))
        on = 0
        for t in range(n + 1):
            x = int(round(x0 + (x1 - x0) * t / n)); y = int(round(y0 + (y1 - y0) * t / n))
            if 0 <= x < W and 0 <= y < H and stems_mask[y, x]:
                on += 1
        if on < frac * (n + 1):
            out.append(sg)
    return out


def chart_furniture_mask(img_bgr, plot_area, span_frac=0.50, max_thick=4,
                         marker_extent=6, blob_area=260):
    """Mask the chart's own lines: axes, gridlines and reference lines (LLOQ).

    These are drawn in black/grey, so an achromatic curve's tube claims them and
    markers get "found" strung along the x-axis or the LLOQ rule.  Furniture is
    told apart from a genuinely flat data curve by thickness alone: a rule is
    thin along its whole length, while a data curve carries markers, so some
    columns are much thicker than the line itself.

    A row (or column) qualifies when it is dark across most of the plot, its
    contiguous band is at most `max_thick` px, and nowhere in the band does the
    ink swell to `marker_extent` px -- i.e. it never carries a marker.
    """
    x0, y0, x1, y1 = [int(v) for v in plot_area]
    H, W = img_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = np.zeros((H, W), bool)
    dark[y0:y1 + 1, x0:x1 + 1] = g[y0:y1 + 1, x0:x1 + 1] < 170
    out = np.zeros((H, W), bool)

    def _bands(counts, need, lo, hi):
        hits = [i for i, c in enumerate(counts) if c >= need]
        groups, cur = [], []
        for i in hits:
            if cur and i == cur[-1] + 1:
                cur.append(i)
            else:
                if cur:
                    groups.append(cur)
                cur = [i]
        if cur:
            groups.append(cur)
        return [(lo + gp[0], lo + gp[-1]) for gp in groups
                if len(gp) <= max_thick and lo + gp[-1] <= hi]

    # A rule's anti-aliased fringe is lighter than the core, so the core band is
    # grown outwards while the neighbouring row/column is still mostly ink --
    # otherwise the fringe survives and carries detections along the axis.
    soft = np.zeros((H, W), bool)
    soft[y0:y1 + 1, x0:x1 + 1] = g[y0:y1 + 1, x0:x1 + 1] < 215

    def _grow_rows(a, b):
        while a - 1 >= y0 and soft[a - 1, x0:x1 + 1].sum() >= need_h:
            a -= 1
        while b + 1 <= y1 and soft[b + 1, x0:x1 + 1].sum() >= need_h:
            b += 1
        return a, b

    def _grow_cols(a, b):
        while a - 1 >= x0 and soft[y0:y1 + 1, a - 1].sum() >= need_v:
            a -= 1
        while b + 1 <= x1 and soft[y0:y1 + 1, b + 1].sum() >= need_v:
            b += 1
        return a, b

    # horizontal rules (x-axis, LLOQ, gridlines)
    need_h = span_frac * (x1 - x0 + 1)
    need_v = span_frac * (y1 - y0 + 1)
    counts = [int(dark[y, x0:x1 + 1].sum()) for y in range(y0, y1 + 1)]
    for by0, by1 in _bands(counts, need_h, y0, y1):
        pad = marker_extent
        band = dark[max(y0, by0 - pad):min(y1 + 1, by1 + 1 + pad), x0:x1 + 1]
        ext = np.array([(np.ptp(np.nonzero(band[:, c])[0]) + 1) if band[:, c].any() else 0
                        for c in range(band.shape[1])])
        # A rule stays thin along its length; a flat data curve swells at every
        # marker, so the typical (median) column is what separates them.
        if ext.size and float(np.median(ext[ext > 0])) < marker_extent:
            by0, by1 = _grow_rows(by0, by1)
            out[by0:by1 + 1, x0:x1 + 1] = True

    # vertical rules (y-axis, gridlines)
    counts = [int(dark[y0:y1 + 1, x].sum()) for x in range(x0, x1 + 1)]
    for bx0, bx1 in _bands(counts, need_v, x0, x1):
        pad = marker_extent
        band = dark[y0:y1 + 1, max(x0, bx0 - pad):min(x1 + 1, bx1 + 1 + pad)]
        ext = np.array([(np.ptp(np.nonzero(band[r, :])[0]) + 1) if band[r, :].any() else 0
                        for r in range(band.shape[0])])
        if ext.size and float(np.median(ext[ext > 0])) < marker_extent:
            bx0, bx1 = _grow_cols(bx0, bx1)
            out[y0:y1 + 1, bx0:bx1 + 1] = True

    # A rule's dashes and its label ("LLOQ = ...") sit on the band and are dark,
    # so they are claimed by an achromatic curve and match as markers. They are
    # small isolated blobs; a real curve crossing the rule is a large component
    # and survives.
    if out.any():
        n_lab, lab, st, _ct = cv2.connectedComponentsWithStats(
            dark.astype(np.uint8), 8)
        for k in range(1, n_lab):
            if st[k, cv2.CC_STAT_AREA] > blob_area:
                continue
            comp = lab == k
            if (comp & out).any():
                out |= comp
    return out


def _tube_dist(sub_rgb, swatch_rgb):
    """Distance of each pixel to the swatch->white blend line."""
    s = np.asarray(swatch_rgb, np.float64)
    d = np.array([255.0, 255.0, 255.0]) - s
    den = float(d @ d) or 1.0
    t = np.clip(((sub_rgb - s) @ d) / den, 0.0, 1.0)
    return np.linalg.norm(sub_rgb - (s + t[..., None] * d), axis=-1)


def swatch_marker_template(img_bgr, swatch_box, swatch_rgb, tol=46.0,
                           other_swatches=None):
    """Cut the MARKER shape out of a curve's legend swatch, as a binary patch.

    A swatch is usually a line with a marker on it (--o--).  The marker is the
    part with a large VERTICAL EXTENT; the connecting line is thin and spans the
    whole swatch, so per-column extent separates them without eroding open
    markers (a ring/hollow triangle survives, which morphological opening would
    destroy).  Error bars are excluded by capping the marker's height to roughly
    its width -- markers are compact, error bars are tall and thin.

    Returns a float32 binary patch usable as a matchTemplate template, or None
    when the swatch carries no marker (line-only legend entry).
    """
    x0, y0, x1, y1 = [int(v) for v in swatch_box]
    H, W = img_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sub = img_bgr[y0:y1, x0:x1, ::-1].astype(np.float64)
    dist = _tube_dist(sub, swatch_rgb)
    ink = (dist <= tol) & (sub.min(axis=-1) < 244)
    # Keep only pixels this swatch owns.  Without it the grey/neighbouring
    # pixels of the connecting line join the template and widen it into a bar.
    for o in (other_swatches or []):
        if np.array_equal(np.asarray(o), np.asarray(swatch_rgb)):
            continue
        ink &= dist <= _tube_dist(sub, o)
    if ink.sum() < 20:
        return None

    ext = np.array([(np.ptp(np.nonzero(ink[:, c])[0]) + 1) if ink[:, c].any() else 0
                    for c in range(ink.shape[1])])
    if ext.max() < 5:
        return None                       # thin everywhere -> line-only swatch
    # Marker columns: tall in absolute terms (the validated swatch rule) and
    # tall relative to this swatch, which separates the marker from the line.
    tall = set(np.nonzero(ext >= max(5, 0.5 * ext.max()))[0].tolist())
    cmax = int(np.argmax(ext))
    lo = hi = cmax
    while lo - 1 in tall:
        lo -= 1
    while hi + 1 in tall:
        hi += 1
    patch = ink[:, lo:hi + 1]
    ry = np.nonzero(patch.any(axis=1))[0]
    if ry.size < 4:
        return None
    patch = patch[ry.min():ry.max() + 1]
    # drop error-bar overhang: keep a compact core centred on the marker
    ph, pw = patch.shape
    if ph > 1.8 * pw:
        cy = ph // 2
        half = int(round(0.9 * pw))
        patch = patch[max(0, cy - half):min(ph, cy + half + 1)]
    if min(patch.shape) < 4:
        return None
    return patch.astype(np.float32)


def template_match_markers(ink_mask, template, plot_area, legend_box=None,
                           thresh=0.40, log_fn=None, name=''):
    """Find this curve's markers by exact-matching the legend marker shape.

    Correlation runs on the curve's BINARY ink map, so the match is on shape
    rather than shade -- a marker dimmed by anti-aliasing or partly covered by
    another curve still correlates on the part that survives.  Peaks are picked
    strongest-first with a separation of roughly one marker width.

    Returns [(cx, cy, score), ...].  Partial occlusion is deliberately NOT
    resolved here: these are handed to the correction stage, which decides what
    is real from the reconstruction error.
    """
    if template is None or ink_mask is None:
        return []
    th, tw = template.shape[:2]
    m = np.asarray(ink_mask, bool).copy()
    H, W = m.shape
    keep = np.zeros((H, W), bool)
    px0, py0, px1, py1 = [int(v) for v in plot_area]
    keep[py0:py1 + 1, px0:px1 + 1] = True
    if legend_box:
        lx0, ly0, lx1, ly1 = [int(v) for v in legend_box]
        keep[ly0:ly1 + 1, lx0:lx1 + 1] = False
    m &= keep
    if m.sum() < template.sum() * 0.5 or H < th or W < tw:
        return []
    res = cv2.matchTemplate(m.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    if res.size == 0:
        return []
    thr = max(thresh, float(res.max()) * 0.55)
    ys, xs = np.where(res >= thr)
    if ys.size == 0:
        return []
    scores = res[ys, xs]
    sep = max(7, int(0.9 * max(tw, th)))
    picks = []
    for k in np.argsort(-scores):
        cx = float(xs[k] + tw / 2.0)
        cy = float(ys[k] + th / 2.0)
        if any((cx - px) ** 2 + (cy - py) ** 2 <= sep * sep for px, py, _ in picks):
            continue
        # A marker occupies real height.  A dash of a reference rule or a
        # gridline can correlate with the template's outline while being only a
        # few px tall, so require the ink here to be as tall as the marker.
        iy0 = max(0, int(cy - th))
        iy1 = min(H, int(cy + th) + 1)
        ix0 = max(0, int(cx - tw // 2))
        ix1 = min(W, int(cx + tw // 2) + 1)
        loc = m[iy0:iy1, ix0:ix1]
        if not loc.any():
            continue
        ext = max((np.ptp(np.nonzero(loc[:, c])[0]) + 1)
                  for c in range(loc.shape[1]) if loc[:, c].any())
        if ext < 0.5 * th:
            continue
        picks.append((cx, cy, float(scores[k])))
    if log_fn:
        log_fn("    [marker-template] %s: %dx%d template, %d match(es) "
               "(thr %.2f)" % (name, tw, th, len(picks), thr))
    return picks


def _resolve_orphan_segments(gray_segments, curves, img_bgr, plot_area,
                             legend_box=None, log_fn=print,
                             ep_tol=6, vote_frac=0.58,
                             dist_gate=62.0, margin=28.0):
    """Assign plot-as-BW segments that no curve's ink claimed, by continuity + colour.

    A greyscale segment taken from the whole plot describes some curve, but in a
    crowded region its body can straddle several curves' dilated ink and so fail
    every curve's `_segments_near_mask` test, leaving it unassigned.  Such a
    segment still belongs to exactly one curve, placed by two signals:

      * own colour -- the ORIGINAL pixels under the segment vote for one legend
                      swatch.  This is the primary signal: it rescues a segment
                      that has no assigned neighbour and, at a crossing, overrides
                      a neighbour belonging to a different curve.
      * continuity -- the segment shares an endpoint (<= ep_tol px) with a segment
                      already assigned to a curve, so it continues that curve's
                      polyline; used to narrow the candidates at a crossing.

    Assignment uses per-curve masks built by NEAREST swatch (a pixel is ink for
    the curve whose swatch->white tube it is closest to), so broad overlapping
    masks cannot let one curve claim a neighbour's segment -- the failure mode of
    the plain swatch masks in dense multi-curve plots.  A segment whose colour
    vote is split across curves (a true crossing the greyscale traced through) is
    left unassigned rather than forced onto a guess.

    Returns {curve_name: [segment, ...]} : segments to ADD to that curve.  The
    stage is purely additive; it never removes or reassigns existing segments.
    """
    names, sw = [], {}
    for ci, cv_ in enumerate(curves):
        s = cv_.get('swatch_rgb')
        if s is None:
            continue
        n = cv_.get('name', 'curve%d' % ci)
        names.append(n); sw[n] = np.asarray(s, np.float64)
    if not gray_segments or len(names) < 2:
        return {}

    H, W = img_bgr.shape[:2]
    rgb = img_bgr[:, :, ::-1].astype(np.float64)

    # Nearest swatch->white tube per pixel -> exclusive per-curve ink masks.
    # (best_i / best_d are kept incrementally so only two H*W planes are held.)
    white = np.array([255.0, 255.0, 255.0])
    best_d = np.full((H, W), 1e9)
    best_i = np.full((H, W), -1, int)
    for i, n in enumerate(names):
        s = sw[n]; d = white - s; den = float(d @ d) or 1.0
        t = np.clip(((rgb - s) @ d) / den, 0.0, 1.0)
        dd = np.linalg.norm(rgb - (s + t[..., None] * d), axis=-1)
        dd[t > 0.70] = 1e9                     # too close to white for this curve
        upd = dd < best_d
        best_d[upd] = dd[upd]; best_i[upd] = i
    nonwhite = rgb.min(axis=-1) < 244
    roi = np.zeros((H, W), bool)
    px0, py0, px1, py1 = [int(v) for v in plot_area]
    roi[py0:py1 + 1, px0:px1 + 1] = True
    if legend_box:
        lx0, ly0, lx1, ly1 = [int(v) for v in legend_box]
        roi[ly0:ly1 + 1, lx0:lx1 + 1] = False
    owned = nonwhite & roi & (best_d <= 40.0)
    dil = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))    # 2px, like the filter
    for i in range(len(names)):
        dil.append(cv2.dilate((owned & (best_i == i)).astype(np.uint8), k).astype(bool))

    def _on(sg, dm):
        x0, y0, x1, y1 = [int(round(v)) for v in sg[:4]]
        st = max(2, int(max(abs(x1 - x0), abs(y1 - y0))))
        hit = 0
        for t in range(st + 1):
            x = int(round(x0 + (x1 - x0) * t / st))
            y = int(round(y0 + (y1 - y0) * t / st))
            if 0 <= x < W and 0 <= y < H and dm[y, x]:
                hit += 1
        return hit >= 0.90 * (st + 1)

    accept = [set() for _ in gray_segments]
    for i, dm in enumerate(dil):
        for j, sg in enumerate(gray_segments):
            if _on(sg, dm):
                accept[j].add(i)
    orphans = [j for j, a in enumerate(accept) if not a]
    if not orphans:
        return {}

    ep = [(np.asarray(s[:2], float), np.asarray(s[2:4], float)) for s in gray_segments]

    def _conn(a, b):
        a0, a1 = ep[a]; b0, b1 = ep[b]
        return (np.linalg.norm(a0 - b0) <= ep_tol or np.linalg.norm(a0 - b1) <= ep_tol
                or np.linalg.norm(a1 - b0) <= ep_tol or np.linalg.norm(a1 - b1) <= ep_tol)

    def _colour(sg):
        x0, y0, x1, y1 = [int(round(v)) for v in sg[:4]]
        st = max(2, int(max(abs(x1 - x0), abs(y1 - y0))))
        cols = []
        for t in range(st + 1):
            x = int(round(x0 + (x1 - x0) * t / st))
            y = int(round(y0 + (y1 - y0) * t / st))
            best = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < W and 0 <= yy < H and rgb[yy, xx].min() < 244:
                        p = rgb[yy, xx]
                        if best is None or p.sum() < best.sum():
                            best = p
            if best is not None:
                cols.append(best)
        if not cols:
            return None
        cols = np.asarray(cols)
        votes = np.zeros(len(names))
        for p in cols:
            votes[int(np.argmin([np.linalg.norm(p - sw[nm]) for nm in names]))] += 1
        return np.median(cols, axis=0), votes, len(cols)

    ostat = {}
    for j in orphans:
        c = _colour(gray_segments[j])
        if c is not None:
            ostat[j] = c

    def _d(med, i):
        return float(np.linalg.norm(med - sw[names[i]]))

    resolved = {}
    pending = list(ostat.keys())
    for _pass in range(6):
        newly = []
        for j in pending:
            med, votes, ncol = ostat[j]
            cand = set()
            for q, a in enumerate(accept):           # continuity: assigned neighbours
                if q != j and a and _conn(j, q):
                    cand |= a
            for rq, ri in resolved.items():          # + neighbours resolved this run
                if rq != j and _conn(j, rq):
                    cand.add(ri)
            own = int(np.argmax(votes))              # own colour: a clear majority
            if votes[own] >= vote_frac * ncol and _d(med, own) <= dist_gate:
                cand.add(own)
            if not cand:
                continue
            ranked = sorted(cand, key=lambda i: _d(med, i))
            best = ranked[0]
            if _d(med, best) <= dist_gate and (
                    len(ranked) == 1 or _d(med, ranked[1]) - _d(med, best) >= margin):
                resolved[j] = best
                newly.append(j)
        for j in newly:
            pending.remove(j)
        if not newly:
            break

    out = {}
    for j, i in resolved.items():
        out.setdefault(names[i], []).append(gray_segments[j])
    log_fn("  [orphan-rescue] %d unassigned BW segment(s): %d recovered by "
           "continuity+colour, %d left (ambiguous crossings)"
           % (len(orphans), len(resolved), len(orphans) - len(resolved)))
    for n in sorted(out):
        log_fn("    [orphan-rescue] %s: +%d segment(s)" % (n, len(out[n])))
    return out


def path_from_mask(ink_mask: np.ndarray):
    """Per-column representative y of the curve = its traced path.

    Returns {x: y}. Columns whose ink splits into several runs use the run with
    the most pixels, so an error-bar stem does not drag the path off the curve.
    """
    m = np.asarray(ink_mask, bool)
    H, W = m.shape
    path = {}
    for x in range(W):
        col = np.nonzero(m[:, x])[0]
        if col.size == 0:
            continue
        runs = np.split(col, np.where(np.diff(col) > 2)[0] + 1)
        best = max(runs, key=len)
        path[x] = float(best.mean())
    return path


def confirmed_stems(ink_mask: np.ndarray) -> np.ndarray:
    """Error-bar stems, using the pipeline's own geometric test.

    Ported from ChartDigitizer.detect_stems in run_A4_auto_v44.py, which builds
    `stems_confirmed` -- the STRICT set that is safe to subtract from a curve.
    Three independent geometric criteria, not a thickness heuristic:

      1. vertical      opening with a (1, 9) kernel  -> only tall vertical ink
      2. not the curve  minus the (4, 1) opening      -> the curve line and marker
                                                         bodies are wide HORIZONTALLY
      3. isolated       drop pixels that have ink 3px to the left AND 3px to the
                        right (i.e. sitting on a horizontal run)
      then keep components with height >= 12 and width <= 3.
    """
    ink = np.asarray(ink_mask, np.uint8)
    stems = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    curve_region = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))).astype(bool)
    st = stems.astype(bool) & ~curve_region
    left = np.zeros_like(ink); left[:, 3:] = ink[:, :-3]
    right = np.zeros_like(ink); right[:, :-3] = ink[:, 3:]
    st = st & ~(left.astype(bool) & right.astype(bool))
    n, lbl, stt, _ = cv2.connectedComponentsWithStats(st.astype(np.uint8), 8)
    conf = np.zeros(ink.shape, bool)
    for i in range(1, n):
        if stt[i, cv2.CC_STAT_HEIGHT] >= 12 and stt[i, cv2.CC_STAT_WIDTH] <= 3:
            conf[lbl == i] = True
    return conf


def vertical_runs(ink_mask: np.ndarray, marker_r: int, min_len_factor: float = 3.0):
    """Columns where the CURVE itself rises vertically (a re-dose spike).

    One x then legitimately carries two data points -- the trough and the peak --
    which x-only NMS would merge. A same-colour error bar is also a tall vertical
    run, so it must not be mistaken for one; the pipeline's own geometric stem
    test (`confirmed_stems`) separates them, and the run's endpoints must sit on
    the curve body rather than on a bare stem.

    Returns [(x, y_top, y_bottom), ...].
    """
    m = np.asarray(ink_mask, bool)
    H, W = m.shape
    stems = confirmed_stems(m)
    # curve body = ink that is wide horizontally (the pipeline's `curve_region`)
    body = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))).astype(bool)
    min_len = max(int(marker_r * min_len_factor), 8)
    out = []
    for x in range(W):
        col = np.nonzero(m[:, x])[0]
        if col.size < min_len:
            continue
        for run in np.split(col, np.where(np.diff(col) > 2)[0] + 1):
            if len(run) < min_len:
                continue
            y0, y1 = int(run[0]), int(run[-1])
            # Stem test on THIS column only: a 3-column average dilutes a 1-2px
            # stem below any useful threshold.
            if stems[y0:y1 + 1, x].mean() > 0.5:
                continue                       # confirmed error-bar stem
            xs = slice(max(0, x - 1), x + 2)
            end = max(2, marker_r)
            if body[y0:y0 + end + 1, xs].any() and body[y1 - end:y1 + 1, xs].any():
                out.append((x, y0, y1))        # curve body at BOTH ends -> a rise
    merged = []
    for x, y0, y1 in out:
        if merged and x - merged[-1][0] <= max(2, marker_r) and abs(y0 - merged[-1][1]) <= marker_r:
            continue
        merged.append((x, y0, y1))
    return merged


def short_curve_points(ink_mask: np.ndarray, marker_r: int, active_pts,
                       grid_step: float = None):
    """Points for a curve too small for the grid to describe.

    A very low-dose series can occupy less than one grid cell (measured on the
    ALPN plot: the 0.012 mg/kg curve spans x 150-158 with 50 px of ink), so
    `refine` prunes every one of its segments -- min_span is half a grid step --
    and the correction is left with nothing to act on.

    Such a curve carries either ONE point or TWO. The ink's extent along its
    principal axis decides: about one marker wide means a single point; clearly
    longer means two, and they sit at the two ENDS of the ink.

    Returns None when the curve is not short, else the list of (x, y) points.
    """
    m = np.asarray(ink_mask, bool)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    xspan = float(xs.max() - xs.min() + 1)
    if grid_step and xspan >= grid_step:
        return None                              # wide enough for the grid
    diam = max(3.0, 2.0 * float(marker_r))
    if grid_step is None and xspan >= 3 * diam:
        return None

    pts = np.stack([xs.astype(float), ys.astype(float)], 1)
    c = pts.mean(0)
    u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]                                  # principal direction
    t = (pts - c) @ axis
    extent = float(t.max() - t.min() + 1)
    if extent <= 1.5 * diam:
        return [(float(c[0]), float(c[1]))]       # a single marker
    lo = c + axis * float(t.min()) + axis * (diam * 0.5)
    hi = c + axis * float(t.max()) - axis * (diam * 0.5)
    return [(float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))]


def grid_path_candidates(ink_mask: np.ndarray, grid_xs, active_pts,
                         min_sep: float = 6.0, half_win: int = 3,
                         vruns=None) -> list:
    """Suppressed pool from GRID x COLUMN intersections with the curve path.

    Data points sit where a shared x-column crosses the curve, so this proposes
    exactly the right places even when the ink has no thickness cue -- the case
    that leaves the black curve with almost no marker-like blobs (its markers are
    hidden under other curves and its error bars are the same neutral colour).
    """
    path = path_from_mask(ink_mask)
    if not path:
        return []
    W = ink_mask.shape[1]
    out = []
    # A vertical rise carries TWO points at the same x -- propose both ends.
    _vr = {int(x): (y0, y1) for (x, y0, y1) in (vruns or [])}
    for gx in grid_xs:
        _hit = [x for x in _vr if abs(x - gx) <= max(half_win, 4)]
        if _hit:
            _x = _hit[0]; _y0, _y1 = _vr[_x]
            for _cy in (_y0, _y1):
                if any((_x - px) ** 2 + (_cy - py) ** 2 <= min_sep * min_sep
                       for (px, py) in active_pts):
                    continue
                out.append({'cx': float(_x), 'cy': float(_cy),
                            'class_name': 'suppressed', 'class_idx': -1})
            continue
        gx = int(round(gx))
        ys = [path[x] for x in range(max(0, gx - half_win), min(W, gx + half_win + 1))
              if x in path]
        if not ys:
            continue
        cy = float(np.median(ys))
        if any((gx - px) ** 2 + (cy - py) ** 2 <= min_sep * min_sep
               for (px, py) in active_pts):
            continue                      # already represented by an active point
        out.append({'cx': float(gx), 'cy': cy,
                    'class_name': 'suppressed', 'class_idx': -1})
    return out


def seed_candidates(step5, ref_bgr, active, cands, cls, cidx, min_gain=0.0008,
                    log_fn=print, max_seed=None, name=''):
    """Greedily add the grid x path candidates that measurably help.

    ACTIVATE only ever looks at the suppressed point nearest each l* endpoint, so
    a proposal sitting mid-segment is never tried even when it is exactly the
    missing marker.  Seeding tests every candidate directly against the same
    render+SSIM objective and keeps the ones that improve it, which is what lets
    a curve whose ink has no thickness cue (markers hidden under other curves,
    neutral error bars) recover its points at all.

    This runs BEFORE the greedy and is not part of `max_iters`, so it can add
    several points even on a single-iteration run; `max_seed` caps how many, and
    0 disables seeding entirely when a run has to be one action exactly.

    Returns (seeded_active_points, leftover_candidates).
    """
    H, W = ref_bgr.shape[:2]
    P = lambda pts: [{'cx': float(x), 'cy': float(y),
                      'class_name': cls, 'class_idx': cidx} for (x, y) in pts]
    dist = lambda pts: step5.ssim_dist(ref_bgr, step5.render_from_points(P(pts), (H, W)))
    cur = list(active)
    pool = [(float(c['cx']), float(c['cy'])) for c in cands]
    if max_seed is not None and int(max_seed) <= 0:
        left = [{'cx': float(x), 'cy': float(y),
                 'class_name': 'suppressed', 'class_idx': -1} for (x, y) in pool]
        log_fn(f"    [seed] {name}: disabled -- {len(pool)} candidate(s) "
               f"left suppressed")
        return cur, left
    base = dist(cur)
    added = 0
    while pool:
        if max_seed is not None and added >= int(max_seed):
            log_fn(f"    [seed] {name}: cap of {int(max_seed)} reached")
            break
        scored = [(dist(sorted(cur + [c])), c) for c in pool]
        scored.sort(key=lambda z: z[0])
        best_d, best_c = scored[0]
        if best_d >= base - min_gain:
            break
        cur = sorted(cur + [best_c]); pool.remove(best_c)
        base = best_d; added += 1
    if added:
        log_fn(f"    [seed] {name}: added {added} grid-path point(s) before "
               f"the greedy (NOT counted in max_iters), 1-SSIM -> {base:.5f}")
    left = [{'cx': float(x), 'cy': float(y),
             'class_name': 'suppressed', 'class_idx': -1} for (x, y) in pool]
    return cur, left


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def correct_colour_curves(img_bgr: np.ndarray,
                          curves: list,
                          plot_area,
                          legend_box=None,
                          marker_class: str = 'filled_circle',
                          max_iters=None,
                          max_seed=None,
                          out_dir=None,
                          grid_xs=None,
                          workers=None,
                          log_fn=print,
                          explicit_payload=None,
                          previous_state=None,
                          engine_dir=None,
                          path_options=None) -> list:
    """Run Step-5 independently on every colour curve.

    Parameters
    ----------
    img_bgr      : original colour plot (BGR)
    curves       : list of dicts with
                     'name'       -- curve id (e.g. 'color06')
                     'swatch_rgb' -- (r, g, b) legend swatch colour
                     'points'     -- [(x, y), ...] detected points, image pixels
                     'ink_mask'   -- optional bool array; built from the swatch
                                     colour when omitted
                     'marker_class' -- optional per-curve marker class name
    plot_area    : (x0, y0, x1, y1) used for filtering and to crop the SSIM
    legend_box   : (x0, y0, x1, y1) excluded from the correction, or None
    marker_class : default marker shape used to render points

    Returns
    -------
    list of dicts: {'name', 'points', 'n_before', 'n_after', 'ssim_before',
                    'ssim_after'}
    """
    if explicit_payload is not None:
        # An authoritative v46 handoff already supplies native masks, geometry,
        # path segments and a tentative pool. None of the legacy pre-seeding or
        # independent mask/segment/template heuristics below should run again.
        from color_step5_payload_v46 import correct_payload_curves
        return correct_payload_curves(img_bgr, curves, explicit_payload,
            out_dir=out_dir or tempfile.mkdtemp(prefix='color_step5_v46_'),
            max_iters=max_iters, previous_state=previous_state, workers=workers,
            engine_dir=engine_dir, log_fn=log_fn,path_options=path_options)

    _shared_step5 = _load_step5()
    cls_names = list(getattr(_shared_step5, 'CLASS_NAMES', ['filled_circle']))

    # Segment set A: the WHOLE plot read as black-and-white. A colour mask can be
    # fragmentary where curves overlap, but the plot's line geometry is intact in
    # the greyscale image, so these segments recover the true path shape. They are
    # filtered per curve so one curve is never offered its neighbour's line.
    _g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _px0, _py0, _px1, _py1 = [int(v) for v in plot_area]
    _gray_bin = np.full(_g.shape, 255, np.uint8)
    _roi = np.zeros(_g.shape, bool); _roi[_py0:_py1 + 1, _px0:_px1 + 1] = True
    if legend_box:
        _lx0, _ly0, _lx1, _ly1 = [int(v) for v in legend_box]
        _roi[_ly0:_ly1 + 1, _lx0:_lx1 + 1] = False
    _gray_bin[_roi & (_g < 215)] = 0
    gray_segments = extract_segments(_gray_bin)
    log_fn(f"  [color-step5] plot-as-BW segments: {len(gray_segments)}")

    # A plot-as-BW segment in a crowded region can straddle several curves' ink,
    # so no curve's near-mask test claims it. Recover the ones that clearly belong
    # to a single curve (endpoint continuity + own colour) and drop the genuinely
    # ambiguous crossings. Purely additive -- each recovered segment is added to
    # its curve's candidate list below; nothing existing is changed.
    _rescued_segments = _resolve_orphan_segments(
        gray_segments, curves, img_bgr, plot_area,
        legend_box=legend_box, log_fn=log_fn)

    # Axes / gridlines / reference rules are drawn in black, so an achromatic
    # curve's colour tube claims them and points get detected strung along the
    # x-axis or the LLOQ rule. Exclude them from every curve's ink.
    _furniture = chart_furniture_mask(img_bgr, plot_area)
    if _furniture.any():
        log_fn(f"  [furniture] {int(_furniture.sum())} px of axis/rule lines "
               f"excluded from curve ink")

    base_out = out_dir or tempfile.mkdtemp(prefix='color_step5_')
    os.makedirs(base_out, exist_ok=True)

    # Curves are independent, so they can run concurrently.  Each task loads its
    # OWN correction module instance: the module carries per-curve state in
    # globals (RENDER_LW, _CV_MARKER_R, _SSIM_CROP, NMS_VERTICAL_KEEP), which
    # would race if the instance were shared.  The correction is already
    # thread-parallel inside a curve, so each instance's worker count is scaled
    # down to keep the total near the CPU count.
    _n = len(curves)
    if workers is None:
        workers = max(1, min(4, os.cpu_count() or 1, _n))
    workers = max(1, int(workers))
    _inner = max(2, (os.cpu_count() or 4) // workers)

    def _process(ci_cv):
        ci, cv_ = ci_cv
        step5 = _load_step5() if workers > 1 else _shared_step5
        step5.PARALLEL_WORKERS = _inner
        name = cv_.get('name', f'curve{ci}')
        pts = list(cv_.get('points') or [])
        # NOTE: a curve with fewer than two points is NOT skipped here -- that is
        # exactly the short-curve case (one detected point where the ink holds
        # two), which is decided from the ink further below.
        mask = cv_.get('ink_mask')
        if mask is None:
            mask = swatch_ink_mask(img_bgr, cv_.get('swatch_rgb', (0, 0, 0)),
                                   plot_area=plot_area)
        mask = np.asarray(mask, dtype=bool)
        if _furniture.any():
            _n_ink = int(mask.sum())
            mask = mask & ~_furniture
            if int(mask.sum()) != _n_ink:
                log_fn(f"    [furniture] {name}: dropped "
                       f"{_n_ink - int(mask.sum())} px on axis/rule lines")
        if not mask.any():
            log_fn(f"  [color-step5] {name}: empty ink mask -- skipped")
            return {'name': name, 'points': pts, 'n_before': len(pts),
                    'n_after': len(pts), 'ssim_before': None, 'ssim_after': None}

        ref = build_reference(img_bgr, mask)
        lw, mr = estimate_stroke(mask)
        step5.RENDER_LW = lw
        step5._CV_MARKER_R = mr
        cls = cv_.get('marker_class', marker_class)
        cidx = cls_names.index(cls) if cls in cls_names else 0

        # A curve smaller than one grid cell cannot be described by segments --
        # `refine` prunes them all -- so decide its 1 or 2 points directly.
        _gstep = None
        if grid_xs and len(grid_xs) >= 2:
            _gstep = float(np.median(np.diff(sorted(grid_xs))))
        _short = short_curve_points(mask, mr, pts, grid_step=_gstep)
        if _short is not None:
            log_fn(f"  [color-step5] {name}: short curve (ink={int(mask.sum())}px) "
                   f"-> {len(_short)} point(s) at the ink ends, correction skipped")
            return {'name': name, 'points': [(float(x), float(y)) for x, y in _short],
                    'n_before': len(pts), 'n_after': len(_short),
                    'ssim_before': None, 'ssim_after': None}

        # Segment set B: this curve's own colour mask. A dashed stroke -- or a
        # mask with holes where curves overlap -- breaks into fragments that
        # `refine` prunes entirely, so bridge the gaps first (measured on the
        # 9-colour ALPN plot: several curves went from 0 surviving segments to
        # 11-15). The bridged mask is used ONLY for segment extraction; the
        # reference image the SSIM scores against stays untouched.
        _dashed, _zf = is_dashed_mask(mask)
        if _dashed:
            _seg_src = build_reference(img_bgr, bridge_dashes(mask))
            log_fn(f"    [dashed] {name}: {_zf*100:.0f}% empty columns "
                   f"-> gaps bridged for segment extraction")
        else:
            _seg_src = ref
        mask_segments = extract_segments(_seg_src)
        near_gray = _segments_near_mask(gray_segments, mask)
        # Add any orphan BW segments recovered for THIS curve (continuity+colour),
        # skipping ones the near-mask test already kept.
        _resc = _rescued_segments.get(name)
        if _resc:
            _near_ids = {id(s) for s in near_gray}
            _resc = [s for s in _resc if id(s) not in _near_ids]
            if _resc:
                near_gray = list(near_gray) + list(_resc)
                log_fn(f"    [orphan-rescue] {name}: +{len(_resc)} recovered "
                       f"BW segment(s) added to candidates")
        # Remove segments that lie along this curve's error bars, so the greedy is
        # never offered an l* that sits on a stem.
        _stems = confirmed_stems(mask)
        _n_before = len(mask_segments) + len(near_gray)
        mask_segments = _segments_off_stems(mask_segments, _stems)
        near_gray = _segments_off_stems(near_gray, _stems)
        segs = list(mask_segments) + list(near_gray)
        if _n_before != len(segs):
            log_fn(f"    [stem filter] {name}: dropped {_n_before - len(segs)} "
                   f"segment(s) on error bars")

        # Suppressed pool = grid x path intersections + thickness (marker) peaks.
        vruns = vertical_runs(mask, mr)
        gcands = (grid_path_candidates(mask, grid_xs or [], pts, vruns=vruns)
                  if grid_xs else [])
        # Let a vertical rise keep both of its points through the NMS.
        step5.NMS_VERTICAL_KEEP = float(mr) * 2.5 if vruns else 0.0
        mcands = marker_candidates(mask, pts, mr)

        # Seed the grid x path proposals that measurably improve the objective,
        # BEFORE the greedy runs (ACTIVATE alone cannot reach mid-segment ones).
        step5._SSIM_CROP = tuple(int(v) for v in plot_area)   # same crop as the greedy
        _n_supplied = len(pts)          # before seeding, for an honest tally
        if gcands:
            _ms = max_seed
            if _ms is None and os.environ.get('COLOR_STEP5_MAX_SEED') is not None:
                _ms = int(os.environ['COLOR_STEP5_MAX_SEED'])
            pts, gcands = seed_candidates(step5, ref, pts, gcands, cls, cidx,
                                          log_fn=log_fn, max_seed=_ms, name=name)
        _n_seeded = len(pts) - _n_supplied

        seen = set(); init_S = []
        for c in list(gcands) + list(mcands):
            key = (round(c['cx'] / 4), round(c['cy'] / 4))
            if key in seen:
                continue
            seen.add(key); init_S.append(c)

        # ACTIVE set = supplied points + markers found by exact-matching the
        # legend marker shape.  A template hit is a data point seen directly in
        # the plot, so it starts active; the path/density proposals above are
        # inferred and start suppressed for the greedy to promote or discard.
        _tm = []
        _sbox = cv_.get('swatch_box')
        if _sbox is not None and cv_.get('swatch_rgb') is not None:
            _others = [c.get('swatch_rgb') for c in curves
                       if c is not cv_ and c.get('swatch_rgb') is not None]
            _tmpl = swatch_marker_template(img_bgr, _sbox, cv_['swatch_rgb'],
                                           other_swatches=_others)
            if _tmpl is not None:
                _tm = template_match_markers(mask, _tmpl, plot_area,
                                             legend_box=legend_box,
                                             log_fn=log_fn, name=name)
            elif log_fn:
                log_fn(f"    [marker-template] {name}: line-only swatch, "
                       f"no marker template")

        _act = [(float(x), float(y)) for (x, y) in pts]
        _n_pts = len(_act)
        for cx, cy, _sc in _tm:                       # de-duplicate against pts
            if all((cx - ax) ** 2 + (cy - ay) ** 2 > PT_MERGE_TOL ** 2
                   for ax, ay in _act):
                _act.append((cx, cy))
        if log_fn and len(_act) > _n_pts:
            log_fn(f"    [marker-template] {name}: +{len(_act) - _n_pts} new "
                   f"active point(s) from template matches")

        init_P = [{'cx': float(x), 'cy': float(y),
                   'class_name': cls, 'class_idx': cidx,
                   'confidence': 1.0} for (x, y) in _act]

        # Keep the suppressed pool clear of anything now active.
        init_S = [c for c in init_S
                  if all((c['cx'] - ax) ** 2 + (c['cy'] - ay) ** 2 > PT_MERGE_TOL ** 2
                         for ax, ay in _act)]

        # Minimal prep_info: the reference is already clean, so no clean_fn.
        prep_info = {
            'plot_area': tuple(int(v) for v in plot_area),
            'user_plot_area': tuple(int(v) for v in plot_area),
            'legend_box': tuple(int(v) for v in legend_box) if legend_box else None,
            'clean_fn': None,
        }

        cur_out = os.path.join(base_out, name)
        os.makedirs(cur_out, exist_ok=True)
        tmp_png = os.path.join(cur_out, 'reference.png')
        cv2.imwrite(tmp_png, ref)

        log_fn(f"  [color-step5] {name}: active {len(init_P)} "
               f"({_n_supplied} supplied + {_n_seeded} seeded + "
               f"{len(init_P) - _n_pts} marker-template), "
               f"ink={int(mask.sum())}px, stroke lw={lw} marker_r={mr} | "
               f"segs {len(mask_segments)}(mask)+{len(near_gray)}(BW) | "
               f"suppressed {len(init_S)} ({len(gcands)} grid-path, "
               f"{len(mcands)} thickness), {len(vruns)} vertical rise(s) "
               f"-> correcting ...")
        try:
            r = step5.run_correction(
                img_path=tmp_png,
                model_path=None,
                detector_py_path=None,      # detector-free: init_points supplied
                known_classes=[cls],
                out_dir=cur_out,
                mode_xs=None,               # inferred from the points
                prep_info=prep_info,
                return_diag_imgs=False,
                max_iters=max_iters,
                init_points=init_P,
                init_suppressed=init_S,
                segments_override=(segs or None),
                grid_xs_override=(list(grid_xs) if grid_xs else None),
            )
            P_out = r.get('P_current') or []
            hist = r.get('history') or []
            s_before = hist[0][3] if hist else None
            s_after = hist[-1][3] if hist else None
            new_pts = [(float(p['cx']), float(p['cy'])) for p in P_out]
            log_fn(f"  [color-step5] {name}: {len(init_P)} -> {len(new_pts)} pts, "
                   f"1-SSIM {s_before:.5f} -> {s_after:.5f}"
                   if s_before is not None else
                   f"  [color-step5] {name}: {len(init_P)} -> {len(new_pts)} pts")
            return {'name': name, 'points': new_pts,
                    'n_before': len(init_P), 'n_after': len(new_pts),
                    'ssim_before': s_before, 'ssim_after': s_after,
                    'out_dir': cur_out}
        except Exception as e:
            log_fn(f"  [color-step5] {name}: FAILED ({e}) -- keeping original points")
            return {'name': name, 'points': pts, 'n_before': len(pts),
                    'n_after': len(pts), 'ssim_before': None, 'ssim_after': None}

    tasks = list(enumerate(curves))
    if workers > 1 and _n > 1:
        log_fn(f"  [color-step5] {_n} curves on {workers} worker(s) "
               f"({_inner} threads each)")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_process, tasks))
    else:
        results = [_process(t) for t in tasks]
    return results

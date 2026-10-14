"""Periodic bitonal screen detection and descreening for native B&W detection.

A bitonal halftone print encodes grey as the DENSITY of ink dots on a fixed
periodic lattice. Measured pixel by pixel, the same grey glyph is a different
set of inked pixels wherever it falls on a different screen phase, so
exact-pixel tests (legend line continuity, strict window core, contour-sector
and orientation checks, glyph connectivity, fill read from dither holes)
measure the lattice phase rather than the marker. The tone a reader sees is
the local ink density over one screen period.

This module detects such a screen ONCE per image and, only then, returns the
tone map: the ink density averaged over exactly one period of the screen's
dominant lattice. Every detection stage (legend extraction, shape/fill
classification, proposals, windows) then sees one consistent continuous-tone
image, legend and plot alike. The caller keeps the untouched source image for
overlays and provenance.

The average also spreads every outline by half a period (thin strokes of a
small glyph merge; text and line bundles become blobs). Restricting the tone to
a closed source-ink support keeps outlines crisp, but a closing narrower than
the full dither matrix leaves the larger gaps of some grey levels as white
holes, i.e. it re-introduces the screen-phase dependence this module removes
(measured: fewer false positives, 8 fewer recovered markers). It is therefore
not used; the outline spread is a resolution limit of the print, shared by
legend and plot alike.

``detect_screen`` is a CANDIDATE test, not proof of a screen. It requires:
  * the image must be bitonal (two grey levels, as a printed/faxed screen is);
  * the ink must carry a 2-D periodic lattice: a sharp spectral peak at a
    period of 2-8 px AND its 90-degree rotated partner (square screens such as
    ordered dither or clustered-dot screens);
  * that lattice must carry a real share of the ink's energy.
It does NOT check that the peak is the fundamental of a fine screen: a regular
bitonal grid, dashed grid or hatch fill has comb harmonics in the 2-8 px band
with a 90-degree partner and passes all three tests (measured on gridded fax
charts, fix_C verification). Spectral and spatial fundamental-period tests were
measured and do not separate those grids from the benchmark's real screens
without a tuned margin (fix_C_rf README), so specificity is not claimed here.

The tone map is therefore only USED after ``tone_map_decision`` confirms, on
the legend itself, that (a) the source pixels fail to resolve at least one
legend marker, (b) the tone map resolves every legend marker into a named
marker model and (c) it does not contradict any marker the source pixels
already resolve. The legend is the identity reference for every plot marker;
a tone map that cannot resolve it (or disagrees with the source where the
source works) is not a better measurement than the source pixels, and the
source path is kept unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

VERSION = 'bw-halftone-screen-v1-legend-gated'

# Screens coarser than 8 px stop being texture at marker scale (a 12 px marker
# would hold barely one cell); finer than 2 px is the Nyquist limit.
MIN_PERIOD, MAX_PERIOD = 2, 8
# A printed screen is ONE phase-coherent lattice over the whole page, so a
# whole-image spectrum concentrates it into sharp peaks. A lattice peak must
# stand this far above the mean power of the surrounding frequency ring. With
# no periodic component the ratio is exponentially distributed (mean 1; the
# maximum over ~10^6 bins is ~15), so 30 means "genuinely periodic".
MIN_PEAK_CONTRAST = 30.
# Share of the ink's total spectral energy (DC excluded) carried by the two
# lattice peaks and their conjugates (+-2 bins each). Measured on the 100-chart
# benchmark (README): the 5 bitonal halftone prints 0.0042-0.027, the 4 bitonal
# fax charts 0.00006-0.00058. One fax chart has a sharp period-8 resampling
# lattice (nearest-neighbour resize, contrast 23-43): energetically negligible.
MIN_SCREEN_SHARE = .001
# Ring around a bin used as its local background, in cycles/px (independent of
# the image size); the inner gap excludes the window's main lobe.
RING_INNER, RING_OUTER = .004, .02
# Two-level test: at most this share of pixels may be strictly between the two
# extreme grey levels (anti-aliased/scanned images have several percent).
MAX_MIDTONE_SHARE = .002
# The spectrum is measured on at most this many pixels per side (the most
# ink-dense window). A printed screen is global, so a 2048 px window holds
# hundreds of periods; it bounds FFT memory on very large scans.
ANALYSIS_SIDE = 2048


def _gray(image):
    image = np.asarray(image)
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _analysis_window(ink):
    """Most ink-dense window of at most ANALYSIS_SIDE x ANALYSIS_SIDE pixels."""
    h, w = ink.shape
    wh, ww = min(h, ANALYSIS_SIDE), min(w, ANALYSIS_SIDE)
    if (wh, ww) == (h, w):
        return ink, (0, 0)
    integral = cv2.integral(ink.astype(np.uint8))
    best, origin = -1., (0, 0)
    stride = max(1, ANALYSIS_SIDE // 16)
    for y in list(range(0, h - wh + 1, stride)) + [h - wh]:
        for x in list(range(0, w - ww + 1, stride)) + [w - ww]:
            count = integral[y + wh, x + ww] - integral[y, x + ww] - integral[y + wh, x] + integral[y, x]
            if count > best:
                best, origin = float(count), (y, x)
    y, x = origin
    return ink[y:y + wh, x:x + ww], origin


def _spectrum(ink):
    """Hann-windowed power spectrum (fftshifted) and its local background
    (mean over a square frequency ring around every bin)."""
    from scipy import fft as sfft
    h, w = ink.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    power = np.fft.fftshift(np.abs(sfft.fft2((ink - ink.mean()) * win)) ** 2).astype(np.float64)
    def box(radius_y, radius_x):
        size = (2 * radius_x + 1, 2 * radius_y + 1)
        return cv2.boxFilter(power, -1, size, normalize=False, borderType=cv2.BORDER_REFLECT), size[0] * size[1]
    outer, n_out = box(max(2, int(round(RING_OUTER * h))), max(2, int(round(RING_OUTER * w))))
    inner, n_in = box(max(1, int(round(RING_INNER * h))), max(1, int(round(RING_INNER * w))))
    background = (outer - inner) / max(n_out - n_in, 1)
    return power, background


def detect_screen(image):
    """Return a JSON-ready report; ``report['detected']`` gates descreening."""
    gray = _gray(image)
    lo, hi = int(gray.min()), int(gray.max())
    report = dict(version=VERSION, detected=False, period=None, levels=[lo, hi])
    if hi - lo < 128:
        report['reason'] = 'not_two_level'
        return report
    midtone = float(((gray > lo) & (gray < hi)).mean())
    report['midtone_share'] = midtone
    if midtone > MAX_MIDTONE_SHARE:
        report['reason'] = 'not_bitonal'
        return report
    ink, origin = _analysis_window((gray < (lo + hi) / 2.).astype(np.float32))
    report['analysis_window'] = [int(origin[1]), int(origin[0]), int(origin[1] + ink.shape[1]), int(origin[0] + ink.shape[0])]
    h, w = ink.shape
    power, background = _spectrum(ink)
    contrast = power / np.maximum(background, 1.e-12)
    fy = (np.arange(h) - h // 2)[:, None] / h
    fx = (np.arange(w) - w // 2)[None, :] / w
    cheb = np.maximum(abs(fx), abs(fy))
    band = (cheb >= 1. / MAX_PERIOD - 1.e-9) & (cheb <= 1. / MIN_PERIOD + 1.e-9) & (fy >= 0)
    local_max = power >= cv2.dilate(power, np.ones((5, 5), np.uint8))
    candidates = band & local_max & (contrast >= MIN_PEAK_CONTRAST)
    best = None
    for iy, ix in zip(*np.nonzero(candidates)):
        bx, by = float(fx[0, ix]), float(fy[iy, 0])
        # 90-degree partner of (bx, by) is (-by, bx); bins differ when h != w.
        jy, jx = int(round(bx * h)) + h // 2, int(round(-by * w)) + w // 2
        if not (2 <= jy < h - 2 and 2 <= jx < w - 2):
            continue
        partner = float(contrast[jy - 2:jy + 3, jx - 2:jx + 3].max())
        lattice = min(float(contrast[iy, ix]), partner)
        # The screen's dominant frequency carries the most power; finer
        # harmonics and coarser sub-lattices are removed or weakened with it.
        if lattice >= MIN_PEAK_CONTRAST and (best is None or power[iy, ix] > best[0]):
            best = (float(power[iy, ix]), bx, by, float(contrast[iy, ix]), partner, lattice)
    if best is None:
        report['reason'] = 'no_two_dimensional_periodic_lattice'
        return report
    _, bx, by, peak, partner, lattice = best
    def lobe(fx_, fy_):
        iy, ix = int(round(fy_ * h)) + h // 2, int(round(fx_ * w)) + w // 2
        return float(power[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3].sum())
    total = float(power.sum() - power[h // 2, w // 2])
    share = 2. * (lobe(bx, by) + lobe(-by, bx)) / max(total, 1.e-12)
    period = int(round(1. / max(abs(bx), abs(by))))
    report.update(peak_frequency=[bx, by], peak_contrast=peak, partner_contrast=partner,
                  lattice_contrast=lattice, screen_energy_share=share, candidate_period=period)
    if share < MIN_SCREEN_SHARE:
        report['reason'] = 'periodic_lattice_without_screen_energy'
        return report
    report.update(detected=True, period=period, reason='bitonal_periodic_screen')
    return report


def _period_kernel(period):
    """1-D box over exactly one period with half-weight ends (symmetric, no
    half-pixel shift). Its response is zero at every multiple of 1/period."""
    k = np.ones(period + 1, np.float32)
    k[0] = k[-1] = .5
    return k / k.sum()


def descreen(image, report):
    """Tone map: ink density averaged over one screen period, same geometry."""
    if not report.get('detected'):
        return image
    k = _period_kernel(int(report['period']))
    src = np.asarray(image)
    out = cv2.sepFilter2D(src.astype(np.float32), -1, k, k, borderType=cv2.BORDER_REPLICATE)
    return np.clip(np.rint(out), 0, 255).astype(src.dtype)


# ---------------------------------------------------------------------------
# Legend consistency gate: use the tone map only where it demonstrably helps.
# ---------------------------------------------------------------------------

def _resolved(report):
    """A legend entry resolved into a named marker model (not unknown/failed)."""
    return (report is not None and report.get('extraction_status') == 'usable'
            and report.get('class_name') not in (None, 'unknown_marker'))


def _legend_reports(image, legend_area, swatches):
    """Per-entry reports of the production B&W legend extractor (no class
    filter: the gate measures whether each legend glyph is resolvable)."""
    from bw_legend_v46 import extract_legend_models, LegendExtractionError
    try:
        return extract_legend_models(image, legend_area, swatches, None)[1]
    except LegendExtractionError as error:      # no usable entry at all
        return list(error.reports)


def _centre(report):
    box = report.get('box') or report.get('glyph_box')
    if not box or len(box) != 4:
        return None
    return (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.


def _align(source, tone, explicit):
    """Pair source and tone entries. Explicit swatch boxes give the same entry
    order on both images; automatically discovered entries are paired when the
    centre of one lies inside the other's box."""
    if explicit:
        return [(s, t) for s, t in zip(source, tone)]
    pairs, used = [], set()
    for s in source:
        match = None
        for j, t in enumerate(tone):
            if j in used:
                continue
            cs, ct = _centre(s), _centre(t)
            bs, bt = s.get('box'), t.get('box')
            if cs and ct and bs and bt and (bt[0] <= cs[0] <= bt[2] and bt[1] <= cs[1] <= bt[3]
                                            or bs[0] <= ct[0] <= bs[2] and bs[1] <= ct[1] <= bs[3]):
                match = j
                break
        if match is None:
            pairs.append((s, None))
        else:
            used.add(match)
            pairs.append((s, tone[match]))
    pairs += [(None, t) for j, t in enumerate(tone) if j not in used]
    return pairs


def legend_gate(source_reports, tone_reports, explicit=True):
    """Decide from the two legend extractions whether the tone map may replace
    the source pixels. Parameter-free; see the module docstring.

    An entry that fails on BOTH images (e.g. a line-only legend entry) has no
    marker to resolve and does not take part."""
    entries, source_fails, tone_unresolved, contradictions = [], 0, 0, 0
    for s, t in _align(list(source_reports), list(tone_reports), explicit):
        s_ok, t_ok = _resolved(s), _resolved(t)
        s_usable = s is not None and s.get('extraction_status') == 'usable'
        t_usable = t is not None and t.get('extraction_status') == 'usable'
        if not s_usable and not t_usable:
            continue                                  # no marker on either image
        source_fails += not s_ok
        tone_unresolved += not t_ok
        contradiction = s_ok and t_ok and s.get('class_name') != t.get('class_name')
        contradictions += contradiction
        entries.append(dict(swatch_id=(s or t).get('swatch_id'),
                            source=(s.get('class_name') if s_usable else 'failed') if s else 'missing',
                            tone=(t.get('class_name') if t_usable else 'failed') if t else 'missing',
                            contradiction=bool(contradiction)))
    if not entries:
        accepted, reason = False, 'no_legend_marker_on_either_image'
    elif source_fails == 0:
        accepted, reason = False, 'source_pixels_resolve_every_legend_marker'
    elif tone_unresolved:
        accepted, reason = False, 'tone_map_leaves_legend_markers_unresolved'
    elif contradictions:
        accepted, reason = False, 'tone_map_contradicts_source_resolved_marker'
    else:
        accepted, reason = True, 'tone_map_resolves_markers_the_source_cannot'
    return dict(accepted=accepted, reason=reason, entries=entries,
                source_unresolved=int(source_fails), tone_unresolved=int(tone_unresolved),
                contradictions=int(contradictions))


def tone_map_decision(image, report, legend_area=None, swatches=None):
    """Return (image for detection, report). The source image is returned
    unless a screen candidate was detected AND the legend gate accepts the
    tone map. ``swatches`` are (name, box) pairs in this image's coordinates;
    the supplied names are passed through so both extractions see the same
    user input."""
    report = dict(report)
    report['tone_map_used'] = False
    if not report.get('detected'):
        return image, report
    if legend_area is None and swatches is None:
        report['legend_gate'] = dict(accepted=False, reason='no_legend_to_verify_the_tone_map')
        return image, report
    entries = None if swatches is None else [(name, tuple(int(round(v)) for v in box)) for name, box in swatches]
    tone = descreen(image, report)
    try:
        gate = legend_gate(_legend_reports(image, legend_area, entries),
                           _legend_reports(tone, legend_area, entries), explicit=entries is not None)
    except Exception as error:                     # the source path must never break
        gate = dict(accepted=False, reason='legend_gate_error', error=repr(error)[:300])
    report['legend_gate'] = gate
    if gate['accepted']:
        report['tone_map_used'] = True
        return tone, report
    return image, report

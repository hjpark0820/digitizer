"""
chart_preprocessing.py
======================
Grayscale/black-and-white chart noise removal module for chartocode2 pipeline.

Adapted from run_A4_auto_v22.py (colour-based pipeline) to work on
monochrome scientific charts (patent figures, journal plots, etc.) where
all curves, axes, text, and legend are drawn in black/grey on white.

Key operations
--------------
1. detect_axes(img_bgr)
   -> (axis_row, axis_col)  pixel coordinates of x-axis row and y-axis column

2. detect_plot_area(img_bgr, axis_row, axis_col)
   -> (x0, y0, x1, y1)  plotting rectangle (inside the axes)

3. detect_legend_box(img_bgr, plot_area)
   -> (x0, y0, x1, y1) or None

4. detect_lloq_line(binary_mask, plot_area)
   -> row index or None

5. remove_noise_from_binary(binary_mask, img_bgr,
                             axis_row, axis_col, plot_area,
                             legend_box, lloq_row)
   -> cleaned binary mask (axes, legend, LLOQ line, text removed)

6. preprocess(img_bgr)  -- convenience wrapper: runs all steps and returns
   -> dict with keys:
        'plot_area'    : (x0,y0,x1,y1) or None
        'axis_row'     : int or None
        'axis_col'     : int or None
        'legend_box'   : (x0,y0,x1,y1) or None
        'lloq_row'     : int or None
        'clean_fn'     : callable(binary_mask) -> cleaned_mask
        'debug_img'    : annotated BGR image showing detections

Public usage in pipeline
------------------------
    from chart_preprocessing import preprocess
    info = preprocess(img_bgr)
    # pass info['clean_fn'] to segment_detection and point_detection
"""

from __future__ import annotations
import cv2
import numpy as np
import math
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ── tuneable constants ────────────────────────────────────────────────────────
AXIS_PAD          = 4      # px to blank around detected axis lines
MIN_AXIS_RUN_FRAC = 0.35   # axis line must span >= this fraction of image
TICK_MARGIN       = 18     # px around axis to consider "axis-adjacent"
TICK_AREA_MAX     = 250    # max component area to remove as tick/noise
CHAR_MAX_AREA     = 150    # max component area to consider as a text character
TEXT_CLUSTER_R    = 65     # radius for clustering text characters

# ── LLOQ detection parameters (relaxed for real-world dashed lines) ──────────
LLOQ_DASH_MIN     = 3      # min dash segments to classify a row as LLOQ line
LLOQ_SPAN_FRAC    = 0.35   # LLOQ dashes must span >= this fraction of plot width
LLOQ_SEG_MAX_LEN  = 80     # individual dash segment must be <= this px long
LLOQ_SEG_MIN_LEN  = 2      # individual dash segment must be >= this px long
LLOQ_Y_BAND       = 18     # px band around detected LLOQ row to clear
LLOQ_COMP_MAX     = 400    # components larger than this are protected (curve/marker)
LLOQ_GAP_MIN      = 2      # minimum gap between dashes (px)
LLOQ_GAP_MAX      = 80     # maximum gap between dashes (px)
LLOQ_CV_MAX       = 0.65   # max coefficient of variation for dash lengths (relaxed)
LLOQ_COVERAGE_MIN = 0.35   # row must have >= this fraction of plot_w filled

SHAPE_LINE_SPAN   = 0.45   # component spanning > this fraction of image = line
SHAPE_LINE_ASPECT = 8.0    # and aspect ratio above this = axis/grid line
SHAPE_ERRBAR_MIN  = 25     # thin (<=2px) strokes longer than this = error bar
LEGEND_MIN_COMPS  = 6      # min connected components inside candidate legend box
LEGEND_MAX_FRAC   = 0.35   # legend box must be <= this fraction of image area
LEGEND_RIGHT_FRAC = 0.50   # legend is usually in the right half or bottom
LEGEND_BOTTOM_FRAC= 0.50   # legend is usually in the bottom half
PLOT_AREA_PAD     = 4      # extra pad inside axes when defining plot area


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _dark_binary(img_bgr: np.ndarray, thresh: int = 128) -> np.ndarray:
    """Binary mask: True where pixel is dark (foreground ink)."""
    gray = _to_gray(img_bgr)
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return (bw > 0).astype(np.uint8)


def _max_run(px_1d: np.ndarray) -> int:
    """Longest contiguous run of True values in a 1-D boolean array."""
    m = run = 0
    for p in px_1d:
        if p:
            run += 1
            m = max(m, run)
        else:
            run = 0
    return m


def _row_segments(xs: np.ndarray):
    """Split sorted x-coords into contiguous run lengths (gap>2 = new segment)."""
    segs = []
    if len(xs) == 0:
        return segs
    s, e = xs[0], xs[0]
    for x in xs[1:]:
        if x - e <= 2:
            e = x
        else:
            segs.append(e - s + 1)
            s, e = x, x
    segs.append(e - s + 1)
    return segs


# ── Step 1: Axis detection ────────────────────────────────────────────────────

def detect_axes(img_bgr: np.ndarray):
    """
    Detect the x-axis (horizontal rule) and y-axis (vertical rule) of a
    black-and-white chart.

    Strategy (adapted from run_A4_auto_v22._detect_axes):
      - Threshold to get dark pixels.
      - For each row/column, compute the longest continuous dark run.
      - A row with run >= MIN_AXIS_RUN_FRAC * W is a candidate x-axis.
      - A column with run >= MIN_AXIS_RUN_FRAC * H is a candidate y-axis.
      - Among candidates, pick the bottom-most row (x-axis) and left-most
        column (y-axis) -- the standard chart frame convention.

    Returns
    -------
    axis_row : int or None   -- y-pixel of the x-axis
    axis_col : int or None   -- x-pixel of the y-axis
    """
    H, W = img_bgr.shape[:2]
    gray = _to_gray(img_bgr)

    # Two passes: strict (very dark) then relaxed (medium dark)
    for thresh in (100, 160):
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        dark = (bw > 0).astype(np.uint8)

        # --- x-axis candidates (horizontal rows) ---
        row_runs = []
        for r in range(2, H - 2):
            run = _max_run(dark[r, :])
            if run >= MIN_AXIS_RUN_FRAC * W:
                row_runs.append((r, run))

        # --- y-axis candidates (vertical columns) ---
        col_runs = []
        for c in range(2, W - 2):
            run = _max_run(dark[:, c])
            if run >= MIN_AXIS_RUN_FRAC * H:
                col_runs.append((c, run))

        if row_runs or col_runs:
            break

    axis_row = None
    axis_col = None

    if row_runs:
        # x-axis: bottom-most strong candidate (below the data region)
        # Group nearby rows and pick the densest
        row_runs.sort(key=lambda x: x[0])
        # take the bottom-most group
        bottom_r = max(r for r, _ in row_runs)
        # cluster: all rows within 5px of bottom_r
        cluster = [r for r, _ in row_runs if abs(r - bottom_r) <= 5]
        axis_row = int(np.median(cluster))

    if col_runs:
        # y-axis: left-most strong candidate
        col_runs.sort(key=lambda x: x[0])
        left_c = min(c for c, _ in col_runs)
        cluster = [c for c, _ in col_runs if abs(c - left_c) <= 5]
        axis_col = int(np.median(cluster))

    # Fallback: if one axis is missing, try to infer from the other
    # and from dense dark bands at the image margins
    if axis_row is None:
        # look for a dense dark row in the bottom 40% of the image
        bottom_start = int(H * 0.55)
        col_sums = dark[bottom_start:, :].sum(axis=1).astype(float)
        if col_sums.max() > W * 0.3:
            best = int(np.argmax(col_sums)) + bottom_start
            axis_row = best

    if axis_col is None:
        # look for a dense dark column in the left 40% of the image
        right_end = int(W * 0.45)
        row_sums = dark[:, :right_end].sum(axis=0).astype(float)
        if row_sums.max() > H * 0.3:
            best = int(np.argmax(row_sums))
            axis_col = best

    return axis_row, axis_col


# ── Step 2: Plot area ─────────────────────────────────────────────────────────

def detect_plot_area(img_bgr: np.ndarray,
                     axis_row: int | None,
                     axis_col: int | None):
    """
    Define the plotting rectangle from the detected axes.

    Returns (x0, y0, x1, y1) or None if axes are missing.
    The rectangle is the region INSIDE the axes (where data lives).

    Strategy for right boundary:
      - Look for a right-side vertical axis line (if the chart has a frame).
      - Otherwise, estimate from the rightmost dense dark column in the
        data region (excluding margin text columns).
    """
    H, W = img_bgr.shape[:2]
    if axis_row is None and axis_col is None:
        return None

    pad = PLOT_AREA_PAD
    x0 = (axis_col + pad) if axis_col is not None else 0
    y0 = 0
    y1 = (axis_row - pad) if axis_row is not None else H - 1

    # ── Detect right boundary ────────────────────────────────────────────
    # Look for a right-side vertical axis line (strong vertical run)
    gray = _to_gray(img_bgr)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    dark = (bw > 0).astype(np.uint8)

    # Search for right-axis in the right half of the image
    right_col = None
    search_start = int(W * 0.5)
    for c in range(W - 2, search_start, -1):
        run = _max_run(dark[:y1 + 1, c] if axis_row else dark[:, c])
        if run >= MIN_AXIS_RUN_FRAC * (y1 if axis_row else H):
            right_col = c
            break

    if right_col is not None:
        x1 = right_col - pad
    else:
        # Fallback: estimate right boundary from data density.
        # Use the plot rows (y0..y1) and find the rightmost column with
        # meaningful dark pixel density, ignoring the far-right margin.
        # Exclude the rightmost 15% of the image (likely margin text).
        max_x_search = int(W * 0.85)
        data_region = dark[y0:y1 + 1, x0:max_x_search]
        col_sums = data_region.sum(axis=0).astype(float)
        if col_sums.max() > 0:
            # Find the rightmost column with >= 5% of peak density
            threshold = col_sums.max() * 0.05
            dense_cols = np.where(col_sums >= threshold)[0]
            if len(dense_cols) > 0:
                x1 = int(dense_cols.max()) + x0 + pad
            else:
                x1 = max_x_search
        else:
            x1 = max_x_search

    x1 = min(x1, W - 1)

    # Sanity: plot area must be at least 10% of image in each dimension
    if (x1 - x0) < W * 0.10 or (y1 - y0) < H * 0.10:
        return None

    return (int(x0), int(y0), int(x1), int(y1))


# ── Step 3: Legend detection ──────────────────────────────────────────────────

def detect_legend_box(img_bgr: np.ndarray,
                      plot_area: tuple | None,
                      axis_row: int | None):
    """
    Detect a legend panel in a black-and-white chart.

    The legend is typically:
      - Below the x-axis (below axis_row) OR in the lower-right of the plot.
      - A rectangular cluster of small connected components (marker glyphs +
        text labels) that is clearly separated from the main data region.

    Strategy:
      1. Look in the region below the x-axis (if detected).
      2. Find connected components; cluster them spatially.
      3. Accept as legend if the cluster has enough components and is compact.

    Returns (x0, y0, x1, y1) or None.
    """
    H, W = img_bgr.shape[:2]
    gray = _to_gray(img_bgr)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Define search region: below x-axis (if known) or bottom 35% of image
    if axis_row is not None:
        search_y0 = axis_row + AXIS_PAD + 2
    else:
        search_y0 = int(H * 0.65)
    search_y1 = H

    if search_y0 >= search_y1:
        return None

    region = bw[search_y0:search_y1, :]
    n, lbl, stats, centroids = cv2.connectedComponentsWithStats(region, 8)

    if n < LEGEND_MIN_COMPS + 1:
        return None

    # Collect all non-trivial components (area 3..500)
    comps = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 3 <= a <= 600:
            cx = float(centroids[i][0])
            cy = float(centroids[i][1]) + search_y0
            comps.append((cx, cy, stats[i, cv2.CC_STAT_LEFT],
                          stats[i, cv2.CC_STAT_TOP] + search_y0,
                          stats[i, cv2.CC_STAT_WIDTH],
                          stats[i, cv2.CC_STAT_HEIGHT]))

    if len(comps) < LEGEND_MIN_COMPS:
        return None

    # Cluster components by proximity (simple grid-based grouping)
    # Use a sliding window: find the densest rectangular cluster
    xs_arr = np.array([c[0] for c in comps])
    ys_arr = np.array([c[1] for c in comps])

    # Exclude components that are likely to be in the right-side margin
    # (e.g. patent text running vertically on the right side of the image).
    # Heuristic: if a component is in the rightmost 15% of the image AND
    # below the plot area, it is likely margin text, not legend.
    right_margin_x = W * 0.85
    comps_filtered = [c for c in comps
                      if not (c[0] > right_margin_x)]
    if len(comps_filtered) >= LEGEND_MIN_COMPS:
        comps = comps_filtered

    # Bounding box of all legend-region components
    bx0 = int(min(c[2] for c in comps))
    by0 = int(min(c[3] for c in comps))
    bx1 = int(max(c[2] + c[4] for c in comps))
    by1 = int(max(c[3] + c[5] for c in comps))

    # Sanity checks
    box_w = bx1 - bx0
    box_h = by1 - by0
    box_area_frac = (box_w * box_h) / float(H * W)

    if box_area_frac > LEGEND_MAX_FRAC:
        # Too large -- probably the whole bottom margin, not a legend box
        # Try to tighten: find the densest sub-region
        # Use row/col density to find the actual legend extent
        sub = bw[by0:by1+1, bx0:bx1+1]
        if sub.size == 0:
            return None
        row_d = sub.sum(axis=1).astype(float)
        col_d = sub.sum(axis=0).astype(float)
        # Keep rows/cols with at least 5% of max density
        thr_r = max(3, row_d.max() * 0.05)
        thr_c = max(3, col_d.max() * 0.05)
        dense_rows = np.where(row_d >= thr_r)[0]
        dense_cols = np.where(col_d >= thr_c)[0]
        if len(dense_rows) == 0 or len(dense_cols) == 0:
            return None
        by0 = by0 + int(dense_rows.min())
        by1 = by0 + int(dense_rows.max()) + 1
        bx0 = bx0 + int(dense_cols.min())
        bx1 = bx0 + int(dense_cols.max()) + 1
        box_area_frac = ((bx1-bx0)*(by1-by0)) / float(H*W)
        if box_area_frac > LEGEND_MAX_FRAC:
            return None

    # Add a small padding
    pad = 6
    return (max(0, bx0 - pad), max(0, by0 - pad),
            min(W - 1, bx1 + pad), min(H - 1, by1 + pad))


# ── Step 4: LLOQ line detection ───────────────────────────────────────────────

def detect_lloq_line(binary_mask: np.ndarray,
                     plot_area: tuple | None,
                     exclude_bottom_frac: float = 0.10):
    """
    Detect a horizontal dashed reference line (e.g. LLOQ = ...) inside the
    plot area.

    Strategy (two-pass):
      Pass A: strict dashed-line detection (segment length + gap CV check).
      Pass B: relaxed coverage-based detection (row has >= LLOQ_COVERAGE_MIN
              filled pixels AND is_dashed with at least LLOQ_DASH_MIN gaps).

    Parameters
    ----------
    binary_mask         : uint8 mask (1=dark, 0=white)
    plot_area           : (x0, y0, x1, y1) or None
    exclude_bottom_frac : fraction of plot height to exclude at the bottom
                          (avoids confusing the x-axis line with LLOQ)

    Returns row index (int, global image coordinates) or None.
    """
    if plot_area is None:
        H, W = binary_mask.shape[:2]
        px0, py0, px1, py1 = 0, 0, W - 1, H - 1
    else:
        px0, py0, px1, py1 = plot_area

    plot_w = px1 - px0
    plot_h = py1 - py0
    if plot_w < 20 or plot_h < 20:
        return None

    # Exclude bottom fraction (x-axis region)
    y_limit = py1 - int(plot_h * exclude_bottom_frac)

    # ── Pass A: strict segment-based detection ────────────────────────────────
    for y in range(py0, y_limit + 1):
        row = binary_mask[y, px0:px1 + 1]
        if row.sum() == 0:
            continue
        xs = np.where(row > 0)[0]
        x_span = int(xs.max()) - int(xs.min()) + 1
        if x_span < plot_w * LLOQ_SPAN_FRAC:
            continue
        seg_lens = np.array(_row_segments(xs), dtype=float)
        if len(seg_lens) < LLOQ_DASH_MIN:
            continue
        if seg_lens.max() > LLOQ_SEG_MAX_LEN:
            continue
        if seg_lens.min() < LLOQ_SEG_MIN_LEN:
            continue
        # Segment lengths must be similar (low CV)
        if seg_lens.mean() > 0 and seg_lens.std() / seg_lens.mean() > LLOQ_CV_MAX:
            continue
        # Gaps between segments
        seg_starts, seg_ends = [], []
        s, e = xs[0], xs[0]
        for x in xs[1:]:
            if x - e <= 2:
                e = x
            else:
                seg_ends.append(e)
                seg_starts.append(x)
                s, e = x, x
        seg_ends.append(e)
        if len(seg_starts) >= 2:
            gaps = np.array([seg_starts[i] - seg_ends[i - 1]
                             for i in range(1, len(seg_starts))], dtype=float)
            if gaps.min() < LLOQ_GAP_MIN or gaps.max() > LLOQ_GAP_MAX:
                continue
            if gaps.mean() > 0 and gaps.std() / gaps.mean() > LLOQ_CV_MAX:
                continue
        else:
            continue
        return int(y)

    # ── Pass B: relaxed coverage + is_dashed check ────────────────────────────
    # Collect candidate rows by coverage, then pick the most dashed one.
    roi = binary_mask[py0:y_limit + 1, px0:px1 + 1]
    row_sums = roi.sum(axis=1).astype(float)
    h_thresh = plot_w * LLOQ_COVERAGE_MIN
    cand_rows = np.where(row_sums > h_thresh)[0]

    def _is_dashed_relaxed(row_arr, min_gaps=3, min_gap_len=4):
        in_gap = False; gaps = 0; gap_len = 0
        for px in row_arr:
            if px == 0:
                if not in_gap:
                    in_gap = True; gap_len = 1
                else:
                    gap_len += 1
            else:
                if in_gap and gap_len >= min_gap_len:
                    gaps += 1
                in_gap = False; gap_len = 0
        return gaps >= min_gaps

    best_row = None
    best_cov = 0.0
    for r in cand_rows:
        row_arr = roi[r]
        if _is_dashed_relaxed(row_arr):
            cov = row_sums[r] / plot_w
            if cov > best_cov:
                best_cov = cov
                best_row = r

    if best_row is not None:
        return int(best_row + py0)

    return None


# ── Axis-line removal with marker preservation ──────────────────────────────────

def _remove_axis_lines_preserve_markers(
        mask,
        plot_area,
        axis_row,
        axis_col,
        H, W,
        border_px=6,
        marker_min_area=15,
        marker_max_aspect=3.5,
):
    # Remove x/y axis lines inside plot area while preserving data markers.
    # Strategy: row/col erase then restore compact blobs (markers).
    if plot_area is None:
        return mask

    m = mask.copy().astype('uint8')
    orig = mask.copy().astype('uint8')
    px0, py0, px1, py1 = plot_area
    px0 = max(0, px0); py0 = max(0, py0)
    px1 = min(W - 1, px1); py1 = min(H - 1, py1)
    plot_w = max(1, px1 - px0)
    plot_h = max(1, py1 - py0)

    erased_rows = set()
    erased_cols = set()

    def _longest_continuous_run(arr):
        """Return the length of the longest continuous run of non-zero values."""
        max_run = cur = 0
        for v in arr:
            if v > 0:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        return max_run

    # ── Find and erase x-axis row ─────────────────────────────────────────────
    # Search the bottom 40% of plot area.
    # Criterion: CONTINUOUS run >= 35% of plot_w  OR  total pixels >= 50% of plot_w
    # (x-axis may have tick gaps, so total pixel count is also checked)
    search_h = max(border_px, int(plot_h * 0.40))
    band_y0 = max(py0, py1 - search_h)
    band_y1 = py1
    best_score_x, best_run, best_total, best_row = 0.0, 0, 0, None
    for r in range(band_y0, band_y1 + 1):
        row_arr = m[r, px0:px1 + 1]
        run   = _longest_continuous_run(row_arr)
        total = int(row_arr.sum())
        # Combined score: max of run-fraction and total-fraction
        score = max(run / plot_w, total / plot_w * 0.7)
        if score > best_score_x:
            best_score_x, best_run, best_total, best_row = score, run, total, r
    # Threshold: continuous run >= 30% OR total >= 35% of plot_w
    # (x-axis with tick gaps may have low run but decent total pixel count)
    x_erase = (best_row is not None and
               (best_run >= plot_w * 0.30 or best_total >= plot_w * 0.35))
    if x_erase:
        # Erase a wider band (±3px) around the x-axis row WITHOUT restore.
        # x-axis is a border line; data points do not lie ON the axis line itself.
        # We erase the entire bottom strip from best_row-3 to py1 to catch
        # all tick marks and axis fragments.
        erase_y0 = max(py0, best_row - 3)
        erase_y1 = min(H - 1, py1)
        m[erase_y0:erase_y1 + 1, px0:px1 + 1] = 0
        # Record only the main axis row for restore logic (not the whole strip)
        for r in range(erase_y0, erase_y1 + 1):
            erased_rows.add(r)

    # ── Find and erase left y-axis col ──────────────────────────────────────────
    search_w = max(border_px, int(plot_w * 0.20))
    band_x0 = px0
    band_x1 = min(px1, px0 + search_w)
    best_score, best_col = 0, None
    for c in range(band_x0, band_x1 + 1):
        col_arr = m[py0:py1 + 1, c]
        longest = _longest_continuous_run(col_arr)
        total   = int(col_arr.sum())
        score = max(longest / plot_h, total / plot_h * 0.8)
        if score > best_score:
            best_score, best_col = score, c
    if best_col is not None and best_score >= 0.40:
        for c in range(max(0, best_col - 1), min(W, best_col + 2)):
            m[py0:py1 + 1, c] = 0
            erased_cols.add(c)

    # ── Find and erase right y-axis col (if present) ──────────────────────────
    search_w_r = max(border_px, int(plot_w * 0.10))
    band_x0_r = max(px0, px1 - search_w_r)
    band_x1_r = px1
    best_score_r, best_col_r = 0, None
    for c in range(band_x0_r, band_x1_r + 1):
        col_arr = m[py0:py1 + 1, c]
        longest = _longest_continuous_run(col_arr)
        total   = int(col_arr.sum())
        score = max(longest / plot_h, total / plot_h * 0.8)
        if score > best_score_r:
            best_score_r, best_col_r = score, c
    # Right y-axis must have a long CONTINUOUS run (not just scattered pixels like error bars)
    if best_col_r is not None and best_score_r >= 0.65:
        col_arr_r = m[py0:py1 + 1, best_col_r]
        longest_r = _longest_continuous_run(col_arr_r)
        if longest_r >= plot_h * 0.55:  # must span >55% continuously
            for c in range(max(0, best_col_r - 1), min(W, best_col_r + 2)):
                m[py0:py1 + 1, c] = 0
                erased_cols.add(c)

    if not erased_rows and not erased_cols:
        return m

    # Restore marker fragments cut by the erase.
    # Strategy: scan the ORIGINAL mask in a wide band around the erased rows/cols.
    # For each connected blob in the original, check if it is compact (marker-like).
    # If compact AND its centroid does NOT lie on an erased row/col, restore it.
    # This prevents restoring the axis line itself while recovering cut markers.
    def _restore_markers(bx0, by0, bx1, by1):
        band_orig = orig[by0:by1 + 1, bx0:bx1 + 1].copy()
        n, lbl, stats, centroids = cv2.connectedComponentsWithStats(band_orig, 8)
        # Max marker size: no larger than ~3x estimated marker diameter
        max_marker_area = int(plot_w * plot_h * 0.005)  # 0.5% of plot area
        for i in range(1, n):
            a  = int(stats[i, cv2.CC_STAT_AREA])
            bw_s = int(stats[i, cv2.CC_STAT_WIDTH])
            bh_s = int(stats[i, cv2.CC_STAT_HEIGHT])
            if a < marker_min_area:
                continue
            # Skip very large blobs (axis line + connected curve)
            if a > max_marker_area:
                continue
            long_side  = max(bw_s, bh_s)
            short_side = max(1, min(bw_s, bh_s))
            aspect     = long_side / max(short_side, 1)
            if aspect > marker_max_aspect:
                continue
            # Get pixel coordinates of this blob in image space
            ys, xs = np.where(lbl == i)
            ys_img = ys + by0
            xs_img = xs + bx0
            # Centroid in image space
            cy_img = float(centroids[i][1]) + by0
            cx_img = float(centroids[i][0]) + bx0
            # If centroid falls ON an erased row/col, this is an axis fragment
            centroid_on_erased = (
                int(round(cy_img)) in erased_rows or
                int(round(cx_img)) in erased_cols
            )
            if centroid_on_erased:
                continue
            # Count how many pixels fall on erased rows or cols
            on_erased = np.sum(
                np.isin(ys_img, list(erased_rows)) |
                np.isin(xs_img, list(erased_cols))
            )
            # If > 40% of the blob's pixels are on erased lines, skip
            if on_erased / max(1, a) > 0.40:
                continue
            m[ys_img, xs_img] = orig[ys_img, xs_img]

    if erased_rows:
        min_er = max(py0, min(erased_rows) - 20)
        max_er = min(py1, max(erased_rows) + 20)
        _restore_markers(px0, min_er, px1, max_er)

    if erased_cols:
        min_ec = max(px0, min(erased_cols) - 20)
        max_ec = min(px1, max(erased_cols) + 20)
        _restore_markers(min_ec, py0, max_ec, py1)

    return m


# ── LLOQ line removal with marker preservation ───────────────────────────────

def _remove_lloq_line_preserve_markers(
        mask: np.ndarray,
        lloq_row: int,
        plot_area: tuple,
        H: int, W: int,
        band: int = None,
        marker_min_area: int = 15,
        marker_max_aspect: float = 3.5,
) -> np.ndarray:
    """
    Remove the LLOQ dashed line (and its nearby text label) from the binary
    mask, while preserving data markers that happen to overlap with the line.

    Strategy:
      1. Erase a band of ±band rows around lloq_row within the plot area.
      2. Restore compact blobs from the original mask whose pixels are NOT
         predominantly on the erased rows (same logic as axis removal).
    """
    if band is None:
        band = LLOQ_Y_BAND

    m    = mask.copy().astype(np.uint8)
    orig = mask.copy().astype(np.uint8)

    px0, py0, px1, py1 = plot_area
    px0 = max(0, px0); py0 = max(0, py0)
    px1 = min(W - 1, px1); py1 = min(H - 1, py1)

    y_lo = max(py0, lloq_row - band)
    y_hi = min(py1, lloq_row + band + 1)

    erased_rows = set(range(y_lo, y_hi))

    # Erase band inside plot area only
    m[y_lo:y_hi, px0:px1 + 1] = 0

    # Restore compact marker blobs
    band_orig = orig[y_lo:y_hi + 1, px0:px1 + 1].copy()
    n, lbl, stats, centroids = cv2.connectedComponentsWithStats(band_orig, 8)
    for i in range(1, n):
        a    = int(stats[i, cv2.CC_STAT_AREA])
        bw_s = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_s = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < marker_min_area:
            continue
        long_side  = max(bw_s, bh_s)
        short_side = max(1, min(bw_s, bh_s))
        aspect     = long_side / max(short_side, 1)
        if aspect > marker_max_aspect:
            continue
        ys, xs = np.where(lbl == i)
        ys_img = ys + y_lo
        xs_img = xs + px0
        on_erased = np.sum(np.isin(ys_img, list(erased_rows)))
        if on_erased / max(1, a) > 0.50:
            continue
        m[ys_img, xs_img] = orig[ys_img, xs_img]

    return m


# ── Step 5: Noise removal from binary mask ────────────────────────────────────

def remove_noise_from_binary(binary_mask: np.ndarray,
                              img_bgr: np.ndarray,
                              axis_row: int | None,
                              axis_col: int | None,
                              plot_area: tuple | None,
                              legend_box: tuple | None,
                              lloq_row: int | None) -> np.ndarray:
    """
    Remove structural noise from a binary (dark-on-white) mask:

    Pass 0: Remove axis lines and the region outside the plot area.
    Pass 1: Remove long thin strokes (axis rules, grid lines, error bars)
            by connected-component shape analysis.
    Pass 2: Remove LLOQ dashed reference line and its inline text label.
    Pass 3: Remove legend box contents (marker glyphs + text labels).
    Pass 4: Remove text characters (small clustered components outside
            the plot area -- axis labels, title, annotations).
    Pass 5: Remove axis-adjacent tick marks and small noise fragments.

    The mask is modified in-place and returned.
    """
    H, W = binary_mask.shape[:2]
    out = binary_mask.copy().astype(np.uint8)

    # ── Pass 0: blank everything outside the plot area ────────────────────────
    if plot_area is not None:
        px0, py0, px1, py1 = plot_area
        outside = np.ones((H, W), dtype=np.uint8)
        outside[py0:py1 + 1, px0:px1 + 1] = 0
        out[outside > 0] = 0

    # ── Pass 1: shape-based axis/grid/error-bar removal ───────────────────────
    out = _drop_long_thin_strokes(out, plot_area, H, W)

    # ── Pass 1b: remove axis border lines inside plot area (preserve markers) ──
    out = _remove_axis_lines_preserve_markers(
        out, plot_area, axis_row, axis_col, H, W
    )

    # ── Pass 2: LLOQ dashed line removal (preserve markers) ──────────────────
    if lloq_row is not None and plot_area is not None:
        out = _remove_lloq_line_preserve_markers(
            out, lloq_row, plot_area, H, W
        )

    # ── Pass 3: legend box removal ────────────────────────────────────────────
    if legend_box is not None:
        lx0, ly0, lx1, ly1 = legend_box
        box_area_frac = ((lx1 - lx0 + 1) * (ly1 - ly0 + 1)) / float(H * W)
        if box_area_frac < LEGEND_MAX_FRAC:
            # Remove all components whose bounding box lies inside the legend
            n, lbl, stats, cen = cv2.connectedComponentsWithStats(out, 8)
            for i in range(1, n):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                inside = (x >= lx0 - 3 and y >= ly0 - 3 and
                          x + w <= lx1 + 3 and y + h <= ly1 + 3)
                if inside:
                    out[lbl == i] = 0

    # ── Pass 4: text character removal (outside plot area) ───────────────────
    out = _remove_text_components(out, plot_area, H, W)

    # ── Pass 4b: aggressive margin text removal ───────────────────────────────
    # Remove ALL components outside the plot area that are small (text-sized).
    # This catches rotated patent text, axis labels, title, etc.
    out = _remove_all_margin_components(out, plot_area, H, W)

    # ── Pass 5: axis-adjacent tick / noise removal ────────────────────────────
    out = _remove_axis_ticks(out, axis_row, axis_col, H, W)

    return out


def _drop_long_thin_strokes(mask: np.ndarray,
                             plot_area: tuple | None,
                             H: int, W: int) -> np.ndarray:
    """Remove connected components that are long, thin, straight rules
    (axis lines, grid lines, error bars). Curve/marker blobs are compact
    and are left untouched."""
    m = mask.copy()
    if m.sum() == 0:
        return m
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ref_w = (plot_area[2] - plot_area[0]) if plot_area else W
    ref_h = (plot_area[3] - plot_area[1]) if plot_area else H
    for i in range(1, n):
        a    = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        if a < 3:
            continue
        long_side  = max(w, h)
        short_side = max(1, min(w, h))
        aspect     = long_side / short_side
        # Spans a large fraction of the plot in one direction
        spans = (w >= ref_w * SHAPE_LINE_SPAN) or (h >= ref_h * SHAPE_LINE_SPAN)
        # Drop: extremely elongated AND spans most of the plot
        if aspect >= SHAPE_LINE_ASPECT and short_side <= 6 and spans:
            m[lbl == i] = 0
        # Drop thin vertical/horizontal error-bar strokes
        elif short_side <= 2 and long_side > SHAPE_ERRBAR_MIN:
            m[lbl == i] = 0
    return m


def _remove_text_components(mask: np.ndarray,
                             plot_area: tuple | None,
                             H: int, W: int) -> np.ndarray:
    """Remove small clustered components that look like text characters.
    Text characters are:
      - Small area (< CHAR_MAX_AREA)
      - Aspect ratio 0.2 .. 4.0 (not extreme vertical ticks)
      - Clustered near other similar components (words have multiple chars)
    Only removes components OUTSIDE the plot area to protect data markers.
    """
    m = mask.copy()
    if m.sum() == 0:
        return m

    # Build an "outside plot" mask for restricting text removal
    outside = np.ones((H, W), dtype=bool)
    if plot_area is not None:
        px0, py0, px1, py1 = plot_area
        outside[py0:py1 + 1, px0:px1 + 1] = False

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    small_comps = []
    largest_area = 0
    largest_idx  = -1

    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > largest_area:
            largest_area = a
            largest_idx  = i
        if a < CHAR_MAX_AREA:
            wi = int(stats[i, cv2.CC_STAT_WIDTH])
            hi = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = stats[i, cv2.CC_STAT_LEFT] + wi / 2.0
            cy = stats[i, cv2.CC_STAT_TOP]  + hi / 2.0
            aspect = wi / max(hi, 1)
            if 0.2 <= aspect <= 4.0:
                # Only consider components outside the plot area
                if outside[int(cy), int(cx)]:
                    small_comps.append((i, cx, cy, a))

    if len(small_comps) < 2:
        return m

    sc_arr = np.array([[cx, cy] for _, cx, cy, _ in small_comps], dtype=float)
    diffs  = sc_arr[:, np.newaxis, :] - sc_arr[np.newaxis, :, :]
    dists  = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    neighbour_count = (dists < TEXT_CLUSTER_R).sum(axis=1)

    for k, (i, cx_i, cy_i, area) in enumerate(small_comps):
        if neighbour_count[k] >= 2 and area < largest_area:
            m[lbl == i] = 0

    return m


def _remove_all_margin_components(mask: np.ndarray,
                                   plot_area: tuple | None,
                                   H: int, W: int) -> np.ndarray:
    """Remove ALL connected components that lie entirely outside the plot area.
    This is a catch-all for axis labels, title text, patent margin text, etc.
    Components that overlap with the plot area are preserved."""
    if plot_area is None:
        return mask
    m = mask.copy()
    if m.sum() == 0:
        return m
    px0, py0, px1, py1 = plot_area
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Component bounding box
        cx1 = cx + cw
        cy1 = cy + ch
        # Check if it overlaps with the plot area
        overlap = not (cx1 < px0 or cx > px1 or cy1 < py0 or cy > py1)
        if not overlap:
            m[lbl == i] = 0
    return m


def _remove_axis_ticks(mask: np.ndarray,
                       axis_row: int | None,
                       axis_col: int | None,
                       H: int, W: int) -> np.ndarray:
    """Remove small thin components adjacent to the detected axes
    (tick marks, axis-line fragments, small noise)."""
    m = mask.copy()
    if m.sum() == 0:
        return m
    axis_rows = [axis_row] if axis_row is not None else []
    axis_cols = [axis_col] if axis_col is not None else []
    n, lbl, stats, cen = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        a   = int(stats[i, cv2.CC_STAT_AREA])
        if a >= TICK_AREA_MAX:
            continue
        wi  = int(stats[i, cv2.CC_STAT_WIDTH])
        hi  = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx_i = float(cen[i][0])
        cy_i = float(cen[i][1])
        near = (any(abs(cy_i - ay) <= TICK_MARGIN for ay in axis_rows) or
                any(abs(cx_i - ac) <= TICK_MARGIN for ac in axis_cols))
        thin = min(wi, hi) <= 3
        if near and thin:
            m[lbl == i] = 0
    return m


# ── Step 6: Convenience wrapper ───────────────────────────────────────────────

# Margin added around user-supplied plot area for preprocessing
# (to include axis lines that sit just outside the dragged rectangle)
AXIS_MARGIN = 15


def preprocess(img_bgr: np.ndarray,
               user_plot_area: tuple | None = None,
               user_legend_box: tuple | None = None,
               verbose: bool = True, *, auto_legend: bool = True) -> dict:
    """
    Run the full preprocessing pipeline on a BGR chart image.

    Parameters
    ----------
    img_bgr          : BGR numpy array (the chart image)
    user_plot_area   : optional (x0,y0,x1,y1) override from GUI drag
                       A margin of AXIS_MARGIN px is added around this box
                       for axis detection/removal, but coordinate conversion
                       always uses the original dragged box.
    user_legend_box  : optional (x0,y0,x1,y1) legend area override from GUI drag
                       If provided, overrides auto-detected legend box.
    verbose          : print progress messages
    auto_legend      : False means an absent user box is explicitly no legend;
                       do not discover or erase a guessed legend in that case.

    Returns
    -------
    dict with keys:
        'axis_row'       : int or None
        'axis_col'       : int or None
        'plot_area'      : (x0,y0,x1,y1) expanded area used for preprocessing
        'user_plot_area' : (x0,y0,x1,y1) original dragged area (coord conversion)
        'legend_box'     : (x0,y0,x1,y1) or None
        'lloq_row'       : int or None
        'clean_fn'       : callable(binary_mask) -> cleaned_mask
        'debug_img'      : annotated BGR image
    """
    H, W = img_bgr.shape[:2]

    # 1. Axes
    axis_row, axis_col = detect_axes(img_bgr)
    if verbose:
        print(f"[preprocess] axis_row={axis_row}, axis_col={axis_col}")

    # 2. Plot area
    # If user supplied a drag box, expand it by AXIS_MARGIN for preprocessing
    # so that axis lines just outside the drag box are also captured.
    user_orig = user_plot_area  # original drag box (for coord conversion)
    if user_plot_area is not None:
        ux0, uy0, ux1, uy1 = user_plot_area
        plot_area = (
            max(0, ux0 - AXIS_MARGIN),
            max(0, uy0 - AXIS_MARGIN),
            min(W - 1, ux1 + AXIS_MARGIN),
            min(H - 1, uy1 + AXIS_MARGIN),
        )
    else:
        plot_area = detect_plot_area(img_bgr, axis_row, axis_col)
        user_orig = plot_area
    if verbose:
        print(f"[preprocess] user_plot_area={user_orig}")
        print(f"[preprocess] expanded plot_area={plot_area}")

    # 3. Legend
    if user_legend_box is not None:
        legend_box = user_legend_box
        if verbose:
            print(f"[preprocess] legend_box (user)={legend_box}")
    elif not auto_legend:
        legend_box = None
        if verbose:
            print('[preprocess] legend_box=None (explicit no legend)')
    else:
        legend_box = detect_legend_box(img_bgr, plot_area, axis_row)
        if verbose:
            print(f"[preprocess] legend_box (auto)={legend_box}")

    # 4. LLOQ line (needs a binary mask of the plot area)
    gray = _to_gray(img_bgr)
    _, bw_full = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    bw_full = (bw_full > 0).astype(np.uint8)
    lloq_row = detect_lloq_line(bw_full, plot_area)
    if verbose:
        print(f"[preprocess] lloq_row={lloq_row}")

    # 5. Build clean_fn closure
    def clean_fn(binary_mask: np.ndarray) -> np.ndarray:
        return remove_noise_from_binary(
            binary_mask, img_bgr,
            axis_row, axis_col, plot_area,
            legend_box, lloq_row
        )

    # 6. Debug visualisation
    debug_img = img_bgr.copy()
    if axis_row is not None:
        cv2.line(debug_img, (0, axis_row), (W - 1, axis_row), (0, 0, 255), 2)
    if axis_col is not None:
        cv2.line(debug_img, (axis_col, 0), (axis_col, H - 1), (255, 0, 0), 2)
    if plot_area is not None:
        px0, py0, px1, py1 = plot_area
        cv2.rectangle(debug_img, (px0, py0), (px1, py1), (0, 200, 0), 2)
    if legend_box is not None:
        lx0, ly0, lx1, ly1 = legend_box
        cv2.rectangle(debug_img, (lx0, ly0), (lx1, ly1), (200, 0, 200), 2)
    if lloq_row is not None:
        cv2.line(debug_img, (0, lloq_row), (W - 1, lloq_row), (0, 200, 200), 2)

    return {
        'axis_row':       axis_row,
        'axis_col':       axis_col,
        'plot_area':      plot_area,      # expanded (for noise removal)
        'user_plot_area': user_orig,      # original drag box (for coord conversion)
        'legend_box':     legend_box,
        'lloq_row':       lloq_row,
        'clean_fn':       clean_fn,
        'debug_img':      debug_img,
    }


# ── Error-bar detection (Function 1) ───────────────────────────────────────

def detect_has_errorbars(
        img_bgr: np.ndarray,
        prep_info: dict | None = None,
        min_stem_len: float = 10.0,
        min_stems: int = 1,
        vert_angle_tol: float = 20.0,
        tcap_aspect_min: float = 3.0,
        tcap_scan_rows: int = 6,
        tcap_min_width: int = 5,
        segments=None,
) -> bool:
    """
    Automatically detect whether a chart image contains error bars.

    Strategy
    --------
    Two independent evidence channels are combined with OR logic:

    Channel A — Vertical stem evidence
        Run the segment detector on the (optionally preprocessed) image and
        look for vertical segments (angle ≈ 90°) whose length exceeds
        ``min_stem_len``.  If ``min_stems`` or more such segments are found,
        Channel A fires.

    Channel B — T-cap evidence
        For each vertical-segment candidate (Channel A), scan a narrow band of
        rows around each endpoint.  If the horizontal run of dark pixels in
        that band is at least ``tcap_min_width`` wide and wider than
        ``tcap_aspect_min`` x the stem width, the endpoint is classified as a
        T-cap, providing strong evidence of an error bar.

    Parameters
    ----------
    img_bgr : np.ndarray
        Full BGR chart image.
    prep_info : dict or None
        Output of ``preprocess()``.  If provided, the cleaned binary mask is
        used instead of the raw binarised image, which suppresses axis lines
        and text that could otherwise be mistaken for stems.
    min_stem_len : float
        Minimum length (px) for a vertical segment to count as a stem candidate.
    min_stems : int
        Minimum number of stem candidates required for Channel A to fire.
    vert_angle_tol : float
        Angular tolerance (degrees from 90°) for classifying a segment as
        vertical.
    tcap_aspect_min : float
        Minimum ratio of T-cap horizontal width to stem width to confirm a
        T-cap (Channel B).
    tcap_scan_rows : int
        Number of rows above/below each stem endpoint to scan for T-cap pixels.
    tcap_min_width : int
        Absolute minimum T-cap width (px) regardless of aspect ratio.
    segments : sequence or None
        Optional already-computed segments for the same image/preprocessing.
        An empty sequence means no segments; None retains the legacy detector
        call. The supplied sequence is read only and never modified.

    Returns
    -------
    bool
        True if error bars are detected, False otherwise.
    """
    # ── Build binary mask ────────────────────────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    bw = (bw > 0).astype(np.uint8)
    if prep_info is not None:
        clean_fn = prep_info.get('clean_fn', None)
        if clean_fn is not None:
            bw = clean_fn(bw)

    # ── Run segment detector ─────────────────────────────────────────────────
    # Import lazily to avoid circular imports.
    if segments is None:
        try:
            import sys as _sys
            import importlib.util as _ilu
            _src_dir = str(Path(__file__).parent.resolve())
            if _src_dir not in _sys.path:
                _sys.path.insert(0, _src_dir)
            if 'segment_detector' in _sys.modules:
                _seg_mod = _sys.modules['segment_detector']
            else:
                _seg_path = Path(__file__).parent / '3_segment_detection_v2.py'
                _spec = _ilu.spec_from_file_location('segment_detector', str(_seg_path))
                _seg_mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_seg_mod)
                _sys.modules['segment_detector'] = _seg_mod
            _seg_result = _seg_mod.detect_debug(img_bgr, prep_info=prep_info)
            segments = _seg_result.get('segments', [])
        except Exception:
            return False

    # ── Channel A: vertical stem candidates ──────────────────────────────────
    def _seg_len(s):
        x1, y1, x2, y2 = s
        return math.hypot(x2 - x1, y2 - y1)

    def _seg_angle(s):
        x1, y1, x2, y2 = s
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        return math.degrees(math.atan2(dy, dx + 1e-9))

    vert_candidates = [
        s for s in segments
        if abs(_seg_angle(s) - 90.0) <= vert_angle_tol
        and _seg_len(s) >= min_stem_len
    ]

    if len(vert_candidates) < min_stems:
        return False

    # Channel A fired.  Run Channel B for T-cap confirmation.
    # ── Channel B: T-cap scan ──────────────────────────────────────────────
    H_img, W_img = bw.shape[:2]
    for seg in vert_candidates:
        x1, y1, x2, y2 = seg
        cx = int(round((x1 + x2) / 2))
        stem_w = max(1, int(round(abs(x2 - x1))) + 1)
        y_top = int(round(min(y1, y2)))
        y_bot = int(round(max(y1, y2)))

        for ep_y in (y_top, y_bot):
            r0 = max(0, ep_y - tcap_scan_rows)
            r1 = min(H_img, ep_y + tcap_scan_rows + 1)
            band = bw[r0:r1, :]
            col_sums = band.sum(axis=0)
            dark_cols = np.where(col_sums > 0)[0]
            if dark_cols.size == 0:
                continue
            # Widest contiguous run of dark columns
            best_run = 0
            run = 1
            for i in range(1, len(dark_cols)):
                if dark_cols[i] == dark_cols[i - 1] + 1:
                    run += 1
                else:
                    run = 1
                best_run = max(best_run, run)
            if best_run == 0 and dark_cols.size > 0:
                best_run = 1
            if (best_run >= tcap_min_width and
                    best_run >= tcap_aspect_min * stem_w):
                return True   # T-cap confirmed → definite error bar

    # Channel A fired but no T-cap found — still report True
    return True


# ── Error-bar stem + T-cap removal (Function 2) ─────────────────────────

def remove_errorbars_from_mask(binary_mask, vert_segs, all_segs,
                               stem_half_w=2, tcap_half_h=4,
                               marker_guard=16,
                               marker_connect_thresh=8):
    """
    Remove error-bar stems and T-caps from a binary mask while preserving
    marker pixels.

    Strategy
    --------
    For each vertical segment (angle ≈ 90°):
      1. Classify each end (TOP / BOT) as MARKER or T-CAP by measuring the
         minimum distance from that endpoint to any non-vertical segment.
         - dist < marker_connect_thresh  →  MARKER end (connected to curve)
         - dist >= marker_connect_thresh →  T-CAP end  (free end)
      2. Erase the stem pixels (±stem_half_w columns around cx, full y range).
      3. Erase the T-cap pixels (±tcap_half_h rows around the T-cap end row,
         full x range within the stem column band extended by P//2).
      4. Restore marker-end pixels by copying from the original mask.

    Parameters
    ----------
    binary_mask : np.ndarray (H, W) uint8
        Cleaned binary mask (1 = dark pixel).
    vert_segs : list of dict
        Vertical segments, each with keys 'x1','y1','x2','y2' (pixel coords).
    all_segs : list of dict
        All segments (vertical + horizontal/diagonal), same key format.
    stem_half_w : int
        Half-width (columns) to erase around the stem cx.
    tcap_half_h : int
        Half-height (rows) to erase around the T-cap end.
    marker_connect_thresh : int
        Distance threshold (px) to classify an end as MARKER vs T-CAP.

    Returns
    -------
    cleaned : np.ndarray (H, W) uint8
        Modified mask with stems and T-caps erased.
    info : list of dict
        Per-stem classification info for debugging.
    """
    import numpy as np

    mask = binary_mask.copy()
    orig = binary_mask.copy()
    info = []

    import math as _math

    def _to_dict(s):
        """Accept either a dict or a (x1,y1,x2,y2) tuple."""
        if isinstance(s, dict):
            return s
        x1, y1, x2, y2 = s
        dx = abs(x2 - x1); dy = abs(y2 - y1)
        angle = _math.degrees(_math.atan2(dy, dx + 1e-9))
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'angle': angle}

    vert_segs_d = [_to_dict(s) for s in vert_segs]
    all_segs_d  = [_to_dict(s) for s in all_segs]

    # Non-vertical segments (curves / diagonals) — used for distance check
    non_vert = [s for s in all_segs_d
                if abs(s.get('angle', 0) - 90) > 20]

    def _endpoint_dist(px, py):
        """
        Min distance from (px,py) to any non-vertical segment.
        Uses two metrics and takes the minimum:
          1. Distance to nearest endpoint of non-vertical segs.
          2. Perpendicular distance from (px,py) to each non-vertical seg line,
             but only if px falls within the segment's x-range (i.e. the stem
             passes through the curve at that x position).
        """
        best = float('inf')
        for s in non_vert:
            # metric 1: endpoint distances
            for ex, ey in [(s['x1'], s['y1']), (s['x2'], s['y2'])]:
                d = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
                if d < best:
                    best = d
            # metric 2: y-interpolation at px (if px within segment x-range)
            sx1, sy1, sx2, sy2 = s['x1'], s['y1'], s['x2'], s['y2']
            xlo, xhi = min(sx1, sx2), max(sx1, sx2)
            if xlo - 5 <= px <= xhi + 5 and abs(sx2 - sx1) > 1:
                t = (px - sx1) / (sx2 - sx1)
                interp_y = sy1 + t * (sy2 - sy1)
                d_y = abs(py - interp_y)
                if d_y < best:
                    best = d_y
        return best

    for seg in vert_segs_d:
        cx  = int(round((seg['x1'] + seg['x2']) / 2))
        y_a = int(round(seg['y1']))
        y_b = int(round(seg['y2']))
        y_top = min(y_a, y_b)
        y_bot = max(y_a, y_b)

        d_top = _endpoint_dist(cx, y_top)
        d_bot = _endpoint_dist(cx, y_bot)

        top_is_marker = d_top < marker_connect_thresh
        bot_is_marker = d_bot < marker_connect_thresh

        # Erase stem columns
        x0s = max(0, cx - stem_half_w)
        x1s = min(mask.shape[1], cx + stem_half_w + 1)
        mask[y_top:y_bot + 1, x0s:x1s] = 0

        # Erase T-cap rows (free ends)
        if not top_is_marker:
            r0 = max(0, y_top - tcap_half_h)
            r1 = min(mask.shape[0], y_top + tcap_half_h + 1)
            mask[r0:r1, x0s:x1s] = 0
        if not bot_is_marker:
            r0 = max(0, y_bot - tcap_half_h)
            r1 = min(mask.shape[0], y_bot + tcap_half_h + 1)
            mask[r0:r1, x0s:x1s] = 0

        # Restore marker-end pixels from original
        # Use marker_guard (larger than tcap_half_h) to protect full marker area
        if top_is_marker:
            r0 = max(0, y_top - marker_guard)
            r1 = min(mask.shape[0], y_top + marker_guard + 1)
            region = orig[r0:r1, x0s:x1s]
            mask[r0:r1, x0s:x1s] = np.maximum(mask[r0:r1, x0s:x1s], region)
        if bot_is_marker:
            r0 = max(0, y_bot - marker_guard)
            r1 = min(mask.shape[0], y_bot + marker_guard + 1)
            region = orig[r0:r1, x0s:x1s]
            mask[r0:r1, x0s:x1s] = np.maximum(mask[r0:r1, x0s:x1s], region)

        info.append({
            'cx': cx, 'y_top': y_top, 'y_bot': y_bot,
            'd_top': round(d_top, 1), 'd_bot': round(d_bot, 1),
            'top_is_marker': top_is_marker, 'bot_is_marker': bot_is_marker,
        })

    return mask, info


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: python chart_preprocessing.py <image_path> [out_dir]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(img_path)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: cannot read {img_path}")
        sys.exit(1)

    info = preprocess(img, verbose=True)

    # Save debug image
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{stem}_preprocess_debug.png"),
                info['debug_img'])

    # Show cleaned binary mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    bw = (bw > 0).astype(np.uint8)
    cleaned = info['clean_fn'](bw)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_cleaned_mask.png"),
                (cleaned * 255).astype(np.uint8))

    print(f"\nSaved debug image and cleaned mask to: {out_dir}")


# ── Optimal scale estimation from legend marker size ─────────────────────────

def estimate_optimal_scale(
        img_bgr: np.ndarray,
        legend_box: tuple,
        vit_window_px: int = 19,
        target_ratio: float = 0.667,
        min_scale: float = 0.25,
        max_scale: float = 4.0,
        snap_values: tuple = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
        min_marker_area: int = 4,
        max_marker_area: int = 800,
        min_aspect: float = 0.3,
        max_aspect: float = 3.5,
        verbose: bool = False,
) -> tuple[float, dict]:
    """
    Estimate the optimal image scale factor by measuring legend marker size.

    The function crops the legend region, binarises it, and finds connected
    components that look like marker glyphs (small, roughly square blobs).
    The median diameter of those glyphs is compared to ``vit_window_px`` (the
    ViT sliding-window side length P).  The scale is chosen so that the
    rendered marker diameter equals ``target_ratio * vit_window_px``.

    Parameters
    ----------
    img_bgr : np.ndarray
        Full BGR chart image.
    legend_box : tuple
        (x0, y0, x1, y1) of the legend region in image pixels.
    vit_window_px : int
        ViT patch / sliding-window side length P (default 19).
    target_ratio : float
        Desired (marker_diameter / vit_window_px) after scaling.
        1.0 means the marker fills exactly one ViT window.
        Values slightly below 1 (e.g. 0.85) give a small margin.
    min_scale / max_scale : float
        Hard clamp on the returned scale.
    snap_values : tuple
        Candidate scale values to snap to (nearest wins).
    min_marker_area / max_marker_area : int
        Connected-component area range (px²) to accept as a marker glyph.
    min_aspect / max_aspect : float
        Aspect-ratio (w/h) range to accept as a marker glyph.
    verbose : bool
        If True, print diagnostic information.

    Returns
    -------
    (scale, info_dict)
        scale     : float — recommended scale factor (snapped to snap_values)
        info_dict : dict  — diagnostic fields:
            'marker_diameters'  : list of measured diameters (px)
            'median_diameter'   : float
            'raw_scale'         : float (before snapping / clamping)
            'snapped_scale'     : float
            'n_glyphs_found'    : int
    """
    lx0, ly0, lx1, ly1 = (int(v) for v in legend_box)
    H_img, W_img = img_bgr.shape[:2]
    lx0 = max(0, lx0); ly0 = max(0, ly0)
    lx1 = min(W_img, lx1); ly1 = min(H_img, ly1)

    legend_crop = img_bgr[ly0:ly1, lx0:lx1]
    if legend_crop.size == 0:
        if verbose:
            print("  [estimate_optimal_scale] Empty legend crop — returning 1.0")
        return 1.0, {'marker_diameters': [], 'median_diameter': None,
                     'raw_scale': 1.0, 'snapped_scale': 1.0, 'n_glyphs_found': 0}

    gray = cv2.cvtColor(legend_crop, cv2.COLOR_BGR2GRAY) \
           if legend_crop.ndim == 3 else legend_crop
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bw, 8)

    # Collect all valid components.
    # Keep min_aspect to exclude thin vertical/horizontal lines (axes, borders).
    # Remove max_aspect upper bound so marker+line combos (large w/h) are included.
    # Increase max_marker_area to include wider marker+line components.
    valid = []
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        if area < min_marker_area or area > 2000:
            continue
        if h == 0:
            continue
        aspect = w / h
        if aspect < min_aspect:  # exclude thin vertical lines (axes, borders)
            continue
        valid.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})

    # Group components by legend row using y-center proximity
    # Use a fixed tolerance based on expected text height (~10-30px)
    legend_h = ly1 - ly0
    # Estimate row height from median component height
    all_hs = sorted(c['h'] for c in valid)
    import statistics as _stats
    med_h = _stats.median(all_hs) if all_hs else 10
    row_tolerance = max(3, int(med_h * 0.6))
    rows: list[list[dict]] = []
    for comp in sorted(valid, key=lambda c: c['y']):
        cy = comp['y'] + comp['h'] / 2
        placed = False
        for row in rows:
            row_cy = sum(c['y'] + c['h'] / 2 for c in row) / len(row)
            if abs(cy - row_cy) <= row_tolerance:
                row.append(comp)
                placed = True
                break
        if not placed:
            rows.append([comp])

    diameters = []
    for row in rows:
        # Use the tallest component's height as marker diameter.
        # If the tallest is a marker+line combo (very wide), use the second-tallest
        # that is more square-ish (w <= 4*h) to avoid inflated estimates.
        candidates = sorted(row, key=lambda c: c['h'], reverse=True)
        chosen = None
        for c in candidates:
            if c['w'] <= 4 * c['h']:  # roughly square or slightly wide
                chosen = c
                break
        if chosen is None:
            chosen = candidates[0]  # fallback: use tallest
        diam = float(chosen['h'])
        diameters.append(diam)

    info: dict = {
        'marker_diameters': diameters,
        'median_diameter':  None,
        'raw_scale':        1.0,
        'snapped_scale':    1.0,
        'n_glyphs_found':   len(diameters),
    }

    if not diameters:
        if verbose:
            print("  [estimate_optimal_scale] No marker glyphs found — returning 1.0")
        return 1.0, info

    import statistics
    median_diam = statistics.median(diameters)
    info['median_diameter'] = median_diam

    # raw scale: diam * (1/target_ratio) * scale = vit_window_px
    # i.e. diam * 1.5 * scale = 19  =>  scale = 19 / (1.5 * diam)
    raw_scale = vit_window_px / ((1.0 / target_ratio) * max(median_diam, 1.0))
    raw_scale = float(np.clip(raw_scale, min_scale, max_scale))
    info['raw_scale'] = raw_scale

    info['snapped_scale'] = raw_scale  # no snapping — use raw value directly

    if verbose:
        print(f"  [estimate_optimal_scale] "
              f"n_glyphs={len(diameters)}  median_diam={median_diam:.1f}px  "
              f"raw_scale={raw_scale:.3f}  (no snap)")

    return raw_scale, info


# ── Legend label extraction ───────────────────────────────────────────────────
def extract_legend_labels(
    img_bgr: np.ndarray,
    legend_box: tuple,
    known_classes: list[str] | None = None,
    min_marker_area: int = 8,
    verbose: bool = False,
) -> dict[str, str]:
    """Extract text labels from legend rows and map them to marker class names.

    For each legend row (detected the same way as in estimate_optimal_scale),
    the swatch/glyph is identified, then the region to the right of the swatch
    is OCR'd with pytesseract to obtain the series label.

    Parameters
    ----------
    img_bgr : np.ndarray
        Full original image (BGR).
    legend_box : tuple
        (lx0, ly0, lx1, ly1) pixel coordinates of the legend area.
    known_classes : list[str] or None
        Ordered list of class names as returned by the detection stage.
        If provided, rows are matched positionally (row 0 → known_classes[0], …).
        If None, keys are generic "row_0", "row_1", …
    verbose : bool
        Print debug info.

    Returns
    -------
    dict mapping class_name (or "row_N") → OCR label string.
    """
    try:
        import ocr_bridge_v46 as pytesseract
        from PIL import Image as _PILImage
    except ImportError:
        if verbose:
            print("  [extract_legend_labels] pytesseract not available — skipping")
        return {}

    lx0, ly0, lx1, ly1 = [int(v) for v in legend_box]
    legend_crop = img_bgr[ly0:ly1, lx0:lx1]
    if legend_crop.size == 0:
        return {}

    gray = cv2.cvtColor(legend_crop, cv2.COLOR_BGR2GRAY) \
           if legend_crop.ndim == 3 else legend_crop.copy()
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bw, 8)

    valid = []
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        if area < min_marker_area or area > 2000:
            continue
        if h == 0:
            continue
        aspect = w / h
        if aspect < 0.2:          # exclude thin vertical lines
            continue
        valid.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area})

    if not valid:
        return {}

    # Group into rows (same logic as estimate_optimal_scale)
    import statistics as _stats
    all_hs = sorted(c['h'] for c in valid)
    med_h = _stats.median(all_hs) if all_hs else 10
    row_tolerance = max(3, int(med_h * 0.6))

    rows: list[list[dict]] = []
    for comp in sorted(valid, key=lambda c: c['y']):
        cy = comp['y'] + comp['h'] / 2
        placed = False
        for row in rows:
            row_cy = sum(c['y'] + c['h'] / 2 for c in row) / len(row)
            if abs(cy - row_cy) <= row_tolerance:
                row.append(comp)
                placed = True
                break
        if not placed:
            rows.append([comp])

    # Sort rows top-to-bottom
    rows.sort(key=lambda r: min(c['y'] for c in r))

    crop_h, crop_w = legend_crop.shape[:2]
    result: dict[str, str] = {}

    for row_idx, row in enumerate(rows):
        # Find the swatch: leftmost component in this row that looks like a marker
        # (roughly square, w <= 4*h)
        candidates = sorted(row, key=lambda c: c['x'])
        swatch = None
        for c in candidates:
            if c['w'] <= 4 * c['h']:
                swatch = c
                break
        if swatch is None:
            swatch = candidates[0]

        # Text region: from right edge of swatch to right edge of legend crop
        # with a small gap
        text_x0 = swatch['x'] + swatch['w'] + 2
        # Vertical band: centred on swatch row with ±50% height padding
        text_y0 = max(0, swatch['y'] - int(swatch['h'] * 0.5))
        text_y1 = min(crop_h, swatch['y'] + swatch['h'] + int(swatch['h'] * 0.5))
        text_x1 = crop_w

        if text_x0 >= text_x1 or text_y0 >= text_y1:
            label = ""
        else:
            text_crop = legend_crop[text_y0:text_y1, text_x0:text_x1]
            # Upscale small crops for better OCR accuracy
            th, tw = text_crop.shape[:2]
            if th < 20:
                scale_up = max(1, int(20 / th))
                text_crop = cv2.resize(text_crop,
                                       (tw * scale_up, th * scale_up),
                                       interpolation=cv2.INTER_CUBIC)
            pil_img = _PILImage.fromarray(cv2.cvtColor(text_crop, cv2.COLOR_BGR2RGB))
            try:
                raw = pytesseract.image_to_string(
                    pil_img,
                    config="--psm 7 --oem 3 -c tessedit_char_whitelist="
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                           "0123456789 .,;:_-+/%()[]<>=μαβγδεζηθικλμνξοπρστυφχψω"
                ).strip()
            except Exception as _e:
                raw = ""
            # Clean up common OCR artefacts
            label = " ".join(raw.split())

        if verbose:
            print(f"  [extract_legend_labels] row {row_idx}: swatch x={swatch['x']} "
                  f"text_x0={text_x0}  label='{label}'")

        # Map to class name
        if known_classes and row_idx < len(known_classes):
            key = known_classes[row_idx]
        else:
            key = f"row_{row_idx}"
        result[key] = label

    return result

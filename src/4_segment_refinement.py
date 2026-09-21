"""
refine_segments.py
==================
Refine raw detected segments using the x-grid from Stage 1b (KDE mode_xs).

Two operations are applied in order:

  1. SHORT-SEGMENT PRUNING
     Segments whose x-span is smaller than `min_span_frac * grid_step` are
     treated as marker-blob fragments and removed.
     Default: min_span_frac = 0.5  →  threshold = 0.5 × median_grid_step

  2. CUT AT GRID BOUNDARIES
     Segments that straddle more than one grid cell are cut at every grid
     column boundary they cross.  Each cut produces two sub-segments; the
     cut point lies on the original segment line at the boundary x-coordinate.
     After cutting, any sub-segment shorter than the pruning threshold is
     also removed.

NOTE: Endpoint x-snapping (extending endpoints to the nearest grid column
centre) has been intentionally removed.  Real datasets have x-coordinate
noise so that the true point positions are not exactly on the KDE grid
column centres.  Forcing endpoints to exact grid positions over-extends
segments, causing the correction algorithm (Stage 4) to place l* endpoints
at wrong positions and select incorrect candidate points.

PUBLIC API
----------
  from refine_segments import infer_x_grid, refine

  grid_xs = infer_x_grid(known_dets)          # list of float x-column centres
  segs_refined, log = refine(L0, grid_xs)     # refined segments + diagnostic log
"""

from __future__ import annotations
import math
import numpy as np
from collections import defaultdict
from typing import Sequence


# ── helpers ───────────────────────────────────────────────────────────────────

def _seg_xspan(seg: tuple) -> float:
    return abs(seg[2] - seg[0])

def _seg_len(seg: tuple) -> float:
    return math.hypot(seg[2] - seg[0], seg[3] - seg[1])


def _cut_segment_at_boundaries(seg: tuple,
                                boundaries: list[float]) -> list[tuple]:
    """
    Cut a segment (x1,y1,x2,y2) at every boundary x-value it crosses.

    The segment is parameterised as P(t) = (x1,y1) + t*((x2-x1),(y2-y1)),
    t in [0,1].  For each boundary bx strictly between x1 and x2 (or x2 and
    x1), the cut point is computed and the segment is split there.

    Returns a list of sub-segments (always at least one element).
    """
    x1, y1, x2, y2 = seg
    dx = x2 - x1

    if abs(dx) < 1e-6:
        # Vertical or near-vertical — no x-boundary crosses possible
        return [seg]

    # Collect t values for boundaries that lie strictly inside the segment
    ts = [0.0]
    x_lo, x_hi = (x1, x2) if dx > 0 else (x2, x1)
    for bx in boundaries:
        if x_lo < bx < x_hi:
            t = (bx - x1) / dx
            ts.append(t)
    ts.append(1.0)
    ts.sort()

    if len(ts) == 2:
        # No interior cuts
        return [seg]

    # Build sub-segments from consecutive t values
    sub_segs = []
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        sx1 = x1 + t0 * dx
        sy1 = y1 + t0 * (y2 - y1)
        sx2 = x1 + t1 * dx
        sy2 = y1 + t1 * (y2 - y1)
        sub_segs.append((sx1, sy1, sx2, sy2))
    return sub_segs


# ── x-grid inference ──────────────────────────────────────────────────────────

def infer_x_grid(known_dets: list[dict],
                 cluster_tol_frac: float = 0.25) -> list[float]:
    """
    Infer the x-column grid from marker detections.

    Algorithm:
      1. Count detections per class; k = modal count (expected columns).
      2. Use 1-D k-means to cluster all cx values into k groups.
         This handles non-uniform spacing (e.g. log-axis plots) correctly.
      3. Each cluster → one grid column (mean of cluster cx values).
      4. Return sorted list of grid column x-coordinates.

    Falls back to gap-based greedy clustering if k-means fails.

    NOTE: The pipeline now uses KDE mode_xs from Stage 1b directly instead of
    calling this function.  infer_x_grid is kept for reference / offline use.

    Parameters
    ----------
    known_dets : list of detection dicts with 'cx' and 'class_name' keys
    cluster_tol_frac : fraction of estimated step used as clustering radius
                       (used only in the fallback greedy path)

    Returns
    -------
    grid_xs : sorted list of float x-column centres
    """
    if not known_dets:
        return []

    cxs = sorted(d['cx'] for d in known_dets)

    # Determine expected number of columns k from modal per-class count
    from collections import Counter
    class_counts = Counter(d['class_name'] for d in known_dets)
    if class_counts:
        k = int(np.median(list(class_counts.values())))
        k = max(k, 1)
    else:
        k = len(cxs)

    # 1-D k-means clustering
    try:
        from sklearn.cluster import KMeans
        cx_arr = np.array(cxs).reshape(-1, 1)
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        km.fit(cx_arr)
        centers = sorted(float(c[0]) for c in km.cluster_centers_)
        return centers
    except Exception:
        pass

    # Fallback: gap-based greedy clustering (original behaviour)
    diffs = [cxs[i+1] - cxs[i] for i in range(len(cxs)-1)]
    large_diffs = [d for d in diffs if d > 5]
    if not large_diffs:
        return [float(np.mean(cxs))]
    estimated_step = float(np.median(large_diffs))
    cluster_tol = cluster_tol_frac * estimated_step

    clusters: list[list[float]] = []
    current = [cxs[0]]
    for cx in cxs[1:]:
        if cx - current[-1] <= cluster_tol:
            current.append(cx)
        else:
            clusters.append(current)
            current = [cx]
    clusters.append(current)

    grid_xs = sorted(float(np.mean(c)) for c in clusters)
    return grid_xs


def grid_step(grid_xs: list[float]) -> float:
    """Median spacing between consecutive grid columns."""
    if len(grid_xs) < 2:
        return 1.0
    diffs = [grid_xs[i+1] - grid_xs[i] for i in range(len(grid_xs)-1)]
    return float(np.median(diffs))


def _grid_boundaries(grid_xs: list[float]) -> list[float]:
    """
    Return the grid column positions as cut boundaries.

    Segments are cut at the grid column x-positions directly.  All interior
    column positions (i.e. all columns) are used as cut points so that each
    resulting sub-segment spans at most one inter-column interval.

    Returns a sorted list of boundary x-values.
    """
    if len(grid_xs) < 2:
        return []
    return list(grid_xs)


# ── main refinement ───────────────────────────────────────────────────────────

def refine(segments: list[tuple],
           grid_xs: list[float],
           min_span_frac: float = 0.5,
           ) -> tuple[list[tuple], dict]:
    """
    Refine segments using the x-grid.

    Only two operations are applied (endpoint snapping has been removed):

      Step 1 — Prune short fragments whose x-span < min_span_frac * grid_step.
      Step 2 — Cut segments that straddle multiple grid cells at every grid
               column boundary, then prune any resulting sub-segments that are
               too short.

    Endpoint snapping (extending endpoints to the nearest grid column centre)
    is intentionally omitted because real datasets have x-coordinate noise,
    so forcing endpoints to exact grid positions over-extends segments and
    causes the correction algorithm to select wrong candidate points.

    Parameters
    ----------
    segments     : list of (x1, y1, x2, y2) tuples
    grid_xs      : sorted list of grid x-column centres (from mode_xs)
    min_span_frac: segments with x-span < min_span_frac * step are pruned

    Returns
    -------
    refined      : list of refined (x1, y1, x2, y2) tuples
    log          : dict with diagnostic information
    """
    step     = grid_step(grid_xs)
    min_span = min_span_frac * step

    boundaries = _grid_boundaries(grid_xs)

    pruned_short: list[tuple] = []
    kept_before_cut: list[tuple] = []

    # ── Step 1: prune short (fragment) segments ───────────────────────────────
    for seg in segments:
        xspan = _seg_xspan(seg)
        if xspan < min_span:
            pruned_short.append(seg)
        else:
            kept_before_cut.append(seg)

    # ── Step 2: cut segments that straddle multiple grid cells ────────────────
    cut_segs: list[tuple] = []
    n_cuts = 0
    for seg in kept_before_cut:
        sub = _cut_segment_at_boundaries(seg, boundaries)
        if len(sub) > 1:
            n_cuts += len(sub) - 1
        cut_segs.extend(sub)

    # After cutting, prune any sub-segments that are now too short
    cut_segs_kept: list[tuple] = []
    cut_segs_pruned: list[tuple] = []
    for seg in cut_segs:
        if _seg_xspan(seg) < min_span:
            cut_segs_pruned.append(seg)
        else:
            cut_segs_kept.append(seg)

    log = {
        'step': step,
        'min_span_threshold': min_span,
        'snap_tol': 0.0,           # snapping disabled
        'boundaries': boundaries,
        'n_input': len(segments),
        'n_pruned_short': len(pruned_short),
        'n_cuts': n_cuts,
        'n_cut_segs_pruned': len(cut_segs_pruned),
        'n_kept': len(cut_segs_kept),
        'n_endpoints_snapped': 0,  # snapping disabled
        'pruned_short': pruned_short,
        'cut_segs_pruned': cut_segs_pruned,
        'snap_log': [],
    }

    return cut_segs_kept, log


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import json, sys
    print("refine_segments module — import and use refine() and infer_x_grid()")

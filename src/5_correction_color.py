"""COLOUR copy of the Step-5 SSIM correction (5_correction_v2.py).

Loaded by color_step5.py so the colour pipeline can be developed without
touching the black-and-white module that is already deployed.  Differences
from 5_correction_v2.py:
  * CLASS_NAMES is a plain constant, so no ViT/torch import is needed
  * the detector / adaptive-NMS modules load only when a detection is run
    (with init_points supplied the correction is detector-free)
  * RENDER_LW lets the caller match the plot's stroke width
  * the [ssim-repro] block runs after I0 exists (it silently failed before)
Everything else -- the greedy, the two l* strategies, the cascade actions,
the removal cost -- is identical.


5_correction.py
===============
Stage 4: Greedy SSIM-based correction pipeline with suppressed-point activation.

This module is the final stage of the chart-digitisation pipeline:

  Stage 1a  1_point_detection_v3.py           -- V3 ViT sliding-window detector
  Stage 1b  2_point_detection_adaptive_nms.py -- adaptive x-column NMS
  Stage 2   3_segment_detection.py            -- directional-probe segment detector
  Stage 3   4_segment_refinement.py           -- mode-grid segment refinement
  Stage 4   5_correction.py  (this file)      -- greedy SSIM correction

Algorithm (per iteration)
-------------------------
1.  Render a synthetic image I_t from the current active point set P.
2.  Detect segments L_t from I_t.
3.  Select l* via two-stage strategy:
    - Current strategy: l* = argmax_{l in L_refined} delta-SSIM(I_t, I_t union l)
      (l* drawn FIRST on white canvas, render on top -- z-order last).
      If gain < L_STAR_GAIN_THRESH, skip to old strategy.
    - Old strategy: exhaustively iterate through ALL L_t segments ranked by
      SSIM gain (l* drawn FIRST on white canvas, I_0 composited on top).
      Try each l* until an improvement is found or all exhausted.
4.  Build candidate locations from the two endpoints of l*.
5.  Generate perturbations: ADD / REPLACE / DELETE at each candidate,
    plus ACTIVATE (promote the single closest suppressed point per endpoint).
    DELETE/REPLACE candidates restricted to 2D Euclidean distance <= PT_TOL.
6.  Filter out perturbations that create NEW same-class close-pair conflicts
    (2D Euclidean distance <= X_APPROX) relative to the baseline P_in.
7.  Apply post-perturbation sliding-window NMS to each candidate that adds
    a point (ADD/ACTIVATE/REPLACE), so the SSIM score reflects the clean set.
8.  Accept the perturbation that minimises 1 - SSIM(I_0, I(P_tilde)).
9.  Apply final post-perturbation NMS on the accepted point set.
10. If no improvement -> stop.

Key design choices
------------------
* l* overlay thickness = 1 px; rendered line segments use linewidth = 3
  for l* identification (Rule i) and linewidth = 1 for SSIM comparison (Rule ii).
* l* is drawn FIRST (z-order last): draw l* on white canvas, then render
  segments + symbols on top.  l* only shows in true gaps of the render.
* ACTIVATE considers only the closest suppressed point per l* endpoint
  (at most 2 suppressed candidates per iteration).
* Conflict check uses 2D Euclidean distance, not x-column buckets.
* Post-perturbation NMS uses a sliding window (0.75 * grid_step).
* The mode_xs array from Stage 1b is used directly as the x-grid for
  segment refinement (replaces the old k-means infer_x_grid).

Public API
----------
run_pipeline(plot_name, img_path, out_dir,
             model_path, detector_py_path,
             known_classes=KNOWN,
             mode_xs=None)
    -> history list  [(iter, action, n_pts, 1-SSIM, improved), ...]

run_correction(img_path, model_path, detector_py_path,
               known_classes=KNOWN, out_dir=None, mode_xs=None)
    -> dict with keys: 'history', 'out_dir'
"""

import os
import sys
import math
import random
import importlib.util
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.metrics import structural_similarity as _ssim_fn
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN = [
    'filled_circle', 'open_circle',
    'filled_square',  'open_square',
]

T_MAX    = 25      # maximum correction iterations
PT_TOL           = 18.0   # px radius for candidate / suppressed search
X_APPROX         = 8.0    # px tolerance for same-class close-pair conflict check
CASCADE_TAIL_MAX = 3      # max same-class points beyond a candidate that a
                          # REPLACE+TAIL / DELETE+TAIL action may drop together
MIN_GAIN_PER_REMOVAL = 0.002
                          # Cost charged per EXTRA net point removed (the first
                          # removal is free). Single-point DELETE/REPLACE keep
                          # their original behaviour -- they were never the
                          # problem -- while a cascade that drops several points
                          # must earn a proportional gain (observed failure: 4
                          # real points deleted for a total gain of 0.0014).
                           # (overridden at runtime to grid_step * 0.5)
RENDER_MARKER_PT = 5.0    # marker size (pt) for rendered plots
# Marker class order. Must match chart_marker_detector_v3.CLASS_NAMES; kept as a
# plain constant so the correction can run WITHOUT the ViT detector (e.g. driven by
# the colour pipeline, which supplies its own points).
CLASS_NAMES = [
    'filled_circle', 'open_circle', 'filled_square', 'open_square',
    'open_triangle', 'open_inv_triangle', 'filled_triangle', 'filled_inv_triangle',
    'open_rhombus', 'filled_rhombus', 'x_marker', 'plus_marker',
]

TRY_BOTH_STRATEGIES = True
                          # Evaluate BOTH l* strategies every iteration and keep
                          # the better one. Previously the extra-ink (old) search
                          # was skipped whenever the missing-ink (current) search
                          # improved at all -- even by 0.00006 -- so large
                          # extra-ink errors were never even evaluated.
L_STAR_GAIN_THRESH = 1e-4 # threshold for current-strategy l* gain; below this
                           # skip directly to old strategy
OLD_STRATEGY_MAX_SEGS = 12 # (D) limit Old strategy to top-N L_t segments
# (E) ThreadPoolExecutor workers: auto-detect physical CPU cores, cap at 16
PARALLEL_WORKERS = min(os.cpu_count() or 4, 16)
BATCH_PATIENCE    = 2       # (F) stop after this many consecutive no-improvement batches
GREEDY_BEST_OF_ALL = True   # evaluate the full candidate set / all ranked segments and
                            # take the LARGEST gain, instead of accepting the first
                            # improvement found. Without it a 0.00003 action can be
                            # accepted while a 0.01 action later in the list is never seen.

SYM_COLORS = {
    'filled_circle':       '#0077BB',
    'open_circle':         '#EE7733',
    'filled_square':       '#009944',
    'open_square':         '#CC3311',
    'open_triangle':       '#33BBEE',
    'open_inv_triangle':   '#44BB99',
    'filled_triangle':     '#EE3377',
    'filled_inv_triangle': '#999933',
    'plus_marker':         '#AA3377',
    'x_marker':            '#BBBBBB',
    'open_rhombus':        '#DDCC77',
    'filled_rhombus':      '#884400',
    'suppressed':          '#AAAAAA',
}

SYM_MPL = {
    'filled_circle':       ('o', True,  1.0),
    'open_circle':         ('o', False, 1.0),
    'filled_square':       ('s', True,  0.9),
    'open_square':         ('s', False, 0.9),
    'open_triangle':       ('^', False, 1.0),
    'open_inv_triangle':   ('v', False, 1.0),
    'filled_triangle':     ('^', True,  1.0),
    'filled_inv_triangle': ('v', True,  1.0),
    'plus_marker':         ('P', True,  1.1),
    'x_marker':            ('X', True,  1.1),
    'open_rhombus':        ('D', False, 0.9),
    'filled_rhombus':      ('D', True,  0.9),
    'suppressed':          ('x', False, 0.8),
}

BASE_MS   = 5
ALPHA_DET = 0.70

ACTION_COLORS = {
    'ADD':      '#009944',
    'DELETE':   '#CC3311',
    'REPLACE':  '#FF8800',
    'ACTIVATE': '#9900CC',
    'NONE':     '#888888',
}


# ---------------------------------------------------------------------------
# Module loading helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Legend-area exclusion helper
# ---------------------------------------------------------------------------

def _in_legend(cx, cy, legend_box):
    """Return True if pixel (cx, cy) falls inside the legend bounding box."""
    if legend_box is None:
        return False
    lx0, ly0, lx1, ly1 = legend_box
    return lx0 <= cx <= lx1 and ly0 <= cy <= ly1


def _filter_legend(pts, legend_box):
    """Remove any point whose (cx, cy) is inside legend_box.
    Works for both 'cx'/'cy' keys (Stage 5 internal) and
    'cx_px'/'cy_px' keys (GUI detections).
    """
    if legend_box is None:
        return pts
    kept, removed = [], []
    for p in pts:
        cx = p.get('cx', p.get('cx_px', None))
        cy = p.get('cy', p.get('cy_px', None))
        if cx is None or cy is None or _in_legend(cx, cy, legend_box):
            removed.append(p)
        else:
            kept.append(p)
    if removed:
        print(f'  [legend filter] removed {len(removed)} pts inside legend_box')
    return kept


def _filter_plot_area(pts, plot_area):
    """Keep only points whose (cx, cy) is INSIDE plot_area; drop the rest.
    Guarantees the corrected output never contains points outside the plot
    region (e.g. segment-endpoint candidates the greedy step may have added).
    Works for both 'cx'/'cy' and 'cx_px'/'cy_px' keys.
    """
    if plot_area is None:
        return pts
    x0, y0, x1, y1 = plot_area
    kept, removed = [], []
    for p in pts:
        cx = p.get('cx', p.get('cx_px', None))
        cy = p.get('cy', p.get('cy_px', None))
        if cx is None or cy is None or (x0 <= cx <= x1 and y0 <= cy <= y1):
            kept.append(p)
        else:
            removed.append(p)
    if removed:
        print(f'  [plot-area filter] removed {len(removed)} pts outside plot area')
    return kept


def _load_module(name, path):
    """Dynamically load a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# (B) Conflict-pair caching helper
# ---------------------------------------------------------------------------

def _conflict_pairs_cached(P, x_approx):
    """Return frozenset of conflict-pair keys for active points in P.
    Compute once per call; callers should cache the result for P_baseline.
    """
    active = [p for p in P if p.get('class_name') != 'suppressed']
    pairs  = set()
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            if active[i]['class_name'] == active[j]['class_name']:
                d = math.hypot(active[i]['cx'] - active[j]['cx'],
                               active[i]['cy'] - active[j]['cy'])
                if d <= x_approx:
                    key = (active[i]['class_name'],
                           round(active[i]['cx']), round(active[i]['cy']),
                           round(active[j]['cx']), round(active[j]['cy']))
                    pairs.add(key)
    return pairs


def has_new_conflicts(baseline_pairs, P_new, x_approx):
    """Return True if P_new introduces any conflict not in baseline_pairs."""
    return bool(_conflict_pairs_cached(P_new, x_approx) - baseline_pairs)


# ---------------------------------------------------------------------------
# (F) Batched parallel perturbation evaluator with early-stop
# ---------------------------------------------------------------------------

def _eval_perturbs_batched(valid_perturbs, eval_fn, current_best_d,
                           batch_size=None, patience=None, cost_fn=None):
    """Evaluate perturbations in parallel batches with early-stop.

    Parameters
    ----------
    valid_perturbs : list of (action, point_list)
        Ordered list of candidate perturbations to evaluate.
    eval_fn : callable
        Function that takes (action, point_list) and returns (dist, pt, act).
        Must be picklable / thread-safe (uses ThreadPoolExecutor).
    current_best_d : float
        Current best 1-SSIM distance; improvement means d < current_best_d - 1e-7.
    batch_size : int or None
        Number of perturbations per parallel batch.  Defaults to PARALLEL_WORKERS.
    patience : int or None
        Stop after this many consecutive no-improvement batches.
        Defaults to BATCH_PATIENCE.

    Returns
    -------
    (best_d, best_P, best_action)  -- best found, or (current_best_d, None, None)
    if no improvement was found.
    """
    if not valid_perturbs:
        return current_best_d, None, None

    bs  = batch_size or PARALLEL_WORKERS
    pat = patience  if patience is not None else BATCH_PATIENCE

    best_d      = current_best_d
    best_score  = current_best_d   # cost-adjusted; "do nothing" has zero cost
    best_P      = None
    best_action = None
    no_improve_streak = 0

    for batch_start in range(0, len(valid_perturbs), bs):
        batch = valid_perturbs[batch_start: batch_start + bs]

        batch_improved = False
        with ThreadPoolExecutor(max_workers=bs) as ex:
            for d, pt, act in ex.map(eval_fn, batch):
                # Score = raw 1-SSIM + cost of the points this perturbation drops,
                # so a multi-point cascade must earn its deletions.
                _sc = d + (cost_fn(act, pt) if cost_fn is not None else 0.0)
                if _sc < best_score - 1e-7:
                    best_d, best_score = d, _sc
                    best_P, best_action = pt, act
                    batch_improved = True

        if batch_improved:
            no_improve_streak = 0
            if not GREEDY_BEST_OF_ALL:
                break          # legacy: accept the first improvement found
        else:
            no_improve_streak += 1
            if no_improve_streak >= pat:
                break

    return best_d, best_P, best_action


# ---------------------------------------------------------------------------
# SSIM distance
# ---------------------------------------------------------------------------

_SSIM_CROP = None   # (x0,y0,x1,y1); if set, SSIM is computed ONLY inside this box


def ssim_dist(A, B):
    """1 - SSIM between two BGR images (lower = more similar).

    If _SSIM_CROP (plot area) is set, both images are cropped to it first, so
    out-of-plot content (border, side text, tick labels) can NEVER drive the
    greedy correction. This makes 1-SSIM reflect only how well the points
    reconstruct the actual data region.
    """
    if _SSIM_CROP is not None:
        x0, y0, x1, y1 = _SSIM_CROP
        A = A[y0:y1, x0:x1]
        B = B[y0:y1, x0:x1]
    a = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    b = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    return 1.0 - float(_ssim_fn(a, b, data_range=1.0))


def save_ssim_residual(I0, points, out_dir, tag, dist=None, plot_area=None):
    """Diagnostic: reference | reconstruction | SSIM-residual heatmap.

    Lets you SEE what 1-SSIM is actually minimising. Bright residual ON real
    markers => SSIM is correctly chasing missing/misplaced points. Bright
    residual ON axis lines / text / edges => SSIM is being driven by structure
    the point-render can never reproduce, so it may add spurious points there.
    Also prints how much of the total residual falls inside vs outside the plot
    data region as a quick numeric sanity check.
    """
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as _plt
        H, W = I0.shape[:2]
        recon = render_from_points(points, (H, W))
        a = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
        b = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
        score, smap = _ssim_fn(a, b, data_range=1.0, full=True)
        resid = 1.0 - smap
        _d = dist if dist is not None else (1.0 - score)
        _msg = ""
        if plot_area is not None:
            px0, py0, px1, py1 = [int(v) for v in plot_area]
            _inside = float(resid[py0:py1, px0:px1].sum())
            _total  = float(resid.sum()) + 1e-9
            _msg = (f", residual inside plot={100*_inside/_total:.0f}%"
                    f" (outside={100*(1-_inside/_total):.0f}% = axis/edges/text)")
        fig, ax = _plt.subplots(1, 3, figsize=(21, 7))
        ax[0].imshow(cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)); ax[0].set_title('Reference I0 (cleaned)'); ax[0].axis('off')
        ax[1].imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f'Reconstruction ({len(points)} pts)'); ax[1].axis('off')
        im = ax[2].imshow(resid, cmap='hot', vmin=0, vmax=max(0.1, float(resid.max())))
        if plot_area is not None:
            ax[2].add_patch(_plt.Rectangle((px0, py0), px1 - px0, py1 - py0,
                                           fill=False, edgecolor='cyan', lw=1.2))
        ax[2].set_title(f'SSIM residual  1-SSIM={_d:.4f}\nbright=mismatch (real markers? or axis/edges?)')
        ax[2].axis('off')
        fig.colorbar(im, ax=ax[2], fraction=0.046)
        _plt.tight_layout()
        p = os.path.join(out_dir, f'ssim_residual_{tag}.png')
        _plt.savefig(p, dpi=90, bbox_inches='tight'); _plt.close(fig)
        print(f"  [ssim-diag] {tag}: 1-SSIM={_d:.4f}  residual_sum={float(resid.sum()):.0f}{_msg}  -> {p}")
        return p
    except Exception as _se:
        print(f"  [ssim-diag] {tag} failed: {_se}")
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OpenCV marker drawing helpers (B optimisation)
# ---------------------------------------------------------------------------

# Marker radius in pixels for the rendered image (RENDER_MARKER_PT is in pt
# units; at DPI=100 one pt ~ 1.39 px, so 5 pt ~ 7 px radius).
_CV_MARKER_R = 4   # half-size in px for cv2 marker drawing
RENDER_LW    = 1   # default polyline width used by render_from_points.
                   # Callers driving the correction on a plot whose strokes are
                   # thicker (e.g. the colour pipeline) should set this to match,
                   # otherwise every segment carries a constant residual and the
                   # greedy "improves" SSIM by deleting points.

# Map class index -> (shape_type, filled)
#   shape_type: 'circle', 'square', 'triangle_up', 'triangle_down',
#               'rhombus', 'x', 'plus'
_CV_SHAPE = [
    ('circle',        True),   # 0  filled_circle
    ('circle',        False),  # 1  open_circle
    ('square',        True),   # 2  filled_square
    ('square',        False),  # 3  open_square
    ('triangle_up',   False),  # 4  open_triangle
    ('triangle_down', False),  # 5  open_inv_triangle
    ('triangle_up',   True),   # 6  filled_triangle
    ('triangle_down', True),   # 7  filled_inv_triangle
    ('rhombus',       False),  # 8  open_rhombus
    ('rhombus',       True),   # 9  filled_rhombus
    ('x',             True),   # 10 x_marker
    ('plus',          True),   # 11 plus_marker
]


def _cv_draw_marker(img, cx, cy, class_idx, r=_CV_MARKER_R):
    """Draw a single marker onto img (BGR, in-place) using OpenCV."""
    ci = int(class_idx) % len(_CV_SHAPE)
    shape, filled = _CV_SHAPE[ci]
    cx, cy = int(round(cx)), int(round(cy))
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    lw = 1  # outline thickness

    if shape == 'circle':
        if filled:
            cv2.circle(img, (cx, cy), r, BLACK, -1)
        else:
            cv2.circle(img, (cx, cy), r, BLACK, lw)
            cv2.circle(img, (cx, cy), max(1, r - lw), WHITE, -1)

    elif shape == 'square':
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        if filled:
            cv2.rectangle(img, (x0, y0), (x1, y1), BLACK, -1)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), BLACK, lw)
            cv2.rectangle(img, (x0 + lw, y0 + lw), (x1 - lw, y1 - lw), WHITE, -1)

    elif shape in ('triangle_up', 'triangle_down'):
        if shape == 'triangle_up':
            pts = np.array([
                [cx,     cy - r],
                [cx - r, cy + r],
                [cx + r, cy + r],
            ], dtype=np.int32)
        else:
            pts = np.array([
                [cx,     cy + r],
                [cx - r, cy - r],
                [cx + r, cy - r],
            ], dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts], BLACK)
        else:
            cv2.polylines(img, [pts], isClosed=True, color=BLACK, thickness=lw)
            # fill interior with white
            inner = (pts * (r - lw) // r).astype(np.int32)
            inner = inner + np.array([[cx, cy]], dtype=np.int32) - (inner.mean(axis=0)).astype(np.int32)
            cv2.fillPoly(img, [inner], WHITE)

    elif shape == 'rhombus':
        pts = np.array([
            [cx,     cy - r],
            [cx + r, cy    ],
            [cx,     cy + r],
            [cx - r, cy    ],
        ], dtype=np.int32)
        if filled:
            cv2.fillPoly(img, [pts], BLACK)
        else:
            cv2.polylines(img, [pts], isClosed=True, color=BLACK, thickness=lw)
            inner = np.array([
                [cx,         cy - r + lw],
                [cx + r - lw, cy        ],
                [cx,         cy + r - lw],
                [cx - r + lw, cy        ],
            ], dtype=np.int32)
            cv2.fillPoly(img, [inner], WHITE)

    elif shape == 'x':
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), BLACK, lw + 1)
        cv2.line(img, (cx + r, cy - r), (cx - r, cy + r), BLACK, lw + 1)

    elif shape == 'plus':
        cv2.line(img, (cx - r, cy), (cx + r, cy), BLACK, lw + 1)
        cv2.line(img, (cx, cy - r), (cx, cy + r), BLACK, lw + 1)


def render_from_points(points, pa_shape, linewidth=None):
    """
    Render a synthetic chart image from a list of active point dicts.
    Uses OpenCV for fast rendering (replaces matplotlib, ~5-10x speedup).

    Parameters
    ----------
    points : list of dict
        Each dict must have 'cx', 'cy', 'class_name', 'class_idx'.
        Points with class_name='suppressed' are excluded automatically.
    pa_shape : (H, W)
        Pixel dimensions of the plotting area.
    linewidth : float
        Width of the connecting lines (default 1.0).

    Returns
    -------
    BGR numpy array of shape (H, W, 3).
    """
    if linewidth is None:
        linewidth = RENDER_LW
    H_pa, W_pa = pa_shape
    img = np.full((H_pa, W_pa, 3), 255, dtype=np.uint8)  # white background

    active = [p for p in points if p.get('class_name') != 'suppressed']

    # Build class_idx for each point
    _CN = ['filled_circle', 'open_circle', 'filled_square', 'open_square',
           'open_triangle', 'open_inv_triangle', 'filled_triangle',
           'filled_inv_triangle', 'open_rhombus', 'filled_rhombus',
           'x_marker', 'plus_marker']

    by_class = defaultdict(list)
    for p in active:
        cn = p.get('class_name', '')
        ci = p.get('class_idx', _CN.index(cn) if cn in _CN else 0)
        by_class[ci].append((p['cx'], p['cy']))

    lw_px = max(1, int(round(linewidth)))

    # Draw connecting lines first (lower z-order)
    for ci, pts in by_class.items():
        if len(pts) < 2:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        for i in range(len(pts_sorted) - 1):
            x0, y0 = int(round(pts_sorted[i][0])),     int(round(pts_sorted[i][1]))
            x1, y1 = int(round(pts_sorted[i + 1][0])), int(round(pts_sorted[i + 1][1]))
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), lw_px)

    # Draw markers on top (higher z-order)
    for ci, pts in by_class.items():
        for (px, py) in pts:
            _cv_draw_marker(img, px, py, ci)

    return img


def _render_on_canvas(points, pa_shape, linewidth=3.0, underlay_bgr=None):
    """
    Render segments and symbols onto a canvas using OpenCV (fast path).
    If underlay_bgr is provided (H x W x 3 uint8), it is used as the
    background instead of white -- so l* drawn beforehand is underneath
    (z-order last for l*).
    """
    H_pa, W_pa = pa_shape

    if underlay_bgr is not None:
        img = cv2.resize(underlay_bgr, (W_pa, H_pa)).copy()
    else:
        img = np.full((H_pa, W_pa, 3), 255, dtype=np.uint8)

    active = [p for p in points if p.get('class_name') != 'suppressed']

    _CN = ['filled_circle', 'open_circle', 'filled_square', 'open_square',
           'open_triangle', 'open_inv_triangle', 'filled_triangle',
           'filled_inv_triangle', 'open_rhombus', 'filled_rhombus',
           'x_marker', 'plus_marker']

    by_class = defaultdict(list)
    for p in active:
        cn = p.get('class_name', '')
        ci = p.get('class_idx', _CN.index(cn) if cn in _CN else 0)
        by_class[ci].append((p['cx'], p['cy']))

    lw_px = max(1, int(round(linewidth)))

    # Draw connecting lines first
    for ci, pts in by_class.items():
        if len(pts) < 2:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        for i in range(len(pts_sorted) - 1):
            x0, y0 = int(round(pts_sorted[i][0])),     int(round(pts_sorted[i][1]))
            x1, y1 = int(round(pts_sorted[i + 1][0])), int(round(pts_sorted[i + 1][1]))
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), lw_px)

    # Draw markers on top
    for ci, pts in by_class.items():
        for (px, py) in pts:
            _cv_draw_marker(img, px, py, ci)

    return img


# ---------------------------------------------------------------------------
# Conflict check (2D Euclidean, new conflicts only)
# ---------------------------------------------------------------------------

def count_new_conflicts(P_baseline, P_new, x_approx):
    """Return number of NEW same-class close-pair conflicts in P_new vs P_baseline.

    A conflict is a pair of same-class active points whose 2D Euclidean distance
    is <= x_approx.  Only conflicts that are NOT already present in P_baseline
    are counted, so pre-existing near-duplicates do not block all perturbations.
    """
    def _conflict_pairs(P):
        active = [p for p in P if p.get('class_name') != 'suppressed']
        pairs  = set()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if active[i]['class_name'] == active[j]['class_name']:
                    d = math.hypot(active[i]['cx'] - active[j]['cx'],
                                   active[i]['cy'] - active[j]['cy'])
                    if d <= x_approx:
                        key = (active[i]['class_name'],
                               round(active[i]['cx']), round(active[i]['cy']),
                               round(active[j]['cx']), round(active[j]['cy']))
                        pairs.add(key)
        return pairs

    new_conflicts = _conflict_pairs(P_new) - _conflict_pairs(P_baseline)
    return len(new_conflicts)


# ---------------------------------------------------------------------------
# Post-perturbation sliding-window NMS
# ---------------------------------------------------------------------------

NMS_VERTICAL_KEEP = 0.0
"""Two same-class points closer than `window` in x are normally merged.  When
this is > 0, a pair separated by MORE than this many pixels in y is kept
instead: a curve that rises vertically (a re-dose spike) legitimately has two
points -- the trough and the peak -- at the same x, and x-only NMS would destroy
one of them.  The colour driver sets it from the measured marker size."""


def post_perturbation_nms(P, S, window, vertical_keep=None):
    """Apply sliding-window NMS to P after a perturbation.

    For each class, sort active points by cx.  Sweep with a sliding window
    of width `window`.  When two same-class points fall within the window,
    keep the one with higher confidence (ties broken randomly); move the
    other to S -- unless they are far enough apart in y (see
    NMS_VERTICAL_KEEP), in which case both survive.

    Returns (P_clean, S_updated).
    """
    _vk = NMS_VERTICAL_KEEP if vertical_keep is None else vertical_keep
    active  = [p for p in P if p.get('class_name') != 'suppressed']
    by_cls  = defaultdict(list)
    for p in active:
        by_cls[p['class_name']].append(p)

    keep_ids = set()
    for cls, pts in by_cls.items():
        pts_sorted = sorted(pts, key=lambda p: p['cx'])
        suppressed_ids = set()
        for i, anchor in enumerate(pts_sorted):
            if id(anchor) in suppressed_ids:
                continue
            for j in range(i + 1, len(pts_sorted)):
                other = pts_sorted[j]
                if other['cx'] - anchor['cx'] > window:
                    break
                if id(other) in suppressed_ids:
                    continue
                if _vk > 0 and abs(other['cy'] - anchor['cy']) > _vk:
                    continue          # vertical rise: keep both trough and peak
                # Within window -- suppress the weaker one
                a_conf = anchor.get('confidence', 0.0)
                o_conf = other.get('confidence', 0.0)
                if o_conf > a_conf or (o_conf == a_conf and random.random() < 0.5):
                    suppressed_ids.add(id(anchor))
                    anchor = other   # promote other as new anchor
                else:
                    suppressed_ids.add(id(other))
            if id(anchor) not in suppressed_ids:
                keep_ids.add(id(anchor))

    P_clean   = [p for p in active if id(p) in keep_ids]
    newly_sup = [p for p in active if id(p) not in keep_ids]
    for p in newly_sup:
        p_copy = dict(p)
        p_copy['class_name'] = 'suppressed'
        S = S + [p_copy]
    return P_clean, S


# ---------------------------------------------------------------------------
# Perturbation generation
# ---------------------------------------------------------------------------

def generate_perturbations(P, sup_near, cand_locs, symbol_set):
    """
    Generate ADD / REPLACE / DELETE / ACTIVATE perturbations.

    Parameters
    ----------
    P          : current active point list
    sup_near   : suppressed points near l* endpoints (at most 2)
    cand_locs  : list of {'cx', 'cy'} dicts (l* endpoints)
    symbol_set : list of symbol class names to try

    Returns
    -------
    list of (action_str, P_tilde) tuples
    """
    try:                       # prefer the detector's list when it is importable
        from chart_marker_detector_v3 import CLASS_NAMES as _CN
    except Exception:          # colour pipeline / no-torch use
        _CN = CLASS_NAMES

    perturbations = []

    for _ic, c in enumerate(cand_locs):
        cx, cy = c['cx'], c['cy']
        # Use 2D Euclidean distance (PT_TOL) to restrict DELETE/REPLACE candidates
        # to points that are genuinely near the anchor -- not just the same x-column.
        Q = [p for p in P
             if math.hypot(p['cx'] - cx, p['cy'] - cy) <= PT_TOL
             and p['class_name'] != 'suppressed']

        # -- Cascade tail -----------------------------------------------------
        # With points 1-2-3-4-5 on one curve and l* = 3-4, acting on point 3
        # alone can be misleading: deleting 3's neighbour 2 would join 1--3 and
        # wreck the SSIM, so greedy prefers deleting the REAL point 3 instead of
        # the spurious run {1,2}. To escape that local optimum we also offer the
        # variants "act on 3 AND drop the run beyond it".
        # tail(q) = same-class points on the far side of q from the other l*
        # endpoint (i.e. 1,2 when l* extends to the right), nearest first,
        # capped at CASCADE_TAIL_MAX so the action count at most doubles.
        _other = cand_locs[1 - _ic] if len(cand_locs) == 2 else None

        def _tail_of(q, _other=_other):
            if _other is None:
                return []
            same = [p for p in P
                    if p is not q and p['class_name'] == q['class_name']]
            if _other['cx'] >= q['cx']:          # segment runs right -> tail is left
                out = sorted((p for p in same if p['cx'] < q['cx']),
                             key=lambda p: -p['cx'])
            else:                                 # segment runs left -> tail is right
                out = sorted((p for p in same if p['cx'] > q['cx']),
                             key=lambda p: p['cx'])
            return out[:CASCADE_TAIL_MAX]

        tails = {id(q): _tail_of(q) for q in Q}

        for s in symbol_set:
            ci     = _CN.index(s)
            new_pt = {'cx': cx, 'cy': cy, 'class_name': s, 'class_idx': ci}
            perturbations.append(('ADD', P + [new_pt]))
            for q in Q:
                P_new = [p for p in P if p is not q] + [new_pt]
                perturbations.append(('REPLACE', P_new))
                _tail = tails[id(q)]
                if _tail:
                    _drop = {id(q)} | {id(t) for t in _tail}
                    perturbations.append(
                        ('REPLACE+TAIL',
                         [p for p in P if id(p) not in _drop] + [new_pt]))

        for q in Q:
            perturbations.append(('DELETE', [p for p in P if p is not q]))
            _tail = tails[id(q)]
            if _tail:
                _drop = {id(q)} | {id(t) for t in _tail}
                perturbations.append(
                    ('DELETE+TAIL', [p for p in P if id(p) not in _drop]))

    # ACTIVATE: promote a suppressed point into P with a real symbol type
    for sup in sup_near:
        for s in symbol_set:
            ci = _CN.index(s)
            activated = {'cx': sup['cx'], 'cy': sup['cy'],
                         'class_name': s, 'class_idx': ci,
                         '_activated_from_suppressed': True}
            perturbations.append(('ACTIVATE', P + [activated]))

    return perturbations


# ---------------------------------------------------------------------------
# l* selection strategies
# ---------------------------------------------------------------------------

def _select_l_star_current(L_refined, I_t, points, lw=1):
    """l* = argmax_{l in L_refined} delta-SSIM(I_t, I_t union l).

    l* is drawn FIRST onto a white canvas (thin, lw=1), then all segments
    and symbols are rendered ON TOP with their normal appearance.  This means
    l* is only visible in the true gaps of the render -- it is hidden beneath
    rendered lines and beneath the white interiors of open symbols (z-order last).
    Parallelised with ThreadPoolExecutor (numpy/OpenCV release the GIL).
    """
    H, W = I_t.shape[:2]

    def _eval_seg(seg):
        underlay = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.line(underlay,
                 (int(seg[0]), int(seg[1])),
                 (int(seg[2]), int(seg[3])),
                 (0, 0, 0), lw)
        I_ov = _render_on_canvas(points, (H, W), linewidth=3.0, underlay_bgr=underlay)
        return ssim_dist(I_t, I_ov), seg

    if not L_refined:
        return None, -float('inf')

    best_l, best_d = None, -float('inf')
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        for d, seg in ex.map(_eval_seg, L_refined):
            if d > best_d:
                best_d, best_l = d, seg
    return best_l, best_d


def _select_l_star_old(L_t, I0, lw=1):
    """l* = argmax_{l in L_t} delta-SSIM(I_0, I_0 union l).

    The overlay segment is drawn FIRST onto a white canvas (lw=1, black),
    then the original image I_0 is composited ON TOP using a dilated mask
    so that l* is hidden inside open-symbol interiors in I_0 (z-order last).
    """
    H, W = I0.shape[:2]
    I0_gray   = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    raw_mask  = (I0_gray < 250)
    # Dilate to fill open-symbol interiors in the original image
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    orig_mask = cv2.dilate(raw_mask.astype(np.uint8), kernel).astype(bool)

    best_l, best_d = None, -float('inf')
    for seg in L_t:
        # 1. Draw l* on a white canvas
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.line(canvas,
                 (int(seg[0]), int(seg[1])),
                 (int(seg[2]), int(seg[3])),
                 (0, 0, 0), lw)
        # 2. Composite: wherever I_0 has content (dilated), overwrite with I_0
        I_ov = canvas.copy()
        I_ov[orig_mask] = I0[orig_mask]
        d = ssim_dist(I0, I_ov)
        if d > best_d:
            best_d, best_l = d, seg
    return best_l, best_d


def _rank_all_l_star_old(L_t, I0, lw=1):
    """Rank ALL L_t segments by SSIM gain (descending) for exhaustive search.

    Same z-order-last compositing as _select_l_star_old, but returns the
    full ranked list instead of just the best.
    Parallelised with ThreadPoolExecutor (numpy/OpenCV release the GIL).
    """
    H, W = I0.shape[:2]
    I0_gray   = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    raw_mask  = (I0_gray < 250)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    orig_mask = cv2.dilate(raw_mask.astype(np.uint8), kernel).astype(bool)
    # Pre-copy I0 slice for thread safety (read-only, but explicit copy avoids
    # any potential numpy view aliasing across threads)
    I0_vals   = I0[orig_mask]  # shape (N, 3), read-only reference is fine

    def _eval_seg(seg):
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.line(canvas,
                 (int(seg[0]), int(seg[1])),
                 (int(seg[2]), int(seg[3])),
                 (0, 0, 0), lw)
        canvas[orig_mask] = I0_vals
        d = ssim_dist(I0, canvas)
        return seg, d

    if not L_t:
        return []

    scored = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        for seg, d in ex.map(_eval_seg, L_t):
            scored.append((seg, d))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# One correction iteration (with all fixes)
# ---------------------------------------------------------------------------

def run_one_iteration(P_in, S_in, I0, L_refined, known_classes,
                      nms_window=None, legend_box=None):
    """
    Execute one greedy correction step.

    Parameters
    ----------
    P_in          : list of active point dicts
    S_in          : list of suppressed point dicts
    I0            : original chart image (BGR)
    L_refined     : refined segments from Stage 3
    known_classes : list of symbol class names
    nms_window    : float or None
        Sliding-window width for post-perturbation NMS.
        If None, NMS is skipped.
    legend_box    : (x0,y0,x1,y1) or None
        If provided, candidate locations and ACTIVATE points that fall
        inside this box are rejected before perturbation evaluation.

    Returns a state dict with keys:
      converged, P_in, P_out, S_in, S_out,
      I_t, I_out, L_t, l_star, l_star_diff, l_star_strategy,
      anchor_left, anchor_right, candidates, sup_near,
      added, removed, action, best_dist, baseline, improved,
      n_valid_perturbs, n_sup_near
    """
    _seg_det_mod = _load_module('segment_detector',
                               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '3_segment_detection_v2.py'))
    detect_debug = _seg_det_mod.detect_debug

    H, W = I0.shape[:2]
    # Rule i: render I_t with thick lines so the l* overlay stands out
    I_t  = render_from_points(P_in, (H, W), linewidth=3.0)

    # (A) Fast path for synthetic renders: fewer directions + smaller probe radius.
    # I_t is a clean white-background image with simple black lines/markers,
    # so the full 24-direction / radius-12 scan is overkill.
    seg_res = detect_debug(I_t, n_dirs=8, probe_radius=6)
    L_t     = seg_res['segments']

    # Stage 1: current l* strategy
    best_l_star, best_l_diff = _select_l_star_current(L_refined, I_t, P_in)
    l_star_strategy = 'current'
    use_old_strategy = (best_l_star is None) or (best_l_diff < L_STAR_GAIN_THRESH)

    if best_l_star is None and not L_t:
        return {'converged': True, 'P_out': P_in, 'S_out': S_in,
                'I_t': I_t, 'L_t': L_t}

    # Rule ii: thin render (lw=1, default) for SSIM scoring
    I_t_thin = render_from_points(P_in, (H, W))   # lw=1
    baseline = ssim_dist(I0, I_t_thin)
    best_P, best_d, best_action = P_in, baseline, 'NONE'
    improved = False

    # Cost charged per NET point removed, so a cascade that drops several points
    # must improve 1-SSIM proportionally. Guards against a single +TAIL action
    # wiping out a whole curve limb for a negligible gain.
    _n_in = len(P_in)
    def _removal_cost(_act, _pt):
        # first removed point is free; each ADDITIONAL one must pay
        return MIN_GAIN_PER_REMOVAL * max(0, (_n_in - len(_pt)) - 1)
    candidates, sup_near, valid_perturbs = [], [], []
    anchor_left, anchor_right = (0.0, 0.0), (0.0, 0.0)

    # --- Current strategy: only run if gain is above threshold ---
    if not use_old_strategy:
        lx = float(min(best_l_star[0], best_l_star[2]))
        ly = float(best_l_star[1] if best_l_star[0] <= best_l_star[2] else best_l_star[3])
        rx = float(max(best_l_star[0], best_l_star[2]))
        ry = float(best_l_star[3] if best_l_star[0] <= best_l_star[2] else best_l_star[1])
        anchor_left  = (lx, ly)
        anchor_right = (rx, ry)
        # Exclude anchor locs that fall inside the legend box
        cand_locs = [c for c in [
                         {'cx': anchor_left[0],  'cy': anchor_left[1]},
                         {'cx': anchor_right[0], 'cy': anchor_right[1]},
                     ] if not _in_legend(c['cx'], c['cy'], legend_box)]

        candidates, seen_ids = [], set()
        for anchor in [anchor_left, anchor_right]:
            if _in_legend(anchor[0], anchor[1], legend_box):
                continue
            for p in P_in:
                pid = id(p)
                if pid in seen_ids:
                    continue
                if math.hypot(p['cx'] - anchor[0], p['cy'] - anchor[1]) <= PT_TOL:
                    candidates.append(p)
                    seen_ids.add(pid)

        # Exclude suppressed points inside the legend box
        sup_near, seen_sup = [], set()
        for anchor in [anchor_left, anchor_right]:
            if _in_legend(anchor[0], anchor[1], legend_box):
                continue
            best_sup, best_d_sup = None, float('inf')
            for s in S_in:
                if _in_legend(s['cx'], s['cy'], legend_box):
                    continue
                d = math.hypot(s['cx'] - anchor[0], s['cy'] - anchor[1])
                if d <= PT_TOL and d < best_d_sup:
                    best_d_sup, best_sup = d, s
            if best_sup is not None and id(best_sup) not in seen_sup:
                sup_near.append(best_sup)
                seen_sup.add(id(best_sup))

        all_perturbs   = generate_perturbations(P_in, sup_near, cand_locs, known_classes)
        # (B) Cache baseline conflict pairs once -- avoids O(n^2) recompute per perturbation
        _baseline_pairs = _conflict_pairs_cached(P_in, X_APPROX)
        # Also filter out any ADD/ACTIVATE result whose new point lands in legend
        valid_perturbs = [
            (a, pt) for a, pt in all_perturbs
            if not has_new_conflicts(_baseline_pairs, pt, X_APPROX)
            and not any(_in_legend(p['cx'], p['cy'], legend_box)
                        for p in pt if p not in P_in)
        ]

        def _eval_perturb(args):
            act, pt = args
            if nms_window is not None and act in ('ADD', 'ACTIVATE', 'REPLACE'):
                pt, _ = post_perturbation_nms(pt, S_in, nms_window)
            d = ssim_dist(I0, render_from_points(pt, (H, W)))
            return d, pt, act

        # (F) Batched early-stop: evaluate PARALLEL_WORKERS at a time;
        #     stop as soon as any batch improves, or after BATCH_PATIENCE
        #     consecutive no-improvement batches.
        _bd, _bp, _ba = _eval_perturbs_batched(
            valid_perturbs, _eval_perturb, best_d, cost_fn=_removal_cost)
        if _bp is not None:
            best_d, best_P, best_action = _bd, _bp, _ba

        improved = best_d < baseline - 1e-7

    # --- Old strategy: top-N L_t segments only (D optimisation) ---
    _cur_d, _cur_action = best_d, best_action   # result of the missing-ink search
    if L_t and (TRY_BOTH_STRATEGIES or use_old_strategy or not improved):
        scored_lt = _rank_all_l_star_old(L_t, I0)
        n_total = len(scored_lt)
        scored_lt = scored_lt[:OLD_STRATEGY_MAX_SEGS]   # limit to top-N
        print(f"  Old strategy: trying top {len(scored_lt)} of {n_total} L_t segments")

        for seg_rank, (old_l_star, old_l_diff) in enumerate(scored_lt, 1):
            lx2 = float(min(old_l_star[0], old_l_star[2]))
            ly2 = float(old_l_star[1] if old_l_star[0] <= old_l_star[2] else old_l_star[3])
            rx2 = float(max(old_l_star[0], old_l_star[2]))
            ry2 = float(old_l_star[3] if old_l_star[0] <= old_l_star[2] else old_l_star[1])
            anchor_left2  = (lx2, ly2)
            anchor_right2 = (rx2, ry2)
            # Exclude anchor locs inside the legend box
            cand_locs2 = [c for c in [
                              {'cx': anchor_left2[0], 'cy': anchor_left2[1]},
                              {'cx': anchor_right2[0], 'cy': anchor_right2[1]},
                          ] if not _in_legend(c['cx'], c['cy'], legend_box)]

            candidates2, seen_ids2 = [], set()
            for anchor in [anchor_left2, anchor_right2]:
                if _in_legend(anchor[0], anchor[1], legend_box):
                    continue
                for p in P_in:
                    pid = id(p)
                    if pid in seen_ids2:
                        continue
                    if math.hypot(p['cx'] - anchor[0], p['cy'] - anchor[1]) <= PT_TOL:
                        candidates2.append(p)
                        seen_ids2.add(pid)

            sup_near2, seen_sup2 = [], set()
            for anchor in [anchor_left2, anchor_right2]:
                if _in_legend(anchor[0], anchor[1], legend_box):
                    continue
                best_sup2, best_d2 = None, float('inf')
                for s in S_in:
                    if _in_legend(s['cx'], s['cy'], legend_box):
                        continue
                    d2 = math.hypot(s['cx'] - anchor[0], s['cy'] - anchor[1])
                    if d2 <= PT_TOL and d2 < best_d2:
                        best_d2, best_sup2 = d2, s
                if best_sup2 is not None and id(best_sup2) not in seen_sup2:
                    sup_near2.append(best_sup2)
                    seen_sup2.add(id(best_sup2))

            all_perturbs2   = generate_perturbations(P_in, sup_near2, cand_locs2, known_classes)
            # (B) Reuse cached baseline pairs (computed once above, or compute here if
            #     current strategy was skipped -- use try/except for safe local var check)
            try:
                _baseline_pairs
            except NameError:
                _baseline_pairs = _conflict_pairs_cached(P_in, X_APPROX)
            # Filter out any ADD/ACTIVATE result whose new point lands in legend
            valid_perturbs2 = [
                (a, pt) for a, pt in all_perturbs2
                if not has_new_conflicts(_baseline_pairs, pt, X_APPROX)
                and not any(_in_legend(p['cx'], p['cy'], legend_box)
                            for p in pt if p not in P_in)
            ]

            local_best_d, local_best_P, local_best_action = best_d, best_P, best_action

            def _eval_perturb2(args):
                act2, pt2 = args
                if nms_window is not None and act2 in ('ADD', 'ACTIVATE', 'REPLACE'):
                    pt2, _ = post_perturbation_nms(pt2, S_in, nms_window)
                d2 = ssim_dist(I0, render_from_points(pt2, (H, W)))
                return d2, pt2, act2

            # (F) Batched early-stop for old strategy too
            _bd2, _bp2, _ba2 = _eval_perturbs_batched(
                valid_perturbs2, _eval_perturb2, local_best_d,
                cost_fn=_removal_cost)
            if _bp2 is not None:
                local_best_d, local_best_P, local_best_action = _bd2, _bp2, _ba2

            if local_best_d < best_d - 1e-7:
                best_d, best_P, best_action = local_best_d, local_best_P, local_best_action
                improved        = True
                best_l_star     = old_l_star
                best_l_diff     = old_l_diff
                l_star_strategy = f'old[{seg_rank}]'
                anchor_left     = anchor_left2
                anchor_right    = anchor_right2
                candidates      = candidates2
                sup_near        = sup_near2
                valid_perturbs  = valid_perturbs2
                print(f"    New best at L_t rank {seg_rank}  "
                      f"l*-score={old_l_diff:.5f}  action={best_action}  "
                      f"1-SSIM={best_d:.5f}")
                if not GREEDY_BEST_OF_ALL:
                    break
            else:
                print(f"    L_t rank {seg_rank}: gain={old_l_diff:.5f}  no improvement")

        if not improved:
            print(f"  Old strategy: all {len(scored_lt)} L_t segments exhausted, "
                  f"no improvement found")
        if _cur_action not in (None, 'NONE'):
            _cg = baseline - _cur_d
            _og = _cur_d - best_d
            print(f"  [strategy] missing-ink gain={_cg:.5f} ({_cur_action})  |  "
                  f"extra-ink extra gain={_og:.5f}  ->  chose "
                  f"{'extra-ink' if _og > 0 else 'missing-ink'}")

    if not improved:
        best_action = 'NONE'
        best_P      = P_in

    # Update S_out: remove activated suppressed point from S
    S_out = S_in
    if improved and best_action == 'ACTIVATE':
        new_pts = [p for p in best_P if p.get('_activated_from_suppressed')]
        if new_pts:
            act_pt = new_pts[0]
            S_out  = [s for s in S_in
                      if not (abs(s['cx'] - act_pt['cx']) < X_APPROX
                              and abs(s['cy'] - act_pt['cy']) < X_APPROX)]
            for p in best_P:
                p.pop('_activated_from_suppressed', None)

    # Final post-perturbation NMS on the accepted point set
    # (split('+') so the cascade variants REPLACE+TAIL / DELETE+TAIL are treated
    #  like their base action here.)
    if improved and nms_window is not None and \
            best_action.split('+')[0] in ('ADD', 'ACTIVATE', 'REPLACE'):
        best_P, S_out = post_perturbation_nms(best_P, S_out, nms_window)

    p_in_ids  = {id(p) for p in P_in}
    p_out_ids = {id(p) for p in best_P}
    added   = [p for p in best_P if id(p) not in p_in_ids]
    removed = [p for p in P_in   if id(p) not in p_out_ids]

    # Rule ii: output render also uses thin lines for SSIM reporting.
    I_out = render_from_points(best_P, (H, W))   # lw=1

    return {
        'converged':        False,
        'P_in':             P_in,
        'P_out':            best_P,
        'S_in':             S_in,
        'S_out':            S_out,
        'sup_near':         sup_near,
        'I_t':              I_t,
        'I_out':            I_out,
        'L_t':              L_t,
        'l_star':           best_l_star,
        'l_star_diff':      best_l_diff,
        'l_star_strategy':  l_star_strategy,
        'anchor_left':      anchor_left,
        'anchor_right':     anchor_right,
        'candidates':       candidates,
        'added':            added,
        'removed':          removed,
        'action':           best_action,
        'best_dist':        best_d,
        'baseline':         baseline,
        'improved':         improved,
        'n_valid_perturbs': len(valid_perturbs),
        'n_sup_near':       len(sup_near),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_det(ax, d, alpha=ALPHA_DET, ms_mult=1.0, zorder=5,
              override_color=None, override_mfc=None):
    cn    = d['class_name']
    color = override_color or SYM_COLORS.get(cn, 'gray')
    mcode, filled, ms_scale = SYM_MPL.get(cn, ('o', True, 1.0))
    ms  = BASE_MS * ms_scale * ms_mult
    mfc = (override_mfc if override_mfc is not None
           else (color if filled else 'white'))
    mec = override_color or ('#111111' if not filled else color)
    ax.plot(d['cx'], d['cy'], marker=mcode, color='white', mfc='white',
            mew=2.0, ms=ms + 2, linestyle='none', alpha=0.45, zorder=zorder - 1)
    ax.plot(d['cx'], d['cy'], marker=mcode, color=color, mfc=mfc,
            mew=1.2, ms=ms, linestyle='none', alpha=alpha, zorder=zorder)


def _draw_seg(ax, seg, color, lw=1.5, alpha=0.85, linestyle='-', zorder=3):
    ax.plot([seg[0], seg[2]], [seg[1], seg[3]],
            color=color, lw=lw, alpha=alpha, linestyle=linestyle,
            solid_capstyle='round', zorder=zorder)


def _legend_handles(P, S=None):
    seen = {}
    for d in list(P) + (list(S) if S else []):
        cn = d['class_name']
        if cn not in seen:
            color = SYM_COLORS.get(cn, 'gray')
            mcode, filled, _ = SYM_MPL.get(cn, ('o', True, 1.0))
            mfc = color if filled else 'white'
            lbl = cn if cn != 'suppressed' else 'suppressed (NMS)'
            seen[cn] = Line2D([0], [0], marker=mcode, color=color, mfc=mfc,
                              mew=1.2, ms=7, linestyle='none', label=lbl)
    return list(seen.values())


def _save_jpg(fig, path):
    tmp = path.replace('.jpg', '_tmp.png')
    fig.savefig(tmp, dpi=130, bbox_inches='tight', facecolor='#F8F8F8')
    plt.close(fig)
    PILImage.open(tmp).convert('RGB').save(path, 'JPEG', quality=92)
    os.remove(tmp)


# ---------------------------------------------------------------------------
# Per-iteration figure (5-panel)
# ---------------------------------------------------------------------------

def build_iter_figure(iter_n, P_in, S_in, state,
                      I0, I0_rgb, L_refined, out_path,
                      return_img=False):
    """Build and save the 5-panel per-iteration diagnostic figure.

    If return_img=True, also return the figure as a BGR numpy array
    (for in-memory GUI diagnostics) in addition to saving the JPEG.
    """
    I_t       = state['I_t']
    I_out     = state['I_out']
    l_star    = state['l_star']
    action    = state['action']
    best_dist = state['best_dist']

    I_t_rgb   = cv2.cvtColor(I_t,   cv2.COLOR_BGR2RGB)
    I_out_rgb = cv2.cvtColor(I_out, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(28, 6.5), facecolor='#F8F8F8')
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.88, bottom=0.10)

    for ax, lbl in zip(axes, ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.02, 1.02, lbl, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='right',
                color='#333333')
        ax.axis('off')

    # A: I0 + P_in + S_in + L_refined
    ax = axes[0]
    ax.imshow(I0_rgb)
    for seg in L_refined:
        _draw_seg(ax, seg, '#00AA44', lw=1.0, alpha=0.50, zorder=2)
    for d in S_in:
        _plot_det(ax, d, alpha=0.35, ms_mult=0.8, zorder=3)
    for d in P_in:
        _plot_det(ax, d, zorder=5)
    ax.set_title(f'A -- I_0 + P_in + S_in + L_ref\n'
                 f'{len(P_in)} active  {len(S_in)} suppressed  {len(L_refined)} segs',
                 fontsize=9, pad=4)

    # B: I_t + L_t + l*
    ax = axes[1]
    ax.imshow(I_t_rgb)
    for seg in state['L_t']:
        _draw_seg(ax, seg, '#4488CC', lw=0.9, alpha=0.55, zorder=2)
    lx = float(min(l_star[0], l_star[2]))
    ly = float(l_star[1] if l_star[0] <= l_star[2] else l_star[3])
    rx = float(max(l_star[0], l_star[2]))
    ry = float(l_star[3] if l_star[0] <= l_star[2] else l_star[1])
    ax.plot([lx, rx], [ly, ry], color='#CC2200', lw=2.5, alpha=0.95, zorder=6,
            solid_capstyle='round')
    ax.plot([lx, rx], [ly, ry], color='#CC2200', lw=7, alpha=0.12, zorder=5)
    strat = state.get('l_star_strategy', 'current')
    strat_str = '(current: L_0->I_t)' if strat == 'current' else f'(fallback: L_t->I_0 {strat})'
    ax.set_title(f'B -- I_t + L_t + l* {strat_str}\ndelta-SSIM={state["l_star_diff"]:.5f}',
                 fontsize=9, pad=4)

    # C: Candidates + suppressed near anchors
    ax = axes[2]
    ax.imshow(I_t_rgb)
    for anchor in [state['anchor_left'], state['anchor_right']]:
        circ = plt.Circle(anchor, PT_TOL, color='#FF8800', fill=False,
                          lw=1.5, alpha=0.75, zorder=4)
        ax.add_patch(circ)
        ax.plot(*anchor, 'D', ms=5, color='#FF8800', mew=0, alpha=0.9, zorder=5)
    for d in state['candidates']:
        _plot_det(ax, d, override_color='#CC2200', ms_mult=1.4, zorder=6)
    for d in state['sup_near']:
        ax.plot(d['cx'], d['cy'], 'D', ms=7, color='#9900CC',
                mew=1.5, mfc='none', alpha=0.85, zorder=7)
        ax.annotate(d.get('original_class_name', '?')[:4],
                    (d['cx'], d['cy']), fontsize=5.5, color='#9900CC',
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', zorder=8)
    ax.set_title(f'C -- Candidates\n{len(state["candidates"])} active  '
                 f'{state["n_sup_near"]} suppressed near anchors  '
                 f'{state["n_valid_perturbs"]} valid perturbs',
                 fontsize=9, pad=4)

    # D: Action taken
    ax = axes[3]
    ax.imshow(I_t_rgb)
    for d in P_in:
        _plot_det(ax, d, alpha=0.25, zorder=3)
    for d in state['removed']:
        ax.plot(d['cx'], d['cy'], 'x', ms=12, mew=2.5, color='#CC3311',
                alpha=0.95, zorder=7)
    for d in state['added']:
        ax.plot(d['cx'], d['cy'], '*', ms=13,
                color=ACTION_COLORS.get(action, '#009944'),
                mew=0, alpha=0.95, zorder=7)
        _plot_det(ax, d, ms_mult=1.5, zorder=8)
    act_col = ACTION_COLORS.get(action, '#888888')
    ax.set_title(f'D -- Action: {action}\n'
                 f'added={len(state["added"])}  removed={len(state["removed"])}',
                 fontsize=9, pad=4, color=act_col, fontweight='bold')

    # E: After-state
    ax = axes[4]
    ax.imshow(I_out_rgb)
    for seg in L_refined:
        _draw_seg(ax, seg, '#00AA44', lw=1.0, alpha=0.45, zorder=2)
    for d in state['S_out']:
        _plot_det(ax, d, alpha=0.30, ms_mult=0.8, zorder=3)
    for d in state['P_out']:
        _plot_det(ax, d, zorder=5)
    ax.set_title(f'E -- I_out + P_out + S_out + L_ref\n'
                 f'{len(state["P_out"])} active  {len(state["S_out"])} suppressed  '
                 f'1-SSIM={best_dist:.5f}',
                 fontsize=9, pad=4)

    handles = _legend_handles(state['P_out'], state['S_out'])
    fig.legend(handles=handles, ncol=min(len(handles), 9),
               fontsize=7.5, loc='lower center',
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.5, columnspacing=0.8)

    strat_label = state.get('l_star_strategy', 'current')
    strat_str2  = ('l*: L_0->I_t' if strat_label == 'current'
                   else f'l*: L_t->I_0 ({strat_label})')
    plt.suptitle(
        f'Greedy Correction -- Iteration {iter_n}  |  '
        f'Action: {action}  |  '
        f'{strat_str2}  delta-SSIM={state["l_star_diff"]:.5f}  |  '
        f'{len(state["candidates"])} active cands  '
        f'{state["n_sup_near"]} suppressed cands  |  '
        f'{state["n_valid_perturbs"]} valid perturbs  |  '
        f'1-SSIM: {best_dist:.5f}',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    if return_img:
        # Render to numpy array before saving
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_bgr = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        img_bgr = None

    _save_jpg(fig, out_path)
    print(f'    Saved: {out_path}')

    return img_bgr


# ---------------------------------------------------------------------------
# Convergence curve
# ---------------------------------------------------------------------------

def build_convergence_figure(history, plot_name, out_path):
    """Plot and save the 1-SSIM convergence curve."""
    ts    = [h[0] for h in history]
    dists = [h[3] for h in history]
    acts  = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#FAFAFA')
    ax.plot(ts, dists, 'o-', color='#0077BB', lw=2.2, ms=8, zorder=4)
    for t_i, d_i, a_i in zip(ts, dists, acts):
        c = ACTION_COLORS.get(a_i, '#888888')
        ax.plot(t_i, d_i, 'o', ms=10, color=c, zorder=5)
        ax.annotate(f'{a_i}\n{d_i:.4f}',
                    xy=(t_i, d_i), xytext=(0, 14), textcoords='offset points',
                    ha='center', fontsize=7.5, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='#AAAAAA', alpha=0.88))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('1 - SSIM  (lower = better)', fontsize=12)
    ax.set_title(f'{plot_name} -- SSIM convergence curve',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(ts)
    ax.grid(True, alpha=0.3)
    leg = [Line2D([0], [0], marker='o', color=c, ms=9, mew=0,
                  linestyle='none', label=a)
           for a, c in ACTION_COLORS.items()]
    ax.legend(handles=leg, fontsize=9, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    _save_jpg(fig, out_path)
    print(f'    Convergence curve: {out_path}')


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def build_combined_figure(iter_paths, out_path):
    """Stack all per-iteration figures vertically into one JPEG."""
    imgs = [cv2.imread(p) for p in iter_paths if os.path.exists(p)]
    if not imgs:
        return
    max_w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[1] < max_w:
            pad = np.full((im.shape[0], max_w - im.shape[1], 3), 245, dtype=np.uint8)
            im  = np.hstack([im, pad])
        padded.append(im)
    combined = np.vstack(padded)
    PILImage.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)).save(
        out_path, 'JPEG', quality=92)
    print(f'    Combined figure: {out_path}')


# ---------------------------------------------------------------------------
# Full pipeline for one plot
# ---------------------------------------------------------------------------

def run_pipeline(plot_name, img_path, out_dir,
                 model_path, detector_py_path,
                 known_classes=None,
                 mode_xs=None,
                 prep_info=None,
                 return_diag_imgs=False,
                 max_iters=None,
                 d_override=None,
                 init_points=None,
                 init_suppressed=None,
                 segments_override=None,
                 grid_xs_override=None):
    """
    Run the full 4-stage pipeline for a single chart image.

    Parameters
    ----------
    plot_name        : str
        Human-readable name for the plot (used in figure titles).
    img_path         : str
        Path to the input image.
    out_dir          : str
        Directory for output figures.
    model_path       : str
        Path to chart_marker_net_v3.pth.
    detector_py_path : str
        Path to 1_point_detection_v3.py.
    known_classes    : list of str, optional
        Symbol class names to keep (default: KNOWN).
    mode_xs          : array-like or None
        KDE mode x-positions from Stage 1b.  If provided, used directly as
        the x-grid for segment refinement (skips k-means infer_x_grid).
        If None, infer_x_grid() is called on the detected points.

    Returns
    -------
    history : list of (iter, action, n_pts, 1-SSIM, improved)
    """
    global X_APPROX   # override at runtime based on grid step

    if known_classes is None:
        known_classes = KNOWN

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  Plot: {plot_name}")
    print(f"{'='*65}")

    # Load detector module (only needed when we actually run a detection; with
    # init_points supplied the correction is detector-free, which lets the colour
    # pipeline drive it without torch).
    if init_points is None and detector_py_path:
        _load_module('chart_marker_detector_v3', detector_py_path)

    # Add deliverable directory to path so sibling modules are importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    _seg_det_mod = _load_module('segment_detector',
                               os.path.join(_this_dir, '3_segment_detection_v2.py'))
    _refine_mod  = _load_module('refine_segments',
                               os.path.join(_this_dir, '4_segment_refinement.py'))
    detect_debug  = _seg_det_mod.detect_debug
    infer_x_grid  = _refine_mod.infer_x_grid
    refine        = _refine_mod.refine
    grid_step     = _refine_mod.grid_step

    # Stage 1b: adaptive NMS detection (skipped entirely when continuing from a
    # supplied point set -- keeps this module importable without torch)
    detect_with_adaptive_nms = None
    if init_points is None:
        _adnms_path = os.path.join(_this_dir, '2_point_detection_adaptive_nms_v2.py')
        _adnms_mod  = _load_module('adaptive_nms_detection_v2', _adnms_path)
        detect_with_adaptive_nms = _adnms_mod.detect_with_adaptive_nms

    # -- Preprocessing: run if not already done externally -----------------
    # If prep_info was passed from pipeline.py, reuse it.
    # Otherwise, try to run preprocessing here (for standalone Stage 4 use).
    if prep_info is None:
        try:
            from chart_preprocessing import preprocess as _cp
            _img_for_prep = cv2.imread(img_path)
            prep_info = _cp(_img_for_prep, verbose=False)
            print('  Preprocessing (axes/legend/text) applied internally.')
        except ImportError:
            pass  # chart_preprocessing not available; proceed without it

    if init_points is not None:
        # Continue from a previous correction: reuse its point/suppressed sets
        # instead of re-detecting, so successive Step-5 runs compose instead of
        # restarting from a fresh (and non-deterministic) detection.
        P0 = [dict(p) for p in init_points]
        S0 = [dict(p) for p in (init_suppressed or [])]
        print(f'  [continue] starting from previous state: '
              f'{len(P0)} active, {len(S0)} suppressed (detection skipped)')
        if mode_xs is not None:
            mode_xs = np.asarray(mode_xs)
        else:
            mode_xs = np.array([])
    else:
        print('  Detecting markers (adaptive NMS) ...')
        det_result = detect_with_adaptive_nms(
            img_path         = img_path,
            model_path       = model_path,
            known_classes    = known_classes,
            detector_py_path = detector_py_path,
            prep_info        = prep_info,
            d_override       = d_override,
        )
        P0 = det_result['kept']
        S0 = det_result['suppressed']
        if mode_xs is None:
            mode_xs = det_result['mode_xs']
        else:
            mode_xs = np.asarray(mode_xs)

    # -- Legend-area exclusion: remove any point inside the legend box ------
    _legend_box = prep_info.get('legend_box', None) if prep_info else None
    if _legend_box is not None:
        print(f'  legend_box = {_legend_box}  -> filtering P0 and S0')
        P0 = _filter_legend(P0, _legend_box)
        S0 = _filter_legend(S0, _legend_box)
    # Constrain to the plot area too: drop any detection outside it (axis labels,
    # border, side text). Applied to the initial sets AND to the final output.
    # Use the USER's dragged plot area (tight), NOT prep_info['plot_area'] which
    # is expanded by AXIS_MARGIN and would let border/margin content through.
    _plot_area_f = None
    if prep_info:
        _plot_area_f = prep_info.get('user_plot_area', None) or prep_info.get('plot_area', None)
    if _plot_area_f is not None:
        print(f'  plot_area = {_plot_area_f}  -> filtering P0 and S0 to inside')
        P0 = _filter_plot_area(P0, _plot_area_f)
        S0 = _filter_plot_area(S0, _plot_area_f)
    # Restrict ALL SSIM comparisons to the plot area so out-of-plot border/text
    # can't drive the greedy step (previously 57-63% of the residual was outside).
    global _SSIM_CROP
    _SSIM_CROP = _plot_area_f
    if _plot_area_f is not None:
        print(f'  [correction] SSIM restricted to plot area {_plot_area_f}')
    # Store for use in run_one_iteration candidate filtering
    _LEGEND_BOX = _legend_box

    cnt = {k: sum(1 for d in P0 if d['class_name'] == k) for k in known_classes}
    print(f"  P0 (active):     {len(P0)} pts  "
          + '  '.join(f'{k}={v}' for k, v in cnt.items() if v > 0))
    print(f"  S0 (suppressed): {len(S0)} pts")

    # Stage 2+3: segment detection & refinement
    I0     = cv2.imread(img_path)
    # Use an axis/text-REMOVED reference for SSIM. The point render can only draw
    # markers + data segments, never axis lines/labels; if those remain in I0 the
    # greedy step adds spurious markers near the axis trying to "explain" that ink.
    # clean_fn removes axis/ticks/labels/LLOQ while preserving markers & curves, so
    # we white out only the pixels it removes (same treatment as the detector).
    if prep_info is not None and prep_info.get('clean_fn', None) is not None:
        try:
            _g0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
            _, _bw0 = cv2.threshold(_g0, 128, 255, cv2.THRESH_BINARY_INV)
            _bw0 = (_bw0 > 0).astype('uint8')
            _cm0 = prep_info['clean_fn'](_bw0)
            _rm0 = (_bw0 > 0) & (np.asarray(_cm0) == 0)
            I0[_rm0] = (255, 255, 255)
            print(f"  [correction] SSIM reference cleaned (axis/text removed, {int(_rm0.sum())} px whited)")
        except Exception as _ce0:
            print(f"  [correction] SSIM reference clean skipped: {_ce0}")
    H, W   = I0.shape[:2]
    I0_rgb = cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)

    # -- [ssim-repro] Save the exact SSIM reference (cleaned I0) + the recipe, so
    #    any point set's 1-SSIM can be reproduced offline from the logged points.
    try:
        cv2.imwrite(os.path.join(out_dir, 'ssim_reference_I0.png'), I0)
        print(f"  [ssim-repro] saved ssim_reference_I0.png  (H={H}, W={W})")
        print(f"  [ssim-repro] recipe: render points on WHITE ({H}x{W}); per class_idx draw a "
              f"BLACK polyline through points sorted by cx (cv2.line, lw=1), then markers; "
              f"crop BOTH render and I0 to plot_area={_SSIM_CROP}; to grayscale/255.0; "
              f"1-SSIM = 1 - structural_similarity(I0_crop, render_crop, data_range=1.0)")
    except Exception as _re:
        print(f"  [ssim-repro] reference save failed: {_re}")


    import time as _ctime
    _t_seg0 = _ctime.perf_counter()
    print('  Detecting & refining segments ...')
    # Pass prep_info so Stage 2 removes axis/legend/text noise from I0.
    # Note: detect_debug is called WITHOUT prep_info inside run_one_iteration
    # (for synthetic renders I_t), which is correct -- they are already clean.
    if segments_override is not None:
        # Caller-supplied segment set. The colour driver builds this from BOTH
        # the whole plot read as greyscale (complete line geometry, even where a
        # colour mask is fragmentary) and the curve's own colour mask.
        L0_raw = list(segments_override)
        print(f'  [segments] using {len(L0_raw)} caller-supplied segments '
              f'(detect_debug on I0 skipped)')
    else:
        seg_result = detect_debug(I0, prep_info=prep_info)
        L0_raw     = seg_result['segments']

    if grid_xs_override is not None and len(grid_xs_override) >= 2:
        grid_xs = list(grid_xs_override)
        print(f'  [grid] using {len(grid_xs)} caller-supplied x columns')
    elif mode_xs is not None and len(mode_xs) >= 2:
        grid_xs = list(mode_xs)
    else:
        grid_xs = infer_x_grid(P0)

    step         = grid_step(grid_xs)
    L_refined, _ = refine(L0_raw, grid_xs)
    print(f"  L_refined: {len(L_refined)} segs  (grid step={step:.1f}px)")

    # Override X_APPROX and compute NMS window based on actual grid step
    X_APPROX   = step * 0.5
    nms_window = step * 0.75
    print(f"  X_APPROX={X_APPROX:.1f}px  NMS_WINDOW={nms_window:.1f}px")

    # Stage 4: greedy SSIM correction
    # Rule ii: initial baseline render uses thin lines (lw=1) for fair SSIM.
    I_init       = render_from_points(P0, (H, W))   # lw=1
    initial_dist = ssim_dist(I0, I_init)
    print(f"  Initial 1-SSIM: {initial_dist:.5f}")

    # -- [ssim-repro] initial active set (P0) + full suppressed pool (S0) with
    #    symbol + coords. Together with ssim_reference_I0.png and the per-iteration
    #    [state-active] logs, ANY alternative action's 1-SSIM can be recomputed offline.
    def _fmt0(_d):
        return f"{_d.get('class_name','?')}@({_d.get('cx',-1):.0f},{_d.get('cy',-1):.0f})"
    print(f"  [state-active iter0 n={len(P0)}] " + "; ".join(_fmt0(d) for d in P0))
    print(f"  [suppressed-pool n={len(S0)}] " + "; ".join(_fmt0(d) for d in S0))

    _pa_resid = _plot_area_f
    if os.environ.get("POINT_DIAG", "1") != "0":
        save_ssim_residual(I0, P0, out_dir, 'initial', dist=initial_dist, plot_area=_pa_resid)

    P_current  = P0
    S_current  = S0
    history    = [(0, '--', len(P0), initial_dist, False)]
    iter_paths = []
    diag_imgs  = []   # list of {title, img_bgr} for GUI diagnostics

    _n_iters = int(max_iters) if max_iters else T_MAX
    print(f"  [correction] running up to {_n_iters} iteration(s) "
          f"({'user-specified' if max_iters else 'default T_MAX'}); stops early on convergence")
    _t_iters0 = _ctime.perf_counter()
    for t in range(1, _n_iters + 1):
        print(f"\n  -- Iteration {t} " + '-' * 40)
        _t_it0 = _ctime.perf_counter()
        state = run_one_iteration(P_current, S_current, I0, L_refined,
                                  known_classes, nms_window=nms_window,
                                  legend_box=_LEGEND_BOX)
        print(f"  [timing] iteration {t}: {_ctime.perf_counter() - _t_it0:.2f}s")

        if state['converged']:
            print(f"  Converged (no l* found) at iteration {t}.")
            break

        strat = state.get('l_star_strategy', 'current')
        print(f"  l* [{strat}] delta-SSIM={state['l_star_diff']:.5f}  "
              f"active_cands={len(state['candidates'])}  "
              f"sup_near={state['n_sup_near']}  "
              f"valid_perturbs={state['n_valid_perturbs']}")
        print(f"  Action: {state['action']}  "
              f"added={len(state['added'])}  removed={len(state['removed'])}  "
              f"1-SSIM={state['best_dist']:.5f}  "
              f"({'improved' if state['improved'] else 'no improvement'})")

        # -- Detailed action + candidate trace (symbol + coords) for offline analysis --
        def _fmt(_d):
            return f"{_d.get('class_name','?')}@({_d.get('cx',-1):.0f},{_d.get('cy',-1):.0f})"
        for _a in state.get('added', []):
            print(f"    [act-add]    {_fmt(_a)}")
        for _r in state.get('removed', []):
            print(f"    [act-remove] {_fmt(_r)}")
        # candidate pool that WAS considered this iteration (active near l* + suppressed near anchors)
        _cands = state.get('candidates', []); _supn = state.get('sup_near', [])
        if _cands:
            print(f"    [cand-active {len(_cands)}] " + "; ".join(_fmt(d) for d in _cands))
        if _supn:
            print(f"    [cand-supp {len(_supn)}] " + "; ".join(_fmt(d) for d in _supn))
        # l* segment endpoints (where candidates come from)
        _ls = state.get('l_star', None)
        if _ls is not None:
            try:
                (_lx0, _ly0), (_lx1, _ly1) = _ls[0], _ls[1]
                print(f"    [l-star] ({_lx0:.0f},{_ly0:.0f})-({_lx1:.0f},{_ly1:.0f})  strat={state.get('l_star_strategy','?')}")
            except Exception:
                pass
        # full ACTIVE set after this iteration -- lets any alternative be reproduced offline
        print(f"    [state-active iter{t} n={len(state['P_out'])}] " +
              "; ".join(_fmt(d) for d in state['P_out']))

        history.append((t, state['action'], len(state['P_out']),
                        state['best_dist'], state['improved']))

        out_path = os.path.join(out_dir, f'iter{t:02d}.jpg')
        _iter_img = build_iter_figure(t, P_current, S_current, state,
                                      I0, I0_rgb, L_refined, out_path,
                                      return_img=return_diag_imgs)
        iter_paths.append(out_path)
        if return_diag_imgs and _iter_img is not None:
            act_str = state['action']
            diag_imgs.append({
                'title': (f'Stage 5 -- Iter {t}  [{act_str}]  '
                          f'1-SSIM={state["best_dist"]:.4f}  '
                          f'({"improved" if state["improved"] else "no improvement"})'),
                'img_bgr': _iter_img,
            })

        P_current = state['P_out']
        S_current = state['S_out']

        if not state['improved']:
            print('  No improvement -- stopping.')
            break

    # Guarantee the corrected output stays inside the plot area and out of the
    # legend (the greedy step can add segment-endpoint candidates outside it).
    if _plot_area_f is not None:
        P_current = _filter_plot_area(P_current, _plot_area_f)
    if _legend_box is not None:
        P_current = _filter_legend(P_current, _legend_box)

    print(f"  [timing] setup (segments/refine/init): {_t_iters0 - _t_seg0:.2f}s  |  "
          f"{len(history) - 1} iteration(s) ran: {_ctime.perf_counter() - _t_iters0:.2f}s")

    final_dist = history[-1][3]
    print(f"\n  Final: {len(P_current)} active pts  {len(S_current)} suppressed  "
          f"|  1-SSIM: {final_dist:.5f}")

    if os.environ.get("POINT_DIAG", "1") != "0":
        save_ssim_residual(I0, P_current, out_dir, 'final', dist=final_dist,
                           plot_area=_pa_resid)

    build_convergence_figure(history, plot_name,
                             os.path.join(out_dir, 'convergence.jpg'))
    build_combined_figure(iter_paths,
                          os.path.join(out_dir, 'all_iterations_combined.jpg'))

    print(f"\n  Iteration log:")
    print(f"  {'Iter':>4}  {'Action':>10}  {'N pts':>6}  {'1-SSIM':>10}  {'Improved':>8}")
    for t_i, act, npts, dist, imp in history:
        print(f"  {t_i:>4}  {act:>10}  {npts:>6}  {dist:>10.5f}  {str(imp):>8}")

    if return_diag_imgs:
        return history, diag_imgs, P_current, S_current
    return history, P_current, S_current


# ---------------------------------------------------------------------------
# Convenience wrapper (single-image, no plot_name required)
# ---------------------------------------------------------------------------

def run_correction(img_path, model_path, detector_py_path,
                   known_classes=None, out_dir=None, mode_xs=None,
                   prep_info=None, return_diag_imgs=False, max_iters=None,
                   d_override=None, init_points=None, init_suppressed=None,
                   segments_override=None, grid_xs_override=None):
    """
    Convenience wrapper that runs the full pipeline on a single image.

    Parameters
    ----------
    prep_info : dict or None
        Preprocessing result from chart_preprocessing.preprocess().
        If provided, axis/legend/text/LLOQ noise removal is applied
        to Stage 1b and Stage 2 detections.

    Returns
    -------
    dict with keys:
        'history'    -- convergence history list
        'out_dir'    -- path to the output directory
        'P_current'  -- final corrected active point list (dicts with cx, cy, class_name, ...)
        'S_current'  -- final suppressed point list
    """
    if known_classes is None:
        known_classes = KNOWN
    if out_dir is None:
        out_dir = os.path.join('/tmp', 'correction_out',
                               os.path.splitext(os.path.basename(img_path))[0])

    plot_name = os.path.splitext(os.path.basename(img_path))[0]
    result = run_pipeline(
        plot_name        = plot_name,
        img_path         = img_path,
        out_dir          = out_dir,
        model_path       = model_path,
        detector_py_path = detector_py_path,
        known_classes    = known_classes,
        mode_xs          = mode_xs,
        prep_info        = prep_info,
        return_diag_imgs = return_diag_imgs,
        max_iters        = max_iters,
        d_override       = d_override,
        init_points       = init_points,
        init_suppressed   = init_suppressed,
        segments_override = segments_override,
        grid_xs_override  = grid_xs_override,
    )
    if return_diag_imgs:
        history, diag_imgs, P_current, S_current = result
        return {'history': history, 'out_dir': out_dir,
                'diag_imgs': diag_imgs,
                'P_current': P_current, 'S_current': S_current}
    history, P_current, S_current = result
    return {'history': history, 'out_dir': out_dir,
            'P_current': P_current, 'S_current': S_current}


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run the full chart-digitisation pipeline (Stages 1-4).')
    parser.add_argument('img_path',   help='Path to the input chart image')
    parser.add_argument('model_path', help='Path to chart_marker_net_v3.pth')
    parser.add_argument('--detector', default=None,
                        help='Path to 1_point_detection_v3.py '
                             '(auto-detected if in same directory)')
    parser.add_argument('--out_dir',  default=None,
                        help='Output directory  (default: /tmp/correction_out/<name>)')
    parser.add_argument('--classes',  nargs='+', default=None,
                        help='Known symbol class names  '
                             '(default: filled_circle open_circle '
                             'filled_square open_square)')
    args = parser.parse_args()

    # Auto-locate detector script in the same directory as this file
    detector_py = args.detector
    if detector_py is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '1_point_detection_v3.py')
        if os.path.isfile(candidate):
            detector_py = candidate

    result = run_correction(
        img_path         = args.img_path,
        model_path       = args.model_path,
        detector_py_path = detector_py,
        known_classes    = args.classes,
        out_dir          = args.out_dir,
    )
    print(f"\nOutput saved to: {result['out_dir']}")

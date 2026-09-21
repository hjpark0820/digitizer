"""
segment_detector.py
===================
Self-contained line-segment detector for scientific charts.

ALGORITHM
---------
  Step 1. Binarise          — threshold to get foreground (black) pixels
  Step 2. Directional probe — for each fg pixel, probe in 24 directions;
                              compute linearity = max_run / perp_run
  Step 3. Segment mask      — keep pixels with linearity >= threshold
                              (rejects marker blobs)
  Step 4. Direction cluster — group segment pixels by direction + connectivity
  Step 5+6. Group→Fit→Extend (single non-iterative step):
              a. Compute PCA segment per cluster (raw fits)
              b. Union-find: group collinear raw segments
              c. Validate groups (anti-chain-drift): use longest member as
                 reference; reject members whose angle or perpendicular
                 distance drifts beyond tolerance
              d. For each validated group, collect ALL pixels → fit ONE PCA
              e. Extend each group-fitted segment along the foreground once
  Step 7. Deduplication     — remove sub-segments that are truly contained
                              within a longer same-line segment
                              (parallel segments are NEVER removed)

USAGE (command line)
--------------------
  # Detect segments in a single image and save visualisation:
  python segment_detector.py --image path/to/chart.png

  # Save results to a specific output directory:
  python segment_detector.py --image path/to/chart.png --out-dir ./results

  # Print detected segments as JSON to stdout (no visualisation):
  python segment_detector.py --image path/to/chart.png --json

  # Tune parameters:
  python segment_detector.py --image chart.png \\
      --binary-thresh 180 \\
      --probe-radius 12 \\
      --linearity-thresh 2.0 \\
      --perp-tol 2.0 \\
      --gap-tol 8.0 \\
      --angle-tol 8.0 \\
      --max-gap 4

PYTHON API
----------
  from segment_detector import detect, detect_debug

  # Simple: returns list of (x1, y1, x2, y2) tuples
  import cv2
  img = cv2.imread('chart.png')
  segments = detect(img)

  # Debug: returns dict with all intermediate results
  result = detect_debug(img)
  segments   = result['segments']       # final (x1,y1,x2,y2) list
  fg_mask    = result['fg_mask']        # Step 1 binary mask
  seg_mask   = result['seg_mask']       # Step 3 linearity mask
  clusters   = result['clusters']       # Step 4 pixel clusters
  segs_raw   = result['segments_raw']   # Step 5a raw PCA segments
  segs_group = result['segments_grouped'] # Step 5d group-fitted segments
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS  (all overridable via detect() kwargs or CLI flags)
# ══════════════════════════════════════════════════════════════════════════════
BINARY_THRESH    = 180    # global threshold (0-255); pixels darker than this are fg
PROBE_RADIUS     = 12     # directional probe half-length in pixels
N_DIRECTIONS     = 24     # number of probe directions (evenly spaced over 180°)
LINEARITY_THRESH = 2.0    # min linearity score to classify pixel as segment
MIN_RUN          = 3      # min run length in best direction
DIR_BIN_TOLERANCE = 1     # direction bin tolerance for clustering (bins of 7.5°)
MIN_CLUSTER_PX   = 3      # min pixels per cluster
MIN_SEG_LEN      = 5.0    # min segment length (px)
MIN_ELONGATION   = 1.5    # min PCA eigenvalue ratio for a cluster to be a line

# Merge tolerances
MERGE_PERP_TOL   = 2.0    # max perpendicular distance to merge two segments (px)
MERGE_GAP_TOL    = 8.0    # max gap along line direction to merge (px)
MERGE_ANGLE_TOL  = 8.0    # max angle difference to merge (degrees)

# Extension
EXTEND_MAX_GAP   = 4      # max consecutive background pixels during extension

# Deduplication
DEDUP_DIST_TOL         = 2.0   # max perp distance to consider same-line duplicate (px)
DEDUP_ANGLE_TOL        = 12.0  # max angle difference for dedup
DEDUP_CONTAINMENT      = 0.85  # min fraction of B inside A to remove B
DEDUP_EXTENSION_RATIO  = 1.5   # A must be >= this × longer than B


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: BINARISE
# ══════════════════════════════════════════════════════════════════════════════
def _binarise(img_bgr: np.ndarray, thresh: int = BINARY_THRESH) -> np.ndarray:
    """Return binary foreground mask (1 = dark/foreground, 0 = background)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=15)
    _, global_bin = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(adaptive, global_bin)
    return (combined > 0).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: DIRECTIONAL LINE PROBING
# ══════════════════════════════════════════════════════════════════════════════
def _make_line_kernels(n_dirs: int, radius: int) -> list[np.ndarray]:
    """Create (2*radius+1)×(2*radius+1) line kernels for each of n_dirs directions."""
    kernels = []
    size = 2 * radius + 1
    centre = radius
    for i in range(n_dirs):
        angle = math.pi * i / n_dirs
        kernel = np.zeros((size, size), dtype=np.float32)
        dx = math.cos(angle)
        dy = math.sin(angle)
        for t_sign in [1, -1]:
            for t in range(radius + 1):
                x = int(round(centre + t_sign * t * dx))
                y = int(round(centre + t_sign * t * dy))
                if 0 <= x < size and 0 <= y < size:
                    kernel[y, x] = 1.0
        kernels.append(kernel)
    return kernels


def _directional_runs(fg_mask: np.ndarray, n_dirs: int, radius: int):
    """
    For each foreground pixel, compute the number of foreground pixels within
    radius steps in each of n_dirs directions.

    Returns:
      max_run      (H, W) — maximum run count across all directions
      best_dir_idx (H, W) — direction index with the maximum run
      all_runs     (H, W, n_dirs) — run count per direction
    """
    fg = fg_mask.astype(np.float32)
    kernels = _make_line_kernels(n_dirs, radius)
    H, W = fg.shape
    all_runs = np.zeros((H, W, n_dirs), dtype=np.float32)
    for i, kernel in enumerate(kernels):
        all_runs[:, :, i] = cv2.filter2D(
            fg, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
    max_run = all_runs.max(axis=2)
    best_dir_idx = all_runs.argmax(axis=2)
    return max_run, best_dir_idx, all_runs


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: CLASSIFY PIXELS → SEGMENT MASK
# ══════════════════════════════════════════════════════════════════════════════
def _classify_pixels(fg_mask, max_run, best_dir_idx, all_runs,
                     linearity_thresh, min_run, n_dirs) -> tuple[np.ndarray, np.ndarray]:
    """
    A foreground pixel is a 'segment pixel' if:
      linearity = max_run / (perp_run + 1) >= linearity_thresh
      AND max_run >= min_run

    perp_run = run count in the direction perpendicular to best_dir.
    High linearity → pixel lies on a thin line.
    Low linearity  → pixel is part of a blob (marker).
    """
    H, W = fg_mask.shape
    perp_idx = (best_dir_idx + n_dirs // 2) % n_dirs
    rows = np.arange(H)[:, None].repeat(W, axis=1)
    cols = np.arange(W)[None, :].repeat(H, axis=0)
    perp_run = all_runs[rows, cols, perp_idx]
    linearity = max_run / (perp_run + 1.0)
    seg_mask = ((fg_mask > 0) &
                (max_run >= min_run) &
                (linearity >= linearity_thresh)).astype(np.uint8)
    return seg_mask, linearity


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: DIRECTION-BASED CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def _cluster_by_direction(seg_mask, best_dir_idx, n_dirs,
                          dir_bin_tol, min_cluster_px):
    """
    Group segment pixels into clusters based on direction similarity and
    spatial connectivity (8-connectivity).

    For each direction bin d, create a mask of pixels whose best_dir is within
    dir_bin_tol bins of d, then find connected components.

    Returns:
      clusters      — list of (N, 2) arrays of (y, x) pixel coordinates
      cluster_labels — (H, W) label image
    """
    H, W = seg_mask.shape
    cluster_labels = np.zeros((H, W), dtype=np.int32)
    clusters = []
    next_label = 1
    assigned = np.zeros((H, W), dtype=bool)

    for d in range(n_dirs):
        dir_mask = np.zeros((H, W), dtype=np.uint8)
        for offset in range(-dir_bin_tol, dir_bin_tol + 1):
            target_dir = (d + offset) % n_dirs
            dir_mask |= (best_dir_idx == target_dir).astype(np.uint8)

        combined = (seg_mask > 0) & (dir_mask > 0) & (~assigned)
        if combined.sum() == 0:
            continue

        n_labels, labels = cv2.connectedComponents(
            combined.astype(np.uint8), connectivity=8)

        for label_id in range(1, n_labels):
            component_mask = (labels == label_id)
            if component_mask.sum() < min_cluster_px:
                continue
            ys, xs = np.where(component_mask)
            clusters.append(np.column_stack([ys, xs]))
            cluster_labels[component_mask] = next_label
            assigned[component_mask] = True
            next_label += 1

    return clusters, cluster_labels


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _seg_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))

def _angle_diff(a1, a2):
    d = abs(a1 - a2)
    return 180.0 - d if d > 90.0 else d

def _point_to_line_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-9:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length

def _project_on_line(px, py, x1, y1, dx, dy, len_sq):
    if len_sq < 1e-9:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / len_sq

def _seg_len(seg):
    if seg is None:
        return 0.0
    return math.hypot(seg[2] - seg[0], seg[3] - seg[1])


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5a: PCA SEGMENT FIT PER CLUSTER
# ══════════════════════════════════════════════════════════════════════════════
def _fit_pca_segment(cluster: np.ndarray,
                     min_len: float = MIN_SEG_LEN,
                     min_elongation: float = MIN_ELONGATION):
    """
    Fit a line segment to a cluster of (y, x) pixels using PCA.

    Returns (x1, y1, x2, y2) or None if the cluster is too short/round.
    """
    ys = cluster[:, 0].astype(float)
    xs = cluster[:, 1].astype(float)
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy

    cov_xx = (dx * dx).mean()
    cov_yy = (dy * dy).mean()
    cov_xy = (dx * dy).mean()

    trace = cov_xx + cov_yy
    det   = cov_xx * cov_yy - cov_xy * cov_xy
    disc  = max(0.25 * trace * trace - det, 0.0)
    sqrt_disc = math.sqrt(disc)

    lam1 = 0.5 * trace + sqrt_disc
    lam2 = 0.5 * trace - sqrt_disc

    if lam1 < 1e-6:
        return None
    if lam1 / (lam2 + 1e-6) < min_elongation:
        return None

    if abs(cov_xy) > 1e-9:
        vx = lam1 - cov_yy
        vy = cov_xy
    elif cov_xx >= cov_yy:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0

    norm = math.sqrt(vx * vx + vy * vy) + 1e-12
    vx /= norm
    vy /= norm

    proj = dx * vx + dy * vy
    t_min, t_max = proj.min(), proj.max()

    x1 = cx + t_min * vx
    y1 = cy + t_min * vy
    x2 = cx + t_max * vx
    y2 = cy + t_max * vy

    if math.hypot(x2 - x1, y2 - y1) < min_len:
        return None

    return (float(x1), float(y1), float(x2), float(y2))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5b: COLLINEARITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
def _are_collinear(seg_a, seg_b, perp_tol, gap_tol, angle_tol):
    """
    Two segments are collinear if:
      - angle difference < angle_tol
      - all 4 endpoints within perp_tol of the longer segment's line
      - gap along line direction < gap_tol
    """
    ax1, ay1, ax2, ay2 = seg_a
    bx1, by1, bx2, by2 = seg_b

    aa = _seg_angle(ax1, ay1, ax2, ay2)
    ba = _seg_angle(bx1, by1, bx2, by2)
    if _angle_diff(aa, ba) > angle_tol:
        return False

    a_len = math.hypot(ax2 - ax1, ay2 - ay1)
    b_len = math.hypot(bx2 - bx1, by2 - by1)
    if a_len < 1e-6 or b_len < 1e-6:
        return False

    ref = seg_a if a_len >= b_len else seg_b
    rx1, ry1, rx2, ry2 = ref

    d1 = _point_to_line_dist(ax1, ay1, rx1, ry1, rx2, ry2)
    d2 = _point_to_line_dist(ax2, ay2, rx1, ry1, rx2, ry2)
    d3 = _point_to_line_dist(bx1, by1, rx1, ry1, rx2, ry2)
    d4 = _point_to_line_dist(bx2, by2, rx1, ry1, rx2, ry2)
    if max(d1, d2, d3, d4) > perp_tol:
        return False

    rdx = rx2 - rx1
    rdy = ry2 - ry1
    r_len_sq = rdx * rdx + rdy * rdy

    ta1 = _project_on_line(ax1, ay1, rx1, ry1, rdx, rdy, r_len_sq)
    ta2 = _project_on_line(ax2, ay2, rx1, ry1, rdx, rdy, r_len_sq)
    tb1 = _project_on_line(bx1, by1, rx1, ry1, rdx, rdy, r_len_sq)
    tb2 = _project_on_line(bx2, by2, rx1, ry1, rdx, rdy, r_len_sq)

    a_lo, a_hi = min(ta1, ta2), max(ta1, ta2)
    b_lo, b_hi = min(tb1, tb2), max(tb1, tb2)

    ref_len = math.sqrt(r_len_sq)
    gap = (max(a_lo, b_lo) - min(a_hi, b_hi)) * ref_len
    return gap <= gap_tol


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5c: ANTI-CHAIN-DRIFT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def _validate_and_split_group(members, raw_segments, perp_tol, angle_tol, gap_tol):
    """
    Validate a union-find group against chain-drift.

    Uses the LONGEST member as the reference line and checks that every other
    member satisfies:
      1. angle within angle_tol of reference
      2. both endpoints within perp_tol of reference line
      3. gap along reference < gap_tol

    Members that fail become singletons.
    Returns a list of sub-groups (each is a list of cluster indices).
    """
    if len(members) <= 1:
        return [members]

    lengths = [_seg_len(raw_segments[m]) for m in members]
    longest_idx = lengths.index(max(lengths))
    ref_member = members[longest_idx]
    ref_seg = raw_segments[ref_member]
    ref_angle = _seg_angle(*ref_seg)

    ref_dx = ref_seg[2] - ref_seg[0]
    ref_dy = ref_seg[3] - ref_seg[1]
    ref_len_sq = ref_dx * ref_dx + ref_dy * ref_dy
    ref_len = math.sqrt(ref_len_sq)

    valid_members = [ref_member]
    outliers = []

    for i, m in enumerate(members):
        if i == longest_idx:
            continue
        seg = raw_segments[m]

        # 1. Angle
        if _angle_diff(_seg_angle(*seg), ref_angle) > angle_tol:
            outliers.append(m)
            continue

        # 2. Perpendicular distance
        d1 = _point_to_line_dist(seg[0], seg[1], *ref_seg)
        d2 = _point_to_line_dist(seg[2], seg[3], *ref_seg)
        if max(d1, d2) > perp_tol:
            outliers.append(m)
            continue

        # 3. Gap along reference
        if ref_len_sq > 1e-6:
            t1 = _project_on_line(seg[0], seg[1], ref_seg[0], ref_seg[1],
                                  ref_dx, ref_dy, ref_len_sq)
            t2 = _project_on_line(seg[2], seg[3], ref_seg[0], ref_seg[1],
                                  ref_dx, ref_dy, ref_len_sq)
            cand_lo, cand_hi = min(t1, t2), max(t1, t2)
            gap = max(0.0, max(cand_lo - 1.0, 0.0 - cand_hi)) * ref_len
            if gap > gap_tol:
                outliers.append(m)
                continue

        valid_members.append(m)

    result = [valid_members]
    for m in outliers:
        result.append([m])
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5+6: GROUP → FIT → EXTEND  (single non-iterative step)
# ══════════════════════════════════════════════════════════════════════════════
def _group_fit_extend(
    clusters: list[np.ndarray],
    fg_mask: np.ndarray,
    seg_mask: np.ndarray = None,
    perp_tol: float = MERGE_PERP_TOL,
    gap_tol: float = MERGE_GAP_TOL,
    angle_tol: float = MERGE_ANGLE_TOL,
    max_gap: int = EXTEND_MAX_GAP,
    min_seg_len: float = MIN_SEG_LEN,
    min_elongation: float = MIN_ELONGATION,
) -> tuple[list[tuple], dict]:
    """
    Single non-iterative merge + extend step.

    1. PCA per cluster → raw segments
    2. Union-find: group collinear raw segments (CC constraint from seg_mask)
    3. Anti-chain-drift validation: use longest member as reference
    4. For each group, collect all pixels → fit ONE combined PCA
    5. Extend each group-fitted segment along the foreground once

    Parallel segments (same angle, different perpendicular position) are
    NEVER merged — the perpendicular tolerance (perp_tol=2px) ensures this.

    Returns (segments_final, debug_info).
    """
    n = len(clusters)
    if n == 0:
        return [], {"segments_raw": [], "groups": {}, "segments_grouped": []}

    # ── 1. PCA per cluster ──────────────────────────────────────────────────
    raw_segments = []
    valid_indices = []
    for i, cluster in enumerate(clusters):
        seg = _fit_pca_segment(cluster, min_seg_len, min_elongation)
        raw_segments.append(seg)
        if seg is not None:
            valid_indices.append(i)

    # ── 2. CC constraint from seg_mask ──────────────────────────────────────
    cc_labels = None
    if seg_mask is not None:
        dilated = cv2.dilate(seg_mask, np.ones((3, 3), np.uint8), iterations=1)
        _, cc_labels = cv2.connectedComponents(dilated, connectivity=8)

    def _get_cc(x, y):
        if cc_labels is None:
            return 1
        H, W = cc_labels.shape
        ix, iy = int(round(x)), int(round(y))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < H and 0 <= nx < W and cc_labels[ny, nx] > 0:
                    return cc_labels[ny, nx]
        return 0

    seg_cc = {}
    for i in valid_indices:
        seg = raw_segments[i]
        mx = (seg[0] + seg[2]) / 2
        my = (seg[1] + seg[3]) / 2
        seg_cc[i] = _get_cc(mx, my)

    # ── 3. Union-find grouping ───────────────────────────────────────────────
    parent = {i: i for i in valid_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ii, i in enumerate(valid_indices):
        for jj in range(ii + 1, len(valid_indices)):
            j = valid_indices[jj]
            if cc_labels is not None:
                if seg_cc.get(i, 0) == 0 or seg_cc.get(j, 0) == 0:
                    continue
                if seg_cc[i] != seg_cc[j]:
                    continue
            if _are_collinear(raw_segments[i], raw_segments[j],
                               perp_tol, gap_tol, angle_tol):
                union(i, j)

    initial_groups = defaultdict(list)
    for i in valid_indices:
        initial_groups[find(i)].append(i)

    # ── 4. Anti-chain-drift validation ──────────────────────────────────────
    validated_groups = []
    for root, members in initial_groups.items():
        sub_groups = _validate_and_split_group(
            members, raw_segments, perp_tol, angle_tol, gap_tol)
        validated_groups.extend(sub_groups)

    groups = {sg[0]: sg for sg in validated_groups}

    # ── 5. Combined PCA per group ────────────────────────────────────────────
    segments_grouped = []
    for root, members in groups.items():
        all_pixels = np.vstack([clusters[i] for i in members])
        seg = _fit_pca_segment(all_pixels, min_seg_len, min_elongation=1.0)
        if seg is None:
            longest_i = max(members, key=lambda i: _seg_len(raw_segments[i]))
            seg = raw_segments[longest_i]
        segments_grouped.append(seg)

    # ── 6. Extend each group-fitted segment along the foreground ─────────────
    H, W = fg_mask.shape
    segments_final = []

    for (x1, y1, x2, y2) in segments_grouped:
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length

        # Extend backward from (x1, y1)
        t_neg = 0
        gap_count = 0
        for t in range(1, int(math.hypot(W, H)) + 1):
            nx = int(round(x1 - t * ux))
            ny = int(round(y1 - t * uy))
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                break
            if fg_mask[ny, nx] > 0:
                t_neg = t
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > max_gap:
                    break

        # Extend forward from (x2, y2)
        t_pos = 0
        gap_count = 0
        for t in range(1, int(math.hypot(W, H)) + 1):
            nx = int(round(x2 + t * ux))
            ny = int(round(y2 + t * uy))
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                break
            if fg_mask[ny, nx] > 0:
                t_pos = t
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > max_gap:
                    break

        ext_x1 = x1 - t_neg * ux
        ext_y1 = y1 - t_neg * uy
        ext_x2 = x2 + t_pos * ux
        ext_y2 = y2 + t_pos * uy

        if math.hypot(ext_x2 - ext_x1, ext_y2 - ext_y1) >= min_seg_len:
            segments_final.append((ext_x1, ext_y1, ext_x2, ext_y2))

    debug_info = {
        "segments_raw": [raw_segments[i] for i in valid_indices],
        "valid_indices": valid_indices,
        "groups": groups,
        "segments_grouped": segments_grouped,
    }
    return segments_final, debug_info


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════
def _deduplicate(
    segments: list[tuple],
    angle_tol: float = DEDUP_ANGLE_TOL,
    dist_tol: float = DEDUP_DIST_TOL,
    containment_thresh: float = DEDUP_CONTAINMENT,
    extension_ratio: float = DEDUP_EXTENSION_RATIO,
) -> list[tuple]:
    """
    Remove segment B only if it is truly CONTAINED within segment A AND both
    lie on the SAME physical line (perpendicular distance < dist_tol = 2px).

    Parallel segments (offset >= dist_tol) are NEVER removed.
    """
    if not segments:
        return segments

    def seg_len(s):
        return math.hypot(s[2] - s[0], s[3] - s[1])

    indexed = sorted(enumerate(segments), key=lambda x: seg_len(x[1]), reverse=True)
    keep = [True] * len(segments)

    for ii in range(len(indexed)):
        i, seg_a = indexed[ii]
        if not keep[i]:
            continue
        ax1, ay1, ax2, ay2 = seg_a
        aa = _seg_angle(ax1, ay1, ax2, ay2)
        a_len = seg_len(seg_a)
        if a_len < 1e-6:
            continue
        ddx, ddy = ax2 - ax1, ay2 - ay1
        len_sq = ddx * ddx + ddy * ddy

        for jj in range(ii + 1, len(indexed)):
            j, seg_b = indexed[jj]
            if not keep[j]:
                continue
            bx1, by1, bx2, by2 = seg_b

            if _angle_diff(aa, _seg_angle(bx1, by1, bx2, by2)) > angle_tol:
                continue

            d1 = _point_to_line_dist(bx1, by1, ax1, ay1, ax2, ay2)
            d2 = _point_to_line_dist(bx2, by2, ax1, ay1, ax2, ay2)
            if max(d1, d2) > dist_tol:
                continue

            t_b1 = _project_on_line(bx1, by1, ax1, ay1, ddx, ddy, len_sq)
            t_b2 = _project_on_line(bx2, by2, ax1, ay1, ddx, ddy, len_sq)
            t_lo = max(min(t_b1, t_b2), 0.0)
            t_hi = min(max(t_b1, t_b2), 1.0)
            overlap_len = max(t_hi - t_lo, 0.0) * a_len
            b_len = seg_len(seg_b)
            if b_len < 1e-6:
                keep[j] = False
                continue
            if overlap_len / b_len < containment_thresh:
                continue
            if a_len < extension_ratio * b_len:
                continue

            keep[j] = False

    return [segments[i] for i in range(len(segments)) if keep[i]]


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def detect_debug(img_bgr: np.ndarray, prep_info=None, **kwargs) -> dict:
    """
    Run the full pipeline and return all intermediate results.

    Parameters (all optional, override defaults):
      binary_thresh    (int)   — binarisation threshold (default 180)
      probe_radius     (int)   — directional probe radius (default 12)
      n_dirs           (int)   — number of probe directions (default 24)
      linearity_thresh (float) — linearity threshold (default 2.0)
      min_run          (int)   — minimum run length (default 3)
      dir_bin_tol      (int)   — direction bin tolerance (default 1)
      min_cluster_px   (int)   — minimum cluster size (default 3)
      min_seg_len      (float) — minimum segment length (default 5.0)
      min_elongation   (float) — minimum PCA elongation (default 1.5)
      perp_tol         (float) — merge perpendicular tolerance (default 2.0)
      gap_tol          (float) — merge gap tolerance (default 8.0)
      angle_tol        (float) — merge angle tolerance (default 8.0)
      max_gap          (int)   — extension max gap (default 4)
    prep_info : dict or None
        Preprocessing result from chart_preprocessing.preprocess().
        If provided, the cleaned binary foreground mask (after axis/legend/
        text/LLOQ removal) is used instead of the raw binarised mask.
        Pass None for synthetic renders (Stage 4 correction iterations).

    Returns dict with keys:
      fg_mask, seg_mask, clusters, cluster_labels,
      segments_raw, segments_grouped, segments, (all intermediate stages)
    """
    p = dict(
        binary_thresh    = kwargs.get("binary_thresh",    BINARY_THRESH),
        probe_radius     = kwargs.get("probe_radius",     PROBE_RADIUS),
        n_dirs           = kwargs.get("n_dirs",           N_DIRECTIONS),
        linearity_thresh = kwargs.get("linearity_thresh", LINEARITY_THRESH),
        min_run          = kwargs.get("min_run",          MIN_RUN),
        dir_bin_tol      = kwargs.get("dir_bin_tol",      DIR_BIN_TOLERANCE),
        min_cluster_px   = kwargs.get("min_cluster_px",   MIN_CLUSTER_PX),
        min_seg_len      = kwargs.get("min_seg_len",      MIN_SEG_LEN),
        min_elongation   = kwargs.get("min_elongation",   MIN_ELONGATION),
        perp_tol         = kwargs.get("perp_tol",         MERGE_PERP_TOL),
        gap_tol          = kwargs.get("gap_tol",          MERGE_GAP_TOL),
        angle_tol        = kwargs.get("angle_tol",        MERGE_ANGLE_TOL),
        max_gap          = kwargs.get("max_gap",          EXTEND_MAX_GAP),
    )

    fg_mask = _binarise(img_bgr, p["binary_thresh"])
    # ── Apply preprocessing noise removal if available ───────────────────────
    # Only applied to the original chart (prep_info != None).
    # Synthetic renders (Stage 4 correction) are clean and use raw fg_mask.
    if prep_info is not None:
        clean_fn = prep_info.get('clean_fn', None)
        if clean_fn is not None:
            fg_mask = clean_fn(fg_mask)
    max_run, best_dir_idx, all_runs = _directional_runs(
        fg_mask, p["n_dirs"], p["probe_radius"])
    seg_mask, linearity = _classify_pixels(
        fg_mask, max_run, best_dir_idx, all_runs,
        p["linearity_thresh"], p["min_run"], p["n_dirs"])
    clusters, cluster_labels = _cluster_by_direction(
        seg_mask, best_dir_idx, p["n_dirs"],
        p["dir_bin_tol"], p["min_cluster_px"])
    segments_extended, dbg = _group_fit_extend(
        clusters, fg_mask, seg_mask,
        perp_tol    = p["perp_tol"],
        gap_tol     = p["gap_tol"],
        angle_tol   = p["angle_tol"],
        max_gap     = p["max_gap"],
        min_seg_len = p["min_seg_len"],
        min_elongation = p["min_elongation"],
    )
    segments = _deduplicate(segments_extended)

    return {
        "fg_mask":           fg_mask,
        "seg_mask":          seg_mask,
        "linearity":         linearity,
        "clusters":          clusters,
        "cluster_labels":    cluster_labels,
        "segments_raw":      dbg["segments_raw"],
        "segments_grouped":  dbg["segments_grouped"],
        "segments_extended": segments_extended,
        "segments":          segments,
    }


def detect(img_bgr: np.ndarray, prep_info=None, **kwargs) -> list[tuple]:
    """
    Detect line segments in a BGR chart image.

    Parameters
    ----------
    img_bgr   : BGR numpy array
    prep_info : dict or None
        Preprocessing result from chart_preprocessing.preprocess().
        If provided, axis/legend/text/LLOQ noise is removed before detection.

    Returns a list of (x1, y1, x2, y2) float tuples.
    """
    return detect_debug(img_bgr, prep_info=prep_info, **kwargs)["segments"]


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _draw_pipeline(img_bgr, result, out_path):
    """Save an 8-panel pipeline visualisation PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    def _seg_overlay(base_rgb, segs, color=(220, 40, 40), thickness=2):
        out = base_rgb.copy()
        for (x1, y1, x2, y2) in segs:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.circle(out, (int(x1), int(y1)), 3, color, -1)
            cv2.circle(out, (int(x2), int(y2)), 3, color, -1)
        return out

    def _raw_pca_overlay(clusters):
        out = (img_rgb.astype(np.float32) * 0.25).astype(np.uint8)
        n = len(clusters)
        for i, cl in enumerate(clusters):
            seg = _fit_pca_segment(cl, MIN_SEG_LEN, MIN_ELONGATION)
            if seg is None:
                continue
            hue = int(180 * i / max(n, 1))
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 220]]]), cv2.COLOR_HSV2RGB)[0, 0].tolist()
            cv2.line(out, (int(seg[0]), int(seg[1])),
                     (int(seg[2]), int(seg[3])), color, 3)
            cv2.circle(out, (int(seg[0]), int(seg[1])), 3, color, -1)
            cv2.circle(out, (int(seg[2]), int(seg[3])), 3, color, -1)
        return out

    fg = result["fg_mask"]
    sm = result["seg_mask"]
    clusters = result["clusters"]

    lost = (fg > 0) & (sm == 0)
    panel_lost = img_rgb.copy() // 2
    panel_lost[sm > 0] = [0, 220, 80]
    panel_lost[lost]   = [255, 60, 60]

    panels = [
        (img_rgb,                                          "1. Input"),
        (np.stack([fg * 255]*3, -1).astype(np.uint8),    f"2. Foreground ({fg.sum()} px)"),
        (np.stack([sm * 255]*3, -1).astype(np.uint8),    f"3. Segment mask ({sm.sum()} px)"),
        (panel_lost,                                       f"3b. Kept (green) / Lost (red)"),
        (_raw_pca_overlay(clusters),                       f"4. Raw PCA ({len(clusters)} clusters)"),
        (_seg_overlay(img_rgb, result["segments_grouped"]),
                                                           f"5. Group-fitted ({len(result['segments_grouped'])})"),
        (_seg_overlay(img_rgb, result["segments_extended"]),
                                                           f"6. Extended ({len(result['segments_extended'])})"),
        (_seg_overlay(img_rgb, result["segments"]),        f"7. Final ({len(result['segments'])})"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    for ax, (panel, title) in zip(axes.flat, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle(
        f"Segment Detection Pipeline\n"
        f"Input {W}×{H}px  →  {len(result['segments'])} final segments",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
def _build_parser():
    p = argparse.ArgumentParser(
        prog="segment_detector.py",
        description="Detect line segments in a scientific chart image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python segment_detector.py --image chart.png
  python segment_detector.py --image chart.png --out-dir ./results
  python segment_detector.py --image chart.png --json
  python segment_detector.py --image chart.png --binary-thresh 160 --perp-tol 1.5
""")
    p.add_argument("--image",            required=True,  help="Path to input chart image")
    p.add_argument("--out-dir",          default=".",    help="Output directory (default: current dir)")
    p.add_argument("--json",             action="store_true",
                   help="Print detected segments as JSON to stdout (no visualisation saved)")
    p.add_argument("--no-vis",           action="store_true",
                   help="Skip saving the pipeline visualisation PNG")
    # Tunable parameters
    p.add_argument("--binary-thresh",    type=int,   default=BINARY_THRESH,
                   help=f"Binarisation threshold 0-255 (default {BINARY_THRESH})")
    p.add_argument("--probe-radius",     type=int,   default=PROBE_RADIUS,
                   help=f"Directional probe radius in px (default {PROBE_RADIUS})")
    p.add_argument("--linearity-thresh", type=float, default=LINEARITY_THRESH,
                   help=f"Linearity threshold (default {LINEARITY_THRESH})")
    p.add_argument("--perp-tol",         type=float, default=MERGE_PERP_TOL,
                   help=f"Merge perpendicular tolerance in px (default {MERGE_PERP_TOL})")
    p.add_argument("--gap-tol",          type=float, default=MERGE_GAP_TOL,
                   help=f"Merge gap tolerance in px (default {MERGE_GAP_TOL})")
    p.add_argument("--angle-tol",        type=float, default=MERGE_ANGLE_TOL,
                   help=f"Merge angle tolerance in degrees (default {MERGE_ANGLE_TOL})")
    p.add_argument("--max-gap",          type=int,   default=EXTEND_MAX_GAP,
                   help=f"Extension max consecutive background pixels (default {EXTEND_MAX_GAP})")
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Load image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"ERROR: Cannot read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Run detection
    result = detect_debug(
        img_bgr,
        binary_thresh    = args.binary_thresh,
        probe_radius     = args.probe_radius,
        linearity_thresh = args.linearity_thresh,
        perp_tol         = args.perp_tol,
        gap_tol          = args.gap_tol,
        angle_tol        = args.angle_tol,
        max_gap          = args.max_gap,
    )
    segments = result["segments"]

    # JSON output
    if args.json:
        out = [{"x1": s[0], "y1": s[1], "x2": s[2], "y2": s[3],
                "length": math.hypot(s[2]-s[0], s[3]-s[1]),
                "angle":  _seg_angle(*s)} for s in segments]
        print(json.dumps(out, indent=2))
        return

    # Print summary to stdout
    base = os.path.splitext(os.path.basename(args.image))[0]
    print(f"Image : {args.image}  ({img_bgr.shape[1]}×{img_bgr.shape[0]} px)")
    print(f"Segments detected: {len(segments)}")
    print(f"{'#':>4}  {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7}  {'len':>7}  {'angle':>6}")
    print("-" * 55)
    for i, (x1, y1, x2, y2) in enumerate(segments):
        length = math.hypot(x2 - x1, y2 - y1)
        angle  = _seg_angle(x1, y1, x2, y2)
        print(f"{i+1:>4}  {x1:>7.1f} {y1:>7.1f} {x2:>7.1f} {y2:>7.1f}  "
              f"{length:>7.1f}  {angle:>6.1f}°")

    # Save visualisation
    if not args.no_vis:
        os.makedirs(args.out_dir, exist_ok=True)
        vis_path = os.path.join(args.out_dir, f"{base}_pipeline.png")
        _draw_pipeline(img_bgr, result, vis_path)
        print(f"\nPipeline visualisation saved to: {vis_path}")

        # Also save a clean overlay
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        for (x1, y1, x2, y2) in segments:
            cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (220, 40, 40), 2)
            cv2.circle(overlay, (int(x1), int(y1)), 3, (220, 40, 40), -1)
            cv2.circle(overlay, (int(x2), int(y2)), 3, (220, 40, 40), -1)
        result_path = os.path.join(args.out_dir, f"{base}_segments.png")
        cv2.imwrite(result_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Segment overlay saved to:        {result_path}")


if __name__ == "__main__":
    main()

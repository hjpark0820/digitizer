"""Scope and segment the complete estimated colour path for v46 Step 5.

Observation flags are provenance, not an instruction to discard estimated
coordinates. The adapter retains in-scope estimated geometry without relabeling
it as observed ink. Pure DP/grid helpers are copied from native v45 so runtime
imports never depend on experimental scripts or import the large v45 runner.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def _points(record):
    values = record.get('path')
    if values is None:
        return np.empty((0, 2), float)
    try:
        points = np.asarray(values, float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Estimated path must be an array of numeric [x, y] coordinates') from exc
    if not points.size:
        return np.empty((0, 2), float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Estimated path coordinates must have shape (N, 2)')
    return points


def _box(values, name, *, optional=False):
    if values is None and optional:
        return None
    try:
        box = np.asarray(values, float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must contain four finite coordinates') from exc
    if (box.shape != (4,) or not np.isfinite(box).all()
            or box[0] > box[2] or box[1] > box[3]):
        raise ValueError(f'{name} must be finite inclusive [left, top, right, bottom] bounds')
    return box


def path_scope(record, plot, legend=None):
    """Mask finite raw path samples inside the inclusive plot and outside legend."""
    points = _points(record)
    plot, legend = _box(plot, 'plot'), _box(legend, 'legend', optional=True)
    x, y = points.T
    keep = np.isfinite(points).all(axis=1)
    keep &= (x >= plot[0]) & (x <= plot[2]) & (y >= plot[1]) & (y <= plot[3])
    if legend is not None:
        keep &= ~((x >= legend[0]) & (x <= legend[2]) & (y >= legend[1]) & (y <= legend[3]))
    return keep


def _dp(pts, eps):
    if len(pts) < 3:
        return [pts[0], pts[-1]]
    a, b = np.asarray(pts[0], float), np.asarray(pts[-1], float)
    ab = b - a; L = float(np.hypot(*ab)); P = np.asarray(pts, float)
    d = (np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L
         if L > 1e-6 else np.linalg.norm(P - a, axis=1))
    i = int(np.argmax(d))
    if d[i] <= eps:
        return [pts[0], pts[-1]]
    return _dp(pts[:i + 1], eps)[:-1] + _dp(pts[i:], eps)


def cut_at_grid(segs, grid_xs, min_len=0.0):
    """Native v45 split rule: no output segment spans a supplied grid column."""
    out = []
    for x0, y0, x1, y1 in segs:
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        px, py = x0, y0
        for g in sorted(q for q in grid_xs if x0 + 1.5 < q < x1 - 1.5):
            t = (g - x0) / max(x1 - x0, 1e-6)
            gy = y0 + (y1 - y0) * t
            out.append((px, py, g, gy)); px, py = g, gy
        out.append((px, py, x1, y1))
    return [s for s in out if np.hypot(s[2] - s[0], s[3] - s[1]) >= min_len]


def path_to_segments(path, filled, grid_xs, dev=2.0, min_len=5.0):
    """Native v45 Douglas–Peucker simplification followed by grid splitting."""
    runs, cur = [], []
    for p, v in zip(path, filled):
        if v:
            cur.append((p[0], p[1]))
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    segs = []
    for r in runs:
        if len(r) < 3:
            continue
        v = _dp(r, dev)
        segs += [(v[i][0], v[i][1], v[i + 1][0], v[i + 1][1]) for i in range(len(v) - 1)]
    return cut_at_grid(segs, grid_xs, min_len)


def full_path_segments(record, plot, legend, grid, callback=None):
    """Return ``{'segments': [...], 'provenance': {...}}`` with native v3 parity.

    Every existing in-scope estimated coordinate may participate; no new
    coordinates are extrapolated. Missing x columns, ROI exclusions and legend
    crossings remain hard boundaries. Missing paths return an empty audit.
    Malformed/nonfinite geometry raises a descriptive validation error rather
    than inventing a bridge across an invalid path.
    """
    # Local import avoids a module-level cycle when the exporter uses this
    # adapter. The safety implementation belongs to the production package.
    try:
        from .color_step5_export_v46 import _safe_segments
    except ImportError:
        from color_step5_export_v46 import _safe_segments
    points = _points(record)
    scope = path_scope(record, plot, legend)
    if not np.isfinite(points).all():
        raise ValueError('Estimated path coordinates must be finite before segment export')
    if len(points) > 1 and np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError('Estimated path x coordinates must be strictly increasing')
    grid = np.asarray([] if grid is None else grid, float)
    if grid.ndim != 1 or not np.isfinite(grid).all():
        raise ValueError('Estimated-path grid must be a finite one-dimensional array')
    filled = np.asarray(record.get('filled', np.ones(len(points), bool)), bool)
    observed = np.asarray(record.get('observed', np.zeros(len(points), bool)), bool)
    confidence = np.asarray(record.get('val', np.ones(len(points))), float)
    if any(values.shape != (len(points),) for values in (filled, observed, confidence)):
        raise ValueError('Estimated path observed, filled, and val arrays must match path length')
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Estimated path val confidence must be finite and in [0, 1]')
    adapter = deepcopy(record)
    adapter['path'] = points.tolist()
    adapter['filled'] = scope.tolist()
    adapter['observed'] = (observed & filled & (confidence > 0)).tolist()
    segments, audit = _safe_segments(adapter, plot, legend, grid.tolist(), callback or path_to_segments)
    audit.update(source='all_existing_estimated_path_in_scope_to_native_v45_path_to_segments',
                 coordinates_changed=False, original_observed_flags_changed=False,
                 original_retained_samples=int((scope & filled).sum()),
                 previously_rejected_samples_included=int((scope & ~filled).sum()),
                 evaluated_path_samples=int(scope.sum()),
                 inferred_path_samples=int((scope & ~np.asarray(adapter['observed'], bool)).sum()),
                 inferred_geometry_is_image_evidence=False, extrapolation=False)
    if not len(points):
        audit['status'] = 'missing_path' if record.get('path') is None else 'empty_path'
    return dict(segments=segments, provenance=audit)

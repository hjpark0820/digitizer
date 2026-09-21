# Production snapshot of experiments/color_path_v45/directional_path.py.
# Algorithms are local to src; no experimental runtime dependency.
"""Experimental colour path with a penalty on heading changes, not height gain.

This module does not modify v45/v46, extract colour masks, accept markers, or use
reference coordinates. All bounds are inclusive, matching v45.viterbi_path().
The returned path is an optimisation hypothesis, including across blank pixels;
callers must distinguish observed colour support from unsupported interpolation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionalConfig:
    # Fixed before inspecting Panel C runs; shared by every colour.
    step: int = 3
    max_slope: float = 12.0
    turn_weight: float = 8.0


def heading_transform(cost, source_angles, target_angles, weight):
    """Exact min_u cost[y,u] + weight*abs(source[u]-target[v]).

    Two monotone sweeps compute prefix/suffix minima. Searchsorted then queries
    the envelopes at arbitrary target angles, including a shorter final edge.
    Returns minimum costs and source indices; all-infinite rows use index -1.
    """
    cost = np.asarray(cost, dtype=np.float64)
    source = np.asarray(source_angles, dtype=np.float64)
    target = np.asarray(target_angles, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[1] != len(source) or not len(source):
        raise ValueError('cost must have one column per nonempty source angle')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('angles must be finite')
    if np.any(np.diff(source) <= 0) or weight < 0 or not np.isfinite(weight):
        raise ValueError('source angles must increase strictly and weight be nonnegative')
    if np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError('cost may contain positive infinity, not NaN/-infinity')
    rows, count = cost.shape
    left = cost - weight * source[None, :]
    right = cost + weight * source[None, :]
    li = np.broadcast_to(np.arange(count), (rows, count)).copy()
    ri = li.copy()
    for k in range(1, count):
        better = left[:, k-1] <= left[:, k]
        left[better, k] = left[better, k-1]
        li[better, k] = li[better, k-1]
    for k in range(count-2, -1, -1):
        better = right[:, k+1] < right[:, k]
        right[better, k] = right[better, k+1]
        ri[better, k] = ri[better, k+1]
    left_at = np.searchsorted(source, target, side='right')-1
    right_at = np.searchsorted(source, target, side='left')
    lc = left[:, np.clip(left_at, 0, count-1)] + weight*target[None, :]
    rc = right[:, np.clip(right_at, 0, count-1)] - weight*target[None, :]
    lc[:, left_at < 0] = np.inf
    rc[:, right_at >= count] = np.inf
    choose_left = lc <= rc
    minimum = np.where(choose_left, lc, rc)
    index = np.where(choose_left, li[:, np.clip(left_at, 0, count-1)],
                     ri[:, np.clip(right_at, 0, count-1)])
    index[~np.isfinite(minimum)] = -1
    return minimum, index


def sample_columns(soft, x, y):
    """Sample unmodified soft values at integer x and fractional y (linear)."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    lower = np.floor(y).astype(np.int64)
    upper = np.minimum(lower+1, soft.shape[0]-1)
    fraction = y-lower
    return (1-fraction)*soft[lower, x] + fraction*soft[upper, x]


def _empty(config):
    return dict(path=np.empty((0, 2)), val=np.empty(0),
                knots=np.empty((0, 2)), edge_dy=np.empty(0, dtype=int),
                edge_dx=np.empty(0, dtype=int), headings_radians=np.empty(0),
                cost=dict(image=0.0, turning=0.0, total=0.0,
                          absolute_height_penalty=0.0),
                input_has_evidence=False, path_has_evidence=False,
                status='empty_domain', config=asdict(config))


def trace_directional(soft, lo, hi, y0, y1, config=None):
    """Globally optimal piecewise-linear y(x) on a discrete heading-state graph.

    Nodes exist every ``step`` x pixels and at every integer y. Edges allow both
    rising and falling slopes up to ``max_slope``. An edge pays sum(1-soft) at
    EVERY original x column, with subpixel-y interpolation. Successive edges pay
    turn_weight*abs(angle_next-angle_previous), in radians. Starting direction,
    starting y, and ending y are free. There is NO total-|dy| penalty.

    The last edge uses its actual width, not padding or an invented observation.
    A blank domain still has an optimum; status/flags explicitly identify that
    unsupported raw hypothesis. Returning a path never confirms a curve/marker.
    """
    cfg = config or DirectionalConfig()
    if not isinstance(cfg, DirectionalConfig):
        raise TypeError('config must be DirectionalConfig')
    if not isinstance(cfg.step, (int, np.integer)) or cfg.step < 1:
        raise ValueError('step must be a positive integer')
    if not np.isfinite(cfg.max_slope) or cfg.max_slope < 0:
        raise ValueError('max_slope must be finite and nonnegative')
    if not np.isfinite(cfg.turn_weight) or cfg.turn_weight < 0:
        raise ValueError('turn_weight must be finite and nonnegative')
    field = np.asarray(soft)
    if field.ndim != 2:
        raise ValueError('soft must be a two-dimensional array')
    if any(not isinstance(k, (int, np.integer)) for k in (lo, hi, y0, y1)):
        raise ValueError('bounds must be integers')
    if not field.size or hi < lo or y1 < y0:
        return _empty(cfg)
    if lo < 0 or hi >= field.shape[1] or y0 < 0 or y1 >= field.shape[0]:
        raise ValueError('inclusive bounds must be inside soft')
    sub = np.asarray(field[y0:y1+1, lo:hi+1], dtype=np.float64)
    if not np.isfinite(sub).all() or np.any((sub < 0) | (sub > 1)):
        raise ValueError('soft evidence must be finite and in [0, 1]')
    height, width = sub.shape
    node_x = list(range(0, width, cfg.step))
    if node_x[-1] != width-1:
        node_x.append(width-1)
    if len(node_x) == 1:
        best_y = int(np.argmax(sub[:, 0]))
        path = np.array([[lo, y0+best_y]], dtype=float)
        value = sub[best_y, :].copy()
        return dict(path=path, val=value, knots=path.copy(),
                    edge_dy=np.empty(0, dtype=int), edge_dx=np.empty(0, dtype=int),
                    headings_radians=np.empty(0),
                    cost=dict(image=float(1-value[0]), turning=0.0,
                              total=float(1-value[0]), absolute_height_penalty=0.0),
                    config=asdict(cfg), input_has_evidence=bool(sub.max() > 0),
                    path_has_evidence=bool(value.max() > 0),
                    status='raw_hypothesis' if sub.max() > 0 else 'no_colour_evidence')
    previous = None
    previous_angles = None
    predecessor_tables = []
    edge_directions = []
    all_rows = np.arange(height, dtype=int)
    for edge, (xa, xb) in enumerate(zip(node_x[:-1], node_x[1:])):
        dx = xb-xa
        maximum_dy = min(height-1, int(np.floor(cfg.max_slope*dx)))
        directions = np.arange(-maximum_dy, maximum_dy+1, dtype=int)
        if len(directions) > np.iinfo(np.int16).max:
            raise ValueError('direction grid too large for compact traceback')
        angles = np.arctan(directions/dx)
        if edge:
            transitioned, predecessor = heading_transform(
                previous, previous_angles, angles, cfg.turn_weight)
        current = np.full((height, len(directions)), np.inf)
        back = np.full(current.shape, -1, dtype=np.int16)
        for k, dy in enumerate(directions):
            destination = all_rows[(all_rows-dy >= 0) & (all_rows-dy < height)]
            source = destination-dy
            edge_cost = np.zeros(len(source), dtype=float)
            for offset in range(1, dx+1):
                ys = source+dy*(offset/dx)
                edge_cost += 1-sample_columns(sub, np.full(len(ys), xa+offset), ys)
            if edge == 0:
                current[destination, k] = 1-sub[source, 0]+edge_cost
            else:
                current[destination, k] = transitioned[source, k]+edge_cost
                back[destination, k] = predecessor[source, k]
        predecessor_tables.append(back)
        edge_directions.append(directions)
        previous, previous_angles = current, angles
    final_y, final_k = np.unravel_index(np.argmin(previous), previous.shape)
    optimal_cost = float(previous[final_y, final_k])
    node_y = np.empty(len(node_x), dtype=int)
    node_y[-1] = final_y
    direction_index = final_k
    for edge in range(len(node_x)-2, -1, -1):
        node_y[edge] = node_y[edge+1]-edge_directions[edge][direction_index]
        direction_index = int(predecessor_tables[edge][node_y[edge+1], direction_index])
    xs = np.arange(width, dtype=int)
    ys = np.interp(xs, node_x, node_y)
    values = sample_columns(sub, xs, ys)
    edge_dx = np.diff(node_x)
    edge_dy = np.diff(node_y)
    headings = np.arctan(edge_dy/edge_dx)
    image_cost = float(np.sum(1-values))
    turning_cost = float(cfg.turn_weight*np.abs(np.diff(headings)).sum())
    total = image_cost+turning_cost
    if not np.isclose(total, optimal_cost, rtol=1e-10, atol=1e-8):
        raise AssertionError(f'Traceback cost {total} differs from DP {optimal_cost}')
    return dict(path=np.column_stack((xs+lo, ys+y0)), val=values,
                knots=np.column_stack((np.asarray(node_x)+lo, node_y+y0)),
                edge_dx=edge_dx, edge_dy=edge_dy, headings_radians=headings,
                cost=dict(image=image_cost, turning=turning_cost, total=total,
                          absolute_height_penalty=0.0),
                input_has_evidence=bool(sub.max() > 0),
                path_has_evidence=bool(values.max() > 0),
                status='raw_hypothesis' if sub.max() > 0 else 'no_colour_evidence',
                config=asdict(cfg))

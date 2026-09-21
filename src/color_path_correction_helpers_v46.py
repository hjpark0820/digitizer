"""Deterministic native action search helpers for v46 path-based correction.

No image similarity objective or detector is invoked. Native actions and NMS
are evaluated against immutable path geometry by color_path_correction_v46.
"""
from copy import deepcopy
import hashlib
from itertools import combinations
import random
from types import SimpleNamespace

import numpy as np

from color_path_metrics_v46 import reconstruct_markers
from color_tentative_policy_v46 import strong_evidence


def point_key(point):
    return (round(float(point['cx']), 6), round(float(point['cy']), 6), point.get('class_name', ''))


def set_key(points):
    return tuple(sorted(point_key(p) for p in points))


def difference(first, second):
    other = set(point_key(p) for p in second)
    return [deepcopy(p) for p in first if point_key(p) not in other]


def coords(points):
    return np.asarray([(p['cx'], p['cy']) for p in points], float).reshape(-1, 2)


def curve(points, model, x_query=None):
    xy = coords(points)
    if len(xy) < 2 or np.ptp(xy[:, 0]) < 1e-8:
        return np.empty((0, 2))
    xs = (np.unique(np.r_[np.arange(np.floor(xy[:, 0].min()), np.ceil(xy[:, 0].max())+1), xy[:, 0]])
          if x_query is None else np.asarray(x_query))
    result = reconstruct_markers(xy, xs, model=model)
    valid = np.isfinite(result['y'])
    return np.column_stack((xs[valid], np.asarray(result['y'])[valid]))


def candidate_segments(points, model, grid):
    if len(points) < 2:
        return []
    shape = curve(points, model, np.unique(np.r_[grid, coords(points)[:, 0]]))
    return [tuple(np.r_[a, b]) for a, b in zip(shape[:-1], shape[1:]) if b[0] > a[0]]


def in_bounds(point, roi, legend):
    x, y = float(point['cx']), float(point['cy'])
    return (np.isfinite([x, y]).all() and roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]
            and not (legend is not None and legend[0] <= x <= legend[2] and legend[1] <= y <= legend[3]))


def rank_segments(segments, details, limit=12):
    xs = np.asarray(details['x'])
    residual, weights = np.asarray(details['local_residual']), np.asarray(details['weights'])
    ranked = []
    for segment in segments:
        lo, hi = sorted((segment[0], segment[2]))
        take = (xs >= lo) & (xs <= hi) & (weights > 0)
        value = float(np.average(residual[take], weights=weights[take])) if take.any() else 0.
        ranked.append((value, tuple(map(float, segment))))
    return sorted(ranked, key=lambda row: (-row[0], row[1]))[:limit]


def propose(eng, active, suppressed, segments, details, model, grid, roi, legend):
    ranked = []
    for source, supplied in [('extracted_path', segments),
                             ('reconstructed_path', candidate_segments(active, model, grid))]:
        ranked.extend((gain, seg, source) for gain, seg in rank_segments(supplied, details))
    pairs = eng._conflict_pairs_cached(active, eng.X_APPROX)
    classes = [active[0]['class_name']] if active else [eng._experiment_class]
    proposed, seen, excluded = [], set(), 0
    for rank, (gain, seg, source) in enumerate(ranked, 1):
        endpoints = [dict(cx=seg[0], cy=seg[1]), dict(cx=seg[2], cy=seg[3])]
        endpoints = [p for p in endpoints if in_bounds(p, roi, legend)]
        near = []
        for endpoint in endpoints:
            neighbours = sorted(((np.hypot(s['cx']-endpoint['cx'], s['cy']-endpoint['cy']), i, s)
                                 for i, s in enumerate(suppressed)), key=lambda row: (row[0], row[1]))
            if neighbours and neighbours[0][0] <= eng.PT_TOL and neighbours[0][2] not in near:
                near.append(neighbours[0][2])
        for action, trial in eng.generate_perturbations(active, near, endpoints, classes):
            if eng.has_new_conflicts(pairs, trial, eng.X_APPROX) or not all(in_bounds(p, roi, legend) for p in trial):
                excluded += 1
                continue
            key = action, set_key(trial)
            if key in seen:
                continue
            seen.add(key)
            proposed.append(dict(action=action, points=trial, l_star=seg, segment_source=source,
                                 segment_rank=rank, segment_residual=gain))
    return proposed, ranked, excluded


def trial_nms(eng, points, suppressed, window, action):
    if action.split('+')[0] not in ('ADD', 'ACTIVATE', 'REPLACE'):
        return deepcopy(points), deepcopy(suppressed)
    seed = int(hashlib.sha256(repr(set_key(points)).encode()).hexdigest()[:16], 16)
    eng.random = SimpleNamespace(random=random.Random(seed).random)
    return eng.post_perturbation_nms(deepcopy(points), deepcopy(suppressed), window)


def activate(candidate, cls, eng):
    return dict(candidate, class_name=cls, class_idx=eng.CLASS_NAMES.index(cls),
                state='active_hypothesis', existence='unknown', tentative=True,
                _activated_from_suppressed=True)


def eligible(candidate):
    return strong_evidence(candidate) or (candidate.get('source') == 'colocated_original_marker'
                                         and candidate.get('activation_eligible') is True)


def add_pool_proposals(eng, active, suppressed, items, roi, legend):
    """Global single activations; only independent strong-image pairs bootstrap."""
    cls = eng._experiment_class
    pool = [s for s in suppressed if eligible(s) and all(np.hypot(s['cx']-p['cx'], s['cy']-p['cy']) > 1 for p in active)]
    trials = [('ACTIVATE' if strong_evidence(s) else 'ACTIVATE_OCCLUSION', [s]) for s in pool]
    if len(set(round(p['cx'], 5) for p in active)) == 0:
        trials.extend(('ACTIVATE_PAIR', list(pair)) for pair in combinations([s for s in pool if strong_evidence(s)], 2))
    seen = {(item['action'], set_key(item['points'])) for item in items}
    baseline = eng._conflict_pairs_cached(active, eng.X_APPROX)
    for action, chosen in trials:
        proposed = active + [activate(s, cls, eng) for s in chosen]
        key = action, set_key(proposed)
        if key in seen or eng.has_new_conflicts(baseline, proposed, eng.X_APPROX):
            continue
        if not all(in_bounds(p, roi, legend) for p in proposed):
            continue
        seen.add(key)
        items.append(dict(action=action, points=proposed, l_star=None, segment_source='eligible_suppressed_global',
                          segment_rank=None, segment_residual=None,
                          activated_candidate_ids=[s.get('candidate_id') for s in chosen]))
    return items


def restore_metadata(points, original_points, pool, previous_active=()):
    """Exact original coordinates retain P0 identity; additions never inherit it."""
    original = {point_key(p): p for p in original_points}
    previous = {point_key(p): p for p in previous_active}
    out = []
    for point in points:
        q = {**previous.get(point_key(point), {}), **point}
        matched = original.get(point_key(point))
        if matched is not None:
            q = {**matched, **q, '_initial_id': matched['_initial_id']}
        elif q.get('_activated_from_suppressed'):
            match = next((s for s in pool if abs(s['cx']-q['cx']) < 1e-6 and abs(s['cy']-q['cy']) < 1e-6), None)
            if match is not None:
                q = {**match, **q, 'state': 'active_hypothesis', 'existence': 'unknown'}
            q.pop('_initial_id', None)
        else:
            q.pop('_initial_id', None)
        out.append(q)
    return out


def pool_after(spool, active, known_structural_ids):
    result, seen = [], set()
    for candidate in spool:
        structural = (candidate.get('candidate_id') in known_structural_ids
                      and candidate.get('source') == 'colocated_original_marker')
        if not (strong_evidence(candidate) or structural):
            continue
        if any(np.hypot(candidate['cx']-p['cx'], candidate['cy']-p['cy']) <= 1 for p in active):
            continue
        key = candidate.get('candidate_id'), round(candidate['cx'], 6), round(candidate['cy'], 6)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(candidate, class_name='suppressed', state='suppressed', class_idx=-1))
    return result

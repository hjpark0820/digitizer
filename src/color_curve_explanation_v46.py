# Production snapshot of experiments/color_path_v45/curve_marker_explanation.py.
# Algorithms are local to src; no experimental runtime dependency.
"""Conditional curve explainability from marker-location hypotheses.

Two intentionally different marker-only interpolation models are evaluated
against an independently estimated colour path.  Neither model sees path y
values while fitting.  Additional path-derived points are DIAGNOSTIC SUPPORT
KNOTS, never inferred or confirmed markers.  This is not a minimum-marker
theorem: interpolation assumptions and an explicit error tolerance determine
the answer, and the small greedy search is not globally optimal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class Config:
    tolerance_diameters: float = 1.0
    duplicate_x_tolerance_px: float = 1.0
    residual_cap_tolerances: float = 3.0
    unbracketed_penalty: float = 1.0
    candidate_spacing_diameters: float = 0.5
    anchor_clearance_diameters: float = 0.35
    max_support_knots: int = 4
    minimum_objective_benefit: float = 0.01


def _clean_json(value: Any):
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def group_anchors(anchors, tolerance=1.0):
    """Deterministically collapse near-duplicate x with median x and y.

    Each group is bounded relative to its leftmost member (no chaining).
    Conflicting measurements at the same x cannot be represented by these
    single-valued interpolation models; the spread is retained for auditing.
    """
    rows = []
    for i, anchor in enumerate(anchors):
        if isinstance(anchor, dict):
            x, y = float(anchor['x']), float(anchor['y'])
            identifier = str(anchor.get('id', f'A{i + 1:02d}'))
        else:
            x, y = map(float, anchor)
            identifier = f'A{i + 1:02d}'
        if not np.isfinite([x, y]).all():
            raise ValueError('anchor coordinates must be finite')
        rows.append((x, y, identifier))
    rows.sort()
    groups = []
    for row in rows:
        if not groups or row[0] - groups[-1][0][0] > tolerance:
            groups.append([row])
        else:
            groups[-1].append(row)
    return [{'x': float(np.median([r[0] for r in g])),
             'y': float(np.median([r[1] for r in g])),
             'ids': [r[2] for r in g], 'n': len(g),
             'y_spread_px': float(np.ptp([r[1] for r in g]))}
            for g in groups]


def fit_marker_models(anchors, x_query, duplicate_x_tolerance_px=1.0):
    """Fit using marker x/y ONLY. x_query controls sampling, not fitting.

    Error is a local-normal approximation: vertical residual divided by
    sqrt(1 + model_slope**2). This measures transverse, not vertical, mismatch
    at the queried x; it is not exact nearest-curve distance. No extrapolation.
    """
    groups = group_anchors(anchors, duplicate_x_tolerance_px)
    xq = np.asarray(x_query, dtype=float)
    empty = np.full(xq.shape, np.nan)
    result = {name: {'y': empty.copy(), 'slope': empty.copy()}
              for name in ('linear', 'pchip')}
    if len(groups) < 2:
        return result, groups
    ax = np.asarray([a['x'] for a in groups])
    ay = np.asarray([a['y'] for a in groups])
    inside = (xq >= ax[0]) & (xq <= ax[-1]) & np.isfinite(xq)
    slopes = np.diff(ay) / np.diff(ax)
    segment = np.clip(np.searchsorted(ax, xq[inside], side='right') - 1,
                      0, len(slopes) - 1)
    result['linear']['y'][inside] = np.interp(xq[inside], ax, ay)
    result['linear']['slope'][inside] = slopes[segment]
    pchip = PchipInterpolator(ax, ay, extrapolate=False)
    result['pchip']['y'][inside] = pchip(xq[inside])
    result['pchip']['slope'][inside] = pchip.derivative()(xq[inside])
    return result, groups


def _bands(path, classes, observed):
    """Maximal consecutive classified observed samples; never bridge gaps."""
    result = []
    start = None
    previous = None
    for i in range(len(path) + 1):
        label = classes[i] if i < len(path) and observed[i] else None
        contiguous = (i == 0 or i == len(path)
                      or path[i, 0] - path[i - 1, 0] <= 1.5)
        if start is not None and (label != previous or not contiguous):
            result.append({'classification': previous,
                           'start_index': start, 'end_index_exclusive': i,
                           'x_start': float(path[start, 0]),
                           'x_end': float(path[i - 1, 0]), 'samples': i - start})
            start = None
        if label is not None and start is None:
            start = i
        previous = label
    return result


def _evaluate(context, anchors):
    cfg = Config(**context['config'])
    path = np.asarray(context['path'], dtype=float).reshape(-1, 2)
    filled = np.asarray(context['filled'], dtype=bool)
    observed = np.asarray(context['observed'], dtype=bool)
    models, groups = fit_marker_models(anchors, path[:, 0],
                                      cfg.duplicate_x_tolerance_px)
    tolerance = float(context['diameter']) * cfg.tolerance_diameters
    classes = np.full(len(path), 'excluded', dtype=object)
    classes[filled & ~observed] = 'weak_context'
    classes[observed] = 'unbracketed'
    both_bracketed = np.isfinite(models['linear']['y']) & observed
    within = []
    model_objectives = []
    for model in models.values():
        error = np.abs(path[:, 1] - model['y']) / np.sqrt(1 + model['slope'] ** 2)
        good = (error <= tolerance) & observed
        within.append(good)
        normalized = np.minimum(error / tolerance,
                                cfg.residual_cap_tolerances) / cfg.residual_cap_tolerances
        normalized[observed & ~np.isfinite(error)] = cfg.unbracketed_penalty
        objective = float(np.mean(normalized[observed])) if observed.any() else None
        bracketed = observed & np.isfinite(error)
        model.update(curve=np.column_stack((path[:, 0], model['y'])),
                     error_px=error, normalized_error=normalized,
                     objective=objective,
                     bracketed_residual_component=(float(np.sum(normalized[bracketed]) / observed.sum())
                                                   if observed.any() else None),
                     unbracketed_coverage_component=(float(np.sum(observed & ~bracketed)
                                                           * cfg.unbracketed_penalty / observed.sum())
                                                    if observed.any() else None),
                     mean_bracketed_normal_error_px=(float(np.mean(error[bracketed]))
                                                    if bracketed.any() else None),
                     observed_within_tolerance=int(good.sum()))
        del model['y']
        model_objectives.append(objective)
    classes[both_bracketed & within[0] & within[1]] = 'explained'
    classes[both_bracketed & (within[0] ^ within[1])] = 'model_sensitive'
    classes[both_bracketed & ~within[0] & ~within[1]] = 'unexplained'
    labels = ['explained', 'model_sensitive', 'unexplained', 'unbracketed']
    counts = {name: int(np.count_nonzero(classes == name)) for name in labels}
    n_observed = int(observed.sum())
    metrics = {'observed_path_samples': n_observed,
               'weak_context_samples': int(np.sum(filled & ~observed)),
               'counts': counts,
               'percent_observed': {k: (100 * n / n_observed if n_observed else None)
                                    for k, n in counts.items()},
               'objective': float(np.mean(model_objectives)) if n_observed else None,
               'objective_definition': 'Mean across linear/PCHIP of mean observed local-normal error, capped at 3 tolerances and divided by 3; unbracketed cost=1.',
               'coverage_definition': 'Observed retained path samples, not marker accuracy or detection recall.'}
    return _clean_json({'models': models, 'grouped_anchors': groups,
                        'classifications': classes, 'metrics': metrics,
                        'bands': _bands(path, classes, observed),
                        'anchor_hull_x': ([groups[0]['x'], groups[-1]['x']]
                                          if len(groups) >= 2 else None)})


def _anchor_leave_one_out(context, anchors, base):
    """Report potential input-anchor inconsistency without changing any anchor."""
    cfg = Config(**context['config'])
    original = base['metrics']
    result = []
    for i, anchor in enumerate(anchors):
        reduced = _evaluate(context, anchors[:i] + anchors[i + 1:])
        objective = reduced['metrics']['objective']
        benefit = (original['objective'] - objective
                   if original['objective'] is not None else 0.0)
        result.append({'id': anchor['id'], 'x': anchor['x'], 'y': anchor['y'],
                       'benefit_if_removed': float(benefit),
                       'objective_before': original['objective'], 'objective_after': objective,
                       'potentially_inconsistent_anchor': bool(benefit > cfg.minimum_objective_benefit),
                       'coverage_before': original, 'coverage_after': reduced['metrics'],
                       'coverage_change_counts': {k: reduced['metrics']['counts'][k] - original['counts'][k]
                                                  for k in original['counts']},
                       'automatic_action': 'none',
                       'interpretation': 'A lower conditional curve error after removal can reflect an incorrect anchor OR an inappropriate interpolation model; it does not prove a false-positive marker.'})
    return result


def evaluate_addition(record_or_context, candidatexy, anchors=None):
    """Conditional explanatory benefit of adding one independently given point.

    Pass an analyze_explanation record (uses original anchors), or its context.
    Optional anchors evaluates a later greedy stage. The point can be backed
    by image evidence, but this function does not evaluate or assume evidence.
    """
    context = record_or_context.get('context', record_or_context)
    current = context['anchors'] if anchors is None else anchors
    before = _evaluate(context, current)
    xy = np.asarray(candidatexy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('candidatexy must contain two finite source coordinates')
    cfg = Config(**context['config'])
    x0, y0, x1, y1 = context['plot_box']
    groups = group_anchors(current, cfg.duplicate_x_tolerance_px)
    reason = None
    if not (x0 <= xy[0] < x1 and y0 <= xy[1] < y1):
        reason = 'candidate_outside_plot'
    elif any(abs(xy[0] - g['x']) <= cfg.duplicate_x_tolerance_px for g in groups):
        reason = 'duplicate_x_not_independent_support'
    added = list(current) + [{'id': 'conditional_candidate', 'x': float(xy[0]), 'y': float(xy[1])}]
    after = _evaluate(context, added) if reason is None else before
    obj_before, obj_after = before['metrics']['objective'], after['metrics']['objective']
    benefit = obj_before - obj_after if obj_before is not None else 0.0
    return {'kind': 'conditional_support_test', 'x': float(xy[0]), 'y': float(xy[1]),
            'status': reason or ('no_observed_path' if obj_before is None else 'evaluated'),
            'benefit': float(benefit), 'objective_before': obj_before,
            'objective_after': obj_after, 'coverage_before': before['metrics'],
            'coverage_after': after['metrics'],
            'explanation_change_counts': {k: after['metrics']['counts'][k] - before['metrics']['counts'][k]
                                          for k in before['metrics']['counts']},
            'existence': 'unknown', 'after': after}


def _candidate_indices(context, current, stage, cfg):
    path = np.asarray(context['path'], dtype=float).reshape(-1, 2)
    observed = np.asarray(context['observed'], dtype=bool)
    classes = np.asarray(stage['classifications'])
    # Model disagreement alone does not require additional support.
    eligible = observed & np.isin(classes, ['unexplained', 'unbracketed'])
    clearance = context['diameter'] * cfg.anchor_clearance_diameters
    for anchor in group_anchors(current, cfg.duplicate_x_tolerance_px):
        eligible &= np.abs(path[:, 0] - anchor['x']) > clearance
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return []
    spacing = max(1.0, context['diameter'] * cfg.candidate_spacing_diameters)
    bins = np.floor((path[indices, 0] - context['plot_box'][0]) / spacing).astype(int)
    # One deterministic sampled support point per x bin, plus the final edge.
    chosen = indices[np.r_[True, bins[1:] != bins[:-1]]].tolist()
    if int(indices[-1]) not in chosen:
        chosen.append(int(indices[-1]))
    return chosen


def analyze_explanation(path, filled, observed, anchors, diameter, plot_box,
                        cfg=Config()):
    """Fit two marker-only curves and search sparse diagnostic support knots.

    The supplied anchors should be the caller's frozen high-evidence marker
    hypotheses, not path-derived positions. Fully occluded markers cannot be
    confirmed by a reduction of reconstruction error.
    """
    path = np.asarray(path, dtype=float)
    filled, observed = np.asarray(filled, bool), np.asarray(observed, bool)
    if path.ndim != 2 or path.shape[1] != 2 or filled.shape != (len(path),) or observed.shape != filled.shape:
        raise ValueError('path must be Nx2 with matching filled/observed vectors')
    if not np.isfinite(path).all() or (len(path) > 1 and np.any(np.diff(path[:, 0]) <= 0)):
        raise ValueError('path must be finite and strictly increasing in x')
    if np.any(observed & ~filled):
        raise ValueError('observed samples must also be retained (filled)')
    if diameter <= 0 or not np.isfinite(diameter):
        raise ValueError('diameter must be positive and finite')
    if (cfg.tolerance_diameters <= 0 or cfg.duplicate_x_tolerance_px < 0
            or cfg.residual_cap_tolerances <= 0 or cfg.unbracketed_penalty < 0
            or cfg.candidate_spacing_diameters <= 0 or cfg.max_support_knots < 0
            or cfg.minimum_objective_benefit < 0):
        raise ValueError('invalid reconstruction configuration')
    x0, y0, x1, y1 = map(float, plot_box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('plot_box must be nonempty and half-open')
    scope = (path[:, 0] >= x0) & (path[:, 0] < x1) & (path[:, 1] >= y0) & (path[:, 1] < y1)
    if np.any(filled & ~scope):
        raise ValueError('retained path must be inside the plot box')
    normalized_anchors = []
    for i, a in enumerate(anchors):
        if isinstance(a, dict):
            normalized_anchors.append({'id': str(a.get('id', f'A{i + 1:02d}')),
                                       'x': float(a['x']), 'y': float(a['y'])})
        else:
            normalized_anchors.append({'id': f'A{i + 1:02d}', 'x': float(a[0]), 'y': float(a[1])})
    if any(not (x0 <= a['x'] < x1 and y0 <= a['y'] < y1) for a in normalized_anchors):
        raise ValueError('anchors must be finite and inside the plot box')
    context = {'path': path.tolist(), 'filled': filled.tolist(), 'observed': observed.tolist(),
               'anchors': normalized_anchors, 'diameter': float(diameter),
               'plot_box': [x0, y0, x1, y1], 'config': asdict(cfg)}
    base = _evaluate(context, normalized_anchors)
    leave_one_out = _anchor_leave_one_out(context, normalized_anchors, base)
    current, stage = list(normalized_anchors), base
    supplemental = []
    sufficient_anchors = len(base['grouped_anchors']) >= 2
    for iteration in range(cfg.max_support_knots if sufficient_anchors else 0):
        best = None
        for index in _candidate_indices(context, current, stage, cfg):
            proposal = evaluate_addition(context, path[index], anchors=current)
            if best is None or proposal['benefit'] > best[1]['benefit'] + 1e-12:
                best = (index, proposal)
        if best is None or best[1]['benefit'] < cfg.minimum_objective_benefit:
            break
        index, proposal = best
        identifier = f'K{iteration + 1:02d}'
        knot = {'id': identifier, 'x': float(path[index, 0]), 'y': float(path[index, 1]),
                'kind': 'diagnostic_support_knot', 'existence': 'unknown',
                'source_path_index': int(index), 'benefit': proposal['benefit'],
                'classification_before': stage['classifications'][index],
                'objective_before': proposal['objective_before'],
                'objective_after': proposal['objective_after'],
                'coverage_before': proposal['coverage_before'],
                'coverage_after': proposal['coverage_after']}
        supplemental.append(knot)
        current.append({'id': identifier, 'x': knot['x'], 'y': knot['y']})
        stage = proposal['after']
    return {'schema_version': 1, 'config': asdict(cfg), 'context': context,
            'tolerance_px': float(diameter * cfg.tolerance_diameters),
            'anchors': normalized_anchors, 'base': base,
            'anchor_leave_one_out': leave_one_out,
            'supplemental_support_knots': supplemental, 'after': stage,
            'status': ('insufficient_distinct_anchors' if not sufficient_anchors
                       else 'no_observed_path' if not np.any(observed) else 'evaluated'),
            'warnings': [
                'Both interpolation models are conditional assumptions, not a known drawing process.',
                'Input strong marker hypotheses may themselves contain false positives or biased centres.',
                'Additional support knots are fitted to the path and are not image-backed marker detections.',
                'Greedy bounded support-knot search is not a proof of the minimum required marker count.',
                'Normal residual is a local linear approximation and can understate mismatch at steep slopes.',
                'Only observed retained path samples enter scores; weak and rejected path samples do not.',
                'Same-x anchors are median grouped; single-valued models cannot resolve multiple y values at one x.',
                'Model disagreement and points outside the anchor hull are not proof of missing markers.']}

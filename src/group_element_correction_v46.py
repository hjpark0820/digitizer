"""Experimental, dashed-style-tolerant bidirectional element Step 5.

The reference is frozen v45 raw scoped segments, including short dashes, not
the refinement stage which prunes them. Bounded collinear gaps can be bridged.
Worst unexplained reference/reconstruction elements locate actions; the fixed
global curve-union objective decides whether an action improves geometry.
Unlabelled same-colour lines cannot prove marker identity or existence.
"""
from copy import deepcopy
import math
import numpy as np

from group_segment_metric_v46 import GlobalSegmentMetric, scoped_segments, _nearest_segments
from bw_step5_v46 import key, setkey, difference, initial_guard, _normalise
from x_singleton_suppressed_v46 import same_missing_column


def bridge_dashes(segments, diameter):
    """Add only bounded, collinear gaps; preserve every original segment.

    The pixel-width allowance handles paired detector edges. Avoid quadratic
    length-changing merges: gaps are computed against the original fragments,
    not recursively against newly invented bridges. No extrapolation.
    """
    raw = np.asarray(segments, float).reshape(-1, 4)
    bridges = []
    for i, a in enumerate(raw):
        va = a[2:]-a[:2]; la = np.linalg.norm(va)
        if la < .4*diameter:
            continue
        ta = va/la
        for b in raw[i+1:]:
            vb = b[2:]-b[:2]; lb = np.linalg.norm(vb)
            if lb < .4*diameter or abs(ta @ (vb/lb)) < math.cos(math.radians(8)):
                continue
            normal = np.array([-ta[1], ta[0]])
            lateral = np.abs((b.reshape(2, 2)-a[:2]) @ normal)
            if lateral.max() > max(2., .16*diameter):
                continue
            ends = b.reshape(2, 2)
            projection = (ends-a[:2]) @ ta
            if projection.min() > la:
                p, q = a[2:], ends[projection.argmin()]
            elif projection.max() < 0:
                p, q = ends[projection.argmax()], a[:2]
            else:
                continue
            gap = np.linalg.norm(q-p)
            if 1. < gap <= 1.8*diameter:
                bridges.append([*p, *q])
    return bridges


def segment_errors(segments, other, diameter):
    """Integrated finite-segment-to-union distance and undirected orientation.

    A crossing does not yield zero distance for the whole segment. One long
    segment can match many short fragments without endpoint one-to-one errors.
    """
    result = []
    for idx, s in enumerate(segments):
        v = s[2:]-s[:2]; length = float(np.linalg.norm(v))
        if length < 1e-9:
            continue
        sample = s[:2]+np.linspace(0, 1, max(3, int(math.ceil(length/2))+1))[:, None]*v
        distance, _, tangent = _nearest_segments(sample, np.asarray(other, float).reshape(-1, 4))
        angular = (1-np.abs(tangent @ (v/length)))*np.exp(-(distance/diameter)**2)
        residual = float(np.mean(np.minimum(distance/(2*diameter), 1)+.15*angular))
        result.append(dict(index=idx, segment=s.tolist(), residual=residual, length_px=length))
    return sorted(result, key=lambda r: (-r['residual'], r['index']))


def trial_actions(active, pool, models):
    """Typed ACTIVATE/REPLACE/DELETE, without a dash-endpoint marker gate."""
    trials = []; seen = set()
    def add(action, pts, target):
        signature = setkey(pts)
        if signature in seen:
            return
        seen.add(signature)
        added, removed = difference(pts, active), difference(active, pts)
        valid, reason = initial_guard(removed, active, pts, models)
        for p in added:
            if any(key(q)!=key(p) and same_missing_column(p,q) for q in pts):
                valid, reason = False, 'missing_series_column_collision'
            if any(q['swatch_id'] == p['swatch_id'] and key(q) != key(p) and
                   math.hypot(q['cx']-p['cx'], q['cy']-p['cy']) < .45*models[p['swatch_id']]['diameter'] for q in pts):
                valid, reason = False, 'same_swatch_collision'
        if removed and any(sum(q['swatch_id'] == p['swatch_id'] for q in pts) < 2 for p in removed):
            valid, reason = False, 'preserve_minimum_two_series_points'
        trials.append(dict(action=action, points=deepcopy(pts), added=added, removed=removed,
                           target=deepcopy(target), admissible=valid, reason=reason))
    for q in pool:
        p = deepcopy(q)
        p.update(point_id=p['candidate_id'], tentative=True, state='active_hypothesis')
        add('ACTIVATE', active+[p], p)
        same = sorted((q for q in active if q['swatch_id'] == p['swatch_id']),
                      key=lambda q: math.hypot(q['cx']-p['cx'], q['cy']-p['cy']))
        for q in same:
            if (q in same[:2] and math.hypot(q['cx']-p['cx'], q['cy']-p['cy']) <= 1.5*models[p['swatch_id']]['diameter']) or same_missing_column(p,q):
                add('REPLACE', [r for r in active if key(r) != key(q)]+[p], p)
    for p in active:
        add('DELETE', [q for q in active if key(q) != key(p)], p)
    return trials


def run(active, pool, reference_segments, diameter, plot, legend=None, max_iter=10,
        evidence_prior_weight=.03, original_points=None, source_gray=None, source_ignore=None,
        action_guard=None):
    if not math.isfinite(evidence_prior_weight) or evidence_prior_weight < 0:
        raise ValueError('evidence_prior_weight must be finite and nonnegative')
    active, pool, models = _normalise(active, pool, plot, legend, diameter)
    raw = scoped_segments(reference_segments, plot, legend)
    bridges = bridge_dashes(raw, diameter)
    reference = scoped_segments([*raw, *bridges], plot, legend)
    metric = GlobalSegmentMetric(reference, diameter, plot, legend)
    endpoint_evidence = None
    if source_gray is not None:
        from group_endpoint_guard_v46 import EndpointEvidence
        endpoint_evidence = EndpointEvidence(source_gray, source_ignore, reference, diameter)
    elif source_ignore is not None:
        raise ValueError('source_ignore requires source_gray')
    initial = deepcopy(active); initial_pool = deepcopy(pool); trace = []
    if not metric.available:
        return dict(initial=initial, initial_suppressed=initial_pool, final=active, suppressed=pool,
                    trace=[], raw_segments=raw.tolist(), bridge_segments=bridges,
                    reference_segments=reference.tolist(), reason='no_reference_segments')
    # Penalize loss of original measured points, not failure to activate a
    # structural prior. Confidence here is a detector score, NOT a probability.
    # A large geometric gain can still remove a confidently misclassified point.
    original_weights = {key(p): float(np.clip(p.get('confidence',0.),0,1))**2
                        for p in (initial if original_points is None else original_points) if p.get('original_detection', True)}
    cache = {}
    def components(points):
        signature = setkey(points)
        if signature not in cache:
            geometry = metric.score(points)
            present = set(signature)
            prior = evidence_prior_weight*sum(w for k,w in original_weights.items() if k not in present)
            cache[signature] = dict(geometry_loss=geometry['objective'], removal_penalty=prior,
                                    objective=geometry['objective']+prior)
        return cache[signature]
    def score(points):
        return components(points)['objective']
    for iteration in range(1, max_iter+1):
        before = deepcopy(active); before_pool = deepcopy(pool); loss = score(before)
        current = metric.candidate_segments(before)
        missing = segment_errors(reference, current, diameter)
        extra = segment_errors(current, reference, diameter)
        trials = trial_actions(active, pool, models)
        improving = []
        for trial in trials:
            if not trial['admissible']:
                continue
            if action_guard is not None:
                evidence = action_guard(trial, before)
                trial['marker_action_evidence'] = evidence
                if not evidence['admissible']:
                    trial.update(admissible=False, reason=evidence['reason'])
                    continue
            trial['score_after'] = score(trial['points'])
            trial['score_components_after'] = components(trial['points'])
            trial['improvement'] = loss-trial['score_after']
            if endpoint_evidence is not None and trial['removed']:
                evidence = endpoint_evidence.check(trial['removed'],before,trial['points'])
                trial['endpoint_evidence'] = evidence
                if evidence['protected']:
                    trial.update(admissible=False,reason='preserve_observed_endpoint_missing_reference')
                    continue
            if trial['improvement'] > 1e-5:
                improving.append(trial)
        best = None; focus = None
        # Start with argmax-min in each direction. If the worst element is
        # noise or has no admissible edit, proceed to the next ranked pair.
        for rank in range(max(len(missing), len(extra))):
            choices = []
            for trial in improving:
                candidate = metric.candidate_segments(trial['points'])
                relevant = []
                if rank < len(missing) and trial['added']:
                    row = missing[rank]
                    updated = segment_errors(np.asarray([row['segment']]), candidate, diameter)[0]['residual']
                    if updated < row['residual']-1e-5:
                        relevant.append(dict(direction='reference_missing', **row, residual_after=updated))
                if rank < len(extra) and trial['removed']:
                    row = extra[rank]
                    # Removal/replacement must actually alter this selected
                    # reconstructed edge, not just lower an unrelated score.
                    s = np.asarray(row['segment'])
                    if not any(np.allclose(s, q) for q in candidate):
                        relevant.append(dict(direction='reconstruction_extra', **row))
                if relevant:
                    choices.append((trial, relevant))
            if choices:
                best, selected = min(choices, key=lambda pair: pair[0]['score_after'])
                focus = dict(rank=rank+1, elements=selected)
                break
        row = dict(iteration=iteration, points_before=before, suppressed_before=before_pool,
                   score_before=loss, missing_ranking=missing, extra_ranking=extra,
                   score_components_before=components(before),
                   focus=focus, action='NONE', added=[], removed=[], score_after=loss,
                   trials=[{k:v for k,v in t.items() if k != 'points'} for t in trials])
        if best is not None:
            active = best['points']
            pool = [p for p in pool if key(p) not in {key(q) for q in active}]
            for p in best['removed']:
                if key(p) not in {key(q) for q in pool}:
                    p = deepcopy(p); p.setdefault('candidate_id', 'REM_'+p['point_id'])
                    p['state'] = 'suppressed'; pool.append(p)
            row.update(action=best['action'], added=best['added'], removed=best['removed'], score_after=best['score_after'])
        row.update(points_after=deepcopy(active), suppressed_after=deepcopy(pool),
                   score_components_after=components(active))
        trace.append(row)
        if best is None:
            break
    return dict(initial=initial, initial_suppressed=initial_pool, final=active, suppressed=pool,
                raw_segments=raw.tolist(), bridge_segments=bridges, reference_segments=reference.tolist(),
                initial_score=score(initial), final_score=score(active), trace=trace, config=metric.config,
                evidence_prior_weight=evidence_prior_weight,
                removal_prior='weight * sum(confidence^2 of absent original active records); score is not probability',
                initial_score_components=components(initial),final_score_components=components(active),
                caveat='geometry-supported hypotheses are not confirmed marker detections')

"""Production v46 colour correction against frozen extracted path geometry.

The objective is not SSIM. Original detections receive an explicit local-change
guard; eligible suppressed candidates are hypotheses, never new observations.
The per-series state supports repeated one-iteration GUI calls without relabeling
previous additions as original detections or reinitializing an empty curve.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np

import color_path_correction_helpers_v46 as search
from color_path_metrics_v46 import METRICS, MetricConfig, PathReference, score_markers
from color_initial_point_guard_v46 import GuardConfig, InitialPointGuard, label_initial_points
from color_colocated_hypotheses_v46 import generate
from color_estimated_path_v46 import path_scope
from color_tentative_policy_v46 import strong_evidence

STATE_VERSION = 'v46_path_correction_state_v1'
_HERE = Path(__file__).resolve().parent


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _save(path, value):
    path.write_text(json.dumps(_plain(value), indent=2, allow_nan=False), encoding='utf-8')


def _load_native(filename):
    """Each series owns an independent native module and mutable NMS RNG."""
    spec = importlib.util.spec_from_file_location('v46_path_' + uuid.uuid4().hex, str(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_for(previous, name):
    if previous is None:
        return None
    if name in previous.get('path_state', {}):
        return previous['path_state'][name]
    row = previous.get('curves', {}).get(name)
    return row.get('path_state') if isinstance(row, dict) else None


def _current_points(values, cls, class_idx, original, pool, previous_active):
    previous_by_key = {search.point_key(p): p for p in previous_active}
    points = []
    for value in values:
        if isinstance(value, dict):
            point = dict(value)
            x = point.get('cx', point.get('x'))
            y = point.get('cy', point.get('y'))
        else:
            x, y = value
            point = {}
        if not np.isfinite([x, y]).all():
            raise ValueError('Current Step-5 points must have finite source coordinates')
        point.update(cx=float(x), cy=float(y), class_name=cls, class_idx=class_idx)
        # Native ADD has no confidence field (NMS treats absence as 0). Preserve
        # that absence on resume rather than accidentally promoting it to 1.
        previous = previous_by_key.get(search.point_key(point))
        if 'confidence' not in point:
            if previous is None:
                point['confidence'] = 1.
            elif 'confidence' in previous:
                point['confidence'] = previous['confidence']
        points.append(point)
    return search.restore_metadata(points, original, pool, previous_active)


def _validated_pool(saved, source_pool, active, roi, legend):
    """Saved flags cannot upgrade a previously review-only structural candidate."""
    structural = {s['candidate_id']: s for s in source_pool if s.get('source') == 'colocated_original_marker'}
    result = []
    for candidate in saved:
        if not search.in_bounds(candidate, roi, legend):
            continue
        source = structural.get(candidate.get('candidate_id'))
        if source is not None:
            if abs(source['cx']-candidate['cx']) > 1e-6 or abs(source['cy']-candidate['cy']) > 1e-6:
                continue
            result.append(deepcopy(source))
        elif strong_evidence(candidate) and candidate.get('source') != 'colocated_original_marker':
            source = next((s for s in source_pool if strong_evidence(s)
                           and s.get('candidate_id') == candidate.get('candidate_id')
                           and abs(s['cx']-candidate['cx']) < 1e-6 and abs(s['cy']-candidate['cy']) < 1e-6), None)
            if source is not None:
                result.append(deepcopy(source))
    return search.pool_after(result, active, set(structural))


def _config(metric, model, reference_policy, inferred_weight):
    if metric not in METRICS:
        raise ValueError(f'Unknown path metric: {metric}')
    if model not in ('linear', 'pchip'):
        raise ValueError(f'Unknown path reconstruction model: {model}')
    if reference_policy not in ('observed', 'estimated_path'):
        raise ValueError(f'Unknown reference policy: {reference_policy}')
    if not np.isfinite(inferred_weight) or not 0 < inferred_weight <= 1:
        raise ValueError('inferred_weight must be finite and in (0, 1]')
    return dict(metric=metric, model=model, reference_policy=reference_policy,
                inferred_weight=float(inferred_weight), metric_config=asdict(MetricConfig()),
                guard_config=asdict(GuardConfig()), version=STATE_VERSION)


def correct_path_payload_curves(image, curves, payload, *, out_dir, max_iters=None,
                                previous_state=None, workers=None, engine_dir=None,
                                log_fn=print, metric='chamfer', model='pchip',
                                reference_policy='estimated_path', inferred_weight=.25,
                                proposal_validator=None, colocated_hypotheses=None):
    """Return per-series point results and JSON-safe resumable path_state.

    ``payload`` must be the validated detector-free v46 handoff, including
    reference_paths and full source-coordinate segments_override. The caller's
    curves contain the CURRENT active coordinates; payload init_points remain
    the immutable original detector registry. Missing paths preserve points.
    """
    config = _config(metric, model, reference_policy, inferred_weight)
    if proposal_validator is not None:
        config['new_point_validation'] = 'colour_series_window_triangle_v1'
    limit = 10 if max_iters is None else int(max_iters)
    if limit < 1 or (max_iters is not None and limit != max_iters):
        raise ValueError('max_iters must be a positive integer')
    if workers is not None and (int(workers) < 1 or int(workers) != workers):
        raise ValueError('workers must be a positive integer')
    rows = {row['name']: row for row in payload['curves']}
    names = [curve['name'] for curve in curves]
    if len(set(names)) != len(names) or any(name not in rows or Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError('Curve identities must be unique, safe names in the payload')
    roi, legend = payload['plot_area'], payload.get('legend_box')
    native_dir = Path(engine_dir) if engine_dir else _HERE
    records = payload.get('reference_paths') or {}
    references, observed_records, unavailable = {}, {}, {}
    for name, row in rows.items():
        record = records.get(name)
        if record is None:
            unavailable[name] = 'missing_path_not_corrected'
            continue
        try:
            scope = path_scope(record, roi, legend)
            reference = PathReference.from_record(record, row['diameter'], reference_policy=reference_policy,
                                                  inferred_weight=inferred_weight, scope_mask=scope)
            # Structural image gates always use actual observations, not inference.
            actual = PathReference.from_record(record, row['diameter'], scope_mask=scope)
        except (KeyError, TypeError, ValueError) as error:
            unavailable[name] = 'invalid_path_not_corrected: ' + str(error)
            continue
        if not len(reference.sample_xy):
            unavailable[name] = 'empty_path_not_corrected'
            continue
        references[name], observed_records[name] = reference, actual
    colocated = (generate(payload, observed_records) if colocated_hypotheses is None
                 else deepcopy(colocated_hypotheses))
    if colocated_hypotheses is not None:
        config['shared_occlusion_policy'] = 'all_series_missing_x_v1'
        config['colocated_digest'] = _digest(colocated)
    evidence = dict(identity=payload.get('identity'), image=_array_digest(image),
                    plot_area=roi, legend_box=legend, grid_xs=payload['grid_xs'], reference_paths=records,
                    curves=[dict(name=row['name'], marker_class=row.get('marker_class'), diameter=row.get('diameter'),
                                 init_points=row.get('init_points', []), init_suppressed=row.get('init_suppressed', []),
                                 segments=row.get('segments_override', []), template_center=row.get('template_center'),
                                 ink_mask=_array_digest(row['ink_mask']) if row.get('ink_mask') is not None else None,
                                 alpha=_array_digest(row['marker_alpha']) if row.get('marker_alpha') is not None else None)
                            for row in payload['curves']])
    native_hashes = {name: hashlib.sha256((native_dir/name).read_bytes()).hexdigest()
                     for name in ('5_correction_color.py', '4_segment_refinement.py')}
    fingerprint = _digest(dict(config=config, evidence=evidence, native_algorithms=native_hashes))
    old_states = {name: _state_for(previous_state, name) for name in names}
    for name, state in old_states.items():
        if previous_state is not None and state is None:
            raise ValueError(f'{name}: previous Step-5 state lacks path provenance; start a fresh path-correction run')
        if state is not None and (state.get('version') != STATE_VERSION or state.get('fingerprint') != fingerprint):
            raise ValueError(f'{name}: saved path correction state does not match current image, evidence or configuration')
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    _save(root/'path_colocated_hypotheses.json', colocated)
    _save(root/'path_correction_manifest.json', dict(config=config, fingerprint=fingerprint,
          series=names, reference_summaries={name: ref.describe() for name, ref in references.items()},
          native_algorithm_sha256=native_hashes,
          image_sha256=evidence['image']['sha256'], unavailable=unavailable, prediction_status='unvalidated_hypotheses'))
    nworkers = max(1, min(len(curves) or 1, int(workers or min(4, os.cpu_count() or 1))))

    def run(current):
        name = current['name']
        return _run_series(current, rows[name], references.get(name), unavailable.get(name),
                           colocated['by_series'].get(name, []), payload, old_states[name],
                           fingerprint, config, root/name, native_dir, limit, log_fn, proposal_validator)

    if nworkers == 1:
        return [run(current) for current in curves]
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        return list(executor.map(run, curves))


def _run_series(current, row, reference, unavailable, structural, payload, previous,
                fingerprint, config, folder, native_dir, limit, log_fn, proposal_validator=None):
    tick = perf_counter()
    name, model, metric = current['name'], config['model'], config['metric']
    eng = _load_native(native_dir/'5_correction_color.py')
    cls = row.get('marker_class')
    unsupported = cls not in eng.CLASS_NAMES and row.get('marker_alpha') is None
    if cls not in eng.CLASS_NAMES:
        cls = eng.CLASS_NAMES[0]
    eng._experiment_class = cls
    original = label_initial_points([dict(p, class_name=cls, class_idx=eng.CLASS_NAMES.index(cls), confidence=1.)
                                     for p in row.get('init_points', [])], prefix=name)
    if previous is not None and _digest(previous.get('original_points')) != _digest(original):
        raise ValueError(f'{name}: saved immutable original registry changed')
    source_pool = deepcopy(row.get('init_suppressed', [])) + deepcopy(structural)
    old_active = previous.get('active_points', []) if previous else []
    active = _current_points(current.get('points', []), cls, eng.CLASS_NAMES.index(cls), original, source_pool, old_active)
    incoming = deepcopy(active)
    pool = _validated_pool(previous.get('suppressed', []) if previous else source_pool,
                           source_pool, active, payload['plot_area'], payload.get('legend_box'))
    structural_ids = {s['candidate_id'] for s in structural}
    initialized = bool(previous.get('initialized')) if previous else bool(active)
    count = int(previous.get('iteration_count', 0)) if previous else 0
    same_active = previous is not None and search.set_key(active) == search.set_key(old_active)
    converged = bool(previous.get('converged', False)) and same_active if previous else False
    guard = InitialPointGuard(reference, original, model=model, config=GuardConfig()) if reference is not None else None
    label = 'directional' if metric == 'directional_chamfer' else metric
    method = dict(id=f'{label}_{model}', metric=metric, model=model)
    folder.mkdir(parents=True, exist_ok=True)
    trace_file = folder/'trace.json'
    history, old_trace = [], {}
    if previous is not None and trace_file.exists():
        old_trace = json.loads(trace_file.read_text(encoding='utf-8'))
        if old_trace.get('fingerprint') == fingerprint:
            # An interrupted invocation may have written speculative iterations
            # beyond the last caller-persisted state. Do not replay that tail.
            history = [entry for entry in old_trace.get('iterations', []) if entry.get('iteration', 0) <= count]
    trace = dict(series=name, method=method, fingerprint=fingerprint, initial_points=deepcopy(original),
                 run_initial_points=deepcopy(active), initial_suppressed=deepcopy(source_pool), iterations=history,
                 requested_iterations=count+limit, supplied_segments=row.get('segments_override', []),
                 metric_config=config['metric_config'], guard_config=config['guard_config'],
                 reference_description=reference.describe() if reference is not None else None,
                 prediction_status='unvalidated_correction_hypotheses', status='running',
                 resumed=previous is not None, previous_iteration_count=count)
    for key in ('refined_segments', 'refinement', 'grid_xs', 'nms_window', 'x_approx'):
        if key in old_trace:
            trace[key] = old_trace[key]
    baseline_first = score_markers(reference, search.coords(active), metric=metric, model=model)['objective'] if reference else None
    reason = ('unsupported_marker_not_corrected' if unsupported else unavailable)
    if any(not search.in_bounds(p, payload['plot_area'], payload.get('legend_box')) for p in active):
        reason = 'out_of_scope_active_points_preserved'
    if not reason and not converged:
        refiner = _load_native(native_dir/'4_segment_refinement.py')
        grid = payload['grid_xs']
        step = refiner.grid_step(grid)
        eng.X_APPROX, window = .5*step, .75*step
        segments, refinement = refiner.refine(row.get('segments_override', []), grid)
        trace.update(refined_segments=segments, refinement=refinement, grid_xs=grid, nms_window=window, x_approx=eng.X_APPROX)
        cache = {}
        def evaluate(points, details=False):
            key = search.set_key(points), details
            if key not in cache:
                cache[key] = score_markers(reference, search.coords(points), metric=metric, model=model, details=details)
            return cache[key]
        for _ in range(limit):
            started = perf_counter()
            before, sbefore = deepcopy(active), deepcopy(pool)
            base = evaluate(active, True)
            baseline = base['objective']
            initializing = not active and not initialized
            if initializing:
                proposals, ranked, excluded = [], [], 0
            else:
                proposals, ranked, excluded = search.propose(eng, active, [s for s in pool if search.eligible(s)],
                    segments, base, model, grid, payload['plot_area'], payload.get('legend_box'))
            proposals = search.add_pool_proposals(eng, active, pool, proposals, payload['plot_area'], payload.get('legend_box'))
            if initializing:
                proposals = [dict(p, action='ACTIVATE_PAIR_INIT') for p in proposals if p['action'] == 'ACTIVATE_PAIR']
                initialized = True
            best, best_total, trials, blocked = None, baseline, [], 0
            for item in proposals:
                points = search.restore_metadata(item['points'], original, pool, active)
                nms_action = 'ACTIVATE' if item['action'].startswith('ACTIVATE') else item['action']
                trial, spool = search.trial_nms(eng, points, pool, window, nms_action)
                trial = search.restore_metadata(trial, original, pool, active)
                scored, check = evaluate(trial), guard.evaluate(active, trial)
                pixel_checks = []
                if proposal_validator is not None:
                    pixel_checks = [dict(x=p['cx'], y=p['cy'], **proposal_validator(name, p))
                                    for p in search.difference(trial, active)]
                    if any(not p['accepted'] for p in pixel_checks):
                        check = dict(check, allowed=False, pixel_gate_rejected=True)
                cost = eng.MIN_GAIN_PER_REMOVAL * max(0, len(active)-len(trial)-1)
                total = scored['objective'] + cost
                record = {k: v for k, v in item.items() if k != 'points'}
                record.update(raw_points=points, evaluated_points=deepcopy(trial), raw_distance=scored['objective'],
                              removal_cost=cost, cost_adjusted_distance=total, components=scored,
                              initial_point_guard=check, admissible=check['allowed'],
                              new_point_pixel_checks=pixel_checks)
                trials.append(record)
                blocked += not check['allowed']
                if check['allowed'] and total < best_total-1e-7:
                    best_total, best = total, (trial, spool, record)
            if best:
                active, spool, winner = best
                pool = search.pool_after(spool, active, structural_ids)
                action = winner['action']
                for point in active:
                    point.pop('_activated_from_suppressed', None)
            else:
                action, winner, converged = 'NONE', None, True
            after, summary = evaluate(active), {}
            for trial in trials:
                entry = summary.setdefault(trial['action'], dict(count=0, guard_blocked=0, best_raw_distance=1e9, best_cost_adjusted_distance=1e9))
                entry['count'] += 1
                entry['guard_blocked'] += not trial['admissible']
                entry['best_raw_distance'] = min(entry['best_raw_distance'], trial['raw_distance'])
                entry['best_cost_adjusted_distance'] = min(entry['best_cost_adjusted_distance'], trial['cost_adjusted_distance'])
            count += 1
            trace['iterations'].append(dict(iteration=count, P_before=before, S_before=sbefore,
                P_out=deepcopy(active), S_out=deepcopy(pool), phase='suppressed_pair_initialization' if initializing else 'correction',
                action=action, baseline=baseline, best_dist=after['objective'], improved=best is not None,
                added=search.difference(active, before), removed=search.difference(before, active),
                l_star=winner['l_star'] if winner else None, l_star_strategy=winner['segment_source'] if winner else None,
                components_before={k: v for k, v in base.items() if np.isscalar(v)}, components_after=after,
                reconstructed_before=search.curve(before, model), reconstructed_after=search.curve(active, model),
                trials=trials, evaluation_summary=summary, excluded_conflict_or_bounds=excluded,
                selected_cost_adjusted_distance=best_total, ranked_segments=ranked, guard_blocked_count=blocked,
                selected_guard=winner['initial_point_guard'] if winner else None, iteration_seconds=perf_counter()-started))
            _save(trace_file, trace)
            if not best:
                break
    reason = reason or ('no_admissible_improvement' if converged else 'maximum_iterations')
    score_after = score_markers(reference, search.coords(active), metric=metric, model=model)['objective'] if reference else None
    state = dict(version=STATE_VERSION, fingerprint=fingerprint, config=config, original_points=original,
                 active_points=active, suppressed=pool, initialized=initialized, iteration_count=count, converged=converged)
    trace.update(status='complete', stop_reason=reason, final_points=active, final_suppressed=pool,
                 original_retained=len(original)-len(search.difference(original, active)), reference_fixed=True,
                 seconds=perf_counter()-tick, path_state=state)
    _save(trace_file, trace)
    log_fn(f'[v46 path Step-5] {name}: {len(incoming)} -> {len(active)} points; {reason}; {metric}/{model}')
    return _plain(dict(name=name, points=[(p['cx'], p['cy']) for p in active], n_before=len(incoming), n_after=len(active),
                      suppressed=pool, status=reason if reason.endswith('preserved') or 'not_corrected' in reason else 'corrected',
                      stop_reason=reason, ssim_before=None, ssim_after=None, path_score_before=baseline_first,
                      path_score_after=score_after, metric=metric, model=model, reference_policy=config['reference_policy'],
                      inferred_weight=config['inferred_weight'], path_state=state, out_dir=str(folder),
                      reconstructed_path=search.curve(active, model),
                      iterations=count-(int(previous.get('iteration_count', 0)) if previous else 0)))

"""Per-series colour Step 5: connection-aware, candidate-only when disconnected.

Fitted curves never define new marker positions. Marker-only/uncertain series
never render fictitious connecting lines. Original detections are removed only
as supported duplicate observations of the same body, not for path disagreement.
"""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np

from color_series_structure_v46 import prepare, clean_fields, classify, Verifier, xy
from color_path_correction_v46 import _plain, _digest, correct_path_payload_curves
from color_estimated_path_v46 import full_path_segments
from color_tentative_policy_v46 import strong_evidence
from color_colocated_hypotheses_v46 import generate, ColocatedConfig
from color_limited_path_objective_v46 import LimitedPathObjective

VERSION = 'colour_series_limited_step5_v3_path_before_admission'


def save(path, obj):
    Path(path).write_text(json.dumps(_plain(obj), indent=2, allow_nan=False), encoding='utf-8')


def key(p):
    return tuple(np.round(xy([p])[0], 6))


def difference(a, b):
    existing = {key(p) for p in b}
    return [deepcopy(p) for p in a if key(p) not in existing]


def regular_slot(x, initial, tol):
    xs = np.unique(xy(initial)[:, 0])
    if len(xs) < 4 or not xs[0] < x < xs[-1]:
        return False
    gaps = np.diff(xs); step = np.median(gaps)
    if step <= 2*tol:
        return False
    multiples = np.rint(gaps/step)
    return bool((multiples == 1).sum() >= 3 and np.all((multiples >= 1) & (multiples <= 3))
        and max(abs(gaps-step*multiples)) <= min(tol, .08*step)
        and abs(x-xs[0]-round((x-xs[0])/step)*step) <= min(tol, .08*step))


def path_runs(record):
    path = np.asarray(record.get('path', []), float).reshape(-1, 2)
    filled = np.asarray(record.get('filled', []), bool)
    result, current = [], []
    for p, valid in zip(path, filled):
        if not valid or (current and p[0]-current[-1][0] > 1.5):
            if len(current) >= 2:
                result.append(current)
            current = []
        if valid:
            current.append(p.tolist())
    if len(current) >= 2:
        result.append(current)
    return result


def limited_series(row, current, structure, all_rows, verifier, limit, folder, previous=None, structural=(),
                   path_options=None, image_cost=None):
    sid = row['name']; mode = structure['mode']; d = row.get('diameter') or 10.
    registry = [dict(p) for p in row.get('init_points', [])]
    pool = [dict(p) for p in row.get('init_suppressed', []) if strong_evidence(p)]
    existing = {key(p) for p in pool}
    for p in structural:
        if p.get('source') == 'colocated_original_marker' and key(p) not in existing:
            pool.append(deepcopy(p)); existing.add(key(p))
    previous_points = {key(p): p for p in (previous or {}).get('active_points', [])}
    active = [dict(previous_points.get((float(x),float(y)), {}), cx=float(x), cy=float(y),
                   class_name=row.get('marker_class')) for x, y in xy(current['points'])]
    incoming = deepcopy(active)
    objective = LimitedPathObjective(structure, d, path_options,original_rows=all_rows)
    objective.bind_image_cost(image_cost,registry)
    soft_image = objective.image_cost is not None
    initial_path_score = objective.score(incoming)
    start_count = count = int((previous or {}).get('iteration_count', 0))
    cache = {}
    def verify(p):
        k = key(p)
        if k not in cache:
            cache[k] = verifier(sid, p) if verifier else dict(accepted=False, score=0., reason='no_template')
        return cache[k]
    trace_path = folder/'trace.json'
    history = []
    if previous and trace_path.exists():
        history = json.loads(trace_path.read_text(encoding='utf-8')).get('iterations', [])[:count]
    other_x = [p['cx'] for r in all_rows if r['name'] != sid for p in r.get('init_points', [])]
    reason = 'maximum_iterations'
    for _ in range(limit):
        before = deepcopy(active); proposals = []; trials = []
        baseline = objective.score(before)
        if mode in ('fitted_curve', 'markers_only'):
            # Same series, almost same centre, independently verified better body.
            # Same x alone or overlapping different series is NEVER a duplicate.
            for i, p in enumerate(active):
                for q in active[i+1:]:
                    if np.linalg.norm(xy([p])[0]-xy([q])[0]) >= max(1., .18*d):
                        continue
                    a, b = verify(p), verify(q)
                    if a['accepted'] and a['score'] > b['score']+.10 and not b['accepted']:
                        proposals.append(dict(action='REMOVE_DUPLICATE', point=q, gain=a['score']-b['score']))
                    elif b['accepted'] and b['score'] > a['score']+.10 and not a['accepted']:
                        proposals.append(dict(action='REMOVE_DUPLICATE', point=p, gain=b['score']-a['score']))
        # Structural uncertainty limits curve-driven edits, not image-candidate
        # review. By default uncertain series add independently supported
        # markers only. Opt-in soft-image ranking can activate hidden hypotheses
        # at the same observed slots, retaining their unconfirmed provenance.
        for p in pool:
            is_structural = p.get('source') == 'colocated_original_marker'
            # Measure the complete ADD reconstruction BEFORE any marker-image,
            # occlusion or same-x admission gate. Rejected trials keep their
            # measured geometry; this is not a hypothetical offline audit.
            comparison = objective.compare(before, before+[p], baseline)
            trial = dict(action='ACTIVATE_OCCLUSION' if is_structural else 'ACTIVATE_NATIVE',
                         candidate_id=p.get('candidate_id'), point=p, admissible=False,
                         path_comparison=comparison, path_gain=comparison['gain'],
                         image_admissible=False)
            if any(np.linalg.norm(xy([p])[0]-xy([q])[0]) <= .45*d for q in active):
                trial['reason']='already_active_nearby'; trials.append(trial); continue
            test = verify(p)
            trial['test'] = test
            trial['image_admissible'] = bool(test['accepted'] and
                (not is_structural or p.get('activation_eligible') is True))
            if soft_image:
                trial['image_policy']='weighted_cost_not_veto'
            if not soft_image and is_structural and p.get('activation_eligible') is not True:
                trial['reason']='occlusion_hypothesis_without_independent_fragment'
            elif not soft_image and not test['accepted']:
                trial['reason']='image_verification_failed'
            elif any(abs(q['cx']-p['cx']) <= .45*d for q in active):
                trial['reason']='target_x_slot_already_active'
            else:
                shared = any(abs(p['cx']-q) <= .45*d for q in other_x)
                regular = regular_slot(p['cx'], registry, .45*d)
                if shared or regular:
                    if objective.ranking_enabled and not comparison['improves']:
                        trial['reason']='no_objective_improvement' if objective.completeness.enabled or soft_image else 'no_path_improvement'
                    else:
                        gain=comparison['objective_gain'] if objective.ranking_enabled else test['score']
                        trial.update(admissible=True, gain=gain, image_score=test['score'],
                            ranking_basis=('path_completeness_image_reduction' if soft_image else
                                'path_plus_completeness_reduction' if objective.completeness.enabled else
                                'path_cost_reduction') if objective.ranking_enabled else 'image_score_no_supported_path',
                            reason='shared_observed_x_slot' if shared else 'repeated_observed_x_lattice')
                        proposals.append(dict(action=trial['action'],point=p,gain=gain,reason=trial['reason'],
                            image_score=test['score'],ranking_basis=trial['ranking_basis'],path_comparison=comparison))
                else:
                    trial['reason']='no_observed_measurement_slot'
            trials.append(trial)
        # Keep image-supported duplicate cleanup as an independent first phase.
        # Its image gain is not numerically compared with a curve-distance gain.
        # Eligible additions compete by path improvement when a supported path
        # exists, and by the legacy image score only when no such path exists.
        proposals.sort(key=lambda p: (0 if p['action']=='REMOVE_DUPLICATE' else 1,
                                      -p['gain'], p['action'], key(p['point'])))
        winner = proposals[0] if proposals else None
        for trial in trials:
            trial['selected'] = bool(winner and winner['action']==trial['action']
                                     and key(winner['point'])==key(trial['point']))
        if winner:
            p = winner['point']
            if winner['action'] == 'REMOVE_DUPLICATE':
                active = [q for q in active if key(q) != key(p)]
            else:
                active.append(dict(p, class_name=row.get('marker_class'), state='active_hypothesis',
                    **(dict(validation_basis='weighted_objective_not_pixel_confirmation',
                            image_confirmed=False) if soft_image else {})))
        count += 1
        history.append(dict(iteration=count, action=winner['action'] if winner else 'NONE',
            P_before=before, P_out=deepcopy(active), added=difference(active, before),
            removed=difference(before, active), selected=winner, proposals=proposals, candidate_reviews=trials,
            score_before=baseline['combined_objective'],score_after=objective.score(active)['combined_objective'],
            objective_before=baseline,objective_after=objective.score(active),
            no_path_y_objective=not objective.ranking_enabled,
            path_comparison_before_image_admission=True))
        if not winner:
            reason = ('no_objective_improving_candidate' if any(t.get('reason')=='no_objective_improvement' for t in trials)
                      else 'no_path_improving_candidate' if any(t.get('reason')=='no_path_improvement' for t in trials)
                      else 'no_available_candidate_slot' if soft_image and pool
                      else 'no_admissible_image_candidate' if pool else 'no_image_candidates')
            break
    remaining = [p for p in pool if key(p) not in {key(q) for q in active}]
    state = dict(version=VERSION, active_points=active, iteration_count=count,
                 mode=mode, original_points=registry, suppressed=deepcopy(remaining))
    runs = path_runs(structure['path_record']) if mode == 'fitted_curve' else []
    # The displayed curve is the extracted fit, NOT a path fitted to markers.
    model = 'fitted_reference' if mode == 'fitted_curve' else 'markers_only'
    trace = dict(version=VERSION, series=sid, mode=mode, initial_points=registry,
        initial_suppressed=pool, run_initial_points=incoming, final_points=active,
        final_suppressed=remaining, iterations=history, requested_iterations=start_count+limit,
        stop_reason=reason, structure=structure, candidate_tests=[dict(x=k[0], y=k[1], **v) for k, v in cache.items()],
        no_virtual_active_points=True, no_path_y_objective=not objective.ranking_enabled,
        path_objective=objective.describe(),path_comparison_before_image_admission=True,
        structural_suppressed_count=sum(p.get('source')=='colocated_original_marker' for p in remaining),
        prediction_status='unvalidated_hypotheses')
    save(trace_path, trace)
    return dict(name=sid, points=xy(active).tolist(), n_before=len(incoming), n_after=len(active),
        suppressed=remaining, status=reason, stop_reason=reason, mode=mode, model=model,
        reconstructed_path=[p for run in runs for p in run], reconstructed_segments=runs,
        reconstruction_source='extracted_reference_not_marker_interpolation' if runs else 'markers_only',
        series_state=state, ssim_before=None, ssim_after=None,
        path_score_before=initial_path_score['objective'],path_score_after=objective.score(active)['objective'],
        action_path_model=objective.model,metric=objective.metric,path_objective=objective.describe(),
        objective_before=initial_path_score,objective_after=objective.score(active),
        candidate_reviews=[dict(q, iteration=h['iteration']) for h in history
                           if h['iteration'] > start_count for q in h['candidate_reviews']])


def correct_series_payload_curves(image, curves, payload, *, out_dir, max_iters=None,
                                 previous_state=None, workers=None, engine_dir=None, log_fn=print,
                                 path_options=None):
    if path_options is None:
        from color_step5_defaults_v46 import production_options
        path_options=deepcopy((previous_state or {}).get('path_options',production_options()))
    limit = 5 if max_iters is None else int(max_iters)
    if limit < 1 or (max_iters is not None and limit != max_iters):
        raise ValueError('max_iters must be a positive integer')
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    rows = {r['name']: r for r in payload['curves']}
    if len({c['name'] for c in curves}) != len(curves) or set(rows) != {c['name'] for c in curves}:
        raise ValueError('Series routing requires every bound series exactly once')
    fingerprint = _digest(dict(version=VERSION, identity=payload['identity'],
        rows=[{k: v for k, v in r.items() if k not in ('ink_mask',)} for r in payload['curves']],
        options=path_options or {}))
    if previous_state is not None and previous_state.get('series_routing_fingerprint') != fingerprint:
        raise ValueError('Previous correction uses a different series policy/evidence; start from a fresh detection')
    evidence = prepare(image, payload)
    verifier = Verifier(evidence) if evidence is not None else None
    structures = {}; connected = []; prepared_payload = deepcopy(payload)
    for i, t in enumerate(evidence['templates'] if evidence else []):
        sid = t['id']; field, valid = clean_fields(evidence, i)
        visible = valid*(1-np.clip(evidence['other'][i], 0, 1))
        structure = classify(rows[sid]['init_points'], field, visible, t['diameter'],
                             np.asarray(payload['plot_area'][:2]))
        structure['path_record']['series_id'] = sid
        structures[sid] = structure
    for sid in rows:
        structures.setdefault(sid, dict(mode='uncertain', reason='no_bound_marker_template', path_record={}))
    # Candidate generation is independent of connected/fitted/uncertain routing.
    # Only immutable detector points donate coordinates; inferred paths never
    # create x measurement slots or move the donor centre.
    colocated = generate(payload, {sid:s['path_record'] for sid,s in structures.items()},
        ColocatedConfig(missing_x_only=True, allow_same_shape=True, retain_without_path=True))
    save(root/'shared_x_occlusion_hypotheses.json', colocated)
    # One frozen series-level decision from original detections, never recomputed
    # from newly added points during iterations or resume.
    for row in prepared_payload['curves']:
        sid = row['name']; structure = structures[sid]
        if structure['mode'] == 'connected':
            connected.append(next(c for c in curves if c['name'] == sid))
            rec = structure['path_record']
            prepared_payload['reference_paths'][sid] = rec
            row['segments_override'] = full_path_segments(rec, payload['plot_area'],
                payload.get('legend_box'), payload['grid_xs'])['segments']
    save(root/'series_structure.json', dict(version=VERSION, fingerprint=fingerprint,
         series=structures, policy='all_series_shared_x_hypotheses; uncertain_P0_preserved_with_image_candidate_review'))
    results = []
    if connected:
        # Even directly connected series need image-backed endpoint additions.
        results = correct_path_payload_curves(image, connected, prepared_payload, out_dir=root,
            max_iters=limit, previous_state=previous_state, workers=workers, engine_dir=engine_dir,
            log_fn=log_fn, proposal_validator=verifier, colocated_hypotheses=colocated,
            **{k:v for k,v in (path_options or {}).items()
               if k not in ('limited_completeness_weight','limited_image_weight','limited_model')})
        for result in results:
            result['mode'] = 'connected'
    for current in curves:
        sid = current['name']; mode = structures[sid]['mode']
        if mode == 'connected':
            continue
        folder = root/sid; folder.mkdir(exist_ok=True, parents=True)
        previous = (previous_state or {}).get('series_state', {}).get(sid)
        if previous_state is not None and previous is None:
            raise ValueError('Missing resumable series state for '+sid)
        image_cost=None
        if ((path_options or {}).get('limited_image_weight') is not None
                and verifier is not None and sid in getattr(verifier,'indices',{})
                and structures[sid].get('mode') in ('fitted_curve','uncertain')
                and structures[sid].get('eligible_pairs',0)>=1
                and structures[sid].get('inter_marker_path_support',0.)>=.55):
            from color_soft_image_cost_v46 import FrozenMarkerImageCost
            image_cost=FrozenMarkerImageCost(rows[sid],list(rows.values()),image,
                payload['plot_area'],payload.get('legend_box'),verifier)
        results.append(limited_series(rows[sid], current, structures[sid], list(rows.values()),
                                      verifier, limit, folder, previous, colocated['by_series'].get(sid,[]),
                                      path_options=path_options,image_cost=image_cost))
    for result in results:
        result['series_routing_fingerprint'] = fingerprint
        log_fn(f"[v46 series Step-5] {result['name']}: {result['mode']}, "
               f"{result['n_before']} -> {result['n_after']}; {result['stop_reason']}")
    by_name = {r['name']: r for r in results}
    return _plain([by_name[c['name']] for c in curves])

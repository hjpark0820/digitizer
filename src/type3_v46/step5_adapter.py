"""Production marker-free line-chart adapter for the v46 path Step-5 engine.

There are no marker templates in type 3. The native class enum is only a
transport token used by the action/NMS code, never a glyph observation or a
rendering instruction. A private copy of the production module substitutes an
explicit structural-evidence admission policy and disables shape-colocation.
No shared module globals are patched; every run owns private engine modules.
"""
from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1]

from color_estimated_path_v46 import full_path_segments

POLICY = 'type3_independent_source_structure_v1'
TRANSPORT_CLASS = 'filled_circle'


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), indent=2, allow_nan=False), encoding='utf-8')


def _private_module(name):
    filename = SRC / name
    spec = importlib.util.spec_from_file_location('type3_' + uuid.uuid4().hex, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def structural_admission(point):
    """Independent image gates, not score rank, path benefit, or tentative flag.

    Shared bar assignments, unreliable heights, grid-dominated paths, and all
    hard detector rejections remain review-only even when a local bar is strong.
    Stable corners require the already reviewed straight-polyline plot premise.
    """
    if 'endpoint_evidence' in point:
        from .type3_endpoint_candidates import endpoint_structural_admission
        return endpoint_structural_admission(point)
    evidence = point.get('evidence') or {}
    reasons = ' '.join(str(v) for v in (point.get('reason'), evidence.get('reason')))
    veto_words = ('shared', 'ambig', 'inconsistent', 'outside', 'duplicate', 'dominated',
                  'multiple_', 'unstable', 'instability', 'not_polyline', 'disagree', 'confound',
                  'alternate_', 'one_sided', 'large_extrapolation',
                  'short_horizontal_arm_may_be_neighboring_cap',
                  'insufficient_curve_support_beyond_cap', 'unconfirmed_vertical_structure')
    if point.get('status') in ('rejected', 'unresolved') or evidence.get('status') in ('rejected', 'unresolved'):
        return False, 'detector_rejection_is_not_strong_suppressed'
    if any(word in reasons for word in veto_words):
        return False, 'unresolved_assignment_or_geometry_veto'
    corner = evidence.get('corner') or {}
    windows = corner.get('per_window') or []
    good_windows = [w for w in windows if w.get('status') == 'supported'
                    and w.get('gates') and all(w['gates'].values())]
    if (corner.get('status') == 'supported' and corner.get('supported_windows', 0) >= 2
            and len(good_windows) >= 2):
        return True, 'stable_multiscale_source_polyline_intersection'
    bar = evidence.get('bar') or {}
    curve = evidence.get('curve') or {}
    bilateral = curve.get('bilateral') is True
    arms = [curve.get(side) or {} for side in ('left', 'right')]
    if (bar.get('status') == 'supported' and bilateral
            and all(arm.get('fit') and arm.get('source_columns', 0) >= 4 for arm in arms)):
        return True, 'observed_vertical_cap_and_bilateral_own_colour_arms'
    return False, 'no_independent_strong_type3_structure'


def type3_strong(point):
    return (point.get('suppressed_policy') == POLICY
            and point.get('type3_strong_structure') is True
            and structural_admission(point)[0])


def _isolated_engine():
    engine = _private_module('color_path_correction_v46.py')
    helpers = _private_module('color_path_correction_helpers_v46.py')
    grid = _private_module('color_step5_export_v46.py')
    helpers.strong_evidence = type3_strong
    grid.strong_evidence = type3_strong
    engine.strong_evidence = type3_strong
    engine.search = helpers
    # Shape alternatives require real legend templates; none exist here.
    engine.generate = lambda payload, paths: dict(
        by_series={r['name']: [] for r in payload['curves']}, sites=[], duplicate_merges=[],
        audit=dict(disabled=True, reason='type3_has_no_marker_glyph_or_template'))
    return engine, grid


def _inclusive(box):
    values = np.asarray(box, float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[2:] <= values[:2]):
        raise ValueError('Boxes must be finite nonempty half-open xyxy source boxes')
    return [float(values[0]), float(values[1]), float(values[2]-1), float(values[3]-1)]


def _input_point(point):
    x, y = float(point['x_px']), float(point['y_px'])
    if not np.isfinite([x, y]).all():
        raise ValueError('Measurement hypotheses must have finite source coordinates')
    return dict(deepcopy(point), cx=x, cy=y, class_name=TRANSPORT_CLASS, class_idx=0,
                candidate_id=str(point.get('id', f'{x:.6f}:{y:.6f}')),
                kind='measurement_center_hypothesis', marker_glyph_detected=False)


def _canonical(point, sid):
    original = '_initial_id' in point
    measurement = original or point.get('kind') == 'measurement_center_hypothesis'
    evidence=point.get('endpoint_evidence') or {}
    prior_only=(evidence.get('step5_admission_policy')=='global_endpoint_strong_prior_v1'
                and evidence.get('treat_as_strong_for_step5') is True
                and evidence.get('strong_suppressed') is not True)
    kind = ('endpoint_prior_correction_hypothesis' if prior_only else
            'measurement_center_hypothesis' if measurement else 'path_only_correction_hypothesis')
    output = {k: deepcopy(v) for k, v in point.items()
              if k not in ('cx', 'cy', 'class_idx', 'class_name', '_activated_from_suppressed', 'evidence')}
    if 'evidence' in point:
        output['evidence_ref'] = dict(source='detection.json', series_id=sid,
                                      candidate_id=point.get('id', point.get('candidate_id')))
    output.update(series_id=sid, x_px=float(point['cx']), y_px=float(point['cy']), kind=kind,
                  marker_glyph_detected=False, original_L0=original,
                  independent_measurement_evidence=bool(measurement and not prior_only),
                  existence='unvalidated', transport_class_is_not_symbol_identity=True)
    output.setdefault('id', point.get('candidate_id', f'{sid}:{point["cx"]:.6f}:{point["cy"]:.6f}'))
    if prior_only:
        output['strong_by_endpoint_prior']=True
    return output


def compact_summary_evidence(summary_path):
    """Remove redundant source pixels from a finished summary, not native traces.

    Detection.json and native trace.json remain the authoritative full evidence.
    Every compacted point preserves its exact coordinates, ID and decision.
    """
    path = Path(summary_path)
    record = json.loads(path.read_text(encoding='utf-8'))
    removed = 0
    def visit(value):
        nonlocal removed
        if isinstance(value, dict):
            if 'evidence' in value and ('x_px' in value or 'cx' in value):
                value.pop('evidence')
                value['evidence_ref'] = dict(source='detection.json', series_id=value.get('series_id'),
                                             candidate_id=value.get('id', value.get('candidate_id')))
                removed += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(record)
    record['summary_evidence_storage'] = dict(full_evidence='parent case detection.json and native trace.json',
                                             redundant_point_evidence_blocks_removed=removed,
                                             point_coordinates_or_actions_changed=False)
    _write(path, record)
    return dict(path=str(path), removed=removed, bytes=path.stat().st_size)


def _difference(first, second):
    keys = {(round(p['cx'], 6), round(p['cy'], 6)) for p in second}
    return [p for p in first if (round(p['cx'], 6), round(p['cy'], 6)) not in keys]


def _record(path, roi, legend):
    xy = np.asarray(path['raw_path_source'], float).reshape(-1, 2)
    n = len(xy)
    obs = np.asarray(path.get('raw_observed', np.zeros(n)), bool)
    valid = np.asarray(path.get('valid', np.ones(n)), bool)
    gl = np.asarray(path.get('grid_likelihood', np.zeros(n)), float)
    confidence = np.asarray(path.get('path_color_values', np.ones(n)), float)
    if any(a.shape != (n,) for a in (obs, valid, gl, confidence)):
        raise ValueError('Raw path flags must match its number of samples')
    if not np.isfinite(xy).all() or (n > 1 and np.any(np.diff(xy[:, 0]) <= 0)):
        raise ValueError('Raw paths must be finite and strictly increasing in x')
    if not np.isfinite(gl).all() or np.any((gl < 0) | (gl > 1)):
        raise ValueError('Grid likelihood must be in [0, 1]')
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Path color confidence must be in [0, 1]')
    scope = valid & (xy[:, 0] >= roi[0]) & (xy[:, 0] <= roi[2]) & (xy[:, 1] >= roi[1]) & (xy[:, 1] <= roi[3])
    if legend is not None:
        scope &= ~((xy[:, 0] >= legend[0]) & (xy[:, 0] <= legend[2]) & (xy[:, 1] >= legend[1]) & (xy[:, 1] <= legend[3]))
    reference_allowed = None
    if 'reference_allowed' in path:
        reference_allowed = np.asarray(path['reference_allowed'])
        if reference_allowed.shape != (n,) or (n and reference_allowed.dtype.kind != 'b'):
            raise ValueError('reference_allowed must be a boolean array matching raw_path_source')
        reference_allowed = reference_allowed.astype(bool)
        for name in ('support_kind', 'original_raw_observed'):
            if name in path and np.asarray(path[name]).shape != (n,):
                raise ValueError(f'{name} must match raw_path_source')
        scope &= reference_allowed
    # Legacy inputs keep all in-scope geometry. Explicit support gates remove
    # rejected coordinates before estimated_path can assign inference weight.
    score_observed = obs & (gl < .5)
    record = dict(series_id=path['series_id'], path=xy[scope].tolist(),
                observed=score_observed[scope].tolist(), filled=np.ones(scope.sum(), bool).tolist(),
                val=confidence[scope].tolist(), raw_observed=obs[scope].tolist(),
                confidence_source=('source_soft_colour_values' if 'path_color_values' in path
                                   else 'unit_confidence_fallback_missing_source_values'),
                grid_likelihood=gl[scope].tolist(), source_indices=np.flatnonzero(scope).tolist(),
                scope_excluded_samples=int((~scope).sum()), raw_path_samples=n,
                grid_samples_retained_as_estimated=int((scope & ~score_observed & obs).sum()))
    if reference_allowed is not None:
        record.update(reference_allowed=reference_allowed[scope].tolist(),
                      support_gate_enabled=True,
                      support_excluded_samples=int((~reference_allowed).sum()))
        for name in ('support_kind', 'original_raw_observed'):
            if name in path:
                record[name] = np.asarray(path[name])[scope].tolist()
    return record


def _retained_runs(record):
    """Return index runs; a removed source index is a boundary at any scale."""
    xy = np.asarray(record['path'], float).reshape(-1, 2)
    indices = np.asarray(record.get('source_indices', np.arange(len(xy))), int)
    if not len(xy):
        return []
    breaks = np.flatnonzero((np.diff(indices) != 1) | (np.diff(xy[:, 0]) > 1.5)) + 1
    return np.split(np.arange(len(xy)), breaks)


def _run_record(record, run):
    n = len(record['path'])
    return {k: [v[i] for i in run] if isinstance(v, list) and len(v) == n else v
            for k, v in record.items()}


def _reference_segments(record, roi, legend, grid):
    if not record.get('support_gate_enabled'):
        return full_path_segments(record, roi, legend, grid)
    parts = [full_path_segments(_run_record(record, run), roi, legend, grid)
             for run in _retained_runs(record)]
    # Keep the familiar aggregate provenance and make each original-index run
    # inspectable; production's x-gap threshold alone cannot see subpixel holes.
    audit = full_path_segments(dict(record, path=[], filled=[], observed=[], val=[]),
                               roi, legend, grid)['provenance']
    for key in ('raw_path_samples', 'retained_path_samples', 'observed_path_samples',
                'short_gap_or_weak_samples', 'retained_runs', 'rejected_outside_or_legend_segments',
                'original_retained_samples', 'previously_rejected_samples_included',
                'evaluated_path_samples', 'inferred_path_samples'):
        audit[key] = sum(p['provenance'].get(key, 0) for p in parts)
    audit.update(source='support_allowed_source_index_runs_to_native_full_path_segments',
                 support_gate_enabled=True, source_index_runs=[
                     [record['source_indices'][int(run[0])], record['source_indices'][int(run[-1])]]
                     for run in _retained_runs(record)],
                 run_provenance=[p['provenance'] for p in parts])
    if parts:
        audit.pop('status', None)
    return dict(segments=[s for p in parts for s in p['segments']], provenance=audit)


class _SupportGate:
    """Admit mathematical knots only near a retained reference run.

    Original L0 coordinates and independently vetted suppressed centers retain
    their source evidence. Merely copying their metadata onto a moved knot
    cannot exempt the moved coordinate from this geometrical gate.
    """
    def __init__(self, record, row):
        self.xy = np.asarray(record['path'], float).reshape(-1, 2)
        self.runs = _retained_runs(record)
        self.tolerance = max(1., float(row['source_line_width']))
        self.originals = {(float(p['cx']), float(p['cy'])) for p in row['init_points']}
        self.structural = {(float(p['cx']), float(p['cy'])) for p in row['init_suppressed']
                           if type3_strong(p)}
        self.rejected_proposals = []

    def coordinate(self, x, y):
        xy = np.asarray([x, y], float)
        containing = [run for run in self.runs
                      if self.xy[run[0], 0] - 1e-6 <= x <= self.xy[run[-1], 0] + 1e-6]
        neighbourhood = self.xy[np.concatenate(containing)] if containing else self.xy
        distance = float(np.linalg.norm(neighbourhood - xy, axis=1).min()) if len(neighbourhood) else None
        near = bool(distance is not None and distance <= self.tolerance + 1e-6)
        return dict(allowed=bool(containing and near), x_px=float(x), y_px=float(y),
                    reason=('allowed_retained_reference_neighbourhood' if containing and near else
                            'outside_retained_reference_x_runs' if not containing else
                            'too_far_from_retained_reference_path'),
                    nearest_reference_distance_px=distance, tolerance_px=self.tolerance)

    def evaluate(self, points):
        checks = []
        for p in points:
            key = float(p['cx']), float(p['cy'])
            if key in self.originals or key in self.structural:
                continue
            checks.append(self.coordinate(*key))
        return dict(allowed=all(p['allowed'] for p in checks), checks=checks,
                    blocked_points=sum(not p['allowed'] for p in checks),
                    original_L0_coordinates_exempt=True,
                    independent_structural_coordinates_exempt=True)

    def clip_segments(self, segments):
        """Clip native/reconstructed segments separately inside each source run."""
        output = []
        for segment in segments:
            a, b = np.asarray(segment, float).reshape(2, 2)
            if a[0] > b[0]:
                a, b = b, a
            if b[0] <= a[0]:
                continue
            for run in self.runs:
                lo, hi = max(a[0], self.xy[run[0], 0]), min(b[0], self.xy[run[-1], 0])
                if hi <= lo:
                    continue
                first = a + (b - a) * ((lo - a[0]) / (b[0] - a[0]))
                last = a + (b - a) * ((hi - a[0]) / (b[0] - a[0]))
                output.append([*first.tolist(), *last.tolist()])
        return output

    def describe(self):
        return dict(enabled=True, tolerance_px=self.tolerance,
                    tolerance_rule='max(1 source pixel, source line_width)',
                    x_runs=[[float(self.xy[r[0], 0]), float(self.xy[r[-1], 0])] for r in self.runs],
                    support_rejected_proposal_count=len(self.rejected_proposals),
                    rejected_proposals=self.rejected_proposals,
                    original_L0_coordinates_exempt=True, independent_structural_coordinates_exempt=True)


def _install_support_gate(engine, payload):
    """Wrap only this run's private modules, preserving native metric/NMS/L0 code."""
    gates = {row['name']: _SupportGate(payload['reference_paths'][row['name']], row)
             for row in payload['curves']
             if payload['reference_paths'].get(row['name'], {}).get('support_gate_enabled')}
    if not gates:
        return gates
    context = ContextVar('type3_reference_support', default=None)
    native_run, native_load = engine._run_series, engine._load_native
    native_guard, native_reference = engine.InitialPointGuard, engine.PathReference
    native_propose = engine.search.propose
    native_pool_proposals = engine.search.add_pool_proposals
    native_candidates = engine.search.candidate_segments

    def run(current, *args, **kwargs):
        token = context.set(gates.get(current['name']))
        try:
            return native_run(current, *args, **kwargs)
        finally:
            context.reset(token)

    def load(filename):
        module = native_load(filename)
        gate = context.get()
        if gate is not None and Path(filename).name == '4_segment_refinement.py':
            native_refine = module.refine
            def refine(segments, grid, *args, **kwargs):
                refined, audit = native_refine(gate.clip_segments(segments), grid, *args, **kwargs)
                clipped = gate.clip_segments(refined)
                return clipped, dict(audit, support_gate_enabled=True,
                                     reference_holes_never_bridged=True, n_kept=len(clipped))
            module.refine = refine
        return module

    class Guard(native_guard):
        def evaluate(self, before, trial):
            original = super().evaluate(before, trial)
            gate = context.get()
            if gate is None:
                return original
            support = gate.evaluate(trial)
            return dict(original, allowed=original['allowed'] and support['allowed'],
                        original_L0_allowed=original['allowed'], support_gate=support)

    class Reference:
        @staticmethod
        def from_record(record, diameter, confidence=None, config=None, **kwargs):
            reference = native_reference.from_record(record, diameter, confidence, config, **kwargs)
            if not record.get('support_gate_enabled') or not len(record['path']):
                return reference
            parts = []
            for indices in _retained_runs(record):
                options = dict(kwargs)
                if options.get('scope_mask') is not None:
                    options['scope_mask'] = np.asarray(options['scope_mask'])[indices]
                conf = np.asarray(confidence)[indices] if confidence is not None else None
                parts.append(native_reference.from_record(_run_record(record, indices), diameter,
                                                          conf, config, **options))
            fields = {}
            for field in ('weights', 'sample_weights', 'sample_tangents', 'sample_tangent_valid',
                          'segment_starts', 'segment_ends'):
                values = np.concatenate([getattr(p, field) for p in parts])
                values.flags.writeable = False
                fields[field] = values
            return replace(reference, **fields)

    def candidates(active, model, grid):
        segments = native_candidates(active, model, grid)
        gate = context.get()
        return gate.clip_segments(segments) if gate else segments

    def propose(eng, active, suppressed, segments, *args):
        gate = context.get()
        return native_propose(eng, active, suppressed,
                              gate.clip_segments(segments) if gate else segments, *args)

    def pool_proposals(eng, active, suppressed, items, roi, legend):
        items = native_pool_proposals(eng, active, suppressed, items, roi, legend)
        gate = context.get()
        if gate is None:
            return items
        retained = []
        for item in items:
            check = gate.evaluate(item['points'])
            if check['allowed']:
                retained.append(dict(item, support_gate_before_nms=check))
            else:
                gate.rejected_proposals.append(dict(action=item['action'],
                    segment_source=item.get('segment_source'), l_star=item.get('l_star'),
                    reason='blocked_by_reference_support_gate', support_gate=check,
                    added_points=deepcopy(_difference(item['points'], active))))
        return retained

    engine._run_series, engine._load_native = run, load
    engine.InitialPointGuard, engine.PathReference = Guard, Reference
    engine.search.propose, engine.search.candidate_segments = propose, candidates
    engine.search.add_pool_proposals = pool_proposals
    return gates


def build_payload(detection, grid_module=None):
    """No image-derived points or path coordinates are generated in this step."""
    if grid_module is None:
        _, grid_module = _isolated_engine()
    roi = _inclusive(detection['plot_box'])
    legend = _inclusive(detection['legend_box']) if detection.get('legend_box') is not None else None
    rows, active, suppressed, audit = [], {}, {}, []
    data_series = [s for s in detection['series'] if s.get('role', 'data_series') == 'data_series']
    if len({s['id'] for s in data_series}) != len(data_series):
        raise ValueError('Series IDs must be unique')
    widths = []
    for s in data_series:
        sid = s['id']
        width = float(s.get('line_width', 2.))
        if not np.isfinite(width) or width <= 0:
            raise ValueError('line_width must be finite and positive')
        # Explicit geometrical support scale replaces nonexistent marker size.
        diameter = 4*width
        widths.append(diameter)
        aa = [_input_point(p) for p in detection.get('points', []) if p['series_id'] == sid]
        ss = []
        for p in detection.get('tentative_points', []):
            if p['series_id'] != sid:
                continue
            okay, reason = structural_admission(p)
            audit.append(dict(series_id=sid, candidate_id=p.get('id'), admitted=okay, reason=reason))
            if okay:
                q = _input_point(p)
                from .type3_endpoint_candidates import endpoint_prior_admission
                prior_admitted=endpoint_prior_admission(p)
                q.update(type3_strong_structure=True, suppressed_policy=POLICY, class_name='suppressed',
                         class_idx=-1, evidence_tier=('strong_endpoint_prior' if prior_admitted
                                                     else 'strong_structural_hypothesis'), state='suppressed',
                         existence='unknown', tentative=True, auto_promote=False)
                if prior_admitted:
                    q['strong_evidence_basis']='user_endpoint_prior_not_new_source_evidence'
                ss.append(q)
        active[sid], suppressed[sid] = aa, ss
        rows.append(dict(name=sid, marker_class=TRANSPORT_CLASS, diameter=diameter,
                         diameter_interpretation='four_source_stroke_widths_not_marker_diameter',
                         source_line_width=width, init_points=aa, init_suppressed=ss,
                         marker_alpha=None, segments_override=[]))
    grid, grid_audit = grid_module.cluster_marker_columns(active, widths, suppressed)
    records = {}
    path_by_id = {p['series_id']: p for p in detection.get('paths', [])}
    segment_audit = {}
    for row in rows:
        sid = row['name']
        if sid in path_by_id:
            record = _record(path_by_id[sid], roi, legend)
            records[sid] = record
            seg = _reference_segments(record, roi, legend, grid)
            row['segments_override'] = seg['segments']
            segment_audit[sid] = seg['provenance']
    payload = dict(identity=dict(case_id=detection['case_id'], source_sha256=detection.get('source_sha256')),
                   plot_area=roi, legend_box=legend, curves=rows, grid_xs=grid, reference_paths=records)
    return payload, dict(suppressed_admission=audit, grid=grid_audit, segments=segment_audit,
                         policy=POLICY, independent_image_gates=True, template_invented=False,
                         reference_scope=('valid source plot samples excluding legend, intersected with '
                                          'reference_allowed when supplied; no invented extrapolation'))


def _iteration(entry, sid):
    before, after = entry['P_before'], entry['P_out']
    trials = []
    outkeys = {(p['cx'], p['cy']) for p in after}
    for t in entry.get('trials', []):
        chosen = (entry.get('improved') and t['action'] == entry['action']
                  and {(p['cx'], p['cy']) for p in t['evaluated_points']} == outkeys
                  and abs(t['cost_adjusted_distance'] - entry['selected_cost_adjusted_distance']) < 1e-9)
        improved = t['cost_adjusted_distance'] < entry['baseline'] - 1e-7
        original_guard = t['initial_point_guard']
        support = original_guard.get('support_gate')
        original_allowed = original_guard.get('original_L0_allowed', original_guard['allowed'])
        why = ('selected_best_admissible_improvement' if chosen else
               'blocked_by_original_L0_and_reference_support_guards' if not original_allowed and support and not support['allowed'] else
               'blocked_by_reference_support_gate' if support and not support['allowed'] else
               'blocked_by_original_L0_guard' if not original_allowed else
               'no_cost_adjusted_improvement' if not improved else 'outperformed_by_other_action')
        trials.append(dict(action=t['action'], selected=bool(chosen), admissible=t['admissible'], reason=why,
                           score_before=entry['baseline'], score_after=t['raw_distance'],
                           cost_adjusted_score=t['cost_adjusted_distance'],
                           added=[_canonical(p, sid) for p in _difference(t['evaluated_points'], before)],
                           removed=[_canonical(p, sid) for p in _difference(before, t['evaluated_points'])],
                           initial_point_guard=dict(original_guard, allowed=original_allowed),
                           support_gate=support,
                           segment_source=t.get('segment_source'), l_star=t.get('l_star')))
    return dict(iteration=entry['iteration'], executed=True, action=entry['action'],
                status='accepted_action' if entry['improved'] else 'no_admissible_improvement',
                phase=entry.get('phase'), score_before=entry['baseline'], score_after=entry['best_dist'],
                points_before=[_canonical(p, sid) for p in before],
                points_after=[_canonical(p, sid) for p in after],
                suppressed_before=[_canonical(p, sid) for p in entry['S_before']],
                suppressed_after=[_canonical(p, sid) for p in entry['S_out']],
                added=[_canonical(p, sid) for p in entry['added']],
                removed=[_canonical(p, sid) for p in entry['removed']], trials=trials,
                guard_blocked_count=sum(not t['initial_point_guard'].get('original_L0_allowed',
                    t['initial_point_guard']['allowed']) for t in entry.get('trials', [])),
                support_gate_blocked_count=sum(not t['initial_point_guard'].get('support_gate',
                    {'allowed': True})['allowed'] for t in entry.get('trials', [])),
                selected_guard=entry.get('selected_guard'),
                reconstructed_before=entry.get('reconstructed_before', []),
                reconstructed_after=entry.get('reconstructed_after', []),
                l_star=entry.get('l_star'), segment_source=entry.get('l_star_strategy'),
                excluded_conflict_or_bounds=entry.get('excluded_conflict_or_bounds', 0))


def run_step5(detection, output_dir, iterations=5, *, previous_state=None, engine_dir=None, current_curves=None):
    """Run unchanged v46 geometric correction orchestration with local type3 policy.

    The JSON contract uses original-image coordinates; input plot/legend boxes
    are HALF OPEN. Five slots may include explicit unexecuted convergence slots.
    """
    if int(iterations) != iterations or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    root = Path(output_dir)
    if (root / 'step5.json').exists() or (root / 'native').exists():
        raise FileExistsError('Refusing to overwrite an existing correction experiment')
    source = Path(detection['source_image_path'])
    expected_hash = detection.get('source_sha256')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_hash and expected_hash != source_hash:
        raise ValueError('Source image hash changed since detection')
    image = np.asarray(Image.open(source).convert('RGB'))
    engine, grid = _isolated_engine()
    payload, input_audit = build_payload(detection, grid)
    if detection.get('identity') is not None:
        payload['identity'] = deepcopy(detection['identity'])
    support_gates = _install_support_gate(engine, payload)
    curves = [dict(name=r['name'], points=deepcopy(r['init_points'])) for r in payload['curves']]
    if previous_state is not None:
        if previous_state.get('identity') != payload['identity']:
            raise ValueError('Previous type3 state does not match this image, geometry and series')
        for curve in curves:
            values = previous_state.get('curves', {}).get(curve['name'])
            if values is None:
                raise ValueError('Previous type3 state is missing a series')
            curve['points'] = deepcopy(values)
    root.mkdir(parents=True, exist_ok=True)
    if current_curves is not None:
        if set(current_curves) != {c['name'] for c in curves}:
            raise ValueError('Edited type3 series identities differ from saved evidence')
        for curve in curves:
            curve['points'] = deepcopy(current_curves[curve['name']])
    _write(root / 'payload.json', payload)
    _write(root / 'input_audit.json', input_audit)
    result = engine.correct_path_payload_curves(image, curves, payload, out_dir=root/'native',
        max_iters=int(iterations), workers=1, metric='chamfer', model='linear',
        reference_policy='estimated_path', inferred_weight=.25,
        previous_state=previous_state, engine_dir=engine_dir)
    metadata = {s['id']: s for s in detection['series']}
    output_series = []
    for row in result:
        sid = row['name']
        trace = json.loads((root/'native'/sid/'trace.json').read_text(encoding='utf-8'))
        entries = [_iteration(t, sid) for t in trace['iterations']]
        previous_iterations = int(trace.get('previous_iteration_count', 0))
        final = [_canonical(p, sid) for p in trace['final_points']]
        suppressed = [_canonical(p, sid) for p in trace['final_suppressed']]
        gate = support_gates.get(sid)
        forbidden_added = []
        if gate is not None:
            for entry in trace['iterations']:
                for point in entry['added']:
                    check = gate.evaluate([point])
                    if not check['allowed']:
                        forbidden_added.append(dict(iteration=entry['iteration'], point=_canonical(point, sid),
                                                    support_gate=check))
            if forbidden_added:
                raise RuntimeError(f'{sid}: native Step5 accepted points outside the explicit reference support gate')
        for n in range(len(entries)+1, int(iterations)+1):
            entries.append(dict(iteration=previous_iterations+n, executed=False, action='NO_OP', phase='not_executed',
                status='not_executed_after_convergence' if row['stop_reason']=='no_admissible_improvement'
                       else 'not_executed_'+row['stop_reason'],
                score_before=row['path_score_after'], score_after=row['path_score_after'],
                points_before=deepcopy(final), points_after=deepcopy(final),
                suppressed_before=deepcopy(suppressed), suppressed_after=deepcopy(suppressed),
                added=[], removed=[], trials=[], guard_blocked_count=0,
                reconstructed_before=row['reconstructed_path'], reconstructed_after=row['reconstructed_path']))
        output_series.append(dict(id=sid, label=metadata[sid].get('label', sid), rgb=metadata[sid]['rgb'],
            initial_points=[_canonical(p, sid) for p in trace.get('run_initial_points', trace['initial_points'])],
            original_points=[_canonical(p, sid) for p in trace['initial_points']], final_points=final,
            initial_suppressed=[_canonical(p, sid) for p in trace['initial_suppressed']], final_suppressed=suppressed,
            stop_reason=row['stop_reason'], score_before=row['path_score_before'], score_after=row['path_score_after'],
            iterations=entries, executed_iterations=len(trace['iterations']),
            previous_iteration_count=previous_iterations,
            raw_trace=str(root/'native'/sid/'trace.json'), reference_description=trace['reference_description'],
            supplied_segments=trace['supplied_segments'], refined_segments=trace.get('refined_segments', []),
            support_gate=gate.describe() if gate is not None else dict(enabled=False),
            forbidden_added_points=forbidden_added, forbidden_added_point_count=len(forbidden_added)))
    summary = dict(case_id=detection['case_id'], requested_iterations=int(iterations),
        point_kind='measurement_center_hypotheses_not_marker_glyphs', source_sha256=source_hash,
        method=dict(metric='chamfer', model='linear', reference_policy='estimated_path', inferred_weight=.25,
                    grid_treatment='method6_non_deletion_path_input',
                    production_function='color_path_correction_v46.correct_path_payload_curves',
                    original_point_guard=True, suppression_policy=POLICY,
                    transport_class=TRANSPORT_CLASS, transport_class_is_not_symbol_identity=True,
                    template_invented=False, shape_colocation_disabled=True,
                    geometry_scale='4 * source line_width, not nonexistent marker diameter',
                    reference_support_gate=bool(support_gates),
                    original_point_guard_logic_modified=False, native_nms_modified=False),
        production_modified=True, source_points_modified=False, reference_accuracy_claimed=False,
        series=output_series, input_audit=input_audit,
        identity=deepcopy(payload['identity']),
        path_state={r['name']:r['path_state'] for r in result},
        limitation='Step5 optimizes agreement with the extracted path, not measurement existence. '
                   'ADD/REPLACE path-only knots are explicitly separated from source-evidenced centers.')
    _write(root/'step5.json', summary)
    return _plain(summary)

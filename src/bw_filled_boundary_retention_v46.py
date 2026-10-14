"""Retain native filled-outline evidence without certifying a missing body.

Only raw grid seeds rejected solely by the centre-fill gate enter this lane.
They retain their own legend geometry and need a fresh foreground-scene test
in Step 5; another already active marker is a proposal prerequisite, not proof.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math

VERSION = 'filled-boundary-foreground-retention-v1'


def retain(results, models, active, suppressed, plot, exclusions,
           series_indices, *, maximum_candidates=48):
    """Return additive S-only hypotheses and a compact audit; never edit inputs."""
    reasons = Counter()
    additions = []
    records = [row for result in results for row in getattr(result, 'filled_occlusion_evidence', [])]
    records.sort(key=lambda r: (-float(r.get('metrics', {}).get('contour_coverage', 0.)),
                                -float(r.get('metrics', {}).get('direct_foreground_coverage', 0.))))
    keys = {(p['swatch_id'], round(float(p['cx']), 6), round(float(p['cy']), 6)) for p in active + suppressed}
    for row in records:
        sid = row.get('template')
        model = models.get(sid)
        if (model is None or not str(model['class_name']).startswith('filled_') or
                row.get('contract') != 'raw_filled_boundary_requires_foreground_v1' or
                row.get('rejection') != 'centre_fill' or row.get('native_passed') is not False or
                row.get('evidence_status') != 'native_verifier'):
            reasons['invalid_native_evidence'] += 1
            continue
        x, y = float(row['x']), float(row['y'])
        scale = float(model['marker_scale'])
        if not all(map(math.isfinite, (x, y, scale))) or scale <= 0:
            reasons['invalid_geometry'] += 1
            continue
        if abs(float(row['proposal_scale']) - scale) > 1.e-6:
            reasons['shared_scale_mismatch'] += 1
            continue
        key = sid, round(x, 6), round(y, 6)
        if key in keys:
            reasons['exact_existing_hypothesis'] += 1
            continue
        if not (plot[0] <= x < plot[2] and plot[1] <= y < plot[3]) or any(a <= x < c and b <= y < d for a, b, c, d in exclusions):
            reasons['outside_source_domain'] += 1
            continue
        diameter = float(model['source_diameter']) * scale
        if any(p['swatch_id'] == sid and math.hypot(p['cx'] - x, p['cy'] - y) < .45 * diameter for p in active):
            reasons['same_series_active_body'] += 1
            continue
        donors = [p for p in active if p.get('original_detection', False)
            and not any(name in p for name in ('pending_identity', 'partial_visibility_certificate',
                'partial_visibility_hypothesis', 'overlap_hypothesis', 'overlap_recheck', 'seed_bootstrap'))
            and math.hypot(p['cx'] - x, p['cy'] - y) < .5 * (diameter + float(p.get('effective_diameter', diameter)))]
        if not donors:
            reasons['no_independent_nearby_active_marker'] += 1
            continue
        if len(additions) >= maximum_candidates:
            reasons['candidate_budget'] += 1
            continue
        token = hashlib.sha256(f'{sid}|{x:.6f}|{y:.6f}|{scale:.8f}'.encode()).hexdigest()[:14]
        cid = 'FB_' + token
        additions.append(dict(swatch_id=sid, template=sid, class_name=model['class_name'],
            shape_hint=model['class_name'], class_idx=series_indices[sid],
            cx=x, cy=y, candidate_id=cid, point_id=cid, source=VERSION,
            original_detection=False, state='suppressed', tentative=True,
            confidence=0., confidence_kind='unverified_boundary_hypothesis_not_probability',
            marker_mask=deepcopy(model['marker_mask']), marker_offset_x=float(model.get('marker_offset_x', 0.)),
            marker_offset_y=float(model.get('marker_offset_y', 0.)),
            marker_scale=scale, marker_aspect=1., source_diameter=float(model['source_diameter']),
            effective_diameter=diameter, activation_eligible=True, auto_promote=False,
            admission_reason='supported_filled_boundary_needs_foreground',
            grid_boundary_evidence=deepcopy(row),
            overlap_hypothesis=dict(version=VERSION, reason='supported_filled_boundary_needs_foreground',
                requires_independent_foreground=True, active_authority=False,
                nearby_active_ids=[p.get('point_id') for p in donors],
                center_policy='unrefined_grid_seed', native_full_window_passed=False)))
        keys.add(key)
    return additions, dict(version=VERSION, status='completed', input_records=len(records),
        added=len(additions), maximum_candidates=maximum_candidates, reasons=dict(reasons),
        active_points_modified=False, actual_occlusion_verified=False,
        added_centers=[dict(swatch_id=p['swatch_id'], cx=p['cx'], cy=p['cy'], candidate_id=p['candidate_id']) for p in additions])

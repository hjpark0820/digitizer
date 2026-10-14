"""Conservative, opt-in cold starts from two existing BW marker hypotheses.

This module proposes atomic edits; it never mutates a point, discovers a new
coordinate, changes a raster, or relaxes the dispatcher's marker tests.  The
caller invokes ``propose_pairs`` only when ordinary single activation has no
admissible positive-gain action for the initially empty series.  A first pair
does not establish the repeated evidence needed for connected-series routing.
"""
from itertools import combinations
import math

import cv2
import numpy as np

from bw_series_structure_v46 import chord_test
from bw_suppressed_v46 import decode_marker_mask

POLICY = 'independent_source_pair_init_v1'
MAX_CANDIDATES = 12
MAX_PAIRS = 66
_CONTEXT_KEYS = ('pending_identity', 'partial_visibility_hypothesis',
                 'partial_visibility_certificate', 'overlap_hypothesis', 'overlap_recheck')
_KERNEL = np.ones((3, 3), np.uint8)


def _ordinary(point, model):
    if any(name in point for name in _CONTEXT_KEYS):
        return 'context_dependent_marker'
    source = str(point.get('source', '')).lower().replace('_', '').replace('-', '').replace(' ', '')
    tier = str(point.get('evidence_tier', '')).lower().replace('_', '').replace('-', '').replace(' ', '')
    if ('structuralprior' in source or source in ('xmissingseriesoverlapv2', 'xsingletonoverlapv1') or
            ('structural' in tier and 'prior' in tier)):
        return 'structural_prior_not_independent'
    if point.get('admission_reason') == 'verified_identity_ambiguous':
        return 'context_dependent_marker'
    if (model is None or model.get('class_name') in (None, '', 'unknown_marker', 'suppressed')
            or model.get('provenance') == 'pending_only_marker_raster'):
        return 'unknown_source_model'
    if point.get('class_name') != model.get('class_name'):
        return 'source_model_identity_mismatch'
    try:
        if abs(float(point['source_diameter'])-float(model['source_diameter'])) > 1.e-6:
            return 'source_model_identity_mismatch'
    except (KeyError, TypeError, ValueError, OverflowError):
        return 'source_model_identity_mismatch'
    return ''


def _raster(record):
    """Crop the saved raster while retaining its measured centre offset."""
    if 'mask' in record and 'anchor' in record:
        mask = np.asarray(record['mask'], dtype=np.uint8)
        anchor = np.asarray(record['anchor'], dtype=float)
    else:
        mask = decode_marker_mask(record['marker_mask']).astype(np.uint8)
        anchor = np.array([mask.shape[1] // 2 + float(record.get('marker_offset_x', 0.)),
                           mask.shape[0] // 2 + float(record.get('marker_offset_y', 0.))])
    if mask.ndim != 2 or not mask.any() or max(mask.shape) > 1024 or not np.isfinite(anchor).all():
        raise ValueError('invalid_saved_marker_raster')
    yy, xx = np.nonzero(mask)
    return mask[yy.min():yy.max()+1, xx.min():xx.max()+1], anchor - [xx.min(), yy.min()]


def _diameter(model):
    return float(max(_raster(model)[0].shape))


def _point_key(point):
    return (str(point.get('swatch_id')), float(point['cx']), float(point['cy']))


def _candidate_id(point):
    return str(point.get('candidate_id') or point.get('point_id') or
               '{}@{:g},{:g}'.format(*_point_key(point)))


def _placed_pixels(point):
    mask, anchor = _raster(point)
    yy, xx = np.nonzero(mask)
    return xx + round(float(point['cx']) - anchor[0]), yy + round(float(point['cy']) - anchor[1])


def independent_point_evidence(point, active, models, ink, scope, tests):
    """Revalidate an ordinary marker using unmodified source ink.

    ``tests(point)`` is the dispatcher's fresh per-series measurement mapping.
    Its ordinary acceptance, score > .70, and rival margin >= .06 remain
    mandatory. Native saved point rasters and the pure legend model may have
    different measured anchors/rasterization. The point's actual frozen raster
    must ALSO pass the unchanged body/hole test, at its saved anchor. Signed
    raster-edge contrast is additional evidence: required boundary ink and
    adjacent paper must both be visible. Other active stamps are never
    subtracted to manufacture a positive marker boundary.
    """
    sid = point.get('swatch_id')
    model = models.get(sid)
    reason = _ordinary(point, model)
    result = dict(valid=False, reason=reason, policy=POLICY,
                  candidate_id=_candidate_id(point), score=0., shape_margin=0.)
    if reason:
        return result
    try:
        model_mask, model_anchor = _raster(model)
        mask, anchor = _raster(point)
        result['saved_geometry'] = dict(point_shape=list(mask.shape), point_anchor=anchor.tolist(),
            model_shape=list(model_mask.shape), model_anchor=model_anchor.tolist(),
            identical_raster=bool(np.array_equal(mask, model_mask)),
            identical_anchor=bool(np.allclose(anchor, model_anchor, atol=1.e-6, rtol=0)),
            policy='validate_both_frozen_rasters_at_saved_anchors_without_search')
        measurements = tests(point)
        own = measurements[sid]
        score = float(own['score'])
        rivals = [float(value['score']) for key, value in measurements.items() if key != sid]
        margin = score - max(rivals) if rivals else 1.
        result.update(score=score, shape_margin=margin, marker_measurement=dict(own))
        if not np.isfinite([score, margin, *rivals]).all():
            return dict(result, reason='nonfinite_marker_measurement')
        if not own.get('accepted'):
            return dict(result, reason='failed_marker_pixels')
        if own.get('partial_visibility', {}).get('valid') or own.get('overlap_recheck', {}).get('valid'):
            return dict(result, reason='context_dependent_marker_measurement')
        if score <= .70:
            return dict(result, reason='insufficient_marker_score')
        if margin < .06:
            return dict(result, reason='ambiguous_shape')
        # Import after the dispatcher is initialized: the dispatcher owns this
        # ordinary measurement contract. Reuse it, rather than copying or
        # weakening its body/hole thresholds for a differently rasterized point.
        from bw_series_correction_v46 import _measure
        saved_model = dict(model, mask=mask, anchor=anchor)
        saved_own = _measure(ink, scope, saved_model, point['cx'], point['cy'])
        result['saved_marker_measurement'] = saved_own
        if not saved_own.get('accepted'):
            return dict(result, reason='failed_saved_point_marker_pixels')
        if not np.isfinite(saved_own['score']) or saved_own['score'] <= .70:
            return dict(result, reason='insufficient_saved_point_marker_score')
        result['score'] = min(score, float(saved_own['score']))
        diameter = float(max(mask.shape))
        pad = 3
        left = round(float(point['cx']) - anchor[0]) - pad
        top = round(float(point['cy']) - anchor[1]) - pad
        expected = np.pad(mask, pad)
        height, width = expected.shape
        if left < 0 or top < 0 or left+width > ink.shape[1] or top+height > ink.shape[0]:
            return dict(result, reason='boundary_outside_image')
        observed = np.asarray(ink[top:top+height, left:left+width]) > 0
        valid = np.asarray(scope[top:top+height, left:left+width]) > 0
        inside = (expected > 0) & (cv2.erode(expected, _KERNEL) == 0)
        outside = (cv2.dilate(expected, _KERNEL) > 0) & (expected == 0)
        if not inside.any() or not outside.any() or not valid[inside | outside].all():
            return dict(result, reason='boundary_outside_scope')
        inside_support = float(observed[inside].mean())
        outside_paper = float((~observed[outside]).mean())
        signed = inside_support + outside_paper - 1.
        yy, xx = np.nonzero(inside)
        angles = np.arctan2(yy - anchor[1] - pad, xx - anchor[0] - pad)
        sectors = np.minimum(7, ((angles + np.pi) * (4. / np.pi)).astype(int))
        sector_support = [float(observed[yy[sectors == sector], xx[sectors == sector]].mean())
                          for sector in range(8) if np.sum(sectors == sector) >= 2]
        result['boundary'] = dict(inside_ink=inside_support, outside_paper=outside_paper,
            signed_contrast=signed, sector_support=sector_support,
            source='original_ink_without_other_marker_subtraction')
        if (inside_support < .80 or outside_paper < .70 or signed < .55 or
                len(sector_support) < 4 or min(sector_support) < .60):
            return dict(result, reason='insufficient_independent_boundary')
        px, py = _placed_pixels(point)
        required = set(zip(px.tolist(), py.tolist()))
        reused = set()
        nearest = None
        for other in active:
            if _point_key(other) == _point_key(point):
                continue
            qx, qy = _placed_pixels(other)
            reused.update(required.intersection(zip(qx.tolist(), qy.tolist())))
            if other.get('swatch_id') != sid:
                other_d = float(max(_raster(other)[0].shape))
                separation = math.hypot(float(other['cx'])-float(point['cx']),
                                        float(other['cy'])-float(point['cy']))
                ratio = separation / max(1., .6*(diameter+other_d))
                nearest = ratio if nearest is None else min(nearest, ratio)
        overlap_fraction = len(reused) / max(1, len(required))
        result['active_ink_overlap'] = dict(pixels=len(reused), fraction=overlap_fraction)
        # At most two pixels, AND at most 2%, can be shared by raster contact.
        if len(reused) > 2 or overlap_fraction > .02:
            return dict(result, reason='active_marker_ink_reuse')
        if nearest is not None and nearest < 1.:
            return dict(result, reason='near_active_other_body')
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return dict(result, reason='invalid_saved_marker_evidence')
    return dict(result, valid=True, reason='independent_marker_pixels_and_boundary')


def _connector_exclusions(active, sid, models, ink, scope, diameter):
    source = (np.asarray(ink) > 0).astype(np.uint8)
    valid = (np.asarray(scope) > 0).astype(np.uint8)
    # Match the existing source-reference policy for long bars/axis-like rails.
    yy, xx = np.nonzero(valid)
    span = int(xx.max()-xx.min()+1) if len(xx) else source.shape[1]
    vertical = cv2.morphologyEx(source, cv2.MORPH_OPEN,
                                np.ones((max(9, round(1.4*diameter)), 1), np.uint8))
    horizontal = cv2.morphologyEx(source, cv2.MORPH_OPEN,
                                  np.ones((1, max(20, round(.65*span))), np.uint8))
    nuisance = cv2.dilate(np.maximum(vertical, horizontal), _KERNEL)
    # Erode even a full-image ROI with an explicit paper border.
    safe = cv2.erode(valid, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    excluded = np.maximum(nuisance, 1-safe)
    other_groups = {}
    for point in active:
        model = models.get(point.get('swatch_id'))
        try:
            mask, anchor = _raster(point)
            radius = max(2, round(.75*max(mask.shape)))
            cv2.circle(excluded, (round(point['cx']), round(point['cy'])), radius, 1, -1)
            if point.get('swatch_id') != sid and not _ordinary(point, model):
                other_groups.setdefault(point['swatch_id'], []).append(point)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            # An unrenderable active marker cannot contribute positive support.
            continue
    corridors = 0
    for other_sid, points in sorted(other_groups.items()):
        points = sorted(points, key=lambda point: (_point_key(point), _candidate_id(point)))
        width = max(3, int(math.ceil(.4*_diameter(models[other_sid]))))
        for first, second in zip(points[:-1], points[1:]):
            cv2.line(excluded, (round(first['cx']), round(first['cy'])),
                     (round(second['cx']), round(second['cy'])), 1, width)
            corridors += 1
    return excluded, corridors


def independent_connector_evidence(first, second, active, models, ink, scope, *, _base=None):
    """Require a direct source chord with enough unclaimed line evidence.

    The existing chord thresholds are unchanged. Both original and nuisance-
    excluded source ink must pass ``chord_test``. Exclusions remove evidence;
    they never draw a connector or remove required missing marker pixels.
    """
    result = dict(valid=False, reason='', policy=POLICY)
    if first.get('swatch_id') != second.get('swatch_id'):
        return dict(result, reason='different_series_pair')
    first, second = sorted((first, second), key=lambda point: (_point_key(point), _candidate_id(point)))
    sid = first['swatch_id']
    try:
        diameter = _diameter(models[sid])
        if second['cx']-first['cx'] <= 2*diameter:
            return dict(result, reason='pair_x_separation_too_small')
        ax, ay = _placed_pixels(first)
        bx, by = _placed_pixels(second)
        if set(zip(ax.tolist(), ay.tolist())).intersection(zip(bx.tolist(), by.tolist())):
            return dict(result, reason='pair_marker_bodies_overlap')
        a, b = [first['cx'], first['cy']], [second['cx'], second['cy']]
        original = chord_test(a, b, (np.asarray(ink) > 0).astype(np.uint8), diameter, scope)
        result['original_chord'] = original
        if not original.get('eligible') or not original.get('connected'):
            return dict(result, reason='missing_original_direct_connector')
        excluded, corridors = (_connector_exclusions(active, sid, models, ink, scope, diameter)
                               if _base is None else _base)
        excluded = excluded.copy()
        for point in (first, second):
            cv2.circle(excluded, (round(point['cx']), round(point['cy'])),
                       max(2, round(.75*diameter)), 1, -1)
        unique_ink = ((np.asarray(ink) > 0) & (excluded == 0)).astype(np.uint8)
        unique = chord_test(a, b, unique_ink, diameter, scope)
        count = sum(unique.get('sample_support', []))
        result.update(unique_chord=unique, unique_support_samples=int(count),
                      minimum_unique_samples=max(12, int(math.ceil(1.2*diameter))),
                      excluded_other_series_corridors=corridors,
                      marker_neighborhood_radius=.75*diameter)
        if not unique.get('eligible') or not unique.get('connected'):
            return dict(result, reason='connector_explained_by_nuisance_or_active_series')
        if count < result['minimum_unique_samples']:
            return dict(result, reason='insufficient_unique_connector_ink')
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return dict(result, reason='invalid_pair_source_evidence')
    return dict(result, valid=True, reason='original_and_unique_direct_connector')


def propose_pairs(active, pool, models, ink, scope, tests, initial_zero_series):
    """Return bounded, deterministic atomic pair trials for still-empty series.

    Only saved pool records are considered. Callers gate ordinary single-edit
    availability, retained contextual rosters, and commit both points together.
    Diagnostics include rejected pairs and candidate-budget truncation. Inputs
    are treated as read-only; no active/suppressed state is changed.
    """
    trials = []
    for sid in sorted(set(initial_zero_series)):
        if any(point.get('swatch_id') == sid for point in active) or sid not in models:
            continue
        candidates = [point for point in pool if point.get('swatch_id') == sid]
        def ordering(point):
            confidence = float(point.get('confidence', 0.))
            # A deferred contract at the same location cannot hide an ordinary
            # independently testable candidate, even with higher saved score.
            return (bool(_ordinary(point, models[sid])),
                    -confidence if np.isfinite(confidence) else 0., _point_key(point), _candidate_id(point))
        candidates = sorted(candidates, key=ordering)
        # Multiple saved contracts at one physical location are not two seeds.
        unique = {}
        for point in candidates:
            unique.setdefault(_point_key(point), point)
        candidates = list(unique.values())
        budget = dict(pool_candidates=len(candidates), maximum_candidates=MAX_CANDIDATES,
                      evaluated_candidates=min(MAX_CANDIDATES, len(candidates)),
                      truncated_candidates=max(0, len(candidates)-MAX_CANDIDATES), maximum_pairs=MAX_PAIRS)
        candidates = candidates[:MAX_CANDIDATES]
        if len(candidates) < 2:
            trials.append(dict(action='ACTIVATE_PAIR_INIT', swatch_id=sid, points=[],
                candidate_ids=[_candidate_id(point) for point in candidates], admissible=False,
                reason='insufficient_saved_pair_candidates', gain=0., evidence=dict(policy=POLICY, budget=budget)))
            continue
        checks = {id(point): independent_point_evidence(point, active, models, ink, scope, tests)
                  for point in candidates}
        base = None
        for first, second in combinations(candidates, 2):
            first, second = sorted((first, second), key=lambda point: (_point_key(point), _candidate_id(point)))
            point_checks = [checks[id(first)], checks[id(second)]]
            failed = next((check['reason'] for check in point_checks if not check['valid']), '')
            connector = None
            if not failed:
                if base is None:
                    base = _connector_exclusions(active, sid, models, ink, scope, _diameter(models[sid]))
                connector = independent_connector_evidence(first, second, active, models, ink, scope, _base=base)
                if not connector['valid']:
                    failed = connector['reason']
            trials.append(dict(action='ACTIVATE_PAIR_INIT', swatch_id=sid, points=[first, second],
                candidate_ids=[_candidate_id(first), _candidate_id(second)], admissible=not failed,
                reason=failed or 'two_independent_markers_and_unique_direct_connector',
                gain=float(np.mean([check['score']-.70 for check in point_checks])),
                evidence=dict(policy=POLICY, point_checks=point_checks, connector=connector, budget=budget)))
    return trials

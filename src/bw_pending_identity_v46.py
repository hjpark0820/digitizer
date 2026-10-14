"""Fresh, source-bound validation of typed pending BW identities.

A pending record is a hypothesis, never permission to add a marker.  The fixed
saved centre and exact measured marker mask compete against every saved legend
identity and a line-plus-active-context null on one shared source domain.  No
curve fit, ground truth, source recolouring or stale ``passed`` flag enters this
check.  The existing untyped scene validator supplies all numerical gates.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

import cv2
import numpy as np

from bw_suppressed_v46 import decode_marker_mask
import bw_untyped_resolver_v46 as source_scene


VERSION = 'bw-pending-identity-v1'
GEOMETRY_KEYS = ('swatch_id', 'class_name', 'cx', 'cy', 'marker_mask',
                 'marker_offset_x', 'marker_offset_y', 'marker_scale',
                 'effective_diameter', 'source_diameter')
MODEL_KEYS = ('swatch_id', 'class_name', 'source_diameter', 'effective_diameter',
              'marker_scale', 'marker_mask', 'marker_offset_x', 'marker_offset_y')


def image_hash(image):
    """Full, unchanged BGR pixel binding; identical to the U resolver's hash."""
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def geometry_hash(point):
    """Canonical geometry binding, excluding mutable confidence/state/labels."""
    bound = {key: deepcopy(point.get(key)) for key in GEOMETRY_KEYS}
    return hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def models_hash(models):
    """Bind all encoded legend geometries, not decoded caches or provenance."""
    bound = {sid: {key: deepcopy(record.get(key)) for key in MODEL_KEYS}
             for sid, record in sorted(models.items())}
    return hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _model(record, sid):
    if not isinstance(record, dict) or record.get('swatch_id') != sid:
        raise ValueError('source_model_identity_mismatch')
    mask = decode_marker_mask(record['marker_mask']).astype(np.uint8)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('empty_source_marker_mask')
    offsets = [float(record.get('marker_offset_x', 0)), float(record.get('marker_offset_y', 0))]
    if not np.isfinite(offsets).all():
        raise ValueError('nonfinite_source_anchor')
    anchor = [mask.shape[1]//2 + offsets[0] - xx.min(),
              mask.shape[0]//2 + offsets[1] - yy.min()]
    result = deepcopy(record)
    result.pop('pending_identity', None)
    result.update(mask=mask[yy.min():yy.max()+1, xx.min():xx.max()+1], anchor=anchor)
    return result


def _same_queried_point(query, active):
    """Exclude only this point, not another series or another overlapping body."""
    key = query.get('point_id', query.get('candidate_id'))
    other = active.get('point_id', active.get('candidate_id'))
    if key is None or key != other:
        return False
    return geometry_hash(query) == geometry_hash(active)


def validate_pending(image_bgr, point, active_points, models, plot, legend=None):
    """Validate one fixed pending candidate without changing any input.

    Required ``point.pending_identity`` keys are version, reason,
    source_image_sha256, source_image_shape, source_models_sha256, original_comparison, and
    original_failed_checks.  ``geometry_sha256`` is optional for older exported
    hypotheses; if present it must match the complete exact point geometry.
    Models use the saved correction model format (encoded ``marker_mask``).
    An input model containing decoded mask/anchor fields is also accepted, but
    those fields never override its encoded, provenance-preserved source mask.
    """
    result = dict(valid=False, version=VERSION, reason='invalid_pending_metadata',
                  score=0., identity_margin=0., source_pixels_modified=False,
                  active_points_modified=False, reference_points_read=False)
    try:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return dict(result, reason='invalid_source_image')
        pending = point.get('pending_identity')
        if not isinstance(pending, dict) or pending.get('version') != VERSION:
            return result
        if (not isinstance(pending.get('reason'), str) or not pending['reason'] or
                not isinstance(pending.get('original_comparison'), dict) or
                not isinstance(pending.get('original_failed_checks'), list) or
                not pending['original_failed_checks'] or
                not all(isinstance(value, str) and value for value in pending['original_failed_checks'])):
            return result
        # Even diagnostic metadata must not smuggle NaN into a saved decision.
        json.dumps(pending, allow_nan=False)
        if (pending.get('source_image_shape') != list(image.shape) or
                pending.get('source_image_sha256') != image_hash(image)):
            return dict(result, reason='source_image_binding_changed')
        geometry = geometry_hash(point)
        if pending.get('geometry_sha256') not in (None, geometry):
            return dict(result, reason='pending_geometry_binding_changed')
        own_sid = point.get('swatch_id')
        if not isinstance(own_sid, str) or not own_sid:
            return dict(result, reason='missing_pending_identity')
        center = [float(point['cx']), float(point['cy'])]
        if not np.isfinite(center).all():
            return dict(result, reason='nonfinite_pending_center')
        if (len(plot) != 4 or not np.isfinite(plot).all() or
                not (0 <= plot[0] < plot[2] <= image.shape[1] and
                     0 <= plot[1] < plot[3] <= image.shape[0])):
            return dict(result, reason='invalid_plot_box')
        if legend is not None and (len(legend) != 4 or not np.isfinite(legend).all()):
            return dict(result, reason='invalid_legend_box')
        if not source_scene._inside(center, plot, legend):
            return dict(result, reason='pending_center_outside_plot')
        chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2).astype(np.int16)
        if np.count_nonzero(chroma > 24) > .01*chroma.size:
            return dict(result, reason='color_source_not_supported')
        if not isinstance(models, dict) or not models or own_sid not in models:
            return dict(result, reason='saved_legend_models_missing')
        if pending.get('source_models_sha256') != models_hash(models):
            return dict(result, reason='source_models_binding_changed')
        original_scores = pending['original_comparison'].get('scores')
        if (not isinstance(original_scores, dict) or
                set(original_scores) != set(models)):
            return dict(result, reason='saved_identity_competitors_incomplete')
        saved_own = models[own_sid]
        if point.get('class_name') != saved_own.get('class_name'):
            return dict(result, reason='pending_class_source_mismatch')
        if 'source_diameter' in point or 'source_diameter' in saved_own:
            own_diameter = float(point['source_diameter'])
            saved_diameter = float(saved_own['source_diameter'])
            if (not math.isfinite(own_diameter) or own_diameter <= 0 or
                    not math.isfinite(saved_diameter) or saved_diameter <= 0 or
                    not math.isclose(own_diameter, saved_diameter, rel_tol=1.e-6, abs_tol=1.e-6)):
                return dict(result, reason='pending_source_diameter_mismatch')
        # Every saved identity competes.  Only the queried identity adopts its
        # exact exported, measured scale/anchor instead of a generic shared one.
        compared = {sid: _model(record, sid) for sid, record in models.items()}
        compared[own_sid] = _model(point, own_sid)
        if point.get('class_name') in (None, '', 'unknown_marker', 'suppressed'):
            return dict(result, reason='source_identity_untyped')
        active = []
        excluded = []
        for other in active_points:
            if _same_queried_point(point, other):
                excluded.append(str(other.get('point_id', other.get('candidate_id'))))
            else:
                active.append(deepcopy(other))
        radius = max(compared[own_sid]['mask'].shape)/2 + 1
        a, b = max(int(math.floor(center[0]-radius)), int(math.ceil(plot[0]))), max(int(math.floor(center[1]-radius)), int(math.ceil(plot[1])))
        c, d = min(int(math.ceil(center[0]+radius+1)), int(math.floor(plot[2]))), min(int(math.ceil(center[1]+radius+1)), int(math.floor(plot[3])))
        region = dict(region_id='pending:' + str(point.get('point_id', own_sid)),
                      bbox=[a, b, c, d], cx=center[0], cy=center[1],
                      candidate_centres=[], source_patch_sha256=image_hash(image[b:d, a:c]))
        cfg = replace(source_scene.Config(), maximum_shift=0, maximum_seed_centres=1,
                      maximum_centres=1, allow_pairs=False)
        proposal, review = source_scene._resolve_region(image, region, compared, active, plot, legend, cfg)
        evidence = review.get('evidence', {})
        result.update(reason=review['reason'], review=review, evidence=evidence,
                      original_reason=pending['reason'],
                      original_failed_checks=deepcopy(pending['original_failed_checks']),
                      source_image_sha256=pending['source_image_sha256'],
                      geometry_sha256=geometry, excluded_self_point_ids=excluded)
        if evidence:
            value = float(evidence.get('scene_loss', math.inf))
            margin = float(evidence.get('identity_margin', 0))
            if not math.isfinite(value) or not math.isfinite(margin):
                return dict(result, reason='nonfinite_source_scene')
            result.update(score=float(np.clip(1-value, 0, 1)), identity_margin=margin)
        if proposal is None:
            return result
        winners = proposal['points']
        if (len(winners) != 1 or winners[0]['swatch_id'] != own_sid or
                [winners[0]['cx'], winners[0]['cy']] != center):
            return dict(result, reason='another_source_identity_wins')
        return dict(result, valid=True, reason='pending_identity_source_revalidated')
    except (KeyError, TypeError, ValueError, OverflowError, cv2.error) as exc:
        return dict(result, reason='invalid_pending_source_or_model', detail=str(exc))

"""Opt-in evidence-preserving deferral; never a marker acceptance shortcut.

Native loss-only failures and grouped-raster identity ties may become typed
pending hypotheses. Existing detections and baseline S/U admission are kept.
Every new hypothesis binds to the current image and an explicit saved marker
geometry. The original raster score does NOT transfer to that geometry.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from bw_raster_identity_v46 import Config as RasterConfig
from bw_suppressed_v46 import SuppressedPolicy, decode_marker_mask, encode_marker_mask
from bw_unknown_candidates_v46 import _evidence


VERSION = 'bw-pending-identity-v1'
CLASSES = ['filled_circle', 'open_circle', 'filled_square', 'open_square',
           'open_triangle', 'open_inv_triangle', 'filled_triangle',
           'filled_inv_triangle', 'open_rhombus', 'filled_rhombus', 'x_marker',
           'plus_marker', 'unknown_marker']


def _number(value):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _sid(row):
    return str(row.get('swatch_id') or row.get('template') or '')


def _centre(row):
    return (_number(row.get('cx', row.get('aligned_x', row.get('x')))),
            _number(row.get('cy', row.get('aligned_y', row.get('y')))))


def _inside(x, y, box):
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _plain(value):
    """Keep finite, serializable diagnostics without retaining ndarray aliases."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported pending diagnostic type: {type(value).__name__}')


def raster_failed_checks(score, cfg=None):
    """Use the production raster gates; no looser acceptance band is invented."""
    cfg = cfg or RasterConfig()
    gates = [('maximum_loss', 'loss', cfg.maximum_loss, False),
             ('minimum_line_improvement', 'line_improvement', cfg.minimum_improvement, True),
             ('minimum_independent_fraction', 'independent_fraction', cfg.minimum_independent_fraction, True),
             ('minimum_independent_visible', 'independent_visible', cfg.minimum_visible, True)]
    failures = []
    for name, key, limit, lower in gates:
        value = _number(score.get(key))
        if value is None or (value < limit if lower else value > limit):
            failures.append(name)
    return failures


def _conflict(record):
    if record.get('excluded_by_confidence_floor') or record.get('exclusion_reason') == 'confidence_floor':
        return True
    # Keep explicit source-body contradictions distinct from context residual.
    for name in ('geometry_first', 'open_interior', 'source_open_tone', 'selected_body_fill'):
        info = record.get(name) or {}
        if info.get('decision') in ('conflict', 'incompatible'):
            return True
    return False


def _model_geometry(model):
    """Encode the actual fixed model and preserve its anchor after cropping."""
    # The encoded source is authoritative, even when a caller also carries
    # decoded convenience arrays. This is the same source used by models_hash
    # and fresh Step 5 validation; no stale decoded array can replace it.
    mask = decode_marker_mask(model['marker_mask'])
    ox, oy = float(model.get('marker_offset_x', 0.)), float(model.get('marker_offset_y', 0.))
    if mask.ndim != 2 or not mask.any() or max(mask.shape) > 1024:
        raise ValueError('Invalid saved marker geometry')
    yy, xx = np.nonzero(mask)
    ax, ay = mask.shape[1] // 2 + ox - xx.min(), mask.shape[0] // 2 + oy - yy.min()
    mask = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    ox, oy = float(ax) - mask.shape[1] // 2, float(ay) - mask.shape[0] // 2
    diameter = _number(model.get('effective_diameter'))
    if diameter is None or not 2 <= diameter <= 128:
        raise ValueError('Missing or invalid fixed model diameter')
    source_diameter = _number(model.get('source_diameter'))
    marker_scale = _number(model.get('marker_scale'))
    if source_diameter is None or source_diameter <= 0 or marker_scale is None or marker_scale <= 0:
        raise ValueError('Missing or invalid source diameter or fixed marker scale')
    if not math.isfinite(ox) or not math.isfinite(oy):
        raise ValueError('Invalid model anchor')
    return dict(marker_mask=encode_marker_mask(mask), marker_offset_x=ox, marker_offset_y=oy,
                effective_diameter=diameter,
                source_diameter=source_diameter, marker_scale=marker_scale)


def build_pending(image, plot, legend_boxes, active, pool, records, layered_report,
                  models, untyped_regions=(), *, max_bodies=32, max_alternatives=3):
    """Return copied S/U pools plus bounded new, mandatory-revalidation S rows.

    ``models`` accepts encoded correction_marker_evidence or the decoded models
    returned by bw_series_correction_v46._models. Neither active points nor any
    input record is modified. Existing S remains baseline-eligible; added
    provenance alone cannot change its admission. Only AP rows carry the
    ``pending_identity`` contract consumed by Step 5's fresh pixel validator.
    """
    from bw_pending_identity_v46 import geometry_hash, image_hash, models_hash

    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
        raise ValueError('Expected nonempty uint8 BGR source image')
    if not isinstance(max_bodies, int) or not 1 <= max_bodies <= 128:
        raise ValueError('max_bodies must be in [1, 128]')
    if not isinstance(max_alternatives, int) or not 1 <= max_alternatives <= 3:
        raise ValueError('max_alternatives must be in [1, 3]')
    plot = tuple(float(v) for v in plot)
    legends = [tuple(float(v) for v in box) for box in legend_boxes]
    if any(len(b) != 4 or not all(math.isfinite(v) for v in b) or b[0] >= b[2] or b[1] >= b[3]
           for b in [plot, *legends]):
        raise ValueError('Invalid plot or legend box')
    result, regions = deepcopy(list(pool)), deepcopy(list(untyped_regions))
    source_hash = image_hash(image)
    source_models_hash = models_hash(models)
    cfg, policy = RasterConfig(), SuppressedPolicy()
    rejected = Counter()
    proposals = []
    record_by_index = {row.get('candidate_index', i): row for i, row in enumerate(records)}
    enriched_existing = 0
    for item in result:
        row = record_by_index.get(item.get('candidate_index'))
        if row and row.get('exclusion_reason') == 'identity_ambiguous' and row.get('raster_identity'):
            item['ambiguity_provenance'] = dict(version=VERSION,
                reason='baseline_identity_ambiguous', baseline_admission_unchanged=True,
                original_comparison=_plain(row['raster_identity']),
                original_deferred_reason=row.get('raster_deferred_reason'))
            enriched_existing += 1

    for index, row in enumerate(records):
        if row.get('decision') != 'verified' or row.get('exclusion_reason') != 'raster_marker_not_supported':
            continue
        if row.get('selected') or _conflict(row):
            rejected['native_selected_or_conflict'] += 1
            continue
        key = _sid(row)
        comparison = row.get('raster_identity') or {}
        score = comparison.get('scores', {}).get(key, {})
        if raster_failed_checks(score, cfg) != ['maximum_loss']:
            rejected['native_not_loss_only'] += 1
            continue
        direct = _number((row.get('ink_evidence') or {}).get('direct_required_recall'))
        boundary, contour = _number(row.get('boundary_recall')), _number(row.get('contour_support'))
        if (direct is None or direct < .80 or boundary is None or boundary < policy.minimum_boundary
                or contour is None or contour < policy.minimum_contour):
            rejected['native_missing_direct_outline'] += 1
            continue
        x, y = _centre(row)
        proposals.append(dict(sid=key, x=x, y=y, comparison=comparison, score=score,
            reason='native_raster_loss_only', failures=['maximum_loss'],
            original_index=row.get('candidate_index', index), source='native_verified_window',
            class_name=row.get('class_name') or row.get('shape_hint'),
            existence=dict(passed=True, source='native_verified_direct_outline',
                direct_required_recall=direct, boundary_recall=boundary, contour_support=contour),
            original_geometry=dict(window_scale=row.get('window_scale'),
                effective_diameter=row.get('effective_diameter'),
                footprint_polygon=row.get('footprint_polygon'))))

    for entry in (layered_report or {}).get('rejected', []):
        if entry.get('reason') != 'unresolved_body':
            continue
        for comparison in entry.get('comparisons', []):
            if comparison.get('reason') != 'ambiguous_raster_identity' or comparison.get('winner') is not None:
                continue
            center = comparison.get('center', ())
            if len(center) != 2:
                rejected['grouped_invalid_center'] += 1
                continue
            ranked = sorted(comparison.get('scores', {}).items(), key=lambda q: (
                _number(q[1].get('loss')) if _number(q[1].get('loss')) is not None else math.inf, q[0]))
            passed = [(sid, score) for sid, score in ranked if not raster_failed_checks(score, cfg)]
            for key, score in passed[:max_alternatives]:
                proposals.append(dict(sid=key, x=_number(center[0]), y=_number(center[1]),
                    comparison=comparison, score=score, reason='grouped_raster_identity_tie',
                    failures=['identity_margin'], source='grouped_raster_recovery',
                    original_body_id=entry.get('body_id'), original_index=None,
                    original_geometry=dict(raster_parameters=score.get('parameters'))))

    model_cache, bodies = {}, []
    new_points = []
    region_enrichments = 0
    for proposal in sorted(proposals, key=lambda q: (float(q['score']['loss']),
                            str(q['sid']), q['x'] if q['x'] is not None else math.inf,
                            q['y'] if q['y'] is not None else math.inf)):
        key, x, y = proposal['sid'], proposal['x'], proposal['y']
        if x is None or y is None or not _inside(x, y, plot) or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            rejected['outside_source_plot'] += 1
            continue
        if any(_inside(x, y, box) for box in legends):
            rejected['inside_legend'] += 1
            continue
        model = models.get(key)
        if not model or model.get('swatch_id', key) != key:
            rejected['missing_fixed_model'] += 1
            continue
        shape = str(model.get('class_name') or '')
        if shape not in CLASSES or shape == 'unknown_marker':
            rejected['unresolved_fixed_model_identity'] += 1
            continue
        if proposal.get('class_name') and proposal['class_name'] != shape:
            rejected['original_fixed_model_shape_mismatch'] += 1
            continue
        try:
            if key not in model_cache:
                model_cache[key] = _model_geometry(model)
            geometry = model_cache[key]
        except (ValueError, KeyError, TypeError):
            rejected['invalid_fixed_model_geometry'] += 1
            continue
        diameter = geometry['effective_diameter']
        active_radius = max(2., policy.same_active_diameter_fraction * diameter)
        if any(_sid(p) == key and all(v is not None for v in _centre(p)) and
               math.hypot(_centre(p)[0] - x, _centre(p)[1] - y) <= active_radius for p in active):
            rejected['same_swatch_active_duplicate'] += 1
            continue
        existence = proposal.get('existence')
        if existence is None:
            # Body existence is measured before carrying an identity tie. A
            # high template recall alone cannot certify a compact marker body.
            existence = _evidence(image, x, y, diameter, plot, legends)
            if not existence['passed']:
                rejected['grouped_' + existence['reason']] += 1
                continue
        provenance = dict(version=VERSION, reason=proposal['reason'],
            source=proposal['source'], original_candidate_index=proposal.get('original_index'),
            original_body_id=proposal.get('original_body_id'),
            original_comparison=_plain(proposal['comparison']),
            original_failed_checks=proposal['failures'],
            original_geometry=_plain(proposal['original_geometry']),
            existence_evidence=_plain(existence), source_score_transferred=False)
        merged = None
        for item in result:
            px, py = _centre(item)
            if _sid(item) == key and px is not None and py is not None and math.hypot(px - x, py - y) <= max(2., policy.merge_diameter_fraction * diameter):
                merged = item
                break
        if merged is not None:
            merged.setdefault('additional_ambiguity_provenance', []).append(provenance)
            rejected['merged_existing_same_swatch'] += 1
            continue
        matches = [b for b in bodies if math.hypot(b['x'] - x, b['y'] - y) <= max(1.5, .35 * min(b['diameter'], diameter))]
        if matches:
            body = min(matches, key=lambda b: math.hypot(b['x'] - x, b['y'] - y))
            if len(body['swatch_ids']) >= max_alternatives and key not in body['swatch_ids']:
                rejected['body_alternative_budget'] += 1
                continue
        elif len(bodies) < max_bodies:
            token = f'{source_hash}:{x:.6f}:{y:.6f}'
            body = dict(body_id='AB' + hashlib.sha256(token.encode()).hexdigest()[:14],
                        x=x, y=y, diameter=diameter, swatch_ids=[])
            bodies.append(body)
        else:
            rejected['body_budget'] += 1
            continue
        if key in body['swatch_ids']:
            rejected['same_body_same_swatch'] += 1
            continue
        body['swatch_ids'].append(key)
        token = f'{source_hash}:{key}:{x:.6f}:{y:.6f}:{body["body_id"]}'
        candidate_id = 'AP' + hashlib.sha256(token.encode()).hexdigest()[:16]
        point = dict(candidate_id=candidate_id, swatch_id=key, template=key,
            class_name=shape, shape_hint=shape, class_idx=CLASSES.index(shape),
            shape_idx=CLASSES.index(shape), cx=x, cy=y, confidence=float(proposal['score']['independent_visible']),
            confidence_kind='proposal_visible_ink_not_existence_probability',
            source=VERSION, status='suppressed_hypothesis', priority=1,
            admission_reason=proposal['reason'], occlusion_claim=False, **deepcopy(geometry))
        point['pending_identity'] = dict(provenance, body_id=body['body_id'],
            source_image_sha256=source_hash, source_image_shape=list(image.shape),
            source_models_sha256=source_models_hash,
            geometry_sha256=geometry_hash(point),
            validation_geometry=dict(source='fixed_saved_correction_marker_model',
                model_provenance=model.get('provenance'),
                marker_scale=geometry['marker_scale'], effective_diameter=diameter,
                original_raster_scale_not_transferred=True),
            revalidation_required=True, allow_activation_without_revalidation=False)
        result.append(point)
        new_points.append(point)
        # U is a source region, not a second marker. Enrich an existing region
        # if nearby; never add an overlapping region just to carry a new label.
        near_regions = [r for r in regions if all(v is not None for v in _centre(r)) and
            math.hypot(_centre(r)[0] - x, _centre(r)[1] - y) <= .65 * diameter]
        if near_regions:
            region = min(near_regions, key=lambda r: math.hypot(_centre(r)[0] - x, _centre(r)[1] - y))
            region.setdefault('pending_identity_candidates', []).append(dict(
                candidate_id=candidate_id, swatch_id=key, cx=x, cy=y,
                body_id=body['body_id'], reason=proposal['reason'], source_score_transferred=False))
            region_enrichments += 1
    report = dict(version=VERSION, status='completed', input_active=len(active),
        input_suppressed=len(pool), output_suppressed=len(result),
        new_pending=len(new_points), new_by_reason=dict(Counter(p['admission_reason'] for p in new_points)),
        existing_s_enriched=enriched_existing, input_regions=len(untyped_regions),
        output_regions=len(regions), region_provenance_enrichments=region_enrichments,
        eligible_proposals=len(proposals), body_groups=len(bodies),
        maximum_bodies=max_bodies, maximum_alternatives=max_alternatives,
        rejections=dict(sorted(rejected.items())), active_points_modified=False,
        source_pixels_modified=False, reference_points_read=False,
        default_acceptance_changed=False, new_candidates_require_fresh_validation=True,
        geometry_policy='fixed_saved_model_not_original_raster_score_geometry')
    # Fail closed on accidental NaN/ndarray export rather than saving a weak
    # contract which a later correction session might misinterpret.
    json.dumps(new_points, allow_nan=False)
    return dict(suppressed=result, regions=regions, report=report)

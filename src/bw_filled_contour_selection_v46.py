"""Opt-in angular source-identity ranking of already verified B&W records.

This adapter is deliberately not a detector or a rescue. Full-window, raster,
line-null, fill, and required-ink gates run first. Only existing records at the
same exact measured center can exchange their existing selection rank values.
No source identity, physical footprint, mask, center, or confidence is invented.
"""
from __future__ import annotations

from dataclasses import asdict
import time

import cv2
import numpy as np

import bw_filled_contour_retry_v46 as polar
from bw_marker_selection import duplicate_reason


VERSION = 'filled-angular-verified-selection-v2-selective-retry'
FILLED_CLASSES = frozenset(('filled_circle', 'filled_square', 'filled_triangle',
                            'filled_inv_triangle'))


def _identity(record):
    return record.get('swatch_id') or record['template']


def finalize_supported_alternatives(records, report):
    """Finalize temporary physical alternatives after optional identity evidence.

    Abstention/errors retain exactly the old raster roster. Only a comparator
    winner with its own verified window and raster support can enter that
    roster; every losing alternative stays in the original typed records.
    """
    accepted = {}
    for detail in report.get('centers', []):
        comparison = detail.get('comparison', {})
        if (detail.get('action') in ('permuted_existing_verified_ranks', 'verified_order_confirmed')
                and comparison.get('accepted')):
            accepted[tuple(map(float, detail['center']))] = comparison.get('winner')
    # Recheck the destination even if a caller supplies a stale report. A
    # missing/unsupported winner must not remove an otherwise valid roster.
    supported = {(float(row['aligned_x']), float(row['aligned_y']), _identity(row))
                 for row in records if row.get('decision') == 'verified'
                 and row.get('raster_identity', {}).get('scores', {}).get(_identity(row), {}).get('passed')}
    accepted = {center: winner for center, winner in accepted.items()
                if (*center, winner) in supported}
    kept, promoted = [], []
    for row in records:
        center = (float(row['aligned_x']), float(row['aligned_y']))
        winner = accepted.get(center)
        legacy = row.get('pre_contour_raster_retained', True)
        if winner is None:
            if legacy:kept.append(row)
            continue
        own = row.get('raster_identity', {}).get('scores', {}).get(_identity(row), {})
        if _identity(row) == winner and row.get('decision') == 'verified' and own.get('passed'):
            if row.get('exclusion_reason') == 'identity_ambiguous':
                row['pre_contour_exclusion_reason'] = row.pop('exclusion_reason')
                row['pre_contour_deferred_reason'] = row.pop('raster_deferred_reason', None)
            row['identity_resolved_by'] = VERSION
            kept.append(row)
            if not legacy:promoted.append(row.get('candidate_index'))
        else:
            row['exclusion_reason'] = 'identity_ambiguous'
            row['raster_deferred_reason'] = 'another_independently_supported_contour_identity_wins'
    report['finalization'] = dict(temporary_input=len(records), retained=len(kept),
        identity_promoted_candidate_indices=promoted,
        policy='independent_support_first_identity_last_abstention_preserves_legacy')
    return kept


def rerank_verified(image, templates, records, *, config=None, maximum_centers=128):
    """Return the same record objects, with bounded within-center rank changes.

    Use pre-composition templates for source centers/extents, and sample the
    original image including connectors. Unknown sources may participate as raw
    sources if the polar comparator confirms interior ink; their label stays
    unknown. Unsupported/open identities at a center make that center ineligible.
    All source rivals participate; a winner without an already eligible record
    is diagnostic only. Missing physical/raster evidence is never resurrected.
    """
    started = time.perf_counter()
    cfg = config or polar.Config()
    output = list(records)
    pending = []
    # Source-only preparation is lazy and shared by ambiguous centers in this
    # run. Accepted baseline identities never enter the pair-weighted retry.
    retry_cache = {}
    report = dict(version=VERSION, status='not_applicable', enabled=True,
                  changed_centers=0, evaluated_centers=0, seconds=0.,
                  selective_retry_attempted_centers=0,
                  selective_retry_accepted_centers=0,
                  centers=[], sources=[], config=asdict(cfg),
                  runtime_policy='already_verified_records_only_no_gate_bypass',
                  default_enabled=False, source_pixels_modified=False,
                  new_candidates_generated=False, reference_points_read=False)

    def finish(status):
        report.update(status=status, seconds=time.perf_counter()-started)
        return output, report

    if image.ndim != 3 or image.shape[2] != 3:
        return finish('non_bgr_route_unchanged')
    chroma = image.max(axis=2).astype(float)-image.min(axis=2)
    if np.count_nonzero(chroma > 24) > .01*chroma.size:
        return finish('colour_route_unchanged')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = [t for t in templates if t.name in FILLED_CLASSES or t.name == 'unknown_marker']
    if len(candidates) < 2:
        return finish('fewer_than_two_source_rivals')
    # Do not silently discard a rival to fit a computational budget.
    if len(candidates) > cfg.maximum_sources:
        return finish('source_budget_exceeded_unchanged')
    if len({t.key for t in candidates}) != len(candidates):
        return finish('duplicate_source_ids_unchanged')
    sources = []
    for template in candidates:
        source = polar.build_source(template.key, gray, template.marker_center,
            float(template.diameter)/2., metadata=dict(class_name=template.name,
                marker_kind=template.marker_kind, swatch_box=list(template.swatch_box)), config=cfg)
        sources.append(source)
        report['sources'].append(dict(sid=template.key, class_name=template.name,
            center=list(template.marker_center), input_diameter=float(template.diameter),
            radius_px=source['reference_radius_px']))
    source_ids = {s['sid'] for s in sources}
    groups = {}
    for record in records:
        center = (float(record['aligned_x']), float(record['aligned_y']))
        groups.setdefault(center, []).append(record)
    for center, rows in groups.items():
        if len({_identity(r) for r in rows}) < 2:
            continue
        if any(_identity(r) not in source_ids or r.get('decision') != 'verified'
               for r in rows):
            report['centers'].append(dict(center=list(center), action='unsupported_or_unverified_rival_unchanged'))
            continue
        # A pre-composition unknown can later become an open glyph. Respect the
        # final record's fill metadata rather than its old source eligibility.
        if any(r.get('class_name') not in FILLED_CLASSES | {'unknown_marker'} for r in rows):
            report['centers'].append(dict(center=list(center), action='final_class_not_filled_or_unknown_unchanged'))
            continue
        if any(not r.get('raster_identity', {}).get('scores', {}).get(_identity(r), {}).get('passed')
               for r in rows):
            report['centers'].append(dict(center=list(center), action='independent_raster_support_required_unchanged'))
            continue
        if report['evaluated_centers'] >= maximum_centers:
            report['center_budget_exhausted'] = True
            break
        diameters = np.asarray([r.get('effective_diameter', np.nan) for r in rows], float)
        if not np.isfinite(diameters).all() or min(diameters) < 4 or max(diameters)/min(diameters) > 1.35:
            report['centers'].append(dict(center=list(center), action='inconsistent_body_extent_unchanged'))
            continue
        # A new identity can change same-series suppression radii even without
        # a rank bonus. Preserve the original neighboring-body relationships.
        row_ids = {id(r) for r in rows}
        neighbor_conflict = False
        for other in records:
            if id(other) in row_ids:
                continue
            relations = {(bool(duplicate_reason(row, other)), bool(duplicate_reason(other, row)))
                         for row in rows}
            interacting_tie = (any(a or b for a, b in relations) and
                               any(row['selection_rank'] == other['selection_rank'] for row in rows))
            if len(relations) > 1 or interacting_tie:
                neighbor_conflict = True
                break
        if neighbor_conflict:
            report['centers'].append(dict(center=list(center), action='neighbor_suppression_relationship_differs_unchanged'))
            continue
        # One shared initial radius; never optimize scale separately per class.
        radius_record = polar.estimate_radius(gray, center, float(np.median(diameters))/2., config=cfg)
        radius = radius_record['radius_px']
        retry_audit = {}
        comparison = polar.compare_sources(gray, center, radius, sources, config=cfg,
                                           cache=retry_cache, audit=retry_audit)
        report['evaluated_centers'] += 1
        report['selective_retry_attempted_centers'] += int(bool(retry_audit.get('attempted')))
        report['selective_retry_accepted_centers'] += int(bool(retry_audit.get('applied')))
        detail = dict(center=list(center), radius=radius_record,
                      comparison=comparison, selective_retry=retry_audit,
                      action='uncertain_unchanged')
        report['centers'].append(detail)
        for row in rows:
            pending.append((row, {'filled_contour_identity': comparison}))
        if not comparison.get('accepted'):
            continue
        winner = comparison['winner']
        if winner not in {_identity(r) for r in rows}:
            detail['action'] = 'winner_has_no_already_verified_record_unchanged'
            continue
        old = sorted(rows, key=lambda r: r['selection_rank'], reverse=True)
        new = sorted(rows, key=lambda r: (_identity(r) != winner, -r['selection_rank']))
        values = sorted((float(r['selection_rank']) for r in rows), reverse=True)
        # Reuse the old rank multiset. No rank bonus may reorder unrelated bodies.
        for row, value in zip(new, values):
            pending.append((row, dict(pre_contour_selection_rank=float(row['selection_rank']),
                                      selection_rank=value)))
        changed = any(a is not b for a, b in zip(old, new))
        if changed:
            report['changed_centers'] += 1
            # Stable sorting also resolves equal ranks without inventing scores.
            indices = [i for i, row in enumerate(output) if any(row is r for r in rows)]
            for index, row in zip(indices, new):
                output[index] = row
        detail['action'] = 'permuted_existing_verified_ranks' if changed else 'verified_order_confirmed'
        detail['before_order'] = [_identity(r) for r in old]
        detail['after_order'] = [_identity(r) for r in new]
    # Commit only after every comparison completed. An optional-stage failure
    # leaves all native records untouched, including their original ranking.
    for row, values in pending:
        row.update(values)
    return finish('completed')

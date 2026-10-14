"""A single native-BW window decision from shared legend/shape evidence.

The native window supplies bounded, own-type localization and physical ink
measurements, not an accepted list. Every proposal, including an ambiguous
native measurement, then receives all-legend and line-null measurements at
that SAME center and approved scale. Only this module emits a final verdict.
There is no independent legend extraction, raster calibration, or post-filter.
"""
import time
import cv2
import numpy as np

from occlusion_aware_window_verifier import verify_marker_window
from bw_suppressed_v46 import WINDOW_METRICS
from bw_joint_raster_window_v46 import ResourceLimit

VERSION = 'shared-legend-single-window-v2'


def combine_evidence(physical, comparison, key, *, fill=None, confidence_floor=None):
    """One explicit verdict, without averaging away missing ink or fill conflicts.

    Legacy physical thresholds and common-raster thresholds are unchanged.
    A rival may challenge the identity, but cannot move/relabel a seed onto a
    neighbouring marker. A supported unresolved identity remains suppressed.
    """
    own = comparison['scores'][key]
    checks = dict(
        required_ink_and_outline=physical['decision'] == 'verified',
        fill_consistent=fill is None or fill.get('decision') not in ('conflict', 'abstain'),
        legend_model_supported=bool(own['passed']),
        same_center_identity=comparison['winner'] == key,
        confidence_floor=confidence_floor is None or physical['required_recall'] >= confidence_floor)
    # Uncertain fill is not a veto in the native policy either.
    if fill is not None and fill.get('style') == 'uncertain' and fill.get('decision') == 'abstain':
        checks['fill_consistent'] = True
    if all(checks.values()):
        return 'verified', None, checks
    if not checks['required_ink_and_outline']:
        return physical['decision'], 'window_required_ink_outline_or_fill', checks
    if not checks['fill_consistent']:
        return ('rejected' if fill['decision'] == 'conflict' else 'ambiguous'), 'interior_fill_conflict', checks
    if not checks['confidence_floor']:
        return 'rejected', 'confidence_floor', checks
    if not checks['legend_model_supported']:
        return 'rejected', 'raster_marker_not_supported', checks
    return 'ambiguous', 'identity_ambiguous', checks


def recovery_guard(competition, image, templates, *, geometry=None,
                   fill_competitor=None, minimum_recall=None):
    """New raster-recovery proposals use the SAME decision, not a back door.

    Already accepted grid points are never rechecked. This guard is called only
    for a proposed new/replacement body, at its locked integer observation.
    """
    from bw_final_alignment_v46 import locked_window
    by_key = {t.key: t for t in templates}
    cache = {}
    def check(key, center, comparison):
        x,y = map(float, center)
        if (x,y) != (round(x),round(y)) or tuple(comparison['center']) != (x,y):
            raise ValueError('Recovery must use the common native pixel observation')
        if (key,x,y) in cache:
            return cache[key,x,y]
        t = by_key[key]
        scale = comparison['scores'][key]['parameters']['raster_scale']
        if abs(scale - competition.scales[key]) > 1.e-8:
            raise ValueError('Recovery cannot substitute a separately calibrated scale')
        v = locked_window(geometry.image_for(t) if geometry is not None else image,
                          t, (x,y), scale=scale)
        if v is None:
            return dict(version=VERSION, decision='rejected', reason='unmeasurable_native_window')
        if geometry is not None:
            geometry.assess(t,v,record=False)
        fill = (None if fill_competitor is None else fill_competitor.evaluate_fill(
            t,x,y,total_scale=scale,aspect=1.))
        decision,reason,checks = combine_evidence(
            dict(decision=v.decision,required_recall=v.required_recall),comparison,key,
            fill=fill,confidence_floor=minimum_recall)
        result = dict(version=VERSION,decision=decision,reason=reason,checks=checks,
            center=[x,y],scale=scale,native_measurement_state=v.decision,
            required_recall=v.required_recall,minimum_sector_recall=v.minimum_sector_recall,
            geometry_first=v.compute_diagnostics.get('geometry_first'),
            ink_evidence=v.compute_diagnostics.get('ink_evidence'),
            new_proposals_only=True,center_locked=True,second_identity_filter=False)
        cache[key,x,y]=result
        return result
    return check


def verify_candidates(competition, image, proposals, templates, plot, exclusions,
                      *, geometry=None, fill_competitor=None, identity_competitor=None,
                      geometry_identity_competitor=None, approved_scales,
                      minimum_recall=None, maximum_centres=6000):
    if not competition.ready:
        raise ValueError('Shared legend comparator is not ready')
    if competition.report.get('source_model_policy') != 'shared_native_window':
        raise ValueError('Shared windows must not silently use independently extracted rasters')
    if len(proposals) > maximum_centres:
        raise ResourceLimit(f'{len(proposals)} window proposals exceed {maximum_centres}')
    started = time.perf_counter()
    by_key = {t.key: t for t in templates}
    measurements = []
    # No proposal is discarded here. The native result is one evidence channel
    # in the final decision, and its center is never optimized by another type.
    for p in proposals:
        t = by_key[p['template']]
        # Native patches use integer rounding. Make that observation center
        # explicit so the raster channel cannot sample a different subpixel
        # patch while reporting an apparently identical center.
        v = verify_marker_window(geometry.image_for(t) if geometry is not None else image,
            t, round(p['x']), round(p['y']), search_scales=(approved_scales[t.key],), lock_aspect=True)
        if geometry is not None:
            geometry.assess(t, v)
        measurements.append(v)
    physical_seconds = time.perf_counter() - started
    centers = list(dict.fromkeys((v.aligned_x, v.aligned_y) for v in measurements))
    scores_started = time.perf_counter()
    scores = dict(zip(centers, competition.score_cached(centers)))
    model_seconds = time.perf_counter() - scores_started
    records = []; verified = []; measured = {}; stats = []
    for index, (p, v) in enumerate(zip(proposals, measurements)):
        key = p['template']; t = by_key[key]
        comparison = scores[v.aligned_x, v.aligned_y]
        own = comparison['scores'][key]
        # The native shared-symbol policy is locked to this very scale.
        if abs(v.scale - approved_scales[key]) > 1.e-8 or abs(own['parameters']['raster_scale'] - v.scale) > 1.e-8:
            raise ValueError('Window and legend comparison used different scales')
        fill = (None if fill_competitor is None else fill_competitor.evaluate_fill(
            t, v.aligned_x, v.aligned_y, total_scale=v.scale, aspect=v.aspect_ratio))
        state, reason, checks = combine_evidence(
            dict(decision=v.decision, required_recall=v.required_recall), comparison, key,
            fill=fill, confidence_floor=minimum_recall)
        x,y = v.aligned_x, v.aligned_y
        if not (plot[0] <= x < plot[2] and plot[1] <= y < plot[3]):
            state, reason = 'rejected', 'aligned_centre_outside_plot'
        elif any(a <= x < c and b <= y < d for a,b,c,d in exclusions):
            state, reason = 'rejected', 'aligned_centre_in_legend'
        diagnostic = v.compute_diagnostics
        record = dict(p, candidate_index=index, template=key, swatch_id=t.swatch_id,
            class_name=t.name, shape_hint=t.shape_hint or t.name, selected=False,
            decision=state, aligned_x=x, aligned_y=y,
            required_recall=v.required_recall, window_scale=v.scale,
            window_scale_reference='legend', scale_policy='shared_symbol',
            matching_profile=VERSION, geometry_fixed=False,
            observed_raster_scale=v.scale, observed_raster_diameter=t.diameter,
            effective_diameter=t.diameter*v.scale, raster_identity=comparison,
            joint_window=dict(version=VERSION, single_decision=True, checks=checks,
                native_measurement_state=v.decision, same_center=True, same_scale=True,
                shared_legend_model=True, independent_reextraction=False,
                localization='own_type_native_boundary', second_identity_filter=False,
                marker_support_passed=checks['required_ink_and_outline'] and checks['legend_model_supported'],
                identity_resolved=checks['same_center_identity']))
        for field in WINDOW_METRICS:
            value = getattr(v, field, None)
            record[field] = None if value is None else float(value)
        for field in ('strict_core_recall','boundary_recall','contour_support','aspect_ratio'):
            if record[field] is None:
                record[field] = 1. if field == 'aspect_ratio' else 0.
        for field in ('open_interior','geometry_first','circle_rim','ink_evidence','final_alignment'):
            if field in diagnostic:
                record[field] = diagnostic[field]
        if fill is not None:
            record['fill_identity'] = fill
        if reason:
            if state == 'ambiguous' and reason == 'window_required_ink_outline_or_fill':
                record['deferred_reason'] = reason
            else:
                record['exclusion_reason'] = reason
        # Preserve the original measured appearance and anchor for Step 5.
        # The common raster's nuisance lines must never enter the marker mask.
        rendered = geometry.appearance(t, v) if geometry is not None else v.template_mask
        measured[index] = dict(mask=rendered, marker_offset_x=x-v.x, marker_offset_y=y-v.y)
        yy,xx = np.nonzero(v.template_mask)
        if len(xx) >= 3:
            hull = cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)).reshape(-1,2)
            hull += np.array([round(v.x)-v.template_mask.shape[1]//2,
                              round(v.y)-v.template_mask.shape[0]//2])
            record['footprint_polygon'] = hull.tolist()
        # Ranking uses only the one common legend comparison. Physical checks
        # above remain constraints; a high average cannot cancel a missing rim.
        record['selection_rank'] = record['base_selection_rank'] = 1.-own['loss']
        records.append(record)
        if state == 'verified':
            verified.append(record)
        stats.append(dict(swatch_id=key, x=float(p['x']), y=float(p['y']),
            backend='cpu', used_cuda=False, joint_window=record['joint_window']))
    report = dict(version=VERSION, status='completed', candidates=len(proposals),
        centres=len(centers), retained=len(verified), seconds=time.perf_counter()-started,
        physical_measurement_seconds=physical_seconds, legend_comparison_seconds=model_seconds,
        single_decision=True, legacy_window_measurements_reused=True,
        independent_legend_extraction=False, independent_scale_calibration=False,
        second_identity_filter=False, reference_points_read=False, source_pixels_modified=False,
        thresholds_unchanged=True)
    return dict(records=records, verified=verified, measured_geometry=measured,
                window_stats=stats, report=report)

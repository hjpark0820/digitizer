"""Native common-ink fallback for a caller-declared single marker series.

All source colours are collapsed into binary non-paper evidence. Recurrent
thick cores propose a common filled-disk hypothesis; each local proposal must
still pass density, off-stroke, long-line and validity gates. Dense repeated
guides can be ignored in scoring, with separate original-ink size voting.
There is no directional path proposal: common-ink paths can follow chart axes.
No experiment modules, saved detections, or reference coordinates are inputs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
from time import perf_counter

import numpy as np

from color_marker_evidence import _paper
from .disk import Config, estimate_common_radius, measure_line_width, scan_path
from .palette_base import PaletteConfig


VERSION = 'legend_optional_v46_common_ink_guide_ignore_v2'
SERIES_ID = 'S01'


def _inputs(image_bgr, plot_box):
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('image_bgr must be a native uint8 BGR image')
    values = np.asarray(plot_box)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError('plot_box must contain four integer half-open coordinates')
    x0, y0, x1, y1 = map(int, values)
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError('plot_box must be nonempty and contained in the source image')
    return image, [x0, y0, x1, y1], image[y0:y1, x0:x1].copy()


def _display_rgb(crop, ink, valid, points, plot, radius):
    """Choose an actually observed source colour for display, never inference."""
    selected = np.zeros(ink.shape, bool)
    if points and radius is not None:
        reach = max(1, math.ceil(.4*radius))
        for point in points:
            x, y = point['x_px']-plot[0], point['y_px']-plot[1]
            a, b = max(0, math.floor(x-reach)), max(0, math.floor(y-reach))
            c, d = min(ink.shape[1], math.ceil(x+reach)+1), min(ink.shape[0], math.ceil(y+reach)+1)
            yy, xx = np.mgrid[b:d, a:c]
            selected[b:d, a:c] |= (xx-x)**2+(yy-y)**2 <= (.4*radius)**2
    selected &= (ink > 0) & valid
    origin = 'original_pixels_in_verified_body_cores'
    if not selected.any():
        selected = (ink > 0) & valid
        origin = 'original_nonpaper_pixels_no_verified_body_colour'
    pixels = crop[selected]
    if not len(pixels):
        return None, 'no_observed_nonpaper_colour'
    # An actual source pixel nearest the median avoids inventing an RGB colour
    # by combining the median channels of differently coloured marker parts.
    median = np.median(pixels, axis=0)
    index = int(np.argmin(np.sum((pixels.astype(float)-median)**2, axis=1)))
    return [int(value) for value in pixels[index, ::-1]], origin


def _deduplicate(candidates, radius, config):
    priority = {'active': 2, 'suppressed': 1, 'rejected': 0}
    selected = []
    for point in sorted(candidates, key=lambda q: (priority[q['status']], q['score']), reverse=True):
        if any(np.hypot(point['x']-q['x'], point['y']-q['y']) <
               config.duplicate_radius_fraction*radius for q in selected):
            continue
        selected.append(dict(point))
    return sorted(selected, key=lambda q: (q['x'], q['y']))


def _local_probes(own, valid, radius, width, size, plot, config, ignore=None):
    """Exact local core-seeded scan from the fixed common-ink experiment."""
    candidates, seed_reports = [], []
    for index, record in enumerate(size['candidates']):
        cx, cy = record['x'], record['y']
        padding = math.ceil(6*radius)
        a, b = max(0, cx-padding), max(0, cy-padding)
        c, d = min(own.shape[1], cx+padding+1), min(own.shape[0], cy+padding+1)
        span = math.ceil(2*radius)
        xs = np.arange(max(a, cx-span), min(c-1, cx+span)+1, dtype=float)
        path = np.column_stack((xs-a, np.full(len(xs), cy-b, dtype=float)))
        patch = own[b:d, a:c]
        scanned = scan_path(path, patch, patch, np.zeros_like(patch), valid[b:d, a:c],
                            radius, width, config,
                            **({'ignore':ignore[b:d,a:c]} if ignore is not None else {}))
        nearby = []
        for raw in scanned['candidates']:
            if np.hypot(raw['x']+a-cx, raw['y']+b-cy) > config.duplicate_radius_fraction*radius:
                continue
            point = dict(raw)
            for key, shift in [('x', plot[0]+a), ('y', plot[1]+b),
                               ('path_x', plot[0]+a), ('path_y', plot[1]+b)]:
                point[key] += shift
            point.update(id=f'common_core{index+1:03d}_'+point['id'],
                x_px=point['x'], y_px=point['y'], series_id=SERIES_ID,
                class_name='filled_circle', kind='common_ink_disk_hypothesis',
                proposal_source='observed_distance_maximum_local_probe',
                marker_glyph_detected=point['status'] == 'active',
                shape_source='analytic_shared_disk_hypothesis',
                hypothesis_verified_by_source_density=point['status'] == 'active',
                local_seed_index=index)
            nearby.append(point)
        candidates.extend(nearby)
        seed_reports.append(dict(index=index, seed_plot_local=dict(record),
            seed_source=[float(cx+plot[0]), float(cy+plot[1])],
            local_candidate_ids=[q['id'] for q in nearby],
            local_candidate_count=len(nearby), all_local_peak_count=len(scanned['candidates'])))
    return _deduplicate(candidates, radius, config), seed_reports


def _template(radius, rgb):
    extent = math.ceil(radius)
    yy, xx = np.mgrid[-extent:extent+1, -extent:extent+1]
    disk = (xx*xx+yy*yy <= radius*radius).astype(np.float32)
    return dict(id=SERIES_ID, kind='shared_disk_hypothesis', soft=disk,
        diameter=2*radius, radius=radius, center=[extent, extent], rgb=rgb,
        legend_box=None, composition=dict(best_model_name='circle'),
        class_name='filled_circle', provenance=dict(kind='shared_disk_hypothesis',
            radius_source='recurrent_observed_thick_cores', not_observed_clean_template=True,
            analytical_hypothesis=True, single_series_condition='caller_supplied'))


def detect_common_ink(image_bgr, plot_box, *, ignore_repeated_guides=True):
    """Return verified local disk hypotheses or explicit unresolved evidence.

    ``plot_box`` is half-open in original pixels. Points use full-image native
    coordinates; ``_fields`` uses plot-local native arrays. No internal axis,
    grid, guide, or source colour is deleted. Repeated-guide ink may be excluded
    from scoring through a separate ignore mask. The one-pixel ROI-edge inset
    remains physically unavailable for verification. Suppressed points remain
    empty because one common ink channel cannot establish coloured occlusion.
    """
    started = perf_counter()
    image, plot, crop = _inputs(image_bgr, plot_box)
    paper = _paper(crop)
    minimum_contrast = PaletteConfig().min_contrast
    contrast = np.linalg.norm(paper-crop.astype(np.float32), axis=-1)
    ink = (contrast >= minimum_contrast).astype(np.float32)
    valid = np.ones(ink.shape, bool)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    own = ink*valid
    width, _ = measure_line_width(own, valid)
    config = Config(use_long_line_guard=True)
    candidates, seeds, radius = [], [], None
    ignore=np.zeros_like(valid)
    guide_audit=dict(enabled=ignore_repeated_guides,groups=[],ignored_pixels=0)
    try:
        radius, size = estimate_common_radius([own], valid)
        size['status'] = 'recurrent_radius_supported'
        if ignore_repeated_guides:
            from .guide_ignore import prepare
            ignore, replacement, guide_audit=prepare(crop,own,valid,size,paper)
            guide_audit['enabled']=True
            if replacement is not None:
                size=replacement
                radius=size.get('radius')
    except ValueError as error:
        size = dict(status='unresolved', reason=str(error), supporting_count=0,
                    failure_implies_no_markers=False)
    size['coordinate_system'] = 'plot-local native pixel centres'
    if radius is not None:
        args=(own, valid, radius, width, size, plot, config)
        candidates, seeds = (_local_probes(*args, ignore) if ignore.any()
                             else _local_probes(*args))
    verified = [point for point in candidates if point['status'] == 'active']
    completed = radius is not None and len(verified) >= 3
    status = 'completed' if completed else 'uncertain'
    reason = ('recurrent_common_radius_and_three_verified_independent_local_bodies' if completed
              else 'common_radius_unresolved' if radius is None
              else 'fewer_than_three_verified_independent_local_bodies')
    rgb, rgb_source = _display_rgb(crop, ink, valid, verified, plot, radius)
    template = _template(radius, rgb) if radius is not None else None
    zero = np.zeros_like(ink)
    fields = dict(membership=ink[None, ...].copy(), colour_confidence=ink[None, ...].copy(),
        other=zero[None, ...].copy(), all_ink=ink.copy(), unknown=zero.copy(),
        paper=1.-ink, paper_bgr=paper.copy(), equivalent_colours=np.ones((1, 1), bool),
        palette_rgb=np.asarray([rgb or [0, 0, 0]], np.uint8), valid=valid.copy(),
        common_ink=ink.copy(), ignore_mask=ignore.copy(), crop_bgr=crop, coordinate_system='plot-local native pixels',
        plot_box_source=plot.copy(), native_shape=list(ink.shape),
        source_center_offset=plot[:2], scale=1., scale_x=1., scale_y=1.,
        evidence_mode='binary_common_ink', palette_rgb_is_display_only=True)
    series = []
    if rgb is not None:
        series.append(dict(id=SERIES_ID, label='Series 1', rgb=rgb, role='data_series',
            line_width=width, diameter=2*radius if radius is not None else None,
            marker_state='present' if completed else 'unknown',
            colour_source=rgb_source, colour_used_for_detection=False,
            accepted_count=len(verified) if completed else 0))
    audit = dict(version=VERSION, decision=status, reason=reason,
        paper_bgr=paper.tolist(), minimum_contrast=minimum_contrast,
        binary_rule='Euclidean BGR distance from estimated paper >= minimum_contrast',
        all_nonpaper_colours_have_identical_ink=True, colour_labels_used=False,
        radius=radius, line_width=width, size_estimation=size, config=asdict(config),
        guide_ignore=guide_audit,
        marker_evidence_colour_independent=True,
        guide_pixel_selection_uses_observed_ink_direction=bool(guide_audit.get('groups')),
        local_seed_count=len(seeds), local_seed_reports=seeds,
        candidate_counts=dict(Counter(q['status'] for q in candidates)),
        rejection_reasons=dict(Counter(q['reason'] for q in candidates if q['status'] == 'rejected')),
        verified_independent_body_count=len(verified), minimum_verified_bodies=3,
        raw_path_candidates_admitted=False, source_maxima_are_proposals_only=True,
        border_candidates_promoted=False, internal_axis_pixels_excluded=False,
        spatial_exclusion_policy='one_native_pixel_roi_edge_only',
        display_colour_source=rgb_source, display_colour_is_series_identity=False,
        uses_reference_points=False, uses_saved_detections=False, native_resolution=True,
        suppressed_policy='No occlusion or strong-shape suppressed points fabricated from common ink',
        limitations=['One series is supplied by the caller, not inferred.',
            'Filled circles with a recurrent common radius are a working hypothesis.',
            'Compact bodies may be annotations; source shape and recurrence do not establish semantic curve membership.',
            'Long strokes can explain real marker ink; failing verification remains uncertain.',
            'Clipped edge bodies may fail validity or have biased centre estimates.'])
    return dict(version=VERSION, status=status, reason=reason, chart_kind='markers' if completed else 'uncertain',
        plot_box=plot, plot_box_source=plot.copy(), source_shape=list(image.shape[:2]),
        coordinate_system='full-image native pixel centres', native_resolution=True,
        points=verified if completed else [], verified_points=verified,
        suppressed_points=[], uncertain_points=[], candidates=candidates,
        template=template, templates=[template] if completed else [],
        series=series, rgb=rgb, radius=radius, line_width=width, paths=[],
        audit=audit, _fields=fields, seconds=perf_counter()-started)

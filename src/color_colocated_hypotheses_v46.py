"""Conditional, co-located marker hypotheses for production v46 path correction.

A donor is an ORIGINAL active detection, never an iteratively added point.
An alternative legend shape may share its centre.  This is an occlusion
hypothesis, not a second observed marker.  Candidate existence and candidate
shape remain uncertain; the target series specifies a conditional legend
assignment for evaluation only.  No production suppressed/strong policy is
changed, and no input point, mask, reference, or template is modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

try:
    from .color_path_metrics_v46 import PathReference
except ImportError:
    from color_path_metrics_v46 import PathReference


@dataclass(frozen=True)
class ColocatedConfig:
    path_distance_diameters: float = 1.0
    x_neighborhood_diameters: float = 2.0
    duplicate_distance_px: float = 1.0
    template_core_alpha: float = .65
    line_corridor_diameters: float = .12
    minimum_own_core_pixels: int = 8
    minimum_off_path_pixels: int = 4
    minimum_occluder_fraction: float = .10
    maximum_unexplained_missing_fraction: float = .30
    minimum_own_sectors: int = 2
    # Enabled by the series router, not by legacy direct path experiments.
    missing_x_only: bool = False
    allow_same_shape: bool = False
    retain_without_path: bool = False

    def __post_init__(self):
        for key in ('path_distance_diameters', 'x_neighborhood_diameters'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('duplicate_distance_px', 'line_corridor_diameters'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        for key in ('template_core_alpha',
                    'minimum_occluder_fraction', 'maximum_unexplained_missing_fraction'):
            value = getattr(self, key)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{key} must be in [0, 1]')
        if self.template_core_alpha == 0:
            raise ValueError('template_core_alpha must be positive')
        for key in ('minimum_own_core_pixels', 'minimum_off_path_pixels', 'minimum_own_sectors'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')


def _inside(x, y, box):
    return box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _valid_center(x, y, payload):
    return (np.isfinite([x, y]).all() and _inside(x, y, payload['plot_area'])
            and not _inside(x, y, payload.get('legend_box')))


def _point_id(point, series, index):
    """Immutable source-index identity; detector IDs are not globally unique."""
    return f'{series}:L0:{index:03d}'


def _distance(query, starts, ends):
    query = np.asarray(query, float).reshape(-1, 2)
    if len(starts) == 0:
        return np.full(len(query), np.inf)
    delta = ends - starts
    den = np.sum(delta*delta, axis=1)
    projection = np.sum((query[:, None, :] - starts[None, :, :])*delta[None, :, :], axis=2)
    t = np.divide(projection, den[None, :], out=np.zeros_like(projection), where=den[None, :] > 0)
    closest = starts[None, :, :] + np.clip(t, 0, 1)[..., None]*delta[None, :, :]
    return np.sqrt(np.min(np.sum((query[:, None, :] - closest)**2, axis=2), axis=1))


def _near_segments(reference, x, diameter, cfg):
    starts, ends = reference.segment_starts, reference.segment_ends
    margin = cfg.x_neighborhood_diameters*diameter
    take = ((np.minimum(starts[:, 0], ends[:, 0]) <= x + margin)
            & (np.maximum(starts[:, 0], ends[:, 0]) >= x - margin))
    return starts[take], ends[take]


def _stamp_crop(row, cx, cy, box):
    """Sample native completed alpha at its immutable source-pixel centre."""
    x0, y0, x1, y1 = box
    alpha = np.asarray(row.get('marker_alpha', []), np.float32)
    if alpha.ndim != 2 or not alpha.size:
        return np.zeros((y1-y0, x1-x0), np.float32)
    tc = row.get('template_center', [(alpha.shape[1]-1)/2, (alpha.shape[0]-1)/2])
    xx, yy = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    mx, my = xx - float(cx) + float(tc[0]), yy - float(cy) + float(tc[1])
    return cv2.remap(alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _pixel_evidence(target, donors, rows, reference, payload, cx, cy, cfg):
    """Fixed image diagnostics; curve improvement is deliberately not an input.

    A potential occluder requires OTHER-series ink inside an original donor's
    template, not merely an arbitrary other-colour line in the window.  Ink
    claimed by multiple colour masks is ambiguous and not positive evidence.
    Off-path own ink is required to avoid accepting a connecting line alone.
    The heuristic eligibility flag is NOT a strong-evidence certification.
    """
    alpha = np.asarray(target.get('marker_alpha', []), np.float32)
    if alpha.ndim != 2 or not alpha.size or 'ink_mask' not in target:
        return dict(activation_eligible=False, status='missing_native_template_or_mask',
                    reason='No independent native-pixel evidence is available.')
    shape = np.asarray(target['ink_mask']).shape
    if len(shape) != 2:
        raise ValueError('ink_mask must be a full source-coordinate 2D mask')
    center = target.get('template_center', [(alpha.shape[1]-1)/2, (alpha.shape[0]-1)/2])
    x0 = max(0, int(math.floor(cx-center[0]))-1)
    y0 = max(0, int(math.floor(cy-center[1]))-1)
    x1 = min(shape[1], int(math.ceil(cx-center[0]+alpha.shape[1]))+1)
    y1 = min(shape[0], int(math.ceil(cy-center[1]+alpha.shape[0]))+1)
    box = [x0, y0, x1, y1]
    expected = _stamp_crop(target, cx, cy, box) >= cfg.template_core_alpha
    yy, xx = np.mgrid[y0:y1, x0:x1]
    plot = payload['plot_area']; legend = payload.get('legend_box')
    valid = ((xx >= plot[0]) & (xx <= plot[2]) & (yy >= plot[1]) & (yy <= plot[3]))
    if legend is not None:
        valid &= ~((xx >= legend[0]) & (xx <= legend[2]) & (yy >= legend[1]) & (yy <= legend[3]))
    expected &= valid
    own = np.asarray(target['ink_mask'], bool)[y0:y1, x0:x1]
    other = np.zeros(expected.shape, bool)
    for name, row in rows.items():
        if name != target['name'] and 'ink_mask' in row:
            ink = np.asarray(row['ink_mask'], bool)
            if ink.shape != shape:
                raise ValueError('All ink masks must have identical full-image shapes')
            other |= ink[y0:y1, x0:x1]
    donor_ink = np.zeros(expected.shape, bool)
    for donor in donors:
        row = rows[donor['series']]
        if 'ink_mask' not in row:
            continue
        silhouette = _stamp_crop(row, donor['cx'], donor['cy'], box) >= cfg.template_core_alpha
        donor_ink |= silhouette & np.asarray(row['ink_mask'], bool)[y0:y1, x0:x1]
    positive = expected & own & ~other
    ambiguous = expected & own & other
    occluded = expected & donor_ink & other & ~own
    missing = expected & ~(positive | occluded | ambiguous)
    starts, ends = (_near_segments(reference, cx, reference.diameter, cfg)
                    if reference is not None else (np.empty((0,2)), np.empty((0,2))))
    q = np.column_stack((xx.ravel(), yy.ravel()))
    off_path = (_distance(q, starts, ends).reshape(xx.shape)
                > max(1., cfg.line_corridor_diameters*target['diameter']))
    if not len(starts):
        # No line reference is not proof that target ink is off a line.
        off_path[:] = False
    off_support = positive & off_path
    # Four angular sectors, with a >=2-pixel minimum in each occupied sector.
    angles = (np.arctan2(yy-cy, xx-cx) + 2*np.pi) % (2*np.pi)
    sectors = np.minimum(3, (angles/(np.pi/2)).astype(int))
    sector_counts = [int(np.sum(off_support & (sectors == k))) for k in range(4)]
    count = int(expected.sum()); den = max(count, 1)
    own_count, off_count = int(positive.sum()), int(off_support.sum())
    occupied = sum(n >= 2 for n in sector_counts)
    checks = dict(nonempty_template=count > 0,
                  own_core_pixels=own_count >= cfg.minimum_own_core_pixels,
                  off_path_pixels=off_count >= cfg.minimum_off_path_pixels,
                  multiple_sectors=occupied >= cfg.minimum_own_sectors,
                  donor_occlusion=int(occluded.sum())/den >= cfg.minimum_occluder_fraction,
                  explained_template=int(missing.sum())/den <= cfg.maximum_unexplained_missing_fraction)
    eligible = all(checks.values())
    return dict(status='evaluated', activation_eligible=eligible,
                classification='independent_fragment_candidate' if eligible else 'review_only',
                expected_core_pixels=count, own_core_pixels=own_count,
                own_core_fraction=own_count/den, off_path_own_pixels=off_count,
                own_sector_counts=sector_counts, own_occupied_sectors=occupied,
                potential_donor_occluder_pixels=int(occluded.sum()),
                potential_donor_occluder_fraction=int(occluded.sum())/den,
                ambiguous_colour_pixels=int(ambiguous.sum()),
                unexplained_missing_pixels=int(missing.sum()),
                unexplained_missing_fraction=int(missing.sum())/den,
                visible_fragment_coverage=own_count/max(int((expected & ~(occluded | ambiguous)).sum()), 1),
                occlusion_compatible=bool(count and int(occluded.sum())/den >= cfg.minimum_occluder_fraction
                    and int(missing.sum())/den <= cfg.maximum_unexplained_missing_fraction),
                line_reference_available=bool(len(starts)),
                checks=checks, failed_checks=[key for key, ok in checks.items() if not ok],
                crop_xyxy_half_open=box,
                eligibility_is_strong_evidence=False, curve_score_used=False,
                reason=('Independent own-colour fragment plus donor-occlusion compatibility; '
                        'still an unvalidated conditional candidate.' if eligible else
                        'Structural plausibility alone cannot activate an invisible marker.'))


def generate(payload, paths, config=None):
    """Return immutable-L0 alternatives, preserving cross-series shared centres.

    ``payload`` is the hydrated native ``load_payload`` result. ``paths`` maps
    each series name to a PathReference or the frozen record used to create it.
    New candidates are in ``by_series``; position-level unresolved alternatives
    are in ``sites``. Existing same-target points are never copied or moved:
    ``duplicate_merges`` links donor provenance to them without changing their
    strength, class, or active state. All values returned are JSON-compatible.
    """
    cfg = config or ColocatedConfig()
    rows = {r['name']: r for r in payload['curves']}
    if len(rows) != len(payload['curves']):
        raise ValueError('Series names must be unique')
    refs = {}
    for name, row in rows.items():
        if name in paths and (isinstance(paths[name], PathReference) or paths[name].get('path') is not None):
            try:
                refs[name] = (paths[name] if isinstance(paths[name], PathReference)
                              else PathReference.from_record(paths[name], row['diameter']))
            except (ValueError, KeyError):
                if not cfg.retain_without_path:
                    raise
    raw_sites = {}; exclusions = []
    for name, row in rows.items():
        for index, point in enumerate(row.get('init_points', [])):
            cx, cy = float(point['cx']), float(point['cy'])
            identity = _point_id(point, name, index)
            if not _valid_center(cx, cy, payload):
                exclusions.append(dict(donor_id=identity, reason='outside_plot_or_inside_legend'))
                continue
            # Exact native centre, no shift, x-clustering, jitter, or averaging.
            key = (cx, cy)
            raw_sites.setdefault(key, []).append(dict(donor_id=identity, series=name,
                original_index=index, detector_id=point.get('candidate_id', point.get('id')),
                cx=cx, cy=cy, marker_class=row.get('marker_class'), source='immutable_L0'))
    output = {name: [] for name in rows}; sites = []; merges = []
    for site_index, ((cx, cy), donors) in enumerate(sorted(raw_sites.items()), 1):
        sid = f'COLOC_{site_index:03d}'
        site = dict(site_id=sid, cx=cx, cy=cy, donors=donors,
                    allowed_series=[], allowed_marker_classes=[], alternatives=[],
                    existence='unknown', symbol_assignment='unresolved_legend_alternatives')
        for name, row in rows.items():
            cls = row.get('marker_class')
            eligible_donors = [d for d in donors if d['series'] != name and
                               cls and d['marker_class'] and (cfg.allow_same_shape or cls != d['marker_class'])]
            if not eligible_donors:
                continue
            if cfg.missing_x_only and any(abs(float(p['cx'])-cx) <= .45*row['diameter']
                                         for p in row.get('init_points', [])):
                exclusions.append(dict(site_id=sid, target_series=name, reason='target_x_slot_already_active'))
                continue
            ref = refs.get(name)
            if ref is None and not cfg.retain_without_path:
                exclusions.append(dict(site_id=sid, target_series=name, reason='no_reference_path'))
                continue
            starts, ends = (_near_segments(ref, cx, ref.diameter, cfg)
                            if ref is not None else (np.empty((0,2)), np.empty((0,2))))
            distance = float(_distance([[cx, cy]], starts, ends)[0])
            if ((len(starts) and distance > cfg.path_distance_diameters*ref.diameter)
                    or (not len(starts) and not cfg.retain_without_path)):
                exclusions.append(dict(site_id=sid, target_series=name,
                    reason='outside_reference_path_neighborhood',
                    distance_px=distance if np.isfinite(distance) else None))
                continue
            site['allowed_series'].append(name)
            if cls not in site['allowed_marker_classes']:
                site['allowed_marker_classes'].append(cls)
            duplicate = None
            for state, field in [('active', 'init_points'), ('suppressed', 'init_suppressed')]:
                for idx, existing in enumerate(row.get(field, [])):
                    if math.hypot(float(existing['cx'])-cx, float(existing['cy'])-cy) <= cfg.duplicate_distance_px:
                        duplicate = dict(site_id=sid, target_series=name, existing_state=state,
                            existing_index=idx, existing_id=existing.get('candidate_id', existing.get('id')),
                            existing_cx=float(existing['cx']), existing_cy=float(existing['cy']),
                            donor_ids=[d['donor_id'] for d in eligible_donors],
                            conditional_marker_class=cls, coordinates_changed=False,
                            strength_changed=False)
                        break
                if duplicate is not None:
                    break
            if duplicate is not None:
                merges.append(duplicate)
                site['alternatives'].append(dict(target_series=name, conditional_marker_class=cls,
                    state='merged_existing', existing_state=duplicate['existing_state']))
                continue
            evidence = _pixel_evidence(row, eligible_donors, rows, ref, payload, cx, cy, cfg)
            if not len(starts) and cfg.retain_without_path and not evidence.get('occlusion_compatible'):
                exclusions.append(dict(site_id=sid, target_series=name, reason='no_path_or_compatible_occluder'))
                continue
            candidate = dict(candidate_id=f'{sid}:{name}', site_id=sid, cx=cx, cy=cy,
                class_name='suppressed', class_idx=-1, target_series=name,
                conditional_marker_class=cls, symbol_assignment='conditional_legend_hypothesis',
                existence='unknown', tentative=True, state='suppressed', auto_promote=False,
                source='colocated_original_marker', evidence_tier='structural_occlusion_hypothesis',
                evidence_supported=False, donor_ids=[d['donor_id'] for d in eligible_donors],
                donor_provenance=eligible_donors, original_center_unchanged=True,
                shared_observed_x=True,
                reference_path_distance_px=distance if np.isfinite(distance) else None,
                reference_path_distance_diameters=distance/ref.diameter if np.isfinite(distance) else None,
                review_status=('partial_fragment' if evidence['activation_eligible'] else
                    'occluded_unresolved' if evidence.get('occlusion_compatible') else 'insufficient_image_evidence'),
                activation_eligible=bool(evidence['activation_eligible']),
                image_evidence=evidence)
            output[name].append(candidate)
            site['alternatives'].append(dict(target_series=name, conditional_marker_class=cls,
                candidate_id=candidate['candidate_id'], state='conditional_suppressed',
                activation_eligible=candidate['activation_eligible']))
        if site['alternatives']:
            for name in site['allowed_series']:
                for candidate in output[name]:
                    if candidate['site_id'] == sid:
                        candidate['allowed_series'] = list(site['allowed_series'])
                        candidate['allowed_marker_classes'] = list(site['allowed_marker_classes'])
            sites.append(site)
    all_new = [c for candidates in output.values() for c in candidates]
    return dict(by_series=output, sites=sites, duplicate_merges=merges,
        audit=dict(config=asdict(cfg), original_active_count=sum(len(r.get('init_points', [])) for r in rows.values()),
            source_sites=len(raw_sites), retained_sites=len(sites), new_candidates=len(all_new),
            activation_eligible=sum(c['activation_eligible'] for c in all_new),
            review_only=sum(not c['activation_eligible'] for c in all_new),
            duplicate_merges=len(merges), exclusions=exclusions,
            original_points_changed=False, production_strong_policy_changed=False,
            fully_hidden_is_positive_evidence=False,
            eligibility_thresholds_are_experimental=True,
            candidate_existence_is_confirmed=False))

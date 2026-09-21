"""Path-local disk-density hypotheses; no image identity or reference points.

Filled round bodies are the working hypothesis for this experiment. Raw colour
fields are never erased. Active means image-supported proposal, not ground truth;
suppressed means possible occlusion, never affirmative proof of a hidden point.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Config:
    active_density: float = .56
    active_offstroke: float = .36
    suppressed_total_density: float = .55
    suppressed_offstroke: float = .30
    suppressed_other_density: float = .16
    minimum_excess: float = .12
    prominence: float = .08
    normal_search_radius_fraction: float = .45
    minimum_valid_fraction: float = .85
    duplicate_radius_fraction: float = 1.20
    use_long_line_guard: bool = False
    own_compact_residual: float = .12
    total_compact_residual: float = .15
    minimum_occluder_overlap: float = .15


def _field(field, shape=None):
    a = np.asarray(field, np.float32)
    if a.ndim != 2 or (shape is not None and a.shape != shape):
        raise ValueError('Expected matching 2D density fields')
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Density fields must be finite and in [0, 1]')
    return a


def disk_kernel(radius):
    if not np.isfinite(radius) or radius < 1:
        raise ValueError('Disk radius must be finite and at least one pixel')
    extent = math.ceil(radius)
    yy, xx = np.mgrid[-extent:extent+1, -extent:extent+1]
    return (xx*xx+yy*yy <= radius*radius).astype(np.float32)


def normalized_disk_density(field, valid, radius):
    """Ink / available disk area, with invalid pixels excluded, not made white."""
    a = _field(field)
    v = np.asarray(valid, bool)
    if v.shape != a.shape:
        raise ValueError('Validity mask differs from image shape')
    kernel = disk_kernel(radius)
    denominator = cv2.filter2D(v.astype(np.float32), -1, kernel,
                              borderType=cv2.BORDER_CONSTANT)
    numerator = cv2.filter2D(a*v, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return np.clip(numerator/np.maximum(denominator, 1.), 0., 1.)


def measure_line_width(own, valid):
    binary = ((_field(own) >= .35) & np.asarray(valid, bool)).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ridge = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    width = max(1., float(2*np.percentile(distance[ridge], 25)-.5)) if ridge.any() else 1.
    return width, distance


def estimate_common_radius(own_fields, valid):
    """Pool repeated thick-core radii across colours; no reference/template input.

    The most populated +/-25% radius neighbourhood must contain at least three
    spatially distinct colour/body observations. Thin line-crossings are filtered
    by their small inscribed radius. The assumption is filled compact markers.
    """
    records = []
    widths = []
    for index, own in enumerate(own_fields):
        width, distance = measure_line_width(own, valid)
        widths.append(width)
        local = cv2.dilate(distance, np.ones((3, 3), np.uint8))
        mask = ((distance >= local-1e-5) & (distance >= max(3., 1.35*width)))
        count, labels, _, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        selected = []
        for label in range(1, count):
            yy, xx = np.nonzero(labels == label)
            j = np.argmin((xx-centers[label, 0])**2+(yy-centers[label, 1])**2)
            x, y = int(xx[j]), int(yy[j]); r = float(distance[y, x])
            selected.append(dict(field_index=index, x=x, y=y, radius=r))
        kept = []
        for record in sorted(selected, key=lambda q: -q['radius']):
            if all(np.hypot(record['x']-q['x'], record['y']-q['y']) >
                   2*max(record['radius'], q['radius']) for q in kept):
                kept.append(record)
        records.extend(kept)
    if len(records) < 3:
        raise ValueError('Cannot establish a shared disk size from three visible thick bodies')
    best = max(([j for j, q in enumerate(records) if .8 <= q['radius']/r['radius'] <= 1.25]
                for r in records), key=lambda group: (len(group), np.median([records[j]['radius'] for j in group])))
    if len(best) < 3:
        raise ValueError('Observed thick bodies do not support a recurrent common radius')
    # Distance transform measures the interior including the central pixel.
    radius = float(np.median([records[j]['radius'] for j in best]))
    return radius, dict(method='shared_recurrent_inscribed_radius', radius=radius,
                        supporting_count=len(best), supporters=[records[j] for j in best],
                        measured_line_widths=widths, candidates=records,
                        uses_marker_coordinates_as_detections=False,
                        limitation='Working hypothesis: filled compact common-size bodies, not a marker-free classifier')


def _sample(field, xy):
    xy = np.asarray(xy, np.float32)
    if len(xy) == 0:
        return np.empty(0, np.float32)
    return cv2.remap(np.asarray(field, np.float32), xy[:, 0:1], xy[:, 1:2],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel()


def long_line_residual(field, valid, radius, line_width):
    """Extra scoring field only; original colour pixels remain untouched.

    Explain sustained straight strokes in any orientation, including adjacent
    other-colour curves and errorbar stems. Marker-sized bodies alone cannot
    support a line kernel twice their diameter.
    """
    binary = ((_field(field) >= .35) & np.asarray(valid, bool)).astype(np.uint8)
    length = int(math.ceil(4*radius)) | 1
    center = length//2
    explained = np.zeros_like(binary)
    for angle in np.linspace(0, np.pi, 24, endpoint=False):
        dx, dy = center*np.cos(angle), center*np.sin(angle)
        kernel = np.zeros((length, length), np.uint8)
        cv2.line(kernel, (round(center-dx), round(center-dy)),
                 (round(center+dx), round(center+dy)), 1, max(1, round(.60*line_width)))
        explained |= cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(field, np.float32)*(1-explained)*np.asarray(valid, bool)


def _resample(path):
    p = np.asarray(path, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('Path must be finite N by 2 coordinates')
    if len(p) < 2:
        return p.copy(), np.zeros(len(p)), np.tile([1., 0.], (len(p), 1))
    segment = np.linalg.norm(np.diff(p, axis=0), axis=1)
    p = p[np.r_[True, segment > 1e-8]]
    if len(p) < 2:
        return p, np.zeros(len(p)), np.tile([1., 0.], (len(p), 1))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    samples = np.linspace(0., arc[-1], max(2, math.ceil(arc[-1])+1))
    xy = np.column_stack([np.interp(samples, arc, p[:, k]) for k in range(2)])
    tangents = np.gradient(xy, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    return xy, samples, tangents


def _local_metrics(own, total, other, valid, x, y, tangent, radius, line_width, ignore=None):
    extent = math.ceil(radius)
    gy, gx = np.mgrid[-extent:extent+1, -extent:extent+1]
    disk = gx*gx+gy*gy <= radius*radius
    coordinates = np.column_stack((gx.ravel()+x, gy.ravel()+y))
    available = _sample(valid.astype(np.float32), coordinates).reshape(disk.shape) > .99
    physical_scope = disk & available
    scope = physical_scope.copy()
    if ignore is not None:
        scope &= _sample(ignore.astype(np.float32), coordinates).reshape(disk.shape) < .01
    half_stroke = max(.8, .65*line_width)
    normal_distance = np.abs(-tangent[1]*gx+tangent[0]*gy)
    # Score only: no source pixels are removed. Exclude the connecting tangent
    # and possible vertical error-bar stem from the independent-body evidence.
    offstroke = scope & (normal_distance > half_stroke) & (np.abs(gx) > half_stroke)
    result = dict(valid_fraction=float(physical_scope.sum()/max(1, disk.sum())))
    if ignore is not None:
        result['evaluated_fraction'] = float(scope.sum()/max(1, physical_scope.sum()))
    for name, field in [('own', own), ('total', total), ('other', other)]:
        patch = _sample(field, coordinates).reshape(disk.shape)
        result[name+'_density'] = float(patch[scope].mean()) if scope.any() else 0.
        result[name+'_offstroke'] = float(patch[offstroke].mean()) if offstroke.any() else 0.
        if ignore is not None:
            result[name+'_observed_ink'] = float(patch[scope].sum())
            quadrants = [(gx>=0)&(gy>=0),(gx<0)&(gy>=0),(gx<0)&(gy<0),(gx>=0)&(gy<0)]
            result[name+'_supported_quadrants'] = sum(
                (patch[offstroke&q]>.5).sum() >= max(2,.02*disk.sum()) for q in quadrants)
    if ignore is not None:
        result['enough_direct_evidence'] = (result['own_observed_ink'] >= max(6,.25*disk.sum())
                                            and result['own_supported_quadrants']>=3)
    return result


def scan_path(path_xy, own, total, other, valid, radius, line_width, config=Config(), *, ignore=None):
    """Sample a fixed path, propose density peaks, then bounded normal refinement.

    Each independent own/total peak is evaluated. Total-ink peaks may only enter
    the suppressed pool unless direct own-colour evidence passes active gates.
    Optional ``ignore`` pixels are absent from numerator and denominator, not
    background evidence. Physical ROI validity remains separate. At least half
    the physical disk, sufficient ink mass and three quadrants must support an
    active claim when this uncertainty mask is used. Default callers unchanged.
    """
    own = _field(own); total = _field(total, own.shape); other = _field(other, own.shape)
    valid = np.asarray(valid, bool)
    if valid.shape != own.shape or not np.isfinite(line_width) or line_width <= 0:
        raise ValueError('Invalid validity mask or stroke width')
    if ignore is not None:
        ignore=np.asarray(ignore,bool)
        if ignore.shape!=valid.shape:
            raise ValueError('Ignore mask differs from density field shape')
    evaluation=valid if ignore is None else valid&~ignore
    kernel = disk_kernel(radius)
    xy, arc, tangent = _resample(path_xy)
    maps = {name: normalized_disk_density(a, evaluation, radius)
            for name, a in [('own', own), ('total', total), ('other', other)]}
    if config.use_long_line_guard:
        maps['own_compact'] = normalized_disk_density(long_line_residual(own, evaluation, radius, line_width), evaluation, radius)
        maps['total_compact'] = normalized_disk_density(long_line_residual(total, evaluation, radius, line_width), evaluation, radius)
    profile = dict(arc_length=arc.copy(), x=xy[:, 0].copy(), y=xy[:, 1].copy())
    for name, a in maps.items():
        profile[name+'_density'] = _sample(a, xy)
    if len(xy) < 3:
        return dict(path_xy=xy, profile=profile, candidates=[], config=asdict(config))
    proposed = []
    for name in ('own', 'total'):
        smooth = gaussian_filter1d(profile[name+'_density'], .7)
        # A long local percentile estimates the connecting-line baseline, not
        # white space beyond an endpoint. A separate full-curve floor prevents
        # continuous dense overlaps from being treated as discrete markers.
        baseline = percentile_filter(smooth, 20, size=(int(math.ceil(8*radius)) | 1), mode='nearest')
        excess = np.maximum(0., smooth-baseline)
        profile[name+'_baseline'] = baseline
        profile[name+'_excess'] = excess
        peaks, _ = find_peaks(excess, height=config.minimum_excess,
                             prominence=config.prominence, distance=max(1, int(radius)))
        indexes = set(map(int, peaks))
        # One-sided real peaks near the source path ends; no mandatory endpoint.
        for lo, hi in [(0, min(len(xy), math.ceil(2*radius)+1)),
                       (max(0, len(xy)-math.ceil(2*radius)-1), len(xy))]:
            j = lo+int(np.argmax(excess[lo:hi]))
            if excess[j] >= config.minimum_excess:
                indexes.add(j)
        proposed.extend((j, name, float(excess[j])) for j in sorted(indexes))
    candidates = []
    for j, origin, peak_strength in proposed:
        normal = np.array([-tangent[j, 1], tangent[j, 0]])
        reach = radius*config.normal_search_radius_fraction
        offsets = np.linspace(-reach, reach, 2*math.ceil(reach)+1)
        samples = []
        for offset in offsets:
            p = xy[j]+offset*normal
            m = _local_metrics(own, total, other, valid, p[0], p[1], tangent[j], radius, line_width, ignore)
            direct_score = .55*m['own_density']+.45*m['own_offstroke']
            foreign_score = .55*m['total_density']+.45*m['total_offstroke']
            rank = direct_score if origin == 'own' else foreign_score
            # A weak geometric preference prevents drifting along a broad cap.
            rank -= .035*abs(offset)/max(1., reach)
            samples.append((rank, float(offset), p, m))
        _, offset, p, m = max(samples, key=lambda item: item[0])
        if config.use_long_line_guard:
            m['own_compact_residual'] = float(_sample(maps['own_compact'], p[None, :])[0])
            m['total_compact_residual'] = float(_sample(maps['total_compact'], p[None, :])[0])
        own_compact = not config.use_long_line_guard or m['own_compact_residual'] >= config.own_compact_residual
        total_compact = not config.use_long_line_guard or m['total_compact_residual'] >= config.total_compact_residual
        if m['valid_fraction'] < config.minimum_valid_fraction:
            status, reason = 'rejected', 'insufficient_valid_disk_area'
        elif ignore is not None and m['evaluated_fraction'] < .50:
            status, reason = 'rejected', 'insufficient_unignored_disk_area'
        elif (m['own_density'] >= config.active_density and m['own_offstroke'] >= config.active_offstroke and own_compact
              and (ignore is None or m['enough_direct_evidence'])):
            status, reason = 'active', 'own_colour_disk_and_offstroke_body'
        elif (m['total_density'] >= config.suppressed_total_density and
              m['total_offstroke'] >= config.suppressed_offstroke and
              m['other_density'] >= config.suppressed_other_density and total_compact):
            status, reason = 'suppressed', 'possible_other_colour_occlusion_not_proven_marker'
        else:
            status = 'rejected'
            reason = 'long_strokes_explain_density' if config.use_long_line_guard and not total_compact else 'density_peak_without_sufficient_compact_body'
        candidates.append(dict(x=float(p[0]), y=float(p[1]), radius=float(radius),
            path_x=float(xy[j, 0]), path_y=float(xy[j, 1]), peak_index=j,
            arc_length=float(arc[j]), origin=origin+'_density_peak', offset=offset,
            tangent=tangent[j].tolist(), status=status, reason=reason,
            score=float(.55*m['own_density']+.45*m['own_offstroke'] if status=='active'
                        else .55*m['total_density']+.45*m['total_offstroke']),
            peak_excess=peak_strength, hidden_marker_proven=False,
            evidence_kind='own_colour_body' if status=='active' else 'occlusion_hypothesis' if status=='suppressed' else 'rejected',
            **m))
    # Collapse both peak streams and overlapping disk proposals, by evidence tier.
    priority = {'active': 2, 'suppressed': 1, 'rejected': 0}
    chosen = []
    for p in sorted(candidates, key=lambda q: (priority[q['status']], q['score']), reverse=True):
        near = next((q for q in chosen if np.hypot(p['x']-q['x'], p['y']-q['y']) <
                     config.duplicate_radius_fraction*radius), None)
        if near:
            continue
        chosen.append(p)
    chosen.sort(key=lambda q: (q['x'], q['y']))
    for k, p in enumerate(chosen, 1):
        p['id'] = f'D{k:03d}'
    return dict(path_xy=xy, profile=profile, candidates=chosen, config=asdict(config))


def disk_overlap_fraction(first, second):
    """Area of the target disk covered by the observed other disk, in [0, 1]."""
    r, R = float(first['radius']), float(second['radius'])
    distance = float(np.hypot(first['x']-second['x'], first['y']-second['y']))
    if distance >= r+R:
        return 0.
    if distance <= abs(r-R):
        return min(r, R)**2/r**2
    a = math.acos(np.clip((distance**2+r**2-R**2)/(2*distance*r), -1., 1.))
    b = math.acos(np.clip((distance**2+R**2-r**2)/(2*distance*R), -1., 1.))
    intersection = r*r*a+R*R*b-.5*math.sqrt(max(0., (-distance+r+R)*(distance+r-R)*(distance-r+R)*(distance+r+R)))
    return intersection/(math.pi*r*r)


def retain_observed_occluders(series, config=Config()):
    """Do not discard a possible hidden point merely because line opening
    explains its visible cover. An independently ACTIVE other-colour compact
    body may explain that cover. This never upgrades the hidden series to active
    and never lets suppressed/rejected candidates support one another.
    """
    output = deepcopy(series)
    active = [(s['id'], p) for s in output for p in s['candidates'] if p['status'] == 'active']
    for s in output:
        for p in s['candidates']:
            if p['status'] != 'rejected' or p['reason'] != 'long_strokes_explain_density':
                continue
            if (p['total_density'] < config.suppressed_total_density or
                p['total_offstroke'] < config.suppressed_offstroke or
                p['other_density'] < config.suppressed_other_density or
                p['valid_fraction'] < config.minimum_valid_fraction):
                continue
            covers = [(sid, q) for sid, q in active if sid != s['id'] and
                      disk_overlap_fraction(p, q) >= config.minimum_occluder_overlap]
            if covers:
                p['pre_occluder_status'] = p['status']
                p['pre_occluder_reason'] = p['reason']
                p.update(status='suppressed', reason='observed_other_marker_covers_target_path',
                         evidence_kind='occlusion_hypothesis', hidden_marker_proven=False,
                         observed_occluder_ids=[q['id'] for _, q in covers],
                         observed_occluder_series=[sid for sid, _ in covers],
                         observed_occluder_overlap=[disk_overlap_fraction(p, q) for _, q in covers])
    return output

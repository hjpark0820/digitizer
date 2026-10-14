"""Native source-raster boundary evidence for hollow, filled and concave marks.

The source alone chooses each ink/paper transition, normal, sample band and
coverage denominator. No class renderer, central-hole assumption, target-fit
selection, interpolation oversampling or fitted occluder is used here. A caller
must independently establish every pixel in ``occlusion``. Passing this pure
helper is evidence for a scene check, never authority to add a detection.
"""
from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np


VERSION = 'bw-partial-source-raster-profiles-v1'
SOURCE_PLAN_VERSION = 'native-ink-paper-transitions-v1'
MAXIMUM_PROFILES = 96
_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1),
               (-1, 0), (-1, -1), (0, -1), (1, -1))


@dataclass(frozen=True)
class _Defaults:
    minimum_visible_quality: float = .72
    minimum_visible_coverage: float = .25
    minimum_visible_profiles: int = 8
    minimum_supported_fraction: float = .65
    maximum_missing_paper: float = .18


@dataclass(frozen=True)
class _Plan:
    fingerprint: str
    anchors: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    expected: np.ndarray
    supported: np.ndarray
    regions: np.ndarray
    orientations: np.ndarray
    has_back_paper: np.ndarray
    native_boundary_pixels: int
    diameter: float


def _field(value, name, shape=None):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape, default=0) < 3 or not np.isfinite(array).all():
        raise ValueError(name + ' must be a finite two-dimensional source array')
    if shape is not None and array.shape != shape:
        raise ValueError(name + ' must have the same shape as the source')
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(name + ' must lie in [0, 1]')
    return array.astype(np.float32, copy=False)


def _readonly(array, dtype=None):
    result = np.asarray(array, dtype=dtype)
    result.flags.writeable = False
    return result


def _band_values(array, plan):
    return np.sum(array.ravel()[plan.indices] * plan.weights, axis=2)


def _source_plan(source, reliability, cache=None, work=None):
    """Freeze at most one profile per native boundary-ink pixel, before target access."""
    fingerprint = hashlib.sha256(str(source.shape).encode('ascii') + source.tobytes() +
                                 reliability.tobytes()).hexdigest()
    key = (SOURCE_PLAN_VERSION, fingerprint)
    if cache is not None and key in cache:
        if work is not None:
            work['source_plan_hits'] = work.get('source_plan_hits', 0) + 1
        return cache[key]
    peak = float(source.max())
    mask = source > max(.12, .25 * peak)
    height, width = source.shape
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    by, bx = np.nonzero(boundary)
    native_count = len(bx)
    if native_count > MAXIMUM_PROFILES:
        selected = np.linspace(0, native_count - 1, MAXIMUM_PROFILES).round().astype(int)
        by, bx = by[selected], bx[selected]
    center = np.array([(xx.min() + xx.max()) / 2., (yy.min() + yy.max()) / 2.])
    diameter = float(max(np.ptp(xx) + 1, np.ptp(yy) + 1))
    smooth = cv2.GaussianBlur(source, (3, 3), .7)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    # A local return to paper is optional: thin strokes have it, solid bodies
    # usually do not. The native source determines the distinction per normal.
    maximum_inward = min(12, max(3, int(math.ceil(.30 * diameter))))
    anchors, normals, bands, regions, orientations, backs = [], [], [], [], [], []
    for x, y in zip(bx, by):
        choices = []
        for dx, dy in _DIRECTIONS:
            px, py = int(x + dx), int(y + dy)
            if not (0 <= px < width and 0 <= py < height) or mask[py, px]:
                continue
            norm = math.hypot(dx, dy)
            # Direction ranking is exclusively source-based. Gradient resolves
            # corners; the paper value resolves flat/symmetric thin strokes.
            alignment = (-float(gx[y, x]) * dx - float(gy[y, x]) * dy) / norm
            choices.append((alignment - float(source[py, px]), dx, dy))
        if not choices:
            continue
        _, dx, dy = max(choices)
        outer = []
        for step in (1, 2):
            px, py = int(x + step * dx), int(y + step * dy)
            if not (0 <= px < width and 0 <= py < height) or mask[py, px]:
                break
            outer.append((px, py))
        ink, back = [(int(x), int(y))], []
        for step in range(1, maximum_inward + 1):
            px, py = int(x - step * dx), int(y - step * dy)
            if not (0 <= px < width and 0 <= py < height):
                break
            if mask[py, px]:
                # Bound local ink-band width without discarding the full-source
                # required-ink check performed separately below.
                if len(ink) < 4:
                    ink.append((px, py))
                continue
            back.append((px, py))
            qx, qy = px - dx, py - dy
            if 0 <= qx < width and 0 <= qy < height and not mask[qy, qx]:
                back.append((qx, qy))
            break
        norm = math.hypot(dx, dy)
        normal = (dx / norm, dy / norm)
        angle = math.atan2(normal[1], normal[0]) % math.pi
        orientation = int(round(angle / (math.pi / 4))) % 4
        region = int(x >= center[0]) + 2 * int(y >= center[1])
        anchors.append((int(x), int(y)))
        normals.append(normal)
        bands.append((outer, ink, back or outer))
        backs.append(bool(back))
        regions.append(region)
        orientations.append(orientation)
    if not anchors:
        return None
    indices = np.zeros((len(anchors), 3, 4), np.intp)
    weights = np.zeros(indices.shape, np.float32)
    for row, profile_bands in enumerate(bands):
        for band, points in enumerate(profile_bands):
            for column, (px, py) in enumerate(points):
                indices[row, band, column] = py * width + px
                weights[row, band, column] = 1. / len(points)
    expected = np.sum(source.ravel()[indices] * weights, axis=2)
    contrast = expected[:, 1] - np.maximum(expected[:, 0], expected[:, 2])
    reliable = np.all((reliability.ravel()[indices] > .99) | (weights == 0), axis=(1, 2))
    supported = (contrast >= .20) & (expected[:, 1] >= .25) & reliable
    plan = _Plan(fingerprint, _readonly(anchors, np.int32), _readonly(normals, np.float32),
                 _readonly(indices), _readonly(weights), _readonly(expected),
                 _readonly(supported), _readonly(regions, np.int32),
                 _readonly(orientations, np.int32), _readonly(backs, bool),
                 native_count, diameter)
    if cache is not None:
        if len(cache) >= 512:
            cache.clear()
        cache[key] = plan
    if work is not None:
        work['source_plan_builds'] = work.get('source_plan_builds', 0) + 1
    return plan


def profile_evidence(observed, marker, *, available=None, occlusion=None, config=None,
                     provisional=False, profile_cache=None, work=None, source_reliability=None):
    """Compare an immutable source boundary plan with visible native target pixels.

    ``provisional=True`` may withhold a profile whose source-paper bands contain
    unexplained dark ink. Missing expected ink never causes withholding. The
    final call derives visibility only from ``available`` and independent
    ``occlusion``. All quality denominators include missing-ink profiles.
    """
    cfg = config or _Defaults()
    obs = _field(observed, 'observed')
    source = _field(marker, 'marker', obs.shape)
    valid = np.ones_like(obs) if available is None else _field(available, 'available', obs.shape)
    cover = np.zeros_like(obs) if occlusion is None else _field(occlusion, 'occlusion', obs.shape)
    reliability = (np.ones_like(source) if source_reliability is None else
                   _field(source_reliability, 'source_reliability', source.shape))
    if work is not None:
        work['profile_calls'] = work.get('profile_calls', 0) + 1
        work['source_raster_profile_calls'] = work.get('source_raster_profile_calls', 0) + 1
    result = dict(version=VERSION, source_plan_version=SOURCE_PLAN_VERSION,
        profile_backend='source_raster', stage_a_passed=False,
        reason='insufficient_source_ink_paper_transitions', visible_fit_quality=0.,
        visible_coverage=0., visible_profiles=0, total_profiles=0, visible_sides=0,
        supported_profiles=0, supported_fraction=0., spatial_diversity=False,
        missing_paper_fraction=1., ink_covered_fraction=0., direct_rim_recall=0.,
        required_ink_visibility=0., required_ink_recall=0., profiles=[],
        source_pixels_unchanged=True, source_plan_uses_observed=False,
        uses_canonical_shape=False, unique_native_boundary_pixels=0,
        unique_native_profile_anchors=0, unknown_ink_profiles=0,
        provisional_fragment_gate=bool(provisional), evidence_only=True,
        supported_source_fraction=0.,
        source_cap_profiles=0, visible_cap_profiles=0, supported_cap_profiles=0,
        cap_fit_quality=0., cap_supported_fraction=0., cap_group_required=False,
        cap_group_passed=True,
        missing_paper_profiles=0, evaluated_before_occluder_fit=bool(provisional),
        unknown_counts_as_positive=False,
        visibility_policy='fixed_source_profiles_minus_independent_scene_coverage',
        selection_basis='source_native_ink_paper_transitions')
    plan = _source_plan(source, reliability, profile_cache, work)
    if plan is None:
        return result
    values = _band_values(obs, plan)
    required = plan.weights > 0
    visible = np.all((valid.ravel()[plan.indices] > .99) | ~required, axis=(1, 2))
    visible &= np.all((cover.ravel()[plan.indices] < .20) | ~required, axis=(1, 2))
    base_visible = visible.copy()
    paper_excess = np.maximum(values[:, 0] - plan.expected[:, 0],
                              values[:, 2] - plan.expected[:, 2])
    unknown = base_visible & plan.supported & (paper_excess > .20)
    if provisional:
        visible &= ~unknown
    source_contrast = plan.expected[:, 1] - np.maximum(plan.expected[:, 0], plan.expected[:, 2])
    contrast = values[:, 1] - np.maximum(values[:, 0], values[:, 2])
    contrast_quality = np.clip(contrast / np.maximum(source_contrast, .20), 0, 1)
    # Actual grayscale amplitude matters in addition to the transition sign.
    error = np.mean(np.abs(values - plan.expected), axis=1)
    raster_quality = np.clip(1. - error / np.maximum(source_contrast, .20), 0, 1)
    qualities = np.minimum(contrast_quality, raster_quality)
    measurable = visible & plan.supported
    supported = measurable & (contrast >= .20) & (values[:, 1] >= .25)
    supported &= qualities >= cfg.minimum_visible_quality
    count = int(measurable.sum())
    supported_count = int(supported.sum())
    quality = float(qualities[measurable].mean()) if count else 0.
    coverage = count / len(plan.anchors)
    fraction = supported_count / max(count, 1)
    orientation_counts = np.bincount(plan.orientations[supported], minlength=4)
    region_counts = np.bincount(plan.regions[supported], minlength=4)
    orientations = np.flatnonzero(orientation_counts >= 2)
    regions = np.flatnonzero(region_counts >= 2)
    points = plan.anchors[supported].astype(float)
    span = float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else 0.
    diverse = bool(len(orientations) >= 2 and len(regions) >= 2 and span >= .35 * plan.diameter)
    mask = source > max(.12, .25 * float(source.max()))
    # Legend connector overlaps can leave source ownership uncertain. Such
    # pixels are not guaranteed marker ink; reliable expected ink still has
    # the same missing-on-white contradiction, regardless of target fit.
    ink_mass = source * mask * valid * (reliability > .99)
    mass = max(float(ink_mass.sum()), 1.e-8)
    independent = ink_mass * (1. - cover)
    missing = float(np.sum(independent * np.clip((source - obs) / np.maximum(source, .05), 0, 1)) / mass)
    recall = float(np.sum(ink_mass * np.clip(obs / np.maximum(source, .05), 0, 1)) / mass)
    ink_visibility = float(independent.sum() / mass)
    visible_recall = float(np.sum(independent * np.clip(obs / np.maximum(source, .05), 0, 1)) /
                           max(float(independent.sum()), 1.e-8))
    minimum_profiles = 4 if provisional else cfg.minimum_visible_profiles
    supported_source_fraction = supported_count / len(plan.anchors)
    # A long inkward run identifies a native filled-body face or stroke end.
    # Long line crossings can match a plus/X's sides while continuing beyond
    # every source end. Averaging sparse end contradictions into many matching
    # sides would invent a compact marker. Preserve this source-defined group.
    source_caps = plan.supported & ~plan.has_back_paper
    visible_caps = source_caps & measurable
    supported_caps = source_caps & supported
    cap_count = int(source_caps.sum())
    visible_cap_count = int(visible_caps.sum())
    supported_cap_count = int(supported_caps.sum())
    cap_quality = float(qualities[visible_caps].mean()) if visible_cap_count else 0.
    cap_fraction = supported_cap_count / max(visible_cap_count, 1)
    cap_required = cap_count >= 2
    cap_passed = bool(not cap_required or (supported_cap_count >= 1 and
        cap_quality >= cfg.minimum_visible_quality and cap_fraction >= cfg.minimum_supported_fraction))
    # A source fragment may propose a center before an independent foreground
    # body explains white erasure. Keep every measured profile and all missing
    # ink in the diagnostics; only proposal admission has a fragment gate.
    # Final evidence still requires the full visible-quality and missing-ink
    # gates after the caller supplies independently justified occlusion.
    if provisional:
        passed = bool(supported_count >= minimum_profiles and diverse and
                      supported_source_fraction >= cfg.minimum_visible_coverage)
    else:
        passed = bool(count >= minimum_profiles and supported_count >= minimum_profiles and
                      coverage >= cfg.minimum_visible_coverage and diverse and
                      quality >= cfg.minimum_visible_quality and
                      fraction >= cfg.minimum_supported_fraction and
                      missing <= cfg.maximum_missing_paper and cap_passed)
    profiles = []
    width = source.shape[1]
    for row, anchor in enumerate(plan.anchors):
        coordinates = []
        for band in range(3):
            indices = plan.indices[row, band, required[row, band]]
            coordinates.append([[int(index % width), int(index // width)] for index in indices])
        profiles.append(dict(index=row, side=int(plan.orientations[row]),
            region=int(plan.regions[row]), orientation=int(plan.orientations[row]),
            anchor=anchor.tolist(), normal=plan.normals[row].tolist(),
            coordinates=coordinates, band_roles=['outward_paper', 'required_ink',
                'inkward_paper' if plan.has_back_paper[row] else 'outward_paper_repeat'],
            has_inkward_paper=bool(plan.has_back_paper[row]),
            visible=bool(visible[row]), independent_visible=bool(base_visible[row]),
            unknown_ink=bool(unknown[row]), source_supported=bool(plan.supported[row]),
            observed_contrast=float(contrast[row]), source_contrast=float(source_contrast[row]),
            quality=float(qualities[row]), supported=bool(supported[row]),
            observed=values[row].tolist(), expected=plan.expected[row].tolist()))
    missing_profiles = int(np.sum(measurable & (values[:, 1] < .30)))
    result.update(stage_a_passed=passed,
        reason='visible_source_profiles_supported' if passed else
            'missing_required_ink_on_paper' if missing > cfg.maximum_missing_paper else
            'source_cap_group_unsupported' if not provisional and not cap_passed else
            'insufficient_independent_profile_evidence',
        source_plan_fingerprint=plan.fingerprint, total_profiles=len(plan.anchors),
        source_supported_profiles=int(plan.supported.sum()),
        reliable_source_ink_fraction=float(np.sum(source * mask * (reliability > .99)) /
                                           max(float(np.sum(source * mask)), 1.e-8)),
        unique_native_boundary_pixels=plan.native_boundary_pixels,
        unique_native_profile_anchors=len(plan.anchors),
        visible_fit_quality=quality, visible_coverage=float(coverage),
        visible_profiles=count, supported_profiles=supported_count,
        supported_fraction=float(fraction), visible_sides=int(len(orientations)),
        supported_source_fraction=float(supported_source_fraction),
        source_cap_profiles=cap_count, visible_cap_profiles=visible_cap_count,
        supported_cap_profiles=supported_cap_count, cap_fit_quality=cap_quality,
        cap_supported_fraction=float(cap_fraction), cap_group_required=cap_required,
        cap_group_passed=cap_passed,
        visible_regions=int(len(regions)), side_support=orientation_counts.tolist(),
        region_support=region_counts.tolist(), supported_span_pixels=span,
        spatial_diversity=diverse, missing_paper_fraction=missing,
        ink_covered_fraction=float(np.sum(ink_mass * cover) / mass),
        direct_rim_recall=recall, required_ink_visibility=ink_visibility,
        required_ink_recall=visible_recall, unknown_ink_profiles=int(unknown.sum()),
        missing_paper_profiles=missing_profiles, profiles=profiles)
    if provisional:
        result['visibility_policy'] = 'fixed_source_profiles_with_unresolved_paper_band_ink_withheld'
        result['selection_basis'] = 'source_fixed_paper_bands_not_best_fitting_required_ink'
    return result

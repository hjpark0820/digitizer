"""Conservative filled-marker identity from measured circular brightness contours.

The source is an observed legend raster, not a rendered circle/square/triangle.
Every source sees the same target centre, radius trials and valid pixels.  The
third and fourth complex Fourier coefficients retain orientation: a triangle
and its inverted counterpart are not made equivalent by rotation fitting.
This module never proposes a centre, changes metadata, or promotes a missing
marker.  An accepted result is advisory source identity, not marker confidence.

The bounded default search uses 15 geometry trials and at most six sources.
No thresholds are calibrated probabilities.  Uncertain comparisons abstain;
the caller must preserve its existing identity when ``accepted`` is false.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from functools import lru_cache
import math
import numpy as np

VERSION = "filled-contour-identity-v46-v2-observed-fill"


@dataclass(frozen=True)
class Config:
    angles: int = 48
    radial_samples: int = 17
    radius_min: float = .25
    radius_max: float = 1.75
    informative_min: float = .75
    informative_max: float = 1.375
    center_offsets_xy: tuple = ((0., 0.), (-.5, 0.), (.5, 0.), (0., -.5), (0., .5))
    radius_factors: tuple = (.9, 1., 1.1)
    maximum_sources: int = 6
    maximum_trials: int = 81
    minimum_contrast: float = .12
    minimum_angular_coverage: float = .70
    minimum_quadrant_coverage: float = .50
    maximum_missing_gap_degrees: float = 90.
    maximum_fit_condition: float = 1e5
    minimum_usable_trial_fraction: float = .80
    minimum_winner_fraction: float = .80
    minimum_loss_margin: float = .0005
    maximum_primary_loss: float = .04
    minimum_source_separation: float = .0001
    minimum_distinctive_energy_explained: float = .50
    angular_loss_weight: float = .65
    third_loss_weight: float = .175
    fourth_loss_weight: float = .175
    minimum_relative_core_fill: float = .50
    minimum_source_core_fill: float = .50


def _validate_config(cfg):
    if not (16 <= cfg.angles <= 256 and 10 <= cfg.radial_samples <= 128):
        raise ValueError("polar sampling must use 16..256 angles and 10..128 radii")
    if not (0 < cfg.radius_min < cfg.informative_min < cfg.informative_max < cfg.radius_max):
        raise ValueError("informative annulus must lie inside the sampled positive radii")
    if not cfg.center_offsets_xy or not cfg.radius_factors:
        raise ValueError("nonempty centre offsets and radius factors required")
    if len(cfg.center_offsets_xy) * len(cfg.radius_factors) > cfg.maximum_trials:
        raise ValueError("configured shared geometry search exceeds maximum_trials")
    for offset in cfg.center_offsets_xy:
        if np.shape(offset) != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("each centre offset must be a finite x,y pair")
    if not all(math.isfinite(float(f)) and f > 0 for f in cfg.radius_factors):
        raise ValueError("radius factors must be finite and positive")
    weights = [cfg.angular_loss_weight, cfg.third_loss_weight, cfg.fourth_loss_weight]
    if not all(math.isfinite(w) and w >= 0 for w in weights) or not math.isclose(sum(weights), 1.):
        raise ValueError("primary loss weights must be nonnegative and sum to one")
    if cfg.maximum_sources < 2:
        raise ValueError("maximum_sources must be at least two")
    if not (0. < cfg.minimum_relative_core_fill <= 1. and 0. < cfg.minimum_source_core_fill <= 1.):
        raise ValueError("relative and source core-fill requirements must lie in (0,1]")


def _gray(image):
    a = np.asarray(image)
    if a.ndim != 2 or min(a.shape) < 2:
        raise ValueError("gray must be a two-dimensional image of at least 2x2")
    if not np.all(np.isfinite(a)) or np.min(a) < 0 or np.max(a) > 255:
        raise ValueError("gray must contain finite observed 0..255 values")
    # Keep an existing uint8/float array as-is; sampling casts only local pixels.
    return a


def _geometry(center_xy, reference_radius):
    center = np.asarray(center_xy, dtype=float)
    radius = float(reference_radius)
    if center.shape != (2,) or not np.all(np.isfinite(center)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("finite center_xy and positive finite reference_radius required")
    return center, radius


def _nuisance(mask, shape):
    if mask is None:
        return None
    a = np.asarray(mask)
    if a.shape != shape or not np.all(np.isfinite(a)) or np.min(a) < 0 or np.max(a) > 1:
        raise ValueError("nuisance must have image shape and finite 0..1 uncertainty values")
    return a


def _bilinear(a, xy):
    h, w = a.shape
    x, y = xy[..., 0], xy[..., 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    value = ((1-dx)*(1-dy)*a[y0, x0] + dx*(1-dy)*a[y0, x1]
             + (1-dx)*dy*a[y1, x0] + dx*dy*a[y1, x1])
    return value, valid


@lru_cache(maxsize=16)
def _basis(count):
    angles = np.arange(count)*2*np.pi/count
    basis = np.column_stack([np.ones(count)] +
                            [f(k*angles) for k in range(1, 5) for f in (np.cos, np.sin)])
    basis.setflags(write=False)
    return basis


def _harmonics(values, weights, cfg):
    basis = _basis(cfg.angles)
    n = len(values)
    coefficients = np.zeros((n, 9), float)
    valid = np.zeros(n, bool)
    ranks = np.zeros(n, int)
    condition = np.zeros(n, float)
    full = np.all(weights == 1., axis=1)
    # Exact equal-angle least-squares projection when nothing is masked.
    coefficients[full] = values[full] @ basis / cfg.angles
    coefficients[full, 1:] *= 2.
    valid[full], ranks[full], condition[full] = True, 9, math.sqrt(2.)
    for i in np.flatnonzero(~full):
        root = np.sqrt(np.clip(weights[i], 0., 1.))
        if np.count_nonzero(root) < 9:
            continue
        design = basis * root[:, None]
        coef, _, rank, singular = np.linalg.lstsq(design, values[i]*root, rcond=None)
        cond = singular[0]/singular[-1] if len(singular) and singular[-1] > 0 else math.inf
        ranks[i], condition[i] = rank, cond
        if rank == 9 and cond <= cfg.maximum_fit_condition:
            coefficients[i], valid[i] = coef, True
    pairs = coefficients[:, 1:].reshape(n, 4, 2)
    return dict(coefficients=coefficients, amplitude=np.linalg.norm(pairs, axis=2),
                phase_rad=np.arctan2(pairs[:, :, 1], pairs[:, :, 0]),
                fit_valid=valid, rank=ranks, condition_number=condition,
                coefficient_order=("intercept", "cos1", "sin1", "cos2", "sin2", "cos3", "sin3", "cos4", "sin4"),
                phase_convention="atan2(sin_k,cos_k); physical maximum phase/k modulo 2pi/k")


def _maximum_gap(mask):
    if np.all(mask):
        return 0
    if not np.any(mask):
        return len(mask)
    run = best = 0
    for missing in np.tile(~mask, 2):
        run = run + 1 if missing else 0
        best = max(best, run)
    return min(best, len(mask))


def _coverage(weights, xy, cfg):
    quadrants = np.array_split(np.arange(cfg.angles), 4)
    angular = np.mean(weights, axis=1)
    quadrant = np.column_stack([np.mean(weights[:, q], axis=1) for q in quadrants])
    gap = np.array([360.*_maximum_gap(w >= .5)/cfg.angles for w in weights])
    # Distinct native cells keep a tiny raster from masquerading as many
    # independent contour samples. This is a guard, not a sample-size estimate.
    cells = np.floor(xy).astype(np.int64)
    unique = np.array([len(set(map(tuple, c[w >= .5]))) for c, w in zip(cells, weights)])
    valid = ((angular >= cfg.minimum_angular_coverage) &
             (np.min(quadrant, axis=1) >= cfg.minimum_quadrant_coverage) &
             (gap <= cfg.maximum_missing_gap_degrees) & (unique >= 9))
    return dict(angular_fraction=angular, quadrant_fraction=quadrant,
                maximum_missing_gap_degrees=gap, distinct_native_sample_cells=unique,
                ring_valid=valid)


def _observed_fill_profile(a, center, radius, nuisance, cfg, rn, directions,
                           darkness, weights, paper, background):
    """Independent fill evidence, without the inner-quantile ink normalization.

    A pale filled body can still have a uniformly dark *relative* interior.
    An open body has a brighter central profile than its observed rim. Two
    central radii restore the domain missing from the shape-only annulus.
    Directional lower quartiles prevent a narrow dark connector from supplying
    all apparent fill. No shape label, rendered glyph or inferred mask is used.
    """
    extra_radii = np.array((0., .125))
    xy = center[None, None, :]+radius*extra_radii[:, None, None]*directions[None, :, :]
    gray, inside = _bilinear(a, xy)
    extra_weights = inside.astype(float)
    if nuisance is not None:
        extra_weights *= 1.-_bilinear(nuisance, xy)[0]
    extra_darkness = np.clip((paper-gray)/255., 0., 1.)
    core = rn <= .4375
    core_darkness = np.concatenate((extra_darkness, darkness[core]), axis=0)
    core_weights = np.concatenate((extra_weights, weights[core]), axis=0)
    tone = (rn >= .4375) & (rn <= 1.)

    def quantiles(values, valid_weights, percentile):
        good = valid_weights >= .5
        enough = good.mean(axis=1) >= cfg.minimum_angular_coverage
        # Fast path is the ordinary unmasked raster; masked evidence retains
        # exactly its measured valid pixels rather than filling holes white.
        if np.all(good):
            measured = np.percentile(values, percentile, axis=1)
        else:
            measured = np.array([np.percentile(v[g], percentile) if np.any(g) else 0.
                                 for v, g in zip(values, good)])
        return measured, enough

    levels, core_good = quantiles(core_darkness, core_weights, 25)
    tone_levels, tone_good = quantiles(darkness[tone], weights[tone], 50)
    contrast = max(float(np.max(tone_levels[tone_good]))-background, 0.) if np.any(tone_good) else 0.
    relative = np.clip((levels-background)/max(contrast, .02), 0., 1.)
    observed = bool(np.all(core_good) and np.any(tone_good) and contrast >= cfg.minimum_contrast)
    return dict(version="observed-central-profile-v1", observed=observed,
                core_support=float(np.median(relative[core_good])) if np.any(core_good) else None,
                core_radii_normalized=np.concatenate((extra_radii, rn[core])),
                core_darkness_lower_quartile=levels, core_relative_support=relative,
                core_valid=core_good, tone_radii_normalized=rn[tone],
                tone_darkness_median=tone_levels, tone_valid=tone_good,
                tone_contrast=float(contrast), background_darkness=float(background),
                policy="source-relative raw brightness; central and inner profile; no metadata inference")


def _fill_compatibility(target, source, cfg):
    target = target.get("observed_fill_profile")
    source = source.get("observed_fill_profile")
    observed = bool(target and source and target.get("observed") and source.get("observed"))
    source_support = source.get("core_support") if source else None
    target_support = target.get("core_support") if target else None
    source_resolved = bool(observed and source_support >= cfg.minimum_source_core_fill)
    ratio = float(target_support/max(source_support, 1e-12)) if source_resolved else None
    return dict(observed=observed, source_resolved=source_resolved,
                source_core_support=source_support, target_core_support=target_support,
                relative_core_support=ratio,
                passed=bool(source_resolved and ratio >= cfg.minimum_relative_core_fill))


def _aggregate_fill_evidence(trials, sid, cfg):
    records = [t["source_fill_evidence"][sid] for t in trials]
    observed = [r for r in records if r["source_resolved"]]
    enough = bool(records and len(observed) >= cfg.minimum_usable_trial_fraction*len(records))
    ratio = float(np.median([r["relative_core_support"] for r in observed])) if observed else None
    passed = bool(enough and ratio >= cfg.minimum_relative_core_fill)
    reason = ("source_relative_core_fill_consistent" if passed else
              "source_relative_core_fill_unresolved" if not enough else
              "source_relative_core_fill_contradiction")
    return dict(passed=passed, reason=reason, resolved_trials=len(observed), total_trials=len(records),
                median_relative_core_support=ratio, required_relative_core_support=cfg.minimum_relative_core_fill,
                median_source_core_support=float(np.median([r["source_core_support"] for r in observed])) if observed else None,
                median_target_core_support=float(np.median([r["target_core_support"] for r in observed])) if observed else None,
                per_trial=records, metadata_changed=False)


def _sample(a, center, radius, nuisance, cfg):
    angles = np.arange(cfg.angles)*2*np.pi/cfg.angles
    rn = np.linspace(cfg.radius_min, cfg.radius_max, cfg.radial_samples)
    rp = rn*radius
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    xy = center[None, None, :] + rp[:, None, None]*directions[None, :, :]
    gray, inside = _bilinear(a, xy)
    weights = inside.astype(float)
    nuisance_polar = np.zeros_like(weights)
    if nuisance is not None:
        nuisance_polar, _ = _bilinear(nuisance, xy)
        weights *= 1.-nuisance_polar
    available = gray[inside]
    paper = float(np.percentile(available, 95)) if available.size else 255.
    darkness = np.clip((paper-gray)/255., 0., 1.)
    inner = (rn[:, None] <= .65) & (weights >= .5)
    outer = (rn[:, None] >= 1.45) & (weights >= .5)
    inner_level = float(np.percentile(darkness[inner], 75)) if np.any(inner) else 0.
    background = float(np.percentile(darkness[outer], 25)) if np.any(outer) else 0.
    contrast = max(inner_level-background, 0.)
    ink = np.clip((darkness-background)/max(contrast, .02), 0., 1.)
    boundary = np.zeros(cfg.angles, float)
    boundary_valid = np.zeros(cfg.angles, bool)
    crossing = ((ink[:-1] >= .5) & (ink[1:] < .5) &
                (np.minimum(weights[:-1], weights[1:]) >= .5) &
                (rn[1:, None] >= .45) & (rn[:-1, None] <= 1.65))
    for j in range(cfg.angles):
        indices = np.flatnonzero(crossing[:, j])
        if len(indices):
            i = indices[0]
            t = (ink[i, j]-.5)/max(ink[i, j]-ink[i+1, j], 1e-12)
            boundary[j] = rp[i]+t*(rp[i+1]-rp[i])
            boundary_valid[j] = contrast >= cfg.minimum_contrast
    measured = float(np.median(boundary[boundary_valid])) if np.mean(boundary_valid) >= .50 else None
    fill_profile = _observed_fill_profile(a, center, radius, nuisance, cfg, rn, directions,
                                         darkness, weights, paper, background)
    return dict(center_xy=center.copy(), reference_radius_px=float(radius),
                angles_rad=angles, radii_normalized=rn, radii_px=rp, sample_xy=xy,
                gray_polar=gray, darkness_polar=darkness, normalized_ink=ink,
                valid_weight=weights, nuisance_polar=nuisance_polar, paper_gray=paper,
                measured_inner_darkness=inner_level, outer_background_darkness=background,
                measured_contrast=contrast, contrast_valid=contrast >= cfg.minimum_contrast,
                inner_ink_median=float(np.median(ink[inner])) if np.any(inner) else 0.,
                boundary_radius_px=boundary, boundary_radius_valid=boundary_valid,
                median_boundary_radius_px=measured,
                boundary_missing_fraction=float(1.-np.mean(boundary_valid)),
                radial_mean_ink=np.sum(ink*weights, axis=1)/np.maximum(weights.sum(axis=1), 1e-12),
                coverage=_coverage(weights, xy, cfg), harmonics=_harmonics(ink, weights, cfg),
                representative_ring_index=int(np.argmin(abs(rn-1.))),
                informative_ring_mask=(rn >= cfg.informative_min) & (rn <= cfg.informative_max),
                angle_convention="radians; zero right, positive toward image-down",
                array_axes="radius,angle", observed_fill_profile=fill_profile)


def sample_polar(gray, center_xy, reference_radius, nuisance=None, config=None):
    """Measure raw 0..255 gray on circular contours, without rotating a source.

    ``nuisance`` is optional pre-established 0..1 uncertainty. Missing data is
    weighted out, never replaced by white or inferred from source mismatch.
    """
    cfg = config or Config()
    _validate_config(cfg)
    a = _gray(gray)
    center, radius = _geometry(center_xy, reference_radius)
    return _sample(a, center, radius, _nuisance(nuisance, a.shape), cfg)


def _radius_record(signature, reference_radius):
    measured = signature["median_boundary_radius_px"]
    refined = float(np.clip(measured, .65*reference_radius, 1.35*reference_radius)) if measured is not None else float(reference_radius)
    return dict(radius_px=refined, reference_radius_px=refined, input_reference_radius_px=float(reference_radius),
                measured_boundary_radius_px=measured,
                radius_method="measured_median_half_intensity_boundary" if measured is not None else "unchanged_initial_radius_no_reliable_boundary",
                radius_refinement_clipped=bool(measured is not None and refined != measured),
                boundary_missing_fraction=signature["boundary_missing_fraction"])


def estimate_radius(gray, center_xy, reference_radius, nuisance=None, config=None):
    """Use the same bounded, source-independent radius rule for either image.

    Median first outward half-ink crossings preserve non-circular shapes; no
    circle is fitted. A line continuing beyond the window supplies no edge.
    """
    signature = sample_polar(gray, center_xy, reference_radius, nuisance, config)
    return _radius_record(signature, float(reference_radius))


def build_source(sid, gray, center_xy, reference_radius, nuisance=None, metadata=None, config=None):
    """Prepare an observed legend source once; unknown metadata stays unknown."""
    cfg = config or Config()
    _validate_config(cfg)
    a = _gray(gray)
    center, radius = _geometry(center_xy, reference_radius)
    mask = _nuisance(nuisance, a.shape)
    initial = _sample(a, center, radius, mask, cfg)
    radius_info = _radius_record(initial, radius)
    signature = _sample(a, center, radius_info["reference_radius_px"], mask, cfg)
    return dict(sid=str(sid), version=VERSION, center_xy=center.tolist(),
                metadata=dict(metadata or {}), signature=signature,
                radius_diagnostics=radius_info,
                reference_radius_px=radius_info["reference_radius_px"],
                input_reference_radius_px=radius, configuration=asdict(cfg),
                class_policy="raw source identity; shape and fill metadata unchanged")


def _weighted_mean(a, w):
    return np.sum(a*w, axis=1)/np.maximum(w.sum(axis=1), 1e-12)


def _features(signature, weights, cfg):
    ink = signature["normalized_ink"]
    mean = _weighted_mean(ink, weights)
    h = _harmonics(ink, weights, cfg)
    return dict(ink=ink, angular=ink-mean[:, None], radial_mean=mean,
                coefficients=h["coefficients"], fit_valid=h["fit_valid"])


def _loss(a, b, weights, use, cfg):
    rings = use & a["fit_valid"] & b["fit_valid"]
    if not np.any(rings):
        return None
    angular = float(np.mean(_weighted_mean((a["angular"]-b["angular"])**2, weights)[rings]))
    c3 = float(np.mean(.5*np.sum((a["coefficients"][:, 5:7]-b["coefficients"][:, 5:7])**2, axis=1)[rings]))
    c4 = float(np.mean(.5*np.sum((a["coefficients"][:, 7:9]-b["coefficients"][:, 7:9])**2, axis=1)[rings]))
    full = float(np.mean(_weighted_mean((a["ink"]-b["ink"])**2, weights)[rings]))
    energy = float(np.mean(.5*np.sum(a["coefficients"][:, 5:9]**2, axis=1)[rings]))
    source_energy = float(np.mean(.5*np.sum(b["coefficients"][:, 5:9]**2, axis=1)[rings]))
    excess_strength = max(math.sqrt(energy)-math.sqrt(source_energy), 0.)**2
    return dict(primary=cfg.angular_loss_weight*angular+cfg.third_loss_weight*c3+cfg.fourth_loss_weight*c4,
                angular_residual=angular, third_harmonic=c3, fourth_harmonic=c4,
                distinctive_target_energy=energy,
                distinctive_source_energy=source_energy,
                distinctive_excess_strength_energy=excess_strength,
                distinctive_strength_explained=(1.-excess_strength/energy) if energy > 1e-12 else None,
                distinctive_energy_explained=(1.-(c3+c4)/energy) if energy > 1e-12 else None,
                full_polar=full, radial_mean=float(np.mean((a["radial_mean"][rings]-b["radial_mean"][rings])**2)),
                informative_ring_count=int(np.count_nonzero(rings)),
                fit_valid=bool(np.all(a["fit_valid"][use]) and np.all(b["fit_valid"][use])))


def _abstention(reason, cfg, total=0):
    return dict(version=VERSION, accepted=False, uncertain=True, winner=None, reason=reason,
                scores={}, usable_trials=0, total_trials=total, robustness_fraction=0.,
                margin=None, margin_q10=None, winning_metadata=None, configuration=asdict(cfg))


def compare_sources(gray, center_xy, target_radius, sources, target_nuisance=None, config=None, diagnostics=False, *, _trial_observer=None):
    """Compare raw-source identities at identical target geometry; never mutate.

    ``target_radius`` is a single shared radius, preferably ``estimate_radius``
    output. Do not pass a different target radius for each source. ``accepted``
    must be checked before using ``winner``. Losing absolute/robustness guards
    is abstention, not a request to change detection thresholds or labels.
    The optional private observer receives read-only feature views and a copied
    trial record. It is for local evidence reuse, never score modification.
    """
    cfg = config or Config()
    _validate_config(cfg)
    models = list(sources.values()) if isinstance(sources, dict) else list(sources)
    if len(models) < 2:
        return _abstention("fewer_than_two_sources", cfg)
    if len(models) > cfg.maximum_sources:
        return _abstention("source_budget_exceeded_no_sources_silently_dropped", cfg)
    sids = [str(m["sid"]) for m in models]
    if len(set(sids)) != len(sids):
        raise ValueError("source identities must be distinct")
    a = _gray(gray)
    center, radius = _geometry(center_xy, target_radius)
    mask = _nuisance(target_nuisance, a.shape)
    nominal = _sample(a, center, radius, mask, cfg)
    signatures = [m["signature"] for m in models]
    for source in signatures:
        if not np.array_equal(source["radii_normalized"], nominal["radii_normalized"]) or not np.array_equal(source["angles_rad"], nominal["angles_rad"]):
            raise ValueError("source and target polar sampling configurations must agree")
    use = nominal["informative_ring_mask"]
    source_weights = np.minimum.reduce([s["valid_weight"] for s in signatures])
    source_features = [_features(s, source_weights, cfg) for s in signatures]
    separations = []
    for i, f in enumerate(source_features):
        for other in source_features[i+1:]:
            loss = _loss(f, other, source_weights, use, cfg)
            separations.append(loss["primary"] if loss else 0.)
    separation = min(separations)
    source_coverage = [_coverage(source_weights, s["sample_xy"], cfg) for s in signatures]
    source_coverage_good = all(np.all(c["ring_valid"][use]) for c in source_coverage)
    source_contrast_good = all(s["contrast_valid"] for s in signatures)
    source_inner_good = all(s["inner_ink_median"] >= .5 for s in signatures)
    trials = []
    for dx, dy in cfg.center_offsets_xy:
        for factor in cfg.radius_factors:
            trial_center = center+np.array([dx, dy])
            target = _sample(a, trial_center, radius*factor, mask, cfg)
            common = np.minimum(source_weights, target["valid_weight"])
            if np.array_equal(common, source_weights):
                features = source_features
                coverage_sources = source_coverage_good
            else:
                features = [_features(s, common, cfg) for s in signatures]
                coverage_sources = all(np.all(_coverage(common, s["sample_xy"], cfg)["ring_valid"][use]) for s in signatures)
            coverage = _coverage(common, target["sample_xy"], cfg)
            ft = _features(target, common, cfg)
            scores = {sid: _loss(ft, fs, common, use, cfg) for sid, fs in zip(sids, features)}
            gates = dict(common_angular_coverage=bool(coverage_sources and np.all(coverage["ring_valid"][use])),
                         measured_contrast=bool(source_contrast_good and target["contrast_valid"]),
                         inner_ink_support=bool(source_inner_good and target["inner_ink_median"] >= .5),
                         full_rank_harmonic_fit=all(v is not None and v["fit_valid"] for v in scores.values()))
            trial = dict(index=len(trials), center_xy=trial_center.tolist(),
                         center_offset_px=[float(dx), float(dy)], radius_factor=float(factor),
                         radius_px=float(radius*factor), usable=all(gates.values()), gates=gates,
                         scores=scores,
                         source_fill_evidence={sid: _fill_compatibility(target, source, cfg)
                                               for sid, source in zip(sids, signatures)})
            trials.append(trial)
            if _trial_observer is not None:
                def readonly(value):
                    view = value.view()
                    view.setflags(write=False)
                    return view
                def feature_view(feature):
                    return {key: readonly(value) if isinstance(value, np.ndarray) else value
                            for key, value in feature.items()}
                _trial_observer(dict(target_features=feature_view(ft),
                                     source_features=[feature_view(f) for f in features],
                                     common_weights=readonly(common), informative_ring_mask=readonly(use),
                                     source_ids=tuple(sids), trial=copy.deepcopy(trial)))
    usable = [t for t in trials if t["usable"]]
    if not usable:
        result = _abstention("no_usable_common_geometry_trials", cfg, len(trials))
    else:
        median = {sid: float(np.median([t["scores"][sid]["primary"] for t in usable])) for sid in sids}
        ordered = sorted(sids, key=lambda sid: (median[sid], sid))
        winner = ordered[0]
        margins = [min(t["scores"][sid]["primary"] for sid in sids if sid != winner)-t["scores"][winner]["primary"] for t in usable]
        fraction = float(np.mean(np.array(margins) > 0.))
        q10 = float(np.quantile(margins, .10))
        reasons = []
        if len(usable) < cfg.minimum_usable_trial_fraction*len(trials):
            reasons.append("insufficient_usable_geometry_trials")
        if separation < cfg.minimum_source_separation:
            reasons.append("raw_sources_not_separable_in_this_feature")
        if fraction < cfg.minimum_winner_fraction:
            reasons.append("winner_changes_under_center_or_scale_perturbation")
        if q10 <= cfg.minimum_loss_margin:
            reasons.append("small_or_reversing_lower_decile_margin")
        if median[winner] > cfg.maximum_primary_loss:
            reasons.append("poor_absolute_source_match")
        # A weak-feature source must not win merely by being less wrong than
        # a strong but oppositely phased rival. This ONE-SIDED guard asks whether
        # target harmonic strength is absent from the source. Complex phase is
        # already scored by the primary loss; imposing the same phase residual
        # again as a relative veto needlessly double-penalizes pixel aliasing.
        # A source stronger than the target is not penalized by this guard;
        # angular/phase mismatch still matters in the unchanged primary loss.
        # The existing evidence floor defines whether strength is measurable.
        resolved = [t["scores"][winner] for t in usable
                    if t["scores"][winner]["distinctive_target_energy"] >= cfg.minimum_source_separation]
        explained = (float(np.median([s["distinctive_energy_explained"] for s in resolved]))
                     if len(resolved) >= cfg.minimum_usable_trial_fraction*len(usable) else None)
        strength_explained = (float(np.median([s["distinctive_strength_explained"] for s in resolved]))
                              if len(resolved) >= cfg.minimum_usable_trial_fraction*len(usable) else None)
        if strength_explained is not None and strength_explained < cfg.minimum_distinctive_energy_explained:
            reasons.append("distinctive_angular_strength_exceeds_winning_source")
        fill_evidence = {sid: _aggregate_fill_evidence(usable, sid, cfg) for sid in sids}
        if not fill_evidence[winner]["passed"]:
            reasons.append(fill_evidence[winner]["reason"])
        result = dict(version=VERSION, accepted=not reasons, uncertain=bool(reasons), winner=winner,
                      reason=";".join(reasons) if reasons else "stable_raw_source_identity",
                      scores=median, usable_trials=len(usable), total_trials=len(trials),
                      robustness_fraction=fraction, margin=median[ordered[1]]-median[winner], margin_q10=q10,
                      distinctive_energy_explained=explained, resolved_distinctive_trials=len(resolved),
                      distinctive_strength_explained=strength_explained,
                      phase_residual_guard_would_reject=bool(explained is not None and explained < cfg.minimum_distinctive_energy_explained),
                      median_distinctive_target_energy=(float(np.median([s["distinctive_target_energy"] for s in resolved])) if resolved else None),
                      median_distinctive_source_energy=(float(np.median([s["distinctive_source_energy"] for s in resolved])) if resolved else None),
                      median_distinctive_excess_strength_energy=(float(np.median([s["distinctive_excess_strength_energy"] for s in resolved])) if resolved else None),
                      median_distinctive_residual=(float(np.median([s["third_harmonic"]+s["fourth_harmonic"] for s in resolved])) if resolved else None),
                      winning_metadata=dict(models[sids.index(winner)].get("metadata", {})),
                      configuration=asdict(cfg), source_relative_fill_evidence=fill_evidence,
                      winning_fill_evidence=fill_evidence[winner])
    result.update(source_ids=sids, input_center_xy=center.tolist(), target_radius_px=radius,
                  source_feature_separation=float(separation),
                  primary_formula="0.65 angular-pattern MSE + 0.175 complex-C3 MSE + 0.175 complex-C4 MSE (default)",
                  geometry_policy="identical target trials and common observation weights for every source; median aggregation",
                  strength_guard_formula="1 - max(sqrt(target_C3_C4_energy)-sqrt(source_C3_C4_energy),0)^2 / target_C3_C4_energy; phase remains in primary loss",
                  classification_policy="advisory source identity only; no coordinates or source metadata changed",
                  fill_policy="independent original-brightness core profile versus observed source; fill disagreement only vetoes identity and never relabels a source",
                  uncertainty_note="deterministic acceptance guards, not calibrated confidence")
    if diagnostics:
        result.update(trials=trials, target_nominal=nominal,
                      source_signatures={sid: s for sid, s in zip(sids, signatures)})
    return result

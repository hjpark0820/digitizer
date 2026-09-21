"""Observed single-series palette with neutral-ink and spatial safeguards.

The base direction estimator is reused without changing marker evidence.
Neutral/colour mixtures are not counted as equally reliable independent
palettes. Their actual pixels remain in the image and in downstream evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from . import palette_base as v1

evidence = v1.evidence


@dataclass(frozen=True)
class PaletteConfig(v1.PaletteConfig):
    min_neutral_angle_degrees: float = 12.
    min_relative_chroma: float = .30
    neutral_hue_tolerance_degrees: float = 25.
    minimum_component_pixels: int = 6
    minimum_coherent_fraction: float = .65
    minimum_x_span_fraction: float = .25
    minimum_x_bins: int = 3


def _colour_fields(crop, paper):
    delta = np.asarray(paper, np.float32)-crop.astype(np.float32)
    magnitude = np.linalg.norm(delta, axis=2)
    unit = delta/np.maximum(magnitude[..., None], 1.)
    neutral_cosine = np.clip(unit.sum(axis=2)/math.sqrt(3.), -1., 1.)
    neutral_angle = np.degrees(np.arccos(neutral_cosine))
    relative_chroma = np.ptp(delta, axis=2)/np.maximum(magnitude, 1.)
    chromatic_part = delta-delta.mean(axis=2, keepdims=True)
    hue = chromatic_part/np.maximum(np.linalg.norm(chromatic_part, axis=2, keepdims=True), 1.)
    return magnitude, neutral_angle, relative_chroma, hue


def _spatial_support(mask, cfg):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    eligible = np.flatnonzero(areas >= cfg.minimum_component_pixels)+1
    coherent = np.isin(labels, eligible) & mask
    ys, xs = np.nonzero(coherent)
    width = mask.shape[1]
    if len(xs):
        lo, hi = np.percentile(xs, [5, 95])
        span = float((hi-lo+1)/width)
        bins = np.minimum(7, 8*xs//width)
        occupied = int((np.bincount(bins, minlength=8) >= cfg.minimum_component_pixels).sum())
    else:
        span, occupied = 0., 0
    fraction = float(len(xs)/max(1, int(mask.sum())))
    sufficient = (fraction >= cfg.minimum_coherent_fraction and
                  span >= cfg.minimum_x_span_fraction and occupied >= cfg.minimum_x_bins)
    return dict(component_count=count-1, coherent_component_count=len(eligible),
                coherent_pixels=int(len(xs)), coherent_fraction=fraction,
                central_90_percent_x_span_fraction=span, x_bins_supported=occupied,
                sufficient_for_single_series_palette=bool(sufficient)), coherent


def estimate_plot_palette(image_bgr, plot_box, *, valid=None, allow_achromatic=False,
                          config=PaletteConfig()):
    """Same public contract as palette.py; source_box is never an exclusion.

    Reliable chromatic cores must differ from neutral ink both in angle and
    relative channel contrast. The original 75% dominant/20% rival tests apply
    within those cores. A separate spatial test prevents narrowly concentrated
    coloured guides from automatically becoming a whole-series palette. This
    cannot identify an arbitrary coloured annotation or solve mixed-ink series.
    """
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image_bgr must be uint8 HxWx3 BGR pixels')
    if not isinstance(config, PaletteConfig):
        raise TypeError('config must be PaletteConfig')
    cfg = config
    positive = [cfg.min_contrast, cfg.min_chroma, cfg.direction_degrees,
                cfg.direction_bin_width, cfg.min_neutral_angle_degrees,
                cfg.min_relative_chroma, cfg.neutral_hue_tolerance_degrees]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError('Palette configuration values must be positive and finite')
    if not (cfg.direction_degrees < 90 and cfg.min_neutral_angle_degrees < 90 and
            cfg.direction_bin_width < 1 and cfg.neutral_hue_tolerance_degrees < 90 and
            0 < cfg.dominant_fraction <= 1 and 0 < cfg.substantial_rival_fraction <= 1 and
            0 < cfg.minimum_coherent_fraction <= 1 and 0 < cfg.minimum_x_span_fraction <= 1 and
            1 <= cfg.minimum_x_bins <= 8 and cfg.minimum_component_pixels >= 1 and
            cfg.minimum_cluster_pixels >= 1 and cfg.sample_candidates >= 1):
        raise ValueError('Invalid palette configuration')
    plot = evidence._box(plot_box, image.shape)
    x0, y0, x1, y1 = plot
    crop = image[y0:y1, x0:x1]
    allowed = np.ones(crop.shape[:2], bool) if valid is None else np.asarray(valid, bool).copy()
    if allowed.shape != crop.shape[:2]:
        raise ValueError('valid must be plot-local spatial permission')
    result = dict(status='unavailable', model=None, source_box=None,
        model_source='plot_observed', source_kind='palette_sample_not_legend',
        legend_box=None, exclusion_boxes=[], plot_box=plot, config=asdict(cfg),
        diagnostic_only=False, clusters=[], method='observed_neutral_separated_spatial_palette_v2',
        warnings=['Known-single-series scope is supplied, not inferred by this helper.',
                  'Palette source_box is an actual colour sample, not a marker, legend, or invalid region.',
                  'Near-neutral ink is retained but not forced into the selected colour model.',
                  'Spatial support is a conservative palette-quality gate, not proof that coloured ink is a data curve.',
                  'A mixed gray-curve/coloured-marker or guide system is not solved by this single-colour model.'])
    if not allowed.any():
        result['status'] = 'no_valid_plot_pixels'
        return result
    paper = evidence._paper(crop[allowed].reshape(-1, 1, 3))
    magnitude, angles, relative, hues = _colour_fields(crop, paper)
    ink = (magnitude >= cfg.min_contrast) & allowed
    result['paper_bgr'] = paper
    if not ink.any():
        result['status'] = 'no_distinguishable_ink'
        return result
    chromatic = ink & (np.ptp(crop.astype(np.float32), axis=2) >= cfg.min_chroma)
    core = chromatic & (angles >= cfg.min_neutral_angle_degrees) & (relative >= cfg.min_relative_chroma)
    near_neutral = chromatic & ~core
    result['ink_quality'] = dict(ink_pixels=int(ink.sum()),
        absolute_chromatic_pixels=int(chromatic.sum()), reliable_chromatic_core_pixels=int(core.sum()),
        near_neutral_chromatic_pixels=int(near_neutral.sum()),
        achromatic_pixels=int((ink & ~chromatic).sum()))
    if core.sum() < cfg.minimum_cluster_pixels:
        if allow_achromatic and chromatic.sum() < cfg.minimum_cluster_pixels:
            base_cfg = v1.PaletteConfig(**{k: getattr(cfg, k) for k in v1.PaletteConfig.__dataclass_fields__})
            fallback = v1.estimate_plot_palette(image, plot, valid=allowed,
                allow_achromatic=True, config=base_cfg)
            fallback.update(method=result['method'], config=result['config'], ink_quality=result['ink_quality'])
            fallback['warnings'].extend(result['warnings'])
            return fallback
        result.update(status='no_reliable_chromatic_core',
                      reason='Too little colour reliably separated from neutral ink. No automatic grayscale fallback.')
        return result
    ys, xs = np.nonzero(core)
    pixels = crop[core]
    labels, clusters = v1._directions(pixels, paper, cfg)
    for item in clusters:
        own = labels == item['cluster_id']
        item.update(fraction=float(own.mean()),
                    median_neutral_angle_degrees=float(np.median(angles[core][own])),
                    median_relative_chroma=float(np.median(relative[core][own])),
                    box_source=[int(xs[own].min()+x0), int(ys[own].min()+y0),
                                int(xs[own].max()+x0+1), int(ys[own].max()+y0+1)])
    clusters.sort(key=lambda c: (-c['pixel_count'], c['cluster_id']))
    result['clusters'] = clusters
    substantial = [c for c in clusters if c['pixel_count'] >= cfg.minimum_cluster_pixels]
    if not substantial:
        result['status'] = 'insufficient_colour_samples'
        return result
    winner = substantial[0]
    if winner['fraction'] < cfg.dominant_fraction or any(
            c['fraction'] >= cfg.substantial_rival_fraction for c in substantial[1:]):
        result.update(status='ambiguous_palette', reason='Reliable chromatic cores contain competing substantial colour directions; original dominance thresholds retained.')
        return result
    own = labels == winner['cluster_id']
    winner_mask = np.zeros(core.shape, bool)
    winner_mask[ys[own], xs[own]] = True
    spatial, coherent = _spatial_support(winner_mask, cfg)
    result['spatial_support'] = spatial
    # A desaturated but differently hued coherent object remains a rival. Neutral
    # mixtures sharing the core hue are only discounted from palette selection,
    # not accepted as matching pixels or deleted from downstream evidence.
    mean_hue = hues[winner_mask].mean(axis=0)
    mean_hue /= max(float(np.linalg.norm(mean_hue)), 1e-9)
    neutral_rival = near_neutral & (hues @ mean_hue < math.cos(math.radians(cfg.neutral_hue_tolerance_degrees)))
    rival_spatial, _ = _spatial_support(neutral_rival, cfg)
    rival_fraction = float(neutral_rival.sum()/max(1, chromatic.sum()))
    result['near_neutral_review'] = dict(same_hue_or_uncertain_pixels=int((near_neutral & ~neutral_rival).sum()),
        different_hue_pixels=int(neutral_rival.sum()), different_hue_fraction=rival_fraction,
        different_hue_spatial_support=rival_spatial,
        interpretation='Not independent palette evidence when nearly neutral and compatible with the reliable core hue; not converted into positive marker evidence.')
    if rival_fraction >= cfg.substantial_rival_fraction and rival_spatial['sufficient_for_single_series_palette']:
        result.update(status='ambiguous_palette', reason='Substantial coherent desaturated ink has a different hue from the reliable core; not dismissed as neutral nuisance.')
        return result
    if not spatial['sufficient_for_single_series_palette']:
        result.update(status='insufficient_spatial_palette_support', reason='Dominant chromatic ink lacks coherent x-distributed support; it may be a guide or annotation. Do not activate a data series automatically.')
        return result
    # The model uses observed dominant-core pixels, not denoised or recoloured
    # synthetic samples. Membership retains the production residual semantics.
    observed = pixels[own]
    model = evidence._estimate_model(observed.reshape(-1, 1, 3), paper_bgr=paper)
    local, sample = v1._sample_box(crop, allowed, model, cfg)
    if local is None:
        result.update(status='no_usable_palette_source_box', sample=sample)
        return result
    a, b, c, d = local
    source = [a+x0, b+y0, c+x0, d+y0]
    result.update(status='estimated', model=model, source_box=source,
        source_center=[(source[0]+source[2]-1)/2, (source[1]+source[3]-1)/2],
        rgb=[int(round(v)) for v in model['bgr'][::-1]], sample=dict(sample, box_source=source),
        selected_cluster_id=winner['cluster_id'], source_pixel_count=int(len(observed)),
        model_provenance='color_marker_evidence._estimate_model on actual dominant reliable-core plot pixels; paper from actual allowed plot pixels.',
        miner_note='Use rgb as color_rgb and keep source sample separate from real legend exclusion; frozen recurrent-body mining is unchanged.')
    return result

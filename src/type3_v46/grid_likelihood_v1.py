"""Non-destructive likelihood that source ink belongs to repeated grid rails.

Unlike deleting horizontal rows, this model returns a *cost feature*. Callers
can penalize a path that follows compatible rail pixels while retaining all
original image/mask evidence, including darker or chromatic curve crossings.
No series IDs, known curve coordinates, or ground-truth points are used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GridLikelihoodConfig:
    contrast_floor: float = 5.0
    minimum_row_coverage: float = 0.78
    minimum_coherent_coverage: float = 0.72
    missing_rail_coherent_coverage: float = 0.52
    minimum_family_rows: int = 3
    minimum_family_span_fraction: float = 0.25
    minimum_spacing_fraction: float = 0.06
    spacing_error_px: float = 2.0
    maximum_band_fraction: float = 0.012
    halo_px: int = 2
    profile_seed_radius: float = 14.0
    minimum_sigma: float = 3.0
    maximum_sigma: float = 7.0
    plateau_radius: float = 4.0


def _row_profile(row, valid, paper, cfg):
    """Robust row-specific color profile, rather than one RGB per rail band."""
    points = row[valid]
    if len(points) < 8:
        return None
    center = np.median(points, axis=0)
    if np.linalg.norm(center - paper) < cfg.contrast_floor:
        return None
    distance = np.linalg.norm(points - center, axis=1)
    inliers = distance <= cfg.profile_seed_radius
    if inliers.sum() < 8:
        return None
    center = np.median(points[inliers], axis=0)
    residual = np.linalg.norm(points[inliers] - center, axis=1)
    sigma = float(np.clip(1.4826 * np.median(residual),
                          cfg.minimum_sigma, cfg.maximum_sigma))
    full_distance = np.linalg.norm(row - center, axis=1)
    coherent = valid & (full_distance <= max(cfg.profile_seed_radius, 2 * sigma))
    # A long source stroke must have support throughout the row, not just a
    # compact plateau that occupies one end of a plot.
    bins = np.array_split(np.arange(len(row)), 8)
    distributed = sum(bool(np.any(coherent[b])) for b in bins)
    return dict(center=center, sigma=sigma, coherent=coherent,
                coverage=float(coherent.sum() / max(valid.sum(), 1)),
                distributed_bins=int(distributed),
                both_edges=bool(np.any(coherent[bins[0]]) and np.any(coherent[bins[-1]])))


def horizontal_grid_likelihood(image_bgr, valid=None, cfg=GridLikelihoodConfig()):
    """Return ``(float32[H,W] rail_compatibility, JSON-safe diagnostics)``.

    Strong repeated near-full-width thin rows establish a grid family. Every
    anti-aliased row in each rail's small halo is modelled separately. Missing
    family members are admitted only if a long coherent source row is present
    near the predicted position; periodic spacing alone adds no penalty.

    The returned values describe similarity to the measured grid appearance,
    not certainty that an underlying data curve is absent. In particular, an
    exactly coincident, same-color curve cannot be distinguished from its grid.
    Inputs are never modified. Color channels may be BGR or RGB, consistently.
    """
    source = np.asarray(image_bgr)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('image_bgr must have shape (height, width, 3)')
    image = source.astype(np.float32, copy=True)
    h, w = image.shape[:2]
    scope = np.ones((h, w), bool) if valid is None else np.asarray(valid, bool)
    if scope.shape != (h, w):
        raise ValueError('valid and image dimensions differ')
    output = np.zeros((h, w), np.float32)
    if not h or not w or not scope.any():
        return output, dict(status='empty_scope', selected_family=None)
    pixels = image[scope]
    brightness = pixels.mean(axis=1)
    paper = np.percentile(pixels[brightness >= np.percentile(brightness, 90)], 75, axis=0)
    ink = (np.linalg.norm(image - paper, axis=2) >= cfg.contrast_floor) & scope
    profiles = [_row_profile(image[y], scope[y], paper, cfg) for y in range(h)]
    row_coverage = ink.sum(axis=1) / np.maximum(scope.sum(axis=1), 1)
    strong = np.array([
        p is not None and row_coverage[y] >= cfg.minimum_row_coverage
        and p['coverage'] >= cfg.minimum_coherent_coverage
        and p['distributed_bins'] >= 7
        for y, p in enumerate(profiles)
    ], bool)
    edges = np.diff(np.r_[False, strong, False].astype(np.int8))
    bands = []
    max_height = max(3, round(cfg.maximum_band_fraction * min(h, w)))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a <= max_height:
            bands.append(dict(y0=int(a), y1=int(b), center=float((a + b - 1) / 2),
                              height=int(b - a), inferred_member=False))
    families = []
    for i, first in enumerate(bands):
        for second in bands[i + 1:]:
            spacing = second['center'] - first['center']
            if spacing < cfg.minimum_spacing_fraction * h:
                continue
            phase = first['center']
            members, errors = [], []
            for j, candidate in enumerate(bands):
                error = abs((candidate['center'] - phase) / spacing
                            - round((candidate['center'] - phase) / spacing)) * spacing
                if error <= cfg.spacing_error_px:
                    members.append(j)
                    errors.append(error)
            span = max(bands[j]['center'] for j in members) - min(bands[j]['center'] for j in members)
            if len(members) >= cfg.minimum_family_rows and span >= cfg.minimum_family_span_fraction * h:
                families.append((len(members), span, -sum(errors), -spacing, members, phase))
    audit = dict(config=asdict(cfg), paper_bgr=paper.tolist(), candidate_bands=bands,
                 selected_family=None, row_profiles=[], penalized_pixels=0,
                 source_arrays_modified=False,
                 limitations=['Coincident data ink identical to grid ink is not identifiable from color alone.',
                              'Several genuinely periodic full-width data plateaus can resemble grid rails.',
                              'A likelihood penalty is evidence for path scoring, not a hard removal mask.'])
    if not families:
        audit['status'] = 'no_repeated_grid_family'
        return output, audit
    _, span, _, negative_spacing, members, phase = max(families, key=lambda f: f[:4])
    spacing = -negative_spacing
    typical_height = float(np.median([bands[i]['height'] for i in members]))
    selected = [dict(bands[i]) for i in members
                if bands[i]['height'] <= max(3.0, 1.5 * typical_height)]
    # Infer a missing rail only when source pixels independently support a
    # coherent, distributed long row at the family phase.
    for multiple in range(int(np.floor(-phase / spacing)), int(np.ceil((h - phase) / spacing)) + 1):
        expected = phase + multiple * spacing
        if expected < 0 or expected >= h or any(abs(b['center'] - expected) <= cfg.spacing_error_px for b in selected):
            continue
        candidates = []
        for y in range(max(0, int(np.floor(expected - cfg.spacing_error_px))),
                       min(h, int(np.ceil(expected + cfg.spacing_error_px)) + 1)):
            p = profiles[y]
            if p is not None and row_coverage[y] >= cfg.minimum_row_coverage and p['coverage'] >= cfg.missing_rail_coherent_coverage and p['distributed_bins'] >= 5 and p['both_edges']:
                candidates.append((p['coverage'], -abs(y - expected), y))
        if candidates:
            y = max(candidates)[2]
            selected.append(dict(y0=y, y1=y + 1, center=float(y), height=1,
                                 inferred_member=True, expected_y=float(expected)))
    rows = set()
    for band in selected:
        rows.update(range(max(0, band['y0'] - cfg.halo_px), min(h, band['y1'] + cfg.halo_px)))
    protected_darker = 0
    protected_chromatic = 0
    for y in sorted(rows):
        p = profiles[y]
        if p is None or p['coverage'] < cfg.missing_rail_coherent_coverage or p['distributed_bins'] < 5 or not p['both_edges']:
            continue
        distance = np.linalg.norm(image[y] - p['center'], axis=1)
        # Keep a small flat-topped tolerance so minor JPEG quantization of a
        # rail does not provide a spurious parallel escape route to a path.
        probability = np.exp(-0.5 * (np.maximum(distance - cfg.plateau_radius, 0) / p['sigma']) ** 2)
        probability[(distance > cfg.plateau_radius + 4 * p['sigma']) | ~ink[y]] = 0
        output[y] = probability * scope[y]
        darker = image[y].mean(axis=1) < p['center'].mean() - (cfg.plateau_radius + 3 * p['sigma'])
        pixel_chroma = image[y] - image[y].mean(axis=1, keepdims=True)
        profile_chroma = p['center'] - p['center'].mean()
        chromatic = np.linalg.norm(pixel_chroma - profile_chroma, axis=1) > cfg.plateau_radius + 3 * p['sigma']
        protected_darker += int((darker & scope[y] & (output[y] < .1)).sum())
        protected_chromatic += int((chromatic & scope[y] & (output[y] < .1)).sum())
        audit['row_profiles'].append(dict(y=int(y), rgb_or_bgr=p['center'].tolist(),
            sigma=float(p['sigma']), coherent_coverage=p['coverage'],
            strong_likelihood_pixels=int((output[y] >= .8).sum())))
    audit.update(status='repeated_grid_likelihood',
        selected_family=dict(spacing_px=float(spacing), phase_px=float(phase), span_px=float(span),
                             typical_thickness_px=typical_height, bands=sorted(selected, key=lambda b: b['center'])),
        penalized_pixels=int((output > .1).sum()), strong_likelihood_pixels=int((output >= .8).sum()),
        protected_darker_residual_pixels=protected_darker,
        protected_chromatic_residual_pixels=protected_chromatic)
    return output, audit

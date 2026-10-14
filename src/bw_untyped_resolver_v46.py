"""Bounded source-pixel proposals for saved, untyped BW regions.

This module never edits the active set. A saved region only limits where to
look: every saved marker identity competes on the same pixels against a line
null, single markers and distinct-series pairs. Saved binary source masks are
used at their fixed scale; no ideal glyph, curve target or new detector is used.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from bw_exterior_line_family_v46 import external_strokes
from bw_layered_composite_v46 import compose
from bw_raster_identity_v46 import features
from bw_suppressed_v46 import decode_marker_mask


VERSION = 'bw-untyped-local-source-scene-v1'


@dataclass(frozen=True)
class Config:
    maximum_regions: int = 24
    maximum_models: int = 10
    maximum_seed_centres: int = 12
    maximum_centres: int = 100
    maximum_crop_side: int = 161
    maximum_diameter: int = 64
    maximum_shift: int = 2
    placements_per_identity: int = 2
    marker_cost: float = .025
    maximum_loss: float = .50
    minimum_improvement: float = .05
    minimum_member_improvement: float = .025
    identity_margin: float = .025
    minimum_independent_fraction: float = .20
    minimum_independent_recall: float = .80
    maximum_missing_paper: float = .14
    boundary_weight: float = .35
    allow_pairs: bool = True


def _hash_image(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _metadata(model):
    return {key: deepcopy(value) for key, value in model.items()
            if key not in ('mask', 'anchor') and not isinstance(value, np.ndarray)}


def _point_binding(point):
    return {key: deepcopy(point.get(key)) for key in (
        'point_id', 'swatch_id', 'class_name', 'cx', 'cy', 'marker_mask',
        'marker_offset_x', 'marker_offset_y', 'marker_scale', 'effective_diameter')}


def _union(*arrays):
    result = np.zeros_like(arrays[0], dtype=np.float32)
    for value in arrays:
        result = 1 - (1 - result) * (1 - value)
    return result


def _stamp(mask, anchor, center, origin, shape):
    return cv2.warpAffine(np.asarray(mask, np.float32), np.float32([
        [1, 0, center[0] - origin[0] - anchor[0]],
        [0, 1, center[1] - origin[1] - anchor[1]]]),
        (shape[1], shape[0]), flags=cv2.INTER_LINEAR)


def _runs(values):
    ids = np.flatnonzero(values)
    return [] if not len(ids) else np.split(ids, np.flatnonzero(np.diff(ids) > 2) + 1)


def _finite_line_families(observed, body, center):
    """Restrict exterior-measured lines to their observed arms and held-out body.

    The shared hinge is the saved region centre, never a best-fitting marker
    centre. A bent alternative joins only exterior arms that already reach the
    body and whose measured strips cross that hinge. Source pixels within the
    held-out envelope never fit or select a line.
    """
    _, records = external_strokes(observed, body)
    yy, xx = np.indices(observed.shape, dtype=float)
    outer = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    by, bx = np.nonzero(body)
    diameter = max(np.ptp(bx) + 1, np.ptp(by) + 1) if len(bx) else 3
    predicted = []; bent = []; audit = []
    for record in records[:6]:
        normal = np.asarray(record['normal'], float)
        offset = float(record['offset'])
        if normal.shape != (2,) or not np.isfinite(normal).all():
            continue
        direction = np.array([normal[1], -normal[0]])
        perp = xx * normal[0] + yy * normal[1] - offset
        along = xx * direction[0] + yy * direction[1]
        body_values = along[body & (np.abs(perp) <= 2.5)]
        if not len(body_values):
            continue
        low, high = float(body_values.min()), float(body_values.max())
        shift = math.floor(float(along.min()))
        support = np.zeros(math.ceil(float(along.max())) - shift + 1, bool)
        dark = outer & (observed > .25) & (np.abs(perp) < .9)
        support[np.clip(np.rint(along[dark]).astype(int) - shift, 0, len(support)-1)] = True
        arms = []
        for run in _runs(support):
            start, end = int(run[0]) + shift, int(run[-1]) + shift
            gap = max(low-end, start-high, 0.)
            if end-start+1 >= max(4., .4*diameter) and gap <= 3.5:
                arms.append((start, end))
        if not arms:
            continue
        exterior = np.zeros_like(body)
        for start, end in arms:
            exterior |= (along >= start-.5) & (along <= end+.5)
        fitting = exterior & outer
        core = observed[fitting & (np.abs(perp) < .9)]
        left = observed[fitting & (perp > 2.5) & (perp < 4.)]
        right = observed[fitting & (perp < -2.5) & (perp > -4.)]
        if min(len(core), len(left), len(right)) < 3:
            continue
        if np.median(core) - max(np.median(left), np.median(right)) < .15:
            continue
        positions = np.arange(-2.5, 2.51, .5)
        profile = []
        for delta in positions:
            samples = observed[fitting & (np.abs(perp-delta) < .5)]
            profile.append(float(np.median(samples)) if len(samples) >= 3 else 0.)
        line = np.interp(perp, positions, profile, left=0., right=0.).astype(np.float32)
        full_extent = np.zeros_like(body); bent_extent = np.zeros_like(body)
        hinge = float(np.dot(center, direction))
        can_bend = abs(float(np.dot(center, normal)) - offset) <= 2.5
        for start, end in arms:
            full_extent |= (along >= min(start, low)-.5) & (along <= max(end, high)+.5)
            lo, hi = (min(start, hinge), max(end, hinge)) if can_bend else (min(start, low), max(end, high))
            bent_extent |= (along >= lo-.5) & (along <= hi+.5)
        full = line * full_extent
        if any(np.mean(np.abs(full - old)) < .001 for old in predicted):
            continue
        predicted.append(full); bent.append(line * bent_extent)
        audit.append(dict(normal=normal.tolist(), offset=offset, exterior_arms=arms,
                          profile=profile, finite=True, hinge_used=can_bend,
                          fit_region='outside_all_candidate_and_active_bodies'))
        if len(predicted) == 4:
            break
    zero = np.zeros_like(observed)
    families = [('no_lines', zero)]
    if predicted:
        full = np.maximum.reduce(predicted)
        families.append(('finite_exterior_lines', full))
        joined = np.maximum.reduce(bent)
        if not np.array_equal(full, joined):
            families.append(('finite_exterior_bent_arms', joined))
    return families, audit


def _member_evidence(observed, member, others, lines, valid, cfg, *, occluders=(), front_lines=None):
    marker = member['marker']
    cover = np.maximum.reduce([p['cover'] for p in others]) if others else np.zeros_like(marker)
    other_ink = _union(*[p['marker'] for p in others]) if others else np.zeros_like(marker)
    mass = marker * marker * valid
    total = max(float(mass.sum()), 1.e-8)
    independent = mass * (1-cover) * (1-lines)
    remaining = np.clip((observed-_union(other_ink, lines)) /
                        np.maximum(1-_union(other_ink, lines), .02), 0, 1)
    amount = float(independent.sum())
    fraction = amount / total
    recall = float(np.sum(independent * np.clip(remaining / np.maximum(marker, .05), 0, 1)) /
                   max(amount, 1.e-8))
    # Actual white paper cannot become invisible merely because a fitted line
    # predicts ink there. A different, source-supported opaque marker can hide it.
    foreground_cover = (np.maximum.reduce([p['cover'] for p in occluders])
                        if occluders else np.zeros_like(marker))
    line_occlusion = np.minimum(front_lines, observed) if front_lines is not None else 0.
    visible = mass * (1-foreground_cover) * (1-line_occlusion)
    missing = float(np.sum(visible * np.clip((marker-observed)/np.maximum(marker, .05), 0, 1)) / total)
    yy, xx = np.indices(marker.shape)
    cx, cy = member['local_center']
    supported = independent * np.clip(remaining / np.maximum(marker, .05), 0, 1)
    sectors = [float(supported[((xx >= cx) == sx) & ((yy >= cy) == sy)].sum())
               for sx, sy in ((False, False), (True, False), (False, True), (True, True))]
    diverse = sum(value >= max(1., .08*amount) for value in sectors) >= 2
    accepted = (fraction >= cfg.minimum_independent_fraction and
                recall >= cfg.minimum_independent_recall and
                missing <= cfg.maximum_missing_paper and amount >= 5 and diverse)
    return dict(accepted=bool(accepted), independent_fraction=fraction,
                independent_recall=recall, independent_pixels=amount,
                missing_paper_fraction=missing, spatial_diversity=bool(diverse),
                missing_paper_occluders=[p['swatch_id'] for p in occluders])


def _spread_centres(values, maximum, initial=()):
    """Deterministic spatial coverage, selected without inspecting target ink."""
    remaining = sorted(set(values))
    selected = [value for value in dict.fromkeys(initial) if value in remaining][:maximum]
    remaining = [value for value in remaining if value not in selected]
    if not selected and remaining:
        selected.append(remaining.pop(0))
    while remaining and len(selected) < maximum:
        best = max(remaining, key=lambda value: (
            min((value[0]-old[0])**2+(value[1]-old[1])**2 for old in selected),
            -value[0], -value[1]))
        selected.append(best); remaining.remove(best)
    return selected


def _inside(center, plot, legend):
    x, y = center
    return (plot[0] <= x < plot[2] and plot[1] <= y < plot[3] and
            not (legend and legend[0] <= x < legend[2] and legend[1] <= y < legend[3]))


def _resolve_region(image, region, models, active, plot, legend, cfg):
    rid = str(region.get('region_id', ''))
    review = dict(region_id=rid, version=VERSION, status='unresolved', reason='',
                  source_model_kind='saved_binary_source_raster', centers_tested=0,
                  single_scenes=0, pair_scenes=0, fixed_scale=True)
    box = region.get('bbox', ())
    if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        return None, dict(review, reason='invalid_region_box')
    a, b, c, d = map(int, box)
    if not (0 <= a < c <= image.shape[1] and 0 <= b < d <= image.shape[0]):
        return None, dict(review, reason='invalid_region_box')
    source_hash = _hash_image(image[b:d, a:c])
    if region.get('source_patch_sha256') != source_hash:
        return None, dict(review, reason='source_region_binding_changed')
    if not models or len(models) > cfg.maximum_models:
        return None, dict(review, reason='source_model_budget_or_missing')
    diameter = max(max(m['mask'].shape) for m in models.values())
    if diameter > cfg.maximum_diameter:
        return None, dict(review, reason='source_diameter_budget')
    center = (float(region.get('cx', (a+c)/2)), float(region.get('cy', (b+d)/2)))
    if not np.isfinite(center).all() or not (a <= center[0] < c and b <= center[1] < d):
        return None, dict(review, reason='invalid_region_center')
    seeds = [center]
    for value in region.get('candidate_centres', ()):
        try:
            seed = (float(value['cx']), float(value['cy']))
        except (KeyError, TypeError, ValueError):
            continue
        if (np.isfinite(seed).all() and a <= seed[0] < c and b <= seed[1] < d and
                seed not in seeds):
            seeds.append(seed)
    seed_count = len(seeds)
    seeds = _spread_centres(seeds, cfg.maximum_seed_centres, [center])
    centers = sorted(set((x+dx, y+dy) for x, y in seeds
                        for dy in range(-cfg.maximum_shift, cfg.maximum_shift+1)
                        for dx in range(-cfg.maximum_shift, cfg.maximum_shift+1)
                        if a <= x+dx < c and b <= y+dy < d and _inside((x+dx, y+dy), plot, legend)))
    available_centers = len(centers)
    centers = sorted(_spread_centres(centers, cfg.maximum_centres, seeds))
    review.update(seed_centres=seed_count, seed_centres_used=len(seeds),
                  centers_available=available_centers,
                  center_budget_limited=available_centers > len(centers) or seed_count > len(seeds))
    if not centers:
        return None, dict(review, reason='no_local_in_scope_centers')
    pad = math.ceil(1.6*diameter) + cfg.maximum_shift
    crop = [max(0, a-pad), max(0, b-pad), min(image.shape[1], c+pad), min(image.shape[0], d+pad)]
    x0, y0, x1, y1 = crop
    if max(x1-x0, y1-y0) > cfg.maximum_crop_side:
        return None, dict(review, reason='local_crop_budget')
    patch = image[y0:y1, x0:x1]
    observed = np.clip((255.-cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32))/255., 0, 1)
    yy, xx = np.indices(observed.shape)
    valid = ((xx+x0 >= plot[0]) & (xx+x0 < plot[2]) & (yy+y0 >= plot[1]) & (yy+y0 < plot[3]))
    if legend:
        valid &= ~((xx+x0 >= legend[0]) & (xx+x0 < legend[2]) & (yy+y0 >= legend[1]) & (yy+y0 < legend[3]))
    origin = (x0, y0)
    placements = []
    for sid, model in sorted(models.items()):
        mask = np.asarray(model['mask'], np.float32)
        cover = binary_fill_holes(mask > .5).astype(np.float32)
        for position in centers:
            marker = _stamp(mask, model['anchor'], position, origin, observed.shape)
            matte = _stamp(cover, model['anchor'], position, origin, observed.shape)
            if float(np.sum(marker*valid)) < .98*float(mask.sum()):
                continue
            occupied = any(q.get('swatch_id') == sid and
                math.hypot(q['cx']-position[0], q['cy']-position[1]) < .65*max(mask.shape) for q in active)
            placements.append(dict(swatch_id=sid, center=position, marker=marker, cover=matte,
                local_center=(position[0]-x0, position[1]-y0), occupied=occupied))
    if not placements:
        return None, dict(review, reason='no_in_scope_source_placements')
    body = np.maximum.reduce([p['cover'] for p in placements]) > .1
    context = []; context_points = []
    for point in active:
        if not (x0-diameter <= point['cx'] < x1+diameter and y0-diameter <= point['cy'] < y1+diameter):
            continue
        try:
            mask = decode_marker_mask(point['marker_mask']).astype(np.float32)
        except (KeyError, TypeError, ValueError):
            return None, dict(review, reason='invalid_active_context_mask')
        anchor = (mask.shape[1]//2+float(point.get('marker_offset_x', 0)),
                  mask.shape[0]//2+float(point.get('marker_offset_y', 0)))
        marker = _stamp(mask, anchor, (point['cx'], point['cy']), origin, observed.shape)
        if not marker.any():
            continue
        outside = marker*valid*(~cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        if (outside.sum() < max(4., .12*marker.sum()) or
                np.sum(outside*np.clip(observed/np.maximum(marker, .05), 0, 1))/max(float(outside.sum()), 1.e-8) < .78):
            return None, dict(review, reason='active_context_not_independently_supported')
        matte = _stamp(binary_fill_holes(mask > .5).astype(np.float32), anchor,
                       (point['cx'], point['cy']), origin, observed.shape)
        context.append(dict(marker=marker, cover=matte)); context_points.append(_point_binding(point))
    if context:
        body |= np.maximum.reduce([p['cover'] for p in context]) > .1
    lines, line_audit = _finite_line_families(observed*valid, body, (center[0]-x0, center[1]-y0))
    domain = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)*valid
    observed_features = features(observed, domain, cfg.boundary_weight)[0]
    denominator = max(float(np.sum(observed_features**2)), 1.e-6)
    def loss(raster):
        difference = features(raster, domain, cfg.boundary_weight)[0] - observed_features
        return float(np.dot(difference, difference))/denominator
    zero = np.zeros_like(observed)
    context_ink = _union(*[p['marker'] for p in context]) if context else zero
    context_cover = np.maximum.reduce([p['cover'] for p in context]) if context else zero
    def scene(members):
        choices = []
        orders = list(itertools.permutations(members)) if len(members) > 1 else [tuple(members)]
        for ordered in orders:
            marker = context_ink.copy(); cover = context_cover.copy()
            for member in ordered:
                marker = member['marker'] + (1-member['cover'])*marker
                cover = np.maximum(cover, member['cover'])
            for line_name, line in lines:
                for drawing in ('front', 'behind'):
                    raster = compose(marker, np.maximum(cover, marker), [line], [drawing])
                    value = loss(raster) + cfg.marker_cost*len(members)
                    choices.append((value, line_name, line, drawing,
                                    [m['swatch_id'] for m in ordered]))
        value, line_name, line, drawing, order = min(choices, key=lambda item: item[0])
        selected_order = sorted(members, key=lambda p: order.index(p['swatch_id']))
        evidence = [_member_evidence(observed, p, [q for q in members if q is not p]+context,
            line, valid, cfg, occluders=selected_order[order.index(p['swatch_id'])+1:],
            front_lines=line if drawing == 'front' else None) for p in members]
        return dict(loss=value, line_family=line_name, line_order=drawing, marker_order=order,
                    members=members, member_evidence=evidence,
                    independent=all(row['accepted'] for row in evidence))
    null = scene([])
    singles = [scene([p]) for p in placements]
    pair_options = []
    for sid in sorted(models):
        own = sorted((row for row in singles if row['members'][0]['swatch_id'] == sid), key=lambda row: row['loss'])
        pair_options.extend(row['members'][0] for row in own[:cfg.placements_per_identity])
    pairs = [scene([first, second]) for first, second in itertools.combinations(pair_options, 2)
             if first['swatch_id'] != second['swatch_id'] and
             math.hypot(first['center'][0]-second['center'][0], first['center'][1]-second['center'][1]) <= 1.4*diameter] if cfg.allow_pairs else []
    scenes = singles+pairs
    # Identity competitors are retained even when their own visible evidence
    # fails: an unobservable close rival is uncertainty, not permission to win.
    by_identity = {}
    for row in scenes:
        identity = tuple(sorted(p['swatch_id'] for p in row['members']))
        if identity not in by_identity or row['loss'] < by_identity[identity]['loss']:
            by_identity[identity] = row
    ranked = sorted(by_identity.items(), key=lambda item: (item[1]['loss'], item[0]))
    identity, best = ranked[0]
    margin = ranked[1][1]['loss']-best['loss'] if len(ranked) > 1 else null['loss']-best['loss']
    improvement = null['loss']-best['loss']
    member_gains = []
    for index in range(len(best['members'])):
        reduced = scene(best['members'][:index]+best['members'][index+1:])
        member_gains.append(reduced['loss']-best['loss'])
    best_single = min(row['loss'] for row in singles)
    known = all(models[sid].get('class_name') not in (None, '', 'unknown_marker', 'suppressed')
                for sid in identity)
    reason = ('source_identity_untyped' if not known else
              'source_scene_loss' if best['loss'] > cfg.maximum_loss else
              'line_only_sufficient' if improvement < cfg.minimum_improvement else
              'insufficient_independent_marker_pixels' if not best['independent'] else
              'single_marker_sufficient' if len(identity) > 1 and best_single-best['loss'] < cfg.minimum_member_improvement else
              'pair_member_not_independently_needed' if min(member_gains) < cfg.minimum_member_improvement else
              'ambiguous_source_identity' if margin < cfg.identity_margin else
              'same_series_active_body_occupied' if any(p['occupied'] for p in best['members']) else '')
    evidence = dict(version=VERSION, config=asdict(cfg), region_id=rid, source_region_sha256=source_hash,
        source_model_kind='saved_binary_source_raster', source_patch_box=crop,
        compared_swatch_ids=sorted(models), source_region_box=[a, b, c, d],
        source_patch_sha256=_hash_image(patch), source_image_shape=list(image.shape),
        source_models_sha256=_hash_json({k: dict(metadata=_metadata(v), anchor=list(v['anchor']),
            mask_sha256=_hash_image(v['mask'])) for k, v in sorted(models.items())}),
        active_context_sha256=_hash_json(sorted(context_points, key=lambda row: (str(row['point_id']), str(row['swatch_id'])))),
        target_centres=[dict(swatch_id=p['swatch_id'], cx=p['center'][0], cy=p['center'][1]) for p in best['members']],
        null_loss=null['loss'], scene_loss=best['loss'], marker_improvement=improvement,
        identity_margin=margin, member_improvements=member_gains, member_evidence=best['member_evidence'],
        line_family=best['line_family'], line_order=best['line_order'], marker_order=best['marker_order'],
        exterior_lines=line_audit, fixed_denominator=denominator, shared_domain_pixels=int(domain.sum()),
        source_pixels_modified=False, active_points_modified=False, reference_points_read=False,
        alternatives=[dict(swatch_ids=list(key), loss=row['loss']) for key, row in ranked])
    review.update(reason=reason or 'independent_source_scene_supported', centers_tested=len(centers),
                  single_scenes=len(singles), pair_scenes=len(pairs), evidence=evidence)
    if reason:
        return None, review
    points = []
    for index, placement in enumerate(best['members']):
        sid = placement['swatch_id']; point = _metadata(models[sid])
        ident = 'U_' + hashlib.sha256((rid+'|'+sid+'|'+repr(placement['center'])).encode()).hexdigest()[:16]
        point.update(swatch_id=sid, template=sid, cx=float(placement['center'][0]), cy=float(placement['center'][1]),
            point_id=ident, candidate_id=ident, source=VERSION, original_detection=False,
            state='active', confidence=float(np.clip(1-best['loss'], 0, 1)),
            confidence_kind='local_source_scene_fit_not_probability', untyped_region_id=rid,
            untyped_scene_evidence=deepcopy(evidence))
        points.append(point)
    review['status'] = 'proposed'
    return dict(region_id=rid, points=points, gain=improvement-cfg.minimum_improvement,
                evidence=evidence), review


def resolve_regions(image_bgr, regions, models, active, plot, legend=None):
    """Return bounded atomic proposals and reviews; inputs remain unchanged.

    Re-run after every accepted action. No persisted ``passed`` flag authorizes
    addition; every call checks source hashes and measures the current scene.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR source image')
    cfg = Config(); proposals = []; reviews = []
    chroma = image.max(axis=2).astype(np.int16)-image.min(axis=2).astype(np.int16)
    if np.count_nonzero(chroma > 24) > .01*chroma.size:
        return [], [dict(region_id=str(region.get('region_id', '')), version=VERSION,
                         status='unresolved', reason='color_source_not_supported') for region in regions]
    for index, region in enumerate(regions):
        if index >= cfg.maximum_regions:
            reviews.append(dict(region_id=str(region.get('region_id', '')), version=VERSION,
                                status='unresolved', reason='region_budget'))
            continue
        try:
            proposal, review = _resolve_region(image, region, models, active, plot, legend, cfg)
        except (KeyError, TypeError, ValueError, OverflowError, cv2.error) as exc:
            proposal = None
            review = dict(region_id=str(region.get('region_id', '')), version=VERSION,
                          status='unresolved', reason='invalid_saved_region_or_model',
                          detail=str(exc))
        reviews.append(review)
        if proposal is not None:
            proposals.append(proposal)
    return proposals, reviews

"""Opt-in, fixed-geometry circle-arc / connected-square-corner diagnostics.

The production default does not call this module. An explicit shadow audit can
call it without changing decisions. Its outputs are diagnostic evidence, not
permission to promote or relabel a marker. Geometry and expected
ink amplitude must come from the source swatch and the caller's existing
alignment, never from a new target-body or per-corner optimization.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from bw_circle_outline_identity_v46 import _field

VERSION = 'fixed-connected-corner-shadow-v1'

# Shared transition conventions come from bw_circle_outline_identity_v46:
# fixed source amplitude, dark-side .75px, paper-side 1px, signed Sobel normal.
# Constants describe visibility/geometry, not a particular plot or GT case.
CONSTANTS = dict(
    minimum_radius_px=4.5,
    maximum_aspect_ratio=1.8,
    source_ink_floor=.03,
    ink_side_offset_px=.75,
    paper_side_offset_px=1.,
    minimum_contrast=.35,
    minimum_direction=.85,
    bridge_minimum_direction=.70,
    nuisance_exclusion_level=.20,
    missing_ink_level=.18,
    missing_ink_penalty=.50,
    minimum_visible_fraction=.60,
    minimum_positive_fraction=.60,
    minimum_arm_normal_concentration=.97,
    minimum_distinct_corners=2,
    minimum_distinct_arcs=2,
    maximum_aa_corner_radius_px=1.5,
    maximum_aa_corner_radius_fraction=.15,
    arm_fractions=(.25, .40, .55, .70, .82),
    bridge_angles_degrees=(0., 15., 30., 45., 60., 75., 90.),
    arc_angles_degrees=(18., 27., 36., 45., 54., 63., 72.),
)


def _pair(value, name):
    data = np.asarray(value, dtype=float)
    if data.shape != (2,) or not np.isfinite(data).all():
        raise ValueError(name + ' must contain two finite values')
    return tuple(float(v) for v in data)


def corner_evidence(observed, *, geometry, source_ink_level,
                    available=None, nuisance=None, inner_radii=None,
                    capture=False):
    """Measure adjacent straight arms + their connecting corner, or curved arcs.

    Args:
        observed: Native grayscale darkness in [0,1], with white=0/black=1.
        geometry: ``{'center': [x,y], 'radii': [rx,ry]}`` from the source
            geometry placed at an existing candidate. Square axes are aligned
            with the image. No translation, scale, corner, or radius is fitted.
        source_ink_level: Fixed source-swatch darkness amplitude in (0,1].
        available: Source-observation availability in [0,1]. Samples outside
            the supplied image are unavailable, never invented white paper.
        nuisance: Independently established nuisance/occlusion in [0,1]. It
            must not be a mask selected from the model's own residual errors.
            Masked samples are unknown; they earn no positive support.
        inner_radii: Optional source inner extent for an open symbol. If
            supplied, corresponding inner arms/bridge or curved arc must also
            support the identity; outer success does not repair inner failure.
        capture: Include actual sample coordinates/brightness/normal arrays as
            JSON lists. Coordinates remain in the caller's image coordinate
            system even though gradients are computed only in a bounded ROI.

    Returns:
        JSON-friendly record with ``status`` equal to ``square_corner``,
        ``circle_arc``, or ``uncertain``. Per-corner and per-arc evidence stays
        available even on abstention. ``shadow_only`` is always true and
        ``promotion_allowed`` always false. Scores are not probabilities.
    """
    started = time.perf_counter()
    ink = _field(observed, 'observed')
    valid = np.ones_like(ink) if available is None else _field(available, 'available', ink.shape)
    excluded = np.zeros_like(ink) if nuisance is None else _field(nuisance, 'nuisance', ink.shape)
    if not isinstance(geometry, dict):
        raise ValueError('geometry must be a source-derived geometry dictionary')
    center = _pair(geometry.get('center'), 'geometry.center')
    radii = _pair(geometry.get('radii'), 'geometry.radii')
    if min(radii) <= 0:
        raise ValueError('geometry.radii must be positive')
    amplitude = float(source_ink_level)
    if not math.isfinite(amplitude) or not 0 < amplitude <= 1:
        raise ValueError('source_ink_level must be in (0, 1]')
    inner = None if inner_radii is None else _pair(inner_radii, 'inner_radii')
    if inner is not None and (min(inner) <= 0 or any(i >= r for i, r in zip(inner, radii))):
        raise ValueError('inner_radii must be positive and strictly inside outer radii')

    out = dict(version=VERSION, status='uncertain', preferred_shape=None,
        reason='insufficient_connected_corner_or_curved_arc',
        geometry=dict(center=list(center), radii=list(radii), inner_radii=None if inner is None else list(inner)),
        source_ink_level=amplitude, corners=[], arcs=[],
        square_supported_corners=0, circle_supported_arcs=0,
        scores=dict(square_corner=0., circle_arc=0.),
        constants=dict(CONSTANTS), source_pixels_modified=False,
        shadow_only=True, promotion_allowed=False, source_geometry_refitted=False,
        per_corner_optimization=False, nuisance_is_positive_evidence=False,
        gray_policy='fixed_source_amplitude_not_target_contrast_stretch',
        score_kind='signed_transition_support_minus_visible_missing_ink_not_probability',
        angle_convention='image coordinates: x right, y down; angles increase clockwise',
        limitations=['Axis-aligned square and circle/ellipse probes only.',
            'A single L corner is reported locally but cannot establish global identity.',
            'Masked or unavailable samples cannot become positive evidence.'])

    def finish(reason=None):
        if reason is not None:
            out['reason'] = reason
        out['timing_seconds'] = float(time.perf_counter() - started)
        return out

    if amplitude < CONSTANTS['source_ink_floor']:
        return finish('source_ink_too_weak')
    if min(radii) <= CONSTANTS['minimum_radius_px'] or max(radii)/min(radii) > CONSTANTS['maximum_aspect_ratio']:
        return finish('small_or_extreme_aspect_source_geometry')
    if inner is not None and min(inner) <= 2.:
        return finish('inner_extent_too_small_for_independent_paired_measurement')
    if abs(float(geometry.get('rotation_degrees', 0.))) > 1.e-6:
        return finish('rotated_geometry_not_supported')

    # Work only around known source geometry. No whole-image gradient or search.
    cx, cy = center
    a, b = radii
    left, top = max(0, int(math.floor(cx-a-4))), max(0, int(math.floor(cy-b-4)))
    right, bottom = min(ink.shape[1], int(math.ceil(cx+a+5))), min(ink.shape[0], int(math.ceil(cy+b+5)))
    if right <= left or bottom <= top:
        return finish('geometry_outside_observed_image')
    obs = np.clip(ink[top:bottom, left:right] / amplitude, 0, 1)
    vis = valid[top:bottom, left:right]
    unknown = excluded[top:bottom, left:right]
    smooth = cv2.GaussianBlur(obs, (3,3), .55)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    out['gradient_roi'] = [left, top, right, bottom]
    out['gradient_pixels'] = int(obs.size)

    def sample(field, points):
        points = np.asarray(points, float)
        return cv2.remap(field, np.float32(points[:,0]-left)[None,:],
            np.float32(points[:,1]-top)[None,:], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]

    def transition(points, normals, polarity, *, bridge=False):
        points, normals = np.asarray(points, float), np.asarray(normals, float)
        ink_points = points + polarity*CONSTANTS['ink_side_offset_px']*normals
        paper_points = points - polarity*CONSTANTS['paper_side_offset_px']*normals
        dark = sample(obs, ink_points)
        paper = sample(obs, paper_points)
        # Outer edge is sampled .25px into ink as in outline_identity.
        edge_points = points + polarity*.25*normals
        edge = np.stack([sample(gx, edge_points), sample(gy, edge_points)], axis=1)
        magnitudes = np.hypot(edge[:,0], edge[:,1])
        direction = np.clip(polarity*np.sum(edge*normals, axis=1)/np.maximum(magnitudes,.05),0,1)
        contrast = np.clip(dark-paper,0,1)
        visible = np.ones(len(points),bool)
        for coords in (ink_points, points, paper_points):
            visible &= sample(vis, coords) > .99
            visible &= sample(unknown, coords) < CONSTANTS['nuisance_exclusion_level']
        minimum_direction = CONSTANTS['bridge_minimum_direction'] if bridge else CONSTANTS['minimum_direction']
        positive = visible & (contrast >= CONSTANTS['minimum_contrast']) & (direction >= minimum_direction)
        missing = np.clip((CONSTANTS['missing_ink_level']-dark)/CONSTANTS['missing_ink_level'],0,1)
        per_point = np.where(visible,contrast*direction-CONSTANTS['missing_ink_penalty']*missing,0.)
        count = int(visible.sum())
        expected = len(points)
        normals_seen = polarity*edge[positive]/np.maximum(magnitudes[positive,None],1.e-6)
        concentration = float(np.hypot(*normals_seen.mean(axis=0))) if positive.any() else 0.
        basic = (count/expected >= CONSTANTS['minimum_visible_fraction']
            and positive.sum()/max(1,count) >= CONSTANTS['minimum_positive_fraction'])
        row = dict(score=float(per_point.mean()), visible_samples=count, total_samples=expected,
            visible_fraction=float(count/expected), positive_samples=int(positive.sum()),
            positive_fraction=float(positive.sum()/max(count,1)),
            normal_concentration=concentration,
            mean_ink=float(dark[visible].mean()) if count else None,
            mean_paper=float(paper[visible].mean()) if count else None,
            missing_ink_penalty=float(CONSTANTS['missing_ink_penalty']*np.where(visible,missing,0.).mean()),
            unknown_samples=int(expected-count), supported=bool(basic),
            reason='positive_transition' if basic else ('insufficient_observable_samples' if count/expected < CONSTANTS['minimum_visible_fraction'] else 'missing_or_incompatible_transition'))
        if capture:
            row['samples'] = dict(xy=points.tolist(), ink_xy=ink_points.tolist(), paper_xy=paper_points.tolist(),
                expected_normal=normals.tolist(), edge_gradient=edge.tolist(), polarity=int(polarity),
                ink=dark.tolist(), paper=paper.tolist(), contrast=contrast.tolist(), direction=direction.tolist(),
                visible=visible.tolist(), positive=positive.tolist(), missing_ink=missing.tolist(), contribution=per_point.tolist())
        return row, dict(positive=positive, visible=visible, edge=edge, magnitudes=magnitudes, contribution=per_point)

    def straight(points, normals, polarity):
        row, samples = transition(points,normals,polarity)
        row['supported'] = bool(row['supported'] and row['normal_concentration'] >= CONSTANTS['minimum_arm_normal_concentration'])
        if not row['supported'] and row['reason'] == 'positive_transition':
            row['reason'] = 'observed_normals_not_straight'
        return row

    fractions = np.asarray(CONSTANTS['arm_fractions'])
    bridge_angles = np.radians(CONSTANTS['bridge_angles_degrees'])

    def square_corner(rx, ry, sx, sy, polarity):
        horizontal = np.column_stack([cx+sx*rx*fractions, np.full(len(fractions),cy+sy*ry)])
        vertical = np.column_stack([np.full(len(fractions),cx+sx*rx), cy+sy*ry*fractions])
        horizontal_row = straight(horizontal,np.tile([0,sy],(len(fractions),1)),polarity)
        vertical_row = straight(vertical,np.tile([sx,0],(len(fractions),1)),polarity)
        # One fixed subpixel-compatible corner, NOT a radius chosen per corner.
        # Sharp corners and ordinary antialiasing lie within this short band.
        rounding = min(CONSTANTS['maximum_aa_corner_radius_px'], CONSTANTS['maximum_aa_corner_radius_fraction']*min(rx,ry))
        bridge_normals = np.column_stack([sx*np.cos(bridge_angles),sy*np.sin(bridge_angles)])
        bridge_points = np.column_stack([cx+sx*(rx-rounding)+rounding*bridge_normals[:,0],
            cy+sy*(ry-rounding)+rounding*bridge_normals[:,1]])
        bridge_row, samples = transition(bridge_points,bridge_normals,polarity,bridge=True)
        # The middle of the corner must connect the arms: the two endpoints
        # alone cannot pass an interrupted L, plus, or two unrelated lines.
        middle = np.array([2,3,4])
        bridge_row.update(middle_visible_samples=int(samples['visible'][middle].sum()),
            middle_positive_samples=int(samples['positive'][middle].sum()), fixed_rounding_radius_px=float(rounding))
        bridge_row['supported'] = bool(bridge_row['supported'] and samples['visible'][middle].sum()>=2 and samples['positive'][middle].sum()>=2)
        if not bridge_row['supported'] and bridge_row['reason']=='positive_transition':
            bridge_row['reason']='corner_bridge_not_connected'
        supported = bool(horizontal_row['supported'] and vertical_row['supported'] and bridge_row['supported'])
        all_parts = (horizontal_row,vertical_row,bridge_row)
        return dict(supported=supported,horizontal_arm=horizontal_row,vertical_arm=vertical_row,bridge=bridge_row,
            score=float(min(r['score'] for r in all_parts)),
            reason='adjacent_straight_arms_and_connected_corner' if supported else
                ('adjacent_straight_arms_missing' if not(horizontal_row['supported'] and vertical_row['supported']) else 'corner_bridge_missing'))

    def curved_arc(rx,ry,quadrant,polarity):
        angles=np.radians(np.asarray(CONSTANTS['arc_angles_degrees'])+90*quadrant)
        ux,uy=np.cos(angles),np.sin(angles)
        distance=1/np.sqrt((ux/rx)**2+(uy/ry)**2)
        points=np.column_stack([cx+distance*ux,cy+distance*uy])
        normals=np.column_stack([ux/rx**2,uy/ry**2])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        row,samples=transition(points,normals,polarity)
        means=[];counts=[]
        for idx in (np.array([0,1]),np.array([2,3,4]),np.array([5,6])):
            take=idx[samples['positive'][idx]]
            counts.append(len(take))
            vec=polarity*samples['edge'][take]/np.maximum(samples['magnitudes'][take,None],1.e-6)
            means.append(math.atan2(float(vec[:,1].mean()),float(vec[:,0].mean())) if len(take) else 0.)
        turns=[math.atan2(math.sin(means[j+1]-means[j]),math.cos(means[j+1]-means[j])) for j in (0,1)]
        row.update(normal_bin_positive_samples=counts,normal_turn_degrees=[float(math.degrees(t)) for t in turns])
        turning = min(counts)>=2 and all(.12<=t<=.80 for t in turns)
        row['supported']=bool(row['supported'] and turning)
        if not row['supported'] and row['reason']=='positive_transition':
            row['reason']='normals_do_not_turn_through_three_visible_arc_parts'
        return row

    # Clockwise quadrants: bottom-right, bottom-left, top-left, top-right.
    signs=((1,1),(-1,1),(-1,-1),(1,-1))
    for quadrant,(sx,sy) in enumerate(signs):
        outer=square_corner(a,b,sx,sy,-1)
        paired=None if inner is None else square_corner(*inner,sx,sy,1)
        supported=outer['supported'] and (paired is None or paired['supported'])
        row=dict(quadrant=quadrant,signs=[sx,sy],supported=bool(supported),**{
            key:outer[key] for key in ('horizontal_arm','vertical_arm','bridge')},
            outer=outer,inner=paired,score=outer['score'] if paired is None else min(outer['score'],paired['score']),
            reason=outer['reason'] if paired is None or paired['supported'] else 'inner_corner_not_independently_supported')
        out['corners'].append(row)
        arc=curved_arc(a,b,quadrant,-1)
        paired_arc=None if inner is None else curved_arc(*inner,quadrant,1)
        arc_supported=arc['supported'] and (paired_arc is None or paired_arc['supported'])
        out['arcs'].append(dict(quadrant=quadrant,supported=bool(arc_supported),outer=arc,inner=paired_arc,
            score=arc['score'] if paired_arc is None else min(arc['score'],paired_arc['score']),
            reason=arc['reason'] if paired_arc is None or paired_arc['supported'] else 'inner_arc_not_independently_supported'))

    out['square_supported_corners']=sum(r['supported'] for r in out['corners'])
    out['circle_supported_arcs']=sum(r['supported'] for r in out['arcs'])
    out['scores']=dict(square_corner=float(np.mean([r['score'] for r in out['corners']])),
        circle_arc=float(np.mean([r['score'] for r in out['arcs']])))
    square=out['square_supported_corners']>=CONSTANTS['minimum_distinct_corners']
    circle=out['circle_supported_arcs']>=CONSTANTS['minimum_distinct_arcs']
    if square and not circle:
        out.update(status='square_corner',preferred_shape='square',reason='multiple_independent_connected_corners')
    elif circle and not square:
        out.update(status='circle_arc',preferred_shape='circle',reason='multiple_independent_curved_arcs')
    elif square and circle:
        out['reason']='conflicting_positive_corner_and_arc_evidence'
    return finish()

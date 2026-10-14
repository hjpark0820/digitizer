"""Bounded, non-consuming exterior connector proposals for partial visibility.

This opt-in helper preserves the legacy nuisance model and only adds a line
when a long, contiguous observed exterior segment reaches a held-out body.
Neither proposal ranking nor fitting reads pixels inside any candidate body.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


LEGACY_POLICY = 'exterior-accumulator-pool-v1'
POLICY = 'exterior-accumulator-pool-v2'


@dataclass(frozen=True)
class Config:
    maximum_proposals: int = 128
    maximum_additions: int = 2


def _runs(support):
    """Intervals with at most one missing integer sample between support."""
    indices = np.flatnonzero(support)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 2)+1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def external_strokes(patch, body, *, config=None):
    """Return nuisance ink and an audit; body pixels never fit extra lines.

    Standard Hough voting does not remove support pixels after finding a
    stronger crossing connector. A proposed strip must independently reach
    a held-out envelope from a sufficiently long exterior segment. Added
    Added coverage stops at that envelope's far edge, rather than extending
    across arbitrary white image regions. Width remains the legacy measured
    ±2.5px. The separately labeled legacy baseline retains its old behavior.
    """
    from bw_raster_context_v46 import external_strokes as legacy_strokes
    cfg = config or Config()
    obs = np.asarray(patch, np.float32)
    excluded = np.asarray(body)
    if (obs.ndim != 2 or min(obs.shape, default=0) < 3 or excluded.shape != obs.shape or
        not np.isfinite(obs).all() or np.any(obs < 0) or np.any(obs > 1)):
        raise ValueError('Finite normalized source and matching body mask required')
    if (not isinstance(cfg.maximum_proposals, int) or isinstance(cfg.maximum_proposals,bool) or cfg.maximum_proposals <= 0 or
        not isinstance(cfg.maximum_additions, int) or isinstance(cfg.maximum_additions,bool) or cfg.maximum_additions <= 0):
        raise ValueError('Positive bounded integer line limits required')
    excluded = excluded.astype(bool)
    baseline, old_records = legacy_strokes(obs, excluded)
    records = [dict(r, proposal_source='legacy_probabilistic_hough', policy=POLICY) for r in old_records]
    yy, xx = np.indices(obs.shape, dtype=float)
    by, bx = np.nonzero(excluded)
    if not len(bx):
        return baseline, records
    diameter = float(max(np.ptp(bx)+1, np.ptp(by)+1))
    outer = cv2.dilate(excluded.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    dark = (obs > .25) & outer
    proposed = cv2.HoughLines(np.uint8(dark)*255, 1, np.pi/180,
                              threshold=max(4, round(.4*diameter)))
    if proposed is None:
        return baseline, records
    proposals = []
    transverse = np.arange(-2.5, 2.51, .5)
    for index, (rho, theta) in enumerate(proposed.reshape(-1, 2)[:cfg.maximum_proposals]):
        normal = np.array([math.cos(float(theta)), math.sin(float(theta))])
        direction = np.array([normal[1], -normal[0]])
        perp = xx*normal[0]+yy*normal[1]-rho
        along = xx*direction[0]+yy*direction[1]
        # Reject a short body edge or disjoint black fragments that only
        # produce a large aggregate Hough vote along an otherwise empty ray.
        near = dark & (abs(perp) < .8)
        if not near.any():
            continue
        lo = math.floor(float(along.min()))
        count = math.ceil(float(along.max()))-lo+1
        support = np.zeros(count, bool)
        support[np.clip(np.rint(along[near]).astype(int)-lo, 0, count-1)] = True
        body_along = along[excluded & (abs(perp) <= 2.5)]
        if not len(body_along):
            continue
        body_bins=np.zeros(count,bool)
        body_bins[np.clip(np.rint(body_along).astype(int)-lo,0,count-1)]=True
        body_intervals=[(first+lo,last+lo) for first,last in _runs(body_bins)]
        segments = []
        for first, last in _runs(support):
            start, end = first+lo, last+lo
            length = end-start+1
            for body_lo,body_hi in body_intervals:
                gap = max(body_lo-end, start-body_hi, 0.)
                # One withheld dilation pixel plus the next exterior lattice
                # pixel costs up to 2*sqrt(2), with half a bin of rounding.
                if length >= max(4., .5*diameter) and gap <= max(2*math.sqrt(2)+.5, .05*diameter):
                    segments.append((start, end, gap,body_lo,body_hi))
        if not segments:
            continue
        # Measure the proposed finite exterior segment itself. Sampling a
        # whole infinite stripe both dilutes finite connectors with blank
        # paper and aliases a one-pixel diagonal against adjacent white rows.
        finite=np.zeros_like(excluded)
        for start,end,_,_,_ in segments:finite |= (along>=start-.5)&(along<=end+.5)
        ts=np.unique(np.concatenate([np.arange(start,end+.25,.5) for start,end,_,_,_ in segments]))
        def measured(delta):
            values=obs[outer&finite&(abs(perp-delta)<.4)]
            if len(values)<3:
                # Half-pixel normal positions can fall between every row of
                # an axis-aligned raster. Resample the same measured finite
                # segment, requiring every interpolation neighbor to remain
                # in the observed exterior; unavailable is never paper.
                coords=direction[None,:]*ts[:,None]+normal[None,:]*(float(rho)+delta)
                px=coords[:,0].astype(np.float32)[None];py=coords[:,1].astype(np.float32)[None]
                valid=cv2.remap((outer&finite).astype(np.float32),px,py,cv2.INTER_LINEAR)[0]>.99
                values=cv2.remap(obs,px,py,cv2.INTER_LINEAR)[0][valid]
            return (float(np.median(values)) if len(values)>=3 else None,int(len(values)))
        core_value,core_count=measured(0.)
        left_value,left_count=measured(-3.5)
        right_value,right_count=measured(3.5)
        if any(v is None for v in (core_value,left_value,right_value)):
            continue
        strength=core_value
        # Both exterior flanks must be lighter: a filled-body edge has only
        # one paper flank and cannot certify an additional narrow connector.
        contrast=strength-max(left_value,right_value)
        if strength<.25 or contrast<.15:
            continue
        measured_profile=[measured(delta) for delta in transverse]
        if any(value is None for value,count in measured_profile):
            continue
        profile = [value for value,count in measured_profile]
        predicted = np.interp(perp, transverse, profile, left=0., right=0.).astype(np.float32)
        permitted = np.zeros_like(excluded)
        for start, end, gap,body_lo,body_hi in segments:
            # Segment coordinates are rounded integer bins; their physical
            # support ends at half a bin, not a whole extra unobserved pixel.
            permitted |= (along >= min(start, body_lo)-.5) & (along <= max(end, body_hi)+.5)
        predicted *= permitted
        proposals.append(dict(predicted=predicted, normal=normal.tolist(), offset=float(rho),
            strength=strength, contrast=contrast, proposal_index=index,
            segments=[dict(along_start=s, along_end=e, body_gap=g,reached_body_start=bs,reached_body_end=be)
                      for s,e,g,bs,be in segments],
            exterior_sample_counts=dict(core=core_count,left_flank=left_count,right_flank=right_count,
                                        transverse=[count for value,count in measured_profile]),
            normal_sampling_policy='source_pixel_bands_with_supported_subpixel_fallback',
            transverse_profile=profile))
    cover = baseline.copy()
    # The fixed exterior image-error objective is independent of which
    # target center/identity will later benefit from nuisance explanation.
    norm = max(float(np.sum(outer*obs*obs)), 1.e-6)
    for _ in range(cfg.maximum_additions):
        best = None
        for row in proposals:
            candidate = np.maximum(cover, row['predicted'])
            gain = float(np.sum(outer*((obs-cover)**2-(obs-candidate)**2)))/norm
            if best is None or gain > best[0]:
                best = (gain, row, candidate)
        if best is None or best[0] <= .01:
            break
        gain, row, cover = best
        records.append({k:v for k,v in row.items() if k != 'predicted'} | dict(
            policy=POLICY, proposal_source='non_consuming_hough_accumulator',
            exterior_error_improvement=gain, exterior_denominator=norm,
            fit_region='outside_union_of_all_family_candidate_and_neighbor_bodies',
            visibility_not_used_for_selection=True, finite_extrapolation=True))
        proposals = [other for other in proposals if other is not row]
    return cover, records

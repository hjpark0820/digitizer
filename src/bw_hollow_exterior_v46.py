"""Common source-supported finite connector background for hollow rivals.

Only exterior pixels propose/select rays and measure their transverse ink.
Opposite rays may differ; neither ray extrapolates beyond the body center.
Unseen caps and arbitrary interior ink are never granted an explanation.
"""
from copy import deepcopy
from functools import lru_cache
import math

import cv2
import numpy as np

from bw_exterior_line_family_v46 import _runs

VERSION = 'held-out-finite-connector-profiles-v1'
MAX_PROPOSALS = 48
MAX_SEGMENTS = 4


def _fit(source, body, available, baseline):
    yy, xx = np.indices(source.shape, dtype=np.float32)
    by, bx = np.nonzero(body)
    if len(bx) < 3:
        return baseline.copy(), dict(version=VERSION, adopted=False, reason='no_common_body', segments=[])
    cx, cy = float((bx.min()+bx.max())/2), float((by.min()+by.max())/2)
    diameter = float(max(np.ptp(bx)+1, np.ptp(by)+1))
    exterior = (cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0) & available
    dark = np.uint8((source > .25) & exterior)*255
    proposals = cv2.HoughLines(dark, 1., np.pi/180., max(4, round(.4*diameter)))
    if proposals is None:
        return baseline.copy(), dict(version=VERSION, adopted=False, reason='no_exterior_segments', segments=[])
    candidates=[]
    transverse=np.arange(-4., 4.01, .5, dtype=np.float32)
    norm=max(float(np.sum(source[exterior]**2)), 1.e-9)
    for index, (rho, theta) in enumerate(proposals.reshape(-1, 2)[:MAX_PROPOSALS]):
        nx,ny=float(np.cos(theta)),float(np.sin(theta))
        ux,uy=ny,-nx
        perp=xx*nx+yy*ny-float(rho)
        along=xx*ux+yy*uy
        center_along=cx*ux+cy*uy
        near=(dark>0)&(abs(perp)<.8)
        if not near.any():continue
        origin=math.floor(float(along.min()))
        support=np.zeros(math.ceil(float(along.max()))-origin+1,bool)
        support[np.clip(np.rint(along[near]).astype(int)-origin,0,len(support)-1)]=True
        through=along[body & (abs(perp)<=2.5)]
        body_start=float(through.min()) if len(through) else None
        body_end=float(through.max()) if len(through) else None
        for first,last in _runs(support):
            start,end=first+origin,last+origin
            if end-start+1 < max(4., .45*diameter):continue
            # Measure each outside segment separately: a kink's second arm
            # must not dictate the first arm's direction, width or opacity.
            ts=np.arange(start, end+.25, .5, dtype=np.float32)
            mx=ux*ts[None,:]+nx*(float(rho)+transverse[:,None])
            my=uy*ts[None,:]+ny*(float(rho)+transverse[:,None])
            values=cv2.remap(source,mx,my,cv2.INTER_LINEAR)
            valid=cv2.remap(exterior.astype(np.float32),mx,my,cv2.INTER_LINEAR)>.99
            counts=valid.sum(axis=1)
            if np.any(counts<3):continue
            profile=np.array([np.median(v[ok]) for v,ok in zip(values,valid)],np.float32)
            strength=float(profile.max())
            if strength<.25 or strength-max(float(profile[0]),float(profile[-1]))<.15:continue
            # Both sides must reach paper. Otherwise an exterior body edge
            # could be extrapolated as an arbitrarily broad dark connector.
            if max(float(profile[0]),float(profile[-1]))>.35*strength:continue
            peak=int(profile.argmax())
            if peak in (0,len(profile)-1):continue
            predicted=np.interp(perp,transverse,profile,left=0.,right=0.).astype(np.float32)
            lo,hi=float(start)-.5,float(end)+.5
            extension='observed_finite_segment'
            if body_start is not None:
                gap=max(body_start-end,start-body_end,0.)
                if gap<=2*math.sqrt(2)+.5:
                    if end<=center_along:
                        hi=max(hi,center_along);extension='exterior_ray_to_center'
                    elif start>=center_along:
                        lo=min(lo,center_along);extension='exterior_ray_to_center'
            predicted *= (along>=lo)&(along<=hi)
            # No exterior profile grants ink outside observed image support.
            predicted *= available
            evidence=dict(proposal_index=index,normal=[nx,ny],offset=float(rho),
                source_segment=[int(start),int(end)],predicted_interval=[lo,hi],
                extension=extension,transverse_positions=transverse.tolist(),
                transverse_ink=profile.tolist(),source_sample_counts=counts.tolist(),
                body_pixels_used=0,maximum_extrapolation='body_center',strength=strength)
            candidates.append((predicted,evidence))

    cover=np.zeros_like(source);selected=[]
    for _ in range(MAX_SEGMENTS):
        best=None
        for i,(predicted,record) in enumerate(candidates):
            combined=np.maximum(cover,predicted)
            gain=float(np.sum((source[exterior]-cover[exterior])**2-
                              (source[exterior]-combined[exterior])**2))/norm
            if best is None or gain>best[0]:best=(gain,i,combined)
        if best is None or best[0]<=.01:break
        gain,i,cover=best
        _,record=candidates.pop(i)
        selected.append(dict(record,exterior_error_improvement=gain))
    old_error=float(np.mean((baseline[exterior]-source[exterior])**2)) if exterior.any() else 0.
    new_error=float(np.mean((cover[exterior]-source[exterior])**2)) if exterior.any() else 0.
    # A source-supported replacement must improve its held-out objective.
    # Otherwise existing validated behavior remains the incumbent.
    adopted=bool(selected and new_error<old_error)
    report=dict(version=VERSION,adopted=adopted,
        reason='lower_exterior_error' if adopted else 'incumbent_exterior_model_retained',
        exterior_pixels=int(exterior.sum()),body_pixels_used=0,
        baseline_exterior_mse=old_error,finite_exterior_mse=new_error,
        proposals_considered=min(len(proposals),MAX_PROPOSALS),maximum_segments=MAX_SEGMENTS,
        segments=selected,common_background_for_all_rivals=True)
    return (cover if adopted else baseline.copy()), report


@lru_cache(maxsize=512)
def _cached(shape, source_bytes, body_bytes, valid_bytes, baseline_bytes):
    source=np.frombuffer(source_bytes,np.float32).reshape(shape)
    body=np.frombuffer(body_bytes,np.bool_).reshape(shape)
    valid=np.frombuffer(valid_bytes,np.bool_).reshape(shape)
    baseline=np.frombuffer(baseline_bytes,np.float32).reshape(shape)
    return _fit(source,body,valid,baseline)


def exterior_background(source, body, available, baseline):
    """Bounded cached fitting; no mutable caller array is cached or modified."""
    source=np.asarray(source,np.float32);body=np.asarray(body,bool)
    available=np.asarray(available,bool);baseline=np.asarray(baseline,np.float32)
    if (source.ndim!=2 or any(v.shape!=source.shape for v in (body,available,baseline))
        or not np.isfinite(source).all() or not np.isfinite(baseline).all()
        or np.any(source<0) or np.any(source>1)):
        raise ValueError('Aligned finite native ink, common body and availability required')
    cover,report=_cached(source.shape,source.tobytes(),body.tobytes(),available.tobytes(),baseline.tobytes())
    return cover.copy(),deepcopy(report)

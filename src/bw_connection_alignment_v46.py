"""Frozen, bounded per-swatch connector translations supported by source L0.

This adjusts rendering anchors, never marker rasters or exported data points.
No reference segment is moved. There is no per-action image registration.
"""
from __future__ import annotations

import hashlib
import json
import math
from time import perf_counter

import cv2
import numpy as np

from bw_suppressed_v46 import decode_marker_mask

POLICY = 'bw_frozen_connection_anchor_v2'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def _inside_markers(points, shifts):
    """Connection anchors must remain inside every observed marker's hull."""
    valid=np.ones(len(shifts),dtype=bool)
    for p in points:
        ink=decode_marker_mask(p['marker_mask']).astype(np.uint8)
        yy,xx=np.nonzero(ink)
        if len(xx)<3:
            return np.zeros(len(shifts),dtype=bool)
        hull=np.zeros_like(ink)
        cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)),1)
        # Match the production raster placement, including saved seed offsets.
        left=round(p['cx']-p.get('marker_offset_x',0.))-ink.shape[1]//2
        top=round(p['cy']-p.get('marker_offset_y',0.))-ink.shape[0]//2
        xs=np.rint(float(p['cx'])+shifts[:,0]-left).astype(int)
        ys=np.rint(float(p['cy'])+shifts[:,1]-top).astype(int)
        ok=(xs>=0)&(xs<ink.shape[1])&(ys>=0)&(ys<ink.shape[0])
        indices=np.flatnonzero(ok)
        ok[indices] &= hull[ys[indices],xs[indices]]>0
        valid &= ok
    return valid


def _distances(samples,refs):
    """Finite-segment distances, vectorized over translations and samples."""
    a=refs[:,:2];v=refs[:,2:]-a
    w=samples[:,:,None,:]-a[None,None,:,:]
    t=np.clip(np.einsum('kqrd,rd->kqr',w,v)/(v*v).sum(axis=1),0.,1.)
    delta=w-t[:,:,:,None]*v[None,None,:,:]
    return np.sqrt((delta*delta).sum(axis=3)).min(axis=2)


def _fit_series(points,reference,diameter):
    limit=min(8.,.23*diameter)
    radius=int(math.floor(limit))
    shifts=np.asarray([(dx,dy) for dx in range(-radius,radius+1)
                       for dy in range(-radius,radius+1)
                       if math.hypot(dx,dy)<=limit+1e-9],dtype=float)
    if not len(shifts):
        shifts=np.zeros((1,2),float)
    allowed=_inside_markers(points,shifts) if points else np.zeros(len(shifts),bool)
    shifts=shifts[allowed]
    zero=np.flatnonzero((shifts==0).all(axis=1))
    base=dict(offset=[0.,0.],max_translation_px=limit,observed_points=len(points),
              tested_translations=len(shifts),edges=[],accepted=False)
    if not len(zero):
        return dict(base,reason='initial_anchor_outside_measured_marker_hull')
    zero=int(zero[0])
    if len(points)<3 or len(reference)==0:
        return dict(base,reason='fewer_than_two_supported_connections')
    rv=reference[:,2:]-reference[:,:2]
    rl=np.linalg.norm(rv,axis=1)
    costs=[];evidence=[]
    for p,q in zip(points,points[1:]):
        a=np.array([p['cx'],p['cy']],float);b=np.array([q['cx'],q['cy']],float)
        v=b-a;length=float(np.linalg.norm(v))
        if length<2.*diameter or abs(v[0])<diameter:
            continue
        tangent=v/length
        # Exclude marker bodies near both endpoints before observing lines.
        margin=max(.65*diameter,5.)
        ts=np.linspace(margin,length-margin,min(32,max(12,int(length/4))))
        core=a+ts[:,None]*tangent
        parallel=np.abs(rv@tangent)/np.maximum(rl,1e-9)>=math.cos(math.radians(10))
        projection=(reference.reshape(-1,2,2)-a)@tangent
        overlap=np.maximum(0.,np.minimum(projection.max(axis=1),length-margin)-
                           np.maximum(projection.min(axis=1),margin))
        suitable=parallel & (rl>=max(8.,.7*diameter)) & (overlap>=max(6.,.15*(length-2*margin)))
        refs=reference[suitable]
        if not len(refs):
            continue
        # Retain only fragments within the small shift search neighborhood.
        near=[]
        for ref in refs:
            distance=_distances(core[None,:,:],ref[None,:])[0]
            if np.quantile(distance,.35)<=limit+2.:
                near.append(ref)
        if not near:
            continue
        refs=np.asarray(near)
        distance=_distances(core[None,:,:]+shifts[:,None,:],refs)
        # A connection needs reference coverage along most of its interior.
        if np.mean(distance[zero]<=limit+2.)<.65:
            continue
        costs.append(np.mean(np.minimum(distance,limit+2.),axis=1))
        evidence.append(dict(from_point=p.get('point_id'),to_point=q.get('point_id'),
            connection=[*a.tolist(),*b.tolist()],reference_fragments=len(refs),samples=len(core)))
    if len(costs)<2:
        return dict(base,edges=evidence,reason='fewer_than_two_supported_connections')
    costs=np.asarray(costs)
    # Every connection has equal weight; duplicate detector fragments cannot
    # become additional independent votes. Mild regularization favors no shift.
    aggregate=np.mean(np.sort(costs,axis=0)[:max(2,math.ceil(.8*len(costs)))],axis=0)
    objective=aggregate+.035*np.linalg.norm(shifts,axis=1)
    best=int(np.argmin(objective));delta=costs[:,zero]-costs[:,best]
    improvement=float(aggregate[zero]-aggregate[best])
    supporters=int(np.sum(delta>.35))
    acceptable=(improvement>=max(.55,.02*diameter) and
        aggregate[best]<=.75*aggregate[zero] and
        aggregate[best]<=max(1.6,.07*diameter) and
        supporters>=max(2,math.ceil(.6*len(costs))) and
        # Do not sacrifice even one measured connection to align a majority.
        # The purpose is a common glyph-anchor correction, not curve warping.
        float(delta.min())>=-.5)
    for e,old,new,gain in zip(evidence,costs[:,zero],costs[:,best],delta):
        e.update(distance_before_px=float(old),distance_after_px=float(new),gain_px=float(gain))
    return dict(base,offset=shifts[best].tolist() if acceptable else [0.,0.],
        proposed_offset=shifts[best].tolist(),edges=evidence,supporting_connections=supporters,
        reference_distance_before_px=float(aggregate[zero]),reference_distance_after_px=float(aggregate[best]),
        improvement_px=improvement,accepted=bool(acceptable),
        reason='consistent_source_line_support' if acceptable else 'insufficient_consistent_improvement')


def fit(points,models,reference_segments):
    started=perf_counter();series={}
    reference=np.asarray(reference_segments,float).reshape(-1,4)
    reference=reference[np.linalg.norm(reference[:,2:]-reference[:,:2],axis=1)>1e-6]
    for sid,model in sorted(models.items()):
        observed=sorted((p for p in points if p['swatch_id']==sid and
            p.get('original_detection',True) and not p.get('tentative',False) and
            float(p.get('confidence',1.) or 0.)>=.85),key=lambda p:(p['cx'],p['cy']))
        series[sid]=_fit_series(observed,reference,float(model['diameter']))
    result=dict(policy=POLICY,series=series,offsets={sid:d['offset'] for sid,d in series.items()},
        source='frozen initial observed points and fixed plot-scoped L0',
        marker_positions_changed=False,reference_segments_changed=False,per_action_search=False,
        fit_seconds=perf_counter()-started)
    result['sha256']=_digest(result)
    return result


def validate(saved,models,allow_legacy=False):
    if saved.get('policy')!=POLICY and not (allow_legacy and saved.get('policy')=='bw_frozen_connection_anchor_v1'):
        raise ValueError('Unsupported saved connection alignment policy')
    payload={k:v for k,v in saved.items() if k!='sha256'}
    if saved.get('sha256')!=_digest(payload):
        raise ValueError('Saved connection alignment integrity mismatch')
    if set(saved['offsets'])!=set(models):
        raise ValueError('Saved connection alignment swatch roster mismatch')
    for sid,delta in saved['offsets'].items():
        if len(delta)!=2 or not all(math.isfinite(float(v)) for v in delta):
            raise ValueError('Invalid saved connection translation')
        if math.hypot(*delta)>min(8.,.23*float(models[sid]['diameter']))+1e-9:
            raise ValueError('Saved connection translation exceeds marker-size bound')
    return saved['offsets']

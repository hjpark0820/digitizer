"""Identity-safe BW v46 SSIM correction, ported from the endpoint-gated trial.

v45 extracts/refines unlabelled segments; their endpoints gate typed edits.
No ViT or colour-path metric is used. A lower SSIM loss is not marker truth.
All coordinates, masks and exclusive ROI boxes use the supplied image pixels.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter
import zlib

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from bw_suppressed_v46 import decode_marker_mask, encode_marker_mask
from x_singleton_suppressed_v46 import generate as generate_overlap_candidates, same_missing_column

VERSION = 'bw_ssim_step5_v46_v1'
ENDPOINT_TOLERANCE = 18.0
OVERLAP_POLICY = 'frozen_missing_series_ssim_v1'


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (float,np.floating)):
        if not np.isfinite(value): raise ValueError('Nonfinite correction state')
        return float(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, Path): return str(value)
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def key(p): return (p['swatch_id'],round(float(p['cx']),6),round(float(p['cy']),6))
def setkey(points): return tuple(sorted(key(p) for p in points))
def difference(a,b):
    keys={key(p) for p in b}
    return [deepcopy(p) for p in a if key(p) not in keys]


def _box(value, image_shape, default=False):
    if value is None: return (0,0,image_shape[1],image_shape[0]) if default else None
    if len(value)!=4 or any(not math.isfinite(float(v)) or int(v)!=float(v) for v in value):
        raise ValueError('ROI must contain four finite integer pixel coordinates')
    x0,y0,x1,y1=map(int,value)
    if not (0<=x0<x1<=image_shape[1] and 0<=y0<y1<=image_shape[0]):
        raise ValueError('ROI must be inside the image with positive extent')
    return (x0,y0,x1,y1)


def _binding(image,plot,legend):
    return dict(image_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                image_shape=list(image.shape),plot_box=plain(plot),legend_box=plain(legend))


def _clip_interval(segment,box):
    a=np.array(segment[:2],float); delta=np.array(segment[2:],float)-a; lo,hi=0.,1.
    for axis in range(2):
        if abs(delta[axis])<1e-12:
            if not box[axis]<=a[axis]<=box[axis+2]: return None
        else:
            ts=sorted(((box[axis]-a[axis])/delta[axis],(box[axis+2]-a[axis])/delta[axis]))
            lo=max(lo,ts[0]); hi=min(hi,ts[1])
            if hi<=lo: return None
    return lo,hi


def scope_segments(segments,plot,legend):
    result=[]
    for seg in segments:
        interval=_clip_interval(seg,(plot[0],plot[1],plot[2]-1,plot[3]-1))
        if interval is None: continue
        spans=[interval]
        hidden=_clip_interval(seg,(legend[0],legend[1],legend[2]-1,legend[3]-1)) if legend else None
        if hidden:
            lo,hi=interval; a,b=hidden
            if b>lo and a<hi: spans=[(lo,min(hi,a-1e-6)),(max(lo,b+1e-6),hi)]
        p=np.array(seg[:2],float); d=np.array(seg[2:],float)-p
        for lo,hi in spans:
            if hi>lo and np.linalg.norm(d)*(hi-lo)>.5: result.append([*(p+lo*d),*(p+hi*d)])
    return plain(result)


def clustered_grid(points,diameter):
    groups=[]
    for x in sorted(float(p['cx']) for p in points):
        if not groups or x-groups[-1][0]>max(2.,.45*diameter): groups.append([x])
        else: groups[-1].append(x)
    return [float(np.median(g)) for g in groups]


@lru_cache(maxsize=2)
def _legacy_module(filename):
    spec=importlib.util.spec_from_file_location('bw_step5_'+filename.replace('.','_'),Path(__file__).with_name(filename))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare_reference(image,plot,legend,points,diameter,*,no_legend=False):
    """Keep the reviewed v45 preprocessing/segment recipe unchanged."""
    from chart_preprocessing import preprocess
    inclusive=lambda b: (b[0],b[1],b[2]-1,b[3]-1) if b else None
    extra={'auto_legend':False} if no_legend else {}
    prep=preprocess(image,user_plot_area=inclusive(plot),user_legend_box=inclusive(legend),verbose=False,**extra)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); ink=(gray<128).astype('uint8')
    removed=(ink>0)&(np.asarray(prep['clean_fn'](ink))==0)
    reference=image.copy(); reference[removed]=255
    raw=_legacy_module('3_segment_detection_v2.py').detect_debug(reference,prep_info=prep)['segments']
    grid=clustered_grid(points,diameter)
    refined,refinement_log=_legacy_module('4_segment_refinement.py').refine(raw,grid)
    x0,y0,x1,y1=plot
    crop=cv2.cvtColor(reference[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    _blank_legend(crop,plot,legend)
    return crop,dict(raw_segments=plain(raw),refined_segments=scope_segments(refined,plot,legend),
        reference_segments=scope_segments(raw,plot,legend),grid_xs=grid,refinement_log=plain(refinement_log))


def _blank_legend(canvas,plot,legend):
    if legend:
        x0,y0,x1,y1=plot; lx0,ly0,lx1,ly1=legend
        if min(x1,lx1)>max(x0,lx0) and min(y1,ly1)>max(y0,ly0):
            canvas[max(y0,ly0)-y0:min(y1,ly1)-y0,max(x0,lx0)-x0:min(x1,lx1)-x0]=255


def _pack_reference(crop):
    return dict(shape=list(crop.shape),zlib_base64=base64.b64encode(zlib.compress(crop.tobytes())).decode('ascii'),
                sha256=hashlib.sha256(crop.tobytes()).hexdigest())


def _unpack_reference(payload,shape):
    if payload['shape']!=list(shape): raise ValueError('Saved SSIM reference dimensions mismatch')
    raw=zlib.decompress(base64.b64decode(payload['zlib_base64']))
    if len(raw)!=int(np.prod(shape)) or hashlib.sha256(raw).hexdigest()!=payload['sha256']:
        raise ValueError('Saved SSIM reference integrity mismatch')
    return np.frombuffer(raw,np.uint8).reshape(shape).copy()


def _normalise(points,pool,plot,legend,d_est):
    points,pool=deepcopy(points),deepcopy(pool)
    models={}
    for active,rows in ((True,points),(False,pool)):
        for i,p in enumerate(rows,1):
            if not p.get('swatch_id') or not p.get('class_name'):
                raise ValueError('BW v46 Step 5 requires typed swatch IDs; rerun v46 detection')
            p['cx']=float(p.get('cx',p.get('cx_px',float('nan'))));p['cy']=float(p.get('cy',p.get('cy_px',float('nan'))))
            if not (plot[0]<=p['cx']<plot[2] and plot[1]<=p['cy']<plot[3]):
                raise ValueError('Correction point outside the plot')
            if legend and legend[0]<=p['cx']<legend[2] and legend[1]<=p['cy']<legend[3]:
                raise ValueError('Correction point inside the legend')
            if not p.get('marker_mask'):
                raise ValueError('Measured marker mask missing; rerun v46 detection before Step 5')
            mask=decode_marker_mask(p['marker_mask'])
            if not mask.any(): raise ValueError('Empty marker mask')
            for field in ('marker_offset_x','marker_offset_y'):
                p[field]=float(p.get(field,0.))
                if not math.isfinite(p[field]): raise ValueError('Nonfinite marker offset')
            p.setdefault('original_detection',active)
            p.setdefault('point_id',f'P{i:03}' if active else p.get('candidate_id',f'S{i:05}'))
            if not active: p.setdefault('candidate_id',p['point_id'])
            diameter=float(p.get('source_diameter',d_est))
            if not math.isfinite(diameter) or diameter<=0: raise ValueError('Invalid swatch diameter')
            model=models.setdefault(p['swatch_id'],dict(diameter=diameter,class_name=p['class_name']))
            if model['class_name']!=p['class_name'] or abs(model['diameter']-diameter)>1e-6:
                raise ValueError('Inconsistent metadata for one swatch ID')
    if len(setkey(points))!=len(set(setkey(points))) or len(setkey(pool))!=len(set(setkey(pool))):
        raise ValueError('Duplicate same-swatch correction points')
    if set(setkey(points)) & set(setkey(pool)): raise ValueError('Active and suppressed pools overlap')
    return plain(points),plain(pool),models


def point_segments(points,connection_offsets=None):
    rows={}
    for p in points: rows.setdefault(p['swatch_id'],[]).append(p)
    segments=[]
    for sid,values in rows.items():
        ordered=sorted(values,key=lambda p:(p['cx'],p['cy']))
        dx,dy=(connection_offsets or {}).get(sid,(0.,0.))
        segments.extend([[a['cx']+dx,a['cy']+dy,b['cx']+dx,b['cy']+dy] for a,b in zip(ordered,ordered[1:])])
    return segments


@lru_cache(maxsize=1024)
def _stamp(encoded):
    ink=decode_marker_mask(json.loads(encoded)).astype('uint8')
    yy,xx=np.nonzero(ink); hull=np.zeros_like(ink)
    if len(xx)>=3: cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((xx,yy)).astype('int32')),1)
    else: hull=ink.copy()
    return ink,hull


def render_crop(points,plot,legend,connection_offsets=None,connected_series=None):
    x0,y0,x1,y1=plot; canvas=np.full((y1-y0,x1-x0),255,np.uint8)
    line_points=points if connected_series is None else [p for p in points if p['swatch_id'] in connected_series]
    for seg in point_segments(line_points,connection_offsets):
        cv2.line(canvas,(round(seg[0]-x0),round(seg[1]-y0)),(round(seg[2]-x0),round(seg[3]-y0)),0,1,cv2.LINE_8)
    for p in sorted(points,key=lambda p:(p['swatch_id'],p['cx'],p['cy'])):
        ink,hull=_stamp(json.dumps(p['marker_mask'],sort_keys=True));h,w=ink.shape
        left=round(p['cx']-p['marker_offset_x'])-x0-w//2
        top=round(p['cy']-p['marker_offset_y'])-y0-h//2
        xa,ya=max(0,left),max(0,top);xb,yb=min(canvas.shape[1],left+w),min(canvas.shape[0],top+h)
        if xa>=xb or ya>=yb: continue
        target=canvas[ya:yb,xa:xb];cut=(slice(ya-top,yb-top),slice(xa-left,xb-left))
        target[hull[cut]>0]=255;target[ink[cut]>0]=0
    _blank_legend(canvas,plot,legend)
    return canvas


def initial_guard(removed,before,after,models):
    for p in removed:
        if not p.get('original_detection'): continue
        sid=p['swatch_id'];diameter=models[sid]['diameter']
        a=sorted([q for q in before if q['swatch_id']==sid],key=lambda q:q['cx'])
        b=sorted([q for q in after if q['swatch_id']==sid],key=lambda q:q['cx'])
        if len(a)<2 or len(b)<2: continue
        xs=np.linspace(max(a[0]['cx'],p['cx']-2*diameter),min(a[-1]['cx'],p['cx']+2*diameter),41)
        outside=(xs<b[0]['cx'])|(xs>b[-1]['cx'])
        delta=np.abs(np.interp(xs,[q['cx'] for q in a],[q['cy'] for q in a])-
                     np.interp(xs,[q['cx'] for q in b],[q['cy'] for q in b]))
        if not outside.any() and float(delta.max(initial=0))<.25*diameter:
            return False,'preserve_original_negligible_curve_change'
    return True,''


def proposals(active,pool,models,segments,evidence=None):
    trials=[];seen=set()
    def add(action,points,source=None,removed_point=None):
        signature=setkey(points)
        if signature in seen: return
        seen.add(signature);added,removed=difference(points,active),difference(active,points)
        valid,reason=initial_guard(removed,active,points,models)
        for p in added:
            if any(key(q)!=key(p) and same_missing_column(p,q) for q in points):
                valid,reason=False,'missing_series_column_collision'
            if any(q['swatch_id']==p['swatch_id'] and key(q)!=key(p) and
                math.hypot(q['cx']-p['cx'],q['cy']-p['cy'])<.45*models[p['swatch_id']]['diameter'] for q in points):
                valid,reason=False,'same_swatch_collision'
        target=source or removed_point
        options=[(math.hypot(target['cx']-s[j],target['cy']-s[j+1]),i,j) for i,s in enumerate(segments) for j in (0,2)]
        selected,endpoint,distance=None,None,None
        if options:
            distance,i,j=min(options);selected=segments[i];endpoint=list(selected[j:j+2])
        structural=None
        if evidence is not None and action=='REASSIGN':
            structural=evidence.reassignment(removed_point,source,active)
            if not structural['accepted']:
                valid,reason=False,structural['reason']
        elif evidence is not None and action=='DELETE':
            structural=evidence.deletion(removed_point,active)
            if structural.get('protect_observed'):
                valid,reason=False,structural['reason']
        rescued=bool(structural and structural['accepted'])
        if (distance is None or distance>ENDPOINT_TOLERANCE) and not rescued:
            valid,reason=False,'outside_v45_endpoint_neighborhood'
        trials.append(dict(action=action,points=deepcopy(points),candidate_id=source.get('candidate_id') if source else None,
            swatch_id=target['swatch_id'],selected_segment=selected,selected_endpoint=endpoint,endpoint_distance_px=distance,
            candidate_origin='typed_suppressed_hypothesis' if source else 'active_point',
            segment_role='raw_fragment_local_structure' if rescued else 'v45_refined_segment_endpoint_gate',
            admissible=valid,reason=reason,added=added,removed=removed))
        if evidence is not None:trials[-1]['structural_evidence']=structural
    for candidate in pool:
        if key(candidate) in {key(p) for p in active}: continue
        p=deepcopy(candidate);p.update(original_detection=bool(candidate.get('original_detection',False)),tentative=True,
                                      point_id=candidate['candidate_id'],state='active_hypothesis')
        add('ACTIVATE',active+[p],p)
        same=sorted([q for q in active if q['swatch_id']==p['swatch_id']],key=lambda q:math.hypot(q['cx']-p['cx'],q['cy']-p['cy']))
        for q in same[:2]:
            if math.hypot(q['cx']-p['cx'],q['cy']-p['cy'])<=1.5*models[p['swatch_id']]['diameter']:
                add('REPLACE',[a for a in active if key(a)!=key(q)]+[p],p,q)
        if evidence is not None:
            for q in active:
                if (q['swatch_id']!=p['swatch_id'] and
                    math.hypot(q['cx']-p['cx'],q['cy']-p['cy'])<=.6*min(models[q['swatch_id']]['diameter'],models[p['swatch_id']]['diameter'])):
                    add('REASSIGN',[a for a in active if key(a)!=key(q)]+[p],p,q)
    for p in active: add('DELETE',[q for q in active if key(q)!=key(p)],removed_point=p)
    return trials


def _augment_overlap_pool(active,pool,state,plot,legend,diameter,enabled):
    """Augment once, AFTER reference extraction; never change SSIM or its grid.

    Old resumable states migrate using frozen P0, not corrected active points.
    Unknown legend identities without any saved target mask cannot be invented.
    A target stamp is copied from its own best measured template verification,
    including offsets, never from the donor's possibly different symbol.
    """
    if not enabled:
        return pool
    if 'overlap_candidates' in state:
        if state['overlap_candidates'].get('policy') != OVERLAP_POLICY:
            raise ValueError('Unsupported saved overlap candidate policy')
        # Consumed/rejected hypotheses remain in the resumable P/S state.
        return pool
    observed=state.get('initial_points',active)
    records=observed+state.get('initial_suppressed',[])+active+pool
    series={}
    for p in sorted(records,key=lambda p:float(p.get('confidence',0.) or 0.),reverse=True):
        sid=p['swatch_id']
        if sid in series:
            continue
        series[sid]=dict(class_name=p['class_name'],
            source_diameter=float(p.get('source_diameter',diameter)),
            marker_mask=deepcopy(p['marker_mask']),
            marker_offset_x=float(p.get('marker_offset_x',0.)),
            marker_offset_y=float(p.get('marker_offset_y',0.)),
            marker_mask_source='target_series_saved_legend_derived_stamp_not_observed_at_candidate')
        # Keep size provenance together with the copied target-series raster.
        # This is geometry metadata, not evidence that the overlap point exists.
        for field in ('marker_scale','marker_aspect','effective_diameter'):
            if field in p:
                series[sid][field]=p[field]
    if len(series)<2 or not observed:
        state['overlap_candidates']=dict(policy=OVERLAP_POLICY,added_count=0,added=[],reused=[],
            status='no_cross_series_candidates',series_ids=sorted(series))
        return pool
    expanded,audit=generate_overlap_candidates(observed,pool,series,plot,legend)
    # On migration an old correction may already have activated a candidate.
    # Do not place that active identity/location back in the suppressed pool.
    occupied={key(p) for p in active}
    excluded=[p for p in expanded if key(p) in occupied]
    expanded=[p for p in expanded if key(p) not in occupied]
    audit.update(policy=OVERLAP_POLICY,generator_policy=audit['policy'],
        excluded_already_active=[key(p) for p in excluded],
        generation_iteration=int(state.get('total_iterations',0)),series_ids=sorted(series),
        roster_source='frozen observed/suppressed target identities with saved masks',
        measurement_changed=False,reference_grid_changed=False)
    state['overlap_candidates']=plain(audit)
    return expanded


def _write_image(path,image):
    if not cv2.imwrite(str(path),image): raise OSError(f'Cannot write {path}')


def _strip(image,plot,before,after,pool,row):
    x0,y0,x1,y1=plot;source=image[y0:y1,x0:x1].copy();end=source.copy()
    def mark(canvas,p,color,style=cv2.MARKER_CROSS):
        cv2.drawMarker(canvas,(round(p['cx']-x0),round(p['cy']-y0)),color,style,13,2,cv2.LINE_AA)
    for p in row['points_before']: mark(source,p,(70,160,0))
    for p in pool: mark(source,p,(0,150,230),cv2.MARKER_DIAMOND)
    for p in row['added']: mark(end,p,(230,110,0))
    for p in row['removed']: mark(end,p,(0,0,230),cv2.MARKER_TILTED_CROSS)
    panels=[source,cv2.cvtColor(before,cv2.COLOR_GRAY2BGR),cv2.cvtColor(after,cv2.COLOR_GRAY2BGR),end]
    titles=['Before: active / suppressed','Reconstruction BEFORE','Reconstruction AFTER','Original: added / removed']
    tiles=[]
    for panel,title in zip(panels,titles):
        tile=cv2.copyMakeBorder(panel,48,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(tile,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(tile,f"Iter {row['iteration']}: {row['action']}  loss {row['score_before']:.6f} -> {row['score_after']:.6f}",
                    (8,38),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1,cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def run_correction(image_bgr,out_dir,init_points=None,init_suppressed=None,plot_area=None,legend_box=None,
                   d_est=None,max_iter=5,init_state=None,log_fn=print,return_diag_imgs=True,no_legend=False,
                   score_ignore_mask=None,overlap_candidates=True,connection_alignment=None,structural_edits=None,
                   editable_series=None):
    """Correct typed BW hypotheses; optional other-colour pixels abstain in SSIM.

    A nonzero ignore mask removes affected 7x7 comparison windows and is bound
    to saved state by a digest. None/all-zero retain the exact existing score.
    Missing-series overlap hypotheses augment the pool once by default; the
    SSIM reference, endpoint gate and score are unchanged. False is an explicit
    legacy replay switch; it does not remove candidates from an existing state.
    connection_alignment=True opts into frozen per-swatch connector offsets.
    None keeps legacy rendering for old/fresh states, or resumes saved offsets.
    Marker rasters, input coordinates and the SSIM formula remain unchanged.
    structural_edits=True enables pixel/finite-raw-fragment gated reassignment
    and spike removal. None preserves legacy behavior or resumes the saved policy.
    """
    started=perf_counter();image=np.asarray(image_bgr)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3: raise ValueError('Expected uint8 BGR image')
    plot=_box(plot_area,image.shape,True);legend=_box(legend_box,image.shape)
    if min(plot[2]-plot[0],plot[3]-plot[1])<7: raise ValueError('SSIM plot crop must be at least 7x7 pixels')
    budget=5 if max_iter is None else int(max_iter)
    if isinstance(max_iter,bool) or (max_iter is not None and budget!=float(max_iter)) or budget<1:
        raise ValueError('max_iter must be a positive integer')
    diameter=float(d_est if d_est is not None else (init_state or {}).get('diameter',15.))
    if not math.isfinite(diameter) or diameter<=0: raise ValueError('d_est must be positive and finite')
    if not isinstance(overlap_candidates,bool):raise ValueError('overlap_candidates must be bool')
    if connection_alignment is not None and not isinstance(connection_alignment,bool):
        raise ValueError('connection_alignment must be bool or None')
    if structural_edits is not None and not isinstance(structural_edits,bool):
        raise ValueError('structural_edits must be bool or None')
    active,pool,models=_normalise(init_points or [],init_suppressed or [],plot,legend,diameter)
    editable=None if editable_series is None else sorted(set(editable_series))
    if editable is not None and not set(editable).issubset(models):
        raise ValueError('Unknown editable BW series')
    binding=_binding(image,plot,legend)
    score_valid=None
    if score_ignore_mask is not None:
        ignored=np.asarray(score_ignore_mask)
        if ignored.shape!=image.shape[:2] or not np.isfinite(ignored).all():
            raise ValueError('Step-5 ignore mask must be finite and image-aligned')
        ignored=(ignored>.5).astype(np.uint8)
        if ignored.any():
            binding['score_ignore_sha256']=hashlib.sha256(ignored.tobytes()).hexdigest()
            x0,y0,x1,y1=plot
            # SSIM uses a 7x7 window: ignore centres whose comparison would
            # include another colour, not just the occluded centre pixel.
            score_valid=~cv2.dilate(ignored,np.ones((7,7),np.uint8))[y0:y1,x0:x1].astype(bool)
            score_valid[:3]=False;score_valid[-3:]=False;score_valid[:,:3]=False;score_valid[:,-3:]=False
            if score_valid.sum()<49:raise ValueError('Too few independently visible pixels for group SSIM')
    if no_legend:
        if legend is not None:raise ValueError('Explicit no-legend correction conflicts with a legend box')
        binding['explicit_no_legend']=True
    if init_state is not None:
        if init_state.get('version')!=VERSION or init_state.get('binding')!=binding:
            raise ValueError('Saved BW SSIM state belongs to a different image, ROI or algorithm')
        if init_state.get('endpoint_tolerance_px')!=ENDPOINT_TOLERANCE:
            raise ValueError('Saved endpoint policy mismatch')
        state=deepcopy(init_state)
        if state.get('editable_series')!=editable:
            raise ValueError('Cannot change saved BW series routing')
        crop=_unpack_reference(state['reference'],(plot[3]-plot[1],plot[2]-plot[0]))
        if state['models']!=models: raise ValueError('Saved swatch geometry differs from current state')
    else:
        extra={'no_legend':True} if no_legend else {}
        crop,segments=prepare_reference(image,plot,legend,active+pool,diameter,**extra)
        state=dict(version=VERSION,binding=binding,reference=_pack_reference(crop),models=models,diameter=diameter,
            endpoint_tolerance_px=ENDPOINT_TOLERANCE,initial_points=deepcopy(active),initial_suppressed=deepcopy(pool),
            total_iterations=0,**segments)
        state['editable_series']=editable
    pool=_augment_overlap_pool(active,pool,state,plot,legend,diameter,overlap_candidates)
    active,pool,_=_normalise(active,pool,plot,legend,diameter)
    connection_offsets=None
    timing=dict(alignment_seconds=0.,score_seconds=0.,score_evaluations=0,score_cache_hits=0)
    if 'connection_alignment' in state or connection_alignment:
        from bw_connection_alignment_v46 import fit,validate,POLICY
        if 'connection_alignment' in state and connection_alignment is False:
            raise ValueError('Cannot disable connection alignment while resuming an aligned state')
        if 'connection_alignment' in state and state['connection_alignment'].get('policy')!=POLICY:
            validate(state['connection_alignment'],models,allow_legacy=True)
            state['previous_alignment_policy']=state.pop('connection_alignment')['policy']
        if 'connection_alignment' not in state:
            anchor_points=[p for p in state['initial_points'] if editable is None or p['swatch_id'] in editable]
            state['connection_alignment']=fit(anchor_points,models,state['reference_segments'])
            timing['alignment_seconds']=state['connection_alignment']['fit_seconds']
        connection_offsets=validate(state['connection_alignment'],models)
        log_fn(f'[v46 BW Step5] Frozen connection offsets (marker positions unchanged): {connection_offsets}')
    evidence=None
    if 'structural_edits' in state or structural_edits:
        from bw_structural_edits_v46 import Evidence,frozen_policy
        expected=frozen_policy(state['reference_segments'])
        if 'structural_edits' in state:
            if structural_edits is False:
                raise ValueError('Cannot disable structural edits while resuming an enabled state')
            if state['structural_edits']!=expected:
                raise ValueError('Saved structural evidence policy or raw reference mismatch')
        state['structural_edits']=expected
        evidence=Evidence(image,models,state['reference_segments'],plot,legend,score_ignore_mask)
        log_fn('[v46 BW Step5] Local structural edits enabled: raw fragments + marker pixels; SSIM unchanged.')
    def render(points):return render_crop(points,plot,legend,connection_offsets,editable)
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    dest=Path(tempfile.mkdtemp(prefix='bw_ssim_',dir=str(out_dir)))
    _write_image(dest/'reference.png',crop)
    cache={}
    def score(points):
        signature=setkey(points)
        if signature not in cache:
            tick=perf_counter()
            if score_valid is None:
                cache[signature]=1-float(structural_similarity(crop,render(points),data_range=255))
            else:
                _,score_map=structural_similarity(crop,render(points),data_range=255,full=True)
                cache[signature]=1-float(score_map[score_valid].mean())
            timing['score_seconds']+=perf_counter()-tick
            timing['score_evaluations']+=1
        else:
            timing['score_cache_hits']+=1
        return cache[signature]
    initial_points=deepcopy(active);initial_pool=deepcopy(pool);initial_score=score(active)
    _write_image(dest/'reconstruction_initial.png',render(active))
    history=[];trace=[];diag=[]
    log_fn(f'[v46 BW Step5] SSIM; P={len(active)}, S={len(pool)}, endpoint radius=18px, max_iter={budget}')
    if state.get('overlap_candidates'):
        overlap=state['overlap_candidates']
        log_fn(f'[v46 BW Step5] Frozen same-x overlap pool: {overlap["added_count"]} generated; '
               f'{len(overlap["reused"])} reused. SSIM/reference/grid unchanged; candidates are not confirmed markers.')
    log_fn('[v46 BW Step5] Score improvement does not confirm marker identity; v45 preprocessing reference is retained.')
    for number in range(1,budget+1):
        before=deepcopy(active);before_pool=deepcopy(pool);baseline=score(before)
        trials=proposals(active,pool,models,state['refined_segments'],evidence)
        if editable is not None:
            for trial in trials:
                changed=difference(trial['points'],active)+difference(active,trial['points'])
                if any(p['swatch_id'] not in editable for p in changed):
                    trial.update(admissible=False,reason='nonconnected_or_uncertain_series_frozen')
        for t in trials: t['score']=score(t['points']) if t['admissible'] else None
        eligible=[t for t in trials if t['admissible']]
        winner=min(eligible,key=lambda t:(t['score'],t['action'],setkey(t['points']))) if eligible else None
        improved=winner is not None and winner['score']<baseline-1e-7
        if improved:
            active=deepcopy(winner['points']);keys={key(p) for p in active}
            pool=[p for p in pool if key(p) not in keys]
            for p in difference(before,active):
                if key(p) not in {key(q) for q in pool}:
                    p.update(candidate_id=p.get('candidate_id') or p['point_id'],state='suppressed',suppression_reason='step5_deactivated')
                    pool.append(p)
        row=dict(iteration=state['total_iterations']+number,run_iteration=number,action=winner['action'] if improved else 'NONE',
            improved=improved,points_before=before,points_after=deepcopy(active),suppressed_before=before_pool,
            suppressed_after=deepcopy(pool),added=difference(active,before),removed=difference(before,active),
            score_before=baseline,score_after=score(active),stop_reason='' if improved else 'no_admissible_improvement',
            trials=[{k:v for k,v in t.items() if k!='points'} for t in trials])
        for field in ('selected_segment','selected_endpoint','endpoint_distance_px','candidate_origin','candidate_id','structural_evidence','segment_role'):
            row[field]=winner.get(field) if improved else None
        before_render=render(before);after_render=render(active)
        before_path=dest/f'iter{number:02}_before.png';after_path=dest/f'iter{number:02}_after.png'
        _write_image(before_path,before_render);_write_image(after_path,after_render)
        strip=_strip(image,plot,before_render,after_render,before_pool,row);strip_path=dest/f'iter{number:02}_comparison.png'
        _write_image(strip_path,strip)
        row.update(reconstruction_before_path=str(before_path),reconstruction_after_path=str(after_path),comparison_path=str(strip_path))
        trace.append(row);history.append((row['iteration'],row['action'],len(active),row['score_after'],improved))
        if return_diag_imgs: diag.append(dict(title=f"Stage 5 BW SSIM iter {row['iteration']}: {row['action']}",img_bgr=strip))
        log_fn(f"[v46 BW Step5] iter={row['iteration']} {row['action']} P={len(active)} 1-SSIM={row['score_after']:.8f}")
        if not improved: break
    state['total_iterations']+=len(trace)
    state.update(P_current=deepcopy(active),S_current=deepcopy(pool),last_score=score(active))
    _write_image(dest/'reconstruction_final.png',render(active))
    report=dict(version=VERSION,metric='1-SSIM',status='completed',initial_points=initial_points,initial_suppressed=initial_pool,
        P_current=active,S_current=pool,initial_score=initial_score,final_score=score(active),iterations=trace,
        max_iterations=budget,iterations_executed=len(trace),seconds=perf_counter()-started,
        warning='Unvalidated image hypotheses: lower SSIM loss is not marker accuracy.')
    report['overlap_candidates']=deepcopy(state.get('overlap_candidates'))
    report['connection_alignment']=deepcopy(state.get('connection_alignment'))
    report['structural_edits']=deepcopy(state.get('structural_edits'))
    report['timing']=timing
    save(dest/'trace.json',report);save(dest/'runtime_state.json',state)
    return dict(P_current=active,S_current=pool,mode_xs=state['grid_xs'],runtime_state=state,
        history=history,diag_imgs=diag,steps=diag,trace=trace,status='completed',artifact_dir=str(dest),
        initial_score=initial_score,final_score=score(active),seconds=perf_counter()-started,timing=timing)

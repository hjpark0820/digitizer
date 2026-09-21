"""Native-pixel, marker-free measurement hypotheses for the v46 colour route.

This is the evaluated method-6 -> source support gate -> endpoint pipeline.
Inputs are current image pixels and explicit series identities, never experiment
results or numerical target points. Boxes are half-open source-image xyxy.
"""
from collections import Counter
from copy import deepcopy
import math
from time import perf_counter

import cv2
import numpy as np

from color_tentative_v46 import support_intervals
from .grid_likelihood_v1 import horizontal_grid_likelihood
from .grid_aware_path_v1 import trace_grid_aware
from .errorbar_neutral_evidence import build_neutral_evidence
from .errorbar_stems import propose_stems
from .errorbar_centers import estimate_centers
from .path_bar_kink_detector_v2 import detect_series as detect_v2
from .path_bar_kink_detector_v3 import detect_series as detect_v3, reconcile_series
from .path_bar_assignment_guard_v3 import preserve_competing_bar_assignments
from .type3_tick_text import repeated_tick_text
from .type3_path_support_gate import gate_path
from .type3_endpoint_candidates import test_path_endpoints, ENDPOINT_PRIOR_POLICY

VERSION = 'type3_detection_v46_v1'


def _independent_candidate(q, sid, width, chromatic):
    selected=q.get('selected') or {};left=selected.get('left');right=selected.get('right')
    bilateral=bool(left and right and not selected.get('cap_confounded') and not selected.get('neighboring_cap_confounded'))
    stem=deepcopy(q['stem'])
    if chromatic:stem['id']=sid+'_'+stem['id']
    strict=any(c.get('strict',False) for c in stem.get('caps',[]))
    curve=dict(bilateral=bilateral,one_side=bool(q.get('left_fits') or q.get('right_fits')),
        bend_degrees=abs(math.degrees(math.atan(right['slope'])-math.atan(left['slope']))) if left and right else 0.)
    for name,fit in [('left',left),('right',right)]:
        curve[name]=dict(fit=fit,source_columns=fit.get('source_columns',0) if fit else 0)
    return dict(id='I_'+q['id'],x=q['x'],y=q['y'],stem=stem,
        bar=dict(status='supported' if strict else 'tentative',caps=stem.get('caps',[]),reason=q['reason'],structural_score=0.),
        kink=None,curve=curve,independent_center=deepcopy(q),origin='independent_off_stem_fits',
        status='accepted' if q['status']=='accepted' else 'tentative',reason=q['reason'],
        score=7. if q['status']=='accepted' else 1.,structural_color_source='same_series' if chromatic else 'neutral',
        interpretation='Observed vertical structure and independent source curve arms; no path or midpoint needed')


def _merge_candidates(path_result, independent, width, sid, chromatic):
    combined=deepcopy(path_result)
    for q in combined['candidates']:
        q['pre_merge_id']=q['id']
        if chromatic and q.get('stem'):q['structural_color_source']='same_series_or_neutral'
    combined['candidates'].extend(_independent_candidate(q,sid,width,chromatic)
        for q in independent['candidates'] if q['y'] is not None)
    rank={'accepted':2,'tentative':1,'rejected':0};chosen=[]
    for p in sorted(combined['candidates'],key=lambda q:(rank[q['status']],q.get('score',0)),reverse=True):
        if p['status']=='rejected':continue
        near=next((q for q in chosen if abs(q['x']-p['x'])<=max(3.,1.5*width) and abs(q['y']-p['y'])<=2*width),None)
        if near:
            p['pre_duplicate_status']=p['status'];p.update(status='rejected',reason='duplicate_local_type3_center',merged_into_x=near['x'])
        else:chosen.append(p)
    combined['candidates'].sort(key=lambda q:(q['x'],q['y']))
    for i,p in enumerate(combined['candidates'],1):p['id']=f'C{i:03d}'
    combined['points']=[p for p in combined['candidates'] if p['status']=='accepted']
    combined['tentative_points']=[p for p in combined['candidates'] if p['status']=='tentative']
    return combined


def _endpoint_point(c):
    return dict(id=c['id'],series_id=c['series_id'],x_px=c['x_px'],y_px=c['y_px'],
        status='accepted' if c['state']=='active' else 'tentative',reason=c['reason'],
        kind='measurement_center_hypothesis',origin='source_tested_global_path_endpoint',
        endpoint_evidence=deepcopy(c['evidence']),evidence=deepcopy(c['evidence']),
        evidence_coordinate_system='plot-local native pixels',marker_glyph_detected=False,
        endpoint_side=c['side'],step5_eligible=c['step5_eligible'])


def spatial_valid(plot, legend=None):
    x0,y0,x1,y1=plot
    valid=np.ones((y1-y0,x1-x0),bool)
    if legend is not None:
        a,b=max(legend[0],x0)-x0,max(legend[1],y0)-y0
        c,d=min(legend[2],x1)-x0,min(legend[3],y1)-y0
        if a<c and b<d:valid[b:d,a:c]=False
    return valid


def detect(image_bgr, plot_box, legend_box, series, own_masks, soft_masks, *, valid=None):
    """Run identical rules for all series; no antibody or file-name branches.

    `series` contains id/rgb/line_width and optional label/role; masks are plot
    local. The caller must explicitly select a measurement-connecting line-only
    chart. A line-only legend does not establish absence of plot marker glyphs.
    """
    started=perf_counter();image=np.asarray(image_bgr)
    plot=list(map(int,plot_box));x0,y0,x1,y1=plot
    if not (0<=x0<x1<=image.shape[1] and 0<=y0<y1<=image.shape[0]):
        raise ValueError('Type3 plot box must be a nonempty image-contained half-open box')
    legend=None if legend_box is None else list(map(int,legend_box))
    crop=image[y0:y1,x0:x1];shape=crop.shape[:2]
    allowed=spatial_valid(plot,legend)
    if valid is not None:
        if np.shape(valid)!=shape:raise ValueError('Type3 valid mask shape differs from plot')
        allowed &= np.asarray(valid,bool)
    metadata=deepcopy(series)
    ids=[s['id'] for s in metadata]
    if len(set(ids))!=len(ids):raise ValueError('Type3 series identities must be unique')
    for s in metadata:
        s.setdefault('role','data_series');s.setdefault('label',s['id'])
        width=float(s['line_width'])
        if not np.isfinite(width) or width<=0:raise ValueError('Invalid source stroke width')
        sid=s['id']
        if np.shape(own_masks[sid])!=shape or np.shape(soft_masks[sid])!=shape:
            raise ValueError('Type3 colour mask shape differs from plot')
    grid,grid_report=horizontal_grid_likelihood(crop,allowed)
    neutral=build_neutral_evidence(crop,allowed,line_width=1.)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    colored=(hsv[:,:,1]>=40)&(hsv[:,:,2]<250)&allowed
    dark=(255-crop.astype(float).mean(axis=2))/255.
    result=dict(version=VERSION,case_id='v46_line_only',display_name='Line-only chart',
        plot_box=plot,legend_box=legend,series=metadata,paths=[],points=[],tentative_points=[],candidates=[],
        method='method6_grid_cost_source_centers_support_gate_endpoint_strong_prior',
        native_resolution=True,marker_glyph_count=0,step5_run=False,
        endpoint_suppressed_policy=ENDPOINT_PRIOR_POLICY,
        warnings=['Measurement-center hypotheses, not detected marker glyphs or reference-verified data.',
            'Suppressed whole-path endpoints are strong by user prior, not new source evidence.',
            'Unsupported raw path samples are retained diagnostically but excluded from correction.'])
    diagnostics={}
    for s in metadata:
        if s['role']=='reference_line':continue
        sid=s['id'];width=s['line_width'];own=np.asarray(own_masks[sid],bool)
        soft=np.asarray(soft_masks[sid])*allowed
        intervals=support_intervals(own&allowed)
        if not intervals:
            s.update(accepted_count=0,tentative_count=0);continue
        tr=trace_grid_aware(soft,grid,intervals[0][0],intervals[-1][1]-1,weight=1.)
        xy=tr['path'];ix=np.rint(xy[:,0]).astype(int);iy=np.rint(xy[:,1]).astype(int)
        pvalid=allowed[iy,ix];observed=tr['nongrid_color_observed']&pvalid;psource=xy+[x0,y0]
        result['paths'].append(dict(series_id=sid,raw_path_source=psource,raw_observed=observed,
            grid_likelihood=tr['grid_likelihood'],valid=pvalid,
            samples=[dict(x_px=float(x),y_px=float(y),observed=bool(o)) for (x,y),o in zip(psource,observed)],
            path_color_values=tr['val'],cost=tr['objective']))
        chromatic=max(s['rgb'])-min(s['rgb'])>25
        structure=(own|neutral['own']) if chromatic else neutral['own']
        stemset=propose_stems(structure,allowed,1.)
        for st in stemset['stems']:st['id']=sid+'_proposal_'+st['id']
        d2=detect_v2(xy,observed,own,soft,allowed,stemset['stems'],structure,dark,width,
            grid_mask=grid>=.5,colored_ink_mask=colored)
        d3=detect_v3(xy,observed,own,soft,allowed,structure,dark,width,d2,
            grid_mask=grid>=.5,colored_ink_mask=colored,polyline_mode=True)
        same=propose_stems(own,allowed,width)
        independent=estimate_centers(own,allowed,same,width,soft=soft)
        diagnostics[sid]=_merge_candidates(d3,independent,width,sid,chromatic)
        print(f'[v46 type3] {sid}: path {len(xy)} pixels; source hypotheses prepared',flush=True)
    diagnostics,physical=reconcile_series(diagnostics)
    diagnostics,assignment=preserve_competing_bar_assignments(diagnostics)
    for s in metadata:
        sid=s['id']
        if sid not in diagnostics:continue
        d=diagnostics[sid]
        for p in d['candidates']:
            q=dict(id=sid+'_'+p['id'],series_id=sid,x_px=p['x']+x0,y_px=p['y']+y0,
                status=p['status'],reason=p['reason'],kind='measurement_center_hypothesis',
                origin=p['origin'],evidence=deepcopy(p),evidence_coordinate_system='plot-local native pixels',
                marker_glyph_detected=False)
            result['candidates'].append(q)
            if p['status']=='accepted':result['points'].append(q)
            elif p['status']=='tentative':result['tentative_points'].append(q)
        s.update(accepted_count=len(d['points']),tentative_count=len(d['tentative_points']))
    result.update(grid_report=grid_report,physical_bar_audit=physical,assignment_audit=assignment)
    text,text_audit=repeated_tick_text(image,plot,allowed)
    active_series={s['id']:s for s in metadata if s['role']!='reference_line'}
    other_masks={};paths=[]
    for path in result['paths']:
        sid=path['series_id'];own=np.asarray(own_masks[sid],bool)&allowed;other=np.zeros_like(own)
        for other_sid in active_series:
            if other_sid!=sid:other |= np.asarray(own_masks[other_sid],bool)
        other &= allowed & ~own & ~text & (grid<.5);other_masks[sid]=other
        updated=gate_path(path,image,plot,own=own,other_ink=other,grid=grid,text_mask=text,
            line_width=active_series[sid]['line_width'])
        updated.setdefault('original_raw_observed',deepcopy(path['raw_observed']))
        paths.append(updated)
    result.update(paths=paths,text_gate_report=text_audit)
    shared=propose_stems(neutral['own']&~text,allowed,1.)['stems'];endpoints=[]
    for path in paths:
        sid=path['series_id'];width=active_series[sid]['line_width'];own=np.asarray(own_masks[sid],bool)
        own_stems=propose_stems(own&~text&(grid<.5),allowed,width)['stems']
        saved=[deepcopy(p['evidence']['stem']) for p in result['candidates']
            if p['series_id']==sid and (p.get('evidence') or {}).get('stem')]
        stems=[];seen=set()
        for st in own_stems+saved+shared:
            token=tuple(round(float(st[k]),2) for k in ('x','y0','y1'))
            if token not in seen:stems.append(st);seen.add(token)
        endpoints.extend(test_path_endpoints(path,image_bgr=image,plot_box=plot,own=own,soft=soft_masks[sid],
            valid=allowed,grid=grid,text_mask=text,other_ink=other_masks[sid],line_width=width,stems=stems))
    result['initial_source_points']=deepcopy(result['points']);merges=[]
    for c in endpoints:
        width=active_series[c['series_id']]['line_width']
        near=next((p for p in result['points'] if p['series_id']==c['series_id'] and
            abs(p['x_px']-c['x_px'])<=max(2,1.5*width) and abs(p['y_px']-c['y_px'])<=max(2,1.5*width)),None)
        if near is not None:
            c.update(coincident_initial_id=near['id'],pool_action='existing_active_preserved_no_duplicate')
            merges.append(dict(candidate_id=c['id'],existing_id=near['id']))
        elif c['state']=='active':
            result['points'].append(_endpoint_point(c));c['pool_action']='added_active_source_endpoint'
        elif c['state']=='suppressed':
            result['tentative_points'].append(_endpoint_point(c));c['pool_action']='suppressed_endpoint_hypothesis'
        else:c['pool_action']='rejected_no_pool_entry'
    result.update(endpoint_candidates=endpoints,endpoint_duplicates=merges,
        endpoint_state_counts=dict(Counter(c['state'] for c in endpoints)),seconds=perf_counter()-started)
    return result,dict(grid_likelihood=grid,text_mask=text,valid=allowed,other_masks=other_masks)

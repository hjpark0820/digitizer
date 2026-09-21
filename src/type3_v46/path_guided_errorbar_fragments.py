"""Paired neutral cap/vertical-fragment evidence around a supplied path bend.

No stem proposal, minimum five-pixel vertical run, safe-deletion mask, marker
template, target coordinate, or antibody identity is used. Missing middle rows
can be explained by actual chromatic ink, but chromatic ink is never counted
as positive vertical evidence. Outputs remain measurement-center hypotheses.
All coordinates are plot-local; bounding boxes and cap extrema are inclusive.
"""
from __future__ import annotations

import math
import numpy as np

from .path_errorbar_evidence import _runs, _closed_parallel_structure

VERSION = 'path_guided_paired_caps_v1'


def _horizontal_caps(mask, kink_x, path_y, lw, cw):
    h,w=mask.shape
    xreach=max(4,int(math.ceil(2*cw)));yreach=max(6,int(math.ceil(4*cw)))
    max_width=max(25,int(math.ceil(24*lw))+1)
    min_width=max(5,int(math.ceil(4*lw))+1)
    max_thickness=max(3,int(math.ceil(2.5*lw)))
    xa=max(0,int(math.floor(kink_x))-xreach-max_width-1)
    xb=min(w,int(math.ceil(kink_x))+xreach+max_width+2)
    ya=max(0,int(math.floor(path_y))-yreach);yb=min(h,int(math.ceil(path_y))+yreach+1)
    rows=[];wide=0
    for y in range(ya,yb):
        for run in _runs(np.flatnonzero(mask[y,xa:xb])):
            a,b=int(run[0]+xa),int(run[-1]+xa);width=b-a+1
            cx=.5*(a+b)
            if abs(cx-kink_x)>xreach:continue
            if width>max_width:wide+=1;continue
            if width<min_width:continue
            rows.append(dict(y=y,x0=a,x1=b,center=cx))
    groups=[]
    for row in rows:
        matches=[g for g in groups if g[-1]['y']==row['y']-1 and
                 abs(g[-1]['center']-row['center'])<=max(2,2*lw) and
                 min(g[-1]['x1'],row['x1'])>=max(g[-1]['x0'],row['x0'])]
        if matches:min(matches,key=lambda g:abs(g[-1]['center']-row['center'])).append(row)
        else:groups.append([row])
    caps=[];rejected=[]
    for group in groups:
        a,b=min(r['x0'] for r in group),max(r['x1'] for r in group)
        top,bottom=group[0]['y'],group[-1]['y']
        cap=dict(x0=a,x1=b,y0=top,y1=bottom,y=float(np.median([r['y'] for r in group])),
                 x=float(np.median([r['center'] for r in group])),width=b-a+1,
                 thickness=bottom-top+1,kind='observed_horizontal_cap_fragment')
        columns=[]
        for x in range(a,b+1):
            ys=np.flatnonzero(mask[top:bottom+1,x])+top
            if len(ys):columns.append([x,float(np.median(ys))])
        slope=float(np.polyfit(*np.asarray(columns).T,1)[0]) if len(columns)>=3 else 0.
        cap['horizontal_slope']=slope
        if bottom-top+1>max_thickness:cap['rejection']='thick_horizontal_body';rejected.append(cap)
        elif b-a+1>max_width:cap['rejection']='wide_horizontal_body';rejected.append(cap)
        elif abs(slope)>.30:cap['rejection']='diagonal_horizontal_fragment';rejected.append(cap)
        else:caps.append(cap)
    return caps,dict(search_bbox=[xa,ya,xb,yb],candidate_center_x_radius=xreach,
                    candidate_y_radius=yreach,maximum_cap_width=max_width,
                    maximum_cap_thickness=max_thickness,too_wide_rows=wide,rejected_caps=rejected)


def _physical_id(top,bottom):
    # Source geometry, not curve identity or a supplied kink's x coordinate.
    fmt=lambda c:'_'.join(str(int(c[k])) for k in ['x0','y0','x1','y1'])
    return 'cap_pair_'+fmt(top)+'__'+fmt(bottom)


def _pair(top,bottom,path,mask,colored,lw,cw):
    h,w=mask.shape
    alignment=abs(top['x']-bottom['x'])
    if alignment>max(3,2*lw):return None
    if max(top['width'],bottom['width'])/min(top['width'],bottom['width'])>3:return None
    ya,yb=top['y1']+1,bottom['y0']-1
    if yb<ya:return None
    base_x=.5*(top['x']+bottom['x'])
    maxwidth=max(3,int(math.ceil(2.5*lw)))
    # Select a narrow actually observed core, never merely the nearest kink.
    alternatives=[]
    for x in range(max(0,int(math.floor(base_x))-int(math.ceil(lw))),min(w,int(math.ceil(base_x))+int(math.ceil(lw))+1)):
        if not path[0,0]<=x<=path[-1,0]:continue
        py=float(np.interp(x,path[:,0],path[:,1]))
        if not top['y1']<py<bottom['y0']:continue
        if min(x-top['x0'],top['x1']-x,x-bottom['x0'],bottom['x1']-x)<1:continue
        rows=[];broad=[];missing=[];covered=[]
        for y in range(ya,yb+1):
            # Width is measured on the original neutral row, not a narrow
            # crop that could turn a horizontal stroke into a fake upright.
            xa=max(0,x-maxwidth-1);xb=min(w,x+maxwidth+2)
            runs=_runs(np.flatnonzero(mask[y,xa:xb]))
            touching=[r for r in runs if r[0]+xa<=x<=r[-1]+xa]
            if touching:
                run=touching[0]
                if len(run)<=maxwidth:rows.append(y)
                else:broad.append(y);missing.append(y)
            else:missing.append(y)
            if y in missing and colored[y,x]:covered.append(y)
        unexplained=[y for y in missing if y not in covered]
        # More observed neutral rows wins; fewer unexplained rows breaks ties.
        alternatives.append((len(rows),-len(unexplained),-abs(x-base_x),x,py,rows,broad,covered,unexplained))
    if not alternatives:return None
    _,_,_,x,py,rows,broad,covered,unexplained=max(alternatives,key=lambda a:a[:3])
    spans=[[int(g[0]),int(g[-1])] for g in _runs(rows)]
    result=dict(status='rejected',reason=None,structural_score=0.,x=float(x),path_y=py,
                caps=[top,bottom],vertical=dict(observed_rows=rows,observed_runs=spans,
                    observed_count=len(rows),between_cap_row_count=yb-ya+1,
                    chromatic_occlusion_rows=covered,unexplained_missing_rows=unexplained,
                    broad_neutral_rows=broad,maximum_core_width=maxwidth),
                diagnostics=dict(cap_center_misalignment=alignment,pixels_filled=0,pixels_erased=0,
                    chromatic_ink_is_positive_vertical_evidence=False),
                stem=dict(id=_physical_id(top,bottom),x=float(x),x0=x,x1=x,
                    y0=top['y0'],y1=bottom['y1'],width=1,caps=[top,bottom],
                    proposal_kind='path_guided_paired_cap_fragments'))
    closed=_closed_parallel_structure(mask,x,top['y0'],bottom['y1'],[top,bottom],lw)
    result['diagnostics']['closed_structure']=closed
    if closed:result['reason']='closed_or_digit_like_parallel_structure'
    elif not rows:result['reason']='parallel_caps_without_vertical_evidence'
    elif broad:result['reason']='broad_middle_ink_not_narrow_stem'
    elif unexplained:result['reason']='unexplained_missing_vertical_rows'
    elif len(rows)<max(2,int(math.ceil(2*lw))):
        result.update(status='tentative',reason='paired_caps_with_insufficient_vertical_fragments',structural_score=.5)
    else:
        result.update(status='supported',reason='paired_caps_observed_fragments_and_chromatic_occlusion',
                      structural_score=float(.6+.25*min(1,len(rows)/max(4,4*lw))+.15*(1-alignment/max(3,2*lw))))
    return result


def score_fragments_near_kink(kink_x,path_xy,neutral_mask,colored_ink_mask,line_width=1.,curve_width=4.):
    """Score local paired caps without requiring any pre-existing stem.

    ``colored_ink_mask`` must mean observed CHROMATIC source ink (all colors),
    not a filled curve model, brightness mask, or uncertain palette assignment.
    Its only use is explaining absent neutral middle rows. Caller owns valid
    plot masking, same-series pixel/bend evidence, and ambiguous-series policy.
    """
    mask=np.asarray(neutral_mask);colored=np.asarray(colored_ink_mask);path=np.asarray(path_xy,float)
    if mask.ndim!=2 or mask.dtype!=np.bool_ or colored.dtype!=np.bool_ or colored.shape!=mask.shape:
        raise ValueError('neutral and chromatic masks must be matching bool[H,W] arrays')
    lw,cw,kx=float(line_width),float(curve_width),float(kink_x)
    if not np.isfinite([lw,cw,kx]).all() or min(lw,cw)<=0:raise ValueError('Finite positive widths and finite kink_x required')
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all() or np.any(np.diff(path[:,0])<=0):
        raise ValueError('path_xy must be finite, with strictly increasing x')
    result=dict(version=VERSION,status='rejected',reason='no_aligned_paired_caps',structural_score=0.,
                x=kx,path_y=None,caps=[],vertical={},diagnostics=dict(pixels_filled=0,pixels_erased=0),stem=None,candidates=[])
    if not path[0,0]<=kx<=path[-1,0]:result['reason']='kink_outside_path';return result
    py=float(np.interp(kx,path[:,0],path[:,1]));result['path_y']=py
    if not (0<=kx<mask.shape[1] and 0<=py<mask.shape[0]):result['reason']='kink_outside_mask';return result
    caps,diagnostics=_horizontal_caps(mask,kx,py,lw,cw)
    result['diagnostics'].update(diagnostics)
    pairs=[]
    for top in caps:
        for bottom in caps:
            if top['y1']>=bottom['y0'] or not top['y']<py<bottom['y']:continue
            pair=_pair(top,bottom,path,mask,colored,lw,cw)
            if pair is not None:pairs.append(pair)
    if not pairs:return result
    rank={'supported':2,'tentative':1,'rejected':0}
    pairs.sort(key=lambda p:(rank[p['status']],p['structural_score'],-abs(p['x']-kx)),reverse=True)
    result.update({k:v for k,v in pairs[0].items() if k!='diagnostics'})
    result['diagnostics'].update(pairs[0]['diagnostics']);result['candidates']=pairs
    viable=[p for p in pairs if p['status']!='rejected']
    if len(viable)>1:
        result.update(status='tentative',reason='multiple_compatible_cap_pairs')
        result['diagnostics']['alternative_physical_ids']=[p['stem']['id'] for p in viable]
    result['diagnostics']['score_is_probability']=False
    result['diagnostics']['limitation']='I-shaped text and error bars need independent curve/text context; no marker glyph is detected.'
    return result

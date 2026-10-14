"""Bounded source-pixel completion of a vertically truncated legend handle.

This is image geometry only: no marker class, data points or fitted template is
used. Source pixels and the caller's horizontal/series identity stay unchanged.
It deliberately cannot repair an attached frame, a word, or a large missing body.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def completion_candidates(image, box, peers=(), *, bounds=None):
    """Return small, evidence-backed vertical crop expansions, then diagnostics.

    Bounds use half-open pixel coordinates. Unrelated components and adjacent
    swatch rows constrain expansion; there is no blanket padding of every box.
    Consumers must still perform their normal line/body/shape validation.
    """
    a,b,c,d=map(int,box)
    height,width=image.shape[:2]
    if not (0<=a<c<=width and 0<=b<d<=height):
        raise ValueError(f'Invalid swatch crop: {box}')
    budget=max(2,min(6,int(math.ceil(.25*(d-b)))))
    limit_top,limit_bottom=max(0,b-budget),min(height,d+budget)
    # Two context rows beyond the allowed output crop distinguish a complete
    # stroke ending at the budget from one still extending farther away.
    top,bottom=max(0,limit_top-2),min(height,limit_bottom+2)
    if bounds is not None:
        e,f,g,h=map(int,bounds)
        if not (e<=a<c<=g and f<=b<d<=h):
            raise ValueError('Swatch must be inside measured context bounds')
        limit_top,limit_bottom=max(limit_top,f),min(limit_bottom,h)
        top,bottom=max(top,f),min(bottom,h)
    for peer in peers:
        e,f,g,h=map(int,peer)
        if (e,f,g,h)==(a,b,c,d) or min(c,g)<=max(a,e):continue
        if h<=b:limit_top=max(limit_top,int(math.ceil((h+b)/2)))
        elif f>=d:limit_bottom=min(limit_bottom,int(math.floor((d+f)/2)))
        elif f<b:limit_top=b
        elif h>d:limit_bottom=d
    diag=dict(policy='bounded_connected_ink_completion_v1',input_box=[a,b,c,d],
              vertical_budget_px=budget,search_box=[a,top,c,bottom],
              output_limits_y=[limit_top,limit_bottom],strong_outside=False)
    gray=cv2.cvtColor(image[top:bottom,a:c],cv2.COLOR_BGR2GRAY)
    strong=gray<235
    n,labels,stats,_=cv2.connectedComponentsWithStats(strong.astype(np.uint8),8)
    inner=np.zeros(gray.shape,bool)
    inset=max(0,int(.15*(c-a)))
    inner[b-top:d-top,inset:max(inset+1,c-a-inset)]=True
    choices=[(int(np.sum((labels==i)&inner)),int(stats[i,cv2.CC_STAT_AREA]),i)
             for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=3]
    if not choices or max(choices)[0]<2:
        return [],dict(diag,reason='no_independent_seed')
    _,_,chosen=max(choices)
    core=labels==chosen
    yy,xx=np.nonzero(core)
    cy0,cy1=int(yy.min())+top,int(yy.max())+top+1
    outside=cy0<b or cy1>d
    diag['strong_outside']=outside
    if cy0<limit_top or cy1>limit_bottom:
        return [],dict(diag,reason='connected_ink_exceeds_budget')
    # The extension must touch the original seed through more than one pixel.
    # A connector or isolated JPEG speck alone is not a cropped marker body.
    joins=[]
    if cy0<b and b>top:
        joins.append(int(np.sum(core[b-top-1] & cv2.dilate(core[b-top:b-top+1].astype(np.uint8),np.ones((1,3),np.uint8))[0].astype(bool))))
    if cy1>d and d<bottom:
        joins.append(int(np.sum(core[d-top] & cv2.dilate(core[d-top-1:d-top].astype(np.uint8),np.ones((1,3),np.uint8))[0].astype(bool))))
    if joins and sum(joins)<2:
        return [],dict(diag,reason='insufficient_border_continuation')
    soft=core | ((gray<248) & cv2.dilate(core.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool))
    sy=np.flatnonzero(soft.any(axis=1))+top
    sy0,sy1=int(sy.min()),int(sy.max())+1
    if not outside and sy0>=b and sy1<=d:
        return [],dict(diag,reason='no_connected_ink_outside')
    candidates=[]
    for y0,y1 in [(cy0,cy1),(sy0,sy1),(sy0-1,sy1),(sy0,sy1+1),(sy0-1,sy1+1)]:
        y0,y1=max(limit_top,min(b,y0)),min(limit_bottom,max(d,y1))
        candidate=(a,y0,c,y1)
        if candidate==(a,b,c,d) or candidate in candidates:continue
        added=np.zeros_like(core)
        added[y0-top:b-top]=True;added[d-top:y1-top]=True
        # Refuse to recruit another substantial body/frame/text component.
        intruders=[i for i in range(1,n) if i!=chosen and stats[i,cv2.CC_STAT_AREA]>=3
                   and np.any((labels==i)&added) and np.sum(labels[b-top:d-top]==i)<3]
        if intruders:continue
        candidates.append(candidate)
    diag.update(reason='measured_connected_extension' if candidates else 'neighbour_blocks_extension',
                strong_extent_y=[cy0,cy1],soft_extent_y=[sy0,sy1],
                candidate_boxes=[list(q) for q in candidates])
    return candidates,diag

"""Held-out local connector fit for neutral marker verification.

Long reference segments supply directions only. Up to two independently
observed strokes are fitted outside the marker; no marker pixels, truth labels,
or LSD detection are used to choose offsets, widths, or amplitudes.
"""
import cv2
import numpy as np
from bw_centered_composite_v46 import stroke
from bw_raster_identity_v46 import features

VERSION='outer_fit_two_connectors_v1'


def fit(comp, center, segments, diameter):
    xy=np.asarray(center,float)
    directions=[]
    for s in segments:
        a=np.asarray(s[:2]);v=np.asarray(s[2:])-a;length=float(np.linalg.norm(v))
        if length<2.5*diameter:continue
        t=v/length;proj=float((xy-a)@t);n=np.array([-t[1],t[0]])
        if min(proj,length-proj)<.7*diameter or abs(float((a-xy)@n))>.8*diameter:continue
        angle=float(np.degrees(np.arctan2(t[1],t[0]))%180)
        if all(abs((angle-q+90)%180-90)>3 for q in directions):directions.append(angle)
        if len(directions)>=4:break
    if not directions:return None
    comp._bank();r=comp.radius;fr=max(r,int(np.ceil(1.6*diameter)));size=2*fr+1
    patch=cv2.getRectSubPix(comp.ink,(size,size),tuple(map(float,xy)))
    valid=1-cv2.getRectSubPix(comp.observation.other,(size,size),tuple(map(float,xy)))
    yy,xx=np.indices((size,size),dtype=np.float32);xx-=fr;yy-=fr
    outer=(np.hypot(xx,yy)>=.70*diameter)*valid
    bank=[];pars=[];sides=[]
    for base in directions:
        for angle in (base-3,base,base+3):
            rad=np.deg2rad(angle);projection=xx*np.cos(rad)+yy*np.sin(rad)
            for offset in np.arange(-.65*diameter,.65*diameter+.01,.5):
                for width in (.8,1.2,1.6,2.):
                    line=stroke(size,(fr-np.sin(rad)*offset,fr+np.cos(rad)*offset),angle,width)
                    bank.append(line);pars.append(dict(angle=float(angle),offset=float(offset),width=width))
                    sides.append(projection)
    bank=np.asarray(bank);proj=np.asarray(sides)
    w=outer[None];den=np.sum(w*bank*bank,axis=(1,2));num=np.sum(w*bank*patch,axis=(1,2))
    amp=np.clip(num/np.maximum(den,1.e-6),0,1)
    # A line must have measured ink and consistent intensity on both sides.
    side_amp=[];side_den=[]
    for sign in (-1,1):
        sw=w*(sign*proj>=.70*diameter)
        sd=np.sum(sw*bank*bank,axis=(1,2));sn=np.sum(sw*bank*patch,axis=(1,2))
        side_den.append(sd);side_amp.append(np.clip(sn/np.maximum(sd,1.e-6),0,1))
    good=(np.minimum(*side_den)>=2)&(np.minimum(*side_amp)>=.15)
    good &= np.minimum(*side_amp)>=.5*np.maximum(*side_amp)
    rendered=bank*amp[:,None,None]
    losses=np.sum(w*(patch-rendered)**2,axis=(1,2))
    chosen=[]
    for i in np.argsort(np.where(good,losses,np.inf)):
        if not good[i]:break
        p=pars[i]
        if any(abs((p['angle']-pars[j]['angle']+90)%180-90)<5 and abs(p['offset']-pars[j]['offset'])<1. for j in chosen):continue
        chosen.append(int(i))
        if len(chosen)>=8:break
    if not chosen:return None
    variants=[(rendered[i],[i]) for i in chosen]
    for ai,i in enumerate(chosen):
        for j in chosen[ai+1:]:
            variants.append((1-(1-rendered[i])*(1-rendered[j]),[i,j]))
    # Complexity cost is paid during selection as well as final comparison.
    energy=max(float(np.sum(outer*patch*patch)),1.e-6)
    model,ids=min(variants,key=lambda q:float(np.sum(outer*(patch-q[0])**2))/energy+len(q[1])*comp.cfg.line_cost)
    sl=slice(fr-r,fr+r+1);small=patch[sl,sl];pred=model[sl,sl];vis=valid[sl,sl]
    f=features(small,comp.weight,comp.cfg.boundary_weight)[0]
    ff=features(pred,comp.weight,comp.cfg.boundary_weight)[0]
    conf=np.r_[vis.ravel(),np.minimum(vis[1:],vis[:-1]).ravel(),np.minimum(vis[:,1:],vis[:,:-1]).ravel()]
    loss=float(np.sum(conf*(f-ff)**2)/max(float(np.sum(conf*f*f)),1.e-6)+len(ids)*comp.cfg.line_cost)
    return dict(version=VERSION,loss=loss,lines=[dict(pars[i],amplitude=float(amp[i]),
        side_amplitudes=[float(a[i]) for a in side_amp]) for i in ids],
        fit_region='outside_marker_body',body_held_out_radius=.70*diameter,
        outer_loss=float(np.sum(outer*(patch-model)**2))/energy,
        source_segment_directions_only=True,source_segments_only=False)

"""Observed plot-palette estimation for an explicitly single-series plot.

No legend is invented. A source box identifies real pixels used for colour
sampling, never an exclusion rectangle or a semantic marker template. Multiple
substantial chromatic directions cause abstention. Achromatic results are only
available as an explicitly requested diagnostic because axes/grid/text share ink.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

import color_marker_evidence as evidence


@dataclass(frozen=True)
class PaletteConfig:
    min_contrast: float = 12.
    min_chroma: float = 24.
    direction_degrees: float = 9.
    direction_bin_width: float = .08
    minimum_cluster_pixels: int = 12
    dominant_fraction: float = .75
    substantial_rival_fraction: float = .20
    sample_candidates: int = 256


def _directions(pixels, paper, cfg):
    """Greedy histogram modes on normalized paper-minus-ink RGB directions.

    Antialiased mixtures along one paper/ink ray retain one direction; no forced
    K=1 clustering. Quantization bounds work, while angular membership uses the
    observed normalized vectors, not quantized colour values.
    """
    delta=paper-pixels.astype(np.float32)
    unit=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1.)
    quantized=np.rint(unit/cfg.direction_bin_width).astype(np.int16)
    bins,inverse,counts=np.unique(quantized,axis=0,return_inverse=True,return_counts=True)
    sums=np.column_stack([np.bincount(inverse,weights=unit[:,c],minlength=len(bins)) for c in range(3)])
    centers=sums/np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-9)
    pending=np.ones(len(bins),bool);labels=np.full(len(pixels),-1,np.int32);records=[]
    cutoff=math.cos(math.radians(cfg.direction_degrees))
    while pending.any():
        available=np.flatnonzero(pending)
        seed=int(available[np.argmax(counts[available])])
        selected=pending & (centers@centers[seed]>=cutoff)
        mean=sums[selected].sum(axis=0);mean/=max(float(np.linalg.norm(mean)),1e-9)
        selected=pending & (centers@mean>=cutoff)
        selected[seed]=True
        own=selected[inverse]
        label=len(records);labels[own]=label
        values=unit[own];center=values.mean(axis=0);center/=max(float(np.linalg.norm(center)),1e-9)
        records.append(dict(cluster_id=label,pixel_count=int(own.sum()),direction_bgr=center.tolist()))
        pending[selected]=False
    return labels,records


def _sample_box(crop, valid, model, cfg):
    """Choose a narrow ordinary stroke if observed; otherwise disclose fallback."""
    soft,fit=evidence._membership(crop,model)
    support=(soft>=.30)&(fit>=.65)&valid
    if support.sum()<3:
        return None,dict(status='no_strong_source_sample')
    distance=cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,5)
    ridges=support & (distance>=cv2.dilate(distance,np.ones((3,3),np.uint8)))
    ys,xs=np.nonzero(ridges)
    if not len(xs):
        return None,dict(status='no_source_ridge')
    lower=float(np.percentile(distance[ridges],25))
    ordinary=distance[ys,xs]<=max(1.01,1.25*lower)
    ys,xs=ys[ordinary],xs[ordinary]
    # Deterministic spatially spread candidates, not a first-pixel selection.
    if len(xs)>cfg.sample_candidates:
        index=np.linspace(0,len(xs)-1,cfg.sample_candidates).round().astype(int)
        ys,xs=ys[index],xs[index]
    h,w=support.shape;choices=[];fallback=[]
    for y,x in zip(ys,xs):
        stroke=max(1.,2*float(distance[y,x])-.5)
        radius=max(5,min(24,int(math.ceil(4*stroke))))
        a,b,c,d=max(0,x-radius),max(0,y-radius),min(w,x+radius+1),min(h,y+radius+1)
        yy,xx=np.nonzero(support[b:d,a:c])
        if len(xx)<6:
            continue
        coords=np.column_stack((xx+a-x,yy+b-y)).astype(float)
        cov=np.cov(coords.T);eigenvalues,eigenvectors=np.linalg.eigh(cov)
        ratio=float(eigenvalues[0]/max(eigenvalues[-1],1.))
        axis=eigenvectors[:,-1];normal=np.array([-axis[1],axis[0]])
        span=float(np.ptp(coords@axis)+1)
        along=max(6.,3.*stroke);across=max(2.,1.25*stroke)
        corners=np.array([np.array([x,y])+u*along*axis+v*across*normal for u in (-1,1) for v in (-1,1)])
        box=[max(0,int(np.floor(corners[:,0].min()))),max(0,int(np.floor(corners[:,1].min()))),
             min(w,int(np.ceil(corners[:,0].max()))+1),min(h,int(np.ceil(corners[:,1].max()))+1)]
        p,q,r,s=box
        if r-p<3 or s-q<3 or not valid[q:s,p:r].all():
            continue
        ink=int(support[q:s,p:r].sum());maximum=float(distance[b:d,a:c].max())
        row=dict(box=box,center=[int(x),int(y)],measured_stroke_width_px=stroke,
                 local_covariance_ratio=ratio,local_span_px=span,source_ink_pixels=ink)
        fallback.append((ratio,stroke,-ink,int(y),int(x),row))
        if ratio<=.16 and span>=max(8.,4*stroke) and maximum<=max(2.,1.75*lower):
            choices.append((stroke,ratio,-span,int(y),int(x),row))
    if choices:
        row=min(choices)[-1];row['status']='ordinary_stroke_sample'
        return row['box'],row
    if fallback:
        row=min(fallback)[-1];row['status']='observed_colour_patch_not_verified_ordinary_stroke'
        return row['box'],row
    return None,dict(status='no_valid_source_box')


def estimate_plot_palette(image_bgr, plot_box, *, valid=None, allow_achromatic=False,
                          config=PaletteConfig()):
    """Return observed palette model or explicit abstention, never a fake legend.

    Input/output boxes are full-source half-open xyxy. ``valid`` is plot-local
    spatial permission (white paper remains valid), not a foreground mask.
    An estimated model is compatible with color_marker_evidence._membership.
    source_box is diagnostic colour provenance and must NOT be fed to exclusion
    logic. A caller using the historical miner must supply color_rgb explicitly,
    keep legend_box=None, and construct valid independently of this sample box.
    """
    image=np.asarray(image_bgr)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('image_bgr must be uint8 HxWx3 BGR pixels')
    if not isinstance(config,PaletteConfig):
        raise TypeError('config must be PaletteConfig')
    cfg=config
    if not 0<cfg.direction_degrees<90 or not 0<cfg.direction_bin_width<1 or not 0<cfg.dominant_fraction<=1:
        raise ValueError('Invalid palette configuration')
    plot=evidence._box(plot_box,image.shape);x0,y0,x1,y1=plot
    crop=image[y0:y1,x0:x1]
    allowed=np.ones(crop.shape[:2],bool) if valid is None else np.asarray(valid,bool).copy()
    if allowed.shape!=crop.shape[:2]:
        raise ValueError('valid must be plot-local spatial permission')
    result=dict(status='unavailable',model=None,source_box=None,model_source='plot_observed',
                source_kind='palette_sample_not_legend',legend_box=None,exclusion_boxes=[],
                plot_box=plot,config=asdict(cfg),diagnostic_only=False,
                warnings=['Known-single-series scope is supplied, not inferred by this helper.',
                          'Colour recurrence cannot distinguish a curve from same-colour text or separate same-colour series.',
                          'Palette source_box is not a marker, legend, or invalid region.'],clusters=[])
    if not allowed.any():
        result['status']='no_valid_plot_pixels';return result
    # Spatially allowed actual plot pixels only; no inferred/painted samples.
    permitted=crop[allowed].reshape(-1,1,3)
    paper=evidence._paper(permitted)
    contrast=np.linalg.norm(crop.astype(np.float32)-paper,axis=2)
    ink=(contrast>=cfg.min_contrast)&allowed
    if not ink.any():
        result.update(status='no_distinguishable_ink',paper_bgr=paper);return result
    chromatic=ink & (np.ptp(crop.astype(float),axis=2)>=cfg.min_chroma)
    if chromatic.sum()<cfg.minimum_cluster_pixels:
        if not allow_achromatic:
            result.update(status='no_chromatic_palette',paper_bgr=paper,
                          achromatic_ink_pixels=int(ink.sum()));return result
        selected=ink
        result['diagnostic_only']=True
        result['warnings'].append('Achromatic diagnostic only: axes, grid, text and curve may be indistinguishable. Do not automatically activate a data series.')
    else:
        selected=chromatic
    ys,xs=np.nonzero(selected);pixels=crop[selected]
    labels,clusters=_directions(pixels,paper,cfg)
    substantial=[]
    for cluster in clusters:
        own=labels==cluster['cluster_id']
        cluster['fraction']=float(own.mean())
        cluster['box_source']=[int(xs[own].min()+x0),int(ys[own].min()+y0),
                               int(xs[own].max()+x0+1),int(ys[own].max()+y0+1)]
        cluster['x_bins_occupied']=int(len(np.unique(np.minimum(7,(8*xs[own]//max(1,crop.shape[1])).astype(int)))))
        cluster['y_bins_occupied']=int(len(np.unique(np.minimum(7,(8*ys[own]//max(1,crop.shape[0])).astype(int)))))
        if cluster['pixel_count']>=cfg.minimum_cluster_pixels:
            substantial.append(cluster)
    result['clusters']=sorted(clusters,key=lambda c:(-c['pixel_count'],c['cluster_id']))
    if not substantial:
        result.update(status='insufficient_colour_samples',paper_bgr=paper);return result
    substantial.sort(key=lambda c:(-c['pixel_count'],c['cluster_id']))
    winner=substantial[0]
    rival=any(c['fraction']>=cfg.substantial_rival_fraction for c in substantial[1:])
    if winner['fraction']<cfg.dominant_fraction or rival:
        result.update(status='ambiguous_palette',paper_bgr=paper,
                      reason='More than one substantial observed colour direction, or no uniquely dominant mode. No forced K=1 assignment.');return result
    observed=pixels[labels==winner['cluster_id']]
    model=evidence._estimate_model(observed.reshape(-1,1,3),paper_bgr=paper)
    local,sample=_sample_box(crop,allowed,model,cfg)
    if local is None:
        result.update(status='no_usable_palette_source_box',paper_bgr=paper,sample=sample);return result
    a,b,c,d=local;source=[a+x0,b+y0,c+x0,d+y0]
    result.update(status='achromatic_diagnostic_only' if result['diagnostic_only'] else 'estimated',
        model=model,source_box=source,source_center=[(source[0]+source[2]-1)/2,(source[1]+source[3]-1)/2],
        rgb=[int(round(v)) for v in model['bgr'][::-1]],paper_bgr=paper,
        sample=dict(sample,box_source=source),selected_cluster_id=winner['cluster_id'],
        source_pixel_count=int(len(observed)),
        model_provenance='color_marker_evidence._estimate_model on actual source pixels belonging to dominant observed plot-colour direction; plot paper sampled separately.',
        miner_note='Legacy _mine_plot_template re-estimates local residual_sigma from source_box. Pass rgb as color_rgb, do not treat source_box as legend/exclusion; fewer than 3 recurrent bodies remains a legitimate marker-mining failure.')
    return result

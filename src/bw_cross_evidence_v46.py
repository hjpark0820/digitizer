"""Source-only evidence for open-stroke X/+ markers, not filled interiors.

Four short ridges must meet at the proposed centre. Paper gaps penalize missing
arms; unavailable/occluded samples cannot provide positive evidence. Long
crossing lines and filled bodies are negative controls, not marker evidence.
"""
import cv2
import numpy as np
from functools import lru_cache

VERSION = 'bw-cross-four-arm-ridges-v1'
NAMES = ('x_marker', 'plus_marker')


def eligible(t):
    return getattr(t, 'name', '') in NAMES and t.ink.achromatic


def model_for(t):
    return getattr(t,'cross_stroke_model',None) or getattr(t,'soft_legend_model',None)


@lru_cache(maxsize=256)
def _rasters(name,width,height,stroke,blur,scale,size):
    from legend_composition_v46 import render_model,Config
    c=size//2;rows=[]
    line=dict(cx=c,cy=c,x0=-2.,x1=-1.,thickness=0.,blur_sigma=0.)
    for dy in (-.5,0.,.5):
        for dx in (-.5,0.,.5):
            p=dict(cx=c+dx,cy=c+dy,width=width*scale,height=height*scale,
                   stroke_width=stroke*scale,marker_blur_sigma=blur*scale)
            rows.append(render_model(name,p,line,(size,size),Config())['marker_alpha'])
    return np.stack(rows)


def raster_check(own,valid,center,model,scale):
    """Symmetric body+paper residual; only external line evidence is exempt."""
    from bw_small_hollow_v46 import nuisance
    p=model['params'];diameter=max(p['width'],p['height'])*scale
    size=2*int(np.ceil(diameter+4))+1;c=size//2
    yy,xx=np.indices((size,size),dtype=np.float32)
    xs=np.float32(xx+center[0]-c);ys=np.float32(yy+center[1]-c)
    a=cv2.remap(own,xs,ys,cv2.INTER_LINEAR)
    seen=cv2.remap(valid.astype(np.float32),xs,ys,cv2.INTER_LINEAR)>.95
    roi=(abs(xx-c)<=diameter/2+1)&(abs(yy-c)<=diameter/2+1)
    if seen[roi].mean()<.85:return dict(decision='abstain',reason='cross_body_unobservable')
    roi &= seen
    base,n=nuisance(a,(c,c),diameter,seen)
    names=(model['name'],'plus_marker' if model['name']=='x_marker' else 'x_marker',
           'circle','square','triangle_up','triangle_down','open_circle','open_square')
    losses={'line_only':float(np.mean((base[roi]-a[roi])**2))}
    args=(p['width'],p['height'],p['stroke_width'],model['blur_sigma'],scale,size)
    for name in names:
        shapes=_rasters(name,*args)
        pred=np.maximum(shapes[:,None]*np.array([.75,1.,1.25])[None,:,None,None],base)
        losses[name]=float(np.min(np.mean((pred[:,:,roi]-a[roi])**2,axis=2)))
    loss=losses[model['name']];rival=min(v for k,v in losses.items() if k!=model['name'])
    margin=rival-loss
    status='compatible' if loss<=.10 and margin>=.003 else 'abstain' if margin>=-.003 and loss<=.10 else 'conflict'
    return dict(decision=status,reason='cross_body_raster_supported' if status=='compatible' else 'cross_body_not_distinct_from_rivals',
                model_losses=losses,loss=loss,rival_margin=margin,crossing_strokes=n,
                domain='common source pixels: marker body and surrounding paper')


def evidence(observed, mask, name, available=None, contrast=1., occlusion=None, model=None, scale=1.):
    observed = np.asarray(observed, np.float32)
    mask = np.asarray(mask, bool)
    if observed.ndim != 2 or observed.shape != mask.shape or name not in NAMES:
        raise ValueError('Cross evidence needs aligned source, mask and X/+ identity')
    valid = np.ones(mask.shape, bool) if available is None else np.asarray(available, bool).copy()
    if valid.shape != mask.shape:
        raise ValueError('Cross availability must align with source')
    if occlusion is not None:
        if np.shape(occlusion) != mask.shape:
            raise ValueError('Cross occlusion must align with source')
        valid &= np.asarray(occlusion) < .5
    result = dict(version=VERSION, decision='abstain', reason='unobserved_cross_arms',
                  shape=name, interior_fill_required=False, source_pixels_unchanged=True)
    yy, xx = np.nonzero(mask)
    if len(xx) < 4:
        return result
    cx, cy = (xx.min()+xx.max())/2., (yy.min()+yy.max())/2.
    rx, ry = max(1., (np.ptp(xx)+1)/2), max(1., (np.ptp(yy)+1)/2)
    # Preserve the fitted aspect/centre. No per-arm relocation or scale search.
    directions = [(rx, ry), (-rx, ry), (-rx, -ry), (rx, -ry)] if name == 'x_marker' else [(rx, 0), (0, ry), (-rx, 0), (0, -ry)]
    own = np.clip(observed/max(.10, float(contrast)), 0, 1)
    def sample(array, x, y):
        return cv2.remap(array.astype(np.float32), np.float32(x).reshape(1, -1),
                         np.float32(y).reshape(1, -1), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
    records = []
    for dx, dy in directions:
        radius = np.hypot(dx, dy); nx, ny = -dy/radius, dx/radius
        u = np.linspace(.35, .85, 5)
        x, y = cx+u*dx, cy+u*dy
        flank = max(1.25, .30*min(rx, ry))
        peak = sample(own, x, y)
        left = sample(own, x+flank*nx, y+flank*ny)
        right = sample(own, x-flank*nx, y-flank*ny)
        visible = np.minimum.reduce([sample(valid, x, y), sample(valid, x+flank*nx, y+flank*ny),
                                     sample(valid, x-flank*nx, y-flank*ny)]) > .95
        # One independently lighter side can survive a transverse error bar.
        # A filled body has no such ridge on the inner part of all four arms.
        ridge = peak - np.minimum(left, right)
        support = visible & (peak >= .22) & (ridge >= .10)
        bilateral = visible & (peak >= .22) & (peak-np.maximum(left,right) >= .08)
        missing = visible & (peak < .12)
        outer = np.array([1.25, 1.5, 1.75])
        beyond = sample(own, cx+outer*dx, cy+outer*dy)
        beyond_valid = sample(valid, cx+outer*dx, cy+outer*dy) > .95
        records.append(dict(visible_samples=int(visible.sum()), support_fraction=float(support.mean()),
            bilateral_fraction=float(bilateral.mean()),
            white_gap_fraction=float(missing.mean()), ridge_mean=float(ridge[visible].mean()) if visible.any() else None,
            peak_mean=float(peak[visible].mean()) if visible.any() else None,
            extended=bool(beyond_valid.all() and np.min(beyond) >= .35)))
    supported = sum(r['support_fraction'] >= .6 for r in records)
    bilateral_arms = sum(r['bilateral_fraction'] >= .6 for r in records)
    extended = sum(r['extended'] for r in records)
    center = float(sample(own, [cx], [cy])[0])
    center_visible = float(sample(valid, [cx], [cy])[0]) > .95
    result.update(arms=records, supported_arms=supported, extended_arms=extended, center_ink=center,
                  bilateral_arms=bilateral_arms,
                  score=float(np.mean([r['support_fraction'] for r in records])),
                  loss=float(1-np.mean([r['support_fraction'] for r in records])))
    if any(r['white_gap_fraction'] >= .6 for r in records) or (center_visible and center < .12):
        result.update(decision='conflict', reason='missing_cross_arm_ink')
    elif extended >= 3:
        result.update(decision='conflict', reason='unbounded_crossing_strokes')
    elif supported == 4 and bilateral_arms >= 3 and center_visible and center >= .22:
        result.update(decision='compatible', reason='four_short_cross_ridges_supported')
    elif all(r['visible_samples'] >= 4 for r in records):
        result.update(decision='conflict', reason='cross_ridges_not_distinct_from_body')
    if result['decision']=='compatible' and model is not None:
        check=raster_check(own,valid,(cx,cy),model,scale)
        result['body_raster']=check
        if check['decision']!='compatible':
            result.update(decision=check['decision'],reason=check['reason'])
    return result

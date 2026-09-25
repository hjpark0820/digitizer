"""Shape evidence for tiny, bare hollow legends, independent of fill texture.

The observed legend raster stays the search template. Only its outer band is
compared on a common ROI with the other observed bare legend shapes. Nothing
inside a body is counted as a positive exterior boundary. This is a narrow
opt-in guard, not a replacement for the normal circle/square or colour paths.
"""
import cv2
import numpy as np
from functools import lru_cache
from scipy.ndimage import binary_fill_holes

VERSION='compact-source-outer-identity-v1'
SHIFTS=(-1.,-.5,0.,.5,1.)


@lru_cache(maxsize=32)
def _line_bank(size):
    """Counter-hypotheses only: thin lines cannot add marker evidence."""
    yy,xx=np.indices((size,size));xx=xx-size//2;yy=yy-size//2
    fields=[]
    for angle in np.linspace(0,np.pi,24,endpoint=False):
        perpendicular=-xx*np.sin(angle)+yy*np.cos(angle)
        for offset in (-2.,-1.,0.,1.,2.):
            for sigma in (.5,.8):
                fields.append(np.exp(-.5*((perpendicular-offset)/sigma)**2))
    return np.asarray(fields,np.float32)


def _line_loss(obs,roi):
    bank=_line_bank(obs.shape[0])[:,roi];values=obs[roi]
    denom=np.maximum(np.sum(bank*bank,axis=1),1.e-6)
    amp=np.clip(np.sum(bank*values,axis=1)/denom,0,1.25)
    predicted=np.minimum(1.,amp[:,None]*bank)
    losses=np.mean((predicted-values)**2,axis=1)
    # Do not fit two arbitrary lines inside the marker: they can appropriate
    # a triangle's genuine sides. Additional nuisance strokes require external
    # flank evidence; this guard deliberately uses only the single-line null.
    return float(losses.min())


def _fields(raw,scale,size):
    c=size//2;h,w=raw.shape
    target=cv2.warpAffine(np.asarray(raw,np.float32),
        np.float32([[scale,0,c-scale*(w-1)/2],[0,scale,c-scale*(h-1)/2]]),(size,size))
    env=binary_fill_holes(target>.22).astype(np.uint8)
    inside=cv2.distanceTransform(env,cv2.DIST_L2,5)
    outside=cv2.distanceTransform(1-env,cv2.DIST_L2,5)
    band=(inside<=1.5)&(outside<=1.5)
    return target,env,band


def evidence(source,center,own_key,models,scales,available=None):
    """Compare every model on the same observed band and translation budget.

    Width/height are fixed by the symbol's common scale, never fitted per point.
    Interior texture is excluded unless it is another model's outer boundary.
    A close tie is uncertainty, not proof of either shape. The diagnostic shifts
    do not move the candidate; the existing window owns the output centre.
    """
    source=np.asarray(source,np.float32)
    if source.ndim!=2 or not np.isfinite(source).all():
        raise ValueError('Finite two-dimensional source ink required')
    if own_key not in models:raise ValueError('Missing own outer model')
    valid=np.ones_like(source,bool) if available is None else np.asarray(available,bool)
    if valid.shape!=source.shape:raise ValueError('Visibility shape must match source')
    diameter=max(max(np.shape(raw))*scales.get(k,1.) for k,raw in models.items())
    size=2*int(np.ceil(diameter/2+4))+1;c=size//2
    yy,xx=np.indices((size,size),dtype=np.float32)
    fields={k:_fields(raw,scales.get(k,1.),size) for k,raw in models.items()}
    roi=np.logical_or.reduce([f[2] for f in fields.values()])
    samples=[]
    for dy in SHIFTS:
        for dx in SHIFTS:
            sx=xx-c+float(center[0])+dx;sy=yy-c+float(center[1])+dy
            obs=cv2.remap(source,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            seen=cv2.remap(valid.astype(np.float32),sx,sy,cv2.INTER_LINEAR)>.99
            samples.append((dx,dy,obs,seen))
    # Fixed intersection prevents a rival from hiding inconvenient pixels by
    # translating its ROI out of view. No unseen pixels become white evidence.
    seen=np.logical_and.reduce([s[3] for s in samples]);fraction=float(seen[roi].mean()) if roi.any() else 0.
    rec=dict(version=VERSION,decision='abstain',reason='outer_visibility_unresolved',
        loss=1.,visible_fraction=fraction,source_pixels_unchanged=True,
        shared_scale_only=True,common_roi_pixels=int((roi&seen).sum()),
        translation_budget_px=1.,models={},fill_status='not_measured_by_outer_band')
    if fraction<.85 or (roi&seen).sum()<12:return rec
    roi &= seen
    losses={}
    for key,(target,env,band) in fields.items():
        use=band&seen;best=None
        # The base can be above OR below the apex. A down triangle must not
        # inherit an up-triangle's three evidence regions.
        base_sign=1 if env[c+1:].sum()>=env[:c].sum() else -1
        sectors=np.where(base_sign*(yy-c)>diameter*.2,2,np.where(xx<c,0,1))
        for dx,dy,obs,_ in samples:
            error=(obs-target)**2;loss=float(error[roi].mean())
            side_loss=[float(error[use&(sectors==k)].mean()) if np.any(use&(sectors==k)) else 1. for k in range(3)]
            rim=use&(target>.35);outside=use&(env==0)
            missing=float(np.maximum(target[rim]-obs[rim],0).sum()/max(float(target[rim].sum()),1.e-6))
            side_missing=[]
            for k in range(3):
                part=rim&(sectors==k);mass=float(target[part].sum())
                side_missing.append(float(np.maximum(target[part]-obs[part],0).sum()/mass) if mass>.5 else 1.)
            row=dict(loss=loss,side_loss=side_loss,missing_rim_fraction=missing,
                side_missing_rim=side_missing,
                exterior_ink=float(obs[outside].mean()) if outside.any() else 1.,
                diagnostic_shift=[dx,dy])
            if best is None or loss<best['loss']:best=row
        losses[key]=best
    own=losses[own_key];rival=min((v['loss'] for k,v in losses.items() if k!=own_key),default=1.)
    margin=rival-own['loss']
    # Same pixels and shift budget as the marker models. This does not mask
    # any source ink, nor promote a marker on an inferred line intersection.
    line_loss=min(_line_loss(s[2],roi) for s in samples)
    rec.update(loss=own['loss'],models=losses,rival_margin=margin,
        worst_side_loss=max(own['side_loss']),missing_rim_fraction=own['missing_rim_fraction'],
        line_only_loss=line_loss,marker_over_line_improvement=line_loss-own['loss'])
    # Scores are descriptive squared alpha errors, NOT probabilities. These
    # bounds require all three regions, rather than many redundant grid votes.
    if own['missing_rim_fraction']>.30 or max(own['side_missing_rim'])>.40:
        rec.update(decision='conflict',reason='outer_required_ink_missing')
    elif margin<-.025:
        rec.update(decision='conflict',reason='other_legend_outer_shape_better')
    elif own['loss']>.15 or max(own['side_loss'])>.22:
        rec.update(reason='outer_sides_not_independently_supported')
    elif line_loss-own['loss']<.010:
        rec.update(reason='thin_lines_explain_outer_band')
    elif margin<.010:
        rec.update(reason='outer_shape_near_tie')
    else:
        rec.update(decision='compatible',reason='observed_outer_shape_supported')
    return rec

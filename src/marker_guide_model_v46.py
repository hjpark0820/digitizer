"""Freeze a repeated neutral-dot nuisance model from unoccluded guide cells.

Fit period/phase from narrow pulses, then fit the gray raster only on cells
without off-guide ink. No marker centre, class or detection is used in fitting.
An unsupported guide stays unresolved, never reverting to a broad row mask.
"""
from __future__ import annotations

import cv2
import numpy as np

from marker_ignore_v46 import _runs


def _lattice(centres, initial_period):
    # A whole-row Fourier phase estimate avoids accumulating one-pixel errors
    # from rounding every gap to the median integer distance.
    periods = np.linspace(.85*initial_period,1.15*initial_period,181)
    waves = np.exp(2j*np.pi*np.asarray(centres)[None,:]/periods[:,None]).mean(axis=1)
    best = int(np.argmax(np.abs(waves)))
    period = float(periods[best])
    phase = float(np.angle(waves[best])*period/(2*np.pi)) % period
    xs = np.asarray(centres,float)
    for _ in range(4):
        indices = np.rint((xs-phase)/period)
        residual = np.abs(xs-(phase+indices*period))
        keep = residual <= max(.55,.12*period)
        if keep.sum() < 8:
            return None
        design = np.column_stack((np.ones(int(keep.sum())),indices[keep]))
        phase,period = np.linalg.lstsq(design,xs[keep],rcond=None)[0]
    return float(period),float(phase % period),float(keep.mean())


def _basis(xs, period, phase):
    angle = 2*np.pi*(np.asarray(xs)-phase)/period
    count = min(3,max(1,int(period//2)))
    return np.column_stack([np.ones(len(angle)), *[f(k*angle) for k in range(1,count+1)
                                                   for f in (np.cos,np.sin)]])


def fit_guides(crop_bgr, structures, *, valid=None, marker_diameter=10.):
    """Return expected neutral ink, uncertainty, and explicit fit diagnostics.

    Fields are native crop-local float32 gray darkness (0=paper, 1=black).
    Error bounds represent repeat-to-repeat raster variability, not a license
    to fit each candidate's nearby marker as another reference dot.
    """
    image = np.asarray(crop_bgr)
    h,w = image.shape[:2]
    allowed = np.ones((h,w),bool) if valid is None else np.asarray(valid,bool)
    darkness = (255-cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32))/255.
    neutral = np.ptp(image.astype(np.int16),axis=2) <= 30
    model = np.zeros((h,w),np.float32)
    uncertainty = model.copy()
    reports = []
    for structure in structures:
        if structure.get('kind') != 'periodic_guide':
            continue
        lo,hi = structure['band'];left,right=structure.get('x_span',[0,w])
        y = int(np.argmax(np.mean(darkness[lo:hi,left:right]*neutral[lo:hi,left:right],axis=1)))+lo
        pulses = [(a+left,b+left) for a,b in _runs((darkness[y,left:right]>.25)&neutral[y,left:right]&allowed[y,left:right])]
        centres = [.5*(a+b-1) for a,b in pulses if b-a <= max(2,.50*marker_diameter)]
        record = dict(band=[lo,hi], reference_y=y, status='unresolved', initial_period=structure['period'])
        reports.append(record)
        if len(centres)<8:
            record['reason']='too_few_narrow_pulses';continue
        fitted = _lattice(centres,structure['period'])
        if fitted is None:
            record['reason']='no_stable_lattice';continue
        period,phase,fraction = fitted
        if fraction < .7:
            record['reason']='inconsistent_pulse_positions';continue
        # Cells are one period wide; inspect a marker-sized vertical context.
        # Stems, glyphs and coloured crossings disqualify a training cell.
        y0,y1=max(0,lo-2),min(h,hi+2)
        cy0,cy1=max(0,y-int(np.ceil(marker_diameter))),min(h,y+int(np.ceil(marker_diameter))+1)
        selected = np.zeros(w,bool);clean=[];rejected=[]
        k0,k1=int(np.ceil((left-phase)/period)),int(np.floor((right-1-phase)/period))
        for k in range(k0,k1+1):
            cx=phase+k*period
            a,b=max(left,int(np.ceil(cx-period/2))),min(right,int(np.ceil(cx+period/2)))
            if b-a<3 or a<=left or b>=right:
                continue
            patch=darkness[cy0:cy1,a:b];domain=allowed[cy0:cy1,a:b]
            yy=np.arange(cy0,cy1)[:,None]
            exterior=(yy<lo)|(yy>=hi)
            off = int(np.count_nonzero((patch>.16)&exterior&domain))
            coloured = int(np.count_nonzero((patch>.10)&~neutral[cy0:cy1,a:b]&domain))
            if off>max(2,.10*(b-a)*(hi-lo)) or coloured>1 or not domain.all():
                rejected.append([a,b]);continue
            if darkness[lo:hi,a:b].max()<.28:
                continue
            selected[a:b]=True;clean.append([a,b])
        if len(clean)<8 or np.ptp(np.flatnonzero(selected))<.45*(right-left):
            record.update(reason='insufficient_unoccluded_reference_cells',clean_cells=clean,rejected_cells=rejected)
            continue
        xs=np.arange(left,right)
        design=_basis(xs,period,phase)
        train=selected[left:right]
        values=darkness[y0:y1,left:right]
        row_models=[];row_errors=[]
        for row in values:
            X=design[train];target=row[train]
            coefficients=np.linalg.lstsq(X,target,rcond=None)[0]
            for _ in range(3):
                errors=np.abs(target-X@coefficients)
                tolerance=max(.025,2.5*float(np.median(errors)))
                weights=np.minimum(1,tolerance/np.maximum(errors,1e-6))
                coefficients=np.linalg.lstsq(X*weights[:,None],target*weights,rcond=None)[0]
            prediction=np.clip(design@coefficients,0,1)
            prediction[prediction<.018]=0
            residual=np.abs(target-X@coefficients)
            error=float(np.clip(np.percentile(residual,90)+.012,.018,.12))
            row_models.append(prediction);row_errors.append(error)
        rendered=np.asarray(row_models,np.float32)
        error=np.broadcast_to(np.asarray(row_errors,np.float32)[:,None],rendered.shape).copy()
        # Uncertainty only where the learned dot raster has plausible ink.
        support=cv2.dilate((rendered>.035).astype(np.uint8),np.ones((3,3),np.uint8))>0
        error*=support
        domain=allowed[y0:y1,left:right]
        rendered*=domain;error*=domain
        model[y0:y1,left:right]=np.maximum(model[y0:y1,left:right],rendered)
        uncertainty[y0:y1,left:right]=np.maximum(uncertainty[y0:y1,left:right],error)
        record.update(status='fitted',period=period,phase=phase,inlier_fraction=fraction,
            model_box=[left,y0,right,y1],clean_cells=clean,rejected_cells=rejected,
            clean_cell_count=len(clean),median_gray_error=float(np.median(row_errors)),
            policy='Fitted only on cells without off-guide or coloured ink; fixed for all candidates')
    return model,uncertainty,dict(version='periodic_gray_guide_v1',guides=reports,
        fitted_count=sum(r['status']=='fitted' for r in reports),source_pixels_changed=False)


def palette_fields(model, error, templates):
    """Convert rendered guide darkness with the SAME colour-membership model."""
    from color_marker_evidence import _membership
    predictions=[]
    support=(model>0)|(error>0)
    for level in (np.maximum(model-error,0),model,np.minimum(model+error,1)):
        gray=np.rint(255*(1-level)).astype(np.uint8)
        raster=np.repeat(gray[...,None],3,axis=2)
        predictions.append(np.stack([_membership(raster,t['model'])[0]*support for t in templates]))
    low=np.minimum.reduce(predictions);high=np.maximum.reduce(predictions)
    return predictions[1],low,high

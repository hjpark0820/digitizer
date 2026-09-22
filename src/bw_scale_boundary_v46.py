"""Bounded, image-only outer-edge evidence for filled shared-scale anchors.

An inscribed template does not explain an edge: ink on BOTH sides supplies no
support. Cross-colour occlusion abstains and never counts as a matching edge.
The full perimeter denominator and distributed quadrants prevent one short
connector fragment from authorizing a marker size. No marker positions are
returned as detections; these measurements only vote on a per-series scale.
"""
import cv2
import numpy as np

VERSION = 'shared-scale-visible-outline-v1'


def eligible(template):
    return getattr(template, 'marker_kind', '') == 'filled'


def model_samples(template):
    mask=np.pad(np.asarray(template.mask,np.uint8),3)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return None
    contour=max(contours,key=cv2.contourArea)[:,0,:].astype(np.float32)
    if len(contour)<12:return None
    # Sample at most 96 equally spaced raster contour points. Sobel normals
    # point inward; half a pixel outward locates the ink/paper transition.
    contour=contour[np.linspace(0,len(contour)-1,min(96,len(contour))).astype(int)]
    soft=cv2.GaussianBlur(mask.astype(np.float32),(5,5),.8)
    gx=cv2.Sobel(soft,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(soft,cv2.CV_32F,0,1,ksize=3)
    xx,yy=contour[:,0].astype(int),contour[:,1].astype(int)
    normals=np.stack([gx[yy,xx],gy[yy,xx]],axis=1)
    norm=np.linalg.norm(normals,axis=1)
    keep=norm>1e-4;contour=contour[keep];normals=normals[keep]/norm[keep,None]
    origin=np.array([(mask.shape[1]-1)/2,(mask.shape[0]-1)/2])
    boundary=contour-.5*normals-origin
    quadrant=(boundary[:,0]>=0).astype(int)+2*(boundary[:,1]>=0).astype(int)
    return boundary.astype(np.float32),normals.astype(np.float32),quadrant


def profile(observed, template, x, y, scales, valid=None, other=None, samples=None):
    samples=model_samples(template) if samples is None else samples
    if samples is None:
        return dict(accepted=False,reason='insufficient_native_outline',profile=[-1.]*len(scales))
    boundary,normals,quadrant=samples
    # Fixed +/-2 px translations at every scale, not progressively moving the
    # anchor to a different nearby marker as scale changes.
    shifts=np.array([(dx,dy) for dy in range(-2,3) for dx in range(-2,3)],np.float32)
    scale=np.asarray(scales,np.float32)[:,None,None,None]
    xy=boundary[None,None,:,:]*scale+shifts[None,:,None,:]+[x,y]
    offset=np.clip(.075*float(template.diameter)*np.asarray(scales),1.25,2.)[:,None,None,None]
    def sample(field,delta):
        pos=xy+delta*offset*normals[None,None,:,:]
        return cv2.remap(np.asarray(field,np.float32),np.float32(pos[...,0]).reshape(-1,len(boundary)),
                         np.float32(pos[...,1]).reshape(-1,len(boundary)),cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0).reshape(xy.shape[:-1])
    inside,outside,mid=sample(observed,1),sample(observed,-1),sample(observed,0)
    usable=np.ones_like(mid,dtype=bool)
    if valid is not None:
        usable &= (sample(valid,1)>.99)&(sample(valid,-1)>.99)
    if other is not None:
        usable &= (sample(other,1)<.25)&(sample(other,0)<.25)&(sample(other,-1)<.25)
    contrast=np.clip(inside-outside,0,1)
    # Location term makes inside-only patches inferior even for a thick edge.
    support=np.where(usable,contrast*np.exp(-((mid-.5)/.45)**2),0.)
    quality=support.mean(axis=-1)
    qs=np.stack([support[...,quadrant==q].mean(axis=-1) if np.any(quadrant==q)
                 else np.zeros(quality.shape) for q in range(4)],axis=-1)
    supported=(qs>=.25).sum(axis=-1)
    allowed=(quality>=.32)&(supported>=3)
    values=np.where(allowed,quality,-1.)
    j=np.argmax(values,axis=1);i=np.arange(len(scales))
    curve=values[i,j];best=int(np.argmax(curve));k=int(j[best])
    return dict(version=VERSION,accepted=bool(curve[best]>=0),
                reason='distributed_visible_outline' if curve[best]>=0 else 'insufficient_visible_outline',
                profile=curve.tolist(),raw_boundary_profile=quality.max(axis=1).tolist(),
                quality=float(curve[best]),best_scale=float(scales[best]),
                center=[float(x+shifts[k,0]),float(y+shifts[k,1])],
                quadrant_scores=qs[best,k].tolist(),supported_quadrants=int(supported[best,k]),
                observable_fraction=float(usable[best,k].mean()),
                samples=int(len(boundary)),unknown_is_positive=False,
                center_by_scale=(shifts[j]+[x,y]).tolist())

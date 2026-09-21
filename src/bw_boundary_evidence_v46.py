"""Positive outline evidence, distinct from foreground/occlusion compatibility.

A filled marker must explain ink-to-paper transitions along its boundary.
Both sides being ink may be occlusion, but never supplies shape evidence.
Engineering scores below are not calibrated probabilities.
"""
import cv2
import numpy as np

VERSION = 'normal-transition-boundary-v1'


def enabled(template):
    return (getattr(template, 'matching_profile', '') == 'bw_v46_uncertain'
            and getattr(template, 'marker_kind', '') == 'filled')


def transition_evidence(observed, expected, valid=None, nuisance=None, other=None):
    """Sample short 1-D profiles normal to a model's measured outer boundary.

    No nearby-edge borrowing or independent per-model translation is allowed.
    Unknown/crossing-line samples earn no positive credit. Support retains its
    full perimeter denominator, preventing one exposed fragment scoring 1.0.
    Inputs share pixel coordinates; observed is source ink, not an opened body.
    """
    observed=np.asarray(observed,np.float32);expected=np.asarray(expected,np.float32)
    if observed.shape != expected.shape or observed.ndim != 2:
        raise ValueError('Boundary ink and model must be aligned 2D arrays')
    mask=(expected>=.5).astype(np.uint8)
    edge=mask & (1-cv2.erode(mask,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0))
    yy,xx=np.nonzero(edge)
    if len(xx)<6:
        return dict(version=VERSION,status='insufficient_outline',score=0.,supported_quadrants=0,
                    samples=0,visible_fraction=0.,quadrant_scores=[0.]*4)
    smooth=cv2.GaussianBlur(expected,(3,3),.55)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    norm=np.hypot(gx[yy,xx],gy[yy,xx]);nx=gx[yy,xx]/np.maximum(norm,1e-6);ny=gy[yy,xx]/np.maximum(norm,1e-6)
    # Sobel points into dark/ink foreground, not out towards paper.
    diameter=max(np.ptp(xx)+1,np.ptp(yy)+1)
    offset=max(1.25,min(2.,.075*diameter))
    def sample(array, displacement):
        return cv2.remap(np.asarray(array,np.float32),np.float32(xx+displacement*nx)[None,:],
                         np.float32(yy+displacement*ny)[None,:],cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]
    inside,outside=sample(observed,offset),sample(observed,-offset)
    expected_contrast=sample(expected,offset)-sample(expected,-offset)
    observable=(norm>1e-5)&(expected_contrast>.25)
    if valid is None:valid=np.ones(observed.shape,np.float32)
    observable &= (sample(valid,offset)>.99)&(sample(valid,-offset)>.99)
    for unknown in (nuisance,other):
        if unknown is not None:
            observable &= (sample(unknown,offset)<.25)&(sample(unknown,0)<.25)&(sample(unknown,-offset)<.25)
    # A dark exterior is not a circular edge, even when the interior is perfect.
    contrast=np.clip((inside-outside)/np.maximum(expected_contrast,.4),0,1)
    support=np.where(observable,contrast,0.)
    cy,cx=(expected.shape[0]-1)/2,(expected.shape[1]-1)/2
    quadrant=(xx>=cx).astype(int)+2*(yy>=cy).astype(int)
    q=[float(support[quadrant==i].mean()) if np.any(quadrant==i) else 0. for i in range(4)]
    return dict(version=VERSION,status='measured',score=float(support.mean()),
                supported_quadrants=sum(v>=.35 for v in q),samples=int(len(xx)),
                observable_samples=int(observable.sum()),visible_fraction=float(observable.mean()),
                quadrant_scores=q,profile_offset_px=float(offset),
                mean_inside=float(inside.mean()),mean_outside=float(outside.mean()),
                unknown_is_positive=False,curvature_test='distributed model-normal transitions')


def grid_score_adjustment(template, observed, previous_support, other=None):
    if not enabled(template):return 0.,{}
    result=transition_evidence(observed,template.soft,other=other)
    result['previous_proximity_or_occlusion_support']=float(previous_support)
    result['score_weight']=.21
    return .21*(result['score']-previous_support),result


def compare_transitions(a,b,name_a,name_b):
    """Require positive observed outline evidence distributed over quadrants."""
    margin=float(a['score']-b['score'])
    sign=1 if margin>0 else -1
    winner=name_a if sign>0 else name_b
    best=a if sign>0 else b
    qs=sign*(np.array(a['quadrant_scores'])-np.array(b['quadrant_scores']))
    quadrants=int(np.sum(qs>=.04))
    supported=(a['samples']>=8 and b['samples']>=8 and
               min(a['visible_fraction'],b['visible_fraction'])>=.5 and
               best['score']>=.45 and abs(margin)>=.07 and quadrants>=2)
    return dict(winner=winner if supported else None,margin=margin,
                supporting_quadrants=quadrants,minimum_margin=.07,
                reason='distributed_normal_transitions' if supported else 'outline_identity_unresolved')

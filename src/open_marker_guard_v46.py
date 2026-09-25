"""Negative evidence for opt-in pure hollow templates; no source pixel edits."""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from partial_swatch_detector import ink_membership


def apply_guard(result,template,occlusion=None):
    if not getattr(template,'model_completed',False) or template.marker_kind!='open':return result
    from bw_small_hollow_v46 import eligible as small_hollow,guard as small_guard
    if small_hollow(template):return small_guard(result,template,occlusion)
    from bw_hollow_boundary_v46 import eligible, source_darkness, square_interior_evidence
    if eligible(template):
        evidence=square_interior_evidence(source_darkness(result.plot_window),result.template_mask,
                                          occlusion=occlusion)
        result.compute_diagnostics['open_interior']=evidence
        if evidence['decision']=='conflict':result.decision='rejected'
        elif evidence['decision']=='abstain' and result.decision=='verified':result.decision='ambiguous'
        return result
    rendered=result.template_mask
    hole=binary_fill_holes(rendered)&~rendered
    hole=cv2.erode(hole.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    membership=ink_membership(result.plot_window,template.ink)
    ink=(membership>=.45).astype(np.uint8);diam=template.diameter*result.scale
    # Dashed connectors may end INSIDE the hole. Requiring a 1.5-diameter
    # continuous stroke wrongly calls those short dashes a filled interior.
    # Recognize short straight thin ink too; the depth bound below prevents
    # a filled disk from explaining its broad interior as many short lines.
    length=max(5,int(round(.35*diam)))|1
    strokes=np.zeros_like(ink)
    for angle in np.linspace(0,np.pi,12,endpoint=False):
        kernel=np.zeros((length,length),np.uint8);r=length//2
        dx,dy=int(round(r*np.cos(angle))),int(round(r*np.sin(angle)))
        cv2.line(kernel,(r-dx,r-dy),(r+dx,r+dy),1,1)
        strokes|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
    depth=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    thin=(strokes>0)&(depth<=max(1.4,.13*diam))
    visible=hole&~thin
    if occlusion is not None:visible&=occlusion<.5
    fraction=float(visible.sum()/max(1,hole.sum()))
    ink_fraction=float(membership[visible].mean()) if visible.any() else 1.
    small_evidence=None
    if (template.name=='open_circle' and diam<=10. and 0<hole.sum()<=9 and
        0<visible.sum()<4 and fraction>=.20 and ink_fraction<=.10):
        # A tiny ring may have only one independent white pixel after the
        # observed crossing strokes are excluded. White evidence is not the
        # same as no observation; require direct rim and round-boundary proof
        # before using that smaller native sample. Never invent hole pixels.
        from bw_circle_boundary_v46 import final_circle_evidence
        small_evidence=final_circle_evidence(source_darkness(result.plot_window),rendered,
                                            occlusion=occlusion)
        rim=result.compute_diagnostics.get('circle_rim',{})
        small_evidence=dict(small_evidence,direct_rim_recall=rim.get('direct_rim_recall'),
            paper_loss=rim.get('paper_loss'),minimum_visible_pixels=1,
            measured_hole_pixels=int(hole.sum()),measured_white_pixels=int(visible.sum()))
        enough=(small_evidence['decision']=='compatible' and
                rim.get('direct_rim_recall',0.)>=.90 and rim.get('paper_loss',1.)<=.03)
    else:enough=False
    if visible.sum()<max(4,.20*hole.sum()) and not enough:status='unobserved'
    elif ink_fraction>.35:status='filled_interior_conflict'
    else:status='hollow_interior_supported'
    result.compute_diagnostics['open_interior']=dict(status=status,hole_pixels=int(hole.sum()),
        independent_visible_pixels=int(visible.sum()),visible_fraction=fraction,own_ink_fraction=ink_fraction)
    if small_evidence is not None:
        result.compute_diagnostics['open_interior']['small_hole_evidence']=small_evidence
    if status=='filled_interior_conflict':result.decision='rejected'
    elif status=='unobserved' and result.decision=='verified':result.decision='ambiguous'
    return result

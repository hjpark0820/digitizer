"""Positive circular arc evidence, independent of interior ink containment.

Fixed radial paper/rim/paper profiles are compared with rectangular profiles
at the SAME centre and outer/inner extents. A white square is not a circle
anchor. Crossings and other-colour occlusion earn no positive boundary credit.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

VERSION='paired-circle-shape-v2'


def eligible(template):
    return (getattr(template,'model_completed',False)
            and getattr(template,'name','')=='open_circle'
            and getattr(template,'marker_kind','')=='open'
            and getattr(template,'matching_profile','')=='bw_v46_uncertain'
            and template.ink.achromatic)


def boundary_profiles(observed,center,outer,inner,available=None,occlusion=None,gradients=None):
    """40 fixed radial directions; no per-ray best-radius fitting.

    Mean support retains all directions in its denominator. The square rival
    uses identical extents and source pixels, never a separately optimized box.
    """
    image=np.asarray(observed,np.float32)
    valid=np.ones_like(image) if available is None else np.asarray(available,np.float32).copy()
    if image.ndim!=2 or valid.shape!=image.shape:raise ValueError('Aligned source/availability required')
    if occlusion is not None:
        if np.shape(occlusion)!=image.shape:raise ValueError('Aligned occlusion required')
        valid*=np.asarray(occlusion)<.5
    cx,cy=center;rx,ry=outer;ix,iy=inner
    empty=dict(version=VERSION,supported=False,score=0.,circle_score=0.,square_score=0.,
               diamond_score=0.,triangle_score=0.,inv_triangle_score=0.,shape_margin=0.,supported_fraction=0.,sector_support=[0.]*8,reason='unresolved_ring_geometry')
    if min(ix,iy)<1.5 or min(rx-ix,ry-iy)<.75:return empty
    angles=np.linspace(0,2*np.pi,40,endpoint=False)+np.pi/40
    ux,uy=np.cos(angles),np.sin(angles)
    def triangle_normals(a,b,kind):
        ns=np.array([[2/a,-1/b],[0,1/b],[-2/a,-1/b]])
        if kind=='inv_triangle':ns[:,1]*=-1
        return ns
    def radius(a,b,kind):
        if kind=='square':return np.minimum(a/np.maximum(abs(ux),1e-6),b/np.maximum(abs(uy),1e-6))
        if kind=='diamond':return 1./(abs(ux)/a+abs(uy)/b)
        if kind in ('triangle','inv_triangle'):
            ns=triangle_normals(a,b,kind)
            return 1./np.max(ns[:,0,None]*ux+ns[:,1,None]*uy,axis=0)
        return 1./np.sqrt((ux/a)**2+(uy/b)**2)
    def sample(array,r):
        return cv2.remap(array,np.float32(cx+ux[:,None]*r),np.float32(cy+uy[:,None]*r),
                         cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    if gradients is None:
        # Work on a bounded ROI, never Sobel the complete source for every ray.
        pad=int(np.ceil(max(rx,ry)))+5
        x0=max(0,int(cx)-pad);y0=max(0,int(cy)-pad)
        crop=image[y0:min(image.shape[0],int(cy)+pad+2),x0:min(image.shape[1],int(cx)+pad+2)]
        gx=cv2.Sobel(crop,cv2.CV_32F,1,0,ksize=3,scale=1/8)
        gy=cv2.Sobel(crop,cv2.CV_32F,0,1,ksize=3,scale=1/8)
    else:gx,gy=gradients;x0=y0=0
    def normal(a,b,kind):
        if kind=='circle':nx,ny=ux/(a*a),uy/(b*b)
        elif kind=='diamond':nx,ny=np.sign(ux)/a,np.sign(uy)/b
        elif kind in ('triangle','inv_triangle'):
            ns=triangle_normals(a,b,kind);idx=np.argmax(ns[:,0,None]*ux+ns[:,1,None]*uy,axis=0)
            nx,ny=ns[idx,0],ns[idx,1]
        else:
            vertical=a/np.maximum(abs(ux),1e-6)<=b/np.maximum(abs(uy),1e-6)
            nx=np.where(vertical,np.sign(ux),0);ny=np.where(vertical,0,np.sign(uy))
        length=np.hypot(nx,ny);return nx/length,ny/length
    def gradient_direction(r,a,b,kind,sign):
        xx=np.float32(cx-x0+ux*r);yy=np.float32(cy-y0+uy*r)
        dx=cv2.remap(gx,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()
        dy=cv2.remap(gy,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).ravel()
        nx,ny=normal(a,b,kind);mag=np.hypot(dx,dy)
        # The same gradient reliability and sign convention apply to rivals.
        return np.clip(sign*(dx*nx+dy*ny)/np.maximum(mag,.06),0,1)
    def profile(kind):
        ro,ri=radius(rx,ry,kind),radius(ix,iy,kind)
        width=ro-ri;gap=np.minimum(1.5,.30*ri)
        rim=ri[:,None]+width[:,None]*np.array([.3,.5,.7])[None,:]
        inside=ri[:,None]-gap[:,None]-np.array([0,.4,.8])[None,:]
        outside=ro[:,None]+np.array([1.,1.5,2.])[None,:]
        vals=[np.median(sample(image,r),axis=1) for r in (outside,rim,inside)]
        seen=np.logical_and.reduce([np.all(sample(valid,r)>.99,axis=1) for r in (outside,rim,inside)])
        contrast=np.minimum(vals[1]-vals[0],vals[1]-vals[2])
        credit=np.where(seen,np.clip(contrast,0,1),0.)
        direction=.5*(gradient_direction(ro,rx,ry,kind,-1)+gradient_direction(ri,ix,iy,kind,1))
        # A proposed stroke on visible paper is NEGATIVE evidence, not merely
        # absent positive contrast. Apply identically to every rival shape.
        paper=np.where(seen,np.clip((.18-vals[1])/.18,0,1),0.)
        score=credit*(.65+.35*direction)-.5*paper
        support=seen & (contrast>=.25) & (vals[1]>=.4)
        return dict(score=float(max(0.,score.mean())),raw_contrast_score=float(credit.mean()),paper_loss=float(paper.mean()),
                    direction_score=float((credit*direction).sum()/max(credit.sum(),1.e-6)),
                    contrasts=contrast.tolist(),credit=credit.tolist(),
                    supported_fraction=float(support.mean()),observable_fraction=float(seen.mean()),
                    sector_support=support.reshape(8,5).mean(axis=1).tolist(),
                    outer=vals[0].tolist(),rim=vals[1].tolist(),inner=vals[2].tolist())
    circle,square,diamond=profile('circle'),profile('square'),profile('diamond')
    triangle,inverted=profile('triangle'),profile('inv_triangle')
    margin=circle['score']-max(square['score'],diamond['score'],triangle['score'],inverted['score']);sectors=np.asarray(circle['sector_support'])
    supported=(circle['score']>=.35 and circle['supported_fraction']>=.55
               and int((sectors>=.4).sum())>=6 and margin>=.055)
    return dict(version=VERSION,supported=bool(supported),score=circle['score'],
                circle_score=circle['score'],square_score=square['score'],diamond_score=diamond['score'],shape_margin=float(margin),
                triangle_score=triangle['score'],inv_triangle_score=inverted['score'],
                supported_fraction=circle['supported_fraction'],sector_support=circle['sector_support'],
                observable_fraction=circle['observable_fraction'],circle=circle,square=square,diamond=diamond,
                triangle=triangle,inv_triangle=inverted,
                center=[float(cx),float(cy)],outer_radii=[float(rx),float(ry)],inner_radii=[float(ix),float(iy)],
                reason='distributed_round_inner_outer_boundaries' if supported else 'round_identity_unresolved',
                independent_radius_per_ray=False,unknown_is_positive=False)


def geometry(mask):
    mask=np.asarray(mask,bool);env=binary_fill_holes(mask);hole=env&~mask
    yy,xx=np.nonzero(env);hy,hx=np.nonzero(hole)
    if len(hx)<6 or len(xx)<12:return None
    return (((xx.min()+xx.max())/2.,(yy.min()+yy.max())/2.),
            ((xx.max()-xx.min()+1)/2.,(yy.max()-yy.min()+1)/2.),
            ((hx.max()-hx.min()+1)/2.,(hy.max()-hy.min()+1)/2.))


def circle_boundary_evidence(observed,mask,available=None,occlusion=None):
    if np.shape(observed)!=np.shape(mask):raise ValueError('Aligned source and circle model required')
    dims=geometry(mask)
    if dims is None:
        return dict(version=VERSION,supported=False,score=0.,circle_score=0.,square_score=0.,
                    shape_margin=0.,supported_fraction=0.,sector_support=[0.]*8,reason='unresolved_ring_geometry')
    return boundary_profiles(observed,*dims,available=available,occlusion=occlusion)


def rival_score(evidence,rival_shapes=None):
    kinds=('square','diamond','triangle','inv_triangle') if rival_shapes is None else rival_shapes
    return max((evidence.get(k+'_score',0.) for k in kinds),default=0.)


def profile_identity(evidence,rival_shapes=None):
    """Compare BOTH shapes over the same whole-body size/translation hypotheses.

    An oversized circle can beat an equally oversized square on a real square.
    Comparing independently best *whole bodies* prevents that containment error.
    The radius is never optimized separately for individual rays.
    """
    if not evidence:
        return dict(accepted=False,reason='no_observable_geometry',circle_score=0.,square_score=0.,shape_margin=0.)
    best=max(evidence,key=lambda e:e['circle_score'])
    square=max(e['square_score'] for e in evidence)
    diamond=max(e.get('diamond_score',0.) for e in evidence)
    triangle=max(e.get('triangle_score',0.) for e in evidence)
    inverted=max(e.get('inv_triangle_score',0.) for e in evidence)
    rival=max(rival_score(e,rival_shapes) for e in evidence)
    margin=best['circle_score']-rival
    accepted=(best['circle_score']>=.35 and best['supported_fraction']>=.55
              and sum(s>=.4 for s in best['sector_support'])>=6 and margin>=.055)
    return dict(accepted=bool(accepted),circle_score=best['circle_score'],square_score=float(square),diamond_score=float(diamond),
                triangle_score=triangle,inv_triangle_score=inverted,rival_score=rival,
                rival_shapes=list(rival_shapes) if rival_shapes is not None else ['square','diamond','triangle','inv_triangle'],
                shape_margin=float(margin),supported_fraction=best['supported_fraction'],
                sector_support=best['sector_support'],
                reason='independent_circular_arc_anchor' if accepted else 'circle_vs_polygon_identity_unresolved',
                comparison='circle/square/diamond/triangles positions and signed normals; same whole-body hypotheses')


def final_circle_evidence(observed,mask,available=None,occlusion=None,rival_shapes=None):
    """Keep the accepted circle size fixed; challenge it with whole polygons.

    Rival size/translation variation is bounded and joint, never fitted per ray.
    Crossing ink earns no contrast/normal credit, but is not erased from source.
    """
    dims=geometry(mask)
    if dims is None:return dict(decision='abstain',reason='unresolved_ring_geometry',version=VERSION)
    center,outer,inner=dims
    image=np.asarray(observed,np.float32)
    grads=(cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3,scale=1/8),
           cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3,scale=1/8))
    actual=boundary_profiles(image,center,outer,inner,available,occlusion,grads)
    rivals=[]
    for scale in (.8,1.,1.2,1.4):
        for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            rivals.append(boundary_profiles(image,(center[0]+dx,center[1]+dy),
                tuple(r*scale for r in outer),tuple(r*scale for r in inner),available,occlusion,grads))
    square=max(e['square_score'] for e in rivals);diamond=max(e['diamond_score'] for e in rivals)
    triangle=max(e['triangle_score'] for e in rivals);inverted=max(e['inv_triangle_score'] for e in rivals)
    rival=max(rival_score(e,rival_shapes) for e in rivals)
    margin=actual['circle_score']-rival
    # A clean calibration anchor is stricter than a partly overlapped marker.
    # Weak shape evidence is not a forced circle label.
    if rival>=.30 and margin<-.035:
        decision,reason='conflict','polygon_boundary_better_than_circle'
    elif actual['circle_score']<.24 or actual['supported_fraction']<.40 or margin<.025:
        decision,reason='abstain','insufficient_independent_round_boundary'
    else:decision,reason='compatible','round_boundary_beats_polygon_rivals'
    return dict(version=VERSION,decision=decision,reason=reason,circle_score=actual['circle_score'],
        square_score=square,diamond_score=diamond,shape_margin=margin,
        triangle_score=triangle,inv_triangle_score=inverted,
        rival_shapes=list(rival_shapes) if rival_shapes is not None else ['square','diamond','triangle','inv_triangle'],
        direction_score=actual.get('circle',{}).get('direction_score'),
        supported_fraction=actual['supported_fraction'],sector_support=actual['sector_support'],
        circle_size_changed=False,per_ray_geometry_fitted=False)

"""Source-pixel BW connection evidence, promoted from the limited-correction pilot.

Negative chord tests do not prove author intent or identify a fitted curve.
Ambiguous series abstain from connector SSIM.
"""
import cv2
import numpy as np

def ink_image(image, plot, legend=None):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    paper=float(np.percentile(gray,90))
    ink=((paper-gray.astype(float))>max(25,.18*paper)).astype(np.uint8)
    scope=np.zeros_like(ink); x0,y0,x1,y1=map(int,plot); scope[y0:y1,x0:x1]=1
    if legend is not None:
        a,b,c,d=map(int,legend); scope[b:d,a:c]=0
    return ink*scope,scope


def sample(field,xy):
    return cv2.remap(field.astype(np.float32),np.asarray(xy[:,0],np.float32)[None,:],
                     np.asarray(xy[:,1],np.float32)[None,:],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=999).ravel()


def chord_test(a,b,ink,diameter,scope=None,distance=None):
    a=np.asarray(a,float); b=np.asarray(b,float); length=np.linalg.norm(b-a)
    if length<2.5*diameter or b[0]-a[0]<diameter:
        return dict(eligible=False,reason='too_short_or_nearly_vertical',a=a.tolist(),b=b.tolist())
    # Exclude marker bodies and near-end error bars from the interior test.
    unit=(b-a)/length; normal=np.array([-unit[1],unit[0]])
    t=np.linspace(.75*diameter,length-.75*diameter,max(12,int(length-1.5*diameter)))
    xy=a+t[:,None]*unit
    dist=distance if distance is not None else cv2.distanceTransform(1-ink,cv2.DIST_L2,5)
    radius=max(1.25,.075*diameter)
    near=sample(dist,xy)<=radius
    valid=np.ones(len(xy),bool) if scope is None else sample(scope,xy)>.99
    if valid.mean()<.8:
        return dict(eligible=False,reason='legend_or_roi_occlusion',a=a.tolist(),b=b.tolist())
    # Directional agreement: compare near-chord occupancy to displaced parallel
    # rails. A transverse error bar supports only a short part of the chord.
    side=.5*(sample(dist,xy+normal*max(4,.25*diameter))<=radius).mean()+.5*(sample(dist,xy-normal*max(4,.25*diameter))<=radius).mean()
    bins=np.array_split(near,8); continuity=float(np.mean([v.mean()>.55 for v in bins]))
    padded=np.r_[False,near,False]; edges=np.flatnonzero(padded[1:]!=padded[:-1])
    runs=list(zip(edges[::2],edges[1::2])); lengths=np.array([j-i for i,j in runs])
    gaps=np.array([runs[i+1][0]-runs[i][1] for i in range(len(runs)-1)])
    dashed=len(runs)>=3 and near.mean()>.30 and len(gaps)>1 and np.std(gaps)/max(1,np.mean(gaps))<.65 and max(gaps)<2*diameter
    connected=bool((near.mean()>=.78 and continuity>=.75 and near.mean()-side>.18) or
                   (dashed and near.mean()-side>.15))
    return dict(eligible=True,a=a.tolist(),b=b.tolist(),ink_support=float(near.mean()),
        continuity=continuity,parallel_background=float(side),direction_contrast=float(near.mean()-side),
        dashed_compatible=bool(dashed),connected=connected,sample_xy=xy.tolist(),sample_support=near.tolist())


def classify_connection(points,ink,diameter,scope=None,path_record=None):
    distance=cv2.distanceTransform(1-ink,cv2.DIST_L2,5)
    # One highest-evidence hypothesis per near-identical x column; record ambiguity.
    groups=[]
    for p in sorted(points,key=lambda p:p['cx']):
        if groups and p['cx']-groups[-1][0]['cx']<.5*diameter: groups[-1].append(p)
        else: groups.append([p])
    selected=[max(g,key=lambda p:p.get('confidence',0)) for g in groups]
    pairs=[chord_test([a['cx'],a['cy']],[b['cx'],b['cy']],ink,diameter,scope,distance) for a,b in zip(selected[:-1],selected[1:])]
    eligible=[p for p in pairs if p['eligible']]
    ratio=float(np.mean([p['connected'] for p in eligible])) if eligible else 0.
    alternative=[]
    if path_record:
        path=np.asarray(path_record['path'],float); obs=np.asarray(path_record['observed'],bool)
        dist=cv2.distanceTransform(1-ink,cv2.DIST_L2,5)
        obs=obs&(sample(dist,path)<=max(1.5,.1*diameter))
        for p in eligible:
            a,b=np.asarray(p['a']),np.asarray(p['b']); good=obs&(path[:,0]>a[0]+diameter)&(path[:,0]<b[0]-diameter)
            if good.sum()<10: continue
            expected=a[1]+(path[good,0]-a[0])*(b[1]-a[1])/(b[0]-a[0])
            alternative.append(float(np.median(np.abs(path[good,1]-expected))/diameter))
    n=len(eligible)
    if n>=2 and ratio>=.80 and not any(len(g)>1 for g in groups): label='connected_supported'; reason='repeated_direct_chord_ink'
    elif n>=2 and ratio<=.25:
        label='nonconnected_supported'; reason='repeated_missing_direct_connectors; fitted_vs_markers_only_not_asserted'
    else: label='uncertain'; reason='insufficient_or_conflicting_connection_evidence'
    return dict(label=label,reason=reason,eligible_pairs=n,connected_fraction=ratio,pairs=pairs,
        off_chord_path_residual_diameters=alternative,ambiguous_x_groups=sum(len(g)>1 for g in groups),
        interpretation='observational compatibility, not proof of author plotting model')




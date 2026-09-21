"""Conservative per-legend-marker scale calibration from independent outlines.

Broad scale search is confined to this calibration pass.  Its output is exactly
1.0 unless several distinct, sufficiently complete plot outlines support a
consistent non-raster-sized change.  Marker detection does not use these anchors
as detections and no annotation/result files are read at runtime.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

import partial_swatch_detector as D


THRESHOLDS={
    'minimum_independent_anchors':3,
    'minimum_outline_score':.72,
    'minimum_directional_match':.70,
    'minimum_side_support':.55,
    'minimum_angular_coverage':.875,
    'minimum_gain_vs_1x':.10,
    'minimum_relative_departure':.12,
    'minimum_absolute_departure_px':2.,
    'maximum_scale_mad':.055,
    'maximum_cluster_deviation':.085,
    'minimum_consensus_fraction':.70,
    'independent_distance_diameters':1.15,
    'edge_tolerance_px':1.6,
    'orientation_tolerance_degrees':40.,
    'minimum_class_outline_margin':.04,
}
SEARCH_SCALES=tuple(round(.75+.05*i,2) for i in range(19))


def _geometry(template):
    """Observed external boundary metadata, not newly synthesized marker ink."""
    mask=template.mask.astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    envelope=np.zeros(mask.shape,np.uint8)
    cv2.drawContours(envelope,contours,-1,1,-1)
    outer=cv2.Canny(envelope*255,40,100)>0
    # Attached horizontal-line entry points do not establish the glyph radius.
    if template.line_nuisance.shape==outer.shape:
        outer &= ~template.line_nuisance
    y,x=np.nonzero(outer)
    centre=np.array([(mask.shape[1]-1)/2,(mask.shape[0]-1)/2])
    vectors=np.column_stack((x,y)).astype(np.float32)-centre
    angles=D._edge_orientation(template.raw_soft)[y,x]
    return vectors,angles


def _kernel(vectors,scale):
    scaled=vectors*scale
    radius=max(3,int(math.ceil(np.max(np.abs(scaled))))+2)
    kernel=np.zeros((2*radius+1,2*radius+1),np.float32)
    xy=np.rint(scaled+radius).astype(int)
    np.add.at(kernel,(xy[:,1],xy[:,0]),1.)
    kernel/=max(1.,float(kernel.sum()))
    return kernel,radius


def _candidate_positions(near_edge,vectors,scale,diameter,allowed):
    kernel,radius=_kernel(vectors,scale)
    if min(near_edge.shape)<kernel.shape[0]:
        return []
    score=cv2.matchTemplate(near_edge,kernel,cv2.TM_CCORR)
    valid=allowed[radius:radius+score.shape[0],radius:radius+score.shape[1]]
    score[~valid]=0
    peak_size=2*max(2,int(round(.18*diameter*scale)))+1
    local=cv2.dilate(score,np.ones((peak_size,peak_size),np.uint8))
    yy,xx=np.nonzero((score>=.62)&(score>=local-1e-6))
    if len(xx)>60:
        order=np.argsort(score[yy,xx])[-60:]
        xx,yy=xx[order],yy[order]
    return [(float(x+radius),float(y+radius)) for x,y in zip(xx,yy)]


def _outline_fit(x,y,scale,vectors,angles,tree,edge_xy,edge_angles,membership,allowed,template):
    locations=vectors*scale+np.array([x,y])
    xy=np.rint(locations).astype(int)
    if (xy[:,0].min()<0 or xy[:,1].min()<0 or xy[:,0].max()>=allowed.shape[1]
            or xy[:,1].max()>=allowed.shape[0] or not np.all(allowed[xy[:,1],xy[:,0]])):
        return None
    distances,indices=tree.query(locations,k=min(5,len(edge_xy)),distance_upper_bound=2.01)
    if distances.ndim==1:
        distances=distances[:,None];indices=indices[:,None]
    exists=indices<len(edge_xy)
    safe=np.minimum(indices,len(edge_xy)-1)
    difference=np.abs(np.angle(np.exp(1j*(edge_angles[safe]-angles[:,None]))))
    compatible=exists & (difference<=math.radians(THRESHOLDS['orientation_tolerance_degrees']))
    best_distance=np.min(np.where(compatible,distances,np.inf),axis=1)
    matched=best_distance<=THRESHOLDS['edge_tolerance_px']
    support=np.maximum(0,1-.30*np.minimum(best_distance,4.))
    support[~np.isfinite(best_distance)]=0.
    score=float(support.mean())
    direction_match=float(matched.mean())
    sector=(np.floor((np.arctan2(vectors[:,1],vectors[:,0])+np.pi)*4/np.pi).astype(int)%8)
    sector_support=[float(matched[sector==i].mean()) for i in range(8) if np.count_nonzero(sector==i)>=2]
    angular=float(np.mean(np.asarray(sector_support)>=.50)) if sector_support else 0.
    sides=[]
    for axis,sign in ((0,-1),(0,1),(1,-1),(1,1)):
        selected=sign*vectors[:,axis]>=.25*template.diameter
        sides.append(float(matched[selected].mean()) if selected.any() else 0.)
    # Calibration needs clear anchors, unlike occlusion-tolerant detection.
    # This stops an open-circle model using the outside of a filled circle.
    radius=max(2,int(round(.19*template.diameter*scale)))
    cy,cx=int(round(y)),int(round(x))
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    disc=xx*xx+yy*yy<=radius*radius
    patch=membership[cy-radius:cy+radius+1,cx-radius:cx+radius+1]
    if patch.shape!=disc.shape:
        return None
    fill=float((patch[disc]>=.45).mean())
    fill_ok=fill<=.30 if template.marker_kind=='open' else fill>=.80
    clear=(score>=THRESHOLDS['minimum_outline_score'] and
           direction_match>=THRESHOLDS['minimum_directional_match'] and
           angular>=THRESHOLDS['minimum_angular_coverage'] and
           min(sides)>=THRESHOLDS['minimum_side_support'] and fill_ok)
    return {'x':float(x),'y':float(y),'scale':float(scale),'outline_score':score,
            'directional_match':direction_match,'angular_coverage':angular,
            'side_support':sides,'interior_fill':fill,'clear_outline':bool(clear)}


def _calibrate_one(image,template,pa,ignore_regions):
    x0,y0,x1,y1=pa
    crop=image[y0:y1,x0:x1]
    membership=D.ink_membership(crop,template.ink)
    allowed=np.ones(membership.shape,bool)
    for a,b,c,d in ignore_regions:
        allowed[max(0,b-y0):max(0,min(y1,d)-y0),max(0,a-x0):max(0,min(x1,c)-x0)]=False
    # Calibration anchors must be complete, so no plot-border fragments.
    margin=max(2,int(math.ceil(.5*template.diameter*.75)))
    allowed[:margin]=False;allowed[-margin:]=False
    allowed[:,:margin]=False;allowed[:,-margin:]=False
    binary=(membership>=.45).astype(np.uint8)
    edge=cv2.Canny(binary*255,40,100)>0
    edge &= allowed
    ey,ex=np.nonzero(edge)
    if len(ex)<12:
        return {'candidates':[],'reason':'too_few_plot_edges'}
    edge_xy=np.column_stack((ex,ey))
    tree=cKDTree(edge_xy)
    edge_angles=D._edge_orientation(membership)[ey,ex]
    distance=cv2.distanceTransform((~edge).astype(np.uint8),cv2.DIST_L2,5)
    near_edge=np.maximum(0,1-.30*np.minimum(distance,4)).astype(np.float32)
    vectors,angles=_geometry(template)
    if len(vectors)<12:
        return {'candidates':[],'reason':'too_few_reliable_template_outline_pixels'}
    raw=[]
    for scale in SEARCH_SCALES:
        for x,y in _candidate_positions(near_edge,vectors,scale,template.diameter,allowed):
            fit=_outline_fit(x,y,scale,vectors,angles,tree,edge_xy,edge_angles,membership,allowed,template)
            if fit is not None and fit['clear_outline']:
                raw.append(fit)
    # One local peak at many scales is ONE physical anchor, not many votes.
    grouped=[]
    for candidate in sorted(raw,key=lambda a:a['outline_score'],reverse=True):
        if any(math.hypot(candidate['x']-a['x'],candidate['y']-a['y'])
               <=.75*template.diameter*max(candidate['scale'],a['scale']) for a in grouped):
            continue
        grouped.append(candidate)
    anchors=[]
    for candidate in grouped:
        # Compare against the best 1x centre within a small raster shift.
        unchanged=[]
        for dx in (-2,-1,0,1,2):
            for dy in (-2,-1,0,1,2):
                fit=_outline_fit(candidate['x']+dx,candidate['y']+dy,1.,vectors,angles,
                                 tree,edge_xy,edge_angles,membership,allowed,template)
                if fit is not None:unchanged.append(fit['outline_score'])
        baseline=max(unchanged,default=0.)
        candidate={**candidate,'score_1x':float(baseline),
                   'gain_vs_1x':float(candidate['outline_score']-baseline),
                   'x':float(candidate['x']+x0),'y':float(candidate['y']+y0)}
        anchors.append(candidate)
    return {'candidates':anchors,'reason':'outline_search_complete'}


def _decide(template,anchors):
    thresholds={**THRESHOLDS,'search_scales':list(SEARCH_SCALES)}
    report={'status':'insufficient_evidence','scale':1.,'reason':'fewer_than_three_clear_independent_anchors',
            'source_diameter':float(template.diameter),'thresholds':thresholds,'anchors':[],
            'evidence':{'independent_sites':0,'consistent_sites':0}}
    independent=[]
    for a in sorted(anchors,key=lambda a:a['outline_score'],reverse=True):
        if any(math.hypot(a['x']-b['x'],a['y']-b['y']) <
               THRESHOLDS['independent_distance_diameters']*template.diameter*max(a['scale'],b['scale'])
               for b in independent):
            continue
        independent.append(a)
    report['anchors']=independent
    report['evidence']['independent_sites']=len(independent)
    if len(independent)<THRESHOLDS['minimum_independent_anchors']:
        return report
    # The dominant scale must explain most anchors, not only a cherry-picked
    # set of large ink masses among many unchanged-size markers.
    ratios=np.asarray([a['scale'] for a in independent])
    median=float(np.median(ratios))
    mad=float(np.median(np.abs(ratios-median)))
    inliers=[a for a in independent if abs(a['scale']-median)<=THRESHOLDS['maximum_cluster_deviation']]
    departure=max(THRESHOLDS['minimum_relative_departure'],THRESHOLDS['minimum_absolute_departure_px']/template.diameter)
    gain=float(np.median([a['gain_vs_1x'] for a in inliers])) if inliers else 0.
    report['evidence'].update(consistent_sites=len(inliers),median_scale=median,scale_mad=mad,
        consensus_fraction=len(inliers)/len(independent),median_gain_vs_1x=gain,
        minimum_detectable_relative_departure=float(departure))
    if len(inliers)<3 or len(inliers)/len(independent)<THRESHOLDS['minimum_consensus_fraction'] or mad>THRESHOLDS['maximum_scale_mad']:
        report['reason']='inconsistent_scale_across_independent_sites'
    elif abs(median-1.)<=departure:
        report.update(status='unchanged',reason='difference_within_raster_uncertainty')
    elif gain<THRESHOLDS['minimum_gain_vs_1x']:
        report['reason']='insufficient_outline_improvement_over_1x'
    elif sum(a['gain_vs_1x']>=THRESHOLDS['minimum_gain_vs_1x'] for a in inliers)<3:
        report['reason']='fewer_than_three_individually_improved_anchors'
    else:
        report.update(status='approved',scale=median,reason='consistent_complete_outline_scale_change')
    return report


def _confirm_shared_raster(image,template,report,plot_area,ignore_regions):
    """Confirm ONE actual raster model across the independent outline anchors.

    Continuous boundary vectors can favour a scale whose nearest-neighbour
    raster has a different stroke thickness/phase.  A shared, narrowly bounded
    calibration search resolves that mismatch before any marker detection.
    It does not let individual detections choose their own scale.
    """
    if report['status']!='approved':
        return report
    from occlusion_aware_window_verifier import verify_marker_window
    outline_scale=float(report['scale'])
    anchors=[a for a in report['anchors'] if abs(a['scale']-outline_scale)<=THRESHOLDS['maximum_cluster_deviation']]
    step=float(SEARCH_SCALES[1]-SEARCH_SCALES[0])
    lower=max(min(SEARCH_SCALES),min(a['scale'] for a in anchors)-step)
    upper=min(max(SEARCH_SCALES),max(a['scale'] for a in anchors)+step)
    departure=report['evidence']['minimum_detectable_relative_departure']
    scales=sorted({outline_scale,*[s for s in SEARCH_SCALES if lower-1e-8<=s<=upper+1e-8]})
    scales=[s for s in scales if abs(s-1.)>departure]
    required=max(THRESHOLDS['minimum_independent_anchors'],
                 int(math.ceil(THRESHOLDS['minimum_consensus_fraction']*len(report['anchors']))))
    pa=plot_area

    def evaluate(scale):
        variant=D.scale_swatch_template(template,scale)
        checks=[]
        for anchor in anchors:
            v=verify_marker_window(image,variant,anchor['x'],anchor['y'],fixed_geometry=True)
            inside=(pa[0]<=v.aligned_x<pa[2] and pa[1]<=v.aligned_y<pa[3] and
                    not any(a<=v.aligned_x<c and b<=v.aligned_y<d for a,b,c,d in ignore_regions))
            checks.append({'x':anchor['x'],'y':anchor['y'],'aligned_x':float(v.aligned_x),
                'aligned_y':float(v.aligned_y),'decision':v.decision,
                'verified':bool(v.decision=='verified' and inside),
                'required_recall':float(v.required_recall),'strict_core_recall':float(v.strict_core_recall),
                'aligned_centre_in_scope':bool(inside)})
        return {'scale':float(scale),'verified_sites':sum(c['verified'] for c in checks),
                'median_required_recall':float(np.median([c['required_recall'] for c in checks])),
                'median_strict_core_recall':float(np.median([c['strict_core_recall'] for c in checks])),
                'checks':checks}

    unit=evaluate(1.)
    tested=[evaluate(s) for s in scales]
    eligible=[r for r in tested if r['verified_sites']>=required]
    validation={'outline_proposed_scale':outline_scale,'supported_scale_interval':[float(lower),float(upper)],
                'required_verified_sites':int(required),'unit_control':unit,'tested_shared_scales':tested,
                'selected_shared_scale':None}
    report['raster_validation']=validation
    if not eligible:
        report.update(status='insufficient_evidence',scale=1.,
                      reason='no_shared_raster_scale_verifies_enough_independent_anchors')
        report['evidence']['shared_raster_verified_sites']=max((r['verified_sites'] for r in tested),default=0)
        return report
    # Number of actually verified independent sites takes precedence over a
    # slightly higher average achieved by sacrificing an anchor. Unit controls
    # are recorded, but solid interiors may legitimately also pass at 1x;
    # the preceding positive-outline improvement establishes the size change.
    chosen=max(eligible,key=lambda r:(r['verified_sites'],
        .7*r['median_required_recall']+.3*r['median_strict_core_recall'],-abs(r['scale']-outline_scale)))
    report.update(scale=chosen['scale'],reason='consistent_outline_change_and_shared_fixed_raster_validation')
    report['evidence']['shared_raster_verified_sites']=chosen['verified_sites']
    report['evidence']['shared_raster_unit_verified_sites']=unit['verified_sites']
    validation['selected_shared_scale']=chosen['scale']
    return report


def calibrate_swatch_scales(image,templates,plot_area,ignore_regions=()):
    """Return per-swatch scale decisions; shape names need not be unique."""
    pa=tuple(map(int,plot_area))
    if len(pa)!=4 or not(0<=pa[0]<pa[2]<=image.shape[1] and 0<=pa[1]<pa[3]<=image.shape[0]):
        raise ValueError('Invalid plot area for scale calibration')
    names=[t.key for t in templates]
    if len(set(names))!=len(names):
        raise ValueError('Scale calibration requires one distinct identity per legend swatch')
    searches={t.key:_calibrate_one(image,t,pa,ignore_regions) for t in templates}
    reports={}
    for t in templates:
        anchors=[]
        for a in searches[t.key]['candidates']:
            # A scale anchor must not be better explained by a different
            # legend shape. This comparison uses positive outline evidence.
            rivals=[b for name,r in searches.items() if name!=t.key for b in r['candidates']
                    if math.hypot(a['x']-b['x'],a['y']-b['y'])<=.35*t.diameter*max(a['scale'],b['scale'])]
            if rivals and a['outline_score']-max(b['outline_score'] for b in rivals)<THRESHOLDS['minimum_class_outline_margin']:
                continue
            anchors.append(a)
        report=_decide(t,anchors)
        report=_confirm_shared_raster(image,t,report,pa,ignore_regions)
        report['search_reason']=searches[t.key]['reason']
        report['evidence']['clear_sites_before_class_competition']=len(searches[t.key]['candidates'])
        report['swatch_id']=t.swatch_id
        report['shape_hint']=t.shape_hint or t.name
        reports[t.key]=report
    return reports

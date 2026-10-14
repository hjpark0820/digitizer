"""Context checks for an observed marker plus centered-line hypothesis.

Local marker similarity alone can explain a letter fragment. Lines used to
explain the body must continue on at least one side outside it. Extra nearby
ink is measured symmetrically, without calling it missing marker ink.
"""
import cv2
import numpy as np
from bw_centered_composite_v46 import stroke
from bw_layered_composite_v46 import compose


def external_strokes(patch, body, *, centered_only=False, finite_exterior=False,
                     available=None):
    """Explain extra ink only with thin strokes measured OUTSIDE the body.

    This mask can reduce a nuisance cost, never increase marker recall.
    Width and strength are fitted on the exterior so a filled glyph cannot
    explain away its own interior as an error bar.
    """
    h,w=patch.shape;yy,xx=np.indices(patch.shape,dtype=float)
    outer=cv2.dilate(body.astype(np.uint8),np.ones((3,3),np.uint8))==0
    if available is not None:
        outer &= np.asarray(available, bool)
    ys,xs=np.nonzero(body);diameter=max(np.ptp(xs)+1,np.ptp(ys)+1) if len(xs) else 3
    # Suppress the held-out body even during proposal generation.
    binary=np.uint8((patch>.25)&outer)*255
    lines=cv2.HoughLinesP(binary,1,np.pi/180,threshold=max(4,round(.4*diameter)),
        minLineLength=max(4.,.5*diameter),maxLineGap=1)
    cover=np.zeros_like(patch);records=[]
    segments=[] if lines is None else list(lines.reshape(-1,4))
    if finite_exterior:
        # Thick horizontal/vertical caps have several equally dark rows.
        # Hough can return only a slanted fragment of such a cap. Add complete
        # finite axial runs, measured on the same held-out exterior pixels.
        axial=[];minimum=max(4,int(round(.5*diameter)))
        for horizontal in (True,False):
            kernel=np.ones((1,minimum) if horizontal else (minimum,1),np.uint8)
            opened=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
            count,labels,stats,_=cv2.connectedComponentsWithStats(opened,8)
            for x,y,sw,sh,area in stats[1:]:
                length,width=(sw,sh) if horizontal else (sh,sw)
                if length<minimum or width>max(3.,.20*diameter):continue
                if horizontal:axial.append((x,y+(sh-1)/2.,x+sw-1,y+(sh-1)/2.))
                else:axial.append((x+(sw-1)/2.,y,x+(sw-1)/2.,y+sh-1))
        # Long, independently observed exterior segments are tested first.
        # The new context route never extrapolates a segment through a body.
        by_length=lambda p:-(float(p[2]-p[0])**2+float(p[3]-p[1])**2)
        segments=sorted(axial,key=by_length)[:12]+sorted(segments,key=by_length)[:24]
    for a,b,c,d in segments:
        vx,vy=float(c-a),float(d-b);length=np.hypot(vx,vy)
        if length<1:continue
        vx/=length;vy/=length;normal=(-vy,vx)
        perp=(xx-a)*normal[0]+(yy-b)*normal[1]
        along=(xx-a)*vx+(yy-b)*vy
        measured_domain=outer
        if finite_exterior:
            measured_domain=outer&(along>=-.5)&(along<=length+.5)
        if centered_only:
            center_distance=abs(((w-1)/2-a)*normal[0]+((h-1)/2-b)*normal[1])
            if center_distance>1.5:continue
            if records and not (abs(normal[0])>.94 or abs(records[0]['normal'][0])>.94):continue
        core=measured_domain&(abs(perp)<.8)
        side=(measured_domain&(abs(perp)>=2.5)&(abs(perp)<=3.5) if finite_exterior
              else measured_domain&(abs(perp)>2.5)&(abs(perp)<3.5))
        if core.sum()<4 or side.sum()<3:continue
        strength=float(np.median(patch[core]));contrast=strength-float(np.median(patch[side]))
        if strength<.25 or contrast<.15:continue
        if finite_exterior:
            if float(np.mean(patch[core]>.25))<.75:continue
            # A broad dark neighbour is not a thin cap. Independently sampled
            # flanks must both be lighter when both exist in the observation.
            flanks=[measured_domain&(perp*sign>=2.5)&(perp*sign<=3.5) for sign in (-1,1)]
            if any(q.sum()>=3 and strength-float(np.median(patch[q]))<.15 for q in flanks):continue
        offset=a*normal[0]+b*normal[1]
        if any(abs(np.dot(normal,q['normal']))>.97 and abs(abs(offset)-abs(q['offset']))<1.5 for q in records):continue
        profile=[];positions=np.arange(-2.5,2.51,.5)
        for delta in positions:
            band=measured_domain&(abs(perp-delta)<.4)
            profile.append(float(np.median(patch[band])) if band.sum()>=3 else 0.)
        predicted=np.interp(perp,positions,profile,left=0,right=0)
        if finite_exterior:
            predicted*=measured_domain
        cover=np.maximum(cover,predicted)
        record=dict(normal=list(normal),offset=float(offset),strength=strength)
        if finite_exterior:
            record.update(endpoints=[[int(a),int(b)],[int(c),int(d)]],
                          finite_exterior_only=True,body_pixels_used=0,
                          explained_body_pixels=0,core_pixels=int(core.sum()))
        records.append(record)
        if len(records)==(2 if centered_only else (8 if finite_exterior else 4)):break
    return cover,records


def evidence(ink, center, marker, cover, parameters, reliability=None, *, finite_exterior=True):
    r=len(marker)//2
    patch=cv2.getRectSubPix(ink,(len(marker),len(marker)),tuple(map(float,center)))
    body=np.maximum(marker,cover)>.08
    yy0,xx0=np.nonzero(body)
    if len(xx0)>=3:
        envelope=np.zeros_like(body,np.uint8)
        cv2.fillConvexPoly(envelope,cv2.convexHull(np.column_stack((xx0,yy0)).astype(np.int32)),1)
        body=envelope>0  # Nuisance exclusion only, never positive marker ink.
    yy,xx=np.indices(marker.shape)
    lines=[p['amplitude']*stroke(len(marker),(r,r),p['angle'],p['width'])
           for p in parameters['lines']]
    scaled=np.clip(marker*parameters['gain'],0,1)
    model=compose(scaled,np.maximum(cover,scaled),lines,
                  [p['order'] for p in parameters['lines']])
    extensions=[]
    # The body band is excluded: marker ink cannot validate its own nuisance.
    outside=cv2.dilate(body.astype(np.uint8),np.ones((3,3),np.uint8))==0
    for p,line in zip(parameters['lines'],lines):
        angle=np.deg2rad(p['angle']);along=(xx-r)*np.cos(angle)+(yy-r)*np.sin(angle)
        ray_scores=[]
        for sign in (-1,1):
            w=line*outside*(sign*along>0)
            if w.sum()<.1:continue
            support=np.clip(patch/np.maximum(line,.05),0,1)
            ray_scores.append(float((w*support).sum()/w.sum()))
        extensions.append(max(ray_scores,default=0.))
    # Require precision as well as recall, on the identical observed canvas.
    # Small antialiasing/model errors are not additional objects.
    weight=np.ones_like(patch) if reliability is None else np.asarray(reliability,np.float32)
    nuisance,measured=external_strokes(patch,body,finite_exterior=finite_exterior,
                                      centered_only=not finite_exterior)
    extra=np.maximum(patch-np.maximum(model,nuisance)-.15,0.)
    extra_fraction=float(np.sum(weight*extra**2)/max(np.sum(weight*patch**2),1.e-6))
    # Common-domain marker consistency and exterior context are distinct.
    # Zero-weight padding is not reintroduced into the marker's score, but
    # unexplained exterior blobs must not vanish from the context decision.
    exterior_weight=(weight<=0).astype(np.float32)
    context_weight=weight+exterior_weight
    context_energy=max(float(np.sum(context_weight*patch**2)),1.e-6)
    exterior_fraction=float(np.sum(exterior_weight*extra**2)/context_energy)
    context_fraction=float(np.sum(context_weight*extra**2)/context_energy)
    return dict(version='observed-raster-context-v2-finite-exterior',line_extensions=extensions,
                minimum_line_extension=min(extensions,default=1.),
                external_strokes=measured,
                extra_ink_fraction=extra_fraction,
                exterior_extra_ink_fraction=exterior_fraction,
                total_context_extra_ink_fraction=context_fraction,
                marker_domain_pixels=int(np.count_nonzero(weight)),
                exterior_context_pixels=int(np.count_nonzero(exterior_weight)),
                historical_context_passed=bool(min(extensions,default=1.)>=.45 and context_fraction<=.20),
                passed=bool(min(extensions,default=1.)>=.45 and extra_fraction<=.20 and context_fraction<=.20))


def selected_body_fill(ink, center, marker, cover, marker_weight, parameters, style):
    """Recheck a native fill contradiction on the EXACT selected raster body.

    A nearby native model may use a different size. Its verdict only requests
    this measurement; it cannot veto a changed raster body without new source
    samples. Exterior strokes are never used to explain internal mismatch.
    """
    from bw_geometry_first_v46 import compare_fill,fill_statistics
    marker=np.asarray(marker,np.float32);cover=np.asarray(cover,np.float32)
    patch=cv2.getRectSubPix(ink,marker.shape[::-1],tuple(map(float,center)))
    reliable=np.asarray(marker_weight)>.2
    body=cover>.5
    yy,xx=np.nonzero(body)
    diameter=max(np.ptp(xx)+1,np.ptp(yy)+1) if len(xx) else 0.
    if style=='open':
        # Only source-observed low-ink islands, not an inferred target hole.
        region=body&(marker<.1)&reliable
    elif style=='solid':
        # Source connector uncertainty can split a solid observed raster into
        # two lobes. The envelope locates its interior only; unreliable gap
        # pixels remain excluded and contribute no invented source ink.
        if len(xx)>=3:
            envelope=np.zeros_like(body,np.uint8)
            cv2.fillConvexPoly(envelope,cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)),1)
            body=envelope>0
        depth=cv2.distanceTransform(body.astype(np.uint8),cv2.DIST_L2,5)
        region=(depth>=max(1.5,.18*diameter))&reliable&(marker>.5)
    else:
        return dict(decision='abstain',reason='fill_style_not_independently_known')
    source=fill_statistics(marker[region])
    expected=dict(style=style,mean_ink=source['mean'],std_ink=source['std'])
    visible=region.copy()
    if style=='open':
        # A fitted front line has already passed its exterior continuation
        # check. A behind line cannot darken an opaque observed source hole.
        r=len(marker)//2
        for p in parameters['lines']:
            if p['order']=='front':
                line=p['amplitude']*stroke(len(marker),(r,r),p['angle'],p['width'])
                visible &= line<.1
    result=compare_fill(expected,fill_statistics(patch[visible]),int(visible.sum()),
                        float(visible.sum()/max(1,region.sum())))
    result.update(version='selected-raster-body-tone-v1',center=list(map(float,center)),
        raster_scale=parameters.get('raster_scale'),source_pixels=int(region.sum()),
        source_tone=source,exact_selected_bank=True,exterior_pixels_used=0,
        fill_thresholds='existing bw_geometry_first_v46.compare_fill')
    return result


def native_identity_supported(record, expected_keys=None):
    """Preserve a full-window winner, not merely a high ink recall.

An explicit same-observation rival comparison and line-null comparison must
both support this identity. A weak grid vote or high confidence is not enough.
"""
    g=record.get('geometry_first',{})
    rivals=g.get('model_losses',{})
    # A specialized hollow-only comparison is not an identity verdict
    # against a triangle/X/filled rival that was never evaluated.
    if expected_keys is not None and not set(expected_keys).issubset(rivals):
        return False
    own=rivals.get(record.get('template'))
    if record.get('decision')!='verified' or g.get('decision')!='compatible' or own is None:
        return False
    if g.get('native_context',{}).get('passed') is False:
        return False
    other=[v for k,v in rivals.items() if k!=record['template']]
    # Scene fallback uses an asymmetric 0.4-weighted surplus-ink loss;
    # its improvement is not on the original verifier's MSE scale.
    minimum_improvement=.012 if g.get('version')=='small-hollow-scene-fallback-v1' else .06
    return bool(other and min(other)>own and own<=.15 and
                g.get('marker_improvement',0.)>=minimum_improvement and
                g.get('missing_rim_fraction',1.)<=.25)


def native_body_supported(records, key, center, diameter):
    """Reuse a verified larger-window body, not its old identity ranking.

    This only supplies the physical-body/context check. The caller must still
    require the complete observed-raster competition to select the same key.
    A rejected or merely high-recall candidate cannot bypass context checks.
    An overlap-abstained hollow can provide body evidence only when the full
    window still prefers that same hollow, explains its rim, and beats the
    line-only alternative. Final identity still requires ALL raster rivals.
    """
    x,y=center
    for p in records:
        g=p.get('geometry_first',{})
        if p.get('template')!=key:continue
        verified=(p.get('decision')=='verified' and g.get('decision')=='compatible'
                  and g.get('geometry_decision')=='verified')
        losses=g.get('model_losses',{});own=losses.get(key)
        supported_overlap=(p.get('decision')=='ambiguous' and g.get('decision')=='abstain'
            and g.get('reason')=='small_hollow_overlap_or_model_residual'
            and own is not None and own<=.15 and own==min(losses.values())
            and g.get('missing_rim_fraction',1.)<=.25 and g.get('marker_improvement',0.)>=.06)
        if not (verified or supported_overlap):continue
        if np.hypot(p.get('aligned_x',p['x'])-x,p.get('aligned_y',p['y'])-y)>max(1.,.15*diameter):
            continue
        if p.get('ink_evidence',{}).get('direct_required_recall',0.)<.80:continue
        return dict(candidate_index=p.get('candidate_index'),center=[p.get('aligned_x',p['x']),p.get('aligned_y',p['y'])],
                    source='verified_native_full_window_body' if verified else 'independently_supported_hollow_overlap',
                    identity_supplied_by='complete_raster_competition')
    return None

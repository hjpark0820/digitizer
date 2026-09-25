"""Production colour legend hybrid: native v45 priors and observed body refinement.

Discovery uses chromatic connected bodies, not a grayscale crop containing the
first label letter. Closing is used only for component grouping; no fabricated
pixels enter raw images, templates, masks or weights. Both horizontal connectors
and vertical error bars are measured by the same transposed span calculation.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import partial_swatch_detector as D
import bw_legend_v46 as L
from bw_legend_shape import classify_observed_shape
from color_membership_v46 import chromatic_evidence, reference_contrast

VERSION = 'colour-legend-hybrid-v46-validated-neutral-priors'
FOREGROUND_THRESHOLD = .50
WEAK_SUPPORT_THRESHOLD = .05


def _box(value, image):
    return L._box(value, image)


def _shape(mask, nuisance=None, return_report=False):
    result = classify_observed_shape(mask, nuisance)
    return result if return_report else result[:2]


def _soft_fill_evidence(mask, soft, central):
    """Do not mistake a low-confidence coloured halo for a filled interior."""
    yy,xx=np.nonzero(mask)
    hull=cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32))
    envelope=np.zeros(mask.shape,np.uint8)
    cv2.fillConvexPoly(envelope,hull,1)
    distance=cv2.distanceTransform(np.pad(envelope,1),cv2.DIST_L2,5)[1:-1,1:-1]
    inside=(distance>=max(1.5,.45*float(distance.max()))) & ~central
    count=int(inside.sum())
    return dict(interior_pixels=count,
                mean_membership=float(soft[inside].mean()) if count else None,
                strong_fraction=float((soft[inside]>=.55).mean()) if count else None,
                weak_fraction=float((soft[inside]<.30).mean()) if count else None)


def _lab(image):
    return cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def _colour_evidence(image, model):
    """Separate colour identity from observed source-paper ink strength.

    JPEG can darken and desaturate a genuinely red edge without moving it
    along an ink-to-white ray. Hue direction remains evidence of identity;
    source-paper contrast, normalized once per model, measures ink strength.
    Weak pink/blue halo pixels stay continuous evidence but are not promoted
    to binary foreground merely because their hue matches the legend.
    """
    pixels=np.asarray(image,np.float32)
    paper=np.asarray(model.paper_bgr,np.float32)
    contrast=np.linalg.norm(np.maximum(paper-pixels,0.),axis=-1)
    reference=float(getattr(model,'colour_reference_contrast',
                            np.linalg.norm(paper-np.asarray(model.core_bgr,float))))
    strength=contrast/max(reference,8.)
    chroma=np.ptp(pixels,axis=-1)
    if model.achromatic:
        # A neutral prior cannot claim distinctly coloured marker pixels.
        colour_fit=np.clip((30.-chroma)/6.,0.,1.)
        owns=chroma<=24.
    else:
        soft, _ = chromatic_evidence(image, paper, model.core_bgr, reference)
        return soft, (chroma >= 4.) & (soft >= FOREGROUND_THRESHOLD)
    soft=(np.clip(strength,0.,1.)*colour_fit).astype(np.float32)
    strong=owns&(strength>=FOREGROUND_THRESHOLD)
    return soft,strong


def _membership(image, model):
    return _colour_evidence(image,model)[0]


def _strong_support(image, model):
    return _colour_evidence(image,model)[1]


def _groups(support):
    grouped = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    result = []
    for idx, (x,y,w,h,_) in enumerate(stats[1:], 1):
        observed = support & (labels == idx)
        if observed.sum() < 6 or min(w,h) < 3:
            continue
        result.append((observed, (int(x),int(y),int(x+w),int(y+h))))
    return result


def _tight(mask):
    yy,xx=np.nonzero(mask)
    if not len(xx):
        raise ValueError('No observed marker support')
    return int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)


def _horizontal_body(support):
    """Measure marker bulge; return bounds and a line band, never filled ink."""
    x0,y0,x1,y1=_tight(support)
    width,height=x1-x0,y1-y0
    if width < 1.25*height:
        return None
    spans=np.array([np.ptp(np.flatnonzero(c))+1 if c.any() else 0 for c in support.T])
    outer=max(1,int(round(.16*width)))
    tails=np.r_[np.arange(x0,x0+outer),np.arange(x1-outer,x1)]
    positive=spans[tails][spans[tails]>0]
    if not len(positive):
        return None
    thickness=float(np.percentile(positive,65))
    elevated=np.flatnonzero(spans>max(thickness+.75,.50*height))
    if height < max(3.,1.6*thickness) or not len(elevated):
        return None
    # A lone JPEG tail pixel must not extend a central bulge to the end of
    # the connector. Seed from the strongest consecutive tall-column run,
    # then expand across adjacent observed shoulders (including triangle bases).
    runs=np.split(elevated,np.flatnonzero(np.diff(elevated)>1)+1)
    strongest=max(runs,key=lambda run:float(np.sum(spans[run]-thickness)))
    a,b=int(strongest[0]),int(strongest[-1])+1
    while a>x0 and spans[a-1]>thickness+.75:
        a-=1
    while b<x1 and spans[b]>thickness+.75:
        b+=1
    if b-a>1.7*height or a<=x0 or b>=x1:
        return None
    if not support[:,x0:a].any() or not support[:,b:x1].any():
        return None
    line_rows=support[:,tails].mean(axis=1)>=.20
    # Include the ambiguous transition columns in the body; the template keeps
    # their actual pixels, never a completed circle/square or convex-hull fill.
    return (max(x0,a-1),y0,min(x1,b+1),y1),line_rows


def _repeated_horizontal_body(support):
    """Recognize a line key containing three repeated compact marker bodies.

    Tail columns cannot estimate line width when the legend itself repeats
    markers at both ends. Use the intervening thin continuous connector, and
    retain only the middle observed body. Pure dashes/letters do not establish
    three aligned bulges plus a continuous shared connector.
    """
    x0,y0,x1,y1 = _tight(support)
    height = y1-y0
    if height < 3 or x1-x0 < 4*height:
        return None
    spans = np.array([np.ptp(np.flatnonzero(c))+1 if c.any() else 0 for c in support.T])
    positive = spans[x0:x1][spans[x0:x1]>0]
    thin = float(np.percentile(positive, 30))
    if height < max(3., 1.6*thin):
        return None
    elevated = np.flatnonzero(spans > max(thin+.75, .60*height))
    runs = np.split(elevated, np.flatnonzero(np.diff(elevated)>1)+1)
    runs = [r for r in runs if max(2., .25*height) <= len(r) <= 1.5*height]
    if len(runs) < 3:
        return None
    centers = np.array([(r[0]+r[-1])/2 for r in runs])
    distances = np.diff(centers)
    if np.ptp(distances) > max(1., .25*float(np.median(distances))):
        return None
    gap_columns = []
    for left,right in zip(runs,runs[1:]):
        gap = np.arange(left[-1]+1,right[0])
        if len(gap)<max(2,height) or not np.all((spans[gap]>0)&(spans[gap]<=thin+.5)):
            return None
        gap_columns.extend(gap.tolist())
    line_rows = support[:,gap_columns].mean(axis=1) >= .90
    if not line_rows.any():
        return None
    # Every compact body must extend on both sides of the same connector.
    ly = np.flatnonzero(line_rows)
    for run in runs:
        ys = np.flatnonzero(support[:,run].any(axis=1))
        if not len(ys) or ys[0]>=ly[0] or ys[-1]<=ly[-1]:
            return None
    middle = runs[int(np.argmin(abs(centers-(x0+x1-1)/2)))]
    return (max(x0,int(middle[0])-1),y0,min(x1,int(middle[-1])+2),y1),line_rows


def _body(support):
    box=_tight(support)
    x0,y0,x1,y1=box
    horizontal=_horizontal_body(support)
    vertical=_horizontal_body(support.T)
    if horizontal is None:
        horizontal=_repeated_horizontal_body(support)
    if vertical is None:
        vertical=_repeated_horizontal_body(support.T)
    nuisance=np.zeros_like(support,bool)
    central_line=np.zeros_like(support,bool)
    direction=None
    if horizontal is not None and (vertical is None or x1-x0>=y1-y0):
        box,rows=horizontal
        central_line[:]=rows[:,None]
        direction='horizontal'
    elif vertical is not None:
        transposed,rows=vertical
        a,b,c,d=transposed;box=(b,a,d,c)
        central_line[:]=rows[None,:]
        direction='vertical'
    else:
        w,h=x1-x0,y1-y0
        if max(w,h)>1.9*min(w,h) or min(w,h)<3:
            raise ValueError('Line-only or elongated word without an observed marker bulge')
    a,b,c,d=box
    body=np.zeros_like(support,bool);body[b:d,a:c]=True
    nuisance=central_line & ~body
    marker=support & body
    if marker.sum()<6 or min(c-a,d-b)<3:
        raise ValueError('Insufficient two-dimensional marker body')
    return box,nuisance,central_line,marker,direction


def _model(pixels, paper, rivals=(), prior_rgb=None):
    # Preserve native early colour priors; never resample a wide label box.
    flat=np.asarray(pixels,np.uint8).reshape(-1,3)
    if not len(flat):
        raise ValueError('No visible ink colour')
    if prior_rgb is not None:
        bgr=tuple(int(v) for v in reversed(prior_rgb))
    else:
        distance=np.linalg.norm(np.asarray(paper,float)-flat,axis=1)
        core=flat[distance>=np.percentile(distance,75)]
        bgr=tuple(int(v) for v in np.median(core,axis=0))
    neutral=max(bgr)-min(bgr)<18
    gray=lambda q:float(cv2.cvtColor(np.asarray(q,np.uint8).reshape(1,1,3),cv2.COLOR_BGR2GRAY)[0,0])
    # Retain the InkModel compatibility radius for existing consumers. The
    # hybrid extractor itself uses hue and source contrast, not this Lab tube.
    model=D.InkModel(neutral,bgr,paper,16.,gray(bgr),gray(paper))
    contrast=np.linalg.norm(np.maximum(np.asarray(paper,float)-flat,0.),axis=1)
    if neutral:
        reference=float(np.linalg.norm(np.asarray(paper,float)-np.asarray(bgr,float)))
        # Do not use an adjacent black letter to calibrate a pale-grey key.
        eligible=(np.ptp(flat.astype(float),axis=1)<24)&(contrast>=8.)&(contrast<=1.30*max(reference,8.))
    else:
        hue=flat.astype(float)-flat.min(axis=1,keepdims=True)
        ink_hue=np.asarray(bgr,float)-min(bgr)
        cosine=(hue@ink_hue)/np.maximum(np.linalg.norm(hue,axis=1)*np.linalg.norm(ink_hue),1.)
        eligible=(cosine>=math.cos(math.radians(15.)))&(contrast>=8.)&(np.ptp(flat.astype(float),axis=1)>=4.)
    # Freeze this source-component normalization before body measurement.
    # Final crop extraction must not recalculate it from a smaller rectangle.
    reference=float(np.percentile(contrast[eligible],90)) if eligible.any() else float(np.linalg.norm(np.asarray(paper,float)-np.asarray(bgr,float)))
    if not neutral:
        reference = reference_contrast(flat, paper, bgr)
    model.colour_reference_contrast=max(reference,8.)
    return model


def _candidate(legend, support, initial_box, paper, prior_model=None):
    model=prior_model if prior_model is not None else _model(legend[support],paper)
    x0,y0,x1,y1=initial_box
    x0,y0=max(0,x0-2),max(0,y0-2)
    x1,y1=min(legend.shape[1],x1+2),min(legend.shape[0],y1+2)
    local=legend[y0:y1,x0:x1]
    near=cv2.dilate(support[y0:y1,x0:x1].astype(np.uint8),np.ones((3,3),np.uint8))>0
    soft=_membership(local,model)
    observed=_strong_support(local,model)&near
    body,nuisance,central,marker,direction=_body(observed)
    a,b,c,d=body
    # Neighbouring dash fragments are not independently trusted as squares.
    # A larger two-dimensional body in the same short row segment wins below.
    return dict(model=model,component_box=(x0,y0,x1,y1),local_body=body,
                body_box=(x0+a,y0+b,x0+c,y0+d),
                center=((x0+a+x0+c-1)/2,(y0+b+y0+d-1)/2),
                marker_pixels=int(marker.sum()),diameter=max(c-a,d-b),
                local_soft=soft,observed=observed,nuisance=nuisance,
                central_line=central,marker=marker,direction=direction)


def _remove_fragments(candidates):
    kept=[];rejected=[]
    for c in candidates:
        cx,cy=c['center'];a,b,e,f=c['body_box'];w,h=e-a,f-b
        rival=next((r for r in candidates if r is not c and
            abs(r['center'][1]-cy)<=max(2.,.45*max(h,r['diameter'])) and
            abs(r['center'][0]-cx)<2.4*max(c['diameter'],r['diameter']) and
            c['diameter']<.76*r['diameter'] and c['marker_pixels']<.75*r['marker_pixels']),None)
        if rival is None:
            kept.append(c)
        else:
            rejected.append(dict(box=list(c['body_box']),reason='Small row fragment beside a more complete marker body'))
    # Components can merge/discover the same body twice, but equal markers in
    # distinct table cells remain separate swatches with separate IDs.
    return kept,rejected


def _coherent_columns(candidates):
    """Repeated rows establish graphical columns, not arbitrary row text."""
    columns=[]
    for entry in candidates:
        x,y=entry['center']
        aligned=[other for other in candidates if
                 abs(x-other['center'][0])<=max(3.,.45*min(entry['diameter'],other['diameter']))]
        if not any(abs(y-other['center'][1])>max(3.,.6*min(entry['diameter'],other['diameter'])) for other in aligned):
            continue
        center=float(np.median([other['center'][0] for other in aligned]))
        diameter=float(np.median([other['diameter'] for other in aligned]))
        if not any(abs(center-c[0])<=max(3.,.4*min(diameter,c[1])) for c in columns):
            columns.append((center,diameter))
    return columns


def _right_label(legend,entry,reference_diameter):
    """A new one-row column needs text to its right, not a plot curve tail."""
    x0,y0,x1,y1=entry['body_box'];_,cy=entry['center']
    h,w=legend.shape[:2]
    if min(x0,y0)<0 or x1>=w or y1>=h:
        return False
    # Use the established table's character scale, not an oversized plot
    # fragment's height, which could accidentally reach text in the row above.
    radius=max(3.,.65*reference_diameter)
    top,bottom=max(0,int(cy-radius)),min(h,int(cy+radius+1))
    left,right=x1,min(w,int(x1+4*reference_diameter+15))
    region=legend[top:bottom,left:right]
    if region.size==0:
        return False
    # Labels can be coloured too. Spatial character structure, not neutrality,
    # supplies the right-hand text evidence (e.g. multi-line cohort legends).
    text=region.min(axis=2)<180
    n,_,stats,_=cv2.connectedComponentsWithStats(text.astype(np.uint8),8)
    bodies=[s for s in stats[1:] if s[4]>=3 and s[3]>=max(3,.30*reference_diameter)]
    return len(bodies)>=2 or any(s[2]>=1.2*reference_diameter and s[4]>=10 for s in bodies)


def _anchored_label_run(legend, entry, anchors):
    """Locate an existing key's first text run, without recognizing its words.

    A local hole/spacing test cannot distinguish '80' or 'BI' from a hollow
    key. Join character columns across small word spaces, but stop at a real
    inter-column gutter. Only the first multi-character run after an observed
    chromatic key is its label; later isolated keys retain their own evidence.
    Coordinates in this diagnostic are relative to the legend crop.
    """
    x,y=entry['center'];h,w=legend.shape[:2]
    spread=np.ptp(legend.astype(np.int16),axis=2)
    neutral=(spread<24)&(legend.min(axis=2)<180)
    for anchor in anchors:
        diameter=anchor['diameter'];ax,ay=anchor['center']
        if x<=ax or abs(y-ay)>max(2.,.3*diameter):
            continue
        # The observed connector belongs to the key, not to its label.
        bx,by,_,_=anchor['component_box']
        _,_,right,_=_tight(anchor['observed'])
        key_right=max(anchor['body_box'][2],bx+right)
        radius=max(3.,.65*diameter)
        top,bottom=max(0,int(ay-radius)),min(h,int(ay+radius+1))
        occupied=neutral[top:bottom].any(axis=0)
        columns=np.flatnonzero(occupied & (np.arange(w)>=key_right))
        if not len(columns) or columns[0]-key_right>4*diameter+15:
            continue
        # Estimate a text run, not an unbounded 'everything to the right' box.
        gaps=np.diff(columns)-1
        breaks=np.flatnonzero(gaps>=max(4.,diameter))
        end=int(breaks[0]+1) if len(breaks) else len(columns)
        run=columns[:end];start,stop=int(run[0]),int(run[-1]+1)
        character_gaps=int(np.count_nonzero(np.diff(run)>1))
        if stop-start>=1.4*diameter and character_gaps>=1 and start<=x<stop:
            return dict(inside=True,label_box=[start,top,stop,bottom],
                anchor_center=[float(ax),float(ay)],character_gaps=character_gaps,
                word_gap_limit_px=float(max(4.,diameter)))
    return dict(inside=False)


def _neutral_row_evidence(legend, entry, reference_diameter, shape, evidence, anchors=()):
    """Separate horizontal neutral swatch identity from exact symbol identity.

    A triangle's bounding rectangle includes paper corners. Measure fill only
    inside its observed envelope, and require an isolated key + right label to
    avoid admitting letters. This returns evidence, never completed mask ink.
    """
    a,b,c,d=entry['body_box']
    if entry['direction'] is not None:
        # Measure separation outside observed connector tails, not inside them.
        bx,by,_,_=entry['component_box']
        x0,y0,x1,y1=_tight(entry['observed'])
        a,b,c,d=bx+x0,by+y0,bx+x1,by+y1
    h,w=legend.shape[:2];cy=entry['center'][1]
    radius=max(3.,.65*reference_diameter)
    top,bottom=max(0,int(cy-radius)),min(h,int(cy+radius+1))
    occupied=(legend[top:bottom].min(axis=2)<180).any(axis=0)
    left=np.flatnonzero(occupied[:a]);right=np.flatnonzero(occupied[c:])
    left_gap=a-int(left[-1])-1 if len(left) else a
    right_gap=int(right[0]) if len(right) else w-c
    min_gap=max(3,int(round(.45*reference_diameter)))
    label=_right_label(legend,entry,reference_diameter)
    isolated=left_gap>=min_gap and right_gap>=min_gap
    interior_count=int(evidence.get('independent_interior_pixels',0))
    interior_fill=float(evidence.get('independent_fill',0.))
    filled=(bool(evidence.get('sufficient_fill_evidence')) and interior_count>=3 and
            interior_fill>=.65 and not evidence.get('strong_hollow_evidence',False) and
            not evidence.get('patterned_internal_evidence',False))
    # Unknown is allowed for a compact, well-supported filled silhouette;
    # uncertainty is retained in shape_hint and the observed template itself.
    unknown_supported=(filled and evidence.get('outer_solidity',0.)>=.84 and
                       evidence.get('convex_occupancy',0.)>=.78 and
                       max(evidence.get('shape_scores',{}).values(),default=0.)>=.60)
    body_supported=(shape!='unknown_marker' or unknown_supported)
    if entry['direction'] is None and not shape.startswith('open_'):
        body_supported &= filled
    reasons=[]
    label_run=_anchored_label_run(legend,entry,anchors)
    if label_run['inside']:reasons.append('inside_existing_key_label_run')
    if not label:reasons.append('no_independent_right_label')
    if not isolated:reasons.append('insufficient_separation_from_text_or_other_ink')
    if not body_supported:reasons.append('insufficient_observed_interior_or_shape_support')
    return dict(accepted=not reasons,reasons=reasons,right_label=bool(label),
        anchored_label_run=label_run,
        left_gap_px=int(left_gap),right_gap_px=int(right_gap),minimum_gap_px=min_gap,
        independent_interior_pixels=interior_count,independent_fill=interior_fill,
        shape_uncertain=shape=='unknown_marker',unknown_supported=bool(unknown_supported),
        policy='isolated_row_key_plus_right_label_and_observed_envelope_interior',
        template_pixels_reconstructed=False)


def _table_context(candidates,legend):
    columns=_coherent_columns(candidates)
    if not columns:
        return candidates,[],columns
    typical=float(np.median([diameter for _,diameter in columns]))
    kept=[];rejected=[]
    for entry in candidates:
        aligned=any(abs(entry['center'][0]-x)<=max(3.,.5*diameter) for x,diameter in columns)
        if aligned or _right_label(legend,entry,typical):
            kept.append(entry)
        else:
            rejected.append(dict(box=list(entry['body_box']),reason=
                'Out-of-column body lacks an independent right-hand legend label or is clipped'))
    return kept,rejected,columns


def _crop_template(legend, entry, origin, slot, rivals):
    lx,ly=origin
    body=entry['body_box'];a,b,c,d=body
    # Body-centred source crop keeps short observed tails visible in the audit
    # without allowing tail length to set the marker centre or diameter.
    pad=max(1,int(math.ceil(.18*entry['diameter'])))
    x0,y0=max(0,a-pad),max(0,b-pad)
    x1,y1=min(legend.shape[1],c+pad),min(legend.shape[0],d+pad)
    raw=legend[y0:y1,x0:x1].copy()
    cx,cy=entry['center']
    model=entry['model']
    raw_soft=_membership(raw,model)
    body_domain=np.zeros(raw.shape[:2],bool);body_domain[b-y0:d-y0,a-x0:c-x0]=True
    observed=_strong_support(raw,model)
    weak_observed=raw_soft>=WEAK_SUPPORT_THRESHOLD
    marker=observed & body_domain
    nuisance=weak_observed & ~body_domain
    # Pad asymmetrically only to restore the measured body's true centre when
    # the supplied legend clips an outer connecting-line tail at an image edge.
    center_x,center_y=cx-x0,cy-y0
    half_w=max(center_x,raw.shape[1]-1-center_x)+2
    half_h=max(center_y,raw.shape[0]-1-center_y)+2
    shape=(int(round(2*half_h+1)),int(round(2*half_w+1)))
    top,left=int(round(half_h-center_y)),int(round(half_w-center_x))
    def pad_array(array):
        result=np.zeros(shape,array.dtype);result[top:top+array.shape[0],left:left+array.shape[1]]=array
        return result
    raw_s=pad_array(raw_soft);raw_m=pad_array(weak_observed)
    mask=pad_array(marker);nuisance=pad_array(nuisance)
    soft=raw_s*pad_array(body_domain)
    if mask.sum()<6:
        raise ValueError('Final colour model has insufficient observed body support')
    inside=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    boundary=mask & (inside<=1.05)
    weight=np.where(mask,np.where(boundary,.40+.35*soft,.82+.18*soft),0.).astype(np.float32)
    valid=np.ones(shape,np.float32);valid[nuisance]=0.
    # Central connector is nuisance for fill classification only, never for
    # the final mask/distribution. Otherwise a line could erase a hollow rim.
    central=np.zeros_like(marker)
    bx,by,ex,ey=entry['component_box']
    full=np.zeros(legend.shape[:2],bool);full[by:ey,bx:ex]=entry['central_line']
    central=pad_array(full[y0:y1,x0:x1])
    name,fill,evidence=_shape(mask,nuisance=central,return_report=True)
    soft_fill=_soft_fill_evidence(mask,soft,central)
    evidence['continuous_fill_evidence']=soft_fill
    filled=bool(name.startswith('filled_') and evidence.get('sufficient_fill_evidence') and
                evidence.get('independent_fill',0.)>=.90 and not evidence.get('strong_hollow_evidence',False) and
                soft_fill['interior_pixels']>=6 and soft_fill['strong_fraction']>=.85 and
                soft_fill['mean_membership']>=.72)
    if name.startswith('filled_') and not filled:
        evidence['pre_uncertainty_shape_hint']=name
        evidence['status']='uncertain_continuous_interior_fill'
        name='unknown_marker'
    uncertain=np.zeros_like(mask)
    if central.any() and not filled:
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Envelope is a measurement domain only, never template ink.
            envelope=np.zeros_like(mask,np.uint8)
            cv2.drawContours(envelope,[cv2.convexHull(max(contours,key=cv2.contourArea))],-1,1,-1)
            interior=cv2.distanceTransform(envelope,cv2.DIST_L2,5)>=max(1.5,.15*entry['diameter'])
            if (interior & ~mask).sum()>=max(2,.12*interior.sum()):
                uncertain=central & interior & mask
                weight[uncertain]*=.08
                valid[uncertain]=.15
                nuisance|=uncertain
    edge=(cv2.Canny(mask.astype(np.uint8)*255,40,100)>0) & ~uncertain
    sid=f'S{slot:02d}'
    absolute=lambda box:[v+(lx if i%2==0 else ly) for i,v in enumerate(box)]
    absolute_crop=absolute((x0,y0,x1,y1));absolute_body=absolute(body)
    template=D.SwatchTemplate(name,tuple(absolute_crop),(cx+lx,cy+ly),float(entry['diameter']),
        'filled' if filled else 'open' if name.startswith('open_') else 'unknown',float(fill),
        raw_s,raw_m,nuisance,valid,soft,mask,edge,D._edge_orientation(soft),model,
        required_weight=weight,matching_profile=VERSION,
        swatch_id=sid,shape_hint=name)
    report=dict(swatch_id=sid,id=sid,shape_hint=name,diameter=template.diameter,
        color_rgb=list(model.core_bgr[::-1]),swatch_box=absolute_crop,glyph_box=absolute_crop,
        marker_body_box=absolute_body,body_dimensions=[c-a,d-b],
        mask=mask.astype(int).tolist(),soft=soft.tolist(),raw_soft=raw_s.tolist(),
        raw_mask=raw_m.astype(int).tolist(),line_nuisance=nuisance.astype(int).tolist(),
        required_weight=weight.tolist(),raw_bgr=raw.tolist(),connected_line=entry['direction'] is not None,
        central_line_uncertainty=uncertain.astype(int).tolist(),
        line_direction=entry['direction'],shape_evidence=evidence,color_tolerance_lab=float(model.lab_tolerance),
        extractor=VERSION, native_palette_rgb=entry.get('native_palette_rgb'),
        native_grid_center=entry.get('native_grid_center'), native_index=entry.get('native_index'),
        prior_used=entry.get('prior_used',False),
        discovery_source=entry.get('discovery_source','observed_colour_body'),
        neutral_row_evidence=entry.get('neutral_row_evidence'),
        source_structure_box=absolute(entry['structure_box']) if entry.get('structure_box') else None,
        soft_threshold=FOREGROUND_THRESHOLD,
        weak_support_threshold=WEAK_SUPPORT_THRESHOLD,
        colour_reference_contrast=float(model.colour_reference_contrast),
        colour_evidence_model='hue_direction_and_observed_source_paper_contrast',
        mask_evidence='Observed hue-compatible pixels >=50% of source-component ink contrast; weak halo retained separately',
        extraction_quality=dict(observed_body_pixels=int(mask.sum()),line_bulge=entry['direction'] is not None,
                                source_center=[float(cx+lx),float(cy+ly)]))
    return template,report


def _source_structure(legend, paper):
    """Independent full-ink keys; neutral bodies need not share line colour.

    Reuse the BW row/label locator, but sample each body's observed colour.
    These anchors constrain text ownership, not marker class or plot points.
    """
    anchors=[]
    area=(0,0,legend.shape[1],legend.shape[0])
    try:boxes=L._automatic_boxes(legend,area)
    except ValueError as error:
        if str(error) not in ('No complete graphical legend entries were found',
                             'No graphical entries with label or column context were found'):raise
        return []
    for source_box in boxes:
        result=None
        for pad in (0,1,2):
            a,b,c,d=source_box
            measured=(max(0,a-pad),max(0,b-pad),min(area[2],c+pad),min(area[3],d+pad))
            try:result=L._glyph_pixels(legend,measured,return_report=True)
            except L.LegendEntryError as error:
                if error.code!='clipped_marker':break
                continue
            break
        if result is None:continue
        gray,line,glyph,connected,report=result
        a,b,c,d=glyph;body=legend[b:d,a:c]
        core=(gray<235)&~line
        if core.sum()<3:continue
        model=_model(body[core],paper)
        support=np.zeros(legend.shape[:2],bool)
        x0,y0,x1,y1=measured
        support[y0:y1,x0:x1]=legend[y0:y1,x0:x1].min(axis=2)<235
        try:entry=_candidate(legend,support,measured,paper,model)
        except ValueError:continue
        if model.achromatic:
            name,_,evidence=_shape(entry['marker'],nuisance=entry['central_line'],return_report=True)
            structure=_neutral_row_evidence(legend,entry,entry['diameter'],name,evidence)
            # A full-source connected key can have a differently coloured
            # line. Its geometric line/body check is independent evidence;
            # a standalone neutral blob still requires the strict row test.
            # Short dashed connectors can be confirmed by the colour/body
            # decomposition even when the BW compact-body policy leaves them
            # attached. Require a measured hollow body as independent evidence
            # before using that route; a horizontal text stroke is insufficient.
            if not structure['accepted']:
                short_hollow_connector=(entry.get('direction')=='horizontal'
                    and bool(evidence.get('strong_hollow_evidence')))
                if not (connected or short_hollow_connector) or not structure['right_label']:continue
                structure=dict(structure,accepted=True,reasons=[],
                    policy='full_source_line_body_and_independent_right_label',
                    original_colour_only_reasons=structure['reasons'])
            entry['neutral_row_evidence']=structure
        entry.update(discovery_source='complete_source_key_row_and_body_colour',
                     structure_box=measured)
        anchors.append(entry)
    # The BW locator can still propose a connected word (e.g. mg/m²).
    # Structural proposals are not exempt from ownership by an earlier key's
    # measured text run. Resolve that ownership before they define columns.
    return [entry for entry in anchors
            if not _anchored_label_run(legend,entry,anchors)['inside']]


def _in_structure_label(center, anchors, shape):
    """Protect label cells, including coloured/multi-line text and neutral keys.

    Real neighbouring key columns and the next key row bound every text cell.
    A candidate intersecting any observed key can never be rejected as text.
    """
    x,y=center;h,w=shape[:2]
    boxes=[a['structure_box'] for a in anchors if a.get('structure_box')]
    if any(a-2<=x<c+2 and b-2<=y<d+2 for a,b,c,d in boxes):return False
    for a,b,c,d in boxes:
        diameter=d-b
        # A key column persists across rows. The same-row key can be faint or
        # missed by this locator while independently visible colour evidence
        # remains; do not absorb that whole column into its neighbour's label.
        right=min((aa for aa,bb,cc,dd in boxes if aa>c),default=w)
        bottom=min((bb for aa,bb,cc,dd in boxes if bb>=d and abs((aa+cc-a-c)/2)<=max(4.,.6*(c-a))),default=h)
        if c+1<=x<right-2 and max(0,b-2)<=y<bottom:
            return True
    return False


def _discover_entries(image,legend_area):
    lx,ly,rx,by=_box(legend_area,image)
    legend=image[ly:by,lx:rx]
    hsv=cv2.cvtColor(legend,cv2.COLOR_BGR2HSV)
    chroma=legend.max(axis=2).astype(int)-legend.min(axis=2).astype(int)
    chromatic=(hsv[...,1]>=30)&(chroma>=18)&(legend.min(axis=2)<242)
    bright=legend.mean(axis=2)>=np.percentile(legend.mean(axis=2),85)
    paper=tuple(int(v) for v in np.median(legend[bright],axis=0))
    structural=_source_structure(legend,paper)
    candidates=[];rejected=[]
    for support,box in _groups(chromatic):
        try:
            candidates.append(_candidate(legend,support,box,paper))
        except (ValueError,cv2.error) as error:
            rejected.append(dict(box=list(box),reason=str(error)))
    candidates,fragments=_remove_fragments(candidates);rejected.extend(fragments)
    candidates,context_rejections,columns=_table_context(candidates,legend)
    rejected.extend(context_rejections)
    kept=[]
    for entry in candidates:
        if _in_structure_label(entry['center'],structural,legend.shape):
            rejected.append(dict(box=list(entry['body_box']),status='source_label_cell_rejected',
                reason='Observed body lies in a full-source key label cell'))
        else:kept.append(entry)
    candidates=kept
    for entry in structural:
        match=next((c for c in candidates if abs(c['center'][0]-entry['center'][0])<.55*entry['diameter']
                    and abs(c['center'][1]-entry['center'][1])<.55*entry['diameter']),None)
        if match is None:candidates.append(entry)
        else:match['structure_box']=entry['structure_box']
    if not candidates:
        return [],rejected+[dict(reason='No trustworthy chromatic marker body; native priors may supply observed neutral keys')]
    anchors=list(candidates)
    # Neutral swatches require an independently observed chromatic column (or
    # matching horizontal row), preventing ordinary black words from becoming
    # standalone neutral marker templates.
    neutral=(chroma<18)&(legend.min(axis=2)<220)
    column_scope=np.zeros(neutral.shape,bool)
    for anchor in anchors:
        x=anchor['center'][0];radius=max(4.,.80*anchor['diameter'])
        column_scope[:,max(0,int(math.floor(x-radius))):min(neutral.shape[1],int(math.ceil(x+radius+1)))]=True
    # Column-limited discovery is especially important for neutral swatches:
    # unlike colour segmentation, grayscale cannot separate an adjacent label.
    neutral_groups=_groups(neutral & column_scope)+_groups(neutral)
    for support,box in neutral_groups:
        try:
            entry=_candidate(legend,support,box,paper)
            x,y=entry['center'];diameter=entry['diameter']
            if _in_structure_label(entry['center'],structural,legend.shape):
                name,_,evidence=_shape(entry['marker'],nuisance=entry['central_line'],return_report=True)
                structure=_neutral_row_evidence(legend,entry,diameter,name,evidence,anchors)
                if 'inside_existing_key_label_run' not in structure['reasons']:
                    structure['reasons'].append('inside_existing_key_label_run')
                structure['accepted']=False
                rejected.append(dict(status='neutral_row_rejected',box=list(entry['body_box']),
                    neutral_row_evidence=structure))
                continue
            column=any(abs(x-a['center'][0])<=max(2.5,.4*a['diameter']) for a in anchors)
            row=any(abs(y-a['center'][1])<=max(2.,.3*a['diameter']) and
                    .7*a['diameter']<=diameter<=1.4*a['diameter'] for a in anchors)
            if not (column or row):
                continue
            if any(abs(x-a['center'][0])<.65*a['diameter'] and abs(y-a['center'][1])<.65*a['diameter'] for a in anchors):
                continue
            name,_,shape_evidence=_shape(entry['marker'],nuisance=entry['central_line'],return_report=True)
            if not column or structural:
                peers=[a['diameter'] for a in anchors if abs(y-a['center'][1])<=max(2.,.3*a['diameter'])]
                reference=float(np.median(peers)) if peers else diameter
                structure=_neutral_row_evidence(legend,entry,reference,name,shape_evidence,anchors)
                if not structure['accepted']:
                    rejected.append(dict(status='neutral_row_rejected',
                        source_body_box=[v+(lx if i%2==0 else ly) for i,v in enumerate(entry['body_box'])],
                        reason='Neutral row key lacks independent label, separation or body evidence',
                        neutral_row_evidence=structure))
                    continue
                entry.update(neutral_row_evidence=structure,
                    discovery_source='observed_neutral_row_key_and_label')
            elif name=='unknown_marker' and entry['marker_pixels']<10:
                continue
            if any(abs(x-a['center'][0])<=.40*min(diameter,a['diameter']) and
                   abs(y-a['center'][1])<=.40*min(diameter,a['diameter']) for a in candidates):
                continue
            candidates.append(entry)
        except (ValueError,cv2.error):
            continue
    candidates,fragments=_remove_fragments(candidates);rejected.extend(fragments)
    return candidates,rejected


def _native_priors(native_grid, native_palette, native_centres):
    """Read only populated v45 cells; table dimensions never create entries."""
    raw=[]
    if native_centres is not None and native_palette is not None:
        if len(native_centres)!=len(native_palette):
            raise ValueError('Native palette and centres must have the same length')
        raw=[(c[0],c[1],rgb) for c,rgb in zip(native_centres,native_palette)]
    elif native_grid:
        cells=native_grid.get('cells',[])
        if isinstance(cells,dict):
            for key,entry in cells.items():
                col,row=(int(v) for v in (key.split(',') if isinstance(key,str) else key))
                if entry.get('rgb') is not None:
                    raw.append((native_grid['cols'][col],native_grid['rows'][row],entry['rgb']))
        else:
            raw=[(c[0],c[1],c[2]) for c in cells if len(c)>=3]
        raw.sort(key=lambda c:(c[1],c[0]))
    result=[]
    for index,(x,y,rgb) in enumerate(raw):
        values=np.asarray([x,y,*rgb],float)
        if len(rgb)!=3 or not np.isfinite(values).all() or min(rgb)<0 or max(rgb)>255:
            raise ValueError('Native centres/RGB palette must be finite and RGB in 0..255')
        result.append(dict(x=float(x),y=float(y),rgb=[int(v) for v in rgb],index=index))
    return result


def _prior_entry(legend, origin, prior, priors, paper, anchors=()):
    lx,ly=origin
    cx,cy=prior['x']-lx,prior['y']-ly
    h,w=legend.shape[:2]
    if not (0<=cx<w and 0<=cy<h):
        raise ValueError('Native centre is outside the supplied legend region')
    if _in_structure_label((cx,cy),anchors,legend.shape):
        prior['_neutral_rejection']=dict(reasons=['inside_existing_key_label_run'],
            policy='source_structure_label_cell')
        return None,'neutral_prior_rejected'
    gaps=[abs(prior['y']-p['y']) for p in priors if abs(prior['y']-p['y'])>4]
    ry=max(6.,min(24.,.46*min(gaps))) if gaps else 17.
    rx=max(12.,min(30.,1.5*ry))
    x0,y0=max(0,int(cx-rx)),max(0,int(cy-ry))
    x1,y1=min(w,int(cx+rx+1)),min(h,int(cy+ry+1))
    patch=legend[y0:y1,x0:x1]
    model=_model(patch.reshape(-1,3),paper,prior_rgb=prior['rgb'])
    membership=_membership(patch,model)
    support=_strong_support(patch,model)
    # Neutral intensity cannot distinguish grey swatches from darker label
    # letters; a neutral prior only owns pixels close to its own intensity ray.
    if model.achromatic:
        support &= np.ptp(patch.astype(int),axis=2)<24
    entries=[];line_only=False;line_support=None
    for group,box in _groups(support):
        full=np.zeros((h,w),bool);full[y0:y1,x0:x1]=group
        a,b,c,d=box
        absolute=(a+x0,b+y0,c+x0,d+y0)
        try:
            entry=_candidate(legend,full,absolute,paper,model)
        except ValueError:
            if c-a>=3*max(1,d-b) and d-b<=5:
                line_only=True
                line_support=group
            continue
        x,y=entry['center'];diameter=entry['diameter']
        if abs(y-cy)>ry or abs(x-cx)>max(9.,.65*diameter):
            continue
        # A crop clipped by the search box cannot establish a complete body.
        a,b,c,d=entry['body_box']
        if (a<=x0 and x0>0) or (c>=x1 and x1<w) or (b<=y0 and y0>0) or (d>=y1 and y1<h):
            continue
        if model.achromatic:
            name, _, shape_evidence = _shape(entry['marker'], nuisance=entry['central_line'], return_report=True)
            structure = _neutral_row_evidence(legend, entry, diameter, name, shape_evidence, anchors)
            if not structure['accepted']:
                prior['_neutral_rejection'] = structure
                continue
            entry['neutral_row_evidence'] = structure
        score=(abs(x-cx)/max(diameter,5.)+abs(y-cy)/max(ry,5.))
        entries.append((score,entry))
    if not entries:
        # Thin lines have height<3 and are intentionally excluded by _groups.
        ys,xs=np.nonzero(support)
        if len(xs)>=6:
            spanx,spany=np.ptp(xs)+1,np.ptp(ys)+1
            thin=spanx>=4*spany and spany<=4
            line_only |= thin
            if thin:
                line_support=support
        if line_only and line_support is not None:
            a,b,c,d=_tight(line_support)
            prior['_line_evidence']=dict(observed_line_box=[a+x0+lx,b+y0+ly,c+x0+lx,d+y0+ly],
                observed_line_mask=line_support[b:d,a:c].astype(int).tolist(),
                observed_line_raw_bgr=patch[b:d,a:c].tolist())
        if prior.get('_neutral_rejection'):
            # A rejected letter must not be reclassified as a line-only key.
            prior.pop('_line_evidence', None)
            return None, 'neutral_prior_rejected'
        return None,'line_only' if line_only else 'no_observed_marker'
    entry=min(entries,key=lambda pair:pair[0])[1]
    entry.update(native_palette_rgb=prior['rgb'],native_grid_center=[prior['x'],prior['y']],
                 native_index=prior['index'],prior_used=True,discovery_source='v45_initial_grid_colour_prior')
    return entry,None


def extract_hybrid_legend(image_bgr, legend_area, native_grid=None,
                          native_palette=None, native_centres=None):
    """Return (SwatchTemplates, JSON-safe reports, explicit rejections).

    Input and report rectangles use exclusive right/bottom source coordinates.
    Native centres are absolute (x,y), native palette entries are RGB in the
    same order. Alternatively supply the original populated v45 grid dict.
    The image and all supplied native data are read-only. Original crops and
    continuous pixel evidence are retained; no ideal-symbol pixels are drawn.
    Line-only keys are reported separately, never turned into marker models.
    """
    if not isinstance(image_bgr,np.ndarray) or image_bgr.ndim!=3 or image_bgr.shape[2]!=3:
        raise ValueError('Expected a BGR image with three channels')
    if image_bgr.dtype!=np.uint8:
        raise ValueError('Expected an uint8 BGR image')
    lx,ly,rx,by=_box(legend_area,image_bgr)
    legend=image_bgr[ly:by,lx:rx]
    bright=legend.mean(axis=2)>=np.percentile(legend.mean(axis=2),85)
    paper=tuple(int(v) for v in np.median(legend[bright],axis=0))
    candidates,rejected=_discover_entries(image_bgr,legend_area)
    anchors=list(candidates)
    priors=_native_priors(native_grid,native_palette,native_centres)
    merged=[];used=set()
    for prior in priors:
        try:
            entry,status=_prior_entry(legend,(lx,ly),prior,priors,paper,anchors)
        except (ValueError,cv2.error) as error:
            entry,status=None,str(error)
        # Native table centres can sit on the connector. Independent body
        # discovery provides the real centre rather than forcing a grid snap.
        matches=[]
        for i,other in enumerate(candidates):
            x,y=other['center'];diameter=other['diameter']
            if i in used or abs(x+lx-prior['x'])>max(10.,diameter) or abs(y+ly-prior['y'])>max(8.,.8*diameter):
                continue
            colour_error=np.linalg.norm(_lab(np.asarray(other['model'].core_bgr,np.uint8).reshape(1,1,3))-
                                        _lab(np.asarray(prior['rgb'][::-1],np.uint8).reshape(1,1,3)))
            source_box=other.get('structure_box')
            same_source_key=bool(source_box and source_box[0]<=prior['x']-lx<source_box[2]
                                 and source_box[1]<=prior['y']-ly<source_box[3])
            if colour_error<55 or same_source_key:
                matches.append((float(colour_error)+(x+lx-prior['x'])**2+(y+ly-prior['y'])**2,i,other))
        if matches:
            _,i,other=min(matches,key=lambda item:item[0]);used.add(i)
            if entry is None or entry['diameter']<.80*other['diameter'] or entry['marker_pixels']<.70*other['marker_pixels']:
                # A washed-out/native colour is a prior, not authority to
                # erase observed body pixels. Keep the independent body and
                # record that the native colour could not preserve it.
                entry=dict(other)
                entry.update(native_palette_rgb=prior['rgb'],native_grid_center=[prior['x'],prior['y']],
                             native_index=prior['index'],prior_used=False,
                             discovery_source='observed_body_preserved_native_prior_disagreed')
        if entry is None:
            rejected.append(dict(native_index=prior['index'],native_grid_center=[prior['x'],prior['y']],
                                 native_palette_rgb=prior['rgb'],status=status,
                                 reason='Native populated cell has no complete independently observed marker body',
                                 neutral_row_evidence=prior.get('_neutral_rejection'),
                                 **prior.get('_line_evidence',{})))
        else:
            merged.append(entry)
    for i,entry in enumerate(candidates):
        if i in used:
            continue
        if any(abs(entry['center'][0]-m['center'][0])<=.60*max(entry['diameter'],m['diameter']) and
               abs(entry['center'][1]-m['center'][1])<=.60*max(entry['diameter'],m['diameter']) for m in merged):
            continue
        merged.append(entry)
    # Native colour priors may measure two fragments of the SAME black body
    # on a coloured connector. Deduplicate by the independently observed key
    # rectangle, never by plot-point proximity or shared RGB.
    unique=[]
    structure_boxes={tuple(e['structure_box']) for e in candidates if e.get('structure_box')}
    for entry in merged:
        x,y=entry['center']
        owner=next((b for b in sorted(structure_boxes) if b[0]<=x<b[2] and b[1]<=y<b[3]),None)
        if owner is not None:entry['structure_box']=owner
        rival=next((e for e in unique if owner is not None and e.get('structure_box')==owner),None)
        if rival is None:unique.append(entry)
        else:
            winner,loser=(entry,rival) if entry['marker_pixels']>rival['marker_pixels'] else (rival,entry)
            if winner is entry:unique[next(i for i,e in enumerate(unique) if e is rival)]=entry
            rejected.append(dict(status='same_observed_source_key',box=list(loser['body_box']),
                reason='Multiple colour fragments belong to one measured legend key'))
    merged=unique
    merged.sort(key=lambda entry:(entry.get('native_index',len(priors)+1000),entry['center'][1],entry['center'][0]))
    models=[];reports=[]
    for entry in merged:
        try:
            model,report=_crop_template(legend,entry,(lx,ly),len(models)+1,[])
            report['native_table_shape']=None if not native_grid else [len(native_grid.get('rows',[])),len(native_grid.get('cols',[]))]
            models.append(model);reports.append(report)
        except (ValueError,cv2.error) as error:
            rejected.append(dict(native_index=entry.get('native_index'),native_grid_center=entry.get('native_grid_center'),
                                 native_palette_rgb=entry.get('native_palette_rgb'),status='invalid_body_template',reason=str(error)))
    return models,reports,rejected

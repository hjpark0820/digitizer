"""One image-estimated scale per legend identity, before any grid detections.

Port of the reviewed Iscalimab shared-symbol experiment. Matching uses gray
ink and paper (including hollow interiors); independent x sites vote equally.
Weak consensus is reported, not upgraded to high-confidence evidence. If no
usable anchor exists, retain 1x rather than silently reopening per-marker fitting.
"""
from __future__ import annotations

import cv2
import numpy as np
from partial_swatch_detector import ink_membership

SEARCH_SCALES = tuple(round(.70 + .025 * i, 3) for i in range(25))


def gray_match_map(observed, template, scale, allowed):
    h, w = template.raw_soft.shape
    ww, hh = max(3, round(w*scale)), max(3, round(h*scale))
    result = np.full(observed.shape, -1, np.float32)
    if hh+6 > observed.shape[0] or ww+6 > observed.shape[1]:
        return result
    model = cv2.resize(template.raw_soft, (ww,hh), interpolation=cv2.INTER_LINEAR)
    nuisance = cv2.resize(template.line_nuisance.astype(np.uint8), (ww,hh),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    model = np.pad(model, 3)
    weight = np.ones(model.shape, np.float32)
    weight[np.flatnonzero(nuisance.any(axis=1))+3] = .15
    model = cv2.GaussianBlur(model, (3,3), .55).astype(np.float32)
    total = float(weight.sum())
    mean = float((weight*model).sum()/total)
    centered = weight*(model-mean)
    variance = float((weight*(model-mean)**2).sum())
    if variance < 1e-6:
        return result
    a = cv2.matchTemplate(observed, weight, cv2.TM_CCORR)
    b = cv2.matchTemplate(observed**2, weight, cv2.TM_CCORR)
    numerator = cv2.matchTemplate(observed, centered, cv2.TM_CCORR)
    denominator = np.sqrt(np.maximum(b-a*a/total, 1e-6)*variance)
    score = np.clip(numerator/denominator, -1, 1)
    valid = cv2.matchTemplate(allowed.astype(np.float32), np.ones(model.shape,np.float32), cv2.TM_CCORR)
    score[valid < model.size-.1] = -1
    ry, rx = model.shape[0]//2, model.shape[1]//2
    result[ry:ry+score.shape[0],rx:rx+score.shape[1]] = score
    return result


def calibrate_swatch_scales(image, templates, plot, ignore_regions=(), log_fn=lambda *a:None,
                            anchor_filter=None, search_scales=None):
    """Return one scale per swatch; default [0.7,1.3], explicit range supported.

    Positions in the report explain calibration only and are NOT detections.
    Retain at most one scale stack at a time. Large crops use a second
    streaming pass instead. Custom profiles carry their own scale coordinates
    into the anchor filter; defaults retain the legacy report contract.
    """
    scales=SEARCH_SCALES if search_scales is None else tuple(float(s) for s in search_scales)
    if not scales or any(not np.isfinite(s) or s<=0 for s in scales):
        raise ValueError('Calibration scales must be positive and finite')
    x0,y0,x1,y1 = plot
    crop = image[y0:y1,x0:x1]
    allowed = np.ones(crop.shape[:2],bool)
    for a,b,c,d in ignore_regions:
        left,right = max(0,a-x0),min(x1,c)-x0
        top,bottom = max(0,b-y0),min(y1,d)-y0
        if left<right and top<bottom:
            allowed[top:bottom,left:right] = False
    best, candidates_by_key = {}, {}
    for t in templates:
        if t.matching_profile != 'bw_v46_uncertain' or not t.ink.achromatic:
            raise ValueError('Shared symbol calibration requires native uncertain B&W templates')
        observed = cv2.GaussianBlur(ink_membership(crop,t.ink).astype(np.float32),(3,3),.55)
        stack = None
        if observed.size*len(scales) <= 16_000_000:
            stack = np.stack([gray_match_map(observed,t,s,allowed) for s in scales])
            best[t.key] = stack.max(axis=0)
        else:
            best[t.key] = np.full(observed.shape,-1,np.float32)
            for scale in scales:
                np.maximum(best[t.key],gray_match_map(observed,t,scale,allowed),out=best[t.key])
        local = cv2.dilate(best[t.key],np.ones((9,9),np.uint8))
        yy,xx = np.nonzero((best[t.key]>=.45)&(best[t.key]>=local-1e-6))
        peaks = sorted(zip(xx,yy),key=lambda xy:-float(best[t.key][xy[1],xy[0]]))
        unique = []
        for x,y in peaks:
            if any(np.hypot(x-q[0],y-q[1])<.85*t.diameter for q in unique):
                continue
            unique.append((int(x),int(y)))
        profiles = np.full((len(unique),len(scales)),-1,np.float32)
        for j,scale in enumerate(scales):
            plane = stack[j] if stack is not None else gray_match_map(observed,t,scale,allowed)
            for i,(x,y) in enumerate(unique):
                cut = plane[max(0,y-2):y+3,max(0,x-2):x+3]
                profiles[i,j] = float(cut.max())
        candidates_by_key[t.key] = [dict(x=x+x0,y=y+y0,quality=float(profile.max()),
            profile=profile.tolist(),all_scales_inside_roi=bool(np.all(profile>-.99)))
            for (x,y),profile in zip(unique,profiles)]
        if search_scales is not None:
            for q in candidates_by_key[t.key]:q['profile_scales']=list(scales)
        del stack, observed
    reports = {}
    for t in templates:
        candidates = candidates_by_key[t.key]
        for q in candidates:
            x,y = q['x']-x0,q['y']-y0
            rivals = [float(v[max(0,y-2):y+3,max(0,x-2):x+3].max())
                      for key,v in best.items() if key!=t.key]
            q['identity_margin'] = q['quality']-max(rivals,default=-1.)
        anchors = []
        for q in sorted(candidates,key=lambda q:-q['quality']):
            if not q['all_scales_inside_roi'] or q['quality']<.55 or q['identity_margin']<.015:
                continue
            if anchor_filter is not None:
                q['independent_anchor_evidence']=anchor_filter(t,q)
                if not q['independent_anchor_evidence']['accepted']:
                    continue
            if any(abs(q['x']-a['x'])<.60*t.diameter for a in anchors):
                continue
            anchors.append(q)
            if len(anchors)==5:
                break
        if anchors:
            curves = np.array([a['profile'] for a in anchors])
            loss = (curves.max(axis=1,keepdims=True)-curves).mean(axis=0)
            index = int(np.argmin(loss));scale = scales[index]
            near = [s for s,v in zip(scales,loss) if v<=loss[index]+.015]
            status = 'multi_site_consensus' if len(anchors)>=3 else 'weak_consensus'
        else:
            loss = np.zeros(len(scales));scale=1.;near=list(scales)
            status = 'no_reliable_anchor_fallback_1x'
        reports[t.key] = dict(scale=scale,status=status,shape=t.shape_hint,
            anchors=anchors,candidates=candidates,scales=list(scales),
            mean_regret=loss.tolist(),near_optimal_scales=near,
            reason='image-only shared symbol scale; one working geometry even with weak evidence',
            criterion='mean per-site weighted grayscale-correlation regret; independent x anchors',
            manual_coordinates_used=False,saved_detections_used=False)
        log_fn(f'[v46 shared scale] {t.key} {t.shape_hint}: scale={scale:.3f}, '
               f'anchors={len(anchors)}, status={status}; all marker windows locked')
    return reports

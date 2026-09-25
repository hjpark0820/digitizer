"""Observed grayscale swatch helpers for v46 centered-X recovery.

Dark seeds locate extent; scoring retains measured darkness and a separate
legend-connector reliability. No idealized glyph or plot reference is used.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.ndimage import minimum_filter

VERSION = "observed-raster-v46-v1"

@dataclass(frozen=True)
class Config:
    scales: tuple = (.75, .875, 1., 1.125, 1.25)
    gains: tuple = (.85, 1., 1.15)
    phases: tuple = (0., .5)
    maximum_loss: float = .48
    minimum_identity_margin: float = .035
    minimum_visible_mass: float = 1.5
    connector_weight_floor: float = .06
    top_anchors: int = 6


def extract(gray, cfg=Config(), *, connected_line=None):
    """Keep every crop gray value; estimate reliability separately, not ink.

    Dark pixels identify the connector row and body extent only. They do not
    replace the grayscale scoring target with a binary silhouette. A source
    connector is not subtracted (subtraction would fabricate white holes).
    """
    gray = np.asarray(gray, np.float32)
    if gray.ndim != 2 or min(gray.shape) < 4 or not np.isfinite(gray).all():
        raise ValueError('Expected a finite, at least 4x4 gray swatch')
    h, w = gray.shape
    paper = float(np.percentile(gray, 95))
    dark = gray[gray < paper-8]
    if len(dark) < 3:
        return dict(status='no_independent_body'), None
    contrast = max(24., paper-float(np.percentile(dark, 10)))
    observed = np.clip((paper-gray)/255., 0, 1)
    relative = np.clip((paper-gray)/contrast, 0, 1)
    side = max(1, int(.20*w))
    flank = np.r_[np.arange(side), np.arange(w-side, w)]
    profile = np.quantile(relative[:, flank], .75, axis=1)
    # Upstream "not separated" can mean a compact STRONG component even
    # though the raw crop retains a pale/dashed connector. Override that hint
    # only with observed tails beyond the off-band body on both sides, and a
    # line row inside (not at the top/bottom of) that body. A tight hollow rim
    # consequently cannot authorize its own removal as a connector.
    peak = int(profile.argmax())
    band = profile > .10*max(float(profile.max()), 1.e-6)
    lo = hi = peak
    while lo > 0 and band[lo-1]: lo -= 1
    while hi < h-1 and band[hi+1]: hi += 1
    protruding = False
    if connected_line is False and float(profile.max()) > .12:
        off = relative.copy(); off[lo:hi+1] = 0
        sums = off.sum(axis=0)
        body_cols = np.flatnonzero(sums > max(.25*float(sums.max()), .20))
        if len(body_cols) >= 2:
            left, right = int(body_cols[0]), int(body_cols[-1])
            rows = np.flatnonzero(off[:, left:right+1].max(axis=1) > .20)
            tail = relative[lo:hi+1].max(axis=0) > .12
            protruding = bool(len(rows) >= 2 and rows[0] < lo and rows[-1] > hi
                and np.count_nonzero(tail[:max(0,left-1)]) >= 2
                and np.count_nonzero(tail[right+2:]) >= 2)
    has_line = float(profile.max()) > .12 and (connected_line is not False or protruding)
    if has_line:
        # Source row intensities, not a rendered line. Only rows connected to
        # the flank peak are nuisance; do not erase other dark horizontal edges.
        nuisance = np.zeros(h, np.float32)
        nuisance[lo:hi+1] = np.clip(profile[lo:hi+1]/max(profile.max(), 1.e-6), 0, 1)
        # Main line and shoulders remain observations with low confidence.
        reliability = 1-(1-cfg.connector_weight_floor)*np.sqrt(nuisance)
    else:
        peak = h//2
        nuisance = np.zeros(h, np.float32)
        reliability = np.ones(h, np.float32)
    independent = relative*reliability[:, None]
    if has_line or connected_line is None:
        independent[:, :side] = 0
        independent[:, w-side:] = 0
    col = independent.sum(axis=0)
    cols = np.flatnonzero(col > max(.25*float(col.max()), .20))
    seed = independent > .20
    if len(cols) < 2 or independent.sum() < cfg.minimum_visible_mass:
        return dict(status='no_independent_body', paper=paper), None
    # Quantiles bound the strongest central evidence without naming its shape.
    x0, x1 = int(cols[0]), int(cols[-1])
    rows = np.flatnonzero(independent[:, x0:x1+1].max(axis=1) > .20)
    if len(rows) < 2:
        return dict(status='line_only_or_unresolved_body', paper=paper), None
    y0, y1 = int(rows[0]), int(rows[-1])
    if y1-y0 < 2:
        return dict(status='line_only_or_unresolved_body', paper=paper), None
    cx, cy = (x0+x1)/2, (y0+y1)/2
    yy, xx = np.indices(gray.shape)
    # One-pixel observed exterior checks containment errors too. No assumed
    # round/square outline, no synthesized rim, no hole filling.
    dx = np.maximum.reduce([x0-xx, xx-x1, np.zeros_like(xx)])
    dy = np.maximum.reduce([y0-yy, yy-y1, np.zeros_like(yy)])
    spatial = np.clip(1-np.maximum(dx, dy)/2., 0, 1)
    weight = spatial*reliability[:, None]
    # A marker's own pale pixels retain their measured amplitude. The outside
    # paper is not an unlimited negative region and cannot swamp the marker.
    bounds = (max(0, x0-2), max(0, y0-2), min(w, x1+3), min(h, y1+3))
    a,b,c,d = bounds
    fields = dict(raw=gray.copy(), observed=observed, relative=relative,
                  line_uncertainty=np.broadcast_to(nuisance[:,None], gray.shape).copy(),
                  weight=weight.astype(np.float32), target=observed[b:d,a:c].copy(),
                  target_weight=weight[b:d,a:c].astype(np.float32),
                  source_gray=gray[b:d,a:c].copy())
    rec = dict(status='observed_template', version=VERSION, paper=paper,
               contrast_for_extent_only=contrast, bbox=[x0,y0,x1+1,y1+1], crop=list(bounds),
               center=[cx,cy], crop_center=[cx-a,cy-b],
               diameter=float(max(x1-x0+1,y1-y0+1)),
               independent_mass=float(independent.sum()), connector_detected=has_line,
               upstream_connected_line=connected_line,
               independent_connector_tails=protruding,
               visible_ink_energy=float(np.sum(weight*observed**2)),
               class_name=None, completed_pixels=0, source_raster_exact=True,
               note='Pixel darkness is not a calibrated ink probability. No shape label used.')
    return rec, fields


def raster_variant(target, weight, center, scale, phase):
    """Resample weights and weighted observations separately across uncertainty."""
    h,w = target.shape
    radius = int(np.ceil(max(w,h)*scale/2))+2
    size = 2*radius+1
    matrix = np.array([[scale,0,radius-center[0]*scale+phase[0]],
                       [0,scale,radius-center[1]*scale+phase[1]]], np.float32)
    ww = cv2.warpAffine(weight, matrix, (size,size), flags=cv2.INTER_LINEAR)
    tw = cv2.warpAffine(target*weight, matrix, (size,size), flags=cv2.INTER_LINEAR)
    tt = np.divide(tw, ww, out=np.zeros_like(tw), where=ww>1.e-6)
    return tt, ww, radius


def score_map(ink, target, weight, gains):
    """Symmetric weighted squared error / expected ink energy; lower is better.

    Missing expected ink AND extra ink on known paper are penalized. Both
    receive low weight only in the source connector-ambiguous band. Blank
    input gives loss 1, not a perfect match. Brightness gains are bounded.
    """
    ink = np.ascontiguousarray(ink, np.float32)
    target = np.ascontiguousarray(target, np.float32)
    weight = np.ascontiguousarray(weight, np.float32)
    energy = float(np.sum(weight*target**2))
    if energy < 1.e-7: raise ValueError('No visible template energy')
    z2 = cv2.matchTemplate(ink**2, weight, cv2.TM_CCORR)
    zt = cv2.matchTemplate(ink, target*weight, cv2.TM_CCORR)
    layers = [(z2-2*g*zt+g*g*energy)/(g*g*energy) for g in gains]
    return np.maximum(np.min(layers,axis=0), 0)


def evaluate_scale(ink, rec, fields, scale, cfg):
    h,w = ink.shape
    best = np.full((h,w), np.inf, np.float32)
    px = np.zeros((h,w), np.float32); py = px.copy()
    for dx in cfg.phases:
        for dy in cfg.phases:
            target, weight, radius = raster_variant(fields['target'],fields['target_weight'],
                                                   rec['crop_center'], scale, (dx,dy))
            if min(h,w) < len(target): continue
            pad = cv2.copyMakeBorder(ink,radius,radius,radius,radius,cv2.BORDER_CONSTANT,value=0)
            score = score_map(pad,target,weight,cfg.gains)
            better = score < best
            best[better] = score[better]; px[better] = dx; py[better] = dy
    return best,px,py


def local_candidates(score, px, py, diameter, permitted, threshold, cap=250):
    win = max(3, int(round(.65*diameter))|1)
    local = score <= minimum_filter(score,size=win,mode='constant',cval=np.inf)+1.e-7
    ys,xs = np.nonzero(local & permitted & (score < threshold))
    order = sorted(zip(xs,ys),key=lambda a:float(score[a[1],a[0]]))
    out=[]
    for x,y in order:
        pt = dict(x=float(x+px[y,x]),y=float(y+py[y,x]),loss=float(score[y,x]),ix=int(x),iy=int(y))
        if all(np.hypot(pt['x']-o['x'],pt['y']-o['y']) > .55*diameter for o in out):
            out.append(pt)
        if len(out)>=cap: break
    return out

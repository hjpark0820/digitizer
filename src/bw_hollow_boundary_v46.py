"""Source-only hollow-square evidence shared by calibration and verification.

An ink-compatible outline is not proof of an empty interior. Require paired
paper/rim/paper transitions at fixed, distributed normals, and measure the
actual hole, not an erosion of the filled envelope. Crossings are nuisance
only when collinear ink exists independently on BOTH external flanks.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

VERSION = 'paired-square-inner-outer-v1'


def eligible(template):
    return (getattr(template, 'model_completed', False)
            and getattr(template, 'name', '') == 'open_square'
            and getattr(template, 'marker_kind', '') == 'open'
            and getattr(template, 'matching_profile', '') == 'bw_v46_uncertain'
            and template.ink.achromatic)


def source_darkness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return np.clip((255. - gray.astype(np.float32)) / 255., 0., 1.)


def _crossing_exemption(observed, envelope, hole, available, diameter):
    """At most one narrow crossing, with independently visible outer flanks.

    A short dash within the hole alone cannot excuse hatch ink. The test uses
    the unmodified source, with no marker/ring pixels as external evidence.
    """
    ink = (observed >= .35) & available
    yy, xx = np.indices(ink.shape)
    ey, ex = np.nonzero(envelope)
    cx, cy = (ex.min()+ex.max())/2., (ey.min()+ey.max())/2.
    dx, dy = xx-cx, yy-cy
    external = (cv2.distanceTransform((~envelope).astype(np.uint8), cv2.DIST_L2, 5) >= 1.5)
    depth = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    half_width = max(1.25, .14*diameter)
    best = np.zeros_like(hole); best_count = 0; evidence = None
    for angle in np.linspace(0, np.pi, 24, endpoint=False):
        along = dx*np.cos(angle)+dy*np.sin(angle)
        across = -dx*np.sin(angle)+dy*np.cos(angle)
        for offset in np.arange(-.35*diameter, .35*diameter+.01, 1.):
            tube = np.abs(across-offset) <= half_width
            removable = tube & hole & ink & (depth <= half_width+.5)
            count = int(removable.sum())
            if count < max(3, best_count+1):
                continue
            flank_counts = []
            for sign in (-1, 1):
                flank = tube & external & ink & (sign*along > .45*diameter) & (sign*along < 1.5*diameter)
                coords = along[flank]
                flank_counts.append(int(flank.sum()) if coords.size and np.ptp(coords) >= max(1., .12*diameter) else 0)
            if min(flank_counts) < max(3, int(.15*diameter)):
                continue
            best, best_count = removable, count
            evidence = dict(angle_degrees=float(np.degrees(angle)), offset_px=float(offset),
                            half_width_px=float(half_width), external_flank_ink=flank_counts)
    return best, evidence


def square_interior_evidence(observed, mask, available=None, occlusion=None):
    """Fixed-geometry profiles; no independent per-normal position search.

    Unknown profiles count as zero support, never positive white space. Shape
    and required-ink gates remain separate and cannot be promoted by this test.
    """
    observed = np.asarray(observed, np.float32)
    mask = np.asarray(mask, bool)
    if observed.ndim != 2 or observed.shape != mask.shape:
        raise ValueError('Hollow evidence requires aligned 2D source and model')
    valid = np.ones_like(mask) if available is None else np.asarray(available, bool).copy()
    if valid.shape != mask.shape:
        raise ValueError('Hollow availability must align with source')
    if occlusion is not None:
        if np.shape(occlusion) != mask.shape:
            raise ValueError('Hollow occlusion must align with source')
        valid &= np.asarray(occlusion) < .5
    envelope = binary_fill_holes(mask)
    hole = envelope & ~mask
    result = dict(version=VERSION, decision='abstain', reason='unresolved_inner_boundary',
                  status='unobserved', hole_pixels=int(hole.sum()), independent_visible_pixels=0,
                  visible_fraction=0., own_ink_fraction=None, paired_support=0., side_support=[0.]*4,
                  unknown_is_positive=False, source_pixels_modified=False)
    yy, xx = np.nonzero(envelope); hy, hx = np.nonzero(hole)
    if len(hx) < 9 or len(xx) < 12:
        return result
    cx, cy = (xx.min()+xx.max())/2., (yy.min()+yy.max())/2.
    rx, ry = (xx.max()-xx.min()+1)/2., (yy.max()-yy.min()+1)/2.
    ix, iy = (hx.max()-hx.min()+1)/2., (hy.max()-hy.min()+1)/2.
    if min(ix, iy) < 2.:
        return result
    contrasts, supported, observable = [], [], []
    def sample(array, x, y):
        return cv2.remap(np.asarray(array, np.float32), np.float32(x)[None,:],
                         np.float32(y)[None,:], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]
    for side in range(4):
        nx, ny = [(0,1),(-1,0),(0,-1),(1,0)][side]
        radius, inner = [(ry,iy),(rx,ix),(ry,iy),(rx,ix)][side]
        thickness = radius-inner
        if thickness < 1.:
            return result
        extent = [ix,iy,ix,iy][side]
        cs, ss, vs = [], [], []
        # Bands use a subpixel margin on both boundaries; never ring pixels as hole.
        gap = min(1.5, .30*inner)
        bands = [np.linspace(-3.,-1.5,7),
                 np.linspace(min(.75,.3*thickness), thickness-min(.75,.3*thickness),7),
                 np.linspace(thickness+gap, thickness+min(3., .65*inner),7)]
        for f in np.linspace(-.65,.65,9):
            tangent = f*extent
            bx, by = [(cx+tangent,cy-ry),(cx+rx,cy+tangent),
                      (cx+tangent,cy+ry),(cx-rx,cy+tangent)][side]
            values, seen = [], True
            for d in bands:
                sx, sy = bx+nx*d, by+ny*d
                values.append(float(np.median(sample(observed,sx,sy))))
                seen &= bool(np.all(sample(valid,sx,sy)>.99))
            outside, rim, inside = values
            c = min(rim-outside, rim-inside)
            cs.append(c); vs.append(seen); ss.append(seen and c>=.25 and rim>=.4)
        contrasts.append(cs); supported.append(ss); observable.append(vs)
    support = np.asarray(supported, bool)
    side_support = support.mean(axis=1)
    paired = (float(support.mean()) >= .55 and int((side_support>=.4).sum())>=3
              and int((side_support>=.65).sum())>=2)
    safe_hole = cv2.erode(hole.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    visible = safe_hole & valid
    raw_mean = float(observed[visible].mean()) if visible.any() else None
    # Do not seek an explanation for negligible background/antialias ink.
    exemption, crossing = (_crossing_exemption(observed,envelope,safe_hole,valid,max(rx,ry)*2)
                           if raw_mean is not None and raw_mean>.10 else (np.zeros_like(hole),None))
    visible &= ~exemption
    count = int(visible.sum()); fraction = count/max(1,int(safe_hole.sum()))
    remaining = float(observed[visible].mean()) if count else None
    result.update(hole_pixels=int(safe_hole.sum()), independent_visible_pixels=count,
        visible_fraction=float(fraction), own_ink_fraction=remaining, raw_hole_mean=raw_mean,
        paired_support=float(support.mean()), side_support=side_support.tolist(),
        paired_boundaries_supported=bool(paired), profile_contrasts=contrasts,
        observable_profiles=int(np.sum(observable)), total_profiles=36,
        crossing=crossing, crossing_exempt_pixels=int(exemption.sum()),
        model_outer_radii=[float(rx),float(ry)],model_inner_radii=[float(ix),float(iy)])
    if count < max(4,.30*safe_hole.sum()):
        return result
    if remaining > .30:
        result.update(decision='conflict', status='filled_interior_conflict',
                      reason='unexplained_ink_in_observed_hole')
    elif paired:
        result.update(decision='compatible', status='hollow_interior_supported',
                      reason='distributed_inner_outer_boundaries')
    return result

"""Class-name evidence from visible outer transitions, not area containment.

The observed raster remains the matching template.  These circle/square/diamond
models are measurement probes only.  Every alternative uses the same centre,
extent, angular samples, availability and nuisance mask.  A polygon's dark
interior is not positive evidence for an inscribed circular boundary.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = 'positive-arc-outline-identity-v1'
KINDS = ('circle', 'square', 'rhombus')


def _field(value, name, shape=None):
    a = np.asarray(value, np.float32)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError(name+' must be a finite 2-D ink field in [0, 1]')
    if shape is not None and a.shape != shape:
        raise ValueError(name+' must match observed shape')
    return a


def _geometry(ink, nuisance=None):
    """Measure the substantial outer body; do not rewrite template pixels.

    Filling the external silhouette makes the measurement identical for open
    and filled markers. A bounded opening removes thin connector protrusions
    only from the extent measurement, never from observed edge samples.
    """
    binary = np.uint8(ink >= .35)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    face = np.zeros(binary.shape, np.uint8)
    cv2.drawContours(face, [contour], -1, 1, cv2.FILLED)
    size = min(7, max(3, int(round(.22*min(w,h))) | 1))
    if nuisance is not None:
        # A separately identified connector can be thicker than seven pixels.
        # Adapt extent measurement to that measured full-band width rather
        # than treating the surviving connector as part of the marker body.
        # This changes neither observed pixels nor which edge probes count.
        crop = nuisance[y:y+h,x:x+w] >= .20
        rows = int(np.count_nonzero(np.all(crop,axis=1)))
        cols = int(np.count_nonzero(np.all(crop,axis=0)))
        band = max(rows,cols)
        if 0 < band <= .35*min(w,h):
            size = max(size,(band+2)|1)
    opened = cv2.morphologyEx(face, cv2.MORPH_OPEN, np.ones((size,size), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0)
    objects, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not objects:
        return None
    body = max(objects, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(body)
    if min(bw,bh) < 7 or np.count_nonzero(opened) < .45*max(1,np.count_nonzero(face)):
        return None
    # An opening shortens a diamond's tips as well as a connector. Restore
    # the shared axial shortening; only the *excess* elongation of one axis is
    # attributable to a thin flank. This avoids fitting an undersized circle
    # to the middle of a diamond. Both-axis large protrusions remain ambiguous.
    common_shortening = min(max(0,w-bw),max(0,h-bh))
    ew, eh = bw+common_shortening, bh+common_shortening
    return dict(center=[bx+(bw-1)/2., by+(bh-1)/2.], radii=[ew/2.,eh/2.],
        extent=[bx,by,bx+bw,by+bh], raw_extent=[x,y,x+w,y+h],
        extent_method='substantial_outer_body_only_no_source_pixel_rewrite', measurement_kernel=size)


def outline_identity(observed, nuisance=None, *, geometry=None, available=None, source_ink_level=None):
    """Return positive circle/polygon evidence or an explicit abstention.

    ``observed`` is bool or native gray ink in [0,1]. ``geometry`` may reuse the
    returned source geometry at a known aligned centre, preventing a candidate
    from fitting a new smaller circle inside a large body. ``nuisance`` must be
    independently established occlusion/connector pixels, never error-selected.
    ``source_ink_level`` fixes the expected ink amplitude from the source, not
    the target's strongest pixel. It must accompany reused geometry for a plot
    probe. Caller-owned arrays are never changed; scores are not probabilities.
    """
    ink = _field(observed, 'observed')
    valid = np.ones_like(ink) if available is None else _field(available,'available',ink.shape)
    unknown = np.zeros_like(ink) if nuisance is None else _field(nuisance,'nuisance',ink.shape)
    if geometry is not None and source_ink_level is None:
        raise ValueError('Reused source geometry requires a fixed source_ink_level')
    if source_ink_level is None:
        positive = ink[ink>0]
        source_ink_level = float(np.quantile(positive,.90)) if positive.size else 0.
    else:
        source_ink_level = float(source_ink_level)
        if not math.isfinite(source_ink_level) or not 0 < source_ink_level <= 1:
            raise ValueError('source_ink_level must be in (0, 1]')
    out = dict(version=VERSION, status='uncertain', preferred_shape=None, circle_supported=False,
        reason='insufficient_native_outline', scores={}, models={}, geometry=None,
        source_pixels_modified=False, uses_area_iou=False, shared_geometry=True,
        source_ink_level=source_ink_level, gray_policy='fixed_source_amplitude_not_target_contrast_stretch',
        score_kind='observed_outer_transition_support_not_probability')
    if source_ink_level < .03:
        out['reason']='source_ink_too_weak_to_measure_outline'
        return out
    # Masks passed by the legend classifier denote complete extracted
    # silhouettes. Padding supplies their background, not additional ink.
    # Explicit candidate availability still bounds all probes to source pixels.
    padding = 4
    obs = np.pad(np.clip(ink/source_ink_level,0,1), padding)
    vis = np.pad(valid, padding, constant_values=1 if available is None else 0)
    excluded = np.pad(unknown, padding)
    if geometry is None:
        measured = _geometry(obs, excluded)
        if measured is None:
            return out
        geom = dict(measured)
        geom['center'] = [v-padding for v in measured['center']]
        for key in ('extent','raw_extent'):
            geom[key] = [v-padding for v in measured[key]]
    else:
        geom = dict(geometry)
        if len(geom.get('center',[])) != 2 or len(geom.get('radii',[])) != 2:
            raise ValueError('Geometry needs centre and two outer radii')
        if not all(math.isfinite(float(v)) for v in geom['center']+geom['radii']):
            raise ValueError('Finite outline geometry required')
    cx, cy = [float(v)+padding for v in geom['center']]
    a, b = map(float, geom['radii'])
    out['geometry'] = geom
    if min(a,b) <= 4.5 or max(a,b)/min(a,b) > 1.8:
        out['reason'] = 'small_or_extreme_aspect_outline'
        return out
    angles = (np.arange(80)+.5)*2*np.pi/80
    ux, uy = np.cos(angles), np.sin(angles)
    quadrants = ((ux >= 0).astype(int)+2*(uy >= 0).astype(int))
    # Arc-specific probes stay away from axes, where a square's straight side
    # may share a tangent and position with a circle.
    off_axis = np.minimum(abs(ux),abs(uy)) >= .30
    smooth = cv2.GaussianBlur(obs,(3,3),.55)
    gx = cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=1/8)
    gy = cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=1/8)

    def sample(field,x,y):
        return cv2.remap(field,np.float32(x)[None,:],np.float32(y)[None,:],
                         cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]

    for kind in KINDS:
        if kind == 'circle':
            radius = 1/np.sqrt((ux/a)**2+(uy/b)**2)
            nx, ny = ux/a**2, uy/b**2
        elif kind == 'square':
            horizontal = a/np.maximum(abs(ux),1.e-6) <= b/np.maximum(abs(uy),1.e-6)
            radius = np.minimum(a/np.maximum(abs(ux),1.e-6),b/np.maximum(abs(uy),1.e-6))
            nx, ny = np.where(horizontal,np.sign(ux),0),np.where(horizontal,0,np.sign(uy))
        else:
            radius = 1/(abs(ux)/a+abs(uy)/b)
            nx, ny = np.sign(ux)/a,np.sign(uy)/b
        length = np.hypot(nx,ny)
        nx, ny = nx/length, ny/length
        x,y = cx+ux*radius,cy+uy*radius
        inside = sample(obs,x-.75*nx,y-.75*ny)
        outside = sample(obs,x+1.0*nx,y+1.0*ny)
        edge_x,edge_y = sample(gx,x-.25*nx,y-.25*ny),sample(gy,x-.25*nx,y-.25*ny)
        direction = np.clip(-(edge_x*nx+edge_y*ny)/np.maximum(np.hypot(edge_x,edge_y),.05),0,1)
        contrast = np.clip(inside-outside,0,1)
        visible = np.ones(len(angles),bool)
        for offset in (-.75,0.,1.):
            visible &= sample(vis,x+offset*nx,y+offset*ny) > .99
            visible &= sample(excluded,x+offset*nx,y+offset*ny) < .20
        support = contrast*direction
        positive = visible & (contrast >= .35) & (direction >= .85)
        count = int(visible.sum())
        quality = float(support[visible].mean()) if count else 0.
        fraction = float(positive.sum()/max(count,1))
        arc_visible = visible & off_axis
        arc_positive = positive & off_axis
        arc_fraction = float(arc_positive.sum()/max(1,arc_visible.sum()))
        sector_support = [int(np.count_nonzero(positive & (quadrants==i))) for i in range(4)]
        arc_quadrants = sum(np.count_nonzero(arc_positive & (quadrants==i)) >= 3 for i in range(4))
        curvature_rows = []
        if kind == 'circle':
            # Positive curvature requires an observed normal that *turns*
            # through three consecutive visible parts of an arc. Constant
            # normals on a diamond side do not become circle evidence merely
            # because they are moderately close to several radial directions.
            phase = np.degrees(angles) % 90
            arc_index = (np.degrees(angles)//90).astype(int)
            for quadrant in range(4):
                means = []
                counts = []
                for low,high in ((18,35),(35,55),(55,72)):
                    take = positive & (arc_index==quadrant) & (phase>=low) & (phase<high)
                    counts.append(int(take.sum()))
                    dx,dy = -edge_x[take],-edge_y[take]
                    norm = np.maximum(np.hypot(dx,dy),1.e-6)
                    means.append(math.atan2(float(np.mean(dy/norm)),float(np.mean(dx/norm))) if take.any() else 0.)
                turns = [math.atan2(math.sin(means[i+1]-means[i]),math.cos(means[i+1]-means[i]))
                         for i in (0,1)]
                # Two independently positive turns are required; a single
                # large corner turn cannot compensate for a straight flank.
                passed = min(counts)>=2 and all(.12<=v<=.80 for v in turns)
                curvature_rows.append(dict(quadrant=quadrant,counts=counts,
                    normal_turn_degrees=[float(math.degrees(v)) for v in turns],supported=bool(passed)))
        straight_rows = []
        if kind != 'circle':
            faces = quadrants if kind=='rhombus' else np.where(abs(nx)>.5,
                np.where(nx>0,0,1),np.where(ny>0,2,3))
            # A polygon face excludes its corner transition. Requiring a
            # constant normal through a *corner* would reject genuine squares
            # with antialiased/rounded corners. The whole perimeter still
            # contributes to score and missing-boundary penalties above.
            face_interior = off_axis if kind=='rhombus' else np.where(abs(nx)>.5,
                np.abs((y-cy)/b)<=.65, np.abs((x-cx)/a)<=.65)
            for face in range(4):
                take = positive & (faces==face) & face_interior
                dx,dy = -edge_x[take],-edge_y[take]
                norms = np.maximum(np.hypot(dx,dy),1.e-6)
                concentration = float(np.hypot(np.mean(dx/norms),np.mean(dy/norms))) if take.any() else 0.
                passed = take.sum()>=4 and concentration>=.985
                straight_rows.append(dict(face=face,positive_samples=int(take.sum()),
                    normal_concentration=concentration,supported=bool(passed)))
        row = dict(score=quality, visible_samples=count,total_samples=len(angles),
            visible_fraction=float(count/len(angles)),positive_samples=int(positive.sum()),
            positive_fraction=fraction, supported_quadrants=sum(v>=4 for v in sector_support),
            arc_visible_samples=int(arc_visible.sum()),arc_positive_samples=int(arc_positive.sum()),
            arc_positive_fraction=arc_fraction,arc_supported_quadrants=int(arc_quadrants),
            mean_inside=float(inside[visible].mean()) if count else 0.,
            mean_outside=float(outside[visible].mean()) if count else 0.,
            curved_arc_quadrants=sum(r['supported'] for r in curvature_rows),curvature=curvature_rows,
            straight_faces=sum(r['supported'] for r in straight_rows),straight=straight_rows,
            positive_arc_is_observed_transition=True)
        out['models'][kind] = row
        out['scores'][kind] = quality
    ranked = sorted(KINDS,key=lambda k:out['scores'][k],reverse=True)
    best, rival = ranked[:2]
    row = out['models'][best]
    margin = out['scores'][best]-out['scores'][rival]
    evidence = (row['visible_fraction'] >= .5 and row['positive_fraction'] >= .55
                and row['supported_quadrants'] >= 2 and row['score'] >= .45 and margin >= .07)
    arc = (row['arc_positive_fraction'] >= .55 and row['arc_supported_quadrants'] >= 2
           and row['curved_arc_quadrants'] >= 2)
    polygon = row['straight_faces'] >= 2
    if evidence and (arc if best=='circle' else polygon):
        out.update(status='supported',preferred_shape=best,circle_supported=best=='circle',
                   reason='positive_visible_arc' if best=='circle' else 'positive_straight_polygon_outline')
    else:
        out['reason'] = ('no_positive_visible_arc' if best=='circle' and not arc else
                         'outline_alternatives_not_resolved')
    out.update(best_shape=best,runner_up=rival,margin=float(margin),minimum_margin=.07)
    return out

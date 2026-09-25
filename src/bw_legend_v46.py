"""Observed-pixel B&W legend models with complete connected swatch support.

The supplied legend region still comes from the GUI/user.  Individual graphical
entries are found across that region, including multiple columns.  There is no
fixed horizontal scan cap and no reconstructed convex-hull ink.  Only matching
weights, never the source pixels, express uncertain boundaries/connecting lines.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import partial_swatch_detector as D


class LegendEntryError(ValueError):
    """Expected source-glyph failure, not an invalid request/programming error."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class LegendExtractionError(ValueError):
    """No usable model remains; retain per-entry diagnostics for callers."""
    def __init__(self, reports):
        self.reports = reports
        failures = [f"{r['swatch_id']}: {r['extraction_error']}" for r in reports
                    if r.get('extraction_status') == 'failed']
        super().__init__('No complete legend marker remains after extraction/class filter'
                         + (': ' + '; '.join(failures) if failures else ''))


def _box(box, image):
    if box is None or len(box) != 4:
        raise ValueError('A legend region or explicit swatch boxes are required')
    x0,y0,x1,y1=map(int,box)
    h,w=image.shape[:2]
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):
        raise ValueError(f'Invalid legend/swatch box: {box}')
    return x0,y0,x1,y1


def _components(gray, threshold=235):
    return _support_components(gray<threshold)


def _support_components(support):
    n, labels, stats, _=cv2.connectedComponentsWithStats(support.astype(np.uint8),8)
    return labels,[{'id':i,'box':tuple(map(int,(s[0],s[1],s[0]+s[2],s[1]+s[3]))),
                   'area':int(s[4]),'height':int(s[3])}
                  for i,s in enumerate(stats[1:],1) if s[4]>=5 and s[3]>=4]


def _frame_ids(labels, components):
    """Ignore enclosing frames for *discovery*, never erase source pixels."""
    h,w=labels.shape
    frames=set()
    for comp in components:
        a,b,c,d=comp['box']
        enclosed=[q for q in components if q['id']!=comp['id'] and
                  a<q['box'][0]<q['box'][2]<c and b<q['box'][1]<q['box'][3]<d]
        # A large enclosing component with several independent interior
        # objects is a frame, not an open marker.  A lone open square is valid.
        if len(enclosed)>=2 and ((c-a)>.65*w or (d-b)>.70*h):
            inner=labels[b+2:max(b+2,d-2),a+2:max(a+2,c-2)]==comp['id']
            if inner.size and inner.mean()<.12:
                frames.add(comp['id'])
    return frames


def _legend_components(gray):
    """Separate verified enclosing border strokes, not attached legend keys.

    Only the discovery support changes. Source grayscale stays untouched.
    Frame-interior bounds also prevent crop padding from recruiting the border
    again during model extraction or line/marker composition.
    """
    labels,components=_components(gray)
    frames=_frame_ids(labels,components)
    if not frames:return labels,components,[]
    support=gray<235
    interiors=[]
    for comp in components:
        if comp['id'] not in frames:continue
        a,b,c,d=comp['box'];local=labels[b:d,a:c]==comp['id']
        limit=max(1,int(math.ceil(.12*min(c-a,d-b))))
        def thickness(profile):
            count=0
            for coverage in profile[:limit]:
                if coverage<.70:break
                count+=1
            return count
        cols,rows=local.mean(axis=0),local.mean(axis=1)
        left,right=thickness(cols),thickness(cols[::-1])
        top,bottom=thickness(rows),thickness(rows[::-1])
        if min(left,right,top,bottom)==0:
            # An incomplete/non-rectangular enclosing object is not sufficient
            # evidence to cut a frame away from a connected graphical key.
            continue
        edge=np.zeros_like(local)
        edge[:top]=True;edge[-bottom:]=True
        edge[:,:left]=True;edge[:,-right:]=True
        support[b:d,a:c] &= ~(edge & local)
        interiors.append((a+left,b+top,c-right,d-bottom))
    labels,components=_support_components(support)
    remaining_frames=_frame_ids(labels,components)
    return labels,[q for q in components if q['id'] not in remaining_frames],interiors


def _inside_frame(box, interiors, anchor):
    """Clip padding to a frame only when the entire measured body is inside."""
    a,b,c,d=box;e,f,g,h=anchor
    for left,top,right,bottom in interiors:
        if left<=e<g<=right and top<=f<h<=bottom:
            a,b,c,d=max(a,left),max(b,top),min(c,right),min(d,bottom)
    return a,b,c,d


def _repeated_horizontal_body(support):
    """Return the middle observed body only when three aligned repeats agree.

    Measure the connector BETWEEN bodies; end bodies are not line tails.
    This is source extent selection, never synthesis of marker pixels.
    """
    ys,xs=np.nonzero(support)
    if not len(xs):return None
    x0,x1=int(xs.min()),int(xs.max()+1)
    y0,y1=int(ys.min()),int(ys.max()+1);height=y1-y0
    if height<3 or x1-x0<4*height:return None
    spans=np.array([np.ptp(np.flatnonzero(c))+1 if c.any() else 0 for c in support.T])
    positive=spans[x0:x1][spans[x0:x1]>0]
    thin=float(np.percentile(positive,30))
    if height<max(3.,1.6*thin):return None
    elevated=np.flatnonzero(spans>max(thin+.75,.60*height))
    runs=np.split(elevated,np.flatnonzero(np.diff(elevated)>1)+1)
    runs=[r for r in runs if max(2.,.25*height)<=len(r)<=1.5*height]
    if len(runs)<3:return None
    expanded=[]
    for run in runs:
        a,b=int(run[0]),int(run[-1])+1
        while a>x0 and spans[a-1]>thin+.75:a-=1
        while b<x1 and spans[b]>thin+.75:b+=1
        if b-a>1.5*height:return None
        expanded.append(np.arange(a,b))
    runs=expanded
    centers=np.array([(r[0]+r[-1])/2 for r in runs]);distances=np.diff(centers)
    if np.ptp(distances)>max(1.,.25*float(np.median(distances))):return None
    gaps=[]
    for left,right in zip(runs,runs[1:]):
        gap=np.arange(left[-1]+1,right[0])
        if len(gap)<max(2,height) or not np.all((spans[gap]>0)&(spans[gap]<=thin+.5)):return None
        gaps.extend(gap.tolist())
    rows=support[:,gaps].mean(axis=1)>=.90
    if not rows.any():return None
    ly=np.flatnonzero(rows)
    for run in runs:
        yy=np.flatnonzero(support[:,run].any(axis=1))
        if not len(yy) or yy[0]>=ly[0] or yy[-1]<=ly[-1]:return None
    middle=runs[int(np.argmin(abs(centers-(x0+x1-1)/2)))]
    return (max(x0,int(middle[0])-1),y0,min(x1,int(middle[-1])+2),y1),rows


def _line_analysis(support):
    """Observed column-span evidence for a marker on a horizontal line.

    Distinguish compact glyphs (no separation needed) from invalid wide blobs.
    This is an extent measurement only; its support is never a new template.
    """
    ys,xs=np.nonzero(support)
    if not len(xs):
        return dict(status='invalid', reason='empty_support', extent=None)
    top,bottom=int(ys.min()),int(ys.max())+1
    left,right=int(xs.min()),int(xs.max())+1
    height=bottom-top
    geometry=dict(width=right-left, height=height, aspect=(right-left)/height)
    def outcome(status, reason, extent=None):
        return dict(status=status, reason=reason, extent=extent, **geometry)
    repeated=_repeated_horizontal_body(support)
    if repeated is not None:
        (a,_,c,_),rows=repeated
        return outcome('separated','repeated_bodies_with_intervening_connector',(a,c,rows))
    compact=right-left<=1.7*height
    # Short legend connectors can still make an almost-square union. Do not
    # let that union's aspect ratio hide independently thin flanks on a ring.
    # Failed line evidence on a compact glyph is NOT an invalid swatch.
    def no_line(reason):
        return outcome('not_needed', 'compact_strong_support') if compact else outcome('invalid', reason)
    if right-left<=1.15*height:
        return outcome('not_needed', 'compact_strong_support')
    if support[top:bottom,left:right].sum(axis=1).max()<.55*(right-left):
        return no_line('no_continuous_horizontal_line')
    spans=np.array([np.ptp(np.flatnonzero(c))+1 if c.any() else 0 for c in support.T])
    outer=max(2,int(round(.16*(right-left))))
    tails=np.r_[np.arange(left,min(right,left+outer)),np.arange(max(left,right-outer),right)]
    positive=spans[tails][spans[tails]>0]
    if not len(positive):
        return no_line('no_observed_tails')
    thickness=float(np.percentile(positive,80))
    if height<max(5,1.65*thickness):
        return no_line('line_without_marker_bulge')
    elevated=np.flatnonzero(spans>max(thickness+1,.24*height))
    # A few thick pixels at a scan's line endpoint are not a second marker
    # body. Retain the dominant contiguous bulge for geometry only; observed
    # pixels remain untouched and the composition fit still scores the tails.
    if len(elevated):
        runs=np.split(elevated,np.flatnonzero(np.diff(elevated)>2)+1)
        main=max(runs,key=len)
        if len(main)>=max(4,.45*height) and len(main)>=.85*len(elevated):
            elevated=main
    if not len(elevated) or elevated[-1]-elevated[0]+1>1.7*height:
        return no_line('no_compact_marker_bulge')
    if abs((elevated[0]+elevated[-1]+1)/2-(left+right)/2)>.28*(right-left):
        return no_line('off_centre_bulge')
    # An attached line should have meaningful thin tails on both sides;
    # a connected word normally has tall ink throughout its width.
    if np.mean(spans[left:right]<=max(thickness+1,.24*height))<.20:
        return no_line('insufficient_thin_tails')
    line_rows=support[:,tails].mean(axis=1)>=.22
    if compact:
        flank_min=max(2,int(round(.10*height)))
        if min(elevated[0]-left,right-1-elevated[-1])<flank_min:
            return no_line('short_or_one_sided_flanks')
        if not line_rows.any() or line_rows.sum()>.35*height:
            return no_line('no_thin_shared_line_band')
        # Both tails, not a lone ring edge or an asymmetric glyph, must
        # independently support the same horizontal band.
        for flank in (np.arange(left,left+outer),np.arange(right-outer,right)):
            if support[line_rows][:,flank].mean()<.45:
                return no_line('unconfirmed_bilateral_line_band')
        # The compact extension is deliberately hollow-only. Compact solid
        # keys already have a separate union-of-line-and-body fitter; changing
        # their discovery geometry would bypass that identity competition.
        body=support.copy();body[:, :elevated[0]]=False;body[:, elevated[-1]+1:]=False
        yy,xx=np.nonzero(body & ~line_rows[:,None])
        envelope=np.zeros(support.shape,np.uint8)
        if len(xx)<3:
            return no_line('no_independent_hollow_body')
        cv2.fillConvexPoly(envelope,cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)),1)
        depth=cv2.distanceTransform(np.pad(envelope,1),cv2.DIST_L2,5)[1:-1,1:-1]
        interior=(depth>=max(1.5,.18*height)) & ~line_rows[:,None]
        if interior.sum()<10 or support[interior].mean()>.40:
            return no_line('no_independent_hollow_body')
    return outcome('separated', 'supported_line_and_marker',
                   (max(left,int(elevated[0])-1),min(right,int(elevated[-1])+2),line_rows))


def _line_extent(support):
    """Compatibility measurement for discovery; use _line_analysis to reject."""
    return _line_analysis(support)['extent']


def _automatic_boxes(image, legend_area):
    """Find every graphical entry using rows, label gaps and glyph centres.

    A row need not repeat in another row: horizontal legends are valid.  The
    centre of a line-attached glyph, rather than the line's left endpoint,
    supplies column alignment.  Tiny leading raster specks cannot displace a
    full-sized graphical entry.  Connected text runs are not swatches.
    """
    lb=_box(legend_area,image)
    x0,y0,x1,y1=lb
    gray=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    labels,components,interiors=_legend_components(gray)
    rows=[]
    # Large text/glyph bodies establish rows before small detached fragments.
    for comp in sorted(components,key=lambda c:(-c['height'],c['box'][1],c['box'][0])):
        a,b,c,d=comp['box']
        if c-a>1.7*(d-b) and _line_extent(labels==comp['id']) is None:
            # A broad connected text word still provides row evidence, but a
            # nearly horizontal frame/plot rule should not establish a row.
            if c-a>12*(d-b):
                continue
        compatible=[]
        for row in rows:
            rtop=float(np.median([q['box'][1] for q in row]))
            rbottom=float(np.median([q['box'][3] for q in row]))
            overlap=min(d,rbottom)-max(b,rtop)
            if overlap >= .30*min(d-b,rbottom-rtop):
                compatible.append((overlap,row))
        if compatible:
            max(compatible,key=lambda q:q[0])[1].append(comp)
        else:
            rows.append([comp])
    candidates=[]
    rows.sort(key=lambda row:float(np.median([(q['box'][1]+q['box'][3])/2 for q in row])))
    for row_index,row in enumerate(rows):
        row=sorted(row,key=lambda q:q['box'][0])
        row_h=float(np.percentile([q['height'] for q in row],70))
        bodies=[q for q in row if q['height']>=max(4,.32*row_h)]
        right=None
        last_graphic=-2
        for i,comp in enumerate(bodies):
            a,b,c,d=comp['box']
            line=_line_extent(labels==comp['id'])
            start=right is None or a-right>1.05*row_h
            next_gap=bodies[i+1]['box'][0]-c if i+1<len(bodies) else float('inf')
            compact=.35*(d-b)<=c-a<=1.85*(d-b)
            # Letters within a word have nearby neighbours.  A standalone
            # graphical body has a label gap, or a large diameter relative to
            # its text row.  Wide line+marker components have direct evidence.
            isolated=next_gap>=max(2,.30*row_h) or (d-b)>1.45*row_h
            line_start=line is not None and (right is None or a-right>.30*row_h)
            if line is not None and i>0 and i-1!=last_graphic:
                # A detached, short, horizontally aligned dash belongs to the
                # same graphical key, not to its preceding text label.
                e,f,g,h=bodies[i-1]['box']
                preceding_dash=(h-f<=.45*(d-b) and g-e>=1.15*(h-f) and
                    0<=a-g<=.75*(d-b) and abs((f+h-b-d)/2)<=.25*(d-b))
                line_start=line_start or preceding_dash
            # The first isolated word immediately following a swatch is its
            # label, not another glyph (e.g. the connected text "F7").
            if i!=last_graphic+1 and (line_start or (start and compact and isolated)):
                center=(line[0]+line[1])/2 if line else (a+c)/2
                candidates.append(dict(comp,center=center,row_height=row_h,row_index=row_index,
                                       row_bodies=len(bodies)))
                last_graphic=i
            right=max(c,right or c)
    if not candidates:
        raise ValueError('No complete graphical legend entries were found')
    # Rows with unusually small detached bodies can be recovered only when
    # their centre and scale are supported by already found graphical rows.
    # This does not merge equal shapes: every row/column retains its own ID.
    for row_index,row in enumerate(rows):
        if any(q['id'] in {c['id'] for c in candidates} for q in row):
            continue
        for q in row:
            a,b,c,d=q['box']; center=(a+c)/2
            aligned=[p for p in candidates if abs(center-p['center'])<=.25*max(d-b,p['height'])
                     and .55*p['height']<=d-b<=1.8*p['height']]
            if aligned and .35*(d-b)<=c-a<=1.85*(d-b):
                candidates.append(dict(q,center=center,row_height=d-b,row_index=row_index,
                                       row_bodies=len(row)))
                break
    typical=float(np.median([q['height'] for q in candidates]))
    candidates=[q for q in candidates if q['height']>=max(4,.35*typical) and
                (q['row_bodies']>1 or any(p['id']!=q['id'] and
                 abs(q['center']-p['center'])<.25*max(q['height'],p['height'])
                 for p in candidates))]
    if not candidates:
        raise ValueError('No graphical entries with label or column context were found')
    boxes=[]
    for comp in sorted(candidates,key=lambda q:(q['row_index'],q['box'][0])):
        a,b,c,d=comp['box']
        # Padding is paper context, not permission to import the first label
        # letter (important when a tiny swatch ends one pixel before its text).
        neighbors=[q for q in components if q['id']!=comp['id'] and
                   min(d,q['box'][3])>max(b,q['box'][1])]
        previous=max((q['box'][2] for q in neighbors if q['box'][2]<=a),default=0)
        following=min((q['box'][0] for q in neighbors if q['box'][0]>=c),default=x1-x0)
        a,b,c,d=_inside_frame((max(previous,a-2),max(0,b-2),min(following,c+2),min(y1-y0,d+2)),
                              interiors,comp['box'])
        boxes.append((x0+a,y0+b,x0+c,y0+d))
    return boxes


def composition_box(image, legend_area, swatch_box, diameter):
    """Grow only through nearby thin connector fragments, never label glyphs.

    Returned pixels are original observations. The base box is unchanged;
    neighboring components are used solely to bound a full-key fitting crop.
    """
    lx,ly,rx,by=_box(legend_area,image)
    a,b,c,d=map(int,swatch_box)
    top,bottom=max(ly,b-3),min(by,d+3)
    # Full legend context is needed to recognize an enclosing frame: a thin
    # row slice makes that same frame look like an enormous connected swatch.
    gray=cv2.cvtColor(image[ly:by,lx:rx],cv2.COLOR_BGR2GRAY)
    _,components,interiors=_legend_components(gray)
    all_rows=[dict(q,box=(q['box'][0]+lx,q['box'][1]+ly,q['box'][2]+lx,q['box'][3]+ly)) for q in components]
    row=[q for q in all_rows if min(d,q['box'][3])-max(b,q['box'][1])>=
         .6*min(d-b,q['height'])]
    # Discovery boxes contain two padding pixels, which can already overlap
    # the first letter. Re-anchor on the graphical component before growing.
    members=[q for q in row if max(0,min(c,q['box'][2])-max(a,q['box'][0]))>=
             .6*min(c-a,q['box'][2]-q['box'][0])]
    if members:
        main=min(members,key=lambda q:(abs((q['box'][1]+q['box'][3]-b-d)/2),-q['area']))
        a,c=main['box'][0],main['box'][2]
        mb,md=main['box'][1],main['box'][3]
        # Include paper, not fragments of a neighboring legend row.
        overlap=[q for q in all_rows if min(c,q['box'][2])>max(a,q['box'][0])]
        top=max(top,max((q['box'][3] for q in overlap if q['box'][3]<=mb),default=ly))
        bottom=min(bottom,min((q['box'][1] for q in overlap if q['box'][1]>=md),default=by))
    cy=(b+d-1)/2
    def thin(q):
        e,f,g,h=q['box']
        return h-f<=.48*diameter and g-e>=1.1*(h-f) and abs((f+h-1)/2-cy)<=.28*diameter
    left,right=a,c
    for q in sorted((q for q in row if q['box'][2]<=a),key=lambda q:-q['box'][2]):
        e,f,g,h=q['box']
        if not thin(q) or left-g>.85*diameter or a-e>2*diameter:break
        left=e
    for q in sorted((q for q in row if q['box'][0]>=c),key=lambda q:q['box'][0]):
        e,f,g,h=q['box']
        if not thin(q) or e-right>.85*diameter or h-f>diameter or g-c>2*diameter:break
        right=g
    # Only paper padding, clipped before any neighboring component. This is
    # essential for scans where the label starts 2 px after the connector.
    previous=max((q['box'][2] for q in row if q['box'][2]<=left),default=lx)
    following=min((q['box'][0] for q in row if q['box'][0]>=right),default=rx)
    box=(max(lx,previous,left-2),top,min(rx,following,right+2),bottom)
    global_interiors=[(e+lx,f+ly,g+lx,h+ly) for e,f,g,h in interiors]
    anchor=main['box'] if members else swatch_box
    return list(_inside_frame(box,global_interiors,anchor))


def _glyph_pixels(image, box, *, return_report=False):
    """Whole observed glyph extent; horizontal line crossings stay observable.

    Only outside tails estimate a line nuisance band.  The complete union of
    elevated column spans keeps both sides of an open glyph, and one additional
    ambiguity column is retained on each side.  No internal stroke is erased.
    """
    x0,y0,x1,y1=box
    gray=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    labels,components=_components(gray)
    frames=_frame_ids(labels,components)
    components=[q for q in components if q['id'] not in frames]
    if not components:
        raise LegendEntryError('no_marker_ink', f'No marker-sized ink in swatch {box}')
    comp=max(components,key=lambda c:c['area'])
    support=labels==comp['id']
    # Join only nearby, substantial observed fragments for the extent.  The
    # matching image below remains the original grayscale rectangle, including
    # all holes and pale pixels.  No closing/filling is applied to that image.
    a,b,c,d=comp['box']; diameter=max(c-a,d-b)
    reach=max(2,int(round(.12*min(c-a,d-b))))
    for q in components:
        if q['id']==comp['id'] or q['area']<max(5,.025*comp['area']):
            continue
        e,f,g,h=q['box']
        if max(a-g,e-c,0)<=reach and max(b-h,f-d,0)<=reach:
            support|=labels==q['id']
    # A second brightness level extends antialiased boundaries that were
    # disconnected at 235, but cannot recruit distant JPEG/background noise.
    strong_support=support.copy()
    pale=(gray<248) & cv2.dilate(support.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    support|=pale
    ys,xs=np.nonzero(support)
    top,bottom=int(ys.min()),int(ys.max())+1
    left,right=int(xs.min()),int(xs.max())+1
    height=bottom-top
    line_rows=np.zeros(gray.shape[0],bool)
    # Both the gate and the decomposition use the SAME strong support. Pale
    # antialias pixels still extend the returned source crop, but a single pale
    # column cannot turn "no separation needed" into "invalid swatch".
    analysis=_line_analysis(strong_support)
    line_report={k:v for k,v in analysis.items() if k!='extent'}
    line_report.update(support='strong_gray_lt_235', extended_width=right-left,
                       extended_height=height)
    if analysis['status']=='invalid':
        raise LegendEntryError(analysis['reason'],
            f'Line-only or text-contaminated legend entry at {box}: {analysis["reason"]}')
    connected_line=analysis['status']=='separated'
    if connected_line:
        left,right,line_rows=analysis['extent']
    if right-left>1.85*height or right-left<.35*height:
        raise LegendEntryError('implausible_extent', f'Implausible or clipped marker extent in swatch {box}')
    # A compact glyph must have a complete source border.  Connected thin
    # tails are allowed to leave the box; a tall cut through the glyph is not.
    touches=[]
    border_support=support.copy()
    if analysis['reason']=='repeated_bodies_with_intervening_connector':
        # The representative middle body must be complete. An end repeat
        # leaving the supplied swatch is not a cut through the chosen body.
        border_support[:,:left]=False
        border_support[:,right:]=False
    if connected_line:
        # A confirmed connector may leave the crop. Only BODY ink outside
        # that band establishes a cut marker, not the endpoint of a dash.
        border_support[line_rows,:]=False
    for edge_name,local,adjacent,extent in (
        ('left',border_support[:,0],image[y0:y1,x0-1:x0] if x0 else None,height),
        ('right',border_support[:,-1],image[y0:y1,x1:x1+1] if x1<image.shape[1] else None,height),
        ('top',border_support[0,:],image[y0-1:y0,x0:x1] if y0 else None,right-left),
        ('bottom',border_support[-1,:],image[y1:y1+1,x0:x1] if y1<image.shape[0] else None,right-left)):
        if local.sum()>.45*extent and adjacent is not None and np.any(cv2.cvtColor(adjacent,cv2.COLOR_BGR2GRAY)<235):
            touches.append(edge_name)
    if touches:
        raise LegendEntryError('clipped_marker', f'Incomplete marker touches {"/".join(touches)} swatch boundary: expand {box}')
    # Keep every original grayscale pixel inside the completed glyph window,
    # including detached raster fragments: no denoising/fill reconstruction.
    local_gray=gray[top:bottom,left:right].copy()
    local_line=np.broadcast_to(line_rows[top:bottom,None],local_gray.shape).copy()
    result=(local_gray,local_line,(x0+left,y0+top,x0+right,y0+bottom),connected_line)
    return (*result,line_report) if return_report else result


def _independent_fill(mask, nuisance):
    """Check actual interior ink away from a connected legend line.

    A convex envelope is used only as a measurement region, never as new ink.
    A dark line drawn through a hollow marker cannot itself establish a fill:
    the region outside that line must contain enough observed dark interior.
    """
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False,0.
    hull=cv2.convexHull(max(contours,key=cv2.contourArea))
    envelope=np.zeros(mask.shape,np.uint8)
    cv2.drawContours(envelope,[hull],-1,1,-1)
    yy,xx=np.nonzero(envelope)
    diameter=max(int(np.ptp(xx))+1,int(np.ptp(yy))+1)
    inner=cv2.distanceTransform(envelope,cv2.DIST_L2,5)>=max(1.5,.15*diameter)
    independent=inner & ~nuisance
    enough=int(independent.sum())>=max(6,int(math.ceil(.20*inner.sum())))
    fill=float(mask[independent].mean()) if independent.any() else 0.
    return bool(enough and fill>=.80),fill


def _model(image, box, supplied=None):
    # Lazy import avoids a module cycle: bw_pipeline_v46 is the public adapter.
    from bw_pipeline_v46 import _shape
    gray,line,glyph_box,connected_line,line_report=_glyph_pixels(image,box,return_report=True)
    paper=float(np.percentile(cv2.cvtColor(image[box[1]:box[3],box[0]:box[2]],cv2.COLOR_BGR2GRAY),95))
    paper=max(paper,float(gray.max()),235.)
    ink_pixels=gray[gray<min(235,paper-8)]
    if not len(ink_pixels):
        raise LegendEntryError('insufficient_contrast', f'Empty or insufficient-contrast marker at {box}')
    # Crop whitespace must not change the definition of black ink.  The 64
    # floor treats normal antialiasing as ink; pale actual glyphs stay pale.
    core=min(paper-12,max(64.,float(np.percentile(ink_pixels,10))))
    ink=D.InkModel(True,(int(core),)*3,(int(paper),)*3,12.,core,paper)
    observed=np.clip((paper-gray.astype(np.float32))/max(24.,paper-core),0,1)
    mask=observed>=.25
    if not mask.any():
        raise LegendEntryError('no_supported_pixels', f'No supported marker pixels at {box}')
    inferred,fill,shape_evidence=_shape(mask,nuisance=line,return_report=True)
    name=supplied.replace('diamond','rhombus') if supplied else inferred
    # Source extent is used for diameter even when one antialiased side has
    # lower confidence.  This prevents tiny crop fragments becoming a model.
    height,width=gray.shape
    diameter=float(max(height,width))
    size=2*max(3,math.ceil(.58*diameter))+1
    top,left=(size-height)//2,(size-width)//2
    raw_soft=np.zeros((size,size),np.float32)
    raw_soft[top:top+height,left:left+width]=observed
    padded_mask=np.zeros((size,size),bool)
    padded_mask[top:top+height,left:left+width]=mask
    nuisance=np.zeros_like(padded_mask)
    nuisance[top:top+height,left:left+width]=line
    distance=cv2.distanceTransform(padded_mask.astype(np.uint8),cv2.DIST_L2,5)
    weights=np.zeros((size,size),np.float32)
    boundary=padded_mask & (distance<=1.05)
    interior=padded_mask & ~boundary
    weights[boundary]=.35+.40*raw_soft[boundary]
    weights[interior]=.85+.15*raw_soft[interior]
    weights[nuisance & padded_mask]=np.minimum(weights[nuisance & padded_mask],.20)
    # A connecting line is ambiguous at the marker boundary, but cannot make
    # the centre of an independently confirmed SOLID marker optional. Restore
    # only observed interior pixels, never white pixels or hollow crossings.
    filled_confirmed,independent_fill=_independent_fill(padded_mask,nuisance)
    # Uncertain geometry must not make independently proven solid ink optional.
    # A hollow/uncertain-fill crossing still cannot pass filled_confirmed.
    restore=(interior & nuisance) if (filled_confirmed and
        (name.startswith('filled_') or name=='unknown_marker')) else np.zeros_like(interior)
    weights[restore]=.85+.15*raw_soft[restore]
    # The raw line band remains present; the grid may use the outline while
    # full-window validation knows line crossings do not prove marker ink.
    edge=cv2.Canny(padded_mask.astype(np.uint8)*255,40,100)>0
    center=((glyph_box[0]+glyph_box[2]-1)/2,(glyph_box[1]+glyph_box[3]-1)/2)
    # Canny places some contour pixels just outside the binary foreground.
    # The grid's validity map must therefore NOT be zero outside ink; only the
    # full-window required-ink weights have that support restriction.
    valid_weight=np.ones((size,size),np.float32)
    valid_weight[nuisance]=.35
    marker_kind=('open' if name.startswith('open_') else 'filled' if name.startswith('filled_')
                 else 'filled' if fill>=.52 else 'open')
    source_gray=np.full((size,size),paper,np.float32)
    source_gray[top:top+height,left:left+width]=gray
    t=D.SwatchTemplate(name,box,center,diameter,marker_kind,fill,
                      raw_soft.copy(),padded_mask.copy(),nuisance,valid_weight,
                      raw_soft.copy(),padded_mask,edge,D._edge_orientation(raw_soft),ink,
                      required_weight=weights,matching_profile='bw_v46_uncertain',source_gray=source_gray)
    report={'box':box,'glyph_box':glyph_box,'class_name':name,
            'classification':'supplied' if supplied else 'geometric',
            'extractor':'complete_observed_multilevel_extent','template_shape':[height,width],
            'shape_hint':name,'shape_evidence':shape_evidence,
            'connected_line':connected_line,'line_analysis':line_report,
            'ink_core_gray':core,'paper_gray':paper,
            'strong_required_pixels':int((weights>=.85).sum()),
            'uncertain_boundary_pixels':int(boundary.sum()),
            'nuisance_pixels':int((nuisance & padded_mask).sum()),
            'independent_interior_fill':independent_fill,
            'restored_filled_interior_pixels':int(restore.sum()),
            'uncertain_line_pixels':int((nuisance & padded_mask & (weights<.30)).sum()),
            'matching_profile':'bw_v46_uncertain'}
    from bw_fill_identity_v46 import describe_fill
    report['fill_evidence'] = describe_fill(t, shape_evidence)
    return t,report


def extract_legend_models(image, legend_area=None, swatches=None, known_classes=None):
    """Return complete B&W ``SwatchTemplate`` models and auditable reports."""
    from bw_pipeline_v46 import CLASSES
    explicit=swatches is not None
    entries=swatches if explicit else [(None,b) for b in _automatic_boxes(image,legend_area)]
    templates=[]
    reports=[]
    for entry_index,(supplied,box) in enumerate(entries,1):
        # Invalid caller coordinates/classes must still fail loudly. Only an
        # expected extraction failure of a valid source entry is recoverable.
        box=_box(box,image)
        if supplied and supplied.replace('diamond','rhombus') not in CLASSES:
            raise ValueError(f'Unsupported marker class: {supplied}')
        identity=dict(swatch_id=f'S{entry_index:02}',series_index=entry_index-1)
        try:
            try:
                t,report=_model(image,box,supplied)
            except LegendEntryError as first:
                if first.code!='clipped_marker':raise
                # Recover at most two original-pixel border rows/columns.
                # Large/incomplete glyphs still fail; no threshold is relaxed.
                for pad in (1,2):
                    a,b,c,d=box
                    expanded=(max(0,a-pad),max(0,b-pad),min(image.shape[1],c+pad),min(image.shape[0],d+pad))
                    try:t,report=_model(image,expanded,supplied)
                    except LegendEntryError:continue
                    report['border_recovery']=dict(input_box=box,measured_box=expanded,
                        padding_px=pad,initial_error=first.code,policy='bounded_source_pixels_revalidated')
                    break
                else:raise first
        except LegendEntryError as error:
            reports.append(dict(identity,box=box,class_name='unknown_marker',shape_hint='unknown_marker',
                classification='unresolved',extraction_status='failed',template_available=False,
                extraction_error=str(error),extraction_error_code=error.code,supplied_class=supplied))
            continue
        t.swatch_id=f'S{entry_index:02}'
        t.shape_hint=t.name
        report.update(identity,extraction_status='usable',template_available=True)
        if t.name not in CLASSES and t.name!='unknown_marker':
            raise ValueError(f'Unsupported marker class: {t.name}')
        if known_classes and t.name not in known_classes:
            report['excluded_by_class_filter']=True
            reports.append(report)
            continue
        templates.append(t)
        reports.append(report)
    if not templates:
        raise LegendExtractionError(reports)
    return templates,reports

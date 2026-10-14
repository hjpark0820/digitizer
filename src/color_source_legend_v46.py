"""Recover contaminated chromatic keys from their observed source body.

Colour proposes a body rectangle, not a shape. Matching keeps source tone,
original RGB/gray, and independently observed crossing-line uncertainty. No
pixels, canonical symbols, or plot/reference point coordinates are generated.
"""
from copy import deepcopy
import cv2
import numpy as np

from color_marker_evidence import build_series_template, _membership
import partial_swatch_detector as D

VERSION = 'source_colour_body_with_context_v1'


class SourceBodyError(ValueError):
    """Expected lack of independently measurable source-body evidence."""


def _vertical_crossing(image, body, model):
    """A marker edge alone cannot establish a vertical nuisance stroke."""
    a,b,c,d=map(int,body);height=d-b
    radius=max(3,min(12,height))
    if b-radius<0 or d+radius>image.shape[0]:
        return [],dict(status='insufficient_external_context')
    upper=image[b-radius:b,a:c];lower=image[d:d+radius,a:c]
    def supported(crop):
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        own,_=_membership(crop,model)
        # A neutral axis is a nuisance, not target marker support. A coloured
        # stem can be a nuisance too, but needs the same two-sided context.
        ink=(gray<220)&((np.ptp(crop.astype(float),axis=2)<24)|(own>=.30))
        return ink.mean(axis=0)>=.70
    columns=np.flatnonzero(supported(upper)&supported(lower))
    if not len(columns):return [],dict(status='not_observed')
    runs=np.split(columns,np.flatnonzero(np.diff(columns)>1)+1)
    center=(c-a-1)/2
    eligible=[r for r in runs if len(r)<=max(3,.35*(c-a)) and abs(float(r.mean())-center)<=.3*(c-a)]
    if not eligible:return [],dict(status='no_thin_centered_stem')
    chosen=min(eligible,key=lambda r:abs(float(r.mean())-center))
    return (chosen+a).tolist(),dict(status='two_sided_source_stem',
        columns=(chosen+a).tolist(),upper_box=[a,b-radius,c,b],lower_box=[a,d,c,d+radius])


def source_body_model(image, box, legend, key, model):
    """Measure a source-colour body, or fail rather than fabricate a model."""
    if model['achromatic']:raise SourceBodyError('Source-colour recovery requires a chromatic key')
    try:
        ct=build_series_template(image,dict(id=key,swatch_box=box,legend_box=legend),_source_model=model)
    except ValueError as error:
        expected=('Fewer than three source ink pixels','No graphical ink intersects',
                  'Legend key is line-only','Legend component has no compact marker body',
                  'Marker is too small for two-dimensional shape evidence')
        if not str(error).startswith(expected):raise
        raise SourceBodyError(str(error)) from error
    a,b,c,d=map(int,ct['marker_box']);sx,sy,ex,ey=map(int,ct['source_crop_box'])
    cx,cy=ct['source_center'];height,width=d-b,c-a
    if not (box[0]<=cx<box[2] and box[1]<=cy<box[3]):
        raise SourceBodyError('Observed colour body is not associated with the original swatch')
    if min(height,width)<3 or max(height,width)>1.9*min(height,width):
        raise SourceBodyError('Colour support does not form a compact two-dimensional body')
    # Expanded source must reveal body endpoints; connector endpoints are not
    # marker clipping. A discovered box touching the expanded vertical limit
    # has no complete body evidence and remains unavailable.
    if b<=sy or d>=ey or (ct['connector_direction'] is None and (a<=sx or c>=ex)):
        raise SourceBodyError('Observed colour body reaches its source context boundary')
    raw=ct['raw_bgr'];gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(np.float32)
    support=ct['soft']>=.22
    nuisance=ct['central_connector'].copy()
    columns,vertical=_vertical_crossing(image,ct['marker_box'],model)
    radius=int(ct['center'][0]);xx=np.arange(gray.shape[1])+cx-radius
    if columns:
        nuisance |= np.broadcast_to(np.min(np.abs(xx[:,None]-np.asarray(columns)[None,:]),axis=1)<=.55,gray.shape)
    independent=support&~nuisance
    yy,xx=np.indices(gray.shape);dx,dy=xx-radius,yy-radius
    quadrants=[int(np.sum(independent&(dx*xs>0)&(dy*ys>0))) for xs,ys in ((1,1),(1,-1),(-1,1),(-1,-1))]
    if independent.sum()<max(6,.18*support.sum()) or sum(q>=2 for q in quadrants)<3:
        raise SourceBodyError('Colour body lacks independent ink beyond supported line crossings')
    paper=max(235.,float(np.percentile(gray,95)))
    core=min(paper-12,max(64.,float(np.percentile(gray[independent],10))))
    tone=np.clip((paper-gray)/max(24.,paper-core),0,1)
    # Colour establishes which source ink belongs to the body. It is not a
    # replacement intensity; preserved source grayscale sets that intensity.
    _,colour_fit=_membership(raw,model)
    observed=tone*colour_fit*np.clip(ct['soft']/np.maximum(ct['raw_soft'],1e-6),0,1)
    mask=observed>=.25
    from bw_pipeline_v46 import _shape
    name,fill,shape=_shape(mask,nuisance=nuisance,return_report=True)
    union_name,_,union_shape=_shape(mask,nuisance=None,return_report=True)
    def family(value):
        return value.removeprefix('filled_').removeprefix('open_')
    disagreement=(name!='unknown_marker' and union_name!='unknown_marker' and family(name)!=family(union_name))
    shape['source_body_family_check']=dict(off_line_class=name,body_union_class=union_name,
        off_line_margin=shape.get('shape_margin'),body_union_margin=union_shape.get('shape_margin'),
        independent_and_body_agree=not disagreement)
    if disagreement:
        # A thick connector may leave two tiny caps whose ranking flips from
        # round to square. Neither line-role interpretation gets to force the
        # family; retain the complete observed raster and independent fill.
        name='unknown_marker'
        shape.update(status='uncertain_family_across_line_roles',shape_hint=name)
    kind='open' if name.startswith('open_') else 'filled' if name.startswith('filled_') else 'open' if fill<.52 else 'filled'
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    boundary=mask&(distance<=1.05)
    weights=np.where(mask,np.where(boundary,.35+.40*observed,.85+.15*observed),0).astype(np.float32)
    weights[nuisance&mask]=np.minimum(weights[nuisance&mask],.20)
    valid=np.ones_like(observed);valid[nuisance]=.35
    ink=D.InkModel(True,(int(core),)*3,(int(paper),)*3,12.,core,paper)
    t=D.SwatchTemplate(name,tuple(box),(float(cx),float(cy)),float(max(width,height)),kind,float(fill),
        observed.copy(),mask.copy(),nuisance,valid,observed.copy(),mask,
        cv2.Canny(mask.astype(np.uint8)*255,40,100)>0,D._edge_orientation(observed),ink,
        required_weight=weights,matching_profile='bw_v46_uncertain',source_gray=gray.copy(),
        swatch_id=key,shape_hint=name)
    t.source_colour_body=dict(version=VERSION,source_box=list(ct['marker_box']),
        source_pixels_modified=False,original_rgb=raw.copy())
    report=dict(swatch_id=key,series_index=int(key[1:])-1,box=list(box),glyph_box=list(ct['marker_box']),
        class_name=name,shape_hint=name,classification='source_colour_observed_geometry',
        extraction_status='usable',template_available=True,extractor=VERSION,
        template_policy='observed_source_colour_body',template_shape=[height,width],
        source_center=[float(cx),float(cy)],connected_line=bool(ct['connector_direction']),
        line_analysis=dict(status='source_context_separated',horizontal=ct['connector_direction'],vertical=vertical),
        source_colour_body=dict(version=VERSION,source_pixels_modified=False,independent_pixels=int(independent.sum()),
            quadrant_pixels=quadrants,marker_box=list(ct['marker_box']),source_crop_box=list(ct['source_crop_box']),
            vertical=vertical),shape_evidence=shape,ink_core_gray=core,paper_gray=paper,
        strong_required_pixels=int((weights>=.85).sum()),nuisance_pixels=int((nuisance&mask).sum()),
        matching_profile='bw_v46_uncertain',model_completed=False)
    from bw_fill_identity_v46 import describe_fill
    report['fill_evidence']=describe_fill(t,shape)
    return t,report


def recover_templates(image,legend,templates,reports,palette):
    """Retry unavailable keys or independently proven polluted unknown rasters."""
    existing={t.key:t for t in templates};output=[];updated=[]
    for original_report in reports:
        r=deepcopy(original_report);key=r['swatch_id'];t=existing.get(key)
        model=palette.get(key)
        if model is None or model['achromatic']:
            if t is not None:output.append(t)
            updated.append(r);continue
        eligible=(not r.get('supplied_class') and r.get('classification')!='supplied' and
            (t is None or (getattr(t,'name',None)=='unknown_marker' and not getattr(t,'model_completed',False))))
        if eligible:
            try:
                proposal,pr=source_body_model(image,r['box'],legend,key,model)
                a,b,c,d=map(int,r['box']);raw=image[b:d,a:c]
                neutral=(np.ptp(raw.astype(float),axis=2)<24)&(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)<220)
                columns=np.arange(a,c)
                outside_body=(columns<pr['glyph_box'][0])|(columns>=pr['glyph_box'][2])
                external_stroke=outside_body&(neutral.mean(axis=0)>=.65)
                contaminated=bool(external_stroke.any())
                contamination=dict(external_neutral_columns=columns[external_stroke].tolist(),
                    retained_frame_pixels=0,retained_frame_fraction=0.,core_tone_shift=0.,
                    corrupt_body_evidence=False)
                if t is not None and contaminated:
                    # A close frame may add only a few columns to an otherwise
                    # compact crop, so a diameter ratio alone misses it. Verify
                    # that the OLD observed marker actually requires that
                    # external, neutral stroke. A frame elsewhere in the legend
                    # cannot authorize replacing a clean compact model.
                    stroke=np.zeros(image.shape[:2],np.uint8)
                    stroke[b:d,a:c]=neutral&external_stroke[None,:]
                    yy,xx=np.indices(t.mask.shape,dtype=np.float32)
                    mx=xx-(t.mask.shape[1]-1)/2+float(t.marker_center[0])
                    my=yy-(t.mask.shape[0]-1)/2+float(t.marker_center[1])
                    retained=cv2.remap(stroke,mx,my,cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)&t.mask
                    retained &= ~t.line_nuisance
                    n=int(retained.sum());fraction=n/max(int(t.mask.sum()),1)
                    tone_shift=float(proposal.ink.core_gray-t.ink.core_gray)
                    corrupt=(n>=3 and fraction>=.05 and tone_shift>=12.)
                    contamination.update(retained_frame_pixels=n,retained_frame_fraction=fraction,
                        core_tone_shift=tone_shift,corrupt_body_evidence=bool(corrupt))
                r['source_colour_proposal']=dict(diameter=proposal.diameter,glyph_box=pr['glyph_box'],
                    external_neutral_columns=columns[external_stroke].tolist(),eligible_context=contaminated,
                    contamination=contamination)
                oversized=bool(t is not None and contaminated and t.diameter>=1.6*proposal.diameter)
                if t is None or oversized or contamination['corrupt_body_evidence']:
                    pr['source_recovery']=dict(previous_status=r.get('extraction_status'),
                        previous_class=t.name if t is not None else None,
                        previous_diameter=t.diameter if t is not None else None,
                        previous_error=r.get('extraction_error_code'),neutral_context_pixels=int(neutral.sum()),
                        policy='source_body_and_external_frame_evidence_v2',
                        selection_reason='unavailable_model' if t is None else
                            'retained_foreign_frame_changes_ink_tone' if contamination['corrupt_body_evidence'] else
                            'oversized_unknown_with_source_context',
                        contamination=contamination)
                    pr['ink_model_available']=True
                    t,r=proposal,pr
            except SourceBodyError as error:
                r['source_colour_recovery']=dict(status='unresolved',reason=str(error))
        if t is not None:output.append(t)
        updated.append(r)
    return output,updated

"""Bridge supported line+symbol decompositions into the native BW grid model."""
from copy import deepcopy
import math
import cv2
import numpy as np
from legend_composition_v46 import Config,fit_legend_composition,render_model
from color_marker_evidence import _swatch_model
import partial_swatch_detector as D
from legend_open_composition_v46 import LineEvidenceError


def compose_templates(image,legend,templates,reports, *, crop_policy='padded', small_hollow=False, stroke_crosses=False):
    """Fit the observed union; retain unsupported swatches without forcing shape.

    Compact BW discovery boxes already contain the entire connected key and a
    paper margin. Expanding those boxes can import label letters or adjacent
    rows. Open/dashed callers grow only through observed thin connector
    fragments and compare transparent/paper-face hollow compositions.
    """
    if crop_policy not in ('padded','observed_swatch'):
        raise ValueError('Unknown composition crop policy: '+str(crop_policy))
    output=[];report_out=[];figures=[]
    cfg=(Config(enable_open_dashed=True,maximum_marker_relative_error=.35,maximum_marker_mae=.04)
         if small_hollow or stroke_crosses else Config(enable_open_dashed=True))
    names=dict(circle='filled_circle',ellipse='filled_circle',open_circle='open_circle',open_ellipse='open_circle',
        open_square='open_square',open_rectangle='open_square',square='filled_square',
        rectangle='filled_square',diamond='filled_rhombus',triangle_up='filled_triangle',triangle_down='filled_inv_triangle',
        open_triangle_up='open_triangle',open_triangle_down='open_inv_triangle',x_marker='x_marker',plus_marker='plus_marker')
    for original,report in zip(templates,reports):
        t=deepcopy(original);r=deepcopy(report);lx,ly,rx,by=legend;a,b,c,d=t.swatch_box
        if crop_policy=='padded':
            from bw_legend_v46 import composition_box
            box=composition_box(image,legend,t.swatch_box,t.diameter)
        else:
            box=[max(lx,a),max(ly,b),min(rx,c),min(by,d)]
        a,b,c,d=map(int,box);raw=image[b:d,a:c];model=_swatch_model(image,t.swatch_box)
        try:
            if crop_policy=='padded' or stroke_crosses:
                from legend_layered_composition_v46 import fit_layered
                rec,fields=fit_layered(raw[...,::-1],model['bgr'][::-1],cfg,small_hollow=small_hollow,include_cross=stroke_crosses)
            else:
                rec,fields=fit_legend_composition(raw[...,::-1],model['bgr'][::-1],cfg)
        except LineEvidenceError as error:
            # Optional inverse rendering must not discard a valid observed
            # template, nor let one unresolved connector abort every series.
            # Programming errors and invalid caller inputs still propagate.
            rec=dict(status='unresolved_connector',best_model_name='unresolved',
                error_code=error.code,error=str(error),
                template_policy='explicit_observed_fallback')
            fields={}
        r['composition']=rec;r['composition_source_box']=box;r['observed_glyph_box']=r['glyph_box']
        r['composition_crop_policy']=crop_policy
        supported=rec['status']=='supported_simple_shape_model'
        if stroke_crosses:
            # Unknown non-convex glyphs may resolve as strokes, never as a
            # fallback convex body merely because the model fits inside one.
            supported &= rec['best_model_name'] in ('x_marker','plus_marker')
        if small_hollow:
            # Native interior measurability, not a fixed diameter, chooses this
            # route. A large dashed swatch can leave only two independent hole
            # samples too. The fitted hollow must still beat filled/line/rivals.
            supported &= rec['best_model_name'].startswith('open_')
        r['template_policy']='pure_model_marker' if supported else 'explicit_observed_fallback'
        if supported:
            pars=rec['best_model']['params'];diameter=max(pars['width'],pars['height']);lp=rec['line_params']
            radius=int(math.ceil(diameter/2+3*lp['blur_sigma']))+2;size=2*radius+1
            # Marker_alpha is independent of every line parameter. Shift the
            # line too for diagnostic consistency, but never use its pixels.
            shiftx=radius-pars['cx'];shifty=radius-pars['cy']
            line=dict(lp,cx=lp['cx']+shiftx,cy=lp['cy']+shifty,x0=lp['x0']+shiftx,x1=lp['x1']+shiftx)
            if line.get('style')=='dashed':line['dash_phase']+=shiftx
            pure=render_model(rec['best_model_name'],dict(pars,cx=radius,cy=radius),line,(size,size),cfg)['marker_alpha']
            transform=np.float32([[1,0,shiftx],[0,1,shifty]])
            source_gray=cv2.warpAffine(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(np.float32),transform,(size,size),borderValue=t.ink.paper_gray)
            observed=cv2.warpAffine(fields['observed_alpha'],transform,(size,size))
            mask=pure>=.25;depth=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
            weight=np.where(mask,np.where(depth<=1.05,.35+.40*pure,.85+.15*pure),0).astype(np.float32)
            name=names.get(rec['best_model_name'],'unknown_marker')
            t.name=name;t.shape_hint=name;t.marker_kind='open' if name.startswith('open_') else 'filled'
            t.marker_center=(a+pars['cx'],b+pars['cy']);t.diameter=float(diameter)
            t.raw_soft=pure.copy();t.soft=pure.copy();t.raw_mask=mask.copy();t.mask=mask
            t.line_nuisance=np.zeros_like(mask);t.valid_weight=np.ones_like(pure)
            t.edge=cv2.Canny(mask.astype(np.uint8)*255,40,100)>0;t.orientation=D._edge_orientation(pure)
            t.required_weight=weight;t.source_gray=source_gray;t.model_completed=True;t.observed_source_soft=observed
            t.interior_fill=0. if name.startswith('open_') else 1.
            if name in ('x_marker','plus_marker'):
                t.cross_stroke_model=dict(name=name,params=pars,
                    blur_sigma=pars.get('marker_blur_sigma',lp['blur_sigma']))
            if small_hollow:
                t.small_hollow_model=dict(name=rec['best_model_name'],params=pars,
                    blur_sigma=lp['blur_sigma'],legend_family_margin=rec['winner_other_family_margin'])
                r['small_hollow_model']=t.small_hollow_model
            r.update(class_name=name,shape_hint=name,glyph_box=[t.marker_center[0]-pars['width']/2,t.marker_center[1]-pars['height']/2,
                t.marker_center[0]+pars['width']/2,t.marker_center[1]+pars['height']/2],model_completed=True,
                fitted_diameter=t.diameter,source_center=t.marker_center,expected_ink_source='pure marker_alpha in BOTH raw_soft and soft',
                observed_ink_source='source_gray / observed_source_soft; NOT required template ink')
        output.append(t);report_out.append(r)
        figures.append(dict(swatch_id=t.key,raw=raw,record=rec,fields=fields,box=box))
    return output,report_out,figures


def compose_small_hollow(image,legend,templates,reports):
    """Resolution-specific inverse composition, never a supplied-shape override.

    Fixed ten-pixel fill samples can be impossible for tiny or line-crossed
    symbols. Use the measured independent sample count, not diameter alone.
    Compare open AND filled shape/line hypotheses before choosing this route.
    A low residual alone is insufficient: the usual family margin remains.
    """
    if legend is None:return templates,reports
    output=[];report_out=[]
    for t,r in zip(templates,reports):
        interior=r.get('fill_evidence',{}).get('interior_pixels')
        few_samples=interior is not None and interior<10
        if (t.ink.achromatic and (t.diameter<=10 or few_samples) and not t.model_completed and
            r.get('classification')!='supplied' and
            r.get('fill_evidence',{}).get('style') in ('uncertain','open') and
            r.get('line_analysis',{}).get('status')=='separated'):
            ts,rs,_=compose_templates(image,legend,[t],[r],small_hollow=True)
            if getattr(ts[0],'small_hollow_model',None):t,r=ts[0],rs[0]
            else:
                r=deepcopy(r);r['small_hollow_attempt']=rs[0].get('composition')
        output.append(t);report_out.append(r)
    return output,report_out


def compose_compact_unknowns(image, legend, templates, reports):
    """Try existing line-union-marker fitting BEFORE requiring a known class.

    Only previously unclassified compact keys enter this recovery. Patterned
    fills keep the geometry/fill route except tiny independently hollow bare
    triangles, whose outer identity is resolved without replacing their raster.
    Explicit classes and calibrated
    colour-group templates are not overwritten. Identity and observations stay
    unchanged even when fitting abstains.
    """
    if legend is None:
        return templates,reports
    output=[];report_out=[]
    for t,r in zip(templates,reports):
        # A native tiny hollow outline may look patterned when its own rim
        # enters a fixed interior sample. Resolve OUTER identity independently,
        # but keep the observed raster and size: a completed tiny triangle can
        # otherwise erase real error-bar/connector evidence in the final check.
        shape=r.get('shape_evidence',{})
        if (t.name=='unknown_marker' and t.ink.achromatic and t.diameter<=16
            and not t.model_completed and r.get('classification')!='supplied'
            and not shape.get('strong_hollow_evidence') and not shape.get('patterned_internal_evidence')
            and shape.get('convex_occupancy',1.)<.77
            and max(shape.get('cross_scores',{}).values(),default=0.)>=.5):
            ts,rs,_=compose_templates(image,legend,[t],[r],crop_policy='observed_swatch',stroke_crosses=True)
            if ts[0].name in ('x_marker','plus_marker') and ts[0].model_completed:
                t,r=ts[0],rs[0]
                r['cross_legend_recovery']=dict(previous='unknown_marker',current=t.name,
                    reason='line_plus_cross_beats_closed_body_rivals')
            else:
                r=deepcopy(r);r['cross_legend_attempt']=rs[0]['composition']
        if (t.name=='unknown_marker' and t.ink.achromatic and t.diameter<=10
            and not t.model_completed and r.get('classification')!='supplied'
            and r.get('line_analysis',{}).get('status')=='not_needed'
            and shape.get('strong_hollow_evidence') and not shape.get('patterned_internal_evidence')
            and shape.get('best_shape') in ('triangle','inv_triangle')):
            _,attempt,_=compose_templates(image,legend,[t],[r],small_hollow=True)
            fit=attempt[0]['composition'];name=fit.get('best_model_name')
            if (fit.get('status')=='supported_simple_shape_model'
                and name in ('open_triangle_up','open_triangle_down')
                and fit.get('winner_other_family_margin',0)>=.005):
                t=deepcopy(t);r=deepcopy(r)
                t.name='open_triangle' if name=='open_triangle_up' else 'open_inv_triangle'
                t.shape_hint=t.name;t.marker_kind='open'
                t.compact_outer_identity=dict(version='bare-hollow-outer-fit-v1',family=name,
                    fit=fit,search_raster='unchanged observed legend',fill_resolved=False)
                r.update(class_name=t.name,shape_hint=t.name,outer_identity=t.compact_outer_identity,
                         original_class='unknown_marker',template_policy='observed_raster_with_outer_identity')
        eligible=(t.name=='unknown_marker' and not t.model_completed
            and r.get('classification')!='supplied'
            and r.get('line_analysis',{}).get('status')=='not_needed'
            and r.get('fill_evidence',{}).get('style') not in ('patterned','partial'))
        if eligible:
            ts,rs,_=compose_templates(image,legend,[t],[r],crop_policy='observed_swatch')
            t,r=ts[0],rs[0]
            r['compact_composition_attempted']=True
        output.append(t);report_out.append(r)
    return output,report_out

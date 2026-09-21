"""Bridge supported line+symbol decompositions into the native BW grid model."""
from copy import deepcopy
import math
import cv2
import numpy as np
from legend_composition_v46 import Config,fit_legend_composition,render_model
from color_marker_evidence import _swatch_model
import partial_swatch_detector as D


def compose_templates(image,legend,templates,reports, *, crop_policy='padded'):
    """Fit the observed union; retain unsupported swatches without forcing shape.

    Compact BW discovery boxes already contain the entire connected key and a
    paper margin. Expanding those boxes can import label letters or adjacent
    rows. Open/dashed callers grow only through observed thin connector
    fragments and compare transparent/paper-face hollow compositions.
    """
    if crop_policy not in ('padded','observed_swatch'):
        raise ValueError('Unknown composition crop policy: '+str(crop_policy))
    output=[];report_out=[];figures=[]
    cfg=Config(enable_open_dashed=True)
    names=dict(circle='filled_circle',ellipse='filled_circle',open_circle='open_circle',open_ellipse='open_circle',
        open_square='open_square',open_rectangle='open_square',square='filled_square',
        rectangle='filled_square',diamond='filled_rhombus',triangle_up='filled_triangle',triangle_down='filled_inv_triangle')
    for original,report in zip(templates,reports):
        t=deepcopy(original);r=deepcopy(report);lx,ly,rx,by=legend;a,b,c,d=t.swatch_box
        if crop_policy=='padded':
            from bw_legend_v46 import composition_box
            box=composition_box(image,legend,t.swatch_box,t.diameter)
        else:
            box=[max(lx,a),max(ly,b),min(rx,c),min(by,d)]
        a,b,c,d=map(int,box);raw=image[b:d,a:c];model=_swatch_model(image,t.swatch_box)
        if crop_policy=='padded':
            from legend_layered_composition_v46 import fit_layered
            rec,fields=fit_layered(raw[...,::-1],model['bgr'][::-1],cfg)
        else:
            rec,fields=fit_legend_composition(raw[...,::-1],model['bgr'][::-1],cfg)
        r['composition']=rec;r['composition_source_box']=box;r['observed_glyph_box']=r['glyph_box']
        r['composition_crop_policy']=crop_policy
        supported=rec['status']=='supported_simple_shape_model'
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
            r.update(class_name=name,shape_hint=name,glyph_box=[t.marker_center[0]-pars['width']/2,t.marker_center[1]-pars['height']/2,
                t.marker_center[0]+pars['width']/2,t.marker_center[1]+pars['height']/2],model_completed=True,
                fitted_diameter=t.diameter,source_center=t.marker_center,expected_ink_source='pure marker_alpha in BOTH raw_soft and soft',
                observed_ink_source='source_gray / observed_source_soft; NOT required template ink')
        output.append(t);report_out.append(r)
        figures.append(dict(swatch_id=t.key,raw=raw,record=rec,fields=fields,box=box))
    return output,report_out,figures


def compose_compact_unknowns(image, legend, templates, reports):
    """Try existing line-union-marker fitting BEFORE requiring a known class.

    Only previously unclassified compact keys enter this recovery. Patterned
    fills keep the geometry/fill route; explicit classes and calibrated
    colour-group templates are not overwritten. Identity and observations stay
    unchanged even when fitting abstains.
    """
    if legend is None:
        return templates,reports
    output=[];report_out=[]
    for t,r in zip(templates,reports):
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

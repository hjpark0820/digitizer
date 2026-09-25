"""Production legend-composition adapter; location and colour stay observed.

The hybrid locator owns entry identity. This module may replace its *shape*
with a supported line-union-primitive explanation of the original full swatch.
It never samples inferred template pixels to redefine a series colour.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math

import cv2
import numpy as np

import color_marker_evidence as evidence_base
from color_marker_evidence_v2 import _template_regions
from legend_composition_v46 import Config, fit_legend_composition, render_model

VERSION = 'v46_observed_locator_inverse_legend_composition_v3_independent_hollow'


def full_swatch_box(image, entry, legend_box):
    """Recover observed connector tails discarded by the locator's audit crop.

    Search only the supplied legend, near the identified row; take the connected
    target-colour component intersecting the body, never a broad text window.
    All returned coordinates are half-open source-image coordinates.
    """
    bx0, by0, bx1, by1 = map(int, entry['box'])
    lx0, ly0, lx1, ly1 = map(int, legend_box)
    diameter = float(entry.get('report', {}).get('diameter', max(bx1-bx0, by1-by0)))
    pad = max(3, int(math.ceil(.70*diameter)))
    area = evidence_base._box([max(lx0, bx0-3*diameter), max(ly0, by0-pad),
                              min(lx1, bx1+3*diameter), min(ly1, by1+pad)], image.shape)
    x0,y0,x1,y1 = area
    model = evidence_base._swatch_model(image, entry['box'], entry['rgb'])
    soft, _ = evidence_base._membership(image[y0:y1,x0:x1], model)
    support = soft >= .25
    n, labels, _, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    seed = np.zeros(support.shape, bool)
    a,b,c,d = max(bx0,x0)-x0,max(by0,y0)-y0,min(bx1,x1)-x0,min(by1,y1)-y0
    seed[b:d,a:c] = True
    choices = [(int(((labels==k)&seed).sum()),k) for k in range(1,n)]
    if not choices or max(choices)[0] < 3:
        raise ValueError('No observed swatch component intersects located body')
    component = labels == max(choices)[1]
    a,b,c,d = evidence_base._tight(component)
    margin = max(3, int(math.ceil(.12*diameter)))
    box = (max(lx0,x0+a-margin),max(ly0,y0+b-margin),
           min(lx1,x0+c+margin),min(ly1,y0+d+margin))
    if min(box[2]-box[0],box[3]-box[1]) < 5:
        raise ValueError('Full observed swatch crop is too small')
    return box


def _warp(value, source_centre, radius, *, paper=None):
    transform = np.float32([[1,0,radius-source_centre[0]],
                            [0,1,radius-source_centre[1]]])
    kwargs = {} if paper is None else {'borderValue':tuple(map(float,paper))}
    return cv2.warpAffine(value,transform,(2*radius+1,2*radius+1),
                          flags=cv2.INTER_LINEAR,**kwargs)


def _base_template(image, entry, soft, raw, raw_soft, source_center, box,
                   swatch, radius, model, provenance):
    support = soft >= .18
    distance = cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,5)
    template = dict(id=entry['id'], label=entry['id'], rgb=list(entry['rgb']),
        model=model, diameter=float(max(box[2]-box[0],box[3]-box[1])),
        soft=np.asarray(soft,np.float32),core=soft>=.60,
        weight=(soft*np.where(distance>=1.5,1.,.65)).astype(np.float32),
        nuisance=(raw_soft>=.15)&~support,
        central_connector=np.zeros(soft.shape,bool),uncertain=np.zeros(soft.shape,bool),
        raw_bgr=raw,raw_soft=raw_soft,center=[radius,radius],
        source_center=list(map(float,source_center)),swatch_box=list(swatch),
        marker_box=list(box),source_crop_box=list(swatch),legend_box=None,
        connector_direction=None,hollow_fraction=0.,evidence_version=VERSION,
        provenance=provenance)
    return _template_regions(template)


def enrich_entry(image_bgr, entry, legend_box, cfg=Config()):
    """Attach a model-completed template without changing observed entry arrays.

    Unsupported compositions leave the original hybrid template available.
    The model bbox centre is explicit; a triangle centroid is *not* substituted
    for a data anchor. Image pixels alone cannot establish plotting conventions.
    """
    image = np.asarray(image_bgr,np.uint8)
    entry['sample_mask'] = np.asarray(entry['mask'],bool).copy()
    entry['composition_template'] = None
    if not entry.get('marker_template',True):
        entry['composition'] = dict(status='line_only_not_fitted',
            template_policy='observed_line_only_no_marker',version=VERSION)
        return entry
    try:
        box = full_swatch_box(image,entry,legend_box)
        x0,y0,x1,y1 = box
        raw = image[y0:y1,x0:x1]
        hint = str(entry.get('report', {}).get('shape_hint', '')).lower()
        shape_evidence = entry.get('report', {}).get('shape_evidence') or {}
        hollow = hint.startswith(('open_', 'hollow_')) or bool(shape_evidence.get('strong_hollow_evidence'))
        if hollow:
            # A filled-only model can explain a thick ring plus its connector
            # well enough to pass the global fit cutoff. It must not erase the
            # independently observed hole before scale/window verification.
            from legend_layered_composition_v46 import fit_layered
            record, fields = fit_layered(raw[...,::-1], entry['rgb'],
                                        replace(cfg, enable_open_dashed=True))
            if record['status'] == 'unknown_poor_fit':
                # A partial connector inside the hole does not invalidate the
                # independently visible rim. Do not loosen fit thresholds or
                # use the legend label as a shape verdict: all primitive
                # families compete on one fixed observable pixel domain.
                retry, retry_fields = fit_layered(raw[...,::-1], entry['rgb'],
                    replace(cfg, enable_open_dashed=True), connector_independent=True)
                retry['full_composition_attempt'] = {k:record.get(k) for k in
                    ('status','best_model_name','best_model','winner_other_family_margin')}
                if retry['status'] == 'supported_simple_shape_model':
                    record, fields = retry, retry_fields
                else:
                    record['connector_independent_attempt'] = {k:retry.get(k) for k in
                        ('status','best_model_name','best_model','independent_marker_fraction')}
        else:
            record, fields = fit_legend_composition(raw[...,::-1],entry['rgb'],cfg)
        record.update(version=VERSION,source_swatch_box=list(box),
            crop_policy='Observed connected component intersecting hybrid body, with paper margin',
            template_policy='observed_hybrid_fallback')
        entry['composition'] = record
        if record['status'] != 'supported_simple_shape_model':
            return entry
        model_open = record['best_model_name'].startswith('open_')
        if hollow and not model_open:
            record['runtime_fallback_reason'] = 'filled_completion_conflicts_with_observed_hollow'
            return entry
        pars = record['best_model']['params']
        lp = record['line_params']
        # The line model requires independently visible flanks, not only a
        # primitive's own wide horizontal cross-section.
        left = pars['cx']-pars['width']/2-lp['x0']
        right = lp['x1']-(pars['cx']+pars['width']/2)
        minimum_flank = max(2.,.12*max(pars['width'],pars['height']))
        record['independent_flanks_px'] = [float(left),float(right)]
        if min(left,right) < minimum_flank:
            record['runtime_fallback_reason'] = 'insufficient_independent_line_flanks'
            return entry
        model = evidence_base._swatch_model(image,box,entry['rgb'])
        observed_soft,_ = evidence_base._membership(raw,model)
        diameter = float(max(pars['width'],pars['height']))
        radius = int(math.ceil(diameter/2+3*lp['blur_sigma']))+2
        source_center = [x0+pars['cx'],y0+pars['cy']]
        # Render directly on the target pixel grid, avoiding a second
        # antialiasing pass when centering a fractional-coordinate fit.
        target_pars = dict(pars,cx=float(radius),cy=float(radius))
        target_line = dict(lp,cx=lp['cx']+radius-pars['cx'],
            cy=lp['cy']+radius-pars['cy'],x0=lp['x0']+radius-pars['cx'],
            x1=lp['x1']+radius-pars['cx'])
        centered = render_model(record['best_model_name'],target_pars,target_line,
                                (2*radius+1,2*radius+1),cfg)
        soft = centered['marker_alpha']
        raw_centered = _warp(raw,(pars['cx'],pars['cy']),radius,paper=model['paper_bgr'])
        raw_soft = _warp(observed_soft,(pars['cx'],pars['cy']),radius)
        marker_box = [source_center[0]-pars['width']/2,source_center[1]-pars['height']/2,
                      source_center[0]+pars['width']/2,source_center[1]+pars['height']/2]
        template = _base_template(image,entry,soft,raw_centered,raw_soft,source_center,
            marker_box,box,radius,model,dict(kind='legend_model_completion',
                extraction=VERSION,source_bbox=list(box),supporting_count=1,
                shape_status=record['status'],shape_family=record['best_model_name'],
                hidden_pixels_are_observed=False,probability_calibrated=False))
        template['diameter'] = diameter
        template['central_connector'] = centered['hidden_marker_by_line']>.15
        if model_open:
            # A transparent hollow key may have a connector through its hole.
            # Keep observed-paper masks separate; the MODEL's negative space
            # still has to survive even when source paper covers <70% of it.
            hole = template['face'] & (soft < .12) & ~template['boundary_uncertain']
            template['hole_core'] = cv2.erode(hole.astype(np.uint8), np.ones((3,3),np.uint8)) > 0
            template['hollow_fraction'] = float(hole.sum()/max(int(template['face'].sum()),1))
            template['provenance']['hole_core_source'] = 'supported_hollow_model_not_observed_paper'
        # A supported filled primitive has no measured paper holes. Keep the
        # completion uncertainty separate from hollow-marker negative evidence.
        if not model_open:
            for key in ('hole_core','enclosed_paper','observed_rim','uncertain'):
                template[key] = np.zeros(soft.shape,bool)
            template['hollow_fraction'] = 0.
        centroid = record.get('inferred_centroid') or [pars['cx'], pars['cy']]
        centroid_offset = [centroid[i]-pars['cx' if i==0 else 'cy'] for i in (0,1)]
        center_info = dict(convention='geometric_bounding_box_center',
            source_geometric_center=source_center,template_geometric_center=[radius,radius],
            source_fractional_offset=[v-round(v) for v in source_center],
            centroid_minus_geometric_center=centroid_offset,
            line_y_minus_geometric_center=float(lp['cy']+lp['slope']*(pars['cx']-lp['cx'])-pars['cy']),
            data_anchor_status='not_identifiable_from_legend_pixels_alone',
            data_anchor_policy='retain_geometric_center; no automatic centroid shift')
        template.update(center_convention=center_info,composition=deepcopy(record),
            completion_hidden_alpha=centered['hidden_marker_by_line'],
            completion_visible_alpha=centered['visible_marker_evidence'])
        record.update(template_policy='supported_composition_standalone_shape',
                      center_convention=center_info)
        template['composition']['template_policy'] = record['template_policy']
        entry['composition_template'] = template
    except (ValueError,cv2.error) as error:
        entry['composition'] = dict(status='runtime_unusable_composition',version=VERSION,
            template_policy='observed_hybrid_fallback',reason=f'{type(error).__name__}: {error}')
    return entry


def marker_template_from_entry(image_bgr, entry, series_id, label=None, legend_box=None):
    """Return the prepared composition or observed hybrid fallback, never re-locate.

    None represents a line-only legend. Callers may keep its colour for path
    extraction, but must not silently manufacture a marker from the line key.
    """
    if not entry.get('marker_template',True):
        return None
    if entry.get('composition_template') is not None:
        template = deepcopy(entry['composition_template'])
    else:
        image = np.asarray(image_bgr,np.uint8)
        box = tuple(map(int,entry['box']))
        x0,y0,x1,y1 = box
        diameter = float(entry.get('report',{}).get('diameter',max(x1-x0,y1-y0)))
        centre = [(x1-x0-1)/2,(y1-y0-1)/2]
        radius = int(math.ceil(max(diameter,x1-x0,y1-y0)/2))+2
        model = evidence_base._swatch_model(image,box,entry['rgb'])
        raw = np.asarray(entry['raw_bgr'],np.uint8)
        observed_soft = np.asarray(entry['soft'],np.float32)
        if raw.shape[:2] != observed_soft.shape:
            raise ValueError('Prepared hybrid raw/soft template shapes disagree')
        soft = _warp(observed_soft,centre,radius)
        raw_centered = _warp(raw,centre,radius,paper=model['paper_bgr'])
        template = _base_template(image,entry,soft,raw_centered,soft.copy(),
            [x0+centre[0],y0+centre[1]],box,
            entry.get('composition',{}).get('source_swatch_box',entry.get('report',{}).get('swatch_box',box)),
            radius,model,dict(kind='legend_observed_hybrid_fallback',extraction=VERSION,
                source_bbox=list(box),supporting_count=1,probability_calibrated=False,
                composition_status=entry.get('composition',{}).get('status','not_attempted')))
        # Preserve any weak/uncertain connector-interior weights already found
        # by the hybrid locator. Fallback must not promote those pixels merely
        # because no primitive completion was supported.
        required = np.asarray(entry.get('required_weight',observed_soft),np.float32)
        if required.shape != observed_soft.shape:
            raise ValueError('Prepared hybrid required-weight shape disagrees')
        preserved = _warp(required,centre,radius)
        template['weight'] = np.minimum(template['weight'],preserved)
        uncertain_source = ((np.asarray(entry['mask'],bool)) &
                            (required < .25*observed_soft)).astype(np.float32)
        prior_uncertain = _warp(uncertain_source,centre,radius)>.25
        template['uncertain'] |= prior_uncertain
        template['central_connector'] |= prior_uncertain
        template['diameter'] = diameter
        template['composition'] = deepcopy(entry.get('composition',{}))
    template.update(id=str(series_id),label=str(label or series_id),legend_box=legend_box,
        legend_shape_hint=entry.get('report',{}).get('shape_hint','unknown_marker'),
        legend_shape_evidence=deepcopy(entry.get('report',{}).get('shape_evidence',{})))
    return template


def shape_fields_from_template(template):
    """Expose the *same* prepared geometry to tentative-marker shape scoring.

    Both confirmed and tentative paths therefore use one scale and one centre.
    A model-completed silhouette is labelled as such, never observed RGB ink.
    """
    soft = np.asarray(template['soft'],np.float32)
    mask = soft >= .5
    edge = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,
                          np.ones((3,3),np.uint8)).astype(bool)
    info = dict(diameter=float(template['diameter']),method=VERSION,
        center=list(template['center']),source_center=list(template['source_center']),
        provenance=deepcopy(template['provenance']),
        center_convention=deepcopy(template.get('center_convention',{})))
    fields = dict(template_expected=soft.copy(),template_mask=mask,
        template_weight=np.asarray(template['weight'],np.float32).copy(),
        template_edge=edge,line_uncertain=np.zeros(mask.shape,bool))
    return info,fields

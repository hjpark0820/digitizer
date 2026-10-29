"""Read-only adapter for opt-in candidate corner/arc diagnostics.

No result of this module changes selection, active points, suppressed points,
or a record's existing identity. Missing source geometry is reported, not fitted
from target pixels. Image and source records must use the same working scale.
"""
from __future__ import annotations

from collections import Counter
import math
import time

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from bw_corner_identity_v46 import corner_evidence

VERSION = 'fixed-corner-candidate-shadow-v1'
SUPPORTED_CLASSES = ('open_circle','filled_circle','open_square','filled_square')


def geometry_from_mask(mask, crop_center, *, open_marker=False,
                       provenance='exact_window_template_mask'):
    """Translate the already rendered model mask into source-image geometry.

    ``crop_center`` is the window crop's ORIGINAL anchor (v.x,v.y), not the
    refined marker centre. A rendered mask already contains that refinement.
    This avoids applying alignment twice. The mask is never target-derived.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError('A nonempty rendered source mask is required')
    env = binary_fill_holes(mask)
    yy, xx = np.nonzero(env)
    if not len(xx):
        raise ValueError('Rendered source mask contains no body')
    origin = np.array([int(round(crop_center[0]))-mask.shape[1]//2,
                       int(round(crop_center[1]))-mask.shape[0]//2], float)
    center = origin + [(float(xx.min())+float(xx.max()))/2,
                       (float(yy.min())+float(yy.max()))/2]
    result = dict(center=center.tolist(),
        radii=[float(xx.max()-xx.min()+1)/2, float(yy.max()-yy.min()+1)/2],
        provenance=provenance, source_mask_shape=list(mask.shape),
        source_mask_origin=origin.tolist(), target_geometry_fitted=False)
    if open_marker:
        from bw_circle_boundary_v46 import _ring_hole
        hole = _ring_hole(env,mask)
        hy,hx = np.nonzero(hole)
        if len(hx)>=6:
            result['inner_radii']=[float(hx.max()-hx.min()+1)/2,float(hy.max()-hy.min()+1)/2]
            result['inner_geometry_provenance']='enclosed_source_model_hole_extents_at_shared_outer_center'
        else:
            result['inner_geometry_unavailable']='source_model_has_no_measurable_enclosed_hole'
    return result


def _source_rows(swatches):
    if isinstance(swatches,dict):
        return {str(key):value for key,value in swatches.items() if isinstance(value,dict)}
    return {str(row.get('swatch_id',row.get('key',''))):row for row in swatches if isinstance(row,dict)}


def _json_geometry(value):
    if isinstance(value,dict):return {str(k):_json_geometry(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_geometry(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def _source_amplitude(source):
    value = source.get('source_ink_level')
    if value is not None:
        value=float(value)
        if math.isfinite(value) and 0<value<=1:
            return value,'explicit_source_ink_level'
    # These are source-swatch photometry fields, never local target maxima.
    core=source.get('ink_core_gray',source.get('core_gray'))
    paper=source.get('paper_gray',255.)
    nested=source.get('ink',{})
    if core is None and isinstance(nested,dict):
        core=nested.get('core_gray')
        paper=nested.get('paper_gray',paper)
    if core is not None:
        value=(float(paper)-float(core))/255.
        if math.isfinite(value) and 0<value<=1:
            return value,'source_swatch_paper_minus_core_gray_over_255'
    # Saved correction models may retain grayscale source pixels but not the
    # extractor's photometric fields. A source-only estimate is still explicit.
    raster=source.get('overlap_source',{}).get('ink')
    if raster is not None:
        array=np.asarray(raster,float)
        positive=array[np.isfinite(array)&(array>0)&(array<=1)]
        if positive.size:
            return float(np.quantile(positive,.90)),'saved_legend_overlap_source_positive_quantile_90'
    return None,'missing_source_photometry'


def _eligibility(record):
    reasons=[]
    if record.get('decision') in ('ambiguous','rejected'):
        reasons.append('window_'+record['decision'])
    native=record.get('geometry_first',{}) or {}
    if native.get('decision') in ('abstain','conflict'):
        reasons.append('geometry_'+native['decision'])
    shape=native.get('circle_shape',{}) or {}
    margin=shape.get('shape_margin')
    if margin is not None and float(margin)<.05:
        reasons.append('circle_shape_margin_below_0_05')
    raster=record.get('raster_identity',{}) or {}
    own=(raster.get('scores',{}) or {}).get(record.get('swatch_id'),{}) or {}
    if own.get('passed') is False:
        reasons.append('own_raster_model_rejected')
    if raster.get('winner') not in (None,record.get('swatch_id')):
        reasons.append('different_swatch_raster_winner')
    if raster.get('margin') is not None and float(raster['margin'])<.05:
        reasons.append('raster_margin_below_0_05')
    exclusion=str(record.get('exclusion_reason',''))
    if exclusion.startswith(('raster_','grid_identity','interior_fill_conflict')):
        reasons.append(exclusion)
    return reasons


def _old_scores(record):
    native=record.get('geometry_first',{}) or {}
    circle=native.get('circle_shape',{}) or {}
    raster=record.get('raster_identity',{}) or {}
    own=(raster.get('scores',{}) or {}).get(record.get('swatch_id'),{}) or {}
    return dict(grid_score=record.get('score'),required_recall=record.get('required_recall'),
        circle_score=circle.get('circle_score'),square_score=circle.get('square_score'),
        circle_shape_margin=circle.get('shape_margin'),circle_decision=circle.get('decision'),
        own_raster_loss=own.get('loss'),own_raster_passed=own.get('passed'),
        raster_winner=raster.get('winner'),raster_margin=raster.get('margin'))


def _selection_priority(reasons):
    """Budget explicit identity disputes before generic weak window proposals."""
    if any(reason in ('own_raster_model_rejected','different_swatch_raster_winner')
           or reason.startswith(('raster_marker_','raster_identity_','grid_identity')) for reason in reasons):
        return 0,'explicit_raster_or_swatch_identity_conflict'
    if 'circle_shape_margin_below_0_05' in reasons:
        return 1,'circle_shape_low_margin'
    if 'raster_margin_below_0_05' in reasons:
        return 2,'raster_low_margin'
    if any(reason.startswith('geometry_') for reason in reasons):
        return 3,'geometry_abstention_or_conflict'
    if reasons:
        return 4,'generic_window_ambiguity_or_rejection'
    return 5,'not_eligible'


def _saved_geometry(record):
    polygon=record.get('footprint_polygon')
    if polygon is not None:
        xy=np.asarray(polygon,float)
        if xy.ndim==2 and xy.shape[1]==2 and len(xy)>=3 and np.isfinite(xy).all():
            low,high=xy.min(axis=0),xy.max(axis=0)
            result=dict(center=((low+high)/2).tolist(),radii=((high-low+1)/2).tolist(),
                provenance='saved_rendered_model_footprint_bbox_outer_only',
                inner_geometry_unavailable='saved_footprint_is_outer_envelope_not_hole',target_geometry_fitted=False)
            # The saved paired-boundary diagnostic sometimes records exact
            # model extents. Use its hole only when the OUTER extent agrees,
            # never substitute a target-estimated hole or repair disagreement.
            hollow=record.get('open_interior',{}) or {}
            outer=hollow.get('model_outer_radii');inner=hollow.get('model_inner_radii')
            if str(record.get('class_name','')).startswith('open_') and outer is not None and inner is not None:
                outer=np.asarray(outer,float);inner=np.asarray(inner,float)
                if (outer.shape==(2,) and inner.shape==(2,) and np.isfinite(np.r_[outer,inner]).all()
                        and np.all(inner>0) and np.all(inner<outer)
                        and np.allclose(outer,result['radii'],rtol=0,atol=1.e-6)):
                    result.update(inner_radii=inner.tolist(),
                        provenance='saved_rendered_footprint_with_matching_paired_model_extents',
                        inner_geometry_provenance='saved_open_interior_model_inner_radii')
                    result.pop('inner_geometry_unavailable',None)
            return result
    encoded=record.get('marker_mask')
    if encoded is not None:
        from bw_suppressed_v46 import decode_marker_mask
        aligned=[record.get('aligned_x',record.get('cx',record.get('x'))),
                 record.get('aligned_y',record.get('cy',record.get('y')))]
        if None not in aligned:
            crop_center=[aligned[0]-float(record.get('marker_offset_x',0.)),
                         aligned[1]-float(record.get('marker_offset_y',0.))]
            return geometry_from_mask(decode_marker_mask(encoded),crop_center,
                open_marker=str(record.get('class_name','')).startswith('open_'),
                provenance='saved_marker_mask_with_recorded_alignment_offset')
    return None


def _global_coordinates(diagnostic, origin):
    """Translate this newly created diagnostic, never caller-owned data."""
    seen=set()
    def walk(value):
        if isinstance(value,dict):
            if id(value) in seen:return
            seen.add(id(value))
            samples=value.get('samples')
            if isinstance(samples,dict) and id(samples) not in seen:
                for key in ('xy','ink_xy','paper_xy'):
                    if key in samples:
                        samples[key]=(np.asarray(samples[key])+origin).tolist()
                seen.add(id(samples))
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(diagnostic)
    diagnostic['geometry']['center']=(np.asarray(diagnostic['geometry']['center'])+origin).tolist()
    if 'gradient_roi' in diagnostic:
        diagnostic['gradient_roi']=(np.asarray(diagnostic['gradient_roi'])+np.tile(origin,2)).tolist()
    diagnostic['coordinate_frame']='global_working_image'


def audit_records(image, records, swatches, *, geometries=None,
                  available=None, max_candidates=64, capture=False):
    """Audit unresolved circle/square candidates without mutating any records.

    ``image`` is the detector's working BGR uint8 raster (including its existing
    descreen transform), grayscale uint8, or 2-D darkness float in [0,1].
    ``geometries`` is optional candidate-index -> global source geometry. Exact
    sidecars may include inner_radii. Old saved model footprints are an explicit
    outer-only fallback. Missing geometry/source amplitude is a recorded skip.
    The bounded budget prioritizes explicit raster identity conflicts, circle
    low margins, and then generic window failures. Ties use saved order, never
    GT, desired outcome, or a coordinate-specific rule. Input order is intact.
    """
    started=time.perf_counter()
    array=np.asarray(image)
    if array.ndim==3 and array.shape[2]==3 and array.dtype==np.uint8:
        gray=cv2.cvtColor(array,cv2.COLOR_BGR2GRAY)
        image_kind='bgr_uint8'
    elif array.ndim==2 and array.dtype==np.uint8:
        gray=array
        image_kind='gray_uint8'
    elif array.ndim==2 and np.issubdtype(array.dtype,np.floating):
        if not np.isfinite(array).all() or np.any(array<0) or np.any(array>1):
            raise ValueError('Darkness image must be finite and in [0,1]')
        gray=array
        image_kind='darkness_float'
    else:
        raise ValueError('image must be BGR/gray uint8 or 2-D darkness float')
    if not gray.size:
        raise ValueError('image must be nonempty')
    if not isinstance(max_candidates,int) or isinstance(max_candidates,bool) or max_candidates<0:
        raise ValueError('max_candidates must be a nonnegative integer')
    visibility=None if available is None else np.asarray(available)
    if visibility is not None and visibility.shape!=gray.shape:
        raise ValueError('available must match working image dimensions')
    if visibility is not None and (not np.isfinite(visibility).all() or np.any(visibility<0) or np.any(visibility>1)):
        raise ValueError('available must be finite and in [0,1]')
    sources=_source_rows(swatches)
    geometries={} if geometries is None else geometries
    rows=[];skips=Counter();counts=Counter();tested=0
    ordered=sorted(enumerate(records),key=lambda item:(_selection_priority(_eligibility(item[1]))[0],item[0]))
    for index,record in ordered:
        class_name=record.get('class_name',record.get('shape_hint'))
        if class_name not in SUPPORTED_CLASSES:
            skips['unsupported_class']+=1
            continue
        candidate_index=record.get('candidate_index',index)
        sid=str(record.get('swatch_id',record.get('template','')))
        center=[record.get('aligned_x',record.get('x',record.get('cx'))),
                record.get('aligned_y',record.get('y',record.get('cy')))]
        reasons=_eligibility(record)
        priority,priority_reason=_selection_priority(reasons)
        row=dict(candidate_index=candidate_index,source_record_index=index,
            selection_priority=priority,selection_priority_reason=priority_reason,
            swatch_id=sid,class_name=class_name,center=center,
            decision=record.get('decision'),selected=bool(record.get('selected',False)),
            exclusion_reason=record.get('exclusion_reason'),original_scores=_old_scores(record),
            eligible=bool(reasons),eligibility_reasons=reasons,status='skipped',diagnostic=None)
        rows.append(row)
        if not reasons:
            row['skip_reason']='no_circle_square_ambiguity_or_conflict';skips[row['skip_reason']]+=1
            continue
        counts['eligible_records']+=1
        if tested>=max_candidates:
            row['skip_reason']='candidate_budget';skips[row['skip_reason']]+=1
            continue
        source=sources.get(sid,{})
        try:
            amplitude,amplitude_provenance=_source_amplitude(source)
            entry=geometries.get(candidate_index,geometries.get(str(candidate_index)))
            geom=dict(entry.get('geometry',entry)) if isinstance(entry,dict) else _saved_geometry(record)
        except (ValueError,TypeError,KeyError) as error:
            row.update(skip_reason='invalid_saved_source_geometry_or_photometry',error=str(error))
            skips[row['skip_reason']]+=1
            continue
        if geom is None:
            row['skip_reason']='missing_saved_source_geometry';skips[row['skip_reason']]+=1
            continue
        # An explicit sidecar amplitude is allowed only as source metadata;
        # the adapter never measures or guesses target brightness.
        if amplitude is None and isinstance(entry,dict) and entry.get('source_ink_level') is not None:
            amplitude,amplitude_provenance=_source_amplitude(entry)
            amplitude_provenance='source_geometry_sidecar:'+amplitude_provenance
        row['source_ink_level']=amplitude
        row['source_ink_level_provenance']=amplitude_provenance
        row['geometry']=_json_geometry(geom)
        row['geometry_provenance']=geom.get('provenance','explicit_source_geometry_sidecar')
        row['paired_inner_measurement']=geom.get('inner_radii') is not None
        if amplitude is None:
            row['skip_reason']='missing_source_photometry';skips[row['skip_reason']]+=1
            continue
        try:
            xy=np.asarray(geom['center'],float);radii=np.asarray(geom['radii'],float)
            if xy.shape!=(2,) or radii.shape!=(2,) or not np.isfinite(np.r_[xy,radii]).all() or np.any(radii<=0):
                raise ValueError('Invalid global source geometry')
            low=np.maximum(0,np.floor(xy-radii-5).astype(int))
            high=np.minimum([gray.shape[1],gray.shape[0]],np.ceil(xy+radii+6).astype(int))
            left,top=map(int,low);right,bottom=map(int,high)
            if right<=left or bottom<=top:
                row['skip_reason']='source_geometry_outside_image';skips[row['skip_reason']]+=1
                continue
            patch=gray[top:bottom,left:right]
            observed=(patch.astype(np.float32) if image_kind=='darkness_float' else (255-patch.astype(np.float32))/255.)
            local_geometry=dict(center=(xy-low).tolist(),radii=radii.tolist())
            if 'rotation_degrees' in geom:local_geometry['rotation_degrees']=geom['rotation_degrees']
            diagnostic=corner_evidence(observed,geometry=local_geometry,source_ink_level=amplitude,
                available=None if visibility is None else visibility[top:bottom,left:right],
                inner_radii=geom.get('inner_radii'),capture=capture)
            _global_coordinates(diagnostic,low)
        except (ValueError,TypeError,KeyError) as error:
            row.update(skip_reason='invalid_source_diagnostic_input',error=str(error))
            skips[row['skip_reason']]+=1
            continue
        tested+=1
        counts[diagnostic['status']]+=1
        row.update(status='measured',diagnostic=diagnostic,patch_box=[left,top,right,bottom],
            nuisance_policy='no_nuisance_inferred_from_residual_or_target',
            shadow_only=True,changes_selection=False)
    return dict(version=VERSION,status='completed',enabled=True,shadow_only=True,
        records_modified=False,active_points_modified=False,suppressed_points_modified=False,
        input_records=len(records),circle_square_records=len(rows),eligible_records=counts['eligible_records'],
        measured_records=tested,max_candidates=max_candidates,skipped=dict(skips),
        decisions={name:counts[name] for name in ('square_corner','circle_arc','uncertain')},
        image_policy='caller_supplied_working_raster_no_additional_preprocessing',
        coordinate_frame='global_working_image',rows=rows,
        selection_priority_policy=['explicit_raster_or_swatch_identity_conflict','circle_shape_low_margin',
            'raster_low_margin','geometry_abstention_or_conflict','generic_window_ambiguity_or_rejection'],
        budget_tie_break='original_record_order_without_mutating_input',
        seconds=float(time.perf_counter()-started))

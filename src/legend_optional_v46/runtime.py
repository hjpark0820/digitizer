"""Explicit v46 routing, exports and evidence for optional legends.

No legend rectangle means the user supplies the one-series assumption. This
module never auto-discovers a hidden legend or invokes v45 no-legend detection.
Native v45 legend helpers may supply colour keys for an explicitly selected ROI.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from . import VERSION


BACKEND = 'v46_legend_optional'


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items() if not str(k).startswith('_')}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def box(value, shape, *, inclusive=True):
    a=np.asarray(value,float)
    if a.shape!=(4,) or not np.isfinite(a).all() or not np.equal(a,np.round(a)).all():
        raise ValueError('Plot/legend ROI must contain four finite integer pixels')
    b=a.astype(int).tolist()
    if inclusive:b[2]+=1;b[3]+=1
    if not (0<=b[0]<b[2]<=shape[1] and 0<=b[1]<b[3]<=shape[0]):
        raise ValueError('Plot/legend ROI must be nonempty and inside the source image')
    return b


def pixel_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def probe_legend(image, plot, legend):
    """Only real observed line keys may replace the ordinary marker route."""
    from .native_legend import extract
    from color_legend_runtime_v46 import native_seeds
    from color_legend_hybrid_v46 import extract_hybrid_legend
    from color_marker_evidence import _swatch_model
    from type3_v46.runtime import _labels, reference_role
    native=extract(image,legend,plot)['diagnostics']
    grid=native.get('unified_grid') or native.get('raw_grid')
    palette,centers=native_seeds(grid,native.get('native_swatch_info',[]))
    models,reports,rejected=extract_hybrid_legend(image,legend,native_grid=grid,
        native_palette=palette,native_centres=centers)
    # Colour-body discovery can sample lettering between native swatch cells.
    # Disregard those unassociated models only when EVERY native cell has its
    # own independently observed plain line. Mixed true marker/line legends
    # retain the marker-bearing route rather than discarding a real swatch.
    line_rows=[r for r in rejected if r.get('status')=='line_only' and r.get('observed_line_box')]
    native_indices={int(r['native_index']) for r in line_rows}
    covered=bool(centers) and native_indices==set(range(len(centers)))
    associated=[m for m in reports if m.get('native_index') is not None or m.get('native_grid_center') is not None or m.get('prior_used')]
    excluded=[]
    if models and covered and not associated:
        excluded=[dict(id=m.get('swatch_id'),box=m.get('box'),reason='unassociated_colour_body_outside_native_line_cells') for m in reports]
        models=[]
    if models:
        return dict(kind='marker_legend',entries=[],reports=reports,rejected=rejected)
    entries=[]
    for r in sorted(rejected,key=lambda q:q.get('native_index',9999)):
        if r.get('status')!='line_only' or not r.get('observed_line_box'):continue
        idx=int(r['native_index']);rgb=list(map(int,r['native_palette_rgb']));b=list(map(int,r['observed_line_box']))
        entries.append(dict(id=f'color{idx+2:02d}',rgb=rgb,box=b,palette_source_box=b,
            colour_source='native_observed_line_legend',model=_swatch_model(image,b,rgb)))
    labels,error=_labels(image,legend,entries) if entries else ({},None)
    for i,e in enumerate(entries):
        e['label']=labels.get(f'color{i+2:02d}',e['id']);e['role']=reference_role(e['label'])
    return dict(kind='line_legend' if entries else 'unresolved_legend',entries=entries,
                reports=reports,rejected=rejected,label_ocr_error=error,excluded_non_native_models=excluded)


def _point(p, sid=None):
    q=deepcopy(p)
    q['x_px']=float(q.get('x_px',q.get('x',q.get('cx_px',q.get('cx')))))
    q['y_px']=float(q.get('y_px',q.get('y',q.get('cy_px',q.get('cy')))))
    q['series_id']=str(sid or q.get('series_id',q.get('swatch_id','S01')))
    # The single-series hybrid uses plot-local x/y plus source x_px/y_px.
    # Every public alias in the runtime/Step5 handoff must use SOURCE pixels;
    # otherwise the existing exporter prioritises a local x over source x_px.
    for axis in ('x','y'):
        source=q[axis+'_px']
        if axis in q and q[axis]!=source:q['detector_local_'+axis]=q[axis]
        q[axis]=source;q['c'+axis]=source
    return q


def detect(image, plot, legend=None, *, mode='color', series_mode='auto', probe=None):
    from .single import detect_single
    from .classifier import classify_series
    from color_marker_evidence import colour_evidence
    from type3_v46.pipeline import detect as detect_type3, spatial_valid
    started=perf_counter()
    diagnostic=dict(version=VERSION,detector_backend=BACKEND,mode=mode,plot_box=plot,legend_box=legend,
        no_legend=legend is None,single_series_assumption=legend is None,
        series_mode_requested=series_mode,legacy_no_legend_disabled=True,
        native_resolution=True,image_sha256=pixel_hash(image),image_shape=list(image.shape))
    single=None
    if legend is None:
        single=detect_single(image,plot,mode=mode)
        diagnostic['single_series_evidence']={k:v for k,v in single.items() if k not in ('fields','evidence','templates','valid','paths')}
        # Preserve successful observed templates and their exact points. This
        # fallback is for unavailable palette/template evidence, not a bypass
        # for a template whose marker candidates failed window verification.
        if series_mode!='line-only' and not single.get('points') and not single.get('templates'):
            from .common_ink import detect_common_ink
            common=detect_common_ink(image,plot)
            diagnostic['common_ink_fallback']={k:v for k,v in common.items()
                if not k.startswith('_') and k not in ('templates','template','points','series')}
            diagnostic['common_ink_fallback']['trigger']=single.get('status','no_observed_template')
            if common['status']=='completed':
                correction_reason=(
                    'Common-ink markers have no validated colour-path reference. '
                    'Point editing/export remain available; colour Step 5 is withheld '
                    'instead of tracing axes or inventing a reference from marker chords.')
                diagnostic.update(status='completed',chart_kind='markers',route='no_legend_common_ink_disk',
                    reason='Observed-template extraction was unavailable; recurrent compact bodies passed local common-ink disk verification',
                    series=common['series'],points=[_point(p,'S01') for p in common['points']],
                    suppressed_points=[_point(p,'S01') for p in common.get('suppressed_points',[])],
                    paths=[],templates=common['templates'],candidates=common.get('candidates',[]),
                    reference_path_policy='withhold_unvalidated_common_ink_path',
                    correction_available=mode=='bw',
                    correction_unavailable_reason=correction_reason if mode=='color' else None,
                    _fields=common['_fields'],seconds=perf_counter()-started)
                return diagnostic
        palette=single.get('palette') or {}
        model=palette.get('model')
        if model is None:
            return dict(diagnostic,status='uncertain',chart_kind='uncertain',reason=single.get('reason') or palette.get('reason') or single.get('status','unresolved_palette'),
                        series=[],points=[],suppressed_points=[],paths=[],seconds=perf_counter()-started)
        model=deepcopy(model)
        for key in ('bgr','paper_bgr'):model[key]=np.asarray(model[key],np.float32)
        entries=[dict(id='S01',label='Series 1',rgb=palette.get('rgb') or single['series'][0]['rgb'],model=model,
                      role='data_series',colour_source='plot_observed',palette_source_box=palette.get('source_box'))]
        valid=np.asarray(single.get('valid',spatial_valid(plot)),bool)
        if single.get('points') and series_mode!='line-only':
            diagnostic.update(chart_kind='markers',status='completed',route='no_legend_observed_template',
                reason='Repeated source marker template and full-window verification support one series',
                series=single['series'],points=[_point(p,'S01') for p in single['points']],
                suppressed_points=[_point(p,'S01') for p in single.get('suppressed_points',single.get('uncertain_points',[]))],
                paths=single.get('paths',[]),templates=single.get('templates',[]))
            fields=colour_evidence(image[plot[1]:plot[3],plot[0]:plot[2]],entries)
            fields['valid']=valid
            diagnostic['_fields']=fields
            diagnostic['seconds']=perf_counter()-started
            return diagnostic
    else:
        probe=probe or probe_legend(image,plot,legend)
        diagnostic['legend_probe']={k:v for k,v in probe.items() if k!='entries'}
        entries=[e for e in probe['entries'] if e.get('role','data_series')!='reference_line']
        if not entries:
            return dict(diagnostic,status='uncertain',chart_kind='uncertain',reason='Selected legend contains no usable data-series colour lines',series=[],points=[],suppressed_points=[],paths=[])
        valid=spatial_valid(plot,legend)
    classified=classify_series(image,plot,entries,valid=valid)
    diagnostic['classification']=classified.get('audit',{})
    fields=classified['fields'];diagnostic['_fields']=fields
    chosen=classified['chart_kind']
    if series_mode=='line-only':chosen='line-only'
    elif series_mode=='markers' and chosen!='markers':chosen='uncertain'
    diagnostic.update(chart_kind=chosen,series=classified.get('series',entries),
        paths=classified.get('paths',[]),points=[],suppressed_points=[],templates=[])
    if chosen=='line-only':
        rows=[];own={};soft={}
        for i,s in enumerate(classified.get('series',entries)):
            rows.append(dict(s,role='data_series',marker_state='absent_by_source_line_evidence' if series_mode=='auto' else 'absent_by_user_selection'))
            own[s['id']]=(np.asarray(fields['membership'][i])>=.18)&valid
            soft[s['id']]=np.asarray(fields['membership'][i])*valid
        typed,maps=detect_type3(image,plot,legend,rows,own,soft,valid=valid)
        typed.update(image_sha256=pixel_hash(image),image_shape=list(image.shape),detector_backend='v46_type3_v1')
        diagnostic.update(status='completed',route='source_line_only_type3',reason='Source thin-curve evidence selected the structural type3 detector',
            series=typed['series'],points=typed['points'],suppressed_points=typed.get('tentative_points',[]),paths=typed['paths'],
            _type3=typed,_type3_maps=maps)
        if not diagnostic['points']:
            diagnostic['reason']+='; no active centres, only path/endpoint hypotheses' if diagnostic['suppressed_points'] else '; no usable measurement hypotheses'
            if not diagnostic['suppressed_points']:diagnostic['status']='uncertain'
    elif chosen=='markers':
        points=classified.get('points',[])
        suppressed=classified.get('suppressed',classified.get('suppressed_points',[]))
        radius=float(classified.get('audit',{}).get('radius',classified.get('radius')) or 0.)
        if not radius:
            radius=next((float(p.get('radius',0.)) for p in points if p.get('radius')),0.)
        if radius<=0:raise ValueError('Disk route returned markers without a measured radius')
        extent=int(np.ceil(radius));gy,gx=np.mgrid[-extent:extent+1,-extent:extent+1]
        alpha=(gx*gx+gy*gy<=radius*radius).astype(np.float32)
        templates=[dict(id=s['id'],soft=alpha,diameter=2*radius,center=[extent,extent],
            provenance=dict(kind='shared_disk_hypothesis',radius_source='recurrent_source_bodies',not_observed_clean_template=True),
            composition=dict(best_model_name='circle')) for s in diagnostic['series']]
        diagnostic.update(status='completed',route='path_disk_density',reason='Repeated source bodies support a shared disk, evaluated by own/total density',
            points=[_point(p) for p in points],suppressed_points=[_point(p) for p in suppressed],
            templates=templates,candidates=classified.get('candidates',[]))
    else:
        diagnostic.update(status='uncertain',route='abstain',reason=classified.get('reason') or classified.get('audit',{}).get('reason') or
            'Source pixels do not confidently distinguish markers from line/error-bar structures')
    diagnostic['seconds']=perf_counter()-started
    return diagnostic


def calibration(plot, x_range=None, y_range=None, x_log=False, y_log=False):
    def axis(p0,p1,bounds,log):
        if bounds is None:
            return dict(p0=float(p0),p1=float(p1),v0=0.,v1=float(abs(p1-p0)),log=False,kind='pixel',source='uncalibrated_pixel_distance')
        lo,hi=map(float,bounds)
        if not np.isfinite([lo,hi]).all() or lo==hi or (log and min(lo,hi)<=0):
            raise ValueError('Axis limits must be finite, distinct and positive for log axes')
        return dict(p0=float(p0),p1=float(p1),v0=lo,v1=hi,log=bool(log),kind='log' if log else 'linear',source='user_supplied')
    return dict(x=axis(plot[0],plot[2]-1,x_range,x_log),y=axis(plot[3]-1,plot[1],y_range,y_log))


def edit_data(image, result, cal):
    curves=[]
    for s in result.get('series',[]):
        if s.get('role')=='reference_line':continue
        sid=s['id'];rows=[p for p in result.get('points',[]) if p.get('series_id')==sid]
        ordered=sorted(rows,key=lambda q:(q['x_px'],q['y_px']))
        if result['chart_kind']=='line-only':
            from type3_v46.gui_adapter import gui_point
            points=[gui_point(p) for p in ordered]
        else:
            points=[dict(x=float(p['x_px']),y=float(p['y_px']),candidate_id=p.get('id',p.get('candidate_id')),
                         original_L0=p.get('original_L0',p.get('original_detection',True)),
                         existence=p.get('existence','image_supported'),kind=p.get('kind','marker_candidate'))
                    for p in ordered]
        curve=dict(name=sid,label=s.get('label') or 'Series 1',rgb=plain(s['rgb']),points=points,
                   series_mode='line-only' if result['chart_kind']=='line-only' else 'markers',
                   marker_glyph_detected=result['chart_kind']=='markers')
        if result['chart_kind']=='line-only':curve.update(marker_class=None)
        curves.append(curve)
    plot=result['plot_box']
    return dict(image=dict(width=image.shape[1],height=image.shape[0]),plot_area=[plot[0],plot[1],plot[2]-1,plot[3]-1],
        calibration=cal,curves=curves,detector_backend=BACKEND,series_mode='line-only' if result['chart_kind']=='line-only' else 'markers',
        legend_optional=dict(status=result['status'],reason=result['reason'],route=result.get('route'),no_legend=result['no_legend']),
        correction_available=result['status']=='completed' and result.get('correction_available',True),
        correction_unavailable_reason=result.get('correction_unavailable_reason'))


def overlay(image, result):
    canvas=image.copy()
    for rows,suppressed in ((result.get('points',[]),False),(result.get('suppressed_points',[]),True)):
        for p in rows:
            x,y=round(p['x_px']),round(p['y_px']);r=max(4,min(9,round(float(p.get('radius',6)))))
            color=(210,35,180) if suppressed else (30,170,25)
            if suppressed:
                poly=np.array([[x,y-r],[x+r,y],[x,y+r],[x-r,y]],np.int32)
                cv2.polylines(canvas,[poly],True,(255,255,255),4,cv2.LINE_AA);cv2.polylines(canvas,[poly],True,color,2,cv2.LINE_AA)
            else:
                cv2.circle(canvas,(x,y),r,(255,255,255),4,cv2.LINE_AA);cv2.circle(canvas,(x,y),r,color,2,cv2.LINE_AA)
    return canvas


def _templates(result):
    values=result.get('templates') or []
    if isinstance(values,dict):return values
    return {t.get('id','S01'):t for t in values}


def export_marker_payload(destination,image,result):
    from color_step5_export_v46 import export_step5_inputs
    from color_path_directional_v46 import trace_directional
    fields=result['_fields'];valid=np.asarray(fields['valid'],bool);plot=result['plot_box'];legend=result['legend_box']
    series=result['series'];names=[s['id'] for s in series];paths=[];own={}
    for i,s in enumerate(series):
        sid=s['id'];field=np.asarray(fields['membership'][i],np.float32)*valid
        mask=(field>=.18)&valid;own[sid]=mask
        xs=np.flatnonzero(mask.any(axis=0))
        # All-ink paths repeatedly followed axes in the source experiments.
        # Preserve masks and measured marker proposals, but do not turn those
        # unvalidated paths (or marker-joining chords) into Step-5 reference data.
        if len(xs)>1 and result.get('reference_path_policy')!='withhold_unvalidated_common_ink_path':
            tr=trace_directional(field,int(xs[0]),int(xs[-1]),0,field.shape[0]-1)
            xy=tr['path'];obs=tr['val']>=.18
            paths.append(dict(series_id=sid,path=xy+[plot[0],plot[1]],val=tr['val'],observed=obs,
                filled=np.ones(len(obs),bool),valid=True))
    active={n:sorted([dict(p,x_px=p['x_px'],y_px=p['y_px']) for p in result['points'] if p['series_id']==n],
                     key=lambda p:(p['x_px'],p['y_px'])) for n in names}
    # Density-only occlusion alternatives remain visible in the source diagnostic
    # but are not relabelled as the pre-existing strong-shape Step5 evidence.
    tentative=dict(version=VERSION,paths=paths,candidates=result.get('suppressed_points',[]))
    from color_estimated_path_v46 import path_to_segments
    payload=export_step5_inputs(destination,image,plot,legend,series,names,_templates(result),active,tentative,own,path_to_segments)
    if result.get('correction_available') is False:
        payload.update(correction_available=False,
            correction_unavailable_reason=result.get('correction_unavailable_reason'),
            reference_path_policy=result.get('reference_path_policy'))
        write(Path(destination)/'step5_inputs.json',payload)
    return payload


def run_file(image_path,out_dir,plot_area,legend_area=None,*,mode='color',series_mode='auto',x_range=None,y_range=None,x_log=False,y_log=False,probe=None):
    image=cv2.imread(str(image_path))
    if image is None:raise ValueError('Cannot read source image')
    plot=box(plot_area,image.shape);legend=box(legend_area,image.shape) if legend_area is not None else None
    result=detect(image,plot,legend,mode=mode,series_mode=series_mode,probe=probe)
    save_result(image_path,out_dir,result,x_range=x_range,y_range=y_range,x_log=x_log,y_log=y_log)
    return result


def save_result(image_path,out_dir,result,*,x_range=None,y_range=None,x_log=False,y_log=False):
    """Export a fresh in-memory detection without rerunning the detector."""
    image=cv2.imread(str(image_path))
    if image is None or pixel_hash(image)!=result['image_sha256']:
        raise ValueError('Source image changed since optional-legend detection')
    plot=result['plot_box'];mode=result['mode']
    result.update(source_image_path=str(Path(image_path).resolve()),source_sha256=hashlib.sha256(Path(image_path).read_bytes()).hexdigest())
    destination=Path(out_dir);destination.mkdir(parents=True,exist_ok=True)
    cal=calibration(plot,x_range,y_range,x_log,y_log)
    result['calibration_warning']='Axis ranges not supplied: uncalibrated pixel distances, not recovered data units' if x_range is None or y_range is None else None
    if result.get('_type3') is not None:
        typed=result['_type3'];typed.update(source_image_path=result['source_image_path'],source_sha256=result['source_sha256'])
        write(destination/'type3_detection_v46.json',typed)
    elif result['status']=='completed' and mode=='color':
        payload=export_marker_payload(destination,image,result)
        result['step5_suppressed_admission']=payload['counts']
    write(destination/'legend_optional_v46.json',result)
    write(destination/'edit_data.json',edit_data(image,result,cal))
    cv2.imwrite(str(destination/'data_points_overlay.png'),overlay(image,result))
    print(f"[v46 optional legend] {result['status']}: {result.get('route','unresolved')}; "
          f"{len(result.get('points',[]))} active / {len(result.get('suppressed_points',[]))} suppressed. {result['reason']}",flush=True)
    return result


def bw_points(result):
    """Adapt observed or explicitly analytic glyphs to the BW SSIM pool.

    A plot-mined template is not an actual legend. Its centre/mask/provenance
    remain attached to every hypothesis; no type3 centre is rendered as a glyph.
    """
    from bw_suppressed_v46 import encode_marker_mask
    if result['chart_kind']!='markers':return [],[],0.
    templates=_templates(result);active=[];suppressed=[];seen=set();diameters=[]
    for rows,target,is_active in ((result.get('points',[]),active,True),
                                  (result.get('suppressed_points',[]),suppressed,False)):
        for i,p in enumerate(rows):
            sid=p['series_id'];t=templates.get(sid)
            if t is None:raise ValueError('BW correction requires a declared marker template for each series')
            alpha=np.asarray(t['soft'],float);mask=alpha>=.35
            if not mask.any():raise ValueError('Empty BW marker template')
            x,y=float(p['x_px']),float(p['y_px']);key=(sid,round(x,3),round(y,3))
            if key in seen:continue
            seen.add(key)
            center=t.get('center',[(mask.shape[1]-1)/2,(mask.shape[0]-1)/2]);diameter=float(t['diameter'])
            diameters.append(diameter)
            point_id=str(p.get('id') or p.get('candidate_id') or f'{sid}_{"P" if is_active else "S"}{i:04d}')
            q=dict(p,swatch_id=sid,class_name=p.get('class_name') or 'observed_marker',
                cx=x,cy=y,cx_px=x,cy_px=y,source_diameter=diameter,marker_mask=encode_marker_mask(mask),
                marker_offset_x=float(center[0]-mask.shape[1]//2),marker_offset_y=float(center[1]-mask.shape[0]//2),
                marker_provenance=t.get('provenance',{}),point_id=point_id,candidate_id=point_id,
                original_detection=p.get('original_detection',is_active),state='active' if is_active else 'suppressed')
            target.append(q)
    return active,suppressed,float(np.median(diameters)) if diameters else 0.


def finish_bw(image_path,out_dir,result,*,correct=False,iterations=5,previous_path=None):
    """Save resumable BW state or execute the existing typed correction engine.

    Returns a result copy with current points/suppressed_points, correction
    status and output directory. Original source hypotheses stay in the saved
    legend_optional_v46.json, independently of Step5 actions.
    """
    destination=Path(out_dir);image=cv2.imread(str(image_path))
    if image is None or pixel_hash(image)!=result['image_sha256']:raise ValueError('BW source changed')
    updated=deepcopy(result)
    status=dict(requested=bool(correct),status='not_requested',engine='bw_series_correction_v46')
    try:
        if result['status']!='completed':raise ValueError('Uncertain optional-legend detection cannot be corrected')
        if result['chart_kind']=='line-only':
            status['engine']='v46_type3_path_v1'
            if correct:
                from type3_v46.gui_adapter import correct_saved_detection
                plot=result['plot_box'];legend=result.get('legend_box')
                summary=correct_saved_detection(image_path,out_dir,iterations=iterations,previous_path=previous_path,
                    plot_area=[plot[0],plot[1],plot[2]-1,plot[3]-1],
                    legend_box=[legend[0],legend[1],legend[2]-1,legend[3]-1] if legend else None)
                updated.update(points=[p for s in summary['series'] for p in s['final_points']],
                    suppressed_points=[p for s in summary['series'] for p in s['final_suppressed']])
                status.update(status='completed',active_count=len(updated['points']))
        else:
            active,suppressed,diameter=bw_points(result)
            identity=dict(version=VERSION,image_sha256=result['image_sha256'],plot_box=result['plot_box'],
                legend_box=result.get('legend_box'),templates=plain(_templates(result)))
            state=dict(backend=BACKEND,identity=identity,P_current=active,S_current=suppressed,runtime_state=None)
            if previous_path:
                state=json.loads(Path(previous_path).read_text(encoding='utf-8'))
                if state.get('backend')!=BACKEND or state.get('identity')!=identity:
                    raise ValueError('Previous optional BW state belongs to a different source/ROI/template')
                active,suppressed=state['P_current'],state['S_current']
            if correct:
                from bw_series_correction_v46 import run_correction
                output=run_correction(image,destination,init_points=active,init_suppressed=suppressed,
                    plot_area=result['plot_box'],legend_box=result.get('legend_box'),d_est=diameter,
                    max_iter=iterations,init_state=state.get('runtime_state'),no_legend=result.get('no_legend',False),
                    connection_alignment=True,structural_edits=True)
                active,suppressed=output['P_current'],output['S_current']
                state.update(P_current=active,S_current=suppressed,runtime_state=output['runtime_state'])
                updated.update(points=[dict(p,series_id=p['swatch_id'],x_px=p['cx'],y_px=p['cy']) for p in active],
                    suppressed_points=[dict(p,series_id=p['swatch_id'],x_px=p['cx'],y_px=p['cy']) for p in suppressed])
                edited=json.loads((destination/'edit_data.json').read_text(encoding='utf-8'))
                curves=edit_data(image,updated,edited['calibration'])['curves'];edited['curves']=curves
                edited['correction']=dict(backend='bw_series_correction_v46',metric='structure_routed',iterations_executed=len(output['trace']))
                write(destination/'edit_data.json',edited)
                cv2.imwrite(str(destination/'data_points_overlay.png'),overlay(image,updated))
                status.update(status='completed',active_count=len(active),suppressed_count=len(suppressed),
                    artifact_dir=output['artifact_dir'],iterations_executed=len(output['trace']))
            write(destination/'correction_state.json',state)
    except Exception as error:
        status.update(status='failed',error=str(error));write(destination/'correction_status.json',status)
        raise
    write(destination/'correction_status.json',status)
    updated.update(correction_status=status,output_dir=str(destination))
    return updated

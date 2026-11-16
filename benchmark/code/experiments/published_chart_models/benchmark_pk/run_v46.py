"""Ten-panel conditional benchmark of the unchanged production v46 CLIs.

Reads only inputs.json and its raster images, never evaluation point references.
Run panels sequentially so their wall times do not compete with one another.
BW has native explicit swatch inputs; colour retains its native legend locator.
Actual observed legend crops are exported for the other benchmark methods.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / 'src'
DEFAULT_INPUTS = HERE / 'inputs.json'
DEFAULT_OUTPUT = HERE / 'results' / 'run_v1' / 'v46'
FORBIDDEN_KEYS = {'ground_truth', 'groundtruth', 'target_points', 'reference_points',
                  'points', 'points_px', 'targets', 'evaluation_path', 'reference_path'}
SOURCE_FILES = ('run_A4_auto_v46.py', 'run_A4_color_v46.py', 'run_A4_auto_v45.py',
    'bw_detect_cli.py', 'bw_pipeline.py', 'bw_pipeline_v46.py',
    'bw_legend_v46.py', 'bw_suppressed_v46.py', 'bw_step5_v46.py',
    'partial_swatch_detector.py', 'occlusion_aware_window_verifier.py',
    'color_legend_runtime_v46.py', 'color_legend_hybrid_v46.py',
    'color_legend_composition_v46.py', 'legend_composition_v46.py',
    'color_marker_runtime_v46.py', 'color_marker_hybrid_v2.py',
    'color_marker_evidence_v2.py', 'color_tentative_v46.py',
    'color_step5_export_v46.py', 'color_correct_cli.py', 'color_step5_payload_v46.py',
    'color_path_correction_v46.py', 'color_path_metrics_v46.py')
ALGORITHM_ENV = ('MARKER_DIAM', 'EXTRACT_MODE', 'STEM_LEVEL', 'AA_TERM',
    'SPIKE_RECOVERY', 'BW_V46_REFINEMENT_BACKEND', 'BW_V46_GRID_BACKEND',
    'BW_V46_WINDOW_BACKEND')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_targets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(f'Inference input contains forbidden evaluation field: {key}')
            reject_targets(item)
    elif isinstance(value, list):
        for item in value:
            reject_targets(item)


def box(value, size):
    if len(value) != 4 or any(not math.isfinite(float(v)) or int(v) != v for v in value):
        raise ValueError('Boxes must have four integral finite half-open coordinates')
    a,b,c,d = map(int, value)
    if not (0 <= a < c <= size[0] and 0 <= b < d <= size[1]):
        raise ValueError(f'Box outside native image: {value} / {size}')
    return [a,b,c,d]


def validate_inputs(data):
    reject_targets(data)
    panels = data['panels']
    if len(panels) != 10 or Counter(p['modality'] for p in panels) != {'bw':5, 'color':5}:
        raise ValueError('The requested benchmark is exactly five BW and five colour panels')
    ids = [p['panel_id'] for p in panels]
    if len(set(ids)) != 10 or any(not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in ids):
        raise ValueError('Panel IDs must be unique safe path components')
    for panel in panels:
        size = panel['image_size']
        crop = box(panel['crop_box_original'], panel['source_image_size'])
        if size != [crop[2]-crop[0], crop[3]-crop[1]]:
            raise ValueError('Native crop dimensions do not match translation-only geometry')
        box(panel['plot_bbox'],size)
        box(panel['legend_bbox'],size)
        sids = [s['series_id'] for s in panel['swatches']]
        if not sids or len(sids) != len(set(sids)):
            raise ValueError('Legend series identities must be nonempty and unique')
        for swatch in panel['swatches']:
            box(swatch['bbox'],size)
        for name in ('image_path','source_image_path','input_sha256','source_sha256'):
            if not panel.get(name):
                raise ValueError(f'Missing image binding: {name}')
    return panels


def native_image(panel):
    """Verify actual pixels and the once-only original->crop translation."""
    import cv2
    import numpy as np
    if sha(panel['image_path']) != panel['input_sha256']:
        raise ValueError('Submitted crop file hash changed')
    if sha(panel['source_image_path']) != panel['source_sha256']:
        raise ValueError('Original image file hash changed')
    image = cv2.imread(panel['image_path'])
    original = cv2.imread(panel['source_image_path'])
    if image is None or original is None:
        raise ValueError('Unreadable input image')
    if list(image.shape[1::-1]) != panel['image_size'] or list(original.shape[1::-1]) != panel['source_image_size']:
        raise ValueError('Native image dimensions changed')
    a,b,c,d = panel['crop_box_original']
    if not np.array_equal(image, original[b:d,a:c]):
        raise ValueError('Input crop is not an exact unresized original-image pixel crop')
    return image


def inclusive(value):
    return [int(value[0]),int(value[1]),int(value[2])-1,int(value[3])-1]


def csv_box(value):
    return ','.join(str(v) for v in value)


def detection_command(panel, destination, python):
    args = [str(python), str(SRC/'run_A4_auto_v46.py'), panel['image_path'], str(destination),
            '--mode',panel['modality']]
    if panel['modality'] == 'bw':
        # BW native public coordinates are half-open; colour v45 CLI is inclusive.
        args += ['--plot-area',csv_box(panel['plot_bbox']), '--legend-area',csv_box(panel['legend_bbox']),
                 '--swatches-json',json.dumps([[None,s['bbox']] for s in panel['swatches']]),
                 '--skip-segments']
    else:
        args += ['--plot-area',csv_box(inclusive(panel['plot_bbox'])),
                 '--legend-box',csv_box(inclusive(panel['legend_bbox']))]
    return args


def correction_command(panel, before, after, python):
    if panel['modality'] == 'bw':
        return detection_command(panel,after,python)+[
            '--prev-state',str(before/'correction_state.json'),'--correct','--correct-iters','5']
    return [str(python),str(SRC/'color_correct_cli.py'),panel['image_path'],str(after),
            '--require-v46-path','--correct-iters','5']


def iou(a,b):
    area = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0.


def match_legends(entries, annotations):
    """One-to-one native legend-box IoU assignment; no chart-point information."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    matrix = np.array([[iou(e['bbox'],a['bbox']) for a in annotations] for e in entries],dtype=float)
    selected = {}
    if entries and annotations:
        rows,cols = linear_sum_assignment(-matrix)
        selected = {int(i):int(j) for i,j in zip(rows,cols) if matrix[i,j] > 0}
    pairs = []
    for i,entry in enumerate(entries):
        for j,annotation in enumerate(annotations):
            accepted = selected.get(i) == j
            pairs.append(dict(native_id=entry['native_id'],series_id=annotation['series_id'],
                iou=float(matrix[i,j]),accepted=accepted,
                reason='maximum_total_iou_assignment' if accepted else
                       'zero_bbox_overlap' if matrix[i,j] == 0 else 'other_one_to_one_assignment'))
    mapping = {entry['native_id']:(annotations[selected[i]]['series_id'] if i in selected
                else f"unmapped_v46_{entry['native_id']}") for i,entry in enumerate(entries)}
    return mapping, dict(policy='maximum-total-IoU one-to-one observed legend boxes; strictly positive overlap; no point references',
                         pairs=pairs,unmatched_annotations=[a['series_id'] for j,a in enumerate(annotations) if j not in selected.values()],
                         unmatched_native=[e['native_id'] for i,e in enumerate(entries) if i not in selected])


def actual_entries(panel,before):
    if panel['modality'] == 'bw':
        diag = read(before/'v46_grid_diagnostics.json')
        return [dict(native_id=e['swatch_id'],bbox=e['box'],native_body_bbox=e.get('glyph_box'),
                     class_name=e.get('class_name'),crop_source='v46_grid_diagnostics.swatches.box')
                for e in diag.get('swatches',[]) if not e.get('excluded_by_class_filter')]
    path = before/'legend_hybrid_v46.json'
    if not path.exists():
        return []
    diag = read(path)
    reports = {r['swatch_id']:r for r in diag.get('hybrid_reports',[]) if r.get('swatch_id')}
    output = []
    for i,entry in enumerate(diag.get('entries',[])):
        composition = entry.get('composition') or {}
        report = reports.get(entry.get('id'),{})
        rectangle = composition.get('source_swatch_box') or report.get('swatch_box') or entry['box']
        names = diag.get('names') or []
        name = names[i] if i < len(names) else f'color{i+2:02d}'
        output.append(dict(native_id=name,native_swatch_id=entry.get('id'),bbox=rectangle,
            native_body_bbox=entry['box'],class_name=report.get('shape_hint'),rgb=entry.get('rgb'),
            crop_source='composition.source_swatch_box' if composition.get('source_swatch_box') else
                        'hybrid_report.swatch_box' if report.get('swatch_box') else 'hybrid_entry.box'))
    return output


def export_legends(panel,before,destination,image):
    import cv2
    tick = perf_counter()
    entries = actual_entries(panel,before)
    mapping,audit = match_legends(entries,panel['swatches'])
    labels = {a['series_id']:a.get('label',a['series_id']) for a in panel['swatches']}
    offset = panel['crop_box_original'][:2]
    for index,entry in enumerate(entries):
        a,b,c,d = box(entry['bbox'],panel['image_size'])
        path = destination/'shared_legend_crops'/f'legend_{index+1:02d}.png'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),image[b:d,a:c]):
            raise OSError('Could not save observed legend crop')
        entry.update(series_id=mapping[entry['native_id']],label=labels.get(mapping[entry['native_id']],mapping[entry['native_id']]),
            image_path=str(path.resolve()),image_size=[c-a,d-b],sha256=sha(path),
            source_native_legend_id=entry['native_id'],crop_path=str(path.resolve()),bbox_crop=[a,b,c,d],
            bbox_original=[a+offset[0],b+offset[1],c+offset[0],d+offset[1]],
            observed_rgb_pixels=True,connector_pixels_removed=False,template_rendered=False)
    value = dict(panel_id=panel['panel_id'],modality=panel['modality'],entries=entries,mapping=mapping,
        legend_mapping_audit=audit,annotations=panel['swatches'],input_sha256=panel['input_sha256'],
        source_sha256=panel['source_sha256'],coordinate_convention='bbox crop-local half-open; bbox_original full original half-open',
        status='available' if entries else 'unavailable',
        protocol='Shared actual v46 observed legend crops; not an independent legend-localization comparison.',
        export_seconds=perf_counter()-tick)
    save(destination/'shared_legends.json',value)
    return value


def predictions(edit_data,panel,mapping,method):
    offset = panel['crop_box_original'][:2]
    result=[]
    for curve in edit_data.get('curves',[]):
        native=str(curve.get('swatch_id') or curve.get('name'))
        series=mapping.get(native,f'unmapped_v46_{native}')
        for index,p in enumerate(curve.get('points',[]),1):
            x,y=float(p['x']),float(p['y'])
            if not all(map(math.isfinite,(x,y))):
                raise ValueError('Production returned nonfinite points')
            result.append(dict(point_id=f'{method}_{native}_{index:04d}',
                x_px=x+offset[0],y_px=y+offset[1],series_id=series,
                score=p.get('confidence',p.get('score')),native_series_id=native,
                crop_x_px=x,crop_y_px=y))
    return result


def publish_prediction(folder,panel,method,status,rows,reason=''):
    record=dict(panel_id=panel['panel_id'],model=method,method=method,status=status,reason=reason,
        predictions=rows,coordinate_convention='full original image pixel centers; crop translation added exactly once',
        score_note='Native export may omit confidence; null is unavailable, not zero. No score threshold added.',
        input_sha256=panel['input_sha256'],source_sha256=panel['source_sha256'])
    save(folder/f'{method}.json',record)
    return record


def timed_cli(argv,timing_path):
    """Time a transparent wrapper around the real colour legend method only.

    The original callable receives identical arguments and its exact return value
    is returned. No image, marker, setting or decision is replaced.
    """
    sys.path.insert(0,str(SRC))
    timing=dict(shared_legend_seconds=None,legend_calls=0,module_preload_seconds=0.,
        wrapper='perf_counter around original LegendRuntime.prepare; unchanged args and return value')
    if '--mode' in argv and argv[argv.index('--mode')+1]=='color':
        tick=perf_counter()
        from color_legend_runtime_v46 import LegendRuntime
        timing['module_preload_seconds']=perf_counter()-tick
        original=LegendRuntime.prepare
        def measured(self,*args,**kwargs):
            started=perf_counter()
            try:
                return original(self,*args,**kwargs)
            finally:
                timing['shared_legend_seconds']=(timing['shared_legend_seconds'] or 0.)+perf_counter()-started
                timing['legend_calls']+=1
        LegendRuntime.prepare=measured
    previous=sys.argv
    started=perf_counter()
    try:
        sys.argv=list(argv)
        runpy.run_path(argv[0],run_name='__main__')
    finally:
        sys.argv=previous
        timing['native_cli_seconds']=perf_counter()-started
        save(timing_path,timing)


def launch(command,folder,phase,env,timeout):
    tick=perf_counter()
    actual_command=[command[0],str(Path(__file__).resolve()),'--timed-cli',
                    '--timing-path',str(folder/f'{phase}_stage_timing.json'),'--',*command[1:]]
    with (folder/f'{phase}.log').open('w',encoding='utf-8') as log:
        try:
            p=subprocess.run(actual_command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            result=dict(exit_code=p.returncode,timed_out=False,wall_seconds=perf_counter()-tick)
        except subprocess.TimeoutExpired:
            result=dict(exit_code=None,timed_out=True,wall_seconds=perf_counter()-tick)
    result.update(command=command,timing_wrapper_command=actual_command,log_path=str(folder/f'{phase}.log'))
    save(folder/f'{phase}_process.json',result)
    return result


def correction_result(panel,after):
    if panel['modality']=='bw':
        status=read(after/'correction_status.json')
        state=read(after/'correction_state.json')
        runtime=state.get('bw_step5_state') or {}
        if status.get('status')!='completed' or not runtime:
            raise ValueError('BW production did not confirm a successful correction')
        iterations={'all_series':int(runtime['total_iterations'])}
        statuses={'all_series':status['status']}
    else:
        state=read(after/'color_correction_state.json')
        if state.get('backend')!='v46_path_shape_v3':
            raise ValueError('Colour correction did not use the current production path engine')
        iterations={k:int(v['iteration_count']) for k,v in state.get('path_state',{}).items()}
        statuses=state.get('status',{})
    if any(n>5 or n<0 for n in iterations.values()):
        raise ValueError('Step 5 exceeded the requested five-iteration cap')
    return dict(iterations_by_series=iterations,status_by_series=statuses,requested_iterations=5,
        status='completed' if statuses and all(s in ('completed','corrected') for s in statuses.values()) else 'completed_partial')


def native_timing(panel,before,after):
    result={}
    for phase,directory in [('detection',before),('correction',after)]:
        if (directory/'pipeline_timing.json').exists():
            result[phase]=read(directory/'pipeline_timing.json')
    detection=result.get('detection',{})
    legend=(detection.get('point_pipeline_timings') or {}).get('legend_seconds')
    measured_path=before.parent/'detection_stage_timing.json'
    measured=read(measured_path) if measured_path.exists() else {}
    if legend is None:
        legend=measured.get('shared_legend_seconds')
    result['shared_legend_seconds']=legend
    result['shared_legend_timing_note']='BW native timer or transparent colour LegendRuntime.prepare timer; included in detection process wall time.' if legend is not None else 'No complete native legend timing available; not estimated.'
    result['model_load_seconds']=detection.get('pipeline_imports_seconds')
    result['model_load_timing_note']='BW native imports/backend initialization, included in process wall time; colour unavailable, not zero.'
    result['timing_wrapper']=measured
    if (before/'color_marker_hybrid_v46.json').exists():
        diag=read(before/'color_marker_hybrid_v46.json')
        result['color_marker_stage_seconds']=diag.get('seconds')
        result['color_marker_analysis_timing']=diag.get('timing')
    return result


def run_panel(panel,output,python,timeout,env):
    tick=perf_counter();folder=output/panel['panel_id']
    if folder.exists():
        raise FileExistsError(f'Refusing to reuse potentially stale results: {folder}')
    folder.mkdir(parents=True);before=folder/'raw_before';after=folder/'raw_after'
    before.mkdir();after.mkdir()
    job=dict(panel_id=panel['panel_id'],modality=panel['modality'],status='running',
        input=panel,step5_max_iterations=5,raw_before=str(before),raw_after=str(after),timing={},
        process_policy='sequential fresh CLI processes; BW correction resumes and skips point detection',
        default_settings=dict(bw_backend='grid_v46',bw_grid_fraction=.5,bw_grid_overlap=.5,
            bw_compute='auto',bw_source_scale=1.,bw_errorbars=False,bw_skip_segments=True,
            color_native_resolution=True,color_scale_search=False,color_selection_threshold=.25,
            color_correction_metric='chamfer',color_correction_model='pchip',
            color_correction_reference='estimated_path',color_inferred_weight=.25))
    save(folder/'job.json',job)
    phase='detection';before_record=None;after_record=None;image=None
    try:
        loaded=perf_counter();image=native_image(panel);job['timing']['input_verification_seconds']=perf_counter()-loaded
        process=launch(detection_command(panel,before,python),folder,'detection',env,timeout)
        job['timing']['detection_process_seconds']=process['wall_seconds'];job['detection_process']=process
        if process['timed_out'] or process['exit_code']!=0:
            raise RuntimeError('Detection CLI timed out' if process['timed_out'] else f"Detection CLI exited {process['exit_code']}")
        shared=export_legends(panel,before,folder,image)
        job['shared_legends_path']=str(folder/'shared_legends.json')
        job['timing']['legend_crop_export_seconds']=shared['export_seconds']
        rows=predictions(read(before/'edit_data.json'),panel,shared['mapping'],'v46_before')
        before_record=publish_prediction(folder,panel,'v46_before','completed',rows)
        job['points_before']=len(rows);job['status']='correcting';save(folder/'job.json',job)
        phase='correction'
        if panel['modality']=='color':
            # Payload masks and relative paths remain resolvable in their copied native layout.
            shutil.copytree(before,after,dirs_exist_ok=True)
            if not (after/'step5_inputs.json').exists():
                raise ValueError('Current colour detector supplied no bound Step5 payload')
        process=launch(correction_command(panel,before,after,python),folder,'correction',env,timeout)
        job['timing']['correction_process_seconds']=process['wall_seconds'];job['correction_process']=process
        if process['timed_out'] or process['exit_code']!=0:
            raise RuntimeError('Correction CLI timed out' if process['timed_out'] else f"Correction CLI exited {process['exit_code']}")
        status=correction_result(panel,after)
        rows=predictions(read(after/'edit_data.json'),panel,shared['mapping'],'v46_after')
        after_record=publish_prediction(folder,panel,'v46_after',status['status'],rows)
        job.update(status=status['status'],points_after=len(rows),correction=status)
    except Exception as error:
        reason=f'{type(error).__name__}: {error}'
        job.update(status=f'{phase}_failed',reason=reason,traceback=traceback.format_exc())
        if before_record is None:
            publish_prediction(folder,panel,'v46_before','failed',[],reason)
        if after_record is None:
            publish_prediction(folder,panel,'v46_after','failed',[],reason)
    finally:
        if not (folder/'shared_legends.json').exists():
            # A failed marker stage may still have produced valid observed legend
            # evidence. Keep that independent artifact available to other methods.
            try:
                if image is None:
                    raise ValueError('Input pixel verification did not complete')
                export_legends(panel,before,folder,image)
            except Exception as error:
                save(folder/'shared_legends.json',dict(panel_id=panel['panel_id'],status='unavailable',
                    entries=[],mapping={},reason=f'Observed native legend unavailable: {error}',
                    input_sha256=panel['input_sha256'],source_sha256=panel['source_sha256']))
        job['timing']['total_wall_seconds']=perf_counter()-tick
        try:
            job['native_timing']=native_timing(panel,before,after)
        except Exception as error:
            job['native_timing']={'unavailable_reason':str(error)}
        save(folder/'job.json',job)
        combined={}
        for model in ('v46_before','v46_after'):
            artifact=read(folder/f'{model}.json')
            artifact['timing']=dict(job['timing'],native=job['native_timing'])
            save(folder/f'{model}.json',artifact)
            combined[model]=artifact
        save(folder/'result.json',dict(panel_id=panel['panel_id'],modality=panel['modality'],
            status=job['status'],models=combined,raw_before=str(before),raw_after=str(after),
            timing=job['timing'],native_timing=job['native_timing'],
            correction=job.get('correction'),reason=job.get('reason','')))
    return job


def scope_partition(rows,panel):
    """Split output scope without deleting any native prediction evidence."""
    plot,legend=panel['plot_bbox'],panel['legend_bbox']
    kept=[];excluded=[]
    for point in rows:
        point=dict(point);point.pop('exclusion_reason',None)
        x,y=point['crop_x_px'],point['crop_y_px'];reasons=[]
        if not (plot[0]<=x<plot[2] and plot[1]<=y<plot[3]):
            reasons.append('outside_provided_plot_roi')
        if legend[0]<=x<legend[2] and legend[1]<=y<legend[3]:
            reasons.append('inside_provided_legend_roi')
        if reasons:
            excluded.append(dict(point,exclusion_reason=reasons))
        else:
            kept.append(point)
    return kept,excluded


def finalize_exports(output=DEFAULT_OUTPUT):
    """Export-only audit/metadata refresh. No native algorithms are called."""
    output=Path(output).resolve();records=[];scope=[]
    folders=sorted(p.parent for p in output.glob('*/job.json'))
    if len(folders)!=10:
        raise ValueError('Only a finished exactly-ten-panel run can be finalized')
    for folder in folders:
        job=read(folder/'job.json');panel=job['input'];result=read(folder/'result.json')
        if job['status'] in ('running','correcting'):
            raise ValueError('Cannot finalize an active native run')
        before=folder/'raw_before';warnings=[]
        if panel['modality']=='bw':
            backend='bw_grid_v46' if job.get('points_before') is not None else 'bw_grid_v46_attempted_failed'
        else:
            diagnostic=read(before/'legend_hybrid_v46.json') if (before/'legend_hybrid_v46.json').exists() else {}
            if diagnostic.get('status')=='native_fallback':
                backend='native_v45_compatibility'
                warnings.append('Current v46 CLI was attempted, but the colour legend had no usable models: the actual before detector was native v45 compatibility, not the strict v46 hybrid detector. Strict v46 correction remained unavailable; no legacy SSIM correction was run.')
            elif diagnostic.get('status')=='hybrid_active':
                backend='color_marker_hybrid_v2'
            else:
                backend='v46_color_cli_attempted_failed'
        if job.get('reason'):
            warnings.append(job['reason'])
        correction=job.get('correction') or {}
        for name in ('v46_before','v46_after'):
            model=read(folder/f'{name}.json')
            all_rows=model.get('predictions',[])+model.get('excluded_predictions',[])
            retained,excluded=scope_partition(all_rows,panel)
            detection_seconds=job.get('timing',{}).get('detection_process_seconds')
            correction_seconds=job.get('timing',{}).get('correction_process_seconds') if name=='v46_after' else 0.
            total=(detection_seconds or 0.)+(correction_seconds or 0.) if detection_seconds is not None else None
            model.update(predictions=retained,excluded_predictions=excluded,
                raw_prediction_count=len(all_rows),prediction_count=len(retained),
                scope_policy='provided crop-local half-open plot ROI minus legend; all excluded native rows retained separately; no point references used',
                detection_backend=backend,backend=backend if name=='v46_before' else
                    ('bw_step5_v46' if panel['modality']=='bw' else 'v46_path_shape_v3') if model['status']!='failed' else 'correction_failed_or_unavailable',
                iterations=0 if name=='v46_before' else correction.get('iterations_by_series',{}),
                iterations_by_series={} if name=='v46_before' else correction.get('iterations_by_series',{}),
                correction_status='not_requested' if name=='v46_before' else correction.get('status_by_series',{'run':'failed_or_unavailable'}),
                warnings=warnings,source_sha256=panel['source_sha256'],input_sha256=panel['input_sha256'])
            model['timing'].update(total_seconds=total,inference_seconds=detection_seconds,
                correction_seconds=correction_seconds,
                shared_legend_seconds=job.get('native_timing',{}).get('shared_legend_seconds'),
                runtime_scope='Wall time of actual fresh production CLI processes, including imports/I/O. Before=one detection; after=that detection plus correction. Shared legend stage is INCLUDED, not an additional cost.')
            save(folder/f'{name}.json',model);result['models'][name]=model
            scope.append(dict(panel_id=panel['panel_id'],method=name,raw_count=len(all_rows),
                retained_count=len(retained),excluded_count=len(excluded),
                excluded_point_ids=[p['point_id'] for p in excluded]))
        result.update(source_sha256=panel['source_sha256'],input_sha256=panel['input_sha256'],
            detection_backend=backend,warnings=warnings,export_adapter_sha256=sha(__file__))
        save(folder/'result.json',result)
        job.update(detection_backend=backend,warnings=warnings,
            evaluated_points_before=result['models']['v46_before']['prediction_count'],
            evaluated_points_after=result['models']['v46_after']['prediction_count'])
        save(folder/'job.json',job);records.append(job)
    audit=dict(algorithm_rerun=False,scope=scope,
        raw_count=sum(r['raw_count'] for r in scope),retained_count=sum(r['retained_count'] for r in scope),
        excluded_count=sum(r['excluded_count'] for r in scope),export_adapter_sha256=sha(__file__))
    save(output/'export_audit.json',audit)
    summary=read(output/'summary.json')
    summary.update(jobs=records,export_audit=str(output/'export_audit.json'),
        detection_backends=dict(Counter(j['detection_backend'] for j in records)))
    save(output/'summary.json',summary)
    return audit


def run(inputs=DEFAULT_INPUTS,output=DEFAULT_OUTPUT,python=sys.executable,timeout=900):
    inputs=Path(inputs).resolve();output=Path(output).resolve()
    panels=validate_inputs(read(inputs))
    output.mkdir(parents=True,exist_ok=True)
    if (output/'settings.json').exists():
        raise FileExistsError('Benchmark already initialized; use a new explicitly named output folder')
    hashes={name:sha(SRC/name) for name in SOURCE_FILES}
    env=os.environ.copy();removed={k:env.pop(k) for k in ALGORITHM_ENV if k in env}
    env['PYTHONIOENCODING']='utf-8'
    save(output/'settings.json',dict(inputs_path=str(inputs),inputs_sha256=sha(inputs),
        python_executable=str(python),source_hashes=hashes,runner_sha256=sha(__file__),
        algorithm_environment_removed=removed,step5_max_iterations=5,panel_count=10,
        modality_counts={'bw':5,'color':5},sequential=True,timeout_per_phase_seconds=timeout,
        reads_evaluation_targets=False,shared_legend_protocol='actual observed v46 legend crops for every method'))
    jobs=[]
    for panel in panels:
        job=run_panel(panel,output,python,timeout,env);jobs.append(job)
        save(output/'summary.json',dict(jobs=jobs,complete=len(jobs)==10,status_counts=dict(Counter(j['status'] for j in jobs))))
        print(f"{panel['panel_id']} {job['status']}: {job.get('points_before')} -> {job.get('points_after')}; {job['timing']['total_wall_seconds']:.2f}s",flush=True)
    unchanged=all(sha(SRC/name)==value for name,value in hashes.items())
    save(output/'summary.json',dict(jobs=jobs,complete=True,production_sources_unchanged=unchanged,
        status_counts=dict(Counter(j['status'] for j in jobs))))
    if not unchanged:
        raise ValueError('Production source changed during the benchmark')
    finalize_exports(output)
    return jobs


if __name__=='__main__':
    if '--timed-cli' in sys.argv:
        separator=sys.argv.index('--')
        timing_path=sys.argv[sys.argv.index('--timing-path')+1]
        timed_cli(sys.argv[separator+1:],timing_path)
        raise SystemExit(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=DEFAULT_INPUTS)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--timeout',type=float,default=900)
    parser.add_argument('--finalize-only',action='store_true',help='Refresh finished exports only; never run detectors/correction')
    args=parser.parse_args()
    if args.finalize_only:
        print(json.dumps(finalize_exports(args.output),indent=2))
    else:
        run(args.inputs,args.output,args.python,args.timeout)

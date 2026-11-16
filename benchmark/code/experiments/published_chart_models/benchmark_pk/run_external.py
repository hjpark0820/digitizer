"""Frozen-input external-model benchmark; never opens reference/target files.

Both modalities use identical code. Shared native crop and supplied legend boxes
are the only inputs. ChartParser's paper-inspired correction and LineEX learned
matcher are the same functions as the preceding experiment. Series IDs are taken
directly from the supplied swatches, never optimized against reference points.
This is an annotated-layout/legend-assisted benchmark, not native end-to-end.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
VERSION = 'external_shared_input_benchmark_v1'
REQUIRED = {'panel_id', 'modality', 'figure_id', 'source_image_path', 'source_image_size',
            'source_sha256', 'image_path', 'image_size', 'input_sha256', 'crop_box_original',
            'plot_bbox', 'legend_bbox', 'swatches'}
FORBIDDEN = {'targets', 'target', 'reference_points', 'points', 'curves', 'own_json',
             'target_path', 'reference_path', 'shape', 'shape_hint', 'symbol', 'marker_type'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module_at(name, relative):
    path = EXPERIMENT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_safe_tree(value):
    """Fail closed if coordinate targets or semantic shape hints leak into input."""
    if isinstance(value, dict):
        bad = FORBIDDEN.intersection(value)
        if bad:
            raise ValueError(f'Forbidden detector-input fields: {sorted(bad)}')
        for child in value.values():
            validate_safe_tree(child)
    elif isinstance(value, list):
        for child in value:
            validate_safe_tree(child)


def box(value, size, *, optional=False):
    if value is None and optional:
        return None
    if value is None or len(value) != 4 or not np.isfinite(value).all():
        raise ValueError(f'Invalid bbox: {value}')
    x0,y0,x1,y1 = map(float,value)
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError(f'Out-of-bounds half-open bbox: {value}, size={size}')
    return [x0,y0,x1,y1]


def normalize_panel(row):
    validate_safe_tree(row)
    if REQUIRED-set(row):
        raise ValueError(f'Missing shared-input fields: {REQUIRED-set(row)}')
    if row['modality'] not in ('bw','color'):
        raise ValueError('Only frozen BW/color panels are supported')
    if any(int(row.get(key,0)) != 0 for key in ('clockwise_rotation_degrees','rotation_degrees')):
        raise ValueError('This frozen PK benchmark uses unrotated native source pixels')
    w,h = map(int,row['image_size'])
    crop = box(row['crop_box_original'],row['source_image_size'])
    if any(value != int(value) for value in crop) or [crop[2]-crop[0],crop[3]-crop[1]] != [w,h]:
        raise ValueError('Native shared crop must have integer edges and unchanged dimensions')
    plot = box(row['plot_bbox'],[w,h]); legend = box(row['legend_bbox'],[w,h],optional=True)
    ids = []
    for swatch in row['swatches']:
        if set(swatch) != {'series_id','label','bbox'}:
            raise ValueError('Swatches must contain only supplied series_id, label and bbox')
        ids.append(str(swatch['series_id']))
        box(swatch['bbox'],[w,h])
    if len(ids) != len(set(ids)):
        raise ValueError('Supplied series IDs must be unique')
    return dict(panel_id=row['panel_id'], modality=row['modality'], figure_id=row['figure_id'],
                image_path=row['image_path'], image_size=[w,h], input_sha256=row['input_sha256'],
                source_sha256=row['source_sha256'], source_filename=Path(row['source_image_path']).name,
                source_image_path=row['source_image_path'], source_image_size=row['source_image_size'],
                crop_box_upright=list(map(int,crop)), crop_box_original=list(map(int,crop)),
                clockwise_rotation_degrees=0, plot_box=plot, legend_box=legend,
                swatches=[dict(s) for s in row['swatches']])


def verify_image_binding(panel):
    if sha(panel['image_path']) != panel['input_sha256']:
        raise ValueError('Shared input crop hash mismatch')
    if sha(panel['source_image_path']) != panel['source_sha256']:
        raise ValueError('Original source image hash mismatch')
    with Image.open(panel['image_path']) as im:
        cropped = np.asarray(im.convert('RGB'))
    with Image.open(panel['source_image_path']) as im:
        if list(im.size) != list(panel['source_image_size']):
            raise ValueError('Original image dimensions mismatch')
        expected = np.asarray(im.convert('RGB').crop(panel['crop_box_original']))
    if list(cropped.shape[:2][::-1]) != panel['image_size'] or not np.array_equal(cropped,expected):
        raise ValueError('Shared image is not exactly the specified native original crop')
    return cropped


def inside(x,y,b):
    return bool(b and b[0] <= x < b[2] and b[1] <= y < b[3])


def scoped(point,panel):
    return inside(point['x'],point['y'],panel['plot_box']) and not inside(point['x'],point['y'],panel['legend_box'])


def original_point(point,panel):
    return dict(point, x_px=float(point['x'])+panel['crop_box_original'][0],
                y_px=float(point['y'])+panel['crop_box_original'][1])


def shared_legend_crops(panel,image,shared_path,directory):
    """Read only shared legend metadata/crops, never v46 point/result files.

    Production colour v46 has native automatic legend extraction, so using the
    original layout swatches only for the external models would be asymmetric.
    Both pipelines therefore share the actual observed source-pixel boxes that
    v46 exported. Native IDs were mapped one-to-one by bbox overlap, not targets.
    """
    metadata=read(shared_path)
    if metadata['input_sha256']!=panel['input_sha256'] or metadata['source_sha256']!=panel['source_sha256']:
        raise ValueError('Shared legend image binding mismatch')
    shared_hash=sha(shared_path)
    records,images=[],[]
    seen=set();dx,dy=panel['crop_box_original'][:2]
    names={s['series_id']:s['label'] for s in panel['swatches']}
    for index,entry in enumerate(metadata['entries']):
        identity=entry['series_id']
        if not isinstance(identity,str) or not identity or identity in seen:
            raise ValueError('Shared legend series IDs must be nonempty and unique, including extras')
        seen.add(identity)
        local=box(entry['bbox_crop'],panel['image_size'])
        if any(v!=int(v) for v in local):
            raise ValueError('Exported source-pixel swatch boxes must have integer edges')
        a,b,c,d=map(int,local)
        original=[a+dx,b+dy,c+dx,d+dy]
        if list(entry['bbox_original'])!=original:
            raise ValueError('Shared swatch crop/full-original offset mismatch')
        path=Path(entry['crop_path'])
        if sha(path)!=entry['sha256']:
            raise ValueError('Shared swatch file hash mismatch')
        with Image.open(path) as im:
            crop=np.asarray(im.convert('RGB'))
        if list(crop.shape[:2][::-1])!=list(entry['image_size']) or not np.array_equal(crop,image[b:d,a:c]):
            raise ValueError('Shared swatch is not exactly the original RGB pixel crop')
        records.append(dict(legend_index=index,series_id=identity,label=names.get(identity,identity),
            source_native_legend_id=entry['source_native_legend_id'],
            box_input=[a,b,c,d],box_original=original,crop_path=str(path.resolve()),
            crop_sha256=entry['sha256'],shape_label=None,
            shared_legends_path=str(Path(shared_path).resolve()),shared_legends_sha256=shared_hash,
            mapping_provenance=metadata.get('mapping'),
            provenance='Shared actual observed v46 legend source-pixel crop; no template pixels or v46 marker coordinates used. IDs supplied by bbox-only mapping.'))
        images.append(crop.copy())
    return records,images


def group_positions(points,panel,image,legends,legend_images,matcher,patch_module,directory):
    """Every patch winner retains the input swatch's series ID, never a GT mapping."""
    started=time.perf_counter()
    patchset=patch_module.extract_pivot_patches(image,[(p['x'],p['y']) for p in points])
    predictions,unavailable = [],[]
    patch_records=[]
    for index,patch,bbox in zip(patchset['valid_indices'],patchset['patches'],patchset['boxes']):
        p=points[index]
        path=directory/'patches'/f"{p['point_id']}.png"
        path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(patch).save(path)
        patch_records.append(dict(point_index=index,point_id=p['point_id'],box_input=list(map(int,bbox)),
                                  patch_path=str(path.resolve())))
    crop_seconds=time.perf_counter()-started
    match_started=time.perf_counter()
    scores=np.empty((len(patchset['patches']),len(legends)),np.float32)
    if legends and patchset['patches']:
        scores=np.asarray(matcher.score_patches(legend_images,patchset['patches']),np.float32)
        if scores.shape!=(len(patchset['patches']),len(legends)) or not np.isfinite(scores).all():
            raise ValueError('Invalid learned similarity matrix')
        for row,patch in zip(scores,patch_records):
            p=points[patch['point_index']];winner=int(np.argmax(row))
            predictions.append(original_point(dict(p,series_id=legends[winner]['series_id'],
                legend_index=winner,score=float(row[winner]),winning_similarity=float(row[winner]),
                similarities=row.tolist(),patch_path=patch['patch_path'],patch_box_input=patch['box_input']),panel))
    elif not legends:
        unavailable += [dict(original_point(p,panel),reason='no_supplied_legend') for p in points]
        predictions += [original_point(dict(p,series_id='unknown_shared_legend_unavailable',score=None,
            winning_similarity=None,similarities=[],legend_index=None,
            grouping_status='unavailable_no_shared_legend'),panel) for p in points]
    if legends:
        for item in patchset['unavailable']:
            point=points[item['index']]
            unavailable.append(dict(original_point(point,panel),reason=item['reason']))
            predictions.append(original_point(dict(point,series_id='unknown_patch_unavailable',score=None,
                winning_similarity=None,similarities=[],legend_index=None,
                grouping_status='unavailable_incomplete_native_patch'),panel))
    match_seconds=time.perf_counter()-match_started
    directory.mkdir(parents=True,exist_ok=True);np.save(directory/'similarity.npy',scores)
    return dict(predictions=predictions,unavailable=unavailable,patches=patch_records,
                similarity_path=str((directory/'similarity.npy').resolve()),
                similarity_row_point_ids=[p['point_id'] for p in patch_records],
                similarity_column_series_ids=[r['series_id'] for r in legends],
                timing=dict(patch_extract_and_export_seconds=crop_seconds,
                            learned_matching_seconds=match_seconds,total_seconds=time.perf_counter()-started))


def correct_chartparser(heads,panel,geometry,directory):
    started=time.perf_counter()
    with np.load(heads['raw_heads_path']) as store:
        segmentation=store['segmentation']
    binary,closed,thin,labels,components=geometry.centreline(segmentation)
    directory.mkdir(parents=True,exist_ok=True)
    for name,mask in [('curve_threshold_512',binary),('curve_closed_512',closed),('centerline_512',thin)]:
        Image.fromarray(mask.astype(np.uint8)*255).save(directory/f'{name}.png')
    raw,corrected,rejected,excluded,public = [],[],[],[],[]
    for i,p in enumerate(heads['raw_points']):
        point=dict(point_id=f'CP{i+1:04d}',x=float(p['x']),y=float(p['y']),
                   model_x=p['model_x'],model_y=p['model_y'],detection_score=p['score'])
        if not scoped(point,panel):
            excluded.append(original_point(point,panel));continue
        raw.append(point)
        ref=geometry.public_position_reference(p['model_x'],p['model_y'],segmentation.argmax(axis=1)[0])
        public.append(dict(point_id=point['point_id'],status='retained' if ref else 'rejected',
                           **(geometry.point_coordinates(*ref,panel) if ref else {})))
        matched=geometry.nearest_centreline(p['model_x'],p['model_y'],thin)
        if matched is None:
            rejected.append(dict(original_point(point,panel),reason='no_centerline_in_band_radius'));continue
        coordinates=geometry.point_coordinates(*matched,panel)
        corr=dict(point_id=point['point_id'],x=coordinates['x'],y=coordinates['y'],
                  model_x=matched[0],model_y=matched[1],raw_x=point['x'],raw_y=point['y'],
                  detection_score=p['score'],component_id=int(labels[matched[1],matched[0]]))
        if not scoped(corr,panel):
            rejected.append(dict(original_point(corr,panel),reason='corrected_outside_scope'));continue
        corrected.append(corr)
    return dict(raw_points=raw,corrected_points=corrected,rejected_points=rejected,excluded_points=excluded,
                public_process_kps_reference=public,components=components,
                centerline_path=str((directory/'centerline_512.png').resolve()),
                implementation_choices=geometry.CHOICES,correction_seconds=time.perf_counter()-started)


def record(panel,method,prediction_data,legends,timing,**diagnostics):
    prediction_data=dict(prediction_data)
    patch_timing=prediction_data.pop('timing')
    timing=dict(timing,total_seconds=timing['method_total_excluding_shared_verification_seconds'],
                shared_legend_frontend_seconds=None,shared_legend_frontend_cost_included=False,
                timing_scope='Resident external heads + correction + matching/export. Model initialization and upstream v46 shared legend extraction excluded.')
    return dict(version=VERSION,panel_id=panel['panel_id'],modality=panel['modality'],method=method,
                status='completed' if legends else 'unavailable_legend',
                coordinate_system='x_px/y_px = full original unrotated image pixels; x/y = shared native crop pixels',
                image_path=panel['image_path'],input_sha256=panel['input_sha256'],
                source_image_path=panel['source_image_path'],source_sha256=panel['source_sha256'],
                crop_box_original=panel['crop_box_original'],plot_box=panel['plot_box'],legend_box=panel['legend_box'],
                legends=legends,**prediction_data,timing=timing,patch_matching_timing=patch_timing,
                timing_breakdown=timing,**diagnostics,
                limitations=[
                    'Annotated shared plot/legend ROI; actual observed source-pixel swatch crops exported by v46 are shared. Not an end-to-end chart parser.',
                    'Native swatch IDs mapped to supplied IDs by bbox-only overlap; then learned swatch argmax. Unmapped extra swatches retain their unknown ID. No reference-point matching.',
                    'If the shared legend front end failed, actual scoped head/corrected locations remain as unknown-series predictions for center-only evaluation; no legend or marker is fabricated.',
                    'A valid detection whose native 40x20 patch crosses the image edge is retained as unknown_patch_unavailable, not dropped or padded.',
                    'Scores are learned similarities, not calibrated correctness probabilities.',
                    'Fixed native 40x20 patches, learned RGB Siamese matcher; no histogram override.',
                    'Pivot locations may not correspond to visible markers. No semantic marker-shape classification.',
                    'No reference targets, reference curves, reference point positions, or shape labels read.',
                    'No deduplication or score threshold added after highest-similarity legend assignment.',
                ])


def run_panel(panel,output,models,modules,matcher,device,shared_legends_root):
    binding_started=time.perf_counter();image=verify_image_binding(panel)
    destination=output/panel['panel_id'];destination.mkdir(parents=True,exist_ok=True)
    binding_seconds=time.perf_counter()-binding_started
    legend_started=time.perf_counter()
    legends,legend_images=shared_legend_crops(panel,image,shared_legends_root/panel['panel_id']/'shared_legends.json',destination)
    shared_metadata=read(shared_legends_root/panel['panel_id']/'shared_legends.json')
    legend_seconds=time.perf_counter()-legend_started
    results={}
    if 'chartparser' in models:
        heads=modules['chartparser'].infer(panel,models['chartparser'],device,output/'raw_heads'/'chartparser')
        geometry=correct_chartparser(heads,panel,modules['geometry'],destination/'chartparser_geometry')
        for key,positions in [('aicp_corrected_grouped',geometry['corrected_points']),
                              ('aicp_raw_grouped',geometry['raw_points'])]:
            grouped=group_positions(positions,panel,image,legends,legend_images,matcher,modules['matcher'],destination/key)
            correction_time=geometry['correction_seconds'] if key=='aicp_corrected_grouped' else 0.
            timing=dict(shared_input_verification_seconds=binding_seconds,legend_crop_seconds=legend_seconds,
                network_seconds=heads['timing']['network_seconds'],
                head_postprocess_and_export_seconds=heads['timing']['total_seconds']-heads['timing']['network_seconds'],
                curve_correction_seconds=correction_time,**grouped['timing'],
                method_total_excluding_shared_verification_seconds=heads['timing']['total_seconds']+legend_seconds+correction_time+grouped['timing']['total_seconds'])
            results[key]=record(panel,key,grouped,legends,timing,
                raw_heads_result_path=str((output/'raw_heads'/'chartparser'/panel['panel_id']/'result.json').resolve()),
                raw_points=[original_point(p,panel) for p in geometry['raw_points']],
                corrected_points=[original_point(p,panel) for p in geometry['corrected_points']],
                rejected_points=geometry['rejected_points'],excluded_points=geometry['excluded_points'],
                centerline_path=geometry['centerline_path'],
                public_process_kps_reference=geometry['public_process_kps_reference'],
                implementation_choices=geometry['implementation_choices'],
                native_legend_stage='unavailable; supplied benchmark swatch crops used, never class-3 remapping')
    if 'lineex' in models:
        line_model,gate=models['lineex']
        heads=modules['lineex'].infer(line_model,gate,panel,str(device))
        write(output/'raw_heads'/'lineex'/f"{panel['panel_id']}.json",heads)
        positions=[dict(point_id=f"LX{p['query_id']:04d}",x=float(p['x']),y=float(p['y']),detection_score=None)
                   for p in heads['points'] if scoped(p,panel)]
        grouped=group_positions(positions,panel,image,legends,legend_images,matcher,modules['matcher'],destination/'lineex_grouped')
        timing=dict(shared_input_verification_seconds=binding_seconds,legend_crop_seconds=legend_seconds,
            network_seconds=heads['timing']['network_seconds'],
            head_postprocess_and_export_seconds=heads['timing']['total_seconds']-heads['timing']['network_seconds'],
            curve_correction_seconds=0.,**grouped['timing'],
            method_total_excluding_shared_verification_seconds=heads['timing']['total_seconds']+legend_seconds+grouped['timing']['total_seconds'])
        results['lineex_grouped']=record(panel,'lineex_grouped',grouped,legends,timing,
            raw_heads_result_path=str((output/'raw_heads'/'lineex'/f"{panel['panel_id']}.json").resolve()),
            raw_points=[original_point(p,panel) for p in positions],
            raw_queries=heads['raw_queries'],head_status=heads['status'],
            implementation_choices={'keypoints':'Official 64-query L+D checkpoint and background gate',
                'grouping':'Same official learned Siamese matcher as ChartParser paper, no histogram override.'})
    for key,result in results.items():
        result['shared_legend_status']=shared_metadata.get('status')
        result['shared_legend_reason']=shared_metadata.get('reason')
        result['shared_legends_sha256']=sha(shared_legends_root/panel['panel_id']/'shared_legends.json')
        result['warnings']=result['limitations']
        write(destination/f'{key}.json',result)
    write(destination/'result.json',dict(version=VERSION,panel_id=panel['panel_id'],modality=panel['modality'],
        input_sha256=panel['input_sha256'],source_sha256=panel['source_sha256'],
        crop_box_original=panel['crop_box_original'],models=results,
        shared_legends_path=str((shared_legends_root/panel['panel_id']/'shared_legends.json').resolve())))
    return {key:dict(status=r['status'],point_count=len(r['predictions']),
                    result_path=str((destination/f'{key}.json').resolve()),
                    timing=r['timing_breakdown']) for key,r in results.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shared-legends-root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--models',nargs='+',choices=['chartparser','lineex'],default=['chartparser','lineex'])
    parser.add_argument('--ids',nargs='*')
    args=parser.parse_args()
    frozen=read(args.manifest);validate_safe_tree(frozen)
    panels=[normalize_panel(p) for p in frozen['panels']]
    if len(panels)!=10 or sorted(p['modality'] for p in panels)!=['bw']*5+['color']*5:
        raise ValueError('User-authorized final cohort is exactly five BW and five color panels')
    if args.ids:
        panels=[p for p in panels if p['panel_id'] in set(args.ids)]
        if len(panels)!=len(set(args.ids)):
            raise ValueError('Unknown or duplicate selected panel IDs')
    # Import/load only after the frozen detector-only contract is validated.
    modules={'matcher':module_at('benchmark_lineex_matcher','lineex/legend_matcher.py'),
             'geometry':module_at('benchmark_chartparser_geometry','chartparser/run_grouped.py')}
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    setup=time.perf_counter();matcher=modules['matcher'].LegendMatcher(device=str(device),batch_size=args.batch_size)
    load_times={'legend_matcher_seconds':time.perf_counter()-setup}
    provenance={'matcher':matcher.provenance};models={}
    if 'chartparser' in args.models:
        modules['chartparser']=module_at('benchmark_chartparser_heads','chartparser/run_heads.py')
        setup=time.perf_counter()
        models['chartparser'],provenance['chartparser']=modules['chartparser'].build_model(
            EXPERIMENT/'chartparser/weights/epoch-19.pth',device)
        load_times['chartparser_seconds']=time.perf_counter()-setup
    if 'lineex' in args.models:
        modules['lineex']=module_at('benchmark_lineex_heads','lineex/run_keypoints.py')
        setup=time.perf_counter()
        model,gate,provenance['lineex']=modules['lineex'].load_model(str(device))
        models['lineex']=(model,gate);load_times['lineex_seconds']=time.perf_counter()-setup
    helper_paths=['chartparser/run_heads.py','chartparser/run_grouped.py','lineex/run_keypoints.py','lineex/legend_matcher.py']
    write(args.output/'runtime.json',dict(version=VERSION,manifest_path=str(args.manifest.resolve()),
        manifest_sha256=sha(args.manifest),runner_sha256=sha(__file__),device=str(device),
        helper_source_sha256={name:sha(EXPERIMENT/name) for name in helper_paths},
        shared_legends_root=str(args.shared_legends_root.resolve()),
        model_load_seconds=load_times,provenance=provenance,
        timing_notes='One run per panel, no warmup; model load excluded from per-panel timing. Raw-position ablation matching timed separately. No batch throughput or v46 speedup claim.'))
    rows=[]
    for panel in panels:
        try:
            result=run_panel(panel,args.output,models,modules,matcher,device,args.shared_legends_root)
            row=dict(panel_id=panel['panel_id'],modality=panel['modality'],status='completed',models=result)
        except Exception:
            row=dict(panel_id=panel['panel_id'],modality=panel['modality'],status='failed',error=traceback.format_exc())
            write(args.output/panel['panel_id']/'failure.json',row)
        rows.append(row)
        write(args.output/'summary.json',dict(expected=len(panels),processed=len(rows),panels=rows))
        print(f"{panel['panel_id']} {row['status']} "+str({k:v['point_count'] for k,v in row.get('models',{}).items()}),flush=True)
    return int(any(row['status']=='failed' for row in rows))


if __name__=='__main__':raise SystemExit(main())

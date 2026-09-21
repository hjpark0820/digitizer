"""v45 writer/GUI contract for explicitly selected marker-free colour charts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from color_legend_runtime_v46 import exclusive_box, inclusive, _plain
from .pipeline import detect, spatial_valid


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def _labels(image, legend, entries):
    names=[f'color{i+2:02d}' for i in range(len(entries))]
    labels={n:n for n in names};error=None
    try:
        import ocr_bridge_v46 as pytesseract
        # A 3px-high legend stroke is at the middle of a much taller text row.
        # Marker-body OCR crops would cut off the upper half of every letter.
        # Use table row midpoints, not stroke thickness, for text boundaries.
        columns=[]
        for name,entry in sorted(zip(names,entries),key=lambda row:row[1]['box'][0]):
            if columns and abs(entry['box'][0]-np.median([e['box'][0] for _,e in columns[-1]]))<=8:
                columns[-1].append((name,entry))
            else:columns.append([(name,entry)])
        for col_index,column in enumerate(columns):
            column.sort(key=lambda row:row[1]['box'][1])
            centers=[.5*(e['box'][1]+e['box'][3]-1) for _,e in column]
            spacing=float(np.median(np.diff(centers))) if len(centers)>1 else float(legend[3]-legend[1])
            right=min(e['box'][0] for _,e in columns[col_index+1])-3 if col_index+1<len(columns) else legend[2]
            for i,(name,entry) in enumerate(column):
                top=max(legend[1],int((centers[i-1]+centers[i])/2) if i else int(centers[i]-.5*spacing))
                bottom=min(legend[3],int((centers[i]+centers[i+1])/2) if i+1<len(column) else int(centers[i]+.5*spacing)+1)
                left=entry['box'][2]+3
                if bottom<=top or right<=left:continue
                crop=cv2.cvtColor(image[top:bottom,left:right],cv2.COLOR_BGR2GRAY)
                crop=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
                _,crop=cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                crop=cv2.copyMakeBorder(crop,10,10,10,10,cv2.BORDER_CONSTANT,value=255)
                label=' '.join(pytesseract.image_to_string(crop,config='--psm 7').split())
                if label:labels[name]=label
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    return labels,error


def reference_role(label):
    """Only explicit source text declares quantification/detection references.

    Gray/black colour or horizontal geometry alone must never exclude a series.
    """
    return 'reference_line' if re.search(r'\b(?:LLOQ|ULOQ|LOQ|LLOD|LOD)\b|limit\s+of\s+(?:quantification|detection)',str(label),re.I) else 'data_series'


def run(marker_runtime, env, legend_runtime):
    if not legend_runtime.active or not legend_runtime.entries:
        raise ValueError('Line-only mode requires usable source legend colour entries; select a legend region')
    image=env['img'];plot=exclusive_box(env['PLOT_AREA'],image.shape)
    legend=exclusive_box(env['LEGEND_BOX'],image.shape);entries=legend_runtime.entries
    names=[f'color{i+2:02d}' for i in range(len(entries))]
    labels,label_error=_labels(image,legend,entries)
    # Use the original native v45 masks exactly as in the evaluated experiment.
    # Do not erase gray grid rows or run the marker-specific furniture filter.
    masks,soft,info=env['colour_masks'](image,inclusive(plot),inclusive(legend),
        swatch_boxes=[inclusive(e['box']) for e in entries],metric='tube',spatial=False,furniture=None)
    if len(info)!=len(entries) or set(masks)!=set(range(len(entries))):
        raise RuntimeError('Native type3 colour mask identities no longer match the source legend')
    x0,y0,x1,y1=plot;valid=spatial_valid(plot,legend);series=[];own_maps={};soft_maps={}
    for i,(name,entry,observed) in enumerate(zip(names,entries,info)):
        box=list(entry['box']);rgb=list(map(int,observed['rgb']))
        series.append(dict(id=name,label=labels[name],rgb=rgb,line_width=max(1.,float(box[3]-box[1])),
            role=reference_role(labels[name]),role_source='explicit_source_legend_text_or_default_data',
            marker_state='absent_by_user_line_only_selection',marker_glyph_detected=False,
            palette_source_box=box,colour_source='native_v45_observed_line_legend'))
        own_maps[name]=np.asarray(masks[i][y0:y1,x0:x1],bool)&valid
        soft_maps[name]=np.asarray(soft[i][y0:y1,x0:x1])*valid
    print('[v46 type3] Line-only mode: native colour masks -> grid-aware paths -> source errorbar/kink centres '
        '-> support gate -> whole-path endpoints (suppressed endpoints treated as strong).',flush=True)
    result,maps=detect(image,plot,legend,series,own_maps,soft_maps,valid=valid)
    source=Path(env['IMG_PATH'])
    result.update(source_image_path=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        image_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
        image_size=[image.shape[1],image.shape[0]],case_id=source.stem,display_name=source.stem,
        label_ocr_error=label_error,detector_backend='v46_type3_v1')
    destination=Path(env['OUT_DIR']);destination.mkdir(parents=True,exist_ok=True)
    _write(destination/'type3_detection_v46.json',result)
    np.savez_compressed(destination/'type3_evidence_v46.npz',valid=valid,grid_likelihood=maps['grid_likelihood'],
        text_mask=maps['text_mask'],**{'own_'+k:v for k,v in own_maps.items()},
        **{'soft_'+k:v for k,v in soft_maps.items()})
    data=[s for s in series if s['role']=='data_series'];data_names=[s['id'] for s in data]
    points={n:[] for n in data_names}
    for p in result['points']:
        # Native writers consume integer display pixels; exact fitted centres
        # and all provenance are restored to edit_data by finalize_outputs.
        points[p['series_id']].append(dict(x=int(np.clip(round(p['x_px']),x0,x1-1)),
            y=int(np.clip(round(p['y_px']),y0,y1-1)),x_px=p['x_px'],y_px=p['y_px'],
            fitness=1.,px=1,source='v46_type3_v1',candidate_id=p['id'],marker_glyph_detected=False))
    for value in points.values():value.sort(key=lambda p:(p['x'],p['y']))
    env.update(_names=data_names,_rgbs=[s['rgb'] for s in data],all_detections=points,
        all_tcaps_out={n:[] for n in data_names},all_results={n:([],0.) for n in data_names},_V45_STEP5=None,
        COLORS=[dict(name=s['id'],mean_rgb=s['rgb'],swatch_rgb=s['rgb'],px_count=0) for s in data])
    env['_cfg'].curve_names=data_names.copy();env['_cfg'].stem_x=[];env['_cfg'].x_values=[]
    digitizer=env['_dig'];digitizer.palette=[tuple(s['rgb']) for s in data]
    digitizer.is_sink=np.zeros(len(data),bool)
    assign=np.full(image.shape[:2],-1,np.int16)
    if data:
        stack=np.stack([soft_maps[s['id']] for s in data]);local=np.argmax(stack,axis=0).astype(np.int16)
        local[(np.max(stack,axis=0)<.15)|~valid]=-1;assign[y0:y1,x0:x1]=local
    digitizer.assign=assign;np.save(destination/'assign.npy',assign)
    marker_runtime.active=True;marker_runtime.type3_detection=result
    marker_runtime.type3_labels={s['id']:s['label'] for s in data}
    marker_runtime.diagnostics=dict(version='v46_type3_v1',status='completed',legend_box=legend,
        plot_box=plot,points=result['points'],templates=[],template_errors=[],
        series_mode='line-only',marker_glyph_count=0)
    legend_runtime.diagnostics.update(scope='Source legend colours for marker-free type3 paths and hypotheses',
        downstream_marker_version='v46_type3_v1',series_mode='line-only')
    legend_runtime._save()
    print(f"[v46 type3] {len(result['points'])} initial active centres; "
        f"{len(result['endpoint_candidates'])} whole-path endpoint tests. -> type3_detection_v46.json",flush=True)


def finalize_outputs(marker_runtime, env):
    """Keep marker-free identity and exact coordinates through legacy writers."""
    result=getattr(marker_runtime,'type3_detection',None)
    if result is None:return
    path=Path(env['OUT_DIR'])/'edit_data.json'
    if not path.is_file():raise RuntimeError('v46 writer did not produce edit_data.json')
    edit=json.loads(path.read_text(encoding='utf-8-sig'))
    by_id={s['id']:s for s in result['series'] if s['role']=='data_series'}
    for curve in edit['curves']:
        sid=curve['name']
        if sid not in by_id:raise RuntimeError('Type3 GUI series identity changed during native export')
        curve.update(series_mode='line-only',marker_class=None,marker_glyph_detected=False,label=by_id[sid]['label'])
        curve['points']=[dict(x=float(p['x_px']),y=float(p['y_px']),candidate_id=p['id'],
            kind=p['kind'],original_L0=True,independent_measurement_evidence=True,
            marker_glyph_detected=False,**({'endpoint_evidence':deepcopy(p['endpoint_evidence'])}
                if 'endpoint_evidence' in p else {}))
            for p in sorted(result['points'],key=lambda q:(q['x_px'],q['y_px'])) if p['series_id']==sid]
    edit.update(series_mode='line-only',detector_backend='v46_type3_v1',type3_detection_file='type3_detection_v46.json')
    _write(path,edit)

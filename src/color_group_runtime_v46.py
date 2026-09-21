"""GUI/CLI bridge for the reviewed colour-group BW detector and typed Step 5.

Selection is image/legend based, never antibody/file based. Source pixels and
legend row IDs survive detection, export, manual editing, and resumed correction.
The colour-group experiment uses segment geometry, not a single path shared
by several same-colour series. Unique-colour charts retain their existing route.
"""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import zlib

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from bw_step5_v46 import _pack_reference, _normalise, save
from bw_suppressed_v46 import encode_marker_mask

VERSION = 'color_group_bw_gui_v46_v1'
STATE_FILE = 'color_group_state_v46.json'


def ambiguous_palette(entries):
    """Same reviewed Lab/hue-family test; brightness alone cannot split shapes."""
    if len(entries) < 2:
        return False
    bgr = np.uint8([[e['rgb'][::-1] for e in entries]])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0].astype(float)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0].astype(float)
    for i in range(len(entries)):
        for j in range(i):
            dh = abs(hsv[i, 0]-hsv[j, 0]); dh = min(dh, 180-dh)*2
            if np.linalg.norm(lab[i]-lab[j]) <= 14 or (min(hsv[i, 1], hsv[j, 1]) >= 40 and dh <= 8):
                return True
    return False


def match_rows(templates, entries, names):
    """One-to-one geometric association, not RGB ties or surviving point order."""
    if len(templates) != len(entries):
        raise ValueError('Colour-group legend roster differs from GUI legend; inspect diagnostics')
    cost = np.empty((len(templates), len(entries)))
    for i, t in enumerate(templates):
        tx, ty = t.marker_center
        for j, e in enumerate(entries):
            a, b, c, d = e['box']
            cost[i, j] = ((tx-(a+c)/2)/max(c-a, t.diameter))**2 + ((ty-(b+d)/2)/max(d-b, t.diameter))**2
    rows, cols = linear_sum_assignment(cost)
    if any(cost[i, j] > 1. for i, j in zip(rows, cols)):
        raise ValueError('Colour-group swatch could not be associated with its GUI legend row')
    return {templates[i].key: names[j] for i, j in zip(rows, cols)}


def binding(image):
    return dict(shape=list(image.shape), sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())


def unpack(plane, shape):
    """Bounded decompression also protects directly supplied CLI sessions."""
    if plane['shape'] != list(shape):
        raise ValueError('Colour-group evidence dimensions changed')
    size = int(np.prod(shape))
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(plane['zlib_base64'], validate=True), size+1)
    if len(raw) != size or not decoder.eof or decoder.unused_data or hashlib.sha256(raw).hexdigest() != plane['sha256']:
        raise ValueError('Colour-group evidence integrity mismatch')
    return np.frombuffer(raw, np.uint8).reshape(shape).copy()


def detect(runtime, env, legend_runtime, plot, legend, report):
    from color_group_bw_v46 import prepare_groups, detection_options
    from bw_pipeline_v46 import detect_points
    from x_singleton_suppressed_v46 import generate

    image = env['img']; destination = Path(env['OUT_DIR'])
    entries = legend_runtime.entries
    names = [f'color{i+2:02d}' for i in range(len(entries))]
    groups, _ = prepare_groups(image, plot, legend, group_policy='hue_family', composition=True,
                               ownership_policy='uncertainty_v1', grid_evidence='confirmed')
    mapping = match_rows([t for g in groups for t in g['templates']], entries, names)
    state = dict(version=VERSION, image=binding(image), plot_box=list(plot), legend_box=list(legend),
                 names=names, groups=[], series_models={}, total_correction_runs=0)
    points, pool, templates, membership, group_reports = [], [], [], {}, []
    for g in groups:
        dest = destination/'colour_groups'/g['id']; dest.mkdir(parents=True, exist_ok=True)
        options = detection_options(g)
        result = detect_points(g['image'], plot, legend,
            prepared_templates=(g['templates'],g['reports']),window_image=g['window_image'],
            window_occlusion_mask=g['occlusion'],window_uncertainty_mask=g['uncertainty'],**options)
        save(dest/'detection.json', {k:v for k,v in result.items() if k != 'diag_steps'})
        cv2.imwrite(str(dest/'confirmed_gray.png'),g['window_image'])
        cv2.imwrite(str(dest/'uncertainty.png'),np.uint8(255*g['uncertainty']))
        group_names = [mapping[t.key] for t in g['templates']]
        for field, target in (('kept',points), ('suppressed',pool)):
            for p in result[field]:
                q = deepcopy(p)
                q.update(swatch_id=mapping[p['swatch_id']], group_id=g['id'])
                target.append(q)
        for t, r in zip(g['templates'], g['reports']):
            name = mapping[t.key]; e = entries[names.index(name)]
            soft = np.asarray(t.soft, np.float32)
            templates.append(dict(id=name, rgb=list(e['rgb']), swatch_box=list(t.swatch_box),
                diameter=float(t.diameter), soft=soft, source_center=list(t.marker_center),
                legend_shape_hint=t.shape_hint, provenance=r.get('template_policy'),
                composition=r.get('composition'), marker_class=t.name))
            observed = next((p for p in points+pool if p['swatch_id']==name), None)
            state['series_models'][name] = deepcopy(observed) if observed else dict(
                swatch_id=name,class_name=t.name,source_diameter=float(t.diameter),
                marker_mask=encode_marker_mask(t.mask),marker_offset_x=0.,marker_offset_y=0.)
            state['series_models'][name]['rgb'] = list(e['rgb'])
            state['series_models'][name]['legend_box'] = list(t.swatch_box)
            membership[name] = 1-cv2.cvtColor(g['window_image'],cv2.COLOR_BGR2GRAY)[plot[1]:plot[3],plot[0]:plot[2]]/255.
        state['groups'].append(dict(id=g['id'], names=group_names, diameter=float(result['d_est']),
            gray=_pack_reference(cv2.cvtColor(g['window_image'],cv2.COLOR_BGR2GRAY)),
            ignore=_pack_reference(np.uint8(g['occlusion']>.5)), runtime=None))
        group_reports.append(dict(id=g['id'], names=group_names, active=len(result['kept']),
            suppressed=len(result['suppressed']), options=options,
            colour_visibility_policy=result['diagnostics'].get('colour_visibility_policy'),
            composition=[dict(id=mapping[t.key], policy=r.get('template_policy'),
                              model=r.get('composition',{}).get('best_model_name')) for t,r in zip(g['templates'],g['reports'])]))
    # One frozen plot-wide pool, including targets in other colour groups.
    pool, overlap = generate(points,pool,state['series_models'],plot,legend)
    state.update(P_current=points,S_current=pool,overlap=overlap,original_points=deepcopy(points))
    save(destination/STATE_FILE,state)
    converted = [dict(p, series_id=p['swatch_id'],x_px=p['cx'],y_px=p['cy'],
                      score=float(p.get('confidence',0.)),candidate_id=p.get('point_id',f'group_{i}'))
                 for i,p in enumerate(points)]
    report.update(version=VERSION,status='completed',same_colour_bw_v3=True,
        selection_config=None,guide_policy='group_bw_native',
        ownership_policy='uncertainty_v1',grid_evidence='confirmed',
        backend=VERSION,groups=group_reports,points=converted,uncertain_points=pool,
        templates=[{k:v for k,v in t.items() if k!='soft'} for t in templates],
        scale_search=True,scale_policy='reviewed_colour_group_per_candidate_0.90_1.12',
        correction_backend='colour_group_bw_elements',step5_state_file=STATE_FILE,
        overlap=overlap,triangle_errorbar_guard=dict(status='native_bw_window_verification',enabled=False),
        mask_note='Hue-family ownership selects ink; BW grid/window distinguish individual legend row identities.')
    ordered = sorted(templates,key=lambda t:names.index(t['id']))
    evidence = dict(templates=ordered,membership=np.stack([membership[n] for n in names]),
                    valid=np.ones((plot[3]-plot[1],plot[2]-plot[0]),bool))
    runtime._handoff(env,entries,names,plot,evidence,report)
    runtime._draw_templates(image,evidence,destination)
    runtime.group_backend = True
    runtime.active = True
    legend_runtime.diagnostics.update(downstream_marker_version=VERSION,
        scope='Colour-family ownership + pure line/symbol composition + full BW grid/window; typed segment correction')
    legend_runtime._save()
    print(f'[v46 colour groups] {len(groups)} groups; {len(points)} active; {len(pool)} suppressed hypotheses. GUI/Step-5 state saved.',flush=True)


def validate(state, image, ed):
    if state.get('version') != VERSION or state.get('image') != binding(image):
        raise ValueError('Colour-group correction image/version mismatch')
    from bw_step5_v46 import _box
    plot = _box(state['plot_box'],image.shape); legend = _box(state['legend_box'],image.shape)
    if list(ed['plot_area']) != [plot[0],plot[1],plot[2]-1,plot[3]-1]:
        raise ValueError('Colour-group correction plot ROI mismatch')
    if sorted(c['name'] for c in ed['curves']) != sorted(state['names']):
        raise ValueError('Colour-group correction legend identities mismatch')
    if sorted(n for g in state['groups'] for n in g['names']) != sorted(state['names']):
        raise ValueError('Invalid colour-group roster')
    if set(state['series_models']) != set(state['names']):
        raise ValueError('Missing colour-group target templates')
    for g in state['groups']:
        if not g['id'].startswith('G') or not g['id'][1:].isdigit():
            raise ValueError('Invalid group ID')
        unpack(g['gray'],image.shape[:2]); unpack(g['ignore'],image.shape[:2])
        active=[p for p in state['P_current'] if p['swatch_id'] in g['names']]
        pool=[p for p in state['S_current'] if p['swatch_id'] in g['names']]
        _normalise(active,pool,plot,legend,g['diameter'])
    if any(p['swatch_id'] not in state['names'] for p in state['P_current']+state['S_current']):
        raise ValueError('Unknown colour-group point identity')
    return state


def update_edit(ed, state):
    """Shape identity is independent of RGB; never join equal-colour curves."""
    for c in ed['curves']:
        name=c['name']; model=state['series_models'][name]
        ps=sorted([p for p in state['P_current'] if p['swatch_id']==name],key=lambda p:(p['cx'],p['cy']))
        metadata=('point_id','candidate_id','tentative','state','evidence_supported','original_detection','source')
        c.update(swatch_id=name,marker_class=model['class_name'],
                 points=[dict({k:p[k] for k in metadata if k in p},x=float(p['cx']),y=float(p['cy'])) for p in ps])
        c.pop('reconstruction',None)
        c['reconstruction']=dict(model='linear',anchors=deepcopy(c['points']),points=deepcopy(c['points']),
            status='typed_marker_chords_not_extracted_path')
    ed.update(detector_backend=VERSION,correction_backend='colour_group_bw_elements',correction_available=True)


def finalize_outputs(env):
    dest=Path(env['OUT_DIR'])
    state=json.loads((dest/STATE_FILE).read_text(encoding='utf-8'))
    ed=json.loads((dest/'edit_data.json').read_text(encoding='utf-8'))
    update_edit(ed,state);validate(state,env['img'],ed)
    save(dest/'edit_data.json',ed)
    from correction_session_v46 import draw_overlay
    draw_overlay(dest,env['img'],ed)


def correct_saved(folder,image,ed,iterations):
    """Resume frozen segment geometry per group without running detection."""
    from bw_step5_v46 import prepare_reference, key
    from group_element_correction_v46 import run
    from correction_session_v46 import draw_overlay
    folder=Path(folder)
    state=validate(json.loads((folder/STATE_FILE).read_text(encoding='utf-8')),image,ed)
    active=[]
    for c in ed['curves']:
        name=c['name']
        old=[p for p in state['P_current'] if p['swatch_id']==name]
        for i,p in enumerate(c['points']):
            exact=next((q for q in old if abs(q['cx']-p['x'])<1e-6 and abs(q['cy']-p['y'])<1e-6),None)
            q=deepcopy(exact or state['series_models'][name])
            q.update(cx=p['x'],cy=p['y'],swatch_id=name)
            if exact is None:
                q.update(point_id=f'{name}_manual_{i}',candidate_id=f'{name}_manual_{i}',
                         original_detection=False,manual_edit=True,confidence=0.)
            active.append(q)
    occupied={key(p) for p in active}
    pool=[p for p in state['S_current'] if key(p) not in occupied]
    after=[];after_pool=[]
    for g in state['groups']:
        original=[p for p in state['original_points'] if p['swatch_id'] in g['names']]
        if g['runtime'] is None:
            _,reference=prepare_reference(cv2.cvtColor(unpack(g['gray'],image.shape[:2]),cv2.COLOR_GRAY2BGR),
                state['plot_box'],state['legend_box'],original,g['diameter'])
            g['runtime']=dict(reference=reference,total_iterations=0,metric='segment_elements',evidence_prior_weight=.03)
        previous=g['runtime']
        if previous.get('metric')!='segment_elements':
            raise ValueError('Saved group correction metric differs; rerun initial detection')
        result=run([p for p in active if p['swatch_id'] in g['names']],
            [p for p in pool if p['swatch_id'] in g['names']],previous['reference']['reference_segments'],
            g['diameter'],state['plot_box'],state['legend_box'],max_iter=iterations,
            evidence_prior_weight=previous['evidence_prior_weight'],original_points=original)
        for row in result['trace']:
            row['iteration']+=previous['total_iterations']
        trace=dict(result,series=g['id'],iterations=result['trace'],metric='segment_elements',
            previous_iteration_count=previous['total_iterations'],
            stop_reason='unavailable_reference_preserved' if result.get('reason') else '')
        dest=folder/'group_step5'/g['id']/f'run_{state["total_correction_runs"]+1:03d}'
        dest.mkdir(parents=True,exist_ok=True);save(dest/'trace.json',trace)
        previous['total_iterations']+=len(result['trace'])
        print(f'[v46 group Step5] {g["id"]}: segment elements, {[r["action"] for r in result["trace"]]}; '
              f'{len(result["initial"])} -> {len(result["final"])} points (hypotheses, not ground truth)',flush=True)
        after.extend(result['final']);after_pool.extend(result['suppressed'])
    state.update(P_current=after,S_current=after_pool,total_correction_runs=state['total_correction_runs']+1)
    update_edit(ed,state)
    ed['correction']=dict(backend='colour_group_bw_elements',metric='segment_elements',detection_skipped=True,
                          hypotheses_are_not_measurements=True)
    save(folder/STATE_FILE,state);save(folder/'edit_data.json',ed)
    draw_overlay(folder,image,ed)

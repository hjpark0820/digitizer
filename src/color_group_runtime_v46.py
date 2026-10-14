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


def palette_routing(entries):
    """Use the same ink-identity relation as prepare_groups, not hue alone.

    Distinct nearby hues still compete in the general colour evidence model.
    Actual repeated inks retain the existing chart-wide grouped BW route;
    this does not mix correction engines within a saved chart.
    """
    from color_palette_identity_v46 import palette_relations
    same, near = palette_relations([e['rgb'][::-1] for e in entries], hue_family=True)
    repeated = [[j, i] for i in range(len(entries)) for j in range(i) if same[i, j]]
    rivals = [[j, i] for i in range(len(entries)) for j in range(i) if near[i, j] and not same[i, j]]
    return dict(policy='same_ink_palette_routing_v1',
                recommended_backend='color_group_bw' if repeated else 'color_marker_hybrid_v2',
                reason='repeated_legend_ink' if repeated else 'distinct_legend_inks',
                same_ink_pairs=repeated, distinct_near_hue_pairs=rivals,
                pair_indices='zero-based legend entry order',
                scope='whole chart; actual repeated inks retain grouped BW')


def ambiguous_palette(entries):
    """Compatibility boolean: only repeated legend inks require grouped BW."""
    return bool(palette_routing(entries)['same_ink_pairs'])


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


def validated_legend_groups(image, plot, legend, entries, names, observation_policy='legacy',
                            *, return_availability=False):
    """Retain every GUI identity, with explicit expected measurement failures.

    Legacy callers keep the strict two-value contract. The runtime requests
    availability and may process the measured subset without renumbering it.
    Invalid or contradictory identities remain errors in either mode.
    """
    from color_group_bw_v46 import prepare_groups
    from color_legend_composition_v46 import full_swatch_box
    if (not entries or len(entries)!=len(names) or len({e['id'] for e in entries})!=len(entries)
            or len(set(names))!=len(names) or any(not isinstance(n,str) or not n for n in names)):
        raise ValueError('Invalid validated colour legend identities')
    swatches=[]
    for entry in entries:
        if not entry.get('marker_template',True):
            raise ValueError(f"Colour-group marker route cannot use line-only key {entry['id']}")
        # Keep the full-source line/body rectangle when the colour-only
        # component was just a fragment (black body on a coloured connector).
        # This changes geometry input, not the accepted roster or series IDs.
        box=entry.get('source_structure_box') or (entry.get('report') or {}).get('source_structure_box')
        if box is None:
            box=(entry.get('composition') or {}).get('source_swatch_box')
        if box is None:
            box=full_swatch_box(image,entry,legend)
        swatches.append((None,box))  # Do not force a supplied shape class.
    groups,diagnostics=prepare_groups(image,plot,legend,group_policy='hue_family',composition=True,
        ownership_policy='uncertainty_v1',grid_evidence='confirmed',
        palette_recovery='calibrate_complete',swatches=swatches,observation_policy=observation_policy,
        identity_entries=entries,neutral_geometry_policy='shared_source')
    templates=[t for g in groups for t in g['templates']]
    # Explicit extraction IDs are assigned before any per-entry failure or
    # colour grouping. Do not renumber survivors or silently relabel a series.
    expected={f'S{i+1:02}':names[i] for i in range(len(entries))}
    observed=[t.key for t in templates]
    if len(set(observed))!=len(observed) or not set(observed)<=set(expected):
        raise ValueError('BW returned duplicate or unknown validated colour legend identities')
    missing=[dict(gui_id=e['id'],name=names[i],extraction_id=f'S{i+1:02}',extraction=next(
        (r for r in diagnostics if r.get('swatch_id')==f'S{i+1:02}'),None))
        for i,e in enumerate(entries) if f'S{i+1:02}' not in observed]
    if missing and not return_availability:
        raise ValueError(f'Validated colour legend could not be measured by BW: {missing}')
    if any(not q['extraction'] or q['extraction'].get('extraction_status')!='failed' for q in missing):
        raise ValueError('Missing colour legend identity has no expected extraction-failure diagnostic')
    # Geometry remains a consistency check, not the source of row identity.
    indices=[i for i in range(len(entries)) if f'S{i+1:02}' in observed]
    active_mapping={f'S{i+1:02}':names[i] for i in indices}
    if templates and match_rows(templates,[entries[i] for i in indices],
                                [names[i] for i in indices])!=active_mapping:
        raise ValueError('BW template centre disagrees with its validated legend identity')
    for g in groups:
        if len(g['templates'])!=len(g['reports']):
            raise ValueError('Colour-group templates and reports are not aligned')
        for t,r in zip(g['templates'],g['reports']):
            if r.get('swatch_id',t.key)!=t.key:
                raise ValueError('Colour-group template/report identities disagree')
            i=int(t.key[1:])-1
            r.update(gui_legend_id=entries[i]['id'],gui_curve_name=names[i],
                     legend_roster_source='validated_colour_legend')
    active_names=[names[i] for i in indices]
    unavailable=[]
    for q in missing:
        extraction=q['extraction']
        unavailable.append(dict(q,status='unavailable',template_available=False,
            correction_available=False,stage='legend_measurement',
            reason_code=extraction.get('extraction_error_code','legend_measurement_failed'),
            reason=extraction.get('extraction_error','Observed legend marker could not be measured.')))
    by_name={q['name']:q for q in unavailable}
    statuses={n:deepcopy(by_name[n]) if n in by_name else dict(name=n,gui_id=entries[i]['id'],
        extraction_id=f'S{i+1:02}',status='available',template_available=True,correction_available=True)
        for i,n in enumerate(names)}
    availability=dict(status='completed' if not unavailable else 'partial' if active_names else 'unavailable',
        active_names=active_names,unavailable_series=unavailable,series_status=statuses,
        correction_available=bool(active_names),availability_policy='explicit_legend_subset_v1')
    return (groups,expected,availability) if return_availability else (groups,expected)


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


def detection_policies(environ=None):
    """Reviewed defaults, with explicit legacy comparison still available.

    Selecting legacy observations alone also selects legacy identity; an
    explicitly contradictory pair is rejected rather than silently changed.
    """
    if environ is None:
        import os
        environ=os.environ
    observation=environ.get('BW_V46_COLOUR_OBSERVATION','source_roles')
    if observation not in ('legacy','source_roles'):
        raise ValueError('BW_V46_COLOUR_OBSERVATION must be legacy or source_roles')
    identity=environ.get('BW_V46_GROUP_RELATIVE_IDENTITY',
                         'common_scene' if observation=='source_roles' else 'off')
    if identity not in ('off','common_scene'):
        raise ValueError('BW_V46_GROUP_RELATIVE_IDENTITY must be off or common_scene')
    if identity!='off' and observation!='source_roles':
        raise ValueError('Group relative identity requires source_roles observations')
    return observation,identity


def detect(runtime, env, legend_runtime, plot, legend, report):
    from color_group_bw_v46 import detection_options
    from bw_pipeline_v46 import detect_points
    from x_singleton_suppressed_v46 import generate

    image = env['img']; destination = Path(env['OUT_DIR'])
    entries = legend_runtime.entries
    names = [f'color{i+2:02d}' for i in range(len(entries))]
    # User-reviewed Opicinumab route: retain pale source ink, then compare
    # group identities on one observation with boundary-verified raster scale.
    observation_policy,relative_identity_policy=detection_policies()
    groups, mapping, availability = validated_legend_groups(image,plot,legend,entries,names,
        observation_policy=observation_policy,return_availability=True)
    report['legend_roster_source']='validated_colour_legend'
    state = dict(version=VERSION, image=binding(image), plot_box=list(plot), legend_box=list(legend),
                 names=names, groups=[], series_models={}, total_correction_runs=0,
                 observation_policy=observation_policy,relative_identity_policy=relative_identity_policy,
                 legend_roster_source='validated_colour_legend',
                 legend_bindings=[dict(gui_id=e['id'],name=n) for e,n in zip(entries,names)])
    state.update(deepcopy(availability))
    recovery=groups[0].get('palette_recovery',{}) if groups else {}
    state['palette_recovery']=deepcopy(recovery)
    save(destination/'palette_recovery.json',recovery)
    if groups:
        cv2.imwrite(str(destination/'palette_recovery_phase.png'),groups[0]['recovery_phase'])
    points, pool, templates, membership, group_reports = [], [], [], {}, []
    for g in groups:
        dest = destination/'colour_groups'/g['id']; dest.mkdir(parents=True, exist_ok=True)
        options = detection_options(g)
        result = detect_points(g['image'], plot, legend,
            prepared_templates=(g['templates'],g['reports']),window_image=g['window_image'],
            window_occlusion_mask=g['occlusion'],window_uncertainty_mask=g['uncertainty'],**options)
        observation=g.get('colour_observation')
        relative_report={'status':'disabled'}
        if relative_identity_policy=='common_scene' and observation is not None:
            from color_group_relative_identity_v46 import refine
            result,relative_report=refine(image,g,result,plot,legend)
        reference_gray=(cv2.cvtColor(g['window_image'],cv2.COLOR_BGR2GRAY) if observation is None else observation.reference_gray())
        save(dest/'detection.json', {k:v for k,v in result.items() if k != 'diag_steps'})
        cv2.imwrite(str(dest/'confirmed_gray.png'),reference_gray)
        if observation is not None:
            save(dest/'source_observation.json',observation.report)
            for role in ('own','other','paper'):
                cv2.imwrite(str(dest/f'role_{role}.png'),np.uint8(255*getattr(observation,role)))
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
            source_body=getattr(t,'source_colour_body',None)
            if source_body:
                state['series_models'][name]['legend_body_box']=list(source_body['source_box'])
                state['series_models'][name]['legend_body_provenance']=deepcopy(r.get('source_colour_body',{}))
            membership[name] = 1-reference_gray[plot[1]:plot[3],plot[0]:plot[2]]/255.
        state['groups'].append(dict(id=g['id'], names=group_names, diameter=float(result['d_est']),
            gray=_pack_reference(reference_gray),
            ignore=_pack_reference(np.uint8(g['occlusion']>.5)), runtime=None,
            palette_competitor_ids=g.get('palette_competitor_ids'),
            unavailable_palette_ids=g.get('unavailable_palette_ids'),
            source_observation=(observation.report if observation is not None else None),
            role_maps=({role:_pack_reference(np.uint8(255*getattr(observation,role))) for role in ('own','other','paper')} if observation is not None else None)))
        group_reports.append(dict(id=g['id'], names=group_names, active=len(result['kept']),
            suppressed=len(result['suppressed']), options={k:v for k,v in options.items() if k!='colour_observation'},
            relative_identity=relative_report,
            source_observation=(observation.report if observation is not None else None),
            colour_visibility_policy=result['diagnostics'].get('colour_visibility_policy'),
            palette_identity=g.get('palette_identity'),
            palette_competitor_ids=g.get('palette_competitor_ids'),
            unavailable_palette_ids=g.get('unavailable_palette_ids'),
            composition=[dict(id=mapping[t.key], policy=r.get('template_policy'),
                              model=r.get('composition',{}).get('best_model_name')) for t,r in zip(g['templates'],g['reports'])]))
    # One frozen plot-wide pool, including targets in other colour groups.
    if state['series_models']:
        pool, overlap = generate(points,pool,state['series_models'],plot,legend)
    else:
        overlap=dict(status='unavailable',reason='no_measured_target_templates',added=[])
    state.update(P_current=points,S_current=pool,overlap=overlap,original_points=deepcopy(points))
    # Neutral groups have no chromatic relative-identity pass. Recheck their
    # observed marker bodies against original-source connectors, not a binary
    # group mask that treats every nearby black stroke as marker evidence.
    from color_neutral_step5_v46 import review_neutral_detections
    if state['series_models']:
        points,pool,neutral_report=review_neutral_detections(image,state,points,pool)
    else:
        neutral_report=dict(status='unavailable',reason='no_measured_target_templates')
    state.update(P_current=points,S_current=pool,original_points=deepcopy(points),
                 neutral_body_review=neutral_report)
    save(destination/'neutral_body_review.json',neutral_report)
    for g in group_reports:
        g['active_before_neutral_review']=g['active']
        g['active']=sum(p['swatch_id'] in g['names'] for p in points)
        g['suppressed']=sum(p['swatch_id'] in g['names'] for p in pool)
    save(destination/STATE_FILE,state)
    converted = [dict(p, series_id=p['swatch_id'],x_px=p['cx'],y_px=p['cy'],
                      score=float(p.get('confidence',0.)),candidate_id=p.get('point_id',f'group_{i}'))
                 for i,p in enumerate(points)]
    report.update(deepcopy(availability))
    report.update(version=VERSION,same_colour_bw_v3=True,
        template_errors=[dict(series_id=q['name'],gui_id=q['gui_id'],extraction_id=q['extraction_id'],
            reason_code=q['reason_code'],error=q['reason']) for q in availability['unavailable_series']],
        selection_config=None,guide_policy='group_bw_native',
        ownership_policy=('source_rgb_colour_roles_v1' if observation_policy=='source_roles' else 'uncertainty_v1'),
        relative_identity_policy=relative_identity_policy,
        grid_evidence=('original_source_edges_with_colour_roles' if observation_policy=='source_roles' else 'confirmed'),
        backend=VERSION,groups=group_reports,points=converted,uncertain_points=pool,
        palette_recovery=recovery,neutral_body_review=neutral_report,
        templates=[{k:v for k,v in t.items() if k!='soft'} for t in templates],
        scale_search=True,scale_policy='shared_symbol',
        scale_scope='per legend series identity; shared by grid and window',
        correction_backend='colour_group_bw_elements',step5_state_file=STATE_FILE,
        overlap=overlap,triangle_errorbar_guard=dict(status='native_bw_window_verification',enabled=False),
        mask_note='Distinct source inks retain brightness/saturation before BW matching. '
                  'Truly same-ink shapes share a group mask, not a colour-only series identity.')
    ordered = sorted(templates,key=lambda t:names.index(t['id']))
    evidence = (dict(templates=ordered,membership=np.stack([membership[t['id']] for t in ordered]),
                    valid=np.ones((plot[3]-plot[1],plot[2]-plot[0]),bool)) if ordered else None)
    runtime._handoff(env,entries,names,plot,evidence,report)
    runtime._draw_templates(image,evidence,destination)
    runtime.group_backend = True
    runtime.active = True
    legend_runtime.diagnostics.update(downstream_marker_version=VERSION,
        downstream_marker_status=availability['status'],
        unavailable_marker_series=len(availability['unavailable_series']),
        scope='Colour-family ownership + pure line/symbol composition + full BW grid/window; typed segment correction')
    legend_runtime._save()
    print(f'[v46 colour groups] {availability["status"]}: {len(groups)} groups; '
          f'{len(availability["active_names"])}/{len(names)} measured series; '
          f'{len(points)} active; {len(pool)} suppressed hypotheses. GUI/Step-5 state saved.',flush=True)


def validate(state, image, ed):
    if state.get('version') != VERSION or state.get('image') != binding(image):
        raise ValueError('Colour-group correction image/version mismatch')
    from bw_step5_v46 import _box
    plot = _box(state['plot_box'],image.shape); legend = _box(state['legend_box'],image.shape)
    if list(ed['plot_area']) != [plot[0],plot[1],plot[2]-1,plot[3]-1]:
        raise ValueError('Colour-group correction plot ROI mismatch')
    names=state['names']
    if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
        raise ValueError('Invalid colour-group legend identities')
    if sorted(c['name'] for c in ed['curves']) != sorted(names):
        raise ValueError('Colour-group correction legend identities mismatch')
    active_names=state.get('active_names',names)
    if len(set(active_names))!=len(active_names) or not set(active_names)<=set(names):
        raise ValueError('Invalid active colour-group identities')
    if 'active_names' in state:
        if state.get('availability_policy')!='explicit_legend_subset_v1':
            raise ValueError('Unsupported colour-group availability policy')
        unavailable=state.get('unavailable_series',[])
        missing=[q['name'] for q in unavailable]
        if len(set(missing))!=len(missing) or set(missing)&set(active_names) or set(missing)|set(active_names)!=set(names):
            raise ValueError('Invalid colour-group availability partition')
        statuses=state.get('series_status',{})
        if set(statuses)!=set(names):
            raise ValueError('Missing colour-group per-series availability')
        bindings=state.get('legend_bindings',[])
        if ([q['name'] for q in bindings]!=names or len({q['gui_id'] for q in bindings})!=len(names)):
            raise ValueError('Invalid colour-group legend bindings')
        for i,name in enumerate(names):
            row=statuses[name];usable=name in active_names
            if (row.get('name')!=name or row.get('gui_id')!=bindings[i]['gui_id']
                    or row.get('extraction_id')!=f'S{i+1:02}'
                    or row.get('status')!=('available' if usable else 'unavailable')
                    or row.get('template_available') is not usable
                    or row.get('correction_available') is not usable):
                raise ValueError('Contradictory colour-group series availability')
        if any(q!=statuses[q['name']] or not q.get('reason_code') for q in unavailable):
            raise ValueError('Invalid unavailable colour-group diagnostic')
        outcome='completed' if not missing else 'partial' if active_names else 'unavailable'
        if state.get('status')!=outcome or state.get('correction_available') is not bool(active_names):
            raise ValueError('Contradictory colour-group completion status')
    if sorted(n for g in state['groups'] for n in g['names']) != sorted(active_names):
        raise ValueError('Invalid colour-group roster')
    if set(state['series_models']) != set(active_names):
        raise ValueError('Missing colour-group target templates')
    # Source-body coordinates are correction evidence, not editable display
    # metadata. Validate them before NumPy slicing can wrap negative indices
    # or silently truncate a forged/fractional imported crop.
    for model in state['series_models'].values():
        if 'legend_body_box' not in model:
            continue  # Existing complete sessions predate source-body recovery.
        def source_box(value):
            try:coords=np.asarray(value,dtype=float)
            except (TypeError,ValueError):
                raise ValueError('Invalid saved source-body coordinates') from None
            if (coords.shape!=(4,) or not np.isfinite(coords).all()
                    or np.any(coords!=np.rint(coords))):
                raise ValueError('Invalid saved source-body coordinates')
            return _box(coords,image.shape)
        body=source_box(model['legend_body_box'])
        swatch=source_box(model.get('legend_box'))
        provenance=model.get('legend_body_provenance') or {}
        crop=source_box(provenance.get('source_crop_box'))
        if (provenance.get('version')!='source_colour_body_with_context_v1'
                or provenance.get('source_pixels_modified') is not False
                or provenance.get('marker_box')!=list(body)
                or not (crop[0]<=body[0]<body[2]<=crop[2] and crop[1]<=body[1]<body[3]<=crop[3])
                or not (swatch[0]<=(body[0]+body[2]-1)/2<swatch[2]
                        and swatch[1]<=(body[1]+body[3]-1)/2<swatch[3])):
            raise ValueError('Saved source body disagrees with its legend provenance')
    if len({g['id'] for g in state['groups']})!=len(state['groups']):
        raise ValueError('Duplicate colour-group IDs')
    for g in state['groups']:
        if not g['id'].startswith('G') or not g['id'][1:].isdigit():
            raise ValueError('Invalid group ID')
        unpack(g['gray'],image.shape[:2]); unpack(g['ignore'],image.shape[:2])
        active=[p for p in state['P_current'] if p['swatch_id'] in g['names']]
        pool=[p for p in state['S_current'] if p['swatch_id'] in g['names']]
        _normalise(active,pool,plot,legend,g['diameter'])
    if any(p['swatch_id'] not in active_names for p in
           state['P_current']+state['S_current']+state.get('original_points',[])):
        raise ValueError('Unknown colour-group point identity')
    return state


def update_edit(ed, state):
    """Shape identity is independent of RGB; never join equal-colour curves."""
    active_names=state.get('active_names',list(state['series_models']))
    for c in ed['curves']:
        name=c['name']
        if name not in active_names:
            # No automatic model exists for this identity. Keep coordinates
            # entered by the user; correction must never manufacture a model
            # for those points or silently delete the manual-only row.
            unavailable=deepcopy(state.get('series_status',{}).get(name,{}))
            c.update(swatch_id=name,detection_status='unavailable',correction_available=False,
                     extraction_diagnostic=unavailable,
                     correction_unavailable_reason=unavailable.get('reason','Legend marker unavailable.'))
            c.pop('marker_class',None)
            c['reconstruction']=dict(model='linear',anchors=deepcopy(c['points']),points=deepcopy(c['points']),
                status='manual_only_unavailable_legend_model')
            continue
        model=state['series_models'][name]
        ps=sorted([p for p in state['P_current'] if p['swatch_id']==name],key=lambda p:(p['cx'],p['cy']))
        metadata=('point_id','candidate_id','tentative','state','evidence_supported','original_detection','source',
                  'confidence','confidence_kind','marker_confidence','detector_confidence','detector_confidence_kind')
        c.update(swatch_id=name,marker_class=model['class_name'],detection_status='available',correction_available=True,
                 points=[dict({k:p[k] for k in metadata if k in p},x=float(p['cx']),y=float(p['cy'])) for p in ps])
        c.pop('correction_unavailable_reason',None)
        c.pop('extraction_diagnostic',None)
        c.pop('reconstruction',None)
        c['reconstruction']=dict(model='linear',anchors=deepcopy(c['points']),points=deepcopy(c['points']),
            status='typed_marker_chords_not_extracted_path')
    ed.update(detector_backend=VERSION,correction_backend='colour_group_bw_elements',
        correction_available=bool(active_names),detection_status=state.get('status','completed'),
        active_names=list(active_names),unavailable_series=deepcopy(state.get('unavailable_series',[])),
        series_status=deepcopy(state.get('series_status',{})),
        correction_scope='measured_series_only')
    if active_names:
        ed.pop('correction_unavailable_reason',None)
    else:
        ed['correction_unavailable_reason']='No legend marker model is available; saved series support manual editing only.'


def finalize_outputs(env):
    dest=Path(env['OUT_DIR'])
    state=json.loads((dest/STATE_FILE).read_text(encoding='utf-8'))
    ed=json.loads((dest/'edit_data.json').read_text(encoding='utf-8'))
    update_edit(ed,state);validate(state,env['img'],ed)
    save(dest/'edit_data.json',ed)
    from correction_session_v46 import draw_overlay
    draw_overlay(dest,env['img'],ed)


def ensure_group_reference(group,gray,plot,legend,original):
    """Upgrade only the reference, never restore/delete any saved markers.

    A pre-policy session still contains its frozen group pixels, so migration
    does not need redetection. Unknown future policies fail instead of silently
    downgrading. Preserve iteration counters and explicit objective settings.
    """
    from bw_step5_v46 import prepare_reference
    from bw_dash_reference_v46 import VERSION as reference_version
    runtime=group.get('runtime')
    if runtime is not None and runtime.get('metric')!='segment_elements':
        raise ValueError('Saved group correction metric differs; rerun initial detection')
    policy=runtime['reference'].get('reference_policy') if runtime is not None else None
    if policy not in (None,reference_version):
        raise ValueError('Saved group reference policy is newer or incompatible')
    if policy==reference_version:
        return runtime
    _,reference=prepare_reference(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),plot,legend,
                                  original,group['diameter'],recover_dashes=True)
    if runtime is None:
        runtime=dict(reference=reference,total_iterations=0,metric='segment_elements',evidence_prior_weight=.03)
    else:
        runtime=deepcopy(runtime)
        runtime['reference_upgrade']=dict(from_policy='legacy_raw_segments',to_policy=reference_version,
            previous_segment_count=len(runtime['reference']['reference_segments']),
            new_segment_count=len(reference['reference_segments']),points_unchanged=True)
        runtime['reference']=reference
    group['runtime']=runtime
    return runtime


def correct_saved(folder,image,ed,iterations):
    """Resume frozen segment geometry per group without running detection."""
    from bw_step5_v46 import key
    from group_element_correction_v46 import run
    from correction_session_v46 import draw_overlay
    folder=Path(folder)
    state=validate(json.loads((folder/STATE_FILE).read_text(encoding='utf-8')),image,ed)
    active_names=state.get('active_names',state['names'])
    if not active_names:
        raise ValueError('Colour-group Step 5 is unavailable: no measured target templates; manual points are preserved')
    active=[]
    for c in ed['curves']:
        name=c['name']
        if name not in active_names:
            continue
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
    import os
    from color_neutral_step5_v46 import routing, run as run_joint_neutral
    policy=os.environ.get('BW_V46_GROUP_STEP5','auto')
    if policy not in ('auto','legacy','joint_gray'):
        raise ValueError('BW_V46_GROUP_STEP5 must be auto, legacy or joint_gray')
    route=routing(image,state,active)
    use_joint=policy=='joint_gray' or (policy=='auto' and (route['enabled'] or state.get('joint_neutral_runtime') is not None))
    if state.get('joint_neutral_runtime') is not None and policy=='legacy':
        raise ValueError('Cannot resume a joint neutral session with the legacy group objective')
    state['step5_connector_route']=dict(route,requested_policy=policy,selected='joint_gray' if use_joint else 'group_gray')
    if use_joint:
        run_joint_neutral(image,state,active,pool,iterations,folder,route)
        update_edit(ed,state)
        ed['correction']=dict(backend='colour_group_bw_joint_neutral',metric='joint_original_gray_segments',
            detection_skipped=True,colour_ids_preserved=True,hypotheses_are_not_measurements=True)
        save(folder/STATE_FILE,state);save(folder/'edit_data.json',ed)
        draw_overlay(folder,image,ed)
        return
    after=[];after_pool=[]
    for g in state['groups']:
        original=[p for p in state['original_points'] if p['swatch_id'] in g['names']]
        gray=unpack(g['gray'],image.shape[:2])
        ignore=unpack(g['ignore'],image.shape[:2])
        previous=ensure_group_reference(g,gray,state['plot_box'],state['legend_box'],original)
        own_active=[p for p in active if p['swatch_id'] in g['names']]
        own_pool=[p for p in pool if p['swatch_id'] in g['names']]
        from color_neutral_step5_v46 import MarkerEvidence
        # Use frozen original colour roles and legend bodies, not line geometry
        # or old confidence alone, for every proposed ADD/DELETE/REPLACE.
        local_state=dict(state,groups=[g])
        marker_evidence=MarkerEvidence(image,local_state,own_active+own_pool,
            previous['reference']['reference_segments'],duplicate_policy='body')
        result=run(own_active,own_pool,previous['reference']['reference_segments'],
            g['diameter'],state['plot_box'],state['legend_box'],max_iter=iterations,
            evidence_prior_weight=previous['evidence_prior_weight'],original_points=original,
            source_gray=gray,source_ignore=ignore,action_guard=marker_evidence.guard)
        for row in result['trace']:
            row['iteration']+=previous['total_iterations']
        trace=dict(result,series=g['id'],iterations=result['trace'],metric='segment_elements',
            reference_policy=previous['reference']['reference_policy'],
            endpoint_evidence_policy='observed_endpoint_missing_reference_v1',
            marker_action_policy='frozen_source_colour_marker_evidence_v1',
            marker_models=marker_evidence.reports,
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

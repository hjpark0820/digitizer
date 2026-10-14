"""Colour ownership adapter for the complete production BW engine.

The colour runtime selects this backend for ambiguous palettes. Boxes stay in native
source coordinates. The source image is never edited. Colour only selects ink;
native BW templates, voting, window verification and typed correction are reused.
"""
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np

from bw_legend_v46 import extract_legend_models, LegendExtractionError
from bw_pipeline_v46 import detect_points, production_detection_options, LEGACY_PROPOSAL_SCALES
from bw_step5_v46 import run_correction, save
from color_marker_evidence import _swatch_model, _membership
from color_palette_identity_v46 import (palette_relations, distinguish_inks, mixing_family_relation, legend_entry_inputs,
                                       MIXING_THIN_PX, MIXING_SOLID_PX)


def prepare_groups(image, plot_area, legend_area, colour_distance=14., group_policy='lab', composition=False,
                   ownership_policy='legacy', grid_evidence='confirmed', palette_recovery='off',
                   swatches=None, neutral_geometry_policy='separate', observation_policy='legacy',
                   identity_entries=None):
    # Shared neutral observations are diagnostic opt-in until shape/tone
    # competition in the grouped BW route is validated. Merely exposing more
    # ink can increase wrong-series assignments from unresolved tiny keys.
    if neutral_geometry_policy not in ('separate','shared_source'):
        raise ValueError('Unknown neutral geometry policy')
    if observation_policy not in ('legacy','source_roles'):
        raise ValueError('Unknown colour observation policy')
    if ownership_policy not in ('legacy','uncertainty_v1'):
        raise ValueError('Unknown colour ownership policy')
    if grid_evidence not in ('confirmed','raw_proposal'):
        raise ValueError('Unknown grid colour evidence policy')
    if group_policy not in ('lab','hue_family'):raise ValueError('Unknown colour grouping policy')
    # GUI callers supply their accepted key boxes in identity order. BW still
    # measures shape from original pixels, but must not rediscover a different
    # series roster (e.g. admit label text or drop a real neutral key).
    try:
        templates,reports=extract_legend_models(image,legend_area,swatches=swatches)
    except LegendExtractionError as error:
        # Valid source entries can all be unresolved. The accepted roster and
        # measured inks still matter; never invent a model just to keep going.
        templates,reports=[],error.reports
    palette={}
    source_templates={t.key:t for t in templates}
    for r in reports:
        if 'box' not in r and r['swatch_id'] in source_templates:
            r['box']=list(source_templates[r['swatch_id']].swatch_box)
        try:palette[r['swatch_id']]=_swatch_model(image,tuple(r['box']))
        except ValueError as error:
            if str(error) not in ('Swatch has no distinguishable ink',
                                   'Legend colour is indistinguishable from its paper'):
                raise
            r['ink_measurement_error']=str(error)
        r['ink_model_available']=r['swatch_id'] in palette
    from color_source_legend_v46 import recover_templates
    templates,reports=recover_templates(image,legend_area,templates,reports,palette)
    unmeasured={t.key for t in templates if t.key not in palette}
    if unmeasured:
        templates=[t for t in templates if t.key not in unmeasured]
        for r in reports:
            if r['swatch_id'] in unmeasured:
                r.update(extraction_status='failed',template_available=False,
                    extraction_error_code='unmeasurable_legend_ink',
                    extraction_error=r.get('ink_measurement_error','No distinguishable legend ink'))
    # The extractor retains failed entry reports. Keep model-indexed arrays
    # aligned by ID, while returning the complete diagnostic list to callers.
    all_reports=reports
    by_id={r['swatch_id']:r for r in reports}
    reports=[by_id[t.key] for t in templates]
    decomposition=[]
    if composition:
        from bw_composed_legend_v46 import compose_templates
        # Source-colour recovery already measured a body and its crossing
        # nuisance. Re-composing the contaminated full swatch would reimport
        # the frame/axis and discard that measured source geometry.
        composed=[];composed_reports=[]
        for t,r in zip(templates,reports):
            if r.get('source_colour_body'):
                composed.append(t);composed_reports.append(r);decomposition.append({})
            else:
                ts,rs,ds=compose_templates(image,legend_area,[t],[r])
                composed.extend(ts);composed_reports.extend(rs);decomposition.extend(ds)
        templates,reports=composed,composed_reports
    if not templates:return [],all_reports
    # Colour identity is measured for the FULL accepted roster, independent
    # of shape availability. An unresolved blue key must remain a negative
    # competitor rather than donating its pixels to a surviving dark key.
    palette_ids=list(palette)
    models=[palette[key] for key in palette_ids]
    same,_=palette_relations([m['bgr'] for m in models],colour_distance,
                             hue_family=group_policy=='hue_family')
    # The summary relation above stays exactly as it is. A thin key (ring,
    # outline, cross) measures darker-and-desaturated or paler than the same
    # ink in a solid key, so the summary cannot pair it with its solid twin;
    # the physical paper/ink mixing test (mixing_family_relation) may ADD
    # such a pair between two summary singletons. It never splits a summary
    # pair and never adds a member to a summary group. Different neutral
    # inks stay separate unless neutral_geometry_policy shares them (below).
    mixing_report=None  # callers without the legend entries keep the summary relation
    if identity_entries is not None:
        # Measure the legend's own colour keys (entry box + entry RGB), the
        # inputs the routing decision used; BW boxes would add connector
        # stubs and a re-estimated key colour.
        entry_models,entry_boxes=legend_entry_inputs(image,identity_entries)
        # Swatch IDs S01.. follow the accepted entry order.
        entry_index=[int(key[1:])-1 if str(key)[:1]=='S' and str(key)[1:].isdigit() else -1
                     for key in palette_ids]
        if (len(set(entry_index))==len(entry_index) and
                all(0<=e<len(entry_models) for e in entry_index)):
            summary=np.eye(len(entry_models),dtype=bool)
            summary[np.ix_(entry_index,entry_index)]=same
            related,_,mixing_report=mixing_family_relation(image,entry_models,entry_boxes,
                colour_distance=colour_distance,hue_family=group_policy=='hue_family',summary=summary)
            same=related[np.ix_(entry_index,entry_index)]
    def compatible(i,j):
        # Neutral antialias shades share one BW observation plane. This is
        # NOT a claim that black and gray are the same ink: original grayscale
        # and per-series fill descriptors retain tone for later identity tests.
        # Hard colour argmax among neutral shades erases black outlines from
        # keys whose small legend happened to be measured as pale gray.
        return bool(same[i,j] or (neutral_geometry_policy=='shared_source' and
                                 models[i]['achromatic'] and models[j]['achromatic']))
    palette_groups=[]
    for i in range(len(models)):
        g=next((g for g in palette_groups if all(compatible(i,j) for j in g)),None)
        if g is None:palette_groups.append([i])
        else:g.append(i)
    template_index={t.key:i for i,t in enumerate(templates)}
    groups=[];blocked=set()
    for members in palette_groups:
        missing=[palette_ids[i] for i in members if palette_ids[i] not in template_index]
        active=[template_index[palette_ids[i]] for i in members if palette_ids[i] in template_index]
        if missing and active:
            # Colour cannot distinguish a missing shape from an available
            # same-ink shape. Until joint unknown-shape competition exists,
            # suspend this group's automatic assignment rather than force it.
            for ti in active:
                blocked.add(templates[ti].key)
                reports[ti].update(extraction_status='failed',template_available=False,
                    extraction_error_code='unavailable_same_ink_rival',
                    extraction_error='Same-ink legend rival has no valid shape model',
                    unavailable_rival_ids=missing)
            active=[]
        groups.append(active)
    if not any(groups):
        updated={r['swatch_id']:r for r in reports}
        return [],[updated.get(r['swatch_id'],r) for r in all_reports]
    group_models=[];fits=[];strengths=[]
    for indices in palette_groups:
        model=deepcopy(models[indices[0]])
        model['bgr']=np.median([models[i]['bgr'] for i in indices],axis=0)
        if group_policy=='hue_family' or (neutral_geometry_policy=='shared_source' and
                                        all(models[i]['achromatic'] for i in indices)):
            measurements=[_membership(image,models[i]) for i in indices]
            membership=np.max([m for m,f in measurements],axis=0)
            fit=np.max([f for m,f in measurements],axis=0)
        else:membership,fit=_membership(image,model)
        group_models.append(model);fits.append(fit);strengths.append(membership)
    fits=np.stack(fits);strengths=np.stack(strengths)
    # A group's identity is the set of inks its members actually use. When a
    # group holds a SOLID key and a THIN key (vivid fill vs thin ring/outline,
    # thickness measured by the mixing test's glyph profile), the median key
    # is a summary no member renders, so paper-mixture fits are taken against
    # every member key. Members that render the ink alike (all solid or all
    # thin) keep the median: there it IS the ink up to measurement noise.
    member_models=None
    if mixing_report is not None and mixing_report.get('keys'):
        thickness={p:mixing_report['keys'][e].get('thickness') for p,e in enumerate(entry_index)}
        def rendering(i):
            t=thickness.get(i)
            return None if t is None else 'thin' if t<=MIXING_THIN_PX else 'solid' if t>=MIXING_SOLID_PX else None
        member_models=[[models[i] for i in indices] if {'thin','solid'}<={rendering(i) for i in indices} else []
                       for indices in palette_groups]
    # Distinct inks must not be collapsed again by hue-only membership.
    # Groups were formed with a complete-link identity test; all remaining
    # groups compete as separate inks, even if their median RGBs are close.
    strengths,fits,identity_report=distinguish_inks(image,group_models,strengths,fits,
                                                  equivalent=np.eye(len(groups),dtype=bool),
                                                  member_models=member_models)
    identity_report['mixing_family']=mixing_report
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
    x0,y0,x1,y1=map(int,plot_area)
    plot=np.zeros(gray.shape,bool);plot[y0:y1,x0:x1]=True
    from color_palette_recovery_v46 import recover
    eligible=plot.copy()
    lx0,ly0,lx1,ly1=map(int,legend_area)
    eligible[ly0:ly1,lx0:lx1]=False
    diameters=[float(np.median([getattr(templates[i],'diameter',10.) for i in indices])) if indices else
               float(np.median([max(3.,min(by_id[palette_ids[j]]['box'][2]-by_id[palette_ids[j]]['box'][0],
                   by_id[palette_ids[j]]['box'][3]-by_id[palette_ids[j]]['box'][1])) for j in palette_groups[gi]]))
               for gi,indices in enumerate(groups)]
    strengths,fits,recovery_report,recovery_phase=recover(
        image,group_models,strengths,fits,eligible,diameters,policy=palette_recovery)
    winner=fits.argmax(axis=0)
    neutral=(np.ptp(image.astype(float),axis=2)<18)&(gray<190)
    from color_pixel_ownership_v46 import ownership_maps
    separated = (ownership_maps(image,fits,strengths,[not m['achromatic'] for m in group_models])
                 if ownership_policy=='uncertainty_v1' else None)
    observations=None
    if observation_policy=='source_roles':
        from color_source_observation_v46 import build_observations
        observations=build_observations(image,group_models,groups,templates,
            group_hollow=[bool(indices) and all(templates[i].marker_kind=='open' for i in indices) for indices in groups],
            member_models=member_models)
    output=[]
    for gi,indices in enumerate(groups):
        if not indices:continue
        # Preserve the original darkness (including antialiasing), not an
        # idealised black disk. Fit controls ownership, not observed intensity.
        ownership=(winner==gi)*np.clip((fits[gi]-.10)/.25,0,1)
        canvas=np.rint(255-(255-gray)*ownership).astype(np.uint8)
        confirmed=canvas.copy();uncertainty=np.zeros(gray.shape,np.float32)
        if separated is not None:
            canvas=separated[gi]['proposal']
            confirmed=separated[gi]['confirmed']
            uncertainty=separated[gi]['uncertainty']*plot
        other=np.zeros(gray.shape,np.float32)
        for j in range(len(groups)):
            if j!=gi:
                # Use the SAME ownership confidence that removed these pixels
                # from this group's image. The old >.8 fit cutoff left pixels
                # fully assigned to another group (> .35) as false white holes.
                # Low-confidence colour blends remain only partial exemptions;
                # actual paper (zero strength) can never become an occluder.
                confidence=(separated[j]['confidence'] if separated is not None else
                    (winner==j)*np.clip((fits[j]-.10)/.25,0,1))
                other=np.maximum(other,(confidence*np.clip(strengths[j]/.25,0,1)).astype(np.float32))
        # Neutral error bars cannot support a coloured marker. They may hide it.
        if not group_models[gi]['achromatic']:other=np.maximum(other,neutral.astype(np.float32))
        other*=plot
        if group_policy=='hue_family' and len(groups)==1:
            # A one-hue chart can use exactly the source BW pixels. Masking
            # would remove neutral error bars and colour-overlap boundaries
            # that the successful full BW reference actually retained.
            canvas=gray.astype(np.uint8);other[:]=0
            confirmed=canvas.copy();uncertainty[:]=0
        output.append(dict(id=f'G{gi+1}',indices=indices,series_ids=[templates[i].key for i in indices],
            rgb=np.rint(group_models[gi]['bgr'][::-1]).astype(int).tolist(),
            image=cv2.cvtColor(canvas if grid_evidence=='raw_proposal' else confirmed,cv2.COLOR_GRAY2BGR),occlusion=other,
            proposal_image=cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR),grid_evidence=grid_evidence,
            ownership_policy=ownership_policy,
            neutral_geometry_shared=bool(len(indices)>1 and group_models[gi]['achromatic']),
            source_tone_preserved=True,
            palette_identity=identity_report,
            palette_competitor_ids=palette_ids,
            unavailable_palette_ids=[key for key in palette_ids if key not in template_index or key in blocked],
            palette_recovery=recovery_report,recovery_phase=recovery_phase,
            window_image=cv2.cvtColor(confirmed,cv2.COLOR_GRAY2BGR),uncertainty=uncertainty,
            decomposition=[decomposition[i] for i in indices] if decomposition else [],
            templates=[templates[i] for i in indices],reports=[reports[i] for i in indices]))
        # Native BW / genuinely neutral series retain their existing route.
        # For chromatic groups the detector receives ORIGINAL RGB and roles,
        # never the diagnostic/reference grayscale export below.
        repeated_observation=(len(indices)>1 or (observations is not None and
            len(observations[gi].report['shared_observation_groups'])>1))
        if observations is not None and repeated_observation and not group_models[gi]['achromatic']:
            obs=observations[gi]
            output[-1].update(image=image,window_image=image,colour_observation=obs,
                occlusion=obs.other,uncertainty=np.zeros(gray.shape,np.float32),
                observation_report=obs.report)
    updated={r['swatch_id']:r for r in reports}
    return output,[updated.get(r['swatch_id'],r) for r in all_reports]


def detection_options(group, proposal_scales=None):
    """One calibrated size per legend identity in the GUI/CLI colour-group route.

    Explicit proposal scales retain the experimental per-candidate override.
    """
    options=production_detection_options()
    options.update(scale_policy='shared_symbol',proposal_scales=None,
                   window_search_scales=None,geometry_first=False,
                   visible_first_detection=False)
    if group.get('colour_observation') is not None:
        options.update(colour_observation=group['colour_observation'])
    if proposal_scales is not None:
        options.update(scale_policy='per_candidate',proposal_scales=proposal_scales)
    if any(t.model_completed and t.marker_kind=='open' for t in group['templates']):options['window_backend']='cpu'
    # Several achromatic keys share one ink image: recall-only ranking cannot tell an X/outline key from the
    # filled body that contains its ink, so compare them on the same body (ink and paper), as plain B&W does.
    if group.get('neutral_geometry_shared') and group.get('colour_observation') is None:
        options['neutral_group_identity']=True
    return options


def run_group(group, plot_area, legend_area, out_dir, max_iter=10, log_fn=print, proposal_scales=None):
    dest=Path(out_dir);dest.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(dest/'observed_gray.png'),group['image'])
    cv2.imwrite(str(dest/'other_colour_ignore.png'),np.uint8(group['occlusion']*255))
    cv2.imwrite(str(dest/'confirmed_gray.png'),group.get('window_image',group['image']))
    if 'uncertainty' in group:
        cv2.imwrite(str(dest/'colour_uncertainty.png'),np.uint8(group['uncertainty']*255))
    # The same common scale controls both grid proposals and native windows.
    options=detection_options(group, proposal_scales)
    result=detect_points(group['image'],plot_area,legend_area,
        prepared_templates=(group['templates'],group['reports']),
        window_occlusion_mask=group['occlusion'],window_image=group.get('window_image'),
        window_uncertainty_mask=group.get('uncertainty'),log_fn=log_fn,**options)
    save(dest/'detection.json',{k:v for k,v in result.items() if k!='diag_steps'})
    for i,step in enumerate(result['diag_steps']):
        cv2.imwrite(str(dest/f'legend_{i:02}.png'),step['img_bgr'])
    observation=group.get('colour_observation')
    correction_image=(group.get('window_image',group['image']) if observation is None else
                      cv2.cvtColor(observation.reference_gray(),cv2.COLOR_GRAY2BGR))
    corrected=run_correction(correction_image,dest/'step5',result['kept'],result['suppressed'],
        plot_area,legend_area,result['d_est'],max_iter=max_iter,log_fn=log_fn,
        return_diag_imgs=False,score_ignore_mask=group['occlusion'])
    save(dest/'result.json',dict(group=group['id'],series_ids=group['series_ids'],rgb=group['rgb'],
        initial=result['kept'],initial_suppressed=result['suppressed'],
        final=corrected['P_current'],suppressed=corrected['S_current'],
        correction_artifacts=corrected['artifact_dir'],
        iterations=[{k:r[k] for k in ('iteration','action','added','removed','score_before','score_after','stop_reason')}
                    for r in corrected['trace']],
        initial_loss=corrected['initial_score'],final_loss=corrected['final_score']))
    return result,corrected

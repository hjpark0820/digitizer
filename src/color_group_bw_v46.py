"""Colour ownership adapter for the complete production BW engine.

The colour runtime selects this backend for ambiguous palettes. Boxes stay in native
source coordinates. The source image is never edited. Colour only selects ink;
native BW templates, voting, window verification and typed correction are reused.
"""
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np

from bw_legend_v46 import extract_legend_models
from bw_pipeline_v46 import detect_points, production_detection_options, LEGACY_PROPOSAL_SCALES
from bw_step5_v46 import run_correction, save
from color_marker_evidence import _swatch_model, _membership
from color_palette_identity_v46 import palette_relations, distinguish_inks


def prepare_groups(image, plot_area, legend_area, colour_distance=14., group_policy='lab', composition=False,
                   ownership_policy='legacy', grid_evidence='confirmed', palette_recovery='off',
                   swatches=None, neutral_geometry_policy='separate', observation_policy='legacy'):
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
    templates,reports=extract_legend_models(image,legend_area,swatches=swatches)
    # The extractor retains failed entry reports. Keep model-indexed arrays
    # aligned by ID, while returning the complete diagnostic list to callers.
    all_reports=reports
    by_id={r['swatch_id']:r for r in reports}
    reports=[by_id[t.key] for t in templates]
    models=[_swatch_model(image,t.swatch_box) for t in templates]
    decomposition=[]
    if composition:
        from bw_composed_legend_v46 import compose_templates
        templates,reports,decomposition=compose_templates(image,legend_area,templates,reports)
    same,_=palette_relations([m['bgr'] for m in models],colour_distance,
                             hue_family=group_policy=='hue_family')
    def compatible(i,j):
        # Neutral antialias shades share one BW observation plane. This is
        # NOT a claim that black and gray are the same ink: original grayscale
        # and per-series fill descriptors retain tone for later identity tests.
        # Hard colour argmax among neutral shades erases black outlines from
        # keys whose small legend happened to be measured as pale gray.
        return bool(same[i,j] or (neutral_geometry_policy=='shared_source' and
                                 models[i]['achromatic'] and models[j]['achromatic']))
    groups=[]
    for i in range(len(models)):
        g=next((g for g in groups if all(compatible(i,j) for j in g)),None)
        if g is None:groups.append([i])
        else:g.append(i)
    group_models=[];fits=[];strengths=[]
    for indices in groups:
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
    # Distinct inks must not be collapsed again by hue-only membership.
    # Groups were formed with a complete-link identity test; all remaining
    # groups compete as separate inks, even if their median RGBs are close.
    strengths,fits,identity_report=distinguish_inks(image,group_models,strengths,fits,
                                                  equivalent=np.eye(len(groups),dtype=bool))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
    x0,y0,x1,y1=map(int,plot_area)
    plot=np.zeros(gray.shape,bool);plot[y0:y1,x0:x1]=True
    from color_palette_recovery_v46 import recover
    eligible=plot.copy()
    lx0,ly0,lx1,ly1=map(int,legend_area)
    eligible[ly0:ly1,lx0:lx1]=False
    diameters=[float(np.median([getattr(templates[i],'diameter',10.) for i in indices])) for indices in groups]
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
        observations=build_observations(image,group_models,groups,templates)
    output=[]
    for gi,indices in enumerate(groups):
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
            neutral_geometry_shared=bool(len(indices)>1 and all(models[i]['achromatic'] for i in indices)),
            source_tone_preserved=True,
            palette_identity=identity_report,
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
                   window_search_scales=None,geometry_first=False)
    if group.get('colour_observation') is not None:
        options.update(colour_observation=group['colour_observation'])
    if proposal_scales is not None:
        options.update(scale_policy='per_candidate',proposal_scales=proposal_scales)
    if any(t.model_completed and t.marker_kind=='open' for t in group['templates']):options['window_backend']='cpu'
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

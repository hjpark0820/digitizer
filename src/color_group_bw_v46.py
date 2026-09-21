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


def prepare_groups(image, plot_area, legend_area, colour_distance=14., group_policy='lab', composition=False,
                   ownership_policy='legacy', grid_evidence='confirmed'):
    if ownership_policy not in ('legacy','uncertainty_v1'):
        raise ValueError('Unknown colour ownership policy')
    if grid_evidence not in ('confirmed','raw_proposal'):
        raise ValueError('Unknown grid colour evidence policy')
    if group_policy not in ('lab','hue_family'):raise ValueError('Unknown colour grouping policy')
    templates,reports=extract_legend_models(image,legend_area)
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
    labs=cv2.cvtColor(np.uint8([[m['bgr'] for m in models]]),cv2.COLOR_BGR2LAB)[0].astype(float)
    hsv=cv2.cvtColor(np.uint8([[m['bgr'] for m in models]]),cv2.COLOR_BGR2HSV)[0].astype(float)
    def compatible(i,j):
        if np.linalg.norm(labs[i]-labs[j])<=colour_distance:return True
        # Hue is circular, in OpenCV half-degree units. Distinct brightness is
        # retained in the grayscale samples and native template models.
        dh=abs(hsv[i,0]-hsv[j,0]);dh=min(dh,180-dh)*2
        return group_policy=='hue_family' and min(hsv[i,1],hsv[j,1])>=40 and dh<=8
    groups=[]
    for i in range(len(models)):
        g=next((g for g in groups if all(compatible(i,j) for j in g)),None)
        if g is None:groups.append([i])
        else:g.append(i)
    group_models=[];fits=[];strengths=[]
    for indices in groups:
        model=deepcopy(models[indices[0]])
        model['bgr']=np.median([models[i]['bgr'] for i in indices],axis=0)
        if group_policy=='hue_family':
            measurements=[_membership(image,models[i]) for i in indices]
            membership=np.max([m for m,f in measurements],axis=0)
            fit=np.max([f for m,f in measurements],axis=0)
        else:membership,fit=_membership(image,model)
        group_models.append(model);fits.append(fit);strengths.append(membership)
    fits=np.stack(fits);strengths=np.stack(strengths);winner=fits.argmax(axis=0)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
    x0,y0,x1,y1=map(int,plot_area)
    plot=np.zeros(gray.shape,bool);plot[y0:y1,x0:x1]=True
    neutral=(np.ptp(image.astype(float),axis=2)<18)&(gray<190)
    from color_pixel_ownership_v46 import ownership_maps
    separated = (ownership_maps(image,fits,strengths,[not m['achromatic'] for m in group_models])
                 if ownership_policy=='uncertainty_v1' else None)
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
            window_image=cv2.cvtColor(confirmed,cv2.COLOR_GRAY2BGR),uncertainty=uncertainty,
            decomposition=[decomposition[i] for i in indices] if decomposition else [],
            templates=[templates[i] for i in indices],reports=[reports[i] for i in indices]))
    updated={r['swatch_id']:r for r in reports}
    return output,[updated.get(r['swatch_id'],r) for r in all_reports]


def detection_options(group, proposal_scales=None):
    """Keep the reviewed colour-group profile independent of BW GUI defaults."""
    options=production_detection_options()
    options.update(scale_policy='per_candidate',proposal_scales=LEGACY_PROPOSAL_SCALES,
                   window_search_scales=None,geometry_first=False)
    if proposal_scales is not None:options['proposal_scales']=proposal_scales
    if any(t.model_completed and t.marker_kind=='open' for t in group['templates']):options['window_backend']='cpu'
    return options


def run_group(group, plot_area, legend_area, out_dir, max_iter=10, log_fn=print, proposal_scales=None):
    dest=Path(out_dir);dest.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(dest/'observed_gray.png'),group['image'])
    cv2.imwrite(str(dest/'other_colour_ignore.png'),np.uint8(group['occlusion']*255))
    cv2.imwrite(str(dest/'confirmed_gray.png'),group.get('window_image',group['image']))
    if 'uncertainty' in group:
        cv2.imwrite(str(dest/'colour_uncertainty.png'),np.uint8(group['uncertainty']*255))
    # Preserve the separately reviewed colour-group workflow; shared-symbol
    # calibration was requested for the B&W legend-marker production route.
    options=detection_options(group, proposal_scales)
    result=detect_points(group['image'],plot_area,legend_area,
        prepared_templates=(group['templates'],group['reports']),
        window_occlusion_mask=group['occlusion'],window_image=group.get('window_image'),
        window_uncertainty_mask=group.get('uncertainty'),log_fn=log_fn,**options)
    save(dest/'detection.json',{k:v for k,v in result.items() if k!='diag_steps'})
    for i,step in enumerate(result['diag_steps']):
        cv2.imwrite(str(dest/f'legend_{i:02}.png'),step['img_bgr'])
    corrected=run_correction(group.get('window_image',group['image']),dest/'step5',result['kept'],result['suppressed'],
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

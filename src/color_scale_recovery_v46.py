"""Bounded shared-scale refit after conflicting independent marker sizes.

The caller evaluates ORIGINAL legend variants at fixed candidate sites. No
per-point sizes, point-count priors, new marker labels or relaxed pixel costs.
Validation sites are frozen before reselection and never enter the refit.
"""
import math
import numpy as np

CONFIG = dict(maximum_iterations=3, minimum_refit_sites=3,
              validation_fraction=.70, minimum_image_gain=.01)


def recover(scales, profiles, sites, seed_index, inspect):
    """inspect(site_index, scale_index) -> JSON-safe score/clean/position record.

Sites are already independent-x, clean-at-own-best calibration candidates.
Holdouts validate this conditional refinement (not a statistical test set:
the original global proposal used all sites). All fallback scales see exactly
the same sites, including failed/unselected sites in their mean image score.
"""
    profiles=np.asarray(profiles,float)
    native=scales.index(1.)
    result=dict(attempted=True,applied=False,scale=1.,status='unresolved',
                seed_scale=float(scales[seed_index]),iterations=[],comparisons=[],
                config=CONFIG.copy(),validation_scope='excluded_from_refit_not_initial_proposal',
                original_legend_variants_only=True)
    if len(sites)<CONFIG['minimum_refit_sites']+1:
        result['reason']='insufficient_independent_sites_for_refit_and_validation'
        return result
    # Spatially distributed, deterministic; never select holdouts by score.
    order=sorted(range(len(sites)),key=lambda i:(sites[i]['x'],sites[i]['y']))
    count=2 if len(sites)>=6 else 1
    holdout=[order[round((k+1)*(len(order)-1)/(count+1))] for k in range(count)]
    training=[i for i in range(len(sites)) if i not in holdout]
    result.update(training_sites=training,validation_sites=holdout,
                  sites=[dict(x=s['x'],y=s['y']) for s in sites])
    cache={}
    def read(i,j):
        if (i,j) not in cache:cache[i,j]=dict(inspect(i,j))
        return cache[i,j]
    def assess(j):
        rows=[dict(read(i,j),site=i) for i in range(len(sites))]
        scores=[float(r['score']) if not r['line_only_reject'] else 0. for r in rows]
        return dict(scale=float(scales[j]),index=int(j),sites=rows,
                    training_clean=sum(bool(rows[i]['clean']) for i in training),
                    validation_clean=sum(bool(rows[i]['clean']) for i in holdout),
                    mean_score=float(np.mean(scores)),
                    validation_score=float(np.mean([scores[i] for i in holdout])))
    baseline=assess(native)
    def supported(a):
        return (a['training_clean']>=CONFIG['minimum_refit_sites'] and
                a['validation_clean']>=math.ceil(CONFIG['validation_fraction']*len(holdout)))
    def improved(a):
        return (supported(a) and a['mean_score']>=baseline['mean_score']+CONFIG['minimum_image_gain']
                and a['validation_score']>=baseline['validation_score']-1e-8)
    j=seed_index;seen=set();proposals={native,seed_index};refitted=None
    for iteration in range(1,CONFIG['maximum_iterations']+1):
        selected=[i for i in training if read(i,j)['clean']]
        row=dict(iteration=iteration,input_scale=float(scales[j]),selected_sites=selected,
                 site_reviews=[dict(read(i,j),site=i) for i in training])
        result['iterations'].append(row)
        if len(selected)<CONFIG['minimum_refit_sites']:
            row['stop']='insufficient_matching_markers';break
        curves=profiles[selected]
        loss=np.mean(curves.max(axis=1,keepdims=True)-curves,axis=0)
        next_j=int(np.argmin(loss));proposals.add(next_j)
        row.update(refit_scale=float(scales[next_j]),mean_regret=loss.tolist())
        if next_j==j:
            refitted=assess(j)
            row['stop']='stable' if improved(refitted) else 'stable_but_validation_failed'
            break
        if next_j in seen:
            row['stop']='scale_cycle';break
        seen.add(j);j=next_j
    else:result['iterations'][-1]['stop']='iteration_limit'
    if refitted is not None and improved(refitted):
        chosen=refitted
        result.update(applied=True,status='shared_reestimated_consensus',reason='stable_refit_validated')
    else:
        # Recovery failed: compare seed, attempted refits and native at the
        # SAME fixed sites. Do not reward cherry-picking only matching markers.
        comparisons=[assess(k) for k in sorted(proposals)]
        eligible=[a for a in comparisons if improved(a)]
        chosen=max(eligible,key=lambda a:(a['mean_score'],-abs(a['scale']-1.))) if eligible else baseline
        if eligible:
            result.update(applied=True,status='shared_image_compared_fallback',
                          reason='refit_failed_best_validated_common_image_score')
        elif supported(baseline):
            result.update(applied=True,status='shared_native_image_confirmed',
                          reason='native_size_supported_and_no_validated_improvement')
        else:
            result['reason']='no_validated_improvement_native_is_unconfirmed_reference'
    result.update(scale=chosen['scale'],selected_scale_index=chosen['index'],
                  comparisons=[assess(k) for k in sorted(proposals)],
                  window_evaluations=len(cache))
    return result

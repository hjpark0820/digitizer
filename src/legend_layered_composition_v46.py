"""Compare line + marker hypotheses, including transparent/paper hollow faces.

Only observed legend pixels enter fitting. Candidate line periods are fitted
on independent flanks; multiple periods survive until whole-swatch comparison.
The winning marker raster is returned separately from its connector. A close
layering contest does not force a drawing-order claim or invent observed ink.
"""
from dataclasses import asdict
import numpy as np
from scipy.optimize import least_squares
from legend_composition_v46 import Config, FAMILIES, _calibrate, render_model
from legend_open_composition_v46 import line_candidates

VERSION = 'layered_hollow_composition_v1'
HOLLOW = ('open_circle', 'open_ellipse', 'open_square', 'open_rectangle')


def family(name):
    return dict(circle='ellipse', square='rectangle', open_circle='open_ellipse',
                open_square='open_rectangle').get(name, name)


def fit_layered(crop_rgb, nominal_rgb=None, cfg=Config(enable_open_dashed=True)):
    alpha, fg, bg, colour_error = _calibrate(crop_rgb, nominal_rgb)
    h, w = alpha.shape
    lines, flank = line_candidates(alpha, cfg, maximum_dashed=2)
    # Estimate a body region from persistent tall column spans, not isolated
    # line-end noise. This measurement is shared by EVERY competing model.
    spans = np.array([np.ptp(np.flatnonzero(c))+1 if c.any() else 0
                      for c in (alpha > .25).T])
    ids = np.flatnonzero(spans > max(5., .30*h))
    runs = np.split(ids, np.flatnonzero(np.diff(ids) > 2)+1) if len(ids) else []
    if runs:
        run = max(runs, key=len)
        ids = np.arange(run[0], run[-1]+1)
    body = (alpha > .25).copy()
    if len(ids):
        body[:, :ids[0]] = False
        body[:, ids[-1]+1:] = False
    ys, xs = np.nonzero(body)
    if len(xs):
        cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
        width, height = max(3., float(np.ptp(xs)+1)), max(3., float(np.ptp(ys)+1))
    else:
        cx, cy, width, height = (w-1)/2, (h-1)/2, max(3., h/3), max(3., h/3)
    pad = max(3., .25*max(width, height))
    roi = [max(0, int(cx-width/2-pad)), 0, min(w, int(cx+width/2+pad+1)), h]
    sl = np.s_[:, roi[0]:roi[2]]
    # Body and flanks each have a fixed area/weight for every hypothesis.
    weights = np.zeros_like(alpha)
    weights[sl] = .8 / max(1, alpha[sl].size)
    if flank.any():
        weights[:, flank] += .2 / (h*int(flank.sum()))
    else:
        weights /= .8
    root_weight = np.sqrt(weights)
    models = []

    for lp in lines:
        line = render_model('line_only', {}, lp, alpha.shape, cfg)['line_alpha']
        excess = np.maximum(alpha-line, 0)

        def assess(name, params, count):
            fields = render_model(name, params, lp, alpha.shape, cfg)
            diff = fields['composite_alpha']-alpha
            loss = float(np.sum(weights*diff**2))
            ink = max(float(excess[sl].sum()), float(fields['marker_alpha'][sl].sum()), 1.)
            penalty = cfg.complexity_cost_per_parameter*(count+(3 if lp['style']=='dashed' else 0))
            return dict(name=name, params=params, line_params=lp, loss=loss+penalty,
                        data_loss=loss, body_mse=float(np.mean(diff[sl]**2)),
                        flank_mse=float(np.mean(diff[:, flank]**2)) if flank.any() else None,
                        mean_absolute_error=float(np.mean(abs(diff[sl]))),
                        marker_relative_error=float(abs(diff[sl]).sum()/ink),
                        complexity_penalty=penalty, free_marker_parameters=count,
                        interior_mode=params.get('interior_mode', 'filled'))

        models.append(assess('line_only', {}, 0))
        if excess.sum() < cfg.minimum_ink_mass:
            continue
        for name in (*FAMILIES, *HOLLOW):
            equal = name in ('circle', 'square', 'open_circle', 'open_square')
            hollow = name in HOLLOW
            for mode in (('transparent', 'paper') if hollow else ('filled',)):
                initial = [cx, cy, (width+height)/2] if equal else [cx, cy, width, height]
                low = [max(0., cx-.35*width), max(0., cy-.35*height), max(2., .55*width)]
                high = [min(w-1., cx+.35*width), min(h-1., cy+.35*height), min(w-1., 1.4*width)]
                if equal:
                    low[2] = max(2., .55*min(width, height))
                    high[2] = min(h-1., w-1., 1.4*max(width, height))
                else:
                    low.append(max(2., .55*height)); high.append(min(h-1., 1.4*height))
                if hollow:
                    # Optimise a fraction of the CURRENT diameter. Independent
                    # pixel bounds could turn a shrunken "hollow" model solid.
                    initial.append(.15)
                    low.append(.02); high.append(.40)

                def unpack(v):
                    p = dict(cx=float(v[0]), cy=float(v[1]), width=float(v[2]),
                             height=float(v[2] if equal else v[3]))
                    if hollow:
                        p.update(stroke_width=float(v[-1]*min(p['width'],p['height'])), interior_mode=mode)
                    return p

                def residual(v):
                    pred = render_model(name, unpack(v), lp, alpha.shape, cfg)['composite_alpha']
                    return ((pred-alpha)*root_weight).ravel()

                fitted = least_squares(residual, np.clip(initial, np.array(low)+1e-5, np.array(high)-1e-5),
                                       bounds=(low, high), diff_step=.003, max_nfev=cfg.maximum_evaluations)
                models.append(assess(name, unpack(fitted.x), len(initial)))

    models.sort(key=lambda m:m['loss'])
    best = models[0]
    other = next((m for m in models if family(m['name']) != family(best['name'])), None)
    line_best = min((m for m in models if m['name']=='line_only'), key=lambda m:m['loss'])
    margin = None if other is None else other['loss']-best['loss']
    improvement = line_best['data_loss']-best['data_loss']
    status = ('blank' if alpha.sum()<cfg.minimum_ink_mass else
              'line_only_or_no_resolved_marker' if best['name']=='line_only' or improvement<cfg.minimum_marker_improvement else
              'unknown_poor_fit' if best['mean_absolute_error']>cfg.maximum_marker_mae or
              best['marker_relative_error']>cfg.maximum_marker_relative_error or
              float(colour_error.mean())>cfg.maximum_colour_residual else
              'ambiguous_shape_family' if margin is None or margin<cfg.minimum_family_margin else
              'supported_simple_shape_model')
    opposite = next((m for m in models if family(m['name'])==family(best['name']) and
                     m['interior_mode']!=best['interior_mode']), None)
    layer_margin = None if opposite is None else opposite['loss']-best['loss']
    fields = render_model(best['name'], best['params'], best['line_params'], alpha.shape, cfg)
    fields.update(observed_alpha=alpha, residual=alpha-fields['composite_alpha'],
                  colour_residual=colour_error, flank_mask=np.broadcast_to(flank, alpha.shape).astype(np.uint8))
    rec = dict(version=VERSION, status=status, best_model_name=best['name'], best_model=best,
               models=models, line_params=best['line_params'], fit_roi=roi, config=asdict(cfg),
               foreground_rgb=(fg*255).tolist(), background_rgb=(bg*255).tolist(),
               winner_other_family_margin=margin,
               winner_runner_margin=models[1]['loss']-best['loss'] if len(models)>1 else None,
               interior_mode_margin=layer_margin,
               interior_mode_status='not_applicable' if opposite is None else
                   'ambiguous' if layer_margin<cfg.minimum_family_margin else 'supported',
               marker_improvement=improvement, line_only_data_loss=line_best['data_loss'],
               colour_residual_mae=float(colour_error.mean()), line_hypotheses=len(lines),
               score_policy='0.8 body MSE + 0.2 flank MSE + parameter cost; fixed regions',
               template_policy='pure_model_marker_if_supported',
               caveats=['Synthesized hidden pixels are not observed evidence.',
                        'An unresolved drawing order is not a confirmed transparent or paper face.'])
    return rec, fields

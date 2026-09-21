"""Infer a simple legend marker by fitting line UNION marker to native RGB.

No observed pixels are deleted. The hidden interior is a model completion, not
observed evidence. The legacy fitter uses filled, axis-aligned primitives.
Config.enable_open_dashed opts into open circles and periodic dashed connectors.
Unsupported/irregular symbols remain unknown. No plot data enter this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage, optimize


@dataclass(frozen=True)
class Config:
    supersample: int = 3
    maximum_evaluations: int = 160
    complexity_cost_per_parameter: float = 0.00015
    minimum_marker_improvement: float = 0.010
    maximum_marker_mae: float = 0.070
    maximum_marker_relative_error: float = 0.25
    minimum_family_margin: float = 0.0015
    minimum_ink_mass: float = 3.0
    maximum_colour_residual: float = 0.050
    # Opt-in until the extended primitive/line family has broad validation.
    enable_open_dashed: bool = False


FAMILIES = ('circle', 'ellipse', 'diamond', 'square', 'rectangle',
            'triangle_up', 'triangle_down', 'triangle_left', 'triangle_right')


def _calibrate(rgb, nominal):
    rgb = np.asarray(rgb, float) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 5:
        raise ValueError('crop_rgb must be an RGB crop at least 5 by 5 pixels')
    bg = np.percentile(rgb.reshape(-1, 3), 98, axis=0)
    contrast = bg - rgb
    norm = np.linalg.norm(contrast, axis=2)
    selected = norm > 0.05
    if nominal is not None:
        direction = bg - np.asarray(nominal, float) / 255.0
        if np.linalg.norm(direction) > 0.05:
            direction /= np.linalg.norm(direction)
            selected &= np.sum(contrast * direction, axis=2) / np.maximum(norm, 1e-9) > .88
    if not selected.any():
        return np.zeros(rgb.shape[:2]), bg, bg, np.zeros(rgb.shape[:2])
    # Foreground is estimated from the image; nominal RGB only selects the ray.
    strong = selected & (norm >= np.percentile(norm[selected], 70))
    direction = np.median(contrast[strong], axis=0)
    direction /= max(np.linalg.norm(direction), 1e-9)
    projected = contrast @ direction
    magnitude = float(np.percentile(projected[strong], 95))
    fg = np.clip(bg - magnitude * direction, 0, 1)
    vector = bg - fg
    alpha = np.clip((contrast @ vector) / max(float(vector @ vector), 1e-12), 0, 1)
    colour_residual = np.linalg.norm(contrast - alpha[..., None] * vector, axis=2)
    return alpha, fg, bg, colour_residual


def _grid(shape, ss):
    h, w = shape
    # Source-pixel centres are integer coordinates; samples span +/- 0.5 pixel.
    return np.meshgrid((np.arange(w * ss) + .5) / ss - .5,
                       (np.arange(h * ss) + .5) / ss - .5)


def _polygon_distance(x, y, vertices):
    distances = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        distances.append((dx * (y - a[1]) - dy * (x - a[0])) / np.hypot(dx, dy))
    return np.minimum.reduce(distances)


def render_model(name, params, line_params, shape, cfg=Config()):
    """Render union on a fine grid BEFORE blur/pixel averaging.

    Subpixel boundary coverage is a linear one-fine-pixel ramp, permitting
    continuous optimisation. Triangle centre is its bounding-box centre.
    """
    ss = int(cfg.supersample)
    if ss < 1:
        raise ValueError('supersample must be positive')
    x, y = _grid(shape, ss)
    lp = line_params
    yl = lp['cy'] + lp.get('slope', 0.0) * (x - lp.get('cx', (shape[1] - 1) / 2))
    ld = np.minimum.reduce([lp['thickness'] / 2 - np.abs(y - yl),
                            x - lp['x0'], lp['x1'] - x])
    if lp.get('style','solid')=='dashed':
        period=float(lp['dash_period'])
        phase=(x-float(lp['dash_phase'])+period/2)%period-period/2
        ld=np.minimum(ld,float(lp['dash_length'])/2-np.abs(phase))
    line = np.clip(.5 + ss * ld, 0, 1)
    marker = np.zeros_like(line)
    if name != 'line_only':
        cx, cy = params['cx'], params['cy']
        rx, ry = params['width'] / 2, params['height'] / 2
        xx, yy = x - cx, y - cy
        if name in ('circle', 'ellipse', 'open_circle', 'open_ellipse'):
            r = np.sqrt((xx / rx) ** 2 + (yy / ry) ** 2)
            signed = (1 - r) * min(rx, ry)
            outer_signed = signed
            if name.startswith('open_'):
                # Outer diameter and positive stroke thickness; not a disk.
                signed=np.minimum(signed,float(params['stroke_width'])-signed)
        elif name in ('square', 'rectangle', 'open_square', 'open_rectangle'):
            signed = np.minimum(rx - np.abs(xx), ry - np.abs(yy))
            outer_signed = signed
            if name.startswith('open_'):
                signed = np.minimum(signed, float(params['stroke_width']) - signed)
        else:
            vertices = {
                'diamond': [(0, -ry), (rx, 0), (0, ry), (-rx, 0)],
                'triangle_up': [(0, -ry), (rx, ry), (-rx, ry)],
                'triangle_down': [(-rx, -ry), (rx, -ry), (0, ry)],
                'triangle_left': [(-rx, 0), (rx, -ry), (rx, ry)],
                'triangle_right': [(-rx, -ry), (rx, 0), (-rx, ry)],
            }[name]
            signed = _polygon_distance(xx, yy, vertices)
        marker = np.clip(.5 + ss * signed, 0, 1)
    union = np.maximum(line, marker)
    if name.startswith('open_') and params.get('interior_mode', 'transparent') == 'paper':
        # A background-filled marker is drawn OVER the connector. Compose on
        # the fine grid before blur: its white interior is not missing ink.
        outer = np.clip(.5 + ss * outer_signed, 0, 1)
        union = np.maximum(line * (1 - outer), marker)
    sigma = lp.get('blur_sigma', .4) * ss

    def sample(a):
        a = ndimage.gaussian_filter(a, sigma, mode='constant') if sigma > 0 else a
        return a.reshape(shape[0], ss, shape[1], ss).mean((1, 3)).astype(np.float32)

    la, ma, ca = sample(line), sample(marker), sample(union)
    visible = np.maximum(ca - la, 0)
    return dict(line_alpha=la, marker_alpha=ma, composite_alpha=ca,
                hidden_marker_by_line=np.maximum(ma - visible, 0),
                visible_marker_evidence=visible)


def _fit_line(alpha, cfg):
    h, w = alpha.shape
    occupied = np.where(alpha.max(axis=0) > .12)[0]
    x0, x1 = (float(occupied[0]) - .5, float(occupied[-1]) + .5) if len(occupied) else (0., float(w-1))
    span = max(x1 - x0, 4.)
    xx = np.arange(w)
    flank = ((xx >= x0 + .04 * span) & (xx <= x0 + .24 * span)) | ((xx >= x1 - .24 * span) & (xx <= x1 - .04 * span))
    if not flank.any():
        flank[np.clip(np.rint([x0+.10*span,x1-.10*span]).astype(int),0,w-1)] = True
    profile = np.mean(alpha[:, flank], axis=1)
    cy = float(np.sum(profile * np.arange(h)) / max(profile.sum(), 1e-9))
    thickness = float(np.clip(profile.sum(), .4, h * .25))
    lp = dict(x0=x0, x1=x1, cy=cy, thickness=thickness, slope=0., cx=(x0+x1)/2, blur_sigma=.4)
    # Use flank columns only, so the marker cannot force the line to thicken.
    def residual(v):
        trial = dict(lp, cy=v[0], thickness=v[1], slope=v[2], blur_sigma=v[3])
        return (render_model('line_only', {}, trial, alpha.shape, cfg)['line_alpha'][:, flank] - alpha[:, flank]).ravel()
    fitted = optimize.least_squares(residual, [cy, thickness, 0, .4],
        bounds=([max(0,cy-5),.2,-.10,.05],[min(h-1,cy+5),max(thickness*2.5,1.),.10,1.3]),
        diff_step=.015, max_nfev=cfg.maximum_evaluations)
    lp.update(cy=float(fitted.x[0]), thickness=float(fitted.x[1]),
              slope=float(fitted.x[2]), blur_sigma=float(fitted.x[3]))
    return lp, flank


def _family(name):
    return {'circle':'ellipse', 'square':'rectangle'}.get(name, name)


def fit_legend_composition(crop_rgb, nominal_rgb=None, cfg=Config()):
    """Return JSON-ready diagnostics and native-size alpha fields.

    Loss is MSE in a fixed data-derived marker ROI plus 0.00015 per free
    marker parameter (three for circle/square, four otherwise). Line parameters
    are shared, so their equal penalty is omitted. This is not a probability.
    """
    if cfg.enable_open_dashed:
        from legend_open_composition_v46 import fit_extended
        return fit_extended(crop_rgb,nominal_rgb,cfg)
    alpha, fg, bg, colour_error = _calibrate(crop_rgb, nominal_rgb)
    h, w = alpha.shape
    lp, flank = _fit_line(alpha, cfg)
    line_fields = render_model('line_only', {}, lp, alpha.shape, cfg)
    excess = np.maximum(alpha - line_fields['line_alpha'], 0)
    # Excess is used only to initialise the model and freeze its evaluation ROI.
    # It is never used as the extracted marker silhouette.
    mass = float(excess.sum())
    roi_mask = excess > .18
    cc, n = ndimage.label(roi_mask)
    if n:
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        primary = cc == np.argmax(sizes)
        # Both halves of a marker may be disconnected across the shared line.
        ys, xs = np.where(primary)
        centre_x = float(np.average(xs, weights=excess[primary]))
        near = roi_mask & (np.abs(np.arange(w)[None, :] - centre_x) <= max(8., (xs.max()-xs.min()+1)*1.3))
        ys, xs = np.where(near)
        cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
        width, height = max(4.,xs.max()-xs.min()+1.), max(4.,ys.max()-ys.min()+1.)
    else:
        cx, cy, width, height = (w-1)/2, lp['cy'], min(w,h)/3, min(w,h)/3
    pad = max(4., .35 * max(width, height))
    roi = [max(0,int(np.floor(cx-width/2-pad))), max(0,int(np.floor(cy-height/2-pad))),
           min(w,int(np.ceil(cx+width/2+pad+1))), min(h,int(np.ceil(cy+height/2+pad+1)))]
    x0,y0,x1,y1 = roi
    sl = np.s_[y0:y1,x0:x1]

    def assess(name, pars, parameters):
        fields = render_model(name, pars, lp, alpha.shape, cfg)
        diff = fields['composite_alpha'][sl] - alpha[sl]
        mse, mae = float(np.mean(diff**2)), float(np.mean(np.abs(diff)))
        # Blank ROI padding must not make an unsupported shape look accurate.
        # Scale L1 reconstruction error by visible marker ink, not total area.
        ink = max(float(excess[sl].sum()),float(fields['visible_marker_evidence'][sl].sum()),1.0)
        relative = float(np.abs(diff).sum()) / ink
        cost = cfg.complexity_cost_per_parameter * parameters
        return dict(name=name,params=pars,loss=mse+cost,data_loss=mse,
                    mean_absolute_error=mae,marker_relative_error=relative,
                    complexity_penalty=cost,free_marker_parameters=parameters)

    models = [assess('line_only', {}, 0)]
    if mass >= cfg.minimum_ink_mass:
        for name in FAMILIES:
            equal = name in ('circle','square')
            count = 3 if equal else 4
            def unpack(v):
                return dict(cx=float(v[0]),cy=float(v[1]),width=float(v[2]),height=float(v[2] if equal else v[3]))
            initial = [cx,cy,(width+height)/2] if equal else [cx,cy,width,height]
            low = [max(0.,cx-width*.65),max(0.,cy-height*.65),max(2.,width*.45)]
            high = [min(w-1.,cx+width*.65),min(h-1.,cy+height*.65),min(w*.8,width*1.8)]
            if not equal:
                low += [max(2.,height*.45)]; high += [min(h*.95,height*1.8)]
            if equal:
                low[2] = max(2.,min(width,height)*.45)
                high[2] = min(min(w,h)*.95,max(width,height)*1.8)
            def residual(v):
                pred = render_model(name,unpack(v),lp,alpha.shape,cfg)['composite_alpha']
                return (pred[sl]-alpha[sl]).ravel()
            # Two starts reduce dependence on a triangle's initial
            # bounding-box displacement. Same starts for every family.
            fitted = []
            for shift in (0., .15):
                start = np.clip(initial, np.asarray(low)+1e-4, np.asarray(high)-1e-4)
                start[1] = np.clip(start[1]+shift*height,low[1]+1e-4,high[1]-1e-4)
                result = optimize.least_squares(residual,start,bounds=(low,high),
                    diff_step=.003,max_nfev=cfg.maximum_evaluations)
                fitted.append(assess(name,unpack(result.x),count))
            models.append(min(fitted,key=lambda m:m['loss']))
    models.sort(key=lambda m:m['loss'])
    best = models[0]
    runner = models[1] if len(models)>1 else None
    alternative = next((m for m in models[1:] if _family(m['name']) != _family(best['name'])),None)
    line_model = next(m for m in models if m['name']=='line_only')
    improvement = line_model['data_loss'] - best['data_loss']
    margin = None if alternative is None else alternative['loss']-best['loss']
    colour_mae = float(np.mean(colour_error[sl]))
    if float(alpha.sum()) < cfg.minimum_ink_mass:
        status = 'blank'
    elif best['name']=='line_only' or improvement < cfg.minimum_marker_improvement:
        status = 'line_only_or_no_resolved_marker'
    elif (best['mean_absolute_error'] > cfg.maximum_marker_mae
          or best['marker_relative_error'] > cfg.maximum_marker_relative_error
          or colour_mae > cfg.maximum_colour_residual):
        status = 'unknown_poor_fit'
    elif margin is None or margin < cfg.minimum_family_margin:
        status = 'ambiguous_shape_family'
    else:
        status = 'supported_simple_shape_model'
    fields = render_model(best['name'],best['params'],lp,alpha.shape,cfg)
    fields.update(observed_alpha=alpha.astype(np.float32),
                  residual=(alpha-fields['composite_alpha']).astype(np.float32),
                  colour_residual=colour_error.astype(np.float32),
                  flank_mask=np.broadcast_to(flank[None,:],alpha.shape).astype(np.uint8))
    centroid = None
    if best['name'] != 'line_only':
        p=best['params']; dx=dy=0.
        if best['name']=='triangle_down': dy=-p['height']/6
        if best['name']=='triangle_up': dy=p['height']/6
        if best['name']=='triangle_left': dx=p['width']/6
        if best['name']=='triangle_right': dx=-p['width']/6
        centroid=[p['cx']+dx,p['cy']+dy]
    record=dict(status=status,best_model=best,best_model_name=best['name'],models=models,
        line_params=lp,foreground_rgb=(fg*255).tolist(),background_rgb=(bg*255).tolist(),
        fit_roi=roi,config=asdict(cfg),winner_runner_margin=None if runner is None else runner['loss']-best['loss'],
        winner_other_family_margin=margin,line_only_data_loss=line_model['data_loss'],
        marker_improvement=improvement,colour_residual_mae=colour_mae,excess_mass=mass,
        inferred_centroid=centroid,coordinate_convention='Crop-local integer pixel centres; cx/cy are marker bounding-box centre, not triangle centroid.',
        hidden_fraction=float(fields['hidden_marker_by_line'].sum()/max(float(fields['marker_alpha'].sum()),1e-9)),
        caveats=['Same-colour line/marker overlap is unidentifiable from pixels alone; completion assumes the selected primitive.',
                 'Filled axis-aligned family only; no hollow, star, rotated-square, arbitrary polygon, or double-line model.',
                 'Loss margins are descriptive, not calibrated probabilities. Marker ROI is fixed across candidate families.',
                 'Foreground/background RGB are estimated assuming a single foreground colour and approximately uniform background.'])
    return record,fields

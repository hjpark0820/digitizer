"""Opt-in, plot-wide overlap hypotheses; not additional marker detections.

Freeze the initial observed active points, cluster their x coordinates, and add
MISSING legend identities at each observed centre in that column. Never
bootstrap from suppressed points or insert a column with zero active points.
Call once across ALL colour groups, before splitting correction by colour.
"""
from copy import deepcopy
import hashlib
import math
import statistics

from bw_suppressed_v46 import decode_marker_mask

VERSION = 'x_missing_series_overlap_v2'
LEGACY_VERSION = 'x_singleton_overlap_v1'


def is_overlap_hypothesis(point):
    return point.get('source') in (LEGACY_VERSION, VERSION)


def same_missing_column(a, b):
    """One missing series may occupy only one alternative y in a sampled x bin.

    Different series may share an exact location. Only v2 records carrying a
    frozen column constraint invoke this rule; old/native records alone do not.
    """
    if a['swatch_id'] != b['swatch_id']:
        return False
    for p in (a,b):
        col = p.get('overlap_column')
        if col is not None:
            lo,hi=map(float,col['x_span']);tol=float(col['x_tolerance_px'])
            if max(hi,float(a['cx']),float(b['cx']))-min(lo,float(a['cx']),float(b['cx'])) <= tol+1.e-8:
                return True
    return False


def generate(active, suppressed, series_models, plot_box, legend_box=None, *, x_tolerance=None,
             policy='missing_series'):
    """Return (copied full pool, audit), leaving all inputs untouched.

    ``series_models`` maps legend IDs to class_name, source_diameter and an
    encoded marker_mask. Masks describe the TARGET legend symbol, never the
    visible donor symbol. They are rendering models, not recovered pixels.
    Each record has a CONDITIONAL fixed identity for compatibility with typed
    correction. Site alternatives are represented as separate records, not by
    mutating class_name independently of swatch_id. The singleton policy keeps
    previous experiments reproducible.
    """
    if policy not in ('missing_series','singleton'):
        raise ValueError('Unknown overlap candidate policy')
    version = VERSION if policy=='missing_series' else LEGACY_VERSION
    def box(value):
        values = tuple(map(float, value))
        if len(values) != 4 or not all(map(math.isfinite, values)) or not (
                values[0] < values[2] and values[1] < values[3]):
            raise ValueError('Expected a finite positive-area ROI')
        return values

    plot_box = box(plot_box)
    legend_box = box(legend_box) if legend_box is not None else None
    diameters = []
    for sid, model in series_models.items():
        d = float(model['source_diameter'])
        if not sid or not model['class_name'] or not math.isfinite(d) or d <= 0:
            raise ValueError('Invalid legend identity or diameter')
        if not decode_marker_mask(model['marker_mask']).any():
            raise ValueError('Target legend model must have a nonempty marker mask')
        if any(not math.isfinite(float(model.get(field,0.))) for field in ('marker_offset_x','marker_offset_y')):
            raise ValueError('Target legend model must have finite marker offsets')
        diameters.append(d)
    if not diameters:
        raise ValueError('At least one legend identity is required')
    tolerance = float(x_tolerance if x_tolerance is not None else max(2., .45*statistics.median(diameters)))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('x_tolerance must be finite and positive')
    pool = deepcopy(suppressed)
    unique = {}
    for p in active:
        sid, x, y = p['swatch_id'], float(p['cx']), float(p['cy'])
        if sid not in series_models or not all(map(math.isfinite, (x, y))):
            raise ValueError('Active point has an unknown identity or invalid coordinates')
        if not (plot_box[0] <= x < plot_box[2] and plot_box[1] <= y < plot_box[3]):
            raise ValueError('Active point is outside the plot ROI')
        if legend_box and legend_box[0] <= x < legend_box[2] and legend_box[1] <= y < legend_box[3]:
            raise ValueError('Active point is inside the legend ROI')
        if is_overlap_hypothesis(p) or p.get('state')=='active_hypothesis' or p.get('tentative'):
            continue
        unique.setdefault((sid, x, y), p)
    columns = []
    for p in sorted(unique.values(), key=lambda p: (p['cx'], p['cy'], p['swatch_id'])):
        # Bounded total span, not single-link chaining through close neighbours.
        if not columns or float(p['cx']) - float(columns[-1][0]['cx']) > tolerance:
            columns.append([p])
        else:
            columns[-1].append(p)
    audit = dict(version=version, x_tolerance_px=tolerance, columns=[], added=[], reused=[],
                 policy=policy, sites=[],
                 generation='missing series at each observed donor; frozen initial x columns; no automatic activation',
                 evidence='structural hypothesis only; no new pixel evidence')
    for col in columns:
        present = sorted({p['swatch_id'] for p in col})
        missing = sorted(set(series_models)-set(present))
        row = dict(x=float(statistics.median(p['cx'] for p in col)),
                   x_span=[float(col[0]['cx']), float(col[-1]['cx'])], active_count=len(col),
                   present_series=present,missing_series=missing,
                   active=[dict(swatch_id=p['swatch_id'], cx=p['cx'], cy=p['cy']) for p in col],
                   created_ids=[])
        audit['columns'].append(row)
        if policy=='singleton' and len(col) != 1:
            row['reason'] = 'multiple_active_markers_in_column'
            continue
        if not missing:
            row['reason']='all_legend_series_present'
            continue
        row['reason']='single_active_marker_overlap_prior' if policy=='singleton' else 'missing_series_overlap_prior'
        donors={}
        for p in col:donors.setdefault((float(p['cx']),float(p['cy'])),[]).append(p)
        for (x,y),observed in donors.items():
            donor=observed[0]
            site_id='XS_'+hashlib.sha256(f'{x:.6f}|{y:.6f}'.encode()).hexdigest()[:14]
            audit['sites'].append(dict(site_id=site_id,cx=x,cy=y,donor_series=[p['swatch_id'] for p in observed],
                                      possible_missing_series=missing,assignment='conditional_not_confirmed'))
            for sid in missing:
                model=series_models[sid]
                # Reuse measured suppressed evidence without rewriting its
                # scores/mask. Position alternatives for a missing series are
                # mutually exclusive in Step 5, NOT multiple observed points.
                radius=max(1.,.15*float(model['source_diameter']))
                existing=next((p for p in pool if p['swatch_id']==sid and
                               math.hypot(float(p['cx'])-x,float(p['cy'])-y)<=radius),None)
                if existing is not None:
                    audit['reused'].append(dict(swatch_id=sid,cx=x,cy=y,candidate_id=existing.get('candidate_id')))
                    if policy=='missing_series':
                        existing.setdefault('overlap_column',dict(x_span=row['x_span'].copy(),x_tolerance_px=tolerance))
                    continue
                token=(f'{sid}|{donor["swatch_id"]}|{x:.6f}|{y:.6f}' if policy=='singleton'
                       else f'{sid}|{x:.6f}|{y:.6f}')
                cid=('XO_' if policy=='singleton' else 'XM_')+hashlib.sha256(token.encode()).hexdigest()[:14]
                candidate=dict(swatch_id=sid,template=sid,class_name=model['class_name'],
                    shape_hint=model['class_name'],cx=x,cy=y,candidate_id=cid,point_id=cid,
                    original_detection=False,tentative=True,state='suppressed',source=version,
                    evidence_tier='structural_overlap_prior',evidence_supported=False,pixel_evidence=None,
                    activation_eligible=True,auto_promote=False,existence='unknown',conditional_marker_class=model['class_name'],
                    marker_mask=deepcopy(model['marker_mask']),
                    marker_mask_source=model.get('marker_mask_source','target_legend_model_not_observed'),
                    marker_offset_x=float(model.get('marker_offset_x',0.)),
                    marker_offset_y=float(model.get('marker_offset_y',0.)),source_diameter=float(model['source_diameter']),
                    overlap_donor=dict(swatch_id=donor['swatch_id'],point_id=donor.get('point_id'),cx=x,cy=y),
                    reason=row['reason'])
                for field in ('marker_scale','marker_aspect','effective_diameter'):
                    if field in model:
                        candidate[field]=model[field]
                if policy=='missing_series':
                    candidate.update(symbol_assignment='conditional_legend_hypothesis',hypothesis_site_id=site_id,
                        overlap_column=dict(x_span=row['x_span'].copy(),x_tolerance_px=tolerance),
                        possible_missing_series=missing.copy())
                pool.append(candidate);audit['added'].append(deepcopy(candidate));row['created_ids'].append(cid)
    audit['added_count'] = len(audit['added'])
    return pool, audit

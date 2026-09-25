"""One recovered colour field for initial geometry, paths and saved correction.

Recovery adds only bounded observed chromatic support. Frozen handoffs prevent
Step 5 from relearning colours with already-scaled marker diameters or silently
reverting to an unadapted legend. Historical handoffs remain unadapted.
"""
from copy import deepcopy

import numpy as np

VERSION = 'frozen_recovered_colour_fields_v1'
SERIES_FIELDS = ('membership', 'colour_confidence', 'other', 'blend_uncertainty')
PLANE_FIELDS = ('paper', 'unknown', 'all_ink')
FIELDS = SERIES_FIELDS + PLANE_FIELDS + ('equivalent_colours', 'palette_rgb', 'paper_bgr')


def apply_recovery(crop, templates, evidence, valid, *, policy):
    """Adapt distinct observed inks, then refresh all dependent colour fields."""
    from color_palette_identity_v46 import palette_relations
    from color_palette_recovery_v46 import recover, POLICIES
    from color_blend_uncertainty_v46 import blend_uncertainty
    if policy not in POLICIES:
        raise ValueError('Unknown palette recovery policy')
    models = [t['model'] for t in templates]
    same, _ = palette_relations([m['bgr'] for m in models])
    if policy != 'off' and np.any(same & ~np.eye(len(models), dtype=bool)):
        evidence['palette_recovery'] = dict(policy=policy, status='same_ink_requires_group_route',
                                            calibration_pixels=0, residual_pixels=0)
        evidence['recovery_phase'] = np.zeros(crop.shape[:2], np.uint8)
        return
    membership, confidence, report, phase = recover(crop, models,
        evidence['membership'], evidence['colour_confidence'], valid,
        [t['diameter'] for t in templates], policy=policy)
    report['anchor_coordinates'] = 'evidence-raster pixel centres, before source-coordinate transform'
    report['series_ids'] = [str(t['id']) for t in templates]
    evidence.update(palette_recovery=report, recovery_phase=phase)
    if not phase.any():
        return
    evidence['membership'], evidence['colour_confidence'] = membership, confidence
    # A recovered own pixel cannot simultaneously remain an unknown hole or
    # retain the old rival-occlusion waiver. Use the original evidence equations.
    other = np.zeros_like(membership)
    for i in range(len(templates)):
        rivals = np.flatnonzero(~evidence['equivalent_colours'][i])
        if len(rivals):
            rival = np.max(membership[rivals] * confidence[rivals], axis=0)
            other[i] = rival * np.clip((rival-membership[i]-.08)/.35, 0., 1.)
    evidence['other'] = other
    evidence['unknown'] = evidence['all_ink'] * (1-confidence.max(axis=0))
    evidence['blend_uncertainty'] = blend_uncertainty(crop, templates, membership, confidence,
        evidence['equivalent_colours'], evidence['paper_bgr'])


def freeze(evidence):
    """Native source-local snapshot. Missing legacy contracts return no snapshot."""
    if evidence is None or evidence.get('palette_recovery', {}).get('policy', 'off') == 'off':
        return None
    if evidence.get('scale_x', 1.) != 1. or evidence.get('scale_y', 1.) != 1.:
        raise ValueError('Frozen colour fields require native resolution')
    return dict(version=VERSION, series_ids=[str(t['id']) for t in evidence['templates']],
        plot_box=list(evidence['plot_box_source']),
        diagnostics={k: deepcopy(evidence[k]) for k in ('palette_identity', 'palette_recovery',
                     'blend_uncertainty_version') if k in evidence},
        arrays={k: np.asarray(evidence[k]).copy() for k in FIELDS})


def validate(snapshot, series_ids, plot_box):
    """Fail closed on stale, reordered, incomplete or malformed saved fields."""
    if (snapshot.get('version') != VERSION or snapshot.get('series_ids') != list(series_ids)
            or snapshot.get('plot_box') != list(plot_box)):
        raise ValueError('Frozen colour evidence identity/geometry mismatch')
    n = len(series_ids)
    x0, y0, x1, y1 = plot_box
    shape = (y1-y0, x1-x0)
    arrays = snapshot.get('arrays', {})
    if set(arrays) != set(FIELDS):
        raise ValueError('Incomplete frozen colour fields')
    for key in FIELDS:
        value = np.asarray(arrays[key])
        expected = ((n, *shape) if key in SERIES_FIELDS else shape if key in PLANE_FIELDS
                    else (n, n) if key == 'equivalent_colours' else (n, 3) if key == 'palette_rgb' else (3,))
        maximum = 255. if key in ('palette_rgb', 'paper_bgr') else 1.
        if (value.shape != expected or not np.isfinite(value).all()
                or np.any((value < 0) | (value > maximum))):
            raise ValueError('Invalid frozen colour field: ' + key)
        if key == 'equivalent_colours' and (not np.isin(value, [0, 1]).all()
                or not np.array_equal(value, value.T) or not np.diag(value).all()):
            raise ValueError('Invalid frozen palette equivalence')


def restore(snapshot, series_ids, plot_box):
    validate(snapshot, series_ids, plot_box)
    result = {k: np.asarray(v).copy() for k, v in snapshot['arrays'].items()}
    result['equivalent_colours'] = result['equivalent_colours'].astype(bool)
    result.update(deepcopy(snapshot.get('diagnostics', {})))
    result['colour_evidence_source'] = VERSION
    return result


def export(snapshot, archive):
    """Store numeric fields in the existing bound evidence NPZ, not JSON lists."""
    if snapshot is None:
        return None
    validate(snapshot, snapshot['series_ids'], snapshot['plot_box'])
    metadata = {k: deepcopy(v) for k, v in snapshot.items() if k != 'arrays'}
    metadata['array_keys'] = {k: 'recovered_colour_' + k for k in FIELDS}
    for key, name in metadata['array_keys'].items():
        value = np.asarray(snapshot['arrays'][key])
        # Portable sessions intentionally allow 2-D numeric arrays only.
        # Flatten the series axis into rows; keep that archive safety contract.
        archive[name] = value.reshape(-1, value.shape[-1])
    return metadata


def hydrate(metadata, archive, series_ids, plot_box):
    if metadata is None:
        return None
    if set(metadata.get('array_keys', {})) != set(FIELDS):
        raise ValueError('Incomplete frozen colour archive keys')
    result = {k: deepcopy(v) for k, v in metadata.items() if k != 'array_keys'}
    try:
        result['arrays'] = {k: np.asarray(archive[name]).copy() for k, name in metadata['array_keys'].items()}
    except KeyError as exc:
        raise ValueError('Missing frozen colour archive array') from exc
    n = len(series_ids)
    h, w = plot_box[3]-plot_box[1], plot_box[2]-plot_box[0]
    for key in SERIES_FIELDS:
        if result['arrays'][key].shape != (n*h, w):
            raise ValueError('Invalid flattened colour field: ' + key)
        result['arrays'][key] = result['arrays'][key].reshape(n, h, w)
    if result['arrays']['paper_bgr'].shape != (1, 3):
        raise ValueError('Invalid frozen paper colour')
    result['arrays']['paper_bgr'] = result['arrays']['paper_bgr'].reshape(3)
    validate(result, series_ids, plot_box)
    return result

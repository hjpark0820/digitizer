"""Validated Step-5 handoff without rerunning full marker detection.

This branch deliberately does not seed proposals into the active point set.
The correction engine alone may promote a supplied suppressed candidate.
Per-series correction can recheck individual candidates against bound templates.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np
from color_tentative_policy_v46 import filter_strong_candidates, policy_info

VERSION = 'v46_tentative_step5_v1'


def pixel_sha256(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _box(value, shape, nullable=False):
    if value is None and nullable:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Step-5 requires an inclusive xyxy box')
    a = np.asarray(value, float)
    if not np.isfinite(a).all() or not np.equal(a, np.round(a)).all():
        raise ValueError('Step-5 boxes must contain finite integer pixels')
    x0, y0, x1, y1 = map(int, a)
    if not (0 <= x0 <= x1 < shape[1] and 0 <= y0 <= y1 < shape[0]):
        raise ValueError('Step-5 box lies outside the source raster')
    return [x0, y0, x1, y1]


def _in(x, y, box):
    return box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _points(points, plot, legend, suppressed=False):
    result = []
    for p in points:
        q = dict(p)
        x, y = float(q['cx']), float(q['cy'])
        if not np.isfinite([x, y]).all():
            raise ValueError('Step-5 points must be finite source coordinates')
        # Edge/subpixel points may be clipped out, never moved to a new centre.
        if not _in(x, y, plot) or _in(x, y, legend):
            continue
        q.update(cx=x, cy=y)
        if suppressed:
            q.update(class_name='suppressed', class_idx=-1, existence='unknown')
        result.append(q)
    return result


def _reference_paths(records, names):
    """Validate embedded source-coordinate paths without relabelling inference.

    Legacy handoffs lack bound raw paths. Do not load an unbound sidecar from
    another run or substitute marker chords: the path corrector preserves those
    series until detection is rerun with the current exporter.
    """
    if records is None:return {}
    if not isinstance(records,dict) or not set(records).issubset(names):
        raise ValueError('Invalid v46 reference path series identities')
    result={}
    for name,record in records.items():
        if not isinstance(record,dict) or record.get('series_id',name)!=name:
            raise ValueError('Reference path identity does not match its curve')
        path=np.asarray(record.get('path',[]),float)
        if path.size==0:path=np.empty((0,2),float)
        if path.ndim!=2 or path.shape[1]!=2 or not np.isfinite(path).all():
            raise ValueError(f'Invalid finite source path for {name}')
        if len(path)>1 and np.any(np.diff(path[:,0])<=0):
            raise ValueError('Reference path x must be strictly increasing')
        for field in ('observed','filled'):
            values=np.asarray(record.get(field,[]))
            if values.shape!=(len(path),) or not np.isin(values,[False,True]).all():
                raise ValueError(f'Invalid {field} reference flags for {name}')
        val=np.asarray(record.get('val',np.ones(len(path))),float)
        if val.shape!=(len(path),) or not np.isfinite(val).all() or (val<0).any() or (val>1).any():
            raise ValueError(f'Invalid reference confidence for {name}')
        result[name]=dict(record,series_id=name,path=path.tolist(),val=val.tolist(),
                          observed=np.asarray(record['observed'],bool).tolist(),
                          filled=np.asarray(record['filled'],bool).tolist())
    return result


def load_payload(out_dir, image, plot_area, legend_box, names):
    """Return a validated v46 payload, or None for the untouched legacy route."""
    base = Path(out_dir)
    source = base / 'step5_inputs.json'
    if not source.exists():
        return None
    raw = json.loads(source.read_text(encoding='utf-8'))
    if raw.get('version') != VERSION:
        return None
    if raw.get('image_shape') != list(image.shape):
        raise ValueError('v46 Step-5 source image shape changed; re-run detection')
    if raw.get('image_sha256') != pixel_sha256(image):
        raise ValueError('v46 Step-5 source image pixels changed; re-run detection')
    plot = _box(raw.get('plot_area'), image.shape)
    legend = _box(raw.get('legend_box'), image.shape, nullable=True)
    if plot != list(plot_area) or legend != (list(legend_box) if legend_box is not None else None):
        raise ValueError('v46 Step-5 plot/legend geometry changed; re-run detection')
    rows = raw.get('curves', [])
    row_names = [r.get('name') for r in rows]
    if any(not isinstance(n, str) or not n or Path(n).name != n for n in row_names):
        raise ValueError('v46 Step-5 curve IDs must be safe directory names')
    if len(set(row_names)) != len(row_names) or len(set(names)) != len(names) or set(row_names) != set(names):
        raise ValueError('v46 Step-5 series identities differ from edit_data.json')
    archive_name = raw.get('masks_npz', '')
    if not archive_name or Path(archive_name).name != archive_name:
        raise ValueError('v46 Step-5 masks_npz must name a sibling evidence archive')
    archive = base / archive_name
    x0, y0, x1, y1 = plot
    local_shape = (y1-y0+1, x1-x0+1)
    with np.load(archive, allow_pickle=False) as arrays:
        hydrated = []
        for row in rows:
            r = dict(row)
            mask = np.asarray(arrays[r['mask_key']])
            if mask.shape != local_shape or not np.isfinite(mask).all():
                raise ValueError(f"Invalid source-local Step-5 mask for {r['name']}")
            full = np.zeros(image.shape[:2], bool)
            full[y0:y1+1, x0:x1+1] = mask.astype(bool)
            if legend is not None:
                a, b, c, d = legend
                full[b:d+1, a:c+1] = False
            r['ink_mask'] = full
            r['init_points'] = _points(r.get('init_points', []), plot, legend)
            r['init_suppressed'], r['suppressed_admission'] = filter_strong_candidates(
                _points(r.get('init_suppressed', []), plot, legend, True))
            segs = np.asarray(r.get('segments_override', []), float)
            if segs.size == 0:
                r['segments_override'] = []
            else:
                if segs.ndim != 2 or segs.shape[1] != 4 or not np.isfinite(segs).all():
                    raise ValueError(f"Invalid source-coordinate segments for {r['name']}")
                if any(not _in(s[0], s[1], plot) or not _in(s[2], s[3], plot) for s in segs):
                    raise ValueError('v46 Step-5 segment endpoint outside plot')
                r['segments_override'] = [tuple(map(float, s)) for s in segs]
            if r.get('template_key'):
                alpha = np.asarray(arrays[r['template_key']], np.float32)
                if alpha.ndim != 2 or not alpha.size or not np.isfinite(alpha).all() or np.min(alpha) < 0 or np.max(alpha) > 1:
                    raise ValueError(f"Invalid native marker alpha for {r['name']}")
                center = r.get('template_center', [(alpha.shape[1]-1)/2, (alpha.shape[0]-1)/2])
                if len(center) != 2 or not np.isfinite(center).all() or not (0 <= center[0] < alpha.shape[1] and 0 <= center[1] < alpha.shape[0]):
                    raise ValueError('Invalid native marker template centre')
                r.update(marker_alpha=alpha.copy(), template_center=list(map(float, center)))
            hydrated.append(r)
        from color_recovery_evidence_v46 import hydrate
        frozen_colours = hydrate(raw.get('frozen_colour_evidence'), arrays,
            [r['name'] for r in hydrated if r.get('template_key')], [x0, y0, x1+1, y1+1])
    grid = np.asarray(raw.get('grid_xs', []), float)
    if grid.ndim != 1 or not np.isfinite(grid).all() or np.any(grid < x0) or np.any(grid > x1):
        raise ValueError('Invalid source-coordinate Step-5 x grid')
    references=_reference_paths(raw.get('reference_paths'),set(row_names))
    result = dict(raw, curves=hydrated, grid_xs=sorted(set(grid.tolist())),
                  frozen_colour_evidence=frozen_colours,
                  reference_paths=references,
                  reference_status='embedded_validated_paths' if references else 'no_bound_reference_rerun_detection',
                  suppressed_policy=policy_info())
    result['suppressed_admission'] = {
        key: sum(r['suppressed_admission'][key] for r in hydrated)
        for key in ('input_count', 'admitted_count', 'excluded_count')}
    result['identity'] = dict(version=VERSION, image_sha256=raw['image_sha256'],
        image_shape=raw['image_shape'], plot_area=plot, legend_box=legend,
        series_names=sorted(row_names))
    return result


def validate_previous_state(previous, payload):
    if previous is None:
        return
    if previous.get('identity') != payload['identity']:
        raise ValueError('Previous Step-5 state does not match v46 image, geometry and series identities')
    if set(previous.get('curves', {})) != set(payload['identity']['series_names']):
        raise ValueError('Previous Step-5 state has different curve IDs')


def alpha_marker_renderer(alpha, center):
    """Native-size alpha stamp; no default-r=4 bug or implicit circle fallback."""
    a = np.asarray(alpha, np.float32)
    cx0, cy0 = center

    def draw(canvas, cx, cy, class_idx, r=None):
        x0, y0 = int(np.floor(cx-cx0)), int(np.floor(cy-cy0))
        fx, fy = float(cx-cx0-x0), float(cy-cy0-y0)
        # One-pixel padding keeps translated fractional boundary ink in bounds.
        shifted = cv2.warpAffine(a, np.array([[1, 0, fx], [0, 1, fy]], np.float32),
                                 (a.shape[1]+1, a.shape[0]+1), flags=cv2.INTER_LINEAR)
        x1, y1 = x0+shifted.shape[1], y0+shifted.shape[0]
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(canvas.shape[1], x1), min(canvas.shape[0], y1)
        if xa >= xb or ya >= yb:
            return
        ink = np.rint(255*(1-shifted[ya-y0:yb-y0, xa-x0:xb-x0])).astype(np.uint8)
        # Composition is the union of connecting-line and marker coverage.
        np.minimum(canvas[ya:yb, xa:xb], ink[..., None], out=canvas[ya:yb, xa:xb])
    return draw


def correct_payload_curves(image, curves, payload, *, out_dir, max_iters=None,
                           previous_state=None, workers=None, engine_dir=None, log_fn=print,
                           path_options=None):
    """Route each colour series by observed connection evidence.

    SSIM remains available only through the unrelated legacy colour pipeline.
    Fitted/marker-only series use image-backed native candidates, not path loss.
    Uncertain series preserve current points; no silent SSIM fallback.
    """
    validate_previous_state(previous_state,payload)
    from color_series_correction_v46 import correct_series_payload_curves
    return correct_series_payload_curves(image,curves,payload,out_dir=out_dir,max_iters=max_iters,
        previous_state=previous_state,workers=workers,engine_dir=engine_dir,log_fn=log_fn,
        path_options=path_options)


def _correct_payload_curves_ssim_legacy(image, curves, payload, *, out_dir, max_iters=None,
                           previous_state=None, workers=None, engine_dir=None, log_fn=print):
    """Use only current active points, supplied tentative pool and path segments."""
    from color_step5 import _load_step5, build_reference, estimate_stroke
    validate_previous_state(previous_state, payload)
    by_name = {c['name']: c for c in payload['curves']}
    nworkers = max(1, min(len(curves) or 1, int(workers or min(4, os.cpu_count() or 1))))

    def run(cv):
        row = by_name[cv['name']]
        if engine_dir:
            filename = Path(engine_dir) / '5_correction_color.py'
            spec = importlib.util.spec_from_file_location('v46_explicit_colour_correction', filename)
            eng = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eng)
        else:
            eng = _load_step5()
        eng.PARALLEL_WORKERS = max(1, (os.cpu_count() or 1)//nworkers)
        cls = row.get('marker_class')
        custom = row.get('marker_alpha') is not None
        if cls not in eng.CLASS_NAMES and not custom:
            log_fn(f"[v46 Step-5] {cv['name']}: unsupported marker without template; preserving points")
            return dict(name=cv['name'], points=list(cv['points']), n_before=len(cv['points']),
                        n_after=len(cv['points']), ssim_before=None, ssim_after=None,
                        suppressed=row['init_suppressed'], status='unsupported_marker_not_corrected')
        # Class is an engine routing token only when a native template replaces drawing.
        if cls not in eng.CLASS_NAMES:
            cls = eng.CLASS_NAMES[0]
            log_fn(f"[v46 Step-5] {cv['name']}: template-only geometry (engine token {cls}, not a circle model)")
        ci = eng.CLASS_NAMES.index(cls)
        active = _points([dict(cx=x, cy=y, class_name=cls, class_idx=ci, confidence=1.)
                          for x, y in cv['points']], payload['plot_area'], payload['legend_box'])
        pool = row['init_suppressed']
        if previous_state is not None:
            pool = previous_state.get('suppressed', {}).get(cv['name'], [])
        # Apply on resume too: an older state can still contain the broad pool.
        suppressed, admission = filter_strong_candidates(
            _points(pool, payload['plot_area'], payload['legend_box'], True))
        if admission['excluded_count']:
            log_fn(f"[v46 Step-5] {cv['name']}: omitted {admission['excluded_count']} non-strong saved hypotheses")
        # Exact/subpixel duplicates are excluded; do not collapse nearby distinct shapes.
        suppressed = [p for p in suppressed if all((p['cx']-q['cx'])**2+(p['cy']-q['cy'])**2 > 1.
                                                 for q in active)]
        mask = row['ink_mask']
        lw, mr = estimate_stroke(mask)
        eng.RENDER_LW = max(1., float(row.get('render_linewidth') or lw))
        if custom:
            eng._cv_draw_marker = alpha_marker_renderer(row['marker_alpha'], row['template_center'])
        else:
            original = eng._cv_draw_marker
            eng._cv_draw_marker = lambda canvas, x, y, c, r=None: original(canvas, x, y, c, r=int(mr if r is None else r))
        folder = Path(out_dir) / cv['name']
        folder.mkdir(parents=True, exist_ok=True)
        reference = build_reference(image, mask)
        cv2.imwrite(str(folder/'reference.png'), reference)
        log_fn(f"[v46 Step-5] {cv['name']}: {len(active)} active, {len(suppressed)} tentative suppressed, "
               f"{len(row['segments_override'])} supplied path segments; no automatic seeding")
        if not mask.any():
            return dict(name=cv['name'], points=[(p['cx'], p['cy']) for p in active],
                        n_before=len(active), n_after=len(active), suppressed=suppressed,
                        ssim_before=None, ssim_after=None, status='empty_reference_not_corrected')
        try:
            result = eng.run_correction(img_path=str(folder/'reference.png'), model_path=None,
                detector_py_path=None, known_classes=[cls], out_dir=str(folder), mode_xs=None,
                prep_info=dict(plot_area=tuple(payload['plot_area']), user_plot_area=tuple(payload['plot_area']),
                               legend_box=tuple(payload['legend_box']) if payload['legend_box'] else None, clean_fn=None),
                max_iters=max_iters, return_diag_imgs=False, init_points=active,
                init_suppressed=suppressed, segments_override=list(row['segments_override']),
                grid_xs_override=list(payload['grid_xs']))
            p = result.get('P_current', [])
            s = _points(result.get('S_current', []), payload['plot_area'], payload['legend_box'], True)
            s, _ = filter_strong_candidates(s)
            # Legacy ADD can choose a tentative centre without marking ACTIVATE.
            # Persist the remaining pool only, regardless of the action label.
            s = [q for q in s if all((q['cx']-a['cx'])**2+(q['cy']-a['cy'])**2 > 1. for a in p)]
            h = result.get('history', [])
            return dict(name=cv['name'], points=[(float(q['cx']), float(q['cy'])) for q in p],
                        n_before=len(active), n_after=len(p), suppressed=s,
                        ssim_before=h[0][3] if h else None, ssim_after=h[-1][3] if h else None,
                        status='corrected', out_dir=str(folder))
        except Exception as error:
            log_fn(f"[v46 Step-5] {cv['name']}: correction failed; preserving supplied active points: {error}")
            return dict(name=cv['name'], points=[(p['cx'], p['cy']) for p in active],
                        n_before=len(active), n_after=len(active), suppressed=suppressed,
                        ssim_before=None, ssim_after=None, status='failed_preserved', error=str(error))

    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        return list(executor.map(run, curves))

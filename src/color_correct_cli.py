"""
color_correct_cli.py -- path-shape Step 5 for v46; legacy colour compatibility.

Runs as a post-process on a finished colour job, so run_A4_auto_v<N>.py itself is
not touched:

    python color_correct_cli.py <input.png> <out_dir> --plot-area x0,y0,x1,y1
           [--legend-area x0,y0,x1,y1] [--correct-iters N] [--prev-state state.json]

It reads <out_dir>/edit_data.json (curves with their display RGB + points),
corrects each colour curve independently via color_step5, writes the corrected
edit_data.json back, redraws the overlay, and saves a state file so a following
run can continue from this result (same behaviour as the B&W Step 5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Windows consoles default to cp949/cp1252, which cannot encode characters the
# correction prints (e.g. an arrow).  Force UTF-8 with replacement so a stray
# glyph can never abort a curve's correction.
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from color_step5 import correct_colour_curves, swatch_ink_mask  # noqa: E402

STATE_NAME = "color_correction_state.json"


def _p4(s):
    if not s or not s.strip():
        return None
    v = [int(round(float(t))) for t in s.split(",")]
    return tuple(v) if len(v) == 4 else None


def _draw_overlay(img_bgr, curves):
    """Original image with the corrected points marked in each curve's colour."""
    ov = img_bgr.copy()
    ref = max(ov.shape[:2])
    ro = max(4, int(ref * 0.011))
    ri = max(2, int(ro * 0.30))
    th = max(1, int(ref * 0.002))
    for c in curves:
        rgb = c.get("rgb") or [200, 0, 0]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        for p in c.get("points", []):
            x, y = int(round(p["x"])), int(round(p["y"]))
            cv2.circle(ov, (x, y), ro, bgr, th)
            cv2.circle(ov, (x, y), ri, bgr, -1)
    return ov


def main(argv=None):
    supplied={str(v).split('=',1)[0] for v in (sys.argv[1:] if argv is None else argv)
              if str(v).startswith('--')}
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--plot-area", default="")
    ap.add_argument("--legend-area", default="")
    ap.add_argument("--correct-iters", default="")
    ap.add_argument("--prev-state", default="")
    ap.add_argument("--engine-dir", default="")
    ap.add_argument('--path-metric',choices=('chamfer','directional_chamfer','hausdorff95'),default='chamfer')
    ap.add_argument('--path-model',choices=('linear','pchip'),default='pchip')
    ap.add_argument('--path-reference',choices=('observed','estimated_path'),default='estimated_path')
    ap.add_argument('--path-inferred-weight',type=float,default=.25)
    ap.add_argument('--limited-path-model',choices=('linear','pchip'),default=None,
        help='Supported limited-series model; fresh production default is linear; connected model is unchanged')
    ap.add_argument('--limited-completeness-weight',type=float,default=0.,
        help='Override completeness weight (fresh v46 default .5; irregular x grids abstain)')
    ap.add_argument('--limited-image-policy',choices=('weighted','hard'),default=None,
        help='Fresh v46 default is weighted; hard is the historical comparison policy')
    ap.add_argument('--limited-image-weight',type=float,default=None,
        help='Override image weight (fresh v46 default .5); explicit 0 is a no-image-cost ablation')
    ap.add_argument('--require-v46-path',action='store_true',help='Never fall back to legacy SSIM when v46 evidence is unavailable')
    ap.add_argument('--use-edit-points',action='store_true',help='Use current editor coordinates with saved evidence')
    a = ap.parse_args(argv)

    img = cv2.imread(a.image)
    if img is None:
        print("[color-correct] ERROR: cannot read", a.image)
        sys.exit(2)
    H, W = img.shape[:2]

    ed_path = os.path.join(a.out_dir, "edit_data.json")
    if not os.path.exists(ed_path):
        print("[color-correct] ERROR: edit_data.json not found in", a.out_dir)
        sys.exit(2)
    with open(ed_path, "r", encoding="utf-8") as f:
        ed = json.load(f)
    if ed.get('correction_available') is False:
        raise ValueError(ed.get('correction_unavailable_reason') or
                         'Correction requires a validated reference; detection output is preserved')

    pa = _p4(a.plot_area) or (0, 0, W - 1, H - 1)
    lg = _p4(a.legend_area)
    iters = int(a.correct_iters) if a.correct_iters.strip() else None
    if iters is not None and iters<1:ap.error('--correct-iters must be positive')
    if not 0<a.path_inferred_weight<=1:ap.error('--path-inferred-weight must be in (0,1]')
    if not np.isfinite(a.limited_completeness_weight) or a.limited_completeness_weight<0:
        ap.error('--limited-completeness-weight must be finite and nonnegative')
    if a.limited_image_weight is not None and (not np.isfinite(a.limited_image_weight) or a.limited_image_weight<0):
        ap.error('--limited-image-weight must be finite and nonnegative')

    # Equal colours contain multiple paths. Reuse their saved typed BW evidence,
    # not a single colour path or a newly extracted marker template.
    from color_group_runtime_v46 import VERSION as GROUP_VERSION, correct_saved as correct_groups
    if ed.get('detector_backend') == GROUP_VERSION:
        if a.limited_completeness_weight or a.limited_image_weight is not None or a.limited_image_policy or a.limited_path_model:
            raise ValueError('Limited completeness/image objectives are not implemented for the colour-group/BW backend')
        if a.plot_area.strip() and list(pa) != ed['plot_area']:
            raise ValueError('Colour-group correction plot ROI mismatch')
        from pathlib import Path
        from color_group_runtime_v46 import STATE_FILE
        state_path=Path(a.out_dir)/STATE_FILE
        if a.prev_state.strip() and Path(a.prev_state).resolve()!=state_path.resolve():
            raise ValueError('Colour-group correction resumes the bound state inside this job; import the desired saved version first')
        if a.legend_area.strip():
            with state_path.open(encoding='utf-8') as handle:group_state=json.load(handle)
            box=group_state['legend_box']
            if list(lg)!=[box[0],box[1],box[2]-1,box[3]-1]:
                raise ValueError('Colour-group correction legend ROI mismatch')
        correct_groups(a.out_dir, img, ed, iters if iters is not None else 5)
        return

    # Marker-free line charts have structural (not glyph) evidence and an
    # explicit endpoint prior. Never pass that pool through marker-only gates.
    from type3_v46.gui_adapter import DETECTION_NAME, correct_saved_detection
    if ed.get('series_mode') == 'line-only' or ed.get('detector_backend') == 'v46_type3_v1':
        if a.limited_completeness_weight or a.limited_image_weight is not None or a.limited_image_policy or a.limited_path_model:
            raise ValueError('Limited completeness/image objectives are not implemented for marker-free type3 correction')
        if not os.path.exists(os.path.join(a.out_dir, DETECTION_NAME)):
            raise ValueError('Type3 correction requires its bound detection evidence; rerun v46 line-only detection')
        correct_saved_detection(a.image, a.out_dir,
            plot_area=_p4(a.plot_area), legend_box=_p4(a.legend_area),
            iterations=iters if iters is not None else 5,
            previous_path=a.prev_state.strip() or None,
            engine_dir=a.engine_dir.strip() or None, use_edit_points=a.use_edit_points)
        return

    # Continue from a previous correction when one is supplied, so repeated
    # presses compose instead of restarting from the raw detection.
    prev = None
    prev_load_error = None
    if a.prev_state.strip() and os.path.exists(a.prev_state.strip()):
        try:
            with open(a.prev_state.strip(), "r", encoding="utf-8") as f:
                prev = json.load(f)
            print(f"[color-correct] continuing from previous state "
                  f"({len(prev.get('curves', {}))} curves)")
        except Exception as e:
            print(f"[color-correct] prev-state load failed ({e}); starting fresh")
            prev = None
            prev_load_error = str(e)

    # v46 makes the marker identity, native mask and tentative candidate pool
    # explicit. Legacy v45 exports have no matching version and keep the old
    # correction route unchanged.
    from color_step5_payload_v46 import load_payload, validate_previous_state, VERSION
    # Automatic detector geometry is authoritative when GUI/CLI boxes were not
    # explicitly supplied. load_payload still checks the image hash and bounds.
    evidence_path=os.path.join(a.out_dir,'step5_inputs.json')
    if os.path.exists(evidence_path):
        with open(evidence_path,encoding='utf-8') as handle:raw_evidence=json.load(handle)
        if raw_evidence.get('version')==VERSION:
            if not a.plot_area.strip():pa=raw_evidence['plot_area']
            if not a.legend_area.strip():lg=raw_evidence.get('legend_box')
    names = [c.get("name") or c.get("label") or f"curve{i}"
             for i, c in enumerate(ed.get("curves", []))]
    payload = load_payload(a.out_dir, img, pa, lg, names)
    if payload is not None:
        if a.prev_state.strip() and (prev_load_error or prev is None):
            raise ValueError('Cannot resume v46 Step-5 from a missing or unreadable state file')
        validate_previous_state(prev, payload)
        print(f"[color-correct] v46 per-series correction; connected metric: {a.path_metric}/{a.path_model}, "
              f"reference={a.path_reference}, inferred_weight={a.path_inferred_weight}; "
              "all modes: shared-x occlusion candidates; limited ADD: score reconstruction/path before image admission; "
              "uncertain: preserve original points")
        admission = payload['suppressed_admission']
        print(f"[color-correct] strong evidence only: {admission['admitted_count']} suppressed; "
              f"{admission['excluded_count']} weak/missing-evidence candidates omitted")
    elif a.engine_dir.strip() or a.require_v46_path:
        raise ValueError('v46 path correction requires a validated Step-5 payload; '
                         'rerun supported legend detection. Legacy SSIM fallback is disabled for this request.')

    from color_step5_defaults_v46 import cli_options
    path_options=cli_options(a,supplied,supported=payload is not None,
                             previous=prev if payload is not None else None)
    a.path_metric=path_options['metric'];a.path_model=path_options['model']
    a.path_reference=path_options['reference_policy'];a.path_inferred_weight=path_options['inferred_weight']
    a.limited_completeness_weight=path_options.get('limited_completeness_weight',0.)
    a.limited_image_weight=path_options.get('limited_image_weight')
    if payload is not None:
        print('[color-correct] '+('saved' if prev is not None else 'new')+' objective settings: '+json.dumps(path_options))

    curves_in = []
    for c in ed.get("curves", []):
        name = c.get("name") or c.get("label") or f"curve{len(curves_in)}"
        pts = [(float(p["x"]), float(p["y"])) for p in c.get("points", [])]
        if prev and not a.use_edit_points and name in (prev.get("curves") or {}):
            pts = [tuple(v) for v in prev["curves"][name]]
        curves_in.append({
            "name": name,
            "swatch_rgb": tuple(c.get("rgb") or (0, 0, 0)),
            "points": pts,
            "ink_mask": None,          # built from the swatch colour
        })

    # Shared x-column grid: data points across curves line up on the same x
    # positions, so cluster every curve's x values into columns.  These columns
    # are what the grid x path suppressed candidates are anchored to.
    _allx = sorted(x for c in curves_in for (x, _y) in c["points"])
    grid_xs = []
    if _allx:
        _tolx = max(4.0, (pa[2] - pa[0]) * 0.012)
        _grp = [_allx[0]]
        for v in _allx[1:]:
            if v - _grp[-1] <= _tolx:
                _grp.append(v)
            else:
                grid_xs.append(sum(_grp) / len(_grp)); _grp = [v]
        grid_xs.append(sum(_grp) / len(_grp))
    if payload is not None:
        grid_xs = payload['grid_xs']

    print(f"[color-correct] {len(curves_in)} curves, "
          f"{sum(len(c['points']) for c in curves_in)} points, "
          f"plot_area={pa}, iters={iters or 'default'}, "
          f"grid={len(grid_xs)} x-columns")

    if a.limited_image_weight is not None:
        print(f'[color-correct] supported limited paths: image veto disabled; weighted image loss mu={a.limited_image_weight}')
    results = correct_colour_curves(
        img, curves_in, pa, legend_box=lg,
        max_iters=iters, out_dir=os.path.join(a.out_dir, "step5"),
        grid_xs=grid_xs,
        explicit_payload=payload,
        previous_state=prev if payload is not None else None,
        engine_dir=a.engine_dir.strip() or None,
        path_options=path_options,
    )
    by_name = {r["name"]: r for r in results}

    # write corrected points back into edit_data.json
    for c in ed.get("curves", []):
        name = c.get("name") or c.get("label")
        r = by_name.get(name)
        if not r:
            continue
        c["points"] = [{"x": float(x), "y": float(y)} if payload is not None
                       else {"x": int(round(x)), "y": int(round(y))}
                       for (x, y) in sorted(r["points"])]
        if payload is not None:
            row=next(row for row in payload['curves'] if row['name']==name)
            c['marker_class']=row.get('marker_class')
            c['correction_status']=r.get('status')
            c['connection_mode']=r.get('mode','connected')
            # Separate review evidence from exported measurement coordinates.
            c['suppressed_candidates']=r.get('suppressed', [])
            c['suppressed_summary']=dict(total=len(r.get('suppressed', [])),
                structural=sum(p.get('source')=='colocated_original_marker' for p in r.get('suppressed', [])),
                note='Unconfirmed alternatives; excluded from active points and numeric data exports.')
            if 'path_objective' in r:
                c['step5_path_objective']=r['path_objective']
                c['step5_objective_before']=r.get('objective_before')
                c['step5_objective_after']=r.get('objective_after')
                c['step5_candidate_reviews']=[dict(
                    candidate_id=q.get('candidate_id'),x=q['point']['cx'],y=q['point']['cy'],
                    iteration=q['iteration'],selected=q.get('selected',False),
                    action=q['action'],reason=q['reason'],admissible=q['admissible'],
                    image_score=q.get('test',{}).get('score'),image_admissible=q.get('image_admissible',False),
                    path_comparison=q['path_comparison']) for q in r.get('candidate_reviews',[])]
                if r['path_objective'].get('image_policy')=='weighted_cost':
                    metadata={(p['cx'],p['cy']):p for p in r['series_state']['active_points']}
                    for point in c['points']:
                        source=metadata.get((point['x'],point['y']),{})
                        if source.get('image_confirmed') is False:
                            point.update({k:source[k] for k in ('state','image_confirmed','validation_basis','candidate_id') if k in source})
            c['reconstruction']=dict(model=r.get('model',a.path_model),
                points=[dict(x=float(x),y=float(y)) for x,y in r.get('reconstructed_path',[])],
                anchors=[dict(p) for p in c['points']],status='unvalidated_path_hypotheses')
            if 'reconstructed_segments' in r:
                c['reconstruction']['segments'] = [[dict(x=float(x),y=float(y)) for x,y in run]
                                                    for run in r['reconstructed_segments']]
                c['reconstruction']['source'] = r.get('reconstruction_source')
    if payload is not None:
        ed['correction']=dict(backend='v46_series_routed_v1',metric=a.path_metric,model=a.path_model,
            reference_policy=a.path_reference,inferred_weight=a.path_inferred_weight,
            limited_completeness_weight=a.limited_completeness_weight,
            existence='unvalidated_hypotheses',status={r['name']:r.get('status') for r in results})
        if a.limited_image_weight is not None:
            ed['correction']['limited_image_weight']=a.limited_image_weight
        ed['correction']['limited_model']=path_options.get('limited_model',a.path_model)
    with open(ed_path, "w", encoding="utf-8") as f:
        json.dump(ed, f)

    cv2.imwrite(os.path.join(a.out_dir, "data_points_overlay.png"),
                _draw_overlay(img, ed.get("curves", [])))

    state = {"curves": {r["name"]: [[float(x), float(y)] for x, y in r["points"]]
                         for r in results}}
    if payload is not None:
        state.update(identity=payload['identity'],
                     suppressed={r['name']: r.get('suppressed', []) for r in results},
                     status={r['name']: r.get('status') for r in results},
                     backend='v46_series_routed_v1',
                     series_routing_fingerprint=results[0].get('series_routing_fingerprint') if results else None,
                     series_state={r['name']:r['series_state'] for r in results if r.get('series_state') is not None},
                     path_state={r['name']:r['path_state'] for r in results if r.get('path_state') is not None},
                     path_options=path_options)
    with open(os.path.join(a.out_dir, STATE_NAME), "w", encoding="utf-8") as f:
        json.dump(state, f)

    for r in results:
        if r.get('path_score_before') is not None:
            print(f"[color-correct]   {r['name']}: {r['n_before']} -> {r['n_after']} hypotheses, "
                  f"{r.get('metric',a.path_metric)}/{r.get('action_path_model',r.get('model',a.path_model))} "
                  f"{r['path_score_before']:.5f} -> {r['path_score_after']:.5f}; {r['status']}")
        elif r.get("ssim_before") is None:
            print(f"[color-correct]   {r['name']}: {r['n_before']} -> {r['n_after']} pts; {r.get('status','legacy')}")
        else:
            print(f"[color-correct]   {r['name']}: {r['n_before']} -> {r['n_after']} pts, "
                  f"1-SSIM {r['ssim_before']:.5f} -> {r['ssim_after']:.5f}")
    print("[color-correct] done ->", a.out_dir)
    failed=[r['name'] for r in results if r.get('status')=='failed_preserved']
    if failed:raise RuntimeError('Correction failed for '+', '.join(failed)+'; affected input points were preserved')


if __name__ == "__main__":
    main()

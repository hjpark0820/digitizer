"""
Command-line wrapper around chartocode2's run_detection(), used by the unified
server so B&W detection runs in its OWN process (main thread) — exactly like the
standalone GUI. This avoids the "partially initialised module 'torch' (circular
import)" error that happens when torch is first imported inside a uvicorn worker
thread. It writes the common edit_data.json (+ overlay) into <out_dir>.

Usage:
  python bw_detect_cli.py <image> <out_dir>
      --plot-area x0,y0,x1,y1
      [--legend-area x0,y0,x1,y1]
      [--x-min V --x-max V --y-min V --y-max V]
      [--x-log] [--y-log]
      [--classes filled_circle,open_square,...]   # empty => automatic legend classes
      [--conf 0.3] [--errorbars 0|1]
      [--refinement-backend auto|cpu|cuda]        # default: automatic GPU/CPU
      [--grid-backend auto|cpu|cuda] [--window-backend auto|cpu|cuda]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
from bw_series_identity import identity_fields, series_color

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BW_ALL_CLASSES = [
    "filled_circle", "open_circle", "filled_square", "open_square",
    "filled_triangle", "open_triangle", "filled_inv_triangle", "open_inv_triangle",
    "filled_rhombus", "open_rhombus", "x_marker", "plus_marker",
]

BW_STATE_VERSION = 'bw_correction_state_v46_v1'


def _json_ready(value):
    """Keep measured masks, candidate evidence and immutable P0 provenance."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, 'tolist'):
        return _json_ready(value.tolist())
    return value


def _state_binding(image, plot_area, legend_area, upscale, backend):
    """Bind saved pixel coordinates to the submitted raster and selected ROIs."""
    return {'image_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'image_shape': list(image.shape), 'plot_area': list(plot_area),
            'legend_area': list(legend_area) if legend_area is not None else None,
            'upscale': float(upscale), 'point_backend': backend}


def _validate_grid_state(state, binding):
    if state.get('format_version') != BW_STATE_VERSION:
        raise ValueError('Previous B&W state predates swatch-safe SSIM correction. '
                         'Run detection once with the current v46 before continuing Step 5.')
    if state.get('binding') != binding:
        raise ValueError('Previous B&W state belongs to a different image, plot/legend '
                         'selection, scale or detector backend. Run a fresh detection.')
    if not isinstance(state.get('P_current'), list) or not isinstance(state.get('S_current'), list):
        raise ValueError('Previous B&W state has invalid active/suppressed point arrays.')


def _p4(s):
    return tuple(int(float(v)) for v in s.split(",")) if s else None


def main():
    # Direct v46 BW calls and the shared runner use the same explicit no-legend
    # implementation. An explicitly requested legacy ViT run is not rewritten.
    if '--point-backend=vit' not in sys.argv and not any(
            a=='--point-backend' and i+1<len(sys.argv) and sys.argv[i+1]=='vit'
            for i,a in enumerate(sys.argv)):
        from legend_optional_v46.cli import maybe_run
        if maybe_run(sys.argv[1:],default_mode='bw'):return
    _t_cli0 = time.perf_counter()
    # Force UTF-8 output. run_detection logs contain unicode (arrows, ellipsis);
    # on Windows the default console/pipe encoding is cp1252 and printing those
    # raises UnicodeEncodeError, killing the process AFTER detection succeeds.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _safe_log(*args):
        msg = " ".join(str(a) for a in args)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"))

    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--plot-area", required=True)
    ap.add_argument("--legend-area", '--legend-box', default="")
    ap.add_argument("--x-min", default="0"); ap.add_argument("--x-max", default="1")
    ap.add_argument("--y-min", default="0"); ap.add_argument("--y-max", default="1")
    ap.add_argument("--x-log", action="store_true")
    ap.add_argument("--y-log", action="store_true")
    ap.add_argument("--classes", default="")
    ap.add_argument("--conf", default="")
    ap.add_argument('--point-backend', choices=('grid_v46','vit'), default='grid_v46')
    ap.add_argument('--grid-fraction', type=float, default=0.5)
    ap.add_argument('--grid-overlap', type=float, default=0.5)
    ap.add_argument('--refinement-backend', choices=('auto', 'cpu', 'cuda'), default=None,
                    help='compute backend; defaults to BW_V46_REFINEMENT_BACKEND or auto; grid/window inherit it')
    ap.add_argument('--grid-backend', choices=('auto', 'cpu', 'cuda'), default=None,
                    help='optional grid-voting override; inherits refinement backend')
    ap.add_argument('--window-backend', choices=('auto', 'cpu', 'cuda'), default=None,
                    help='optional full-window rank-screening override; final decisions stay on CPU')
    ap.add_argument('--gpu-batch-size', type=int, default=512,
                    help='CUDA batch limit for center/window alignments or grid edge pixels')
    ap.add_argument('--diagnostic-detail', action='store_true', default=None,
                    help='render detailed intermediate diagnostic images (grid v46 default is lightweight)')
    ap.add_argument('--swatches-json', default='',
                    help='JSON list of [class_name, [x0,y0,x1,y1]]; exclusive right/bottom')
    ap.add_argument("--errorbars", default="")
    ap.add_argument('--skip-segments', action='store_true',
                    help='skip line/errorbar detection; preserve original marker pixels (B&W web workflow)')
    ap.add_argument("--scale", default="",
                    help="manual scale factor; v46 default=1x, legacy ViT default=legend estimate")
    ap.add_argument("--correct", action="store_true",
                    help="run structure-aware Step 5 (connected SSIM / restricted marker checks)")
    ap.add_argument("--correct-iters", default="",
                    help="max number of Step-5 correction iterations (empty = default cap)")
    ap.add_argument("--prev-state", default="",
                    help="path to a previous correction_state.json; when given, Step-5 "
                         "continues from that point set instead of re-detecting")
    a = ap.parse_args()
    if a.correct_iters.strip() and int(a.correct_iters) < 1:
        ap.error('--correct-iters must be a positive integer')

    os.makedirs(a.out_dir, exist_ok=True)

    # Resolve/probe on this process's main thread before importing bw_pipeline.
    # Automatic mode tolerates absent/broken CUDA; explicit CPU never needs
    # torch. Keep the original None/auto choices when forwarding below so the
    # pipeline retains automatic-runtime-fallback provenance.
    import time as _tmod
    _t_imp0 = _tmod.perf_counter()
    try:
        if a.point_backend == 'vit':
            # Legacy trained detection still requires torch, even when its
            # tensor operations run on CPU. Preserve that existing contract.
            import torch
            _cuda = torch.cuda.is_available()
            print(f'[bw-cli] torch {torch.__version__} (cuda_available={_cuda})')
            if not _cuda and "+cpu" in torch.__version__:
                print("[bw-cli] NOTE: this is a CPU-ONLY torch build (+cpu) — inference "
                      "runs on CPU (slow). Install a CUDA build to use the GPU.")
        else:
            from bw_compute_backend import resolve_compute_backends, ensure_cuda_available
            _compute = resolve_compute_backends(a.refinement_backend, a.grid_backend,
                                               a.window_backend, log_fn=_safe_log)
            # The resolver keeps an explicit CUDA choice strict, without
            # importing torch for selection alone. Preload it here as promised
            # by this CLI; a prior successful auto probe already initialized it.
            if ('cuda' in _compute['requested'].values()
                    and _compute['cuda_probe']['available'] is not True):
                ensure_cuda_available()
    except Exception as e:
        context = 'torch not available' if a.point_backend == 'vit' else 'compute backend initialization failed'
        print(f"[bw-cli] ERROR: {context}: {e}")
        sys.exit(3)

    import bw_pipeline           # GUI-free shared detection runtime
    import bw_to_edit_data
    _imports_seconds = _tmod.perf_counter()-_t_imp0
    print(f"[bw-cli] [timing] pipeline imports: {_imports_seconds:.2f}s")

    # BUILD SIGNATURE — print key module paths + mtime/size so the ACTUAL deployed
    # files can be verified (if a fix "does nothing", check these match your edits).
    try:
        import time as _tsig
        _srcdir = os.path.dirname(os.path.abspath(__file__))
        print("[bw-cli] BUILD SIGNATURE (src=%s):" % _srcdir)
        _build_files = (('run_A4_auto_v46.py', 'bw_detect_cli.py', 'bw_pipeline.py',
                         'bw_pipeline_v46.py', 'bw_grid_identity_v46.py',
                         'bw_step5_v46.py', 'bw_series_correction_v46.py', 'bw_series_structure_v46.py',
                         'bw_structural_edits_v46.py', 'bw_connection_alignment_v46.py',
                         'x_singleton_suppressed_v46.py', 'partial_swatch_detector.py',
                         'bw_compute_backend.py', 'bw_gpu_grid_votes.py',
                         'bw_gpu_candidate_centres.py', 'bw_gpu_candidate_verifier.py',
                         'bw_gpu_refinement.py', 'bw_gpu_window_verifier.py',
                         'bw_gpu_window_fused.py',
                         'occlusion_aware_window_verifier.py') if a.point_backend == 'grid_v46'
                        else ('bw_pipeline.py', '2_point_detection_adaptive_nms_v2.py',
                              '5_correction_v2.py', 'chart_marker_detector_v3.py'))
        for _sf in _build_files:
            _p = os.path.join(_srcdir, _sf)
            if os.path.exists(_p):
                _mt = _tsig.strftime('%Y-%m-%d %H:%M', _tsig.localtime(os.path.getmtime(_p)))
                print("[bw-cli]   %-38s mtime=%s size=%d" % (_sf, _mt, os.path.getsize(_p)))
            else:
                print("[bw-cli]   %-38s NOT FOUND" % _sf)
    except Exception as _sige:
        print("[bw-cli] BUILD SIGNATURE failed: %s" % _sige)

    _t_image0 = _tmod.perf_counter()
    img = cv2.imread(a.image)
    _image_read_seconds = _tmod.perf_counter()-_t_image0
    if img is None:
        print("[bw-cli] ERROR: could not read image"); sys.exit(2)
    H, W = img.shape[:2]
    pa = _p4(a.plot_area)
    xr = (float(a.x_min), float(a.x_max))
    yr = (float(a.y_min), float(a.y_max))
    kc = [c.strip() for c in a.classes.split(",") if c.strip()]
    if not kc:
        kc = None if a.point_backend == 'grid_v46' else list(BW_ALL_CLASSES)
    eb = None if a.errorbars == "" else (a.errorbars.strip() in ("1", "true", "yes", "on"))

    # Load a previous Step-5 state up front: when present the ViT detection is
    # redundant (its points are replaced by the saved ones), so we skip it.
    _prev_state = None
    _binding = _state_binding(img, pa, _p4(a.legend_area),
                             float(a.scale) if a.scale.strip() else 1.0, a.point_backend)
    if a.correct and a.prev_state.strip():
        try:
            with open(a.prev_state.strip(), "r", encoding="utf-8") as _pf:
                _prev_state = json.load(_pf)
            if a.point_backend == 'grid_v46':
                _validate_grid_state(_prev_state, _binding)
            print(f"[bw-cli] prev-state loaded: "
                  f"{len(_prev_state.get('P_current') or [])} active, "
                  f"{len(_prev_state.get('S_current') or [])} suppressed")
        except Exception as _pe:
            if a.point_backend == 'grid_v46':
                _safe_log(f'[bw-cli] Step-5 state validation failed: {_pe}')
                with open(os.path.join(a.out_dir, 'correction_status.json'), 'w', encoding='utf-8') as _sf:
                    json.dump({'requested': True, 'status': 'failed', 'engine': 'bw_step5_v46',
                               'error': str(_pe)}, _sf, indent=2)
                sys.exit(4)
            print(f"[bw-cli] prev-state load failed ({_pe}); starting fresh")
            _prev_state = None

    _t_det0 = _tmod.perf_counter()
    result = bw_pipeline.run_detection(
        **({'diagnostic_detail': a.diagnostic_detail} if a.diagnostic_detail is not None else {}),
        point_backend=a.point_backend,
        grid_fraction=a.grid_fraction, grid_overlap=a.grid_overlap,
        refinement_backend=a.refinement_backend, gpu_batch_size=a.gpu_batch_size,
        grid_backend=a.grid_backend, window_backend=a.window_backend,
        swatches=(json.loads(a.swatches_json) if a.swatches_json else None),
        skip_point_detection=bool(_prev_state),
        img_bgr=img, plot_area_px=pa, legend_area_px=_p4(a.legend_area),
        known_classes=kc, x_range=xr, y_range=yr,
        x_log=a.x_log, y_log=a.y_log, has_errorbars=False if a.skip_segments else eb,
        upscale=(float(a.scale) if a.scale.strip() else None),
        conf_thresh=(float(a.conf) if a.conf else None),
        stride=None, has_lines=not a.skip_segments, log_fn=_safe_log,
    )
    _detection_seconds = _tmod.perf_counter()-_t_det0
    _t_post0 = _tmod.perf_counter()
    print(f"[bw-cli] [timing] run_detection total: {_detection_seconds:.2f}s")
    if _prev_state:
        # If optional correction fails, keep the previous visible point set,
        # not the empty result from deliberately skipped point detection.
        up = float(result.get('upscale',1.0))
        result['detections'] = [
            {'class_name':p['class_name'],
             **identity_fields(p),
             'cx_px':float(p.get('cx',p.get('cx_px',0)))/up,
             'cy_px':float(p.get('cy',p.get('cy_px',0)))/up}
            for p in (_prev_state.get('P_current') or [])
            if p.get('class_name') != 'suppressed']
        # Corresponding fallback overlay; correction redraws on success.
        from bw_pipeline import MARKER_COLORS
        ov = img.copy()
        for p in result['detections']:
            c=series_color(p,MARKER_COLORS) if p.get('swatch_id') else MARKER_COLORS.get(p['class_name'],(20,160,20))
            cv2.drawMarker(ov,(round(p['cx_px']),round(p['cy_px'])),c[::-1],cv2.MARKER_CROSS,15,2,cv2.LINE_AA)
        result['overlay_img'] = ov
    if a.point_backend == 'grid_v46':
        diag = result.get('grid_diagnostics') or {}
        legend_classes = [s['class_name'] for s in diag.get('swatches', [])
                          if not s.get('excluded_by_class_filter') and s.get('template_available',True)]
        # Correct only actual legend types; do not invent the other classes.
        if legend_classes:
            kc = list(dict.fromkeys(legend_classes))
        elif _prev_state:
            kc = list(dict.fromkeys(p['class_name'] for p in _prev_state.get('P_current', [])))

    _binding = _state_binding(img, pa, _p4(a.legend_area), result.get('upscale', 1.0), a.point_backend)

    def _write_state(_P, _S, _what, _mode_xs=None, _runtime_state=None):
        """Persist the point state (SCALED coords) so the next Step-5 continues
        from exactly these points instead of re-detecting on its own."""
        try:
            from bw_series_identity import legend_series_catalog
            catalog=legend_series_catalog((result.get('grid_diagnostics') or {}).get('swatches',[]),
                                          result.get('legend_labels'))
            if not catalog and _prev_state:
                catalog=_prev_state.get('series_catalog',[])
            def _plain(_pts):
                out = []
                for _p in (_pts or []):
                    out.append({**_json_ready(_p),
                                "cx": float(_p.get("cx", _p.get("cx_px", 0.0))),
                                "cy": float(_p.get("cy", _p.get("cy_px", 0.0))),
                                "class_name": str(_p.get("class_name", "")),
                                "class_idx": int(_p.get("class_idx", 0)),
                                **identity_fields(_p)})
                return out
            _mx = _mode_xs
            if _mx is None and _prev_state is not None:
                _mx = _prev_state.get("mode_xs")
            try:
                _mx = [float(v) for v in (_mx if _mx is not None else [])]
            except Exception:
                _mx = []
            with open(os.path.join(a.out_dir, "correction_state.json"), "w",
                      encoding="utf-8") as _sf:
                json.dump({'format_version': BW_STATE_VERSION, 'binding': _binding,
                           'point_backend': a.point_backend, 'state_origin': _what,
                           "P_current": _plain(_P), "S_current": _plain(_S),
                           'series_catalog': catalog,
                           'marker_evidence': ((result.get('grid_diagnostics') or {}).get('correction_marker_evidence')
                               or (_prev_state or {}).get('marker_evidence',{})),
                           "mode_xs": _mx,
                           'bw_step5_state': _json_ready(_runtime_state)}, _sf, allow_nan=False)
            _safe_log(f"[bw-cli] saved correction_state.json from {_what} "
                      f"({len(_P or [])} active, {len(_S or [])} suppressed)")
        except Exception as _se:
            _safe_log(f"[bw-cli] state save failed: {_se}")
            if a.point_backend == 'grid_v46':
                raise

    # Save the DETECTION state too, so the first Step-5 press continues from the
    # points the user just saw rather than starting from a fresh detection.
    if not _prev_state:
        _write_state(result.get("kept_scaled"), result.get("suppressed_scaled"),
                     "detection", result.get("mode_xs"))
    else:
        # Every output job must remain resumable even if correction fails.
        _write_state(_prev_state.get('P_current'), _prev_state.get('S_current'),
                     'resumed state before correction', _prev_state.get('mode_xs'),
                     _prev_state.get('bw_step5_state'))

    # ── Optional Step-5 SSIM greedy correction: refine the point set ───────────
    _correction_error = None
    _correction_status = {'requested': bool(a.correct), 'status': 'not_requested',
                          'engine': 'bw_series_correction_v46' if a.point_backend == 'grid_v46' else '5_correction_v2'}
    if a.correct:
        _tmp = None
        try:
            up = float(result.get("upscale", 1.0) or 1.0)
            scaled = result.get("scaled_img_bgr")
            _safe_log("[bw-cli] running Step-5 correction (v46: structure-aware routing) ...")
            _t_corr0 = _tmod.perf_counter()
            # Continue from a previous Step-5 result when one is supplied, so
            # successive corrections compose instead of re-detecting from scratch.
            _init_P = (_prev_state.get("P_current") or []) if _prev_state else result.get('kept_scaled', [])
            _init_S = _prev_state.get("S_current") if _prev_state else result.get('suppressed_scaled', [])
            _mode_xs = _prev_state.get("mode_xs") if _prev_state else result.get('mode_xs')
            if _prev_state:
                _safe_log(f"[bw-cli] continuing from previous state: "
                          f"{len(_init_P or [])} active, {len(_init_S or [])} suppressed")
            if a.point_backend == 'grid_v46':
                import bw_series_correction_v46
                _scaled_plot = tuple(int(v * up) for v in pa)
                _legend = _p4(a.legend_area)
                _scaled_legend = tuple(int(v * up) for v in _legend) if _legend is not None else None
                r5 = bw_series_correction_v46.run_correction(
                    image_bgr=scaled if scaled is not None else img,
                    out_dir=os.path.join(a.out_dir, 'correction_diagnostics'),
                    init_points=_init_P, init_suppressed=_init_S,
                    plot_area=_scaled_plot, legend_box=_scaled_legend,
                    d_est=result.get('d_override'),
                    max_iter=int(a.correct_iters) if a.correct_iters.strip() else 5,
                    init_state=_prev_state.get('bw_step5_state') if _prev_state else None,
                    connection_alignment=True, structural_edits=True,
                    marker_evidence=((result.get('grid_diagnostics') or {}).get('correction_marker_evidence')
                        or (_prev_state or {}).get('marker_evidence')),
                    log_fn=_safe_log,
                )
                if not isinstance(r5.get('runtime_state'), dict):
                    raise ValueError('v46 Step-5 returned no resumable runtime state.')
            else:
                # Explicit legacy ViT remains on its original correction engine.
                import importlib.util, tempfile
                model_path = str(bw_pipeline.MODEL_PATH)
                detector_py = str(bw_pipeline.SRC_DIR / "1_point_detection_v3.py")
                _spec = importlib.util.spec_from_file_location(
                    "correction_v2", os.path.join(HERE, "5_correction_v2.py"))
                mod5 = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(mod5)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as _tf:
                    _tmp = _tf.name
                cv2.imwrite(_tmp, scaled if scaled is not None else img)
                r5 = mod5.run_correction(
                    img_path=_tmp, model_path=model_path, detector_py_path=detector_py,
                    out_dir=os.path.join(a.out_dir, 'correction_diagnostics'),
                    known_classes=kc, mode_xs=_mode_xs,
                    prep_info=result.get("prep_info"), return_diag_imgs=False,
                    max_iters=(int(a.correct_iters) if a.correct_iters.strip() else None),
                    d_override=result.get("d_override"),
                    init_points=_init_P, init_suppressed=_init_S,
                )
            if not isinstance(r5.get('P_current'), list) or not isinstance(r5.get('S_current'), list):
                raise ValueError('Step-5 returned invalid active/suppressed point arrays.')
            _safe_log(f"[bw-cli] [timing] correction total: "
                      f"{_tmod.perf_counter() - _t_corr0:.2f}s")
            # Persist the corrected state so the NEXT Step-5 can continue from it.
            _write_state(r5['P_current'], r5['S_current'], "correction",
                         r5.get('mode_xs', _mode_xs), r5.get('runtime_state'))
            result['kept_scaled'] = r5['P_current']
            result['suppressed_scaled'] = r5['S_current']
            inv = 1.0 / up if up else 1.0
            corrected = []
            for p in r5.get("P_current", []):
                if p.get("class_name") == "suppressed":
                    continue
                cx = float(p.get("cx", p.get("cx_px", 0))) * inv
                cy = float(p.get("cy", p.get("cy_px", 0))) * inv
                corrected.append({"class_name": p.get("class_name", "filled_circle"),
                                  **identity_fields(p),
                                  "cx_px": cx, "cy_px": cy})
            result["detections"] = corrected
            if corrected:
                _safe_log(f"[bw-cli] Step-5 correction: {len(corrected)} active points")
                # Redraw the overlay from the CORRECTED points so the overlay image
                # (plot 2) matches edit_data.json / the live reconstruction (plot 3).
                # Mirrors run_detection's Step-5 overlay style (same colours/sizes).
                try:
                    _MC = getattr(bw_pipeline, "MARKER_COLORS", {})
                    _ov2 = img.copy()
                    cv2.rectangle(_ov2, (pa[0], pa[1]), (pa[2], pa[3]), (0, 200, 0), 2)
                    _lg = _p4(a.legend_area)
                    if _lg is not None:
                        cv2.rectangle(_ov2, (_lg[0], _lg[1]), (_lg[2], _lg[3]), (200, 0, 200), 2)
                    _ref = max(_ov2.shape[:2])
                    _ro = max(4, int(_ref * 0.013)); _ri = max(2, int(_ro * 0.25))
                    _fs = max(0.30, _ref * 0.00065); _th = max(1, int(_ref * 0.002))
                    for _d in corrected:
                        _cx = int(round(float(_d['cx_px']))); _cy = int(round(float(_d['cy_px'])))
                        _c = series_color(_d, _MC); _cbgr = (_c[2], _c[1], _c[0])
                        cv2.circle(_ov2, (_cx, _cy), _ro, _cbgr, _th)
                        cv2.circle(_ov2, (_cx, _cy), _ri, _cbgr, -1)
                        _sh = _d['class_name'].replace('_marker', '').replace('_', ' ')
                        cv2.putText(_ov2, _sh, (_cx + _ro + 2, _cy - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, _fs, _cbgr, _th, cv2.LINE_AA)
                    result["overlay_img"] = _ov2
                    _safe_log("[bw-cli] overlay redrawn from corrected points (plot 2 = plot 3)")
                except Exception as _oe:
                    _safe_log("[bw-cli] overlay redraw failed: " + str(_oe))
            else:
                result['overlay_img'] = img.copy()
                _safe_log("[bw-cli] Step-5 produced no points; saved an empty point set and overlay")
            _correction_status.update(status='completed', active_count=len(corrected),
                                      suppressed_count=len(r5['S_current']))
        except Exception as _ce:
            import traceback
            _correction_error = str(_ce)
            _correction_status.update(status='failed', error=_correction_error)
            _safe_log("[bw-cli] Step-5 correction failed (keeping raw detections): "
                      + str(_ce))
            _safe_log(traceback.format_exc())
        finally:
            if _tmp is not None:
                try:
                    os.unlink(_tmp)
                except OSError:
                    pass
    with open(os.path.join(a.out_dir, 'correction_status.json'), 'w', encoding='utf-8') as _sf:
        json.dump(_correction_status, _sf, indent=2)

    _t_save0 = _tmod.perf_counter()
    ed = bw_to_edit_data.bw_to_edit_data(result, pa, xr, yr, a.x_log, a.y_log, (W, H))
    if result.get('grid_diagnostics') is not None:
        with open(os.path.join(a.out_dir, 'v46_grid_diagnostics.json'), 'w', encoding='utf-8') as f:
            json.dump(result['grid_diagnostics'], f, indent=2)
        for step in result.get('diag_steps', []):
            filename = step.get('output_filename')
            if filename in {'v46_legend_templates.png', 'v46_legend_diagnostic.png',
                            'v46_legend_overlay.png'}:
                cv2.imwrite(os.path.join(a.out_dir, filename), step['img_bgr'])
    with open(os.path.join(a.out_dir, "edit_data.json"), "w") as f:
        json.dump(ed, f, indent=2)
    ov = result.get("overlay_img")
    if ov is not None:
        cv2.imwrite(os.path.join(a.out_dir, "data_points_overlay.png"), ov)

    # Keep timing/provenance separate from edit_data and correction state so
    # performance instrumentation cannot mutate the user's detected points.
    _timing = {
        'pipeline_imports_seconds': _imports_seconds,
        'image_read_seconds': _image_read_seconds,
        'run_detection_seconds': _detection_seconds,
        'post_detection_seconds': _tmod.perf_counter()-_t_post0,
        'output_save_seconds': _tmod.perf_counter()-_t_save0,
        'cli_main_seconds': _tmod.perf_counter()-_t_cli0,
        'pipeline_timings': result.get('pipeline_timings', {}),
        'pipeline_timing_scope': result.get('pipeline_timing_scope'),
        'point_pipeline_timings': (result.get('grid_diagnostics') or {}).get('point_pipeline_timings', {}),
        'settings': {'point_backend': a.point_backend, 'gpu_batch_size': a.gpu_batch_size,
                     'diagnostic_detail_requested': a.diagnostic_detail, 'skip_segments': a.skip_segments,
                     'diagnostic_detail': result.get('diagnostic_detail', a.diagnostic_detail), 'image_shape': [H, W],
                     'plot_area': list(pa), 'legend_area': _p4(a.legend_area)},
        'timing_scope': 'perf_counter wall seconds; imports include backend initialization; post_detection includes output_save; cli_main excludes process launch/module-level imports and this timing file write',
    }
    with open(os.path.join(a.out_dir, 'pipeline_timing.json'), 'w', encoding='utf-8') as _tf:
        json.dump(_timing, _tf, indent=2)
    print(f"[bw-cli] [timing] image read: {_image_read_seconds:.2f}s; output save: {_timing['output_save_seconds']:.2f}s; CLI main: {_timing['cli_main_seconds']:.2f}s")

    n = sum(len(c["points"]) for c in ed["curves"])
    print(f"[bw-cli] done: {len(ed['curves'])} series, {n} points -> {a.out_dir}")
    if _correction_error is not None:
        # Keep raw/resumed artifacts available, but never report them as a
        # successful Step-5 result to the unified server or command-line caller.
        sys.exit(4)


if __name__ == "__main__":
    main()

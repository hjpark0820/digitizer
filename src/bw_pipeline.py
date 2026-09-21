"""GUI-free B&W runtime shared by unified_server.py and the v46 CLI.

The retired Tkinter front ends have been removed. Keep orchestration, pixel/data
conversion, native no-legend routing and legacy explicit ViT support here.
Legend-grid detection uses the automatic production profile in bw_pipeline_v46.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from bw_series_identity import series_key, series_color, identity_fields

# ── Resolve paths ─────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "chart_marker_net_v3.pth"

# ── Marker class definitions (from 1_point_detection_v3.py) ──────────────────
ALL_MARKERS = [
    ("filled_circle",       "●  Filled Circle"),
    ("open_circle",         "○  Open Circle"),
    ("filled_square",       "■  Filled Square"),
    ("open_square",         "□  Open Square"),
    ("open_triangle",       "△  Open Triangle (up)"),
    ("open_inv_triangle",   "▽  Open Triangle (down)"),
    ("filled_triangle",     "▲  Filled Triangle (up)"),
    ("filled_inv_triangle", "▼  Filled Triangle (down)"),
    ("open_rhombus",        "◇  Open Rhombus"),
    ("filled_rhombus",      "◆  Filled Rhombus"),
    ("x_marker",            "✕  X Marker"),
    ("plus_marker",         "+  Plus Marker"),
]

# Colour for each marker class overlay
MARKER_COLORS = {
    "filled_circle":       (220,  30,  30),   # vivid red
    "open_circle":         (230, 100,   0),   # deep orange
    "filled_square":       ( 20, 160,  20),   # vivid green
    "open_square":         (  0, 130, 200),   # sky blue
    "open_triangle":       ( 80,  30, 220),   # deep violet
    "open_inv_triangle":   (200,   0, 200),   # magenta
    "filled_triangle":     (  0, 190, 190),   # cyan
    "filled_inv_triangle": (180,  60, 180),   # purple
    "open_rhombus":        (220, 180,   0),   # gold/yellow
    "filled_rhombus":      (160,  80,   0),   # brown
    "x_marker":            ( 30, 180, 100),   # teal-green
    "plus_marker":         (255,  20, 120),   # hot pink
}

def grid_step5_detections(points, upscale, plot_area, x_range, y_range, x_log, y_log):
    """Convert typed runtime points once, keeping series identity even if empty."""
    result=[]
    for p in points:
        x=float(p['cx'])/upscale; y=float(p['cy'])/upscale
        xd,yd=px_to_data(x,y,plot_area,x_range,y_range,x_log,y_log)
        result.append(dict(class_name=p['class_name'],**identity_fields(p),
                           cx_px=x,cy_px=y,x_data=xd,y_data=yd,confidence=p.get('confidence',1.)))
    return sorted(result,key=lambda p:p['x_data'])


# ── Dynamic module loader ─────────────────────────────────────────────────────
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


# ── Coordinate conversion helpers ────────────────────────────────────────────
def px_to_data(px: float, py: float,
               plot_area_px: tuple,
               x_range: tuple, y_range: tuple,
               x_log: bool, y_log: bool) -> tuple[float, float]:
    """Convert pixel coordinates inside the plot area to data coordinates."""
    ax0, ay0, ax1, ay1 = plot_area_px
    # Normalise to [0,1]
    fx = (px - ax0) / max(ax1 - ax0, 1)
    fy = (py - ay0) / max(ay1 - ay0, 1)
    fy = 1.0 - fy   # y-axis is inverted in image coords

    x_min, x_max = x_range
    y_min, y_max = y_range

    if x_log:
        lx0 = math.log10(max(x_min, 1e-300))
        lx1 = math.log10(max(x_max, 1e-300))
        x_data = 10 ** (lx0 + fx * (lx1 - lx0))
    else:
        x_data = x_min + fx * (x_max - x_min)

    if y_log:
        ly0 = math.log10(max(y_min, 1e-300))
        ly1 = math.log10(max(y_max, 1e-300))
        y_data = 10 ** (ly0 + fy * (ly1 - ly0))
    else:
        y_data = y_min + fy * (y_max - y_min)

    return x_data, y_data


# ── Dual Y-axis boundary detection ──────────────────────────────────────────
def detect_right_yaxis(img_bgr: np.ndarray,
                       plot_area_px: tuple,
                       min_dark_frac: float = 0.25) -> int | None:
    """
    Detect the x-coordinate of a right-side Y-axis line inside the plot area.
    Scans the right half of the plot area for a column with high dark-pixel
    fraction (i.e. a continuous vertical black line).

    Returns the x-coordinate of the right Y-axis, or None if not found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ax0, ay0, ax1, ay1 = plot_area_px
    mid_x = (ax0 + ax1) // 2

    best_x, best_frac = None, 0.0
    for x in range(mid_x, min(ax1 + 30, gray.shape[1])):
        col = gray[ay0:ay1, x]
        dark_frac = float(np.sum(col < 50)) / max(len(col), 1)
        if dark_frac > best_frac:
            best_frac = dark_frac
            best_x = x

    # Only accept if clearly a solid vertical line (>25% dark)
    if best_frac >= min_dark_frac:
        return best_x
    return None


# ── Detection logic ───────────────────────────────────────────────────────────
def _optional_display_points(result):
    """Keep structural curve anchors visible without attaching glyph masks."""
    labels = {series['id']: series.get('label', 'Series 1') for series in result.get('series', [])}
    fallback = 'observed_marker' if result.get('chart_kind') == 'markers' else 'curve_anchor'
    return [dict(point, cx=float(point['x_px']), cy=float(point['y_px']),
                 class_name=point.get('class_name') or fallback, swatch_id=point['series_id'],
                 series_label=labels.get(point['series_id'], 'Series 1'))
            for point in result.get('points', [])]


def _run_no_legend_detection(img_bgr, plot_area_px, x_range, y_range, x_log, y_log,
                             *, diagnostic_detail=False, log_fn=print):
    """Adapt the shared no-legend detector to web/CLI outputs and BW Step-5 state."""
    from legend_optional_v46.runtime import box, bw_points, detect, overlay
    started = time.perf_counter()
    plot = box(plot_area_px, img_bgr.shape, inclusive=True)
    result = detect(img_bgr, plot, None, mode='bw')
    active, suppressed, diameter = bw_points(result)
    detections = grid_step5_detections(_optional_display_points(result), 1., plot_area_px,
        x_range, y_range, x_log, y_log)
    reports = []
    for index, series in enumerate(result.get('series', [])):
        sid = series['id']
        observed = next((p for p in active+suppressed if p.get('swatch_id') == sid), {})
        fallback = 'unknown_marker' if result.get('chart_kind') != 'line-only' else 'curve_anchor'
        name = observed.get('class_name', series.get('class_name') or fallback)
        reports.append(dict(class_name=name, swatch_id=sid, class_idx=index,
            shape_hint=observed.get('shape_hint', name), series_label=series.get('label', 'Series 1'),
            marker_state=series.get('marker_state', 'unknown'),
            source_kind='plot_observed_not_legend'))
    labels = {s['id']: s.get('label', 'Series 1') for s in result.get('series', [])}
    for detection in detections:
        detection['series_label'] = labels.get(series_key(detection), 'Series 1')
    visual = overlay(img_bgr, result)
    cv2.rectangle(visual, tuple(plot_area_px[:2]), tuple(plot_area_px[2:]), (0, 200, 0), 2)
    steps = [dict(title=f"No legend — {result.get('chart_kind', 'uncertain')} ({len(detections)} points)",
                  img_bgr=visual)]
    elapsed = time.perf_counter()-started
    diagnostics = dict(backend='v46_legend_optional', swatches=reports,
        no_legend=True, single_series_assumption=True, native_resolution=True,
        legacy_no_legend_disabled=True, status=result['status'],
        chart_kind=result.get('chart_kind', 'uncertain'), route=result.get('route'),
        reason=result.get('reason'), legend_labels=labels)
    log_fn('[v46 no legend] One series is assumed; source pixels are evaluated at 1x scale.')
    log_fn(f"[v46 no legend] {result['status']}: {len(detections)} active / {len(result.get('suppressed_points', []))} suppressed. "
           f"{result.get('reason', '')}")
    return dict(detections=detections, overlay_img=visual, segs=[], legend_labels=labels,
        diag_steps=steps, mode_xs=np.asarray(sorted({p['cx'] for p in active}), float),
        prep_info=None, scaled_img_bgr=img_bgr, upscale=1., d_override=diameter,
        kept_scaled=active, suppressed_scaled=suppressed, point_backend='grid_v46',
        diagnostic_detail=diagnostic_detail, pipeline_timings={'points': elapsed, 'total': elapsed},
        pipeline_timing_scope='Native no-legend source evidence and GUI result conversion',
        grid_diagnostics=diagnostics, legend_optional_result=result)


def run_detection(img_bgr: np.ndarray,
                  plot_area_px: tuple,
                  legend_area_px: tuple | None,
                  known_classes: list[str] | None,
                  x_range: tuple,
                  y_range: tuple,
                  x_log: bool,
                  y_log: bool,
                  has_errorbars: bool | None = None,
                  upscale: float = 1.0,
                  conf_thresh: float | None = None,
                  stride: int | None = None,
                  has_lines: bool = True,
                  skip_point_detection: bool = False,
                  log_fn=print,
                  point_backend: str = 'grid_v46',
                  grid_fraction: float = 0.5,
                  grid_overlap: float = 0.5,
                  swatches=None,
                  refinement_backend=None,
                  gpu_batch_size: int = 512,
                  grid_backend=None,
                  window_backend=None,
                  diagnostic_detail: bool | None = None) -> dict:
    """
    Run the chartocode2 pipeline restricted to the user-specified areas.
    B&W computation defaults to automatic CUDA availability selection, with
    the original CPU path when CUDA/PyTorch cannot be used. Explicit backend
    arguments or BW_V46_REFINEMENT_BACKEND override this automatic default.
    ``diagnostic_detail`` enables full-size segment/error-bar diagnostic panels.
    None defaults to False for grid_v46 and True for legacy ViT. Compact mode
    still retains input, legend-template and original-image final overlays.

    Returns
    -------
    dict with keys:
        'detections'  : list of {class_name, cx_px, cy_px, x_data, y_data}
        'overlay_img' : BGR image with detections drawn
    """
    _pipeline_started = time.perf_counter()
    pipeline_timings = {name: 0. for name in (
        'setup', 'preprocessing', 'segments', 'errorbar', 'points',
        'filter_coordinates', 'overlay', 'legend_ocr', 'diagnostics')}
    def _record_timing(name, started, subtract=0.):
        pipeline_timings[name] += max(0., time.perf_counter()-started-subtract)
    H, W = img_bgr.shape[:2]
    if point_backend not in ('grid_v46', 'vit'):
        raise ValueError(f'Unknown point backend: {point_backend}')
    if diagnostic_detail is None:
        diagnostic_detail = point_backend != 'grid_v46'
    if not isinstance(diagnostic_detail, bool):
        raise ValueError('diagnostic_detail must be bool or None')
    if point_backend == 'grid_v46' and not skip_point_detection:
        if legend_area_px is None and swatches is None:
            return _run_no_legend_detection(img_bgr, plot_area_px, x_range, y_range, x_log, y_log,
                diagnostic_detail=diagnostic_detail, log_fn=log_fn)
        # Older clients' all-twelve checkbox state means no shape restriction;
        # it must not silently discard an observed but unnamed marker.
        if known_classes and set(known_classes)=={k for k,_ in ALL_MARKERS}:
            known_classes=None
    # ViT's auto-resize target is unrelated to a pixel-template detector.
    if point_backend == 'grid_v46' and upscale is None:
        upscale = 1.0
    if point_backend == 'grid_v46' and (not math.isfinite(float(upscale)) or float(upscale)<=0):
        raise ValueError('v46 manual scale must be positive and finite')
    ax0, ay0, ax1, ay1 = plot_area_px
    _orig_img_bgr      = img_bgr          # keep original for overlay
    _orig_plot_area_px = plot_area_px     # keep original for overlay
    _orig_legend_px    = legend_area_px   # keep original for overlay

    # ── [point-diag] echo the user-specified areas so they can be verified ────
    log_fn(f"[input] image {img_bgr.shape[1]}x{img_bgr.shape[0]}  "
           f"plot_area(x0,y0,x1,y1)={tuple(int(v) for v in plot_area_px)}  "
           f"legend_area={tuple(int(v) for v in legend_area_px) if legend_area_px else 'None'}  "
           f"has_errorbars={has_errorbars}")

    # ── Upscale: auto-detect from legend if upscale=None ──────────────────────
    if upscale is None and legend_area_px is not None:
        try:
            from chart_preprocessing import estimate_optimal_scale as _eos
            _scale_result = _eos(img_bgr, legend_box=legend_area_px)
            # estimate_optimal_scale returns (scale, info_dict) or just scale
            if isinstance(_scale_result, tuple):
                upscale = float(_scale_result[0])
            else:
                upscale = float(_scale_result)
            log_fn(f"[Step 0a] Auto upscale from legend: {upscale}x")
        except Exception as _e:
            log_fn(f"[Step 0a] Auto upscale failed ({_e}); using 1.0")
            upscale = 1.0
    # Support both upscale (>1) and downscale (<1); 1.0 = no resize.
    # Formula: diameter * 2 * scale = 19px  →  scale = 9.5 / diameter
    _upscale = float(upscale) if upscale is not None else 1.0
    if _upscale == 0.0:
        _upscale = 1.0

    if _upscale != 1.0:
        new_w = int(round(W * _upscale))
        new_h = int(round(H * _upscale))
        interp = cv2.INTER_CUBIC if _upscale > 1.0 else cv2.INTER_AREA
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
        plot_area_px   = tuple(int(v * _upscale) for v in plot_area_px)
        if legend_area_px is not None:
            legend_area_px = tuple(int(v * _upscale) for v in legend_area_px)
        ax0, ay0, ax1, ay1 = plot_area_px
        H, W = img_bgr.shape[:2]
        log_fn(f"[Step 0a] Scaled x{_upscale}: {W}x{H}  plot_area={plot_area_px}")

    # ── Build preprocessing info from user-supplied areas ─────────────────
    _record_timing('setup', _pipeline_started)
    _preprocessing_started = time.perf_counter()
    log_fn("[Step 0] Building preprocessing info from user areas …")
    try:
        from chart_preprocessing import preprocess as _cp
        # preprocess() will expand plot_area_px by AXIS_MARGIN for noise removal
        # but keeps user_plot_area for coordinate conversion.
        prep_info = _cp(img_bgr,
                         user_plot_area=plot_area_px,
                         user_legend_box=legend_area_px,
                         verbose=False)
        log_fn(f"  expanded plot_area = {prep_info['plot_area']}")
        log_fn(f"  user plot_area     = {prep_info.get('user_plot_area', plot_area_px)}")
        log_fn(f"  legend_box = {prep_info['legend_box']}")
        log_fn(f"  lloq_row   = {prep_info['lloq_row']}")
    except ImportError:
        log_fn("  chart_preprocessing not found; skipping noise removal.")
        prep_info = None
    _record_timing('preprocessing', _preprocessing_started)

    # ── Segment detection + error-bar removal (lines mode only) ─────────────
    segs = []
    _img_for_vit = img_bgr
    _prep_info_for_vit = prep_info
    _eb_info_list = []
    _d_override_used = None   # d_est passed to NMS; shared with Step-5 correction for consistency
    _diag_steps: list[dict] = []  # list of {title, img_bgr}

    # ── [input-diag] Visual confirmation of the areas the USER specified in the GUI,
    #    overlaid on the ORIGINAL image with a PIXEL GRID so coordinates can be read off.
    #    Green = plot area, Red = legend area. Uses original (unscaled) coordinates.
    _input_diag_started = time.perf_counter()
    if os.environ.get("POINT_DIAG", "1") != "0":
        try:
            _in_diag = _orig_img_bgr.copy()
            _Hd, _Wd = _in_diag.shape[:2]
            # subtle pixel grid every 50 px (draw on overlay, then blend)
            _grid = _in_diag.copy()
            for _gx in range(0, _Wd, 50):
                cv2.line(_grid, (_gx, 0), (_gx, _Hd), (235, 190, 100), 1)
            for _gy in range(0, _Hd, 50):
                cv2.line(_grid, (0, _gy), (_Wd, _gy), (235, 190, 100), 1)
            cv2.addWeighted(_grid, 0.35, _in_diag, 0.65, 0, dst=_in_diag)
            # coordinate labels every 100 px (top edge = x, left edge = y)
            for _gx in range(0, _Wd, 100):
                cv2.putText(_in_diag, str(_gx), (_gx + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (200, 90, 0), 1, cv2.LINE_AA)
            for _gy in range(0, _Hd, 100):
                cv2.putText(_in_diag, str(_gy), (2, _gy + 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (200, 90, 0), 1, cv2.LINE_AA)
            # user-specified areas
            _pa = tuple(int(v) for v in _orig_plot_area_px)
            cv2.rectangle(_in_diag, (_pa[0], _pa[1]), (_pa[2], _pa[3]), (0, 200, 0), 2)
            cv2.putText(_in_diag, f"plot area {_pa}", (_pa[0], max(14, _pa[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1, cv2.LINE_AA)
            if _orig_legend_px is not None:
                _lg = tuple(int(v) for v in _orig_legend_px)
                cv2.rectangle(_in_diag, (_lg[0], _lg[1]), (_lg[2], _lg[3]), (0, 0, 230), 2)
                cv2.putText(_in_diag, f"legend {_lg}", (_lg[0], max(14, _lg[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
            _diag_steps.append({'title': 'Input — plot area (green) + legend (red) + pixel grid',
                                'img_bgr': _in_diag})
        except Exception as _ind_e:
            log_fn(f"  [input-diag] area+grid overlay failed: {_ind_e}")
    _record_timing('diagnostics', _input_diag_started)
    if point_backend == 'grid_v46' and not diagnostic_detail:
        log_fn('[v46 diagnostics] Compact: input, legend and final overlay retained; '
               'enable diagnostic_detail for segment/error-bar panels.')

    if not has_lines:
        log_fn("[Step 1] No-lines mode: segment detection skipped.")
        log_fn("[Step 1b] No-lines mode: error-bar removal skipped.")
    else:
        # ── Error-bar gate (Function 1 auto-detect or GUI override) ──────────
        # has_errorbars=True  → always run stem removal (Function 2)
        # has_errorbars=False → always skip
        # has_errorbars=None  → auto-detect via detect_has_errorbars() (Function 1)
        if has_errorbars is None and point_backend != 'grid_v46':
            _errorbar_gate_started = time.perf_counter()
            try:
                from chart_preprocessing import detect_has_errorbars as _deb
                has_errorbars = _deb(img_bgr, prep_info=prep_info)
                log_fn(f"  [Function 1] detect_has_errorbars → {has_errorbars}")
            except Exception:
                has_errorbars = False
                log_fn("  [Function 1] auto-detect failed; assuming no error bars.")
            _record_timing('errorbar', _errorbar_gate_started)

        # ── Stage 2: segment detection ────────────────────────────────────────
        _segments_started = time.perf_counter()
        _segment_diag_before = pipeline_timings['diagnostics']
        log_fn("[Step 1] Segment detection …")
        seg_v2   = SRC_DIR / "3_segment_detection_v2.py"
        seg_orig = SRC_DIR / "3_segment_detection.py"
        seg_path = seg_v2 if seg_v2.exists() else seg_orig
        log_fn(f"  Loading: {seg_path.name}")
        mod3 = _load("segment_detector", seg_path)
        import inspect as _insp
        _seg_sig = _insp.signature(mod3.detect)
        _seg_debug_result = None
        if hasattr(mod3, 'detect_debug'):
            try:
                _seg_debug_result = mod3.detect_debug(img_bgr, prep_info=prep_info)
                segs = _seg_debug_result['segments']
                if diagnostic_detail:
                    _segment_diag_started = time.perf_counter()
                    try:
                        # Build 8-panel diagnostic image in-memory
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as _plt_seg
                        import io as _io_seg
                        _img_rgb_d = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        _fg = _seg_debug_result['fg_mask']
                        _sm = _seg_debug_result['seg_mask']
                        _clusters = _seg_debug_result['clusters']
                        _lost = (_fg > 0) & (_sm == 0)
                        _panel_lost = _img_rgb_d.copy() // 2
                        _panel_lost[_sm > 0] = [0, 220, 80]
                        _panel_lost[_lost]   = [255, 60, 60]
                        def _seg_ov(base, segs_list, color=(220,40,40)):
                            out = base.copy()
                            for (x1,y1,x2,y2) in segs_list:
                                cv2.line(out,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
                                cv2.circle(out,(int(x1),int(y1)),3,color,-1)
                                cv2.circle(out,(int(x2),int(y2)),3,color,-1)
                            return out
                        _panels_seg = [
                            (_img_rgb_d,                                              f"1. Input"),
                            (np.stack([_fg*255]*3,-1).astype(np.uint8),              f"2. Foreground ({_fg.sum()} px)"),
                            (np.stack([_sm*255]*3,-1).astype(np.uint8),              f"3. Segment mask ({_sm.sum()} px)"),
                            (_panel_lost,                                              f"3b. Kept(green)/Lost(red)"),
                            (_seg_ov(_img_rgb_d, _seg_debug_result['segments_raw']),  f"4. Raw ({len(_seg_debug_result['segments_raw'])})"),
                            (_seg_ov(_img_rgb_d, _seg_debug_result['segments_grouped']), f"5. Grouped ({len(_seg_debug_result['segments_grouped'])})"),
                            (_seg_ov(_img_rgb_d, _seg_debug_result['segments_extended']), f"6. Extended ({len(_seg_debug_result['segments_extended'])})"),
                            (_seg_ov(_img_rgb_d, segs),                               f"7. Final ({len(segs)})"),
                        ]
                        _fig_seg, _axes_seg = _plt_seg.subplots(2, 4, figsize=(22, 12))
                        for _ax_s, (_pan, _tit) in zip(_axes_seg.flat, _panels_seg):
                            _ax_s.imshow(_pan); _ax_s.set_title(_tit, fontsize=9); _ax_s.axis('off')
                        _plt_seg.suptitle(f"Segment Detection Pipeline  →  {len(segs)} final segments", fontsize=12, fontweight='bold')
                        _plt_seg.tight_layout()
                        _buf_seg = _io_seg.BytesIO()
                        _plt_seg.savefig(_buf_seg, dpi=120, bbox_inches='tight', format='png')
                        _plt_seg.close()
                        _buf_seg.seek(0)
                        _seg_diag_arr = cv2.imdecode(np.frombuffer(_buf_seg.read(), np.uint8), cv2.IMREAD_COLOR)
                        _diag_steps.append({'title': f'Step 1 — Segment Detection ({len(segs)} segments)', 'img_bgr': _seg_diag_arr})
                    except Exception as _segment_render_error:
                        log_fn(f'  [diag] segment rendering failed: {_segment_render_error}')
                    finally:
                        _record_timing('diagnostics', _segment_diag_started)
            except Exception as _seg_dbg_e:
                log_fn(f"  [diag] detect_debug failed: {_seg_dbg_e}")
                if 'prep_info' in _seg_sig.parameters:
                    segs = mod3.detect(img_bgr, prep_info=prep_info)
                else:
                    segs = mod3.detect(img_bgr)
        else:
            if 'prep_info' in _seg_sig.parameters:
                segs = mod3.detect(img_bgr, prep_info=prep_info)
            else:
                segs = mod3.detect(img_bgr)
        log_fn(f"  {len(segs)} segments detected")
        _record_timing('segments', _segments_started,
                       pipeline_timings['diagnostics']-_segment_diag_before)

        # grid_v46 consumes the same segments for the error-bar gate instead
        # of running the entire CPU segment detector a second time.
        if has_errorbars is None and point_backend == 'grid_v46':
            _errorbar_gate_started = time.perf_counter()
            try:
                from chart_preprocessing import detect_has_errorbars as _deb
                has_errorbars = _deb(img_bgr, prep_info=prep_info, segments=segs)
                log_fn(f"  [Function 1] detect_has_errorbars (reused segments) → {has_errorbars}")
            except Exception:
                has_errorbars = False
                log_fn("  [Function 1] auto-detect failed; assuming no error bars.")
            _record_timing('errorbar', _errorbar_gate_started)

        # ── Function 2: Error-bar stem + T-cap removal (if has_errorbars) ────
        _errorbar_started = time.perf_counter()
        _errorbar_diag_before = pipeline_timings['diagnostics']
        if has_errorbars and prep_info is not None and (point_backend != 'grid_v46' or diagnostic_detail):
            log_fn("[Step 1b] Error-bar stem + T-cap removal (Function 2) …")
            try:
                from chart_preprocessing import remove_errorbars_from_mask as _rem_eb
                import math as _math_eb

                def _seg_to_dict_eb(s):
                    x1, y1, x2, y2 = s
                    dx = abs(x2 - x1); dy = abs(y2 - y1)
                    angle = _math_eb.degrees(_math_eb.atan2(dy, dx + 1e-9))
                    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'angle': angle}

                segs_dicts = [_seg_to_dict_eb(s) for s in segs]
                vert_segs  = [s for s in segs_dicts if abs(s['angle'] - 90) <= 20]
                if vert_segs:
                    _gray_eb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, _bw_eb = cv2.threshold(_gray_eb, 128, 255, cv2.THRESH_BINARY_INV)
                    _bw_eb = (_bw_eb > 0).astype('uint8')
                    _mask_eb = prep_info['clean_fn'](_bw_eb)
                    _mask_no_stem, _eb_info_list = _rem_eb(_mask_eb, vert_segs, segs)
                    log_fn(f"  Removed {len(vert_segs)} stem(s); "
                           f"{len(_eb_info_list)} stem(s) processed.")
                    # Build stem-erased BGR image for ViT
                    _img_no_stem = img_bgr.copy()
                    _stem_px = (_mask_eb.astype('uint8') - _mask_no_stem.astype('uint8')).clip(0, 1)
                    _img_no_stem[_stem_px == 1] = 255
                    _img_for_vit = _img_no_stem
                    if diagnostic_detail:
                        # Build error-bar before/after diagnostic image
                        _errorbar_diag_started = time.perf_counter()
                        try:
                            import io as _io_eb2
                            import matplotlib.pyplot as _plt_eb2
                            _eb_before_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            _eb_after_rgb  = cv2.cvtColor(_img_no_stem, cv2.COLOR_BGR2RGB)
                            # Overlay stem positions on before image
                            _eb_before_ann = _eb_before_rgb.copy()
                            for _ebi in _eb_info_list:
                                _ecx = int(_ebi.get('cx', 0))
                                _ey0 = int(_ebi.get('y_top', 0))
                                _ey1 = int(_ebi.get('y_bot', 0))
                                cv2.line(_eb_before_ann, (_ecx, _ey0), (_ecx, _ey1), (255, 0, 0), 2)
                                cv2.circle(_eb_before_ann, (_ecx, _ey0), 4, (0, 255, 0) if _ebi.get('top_is_marker') else (255, 165, 0), -1)
                                cv2.circle(_eb_before_ann, (_ecx, _ey1), 4, (0, 255, 0) if _ebi.get('bot_is_marker') else (255, 165, 0), -1)
                            _fig_eb, _axes_eb = _plt_eb2.subplots(1, 3, figsize=(18, 6))
                            _axes_eb[0].imshow(_eb_before_rgb);     _axes_eb[0].set_title('Before removal', fontsize=10); _axes_eb[0].axis('off')
                            _axes_eb[1].imshow(_eb_before_ann);     _axes_eb[1].set_title(f'Stems annotated ({len(_eb_info_list)} stems)\nBlue=stem, Green=marker end, Orange=T-cap end', fontsize=9); _axes_eb[1].axis('off')
                            _axes_eb[2].imshow(_eb_after_rgb);      _axes_eb[2].set_title('After removal (ViT input)', fontsize=10); _axes_eb[2].axis('off')
                            _plt_eb2.suptitle(f'Step 1b — Error-bar Removal  ({len(_eb_info_list)} stems processed)', fontsize=12, fontweight='bold')
                            _plt_eb2.tight_layout()
                            _buf_eb = _io_eb2.BytesIO()
                            _plt_eb2.savefig(_buf_eb, dpi=120, bbox_inches='tight', format='png')
                            _plt_eb2.close()
                            _buf_eb.seek(0)
                            _eb_diag_arr = cv2.imdecode(np.frombuffer(_buf_eb.read(), np.uint8), cv2.IMREAD_COLOR)
                            _diag_steps.append({'title': f'Step 1b — Error-bar Removal ({len(_eb_info_list)} stems)', 'img_bgr': _eb_diag_arr})
                        except Exception as _eb_diag_e:
                            log_fn(f"  [diag] error-bar diag failed: {_eb_diag_e}")
                        finally:
                            _record_timing('diagnostics', _errorbar_diag_started)
                    # Wrap stem-free mask into prep_info for ViT
                    _orig_clean_fn = prep_info['clean_fn']
                    def _make_sfn(sfm, ofn):
                        def _sfn(bw): return np.minimum(ofn(bw), sfm)
                        return _sfn
                    _prep_info_for_vit = dict(prep_info)
                    _prep_info_for_vit['clean_fn'] = _make_sfn(_mask_no_stem, _orig_clean_fn)
                else:
                    log_fn("  No vertical stems found; skipping removal.")
            except Exception as _e_eb:
                import traceback as _tb_eb
                log_fn(f"  Error-bar removal failed: {_e_eb}")
                log_fn(_tb_eb.format_exc())
        elif not has_errorbars:
            log_fn("[Step 1b] Error-bar removal skipped (has_errorbars=False).")
        elif point_backend == 'grid_v46' and not diagnostic_detail:
            log_fn('[Step 1b] v46 preserves original marker/error-bar ink; '
                   'ViT-only stem-erased copies skipped. Segments remain available.')
        _record_timing('errorbar', _errorbar_started,
                       pipeline_timings['diagnostics']-_errorbar_diag_before)

    # ── Marker detection: v46 grid by default; explicit legacy ViT option ──
    _points_started = time.perf_counter()
    kept = []
    if skip_point_detection:
        # Step-5 continuing from a saved state: the ViT result would be thrown
        # away anyway, so skip the expensive scan (this is the dominant cost).
        log_fn("[Step 2] Point detection SKIPPED (continuing from saved state).")
        if point_backend == 'grid_v46' and (legend_area_px is not None or swatches is not None):
            from bw_pipeline_v46 import extract_templates, VERSION
            scaled_swatches = None if swatches is None else [
                (name,tuple(int(round(v*_upscale)) for v in box)) for name,box in swatches]
            templates, swatch_report = extract_templates(img_bgr,legend_area_px,scaled_swatches,known_classes)
            from bw_legend_diagnostics import legend_diagnostic_steps
            _diag_steps.extend(legend_diagnostic_steps(img_bgr, legend_area_px, templates, swatch_report))
            result2 = {'suppressed':[], 'mode_xs':np.array([]),
                       'diagnostics':{'backend':VERSION,'swatches':swatch_report,
                                      'candidates':[],'resumed_from_state':True}}
            _d_override_used = float(np.median([t.diameter for t in templates]))
    elif point_backend == 'grid_v46':
        from bw_pipeline_v46 import detect_points, production_detection_options, PRODUCTION_PROFILE
        scaled_swatches = None if swatches is None else [
            (name, tuple(int(round(v*_upscale)) for v in box)) for name,box in swatches]
        # Deliberately read original/scaled ink, not the stem-erased ViT image.
        log_fn(f'[v46 matching profile] {PRODUCTION_PROFILE} (automatic)')
        result2 = detect_points(img_bgr, plot_area_px, legend_area_px,
                                known_classes=known_classes, swatches=scaled_swatches,
                                grid_fraction=grid_fraction, grid_overlap=grid_overlap,
                                min_required_recall=conf_thresh,
                                refinement_backend=refinement_backend,
                                gpu_batch_size=gpu_batch_size,
                                grid_backend=grid_backend, window_backend=window_backend,
                                log_fn=log_fn, **production_detection_options())
        result2['diagnostics']['matching_profile'] = PRODUCTION_PROFILE
        kept = result2['kept']
        _d_override_used = result2['d_est']
        _diag_steps.extend(result2['diag_steps'])
    elif MODEL_PATH.exists() and known_classes:
        log_fn("[Step 2] ViT point detection (adaptive NMS) …")
        try:
            # Prefer _v2 version (supports prep_info); fall back to original
            nms_v2   = SRC_DIR / "2_point_detection_adaptive_nms_v2.py"
            nms_orig = SRC_DIR / "2_point_detection_adaptive_nms.py"
            nms_path = nms_v2 if nms_v2.exists() else nms_orig
            log_fn(f"  Loading: {nms_path.name}")
            mod2 = _load("adaptive_nms", nms_path)

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = tf.name
            cv2.imwrite(tmp_path, _img_for_vit)

            import inspect
            sig = inspect.signature(mod2.detect_with_adaptive_nms)
            supports_prep = 'prep_info' in sig.parameters

            call_kwargs = dict(
                img_path         = tmp_path,
                model_path       = str(MODEL_PATH),
                known_classes    = known_classes,
                detector_py_path = str(SRC_DIR / "1_point_detection_v3.py"),
            )
            if conf_thresh is not None:
                call_kwargs['conf_thresh'] = conf_thresh
            if stride is not None:
                call_kwargs['stride'] = stride
            # When upscaled, KDE-based d_est can be unstable (more raw detections
            # cause finer mode splitting). Instead, estimate d on the ORIGINAL image
            # and scale it by _upscale to get the correct NMS bin width.
            import math as _math
            _sig_nms = inspect.signature(mod2.detect_with_adaptive_nms)
            if _upscale > 1.0 and 'd_override' in _sig_nms.parameters:
                try:
                    import tempfile as _tf2, os as _os2
                    with _tf2.NamedTemporaryFile(suffix='.png', delete=False) as _tf2f:
                        _tmp2 = _tf2f.name
                    cv2.imwrite(_tmp2, _orig_img_bgr)
                    _r_orig = mod2.detect_with_adaptive_nms(
                        img_path         = _tmp2,
                        model_path       = str(MODEL_PATH),
                        known_classes    = known_classes,
                        detector_py_path = str(SRC_DIR / '1_point_detection_v3.py'),
                        plot_area        = _orig_plot_area_px,
                    )
                    _os2.unlink(_tmp2)
                    _d_orig = _r_orig.get('d_est', None)
                    if _d_orig is not None:
                        _d_scaled = _d_orig * _upscale
                        call_kwargs['d_override'] = _d_scaled
                        _d_override_used = _d_scaled
                        log_fn(f"  d_override={_d_scaled:.1f}px (orig d_est={_d_orig:.1f} x {_upscale})")
                except Exception as _de:
                    log_fn(f"  d_override estimation failed: {_de}")
            if supports_prep:
                # Compute tight scan boundary:
                # - Left  boundary: axis_col (left Y-axis x) from prep_info
                # - Right boundary: right Y-axis x (detected) or user plot_area x1
                # This prevents scanning axis tick/label areas on either side.
                _vit_prep = dict(_prep_info_for_vit)
                _scan_x0, _scan_y0, _scan_x1, _scan_y1 = plot_area_px

                # Y-axis boundary detection with margin
                # Margin keeps ViT windows clear of the axis lines themselves.
                _AXIS_MARGIN_PX = 10

                # Left Y-axis: use axis_col from prep_info
                _axis_col = prep_info.get('axis_col', None) if prep_info else None
                if _axis_col is not None:
                    _cand_x0 = _axis_col + _AXIS_MARGIN_PX
                    if _cand_x0 > _scan_x0:
                        _scan_x0 = _cand_x0
                        log_fn(f"  Left Y-axis at x={_axis_col}; scan x0 \u2192 {_scan_x0} (+{_AXIS_MARGIN_PX}px margin)")

                # Right Y-axis: detect vertical line in right half of plot area
                _right_yaxis = detect_right_yaxis(img_bgr, plot_area_px)
                if _right_yaxis is not None:
                    _cand_x1 = _right_yaxis - _AXIS_MARGIN_PX
                    if _cand_x1 < _scan_x1:
                        _scan_x1 = _cand_x1
                        log_fn(f"  Right Y-axis at x={_right_yaxis}; scan x1 \u2192 {_scan_x1} (-{_AXIS_MARGIN_PX}px margin)")

                _vit_prep['plot_area'] = (_scan_x0, _scan_y0, _scan_x1, _scan_y1)
                call_kwargs['prep_info'] = _vit_prep
                log_fn(f"  ViT scan range: x=[{_scan_x0},{_scan_x1}] y=[{_scan_y0},{_scan_y1}]")
            else:
                log_fn("  (prep_info not supported by this version – skipping noise filter)")

            result2 = mod2.detect_with_adaptive_nms(**call_kwargs)
            os.unlink(tmp_path)
            kept = result2["kept"]
            log_fn(f"  {len(kept)} markers detected by ViT")
            # ── [point-diag] Why were points dropped? (NMS is X-only) ──────────
            if os.environ.get("POINT_DIAG", "1") != "0":
                try:
                    _sup  = result2.get("suppressed", [])
                    _binw = result2.get("bin_width", 0) or 0
                    # per-class kept vs suppressed summary (spot under-detected classes)
                    from collections import Counter as _Cnt
                    _kc = _Cnt(_kd.get("class_name", "?") for _kd in kept)
                    _sc = _Cnt(_sd.get("original_class_name", _sd.get("class_name", "?")) for _sd in _sup)
                    for _cn in sorted(set(_kc) | set(_sc)):
                        _flag = "  <-- many suppressed" if _sc.get(_cn, 0) > _kc.get(_cn, 0) else ""
                        log_fn(f"  [point-diag] class '{_cn}': kept={_kc.get(_cn,0)}, "
                               f"suppressed={_sc.get(_cn,0)}{_flag}")
                    log_fn(f"  [point-diag] raw ViT detections={len(kept)+len(_sup)} "
                           f"(kept={len(kept)}, NMS-suppressed={len(_sup)}); "
                           f"conf_thresh={conf_thresh if conf_thresh is not None else 'default'}, "
                           f"nms_window={_binw:.0f}px")
                    _stacked = 0
                    for _sd in _sup:
                        _scx, _scy = _sd.get("cx", -1), _sd.get("cy", -1)
                        _scls = _sd.get("original_class_name", _sd.get("class_name", "?"))
                        _sconf = _sd.get("confidence", 0.0)
                        _cands = [(abs(_kd.get("cx", 1e9) - _scx), _kd) for _kd in kept
                                  if _kd.get("class_name") == _scls]
                        if _cands:
                            _dx, _anc = min(_cands, key=lambda z: z[0])
                            _dy = abs(_anc.get("cy", 0) - _scy)
                            _same_x = _dx < max(3.0, _binw * 0.6) and _dy > 6
                            _stacked += 1 if _same_x else 0
                            log_fn(f"    [suppressed] {_scls} @({_scx:.0f},{_scy:.0f}) "
                                   f"conf={_sconf:.2f} -> merged into kept "
                                   f"({_anc.get('cx',0):.0f},{_anc.get('cy',0):.0f}) "
                                   f"dx={_dx:.0f} dy={_dy:.0f}"
                                   f"{'  [SAME-X STACKED]' if _same_x else ''}")
                        else:
                            log_fn(f"    [suppressed] {_scls} @({_scx:.0f},{_scy:.0f}) "
                                   f"conf={_sconf:.2f} -> no same-class kept nearby")
                    if _stacked:
                        log_fn(f"  [point-diag] {_stacked} suppressed are SAME-X stacked: "
                               f"NMS keeps one marker per class per x-window, so vertically-"
                               f"stacked markers at ~same x are merged (not recoverable by ViT+NMS).")
                except Exception as _pde:
                    log_fn(f"  [point-diag] suppressed-diag failed: {_pde}")
            # Build ViT/NMS diagnostic image
            try:
                import io as _io_vit
                import matplotlib.pyplot as _plt_vit
                _suppressed = result2.get('suppressed', [])
                _d_est_v    = result2.get('d_est', None)
                _bin_w      = result2.get('bin_width', None)
                _vit_img_rgb = cv2.cvtColor(_img_for_vit, cv2.COLOR_BGR2RGB)
                _vit_ann = _vit_img_rgb.copy()
                for _sd in _suppressed:
                    cv2.circle(_vit_ann, (int(round(_sd['cx'])), int(round(_sd['cy']))), 5, (0, 80, 255), -1)
                for _kd in kept:
                    _kcx, _kcy = int(round(_kd['cx'])), int(round(_kd['cy']))
                    cv2.circle(_vit_ann, (_kcx, _kcy), 7, (255, 60, 60), 2)
                    cv2.circle(_vit_ann, (_kcx, _kcy), 2, (255, 60, 60), -1)
                _fig_vit, _axes_vit = _plt_vit.subplots(1, 2, figsize=(16, 7))
                _axes_vit[0].imshow(_vit_img_rgb); _axes_vit[0].set_title('ViT input image', fontsize=10); _axes_vit[0].axis('off')
                _axes_vit[1].imshow(_vit_ann)
                _axes_vit[1].set_title(
                    f'NMS result  kept={len(kept)}, suppressed={len(_suppressed)}\n'
                    f'd_est={_d_est_v:.1f}px  bin_width={_bin_w:.1f}px' if _d_est_v else
                    f'NMS result  kept={len(kept)}, suppressed={len(_suppressed)}',
                    fontsize=9)
                _axes_vit[1].axis('off')
                if _bin_w:
                    _W_vit = _img_for_vit.shape[1]
                    for _bv in range(int(np.ceil(_W_vit / _bin_w)) + 1):
                        _axes_vit[1].axvline(_bv * _bin_w, color='lime', lw=0.8, ls='--', alpha=0.5)
                _plt_vit.suptitle(f'Step 2 — ViT Detection + Adaptive NMS', fontsize=12, fontweight='bold')
                _plt_vit.tight_layout()
                _buf_vit = _io_vit.BytesIO()
                _plt_vit.savefig(_buf_vit, dpi=120, bbox_inches='tight', format='png')
                _plt_vit.close()
                _buf_vit.seek(0)
                _vit_diag_arr = cv2.imdecode(np.frombuffer(_buf_vit.read(), np.uint8), cv2.IMREAD_COLOR)
                _diag_steps.append({'title': f'Step 2 — ViT Detection (kept={len(kept)}, suppressed={len(_suppressed)})', 'img_bgr': _vit_diag_arr})
            except Exception as _vit_diag_e:
                log_fn(f"  [diag] ViT diag failed: {_vit_diag_e}")
        except Exception as e:
            import traceback
            log_fn(f"  ViT detection failed: {e}")
            log_fn(traceback.format_exc())
            kept = []
    else:
        if not MODEL_PATH.exists():
            log_fn("[Step 2] Model not found – ViT detection skipped.")
            log_fn(f"  (expected: {MODEL_PATH})")
        else:
            log_fn("[Step 2] No marker classes selected – ViT detection skipped.")

    _record_timing('points', _points_started)
    _filter_started = time.perf_counter()
    # ── Filter detections to plot area (excluding legend area) ─────────────
    log_fn("[Step 3] Filtering detections to plot area …")
    def _in_plot_not_legend(d):
        cx = d.get('cx', d.get('cx_px', -1))
        cy = d.get('cy', d.get('cy_px', -1))
        # Must be inside plot area
        if not (ax0 <= cx <= ax1 and ay0 <= cy <= ay1):
            return False
        # Must NOT be inside legend area (if specified)
        if legend_area_px is not None:
            lx0, ly0, lx1, ly1 = legend_area_px
            if lx0 <= cx <= lx1 and ly0 <= cy <= ly1:
                return False
        return True
    kept_in, _filtered_out = [], []
    for d in kept:
        (kept_in if _in_plot_not_legend(d) else _filtered_out).append(d)
    # Scaled-space detection state, handed to Step-5 so a correction continues
    # from exactly the points shown here instead of re-detecting on its own.
    # NOTE: deep-copy each dict — the scale-back loop below divides 'cx'/'cy'
    # in place, which would otherwise rewrite this state into original coords.
    _kept_scaled = [dict(d) for d in kept_in]
    try:
        _sup_scaled = [dict(d) for d in result2.get("suppressed", [])]
    except Exception:
        _sup_scaled = []
    if _filtered_out and os.environ.get("POINT_DIAG", "1") != "0":
        for d in _filtered_out:
            cx = d.get('cx', d.get('cx_px', -1)); cy = d.get('cy', d.get('cy_px', -1))
            _reason = "outside plot area"
            if legend_area_px is not None:
                lx0, ly0, lx1, ly1 = legend_area_px
                if lx0 <= cx <= lx1 and ly0 <= cy <= ly1:
                    _reason = "inside legend area"
            log_fn(f"    [filtered] {d.get('class_name','?')} @({cx:.0f},{cy:.0f}) removed: {_reason}")
    log_fn(f"  {len(kept_in)} markers inside plot area (legend excluded)")

    # ── Scale coordinates back to original image space if scaled ───────────
    if _upscale != 1.0:
        for d in kept_in:
            if 'cx' in d: d['cx'] = d['cx'] / _upscale
            if 'cy' in d: d['cy'] = d['cy'] / _upscale
        # Restore original-scale plot_area_px for coordinate conversion
        ax0, ay0, ax1, ay1 = plot_area_px
        ax0 = int(ax0 / _upscale); ay0 = int(ay0 / _upscale)
        ax1 = int(ax1 / _upscale); ay1 = int(ay1 / _upscale)
        plot_area_px = (ax0, ay0, ax1, ay1)
        if legend_area_px is not None:
            legend_area_px = tuple(int(v / _upscale) for v in legend_area_px)

    # ── Convert pixel → data coordinates ────────────────────────────────────────────────────────────────────
    log_fn("[Step 4] Converting pixel → data coordinates …")
    detections = []
    for d in kept_in:
        cx_px = d.get('cx', d.get('cx_px', 0))
        cy_px = d.get('cy', d.get('cy_px', 0))
        xd, yd = px_to_data(cx_px, cy_px, plot_area_px,
                             x_range, y_range, x_log, y_log)
        detections.append({
            'class_name': d['class_name'],
            **identity_fields(d),
            'cx_px':      cx_px,
            'cy_px':      cy_px,
            'x_data':     xd,
            'y_data':     yd,
            'confidence': d.get('confidence', 1.0),
        })

    # Sort by x_data
    detections.sort(key=lambda d: d['x_data'])

    # ── [point-diag] FINAL output points — exactly what the overlay/GUI shows.
    #    Coordinates are ORIGINAL-image pixels (already scaled back), so they match
    #    the overlay and any pixel grid drawn on the original image.
    if os.environ.get("POINT_DIAG", "1") != "0":
        try:
            from collections import Counter as _Cnt2
            _fc = _Cnt2(d['class_name'] for d in detections)
            log_fn(f"  [final] {len(detections)} output points, by class: {dict(_fc)}")
            for d in detections:
                log_fn(f"    [final] {d['class_name']} @px({d['cx_px']:.0f},{d['cy_px']:.0f}) "
                       f"data=({d['x_data']:.4g},{d['y_data']:.4g}) conf={d.get('confidence', 1.0):.2f}")
        except Exception as _fde:
            log_fn(f"  [final] final-points log failed: {_fde}")


    # ── Build overlay image (always on original-scale image) ────────────
    _record_timing('filter_coordinates', _filter_started)
    _overlay_started = time.perf_counter()
    log_fn("[Step 5] Building overlay image …")
    overlay = _orig_img_bgr.copy()
    _oax0, _oay0, _oax1, _oay1 = _orig_plot_area_px

    # Draw plot area rectangle
    cv2.rectangle(overlay, (_oax0, _oay0), (_oax1, _oay1), (0, 200, 0), 2)

    # Draw legend area rectangle
    if _orig_legend_px is not None:
        lx0, ly0, lx1, ly1 = _orig_legend_px
        cv2.rectangle(overlay, (lx0, ly0), (lx1, ly1), (200, 0, 200), 2)

    # Dynamic overlay sizes proportional to image dimensions
    _oh, _ow = overlay.shape[:2]
    _ref_dim  = max(_oh, _ow)                        # longest side
    _r_outer  = max(4, int(_ref_dim * 0.013))        # outer circle radius
    _r_inner  = max(2, int(_r_outer * 0.25))         # inner dot radius
    _font_sc  = max(0.30, _ref_dim * 0.00065)        # font scale
    _thickness = max(1, int(_ref_dim * 0.002))       # line thickness

    # Draw detected markers (cx_px/cy_px already in original-scale coords)
    for d in detections:
        cx = int(round(float(d['cx_px'])))
        cy = int(round(float(d['cy_px'])))
        color = series_color(d, MARKER_COLORS)
        # BGR order
        color_bgr = (color[2], color[1], color[0])
        cv2.circle(overlay, (cx, cy), _r_outer, color_bgr, _thickness)
        cv2.circle(overlay, (cx, cy), _r_inner, color_bgr, -1)
        # Label
        short = (d.get('swatch_id','')+' '+d['class_name'].replace('_marker', '').replace('_', ' ')).strip()
        cv2.putText(overlay, short, (cx + _r_outer + 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, _font_sc, color_bgr, _thickness, cv2.LINE_AA)

    _record_timing('overlay', _overlay_started)
    _ocr_started = time.perf_counter()
    # ── Extract legend text labels ───────────────────────────────────────────────
    legend_labels: dict[str, str] = {}
    if _orig_legend_px is not None:
        try:
            from chart_preprocessing import extract_legend_labels as _ell
            # Use the original-scale legend box and image
            _detected_classes = list(dict.fromkeys(d['class_name'] for d in detections))
            if point_backend == 'grid_v46' and 'result2' in locals():
                from bw_pipeline_v46 import legend_labels_from_swatches
                entries=[{**s,'box':tuple(int(round(v/_upscale)) for v in s['box'])}
                         for s in result2.get('diagnostics', {}).get('swatches', [])]
                legend_labels = legend_labels_from_swatches(_orig_img_bgr,_orig_legend_px,entries)
                result2['diagnostics']['legend_labels'] = legend_labels
            else:
                legend_labels = _ell(
                    _orig_img_bgr,
                    _orig_legend_px,
                    known_classes=_detected_classes,
                    verbose=False,
                )
            log_fn(f"  Legend labels: {legend_labels}")
        except Exception as _le:
            log_fn(f"  Legend label extraction failed: {_le}")

    # Attach series_label to each detection
    for d in detections:
        d['series_label'] = legend_labels.get(series_key(d), '')
    _record_timing('legend_ocr', _ocr_started)

    log_fn(f"[Done] {len(detections)} data points found.")
    # Build final overlay step for diagnostics
    _final_diag_started = time.perf_counter()
    if diagnostic_detail:
        try:
            import io as _io_fin
            import matplotlib.pyplot as _plt_fin
            _fin_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            _fig_fin, _ax_fin = _plt_fin.subplots(1, 1, figsize=(12, 8))
            _ax_fin.imshow(_fin_rgb); _ax_fin.set_title(f'Final overlay  ({len(detections)} data points)', fontsize=11); _ax_fin.axis('off')
            _plt_fin.suptitle('Step 3–4 — Filter + Coordinate Conversion + Overlay', fontsize=12, fontweight='bold')
            _plt_fin.tight_layout()
            _buf_fin = _io_fin.BytesIO()
            _plt_fin.savefig(_buf_fin, dpi=120, bbox_inches='tight', format='png')
            _plt_fin.close()
            _buf_fin.seek(0)
            _fin_diag_arr = cv2.imdecode(np.frombuffer(_buf_fin.read(), np.uint8), cv2.IMREAD_COLOR)
            _diag_steps.append({'title': f'Step 3–4 — Final Overlay ({len(detections)} points)', 'img_bgr': _fin_diag_arr})
        except Exception:
            pass
    else:
        _diag_steps.append({'title': f'Step 3–4 — Final Overlay ({len(detections)} points)',
                            'img_bgr': overlay})
    _record_timing('diagnostics', _final_diag_started)
    pipeline_timings['total'] = time.perf_counter()-_pipeline_started
    for _stage, _seconds in pipeline_timings.items():
        log_fn(f'[pipeline timing] {_stage}: {_seconds:.6f}s')
    return {
        'detections':    detections,
        'overlay_img':   overlay,
        'segs':          segs,
        'legend_labels': legend_labels,
        'diag_steps':    _diag_steps,
        'mode_xs':       _result2_ref.get('mode_xs', np.array([])) if (_result2_ref := locals().get('result2')) else np.array([]),
        'prep_info':     prep_info,
        'scaled_img_bgr': img_bgr,   # upscaled (or original if upscale=1) image
        'upscale':        _upscale,  # scale factor applied
        'd_override':     _d_override_used,   # d_est used for NMS (for Step-5 correction consistency)
        'kept_scaled':       _kept_scaled,     # detection state in SCALED coords (Step-5 handoff)
        'suppressed_scaled': _sup_scaled,
        'point_backend': point_backend,
        'diagnostic_detail': diagnostic_detail,
        'pipeline_timings': pipeline_timings,
        'pipeline_timing_scope': ('run_detection wall seconds; named stages are non-overlapping; '
            'segments/errorbar exclude separately timed diagnostic rendering. points includes '
            'the point detector and its own diagnostics. total includes minor glue/logging; '
            'CLI imports, image decoding, optional correction and file export are outside.'),
        'grid_diagnostics': result2.get('diagnostics') if 'result2' in locals() else None,
    }

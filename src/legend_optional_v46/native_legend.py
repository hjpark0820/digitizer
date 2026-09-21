"""Read-only v46 colour legend analysis.

Public boxes are half-open xyxy. Shared analysis uses inclusive right/bottom
edges; conversion happens once at entry. All palette stages are explicit
functions. No historical CLI source or source-line ranges are required.
"""
from __future__ import annotations

import contextlib
import io
import time
import types

import cv2
import numpy as np

from analysis_session_v46 import AnalysisSession
from legend_palette_v46 import (filter_palette_noise, recover_palette,
    set_palette_names, lock_palette, sync_palette, select_swatch_boxes)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _inclusive(box, shape):
    x0, y0, x1, y1 = [int(v) for v in box]
    height, width = shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'Expected in-bounds half-open box, got {box}')
    return x0, y0, x1 - 1, y1 - 1


def extract(image_bgr, legend_area, plot_area):
    """Execute shared v46 extraction for one supplied legend/plot rectangle.

    Returns JSON-ready native sampling boxes, raw colour crops, palette values,
    and native line-stripped marker templates. `box` is half-open; original
    `native_box_inclusive` remains available without silent clipping/recentering.
    """
    started = time.perf_counter()
    session = AnalysisSession()
    env = session.values
    img = np.asarray(image_bgr, np.uint8)
    height, width = img.shape[:2]
    lb = _inclusive(legend_area, img.shape)
    pa = _inclusive(plot_area, img.shape)
    env.update(img=img, img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
               img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32),
               img_lab=cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32),
               H=height, W=width, USER_LEGEND_BOX=lb, USER_PLOT_AREA=pa,
               PLOT_AREA=pa, LEGEND_IMG_PATH=None, _HAS_OCR=False,
               AXIS_ROWS=np.asarray([pa[3]], int),
               AXIS_COLS=np.asarray([pa[0]], int),
               _LAST_LEGEND_GRID=None, _LAST_UNIFIED_GRID=None,
               _LAST_ACHRO_SWATCHES=[], _LEGEND_SWATCH_INFO=[],
               _LEGEND_1TO1=False, _OCR_LEGEND_USED=False)
    env['DARK_BG'] = env['_detect_dark_background']()
    env['_BORDER_V'] = env['_border_brightness']()
    env['GREY_PEAK_DIST_EFF'] = 40 if env['_BORDER_V'] >= 248 else 60
    env['BG_MASK'] = env['_pcm_background_mask'](env['img_hsv'])
    env['PLOT_MASK'] = env['_plot_area_mask']()
    env['NOISE_LAB'] = env['_detect_noise_color']()
    env['NOISE_MATCH_DIST'] = 45.0
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        # Exact original user-box discovery, including original colour masks.
        # No OCR/hue-histogram branch is eligible with USER_LEGEND_BOX supplied.
        colors, legend_box = env['_discover_colors']()
        env.update(COLORS=colors, LEGEND_BOX=legend_box)
        discovery_colors = [{k: v for k, v in c.items()
                             if not k.startswith('_')} for c in colors]
        # Keep the original count-affecting noise-filter control flow as well.
        session.run(filter_palette_noise)
        env['_lb'] = legend_box
        env['_cfg'] = types.SimpleNamespace(curve_names=[])
        # Palette/table prelude, including the genuine achromatic row recovery.
        session.run(recover_palette)
        session.run(set_palette_names)
        # Original grid palette lock (skip JSON output and digitizer stages).
        session.run(lock_palette)
        # Calling set_palette on a lightweight object performs the original
        # palette/sink bookkeeping without constructor plot clustering or I/O.
        digitizer = types.SimpleNamespace(cfg=env['_cfg'])
        env['PlotDigitizer'].set_palette(digitizer, env['_rgbs'], add_black_if_missing=False,
                           add_black_sink=env['_add_sink'])
        env['_dig'] = digitizer
        session.run(sync_palette)
        # Palette preference: unified table, raw table, native own scan;
        # prefer count matching real curves, otherwise largest candidate set.
        session.run(select_swatch_boxes)
        swatch_boxes = env['_sw_boxes']
        if swatch_boxes is None:
            swatch_boxes = env['find_legend_swatches'](img, lb)
        # Native colour_masks establishes sampled ink and tolerance. Furniture
        # changes reported plot masks only, not swatch samples or templates.
        if swatch_boxes:
            _, _, info = env['colour_masks'](img, pa, lb, metric='tube',
                                              spatial=False, furniture=None,
                                              swatch_boxes=swatch_boxes)
        else:
            info = []
        swatches = []
        for i, item in enumerate(info):
            native = tuple(int(v) for v in item['box'])
            mask = env['marker_template'](
                img, native, item['rgb'],
                [other['rgb'] for other in info if other is not item], item['tol'])
            x0, y0, x1, y1 = native
            raw = img[y0:y1 + 1, x0:x1 + 1]
            swatches.append(dict(
                id=f'S{i + 1:02d}', box=[x0, y0, x1 + 1, y1 + 1],
                native_box_inclusive=native, color_rgb=item['rgb'], raw_bgr=raw,
                template_mask=None if mask is None else mask.astype(np.uint8),
                template_shape=None if mask is None else list(mask.shape),
                template_pixels=0 if mask is None else int(mask.sum()),
                shape_name=None, palette_info=item,
                centre=[(x0 + x1) / 2., (y0 + y1) / 2.],
                native_box_out_of_image=bool(x0 < 0 or y0 < 0 or x1 >= width or y1 >= height)))
    return _plain(dict(
        version='v46-colour-user-box-legend-adapter', swatches=swatches,
        diagnostics=dict(
            source_module='chart_analysis_v46',
            source_validation=dict(status='standalone_v46_library'),
            legend_area_half_open=list(legend_area), plot_area_half_open=list(plot_area),
            native_legend_box_inclusive=lb, native_plot_area_inclusive=pa,
            source_box_selection=env.get('_sw_src'),
            native_box_candidates=[dict(boxes=b, source=s) for b, s in env.get('_cands', [])],
            discovery_colors=discovery_colors,
            native_swatch_info=env.get('_LEGEND_SWATCH_INFO', []),
            final_real_colors=[{k: v for k, v in c.items() if not k.startswith('_')}
                               for c in env['COLORS'] if not str(c.get('name', '')).endswith('_sink')],
            raw_grid=env.get('_LAST_LEGEND_GRID'),
            achro_swatches=env.get('_LAST_ACHRO_SWATCHES'),
            unified_grid=env.get('_LAST_UNIFIED_GRID'),
            count=len(swatches), logs=log.getvalue().splitlines(),
            elapsed_seconds=time.perf_counter() - started,
            caveats=[
                'Same provided legend rectangle; automatic whole-image legend localization not compared.',
                'User-box discovery does not enter OCR or whole-plot hue-histogram fallbacks.',
                'Line-stripped template is not a semantic shape classification.',
                'Native sampling boxes can extend outside the provided legend rectangle.',
                'Full colour-mask creation is executed for original palette behavior; no data-marker detection runs.',
                'Furniture argument omitted only in final colour_masks; it affects plot-mask diagnostics, not ink/tolerance/template.',
            ])))

"""Explicit v46 legend palette stages, run in an AnalysisSession."""

def filter_palette_noise():
    global COLORS, _agree, _axis_cols_list, _axis_rows_list, _cd, _is_noise, _kept_colors, _legend_1to1, _mem, _rm, _xcov
    _axis_rows_list = list(AXIS_ROWS) if 'AXIS_ROWS' in globals() else None
    _axis_cols_list = list(AXIS_COLS) if 'AXIS_COLS' in globals() else None
    _legend_1to1 = bool(globals().get('_LEGEND_1TO1', False))
    if _legend_1to1:
        print('  [noise filter] skipped (legend matched 1:1; all colours trusted)')
    else:
        _kept_colors = []
        for _cd in COLORS:
            _rm = _cd.get('_raw_mask')
            if _rm is None:
                _kept_colors.append(_cd)
                continue
            _mem = _rm > 0 if _rm.dtype != bool else _rm
            if _mem.ndim == 3:
                _mem = _mem.any(axis=2)
            _agree, _xcov = _seed_walk_agreement(_mem, _axis_rows_list, _axis_cols_list)
            _is_noise = _agree > NOISE_AGREE_MAX or _xcov < NOISE_XCOV_MIN
            if _is_noise:
                print(f'  [noise filter] dropping {_cd['name']} (agree={_agree:.1f}, xcov={_xcov:.2f})')
            else:
                _kept_colors.append(_cd)
        if _kept_colors:
            COLORS = _kept_colors
        else:
            print('  [noise filter] all colours looked like noise; keeping original set')


def recover_palette():
    global _ac, _ach_from_legend, _added, _are, _chrom_legend, _cur, _grid_now, _names, _recover_neutral_columns, _rgbs, _unified, a, c, cd
    _names = [cd['name'] for cd in COLORS]
    _rgbs = [tuple((int(c) for c in cd['mean_rgb'])) for cd in COLORS]
    _grid_now = globals().get('_LAST_LEGEND_GRID')
    _chrom_legend = [tuple(c[2]) for c in _grid_now['cells']] if _grid_now and _grid_now.get('cells') else _rgbs
    if globals().get('_OCR_LEGEND_USED'):
        _ach_from_legend = []
    else:
        _ach_from_legend = _classify_legend_swatches(_lb, existing_rgbs=_chrom_legend)
    for _ac in _ach_from_legend:
        _rgbs.append(_ac)
        _names.append('black' if sum(_ac) / 3.0 < 90 else 'grey')
    if _ach_from_legend:
        print(f'  [legend achromatic swatches: {_ach_from_legend}]')
    try:
        if _grid_now and _grid_now.get('cells') and (_lb is not None):
            from legend_achromatic_recovery_v46 import recover as _recover_neutral_columns
            _cur = list(globals().get('_LAST_ACHRO_SWATCHES', []))
            _added = _recover_neutral_columns(img_rgb, _lb, _grid_now['cells'], _cur)
            if _added:
                globals()['_LAST_ACHRO_SWATCHES'] = _cur + _added
                print(f'  [achro column recovery: +{len(_added)} dark swatch(es) at xy={[(a[0], a[1]) for a in _added]}]')
    except Exception as _are:
        print(f'  [achro row recovery skipped: {_are}]')
    _unified = _build_unified_legend_grid((_grid_now or {}).get('cells'), globals().get('_LAST_ACHRO_SWATCHES', []))
    globals()['_LAST_UNIFIED_GRID'] = _unified
    if _unified is not None:
        print(f'  [unified legend grid: {len(_unified['cols'])} cols x {len(_unified['rows'])} rows, {len(_unified['cells'])} filled / {len(_unified['empty'])} empty, aligned={_unified['aligned']} (residual={_unified['residual']}px)]')


def set_palette_names():
    _cfg.curve_names = list(_names)


def lock_palette():
    global _add_sink, _cell, _cell_items, _close, _grid_names, _grid_rgbs, _has_achro, _key, _names, _rgb, _rgbs
    _add_sink = True
    if _unified is not None and _unified.get('cells') and globals().get('_LEGEND_1TO1'):
        _cell_items = sorted(_unified['cells'].items(), key=lambda kv: (int(kv[0].split(',')[1]), int(kv[0].split(',')[0])))
        _grid_rgbs = []
        _grid_names = []
        _has_achro = False

        def _close(a, b, t=18):
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) <= t
        for _key, _cell in _cell_items:
            _rgb = tuple((int(v) for v in _cell['rgb']))
            if any((_close(_rgb, g) for g in _grid_rgbs)):
                continue
            _grid_rgbs.append(_rgb)
            if _cell.get('type') == 'achro':
                _has_achro = True
                _grid_names.append('black' if sum(_rgb) / 3.0 < 90 else 'grey')
            else:
                _grid_names.append(f'color{len(_grid_rgbs) + 1:02d}')
        if len(_grid_rgbs) >= LEGEND_MIN_ENTRIES:
            _rgbs = _grid_rgbs
            _names = _grid_names
            _cfg.curve_names = list(_names)
            _add_sink = not _has_achro
            print(f'  [extraction palette locked to legend grid: {len(_rgbs)} colours{(' (achro present -> no sink)' if _has_achro else '')}]')


def sync_palette():
    global COLORS, _cd, _i, _mrgb, _names, _new_colors, _nm, _pal_rgbs, _prev_by_name, cd, v
    _names = list(_dig.cfg.curve_names)
    _prev_by_name = {cd.get('name'): cd for cd in COLORS}
    _pal_rgbs = getattr(_dig, 'palette', None)
    _new_colors = []
    for _i, _nm in enumerate(_names):
        _cd = _prev_by_name.get(_nm, {})
        if _pal_rgbs is not None and _i < len(_pal_rgbs):
            _mrgb = [int(v) for v in _pal_rgbs[_i]]
        else:
            _mrgb = _cd.get('mean_rgb', [0, 0, 0])
        _new_colors.append({'name': _nm, 'mean_rgb': _mrgb, 'px_count': _cd.get('px_count', 0)})
    COLORS = _new_colors


def select_swatch_boxes():
    global _boxes_from, _cands, _cc, _ci, _exact, _k, _lb, _pa, _real0, _rg, _ri, _scan, _sw_boxes, _sw_src, _ug, b, c, q, t
    _pa = tuple((int(v) for v in PLOT_AREA))
    _lb = tuple((int(v) for v in LEGEND_BOX or (0, 0, 1, 1)))
    _real0 = [c for c in COLORS if not str(c.get('name', '')).endswith('_sink')]

    def _boxes_from(centres, lbl):
        if not centres:
            return (None, lbl)
        _ry = sorted({int(round(c[1])) for c in centres})
        _p = float(np.median(np.diff(_ry))) if len(_ry) > 1 else max(10.0, (_lb[3] - _lb[1]) / 2.0)
        _hh = max(4, int(round(0.45 * _p)))
        _hw = max(8, int(round(0.9 * _p)))
        return ([(int(round(c[0])) - _hw, int(round(c[1])) - _hh, int(round(c[0])) + _hw, int(round(c[1])) + _hh) for c in sorted(centres, key=lambda q: (q[0], q[1]))], f'{lbl} (row pitch {_p:.0f}px)')
    _cands = []
    _ug = globals().get('_LAST_UNIFIED_GRID') or {}
    if _ug.get('cols') and _ug.get('rows') and _ug.get('cells'):
        _cc = []
        for _k in sorted(_ug['cells'], key=lambda k: (int(k.split(',')[0]), int(k.split(',')[1]))):
            _ci, _ri = (int(q) for q in _k.split(','))
            _cc.append((_ug['cols'][_ci], _ug['rows'][_ri]))
        _cands.append(_boxes_from(_cc, 'unified legend grid'))
    _rg = globals().get('_LAST_LEGEND_GRID') or {}
    if _rg.get('cells'):
        _cands.append(_boxes_from([(c[0], c[1]) for c in _rg['cells']], 'raw legend table'))
    try:
        _scan = find_legend_swatches(img, _lb)
        _cands.append(_boxes_from([((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in _scan], 'own swatch scan'))
    except Exception:
        pass
    _cands = [(b, t) for b, t in _cands if b]
    _sw_boxes, _sw_src = (None, 'auto')
    if _cands:
        _exact = [q for q in _cands if len(q[0]) == len(_real0)]
        _sw_boxes, _sw_src = _exact[0] if _exact else max(_cands, key=lambda q: len(q[0]))
        print(f'  [v46] {len(_sw_boxes)} swatch box(es) from {_sw_src}' + ('' if len(_sw_boxes) == len(_real0) else f'  -- WARNING {len(_real0)} curve(s) expected'))



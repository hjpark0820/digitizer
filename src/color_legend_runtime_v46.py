"""v46 legend composition and explicit colour-sample adapters."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def exclusive_box(box, image_shape):
    """Clip analysis inclusive coordinates once, returning nonempty half-open xyxy."""
    height, width = image_shape[:2]
    x0, y0, x1, y1 = map(int, box)
    result = max(0, x0), max(0, y0), min(width, x1 + 1), min(height, y1 + 1)
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f'Empty legend rectangle after clipping: {box}')
    return result


class LegendRuntime:
    def __init__(self, source_path=None, *, allow_line_only=False):
        # Optional positional argument retained for old experiment callers only.
        # Analysis always uses the standalone library, regardless of that argument.
        self.source_path = Path(__file__).with_name("chart_analysis_v46.py")
        self.allow_line_only = allow_line_only
        self.active = False
        self.entries = []
        self.env = None
        self.diagnostics = {
            'version': 'v46-hybrid-colour-legend-runtime',
            'analysis_source_sha256': hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            'hybrid_source_sha256': hashlib.sha256(Path(__file__).with_name('color_legend_hybrid_v46.py').read_bytes()).hexdigest(),
            'composition_source_sha256': hashlib.sha256(Path(__file__).with_name('legend_composition_v46.py').read_bytes()).hexdigest(),
            'scope': 'v46 legend composition and standalone colour analysis',
            'coordinate_convention': 'half-open xyxy except native_box_inclusive',
            'grid_refinement_guard': {
                'calls': 0, 'native_nonempty_calls': 0, 'empty_grid_calls': 0,
                'observed_markers_returned': 0, 'empty_grid_events': [],
                'policy': 'Empty shared grid keeps only observed marker coordinates; '
                          'no interpolation, column snapping or v44 fallback. '
                          'Original downstream Scorer/verdict still filters the returned candidates.',
            },
        }

    def discover(self, original, env):
        # Preserve initial products before later palette filtering changes them.
        colors, legend_box = original()
        self.diagnostics['initial_palette'] = _plain([
            {k: v for k, v in item.items() if not k.startswith('_')}
            for item in colors])
        self.diagnostics['initial_grid'] = _plain(env.get('_LAST_LEGEND_GRID'))
        return colors, legend_box

    def _save(self):
        if self.env is not None and self.env.get('OUT_DIR'):
            target = Path(self.env['OUT_DIR']) / 'legend_hybrid_v46.json'
            target.write_text(json.dumps(_plain(self.diagnostics), indent=2), encoding='utf-8')

    def prepare(self, env):
        """Lock every downstream list to the exact same accepted-entry order."""
        from color_legend_hybrid_v46 import extract_hybrid_legend
        from color_legend_composition_v46 import enrich_entry

        self.env = env
        legend_box = env.get('LEGEND_BOX')
        if not legend_box:
            raise ValueError('v46 no-legend detection must use legend_optional_v46; legacy no-legend fallback is disabled')
        image = env['img']
        area = exclusive_box(legend_box, image.shape)
        native_grid = env.get('_LAST_UNIFIED_GRID') or env.get('_LAST_LEGEND_GRID')
        palette, centres = native_seeds(native_grid, env.get('_LEGEND_SWATCH_INFO', []))
        self.diagnostics.update(legend_area=area, native_grid=_plain(native_grid),
                                native_palette=palette, native_centres=centres)
        try:
            models, reports, rejected = extract_hybrid_legend(
                image, area, native_grid=native_grid,
                native_palette=palette, native_centres=centres)
        except Exception as error:
            self.diagnostics.update(status='error', error=f'{type(error).__name__}: {error}')
            self._save()
            raise RuntimeError('v46 hybrid legend extraction failed; no silent detector fallback') from error
        self.diagnostics['rejected'] = _plain(rejected)
        self.diagnostics['hybrid_reports'] = _plain(reports)
        has_line_entries = any(r.get('status') == 'line_only' and r.get('observed_line_box') for r in rejected)
        if self.allow_line_only and has_line_entries:
            # An explicitly marker-free chart consumes observed native line
            # keys only. Extra hybrid shape proposals can be adjacent text;
            # they must not manufacture palette identities in this mode.
            self.diagnostics['line_only_excluded_marker_models'] = [m.swatch_id for m in models]
            models = []
        if not models and not (self.allow_line_only and has_line_entries):
            self.diagnostics.update(status='no_marker_models', fallback_enabled=False)
            self._save()
            raise ValueError('v46 found no usable marker legend. Check the legend rectangle '
                             'or select Auto / line-only for a chart without markers.')
        report_by_id = {r['swatch_id']: r for r in reports if r.get('swatch_id')}
        for model in models:
            report = report_by_id.get(model.swatch_id, {})
            box = tuple(map(int, report['marker_body_box']))
            x0, y0, x1, y1 = box
            padded = np.asarray(model.mask, dtype=bool)
            ox = int(round(model.marker_center[0] - (padded.shape[1] - 1) / 2))
            oy = int(round(model.marker_center[1] - (padded.shape[0] - 1) / 2))
            domain = np.s_[y0-oy:y1-oy, x0-ox:x1-ox]
            mask = padded[domain].copy()
            raw = image[y0:y1, x0:x1].copy()
            if raw.shape[:2] != mask.shape or not mask.any():
                raise ValueError(f'Hybrid model/crop mismatch for {model.swatch_id}: '
                                 f'{raw.shape[:2]} != {mask.shape}')
            rgb = tuple(map(int, report['color_rgb']))
            self.entries.append(dict(id=model.swatch_id, rgb=rgb, box=box,
                                     mask=mask.copy(), raw_bgr=raw,
                                     required_weight=np.asarray(model.required_weight)[domain].copy(),
                                     soft=np.asarray(model.raw_soft)[domain].copy(),
                                     source_structure_box=report.get('source_structure_box'),
                                     marker_template=True, report=report))
        # A line-only legend item is a valid colour series, but never a marker
        # template. Retain its actually observed narrow support and RGB without
        # manufacturing a marker or taking pixels from a broad text window.
        for rejection in rejected:
            if rejection.get('status') != 'line_only' or not rejection.get('observed_line_box'):
                continue
            box = tuple(map(int, rejection['observed_line_box']))
            raw = image[box[1]:box[3], box[0]:box[2]].copy()
            mask = np.asarray(rejection['observed_line_mask'], bool)
            if not mask.any() or mask.shape != raw.shape[:2]:
                raise ValueError('Invalid observed line-only evidence from hybrid extractor')
            self.entries.append(dict(id=f"L{int(rejection['native_index'])+1:02d}",
                rgb=tuple(map(int, rejection['native_palette_rgb'])), box=box,
                mask=mask, raw_bgr=raw, required_weight=mask.astype(np.float32),
                soft=mask.astype(np.float32), marker_template=False, report=rejection))
        # Locate with the robust hybrid extractor, then explain the original
        # full connected swatch as line UNION primitive. Unsupported fits keep
        # their observed hybrid body; raw colour samples are never inferred.
        for entry in self.entries:
            enrich_entry(image, entry, area)
        self.entries.sort(key=lambda e: (e['report'].get('native_index')
                                        if e['report'].get('native_index') is not None else 10**9,
                                        e['box'][1], e['box'][0]))
        # Entry order, NOT nearest RGB, defines identity (same RGB can encode
        # filled/open symbols). The native set_palette does not deduplicate.
        rgbs = [entry['rgb'] for entry in self.entries]
        names = [f'color{i + 2:02d}' for i in range(len(self.entries))]
        env['_rgbs'], env['_names'] = rgbs, names
        env['_cfg'].curve_names = names.copy()
        env['_add_sink'] = not any(max(rgb) - min(rgb) <= 25 and np.mean(rgb) < 80
                                   for rgb in rgbs)
        env['COLORS'] = [{'name': name, 'mean_rgb': list(rgb), 'swatch_rgb': list(rgb),
                          'px_count': 0} for name, rgb in zip(names, rgbs)]
        grid = output_grid(self.entries, native_grid)
        env['_LAST_UNIFIED_GRID'] = grid
        env['_LAST_LEGEND_GRID'] = dict(cols=grid['cols'], rows=grid['rows'],
                                        cells=[[(e['box'][0] + e['box'][2] - 1) / 2,
                                                (e['box'][1] + e['box'][3] - 1) / 2,
                                                list(e['rgb'])] for e in self.entries])
        env['_LEGEND_SWATCH_INFO'] = [dict(bbox=(e['box'][1], e['box'][3]-1,
                                                                e['box'][0], e['box'][2]-1),
                                          lab_samples=cv2.cvtColor(e['raw_bgr'], cv2.COLOR_BGR2Lab)[e['sample_mask']],
                                          s_samples=cv2.cvtColor(e['raw_bgr'], cv2.COLOR_BGR2HSV)[..., 1][e['sample_mask']])
                                        for e in self.entries]
        env['_LEGEND_1TO1'] = True
        self.active = True
        self.diagnostics.update(status='hybrid_active', count=len(self.entries),
                                composition_supported=sum(e.get('composition_template') is not None for e in self.entries),
                                names=names, output_palette=rgbs, output_grid=grid,
                                entries=[{k: v for k, v in e.items() if k != 'report'}
                                         for e in self.entries])
        self._save()
        # The preparation stage already wrote its initial grid before prepare; write the actual
        # grid consumed by labels and GUI, retaining original in diagnostic JSON.
        (Path(env['OUT_DIR']) / 'legend_grid.json').write_text(json.dumps(_plain(dict(
            legend_box=legend_box, grid=env['_LAST_LEGEND_GRID'],
            achro=[], unified=grid, extractor='v46_hybrid')), indent=2), encoding='utf-8')
        print(f'  [v46 colour legend] Hybrid active: {len(self.entries)} swatches; '
              f"{self.diagnostics['composition_supported']} supported line+shape compositions; "
              'observed hybrid fallback for unsupported shapes. Wide-box RGB resampling OFF; '
              'legacy line stripping OFF. -> legend_hybrid_v46.json')

    def sample_arrays(self, image):
        inks, cores, achro, boxes, alls = [], [], [], [], []
        for entry in self.entries:
            rgb = entry['rgb']
            pixels = entry['raw_bgr'][..., ::-1][entry.get('sample_mask', entry['mask'])].astype(np.float32)
            # Core samples only from actual body pixels. Neither white paper
            # nor adjacent text/connector tails can redefine the palette colour.
            distance = np.linalg.norm(pixels - np.asarray(rgb), axis=1)
            core = pixels[distance <= np.percentile(distance, 50)]
            inks.append(rgb)
            cores.append(core)
            achro.append(max(rgb) - min(rgb) <= 25)
            boxes.append(inclusive(entry['box']))
            alls.append(pixels)
        return inks, cores, achro, boxes, alls

    def colour_masks(self, original, *args, **kwargs):
        if not self.active:
            return original(*args, **kwargs)
        kwargs['swatch_boxes'] = [inclusive(e['box']) for e in self.entries]
        image = args[0] if args else kwargs['img_bgr']
        kwargs['samples'] = self.sample_arrays(image)
        result = original(*args, **kwargs)
        self.diagnostics['downstream_info'] = _plain(result[2])
        self._save()
        return result

    def marker_template(self, original, image, box, ink_rgb, rivals_rgb, tol_lab, **kwargs):
        if not self.active:
            return original(image, box, ink_rgb, rivals_rgb, tol_lab, **kwargs)
        matches = [e for e in self.entries if inclusive(e['box']) == tuple(map(int, box))]
        if len(matches) != 1 or tuple(map(int, ink_rgb)) != matches[0]['rgb']:
            raise RuntimeError('Hybrid template identity lost between palette, box and downstream slot')
        entry = matches[0]
        if not entry['marker_template']:
            return None
        if entry.get('composition_template') is not None:
            return entry['composition_template']['soft'] >= .5
        return entry['mask'].copy()

    def record_templates(self, info, templates, pairs):
        self.diagnostics['exact_downstream_templates'] = {
            str(i): None if template is None else np.asarray(template, np.uint8).tolist()
            for i, template in templates.items()}
        self.diagnostics['exact_downstream_info'] = _plain(info)
        self.diagnostics['curve_to_template_slot'] = _plain(pairs)
        self._save()

    def refine_on_grid(self, original, path, filled, grid_xs, markers, lo, *args, **kwargs):
        """Handle absent shared columns without inventing or moving measurements.

        The grid refinement assumes the shared x-grid is nonempty after checking marker count.
        Real template detections can exist even when grid consensus returns no
        columns. Passing those observed coordinates to the *existing* downstream
        scorer is safe; reading a path or creating columns in that case is not.
        """
        report = self.diagnostics['grid_refinement_guard']
        report['calls'] += 1
        if len(grid_xs):
            report['native_nonempty_calls'] += 1
            self._save()
            return original(path, filled, grid_xs, markers, lo, *args, **kwargs)
        # Bind the actual native signature rather than hard-coding where an
        # optional positional min_markers argument happens to sit.
        bound = inspect.signature(original).bind(path, filled, grid_xs, markers, lo,
                                                 *args, **kwargs)
        bound.apply_defaults()
        minimum = bound.arguments['min_markers']
        accepted = len(markers) >= minimum
        result = [(float(marker[0]), float(marker[1]), 'detected') for marker in markers] if accepted else []
        report['empty_grid_calls'] += 1
        report['observed_markers_returned'] += len(result)
        report['empty_grid_events'].append(dict(
            call=report['calls'], input_markers=len(markers), min_markers=minimum,
            returned_markers=len(result), coordinates=[[x,y] for x,y,_ in result],
            status='observed_only' if accepted else 'insufficient_observed_markers'))
        self._save()
        print(f'  [v46 colour grid guard] Shared x-grid is empty: '
              f'{len(result)}/{len(markers)} observed marker candidates retained '
              f'(min_markers={minimum}); no grid interpolation; downstream scoring retained.')
        return result


def inclusive(box):
    return int(box[0]), int(box[1]), int(box[2]) - 1, int(box[3]) - 1


def native_seeds(grid, swatch_info):
    palette, centres = [], []
    if grid and isinstance(grid.get('cells'), dict):
        for key, cell in sorted(grid['cells'].items(), key=lambda item: tuple(map(int, item[0].split(',')))):
            column, row = map(int, key.split(','))
            centres.append((grid['cols'][column], grid['rows'][row]))
            palette.append(tuple(cell['rgb']))
    elif grid and grid.get('cells'):
        for x, y, rgb in grid['cells']:
            centres.append((x, y)); palette.append(tuple(rgb))
    else:
        for item in swatch_info:
            if 'bbox' not in item or not len(item.get('lab_samples', [])):
                continue
            # Native sampler stores y0,y1,x0,x1, not xyxy.
            y0,y1,x0,x1 = item['bbox']
            lab = np.median(item['lab_samples'], axis=0).astype(np.uint8)
            palette.append(tuple(int(v) for v in cv2.cvtColor(lab[None, None], cv2.COLOR_Lab2RGB)[0, 0]))
            centres.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return palette, centres


def output_grid(entries, native_grid=None):
    centres = [((e['box'][0] + e['box'][2] - 1) / 2,
                (e['box'][1] + e['box'][3] - 1) / 2) for e in entries]
    # Native populated cells supply the table topology; refined bodies retain
    # their own exact measured centres. Tiny body-centre differences must not
    # turn a real 2-column/3-row table into six columns and six rows.
    anchors = [tuple(e.get('report', {}).get('native_grid_center') or centre)
               for e, centre in zip(entries, centres)]
    def coordinates(axis):
        native = list((native_grid or {}).get('cols' if axis == 0 else 'rows', []))
        groups = [[float(v)] for v in native]
        for anchor in anchors:
            value = float(anchor[axis])
            nearest = min(groups, key=lambda group: abs(np.mean(group)-value)) if groups else None
            if nearest is not None and abs(np.mean(nearest)-value) <= 3.:
                nearest.append(value)
            else:
                groups.append([value])
        return sorted(float(np.mean(group)) for group in groups)
    cols, rows = coordinates(0), coordinates(1)
    cells = {}
    residuals = []
    for index, ((x, y), anchor, entry) in enumerate(zip(centres, anchors, entries)):
        ci = min(range(len(cols)), key=lambda i: abs(cols[i] - anchor[0]))
        ri = min(range(len(rows)), key=lambda i: abs(rows[i] - anchor[1]))
        key = f'{ci},{ri}'
        if key in cells:
            raise ValueError('Two hybrid entries occupy the same centre')
        cells[key] = dict(rgb=list(entry['rgb']), type='achro' if max(entry['rgb']) - min(entry['rgb']) <= 25
                         else 'chromatic', idx=index, swatch_id=entry['id'], measured_center=[x,y])
        residuals.append(max(abs(cols[ci] - x), abs(rows[ri] - y)))
    empty = [f'{ci},{ri}' for ci in range(len(cols)) for ri in range(len(rows))
             if f'{ci},{ri}' not in cells]
    residual = float(max(residuals, default=0))
    return dict(cols=cols, rows=rows, cells=cells, empty=empty, aligned=residual <= 4.,
                residual=residual, measured_centres=True, identity_order='idx')

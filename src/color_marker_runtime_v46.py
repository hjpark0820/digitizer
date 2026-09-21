"""Production adapter for the frozen image-only colour hybrid v2 detector.

Only CLI/GUI contracts live here. The evaluated proposal, evidence, window and
identity-selection modules are imported unchanged; no experiment files or point
references are read at runtime. The standalone v46 workflow supplies shared analysis helpers. Missing legend
evidence never selects an older detector.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from color_legend_runtime_v46 import _plain, exclusive_box, inclusive
from color_legend_composition_v46 import marker_template_from_entry, shape_fields_from_template
from color_marker_evidence_v2 import prepare_evidence
from color_marker_hybrid_v2 import SelectionConfig, analyze_evidence, select_candidates
from color_paper_contradiction_v46 import VERSION as PAPER_CENTER_VERSION, DEFAULT_WEIGHT as PAPER_CENTER_WEIGHT
from color_window_physical_v46 import VERSION as PAPER_WINDOW_VERSION, Config as PaperWindowConfig
from triangle_errorbar_v46 import (triangle_prior, apply_guards, weak_suppressed,
                                   protect_weak_candidates, draw_review)

MARKER_VERSION = 'color_marker_hybrid_v2'


def production_config():
    """Snapshot of the development-selected v2 settings, not its .50 API default."""
    return SelectionConfig(threshold=0.24999999999999997,
        same_series_spacing=.36, competing_center_distance=.55,
        minimum_independent_fraction=.22, minimum_independent_area_fraction=.025,
        identity_ambiguity_margin=.025)


def _inside(point, box):
    return box[0] <= point['x_px'] < box[2] and box[1] <= point['y_px'] < box[3]


def _export_pixel(point, plot, legend):
    """Nearest allowed integer pixel; rounding must not enter the legend."""
    x = int(np.clip(round(point['x_px']), plot[0], plot[2]-1))
    y = int(np.clip(round(point['y_px']), plot[1], plot[3]-1))
    if not _inside(dict(x_px=x, y_px=y), legend):
        return x, y
    choices = []
    for a in (np.floor(point['x_px']), np.ceil(point['x_px'])):
        for b in (np.floor(point['y_px']), np.ceil(point['y_px'])):
            candidate = dict(x_px=int(a), y_px=int(b))
            if _inside(candidate, plot) and not _inside(candidate, legend):
                choices.append(((a-point['x_px'])**2+(b-point['y_px'])**2, int(a), int(b)))
    if not choices:
        raise ValueError('No valid integer export pixel for a retained marker centre')
    _, x, y = min(choices)
    return x, y


class MarkerRuntime:
    def __init__(self, *, enable_tentative=True, use_prepared_templates=True, series_mode='markers',
                 guide_policy='auto', enable_triangle_guard=True, enable_shared_scale=True):
        self.active = False
        self.diagnostics = None
        # The off switches isolate historical detector-contract unit tests;
        # the production CLI always uses both defaults.
        self.enable_tentative = enable_tentative
        self.use_prepared_templates = use_prepared_templates
        if guide_policy not in {'auto','model','legacy_band','none'}:
            raise ValueError('Unknown guide policy')
        self.guide_policy = guide_policy
        self.enable_triangle_guard = enable_triangle_guard
        self.enable_shared_scale = bool(enable_shared_scale)
        if series_mode not in ('markers', 'line-only'):
            raise ValueError('Unknown colour series mode')
        self.series_mode = series_mode

    def output_detections(self, detections):
        return dict(detections) if self.active else {k: v for k, v in detections.items() if v}

    def labels(self, original, env, legend_runtime):
        """Keep labels attached to swatch identity even when a rival has zero points."""
        if self.active and self.series_mode == 'line-only':
            return dict(self.type3_labels)
        if not self.active:
            return original()
        labels = {name: name for name in env['all_detections_out']}
        # Reuse the existing swatch-anchored OCR helper, not the BW detector.
        # Include ALL legend entries in layout, including failed/zero-point
        # series; a missing series must not shift the remaining text rows.
        entries = [dict(swatch_id=f'color{i+2:02d}', box=entry['box'])
                   for i, entry in enumerate(legend_runtime.entries)]
        status, error = 'swatch_anchored_ocr', None
        if env.get('_HAS_OCR', False):
            try:
                from bw_pipeline_v46 import legend_labels_from_swatches
                observed = legend_labels_from_swatches(env['img'],
                    self.diagnostics['legend_box'], entries)
                labels.update({name: text for name, text in observed.items() if name in labels})
            except Exception as exc:
                status, error = 'identity_ids_after_ocr_failure', f'{type(exc).__name__}: {exc}'
        else:
            status = 'identity_ids_ocr_unavailable'
        self.diagnostics['labels'] = dict(source=status, values=labels, error=error,
            policy='Source swatch position -> curve ID; never rematch equal RGB to surviving curves')
        (Path(env['OUT_DIR'])/'color_marker_hybrid_v46.json').write_text(
            json.dumps(_plain(self.diagnostics), indent=2), encoding='utf-8')
        return labels

    def run(self, env, legend_runtime):
        if self.series_mode == 'line-only':
            from type3_v46.runtime import run
            return run(self, env, legend_runtime)
        started = perf_counter()
        image = env['img']
        plot = exclusive_box(env['PLOT_AREA'], image.shape)
        legend = exclusive_box(env['LEGEND_BOX'], image.shape)
        entries = legend_runtime.entries
        names = [f'color{i+2:02d}' for i in range(len(entries))]
        priors = {name: triangle_prior(entry.get('report', {}))
                  for name, entry in zip(names, entries)}
        guard_active = self.enable_triangle_guard and any(p.get('applicable') for p in priors.values())
        # Only triangle-bearing marker runs switch from broad guide bands to
        # the jointly tested repeated-dot explanation. Other routes are unchanged.
        guide_policy = ('model' if guard_active else 'legacy_band') if self.guide_policy == 'auto' else self.guide_policy
        # Keep identity by entry index. Equal RGB values must not merge shapes.
        specs = [dict(id=name, label=name,
            swatch_box=list(entry.get('report', {}).get('swatch_box') or entry['box']),
            legend_box=list(legend)) for name, entry in zip(names, entries)]
        config = production_config()
        report = dict(version=MARKER_VERSION, status='running',
            guide_policy=guide_policy, requested_guide_policy=self.guide_policy,
            triangle_errorbar_guard=dict(enabled=self.enable_triangle_guard,
                status='pending' if guard_active else 'not_applicable' if self.enable_triangle_guard else 'disabled',
                priors=priors), weak_suppressed_points=[],
            selection_config=asdict(config), native_resolution=True, max_side=0,
            scale_search=self.enable_shared_scale,scale_policy='shared_symbol' if self.enable_shared_scale else 'fixed_1x',
            plot_box=list(plot), legend_box=list(legend),
            coordinate_convention='half-open xyxy; x_px/y_px are source-image pixel centres',
            series_specs=specs, points=[], uncertain_points=[], rejected=[],
            templates=[], template_errors=[], scale_x=1., scale_y=1.,
            legacy_detection_executed=False, same_colour_bw_v3=False,
            gui_coordinate_policy='nearest allowed integer outside legend; unrounded coordinates retained in this file',
            mask_note='Display-only exclusive colour ownership; not marker shape identity. '
                      'Detection uses continuous overlapping evidence.',
            score_note='Reconstruction quality score, not a calibrated probability')
        destination = Path(env['OUT_DIR'])
        destination.mkdir(parents=True, exist_ok=True)
        self.diagnostics = report
        evidence = None
        try:
            from color_group_runtime_v46 import ambiguous_palette, detect as detect_groups
            if self.use_prepared_templates and ambiguous_palette(entries):
                print('[v46 color markers] Repeated legend colours: switching to reviewed '
                      'colour-group BW grid/window and typed group Step 5.', flush=True)
                detect_groups(self, env, legend_runtime, plot, legend, report)
                return
            report['per_marker_scale_search']=False
            print('[v46 color markers] Hybrid v2: path-density + overlapping grid proposals; '
                  'connector/marker window reconstruction; joint same-colour identities. '
                  f'threshold=0.25, native resolution, scale policy={report["scale_policy"]}.', flush=True)
            prepared = ({name: marker_template_from_entry(image, entry, name, legend_box=list(legend))
                         for name, entry in zip(names, entries)} if self.use_prepared_templates else None)
            try:
                evidence = prepare_evidence(image, plot, specs, max_side=0, template_overrides=prepared,
                                            guide_policy=guide_policy,scale_policy=report['scale_policy'])
            except ValueError as error:
                if not str(error).startswith('No usable legend templates:'):
                    raise
                report.update(status='no_usable_templates', template_errors=[str(error)])
            if evidence is not None:
                evidence['priors'] = priors
                report['symbol_scale_calibration']=evidence.get('symbol_scale_calibration')
                report['marker_recovery_policy']=dict(
                    version='confirmed_T_multicentre_local_blend_v1',
                    blend_uncertainty=evidence.get('blend_uncertainty_version'),
                    blend_is_positive_evidence=False,legacy_minimum_missing_weight=.35,
                    alternative_centres='up to two nearby 2D image-response peaks; same shared size and final verifier',
                    triangle_nuisance='full T only with independent support; otherwise confidence-weighted; required-ink and interior checks remain')
                report['marker_recovery_policy']['center_score'] = dict(
                    version=PAPER_CENTER_VERSION,paper_weight=PAPER_CENTER_WEIGHT,
                    scope='ordinary colour filled-marker centre refinement and alternative peaks',
                    acceptance_threshold_changed=False)
                report['marker_recovery_policy']['window_physical'] = dict(
                    version=PAPER_WINDOW_VERSION,configuration=asdict(PaperWindowConfig()),
                    scope='confirmed filled colour windows; raw-paper contradiction, foreign hue waiver, soft unknown ink',
                    acceptance_threshold_changed=False,positive_support_changed=False)
                if report['symbol_scale_calibration']:
                    for sid,decision in report['symbol_scale_calibration']['series'].items():
                        print(f'[v46 colour shared scale] {sid}: {decision["scale"]:.3f}x; '
                              f'anchors={decision["anchor_count"]}; {decision["status"]}; grid/window locked.',flush=True)
                    (destination/'color_symbol_scale_v46.json').write_text(
                        json.dumps(_plain(report['symbol_scale_calibration']),indent=2),encoding='utf-8')
                report['initial_marker_ignore_mask'] = evidence['ignore_report']
                cv2.imwrite(str(destination/'initial_marker_ignore_mask.png'),
                            np.rint(255*evidence['ignore_mask']).astype(np.uint8))
                if 'guide_report' in evidence:
                    report['initial_marker_guide_model']=evidence['guide_report']
                    cv2.imwrite(str(destination/'initial_marker_guide_model.png'),
                                np.rint(255*(1-evidence['guide_darkness'])).astype(np.uint8))
                analysis = analyze_evidence(evidence)
                sx, sy = evidence['scale_x'], evidence['scale_y']
                ox, oy = evidence['source_center_offset']
                for point in analysis['candidates']:
                    point.update(x_px=point['x']/sx+ox, y_px=point['y']/sy+oy,
                                 diameter_source=point['diameter']/(.5*(sx+sy)))
                if guard_active:
                    before = select_candidates(analysis['candidates'], config=config,
                                               joint=True, line_gate=True)
                    guard = apply_guards(evidence, analysis['candidates'], priors, config.threshold)
                    report['weak_suppressed_points'] = weak_suppressed(
                        before['points'], analysis['candidates'], plot, legend)
                    guard['weak_suppressed_count'] = len(report['weak_suppressed_points'])
                    guard['before_active_count'] = len(before['points'])
                    report['triangle_errorbar_guard'] = guard
                selection = select_candidates(analysis['candidates'], config=config,
                                              joint=True, line_gate=True)
                # Final ROI guard also covers subpixel centre refinement near an
                # edge. An excluded legend never produces data points.
                for field in ('points', 'uncertain_points'):
                    keep = []
                    for point in selection[field]:
                        if _inside(point, plot) and not _inside(point, legend):
                            keep.append(point)
                        else:
                            selection['rejected'].append(dict(candidate_id=point['candidate_id'],
                                reason='final_center_outside_plot_or_in_legend'))
                    selection[field] = keep
                keys = ('id', 'label', 'rgb', 'diameter', 'source_center',
                        'marker_box', 'swatch_box', 'provenance', 'legend_shape_hint', 'legend_shape_evidence',
                        'symbol_scale','legend_diameter','symbol_scale_status','symbol_scale_version')
                report.update(**selection, status='completed',
                    candidates=analysis['candidates'],
                    templates=[{k: t[k] for k in keys if k in t} for t in evidence['templates']],
                    template_errors=evidence.get('template_errors', []),
                    scale_x=sx, scale_y=sy, timing=analysis['timing'],
                    proposal_diagnostics=analysis['proposal_diagnostics'])
                if report['template_errors']:
                    report['status'] = 'completed_partial_legend'
                if guard_active:
                    report['triangle_errorbar_guard']['after_active_count'] = len(report['points'])
                    draw_review(image, report['points'], report['weak_suppressed_points'], destination)
                    (destination/'triangle_errorbar_weak_suppressed_v46.json').write_text(
                        json.dumps(_plain(dict(points=report['weak_suppressed_points'],
                            coordinate_convention='x_px/y_px: source-image pixel centres',
                            step5_suppressed_eligible=False)), indent=2), encoding='utf-8')
                    print(f"[v46 triangle/error-bar] {report['triangle_errorbar_guard']['checked_candidates']} windows checked; "
                          f"{len(report['weak_suppressed_points'])} weak suppressed; guide={guide_policy}. "
                          'Weak candidates are review-only, not strong Step-5 input.', flush=True)
            self._handoff(env, entries, names, plot, evidence, report)
            if self.enable_tentative:
                # Reuse the working geometry, not the unscaled prepared legend.
                # Keep explicit None entries for line-only/failed identities.
                templates = dict(prepared or {})
                templates.update({t['id']:t for t in (evidence or {}).get('templates',[])})
                self._tentative_handoff(env, legend_runtime, names, plot, legend, templates, report)
            self.active = True
            legend_runtime.diagnostics.update(
                scope='observed legend + supported composition feeding hybrid v2 and tentative Step 5',
                downstream_marker_version=MARKER_VERSION)
            legend_runtime._save()
        except Exception as error:
            report.update(status='error', error=f'{type(error).__name__}: {error}')
            raise RuntimeError('v46 hybrid v2 marker detection failed; no silent legacy fallback') from error
        finally:
            report['seconds'] = perf_counter()-started
            (destination/'color_marker_hybrid_v46.json').write_text(
                json.dumps(_plain(report), indent=2), encoding='utf-8')
        self._draw_templates(image, evidence, destination)
        print(f"[v46 color markers] {len(report['points'])} active detections, "
              f"{len(report['uncertain_points'])} uncertain, "
              f"{len(report['template_errors'])} template errors; "
              f"{report['seconds']:.2f}s. -> color_marker_hybrid_v46.json", flush=True)

    @staticmethod
    def _tentative_handoff(env, legend_runtime, names, plot, legend, templates, report):
        """Shared v46 colour evidence -> retained paths -> suppressed candidates.

        Active detections and geometric support knots never seed this review
        pool. The native masks are separate from the display-only assign.npy.
        """
        from color_tentative_v46 import build_tentative_candidates
        from color_step5_export_v46 import export_step5_inputs

        tick = perf_counter()
        image, entries = env['img'], legend_runtime.entries
        furniture, _, furniture_report = env['furniture_mask'](image, inclusive(plot))
        masks, soft, info = legend_runtime.colour_masks(env['colour_masks'], image,
            inclusive(plot), inclusive(legend), metric='tube', spatial=False, furniture=furniture)
        if len(info) != len(names) or set(masks) != set(range(len(names))):
            raise RuntimeError('Native colour fields lost the prepared legend series identities')
        x0, y0, x1, y1 = plot
        allowed = ~np.asarray(furniture[y0:y1, x0:x1], bool)
        a, b = max(legend[0], x0)-x0, max(legend[1], y0)-y0
        c, d = min(legend[2], x1)-x0, min(legend[3], y1)-y0
        if a < c and b < d:
            allowed[b:d, a:c] = False
        series, own_masks = [], {}
        for i, (name, entry) in enumerate(zip(names, entries)):
            if not np.allclose(info[i]['rgb'], entry['rgb'], atol=1.):
                raise RuntimeError('Native colour field RGB no longer corresponds to its legend slot')
            # Native ownership is exclusive and ties favour one slot. Identical
            # colours share that observed ink, but retain separate shape IDs.
            peers = [j for j, other in enumerate(entries)
                     if np.allclose(other['rgb'], entry['rgb'], atol=1.)]
            own = np.logical_or.reduce([masks[j][y0:y1, x0:x1] for j in peers]) & allowed
            field = np.asarray(soft[i][y0:y1, x0:x1], np.float32) * allowed
            template = templates.get(name)
            diameter = float(template['diameter']) if template is not None else max(
                2., float(entry.get('report', {}).get('diameter', max(entry['mask'].shape))))
            spec = dict(id=name, rgb=entry['rgb'], own=own, soft=field, diameter=diameter,
                active_points=env['all_detections'][name], marker_allowed=template is not None)
            if template is not None:
                spec['shape_info'], spec['shape_fields'] = shape_fields_from_template(template)
            own_masks[name] = own
            series.append(spec)
        print('[v46 tentative] Shared v46 colour masks; directional retained paths and '
              'image/shape hypotheses. Only strong image evidence enters suppressed.', flush=True)
        tentative = build_tentative_candidates(image, plot, series, legend_boxes=[legend],
            valid=allowed, native_helpers={k: env[k] for k in ('_edge_norm_density', 'fragment_profile')})
        if report.get('weak_suppressed_points'):
            report['triangle_errorbar_guard']['tentative_proposals_kept_weak'] = protect_weak_candidates(
                tentative, report['weak_suppressed_points'])
        destination = Path(env['OUT_DIR'])
        (destination/'color_tentative_v46.json').write_text(
            json.dumps(_plain(tentative), indent=2), encoding='utf-8')
        payload = export_step5_inputs(destination, image, plot, legend, entries, names,
            templates, env['all_detections'], tentative, own_masks, env['path_to_segments'])
        report['tentative'] = dict(version=tentative['version'],
            tentative_candidates=len(tentative['candidates']),
            suppressed_candidates=payload['counts']['tentative_suppressed'],
            excluded_non_strong=payload['counts']['excluded_non_strong'],
            active_points_changed=False,
            mathematical_knots_exported=False, seconds=perf_counter()-tick,
            furniture=furniture_report, native_colour_info=info,
            payload_file='step5_inputs.json', payload_version=payload['version'])
        print(f"[v46 tentative] {payload['counts']['tentative_suppressed']} strong suppressed "
              f"from {len(tentative['candidates'])} tentative candidates; "
              f"{report['tentative']['seconds']:.2f}s including masks/paths/export. "
              '-> step5_inputs.json (not auto-accepted)', flush=True)

    @staticmethod
    def _handoff(env, entries, names, plot, evidence, report):
        """Populate the existing writers without invoking the old point detector."""
        image = env['img']
        detections = {name: [] for name in names}
        for point in report['points']:
            # A valid subpixel edge centre can round outside a half-open ROI.
            x, y = _export_pixel(point, plot, report['legend_box'])
            detections[point['series_id']].append(dict(x=x, y=y,
                fitness=float(point['score']), px=1, source=report.get('version', MARKER_VERSION),
                x_px=float(point['x_px']), y_px=float(point['y_px']),
                candidate_id=point['candidate_id']))
        for points in detections.values():
            points.sort(key=lambda p: (p['x'], p['y']))
        env.update(_names=names, _rgbs=[e['rgb'] for e in entries],
            all_detections=detections, all_tcaps_out={n: [] for n in names},
            all_results={n: ([], 0.) for n in names}, _V45_STEP5=None,
            COLORS=[dict(name=n, mean_rgb=list(e['rgb']), swatch_rgb=list(e['rgb']),
                         swatch_id=e['id'], px_count=0) for n, e in zip(names, entries)])
        env['_cfg'].curve_names = names.copy()
        env['_cfg'].stem_x = []
        env['_cfg'].x_values = []
        digitizer = env['_dig']
        digitizer.palette = [tuple(e['rgb']) for e in entries]
        digitizer.is_sink = np.zeros(len(entries), bool)
        assign = np.full(image.shape[:2], -1, np.int16)
        if evidence is not None:
            membership = evidence['membership']
            winners = np.argmax(membership, axis=0)
            strength = np.max(membership, axis=0)
            indices = np.array([names.index(t['id']) for t in evidence['templates']], np.int16)
            local = indices[winners]
            local[(strength < .15) | ~evidence['valid']] = -1
            assign[plot[1]:plot[3], plot[0]:plot[2]] = local
        digitizer.assign = assign
        # assign.npy remains useful to existing downstream correction tooling.
        np.save(Path(env['OUT_DIR'])/'assign.npy', assign)

    @staticmethod
    def _draw_templates(image, evidence, destination):
        """Actual source swatches and retained continuous templates, not ideal glyphs."""
        templates = evidence['templates'] if evidence is not None else []
        canvas = np.full((70+110*max(1, len(templates)), 800, 3), 255, np.uint8)
        cv2.putText(canvas, 'v46: source swatch | detection template | identity',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (30,30,30), 1, cv2.LINE_AA)
        for i, template in enumerate(templates):
            a,b,c,d = map(int, template['swatch_box'])
            soft = np.asarray(template['soft'], np.float32)
            bgr = np.asarray(template['rgb'][::-1], float)
            model = np.clip(255*(1-soft[...,None])+bgr*soft[...,None], 0,255).astype(np.uint8)
            source = image[b:d,a:c]
            # Padding differs, but source and retained ink must have the same
            # display magnification; independent resizing implies a fake scale change.
            scale = min(245/max(source.shape[1], model.shape[1]),
                        90/max(source.shape[0], model.shape[0]))
            for x, patch in ((15,source), (285,model)):
                patch = cv2.resize(patch, (max(1,round(patch.shape[1]*scale)),
                                         max(1,round(patch.shape[0]*scale))),
                                   interpolation=cv2.INTER_NEAREST)
                canvas[55+i*110:55+i*110+patch.shape[0], x:x+patch.shape[1]] = patch
            cv2.putText(canvas, f"{template['id']} d={template['diameter']:.1f}px x{template.get('symbol_scale',1.):.3f}",
                (550,85+i*110), cv2.FONT_HERSHEY_SIMPLEX, .55, (30,30,30), 1, cv2.LINE_AA)
        if not templates:
            cv2.putText(canvas, 'No usable observed templates; no legacy points substituted.',
                        (15,100), cv2.FONT_HERSHEY_SIMPLEX, .55, (30,30,170), 1, cv2.LINE_AA)
        cv2.imwrite(str(destination/'color_marker_templates_v46.png'), canvas)

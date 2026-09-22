"""B&W v46: legend pixels -> grid votes -> asymmetric full-window validation.

No trained point detector is imported. Shared pixel helpers are imported from
the standalone v46 analysis library, which has no CLI side effects.
All public boxes use exclusive right/bottom edges, like partial_swatch_detector.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import os
import time

import cv2
import numpy as np

import partial_swatch_detector as D
from bw_marker_selection import select_verified
from bw_suppressed_v46 import WINDOW_METRICS, build_suppressed, encode_marker_mask
from bw_compute_backend import resolve_compute_backends, is_cuda_runtime_error
from occlusion_aware_window_verifier import verify_marker_window, verify_marker_windows_many

VERSION = 'bw-grid-v46'
PRODUCTION_PROFILE = 'two-stage-shared-symbol-scale-geometry-fill-v3'
LEGACY_PROPOSAL_SCALES = (.90, .94, .98, 1., 1.04, 1.08, 1.12)
# Compatibility constant for old explicit experiments; no longer a set of
# production proposals. The new production profile calibrates once per swatch.
PRODUCTION_PROPOSAL_SCALES = LEGACY_PROPOSAL_SCALES
CLASSES = ['filled_circle', 'open_circle', 'filled_square', 'open_square',
           'open_triangle', 'open_inv_triangle', 'filled_triangle',
           'filled_inv_triangle', 'open_rhombus', 'filled_rhombus',
           'x_marker', 'plus_marker', 'unknown_marker']


def production_detection_options():
    """Shared GUI/CLI B&W profile; standalone experiments may override options."""
    return dict(scale_policy='shared_symbol', window_scale_reference='legend',
                grid_identity_competition=True, geometry_first=True)


def _legend_functions():
    import chart_analysis_v46 as analysis
    names = ('_to_lab', '_lab_image', '_tube', 'find_legend_swatches',
             'swatch_ink_colour', 'adaptive_tolerance', 'marker_template')
    return {name: getattr(analysis, name) for name in names}


def _box(box, image):
    if box is None or len(box) != 4:
        raise ValueError('v46 requires a plot area and a legend area (or explicit swatches)')
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = image.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f'Invalid image box: {box}; image size {w}x{h}')
    return x0, y0, x1, y1


def _tight(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Empty legend marker')
    return mask[y.min():y.max()+1, x.min():x.max()+1]


def _raw_marker(image, box):
    """Keep observed pixels; trim only lateral thin-line tails.

    The v45 contiguous-tall-column crop can cut a triangle or split an open
    glyph. This alternative keeps the whole span of tall columns, then expands
    across adjacent columns containing more than a thin horizontal stroke.
    No convex-hull pixels are written into the returned template.
    """
    x0, y0, x1, y1 = box
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    ink = gray < 210
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    if n <= 1:
        raise ValueError('No ink in legend swatch')
    ink = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    component = _tight(ink)
    if component.shape[1] <= 1.6*component.shape[0]:
        # Compact legend glyph without long lateral tails: there is no line
        # to strip. In particular, preserve the triangle's entire base.
        return component
    spans = np.array([np.ptp(np.flatnonzero(c))+1 if c.any() else 0 for c in ink.T])
    height = int(spans.max())
    if height < 4:
        raise ValueError('Line-only legend swatch')
    cols = np.flatnonzero(spans >= max(4, .5*height))
    left, right = int(cols[0]), int(cols[-1])
    # Expand to triangle bases / curved extrema but not long connecting tails.
    thin = max(2, int(round(.22*height)))
    # A thick legend line can be 6px high beside an 18px square. Estimate
    # its thickness from both outer tails, instead of assuming a 1-4px line.
    tail = max(1,len(spans)//5)
    outer = np.r_[spans[:tail],spans[-tail:]]
    outer = outer[(outer>0) & (outer < .5*height)]
    if len(outer):
        thin = max(thin,int(np.median(outer)))
    while left > 0 and spans[left-1] > thin:
        left -= 1
    while right+1 < len(spans) and spans[right+1] > thin:
        right += 1
    return _tight(ink[:, left:right+1])


def _shape(mask, nuisance=None, return_report=False):
    """Observed shape hint; uncertain evidence never defaults to a circle."""
    from bw_legend_shape import classify_observed_shape
    name, fill, report = classify_observed_shape(mask, nuisance)
    return (name, fill, report) if return_report else (name, fill)


def _adapt(image, box, name, raw):
    h, w = raw.shape
    diameter = float(max(h,w))
    size = 2*max(3, math.ceil(.58*diameter))+1
    mask = np.zeros((size,size),bool)
    top, left = (size-h)//2, (size-w)//2
    mask[top:top+h,left:left+w] = raw
    soft = mask.astype(np.float32)
    _, fill = _shape(raw)
    edge = cv2.Canny(mask.astype(np.uint8)*255,40,100)>0
    ink = D.estimate_ink_model(image[box[1]:box[3],box[0]:box[2]])
    return D.SwatchTemplate(name,box,((box[0]+box[2]-1)/2,(box[1]+box[3]-1)/2),
        diameter,'filled' if fill>=.52 else 'open',fill,soft.copy(),mask.copy(),
        np.zeros_like(mask),np.ones_like(soft),soft,mask,edge,D._edge_orientation(soft),ink)


def extract_templates(image, legend_area=None, swatches=None, known_classes=None):
    """Extract complete observed B&W glyphs with explicit pixel uncertainty.

    The full legend region may contain several graphical columns. A clipped
    line fragment is not an acceptable marker, even if it matches plot ink.
    """
    from bw_legend_v46 import extract_legend_models
    return extract_legend_models(image, legend_area, swatches, known_classes)


def _legacy_extract_templates(image, legend_area=None, swatches=None, known_classes=None):
    """v45 box/ink extraction plus a conservative observed-pixel fallback.

    Retained only as an explicit regression/ablation reference. The production
    entry above never silently falls back to this potentially clipped model.
    """
    v45 = _legend_functions()
    explicit = swatches is not None
    if not explicit:
        lb = _box(legend_area,image)
        boxes = v45['find_legend_swatches'](image,(lb[0],lb[1],lb[2]-1,lb[3]-1))
        # Wrapped label lines and plot strokes can produce extra row bands.
        # Retain repeated graphical-column starts, not arbitrary text rows.
        # Keep a single-entry legend usable when there is no repeated column.
        heights = [d-b+1 for a,b,c,d in boxes if d-b+1>=4]
        tolerance = max(6,1.2*float(np.median(heights))) if heights else 6
        groups=[]
        for b in sorted(boxes,key=lambda b:b[0]):
            if groups and abs(b[0]-np.median([q[0] for q in groups[-1]])) <= tolerance:
                groups[-1].append(b)
            else:
                groups.append([b])
        repeated = [g for g in groups if sum(q[3]-q[1]+1>=4 for q in g)>=2]
        if repeated:
            boxes = sorted([b for g in repeated for b in g],key=lambda b:(b[1],b[0]))
        # Scan can stop part way along a long swatch line; add room for marker.
        swatches = [(None,(max(lb[0],a-2),max(lb[1],b-2),
                           min(lb[2],c+max(3,d-b)+1),min(lb[3],d+3))) for a,b,c,d in boxes]
    if not swatches:
        raise ValueError('v46 found no legend swatches; select the legend graphical column or supply swatch boxes')
    templates=[]; report=[]
    for supplied, box in swatches:
        box = _box(box,image)
        inc = box[0],box[1],box[2]-1,box[3]-1
        rgb, core, _ = v45['swatch_ink_colour'](image,inc)
        if rgb is None:
            raise ValueError(f'Empty swatch at {box}')
        tol = v45['adaptive_tolerance'](rgb,core,[])
        original = v45['marker_template'](image,inc,rgb,[],tol)
        try:
            observed = _raw_marker(image,box)
        except ValueError:
            if original is None:
                if explicit:
                    raise
                # The v45 scan deliberately returns line-only rows too.
                continue
            # Pale achromatic ink may be valid under v45's ink-relative
            # threshold but absent under the darker fallback threshold.
            observed = original
        # A large missing extent means the v45 crop discarded glyph geometry.
        plausible_observed = observed.shape[1] <= 1.6*observed.shape[0]
        truncated = original is None or (plausible_observed and
                    min(original.shape[i]/observed.shape[i] for i in (0,1)) < .80)
        mask = observed if truncated else original
        cls, _ = _shape(mask)
        if supplied:
            cls = supplied.replace('diamond','rhombus')
        if cls not in CLASSES:
            raise ValueError(f'Unsupported marker class: {cls}')
        if known_classes and cls not in known_classes:
            report.append({'box':box,'class_name':cls,'excluded_by_class_filter':True})
            continue
        if any(t.name == cls for t in templates):
            raise ValueError(f'Multiple legend entries classified as {cls}; supply corrected swatch classes/boxes. Refusing to silently merge series.')
        templates.append(_adapt(image,box,cls,mask))
        report.append({'box':box,'class_name':cls,'classification':'supplied' if supplied else 'geometric',
                       'extractor':'observed_extent_fallback' if truncated else 'v45_marker_template',
                       'v45_shape':None if original is None else list(original.shape),
                       'template_shape':list(mask.shape)})
    if not templates:
        raise ValueError('No legend templates remain after the class filter')
    return templates,report


def legend_labels_from_swatches(image, legend_area, entries, ocr=None):
    """Anchor OCR to retained swatches, never independently re-number text rows.

    Wrapped text is included up to the next marker row. A text-only row cannot
    shift every later class/label association as in the legacy row-index mapper.
    """
    lb = _box(legend_area,image)
    if ocr is None:
        try:
            import ocr_bridge_v46 as pytesseract
        except ImportError:
            return {}
        def ocr(crop):
            return pytesseract.image_to_string(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),config='--psm 6')
    entries=[e for e in entries if not e.get('excluded_by_class_filter')]
    if not entries:
        return {}
    tolerance=max(6.,.95*float(np.median([e['box'][3]-e['box'][1] for e in entries])))
    columns=[]
    for entry in sorted(entries,key=lambda e:e['box'][0]):
        if columns and abs(entry['box'][0]-np.median([e['box'][0] for e in columns[-1]]))<=tolerance:
            columns[-1].append(entry)
        else:
            columns.append([entry])
    ordered=[]
    for column_index,column in enumerate(columns):
        column=sorted(column,key=lambda e:e['box'][1])
        text_right=(min(e['box'][0] for e in columns[column_index+1])-3
                    if column_index+1<len(columns) else lb[2])
        for i,entry in enumerate(column):
            bottom=(column[i+1]['box'][1]-2 if i+1<len(column)
                    else min(lb[3],entry['box'][3]+4))
            ordered.append((entry,bottom,text_right))
    labels={}
    for entry,bottom,text_right in ordered:
        a,b,c,d = _box(entry['box'],image)
        gray=cv2.cvtColor(image[b:d,a:text_right],cv2.COLOR_BGR2GRAY)
        n,_,stats,_=cv2.connectedComponentsWithStats((gray<235).astype(np.uint8),8)
        glyphs=[s for s in stats[1:] if s[cv2.CC_STAT_LEFT] <= max(4,.35*(d-b))
                and s[cv2.CC_STAT_HEIGHT] >= .4*(d-b)]
        text_left=c
        if glyphs:
            glyph=max(glyphs,key=lambda s:s[cv2.CC_STAT_AREA])
            text_left=a+int(glyph[cv2.CC_STAT_LEFT]+glyph[cv2.CC_STAT_WIDTH])+2
        top=max(lb[1],b-2)
        if text_left>=text_right or bottom<=top:
            continue
        crop=image[top:bottom,text_left:text_right]
        try:
            label=' '.join(str(ocr(crop)).split())
        except Exception:
            label=''
        if label:
            labels[entry.get('swatch_id') or entry['class_name']]=label
    return labels


def detect_points(image, plot_area, legend_area=None, known_classes=None,
                  swatches=None, grid_fraction=.5, grid_overlap=.5, log_fn=print,
                  min_required_recall=None, proposal_scales=None,
                  scale_policy='per_candidate', refinement_backend=None,
                  gpu_batch_size=512, grid_backend=None, window_backend=None,
                  grid_identity_competition=False, window_scale_reference='proposal',
                  fill_identity=False, observed_body_scale_range=None,
                  prepared_templates=None, window_occlusion_mask=None,
                  window_image=None, window_uncertainty_mask=None,
                  window_search_scales=None, geometry_first=False):
    """Detect with the existing per-candidate scale/aspect search by default.

    ``conservative`` locks each swatch to one explicitly approved size (or 1x).
    ``fixed_1x`` disables calibration and locks every swatch to its original size.
    ``shared_symbol`` estimates one 0.70-1.30 scale per swatch from image-only
    gray-pattern consensus, then locks both stages and aspect ratio to that
    geometry. Weak evidence stays labelled weak; no anchors means fixed 1x.
    ``grid_identity_competition`` opts into common-cell legend discrimination
    before vote clustering and full-window verification (filled-body pilot).
    ``window_scale_reference='legend'`` searches the original legend size in
    full windows, independently of the proposal size (no compounded scaling).
    ``proposal_scales`` can also map every swatch ID to its own scale sequence;
    this controls grid proposals only, not the independent window scale search.
    ``window_search_scales`` explicitly configures both CPU and CUDA full
    windows for per-candidate fitting. None retains the older window range.
    Shared-symbol fitting supplies its per-swatch locks internally instead.
    The GUI and shared B&W CLI pass ``production_detection_options()``; direct
    experiment calls retain their previous defaults unless explicitly changed.
    ``fill_identity`` tests separate soft-interior evidence at grid and window
    stages; it is experimental and does not change the GUI production default.
    ``geometry_first`` uses filled envelopes for patterned-marker proposals,
    verifies fill on original grayscale, and decomposes open circles from
    connecting legend lines. Enabled automatically by the native BW GUI/CLI
    profile when the legend contains a patterned marker or open circle.
    Prepared colour-group templates retain their separately reviewed route.
    ``refinement_backend='cuda'`` enables the CUDA center, grid and window paths.
    Explicit ``grid_backend`` / ``window_backend`` overrides allow ablations.
    When omitted, BW_V46_REFINEMENT_BACKEND selects it (default: auto).
    Auto uses a working CUDA runtime, otherwise the original CPU path. Explicit
    cuda remains strict; explicit cpu never probes or imports optional PyTorch.
    ``prepared_templates`` optionally reuses an aligned (templates, reports)
    pair extracted by the native BW legend module, preserving swatch identities.
    ``window_occlusion_mask`` is image-aligned other-colour visibility [0,1];
    despite its historical name it also reaches candidate/centre/identity checks.
    Nonzero masks use CPU candidate, centre and window scoring and cannot supply
    positive marker ink. Grid voting may remain CUDA; centre overrides are logged.
    ``window_image`` optionally separates confirmed ink from grid proposals.
    ``window_uncertainty_mask`` discounts ambiguous-colour loss only; it never
    supplies positive ink or completely exempts a missing required pixel.
    """
    started = time.perf_counter()
    if window_image is None:
        window_image=image
    elif window_image.shape!=image.shape or window_image.dtype!=image.dtype:
        raise ValueError('window_image must have the same shape and dtype as image')
    if window_uncertainty_mask is not None:
        window_uncertainty_mask=np.asarray(window_uncertainty_mask,np.float32)
        if (window_uncertainty_mask.shape!=image.shape[:2] or not np.isfinite(window_uncertainty_mask).all()
                or np.any((window_uncertainty_mask<0)|(window_uncertainty_mask>1))):
            raise ValueError('window_uncertainty_mask must be finite, image-aligned and in [0,1]')
        if window_uncertainty_mask.any():
            if window_backend not in (None,'auto','cpu'):
                raise ValueError('Colour uncertainty requires window_backend=cpu/auto')
            window_backend='cpu'
        else:
            window_uncertainty_mask=None
    if window_occlusion_mask is not None:
        window_occlusion_mask=np.asarray(window_occlusion_mask,np.float32)
        if (window_occlusion_mask.shape!=image.shape[:2] or not np.isfinite(window_occlusion_mask).all()
                or np.any((window_occlusion_mask<0)|(window_occlusion_mask>1))):
            raise ValueError('window_occlusion_mask must be finite, image-aligned and in [0,1]')
        if window_occlusion_mask.any():
            if window_backend not in (None,'auto','cpu'):
                raise ValueError('Explicit colour occlusion requires window_backend=cpu/auto')
            window_backend='cpu'
            log_fn('[v46 colour adapter] Three-state candidate/centre/window verification; CPU visibility scoring, grid votes unchanged')
        else:
            window_occlusion_mask=None
    if not isinstance(gpu_batch_size, int) or gpu_batch_size <= 0:
        raise ValueError('gpu_batch_size must be a positive integer')
    pa = _box(plot_area,image)
    if not (0 < grid_fraction <= 1 and 0 <= grid_overlap < 1):
        raise ValueError('grid_fraction must be in (0,1]; grid_overlap in [0,1)')
    if min_required_recall is not None and not 0 <= min_required_recall <= 1:
        raise ValueError('v46 confidence floor must be in [0,1] (required-ink recall)')
    if scale_policy not in {'conservative','fixed_1x','per_candidate','shared_symbol'}:
        raise ValueError('Unknown v46 scale policy')
    if proposal_scales is not None and scale_policy != 'per_candidate':
        raise ValueError('proposal_scales requires per_candidate policy')
    if window_search_scales is not None:
        if scale_policy != 'per_candidate':
            raise ValueError('window_search_scales requires per_candidate policy')
        window_search_scales=tuple(float(s) for s in window_search_scales)
        if not window_search_scales or any(not np.isfinite(s) or s<=0 for s in window_search_scales):
            raise ValueError('window_search_scales must contain positive finite scales')
    if window_scale_reference not in {'proposal', 'legend'}:
        raise ValueError('window_scale_reference must be proposal or legend')
    if scale_policy=='shared_symbol' and window_scale_reference!='legend':
        raise ValueError('shared_symbol requires window_scale_reference=legend')
    # Older fixed/conservative policies verify a pre-scaled raster at 1x.
    # Shared-symbol fitting locks one original-legend-referenced scale instead.
    use_legend_windows = window_scale_reference == 'legend' and scale_policy in {'per_candidate','shared_symbol'}
    fixed_window_geometry = scale_policy in {'conservative','fixed_1x'}
    effective_window_reference = 'legend' if use_legend_windows else 'proposal'
    log_fn(f'[v46 matching] grid identity competition={bool(grid_identity_competition)}; '
           f'full-window scale reference={effective_window_reference}')
    if window_search_scales is not None:
        log_fn(f'[v46 scale search] per-marker full-window scales={window_search_scales}; '
               'aspect search=(1.0, 0.94, 1.06); no per-symbol scale lock')
    compute_plan = resolve_compute_backends(refinement_backend, grid_backend, window_backend, log_fn=log_fn)
    compute_requested, compute_selected = compute_plan['requested'], compute_plan['selected']
    refinement_backend, grid_backend, window_backend = (
        compute_selected[name] for name in ('refinement', 'grid', 'window'))
    active_refinement, active_grid = refinement_backend, grid_backend
    compute_fallbacks = []
    if window_occlusion_mask is not None:
        if active_refinement != 'cpu':
            compute_fallbacks.append(dict(stage='refinement',**{'from':active_refinement,'to':'cpu'},
                reason='Three-state visibility requires CPU candidate/refinement kernels'))
        refinement_backend = active_refinement = 'cpu'
        compute_selected = {**compute_selected,'refinement':'cpu'}
    grid_fallback_reason = None
    candidate_backend='cpu' if window_occlusion_mask is not None else grid_backend
    log_fn(f'[v46 compute] grid voting / center aggregation: {grid_backend}; candidate scoring: {candidate_backend}; center refinement: {refinement_backend}; full-window rank screening: {window_backend}; final window decisions / selection: CPU')
    backend_setup_seconds = time.perf_counter()-started
    stage_started = time.perf_counter()
    if prepared_templates is None:
        templates,report = extract_templates(image,legend_area,swatches,known_classes)
    else:
        from copy import deepcopy
        templates,report=deepcopy(prepared_templates)
        if not templates or len(templates)!=len(report):
            raise ValueError('prepared_templates requires nonempty aligned BW templates and reports')
        for t,r in zip(templates,report):
            if t.matching_profile!='bw_v46_uncertain' or not t.ink.achromatic or t.key!=r.get('swatch_id'):
                raise ValueError('Prepared templates must preserve the native BW profile and swatch IDs')
    legend_seconds = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    exclusions = [tuple(legend_area)] if legend_area is not None else [t.swatch_box for t in templates]
    geometry = None
    if geometry_first and prepared_templates is None:
        from bw_geometry_first_v46 import GeometryFirst
        # Reports include failed/filtered entries so a missing model cannot
        # shift the shape/fill metadata of every subsequent series.
        report_by_id={r['swatch_id']:r for r in report}
        from bw_composed_legend_v46 import compose_compact_unknowns
        templates,composed_reports=compose_compact_unknowns(
            image,legend_area,templates,[report_by_id[t.key] for t in templates])
        transformed={r['swatch_id']:r for r in composed_reports}
        report=[transformed.get(r['swatch_id'],r) for r in report]
        attempted=[r for r in composed_reports if r.get('compact_composition_attempted')]
        if attempted:
            adopted=sum(r.get('template_policy')=='pure_model_marker' for r in attempted)
            log_fn(f'[v46 compact legend] line + marker fits: {len(attempted)}; '
                   f'supported pure markers: {adopted}; observed fallbacks: {len(attempted)-adopted}')
        report_by_id={r['swatch_id']:r for r in report}
        controller = GeometryFirst(image,templates,[report_by_id[t.key] for t in templates],pa,legend_area,exclusions)
        if controller.enabled:
            geometry=controller
            templates=geometry.templates
            transformed={r['swatch_id']:r for r in geometry.reports}
            report=[transformed.get(r['swatch_id'],r) for r in report]
            log_fn('[v46 geometry/fill] filled silhouette proposals; original grayscale fill verification; line + open-circle decomposition')
            # Separate proposal rasters have source-only tone checks. This
            # explicit CPU window path does not alter GPU grid computation.
            if compute_requested['window']=='cuda':
                raise ValueError('Geometry-first source-tone verification needs window_backend=cpu/auto; CUDA grid/refinement remain available')
            window_backend='cpu'
            compute_selected['window']='cpu'
            log_fn('[v46 geometry/fill compute] CPU full windows for separate geometry and source-tone rasters; grid/refinement backend unchanged')
    by_key={t.key:t for t in templates}
    if len(by_key)!=len(templates):
        raise ValueError('Each retained legend swatch needs a distinct identity')
    from collections.abc import Mapping
    if isinstance(proposal_scales, Mapping):
        if set(proposal_scales) != set(by_key):
            raise ValueError('proposal_scales mapping must contain exactly the retained swatch IDs')
    source_indices={r.get('swatch_id'):r.get('series_index') for r in report}
    series_indices={t.key:((source_indices.get(t.key) if source_indices.get(t.key) is not None else i)
                         if t.swatch_id else CLASSES.index(t.name))
                    for i,t in enumerate(templates)}
    legend_failures=[r for r in report if r.get('extraction_status')=='failed']
    for entry in legend_failures:
        log_fn(f'[v46 legend WARNING] {entry["swatch_id"]}: unresolved template; '
               f'{entry["extraction_error"]}. Identity retained; no points invented.')
    exclusions = [tuple(legend_area)] if legend_area is not None else [t.swatch_box for t in templates]
    identity_competitor = None
    geometry_identity_competitor = None
    if grid_identity_competition and geometry is None:
        from bw_grid_identity_v46 import GridIdentityCompetition
        identity_competitor = GridIdentityCompetition(image,templates,pa,exclusions,
            boundary_priority=True,
            **({'occlusion_mask':window_occlusion_mask} if window_occlusion_mask is not None else {}),
            **({'observed_scale_range':observed_body_scale_range} if observed_body_scale_range else {}))
    elif grid_identity_competition and geometry is not None:
        # Geometry-first proposals must not match hatch phase in small cells.
        # But full-window opaque candidates still need shape competition on
        # the SAME source body, including symmetric boundary evidence.
        from bw_grid_identity_v46 import GridIdentityCompetition
        opaque=[t for t in templates if geometry.descriptors[t.key].get('opaque_body_supported') or
                geometry.descriptors[t.key]['style']=='solid']
        if len(opaque)>1:
            geometry_identity_competitor=GridIdentityCompetition(image,opaque,pa,exclusions,
                silhouette_models=True,body_opening_fraction=.30,boundary_priority=True,
                **({'observed_scale_range':observed_body_scale_range} if observed_body_scale_range else {}))
    fill_competitor = None
    if fill_identity and geometry is None:
        from bw_fill_identity_v46 import FillIdentityCompetition
        fill_competitor = FillIdentityCompetition(image, templates, pa, exclusions, base=identity_competitor)
        identity_competitor = fill_competitor
    if scale_policy == 'shared_symbol':
        from bw_shared_scale_v46 import calibrate_swatch_scales
        calibration = (geometry.calibrate(log_fn) if geometry is not None else
                       calibrate_swatch_scales(image,templates,pa,ignore_regions=exclusions,log_fn=log_fn,
                                              occlusion_mask=window_occlusion_mask))
    elif scale_policy == 'conservative':
        from bw_scale_calibration import calibrate_swatch_scales
        calibration = calibrate_swatch_scales(image,templates,pa,ignore_regions=exclusions)
    else:
        calibration = {t.key:{'status':('unchanged' if scale_policy=='fixed_1x'
                                        else 'automatic_per_candidate'),
                               'scale':1.,'reason':scale_policy,'anchors':[]}
                       for t in templates}
    approved_scales={}
    for t in templates:
        decision=calibration.setdefault(t.key,{'status':'insufficient_evidence',
                                               'scale':1.,'reason':'no_calibration_report','anchors':[]})
        # The older conservative policy requires explicit approval. The reviewed
        # shared-symbol policy uses a working common scale even with weak
        # consensus, retaining that uncertainty in the exported diagnostics.
        apply_scale=scale_policy=='shared_symbol' or decision.get('status')=='approved'
        applied=float(decision.get('scale',1.)) if apply_scale else 1.
        if not np.isfinite(applied) or applied<=0:
            raise ValueError(f'Invalid approved scale for {t.name}: {applied}')
        approved_scales[t.key]=applied
        # Free fitting has no single applied class scale. Individual candidate
        # proposal/window scales below are authoritative in that policy.
        decision['applied_scale']=None if scale_policy=='per_candidate' else applied
        size_log=('automatic per-candidate scale/aspect' if scale_policy=='per_candidate'
                  else f'applied={applied:g}')
        log_fn(f'[v46 scale] {t.name}: {decision["status"]}; {size_log}')
    for entry in report:
        name=entry.get('swatch_id') or entry['class_name']
        if name in calibration:
            entry['scale_status']=calibration[name]['status']
            entry['applied_scale']=calibration[name]['applied_scale']
            entry['series_index']=series_indices[name]
    # Preserve unmodified image ink: do not delete errorbar strokes through markers.
    scale_policy_seconds = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    results=[]
    preprocessing_cache = {}
    for t in templates:
        proposal_image=geometry.image_for(t) if geometry is not None else image
        if scale_policy == 'per_candidate':
            configured = proposal_scales[t.key] if isinstance(proposal_scales, Mapping) else proposal_scales
            scales = configured if configured is not None else (
                (1.,1.25,1.5) if t.marker_kind == 'open' else (1.,))
        else:
            # One raster model per swatch, shared by grid and full-window checks.
            scales = (approved_scales[t.key],)
        scales = sorted(set(float(s) for s in scales))
        if not scales or any(not np.isfinite(s) or s<=0 for s in scales):
            raise ValueError('proposal_scales must contain positive finite scales')
        for scale in scales:
            variant = D.scale_swatch_template(t,scale)
            log_fn(f'[v46] {t.name}: diameter={variant.diameter:g}; proposal-scale={scale:g}; grid={grid_fraction:g}; overlap={grid_overlap:g}')
            compute_options = ({'refinement_backend': active_refinement, 'gpu_batch_size': gpu_batch_size}
                               if active_refinement != 'cpu' else {})
            # These do not affect accepted markers. The baseline was only an
            # experiment comparison; the immutable, call-scoped image cache
            # also avoids repeated preprocessing on the CPU path.
            compute_options.update(compute_baseline=False, preprocessing_cache=preprocessing_cache)
            if window_occlusion_mask is not None:
                compute_options['occlusion_mask'] = window_occlusion_mask
            if identity_competitor is not None:
                compute_options['grid_identity_competitor'] = identity_competitor
            if active_grid != 'cpu':
                compute_options.update(grid_backend=active_grid, gpu_batch_size=gpu_batch_size)
            try:
                detected = D.detect_template(proposal_image,variant,pa,grid_fraction=grid_fraction,
                    grid_overlap=grid_overlap,ignore_regions=exclusions,defer_nms=True, **compute_options)
            except Exception as error:
                # Only auto-selected CUDA can fail over. Explicit CUDA is a
                # strict diagnostic/benchmark choice and must never masquerade
                # as a successful CPU measurement. Do not hide algorithm bugs.
                cuda_requests = [compute_requested[name] for name, active in
                                 (('grid', active_grid), ('refinement', active_refinement)) if active=='cuda']
                if not cuda_requests or any(request!='auto' for request in cuda_requests) or not is_cuda_runtime_error(error):
                    raise
                grid_fallback_reason = f'{type(error).__name__}: {error}'
                compute_fallbacks.append({'stage':'grid','from':'cuda','to':'cpu',
                    'swatch_id':t.swatch_id,'reason':grid_fallback_reason})
                log_fn(f'[v46 compute] auto CUDA grid/center failed for {t.swatch_id}; CPU retry: {grid_fallback_reason}')
                active_grid = active_refinement = 'cpu'
                # Retry only this unfinished template. Prior completed models
                # stay intact; following models avoid a repeatedly broken GPU.
                detected = D.detect_template(proposal_image,variant,pa,grid_fraction=grid_fraction,
                    grid_overlap=grid_overlap,ignore_regions=exclusions,defer_nms=True,
                    compute_baseline=False,preprocessing_cache=preprocessing_cache,
                    **({'occlusion_mask':window_occlusion_mask} if window_occlusion_mask is not None else {}),
                    **({'grid_identity_competitor':identity_competitor} if identity_competitor is not None else {}))
            if grid_fallback_reason:
                for field in ('grid_diagnostics','refinement_diagnostics','candidate_diagnostics','clustering_diagnostics'):
                    stats = dict(getattr(detected, field, {}))
                    stats.update(backend='cpu', used_cuda=False, fallback_reason=grid_fallback_reason)
                    setattr(detected, field, stats)
            results.append(detected)
    grid_models_seconds = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    # Keep score gates, but do not call the legacy competition function here:
    # it suppresses even SAME-type alternatives before their full windows pass.
    for r in results:
        diameter = r.template.diameter
        if fixed_window_geometry:
            # A size correction must not also loosen the grid score threshold.
            diameter /= r.template.proposal_scale
        floor = (.72 if diameter<9 else .68 if diameter<20 else .66)
        r.detections = [d for d in r.detections if d.score >= floor or
            getattr(d,'colour_visibility',{}).get('candidate_supported',False)]
    kept=[]; records=[]; verified=[]; window_stats=[]
    # Keep only small measured masks until selection. Do not retain the full
    # WindowVerification image arrays or perform a second verification pass.
    measured_geometry = {}
    batch_windows = None
    window_fallback_reason = None
    def window_template(result):
        return by_key[result.template.key] if use_legend_windows else result.template
    window_search_options = ({'search_scales':window_search_scales}
                             if window_search_scales is not None else {})
    if scale_policy=='shared_symbol':
        window_search_options={'search_scales':{k:(s,) for k,s in approved_scales.items()},
                               'lock_aspect':True}

    if window_backend == 'cuda':
        # Preserve template/proposal/candidate order while batching independent
        # windows. This removes the old per-candidate CPU->GPU->CPU call chain.
        requests = [(window_template(r), d.x, d.y, fixed_window_geometry)
                    for r in results for d in r.detections]
        try:
            batch_windows = iter(verify_marker_windows_many(window_image, requests,
                backend='cuda', gpu_batch_size=gpu_batch_size, **window_search_options)) if requests else iter(())
        except Exception as error:
            if compute_requested['window']!='auto' or not is_cuda_runtime_error(error):
                raise
            window_fallback_reason = f'{type(error).__name__}: {error}'
            compute_fallbacks.append({'stage':'window','from':'cuda','to':'cpu','reason':window_fallback_reason})
            log_fn(f'[v46 compute] auto CUDA window batch failed; original CPU windows: {window_fallback_reason}')
    for r in results:
        for d in r.detections:
            window_options = ({'backend': window_backend, 'gpu_batch_size': gpu_batch_size}
                              if window_backend != 'cpu' and not window_fallback_reason else {})
            window_options.update(window_search_options)
            if window_occlusion_mask is not None:
                window_options['occlusion_mask']=window_occlusion_mask
            if window_uncertainty_mask is not None:
                window_options['uncertainty_mask']=window_uncertainty_mask
            v = (next(batch_windows) if batch_windows is not None else
                 verify_marker_window(geometry.image_for(r.template) if geometry is not None else window_image,window_template(r),d.x,d.y,
                                      fixed_geometry=fixed_window_geometry, **window_options))
            if geometry is not None:
                geometry.assess(window_template(r),v)
            absolute_scale = v.scale * (1. if use_legend_windows else r.template.proposal_scale)
            diagnostic = getattr(v, 'compute_diagnostics', {})
            if window_fallback_reason:
                diagnostic = {**diagnostic, 'backend':'cpu','used_cuda':False,'fallback_reason':window_fallback_reason}
            if diagnostic:
                window_stats.append({'swatch_id': r.template.swatch_id,
                    'x':float(d.x), 'y':float(d.y), **diagnostic})
            record = {**asdict(d),'decision':v.decision,'required_recall':v.required_recall,
                      'candidate_index':len(records),
                      'template':r.template.key,'swatch_id':r.template.swatch_id,
                      'shape_hint':r.template.shape_hint or r.template.name,
                      'class_name':r.template.name,
                      'window_scale':absolute_scale,
                      'window_scale_reference':effective_window_reference,
                      'proposal_scale':r.template.proposal_scale,
                      'scale_policy':scale_policy,
                      'scale_status':calibration[r.template.key]['status'],
                      'geometry_fixed':bool(getattr(v,'geometry_fixed',scale_policy!='per_candidate')),
                      'matching_profile':r.template.matching_profile,
                      'aligned_x':v.aligned_x,'aligned_y':v.aligned_y,
                      'selected':False}
            if 'open_interior' in diagnostic:
                record['open_interior'] = diagnostic['open_interior']
            if 'geometry_first' in diagnostic:
                record['geometry_first']=diagnostic['geometry_first']
            if 'circle_rim' in diagnostic:
                record['circle_rim']=diagnostic['circle_rim']
            if 'colour_uncertainty' in diagnostic:
                record['colour_uncertainty'] = diagnostic['colour_uncertainty']
            for key in WINDOW_METRICS:
                value = getattr(v,key,None)
                record[key] = None if value is None else float(value)
            # Compatibility adapters may omit these legacy selection fields.
            for key in ('strict_core_recall','boundary_recall','contour_support','aspect_ratio'):
                if record[key] is None:
                    record[key] = 1. if key == 'aspect_ratio' else 0.
            if window_occlusion_mask is not None:
                # Recheck at the geometry actually chosen by the large window,
                # not at the earlier grid seed/scale. A window recall boosted
                # by ignored constraints alone cannot confirm a hidden shape.
                from bw_colour_visibility_v46 import measure
                from occlusion_aware_window_verifier import _uncertain_shape
                wt=window_template(r)
                mask,_,_,_,_=_uncertain_shape(wt,v.scale,record['aspect_ratio'])
                proxy=replace(wt,mask=mask,valid_weight=np.ones(mask.shape,np.float32),
                    edge=cv2.Canny(mask.astype(np.uint8)*255,40,100)>0,
                    diameter=wt.diameter*v.scale)
                other=D._extract_aligned_patch(window_occlusion_mask,v.aligned_x,v.aligned_y,mask.shape)
                if float((other*mask).sum()) / max(int(mask.sum()),1) >= .03:
                    patch=np.stack([D._extract_aligned_patch(window_image[:,:,c],v.aligned_x,v.aligned_y,mask.shape,fill=255)
                                    for c in range(3)],axis=-1)
                    state=measure(proxy,D.ink_membership(patch,wt.ink),other)
                    record['window_colour_visibility']=state
                    diameter=r.template.diameter
                    floor=.72 if diameter<9 else .68 if diameter<20 else .66
                    if not state['candidate_supported']:
                        record['decision']='rejected'
                        record['exclusion_reason']='insufficient_visible_own_ink'
                    elif v.decision in {'verified','ambiguous'} and (not state['identity_observable'] or d.score<floor):
                        record['decision']='ambiguous'
                        record['deferred_reason']='other_colour_identity_unobservable'
                elif getattr(d,'colour_visibility',{}):
                    # A masked seed may align elsewhere. A vanishing rival
                    # colour tail cannot justify bypassing the ordinary grid
                    # floor at the final unoccluded geometry.
                    diameter=r.template.diameter
                    if d.score < (.72 if diameter<9 else .68 if diameter<20 else .66):
                        record['decision']='rejected'
                        record['exclusion_reason']='occlusion_not_present_at_final_geometry'
            if fill_competitor is not None:
                fill = fill_competitor.evaluate_fill(r.template, v.aligned_x, v.aligned_y,
                            total_scale=absolute_scale, aspect=record['aspect_ratio'])
                record['fill_identity'] = fill
                if fill['decision'] == 'conflict':
                    record['exclusion_reason'] = 'interior_fill_conflict'
                    fill_competitor.counts['window_conflicts'] += 1
                elif (fill_competitor.enabled and fill['decision'] == 'abstain'
                      and fill['style'] != 'uncertain' and v.decision == 'verified'):
                    # Not enough unoccluded fill evidence is not an active
                    # identity. Preserve near-pass hypotheses in the typed S
                    # pool instead of treating abstention as a positive match.
                    record['decision'] = 'ambiguous'
                    record['deferred_reason'] = fill['reason']
                    fill_competitor.counts['window_deferred'] += 1
            if v.decision in {'verified','ambiguous'}:
                rendered = getattr(v,'template_mask',None)
                measured_geometry[len(records)] = {
                    'mask':(geometry.appearance(window_template(r),v) if geometry is not None else
                            r.template.mask if rendered is None else rendered),
                    'marker_offset_x':0. if rendered is None else float(v.aligned_x-d.x),
                    'marker_offset_y':0. if rendered is None else float(v.aligned_y-d.y),
                }
            records.append(record)
            if record.get('exclusion_reason') == 'interior_fill_conflict':
                continue
            if record['decision'] != 'verified':
                continue
            final_identity = identity_competitor
            if (geometry_identity_competitor is not None and
                    r.template.key in geometry_identity_competitor.bodies):
                final_identity = geometry_identity_competitor
            if final_identity is not None:
                # Re-evaluate the SAME cell evidence at the full window's final
                # alignment/absolute scale. A coarse wrong-size model must not
                # override the identity found at the verified marker geometry.
                identity=final_identity.for_template(r.template,v.aligned_x,v.aligned_y,
                    total_scale=absolute_scale)
                record['grid_identity']=identity
                if identity['winner'] not in (None,r.template.key):
                    record['exclusion_reason']='grid_identity_conflict'
                    continue
            if min_required_recall is not None and v.required_recall < min_required_recall:
                record['excluded_by_confidence_floor']=True
                record['exclusion_reason']='confidence_floor'
                continue
            # Export the centre of the alignment which actually passed, not
            # the pre-verification grid seed. Recheck ROI after the shift.
            d = replace(d,x=v.aligned_x,y=v.aligned_y)
            if not (pa[0]<=d.x<pa[2] and pa[1]<=d.y<pa[3]):
                record['exclusion_reason']='aligned_centre_outside_plot'
                continue
            if any(b[0]<=d.x<b[2] and b[1]<=d.y<b[3] for b in exclusions):
                record['exclusion_reason']='aligned_centre_in_legend'
                continue
            # Positive visible-boundary evidence resolves competing shapes;
            # raw ink area alone rewarded a circle inside a real diamond.
            rank = v.required_recall + .12*getattr(v,'contour_support',0.) + .03*d.score
            record['base_selection_rank']=float(rank)
            body_evidence=record.get('grid_identity',{}).get('body_boundary',{}).get(r.template.key,{})
            if body_evidence.get('score') is not None:
                record['body_boundary_evidence']=body_evidence
                record['body_boundary_rank_weight']=.35
                rank+=.35*body_evidence['score']
            if geometry is not None:
                evidence=record['geometry_first']
                rank-=.50*evidence['loss']
                if evidence['compact_outer_iou'] is not None:
                    rank+=.40*evidence['compact_outer_iou']
            if fill_competitor is not None:
                rank -= .40 * record['fill_identity']['loss']
            record['selection_rank']=float(rank)
            record['effective_diameter']=float(by_key[r.template.key].diameter*absolute_scale)
            # The convex footprint is selection metadata only. It never
            # contributes reconstructed pixels to matching or validation.
            rendered=getattr(v,'template_mask',None)
            if rendered is None:  # compatibility with explicit test adapters
                rendered=r.template.mask
                cx,cy=v.aligned_x,v.aligned_y
            else:
                cx,cy=record['x'],record['y']
            yy,xx=np.nonzero(rendered)
            if len(xx)>=3:
                hull=cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32)).reshape(-1,2)
                hull+=np.array([int(round(cx))-rendered.shape[1]//2,
                                int(round(cy))-rendered.shape[0]//2])
                record['footprint_polygon']=hull.tolist()
            verified.append(record)
    window_and_records_seconds = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    def attach_geometry(point, record):
        """Export the exact verified raster at its grid-seed anchor, not a glyph."""
        geometry = measured_geometry[record['candidate_index']]
        template = by_key[record['template']]
        point.update(marker_mask=encode_marker_mask(geometry['mask']),
            marker_offset_x=geometry['marker_offset_x'],marker_offset_y=geometry['marker_offset_y'],
            marker_scale=float(record['window_scale']),marker_aspect=float(record['aspect_ratio']),
            source_diameter=float(template.diameter),
            effective_diameter=float(template.diameter*record['window_scale']))

    for record in select_verified(verified):
        template=by_key[record['template']]
        name=template.name
        point={'class_name':name,'shape_hint':template.shape_hint or name,'template':template.key,
                     'swatch_id':template.swatch_id,'class_idx':series_indices[template.key],
                     'shape_idx':CLASSES.index(name),
                     'cx':record['aligned_x'],'cy':record['aligned_y'],
                     'confidence':float(record['required_recall']),'source':VERSION,
                     'point_id':f'P{len(kept)+1:03d}','original_detection':True,
                     'candidate_index':record['candidate_index']}
        attach_geometry(point,record)
        if fill_competitor is not None:
            point['fill_identity'] = record['fill_identity']
            point['fill_style'] = fill_competitor.descriptors[template.key]['style']
        if geometry is not None:
            point['fill_identity']=record['geometry_first']
            point['fill_style']=geometry.descriptors[template.key]['style']
        kept.append(point)
    # Preserve the reviewed experiment's legacy reference diameter for
    # suppression/Step-5 thresholds; physical sizes live in effective_diameter.
    d_est = float(np.median([t.diameter*(approved_scales[t.key] if fixed_window_geometry else 1.)
                            for t in templates]))
    metadata = {t.key:{'source_diameter':float(t.diameter), 'class_name':t.name,
                'shape_hint':t.shape_hint or t.name, 'class_idx':series_indices[t.key],
                'shape_idx':CLASSES.index(t.name)} for t in templates}
    pool = build_suppressed(records,kept,pa,exclusions,default_diameter=d_est,
                            swatch_metadata=metadata)
    for point in pool['suppressed']:
        point['original_detection'] = False
        attach_geometry(point,records[point['candidate_index']])
        if fill_competitor is not None:
            point['fill_identity'] = records[point['candidate_index']]['fill_identity']
    # Actual template pixels and a compact plot overlay for the existing UI.
    selection_seconds = time.perf_counter()-stage_started
    stage_started = time.perf_counter()
    panels=[]
    for t in templates:
        m = cv2.cvtColor(np.where(t.mask,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        m = cv2.resize(m,(160,160),interpolation=cv2.INTER_NEAREST)
        panel = np.full((218,230,3),255,np.uint8)
        panel[:160,:160] = m
        cv2.putText(panel,f'{t.swatch_id} {t.name}'.strip(),(3,182),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1,cv2.LINE_AA)
        size_label = (f'plot scale {approved_scales[t.key]:.3f}' if scale_policy!='per_candidate'
                      else 'auto candidate scale')
        cv2.putText(panel,size_label,(3,204),cv2.FONT_HERSHEY_SIMPLEX,.40,(20,20,20),1,cv2.LINE_AA)
        panels.append(panel)
    from bw_legend_diagnostics import legend_diagnostic_steps
    legend_steps = legend_diagnostic_steps(image, legend_area, templates, report)
    log_fn(f'[v46] {len(kept)} full-window verified points; {len(pool["suppressed"])} typed suppressed hypotheses; no ViT inference')
    if window_backend == 'cuda':
        fallbacks = sum(not row.get('used_cuda', False) for row in window_stats)
        log_fn(f'[v46 compute] CUDA window screening: {len(window_stats)-fallbacks}/{len(window_stats)}; exact CPU fallbacks: {fallbacks}')
    point_timings = dict(backend_setup_seconds=backend_setup_seconds,
        legend_seconds=legend_seconds,scale_policy_seconds=scale_policy_seconds,
        grid_models_seconds=grid_models_seconds,window_and_records_seconds=window_and_records_seconds,
        selection_seconds=selection_seconds,legend_panels_seconds=time.perf_counter()-stage_started,
        total_seconds=time.perf_counter()-started,
        timing_scope='Sequential detect_points wall time; per-model/window diagnostics are nested')
    log_fn('[v46 timing] '+', '.join(f'{key}={value:.3f}s' for key,value in point_timings.items()
                                  if key.endswith('_seconds')))
    from bw_series_correction_v46 import template_evidence
    correction_models=template_evidence(templates,approved_scales,report)
    return {'kept':kept,'suppressed':pool['suppressed'], 'mode_xs':np.array([]),
            'd_est':d_est,
            'diagnostics':{'backend':VERSION,'grid_fraction':grid_fraction,'grid_overlap':grid_overlap,
                           'correction_marker_evidence':correction_models,
                           'colour_visibility_policy':('bw_three_state_visibility_v1' if window_occlusion_mask is not None else None),
                           'fill_identity_enabled':bool(fill_identity or geometry is not None),
                           'geometry_first':geometry.report() if geometry is not None else {'enabled':False},
                           'grid_identity_competition':identity_competitor.report() if identity_competitor is not None else {'enabled':False},
                           'geometry_window_identity_competition':(geometry_identity_competitor.report()
                               if geometry_identity_competitor is not None else {'enabled':False}),
                           'suppressed_pool':pool['summary'],
                           'suppressed_rejections':pool['rejected'],
                           'point_pipeline_timings':point_timings,
                           'template_timings':[{'swatch_id':r.template.swatch_id,
                               'scale':r.template.proposal_scale,
                               **getattr(r,'performance_diagnostics',{})} for r in results],
                           'compute_requested':compute_requested,
                           'compute_selection':compute_selected,
                           'cuda_probe':compute_plan['cuda_probe'],
                           'compute_fallbacks':compute_fallbacks,
                           'refinement_backend':refinement_backend,
                           'grid_backend':grid_backend,'window_backend':window_backend,
                           'grid_models':[{'swatch_id':r.template.swatch_id,
                               'scale':r.template.proposal_scale,
                               **getattr(r,'grid_diagnostics',{})} for r in results],
                           'window_models':window_stats,
                           'window_dispatch':'multi_window_batch' if window_backend=='cuda' and not window_fallback_reason else 'scalar_cpu',
                           'candidate_models':[{'swatch_id':r.template.swatch_id,
                               'scale':r.template.proposal_scale,
                               **getattr(r,'candidate_diagnostics',{})} for r in results],
                           'clustering_models':[{'swatch_id':r.template.swatch_id,
                               'scale':r.template.proposal_scale,
                               **getattr(r,'clustering_diagnostics',{})} for r in results],
                           'refinement_models':[{'swatch_id':r.template.swatch_id,
                               'scale':r.template.proposal_scale,
                               **getattr(r,'refinement_diagnostics',{})} for r in results],
                           'model_revision':('swatch-grid-identity-v5' if grid_identity_competition else 'swatch-identity-v4'),
                           'window_scale_reference':effective_window_reference,
                           'window_search_scales':list(window_search_scales) if window_search_scales is not None else None,
                           'shared_symbol_scales':dict(approved_scales) if scale_policy=='shared_symbol' else None,
                           'window_aspect_locked':scale_policy=='shared_symbol',
                           'scale_policy':scale_policy,'scale_calibration':calibration,
                           'swatches':report,'candidates':records,
                           'legend_extraction':{'status':'partial' if legend_failures else 'complete',
                               'entry_count':len(report),'usable_models':len(templates),
                               'failed_ids':[r['swatch_id'] for r in legend_failures]},
                           'proposal_models':[{'template':r.template.key,
                               'swatch_id':r.template.swatch_id,'shape_hint':r.template.shape_hint or r.template.name,
                               'scale':r.template.proposal_scale,'diameter':r.template.diameter}
                               for r in results]},
            'diag_steps':[{'title':'v46 — observed legend templates and plot-scale decisions',
                           'img_bgr':np.concatenate(panels,axis=1),
                           'output_filename':'v46_legend_templates.png'}, *legend_steps]}

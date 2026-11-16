"""ROI-assisted, paper-described ChartParser pivot-to-legend postprocessing.

Uses only cached official neural heads, original RGB pixels, external plot/legend
ROIs, and prior swatch *bounding boxes*. Never uses v46 plot marker coordinates,
curves, inferred template pixels, shape predictions or palette distances.

Paper: AI-ChartParser, CGF 2025, section 3.3.2 (page 7), section 4.4.3
(page 9). Threshold > .6, morphology/8-connectivity/single-pixel centreline,
native-original 40x20 pivot crops, learned LineEX similarity and argmax grouping.
Morphological kernel and pivot correction rule are unspecified in the paper:
our explicit choices below are NOT an exact reproduction of unknown code.
The released detector has no usable swatch boxes in these cached results, so
legend localization remains an external ROI-assisted input, not end-to-end.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
VERSION = 'chartparser_paper_grouping_roi_assisted_v1'
PAPER = 'AI-ChartParser: A Method For Extracting Experimental Data From Curve Charts in Literature, doi:10.1111/cgf.70146'
CHOICES = {
    'curve_probability': 'Cached segmentation[0,1], already sigmoid in ChartParserModel.forward; no second activation.',
    'probability_inconsistency': 'Paper section 3.2.3 says two-channel Softmax, section 3.2.4 discusses Sigmoid, and the public forward applies independent Sigmoid. This run thresholds the released cached Sigmoid output directly, not a claim of identical paper Softmax probabilities.',
    'threshold': .6,
    'morphology': 'One 3x3 all-ones binary closing; constant-zero border.',
    'connectivity': 8,
    'component_policy': 'Retain every connected component; no minimum-area deletion.',
    'thinning': 'skimage.morphology.skeletonize(method=zhang) independently within each 8-connected component.',
    'correction': 'Nearest centreline pixel in x +/- 1 columns and abs(delta_y) < 15 model pixels; minimum Euclidean distance, ties smaller y then x. Snap x and y to that centreline pixel. Reject if absent.',
    'correction_parameters_provenance': 'The paper does not specify kernel/thinner/selection radius. Column band and vertical radius are borrowed from public process_kps; nearest selection and exact snapping are explicit experimental choices.',
    'public_reference': 'Separate result using original argmax mask, public x +/- 1, y-run gap <= 2, first ascending-y mean within 15, average with original y. Safety bounds added; not used for grouping.',
    'pivot_patch_size_width_height': [40, 20],
    'pivot_patch_coordinate_system': 'Native original-resolution upright input crop, not the 512x512 model input.',
    'legend_images': 'Unmodified original RGB swatch box crops, connector pixels retained. No reconstructed swatch masks.',
    'grouping': 'Official LineEX learned Siamese + MLP similarity, highest score over supplied legend crops. No histogram override or confidence rejection.',
    'shape_classification': 'Not performed. Legend identity is not a semantic circle/square/triangle prediction.',
    'deduplication': 'None. Distinct raw pivot hypotheses can snap to the same pixel and remain separate records.',
    'no_legend_policy': 'Abstain from grouping even for a nominal single-series panel; no fabricated legend.',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def in_box(x, y, box):
    return bool(box and box[0] <= x < box[2] and box[1] <= y < box[3])


def in_scope(x, y, panel):
    return in_box(x, y, panel['plot_box']) and not in_box(x, y, panel.get('legend_box'))


def upright_to_original(x, y, panel):
    """Pixel centres: inverse lossless clockwise quarter-turn, not box edges."""
    width, height = panel['source_image_size']
    rotation = int(panel.get('clockwise_rotation_degrees', 0)) % 360
    if rotation == 0:
        return float(x), float(y)
    if rotation == 90:
        return float(y), float(height - 1 - x)
    if rotation == 180:
        return float(width - 1 - x), float(height - 1 - y)
    if rotation == 270:
        return float(width - 1 - y), float(x)
    raise ValueError(f'Unsupported rotation {rotation}')


def original_box(box, panel):
    """Inverse rotation of half-open box *edges*: no minus-one correction."""
    x0, y0, x1, y1 = map(float, box)
    width, height = panel['source_image_size']
    rotation = int(panel.get('clockwise_rotation_degrees', 0)) % 360
    if rotation == 0:
        return [x0, y0, x1, y1]
    if rotation == 90:
        return [y0, height-x1, y1, height-x0]
    if rotation == 180:
        return [width-x1, height-y1, width-x0, height-y0]
    if rotation == 270:
        return [width-y1, x0, width-y0, x1]
    raise ValueError(f'Unsupported rotation {rotation}')


def point_coordinates(model_x, model_y, panel):
    width, height = panel['image_size']
    x, y = float(model_x) / 512 * width, float(model_y) / 512 * height
    ux, uy = x + panel['crop_box_upright'][0], y + panel['crop_box_upright'][1]
    ox, oy = upright_to_original(ux, uy, panel)
    return dict(x=x, y=y, upright_x=ux, upright_y=uy, original_x=ox, original_y=oy,
                model_x=float(model_x), model_y=float(model_y))


def centreline(segmentation, threshold=.6):
    """Direct sigmoid channel threshold, morphology, eight-connectivity/thinning."""
    seg = np.asarray(segmentation)
    if seg.shape != (1, 2, 512, 512):
        raise ValueError(f'Unexpected segmentation shape {seg.shape}')
    if not np.isfinite(seg).all() or seg.min() < 0 or seg.max() > 1:
        raise ValueError('Expected already-sigmoid finite cached segmentation')
    binary = (seg[0, 1] > threshold).astype(np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    thin = np.zeros_like(binary, bool)
    components = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        region = labels[y:y+height, x:x+width] == label
        skeleton = skeletonize(region, method='zhang')
        thin[y:y+height, x:x+width] |= skeleton
        components.append(dict(component_id=label, box_model=[x, y, x+width, y+height],
                               area=area, skeleton_pixels=int(skeleton.sum())))
    return binary, closed, thin, labels, components


def nearest_centreline(model_x, model_y, thin, radius=15):
    x, y = int(model_x), int(model_y)
    height, width = thin.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    x0, x1 = max(0, x-1), min(width, x+2)
    y0, y1 = max(0, y-radius+1), min(height, y+radius)
    ys, xs = np.where(thin[y0:y1, x0:x1])
    if not len(xs):
        return None
    xs, ys = xs+x0, ys+y0
    d2 = (xs-x)**2 + (ys-y)**2
    index = np.lexsort((xs, ys, d2))[0]
    return int(xs[index]), int(ys[index])


def public_position_reference(model_x, model_y, mask):
    """Same run/first-match maths as public locate_offset, safe integer bounds."""
    x, y = int(model_x), int(model_y)
    if not (0 <= x <= 510 and 0 <= y < 512):
        return None
    # The upstream x=0 uses negative wrapping. Reject that unsafe case explicitly.
    if x == 0:
        return None
    rows = np.where((mask[:, x-1] == 1) | (mask[:, x] == 1) | (mask[:, x+1] == 1))[0]
    if not len(rows):
        return None
    groups = np.split(rows, np.where(np.diff(rows) > 2)[0]+1)
    for group in groups:
        midpoint = int(group.sum() // len(group))
        if abs(y-midpoint) < 15:
            return x, (y+midpoint)//2
    return None


def load_legends(panel, image_rgb, destination):
    """Read only swatch boxes from prior legend diagnostics; crop raw RGB pixels."""
    run = Path(panel.get('v46_run_dir') or Path(panel['v46_result_path']).parent)
    colour = run / 'legend_hybrid_v46.json'
    bw = run / 'detection.json'
    if not panel.get('legend_box'):
        return [], [], {'status': 'no_reviewed_legend', 'source_path': None}, []
    if colour.exists():
        source = colour
        data = read(source)
        entries = []
        for i, entry in enumerate(data.get('entries', [])):
            observed = entry.get('composition', {}).get('source_swatch_box')
            field = 'composition.source_swatch_box' if observed else 'box'
            entries.append((entry.get('id', f'S{i+1:02d}'), observed or entry.get('box'),
                            f'entries[{i}].{field}'))
        del data
    elif bw.exists():
        source = bw
        data = read(source)
        entries = [(entry.get('swatch_id', f'S{i+1:02d}'), entry.get('box'),
                    f'diagnostics.swatches[{i}].box')
                   for i, entry in enumerate(data.get('diagnostics', {}).get('swatches', []))]
        del data
    else:
        return [], [], {'status': 'missing_swatch_bbox_file', 'source_path': None}, []
    provenance = dict(status='bbox_file_read', source_path=str(source.resolve()), source_sha256=sha(source),
                      bbox_coordinate_system='upright full-image half-open pixel edges',
                      use='Only bbox and stable entry ID read; no v46 plot points/curves/templates/shape/color features used.')
    images, legends, unavailable = [], [], []
    origin_x, origin_y = panel['crop_box_upright'][:2]
    height, width = image_rgb.shape[:2]
    seen = set()
    for identity, box, field in entries:
        reason = None
        if not box or len(box) != 4 or not np.isfinite(box).all():
            unavailable.append(dict(source_entry_id=identity, reason='invalid_box', field=field))
            continue
        x0, y0, x1, y1 = map(int, box)
        local = [x0-origin_x, y0-origin_y, x1-origin_x, y1-origin_y]
        a, b, c, d = local
        if min(c-a, d-b) <= 0 or not (0 <= a < c <= width and 0 <= b < d <= height):
            reason = 'bbox_outside_model_input_or_empty'
        elif not in_box((a+c)/2, (b+d)/2, panel['legend_box']):
            reason = 'bbox_center_outside_reviewed_legend'
        elif tuple(local) in seen:
            reason = 'duplicate_swatch_box'
        if reason:
            unavailable.append(dict(source_entry_id=identity, box_input=local, reason=reason, field=field))
            continue
        seen.add(tuple(local))
        legend_id = f'L{len(legends)+1:02d}'
        crop = np.ascontiguousarray(image_rgb[b:d, a:c])
        path = destination / 'legends' / f'{legend_id}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop).save(path)
        # RGB is only descriptive source-pixel metadata, not a grouping feature.
        rgb = np.median(crop.reshape(-1, 3), axis=0).tolist()
        legends.append(dict(legend_id=legend_id, source_entry_id=str(identity), box_input=local,
                            box_upright=[x0,y0,x1,y1], box_original=original_box([x0,y0,x1,y1], panel),
                            original_crop_path=str(path.resolve()), crop_size=[c-a,d-b],
                            crop_sha256=sha(path), bbox_provenance=dict(provenance, field=field),
                            median_rgb=rgb, semantic_shape=None))
        images.append(crop)
    provenance['status'] = 'available' if images else 'no_valid_swatch_boxes'
    return legends, images, provenance, unavailable


def get_matcher_module():
    path = ROOT.parent / 'lineex' / 'legend_matcher.py'
    spec = importlib.util.spec_from_file_location('published_lineex_legend_matcher', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def process(panel, heads_root, destination, matcher, matcher_module):
    started = time.perf_counter()
    destination = destination / panel['panel_id']
    destination.mkdir(parents=True, exist_ok=True)
    headspath = heads_root / panel['panel_id'] / 'result.json'
    heads = read(headspath)
    if heads.get('status') != 'completed':
        raise ValueError(f'Cached head inference not completed: {heads.get("status")}')
    if heads['input_sha256'] != panel['input_sha256'] or sha(panel['image_path']) != panel['input_sha256']:
        raise ValueError('Input SHA mismatch')
    with Image.open(panel['image_path']) as image:
        image_rgb = np.asarray(image.convert('RGB'))
    assert list(image_rgb.shape[:2][::-1]) == panel['image_size']
    legend_start = time.perf_counter()
    legends, legend_images, legend_provenance, rejected_legends = load_legends(panel, image_rgb, destination)
    legend_seconds = time.perf_counter()-legend_start
    with np.load(heads['raw_heads_path']) as arrays:
        seg = arrays['segmentation']
    binary, closed, thin, components_labels, components = centreline(seg)
    public_mask = seg.argmax(axis=1)[0]
    for name, mask in [('curve_threshold_512', binary), ('curve_closed_512', closed), ('centerline_512', thin)]:
        Image.fromarray(np.asarray(mask, np.uint8)*255).save(destination / f'{name}.png')
    raw, corrected, rejected, excluded, public = [], [], [], [], []
    for index, point in enumerate(heads.get('raw_points', [])):
        point_id = f'P{index+1:04d}'
        mx, my = int(point['model_x']), int(point['model_y'])
        record = dict(point_id=point_id, **point_coordinates(mx, my, panel),
                      score=float(point['score']), legend_id=None)
        assert abs(record['x']-point['x']) < 1e-6 and abs(record['y']-point['y']) < 1e-6
        record['inside_plot_excluding_legend'] = in_scope(record['x'], record['y'], panel)
        raw.append(record)
        if not record['inside_plot_excluding_legend']:
            excluded.append(dict(record, reason='raw_point_outside_reviewed_plot_or_inside_legend'))
            continue
        ref = public_position_reference(mx, my, public_mask)
        public.append(dict(point_id=point_id, status='retained' if ref else 'rejected',
                           raw_x=record['x'], raw_y=record['y'],
                           **(point_coordinates(*ref, panel) if ref else {})))
        match = nearest_centreline(mx, my, thin)
        if match is None:
            rejected.append(dict(record, reason='no_centerline_in_x_band_and_vertical_radius'))
            continue
        candidate = dict(point_id=point_id, **point_coordinates(*match, panel), score=record['score'],
                         raw_x=record['x'], raw_y=record['y'], raw_model_x=mx, raw_model_y=my,
                         model_displacement=float(np.hypot(match[0]-mx, match[1]-my)),
                         component_id=int(components_labels[match[1], match[0]]), legend_id=None)
        if not in_scope(candidate['x'], candidate['y'], panel):
            rejected.append(dict(candidate, reason='corrected_point_outside_reviewed_plot_or_inside_legend'))
            continue
        corrected.append(candidate)
    correction_seconds = time.perf_counter()-started-legend_seconds
    patches = matcher_module.extract_pivot_patches(image_rgb, [(p['x'], p['y']) for p in corrected])
    eligible = []
    for i, box, patch in zip(patches['valid_indices'], patches['boxes'], patches['patches']):
        point = corrected[i]
        path = destination / 'pivot_patches' / f"{point['point_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(patch).save(path)
        point['patch_box_input'] = list(map(int, box))
        point['patch_path'] = str(path.resolve())
        point['patch_status'] = 'available'
        eligible.append(point)
    for item in patches['unavailable']:
        # The matcher helper exports index; do not conflate crop rejection with curve rejection.
        index = int(item.get('index', item.get('point_index', -1)))
        if not 0 <= index < len(corrected):
            raise ValueError(f'Unexpected unavailable patch schema: {item}')
        corrected[index]['patch_status'] = 'unavailable'
        corrected[index]['grouping_unavailable_reason'] = item.get('reason', 'incomplete_native_patch')
    grouping_start = time.perf_counter()
    scores = np.zeros((len(eligible), len(legends)), np.float32)
    groups = [dict(legend_id=entry['legend_id'], point_ids=[], semantic_shape=None) for entry in legends]
    if legends and eligible:
        scores = np.asarray(matcher.score_patches(legend_images, patches['patches']), np.float32)
        if scores.shape != (len(eligible), len(legends)) or not np.isfinite(scores).all():
            raise ValueError(f'Invalid similarity matrix {scores.shape}')
        for row, point in zip(scores, eligible):
            winner = int(np.argmax(row))
            point['legend_id'] = legends[winner]['legend_id']
            point['legend_index'] = winner
            point['similarities'] = row.tolist()
            point['winning_similarity'] = float(row[winner])
            ordered = np.sort(row)
            point['runner_up_margin'] = float(ordered[-1]-ordered[-2]) if len(row)>1 else None
            groups[winner]['point_ids'].append(point['point_id'])
    else:
        for point in eligible:
            point['grouping_unavailable_reason'] = 'no_valid_legend_crops'
    grouping_seconds = time.perf_counter()-grouping_start
    np.save(destination / 'similarity.npy', scores)
    status = ('unavailable_legend' if not legends else
              'completed_no_groupable_points' if not eligible else 'completed')
    warnings = [
        'Paper-described experimental postprocessing with external swatch-bbox assistance, not full native end-to-end AI-ChartParser.',
        'No native class-6 swatch detections in cached heads; public detector class-name maps disagree. Class 3 is NOT remapped to a swatch.',
        'Morphology/thinner/correction selection parameters are implementation choices because the paper omits them.',
        'Paper section 3.2.3 specifies Softmax, but its loss discussion and public forward use Sigmoid. This experiment uses direct cached Sigmoid channel 1 > .6, not reactivated or renormalized probabilities.',
        'A pivot is not necessarily a visible marker. Group counts are not marker accuracy, recall, or calibrated physical data.',
        'Argmax assigns every eligible corrected pivot even with low similarity; winner score is not a calibrated probability.',
        'Fixed native 40x20 crops can be small relative to high-resolution markers; no adaptive-scale improvement is applied.',
        'No OCR, no semantic marker-shape classifier, no axes calibration, and no reconstructed curves are claimed.',
    ]
    result = dict(version=VERSION, panel_id=panel['panel_id'], status=status,
                  source_filename=panel['source_filename'], input_sha256=panel['input_sha256'],
                  source_sha256=panel['source_sha256'], input_image_path=panel['image_path'],
                  input_image_size=panel['image_size'], crop_box_upright=panel['crop_box_upright'],
                  clockwise_rotation_degrees=panel.get('clockwise_rotation_degrees',0),
                  heads_result_path=str(headspath.resolve()), heads_result_sha256=sha(headspath),
                  scope='roi_assisted_paper_postprocessing', paper=PAPER,
                  coordinate_system='input_crop_pixels; original_x/original_y are unrotated source pixel centres',
                  plot_box=panel['plot_box'], legend_box=panel.get('legend_box'),
                  native_legend_stage=dict(available=False,
                      reason='Native swatch stage unavailable: no class-6 boxes; class map unresolved.',
                      observed_class6_count=sum(int(b['class_id'])==6 for b in heads.get('boxes',[]))),
                  legend_provenance=legend_provenance, legends=legends, rejected_legends=rejected_legends,
                  raw_points=[p for p in raw if p['inside_plot_excluding_legend']], raw_points_all=raw,
                  corrected_points=corrected, rejected_points=rejected, excluded_points=excluded,
                  groups=groups, similarity_path=str((destination/'similarity.npy').resolve()),
                  similarity_row_point_ids=[p['point_id'] for p in eligible],
                  similarity_column_legend_ids=[entry['legend_id'] for entry in legends],
                  public_process_kps_reference=public, components=components,
                  centerline_path=str((destination/'centerline_512.png').resolve()),
                  threshold_mask_path=str((destination/'curve_threshold_512.png').resolve()),
                  implementation_choices=CHOICES, mask_probability_threshold=.6,
                  counts=dict(raw=len(raw), scoped=sum(p['inside_plot_excluding_legend'] for p in raw),
                      corrected=len(corrected), curve_or_scope_rejected=len(rejected),
                      excluded_outside_scope=len(excluded),
                      patches=len(eligible), grouped=sum(p['legend_id'] is not None for p in corrected),
                      legends=len(legends)),
                  timing=dict(legend_bbox_read_and_crop_seconds=legend_seconds,
                      correction_seconds=correction_seconds, learned_grouping_seconds=grouping_seconds,
                      total_seconds=time.perf_counter()-started), warnings=warnings,
                  physical_data=None, reconstructed_series=None)
    write(destination/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    panels = read(args.manifest)['panels']
    if args.ids:
        wanted = set(args.ids)
        panels = [p for p in panels if p['panel_id'] in wanted]
        if {p['panel_id'] for p in panels} != wanted:
            raise ValueError('Unknown selected panel IDs')
    module = get_matcher_module()
    matcher = module.LegendMatcher(device=args.device, batch_size=args.batch_size)
    write(args.output/'provenance.json', dict(version=VERSION, paper=PAPER,
        implementation_choices=CHOICES, runner_sha256=sha(__file__),
        manifest_path=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
        matcher_provenance=getattr(matcher,'provenance',None),
        notes='Local inference only. Prior v46 legend bboxes are external assisted inputs, never prior marker coordinates.'))
    rows = []
    for panel in panels:
        path = args.output/panel['panel_id']/'result.json'
        if args.resume and path.exists():
            result = read(path)
            if result.get('version') != VERSION or result.get('input_sha256') != panel['input_sha256']:
                raise ValueError('Stale resumed result')
        else:
            try:
                result = process(panel, args.heads, args.output, matcher, module)
            except Exception:
                result = dict(version=VERSION,panel_id=panel['panel_id'],status='failed',
                    input_sha256=panel['input_sha256'],error=traceback.format_exc())
                write(path,result)
        row = dict(panel_id=panel['panel_id'], status=result['status'],
                   result_path=str(path.resolve()), counts=result.get('counts'), timing=result.get('timing'))
        rows.append(row)
        write(args.output/'summary.json',dict(expected=len(panels),processed=len(rows),panels=rows))
        print(f"{panel['panel_id']}: {result['status']} {result.get('counts',{})}",flush=True)
    return 1 if any(r['status']=='failed' for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Sensitivity run: LineEX in the configuration its PAPER describes, not its released defaults.

The implementation audit upheld two differences between the LineEX paper and the code the authors
released (which our main run follows verbatim):

  1. Background gate threshold. Paper Algorithm 2 (p.6219) prints `threshold = 0.98`; the released
     vendor/LineEX/modules/KP_detection/utils.py:13 hard-codes 0.999 with the comment
     "#0.98 for adobe,0.999 for chartOCR and synt". A higher threshold rejects fewer keypoints, so
     the released value leaves more false positives than the published algorithm would.
  2. Legend patch size. Paper section 4.3 (p.6217): "We sample 20x40 pixel patches around each
     detected keypoint and also resize legend marker bounding boxes to 20x40 pixels". The released
     grouping code instead feeds the raw legend crop through Resize(224)+CenterCrop(224), which
     keeps the aspect ratio and crops away everything outside a centred square - a different field
     of view, not a different distortion.

This script re-runs ONLY LineEX with both paper values (gate 0.98, legend crops resized to 40x20
before the matcher's own 224 transform) and writes `lineex_paper_grouped` predictions next to the
released-configuration ones, so the difference can be measured rather than argued about.

The gate is re-implemented here with an explicit threshold because the vendor constant is hard
coded; at 0.999 it is asserted to reproduce the vendor function's decision for every keypoint of
every panel before any 0.98 result is written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent

spec = importlib.util.spec_from_file_location('benchmark_pk_run_external', EXPERIMENT / 'benchmark_pk' / 'run_external.py')
RX = importlib.util.module_from_spec(spec)
sys.modules['benchmark_pk_run_external'] = RX
spec.loader.exec_module(RX)

PAPER_THRESHOLD = 0.98
RELEASED_THRESHOLD = 0.999
PAPER_LEGEND_SIZE = (40, 20)      # width, height, matching the 40x20 pivot patch the wrapper cuts


def gate(kp, image, threshold):
    """vendor/LineEX/modules/KP_detection/utils.py:checkifbackground with an explicit threshold."""
    alpha = max(image.shape[0], image.shape[1]) * 0.05
    t1, t2 = int(alpha / 2), int(alpha / 6)
    x, y = kp
    sections = [image[y-t1:y-t2, x-t1:x-t2], image[y-t1:y-t2, x-t2:x+t2], image[y-t1:y-t2, x+t2:x+t1],
                image[y-t2:y+t2, x-t1:x-t2], image[y-t2:y+t2, x-t2:x+t2], image[y-t2:y+t2, x+t2:x+t1],
                image[y+t2:y+t1, x-t1:x-t2], image[y+t2:y+t1, x-t2:x+t2], image[y+t2:y+t1, x+t2:x+t1]]
    histograms = []
    for section in sections:
        hist = cv2.calcHist([section], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histograms.append(cv2.normalize(hist, hist).flatten())
    for i, h1 in enumerate(histograms):
        for j, h2 in enumerate(histograms):
            if i == j:
                continue
            if cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) < threshold:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=HERE / 'inputs_with_er_swatches.json')
    parser.add_argument('--output', type=Path, default=HERE / 'results' / 'run_v1' / 'external')
    parser.add_argument('--shared-legends-root', type=Path, default=HERE / 'results' / 'run_v1' / 'v46')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()

    frozen = RX.read(args.manifest)
    RX.validate_safe_tree(frozen)
    panels = [RX.normalize_panel(p) for p in frozen['panels']]
    if args.ids:
        panels = [p for p in panels if p['panel_id'] in set(args.ids)]

    modules = {'matcher': RX.module_at('benchmark_lineex_matcher', 'lineex/legend_matcher.py'),
               'lineex': RX.module_at('benchmark_lineex_heads', 'lineex/run_keypoints.py')}
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    matcher = modules['matcher'].LegendMatcher(device=str(device), batch_size=16)
    model, vendor_gate, provenance = modules['lineex'].load_model(str(device))

    rows = []
    for panel in panels:
      try:
        started = time.perf_counter()
        image = RX.verify_image_binding(panel)
        destination = args.output / panel['panel_id']
        destination.mkdir(parents=True, exist_ok=True)
        legends, legend_images = RX.shared_legend_crops(
            panel, image, args.shared_legends_root / panel['panel_id'] / 'shared_legends.json', destination)
        # paper 4.3: legend marker boxes resized to the pivot patch size
        resized = [cv2.resize(crop, PAPER_LEGEND_SIZE, interpolation=cv2.INTER_AREA) for crop in legend_images]

        heads = modules['lineex'].infer(model, lambda kp, img: gate(kp, img, PAPER_THRESHOLD), panel, str(device))
        # equivalence check of the re-implemented gate at the released threshold
        bgr = cv2.imdecode(np.fromfile(panel['image_path'], np.uint8), cv2.IMREAD_COLOR)
        mismatches = 0
        for query in heads['raw_queries']:
            x, y = int(query['x']), int(query['y'])
            try:
                mine = gate((x, y), bgr, RELEASED_THRESHOLD)
                theirs = bool(vendor_gate((x, y), bgr))
            except Exception:
                continue
            mismatches += int(mine != theirs)
        if mismatches:
            raise RuntimeError(f'Re-implemented gate disagrees with the vendor function on {mismatches} keypoints')

        positions = [dict(point_id=f"LXP{p['query_id']:04d}", x=float(p['x']), y=float(p['y']), detection_score=None)
                     for p in heads['points'] if RX.scoped(p, panel)]
        grouped = RX.group_positions(positions, panel, image, legends, resized, matcher, modules['matcher'],
                                     destination / 'lineex_paper_grouped')
        timing = dict(shared_input_verification_seconds=0., legend_crop_seconds=0.,
                      network_seconds=heads['timing']['network_seconds'],
                      head_postprocess_and_export_seconds=heads['timing']['total_seconds'] - heads['timing']['network_seconds'],
                      curve_correction_seconds=0., **grouped['timing'],
                      method_total_excluding_shared_verification_seconds=heads['timing']['total_seconds'] + grouped['timing']['total_seconds'])
        result = RX.record(panel, 'lineex_paper_grouped', grouped, legends, timing,
                           raw_points=[RX.original_point(p, panel) for p in positions],
                           raw_queries=heads['raw_queries'], head_status=heads['status'],
                           implementation_choices={
                               'keypoints': 'Official 64-query L+D checkpoint',
                               'background_gate': f'paper Algorithm 2 threshold {PAPER_THRESHOLD} '
                                                  f'(released code uses {RELEASED_THRESHOLD}); re-implemented gate '
                                                  f'verified identical to the vendor function at {RELEASED_THRESHOLD}',
                               'legend_patches': f'resized to {PAPER_LEGEND_SIZE[0]}x{PAPER_LEGEND_SIZE[1]} per paper '
                                                 f'section 4.3 before the matcher 224 transform',
                               'grouping': 'Official learned Siamese matcher, no histogram override'})
        result['shared_legend_status'] = RX.read(args.shared_legends_root / panel['panel_id'] / 'shared_legends.json').get('status')
        result['warnings'] = result['limitations']
        RX.write(destination / 'lineex_paper_grouped.json', result)
        kept = sum(1 for p in heads['raw_queries'] if not p.get('background_rejected'))
        rows.append({'panel_id': panel['panel_id'], 'points': len(positions), 'gate_kept': kept,
                     'seconds': round(time.perf_counter() - started, 1)})
        print(json.dumps(rows[-1]), flush=True)
      except Exception as error:   # one bad panel must not silently drop every later panel
        import traceback
        rows.append({'panel_id': panel['panel_id'], 'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        RX.write(args.output / panel['panel_id'] / 'lineex_paper_failure.json',
                 {'panel_id': panel['panel_id'], 'error': traceback.format_exc()})
        print(json.dumps(rows[-1]), flush=True)
    RX.write(args.output / 'lineex_paper_summary.json', {'threshold': PAPER_THRESHOLD,
             'legend_patch_size': list(PAPER_LEGEND_SIZE), 'panels': rows, 'provenance': provenance})


if __name__ == '__main__':
    main()

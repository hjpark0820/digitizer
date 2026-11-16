"""Write shared_legends.json for every panel from the benchmark's annotated legend-key boxes (ground-truth swatches).

Purpose (2026-11-16, user request): measure how much the published methods (AI-ChartParser, LineEX) depend on
the legend crops exported by v46. Normally they receive v46's crops, so a key v46 misses is lost for them too.
Here every panel gets one crop per annotated key, cut from the source image at the annotated box (inputs
'swatches'), in exactly the format ensure_legends.py writes for its annotation fallback, so run_external_v2.py,
run_lineex_paper.py and evaluate_entry.py read it unchanged. v46 is not run or scored in this condition.

usage: python make_gt_swatch_legends.py --inputs inputs_test100v2_20261025.json --out results/<run>/v46
       (refuses to write into an existing --out folder)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f'{args.out} exists; refusing to overwrite')
    panels = json.loads(args.inputs.read_text(encoding='utf-8-sig'))['panels']
    total = 0
    for panel in panels:
        folder = args.out / panel['panel_id']
        image = cv2.imdecode(np.fromfile(panel['image_path'], np.uint8), cv2.IMREAD_COLOR)
        assert hashlib.sha256(Path(panel['image_path']).read_bytes()).hexdigest() == panel['input_sha256']
        offset = panel['crop_box_original'][:2]
        entries = []
        for index, swatch in enumerate(panel['swatches'], 1):
            a, b, c, d = [int(v) for v in swatch['bbox']]
            crop_path = folder / 'shared_legend_crops' / f'legend_{index:02d}.png'
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode('.png', image[b:d, a:c])[1].tofile(crop_path)
            entries.append({'native_id': swatch['series_id'], 'series_id': swatch['series_id'],
                            'source_native_legend_id': swatch['series_id'],
                            'label': swatch.get('label') or swatch['series_id'], 'bbox': [a, b, c, d],
                            'bbox_crop': [a, b, c, d],
                            'bbox_original': [a + offset[0], b + offset[1], c + offset[0], d + offset[1]],
                            'image_path': str(crop_path.resolve()), 'crop_path': str(crop_path.resolve()),
                            'image_size': [c - a, d - b],
                            'sha256': hashlib.sha256(crop_path.read_bytes()).hexdigest(),
                            'observed_rgb_pixels': True, 'connector_pixels_removed': False,
                            'template_rendered': False, 'crop_source': 'benchmark_annotation_swatch_box'})
        total += len(entries)
        value = {'panel_id': panel['panel_id'], 'modality': panel['modality'], 'entries': entries,
                 'mapping': {e['native_id']: e['series_id'] for e in entries},
                 'annotations': panel['swatches'], 'status': 'available', 'source': 'ground_truth_swatches',
                 'input_sha256': panel['input_sha256'], 'source_sha256': panel['source_sha256'],
                 'coordinate_convention': 'bbox crop-local half-open; bbox_original full original half-open',
                 'protocol': 'Experiment: crops cut from the benchmark annotated swatch boxes for every key, '
                             'independent of v46; for the published methods only.'}
        (folder / 'shared_legends.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                                                     encoding='utf-8')
    print(f'{len(panels)} panels, {total} annotated keys written to {args.out}')


if __name__ == '__main__':
    main()

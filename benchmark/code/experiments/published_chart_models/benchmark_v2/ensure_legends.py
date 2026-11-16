"""Give the external models a legend front end even where v46's own legend stage failed.

The external benchmark reuses the legend crops that the v46 run exported (shared_legends.json).
When v46 fails before that export - e.g. bavituximab1__p02, where BW legend extraction rejects
both entries as "line_without_marker_bulge" - the published models would be scored with no legend
at all, i.e. punished for OUR failure. For panels that carry annotated swatch boxes (the PK
dataset) this script writes the missing shared_legends.json directly from those annotations, with
source='annotation_fallback'. ER panels have no swatch annotation, so nothing is fabricated there
and they stay legend-unavailable for every method alike.

Existing successful v46 exports are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=HERE / 'inputs_with_er_swatches.json')
    parser.add_argument('--results', type=Path, default=HERE / 'results' / 'run_v1' / 'v46')
    args = parser.parse_args()
    if not args.inputs.exists():
        args.inputs = HERE / 'inputs.json'
    report = []
    for panel in read(args.inputs)['panels']:
        folder = args.results / panel['panel_id']
        path = folder / 'shared_legends.json'
        current = read(path) if path.exists() else {'status': 'missing', 'entries': []}
        if (current.get('status') == 'available' and current.get('entries')
                and current.get('source') != 'annotation_fallback'):
            report.append({'panel_id': panel['panel_id'], 'action': 'kept v46 export',
                           'entries': len(current['entries'])})
            continue
        if not panel['swatches']:
            report.append({'panel_id': panel['panel_id'], 'action': 'left unavailable (no annotated swatches)',
                           'reason': current.get('reason', '')})
            continue
        image = cv2.imdecode(np.fromfile(panel['image_path'], np.uint8), cv2.IMREAD_COLOR)
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
        write(path, {'panel_id': panel['panel_id'], 'modality': panel['modality'], 'entries': entries,
                     'mapping': {e['native_id']: e['series_id'] for e in entries},
                     'annotations': panel['swatches'], 'status': 'available',
                     'source': 'annotation_fallback',
                     'input_sha256': panel['input_sha256'], 'source_sha256': panel['source_sha256'],
                     'coordinate_convention': 'bbox crop-local half-open; bbox_original full original half-open',
                     'protocol': 'v46 legend export was unavailable for this panel; crops cut from the benchmark '
                                 'annotated swatch boxes so the external models are not penalised for a v46 failure.',
                     'previous_status': current.get('status'), 'previous_reason': current.get('reason', '')})
        report.append({'panel_id': panel['panel_id'], 'action': 'wrote annotation fallback',
                       'entries': len(entries)})
    write(HERE / 'legend_availability.json', report)
    for row in report:
        print(f"{row['panel_id']:38s} {row['action']}", flush=True)


if __name__ == '__main__':
    main()

"""Run run_parallel.py on the Astra/Sol 100 simulated charts, including the 11 charts without a legend.

Charts without a rendered legend carry "no_legend": true and a placeholder legend box [0,0,1,1] (the shared
runner validates a legend box; a 1-pixel box at the image corner removes no plot point). For them only:
  detection  run_A4_auto_v46.py ... --mode <bw|color> --plot-area <box> --no-legend   (v46's optional-legend
             single-series route; v46 abstains when the source shows competing series)
  legends    no observed legend crops; shared_legends.json records status 'no_legend' and an empty mapping, so
             their points stay 'unmapped_v46_S01' (position-only scoring decides single-series charts; on a
             multi-series chart they cannot carry a legend entry).
Everything else is run_parallel.py unchanged (snapshot-pinned CLIs, GUI-equivalent Step 5, timing).
Usage: python run_astra100.py --inputs inputs_astra100_run.json --results results/run_astra100_<hash> --snapshot ... [run_parallel args]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v46_v2 as V2  # noqa: E402  (patched benchmark_pk/run_v46.py)

R46 = V2.R46
_detection = R46.detection_command
_export = R46.export_legends


def detection_command(panel, destination, python):
    if panel.get('no_legend'):
        return [str(python), str(R46.SRC / 'run_A4_auto_v46.py'), panel['image_path'], str(destination),
                '--mode', panel['modality'], '--plot-area', R46.csv_box(panel['plot_bbox']), '--no-legend']
    return _detection(panel, destination, python)


def export_legends(panel, before, destination, image):
    if panel.get('no_legend'):
        value = dict(panel_id=panel['panel_id'], modality=panel['modality'], entries=[], mapping={},
                     legend_mapping_audit=[], annotations=[], input_sha256=panel['input_sha256'],
                     source_sha256=panel['source_sha256'], status='no_legend',
                     protocol='Chart has no legend; v46 run with --no-legend.', export_seconds=0.0)
        R46.save(destination / 'shared_legends.json', value)
        return value
    return _export(panel, before, destination, image)


R46.detection_command = detection_command
R46.export_legends = export_legends

if __name__ == '__main__':
    import run_parallel
    sys.argv = ['run_parallel.py', *sys.argv[1:]]
    run_parallel.main()

"""v46 colour CLI: composed legend, hybrid v2 and tentative Step-5 evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv=None):
    # Compatible with both direct python execution and runpy dispatch from GUI.
    forwarded = list(sys.argv[1:] if argv is None else argv)
    from legend_optional_v46.cli import maybe_run
    if maybe_run(forwarded):return
    correction_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    correction_parser.add_argument('--color-series-mode', choices=('auto','markers','line-only'), default='auto')
    correction_parser.add_argument('--step5')
    correction_parser.add_argument('--step5-iters', type=int, default=1)
    correction_parser.add_argument('--step5-metric',choices=('chamfer','directional_chamfer','hausdorff95'),default='chamfer')
    correction_parser.add_argument('--step5-model',choices=('linear','pchip'),default=None)
    correction_parser.add_argument('--step5-reference',choices=('observed','estimated_path'),default='estimated_path')
    correction_parser.add_argument('--step5-inferred-weight',type=float,default=.25)
    correction, forwarded = correction_parser.parse_known_args(forwarded)
    # Auto has already established a marker-bearing legend before declining
    # optional routing. Only the established marker detector reaches this block.
    if correction.color_series_mode=='auto':correction.color_series_mode='markers'
    if correction.step5:
        if correction.step5_iters < 1:
            correction_parser.error('--step5-iters must be a positive integer')
        if not (Path(correction.step5) / '5_correction_color.py').is_file():
            correction_parser.error('--step5 DIR must contain 5_correction_color.py')
    directory = str(Path(__file__).resolve().parent)
    prior_path, prior_argv = sys.path[:], sys.argv
    try:
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from color_legend_runtime_v46 import LegendRuntime
        from color_marker_runtime_v46 import MarkerRuntime
        from color_pipeline_v46 import run
        runtime = LegendRuntime(allow_line_only=correction.color_series_mode == 'line-only')
        if correction.color_series_mode == 'line-only':
            print('[v46 type3] Marker-free colour path/centre pipeline enabled, including '
                  'strong suppressed outer endpoints. Standalone v46 analysis and export.', flush=True)
        else:
            print('[v46 color] Legend composition + colour marker hybrid v2 + tentative '
                  'Step-5 evidence enabled. Standalone v46 analysis and export.', flush=True)
        marker_runtime = MarkerRuntime(series_mode=correction.color_series_mode)
        namespace = run(forwarded, runtime, marker_runtime)
        if getattr(marker_runtime, 'group_backend', False) is True:
            from color_group_runtime_v46 import finalize_outputs
            finalize_outputs(namespace)
        if correction.color_series_mode == 'line-only':
            from type3_v46.runtime import finalize_outputs
            finalize_outputs(marker_runtime, namespace)
        if correction.step5:
            # Correction consumes the v46 evidence exported by detection.
            from color_correct_cli import main as correct_main
            arguments = [str(namespace['IMG_PATH']), str(namespace['OUT_DIR']),
                         '--plot-area', ','.join(str(int(v)) for v in namespace['PLOT_AREA']),
                         '--correct-iters', str(correction.step5_iters),
                         '--engine-dir', str(Path(correction.step5).resolve()),
                         '--require-v46-path','--path-metric',correction.step5_metric,
                         '--path-reference',correction.step5_reference,
                         '--path-inferred-weight',str(correction.step5_inferred_weight)]
            if correction.step5_model is not None:
                arguments += ['--path-model',correction.step5_model]
            if namespace.get('LEGEND_BOX') is not None:
                arguments += ['--legend-area', ','.join(str(int(v)) for v in namespace['LEGEND_BOX'])]
            print('[v46 color] Running path-shape Step 5 with P0 protection and tentative hypotheses.', flush=True)
            correct_main(arguments)
    finally:
        sys.path[:] = prior_path
        sys.argv = prior_argv


if __name__ == '__main__':
    main()

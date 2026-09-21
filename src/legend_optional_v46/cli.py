"""Optional-legend entry shared by the v46 command line and web GUI.

No legacy runner is imported until this adapter explicitly declines the route.
An absent legend supplies the one-series assumption; it is not auto-detection
of the number of series. Uncertain evidence is a failed detection, not fallback.
"""
import argparse
import json
from pathlib import Path

import cv2

from . import runtime


def _roi(value):
    if value is None or not value.strip():return None
    try:
        numbers=[float(v) for v in value.split(',')]
        if len(numbers)!=4 or any(not v.is_integer() for v in numbers):raise ValueError()
        return [int(v) for v in numbers]
    except (ValueError,OverflowError):
        raise argparse.ArgumentTypeError('ROI must contain four finite integer pixels')


def parser(default_mode):
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('image',nargs='?');p.add_argument('out_dir',nargs='?')
    p.add_argument('--mode',choices=('color','bw'),default=default_mode)
    p.add_argument('--no-legend',action='store_true')
    p.add_argument('--legend-area','--legend-box',type=_roi)
    p.add_argument('--plot-area',type=_roi)
    p.add_argument('--color-series-mode',choices=('auto','markers','line-only'),default='auto')
    p.add_argument('--swatches-json',default='')
    for name in ('x-min','x-max','y-min','y-max'):p.add_argument('--'+name,type=float)
    p.add_argument('--x-log',action='store_true');p.add_argument('--y-log',action='store_true')
    p.add_argument('--correct',action='store_true');p.add_argument('--correct-iters',type=int,default=5)
    p.add_argument('--prev-state')
    p.add_argument('--step5');p.add_argument('--step5-iters',type=int,default=5)
    p.add_argument('--step5-metric',default='chamfer');p.add_argument('--step5-model',default=None)
    p.add_argument('--step5-reference',default='estimated_path');p.add_argument('--step5-inferred-weight',default='.25')
    return p


def _bounds(a,axis):
    lo,hi=getattr(a,axis+'_min'),getattr(a,axis+'_max')
    if (lo is None)!=(hi is None):raise ValueError(f'Both {axis}-min and {axis}-max must be supplied together')
    return None if lo is None else [lo,hi]


def maybe_run(argv, *, default_mode='color'):
    """Return False for established legend routes; execute other routes fully."""
    argv=list(argv)
    if '--help' in argv or '-h' in argv:return False
    a,unused=parser(default_mode).parse_known_args(argv)
    if a.no_legend and (a.legend_area is not None or a.swatches_json):
        raise ValueError('--no-legend cannot be combined with a legend ROI or swatches')
    if a.swatches_json:return False
    if a.legend_area is not None and (a.mode=='bw' or a.color_series_mode in ('markers','line-only')):
        return False
    if not a.image or not a.out_dir:raise ValueError('Image and output directory are required')
    if a.plot_area is None:raise ValueError('Select the plot area before v46 optional-legend detection')
    image=cv2.imread(a.image)
    if image is None:raise ValueError('Cannot read source image')
    plot=runtime.box(a.plot_area,image.shape)
    probe=None
    if a.legend_area is not None:
        legend=runtime.box(a.legend_area,image.shape)
        probe=runtime.probe_legend(image,plot,legend)
        if probe['kind']=='marker_legend':return False
    if a.correct_iters<1 or a.step5_iters<1:raise ValueError('Correction iterations must be positive')
    result=runtime.run_file(a.image,a.out_dir,a.plot_area,a.legend_area,mode=a.mode,
        series_mode=a.color_series_mode,probe=probe,x_range=_bounds(a,'x'),y_range=_bounds(a,'y'),
        x_log=a.x_log,y_log=a.y_log)
    if result['status']!='completed':
        raise RuntimeError('v46 abstained: '+result['reason']+'; see legend_optional_v46.json')
    if a.mode=='bw':
        runtime.finish_bw(a.image,a.out_dir,result,correct=a.correct,
            iterations=a.correct_iters,previous_path=a.prev_state)
    if a.mode=='color' and (a.step5 or a.correct):
        if result.get('correction_available') is False:
            raise RuntimeError(result.get('correction_unavailable_reason') or
                               'Colour correction is unavailable for this detection evidence')
        from color_correct_cli import main as correct_main
        args=[a.image,a.out_dir,'--plot-area',','.join(map(str,a.plot_area)),
            '--correct-iters',str(a.step5_iters if a.step5 else a.correct_iters),'--require-v46-path',
            '--path-metric',a.step5_metric,'--path-reference',a.step5_reference,
            '--path-inferred-weight',a.step5_inferred_weight]
        if a.step5_model is not None:args+=['--path-model',a.step5_model]
        if a.legend_area is not None:args+=['--legend-area',','.join(map(str,a.legend_area))]
        if a.step5:args+=['--engine-dir',a.step5]
        if a.prev_state:args+=['--prev-state',a.prev_state]
        correct_main(args)
    return True

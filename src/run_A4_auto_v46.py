"""v46 shared entry: colour hybrid, legend-grid BW, or optional single series.

python run_A4_auto_v46.py IMAGE OUT --mode color [colour options]
python run_A4_auto_v46.py IMAGE OUT --mode bw --plot-area ... --legend-area ...
python run_A4_auto_v46.py IMAGE OUT --mode bw --plot-area ... --no-legend

Selected legends keep hybrid extraction; an absent legend explicitly infers one
series from source pixels and never invokes the legacy no-legend detectors.
It remains the default. This file intentionally does not
import either pipeline until after selecting the mode.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def dispatch_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=('color', 'bw'), default='color')
    args, forwarded = parser.parse_known_args(argv)
    if args.mode == 'bw':
        # v46 must never silently select the old trained point detector.
        if any(a == '--point-backend' or a.startswith('--point-backend=') for a in forwarded):
            raise ValueError('v46 B&W always uses grid_v46; use bw_detect_cli.py for legacy ViT')
        forwarded += ['--point-backend', 'grid_v46']
        return 'bw_detect_cli.py', forwarded
    return 'run_A4_color_v46.py', forwarded


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    target, forwarded = dispatch_args(arguments)
    if target=='bw_detect_cli.py':
        from legend_optional_v46.cli import maybe_run
        if maybe_run(arguments,default_mode='bw'):return
    script = Path(__file__).with_name(target)
    previous = sys.argv
    try:
        sys.argv = [str(script), *forwarded]
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()

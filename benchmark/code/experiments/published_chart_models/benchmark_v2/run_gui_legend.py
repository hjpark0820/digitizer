"""run_astra100.py / run_parallel.py with the B&W detection command reduced to what the web GUI sends.

Why: benchmark_pk/run_v46.py::detection_command passes the benchmark's annotated legend-key boxes to the
B&W CLI (--swatches-json). With that option bw_legend_v46.extract_legend_models skips its automatic key
discovery (_automatic_boxes) inside the legend rectangle. The web GUI (unified_server.py, B&W branch) never
sends --swatches-json: a user gives the plot area and the legend rectangle only. So B&W benchmark numbers
measured with --swatches-json are not the GUI condition.

Change: for B&W panels the '--swatches-json <value>' pair is removed from the command that the unchanged
runner builds. The command is then asserted to carry exactly the GUI's B&W option set
(--mode bw, --plot-area, --legend-area, --skip-segments). Colour panels are unchanged (they already get only
--plot-area and --legend-box). Everything else - snapshot-pinned CLIs, legend export, GUI Step 5, timing -
is run_parallel.py unchanged.

Usage: same arguments as run_parallel.py, e.g.
  python run_gui_legend.py --inputs inputs_test100v2_20261025.json --results results/<run> \
      --snapshot src_snapshots/0fd678a840bc/src --workers 7 --gpu-slots 7 --threads 4 --ids <B&W ids>
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_astra100  # noqa: E402,F401  (patches benchmark_pk/run_v46.py like the original run)

R46 = run_astra100.R46
_previous = R46.detection_command
GUI_BW_FLAGS = {'--mode', '--plot-area', '--legend-area', '--skip-segments'}


def detection_command(panel, destination, python):
    command = list(_previous(panel, destination, python))
    if panel['modality'] != 'bw':
        return command
    if '--swatches-json' in command:
        i = command.index('--swatches-json')
        del command[i:i + 2]
    flags = {a for a in command if a.startswith('--')}
    if flags != GUI_BW_FLAGS or command[command.index('--mode') + 1] != 'bw':
        raise ValueError(f'B&W command is not the GUI option set: {sorted(flags)}')
    return command


R46.detection_command = detection_command

if __name__ == '__main__':
    import run_parallel
    sys.argv = ['run_parallel.py', *sys.argv[1:]]
    run_parallel.main()

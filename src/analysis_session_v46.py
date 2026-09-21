"""Isolate the image state used by shared analysis helpers.

The numerical helpers historically use module globals for image arrays and
palette state. Each session loads only the import-safe v46 helper library into
its own module. Binding an ordinary stage function to that namespace lets the
helpers and stages share one image without leaking state between requests.
No CLI script is imported and no source is parsed or transformed at runtime.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import FunctionType


class AnalysisSession:
    def __init__(self):
        path = Path(__file__).with_name('chart_analysis_v46.py')
        spec = spec_from_file_location('_chart_analysis_session_v46', path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        self.values = vars(module)

    def install(self, stage):
        bound = FunctionType(stage.__code__, self.values, stage.__name__,
                             stage.__defaults__, stage.__closure__)
        bound.__kwdefaults__ = stage.__kwdefaults__
        self.values[stage.__name__] = bound
        return bound

    def run(self, stage, *args, **kwargs):
        return self.install(stage)(*args, **kwargs)

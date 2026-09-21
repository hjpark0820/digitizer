"""Shared, lazy CPU/CUDA selection for B&W v46 entry points.

Automatic selection checks that CUDA can actually allocate and execute a tiny
operation, not just that a PyTorch CUDA build is installed. Explicit CPU never
imports torch. Explicit CUDA remains strict and is never silently changed to
CPU. No availability result is cached across independent resolutions.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import time


BACKENDS=frozenset(('auto','cpu','cuda'))
BACKEND_ENV='BW_V46_REFINEMENT_BACKEND'


class CudaBackendUnavailable(RuntimeError):
    """An explicitly required CUDA runtime could not pass its usable probe."""
    def __init__(self,probe):
        self.probe=dict(probe)
        super().__init__(f"CUDA is unavailable ({probe.get('reason','unknown')}): {probe.get('detail','')}")


def _load_torch():
    """Single patchable import boundary; importing this module is CPU-only."""
    return importlib.import_module('torch')


def probe_cuda_available():
    """Return a fresh usable-CUDA report; never raise ordinary probe failures.

    CUDA:0 is the device used by the accelerated v46 helpers. The check includes
    allocation, one arithmetic kernel, a scalar readback, and synchronization.
    KeyboardInterrupt/SystemExit are intentionally not swallowed.
    """
    started=time.perf_counter()
    report=dict(available=False,reason='not_probed',detail='',device=None,
                device_index=0,torch_version=None,cuda_version=None,error_type=None)
    phase='torch_import'
    try:
        torch=_load_torch()
        report['torch_version']=str(getattr(torch,'__version__','unknown'))
        report['cuda_version']=getattr(getattr(torch,'version',None),'cuda',None)
        phase='availability'
        if not torch.cuda.is_available():
            report.update(reason='cuda_not_available',
                          detail='PyTorch reports no usable CUDA device or CUDA-enabled runtime.')
            return report
        phase='initialization'
        torch.cuda.init()
        phase='allocation_or_compute'
        with torch.inference_mode():
            sample=torch.ones((1,),device='cuda:0',dtype=torch.float32)
            actual=(sample+1.0).item()
            if actual!=2.0:
                raise RuntimeError(f'CUDA arithmetic probe returned {actual!r}, expected 2.0')
            torch.cuda.synchronize(0)
        # Device-name metadata is useful but not an additional usability gate.
        # A successful allocation/kernel/readback already establishes usability.
        try:
            report['device']=str(torch.cuda.get_device_name(0))
        except Exception:
            report['device']='CUDA device 0'
        report.update(available=True,reason='cuda_ready',
                      detail='CUDA:0 allocation, arithmetic and synchronization succeeded.')
        return report
    except Exception as error:
        if phase=='torch_import':
            missing=isinstance(error,ModuleNotFoundError) and getattr(error,'name',None)=='torch'
            reason='torch_missing' if missing else 'torch_import_failed'
        else:
            reason={'availability':'cuda_availability_check_failed',
                    'initialization':'cuda_init_failed',
                    'allocation_or_compute':'cuda_allocation_or_compute_failed'}[phase]
        report.update(available=False,reason=reason,error_type=type(error).__name__,detail=str(error))
        return report
    finally:
        report['seconds']=time.perf_counter()-started


def ensure_cuda_available():
    """Strict, main-thread-friendly preload for an explicit CUDA entry point."""
    report=probe_cuda_available()
    if not report.get('available',False):
        raise CudaBackendUnavailable(report)
    return report


def _backend(value,label):
    if not isinstance(value,str) or value not in BACKENDS:
        raise ValueError(f'{label} must be auto, cpu or cuda; got {value!r}')
    return value


def resolve_compute_backends(refinement_backend=None,grid_backend=None,window_backend=None,
                             *,environ=None,log_fn=None):
    """Resolve requested/selected backends without importing the detector.

    Base None reads BW_V46_REFINEMENT_BACKEND, defaulting to auto. A stage None
    inherits the *requested* base policy. Every auto stage shares one probe;
    explicit cuda/cpu stages preserve the user's choice regardless of its result.
    Explicit CUDA-only calls may use ensure_cuda_available() for eager preload;
    otherwise their normal strict execution path performs the device checks.
    """
    environment=os.environ if environ is None else environ
    base=_backend(environment.get(BACKEND_ENV,'auto') if refinement_backend is None
                  else refinement_backend,'refinement_backend')
    requested=dict(refinement=base,
                   grid=_backend(base if grid_backend is None else grid_backend,'grid_backend'),
                   window=_backend(base if window_backend is None else window_backend,'window_backend'))
    if 'auto' in requested.values():
        probe=probe_cuda_available()
        automatic='cuda' if probe.get('available',False) else 'cpu'
    else:
        explicit_cuda='cuda' in requested.values()
        probe=dict(available=None,reason='explicit_cuda_unprobed' if explicit_cuda else 'not_requested',
                   detail='Explicit CUDA remains strict; runtime/preload checks are deferred.' if explicit_cuda
                   else 'All stages explicitly select CPU; PyTorch was not imported.',
                   device=None,device_index=None,seconds=0.)
        automatic=None
    selected={stage:(automatic if backend=='auto' else backend)
              for stage,backend in requested.items()}
    result=dict(requested=requested,selected=selected,cuda_probe=probe)
    if log_fn is not None:
        log_fn('[v46 compute] requested '+', '.join(f'{key}={value}' for key,value in requested.items())+
               '; selected '+', '.join(f'{key}={value}' for key,value in selected.items()))
        if 'auto' in requested.values():
            log_fn(f"[v46 compute] auto CUDA probe: {probe.get('reason')}; {probe.get('detail','')}")
    return result


def is_cuda_runtime_error(error):
    """Recognize GPU-runtime/known-helper failures, not arbitrary data errors.

    Already-loaded helper classes are checked by actual isinstance, without
    importing torch or GPU modules. Generic MemoryError/ValueError and CPU
    allocator failures are not CUDA failures. The explicit exception cause of
    a RuntimeError wrapper may preserve an identifiable CUDA failure.
    """
    if isinstance(error,CudaBackendUnavailable):
        return True
    for module_name,class_name in (
            ('bw_gpu_refinement','GpuRefinementUnavailable'),
            ('bw_gpu_grid_votes','GpuGridVotesUnavailable'),
            ('bw_gpu_window_verifier','_Uncertifiable')):
        module=sys.modules.get(module_name)
        cls=getattr(module,class_name,None) if module is not None else None
        if isinstance(cls,type) and isinstance(error,cls):
            return True
    cuda_module=sys.modules.get('torch.cuda')
    oom=getattr(cuda_module,'OutOfMemoryError',None) if cuda_module is not None else None
    if isinstance(oom,type) and isinstance(error,oom):
        return True
    text=str(error).lower()
    # These tokens identify CUDA subsystem failures, including status names
    # such as CUDNN_STATUS_ALLOC_FAILED. Merely containing 'out of memory' is
    # insufficient: CPU allocation failures must not trigger GPU fallback.
    gpu_token=re.search(r'cuda|cudnn|cublas|cusolver|cusparse|curand|cufft|nvrtc|nvidia driver',text)
    if isinstance(error,RuntimeError):
        if gpu_token or ('gpu' in text and ('out of memory' in text or 'allocation' in text)):
            return True
        cause=getattr(error,'__cause__',None)
        if cause is not None and cause is not error:
            # Follow only a short acyclic explicit cause chain. Never turn an
            # unrelated ValueError wrapper into an automatically retried job.
            seen={id(error)}
            for _ in range(8):
                if id(cause) in seen:
                    break
                seen.add(id(cause))
                if not isinstance(cause,RuntimeError):
                    break
                if re.search(r'cuda|cudnn|cublas|cusolver|cusparse|curand|cufft|nvrtc|nvidia driver',str(cause).lower()):
                    return True
                cause=getattr(cause,'__cause__',None)
                if cause is None:
                    break
        return False
    if isinstance(error,(ImportError,OSError)):
        # DLL failures in a named CUDA/PyTorch CUDA library are runtime issues;
        # a missing user file or generic DLL failure is not classified as one.
        return bool(gpu_token and re.search(r'dll|library|cannot load|could not load|not found',text))
    if isinstance(error,AssertionError):
        return 'torch not compiled with cuda enabled' in text
    return False

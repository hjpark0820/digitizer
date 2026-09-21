"""Sparse fused CUDA ranks for certified B&W window preselection.

Each block evaluates one independent window/alignment pair. Only active shape
pixels are visited; no pair-by-height-by-width intermediate arrays exist.
Ranks are float64 approximations enclosed by the unchanged verifier bounds;
the caller still recomputes every competitive score with original NumPy code.
NVRTC is loaded from the existing PyTorch installation, never installed here.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np


class FusedWindowUnavailable(RuntimeError):
    pass


_SOURCE = r'''
extern "C" __global__ void sparse_window_ranks(
    const float *observed, const float *nearby, int window_area, int windows,
    const int *shape_offsets, const int *pixel_offsets, const float *weights,
    const float *expected, const unsigned char *flags,
    const int *shape_ids, const int *origins, const double *penalties,
    const double *denominators, const double *boundary_denominators,
    const int *core_counts, const unsigned char *has_boundary,
    long long work_start, long long work_stop, double *ranks) {
    long long work = work_start + blockIdx.x;
    if (work >= work_stop) return;
    int alignment = (int)(work / windows), window = (int)(work % windows);
    int shape = shape_ids[alignment];
    long long origin = (long long)window * window_area + origins[alignment];
    double weighted = 0., core = 0., boundary = 0.;
    for (int p = shape_offsets[shape] + threadIdx.x;
         p < shape_offsets[shape + 1]; p += blockDim.x) {
        long long at = origin + pixel_offsets[p];
        double e = (double)expected[p];
        double support = fmin((double)observed[at] / e, 1.);
        unsigned char kind = flags[p];
        if (kind & 2) {
            double neighbor = 0.85 * fmin((double)nearby[at] / e, 1.);
            support = fmax(support, neighbor);
        }
        double ink = support * (double)weights[p];
        weighted += ink;
        if (kind & 1) core += support;
        if (kind & 2) boundary += ink;
    }
    __shared__ double sums[3][128];
    sums[0][threadIdx.x] = weighted;
    sums[1][threadIdx.x] = core;
    sums[2][threadIdx.x] = boundary;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[0][threadIdx.x] += sums[0][threadIdx.x + stride];
            sums[1][threadIdx.x] += sums[1][threadIdx.x + stride];
            sums[2][threadIdx.x] += sums[2][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double required = sums[0][0] / denominators[shape];
        double core_recall = core_counts[shape] ? sums[1][0] / core_counts[shape] : 1.;
        double boundary_recall = has_boundary[shape] ?
            sums[2][0] / boundary_denominators[shape] : 1.;
        double score = .85 * required + .10 * core_recall + .05 * boundary_recall;
        ranks[work] = score - penalties[alignment] - .30 * fmax(.85 - core_recall, 0.);
    }
}
'''


class _Kernel:
    def __init__(self, torch, device):
        self.module = self.module_unload = self.directory_handle = None
        try:
            self._initialize(torch, device)
        except BaseException:
            self.close()
            raise

    def _initialize(self, torch, device):
        # Reuse the platform-specific library discovery/signatures, not the
        # centre-clustering kernel or its numerical algorithm.
        from bw_gpu_candidate_centres import _library_paths, _set_signature
        self.torch, self.device = torch, device
        if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
            self.directory_handle = os.add_dll_directory(str(Path(torch.__file__).resolve().parent/'lib'))
        failures = []
        self.nvrtc = None
        for name in _library_paths(torch):
            try:
                self.nvrtc = ctypes.CDLL(name)
                break
            except OSError as error:
                failures.append(str(error))
        if self.nvrtc is None:
            raise FusedWindowUnavailable('CUDA NVRTC is unavailable: '+'; '.join(failures))
        p = ctypes.c_void_p
        pp, cp = ctypes.POINTER(p), ctypes.POINTER(ctypes.c_char_p)
        create = _set_signature(self.nvrtc, 'nvrtcCreateProgram', [pp, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, cp, cp])
        compile_program = _set_signature(self.nvrtc, 'nvrtcCompileProgram', [p, ctypes.c_int, cp])
        log_size = _set_signature(self.nvrtc, 'nvrtcGetProgramLogSize', [p, ctypes.POINTER(ctypes.c_size_t)])
        get_log = _set_signature(self.nvrtc, 'nvrtcGetProgramLog', [p, p])
        ptx_size = _set_signature(self.nvrtc, 'nvrtcGetPTXSize', [p, ctypes.POINTER(ctypes.c_size_t)])
        get_ptx = _set_signature(self.nvrtc, 'nvrtcGetPTX', [p, p])
        destroy = _set_signature(self.nvrtc, 'nvrtcDestroyProgram', [pp])
        program = p()
        self._check(create(ctypes.byref(program), _SOURCE.encode(), b'bw_window_sparse.cu', 0, None, None), 'nvrtcCreateProgram')
        major, minor = torch.cuda.get_device_capability(device)
        options = [f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--std=c++14',
                   b'--fmad=false', b'--ftz=false', b'--prec-div=true']
        try:
            result = compile_program(program, len(options), (ctypes.c_char_p*len(options))(*options))
            if result:
                size = ctypes.c_size_t()
                log_size(program, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                get_log(program, log)
                raise FusedWindowUnavailable('CUDA fused window compilation failed: '+log.value.decode(errors='replace'))
            size = ctypes.c_size_t()
            self._check(ptx_size(program, ctypes.byref(size)), 'nvrtcGetPTXSize')
            ptx = ctypes.create_string_buffer(size.value)
            self._check(get_ptx(program, ptx), 'nvrtcGetPTX')
        finally:
            destroy(ctypes.byref(program))
        self.driver = ctypes.CDLL('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
        load = _set_signature(self.driver, 'cuModuleLoadData', [pp, p])
        self.module_unload = _set_signature(self.driver, 'cuModuleUnload', [p])
        function = _set_signature(self.driver, 'cuModuleGetFunction', [pp, p, ctypes.c_char_p])
        self.launch = _set_signature(self.driver, 'cuLaunchKernel',
            [p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
             ctypes.c_uint, ctypes.c_uint, p, pp, pp])
        self.module, self.function = p(), p()
        self._check(load(ctypes.byref(self.module), ptx), 'cuModuleLoadData')
        self._check(function(ctypes.byref(self.function), self.module, b'sparse_window_ranks'), 'cuModuleGetFunction')

    @staticmethod
    def _check(result, operation):
        if result:
            raise FusedWindowUnavailable(f'CUDA fused window {operation} failed with code {result}')

    def close(self):
        module, self.module = self.module, None
        if module is not None and self.module_unload is not None:
            try:
                self.module_unload(module)
            except BaseException:
                pass
        handle, self.directory_handle = self.directory_handle, None
        if handle is not None:
            try:
                handle.close()
            except BaseException:
                pass

    def __del__(self):
        self.close()

    def run(self, observed, nearby, pack, output, start, stop):
        p = ctypes.c_void_p
        values = [p(observed.data_ptr()), p(nearby.data_ptr()), ctypes.c_int(pack['window_size']**2),
                  ctypes.c_int(observed.shape[0])]
        keys = ('shape_offsets', 'pixel_offsets', 'weights', 'expected', 'flags', 'shape_ids',
                'origins', 'penalties', 'denominators', 'boundary_denominators', 'core_counts', 'has_boundary')
        values += [p(pack[key].data_ptr()) for key in keys]
        values += [ctypes.c_longlong(start), ctypes.c_longlong(stop), p(output.data_ptr())]
        pointers = (p*len(values))(*(ctypes.cast(ctypes.byref(value), p) for value in values))
        stream = p(self.torch.cuda.current_stream(observed.device).cuda_stream)
        self._check(self.launch(self.function, stop-start, 1, 1, 128, 1, 1, 0, stream, pointers, None), 'cuLaunchKernel')


@lru_cache(maxsize=8)
def _kernel_for_device(device):
    import torch
    with torch.cuda.device(device):
        torch.cuda.init()
        torch.empty(1, device=f'cuda:{device}')
        try:
            return _Kernel(torch, device)
        except (OSError, AttributeError) as error:
            raise FusedWindowUnavailable('CUDA fused window runtime unavailable: '+str(error)) from error


def sparse_arrays(usable, alignments, window_size):
    """CPU packing only: retain zero-weight core pixels in the core mean."""
    from bw_gpu_window_verifier import _rank_error_bound
    offsets = [0]
    pixels, weights, expected, flags = [], [], [], []
    errors, denominators, boundary_denominators, cores, boundaries = [], [], [], [], []
    for _, _, mask, weight, expect, core, boundary in usable:
        active = (weight > 0.) | core
        yy, xx = np.nonzero(active)
        pixels.append((yy*window_size+xx).astype(np.int32))
        weights.append(weight[active]); expected.append(expect[active])
        flags.append(core[active].astype(np.uint8)+2*boundary[active].astype(np.uint8))
        offsets.append(offsets[-1]+len(xx))
        error, denominator, boundary_denominator, _ = _rank_error_bound(weight, core, boundary)
        errors.append(error); denominators.append(denominator)
        boundary_denominators.append(boundary_denominator)
        cores.append(int(core.sum())); boundaries.append(bool(boundary.any()))
    if offsets[-1] > np.iinfo(np.int32).max:
        raise FusedWindowUnavailable('CUDA sparse template offsets exceed int32 capacity')
    return dict(shape_offsets=np.asarray(offsets, np.int32), pixel_offsets=np.concatenate(pixels),
        weights=np.concatenate(weights), expected=np.concatenate(expected), flags=np.concatenate(flags),
        shape_ids=np.asarray([a[0] for a in alignments], np.int32),
        origins=np.asarray([a[4]*window_size+a[3] for a in alignments], np.int32),
        penalties=np.asarray([a[5] for a in alignments], np.float64),
        denominators=np.asarray(denominators, np.float64),
        boundary_denominators=np.asarray(boundary_denominators, np.float64),
        core_counts=np.asarray(cores, np.int32), has_boundary=np.asarray(boundaries, np.uint8),
        errors=np.asarray([errors[a[0]] for a in alignments], np.float64))


def prepare_resources(torch, usable, alignments, window_size):
    arrays = sparse_arrays(usable, alignments, window_size)
    pack = {key: torch.as_tensor(np.ascontiguousarray(value), device='cuda:0')
            for key, value in arrays.items() if key != 'errors'}
    pack.update(errors=arrays['errors'], window_size=window_size,
        active_pixels=int(arrays['weights'].size),
        full_shape_pixels=sum(shape[2].size for shape in usable),
        max_area=max(shape[2].size for shape in usable),
        sparse_resource_bytes=sum(value.nbytes for value in arrays.values()))
    return pack


def rank_windows(torch, memberships, nearby_windows, pack, batch_size):
    """Float64 ranks and actual bounded launch diagnostics; one rank readback."""
    kernel = _kernel_for_device(0)
    observed = torch.as_tensor(np.stack(memberships), device='cuda:0', dtype=torch.float32)
    nearby = torch.as_tensor(np.stack(nearby_windows), device='cuda:0', dtype=torch.float32)
    pairs = len(pack['errors'])*len(memberships)
    ranks = torch.empty(pairs, device='cuda:0', dtype=torch.float64)
    # Preserve the public batch-size maximum exactly. Unlike the tensor path,
    # large marker rasters no longer force that maximum down to 18-42 pairs.
    # Each block uses three fixed shared reduction arrays (3072 bytes).
    launch_cap = min(65535, batch_size)
    launches = 0
    for start in range(0, pairs, launch_cap):
        kernel.run(observed, nearby, pack, ranks, start, min(pairs, start+launch_cap))
        launches += 1
    output = ranks.cpu().numpy().reshape(len(pack['errors']), len(memberships)).T
    stats = dict(rank_engine='nvrtc_sparse_fused', fused_kernel_launches=launches,
        fused_pairs_per_launch=launch_cap, fused_shared_bytes_per_block=3072,
        fused_active_shape_pixels=pack['active_pixels'], fused_full_shape_pixels=pack['full_shape_pixels'],
        fused_sparse_resource_bytes=pack['sparse_resource_bytes'],
        fused_pair_raster_intermediate_bytes=0, fused_fallback_reason=None,
        rank_launch_policy='min(65535,batch_size) independent pairs; no raster-area-dependent reduction')
    return output, launch_cap, launches, stats

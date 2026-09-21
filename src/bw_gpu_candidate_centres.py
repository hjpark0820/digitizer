"""Exact CUDA aggregates for B&W grid-hypothesis centre clustering.

The spatial tree, its original neighbour order, stable density ranking, greedy
NMS and Gaussian vote-map rendering stay on CPU. Independent weighted sums and
distinct-cell counts run in bounded CUDA batches. A tiny NVRTC kernel preserves
NumPy's float32 reduction order; a generic torch.sum would change tie ordering.
PyTorch and the CUDA runtime compiler are loaded only on an actual CUDA call.
No system CUDA toolkit, nvcc executable or additional Python package is needed
when PyTorch ships its usual NVRTC runtime library.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from bw_gpu_refinement import GpuRefinementUnavailable


class GpuCandidateCentresUnavailable(GpuRefinementUnavailable):
    """The installed CUDA stack cannot provide certified centre aggregates."""


_CUDA_SOURCE = r'''
struct Pair { float weight; float clipped; };
__device__ __forceinline__ Pair add_pair(Pair a, Pair b) {
    return {__fadd_rn(a.weight, b.weight), __fadd_rn(a.clipped, b.clipped)};
}
__device__ __forceinline__ Pair value_at(
        const int *indices, const float *scores, int position) {
    float value = scores[indices[position]];
    return {fmaxf(value, 0.0001f), fminf(fmaxf(value, 0.0f), 2.0f)};
}
// NumPy's real-float pairwise sum uses eight lanes and a 128-element leaf.
// An explicit stack avoids recursive-device linking and has depth < 32 for
// every supported (signed int32) neighbour count.
__device__ Pair pairwise(const int *indices, const float *scores, int count) {
    int starts[32], sizes[32], state[32];
    Pair saved[32];
    int start = 0, size = count, depth = 0;
    Pair result;
    while (true) {
        if (size > 128) {
            int half = (size / 2) - (size / 2) % 8;
            starts[depth] = start + half;
            sizes[depth] = size - half;
            state[depth] = 0;
            ++depth;
            size = half;
            continue;
        }
        if (size < 8) {
            result = {-0.0f, -0.0f};
            for (int i = 0; i < size; ++i)
                result = add_pair(result, value_at(indices, scores, start + i));
        } else {
            Pair lanes[8];
            for (int lane = 0; lane < 8; ++lane)
                lanes[lane] = value_at(indices, scores, start + lane);
            int i = 8;
            for (; i < size - size % 8; i += 8)
                for (int lane = 0; lane < 8; ++lane)
                    lanes[lane] = add_pair(lanes[lane],
                        value_at(indices, scores, start + i + lane));
            result = add_pair(add_pair(add_pair(lanes[0], lanes[1]),
                                      add_pair(lanes[2], lanes[3])),
                              add_pair(add_pair(lanes[4], lanes[5]),
                                      add_pair(lanes[6], lanes[7])));
            for (; i < size; ++i)
                result = add_pair(result, value_at(indices, scores, start + i));
        }
        bool next_right = false;
        while (depth) {
            int parent = depth - 1;
            if (state[parent] == 0) {
                saved[parent] = result;
                state[parent] = 1;
                start = starts[parent];
                size = sizes[parent];
                next_right = true;
                break;
            }
            result = add_pair(saved[parent], result);
            --depth;
        }
        if (!next_right) return result;
    }
}
extern "C" __global__ void aggregate_centres(
        const float *points, const float *scores, const int *indices,
        const int *offsets, int rows, float *out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int start = offsets[row], end = offsets[row + 1];
    float x = 0.0f, y = 0.0f;
    for (int position = start; position < end; ++position) {
        int index = indices[position];
        float weight = fmaxf(scores[index], 0.0001f);
        // np.average first multiplies float32 arrays, then sums axis 0 of an
        // N x 2 C-contiguous array sequentially. FMA must not combine these.
        x = __fadd_rn(x, __fmul_rn(points[2 * index], weight));
        y = __fadd_rn(y, __fmul_rn(points[2 * index + 1], weight));
    }
    Pair sums = pairwise(indices + start, scores, end - start);
    out[3 * row] = __fdiv_rn(x, sums.weight);
    out[3 * row + 1] = __fdiv_rn(y, sums.weight);
    out[3 * row + 2] = sums.clipped;
}
'''


def _library_paths(torch):
    """Only inspect conventional runtime locations, never install packages."""
    torch_root = Path(torch.__file__).resolve().parent
    if os.name == 'nt':
        paths = sorted((torch_root / 'lib').glob('nvrtc64_*.dll'))
        return [str(p) for p in paths if '.alt.' not in p.name] + ['nvrtc64_120_0.dll']
    candidates = []
    for folder in (torch_root / 'lib', torch_root.parent / 'nvidia' / 'cuda_nvrtc' / 'lib'):
        candidates.extend(str(p) for p in sorted(folder.glob('libnvrtc.so*')))
    found = ctypes.util.find_library('nvrtc')
    return candidates + ([found] if found else []) + ['libnvrtc.so.12', 'libnvrtc.so']


def _set_signature(library, name, args):
    function = getattr(library, name)
    function.argtypes = args
    function.restype = ctypes.c_int
    return function


class _CudaKernel:
    def __init__(self, torch, device):
        self.module = None
        self.module_unload = None
        self.directory_handle = None
        try:
            self._initialize(torch, device)
        except BaseException:
            self.close()
            raise

    def _initialize(self, torch, device):
        self.torch, self.device = torch, device
        if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
            self.directory_handle = os.add_dll_directory(str(Path(torch.__file__).resolve().parent / 'lib'))
        attempts = []
        self.nvrtc = None
        for name in _library_paths(torch):
            try:
                self.nvrtc = ctypes.CDLL(name)
                break
            except OSError as error:
                attempts.append(str(error))
        if self.nvrtc is None:
            raise GpuCandidateCentresUnavailable('CUDA NVRTC runtime compiler is unavailable: ' + '; '.join(attempts))
        pointer = ctypes.c_void_p
        ppointer = ctypes.POINTER(pointer)
        char_pp = ctypes.POINTER(ctypes.c_char_p)
        self.create = _set_signature(self.nvrtc, 'nvrtcCreateProgram',
            [ppointer, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, char_pp, char_pp])
        self.compile = _set_signature(self.nvrtc, 'nvrtcCompileProgram', [pointer, ctypes.c_int, char_pp])
        self.log_size = _set_signature(self.nvrtc, 'nvrtcGetProgramLogSize', [pointer, ctypes.POINTER(ctypes.c_size_t)])
        self.log = _set_signature(self.nvrtc, 'nvrtcGetProgramLog', [pointer, pointer])
        self.ptx_size = _set_signature(self.nvrtc, 'nvrtcGetPTXSize', [pointer, ctypes.POINTER(ctypes.c_size_t)])
        self.ptx = _set_signature(self.nvrtc, 'nvrtcGetPTX', [pointer, pointer])
        self.destroy = _set_signature(self.nvrtc, 'nvrtcDestroyProgram', [ppointer])
        program = pointer()
        self._check(self.create(ctypes.byref(program), _CUDA_SOURCE.encode(), b'bw_candidate_centres.cu', 0, None, None), 'nvrtcCreateProgram')
        major, minor = torch.cuda.get_device_capability(device)
        options = [f'--gpu-architecture=compute_{major}{minor}'.encode(),
                   b'--std=c++14', b'--fmad=false', b'--ftz=false', b'--prec-div=true']
        try:
            result = self.compile(program, len(options), (ctypes.c_char_p * len(options))(*options))
            if result:
                size = ctypes.c_size_t()
                self.log_size(program, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                self.log(program, log)
                raise GpuCandidateCentresUnavailable('CUDA centre kernel compilation failed: ' + log.value.decode(errors='replace'))
            size = ctypes.c_size_t()
            self._check(self.ptx_size(program, ctypes.byref(size)), 'nvrtcGetPTXSize')
            ptx = ctypes.create_string_buffer(size.value)
            self._check(self.ptx(program, ptx), 'nvrtcGetPTX')
        finally:
            self.destroy(ctypes.byref(program))
        self.driver = ctypes.CDLL('nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1')
        self.module_load = _set_signature(self.driver, 'cuModuleLoadData', [ppointer, pointer])
        self.module_unload = _set_signature(self.driver, 'cuModuleUnload', [pointer])
        self.get_function = _set_signature(self.driver, 'cuModuleGetFunction', [ppointer, pointer, ctypes.c_char_p])
        self.launch = _set_signature(self.driver, 'cuLaunchKernel',
            [pointer, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
             ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, pointer, ppointer, ppointer])
        self.module, self.function = pointer(), pointer()
        self._check(self.module_load(ctypes.byref(self.module), ptx), 'cuModuleLoadData')
        self._check(self.get_function(ctypes.byref(self.function), self.module, b'aggregate_centres'), 'cuModuleGetFunction')

    def close(self):
        """Best-effort, idempotent release after failed initialization/probing.

        Cleanup must never replace the exception that triggered CPU fallback.
        A successful instance stays resident in the per-device kernel cache.
        """
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

    @staticmethod
    def _check(result, operation):
        if result:
            raise GpuCandidateCentresUnavailable(f'CUDA centre aggregation {operation} failed with code {result}')

    def __call__(self, points, scores, indices, offsets):
        torch = self.torch
        rows = offsets.numel() - 1
        output = torch.empty((rows, 3), dtype=torch.float32, device=points.device)
        values = [ctypes.c_void_p(v.data_ptr()) for v in (points, scores, indices, offsets)]
        values += [ctypes.c_int(rows), ctypes.c_void_p(output.data_ptr())]
        pointers = (ctypes.c_void_p * len(values))(*(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values))
        stream = ctypes.c_void_p(torch.cuda.current_stream(points.device).cuda_stream)
        self._check(self.launch(self.function, (rows + 127) // 128, 1, 1, 128, 1, 1,
                               0, stream, pointers, None), 'cuLaunchKernel')
        return output


@lru_cache(maxsize=8)
def _kernel_for_device(device):
    import torch
    with torch.cuda.device(device):
        torch.cuda.init()
        # A tensor ensures the runtime's primary context is current for the
        # driver API, including on freshly-created application worker threads.
        torch.empty(1, device=f'cuda:{device}')
        try:
            kernel = _CudaKernel(torch, device)
        except (OSError, AttributeError) as error:
            raise GpuCandidateCentresUnavailable('CUDA centre runtime library is incompatible or unavailable: '
                                                 + str(error)) from error
        try:
            _certify_numpy_reductions(torch, kernel, device)
        except BaseException:
            kernel.close()
            raise
        return kernel


def _certify_numpy_reductions(torch, kernel, device):
    """One tiny per-device check rejects unsupported NumPy reduction layouts."""
    rng = np.random.default_rng(4631)
    sizes = [1, 2, 7, 8, 9, 15, 31, 64, 127, 128, 129, 255, 1025, 8193]
    points = rng.uniform(-1000, 1000, (max(sizes), 2)).astype(np.float32)
    scores = rng.uniform(-.1, 2.2, max(sizes)).astype(np.float32)
    rows = [rng.permutation(size).astype(np.int32) for size in sizes]
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int32)
    indices = np.concatenate(rows)
    values = [torch.as_tensor(v, device=f'cuda:{device}') for v in (points, scores, indices, offsets)]
    actual = kernel(*values).cpu().numpy()
    expected = np.asarray([(*np.average(points[row], axis=0, weights=np.maximum(scores[row], 1e-4)),
                             np.clip(scores[row], 0., 2.).sum()) for row in rows], dtype=np.float32)
    if not np.array_equal(actual, expected):
        differences = np.argwhere(actual != expected)
        raise GpuCandidateCentresUnavailable('CUDA centre aggregates cannot certify installed NumPy float32 reduction order; '
                                             f'{len(differences)} differing values')


def _neighbour_batches(tree, points, radius, batch_size, max_pairs):
    """Bound query output by counted neighbours before materializing lists.

    return_sorted=False is essential: scalar cKDTree queries are unsorted and
    sorting the batched results would change the CPU float32 summation order.
    No N x N distance table is allocated on either device.
    """
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        counts = np.asarray(tree.query_ball_point(points[start:stop], radius,
                            return_length=True), dtype=np.int64)
        offset = 0
        while offset < len(counts):
            count = 0
            end = offset
            while end < len(counts) and count + int(counts[end]) <= max_pairs:
                count += int(counts[end])
                end += 1
            if end == offset:
                raise GpuCandidateCentresUnavailable('CUDA centre neighbourhood exceeds bounded pair capacity')
            rows = tree.query_ball_point(points[start + offset:start + end], radius, return_sorted=False)
            yield rows
            offset = end


def _rank_and_map(aggregates, template, plot_area, overlap_multiplicity):
    """Original compact CPU scalar arithmetic, tie order, NMS and blur."""
    x0, y0, x1, y1 = plot_area
    vote_map = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    ranked = []
    for cx, cy, weight_sum, cell_count in aggregates:
        effective_cells = int(cell_count) / max(overlap_multiplicity, 1.0)
        density = float(effective_cells + weight_sum / max(overlap_multiplicity, 1.0))
        ranked.append((float(cx), float(cy), int(math.ceil(effective_cells)), density))
        px, py = int(round(cx)) - x0, int(round(cy)) - y0
        if 0 <= px < vote_map.shape[1] and 0 <= py < vote_map.shape[0]:
            vote_map[py, px] = max(vote_map[py, px], density)
    ranked.sort(key=lambda item: item[3], reverse=True)
    selected = []
    nms_radius = max(2.0, .28 * template.diameter)
    for candidate in ranked:
        if not any((candidate[0] - kept[0]) ** 2 + (candidate[1] - kept[1]) ** 2
                   <= nms_radius ** 2 for kept in selected):
            selected.append(candidate)
    if vote_map.any():
        vote_map = cv2.GaussianBlur(vote_map, (0, 0), max(.8, .10 * template.diameter))
    return selected, vote_map


def candidate_centres_cuda(hypotheses, template, plot_area,
                           overlap_multiplicity=1.0, batch_size=512):
    """Return (centres, vote_map, diagnostics), preserving the CPU algorithm."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    started = time.perf_counter()
    stats = dict(backend='cuda_candidate_centres', used_cuda=False, hypotheses=len(hypotheses),
                 neighbour_pairs=0, gpu_batches=0, max_hypotheses_per_batch=0,
                 max_neighbour_pairs_per_batch=0, neighbour_lookup_seconds=0.,
                 aggregate_seconds=0., selection_seconds=0., compile_seconds=0.,
                 timing_scope='Wall time: CPU cKDTree lookup; CUDA aggregates including transfers/synchronization; CPU stable NMS and vote map',
                 policy='Exact float32 NumPy-order CUDA weighted sums and distinct-cell counts; unchanged CPU stable decisions')
    if not hypotheses:
        selected, vote_map = _rank_and_map([], template, plot_area, overlap_multiplicity)
        stats['total_seconds'] = time.perf_counter() - started
        return selected, vote_map, stats
    points = np.asarray([(h.x, h.y) for h in hypotheses], dtype=np.float32)
    scores = np.asarray([h.local_score for h in hypotheses], dtype=np.float32)
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise GpuCandidateCentresUnavailable('CUDA centre aggregation requires finite float32 coordinates and scores')
    if len(hypotheses) >= 2**24:
        raise GpuCandidateCentresUnavailable('CUDA centre aggregation input exceeds exact cell-count capacity')
    try:
        import torch
    except (ImportError, OSError) as error:
        raise GpuCandidateCentresUnavailable('CUDA centre aggregation requires installed CUDA PyTorch') from error
    if not torch.cuda.is_available():
        raise GpuCandidateCentresUnavailable('CUDA centre aggregation device is unavailable')
    device = torch.cuda.current_device()
    setup_started = time.perf_counter()
    kernel = _kernel_for_device(device)
    stats['compile_seconds'] = time.perf_counter() - setup_started
    lookup_started = time.perf_counter()
    tree = cKDTree(points)
    cells = {}
    cell_ids = np.empty(len(hypotheses), dtype=np.int64)
    for index, item in enumerate(hypotheses):
        key = item.cell_left, item.cell_top
        cell_ids[index] = cells.setdefault(key, len(cells))
    stats['neighbour_lookup_seconds'] += time.perf_counter() - lookup_started
    radius = max(2.0, .22 * template.diameter)
    # CSR indices plus row/cell keys, sort workspace and compact outputs stay
    # below a conservative ~64 MiB temporary budget; points are uploaded once.
    max_pairs = 64 * 1024**2 // 96
    batches = _neighbour_batches(tree, points, radius, batch_size, max_pairs)
    output = []
    with torch.inference_mode():
        aggregate_started = time.perf_counter()
        point_gpu, score_gpu, cell_gpu = [torch.as_tensor(a, device=f'cuda:{device}')
                                         for a in (points, scores, cell_ids)]
        torch.cuda.synchronize(device)
        stats['aggregate_seconds'] += time.perf_counter() - aggregate_started
        while True:
            lookup_started = time.perf_counter()
            try:
                rows = next(batches)
            except StopIteration:
                stats['neighbour_lookup_seconds'] += time.perf_counter() - lookup_started
                break
            lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
            indices = np.concatenate(rows).astype(np.int32)
            offsets = np.r_[0, np.cumsum(lengths)].astype(np.int32)
            stats['neighbour_lookup_seconds'] += time.perf_counter() - lookup_started
            aggregate_started = time.perf_counter()
            index_gpu, offset_gpu, length_gpu = [torch.as_tensor(a, device=f'cuda:{device}')
                                                  for a in (indices, offsets, lengths)]
            aggregates = kernel(point_gpu, score_gpu, index_gpu, offset_gpu)
            row_ids = torch.repeat_interleave(torch.arange(len(rows), device=f'cuda:{device}'),
                                              length_gpu, output_size=len(indices))
            keys = row_ids * len(cells) + cell_gpu[index_gpu.to(torch.int64)]
            unique = torch.unique_consecutive(torch.sort(keys).values)
            counts = torch.bincount(torch.div(unique, len(cells), rounding_mode='floor'), minlength=len(rows))
            compact = torch.cat((aggregates, counts.to(torch.float32)[:, None]), dim=1).cpu().numpy()
            output.append(compact)
            stats['aggregate_seconds'] += time.perf_counter() - aggregate_started
            stats['used_cuda'] = True
            stats['gpu_batches'] += 1
            stats['neighbour_pairs'] += len(indices)
            stats['max_hypotheses_per_batch'] = max(stats['max_hypotheses_per_batch'], len(rows))
            stats['max_neighbour_pairs_per_batch'] = max(stats['max_neighbour_pairs_per_batch'], len(indices))
    select_started = time.perf_counter()
    selected, vote_map = _rank_and_map(np.concatenate(output), template, plot_area, overlap_multiplicity)
    stats['selection_seconds'] = time.perf_counter() - select_started
    stats['total_seconds'] = time.perf_counter() - started
    return selected, vote_map, stats

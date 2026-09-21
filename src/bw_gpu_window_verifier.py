"""Certified scalar and independent-window CUDA preselection for B&W search.

Only alignments proven unable to enter the original CPU top eight are skipped.
Float64 CUDA ranks receive conservative intervals enclosing the original
float32 NumPy arithmetic. Every potentially competitive alignment is evaluated
again by the unmodified CPU helpers, in their original stable order. Required
ink, strict cores, boundary credit, sector checks and final decisions are not
changed. Torch is imported lazily; unsupported inputs use the full CPU search.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
import time

import numpy as np


_RESOURCE_CACHE = OrderedDict()
_CACHE_LIMIT = 4
_U32 = 2.0 ** -24


class _Uncertifiable(ValueError):
    pass


def _rank_error_bound(weight, core, boundary):
    """Absolute enclosure of CPU float32 rank around ideal float64 arithmetic.

    Inputs lie in [0, 1], expected ink in [.25, 1]. A deliberately loose 16u
    support allowance covers division, clipping, float32 .85 and multiplication
    in the boundary maximum (and the much smaller GPU float64 roundoff). A sum
    of n nonnegative float32 terms has relative error at most gamma_n; using
    the entire raster size also covers NumPy's pairwise reduction order.
    Denominators are the EXACT CPU float32 weight sums, not GPU recomputations.
    The core term's coefficient is .10+.30 because it also enters the hinge
    penalty. A further 1e-9 encloses float64 score/penalty combination roundoff.
    Gradual underflow contributes less than this absolute allowance.
    """
    n = int(weight.size)
    if n == 0 or n * _U32 >= .01:
        raise _Uncertifiable('raster too large for the conservative reduction bound')
    gamma = n * _U32 / (1. - n * _U32)
    support_error = 16. * _U32
    product_error = support_error + _U32 * (1. + support_error)

    def weighted_bound(values):
        denominator = float(values.sum())
        if not np.isfinite(denominator) or denominator < 1e-6:
            raise _Uncertifiable('required-weight denominator is too small')
        exact_sum = float(np.sum(values, dtype=np.float64))
        ratio = exact_sum / denominator
        # Product rounding, reduction rounding, then optional final float32
        # division (NumPy scalar promotion differs between installed versions).
        error = ratio * (product_error + gamma * (1. + product_error)
                         + _U32 * (1. + product_error) * (1. + gamma))
        return denominator, error

    denominator, weighted_error = weighted_bound(weight)
    core_count = int(core.sum())
    core_error = (support_error + gamma * (1. + support_error)
                  + _U32 * (1. + support_error) * (1. + gamma)) if core_count else 0.
    if boundary.any():
        boundary_denominator, boundary_error = weighted_bound(weight[boundary])
    else:
        boundary_denominator, boundary_error = 1., 0.
    error = .85 * weighted_error + .40 * core_error + .05 * boundary_error + 1e-9
    if not np.isfinite(error) or error > .02:
        raise _Uncertifiable('rank interval is too wide to certify efficiently')
    return float(error), denominator, boundary_denominator, max(core_count, 1)


def _alignment_plan(shapes, window_size, maximum_shift):
    usable, alignments = [], []
    for shape in shapes:
        scale, aspect, mask, weight, expected, core, boundary = shape
        height, width = mask.shape
        left, top = (window_size - width) // 2, (window_size - height) // 2
        if min(left, top) < maximum_shift:
            continue
        index = len(usable)
        usable.append(shape)
        for dy in range(-maximum_shift, maximum_shift + 1):
            for dx in range(-maximum_shift, maximum_shift + 1):
                penalty = (.004 * math.hypot(dx, dy) + .018 * abs(scale - 1.)
                           + .01 * abs(aspect - 1.))
                alignments.append((index, dx, dy, left + dx, top + dy, penalty))
    return usable, alignments


def _cpu_shortlist(membership, nearby, usable, alignments, selected=None):
    # Lazy import avoids a cycle when the production verifier calls this module.
    from occlusion_aware_window_verifier import _uncertain_scores, _uncertain_support
    shortlist = []
    indices = range(len(alignments)) if selected is None else selected
    for index in indices:
        shape_id, dx, dy, left, top, penalty = alignments[index]
        scale, aspect, mask, weight, expected, core, boundary = usable[shape_id]
        h, w = mask.shape
        support = _uncertain_support(membership[top:top+h, left:left+w],
                                     nearby[top:top+h, left:left+w],
                                     expected, core, boundary)
        weighted, core_recall, boundary_recall, score = _uncertain_scores(
            support, weight, core, boundary)
        rank = score - penalty - .30 * max(.85 - core_recall, 0)
        item = (rank, scale, aspect, dx, dy, mask, weight, support, core,
                boundary, weighted, core_recall, boundary_recall, score)
        if len(shortlist) < 8 or rank > shortlist[-1][0]:
            shortlist.append(item)
            shortlist.sort(key=lambda value: value[0], reverse=True)
            del shortlist[8:]
    return shortlist


def _validate_arrays(membership, nearby, usable, window_size, validate_shapes=True):
    for name, array in (('membership', membership), ('nearby', nearby)):
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise _Uncertifiable(f'{name} must be the production float32 raster')
        if array.shape != (window_size, window_size):
            raise _Uncertifiable(f'{name} must match the full square window')
        if not np.isfinite(array).all() or np.any(array < 0.) or np.any(array > 1.):
            raise _Uncertifiable(f'{name} lies outside the certified finite [0,1] range')
    if not validate_shapes:
        return
    for scale, aspect, mask, weight, expected, core, boundary in usable:
        if (not np.isfinite(scale) or not np.isfinite(aspect)
                or not .25 <= scale <= 4. or not .25 <= aspect <= 4.):
            raise _Uncertifiable('geometry lies outside the certified finite [.25,4] range')
        for array in (weight, expected):
            if array.dtype != np.float32 or array.shape != mask.shape:
                raise _Uncertifiable('weight and expected must be aligned float32 arrays')
        for array in (mask, core, boundary):
            if array.dtype != np.bool_ or array.shape != mask.shape:
                raise _Uncertifiable('mask, core and boundary must be aligned Boolean arrays')
        if (not np.isfinite(weight).all() or np.any(weight < 0.) or np.any(weight > 1.)
                or not np.isfinite(expected).all() or np.any(expected < .25)
                or np.any(expected > 1.)):
            raise _Uncertifiable('template lies outside the certified weight/expected range')
        if np.any(core & boundary) or not np.array_equal(core | boundary, mask):
            raise _Uncertifiable('strict core and uncertain boundary must partition the mask')
        if np.any(weight[~mask] != 0.):
            raise _Uncertifiable('required weights outside the mask are unsupported')


def _resource_key(usable, window_size, maximum_shift):
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str((window_size, maximum_shift)).encode('ascii'))
    for shape in usable:
        digest.update(repr(shape[:2]).encode('ascii'))
        for array in shape[2:]:
            digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.digest()


def _resources(torch, usable, alignments, window_size, maximum_shift, sparse=False):
    key = _resource_key(usable, window_size, maximum_shift)
    if sparse:
        key = b'sparse-fused:' + key
    if key in _RESOURCE_CACHE:
        _RESOURCE_CACHE.move_to_end(key)
        return _RESOURCE_CACHE[key], True
    if sparse:
        from bw_gpu_window_fused import prepare_resources
        pack = prepare_resources(torch, usable, alignments, window_size)
        pack['dense_fallback_args'] = (usable, alignments, window_size, maximum_shift)
        _RESOURCE_CACHE[key] = pack
        while len(_RESOURCE_CACHE) > _CACHE_LIMIT:
            _RESOURCE_CACHE.popitem(last=False)
        return pack, False
    max_h = max(shape[2].shape[0] for shape in usable)
    max_w = max(shape[2].shape[1] for shape in usable)
    n_shapes = len(usable)
    shape = (n_shapes, max_h, max_w)
    weights = np.zeros(shape, np.float64)
    expected = np.ones(shape, np.float64)
    cores = np.zeros(shape, bool)
    boundaries = np.zeros(shape, bool)
    errors, denominators, boundary_denominators, core_counts = [], [], [], []
    has_core, has_boundary = [], []
    for index, (_, _, mask, weight, expect, core, boundary) in enumerate(usable):
        h, w = mask.shape
        weights[index, :h, :w] = weight
        expected[index, :h, :w] = expect
        cores[index, :h, :w] = core
        boundaries[index, :h, :w] = boundary
        error, denominator, boundary_denominator, core_count = _rank_error_bound(weight, core, boundary)
        errors.append(error); denominators.append(denominator)
        boundary_denominators.append(boundary_denominator); core_counts.append(core_count)
        has_core.append(bool(core.any())); has_boundary.append(bool(boundary.any()))
    device = torch.device('cuda:0')
    def gpu(array, dtype=None):
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)
    pack = {
        'weights': gpu(weights), 'expected': gpu(expected),
        'cores': gpu(cores), 'boundaries': gpu(boundaries),
        'denominators': gpu(denominators, torch.float64),
        'boundary_denominators': gpu(boundary_denominators, torch.float64),
        'core_counts': gpu(core_counts, torch.float64),
        'has_core': gpu(has_core, torch.bool), 'has_boundary': gpu(has_boundary, torch.bool),
        'shape_ids': gpu([a[0] for a in alignments], torch.int64),
        'lefts': gpu([a[3] for a in alignments], torch.int64),
        'tops': gpu([a[4] for a in alignments], torch.int64),
        'penalties': gpu([a[5] for a in alignments], torch.float64),
        'dy': torch.arange(max_h, device=device).view(1, max_h, 1),
        'dx': torch.arange(max_w, device=device).view(1, 1, max_w),
        'errors': np.asarray([errors[a[0]] for a in alignments], np.float64),
        'max_area': max_h * max_w,
    }
    _RESOURCE_CACHE[key] = pack
    while len(_RESOURCE_CACHE) > _CACHE_LIMIT:
        _RESOURCE_CACHE.popitem(last=False)
    return pack, False


def certified_shortlist(membership, nearby, shapes, window_size, maximum_shift,
                        batch_size=512):
    """Return ``(original_CPU_format_top8, diagnostics)`` for one B&W window.

    The caller retains the original sector checks, winner selection, final
    metrics, aligned coordinates and decision thresholds. Any unsupported
    CUDA/raster condition explicitly runs the full original CPU search instead.
    This helper never returns a GPU-computed acceptance decision or CPU score.
    """
    started = time.perf_counter()
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    usable, alignments = _alignment_plan(shapes, window_size, maximum_shift)
    stats = {'backend': 'cuda_certified_window_preselection', 'used_cuda': False,
             'total_alignments': len(alignments), 'cpu_recomputed_alignments': 0,
             'pruned_alignments': 0, 'batch_size': batch_size,
             'resource_cache_hit': False, 'rank_error_bound_max': 0.,
             'setup_seconds': 0., 'gpu_rank_seconds': 0., 'cpu_recheck_seconds': 0.,
             'total_seconds': 0., 'fallback_reason': None,
             'score_policy': 'float64 GPU intervals; original float32 NumPy top8 recheck',
             'rank_tie_policy': 'original shape/dy/dx order and stable strict-greater top8',
             'asymmetric_ink_policy_changed': False,
             'gpu_rank_seconds_meaning': 'synchronized wall time including transfers; not kernel-only'}
    if not alignments:
        stats['total_seconds'] = time.perf_counter() - started
        return [], stats
    selected = None
    try:
        if not 0 <= maximum_shift <= 1024 or not 1 <= window_size <= 4096:
            raise _Uncertifiable('window/shift exceeds the certified double-precision penalty bound')
        _validate_arrays(membership, nearby, usable, window_size)
        import torch
        if not torch.cuda.is_available():
            raise _Uncertifiable('CUDA is unavailable')
        with torch.inference_mode():
            pack, cache_hit = _resources(torch, usable, alignments, window_size, maximum_shift)
            stats['resource_cache_hit'] = cache_hit
            stats['rank_error_bound_max'] = float(pack['errors'].max())
            observed_gpu = torch.as_tensor(np.ascontiguousarray(membership), device='cuda:0', dtype=torch.float64)
            nearby_gpu = torch.as_tensor(np.ascontiguousarray(nearby), device='cuda:0', dtype=torch.float64)
            torch.cuda.synchronize()
            stats['setup_seconds'] = time.perf_counter() - started
            rank_started = time.perf_counter()
            ranks = np.empty(len(alignments), np.float64)
            # Bound temporary double arrays below approximately 128 MiB. The
            # public batch size is an upper limit, not a memory commitment.
            chunk_size = min(batch_size, max(1, 128 * 1024**2 // (pack['max_area'] * 8 * 14)))
            stats['effective_batch_size'] = chunk_size
            for start in range(0, len(alignments), chunk_size):
                stop = min(start + chunk_size, len(alignments))
                ids = pack['shape_ids'][start:stop]
                ys = pack['tops'][start:stop, None, None] + pack['dy']
                xs = pack['lefts'][start:stop, None, None] + pack['dx']
                # Padding belongs only to smaller shape variants. Such pixels
                # carry zero weight and no core/boundary, so clamping is inert.
                ys = ys.clamp(0, window_size - 1); xs = xs.clamp(0, window_size - 1)
                expected = pack['expected'][ids]
                direct = (observed_gpu[ys, xs] / expected).clamp(max=1.)
                neighbor = .85 * (nearby_gpu[ys, xs] / expected).clamp(max=1.)
                boundary = pack['boundaries'][ids]
                support = torch.where(boundary, torch.maximum(direct, neighbor), direct)
                weighted_ink = support * pack['weights'][ids]
                weighted = weighted_ink.sum(dim=(1, 2)) / pack['denominators'][ids]
                core = (support * pack['cores'][ids]).sum(dim=(1, 2)) / pack['core_counts'][ids]
                core = torch.where(pack['has_core'][ids], core, 1.)
                boundary_recall = (weighted_ink * boundary).sum(dim=(1, 2)) / pack['boundary_denominators'][ids]
                boundary_recall = torch.where(pack['has_boundary'][ids], boundary_recall, 1.)
                score = .85 * weighted + .10 * core + .05 * boundary_recall
                rank = score - pack['penalties'][start:stop] - .30 * (.85 - core).clamp(min=0.)
                ranks[start:stop] = rank.cpu().numpy()
            torch.cuda.synchronize()
            stats['gpu_rank_seconds'] = time.perf_counter() - rank_started
            if not np.isfinite(ranks).all():
                raise _Uncertifiable('CUDA rank is nonfinite')
            lower, upper = ranks - pack['errors'], ranks + pack['errors']
            kth = min(8, len(lower))
            cutoff = float(np.partition(lower, len(lower) - kth)[len(lower) - kth])
            # Any omitted alignment has rank < at least eight other CPU ranks.
            # Include equality to preserve the original stable tie ordering.
            selected = np.flatnonzero(upper >= cutoff).tolist()
            stats['certified_cutoff_lower_bound'] = cutoff
            stats['used_cuda'] = True
    except (ImportError, RuntimeError, MemoryError, _Uncertifiable) as error:
        stats['fallback_reason'] = f'{type(error).__name__}: {error}'
        selected = None
    cpu_started = time.perf_counter()
    if selected is None:
        from occlusion_aware_window_verifier import _uncertain_cpu_shortlist
        shortlist = _uncertain_cpu_shortlist(membership, nearby, shapes, window_size, maximum_shift)
    else:
        shortlist = _cpu_shortlist(membership, nearby, usable, alignments, selected)
    stats['cpu_recheck_seconds'] = time.perf_counter() - cpu_started
    stats['cpu_recomputed_alignments'] = len(alignments) if selected is None else len(selected)
    stats['pruned_alignments'] = len(alignments) - stats['cpu_recomputed_alignments']
    stats['total_seconds'] = time.perf_counter() - started
    return shortlist, stats


def _rank_windows_together_torch(torch, memberships, nearby_windows, pack, window_size,
                                 batch_size):
    """Evaluate independent windows in the same CUDA launches.

    Flattened work is alignment-major: adjacent lanes belong to different
    windows. All rank chunks remain on CUDA until ONE group readback, unlike
    the scalar path's per-alignment-chunk synchronization. Shapes are shared,
    not copied once per window. The public batch size bounds evaluated
    window/alignment pairs; the temporary tensor budget also caps each launch.
    """
    n_windows = len(memberships)
    n_alignments = len(pack['errors'])
    observed = torch.as_tensor(np.stack(memberships), device='cuda:0', dtype=torch.float64)
    nearby = torch.as_tensor(np.stack(nearby_windows), device='cuda:0', dtype=torch.float64)
    ranks = torch.empty(n_alignments * n_windows, device='cuda:0', dtype=torch.float64)
    chunk_size = min(batch_size, max(1, 128 * 1024**2 // (pack['max_area'] * 8 * 16)))
    for start in range(0, ranks.numel(), chunk_size):
        stop = min(start + chunk_size, ranks.numel())
        work = torch.arange(start, stop, device='cuda:0')
        windows = work.remainder(n_windows)
        alignments = torch.div(work, n_windows, rounding_mode='floor')
        ids = pack['shape_ids'][alignments]
        ys = (pack['tops'][alignments, None, None] + pack['dy']).clamp(0, window_size - 1)
        xs = (pack['lefts'][alignments, None, None] + pack['dx']).clamp(0, window_size - 1)
        expected = pack['expected'][ids]
        direct = (observed[windows[:, None, None], ys, xs] / expected).clamp(max=1.)
        neighbor = .85 * (nearby[windows[:, None, None], ys, xs] / expected).clamp(max=1.)
        boundary = pack['boundaries'][ids]
        support = torch.where(boundary, torch.maximum(direct, neighbor), direct)
        weighted_ink = support * pack['weights'][ids]
        weighted = weighted_ink.sum(dim=(1, 2)) / pack['denominators'][ids]
        core = (support * pack['cores'][ids]).sum(dim=(1, 2)) / pack['core_counts'][ids]
        core = torch.where(pack['has_core'][ids], core, 1.)
        boundary_recall = (weighted_ink * boundary).sum(dim=(1, 2)) / pack['boundary_denominators'][ids]
        boundary_recall = torch.where(pack['has_boundary'][ids], boundary_recall, 1.)
        score = .85 * weighted + .10 * core + .05 * boundary_recall
        ranks[start:stop] = score - pack['penalties'][alignments] - .30 * (.85 - core).clamp(min=0.)
    # This is the only rank readback in a bounded window group. It waits for
    # queued CUDA work without a synchronize call inside the chunk loop.
    cpu_ranks = ranks.cpu().numpy().reshape(n_alignments, n_windows).T
    return cpu_ranks, chunk_size, math.ceil(n_alignments * n_windows / chunk_size)


def _rank_windows_together(torch, memberships, nearby_windows, pack, window_size,
                           batch_size):
    """Fused sparse ranks, or the unchanged tensor ranks on missing NVRTC.

    Both implementations evaluate every alignment and use the same float64
    enclosure. The existing 1e-9 double-roundoff allowance also covers the
    fused 128-lane nonnegative reduction: fewer than n/128 sequential additions
    plus seven tree levels, with n*u32 < .01 enforced by _rank_error_bound.
    No CPU shortlist arithmetic or strict-core rule changes.
    """
    if 'dense_fallback_args' not in pack:
        # Preserve direct internal callers that supply the historical dense
        # _resources() pack (the production multiwindow caller requests sparse).
        result = _rank_windows_together_torch(torch, memberships, nearby_windows, pack, window_size, batch_size)
        pack['rank_execution'] = dict(rank_engine='torch_tensor', fused_kernel_launches=0,
            fused_fallback_reason=None, rank_launch_policy='explicit historical dense resource pack')
        return result
    from bw_gpu_window_fused import FusedWindowUnavailable, rank_windows
    try:
        ranks, size, launches, execution = rank_windows(torch, memberships, nearby_windows, pack, batch_size)
        pack['rank_execution'] = execution
        return ranks, size, launches
    except (FusedWindowUnavailable, ImportError, OSError) as error:
        # Missing compiler support changes only the execution mechanism. Keep
        # the prior CUDA tensor implementation before considering full CPU.
        dense, _ = _resources(torch, *pack['dense_fallback_args'])
        result = _rank_windows_together_torch(torch, memberships, nearby_windows, dense, window_size, batch_size)
        pack['rank_execution'] = dict(rank_engine='torch_tensor', fused_kernel_launches=0,
            fused_fallback_reason=f'{type(error).__name__}: {error}',
            rank_launch_policy='legacy tensor batching with the original 128 MiB workspace cap')
        return result


def certified_shortlists_many(jobs, batch_size=512):
    """Return ordered ``[(original_CPU_top8, diagnostics), ...]``.

    Each job is ``(membership, nearby, shapes, window_size, maximum_shift)``.
    Equal geometry/content shares CUDA resources. Independent windows are
    evaluated together; interval certification and stable CPU rechecking are
    exactly the scalar algorithm. Large groups are split by input/rank memory
    bounds, never by dropping candidates or changing acceptance rules.

    Group timings are apportioned among its windows, so summing per-window
    setup/gpu times does not multiply batch wall time. Each record identifies
    the actual CUDA window group and the number of independent windows in it.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    started = time.perf_counter()
    jobs = list(jobs)
    if not jobs:
        return []
    records, groups, plans, validated_shapes = [], OrderedDict(), {}, set()
    for index, job in enumerate(jobs):
        membership, nearby, shapes, window_size, maximum_shift = job
        stats = {
            'backend': 'cuda_certified_multiwindow_preselection', 'used_cuda': False,
            'total_alignments': 0, 'cpu_recomputed_alignments': 0, 'pruned_alignments': 0,
            'batch_size': batch_size, 'resource_cache_hit': False, 'rank_error_bound_max': 0.,
            'setup_seconds': 0., 'gpu_rank_seconds': 0., 'cpu_recheck_seconds': 0.,
            'total_seconds': 0., 'fallback_reason': None, 'window_request_index': index,
            'window_batch_id': None, 'window_batch_size': 0, 'gpu_launch_count': 0,
            'gpu_rank_readbacks': 0, 'independent_windows_per_launch_max': 0,
            'score_policy': 'float64 GPU intervals; original float32 NumPy top8 recheck',
            'rank_tie_policy': 'original shape/dy/dx order and stable strict-greater top8',
            'asymmetric_ink_policy_changed': False,
            'timing_policy': 'group setup/rank wall times allocated equally across windows; CPU recheck individual',
            'gpu_rank_seconds_meaning': 'group synchronized wall time including transfers; not kernel-only',
        }
        record = {'stats': stats, 'selected': None, 'usable': [], 'alignments': []}
        records.append(record)
        try:
            if not 0 <= maximum_shift <= 1024 or not 1 <= window_size <= 4096:
                raise _Uncertifiable('window/shift exceeds the certified double-precision penalty bound')
            # Production shape tuples are reused by the template cache. This
            # identity cache avoids hashing the same arrays for every window;
            # the content key still merges distinct but equal template objects.
            identity = (id(shapes), window_size, maximum_shift)
            if identity not in plans:
                usable, alignments = _alignment_plan(shapes, window_size, maximum_shift)
                plans[identity] = (usable, alignments, _resource_key(usable, window_size, maximum_shift))
            usable, alignments, key = plans[identity]
            record.update(usable=usable, alignments=alignments)
            stats['total_alignments'] = len(alignments)
            if not alignments:
                record['selected'] = []
                continue
            _validate_arrays(membership, nearby, usable, window_size, key not in validated_shapes)
            validated_shapes.add(key)
            groups.setdefault(key, []).append(index)
        except _Uncertifiable as error:
            stats['fallback_reason'] = f'{type(error).__name__}: {error}'
    preflight_seconds = time.perf_counter() - started
    torch_error = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise _Uncertifiable('CUDA is unavailable')
    except (ImportError, RuntimeError, _Uncertifiable) as error:
        torch_error = f'{type(error).__name__}: {error}'

    batch_id = 0
    for indices in groups.values():
        first = indices[0]
        _, _, _, window_size, maximum_shift = jobs[first]
        usable, alignments = records[first]['usable'], records[first]['alignments']
        # Bound two raster stacks and rank readback/output separately using the
        # conservative float64 size needed by the optional tensor fallback.
        # At most 256 independent windows share a batch; ordinary swatches fit
        # far below these 64 MiB caps. Tensor fallback chunks retain their
        # 128 MiB cap; fused blocks need only fixed 3072-byte shared reductions.
        windows_per_group = min(256, max(1, 64 * 1024**2 // (window_size**2 * 8 * 2)),
                                max(1, 64 * 1024**2 // (len(alignments) * 8)))
        for offset in range(0, len(indices), windows_per_group):
            batch_indices = indices[offset:offset + windows_per_group]
            batch_id += 1
            group_started = time.perf_counter()
            try:
                if torch_error is not None:
                    raise _Uncertifiable(torch_error)
                with torch.inference_mode():
                    pack, cache_hit = _resources(torch, usable, alignments, window_size, maximum_shift, sparse=True)
                    setup_seconds = time.perf_counter() - group_started
                    rank_started = time.perf_counter()
                    ranks, effective_size, launches = _rank_windows_together(
                        torch, [jobs[i][0] for i in batch_indices], [jobs[i][1] for i in batch_indices],
                        pack, window_size, batch_size)
                    rank_seconds = time.perf_counter() - rank_started
                if not np.isfinite(ranks).all():
                    raise _Uncertifiable('CUDA rank is nonfinite')
                lower, upper = ranks - pack['errors'][None, :], ranks + pack['errors'][None, :]
                kth = min(8, len(alignments))
                cutoffs = np.partition(lower, len(alignments) - kth, axis=1)[:, len(alignments) - kth]
                for local_index, index in enumerate(batch_indices):
                    records[index]['selected'] = np.flatnonzero(upper[local_index] >= cutoffs[local_index]).tolist()
                    stats = records[index]['stats']
                    stats.update(used_cuda=True, resource_cache_hit=cache_hit,
                        rank_error_bound_max=float(pack['errors'].max()),
                        setup_seconds=setup_seconds / len(batch_indices),
                        gpu_rank_seconds=rank_seconds / len(batch_indices),
                        effective_batch_size=effective_size, window_batch_id=batch_id,
                        window_batch_size=len(batch_indices),
                        # Count launch/readback work once in the batch's first
                        # record; size/id repeats are metadata, not additive.
                        gpu_launch_count=launches if local_index == 0 else 0,
                        gpu_rank_readbacks=1 if local_index == 0 else 0,
                        independent_windows_per_launch_max=min(len(batch_indices), effective_size),
                        certified_cutoff_lower_bound=float(cutoffs[local_index]))
                    execution = dict(pack.get('rank_execution', {}))
                    # Kernel launches are additive work, so record them only
                    # once; resource sizes/policies are repeated metadata.
                    if local_index:
                        execution['fused_kernel_launches'] = 0
                    stats.update(execution)
            except (ImportError, RuntimeError, MemoryError, _Uncertifiable) as error:
                elapsed = time.perf_counter() - group_started
                for index in batch_indices:
                    records[index]['selected'] = None
                    records[index]['stats'].update(fallback_reason=f'{type(error).__name__}: {error}',
                        setup_seconds=elapsed / len(batch_indices))

    output = []
    for index, (membership, nearby, shapes, window_size, maximum_shift) in enumerate(jobs):
        record = records[index]
        stats, selected = record['stats'], record['selected']
        cpu_started = time.perf_counter()
        if selected is None:
            from occlusion_aware_window_verifier import _uncertain_cpu_shortlist
            shortlist = _uncertain_cpu_shortlist(membership, nearby, shapes, window_size, maximum_shift)
            if not record['alignments']:
                _, alignments = _alignment_plan(shapes, window_size, maximum_shift)
                stats['total_alignments'] = len(alignments)
        else:
            shortlist = _cpu_shortlist(membership, nearby, record['usable'], record['alignments'], selected)
        stats['cpu_recheck_seconds'] = time.perf_counter() - cpu_started
        stats['cpu_recomputed_alignments'] = stats['total_alignments'] if selected is None else len(selected)
        stats['pruned_alignments'] = stats['total_alignments'] - stats['cpu_recomputed_alignments']
        stats['preflight_seconds'] = preflight_seconds / max(1, len(jobs))
        stats['total_seconds'] = (stats['preflight_seconds'] + stats['setup_seconds']
                                  + stats['gpu_rank_seconds'] + stats['cpu_recheck_seconds'])
        output.append((shortlist, stats))
    return output

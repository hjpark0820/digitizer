"""CUDA acceleration of the *existing* B&W integer-centre refinement score.

No detector policy, candidate radius, source pixels, template geometry, or
acceptance threshold is changed. Torch is imported only when constructing the
backend. GPU work consists of patch-local binary morphology and integer counts;
the original double-precision scalar score and stable y/x selection run on CPU.
The caller owns an explicit CPU fallback when CUDA is unavailable/unsupported.
"""
from __future__ import annotations

from functools import lru_cache
import math
import time
from typing import Sequence

import cv2
import numpy as np


class GpuRefinementUnavailable(RuntimeError):
    """CUDA is unavailable or exact CPU score semantics cannot be certified."""


def _distance_coverage_footprint(edge_tolerance: float) -> np.ndarray:
    """Threshold the same OpenCV 3x3 chamfer metric around one zero seed.

    A local distance-to-edge threshold is a binary dilation by this footprint,
    provided the dilation sees only edges within that particular extracted patch.
    A global image distance map is deliberately NOT used.
    """
    tolerance = float(edge_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('edge_tolerance must be finite and nonnegative')
    # DIST_L2 mask=3 axial cost is approximately .955, therefore radius must
    # exceed ceil(tolerance), unlike a Euclidean disk constructed by geometry.
    radius = max(2, int(math.ceil(tolerance / .90)) + 2)
    seed = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    seed[radius, radius] = 0
    distance = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
    footprint = distance <= tolerance
    assert footprint[radius, radius]
    assert not footprint[0].any() and not footprint[-1].any()
    assert not footprint[:, 0].any() and not footprint[:, -1].any()
    return np.ascontiguousarray(footprint)


def _validate_footprint_equivalence(footprint: np.ndarray, edge_tolerance: float,
                                  patch_shape: tuple[int, int]) -> dict:
    """Deterministic CPU parity checks, including local patch-border behavior.

    All single-edge positions are tested for small patches. Larger patches use
    corners, edges, a regular grid and deterministic interior positions, plus
    multi-edge patterns. Equality is bitwise on the thresholded support mask.
    The distance transform is a min-of-seed chamfer metric; the multi-edge tests
    additionally guard implementation/border differences in the installed cv2.
    """
    h, w = map(int, patch_shape)
    if h <= 0 or w <= 0:
        raise ValueError('patch dimensions must be positive')
    kernel = np.ascontiguousarray(footprint, dtype=np.uint8)
    rng = np.random.default_rng(17823)
    if h * w <= 2304:
        positions = [(y, x) for y in range(h) for x in range(w)]
        exhaustive = True
    else:
        ys = {0, h - 1, h // 2, h // 4, 3 * h // 4}
        xs = {0, w - 1, w // 2, w // 4, 3 * w // 4}
        positions = sorted({(y, x) for y in ys for x in xs} |
                           {(int(rng.integers(h)), int(rng.integers(w))) for _ in range(32)})
        exhaustive = False
    def compare(edge):
        expected = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3) <= edge_tolerance
        actual = cv2.dilate(edge.astype(np.uint8), kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if not np.array_equal(expected, actual):
            raise GpuRefinementUnavailable('Patch-local chamfer footprint differs from installed OpenCV distanceTransform')
    for y, x in positions:
        edge = np.zeros((h, w), bool)
        edge[y, x] = True
        compare(edge)
    patterns = [np.zeros((h, w), bool), np.ones((h, w), bool)]
    for density in (.01, .08, .25, .60):
        for _ in range(3):
            patterns.append(rng.random((h, w)) < density)
    for edge in patterns:
        compare(edge)
    return {'single_seed_cases': len(positions), 'exhaustive_single_seed_positions': exhaustive,
            'multi_seed_cases': len(patterns), 'threshold_mask_mismatches': 0,
            'patch_shape': [h, w], 'edge_tolerance': float(edge_tolerance)}


@lru_cache(maxsize=64)
def _checked_footprint(edge_tolerance: float, h: int, w: int):
    footprint = _distance_coverage_footprint(edge_tolerance)
    # Near a float32 chamfer boundary, different shortest-path addition orders
    # could cross the scalar threshold. Refuse that rare case, rather than
    # claim exactness merely because representative probes happened to pass.
    radius = footprint.shape[0] // 2
    seed = np.ones(footprint.shape, np.uint8); seed[radius, radius] = 0
    distance = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
    guard = float(max(1e-5, 32 * np.finfo(np.float32).eps * radius * max(1., edge_tolerance)))
    nearest = float(np.min(np.abs(distance.astype(np.float64) - edge_tolerance)))
    if nearest <= guard:
        raise GpuRefinementUnavailable('Contour tolerance is too close to a float32 chamfer boundary for exact accelerated thresholding')
    report = _validate_footprint_equivalence(footprint, edge_tolerance, (h, w))
    report = {**report, 'nearest_metric_threshold_gap': nearest, 'float32_threshold_guard': guard}
    return footprint, report


class GpuCenterRefiner:
    """Batch/cache score locations and refine seeds in their original order.

    Arrays are uploaded once per instance/template. Candidate patches are
    processed in bounded chunks; arrays outside the plot are *not* zeroed here.
    Only true image boundaries receive zero padding, exactly as the CPU helper.
    """
    def __init__(self, template, membership, plot_edge, plot_area,
                 batch_size: int = 256):
        started = time.perf_counter()
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as error:
            raise GpuRefinementUnavailable('GPU refinement requires installed PyTorch with CUDA support') from error
        if not torch.cuda.is_available():
            raise GpuRefinementUnavailable('PyTorch CUDA device is unavailable; select the CPU refinement backend')
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.torch, self.functional = torch, functional
        self.device = torch.device('cuda:0')
        self.batch_size = int(batch_size)
        self.template = template
        self.plot_area = tuple(map(int, plot_area))
        self.membership = np.asarray(membership)
        self.plot_edge = np.asarray(plot_edge)
        if self.membership.ndim != 2 or self.membership.shape != self.plot_edge.shape:
            raise ValueError('membership and plot_edge must have matching two-dimensional image shapes')
        if self.membership.dtype != np.float32:
            raise GpuRefinementUnavailable('Exact GPU refinement currently requires float32 membership, matching the v46 production path')
        if not np.isfinite(self.membership).all():
            raise GpuRefinementUnavailable('Nonfinite membership is unsupported for exact GPU refinement')
        mask = np.asarray(template.mask, dtype=bool)
        edge = np.asarray(template.edge, dtype=bool)
        if mask.shape != edge.shape or mask.ndim != 2:
            raise ValueError('Template mask and edge must have equal two-dimensional shapes')
        self.h, self.w = mask.shape
        # A larger public batch can amortize independent-window launches while
        # large raster patches must still fit bounded refinement work buffers.
        requested_batch_size = self.batch_size
        self.batch_size = min(self.batch_size, max(1, 128 * 1024**2 // (self.h*self.w*48)))
        self.diameter = float(template.diameter)
        if not np.isfinite(self.diameter) or self.diameter <= 0:
            raise ValueError('Template diameter must be positive and finite')
        self.radius = max(2, int(round(.30 * self.diameter)))
        self.threshold = .45 if template.ink.achromatic else .24
        tolerance = max(1.25, .08 * self.diameter)
        footprint, footprint_report = _checked_footprint(tolerance, self.h, self.w)
        dilation_radius = max(1, int(round(.06 * self.diameter)))
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * dilation_radius + 1, 2 * dilation_radius + 1)) > 0
        kernel_side = max(footprint.shape[0], ellipse.shape[0])
        weights = np.zeros((2, 1, kernel_side, kernel_side), np.float32)
        for i, kernel in enumerate((footprint, ellipse)):
            top, left = (kernel_side-kernel.shape[0])//2, (kernel_side-kernel.shape[1])//2
            weights[i, 0, top:top+kernel.shape[0], left:left+kernel.shape[1]] = kernel
        yy, xx = np.indices(mask.shape)
        centre_radius = max(1.5, .22 * self.diameter)
        centre_disk = ((xx - .5 * (self.w-1))**2 + (yy - .5 * (self.h-1))**2 <= centre_radius**2)
        self.edge_count = max(int(edge.sum()), 1)
        self.mask_count = max(int(mask.sum()), 1)
        self.centre_count = int(centre_disk.sum())
        if not self.centre_count:
            raise GpuRefinementUnavailable('Template has an empty center disk')
        self.template_centre_fill = float(mask[centre_disk].mean())
        self.kernel_padding = kernel_side // 2
        self._scores: dict[tuple[int, int], float] = {}
        self.stats = {
            'backend': 'cuda_batched_patch_local_refinement', 'batch_size': self.batch_size,
            'requested_batch_size': requested_batch_size,
            'workspace_policy': 'adaptive patch batch, approximately 128 MiB temporary-array budget',
            'device': torch.cuda.get_device_name(self.device), 'torch_version': torch.__version__,
            'diameter': self.diameter, 'patch_shape': [self.h, self.w],
            'footprint_validation': footprint_report,
            'refine_calls': 0, 'input_centers': 0, 'enumerated_locations': 0,
            'unique_scored_locations': 0, 'location_cache_hits': 0, 'gpu_batches': 0,
            'setup_seconds': 0., 'gpu_scoring_seconds': 0., 'refine_seconds': 0.,
            'score_selection_seconds': 0., 'total_seconds': 0.,
            'score_policy': 'GPU exact integer counts; original Python float64 scalar objective/order',
            'gpu_scoring_seconds_meaning': 'synchronized scoring wall time, including transfers and CPU scalar combination; not CUDA-kernel-only time',
            'patch_boundary_policy': 'zero only outside full image; morphology confined to each extracted patch',
            'shape_or_radius_policy_changed': False,
        }
        try:
            with torch.inference_mode():
                self._membership_gpu = torch.as_tensor(np.ascontiguousarray(self.membership), device=self.device)
                self._edge_gpu = torch.as_tensor(np.ascontiguousarray(self.plot_edge, dtype=bool), device=self.device)
                self._mask_gpu = torch.as_tensor(np.ascontiguousarray(mask), device=self.device)
                self._template_edge_gpu = torch.as_tensor(np.ascontiguousarray(edge), device=self.device)
                self._centre_gpu = torch.as_tensor(np.ascontiguousarray(centre_disk), device=self.device)
                self._kernels_gpu = torch.as_tensor(weights, device=self.device)
                self._dy_gpu = torch.arange(self.h, device=self.device, dtype=torch.int64).view(1, self.h, 1)
                self._dx_gpu = torch.arange(self.w, device=self.device, dtype=torch.int64).view(1, 1, self.w)
                torch.cuda.synchronize(self.device)
        except (RuntimeError, MemoryError) as error:
            raise GpuRefinementUnavailable(f'Unable to initialize CUDA refinement arrays: {error}') from error
        self.stats['setup_seconds'] = time.perf_counter() - started
        self.stats['total_seconds'] = self.stats['setup_seconds']

    def _score_locations(self, locations: Sequence[tuple[int, int]]) -> dict[tuple[int, int], float]:
        """Score only previously unseen integer locations; cache survives calls."""
        pending = list(dict.fromkeys((int(x), int(y)) for x, y in locations if (int(x), int(y)) not in self._scores))
        if not pending:
            return {tuple(location): self._scores[tuple(location)] for location in locations}
        torch, functional = self.torch, self.functional
        started = time.perf_counter()
        image_h, image_w = self.membership.shape
        try:
            with torch.inference_mode(), torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
                for start in range(0, len(pending), self.batch_size):
                    chunk = pending[start:start+self.batch_size]
                    # Python round is intentional: even patch sizes alternate
                    # half-tie origins. x-width//2 is NOT equivalent.
                    lefts = [int(round(x - .5 * (self.w-1))) for x, _ in chunk]
                    tops = [int(round(y - .5 * (self.h-1))) for _, y in chunk]
                    left = torch.tensor(lefts, device=self.device, dtype=torch.int64).view(-1, 1, 1)
                    top = torch.tensor(tops, device=self.device, dtype=torch.int64).view(-1, 1, 1)
                    ys, xs = top+self._dy_gpu, left+self._dx_gpu
                    valid = (ys >= 0) & (ys < image_h) & (xs >= 0) & (xs < image_w)
                    cy, cx = ys.clamp(0, image_h-1), xs.clamp(0, image_w-1)
                    edge_patch = self._edge_gpu[cy, cx] & valid
                    member_patch = torch.where(valid, self._membership_gpu[cy, cx], 0.)
                    plot_ink = member_patch >= self.threshold
                    channels = torch.stack((edge_patch, plot_ink), dim=1).to(torch.float32)
                    # Both kernels and inputs are binary. A 0.5 separator avoids
                    # accepting numerical near-zero convolution artifacts; any
                    # mathematically positive sum is an integer at least one.
                    dilated = functional.conv2d(channels, self._kernels_gpu,
                                                padding=self.kernel_padding, groups=2) >= .5
                    any_edge = edge_patch.any(dim=(1, 2))
                    counts = torch.stack((
                        any_edge.to(torch.int64),
                        (dilated[:, 0] & self._template_edge_gpu).sum(dim=(1, 2), dtype=torch.int64),
                        (dilated[:, 1] & self._mask_gpu).sum(dim=(1, 2), dtype=torch.int64),
                        (plot_ink & self._mask_gpu).sum(dim=(1, 2), dtype=torch.int64),
                        (plot_ink & self._centre_gpu).sum(dim=(1, 2), dtype=torch.int64),
                    ), dim=1).cpu().numpy()
                    # Integer CPU transfer synchronizes the work before scoring;
                    # scalar operations deliberately match _alignment_objective.
                    for location, (has_edge, contour_n, foreground_n, direct_n, centre_n) in zip(chunk, counts, strict=True):
                        if not has_edge:
                            score = -1.
                        else:
                            contour_coverage = float(contour_n / self.edge_count)
                            foreground_coverage = float(foreground_n / self.mask_count)
                            direct_coverage = float(direct_n / self.mask_count)
                            fill_similarity = 1. - abs(self.template_centre_fill - float(centre_n / self.centre_count))
                            score = (.44 * contour_coverage + .23 * foreground_coverage +
                                     .18 * direct_coverage + .15 * fill_similarity)
                        self._scores[location] = float(score)
                    self.stats['gpu_batches'] += 1
                torch.cuda.synchronize(self.device)
        except (RuntimeError, MemoryError) as error:
            raise GpuRefinementUnavailable(f'CUDA refinement scoring failed; caller may explicitly retry CPU: {error}') from error
        self.stats['unique_scored_locations'] += len(pending)
        self.stats['gpu_scoring_seconds'] += time.perf_counter() - started
        return {tuple(location): self._scores[tuple(location)] for location in locations}

    def refine_many(self, centers: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        started = time.perf_counter()
        centers = [(float(x), float(y)) for x, y in centers]
        if any(not np.isfinite(x) or not np.isfinite(y) for x, y in centers):
            raise ValueError('Candidate centers must be finite')
        x0, y0, x1, y1 = self.plot_area
        sequences, distinct = [], {}
        enumerated = 0
        for x, y in centers:
            locations = []
            for cy in range(int(round(y))-self.radius, int(round(y))+self.radius+1):
                if not y0 <= cy < y1:
                    continue
                for cx in range(int(round(x))-self.radius, int(round(x))+self.radius+1):
                    if not x0 <= cx < x1:
                        continue
                    point = (cx, cy)
                    locations.append(point)
                    distinct.setdefault(point, None)
            enumerated += len(locations)
            sequences.append(locations)
        new_count = sum(point not in self._scores for point in distinct)
        self._score_locations(list(distinct))
        selected_started = time.perf_counter()
        result = []
        for original, locations in zip(centers, sequences, strict=True):
            best, best_score = original, -1.
            for location in locations:
                score = self._scores[location]
                if score > best_score:  # First row-major maximum wins, not >=.
                    best_score = score
                    best = (float(location[0]), float(location[1]))
            result.append(best)
        self.stats['score_selection_seconds'] += time.perf_counter() - selected_started
        self.stats['refine_calls'] += 1
        self.stats['input_centers'] += len(centers)
        self.stats['enumerated_locations'] += enumerated
        self.stats['location_cache_hits'] += enumerated - new_count
        self.stats['refine_seconds'] += time.perf_counter() - started
        self.stats['total_seconds'] = self.stats['setup_seconds'] + self.stats['refine_seconds']
        return result

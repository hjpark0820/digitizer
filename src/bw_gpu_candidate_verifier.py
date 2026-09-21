"""Batched, full numerical verification of independent B&W grid candidates.

``verify_many`` keeps patches, morphology, asymmetric contour evidence, scores
and acceptance gates on CUDA. Only compact result rows return to the CPU.
``prepare_many`` remains a compatibility/reference path for morphology tests.
"""
from __future__ import annotations

import time
import math

import cv2
import numpy as np


def _numpy_pairwise_sum(values):
    """Match NumPy's contiguous float32 sum, including its eight accumulators.

    A mathematically equivalent CUDA reduction can round differently and alter
    candidate ordering. This batched reduction deliberately uses NumPy's
    128-element pairwise blocks without sending pixel arrays back to the host.
    The input has one flat pixel axis at the end; other axes are batched.
    """
    import torch
    n = values.shape[-1]
    if n < 8:
        total = torch.zeros_like(values[..., 0]) if n else values.sum(-1)
        for index in range(n):
            total = total + values[..., index]
        return total
    leaves = []
    def tree(start, length):
        if length <= 128:
            leaves.append((start, length))
            return len(leaves)-1
        half = (length//2)//8*8
        return tree(start, half), tree(start+half, length-half)
    topology = tree(0, n)
    starts = torch.tensor([a for a, _ in leaves], device=values.device)
    lengths = torch.tensor([b for _, b in leaves], device=values.device)
    width = max(length for _, length in leaves)
    lane_width = width//8*8
    offsets = torch.arange(lane_width, device=values.device)
    indices = starts[:, None]+offsets[None]
    body = values[..., indices.clamp(max=n-1)]
    body = body * (offsets[None] < (lengths//8*8)[:, None])
    accum = body[..., :8]
    for index in range(8, lane_width, 8):
        accum = accum + body[..., index:index+8]
    total = ((accum[..., 0]+accum[..., 1])+(accum[..., 2]+accum[..., 3])) + \
            ((accum[..., 4]+accum[..., 5])+(accum[..., 6]+accum[..., 7]))
    for index in range(max(length % 8 for _, length in leaves)):
        tail_index = starts+lengths//8*8+index
        tail = values[..., tail_index.clamp(max=n-1)]
        total = total + tail*(index < lengths % 8)
    def combine(node):
        return total[..., node] if isinstance(node, int) else combine(node[0])+combine(node[1])
    return combine(topology)


class GpuCandidateVerifier:
    def __init__(self, template, membership, plot_edge, batch_size=512):
        from bw_gpu_refinement import GpuRefinementUnavailable, _checked_footprint
        import torch
        import torch.nn.functional as functional
        if not torch.cuda.is_available():
            raise GpuRefinementUnavailable('CUDA candidate verification is unavailable')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be positive')
        started = time.perf_counter()
        self.torch, self.functional = torch, functional
        self.template, self.membership, self.plot_edge = template, membership, plot_edge
        self.h, self.w = template.edge.shape
        self.batch_size = min(batch_size, max(1, 128 * 1024**2 // (self.h*self.w*64)))
        self.threshold = .45 if template.ink.achromatic else .24
        if membership.dtype != np.float32 or membership.shape != plot_edge.shape:
            raise GpuRefinementUnavailable('Candidate verification requires aligned float32 membership')
        footprint, check = _checked_footprint(max(1.25, .08*template.diameter), self.h, self.w)
        small_r = max(1, int(round(.06*template.diameter)))
        contour_r = max(1, int(round(.08*template.diameter)))
        small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*small_r+1, 2*small_r+1))
        contour = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*contour_r+1, 2*contour_r+1))
        band = cv2.dilate(template.edge.astype(np.uint8), contour) > 0
        self.inside = band & template.mask
        self.outside = band & ~template.mask
        kernels = (footprint, small, contour, contour, contour)
        side = max(k.shape[0] for k in kernels)
        weights = np.zeros((5, 1, side, side), np.float32)
        for index, kernel in enumerate(kernels):
            pad = (side-kernel.shape[0])//2
            weights[index, 0, pad:pad+kernel.shape[0], pad:pad+kernel.shape[1]] = kernel
        self.kernels = torch.as_tensor(weights, device='cuda')
        self.padding = side//2
        self.stats = dict(backend='cuda_multi_candidate_morphology', used_cuda=False,
            input_candidates=0, gpu_batches=0, max_candidates_per_batch=0,
            effective_batch_size=self.batch_size, setup_seconds=0., prepare_seconds=0.,
            footprint_validation=check,
            policy='GPU binary morphology across candidates; CPU original weighted reductions and decisions',
            timing_scope='wall time including patch gather, transfers and synchronization')
        self.stats['setup_seconds'] = time.perf_counter()-started
        self._resident = None

    def _scoring_setup(self, plot_orientation, geometry):
        from partial_swatch_detector import _candidate_geometry, _orientation_bin
        from bw_gpu_refinement import GpuRefinementUnavailable
        torch = self.torch
        if self._resident is not None:
            if self._resident['orientation_source'] is not plot_orientation:
                raise ValueError('A candidate verifier cannot reuse a different orientation image')
            return self._resident
        started = time.perf_counter()
        g = _candidate_geometry(self.template) if geometry is None else geometry
        if plot_orientation.shape != self.membership.shape or plot_orientation.dtype != np.float32:
            raise GpuRefinementUnavailable('CUDA full candidate scoring requires aligned float32 plot orientation')
        if self.template.orientation.dtype != np.float32 or any(
                g[key].dtype != np.float32 for key in ('edge_weight', 'marker_weight')):
            raise GpuRefinementUnavailable('CUDA full candidate scoring requires float32 template orientation and weights')
        def gpu(value, dtype=None):
            return torch.as_tensor(np.ascontiguousarray(value), device='cuda', dtype=dtype)
        yy, xx = np.indices((self.h, self.w))
        t = self.template
        radial = _orientation_bin(np.mod(np.arctan2(yy-.5*(self.h-1), xx-.5*(self.w-1)),
                                        2.*np.pi), 8)
        state = dict(orientation_source=plot_orientation,
            membership=gpu(self.membership), edge=gpu(self.plot_edge, torch.bool),
            orientation=gpu(plot_orientation, torch.float32),
            template_edge=gpu(t.edge, torch.bool), template_mask=gpu(t.mask, torch.bool),
            inside=gpu(self.inside), outside=gpu(self.outside),
            edge_weight=gpu(g['edge_weight']), marker_weight=gpu(g['marker_weight']),
            distance_ok=gpu(g['distance_to_template'] <= max(1.25, .08*t.diameter)),
            centre_disk=gpu(g['centre_disk']),
            centre_count=int(g['centre_disk'].sum()),
            template_centre_fill=float(t.mask[g['centre_disk']].mean()),
            edge_denominator=max(float(g['edge_weight'].sum()), 1.),
            marker_denominator=max(float(g['marker_weight'].sum()), 1.),
            edge_count=max(int(t.edge.sum()), 1),
            template_bins=gpu(_orientation_bin(t.orientation, 24)),
            sector=gpu(g['contour']['sector_index']), radial=gpu(radial),
            xx=gpu(xx), yy=gpu(yy))
        self._resident = state
        self.stats.update(backend='cuda_multi_candidate_scoring',
            policy='GPU patch gather, morphology, asymmetric metrics, gates and score; CPU indexed hypothesis counts and Detection construction',
            scored_candidates=0, compact_result_rows=0, mask_cpu_readbacks=0,
            cpu_reference_candidates=0, cpu_evidence_seconds=0., scoring_seconds=0.,
            resident_upload_seconds=time.perf_counter()-started)
        return state

    def verify_many(self, centers, plot_orientation, plot_area, grid_step, evidence, geometry=None):
        """Return input-ordered Detection/None results without CPU pixel scoring.

        ``evidence`` contains (grid_stride, hypotheses) per center. Its compact
        nearby-hypothesis counts are prepared on CPU with the existing exact
        spatial index; all pixel-level metrics and acceptance decisions are
        calculated in bounded GPU batches. No template/marker thresholds change.
        """
        from partial_swatch_detector import Detection, _IndexedHypotheses
        if len(evidence) != len(centers):
            raise ValueError('Every candidate requires its own voting evidence')
        if grid_step < 1:
            raise ValueError('grid_step must be positive')
        if not centers:
            return []
        torch = self.torch
        s = self._scoring_setup(plot_orientation, geometry)
        started = time.perf_counter()
        evidence_started = time.perf_counter()
        compact_evidence = []
        radius = max(2., .22*self.template.diameter)
        for (x, y), (stride, hypotheses) in zip(centers, evidence, strict=True):
            nearby = hypotheses.near(x, y, radius) if isinstance(hypotheses, _IndexedHypotheses) else [
                h for h in hypotheses if (h.x-x)**2+(h.y-y)**2 <= radius**2]
            cells = len({(h.cell_left, h.cell_top) for h in nearby})
            multiplicity = float(max(1, int(round(grid_step/max(stride, 1))))**2)
            compact_evidence.append((len(nearby), math.ceil(cells/multiplicity), multiplicity))
        self.stats['cpu_evidence_seconds'] += time.perf_counter()-evidence_started
        self.stats['input_candidates'] += len(centers)
        results = []
        h_image, w_image = self.membership.shape
        # NumPy 2 uses float32 for np.float32 / Python float; NumPy 1 promotes
        # that expression to float64. Follow the CPU oracle on either host.
        scalar_division_is_float32 = np.asarray(np.float32(1.)/float(3.)).dtype == np.float32
        # CUDA's scalar-divisor optimization multiplies by its reciprocal;
        # NumPy/Python perform a correctly rounded divide. A device scalar
        # denominator requests true division and avoids one-ULP score changes.
        divisors = {}
        def divide(value, denominator):
            if denominator not in divisors:
                divisors[denominator] = torch.tensor(denominator, device='cuda', dtype=torch.float64)
            return value/divisors[denominator]
        def coverage(numerator, denominator):
            value = divide(numerator.to(torch.float64), denominator)
            return value.to(torch.float32).to(torch.float64) if scalar_division_is_float32 else value
        with torch.inference_mode(), torch.backends.cudnn.flags(
                benchmark=False, deterministic=True, allow_tf32=False):
            for start in range(0, len(centers), self.batch_size):
                chunk = centers[start:start+self.batch_size]
                count = len(chunk)
                coords = torch.tensor(chunk, device='cuda', dtype=torch.float64)
                left = torch.round(coords[:, 0]-.5*(self.w-1)).to(torch.int64)
                top = torch.round(coords[:, 1]-.5*(self.h-1)).to(torch.int64)
                px = left[:, None, None]+s['xx']
                py = top[:, None, None]+s['yy']
                valid = (px >= 0) & (px < w_image) & (py >= 0) & (py < h_image)
                iy, ix = py.clamp(0, h_image-1), px.clamp(0, w_image-1)
                edge = s['edge'][iy, ix] & valid
                ink = (s['membership'][iy, ix] >= self.threshold) & valid
                orientation = torch.where(valid, s['orientation'][iy, ix], 0.)
                channels = torch.stack((edge, ink, ink, ink & s['inside'], ink & s['outside']), 1)
                masks = self.functional.conv2d(channels.to(torch.float32), self.kernels,
                    padding=self.padding, groups=5) >= .5
                edge_near, dilated, ink_near, inside_near, outside_near = masks.unbind(1)
                matched = s['template_edge'] & edge_near
                foreground = s['template_mask'] & dilated
                direct = s['template_mask'] & ink
                weighted = torch.stack((matched*s['edge_weight'], foreground*s['marker_weight'],
                                        direct*s['marker_weight']), 1).flatten(2)
                sums = _numpy_pairwise_sum(weighted)
                contour_coverage = coverage(sums[:, 0], s['edge_denominator'])
                foreground_coverage = coverage(sums[:, 1], s['marker_denominator'])
                direct_coverage = coverage(sums[:, 2], s['marker_denominator'])
                def number(mask):
                    return mask.flatten(1).sum(1)
                edge_count = number(edge)
                precision = number(edge & s['distance_ok']).to(torch.float64)/edge_count.clamp(min=1)
                visible = matched
                occluded = s['template_edge'] & ~visible & inside_near & outside_near
                mismatch = s['template_edge'] & ~visible & ~occluded
                sector_ids = torch.arange(8, device='cuda')[:, None, None]
                sector_mask = s['sector'][None] == sector_ids
                sector_edge = sector_mask & s['template_edge']
                required = sector_mask & s['inside']
                sector_counts = sector_edge.flatten(1).sum(1)
                required_counts = required.flatten(1).sum(1)
                def sector_fraction(mask, template_mask, denominator):
                    counts = (mask[:, None] & template_mask[None]).flatten(2).sum(2)
                    return counts.to(torch.float64)/denominator.clamp(min=1)
                required_coverage = sector_fraction(ink_near, required, required_counts)
                visible_fraction = sector_fraction(visible, sector_edge, sector_counts)
                occluded_fraction = sector_fraction(occluded, sector_edge, sector_counts)
                mismatch_fraction = sector_fraction(mismatch, sector_edge, sector_counts)
                sector_exists = sector_counts > 0
                visible_sectors = ((visible_fraction >= .18) & sector_exists).sum(1)
                mismatch_sectors = ((required_coverage < .45) & (mismatch_fraction >= .35) & sector_exists).sum(1)
                supported_sectors = ((required_coverage >= .55) &
                    (visible_fraction+occluded_fraction >= .18) & sector_exists).sum(1)
                plot_centre_fill = divide(number(ink & s['centre_disk']).to(torch.float64), s['centre_count'])
                similarity = 1.-torch.abs(s['template_centre_fill']-plot_centre_fill)
                radial = ((foreground[:, None] & (s['radial'][None] == sector_ids)[None])
                          .flatten(2).any(2)).sum(1)
                occupied_x, occupied_y = ink.any(1), ink.any(2)
                xrange, yrange = torch.arange(self.w, device='cuda'), torch.arange(self.h, device='cuda')
                horizontal = torch.where(occupied_x.any(1),
                    torch.where(occupied_x, xrange, -1).amax(1)-torch.where(occupied_x, xrange, self.w).amin(1)+1, 0)
                vertical = torch.where(occupied_y.any(1),
                    torch.where(occupied_y, yrange, -1).amax(1)-torch.where(occupied_y, yrange, self.h).amin(1)+1, 0)
                tau = float(np.float32(2.*np.pi))
                angles = torch.remainder(orientation, tau)*24
                plot_bins = torch.floor(divide(angles.to(torch.float64), tau).to(torch.float32)).to(torch.int64) % 24
                difference = torch.minimum((s['template_bins']-plot_bins) % 24,
                                           (plot_bins-s['template_bins']) % 24)
                orientation_match = matched & (difference <= 2)
                bin_counts = torch.zeros((count, 24), device='cuda', dtype=torch.int64)
                bin_counts.scatter_add_(1, s['template_bins'].flatten()[None].expand(count, -1),
                                        orientation_match.flatten(1).to(torch.int64))
                used_bins = (bin_counts > 0).sum(1)
                # Python rounds each (center-offset+pixel), not the patch's
                # rounded origin. Keep float64 and ties-to-even for half pixels.
                gx = torch.div(torch.round(coords[:, 0, None, None]-.5*(self.w-1)+s['xx']).to(torch.int64)-plot_area[0],
                               grid_step, rounding_mode='floor')
                gy = torch.div(torch.round(coords[:, 1, None, None]-.5*(self.h-1)+s['yy']).to(torch.int64)-plot_area[1],
                               grid_step, rounding_mode='floor')
                gx = gx-gx.flatten(1).amin(1)[:, None, None]
                gy = gy-gy.flatten(1).amin(1)[:, None, None]
                cells_w = math.ceil(self.w/grid_step)+2
                cells_h = math.ceil(self.h/grid_step)+2
                grid_counts = torch.zeros((count, cells_w*cells_h), device='cuda', dtype=torch.int64)
                grid_counts.scatter_add_(1, (gy*cells_w+gx).flatten(1), matched.flatten(1).to(torch.int64))
                support_grid = (grid_counts > 0).sum(1)
                ev = torch.tensor(compact_evidence[start:start+count], device='cuda', dtype=torch.float64)
                supporting = torch.maximum(support_grid, ev[:, 1].to(torch.int64))
                diameter = self.template.diameter
                template_fill = s['template_centre_fill']
                if template_fill >= .65:
                    centre_ok = plot_centre_fill >= max(.32, .45*template_fill)
                elif template_fill <= .35:
                    centre_ok = plot_centre_fill <= .70
                else:
                    centre_ok = similarity >= .45
                accepted = ((edge_count > 0) & (number(matched) > 0) &
                    (supporting >= (3 if diameter < 9 else 4)) &
                    (supported_sectors >= (3 if diameter < 9 else 4)) & (mismatch_sectors < 3) &
                    (used_bins >= (4 if diameter < 9 else 5)) & (radial >= 4) &
                    (horizontal >= .48*diameter) & (vertical >= .48*diameter) &
                    (contour_coverage >= (.28 if diameter >= 9 else .36)) &
                    (foreground_coverage >= (.34 if diameter >= 9 else .42)) &
                    (direct_coverage >= (.45 if diameter < 9 else .32 if diameter < 20 else .25)) & centre_ok)
                vote = ((ev[:, 0]/ev[:, 2])/supporting.clamp(min=3)).clamp(max=1.)
                spatial = divide(supporting.to(torch.float64), 5.).clamp(max=1.)
                orientation_strength = (used_bins.to(torch.float64)/8.).clamp(max=1.)
                asymmetric = (divide(number(visible).to(torch.float64), s['edge_count'])+
                              divide(number(occluded).to(torch.float64), s['edge_count'])).clamp(max=1.)
                signature = (visible_sectors.to(torch.float64)/2.).clamp(max=1.)
                score = (.21*asymmetric+.20*foreground_coverage+.17*direct_coverage+
                         .10*similarity+.10*vote+.10*spatial+.07*orientation_strength+
                         .05*signature-.18*(mismatch_sectors.to(torch.float64)/8.))
                rows = torch.stack((accepted, score, contour_coverage, precision, foreground_coverage,
                    direct_coverage, plot_centre_fill, similarity, supporting, used_bins, radial, ev[:, 0],asymmetric), 1).cpu().numpy()
                self.stats['used_cuda'] = True
                self.stats['gpu_batches'] += 1
                self.stats['max_candidates_per_batch'] = max(self.stats['max_candidates_per_batch'], count)
                self.stats['scored_candidates'] += count
                self.stats['compact_result_rows'] += count
                for (x, y), row in zip(chunk, rows, strict=True):
                    if not row[0]:
                        results.append(None)
                        continue
                    from partial_swatch_detector import _extract_aligned_patch
                    from bw_boundary_evidence_v46 import grid_score_adjustment, enabled
                    delta,boundary=0.,{}
                    if enabled(self.template):
                        patch=_extract_aligned_patch(self.membership,x,y,self.template.mask.shape)
                        delta,boundary=grid_score_adjustment(self.template,patch,float(row[12]))
                        self.stats['boundary_evidence_backend']='cpu_shared_normal_profiles'
                    results.append(Detection(template=self.template.key, x=float(x), y=float(y),
                        score=float(row[1])+delta, contour_coverage=float(row[2]), contour_precision=float(row[3]),
                        foreground_coverage=float(row[4]), direct_foreground_coverage=float(row[5]),
                        centre_fill=float(row[6]), centre_fill_similarity=float(row[7]),
                        supporting_cells=int(row[8]), orientation_bins=int(row[9]), radial_sectors=int(row[10]),
                        raw_hypotheses=int(row[11]), swatch_id=self.template.swatch_id,
                        shape_hint=self.template.shape_hint or self.template.name,boundary_evidence=boundary))
        self.stats['scoring_seconds'] += time.perf_counter()-started
        return results

    def prepare_many(self, centers):
        """Yield bounded batches of (original position, exact Boolean masks)."""
        from partial_swatch_detector import _extract_aligned_patch
        torch = self.torch
        started = time.perf_counter()
        self.stats['input_candidates'] += len(centers)
        names = ('edge_near', 'dilated_plot_ink', 'ink_near', 'inside_ink_near', 'outside_ink_near')
        with torch.inference_mode(), torch.backends.cudnn.flags(
                benchmark=False, deterministic=True, allow_tf32=False):
            for start in range(0, len(centers), self.batch_size):
                chunk = centers[start:start+self.batch_size]
                edge = np.stack([_extract_aligned_patch(self.plot_edge, x, y, (self.h, self.w))
                                 for x, y in chunk])
                ink = np.stack([_extract_aligned_patch(self.membership, x, y, (self.h, self.w))
                                >= self.threshold for x, y in chunk])
                channels = np.stack((edge, ink, ink, ink & self.inside, ink & self.outside), axis=1)
                value = torch.as_tensor(channels, device='cuda', dtype=torch.float32)
                masks = (self.functional.conv2d(value, self.kernels, padding=self.padding, groups=5)
                         >= .5).cpu().numpy()
                self.stats['used_cuda'] = True
                self.stats['gpu_batches'] += 1
                self.stats['max_candidates_per_batch'] = max(self.stats['max_candidates_per_batch'], len(chunk))
                for offset, value in enumerate(masks):
                    yield start+offset, {name: value[i] for i, name in enumerate(names)}
        self.stats['prepare_seconds'] += time.perf_counter()-started
        # The generator includes caller's CPU decisions between yields, so this
        # is deliberately a whole prepare/decision wall time, not GPU-only time.

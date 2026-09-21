"""Exact, optional CUDA acceleration of grid-local edge-to-centre votes.

Template orientation bins and every acceptance rule come from the existing CPU
detector. A global edge's votes are generated once and reused by overlapping
cells. Integer CUDA arithmetic represents the template's exact half-pixel
offset lattice. Many cells are grouped together on CUDA, retaining the CPU
dictionary's first-seen tie ordering. Only the compact top candidates return
to CPU for the original double-precision scalar score and output objects.
Torch is deliberately imported only when CUDA work is actually requested.
"""
from __future__ import annotations

import math
import time

import numpy as np


class GpuGridVotesUnavailable(RuntimeError):
    """Exact CUDA voting is unavailable; the caller may use the CPU helper."""


def _origins(plot_area, grid_step, grid_stride):
    x0, y0, x1, y1 = plot_area
    phase_count = max(1, int(round(grid_step / max(grid_stride, 1))))
    phases = sorted({int(round(i * grid_step / phase_count))
                     for i in range(phase_count)})
    tops = sorted({top for phase in phases for top in range(y0 + phase, y1, grid_step)})
    lefts = sorted({left for phase in phases for left in range(x0 + phase, x1, grid_step)})
    return tops, lefts


def _offset_bank(template, bins, detector):
    table = detector._template_r_table(template, bins)
    lists = []
    for edge_bin in range(bins):
        arrays = [table[(edge_bin + delta) % bins] for delta in (-1, 0, 1)
                  if len(table[(edge_bin + delta) % bins])]
        lists.append(np.concatenate(arrays, axis=0) if arrays else np.empty((0, 2), np.float32))
    lengths = np.asarray([len(values) for values in lists], np.int64)
    width = int(lengths.max(initial=0))
    bank = np.zeros((bins, width, 2), np.int64)
    for index, values in enumerate(lists):
        doubled = values.astype(np.float64) * 2
        if not np.isfinite(doubled).all() or not np.array_equal(doubled, np.rint(doubled)):
            raise GpuGridVotesUnavailable('Template offsets are not an exact half-pixel lattice')
        if np.abs(doubled).max(initial=0) >= 2**20:
            raise GpuGridVotesUnavailable('Template dimensions exceed certified float32 lattice range')
        bank[index, :len(values)] = doubled.astype(np.int64)
    return bank, lengths


def _round_half_units(torch, doubled, quantum):
    """Round (doubled / 2) / quantum to nearest integer, ties to even.

    Within the certified coordinate range, the CPU float32 addition is exact
    and division cannot round a non-tie across a half-integer boundary. Exact
    ties themselves are representable. Integer division avoids GPU float
    division variations while reproducing NumPy/Python banker rounding.
    """
    denominator = 2 * quantum
    floor = torch.div(doubled, denominator, rounding_mode='floor')
    remainder = doubled - floor * denominator
    up = (remainder > quantum) | ((remainder == quantum) & ((floor & 1) != 0))
    return floor + up.to(torch.int64)


def _stable_top_votes(keys, local_top_k):
    """Reference-only CPU ordering helper; not used by the CUDA execution path."""
    if not len(keys):
        return []
    unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
    order = np.lexsort((first, -counts))[:local_top_k]
    return [(int(unique[i]), int(counts[i])) for i in order]


def _aggregate_cells_cuda(torch, all_keys, pointers, counts, pixel_ids,
                          cell_origins, plot_area, grid_step, centre_space,
                          local_top_k, cell_batch_size, max_cell_vote_pairs,
                          bank_width):
    """Group resident votes for many cells; transfer only their top candidates.

    Cell-major, row-major pixel order followed by offset order is the original
    dictionary insertion stream. Scatter-min obtains its first occurrence for
    each (cell, centre), and a unique integer sort key orders count-descending
    ties by that first occurrence. No float reduction or unstable tie is used.
    """
    device=all_keys.device
    x0,y0,x1,y1=plot_area
    roi_height,roi_width=pixel_ids.shape
    height,width=min(grid_step,roi_height),min(grid_step,roi_width)
    upper_per_cell=height*width*bank_width
    if upper_per_cell>max_cell_vote_pairs:
        raise GpuGridVotesUnavailable('One cell exceeds the bounded GPU vote-pair workspace; no votes truncated')
    batch=min(cell_batch_size,max(1,max_cell_vote_pairs//max(upper_per_cell,1)))
    # The integer ranking key is bounded before arithmetic, rather than letting
    # int64 wrap alter the count/first-seen order.
    if batch*(max_cell_vote_pairs+1)**2>=np.iinfo(np.int64).max:
        raise GpuGridVotesUnavailable('Cell-batch stable ranking key could exceed int64')
    if batch*centre_space>=np.iinfo(np.int64).max:
        raise GpuGridVotesUnavailable('Cell/centre grouping key could exceed int64')
    dy,dx=torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing='ij')
    dy,dx=dy.reshape(1,-1),dx.reshape(1,-1)
    origins=torch.as_tensor(cell_origins,dtype=torch.int64,device=device)
    output=[]
    report=dict(gpu_cell_batches=0,max_cells_per_batch=0,cell_batch_size=int(batch),
                gpu_grouped_cell_votes=0,compact_hypothesis_transfers=0,
                aggregation_backend='cuda',global_vote_cpu_transfers=0,
                gpu_resident_votes=True)
    local_cells=torch.zeros((),dtype=torch.int64,device=device)
    torch.cuda.synchronize(device)
    started=time.perf_counter()
    for start in range(0,len(cell_origins),batch):
        stop=min(start+batch,len(cell_origins)); n=stop-start
        ys=origins[start:stop,0,None]-y0+dy
        xs=origins[start:stop,1,None]-x0+dx
        inside=(ys<roi_height)&(xs<roi_width)
        ids=pixel_ids[ys.clamp(max=roi_height-1),xs.clamp(max=roi_width-1)].to(torch.int64)
        visible=inside&(ids>=0)
        cell_counts=visible.sum(dim=1)
        eligible=cell_counts>=2
        local_cells+=eligible.sum()
        visible&=eligible[:,None]
        edge_ids=ids[visible]
        pixel_cells=torch.arange(n,device=device)[:,None].expand_as(ids)[visible]
        amounts=counts[edge_ids]
        # One scalar controls bounded allocation for this entire cell batch;
        # no global edge votes, counts, or accumulators are copied to the host.
        total=int(amounts.sum().item())
        report['gpu_cell_batches']+=1
        report['max_cells_per_batch']=max(report['max_cells_per_batch'],n)
        report['gpu_grouped_cell_votes']+=total
        if not total:
            continue
        if total>max_cell_vote_pairs:
            raise GpuGridVotesUnavailable('Cell batch exceeded its certified workspace bound')
        local_starts=torch.cumsum(amounts,dim=0)-amounts
        vote_indices=(torch.repeat_interleave(pointers[edge_ids],amounts,output_size=total)
                      +torch.arange(total,device=device)
                      -torch.repeat_interleave(local_starts,amounts,output_size=total))
        vote_cells=torch.repeat_interleave(pixel_cells,amounts,output_size=total)
        grouped_keys=vote_cells*centre_space+all_keys[vote_indices]
        unique,inverse,occurrences=torch.unique(grouped_keys,sorted=True,return_inverse=True,return_counts=True)
        first=torch.full_like(unique,total)
        first.scatter_reduce_(0,inverse,torch.arange(total,device=device),reduce='amin',include_self=True)
        cells=torch.div(unique,centre_space,rounding_mode='floor')
        rank_key=(cells*(total+1)**2+(total-occurrences)*(total+1)+first)
        order=torch.argsort(rank_key)
        ordered_cells=cells[order]
        position=torch.arange(len(order),device=device)
        starts=torch.cat((torch.ones(1,dtype=torch.bool,device=device),ordered_cells[1:]!=ordered_cells[:-1]))
        first_in_cell=torch.cummax(torch.where(starts,position,0),dim=0)[0]
        retain=position-first_in_cell<local_top_k
        selected=order[retain]
        selected_cells=cells[selected]
        compact=torch.stack((selected_cells+start,unique[selected]%centre_space,
                             occurrences[selected],cell_counts[selected_cells]),dim=1)
        output.append(compact.cpu().numpy())
        report['compact_hypothesis_transfers']+=1
    torch.cuda.synchronize(device)
    report['gpu_aggregation_seconds']=time.perf_counter()-started
    report['local_cells']=int(local_cells.item())
    return (np.concatenate(output) if output else np.empty((0,4),np.int64)),report


def cell_hypotheses_cuda(plot_edge, plot_orientation, template, plot_area,
                         grid_step, grid_stride, orientation_bins=24,
                         local_top_k=4, batch_size=4096, *,
                         max_vote_pairs=16_000_000, return_report=False,
                         diagnostics=None, cell_batch_size=None,
                         max_cell_vote_pairs=2_000_000):
    """Drop-in output for ``partial_swatch_detector._cell_hypotheses``.

    ``batch_size`` bounds global edge pixels per CUDA chunk (also capped at
    two million padded edge/offset pairs). ``max_vote_pairs`` bounds resident vote
    storage; exceeding a bound raises GpuGridVotesUnavailable, never truncates.
    ``cell_batch_size`` defaults to batch_size, independently grouping multiple
    cells; ``max_cell_vote_pairs`` bounds that aggregation workspace.
    ``return_report=True`` returns (hypotheses, diagnostic_dict).
    Optional ``diagnostics`` is updated with the same report for callers that
    need the ordinary list return. Aggregation is CUDA; the final scalar score
    retains the original CPU math.sqrt/division operations.
    The ordinary return is the exact ordered list of CPU GridHypothesis values.
    """
    started = time.perf_counter()
    import partial_swatch_detector as detector

    edge = np.asarray(plot_edge)
    orientation = np.asarray(plot_orientation)
    if edge.ndim != 2 or orientation.shape != edge.shape:
        raise ValueError('plot edge/orientation must have equal two-dimensional shapes')
    area = tuple(plot_area)
    if len(area) != 4 or any(int(v) != v for v in area):
        raise ValueError('plot_area must contain four integer coordinates')
    x0, y0, x1, y1 = map(int, area)
    area = (x0, y0, x1, y1)
    if not (0 <= x0 < x1 <= edge.shape[1] and 0 <= y0 < y1 <= edge.shape[0]):
        raise ValueError('plot_area must be a nonempty half-open box inside the image')
    cell_batch_size=batch_size if cell_batch_size is None else cell_batch_size
    for name, value in [('grid_step', grid_step), ('grid_stride', grid_stride),
                        ('orientation_bins', orientation_bins), ('batch_size', batch_size),
                        ('max_vote_pairs', max_vote_pairs), ('cell_batch_size',cell_batch_size),
                        ('max_cell_vote_pairs',max_cell_vote_pairs)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isinstance(local_top_k, (int, np.integer)) or local_top_k < 0:
        raise ValueError('local_top_k must be a nonnegative integer')
    if not np.isfinite(template.diameter) or template.diameter <= 0:
        raise ValueError('template diameter must be positive and finite')
    if max(x1, y1) >= 2**19:
        raise GpuGridVotesUnavailable('Image coordinates exceed certified float32 lattice range')
    quantum = max(1, int(round(template.diameter * .07)))
    if quantum >= 2**20:
        raise GpuGridVotesUnavailable('Centre quantum exceeds certified float32 division range')
    template_edge = np.asarray(template.edge)
    if template_edge.ndim != 2 or np.asarray(template.orientation).shape != template_edge.shape:
        raise ValueError('template edge/orientation shapes must match')
    bank, lengths = _offset_bank(template, orientation_bins, detector)
    local_y, local_x = np.nonzero(edge[y0:y1, x0:x1])
    global_x, global_y = local_x + x0, local_y + y0
    edge_count = len(global_x)
    stats = dict(backend='cuda', used_cuda=False, global_edges=edge_count,
                 grid_step=int(grid_step), grid_stride=int(grid_stride),
                 centre_quantum=quantum, offset_bank_width=bank.shape[1],
                 generated_votes=0, cuda_batches=0, local_cells=0,
                 quantization='exact half-pixel integer banker rounding',
                 aggregation='CUDA multi-cell integer counts + first-seen stable ties; CPU scalar score only',
                 aggregation_backend='cuda',gpu_cell_batches=0,max_cells_per_batch=0,
                 cell_batch_size=cell_batch_size,gpu_aggregation_seconds=0.,
                 gpu_grouped_cell_votes=0,gpu_resident_votes=True,
                 global_vote_cpu_transfers=0,compact_hypothesis_transfers=0,
                 gpu_pixel_index_bytes=0)
    if not edge_count or not bank.shape[1] or not local_top_k:
        stats['seconds'] = time.perf_counter() - started
        if diagnostics is not None:
            diagnostics.update(stats)
        return ([], stats) if return_report else []
    angles = orientation[global_y, global_x]
    if not np.isfinite(angles).all():
        raise GpuGridVotesUnavailable('Nonfinite plot orientations are unsupported')
    edge_bins = detector._orientation_bin(angles, orientation_bins)
    proposed_pairs = int(lengths[edge_bins].sum())
    if proposed_pairs > max_vote_pairs:
        raise GpuGridVotesUnavailable(f'Exact vote storage needs {proposed_pairs} pairs; bound is {max_vote_pairs}')
    # Every possible centre is below 2**20 in absolute value, where half-pixel
    # float32 additions are exact and nearest-half-boundary distance dominates
    # division roundoff. This includes candidates filtered outside the ROI.
    if (max(x1, y1) * 2 + int(np.abs(bank).max(initial=0))) >= 2**21:
        raise GpuGridVotesUnavailable('Candidate coordinates exceed certified lattice range')
    try:
        import torch
    except ImportError as error:
        raise GpuGridVotesUnavailable('CUDA voting requires PyTorch') from error
    if not torch.cuda.is_available():
        raise GpuGridVotesUnavailable('PyTorch CUDA device is unavailable')
    device = torch.device('cuda:0')
    gpu_bank = torch.as_tensor(bank, device=device)
    gpu_lengths = torch.as_tensor(lengths, device=device)
    offset_index = torch.arange(bank.shape[1], device=device)
    chunk_size = min(int(batch_size), max(1, 2_000_000 // bank.shape[1]))
    stride_y = int(round(y1 / quantum)) + 2
    centre_space=(int(round(x1/quantum))+2)*stride_y
    counts = torch.zeros(edge_count,dtype=torch.int64,device=device)
    gpu_bins=torch.as_tensor(edge_bins,dtype=torch.int64,device=device)
    gpu_xy=torch.as_tensor(np.column_stack((global_x,global_y))*2,dtype=torch.int64,device=device)
    key_chunks = []
    torch.cuda.synchronize(device)
    gpu_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, edge_count, chunk_size):
            stop = min(start + chunk_size, edge_count)
            b = gpu_bins[start:stop]
            centres = gpu_bank[b] + gpu_xy[start:stop,None,:]
            cx, cy = centres[..., 0], centres[..., 1]
            valid = ((offset_index[None, :] < gpu_lengths[b, None]) &
                     (cx >= 2*x0) & (cx < 2*x1) & (cy >= 2*y0) & (cy < 2*y1))
            qx = _round_half_units(torch, cx, quantum)
            qy = _round_half_units(torch, cy, quantum)
            keys = qx * stride_y + qy
            # Boolean indexing traverses pixel-major/offset-major order, the
            # same order as the original nested pixel, delta and offset loops.
            key_chunks.append(keys[valid])
            counts[start:stop] = valid.sum(dim=1)
            stats['cuda_batches'] += 1
        all_keys = torch.cat(key_chunks)
        del key_chunks
        pointers = torch.cat((torch.zeros(1,dtype=torch.int64,device=device),torch.cumsum(counts,dim=0)))
        torch.cuda.synchronize(device)
        stats.update(used_cuda=True,device=torch.cuda.get_device_name(device),
                     generated_votes=len(all_keys),cuda_vote_seconds=time.perf_counter()-gpu_started)
        # Four bytes per ROI pixel, on GPU only. No full-source or host int64
        # index is allocated; edges outside the original plot never participate.
        if edge_count>np.iinfo(np.int32).max:
            raise GpuGridVotesUnavailable('ROI edge index exceeds int32 capacity')
        pixel_ids=torch.full((y1-y0,x1-x0),-1,dtype=torch.int32,device=device)
        pixel_ids[gpu_xy[:,1]//2-y0,gpu_xy[:,0]//2-x0]=torch.arange(edge_count,dtype=torch.int32,device=device)
        stats['gpu_pixel_index_bytes']=pixel_ids.numel()*pixel_ids.element_size()
        tops,lefts=_origins(area,grid_step,grid_stride)
        tt,ll=np.meshgrid(tops,lefts,indexing='ij')
        origins=np.column_stack((tt.ravel(),ll.ravel()))
        compact,aggregation_report=_aggregate_cells_cuda(torch,all_keys,pointers,counts,pixel_ids,
            origins,area,grid_step,centre_space,local_top_k,cell_batch_size,max_cell_vote_pairs,bank.shape[1])
        stats.update(aggregation_report)
    template_count=max(int(template_edge.sum()),1)
    hypotheses=[]
    for cell_id,key,votes,edge_pixels in compact:
        cell_id,key,votes,edge_pixels=map(int,(cell_id,key,votes,edge_pixels))
        normalizer=max(2.,math.sqrt(edge_pixels*template_count))
        score=float(votes/normalizer)
        if votes<2 or score<.10:
            continue
        cell_y,cell_x=divmod(cell_id,len(lefts))
        qx,qy=divmod(key,stride_y)
        hypotheses.append(detector.GridHypothesis(
            x=float(qx*quantum),y=float(qy*quantum),cell_x=cell_x,cell_y=cell_y,
            cell_left=lefts[cell_x],cell_top=tops[cell_y],local_votes=votes,local_score=score))
    stats.update(hypotheses=len(hypotheses), seconds=time.perf_counter()-started)
    if diagnostics is not None:
        diagnostics.update(stats)
    return (hypotheses, stats) if return_report else hypotheses

"""Experimental grid-aware cost; original image/color evidence are untouched."""
from dataclasses import replace

import numpy as np

from color_path_directional_v46 import DirectionalConfig, sample_columns, trace_directional


def trace_grid_aware(soft, grid_likelihood, lo, hi, *, weight=1., config=None):
    """Minimize sum(1-color_support + weight*grid_likelihood) + turning.

    An affine conversion lets the unchanged directional solver evaluate this
    objective exactly. The converted field is a COST encoding, not observed
    color evidence. Returned ``val`` always samples original ``soft``.
    """
    original = np.asarray(soft, float)
    grid = np.asarray(grid_likelihood, float)
    if original.ndim != 2 or original.shape != grid.shape:
        raise ValueError("soft/grid must be matching 2D arrays")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in (original, grid)):
        raise ValueError("soft/grid must be finite in [0,1]")
    cfg = config or DirectionalConfig()
    cost_encoding = (original + weight * (1-grid)) / (1+weight)
    solved = trace_directional(cost_encoding, lo, hi, 0, original.shape[0]-1,
                               replace(cfg, turn_weight=cfg.turn_weight/(1+weight)))
    xy = solved["path"]
    values = sample_columns(original, xy[:, 0].astype(int), xy[:, 1])
    penalties = sample_columns(grid, xy[:, 0].astype(int), xy[:, 1])
    color_cost = float(np.sum(1-values))
    grid_cost = float(weight * penalties.sum())
    turn_cost = float(solved["cost"]["turning"]*(1+weight))
    total = color_cost + grid_cost + turn_cost
    assert np.isclose(total, solved["cost"]["total"]*(1+weight), rtol=1e-9, atol=1e-7)
    return dict(path=xy, val=values, grid_likelihood=penalties,
                original_color_observed=values >= .18,
                nongrid_color_observed=(values >= .18) & (penalties < .5),
                objective=dict(color=color_cost, grid=grid_cost, turning=turn_cost, total=total),
                weight=float(weight), config=cfg.__dict__,
                interpretation="Optimized path samples, not measured data points; source pixels unchanged")

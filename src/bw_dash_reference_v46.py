"""Recover short dash chains for v46 references, never isolated short ink.

The legacy detector keeps its 5-pixel minimum. This supplement groups at least
three independently connected short fragments BEFORE applying a total-length
test. No marker locations or expected concentration values are used.
"""
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

VERSION = 'v46_short_dash_chains_v1'


def recover(debug, fit_segment, diameter):
    fg = np.asarray(debug['fg_mask'], np.uint8)
    _, components = cv2.connectedComponents(fg, connectivity=8)
    fragments = []
    for cluster in debug['clusters']:
        segment = fit_segment(cluster, min_len=2., min_elongation=1.5)
        if segment is None or fit_segment(cluster) is not None:
            continue
        segment = np.asarray(segment, float)
        vector = segment[2:]-segment[:2]
        labels, counts = np.unique(components[cluster[:, 0], cluster[:, 1]], return_counts=True)
        component = int(labels[np.argmax(counts)])
        if component:
            fragments.append(dict(segment=segment, center=(segment[:2]+segment[2:])/2,
                                  tangent=vector/np.linalg.norm(vector), component=component))
    report = dict(version=VERSION, short_fragment_count=len(fragments), accepted_chain_count=0,
                  minimum_independent_fragments=3, minimum_fragment_length_px=2.,
                  legacy_minimum_unchanged=True, chains=[])
    if len(fragments) < 3:
        return [], report
    centers = np.array([f['center'] for f in fragments])
    ends = np.array([f['segment'].reshape(2, 2) for f in fragments])
    directions = np.array([f['tangent'] for f in fragments])
    ids = np.array([f['component'] for f in fragments])
    tree = cKDTree(centers)
    max_gap = min(8., max(3., .5*float(diameter)))
    radius = 3*(max_gap+5.)
    cos_angle = math.cos(math.radians(25))  # 2–4px raster fragments have quantised angles
    chains = {}
    for i, first in enumerate(fragments):
        # Bound both search extent and density; do not link distant text/noise.
        distances, nearby = tree.query(centers[i], k=min(64, len(fragments)), distance_upper_bound=radius)
        nearby = np.atleast_1d(nearby)[np.isfinite(np.atleast_1d(distances))]
        for j in nearby:
            if j <= i or ids[j] == ids[i]:
                continue
            delta = centers[j]-centers[i]
            length = float(np.linalg.norm(delta))
            if not 2. < length <= max_gap+5.:
                continue
            tangent = delta/length
            normal = np.array([-tangent[1], tangent[0]])
            lateral = np.max(np.abs((ends[nearby]-centers[i]) @ normal), axis=1)
            eligible = nearby[(lateral <= 1.6) & (np.abs(directions[nearby] @ tangent) >= cos_angle)]
            if len(np.unique(ids[eligible])) < 3:
                continue
            # One fragment per foreground component: duplicate direction bins
            # or several edges of ONE marker cannot become three dashes.
            chosen = {}
            for k in eligible:
                chosen.setdefault(int(ids[k]), int(k))
            members = sorted(chosen.values(), key=lambda k: float((centers[k]-centers[i]) @ tangent))
            intervals = [(float(((ends[k]-centers[i]) @ tangent).min()),
                          float(((ends[k]-centers[i]) @ tangent).max()), k) for k in members]
            runs = [[]]
            for entry in intervals:
                if runs[-1] and entry[0]-runs[-1][-1][1] > max_gap:
                    runs.append([])
                runs[-1].append(entry)
            for run in runs:
                keys = tuple(sorted(e[2] for e in run))
                if len(run) < 3 or i not in keys or j not in keys or keys in chains:
                    continue
                pixels = np.concatenate([ends[k] for k in keys])
                center = pixels.mean(axis=0)
                _, _, vt = np.linalg.svd(pixels-center, full_matrices=False)
                direction = vt[0]
                perpendicular = np.array([-direction[1], direction[0]])
                if np.max(np.abs((pixels-center) @ perpendicular)) > 1.6:
                    continue
                projection = (pixels-center) @ direction
                span = float(np.ptp(projection))
                observed_length = sum(np.linalg.norm(ends[k, 1]-ends[k, 0]) for k in keys)
                if span < 9. or observed_length/span < .3:
                    continue
                # Output ends at the first/last observed fragment, no extrapolation.
                result = np.r_[center+projection.min()*direction, center+projection.max()*direction]
                chains[keys] = dict(segment=result.tolist(), independent_fragments=len(keys),
                                    span_px=span, observed_fraction=min(1., float(observed_length/span)))
    # Remove strict subsets of already validated local chains, not parallel lines.
    accepted = []
    accepted_keys = []
    for keys, row in sorted(chains.items(), key=lambda item: (-len(item[0]), item[0])):
        if any(set(keys).issubset(prior) for prior in accepted_keys):
            continue
        accepted_keys.append(set(keys)); accepted.append(row)
    # Consolidate overlapping local windows only while ALL observed fragments
    # still fit one straight chain. Prevent repeated windows overweighting ink.
    changed = True
    while changed:
        changed = False
        for i in range(len(accepted_keys)):
            for j in range(i+1, len(accepted_keys)):
                if len(accepted_keys[i] & accepted_keys[j]) < 2:
                    continue
                keys = sorted(accepted_keys[i] | accepted_keys[j])
                pixels = np.concatenate([ends[k] for k in keys])
                center = pixels.mean(axis=0)
                _, _, vt = np.linalg.svd(pixels-center, full_matrices=False)
                direction = vt[0]
                normal = np.array([-direction[1], direction[0]])
                if np.max(np.abs((pixels-center) @ normal)) > 1.6:
                    continue
                projection = (pixels-center) @ direction
                span = float(np.ptp(projection))
                accepted_keys[i] |= accepted_keys[j]
                accepted[i] = dict(segment=np.r_[center+projection.min()*direction,
                                                 center+projection.max()*direction].tolist(),
                    independent_fragments=len(set(ids[keys])), span_px=span,
                    observed_fraction=min(1.,float(sum(np.linalg.norm(ends[k,1]-ends[k,0]) for k in keys)/span)))
                del accepted_keys[j]; del accepted[j]
                changed = True
                break
            if changed:
                break
    report.update(accepted_chain_count=len(accepted), chains=accepted, max_gap_px=max_gap)
    return [row['segment'] for row in accepted], report

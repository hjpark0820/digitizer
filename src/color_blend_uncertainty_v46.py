"""Bounded negative-evidence uncertainty at observed palette intersections.

This does NOT recolour pixels, add marker support, or declare an occluder.
Only a convex mixture of two distinct known inks, near observed pixels of
both inks, can soften a missing-ink penalty. Paper and arbitrary unknown
colours retain their original penalty. All quantities use image pixels.
"""
import cv2
import numpy as np

VERSION = 'local_palette_blend_uncertainty_v1'


def blend_uncertainty(crop, templates, membership, confidence, equivalent, paper_bgr):
    values = np.asarray(crop, np.float32)
    paper = np.asarray(paper_bgr, np.float32)
    delta = paper-values
    result = np.zeros_like(membership, dtype=np.float32)
    # Fixed small neighbourhood models raster mixing, not marker-scale gaps.
    nearby = [cv2.dilate((m*c >= .45).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
              for m, c in zip(membership, confidence)]
    ink = np.clip((np.linalg.norm(delta, axis=-1)-12.)/28., 0., 1.)
    vectors = [paper-np.asarray(t['model']['bgr'], np.float32) for t in templates]
    for i, u in enumerate(vectors):
        for j in range(i+1, len(vectors)):
            if equivalent[i, j]:
                continue
            v = vectors[j]
            uu, vv, uv = float(u@u), float(v@v), float(u@v)
            det = uu*vv-uv*uv
            if det <= .015*max(uu*vv, 1.):
                continue
            du, dv = delta@u, delta@v
            a, b = (du*vv-dv*uv)/det, (dv*uu-du*uv)/det
            # At least some of EACH ink, inside the paper/ink colour simplex.
            bounded = (a >= 0) & (b >= 0) & (a+b <= 1.05)
            balance = np.clip((np.minimum(a, b)-.06)/.14, 0., 1.)
            residual = np.linalg.norm(delta-a[..., None]*u-b[..., None]*v, axis=-1)
            fit = np.exp(-.5*(residual/10.)**2)
            mixed = fit*balance*bounded*nearby[i]*nearby[j]*ink
            for k in (i, j):
                # A confidently identified target pixel needs no waiver.
                uncertain = mixed*np.clip((.85-membership[k])/.55, 0., 1.)
                result[k] = np.maximum(result[k], uncertain)
    return result


def window_kwargs(evidence, index):
    """One contract for calibration, final windows and the triangle guard."""
    maps = evidence.get('blend_uncertainty')
    t=evidence['templates'][index]
    return dict(blend_uncertainty=None if maps is None else maps[index],
        target_confidence=None if evidence.get('colour_confidence') is None else evidence['colour_confidence'][index],
        raw_bgr=evidence.get('crop_bgr'),filled_prior=evidence.get('priors',{}).get(str(t['id'])),
        native_pixel_area=float(evidence.get('scale_x',evidence.get('scale',1.)))*
                          float(evidence.get('scale_y',evidence.get('scale',1.))))

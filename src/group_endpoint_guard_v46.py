"""Do not use a locally missing line reference to erase an observed endpoint.

This is a narrow deletion safeguard, NOT an alternate marker detector. Only
high-confidence original detections with same-group ink support qualify.
Other-colour occlusion is neither white contradiction nor positive evidence.
"""
import cv2
import numpy as np

from bw_suppressed_v46 import decode_marker_mask
from group_segment_metric_v46 import _nearest_segments

VERSION = 'observed_endpoint_missing_reference_v1'


class EndpointEvidence:
    def __init__(self, gray, ignore, reference, diameter):
        self.gray = np.asarray(gray)
        if self.gray.ndim != 2 or self.gray.dtype != np.uint8:
            raise ValueError('Endpoint evidence must be a native-pixel uint8 gray plane')
        self.ignore = np.zeros_like(self.gray, bool) if ignore is None else np.asarray(ignore, bool)
        if self.ignore.shape != self.gray.shape:
            raise ValueError('Endpoint visibility dimensions mismatch')
        self.reference = np.asarray(reference, float).reshape(-1, 4)
        self.diameter = float(diameter)
        # A one-pixel tolerance handles antialiasing, without crediting occluders.
        observed = (self.gray < 160) & ~self.ignore
        self.near_ink = cv2.dilate(np.uint8(observed), np.ones((3,3),np.uint8)).astype(bool)
        self.cache = {}

    def marker(self, point):
        identity = (point['swatch_id'],point['cx'],point['cy'])
        if identity in self.cache:
            return self.cache[identity]
        # Saved masks already include the detector scale/aspect. Do not scale twice.
        mask = decode_marker_mask(point['marker_mask'])
        yy,xx = np.nonzero(mask)
        x = xx + round(point['cx']-point.get('marker_offset_x',0.))-mask.shape[1]//2
        y = yy + round(point['cy']-point.get('marker_offset_y',0.))-mask.shape[0]//2
        inside = (x>=0)&(x<self.gray.shape[1])&(y>=0)&(y<self.gray.shape[0])
        visible = np.zeros(len(x),bool)
        visible[inside] = ~self.ignore[y[inside],x[inside]]
        support = np.zeros(len(x),bool)
        support[visible] = self.near_ink[y[visible],x[visible]]
        # Evidence must be distributed around the marker, not a single dash.
        angles = np.arctan2(y-point['cy'],x-point['cx'])
        sectors = np.clip(((angles+np.pi)/(2*np.pi)*8).astype(int),0,7)
        supported = sum(np.count_nonzero(support & (sectors==i)) >= 2 for i in range(8))
        row = dict(visible_fraction=float(visible.mean()),
                   ink_support=float(support.sum()/max(1,visible.sum())),
                   supported_sectors=int(supported), visible_ink_pixels=int(support.sum()))
        row['strong_observed_marker'] = bool(row['visible_fraction']>=.5 and row['ink_support']>=.7
                                            and supported>=4 and support.sum()>=8)
        self.cache[identity] = row
        return row

    def check(self, removed, before, after):
        checks = []
        for p in removed:
            if (not p.get('original_detection',False) or p.get('tentative',False)
                    or p.get('manual_edit',False) or float(p.get('confidence',0.) or 0.)<.8):
                continue
            same = sorted((q for q in before if q['swatch_id']==p['swatch_id']),key=lambda q:q['cx'])
            remaining = sorted((q for q in after if q['swatch_id']==p['swatch_id']),key=lambda q:q['cx'])
            if len(same)<3 or len(remaining)<2:
                continue
            diameter = float(p.get('effective_diameter',p.get('source_diameter',self.diameter)))
            if p['cx']==same[0]['cx'] and remaining[0]['cx']-p['cx'] > max(2.,.5*diameter):
                neighbor = same[1]
            elif p['cx']==same[-1]['cx'] and p['cx']-remaining[-1]['cx'] > max(2.,.5*diameter):
                neighbor = same[-2]
            else:
                continue
            a=np.array([p['cx'],p['cy']]); b=np.array([neighbor['cx'],neighbor['cy']])
            length=float(np.linalg.norm(b-a))
            if length<=2*diameter:
                continue
            t=np.linspace(diameter/length,1-diameter/length,41)
            distance,_,_=_nearest_segments(a+t[:,None]*(b-a),self.reference)
            missing=float(np.mean(distance>.5*diameter))
            row=dict(point_id=p.get('point_id'),reference_missing_fraction=missing,**self.marker(p))
            row['protected']=bool(missing>.5 and row['strong_observed_marker'])
            checks.append(row)
        return dict(policy=VERSION,protected=any(r['protected'] for r in checks),checks=checks)

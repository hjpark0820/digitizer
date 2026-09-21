"""Opt-in, frozen uniform-x completeness prior; never marker-image evidence.

Coordinates are source-image x pixels (therefore log-x when the drawn axis is
logarithmic). Only original active detections establish the common lattice.
Each series is scored separately inside its own frozen path support.
"""
import numpy as np


def _x(p):
    return float(p.get('cx',p.get('x'))) if isinstance(p,dict) else float(p[0])


class XCompleteness:
    def __init__(self, rows, reference, diameter, enabled=True):
        self.enabled=False
        self.reason='disabled' if not enabled else 'insufficient_original_x_support'
        self.slots=np.array([],float);self.weights=np.array([],float)
        self.audit=dict(coordinates='source_image_x_pixels',grid_frozen=True,
                        source='original_active_only',same_x_rule='within_each_series_only')
        if not enabled or reference is None:
            return
        samples=sorted((_x(p),r['name']) for r in rows for p in r.get('init_points',[]))
        if not samples or not all(np.isfinite(x) for x,_ in samples):
            return
        diameters=[float(r['diameter']) for r in rows if r.get('diameter',0)>0]
        tol=max(2.,.35*float(np.median(diameters or [diameter])))
        clusters=[]
        for x,sid in samples:
            if (clusters and x-clusters[-1][0][0]<=2*tol
                and abs(x-np.median([p[0] for p in clusters[-1]]))<=tol):
                clusters[-1].append((x,sid))
            else:clusters.append([(x,sid)])
        # Multiple points from one series cannot outvote the other series.
        min_series=2 if len({s for _,s in samples})>1 else 1
        centres=[]
        for group in clusters:
            members={sid for _,sid in group}
            if len(members)>=min_series:
                centres.append(float(np.median([np.median([x for x,s in group if s==sid]) for sid in members])))
        self.audit.update(cluster_tolerance_px=tol,cluster_centres=centres,
                          minimum_series_per_grid_anchor=min_series)
        if len(centres)<4:return
        centres=np.asarray(centres);gaps=np.diff(centres);period=float(np.median(gaps))
        if period<=2*tol:return
        indices=np.rint((centres-centres[0])/period).astype(int)
        if len(set(indices))!=len(indices):return
        period,origin=np.polyfit(indices,centres,1)
        residual=np.abs(centres-(origin+period*indices))
        self.audit.update(period_px=float(period),origin_px=float(origin),
            anchor_indices=indices.tolist(),maximum_fit_residual_px=float(residual.max()))
        if period<=2*tol or residual.max()>max(2.,.12*period) or np.max(np.diff(indices))>3:
            self.reason='original_x_not_regular';return
        # No extrapolation beyond the originally observed common x extent.
        grid=origin+period*np.arange(indices[0],indices[-1]+1)
        lo,hi=reference.sample_xy[[0,-1],0]
        endpoint_tol=min(.25*period,.75*diameter)
        keep=(grid>=lo-endpoint_tol)&(grid<=hi+endpoint_tol)
        self.slots=grid[keep]
        # A fit endpoint is not necessarily a measurement endpoint. If the
        # nearest observed common slot is just outside the path, downweight it.
        self.weights=np.where((self.slots>=lo-1e-7)&(self.slots<=hi+1e-7),1.,.5)
        self.tolerance=min(.18*period,max(2.,.45*diameter))
        self.period=float(period)
        self.audit.update(path_x_range=[float(lo),float(hi)],endpoint_snap_tolerance_px=float(endpoint_tol),
                          slot_match_tolerance_px=float(self.tolerance))
        if len(self.slots)<2:
            self.reason='insufficient_slots_in_path';return
        self.enabled=True;self.reason='validated_original_uniform_x_lattice'
        self.slots.flags.writeable=False;self.weights.flags.writeable=False

    def describe(self):
        return dict(self.audit,enabled=self.enabled,reason=self.reason,slots=self.slots.tolist(),
                    weights=self.weights.tolist(),marker_existence_evidence=False)

    def score(self,points):
        if not self.enabled:
            return dict(enabled=False,reason=self.reason,loss=0.,quality=None)
        counts=np.zeros(len(self.slots),int);offset=0.;assignments=[]
        for p in points:
            x=_x(p);j=int(np.argmin(abs(self.slots-x)));distance=abs(self.slots[j]-x)
            matched=distance<=self.tolerance
            if matched:counts[j]+=1
            # Even numerous off-lattice points cannot create coverage credit.
            offset+=min(1.,(distance/(.5*self.period))**2)
            assignments.append(int(j) if matched else None)
        denom=float(self.weights.sum())
        missing=float(self.weights[counts==0].sum()/denom)
        duplicate=float(np.dot(self.weights,np.maximum(counts-1,0))/denom)
        offgrid=float(offset/denom)
        loss=missing+duplicate+offgrid
        return dict(enabled=True,loss=loss,quality=max(0.,1.-loss),missing=missing,
                    duplicate=duplicate,offgrid=offgrid,counts=counts.tolist(),
                    occupied_slots=int((counts>0).sum()),expected_slots=len(counts),
                    assignments=assignments,fixed_denominator=denom)

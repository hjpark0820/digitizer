"""Post-verification competition of observed marker footprints.

Different series may have overlapping glyphs. Centre proximity alone must not
delete them: cross-type duplicates must largely explain the same footprint.
"""
import math

import cv2
import numpy as np


IDENTITY_RANK_MARGIN = .03


def resolve_body_identities(records):
    """Resolve only verified fits to the SAME independently isolated body.

    Pairwise contradictions precede scalar rank. A near-tie keeps its plausible
    typed alternatives in the suppressed pool; rank order is not identity proof.
    Non-isolated overlaps and legacy records keep the original selection path.
    """
    groups={}
    for r in records:
        evidence=r.get('grid_identity',{})
        if evidence.get('version')!='pooled-body-raster-identity-v2' or not evidence.get('body_id'):
            continue
        groups.setdefault(evidence['body_id'],[]).append(r)
    blocked=set()
    for rows in groups.values():
        by_id={}
        for r in rows:
            sid=identity(r)
            if sid not in by_id or r['selection_rank']>by_id[sid]['selection_rank']:
                by_id[sid]=r
        if len(by_id)<2:continue
        defeated=set()
        for row in by_id.values():
            for pair in row['grid_identity'].get('pairs',[]):
                winner=pair.get('winner')
                if winner in by_id:
                    defeated.add(pair['b'] if pair['a']==winner else pair['a'])
        contenders=[r for sid,r in by_id.items() if sid not in defeated]
        # Contradictory comparisons form a cycle: no forced winner.
        cycle=not contenders
        if cycle:contenders=list(by_id.values())
        contenders.sort(key=lambda r:(-r['selection_rank'],identity(r)))
        gap=(contenders[0]['selection_rank']-contenders[1]['selection_rank']) if len(contenders)>1 else None
        boundary_priority=any(r['grid_identity'].get('boundary_priority') for r in rows)
        # Internal ink compatibility cannot break an unresolved shape contest.
        # Keep the typed alternatives suppressed until positive outlines agree.
        deferred=cycle or (gap is not None and (boundary_priority or gap<IDENTITY_RANK_MARGIN))
        winner=None if deferred else contenders[0]
        plausible={identity(r) for r in contenders}
        for r in rows:
            r['identity_selection']=dict(status='ambiguous' if deferred else 'resolved',
                contenders=sorted(plausible),rank_margin=gap,minimum_margin=IDENTITY_RANK_MARGIN,
                winner=identity(winner) if winner else None,comparison_cycle=cycle)
            r['identity_selection']['boundary_priority']=boundary_priority
            if deferred:
                r['selected']=False
                r['exclusion_reason']='identity_ambiguous' if identity(r) in plausible else 'grid_identity_conflict'
                blocked.add(id(r))
            elif identity(r)!=identity(winner):
                r['selected']=False;r['exclusion_reason']='verified_duplicate'
                r['duplicate_kind']='isolated_body_identity_competition'
                r['suppressed_by']={'template':winner['template'],'swatch_id':identity(winner),
                                    'x':winner['aligned_x'],'y':winner['aligned_y']}
                blocked.add(id(r))
    return blocked


def identity(record):
    """Same geometry does not imply the same legend series."""
    return record.get('swatch_id') or record['template']


def footprint_overlap(left,right):
    """Intersection divided by the smaller convex observed-glyph footprint."""
    a=np.asarray(left.get('footprint_polygon',[]),np.float32)
    b=np.asarray(right.get('footprint_polygon',[]),np.float32)
    if len(a)<3 or len(b)<3:
        return None
    area_a=abs(cv2.contourArea(a)); area_b=abs(cv2.contourArea(b))
    if min(area_a,area_b)<1e-6:
        return 0.
    intersection,_=cv2.intersectConvexConvex(a,b)
    return float(max(0.,intersection)/min(area_a,area_b))


def duplicate_reason(candidate,kept):
    distance=math.hypot(candidate['aligned_x']-kept['aligned_x'],
                        candidate['aligned_y']-kept['aligned_y'])
    da,db=candidate['effective_diameter'],kept['effective_diameter']
    if identity(candidate)==identity(kept):
        # Equivalent coarse-scale fits of ONE marker can have slightly
        # different diameters. A minimum radius lets the smaller fit escape.
        radius=max(2.,.88*(da+db)/2)
        return 'same_type_neighbour' if distance<=radius else None
    if distance<=max(2.,.25*min(da,db)):
        return 'same_centre_different_type'
    overlap=footprint_overlap(candidate,kept)
    if overlap is None:
        # Compatibility for explicit synthetic/legacy records without masks.
        return 'legacy_centre_duplicate' if distance<=.65*min(da,db) else None
    return 'shared_marker_footprint' if overlap>=.55 else None


def select_verified(records):
    """Select only supplied eligible records; never create a new candidate."""
    accepted=[]
    for record in records:
        record['selected']=False
        record.pop('suppressed_by',None)
        record.pop('duplicate_kind',None)
        previous_identity=record.pop('identity_selection',None)
        if (record.get('exclusion_reason') in ('verified_duplicate','identity_ambiguous') or
                (previous_identity and record.get('exclusion_reason')=='grid_identity_conflict')):
            record.pop('exclusion_reason')
    blocked=resolve_body_identities(records)
    for record in sorted(records,key=lambda c:c['selection_rank'],reverse=True):
        if id(record) in blocked:continue
        for kept in accepted:
            reason=duplicate_reason(record,kept)
            if reason:
                record['exclusion_reason']='verified_duplicate'
                record['duplicate_kind']=reason
                record['suppressed_by']={'template':kept['template'],
                                        'x':kept['aligned_x'],'y':kept['aligned_y']}
                if kept.get('swatch_id'):
                    record['suppressed_by']['swatch_id']=kept['swatch_id']
                break
        else:
            record['selected']=True
            accepted.append(record)
    return accepted

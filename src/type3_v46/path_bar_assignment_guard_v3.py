"""Preserve competing bar assignments even after local point deduplication.

A green source-only corner can replace a duplicate green bar-based point,
but that does not prove the shared neutral bar belongs to cyan. Independent
corner acceptance and bar ownership are distinct decisions.
"""
from copy import deepcopy
from .path_bar_kink_detector_v2 import canonicalize_physical_bars


def preserve_competing_bar_assignments(results):
    output=deepcopy(results);witnesses={};active={}
    for sid,d in output.items():
        chosen=[]
        for p in d['candidates']:
            if p.get('stem') is None:continue
            was_accepted=p['status']=='accepted' or p.get('pre_duplicate_status')=='accepted'
            if not was_accepted:continue
            if p['status']=='rejected' and not p['reason'].startswith('duplicate_'):continue
            q=deepcopy(p);q['status']='accepted';q['witness_candidate_id']=p['id'];chosen.append(q)
            if p['status']=='accepted':active[(sid,p['id'])]=p
        witnesses[sid]={'candidates':chosen}
    grouped,identity_audit=canonicalize_physical_bars(witnesses)
    groups={}
    # canonicalization retains rejected same-series duplicate witnesses; all
    # retain the shared identity, so they can still map an active bar point.
    for sid,d in grouped.items():
        for p in d['candidates']:groups.setdefault(p['stem']['id'],[]).append((sid,p))
    decisions=[]
    for identity,members in groups.items():
        series=sorted({sid for sid,p in members})
        if len(series)<2:continue
        record=dict(ambiguity_group=identity,series_ids=series,
                    witnesses=[dict(series_id=sid,candidate_id=p['witness_candidate_id']) for sid,p in members],
                    downgraded=[])
        for sid,p in members:
            key=(sid,p['witness_candidate_id']);live=active.get(key)
            if live is not None and live['status']=='accepted':
                live['before_assignment_guard_status']='accepted'
                live['before_assignment_guard_reason']=live['reason']
                live.update(status='tentative',reason='shared_errorbar_assignment_survives_point_dedup')
                record['downgraded'].append(dict(series_id=sid,candidate_id=live['id']))
        decisions.append(record)
    for d in output.values():
        d['points']=[p for p in d['candidates'] if p['status']=='accepted']
        d['tentative_points']=[p for p in d['candidates'] if p['status']=='tentative']
    return output,dict(identity_audit=identity_audit,decisions=decisions,
        interpretation='Shared-cap assignments remain uncertain even when one color also has an independent corner')

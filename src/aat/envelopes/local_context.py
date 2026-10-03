"""Audit repeated acoustic events inside the actual input to each inference.

Segment length and model padding never enter the evidence window definition.
Targets stay untouched, including silence between repeated notes.
"""
from __future__ import annotations
import math
import numpy as np
from aat.windowing import centered_window_bounds

CONSTRAINT_VERSION='aat-c1-local-cooccurrence-v1'
CONTEXT_SECONDS=2.0
EDGE_GUARD_SECONDS=.05
RMS_THRESHOLD=.001


def audit_local_context(labels,centers,*,audio_frames,origin_seconds=0.,
                        context_seconds=CONTEXT_SECONDS,guard_seconds=EDGE_GUARD_SECONDS,
                        threshold=RMS_THRESHOLD):
    centers=np.asarray(centers,dtype=np.float64)
    if centers.ndim!=1 or not len(centers) or not np.isfinite(centers).all() or (np.diff(centers)<=0).any():
        raise ValueError('nonempty increasing centers required')
    if not all(math.isfinite(x) for x in (origin_seconds,context_seconds,guard_seconds,threshold)):
        raise ValueError('finite context settings required')
    if context_seconds<=0 or guard_seconds<0 or 2*guard_seconds>=context_seconds or threshold<=0 or audio_frames<1:
        raise ValueError('invalid context bounds/threshold')
    label_indices=np.round((centers-labels.center_times[0])/labels.hop_seconds).astype(np.int64)
    if (label_indices<0).any() or (label_indices>=len(labels.center_times)).any():
        raise ValueError('centers outside label grid')
    if not np.allclose(labels.center_times[label_indices],centers,atol=1e-8,rtol=0) or not labels.valid[label_indices].all():
        raise ValueError('centers must be valid points of the label grid')
    width=round(context_seconds*labels.sample_rate)
    samples=np.floor((centers-origin_seconds)*labels.sample_rate+.5).astype(np.int64)
    bounds=np.array([centered_window_bounds(int(i),width) for i in samples],dtype=np.int64)
    full_context=(bounds[:,0]>=0)&(bounds[:,1]<=audio_frames)
    absolute=bounds/labels.sample_rate+origin_seconds
    guarded=absolute+np.array([guard_seconds,-guard_seconds])
    n,s=len(centers),len(labels.source_ids)
    counts=np.zeros((n,s),dtype=np.int64)
    active=(labels.rms[label_indices]>=threshold)
    bridges=np.zeros((n,s),dtype=bool)
    active_ok=np.zeros((n,s),dtype=bool)
    bridge_ok=np.zeros((n,s),dtype=bool)
    sources=[]
    # Conservative expansion accounts for the measurement window and grid,
    # rather than interpreting first/last positive centers as exact edges.
    expansion=(labels.hop_seconds+labels.energy_window_seconds)/2
    violations=[]
    violation_count=0

    def fail(i,source_id,reason):
        nonlocal violation_count
        violation_count+=1
        if len(violations)<20:
            violations.append({'center_seconds':float(centers[i]),'source_id':source_id,'reason':reason})

    for source,source_id in enumerate(labels.source_ids):
        activity=(labels.rms[:,source]>=threshold)&labels.valid
        edges=np.diff(np.r_[False,activity,False].astype(np.int8))
        starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        left=labels.center_times[starts]-expansion
        right=labels.center_times[ends]+expansion
        visible=(left[None,:]>=guarded[:,0,None]-1e-9)&(right[None,:]<=guarded[:,1,None]+1e-9)
        counts[:,source]=visible.sum(1)
        for i,label_index in enumerate(label_indices):
            if not full_context[i]:fail(i,source_id,'input_requires_padding')
            if counts[i,source]<2:fail(i,source_id,'fewer_than_two_complete_same_source_events')
            if active[i,source]:
                current=int(np.flatnonzero((starts<=label_index)&(ends>=label_index))[0])
                neighbours=[j for j in (current-1,current+1) if 0<=j<len(starts)]
                active_ok[i,source]=full_context[i] and visible[i,current] and any(visible[i,j] for j in neighbours)
                if not active_ok[i,source]:fail(i,source_id,'active_center_has_no_complete_adjacent_note')
            for j in range(len(starts)-1):
                if ends[j]<label_index<starts[j+1]:
                    bridges[i,source]=True
                    bridge_ok[i,source]=full_context[i] and visible[i,j] and visible[i,j+1]
                    if not bridge_ok[i,source]:fail(i,source_id,'silence_bridge_endpoints_not_co_visible')
                    break
        sources.append({'source_id':source_id,'events':[
            {'event_index':j,'start_seconds':float(left[j]),'end_seconds':float(right[j]),
             'first_positive_center_seconds':float(labels.center_times[a]),
             'last_positive_center_seconds':float(labels.center_times[b])}
            for j,(a,b) in enumerate(zip(starts,ends))],
            'event_count':len(starts),'minimum_complete_events_per_window':int(counts[:,source].min()),
            'active_centers':int(active[:,source].sum()),'active_centers_with_adjacent_evidence':int(active_ok[:,source].sum()),
            'silence_bridge_centers':int(bridges[:,source].sum()),
            'silence_bridge_centers_with_both_endpoints':int(bridge_ok[:,source].sum()),
            'bridge_targets_exactly_zero':int(((labels.rms[label_indices,source]==0)&bridges[:,source]).sum())})
    report={'constraint_version':CONSTRAINT_VERSION,'evidence':'actual unpadded application waveform input',
            'context_seconds':context_seconds,'edge_guard_seconds':guard_seconds,'rms_threshold':threshold,
            'event_boundary_expansion_seconds':expansion,'model_padding_is_evidence':False,
            'center_count':n,'first_center_seconds':float(centers[0]),'last_center_seconds':float(centers[-1]),
            'full_context_centers':int(full_context.sum()),'sources':sources,
            'violation_count':violation_count,'violations_first_20':violations,'passed':violation_count==0}
    arrays={'center_times':centers,'input_bounds_seconds':absolute,'guarded_bounds_seconds':guarded,
            'complete_event_counts':counts,'center_active':active,'silence_bridge':bridges,
            'active_neighbor_covered':active_ok,'bridge_endpoints_covered':bridge_ok,'full_context':full_context}
    return report,arrays

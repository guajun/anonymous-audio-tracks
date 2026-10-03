"""Threshold diagnostics on the absolute envelope grid; never time-aligned."""
from __future__ import annotations

import numpy as np
from aat.tracking.matching import maximum_assignment


def stitch_diagnostics(predicted,target,times,assignment,*,maximum_endpoint_distance=.91,threshold=.002):
    """Reference-adjacent event evidence, on ONE fixed whole-segment PIT map.

    Event-best fragment IDs are diagnostics only; they never remap the loss.
    A same-fragment result alone is not source accuracy (see shape/extra events).
    """
    p,y,times=map(np.asarray,(predicted,target,times))
    records=[]
    for source,selected in enumerate(assignment.assignments[0]):
        edges=np.diff(np.r_[False,y[:,source]>=threshold,False].astype(np.int8))
        starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
        for previous,current in zip(range(len(starts)-1),range(1,len(starts))):
            a,b=starts[previous],ends[previous];c,d=starts[current],ends[current]
            distance=float(times[c]-times[b])
            event_mass=p[a:b+1].sum(0);next_mass=p[c:d+1].sum(0)
            first_fragment=int(event_mass.argmax());second_fragment=int(next_mass.argmax())
            represented=bool((p[a:b+1,selected]>=threshold).any() and (p[c:d+1,selected]>=threshold).any())
            both_detected=bool(p[a:b+1].max()>=threshold and p[c:d+1].max()>=threshold)
            records.append({'source_index':source,'previous_last_positive_seconds':float(times[b]),
                            'next_first_positive_seconds':float(times[c]),'endpoint_distance_seconds':distance,
                            'local_evidence_eligible':distance<maximum_endpoint_distance,
                            'both_events_detected_any_fragment':both_detected,
                            'both_events_on_fixed_pit_fragment':represented,
                            'diagnostic_event_best_fragments':[first_fragment,second_fragment],
                            'event_best_fragment_break':both_detected and first_fragment!=second_fragment})
    local=[r for r in records if r['local_evidence_eligible']]
    return {'reference_short_gap_pairs':len(local),'short_gap_both_detected':sum(r['both_events_detected_any_fragment'] for r in local),
            'short_gap_pit_stitched':sum(r['both_events_on_fixed_pit_fragment'] for r in local),
            'short_gap_pit_stitch_fraction':sum(r['both_events_on_fixed_pit_fragment'] for r in local)/len(local) if local else None,
            'diagnostic_short_gap_fragment_breaks':sum(r['event_best_fragment_break'] for r in local),
            'reference_long_gap_pairs':len(records)-len(local),'stitch_pairs':records}


def event_diagnostics(predicted, target, times, *, threshold=0.002):
    """One-to-one overlapping-run matches; missed events have no boundary score.

    The threshold is in linear RMS, not probability. First/last active centers
    define edges, so the sampling hop limits boundary resolution. No dilation,
    time shifts, minimum duration filter, or event-wise realignment is applied.
    """
    p, y, times = map(np.asarray, (predicted, target, times))
    if p.ndim != 1 or p.shape != y.shape or p.shape != times.shape or len(times)<2:
        raise ValueError("expected equal nonempty one-dimensional envelope/time arrays")
    if not all(np.isfinite(v).all() for v in (p,y,times)) or threshold<=0 or not np.isfinite(threshold):
        raise ValueError("finite envelopes and positive threshold required")
    if (p<0).any() or (y<0).any() or (np.diff(times)<=0).any():
        raise ValueError("nonnegative envelopes and increasing times required")
    active_p, active_y = p>=threshold, y>=threshold

    def runs(active):
        edges=np.diff(np.r_[False,active,False].astype(np.int8))
        return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)-1))

    reference, predicted_runs = runs(active_y), runs(active_p)
    scores=np.zeros((len(reference),len(predicted_runs)))
    for i,(a,b) in enumerate(reference):
        for j,(c,d) in enumerate(predicted_runs):
            overlap=max(0,min(b,d)-max(a,c)+1)
            scores[i,j]=overlap/(max(b,d)-min(a,c)+1)
    pairs=maximum_assignment(scores,threshold=1e-12,objective="cardinality_then_score")
    onset=[float(times[predicted_runs[j][0]]-times[reference[i][0]]) for i,j in pairs]
    offset=[float(times[predicted_runs[j][1]]-times[reference[i][1]]) for i,j in pairs]
    tp=int((active_p&active_y).sum())
    fp=int((active_p&~active_y).sum())
    fn=int((~active_p&active_y).sum())
    return {"rms_threshold":threshold,"matched_events":len(pairs),
            "missed_events":len(reference)-len(pairs),"extra_events":len(predicted_runs)-len(pairs),
            "onset_mae_seconds":float(np.abs(onset).mean()) if onset else None,
            "offset_mae_seconds":float(np.abs(offset).mean()) if offset else None,
            "onset_signed_seconds":float(np.mean(onset)) if onset else None,
            "offset_signed_seconds":float(np.mean(offset)) if offset else None,
            "frame_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}

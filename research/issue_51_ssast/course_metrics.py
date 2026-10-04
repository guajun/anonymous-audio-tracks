"""Evaluation distinguishes raw/transport/gates and mixture counterexamples."""
import numpy as np
import torch
from course_loss import pit


def curve(a,b,mask):
    if not mask.any():
        return dict(centers=0, per_source_iou=None, per_source_mae=None)
    x,y=a[mask],b[mask]
    union=np.maximum(x,y).sum(0)
    iou=np.divide(np.minimum(x,y).sum(0),union,out=np.zeros_like(union),where=union>0)
    return dict(centers=int(mask.sum()),per_source_iou=iou.tolist(),
        per_source_mae=np.abs(x-y).mean(0).tolist())


def measure(a,b,times,scale, *, order_override=None):
    aa=torch.tensor(a/scale)
    bb=torch.tensor(b/scale)
    order,_,gap=pit(aa,bb)
    order=order.numpy()
    if order_override is not None:
        order=np.asarray(order_override,dtype=np.int64)
    x=a[:,order]
    active=b>.001
    overlap=active.sum(1)>=2
    mask=np.ones(a.shape[1],dtype=bool); mask[order]=False
    norm=np.linalg.norm(a,axis=0)
    cosine=a.T@a/(norm[:,None]*norm[None,:]).clip(1e-12)
    areas=a.sum(0)
    copied=[(int(i),int(j)) for i in range(a.shape[1]) for j in range(i+1,a.shape[1])
        if cosine[i,j]>.99 and min(areas[i],areas[j])>.05*b.sum()]
    copied_windows=active_windows=0
    for start in range(0,len(a),8):
        xw=a[start:start+8]; bw=b[start:start+8]
        reference_area=float(bw.sum())
        if reference_area<=.001:continue
        active_windows+=1
        nw=np.linalg.norm(xw,axis=0)
        cw=xw.T@xw/(nw[:,None]*nw[None,:]).clip(1e-12)
        aw=xw.sum(0)
        copied_windows+=any(cw[i,j]>.99 and min(aw[i],aw[j])>.05*reference_area
            for i in range(a.shape[1]) for j in range(i+1,a.shape[1]))
    missed=(x<=.001)&active
    per_source_miss=[float(missed[:,s].sum()/max(1,active[:,s].sum())) for s in range(b.shape[1])]
    per_source_false=[float(((x[:,s]>.001)&~active[:,s]).sum()/max(1,(~active[:,s]).sum())) for s in range(b.shape[1])]
    onset_offset=[]
    for s in range(b.shape[1]):
        edges=[]
        for xx in (x[:,s]>.001, active[:,s]):
            d=np.diff(np.r_[False,xx,False].astype(int))
            edges.append((np.flatnonzero(d==1),np.flatnonzero(d==-1)-1))
        edge={}
        for kind,index in (('onset',0),('offset',1)):
            pred,ref=edges[0][index],edges[1][index]
            edge[kind]=dict(pred_count=len(pred),reference_count=len(ref),
                median_abs_seconds=float(np.median(np.abs(times[pred]-times[ref]))) if len(pred)==len(ref) and len(ref) else None)
        onset_offset.append(edge)
    all_mask=np.ones(len(b),dtype=bool)
    return dict(order=order.tolist(),assignment='whole-sequence PIT' if order_override is None else 'supplied teacher source rows',all=curve(x,b,all_mask),
        nonoverlap=curve(x,b,~overlap),overlap=curve(x,b,overlap),
        normalized_mae=float(np.abs(x-b).mean()/scale),
        per_source_missed_fraction=per_source_miss,
        per_source_silence_false_positive_fraction=per_source_false,
        second_source_missed_fraction=per_source_miss[1] if b.shape[1]>1 else None,
        unused_false_positive_fraction=float((a[:,mask]>.001).mean()),
        extra_candidate_area_fraction=float(a[:,mask].sum()/max(1e-12,b.sum())),
        source_count_mae=float(np.abs((a>.001).sum(1)-active.sum(1)).mean()),
        global_copy_pairs=copied,candidate_copy_window_fraction=copied_windows/max(1,active_windows),
        copy_window_seconds=.16,pit_gap=float(gap),pit_tied=bool(float(gap)<=1e-4),
        reference_shape_difference=float(np.abs(b[:,0]-b[:,1]).mean()/scale) if b.shape[1]>1 else None,
        onset_offset=onset_offset,boundary_note=(
            'C0 includes padded boundary inputs; see historical C0 boundary audit'
            if times[0]<1 else 'course centers1..11 have complete2s input'))


def endpoint_evidence(raw, transported, c,b,times,scale,horizon=.9):
    # Truth is used ONLY for evaluation, never to select training matches.
    track_order=measure(transported,b,times,scale)['order']
    source_metrics=[]
    for s in range(b.shape[1]):
        active=b[:,s]>.001
        edges=np.diff(np.r_[False,active,False].astype(int))
        starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)-1
        covered=correct=missing=0
        masses=[]
        for j,(end,start) in enumerate(zip(ends[:-1],starts[1:])):
            # centers <=.9 apart, both genuine2s windows see both endpoints.
            if times[start]-times[end]>horizon:
                continue
            covered+=1
            previous_order=measure(raw[starts[j]:end+1],b[starts[j]:end+1],times[starts[j]:end+1],scale)['order']
            next_order=measure(raw[start:ends[j+1]+1],b[start:ends[j+1]+1],times[start:ends[j+1]+1],scale)['order']
            previous_slot=previous_order[s]; slot=next_order[s]; row=track_order[s]
            if raw[end,previous_slot]<=.001 or raw[start,slot]<=.001:
                missing+=1; continue
            matched_row=int(c[start,:,slot].argmax())
            previous_row=int(c[end,:,previous_slot].argmax())
            masses.append(float(c[start,row,slot]))
            correct+=matched_row==row and previous_row==row
        source_metrics.append(dict(source=s,co_visible_endpoint_pairs=covered,
            correct_pairs=int(correct),candidate_missing_pairs=missing,
            correct_fraction=correct/max(1,covered),assigned_mass=masses))
    return dict(sources=source_metrics,definition='evaluation-only eventwise raw-candidate truth alignment; both endpoint columns must map to the same whole-sequence truth-aligned transported row; missing candidate counts as failure',
        limitation='eventwise envelope matching is an evaluation oracle, never a training target; ambiguous identical shapes cannot certify identity')

"""Post-selection diagnostics of joint-teacher C3 endpoint failures."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('runs/joint-teacher/fit-v2/joint-id/C3');records=[];total=0
    for path in sorted(root.glob('C3-*.npz')):
        z=np.load(path);b=z['target'];v=z['v'];a=z['raw_a'];order=z['teacher_order'];pred=z['predicted_order'];t=z['times']
        inverse=np.empty_like(pred);inverse[np.arange(len(pred))[:,None],pred]=np.arange(8)[None,:]
        for source in range(b.shape[1]):
            active=b[:,source]>0;edges=np.diff(np.r_[False,active,False].astype(int))
            starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
            for left,right in zip(ends[:-1],starts[1:]):
                if right-left>45:continue
                total+=1;c0=order[left,source];c1=order[right,source]
                row0=inverse[left,c0];row1=inverse[right,c1]
                if row0==row1:continue
                changes=[]
                # For this same-slot teacher case, locate which candidate-to-row
                # relation changes in the gap; this is evaluation, never tuning.
                if c0==c1:
                    for j in range(left,right):
                        if inverse[j,c0]==inverse[j+1,c0]:continue
                        unit0=v[j,c0]/max(a[j,c0],1e-30)
                        unit1=v[j+1,c0]/max(a[j+1,c0],1e-30)
                        changes.append(dict(left_frame=int(j),right_frame=int(j+1),time=float(t[j+1]),
                            amplitude_left=float(a[j,c0]),amplitude_right=float(a[j+1,c0]),
                            gt_left=float(b[j,source]),gt_right=float(b[j+1,source]),
                            candidate_cosine=float(np.dot(unit0,unit1)),row_before=int(inverse[j,c0]),row_after=int(inverse[j+1,c0])))
                records.append(dict(sample_id=path.stem[3:],source=source,left_frame=int(left),right_frame=int(right),
                    left_time=float(t[left]),right_time=float(t[right]),gap_seconds=float(t[right]-t[left]),
                    teacher_slots=[int(c0),int(c1)],predicted_rows=[int(row0),int(row1)],
                    endpoint_gt=[float(b[left,source]),float(b[right,source])],
                    endpoint_amplitude=[float(a[left,c0]),float(a[right,c1])],changes=changes))
    if total!=62:raise ValueError('C3 expected62 GT endpoint pairs')
    result=dict(total_endpoint_pairs=total,failed_pairs=len(records),correct_pairs=total-len(records),records=records,
        definition='post-selected C3 test, GT exactly nonzero event endpoints <=.9s; diagnostic only, no checkpoint/threshold changes',
        limitation='describes observed low-amplitude association errors; does not isolate a causal fix or validate identity during silence')
    out=Path('runs/joint-teacher/evidence/issue-51-joint-gap-failures.json')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(total=total,failed=len(records),correct=total-len(records))))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Rebuild endpoint result and numerical boundary figures from published evidence."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path('docs/reports/evidence/issue-46-endpoint.json'))
    parser.add_argument('--out',type=Path,default=Path('docs/reports/figures'));args=parser.parse_args()
    data=json.loads(args.evidence.read_text(encoding='utf-8'));args.out.mkdir(parents=True,exist_ok=True)
    colors=('#31688e','#d59b13')
    rows=[r for r in data['groups']['v3']['runs'] if r['summary']['runs'][0]['family']=='huber_iou']
    fig,axes=plt.subplots(2,3,figsize=(14,6.5),sharex=True,sharey=True,layout='constrained')
    for column,(row,title) in enumerate(zip(rows,('Channel index','Local soft, cycle=0','Local soft, cycle=0.001'))):
        curve=row['validation_example'];t=curve['center_times']
        for source in range(2):
            ax=axes[source,column]
            ax.plot(t,np.array(curve['target'])[:,source],color='#222222',lw=1.5,label='Reference')
            ax.plot(t,np.array(curve['predicted'])[:,source],color=colors[source],lw=1.2,label='Whole-PIT selected fragment')
            ax.plot(t,np.array(curve['unused']).max(1),color='#bf3868',ls='--',lw=.8,label='Max unused fragment')
            ax.axhline(.002,color='gray',ls=':',lw=.7);ax.grid(alpha=.2)
            if source==0:ax.set_title(title)
            if column==0:ax.set_ylabel(f'Source {source+1}: linear RMS')
            if source==1:ax.set_xlabel('Original-track seconds')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Local endpoints v3: Huber + IoU, seed 46, 100 steps\nActual c1-local-val-00; event coverage improved, candidate mixtures remain')
    fig.savefig(args.out/'issue-46-endpoint-curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),sharex=True,sharey=True,layout='constrained')
    for ax,row,title in zip(axes,rows,('Channel index','Local soft, cycle=0','Local soft, cycle=0.001')):
        curve=row['validation_example'];t=curve['center_times']
        ax.plot(t,np.array(curve['pregate_candidates']).sum(1),color='#d59b13',lw=1,label='Head before gate')
        ax.plot(t,np.array(curve['raw_candidates']).sum(1),color='#31688e',lw=1,label='Head after gate')
        ax.plot(t,np.array(curve['tracked_candidates']).sum(1),color='#bf3868',lw=1,label='Transported after gate')
        ax.plot(t,np.array(curve['target']).sum(1),color='#222222',lw=1,label='Reference sum')
        ax.set_title(title);ax.set_xlabel('Original-track seconds');ax.grid(alpha=.2)
    axes[0].set_ylabel('Sum linear RMS');axes[0].legend(fontsize=7)
    fig.suptitle('Measured activity through the two gates\nAggregate mass is a diagnostic, not source identity accuracy')
    fig.savefig(args.out/'issue-46-endpoint-gates.png',dpi=150);plt.close(fig)
    probe=data['boundary_probe'];long=probe['long_gap_example']
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
    for ax,example,title in zip(axes,(probe,long),('One source resumes after short zero gaps while another remains audible','Beyond local endpoint evidence: new fragment, even with identical E')):
        time=np.array(example['center_times']);tracks=np.array(example['tracked'])
        for fragment in range(tracks.shape[1]):
            if tracks[:,fragment].max()>0:
                ax.plot(time,tracks[:,fragment],marker='o',ms=3,lw=1,label=f'Fragment {fragment}')
        ax.set_title(title);ax.set_ylabel('Activity');ax.set_xlabel('Original-grid seconds');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=4)
    fig.suptitle('Controlled numerical boundary check (not an audio performance claim)\nZero candidate E is invalid; finite nonzero endpoints carry the correspondence')
    fig.savefig(args.out/'issue-46-endpoint-boundaries.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()

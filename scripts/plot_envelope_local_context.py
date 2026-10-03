#!/usr/bin/env python3
"""Rebuild C1-local figures from published evidence; no audio/model dependency."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path('docs/reports/evidence/issue-46-local-context.json'))
    parser.add_argument('--out',type=Path,default=Path('docs/reports/figures'))
    args=parser.parse_args()
    data=json.loads(args.evidence.read_text(encoding='utf-8'));args.out.mkdir(parents=True,exist_ok=True)
    colors=('#31688e','#d59b13')
    ref=data['reference_example'];times=np.array(ref['center_times']);rms=np.array(ref['rms'])
    window=data['window_example'];centers=np.array(window['center_times'])
    first=data['runs'][0]['validation_example'];target=np.array(first['target'])
    silent=np.flatnonzero((target==0).all(1)&(centers>=2)&(centers<=4))
    i=int(silent[0]);center=centers[i];left,right=window['input_bounds_seconds'][i]
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
    guard=.05
    for source in range(2):
        axes[0].plot(times,rms[:,source],color=colors[source],label=f'Source {source+1} reference RMS')
    axes[0].axvspan(left,left+guard,color='gray',alpha=.25)
    axes[0].axvspan(right-guard,right,color='gray',alpha=.25)
    axes[0].axvline(center,color='#bf3868',ls='--',label=f'Actual inference center: {center:.2f}s, zero targets')
    axes[0].set_xlim(left-.12,right+.12);axes[0].set_ylabel('Linear RMS')
    axes[0].set_title(f'Actual 2s input [{left:.2f}, {right:.2f}]s; 50ms guards; no invented sound')
    axes[0].legend(fontsize=8,ncol=2);axes[0].grid(alpha=.2)
    for source in range(2):
        axes[1].step(centers,np.array(window['complete_event_counts'])[:,source],where='mid',
                     color=colors[source],label=f'Source {source+1} complete events')
    axes[1].axhline(2,color='gray',ls=':',label='Required minimum')
    axes[1].set_ylim(0,5);axes[1].set_yticks(range(5));axes[1].set_xlabel('Original-track seconds')
    axes[1].set_ylabel('Complete same-source events');axes[1].legend(fontsize=8,ncol=3);axes[1].grid(alpha=.2)
    fig.suptitle('C1-local: event evidence inside each actual inference input\nc1-local-val-00; all train/val/test windows independently pass the same contract')
    fig.savefig(args.out/'issue-46-local-context-coverage.png',dpi=150);plt.close(fig)
    for family,label in (('l1','L1'),('huber_iou','Huber + IoU')):
        rows=[r for r in data['runs'] if r['summary']['runs'][0]['family']==family]
        fig,axes=plt.subplots(2,3,figsize=(14,6.5),sharex=True,sharey=True,layout='constrained')
        for column,(row,title) in enumerate(zip(rows,('Channel index','Soft, cycle=0','Soft, cycle=0.001'))):
            curve=row['validation_example'];t=np.array(curve['center_times'])
            for source in range(2):
                ax=axes[source,column]
                ax.plot(t,np.array(curve['target'])[:,source],color='#222222',lw=1.5,label='Reference')
                ax.plot(t,np.array(curve['predicted'])[:,source],color=colors[source],lw=1.2,label='PIT-selected transported A')
                ax.plot(t,np.array(curve['unused']).max(1),color='#bf3868',ls='--',lw=.8,label='Max unused track')
                ax.axhline(.002,color='gray',ls=':',lw=.7);ax.grid(alpha=.2)
                if source==0:ax.set_title(title)
                if column==0:ax.set_ylabel(f'Source {source+1}: linear RMS')
                if source==1:ax.set_xlabel('Original-track seconds')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'C1-local: {label}, seed 46, 100 steps\nc1-local-val-00; fixed pad/pluck; input co-occurrence does not guarantee correct association')
        fig.savefig(args.out/f'issue-46-local-context-{family}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,6.2),sharex=True,sharey=True,layout='constrained')
    for row_index,family in enumerate(('l1','huber_iou')):
        rows=[r for r in data['runs'] if r['summary']['runs'][0]['family']==family]
        for column,(row,title) in enumerate(zip(rows,('Channel index','Soft, cycle=0','Soft, cycle=0.001'))):
            curve=row['validation_example'];ax=axes[row_index,column]
            ax.plot(curve['center_times'],np.array(curve['raw_candidates']).sum(1),color='#31688e',lw=1,label='Sum raw A')
            ax.plot(curve['center_times'],np.array(curve['tracked_candidates']).sum(1),color='#bf3868',lw=1,label='Sum transported A')
            ax.plot(curve['center_times'],np.array(curve['target']).sum(1),color='#222222',lw=1,label='Sum reference')
            if row_index==0:ax.set_title(title)
            if column==0:ax.set_ylabel(f'{family}: sum linear RMS')
            if row_index==1:ax.set_xlabel('Original-track seconds')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('C1-local: raw head activity versus transported activity\nAggregate diagnostics are not source accuracy; original time axis, no alignment')
    fig.savefig(args.out/'issue-46-local-context-transport.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()

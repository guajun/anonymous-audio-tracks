#!/usr/bin/env python3
"""Rebuild issue #46 C0 figures from the published numeric evidence.

Requires numpy and matplotlib; does not load models or audio.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path('docs/reports/evidence/issue-46-c0.json'))
    parser.add_argument('--out',type=Path,default=Path('docs/reports/figures'))
    args=parser.parse_args()
    data=json.loads(args.evidence.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True,exist_ok=True)
    families=['l1','huber','area_iou','huber_iou','multiscale']
    labels=['L1','Huber / delta','Area IoU','Huber + IoU','Multiscale Huber']
    colors=['#31688e','#35b779','#e76f51','#9b59b6','#d59b13']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    group=data['groups']['v2_calibrated_1000']
    for i,(family,label,color) in enumerate(zip(families,labels,colors)):
        runs=[next(r for r in d['runs'] if r['family']==family) for d in group]
        val=[r['val']['mean_area_iou'] for r in runs]
        extra=[np.mean([s['extra_active_fraction'] for s in r['val']['songs']])*100 for r in runs]
        for ax,values in zip(axes,(val,extra)):
            ax.scatter(np.array([i]*3)+[-.08,0,.08],values,color=color,s=32)
            ax.plot([i-.18,i+.18],[np.mean(values)]*2,color=color,lw=2)
    for ax in axes:
        ax.set_xticks(range(5),labels,rotation=20,ha='right')
        ax.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Validation area IoU')
    axes[0].set_ylim(.75,.9)
    axes[1].set_ylabel('Time with any extra active slot (%)')
    axes[1].set_ylim(0,105)
    fig.suptitle('C0: 8 candidates, 1,000 steps, seeds 46/47/48\nEach point averages two held-out sequences; fixed timbre')
    fig.savefig(args.out/'issue-46-c0-screen.png',dpi=150)
    plt.close(fig)

    fig,axes=plt.subplots(5,1,figsize=(12,11),sharex=True,layout='constrained')
    for ax,family,label,color in zip(axes,families,labels,colors):
        curve=data['validation_example']['curves'][family]
        ax.plot(curve['times'],curve['target'],color='#222222',lw=2,label='Reference stem RMS')
        ax.plot(curve['times'],curve['predicted'],color=color,lw=1.8,label='Matched candidate')
        ax.plot(curve['times'],curve['unused_max'],color='#bf3868',ls='--',lw=1.4,label='Max of 7 unused candidates')
        ax.axhline(.002,color='gray',lw=.8,ls=':')
        ax.set_ylabel('Linear RMS')
        ax.set_title(label,loc='left',fontsize=11)
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper right',bbox_to_anchor=(1,1.22),ncol=3,fontsize=8)
    axes[-1].set_xlabel('Original-track time (seconds); no time alignment')
    fig.suptitle('Actual C0 validation envelopes: c0-val-00, seed 46, 1,000 steps\nDotted line: 0.002 RMS event threshold; this is not a separation benchmark')
    fig.savefig(args.out/'issue-46-c0-curves.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()

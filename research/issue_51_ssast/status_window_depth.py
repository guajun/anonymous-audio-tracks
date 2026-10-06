"""Compact durable progress; safe while writers replace JSON contents."""
import json
from pathlib import Path
import argparse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',action='store_true')
    args=parser.parse_args()
    root=Path('runs/window-depth/fit-v1')
    for depth in (2,4,8):
        d=root/f'depth{depth}'
        try:
            files=list(d.glob('*/progress.json'))
            last=max(files,key=lambda p:p.stat().st_mtime)
            history=json.loads(last.read_text())
            complete=list(d.glob('*/result.json'))
            print(json.dumps(dict(depth=depth,current=last.parent.name,
                update=history[-1]['update'],val_score=history[-1]['val_score'],
                loop_seconds=history[-1]['elapsed_seconds'],
                completed=[p.parent.name for p in complete])),flush=True)
            if args.results:
                for path in sorted(complete):
                    r=json.loads(path.read_text());rr=[x for x in r['test'] if x['stage']==r['stage']]
                    iou=[x['transported']['all']['per_source_iou'] for x in rr]
                    print(json.dumps(dict(depth=depth,finished=path.parent.name,updates=r['updates'],
                        seconds=r['seconds'],stop=r['stop_reason'],test_source_iou=[sum(z)/len(z) for z in zip(*iou)],
                        unused_fp=sum(x['transported']['unused_false_positive_fraction'] for x in rr)/len(rr))),flush=True)
        except (ValueError,IndexError,FileNotFoundError):
            print(json.dumps(dict(depth=depth,status='initializing or writing progress')),flush=True)


if __name__=='__main__':main()

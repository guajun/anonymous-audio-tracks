#!/usr/bin/env python3
"""Issue #46 C0/C1: prepare, cache, and train shapes through optional association."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    prepare=sub.add_parser("prepare-c0")
    prepare.add_argument("--out",required=True)
    prepare.add_argument("--seed",type=int,default=4600)
    prepare_c1=sub.add_parser("prepare-c1")
    prepare_c1.add_argument("--out",required=True)
    prepare_c1.add_argument("--seed",type=int,default=4610)
    cache=sub.add_parser("cache")
    cache.add_argument("--data",required=True)
    cache.add_argument("--out",required=True)
    cache.add_argument("--device",default="cuda:0")
    cache.add_argument("--hop-seconds",type=float,default=0.04)
    cache.add_argument("--batch-windows",type=int,default=4)
    train=sub.add_parser("train")
    train.add_argument("--cache",required=True)
    train.add_argument("--out",required=True)
    train.add_argument("--device",default="cuda:0")
    train.add_argument("--steps",type=int,default=100)
    train.add_argument("--seed",type=int,default=46)
    train.add_argument("--lr",type=float,default=3e-4)
    train.add_argument("--segment-centers",type=int,default=40)
    train.add_argument("--huber-delta",type=float,default=0.05)
    train.add_argument("--iou-weight",type=float,default=0.1)
    train.add_argument("--empty-weight",type=float,default=1.0)
    train.add_argument("--slots",type=int,default=8,help="candidate capacity; 1 is a C0 diagnostic")
    train.add_argument("--null-weight",type=float,default=None,help="separate unused-track weight; defaults to empty-weight")
    train.add_argument("--association",choices=["index","soft"],default="index")
    train.add_argument("--cycle-weight",type=float,default=0)
    train.add_argument("--association-temperature",type=float,default=.1)
    train.add_argument("--association-iterations",type=int,default=12)
    train.add_argument("--max-gap-frames",type=int,default=400)
    train.add_argument("--shuffle-candidates",action="store_true")
    train.add_argument("--tensorboard-root")
    train.add_argument("--eval-every",type=int,default=0)
    train.add_argument("--allow-partial-recurrence",action="store_true",help="reproduce the early C1 wiring probe without the complete-event filter")
    train.add_argument("--raw-huber",action="store_true",help="do not divide Huber by delta; diagnostic legacy scale")
    train.add_argument("--scales-seconds",nargs="+",type=float,default=[0,0.02,0.05,0.10])
    train.add_argument("--families",nargs="+",default=["l1","huber","area_iou","huber_iou","multiscale"])
    args=parser.parse_args()
    if args.command=="prepare-c0":
        from aat.envelopes.curriculum import prepare_c0
        result=prepare_c0(args.out,seed=args.seed)
    elif args.command=="prepare-c1":
        from aat.envelopes.curriculum import prepare_c1
        result=prepare_c1(args.out,seed=args.seed)
    elif args.command=="cache":
        from aat.envelopes.experiment import build_cache
        result=build_cache(args.data,args.out,device=args.device,hop_seconds=args.hop_seconds,batch_windows=args.batch_windows)
    else:
        from aat.envelopes.experiment import train_combinations
        from aat.envelopes.association import AssociationConfig
        result=train_combinations(args.cache,args.out,device=args.device,steps=args.steps,seed=args.seed,lr=args.lr,
                                  segment_centers=args.segment_centers,families=tuple(args.families),
                                  delta=args.huber_delta,iou_weight=args.iou_weight,empty_weight=args.empty_weight,
                                  scales_seconds=tuple(args.scales_seconds),normalize_huber=not args.raw_huber,slots=args.slots,
                                  null_weight=args.null_weight,association=args.association,cycle_weight=args.cycle_weight,
                                  association_config=AssociationConfig(temperature=args.association_temperature,
                                      iterations=args.association_iterations,max_gap_frames=args.max_gap_frames),
                                  shuffle_candidates=args.shuffle_candidates,tensorboard_root=args.tensorboard_root,eval_every=args.eval_every,
                                  require_recurrence=False if args.allow_partial_recurrence else None)
    print(f"completed {args.command}: {len(result.get('entries',result.get('runs',[])))} records",flush=True)


if __name__=="__main__":
    main()

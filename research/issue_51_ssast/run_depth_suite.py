"""Persist three independent matched experiments on explicitly selected GPU0."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('runs/window-depth/fit-v1')
    root.mkdir(parents=True,exist_ok=True)
    processes=[];started=time.time()
    for depth in (2,4,8):
        out=root/f'depth{depth}'
        if out.exists():raise ValueError('refusing to overwrite experiment')
        log=(root/f'depth{depth}.log').open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH='src',MPLCONFIGDIR='runs/matplotlib')
        command=[sys.executable,'research/issue_51_ssast/train_window_depth.py',
            '--depth',str(depth),'--cache','runs/window-depth/cache',
            '--c0-cache','runs/p1/frame-c0-cache','--out',str(out),
            '--steps','1000','--c0-steps','2000','--seconds','7200']
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((depth,p,log,command))
    while True:
        status=dict(started_unix=started,elapsed_seconds=time.time()-started,
            gpu=0,concurrent_processes=3,
            processes=[dict(depth=d,pid=p.pid,returncode=p.poll(),command=c)
                for d,p,log,c in processes])
        (root/'suite-status.json').write_text(json.dumps(status,indent=2)+'\n')
        if all(p.poll() is not None for d,p,log,c in processes):break
        time.sleep(5)
    for d,p,log,c in processes:log.close()
    if any(p.returncode for d,p,log,c in processes):raise SystemExit(1)
    print('SUITE COMPLETE',flush=True)


if __name__=='__main__':main()

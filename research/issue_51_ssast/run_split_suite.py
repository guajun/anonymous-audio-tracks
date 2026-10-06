"""Durable sequential GPU0-only pilots and matched four-course arms."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    root=Path('runs/split-head/fit-v1');root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='src:research/issue_51_ssast')
    start=time.time();jobs=[]
    for pilot in (True,False):
        for arm in ('coupled','split'):
            name=arm+('-pilot' if pilot else '')
            cmd=[sys.executable,'research/issue_51_ssast/train_joint_teacher.py','--cache','runs/window-depth/cache',
                '--c0-cache','runs/p1/frame-c0-cache','--initial',str(root/(arm+'-initial.pth')),
                '--out',str(root/name),'--identity-weight','.2','--teacher-mode','joint','--head-mode',arm]
            if pilot:cmd+=['--pilot','--steps','20']
            with (root/(name+'.log')).open('w') as log:
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                job=dict(name=name,pid=process.pid,command=cmd,exit_code=None);jobs.append(job)
                while process.poll() is None:
                    (root/'status.json').write_text(json.dumps(dict(start_epoch=start,elapsed_seconds=time.time()-start,gpu=0,jobs=jobs,complete=False),indent=2)+'\n')
                    time.sleep(10)
                job['exit_code']=process.returncode
            if job['exit_code']:raise SystemExit('failed '+name)
    (root/'status.json').write_text(json.dumps(dict(start_epoch=start,elapsed_seconds=time.time()-start,gpu=0,jobs=jobs,complete=True),indent=2)+'\n')

if __name__=='__main__':main()

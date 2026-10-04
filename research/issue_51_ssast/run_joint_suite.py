"""Durable paired GPU0 experiment runner; no shared checkout/process mutation."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root=Path('runs/joint-teacher/fit-v2');root.mkdir(parents=True,exist_ok=True)
    jobs=[];handles=[];start=time.time()
    for name,mode in [('envelope-id','envelope'),('joint-id','joint')]:
        cmd=[sys.executable,'research/issue_51_ssast/train_joint_teacher.py','--cache','runs/window-depth/cache',
            '--c0-cache','runs/p1/frame-c0-cache','--initial','runs/window-depth/fit-v1/depth4/C1-raw/head.pth',
            '--out',str(root/name),'--identity-weight','.2','--teacher-mode',mode]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='src:research/issue_51_ssast')
        log=(root/(name+'.log')).open('w');handles.append(log)
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(dict(name=name,process=process,pid=process.pid,command=cmd,log=str(root/(name+'.log'))))
    while True:
        status=dict(start_epoch=start,elapsed_seconds=time.time()-start,gpu=0,
            jobs=[dict(name=j['name'],pid=j['pid'],command=j['command'],log=j['log'],exit_code=j['process'].poll()) for j in jobs])
        status['complete']=all(j['exit_code'] is not None for j in status['jobs'])
        (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        if status['complete']:break
        time.sleep(10)
    for handle in handles:handle.close()
    if any(j['exit_code'] for j in status['jobs']):raise SystemExit('experiment child failed; see its own log')


if __name__=='__main__':main()

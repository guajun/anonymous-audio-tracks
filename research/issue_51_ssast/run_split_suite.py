"""Durable GPU0-only pilots and concurrent matched four-course arms."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('runs/split-head/fit-v3'));parser.add_argument('--phase',choices=('all','pilot','formal'),default='all');args=parser.parse_args()
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='src:research/issue_51_ssast')
    start=time.time();jobs=[]
    status_path=root/('pilot-status.json' if args.phase=='pilot' else 'status.json')
    phases=(True,False) if args.phase=='all' else (args.phase=='pilot',)
    if args.phase=='formal':
        previous=json.loads((root/'pilot-status.json').read_text())
        if not previous['complete'] or any(j['exit_code'] for j in previous['jobs']):raise ValueError('pilot not complete')
        running=[]
        for arm in ('coupled','split'):
            cmd=[sys.executable,'research/issue_51_ssast/train_joint_teacher.py','--cache','runs/window-depth/cache',
                '--c0-cache','runs/p1/frame-c0-cache','--initial',str(root/(arm+'-initial.pth')),
                '--amplitude-transform','abs','--out',str(root/arm),'--identity-weight','.2','--teacher-mode','joint','--head-mode',arm]
            log=(root/(arm+'.log')).open('w')
            process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            job=dict(name=arm,pid=process.pid,command=cmd,start_epoch=time.time(),exit_code=None);jobs.append(job)
            running.append((job,process,log))
        while True:
            for job,process,log in running:
                code=process.poll()
                if code is not None and job['exit_code'] is None:
                    job.update(exit_code=code,seconds=time.time()-job['start_epoch']);log.close()
            complete=all(j['exit_code'] is not None for j in jobs)
            status_path.write_text(json.dumps(dict(start_epoch=start,elapsed_seconds=time.time()-start,
                runner_pid=os.getpid(),phase=args.phase,gpu=0,jobs=jobs,complete=complete,
                pilot_status=str(root/'pilot-status.json'),concurrent_arms=True),indent=2)+'\n')
            if complete:break
            time.sleep(10)
        if any(j['exit_code'] for j in jobs):raise SystemExit('formal arm failed')
        return
    for pilot in phases:
        for arm in ('coupled','split'):
            name=arm+('-pilot' if pilot else '')
            cmd=[sys.executable,'research/issue_51_ssast/train_joint_teacher.py','--cache','runs/window-depth/cache',
                '--c0-cache','runs/p1/frame-c0-cache','--initial',str(root/(arm+'-initial.pth')),
                '--amplitude-transform','abs','--out',str(root/name),'--identity-weight','.2','--teacher-mode','joint','--head-mode',arm]
            if pilot:cmd+=['--pilot','--steps','20']
            with (root/(name+'.log')).open('w') as log:
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                job=dict(name=name,pid=process.pid,command=cmd,start_epoch=time.time(),exit_code=None);jobs.append(job)
                while process.poll() is None:
                    status_path.write_text(json.dumps(dict(start_epoch=start,elapsed_seconds=time.time()-start,runner_pid=os.getpid(),phase=args.phase,gpu=0,jobs=jobs,complete=False),indent=2)+'\n')
                    time.sleep(10)
                job['exit_code']=process.returncode;job['seconds']=time.time()-job['start_epoch']
            if job['exit_code']:raise SystemExit('failed '+name)
    status_path.write_text(json.dumps(dict(start_epoch=start,elapsed_seconds=time.time()-start,runner_pid=os.getpid(),phase=args.phase,gpu=0,jobs=jobs,complete=True),indent=2)+'\n')

if __name__=='__main__':main()

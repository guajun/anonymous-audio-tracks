"""Durable dependent evaluation pipeline; does not change selected checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('runs/window-depth/fit-v1');status_path=root/'postprocess-status.json'
    while True:
        try:
            status=json.loads((root/'suite-status.json').read_text())
            codes=[p['returncode'] for p in status['processes']]
            if all(code is not None for code in codes):break
        except (ValueError,FileNotFoundError):pass
        time.sleep(10)
    if any(codes):raise SystemExit('training suite failed; refusing partial comparison')
    scripts=[('evaluate_window_depth.py',['--c0-cache','runs/p1/frame-c0-cache']),
        ('audit_trained_association.py',[]),('inspect_window_context.py',[]),
        ('summarize_window_depth.py',None)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH='src',MPLCONFIGDIR='runs/matplotlib')
    evidence=[]
    for name,extra in scripts:
        command=[sys.executable,'research/issue_51_ssast/'+name,'--fit',str(root)]
        if extra is not None:command+=['--cache','runs/window-depth/cache']+extra
        status_path.write_text(json.dumps(dict(current=name,completed=evidence),indent=2)+'\n')
        start=time.monotonic();p=subprocess.run(command,env=env)
        evidence.append(dict(script=name,returncode=p.returncode,seconds=time.monotonic()-start))
        if p.returncode:
            status_path.write_text(json.dumps(dict(failed=name,completed=evidence),indent=2)+'\n')
            raise SystemExit(p.returncode)
    status_path.write_text(json.dumps(dict(complete=True,completed=evidence),indent=2)+'\n')
    print('POSTPROCESS COMPLETE',flush=True)


if __name__=='__main__':main()

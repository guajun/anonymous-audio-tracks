"""Prove full-window recache did not change course labels or former inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from features import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--full',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();audit=[]
    for stage in ('C1','C2-low','C2-high','C3'):
        previous=json.loads((args.old/stage/'manifest.json').read_text())
        current=json.loads((args.full/stage/'manifest.json').read_text())
        if previous['data_index']!=current['data_index']:raise ValueError('data index changed')
        for a,b in zip(previous['entries'],current['entries']):
            x=np.load(args.old/stage/a['cache']);y=np.load(args.full/stage/b['cache'])
            for key in ('targets','center_times','mix_rms_oracle'):
                np.testing.assert_array_equal(x[key],y[key])
            np.testing.assert_array_equal(x['token_rms'],y['token_rms'][:,96:102])
            diff=float(np.abs(x['tokens'].astype(np.float32)-y['tokens'][:,96:102].astype(np.float32)).max())
            if diff>1e-3:raise ValueError('encoder recache changed previous features')
            if y['tokens'].shape[1:]!=(197,768):raise ValueError('incomplete window')
            audit.append(dict(stage=stage,sample_id=a['sample_id'],labels_and_times_identical=True,
                old_token_max_abs_difference=diff,token_shape=list(y['tokens'].shape),
                old_sha256=a['cache_sha256'],full_sha256=sha256(args.full/stage/b['cache'])))
    args.out.write_text(json.dumps(dict(samples=audit,data_changed=False),indent=2)+'\n')
    print('AUDIT PASSED',len(audit),flush=True)


if __name__=='__main__':main()

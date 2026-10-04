"""Package complete raw JSON and predictions without tracking model caches."""
import gzip
import json
from pathlib import Path
import shutil
import zipfile


def main():
    root=Path('runs/window-depth/fit-v1');out=Path('runs/window-depth/evidence')
    status=json.loads((root/'postprocess-status.json').read_text())
    if not status.get('complete'):raise ValueError('postprocessing incomplete')
    out.mkdir(parents=True,exist_ok=True)
    names={'compact.json':'issue-51-window-depth-summary.json',
        'tables.md':'issue-51-window-depth-tables.md',
        'suite-status.json':'issue-51-window-suite-status.json',
        'postprocess-status.json':'issue-51-window-postprocess-status.json',
        'trained-association-audit.json':'issue-51-window-trained-association-audit.json',
        'context-sensitivity.json':'issue-51-window-context-sensitivity.json',
        'depth-course-comparison.png':'issue-51-window-depth-comparison.png',
        'depth-validation-curves.png':'issue-51-window-validation-curves.png',
        'depth-c3-curves.png':'issue-51-window-c3-curves.png'}
    for src,dest in names.items():shutil.copyfile(root/src,out/dest)
    for depth in (2,4,8):
        shutil.copyfile(root/f'depth{depth}-full.json.gz',out/f'issue-51-window-depth{depth}-full.json.gz')
        architecture=json.loads((root/f'depth{depth}'/'architecture.json').read_text())
        architecture['center_interpolated_local_receptive_tokens']=2+2*depth
        (out/f'issue-51-window-depth{depth}-architecture.json').write_text(json.dumps(architecture,indent=2)+'\n')
    for stage in ('C1','C2-low','C2-high','C3'):
        manifest=Path('runs/window-depth/cache')/stage/'manifest.json'
        (out/f'issue-51-window-{stage}-cache.json.gz').write_bytes(gzip.compress(manifest.read_bytes(),mtime=0))
    predictions=Path('runs/window-depth/issue-51-window-predictions.zip')
    with zipfile.ZipFile(predictions,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.glob('depth*/*/*.npz'):archive.write(path,path.relative_to(root))
    print('EVIDENCE PACKAGED',flush=True)


if __name__=='__main__':main()

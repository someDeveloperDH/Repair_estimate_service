"""Build a portable task2-only GPU bundle without caches or smoke weights."""
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    destination=root/'exports'/'task2_rf_detr_small_gpu.zip'
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    files=[root/'requirements_task2.txt',root/'prepare_task2_gpu.py']
    files += [root/'pipeline'/name for name in ('__init__.py','run.py','metrics.py','rf_backend.py','rf_checkpoint.py')]
    files += list((root/'scripts'/'data2').glob('*.py'))
    files += [root/'docs'/'task2_rf_detr_guide.md']
    files += [root/'docs'/'task2_run_report.md']
    files += [p for p in (root/'datasets'/'data2').rglob('*') if p.is_file() and p.name!='data.yaml']
    config="path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: 'outsole'\n  1: 'heel lining'\n  2: 'not_sneakers'\n"
    manifest=[]
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in sorted(files):
            relative=path.relative_to(root).as_posix()
            content=path.read_bytes()
            archive.writestr(relative,content)
            manifest.append(dict(path=relative,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        archive.writestr('datasets/data2/data.yaml',config)
        archive.writestr('bundle_manifest.json',json.dumps(dict(task='data2',model='RFDETRSmall',
            version='1.11.2',files=manifest,note='data.yaml is rewritten to a portable relative path. No pretrained or smoke model is bundled.'),indent=2))
    with zipfile.ZipFile(destination) as archive:
        bad=archive.testzip()
        if bad: raise RuntimeError(f'Corrupt archive entry: {bad}')
    print(destination)
    print(f'{destination.stat().st_size/2**20:.1f} MiB; {len(files)} source/data files')


if __name__=='__main__': main()

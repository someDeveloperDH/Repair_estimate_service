"""Package both full prepared datasets and generate a ready-to-upload Colab notebook."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    exports=root/'exports';exports.mkdir(exist_ok=True)
    archive=exports/'men_dii_all_colab.zip'
    if archive.exists(): raise FileExistsError(archive)
    files=[root/'requirements_task2.txt',root/'run_colab_all.py',root/'docs'/'colab_all_guide.md']
    files += [root/'pipeline'/n for n in ('__init__.py','run.py','metrics.py','rf_backend.py','rf_checkpoint.py')]
    for task in ('data1','data2'):
        files+=list((root/'scripts'/task).glob('*.py'))
        files += [p for p in (root/'datasets'/task).rglob('*') if p.is_file() and p.name!='data.yaml']
    manifest=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(files):
            name=path.relative_to(root).as_posix();content=path.read_bytes()
            z.writestr(name,content);manifest.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        for task in ('data1','data2'):
            names=['sneakers','formal','boots','sandals','slippers','heels','others'] if task=='data1' else ['outsole','heel lining','not_sneakers']
            config='path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'
            config+=''.join(f"  {i}: '{name}'\n" for i,name in enumerate(names))
            z.writestr(f'datasets/{task}/data.yaml',config)
        z.writestr('bundle_manifest.json',json.dumps(dict(files=manifest,
            data1_images=5746,data2_images=1465,note='All prepared, matched labeled data; raw unlabeled/unmatched source files excluded. Relative dataset paths.'),indent=2))
    cells=[]
    def cell(kind,text):
        value=dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
        if kind=='code': ast.parse(text);value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell('markdown', '''# 신발 Task1 + Task2 전체 데이터 GPU 학습
RF-DETR Small을 두 task에서 각각 학습합니다. **런타임 → 런타임 유형 변경 → GPU** 선택 후 **모두 실행**하세요.
파일 선택창에는 `men_dii_all_colab.zip`을 업로드합니다. task1 5,746장, task2 1,465장의 준비된 전체 데이터가 포함됩니다.
기본값은 전체 train split의 10 epochs이며, task별 best.pth 하나를 Google Drive에 자동 저장합니다. 최초 Drive 연결을 직접 승인하세요. val 평가·전후 EDA·샘플 인퍼런스 후 결과 ZIP도 Drive에 저장하고 다운로드합니다.
test는 기본적으로 사용하지 않습니다. 무료 GPU 배정·연속 실행 시간은 보장되지 않지만, 이미 Drive에 저장된 최적 모델과 결과는 런타임 종료 후에도 남습니다.
''')
    cell('code', '''# 실행 설정
EPOCHS = 10
BATCH = 2
GRAD_ACCUM = 8  # 유효 batch 16. 메모리 부족 시 BATCH=1, GRAD_ACCUM=16
FINAL_TEST = False  # 모델·설정을 확정한 최종 평가 때만 True
SAVE_TO_DRIVE = True  # 기본: Drive 연결 승인 후 최적 모델과 모든 결과를 Drive에 자동 저장
''')
    cell('code', '''from google.colab import files
from pathlib import Path
import zipfile, json, hashlib, os, subprocess, sys, time
from datetime import datetime

uploaded = files.upload()
filename = 'men_dii_all_colab.zip'
if filename not in uploaded:
    raise ValueError('men_dii_all_colab.zip을 선택하세요.')
PROJECT = Path('/content/men_dii')
if PROJECT.exists():
    raise FileExistsError('이 런타임에 이미 프로젝트가 있습니다. 새 런타임을 사용하거나 다음 셀부터 진행하세요.')
PROJECT.mkdir()
with zipfile.ZipFile(filename) as z:
    for entry in z.infolist():
        target = (PROJECT / entry.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            raise ValueError('안전하지 않은 압축 경로')
    z.extractall(PROJECT)
manifest = json.loads((PROJECT/'bundle_manifest.json').read_text())
for item in manifest['files']:
    if hashlib.sha256((PROJECT/item['path']).read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('파일 손상: '+item['path'])
# 다른 PC/Colab으로 옮긴 경로를 명시적으로 고정
for task in ('data1','data2'):
    config = PROJECT/'datasets'/task/'data.yaml'
    text = config.read_text().replace('path: .', 'path: '+str(config.parent))
    config.write_text(text)
del uploaded
print('전체 데이터 압축 해제 및 파일 무결성 확인 완료')
''')
    cell('code', '''# 패키지 설치: 학습은 별도 Python 프로세스로 실행하여 커널의 기존 import와 분리
subprocess.run([sys.executable,'-m','pip','install','-r',str(PROJECT/'requirements_task2.txt')],check=True)
subprocess.run([sys.executable,'-c',
    "import torch; print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'); assert torch.cuda.is_available(), 'Colab GPU 런타임을 선택하세요.'"],check=True)
RUN_ID = 'colab_'+datetime.now().strftime('%Y%m%d_%H%M%S')
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RUN_ROOT = Path('/content/drive/MyDrive/men_dii_results')/RUN_ID
else:
    RUN_ROOT = PROJECT/'results'/RUN_ID
print('결과 저장 위치:',RUN_ROOT)
''')
    cell('code', '''# 두 task를 순서대로 전체 데이터 학습 → val 평가 → 결과 EDA → 샘플 인퍼런스
command=[sys.executable,'-u',str(PROJECT/'run_colab_all.py'),
         '--epochs',str(EPOCHS),'--batch',str(BATCH),'--grad-accum',str(GRAD_ACCUM),'--output',str(RUN_ROOT)]
if FINAL_TEST: command.append('--final-test')
completion=subprocess.run(command,cwd=PROJECT)
if completion.returncode:
    print('일부 단계가 실패했습니다. 다음 셀에서 로그를 포함한 결과를 다운로드할 수 있습니다.')
else:
    print('두 task 실행 완료')
''')
    cell('code', '''# 실제 결과를 확인하고 다운로드 (점수는 실행 전까지 비어 있음)
result_archive = Path(str(RUN_ROOT)+'.zip')
if (RUN_ROOT/'run_status.json').exists():
    print((RUN_ROOT/'run_status.json').read_text())
for task in ('data1','data2'):
    metrics_file = RUN_ROOT/task/'val'/'metrics.json'
    if metrics_file.exists():
        metrics=json.loads(metrics_file.read_text())
        print(task,'AP:',metrics['primary_ap'],'고정 임계값:',metrics['primary_operating'])
if not result_archive.exists():
    import shutil
    if RUN_ROOT.exists():
        result_archive=Path(shutil.make_archive(str(RUN_ROOT),'zip',root_dir=RUN_ROOT))
    else:
        raise FileNotFoundError('학습 결과 폴더가 없습니다.')
files.download(str(result_archive))
''')
    notebook=dict(cells=cells,metadata=dict(colab=dict(name='men_dii_all_colab.ipynb'),
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    (exports/'men_dii_all_colab.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for task,count in [('data1',5746),('data2',1465)]:
            assert sum(n.startswith(f'datasets/{task}/images/') for n in z.namelist())==count
            assert sum(n.startswith(f'datasets/{task}/labels/') for n in z.namelist())==count
    print(f'ZIP: {archive} ({archive.stat().st_size/2**20:.1f} MiB)')
    print(f'Notebook: {exports/"men_dii_all_colab.ipynb"}')


if __name__=='__main__': main()

"""Run both full datasets sequentially on a Colab GPU; save all results.

Use from the supplied notebook after uploading/extracting the project bundle.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--grad-accum',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--final-test',action='store_true',help='Optional final holdout evaluation after freezing settings')
    args=parser.parse_args()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU not available. Select a GPU runtime in Colab.')
    if args.epochs<1 or args.batch<1 or args.grad_accum<1: raise ValueError('Training settings must be positive.')
    root=Path(__file__).resolve().parent
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    statuses={}
    def save_status():
        (out/'run_status.json').write_text(json.dumps(dict(model='RFDETRSmall',epochs=args.epochs,
            batch=args.batch,grad_accum=args.grad_accum,gpu=torch.cuda.get_device_name(0),
            full_training_splits=True,final_test_requested=args.final_test,tasks=statuses),indent=2),encoding='utf-8')
    def step(task,stage,script,extra):
        task_root=out/task;task_root.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(root/'scripts'/task/f'{script}.py'),'--model','rf-detr-small',
                 '--output',str(task_root/stage),*map(str,extra)]
        print(f'[{task}] {stage} started',flush=True)
        started=time.monotonic();last=started
        with (task_root/f'{stage}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                log.write(line);log.flush()
                if time.monotonic()-last>20 or 'checkpoint saved' in line.lower():
                    print(line.rstrip(),flush=True);last=time.monotonic()
            code=process.wait()
        statuses[task]['stages'][stage]=dict(exit_code=code,seconds=round(time.monotonic()-started,1))
        save_status()
        if code: raise RuntimeError(f'{task}/{stage} failed. See {task_root/stage}.log')
        print(f'[{task}] {stage} completed',flush=True)
    try:
        for task in ('data1','data2'):
            statuses[task]=dict(status='running',stages={});save_status()
            try:
                base=root/'datasets'/task
                step(task,'pre_eda','pre_eda',[])
                step(task,'train','train',['--device','0','--epochs',args.epochs,'--batch',args.batch,
                     '--grad-accum',args.grad_accum,'--workers','0','--patience','10'])
                weights=out/task/'train'/'fit'/'best.pth'
                step(task,'val','evaluate',['--device','0','--weights',weights,'--split','val','--select-threshold'])
                threshold=out/task/'val'/'threshold.json'
                step(task,'post_val','post_eda',['--evaluation',out/task/'val','--training',out/task/'train'/'fit'])
                # A validation inference sample is only an execution demonstration, not another test set.
                source=out/task/'inference_inputs';source.mkdir()
                for path in sorted((base/'images'/'val').iterdir())[:20]: shutil.copy2(path,source/path.name)
                step(task,'inference','infer',['--device','0','--weights',weights,'--threshold-config',threshold,'--source',source])
                config=json.loads(threshold.read_text(encoding='utf-8'))
                statuses[task]['validation_constraints_met']=config['constraints_met']
                if args.final_test and config['constraints_met']:
                    step(task,'test','evaluate',['--device','0','--weights',weights,'--split','test','--threshold-config',threshold])
                    step(task,'post_test','post_eda',['--evaluation',out/task/'test','--training',out/task/'train'/'fit'])
                statuses[task]['status']='completed'
                statuses[task]['test_evaluated']=bool(args.final_test and config['constraints_met'])
            except Exception as exc:
                statuses[task]['status']='failed';statuses[task]['error']=str(exc)
                print(str(exc),flush=True)
            save_status()
    finally:
        save_status()
        archive=shutil.make_archive(str(out),'zip',root_dir=out)
        print(f'RESULT_ARCHIVE={archive}',flush=True)
    if any(s['status']=='failed' for s in statuses.values()): sys.exit(1)


if __name__=='__main__': main()

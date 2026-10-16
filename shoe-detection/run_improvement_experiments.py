"""Local validation-only ablations with no additional source data."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from pipeline.run import dataset, images_in, labels_for, write_json, write_csv
from pipeline.metrics import evaluate, iou

ROOT = Path(__file__).resolve().parent


def windows(width, height):
    """Four overlapping windows, each 62.5% of the original dimensions."""
    tw, th = round(width*.625), round(height*.625)
    return [(x,y,x+tw,y+th) for y in (0,height-th) for x in (0,width-tw)]


def crop_boxes(boxes, window, minimum_visibility=.5):
    x,y,right,bottom = window
    result=[]
    for g in boxes:
        a,b,c,d=g['box']
        clipped=[max(a,x),max(b,y),min(c,right),min(d,bottom)]
        ca,cb,cc,cd=clipped
        visible=max(0,cc-ca)*max(0,cd-cb)/max((c-a)*(d-b),1e-9)
        if visible>=minimum_visibility and cc-ca>=2 and cd-cb>=2:
            result.append(dict(**{'class':g['class']},box=[ca-x,cb-y,cc-x,cd-y]))
    return result


def link(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(source,destination)
    except OSError: shutil.copy2(source,destination)


def prepare(out, balanced=False):
    """Original full train frames + deterministic crops; val stays full frame."""
    from PIL import Image
    import yaml
    source=ROOT/'datasets/data2'
    names=yaml.safe_load((source/'data.yaml').read_text())['names']
    out.mkdir(parents=True,exist_ok=False)
    manifest=[]
    train_sources={p.name for p in (source/'images/train').iterdir()}
    val_sources={p.name for p in (source/'images/val').iterdir()}
    if train_sources & val_sources:
        raise ValueError('Source filename leakage between train and val')
    for split in ('train','val'):
        for folder in ('images','labels'): (out/folder/split).mkdir(parents=True)
        for path in sorted((source/'images'/split).iterdir()):
            label=source/'labels'/split/(path.stem+'.txt')
            link(path,out/'images'/split/path.name)
            link(label,out/'labels'/split/label.name)
            with Image.open(path) as im:
                boxes=labels_for(path,source,split,im.width,im.height,len(names))
                entries=[(path.name,boxes)]
                if split=='train':
                    for index,window in enumerate(windows(im.width,im.height)):
                        cropped=crop_boxes(boxes,window)
                        # Keep even empty crops: these are backgrounds in labeled images,
                        # not independently confirmed healthy-shoe photographs.
                        name=f'{path.stem}__tile{index}.jpg'
                        im.crop(window).convert('RGB').save(out/'images'/split/name,quality=95)
                        w,h=window[2]-window[0],window[3]-window[1]
                        lines=[]
                        for g in cropped:
                            a,b,c,d=g['box']
                            assert 0<=a<c<=w and 0<=b<d<=h
                            lines.append(f"{g['class']} {(a+c)/2/w:.9f} {(b+d)/2/h:.9f} {(c-a)/w:.9f} {(d-b)/h:.9f}")
                        (out/'labels'/split/Path(name).with_suffix('.txt')).write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf-8')
                        entries.append((name,cropped))
            for name,boxes in entries:
                manifest.append(dict(split=split,file_name=name,source=path.name,repeat=0))
                # Target the weakest damage class; exclude six not_sneakers
                # samples from aggressive repeats. No validation repetition.
                repeats=2 if balanced and split=='train' and any(g['class']==0 for g in boxes) else 0
                for repeat in range(1,repeats+1):
                    new=Path(name).stem+f'__repeat{repeat}'+Path(name).suffix
                    link(out/'images'/split/name,out/'images'/split/new)
                    link(out/'labels'/split/Path(name).with_suffix('.txt'),out/'labels'/split/Path(new).with_suffix('.txt'))
                    manifest.append(dict(split=split,file_name=new,source=path.name,repeat=repeat))
    config=dict(path=str(out.resolve()),train='images/train',val='images/val',test='images/test',names=names)
    (out/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    write_json(out/'manifest.json',manifest)
    write_json(out/'preparation.json',dict(train_images=sum(r['split']=='train' for r in manifest),
        val_images=sum(r['split']=='val' for r in manifest),tile_fraction=.625,
        minimum_box_visibility=.5,outsole_total_repetitions=3 if balanced else 1,
        original_dataset_unchanged=True,test_materialized=False))
    return out/'data.yaml'


def merge_predictions(predictions, threshold=.5, max_det=300):
    """Class-aware greedy NMS only for merging overlapping tiled views."""
    kept=[]
    for p in sorted(predictions,key=lambda p:p['confidence'],reverse=True):
        if not any(q['class']==p['class'] and iou(q['box'],p['box'])>threshold for q in kept):
            kept.append(p)
            if len(kept)>=max_det: break
    return kept


def score(records,names,out):
    from pipeline.rf_backend import coco_ap
    out.mkdir(parents=True,exist_ok=False)
    ap=coco_ap(records,names,out)
    fixed=evaluate(records,names,.25)
    for c in fixed['classes']: c.update(ap[c['class_id']])
    write_json(out/'metrics.json',dict(confidence=.25,metrics=fixed))
    write_csv(out/'class_metrics.csv',fixed['classes'])
    write_json(out/'records.json',dict(names=names,records=records))
    sweep=[]
    for n in range(1,20):
        for c in evaluate(records,names,n/20)['classes']:
            sweep.append(dict(confidence=n/20,**c))
    write_csv(out/'class_threshold_sweep.csv',sweep)
    recommended=[]
    for i,name in enumerate(names):
        candidates=[r for r in sweep if r['class_id']==i and r['support']]
        if not candidates: continue
        target=.95 if name=='sneakers' else .90
        feasible=[r for r in candidates if r['recall']>=target]
        best=max(feasible or candidates,key=lambda r:(r['precision'] if feasible else r['f1'],r['confidence']))
        recommended.append(dict(target=target,target_met=bool(feasible),
            selection='precision_under_recall_constraint' if feasible else 'exploratory_max_f1',**best))
    write_json(out/'class_threshold_candidates.json',recommended)
    return fixed


def tiled_evaluate(weights,task,out):
    from PIL import Image
    from pipeline.rf_backend import setup
    from rfdetr import RFDETR
    settings=SimpleNamespace(device='0',model='rf-detr-small',threads=4)
    setup(settings)
    model=RFDETR.from_checkpoint(str(weights),device=settings.device)
    base,names,config=dataset(SimpleNamespace(data=ROOT/f'datasets/{task}/data.yaml'))
    records=[]
    for index,path in enumerate(images_in(base,config,'val')):
        with Image.open(path) as source:
            im=source.convert('RGB'); w,h=im.size
            predictions=[]
            for window in [(0,0,w,h)]+windows(w,h):
                detection=model.predict(im.crop(window),threshold=.001)
                for box,c,s in zip(detection.xyxy,detection.class_id,detection.confidence):
                    if not 0<=int(c)<len(names): continue
                    a,b,right,bottom=map(float,box);x,y=window[:2]
                    predictions.append(dict(**{'class':int(c)},confidence=float(s),box=[a+x,b+y,right+x,bottom+y]))
            records.append(dict(file_name=path.name,path=str(path),width=w,height=h,
                ground_truth=labels_for(path,base,'val',w,h,len(names)),predictions=merge_predictions(predictions)))
        if (index+1)%20==0: print('Tiled validation',index+1,flush=True)
    return score(records,names,out)


def report(out,baseline):
    rows=[]
    for task in ('data1','data2'):
        saved=json.loads((baseline/task/'val/records.json').read_text())
        destination=out/'baseline'/task/'full_frame'
        if not destination.exists(): score(saved['records'],saved['names'],destination)
    for metrics in sorted(out.rglob('metrics.json')):
        if metrics.parent.name not in ('full_frame','tiled'): continue
        value=json.loads(metrics.read_text())
        for c in value['metrics']['classes']:
            rows.append(dict(experiment=str(metrics.parent.relative_to(out)),**c))
    write_csv(out/'comparison.csv',rows)
    lines=['# Validation experiment results','','All fixed-threshold metrics use confidence=0.25 and IoU=0.5. AP uses maxDets=300. No final test evaluation.','','| Experiment | Class | AP50:95 | Precision | Recall |','|---|---|---:|---:|---:|']
    for r in rows:
        def fmt(key): return 'N/A' if r[key] is None else f"{r[key]*100:.2f}%"
        lines.append(f"| {r['experiment']} | {r['name']} | {fmt('ap50_95')} | {fmt('precision')} | {fmt('recall')} |")
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_suite(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    baseline=args.baseline.resolve()
    status=dict(status='running',started=datetime.now().astimezone().isoformat(),experiments={})
    def save(): write_json(out/'status.json',status)
    def command(label,arguments):
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{label} failed; inspect log')
    save();report(out,baseline)
    prepared={}
    status['preparation']='running';save()
    for variant in ('tiles','balanced'):
        prepared[variant]=prepare(out/'prepared'/variant,balanced=variant=='balanced')
    status['preparation']='completed';save()
    plan=[('data1_long30','data1',None),('data2_long30','data2',None),
          ('data2_tiles30','data2','tiles'),('data2_tiles_balanced30','data2','balanced')]
    for name,task,variant in plan:
        status['experiments'][name]=dict(status='running');save()
        try:
            folder=out/name;folder.mkdir()
            data=prepared[variant] if variant else ROOT/f'datasets/{task}/data.yaml'
            common=['--task',task,'--model','rf-detr-small','--device','0','--batch','2','--grad-accum','8','--workers','0']
            command(name+'_train',['-m','pipeline.run',*common,'--action','train','--data',data,
                '--epochs',args.epochs,'--patience','10','--output',folder/'train'])
            weights=folder/'train/fit/best.pth'
            command(name+'_val',['-m','pipeline.run',*common,'--action','evaluate','--weights',weights,
                '--split','val','--conf','.25','--output',folder/'val_raw'])
            saved=json.loads((folder/'val_raw/records.json').read_text())
            score(saved['records'],saved['names'],folder/'full_frame')
            command(name+'_post',['-m','pipeline.run',*common,'--action','post_eda','--evaluation',folder/'val_raw',
                '--training',folder/'train/fit','--output',folder/'post_val'])
            if variant:
                tiled_evaluate(weights,task,folder/'tiled')
            status['experiments'][name]=dict(status='completed')
        except Exception as exc:
            status['experiments'][name]=dict(status='failed',error=str(exc))
        save();report(out,baseline)
    try:
        tiled_evaluate(baseline/'data2/train/fit/best.pth','data2',out/'baseline/data2/tiled')
    except Exception as exc:
        status['baseline_tile_error']=str(exc)
    report(out,baseline)
    status['status']='completed' if all(x['status']=='completed' for x in status['experiments'].values()) else 'completed_with_failures'
    status['finished']=datetime.now().astimezone().isoformat();save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    os.environ['PYTHONIOENCODING']='utf-8'
    run_suite(args)

"""RF-DETR Small backend for both tasks (Apache-2.0 upstream, pinned to 1.11.2).

No Ultralytics import is used by this backend.
"""
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

from .metrics import evaluate, iou, damage_presence, any_overlap_ap
from .run import (ROOT, IMAGE_EXTENSIONS, dataset, images_in, labels_for, write_json,
                  write_csv, load_json, overlay, require_weights, fixed_conf,
                  select_threshold)


def setup(args):
    # Windows Korean console defaults to cp949; upstream Rich tables use Unicode.
    os.environ.setdefault('PYTHONIOENCODING','utf-8')
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    for variable, folder in [('HF_HOME','huggingface'), ('TORCH_HOME','torch'),
                             ('RF_HOME','rfdetr'), ('MPLCONFIGDIR','matplotlib')]:
        location = ROOT/'.tools'/'cache'/folder
        location.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(location))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    import torch
    torch.set_num_threads(args.threads)
    if args.device.isdigit():
        args.device = f'cuda:{args.device}'
    if args.model != 'rf-detr-small':
        raise ValueError('This backend supports --model rf-detr-small only.')


def train(args, out):
    setup(args)
    import rfdetr.training as training
    from rfdetr import RFDETRSmall
    from .rf_checkpoint import SingleBestCallback
    from pytorch_lightning.callbacks import ModelCheckpoint
    base, names, _ = dataset(args)
    original_builder = training.build_trainer
    selected = {}

    def builder(*positional, **kwargs):
        trainer = original_builder(*positional, **kwargs)
        callback = SingleBestCallback(str(out/'fit'))
        # Remove both latest/periodic archives and the framework's multi-file best callback.
        trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)] + [callback]
        if args.limit_train_batches:
            trainer.limit_train_batches = args.limit_train_batches
        if args.limit_val_batches:
            trainer.limit_val_batches = args.limit_val_batches
        selected['callback'] = callback
        return trainer

    model = RFDETRSmall(device=args.device)
    started = time.perf_counter()
    with patch.object(training, 'build_trainer', builder):
        model.train(dataset_dir=str(base), dataset_file='yolo', output_dir=str(out/'fit'),
                    epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
                    num_workers=args.workers, resolution=args.imgsz, device=args.device,
                    seed=args.seed, use_ema=False, multi_scale=False, expanded_scales=False,
                    tensorboard=False, wandb=False, mlflow=False, clearml=False,
                    run_test=False, progress_bar='tqdm', num_sanity_val_steps=0,
                    early_stopping=True, early_stopping_patience=args.patience,
                    early_stopping_min_delta=0.0, early_stopping_use_ema=False,
                    eval_max_dets=args.max_det, log_per_class_metrics=True,
                    class_names=names, compute_val_loss=True)
    callback = selected['callback']
    best = out/'fit'/'best.pth'
    if not best.is_file():
        raise RuntimeError('No finite validation result produced a best checkpoint.')
    write_json(out/'training.json', dict(backend='rfdetr', version='1.11.2', model='RFDETRSmall',
               best=str(best.resolve()), best_epoch=callback.selected_epoch,
               best_map50_95=callback.selected_score, elapsed_seconds=time.perf_counter()-started,
               smoke_test=bool(args.limit_train_batches or args.limit_val_batches),
               limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
               classes=names, use_ema=False, checkpoint_policy='strictly_best_validation_map_only',
               note='Partial-batch smoke metrics are not full validation performance. No automatic test evaluation.'))
    print(f'Best checkpoint: {best}')


def model_for(args, names):
    setup(args)
    from rfdetr import RFDETR
    model = RFDETR.from_checkpoint(str(require_weights(args)), device=args.device)
    if list(model.class_names) != names:
        raise ValueError(f'Checkpoint class mismatch: {model.class_names} != {names}')
    return model


def predict(model, args, paths, names, base=None, split=None, floor=.001):
    from PIL import Image
    records=[]
    for index,path in enumerate(paths):
        with Image.open(path) as source:
            image=source.convert('RGB')
        started=time.perf_counter()
        result=model.predict(image, threshold=floor)
        predictions=[]
        reserved_slots=0
        for box,c,score in zip(result.xyxy,result.class_id,result.confidence):
            if int(c)==len(names):
                # RF-DETR's sigmoid classifier has num_classes+1 output slots.
                # YOLO targets are 0..num_classes-1; the final slot is not a task class.
                reserved_slots+=1
                continue
            if not 0<=int(c)<len(names):
                raise ValueError(f'Unexpected prediction class ID: {c}')
            predictions.append(dict(**{'class':int(c)},confidence=float(score),box=[float(v) for v in box]))
        predictions=sorted(predictions,key=lambda p:p['confidence'],reverse=True)[:args.max_det]
        records.append(dict(file_name=path.name,path=str(path.resolve()),width=image.width,height=image.height,
                            ground_truth=labels_for(path,base,split,image.width,image.height,len(names)) if base else [],
                            predictions=predictions,reserved_slot_predictions_removed=reserved_slots,
                            elapsed_ms=(time.perf_counter()-started)*1000))
        if (index+1)%20==0: print(f'Predicted {index+1}/{len(paths)}',flush=True)
    return records


def infer(args,out):
    _,names,_=dataset(args)
    model=model_for(args,names)
    conf=fixed_conf(args,load_json(args.threshold_config) if args.threshold_config else None)
    if not args.source: raise ValueError('--source is required.')
    source=Path(args.source)
    paths=sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if source.is_dir() else [source]
    if not paths or any(not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS for p in paths):
        raise ValueError('RF-DETR task2 currently accepts an image or image directory.')
    records=predict(model,args,paths,names,floor=conf)
    if args.task=='data2':
        for r in records:r['damage_presence']=damage_presence(r['predictions'],conf)
    write_json(out/'predictions.json',dict(backend='rfdetr',confidence=conf,records=records))
    for r in records: overlay(Path(r['path']),[],r['predictions'],names,out/'visuals'/r['file_name'],conf)


def coco_ap(records,names,out,max_det=300,return_precision=False):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    data=dict(info={},images=[],annotations=[],categories=[dict(id=i,name=n) for i,n in enumerate(names)])
    detections=[]
    for i,r in enumerate(records):
        data['images'].append(dict(id=i,file_name=r['file_name'],width=r['width'],height=r['height']))
        for g in r['ground_truth']:
            x1,y1,x2,y2=g['box']; b=[x1,y1,x2-x1,y2-y1]
            data['annotations'].append(dict(id=len(data['annotations'])+1,image_id=i,category_id=g['class'],bbox=b,area=b[2]*b[3],iscrowd=0))
        for p in r['predictions']:
            x1,y1,x2,y2=p['box']
            detections.append(dict(image_id=i,category_id=p['class'],score=p['confidence'],bbox=[x1,y1,x2-x1,y2-y1]))
    write_json(out/'ground_truth_coco.json',data);write_json(out/'predictions_coco.json',detections)
    gt=COCO();gt.dataset=data;gt.createIndex()
    if detections: dt=gt.loadRes(detections)
    else:
        dt=COCO();dt.dataset=dict(info={},images=data['images'],categories=data['categories'],annotations=[]);dt.createIndex()
    evaluator=COCOeval(gt,dt,'bbox');evaluator.params.maxDets=[1,10,max_det]
    evaluator.evaluate();evaluator.accumulate()
    precision=evaluator.eval['precision'] # IoU, recall, category, area, max detections
    result={}
    for i in range(len(names)):
        p=precision[:,:,i,0,-1]; p50=precision[0,:,i,0,-1]
        result[i]=dict(ap50=float(p50[p50>=0].mean()) if np.any(p50>=0) else None,
                       ap50_95=float(p[p>=0].mean()) if np.any(p>=0) else None)
    return (result,precision) if return_precision else result


def plot_diagnostics(records,names,conf,out,rule='iou50'):
    import numpy as np
    from .run import plot_setup
    _,precision=coco_ap(records,names,out/'coco_analysis',return_precision=True)
    plt=plot_setup()
    fig,ax=plt.subplots(figsize=(8,6))
    for i,name in enumerate(names):
        p=precision[0,:,i,0,-1]
        valid=p>=0
        if valid.any(): ax.plot(np.linspace(0,1,101)[valid],p[valid],label=name)
    ax.set(xlabel='Recall',ylabel='Interpolated precision',title='COCO PR at IoU=0.50',xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(out/'pr_curve.png',dpi=140);plt.close(fig)
    matrix=np.zeros((len(names)+1,len(names)+1),dtype=int)
    background=len(names)
    for r in records:
        used=set()
        for p in sorted(r['predictions'],key=lambda p:p['confidence'],reverse=True):
            if p['confidence']<conf: continue
            candidates=[(iou(g['box'],p['box']),j) for j,g in enumerate(r['ground_truth']) if j not in used]
            overlap,j=max(candidates,default=(0,-1))
            if j>=0 and (overlap>0 if rule=='any_overlap' else overlap>=.5):
                used.add(j);matrix[r['ground_truth'][j]['class'],p['class']]+=1
            else: matrix[background,p['class']]+=1
        for j,g in enumerate(r['ground_truth']):
            if j not in used: matrix[g['class'],background]+=1
    labels=names+['background']
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(matrix,cmap='Blues')
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col,row,str(matrix[row,col]),ha='center',va='center',color='white' if matrix[row,col]>matrix.max()/2 else 'black')
    ax.set_xticks(range(len(labels)),labels,rotation=30);ax.set_yticks(range(len(labels)),labels)
    ax.set(xlabel='Predicted class',ylabel='Ground truth class',title=f'Spatial confusion, {rule}, confidence={conf:.2f}')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_operating.png',dpi=140);plt.close(fig)
    write_json(out/'confusion_operating.json',dict(labels=labels,matrix=matrix.tolist(),
        matching_rule=rule,note='Class-agnostic spatial matching for confusion analysis; off-diagonal class mistakes differ from class-aware TP matching. Background-background is undefined, stored as zero.'))


def evaluate_task(args,out):
    setup(args)
    base,names,data=dataset(args)
    if args.select_threshold and args.split!='val': raise ValueError('Select thresholds on val only.')
    if args.split=='test' and not args.threshold_config: raise ValueError('Test requires a validation threshold config.')
    if args.select_threshold and args.threshold_config: raise ValueError('Choose threshold selection or loading, not both.')
    config=load_json(args.threshold_config) if args.threshold_config else None
    if not args.select_threshold: fixed_conf(args,config)
    model=model_for(args,names)
    records=predict(model,args,images_in(base,data,args.split),names,base,args.split)
    if args.select_threshold: config=select_threshold(args,records,names,out)
    conf=fixed_conf(args,config)
    rule=getattr(args,'match_rule','iou50')
    metrics=evaluate(records,names,conf,rule=rule)
    reference=evaluate(records,names,conf,rule='iou50')
    overlap_ap=any_overlap_ap(records,names,args.max_det)
    if args.task=='data2':
        for r in records:r['damage_presence']=damage_presence(r['predictions'],conf)
    ap=coco_ap(records,names,out,args.max_det)
    for c in metrics['classes']:
        c.update(ap[c['class_id']]);c.update(overlap_ap[c['class_id']])
    scope=list(range(len(names))) if args.task=='data1' else [0,1]
    available=[i for i in scope if ap[i]['ap50'] is not None]
    primary={k:sum(ap[i][k] for i in available)/len(available) if available else None for k in ('ap50','ap50_95')}
    primary.update(requested_class_ids=scope,included_class_ids=available,complete=len(available)==len(scope))
    supported=[metrics['classes'][i] for i in scope if metrics['classes'][i]['support']]
    operating={k:sum(c[k] for c in supported)/len(supported) if supported else None for k in ('precision','recall','f1')}
    operating['included_class_ids']=[c['class_id'] for c in supported]
    custom_available=[i for i in scope if overlap_ap[i]['ap_any_overlap'] is not None]
    custom_primary=dict(map_any_overlap=sum(overlap_ap[i]['ap_any_overlap'] for i in custom_available)/len(custom_available) if custom_available else None,
                        included_class_ids=custom_available,matching_rule='any_overlap',note='Custom AP, not COCO AP50 or mAP50:95')
    write_json(out/'records.json',dict(task=args.task,backend='rfdetr',split=args.split,names=names,confidence=conf,matching_rule=rule,records=records))
    write_csv(out/'image_metrics.csv',metrics.pop('images'));write_csv(out/'class_metrics.csv',metrics['classes'])
    write_csv(out/'image_metrics_iou50.csv',reference.pop('images'))
    write_json(out/'metrics.json',dict(task=args.task,backend='rfdetr',split=args.split,confidence=conf,
               matching_rule=rule,fixed_threshold=metrics,primary_ap=primary,primary_operating=operating,
               primary_overlap_ap=custom_primary,reference_iou50=reference,
               threshold_constraints_met=config.get('constraints_met') if config else None,
               note='Operating P/R/F1 use matching_rule. ap_any_overlap is custom; ap50 and ap50_95 retain standard COCO criteria. RF-DETR has no native NMS.'))
    if args.normal_source or args.benchmark:
        raise ValueError('RF-DETR optional normal/benchmark evaluation is not enabled in this backend yet.')

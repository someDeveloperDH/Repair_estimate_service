"""RF-DETR (Apache-2.0 sizes only) training, evaluation and inference.

Reuses the project's fixed-threshold matching, threshold selection and post-EDA from run.py so
results stay comparable with the YOLO baseline. AP is computed with pycocotools on the same records.
"""
import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

from . import run
from .metrics import evaluate

ROOT = run.ROOT
# Only the Apache-2.0 detection sizes. A/F/P/XL/2XL are under Roboflow PML and are intentionally absent.
MODELS = {'nano': 'RFDETRNano', 'small': 'RFDETRSmall', 'medium': 'RFDETRMedium', 'large': 'RFDETRLarge'}
BEST = 'checkpoint_best_total.pth'


def model_class(size):
    import rfdetr
    return getattr(rfdetr, MODELS[size])


def subset_dataset(args, out):
    """Hardlink a small deterministic subset (pipeline smoke tests and timing only)."""
    base, names, data = run.dataset(args)
    rng = random.Random(args.seed)
    root = out / 'subset_dataset'
    for split in ('train', 'val', 'test'):
        paths = run.images_in(base, data, split)
        count = args.subset if split == 'train' else max(1, args.subset // 4)
        for path in sorted(rng.sample(paths, min(count, len(paths)))):
            for source, folder in ((path, 'images'), (base/'labels'/split/(path.stem+'.txt'), 'labels')):
                target = root/folder/split/source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    target.write_bytes(source.read_bytes())
    lines = [f"path: '{root.as_posix()}'", 'train: images/train', 'val: images/val', 'test: images/test', 'names:']
    lines += [f"  {i}: '{n}'" for i, n in enumerate(names)]
    (root/'data.yaml').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    args.data = root/'data.yaml'


def train(args, out):
    import rfdetr
    import torch
    if args.subset:
        subset_dataset(args, out)
    base, names, _ = run.dataset(args)
    torch.set_num_threads(args.threads or os.cpu_count())
    model = model_class(args.model)()
    fit = out/'fit'
    started = time.time()
    model.train(dataset_dir=str(base), dataset_file='yolo', output_dir=str(fit), class_names=names,
                epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
                resolution=args.imgsz, device=args.device, num_workers=args.workers, seed=args.seed,
                early_stopping=args.patience > 0, early_stopping_patience=max(args.patience, 1),
                checkpoint_interval=max(args.epochs, 1), tensorboard=False, progress_bar='tqdm',
                run_test=False)
    best = fit/BEST
    if not best.is_file():
        raise RuntimeError(f'Training did not produce {best}')
    run.write_json(out/'training.json', dict(best=str(best.resolve()), model=args.model, classes=names,
                   data=str(args.data.resolve()), epochs=args.epochs, batch=args.batch,
                   grad_accum=args.grad_accum, resolution=args.imgsz, seconds=round(time.time()-started, 1),
                   rfdetr=rfdetr.__version__ if hasattr(rfdetr, '__version__') else None,
                   torch=torch.__version__, license='Apache-2.0 (RF-DETR N/S/M/L)',
                   note='Best checkpoint selected on val mAP@50:95 by RF-DETR. No automatic test evaluation.'))
    print(f'Best weights: {best}')


def load_model(args, names):
    from rfdetr import RFDETR
    model = RFDETR.from_checkpoint(str(run.require_weights(args)), trust_checkpoint=True)
    if list(model.class_names) != list(names):
        raise ValueError(f'Model/dataset class mismatch: {list(model.class_names)} != {list(names)}')
    return model


def detections_to_predictions(detections, nc):
    # The classifier has an extra no-object slot (id == nc) that surfaces at low confidence; it is not a task class.
    return [dict(**{'class': int(c)}, confidence=float(s), box=[float(v) for v in box])
            for box, c, s in zip(detections.xyxy.tolist(), detections.class_id.tolist(),
                                 detections.confidence.tolist()) if 0 <= c < nc]


def predict_records(args, model, paths, names, base=None, split=None, floor=.001):
    from PIL import Image
    records = []
    for start in range(0, len(paths), args.batch):
        chunk = paths[start:start+args.batch]
        images = []
        for path in chunk:
            with Image.open(path) as source:
                images.append(source.convert('RGB'))
        began = time.perf_counter()
        results = model.predict(images, threshold=floor, include_source_image=False)
        per_image = (time.perf_counter()-began)*1000/len(chunk)
        results = results if isinstance(results, list) else [results]
        for path, image, detections in zip(chunk, images, results):
            w, h = image.size
            predictions = detections_to_predictions(detections, len(names))[:args.max_det]
            records.append(dict(file_name=path.name, path=str(path.resolve()), width=w, height=h,
                                ground_truth=run.labels_for(path, base, split, w, h, len(names)) if base else [],
                                predictions=predictions, speed_ms=dict(batch_mean_total=per_image)))
    return records


def coco_ap(records, names):
    """Per-class COCO AP50 and AP50:95 from records; classes without ground truth are omitted."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, annotations, detections = [], [], []
    for i, r in enumerate(records, 1):
        images.append(dict(id=i, file_name=r['file_name'], width=r['width'], height=r['height']))
        for g in r['ground_truth']:
            x1, y1, x2, y2 = g['box']
            annotations.append(dict(id=len(annotations)+1, image_id=i, category_id=g['class'],
                                    bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1), iscrowd=0))
        for p in r['predictions']:
            x1, y1, x2, y2 = p['box']
            detections.append(dict(image_id=i, category_id=p['class'], bbox=[x1, y1, x2-x1, y2-y1], score=p['confidence']))
    gt = COCO()
    gt.dataset = dict(images=images, annotations=annotations,
                      categories=[dict(id=i, name=n) for i, n in enumerate(names)])
    gt.createIndex()
    if not detections:
        return {}
    dt = gt.loadRes(detections)
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.evaluate(); evaluator.accumulate()
    precision = evaluator.eval['precision']  # [IoU, recall, class, area, maxDets]
    result = {}
    for k, class_id in enumerate(evaluator.params.catIds):
        if not any(a['category_id'] == class_id for a in annotations):
            continue
        p = precision[:, :, k, 0, -1]
        valid = p[p > -1]
        p50 = precision[0, :, k, 0, -1]
        result[class_id] = dict(ap50=float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else 0.0,
                                ap50_95=float(np.mean(valid)) if valid.size else 0.0)
    return result


def evaluate_task(args, out):
    base, names, data = run.dataset(args)
    if args.select_threshold and args.split != 'val':
        raise ValueError('Threshold selection is allowed on val only.')
    if args.split == 'test' and not args.threshold_config:
        raise ValueError('Final test requires --threshold-config selected on validation.')
    if args.select_threshold and args.threshold_config:
        raise ValueError('Use either --select-threshold or --threshold-config.')
    if not args.select_threshold:
        run.fixed_conf(args, run.load_json(args.threshold_config) if args.threshold_config else None)
    model = load_model(args, names)
    paths = run.images_in(base, data, args.split)
    records = predict_records(args, model, paths, names, base, args.split)
    ap_by_class = coco_ap(records, names)
    config = run.select_threshold(args, records, names, out) if args.select_threshold else (
        run.load_json(args.threshold_config) if args.threshold_config else None)
    conf = run.fixed_conf(args, config)
    metrics = evaluate(records, names, conf)
    for c in metrics['classes']:
        c.update(ap_by_class.get(c['class_id'], dict(ap50=None, ap50_95=None)))
    scope = list(range(len(names))) if args.task == 'data1' else [0, 1]
    available = [i for i in scope if i in ap_by_class]
    primary = {k: sum(ap_by_class[i][k] for i in available)/len(available) if available else None
               for k in ('ap50', 'ap50_95')}
    primary.update(requested_class_ids=scope, included_class_ids=available,
                   complete=all(metrics['classes'][i]['support'] > 0 for i in scope))
    supported = [metrics['classes'][i] for i in scope if metrics['classes'][i]['support']]
    operating = {k: sum(c[k] for c in supported)/len(supported) if supported else None
                 for k in ('precision', 'recall', 'f1')}
    operating.update(included_class_ids=[c['class_id'] for c in supported])
    run.write_json(out/'records.json', dict(task=args.task, split=args.split, names=names, confidence=conf, records=records))
    run.write_csv(out/'image_metrics.csv', metrics.pop('images'))
    run.write_csv(out/'class_metrics.csv', metrics['classes'])
    run.write_json(out/'metrics.json', dict(task=args.task, split=args.split, confidence=conf, model=f'rfdetr-{args.model_size}',
                   matching='confidence ordered, class aware, one-to-one, IoU >= 0.50',
                   ap_method='pycocotools COCOeval bbox, maxDets 300, per-class mean over IoU 0.50:0.95',
                   fixed_threshold=metrics, primary_ap=primary, primary_operating=operating,
                   threshold_constraints_met=config.get('constraints_met') if config else None,
                   caution='No formal acceptance with insufficient independent per-class samples. AP uses confidence floor 0.001; fixed-threshold metrics are separate.'))
    if args.benchmark:
        benchmark(args, model, paths, conf, out)


def benchmark(args, model, paths, conf, out):
    import numpy as np
    from PIL import Image
    timings = []
    sample = (paths * (200//len(paths)+1))[:200]
    for j, path in enumerate([paths[0]]*10 + sample):
        with Image.open(path) as source:
            image = source.convert('RGB')
        start = time.perf_counter()
        model.predict(image, threshold=conf, include_source_image=False)
        elapsed = (time.perf_counter()-start)*1000
        if j >= 10:
            timings.append(elapsed)
    p95 = float(np.percentile(timings, 95))
    run.write_json(out/'latency.json', dict(images=200, unique_images=min(len(paths), 200), warmup=10, batch=1,
                   mean_ms=float(np.mean(timings)), median_ms=float(np.median(timings)), p95_ms=p95,
                   device=args.device, hardware=platform.processor(), p95_target_ms=1000, observed_target_met=p95 <= 1000,
                   note='Includes preprocessing, inference and postprocessing from decoded in-memory images. Excludes disk IO and model loading.'))


def post_eda(args, out):
    run.post_eda(argparse.Namespace(**{**vars(args), 'training': None}), out)
    if args.training:
        curve_file = args.training/'metrics.csv'
        if not curve_file.is_file():
            raise FileNotFoundError(f'Expected RF-DETR fit/metrics.csv: {curve_file}')
        with curve_file.open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        series = {}
        for row in rows:
            if not row.get('epoch'):
                continue
            for k, v in row.items():
                if v and k != 'epoch' and ('loss' in k or 'mAP' in k) and '/' in k:
                    series.setdefault(k, {})[int(float(row['epoch']))] = float(v)
        plt = run.plot_setup()
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for k, points in sorted(series.items()):
            ax = axes[0] if 'loss' in k else axes[1]
            ax.plot(sorted(points), [points[e] for e in sorted(points)], label=k)
        for ax in axes:
            ax.legend(fontsize=7); ax.set_xlabel('epoch')
        fig.tight_layout(); fig.savefig(out/'training_curves.png', dpi=140); plt.close(fig)


def infer(args, out):
    _, names, _ = run.dataset(args)
    if args.source is None:
        raise ValueError('--source is required (image or directory).')
    source = Path(args.source)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in run.IMAGE_EXTENSIONS) if source.is_dir() else [source]
    conf = run.fixed_conf(args, run.load_json(args.threshold_config) if args.threshold_config else None)
    model = load_model(args, names)
    records = predict_records(args, model, paths, names, floor=conf)
    with (out/'predictions.jsonl').open('w', encoding='utf-8') as f:
        for index, r in enumerate(records):
            detections = [dict(class_id=p['class'], name=names[p['class']], confidence=p['confidence'], xyxy=p['box'])
                          for p in r['predictions']]
            f.write(json.dumps(dict(sequence_index=index, path=r['path'], detections=detections, speed_ms=r['speed_ms']))+'\n')
            run.overlay(Path(r['path']), [], r['predictions'], names, out/'visuals'/r['file_name'], conf)


def main(task=None, action=None):
    parser = argparse.ArgumentParser(description='RF-DETR shoe detection pipeline (Apache-2.0 sizes)')
    if task is None:
        parser.add_argument('--task', choices=['data1', 'data2'], required=True)
    if action is None:
        parser.add_argument('--action', choices=['train', 'evaluate', 'post_eda', 'infer'], required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, help='New directory; existing directories are not overwritten')
    parser.add_argument('--model', dest='model_size', choices=sorted(MODELS), default='small')
    parser.add_argument('--weights', type=Path, help=f'Trained {BEST}')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=512, help='RF-DETR resolution; must be divisible by 32')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=4, help='Effective batch = batch * grad-accum')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, help='torch CPU threads (default: all logical cores)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience in epochs; 0 disables')
    parser.add_argument('--subset', type=int, default=0, help='Train on N train images (val/test N/4) for smoke tests')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--select-threshold', action='store_true')
    parser.add_argument('--threshold-config', type=Path)
    parser.add_argument('--conf', type=float)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--source')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--training', type=Path, help='RF-DETR fit directory containing metrics.csv')
    parser.add_argument('--benchmark', action='store_true', help='Also measure batch-1 latency on 200 inputs')
    args = parser.parse_args()
    if task: args.task = task
    if action: args.action = action
    args.model = args.model_size
    args.nms_iou = None  # DETR decoding has no NMS; kept so threshold configs record it explicitly
    if args.data is None: args.data = ROOT/'datasets'/args.task/'data.yaml'
    if args.conf is not None and not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    if args.imgsz % 32:
        parser.error('--imgsz must be divisible by 32 for RF-DETR')
    action_name = args.action
    args.action = f'rfdetr_{action_name}'  # keeps default artifact folders apart from YOLO runs
    out = run.output_dir(args)
    try:
        {'train': train, 'evaluate': evaluate_task, 'post_eda': post_eda, 'infer': infer}[action_name](args, out)
    except Exception as exc:
        run.write_json(out/'error.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    print(f'Output: {out}')


if __name__ == '__main__':
    main()

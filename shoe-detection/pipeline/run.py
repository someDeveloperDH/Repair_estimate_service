"""Reusable implementation behind task-specific command entry points."""
import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from .metrics import evaluate, match, damage_presence, any_overlap_ap

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dataset(args):
    import yaml
    config = args.data.resolve()
    data = yaml.safe_load(config.read_text(encoding='utf-8-sig'))
    base = Path(data.get('path', config.parent))
    if not base.is_absolute():
        base = (config.parent / base).resolve()
    names = data['names']
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    return base, names, data


def images_in(base, data, split):
    directory = base / data[split]
    if not directory.is_dir():
        raise ValueError(f'Expected generated dataset image directory: {directory}')
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f'Empty split: {directory}')
    return paths


def labels_for(path, base, split, width, height, nc):
    label = base / 'labels' / split / (path.stem + '.txt')
    if not label.exists():
        raise FileNotFoundError(f'Missing label, not silently treated as negative: {label}')
    result = []
    for line in label.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = map(float, line.split())
        if not c.is_integer() or not 0 <= c < nc or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'Invalid label: {label}: {line}')
        result.append(dict(**{'class': int(c)}, box=[(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]))
    return result


def output_dir(args):
    out = args.output or ROOT / 'artifacts' / args.task / args.action / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(python=sys.version, platform=platform.platform())
    write_json(out / 'command.json', metadata)
    return out


def plot_setup():
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.tools'/'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def overlay(path, gt, predictions, names, destination, conf=0):
    from PIL import Image, ImageDraw
    with Image.open(path) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    for entries, color, tag in ((gt, '#00c853', 'GT'), (predictions, '#ff1744', 'P')):
        for b in entries:
            if b.get('confidence', 1) < conf:
                continue
            box = b['box']
            draw.rectangle(box, outline=color, width=2)
            text = f"{tag} {names[b['class']]}"
            if 'confidence' in b:
                text += f" {b['confidence']:.2f}"
            draw.text((max(0, box[0]), max(0, box[1]-12)), text, fill=color, stroke_width=1, stroke_fill='black')
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def pre_eda(args, out):
    from PIL import Image, ImageStat
    base, names, data = dataset(args)
    rows, boxes, hashes, dates, perceptual = [], [], {}, {}, {}
    rng = random.Random(args.seed)
    per_class_samples = {i: [] for i in range(len(names))}
    for split in ('train', 'val', 'test'):
        for path in images_in(base, data, split):
            with Image.open(path) as image:
                width, height = image.size
                brightness = ImageStat.Stat(image.convert('L')).mean[0]
                pixels = list(image.convert('L').resize((9, 8)).getdata())
                dhash = sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x)
                            for y in range(8) for x in range(8))
                perceptual.setdefault(str(dhash), []).append(dict(split=split, file_name=path.name))
            gt = labels_for(path, base, split, width, height, len(names))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(dict(split=split, file_name=path.name))
            date = path.stem[:8] if path.stem[:8].isdigit() else path.stem
            dates.setdefault(date, set()).add(split)
            rows.append(dict(split=split, file_name=path.name, date=date, width=width, height=height,
                             brightness=brightness, objects=len(gt), sha256=digest))
            for g in gt:
                x1, y1, x2, y2 = g['box']
                boxes.append(dict(split=split, file_name=path.name, class_id=g['class'], name=names[g['class']],
                                  center_x=(x1+x2)/2/width, center_y=(y1+y2)/2/height,
                                  area_fraction=(x2-x1)*(y2-y1)/width/height, aspect_ratio=(x2-x1)/(y2-y1)))
            for c in {g['class'] for g in gt}:
                per_class_samples[c].append((path, gt))
    counts = []
    for split in ('train', 'val', 'test'):
        for i, name in enumerate(names):
            selected = [b for b in boxes if b['split'] == split and b['class_id'] == i]
            counts.append(dict(split=split, class_id=i, name=name, boxes=len(selected),
                               images=len({b['file_name'] for b in selected})))
    for c, candidates in per_class_samples.items():
        for j, (path, gt) in enumerate(rng.sample(candidates, min(args.samples, len(candidates)))):
            overlay(path, gt, [], names, out / 'label_samples' / names[c].replace(' ', '_') / f'{j}_{path.name}')
    write_csv(out/'images.csv', rows)
    write_csv(out/'boxes.csv', boxes)
    write_csv(out/'class_distribution.csv', counts)
    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    leakage = [g for g in duplicate_groups if len({x['split'] for x in g}) > 1]
    write_json(out/'checks.json', dict(exact_duplicate_groups=duplicate_groups,
               exact_duplicate_cross_split=leakage,
               identical_dhash_cross_split=[g for g in perceptual.values() if len({x['split'] for x in g})>1],
               date_cross_split={d: sorted(s) for d, s in dates.items() if len(s)>1},
               note='SHA256 plus identical dHash candidates. dHash collisions need visual review; this does not detect all near duplicates or cross-date shoe identities.'))
    plt = plot_setup()
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for split in ('train', 'val', 'test'):
        values = [c['boxes'] for c in counts if c['split'] == split]
        axes[0, 0].plot(names, values, marker='o', label=split)
    axes[0, 0].legend(); axes[0, 0].tick_params(axis='x', rotation=35); axes[0, 0].set_title('Class box counts')
    axes[0, 1].hist([b['area_fraction'] for b in boxes], bins=40); axes[0, 1].set_title('Box area / image area')
    axes[0, 2].hist([b['aspect_ratio'] for b in boxes], bins=40); axes[0, 2].set_title('Box aspect ratio')
    axes[1, 0].hexbin([b['center_x'] for b in boxes], [b['center_y'] for b in boxes], gridsize=30); axes[1, 0].set_title('Box centers (normalized)')
    axes[1, 1].hist([r['brightness'] for r in rows], bins=40); axes[1, 1].set_title('Mean grayscale brightness')
    axes[1, 2].hist([r['objects'] for r in rows], bins=range(max(r['objects'] for r in rows)+2)); axes[1, 2].set_title('Objects per image')
    fig.tight_layout(); fig.savefig(out/'distributions.png', dpi=140); plt.close(fig)
    date_counts = []
    for date in sorted(dates):
        for split in sorted(dates[date]):
            date_counts.append(dict(date=date, split=split, images=sum(r['date']==date and r['split']==split for r in rows)))
    write_csv(out/'date_distribution.csv', date_counts)
    write_json(out/'summary.json', dict(images=len(rows), boxes=len(boxes), classes=names,
               empty_classes_by_split=[c for c in counts if not c['boxes']],
               exact_duplicate_cross_split_groups=len(leakage)))


def train(args, out):
    if args.model == 'rf-detr-small':
        from .rf_backend import train as rf_train
        return rf_train(args, out)
    import ultralytics
    from ultralytics import YOLO
    from .trainer import BestOnlyDetectionTrainer
    _, names, _ = dataset(args)
    model = YOLO(args.model)
    model.train(trainer=BestOnlyDetectionTrainer, save=True, save_period=-1,
                data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers, seed=args.seed,
                deterministic=True, patience=args.patience, project=str(out), name='fit',
                exist_ok=False, plots=True, degrees=5, translate=.05, scale=.2,
                fliplr=.5, flipud=0, mosaic=.3 if args.task=='data1' else .1,
                mixup=0, hsv_h=.01, hsv_s=.2, hsv_v=.2, close_mosaic=10)
    if not Path(model.trainer.best).is_file():
        raise RuntimeError('Training did not produce a finite validation best checkpoint.')
    write_json(out/'training.json', dict(best=str(Path(model.trainer.best).resolve()),
               best_epoch=model.trainer._saved_best_epoch,
               best_fitness=model.trainer._saved_best_score,
               checkpoint_policy='best_only_strict_validation_fitness_improvement', classes=names,
               ultralytics=ultralytics.__version__, note='No automatic test evaluation.'))
    print(f'Best weights: {model.trainer.best}')


def require_weights(args):
    if args.weights is None or not args.weights.is_file():
        raise ValueError('--weights must point to an existing task-trained best.pt')
    return args.weights.resolve()


def predict_records(args, paths, names, base=None, split=None, floor=.001):
    from ultralytics import YOLO
    model = YOLO(str(require_weights(args)))
    model_names = [model.names[i] for i in range(len(model.names))]
    if model_names != names:
        raise ValueError(f'Model/dataset class mismatch: {model_names} != {names}')
    records = []
    results = model.predict(source=[str(p) for p in paths], stream=True, conf=floor,
                            iou=args.nms_iou, max_det=args.max_det, imgsz=args.imgsz,
                            device=args.device, batch=args.batch, verbose=False, save=False)
    for result in results:
        path = Path(result.path)
        h, w = result.orig_shape
        predictions = [dict(**{'class': int(c)}, confidence=float(s), box=[float(v) for v in box])
                       for box, c, s in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]
        records.append(dict(file_name=path.name, path=str(path.resolve()), width=w, height=h,
                            ground_truth=labels_for(path, base, split, w, h, len(names)) if base else [],
                            predictions=predictions, speed_ms={k: float(v) for k, v in result.speed.items()}))
    return records


def infer(args, out):
    if args.model == 'rf-detr-small':
        from .rf_backend import infer as rf_infer
        return rf_infer(args, out)
    from ultralytics import YOLO
    _, names, _ = dataset(args)
    model = YOLO(str(require_weights(args)))
    if [model.names[i] for i in range(len(model.names))] != names:
        raise ValueError('Weights do not belong to the selected task.')
    if args.source is None:
        raise ValueError('--source is required (image, directory, video, or stream).')
    config = load_json(args.threshold_config) if args.threshold_config else None
    conf = fixed_conf(args, config)
    result_file = out/'predictions.jsonl'
    with result_file.open('w', encoding='utf-8') as f:
        for index, r in enumerate(model.predict(source=args.source, stream=True, conf=conf, iou=args.nms_iou,
                               imgsz=args.imgsz, device=args.device, max_det=args.max_det,
                               project=str(out), name='visuals', save=True, save_txt=True,
                               save_conf=True, verbose=False)):
            detections = [dict(class_id=int(c), name=names[int(c)], confidence=float(s), xyxy=[float(v) for v in box])
                          for box, c, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(), r.boxes.conf.cpu().tolist())]
            f.write(json.dumps(dict(sequence_index=index, path=str(r.path), detections=detections, speed_ms=r.speed))+'\n')


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_conf(args, config=None):
    if config:
        if config['task'] != args.task or config['weights_sha256'] != fingerprint(require_weights(args)):
            raise ValueError('Threshold config belongs to another task or weights file.')
        for key in ('imgsz', 'nms_iou', 'max_det'):
            if config[key] != getattr(args, key):
                raise ValueError(f'Threshold config {key} differs. Use the same prediction settings.')
        if config.get('selected_on') != 'val':
            raise ValueError('Threshold configuration was not selected on validation.')
        if config.get('matching_rule','iou50')!=getattr(args,'match_rule','iou50'):
            raise ValueError('Threshold config matching rule differs; select a new threshold on validation.')
        return float(config['confidence'])
    if args.conf is None:
        raise ValueError('Provide --threshold-config from validation, or explicit --conf for exploratory use.')
    return args.conf


def select_threshold(args, records, names, out):
    rows, candidates = [], []
    for step in range(1, 20):
        conf = step / 20
        metrics = evaluate(records, names, conf, rule=getattr(args,'match_rule','iou50'))
        required = [0] if args.task == 'data1' else [0, 1]
        target = .95 if args.task == 'data1' else .90
        feasible = all(metrics['classes'][i]['recall'] is not None and metrics['classes'][i]['recall'] >= target for i in required)
        chosen = metrics['classes'] if args.task == 'data1' else metrics['classes'][:2]
        supported = [c for c in chosen if c['support']]
        macro = {k: sum(c[k] for c in supported)/len(supported) if supported else 0 for k in ('precision','recall','f1')}
        row = dict(confidence=conf, feasible=feasible, **macro)
        rows.append(row)
        score = macro['f1'] if args.task=='data1' else macro['precision']
        candidates.append((feasible, score, conf))
    feasible = [c for c in candidates if c[0]]
    best = max(feasible or candidates, key=lambda c: (c[1], c[2]))
    config = dict(task=args.task, confidence=best[2], constraints_met=bool(feasible),
                  weights_sha256=fingerprint(require_weights(args)), weights=str(require_weights(args)),
                  imgsz=args.imgsz, nms_iou=args.nms_iou, max_det=args.max_det, selected_on='val',
                  matching_rule=getattr(args,'match_rule','iou50'),
                  note='If constraints_met is false, exploratory fallback only; target has not been met.')
    write_json(out/'threshold.json', config)
    write_csv(out/'threshold_sweep.csv', rows)
    return config


def evaluate_task(args, out):
    if args.model == 'rf-detr-small':
        from .rf_backend import evaluate_task as rf_evaluate
        return rf_evaluate(args, out)
    import ultralytics
    from ultralytics import YOLO
    base, names, data = dataset(args)
    if args.select_threshold and args.split != 'val':
        raise ValueError('Threshold selection is allowed on val only.')
    if args.split == 'test' and not args.threshold_config:
        raise ValueError('Final test requires --threshold-config selected on validation.')
    if args.select_threshold and args.threshold_config:
        raise ValueError('Use either --select-threshold or --threshold-config.')
    if not args.select_threshold:
        fixed_conf(args, load_json(args.threshold_config) if args.threshold_config else None)
    model = YOLO(str(require_weights(args)))
    if [model.names[i] for i in range(len(model.names))] != names:
        raise ValueError('Model class names do not match the task.')
    ap = model.val(data=str(args.data.resolve()), split=args.split, conf=.001,
                   iou=args.nms_iou, max_det=args.max_det, imgsz=args.imgsz,
                   device=args.device, batch=args.batch, workers=args.workers,
                   plots=True, project=str(out), name='ap', verbose=False)
    paths = images_in(base, data, args.split)
    records = predict_records(args, paths, names, base, args.split)
    config = select_threshold(args, records, names, out) if args.select_threshold else (
        load_json(args.threshold_config) if args.threshold_config else None)
    conf = fixed_conf(args, config)
    rule=getattr(args,'match_rule','iou50')
    metrics = evaluate(records, names, conf, rule=rule)
    ap_by_class = {int(c): dict(ap50=float(ap.box.ap50[i]), ap50_95=float(ap.box.ap[i]))
                   for i, c in enumerate(ap.box.ap_class_index)}
    for c in metrics['classes']:
        c.update(ap_by_class.get(c['class_id'], dict(ap50=None, ap50_95=None)))
    scope = list(range(len(names))) if args.task=='data1' else [0,1]
    available = [i for i in scope if i in ap_by_class]
    primary = {k: sum(ap_by_class[i][k] for i in available)/len(available) if available else None
               for k in ('ap50','ap50_95')}
    primary.update(requested_class_ids=scope, included_class_ids=available,
                   complete=all(metrics['classes'][i]['support'] > 0 for i in scope))
    supported = [metrics['classes'][i] for i in scope if metrics['classes'][i]['support']]
    operating = {k: sum(c[k] for c in supported)/len(supported) if supported else None
                 for k in ('precision','recall','f1')}
    operating.update(included_class_ids=[c['class_id'] for c in supported])
    overlap_ap=any_overlap_ap(records,names,args.max_det)
    for c in metrics['classes']:c.update(overlap_ap[c['class_id']])
    if args.task=='data2':
        for r in records:r['damage_presence']=damage_presence(r['predictions'],conf)
    write_json(out/'records.json', dict(task=args.task, split=args.split, names=names, confidence=conf, matching_rule=rule,records=records))
    image_metrics = metrics.pop('images')
    write_csv(out/'image_metrics.csv', image_metrics)
    write_csv(out/'class_metrics.csv', metrics['classes'])
    write_json(out/'metrics.json', dict(task=args.task, split=args.split, confidence=conf,
               matching_rule=rule,matching='confidence ordered, class aware, one-to-one; selected matching_rule',
               fixed_threshold=metrics, primary_ap=primary, primary_operating=operating,
               ultralytics_version=ultralytics.__version__,
               threshold_constraints_met=config.get('constraints_met') if config else None,
               caution='No formal acceptance with insufficient independent per-class samples. Native AP plots use low confidence; fixed-threshold metrics are separate.'))
    if args.normal_source:
        if args.task != 'data2':
            raise ValueError('--normal-source is supported for data2 only.')
        normal_paths = sorted(p for p in args.normal_source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not normal_paths:
            raise ValueError('No confirmed normal images in --normal-source.')
        normal = predict_records(args, normal_paths, names, floor=conf)
        rows = [dict(file_name=r['file_name'], false_damage_detection=any(
            p['class'] in (0,1) and p['confidence'] >= conf for p in r['predictions'])) for r in normal]
        rate = sum(r['false_damage_detection'] for r in rows)/len(rows)
        write_csv(out/'normal_image_results.csv', rows)
        write_json(out/'normal_metrics.json', dict(images=len(rows), false_positive_image_rate=rate,
                   target=.05, observed_target_met=rate<=.05, minimum_200_images_met=len(rows)>=200,
                   note='Caller must verify these are normal sneaker photos. No automatic ground-truth validation.'))
    if args.benchmark:
        import numpy as np
        import torch
        from PIL import Image
        model = YOLO(str(require_weights(args)))
        def synchronize():
            if args.device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize(int(args.device) if args.device.isdigit() else None)
        timings = []
        benchmark_paths = (paths * (200//len(paths)+1))[:200]
        for j, path in enumerate([paths[0]]*10 + benchmark_paths):
            with Image.open(path) as image:
                pixels=np.asarray(image.convert('RGB'))[:, :, ::-1].copy()  # ndarray input is BGR
            synchronize(); start=time.perf_counter()
            model.predict(source=pixels, conf=conf, iou=args.nms_iou, max_det=args.max_det,
                          imgsz=args.imgsz, device=args.device, batch=1, verbose=False)
            synchronize(); elapsed=(time.perf_counter()-start)*1000
            if j>=10: timings.append(elapsed)
        write_json(out/'latency.json', dict(images=200, unique_images=min(len(paths),200),
                   warmup=10, batch=1, mean_ms=float(np.mean(timings)), median_ms=float(np.median(timings)),
                   p95_ms=float(np.percentile(timings,95)), device=args.device,
                   hardware=platform.processor(), gpu=torch.cuda.get_device_name(int(args.device) if args.device.isdigit() else None) if args.device!='cpu' and torch.cuda.is_available() else None,
                   p95_target_ms=1000, observed_target_met=float(np.percentile(timings,95))<=1000,
                   note='Includes preprocessing, inference and postprocessing from in-memory images. Excludes disk IO, image decoding, and initial model loading. Images repeat if split has fewer than 200.'))


def post_eda(args, out):
    if not args.evaluation or not (args.evaluation/'records.json').is_file():
        raise ValueError('--evaluation must be a completed evaluation output directory.')
    saved = load_json(args.evaluation/'records.json')
    if saved['task'] != args.task:
        raise ValueError('Evaluation belongs to another task.')
    records, names, conf = saved['records'], saved['names'], saved['confidence']
    rule=saved.get('matching_rule','iou50')
    metrics = evaluate(records, names, conf, rule=rule)
    plt = plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key in ('precision','recall','f1'):
        axes[0].plot(names, [c[key] if c[key] is not None else float('nan') for c in metrics['classes']], marker='o', label=key)
    axes[0].set_ylim(0,1.05); axes[0].tick_params(axis='x', rotation=30); axes[0].legend(); axes[0].set_title(f'Fixed confidence={conf:.2f}')
    sizes = metrics['size_recall']
    axes[1].bar(list(sizes), [v['recall'] if v['recall'] is not None else float('nan') for v in sizes.values()]); axes[1].set_ylim(0,1.05); axes[1].set_title('Recall by relative box area')
    fig.tight_layout(); fig.savefig(out/'metrics.png', dpi=140); plt.close(fig)
    examples = {k: [] for k in ('false_positive','false_negative','success')}
    for r in records:
        pairs, fps, fns = match(r['ground_truth'], r['predictions'], conf, rule=rule)
        if fps: examples['false_positive'].append(r)
        if fns: examples['false_negative'].append(r)
        if pairs and not fps and not fns: examples['success'].append(r)
    rng = random.Random(args.seed)
    for kind, candidates in examples.items():
        for r in rng.sample(candidates, min(args.samples, len(candidates))):
            overlay(Path(r['path']), r['ground_truth'], r['predictions'], names, out/kind/r['file_name'], conf)
    write_csv(out/'image_metrics.csv', metrics['images'])
    write_csv(out/'size_recall.csv', [dict(size=k, **v) for k,v in sizes.items()])
    saved_metrics = load_json(args.evaluation/'metrics.json')
    ap_by_class = saved_metrics['fixed_threshold']['classes']
    fig, ax = plt.subplots(figsize=(10,5))
    for key in ('ap50','ap50_95'):
        ax.plot(names, [c[key] if c[key] is not None else float('nan') for c in ap_by_class], marker='o', label=key)
    ax.set_ylim(0,1.05); ax.tick_params(axis='x', rotation=30); ax.legend()
    fig.tight_layout(); fig.savefig(out/'class_ap.png', dpi=140); plt.close(fig)
    targets = ([.95,.85,.90,.80,.80,.80,.75] if args.task=='data1' else [.80,.90,.80])
    recalls = ([.95,.85,.90,.80,.80,.80,.75] if args.task=='data1' else [.90,.90,.80])
    target_rows = []
    for c in ap_by_class:
        i=c['class_id']
        for key, target in (('ap50',targets[i]),('precision',targets[i]),('recall',recalls[i])):
            value = c[key]
            enough = c['support']>=50 and c['images']>=20
            status = 'N/A' if value is None else ('INSUFFICIENT_SAMPLE' if not enough else ('MET_OBSERVED' if value>=target else 'BELOW_TARGET'))
            target_rows.append(dict(name=c['name'], metric=key, target=target, actual=value,
                                    support=c['support'], images=c['images'], status=status))
    write_csv(out/'class_target_comparison.csv', target_rows)
    global_targets = dict(ap50=.90, ap50_95=.65, precision=.90, recall=.90, f1=.90) if args.task=='data1' else dict(ap50=.85, ap50_95=.50, precision=.85, recall=.90, f1=.87)
    overall = []
    scope = saved_metrics['primary_ap']['requested_class_ids']
    enough = all(ap_by_class[i]['support']>=50 and ap_by_class[i]['images']>=20 for i in scope)
    for key,target in global_targets.items():
        source = saved_metrics['primary_ap'] if key.startswith('ap') else saved_metrics['primary_operating']
        value = source[key]
        status = 'N/A' if value is None else ('INSUFFICIENT_SAMPLE' if not enough else ('MET_OBSERVED' if value>=target else 'BELOW_TARGET'))
        overall.append(dict(metric=key, target=target, actual=value, included_class_ids=str(source['included_class_ids']), status=status))
    write_csv(out/'overall_target_comparison.csv', overall)
    # Export actual confusion and PR artifacts from AP evaluation with their own settings.
    native = args.evaluation/'ap'
    if native.exists():
        for p in native.glob('*.png'):
            shutil.copy2(p, out/('native_'+p.name))
    if args.training:
        curve_file = args.training/'results.csv'
        if not curve_file.is_file():
            curve_file = args.training/'metrics.csv'
        if not curve_file.is_file():
            raise FileNotFoundError(f'Expected training fit/results.csv or metrics.csv: {curve_file}')
        with curve_file.open(encoding='utf-8-sig') as f:
            history = [{k.strip(): v for k,v in row.items()} for row in csv.DictReader(f)]
        if history:
            keys = [k for k in history[0] if 'loss' in k or 'mAP' in k]
            fig, axes = plt.subplots(1,2,figsize=(15,6))
            for k in keys:
                ax = axes[0] if 'loss' in k else axes[1]
                selected = [r for r in history if r.get('epoch') not in ('',None) and r.get(k) not in ('',None)]
                if selected:
                    ax.plot([float(r['epoch']) for r in selected], [float(r[k]) for r in selected], marker='o',label=k)
            for ax in axes:
                ax.legend(fontsize=8); ax.set_xlabel('epoch')
            fig.tight_layout(); fig.savefig(out/'training_curves.png', dpi=140); plt.close(fig)
    write_json(out/'summary.json', dict(task=args.task, split=saved['split'], confidence=conf,matching_rule=rule,
               example_image_counts={k: len(v) for k,v in examples.items()},
               size_recall=sizes, note='Green=ground truth; red=prediction. Native plots use AP settings, not the operating threshold. Library speeds are not synchronized batch-1 p95 benchmarks.'))
    if saved.get('backend')=='rfdetr':
        from .rf_backend import plot_diagnostics
        plot_diagnostics(records,names,conf,out,rule=rule)


def main(task=None, action=None):
    parser = argparse.ArgumentParser(description='Shoe detection pipeline')
    if task is None:
        parser.add_argument('--task', choices=['data1','data2'], required=True)
    if action is None:
        parser.add_argument('--action', choices=['pre_eda','train','infer','evaluate','post_eda'], required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, help='New directory; existing directories are not overwritten')
    parser.add_argument('--model', default=None)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--select-threshold', action='store_true')
    parser.add_argument('--threshold-config', type=Path)
    parser.add_argument('--conf', type=float)
    parser.add_argument('--match-rule', choices=['any_overlap','iou50'], default='any_overlap',
                        help='Same-class positive-area overlap is correct by default; iou50 retains the old criterion')
    parser.add_argument('--nms-iou', type=float, default=.7)
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--source')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--training', type=Path, help='Training output fit directory containing results.csv')
    parser.add_argument('--normal-source', type=Path, help='Directory of independently confirmed normal sneaker photos, data2 only')
    parser.add_argument('--benchmark', action='store_true', help='Also measure synchronized batch-1 latency on 200 inputs')
    parser.add_argument('--limit-train-batches', type=int, default=0, help='RF-DETR smoke test only; 0 means full split')
    parser.add_argument('--limit-val-batches', type=int, default=0, help='RF-DETR smoke test only; 0 means full split')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4, help='RF-DETR CPU intra-op thread count')
    args = parser.parse_args()
    if task: args.task = task
    if action: args.action = action
    if args.model is None: args.model = 'rf-detr-small'
    if args.data is None: args.data = ROOT/'datasets'/args.task/'data.yaml'
    if args.conf is not None and not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    if args.samples < 0 or not 0 < args.nms_iou <= 1:
        parser.error('Invalid --samples or --nms-iou')
    out = output_dir(args)
    try:
        {'pre_eda': pre_eda, 'train': train, 'infer': infer,
         'evaluate': evaluate_task, 'post_eda': post_eda}[args.action](args, out)
    except Exception as exc:
        write_json(out/'error.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    print(f'Output: {out}')


if __name__ == '__main__':
    main()

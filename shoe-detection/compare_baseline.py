"""Compare reconstructed pre-fit RF-DETR with saved validation predictions.

Intercept Trainer.fit before any optimizer step, using the original training
initialization path. Also retain untouched COCO model predictions; its labels
have no semantic mapping to this project's labels.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.run import dataset, images_in, write_json, write_csv, fingerprint
from pipeline.rf_backend import setup, predict, coco_ap
from pipeline.metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.run/'baseline_comparison'
    out.mkdir(exist_ok=False)
    settings = SimpleNamespace(device='0', model='rf-detr-small', threads=4,
                               max_det=300, imgsz=512)
    setup(settings)
    import torch
    import rfdetr.training as training
    from rfdetr import RFDETRSmall
    rows = []
    for task in ('data1', 'data2'):
        dest = out/task
        dest.mkdir()
        settings.data = root/'datasets'/task/'data.yaml'
        base, names, config = dataset(settings)
        trained = json.loads((args.run/task/'val/records.json').read_text())
        paths = images_in(base, config, 'val')
        assert [p.name for p in paths] == [r['file_name'] for r in trained['records']]
        model = RFDETRSmall(device=settings.device)
        native = []
        from PIL import Image
        for i, path in enumerate(paths):
            with Image.open(path) as im:
                detection = model.predict(im.convert('RGB'), threshold=.001)
            native.append(dict(file_name=path.name, predictions=[dict(
                class_id=int(c), class_name=str(n), confidence=float(s), box=list(map(float,b)))
                for c,n,s,b in zip(detection.class_id,detection.data['class_name'],detection.confidence,detection.xyxy)]))
            if (i+1)%100 == 0: print(task, 'native', i+1, flush=True)
        write_json(dest/'native_coco_predictions.json', dict(class_names=model.class_names, records=native,
            note='No semantic class mapping to task labels; class IDs must not be compared numerically.'))
        captured = {}
        class BeforeFit(Exception): pass
        class CaptureTrainer:
            def fit(self, module, datamodule, **kwargs):
                captured['module'] = module
                raise BeforeFit()
        try:
            with patch.object(training, 'build_trainer', return_value=CaptureTrainer()):
                model.train(dataset_dir=str(base), dataset_file='yolo', output_dir=str(dest/'initialization'),
                    epochs=10, batch_size=2, grad_accum_steps=8, num_workers=0,
                    resolution=512, device=settings.device, seed=42, use_ema=False,
                    multi_scale=False, expanded_scales=False, tensorboard=False, wandb=False,
                    mlflow=False, clearml=False, run_test=False, num_sanity_val_steps=0,
                    early_stopping=True, early_stopping_patience=10, early_stopping_use_ema=False,
                    eval_max_dets=300, log_per_class_metrics=True, class_names=names,
                    compute_val_loss=True, save_dataset_grids=False)
        except BeforeFit:
            pass
        model.model.model = captured.pop('module').model.to(settings.device)
        model.model.args.num_classes = len(names)
        model.model.class_names = names
        baseline = predict(model, settings, paths, names, base, 'val')
        assert [(r['file_name'],r['ground_truth']) for r in baseline] == [(r['file_name'],r['ground_truth']) for r in trained['records']]
        write_json(dest/'baseline_records.json', dict(names=names, records=baseline))
        for variant, records in [('before_fit',baseline),('after_fit',trained['records'])]:
            ap = coco_ap(records,names,dest/variant)
            for conf in (.25, trained['confidence']):
                if conf == .25 and trained['confidence'] == .25 and any(r['task']==task and r['variant']==variant for r in rows): continue
                metrics = evaluate(records,names,conf)
                for c in metrics['classes']:
                    rows.append(dict(task=task,variant=variant,confidence=conf,**c,**ap[c['class_id']]))
        del model
        torch.cuda.empty_cache()
    write_csv(out/'comparison.csv',rows)
    write_json(out/'method.json', dict(validation_images=dict(data1=557,data2=131),
        pretrained_sha256=fingerprint(root/'.tools/cache/rfdetr/rf-detr-small.pth'),
        seed=42, resolution=512, prediction_floor=.001, max_detections=300,
        matching_iou=.5, optimizer_steps_before_fit=0,
        baseline='Reconstructed original pre-fit module via intercepted Trainer.fit; not an archived epoch-zero checkpoint.',
        native='COCO predictions retained separately; unsupported task labels have no valid native class-aware metric.',
        after='Original saved best-checkpoint predictions; identical filenames and ground truth asserted.',
        test_used=False))
    print('COMPLETED',out,flush=True)


if __name__ == '__main__':
    main()

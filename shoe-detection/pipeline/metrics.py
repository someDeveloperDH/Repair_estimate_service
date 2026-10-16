"""Fixed-threshold, class-aware, greedy one-to-one detection evaluation.

AP is evaluated separately with Ultralytics. No true-negative boxes are defined.
"""
import math


def iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def match(gt, predictions, conf, threshold=0.5, rule='iou50'):
    """Return matched pairs, unmatched predictions and unmatched ground truths.

    gt: [{'class': int, 'box': [x1,y1,x2,y2]}]; predictions also have confidence.
    Matching is confidence ordered and only accepts equal classes.
    """
    if rule not in ('iou50','any_overlap'):raise ValueError(f'Unknown matching rule: {rule}')
    selected = sorted((p for p in predictions if p['confidence'] >= conf),
                      key=lambda p: p['confidence'], reverse=True)
    used = set()
    pairs, false_positives = [], []
    for p in selected:
        candidates = [(iou(g['box'], p['box']), j) for j, g in enumerate(gt)
                      if j not in used and g['class'] == p['class']]
        overlap, j = max(candidates, default=(0.0, -1))
        accepted=overlap>0 if rule=='any_overlap' else overlap>=threshold
        if j >= 0 and accepted:
            used.add(j)
            pairs.append((j, p, overlap))
        else:
            false_positives.append(p)
    return pairs, false_positives, [j for j in range(len(gt)) if j not in used]


def evaluate(records, names, conf, rule='iou50'):
    counts = [dict(class_id=i, name=n, support=0, images=0, tp=0, fp=0, fn=0)
              for i, n in enumerate(names)]
    rows, size_counts = [], {k: [0, 0] for k in ('under_1pct', '1_to_10pct', 'over_10pct')}
    for record in records:
        gt = record['ground_truth']
        pairs, fps, fns = match(gt, record['predictions'], conf, rule=rule)
        for c in {g['class'] for g in gt}:
            counts[c]['images'] += 1
        matched = {j for j, _, _ in pairs}
        for j, g in enumerate(gt):
            counts[g['class']]['support'] += 1
            b = g['box']
            fraction = (b[2] - b[0]) * (b[3] - b[1]) / (record['width'] * record['height'])
            size = 'under_1pct' if fraction < .01 else ('1_to_10pct' if fraction <= .1 else 'over_10pct')
            size_counts[size][1] += 1
            size_counts[size][0] += int(j in matched)
        for j, p, overlap in pairs:
            counts[gt[j]['class']]['tp'] += 1
        for p in fps:
            counts[p['class']]['fp'] += 1
        for j in fns:
            counts[gt[j]['class']]['fn'] += 1
        rows.append(dict(file_name=record['file_name'], tp=len(pairs), fp=len(fps),
                         fn=len(fns), mean_matched_iou=(sum(x[2] for x in pairs) / len(pairs) if pairs else None)))
    for c in counts:
        c['precision'] = c['tp'] / (c['tp'] + c['fp']) if c['tp'] + c['fp'] else (0.0 if c['support'] else None)
        c['recall'] = c['tp'] / c['support'] if c['support'] else None
        c['f1'] = 2 * c['tp'] / (2 * c['tp'] + c['fp'] + c['fn']) if c['support'] else None
    supported = [c for c in counts if c['support']]
    macro = {key: sum(c[key] for c in supported) / len(supported) if supported else None
             for key in ('precision', 'recall', 'f1')}
    total_tp, total_fp, total_fn = (sum(c[k] for c in counts) for k in ('tp', 'fp', 'fn'))
    micro = dict(precision=total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
                 recall=total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
                 f1=2 * total_tp / (2 * total_tp + total_fp + total_fn) if 2 * total_tp + total_fp + total_fn else 0.0)
    return dict(matching_rule=rule,classes=counts, macro_supported_classes=macro, micro=micro,
                supported_class_ids=[c['class_id'] for c in supported], images=rows,
                size_recall={k: dict(tp=v[0], support=v[1], recall=v[0]/v[1] if v[1] else None)
                             for k, v in size_counts.items()})


def damage_presence(predictions, conf):
    """Independent presence of both damage classes; empty is not proof of health."""
    counts={i:sum(p['class']==i and p['confidence']>=conf for p in predictions) for i in (0,1)}
    state='both' if all(counts.values()) else 'outsole_only' if counts[0] else 'heel_lining_only' if counts[1] else 'no_damage_detected'
    return dict(state=state,outsole_count=counts[0],heel_lining_count=counts[1],
                checked_class_ids=[0,1],detected_class_ids=[i for i in (0,1) if counts[i]])


def any_overlap_ap(records,names,max_det=300):
    """Custom 101-point interpolated AP at positive intersection, not COCO AP.

    Confidence-ranked one-to-one matching uses exactly the same greedy rule as
    operating metrics. Touching edges (zero area), wrong classes and duplicate
    predictions remain false positives. Absent GT classes are N/A.
    """
    ranked={i:[] for i in range(len(names))};support={i:0 for i in range(len(names))}
    for r in records:
        for g in r['ground_truth']:support[g['class']]+=1
        predictions=sorted(r['predictions'],key=lambda p:p['confidence'],reverse=True)[:max_det]
        pairs,_,_=match(r['ground_truth'],predictions,0,rule='any_overlap')
        matched={id(p) for _,p,_ in pairs}
        for p in predictions:ranked[p['class']].append((p['confidence'],int(id(p) in matched)))
    result={}
    for c in range(len(names)):
        if not support[c]:result[c]=dict(ap_any_overlap=None);continue
        tp=0;curve=[]
        for n,(_,correct) in enumerate(sorted(ranked[c],key=lambda x:x[0],reverse=True),1):
            tp+=correct;curve.append((tp/support[c],tp/n))
        best=[0.0]*101
        for recall,precision in curve:
            for step in range(101):
                if recall>=step/100:best[step]=max(best[step],precision)
        result[c]=dict(ap_any_overlap=sum(best)/101)
    return result

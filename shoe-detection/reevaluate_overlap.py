"""Rescore cached validation detections under positive-area overlap matching."""
import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from pipeline.metrics import evaluate,match,any_overlap_ap,damage_presence
from pipeline.run import write_json,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--conf',type=float,default=.25)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plans=[('baseline_full',args.run/'baseline/data2/full_frame'),
           ('baseline_tiled',args.run/'baseline/data2/tiled'),
           ('long30_full',args.run/'data2_long30/full_frame'),
           ('tiles30_full',args.run/'data2_tiles30/full_frame'),
           ('tiles30_tiled',args.run/'data2_tiles30/tiled'),
           ('balanced30_full',args.run/'data2_tiles_balanced30/full_frame'),
           ('balanced30_tiled',args.run/'data2_tiles_balanced30/tiled')]
    rows=[];states=[]
    for name,source in plans:
        saved=json.loads((source/'records.json').read_text(encoding='utf-8'))
        records,names=saved['records'],saved['names']
        custom=evaluate(records,names,args.conf,rule='any_overlap')
        old=evaluate(records,names,args.conf,rule='iou50')
        ap=any_overlap_ap(records,names)
        dest=args.output/name;dest.mkdir()
        presence=[];narratives=[]
        for r in records:
            predicted=damage_presence(r['predictions'],args.conf)
            expected=damage_presence([dict(g,confidence=1) for g in r['ground_truth']],0)
            presence.append(dict(file_name=r['file_name'],predicted=predicted,annotated=expected))
            pairs,fps,fns=match(r['ground_truth'],r['predictions'],args.conf,rule='any_overlap')
            narratives.append(dict(file_name=r['file_name'],matching_rule='any_overlap',
                observation=f"밑창·안감 두 종류의 점수를 함께 검사했습니다. 밑창 후보 {predicted['outsole_count']}개, 안감 후보 {predicted['heel_lining_count']}개입니다. 같은 클래스에서 박스 면적이 조금이라도 겹치는 일대일 매칭 기준으로 정탐 {len(pairs)}개, 오탐 {len(fps)}개, 누락 {len(fns)}개입니다.",
                presence=predicted,matched_pairs=[dict(gt_index=j,prediction=p,iou=overlap) for j,p,overlap in pairs],
                note='Post-inference evaluation summary; not a model thought transcript. No detection is not confirmed normal.'))
        for c,reference in zip(custom['classes'],old['classes']):
            c.update(ap[c['class_id']])
            rows.append(dict(experiment=name,matching_rule='any_overlap',**c,
                             iou50_tp=reference['tp'],iou50_fp=reference['fp'],iou50_fn=reference['fn'],
                             iou50_precision=reference['precision'],iou50_recall=reference['recall']))
        write_json(dest/'metrics.json',dict(confidence=args.conf,matching_rule='any_overlap',
            custom=custom,reference_iou50=old,custom_ap=ap,source_records=str(source/'records.json')))
        write_json(dest/'damage_presence.json',presence)
        with (dest/'image_reports.jsonl').open('w',encoding='utf-8') as stream:
            for line in narratives:stream.write(json.dumps(line,ensure_ascii=False)+'\n')
        states.append(dict(experiment=name,predicted_state_counts=dict(Counter(p['predicted']['state'] for p in presence)),
                           annotated_state_counts=dict(Counter(p['annotated']['state'] for p in presence))))
    write_csv(args.output/'comparison.csv',rows)
    write_json(args.output/'presence_summary.json',states)
    lines=['# 박스 교집합 기준 재평가','',
        '동일한 검증 사진 131장과 기존 예측을 재사용했습니다. 재학습·최종 test 평가는 하지 않았습니다. confidence는 '+str(args.conf)+'입니다.','',
        '**같은 클래스이고 교집합 면적이 0보다 크면 정탐**입니다. 선·점으로만 닿는 박스, 다른 클래스, 같은 정답의 중복 예측은 정탐으로 추가 집계하지 않습니다. 한 예측과 한 정답은 일대일로 대응합니다.','',
        '새 AP(any overlap)는 전체 점수 순위의 101점 보간 AP입니다. COCO AP50·mAP50:95와 다른 사용자 정의 지표이며 이전 표준 mAP와 숫자를 직접 비교하지 않습니다. 점수 변화는 평가 기준 변경의 효과입니다.','',
        '| 실험 | 부위 | IoU≥0.5 정탐/오탐 | 겹침 기준 정탐/오탐 | 겹침 정밀도 | 겹침 검출률 | AP(any overlap) |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['class_id'] not in (0,1):continue
        def pct(key):return 'N/A' if r[key] is None else f"{100*r[key]:.2f}%"
        lines.append(f"| {r['experiment']} | {r['name']} | {r['iou50_tp']}/{r['iou50_fp']} | {r['tp']}/{r['fp']} | {pct('precision')} | {pct('recall')} | {pct('ap_any_overlap')} |")
    lines+=['','## 사진별 두 손상 검사','',
        '밑창·안감 둘 다 / 밑창만 / 안감만 / 두 종류 미검출의 네 상태를 기록합니다. 미검출은 정상 확정이 아닙니다. not_sneakers는 별도 예외 클래스이며 손상 종류에 포함하지 않습니다. 원래 라벨에 정답이 없는 것도 정상 확인과 구분합니다.','',
        '| 실험 | 둘 다 검출 | 밑창만 | 안감만 | 두 종류 미검출 |','|---|---:|---:|---:|---:|']
    for row in states:
        counts=row['predicted_state_counts']
        lines.append('| '+row['experiment']+' | '+' | '.join(str(counts.get(k,0)) for k in ['both','outsole_only','heel_lining_only','no_damage_detected'])+' |')
    lines+=['','각 실험 폴더의 `damage_presence.json`에 사진별 상태, `image_reports.jsonl`에 자연어 평가 기록이 있습니다. 기존 추론 추적기의 단계별 분석·변형 실험과는 별도의 재채점 기록입니다.','']
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(args.output/'report.md')


if __name__=='__main__':main()

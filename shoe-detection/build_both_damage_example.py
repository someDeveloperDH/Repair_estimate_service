"""Render the two-class example from actual completed inference trace records."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/examples/both_damage_predictions_20261016'
index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
assert index['status']=='completed'
traces=[json.loads((OUT/Path(t['report']).parent/'trace.json').read_text(encoding='utf-8')) for t in index['targets']]
assert {d['focus']['class'] for d in traces}=={0,1}
assert len(traces)==2
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
title=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',20)
with Image.open(index['source']) as source:photo=source.convert('RGB')
canvas=Image.new('RGB',(640,760),'#fafaf7')
canvas.paste(photo.resize((640,640)),(0,52))
draw=ImageDraw.Draw(canvas)
draw.text((18,13),'밑창·안감 동시 예측',font=title,fill='#263c35')
draw.text((402,17),f"confidence ≥ {index['confidence']:.2f}",font=small,fill='#4e5c55')
for d in traces:
    p=d['focus'];c=p['class'];color='#d94b2e' if c==0 else '#1266b0'
    box=[p['box'][0]*1.25,p['box'][1]*1.25+52,p['box'][2]*1.25,p['box'][3]*1.25+52]
    draw.rectangle(box,outline=color,width=3)
    label=('밑창 예측' if c==0 else '안감 예측')+f" · {p['confidence']:.3f}"
    bounds=draw.textbbox((0,0),label,font=font)
    width=bounds[2]-bounds[0]+14
    x=max(6,min(634-width,box[0]));y=max(54,box[1]-30)
    draw.rectangle((x,y,x+width,y+29),fill=color)
    draw.text((x+7,y+3),label,font=font,fill='#ffffff')
draw.text((18,706),'빨강·파랑 모두 모델 예측 박스입니다.',font=small,fill='#36463f')
draw.text((18,732),'정답 평가: 안감은 정탐 / 밑창은 라벨 기준 오탐',font=small,fill='#7a493d')
canvas.save(OUT/'both_predictions.png')

lines=['# 밑창과 안감을 동시에 예측한 실제 사진','',
    f"사진: `{Path(index['source']).name}` / 표시 임계값 {index['confidence']:.2f} / 평가 기준 `{index['matching_rule']}`",'',
    '![두 종류의 실제 예측](both_predictions.png)','',
    '빨간색과 파란색 박스는 모두 모델 예측입니다. 현재 검증 사진 중 두 클래스의 정답 라벨이 동시에 등록된 사진은 없습니다. 이 사례는 동시 예측 예시이며, 두 손상을 모두 맞힌 사례로 제시하지 않습니다.','']
for number,d in enumerate(traces,1):
    p=d['focus'];label='뒤꿈치 안감' if p['class']==1 else '밑창'
    layer_scores=' → '.join(f"{l['scores'][p['class']]:.3f}" for l in d['layer_trace'])
    whole=next(x for x in d['perturbations'] if x['id']=='box_full')
    origin='전체 사진' if p['view']=='full_frame' else '왼쪽 위 타일을 확대해 입력한 사진'
    assessment='정답 라벨과 매칭되었습니다.' if d['evaluation']['status']=='true_positive' else '이 위치에는 대응하는 밑창 정답 라벨이 없어 오탐으로 평가되었습니다. 실제 상태와 라벨 누락 여부는 별도 검수가 필요합니다.'
    lines += [f'## {number}. {label} 예측','',
        f"모델은 {origin}에서 이 영역을 `{d['classes'][p['class']]}` 후보로 출력했습니다. 디코더 단계별로 동일 후보 슬롯의 점수는 **{layer_scores}**로 변했고, 최종 점수 {p['confidence']:.3f}가 표시 임계값 {index['confidence']:.2f}를 넘어 결과에 남았습니다.",'',
        f"이 후보의 출처 뷰에서 예측 영역 전체를 흐리게 처리하자, 원래 위치와 겹치는 같은 클래스 예측의 최고 점수는 **{d['probe_baseline']['spatial_best_score']:.3f} → {whole['spatial_best_score']:.3f}**로 감소했습니다. 해당 영역의 시각 정보가 예측에 영향을 준다는 관측이지만, 특정 질감·색상·경계가 원인이라고 확정한 것은 아닙니다.",'',
        assessment,'',f"[항목별 원시 근거와 전체 자연어 기록]({index['targets'][number-1]['report']})",'']
lines += ['## 두 예측의 관계','',
    '한 영역을 밑창으로 예측했다는 이유로 안감 검사를 중단하지 않았습니다. 두 클래스의 점수를 함께 계산하고 여러 후보를 유지했습니다. 따라서 안감 정탐과 밑창 오탐이 같은 사진에서 동시에 나올 수 있습니다. 점수는 보정된 정답 확률이 아닙니다.','',
    '이 글은 실제 후보·단계별 점수·입력 변경 실험을 요약한 개발 기록이며 모델이 직접 생성한 생각을 옮긴 문서는 아닙니다.','']
(OUT/'example_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(OUT/'both_predictions.png')
print(OUT/'example_report.md')

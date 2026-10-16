"""Fresh, frozen-configuration inference with per-image visual/narrative reports.

Reuses one loaded model and one captured forward per view for all detections.
Every displayed detection receives five measured blur comparisons. No LLM,
training, threshold search, or copying of old prediction records is performed.
"""
import argparse
import html
import json
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .inference_trace import Capture,absolute_boxes,find_query,merge_with_lineage
from .metrics import iou,match,evaluate,damage_presence,any_overlap_ap
from .run import ROOT,dataset,images_in,labels_for,write_json,write_csv,fingerprint

DISPLAY={'outsole':'밑창','heel lining':'뒤꿈치 안감','not_sneakers':'운동화 아님',
         'sneakers':'운동화','formal':'구두','boots':'부츠','sandals':'샌들','slippers':'슬리퍼','heels':'하이힐','others':'기타 신발'}
PRESENCE={'both':'밑창·안감 모두 검출','outsole_only':'밑창만 검출','heel_lining_only':'안감만 검출','no_damage_detected':'두 손상 미검출'}


def plans(root):
    return [dict(name='data1_long30',task='data1',tiled=False),
            dict(name='data2_long30',task='data2',tiled=False),
            dict(name='data2_tiles30',task='data2',tiled=True),
            dict(name='data2_tiles_balanced30',task='data2',tiled=True)]


def annotation_image(image,boxes,title,gt=False):
    from PIL import Image,ImageDraw,ImageFont
    font_path=Path('C:/Windows/Fonts/malgun.ttf')
    font=ImageFont.truetype(str(font_path),14) if font_path.exists() else ImageFont.load_default()
    canvas=Image.new('RGB',(image.width,image.height+70),'#fafaf7')
    canvas.paste(image,(0,40));draw=ImageDraw.Draw(canvas)
    draw.text((12,9),title,font=font,fill='#263c35')
    for n,p in enumerate(boxes,1):
        color='#07865e' if gt else '#d64c2e' if p['class']==0 else '#1c6cb0'
        a,b,c,d=p['box'];b+=40;d+=40
        draw.rectangle((a,b,c,d),outline=color,width=2)
        label=('G' if gt else 'P')+str(n)+' '+p['display_name']+(f" {p['confidence']:.3f}" if not gt else '')
        textbox=draw.textbbox((0,0),label,font=font);tw=textbox[2]-textbox[0]+8
        x=max(0,min(image.width-tw,a));y=max(40,b-23)
        draw.rectangle((x,y,x+tw,y+22),fill=color)
        draw.text((x+4,y+1),label,font=font,fill='#ffffff')
    if not boxes:draw.text((12,48),'등록된 정답 없음' if gt else '임계값 이상 예측 없음',font=font,fill='#5c665c')
    draw.text((12,image.height+48),'초록 G = 등록 정답' if gt else '빨강·파랑 P = 모델 예측 / 점수는 보정 확률 아님',font=font,fill='#53604e')
    return canvas


def draw_images(image,displayed,gt,names,out):
    from PIL import Image,ImageDraw,ImageFont
    predictions=[dict(p,display_name=DISPLAY.get(names[p['class']],names[p['class']])) for p in displayed]
    truths=[dict(g,display_name=DISPLAY.get(names[g['class']],names[g['class']])) for g in gt]
    pred=annotation_image(image,predictions,'모델 예측 — P 번호')
    truth=annotation_image(image,truths,'정답 라벨 — G 번호',gt=True)
    pred.save(out/'predictions.png');truth.save(out/'ground_truth.png')
    overlay=pred.copy();draw=ImageDraw.Draw(overlay)
    # Dashed GT borders remain distinguishable from solid predicted borders.
    font_path=Path('C:/Windows/Fonts/malgun.ttf')
    font=ImageFont.truetype(str(font_path),13) if font_path.exists() else ImageFont.load_default()
    for n,g in enumerate(gt,1):
        a,b,c,d=map(int,g['box']);b+=40;d+=40
        for x in range(a,c,10):
            draw.line((x,b,min(c,x+5),b),fill='#07865e',width=2)
            draw.line((x,d,min(c,x+5),d),fill='#07865e',width=2)
        for y in range(b,d,10):
            draw.line((a,y,a,min(d,y+5)),fill='#07865e',width=2)
            draw.line((c,y,c,min(d,y+5)),fill='#07865e',width=2)
        draw.text((max(a,0),min(d+2,image.height+22)),f'G{n}',font=font,fill='#07865e')
    draw.rectangle((0,0,image.width,39),fill='#fafaf7')
    draw.text((12,9),'겹쳐 보기 — 실선 P 예측 / 초록 점선 G 정답',font=font,fill='#263c35')
    overlay.save(out/'overlay.png')
    combined=Image.new('RGB',(image.width*3,image.height+70),'#fafaf7')
    combined.paste(pred,(0,0));combined.paste(truth,(image.width,0));combined.paste(overlay,(image.width*2,0))
    combined.save(out/'comparison.jpg',quality=92)


def render_report(folder,title,paragraphs):
    md=[f'# {title}','','![예측·정답·겹침 비교](comparison.jpg)','']
    md.extend(paragraphs)
    (folder/'report.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    sections=[]
    for line in paragraphs:
        if line.startswith('## '):sections.append('<h2>'+html.escape(line[3:])+'</h2>')
        else:sections.append('<p>'+html.escape(line).replace('\n','<br>')+'</p>')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{font:15px/1.9 system-ui,"Malgun Gothic",sans-serif;max-width:1280px;margin:30px auto;padding:0 20px;color:#26352e;background:#fbfcf8}img{max-width:100%;height:auto}h1{font-size:25px}h2{font-size:20px;margin-top:36px}p{white-space:normal}a{color:#28664b}</style><h1>'+html.escape(title)+'</h1><p><a href="../../../index.html">전체 목록</a> · <a href="predictions.png">예측 이미지</a> · <a href="ground_truth.png">정답 이미지</a> · <a href="overlay.png">겹쳐 보기</a> · <a href="trace.json">원시 근거</a></p><img src="comparison.jpg" alt="왼쪽 예측, 가운데 정답, 오른쪽 겹침 비교">'+''.join(sections)+'</html>'
    (folder/'report.html').write_text(page,encoding='utf-8')


def analyze_image(model,capture,path,task,base,names,tiled,conf,folder):
    import torch
    from PIL import Image,ImageFilter
    from run_improvement_experiments import windows
    started=time.monotonic();folder.mkdir(parents=True,exist_ok=False)
    with Image.open(path) as source:image=source.convert('RGB')
    shutil.copy2(path,folder/'original.jpg')
    views={};candidates=[]
    spans=[(0,0,image.width,image.height)]+(windows(image.width,image.height) if tiled else [])
    for n,window in enumerate(spans):
        view='full_frame' if n==0 else f'tile_{n}'
        source=image.crop(window);capture.snapshot={}
        with torch.inference_mode():detections=model.predict(source,threshold=.001)
        snap=capture.snapshot
        views[view]=dict(window=list(window),image=source,snapshot=snap)
        for j,(box,c,s) in enumerate(zip(detections.xyxy,detections.class_id,detections.confidence)):
            c=int(c)
            if not 0<=c<len(names):continue
            a,b,right,bottom=map(float,box);x,y=window[:2]
            candidates.append(dict(id=f'{view}/p{j}',view=view,**{'class':c},confidence=float(s),
                local_box=[a,b,right,bottom],box=[a+x,b+y,right+x,bottom+y]))
    kept,lineage=merge_with_lineage(candidates)
    displayed=[p for p in kept if p['confidence']>=conf]
    # GT is loaded only after inference and never passed to the model.
    split=path.parent.name
    gt=labels_for(path,base,split,image.width,image.height,len(names))
    pairs,fp,fn=match(gt,kept,conf,rule='any_overlap')
    matched={p['id']:(j,overlap) for j,p,overlap in pairs}
    draw_images(image,displayed,gt,names,folder)
    presence=damage_presence(kept,conf) if task=='data2' else None
    paragraphs=[f'평가 기준: 같은 클래스이며 교집합 면적이 0보다 큰 일대일 매칭. 표시 임계값 {conf:.2f}. 정탐 {len(pairs)}개, 오탐 {len(fp)}개, 누락 {len(fn)}개.',
        '이 설명은 측정된 후보·단계별 점수·입력 변경 실험의 자연어 요약입니다. 모델이 생성한 생각이나 확정된 원인 설명은 아닙니다.']
    if presence:paragraphs.append(f"밑창과 안감을 동시에 검사했습니다. 결과: {PRESENCE[presence['state']]}. 밑창 후보 {presence['outsole_count']}개, 안감 후보 {presence['heel_lining_count']}개입니다. 미검출은 정상 확정이 아닙니다.")
    paragraphs.append(f'원본을 {len(spans)}개 뷰로 입력했습니다. 임계값 적용 전 후보 {len(candidates)}개를 좌표 복원·중복 병합해 {len(kept)}개로 정리했습니다. 정답 라벨은 추론이 끝난 뒤 평가에만 사용했습니다.')
    targets=[];probe_cache={}
    for number,p in enumerate(displayed,1):
        view=views[p['view']];snap=view['snapshot'];source=view['image'];w,h=source.size
        final=snap['layers'][-1];raw_boxes=absolute_boxes(final['boxes'][0].tolist(),w,h)
        scores=final['logits'][0].sigmoid()
        query=find_query(raw_boxes,scores[:,p['class']].tolist(),p['local_box'],p['confidence'])
        layers=[dict(layer=i+1,scores=l['logits'][0,query].sigmoid().tolist(),
                     box=absolute_boxes([l['boxes'][0,query].tolist()],w,h)[0]) for i,l in enumerate(snap['layers'])]
        baseline=max((q['confidence'] for q in candidates if q['view']==p['view'] and q['class']==p['class'] and iou(q['local_box'],p['local_box'])>=.3),default=0)
        a,b,c,d=p['local_box']
        regions=[('whole','예측 영역 전체',(a,b,c,d)),('upper','예측 영역 상부 60%',(a,b,c,b+(d-b)*.6)),
                 ('lower','예측 영역 하부 35%',(a,d-(d-b)*.35,c,d)),
                 ('context','입력 뷰 위쪽 중앙',(w*.35,h*.08,w*.65,h*.3)),
                 ('control','입력 뷰 좌상단 대조 영역',(0,0,w*.22,h*.28))]
        probes=[];blurred=source.filter(ImageFilter.GaussianBlur(12))
        for key,label,region in regions:
            rect=tuple(map(int,region));cache_key=(p['view'],rect)
            if cache_key not in probe_cache:
                changed=source.copy();changed.paste(blurred.crop(rect),rect);capture.snapshot={}
                with torch.inference_mode():result=model.predict(changed,threshold=.001)
                raw=capture.snapshot['layers'][-1]
                probe_cache[cache_key]=(raw,result)
            raw,result=probe_cache[cache_key]
            fixed=float(raw['logits'][0,query,p['class']].sigmoid())
            box=absolute_boxes([raw['boxes'][0,query].tolist()],w,h)[0]
            spatial=max((float(s) for box2,c2,s in zip(result.xyxy,result.class_id,result.confidence)
                         if int(c2)==p['class'] and iou(list(map(float,box2)),p['local_box'])>=.3),default=0)
            probes.append(dict(id=key,label=label,window=list(rect),fixed_query_score=fixed,
                fixed_query_box_iou=iou(box,p['local_box']),spatial_best_score=spatial,
                spatial_score_drop=baseline-spatial))
        label=DISPLAY.get(names[p['class']],names[p['class']])
        stage=' → '.join(f"{l['scores'][p['class']]:.3f}" for l in layers)
        origin='전체 사진' if p['view']=='full_frame' else f"원본 좌표 {view['window']}의 타일을 확대해 입력한 뷰"
        evaluation=(f"정답 G{matched[p['id']][0]+1}과 매칭된 정탐입니다. IoU는 {matched[p['id']][1]:.4f}입니다." if p['id'] in matched else '같은 클래스의 아직 매칭되지 않은 정답과 겹치지 않아 라벨 기준 오탐입니다. 정답 누락 여부와 실제 손상 종류는 별도 검수가 필요합니다.')
        whole=probes[0];switches=[q for q in probes if q['fixed_query_score']<conf<=q['spatial_best_score']]
        paragraphs += [f'## P{number}. {label} 예측',
            f"{origin}의 후보 슬롯 {query}에서 이 박스를 출력했습니다. 디코더 단계별로 동일 슬롯 점수는 {stage}였으며, 최종 점수 {p['confidence']:.3f}가 표시 기준 {conf:.2f}를 넘어 남았습니다. 좌표는 {[round(x,1) for x in p['box']]}입니다.",
            f"이 출처 뷰에서 예측 영역 전체를 흐리게 처리하자 같은 위치의 같은 클래스 예측 최고 점수는 {baseline:.3f} → {whole['spatial_best_score']:.3f}로 변했습니다. 입력 영역의 영향에 대한 실험이며 질감·색상·형태 중 특정 원인을 확정하지 않습니다.",evaluation]
        if switches:paragraphs.append('후보 슬롯의 점수가 내려가도 다른 후보가 같은 위치를 계속 검출한 실험: '+', '.join(q['label'] for q in switches)+'. 고정 슬롯 점수 감소만으로 검출이 사라졌다고 해석하지 않았습니다.')
        targets.append(dict(number=number,prediction=p,query=query,decoder_layers=layers,
                            evaluation='true_positive' if p['id'] in matched else 'false_positive_by_annotations',
                            matched_gt_index=matched[p['id']][0] if p['id'] in matched else None,
                            probe_baseline=baseline,probes=probes))
    if not displayed:paragraphs+=['## 임계값 이상 검출 없음','모델은 모든 과제 클래스 점수를 계산했지만 표시 기준을 넘는 후보가 남지 않았습니다. 이것만으로 상품에 손상이 없다고 확정하지 않습니다.']
    if fn:paragraphs+=['## 누락된 정답','예측과 매칭되지 않은 등록 정답: '+', '.join(f"G{j+1} {DISPLAY.get(names[gt[j]['class']],names[gt[j]['class']])}" for j in fn)+'.']
    paragraphs+=['## 해석 범위','클래스는 부위·상품 종류입니다. 마모·오염·벌어짐을 자동 구분한 것은 아닙니다. 서로 다른 박스가 같은 실제 손상의 조각일 수 있고, 다른 사진의 동일 손상 연결은 별도 판단입니다. 같은 슬롯이 단계마다 같은 개념을 뜻한다고 단정하지 않습니다.']
    record=dict(file_name=path.name,path=str(path),width=image.width,height=image.height,ground_truth=gt,predictions=kept)
    trace=dict(file_name=path.name,classes=names,confidence=conf,matching_rule='any_overlap',damage_presence=presence,
        views=[dict(id=k,window=v['window'],input_shape=list(v['snapshot']['input'].shape),feature_shapes=v['snapshot'].get('feature_shapes')) for k,v in views.items()],
        candidates=candidates,merge_decisions=lineage,targets=targets,unmatched_gt_indices=fn,ground_truth=gt,
        note='Measured computation and blur experiments, not a hidden reasoning transcript.')
    write_json(folder/'trace.json',trace);render_report(folder,path.name,paragraphs)
    return record,dict(file_name=path.name,report=f'images/{path.stem}/report.html',predictions=len(displayed),
                       tp=len(pairs),fp=len(fp),fn=len(fn),presence=presence,seconds=round(time.monotonic()-started,3))


def write_index(out,status,reports):
    rows=[]
    for model,images in reports.items():
        for r in images:
            label=PRESENCE.get((r.get('presence') or {}).get('state'),'신발 종류 검사')
            link=f"{model}/{r['report']}"
            rows.append('<tr><td>'+html.escape(model)+'</td><td><a href="'+html.escape(link)+'">'+html.escape(r['file_name'])+'</a></td><td>'+html.escape(label)+f"</td><td>{r['predictions']}</td><td>{r['tp']}/{r['fp']}/{r['fn']}</td></tr>")
    content='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>모델별 이미지·자연어 재평가</title><style>body{font:14px/1.8 system-ui,"Malgun Gothic",sans-serif;padding:20px;max-width:1300px;margin:auto;color:#26352e;background:#fbfcf8}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #dce4d9}a{color:#28664b}.wrap{overflow-x:auto}</style><h1>모델별 이미지·자연어 재평가</h1><p>'+html.escape(f"분할: {status['split']} / 상태: {status['status']} / 완료 {status['completed_images']} / {status['planned_images']}건 / confidence {status['confidence']} / 같은 클래스의 양의 면적 교집합 매칭")+'</p><p>사진명을 누르면 예측·정답 이미지와 모든 검출의 설명을 볼 수 있습니다. 검출 수는 실제 손상 개수와 다를 수 있습니다.</p><div class="wrap"><table><thead><tr><th>모델</th><th>사진·보고서</th><th>두 손상 검사</th><th>예측 수</th><th>정탐/오탐/누락</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></html>'
    (out/'index.html').write_text(content,encoding='utf-8')


def run(args):
    from .rf_backend import setup,coco_ap
    setup(SimpleNamespace(device=args.device,model='rf-detr-small',threads=4))
    from rfdetr import RFDETR
    import torch
    selected=plans(args.models_root)
    if args.models:selected=[p for p in selected if p['name'] in args.models]
    inventory=[]
    for plan in selected:
        base,names,config=dataset(SimpleNamespace(data=ROOT/f"datasets/{plan['task']}/data.yaml"))
        paths=images_in(base,config,args.split)
        if args.limit:paths=paths[:args.limit]
        training=json.loads((args.models_root/plan['name']/'train/training.json').read_text(encoding='utf-8'))
        weight=Path(training['best'])
        inventory.append(dict(**plan,base=base,names=names,paths=paths,weights=weight,best_epoch=training['best_epoch']))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    frozen=dict(split=args.split,confidence=args.conf,matching_rule='any_overlap',blur_radius=12,
        source='fresh model inference, not cached detections',created=datetime.now().astimezone().isoformat(),
        exploratory_trace_on_test=args.split=='test',
        models=[dict(name=p['name'],task=p['task'],tiled=p['tiled'],weights=str(p['weights']),weights_sha256=fingerprint(p['weights']),
                     best_epoch=p['best_epoch'],image_count=len(p['paths'])) for p in inventory])
    write_json(out/'frozen_config.json',frozen)
    status=dict(status='running',split=args.split,confidence=args.conf,started=frozen['created'],completed_images=0,
                planned_images=sum(len(p['paths']) for p in inventory),models={},errors=[])
    reports={};summary=[];start=time.monotonic()
    write_json(out/'status.json',status);write_index(out,status,reports)
    for plan in inventory:
        name=plan['name'];dest=out/name;dest.mkdir();records=[];reports[name]=[]
        status['models'][name]=dict(status='running',completed=0,total=len(plan['paths']))
        model=RFDETR.from_checkpoint(str(plan['weights']),device='cuda:0' if args.device=='0' else args.device)
        if list(model.class_names)!=plan['names']:raise ValueError('Checkpoint task class mismatch')
        capture=Capture(model)
        try:
            for path in plan['paths']:
                try:
                    record,report=analyze_image(model,capture,path,plan['task'],plan['base'],plan['names'],plan['tiled'],args.conf,dest/'images'/path.stem)
                    records.append(record);reports[name].append(report)
                    with (dest/'records.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    with (dest/'image_summary.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report,ensure_ascii=False)+'\n')
                except Exception as exc:
                    error=dict(model=name,file_name=path.name,error=str(exc));status['errors'].append(error)
                    write_json(dest/'images'/path.stem/'error.json',error)
                status['completed_images']+=1;status['models'][name]['completed']+=1
                elapsed=time.monotonic()-start
                status['elapsed_seconds']=round(elapsed,1)
                status['estimated_remaining_seconds']=round(elapsed/status['completed_images']*(status['planned_images']-status['completed_images']))
                write_json(out/'status.json',status)
                if status['completed_images']%10==0:
                    write_index(out,status,reports)
                    print(f"PROGRESS {status['completed_images']}/{status['planned_images']} errors={len(status['errors'])}",flush=True)
        finally:capture.close()
        metrics=evaluate(records,plan['names'],args.conf,rule='any_overlap')
        reference=evaluate(records,plan['names'],args.conf,rule='iou50')
        custom=any_overlap_ap(records,plan['names']);standard=coco_ap(records,plan['names'],dest/'standard_coco')
        for c in metrics['classes']:
            c.update(custom[c['class_id']]);c.update(standard[c['class_id']]);summary.append(dict(model=name,**c))
        write_json(dest/'metrics.json',dict(processed_images=len(records),planned_images=len(plan['paths']),matching_rule='any_overlap',
            confidence=args.conf,selected=metrics,reference_iou50=reference))
        write_json(dest/'records.json',dict(task=plan['task'],split=args.split,names=plan['names'],confidence=args.conf,matching_rule='any_overlap',records=records))
        status['models'][name]['status']='completed' if len(records)==len(plan['paths']) else 'completed_with_failures'
        write_csv(out/'comparison.csv',summary);write_json(out/'status.json',status);write_index(out,status,reports)
        del capture,model;torch.cuda.empty_cache()
    status['status']='completed' if not status['errors'] else 'completed_with_failures'
    status['finished']=datetime.now().astimezone().isoformat();write_json(out/'status.json',status);write_index(out,status,reports)
    (out/'README.md').write_text('# 모델별 재평가 결과\n\n`index.html`을 열고 사진명을 누르면 예측·정답·겹침 비교와 검출별 자연어 기록을 볼 수 있습니다.\n\n설정은 frozen_config.json, 상태는 status.json, 비교 지표는 comparison.csv입니다. 원시 추적은 각 사진 폴더의 trace.json에 있습니다. 기록은 실측 요약이며 모델의 생각을 읽은 결과가 아닙니다.\n',encoding='utf-8')
    if status['errors']:raise RuntimeError('Some images failed; inspect status.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root',type=Path,default=ROOT/'artifacts/no_new_data_20261016_014558')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],required=True)
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--device',default='0')
    parser.add_argument('--models',nargs='+',choices=[p['name'] for p in plans(None)])
    parser.add_argument('--limit',type=int,default=0,help='Preflight sample only; 0 processes the complete split')
    args=parser.parse_args()
    if not 0<=args.conf<=1 or args.limit<0:parser.error('Invalid confidence or image limit')
    try:run(args)
    except Exception as exc:
        status_file=args.output.resolve()/'status.json'
        if status_file.exists():
            current=json.loads(status_file.read_text(encoding='utf-8'))
            current.update(status='failed',fatal_error=str(exc),finished=datetime.now().astimezone().isoformat())
            write_json(status_file,current)
        raise


if __name__=='__main__':main()

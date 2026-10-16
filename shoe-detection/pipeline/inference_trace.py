"""Development-only RF-DETR evidence logger, not a generated thought transcript.

Run: python -m pipeline.inference_trace --source ... --weights ... --output ...
No model, labels, or service inference code is modified. Reports use templates
and measured tensors; no external language-model service or image upload.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .metrics import iou, match, damage_presence
from .run import ROOT, dataset, labels_for, write_json


def merge_with_lineage(predictions, threshold=.5, max_det=300):
    kept, decisions = [], []
    for p in sorted(predictions, key=lambda p: p['confidence'], reverse=True):
        blocker = next((q for q in kept if q['class'] == p['class'] and iou(q['box'], p['box']) > threshold), None)
        if blocker:
            decisions.append(dict(candidate=p['id'], action='suppressed', by=blocker['id'], overlap=iou(p['box'], blocker['box'])))
        elif len(kept) >= max_det:
            decisions.append(dict(candidate=p['id'], action='max_det_excluded'))
        else:
            kept.append(p)
            decisions.append(dict(candidate=p['id'], action='kept'))
    return kept, decisions


def absolute_boxes(normalized, width, height):
    result=[]
    for cx,cy,w,h in normalized:
        result.append([max(0,min(width,(cx-w/2)*width)),max(0,min(height,(cy-h/2)*height)),
                       max(0,min(width,(cx+w/2)*width)),max(0,min(height,(cy+h/2)*height))])
    return result


def find_query(raw_boxes, scores, box, score):
    """Link postprocessed detection to its actual raw query, never by attention."""
    index=min(range(len(raw_boxes)),key=lambda q:sum(abs(a-b) for a,b in zip(raw_boxes[q],box))+100*abs(scores[q]-score))
    error=max(abs(a-b) for a,b in zip(raw_boxes[index],box))
    if error>.05 or abs(scores[index]-score)>1e-4:
        raise ValueError(f'Cannot reliably link detection to raw query: box error={error}')
    return index


def describe_structure(value):
    if hasattr(value,'shape'):
        return dict(shape=list(value.shape),dtype=str(value.dtype))
    if hasattr(value,'tensors'):
        return dict(tensors=describe_structure(value.tensors))
    if isinstance(value,(tuple,list)):
        return [describe_structure(v) for v in value[:8]]
    if isinstance(value,dict):
        return {str(k):describe_structure(v) for k,v in value.items()}
    return type(value).__name__


def strongest_spatial_change(perturbations):
    return max(perturbations,key=lambda p:p['spatial_score_drop'])


class Capture:
    def __init__(self,detector):
        self.detector=detector
        self.module=detector.model.model
        self.snapshot={}
        self.handles=[self.module.register_forward_pre_hook(self.before),
                      self.module.register_forward_hook(self.after)]
        if hasattr(self.module,'backbone'):
            self.handles.append(self.module.backbone.register_forward_hook(self.features))

    def before(self,module,arguments):
        tensor=arguments[0]
        if hasattr(tensor,'tensors'): tensor=tensor.tensors
        self.snapshot['input']=tensor.detach().cpu()

    def features(self,module,arguments,output):
        self.snapshot['feature_shapes']=describe_structure(output)

    def after(self,module,arguments,output):
        if not isinstance(output,dict) or 'pred_logits' not in output:
            raise ValueError('Trace requires unoptimized RF-DETR detection output dictionary')
        layers=list(output.get('aux_outputs',[]))+[output]
        self.snapshot['layers']=[dict(logits=layer['pred_logits'].detach().cpu(),
                                     boxes=layer['pred_boxes'].detach().cpu()) for layer in layers]
        encoder=output.get('enc_outputs')
        if encoder: self.snapshot['encoder_shapes']=describe_structure(encoder)

    def close(self):
        for handle in self.handles: handle.remove()


class TraceEvents:
    def __init__(self,out):
        self.path=out/'events.jsonl';self.events=[]

    def add(self,stage,message,evidence=None,kind='observation'):
        event=dict(id=f'E{len(self.events)+1:03}',timestamp=datetime.now().astimezone().isoformat(),
                   stage=stage,kind=kind,message=message,evidence=evidence or {})
        self.events.append(event)
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        print(f"[{event['id']}] {message}",flush=True)
        return event['id']


def trace_image(args,model=None):
    import numpy as np
    from PIL import Image, ImageFilter, ImageDraw
    from .rf_backend import setup
    from run_improvement_experiments import windows
    setup(SimpleNamespace(device=args.device,model='rf-detr-small',threads=4))
    from rfdetr import RFDETR
    import torch
    import rfdetr

    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'inputs').mkdir();(out/'experiments').mkdir()
    events=TraceEvents(out)
    base,names,_=dataset(SimpleNamespace(data=ROOT/f'datasets/{args.task}/data.yaml'))
    source=args.source.resolve()
    if source.parent==(base/'images/test').resolve():
        raise ValueError('Exploratory debugging of the held-out test split is disabled')
    with Image.open(source) as original:
        im=original.convert('RGB')
    im.save(out/'original.png')
    events.add('input',f'입력 사진 {source.name}, 크기 {im.width}×{im.height}를 읽었습니다. 정답 라벨은 모델 입력에 포함하지 않습니다.')
    device='cuda:0' if args.device=='0' else args.device
    if model is None:model=RFDETR.from_checkpoint(str(args.weights.resolve()),device=device)
    if list(model.class_names)!=names: raise ValueError('Checkpoint class names do not match task')
    capture=Capture(model)
    views=[];candidates=[]
    spans=[(0,0,im.width,im.height)]+(windows(im.width,im.height) if args.tiled else [])
    events.add('views',f'추론 입력을 {len(spans)}개 뷰로 구성했습니다. 각 뷰는 모델 입력 크기로 변환됩니다.',dict(windows=spans))
    try:
        for index,window in enumerate(spans):
            view_id='full_frame' if index==0 else f'tile_{index}'
            image=im.crop(window);capture.snapshot={}
            with torch.inference_mode(): result=model.predict(image,threshold=.001)
            snap=capture.snapshot
            tensor=snap['input'][0]
            means=torch.as_tensor(model.means).reshape(-1,1,1)
            stds=torch.as_tensor(model.stds).reshape(-1,1,1)
            pixels=((tensor*stds+means).clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
            Image.fromarray(pixels).save(out/'inputs'/f'{view_id}_actual_input.png')
            final=snap['layers'][-1]
            scores=final['logits'][0].sigmoid().tolist()
            raw_boxes=absolute_boxes(final['boxes'][0].tolist(),image.width,image.height)
            raw_queries=[dict(query=q,scores=scores[q],box=raw_boxes[q]) for q in range(len(scores))]
            view=dict(id=view_id,window=list(window),source_size=list(image.size),
                input_shape=list(snap['input'].shape),normalized_min=float(tensor.min()),
                normalized_max=float(tensor.max()),feature_shapes=snap.get('feature_shapes'),
                encoder_shapes=snap.get('encoder_shapes'),raw_queries=raw_queries,
                layers=[dict(logits=l['logits'][0].tolist(),boxes=l['boxes'][0].tolist()) for l in snap['layers']])
            views.append(view)
            count=0
            for n,(box,c,confidence) in enumerate(zip(result.xyxy,result.class_id,result.confidence)):
                c=int(c)
                if not 0<=c<len(names):continue
                local=list(map(float,box));confidence=float(confidence)
                query=find_query(raw_boxes,[s[c] for s in scores],local,confidence)
                x,y=window[:2]
                candidates.append(dict(id=f'{view_id}/q{query}/c{c}',view=view_id,query=query,
                    **{'class':c},confidence=confidence,local_box=local,
                    box=[local[0]+x,local[1]+y,local[2]+x,local[3]+y]))
                count+=1
            events.add('model_output',f'{view_id}: 입력 텐서 {list(snap["input"].shape)}, 원시 후보 슬롯 {len(scores)}개, 기록된 디코더 단계 {len(snap["layers"])}개입니다.',
                dict(view=view_id,postprocessed_candidates=count,feature_shapes=view['feature_shapes']))
        kept,decisions=merge_with_lineage(candidates)
        displayed=[p for p in kept if p['confidence']>=args.conf]
        presence=damage_presence(displayed,args.conf) if args.task=='data2' else None
        if presence:events.add('damage_presence',f'밑창과 안감 두 클래스를 함께 검사했습니다. 밑창 후보 {presence["outsole_count"]}개, 안감 후보 {presence["heel_lining_count"]}개이며 상태는 {presence["state"]}입니다. 미검출은 정상 확정을 뜻하지 않습니다.',presence)
        focus_candidates=[p for p in displayed if p['class']==args.focus_class]
        if not focus_candidates:raise ValueError('No target-class prediction above requested confidence')
        focus=focus_candidates[args.focus_rank]
        origin=next(v for v in views if v['id']==focus['view'])
        siblings=[p for p in candidates if p['class']==focus['class'] and p['id']!=focus['id'] and iou(p['box'],focus['box'])>.5]
        events.add('merge',f'뷰별 후보 {len(candidates)}개를 좌표 복원·중복 병합해 {len(kept)}개를 남겼고, 표시 임계값 {args.conf:.2f} 이상은 {len(displayed)}개입니다.',
            dict(nms_iou=.5,max_det=300,prediction_floor=.001))
        events.add('focus',f'선택한 {names[focus["class"]]} 예측은 {focus["view"]}의 query 슬롯 {focus["query"]}에서 나왔으며 최종 점수는 {focus["confidence"]:.4f}입니다.',
            dict(candidate=focus,overlapping_candidates=[p['id'] for p in siblings]))
        layer_trace=[]
        for n,layer in enumerate(origin['layers']):
            logits=torch.tensor(layer['logits'][focus['query']]);values=logits.sigmoid().tolist()
            box=absolute_boxes([layer['boxes'][focus['query']]],*origin['source_size'])[0]
            layer_trace.append(dict(layer=n+1,scores=values,box=box))
            events.add('decoder_layer',f'동일 query 슬롯의 디코더 {n+1}단계에서 {names[focus["class"]]} 점수는 {values[focus["class"]]:.4f}였습니다. 중간 단계는 실제 표시 판정이 아닙니다.',dict(layer=n+1,scores=values,local_box=box))
        gt=[];evaluation=None
        if source.parent==(base/'images/val').resolve():
            gt=labels_for(source,base,'val',im.width,im.height,len(names))
            pairs,fps,_=match(gt,kept,args.conf,rule=args.match_rule)
            matched=next((g for g,p,_ in pairs if p['id']==focus['id']),None)
            evaluation=dict(status='true_positive' if matched is not None else 'false_positive_by_annotations',matching_rule=args.match_rule,
                max_same_class_iou=max((iou(g['box'],focus['box']) for g in gt if g['class']==focus['class']),default=0),
                ground_truth=gt)
            events.add('evaluation',f'추론 후 정답과 비교한 결과는 {evaluation["status"]}입니다. 같은 클래스 정답과 최대 IoU는 {evaluation["max_same_class_iou"]:.4f}입니다.',evaluation)

        # Freeze the selected source view, query slot, class and reference box.
        # Also track spatially corresponding postprocessed detections to expose
        # query movement/switching instead of treating every global max as it.
        focus_image=im.crop(origin['window'])
        def probe(image):
            capture.snapshot={}
            with torch.inference_mode(): result=model.predict(image,threshold=.001)
            raw=capture.snapshot['layers'][-1]
            q=focus['query'];score=float(raw['logits'][0,q,focus['class']].sigmoid())
            box=absolute_boxes([raw['boxes'][0,q].tolist()],*image.size)[0]
            spatial=[float(s) for b,c,s in zip(result.xyxy,result.class_id,result.confidence)
                     if int(c)==focus['class'] and iou(list(map(float,b)),focus['local_box'])>=.3]
            return dict(fixed_query_score=score,fixed_query_box_iou=iou(box,focus['local_box']),
                        spatial_best_score=max(spatial,default=0),spatial_match_found=bool(spatial))
        baseline=probe(focus_image)
        if abs(baseline['fixed_query_score']-focus['confidence'])>1e-4:
            raise RuntimeError('Focus score could not be reproduced')
        a,b,c,d=focus['local_box'];w,h=focus_image.size
        regions=[('box_full','예측 박스 전체 흐림',(a,b,c,d)),
                 ('box_upper','예측 박스 상부 60% 흐림',(a,b,c,b+(d-b)*.6)),
                 ('box_lower','예측 박스 하부 35% 흐림',(a,d-(d-b)*.35,c,d)),
                 ('upper_context','사진 위쪽 중앙 흐림',(w*.35,h*.08,w*.65,h*.3)),
                 ('corner_control','좌상단 대조 영역 흐림',(0,0,w*.22,h*.28))]
        blurred=focus_image.filter(ImageFilter.GaussianBlur(12))
        perturbations=[]
        for key,label,region in regions:
            rect=tuple(map(int,region));changed=focus_image.copy();changed.paste(blurred.crop(rect),rect)
            changed.save(out/'experiments'/f'{key}.png')
            value=probe(changed)
            entry=dict(id=key,label=label,region=list(rect),**value,
                       score_drop=baseline['fixed_query_score']-value['fixed_query_score'],
                       spatial_score_drop=baseline['spatial_best_score']-value['spatial_best_score'])
            perturbations.append(entry)
            events.add('perturbation',f'{label}: 동일 query 점수 {value["fixed_query_score"]:.4f}, 원본 대비 감소량 {entry["score_drop"]:+.4f}, 원래 박스와 겹치는 예측의 최고 점수 {value["spatial_best_score"]:.4f}입니다.',entry)
            if value['fixed_query_score']<args.conf<=value['spatial_best_score']:
                events.add('query_switch',f'「{label}」에서 선택 슬롯 점수는 표시 기준 아래로 떨어졌지만, 다른 후보가 같은 위치를 기준 이상 점수로 검출했습니다. 선택 슬롯 점수 감소를 해당 위치 검출의 소실로 해석할 수 없습니다.',dict(experiment=key,threshold=args.conf,**value))

        patch=args.patch;stride=args.stride
        heat=np.zeros((h,w),dtype=np.float32);spatial_heat=np.zeros_like(heat);coverage=np.zeros_like(heat);grid=[]
        if args.occlusion:
            locations=[(x,y,min(x+patch,w),min(y+patch,h)) for y in range(0,h,stride) for x in range(0,w,stride)]
            for index,rect in enumerate(locations):
                changed=focus_image.copy();changed.paste(blurred.crop(rect),rect)
                value=probe(changed);drop=baseline['fixed_query_score']-value['fixed_query_score']
                spatial_drop=baseline['spatial_best_score']-value['spatial_best_score']
                x,y,right,bottom=rect;heat[y:bottom,x:right]+=drop;coverage[y:bottom,x:right]+=1
                spatial_heat[y:bottom,x:right]+=spatial_drop
                grid.append(dict(window=list(rect),score_drop=drop,spatial_score_drop=spatial_drop,**value))
                if (index+1)%16==0:print(f'Occlusion {index+1}/{len(locations)}',flush=True)
            heat/=np.maximum(coverage,1);spatial_heat/=np.maximum(coverage,1)
            np.save(out/'occlusion_score_drop.npy',heat);np.save(out/'occlusion_spatial_score_drop.npy',spatial_heat)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            limit=max(float(np.abs(heat).max()),float(np.abs(spatial_heat).max()),1e-6)
            fig,axes=plt.subplots(1,3,figsize=(15,4.8))
            axes[0].imshow(focus_image);axes[0].set_title('Selected source view')
            axes[1].imshow(focus_image);plot=axes[1].imshow(heat,cmap='RdBu_r',vmin=-limit,vmax=limit,alpha=.65)
            axes[1].set_title('Blur sensitivity: fixed-query score drop')
            axes[2].imshow(focus_image);plot=axes[2].imshow(spatial_heat,cmap='RdBu_r',vmin=-limit,vmax=limit,alpha=.65)
            axes[2].set_title('Blur sensitivity: spatial score drop')
            for ax in axes:ax.axis('off')
            fig.colorbar(plot,ax=axes[2],fraction=.046,pad=.04,label='Original score - perturbed score')
            fig.tight_layout();fig.savefig(out/'occlusion.png',dpi=160);plt.close(fig)
            events.add('occlusion',f'선택된 원본 뷰에서 {len(locations)}개 영역을 차례로 흐리게 처리해 동일 query 점수 변화를 기록했습니다. 이는 민감도 실험이며 모델의 생각이나 확정된 원인은 아닙니다.',dict(patch=patch,stride=stride,blur_radius=12))
        strongest=strongest_spatial_change(perturbations)
        explanation=(f'이번에 비교한 영역 중 원래 위치와 겹치는 {names[focus["class"]]} 검출 점수가 가장 크게 감소한 처리는 「{strongest["label"]}」입니다. 위치 기준 최고 점수는 {baseline["spatial_best_score"]:.4f}에서 {strongest["spatial_best_score"]:.4f}로 변했습니다. 해당 영역의 시각 정보에 대한 민감성이 관측됐지만, 질감·색상·형태 중 무엇 때문인지는 이 실험만으로 분리할 수 없습니다.'
                     if strongest['spatial_score_drop']>0 else '이번에 비교한 영역 흐림 실험에서는 같은 위치의 검출 최고 점수 감소가 관측되지 않았습니다. 이 실험만으로 의심한 영역에 대한 의존을 확인할 수 없습니다.')
        events.add('hypothesis',explanation,dict(experiment=strongest['id'],spatial_score_drop=strongest['spatial_score_drop']),kind='hypothesis')
        overlay=im.copy();draw=ImageDraw.Draw(overlay)
        for g in gt:draw.rectangle(g['box'],outline='#05846b' if g['class']==0 else '#2984d5',width=2)
        draw.rectangle(focus['box'],outline='#eb622f',width=3);overlay.save(out/'selected_prediction.png')
        trace=dict(schema_version=1,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            weights=str(args.weights.resolve()),weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            rfdetr_version=getattr(rfdetr,'__version__','unknown'),torch_version=torch.__version__,
            task=args.task,classes=names,confidence=args.conf,prediction_floor=.001,nms_iou=.5,matching_rule=args.match_rule,damage_presence=presence,
            tiled=args.tiled,views=views,candidates=candidates,merge_decisions=decisions,displayed=displayed,
            focus=focus,layer_trace=layer_trace,evaluation=evaluation,probe_baseline=baseline,
            perturbations=perturbations,occlusion=grid,events=events.events,
            interpretation='Measured computation and perturbation summary; not the model internal thoughts, reasoning transcript, or calibrated probability.')
        write_json(out/'trace.json',trace)
        write_report(out,trace)
        print(f'REPORT={out / "report.md"}',flush=True)
        return trace
    finally:
        capture.close()


def write_report(out,trace):
    focus=trace['focus'];names=trace['classes']
    lines=['# 개발용 추론 추적 보고서','',
        '> 이 문서는 실제 텐서·후보·후처리·입력 변경 실험을 자연어로 요약합니다. 모델이 말한 생각이나 사고 과정이 아닙니다. 관측 사실과 원인 가설을 구분합니다.','',
        f"사진: `{Path(trace['source']).name}` / 추적 대상: `{focus['id']}` / 최종 점수: **{focus['confidence']:.4f}** / 평가 매칭: `{trace.get('matching_rule','iou50')}`",'',
        '![선택 예측: 주황, 밑창 정답: 초록, 다른 정답: 파랑](selected_prediction.png)','',
        '## 관측한 단계별 기록','']
    lines += [f"- **{e['id']} · {e['stage']}** — {e['message']}" for e in trace['events'] if e['kind']=='observation']
    lines+=['','## 선택된 query 슬롯의 디코더 단계별 출력','',
        '최종 후보와 동일한 슬롯 번호를 따라간 기록입니다. 단계 사이에 박스가 이동할 수 있으며, 슬롯을 사람의 일관된 개념이나 생각으로 해석하지 않습니다. 중간 점수가 임계값을 넘었다고 그 단계에서 실제로 손상을 확정한 것은 아닙니다.','',
        '| 디코더 단계 | '+ ' | '.join(names)+' | 해당 뷰의 박스 좌표 |','|---|'+ '---:|'*len(names)+'---|']
    for layer in trace['layer_trace']:
        lines.append('| '+str(layer['layer'])+' | '+' | '.join(f'{s:.4f}' for s in layer['scores'][:len(names)])+' | '+str([round(v,1) for v in layer['box']])+' |')
    lines+=['','클래스 점수는 sigmoid 출력이며 합계 1의 확률 분포가 아닙니다. 별도 출력 슬롯은 과제 클래스에서 제외합니다.','',
        '## 입력 일부를 흐리게 처리한 실험','',
        f"선택 후보의 출처 뷰 `{focus['view']}`만 재추론했습니다. 원본 query 점수는 {trace['probe_baseline']['fixed_query_score']:.4f}입니다. 전체 타일 병합 결과에 대한 실험은 아닙니다.",'',
        '| 변경 영역 | 동일 query 점수 | 점수 감소량 | query 박스 IoU | 같은 위치의 최고 점수 |','|---|---:|---:|---:|---:|']
    for p in trace['perturbations']:
        lines.append(f"| {p['label']} | {p['fixed_query_score']:.4f} | {p['score_drop']:+.4f} | {p['fixed_query_box_iou']:.3f} | {p['spatial_best_score']:.4f} |")
    lines+=['','같은 위치는 원래 예측 박스와 IoU ≥ 0.3인 같은 클래스 후보입니다. 후보가 없으면 최고 점수는 0으로 기록합니다. 같은 query의 점수 감소와 실제 같은 위치 검출의 소실은 다를 수 있으므로 두 값을 함께 봅니다.','',
        '흐림 실험은 질감·경계·색상을 동시에 바꿀 수 있습니다. 영역 크기와 위치도 다르므로 각 값을 독립적인 원인 기여도로 간주하지 않습니다.','']
    if trace['occlusion']:lines+=['![영역별 흐림 민감도](occlusion.png)','',
        '왼쪽은 입력, 가운데는 고정 query 점수 변화, 오른쪽은 원래 위치와 겹치는 검출의 최고 점수 변화입니다. 붉은 쪽은 영역을 흐렸을 때 점수가 감소한 위치, 푸른 쪽은 증가한 위치입니다. 두 지도를 비교해 슬롯 교체와 실제 검출 민감성을 구분합니다. 이는 attention 지도나 확정된 인과 설명이 아닙니다.','']
    lines+=['## 원인 가설','']+[f"- **{e['id']} · 가설** — {e['message']}" for e in trace['events'] if e['kind']=='hypothesis']
    lines+=['','## 개선 판단 시 주의할 점','',
        '- 이 모델은 부위·상품 종류 라벨을 학습했습니다. 벌어짐·오염·표면 마모를 분리 학습한 결과가 아니므로 손상 종류를 단정하지 않습니다.',
        '- 특징 추출기의 출력 형태를 기록했지만 개별 특징 채널을 의미 있는 부위·손상 개념으로 해석하지 않았습니다.',
        '- 정답 미등록 영역이 실제 정상인지, 라벨 누락인지는 사진 검수가 필요합니다.',
        '- 가설을 확인하려면 동일 오류 유형의 여러 검증 사례와 정상 검출 사례에 같은 실험을 적용해야 합니다. 한 장의 결과를 전체 모델의 원인으로 일반화하지 않습니다.',
        '- 최종 test는 이번 디버깅에 사용하지 않았습니다. 가중치·라벨·서비스 추론 코드를 수정하지 않았습니다.','',
        '원시 근거: [trace.json](trace.json), 실행 중 자연어 기록: [events.jsonl](events.jsonl), 실제 전처리 입력: [inputs](inputs), 변경 실험 사진: [experiments](experiments).','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def target_plan(predictions,confidence):
    """Keep distinct boxes, including repeated classes; never collapse by label."""
    ranks={};plan=[]
    for p in predictions:
        if p['confidence']<confidence:continue
        c=p['class'];rank=ranks.get(c,0);ranks[c]=rank+1
        plan.append(dict(class_id=c,class_rank=rank,confidence=p['confidence'],box=p['box']))
    return plan


def trace_all_detections(args):
    from PIL import Image,ImageDraw
    from .rf_backend import setup
    from run_improvement_experiments import windows
    setup(SimpleNamespace(device=args.device,model='rf-detr-small',threads=4))
    from rfdetr import RFDETR
    import torch
    base,names,_=dataset(SimpleNamespace(data=ROOT/f'datasets/{args.task}/data.yaml'))
    if args.source.resolve().parent==(base/'images/test').resolve():
        raise ValueError('Exploratory debugging of the held-out test split is disabled')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    device='cuda:0' if args.device=='0' else args.device
    model=RFDETR.from_checkpoint(str(args.weights.resolve()),device=device)
    if list(model.class_names)!=names:raise ValueError('Checkpoint class names do not match task')
    with Image.open(args.source) as source:im=source.convert('RGB')
    candidates=[]
    spans=[(0,0,im.width,im.height)]+(windows(im.width,im.height) if args.tiled else [])
    for v,window in enumerate(spans):
        with torch.inference_mode():result=model.predict(im.crop(window),threshold=.001)
        for n,(box,c,score) in enumerate(zip(result.xyxy,result.class_id,result.confidence)):
            c=int(c)
            if not 0<=c<len(names):continue
            a,b,right,bottom=map(float,box);x,y=window[:2]
            candidates.append(dict(id=f'view{v}/candidate{n}',**{'class':c},confidence=float(score),box=[a+x,b+y,right+x,bottom+y]))
    kept,_=merge_with_lineage(candidates)
    plan=target_plan(kept,args.conf)
    presence=damage_presence(kept,args.conf) if args.task=='data2' else None
    index=dict(source=str(args.source.resolve()),confidence=args.conf,status='running',targets=[],matching_rule=args.match_rule,damage_presence=presence,
               note='Each item is a detection candidate, not a verified distinct physical damage instance.')
    write_json(out/'index.json',index)
    image=im.copy();draw=ImageDraw.Draw(image)
    lines=['# 사진 전체 검출 항목별 추론 보고서','',
        f'사진: `{args.source.name}` / 표시 임계값: {args.conf:.2f} / 검출 후보 **{len(plan)}개** / 평가 매칭 `{args.match_rule}`','',
        '한 사진에서 다른 부위와 같은 부위의 여러 박스를 모두 별도 추적합니다. 예측 박스 수가 실제 독립 손상 개수와 같다는 뜻은 아닙니다. 다른 사진 간 동일 손상 병합은 별도의 판단입니다.','',
        '![전체 검출 후보 번호](all_detections.png)','',
        '| 번호 | 예측 클래스 | 점수 | 정답 평가 | 자연어 보고서 |','|---|---|---:|---|---|']
    for n,target in enumerate(plan,1):
        folder=f'target_{n:02}_{names[target["class_id"]].replace(" ","_")}'
        child=SimpleNamespace(**vars(args));child.output=out/folder
        child.focus_class=target['class_id'];child.focus_rank=target['class_rank']
        try:
            trace=trace_image(child,model=model)
            focus=trace['focus']
            if iou(focus['box'],target['box'])<.999 or abs(focus['confidence']-target['confidence'])>1e-4:
                raise ValueError('Repeated inference did not reproduce the requested detection')
            evaluation=trace['evaluation']
            label=('정답 매칭' if evaluation['status']=='true_positive' else '라벨 기준 오탐') if evaluation else '정답 미제공'
            entry=dict(number=n,**target,class_name=names[target['class_id']],status='completed',evaluation=label,report=f'{folder}/report.md')
            lines.append(f"| {n} | {entry['class_name']} | {target['confidence']:.4f} | {label} | [항목 {n} 추적]({folder}/report.md) |")
        except Exception as exc:
            entry=dict(number=n,**target,class_name=names[target['class_id']],status='failed',error=str(exc))
            lines.append(f"| {n} | {entry['class_name']} | {target['confidence']:.4f} | 추적 실패 | index.json 오류 확인 |")
        index['targets'].append(entry);write_json(out/'index.json',index)
        draw.rectangle(target['box'],outline='#eb622f' if target['class_id']==0 else '#2984d5',width=2)
        draw.text((target['box'][0]+3,target['box'][1]+3),f"{n}: {names[target['class_id']]} {target['confidence']:.3f}",fill='#eb622f' if target['class_id']==0 else '#2984d5',stroke_width=1,stroke_fill='#ffffff')
    image.save(out/'all_detections.png')
    if not plan:lines+=['','임계값 이상 검출이 없습니다. 실제 손상이 없다는 판정은 아닙니다.']
    index['status']='completed' if all(t['status']=='completed' for t in index['targets']) else 'completed_with_failures'
    write_json(out/'index.json',index)
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'MULTI_REPORT={out / "report.md"}',flush=True)
    if index['status']!='completed':raise RuntimeError('Some detection traces failed; see index.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=['data1','data2'],default='data2')
    parser.add_argument('--device',default='0')
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--match-rule',choices=['any_overlap','iou50'],default='any_overlap')
    parser.add_argument('--focus-class',type=int,default=0)
    parser.add_argument('--focus-rank',type=int,default=0)
    parser.add_argument('--all-detections',action='store_true',help='Write one trace per displayed detection, including multiple boxes of the same class')
    parser.add_argument('--tiled',action='store_true')
    parser.add_argument('--occlusion',action='store_true')
    parser.add_argument('--patch',type=int,default=96)
    parser.add_argument('--stride',type=int,default=64)
    args=parser.parse_args()
    if not 0<=args.conf<=1 or args.patch<=0 or args.stride<=0 or args.stride>args.patch:
        parser.error('Invalid threshold or occlusion settings')
    if args.all_detections:trace_all_detections(args)
    else:trace_image(args)


if __name__=='__main__':main()

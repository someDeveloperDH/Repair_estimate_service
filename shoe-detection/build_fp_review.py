"""Prepare validation-only, embedded-image outsole false-positive review data."""
import base64
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from pipeline.metrics import match, iou

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/no_new_data_20261016_014558/outsole_fp_review'
OUT.mkdir(exist_ok=True)
saved=json.loads((OUT.parent/'data2_tiles30/tiled/records.json').read_text())
cases=[]
sources={}
for r in saved['records']:
    pairs,fps,_=match(r['ground_truth'],r['predictions'],.25)
    for p in fps:
        if p['class']!=0: continue
        same=max((iou(g['box'],p['box']) for g in r['ground_truth'] if g['class']==0),default=0)
        other=max((iou(g['box'],p['box']) for g in r['ground_truth'] if g['class']!=0),default=0)
        group='duplicate' if same>=.5 else 'misaligned' if same>=.1 else 'other_class' if other>=.5 else 'unmatched'
        index=len(cases)
        cases.append(dict(id=f'FP-{index+1:03}',file=r['file_name'],group=group,
            score=round(p['confidence'],4),box=[round(x,2) for x in p['box']],
            gt=r['ground_truth'],same_iou=round(same,4),other_iou=round(other,4),
            matched_predictions=[x[1] for x in pairs if x[1]['class']==0] if group=='duplicate' else [],
            width=r['width'],height=r['height']))
        sources[r['file_name']]=ROOT/'datasets/data2/images/val'/r['file_name']
cases.sort(key=lambda x:(x['group'],-x['score']))
assert len(cases)==109
counts=Counter(c['group'] for c in cases)
assert counts==dict(unmatched=82,misaligned=26,duplicate=1)

def encoded(image,width,quality):
    image=image.copy();image.thumbnail((width,width))
    buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=quality,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()

def crop_window(case):
    a,b,c,d=case['box'];w,h=case['width'],case['height']
    nearby=[g['box'] for g in case['gt'] if g['class']==0 and iou(g['box'],case['box'])>=.1]
    if nearby:
        a=min(a,*[g[0] for g in nearby]);b=min(b,*[g[1] for g in nearby]);c=max(c,*[g[2] for g in nearby]);d=max(d,*[g[3] for g in nearby])
    size=min(max(c-a,d-b)*1.5+20,max(w,h))
    size=max(size,90);cx=(a+c)/2;cy=(b+d)/2
    x=max(0,min(w-size,cx-size/2));y=max(0,min(h-size,cy-size/2))
    return [int(x),int(y),int(min(w,x+size)),int(min(h,y+size))]

for size,quality in [(340,48),(320,40),(280,36),(256,32)]:
    images={name:encoded(Image.open(path).convert('RGB'),size,quality) for name,path in sources.items()}
    crops={}
    for case in cases:
        window=crop_window(case);case['crop']=window
        with Image.open(sources[case['file']]) as im:
            crops[case['id']]=encoded(im.convert('RGB').crop(window),224,quality)
    payload=dict(cases=cases,images=images,crops=crops,counts=counts)
    text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    if len(text.encode())<820000:break
(OUT/'review_data.json').write_text(text,encoding='utf-8')

# Representative examples for visual QA. In the interactive review, all 109
# predictions are available, not only these selected examples.
representatives=[]
for group in ('unmatched','misaligned','duplicate'):
    selected=[c for c in cases if c['group']==group]
    representatives+=selected[:3]
sheet=Image.new('RGB',(960,320*len(representatives)), '#f5f5f5')
draw=ImageDraw.Draw(sheet)
for row,case in enumerate(representatives):
    with Image.open(sources[case['file']]) as source:
        full=source.convert('RGB')
    for g in case['gt']:
        color='#008b62' if g['class']==0 else '#267dd1'
        ImageDraw.Draw(full).rectangle(g['box'],outline=color,width=2)
    ImageDraw.Draw(full).rectangle(case['box'],outline='#e34927',width=3)
    full.thumbnail((300,285));sheet.paste(full,(0,row*320+30))
    crop=source.convert('RGB').crop(case['crop'])
    dc=ImageDraw.Draw(crop);x,y=case['crop'][:2]
    for g in case['gt']:
        dc.rectangle([g['box'][0]-x,g['box'][1]-y,g['box'][2]-x,g['box'][3]-y],outline='#008b62' if g['class']==0 else '#267dd1',width=2)
    dc.rectangle([case['box'][0]-x,case['box'][1]-y,case['box'][2]-x,case['box'][3]-y],outline='#e34927',width=3)
    crop.thumbnail((300,285));sheet.paste(crop,(330,row*320+30))
    draw.text((5,row*320+8),f"{case['id']} {case['group']} {case['file']} score={case['score']} same-IoU={case['same_iou']}",fill='#222222')
sheet.save(OUT/'representative_contact_sheet.jpg',quality=90)
print(json.dumps(dict(cases=len(cases),images=len(images),counts=counts,payload_bytes=len(text.encode()),output=str(OUT)),indent=2))

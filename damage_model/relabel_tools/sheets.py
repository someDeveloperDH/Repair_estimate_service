"""바뀐 사진 확인 시트: 빨강 점선 = 예전 CVAT outsole, 파랑 = 새 박스. 종류별(deleted/added/modified) 12장씩.
  python sheets.py <relabel json> <출력 폴더>"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rl = json.load(open(sys.argv[1])); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
cur = json.load(open(f"{D}/cur_boxes.json"))
T = 480; fnt = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 17)
def dashed(d, x0, y0, x1, y1, col):
    for (a, b) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        for i in range(0, 30, 2):
            d.line([(a[0] + (b[0] - a[0]) * i / 30, a[1] + (b[1] - a[1]) * i / 30), (a[0] + (b[0] - a[0]) * (i + 1) / 30, a[1] + (b[1] - a[1]) * (i + 1) / 30)], fill=col, width=4)
made = []
for c in ("deleted", "added", "modified"):
    fs = sorted(f for f, k in rl["category"].items() if k == c)
    for s in range(0, len(fs), 12):
        part = fs[s:s + 12]; rows = (len(part) + 3) // 4
        sheet = Image.new("RGB", (T * 4, T * rows + 36), "white"); d0 = ImageDraw.Draw(sheet)
        d0.text((8, 8), f"{c} {s + 1}-{s + len(part)} / {len(fs)}   red dashed = old CVAT outsole, blue = new", fill="black", font=fnt)
        for i, f in enumerate(part):
            im = Image.open(f"{D}/img/{f}").convert("RGB").resize((T, T)); d = ImageDraw.Draw(im)
            for b in cur.get(f, []):
                if b[0] == 0:
                    dashed(d, (b[1] - b[3] / 2) * T, (b[2] - b[4] / 2) * T, (b[1] + b[3] / 2) * T, (b[2] + b[4] / 2) * T, (230, 30, 30))
            for x0, y0, x1, y1 in rl["images"][f]:
                d.rectangle([x0 * T, y0 * T, x1 * T, y1 * T], outline=(30, 100, 255), width=4)
            d.rectangle([0, 0, T, 22], fill=(0, 0, 0)); d.text((4, 2), f"{c[0].upper()}{s + i + 1} {f}", fill="white", font=fnt)
            sheet.paste(im, ((i % 4) * T, 36 + (i // 4) * T))
        p = f"{outdir}/{c}_{s // 12 + 1:02d}.jpg"; sheet.save(p, quality=85); made.append(p)
print(len(made), "sheets"); [print(p) for p in made]

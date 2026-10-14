"""사진 한 장을 1000px로 보고 박스를 겹쳐 그린다 — 20×20 격자(칸 번호), 좌표는 격자 칸(0~20, 소수 가능).

  python view.py <파일명.jpg> [cur|new|both] [출력.jpg]
  cur  = 지금 라벨(outsole 파랑 'O', heel lining 초록 'H')
  new  = 다시 그린 outsole(results/*.jsonl, 파랑) + 지금 heel lining(초록)
  both = 지금 outsole(빨강 점선 'old') + 다시 그린 outsole(파랑) + heel lining(초록)
출력 기본값: <osole>/views/<파일명>_<모드>.jpg
"""
import glob, json, os, sys
from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "cur"
out = sys.argv[3] if len(sys.argv) > 3 else f"{D}/views/{f[:-4]}_{mode}.jpg"
os.makedirs(os.path.dirname(out), exist_ok=True)
cur = json.load(open(f"{D}/cur_boxes.json")).get(f, [])
to_grid = lambda b: [(b[1] - b[3] / 2) * 20, (b[2] - b[4] / 2) * 20, (b[1] + b[3] / 2) * 20, (b[2] + b[4] / 2) * 20]
old_o = [to_grid(b) for b in cur if b[0] == 0]
heel = [to_grid(b) for b in cur if b[0] == 1]
new_o = None
for p in sorted(glob.glob(f"{D}/results/*.jsonl")):
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("file") == f:
            new_o = r.get("outsole", [])          # 같은 사진이 여러 번 있으면 마지막 것
T, P = 1000, 26
im = Image.open(f"{D}/img/{f}").convert("RGB").resize((T, T))
canvas = Image.new("RGB", (T + P, T + P), "white"); canvas.paste(im, (P, P))
d = ImageDraw.Draw(canvas)
fnt = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 14); big = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 18)
for g in range(21):
    c = (255, 0, 255) if g % 5 == 0 else (255, 185, 255)
    d.line([(P + g * T / 20, P), (P + g * T / 20, P + T)], fill=c, width=1); d.line([(P, P + g * T / 20), (P + T, P + g * T / 20)], fill=c, width=1)
    if g < 20:
        d.text((P + g * T / 20 + 18, 5), str(g), fill="black", font=fnt); d.text((4, P + g * T / 20 + 16), str(g), fill="black", font=fnt)


def box(b, col, tag, dashed=False):
    x0, y0, x1, y1 = [P + v * T / 20 for v in b]
    if dashed:
        for (a, bb) in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
            n = 40
            for i in range(0, n, 2):
                d.line([(a[0] + (bb[0] - a[0]) * i / n, a[1] + (bb[1] - a[1]) * i / n),
                        (a[0] + (bb[0] - a[0]) * (i + 1) / n, a[1] + (bb[1] - a[1]) * (i + 1) / n)], fill=col, width=3)
    else:
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
    ty = y0 - 20 if y0 > P + 20 else y0
    d.rectangle([x0, ty, x0 + 12 + 10 * len(tag), ty + 20], fill=col); d.text((x0 + 3, ty + 1), tag, fill="white", font=big)


for k, b in enumerate(heel):
    box(b, (0, 170, 60), f"{k + 1}H")
if mode in ("cur", "both"):
    for k, b in enumerate(old_o):
        box(b, (230, 30, 30) if mode == "both" else (30, 100, 255), f"old{k + 1}" if mode == "both" else f"{k + 1}O", dashed=mode == "both")
if mode in ("new", "both"):
    if new_o is None:
        d.text((P + 10, P + 10), "NO NEW RESULT", fill=(255, 0, 0), font=big)
    for k, b in enumerate(new_o or []):
        box(b, (30, 100, 255), f"{k + 1}O")
canvas.save(out, quality=90); print(out)

"""2048px 사진에서 한 부분을 크게 잘라 보기. 좌표는 view.py와 같은 20×20 격자 칸(소수 가능).

  python zoom.py <파일명.jpg> <x0> <y0> <x1> <y1> [출력.jpg]
잘라 낸 그림에도 '전체 사진 기준' 격자 번호(0.5칸 간격)를 그려서, 읽은 숫자를 그대로 박스 좌표로 쓸 수 있다.
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f, x0, y0, x1, y1 = sys.argv[1], *map(float, sys.argv[2:6])
x0, y0, x1, y1 = max(0, x0), max(0, y0), min(20, x1), min(20, y1)
out = sys.argv[6] if len(sys.argv) > 6 else f"{D}/views/{f[:-4]}_z{x0:g}_{y0:g}_{x1:g}_{y1:g}.jpg"
os.makedirs(os.path.dirname(out), exist_ok=True)
im = Image.open(f"{D}/img/{f}").convert("RGB"); W, H = im.size
c = im.crop((int(x0 / 20 * W), int(y0 / 20 * H), int(x1 / 20 * W), int(y1 / 20 * H)))
S = 900 / max(c.size); c = c.resize((int(c.width * S), int(c.height * S)))
d = ImageDraw.Draw(c); font = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", 14)
g = x0 - (x0 % 0.5) + 0.5 if x0 % 0.5 else x0
while g <= x1:
    px = (g - x0) / (x1 - x0) * c.width; d.line([(px, 0), (px, c.height)], fill=(255, 0, 255) if g == int(g) else (255, 170, 255), width=1)
    d.text((px + 2, 2), f"{g:g}", fill=(255, 0, 255), font=font); g += 0.5
g = y0 - (y0 % 0.5) + 0.5 if y0 % 0.5 else y0
while g <= y1:
    py = (g - y0) / (y1 - y0) * c.height; d.line([(0, py), (c.width, py)], fill=(255, 0, 255) if g == int(g) else (255, 170, 255), width=1)
    d.text((2, py + 2), f"{g:g}", fill=(255, 0, 255), font=font); g += 0.5
c.save(out, quality=90); print(out)

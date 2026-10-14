"""S3 원본을 받아 2048px로 줄여 저장 (원본은 바로 지움). 키는 환경변수."""
import json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
D = sys.argv[1]
t = json.load(open(f"{D}/targets.json"))
names = sorted(set(t["outsole"]) | set(t["near"]))
out = f"{D}/img"
todo = [n for n in names if not os.path.exists(f"{out}/{n}")]
print("todo", len(todo), "of", len(names), flush=True)
def one(n):
    with tempfile.TemporaryDirectory() as td:
        p = f"{td}/o.jpg"
        r = subprocess.run(["aws", "s3", "cp", "--quiet", f"s3://mendii-sneakers/{n}", p], capture_output=True, text=True)
        if r.returncode:
            return n, r.stderr[:100]
        im = Image.open(p); im.draft("RGB", (2048, 2048))
        im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((2048, 2048), Image.LANCZOS)
        im.save(f"{out}/{n}.part", "JPEG", quality=90); os.replace(f"{out}/{n}.part", f"{out}/{n}")
    return n, None
done = 0; fails = []
with ThreadPoolExecutor(16) as ex:
    for n, err in ex.map(one, todo):
        done += 1
        if err: fails.append((n, err))
        if done % 100 == 0: print(done, flush=True)
print("done", done, "fails", len(fails), fails[:3], flush=True)

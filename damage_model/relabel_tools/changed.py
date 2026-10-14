"""labeler 결과에서 지금 라벨과 outsole이 달라진 사진 + 'uncertain' 메모 사진 목록.  python changed.py <chunk번호>"""
import json, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = int(sys.argv[1]); cur = json.load(open(f"{D}/cur_boxes.json"))
rows = {}
for l in open(f"{D}/results/chunk_{k:02d}.jsonl"):
    r = json.loads(l); rows[r["file"]] = r
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
out = []
for f, r in rows.items():
    old = [[(b[1] - b[3] / 2) * 20, (b[2] - b[4] / 2) * 20, (b[1] + b[3] / 2) * 20, (b[2] + b[4] / 2) * 20] for b in cur.get(f, []) if b[0] == 0]
    new = r["outsole"]
    same = len(old) == len(new) and all(max([iou(o, n) for n in new], default=0) >= 0.7 for o in old)
    if not same or "uncertain" in r.get("note", "").lower():
        out.append((f, r["action"], old, new, r.get("note", "")))
print(f"chunk {k}: 결정 {len(rows)}장, 확인할 사진 {len(out)}장")
fmt = lambda bs: ";".join(",".join(f"{v:.1f}" for v in b) for b in bs) or "-"
for f, a, old, new, note in out:
    print(f"{f}\t{a}\told [{fmt(old)}] → new [{fmt(new)}]\t{note}")

"""results/*.jsonl → 최종 outsole 박스(검증 결과 우선) + 통계 + outsole_relabel.json(0~1 좌표).
  python compile.py <출력 json>"""
import glob, json, os, sys
from collections import Counter
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chunks = json.load(open(f"{D}/chunks.json")); cur = json.load(open(f"{D}/cur_boxes.json"))
lab, ver = {}, {}
for p in sorted(glob.glob(f"{D}/results/chunk_*.jsonl")):
    for l in open(p):
        r = json.loads(l); lab[r["file"]] = r
for p in sorted(glob.glob(f"{D}/results/verify_*.jsonl")):
    for l in open(p):
        r = json.loads(l); ver[r["file"]] = r
allf = [f for c in chunks for f in c]
missing = [f for f in allf if f not in lab]
final, kind = {}, {}
for f in allf:
    if f not in lab:
        continue
    r = ver.get(f, lab[f])
    final[f] = r["outsole"]
    kind[f] = (lab[f]["action"], ver[f]["action"] if f in ver else "-")
old = {f: [[(b[1] - b[3] / 2) * 20, (b[2] - b[4] / 2) * 20, (b[1] + b[3] / 2) * 20, (b[2] + b[4] / 2) * 20] for b in cur.get(f, []) if b[0] == 0] for f in allf}
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
def cat(f):
    o, n = old[f], final[f]
    if not o and not n: return "none"
    if o and not n: return "deleted"
    if not o and n: return "added"
    same = len(o) == len(n) and all(max(iou(a, b) for b in n) >= 0.7 for a in o)
    return "kept" if same else "modified"
cats = {f: cat(f) for f in final}
print("missing (not relabeled):", len(missing))
print("final categories:", dict(Counter(cats.values())))
print("labeler→verifier actions:", dict(Counter(kind.values()).most_common(12)))
print("old outsole boxes:", sum(len(old[f]) for f in final), "→ new:", sum(len(final[f]) for f in final))
out = {"rule": "outsole = 밑창 벌어짐(본드 수선)만, 넉넉한 박스 (2026-10-14 사용자 확인). 닳음·구멍·깔창·갑피 손상 제외.",
       "images": {f: [[round(v / 20, 5) for v in b] for b in final[f]] for f in sorted(final)},
       "category": cats}
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
    print("wrote", sys.argv[1], len(out["images"]), "images")

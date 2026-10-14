"""chunk에서 아직 결정을 적지 않은 사진 목록 (순서대로).  python todo.py <chunk번호>"""
import json, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = int(sys.argv[1]); chunk = json.load(open(f"{D}/chunks.json"))[k]
p = f"{D}/results/chunk_{k:02d}.jsonl"
done = {json.loads(l)["file"] for l in open(p)} if os.path.exists(p) else set()
cur = json.load(open(f"{D}/cur_boxes.json"))
left = [f for f in chunk if f not in done]
print(f"chunk {k}: {len(chunk)}장 중 {len(done)}장 완료, 남은 {len(left)}장")
for f in left:
    n_o = sum(b[0] == 0 for b in cur.get(f, [])); n_h = sum(b[0] == 1 for b in cur.get(f, []))
    print(f, f"(지금 outsole {n_o}, heel {n_h})")

"""결정 한 줄을 결과 파일에 덧붙인다 (형식 검사 포함).

  python record.py <chunk번호> <파일명.jpg> <action> "<박스들>" "<메모>" [--verify]
  action: keep | modify | delete | add | none   (verify 때: agree | fix | revert)
  박스들: 격자 칸 좌표 "x0,y0,x1,y1;x0,y0,x1,y1" (0~20, 소수 한 자리). outsole 박스가 없으면 "" (빈 문자열).
  --verify 이면 results/verify_<chunk>.jsonl 에, 아니면 results/chunk_<chunk>.jsonl 에 쓴다.
"""
import json, os, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if a != "--verify"]; verify = "--verify" in sys.argv
k, f, action, bstr, note = args[0], args[1], args[2], args[3], (args[4] if len(args) > 4 else "")
chunks = json.load(open(f"{D}/chunks.json"))
assert f in chunks[int(k)], f"{f} 는 chunk {k} 에 없습니다"
ok_actions = {"agree", "fix", "revert"} if verify else {"keep", "modify", "delete", "add", "none"}
assert action in ok_actions, f"action은 {sorted(ok_actions)} 중 하나"
boxes = []
for part in [p for p in bstr.replace(" ", "").split(";") if p]:
    x0, y0, x1, y1 = map(float, part.split(","))
    assert 0 <= x0 < x1 <= 20 and 0 <= y0 < y1 <= 20, f"좌표 오류: {part}"
    boxes.append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)])
os.makedirs(f"{D}/results", exist_ok=True)
out = f"{D}/results/{'verify' if verify else 'chunk'}_{int(k):02d}.jsonl"
with open(out, "a") as fh:
    fh.write(json.dumps({"file": f, "action": action, "outsole": boxes, "note": note}, ensure_ascii=False) + "\n")
print("ok", out, f, action, boxes)

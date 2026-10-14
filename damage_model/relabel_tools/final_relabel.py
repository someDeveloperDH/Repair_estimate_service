"""1·2차 합본 + 3차(넓힌 기준) → colab/outsole_relabel.json. 마지막 판단의 메모가 uncertain이고 박스가 없으면 'uncertain' 목록에 넣는다."""
import glob, json, sys, collections
SP, out = sys.argv[1], sys.argv[2]
r12 = json.load(open(f"{SP}/relabel_r12.json")); r3 = json.load(open(f"{SP}/osole3/relabel_r3.json"))
imgs = dict(r12["images"]); cat = dict(r12["category"])
for f, b in r3["images"].items():
    imgs[f] = b; cat[f] = "r3:" + r3["category"][f]
last = {}
for rd in ("osole", "osole2", "osole3"):        # 라운드 순서대로, 같은 라운드에서는 labeler → verify 순
    for kind in ("chunk", "verify"):
        for p in sorted(glob.glob(f"{SP}/{rd}/results/{kind}_*.jsonl")):
            for l in open(p):
                r = json.loads(l); last[r["file"]] = r
unc = sorted(f for f, r in last.items() if f in imgs and "uncertain" in r.get("note", "").lower() and not imgs[f])
res = {"rule": "outsole = 밑창 쪽을 본드로 붙이는 수선: 갑피-밑창 분리(벌어짐·들뜸·완전 탈락) + 밑창 옆벽·가장자리·중창·폭싱·랜드의 찢어짐/갈라짐. "
               "바닥 구멍·닳음·뒷굽 닳음·바닥면 금·갑피 손상·깔창·안감 제외. 넉넉한 박스. (2026-10-14 사용자 확인)",
       "images": dict(sorted(imgs.items())), "category": cat, "uncertain": unc,
       "rounds": {**r12.get("rounds", {}), "r3": "2026-10-14 넓힌 기준(본드로 붙이는 찢어짐 포함)으로 142장 재검토"}}
json.dump(res, open(out, "w"), ensure_ascii=False)
print("images", len(imgs), "boxes", sum(map(len, imgs.values())), "uncertain(no box)", len(unc))
print(dict(collections.Counter(c.split(":")[0] + ":" + c.split(":")[-1] for c in cat.values())))

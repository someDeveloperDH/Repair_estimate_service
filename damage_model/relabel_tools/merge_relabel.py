"""1차(outsole/ 871장, 10/07 라벨 기준) + 2차(outsole2/ 865장, 최종 export 기준) → outsole_relabel.json.
같은 사진이 둘 다 있으면 2차(최종 export를 보고 다시 판단한 것)를 쓴다."""
import json, sys, collections
r1 = json.load(open(sys.argv[1])); r2 = json.load(open(sys.argv[2])); out = sys.argv[3]
imgs = dict(r1["images"]); cat = {f: "r1:" + c for f, c in r1["category"].items()}
for f, b in r2["images"].items():
    imgs[f] = b; cat[f] = "r2:" + r2["category"][f]
res = {"rule": r1["rule"], "images": dict(sorted(imgs.items())), "category": cat,
       "rounds": {"r1": "2026-10-14 10/07 export 기준 871장", "r2": "2026-10-14 최종 export(mendii_shoes_20261014_0900) 기준 865장"}}
json.dump(res, open(out, "w"), ensure_ascii=False)
print("images", len(imgs), "boxes", sum(map(len, imgs.values())), dict(collections.Counter(cat.values())))

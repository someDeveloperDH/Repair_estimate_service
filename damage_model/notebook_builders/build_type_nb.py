"""신발 종류 YOLO26 Colab 노트북. 예전 a52_shoe_yolo_colab.ipynb(YOLO11s)와 같은 데이터·같은 세션 분리, 모델만 YOLO26."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mendii_shoe_type_yolo26_colab.ipynb")

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(r'''
# 멘디 신발 종류 — YOLO26 학습과 성능 (Colab GPU)

CVAT 신발 박스 라벨(9/29 23:00 export, 8,085장)로 YOLO26 검출 모델을 학습해 신발을
**sneakers / formal / boots / sandals / slippers / heels / others**로 구분합니다.
예전 노트북(`a52_shoe_yolo_colab.ipynb`, YOLO11s)과 데이터·세션 분리·채점이 같고 **모델만 YOLO26**입니다.

**실행 순서**: 런타임 → 런타임 유형 변경 → **T4 GPU** → 런타임 → **모두 실행**.

**Drive 준비** (예전 노트북과 같은 파일, 이미 올려 두었으면 그대로): `MyDrive/mendii/`에
- `a52_512_shoe.zip` — 입고 사진 8,754장(512px). S3에는 운동화만 있어서 부츠·구두·샌들이 없으므로 이 zip을 씁니다.
- `mendii_shoe_20260929_2300.zip` — CVAT의 COCO 1.0 export.

**검증 분리**: 파일명 시각 간격 300초 이하를 한 세션으로 묶어 세션 단위로 나눕니다(검증 56세션 · 1,319장).
지금 앱의 종류 판독(SigLIP2 + 선형 분류기)도 **같은 1,319장 · 같은 정답**으로 채점해 두었으므로 9번 셀에서 바로 비교됩니다.

학습 중에는 검증을 하지 않고 정해진 epoch까지 학습한 **마지막 모델**로 채점합니다. 검증 사진으로 가장 좋은 epoch를 고르면 점수가 부풀어
SigLIP2(검증 사진을 전혀 보지 않고 고름)와 공정하게 비교되지 않기 때문입니다.

결과(`last.pt`, 그래프, 혼동행렬, `summary.json`)는 Drive의 `mendii/yolo26_shoe_0929/`에 저장되어 런타임이 끊겨도 남습니다.
사진에 작업지시서(개인정보)가 찍혀 있으니 결과 그림이 남은 노트북은 남에게 공유하지 마세요.
라이선스: YOLO26(Ultralytics)은 AGPL-3.0입니다.
''')

code(r'''
# 1. 런타임 확인. CUDA가 False면 런타임 > 런타임 유형 변경에서 GPU(T4)를 고르고 다시 실행하세요.
import torch

print("torch", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CPU로는 학습이 사실상 불가능합니다. GPU 런타임으로 바꾸세요.")
''')

code(r'''
# 2. 설치. 버전을 고정해 두어야 나중에 다시 돌려도 같은 결과가 나옵니다.
!pip install -q "ultralytics==8.4.169"

import ultralytics
ultralytics.checks()
''')

code(r'''
# 3. Drive에서 이미지와 라벨 가져오기.
IMG_ZIP    = "MyDrive/mendii/a52_512_shoe.zip"
LABEL_ZIP  = "MyDrive/mendii/mendii_shoe_20260929_2300.zip"
RUN_DIR    = "MyDrive/mendii/yolo26_shoe_0929"         # 학습 결과를 저장할 Drive 폴더
QUICK_TEST = False      # True면 사진 200장·2 epoch로 노트북 전체가 도는지만 5분 안에 확인합니다.

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive")
    WORK = Path("/content")
else:                                   # Colab 밖에서 돌릴 때: MENDII_DRIVE 아래에 같은 경로로 둡니다.
    DRIVE = Path(os.environ.get("MENDII_DRIVE", "drive")).resolve()
    WORK = Path(os.environ.get("MENDII_ROOT", "work")).resolve()
    WORK.mkdir(parents=True, exist_ok=True)
for p in (IMG_ZIP, LABEL_ZIP):
    assert (DRIVE / p).is_file(), f"Drive에서 찾을 수 없습니다: {DRIVE / p}"

RAW = WORK / "raw"
if not RAW.exists():
    # Drive에서 바로 풀면 느리므로 로컬 디스크로 복사한 뒤 풉니다.
    local_zip = WORK / "a52_512_shoe.zip"
    shutil.copy(DRIVE / IMG_ZIP, local_zip)
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(RAW)
    local_zip.unlink()

# 폴더가 몇 겹이든 파일명으로 찾습니다.
img_paths = {p.name: p for p in RAW.rglob("*.jpg")}
print("이미지", len(img_paths))

with zipfile.ZipFile(DRIVE / LABEL_ZIP) as z:
    coco = json.loads(z.read("annotations/instances_default.json"))
print("라벨 파일의 이미지", len(coco["images"]), "/ 박스", len(coco["annotations"]))
''')

code(r'''
# 4. COCO → YOLO 변환. 라벨이 하나라도 있는 이미지만 남깁니다.
from collections import Counter, defaultdict

# CVAT 카테고리 id는 1부터, YOLO 클래스는 0부터입니다. id 순서를 그대로 유지합니다.
cats = sorted(coco["categories"], key=lambda c: c["id"])
CLASSES = [c["name"] for c in cats]
cat_to_cls = {c["id"]: i for i, c in enumerate(cats)}
print("클래스", dict(enumerate(CLASSES)))

images = {im["id"]: im for im in coco["images"]}
boxes = defaultdict(list)                       # file_name -> [(cls, xc, yc, w, h)]
for a in coco["annotations"]:
    im = images[a["image_id"]]
    W, H = im["width"], im["height"]
    x, y, w, h = a["bbox"]                      # COCO: 왼쪽 위 + 폭/높이, 픽셀
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        continue
    boxes[im["file_name"]].append((
        cat_to_cls[a["category_id"]],
        (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H,
    ))

missing = [f for f in boxes if f not in img_paths]
assert not missing, f"ZIP에 없는 이미지 {len(missing)}장: {missing[:5]}"

labeled = sorted(boxes)
print(f"학습 대상 {len(labeled)}장 (라벨 없는 {len(coco['images']) - len(labeled)}장 제외)")
''')

code(r'''
# 5. 세션 단위 train/val 분리 (예전 노트북과 똑같은 방식·seed → 같은 검증 사진).
# 세션마다 가장 많은 클래스를 대표로 삼고, 클래스별로 세션의 약 VAL_RATIO만큼을 val로 보냅니다.
# 그래야 slippers·heels처럼 적은 클래스도 val에 몇 장은 들어갑니다.
import random
from datetime import datetime

GAP_SECONDS = 300
VAL_RATIO = 0.15
SEED = 0


def taken_at(name):
    return datetime.strptime(name[:15], "%Y%m%d_%H%M%S")


# 세션은 라벨 없는 사진까지 포함한 전체 순서로 끊습니다. 로컬 스크립트와 같은 방식입니다.
all_names = sorted((im["file_name"] for im in coco["images"]), key=lambda n: (taken_at(n), n))
session_of, sid, prev = {}, 0, None
for n in all_names:
    t = taken_at(n)
    if prev is None or (t - prev).total_seconds() > GAP_SECONDS:
        sid += 1
    session_of[n] = sid
    prev = t

sessions = defaultdict(list)
for n in labeled:
    sessions[session_of[n]].append(n)

by_major = defaultdict(list)
for s, names in sessions.items():
    major = Counter(b[0] for n in names for b in boxes[n]).most_common(1)[0][0]
    by_major[major].append(s)

rng = random.Random(SEED)
val_sessions = set()
for cls, ss in sorted(by_major.items()):
    ss = sorted(ss)
    rng.shuffle(ss)
    target = sum(len(sessions[s]) for s in ss) * VAL_RATIO
    got = 0
    for s in ss:
        if got >= target:
            break
        val_sessions.add(s)
        got += len(sessions[s])

split = {n: ("val" if session_of[n] in val_sessions else "train") for n in labeled}
print(f"세션 {len(sessions)}개 → val {len(val_sessions)}개")
if QUICK_TEST:                           # 분리는 그대로 두고 장수만 줄입니다.
    tr = [n for n in labeled if split[n] == "train"][::40]
    va = [n for n in labeled if split[n] == "val"][::20]
    labeled = sorted(tr + va)

rows = []
for part in ("train", "val"):
    names = [n for n in labeled if split[n] == part]
    box_cnt = Counter(b[0] for n in names for b in boxes[n])
    img_cnt = Counter(c for n in names for c in {b[0] for b in boxes[n]})
    rows.append((part, len(names), box_cnt, img_cnt))

print(f"\n{'class':<10}{'train img':>10}{'val img':>9}{'train box':>11}{'val box':>9}")
for i, c in enumerate(CLASSES):
    print(f"{c:<10}{rows[0][3][i]:>10}{rows[1][3][i]:>9}{rows[0][2][i]:>11}{rows[1][2][i]:>9}")
print(f"{'합계(장)':<10}{rows[0][1]:>10}{rows[1][1]:>9}")
''')

code(r'''
# 6. YOLO 폴더 구조와 data.yaml 만들기. 이미지는 복사하지 않고 심볼릭 링크로 겁니다.
import yaml

DS = WORK / "yolo_shoe"
if DS.exists():
    shutil.rmtree(DS)
for part in ("train", "val"):
    (DS / "images" / part).mkdir(parents=True)
    (DS / "labels" / part).mkdir(parents=True)

for n in labeled:
    part = split[n]
    os.symlink(img_paths[n], DS / "images" / part / n)
    lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes[n]]
    (DS / "labels" / part / (Path(n).stem + ".txt")).write_text("\n".join(lines) + "\n")

DATA_YAML = DS / "data.yaml"
DATA_YAML.write_text(yaml.safe_dump({
    "path": str(DS),
    "train": "images/train",
    "val": "images/val",
    "names": dict(enumerate(CLASSES)),
}, allow_unicode=True, sort_keys=False))
print(DATA_YAML.read_text())
''')

code(r'''
# 7. 학습. T4에서 yolo26s · 512px · 100 epoch는 1시간 남짓입니다.
# 원본 사진이 512px이라 입력을 더 키워도 얻는 것이 없습니다.
# 학습 중에는 검증을 하지 않고(val=False) 마지막 epoch에만 한 번 채점합니다. 시간이 줄고, 검증 사진으로 모델을 고르지 않습니다.
# 끊겼다면 RESUME = True로 바꾸고 모두 실행하면 마지막 epoch부터 이어 갑니다.
from ultralytics import YOLO

MODEL   = "yolo26s.pt"      # 더 빠르게: yolo26n.pt / 더 정확하게: yolo26m.pt
EPOCHS  = 100
IMGSZ   = 512
BATCH   = 32
NAME    = "yolo26s_512"
RESUME  = False
if QUICK_TEST:
    EPOCHS, IMGSZ, BATCH, NAME = 2, 320, 8, "quick_test"

project = DRIVE / RUN_DIR
WEIGHTS = project / NAME / "weights" / "last.pt"

if RESUME and WEIGHTS.exists():
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if ckpt.get("optimizer") is None:        # 끝까지 학습한 모델은 이어 갈 것이 없습니다.
        print("이미 끝난 학습입니다. 그대로 8번부터 채점합니다.")
    else:
        print(f"{ckpt.get('epoch', -1) + 1} epoch 다음부터 이어 갑니다.")
        YOLO(str(WEIGHTS)).train(resume=True)
    del ckpt
else:
    assert not RESUME, f"이어 갈 학습이 없습니다: {WEIGHTS}"
    YOLO(MODEL).train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        val=False,              # 마지막 epoch에만 채점
        project=str(project),
        name=NAME,
        exist_ok=True,
        seed=SEED,
        workers=2,
        plots=True,
    )

print("모델:", WEIGHTS)
''')

code(r'''
# 8. val 성능. 클래스별 검출 정확도와 혼동행렬입니다.
import gc

import pandas as pd
from IPython.display import Image as IPImage, display

gc.collect()                                 # 학습에 쓰던 GPU 메모리를 풉니다.
torch.cuda.empty_cache()

net = YOLO(str(WEIGHTS))
metrics = net.val(data=str(DATA_YAML), imgsz=IMGSZ, batch=BATCH, split="val",
                  project=str(WORK / "val_runs"), name="val", exist_ok=True)

per_class = pd.DataFrame({
    "class": [CLASSES[i] for i in metrics.box.ap_class_index],
    "precision": metrics.box.p,
    "recall": metrics.box.r,
    "mAP50": metrics.box.ap50,
    "mAP50-95": metrics.box.ap,
}).round(3)
display(per_class)
print(f"전체 mAP50 {metrics.box.map50:.3f} / mAP50-95 {metrics.box.map:.3f}")

cm = Path(metrics.save_dir) / "confusion_matrix_normalized.png"
if cm.exists():
    display(IPImage(filename=str(cm), width=700))
''')

code(r'''
# 9. 이미지 단위 분류 정확도와 지금 앱 모델(SigLIP2 + 선형 분류기)과의 비교.
# 한 장에 신발이 여러 짝 있어도 대개 같은 종류이므로, 가장 확신이 높은 박스의 클래스를 그 사진의 답으로 봅니다.
# 정답은 그 사진에서 가장 많은 박스의 클래스입니다.
PRED_BATCH = 32
val_names = [n for n in labeled if split[n] == "val"]


def predict_all(names):
    # 경로 리스트를 통째로 넘기면 전부 한 배치로 GPU에 올라가 메모리가 부족해집니다. 나눠서 넘깁니다.
    for i in range(0, len(names), PRED_BATCH):
        chunk = [str(img_paths[n]) for n in names[i:i + PRED_BATCH]]
        yield from net.predict(chunk, imgsz=IMGSZ, conf=0.25, verbose=False)


pred_rows = []
for res in predict_all(val_names):
    n = Path(res.path).name
    truth = CLASSES[Counter(b[0] for b in boxes[n]).most_common(1)[0][0]]
    if len(res.boxes):
        top = int(res.boxes.conf.argmax())
        pred, conf = CLASSES[int(res.boxes.cls[top])], float(res.boxes.conf[top])
    else:
        pred, conf = "(없음)", 0.0
    pred_rows.append({"file": n, "truth": truth, "pred": pred, "conf": conf})

pred_df = pd.DataFrame(pred_rows)
acc = (pred_df.truth == pred_df.pred).mean()
present = [c for c in CLASSES if (pred_df.truth == c).any()]
recall = {c: ((pred_df.truth == c) & (pred_df.pred == c)).sum() / (pred_df.truth == c).sum() for c in present}
f1 = {}
for c in present:
    tp = ((pred_df.truth == c) & (pred_df.pred == c)).sum()
    p = tp / max((pred_df.pred == c).sum(), 1)
    f1[c] = 0.0 if tp == 0 else 2 * p * recall[c] / (p + recall[c])
balanced, macro_f1 = sum(recall.values()) / len(present), sum(f1.values()) / len(present)

print(f"이미지 단위 정확도 {acc:.1%} ({len(pred_df)}장) / 종류 평균(클래스별 맞힌 비율의 평균) {balanced:.1%} / 매크로 F1 {macro_f1:.2f}")
display(pd.crosstab(pred_df.truth, pred_df.pred, rownames=["정답"], colnames=["예측"]))

# 지금 앱의 SigLIP2 + 선형 분류기 = app/ml/out/type_head.json "new_val" (같은 1,319장·같은 정답, 그 세션을 뺀 나머지로 학습,
# 설정은 train 세션 안의 교차검증으로만 고름). YOLO26도 검증 사진으로 고르지 않은 마지막 모델이라 공정하게 비교됩니다.
# 예전 YOLO11s는 9/24 라벨·다른 검증 934장 결과라 대략만 비교됩니다(MODEL.md §9).
SIGLIP = {"정확도": 0.907, "종류 평균": 0.572, "매크로 F1": 0.563,
          "recall": {"sneakers": 0.982, "formal": 0.876, "boots": 0.805, "sandals": 0.974,
                     "slippers": 0.300, "heels": 0.000, "others": 0.069}}
same_val = len(pred_df) == 1319 and not QUICK_TEST
compare = pd.DataFrame([
    {"모델": f"{MODEL} (이번)", "정확도": f"{acc:.1%}", "종류 평균": f"{balanced:.1%}", "매크로 F1": f"{macro_f1:.2f}",
     "검증": f"{len(pred_df)}장"},
    {"모델": "SigLIP2 + 선형 (지금 앱)", "정확도": "90.7%", "종류 평균": "57.2%", "매크로 F1": "0.56",
     "검증": "같은 1,319장" if same_val else "1,319장 (이번과 다름!)"},
    {"모델": "YOLO11s (예전 노트북)", "정확도": "83.9%", "종류 평균": "47.7%", "매크로 F1": "0.50", "검증": "9/24 라벨 934장"},
])
display(compare)
display(pd.DataFrame({"YOLO26 recall": pd.Series(recall).round(3), "SigLIP2 recall": pd.Series(SIGLIP["recall"])}))

summary = {"task": "shoe_type", "model": MODEL, "imgsz": IMGSZ, "epochs": EPOCHS, "weights": "last.pt", "val_images": len(pred_df),
           "accuracy": round(float(acc), 4), "balanced": round(float(balanced), 4), "macro_f1": round(float(macro_f1), 4),
           "recall": {c: round(float(v), 3) for c, v in recall.items()},
           "map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4),
           "quick_test": QUICK_TEST}
(project / NAME / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("아래 한 줄을 복사해 Claude에게 보내면 결과를 같이 볼 수 있습니다:\n")
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
''')

code(r'''
# 10. 틀린 사진 훑어보기. 라벨 실수인지 모델 실수인지 여기서 가려집니다.
import matplotlib.pyplot as plt

wrong = pred_df[pred_df.truth != pred_df.pred].sort_values("conf", ascending=False).head(16)
if wrong.empty:
    print("틀린 사진이 없습니다.")
else:
    cols = 4
    rows_ = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(4 * cols, 4 * rows_), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, wrong.iterrows()):
        res = net.predict(str(img_paths[r.file]), imgsz=IMGSZ, conf=0.25, verbose=False)[0]
        ax.imshow(res.plot()[:, :, ::-1])
        # Colab 기본 폰트에 한글이 없어 제목은 영어로 씁니다.
        ax.set_title(f"{r.file}\nGT {r.truth} / pred {r.pred} {r.conf:.2f}", fontsize=9)
    plt.tight_layout()
    plt.show()

pred_df.to_csv(project / NAME / "val_image_predictions.csv", index=False)
print("저장:", project / NAME / "val_image_predictions.csv")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", OUT, len(cells), "cells")

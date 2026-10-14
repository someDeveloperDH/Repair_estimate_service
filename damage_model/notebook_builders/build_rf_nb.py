"""RF-DETR 파손 검출 Colab 노트북. 데이터 셀(3~6)은 YOLO 노트북(build_nb.py)과 같은 코드를 그대로 가져와
같은 사진·같은 정답·같은 train/val 분리를 쓰고, 9번 셀에서 RF-DETR과 YOLO26을 같은 채점 코드로 비교한다."""
import json
import os
import sys
import tempfile
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mendii_damage_rfdetr_colab.ipynb")
HERE = Path(__file__).resolve().parent

# YOLO 노트북 셀 소스 가져오기
ns = {"__name__": "build_nb", "__file__": str(HERE / "build_nb.py")}
tmp = tempfile.mktemp(suffix=".ipynb")
sys_argv = sys.argv
sys.argv = ["build_nb.py", tmp]
exec(compile((HERE / "build_nb.py").read_text(encoding="utf-8"), "build_nb.py", "exec"), ns)
sys.argv = sys_argv
os.remove(tmp)
y = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in ns["cells"]]


def must_replace(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r'''
# 멘디 파손 검출 — RF-DETR 학습과 YOLO26 비교 (Colab GPU)

RF-DETR(Roboflow, **Apache-2.0** — 상용 앱에 라이선스 비용 없이 넣을 수 있음)로 파손 박스(**outsole** = 밑창 벌어짐,
**heel lining** = 뒤꿈치 안감 손상)를 학습하고, YOLO26 노트북과 **같은 사진 · 같은 정답 · 같은 train/val 분리 · 같은 채점 코드**로 비교합니다.

**실행 순서**: 런타임 → 런타임 유형 변경 → **T4 GPU** → 런타임 → **모두 실행**. 3번 셀에서 파일을 올리라고 하면 한꺼번에 골라 올립니다.

**올릴 파일 3개** — YOLO 노트북과 같습니다 (이미 `MyDrive/mendii/yolo26_damage/`에 있으면 올리지 않아도 됩니다)
- CVAT 프로젝트 최종 export — `mendii_shoes_20261014_0900.zip`
- 밑창 재라벨 — `outsole_relabel.json`
- S3 읽기 전용 키 — `mendii-reader_accessKeys.csv`

**YOLO 노트북을 먼저 돌렸다면** 같은 Drive 폴더의 사진 캐시(`img1280.zip`)와 분리 목록(`runs/yolo26s_1024_final_e60/split.json`)을 그대로 씁니다.
그래서 학습·검증 사진이 YOLO와 한 장도 다르지 않고, 9번 셀이 YOLO 모델(`last.pt`)도 같은 코드로 다시 채점해 나란히 보여 줍니다.

**공정한 비교를 위해**: YOLO처럼 학습 중에는 검증하지 않고(마지막 epoch에 한 번만) 마지막 EMA 가중치(`last_ema.pth`)로 채점합니다.
검증 사진으로 가장 좋은 epoch를 고르지 않습니다. 결과는 Drive `MyDrive/mendii/yolo26_damage/runs_rfdetr/`에 저장됩니다.
사진에 작업지시서(개인정보)가 찍혀 있으니 결과 그림이 남은 노트북은 남에게 공유하지 마세요.
''')

code(y[1])

code(r'''
# 2. 설치. 버전을 고정해 두어야 나중에 다시 돌려도 같은 결과가 나옵니다.
# ultralytics는 YOLO 모델을 같은 코드로 다시 채점할 때만 씁니다(앱에 넣는 것은 RF-DETR).
!pip install -q "rfdetr[train]==1.11.2" "ultralytics==8.4.169" boto3

import importlib.metadata as md

import pytorch_lightning  # noqa: F401  학습용 추가 패키지가 깔렸는지 여기서 바로 확인합니다.
print("rfdetr", md.version("rfdetr"), "| torch", md.version("torch"))
if md.version("torch").split("+")[0] != torch.__version__.split("+")[0]:
    print("⚠️ 설치하면서 torch 버전이 바뀌었습니다. 런타임 > 세션 다시 시작을 누른 뒤 다시 모두 실행하세요.")
''')

s3 = y[3]
s3 = must_replace(s3, '''# 학습. 안감 파손 박스가 작아서(중앙값이 사진 한 변의 7%) 640px보다 1024px이 유리합니다.
MODEL   = "yolo26s.pt"  # 더 크게: yolo26m.pt (계산량 3.3배, T4에서 BATCH 8)
EPOCHS  = 60            # 최종본(학습 사진 약 4,350장) 기준 T4에서 epoch당 약 3.3분 → 약 3.3시간. 끊기면 RESUME으로 이어 갑니다.
IMGSZ   = 1024
BATCH   = 16
NAME    = "yolo26s_1024_final_e60"    # 결과 폴더 이름. 설정을 바꿔 다시 학습하면 이름도 바꾸세요.
RESUME  = False         # 끊긴 학습을 이어 갈 때 True로 바꾸고 모두 실행 (같은 train/val 분리를 다시 씁니다)
SPLIT_FROM = "yolo26s_1024"   # 첫 학습의 검증 세션은 그대로 검증에 두고, 그 학습에 쓴 세션은 검증에 넣지 않습니다 (없으면 새로 나눔)
QUICK_TEST = False      # True면 사진 180장·2 epoch로 노트북 전체가 도는지만 5분 안에 확인합니다.''',
'''# 학습. 안감 파손 박스가 작아서(중앙값이 사진 한 변의 7%) 기본 해상도(512)보다 키웁니다. 32의 배수여야 합니다.
RF_MODEL   = "small"       # small(기본, 학습 빠름) / medium(조금 더 정확, 약 1.4배 느림)
RESOLUTION = 768           # 896·1024도 되지만 T4에서 느려집니다(해상도 제곱에 비례).
EPOCHS     = 20            # 8번 셀이 첫 epoch 시간을 보여 줍니다. 너무 길면 멈추고 줄이세요.
NAME       = "rfdetr_s_768_e20"   # 결과 폴더 이름. 설정을 바꿔 다시 학습하면 이름도 바꾸세요.
RESUME     = False         # 끊긴 학습을 이어 갈 때 True로 바꾸고 모두 실행
YOLO_RUN   = "yolo26s_1024_final_e60"   # 비교할 YOLO 학습(같은 Drive 폴더의 runs/). 그 분리 목록과 last.pt를 씁니다.
YOLO_IMGSZ = 1024
SPLIT_FROM = "yolo26s_1024"   # YOLO_RUN의 split.json이 없을 때만: YOLO 노트북과 같은 규칙으로 새로 나눔
QUICK_TEST = False         # True면 사진 180장·1 epoch로 노트북 전체가 도는지만 확인합니다.''')
s3 = must_replace(s3, '''if QUICK_TEST:
    EPOCHS, IMGSZ, BATCH, NAME = 2, 320, 8, "quick_test"''', '''if QUICK_TEST:
    RESOLUTION, EPOCHS, NAME = 320, 1, "quick_test"''')
code(s3)

code(y[4])

s5 = y[5]
s5 = must_replace(s5, '''SPLIT_JSON = OUT_DIR / "runs" / NAME / "split.json"
inputs_now = [main_path.name] + [p.name for p, _, _ in overrides] + ([RELABEL.name] if RELABEL else [])
if RESUME and SPLIT_JSON.exists():
    saved = json.loads(SPLIT_JSON.read_text())''', '''SPLIT_JSON = OUT_DIR / "runs_rfdetr" / NAME / "split.json"
YOLO_SPLIT = OUT_DIR / "runs" / YOLO_RUN / "split.json"
inputs_now = [main_path.name] + [p.name for p, _, _ in overrides] + ([RELABEL.name] if RELABEL else [])
use_split = YOLO_SPLIT if (YOLO_SPLIT.exists() and not QUICK_TEST) else (SPLIT_JSON if RESUME and SPLIT_JSON.exists() else None)
if use_split:
    SPLIT_JSON = use_split
    saved = json.loads(SPLIT_JSON.read_text())''')
s5 = must_replace(s5, '''# 5. 세션 단위 train / val 분리.''', '''# 5. 세션 단위 train / val 분리. YOLO 학습의 split.json이 있으면 그대로 써서 YOLO와 같은 사진으로 학습·채점합니다.''')
code(s5)

code(y[6])

code(r'''
# 7. RF-DETR용 COCO 폴더(train / valid). 사진은 심볼릭 링크로 걸고, 박스는 1280px 사진의 픽셀 좌표로 씁니다.
# 파손 없는 사진은 박스 없는 이미지로 넣어 배경으로 학습합니다.
from PIL import Image

DS = ROOT / "rfdetr_damage"
if DS.exists():
    shutil.rmtree(DS)
CATS = [{"id": i, "name": c, "supercategory": "damage"} for i, c in enumerate(CLASSES)]
img_size = {}
for part, names in (("train", train_names), ("valid", val_names)):
    d = DS / part
    d.mkdir(parents=True)
    imgs, anns = [], []
    # train에서는 outsole 사진을 이름만 바꾼 링크로 REPEAT_OUTSOLE번 넣습니다(YOLO 노트북과 같은 반복 샘플링).
    entries = [(n, n) for n in names]
    if part == "train":
        entries += [(n, f"{Path(n).stem}__r{r}.jpg") for n in names if any(b[0] == 0 for b in boxes[n])
                    for r in range(1, REPEAT_OUTSOLE)]
    for k, (n, fname) in enumerate(entries, 1):
        os.symlink(IMG_DIR / n, d / fname)
        if n not in img_size:
            with Image.open(IMG_DIR / n) as im:
                img_size[n] = im.size
        W, H = img_size[n]
        imgs.append({"id": k, "file_name": fname, "width": W, "height": H})
        for c, xc, yc, w, h in boxes[n]:
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": c, "iscrowd": 0,
                         "bbox": [(xc - w / 2) * W, (yc - h / 2) * H, w * W, h * H], "area": w * W * h * H})
    (d / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": CATS}))
    print(f"{part}: 사진 {len(imgs)}장, 박스 {len(anns)}개")
''')

code(r'''
# 8. 학습. 학습 중에는 검증하지 않고(eval_interval = EPOCHS → 마지막 epoch에만 채점) 마지막 EMA 가중치로 채점합니다.
# 배치는 T4 메모리에 맞게 자동으로 정하고(batch_size="auto", 실효 배치 16), T4는 fp16으로 학습합니다.
# 끊겼다면 3번 셀에서 RESUME = True로 바꾸고 모두 실행하면 마지막 epoch(last.ckpt)부터 이어 갑니다.
import time

from rfdetr import RFDETR, RFDETRMedium, RFDETRSmall

RUN = OUT_DIR / "runs_rfdetr" / NAME
FINAL = RUN / "last_ema.pth"                # 학습이 끝나면 생기는 마지막 EMA 가중치
Model = {"small": RFDETRSmall, "medium": RFDETRMedium}[RF_MODEL]
train_kwargs = dict(dataset_dir=str(DS), output_dir=str(RUN), epochs=EPOCHS, batch_size="auto", resolution=RESOLUTION,
                    eval_interval=EPOCHS, amp_dtype="fp16", num_workers=2, seed=SEED, tensorboard=False,
                    progress_bar="tqdm")

t0 = time.time()
if FINAL.exists():
    print("이미 끝난 학습입니다. 그대로 9번부터 채점합니다.")
elif RESUME and (RUN / "last.ckpt").exists():
    print("last.ckpt부터 이어 갑니다.")
    Model().train(**train_kwargs, resume=str(RUN / "last.ckpt"))
else:
    assert not RESUME, f"이어 갈 학습이 없습니다: {RUN / 'last.ckpt'}"
    RUN.mkdir(parents=True, exist_ok=True)
    Model().train(**train_kwargs)
print(f"학습 {(time.time() - t0) / 60:.0f}분 | 모델: {FINAL}")
assert FINAL.exists(), "last_ema.pth가 없습니다. 위 학습 로그를 확인하세요."
''')

code(r'''
# 9. 같은 검증 사진 · 같은 정답 · 같은 채점 코드로 RF-DETR과 YOLO26을 나란히 채점합니다.
#  - AP50 / AP50-95: 박스 단위 정확도(COCO 방식 101점 보간, 같은 클래스·한 정답에 한 예측)
#  - 파손 사진 찾음 / 잘못 경보: 사진 한 장 단위(앱이 "추가 발견"을 띄울지 정하는 기준)
#  - 크기별 찾은 비율: 정답 박스가 사진 한 변의 8% 미만(멀리서 찍은 작은 파손) / 8~20% / 20% 이상(근접 사진)
#  - 차이의 90% 구간: 검증 세션을 1,000번 다시 뽑아(세션 단위 부트스트랩) 계산. 0을 걸치면 우연일 수 있는 차이입니다.
import gc

import numpy as np
import pandas as pd
from IPython.display import display

PRED_BATCH = 8
THRS = np.arange(0.5, 0.96, 0.05)
LOOSE = 0.3                                  # 넉넉한 박스라 범위가 사람마다 다른 outsole은 IoU 0.3 기준도 같이 봅니다.
gc.collect()
torch.cuda.empty_cache()


def predict_rfdetr(names):
    net = RFDETR.from_checkpoint(str(FINAL), trust_checkpoint=True)
    assert set(CLASSES) <= set(net.class_names), f"RF-DETR 클래스 이름이 다릅니다: {net.class_names}"
    name_to_cls = {nm: CLASSES.index(nm) for nm in CLASSES}
    out = {}
    for i in range(0, len(names), PRED_BATCH):
        chunk = names[i:i + PRED_BATCH]
        dets = net.predict([str(IMG_DIR / n) for n in chunk], threshold=0.01, include_source_image=False)
        dets = dets if isinstance(dets, list) else [dets]
        for n, det in zip(chunk, dets):
            W, H = img_size[n]
            rows = []
            # predict()가 붙여 주는 이름을 씁니다. '물체 없음' 칸(__background__)은 여기서 걸러집니다.
            for xyxy, s, nm in zip(det.xyxy, det.confidence, det.data["class_name"]):
                if nm in name_to_cls:
                    rows.append((name_to_cls[nm], float(s), (xyxy[0] / W, xyxy[1] / H, xyxy[2] / W, xyxy[3] / H)))
            out[n] = rows
    del net
    return out


def predict_yolo(names, weights):
    from ultralytics import YOLO
    net = YOLO(str(weights))
    out = {}
    for i in range(0, len(names), PRED_BATCH):
        chunk = [str(IMG_DIR / n) for n in names[i:i + PRED_BATCH]]
        for res in net.predict(chunk, imgsz=YOLO_IMGSZ, conf=0.01, verbose=False):
            b = res.boxes
            out[Path(res.path).name] = [(int(c), float(s), tuple(map(float, xy)))
                                        for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]
    del net
    return out


def gt_xyxy(n):
    return [(c, (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)) for c, xc, yc, w, h in boxes[n]]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def match(preds, n, ci, thr):
    """그 사진의 클래스 ci 예측마다 (conf, 맞음?) — 높은 conf부터 아직 안 맞힌 정답과 IoU ≥ thr면 맞음."""
    g = [b for c, b in gt_xyxy(n) if c == ci]
    used = [False] * len(g)
    out = []
    for c, s, p in sorted([q for q in preds[n] if q[0] == ci], key=lambda q: -q[1]):
        best, bj = thr, -1
        for j, gb in enumerate(g):
            v = iou(p, gb)
            if not used[j] and v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = True
        out.append((s, bj >= 0))
    return out, len(g)


def ap_from(conf, tp, img, n_gt_img, w):
    """가중치 w(사진별 부트스트랩 횟수)로 COCO 101점 AP."""
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-conf, kind="stable")
    ww = w[img[order]]
    ctp = np.cumsum(ww * tp[order])
    cfp = np.cumsum(ww * (1 - tp[order]))
    total = float((w * n_gt_img).sum())
    if total == 0:
        return float("nan")
    rec = ctp / total
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def tables(preds, names):
    """클래스별·IoU 문턱별 (conf, tp, 사진 번호) 배열과 사진별 정답 수."""
    T = {}
    for ci in range(len(CLASSES)):
        for thr in list(THRS) + [LOOSE]:
            conf, tp, img, ngt = [], [], [], np.zeros(len(names))
            for k, n in enumerate(names):
                m, g = match(preds, n, ci, thr)
                ngt[k] = g
                for s, ok in m:
                    conf.append(s); tp.append(1.0 if ok else 0.0); img.append(k)
            T[(ci, round(float(thr), 2))] = (np.array(conf), np.array(tp), np.array(img, dtype=int), ngt)
    return T


def image_level(preds, names, t):
    rows = {}
    clean = [n for n in names if not boxes[n]]
    for ci, cname in enumerate(CLASSES):
        has = [n for n in names if any(c == ci for c, _ in gt_xyxy(n))]
        hs = set(has)
        score = {n: max([s for c, s, _ in preds[n] if c == ci], default=0.0) for n in names}
        found = sum(score[n] >= t for n in has)
        fa = sum(score[n] >= t for n in clean)
        size_hit = {"작은(<8%)": [0, 0], "중간(8~20%)": [0, 0], "큰(≥20%)": [0, 0]}
        for n in has:
            pb = [p for c, s, p in preds[n] if c == ci and s >= t]
            for c, g in gt_xyxy(n):
                if c != ci:
                    continue
                side = ((g[2] - g[0]) * (g[3] - g[1])) ** 0.5
                key = "작은(<8%)" if side < 0.08 else "중간(8~20%)" if side < 0.2 else "큰(≥20%)"
                size_hit[key][1] += 1
                size_hit[key][0] += any(iou(g, p) >= 0.3 for p in pb)
        rows[cname] = {"파손 사진 찾음": f"{found}/{len(has)}", "잘못 경보(파손 없는 사진)": f"{fa}/{len(clean)}",
                       **{f"박스 찾음 {k}": f"{a}/{b}" for k, (a, b) in size_hit.items()}}
    return rows


names = list(val_names)
models = {"RF-DETR": predict_rfdetr(names)}
yolo_w = OUT_DIR / "runs" / YOLO_RUN / "weights" / "last.pt"
if yolo_w.exists() and not QUICK_TEST:
    models["YOLO26"] = predict_yolo(names, yolo_w)
else:
    print(f"YOLO 모델이 없어 RF-DETR만 채점합니다: {yolo_w}")

TB = {m: tables(p, names) for m, p in models.items()}
ones = np.ones(len(names))
rows = []
for m in models:
    r = {"모델": m}
    for ci, cname in enumerate(CLASSES):
        ap50 = ap_from(*TB[m][(ci, 0.5)][:3], TB[m][(ci, 0.5)][3], ones)
        ap = np.nanmean([ap_from(*TB[m][(ci, round(float(t), 2))][:3], TB[m][(ci, round(float(t), 2))][3], ones) for t in THRS])
        r[f"{cname} AP30"] = round(ap_from(*TB[m][(ci, LOOSE)][:3], TB[m][(ci, LOOSE)][3], ones), 3)
        r[f"{cname} AP50"] = round(ap50, 3)
        r[f"{cname} AP50-95"] = round(float(ap), 3)
    rows.append(r)
display(pd.DataFrame(rows))

def session_level(preds, names, t):
    """신발(세션) 단위: 같은 세션 사진 중 한 장이라도 그 클래스 박스(conf ≥ t)를 내면 찾음."""
    by_s = {}
    for n in names:
        by_s.setdefault(session_of[n], []).append(n)
    out = {}
    for ci, cname in enumerate(CLASSES):
        top = {n: max([s for c, s, _ in preds[n] if c == ci], default=0.0) for n in names}
        pos_s = [s for s, ns in by_s.items() if any(c == ci for n in ns for c, _ in gt_xyxy(n))]
        neg_s = [s for s, ns in by_s.items() if all(not boxes[n] for n in ns)]
        hit = sum(any(top[n] >= t for n in by_s[s]) for s in pos_s)
        fa = sum(any(top[n] >= t for n in by_s[s]) for s in neg_s)
        out[cname] = {"신발(세션) 찾음": f"{hit}/{len(pos_s)}", "잘못 경보(파손 없는 세션)": f"{fa}/{len(neg_s)}"}
    return out


for t in (0.25, 0.5):
    print(f"\n사진 단위 (conf ≥ {t})")
    display(pd.DataFrame({f"{m} · {c}": v for m, p in models.items() for c, v in image_level(p, names, t).items()}))
    print(f"신발(세션) 단위 (conf ≥ {t}) — 앱은 한 켤레를 여러 장 찍어 올리므로 이 숫자가 실제 사용에 가깝습니다")
    display(pd.DataFrame({f"{m} · {c}": v for m, p in models.items() for c, v in session_level(p, names, t).items()}))

# 세션 단위 부트스트랩: RF-DETR − YOLO26 AP50 차이의 90% 구간
summary = {"val_images": len(names), "rf_model": RF_MODEL, "resolution": RESOLUTION, "epochs": EPOCHS,
           "yolo_run": YOLO_RUN if "YOLO26" in models else None, "box": rows,
           "image_level": {m: {str(t): image_level(p, names, t) for t in (0.25, 0.5)} for m, p in models.items()},
           "session_level": {m: {str(t): session_level(p, names, t) for t in (0.25, 0.5)} for m, p in models.items()}}
if "YOLO26" in models:
    rng = np.random.default_rng(SEED)
    sess = np.array([session_of[n] for n in names])
    uniq = np.unique(sess)
    pos = {s: np.where(sess == s)[0] for s in uniq}
    diffs = {c: [] for c in CLASSES}
    for _ in range(1000):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        w = np.zeros(len(names))
        for s in draw:
            w[pos[s]] += 1
        for ci, c in enumerate(CLASSES):
            a = ap_from(*TB["RF-DETR"][(ci, 0.5)][:3], TB["RF-DETR"][(ci, 0.5)][3], w)
            b = ap_from(*TB["YOLO26"][(ci, 0.5)][:3], TB["YOLO26"][(ci, 0.5)][3], w)
            diffs[c].append(a - b)
    summary["ap50_diff_rfdetr_minus_yolo_90ci"] = {}
    for c in CLASSES:
        lo, hi = np.nanpercentile(diffs[c], [5, 95])
        summary["ap50_diff_rfdetr_minus_yolo_90ci"][c] = [round(float(lo), 3), round(float(hi), 3)]
        verdict = "RF-DETR이 낫다" if lo > 0 else "YOLO26이 낫다" if hi < 0 else "차이가 우연일 수 있음"
        print(f"{c}: AP50 차이(RF-DETR − YOLO26) 90% 구간 [{lo:+.3f}, {hi:+.3f}] → {verdict}")

(RUN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("\n저장:", RUN / "summary.json")
print("아래 한 줄을 복사해 Claude에게 보내면 결과를 같이 볼 수 있습니다:\n")
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
''')

code(r'''
# 10. 두 모델이 엇갈린 사진 (초록 = 정답, 파랑 = RF-DETR, 빨강 = YOLO26, conf ≥ 0.25).
import cv2
import matplotlib.pyplot as plt

SHOW_CONF = 0.25
COL = {"RF-DETR": (40, 90, 255), "YOLO26": (255, 40, 40)}


def found(m, n):
    return any(any(iou(g, p) >= 0.3 for c2, s, p in models[m][n] if c2 == c and s >= SHOW_CONF) for c, g in gt_xyxy(n))


def draw(n):
    im = cv2.imread(str(IMG_DIR / n))[:, :, ::-1].copy()
    H, W = im.shape[:2]
    for c, g in gt_xyxy(n):
        cv2.rectangle(im, (int(g[0] * W), int(g[1] * H)), (int(g[2] * W), int(g[3] * H)), (0, 220, 0), 7)
    for m, p in models.items():
        for c, s, b in p[n]:
            if s >= SHOW_CONF:
                cv2.rectangle(im, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)), COL[m], 4)
    return im


dmg = [n for n in names if boxes[n]]
if len(models) == 2:
    only_rf = [n for n in dmg if found("RF-DETR", n) and not found("YOLO26", n)][:6]
    only_yolo = [n for n in dmg if found("YOLO26", n) and not found("RF-DETR", n)][:6]
    print(f"RF-DETR만 찾은 파손 사진 {sum(found('RF-DETR', n) and not found('YOLO26', n) for n in dmg)}장 / "
          f"YOLO26만 찾은 사진 {sum(found('YOLO26', n) and not found('RF-DETR', n) for n in dmg)}장 (파손 사진 {len(dmg)}장 중)")
    groups = [("Only RF-DETR found", only_rf), ("Only YOLO26 found", only_yolo)]
else:
    groups = [("RF-DETR on damaged photos", dmg[:8])]
for title, ns_ in groups:
    if not ns_:
        continue
    cols = 3
    rows_ = (len(ns_) + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 5 * rows_), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, n in zip(axes.flat, ns_):
        ax.imshow(draw(n))
        ax.set_title(n, fontsize=8)
    fig.suptitle(title)                      # Colab 기본 폰트에 한글이 없어 영어로 씁니다.
    plt.tight_layout()
    plt.show()
''')

md(r'''
## 결과 읽는 법

- **AP50**: 박스가 정답과 절반 이상 겹치면 맞은 것으로 본 정확도(0~1). **AP50-95**는 더 꼭 맞아야 점수가 나옵니다.
- **사진 단위**: "파손 사진 찾음"은 높고 "잘못 경보"는 낮아야 앱의 AI 추가 발견에 쓸 수 있습니다.
- **크기별**: 고객이 신고한 파손은 근접 사진(10~15cm)을 받으므로 "큰(≥20%)"이 실제 주 사용처이고, "작은(<8%)"은 AI 추가 발견에 해당합니다.
- **90% 구간이 0을 걸치면** 두 모델의 차이는 이 검증 사진 수로는 가릴 수 없는 수준입니다. 그때는 라이선스(RF-DETR Apache-2.0 vs YOLO AGPL-3.0)와 속도로 고르면 됩니다.
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

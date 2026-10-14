"""사진 단위 파손 분류 Colab 노트북. 데이터 셀(3~6)은 YOLO 노트북(build_nb.py)과 같은 코드를 그대로 가져와
같은 사진·같은 정답·같은 train/val 분리를 쓰고, 8번 셀에서 분류 모델과 YOLO26을 '사진 단위'로 같은 채점 코드로 비교한다.

검출(박스)이 아니라 "이 사진에 outsole / heel lining 파손이 있나"를 클래스마다 확률로 낸다(다중 라벨).
둘 다 낮으면 '그 외'(두 부위 파손 없음). 앱에서 '이 사진에 어떤 수선이 필요한가'를 빠르게 거르는 용도.

    python build_cls_nb.py ../notebooks/mendii_damage_cls_colab.ipynb
"""
import json
import os
import sys
import tempfile
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mendii_damage_cls_colab.ipynb")
HERE = Path(__file__).resolve().parent

# YOLO 노트북 셀 소스 가져오기 (build_rf_nb.py와 같은 방식)
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
# 멘디 파손 분류 — 사진 단위 outsole / heel lining / 그 외 (Colab GPU)

검출 모델(YOLO26·RF-DETR)이 **박스**를 찾는 것과 달리, 이 노트북은 사진 한 장에
**outsole**(밑창 벌어짐) · **heel lining**(뒤꿈치 안감 손상) 파손이 **있는지**를 클래스마다 확률로 냅니다(다중 라벨).
둘 다 기준(0.5)보다 낮으면 **그 외**(두 부위 파손 없음)로 봅니다.

모델은 timm `convnext_tiny`(Apache-2.0, ImageNet-22k 사전학습)를 파인튜닝합니다. 512px · 15 epoch, T4에서 약 30~40분(추정).

**실행 순서**: 런타임 → 런타임 유형 변경 → **T4 GPU** → 런타임 → **모두 실행**. 3번 셀에서 파일을 올리라고 하면 한꺼번에 골라 올립니다.

**올릴 파일 3개** — YOLO 노트북과 같습니다 (이미 `MyDrive/mendii/yolo26_damage/`에 있으면 올리지 않아도 됩니다)
- CVAT 프로젝트 최종 export — `mendii_shoes_20261014_0900.zip`
- 밑창 재라벨 — `outsole_relabel.json`
- S3 읽기 전용 키 — `mendii-reader_accessKeys.csv`

**YOLO 노트북을 먼저 돌렸다면** 같은 Drive 폴더의 사진 캐시(`img1280.zip`)와 분리 목록(`runs/yolo26s_1024_final_e60/split.json`)을 그대로 씁니다.
그래서 학습·검증 사진이 YOLO와 같고, 8번 셀이 YOLO `last.pt`도 사진 단위로 다시 채점해 나란히 보여 줍니다.

**공정한 비교를 위해**: 학습 중에는 검증하지 않고 마지막 가중치로 채점합니다. 기준값(0.5)도 검증 사진으로 고르지 않습니다.
결과는 Drive `MyDrive/mendii/yolo26_damage/runs_cls/`에 저장됩니다.
사진에 작업지시서(개인정보)가 찍혀 있으니 결과 그림이 남은 노트북은 남에게 공유하지 마세요.
''')

code(y[1])

code(r'''
# 2. 설치. 버전을 고정해 두어야 나중에 다시 돌려도 같은 결과가 나옵니다.
# ultralytics는 YOLO 모델을 같은 코드로 다시 채점할 때만 씁니다.
!pip install -q "timm==1.0.15" "ultralytics==8.4.169" boto3

import importlib.metadata as md

print("timm", md.version("timm"), "| torch", md.version("torch"))
''')

s3 = must_replace(y[3], '''# 학습. 안감 파손 박스가 작아서(중앙값이 사진 한 변의 7%) 640px보다 1024px이 유리합니다.
MODEL   = "yolo26s.pt"  # 더 크게: yolo26m.pt (계산량 3.3배, T4에서 BATCH 8)
EPOCHS  = 60            # 최종본(학습 사진 약 4,350장) 기준 T4에서 epoch당 약 3.3분 → 약 3.3시간. 끊기면 RESUME으로 이어 갑니다.
IMGSZ   = 1024
BATCH   = 16
NAME    = "yolo26s_1024_final_e60"    # 결과 폴더 이름. 설정을 바꿔 다시 학습하면 이름도 바꾸세요.
RESUME  = False         # 끊긴 학습을 이어 갈 때 True로 바꾸고 모두 실행 (같은 train/val 분리를 다시 씁니다)
SPLIT_FROM = "yolo26s_1024"   # 첫 학습의 검증 세션은 그대로 검증에 두고, 그 학습에 쓴 세션은 검증에 넣지 않습니다 (없으면 새로 나눔)
QUICK_TEST = False      # True면 사진 180장·2 epoch로 노트북 전체가 도는지만 5분 안에 확인합니다.''',
'''# 학습. 안감 파손이 작아서(중앙값이 사진 한 변의 7%) 분류 모델 기본(224px)보다 크게 봅니다.
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"   # timm 모델 이름 (Apache-2.0). 더 크게: convnext_small.fb_in22k_ft_in1k
IMGSZ    = 512          # 640이면 조금 더 정확할 수 있지만 약 1.6배 느려집니다.
EPOCHS   = 15           # 8번 셀 전 7번 셀이 첫 epoch 시간을 보여 줍니다. 너무 길면 멈추고 줄이세요.
BATCH    = 16
LR       = 2e-4
THRESH   = 0.5          # 확률이 이 값 이상이면 그 부위 파손 있음. 검증 사진으로 고르지 않습니다.
NAME     = "cls_convnext_t_512_e15"   # 결과 폴더 이름. 설정을 바꿔 다시 학습하면 이름도 바꾸세요.
RESUME   = False        # True면 저장된 분리(split.json)를 다시 씁니다. 학습은 last.pt가 있으면 건너뜁니다.
YOLO_RUN = "yolo26s_1024_final_e60"   # 비교할 YOLO 학습(같은 Drive 폴더의 runs/). 그 분리 목록과 last.pt를 씁니다.
YOLO_IMGSZ = 1024
SPLIT_FROM = "yolo26s_1024"   # YOLO_RUN의 split.json이 없을 때만: YOLO 노트북과 같은 규칙으로 새로 나눔
QUICK_TEST = False      # True면 사진 120장·2 epoch로 노트북 전체가 도는지만 확인합니다.''')
code(s3)

code(y[4])

s5 = must_replace(y[5], '''SPLIT_JSON = OUT_DIR / "runs" / NAME / "split.json"
inputs_now = [main_path.name] + [p.name for p, _, _ in overrides] + ([RELABEL.name] if RELABEL else [])
if RESUME and SPLIT_JSON.exists():
    saved = json.loads(SPLIT_JSON.read_text())''', '''SPLIT_JSON = OUT_DIR / "runs_cls" / NAME / "split.json"
YOLO_SPLIT = OUT_DIR / "runs" / YOLO_RUN / "split.json"
inputs_now = [main_path.name] + [p.name for p, _, _ in overrides] + ([RELABEL.name] if RELABEL else [])
use_split = YOLO_SPLIT if (YOLO_SPLIT.exists() and not QUICK_TEST) else (SPLIT_JSON if RESUME and SPLIT_JSON.exists() else None)
if use_split:
    SPLIT_JSON = use_split
    saved = json.loads(SPLIT_JSON.read_text())''')
s5 = must_replace(s5, '''# 5. 세션 단위 train / val 분리.''',
                  '''# 5. 세션 단위 train / val 분리. YOLO 학습의 split.json이 있으면 그대로 써서 YOLO와 같은 사진으로 학습·채점합니다.''')
code(s5)

code(y[6])

code(r'''
# 7. 사진 단위 정답과 학습.
# 정답 = 그 사진에 클래스별 박스가 하나라도 있는지(0/1). 파손 없는 사진은 [0, 0].
# outsole 사진은 YOLO 노트북처럼 학습에 REPEAT_OUTSOLE번 넣습니다(드문 클래스 반복 샘플링).
# 학습 중에는 검증하지 않고 마지막 가중치(last.pt)를 씁니다. 끊기면 다시 실행하면 처음부터 학습합니다.
import math
import time

import numpy as np
import timm
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

RUN = OUT_DIR / "runs_cls" / NAME
RUN.mkdir(parents=True, exist_ok=True)
FINAL = RUN / "last.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def target(n):
    has = {b[0] for b in boxes[n]}
    return [float(i in has) for i in range(len(CLASSES))]


train_items = list(train_names) + [n for n in train_names if any(b[0] == 0 for b in boxes[n])
                                   for _ in range(REPEAT_OUTSOLE - 1)]
if QUICK_TEST:
    train_items = train_items[:120]
    EPOCHS = 2
print(f"학습 사진 {len(train_items)}장 (outsole 반복 포함) | "
      + " · ".join(f"{c} {int(sum(target(n)[i] for n in train_items))}" for i, c in enumerate(CLASSES)))

# 크게 자르면 작은 안감 파손이 잘려 나가므로 자르기는 85% 이상만 합니다. 좌우 뒤집기는 신발 좌우 짝과 같아 안전합니다.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMGSZ, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.25, 0.25, 0.15, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([transforms.Resize((IMGSZ, IMGSZ)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Photos(Dataset):
    def __init__(self, names, tf):
        self.names, self.tf = names, tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        with Image.open(IMG_DIR / n) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor(target(n))


def make_model(pretrained):
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=len(CLASSES))


if FINAL.exists() and not QUICK_TEST:
    print("이미 끝난 학습입니다. 그대로 8번부터 채점합니다:", FINAL)
else:
    torch.manual_seed(SEED)
    dl = DataLoader(Photos(train_items, train_tf), batch_size=BATCH, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
    net = make_model(True).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    total, warm = EPOCHS * len(dl), len(dl)          # 첫 epoch 동안 lr을 서서히 올린 뒤 코사인으로 내립니다.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        run_loss, te = 0.0, time.time()
        for x, t in dl:
            x, t = x.to(DEVICE, non_blocking=True), t.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = loss_fn(net(x), t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(x)
        msg = f"epoch {ep}/{EPOCHS}  loss {run_loss / (len(dl) * BATCH):.4f}  {time.time() - te:.0f}초"
        if ep == 1:
            msg += f"  (전체 예상 {(time.time() - te) * EPOCHS / 60:.0f}분)"
        print(msg)

    torch.save({"model": BACKBONE, "classes": CLASSES, "img_size": IMGSZ, "multilabel": True,
                "threshold": THRESH, "state_dict": net.state_dict()}, FINAL)
    print(f"학습 {(time.time() - t0) / 60:.0f}분 | 모델: {FINAL}")
    del net, opt
''')

code(r'''
# 8. 같은 검증 사진 · 같은 정답으로 분류 모델과 YOLO26을 '사진 단위'로 채점합니다.
#  - 분류 모델 점수 = 클래스별 확률, YOLO 점수 = 그 사진에서 그 클래스 박스의 최고 conf (박스 없으면 0)
#  - AP / AUROC: 기준값 없이 순서만 보는 점수(0~1). 기준값을 바꿔도 변하지 않아 두 모델을 공정하게 비교합니다.
#  - 찾은 비율 @ 잘못 경보 10%: 파손 없는 사진의 10%에만 경보가 나도록 기준을 맞췄을 때 파손 사진을 찾은 비율
#  - 신발(세션) 단위: 세션 점수 = 그 세션 사진들의 최고 점수. 한 장이라도 찾으면 찾은 것(앱이 판단하는 방식)
#  - 차이의 90% 구간: 검증 세션을 1,000번 다시 뽑아(세션 단위 부트스트랩) 계산. 0을 걸치면 우연일 수 있는 차이입니다.
import gc

import pandas as pd
from IPython.display import display
from sklearn.metrics import average_precision_score, roc_auc_score

gc.collect()
torch.cuda.empty_cache()
VAL = sorted(val_names)
Y = np.array([target(n) for n in VAL])                 # [사진, 클래스] 정답 0/1


@torch.no_grad()
def predict_cls(names):
    ck = torch.load(FINAL, map_location="cpu", weights_only=False)
    net = make_model(False)
    net.load_state_dict(ck["state_dict"])
    net.to(DEVICE).eval()
    dl = DataLoader(Photos(names, eval_tf), batch_size=32, num_workers=2)
    out = []
    for x, _ in dl:
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(torch.sigmoid(net(x.to(DEVICE)).float()).cpu())
    del net
    return torch.cat(out).numpy()


def predict_yolo(names, weights):
    from ultralytics import YOLO
    net = YOLO(str(weights))
    s = np.zeros((len(names), len(CLASSES)))
    idx = {n: i for i, n in enumerate(names)}
    for i in range(0, len(names), 8):
        chunk = [str(IMG_DIR / n) for n in names[i:i + 8]]
        for res in net.predict(chunk, imgsz=YOLO_IMGSZ, conf=0.01, verbose=False):
            k = idx[Path(res.path).name]
            for c, conf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                s[k, int(c)] = max(s[k, int(c)], conf)
    del net
    return s


def recall_at_fa(y, s, fa=0.10):
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(np.floor(fa * len(neg))))]
    return float((s[y == 1] > thr).mean())


def safe(f, y, s):
    return float(f(y, s)) if 0 < y.sum() < len(y) else float("nan")


SESS = np.array([session_of[n] for n in VAL])


def by_session(y, s):
    keys = np.unique(SESS)
    return (np.array([y[SESS == k].max() for k in keys]), np.array([s[SESS == k].max() for k in keys]))


def table(S, name):
    rows = []
    cols = [(c, Y[:, i], S[:, i]) for i, c in enumerate(CLASSES)] + [("파손 있음(둘 중 하나)", Y.max(1), S.max(1))]
    for c, y, s in cols:
        ys, ss = by_session(y, s)
        rows.append({"모델": name, "클래스": c, "검증 파손 사진": int(y.sum()),
                     "AP": safe(average_precision_score, y, s), "AUROC": safe(roc_auc_score, y, s),
                     "찾음@경보10%": recall_at_fa(y, s),
                     "신발 AP": safe(average_precision_score, ys, ss), "신발 찾음@경보10%": recall_at_fa(ys, ss)})
    return rows


S_cls = predict_cls(VAL)
results = {"cls": S_cls}
yolo_w = OUT_DIR / "runs" / YOLO_RUN / "weights" / "last.pt"
if yolo_w.exists() and not QUICK_TEST:
    results["yolo"] = predict_yolo(VAL, yolo_w)
else:
    print("YOLO last.pt가 없어 분류 모델만 채점합니다:", yolo_w)

df = pd.DataFrame([r for k, S in results.items() for r in table(S, {"cls": NAME, "yolo": YOLO_RUN}[k])])
pd.set_option("display.float_format", "{:.3f}".format)
display(df)

# 앱 관점 3갈래(outsole / heel lining / 그 외) — THRESH 기준. 두 부위가 다 있는 사진은 확률이 높은 쪽을 맞힌 것으로 봅니다.
def three_way(y_row, s_row):
    truth = "그 외" if y_row.max() == 0 else None
    pred = "그 외" if s_row.max() < THRESH else CLASSES[int(s_row.argmax())]
    if truth is None:
        truth = pred if pred in [c for i, c in enumerate(CLASSES) if y_row[i]] else CLASSES[int(np.argmax(y_row))]
    return truth, pred


tw = pd.DataFrame([three_way(Y[i], S_cls[i]) for i in range(len(VAL))], columns=["정답", "예측"])
print(f"\n3갈래 정확도 {(tw['정답'] == tw['예측']).mean():.3f} (THRESH = {THRESH})")
display(pd.crosstab(tw["정답"], tw["예측"]))

# 세션 부트스트랩: 분류 AP − YOLO AP의 90% 구간
boot = {}
if "yolo" in results:
    rng = np.random.default_rng(SEED)
    keys = np.unique(SESS)
    members = {k: np.where(SESS == k)[0] for k in keys}
    for i, c in enumerate(CLASSES):
        diffs = []
        for _ in range(1000):
            ix = np.concatenate([members[k] for k in rng.choice(keys, len(keys), replace=True)])
            y = Y[ix, i]
            if 0 < y.sum() < len(y):
                diffs.append(average_precision_score(y, results["cls"][ix, i]) - average_precision_score(y, results["yolo"][ix, i]))
        lo, hi = np.percentile(diffs, [5, 95])
        boot[c] = [float(lo), float(hi)]
        side = "분류가 낫다" if lo > 0 else ("YOLO가 낫다" if hi < 0 else "0을 걸침(우연일 수 있음)")
        print(f"{c}: 사진 AP 차이(분류 − YOLO) 90% 구간 [{lo:+.3f}, {hi:+.3f}] → {side}")

summary = {"name": NAME, "backbone": BACKBONE, "imgsz": IMGSZ, "epochs": EPOCHS, "threshold": THRESH,
           "split": str(SPLIT_JSON), "n_val": len(VAL), "table": df.to_dict("records"),
           "three_way_acc": float((tw["정답"] == tw["예측"]).mean()), "ap_diff_90ci": boot}
(RUN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print("\n저장:", RUN / "summary.json")
print(json.dumps({k: summary[k] for k in ("name", "n_val", "three_way_acc", "ap_diff_90ci")}, ensure_ascii=False))
''')

code(r'''
# 9. 분류 모델이 놓친 사진 / 잘못 경보한 사진 (각 클래스 상위 8장). 제목은 Colab 폰트에 한글이 없어 영어로 씁니다.
import matplotlib.pyplot as plt


def grid(names, title):
    if not names:
        print("없음:", title)
        return
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, n in zip(axes.flat, names):
        with Image.open(IMG_DIR / n) as im:
            ax.imshow(im.convert("RGB"))
        k = VAL.index(n)
        ax.set_title(f"{n}\n" + " ".join(f"{c[:4]} {S_cls[k, i]:.2f}" for i, c in enumerate(CLASSES)), fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


for i, c in enumerate(CLASSES):
    missed = sorted([n for k, n in enumerate(VAL) if Y[k, i] and S_cls[k, i] < THRESH], key=lambda n: S_cls[VAL.index(n), i])
    alarm = sorted([n for k, n in enumerate(VAL) if not Y[k, i] and S_cls[k, i] >= THRESH], key=lambda n: -S_cls[VAL.index(n), i])
    print(f"{c}: 놓침 {len(missed)}장 / 잘못 경보 {len(alarm)}장")
    grid(missed[:8], f"{c}: missed")
    grid(alarm[:8], f"{c}: false alarm (check: unlabeled damage?)")
''')

md(r'''
## 결과 읽는 법

- **8번 표**의 `AP`가 사진 단위 핵심 점수입니다(0~1, 높을수록 좋음). `신발 AP`는 세션(신발 한 켤레) 단위입니다.
- YOLO 줄은 박스 모델을 "그 사진에서 가장 확신한 박스 점수"로 바꿔 같은 방식으로 채점한 것입니다.
  박스 위치는 보지 않으므로, **위치가 필요 없는 '이 사진에 어떤 수선이 필요한가' 판단만** 비교합니다.
- **결정 규칙**: 사진 단위 AP 차이의 90% 구간이 0을 걸치면 성능이 비슷한 것으로 보고, 위치까지 주는 검출 모델을 씁니다.
  분류 모델이 확실히 이길 때(구간 전체가 +)만 앱의 1차 필터로 검토합니다.
- 서빙: `damage_model/cls_serving/`의 `serve.py --ckpt last.pt`로 같은 가중치를 API로 띄우고 `shoe_app.py`로 시험합니다.
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

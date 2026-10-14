"""멘디 파손 검출 YOLO26 Colab 노트북 만들기. 셀 소스를 여기서 관리하고 .ipynb로 씁니다."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mendii_damage_yolo26_colab.ipynb")

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(r'''
# 멘디 파손 검출 — YOLO26 학습과 성능 (Colab GPU)

CVAT 프로젝트 `mendii_sneakers`의 파손 박스(**outsole** = 밑창 벌어짐, **heel lining** = 뒤꿈치 안감 손상)로
YOLO26 검출 모델을 학습하고, 학습에 쓰지 않은 촬영 세션의 사진으로 성능을 봅니다.

**실행 순서**: 런타임 → 런타임 유형 변경 → **T4 GPU** → 런타임 → **모두 실행**. 3번 셀에서 파일을 올리라고 하면 한꺼번에 골라 올립니다.

**올릴 파일 3개** (`다운로드` 폴더, `outsole_relabel.json`은 `Desktop/mendii/colab/`)
- CVAT 프로젝트 최종 export — `mendii_shoes_20261014_0900.zip` (2024년 사진 12,599장, job 24·25 수정 포함)
- 밑창 재라벨 — `outsole_relabel.json` (outsole 박스를 "벌어짐만 · 넉넉한 박스" 기준으로 다시 그린 것. 그 사진들의 outsole 박스를 통째로 바꿉니다)
- S3 읽기 전용 키 — `mendii-reader_accessKeys.csv` (읽은 뒤 Colab에서 바로 지웁니다)

프로젝트 export(사진이 가장 많은 zip)를 바탕으로 씁니다. 그보다 나중에 받은 job export를 같이 올리면 그 사진들의 박스를 job export로 바꿉니다(최종본에는 필요 없음).
`MyDrive/mendii/yolo26_damage/`에 파일을 미리 넣어 두면 올리지 않고 거기서 찾습니다.
키는 Colab 비밀(왼쪽 🔑, 이름 `AWS_ACCESS_KEY_ID` · `AWS_SECRET_ACCESS_KEY`)에 넣어도 됩니다.

**사진**은 S3 `mendii-sneakers`(서울)에서 원본(3456px)을 받아 1280px로 줄여 씁니다. 처음 한 번은 받는 데 10~20분 걸리고,
줄인 사진은 Drive에 zip 한 개(약 1GB)로 저장해 두어 다음부터는 바로 씁니다. 사진에 작업지시서(개인정보)가 찍혀 있으니
**결과 그림이 남은 노트북이나 Drive 폴더를 남에게 공유하지 마세요.**

**어떤 사진을 쓰나**: CVAT에는 아직 검수하지 않은 task(2025년 전부, 2024년 일부)도 들어 있습니다.
박스가 없는 사진을 전부 "파손 없음"으로 쓰면 아직 안 본 파손 사진을 정답 없음으로 가르치게 되므로,
박스가 달린 사진 사이 간격이 `REVIEW_GAP`(100장) 이하인 구간과 다 검수한 job 24·25만 "검수된 사진"으로 봅니다.

**검증 분리**: 같은 신발을 연달아 찍은 사진이 학습과 검증에 나뉘면 점수가 부풀므로, 촬영 시각 간격 300초 이하를 한 세션으로 묶어
세션 단위로 20%를 검증으로 뺍니다. 검증에는 그 세션의 검수된 사진을 모두(파손 없음 포함) 넣어 실제와 비슷하게 채점합니다.
학습 중에는 검증을 하지 않고 정해진 epoch까지 학습한 **마지막 모델**로 채점합니다. 검증 사진으로 가장 좋은 epoch를 고르면 점수가 부풀기 때문입니다.

결과(`last.pt`, 그래프, 성능 요약 `summary.json`, 분리 목록 `split.json`)는 Drive `MyDrive/mendii/yolo26_damage/runs/`에 저장되므로 런타임이 끊겨도 남습니다.
라이선스: YOLO26(Ultralytics)은 AGPL-3.0입니다. 앱 서버에 넣어 서비스하려면 소스 공개 또는 Ultralytics 기업 라이선스가 필요합니다.
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
!pip install -q "ultralytics==8.4.169" boto3

import ultralytics
ultralytics.checks()
''')

code(r'''
# 3. 설정과 입력 파일. 바꿀 것은 이 셀 위쪽의 대문자 값뿐입니다.
DRIVE_DIR   = "MyDrive/mendii/yolo26_damage"   # 결과·사진 캐시를 저장할 Drive 폴더
BUCKET      = "mendii-sneakers"
REGION      = "ap-northeast-2"
STORE_SIZE  = 1280      # S3 원본(3456px)을 이 크기로 줄여 저장합니다.
REVIEW_GAP  = 100       # 박스 달린 사진 사이가 이 장수보다 멀면 그 사이는 아직 검수 안 한 것으로 봅니다.
SESSION_GAP = 300       # 촬영 간격이 이 초 이하면 같은 신발(같은 세션)로 봅니다.
VAL_RATIO   = 0.2       # 검증으로 뺄 세션 비율
NEG_PER_POS = 1.0       # 학습에 넣을 '파손 없음' 사진 수 = 파손 사진 수 × 이 값 (검증에는 전부 넣음)
REPEAT_OUTSOLE = 2      # outsole(밑창 벌어짐) 사진을 학습에 몇 번 넣을지. 드문 클래스라 2번 넣습니다(LVIS식 반복 샘플링).
SEED        = 0

# 학습. 안감 파손 박스가 작아서(중앙값이 사진 한 변의 7%) 640px보다 1024px이 유리합니다.
MODEL   = "yolo26s.pt"  # 더 크게: yolo26m.pt (계산량 3.3배, T4에서 BATCH 8)
EPOCHS  = 60            # 최종본(학습 사진 약 4,350장) 기준 T4에서 epoch당 약 3.3분 → 약 3.3시간. 끊기면 RESUME으로 이어 갑니다.
IMGSZ   = 1024
BATCH   = 16
NAME    = "yolo26s_1024_final_e60"    # 결과 폴더 이름. 설정을 바꿔 다시 학습하면 이름도 바꾸세요.
RESUME  = False         # 끊긴 학습을 이어 갈 때 True로 바꾸고 모두 실행 (같은 train/val 분리를 다시 씁니다)
SPLIT_FROM = "yolo26s_1024"   # 첫 학습의 검증 세션은 그대로 검증에 두고, 그 학습에 쓴 세션은 검증에 넣지 않습니다 (없으면 새로 나눔)
QUICK_TEST = False      # True면 사진 180장·2 epoch로 노트북 전체가 도는지만 5분 안에 확인합니다.

# 다 검수했다고 확인된 CVAT job (첫 사진, 마지막 사진). 박스가 적어도 이 범위는 검수된 사진입니다.
KNOWN_REVIEWED = {
    "job 24": ("20240819_093048.jpg", "20240902_133744.jpg"),
    "job 25": ("20240903_092559.jpg", "20240913_154100.jpg"),
}

import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

if QUICK_TEST:
    EPOCHS, IMGSZ, BATCH, NAME = 2, 320, 8, "quick_test"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content")
    OUT_DIR = Path("/content/drive") / DRIVE_DIR
else:                                   # Colab 밖(로컬 GPU 등)에서 돌릴 때
    ROOT = Path(os.environ.get("MENDII_ROOT", "mendii_yolo26")).resolve()
    OUT_DIR = ROOT / "out"
ROOT.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANN = "annotations/instances_default.json"


def read_cvat(p):
    """CVAT COCO export면 (export 시각, coco) 아니면 None."""
    try:
        with zipfile.ZipFile(p) as z:
            return z.getinfo(ANN).date_time, json.loads(z.read(ANN))
    except (zipfile.BadZipFile, KeyError, OSError):
        return None


def find_inputs():
    places = [ROOT, OUT_DIR]
    found = {}
    for p in sorted(q for d in places for q in d.glob("*.zip")):
        r = read_cvat(p)
        if r and p.name not in found:
            found[p.name] = (p, *r)
    keys = [p for d in places for p in d.glob("*.csv") if "accesskey" in p.name.lower()]
    return list(found.values()), (max(keys, key=lambda p: p.stat().st_mtime) if keys else None)


def find_relabel():
    ps = [p for d in (ROOT, OUT_DIR) for p in d.glob("outsole_relabel*.json")]
    return max(ps, key=lambda p: p.stat().st_mtime) if ps else None


def secret(name):
    if not IN_COLAB:
        return os.environ.get(name)
    from google.colab import userdata
    try:
        return userdata.get(name)
    except Exception:                    # 비밀이 없거나 이 노트북에 접근 권한을 안 준 경우
        return None


PROJECT_MIN_IMAGES = 5000   # 프로젝트 전체 export는 사진 18,168장, job export는 400여 장입니다.


def missing_inputs(exports, key_csv):
    miss = []
    if not any(len(e[2]["images"]) >= PROJECT_MIN_IMAGES for e in exports):
        miss.append("CVAT 프로젝트 최종 export (mendii_shoes_20261014_0900.zip, 사진 12,599장)")
    if not all(AWS_KEY) and key_csv is None:
        miss.append("S3 키 (mendii-reader_accessKeys.csv)")
    if find_relabel() is None:
        miss.append("밑창 재라벨 (outsole_relabel.json)")
    return miss


AWS_KEY = (secret("AWS_ACCESS_KEY_ID"), secret("AWS_SECRET_ACCESS_KEY"))
exports, key_csv = find_inputs()
for k in range(3):                       # 꼭 필요한 파일(프로젝트 export·키)이 올라올 때까지 다시 묻습니다.
    miss = missing_inputs(exports, key_csv)
    blocking = [m for m in miss if not m.startswith("밑창")]
    if not miss or not IN_COLAB or (k > 0 and not blocking):
        break
    from google.colab import files
    print("올릴 파일 (여러 개를 한꺼번에 골라도 됩니다):\n  - " + "\n  - ".join(miss))
    print("이미 있는 것:", [e[0].name for e in exports] or "없음")
    for name, data in files.upload().items():
        (ROOT / name).write_bytes(data)
    exports, key_csv = find_inputs()
miss = missing_inputs(exports, key_csv)
assert not [m for m in miss if not m.startswith("밑창")], "다음 파일이 없습니다: " + ", ".join(miss)
RELABEL = find_relabel()
print("밑창 재라벨:", RELABEL.name if RELABEL else "⚠️ 없음 — outsole은 CVAT 박스 그대로(기준이 섞여 있음)")

# 사진이 가장 많은 zip = 프로젝트 export(바탕). 그보다 나중에 받은 job export는 그 사진들의 박스를 바꿉니다.
main_path, main_time, coco = max(exports, key=lambda e: (len(e[2]["images"]), e[1]))
overrides = sorted((e for e in exports if e[0] != main_path and e[1] > main_time), key=lambda e: e[1])
ignored = [e[0].name for e in exports if e[0] != main_path and e[1] <= main_time]
fmt = lambda t: "%04d-%02d-%02d %02d:%02d" % t[:5]
print(f"바탕: {main_path.name} (export {fmt(main_time)} UTC) | 사진 {len(coco['images'])} | 박스 {len(coco['annotations'])}"
      f" | 라벨 {[c['name'] for c in coco['categories']]}")
for p, t, c in overrides:
    print(f"덮어쓰기: {p.name} (export {fmt(t)} UTC) | 사진 {len(c['images'])} | 박스 {len(c['annotations'])}")
if ignored:
    print("바탕보다 먼저 받은 export라 쓰지 않음:", ignored)
if not overrides and main_time < (2026, 10, 12, 17, 43, 0):
    print("⚠️ 10/12 이전 export입니다. job 24는 복원 전 상태일 수 있으니 job 24·25 수정본(10/12 export)을 같이 올리세요.")

if not all(AWS_KEY):
    assert key_csv is not None, "S3 키(mendii-reader_accessKeys.csv)를 찾지 못했습니다."
    row = next(csv.DictReader(io.StringIO(key_csv.read_text(encoding="utf-8-sig"))))
    AWS_KEY = (row["Access key ID"].strip(), row["Secret access key"].strip())
    if key_csv.parent == ROOT and IN_COLAB:
        key_csv.unlink()                 # 런타임 디스크에 키를 남기지 않습니다.
''')

code(r'''
# 4. 검수된 사진 고르기, job export로 덮어쓰기, COCO → YOLO 박스 변환.
from collections import Counter, defaultdict

CLASSES = ["outsole", "heel lining"]                      # YOLO 클래스 0, 1


def labels_by_name(c):
    """file_name -> [(라벨 이름, [x, y, w, h])]. 박스 없는 사진도 빈 목록으로 들어갑니다."""
    cat = {x["id"]: x["name"] for x in c["categories"]}
    ims = {im["id"]: im for im in c["images"]}
    out = {im["file_name"]: [] for im in c["images"]}
    for a in c["annotations"]:
        out[ims[a["image_id"]]["file_name"]].append((cat[a["category_id"]], a["bbox"]))
    return out


size = {im["file_name"]: (im["width"], im["height"]) for im in coco["images"]}
labels = labels_by_name(coco)

# 프로젝트 export의 사진 id 순서 = task·프레임 순서(대체로 촬영 순서). 박스가 달린 사진의 위치로 검수 구간을 추정합니다.
order = [im["file_name"] for im in sorted(coco["images"], key=lambda im: im["id"])]
hit = [k for k, n in enumerate(order) if labels[n]]
runs, start = [], hit[0]
for prev, k in zip(hit, hit[1:]):
    if k - prev > REVIEW_GAP:
        runs.append((start, prev))
        start = k
runs.append((start, hit[-1]))
reviewed = {n for s, e in runs for n in order[s:e + 1]}
for lo, hi in KNOWN_REVIEWED.values():
    reviewed |= {n for n in order if lo <= n <= hi}

# 나중에 받은 job export가 있으면 그 사진들의 박스를 통째로 바꾸고, 검수된 사진으로 봅니다.
for p, t, c in overrides:
    new = labels_by_name(c)
    changed = sum(sorted(map(str, labels.get(n, []))) != sorted(map(str, b)) for n, b in new.items())
    labels.update(new)
    size.update({im["file_name"]: (im["width"], im["height"]) for im in c["images"]})
    reviewed |= set(new)
    print(f"{p.name}: 사진 {len(new)}장 중 {changed}장의 박스가 바뀜")

# 밑창 재라벨: 그 사진들의 outsole 박스를 통째로 바꿉니다(빈 목록 = outsole 없음). heel lining 박스는 그대로 둡니다.
# 재라벨에서 '판단 불가'로 남긴 사진은 학습·채점에서 빼고, '파손 아님'으로 확인한 사진은 5번 셀에서 음성으로 먼저 씁니다.
unsure, hard_neg = set(), {}
if RELABEL:
    rl = json.loads(RELABEL.read_text())
    unsure = set(rl.get("uncertain", []))
    hard_neg = {f: c.split(":")[-1] for f, c in rl.get("category", {}).items() if c.split(":")[-1] in ("deleted", "none")}
    n_box = n_changed = 0
    for n, bs in rl["images"].items():
        if n not in size:
            continue
        W, H = size[n]
        old = sorted(round(v) for name, b in labels.get(n, []) if name == "outsole" for v in b)
        new = [("outsole", [x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H]) for x0, y0, x1, y1 in bs]
        labels[n] = [(name, b) for name, b in labels.get(n, []) if name != "outsole"] + new
        reviewed.add(n)
        n_box += len(new)
        n_changed += old != sorted(round(v) for _, b in new for v in b)
    print(f"{RELABEL.name}: 사진 {len(rl['images'])}장 → outsole 박스 {n_box}개 (바뀐 사진 {n_changed}장) | 기준: {rl.get('rule', '')}")

unknown = sorted({name for bs in labels.values() for name, _ in bs} - set(CLASSES) - {"not_sneakers"})
assert not unknown, f"모르는 라벨이 있습니다: {unknown}. CLASSES를 고치세요."

boxes = {}                                   # file_name -> [(cls, xc, yc, w, h)] (0~1 비율)
dropped = Counter()
for n in sorted(reviewed):
    W, H = size[n]
    if any(name == "not_sneakers" for name, _ in labels[n]):   # 운동화가 아닌 사진은 빼고 학습합니다.
        dropped["not_sneakers 사진"] += 1
        continue
    if n in unsure and not any(name == "outsole" for name, _ in labels[n]):
        dropped["판단 불가(재라벨 uncertain)"] += 1
        continue
    out = []
    for name, (x, y, w, h) in labels[n]:
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            dropped["너무 작은 박스"] += 1
            continue
        out.append((CLASSES.index(name), (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    boxes[n] = out

n_pos = sum(1 for b in boxes.values() if b)
print(f"\n검수 구간 {len(runs)}개 + 확인된 job → 검수된 사진 {len(boxes)}장 (파손 박스 있음 {n_pos} / 없음 {len(boxes) - n_pos})")
print(f"제외: 검수 안 된 사진 {len(size) - len(reviewed)}장, {dict(dropped)}")
print("월별 검수된 사진:", dict(sorted(Counter(n[:6] for n in boxes).items())))
print("박스 수:", {c: sum(b[0] == i for bs in boxes.values() for b in bs) for i, c in enumerate(CLASSES)})
for job, (lo, hi) in KNOWN_REVIEWED.items():    # CVAT 화면의 박스 수와 같은지 확인하세요.
    names = [n for n in boxes if lo <= n <= hi]
    print(f"{job}: 사진 {len(names)}장, 박스 {sum(len(boxes[n]) for n in names)}개")
''')

code(r'''
# 5. 세션 단위 train / val 분리.
# 세션을 (밑창 박스 있음, 안감 박스 있음) 네 종류로 나누고 종류마다 세션의 VAL_RATIO만큼을 val로 보냅니다.
# 그래야 적은 밑창 클래스도 val에 고르게 들어갑니다. 분리 목록은 Drive에 저장하고, RESUME 때는 그대로 다시 씁니다.
import random
from datetime import datetime


def taken_at(name):
    return datetime.strptime(name[:15], "%Y%m%d_%H%M%S")


# 세션은 검수 안 된 사진까지 포함한 전체 촬영 순서로 끊습니다.
all_names = sorted(size, key=lambda n: (taken_at(n), n))
session_of, sid, prev = {}, 0, None
for n in all_names:
    t = taken_at(n)
    if prev is None or (t - prev).total_seconds() > SESSION_GAP:
        sid += 1
    session_of[n] = sid
    prev = t

SPLIT_JSON = OUT_DIR / "runs" / NAME / "split.json"
inputs_now = [main_path.name] + [p.name for p, _, _ in overrides] + ([RELABEL.name] if RELABEL else [])
if RESUME and SPLIT_JSON.exists():
    saved = json.loads(SPLIT_JSON.read_text())
    train_names = [n for n in saved["train"] if n in boxes]
    val_names = [n for n in saved["val"] if n in boxes]
    print(f"저장된 분리를 다시 씁니다: {SPLIT_JSON}")
    if saved["inputs"] != inputs_now:
        print("⚠️ 처음 학습 때와 CVAT 파일이 다릅니다:", saved["inputs"], "→", inputs_now)
else:
    sessions = defaultdict(list)
    for n in boxes:
        sessions[session_of[n]].append(n)
    strata = defaultdict(list)
    for s, names in sessions.items():
        has = {b[0] for n in names for b in boxes[n]}
        strata[(0 in has, 1 in has)].append(s)

    rng = random.Random(SEED)
    prev_split = OUT_DIR / "runs" / SPLIT_FROM / "split.json"
    old_val, old_train = set(), set()
    if SPLIT_FROM and prev_split.exists() and not QUICK_TEST:
        prev = json.loads(prev_split.read_text())
        old_val = {session_of[n] for n in prev["val"] if n in session_of}
        old_train = {session_of[n] for n in prev["train"] if n in session_of} - old_val
    # 첫 학습의 검증 세션은 그대로 val, 첫 학습에 쓴 세션은 train, 나머지(새로 검수된 세션)만 층마다 VAL_RATIO씩 val로.
    val_sessions = set(old_val)
    for key in sorted(strata):
        ss = sorted(s for s in strata[key] if s not in old_val and s not in old_train)
        rng.shuffle(ss)
        val_sessions |= set(ss[:round(len(ss) * VAL_RATIO)])
    if old_val:
        print(f"{SPLIT_FROM}의 검증 세션 {len(old_val)}개 유지 + 새로 검수된 세션에서 {len(val_sessions) - len(old_val)}개 추가"
              f" (그 학습에 쓴 세션 {len(old_train)}개는 val에 넣지 않음)")

    val_names = sorted(n for n in boxes if session_of[n] in val_sessions)
    pos = sorted(n for n in boxes if session_of[n] not in val_sessions and boxes[n])
    neg_all = sorted(n for n in boxes if session_of[n] not in val_sessions and not boxes[n])
    n_neg = min(len(neg_all), round(len(pos) * NEG_PER_POS))
    # 음성은 재라벨에서 '파손 아님'으로 확인한 사진부터: 지운 오답(뒷굽 닳음·구멍 등) → 같은 신발의 다른 사진 → 나머지 무작위.
    tiers = [[n for n in neg_all if hard_neg.get(n) == "deleted"], [n for n in neg_all if hard_neg.get(n) == "none"],
             [n for n in neg_all if n not in hard_neg]]
    neg = []
    for tier in tiers:
        take = min(len(tier), n_neg - len(neg))
        neg += rng.sample(tier, take)
    neg = sorted(neg)
    print(f"음성 {len(neg)}장 = 지운 오답 {sum(hard_neg.get(n) == 'deleted' for n in neg)} + 같은 신발 확인 "
          f"{sum(hard_neg.get(n) == 'none' for n in neg)} + 무작위 {sum(n not in hard_neg for n in neg)}")
    if QUICK_TEST:
        pos, neg = pos[:90], neg[:30]
        val_names = [n for n in val_names if boxes[n]][:40] + [n for n in val_names if not boxes[n]][:20]
    train_names = sorted(pos + neg)
    print(f"세션 {len(sessions)}개 → val {len(val_sessions)}개 (train 파손 없음 후보 {len(neg_all)}장 중 {len(neg)}장 뽑음)")
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps({"inputs": inputs_now, "train": train_names, "val": val_names}))

split = {**{n: "train" for n in train_names}, **{n: "val" for n in val_names}}
assert not {session_of[n] for n in train_names} & {session_of[n] for n in val_names}, "같은 세션이 train과 val에 섞였습니다."

print(f"\n{'':<14}{'train':>8}{'val':>8}")
print(f"{'사진':<14}{len(train_names):>8}{len(val_names):>8}")
print(f"{'  파손 없음':<14}{sum(not boxes[n] for n in train_names):>8}{sum(not boxes[n] for n in val_names):>8}")
for i, c in enumerate(CLASSES):
    ti = sum(any(b[0] == i for b in boxes[n]) for n in train_names)
    vi = sum(any(b[0] == i for b in boxes[n]) for n in val_names)
    tb = sum(b[0] == i for n in train_names for b in boxes[n])
    vb = sum(b[0] == i for n in val_names for b in boxes[n])
    print(f"{c:<14}{ti:>8}{vi:>8}   사진 / 박스 {tb} · {vb}")
''')

code(r'''
# 6. S3 원본을 받아 STORE_SIZE로 줄여 저장. 한 번 받은 사진은 Drive의 zip 한 개로 저장해 다음부터 다시 받지 않습니다.
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from PIL import Image, ImageOps

IMG_DIR = ROOT / f"img{STORE_SIZE}"
IMG_DIR.mkdir(exist_ok=True)
CACHE_ZIP = OUT_DIR / f"img{STORE_SIZE}.zip"

need = sorted(split)
have = {p.name for p in IMG_DIR.glob("*.jpg")}
if CACHE_ZIP.exists() and not set(need) <= have:
    t0 = time.time()
    local = ROOT / CACHE_ZIP.name
    shutil.copy(CACHE_ZIP, local)            # Drive에서 바로 풀면 느려서 로컬로 복사한 뒤 풉니다.
    try:
        with zipfile.ZipFile(local) as z:
            z.extractall(IMG_DIR)
    except zipfile.BadZipFile:
        print("Drive 캐시 zip이 깨져 있어 S3에서 다시 받습니다.")
    local.unlink()
    have = {p.name for p in IMG_DIR.glob("*.jpg")}
    print(f"Drive 캐시에서 {len(have)}장 ({time.time() - t0:.0f}초)")

todo = [n for n in need if n not in have]
s3 = boto3.client("s3", region_name=REGION, aws_access_key_id=AWS_KEY[0], aws_secret_access_key=AWS_KEY[1],
                  config=Config(max_pool_connections=32, retries={"max_attempts": 8, "mode": "adaptive"}))


def fetch(name):
    body = s3.get_object(Bucket=BUCKET, Key=name)["Body"].read()
    im = Image.open(io.BytesIO(body))
    im.draft("RGB", (STORE_SIZE, STORE_SIZE))   # JPEG를 1/2 크기로 바로 풀어 빠르게 줄입니다.
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((STORE_SIZE, STORE_SIZE), Image.LANCZOS)
    tmp = IMG_DIR / (name + ".part")
    im.save(tmp, "JPEG", quality=90)
    tmp.rename(IMG_DIR / name)
    return len(body)


if todo:
    t0, done, nbytes, failed = time.time(), 0, 0, []
    with ThreadPoolExecutor(24) as ex:
        futs = {ex.submit(fetch, n): n for n in todo}
        for f in futs:
            try:
                nbytes += f.result()
            except Exception as e:           # 받기 실패한 사진은 학습에서 뺍니다.
                failed.append((futs[f], repr(e)[:120]))
            done += 1
            if done % 500 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)}장 {nbytes / 1e9:.1f}GB {el:.0f}초 (남은 시간 약 {el / done * (len(todo) - done) / 60:.0f}분)")
    if failed:
        print(f"받기 실패 {len(failed)}장 — 학습에서 뺍니다. 예:", failed[:3])
        assert len(failed) < 0.05 * len(todo), "실패가 너무 많습니다. 키·인터넷을 확인하고 이 셀을 다시 실행하세요."
        for n, _ in failed:
            split.pop(n, None)
    if not QUICK_TEST and len(failed) < len(todo):
        # 캐시 zip 다시 쓰기 (사진은 이미 JPEG라 압축 없이 묶기만). 다 쓴 뒤에 이름을 바꿔 깨진 zip이 남지 않게 합니다.
        tmp_zip = ROOT / "cache.zip"
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as z:
            for p in sorted(IMG_DIR.glob("*.jpg")):
                z.write(p, p.name)
        part = CACHE_ZIP.with_name(CACHE_ZIP.name + ".part")
        shutil.copy(tmp_zip, part)
        os.replace(part, CACHE_ZIP)
        tmp_zip.unlink()
        print("Drive 캐시 저장:", CACHE_ZIP, f"{CACHE_ZIP.stat().st_size / 1e9:.2f}GB")

train_names = [n for n in train_names if n in split]
val_names = [n for n in val_names if n in split]
print(f"준비 완료: train {len(train_names)}장 / val {len(val_names)}장")
''')

code(r'''
# 7. YOLO 폴더 구조와 data.yaml. 사진은 복사하지 않고 심볼릭 링크로 겁니다.
import yaml

DS = ROOT / "yolo_damage"
if DS.exists():
    shutil.rmtree(DS)
for part in ("train", "val"):
    (DS / "images" / part).mkdir(parents=True)
    (DS / "labels" / part).mkdir(parents=True)

for n, part in split.items():
    os.symlink(IMG_DIR / n, DS / "images" / part / n)
    lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes[n]]
    # 파손 없는 사진은 빈 라벨 파일 = 배경 사진으로 학습합니다.
    (DS / "labels" / part / (Path(n).stem + ".txt")).write_text("".join(l + "\n" for l in lines))

# outsole 사진은 이름만 바꾼 링크로 한 번 더 넣어 epoch마다 더 자주 보게 합니다(train만).
n_rep = 0
for n in [n for n in train_names if any(b[0] == 0 for b in boxes[n])]:
    for k in range(1, REPEAT_OUTSOLE):
        dup = f"{Path(n).stem}__r{k}"
        os.symlink(IMG_DIR / n, DS / "images" / "train" / (dup + ".jpg"))
        shutil.copy(DS / "labels" / "train" / (Path(n).stem + ".txt"), DS / "labels" / "train" / (dup + ".txt"))
        n_rep += 1
print(f"outsole 사진 {n_rep}장을 학습에 한 번 더 넣었습니다 (REPEAT_OUTSOLE = {REPEAT_OUTSOLE})")

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
# 8. 학습. 설정은 3번 셀에 있습니다.
# 학습 중에는 검증을 하지 않고(val=False) 마지막 epoch에만 한 번 채점합니다. 시간이 줄고, 검증 사진으로 모델을 고르지 않습니다.
# 끊겼다면 3번 셀에서 RESUME = True로 바꾸고 모두 실행하면 마지막 epoch부터 이어 갑니다.
from ultralytics import YOLO

project = OUT_DIR / "runs"
WEIGHTS = project / NAME / "weights" / "last.pt"

if RESUME and WEIGHTS.exists():
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if ckpt.get("optimizer") is None:        # 끝까지 학습한 모델은 이어 갈 것이 없습니다.
        print("이미 끝난 학습입니다. 그대로 9번부터 채점합니다.")
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
        workers=2,              # Colab 무료 CPU가 2개입니다.
        plots=True,
    )

print("모델:", WEIGHTS)
''')

code(r'''
# 9. 박스 단위 성능 (Ultralytics 기본 채점). 검증에는 파손 없는 사진도 들어 있어 잘못 그린 박스가 그대로 감점됩니다.
import gc

import pandas as pd
from IPython.display import Image as IPImage, display

gc.collect()                                 # 학습에 쓰던 GPU 메모리를 풉니다.
torch.cuda.empty_cache()

net = YOLO(str(WEIGHTS))
metrics = net.val(data=str(DATA_YAML), imgsz=IMGSZ, batch=BATCH, split="val",
                  project=str(ROOT / "val_runs"), name="val", exist_ok=True)

per_class = pd.DataFrame({
    "class": [CLASSES[i] for i in metrics.box.ap_class_index],
    "precision": metrics.box.p,
    "recall": metrics.box.r,
    "mAP50": metrics.box.ap50,
    "mAP50-95": metrics.box.ap,
}).round(3)
display(per_class)
print(f"전체 mAP50 {metrics.box.map50:.3f} / mAP50-95 {metrics.box.map:.3f}")

# 이전 학습(SPLIT_FROM) 모델도 지금 정답(재라벨 반영)으로 같은 검증 사진에서 채점해 나란히 봅니다.
prev_w = OUT_DIR / "runs" / SPLIT_FROM / "weights" / "last.pt"
if SPLIT_FROM and SPLIT_FROM != NAME and prev_w.exists() and not QUICK_TEST:
    pm = YOLO(str(prev_w)).val(data=str(DATA_YAML), imgsz=IMGSZ, batch=BATCH, split="val", plots=False,
                               project=str(ROOT / "val_runs"), name="val_prev", exist_ok=True, verbose=False)
    prev_ap = dict(zip(pm.box.ap_class_index.tolist(), pm.box.ap50.tolist()))
    display(pd.DataFrame({
        "class": per_class["class"],
        f"mAP50 {SPLIT_FROM}": [round(prev_ap.get(i, float("nan")), 3) for i in metrics.box.ap_class_index],
        f"mAP50 {NAME} (이번)": per_class["mAP50"],
    }))
    print(f"전체 mAP50 {SPLIT_FROM} {pm.box.map50:.3f} → 이번 {metrics.box.map50:.3f}  (같은 검증 사진·같은 정답)")

for png in ("confusion_matrix.png", "BoxPR_curve.png"):
    p = Path(metrics.save_dir) / png
    if p.exists():
        display(IPImage(filename=str(p), width=650))
''')

code(r'''
# 10. 앱에서 쓰는 방식으로 채점 (사진 한 장 단위).
#  - 파손 사진을 찾았나: 그 클래스 박스가 있는 사진에서 그 클래스 박스를 하나라도 냈나
#  - 잘못 경보: 그 클래스 박스가 없는 사진에서 그 클래스 박스를 냈나 (파손 없는 사진만 따로도 셈)
#  - 박스가 맞나: 정답 박스마다 같은 클래스 예측과 IoU가 0.3 / 0.5 이상인가
#  - 가장 자신 있는 박스의 중심이 파손 위인가 (앱의 지금 위치 모델 채점과 같은 방식)
import numpy as np

CONF_LIST = [0.25, 0.5]
PRED_BATCH = 16


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def gt_xyxy(n):
    return [(c, (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)) for c, xc, yc, w, h in boxes[n]]


preds = {}                                   # file_name -> [(cls, conf, (x0,y0,x1,y1) 0~1)]
for i in range(0, len(val_names), PRED_BATCH):
    chunk = [str(IMG_DIR / n) for n in val_names[i:i + PRED_BATCH]]
    for res in net.predict(chunk, imgsz=IMGSZ, conf=0.01, verbose=False):
        b = res.boxes
        preds[Path(res.path).name] = [(int(c), float(s), tuple(map(float, xy)))
                                      for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]


def auc(pos, neg):                           # 사진 점수(그 클래스 최고 conf)로 파손 사진 / 아닌 사진 가르기
    if not pos or not neg:
        return float("nan")
    s = np.array(pos + neg)
    r = s.argsort().argsort() + 1.0
    for v in np.unique(s):                   # 같은 점수는 평균 순위
        m = s == v
        r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


clean = [n for n in val_names if not boxes[n]]
rows, summary = [], {"val_images": len(val_names), "val_clean_images": len(clean),
                     "map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4),
                     "classes": {}}
for ci, cname in enumerate(CLASSES):
    has = [n for n in val_names if any(c == ci for c, _ in gt_xyxy(n))]
    has_set = set(has)
    not_has = [n for n in val_names if n not in has_set]
    score = {n: max([s for c, s, _ in preds[n] if c == ci], default=0.0) for n in val_names}
    cs = {"images_with_damage": len(has), "boxes": sum(c == ci for n in has for c, _ in gt_xyxy(n)),
          "image_auc": round(float(auc([score[n] for n in has], [score[n] for n in not_has])), 3)}
    for t in CONF_LIST:
        found = sum(score[n] >= t for n in has)
        false_any = sum(score[n] >= t for n in not_has)
        false_clean = sum(score[n] >= t for n in clean)
        hit30 = hit50 = n_gt = 0
        for n in has:
            pb = [xy for c, s, xy in preds[n] if c == ci and s >= t]
            for c, g in gt_xyxy(n):
                if c != ci:
                    continue
                n_gt += 1
                best_iou = max([iou_xyxy(g, p) for p in pb], default=0.0)
                hit30 += best_iou >= 0.3
                hit50 += best_iou >= 0.5
        cs[f"conf{t}"] = {"found_images": f"{found}/{len(has)}", "false_alarm_images": f"{false_any}/{len(not_has)}",
                          "false_alarm_clean": f"{false_clean}/{len(clean)}",
                          "box_iou30": f"{hit30}/{n_gt}", "box_iou50": f"{hit50}/{n_gt}"}
        rows.append({"class": cname, "conf": t,
                     "파손 사진 찾음": f"{found}/{len(has)} ({found / max(len(has), 1):.0%})",
                     "잘못 경보(그 파손 없는 사진)": f"{false_any}/{len(not_has)} ({false_any / max(len(not_has), 1):.1%})",
                     "잘못 경보(파손 없는 사진)": f"{false_clean}/{len(clean)} ({false_clean / max(len(clean), 1):.1%})",
                     "박스 IoU≥0.3": f"{hit30}/{n_gt} ({hit30 / max(n_gt, 1):.0%})",
                     "박스 IoU≥0.5": f"{hit50}/{n_gt} ({hit50 / max(n_gt, 1):.0%})",
                     "사진 AUC": cs["image_auc"]})
    summary["classes"][cname] = cs

# 가장 자신 있는 박스(클래스 무관)의 중심이 정답 박스 안인가 — 파손 사진만
dmg = [n for n in val_names if boxes[n]]
center_hit = 0
for n in dmg:
    if preds[n]:
        c, s, (x0, y0, x1, y1) = max(preds[n], key=lambda p: p[1])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        center_hit += any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for _, g in gt_xyxy(n))
summary["top_box_center_on_damage"] = f"{center_hit}/{len(dmg)}"

# 신발(세션) 단위: 앱은 한 켤레를 여러 장 찍어 올리므로, 같은 세션 사진 중 한 장이라도 박스를 내면 '찾음'으로 봅니다.
# 세션 = 촬영 간격 300초 이하로 이어진 사진 묶음(대개 신발 한 켤레, 가끔 여러 켤레).
by_sess = defaultdict(list)
for n in val_names:
    by_sess[session_of[n]].append(n)
sess_rows = []
summary["session_level"] = {}
for ci, cname in enumerate(CLASSES):
    pos_s = [s for s, ns in by_sess.items() if any(c == ci for n in ns for c, _ in gt_xyxy(n))]
    neg_s = [s for s, ns in by_sess.items() if all(not boxes[n] for n in ns)]
    top = {n: max([s for c, s, _ in preds[n] if c == ci], default=0.0) for n in val_names}
    for t in CONF_LIST:
        hit = sum(any(top[n] >= t for n in by_sess[s]) for s in pos_s)
        fa = sum(any(top[n] >= t for n in by_sess[s]) for s in neg_s)
        sess_rows.append({"class": cname, "conf": t, "신발(세션) 찾음": f"{hit}/{len(pos_s)}",
                          "잘못 경보(파손 없는 세션)": f"{fa}/{len(neg_s)}"})
        summary["session_level"][f"{cname}@{t}"] = {"found": f"{hit}/{len(pos_s)}", "false_alarm": f"{fa}/{len(neg_s)}"}

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
display(pd.DataFrame(rows))
display(pd.DataFrame(sess_rows))
print(f"가장 자신 있는 박스의 중심이 파손 위: {center_hit}/{len(dmg)} ({center_hit / max(len(dmg), 1):.0%})")

summary.update({"model": MODEL, "imgsz": IMGSZ, "epochs": EPOCHS, "inputs": inputs_now,
                "train_images": len(train_names), "train_damage_images": sum(bool(boxes[n]) for n in train_names),
                "review_gap": REVIEW_GAP, "session_gap": SESSION_GAP, "quick_test": QUICK_TEST, "repeat_outsole": REPEAT_OUTSOLE,
                "per_class_box": per_class.to_dict("records")})
(project / NAME / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
pd.DataFrame([{"file": n, "session": session_of[n], "gt": json.dumps(boxes[n]),
               "pred": json.dumps([p for p in preds[n] if p[1] >= 0.1])} for n in val_names]
             ).to_csv(project / NAME / "val_predictions.csv", index=False)
print("\n저장:", project / NAME / "summary.json")
print("아래 한 줄을 복사해 Claude에게 보내면 결과를 같이 볼 수 있습니다:\n")
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
''')

code(r'''
# 11. 틀린 사진 훑어보기 (초록 = 정답, 빨강 = 예측 conf≥0.25, O = outsole, H = heel lining).
# '잘못 경보'가 실제로는 라벨이 빠진 파손인 경우가 많습니다. 그런 사진은 CVAT에서 박스를 추가하면 다음 학습이 좋아집니다.
import cv2
import matplotlib.pyplot as plt

SHOW_CONF = 0.25


def draw(n):
    im = cv2.imread(str(IMG_DIR / n))[:, :, ::-1].copy()
    H, W = im.shape[:2]
    for c, g in gt_xyxy(n):
        cv2.rectangle(im, (int(g[0] * W), int(g[1] * H)), (int(g[2] * W), int(g[3] * H)), (0, 220, 0), 6)
    for c, s, p in preds[n]:
        if s >= SHOW_CONF:
            cv2.rectangle(im, (int(p[0] * W), int(p[1] * H)), (int(p[2] * W), int(p[3] * H)), (255, 40, 40), 4)
            cv2.putText(im, f"{'O' if c == 0 else 'H'} {s:.2f}", (int(p[0] * W) + 6, int(p[1] * H) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 40, 40), 3)
    return im


def grid(names, title):
    if not names:
        print(title, "— 없음")
        return
    cols = 4
    rows_ = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(4.2 * cols, 4.2 * rows_), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, n in zip(axes.flat, names):
        ax.imshow(draw(n))
        ax.set_title(n, fontsize=8)
    fig.suptitle(title)                      # Colab 기본 폰트에 한글이 없어 영어로 씁니다.
    plt.tight_layout()
    plt.show()


def top_score(n):
    return max([s for _, s, _ in preds[n]], default=0.0)


missed = [n for n in val_names if boxes[n] and top_score(n) < SHOW_CONF]
false_alarm = sorted([n for n in val_names if not boxes[n] and top_score(n) >= SHOW_CONF], key=top_score, reverse=True)
found = [n for n in val_names if boxes[n] and top_score(n) >= SHOW_CONF]
print(f"파손 사진 중 아무 박스도 못 낸 사진 {len(missed)}장 / 파손 없는 사진에 박스를 낸 사진 {len(false_alarm)}장")
grid(missed[:12], "Missed (GT green, nothing predicted)")
grid(false_alarm[:12], "False alarms on 'no damage' photos (check: unlabeled damage?)")
grid(found[:8], "Found (GT green, pred red)")
''')

md(r'''
## 결과 읽는 법

- **9번 표**는 Ultralytics 기본 채점입니다. `mAP50`은 박스가 정답과 절반 이상 겹치면 맞은 것으로 보는 점수(0~1)입니다.
- **10번 표**가 앱 관점입니다. "파손 사진 찾음"이 높고 "잘못 경보(파손 없는 사진)"가 낮아야 앱의 AI 추가 발견에 쓸 수 있습니다.
  `conf`를 올리면 잘못 경보가 줄고 찾는 비율도 줄어듭니다.
- 지금 앱의 위치 모델(SigLIP2 패치 선형, `app/ml/loc_probe`)은 시험지 42장에서 "가장 의심한 곳이 파손 위" 33/42였습니다.
  10번의 마지막 줄이 같은 방식이지만 시험 사진이 달라 대략만 비교됩니다.
- 성능을 더 올리려면: 11번에서 잘못 경보로 나온 사진 중 실제 파손을 CVAT에 박스로 추가 → 2025년 task 검수 → `IMGSZ = 1280` 또는 `yolo26m.pt`.
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

# 학습·인퍼런스·평가·전후 EDA 실행 가이드

> **Task2 변경 안내 (2026-10-14):** task2는 RF-DETR Small로 전환했으며 CPU의 짧은 학습·검증·인퍼런스·전후 EDA를 실행했습니다. 아래 문서의 task2 YOLO 관련 설정과 미실행 상태는 초기 기록입니다. 현재 설정은 [RF-DETR 실행 가이드](task2_rf_detr_guide.md), 실측 점검 결과는 [실행 보고서](task2_run_report.md)를 기준으로 확인하세요.

작성일: 2026-10-14

## 1. 우선 사용할 모델

두 task 모두 **COCO 사전학습 YOLO11n 객체 탐지 모델(`yolo11n.pt`)**을 별도로 미세조정합니다. 하나의 가중치를 두 task가 공유하는 구조가 아니라, task별 `best.pt`가 따로 생성됩니다. 작은 모델로 기준 성능과 실행 시간을 확인한 뒤, 검증 성능이 부족하면 `yolo11s.pt`와 비교합니다. YOLO11은 탐지용 사전학습 가중치와 학습·예측·검증 기능을 제공합니다. [공식 모델 문서](https://docs.ultralytics.com/models/yolo11/)

| 설정 | data1 | data2 |
|---|---|---|
| 목적 | 7종 신발 위치·종류 | 손상 부위 2종 + 원본 예외 클래스 |
| 기본 모델 | YOLO11n | YOLO11n |
| 클래스 수 | 7 | 3 (not_sneakers 포함) |
| 입력 크기 | 512 | 512 |
| 최대 epochs | 100 | 100 |
| early stopping patience | 20 | 20 |
| batch / workers | 8 / 0 | 8 / 0 |
| 증강 | 약한 회전·이동·밝기·색상, 좌우 반전, mosaic 0.3 | 같은 약한 증강, mosaic 0.1 |
| 최적 체크포인트 | 라이브러리 검증 fitness 기준 best.pt | 동일 |

test 성능으로 가중치를 고르지 않습니다. data2의 2종만 학습하는 실험은 별도 데이터 설정과 클래스 재매핑이 필요하며, 현재 코드의 기본 실험은 원본 3종입니다. 고해상도 손상 복원이 필요한 경우 실제 고해상도 사진 확보가 우선입니다. 512 사진을 더 큰 입력 크기로 확대해도 원본 세부 정보가 복원되지는 않습니다.

## 2. 코드 위치

각 task 폴더에 동일한 역할의 독립 실행 파일 5개를 만들었습니다. 내부 구현은 공통 모듈을 사용하여 평가 방식이 서로 달라지는 것을 방지합니다.

| 역할 | data1 | data2 |
|---|---|---|
| 학습 전 EDA | [pre_eda.py](../scripts/data1/pre_eda.py) | [pre_eda.py](../scripts/data2/pre_eda.py) |
| 학습 | [train.py](../scripts/data1/train.py) | [train.py](../scripts/data2/train.py) |
| 인퍼런스 | [infer.py](../scripts/data1/infer.py) | [infer.py](../scripts/data2/infer.py) |
| 평가·임계값 선택 | [evaluate.py](../scripts/data1/evaluate.py) | [evaluate.py](../scripts/data2/evaluate.py) |
| 학습 후 EDA | [post_eda.py](../scripts/data1/post_eda.py) | [post_eda.py](../scripts/data2/post_eda.py) |

공통 구현: [pipeline/run.py](../pipeline/run.py), [고정 임계값 평가 계산](../pipeline/metrics.py).

기존 루트 `train.py --task data1` 호출도 지원하지만 **이제 학습만 수행하며 테스트를 자동 실행하지 않습니다.**

## 3. 환경 준비

일반 Python 3.10 이상(3.11 권장)이 설치된 Windows PowerShell에서 다음을 실행합니다. `.tools/python311`은 휴대용 Python이며 venv를 만들 수 없습니다. 현재 이 Python에는 RF-DETR 실행 환경만 설치되어 있고 Ultralytics는 설치되어 있지 않습니다(9절 참고).

```powershell
cd C:\Users\user\Desktop\final\men_dii
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify_data.ps1
```

명령의 `python`이 Windows 설치 안내만 출력한다면 일반 Python을 설치하고 실제 실행 파일 경로로 venv를 생성합니다. NVIDIA GPU는 해당 장치에 맞는 CUDA 지원 PyTorch 환경을 먼저 준비하고, 아래 명령의 `--device cpu`를 `--device 0`으로 바꿉니다. CPU 실행도 가능하지만 학습에는 오래 걸릴 수 있습니다. 최초 학습에는 사전학습 가중치 다운로드가 필요합니다.

## 4. 실행 순서

**데이터 검증 → 학습 전 EDA → 라벨 샘플 검토 → 학습 → val 평가·임계값 선택 → val 결과 EDA·개선 → 설정 확정 → test 평가 → test 결과 EDA → 새 사진 인퍼런스** 순서입니다.

학습 전 EDA의 라벨 샘플과 중복 후보를 먼저 검토하세요. 테스트를 확인한 뒤 같은 테스트 점수를 보며 모델을 반복 수정하지 않습니다. 아래는 기준 실험 한 번의 예시이며, 개선 실험은 출력 이름을 바꾸고 검증 단계까지 반복합니다.

### data1 수동 실행

```powershell
$py = '.\.venv\Scripts\python.exe'
$run = 'artifacts/data1/baseline_v1'

# 1. 학습 전 EDA: 먼저 결과를 검토
& $py scripts/data1/pre_eda.py --output "$run/pre_eda"

# 2. 학습
& $py scripts/data1/train.py --device cpu --output "$run/train"
$weights = "$run/train/fit/weights/best.pt"

# 3. 검증에서 confidence 선택
& $py scripts/data1/evaluate.py --weights $weights --split val --select-threshold --device cpu --output "$run/val"

# 4. 검증 결과 분석
& $py scripts/data1/post_eda.py --evaluation "$run/val" --training "$run/train/fit" --output "$run/post_val"

# 5. 모델·임계값을 확정한 뒤 최종 테스트
& $py scripts/data1/evaluate.py --weights $weights --split test --threshold-config "$run/val/threshold.json" --device cpu --output "$run/test"

# 6. 테스트 결과 분석
& $py scripts/data1/post_eda.py --evaluation "$run/test" --training "$run/train/fit" --output "$run/post_test"

# 7. 새 이미지 또는 폴더 인퍼런스: 경로를 실제 입력으로 변경
& $py scripts/data1/infer.py --weights $weights --threshold-config "$run/val/threshold.json" --source 'C:/path/to/new_images' --device cpu --output "$run/inference"
```

### data2 수동 실행

```powershell
$py = '.\.venv\Scripts\python.exe'
$run = 'artifacts/data2/baseline_v1'

& $py scripts/data2/pre_eda.py --output "$run/pre_eda"
& $py scripts/data2/train.py --device cpu --output "$run/train"
$weights = "$run/train/fit/weights/best.pt"

& $py scripts/data2/evaluate.py --weights $weights --split val --select-threshold --device cpu --output "$run/val"
& $py scripts/data2/post_eda.py --evaluation "$run/val" --training "$run/train/fit" --output "$run/post_val"

& $py scripts/data2/evaluate.py --weights $weights --split test --threshold-config "$run/val/threshold.json" --device cpu --output "$run/test"
& $py scripts/data2/post_eda.py --evaluation "$run/test" --training "$run/train/fit" --output "$run/post_test"

& $py scripts/data2/infer.py --weights $weights --threshold-config "$run/val/threshold.json" --source 'C:/path/to/new_sneaker_images' --device cpu --output "$run/inference"
```

`threshold.json`의 `constraints_met`를 확인합니다. false라면 val에서 Recall 목표를 만족하는 임계값이 없어 탐색용 대체 임계값이 저장된 상태입니다. 개선 후 다시 검증하며, 이 상태를 목표 달성으로 해석하지 않습니다.

임계값은 0.05부터 0.95까지 0.05 간격으로 탐색합니다. data1은 sneakers Recall ≥ 95%인 후보 중 지원 클래스 Macro-F1을 최대화하고, data2는 손상 2종 각각 Recall ≥ 90%인 후보 중 두 종류의 Macro-Precision을 최대화합니다. 없는 클래스는 평균에서 제외하고 포함 클래스를 기록합니다.

### 선택 사항: 단계 자동 실행

EDA 사전 검토를 마친 후 [run_task.ps1](../run_task.ps1)을 사용할 수 있습니다. 두 task는 별도로 실행합니다. 자동 실행은 val Recall 조건을 만족하지 못하면 post_val까지 저장하고 테스트 전에 중단합니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_task.ps1 -Task data1 -Device cpu -RunName auto_v1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_task.ps1 -Task data2 -Device cpu -RunName auto_v1
```

`-Epochs 1 -RunName smoke_v1`은 실행 경로를 확인하는 짧은 실험에 사용할 수 있습니다. 짧은 학습이 Recall 조건을 못 맞추면 테스트 전 중단하는 것이 정상입니다. 실제 평가용 모델과 구분하세요. `-Source 'C:/path/to/images'`를 추가하면 마지막에 인퍼런스도 실행합니다.

## 5. 생성 결과

| 단계 | 주요 결과 | 확인할 내용 |
|---|---|---|
| pre_eda | images.csv, boxes.csv, class_distribution.csv, date_distribution.csv | 이미지 수, 클래스 편중, 날짜별 구성, 해상도·밝기·박스 분포 |
| pre_eda | distributions.png, label_samples/ | 박스 위치, 크기, 클래스 라벨 육안 검토 |
| pre_eda | checks.json | 날짜 중복, 동일 파일 해시 중복, 동일 dHash의 유사 사진 후보 |
| train | fit/weights/best.pt, fit/results.csv, training.json | 가중치 및 학습 이력; best.pt의 정확한 경로 |
| val evaluate | threshold.json, threshold_sweep.csv | 선택한 임계값, Recall 조건 만족 여부 |
| evaluate | metrics.json, class_metrics.csv, image_metrics.csv | AP와 고정 임계값 P/R/F1, 클래스별 표본 수, 크기별 Recall |
| evaluate | records.json, ap/ | 정답·예측 원시 결과 및 라이브러리 PR 곡선·혼동행렬 |
| post_eda | metrics.png, class_ap.png, training_curves.png | 클래스별 결과 및 학습/검증 손실·mAP 곡선 |
| post_eda | class_target_comparison.csv, overall_target_comparison.csv | 목표 대비 수치, 표본 부족 및 N/A 표시 |
| post_eda | false_positive/, false_negative/, success/ | 초록 정답 박스·빨강 예측 박스를 겹친 사례 |
| post_eda | size_recall.csv, image_metrics.csv, native_*.png | 박스 크기별 오류와 원본 평가 시각화 |
| infer | predictions.jsonl, visuals/ | 좌표·클래스·confidence 및 탐지 결과 이미지/영상·YOLO 라벨 |

출력 폴더를 지정하지 않으면 `artifacts/<task>/<단계>/<시각>/`에 저장합니다. 기존 출력 폴더는 덮어쓰지 않습니다. `records.json`은 원본 이미지 절대 경로를 포함하므로 학습 후 EDA를 수행할 때 이미지를 옮기지 마세요.

학습 후 예시 이미지는 오류 유형별로 최대 `--samples 20`장을 저장합니다. 클래스별 통계는 전체 표본으로 계산합니다. 모든 소수 클래스의 예시가 자동으로 포함되는 것은 아니므로 `class_metrics.csv`와 원시 예측을 함께 확인하세요.

## 6. 평가 해석 시 주의점

- AP는 confidence 0.001의 낮은 후보 수집 기준으로 별도 계산합니다. 운영용 고정 임계값 P/R/F1은 저장한 예측을 confidence 순으로 정렬한 뒤 클래스 일치·IoU ≥ 0.50·일대일 매칭으로 계산합니다. 라이브러리 자동 best-F1 임계값의 P/R과 혼용하지 않습니다.
- `ap/`와 `native_*.png`는 AP 평가 설정의 시각화입니다. 운영 confidence에서 계산한 오류 사례 및 수치는 별도입니다.
- data2의 `primary_ap`, `primary_operating`은 손상 2종만 평균합니다. 원본 3종 클래스별 결과는 별도로 남깁니다. 정답 없는 클래스의 AP·Recall·F1은 N/A이며 0점이나 합격으로 처리하지 않습니다.
- 목표 비교는 클래스별 정답 50박스·20사진 미만을 `INSUFFICIENT_SAMPLE`로 표시합니다. 표본 조건을 넘더라도 독립 촬영 날짜, 검증 표본 충족, 라벨 품질까지 자동 보증하지 않으므로 `MET_OBSERVED`는 최종 승인 판정이 아닙니다.
- dHash가 동일한 사진은 유사 사진 후보입니다. 오탐 후보와 놓치는 유사 사진이 모두 있을 수 있어 육안 확인이 필요합니다.
- 정상 사진 오탐률과 두 모델 연결 성능은 일반 손상 양성 세트의 mAP와 다른 평가입니다. 연결 모델 구현·평가는 이번 독립 task 스크립트에 포함되지 않습니다.

목표 수치와 정의는 [평가 계획](evaluation_plan.md)에 정리되어 있습니다. 평가 API 참고: [공식 검증 문서](https://docs.ultralytics.com/modes/val/), [공식 예측 문서](https://docs.ultralytics.com/modes/predict/).

## 7. 추가 평가 옵션

이미 검수된 정상 운동화 사진 폴더가 있으면 data2 평가 명령에 `--normal-source 'C:/path/to/confirmed_normal'`을 추가합니다. 손상 2종이 하나 이상 탐지된 정상 사진 비율을 `normal_metrics.json`으로 저장합니다. 정상 여부는 코드가 판정하지 않으며 사용자가 확인한 자료여야 합니다. 임계값을 정상 테스트 결과에 맞춰 다시 튜닝하지 않습니다.

평가 명령에 `--benchmark`를 추가하면 워밍업 10회 후 batch 1로 200회 처리 시간을 측정하고 `latency.json`에 평균·중앙값·p95를 기록합니다. 이미지 읽기/디코딩과 모델 최초 로딩은 제외하며 전처리·추론·후처리를 포함합니다. 표본이 200장보다 적으면 이미지를 반복합니다. CUDA는 동기화하여 측정합니다. 기본 라이브러리의 개별 speed 값과 이 p95를 혼용하지 않습니다.

다른 모델은 학습 명령에 `--model yolo11s.pt`, 다른 설정은 `--epochs`, `--imgsz`, `--batch`, `--data`로 지정합니다. 입력 크기·NMS IoU·max_det를 바꾸면 임계값을 val에서 새로 선택해야 합니다. 임계값 파일에는 가중치 SHA256과 이 설정을 기록하며, 다른 모델이나 다른 설정에서 사용하면 오류를 냅니다.

## 8. 현재 검증 범위

코드 문법 검사, 10개 실행 파일의 도움말 진입, 박스 IoU·중복 탐지·클래스 오류·미탐·표본 없는 클래스 처리 등 평가 계산 단위 테스트를 수행했습니다. 학습 라이브러리 설치, 실제 EDA 그래프 생성, 실제 모델 학습·예측·AP 평가 실행은 아직 수행하지 않았습니다. 이 문서에 실측 모델 성능은 없습니다.

```powershell
python -m unittest discover -s tests -v
```


## 9. RF-DETR (상업 이용 가능 대안)

YOLO11은 AGPL-3.0이라 비공개 상용 배포에는 Ultralytics Enterprise 라이선스가 필요합니다. RF-DETR은 **N/S/M/L 크기만 Apache-2.0**이므로 코드에서도 이 4개만 선택할 수 있습니다(A/F/P/XL/2XL은 Roboflow PML 라이선스). 구현: [pipeline/rfdetr_run.py](../pipeline/rfdetr_run.py), 자동 실행: [run_rfdetr_task.ps1](../run_rfdetr_task.ps1).

- 데이터: 기존 `datasets/<task>/data.yaml`(YOLO 형식)을 변환 없이 사용합니다. 분할도 동일합니다.
- 평가: 고정 임계값 매칭·임계값 선택·결과 EDA는 `run.py`의 함수를 그대로 사용하므로 YOLO 결과와 같은 기준입니다. AP는 같은 예측 기록으로 pycocotools `COCOeval`(maxDets 300)에서 계산합니다.
- RF-DETR은 NMS를 사용하지 않습니다. `threshold.json`의 `nms_iou`는 `null`로 기록합니다. 분류기의 no-object 출력(클래스 ID = 클래스 수)은 예측에서 제외합니다.
- 최적 가중치는 RF-DETR이 val mAP@50:95(EMA)로 고른 `train/fit/checkpoint_best_total.pth`입니다.

### 환경 (휴대용 Python에 설치)

`.tools/python311`의 `python311._pth`에서 `import site`와 `Lib\site-packages`를 활성화하고 pip로 설치했습니다.

```powershell
$py = '.\.tools\python311\python.exe'
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements-rfdetr.txt
```

### 실행

```powershell
# 전 단계 자동 실행: pre_eda → train → val 평가·임계값·속도 → post_val → (Recall 조건 충족 시) test → post_test
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_rfdetr_task.ps1 -Task data1 -Model nano -ImgSize 384 -Epochs 10 -RunName rfdetr_nano_v1

# 개별 단계 예시
& $py -m pipeline.rfdetr_run --task data1 --action train --model small --imgsz 512 --epochs 10 --output artifacts/data1/rfdetr_small_v1/train
& $py -m pipeline.rfdetr_run --task data1 --action evaluate --model small --imgsz 512 --weights artifacts/data1/rfdetr_small_v1/train/fit/checkpoint_best_total.pth --split val --select-threshold --benchmark --output artifacts/data1/rfdetr_small_v1/val
```

`--subset 64`를 주면 train 64장·val/test 16장의 하드링크 부분 데이터로 학습하여 실행 경로와 시간을 확인할 수 있습니다(성능 평가용 아님).

### 이 PC(i7-10700, RAM 16GB, GPU 없음) 실측 시간

| 항목 | Nano (384) | Small (512) |
|---|---:|---:|
| 학습, 이미지 1장당 | 약 1.5초 | 약 2.6초 |
| 학습, 1 epoch(train 4,368장) 추정 | 약 1.7~1.8시간 | 약 3.2시간 |
| 추론 batch 1 p95 (2 epoch 시험 모델) | — | 480 ms |

시간은 64장 시험 학습 기준입니다. GPU 환경에서는 `--device cuda`를 사용합니다.

## 체크포인트 저장 정책

두 task 모두 학습 중 검증 fitness가 이전 최고값보다 높을 때만 동일한 best.pt를 갱신합니다. 동점은 먼저 저장한 모델을 유지하고 NaN/Inf 점수는 저장하지 않습니다. last.pt와 epoch별 체크포인트는 생성하지 않습니다. training.json에는 선택된 epoch와 fitness를 기록합니다. 학습 로그와 EDA 결과는 계속 저장합니다. 최신 epoch 체크포인트가 없으므로 중단 직전 상태로 복구하는 용도의 저장은 제공하지 않습니다. 현재 구현은 기존 Ultralytics 학습 백엔드에 적용되며, RF-DETR 등으로의 모델 교체는 아직 수행하지 않았습니다.

# Task2 — RF-DETR Small 실행 가이드

작성일: 2026-10-14. Task2 모델은 사용자의 선택에 따라 YOLO에서 **RF-DETR Small**로 변경했습니다. 이 문서는 이전 통합 가이드의 task2 YOLO 설정을 대체합니다. task1 코드는 이 변경의 대상이 아닙니다.

## 현재 모델과 데이터

- RF-DETR Small, `rfdetr[train]==1.11.2`, 공식 Apache-2.0 대상 Small 모델.
- 독립 학습 데이터 1,465장·2,465박스. train 1,222장, val 131장, test 112장.
- 클래스: 0 outsole, 1 heel lining, 2 not_sneakers. 손상 주 지표는 앞의 2종 평균이며 예외 클래스는 별도 기록.
- 기존 YOLO 형식의 이미지·라벨을 RF-DETR의 YOLO 데이터 로더로 읽습니다. 좌표와 날짜별 분할은 재사용합니다.
- 입력 512, 최대 100 epochs, patience 20, seed 42. multi-scale·expanded scales·EMA는 꺼 두었습니다. 학습 증강은 RF-DETR 1.11.2의 기본 torchvision 경로를 사용합니다. 이전 YOLO의 mosaic·HSV 설정을 사용하는 것은 아닙니다.
- 모델 선택은 val mAP@0.5:0.95 기준입니다. **`fit/best.pth` 하나만** 갱신하며 last/epoch별/EMA별 체크포인트와 마지막 모델 복제는 저장하지 않습니다.
- 외부 학습 서비스 업로드 및 wandb/mlflow/clearml/tensorboard 로깅을 사용하지 않습니다. 초기 실행 시 공식 사전학습 가중치를 내려받습니다.

## 이 PC의 짧은 CPU 학습

이 PC는 i7-10700, Intel UHD 630, RAM 약 16GB이며 CUDA GPU가 없습니다. 사용자는 **CPU의 짧은 학습 확인 후 GPU 본학습 준비**를 선택했습니다.

CPU 점검은 batch 1, 학습 16개 배치, 검증 8개 배치, 1 epoch로 수행합니다. 전체 학습 1,222장이나 전체 검증 131장을 모두 사용한 실험이 아닙니다. 여기서 얻은 점수와 모델은 실행 점검용이며 성능 목표 달성이나 배포용 모델로 해석하면 안 됩니다. test는 이 점검에서 사용하지 않습니다.

```powershell
.\.tools\python311\python.exe scripts/data2/train.py --epochs 1 --batch 1 --workers 0 --threads 4 --limit-train-batches 16 --limit-val-batches 8 --output artifacts/data2/cpu_smoke_new
```

점검 실행 결과는 [task2 실행 보고서](task2_run_report.md)에 기록합니다. 새로운 실행은 기존 출력과 다른 폴더 이름을 사용합니다.

## GPU 본학습 준비

`exports/task2_rf_detr_small_gpu.zip`은 task2 코드·라벨·사진과 이 문서를 포함하는 이식용 묶음입니다. 캐시, 테스트용 모델, 기존 YOLO 라이브러리는 포함하지 않습니다. 파일별 SHA256은 `bundle_manifest.json`에 기록합니다.

GPU PC/서버에 압축을 해제하고 해당 폴더를 작업 디렉터리로 사용합니다. 일반 Python 3.11과 GPU에 맞는 CUDA 지원 PyTorch가 필요합니다. 프로젝트의 `.tools` 휴대용 CPU 환경을 GPU 환경으로 복사하지 않습니다.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements_task2.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA 확인이 false면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 GPU/드라이버와 호환되는 패키지를 설치한 뒤 진행합니다. 설치된 실제 패키지 버전은 `python -m pip freeze`로 기록하세요.

## GPU 실행 순서

```bash
# 1. 전체 데이터 EDA와 라벨 확인
python scripts/data2/pre_eda.py --output artifacts/data2/gpu_v1/pre_eda

# 2. 전체 학습: limit 옵션을 지정하지 않음
python scripts/data2/train.py --device 0 --epochs 100 --batch 4 --grad-accum 4 --workers 0 --output artifacts/data2/gpu_v1/train

# 3. val 전체에서 평가하고 confidence 선택
python scripts/data2/evaluate.py --weights artifacts/data2/gpu_v1/train/fit/best.pth --device 0 --split val --select-threshold --output artifacts/data2/gpu_v1/val

# 4. val 성능·오류·학습 곡선 분석
python scripts/data2/post_eda.py --evaluation artifacts/data2/gpu_v1/val --training artifacts/data2/gpu_v1/train/fit --output artifacts/data2/gpu_v1/post_val
```

물리 batch 4·누적 4는 유효 batch 16입니다. GPU 메모리가 부족하면 batch 2·누적 8 또는 batch 1·누적 16으로 조정합니다. CUDA 환경의 batch 크기와 실행 시간은 아직 이 PC에서 검증하지 않았습니다.

val 결과와 `threshold.json`의 `constraints_met`를 검토하고 모델과 설정을 확정한 뒤에만 최종 test를 실행합니다. 손상 2종 각각 Recall 90%를 만족하는 confidence 후보 중 평균 Precision을 최대화합니다. 만족 후보가 없으면 false를 저장하므로 개선은 val 기준으로 진행하세요.

```bash
# 5. 최종 테스트: 튜닝에 사용하지 않음
python scripts/data2/evaluate.py --weights artifacts/data2/gpu_v1/train/fit/best.pth --device 0 --split test --threshold-config artifacts/data2/gpu_v1/val/threshold.json --output artifacts/data2/gpu_v1/test

# 6. 테스트 결과 EDA
python scripts/data2/post_eda.py --evaluation artifacts/data2/gpu_v1/test --training artifacts/data2/gpu_v1/train/fit --output artifacts/data2/gpu_v1/post_test

# 7. 새 사진/이미지 폴더 인퍼런스
python scripts/data2/infer.py --weights artifacts/data2/gpu_v1/train/fit/best.pth --device 0 --source /path/to/new_images --threshold-config artifacts/data2/gpu_v1/val/threshold.json --output artifacts/data2/gpu_v1/inference
```

## 출력과 평가 차이

- 학습: `fit/best.pth`, `fit/metrics.csv`, `fit/training_config.json`, `training.json`.
- 평가: `metrics.json`, `class_metrics.csv`, `image_metrics.csv`, `records.json`, COCO 정답/예측 JSON. val 임계값 선택 시 `threshold.json`과 `threshold_sweep.csv` 추가.
- 인퍼런스: `predictions.json`과 박스를 그린 `visuals/`. 현재 RF 백엔드는 이미지/이미지 폴더를 지원하며 영상 입력은 지원하지 않습니다.
- 학습 후 EDA: 클래스별 AP·P/R/F1, 목표 대비표, 크기별 Recall, 성공·오탐·미탐 이미지, 학습 곡선. RF 체크포인트 학습 로그는 기존 YOLO의 results.csv 대신 metrics.csv를 읽습니다.
- RF-DETR는 NMS 없이 동작합니다. 현재 공통 CLI의 `--nms-iou`는 RF에서 예측에 적용되지 않으며 임계값 파일 호환 확인용 필드로만 남아 있습니다.
- RF 평가 AP는 pycocotools의 bbox 평가와 maxDets 300을 사용합니다. 고정 confidence P/R/F1은 공통 클래스 일치·IoU 0.50·일대일 매칭 계산입니다.
- `--normal-source`와 `--benchmark`는 RF 백엔드에 아직 연결하지 않았습니다. 기존 YOLO 가이드의 해당 옵션을 task2 RF 모델에 사용하지 마세요.
- 현재 test의 outsole 17박스, not_sneakers 0박스 등 표본 부족은 모델 교체로 해결되지 않습니다. 정상 오탐률 평가도 확인된 정상 세트가 필요합니다.

## 라이선스 참고

모델 Small과 공식 rfdetr 코드는 Apache-2.0 대상입니다. XL/2XL 등 다른 변형의 조건을 동일하게 가정하지 않습니다. 사용한 체크포인트·의존성·데이터의 조건은 배포 전에 함께 확인합니다. [공식 RF-DETR 저장소](https://github.com/roboflow/rf-detr)

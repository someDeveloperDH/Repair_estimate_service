# 신발 학습 데이터

> **Task2 변경 안내 (2026-10-14):** task2는 RF-DETR Small로 전환했으며 CPU의 짧은 학습·검증·인퍼런스·전후 EDA를 실행했습니다. 아래 문서의 task2 YOLO 관련 설정과 미실행 상태는 초기 기록입니다. 현재 설정은 [RF-DETR 실행 가이드](task2_rf_detr_guide.md), 실측 점검 결과는 [실행 보고서](task2_run_report.md)를 기준으로 확인하세요.

원본 ZIP은 수정하지 않았습니다. 원본 라벨이 COCO 바운딩 박스이므로 두 task를 **YOLO 객체 탐지** 데이터로 변환했습니다. data1은 사진 안의 신발 종류 및 위치, data2는 손상 부위 및 위치를 학습합니다. 이미지 전체를 한 종류로 분류하는 classification 데이터셋은 아닙니다.

## 생성 결과

| Task | train | val | test | 총 이미지 | 총 박스 |
|---|---:|---:|---:|---:|---:|
| data1 신발 종류 | 4,368 | 557 | 821 | 5,746 | 10,912 |
| data2 손상 부위 | 1,222 | 131 | 112 | 1,465 | 2,465 |

- data1 클래스: `0 sneakers`, `1 formal`, `2 boots`, `3 sandals`, `4 slippers`, `5 heels`, `6 others`.
- data2 클래스: `0 outsole`, `1 heel lining`, `2 not_sneakers`.
- data2 내 실제 사용 박스: outsole 358개, heel lining 2,101개, not_sneakers 6개.
- data2에는 원본 사진이 없는 이미지가 6,198장이며, 이 중 라벨이 있는 사진은 1,321장입니다. `missing_images.csv`에서 확인할 수 있습니다.
- 매칭되지만 라벨 없는 사진(data1 3,008장, data2 4,936장)은 정상/미작업 여부가 불명확하여 제외했습니다. 따라서 정상 이미지 탐지 성능을 평가하기 위한 음성 표본은 별도 확보해야 합니다.
- 파일명으로 매칭하며 사진 이름 중복은 없습니다. data2의 고해상도 라벨 좌표는 라벨에 기록된 너비/높이로 정규화했습니다. 사용 사진 전체의 종횡비를 검사했으며 불일치는 없었습니다. 같은 파일명의 축소본이라는 전제이며 개별 이미지의 크롭 여부까지 수동 확인한 것은 아닙니다.
- data1에서 경계를 조금 벗어난 박스 4개를 이미지 경계로 잘랐습니다. data2는 0개입니다.

## 파일 구조

```text
men_dii/
  prepare_data.ps1       # 원본 ZIP → 이미지 매칭, 좌표 변환, 분할
  verify_data.ps1        # 이미지/라벨 대응, 좌표 범위, 날짜 분할 검증
  train.py               # task별 독립 모델 학습 및 test 평가
  requirements.txt
  datasets/
    data1/ (data2도 동일)
      data.yaml
      images/train/ val/ test/
      labels/train/ val/ test/
      manifest.csv
      missing_images.csv
      invalid_annotations.json
      summary.json
    verification.json
```

라벨 한 줄: `class_id center_x center_y width height`. 좌표는 0~1이며 클래스 ID는 COCO의 1부터 시작하는 ID를 YOLO의 0부터 시작하는 ID로 바꿨습니다.

## 재생성 / 검증

Windows PowerShell에서 프로젝트 폴더로 이동한 뒤 실행합니다. 변환은 Python이나 별도 라이브러리 없이 실행됩니다.

```powershell
cd C:\Users\user\Desktop\final\men_dii
powershell -NoProfile -ExecutionPolicy Bypass -File .\prepare_data.ps1 -OutputDir C:\Users\user\Desktop\final\men_dii\datasets_new
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify_data.ps1 -DatasetDir C:\Users\user\Desktop\final\men_dii\datasets_new
```

기존 출력 폴더는 덮어쓰지 않습니다. `data.yaml`에는 생성 위치의 절대 경로가 들어 있으므로 폴더를 옮기면 `path`를 수정합니다.

## 학습

Python 3.10 이상이 설치된 환경에서 실행합니다. 일반 Python 학습 환경은 아직 구성하지 않았으며 실제 모델 학습은 수행하지 않았습니다. 휴대용 Python으로 코드 문법 및 평가 계산 단위 테스트를 확인했습니다. 데이터 변환 및 독립 검증 스크립트도 실행하여 통과했습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --task data1 --device cpu
.\.venv\Scripts\python.exe train.py --task data2 --device cpu
```

CUDA 지원 PyTorch가 설치된 NVIDIA GPU 환경은 `--device 0`으로 변경합니다. 기본 모델은 `yolo11n.pt`, 입력 크기는 512, batch는 8, epochs는 100입니다. 초기 실행에는 사전학습 가중치 다운로드가 필요합니다. 다른 데이터셋은 `--data datasets_new/data1/data.yaml` 등으로 지정할 수 있습니다. 결과는 `artifacts/`에 저장됩니다. 학습은 test 평가를 자동 실행하지 않습니다. 별도 평가 스크립트에서 val 임계값 선택 후 test를 평가합니다. 상세한 실행 순서는 [파이프라인 가이드](pipeline_guide.md)를 참고하세요.

학습 API 참고: https://docs.ultralytics.com/modes/train/

## 분할과 평가의 한계

촬영 날짜(파일명 앞 8자리)별 SHA256 해시로 train/val/test에 약 80/10/10 비율로 배정합니다. 날짜별 이미지 수가 달라 실제 장수 비율은 정확히 80/10/10이 아닙니다. 같은 날짜의 사진은 두 task에서 모두 같은 split에 들어가며 날짜 중복 검증을 통과했습니다. 신발 개체 ID가 없어 다른 날짜에 같은 신발이 등장하는지까지 검증하지 못합니다.

소수 클래스는 날짜 그룹 분할 시 평가 세트에서 빠질 수 있습니다. 현재 data1의 slippers는 train 66개, val 0개, test 1개입니다. data2의 not_sneakers는 train 6개, val/test 0개이므로 이 클래스의 일반화 성능을 평가할 수 없습니다. 각 클래스에 대한 독립 촬영 데이터 추가가 필요합니다. not_sneakers는 손상 부위가 아니지만 원본 클래스 정의를 보존했습니다.


문서: [data1 정보](data1_info.md), [data2 정보](data2_info.md), [평가 목표](evaluation_plan.md), [학습·인퍼런스·평가·EDA 실행 가이드](pipeline_guide.md).


현재 두 task를 RF-DETR Small로 한 번에 실행하는 Colab 묶음은 [Colab 전체 실행 안내](colab_all_guide.md)를 참고하세요. 준비된 전체 데이터는 task1 5,746장, task2 1,465장입니다.

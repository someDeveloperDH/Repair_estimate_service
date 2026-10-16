# Task1 + Task2 Colab 전체 실행

두 task를 RF-DETR Small로 순차 실행합니다. 각 task는 준비된 전체 train split을 10 epochs 학습하고 val 평가, confidence 선택, 전후 EDA와 val 사진 최대 20장 인퍼런스를 수행합니다. task1 5,746장, task2 1,465장의 전체 학습용 데이터가 포함됩니다. 사진 누락·미라벨 원본은 포함하지 않으며 기존 제외 정책과 날짜 분할을 유지합니다.

## 실행

1. Google Colab에서 `exports/men_dii_all_colab.ipynb`를 업로드하여 엽니다.
2. 런타임 유형을 GPU로 변경합니다.
3. 모두 실행을 누르고 파일 선택창에서 `exports/men_dii_all_colab.zip`을 선택합니다.
4. 두 task가 끝나면 모델·평가지표·그림·오류 사례·로그가 들어 있는 결과 ZIP을 다운로드합니다.

기본 설정은 epochs 10, 물리 batch 2, 누적 8(유효 batch 16), 입력 512입니다. GPU 메모리 부족 시 batch 1·누적 16으로 변경하여 새 실행을 진행합니다. epoch·batch 등은 노트북 첫 코드 셀에서 수정합니다. 학습 배치 수를 줄이는 점검 옵션은 사용하지 않습니다.

각 task의 체크포인트는 `train/fit/best.pth` 한 개입니다. val mAP@0.5:0.95가 개선될 때만 갱신하며 last/epoch별 모델은 생성하지 않습니다. 초기 사전학습 가중치는 별도 캐시이며 최종 결과 묶음에 포함하지 않습니다.

## 결과 보관

기본값은 `SAVE_TO_DRIVE=True`입니다. 최초 Google Drive 연결을 직접 승인하면 최적 모델, 학습 로그, 평가 결과와 EDA 그림이 `MyDrive/men_dii_results/<실행ID>/`에 자동 저장됩니다. 데이터는 Colab 로컬 디스크에서 읽습니다. 학습 중 검증 점수가 개선될 때마다 task별 best.pth 하나를 갱신하며, 별도 epoch 모델을 계속 쌓지 않습니다. 완료 후 결과 ZIP도 Drive에 저장하고 다운로드합니다. 이미 저장된 파일은 Colab 런타임이 종료되어도 Drive에 남습니다.

최적 모델 경로:

```text
내 드라이브/men_dii_results/<실행ID>/data1/train/fit/best.pth
내 드라이브/men_dii_results/<실행ID>/data2/train/fit/best.pth
```

새 실행은 다른 실행ID를 사용하므로 이전 실험 결과는 보존됩니다. 모델은 검증을 한 번 완료한 뒤 처음 저장됩니다. Drive 연결 실패 시 학습을 시작하지 않습니다. `SAVE_TO_DRIVE=False`로 직접 변경한 경우에만 로컬 런타임 저장으로 전환됩니다.

`FINAL_TEST=False`가 기본입니다. 첫 10 epochs는 모델 개선을 위한 기준 실험으로 val 결과를 확인하고 test는 남겨 둡니다. 최종 설정을 확정한 뒤 True로 변경하면 val Recall 조건을 만족한 task에 한해 test 및 test EDA도 실행합니다. 임계값이 조건을 못 맞추더라도 val 결과와 오류 분석은 저장됩니다.

task1은 sneakers Recall 95%를 만족하는 후보 중 Macro-F1을, task2는 손상 2종 각각 Recall 90%를 만족하는 후보 중 두 종류의 Macro-Precision을 우선합니다. 소수 클래스의 표본 부족과 not_sneakers의 평가 불가는 그대로 기록합니다. 정상 이미지 오탐률·두 모델 연결 평가는 이 실행 범위에 포함되지 않습니다.

## 검증 범위

Colab 연결 브라우저가 현재 세션에 없어 실제 원격 GPU 실행은 아직 수행하지 못했습니다. 이 노트북은 로컬에서 코드 셀 문법, 묶음의 파일별 해시와 ZIP 무결성, task별 전체 이미지·라벨 수를 확인하여 제공합니다. GPU 드라이버·메모리·무료 할당량 및 장시간 실행은 실제 Colab에서 확인해야 합니다.

RF-DETR 공통 백엔드의 task2 CPU 짧은 학습·재로딩·전체 val 평가·전후 EDA·인퍼런스는 실행 확인했습니다. 이번 통합 변경은 두 task의 기본 모델을 RF-DETR Small로 설정하고 task1의 7개 클래스 평가 범위를 반영합니다. 기존 별도 `pipeline/rfdetr_run.py`는 유지하지만 Colab 실행에는 사용하지 않습니다.

# Task2 실행 보고서

작성일: 2026-10-14

사용자 선택: RF-DETR Small로 전환하고 CPU에서 짧은 학습을 확인한 뒤 GPU 본학습을 준비합니다.

## 데이터와 학습 전 EDA

- 데이터 검증 통과: task2 1,465장·2,465박스.
- 전체 EDA: `artifacts/data2/rf_small_v1/pre_eda/`에 클래스·박스·날짜 분포, 라벨 샘플, 중복 검사 결과 저장.
- 동일 파일 해시·동일 dHash의 split 간 중복 후보와 촬영 날짜의 split 간 중복은 발견되지 않았습니다. 근접 중복이나 다른 날짜의 동일 신발을 모두 배제했다는 의미는 아닙니다.
- 밑창과 뒤꿈치 안감의 라벨 시각화 샘플을 각각 확인했으며 검사한 두 샘플에서는 박스가 해당 부위에 대응했습니다. 전체 라벨의 육안 검수가 완료된 것은 아닙니다.

## 실행 환경

- Intel Core i7-10700, Intel UHD 630, RAM 약 15.9GB, CUDA GPU 없음.
- 프로젝트 내부 Python 3.11.9, CPU PyTorch 2.8.0, torchvision 0.23.0, rfdetr 1.11.2 설치.
- RF-DETR Small 공식 사전학습 가중치 다운로드 및 upstream MD5 검증 완료.
- 다른 task1 학습 프로세스가 함께 실행 중인 것을 확인했습니다. 이 프로세스는 중단하지 않았으며, CPU 시간과 사용 가능 메모리는 실행 도중 영향을 받습니다.

## 짧은 학습 점검

- 설정: 1 epoch, batch 1, 학습 16배치, 검증 8배치, CPU threads 4.
- 최초 실행은 학습과 검증 계산 후 Windows cp949 콘솔이 평가표의 Unicode 문자를 출력하지 못해 중단됐습니다. 이 실행에서 최종 모델이 저장되지는 않았습니다.
- 표준 출력·오류를 UTF-8로 설정하여 같은 조건으로 재실행했습니다. 실패 로그는 보존합니다.
- UTF-8 수정 후 짧은 학습 정상 완료. 학습·검증 루프 시간 약 79.6초(패키지 import 및 모델 초기 로딩 제외). epoch 1의 best.pth 1개만 생성됐으며 약 121.5MiB입니다.

이번 점검은 전체 데이터 학습이나 최종 성능 평가가 아닙니다. test는 사용하지 않습니다. GPU 본학습·평가 실행은 [RF-DETR 가이드](task2_rf_detr_guide.md)를 참고합니다.

## 저장 모델 재로딩·전체 val 평가

학습은 16배치에 한정했지만, 저장 모델 재로딩과 평가 코드의 점검은 val 131장 전체(손상 정답 207박스)에서 수행했습니다. 이 결과는 **극히 짧게 학습한 점검 모델**의 결과이며 본학습 성능이 아닙니다.

| 항목 | 점검 모델 결과 |
|---|---:|
| 손상 2종 mAP@0.5 | 0.432% |
| 손상 2종 mAP@0.5:0.95 | 0.122% |
| 탐색용 confidence | 0.15 |
| outsole Recall | 0% |
| heel lining Recall | 1.075% |
| Recall 조건 충족 | 아니오 |

`threshold.json`의 constraints_met는 false입니다. 선택된 값은 조건을 만족하지 못한 경우의 탐색용 대체 임계값입니다. 배포용으로 사용하지 않습니다. not_sneakers는 val 정답이 없어 AP/Recall/F1이 N/A입니다.

초기 평가 실행에서 RF-DETR의 num_classes+1 출력 중 사용자 클래스가 아닌 마지막 슬롯 ID가 반환되는 것을 확인했습니다. YOLO 타깃이 0~2이고 모델 헤드가 4개 슬롯을 사용하는 것을 소스로 확인한 뒤 마지막 슬롯을 제외하도록 처리했습니다. 실제 제외 개수는 records.json에 기록합니다. 수정 후 131장 평가가 정상 완료됐습니다.

## 학습 후 EDA·인퍼런스

- 학습 후 EDA 정상 실행: 클래스별 AP/P/R/F1, 박스 크기별 Recall, 목표 비교표, PR 곡선, 운영 confidence의 공간 매칭 혼동행렬, 오류 사례 및 학습 곡선 생성.
- 인퍼런스 정상 실행: val에서 가져온 사진 3장으로 모델 재로딩·예측·결과 이미지 저장 경로를 확인했습니다. 새 독립 데이터의 성능 평가가 아닙니다. 해당 3장에서는 confidence 0.15 이상인 유효 사용자 클래스 예측이 없었습니다.
- 코드 문법과 평가·임계값·AP 계산 등 15개 단위 테스트 통과.
- test 112장은 사용하지 않았습니다.
- 짧은 학습은 compute_val_loss=auto로 실행되어 val 손실 계산이 생략됐습니다. 이후 GPU 본학습 코드에는 compute_val_loss=True를 지정하여 학습·검증 손실 곡선을 함께 기록하도록 했습니다.

## 결과 위치

- 학습 전 EDA: `artifacts/data2/rf_small_v1/pre_eda/`
- 정상 완료한 짧은 학습: `artifacts/data2/rf_small_v1/smoke_train_v2/`
- 모델: `artifacts/data2/rf_small_v1/smoke_train_v2/fit/best.pth`
- 전체 val 점검: `artifacts/data2/rf_small_v1/smoke_val_v2/`
- 학습 후 EDA: `artifacts/data2/rf_small_v1/post_smoke_val/`
- 인퍼런스 점검: `artifacts/data2/rf_small_v1/smoke_inference/`
- 패키지 버전: `artifacts/data2/rf_small_v1/environment_freeze.txt`
- GPU 이식용 묶음: `exports/task2_rf_detr_small_gpu.zip`

GPU 본학습은 아직 실행하지 않았습니다. 제공 묶음에는 전체 task2 사진·라벨과 task2 전용 실행 코드가 포함되며 점검 모델은 포함되지 않습니다. 목표 성능은 GPU 본학습 후 val에서 확인하고, 최종 설정을 확정한 뒤 test에서 측정해야 합니다.
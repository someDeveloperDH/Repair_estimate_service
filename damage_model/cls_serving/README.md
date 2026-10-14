# 사진 단위 파손 분류 (outsole / heel lining / 그 외)

사진 한 장에 **outsole**(밑창 벌어짐) · **heel lining**(뒤꿈치 안감 손상) 파손이 있는지를 클래스마다 확률로 냅니다(다중 라벨).
둘 다 0.5 미만이면 **그 외**(두 부위 파손 없음)입니다. 박스(위치)는 내지 않고 "이 사진에 어떤 수선이 필요한가"만 판단합니다.

## 파일

| 경로 | 내용 |
|---|---|
| `notebooks/mendii_damage_cls_colab.ipynb` | 데이터 준비 → 학습 → 채점 Colab 노트북 (이것만으로 실행됩니다) |
| `notebook_builders/build_cls_nb.py` | 위 노트북을 만드는 스크립트 |
| `cls_serving/model.py` | 학습된 `last.pt`를 불러 판단 |
| `cls_serving/serve.py` | API 서버 (`GET /health`, `POST /predict`) |
| `cls_serving/predict.py` | 서버 없이 사진·폴더 판단 |
| `cls_serving/shoe_app.py` | 서버 시험 GUI (표준 라이브러리, 일괄 추론·재시도·CSV 저장) |

## 데이터와 채점

검출 모델 작업(`jake` 브랜치의 `damage_model/`)과 **같은 데이터 규칙**을 씁니다. 노트북 3~6번 셀이 그 노트북의 데이터 셀과 같은 코드입니다.

- CVAT 최종 export `mendii_shoes_20261014_0900.zip` + outsole 재라벨 `outsole_relabel.json`, 사진은 S3 `mendii-sneakers`에서 받음
- 검수된 사진만, 촬영 간격 300초 세션 단위로 train/val 분리, 파손 없는 사진 1:1, outsole 사진 2회 반복
- YOLO 노트북을 먼저 돌렸으면 그 `split.json`과 사진 캐시를 그대로 써서 **같은 검증 사진**으로 YOLO26과 사진·신발 단위 AP를 비교하고, 차이의 세션 부트스트랩 90% 구간을 냅니다.
- 학습 중 검증하지 않고 마지막 가중치로 채점하며, 기준값 0.5도 검증 사진으로 고르지 않습니다.

모델: timm `convnext_tiny.fb_in22k_ft_in1k`(Apache-2.0), 512px, 15 epoch, T4에서 추정 30~40분.

> 사진·CVAT export·S3 키는 저장소에 올리지 않습니다. 사진에 작업지시서(개인정보)가 찍혀 있습니다.

## 실행

**학습 (Colab)**: 노트북을 열고 런타임을 T4 GPU로 바꾼 뒤 모두 실행합니다. 3번 셀에서 CVAT export zip, `outsole_relabel.json`, S3 읽기 키 csv를 올립니다.
처음에는 3번 셀에서 `QUICK_TEST = True`로 몇 분 안에 끝까지 도는지 확인하세요. 결과는 Drive `MyDrive/mendii/yolo26_damage/runs_cls/<NAME>/`에 저장됩니다.

**서빙**

```bash
pip install torch torchvision timm fastapi uvicorn python-multipart
python serve.py --ckpt last.pt              # http://127.0.0.1:8000
python predict.py --ckpt last.pt --image photos/
python shoe_app.py                          # GUI에서 API 주소 입력 → 서버 점검 → 추론 시작
```

```json
{"part": "heel", "part_ko": "뒤꿈치", "confidence": 0.81,
 "probs": {"outsole": 0.12, "heel": 0.81}, "threshold": 0.5, "inference_ms": 18.4}
```

## 노트북 다시 만들기

`build_cls_nb.py`는 검출 노트북 빌더 `build_nb.py`(현재 `jake` 브랜치)의 데이터 셀을 읽어 옵니다.
두 브랜치가 합쳐진 뒤 `python build_cls_nb.py ../notebooks/mendii_damage_cls_colab.ipynb`로 다시 만듭니다.

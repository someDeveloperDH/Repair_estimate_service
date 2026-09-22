"""
SigLIP2 zero-shot 분류 엔진.

왜 zero-shot인가
    수선 접수 사진에는 정답 라벨이 없다. 지도학습(YOLO-cls, ResNet 파인튜닝 등)은
    라벨링을 먼저 해야 시작할 수 있다. 반면 zero-shot은 텍스트 프롬프트만으로
    바로 동작하고, 클래스를 바꿀 때 재학습 없이 taxonomy.py만 고치면 된다.

왜 YOLO가 아닌가
    YOLO는 객체 검출 모델이다. 접수 사진은 품목이 화면을 거의 채우고 있어
    검출할 것이 없고, 필요한 것은 "무엇인가"라는 한 줄이다. 검출이 필요한 단계
    (파손 부위 찾기)는 이 모듈이 아니라 별도 detector가 맡는다.

신뢰도 계산 주의
    SigLIP은 sigmoid 기반이라 logit_scale(~118)에서 확률이 전부 0.999로 포화된다.
    축 안에서의 순위 비교가 목적이므로 softmax(온도 100)를 쓴다. argmax는 동일하고
    신뢰도만 살아난다. 1위-2위 격차(margin)를 함께 내보내 abstention에 쓴다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

from . import taxonomy

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODELS = {
    "base": "google/siglip2-base-patch16-256",     # 빠름. M1 Pro에서 약 40 img/s
    "large": "google/siglip2-so400m-patch16-384",  # 정확. 대략 6~8배 느림
}

SOFTMAX_TEMP = 100.0


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tensor(out) -> torch.Tensor:
    """transformers 5.x는 ModelOutput을 반환한다 — 텐서로 언랩."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "last_hidden_state", "text_embeds", "image_embeds"):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v if v.dim() == 2 else v[:, 0]
    raise TypeError(f"feature 추출 실패: {type(out)}")


@dataclass
class Prediction:
    label: str          # 클래스키
    confidence: float   # 축 내 softmax 확률
    margin: float       # 1위 - 2위. 낮으면 사람 확인 대상


class ZeroShotClassifier:
    """여러 축을 한 번의 이미지 인코딩으로 동시에 분류한다."""

    def __init__(self, axes: Sequence[str] = ("item", "view", "shoe_type"),
                 model: str = "base", device: str | None = None):
        from transformers import AutoModel, AutoProcessor

        self.device = pick_device(device)
        self.model_name = MODELS[model]
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.axes = list(axes)
        self._banks = {name: self._encode_axis(name) for name in self.axes}

    @torch.no_grad()
    def _encode_axis(self, axis_name: str) -> Tuple[List[str], torch.Tensor]:
        """클래스별 프롬프트 임베딩을 평균내 하나의 기준 벡터로."""
        keys, vecs = [], []
        for key, (_, prompts) in taxonomy.AXES[axis_name].items():
            inp = self.processor(text=prompts, padding="max_length",
                                 truncation=True, return_tensors="pt").to(self.device)
            e = _tensor(self.model.get_text_features(**inp))
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            vecs.append(e / e.norm())
            keys.append(key)
        return keys, torch.stack(vecs)

    @torch.no_grad()
    def embed(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inp = self.processor(images=list(images), return_tensors="pt").to(self.device)
        e = _tensor(self.model.get_image_features(**inp))
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify(self, images: Sequence[Image.Image]
                 ) -> Tuple[Dict[str, List[Prediction]], np.ndarray]:
        """이미지 배치 → {축: [Prediction]}, 그리고 임베딩 배열."""
        emb = self.embed(images)
        result: Dict[str, List[Prediction]] = {}
        for axis_name in self.axes:
            keys, bank = self._banks[axis_name]
            probs = torch.softmax(emb @ bank.T * SOFTMAX_TEMP, dim=-1)
            top2 = probs.topk(min(2, len(keys)), dim=-1)
            conf = top2.values[:, 0]
            margin = conf - (top2.values[:, 1] if top2.values.shape[1] > 1 else 0.0)
            result[axis_name] = [
                Prediction(keys[i], float(c), float(m))
                for i, c, m in zip(top2.indices[:, 0].tolist(),
                                   conf.tolist(),
                                   margin.tolist() if torch.is_tensor(margin) else margin)
            ]
        return result, emb.float().cpu().numpy()


def load_images(paths: Sequence[str]) -> List[Image.Image]:
    """깨진 파일이 있어도 배치 전체를 죽이지 않는다."""
    out = []
    for p in paths:
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! 읽기 실패 {os.path.basename(p)}: {exc}", flush=True)
            out.append(Image.new("RGB", (512, 512), (255, 255, 255)))
    return out

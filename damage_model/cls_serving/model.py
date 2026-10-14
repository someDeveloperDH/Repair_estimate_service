"""mendii_damage_cls_colab.ipynb가 저장한 last.pt를 불러 사진 한 장의 파손 부위를 판단한다.

출력 (사진 한 장)
  part        outsole | heel | other   (두 부위 확률이 모두 threshold 미만이면 other = 그 외)
  confidence  part가 outsole/heel이면 그 확률, other면 1 - 두 확률 중 큰 값
  probs       {"outsole": p, "heel": p}  클래스마다 독립 확률(다중 라벨이라 합이 1이 아님)
"""
from pathlib import Path

import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms

KEY = {"outsole": "outsole", "heel lining": "heel"}          # 노트북 클래스 이름 → API 키
KO = {"outsole": "아웃솔", "heel": "뒤꿈치", "other": "그 외"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class DamageClassifier:
    def __init__(self, ckpt_path, device=None, threshold=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.name = ck["model"]
        self.keys = [KEY.get(c, c) for c in ck["classes"]]
        self.threshold = float(threshold if threshold is not None else ck.get("threshold", 0.5))
        self.model = timm.create_model(ck["model"], pretrained=False, num_classes=len(ck["classes"]))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        size = ck["img_size"]
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    @torch.no_grad()
    def predict(self, img):
        x = self.tf(ImageOps.exif_transpose(img).convert("RGB")).unsqueeze(0).to(self.device)
        p = torch.sigmoid(self.model(x).float())[0].cpu().tolist()
        probs = {k: round(v, 4) for k, v in zip(self.keys, p)}
        top = max(probs, key=probs.get)
        if probs[top] >= self.threshold:
            part, conf = top, probs[top]
        else:
            part, conf = "other", round(1 - probs[top], 4)
        return {"part": part, "part_ko": KO.get(part, part), "confidence": conf,
                "probs": probs, "threshold": self.threshold}

    def predict_path(self, path):
        with Image.open(Path(path)) as im:
            return self.predict(im)

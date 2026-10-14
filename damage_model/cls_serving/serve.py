"""파손 분류 API 서버. shoe_app.py GUI로 시험한다.

    python serve.py --ckpt last.pt                 # http://127.0.0.1:8000
    python serve.py --ckpt last.pt --ngrok         # Colab 등 (환경변수 NGROK_AUTHTOKEN 필요)

GET  /health   모델 정보
POST /predict  form-data 'file' 사진 한 장 → model.py의 출력 + inference_ms
"""
import argparse
import io
import os
import time

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model import DamageClassifier

app = FastAPI(title="mendii damage part classifier")
CLF = {}


@app.get("/health")
def health():
    c = CLF["clf"]
    return {"status": "ok", "model": c.name, "classes": c.keys + ["other"], "threshold": c.threshold, "device": c.device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "이미지 파일이 아닙니다.")
    t0 = time.perf_counter()
    out = CLF["clf"].predict(img)
    out["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--threshold", type=float, help="기본은 체크포인트에 저장된 값(0.5)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ngrok", action="store_true")
    args = p.parse_args()

    CLF["clf"] = DamageClassifier(args.ckpt, threshold=args.threshold)
    if args.ngrok:
        from pyngrok import ngrok
        ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
        print("공개 주소 (GUI의 API 주소 칸에 입력):", ngrok.connect(args.port).public_url, flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

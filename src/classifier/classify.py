"""
폴더 안의 이미지를 분류해 labels.csv와 임베딩을 남긴다.

    python -m src.classifier.classify --src ./images --out ./out
    python -m src.classifier.classify --src ./images --out ./out --model large
    python -m src.classifier.classify --src ./images --out ./out --limit 64   # 벤치마크

출력
    out/labels.csv       품목·뷰·신발종류 + 신뢰도 + margin
    out/embeddings.npy   이미지 임베딩 (N x D). 유사검색·교차검증에 쓴다
    out/taxonomy.json    이 실행에 쓰인 클래스 정의 (재현성)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import taxonomy
from .model import MODELS, ZeroShotClassifier, load_images

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(src: str) -> list[str]:
    return sorted(f for f in os.listdir(src) if f.lower().endswith(IMG_EXT))


def run(src: str, out: str, model: str = "base", batch: int = 32,
        limit: int = 0, device: str | None = None) -> pd.DataFrame:
    os.makedirs(out, exist_ok=True)
    files = list_images(src)
    if limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    if not files:
        raise SystemExit(f"이미지가 없습니다: {src}")

    clf = ZeroShotClassifier(model=model, device=device)
    print(f"[모델] {clf.model_name}  device={clf.device}")
    print(f"[데이터] {len(files)}장")

    rows, embs = [], []
    t0 = time.time()
    for start in range(0, len(files), batch):
        chunk = files[start:start + batch]
        images = load_images([os.path.join(src, f) for f in chunk])
        preds, emb = clf.classify(images)
        embs.append(emb)

        for j, fname in enumerate(chunk):
            item = preds["item"][j]
            view = preds["view"][j]
            shoe = preds["shoe_type"][j]
            is_shoe = item.label == "shoe"
            rows.append({
                "file": fname,
                "item_type": item.label,
                "item_ko": taxonomy.korean("item", item.label),
                "item_conf": round(item.confidence, 4),
                "item_margin": round(item.margin, 4),
                "view": view.label,
                "view_ko": taxonomy.korean("view", view.label),
                "view_conf": round(view.confidence, 4),
                # 신발이 아니면 종류는 의미가 없다 — 비워둔다
                "shoe_type": shoe.label if is_shoe else "",
                "shoe_type_ko": taxonomy.korean("shoe_type", shoe.label) if is_shoe else "",
                "shoe_type_conf": round(shoe.confidence, 4) if is_shoe else None,
                "shoe_type_margin": round(shoe.margin, 4) if is_shoe else None,
            })

        done = start + len(chunk)
        if done % (batch * 10) == 0 or done == len(files):
            rate = done / (time.time() - t0)
            eta = (len(files) - done) / rate / 60
            print(f"  {done}/{len(files)}  {rate:.1f} img/s  ETA {eta:.1f}분", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "labels.csv"), index=False, encoding="utf-8-sig")
    np.save(os.path.join(out, "embeddings.npy"), np.concatenate(embs))
    with open(os.path.join(out, "taxonomy.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "model": clf.model_name,
            "item": taxonomy.as_dict("item"),
            "view": taxonomy.as_dict("view"),
            "shoe_type": taxonomy.as_dict("shoe_type"),
        }, fp, ensure_ascii=False, indent=1)

    summarize(df, time.time() - t0)
    return df


def summarize(df: pd.DataFrame, elapsed: float) -> None:
    print(f"\n[완료] {elapsed:.0f}초 · {len(df)}장")
    print("\n── 품목 ──")
    print(df.item_ko.value_counts().to_string())
    print("\n── 촬영 뷰 ──")
    print(df.view_ko.value_counts().to_string())
    shoes = df[df.item_type == "shoe"]
    if len(shoes):
        print(f"\n── 신발 종류 (신발 {len(shoes)}장) ──")
        print(shoes.shoe_type_ko.value_counts().to_string())
        low = (shoes.shoe_type_margin < 0.15).sum()
        print(f"\n판정 애매(margin<0.15): {low}장 ({low/len(shoes)*100:.1f}%) — 사람 확인 권장")


def main() -> None:
    ap = argparse.ArgumentParser(description="신발/의류 분류 (SigLIP2 zero-shot)")
    ap.add_argument("--src", required=True, help="이미지 폴더")
    ap.add_argument("--out", default="./out", help="결과 폴더")
    ap.add_argument("--model", default="base", choices=list(MODELS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="샘플링 (벤치마크용)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (기본: 자동)")
    a = ap.parse_args()
    run(a.src, a.out, a.model, a.batch, a.limit, a.device)


if __name__ == "__main__":
    main()

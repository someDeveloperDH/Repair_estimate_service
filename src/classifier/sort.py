"""
labels.csv를 폴더 구조로 배치하고, 임베딩으로 결과를 교차검증한다.

    python -m src.classifier.sort --src ./images --out ./out

원본은 건드리지 않는다. 하드링크로 배치하므로 추가 디스크가 0이고,
out/classified를 통째로 지워도 원본에 영향이 없다.

라우팅 우선순위
    1. 의류/가방/기타            → 0x_품목
    2. 클로즈업                  → 03_클로즈업     (종류 판별 불가)
    3. 밑창                      → 02_밑창샷       (종류 신뢰도 낮음)
    4. margin < MARGIN_MIN       → 09_검토필요/<추정종류>
    5. 나머지                    → 01_신발/<종류>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

MARGIN_MIN = 0.15


def crossval(out: str, df: pd.DataFrame) -> None:
    """프롬프트 라벨 vs 임베딩 군집의 일치도.

    주의: 순도가 낮다고 라벨이 틀린 것은 아니다. k-means는 임베딩을 색·소재·촬영각도로
    나누기 때문에, 시각적으로 다양한 클래스(스니커즈 등)는 여러 군집에 흩어진다.
    순도는 '이 클래스가 시각적으로 응집돼 있는가'를 보는 지표로 읽어야 한다.
    """
    path = os.path.join(out, "embeddings.npy")
    if not os.path.exists(path):
        print("[교차검증] embeddings.npy 없음 — 건너뜀")
        return
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        print("[교차검증] scikit-learn 미설치 — 건너뜀")
        return

    emb = np.load(path)
    mask = ((df.item_type == "shoe") & (df.view == "full")).values
    if mask.sum() < 50:
        print("[교차검증] 전체샷 신발이 너무 적음 — 건너뜀")
        return

    feats, labels = emb[mask], df.loc[mask, "shoe_type_ko"].values
    k = len(set(labels))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(feats)
    print(f"\n[교차검증] 전체샷 신발 {len(feats)}장, k={k}")
    print(f"  ARI {adjusted_rand_score(labels, km.labels_):.3f} · "
          f"NMI {normalized_mutual_info_score(labels, km.labels_):.3f}")

    ct = pd.crosstab(labels, km.labels_)
    purity = (ct.max(axis=1) / ct.sum(axis=1)).sort_values()
    print("  클래스별 군집 순도 (낮은 순):")
    for name, p in purity.items():
        print(f"    {name:18s} {p:.2f}  (n={ct.loc[name].sum()})")


def route(row) -> tuple[str, str | None]:
    if row.item_type != "shoe":
        return {"clothing": "04_의류", "bag": "05_가방"}.get(row.item_type, "06_기타"), None
    if row.view == "closeup":
        return "03_클로즈업", None
    if row.view == "sole":
        return "02_밑창샷", None
    if pd.notna(row.shoe_type_margin) and row.shoe_type_margin < MARGIN_MIN:
        return "09_검토필요", row.shoe_type_ko
    return "01_신발", row.shoe_type_ko


def link(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)          # 하드링크 — 추가 용량 0
    except OSError:
        shutil.copy2(src, dst)     # 다른 볼륨이면 복사


def sort_files(src: str, out: str, df: pd.DataFrame) -> pd.DataFrame:
    dst_root = os.path.join(out, "classified")
    if os.path.isdir(dst_root):
        shutil.rmtree(dst_root)

    rows = []
    for row in df.itertuples():
        top, sub = route(row)
        folder = os.path.join(dst_root, top, sub) if sub else os.path.join(dst_root, top)
        os.makedirs(folder, exist_ok=True)
        link(os.path.join(src, row.file), os.path.join(folder, row.file))
        rows.append({"파일명": row.file, "분류": top,
                     "품목": row.item_ko, "촬영뷰": row.view_ko,
                     "신발종류": row.shoe_type_ko or "",
                     "신뢰도": row.shoe_type_conf, "판정여유": row.shoe_type_margin,
                     "경로": os.path.relpath(os.path.join(folder, row.file), out)})

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(out, "classified.csv"), index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="분류 결과를 폴더로 배치 + 교차검증")
    ap.add_argument("--src", required=True, help="원본 이미지 폴더")
    ap.add_argument("--out", default="./out", help="classify.py가 쓴 결과 폴더")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.out, "labels.csv"))
    crossval(a.out, df)

    print("\n[배치] 하드링크 생성 중…")
    res = sort_files(a.src, a.out, df)

    print(f"\n[완료] {len(res)}장 → {os.path.join(a.out, 'classified')}")
    print("\n── 최상위 ──")
    print(res.분류.value_counts().to_string())
    shoes = res[res.분류 == "01_신발"]
    if len(shoes):
        print("\n── 01_신발 ──")
        print(shoes.신발종류.value_counts().to_string())
    review = res[res.분류 == "09_검토필요"]
    if len(review):
        print(f"\n── 09_검토필요 {len(review)}장 ──")
        print(review.신발종류.value_counts().to_string())


if __name__ == "__main__":
    main()

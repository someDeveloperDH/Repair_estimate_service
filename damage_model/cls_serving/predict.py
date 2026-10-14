"""서버 없이 사진(또는 폴더)을 바로 판단한다.

    python predict.py --ckpt last.pt --image shoe.jpg
    python predict.py --ckpt last.pt --image photos/
"""
import argparse
from pathlib import Path

from model import DamageClassifier

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--threshold", type=float)
    args = p.parse_args()

    clf = DamageClassifier(args.ckpt, threshold=args.threshold)
    path = Path(args.image)
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in EXT) if path.is_dir() else [path]
    for f in files:
        r = clf.predict_path(f)
        print(f"{f.name}: {r['part_ko']}  확신도 {r['confidence']:.2f}  {r['probs']}")


if __name__ == "__main__":
    main()

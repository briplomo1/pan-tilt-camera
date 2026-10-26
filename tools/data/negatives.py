"""Add verified no-face images to a dataset (false-positive test frames).

    python tools/data/negatives.py D:/photos/rooms --into data/a/test --count 100

Takes images from a folder you supply (rooms, offices, outdoors, posters without people; check
the licence of whatever public set they come from), drops any where the PC detector (YuNet, low
threshold) sees anything face-like, and copies up to --count of the rest into <into>/negatives/.
Each gets an empty entry in <into>/gt.json, which marks it as a frame with no face.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.pc_detectors import YuNet  # noqa: E402
from tools.common.sources import IMAGE_TYPES, load_ground_truth, save_ground_truth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="folder of candidate images")
    parser.add_argument("--into", required=True, help="dataset folder holding gt.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.3, help="reject images with any detection above this")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.source).rglob("*") if p.suffix.lower() in IMAGE_TYPES)
    random.Random(args.seed).shuffle(paths)
    into = Path(args.into)
    gt_path = into / "gt.json"
    gt = load_ground_truth(gt_path if gt_path.exists() else None)
    detector = YuNet(score_threshold=args.threshold)

    added = rejected = 0
    for path in paths:
        if added >= args.count:
            break
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if detector.detect(rgb):
            rejected += 1
            continue
        key = f"negatives/{path.stem}{path.suffix.lower()}"
        if key in gt:
            continue
        (into / "negatives").mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, into / key)
        gt[key] = []
        added += 1
    save_ground_truth(gt_path, gt)
    print(f"Added {added} no-face images to {into / 'negatives'} ({rejected} rejected as possibly containing a face)")
    if added < args.count:
        print(f"Only {added} of {args.count} wanted: supply more candidate images")


if __name__ == "__main__":
    main()

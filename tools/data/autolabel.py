"""Propose face boxes for a folder, capture session or video with a strong PC detector (YuNet).

    python tools/data/autolabel.py data/captures/desk_1m
    python tools/data/autolabel.py data/wide/walk.mp4 --threshold 0.5
    python tools/data/autolabel.py data/captures/desk_1m --accept       # trust them as ground truth

Writes <source>.proposals.json next to the source (or --out): {key: [[x0, y0, x1, y1, score], ...]}
in original pixels, keys as in tools/common/sources.py. Review and correct them with
tools/data/label.py, which turns them into ground truth. --accept skips review and writes
<source>.gt.json directly (fine for video ground truth in the virtual gimbal, where label.py
then only spot-checks; never for the test sets).

YuNet is OpenCV's face detector (opencv_zoo, downloaded once to build/models/); it is much
stronger than anything that fits on the ESP32, which is what a labeller should be.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.pc_detectors import YuNet  # noqa: E402
from tools.common.sources import open_source, save_ground_truth  # noqa: E402


def default_path(source: Path, suffix: str) -> Path:
    base = source if source.is_dir() else source.with_suffix("")
    return base.parent / f"{base.name}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="image folder, capture session or video")
    parser.add_argument("--threshold", type=float, default=0.6, help="YuNet score threshold")
    parser.add_argument("--stride", type=int, default=1, help="video: every Nth frame")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", help="output JSON (default <source>.proposals.json, or .gt.json with --accept)")
    parser.add_argument("--accept", action="store_true", help="write ground truth directly, no review")
    args = parser.parse_args()

    src = Path(args.source)
    detector = YuNet(score_threshold=args.threshold)
    proposals: dict[str, list[list[float]]] = {}
    with_faces = 0
    for i, frame in enumerate(open_source(src, stride=args.stride, limit=args.limit)):
        faces = detector.detect(frame.rgb)
        proposals[frame.key] = [[round(v, 1) for v in f[:4]] + [round(f[4], 3)] for f in faces]
        with_faces += bool(faces)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} frames, {with_faces} with faces")

    if args.accept:
        out = Path(args.out) if args.out else default_path(src, ".gt.json")
        save_ground_truth(out, {k: [b[:4] for b in v] for k, v in proposals.items()})
    else:
        out = Path(args.out) if args.out else default_path(src, ".proposals.json")
        out.write_text(json.dumps(proposals, indent=1))
    print(f"{len(proposals)} frames, {with_faces} with at least one face; wrote {out}")
    if not args.accept:
        print(f"Next: python tools/data/label.py {src} --proposals {out}")


if __name__ == "__main__":
    main()

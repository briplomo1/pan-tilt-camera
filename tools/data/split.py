"""Split a labelled set into calibration and test parts by session, not by frame (dataset B).

    python tools/data/split.py data/captures --gt data/b/gt.json --out data/b --cal 0.3

Frames from one capture session (or one video) are near-duplicates, so a frame-level split would
put almost the same image in both parts. Whole groups go to one side instead: a group is the first
folder of the key (the session, or the video stem). Groups are shuffled with --seed and added to
B-cal until it holds --cal of the frames; the rest is B-test.

Input: ground truth for every frame ({key: boxes}, keys relative to the root, as label.py writes
per session; pass several --gt files to merge sessions, each relative to its own folder's parent).
Writes <out>/gt_cal.json, <out>/gt_test.json and <out>/split.json (groups on each side, counts).
Score with --gt <out>/gt_test.json and pick thresholds with gt_cal.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.sources import load_ground_truth, save_ground_truth  # noqa: E402


def merge(gt_files: list[Path], root: Path) -> dict[str, list]:
    """Re-key each session's ground truth relative to `root`."""
    merged: dict[str, list] = {}
    for path in gt_files:
        gt = load_ground_truth(path)
        # label.py writes <session>.gt.json next to the session folder, keyed inside the folder
        session_dir = path.with_name(path.name.removesuffix(".gt.json"))
        prefix = session_dir.relative_to(root).as_posix() + "/" if session_dir.is_dir() else ""
        for key, boxes in gt.items():
            merged[prefix + key] = boxes
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="folder the merged keys are relative to (e.g. data/captures)")
    parser.add_argument("--gt", action="append", required=True,
                        help="ground truth file(s): one for the whole root, or <session>.gt.json per session")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cal", type=float, default=0.3, help="fraction of frames for calibration")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    root = Path(args.root)
    gt = merge([Path(g) for g in args.gt], root)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in gt:
        groups[key.split("/", 1)[0] if "/" in key else "(root)"].append(key)
    if len(groups) < 2:
        raise SystemExit("Only one session: capture several sessions so whole sessions can be held out")

    names = sorted(groups)
    random.Random(args.seed).shuffle(names)
    target = args.cal * len(gt)
    cal_groups, test_groups, cal_frames = [], [], 0
    for name in names:
        if cal_frames < target and cal_frames + len(groups[name]) <= target * 1.5:
            cal_groups.append(name)
            cal_frames += len(groups[name])
        else:
            test_groups.append(name)
    if not test_groups:
        test_groups.append(cal_groups.pop())

    out = Path(args.out)
    cal = {k: gt[k] for g in cal_groups for k in groups[g]}
    test = {k: gt[k] for g in test_groups for k in groups[g]}
    save_ground_truth(out / "gt_cal.json", cal)
    save_ground_truth(out / "gt_test.json", test)

    def counts(part: dict) -> dict:
        return {"frames": len(part), "faces": sum(len(b) for b in part.values()),
                "no_face_frames": sum(1 for b in part.values() if not b)}

    (out / "split.json").write_text(json.dumps({
        "root": str(root), "seed": args.seed, "cal_fraction_target": args.cal,
        "cal": {"groups": sorted(cal_groups), **counts(cal)}, "test": {"groups": sorted(test_groups), **counts(test)},
    }, indent=2))
    print(f"B-cal:  {len(cal)} frames from {len(cal_groups)} sessions ({len(cal) / len(gt):.0%})")
    print(f"B-test: {len(test)} frames from {len(test_groups)} sessions")
    print(f"Wrote {out / 'gt_cal.json'}, {out / 'gt_test.json'}, {out / 'split.json'}")


if __name__ == "__main__":
    main()

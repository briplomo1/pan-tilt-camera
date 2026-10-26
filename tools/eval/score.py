"""Score detection runs against ground truth: AP, PR curve, operating point, size buckets.

    python tools/eval/score.py results/20261025-101500_a_test_espdl-msrmnp-s8.jsonl --gt data/a_test/gt.json
    python tools/eval/score.py results/*_b_test_*.jsonl --gt data/b/gt.json --cal-run results/X_b_cal_X.jsonl
    python tools/eval/score.py results/pi_haar.jsonl --gt data/a_test/gt.json --images data/a_test

Reads the per-frame JSONL from tools/bench/run_bench.py (ESP32) or rsbpi/face_tracker.py --images
(Pi). Only frames that have a ground-truth entry are scored; an empty GT list marks a no-face frame.
Writes next to each run:

    <run>.score.json    AP50 [95 % CI], AP50:95, operating point (recall, FP/frame, size buckets,
                        centre error), the run's latency/memory summary if <run>.summary.json exists
    <run>.pr.csv        precision-recall curve at IoU 0.5

The operating threshold is, in order of preference: --threshold; the threshold chosen on
--cal-run (same model, calibration split) for FP per no-face frame <= --max-fp; or chosen on the
scored run itself (flagged "chosen_on": "test" because it leaks).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.imaging import FRAME_HEIGHT, FRAME_WIDTH  # noqa: E402
from tools.common.sources import load_ground_truth  # noqa: E402
from tools.eval import metrics as m  # noqa: E402


def frame_scale(width: int, height: int) -> float:
    return min(FRAME_WIDTH / width, FRAME_HEIGHT / height)


def load_entries(rows_path: Path, gt: dict, images_root: Path | None = None) -> tuple[list[m.Entry], dict]:
    """Entries for every scored frame, plus counts of what was skipped."""
    entries: list[m.Entry] = []
    skipped_unlabelled = 0
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("repeat", 0):
            continue                       # --repeat runs: score each frame once
        key = row["image"]
        if key not in gt:
            skipped_unlabelled += 1
            continue
        scale = None
        if "letterbox" in row:
            scale = row["letterbox"]["scale"]
        elif images_root is not None:
            from PIL import Image

            with Image.open(images_root / key) as im:
                scale = frame_scale(*im.size)
        dets = [(f["box"], f["score"]) for f in row["faces"]]
        entries.append(m.Entry(key, dets, [list(b) for b in gt[key]], scale, row.get("time_s", 0.0)))
    missing = len(set(gt) - {e.key for e in entries})
    return entries, {"unlabelled_frames": skipped_unlabelled, "gt_frames_not_run": missing}


def summarise(entries: list[m.Entry], threshold: float, chosen_on: str, bootstrap: int) -> dict:
    ap50 = m.average_precision(entries, 0.5)
    lo, hi = m.bootstrap_ci(entries, lambda e: m.average_precision(e, 0.5), n_resamples=bootstrap)
    op = m.evaluate_at_threshold(entries, threshold, 0.5)
    op["chosen_on"] = chosen_on
    return {
        "frames": len(entries),
        "gt_boxes": sum(len(e.gt) for e in entries),
        "no_face_frames": sum(1 for e in entries if not e.gt),
        "ap50": ap50, "ap50_ci": [lo, hi],
        "ap50_95": m.ap50_95(entries),
        "operating_point": op,
    }


def write_pr(path: Path, entries: list[m.Entry]) -> None:
    scores, precision, recall = m.pr_curve(entries, 0.5)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "precision", "recall"])
        for row in zip(scores, precision, recall):
            w.writerow([round(v, 5) for v in row])


def run_name(rows_path: Path) -> str:
    return rows_path.name[:-len(".jsonl")] if rows_path.name.endswith(".jsonl") else rows_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="results JSONL files")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--threshold", type=float, help="fixed operating threshold")
    parser.add_argument("--cal-run", help="run on the calibration split, used to choose the threshold")
    parser.add_argument("--cal-gt", help="ground truth for --cal-run (default: --gt)")
    parser.add_argument("--max-fp", type=float, default=0.05, help="FP per no-face frame at the operating point")
    parser.add_argument("--images", help="image root, for frame scale when rows have no letterbox (Pi runs)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    gt = load_ground_truth(args.gt)
    images_root = Path(args.images) if args.images else None
    cal_threshold = None
    if args.cal_run and args.threshold is None:
        cal_entries, _ = load_entries(Path(args.cal_run), load_ground_truth(args.cal_gt or args.gt), images_root)
        cal_threshold = m.operating_threshold(cal_entries, args.max_fp)
        print(f"Operating threshold from {Path(args.cal_run).name}: {cal_threshold:.4f}")

    for run in args.runs:
        rows_path = Path(run)
        entries, skipped = load_entries(rows_path, gt, images_root)
        if not entries:
            print(f"{rows_path.name}: no frames with ground truth - skipped")
            continue
        if args.threshold is not None:
            threshold, chosen_on = args.threshold, "fixed"
        elif cal_threshold is not None:
            threshold, chosen_on = cal_threshold, "cal"
        else:
            threshold, chosen_on = m.operating_threshold(entries, args.max_fp), "test"

        score = {"run": run_name(rows_path), "rows": str(rows_path), "gt": args.gt, **skipped,
                 **summarise(entries, threshold, chosen_on, args.bootstrap)}
        summary_path = rows_path.with_name(run_name(rows_path) + ".summary.json")
        if summary_path.exists():
            score["summary"] = json.loads(summary_path.read_text())
        out = rows_path.with_name(run_name(rows_path) + ".score.json")
        out.write_text(json.dumps(score, indent=2))
        write_pr(rows_path.with_name(run_name(rows_path) + ".pr.csv"), entries)

        op = score["operating_point"]
        sizes = "  ".join(f"{k} {v:.2f}" if v is not None else f"{k} -" for k, v in op["recall_by_size"].items())
        fp = op["fp_per_negative_frame"]
        print(f"{score['run']}: {score['frames']} frames, AP50 {score['ap50']:.3f} "
              f"[{score['ap50_ci'][0]:.3f}, {score['ap50_ci'][1]:.3f}], AP50:95 {score['ap50_95']:.3f}")
        print(f"    @ {threshold:.3f} ({chosen_on}): recall {op['recall']:.3f}, "
              f"FP/no-face frame {'-' if fp is None else f'{fp:.3f}'}, recall by size {sizes}")
        if chosen_on == "test":
            print("    note: threshold chosen on the scored frames themselves; pass --cal-run for an unbiased one")
        print(f"    wrote {out.name}")


if __name__ == "__main__":
    main()

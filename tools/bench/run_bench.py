"""Run a model on a folder of images over USB and record detections, latency and memory.

    python tools/bench/run_bench.py data/sample --model espdl-msrmnp-s8
    python tools/bench/run_bench.py data/sample --repeat 5 --show
    python tools/bench/run_bench.py data/wider_test --gt data/wider_test/gt.json --threshold 0.05

Each image is letterboxed to 320x240 and sent as RGB565, exactly what the camera delivers, so
the model does its own on-device resize just like in live use. The gimbal is paused (bench
mode) for the run. Output, in --out-dir:

    <run>.jsonl          one line per inference: faces (frame and original coords), timing, heap
    <run>.summary.json   model info and latency statistics

Ground truth (optional, for --show): JSON {"relative/path.jpg": [[x0, y0, x1, y1], ...]} in
original image pixels. Accuracy scoring itself is Phase 1 (score.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.imaging import FRAME_HEIGHT, FRAME_WIDTH, letterbox, rgb888_to_rgb565be  # noqa: E402
from tools.link import protocol as p  # noqa: E402
from tools.link.config import PROJECT_ROOT  # noqa: E402
from tools.link.device import DeviceLink  # noqa: E402

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp"}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def latency_stats(values_us: list[int]) -> dict:
    ms = [v / 1000 for v in values_us]
    if not ms:
        return {}
    return {"n": len(ms), "mean_ms": statistics.fmean(ms), "median_ms": statistics.median(ms),
            "p95_ms": percentile(ms, 95), "p99_ms": percentile(ms, 99), "min_ms": min(ms), "max_ms": max(ms)}


def show(rgb: np.ndarray, gt: list[list[float]], faces: list[p.Face], title: str) -> bool:
    """Draw ground truth (green) and predictions (red). Returns False if the user pressed q."""
    import cv2

    canvas = cv2.cvtColor(cv2.resize(rgb, (FRAME_WIDTH * 2, FRAME_HEIGHT * 2)), cv2.COLOR_RGB2BGR)
    for x0, y0, x1, y1 in gt:
        cv2.rectangle(canvas, (round(x0 * 2), round(y0 * 2)), (round(x1 * 2), round(y1 * 2)), (80, 220, 80), 2)
    for f in faces:
        cv2.rectangle(canvas, (f.x0 * 2, f.y0 * 2), (f.x1 * 2, f.y1 * 2), (60, 60, 230), 2)
        cv2.putText(canvas, f"{f.score:.2f}", (f.x0 * 2, max(f.y0 * 2 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 60, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("bench (green = ground truth, red = model)", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        key = cv2.waitKey(0) & 0xFF
    return key not in (ord("q"), 27)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", help="folder of images (searched recursively)")
    parser.add_argument("--port")
    parser.add_argument("--model", default="espdl-msrmnp-s8")
    parser.add_argument("--threshold", type=float, help="score threshold (default: the model's own)")
    parser.add_argument("--gt", help="ground-truth JSON, used by --show")
    parser.add_argument("--repeat", type=int, default=1, help="inferences per image (latency statistics)")
    parser.add_argument("--warmup", type=int, default=10, help="unrecorded inferences before the run")
    parser.add_argument("--limit", type=int, help="only the first N images")
    parser.add_argument("--show", action="store_true", help="display each result")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--stay-bench", action="store_true", help="leave the gimbal paused afterwards")
    args = parser.parse_args()

    root = Path(args.images)
    paths = sorted(q for q in root.rglob("*") if q.suffix.lower() in IMAGE_TYPES)[:args.limit]
    if not paths:
        raise SystemExit(f"No images in {root}")
    ground_truth = json.loads(Path(args.gt).read_text()) if args.gt else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model.replace('/', '_')}"
    rows_path = out_dir / f"{run}.jsonl"

    with DeviceLink(args.port, on_log=None) as device:
        hello = device.hello()
        device.set_mode(p.Mode.BENCH)
        try:
            info = device.load_model(args.model)
            if not info.ok:
                raise SystemExit(f"Model {args.model} failed to load")
            if args.threshold is not None:
                device.set_threshold(args.threshold)
            print(f"{info.id} ({info.backend}, {info.quantization}), input {info.input_width}x{info.input_height}, "
                  f"weights {info.model_bytes} B, load {info.load_us / 1000:.0f} ms, "
                  f"RAM internal {info.internal_bytes:+} B psram {info.psram_bytes:+} B")

            first, _ = letterbox(Image.open(paths[0]))
            first_bytes = rgb888_to_rgb565be(first)
            for _ in range(args.warmup):
                device.infer(first_bytes, FRAME_WIDTH, FRAME_HEIGHT)

            totals: list[int] = []
            min_free_internal = min_free_psram = None
            frame_id = 0
            started = time.monotonic()
            with open(rows_path, "w") as rows:
                for index, path in enumerate(paths):
                    rgb, box = letterbox(Image.open(path))
                    pixels = rgb888_to_rgb565be(rgb)
                    rel = path.relative_to(root).as_posix()
                    result = None
                    for repeat in range(args.repeat):
                        result = device.infer(pixels, FRAME_WIDTH, FRAME_HEIGHT, frame_id)
                        frame_id += 1
                        totals.append(result.timing.total_us)
                        min_free_internal = (result.min_free_internal if min_free_internal is None
                                             else min(min_free_internal, result.min_free_internal))
                        min_free_psram = (result.min_free_psram if min_free_psram is None
                                          else min(min_free_psram, result.min_free_psram))
                        rows.write(json.dumps({
                            "image": rel, "repeat": repeat, "frame_id": result.frame_id,
                            "timing_us": result.timing.as_dict(),
                            "min_free_internal": result.min_free_internal, "min_free_psram": result.min_free_psram,
                            "letterbox": box.as_dict(),
                            "faces": [{"box_frame": f.box, "box": [round(v, 1) for v in box.to_original(f.box)],
                                       "score": round(f.score, 4)} for f in result.faces],
                        }) + "\n")

                    print(f"[{index + 1}/{len(paths)}] {rel}: {len(result.faces)} face(s), "
                          f"{result.timing.total_us / 1000:.1f} ms")
                    if args.show:
                        gt = [box.to_frame(b) for b in ground_truth.get(rel, [])]
                        if not show(rgb, gt, result.faces, f"{rel}  {result.timing.total_us / 1000:.1f} ms"):
                            break

            summary = {
                "run": run, "images": str(root), "count": len(paths), "repeat": args.repeat,
                "warmup": args.warmup, "threshold": args.threshold, "model": asdict(info),
                "firmware": hello.firmware, "detect_total": latency_stats(totals),
                "min_free_internal": min_free_internal, "min_free_psram": min_free_psram,
                "wall_time_s": round(time.monotonic() - started, 1), "corrupt_frames": device.corrupt_frames,
            }
            (out_dir / f"{run}.summary.json").write_text(json.dumps(summary, indent=2))
            stats = summary["detect_total"]
            print(f"\nDetect time over {stats['n']} runs: median {stats['median_ms']:.2f} ms, "
                  f"mean {stats['mean_ms']:.2f}, p95 {stats['p95_ms']:.2f}, p99 {stats['p99_ms']:.2f}")
            print(f"Lowest free heap: internal {min_free_internal} B, psram {min_free_psram} B")
            print(f"Wrote {rows_path} and {run}.summary.json")
        finally:
            if not args.stay_bench:
                device.set_mode(p.Mode.TRACK)


if __name__ == "__main__":
    main()

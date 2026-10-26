"""Stream images or a video over USB to one or more models; record detections, latency and memory.

    python tools/bench/run_bench.py data/sample --model espdl-msrmnp-s8
    python tools/bench/run_bench.py data/sample --repeat 5 --show
    python tools/bench/run_bench.py recordings/walk.mp4 --stride 2 --show --record results/walk_b0.mp4
    python tools/bench/run_bench.py data/a_test --gt data/a_test/gt.json --threshold 0.05 \
        --model espdl-msrmnp-s8 --model tflite-micro/yunet-q8 --tag a_test

The source is a folder of images, a capture.py session or a video file (tools/common/sources.py).
Each frame is letterboxed to 320x240 and sent as RGB565, exactly what the camera delivers, so
the model does its own on-device resize just like in live use. With several --model options the
same frames go to each model in turn. The gimbal is paused (bench mode) for the run.

Output per model, in --out-dir (score them with tools/eval/score.py, compare with compare.py):

    <run>.jsonl          one line per inference: key, faces (frame and original coords), timing, heap
    <run>.summary.json   model info and latency statistics per stage

Ground truth (optional; for --show and later scoring): JSON {key: [[x0, y0, x1, y1], ...]} in
original image pixels. Keys are relative paths for folders, "<video stem>/<frame:05d>" for videos.
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
from tools.common.sources import load_ground_truth, open_source  # noqa: E402
from tools.link import protocol as p  # noqa: E402
from tools.link.config import PROJECT_ROOT  # noqa: E402
from tools.link.device import DeviceLink  # noqa: E402

STAGES = ("pre", "infer", "post", "total")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def latency_stats(values_us: list[int]) -> dict:
    ms = [v / 1000 for v in values_us]
    if not ms:
        return {}
    return {"n": len(ms), "mean_ms": statistics.fmean(ms), "median_ms": statistics.median(ms),
            "p95_ms": percentile(ms, 95), "p99_ms": percentile(ms, 99), "min_ms": min(ms), "max_ms": max(ms)}


class Overlay:
    """Ground truth (green) and predictions (red) on the 320x240 frame, shown and/or recorded."""

    def __init__(self, show: bool, record: Path | None, fps: float):
        import cv2

        self.cv2 = cv2
        self.show = show
        self.writer = None
        if record:
            record.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(record), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                          (FRAME_WIDTH * 2, FRAME_HEIGHT * 2))

    def add(self, rgb: np.ndarray, gt: list[list[float]], faces: list[p.Face], title: str) -> bool:
        """Returns False if the user pressed q."""
        cv2 = self.cv2
        canvas = cv2.cvtColor(cv2.resize(rgb, (FRAME_WIDTH * 2, FRAME_HEIGHT * 2)), cv2.COLOR_RGB2BGR)
        for x0, y0, x1, y1 in gt:
            cv2.rectangle(canvas, (round(x0 * 2), round(y0 * 2)), (round(x1 * 2), round(y1 * 2)), (80, 220, 80), 2)
        for f in faces:
            cv2.rectangle(canvas, (f.x0 * 2, f.y0 * 2), (f.x1 * 2, f.y1 * 2), (60, 60, 230), 2)
            cv2.putText(canvas, f"{f.score:.2f}", (f.x0 * 2, max(f.y0 * 2 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (60, 60, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if self.writer:
            self.writer.write(canvas)
        if not self.show:
            return True
        cv2.imshow("bench (green = ground truth, red = model)", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            key = cv2.waitKey(0) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        if self.writer:
            self.writer.release()
        if self.show:
            self.cv2.destroyAllWindows()


def run_model(device: DeviceLink, model: str, frames: list, args, ground_truth: dict, out_dir: Path,
              hello: p.Hello, source_fps: float) -> Path | None:
    info = device.load_model(model)
    if not info.ok:
        print(f"Model {model} failed to load - skipped")
        return None
    if args.threshold is not None:
        device.set_threshold(args.threshold)
    print(f"\n{info.id} ({info.backend}, {info.quantization}), input {info.input_width}x{info.input_height}, "
          f"weights {info.model_bytes} B, load {info.load_us / 1000:.0f} ms, "
          f"RAM internal {info.internal_bytes:+} B psram {info.psram_bytes:+} B")

    for _ in range(args.warmup):
        device.infer(frames[0][2], FRAME_WIDTH, FRAME_HEIGHT)

    run = f"{time.strftime('%Y%m%d-%H%M%S')}_{args.tag + '_' if args.tag else ''}{model.replace('/', '_')}"
    rows_path = out_dir / f"{run}.jsonl"
    record = None
    if args.record:
        record = Path(args.record)
        if len(args.model) > 1:
            record = record.with_name(f"{record.stem}_{model.replace('/', '_')}{record.suffix}")
    overlay = Overlay(args.show, record, source_fps) if (args.show or record) else None

    stage_us: dict[str, list[int]] = {s: [] for s in STAGES}
    min_free_internal = min_free_psram = None
    frame_id = 0
    started = time.monotonic()
    with open(rows_path, "w") as rows:
        for index, (key, rgb, pixels, box, time_s) in enumerate(frames):
            result = None
            for repeat in range(args.repeat):
                result = device.infer(pixels, FRAME_WIDTH, FRAME_HEIGHT, frame_id)
                frame_id += 1
                for stage, value in result.timing.as_dict().items():
                    stage_us[stage].append(value)
                min_free_internal = (result.min_free_internal if min_free_internal is None
                                     else min(min_free_internal, result.min_free_internal))
                min_free_psram = (result.min_free_psram if min_free_psram is None
                                  else min(min_free_psram, result.min_free_psram))
                rows.write(json.dumps({
                    "image": key, "repeat": repeat, "frame_id": result.frame_id, "time_s": round(time_s, 4),
                    "timing_us": result.timing.as_dict(),
                    "min_free_internal": result.min_free_internal, "min_free_psram": result.min_free_psram,
                    "letterbox": box.as_dict(),
                    "faces": [{"box_frame": f.box, "box": [round(v, 1) for v in box.to_original(f.box)],
                               "score": round(f.score, 4)} for f in result.faces],
                }) + "\n")

            if not args.quiet:
                print(f"[{index + 1}/{len(frames)}] {key}: {len(result.faces)} face(s), "
                      f"{result.timing.total_us / 1000:.1f} ms")
            if overlay:
                gt = [box.to_frame(b) for b in ground_truth.get(key, [])]
                if not overlay.add(rgb, gt, result.faces, f"{info.id}  {key}  {result.timing.total_us / 1000:.1f} ms"):
                    break
    if overlay:
        overlay.close()

    summary = {
        "run": run, "source": args.source, "count": len(frames), "repeat": args.repeat, "stride": args.stride,
        "warmup": args.warmup, "threshold": args.threshold, "tag": args.tag, "model": asdict(info),
        "firmware": hello.firmware,
        "detect_total": latency_stats(stage_us["total"]),
        "stages": {s: latency_stats(v) for s, v in stage_us.items() if any(v)},
        "min_free_internal": min_free_internal, "min_free_psram": min_free_psram,
        "wall_time_s": round(time.monotonic() - started, 1), "corrupt_frames": device.corrupt_frames,
    }
    (out_dir / f"{run}.summary.json").write_text(json.dumps(summary, indent=2))
    stats = summary["detect_total"]
    print(f"Detect time over {stats['n']} runs: median {stats['median_ms']:.2f} ms, "
          f"mean {stats['mean_ms']:.2f}, p95 {stats['p95_ms']:.2f}, p99 {stats['p99_ms']:.2f}")
    print(f"Lowest free heap: internal {min_free_internal} B, psram {min_free_psram} B")
    print(f"Wrote {rows_path} and {run}.summary.json")
    return rows_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="folder of images (searched recursively), capture session or video file")
    parser.add_argument("--port")
    parser.add_argument("--model", action="append", help="model id; repeat to run several (default espdl-msrmnp-s8)")
    parser.add_argument("--threshold", type=float, help="score threshold (default: the model's own)")
    parser.add_argument("--gt", help="ground-truth JSON, drawn by --show / --record")
    parser.add_argument("--repeat", type=int, default=1, help="inferences per frame (latency statistics)")
    parser.add_argument("--warmup", type=int, default=10, help="unrecorded inferences before each model's run")
    parser.add_argument("--limit", type=int, help="only the first N frames")
    parser.add_argument("--stride", type=int, default=1, help="video: keep every Nth frame")
    parser.add_argument("--start", type=float, default=0.0, help="video: start at this many seconds")
    parser.add_argument("--duration", type=float, help="video: only this many seconds")
    parser.add_argument("--show", action="store_true", help="display each result")
    parser.add_argument("--record", help="save the overlay as MP4 (one per model with several models)")
    parser.add_argument("--tag", default="", help="label folded into the run name, e.g. the dataset")
    parser.add_argument("--quiet", action="store_true", help="no line per frame")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--stay-bench", action="store_true", help="leave the gimbal paused afterwards")
    args = parser.parse_args()
    args.model = args.model or ["espdl-msrmnp-s8"]

    source = open_source(args.source, stride=args.stride, start=args.start, duration=args.duration, limit=args.limit)
    source_fps = getattr(source, "fps", 15.0) / args.stride
    # Convert once up front so every model gets byte-identical frames and the link isn't kept waiting.
    # A QVGA frame is 150 KB, so a 2-minute 30 fps clip is ~0.5 GB: use --stride/--duration on long videos.
    frames = []
    for frame in source:
        rgb, box = letterbox(Image.fromarray(frame.rgb))
        frames.append((frame.key, rgb, rgb888_to_rgb565be(rgb), box, frame.time_s))
    if not frames:
        raise SystemExit(f"No frames in {args.source}")
    print(f"{len(frames)} frames from {args.source}")
    ground_truth = load_ground_truth(args.gt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with DeviceLink(args.port, on_log=None) as device:
        hello = device.hello()
        device.set_mode(p.Mode.BENCH)
        try:
            for model in args.model:
                path = run_model(device, model, frames, args, ground_truth, out_dir, hello, source_fps)
                if path:
                    written.append(path)
        finally:
            if not args.stay_bench:
                device.set_mode(p.Mode.TRACK)
    if args.gt and written:
        print("\nScore with:  python tools/eval/score.py " + " ".join(str(w) for w in written) + f" --gt {args.gt}")


if __name__ == "__main__":
    main()

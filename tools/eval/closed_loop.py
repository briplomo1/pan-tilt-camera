"""Tracking metrics from a tracker log: live ESP32 telemetry, a Pi clip, or a virtual-gimbal run.

    python tools/eval/closed_loop.py recordings/live-20261018-211259.jsonl
    python tools/eval/closed_loop.py recordings/clip-20261016-020936.jsonl          # Pi, 640x480
    python tools/eval/closed_loop.py results/vg-*.jsonl --out results/vg_compare.md

Accepted inputs (detected from the rows):
    ESP32   tools/view/live_view.py  live-*.jsonl       (telemetry, every tracker frame)
    Pi      rsbpi/face_tracker.py    clip-*.jsonl       (faces as [x, y, w, h], 640x480 by default)
    virtual tools/rig/virtual_gimbal.py vg-*.jsonl      (frame rows + 20 ms mount ticks with ground truth)

Always reported (the checkpoint metrics): frames, loop fps, face detected %, lost count and
longest gap, median offset from centre, % of frames with the face in the central box, how fast
the mount moved and detection/centring by mount speed. With ground-truth face angles (virtual
gimbal, rig) also: RMS angular error, time on target (< 3 deg), lock ratio, reacquire time.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

SPEED_WINDOW_S = 0.3                       # mount speed = movement over the previous 0.3 s
SPEED_BINS = [(5, 20), (20, 35), (35, 50), (50, math.inf)]
ON_TARGET_DEG = 3.0


@dataclass
class Frame:
    t: float                               # seconds
    state: str                             # SCAN / TRACK / PAUSED
    pan: float
    tilt: float
    target: list[float] | None             # x0, y0, x1, y1 in frame pixels
    width: int
    height: int
    detect_ms: float | None


@dataclass
class Tick:
    t: float
    pan: float
    tilt: float
    gt_pan: float | None
    gt_tilt: float | None
    state: str
    locked: bool                           # tracker has a target this instant


def load(path: Path, frame_size: tuple[int, int]) -> tuple[list[Frame], list[Tick], str]:
    frames, ticks = [], []
    kind = "esp32"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "tick":
            ticks.append(Tick(r["t"], r["pan"], r["tilt"], r.get("gt_pan"), r.get("gt_tilt"), r["state"],
                              r.get("locked", False)))
            continue
        if r.get("kind") == "frame":
            kind = "virtual"
            w, h = r["source"]
            frames.append(Frame(r["t"], r["state"], r["pan"], r["tilt"], r["target"], w, h, r.get("detect_ms")))
            continue
        timing = r.get("timing_us", {})
        if "total_us" in timing:           # Pi: faces as x, y, w, h
            kind = "pi"
            target = None
            if r.get("target_index", -1) >= 0:
                x, y, w, h = r["faces"][r["target_index"]]["box"]
                target = [x, y, x + w, y + h]
            frames.append(Frame(r["time_us"] / 1e6, r["state"].upper(), r["pan"], r["tilt"], target,
                                frame_size[0], frame_size[1], timing["total_us"] / 1000))
        else:                              # ESP32 telemetry
            w, h = r.get("source") or (320, 240)
            target = r["target"]["box"] if r.get("target") else None
            frames.append(Frame(r["time_us"] / 1e6, r["state"].upper(), r["pan"], r["tilt"], target, w, h,
                                timing.get("total", 0) / 1000))
    frames.sort(key=lambda f: f.t)
    return frames, ticks, kind


def offsets(f: Frame) -> tuple[float, float]:
    """Face centre offset from the frame centre, as a fraction of half the frame (0 centred, 1 edge)."""
    cx = (f.target[0] + f.target[2]) / 2
    cy = (f.target[1] + f.target[3]) / 2
    return (cx - f.width / 2) / (f.width / 2), (cy - f.height / 2) / (f.height / 2)


def mount_speeds(frames: list[Frame]) -> list[float | None]:
    speeds: list[float | None] = []
    j = 0
    for i, f in enumerate(frames):
        while frames[j].t < f.t - SPEED_WINDOW_S:
            j += 1
        dt = f.t - frames[j].t
        speeds.append(math.hypot(f.pan - frames[j].pan, f.tilt - frames[j].tilt) / dt if dt > 0.1 else None)
    return speeds


def median(values):
    return statistics.median(values) if values else None


def frame_metrics(frames: list[Frame], central: float) -> dict:
    n = len(frames)
    duration = frames[-1].t - frames[0].t if n > 1 else 0.0
    seen = [f.target is not None for f in frames]
    first = next((i for i, s in enumerate(seen) if s), None)

    gaps = []                              # (length s, closed) for each run of misses after the first lock
    if first is not None:
        last_seen_t = frames[first].t
        in_gap = False
        for f, s in zip(frames[first:], seen[first:]):
            if s:
                if in_gap:
                    gaps.append((f.t - last_seen_t, True))
                in_gap = False
                last_seen_t = f.t
            else:
                in_gap = True
        if in_gap:
            gaps.append((frames[-1].t - last_seen_t, False))
    closed = [g for g, c in gaps if c]

    tracked = [f for f in frames if f.target is not None]
    offs = [offsets(f) for f in tracked]
    speeds = mount_speeds(frames)
    valid_speeds = [s for s in speeds if s is not None]

    by_speed = []
    for lo, hi in SPEED_BINS:
        sel = [(f, s) for f, s in zip(frames, speeds) if s is not None and lo <= s < hi]
        hit = [f for f, _ in sel if f.target is not None]
        by_speed.append({
            "speed": f"{lo}-{hi}" if hi != math.inf else f">{lo}", "frames": len(sel),
            "detected": len(hit) / len(sel) if sel else None,
            "median_h_offset_pct": (100 * median([abs(offsets(f)[0]) for f in hit])) if hit else None,
        })

    detect = [f.detect_ms for f in frames if f.detect_ms]
    return {
        "frames": n,
        "duration_s": duration,
        "loop_fps": (n - 1) / duration if duration > 0 else None,
        "detect_median_ms": median(detect),
        "face_detected": sum(seen) / n if n else 0.0,
        "face_detected_after_first_lock": sum(seen[first:]) / (n - first) if first is not None else None,
        "times_lost": len(gaps),
        "lost_under_0_2s": sum(1 for g in closed if g < 0.2),
        "longest_gap_s": max(closed) if closed else 0.0,
        "median_offset_h_pct": 100 * median([abs(x) for x, _ in offs]) if offs else None,
        "median_offset_v_pct": 100 * median([abs(y) for _, y in offs]) if offs else None,
        "central_box": central,
        "in_central_box": sum(1 for x, y in offs if abs(x) <= central and abs(y) <= central) / len(offs) if offs else None,
        "mount_speed_median_dps": median(valid_speeds),
        "mount_speed_max_dps": max(valid_speeds) if valid_speeds else None,
        "time_above_35dps": sum(1 for s in valid_speeds if s > 35) / len(valid_speeds) if valid_speeds else None,
        "pan_range": [min(f.pan for f in frames), max(f.pan for f in frames)] if frames else None,
        "tilt_range": [min(f.tilt for f in frames), max(f.tilt for f in frames)] if frames else None,
        "by_mount_speed": by_speed,
    }


def tick_metrics(ticks: list[Tick]) -> dict:
    """Ground-truth metrics, time-weighted over the uniform mount ticks."""
    visible = [t for t in ticks if t.gt_pan is not None]
    if not visible:
        return {}
    dt = ticks[1].t - ticks[0].t if len(ticks) > 1 else 0.02
    errors = [math.hypot(t.pan - t.gt_pan, t.tilt - t.gt_tilt) for t in visible]
    reacquire = []
    hidden_since = None
    waiting_from = None
    for t in ticks:
        if t.gt_pan is None:
            if hidden_since is None:
                hidden_since = t.t
            waiting_from = None
        else:
            if hidden_since is not None:
                waiting_from = t.t        # face visible again: time until the tracker has it
                hidden_since = None
            if waiting_from is not None and t.locked:
                reacquire.append(t.t - waiting_from)
                waiting_from = None
    return {
        "gt_visible_s": len(visible) * dt,
        "rms_error_deg": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "median_error_deg": median(errors),
        "time_on_target": sum(1 for e in errors if e < ON_TARGET_DEG) / len(errors),
        "lock_ratio": sum(1 for t in visible if t.state == "TRACK") / len(visible),
        "reacquire_s": reacquire,
        "reacquire_median_s": median(reacquire),
    }


def analyse(path: Path, frame_size: tuple[int, int], central: float) -> dict:
    frames, ticks, kind = load(path, frame_size)
    if not frames:
        raise SystemExit(f"{path}: no tracker frames")
    return {"file": path.name, "kind": kind, **frame_metrics(frames, central), **tick_metrics(ticks)}


def pct(v):
    return "-" if v is None else f"{100 * v:.0f} %"


def num(v, digits=1):
    return "-" if v is None else f"{v:.{digits}f}"


def table(results: list[dict]) -> str:
    rows = [
        ("Frames / duration", lambda r: f"{r['frames']} / {r['duration_s']:.1f} s"),
        ("Loop rate (fps)", lambda r: num(r["loop_fps"])),
        ("Detect median (ms)", lambda r: num(r["detect_median_ms"])),
        ("Face detected", lambda r: pct(r["face_detected"])),
        ("Times lost (all < 0.2 s)", lambda r: f"{r['times_lost']} ({r['lost_under_0_2s']})"),
        ("Longest gap (s)", lambda r: num(r["longest_gap_s"], 2)),
        ("Median offset h / v", lambda r: f"{num(r['median_offset_h_pct'])} % / {num(r['median_offset_v_pct'])} %"),
        ("Face in central box", lambda r: pct(r["in_central_box"])),
        ("Mount speed median / max (deg/s)", lambda r: f"{num(r['mount_speed_median_dps'], 0)} / {num(r['mount_speed_max_dps'], 0)}"),
        ("Time above 35 deg/s", lambda r: pct(r["time_above_35dps"])),
    ]
    for i, (lo, hi) in enumerate(SPEED_BINS):
        label = f"{lo}-{hi}" if hi != math.inf else f">{lo}"
        rows.append((f"Detected / h offset at {label} deg/s", lambda r, i=i: (
            f"{pct(r['by_mount_speed'][i]['detected'])} / {num(r['by_mount_speed'][i]['median_h_offset_pct'])} %"
            if r["by_mount_speed"][i]["frames"] >= 10 else f"{r['by_mount_speed'][i]['frames']} frames")))
    if any("rms_error_deg" in r for r in results):
        rows += [
            ("RMS angular error (deg)", lambda r: num(r.get("rms_error_deg"), 2)),
            ("Time on target (< 3 deg)", lambda r: pct(r.get("time_on_target"))),
            ("Lock ratio", lambda r: pct(r.get("lock_ratio"))),
            ("Reacquire median (s)", lambda r: num(r.get("reacquire_median_s"), 2)),
        ]
    out = ["| Metric | " + " | ".join(r["file"] for r in results) + " |", "|---" * (len(results) + 1) + "|"]
    out += ["| " + label + " | " + " | ".join(fn(r) for r in results) + " |" for label, fn in rows]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--frame-size", default="640x480", help="Pi clips: camera frame size (default 640x480)")
    parser.add_argument("--central", type=float, default=0.25,
                        help="central box half-size as a fraction of half the frame (default 0.25)")
    parser.add_argument("--out", help="write the table (.md) and full results (.json) here, without extension")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.frame_size.lower().split("x"))

    results = [analyse(Path(p), size, args.central) for p in args.logs]
    text = table(results)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(text + "\n")
        out.with_suffix(".json").write_text(json.dumps(results, indent=2))
        print(f"\nWrote {out.with_suffix('.md')} and .json")


if __name__ == "__main__":
    sys.exit(main())

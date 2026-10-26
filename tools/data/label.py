"""Review and correct face boxes: proposals in, ground truth out (replaces a CVAT/Label Studio step).

    python tools/data/label.py data/captures/desk_1m
    python tools/data/label.py data/wide/walk.mp4 --stride 15        # spot-check every 15th frame

Opens each frame with its boxes: saved ground truth if the frame was reviewed before, otherwise
the auto-labeller's proposals (<source>.proposals.json, from autolabel.py). Fix them and move on;
every move saves <source>.gt.json (or --gt), so you can stop and resume any time. Which frames
you reviewed is kept beside it in <gt>.reviewed.txt.

Mouse   drag            draw a new box
        right click     delete the box under the cursor
Keys    n / space       accept the boxes, next frame          b   previous frame
        c               clear all boxes (frame has no face)   j   jump to the next unreviewed frame
        x               un-review this frame (drop its ground truth)
        + / -           show proposals down to a lower / higher score
        q / Esc         save and quit

Ground truth: {key: [[x0, y0, x1, y1], ...]} in original pixels; [] = reviewed, no face. Frames
never reviewed are absent, and nothing scores them. With a video, frames between reviewed ones
keep whatever --base holds (e.g. autolabel.py --accept output), so a whole clip gets ground truth
from spot checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.sources import VideoFile, load_ground_truth, open_source, save_ground_truth, video_key  # noqa: E402

WINDOW = "label (drag: new box, right click: delete, n: next, b: back, c: no face, q: quit)"
MAX_W, MAX_H = 1280, 800


class Frames:
    """Random access to the frames of any source, by position."""

    def __init__(self, source, stride: int):
        self.source = source
        if isinstance(source, VideoFile):
            self.indices = list(range(source.first, source.last, max(1, stride)))
            self.keys = [video_key(source.path.stem, i) for i in self.indices]
        else:
            self.keys = [p.relative_to(source.root).as_posix() for p in source.paths]

    def __len__(self) -> int:
        return len(self.keys)

    def load(self, i: int) -> np.ndarray:
        if isinstance(self.source, VideoFile):
            return self.source.frame_at(self.indices[i])
        from PIL import Image

        return np.asarray(Image.open(self.source.paths[i]).convert("RGB"))


def default_path(source: Path, suffix: str) -> Path:
    base = source if source.is_dir() else source.with_suffix("")
    return base.parent / f"{base.name}{suffix}"


def reviewed_path(gt_path: Path) -> Path:
    return gt_path.with_name(gt_path.name.removesuffix(".json") + ".reviewed.txt")


class Labeller:
    """`gt` is the whole ground-truth file (other frames included, never dropped); `reviewed` the keys a
    person has checked. Unreviewed frames show proposals, else whatever the file or --base holds."""

    def __init__(self, frames: Frames, proposals: dict, gt: dict, reviewed: set[str], gt_path: Path, base: dict,
                 min_score: float):
        self.frames, self.proposals, self.gt, self.gt_path, self.base = frames, proposals, gt, gt_path, base
        self.reviewed = reviewed
        self.min_score = min_score
        self.i = 0
        self.boxes: list[list[float]] = []
        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None
        self.scale = 1.0
        self.image: np.ndarray | None = None

    def boxes_for(self, key: str) -> list[list[float]]:
        if key in self.reviewed:
            return [list(b) for b in self.gt.get(key, [])]
        if key in self.proposals:
            return [b[:4] for b in self.proposals[key] if len(b) < 5 or b[4] >= self.min_score]
        return [list(b) for b in self.gt.get(key, self.base.get(key, []))]

    def open(self, i: int) -> None:
        self.i = max(0, min(i, len(self.frames) - 1))
        self.image = self.frames.load(self.i)
        h, w = self.image.shape[:2]
        self.scale = min(MAX_W / w, MAX_H / h, 3.0)
        self.boxes = self.boxes_for(self.frames.keys[self.i])

    def accept(self) -> None:
        key = self.frames.keys[self.i]
        self.gt[key] = [[round(v, 1) for v in b] for b in self.boxes]
        self.reviewed.add(key)
        self.save()

    def unreview(self) -> None:
        key = self.frames.keys[self.i]
        self.reviewed.discard(key)
        if key in self.base:
            self.gt[key] = self.base[key]
        else:
            self.gt.pop(key, None)
        self.save()

    def save(self) -> None:
        save_ground_truth(self.gt_path, {**self.base, **self.gt})
        reviewed_path(self.gt_path).write_text("\n".join(sorted(self.reviewed)) + "\n")

    def on_mouse(self, event, x, y, flags, _param) -> None:
        ox, oy = x / self.scale, y / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            (x0, y0), (x1, y1) = self.drag_start, (x, y)
            if abs(x1 - x0) > 4 and abs(y1 - y0) > 4:
                self.boxes.append([min(x0, x1) / self.scale, min(y0, y1) / self.scale,
                                   max(x0, x1) / self.scale, max(y0, y1) / self.scale])
            self.drag_start = self.drag_now = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hits = [b for b in self.boxes if b[0] <= ox <= b[2] and b[1] <= oy <= b[3]]
            if hits:   # the smallest box under the cursor
                self.boxes.remove(min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))

    def render(self) -> np.ndarray:
        s = self.scale
        h, w = self.image.shape[:2]
        canvas = cv2.cvtColor(cv2.resize(self.image, (round(w * s), round(h * s))), cv2.COLOR_RGB2BGR)
        key = self.frames.keys[self.i]
        reviewed = key in self.reviewed
        colour = (80, 220, 80) if reviewed else (40, 200, 240)
        for b in self.boxes:
            cv2.rectangle(canvas, (round(b[0] * s), round(b[1] * s)), (round(b[2] * s), round(b[3] * s)), colour, 2)
        if self.drag_start and self.drag_now:
            cv2.rectangle(canvas, self.drag_start, self.drag_now, (255, 255, 255), 1)
        done = sum(1 for k in self.frames.keys if k in self.reviewed)
        status = (f"[{self.i + 1}/{len(self.frames)}] {key}   {'REVIEWED' if reviewed else 'proposal'}   "
                  f"boxes {len(self.boxes)}   reviewed {done}/{len(self.frames)}   min score {self.min_score:.2f}")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(canvas, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def loop(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.open(self.i)
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("n"), ord(" ")):
                self.accept()
                if self.i + 1 < len(self.frames):
                    self.open(self.i + 1)
            elif key == ord("b"):
                self.open(self.i - 1)
            elif key == ord("c"):
                self.boxes = []
            elif key == ord("x"):
                self.unreview()
                self.open(self.i)
            elif key == ord("j"):
                nxt = next((j for j in range(self.i + 1, len(self.frames)) if self.frames.keys[j] not in self.reviewed),
                           None)
                if nxt is not None:
                    self.open(nxt)
            elif key in (ord("+"), ord("=")):
                self.min_score = max(0.0, self.min_score - 0.05)
                self.open(self.i)
            elif key in (ord("-"), ord("_")):
                self.min_score = min(1.0, self.min_score + 0.05)
                self.open(self.i)
        cv2.destroyAllWindows()
        self.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="image folder, capture session or video")
    parser.add_argument("--proposals", help="default <source>.proposals.json")
    parser.add_argument("--gt", help="ground truth to write (and resume from); default <source>.gt.json")
    parser.add_argument("--base", help="video: ground truth for frames you don't review (autolabel --accept output)")
    parser.add_argument("--stride", type=int, default=1, help="video: review every Nth frame")
    parser.add_argument("--min-score", type=float, default=0.6, help="hide proposals below this score")
    parser.add_argument("--start", type=int, default=0, help="start at this position (default: first unreviewed)")
    args = parser.parse_args()

    src = Path(args.source)
    source = open_source(src)
    frames = Frames(source, args.stride)
    if not len(frames):
        raise SystemExit(f"No frames in {src}")
    proposals_path = Path(args.proposals) if args.proposals else default_path(src, ".proposals.json")
    proposals = json.loads(proposals_path.read_text()) if proposals_path.exists() else {}
    if not proposals:
        print(f"No proposals at {proposals_path}: drawing every box by hand (run autolabel.py first to save time)")
    gt_path = Path(args.gt) if args.gt else default_path(src, ".gt.json")
    base = load_ground_truth(args.base) if args.base else {}
    gt = load_ground_truth(gt_path if gt_path.exists() else None)
    marks = reviewed_path(gt_path)
    # Without the sidecar (older file, or written by hand), everything in the file counts as reviewed
    reviewed = set(marks.read_text().split()) if marks.exists() else set(gt) - set(base)

    labeller = Labeller(frames, proposals, gt, reviewed, gt_path, base, args.min_score)
    labeller.i = args.start or next((i for i, k in enumerate(frames.keys) if k not in reviewed), 0)
    labeller.loop()
    done = sum(1 for k in frames.keys if k in labeller.reviewed)
    print(f"Reviewed {done}/{len(frames)} frames; ground truth in {gt_path}")


if __name__ == "__main__":
    main()

"""Open-loop detection metrics (docs/EXPERIMENT_PLAN.md §7), shared by the ESP32 and the Pi.

Definitions are the ones rsbpi/face_tracker.py already uses for the Pi baseline (greedy matching
by best IoU in score order, all-point interpolated AP), so both platforms are scored the same way.
Pure Python + numpy; no device or OpenCV needed.

An Entry is one image: its detections [(box, score)], its ground-truth boxes, and the scale from
original pixels to the 320x240 frame (for size buckets and centre error in frame pixels).
Boxes are (x0, y0, x1, y1) in original image pixels.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

Box = Sequence[float]

FRAME_WIDTH = 320
# Size buckets by box height in the 320x240 frame (EXPERIMENT_PLAN §6)
BUCKETS = (("S", 0.0, 24.0), ("M", 24.0, 64.0), ("L", 64.0, math.inf))
IOU_RANGE = [0.5 + 0.05 * i for i in range(10)]   # COCO AP50:95


@dataclass
class Entry:
    key: str
    dets: list[tuple[Box, float]]
    gt: list[Box]
    scale: float | None = None        # original px -> 320x240 frame px
    time_s: float = 0.0
    meta: dict = field(default_factory=dict)


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def match(dets: Sequence[tuple[Box, float]], gt: Sequence[Box], iou_thresh: float) -> list[int]:
    """For each detection (in the given order) the matched GT index, or -1. Greedy: the detection
    takes the unmatched GT box it overlaps most."""
    taken = [False] * len(gt)
    result = []
    for box, _ in dets:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt):
            if taken[j]:
                continue
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            taken[best_j] = True
            result.append(best_j)
        else:
            result.append(-1)
    return result


def pr_curve(entries: Sequence[Entry], iou_thresh: float = 0.5) -> tuple[list[float], list[float], list[float]]:
    """(scores, precision, recall) after each detection in descending score order, over all images.
    Matching is per image in score order, as in VOC."""
    total_gt = sum(len(e.gt) for e in entries)
    hits: list[tuple[float, bool]] = []
    for e in entries:
        dets = sorted(e.dets, key=lambda d: d[1], reverse=True)
        for (box, score), j in zip(dets, match(dets, e.gt, iou_thresh)):
            hits.append((score, j >= 0))
    hits.sort(key=lambda h: h[0], reverse=True)
    scores, precision, recall = [], [], []
    tp = fp = 0
    for score, hit in hits:
        tp += hit
        fp += not hit
        scores.append(score)
        precision.append(tp / (tp + fp))
        recall.append(tp / total_gt if total_gt else 0.0)
    return scores, precision, recall


def average_precision(entries: Sequence[Entry], iou_thresh: float = 0.5) -> float:
    """All-point interpolated AP (VOC 2010+), as compute_ap in rsbpi/face_tracker.py."""
    if not sum(len(e.gt) for e in entries):
        return 0.0
    _, precision, recall = pr_curve(entries, iou_thresh)
    precision = precision[:]
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap, prev = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev)
        prev = r
    return ap


def ap50_95(entries: Sequence[Entry]) -> float:
    return sum(average_precision(entries, t) for t in IOU_RANGE) / len(IOU_RANGE)


def bucket_of(box: Box, scale: float | None) -> str | None:
    if scale is None:
        return None
    height = (box[3] - box[1]) * scale
    for name, lo, hi in BUCKETS:
        if lo <= height < hi:
            return name
    return None


def evaluate_at_threshold(entries: Sequence[Entry], score_thresh: float, iou_thresh: float = 0.5) -> dict:
    """Recall, false positives and centre error with detections below `score_thresh` dropped."""
    total_gt = tp = 0
    fp_negative = negatives = fp_all = 0
    bucket_gt = {name: 0 for name, _, _ in BUCKETS}
    bucket_hit = {name: 0 for name, _, _ in BUCKETS}
    centre_px: list[float] = []        # frame pixels when the scale is known, else original pixels
    for e in entries:
        kept = sorted(((b, s) for b, s in e.dets if s >= score_thresh), key=lambda d: d[1], reverse=True)
        matched = match(kept, e.gt, iou_thresh)
        total_gt += len(e.gt)
        hit_gt = set()
        for (box, _), j in zip(kept, matched):
            if j < 0:
                fp_all += 1
                if not e.gt:
                    fp_negative += 1
                continue
            tp += 1
            hit_gt.add(j)
            (px, py), (gx, gy) = centre(box), centre(e.gt[j])
            centre_px.append(math.hypot(px - gx, py - gy) * (e.scale or 1.0))
        if not e.gt:
            negatives += 1
        for j, g in enumerate(e.gt):
            name = bucket_of(g, e.scale)
            if name:
                bucket_gt[name] += 1
                bucket_hit[name] += j in hit_gt
    centre_sorted = sorted(centre_px)
    median_centre = centre_sorted[len(centre_sorted) // 2] if centre_sorted else None
    return {
        "threshold": score_thresh,
        "recall": tp / total_gt if total_gt else 0.0,
        "precision": tp / (tp + fp_all) if tp + fp_all else 0.0,
        "tp": tp, "fn": total_gt - tp, "fp": fp_all,
        "fp_per_negative_frame": fp_negative / negatives if negatives else None,
        "fp_per_frame": fp_all / len(entries) if entries else 0.0,
        "recall_by_size": {n: (bucket_hit[n] / bucket_gt[n] if bucket_gt[n] else None) for n in bucket_gt},
        "gt_by_size": bucket_gt,
        "centre_error_px": median_centre,
        "centre_error_pct": (100 * median_centre / FRAME_WIDTH
                             if median_centre is not None and all(e.scale for e in entries) else None),
    }


def operating_threshold(entries: Sequence[Entry], max_fp: float = 0.05, iou_thresh: float = 0.5) -> float:
    """Lowest score threshold whose false positives per no-face frame stay within `max_fp` (EXPERIMENT_PLAN §7).
    Without no-face frames, false positives per frame over all frames are used instead.
    Matching runs in score order, so a detection's hit/miss doesn't depend on the threshold and
    the false positives can be counted once."""
    negatives = sum(1 for e in entries if not e.gt)
    frames = negatives or len(entries)
    false_scores = []
    for e in entries:
        if negatives and e.gt:
            continue
        dets = sorted(e.dets, key=lambda d: d[1], reverse=True)
        false_scores += [s for (_, s), j in zip(dets, match(dets, e.gt, iou_thresh)) if j < 0]
    if not frames:
        return 0.5
    allowed = math.floor(max_fp * frames + 1e-9)   # false positives the budget allows
    false_scores.sort(reverse=True)
    if len(false_scores) <= allowed:
        return min((s for e in entries for _, s in e.dets), default=0.5)
    # Keep everything strictly above the first false positive that would break the budget
    return math.nextafter(false_scores[allowed], math.inf)


def centre_jitter(entries: Sequence[Entry]) -> float | None:
    """Spread (px, frame coordinates) of the top-scoring detection's centre over a static sequence:
    sqrt(var x + var y). This is what turns into servo jitter."""
    points = []
    for e in entries:
        if e.dets:
            box, _ = max(e.dets, key=lambda d: d[1])
            cx, cy = centre(box)
            s = e.scale or 1.0
            points.append((cx * s, cy * s))
    if len(points) < 2:
        return None
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    var = sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in points) / (len(points) - 1)
    return math.sqrt(var)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return float(s[f]) if f == c else s[f] + (s[c] - s[f]) * (k - f)


def bootstrap_ci(entries: Sequence[Entry], statistic: Callable[[Sequence[Entry]], float],
                 n_resamples: int = 1000, seed: int = 0, level: float = 95.0) -> tuple[float, float]:
    """Percentile bootstrap over images."""
    n = len(entries)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = [statistic([entries[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)]
    tail = (100 - level) / 2
    return percentile(samples, tail), percentile(samples, 100 - tail)


def paired_bootstrap(a: Sequence[Entry], b: Sequence[Entry], statistic: Callable[[Sequence[Entry]], float],
                     n_resamples: int = 1000, seed: int = 0) -> dict:
    """Is model A better than model B on the same images? Resamples image keys jointly.
    Returns the observed difference A - B, its 95 % CI and a one-sided p-value for A <= B."""
    by_key_b = {e.key: e for e in b}
    pairs = [(e, by_key_b[e.key]) for e in a if e.key in by_key_b]
    if not pairs:
        raise ValueError("The two runs share no images")
    rng = random.Random(seed)
    n = len(pairs)
    observed = statistic([p[0] for p in pairs]) - statistic([p[1] for p in pairs])
    diffs = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(statistic([p[0] for p in sample]) - statistic([p[1] for p in sample]))
    return {"images": n, "difference": observed, "ci": (percentile(diffs, 2.5), percentile(diffs, 97.5)),
            "p_a_not_better": sum(d <= 0 for d in diffs) / n_resamples}

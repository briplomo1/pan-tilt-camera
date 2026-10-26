"""Compare models side by side: one table in the EXPERIMENT_PLAN §7 layout plus a Pareto plot.

    python tools/eval/compare.py results/*_a_test_*.score.json --baseline espdl-msrmnp-s8
    python tools/eval/compare.py results/*.summary.json --out results/latency_only
    python tools/eval/compare.py results/*_b_test_*.score.json --paired espdl-msrmnp-s8 --gt data/b/gt.json

Inputs are score.json files (accuracy + the run's latency/memory) or plain summary.json files
(latency/memory only), from the ESP32 (run_bench.py) or the Pi (face_tracker.py). Writes
<out>.md, <out>.csv and <out>.png (AP50 against p95 detect time, bubble size = working memory,
Pareto-optimal models outlined). --paired adds a paired bootstrap test of each model's AP50
against the named one on the shared frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.link.config import PROJECT_ROOT  # noqa: E402

COLUMNS = [
    ("model", "Model"), ("backend", "Backend"), ("quant", "Quant"), ("input", "Input"),
    ("ap50", "AP50"), ("ap50_ci", "AP50 95% CI"), ("ap50_95", "AP50:95"),
    ("threshold", "Op. thr"), ("recall", "Recall@op"), ("fp_frame", "FP/no-face frame"),
    ("recall_s", "Recall S"), ("recall_m", "Recall M"), ("recall_l", "Recall L"), ("centre_px", "Centre err px"),
    ("pre_ms", "Pre ms"), ("infer_ms", "Infer ms"), ("post_ms", "Post ms"),
    ("median_ms", "Detect median ms"), ("p95_ms", "Detect p95 ms"), ("p99_ms", "Detect p99 ms"),
    ("model_kb", "Model KB"), ("ram_internal_kb", "RAM internal KB"), ("ram_psram_kb", "RAM PSRAM KB"),
    ("min_free_internal_kb", "Min free internal KB"), ("ap50_per_ms", "AP50 per ms"), ("pareto", "Pareto"),
]


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "ap50" in data:                      # score.json
        return {"score": data, "summary": data.get("summary", {}), "path": path}
    return {"score": None, "summary": data, "path": path}


def row_for(item: dict) -> dict:
    s, sc = item["summary"], item["score"]
    model = s.get("model", {})
    row: dict = {"file": item["path"].name}
    if isinstance(model, dict):             # ESP32 run_bench summary
        row.update(model=model.get("id"), backend=model.get("backend"), quant=model.get("quantization"),
                   input=f"{model.get('input_width')}x{model.get('input_height')}",
                   model_kb=_kb(model.get("model_bytes")), ram_internal_kb=_kb(model.get("internal_bytes")),
                   ram_psram_kb=_kb(model.get("psram_bytes")), min_free_internal_kb=_kb(s.get("min_free_internal")))
    else:                                   # Pi summary: flat fields
        row.update(model=model or None, backend=s.get("backend"), quant=s.get("precision"), input="-",
                   model_kb=_kb(s.get("model_bytes")),
                   ram_internal_kb=(s.get("memory") or {}).get("rss_delta_kb"))
    if not row.get("model"):
        row["model"] = (sc or {}).get("run") or item["path"].stem
    total = s.get("detect_total") or {}
    row.update(median_ms=total.get("median_ms"), p95_ms=total.get("p95_ms"), p99_ms=total.get("p99_ms"))
    for stage, key in (("pre", "pre_ms"), ("infer", "infer_ms"), ("post", "post_ms")):
        row[key] = ((s.get("stages") or {}).get(stage) or {}).get("median_ms")
    if sc:
        op = sc["operating_point"]
        sizes = op.get("recall_by_size", {})
        row.update(ap50=sc["ap50"], ap50_ci=sc["ap50_ci"], ap50_95=sc["ap50_95"], threshold=op["threshold"],
                   recall=op["recall"], fp_frame=op.get("fp_per_negative_frame"), recall_s=sizes.get("S"),
                   recall_m=sizes.get("M"), recall_l=sizes.get("L"), centre_px=op.get("centre_error_px"))
        if row.get("median_ms"):
            row["ap50_per_ms"] = sc["ap50"] / row["median_ms"]
    return row


def _kb(value) -> float | None:
    return None if value is None else value / 1024


def memory_kb(row: dict) -> float:
    return (row.get("ram_internal_kb") or 0) + (row.get("ram_psram_kb") or 0)


def mark_pareto(rows: list[dict]) -> None:
    """Pareto-optimal on (AP50 higher, p95 lower, working memory lower)."""
    scored = [r for r in rows if r.get("ap50") is not None and r.get("p95_ms") is not None]
    for r in scored:
        r["pareto"] = not any(
            o is not r and o["ap50"] >= r["ap50"] and o["p95_ms"] <= r["p95_ms"] and memory_kb(o) <= memory_kb(r)
            and (o["ap50"] > r["ap50"] or o["p95_ms"] < r["p95_ms"] or memory_kb(o) < memory_kb(r))
            for o in scored)


def fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key.startswith("d_"):
        return f"{value:+.3f}" if key == "d_ap50" else f"{value:+.1f}"
    if key == "ap50_ci":
        return f"{value[0]:.3f}-{value[1]:.3f}"
    if key == "pareto":
        return "Y" if value else ""
    if isinstance(value, float):
        if key in ("ap50", "ap50_95", "recall", "recall_s", "recall_m", "recall_l", "fp_frame", "threshold"):
            return f"{value:.3f}"
        if key == "ap50_per_ms":
            return f"{value:.4f}"
        return f"{value:.1f}"
    return str(value)


def add_baseline_deltas(rows: list[dict], baseline: str) -> list[tuple[str, str]]:
    base = next((r for r in rows if r["model"] == baseline), None)
    if not base:
        print(f"Baseline {baseline} not among the inputs - no delta columns")
        return []
    extra = []
    for key, label in (("ap50", f"AP50 vs {baseline}"), ("median_ms", f"Median ms vs {baseline}"),
                       ("p95_ms", f"p95 ms vs {baseline}")):
        if base.get(key) is None:
            continue
        for r in rows:
            r[f"d_{key}"] = None if r.get(key) is None else r[key] - base[key]
        extra.append((f"d_{key}", label))
    return extra


def add_paired(rows: list[dict], items: list[dict], reference: str, gt_path: str) -> list[tuple[str, str]]:
    from tools.common.sources import load_ground_truth
    from tools.eval import metrics as m
    from tools.eval.score import load_entries

    gt = load_ground_truth(gt_path)
    entries = {}
    for row, item in zip(rows, items):
        if item["score"]:
            entries[row["model"]], _ = load_entries(Path(item["score"]["rows"]), gt)
    if reference not in entries:
        print(f"--paired: {reference} has no score file among the inputs")
        return []
    for row in rows:
        if row["model"] in entries and row["model"] != reference:
            res = m.paired_bootstrap(entries[row["model"]], entries[reference], lambda e: m.average_precision(e, 0.5))
            row["paired"] = f"{res['difference']:+.3f} [{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}] p={res['p_a_not_better']:.3f}"
    return [("paired", f"AP50 vs {reference} (paired)")]


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r.get("ap50") is not None and r.get("p95_ms") is not None]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mem = [max(memory_kb(r), 1.0) for r in pts]
    biggest = max(mem)
    for r, m_kb in zip(pts, mem):
        ax.scatter(r["p95_ms"], r["ap50"], s=60 + 900 * m_kb / biggest, alpha=0.55,
                   edgecolors="black" if r.get("pareto") else "none", linewidths=1.5)
        ax.annotate(r["model"], (r["p95_ms"], r["ap50"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    front = sorted((r for r in pts if r.get("pareto")), key=lambda r: r["p95_ms"])
    if len(front) > 1:
        ax.plot([r["p95_ms"] for r in front], [r["ap50"] for r in front], "--", color="grey", linewidth=1)
    ax.set_xlabel("Detect time p95 (ms)")
    ax.set_ylabel("AP50")
    ax.set_title("Accuracy vs latency (bubble = working memory, outlined = Pareto)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="score.json and/or summary.json files")
    parser.add_argument("--baseline", help="model id for Δ columns, e.g. espdl-msrmnp-s8")
    parser.add_argument("--paired", help="model id to test every other model's AP50 against (needs --gt)")
    parser.add_argument("--gt", help="ground truth, for --paired")
    parser.add_argument("--out", help="output path without extension (default results/compare-<time>)")
    args = parser.parse_args()

    items = [load(Path(f)) for f in args.files]
    rows = [row_for(i) for i in items]
    mark_pareto(rows)
    for r in rows:
        r.setdefault("pareto", None)
    columns = [c for c in COLUMNS if any(r.get(c[0]) is not None for r in rows)]
    if args.baseline:
        columns += add_baseline_deltas(rows, args.baseline)
    if args.paired:
        if not args.gt:
            raise SystemExit("--paired needs --gt")
        columns += add_paired(rows, items, args.paired, args.gt)
    rows.sort(key=lambda r: (-(r.get("ap50") or 0), r.get("p95_ms") or 0))

    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"compare-{time.strftime('%Y%m%d-%H%M%S')}"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(k, r.get(k)) for k, _ in columns) + " |")
    table = "\n".join(lines)
    out.with_suffix(".md").write_text(table + "\n\nSources: " + ", ".join(i["path"].name for i in items) + "\n")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([label for _, label in columns])
        for r in rows:
            w.writerow([fmt(k, r.get(k)) for k, _ in columns])
    plot(rows, out.with_suffix(".png"))
    print(table)
    print(f"\nWrote {out.with_suffix('.md')}, .csv" + (", .png" if any(r.get("ap50") is not None for r in rows) else ""))


if __name__ == "__main__":
    main()

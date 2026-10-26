"""Detection metrics on hand-made cases, and agreement with the Pi's own scorer.

    python -m unittest discover -s tools/tests -t .
"""
from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.eval import metrics as m  # noqa: E402

A = [10, 10, 50, 50]
B = [100, 100, 140, 160]


def entry(key, dets, gt, scale=1.0):
    return m.Entry(key, [(list(b), s) for b, s in dets], [list(g) for g in gt], scale)


class IouTest(unittest.TestCase):
    def test_cases(self):
        self.assertAlmostEqual(m.iou(A, A), 1.0)
        self.assertEqual(m.iou(A, B), 0.0)
        self.assertAlmostEqual(m.iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)


class ApTest(unittest.TestCase):
    def test_perfect(self):
        es = [entry("a", [(A, 0.9)], [A]), entry("b", [(B, 0.8)], [B])]
        self.assertAlmostEqual(m.average_precision(es), 1.0)
        self.assertAlmostEqual(m.ap50_95(es), 1.0)

    def test_missed_face_halves_recall(self):
        es = [entry("a", [(A, 0.9)], [A]), entry("b", [], [B])]
        self.assertAlmostEqual(m.average_precision(es), 0.5)

    def test_duplicate_is_a_false_positive(self):
        es = [entry("a", [(A, 0.9), ([11, 11, 51, 51], 0.8)], [A])]
        _, precision, recall = m.pr_curve(es)
        self.assertEqual(recall, [1.0, 1.0])
        self.assertEqual(precision, [1.0, 0.5])
        self.assertAlmostEqual(m.average_precision(es), 1.0)    # the FP ranks below the hit

    def test_high_scoring_false_positive_costs_precision(self):
        es = [entry("a", [(B, 0.95), (A, 0.9)], [A])]
        self.assertAlmostEqual(m.average_precision(es), 0.5)

    def test_offset_box_fails_strict_iou(self):
        shifted = [20, 10, 60, 50]                              # IoU 0.6 with A
        es = [entry("a", [(shifted, 0.9)], [A])]
        self.assertAlmostEqual(m.average_precision(es, 0.5), 1.0)
        self.assertAlmostEqual(m.average_precision(es, 0.75), 0.0)

    def test_no_ground_truth(self):
        self.assertEqual(m.average_precision([entry("n", [(A, 0.9)], [])]), 0.0)


class ThresholdTest(unittest.TestCase):
    def test_operating_point_respects_fp_budget(self):
        faces = [entry(f"f{i}", [(A, 0.9 - i * 0.01)], [A]) for i in range(10)]
        # 20 no-face frames; one false positive allowed at 0.05 FP/frame
        negatives = [entry(f"n{i}", [(B, s)], []) for i, s in enumerate([0.7, 0.5, 0.3])]
        negatives += [entry(f"e{i}", [], []) for i in range(17)]
        t = m.operating_threshold(faces + negatives, max_fp=0.05)
        r = m.evaluate_at_threshold(faces + negatives, t)
        self.assertLessEqual(r["fp_per_negative_frame"], 0.05)
        self.assertGreater(t, 0.5)
        self.assertLessEqual(t, 0.7)
        self.assertEqual(r["recall"], 1.0)

    def test_size_buckets_and_centre_error(self):
        small = [0, 0, 20, 20]       # 20 px tall at scale 1 -> S
        large = [0, 0, 80, 80]       # L
        es = [entry("a", [([2, 0, 22, 20], 0.9)], [small]), entry("b", [], [large])]
        r = m.evaluate_at_threshold(es, 0.5)
        self.assertEqual(r["recall_by_size"]["S"], 1.0)
        self.assertEqual(r["recall_by_size"]["L"], 0.0)
        self.assertIsNone(r["recall_by_size"]["M"])
        self.assertAlmostEqual(r["centre_error_px"], 2.0)

    def test_jitter(self):
        es = [entry(str(i), [([10 + d, 10, 50 + d, 50], 0.9)], []) for i, d in enumerate([0, 2, 0, 2])]
        self.assertAlmostEqual(m.centre_jitter(es), (4 / 3) ** 0.5)


class BootstrapTest(unittest.TestCase):
    def test_ci_contains_estimate(self):
        rng = random.Random(1)
        es = [entry(str(i), [(A, rng.random())] if rng.random() < 0.8 else [], [A]) for i in range(60)]
        ap = m.average_precision(es)
        lo, hi = m.bootstrap_ci(es, m.average_precision, n_resamples=200)
        self.assertLessEqual(lo, ap)
        self.assertGreaterEqual(hi, ap)

    def test_paired(self):
        good = [entry(str(i), [(A, 0.9)], [A]) for i in range(30)]
        bad = [entry(str(i), [(A, 0.9)] if i % 2 else [], [A]) for i in range(30)]
        res = m.paired_bootstrap(good, bad, m.average_precision, n_resamples=200)
        self.assertGreater(res["difference"], 0.3)
        self.assertLess(res["p_a_not_better"], 0.05)


class PiAgreementTest(unittest.TestCase):
    """Same AP as rsbpi/face_tracker.py's compute_ap on random cases (that file needs cv2 to import)."""

    def test_matches_pi_compute_ap(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 not installed")
        path = Path(__file__).resolve().parents[2] / "rsbpi" / "face_tracker.py"
        spec = importlib.util.spec_from_file_location("pi_face_tracker", path)
        pi = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(pi)
        except ImportError as e:     # Pi-only imports (picamera2 etc.) are optional
            self.skipTest(f"face_tracker.py not importable here: {e}")
        rng = random.Random(7)
        for _ in range(20):
            es = []
            for i in range(15):
                gt = [[rng.uniform(0, 200), rng.uniform(0, 150)] for _ in range(rng.randint(0, 3))]
                gt = [[x, y, x + rng.uniform(15, 60), y + rng.uniform(15, 60)] for x, y in gt]
                dets = [([g[0] + rng.uniform(-8, 8), g[1] + rng.uniform(-8, 8), g[2] + rng.uniform(-8, 8),
                          g[3] + rng.uniform(-8, 8)], rng.random()) for g in gt if rng.random() < 0.85]
                dets += [([rng.uniform(0, 200), rng.uniform(0, 150), 250, 200], rng.random() * 0.6)
                         for _ in range(rng.randint(0, 2))]
                es.append(m.Entry(str(i), dets, gt))
            pi_entries = [([(tuple(b), s) for b, s in e.dets], [tuple(g) for g in e.gt]) for e in es]
            self.assertAlmostEqual(m.average_precision(es, 0.5), pi.compute_ap(pi_entries, 0.5), places=9)


if __name__ == "__main__":
    unittest.main()

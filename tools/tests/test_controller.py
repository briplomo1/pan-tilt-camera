"""The Python mirror of the tracking controller (tools/rig/controller.py) and the virtual gimbal.

    python -m unittest discover -s tools/tests -t .

The filter tests state the behaviour src/FaceFilter.h and src/Tracker.cpp are meant to have; the
virtual-gimbal test runs the whole loop on a synthetic video with a perfect detector.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.rig import controller as ctl  # noqa: E402

DT = 0.055        # one B0 tracker frame


class ConstantsTest(unittest.TestCase):
    def test_read_from_firmware(self):
        for name in ("kFaceAccelNoise", "kFaceMeasNoise", "kFaceGate", "kMaxRejects", "kLeadS", "kCoastS",
                     "kFilterStaleUs", "kGain", "kDeadband", "kTrackSpeedDegS", "kLostTimeoutUs", "kHFovDeg",
                     "kVFovDeg", "kPanDir", "kTiltDir", "kScanRowOffsets", "kInitialVelocitySigma"):
            self.assertIn(name, ctl.C)
        self.assertIsInstance(ctl.C["kScanRowOffsets"], list)


class AxisFilterTest(unittest.TestCase):
    def test_learns_constant_velocity(self):
        f = ctl.AxisFilter()
        angle = 80.0
        for _ in range(40):
            f.predict(DT)
            self.assertTrue(f.update(angle))
            angle += 30.0 * DT
        self.assertAlmostEqual(f.velocity, 30.0, delta=2.0)
        self.assertAlmostEqual(f.angle, angle - 30.0 * DT, delta=0.5)

    def test_smooths_noise(self):
        rng = np.random.default_rng(3)
        f = ctl.AxisFilter()
        errors = []
        for i in range(100):
            f.predict(DT)
            f.update(90.0 + rng.normal(0, 0.6))
            if i > 20:
                errors.append(f.angle - 90.0)
        # Less jitter than the raw detections. Only mildly: kFaceAccelNoise lets heads change speed
        # fast, so the filter weights each new detection heavily.
        self.assertLess(float(np.std(errors)), 0.6 * 0.9)

    def test_gate_rejects_outlier(self):
        f = ctl.AxisFilter()
        for _ in range(20):
            f.predict(DT)
            f.update(90.0)
        f.predict(DT)
        self.assertFalse(f.update(120.0))
        self.assertAlmostEqual(f.angle, 90.0, delta=0.1)


def make_tracker(**options):
    mount = ctl.Mount(ctl.Range(0, 180), ctl.Range(60, 180), 150, 90, 150)
    return mount, ctl.Tracker(mount, ctl.Options(**options))


def face_at(pan_offset_deg: float) -> ctl.Face:
    """A face this many degrees right of where the mount points, on a 320x240 frame."""
    x = 160 + pan_offset_deg / (ctl.C["kHFovDeg"] / 2) * 160
    return ctl.Face(x - 20, 100, x + 20, 140, 0.9)


class TrackerTest(unittest.TestCase):
    def run_frames(self, tracker, mount, face_angle_fn, frames, detect_us=45_000, loop_us=55_000, drop=()):
        t = 0
        for i in range(frames):
            pan_cap = mount.pan
            # face direction in mount pan degrees; a face to the right is at lower pan (kPanDir = -1)
            offset = (face_angle_fn(t / 1e6) - pan_cap) / ctl.C["kPanDir"]
            faces = [] if i in drop else [face_at(offset)]
            now = t + detect_us
            mount.advance(now)
            tracker.step(faces, 320, 240, t, pan_cap, mount.tilt, now)
            t += loop_us
            mount.advance(t)
        return t

    def test_locks_onto_still_face(self):
        mount, tracker = make_tracker()
        self.run_frames(tracker, mount, lambda t: 100.0, 60)
        self.assertEqual(tracker.state, "TRACK")
        self.assertAlmostEqual(mount.pan, 100.0, delta=ctl.C["kDeadband"] * ctl.C["kHFovDeg"] / 2 + 0.5)

    def test_filter_lags_less_on_a_moving_face(self):
        lag = {}
        for on in (True, False):
            mount, tracker = make_tracker(filter=on)
            speed = 30.0
            end = self.run_frames(tracker, mount, lambda t: 70.0 + speed * t, 40)   # stays inside 0-180
            lag[on] = abs((70.0 + speed * end / 1e6) - mount.pan)
        self.assertLess(lag[True], lag[False])

    def test_coasts_through_a_short_miss_and_holds_after(self):
        mount, tracker = make_tracker()
        speed = 30.0
        self.run_frames(tracker, mount, lambda t: 90.0 + speed * t, 40)
        before = tracker.pan_target
        tracker.step([], 320, 240, 40 * 55_000, mount.pan, mount.tilt, 40 * 55_000 + 45_000)
        self.assertTrue(tracker.coasting)
        self.assertGreater(tracker.pan_target, before)          # kept moving with the face
        # Well past the coast window: hold, still TRACK until the lost timeout
        late = 40 * 55_000 + int(1.0e6)
        tracker.step([], 320, 240, late, mount.pan, mount.tilt, late + 45_000)
        self.assertFalse(tracker.coasting)
        self.assertEqual(tracker.state, "TRACK")

    def test_no_coast_without_filter(self):
        mount, tracker = make_tracker(filter=False)
        self.run_frames(tracker, mount, lambda t: 90.0 + 30.0 * t, 30)
        target = tracker.pan_target
        tracker.step([], 320, 240, 30 * 55_000, mount.pan, mount.tilt, 30 * 55_000 + 45_000)
        self.assertFalse(tracker.coasting)
        self.assertEqual(tracker.pan_target, target)

    def test_lead_is_bounded(self):
        mount, tracker = make_tracker()
        tracker.started = True
        tracker.enter_track()
        tracker.pan_filter.init(100.0)
        tracker.pan_filter.velocity = 1000.0                    # absurd estimate
        aim = tracker.aim(tracker.pan_filter, 90.0, 0.1, ctl.C["kHFovDeg"] / 2)
        self.assertLessEqual(aim, 100.0 + ctl.C["kTrackSpeedDegS"] * 0.1 + 1e-6)

    def test_stale_filter_restarts(self):
        mount, tracker = make_tracker()
        self.run_frames(tracker, mount, lambda t: 90.0 + 30.0 * t, 30)
        self.assertNotAlmostEqual(tracker.pan_filter.velocity, 0.0, places=1)
        later = 30 * 55_000 + ctl.C["kFilterStaleUs"] + 100_000
        tracker.step([face_at(0)], 320, 240, later, mount.pan, mount.tilt, later + 45_000)
        self.assertEqual(tracker.pan_filter.velocity, 0.0)

    def test_lost_face_returns_to_scan(self):
        mount, tracker = make_tracker()
        self.run_frames(tracker, mount, lambda t: 100.0, 10)
        t = 10 * 55_000 + ctl.C["kLostTimeoutUs"] + 100_000
        tracker.step([], 320, 240, t, mount.pan, mount.tilt, t + 45_000)
        self.assertEqual(tracker.state, "SCAN")


class SyntheticVideo:
    """Black frames with a white square 'face' moving on a sine; its box is the ground truth."""
    fps = 30.0
    width = 1280
    height = 720
    frame_count = 300

    def box(self, index: int) -> list[float]:
        t = index / self.fps
        cx = self.width / 2 + 250 * math.sin(2 * math.pi * 0.25 * t)
        cy = self.height / 2
        return [cx - 40, cy - 50, cx + 40, cy + 50]

    def frame(self, index: int):
        if index >= self.frame_count:
            return None
        img = np.zeros((self.height, self.width, 3), np.uint8)
        x0, y0, x1, y1 = (round(v) for v in self.box(index))
        img[y0:y1, x0:x1] = 255
        return img

    def faces(self, index: int):
        return [self.box(index)]


def white_box_detector(detect_us: int):
    def detect(view):
        ys, xs = np.nonzero(view[..., 0] > 128)
        if len(xs) < 20:
            return [], detect_us
        return [(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), 0.99)], detect_us
    return detect


class VirtualGimbalTest(unittest.TestCase):
    def run_vg(self, detect_us: int, filter_on: bool = True):
        from tools.eval import closed_loop as cl
        from tools.rig import virtual_gimbal as vg

        rows = []
        cfg = vg.RunConfig(hfov=100.0, overhead_us=10_000, options=ctl.Options(filter=filter_on))
        vg.run(SyntheticVideo(), white_box_detector(detect_us), cfg, rows.append)
        frames = [r for r in rows if r["kind"] == "frame"]
        ticks = [cl.Tick(r["t"], r["pan"], r["tilt"], r["gt_pan"], r["gt_tilt"], r["state"], r["locked"])
                 for r in rows if r["kind"] == "tick"]
        return frames, cl.tick_metrics(ticks)

    def test_tracks_the_moving_face(self):
        frames, metrics = self.run_vg(45_000)
        self.assertEqual(frames[-1]["state"], "TRACK")
        self.assertGreater(metrics["lock_ratio"], 0.8)
        self.assertLess(metrics["median_error_deg"], 3.0)

    def test_slow_model_skips_frames(self):
        fast, _ = self.run_vg(15_000)
        slow, _ = self.run_vg(150_000)
        self.assertGreater(len(fast), 2.5 * len(slow))
        steps = np.diff([r["frame_index"] for r in slow])
        self.assertGreaterEqual(int(np.median(steps)), 4)        # 160 ms per frame at 30 fps


if __name__ == "__main__":
    unittest.main()

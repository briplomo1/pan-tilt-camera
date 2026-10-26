"""Python mirror of the firmware's tracking controller, for the virtual gimbal.

    AxisFilter   src/FaceFilter.h     constant-velocity Kalman filter, one axis
    Mount        src/PanTilt.cpp      rate-limited servo targets, updated every 20 ms
    Tracker      src/Tracker.cpp      scan / track / coast / hold logic, steering and filter use

The tuning constants are read from src/Tracker.cpp and src/FaceFilter.h at import, so the mirror
can't drift from the firmware's numbers. The logic itself is a line-by-line port; keep it in step
with Tracker.cpp (tools/tests/test_controller.py checks the behaviour both should have).
Times are integer microseconds, like esp_timer.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CONST = re.compile(r"constexpr\s+(?:const\s+)?[\w:<>,\s]+?\s+(k\w+)\s*=\s*([^;]+);")


def _value(expr: str):
    expr = expr.strip()
    if expr in ("true", "false"):
        return expr == "true"
    if expr.startswith("{"):
        return [_value(v) for v in expr.strip("{}").split(",") if v.strip()]
    cleaned = re.sub(r"(\d+(?:\.\d*)?)f\b", r"\1", expr)
    if not re.fullmatch(r"[\d\s.+\-*/()]+", cleaned):
        return None                        # refers to other names; not needed here
    return eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307 - digits and operators only


def read_constants(path: Path) -> dict:
    return {name: v for name, expr in _CONST.findall(path.read_text()) if (v := _value(expr)) is not None}


C = read_constants(ROOT / "src" / "Tracker.cpp")
C.update({k: v for k, v in read_constants(ROOT / "src" / "FaceFilter.h").items() if k not in C})

UPDATE_US = 20_000                         # PanTilt motion task period (kUpdateMs)


class AxisFilter:
    """src/FaceFilter.h"""

    def __init__(self, accel_noise: float = C["kFaceAccelNoise"], meas_noise: float = C["kFaceMeasNoise"],
                 gate: float = C["kFaceGate"]):
        self.accel_noise = accel_noise
        self.meas_noise = meas_noise
        self.gate = gate
        self.ready = False
        self.angle = self.velocity = 0.0
        self.p00 = self.p01 = self.p11 = 0.0

    def reset(self) -> None:
        self.ready = False

    def init(self, angle: float) -> None:
        self.ready = True
        self.angle = angle
        self.velocity = 0.0
        self.p00 = self.meas_noise ** 2
        self.p01 = 0.0
        self.p11 = C["kInitialVelocitySigma"] ** 2

    def predict(self, dt: float) -> None:
        if not self.ready or dt <= 0:
            return
        self.angle += self.velocity * dt
        q = self.accel_noise ** 2
        dt2 = dt * dt
        self.p00 += dt * (2 * self.p01 + dt * self.p11) + q * dt2 * dt2 / 4
        self.p01 += dt * self.p11 + q * dt2 * dt / 2
        self.p11 += q * dt2

    def update(self, measured: float) -> bool:
        if not self.ready:
            self.init(measured)
            return True
        innovation = measured - self.angle
        s = self.p00 + self.meas_noise ** 2
        if innovation * innovation > self.gate * self.gate * s:
            return False
        k0, k1 = self.p00 / s, self.p01 / s
        self.angle += k0 * innovation
        self.velocity += k1 * innovation
        p00, p01 = self.p00, self.p01
        self.p00 -= k0 * p00
        self.p01 -= k0 * p01
        self.p11 -= k1 * p01
        return True

    def predicted(self, ahead: float) -> float:
        return self.angle + self.velocity * ahead


@dataclass
class Range:
    min: float
    max: float

    def clamp(self, v: float) -> float:
        return min(max(v, self.min), self.max)

    def center(self) -> float:
        return (self.min + self.max) / 2


class Mount:
    """src/PanTilt.cpp: every 20 ms each axis moves toward its target by at most max_speed * 20 ms."""

    def __init__(self, pan_range: Range, tilt_range: Range, tilt_level: float, pan: float, tilt: float):
        self.pan_range, self.tilt_range = pan_range, tilt_range
        self.tilt_level = tilt_range.clamp(tilt_level)
        self.pan = self.pan_target = pan_range.clamp(pan)
        self.tilt = self.tilt_target = tilt_range.clamp(tilt)
        self.max_speed = 60.0
        self.time_us = 0

    def set_target(self, pan: float, tilt: float) -> None:
        self.pan_target = self.pan_range.clamp(pan)
        self.tilt_target = self.tilt_range.clamp(tilt)

    def set_max_speed(self, deg_s: float) -> None:
        self.max_speed = deg_s

    def advance(self, to_us: int, on_tick=None) -> None:
        """Run the motion task's 20 ms ticks up to `to_us`; `on_tick(time_us)` after each."""
        next_tick = (self.time_us // UPDATE_US + 1) * UPDATE_US
        while next_tick <= to_us:
            step = self.max_speed * UPDATE_US / 1e6
            self.pan = _approach(self.pan, self.pan_target, step)
            self.tilt = _approach(self.tilt, self.tilt_target, step)
            self.time_us = next_tick
            if on_tick:
                on_tick(next_tick)
            next_tick += UPDATE_US
        self.time_us = max(self.time_us, to_us)


def _approach(pos: float, target: float, step: float) -> float:
    diff = target - pos
    if abs(diff) <= step:
        return target
    return pos + (step if diff > 0 else -step)


@dataclass
class Face:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class Options:
    """TrackerOptions; defaults from Tracker.cpp"""
    filter: bool = C["kFilterFace"]
    lead_s: float = C["kLeadS"]
    coast_s: float = C["kCoastS"]


@dataclass
class StepResult:
    state: str
    target_index: int
    coasting: bool
    face_pan: float | None = None          # filter estimate when ready
    face_tilt: float | None = None
    extra: dict = field(default_factory=dict)


class Tracker:
    """src/Tracker.cpp, minus the camera and detector: the caller passes each frame's faces."""

    def __init__(self, mount: Mount, options: Options | None = None):
        self.mount = mount
        self.options = options or Options()
        self.state = "SCAN"
        self.scan_dir = 1
        self.scan_row = 0
        self.scan_arrived_us = 0
        self.last_seen_us = 0
        self.last_face: Face | None = None
        self.pan_target = self.tilt_target = 0.0
        self.pan_filter = AxisFilter()
        self.tilt_filter = AxisFilter()
        self.filter_us = 0
        self.rejects = 0
        self.coasting = False
        self.started = False

    def step(self, faces: list[Face], frame_w: int, frame_h: int, capture_us: int, pan_at_capture: float,
             tilt_at_capture: float, now_us: int) -> StepResult:
        """One tracker frame. `pan_at_capture`/`tilt_at_capture`: mount angles when the frame was taken;
        `now_us`: when detection finished (the mount keeps moving in between)."""
        if not self.started:
            self.started = True
            self.enter_scan()
        target = self.pick_face(faces)
        if target >= 0:
            face = faces[target]
            if self.state == "SCAN":
                self.enter_track()
            self.last_seen_us = now_us
            self.last_face = face
            self.coasting = False
            self.steer_toward(face, frame_w, frame_h, pan_at_capture, tilt_at_capture, capture_us, now_us)
        elif self.state == "TRACK" and now_us - self.last_seen_us > C["kLostTimeoutUs"]:
            self.enter_scan()
        elif self.state == "TRACK":
            self.coast(pan_at_capture, tilt_at_capture, now_us)
        if self.state == "SCAN":
            self.scan_step(now_us)
        ready = self.pan_filter.ready
        return StepResult(self.state, target, self.coasting,
                          self.pan_filter.angle if ready else None, self.tilt_filter.angle if ready else None)

    def enter_scan(self) -> None:
        self.state = "SCAN"
        self.last_face = None
        self.pan_filter.reset()
        self.tilt_filter.reset()
        self.coasting = False
        self.mount.set_max_speed(C["kScanMoveSpeedDegS"])
        self.scan_dir = 1 if self.mount.pan < self.mount.pan_range.center() else -1
        self.pan_target = self.mount.pan
        self.tilt_target = self.scan_row_tilt()
        self.scan_arrived_us = 0
        self.mount.set_target(self.pan_target, self.tilt_target)

    def enter_track(self) -> None:
        self.state = "TRACK"
        self.mount.set_max_speed(C["kTrackSpeedDegS"])

    def scan_step(self, now_us: int) -> None:
        arrived = abs(self.mount.pan - self.pan_target) < 0.5 and abs(self.mount.tilt - self.tilt_target) < 0.5
        if not arrived:
            self.scan_arrived_us = 0
            return
        if self.scan_arrived_us == 0:
            self.scan_arrived_us = now_us
        if now_us - self.scan_arrived_us < C["kScanDwellUs"]:
            return
        pan = self.mount.pan_range
        end = pan.max if self.scan_dir > 0 else pan.min
        if abs(self.pan_target - end) < 0.5:
            self.scan_dir = -self.scan_dir
            self.scan_row = (self.scan_row + 1) % len(C["kScanRowOffsets"])
        else:
            self.pan_target = pan.clamp(self.pan_target + self.scan_dir * C["kScanStepDeg"])
        self.tilt_target = self.scan_row_tilt()
        self.scan_arrived_us = 0
        self.mount.set_target(self.pan_target, self.tilt_target)

    def scan_row_tilt(self) -> float:
        return self.mount.tilt_range.clamp(self.mount.tilt_level + C["kScanRowOffsets"][self.scan_row])

    def steer_toward(self, face: Face, frame_w: int, frame_h: int, pan_at_capture: float, tilt_at_capture: float,
                     capture_us: int, now_us: int) -> None:
        raw_x = (face.cx - frame_w / 2) / (frame_w / 2)
        raw_y = (face.cy - frame_h / 2) / (frame_h / 2)
        err_x = 0.0 if abs(raw_x) < C["kDeadband"] else raw_x
        err_y = 0.0 if abs(raw_y) < C["kDeadband"] else raw_y
        if self.options.filter:
            face_pan = pan_at_capture + C["kPanDir"] * raw_x * (C["kHFovDeg"] / 2)
            face_tilt = tilt_at_capture + C["kTiltDir"] * raw_y * (C["kVFovDeg"] / 2)
            self.filter_face(face_pan, face_tilt, capture_us)
            ahead = (now_us - capture_us) / 1e6 + self.options.lead_s
            pan = self.aim(self.pan_filter, pan_at_capture, ahead, C["kHFovDeg"] / 2)
            tilt = self.aim(self.tilt_filter, tilt_at_capture, ahead, C["kVFovDeg"] / 2)
        else:
            pan = pan_at_capture + C["kPanDir"] * C["kGain"] * err_x * (C["kHFovDeg"] / 2)
            tilt = tilt_at_capture + C["kTiltDir"] * C["kGain"] * err_y * (C["kVFovDeg"] / 2)
        self.set_mount_target(pan, tilt)

    def set_mount_target(self, pan: float, tilt: float) -> None:
        self.pan_target = self.mount.pan_range.clamp(pan)
        self.tilt_target = self.mount.tilt_range.clamp(tilt)
        self.mount.set_target(self.pan_target, self.tilt_target)

    @staticmethod
    def aim(f: AxisFilter, at_capture: float, ahead: float, half_fov: float) -> float:
        if abs(f.angle - at_capture) < C["kDeadband"] * half_fov:
            return at_capture
        speed = C["kTrackSpeedDegS"]
        return f.angle + min(max(f.velocity, -speed), speed) * ahead

    def coast(self, pan_at_capture: float, tilt_at_capture: float, now_us: int) -> None:
        self.coasting = (self.options.filter and self.pan_filter.ready
                         and now_us - self.last_seen_us <= int(self.options.coast_s * 1e6))
        if not self.coasting:
            return
        ahead = (now_us - self.filter_us) / 1e6 + self.options.lead_s
        self.set_mount_target(self.aim(self.pan_filter, pan_at_capture, ahead, C["kHFovDeg"] / 2),
                              self.aim(self.tilt_filter, tilt_at_capture, ahead, C["kVFovDeg"] / 2))

    def filter_face(self, face_pan: float, face_tilt: float, capture_us: int) -> None:
        if not self.pan_filter.ready or capture_us - self.filter_us > C["kFilterStaleUs"]:
            self.pan_filter.init(face_pan)
            self.tilt_filter.init(face_tilt)
            self.filter_us = capture_us
            self.rejects = 0
            return
        dt = (capture_us - self.filter_us) / 1e6
        self.filter_us = capture_us
        self.pan_filter.predict(dt)
        self.tilt_filter.predict(dt)
        pan_ok = self.pan_filter.update(face_pan)
        tilt_ok = self.tilt_filter.update(face_tilt)
        if pan_ok and tilt_ok:
            self.rejects = 0
        else:
            self.rejects += 1
            if self.rejects >= C["kMaxRejects"]:
                self.pan_filter.init(face_pan)
                self.tilt_filter.init(face_tilt)
                self.rejects = 0

    def pick_face(self, faces: list[Face]) -> int:
        if not faces:
            return -1
        best = 0
        for i in range(1, len(faces)):
            f = faces[i]
            if self.last_face:
                def dist(a: Face) -> float:
                    return (a.cx - self.last_face.cx) ** 2 + (a.cy - self.last_face.cy) ** 2
                if dist(f) < dist(faces[best]):
                    best = i
            elif f.area > faces[best].area:
                best = i
        return best


def angle_to_offset(angle_deg: float, mount_deg: float, axis: str) -> float:
    """Inverse of the tracker's pixel-to-angle mapping: a mount-angle direction -> offset from the
    frame centre as a fraction of half the frame (-1..1), seen from a mount pointing at `mount_deg`."""
    if axis == "pan":
        return (angle_deg - mount_deg) / (C["kPanDir"] * C["kHFovDeg"] / 2)
    return (angle_deg - mount_deg) / (C["kTiltDir"] * C["kVFovDeg"] / 2)


def describe() -> str:
    keys = ["kFilterFace", "kLeadS", "kCoastS", "kFaceAccelNoise", "kFaceMeasNoise", "kFaceGate", "kMaxRejects",
            "kGain", "kDeadband", "kTrackSpeedDegS", "kLostTimeoutUs", "kHFovDeg", "kVFovDeg"]
    return ", ".join(f"{k}={C[k]}" for k in keys if k in C)


if __name__ == "__main__":
    print(describe())
    assert math.isfinite(C["kHFovDeg"])

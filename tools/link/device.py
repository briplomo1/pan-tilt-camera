"""Talk to the pan/tilt firmware over USB.

    with DeviceLink() as device:
        print(device.hello())
        result = device.infer(rgb565_bytes, 320, 240)

Requests are stop-and-wait. Preview, Telemetry and Log packets arrive at any time and go to
callbacks registered with on(); they run on the reader thread, so keep them short.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Callable, Iterable

import serial

from tools.link import protocol as p
from tools.link.config import default_port

STREAM_TYPES = {p.Msg.PREVIEW, p.Msg.TELEMETRY, p.Msg.LOG}


class DeviceError(RuntimeError):
    def __init__(self, request: int, code: int, message: str):
        try:
            name = p.ErrorCode(code).name
        except ValueError:
            name = str(code)
        super().__init__(f"{p.Msg(request).name if request in p.Msg._value2member_map_ else request} failed: "
                         f"{name}: {message}")
        self.request = request
        self.code = code


def _print_log(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


class DeviceLink:
    def __init__(self, port: str | None = None, *, on_log: Callable[[str], None] | None = _print_log,
                 timeout: float = 5.0):
        self.port = port or default_port()
        self.timeout = timeout
        self._serial = serial.Serial()
        self._serial.port = self.port
        self._serial.baudrate = 115200          # ignored by USB-Serial-JTAG, required by pyserial
        self._serial.timeout = 0.05
        # Opening with DTR/RTS asserted can reset the ESP32-S3; keep both released
        self._serial.dtr = False
        self._serial.rts = False
        self._serial.open()

        self._parser = p.StreamParser()
        self._replies: queue.Queue[p.Packet] = queue.Queue()
        self._handlers: dict[int, list[Callable[[p.Packet], None]]] = {t: [] for t in STREAM_TYPES}
        self._request_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._seq = 0
        self._stop = threading.Event()
        if on_log:
            self.on(p.Msg.LOG, lambda packet: on_log(packet.payload.decode("utf-8", "replace")))
        self.on_text: Callable[[str], None] | None = on_log
        self._reader = threading.Thread(target=self._read_loop, name="device-reader", daemon=True)
        self._reader.start()

    # --- plumbing ------------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._reader.join(timeout=1.0)
        self._serial.close()

    def __enter__(self) -> "DeviceLink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def corrupt_frames(self) -> int:
        return self._parser.corrupt

    def on(self, msg: p.Msg, callback: Callable[[p.Packet], None]) -> None:
        """Call `callback` for every Preview / Telemetry / Log packet."""
        if msg not in STREAM_TYPES:
            raise ValueError(f"{msg.name} is a reply, not a stream")
        self._handlers[msg].append(callback)

    def _read_loop(self) -> None:
        last_data = time.monotonic()
        while not self._stop.is_set():
            try:
                data = self._serial.read(self._serial.in_waiting or 1)
            except serial.SerialException as e:
                if not self._stop.is_set():
                    _print_log(f"[link] serial error: {e}")
                return
            now = time.monotonic()
            if data:
                last_data = now
                events = self._parser.feed(data)
            else:
                events = []
                if now - last_data > 0.2:
                    text = self._parser.flush_text()
                    if text:
                        events.append(("text", text))
            for kind, item in events:
                if kind == "text":
                    if self.on_text:
                        self.on_text(item)
                elif item.type in self._handlers:
                    for handler in self._handlers[item.type]:
                        try:
                            handler(item)
                        except Exception as e:  # a bad callback must not kill the reader
                            _print_log(f"[link] handler error: {e!r}")
                else:
                    self._replies.put(item)

    def send(self, msg: p.Msg, payload: bytes = b"") -> None:
        with self._write_lock:
            self._seq = (self._seq + 1) & 0xFFFF
            self._serial.write(p.build_packet(msg, payload, self._seq))

    def _wait(self, expect: Iterable[p.Msg], request: p.Msg, timeout: float | None) -> p.Packet:
        expect = set(expect)
        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"No reply to {request.name} within {timeout or self.timeout:.1f} s")
            try:
                packet = self._replies.get(timeout=remaining)
            except queue.Empty:
                continue
            if packet.type == p.Msg.ERROR:
                raise DeviceError(*p.parse_error(packet.payload))
            if packet.type in expect:
                return packet
            # Anything else is a late reply to an earlier, timed-out request

    def request(self, msg: p.Msg, payload: bytes = b"", expect: Iterable[p.Msg] = (p.Msg.ACK,),
                timeout: float | None = None) -> p.Packet:
        with self._request_lock:
            while not self._replies.empty():
                self._replies.get_nowait()
            self.send(msg, payload)
            return self._wait(expect, msg, timeout)

    # --- commands ------------------------------------------------------------

    def hello(self) -> p.Hello:
        return p.Hello.parse(self.request(p.Msg.HELLO, expect=[p.Msg.HELLO_ACK]).payload)

    def list_models(self) -> list[tuple[str, str]]:
        return p.parse_model_list(self.request(p.Msg.LIST_MODELS, expect=[p.Msg.MODEL_LIST]).payload)

    def load_model(self, model_id: str) -> p.ModelInfo:
        packet = self.request(p.Msg.LOAD_MODEL, p.LOAD_MODEL.pack(p.fixed(model_id, 32)),
                              expect=[p.Msg.MODEL_INFO], timeout=60.0)
        return p.ModelInfo.parse(packet.payload)

    def set_threshold(self, threshold: float) -> None:
        self.request(p.Msg.SET_THRESHOLD, p.SET_THRESHOLD.pack(threshold))

    def set_mode(self, mode: p.Mode) -> None:
        self.request(p.Msg.SET_MODE, p.SET_MODE.pack(int(mode)))

    def subscribe(self, telemetry: bool) -> None:
        self.request(p.Msg.SUBSCRIBE, p.SUBSCRIBE.pack(int(telemetry)))

    def sync(self, marker: int) -> int:
        """Device clock (µs) when it received the marker."""
        packet = self.request(p.Msg.SYNC, p.SYNC.pack(marker), expect=[p.Msg.SYNC_ACK])
        _marker, time_us = p.SYNC_ACK.unpack_from(packet.payload)
        return time_us

    def set_preview(self, enabled: bool, fmt: p.ImageFormat = p.ImageFormat.JPEG, scale_shift: int = 1,
                    every_n: int = 1, quality: int = 60) -> None:
        packet = self.request(p.Msg.SET_PREVIEW,
                              p.SET_PREVIEW.pack(int(enabled), int(fmt), scale_shift, every_n, quality))
        if not p.ACK.unpack_from(packet.payload)[1]:
            raise DeviceError(p.Msg.SET_PREVIEW, p.ErrorCode.BAD_REQUEST, "preview settings rejected")

    def set_tracker(self, filter_on: bool, lead_s: float = 0.05, coast_s: float = 0.3) -> None:
        """Tracker controller for A/B runs: Kalman filter + lead on or off (defaults: src/Tracker.cpp)."""
        self.request(p.Msg.SET_TRACKER, p.SET_TRACKER.pack(int(filter_on), lead_s, coast_s))

    def infer(self, pixels: bytes, width: int, height: int, frame_id: int = 0,
              fmt: p.ImageFormat = p.ImageFormat.RGB565BE, timeout: float = 10.0) -> p.InferResult:
        payload = p.IMAGE_HEADER.pack(frame_id, width, height, int(fmt)) + pixels
        packet = self.request(p.Msg.INFER, payload, expect=[p.Msg.INFER_RESULT], timeout=timeout)
        return p.InferResult.parse(packet.payload)

    def capture(self, count: int = 1, fmt: p.ImageFormat = p.ImageFormat.RGB565BE,
                quality: int = 90) -> list[p.CapturedFrame]:
        with self._request_lock:
            while not self._replies.empty():
                self._replies.get_nowait()
            self.send(p.Msg.CAPTURE, p.CAPTURE.pack(count, int(fmt), quality))
            return [p.CapturedFrame.parse(self._wait([p.Msg.FRAME], p.Msg.CAPTURE, 10.0).payload)
                    for _ in range(count)]

    def echo(self, payload: bytes, timeout: float = 10.0) -> bytes:
        return self.request(p.Msg.ECHO, payload, expect=[p.Msg.ECHO_REPLY], timeout=timeout).payload

    def list_store(self) -> list[dict]:
        return p.parse_store_list(self.request(p.Msg.LIST_STORE, expect=[p.Msg.STORE_LIST]).payload)

    def load_blob(self, name: str, placement: p.Placement) -> p.BlobInfo:
        packet = self.request(p.Msg.LOAD_BLOB, p.LOAD_BLOB.pack(p.fixed(name, 32), int(placement)),
                              expect=[p.Msg.BLOB_INFO], timeout=30.0)
        return p.BlobInfo.parse(packet.payload)

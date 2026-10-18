"""USB protocol shared with the firmware.

Mirrors src/link/Protocol.h and src/link/Cobs.cpp; keep them in sync and bump
PROTOCOL_VERSION on any change.

Packet on the wire:  0x00 | COBS( Header | payload | CRC32(Header + payload) ) | 0x00
All fields are little-endian.
"""
from __future__ import annotations

import enum
import struct
import zlib
from dataclasses import dataclass, field

PROTOCOL_VERSION = 1
MAX_PAYLOAD = 160 * 1024


class Msg(enum.IntEnum):
    # Host -> device
    HELLO = 0x01
    LIST_MODELS = 0x02
    LOAD_MODEL = 0x03
    SET_THRESHOLD = 0x04
    INFER = 0x05
    CAPTURE = 0x06
    SET_MODE = 0x07
    SUBSCRIBE = 0x08
    SYNC = 0x09
    ECHO = 0x0A
    SET_PREVIEW = 0x0B
    LIST_STORE = 0x0C
    LOAD_BLOB = 0x0D
    # Device -> host
    HELLO_ACK = 0x81
    MODEL_LIST = 0x82
    MODEL_INFO = 0x83
    ACK = 0x84
    INFER_RESULT = 0x85
    FRAME = 0x86
    PREVIEW = 0x87
    TELEMETRY = 0x88
    SYNC_ACK = 0x89
    ECHO_REPLY = 0x8A
    STORE_LIST = 0x8C
    BLOB_INFO = 0x8D
    LOG = 0xF0
    ERROR = 0xFF


class ImageFormat(enum.IntEnum):
    RGB565BE = 0
    RGB888 = 1
    GRAY8 = 2
    JPEG = 3


class Mode(enum.IntEnum):
    TRACK = 0
    BENCH = 1


class Placement(enum.IntEnum):
    FLASH_XIP = 0
    PSRAM = 1
    INTERNAL = 2


class ErrorCode(enum.IntEnum):
    BAD_REQUEST = 1
    UNKNOWN_MODEL = 2
    NO_MODEL = 3
    NO_MEMORY = 4
    CAMERA_FAILED = 5
    UNSUPPORTED = 6


TRACKER_STATES = {0: "SCAN", 1: "TRACK", 2: "PAUSED"}

# Struct layouts (sizes are checked against the firmware's static_asserts in the tests)
HEADER = struct.Struct("<BBHI")                      # 8
CRC = struct.Struct("<I")
WIRE_FACE = struct.Struct("<hhhhf")                  # 12
WIRE_TIMING = struct.Struct("<IIII")                 # 16
IMAGE_HEADER = struct.Struct("<IHHB3x")              # 12
HELLO_ACK = struct.Struct("<BB2xIIII32s32s")         # 84
COUNT_HEADER = struct.Struct("<B3x")                 # ModelListHeader / StoreListHeader
MODEL_LIST_ENTRY = struct.Struct("<32s16s")          # 48
LOAD_MODEL = struct.Struct("<32s")
MODEL_INFO = struct.Struct("<B3x32s16s32sHHIIiiII")  # 112
SET_THRESHOLD = struct.Struct("<f")
ACK = struct.Struct("<BB2x")
INFER_RESULT = struct.Struct("<IIIIIIIB3x")          # 32
CAPTURE = struct.Struct("<HBB")                      # 4
FRAME_MSG = struct.Struct("<qffIHHB3x")              # 28
SET_MODE = struct.Struct("<B3x")
SUBSCRIBE = struct.Struct("<B3x")
TELEMETRY = struct.Struct("<qIBBBxffffhhhhfIIII")    # 60
SYNC = struct.Struct("<I")
SYNC_ACK = struct.Struct("<I4xq")                    # 16
SET_PREVIEW = struct.Struct("<BBBBB3x")              # 8
PREVIEW = struct.Struct("<qIBBBBffIIIIIHH32sIHHB3xI")  # 96
STORE_LIST_ENTRY = struct.Struct("<32s8sII")         # 48
LOAD_BLOB = struct.Struct("<32sB3x")                 # 36
BLOB_INFO = struct.Struct("<BB2x32sIIIiiI")          # 60
ERROR = struct.Struct("<BB2x")


# --- Framing -----------------------------------------------------------------

def cobs_encode(data: bytes) -> bytes:
    """COBS-encode without delimiters (same output as CobsEncoder in the firmware)."""
    out = bytearray()
    block = bytearray()
    last_full = False
    for byte in data:
        if byte == 0:
            out.append(len(block) + 1)
            out += block
            block.clear()
            last_full = False
        else:
            block.append(byte)
            last_full = False
            if len(block) == 254:
                out.append(0xFF)
                out += block
                block.clear()
                last_full = True
    if block or not last_full:
        out.append(len(block) + 1)
        out += block
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    """Decode one COBS frame (without delimiters). Raises ValueError if it is malformed."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        code = data[i]
        if code == 0:
            raise ValueError("zero byte inside a COBS frame")
        end = i + code
        if end > n:
            raise ValueError("truncated COBS block")
        block = data[i + 1:end]
        if 0 in block:
            raise ValueError("zero byte inside a COBS block")
        out += block
        i = end
        if code != 0xFF and i < n:
            out.append(0)
    return bytes(out)


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass
class Packet:
    type: int
    seq: int
    payload: bytes

    @property
    def msg(self) -> Msg | int:
        try:
            return Msg(self.type)
        except ValueError:
            return self.type


def build_packet(msg: int, payload: bytes = b"", seq: int = 0) -> bytes:
    """Bytes to write to the port for one packet, delimiters included."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload of {len(payload)} bytes exceeds {MAX_PAYLOAD}")
    body = HEADER.pack(PROTOCOL_VERSION, int(msg), seq & 0xFFFF, len(payload)) + payload
    return b"\x00" + cobs_encode(body + CRC.pack(crc32(body))) + b"\x00"


def parse_packet(decoded: bytes) -> Packet:
    """Check and split a decoded frame. Raises ValueError if it isn't a valid packet."""
    if len(decoded) < HEADER.size + CRC.size:
        raise ValueError("frame too short")
    version, msg, seq, length = HEADER.unpack_from(decoded)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"protocol version {version}, expected {PROTOCOL_VERSION}")
    if length != len(decoded) - HEADER.size - CRC.size:
        raise ValueError("length mismatch")
    (expected,) = CRC.unpack_from(decoded, len(decoded) - CRC.size)
    if crc32(decoded[:-CRC.size]) != expected:
        raise ValueError("CRC mismatch")
    return Packet(msg, seq, decoded[HEADER.size:HEADER.size + length])


def _printable(data: bytes) -> bool:
    return all(32 <= b < 127 or b in (9, 10, 13, 27) for b in data)


class StreamParser:
    """Splits the device's byte stream into packets and plain text.

    Text appears before the link starts (boot logs), on a crash, or from stray printf calls.
    Every packet starts and ends with 0x00, so text never merges into a packet.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.corrupt = 0

    def feed(self, data: bytes) -> list[tuple[str, Packet | str]]:
        events: list[tuple[str, Packet | str]] = []
        self._buffer += data
        while True:
            end = self._buffer.find(0)
            if end < 0:
                break
            chunk = bytes(self._buffer[:end])
            del self._buffer[:end + 1]
            if not chunk:
                continue
            try:
                events.append(("packet", parse_packet(cobs_decode(chunk))))
            except ValueError:
                if _printable(chunk):
                    events.append(("text", chunk.decode("ascii", "replace")))
                else:
                    self.corrupt += 1
        return events

    def flush_text(self) -> str | None:
        """Hand over pending bytes as text if they look like text (call when the line is idle)."""
        if self._buffer and _printable(self._buffer):
            text = self._buffer.decode("ascii", "replace")
            self._buffer.clear()
            return text
        return None


# --- Message types -----------------------------------------------------------

def cstr(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def fixed(text: str, size: int) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) >= size:
        raise ValueError(f"'{text}' is longer than {size - 1} bytes")
    return raw


@dataclass
class Face:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float

    @property
    def box(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class Timing:
    preprocess_us: int
    inference_us: int
    postprocess_us: int
    total_us: int

    def as_dict(self) -> dict:
        return {"pre": self.preprocess_us, "infer": self.inference_us,
                "post": self.postprocess_us, "total": self.total_us}


def parse_faces(payload: bytes, offset: int, count: int) -> list[Face]:
    return [Face(*WIRE_FACE.unpack_from(payload, offset + i * WIRE_FACE.size)) for i in range(count)]


@dataclass
class Hello:
    protocol: int
    mode: Mode
    free_internal: int
    free_psram: int
    total_internal: int
    total_psram: int
    firmware: str
    active_model: str

    @classmethod
    def parse(cls, payload: bytes) -> "Hello":
        p, mode, fi, fp, ti, tp, fw, model = HELLO_ACK.unpack_from(payload)
        return cls(p, Mode(mode), fi, fp, ti, tp, cstr(fw), cstr(model))


@dataclass
class ModelInfo:
    ok: bool
    id: str
    backend: str
    quantization: str
    input_width: int
    input_height: int
    model_bytes: int
    load_us: int
    internal_bytes: int
    psram_bytes: int
    min_free_internal: int
    min_free_psram: int

    @classmethod
    def parse(cls, payload: bytes) -> "ModelInfo":
        ok, mid, backend, quant, w, h, size, load, ib, pb, mfi, mfp = MODEL_INFO.unpack_from(payload)
        return cls(bool(ok), cstr(mid), cstr(backend), cstr(quant), w, h, size, load, ib, pb, mfi, mfp)


@dataclass
class InferResult:
    frame_id: int
    timing: Timing
    min_free_internal: int
    min_free_psram: int
    faces: list[Face]

    @classmethod
    def parse(cls, payload: bytes) -> "InferResult":
        fid, pre, inf, post, total, mfi, mfp, count = INFER_RESULT.unpack_from(payload)
        return cls(fid, Timing(pre, inf, post, total), mfi, mfp, parse_faces(payload, INFER_RESULT.size, count))


@dataclass
class Preview:
    time_us: int
    frame_id: int
    tracker_state: int
    target_index: int | None
    mode: Mode
    pan: float
    tilt: float
    timing: Timing
    free_internal: int
    source_width: int
    source_height: int
    model_id: str
    width: int
    height: int
    format: ImageFormat
    faces: list[Face]
    image: bytes

    @classmethod
    def parse(cls, payload: bytes) -> "Preview":
        (time_us, frame_id, state, count, target, mode, pan, tilt, pre, inf, post, total, free_internal,
         src_w, src_h, model, _image_id, w, h, fmt, image_bytes) = PREVIEW.unpack_from(payload)
        faces_end = PREVIEW.size + count * WIRE_FACE.size
        image = payload[faces_end:faces_end + image_bytes]
        return cls(time_us, frame_id, state, None if target == 0xFF else target, Mode(mode), pan, tilt,
                   Timing(pre, inf, post, total), free_internal, src_w, src_h, cstr(model), w, h,
                   ImageFormat(fmt), parse_faces(payload, PREVIEW.size, count), image)


@dataclass
class Telemetry:
    time_us: int
    frame_id: int
    state: int
    face_count: int
    target: Face | None
    pan: float
    tilt: float
    pan_target: float
    tilt_target: float
    timing: Timing

    @classmethod
    def parse(cls, payload: bytes) -> "Telemetry":
        (time_us, frame_id, state, count, has_target, pan, tilt, pan_t, tilt_t,
         x0, y0, x1, y1, score, pre, inf, post, total) = TELEMETRY.unpack_from(payload)
        target = Face(x0, y0, x1, y1, score) if has_target else None
        return cls(time_us, frame_id, state, count, target, pan, tilt, pan_t, tilt_t,
                   Timing(pre, inf, post, total))


@dataclass
class CapturedFrame:
    time_us: int
    pan: float
    tilt: float
    frame_id: int
    width: int
    height: int
    format: ImageFormat
    data: bytes = field(repr=False)

    @classmethod
    def parse(cls, payload: bytes) -> "CapturedFrame":
        time_us, pan, tilt, fid, w, h, fmt = FRAME_MSG.unpack_from(payload)
        return cls(time_us, pan, tilt, fid, w, h, ImageFormat(fmt), payload[FRAME_MSG.size:])


@dataclass
class BlobInfo:
    ok: bool
    placement: Placement
    name: str
    size: int
    crc32: int
    load_us: int
    internal_bytes: int
    psram_bytes: int
    address: int

    @classmethod
    def parse(cls, payload: bytes) -> "BlobInfo":
        ok, placement, name, size, crc, load, ib, pb, addr = BLOB_INFO.unpack_from(payload)
        return cls(bool(ok), Placement(placement), cstr(name), size, crc, load, ib, pb, addr)


def parse_model_list(payload: bytes) -> list[tuple[str, str]]:
    (count,) = COUNT_HEADER.unpack_from(payload)
    entries = []
    for i in range(count):
        mid, backend = MODEL_LIST_ENTRY.unpack_from(payload, COUNT_HEADER.size + i * MODEL_LIST_ENTRY.size)
        entries.append((cstr(mid), cstr(backend)))
    return entries


def parse_store_list(payload: bytes) -> list[dict]:
    (count,) = COUNT_HEADER.unpack_from(payload)
    files = []
    for i in range(count):
        name, fmt, size, crc = STORE_LIST_ENTRY.unpack_from(payload, COUNT_HEADER.size + i * STORE_LIST_ENTRY.size)
        files.append({"name": cstr(name), "format": cstr(fmt), "size": size, "crc32": crc})
    return files


def parse_error(payload: bytes) -> tuple[int, int, str]:
    for_type, code, = ERROR.unpack_from(payload)
    return for_type, code, payload[ERROR.size:].decode("utf-8", "replace")

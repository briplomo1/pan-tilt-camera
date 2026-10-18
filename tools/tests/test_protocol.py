"""Host-side checks of the link protocol and the model pack format.

    python -m unittest discover -s tools/tests -t .      (from the project root)
    python -m pytest tools/tests                         (if pytest is installed)

Struct sizes here must match the static_asserts in src/link/Protocol.h and
src/models/ModelPack.h; the CRC vector matches the firmware's boot self-test in Link.cpp.
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.link import protocol as p  # noqa: E402
from tools.models import pack_models  # noqa: E402

R1_254 = bytes(range(1, 255))   # 01..FE

# COBS reference vectors (Cheshire & Baker; the same set Wikipedia lists), encoded form
# shown with its trailing delimiter
COBS_VECTORS = [
    (b"", b"\x01\x00"),
    (b"\x00", b"\x01\x01\x00"),
    (b"\x00\x00", b"\x01\x01\x01\x00"),
    (b"\x00\x11\x00", b"\x01\x02\x11\x01\x00"),
    (b"\x11\x22\x00\x33", b"\x03\x11\x22\x02\x33\x00"),
    (b"\x11\x22\x33\x44", b"\x05\x11\x22\x33\x44\x00"),
    (b"\x11\x00\x00\x00", b"\x02\x11\x01\x01\x01\x00"),
    (R1_254, b"\xff" + R1_254 + b"\x00"),
    (b"\x00" + R1_254, b"\x01\xff" + R1_254 + b"\x00"),
    (bytes(range(1, 256)), b"\xff" + R1_254 + b"\x02\xff\x00"),
    (bytes(range(2, 256)) + b"\x00", b"\xff" + bytes(range(2, 256)) + b"\x01\x01\x00"),
    (bytes(range(3, 256)) + b"\x00\x01", b"\xfe" + bytes(range(3, 256)) + b"\x02\x01\x00"),
]


class CobsTest(unittest.TestCase):
    def test_reference_vectors(self):
        for raw, encoded in COBS_VECTORS:
            with self.subTest(raw=raw[:8]):
                self.assertEqual(p.cobs_encode(raw) + b"\x00", encoded)
                self.assertEqual(p.cobs_decode(encoded[:-1]), raw)

    def test_round_trip_random(self):
        rng = random.Random(1234)
        for size in [1, 2, 253, 254, 255, 256, 508, 509, 1000, 70_000]:
            for zero_rate in (0.0, 0.01, 0.5):
                data = bytes(0 if rng.random() < zero_rate else rng.randrange(1, 256) for _ in range(size))
                encoded = p.cobs_encode(data)
                self.assertNotIn(0, encoded)
                self.assertEqual(p.cobs_decode(encoded), data)

    def test_rejects_malformed(self):
        with self.assertRaises(ValueError):
            p.cobs_decode(b"\x05\x11\x22")        # block runs past the end
        with self.assertRaises(ValueError):
            p.cobs_decode(b"\x03\x11\x00")        # zero inside a block


class PacketTest(unittest.TestCase):
    def test_crc_matches_firmware_self_test(self):
        self.assertEqual(p.crc32(b"123456789"), 0xCBF43926)

    def test_round_trip(self):
        payload = os.urandom(5000) + b"\x00" * 300
        wire = p.build_packet(p.Msg.ECHO, payload, seq=7)
        self.assertEqual(wire[0], 0)
        self.assertEqual(wire[-1], 0)
        packet = p.parse_packet(p.cobs_decode(wire[1:-1]))
        self.assertEqual((packet.type, packet.seq, packet.payload), (p.Msg.ECHO, 7, payload))

    def test_corruption_is_detected(self):
        decoded = bytearray(p.cobs_decode(p.build_packet(p.Msg.HELLO, b"abc")[1:-1]))
        decoded[9] ^= 0x01
        with self.assertRaises(ValueError):
            p.parse_packet(bytes(decoded))

    def test_stream_parser_separates_text_and_packets(self):
        parser = p.StreamParser()
        stream = (b"I (123) boot: hello\r\n" + p.build_packet(p.Msg.LOG, b"from link")
                  + b"\x00\x13\x37garbage\x99\x00" + p.build_packet(p.Msg.ACK, p.ACK.pack(p.Msg.HELLO, 1)))
        events = []
        for i in range(0, len(stream), 7):           # arrives in small pieces
            events += parser.feed(stream[i:i + 7])
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds, ["text", "packet", "packet"])
        self.assertIn("boot: hello", events[0][1])
        self.assertEqual(events[1][1].payload, b"from link")
        self.assertEqual(events[2][1].type, p.Msg.ACK)
        self.assertEqual(parser.corrupt, 1)


class StructSizeTest(unittest.TestCase):
    """Sizes from the static_asserts in src/link/Protocol.h."""

    def test_sizes(self):
        expected = {
            "HEADER": 8, "WIRE_FACE": 12, "WIRE_TIMING": 16, "IMAGE_HEADER": 12, "HELLO_ACK": 84,
            "MODEL_LIST_ENTRY": 48, "MODEL_INFO": 112, "INFER_RESULT": 32, "CAPTURE": 4, "FRAME_MSG": 28,
            "TELEMETRY": 60, "SYNC_ACK": 16, "SET_PREVIEW": 8, "PREVIEW": 96, "STORE_LIST_ENTRY": 48,
            "LOAD_BLOB": 36, "BLOB_INFO": 60, "ERROR": 4, "ACK": 4, "COUNT_HEADER": 4, "SET_MODE": 4,
        }
        for name, size in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(p, name).size, size)

    def test_preview_parse(self):
        faces = [p.Face(10, 20, 60, 80, 0.9), p.Face(100, 50, 140, 110, 0.6)]
        image = b"\xff\xd8jpeg-bytes"
        payload = p.PREVIEW.pack(123456, 42, 1, len(faces), 0, 0, 10.5, 90.0, 1, 2, 3, 4000, 200_000,
                                 320, 240, b"espdl-msrmnp-s8", 42, 160, 120, p.ImageFormat.JPEG, len(image))
        payload += b"".join(p.WIRE_FACE.pack(*f.box, f.score) for f in faces) + image
        preview = p.Preview.parse(payload)
        self.assertEqual(preview.model_id, "espdl-msrmnp-s8")
        self.assertEqual(preview.target_index, 0)
        self.assertEqual([f.box for f in preview.faces], [f.box for f in faces])
        self.assertEqual(preview.image, image)
        self.assertEqual((preview.width, preview.height, preview.format), (160, 120, p.ImageFormat.JPEG))


class ModelPackTest(unittest.TestCase):
    """Layout from src/models/ModelPack.h."""

    def test_header_sizes(self):
        self.assertEqual(pack_models.HEADER.size, 16)
        self.assertEqual(pack_models.ENTRY.size, 56)

    def test_round_trip_and_alignment(self):
        files = [pack_models.PackedFile("yunet-q8", "tflite", os.urandom(1001)),
                 pack_models.PackedFile("blazeface", "tflite", os.urandom(16)),
                 pack_models.PackedFile("dummy", "bin", os.urandom(77_777))]
        image = pack_models.build_pack(files)
        entries = pack_models.read_pack(image)
        self.assertEqual([e["name"] for e in entries], ["yunet-q8", "blazeface", "dummy"])
        for entry, f in zip(entries, files):
            self.assertTrue(entry["crc_ok"])
            self.assertEqual(entry["offset"] % pack_models.ALIGNMENT, 0)
            self.assertEqual(image[entry["offset"]:entry["offset"] + entry["size"]], f.data)


if __name__ == "__main__":
    unittest.main()

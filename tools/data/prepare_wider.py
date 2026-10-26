"""Build dataset A from WIDER FACE: A-test (scoring), A-cal (PTQ calibration), A-train list.

    python tools/data/prepare_wider.py --wider D:/datasets/WIDER --out data/a

Download WIDER FACE yourself (http://shuoyang1213.me/WIDERFACE/; images WIDER_train.zip,
WIDER_val.zip and wider_face_split.zip) and unpack them under one folder, so it contains
WIDER_train/images, WIDER_val/images and wider_face_split/*.txt. Then this writes:

    <out>/test/<event>/<image>.jpg + <out>/test/gt.json    A-test: val images, ~400
    <out>/cal/<event>/<image>.jpg  + <out>/cal/gt.json     A-cal: train images, <= 500, for calibration
    <out>/train_gt.json                                    A-train: every usable train image, by path
                                                           relative to WIDER_train/images (not copied)
    <out>/README.json                                      the filter, counts and seed

Filter (EXPERIMENT_PLAN §6), applied to A-test and A-cal: 1-3 faces, every face at least
--min-height px tall once the image is letterboxed to 320x240, and no face marked invalid or
heavily occluded (those images are dropped, not just the face, so the ground truth stays complete
and an unlabelled face is never counted as a false positive). Selection is random with --seed.
No-face images for the test set come from tools/data/negatives.py.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.common.imaging import FRAME_HEIGHT, FRAME_WIDTH  # noqa: E402
from tools.common.sources import save_ground_truth  # noqa: E402


@dataclass
class WiderFace:
    box: list[float]          # x0, y0, x1, y1
    blur: int
    occlusion: int            # 0 none, 1 partial, 2 heavy
    invalid: bool


def parse_annotations(path: Path) -> dict[str, list[WiderFace]]:
    """wider_face_<split>_bbx_gt.txt: name, count, then one line per face
    'x y w h blur expression illumination invalid occlusion pose' (a zero-face image has one dummy line)."""
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    images: dict[str, list[WiderFace]] = {}
    i = 0
    while i < len(lines):
        name = lines[i]
        if not name:
            i += 1
            continue
        count = int(lines[i + 1])
        faces = []
        for ln in lines[i + 2:i + 2 + max(count, 1)]:
            v = [int(x) for x in ln.split()[:10]]
            if count and v[2] > 0 and v[3] > 0:
                faces.append(WiderFace([v[0], v[1], v[0] + v[2], v[1] + v[3]], v[4], v[8], bool(v[7])))
        images[name] = faces
        i += 2 + max(count, 1)
    return images


def usable(faces: list[WiderFace], width: int, height: int, min_height: float, max_faces: int) -> bool:
    if not 1 <= len(faces) <= max_faces:
        return False
    scale = min(FRAME_WIDTH / width, FRAME_HEIGHT / height)
    return all(not f.invalid and f.occlusion < 2 and (f.box[3] - f.box[1]) * scale >= min_height for f in faces)


def select(images: dict[str, list[WiderFace]], root: Path, count: int, rng: random.Random, min_height: float,
           max_faces: int) -> list[str]:
    # Cheap checks first, then image sizes only for the candidates
    names = [n for n, faces in images.items() if 1 <= len(faces) <= max_faces
             and all(not f.invalid and f.occlusion < 2 for f in faces)]
    rng.shuffle(names)
    chosen = []
    for name in names:
        with Image.open(root / name) as im:
            if usable(images[name], *im.size, min_height, max_faces):
                chosen.append(name)
        if len(chosen) >= count:
            break
    return sorted(chosen)


def copy_set(names: list[str], images: dict[str, list[WiderFace]], src_root: Path, out: Path) -> None:
    gt = {}
    for name in names:
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / name, dst)
        gt[name] = [f.box for f in images[name]]
    save_ground_truth(out / "gt.json", gt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wider", required=True, help="folder holding WIDER_train, WIDER_val, wider_face_split")
    parser.add_argument("--out", default="data/a")
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--cal-count", type=int, default=500)
    parser.add_argument("--min-height", type=float, default=12.0, help="px at 320x240")
    parser.add_argument("--max-faces", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    wider = Path(args.wider)
    split = wider / "wider_face_split"
    val_root, train_root = wider / "WIDER_val" / "images", wider / "WIDER_train" / "images"
    for p in (split / "wider_face_val_bbx_gt.txt", split / "wider_face_train_bbx_gt.txt", val_root, train_root):
        if not p.exists():
            raise SystemExit(f"Missing {p} - see the instructions at the top of this script")
    out = Path(args.out)
    rng = random.Random(args.seed)

    val = parse_annotations(split / "wider_face_val_bbx_gt.txt")
    train = parse_annotations(split / "wider_face_train_bbx_gt.txt")
    print(f"WIDER: {len(val)} val images, {len(train)} train images")

    test = select(val, val_root, args.test_count, rng, args.min_height, args.max_faces)
    copy_set(test, val, val_root, out / "test")
    print(f"A-test: {len(test)} images -> {out / 'test'}")

    cal = select(train, train_root, args.cal_count, rng, args.min_height, args.max_faces)
    copy_set(cal, train, train_root, out / "cal")
    print(f"A-cal: {len(cal)} images -> {out / 'cal'}")

    # Training uses every image with at least one valid face; invalid boxes are left out
    train_gt = {n: [f.box for f in faces if not f.invalid] for n, faces in train.items()}
    train_gt = {n: b for n, b in train_gt.items() if b}
    save_ground_truth(out / "train_gt.json", train_gt)
    print(f"A-train: {len(train_gt)} images listed in {out / 'train_gt.json'} (paths under {train_root})")

    (out / "README.json").write_text(json.dumps({
        "source": str(wider), "seed": args.seed, "min_height_px_at_320x240": args.min_height,
        "faces_per_image": [1, args.max_faces], "dropped": "images with any invalid or heavily occluded face",
        "test": {"split": "val", "images": len(test), "faces": sum(len(val[n]) for n in test)},
        "cal": {"split": "train", "images": len(cal), "faces": sum(len(train[n]) for n in cal)},
        "train": {"images": len(train_gt), "root": str(train_root)},
    }, indent=2))
    print(f"Next: python tools/data/negatives.py <folder of no-face photos> --into {out / 'test'}")


if __name__ == "__main__":
    main()

"""
curate_staging.py — Kurasi sumber BARU ke dataset/_staging/ (foto baru, belum ada di
train/val/test). TIDAK menyentuh train/val/test, _raw/, _clean/. TIDAK menyentuh git.

Sumber & target:
  acne   <- dataset/_raw/imtkaggleteam            (320, prioritas wajah paling jelas)
  dry    <- dataset/_raw/muhammadakbar404 .../Dry    (75)
  normal <- dataset/_raw/muhammadakbar404 .../Normal (63)
  oily   <- dataset/_raw/muhammadakbar404 .../Oily   (63)
  sensitive: TANPA sumber baru -> augmentasi terkontrol dari train/val/test (lihat bawah).

Pipeline per gambar: validasi -> deteksi wajah (DNN res10) -> crop -> resize 224 ->
dedup perceptual-hash ANTAR sumber baru (tidak dedup dgn data existing) -> ambil N terbaik
(by confidence). Output + dataset/_staging/manifest_staging.csv.
"""

import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "_raw"
STAGING = ROOT / "dataset" / "_staging"
IMG_SIZE = 224
CONF_THRESHOLD = 0.5
PHASH_HAMMING = 5
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png"}

SENS_TARGET_TOTAL = 200  # existing(135) + aug -> 200  => 65 aug

random.seed(SEED)
np.random.seed(SEED)

_MODEL_DIR = ROOT / "tools" / "models"
NET = cv2.dnn.readNetFromCaffe(
    str(_MODEL_DIR / "deploy.prototxt"),
    str(_MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"),
)


# ---------------------------------------------------------------------------
# Deteksi wajah / load / crop  (sama dgn prepare_dataset.py)
# ---------------------------------------------------------------------------
def load_bgr(path: Path):
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        return None
    try:
        with Image.open(path) as im:
            arr = np.array(im.convert("RGB"))
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def detect_face(bgr):
    """Return (box, conf) wajah confidence tertinggi, atau (None, 0.0)."""
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    NET.setInput(blob)
    det = NET.forward()
    best, best_conf = None, 0.0
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < CONF_THRESHOLD or conf <= best_conf:
            continue
        x1 = max(0, int(det[0, 0, i, 3] * w)); y1 = max(0, int(det[0, 0, i, 4] * h))
        x2 = min(w, int(det[0, 0, i, 5] * w)); y2 = min(h, int(det[0, 0, i, 6] * h))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        best, best_conf = (x1, y1, x2 - x1, y2 - y1), conf
    return best, best_conf


def crop_resize(bgr, box, margin=0.25):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    H, W = bgr.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(W, int(cx + half)), min(H, int(cy + half))
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def phash_of(bgr):
    return imagehash.phash(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


# ---------------------------------------------------------------------------
# Kumpulkan kandidat per kelas
# ---------------------------------------------------------------------------
def imtkaggle_base(name):
    m = re.match(r"(.+?)_jpg\.rf\.[0-9a-f]+", name)
    return m.group(1) if m else Path(name).stem


def collect(kelas):
    """Return list dict(src, group_id, sumber)."""
    out = []
    if kelas == "acne":
        base = RAW / "imtkaggleteam"
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                out.append({"src": p, "group_id": f"imt_{imtkaggle_base(p.name)}", "sumber": "imtkaggleteam"})
    else:
        folder = {"dry": "Dry", "normal": "Normal", "oily": "Oily"}[kelas]
        base = RAW / "muhammadakbar404"
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS and p.parent.name.lower() == folder.lower():
                out.append({"src": p, "group_id": f"akbar_{p.name}", "sumber": "muhammadakbar404"})
    return out


def curate_class(kelas, target, stats, manifest_rows):
    out_dir = STAGING / kelas
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = collect(kelas)
    passed = []      # dict(conf, phash, group_id, sumber, src)
    n_noface = n_corrupt = 0
    for item in pool:
        bgr = load_bgr(item["src"])
        if bgr is None:
            n_corrupt += 1
            continue
        box, conf = detect_face(bgr)
        if box is None:
            n_noface += 1
            continue
        face = crop_resize(bgr, box)
        if face is None:
            n_noface += 1
            continue
        passed.append({"conf": conf, "phash": phash_of(face),
                       "group_id": item["group_id"], "sumber": item["sumber"], "src": item["src"]})

    # Urutkan by confidence (wajah paling jelas dulu), lalu greedy dedup-select sampai target.
    passed.sort(key=lambda d: -d["conf"])
    selected, seen = [], []
    n_dup = 0
    for d in passed:
        if any((d["phash"] - h) <= PHASH_HAMMING for h in seen):
            n_dup += 1
            continue
        seen.append(d["phash"])
        selected.append(d)
        if len(selected) >= target:
            break

    # Simpan
    for i, d in enumerate(selected, 1):
        bgr = load_bgr(d["src"])
        face = crop_resize(bgr, detect_face(bgr)[0])
        name = f"{kelas}_{d['sumber']}_{i:04d}.jpg"
        cv2.imwrite(str(out_dir / name), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest_rows.append({"path": str((out_dir / name).relative_to(ROOT).as_posix()),
                              "kelas": kelas, "group_id": d["group_id"], "sumber": d["sumber"]})

    stats[kelas] = {"pool": len(pool), "lolos_face": len(passed), "no_face": n_noface,
                    "corrupt": n_corrupt, "dedup_dibuang": n_dup, "final": len(selected), "target": target}


# ---------------------------------------------------------------------------
# Augmentasi sensitive (TANPA hue/sat) — sumber: train/val/test (read-only)
# ---------------------------------------------------------------------------
def augment(img, rng):
    out = img.copy(); h, w = out.shape[:2]
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.8:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.7:
        scale = rng.uniform(0.82, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0 = rng.randint(0, w - cw); y0 = rng.randint(0, h - ch)
        out = cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        M = np.float32([[1, 0, rng.uniform(-0.08, 0.08) * w], [0, 1, rng.uniform(-0.08, 0.08) * h]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.6:
        out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
    return out


def curate_sensitive(stats, manifest_rows):
    out_dir = STAGING / "sensitive"
    out_dir.mkdir(parents=True, exist_ok=True)
    # kumpulkan existing dari train/val/test (read-only)
    srcs = []
    for split in ["train", "val", "test"]:
        d = ROOT / "dataset" / split / "sensitive"
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    srcs.append(p)
    n_existing = len(srcs)
    n_aug = max(0, SENS_TARGET_TOTAL - n_existing)

    rng = random.Random(SEED)
    order = srcs[:]
    rng.shuffle(order)
    aug_rng = random.Random(SEED)
    made = 0
    idx = 0
    while made < n_aug and order:
        src = order[idx % len(order)]
        idx += 1
        img = cv2.imread(str(src))
        if img is None:
            continue
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        made += 1
        name = f"aug_sensitive_{made:04d}.jpg"
        cv2.imwrite(str(out_dir / name), augment(img, aug_rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest_rows.append({"path": str((out_dir / name).relative_to(ROOT).as_posix()),
                              "kelas": "sensitive", "group_id": f"sens_src_{src.name}", "sumber": "aug-sensitive"})
    stats["sensitive"] = {"existing": n_existing, "aug_dibuat": made,
                          "total_setelah": n_existing + made, "target_total": SENS_TARGET_TOTAL}


# ---------------------------------------------------------------------------
def main():
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    stats = {}
    manifest_rows = []
    for kelas, target in [("acne", 320), ("dry", 75), ("normal", 63), ("oily", 63)]:
        curate_class(kelas, target, stats, manifest_rows)
        s = stats[kelas]
        print(f"[{kelas:7s}] pool={s['pool']:5d} lolos_face={s['lolos_face']:5d} "
              f"no_face={s['no_face']:5d} dedup={s['dedup_dibuang']:4d} corrupt={s['corrupt']:3d} "
              f"-> final={s['final']}/{s['target']}")

    curate_sensitive(stats, manifest_rows)
    s = stats["sensitive"]
    print(f"[sensitive] existing={s['existing']} + aug={s['aug_dibuat']} "
          f"= {s['total_setelah']} (target {s['target_total']})")

    with open(STAGING / "manifest_staging.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["path", "kelas", "group_id", "sumber"])
        wcsv.writeheader(); wcsv.writerows(manifest_rows)
    print(f"\nmanifest: {(STAGING / 'manifest_staging.csv').relative_to(ROOT).as_posix()} "
          f"({len(manifest_rows)} baris)")


if __name__ == "__main__":
    main()

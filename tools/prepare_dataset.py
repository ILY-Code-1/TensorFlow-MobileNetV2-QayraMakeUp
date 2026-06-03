"""
prepare_dataset.py — Kurasi dataset kondisi kulit ke dataset/_clean/.

Tahap yang dilakukan (HANYA kurasi):
  1. Kumpulkan sumber per kelas ke pool (termasuk SALIN 100 gambar curated lama
     dari dataset/{train,val,test}/<kelas>/ — copy, bukan pindah).
  2. Validasi (buang corrupt) -> deteksi+crop wajah -> resize 224x224 -> simpan .jpg.
     Wajah tak terdeteksi = dibuang, dihitung per kelas.
  3. Dedup perceptual-hash HANYA untuk 4 kelas besar (oily/dry/normal/acne).
     Kelas 'sensitive' TIDAK di-dedup (augmentasi & angle sengaja dipertahankan).
  4. group_id untuk cegah leakage saat split nanti.
  5. 4 kelas besar dibatasi ~300 kandidat acak/kelas sebelum crop.
  6. Output dataset/_clean/<kelas>/*.jpg + manifest dataset/_clean/manifest.csv.

CATATAN: script ini TIDAK menyentuh isi dataset/{train,val,test} (hanya membaca/menyalin).
"""

import csv
import os
import random
import re
import shutil
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False  # biar gambar korup ketahuan & dibuang

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "_raw"
CLEAN = ROOT / "dataset" / "_clean"
OLD_SPLITS = ["train", "val", "test"]  # folder curated lama (perhatikan: 'val', bukan 'valid')

IMG_SIZE = 224
BIG_CLASSES = ["oily", "dry", "normal", "acne"]

# Cap kandidat acak per kelas (curated lama SELALU disertakan):
ODN_CAP = 400          # oily/dry/normal: 400 kandidat acak per kelas
ACNE_MIN_FACES = 250   # acne: proses seluruh pool (~1.832) sampai dapat >=250 wajah ATAU habis
# sensitive: tanpa cap (proses seluruh pool untuk maksimalkan grup unik)

PHASH_HAMMING_THRESHOLD = 5  # jarak <= ini dianggap duplikat (kelas besar saja)
CONF_THRESHOLD = 0.5         # ambang confidence detektor wajah DNN
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Detektor wajah: OpenCV DNN res10 SSD (jauh lebih kuat dari Haar).
# Catatan: mediapipe tidak dipakai — build yang tersedia memaksa numpy>=2 (bentrok TensorFlow)
# dan tidak menyediakan API solutions. DNN res10 kompatibel numpy<2 (TensorFlow aman).
_MODEL_DIR = ROOT / "tools" / "models"
NET = cv2.dnn.readNetFromCaffe(
    str(_MODEL_DIR / "deploy.prototxt"),
    str(_MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"),
)


# ---------------------------------------------------------------------------
# Kumpulkan kandidat per kelas: list of dict(src, group_id, sumber)
# ---------------------------------------------------------------------------
def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def collect_curated_old(kelas: str):
    """Salin-kandidat dari dataset/{train,val,test}/<kelas>/ (copy logis: cuma daftar)."""
    out = []
    for split in OLD_SPLITS:
        d = ROOT / "dataset" / split / kelas
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if is_img(p):
                out.append({"src": p, "group_id": f"curated_{split}_{p.name}", "sumber": "curated-old"})
    return out


def collect_skin_types(kelas: str):
    """oily/dry/normal dari skin-types (gabung semua split sumber)."""
    base = RAW / "skin-types" / "Oily-Dry-Skin-Types"
    out = []
    for split in ["train", "valid", "test"]:
        d = base / split / kelas
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if is_img(p):
                # kelas besar = file dianggap independen -> group_id = nama file
                out.append({"src": p, "group_id": f"{kelas}_{p.name}", "sumber": "skin-types"})
    return out


def collect_acne():
    d = RAW / "acne" / "Acne"
    out = []
    if d.is_dir():
        for p in sorted(d.rglob("*")):
            if p.is_file() and is_img(p):
                out.append({"src": p, "group_id": f"acne_{p.name}", "sumber": "acne-dataset"})
    return out


def collect_sensitive():
    out = []
    # redness: folder individu 20-29, 3 angle = 1 grup
    redness = RAW / "skin-defects" / "files" / "redness"
    if redness.is_dir():
        for sub in sorted(redness.iterdir()):
            if sub.is_dir():
                gid = f"redness_{sub.name}"
                for p in sorted(sub.iterdir()):
                    if is_img(p):
                        out.append({"src": p, "group_id": gid, "sumber": "redness"})
    # rosacea: base id 'ro<N>', 3 augmentasi .rf.<hash> = 1 grup
    rosacea = RAW / "facial-skin-diseases" / "train" / "rosacea"
    if rosacea.is_dir():
        for p in sorted(rosacea.iterdir()):
            if is_img(p):
                m = re.match(r"^([A-Za-z]+\d+)", p.name)
                base = m.group(1) if m else p.stem
                out.append({"src": p, "group_id": f"rosacea_{base}", "sumber": "rosacea"})
    return out


def build_pool(kelas: str):
    if kelas in ("oily", "dry", "normal"):
        raw = collect_skin_types(kelas)
    elif kelas == "acne":
        raw = collect_acne()
    elif kelas == "sensitive":
        raw = collect_sensitive()
    else:
        raw = []

    curated = collect_curated_old(kelas)

    if kelas in ("oily", "dry", "normal"):
        # curated lama SELALU disertakan; sisanya diisi acak dari raw sampai cap 400.
        random.shuffle(raw)
        slots = max(0, ODN_CAP - len(curated))
        pool = curated + raw[:slots]
    elif kelas == "acne":
        # proses SELURUH pool (diacak); early-stop di loop utama saat >=ACNE_MIN_FACES wajah.
        random.shuffle(raw)
        pool = curated + raw
    else:
        # sensitive: ambil semua (langka), tanpa cap -> maksimalkan grup unik
        pool = curated + raw
    return pool


# ---------------------------------------------------------------------------
# Pemrosesan gambar
# ---------------------------------------------------------------------------
def load_bgr(path: Path):
    """Validasi + load sebagai BGR uint8. Return None kalau korup/tak terbaca."""
    try:
        with Image.open(path) as im:
            im.verify()  # cek integritas
    except Exception:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.array(im)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def detect_face(bgr):
    """Deteksi wajah pakai DNN res10 SSD. Return (x,y,w,h) wajah confidence tertinggi, atau None."""
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    NET.setInput(blob)
    det = NET.forward()
    best, best_conf = None, 0.0
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < CONF_THRESHOLD or conf <= best_conf:
            continue
        x1 = int(det[0, 0, i, 3] * w); y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w); y2 = int(det[0, 0, i, 6] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        best, best_conf = (x1, y1, x2 - x1, y2 - y1), conf
    return best


def crop_resize(bgr, box, margin=0.25):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    H, W = bgr.shape[:2]
    x0 = max(0, int(cx - half)); y0 = max(0, int(cy - half))
    x1 = min(W, int(cx + half)); y1 = min(H, int(cy + half))
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def phash_of(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True, exist_ok=True)

    classes = ["oily", "dry", "normal", "acne", "sensitive"]
    manifest_rows = []
    stats = {}

    for kelas in classes:
        out_dir = CLEAN / kelas
        out_dir.mkdir(parents=True, exist_ok=True)

        pool = build_pool(kelas)
        dedup = kelas in BIG_CLASSES
        seen_hashes = []  # untuk dedup kelas besar

        n_clean = n_noface = n_corrupt = n_dup = 0
        groups = set()
        idx = 0

        for item in pool:
            # acne: berhenti begitu sudah dapat minimal ACNE_MIN_FACES wajah bersih
            if kelas == "acne" and n_clean >= ACNE_MIN_FACES:
                break
            bgr = load_bgr(item["src"])
            if bgr is None:
                n_corrupt += 1
                continue
            box = detect_face(bgr)
            if box is None:
                n_noface += 1
                continue
            face = crop_resize(bgr, box)
            if face is None:
                n_noface += 1
                continue

            if dedup:
                hsh = phash_of(face)
                if any((hsh - prev) <= PHASH_HAMMING_THRESHOLD for prev in seen_hashes):
                    n_dup += 1
                    continue
                seen_hashes.append(hsh)

            idx += 1
            out_name = f"{kelas}_{item['sumber']}_{idx:04d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])

            groups.add(item["group_id"])
            manifest_rows.append(
                {
                    "path": str(out_path.relative_to(ROOT).as_posix()),
                    "kelas": kelas,
                    "group_id": item["group_id"],
                    "sumber": item["sumber"],
                }
            )
            n_clean += 1

        stats[kelas] = {
            "pool": len(pool),
            "clean": n_clean,
            "noface": n_noface,
            "corrupt": n_corrupt,
            "dup": n_dup,
            "groups": len(groups),
        }
        print(
            f"[{kelas:9s}] pool={len(pool):4d} -> bersih={n_clean:4d} | "
            f"no-face={n_noface:4d} | dup={n_dup:4d} | corrupt={n_corrupt:3d} | grup={len(groups)}"
        )

    # tulis manifest
    with open(CLEAN / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "kelas", "group_id", "sumber"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nmanifest: {(CLEAN / 'manifest.csv').relative_to(ROOT).as_posix()} "
          f"({len(manifest_rows)} baris)")
    return stats


if __name__ == "__main__":
    main()

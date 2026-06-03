"""
split_dataset.py — Tahap FINAL: split + augmentasi -> tulis ke dataset/{train,val,test}.

Langkah:
  0. Kosongkan dataset/{train,val,test}/<kelas>/ dari gambar lama (.gitkeep dipertahankan).
  1. 4 kelas besar (oily/dry/normal/acne): ambil acak 200 dari _clean, split 140/40/20.
  2. sensitive: split GROUP-AWARE pakai group_id (tanpa memecah grup), proporsi ~70/20/10.
     - train diaugmentasi sampai total 140; val & test ASLI saja (apa adanya).
  3. Augmentasi sensitive: GEOMETRIC (flip/rotasi kecil/zoom/translasi) + brightness ringan.
     TIDAK mengubah hue/saturation (kemerahan = sinyal warna yang harus dijaga). Prefiks 'aug_'.
  4. labels.txt ditulis ulang URUTAN ALFABET (cocok dengan flow_from_directory).

Reproducible: seed tetap.
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "dataset" / "_clean"
MANIFEST = CLEAN / "manifest.csv"

IMG_SIZE = 224
SEED = 42
SPLITS = ["train", "val", "test"]

BIG_CLASSES = ["oily", "dry", "normal", "acne"]
BIG_TAKE = 200
BIG_SPLIT = {"train": 140, "val": 40, "test": 20}

SENS_TRAIN_TARGET = 140                       # train sensitive diaugmentasi sampai sebanyak ini
SENS_FRAC = {"train": 0.70, "val": 0.20, "test": 0.10}

ALPHA_CLASSES = ["acne", "dry", "normal", "oily", "sensitive"]  # urutan flow_from_directory

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def clear_old_images():
    """Hapus *.jpg/*.jpeg/*.png lama di train/val/test, pertahankan .gitkeep."""
    n = 0
    for split in SPLITS:
        for kelas in ALPHA_CLASSES:
            d = ROOT / "dataset" / split / kelas
            d.mkdir(parents=True, exist_ok=True)
            for p in d.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    p.unlink()
                    n += 1
    return n


def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["kelas"]].append(r)
    return by_class


def copy_to(split, kelas, src_path: Path, name: str):
    dst = ROOT / "dataset" / split / kelas / name
    shutil.copy2(src_path, dst)
    return dst


# ---------------------------------------------------------------------------
# Augmentasi sensitive (geometric + brightness, TANPA hue/sat)
# ---------------------------------------------------------------------------
def augment(img, rng):
    out = img.copy()
    h, w = out.shape[:2]

    # flip horizontal
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)

    # rotasi kecil ±15 derajat
    if rng.random() < 0.8:
        ang = rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # zoom/crop ringan (skala 0.82–1.0 -> resize balik)
    if rng.random() < 0.7:
        scale = rng.uniform(0.82, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0 = rng.randint(0, w - cw); y0 = rng.randint(0, h - ch)
        out = cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)

    # translasi ringan (±8% )
    if rng.random() < 0.5:
        tx = rng.uniform(-0.08, 0.08) * w
        ty = rng.uniform(-0.08, 0.08) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # brightness ringan: skala SEMUA channel sama -> hue & rasio saturasi terjaga
    if rng.random() < 0.6:
        factor = rng.uniform(0.85, 1.15)
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# Split kelas besar
# ---------------------------------------------------------------------------
def split_big(kelas, rows, report):
    paths = [ROOT / r["path"] for r in rows]
    random.shuffle(paths)
    chosen = paths[:BIG_TAKE]
    i = 0
    for split in SPLITS:
        n = BIG_SPLIT[split]
        for p in chosen[i:i + n]:
            copy_to(split, kelas, p, p.name)
        report[kelas][split] = {"asli": n, "aug": 0}
        i += n


# ---------------------------------------------------------------------------
# Split sensitive (group-aware) + augmentasi train
# ---------------------------------------------------------------------------
def split_sensitive(rows, report):
    kelas = "sensitive"
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(ROOT / r["path"])

    total = sum(len(v) for v in groups.values())
    targets = {s: SENS_FRAC[s] * total for s in SPLITS}
    current = {s: 0 for s in SPLITS}
    assign = {s: [] for s in SPLITS}

    # grup besar dulu, tiap grup ke split dengan defisit terbesar (tanpa memecah grup)
    group_items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    # acak tie-break secara deterministik
    rng = random.Random(SEED)
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: -len(kv[1]))
    for gid, files in group_items:
        s = max(SPLITS, key=lambda s: targets[s] - current[s])
        assign[s].extend(files)
        current[s] += len(files)

    # tulis ASLI untuk semua split
    for split in SPLITS:
        for p in assign[split]:
            copy_to(split, kelas, p, p.name)

    n_train_asli = len(assign["train"])
    n_val = len(assign["val"])
    n_test = len(assign["test"])

    # augmentasi HANYA train -> sampai total SENS_TRAIN_TARGET
    aug_rng = random.Random(SEED)
    n_aug = 0
    src_imgs = assign["train"][:]  # sumber augmentasi = gambar asli train
    if src_imgs:
        idx = 0
        while n_train_asli + n_aug < SENS_TRAIN_TARGET:
            src = src_imgs[idx % len(src_imgs)]
            img = cv2.imread(str(src))
            if img is None:
                idx += 1
                continue
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            aug = augment(img, aug_rng)
            n_aug += 1
            out = ROOT / "dataset" / "train" / kelas / f"aug_sensitive_{n_aug:04d}.jpg"
            cv2.imwrite(str(out), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
            idx += 1

    report[kelas]["train"] = {"asli": n_train_asli, "aug": n_aug}
    report[kelas]["val"] = {"asli": n_val, "aug": 0}
    report[kelas]["test"] = {"asli": n_test, "aug": 0}
    report["_sensitive_groups"] = {s: len({_gid_of(p) for p in assign[s]}) for s in SPLITS}


def _gid_of(path: Path):
    """Map balik file ke group_id via lookup manifest (di-cache)."""
    return _PATH2GID.get(path.as_posix(), path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_PATH2GID = {}


def main():
    global _PATH2GID
    rows_all = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    _PATH2GID = {(ROOT / r["path"]).as_posix(): r["group_id"] for r in rows_all}

    n_cleared = clear_old_images()
    print(f"gambar lama dihapus: {n_cleared}")

    by_class = load_manifest()
    report = defaultdict(dict)

    for kelas in BIG_CLASSES:
        split_big(kelas, by_class[kelas], report)
    split_sensitive(by_class["sensitive"], report)

    # labels.txt urutan alfabet
    (ROOT / "labels.txt").write_text("\n".join(ALPHA_CLASSES) + "\n", encoding="utf-8")

    # laporan
    print("\n=== HASIL SPLIT (gambar per kelas per split) ===")
    print(f"{'kelas':10s} {'train':>14s} {'val':>10s} {'test':>10s}")
    for kelas in ALPHA_CLASSES:
        cells = []
        for split in SPLITS:
            d = report[kelas][split]
            if kelas == "sensitive" and split == "train":
                cells.append(f"{d['asli']}+{d['aug']}aug={d['asli']+d['aug']}")
            else:
                cells.append(str(d["asli"]))
        print(f"{kelas:10s} {cells[0]:>14s} {cells[1]:>10s} {cells[2]:>10s}")

    sg = report["_sensitive_groups"]
    print(f"\nsensitive grup unik per split (cek leakage): "
          f"train={sg['train']} val={sg['val']} test={sg['test']} (harus 0 irisan)")
    print("labels.txt ->", ALPHA_CLASSES)


if __name__ == "__main__":
    main()

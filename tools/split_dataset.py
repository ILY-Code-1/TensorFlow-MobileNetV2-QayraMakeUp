"""
split_dataset.py — Tahap FINAL: split GROUP-AWARE + augmentasi -> tulis ke dataset/{train,val,test}.

Langkah:
  0. Kosongkan dataset/{train,val,test}/<kelas>/ dari gambar lama (.gitkeep dipertahankan).
  1. Semua kelas displit GROUP-AWARE pakai group_id dari manifest (tanpa memecah grup),
     proporsi 70/20/10. Target per kelas:
       - normal / dry / oily : 350  (gabungan skin-types + killa92, sudah deduped)
       - acne                : 250
       - sensitive           : semua yang ada (lalu train diaugmentasi -> 140)
  2. sensitive: train diaugmentasi sampai 140; val & test ASLI saja (apa adanya).
     Augmentasi GEOMETRIC (flip/rotasi kecil/zoom/translasi) + brightness ringan,
     TANPA ubah hue/saturation (kemerahan = sinyal warna yang harus dijaga). Prefiks 'aug_'.
  3. labels.txt ditulis ulang URUTAN ALFABET (cocok dengan flow_from_directory).
  4. Hitung dict class_weight (balanced) dari jumlah TRAIN final -> dipakai di train.py.

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
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}

# Target jumlah gambar per kelas (None = pakai semua yang tersedia).
TARGETS = {"normal": 350, "dry": 350, "oily": 350, "acne": 250, "sensitive": None}
SENS_TRAIN_TARGET = 140  # sensitive: train diaugmentasi sampai sebanyak ini

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


def copy_to(split, kelas, src_path: Path, name: str):
    shutil.copy2(src_path, ROOT / "dataset" / split / kelas / name)


# ---------------------------------------------------------------------------
# Split GROUP-AWARE terpadu
# ---------------------------------------------------------------------------
def group_aware_assign(rows, target_total):
    """rows: list manifest-row utk satu kelas. Return assign{split:[(gid,path)]}.
    - Pilih grup (acak deterministik) sampai jumlah file >= target_total (None=semua).
    - Bagi grup terpilih ke split menurut RATIOS, grup besar dulu ke split paling defisit.
      Satu grup TIDAK pernah terpecah -> tidak ada leakage antar split."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(ROOT / r["path"])

    items = list(groups.items())
    rng = random.Random(SEED)
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))  # grup besar dulu (stabil setelah shuffle)

    if target_total is None:
        selected = items
    else:
        selected, cnt = [], 0
        for gid, files in items:
            if cnt >= target_total:
                break
            selected.append((gid, files))
            cnt += len(files)

    sel_total = sum(len(f) for _, f in selected)
    targets = {s: RATIOS[s] * sel_total for s in SPLITS}
    current = {s: 0 for s in SPLITS}
    assign = {s: [] for s in SPLITS}
    for gid, files in selected:  # sudah terurut grup besar dulu
        s = max(SPLITS, key=lambda s: targets[s] - current[s])
        assign[s].extend((gid, p) for p in files)
        current[s] += len(files)
    return assign


# ---------------------------------------------------------------------------
# Augmentasi sensitive (geometric + brightness, TANPA hue/sat)
# ---------------------------------------------------------------------------
def augment(img, rng):
    out = img.copy()
    h, w = out.shape[:2]
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.8:
        ang = rng.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.7:
        scale = rng.uniform(0.82, 1.0)
        cw, ch = int(w * scale), int(h * scale)
        x0 = rng.randint(0, w - cw); y0 = rng.randint(0, h - ch)
        out = cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        tx = rng.uniform(-0.08, 0.08) * w
        ty = rng.uniform(-0.08, 0.08) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.6:
        factor = rng.uniform(0.85, 1.15)
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def augment_train_to(kelas, assign, target):
    """Augmentasi bagian TRAIN sebuah kelas sampai total `target` gambar."""
    src_imgs = [p for _, p in assign["train"]]
    n_asli = len(src_imgs)
    n_aug = 0
    if src_imgs and n_asli < target:
        aug_rng = random.Random(SEED)
        idx = 0
        while n_asli + n_aug < target:
            src = src_imgs[idx % len(src_imgs)]
            img = cv2.imread(str(src))
            idx += 1
            if img is None:
                continue
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            n_aug += 1
            out = ROOT / "dataset" / "train" / kelas / f"aug_{kelas}_{n_aug:04d}.jpg"
            cv2.imwrite(str(out), augment(img, aug_rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return n_asli, n_aug


# ---------------------------------------------------------------------------
# class_weight (balanced) dari jumlah TRAIN final
# ---------------------------------------------------------------------------
def compute_class_weight():
    counts = {}
    for i, kelas in enumerate(ALPHA_CLASSES):
        d = ROOT / "dataset" / "train" / kelas
        counts[i] = sum(1 for p in d.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    total = sum(counts.values())
    n = len(counts)
    weights = {i: round(total / (n * counts[i]), 4) if counts[i] else 0.0 for i in counts}
    return counts, weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows_all = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_class = defaultdict(list)
    for r in rows_all:
        by_class[r["kelas"]].append(r)

    n_cleared = clear_old_images()
    print(f"gambar lama dihapus: {n_cleared}")

    report = {}
    groups_per_split = {}
    for kelas in ALPHA_CLASSES:
        assign = group_aware_assign(by_class[kelas], TARGETS[kelas])
        # tulis gambar ASLI
        for split in SPLITS:
            for _, p in assign[split]:
                copy_to(split, kelas, p, p.name)
        rep = {s: {"asli": len(assign[s]), "aug": 0} for s in SPLITS}
        # sensitive: augmentasi train -> 140
        if kelas == "sensitive":
            n_asli, n_aug = augment_train_to(kelas, assign, SENS_TRAIN_TARGET)
            rep["train"] = {"asli": n_asli, "aug": n_aug}
        report[kelas] = rep
        groups_per_split[kelas] = {s: {gid for gid, _ in assign[s]} for s in SPLITS}

    # labels.txt urutan alfabet
    (ROOT / "labels.txt").write_text("\n".join(ALPHA_CLASSES) + "\n", encoding="utf-8")

    # ---- laporan ----
    print("\n=== HASIL SPLIT (gambar per kelas per split) ===")
    print(f"{'kelas':10s} {'train':>16s} {'val':>8s} {'test':>8s} {'total':>8s}")
    for kelas in ALPHA_CLASSES:
        r = report[kelas]
        tr = r["train"]["asli"] + r["train"]["aug"]
        tr_cell = (f"{r['train']['asli']}+{r['train']['aug']}a={tr}"
                   if r["train"]["aug"] else str(tr))
        total = tr + r["val"]["asli"] + r["test"]["asli"]
        print(f"{kelas:10s} {tr_cell:>16s} {r['val']['asli']:>8d} {r['test']['asli']:>8d} {total:>8d}")

    # cek leakage grup antar split (harus tidak ada irisan)
    print("\n=== Cek leakage grup antar split ===")
    any_leak = False
    for kelas in ALPHA_CLASSES:
        g = groups_per_split[kelas]
        leak = (g["train"] & g["val"]) | (g["train"] & g["test"]) | (g["val"] & g["test"])
        any_leak = any_leak or bool(leak)
        print(f"  {kelas:10s} grup train/val/test = {len(g['train'])}/{len(g['val'])}/{len(g['test'])}"
              f" | irisan: {len(leak)}")
    print("  LEAKAGE ADA?" , any_leak)

    # class_weight
    counts, weights = compute_class_weight()
    print("\n=== class_weight (balanced, dari jumlah TRAIN final) ===")
    print("  index alfabet:", {i: ALPHA_CLASSES[i] for i in range(len(ALPHA_CLASSES))})
    print("  train counts :", counts)
    print("  class_weight :", weights)
    print("\nlabels.txt ->", ALPHA_CLASSES)


if __name__ == "__main__":
    main()

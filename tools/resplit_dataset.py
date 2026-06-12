"""
resplit_dataset.py — Kumpulkan SEMUA foto tersisa di train/val/test per kelas,
lalu bagi ulang 70/20/10 secara GROUP-AWARE.

Constraint: TIDAK menyentuh _raw/ atau _clean/ (manifest hanya DIBACA, read-only).
TIDAK mengubah file lain. TIDAK menyentuh git.

group_id (anti-leakage):
  Sumber utama = dataset/_clean/manifest.csv (basename -> group_id asli), karena file di
  train/val/test sudah di-rename pipeline (pola .rf./front/aug_ hilang). Manifest memetakan
  100% basename -> grup asli: redness_<individu>, rosacea_<base-id>, killa92_<base>, dst.
  FALLBACK (bila ada file di luar manifest) memakai aturan nama-file:
    - prefix 'aug_'        -> nama tanpa 'aug_' & tanpa suffix angka (semua aug 1 sumber = 1 grup)
    - mengandung '.rf.'    -> base-id sebelum '.rf.'
    - lainnya              -> nama file itu sendiri (independen)
"""

import csv
import random
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test"]
CLASSES = ["acne", "dry", "normal", "oily", "sensitive"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42
MANIFEST = ROOT / "dataset" / "_clean" / "manifest.csv"

# Ambang penilaian retrain
MIN_TRAIN_PER_CLASS = 30
MAX_IMBALANCE_RATIO = 5.0


# ---------------------------------------------------------------------------
# group_id
# ---------------------------------------------------------------------------
def load_manifest_map():
    """basename -> group_id (read-only). Kosong jika manifest tak ada."""
    m = {}
    if MANIFEST.is_file():
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
            m[Path(r["path"]).name] = r["group_id"]
    return m


def fallback_group_id(name: str) -> str:
    if name.startswith("aug_"):
        stem = Path(name).stem[len("aug_"):]          # buang prefix aug_
        return re.sub(r"_\d+$", "", stem)             # buang suffix angka
    if ".rf." in name:
        return name.split(".rf.")[0]
    return name                                       # independen


# ---------------------------------------------------------------------------
# Group-aware split 70/20/10 (grup tidak pernah terpecah)
# ---------------------------------------------------------------------------
def group_split(groups):
    """groups: dict gid -> [items]. Return assign{split:[items]}.
    Grup besar dulu -> ditaruh ke split dengan defisit terbesar (target = ratio x total file)."""
    items = list(groups.items())
    rng = random.Random(SEED)
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))

    total = sum(len(v) for _, v in items)
    targets = {s: RATIOS[s] * total for s in SPLITS}
    current = {s: 0 for s in SPLITS}
    assign = {s: [] for s in SPLITS}
    for gid, files in items:
        s = max(SPLITS, key=lambda s: targets[s] - current[s])
        assign[s].extend(files)
        current[s] += len(files)
    return assign


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    man = load_manifest_map()

    # 1. KUMPULKAN pool per kelas (dari train+val+test) + group_id.
    pool = {k: [] for k in CLASSES}     # item: dict(name, gid)
    staging = Path(tempfile.mkdtemp(prefix="resplit_"))
    pre_counts = {k: 0 for k in CLASSES}
    for k in CLASSES:
        (staging / k).mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            d = ROOT / "dataset" / split / k
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                gid = man.get(p.name) or fallback_group_id(p.name)
                # staging copy supaya aman saat folder asli dikosongkan
                shutil.copy2(p, staging / k / p.name)
                pool[k].append({"name": p.name, "gid": gid})
                pre_counts[k] += 1

    # 2+3. Split group-aware per kelas.
    assigns = {}
    for k in CLASSES:
        groups = defaultdict(list)
        for it in pool[k]:
            groups[it["gid"]].append(it["name"])
        assigns[k] = group_split(groups)

    # 4. Kosongkan train/val/test (pertahankan .gitkeep), lalu tulis hasil split.
    n_cleared = 0
    for split in SPLITS:
        for k in CLASSES:
            d = ROOT / "dataset" / split / k
            d.mkdir(parents=True, exist_ok=True)
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    p.unlink(); n_cleared += 1
    for k in CLASSES:
        for split in SPLITS:
            for name in assigns[k][split]:
                shutil.copy2(staging / k / name, ROOT / "dataset" / split / k / name)

    shutil.rmtree(staging, ignore_errors=True)

    # ---- 6. LAPORAN ----
    print(f"gambar lama dihapus: {n_cleared}\n")
    print("=== HASIL RE-SPLIT (file per kelas per split) ===")
    hdr = f"{'kelas':<11}{'train':>8}{'val':>7}{'test':>7}{'total':>8}{'grup':>7}"
    print(hdr); print("-" * len(hdr))
    train_counts = {}
    grand = {"train": 0, "val": 0, "test": 0}
    for k in CLASSES:
        cnt = {s: len(assigns[k][s]) for s in SPLITS}
        tot = sum(cnt.values())
        n_groups = len({it["gid"] for it in pool[k]})
        train_counts[k] = cnt["train"]
        for s in SPLITS:
            grand[s] += cnt[s]
        print(f"{k:<11}{cnt['train']:>8}{cnt['val']:>7}{cnt['test']:>7}{tot:>8}{n_groups:>7}")
    print("-" * len(hdr))
    gtot = sum(grand.values())
    print(f"{'TOTAL':<11}{grand['train']:>8}{grand['val']:>7}{grand['test']:>7}{gtot:>8}")

    # cek leakage grup antar split
    print("\n=== Cek leakage grup antar split ===")
    any_leak = False
    for k in CLASSES:
        gsets = {s: set() for s in SPLITS}
        name2gid = {it["name"]: it["gid"] for it in pool[k]}
        for s in SPLITS:
            for name in assigns[k][s]:
                gsets[s].add(name2gid[name])
        leak = (gsets["train"] & gsets["val"]) | (gsets["train"] & gsets["test"]) | (gsets["val"] & gsets["test"])
        any_leak = any_leak or bool(leak)
        print(f"  {k:<11} irisan grup: {len(leak)}")
    print("  LEAKAGE ADA?", any_leak)

    # ketimpangan train + class_weight
    mx = max(train_counts.values()); mn = min(train_counts.values())
    ratio = mx / mn if mn else float("inf")
    kmx = next(k for k in CLASSES if train_counts[k] == mx)
    kmn = next(k for k in CLASSES if train_counts[k] == mn)
    total_tr = sum(train_counts.values()); n = len(CLASSES)
    cw = {i: round(total_tr / (n * train_counts[CLASSES[i]]), 4) if train_counts[CLASSES[i]] else 0.0
          for i in range(n)}
    print(f"\n=== Ketimpangan train ===")
    print(f"  terbesar={mx} ({kmx}) / terkecil={mn} ({kmn}) = {ratio:.2f}x")
    print(f"\n=== Perkiraan class_weight BARU (balanced, dari train hasil split) ===")
    print(f"  index alfabet: {{0:acne,1:dry,2:normal,3:oily,4:sensitive}}")
    print(f"  train counts : {{{', '.join(f'{i}:{train_counts[CLASSES[i]]}' for i in range(n))}}}")
    print(f"  class_weight : {cw}")
    print("  -> train.py menghitung class_weight ini DINAMIS dari train_data.classes;")
    print("     tidak ada perubahan manual yang diperlukan.")

    # penilaian retrain
    too_small = [k for k in CLASSES if train_counts[k] < MIN_TRAIN_PER_CLASS]
    safe = (not too_small) and (ratio < MAX_IMBALANCE_RATIO)
    print(f"\n=== AMAN UNTUK RETRAIN? ===")
    print(f"  patokan: train >= {MIN_TRAIN_PER_CLASS}/kelas DAN rasio < {MAX_IMBALANCE_RATIO}x")
    if too_small:
        print(f"  - kelas train < {MIN_TRAIN_PER_CLASS}: {too_small}")
    if ratio >= MAX_IMBALANCE_RATIO:
        print(f"  - rasio ketimpangan {ratio:.2f}x >= {MAX_IMBALANCE_RATIO}x")
    print(f"  STATUS: {'AMAN' if safe else 'TIDAK AMAN'}")


if __name__ == "__main__":
    main()

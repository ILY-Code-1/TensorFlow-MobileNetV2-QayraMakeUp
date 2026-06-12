"""
merge_resplit.py — Merge existing (train/val/test) + _staging lalu RESPLIT group-aware 70/20/10.

TIDAK menyentuh _raw/ atau _clean/. TIDAK menyentuh git. TIDAK training.

Pool per kelas (di memori):
  big (acne/dry/normal/oily): file existing (group_id=nama file) + file _staging
      (group_id dari manifest_staging.csv).
  sensitive: 135 foto ASLI dari train/val/test (abaikan aug_* yang mungkin tersisa)
      + 65 aug_* dari _staging. group_id aug = nama file ASLI sumbernya -> 1 grup dgn
      sumbernya.

Split:
  big      : group-aware 70/20/10 (grup utuh, grup besar dulu ke split paling defisit).
  sensitive: val & test HANYA foto ASLI (bebas aug_). Caranya: semua grup yang mengandung
      aug_ dipaksa ke TRAIN (mencegah aug bocor & menjaga val/test murni asli); val/test
      diisi dari grup foto-asli-tanpa-aug. Tetap group-aware, proporsi ~70/20/10.
"""

import csv
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test"]
CLASSES = ["acne", "dry", "normal", "oily", "sensitive"]
BIG = ["acne", "dry", "normal", "oily"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED = 42
STAGING = ROOT / "dataset" / "_staging"
STAGING_MANIFEST = STAGING / "manifest_staging.csv"


def staging_gid_map():
    m = {}
    if STAGING_MANIFEST.is_file():
        for r in csv.DictReader(open(STAGING_MANIFEST, encoding="utf-8")):
            m[Path(r["path"]).name] = r["group_id"]
    return m


def collect_pool(kelas, sgid):
    """Return list of dict(src, name, gid, is_aug)."""
    items = []
    # existing dari train/val/test
    for split in SPLITS:
        d = ROOT / "dataset" / split / kelas
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if kelas == "sensitive" and p.name.startswith("aug_"):
                continue  # abaikan aug_* sisa di folder utama
            items.append({"src": p, "name": p.name, "gid": p.name, "is_aug": False})
    # _staging
    sdir = STAGING / kelas
    if sdir.is_dir():
        for p in sorted(sdir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            is_aug = p.name.startswith("aug_")
            gid = sgid.get(p.name, p.name)
            if kelas == "sensitive" and is_aug:
                gid = gid.replace("sens_src_", "")  # samakan dgn group_id foto asli sumbernya
            items.append({"src": p, "name": p.name, "gid": gid, "is_aug": is_aug})
    return items


def group_split_big(items):
    """Group-aware 70/20/10 (grup besar dulu -> split paling defisit)."""
    groups = defaultdict(list)
    for it in items:
        groups[it["gid"]].append(it)
    glist = list(groups.items())
    rng = random.Random(SEED)
    rng.shuffle(glist)
    glist.sort(key=lambda kv: -len(kv[1]))
    total = sum(len(v) for _, v in glist)
    targets = {s: RATIOS[s] * total for s in SPLITS}
    cur = {s: 0 for s in SPLITS}
    assign = {s: [] for s in SPLITS}
    for _, files in glist:
        s = max(SPLITS, key=lambda s: targets[s] - cur[s])
        assign[s].extend(files)
        cur[s] += len(files)
    return assign


def split_sensitive(items):
    """val/test HANYA foto asli; grup ber-aug dipaksa ke train."""
    groups = defaultdict(list)
    for it in items:
        groups[it["gid"]].append(it)
    aug_groups = {g: fs for g, fs in groups.items() if any(f["is_aug"] for f in fs)}
    orig_groups = {g: fs for g, fs in groups.items() if not any(f["is_aug"] for f in fs)}

    total = sum(len(v) for v in groups.values())
    n_val = round(RATIOS["val"] * total)
    n_test = round(RATIOS["test"] * total)

    og = list(orig_groups.items())
    rng = random.Random(SEED)
    rng.shuffle(og)

    assign = {s: [] for s in SPLITS}
    # test dulu, lalu val, dari grup foto-asli (semuanya 1 file)
    i = 0
    while sum(len(f) for f in [assign["test"]]) < n_test and i < len(og):
        assign["test"].extend(og[i][1]); i += 1
    while len(assign["val"]) < n_val and i < len(og):
        assign["val"].extend(og[i][1]); i += 1
    # sisa grup asli -> train; semua grup ber-aug -> train
    for g, fs in og[i:]:
        assign["train"].extend(fs)
    for g, fs in aug_groups.items():
        assign["train"].extend(fs)
    return assign


def main():
    sgid = staging_gid_map()
    pools = {k: collect_pool(k, sgid) for k in CLASSES}

    # stage semua file terpilih ke temp (karena train/val/test akan dikosongkan)
    tmp = Path(tempfile.mkdtemp(prefix="mergeresplit_"))
    assigns = {}
    for k in CLASSES:
        (tmp / k).mkdir(parents=True, exist_ok=True)
        for it in pools[k]:
            shutil.copy2(it["src"], tmp / k / it["name"])
        assigns[k] = split_sensitive(pools[k]) if k == "sensitive" else group_split_big(pools[k])

    # kosongkan train/val/test (pertahankan .gitkeep)
    n_cleared = 0
    for split in SPLITS:
        for k in CLASSES:
            d = ROOT / "dataset" / split / k
            d.mkdir(parents=True, exist_ok=True)
            for p in d.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    p.unlink(); n_cleared += 1
    # tulis hasil split
    for k in CLASSES:
        for split in SPLITS:
            for it in assigns[k][split]:
                shutil.copy2(tmp / k / it["name"], ROOT / "dataset" / split / k / it["name"])
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- laporan ----
    print(f"gambar lama dihapus: {n_cleared}\n")
    hdr = f"{'kelas':<10}{'train':>8}{'val':>7}{'test':>7}{'total':>8}{'grup':>7}"
    print("=== HASIL MERGE + RESPLIT ===")
    print(hdr); print("-" * len(hdr))
    train_counts = {}
    grand = {"train": 0, "val": 0, "test": 0}
    for k in CLASSES:
        cnt = {s: len(assigns[k][s]) for s in SPLITS}
        train_counts[k] = cnt["train"]
        for s in SPLITS:
            grand[s] += cnt[s]
        ng = len({it["gid"] for it in pools[k]})
        print(f"{k:<10}{cnt['train']:>8}{cnt['val']:>7}{cnt['test']:>7}{sum(cnt.values()):>8}{ng:>7}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<10}{grand['train']:>8}{grand['val']:>7}{grand['test']:>7}{sum(grand.values()):>8}")

    # sensitive: konfirmasi val/test bebas aug_
    print("\n=== Sensitive: val/test bebas aug_? ===")
    for s in SPLITS:
        d = ROOT / "dataset" / s / "sensitive"
        naug = sum(1 for p in d.iterdir() if p.name.startswith("aug_"))
        tot = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        print(f"  {s:<6}: total={tot:>3} | aug_={naug}")

    # leakage check semua kelas
    print("\n=== Cek leakage grup antar split ===")
    any_leak = False
    for k in CLASSES:
        gs = {s: {it["gid"] for it in assigns[k][s]} for s in SPLITS}
        leak = (gs["train"] & gs["val"]) | (gs["train"] & gs["test"]) | (gs["val"] & gs["test"])
        any_leak = any_leak or bool(leak)
        print(f"  {k:<10} irisan grup: {len(leak)}")
    print("  LEAKAGE ADA?", any_leak)

    # ketimpangan + class_weight
    mx = max(train_counts.values()); mn = min(train_counts.values())
    ratio = mx / mn if mn else float("inf")
    kmx = next(k for k in CLASSES if train_counts[k] == mx)
    kmn = next(k for k in CLASSES if train_counts[k] == mn)
    tot_tr = sum(train_counts.values()); n = len(CLASSES)
    cw = {i: round(tot_tr / (n * train_counts[CLASSES[i]]), 4) for i in range(n)}
    print(f"\n=== Ketimpangan train: {mx} ({kmx}) / {mn} ({kmn}) = {ratio:.2f}x ===")
    print("=== Perkiraan class_weight BARU (train.py hitung otomatis) ===")
    print(f"  index alfabet: {{0:acne,1:dry,2:normal,3:oily,4:sensitive}}")
    print(f"  train counts : {{{', '.join(f'{i}:{train_counts[CLASSES[i]]}' for i in range(n))}}}")
    print(f"  class_weight : {cw}")

    too_small = [k for k in CLASSES if train_counts[k] < 30]
    safe = (not too_small) and ratio < 5
    print(f"\n=== AMAN UNTUK RETRAIN? (train>=30/kelas & rasio<5x) -> {'AMAN' if safe else 'TIDAK AMAN'} ===")
    if too_small:
        print(f"  kelas train<30: {too_small}")


if __name__ == "__main__":
    main()

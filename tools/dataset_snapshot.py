"""
dataset_snapshot.py — Snapshot kondisi dataset + baseline evaluasi model saat ini.

HANYA membaca/menghitung. TIDAK mengubah/menghapus gambar. TIDAK menyentuh git.

Output:
  - Tabel jumlah gambar per kelas per split (train/val/test) + total per split + grand total.
  - Baseline evaluasi model terakhir (di-hardcode dari hasil yang diberikan).
  - Dicetak ke terminal + disimpan ke docs/dataset_snapshot_before.txt.
"""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test"]
CLASSES = ["acne", "dry", "normal", "oily", "sensitive"]  # urutan alfabet
IMG_EXTS = {".jpg", ".jpeg", ".png"}
OUT = ROOT / "docs" / "dataset_snapshot_before.txt"

# --- Baseline evaluasi model terakhir (hardcode dari hasil yang diberikan) ---
BASELINE_ACC = "53.02%"
BASELINE_PER_CLASS = [
    # kelas, precision, recall, f1, support
    ("acne",      0.7059, 0.9600, 0.8136, 25),
    ("dry",       0.3953, 0.4857, 0.4359, 35),
    ("normal",    0.4250, 0.4857, 0.4533, 35),
    ("oily",      0.5625, 0.2571, 0.3529, 35),
    ("sensitive", 0.7500, 0.6316, 0.6857, 19),
]
BASELINE_CM = [
    [24,  0,  0, 0,  1],
    [1, 17, 13, 4,  0],
    [2, 11, 17, 3,  2],
    [2, 13, 10, 9,  1],
    [5,  2,  0, 0, 12],
]


def count_images(split, kelas):
    d = ROOT / "dataset" / split / kelas
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def build_report():
    lines = []
    w = lines.append

    w("=" * 70)
    w("SNAPSHOT DATASET — SEBELUM CLEANING MANUAL")
    w(f"Tanggal snapshot : {date.today().isoformat()}")
    w("=" * 70)
    w("")

    # ---- 1. Jumlah gambar per kelas per split ----
    w("1. JUMLAH GAMBAR AKTUAL PER KELAS PER SPLIT (.jpg/.jpeg/.png)")
    w("")
    header = f"{'kelas':<12}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'total':>10}"
    w(header)
    w("-" * len(header))

    grid = {kelas: {s: count_images(s, kelas) for s in SPLITS} for kelas in CLASSES}
    grand = 0
    for kelas in CLASSES:
        row_total = sum(grid[kelas][s] for s in SPLITS)
        grand += row_total
        w(f"{kelas:<12}" + "".join(f"{grid[kelas][s]:>10}" for s in SPLITS) + f"{row_total:>10}")

    w("-" * len(header))
    split_totals = {s: sum(grid[k][s] for k in CLASSES) for s in SPLITS}
    w(f"{'TOTAL':<12}" + "".join(f"{split_totals[s]:>10}" for s in SPLITS) + f"{grand:>10}")
    w("")
    w(f"Grand total keseluruhan: {grand} gambar")
    w("")

    # ---- 2. Baseline evaluasi model saat ini ----
    w("=" * 70)
    w("2. BASELINE EVALUASI MODEL SAAT INI (hasil terakhir)")
    w("=" * 70)
    w("")
    w(f"Akurasi keseluruhan : {BASELINE_ACC}")
    w("")
    w("Classification report (per kelas):")
    ch = f"{'kelas':<12}{'precision':>11}{'recall':>10}{'f1-score':>11}{'support':>10}"
    w(ch)
    w("-" * len(ch))
    for kelas, p, r, f1, sup in BASELINE_PER_CLASS:
        w(f"{kelas:<12}{p:>11.4f}{r:>10.4f}{f1:>11.4f}{sup:>10d}")
    w("")
    w("Confusion matrix (baris = label asli, kolom = prediksi):")
    w(f"  labels: {', '.join(CLASSES)}")
    colhdr = " " * 12 + "".join(f"{c[:8]:>10}" for c in CLASSES)
    w(colhdr)
    for i, kelas in enumerate(CLASSES):
        w(f"{kelas:<12}" + "".join(f"{BASELINE_CM[i][j]:>10d}" for j in range(len(CLASSES))))
    w("")
    w("=" * 70)

    return "\n".join(lines) + "\n"


def main():
    report = build_report()
    print(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"Tersimpan ke: {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

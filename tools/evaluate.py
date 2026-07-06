"""
evaluate.py — Evaluasi model kondisi kulit pada test set (dataset/test/).

- Load skin_model_finetuned.h5 (fallback: skin_model.h5).
- Baca dataset/test/ dengan preprocessing SAMA seperti train.py
  (target_size=224, rescale=1./255, categorical, batch=32) dan shuffle=False.
- Nama kelas diambil dari generator.class_indices (urutan model), BUKAN labels.txt.
- Hitung akurasi, classification_report, confusion_matrix.
- Output: cetak ke terminal + confusion_matrix.png + evaluate_report.txt.

Path relatif terhadap root project (parent dari folder tools/).
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def focal_loss(gamma=2.0):
    def _focal(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = K.pow(1.0 - y_pred, gamma) * y_true
        return K.sum(weight * cross_entropy, axis=-1)
    return _focal

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif (aman tanpa display)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path (root = parent dari folder tools/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "dataset" / "test"
MODEL_FINETUNED = ROOT / "skin_model_finetuned.h5"
MODEL_BASE = ROOT / "skin_model.h5"
CM_PNG = ROOT / "confusion_matrix.png"
REPORT_TXT = ROOT / "evaluate_report.txt"

IMG_SIZE = 224
BATCH_SIZE = 32


def pick_model_path():
    if MODEL_FINETUNED.exists():
        return MODEL_FINETUNED
    if MODEL_BASE.exists():
        print(f"[!] {MODEL_FINETUNED.name} tidak ada, fallback ke {MODEL_BASE.name}")
        return MODEL_BASE
    sys.exit(
        f"ERROR: tidak menemukan model. Jalankan train.py dulu "
        f"({MODEL_FINETUNED.name} / {MODEL_BASE.name} tidak ada)."
    )


def main():
    if not TEST_DIR.is_dir():
        sys.exit(f"ERROR: folder test tidak ada: {TEST_DIR}")

    model_path = pick_model_path()
    print(f"Memuat model: {model_path.name}")
    with custom_object_scope({'_focal': focal_loss()}):
        model = tf.keras.models.load_model(str(model_path))

    # Preprocessing SAMA seperti train.py untuk val (rescale saja, tanpa augmentasi).
    test_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_data = test_gen.flow_from_directory(
        str(TEST_DIR),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # WAJIB: urutan prediksi sejajar dengan label asli
    )

    # Urutan kelas dari generator (yang dipakai model), bukan labels.txt.
    class_indices = test_data.class_indices  # {nama: index}
    class_names = [name for name, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]
    print("Kelas (urutan model):", class_names)

    # Prediksi seluruh test set.
    y_true = test_data.classes
    probs = model.predict(test_data, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # Metrik.
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Cetak ke terminal.
    print(f"\n=== Akurasi keseluruhan: {acc:.4f} ({acc*100:.2f}%) ===\n")
    print("=== Classification Report ===")
    print(report)
    print("=== Confusion Matrix (baris=asli, kolom=prediksi) ===")
    print("labels:", class_names)
    print(cm)

    # Simpan classification_report ke file.
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path.name}\n")
        f.write(f"Akurasi keseluruhan: {acc:.4f} ({acc*100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix (baris=asli, kolom=prediksi):\n")
        f.write("labels: " + ", ".join(class_names) + "\n")
        f.write(np.array2string(cm) + "\n")
    print(f"\nReport disimpan: {REPORT_TXT.name}")

    # Heatmap confusion matrix -> confusion_matrix.png (seaborn jika ada).
    plt.figure(figsize=(7, 6))
    try:
        import seaborn as sns

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=True,
        )
    except ImportError:
        # Fallback murni matplotlib.
        ax = plt.gca()
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        thresh = cm.max() / 2 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.title(f"Confusion Matrix (acc={acc*100:.2f}%)")
    plt.ylabel("Label Asli")
    plt.xlabel("Prediksi")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150)
    plt.close()
    print(f"Heatmap disimpan: {CM_PNG.name}")


if __name__ == "__main__":
    main()

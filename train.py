import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

IMG_SIZE = 224

# Focal Loss: memberi bobot lebih besar pada sampel yang sulit diklasifikasi
# (probabilitas prediksi rendah untuk kelas benar). gamma=2.0 fokus pada hard examples,
# membantu kelas lemah (oily, dry) yang sering tertukar dengan kelas mirip.
# class_weight tetap aktif untuk menangani ketimpangan jumlah sampel.
def focal_loss(gamma=2.0):
    def _focal(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = K.pow(1.0 - y_pred, gamma) * y_true
        return K.sum(weight * cross_entropy, axis=-1)
    return _focal
BATCH_SIZE = 32
# Fase 1 dinaikkan 15 -> 20 epoch: beri model lebih banyak kesempatan belajar fitur
# sebelum fine-tuning (terutama membantu kelas lemah seperti oily).
EPOCHS = 20

# Augmentasi train diperkuat: variasi brightness, geser posisi, & shear membuat model
# lebih robust terhadap perbedaan pencahayaan/framing antar sumber data dan mengurangi
# overfitting (rotation/zoom/flip lama tetap dipertahankan).
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Kelas tidak seimbang (sensitive jauh lebih sedikit dari normal/dry/oily) ->
# hitung class_weight 'balanced' dari jumlah train tiap kelas (urutan = class_indices model).
counts = np.bincount(train_data.classes, minlength=train_data.num_classes)
total = counts.sum()
class_weight = {
    i: float(total / (train_data.num_classes * c)) if c else 0.0
    for i, c in enumerate(counts)
}
print("class_indices:", train_data.class_indices)
print("train counts :", counts.tolist())
print("class_weight :", class_weight)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    # Dropout diturunkan 0.5 -> 0.3: train acc ~70% vs val ~56% menunjukkan
    # Dropout terlalu agresif memotong informasi (underfitting di validasi).
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss=focal_loss(),
    metrics=["accuracy"]
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
# Checkpoint GLOBAL: instance SAMA dipakai di KEDUA fase. Karena ModelCheckpoint
# menyimpan 'self.best' di __init__ dan on_train_begin TIDAK mereset-nya, nilai best
# bertahan lintas fase. Jadi skin_model_finetuned.h5 HANYA ditimpa bila ada epoch
# (di fase mana pun) dengan val_accuracy yang benar-benar lebih tinggi -> hasil akhir =
# model TERBAIK GLOBAL. Kalau fine-tuning tidak pernah lebih baik, file tetap berisi
# model terbaik fase 1 (fine-tune yang merugikan otomatis diabaikan).
ckpt_global = ModelCheckpoint(
    "skin_model_finetuned.h5",
    monitor="val_accuracy", mode="max",
    save_best_only=True, verbose=1,
)
# Checkpoint khusus fase 1 -> model TERBAIK selama fase 1 saja.
ckpt_phase1 = ModelCheckpoint(
    "skin_model.h5",
    monitor="val_accuracy", mode="max",
    save_best_only=True, verbose=1,
)
# EarlyStopping fase 1: hentikan bila val_accuracy tak membaik 5 epoch, pulihkan bobot terbaik.
early_phase1 = EarlyStopping(
    monitor="val_accuracy", mode="max",
    patience=5, restore_best_weights=True, verbose=1,
)
# EarlyStopping fase 2: fine-tuning kini lebih agresif (lr 1e-4, 10 epoch) sehingga rawan
# overfit -> hentikan & pulihkan bobot terbaik bila val_accuracy tak membaik 5 epoch.
early_phase2 = EarlyStopping(
    monitor="val_accuracy", mode="max",
    patience=5, restore_best_weights=True, verbose=1,
)

# ---- Fase 1: feature extraction (base di-freeze) ----
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[ckpt_phase1, ckpt_global, early_phase1],
)

# ===========================================================
# ---- Fase 2: fine-tuning (unfreeze 50 layer terakhir) ----
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

# Fine-tuning lebih agresif: lr dinaikkan 1e-5 -> 1e-4 agar 50 layer terakhir benar-benar
# beradaptasi ke domain kulit wajah (lr 1e-5 sebelumnya terbukti nyaris tak menggeser val acc).
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=["accuracy"]
)

# Fase 2 dinaikkan 5 -> 10 epoch (diiringi EarlyStopping agar berhenti sebelum overfit).
# ckpt_global = instance SAMA -> 'best' fase 1 tetap dipertahankan.
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weight,
    callbacks=[ckpt_global, early_phase2],
)

# ---------------------------------------------------------------------------
# Ringkasan: dari fase/epoch mana model akhir berasal
# ---------------------------------------------------------------------------
v1 = history1.history.get("val_accuracy", [])
v2 = history2.history.get("val_accuracy", [])
best_p1 = max(v1) if v1 else float("nan")
ep_p1 = (v1.index(best_p1) + 1) if v1 else 0
best_p2 = max(v2) if v2 else None
ep_p2 = (v2.index(best_p2) + 1) if v2 else 0

# Checkpoint pakai np.greater (strict) -> fase 2 harus MELAMPAUI best fase 1 utk menimpa.
if best_p2 is not None and best_p2 > best_p1:
    best_global = best_p2
    asal = f"Fase 2 (fine-tune), epoch {ep_p2}"
else:
    best_global = best_p1
    asal = f"Fase 1, epoch {ep_p1}"

print("\n==================== RINGKASAN TRAINING ====================")
print(f"Val accuracy terbaik Fase 1             : {best_p1:.4f} (epoch {ep_p1})")
if best_p2 is not None:
    print(f"Val accuracy terbaik Fase 2 (fine-tune) : {best_p2:.4f} (epoch {ep_p2})")
else:
    print("Val accuracy terbaik Fase 2 (fine-tune) : - (fase 2 tidak berjalan)")
print(f"Val accuracy terbaik GLOBAL             : {best_global:.4f}")
print(f"Model akhir (skin_model_finetuned.h5) berasal dari: {asal}")
print("  - skin_model.h5           = model terbaik Fase 1")
print("  - skin_model_finetuned.h5 = model terbaik GLOBAL (dipakai evaluate.py & export_tflite.py)")
print("============================================================")

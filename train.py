import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
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
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ckpt_global = instance SAMA -> 'best' fase 1 tetap dipertahankan.
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    class_weight=class_weight,
    callbacks=[ckpt_global],
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

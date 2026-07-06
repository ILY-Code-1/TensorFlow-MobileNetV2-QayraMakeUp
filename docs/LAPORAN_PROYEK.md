# Laporan Proyek — Klasifikasi Kondisi Kulit Wajah (MobileNetV2 → TFLite)

**Proyek:** QayraMakeUp — TensorFlow MobileNetV2
**Tanggal:** 7 Juli 2026

---

## a. Tujuan Proyek

Membangun model klasifikasi **kondisi kulit wajah** ke dalam 5 kategori — **acne, dry,
normal, oily, sensitive** — menggunakan arsitektur **MobileNetV2** dengan *transfer
learning*, lalu mengekspornya ke **TensorFlow Lite (`skin_condition.tflite`)** agar ringan
dan dapat dijalankan di perangkat mobile/edge untuk aplikasi **QayraMakeUp**.

Model menerima input gambar wajah **224×224 piksel RGB** (dinormalkan ke rentang `[0,1]`)
dan menghasilkan probabilitas *softmax* untuk 5 kelas.

---

## b. Dataset & Sumbernya

Dataset dibangun secara bertahap dari berbagai sumber Kaggle dan Roboflow. Kelas
`sensitive` tidak memiliki dataset visual "kulit sensitif" yang valid, sehingga
digunakan **proxy** berbasis kemerahan kulit:

| Kelas | Sumber | Keterangan |
|---|---|---|
| oily, dry, normal | Kaggle — *Oily-Dry-and-Normal-Skin-Types* + Roboflow Universe | dataset wajah penuh |
| acne | Kaggle — *Acne Dataset* + Roboflow Universe | banyak *close-up* kulit |
| **sensitive** | **proxy:** *redness* (TrainingDataPro) **+** *rosacea* (Roboflow) | tidak ada dataset "sensitif" langsung |

### Penambahan dataset secara bertahap

- **Batch 1 (V6):** penambahan dari Kaggle/Roboflow untuk seluruh 5 kelas → total
  ~5.440 gambar. Akurasi melonjak dari 60,58% → **71,22%** (+10,64 poin).
- **Batch 2 (V7–V8):** penambahan dari Kaggel batch kedua + Roboflow untuk kelas
  dry, normal, dan oily → total **8.917 gambar**. Akurasi meningkat dari
  71,22% → **74,39%** → **77,17%** setelah optimalisasi hyperparameter.

### Tantangan kelas `sensitive` dan penanganannya

- **Masalah awal:** sumber *redness* hanya berisi **30 foto dari 10 orang** (3 angle per
  orang). Jumlah dan keragaman ini terlalu kecil dan sangat rawan **kebocoran data
  (*leakage*)** bila foto orang yang sama tersebar antar split.
- **Penambahan sumber:** ditambahkan kelas **rosacea** dari dataset Roboflow
  *facial-skin-diseases* (kelas lain seperti acne/eksim/herpes/panu **diabaikan**).
- **Split *group-aware* (anti-leakage):** pembagian train/val/test dilakukan **per grup**
  sehingga semua gambar dari satu subjek/satu gambar asli (termasuk augmentasinya)
  **selalu berada di split yang sama**.
- **Augmentasi hanya di train:** bagian *train* `sensitive` diaugmentasi hingga genap,
  sedangkan **val & test memakai gambar asli saja** (tidak diaugmentasi) agar evaluasi
  jujur.

---

## c. Pipeline Kurasi Data

Script: [`tools/prepare_dataset.py`](../tools/prepare_dataset.py)

1. **Kumpulkan pool** per kelas dari sumber mentah — Kaggle dan Roboflow Universe.
2. **Deteksi wajah** — detektor OpenCV DNN res10 SSD. Gambar tanpa wajah terdeteksi
   tetap dibuang (dihitung).
3. **Crop wajah → resize 224×224**, normalisasi semua gambar ke `.jpg`.
4. **Dedup perceptual-hash** untuk menghilangkan gambar duplikat.
5. **Pembersihan manual** — menghapus gambar non-face, screenshot, stock photo, dan
   gambar dengan kategori salah (mis. eczema di folder sensitive).

---

## d. Pembagian Akhir Dataset (70 / 20 / 10)

Dataset final (V8) setelah seluruh batch digabungkan dan dibersihkan:

| Kelas | Train | Val | Test | Total |
|---|---|---|---|---|
| acne | 565 | 162 | 77 | 804 |
| dry | 1.445 | 417 | 208 | 2.070 |
| normal | 2.029 | 580 | 296 | 2.905 |
| oily | 1.785 | 511 | 257 | 2.553 |
| sensitive | 405 | 115 | 65 | 585 |
| **TOTAL** | **6.229** | **1.785** | **903** | **8.917** |

Total: **6.229 train / 1.785 val / 903 test** (8.917 gambar). **Ketimpangan antar kelas:**
`normal` paling banyak (**2.905**), `sensitive` paling sedikit (**585** — 5× lebih sedikit
dari normal). Ditangani dengan **`class_weight` dinamis** yang di-*clamp* pada rentang
**[0.8, 1.5]** untuk mencegah bobot terlalu ekstrem yang dapat mendistorsi pembelajaran.

---

## e. Training

**Arsitektur:** MobileNetV2 (bobot ImageNet, `include_top=False`, *base di-freeze*) + head:
`GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(5, softmax)`.

**Hyperparameter & pipeline (V8 — final):**

- **Fase 1 (feature extraction):** 30 epoch, optimizer **Adam (lr default)**, *base* dibekukan.
  `EarlyStopping` patience=7 (`restore_best_weights=True`). Loss: **Focal Loss (gamma=2.0)**.
  Augmentasi train: rotasi 20°, zoom 0.2, horizontal flip, brightness [0.8–1.2], width/height
  shift 0.1, shear 0.1.
- **Fase 2 (fine-tuning):** *unfreeze* 50 layer terakhir, Adam **lr=5e-5**, maks 20 epoch,
  `EarlyStopping` patience=5. Loss: **Focal Loss (gamma=2.0)**. **Terbukti efektif** —
  fine-tuning berhasil meningkatkan val accuracy dari 65,00% → **75,56%** (+10,56 poin)
  dan test accuracy menjadi **77,17%**.
- **`ModelCheckpoint(save_best_only=True)`** dipakai di **kedua fase dengan instance yang sama**
  → `skin_model_finetuned.h5` hanya ditimpa bila ada epoch yang **benar-benar lebih baik**,
  sehingga melindungi hasil dari fase mana pun.
- **`class_weight`** dihitung **dinamis** dari jumlah train aktual (*balanced*), kemudian
  di-*clamp* ke rentang **[0.8, 1.5]**.
- **Dropout 0.3** — diturunkan dari 0.5 untuk mengurangi underfitting.
- **Focal Loss (gamma=2.0)** untuk menangani ketidakseimbangan kelas.

**Ringkasan hasil per fase (V8):**

| Fase | Train acc | Val acc |
|---|---|---|
| Fase 1 — awal (ep1) | ~0.40 | ~0.43 |
| Fase 1 — terbaik (ep27) | ~0.63 | **~0.65** ← model fase 1 terbaik |
| Fase 1 — EarlyStopping | ep27 | — |
| Fase 2 — terbaik (ep19) | ~0.82 | **~0.76** ← model final |
| Fase 2 — EarlyStopping | ep20 | — |

**Temuan utama:**

- **Fine-tuning kini efektif** — berbeda dengan V5 di mana fine-tuning selalu menurunkan
  performa. Dengan dataset 8.917 gambar dan lr=5e-5, fine-tuning memberikan peningkatan
  val accuracy dari 65% → 76% dan test accuracy dari 65,48% (sebelum optimalisasi)
  menjadi **77,17%**.
- **`class_weight` clamp [0.8–1.5] penting** — tanpa clamping, model dengan dataset tidak
  seimbang mengalami distorsi: terlalu takut salah prediksi kelas minoritas dan terlalu
  abai pada kelas mayoritas, menjatuhkan akurasi ke 65,48%.
- **Epoch lebih panjang + patience lebih tinggi membantu** — Fase 1 30 epoch dengan
  patience=7 memberi model cukup waktu untuk konvergensi penuh sebelum masuk fine-tuning.
- **ReduceLROnPlateau tidak membantu** — menurunkan akurasi dari 74,39% → 73,27%.
- **Model final** (`skin_model_finetuned.h5`) berasal dari **Fase 2 epoch 19**,
  val accuracy **75,56%** pada data validasi — dan **77,17%** pada test set.

---

## f. Evaluasi di Test Set

Script: [`tools/evaluate.py`](../tools/evaluate.py) — 898 gambar test, `shuffle=False`,
tanpa augmentasi. Hasil **final (V8)**: dataset 8.917 gambar dengan fine-tuning optimal.

**Akurasi keseluruhan: 77,17%.**

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| acne | 0.89 | 0.82 | **0.85** | 76 |
| dry | 0.67 | 0.85 | 0.75 | 207 |
| normal | 0.81 | 0.78 | **0.80** | 295 |
| oily | 0.77 | 0.66 | **0.71** | 256 |
| sensitive | 0.88 | 0.89 | **0.88** | 64 |
| **accuracy** | – | – | **0.77** | 898 |
| macro avg | 0.80 | 0.80 | 0.80 | 898 |
| weighted avg | 0.78 | 0.77 | 0.77 | 898 |

**Model akhir berasal dari Fase 2 (fine-tune) epoch 19.** Fine-tuning kini terbukti
efektif setelah dataset mencapai ukuran yang memadai (5.440+ gambar).

**Confusion Matrix** (baris = label asli, kolom = prediksi):

![Confusion Matrix](../confusion_matrix.png)

Matriks (label urut: acne, dry, normal, oily, sensitive):

```
[[ 62   3   1   3   7]
 [  1 175  16  15   0]
 [  1  33 230  31   0]
 [  5  45  36 169   1]
 [  1   5   0   1  57]]
```

---

## g. Analisis

- **Kelas terkuat & konsisten:** `sensitive` (F1 **0.88**, recall 89%, precision 88%)
  dan `acne` (F1 **0.85**, precision 89%). Keduanya kuat di semua iterasi karena ciri
  visual yang khas (kemerahan; tekstur jerawat).
- **Peningkatan paling dramatis — `oily`:** recall naik dari **20%** (V5) → **56%**
  (V7) → **66%** (V8). Kenaikan +46 poin recall, dengan hanya 45 dari 256 sampel oily
  yang kini salah diprediksi sebagai dry (sebelumnya 65 di V7). F1-score naik dari
  0,30 → 0,66 → **0,71**.
- **Sumber error utama — trio dry/normal/oily masih tumpang-tindih:**
  pada confusion matrix V8, normal banyak bocor ke oily (31) dan dry (33); oily bocor
  ke dry (45) dan normal (36). Tumpang-tindih ini adalah **batas alami visual**.
- **Semua kelas kini di atas F1 0,70** — tidak ada lagi kelas "sangat lemah" seperti
  di iterasi awal. Ini menunjukkan model sudah mencapai keseimbangan yang baik.
- **Fine-tuning kini efektif.** Berbeda dengan dataset kecil (V5, 2.084 gambar) di mana
  fine-tuning selalu menurunkan performa, pada dataset besar (8.917 gambar) fine-tuning
  dengan lr=5e-5 berhasil memberikan peningkatan +2,78 poin (74,39% → 77,17%).
- **Cleaning manual terlalu agresif sempat merusak hasil (V3).** Pembersihan menghapus
  ~70% data `acne`, menjatuhkan akurasi ke **46,72%**. Dipulihkan dengan menambah data
  baru dari tahap *staging* di V4.
- **Penambahan dataset adalah strategi paling efektif.** Penambahan dari 2.084 → 5.440
  gambar (V5→V6) menghasilkan lonjakan akurasi +10,64 poin (60,58% → 71,22%) — peningkatan
  terbesar sepanjang iterasi.
- **Augmentasi lebih kaya + epoch lebih panjang + fine-tuning optimal** berkontribusi
  pada hasil terbaik **77,17%** (V8).

---

## h. Keterbatasan

1. **Test set tetap berisik** (898 gambar) — meskipun 4× lebih besar dari V5 (208),
   selisih beberapa gambar masih dapat menggeser persentase, terutama untuk kelas
   minoritas (sensitive: 65 test).
2. **Bias sumber pada `sensitive`** — kelas ini berasal dari sumber berbeda (rosacea +
   redness), sehingga model bisa saja belajar **ciri sumber/dataset**, bukan murni
   "sensitivitas kulit".
3. **Kelas `oily` tetap yang paling menantang** (recall **66%**) — kemiripan visual
   dengan dry dan normal masih menjadi sumber error utama; kilap minyak sulit
   terdeteksi dari foto pencahayaan biasa.
4. **Dataset tidak seimbang** — normal (2.905) 5× lebih banyak dari sensitive (585).
   Clamp class_weight [0.8–1.5] membantu namun bukan solusi optimal.
5. **Fine-tuning hanya efektif pada dataset besar** — pada dataset kecil (<3.000 gambar),
   fine-tuning konsisten menurunkan performa. Temuan ini penting untuk penelitian
   serupa di masa depan.
6. **Arsitektur MobileNetV2 mulai mendekati batas** — peningkatan lebih lanjut
   kemungkinan membutuhkan arsitektur yang lebih canggih (EfficientNet, ViT).

---

## i. Rekomendasi Perbaikan

1. **Pertahankan strategi fine-tuning saat ini** — lr=5e-5, 20 epoch, unfreeze 50 layer
   terakhir — yang terbukti efektif pada dataset besar.
2. **Prioritaskan penambahan data `oily`** dengan kilap wajah yang jelas terlihat untuk
   menaikkan recall yang masih 66%.
3. **Tambah data `sensitive` asli** (bukan proxy redness/rosacea) untuk mengurangi bias
   sumber dan meningkatkan representasi.
4. **Pertimbangkan arsitektur yang lebih canggih** — EfficientNetV2, ConvNeXt, atau
   Vision Transformer — untuk melampaui batas MobileNetV2.
5. **Implementasikan cross-validation** (k-fold) untuk estimasi performa yang lebih
   stabil dan mengurangi ketergantungan pada satu split test.
6. **Eksplorasi ensemble model** — menggabungkan beberapa model dengan arsitektur
   berbeda dapat menaikkan akurasi 2–5 poin.

---

## j. Riwayat Iterasi

| Versi | Dataset | Akurasi | Catatan singkat |
|---|---|---|---|
| **V1** | 1.000 foto (5 kelas) | 58.59% | Baseline awal |
| **V2** | +killa92 (1.498 foto) | 53.02% | Test set lebih berat & beragam |
| **V3** | +cleaning manual (1.214 foto) | 46.72% | Cleaning terlalu agresif di `acne` |
| **V4** | +staging baru (2.084 foto) | 58.65% | Augmentasi lebih kaya |
| **V5** | 2.084 foto + Dropout 0.3 + Focal Loss | 60.58% | Dropout 0.3 + Focal Loss |
| **V6** | +Kaggle Batch 1 (~5.440 foto) | 71.22% | Lompatan +10.64 poin |
| **V7** | +Kaggle Batch 2 (8.917 foto) | 74.39% | +clamp class weight [0.8–1.5] |
| **V8** | 8.917 foto + fine-tune optimal | **77.17%** | **Terbaik** — lr 5e-5 + 20 ep FT |

Pelajaran utama: **menambah data** adalah strategi paling efektif (+10.64 poin,
V5→V6); **class weight clamping** menstabilkan training pada dataset tidak seimbang;
**fine-tuning hanya efektif setelah dataset cukup besar** — pada dataset kecil,
fine-tuning justru kontraproduktif.

---

*Laporan ini dibuat berdasarkan `training_log.txt`, `evaluate_report.txt`,
`confusion_matrix.png`, `labels.txt`, dan script di folder `tools/`. Angka diambil langsung
dari artefak hasil training/evaluasi.*

# Laporan Proyek — Klasifikasi Kondisi Kulit Wajah (MobileNetV2 → TFLite)

**Proyek:** QayraMakeUp — TensorFlow MobileNetV2
**Tanggal:** 3 Juni 2026

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

Empat kelas memiliki sumber visual yang jelas, namun **kelas `sensitive` tidak memiliki
dataset visual "kulit sensitif" yang valid**, sehingga digunakan **proxy** berbasis
kemerahan kulit:

| Kelas | Sumber | Keterangan |
|---|---|---|
| oily, dry, normal | Kaggle — *Oily-Dry-and-Normal-Skin-Types* | dataset wajah penuh |
| acne | Kaggle — *Acne Dataset* | banyak *close-up* kulit |
| **sensitive** | **proxy:** *redness* (TrainingDataPro) **+** *rosacea* (Roboflow) | tidak ada dataset "sensitif" langsung |

### Tantangan kelas `sensitive` dan penanganannya

- **Masalah awal:** sumber *redness* hanya berisi **30 foto dari 10 orang** (3 angle per
  orang). Jumlah dan keragaman ini terlalu kecil dan sangat rawan **kebocoran data
  (*leakage*)** bila foto orang yang sama tersebar antar split.
- **Penambahan sumber:** ditambahkan kelas **rosacea** dari dataset Roboflow
  *facial-skin-diseases* (kelas lain seperti acne/eksim/herpes/panu **diabaikan**).
  Setelah kurasi, kelas `sensitive` menjadi **192 gambar bersih** yang berasal dari
  **±89 grup unik** (gabungan rosacea, redness, dan data kurasi lama).
- **Split *group-aware* (anti-leakage):** pembagian train/val/test dilakukan **per grup**
  menggunakan `group_id`, sehingga semua gambar dari satu subjek/satu gambar asli
  (termasuk augmentasinya) **selalu berada di split yang sama**.
- **Augmentasi hanya di train:** bagian *train* `sensitive` diaugmentasi hingga genap,
  sedangkan **val & test memakai gambar asli saja** (tidak diaugmentasi) agar evaluasi
  jujur.
- **Augmentasi menjaga sinyal warna:** karena kemerahan adalah **fitur warna**, augmentasi
  dibatasi pada transformasi **geometric** (flip horizontal, rotasi kecil, zoom/crop
  ringan, translasi) **+ brightness ringan**, **TANPA mengubah hue/saturation** agar
  sinyal kemerahan yang justru ingin dipelajari tidak rusak.

---

## c. Pipeline Kurasi Data

Script: [`tools/prepare_dataset.py`](../tools/prepare_dataset.py)

1. **Kumpulkan pool** per kelas dari sumber mentah + **salin** 100 gambar kurasi manual
   lama (copy, bukan dipindah).
2. **Deteksi wajah** — detektor diganti dari **Haar Cascade → OpenCV DNN res10 SSD** yang
   jauh lebih kuat. Dampaknya signifikan: jumlah *no-face* yang salah dibuang turun drastis
   (mis. `oily` dari 101 → 10). Gambar tanpa wajah terdeteksi tetap dibuang (dihitung).
3. **Crop wajah → resize 224×224**, normalisasi semua gambar ke `.jpg`.
4. **Dedup perceptual-hash** hanya untuk **4 kelas besar** (oily/dry/normal/acne);
   kelas `sensitive` **tidak** di-dedup agar augmentasi & variasi angle dipertahankan.
5. **Manifest** `dataset/_clean/manifest.csv` (`path, kelas, group_id, sumber`) sebagai
   dasar split *group-aware*.

Hasil kurasi (pool bersih): oily **350**, dry **369**, normal **376**, acne **250**,
sensitive **192** (≈89 grup unik).

---

## d. Pembagian Akhir Dataset (70 / 20 / 10)

Script: [`tools/split_dataset.py`](../tools/split_dataset.py) — kumpulkan acak 200 per kelas
besar, lalu split **140 train / 40 val / 20 test**.

| Kelas | Train | Val | Test | Total |
|---|---|---|---|---|
| acne | 140 | 40 | 20 | 200 |
| dry | 140 | 40 | 20 | 200 |
| normal | 140 | 40 | 20 | 200 |
| oily | 140 | 40 | 20 | 200 |
| **sensitive** | **140** (134 asli + **6 aug**) | **39** (asli) | **19** (asli) | 198 |

Total: **700 train / 199 val / 99 test**. Split `sensitive` terbukti **bebas leakage**
(irisan grup antar split = 0).

**Perbaikan `labels.txt`:** file ditulis ulang dalam **urutan alfabet**
(`acne, dry, normal, oily, sensitive`) agar **cocok dengan urutan baca
`flow_from_directory`**. Ini sekaligus **menutup bug** urutan lama di mana `oily` dan `dry`
tertukar.

---

## e. Training

**Arsitektur:** MobileNetV2 (bobot ImageNet, `include_top=False`, *base di-freeze*) + head:
`GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(5, softmax)`.

- **Fase 1 (feature extraction):** 15 epoch, optimizer Adam, *base* dibekukan.
- **Fase 2 (fine-tuning):** *unfreeze* 50 layer terakhir, learning rate **1e-5**, 5 epoch.

**Ringkasan hasil per fase:**

| Fase | Train acc | Val acc |
|---|---|---|
| Fase 1 — awal (ep1) | 0.32 | 0.46 |
| Fase 1 — akhir (ep15) | **0.697 (~70%)** | **0.578 (~57.8%)** |
| Fase 2 — akhir (ep20) | 0.56 | 0.578 |

**Temuan:** akurasi *train* naik hingga **~70%** sementara akurasi *validasi* **mentok di
kisaran 55–58%** (puncak ~57.8%) = mulai terjadi **overfitting ringan**. **Fine-tuning
TIDAK menaikkan akurasi validasi** (tetap di ~57–57.8% sepanjang fase 2).

---

## f. Evaluasi di Test Set

Script: [`tools/evaluate.py`](../tools/evaluate.py) — 99 gambar test, `shuffle=False`,
tanpa augmentasi.

**Akurasi keseluruhan: 58.59%.**

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| acne | 0.71 | 0.85 | **0.77** | 20 |
| dry | 0.44 | 0.70 | 0.54 | 20 |
| normal | 0.63 | 0.25 | **0.36** | 20 |
| oily | 0.48 | 0.55 | 0.51 | 20 |
| sensitive | 0.92 | 0.58 | **0.71** | 19 |
| **accuracy** | – | – | **0.59** | 99 |
| macro avg | 0.63 | 0.59 | 0.58 | 99 |
| weighted avg | 0.63 | 0.59 | 0.58 | 99 |

**Confusion Matrix** (baris = label asli, kolom = prediksi):

![Confusion Matrix](confusion_matrix.png)

---

## g. Analisis

- **Kelas terkuat:** `acne` (F1 **0.77**) dan `sensitive` (F1 **0.71**). Keduanya punya ciri
  visual yang relatif khas (tekstur jerawat; kemerahan), sehingga lebih mudah dipisahkan.
- **Sumber error utama:** trio **dry / normal / oily** saling **tumpang-tindih**. Pada
  confusion matrix terlihat `dry` 14 benar tapi 5 bocor ke `oily`; `oily` 11 benar dengan
  7 bocor ke `dry`.
- **`normal` paling lemah** (recall **0.25**) — dari 20 sampel hanya 5 benar; sering
  tertukar menjadi `dry` (7) atau `oily` (6).
- **`dry` menjadi "keranjang" over-prediksi** — banyak kelas lain salah diprediksi sebagai
  `dry` (precision rendah 0.44 meski recall tinggi 0.70).
- Hal ini **wajar**: perbedaan visual antara kulit *dry*, *normal*, dan *oily* memang
  **halus** dan sangat bergantung pencahayaan/kualitas foto.

---

## h. Keterbatasan

1. **Test set kecil** (~20 gambar/kelas, total 99) → angka metrik **berisik**; selisih
   beberapa gambar dapat menggeser persentase cukup besar.
2. **Bias sumber pada `sensitive`** — kelas ini berasal dari sumber berbeda (rosacea +
   redness), sehingga model bisa saja belajar **ciri sumber/dataset**, bukan murni
   "sensitivitas kulit". **Uji sebenarnya** adalah foto wajah baru di luar dataset.
3. **Fine-tuning belum efektif** — pada lr 1e-5, fase 2 tidak meningkatkan akurasi validasi.
4. Model yang disimpan adalah **epoch terakhir**, bukan epoch dengan validasi terbaik.

---

## i. Rekomendasi Perbaikan

1. **Perbaiki data dry/normal/oily** (terutama `normal`): tambah jumlah, perbaiki kualitas
   & konsistensi pencahayaan, kurangi ambiguitas label.
2. **Fine-tuning lebih agresif & cerdas:** coba lr **1e-4**, dan gunakan
   `ModelCheckpoint`/`EarlyStopping` untuk **menyimpan model dengan val accuracy terbaik**
   (bukan epoch terakhir).
3. **Kumpulkan foto kulit sensitif asli** untuk menggantikan proxy redness/rosacea, agar
   kelas `sensitive` benar-benar merepresentasikan target aplikasi.
4. **Pertimbangkan konsolidasi kelas** (mis. menggabungkan kategori yang ambigu) bila sesuai
   dengan kebutuhan produk QayraMakeUp.

---

*Laporan ini dibuat berdasarkan `training_log.txt`, `evaluate_report.txt`,
`confusion_matrix.png`, `labels.txt`, dan script di folder `tools/`. Angka diambil langsung
dari artefak hasil training/evaluasi.*

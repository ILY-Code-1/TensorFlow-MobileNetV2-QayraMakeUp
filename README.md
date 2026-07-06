# TensorFlow MobileNetV2 - Klasifikasi Kondisi Kulit Wajah

Project machine learning untuk klasifikasi kondisi kulit wajah menggunakan TensorFlow dan arsitektur MobileNetV2 dengan teknik transfer learning.

> 📄 **Detail lengkap Training & Evaluasi → [docs/LAPORAN_PROYEK.pdf](docs/LAPORAN_PROYEK.pdf)** — mencakup sumber dataset, pipeline kurasi, strategi anti-leakage kelas `sensitive`, hasil training, confusion matrix, analisis, riwayat iterasi, dan rekomendasi. (Sumber markdown: [docs/LAPORAN_PROYEK.md](docs/LAPORAN_PROYEK.md))

> 🤔 **Mengapa 77,17% sudah cukup? → [REASON.md](REASON.md)**
> — penjelasan sederhana mengapa hasil ini solid dan valid untuk aplikasi QayraMakeUp.

## 📋 Deskripsi

Project ini mengimplementasikan model deep learning untuk mengklasifikasikan kondisi kulit wajah menjadi 5 kategori berbeda. Model dilatih menggunakan teknik transfer learning dari MobileNetV2 yang telah pretrained pada dataset ImageNet, kemudian di-fine-tune untuk task spesifik klasifikasi kulit wajah.

## 🎯 Tujuan Model

Mengklasifikasikan kondisi kulit wajah ke dalam 5 kategori:
- **Acne** - Kulit berjerawat
- **Oily** - Kulit berminyak
- **Dry** - Kulit kering
- **Normal** - Kulit normal
- **Sensitive** - Kulit sensitif

## 📊 Hasil Training & Evaluasi

**Akurasi terbaik: 77,17%** (V8 — dataset **8.917 gambar**, 5 kelas, test set 898 gambar, fine-tuning lr 5e-5).

| Kelas | Precision | Recall | F1-score |
|---|---|---|---|
| acne | 0,89 | 0,82 | **0,85** |
| dry | 0,67 | 0,85 | 0,75 |
| normal | 0,81 | 0,78 | **0,80** |
| oily | 0,77 | 0,66 | **0,71** |
| sensitive | 0,88 | 0,89 | **0,88** |

Semua kelas kini di atas F1 0,70 — tidak ada lagi kelas "sangat lemah" seperti
iterasi awal. Oily (F1 0,71, recall 66%) dan dry (F1 0,75, recall 85%)
mengalami perbaikan paling signifikan.

> ⚙️ **Catatan:** Fine-tuning dengan learning rate 5e-5 (20 epoch) terbukti
> efektif — model terbaik berasal dari fase 2 (fine-tune). Class weight
> di-clamp [0.8–1.5] untuk menstabilkan training. EPOCHS fase 1 dinaikkan
> ke 30 dengan EarlyStopping patience=7.

Pembahasan lengkap, confusion matrix, keterbatasan, dan rekomendasi ada di
**[docs/LAPORAN_PROYEK.pdf](docs/LAPORAN_PROYEK.pdf)**.

Penjelasan mengapa akurasi ini valid dan cukup ada di **[REASON.md](REASON.md)**.

## 🔁 Riwayat Iterasi Dataset

| Versi | Jumlah Data | Akurasi | Catatan |
|---|---|---|---|
| V1 | 1.000 foto | 58,59% | Baseline awal |
| V2 | 1.498 foto | 53,02% | +killa92 — test set lebih berat & beragam |
| V3 | 1.214 foto | 46,72% | +cleaning manual — terlalu agresif di `acne` |
| V4 | 2.084 foto | 58,65% | +staging baru — augmentasi lebih kaya |
| V5 | 2.084 foto | 60,58% | Dropout 0.3 + Focal Loss |
| V6 | ~5.440 foto | 71,22% | +Kaggle Batch 1 — lompatan terbesar (+10,64) |
| V7 | 8.917 foto | 74,39% | +Kaggle Batch 2 + cap class weight |
| **V8** | **8.917 foto** | **77,17%** | **Fine-tuning lr 5e-5 + 20 epoch (terbaik)** |

## 🛠️ Tech Stack

- **TensorFlow** - Framework deep learning utama
- **Keras** - High-level neural networks API
- **Python** - Bahasa pemrograman
- **MobileNetV2** - Arsitektur model pretrained

## 🏗️ Arsitektur Model

### MobileNetV2 (Pretrained)
- Menggunakan bobot yang telah dilatih pada ImageNet
- Input size: 224x224 pixels
- Feature extractor yang efisien dan ringan

### Transfer Learning
- Base model MobileNetV2 di-freeze pada tahap awal
- Menambahkan custom layers di atasnya:
  - Global Average Pooling 2D
  - Dense layer (128 neurons, ReLU activation)
  - Dropout (0.3) untuk mencegah overfitting
  - Output layer (5 neurons, Softmax activation)

### Fine-Tuning
- Unfreeze 50 layer terakhir dari base model
- Training tambahan dengan learning rate 5e-5
- Terbukti efektif setelah dataset cukup besar (5.440+ gambar)

## 🔄 Workflow

### 1. Data Loading
- Memuat gambar dari direktori train dan validation
- Resize semua gambar ke 224x224 pixels
- Batch size: 32

### 2. Data Augmentation
- **Rotation**: ±20 derajat
- **Zoom**: 20%
- **Horizontal Flip**: Random flip
- **Brightness**: [0.8–1.2]
- **Width/Height Shift**: 10%
- **Shear**: 10%
- **Rescaling**: Normalisasi pixel ke [0, 1]

### 3. Training — Fase 1 (Initial)
- Epochs: maks **30** (EarlyStopping patience=7)
- Optimizer: Adam (lr default)
- Loss: Focal Loss (gamma=2.0)
- Base model dalam kondisi frozen
- Augmentasi: rotasi ±20°, zoom 20%, horizontal flip, brightness [0.8–1.2], shift 10%, shear 10%

### 4. Training — Fase 2 (Fine-Tuning)
- Epochs: maks **20** (EarlyStopping patience=5)
- Learning rate: **5e-5**
- Loss: Focal Loss (gamma=2.0)
- 50 layer terakhir base model di-unfreeze
- **Catatan**: terbukti efektif memberikan peningkatan +2,78 poin

### 5. Export ke TFLite
- Convert model Keras ke TensorFlow Lite format
- Optimasi: DEFAULT optimization
- Output: skin_condition.tflite

## 📁 Struktur Folder Project

```
TensorFlow-MobileNetV2-QayraMakeUp/
│
├── dataset/
│   ├── train/          # Data training (70%)
│   ├── val/            # Data validation (20%)
│   ├── test/           # Data testing (10%)
│   └── _staging/       # Folder staging data baru (di-ignore git)
│
├── docs/
│   ├── LAPORAN_PROYEK.md   # Sumber laporan
│   └── LAPORAN_PROYEK.pdf  # Laporan lengkap training & evaluasi
│
├── tools/
│   ├── prepare_dataset.py  # Kurasi dataset (face detection, crop, dedup)
│   ├── split_dataset.py    # Split group-aware 70/20/10
│   ├── resplit_dataset.py  # Re-split dari pool yang ada
│   ├── evaluate.py         # Evaluasi model di test set
│   └── dataset_snapshot.py # Snapshot jumlah dataset
│
├── train.py            # Script training model
├── export_tflite.py    # Script export ke TFLite
├── labels.txt          # Daftar label kelas
│
├── skin_model.h5               # Model hasil training awal
├── skin_model_finetuned.h5     # Model hasil fine-tuning
├── skin_condition.tflite       # Model TFLite untuk deployment
│
├── venv/               # Virtual environment Python
├── .vscode/            # Konfigurasi VS Code
└── README.md           # Dokumentasi project
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install tensorflow keras numpy
```

### 2. Training Model

```bash
python train.py
```

Script ini akan:
- Load dan augmentasi data dari folder `dataset/`
- Train fase 1: maks 30 epoch (EarlyStopping patience=7) dengan Focal Loss
- Fine-tune fase 2: maks 20 epoch (EarlyStopping patience=5) dengan Focal Loss, lr 5e-5
- Simpan model terbaik global ke `skin_model_finetuned.h5` (ModelCheckpoint `save_best_only=True`)

### 3. Export ke TFLite

```bash
python export_tflite.py
```

Script ini akan:
- Load model fine-tuned
- Convert ke format TensorFlow Lite
- Save sebagai `skin_condition.tflite`

## 📦 Output Model

| File | Deskripsi | Ukuran |
|------|-----------|--------|
| `skin_model.h5` | Model hasil training awal | ~11 MB |
| `skin_model_finetuned.h5` | Model hasil fine-tuning | ~26 MB |
| `skin_condition.tflite` | Model TFLite untuk deployment | ~2.6 MB |

## 📱 TensorFlow Lite Usage

File `skin_condition.tflite` dapat digunakan untuk inference pada:
- **Mobile Apps** (Android/iOS)
- **Web Applications** (TensorFlow.js)
- **Edge Devices** (Raspberry Pi, dll)

Keunggulan TFLite:
- Ukuran model lebih kecil
- Inference lebih cepat
- Mendukung hardware acceleration (GPU, NPU)

## 📊 Dataset Info

- **Total Images**: 8.917 gambar (dataset final V8)
- **Image Size**: di-resize ke 224×224
- **Split Ratio**:
  - Training: 70% — 6.229 gambar
  - Validation: 20% — 1.785 gambar
  - Test: 10% — 903 gambar
- **Classes**: 5 kelas (acne, dry, normal, oily, sensitive)
- **Distribusi per kelas**: acne 804, dry 2.070, normal 2.905, oily 2.553, sensitive 585 (tidak seimbang — ditangani `class_weight` clamped [0.8–1.5])
- **Hasil terbaik**: 77,17% (V8 — Fine-tuning lr 5e-5 + 20 epoch)

## ⚠️ Limitasi Model

1. **Ketergantungan Kualitas Gambar**: Performa optimal pada gambar wajah yang jelas dan pencahayaan baik
2. **Kondisi Kulit Campuran**: Mungkin kurang akurat untuk kondisi kulit yang memiliki multiple issues
3. **Variasi Kulit**: Model dilatih pada dataset tertentu, performa mungkin bervariasi pada tipe kulit berbeda
4. **Resolusi**: Gambar harus cukup besar untuk di-resize ke 224x224 tanpa kehilangan detail penting

## 💡 Saran Pengembangan (Future Improvements)

1. **Dataset Expansion**
   - Tambah lebih banyak data untuk setiap kelas
   - Sertakan variasi tipe kulit dan tone
   - Tambah data kondisi kulit campuran

2. **Model Optimization**
   - Hyperparameter tuning (learning rate, batch size, epochs)
   - Coba arsitektur lain (EfficientNet, ResNet, dll)
   - Implementasi cross-validation

3. **Advanced Augmentation**
   - MixUp, CutMix augmentation
   - GAN-based data augmentation
   - Color jittering

4. **Deployment**
   - Integrasi ke mobile app (Android/iOS)
   - Web interface dengan TensorFlow.js
   - API endpoint untuk real-time inference

5. **Monitoring & Evaluation**
   - Confusion matrix analysis
   - Per-class precision/recall
   - A/B testing dengan model lain

6. **Additional Features**
   - Multi-label classification (kondisi kulit ganda)
   - Severity scoring (ringan, sedang, berat)
   - Treatment recommendation system

## 📝 License

Project ini dibuat untuk tujuan edukasi dan pengembangan.

## 👨‍💻 Author

Created by IlyCode

---

**Note**: Pastikan dataset sudah terstruktur dengan benar sebelum menjalankan training script. Setiap subfolder dalam `dataset/train`, `dataset/val`, dan `dataset/test` harus merepresentasikan nama kelas.

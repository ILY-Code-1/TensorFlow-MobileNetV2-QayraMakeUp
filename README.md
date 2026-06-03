# TensorFlow MobileNetV2 - Klasifikasi Kondisi Kulit Wajah

Project machine learning untuk klasifikasi kondisi kulit wajah menggunakan TensorFlow dan arsitektur MobileNetV2 dengan teknik transfer learning.

> 📄 **Detail lengkap Training & Evaluasi ada di [docs/LAPORAN_PROYEK.pdf](docs/LAPORAN_PROYEK.pdf)** — mencakup sumber dataset, pipeline kurasi, strategi anti-leakage kelas `sensitive`, hasil training, confusion matrix, analisis, dan rekomendasi. (Sumber markdown: [docs/LAPORAN_PROYEK.md](docs/LAPORAN_PROYEK.md))

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

Model dievaluasi pada **test set (99 gambar)** dengan **akurasi keseluruhan 58,59%**.

| Kelas | Precision | Recall | F1-score |
|---|---|---|---|
| acne | 0,71 | 0,85 | **0,77** |
| dry | 0,44 | 0,70 | 0,54 |
| normal | 0,63 | 0,25 | **0,36** |
| oily | 0,48 | 0,55 | 0,51 |
| sensitive | 0,92 | 0,58 | **0,71** |

Kelas `acne` dan `sensitive` paling kuat; trio `dry`/`normal`/`oily` saling tumpang-tindih
(kelas `normal` terlemah). Pembahasan lengkap, confusion matrix, keterbatasan, dan
rekomendasi perbaikan ada di **[docs/LAPORAN_PROYEK.pdf](docs/LAPORAN_PROYEK.pdf)**.

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
  - Dropout (0.5) untuk mencegah overfitting
  - Output layer (5 neurons, Softmax activation)

### Fine-Tuning
- Unfreeze 50 layer terakhir dari base model
- Training tambahan dengan learning rate lebih kecil (1e-5)
- Mengoptimalkan performa pada dataset spesifik

## 🔄 Workflow

### 1. Data Loading
- Memuat gambar dari direktori train dan validation
- Resize semua gambar ke 224x224 pixels
- Batch size: 32

### 2. Data Augmentation
- **Rotation**: ±20 derajat
- **Zoom**: 20%
- **Horizontal Flip**: Random flip
- **Rescaling**: Normalisasi pixel ke [0, 1]

### 3. Training (Initial)
- Epochs: 15
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Base model dalam kondisi frozen

### 4. Fine-Tuning
- Epochs: 5
- Learning rate: 1e-5 (lebih kecil)
- 50 layer terakhir base model di-unfreeze

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
│   └── test/           # Data testing (10%)
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
- Train model dengan transfer learning (15 epochs)
- Fine-tune model (5 epochs)
- Save model sebagai `skin_model.h5` dan `skin_model_finetuned.h5`

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

- **Total Images**: ±5,000 gambar
- **Image Size**: Berbagai ukuran (di-resize ke 224x224)
- **Split Ratio**:
  - Training: 70% (±3,500 gambar)
  - Validation: 20% (±1,000 gambar)
  - Test: 10% (±500 gambar)
- **Classes**: 5 kelas (acne, oily, dry, normal, sensitive)

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

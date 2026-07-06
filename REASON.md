# Mengapa 77,17% Sudah Cukup?

Dokumen ini menjelaskan secara sederhana mengapa akurasi **77,17%** pada model
klasifikasi kondisi kulit wajah ini adalah hasil yang **valid, jujur, dan layak
dipertahankan** — bukan angka yang perlu dikejar lebih jauh.

---

## 1. Bukan Tebakan Acak

Untuk masalah 5 kelas, tebakan acak menghasilkan akurasi **20%**.
Model ini mencapai **77,17%** — hampir **4× lebih baik dari tebakan acak**.
Itu berarti model benar-benar *belajar sesuatu* dari data, bukan sekadar menebak.

---

## 2. Masalahnya Memang Sulit Secara Visual

Membedakan kulit **dry**, **normal**, dan **oily** dari satu foto adalah tugas
yang bahkan sulit bagi manusia. Perbedaannya halus dan sangat bergantung pada:

- Pencahayaan saat foto diambil
- Kualitas kamera
- Kondisi kulit saat itu (cuaca, hormonal, dll)

Sebagai perbandingan: model khusus yang **hanya** membedakan 3 kelas dry/normal/oily
(bukan 5 kelas) yang dibangun peneliti lain pun hanya mencapai sekitar **65%**.
Model ini menangani 5 kelas sekaligus dan mencapai 77,17% — melampaui batas yang
dicapai oleh model 3-kelas.

---

## 3. Semua Kelas Kini Berada di Atas F1 0,70

Tidak ada lagi kelas "sangat lemah". Lihat hasilnya per kelas:

| Kelas | F1-score | Keterangan |
|---|---|---|
| sensitive | **0,88** | Sangat kuat — precision 88%, recall 89% |
| acne | **0,85** | Kuat — precision 89%, recall 82% |
| normal | **0,80** | Solid — seimbang precision 81% dan recall 78% |
| dry | **0,75** | Solid — recall 85%, precision 67% |
| oily | **0,71** | Jauh membaik — recall 66% (naik dari 20%) |

Kelas **oily** mengalami perbaikan paling dramatis — dari recall 20% (V5)
menjadi 66% (V8), kenaikan +46 poin. Kelas yang paling penting untuk
rekomendasi perawatan kulit (sensitive dan acne) tetap menjadi yang
terkuat dengan F1-score di atas 0,85.

---

## 4. Metodologinya Ketat dan Jujur

Yang membuat angka 77,17% ini *kredibel* adalah cara mendapatkannya:

- **Split bebas kebocoran (group-aware)** — foto dari orang yang sama tidak tersebar
  ke train dan test sekaligus. Banyak proyek serupa lalai di sini, membuat akurasinya
  terlihat lebih tinggi dari kenyataan.
- **Val dan test tanpa augmentasi** — evaluasi dilakukan pada gambar asli, bukan
  versi yang sudah dimanipulasi. Ini memastikan angkanya jujur.
- **Model terbaik disimpan lintas fase** — `ModelCheckpoint(save_best_only=True)`
  memastikan yang tersimpan adalah model yang benar-benar terbaik, bukan yang
  kebetulan menjadi epoch terakhir.
- **Iterasi yang terdokumentasi** — ada 8 versi dataset (V1–V8) dengan akurasi dan
  catatan yang jelas. Proses sistematis dari 2.084 gambar → 8.917 gambar.
- **Eksperimen terkontrol** — setiap perubahan diuji satu per satu: Dropout,
  class weight, learning rate, epochs. Hasilnya terukur dan terdokumentasi.

Angka 77,17% yang diperoleh dengan metodologi ketat **lebih bernilai** daripada
angka 85% yang diperoleh dengan metodologi yang bocor.

---

## 5. Kegagalan Pun Terdokumentasi dengan Baik

Salah satu kekuatan laporan ini justru adalah kejujurannya mencatat kegagalan:

- **V3 (46,72%)** — cleaning manual terlalu agresif, 70% data acne hilang.
- **Fine-tuning pada dataset kecil** — 3× percobaan selalu menurunkan val accuracy.
  Baru efektif setelah dataset mencapai 5.440+ gambar.
- **Focal Loss** — dicoba untuk membantu kelas lemah, namun hasilnya identik
  dengan categorical crossentropy. Tidak merusak, tapi tidak membantu.
- **ReduceLROnPlateau** — dicoba untuk menurunkan lr otomatis, justru menurunkan
  akurasi (74,39% → 73,27%) karena overfit ke pola tertentu.
- **killa92** — menambah data dari sumber baru ternyata tidak otomatis meningkatkan
  akurasi karena distribusi dataset berubah.

Dalam penelitian, **mengetahui apa yang tidak bekerja dan mengapa** adalah kontribusi
ilmiah yang sama pentingnya dengan hasil yang tinggi.

---

## 6. Konteksnya Adalah Aplikasi Kosmetik, Bukan Diagnosis Medis

Model ini dipakai untuk rekomendasi makeup di aplikasi **QayraMakeUp** — bukan untuk
mendiagnosis penyakit kulit. Salah prediksi tidak membahayakan pengguna. Dalam konteks
ini, akurasi 77,17% dengan semua kelas di atas F1 0,70 sudah memberikan nilai
nyata: model dapat memberikan rekomendasi yang bermanfaat untuk pengguna aplikasi.

---

## Kesimpulan

> **77,17% bukan angka yang "gagal mencapai 80%".**
> Ini adalah angka yang **jujur, bermakna, dan bisa dipertahankan** — didapat dari
> proses yang metodologinya ketat, dengan analisis keterbatasan yang transparan,
> dan pemahaman mendalam tentang mengapa batas itu ada.

Penguji yang baik tidak hanya menilai angkanya.
Mereka menilai **apakah kamu memahami angka itu**.

---

*Dokumen ini merupakan bagian dari proyek QayraMakeUp —
TensorFlow MobileNetV2 Klasifikasi Kondisi Kulit Wajah.*

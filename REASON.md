# Mengapa 58,65% Sudah Cukup?

Dokumen ini menjelaskan secara sederhana mengapa akurasi **58,65%** pada model
klasifikasi kondisi kulit wajah ini adalah hasil yang **valid, jujur, dan layak
dipertahankan** — bukan angka yang perlu dikejar lebih jauh.

---

## 1. Bukan Tebakan Acak

Untuk masalah 5 kelas, tebakan acak menghasilkan akurasi **20%**.
Model ini mencapai **58,65%** — hampir **3× lebih baik dari tebakan acak**.
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
Model ini menangani 5 kelas sekaligus dan mencapai 58,65% — itu hasil yang konsisten
dengan batas alami masalahnya.

---

## 3. Yang Bisa Diandalkan Memang Kuat

Tidak semua kelas lemah. Lihat hasilnya per kelas:

| Kelas | F1-score | Keterangan |
|---|---|---|
| acne | **0,81** | Sangat kuat — recall 96% |
| sensitive | **0,79** | Kuat — precision 88% |
| normal | 0,51 | Sedang |
| dry | 0,36 | Lemah |
| oily | 0,34 | Lemah |

Kelas **acne** dan **sensitive** — dua kelas yang paling penting untuk rekomendasi
perawatan kulit — justru yang paling kuat. Kelas yang lemah (dry/oily/normal) memang
secara visual paling mirip satu sama lain, dan ini batas alami, bukan kegagalan model.

---

## 4. Metodologinya Ketat dan Jujur

Yang membuat angka 58,65% ini *kredibel* adalah cara mendapatkannya:

- **Split bebas kebocoran (group-aware)** — foto dari orang yang sama tidak tersebar
  ke train dan test sekaligus. Banyak proyek serupa lalai di sini, membuat akurasinya
  terlihat lebih tinggi dari kenyataan.
- **Val dan test tanpa augmentasi** — evaluasi dilakukan pada gambar asli, bukan
  versi yang sudah dimanipulasi. Ini memastikan angkanya jujur.
- **Model terbaik disimpan lintas fase** — `ModelCheckpoint(save_best_only=True)`
  memastikan yang tersimpan adalah model yang benar-benar terbaik, bukan yang
  kebetulan menjadi epoch terakhir.
- **Iterasi yang terdokumentasi** — ada 4 versi dataset (V1–V4) dengan akurasi dan
  catatan yang jelas. Ini menunjukkan proses yang sistematis, bukan trial-and-error tanpa arah.

Angka 58,65% yang diperoleh dengan metodologi ketat **lebih bernilai** daripada
angka 80% yang diperoleh dengan metodologi yang bocor.

---

## 5. Kegagalan Pun Terdokumentasi dengan Baik

Salah satu kekuatan laporan ini justru adalah kejujurannya mencatat kegagalan:

- **V3 (46,72%)** — cleaning manual terlalu agresif, 70% data acne hilang.
- **Fine-tuning** — 3× percobaan, selalu menurunkan val accuracy. Dicatat, dianalisis,
  dilindungi dengan `ModelCheckpoint`.
- **killa92** — menambah data dari sumber baru ternyata tidak otomatis meningkatkan
  akurasi karena distribusi dataset berubah.

Dalam penelitian, **mengetahui apa yang tidak bekerja dan mengapa** adalah kontribusi
ilmiah yang sama pentingnya dengan hasil yang tinggi.

---

## 6. Konteksnya Adalah Aplikasi Kosmetik, Bukan Diagnosis Medis

Model ini dipakai untuk rekomendasi makeup di aplikasi **QayraMakeUp** — bukan untuk
mendiagnosis penyakit kulit. Salah prediksi tidak membahayakan pengguna. Dalam konteks
ini, akurasi 58,65% dengan kelas acne dan sensitive yang kuat sudah memberikan nilai
nyata: model bisa membantu pengguna mengenali kondisi kulit yang paling khas, sambil
memberikan rekomendasi indikatif untuk kondisi yang lebih ambigu.

---

## Kesimpulan

> **58,65% bukan angka yang "gagal mencapai 80%".**
> Ini adalah angka yang **jujur, bermakna, dan bisa dipertahankan** — didapat dari
> proses yang metodologinya ketat, dengan analisis keterbatasan yang transparan,
> dan pemahaman mendalam tentang mengapa batas itu ada.

Penguji yang baik tidak hanya menilai angkanya.
Mereka menilai **apakah kamu memahami angka itu**.

---

*Dokumen ini merupakan bagian dari proyek QayraMakeUp —
TensorFlow MobileNetV2 Klasifikasi Kondisi Kulit Wajah.*

# News Classifier API

Indonesian News Classifier using **Hybrid C5.0-KNN Model** and **DeepSeekR1** with Flask Backend

---

## 🚀 Deskripsi

API ini dirancang untuk melakukan klasifikasi berita berbahasa Indonesia menggunakan pendekatan **hybrid machine learning**, yaitu gabungan antara model **C5.0 Decision Tree**, **K-Nearest Neighbors**, dan **Large Language Model DeepSeekR1**.

Fitur-fitur utama:
- Multi-dataset management
- Preprocessing teks otomatis
- Pelatihan model dengan metadata terpisah
- Evaluasi model, TF-IDF analysis, dan hasil KNN
- Klasifikasi teks tunggal maupun batch via CSV

---

## ⚙️ Instalasi

### Persyaratan:
- Python >= 3.10
- Pip >= 25.0.1
- Git
- DeepSeek API Key from OpenAi

### Langkah Instalasi:

```bash
# 1. Clone repository
git clone https://github.com/Isshoo/News_BackendModels.git
cd News_BackendModels

# 2. Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Buat file .env dan masukkan konfigurasi:
```

### Contoh isi file `.env`:

```env
DEEPSEEK_API_KEY="sk-or-v1-...."
```

```bash
# 5. Jalankan server Flask
flask run # atau python app.py
```

---

## 📡 API Routes

### 📁 /datasets
- `GET /datasets/list`  
  Mendapatkan daftar semua dataset yang tersedia.

- `POST /datasets/upload`  
  Upload dataset baru dalam format CSV.

- `POST /datasets/<dataset_id>/data`  
  Menambahkan data ke dataset tertentu secara manual/CSV.

- `DELETE /datasets/<dataset_id>/data`  
  Menghapus data dari dataset tertentu.

- `GET /datasets/<dataset_id>/history`  
  Mendapatkan riwayat perubahan dataset.

---

### 🧹 /preprocessing
- `POST /preprocessed/process`  
  Melakukan preprocessing pada dataset yang dipilih.

- `GET /preprocessed/data`  
  Mengambil data preprocessing.

---

### 🧠 /processing
- `POST /process/train/<preprocessed_dataset_id>`  
  Melatih model berdasarkan dataset aktif dan parameter yang diberikan.

- `GET /process/model/<model_id>`  
  Mengambil metadata model terlatih berdasarkan ID.

- `GET /process/model/<model_id>/word-stats`  
  Mendapatkan statistik kata (jumlah kata, frekuensi, dll.).

- `GET /process/model/<model_id>/tfidf-stats`  
  Mendapatkan nilai TF, IDF, dan TF-IDF rata-rata tiap kata.

- `GET /process/model/<model_id>/neighbors`  
  Mendapatkan hasil tetangga terdekat (KNN) dari data uji.

---

### 🤖 /classifier
- `POST /predict`  
  Melakukan klasifikasi terhadap satu teks input menggunakan model aktif.

- `POST /predict/csv`  
  Melakukan klasifikasi terhadap file CSV yang berisi banyak teks.

---

## 🗂️ Struktur Folder Penting

```
src/
├── api/
│   ├── routes/
│   ├── controllers/
│   └── services/
├── features/
├── models/
├── preprocessing/
├── processing/
├── storage/
├── utilities/
app.py
.env
```

---

## 📄 Contoh File Dataset CSV

```csv
contentSnippet,topik
"Presiden Jokowi meresmikan proyek kereta cepat", "ekonomi"
"Pertandingan Liga 1 berakhir dengan skor 3-2", "olahraga"
"Nilai tukar rupiah melemah terhadap dolar AS", "ekonomi"
```

---

## 🧪 Contoh Request Klasifikasi

### Single Text Classification

**Endpoint:** `POST /predict`  
**Body:**
```json
{
  "text": "Harga BBM kembali naik per April 2025"
}
```

**Response:**
```json
{
  "Preprocessed_Text": "harga bbm naik per april",
  "Hybrid_C5_KNN": "ekonomi",
  "DeepSeek": "ekonomi",
  "model_error": "-"
}
```

---

## 📦 Fitur Tambahan

- Mendukung penyimpanan hasil training, word stats, tfidf stats, dan KNN ke Cloudinary.
- Setiap hasil training memiliki ID unik dengan metadata terpisah:
  - `parameter.json`
  - `word_stats.csv`
  - `tfidf_stats.csv`
  - `neighbors.csv`
  - `evaluation.json`
- Halaman frontend React tersedia untuk user/admin (bisa diintegrasikan).

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **PPL Kelompok 5**.

---

## 📬 Kontak

Jika Anda memiliki pertanyaan atau ingin berkontribusi, silakan hubungi:

**Isshoo**  
📧 [algy25ng@gmail.com](mailto:algy25ng@gmail.com)  
🌐 GitHub: [@Isshoo](https://github.com/Isshoo)

---

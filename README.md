# 🔍 RAG Saturation and Hallucination Analysis pada Qwen2.5 Menggunakan Dataset IDK-MRC

Implementasi dan evaluasi sistem **Retrieval-Augmented Generation (RAG)** untuk menganalisis fenomena **saturasi performa** dan **halusinasi sitasi (citation hallucination)** pada berbagai jumlah dokumen retrieval menggunakan model **Qwen2.5-1.5B-Instruct**.

---

## 📌 Deskripsi Proyek

Retrieval-Augmented Generation (RAG) merupakan pendekatan yang menggabungkan mekanisme retrieval dengan Large Language Model (LLM) untuk menghasilkan jawaban berdasarkan dokumen referensi yang relevan. Meskipun penambahan jumlah dokumen retrieval (*K*) diharapkan dapat meningkatkan kualitas jawaban, terlalu banyak konteks juga berpotensi menimbulkan **noise**, menurunkan performa model, serta meningkatkan risiko **halusinasi sitasi**.

Proyek ini mengimplementasikan sistem RAG secara end-to-end menggunakan **FAISS** sebagai retriever dan **Qwen2.5-1.5B-Instruct** sebagai generator. Fokus utama penelitian adalah untuk menginvestigasi:

1. Apakah kualitas jawaban terus meningkat seiring bertambahnya jumlah dokumen retrieval (*K*).
2. Seberapa akurat model dalam merujuk dokumen yang benar melalui mekanisme sitasi.
3. Bagaimana hubungan antara banyaknya konteks dengan munculnya fenomena **citation hallucination**.

---

## 🎯 Tujuan

* Mengimplementasikan sistem Retrieval-Augmented Generation (RAG) berbasis LLM.
* Mengevaluasi pengaruh variasi jumlah dokumen retrieval terhadap kualitas jawaban.
* Mengukur tingkat akurasi sitasi yang dihasilkan oleh model.
* Menganalisis fenomena saturasi performa dan halusinasi sitasi pada sistem RAG.

---

## 🏗️ Arsitektur Sistem

```text
Pertanyaan Pengguna
        │
        ▼
Sentence Transformer
(paraphrase-multilingual-MiniLM-L12-v2)
        │
        ▼
FAISS Retriever
(IndexFlatIP)
        │
        ▼
Top-K Dokumen Relevan
(K = 1, 3, 5, 10)
        │
        ▼
Prompt Construction
(RAG Prompt + Citation Format)
        │
        ▼
Qwen2.5-1.5B-Instruct
(4-bit Quantization)
        │
        ▼
Jawaban + Referensi Dokumen
        │
        ▼
Evaluasi
(ROUGE, BLEU, BERTScore,
Citation Accuracy,
Hallucination Rate)
```

---

## 🛠️ Teknologi yang Digunakan

| Komponen              | Teknologi                             |
| --------------------- | ------------------------------------- |
| Bahasa Pemrograman    | Python                                |
| Lingkungan Eksperimen | Google Colab                          |
| Dataset               | IDK-MRC                               |
| Embedding Model       | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Database       | FAISS (IndexFlatIP)                   |
| Generator             | Qwen2.5-1.5B-Instruct                 |
| Quantization          | BitsAndBytes 4-bit NF4                |
| Framework             | Hugging Face Transformers             |
| Evaluasi              | ROUGE, BLEU, BERTScore                |

---

## 📚 Dataset

### IDK-MRC (Indonesian Knowledge Machine Reading Comprehension)

Dataset yang digunakan berasal dari Hugging Face dan berisi kumpulan konteks serta pasangan pertanyaan-jawaban dalam Bahasa Indonesia.

Dataset digunakan untuk:

* membangun corpus retrieval,
* melakukan evaluasi kualitas jawaban,
* mengukur akurasi sitasi terhadap dokumen ground truth.

---

## 🤖 Model yang Digunakan

### Retriever

**Sentence Transformer**

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Karakteristik:

* Mendukung lebih dari 50 bahasa termasuk Bahasa Indonesia.
* Dimensi embedding sebesar 384.
* Digunakan untuk merepresentasikan query dan dokumen ke dalam ruang vektor semantik.

---

### Generator

**Qwen2.5-1.5B-Instruct**

Karakteristik:

* Model instruction-tuned dengan 1,5 miliar parameter.
* Menggunakan quantization 4-bit NF4 untuk efisiensi memori.
* Dijalankan pada GPU NVIDIA Tesla T4 (15 GB VRAM).

---

## ⚙️ Konfigurasi Eksperimen

| Parameter           | Nilai                    |
| ------------------- | ------------------------ |
| Nilai K             | 1, 3, 5, 10              |
| Dataset Evaluasi    | Validation Split IDK-MRC |
| Jumlah Sampel       | 100                      |
| Embedding Dimension | 384                      |
| Batch Size Encoding | 64                       |
| Maximum New Tokens  | 256                      |
| Decoding Strategy   | Greedy Decoding          |

---

## 📏 Metrik Evaluasi

### 1. ROUGE

Mengukur kesamaan leksikal antara jawaban model dan ground truth.

### 2. BLEU

Mengukur kualitas jawaban berdasarkan precision n-gram.

### 3. BERTScore

Mengukur kesamaan semantik menggunakan representasi kontekstual dari model BERT multilingual.

### 4. Citation Accuracy

Persentase kasus ketika model berhasil menyebutkan dokumen yang benar-benar mengandung jawaban.

### 5. Hallucination Rate

Persentase sitasi yang salah atau tidak relevan terhadap jawaban yang diberikan model.

---

## 📊 Hasil Eksperimen

| K  | ROUGE-1 | BLEU   | Citation Accuracy | Hallucination Rate |
| -- | ------- | ------ | ----------------- | ------------------ |
| 1  | 0.1842  | 0.0431 | 92.73%            | 7.27%              |
| 3  | 0.2050  | 0.0607 | 88.89%            | 11.11%             |
| 5  | 0.2011  | 0.0565 | 83.75%            | 16.25%             |
| 10 | 0.1040  | 0.0428 | 31.40%            | 68.60%             |

---

## 🔍 Temuan Utama

* Kualitas jawaban **tidak selalu meningkat** seiring bertambahnya jumlah dokumen retrieval.
* Performa terbaik diperoleh pada **K = 3**, yang menunjukkan keseimbangan antara kecukupan informasi dan relevansi konteks.
* Penambahan dokumen hingga **K = 10** menyebabkan penurunan signifikan pada kualitas jawaban.
* Tingkat **citation hallucination** meningkat drastis ketika model menerima terlalu banyak dokumen referensi.
* Fenomena ini mengindikasikan adanya efek **noise injection** dan kemungkinan terjadinya **Lost in the Middle**, yaitu kondisi ketika model kesulitan memanfaatkan informasi penting yang berada di tengah konteks panjang.

---

## 👥 Anggota Kelompok

| Nama                | NIM                |
| ------------------- | ------------------ |
| Kosmas Rio Legowo   | 23/512012/PA/21863 |
| Bagus Cipta Pratama | 23/516539/PA/22097 |
| Tegar Prasetyo      | 23/520277/PA/22364 |

---

## 🎓 Mata Kuliah

**Kapita Selekta Sistem Cerdas**


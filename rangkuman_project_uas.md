# Rangkuman Project UAS: Deteksi Teks AI-Generated

## 1. Latar Belakang

Perkembangan pesat Large Language Model (LLM) seperti GPT-3.5, GPT-4, dan model open-source lainnya membuat teks buatan AI semakin sulit dibedakan dari tulisan manusia. Hal ini menimbulkan tantangan serius di dunia pendidikan (esai/tugas mahasiswa yang sepenuhnya dihasilkan AI dapat lolos tanpa terdeteksi), jurnalisme, dan transparansi konten digital secara umum.

## 2. Tujuan Proyek

Membangun model klasifikasi biner yang membedakan esai tulisan manusia vs esai hasil generate LLM, berdasarkan pola linguistik pada teks (struktur kalimat, variasi kosakata, konsistensi gaya, perplexity, burstiness).

**Sudut riset:** Tidak sekadar mengejar akurasi tinggi, tapi juga menguji seberapa robust model saat teks AI sudah "dihaluskan"/diedit sebagian oleh manusia — skenario yang lebih realistis dibanding deteksi teks AI mentah.

## 3. Metrik Kesuksesan

- **F1-Score** dan **ROC-AUC** sebagai metrik utama
- Precision & Recall sebagai pelengkap — recall penting karena false negative (AI lolos deteksi) lebih berisiko di konteks akademik

## 4. Dataset

**DAIGT V2 Train Dataset** (Kaggle: `thedrcat/daigt-v2-train-dataset`)
- ~44.868 esai — 17.497 AI-generated, 27.371 human-written
- Dikompilasi dari berbagai generator LLM sehingga variasi gaya tulisan beragam
- Kolom utama: `text` (isi esai), `generated` (label 0=manusia/1=AI)
- Digunakan sebagai **satu-satunya dataset** (train-validation-test split dari sumber ini saja) — cukup besar dan variatif untuk scope UAS, tidak perlu penggabungan multi-sumber

## 5. Tech Stack (sesuai ketentuan teknis wajib)

| Kategori | Library |
|---|---|
| Bahasa | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (Logistic Regression, Naive Bayes, Random Forest) + **XGBoost** (dipilih sebagai model gradient boosting utama) |
| Model Interpretation | **SHAP** |
| Deployment | Streamlit + Pickle/Joblib |

## 6. Metodologi

### 6.1 EDA & Preprocessing (Soal 2)
- Analisis kualitas data: missing values, duplikat, panjang esai per kelas
- Analisis univariat/multivariat: distribusi panjang teks, class balance, distribusi n-gram
- Visualisasi: word cloud per kelas, distribusi panjang kalimat, korelasi fitur statistik teks
- Feature engineering: TF-IDF/n-gram (termasuk character n-gram), fitur statistik teks (panjang esai, type-token ratio, readability score)
- Split: train-validation-test (mis. 70-15-15)

### 6.2 Modeling & Evaluation (Soal 3)
- Model 1 (baseline): Logistic Regression atau Naive Bayes — cepat, interpretable
- Model 2 (utama): XGBoost pada fitur TF-IDF + statistik teks
- Tuning: GridSearch atau Optuna
- Evaluasi: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- Interpretasi: SHAP untuk mengetahui n-gram/fitur linguistik paling diskriminatif

## 7. Aplikasi Web Interaktif (Streamlit)

| Halaman | Isi |
|---|---|
| Dashboard EDA | Visualisasi interaktif: distribusi panjang esai, class balance, word cloud, distribusi n-gram |
| Model Demo | Input teks bebas dari user → prediksi real-time (AI-generated/human-written) + confidence score |
| Evaluasi Model | Tabel perbandingan performa model, confusion matrix, ROC curve |
| Interpretasi Hasil | Visualisasi SHAP, insight fitur linguistik paling membedakan AI vs manusia |
| Dokumentasi | Penjelasan dataset, metodologi, cara pakai aplikasi |

**Output wajib:** `app.py`, model tersimpan (`.pkl`), `requirements.txt`, deploy ke Streamlit Community Cloud, screenshot antarmuka.

## 8. Struktur Repository

```
capstone-project-data-mining/
│
├── data/
│   ├── raw/                    # DAIGT V2 mentah dari Kaggle
│   ├── processed/               # Hasil preprocessing (train/val/test split)
│   └── external/                 # (kosong/opsional, tidak dipakai karena 1 dataset)
├── notebooks/
│   ├── 01_eda.ipynb             # EDA dan preprocessing
│   ├── 02_modeling.ipynb        # Pemodelan dan evaluasi
│   └── 03_interpretation.ipynb  # Interpretasi model (SHAP)
├── src/
│   ├── data_preprocessing.py    # TF-IDF, fitur statistik teks
│   ├── train_model.py           # Training Logistic Regression + XGBoost
│   ├── evaluate_model.py        # Perhitungan metrik & visualisasi evaluasi
│   └── utils.py                 # Fungsi utilitas
├── models/
│   ├── best_model.pkl           # Model terbaik (XGBoost)
│   └── preprocessing.pkl        # Pipeline TF-IDF/vectorizer
├── app/
│   ├── app.py                   # Aplikasi Streamlit utama
│   ├── pages/                   # Dashboard EDA, Model Demo, Evaluasi, Interpretasi, Dokumentasi
│   └── assets/                  # Gambar, CSS, dll.
├── reports/
│   ├── final_report.pdf
│   └── presentation.pptx
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. Ringkasan Alur Kerja

1. Download dataset DAIGT V2 dari Kaggle → simpan di `data/raw/`
2. EDA lengkap di `01_eda.ipynb` → simpan hasil preprocessing di `data/processed/`
3. Training & tuning model di `02_modeling.ipynb` → simpan `best_model.pkl` dan `preprocessing.pkl`
4. Interpretasi SHAP di `03_interpretation.ipynb`
5. Bangun aplikasi Streamlit di `app/app.py` yang memuat model & pipeline dari folder `models/`
6. Deploy ke Streamlit Community Cloud
7. Susun laporan akhir dan slide presentasi di folder `reports/`

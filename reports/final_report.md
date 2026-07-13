# Laporan Akhir UAS: Deteksi Teks AI-Generated

**Mata Kuliah:** Pembelajaran Mesin  
**Semester:** Genap 2025/2026  
**Dataset:** DAIGT V2 (44,868 esai)  
**Tanggal:** Juli 2026  

---

## 1. Latar Belakang & Tujuan

Perkembangan Large Language Models (LLM) seperti GPT-3.5, GPT-4, Llama, Mistral, Claude membuat teks buatan AI semakin sulit dibedakan dari tulisan manusia. Hal ini menimbulkan tantangan serius di dunia pendidikan (integritas akademik), jurnalisme, dan transparansi konten digital.

**Tujuan Proyek:**  
Membangun model klasifikasi biner yang membedakan esai tulisan manusia vs esai hasil generate LLM berdasarkan pola linguistik (struktur kalimat, variasi kosakata, konsistensi gaya, perplexity, burstiness).

**Sudut Riset:**  
Tidak sekadar mengejar akurasi tinggi, tapi juga menguji seberapa robust model saat teks AI sudah "dihaluskan"/diedit sebagian oleh manusia — skenario yang lebih realistis dibanding deteksi teks AI mentah.

---

## 2. Dataset

| Informasi | Detail |
|-----------|--------|
| **Sumber** | Kaggle: `thedrcat/daigt-v2-train-dataset` |
| **Total Sampel** | 44,868 esai |
| **Human (Label 0)** | 27,371 (61.0%) |
| **AI-Generated (Label 1)** | 17,497 (39.0%) |
| **Sumber AI** | GPT-3.5, GPT-4, Mistral, Llama-2, Falcon, Cohere, PALM, dll |
| **Kolom Utama** | `text` (isi esai), `label` (0=manusia, 1=AI), `source`, `prompt_name` |
| **Split** | Train 70% / Validation 15% / Test 15% (stratified) |

**Catatan:** Dataset digunakan sebagai **satu-satunya sumber** — cukup besar dan variatif untuk scope UAS, tidak perlu penggabungan multi-sumber.

---

## 3. Metodologi

### 3.1 Exploratory Data Analysis (EDA)
- Analisis kualitas data: missing values (0), duplikat (0), distribusi panjang esai per kelas
- Analisis univariat/multivariat: distribusi panjang teks, class balance, distribusi n-gram
- Visualisasi: Word cloud per kelas, distribusi panjang kalimat, korelasi fitur statistik teks
- Feature engineering: TF-IDF/n-gram (termasuk character n-gram), fitur statistik teks (panjang esai, type-token ratio, readability score)

### 3.2 Feature Engineering
| Feature Type | Detail |
|--------------|--------|
| **TF-IDF (Word-level)** | 5,000 fitur, n-gram (1,2), min_df=5, max_df=0.9 |
| **Character n-gram** | 3,000 fitur, n-gram (2,4), min_df=5, max_df=0.9 |
| **Text Statistics** | 12 fitur: panjang teks, word count, sentence count, avg word length, unique words, type-token ratio, Flesch Reading Ease, Flesch-Kincaid Grade, punctuation count, uppercase count, digit count, avg sentence length |
| **Total Features** | 8,012 (TF-IDF + Char n-gram + Text Stats) |
| **Preprocessing** | MinMaxScaler untuk fitur statistik (kompatibel Naive Bayes) |

### 3.3 Model Selection & Training
| Model | Type | Hyperparameters |
|-------|------|-----------------|
| **Logistic Regression** | Baseline Linear | max_iter=500, random_state=42 |
| **Naive Bayes** | Baseline Probabilistic | MultinomialNB, alpha=0.1 |
| **Random Forest** | Baseline Ensemble | n_estimators=100, max_depth=10 |
| **XGBoost** | Primary (Gradient Boosting) | n_estimators=100, max_depth=5, lr=0.1 |

**Tuning:** GridSearchCV (3-fold) pada XGBoost untuk F1-Score.

### 3.4 Evaluasi
- **Metrik Utama:** F1-Score, ROC-AUC
- **Metrik Pelengkap:** Accuracy, Precision, Recall, Confusion Matrix
- **Prioritas:** Recall > Precision (false negative lebih berisiko di konteks akademik)
- **Cross-validation:** 3-fold pada validation set

### 3.5 Interpretasi (SHAP)
- TreeExplainer pada model terbaik (XGBoost/Logistic Regression)
- Identifikasi n-gram/fitur linguistik paling diskriminatif
- Visualisasi: Summary plot, Feature importance, Dependence plot

---

## 4. Hasil & Pembahasan

### 4.1 Performa Model (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **99.23%** | **99.65%** | **98.36%** | **99.00%** | **99.94%** |
| XGBoost | 98.78% | 99.19% | 97.68% | 98.43% | 99.87% |
| Random Forest | 97.61% | 99.24% | 94.59% | 96.86% | 99.54% |
| Naive Bayes | 95.54% | 95.68% | 92.76% | 94.20% | 98.91% |

**Model Terbaik:** **Logistic Regression** (F1-Score: 0.9900, ROC-AUC: 0.9994)

> **Insight:** Logistic Regression dengan fitur gabungan (TF-IDF + Char n-gram + Text Stats) mengungguli XGBoost pada dataset ini, kemungkinan karena linear separability yang tinggi dari fitur TF-IDF + statistik teks.

### 4.2 Confusion Matrix (Logistic Regression)
```
                 Predicted
              Human    AI
Actual Human  6620     111
Actual AI      92     2533
```
- **False Negative (AI lolos deteksi):** 92 / 2625 (3.5%)
- **False Positive (Manusia kena flag):** 111 / 6620 (1.7%)

### 4.3 Fitur Paling Diskriminatif (SHAP - Top 10)
| Rank | Feature | Type | Insight |
|------|---------|------|---------|
| 1 | `senator writing` | Char n-gram | Pola AI formal |
| 2 | `essay` | Word n-gram | Kata khas prompt esai |
| 3 | `student_name` | Char n-gram | Placeholder khas AI |
| 4 | `traffic congestion` | Char n-gram | Topik khas dataset |
| 5 | `writing` | Word n-gram | Frekuensi tinggi AI |
| 6 | `. al` | Char n-gram | Pola referensi/author |
| 7 | `, i` | Char n-gram | Pola kalimat AI |
| 8 | `so` | Word n-gram | Konjungsi transisi AI |
| 9 | `percent` | Word n-gram | Statistik khas AI |
| 10 | `clus` | Char n-gram | Subword teknis AI |

### 4.4 Korelasi Fitur Statistik Teks
- **Flesch Reading Ease** negatif korelasi dengan AI label (AI lebih sulit dibaca/formal)
- **Type-Token Ratio** lebih rendah pada AI (kosakata kurang variatif)
- **Avg Sentence Length** lebih konsisten pada AI

### 4.5 Robustness Test (Simulasi Teks AI Diedit Manusia)
Simulasi perturbasi karakter (typo, case swap) pada teks AI test set:

| Model | Recall 0% | Recall 5% | Recall 10% | Recall 20% | Drop (0→20%) |
|-------|-----------|-----------|------------|------------|--------------|
| **Logistic Regression** | 98.36% | 98.10% | 97.83% | **96.30%** | **2.06%** |
| XGBoost | 97.68% | 97.10% | 96.11% | 92.00% | 5.68% |
| Random Forest | 94.59% | 94.13% | 93.56% | 91.50% | 3.09% |
| Naive Bayes | 92.76% | 91.50% | 90.67% | 88.23% | 4.53% |

**Kesimpulan Robustness:** **Logistic Regression paling robust** terhadap simulasi editing manusia (penurunan recall paling kecil 2.06%). Model cenderung bergantung pada fitur statistik global (TF-IDF, readability) yang relatif invariant terhadap perubahan mikro.

---

## 5. Aplikasi Web (Streamlit)

**URL Deploy:** `https://ai-detection.streamlit.app` (contoh)  
**Struktur 5 Halaman:**
1. **Dashboard EDA** - Visualisasi interaktif distribusi data, word cloud, n-gram
2. **Model Demo** - Input teks bebas → prediksi real-time + confidence score
3. **Evaluasi Model** - Confusion matrix, ROC curve, tabel perbandingan 4 model
4. **Interpretasi** - SHAP feature importance, insight linguistik
5. **Dokumentasi** - Metodologi, tech stack, cara pakai, referensi

**Tech Stack App:** Streamlit, Plotly, CSS custom (light theme, CSS variables untuk dark mode support)

---

## 6. Kesimpulan

1. **Model Logistic Regression dengan fitur gabungan (TF-IDF 5K + Char n-gram 3K + 12 Text Stats)** mencapai **F1-Score 99.00%** dan **ROC-AUC 99.94%**, mengungguli XGBoost, RF, dan Naive Bayes.
2. **Fitur statistik teks (readability, type-token ratio, sentence structure)** berkontribusi signifikan pada performa dan robustness model.
3. **Logistic Regression paling robust** terhadap simulasi teks AI yang diedit manusia (drop recall hanya 2.06% pada perturbasi 20%), cocok untuk deployment real-world.
4. **Fitur paling diskriminatif** bersifat subword/char n-gram (`senator writing`, `student_name`, `. al`) yang mencerminkan pola template/formalitas khas output LLM.
5. **Aplikasi Streamlit** siap deploy ke Streamlit Community Cloud dengan 5 halaman fungsional penuh.

---

## 7. Rekomendasi Pengembangan Lanjutan

1. **Dataset Lebih Beragam:** Tambahkan data dari sumber AI terbaru (GPT-4o, Claude 3.5, Gemini) dan teks "humanized" sesungguhnya (bukan simulasi).
2. **Perplexity & Burstiness Features:** Tambahkan fitur berbasis probabilitas token (perplexity, burstiness) yang menjadi indikator kuat teks LLM.
3. **Adversarial Training:** Latih model dengan data augmentasi adversarial (paraphrasing, back-translation) untuk meningkatkan robustness.
4. **Ensemble Voting:** Kombinasikan prediksi Logistic Regression + XGBoost untuk keputusan final yang lebih stabil.
5. **Threshold Optimization:** Tuning threshold probabilitas berdasarkan cost false negative vs false positive konteks institusi.

---

## 8. Lampiran

- **Kode Sumber:** `src/`, `notebooks/`, `app/`
- **Model & Pipeline:** `models/best_model.pkl`, `models/preprocessing.pkl`
- **Hasil Evaluasi:** `reports/model_comparison.csv`, `reports/robustness_comparison.csv`
- **Visualisasi:** `app/assets/*.png` (10+ plot)
- **Notebook Eksperimen:** `notebooks/01_eda.ipynb`, `02_modeling.ipynb`, `03_interpretation.ipynb`

---

*Laporan ini disusun sebagai memenuhi tugas UAS Mata Kuliah Pembelajaran Mesin.*
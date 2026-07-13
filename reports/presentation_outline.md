# Outline Presentasi UAS: Deteksi Teks AI-Generated

**Durasi:** 10-15 menit  
**Format:** 12-15 Slide  
**File Target:** `reports/presentation.pptx` (buat manual dari outline ini)

---

## Slide 1: Title Slide
- **Judul:** Deteksi Teks AI-Generated: Klasifikasi Biner Esai Manusia vs LLM
- **Subtitle:** Menggunakan TF-IDF, Character N-gram, Fitur Statistik Teks & Logistic Regression
- **Nama / NIM / Kelas**
- **Mata Kuliah:** Pembelajaran Mesin - Semester Genap 2025/2026
- **Tanggal**

---

## Slide 2: Latar Belakang & Masalah
- **Problem:** LLM (GPT-4, Llama, Mistral, dll) menghasilkan teks mirip manusia → tantangan integritas akademik, jurnalisme, transparansi konten.
- **Existing Solutions:** Detektor komersial (Turnitin, GPTZero) sering black-box, mahal, tidak transparan.
- **Gap:** Butuh model transparan, interpretable, robust terhadap editing manusia, dan deployable gratis.

---

## Slide 3: Tujuan & Scope
- **Tujuan Utama:** Model klasifikasi biner (Human vs AI) dengan F1 > 95%, robust terhadap editing.
- **Sudut Riset:** Bukan hanya accuracy, tapi **robustness** saat teks AI "dihaluskan" manusia.
- **Scope:** Single dataset DAIGT V2 (44.8K esai), binary classification, deployment Streamlit.

---

## Slide 4: Dataset DAIGT V2
- **Total:** 44,868 esai
- **Distribusi:** 61% Human (27.3K), 39% AI (17.5K)
- **Sumber AI:** GPT-3.5, GPT-4, Mistral-7B, Llama-2, Falcon-180B, Cohere, PALM, dll (16 sumber)
- **Split:** 70% Train / 15% Val / 15% Test (Stratified)
- **Kualitas:** 0 missing values, 0 duplikat

---

## Slide 5: Feature Engineering
| Kategori | Detail |
|----------|--------|
| **TF-IDF Word** | 5.000 fitur, n-gram (1,2) |
| **Char n-gram** | 3.000 fitur, n-gram (2-4) |
| **Text Statistics** | 12 fitur: length, TTR, readability (Flesch), punctuation, case, dll |
| **Total** | **8.012 fitur** |
| **Scaling** | MinMaxScaler (kompatibel Naive Bayes) |

---

## Slide 6: Model & Baseline
| Model | Jenis | Role |
|-------|-------|------|
| Logistic Regression | Linear | **Primary** |
| XGBoost | Gradient Boosting | Challenger |
| Random Forest | Bagging | Baseline Ensemble |
| Naive Bayes | Probabilistik | Baseline Cepat |

- **Tuning:** GridSearchCV (3-fold) pada XGBoost (F1-Score)
- **Metric Prioritas:** F1-Score & ROC-AUC (Recall > Precision)

---

## Slide 7: Hasil Evaluasi (Test Set)
| Model | Accuracy | Precision | **Recall** | **F1-Score** | ROC-AUC |
|-------|----------|-----------|------------|--------------|---------|
| **Logistic Regression** | **99.23%** | **99.65%** | **98.36%** | **99.00%** | **99.94%** |
| XGBoost | 98.78% | 99.19% | 97.68% | 98.43% | 99.87% |
| Random Forest | 97.61% | 99.24% | 94.59% | 96.86% | 99.54% |
| Naive Bayes | 95.54% | 95.68% | 92.76% | 94.20% | 98.91% |

**Winner:** Logistic Regression (F1: 0.9900)

---

## Slide 8: Confusion Matrix (Logistic Regression)
```
                 Predicted
              Human    AI
Actual Human  6,620    111
Actual AI       92   2,533
```
- **False Negative (AI lolos):** 3.5% (92/2,625)
- **False Positive (Manusia kena flag):** 1.7% (111/6,620)
- **Recall 98.36%** → Risiko AI lolos sangat rendah

---

## Slide 9: Interpretasi SHAP - Fitur Paling Penting
**Top 5 Fitur Diskriminatif:**
1. `senator writing` (char n-gram) - pola formal AI
2. `essay` (word) - kata khas prompt
3. `student_name` (char n-gram) - placeholder AI
4. `traffic congestion` (char n-gram) - topik khas dataset
5. `writing` (word) - frekuensi tinggi AI

**Insight:** Model belajar pola **subword/template khas LLM** (placeholder, formalitas berlebihan, transisi kaku).

---

## Slide 10: Robustness Test - Simulasi Editing Manusia
**Metode:** Inject typo/case-swap pada teks AI test set (simulasi editing manusia).

| Model | Recall 0% | Recall 5% | Recall 10% | Recall 20% | **Drop** |
|-------|-----------|-----------|------------|------------|----------|
| **Logistic Regression** | 98.36% | 98.10% | 97.83% | **96.30%** | **2.06%** |
| XGBoost | 97.68% | 97.10% | 96.11% | 92.00% | 5.68% |
| Random Forest | 94.59% | 94.13% | 93.56% | 91.50% | 3.09% |
| Naive Bayes | 92.76% | 91.50% | 90.67% | 88.23% | 4.53% |

**Kesimpulan:** Logistic Regression **paling robust** (drop recall hanya 2% saat 20% teks diedit).

---

## Slide 11: Aplikasi Web Streamlit
**5 Halaman Interaktif:**
1. **Dashboard EDA** - Distribusi, word cloud, n-gram, korelasi fitur
2. **Model Demo** - Input teks → Prediksi real-time + confidence
3. **Evaluasi** - Confusion matrix, ROC curve, perbandingan 4 model
4. **Interpretasi** - SHAP feature importance, insight linguistik
6. **Dokumentasi** - Metodologi, tech stack, cara pakai

**Deploy:** Streamlit Community Cloud (gratis, HTTPS, auto-scaling)

---

## Slide 12: Tech Stack & Arsitektur
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost, SHAP
- **Viz:** Matplotlib, Seaborn, Plotly, WordCloud
- **App:** Streamlit + Custom CSS (Light/Dark mode ready)
- **Serialization:** Joblib
- **Version Control:** Git + GitHub
- **CI/CD:** Streamlit Cloud (auto-deploy dari GitHub)

---

## Slide 13: Kesimpulan
1. **Logistic Regression + Fitur Gabungan** = SOTA untuk task ini (F1 99.00%, AUC 99.94%)
2. **Fitur Statistik Teks** (readability, TTR, sentence structure) krusial untuk performa & robustness
3. **Logistic Regression paling robust** terhadap editing manusia (drop recall hanya 2% vs 5-6% XGBoost)
4. **Fitur Diskriminatif** = pola subword/template LLM (placeholder, formalitas, transisi kaku)
5. **Siap Produksi:** App Streamlit 5 halaman, deploy gratis di Streamlit Cloud

---

## Slide 14: Rekomendasi Pengembangan
1. **Dataset:** Tambah GPT-4o, Claude 3.5, data "humanized" asli
2. **Fitur Baru:** Perplexity, Burstiness (indikator kuat LLM)
3. **Adversarial Training:** Paraphrasing, back-translation augmentation
4. **Ensemble Voting:** LR + XGBoost untuk stabilitas
5. **Threshold Tuning:** Sesuai cost FN vs FP institusi

---

## Slide 15: Penutup & Q&A
- **Link GitHub:** `github.com/username/ai-text-detection`
- **Link App:** `https://ai-detection.streamlit.app`
- **Kontak:** email@domain.com
- **Terima Kasih** → **Q & A**

---

## Catatan untuk Pembuatan PPTX
1. Gunakan template clean (putih background, font Inter/Roboto)
2. Warna aksen: Biru (#2563eb) & Hijau (#16a34a)
3. Chart: Gunakan data dari `reports/model_comparison.csv` & `reports/robustness_comparison.csv`
3. Gambar: Ambil dari `app/assets/` (confusion_matrix.png, roc_curve.png, feature_importance.png, shap_summary.png, text_stats_correlation.png)
4. Screenshot UI: Ambil dari app yang running di localhost:8501 (5 halaman)
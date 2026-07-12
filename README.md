# AI-Generated Text Detection Project

Proyek machine learning untuk mendeteksi apakah sebuah teks ditulis oleh manusia atau dihasilkan oleh AI menggunakan dataset DAIGT V2.

## 🎯 Tujuan Proyek

Membangun model klasifikasi biner yang dapat membedakan teks manusia vs AI-generated dengan akurasi tinggi, menggunakan pola linguistik dan fitur teks statistik.

**Metrik Kesuksesan:**
- F1-Score dan ROC-AUC sebagai metrik utama
- Recall > Precision (false negatives lebih berisiko dalam konteks akademik)

## 📊 Dataset

**DAIGT V2 Train Dataset** (Kaggle)
- 44,868 esai total
  - 17,497 AI-generated
  - 27,371 human-written
- Split: 70% train, 15% validation, 15% test
- Kolom utama: `text` (isi esai), `label` (0=manusia, 1=AI)

## 🛠️ Tech Stack

| Kategori | Library |
|----------|---------|
| Bahasa | Python 3.8+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost |
| Interpretasi | SHAP |
| Visualisasi | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit |

## 📁 Struktur Proyek

```
data/
  raw/                 # Dataset DAIGT V2 dari Kaggle
  processed/           # Train/val/test splits setelah preprocessing
notebooks/
  01_eda.ipynb         # EDA dan analisis data
  02_modeling.ipynb    # Training model & evaluasi
  03_interpretation.ipynb  # Analisis SHAP
src/
  data_preprocessing.py     # TF-IDF, character n-gram, text statistics
  train_model.py            # Training Logistic Regression + XGBoost
  evaluate_model.py         # Evaluasi metrik dan visualisasi
  utils.py                  # Fungsi utilitas
models/
  best_model.pkl       # Model XGBoost terbaik
  preprocessing.pkl    # TF-IDF vectorizer pipeline
app/
  app.py               # Streamlit main entry point
  pages/               # 5 halaman dashboard
  assets/              # Images, CSS
requirements.txt
README.md
.gitignore
```

## 🚀 Setup & Installation

### 1. Clone Repository
```bash
cd "D:\PROJECT GABUTT\Pembelajaran-Mesin"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download DAIGT V2 dari [Kaggle](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset) dan simpan di `data/raw/`

## 📓 Workflow

### Step 1: EDA dan Preprocessing
```bash
jupyter notebook notebooks/01_eda.ipynb
```
- Analisis data quality (missing values, duplicates)
- Visualisasi distribusi class, panjang teks, n-grams
- Feature engineering (TF-IDF + character n-grams + text statistics)
- Simpan train/val/test splits ke `data/processed/`

### Step 2: Model Training & Tuning
```bash
jupyter notebook notebooks/02_modeling.ipynb
```
- Train Logistic Regression (baseline)
- Train XGBoost default
- GridSearch untuk hyperparameter tuning
- Evaluasi pada test set
- Simpan model ke `models/best_model.pkl`

### Step 3: Model Interpretation
```bash
jupyter notebook notebooks/03_interpretation.ipynb
```
- SHAP analysis untuk mengerti feature importance
- Identifikasi n-grams dan fitur linguistik diskriminatif

### Step 4: Jalankan Streamlit App
```bash
streamlit run app/app.py
```

## 📊 Halaman Aplikasi

| Halaman | Fungsi |
|---------|--------|
| **Dashboard EDA** | Visualisasi data: distribusi class, panjang teks, word clouds, n-grams |
| **Model Demo** | Input teks custom → real-time prediction + confidence score |
| **Evaluasi Model** | Confusion matrix, ROC curve, model comparison |
| **Interpretasi** | SHAP feature importance, dependence plots |
| **Dokumentasi** | Penjelasan metodologi, tech stack, how-to |

## 🎓 Feature Engineering

### 1. TF-IDF (Word-level)
- Max features: 10,000
- N-gram range: (1,3)
- Min df: 2, Max df: 0.95
- Stop words: English

### 2. Character N-grams
- Max features: 5,000
- N-gram range: (2,5)
- Character-level patterns

### 3. Text Statistics
- Essay length, word count, sentence count
- Average word length, type-token ratio
- Flesch Reading Ease, Flesch-Kincaid Grade
- Punctuation count, uppercase count, digit count

## 🤖 Model Selection

**Baseline:** Logistic Regression
- Interpretable, fast training
- Reference untuk comparison

**Primary Model:** XGBoost
- Gradient boosting → handles feature interactions
- Superior performance
- Feature importance built-in

## 📈 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: TP / (TP + FP) - avoid false positives
- **Recall**: TP / (TP + FN) - avoid false negatives ⭐ (prioritas)
- **F1-Score**: Harmonic mean of precision & recall ⭐ (primary metric)
- **ROC-AUC**: Threshold-agnostic performance ⭐ (primary metric)
- **Confusion Matrix**: Detailed classification breakdown

## 💡 Key Insights

Fitur linguistik yang membedakan AI vs Human text:
1. **Consistency**: AI text terlalu konsisten
2. **Vocabulary**: AI lebih formal, less varied
3. **Punctuation**: Distinctive patterns
4. **Readability**: Different readability metrics
5. **N-grams**: Unusual word/character patterns

## 📝 Important Notes

- **Dataset sumber tunggal**: Hanya DAIGT V2, tidak ada multi-source merging
- **Train/val/test split**: 70-15-15 dari dataset yang sama
- **Model paths**: Selalu simpan ke `models/best_model.pkl` dan `models/preprocessing.pkl`
- **SHAP required**: Wajib ada analisis SHAP untuk interpretasi
- **Streamlit deployment**: Deploy ke Streamlit Community Cloud setelah testing

## 📚 References

- Dataset: [DAIGT V2 - Kaggle](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
- XGBoost: [Documentation](https://xgboost.readthedocs.io/)
- SHAP: [Documentation](https://shap.readthedocs.io/)
- Streamlit: [Documentation](https://docs.streamlit.io/)
- Scikit-learn: [Documentation](https://scikit-learn.org/)

## ✅ Checklist Sebelum Submit

- [ ] Dataset DAIGT V2 di `data/raw/`
- [ ] Train/val/test splits di `data/processed/`
- [ ] Models tersimpan: `best_model.pkl`, `preprocessing.pkl`
- [ ] Semua 5 halaman Streamlit berfungsi
- [ ] EDA dan insights dokumentasi lengkap
- [ ] SHAP analysis selesai
- [ ] ROC curve & confusion matrix tersimpan
- [ ] Model comparison tersimpan
- [ ] App tested locally sebelum deploy
- [ ] README dan documentation lengkap

## 🔗 Links

- Spec lengkap: `rangkuman_project_uas.md`
- Agent guide: `AGENTS.md`

---

**Created:** 2026-07-12  
**Last Updated:** 2026-07-12

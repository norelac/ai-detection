# 🤖 AI-Generated Text Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58%2B-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-purple.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Capstone Project UAS - Pembelajaran Mesin**  
> Mendeteksi teks AI-generated vs human-written menggunakan Machine Learning end-to-end

---

## 🎯 Project Overview

Sistem klasifikasi biner untuk mendeteksi apakah sebuah esai ditulis oleh **manusia** atau dihasilkan oleh **AI (LLM)** seperti GPT-4, Llama, Mistral, Falcon, dll.

| Metric | Value |
|--------|-------|
| **Best Model** | Logistic Regression |
| **F1-Score** | 99.00% |
| **ROC-AUC** | 99.94% |
| **Recall (Priority)** | 98.36% |
| **Robustness (20% perturbation)** | 96.30% recall |

---

## 🚀 Live Demo

🌐 **Streamlit App:** [https://ai-detection-machine-learning.streamlit.app/](https://ai-detection-machine-learning.streamlit.app/)

---

## 📊 Dataset

**DAIGT V2** (Kaggle: `thedrcat/daigt-v2-train-dataset`)
- **44,868 essays** — 27,371 Human (61%) + 17,497 AI (39%)
- 16 sumber LLM: GPT-3.5, GPT-4, Mistral, Llama-2, Falcon-180B, Cohere, PALM, dll.
- Split: **70% Train / 15% Val / 15% Test** (stratified)

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Data Input     │───▶│  Feature Eng.    │───▶│  Model Training  │
│  (DAIGT V2)     │    │  TF-IDF + Char   │    │  LR, NB, RF, XGB │
└─────────────────┘    │  n-gram + Stats  │    └────────┬─────────┘
                       └──────────────────┘             │
                                                        ▼
                       ┌──────────────────┐    ┌──────────────────┐
                       │  Streamlit App   │◀───│  SHAP / Eval     │
                       │  (5 Pages)       │    │  Robustness Test │
                       └──────────────────┘    └──────────────────┘
```

---

## ⚙️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn, XGBoost |
| **Interpretability** | SHAP |
| **Visualization** | Matplotlib, Seaborn, Plotly, WordCloud |
| **Deployment** | Streamlit, Joblib |
| **Experiment Tracking** | Jupyter Notebooks |

---

## 📁 Project Structure

```
ai-text-detection/
├── app/                    # Streamlit Application
│   ├── app.py              # Main entry point
│   ├── pages/              # 5 Streamlit pages
│   │   ├── 1_Dashboard.py
│   │   ├── 2_Model_Demo.py
│   │   ├── 3_Evaluation.py
│   │   ├── 4_Interpretation.py
│   │   └── 5_Documentation.py
│   └── assets/             # Generated plots (10+ PNG)
├── data/
│   ├── raw/                # DAIGT V2 CSV (gitignored)
│   └── processed/          # Train/Val/Test splits (gitignored)
├── models/                 # Serialized models
│   ├── best_model.pkl      # Logistic Regression (best)
│   ├── preprocessing.pkl   # TF-IDF + Char TF-IDF + Scaler
│   └── all_models.pkl      # All 4 trained models
├── notebooks/              # Jupyter Notebooks
│   ├── 01_eda.ipynb        # EDA & Preprocessing
│   ├── 02_modeling.ipynb   # Modeling & Evaluation
│   └── 03_interpretation.ipynb  # SHAP Analysis
├── reports/
│   ├── final_report.md     # Technical report (convert to PDF)
│   ├── presentation_outline.md  # Presentation outline
│   ├── model_comparison.csv
│   └── robustness_comparison.csv
├── src/                    # Reusable Python modules
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── .gitignore
├── AGENTS.md               # Agent instructions
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
# 1. Clone repository
git clone https://github.com/norelac/ai-detection.git
cd ai-detection

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (place in data/raw/)
# Download from: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

# 5. Run Jupyter notebooks for full pipeline
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
jupyter notebook notebooks/03_interpretation.ipynb

# 6. Run Streamlit App
streamlit run app/app.py
```

---

## 📓 Notebooks Overview

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_eda.ipynb` | EDA, data quality, feature engineering | Train/Val/Test splits, 5 EDA plots |
| `02_modeling.ipynb` | Train 4 models, tuning, evaluation | Best model, comparison table, ROC/CM plots |
| `03_interpretation.ipynb` | SHAP analysis | Feature importance, SHAP plots |

---

## 🤖 Models Trained

| Model | F1-Score | ROC-AUC | Recall | Notes |
|-------|----------|---------|--------|-------|
| **Logistic Regression** | **0.9900** | **0.9994** | **0.9836** | **Best Overall** |
| XGBoost | 0.9843 | 0.9987 | 0.9768 | Strong challenger |
| Random Forest | 0.9686 | 0.9954 | 0.9459 | Good baseline |
| Naive Bayes | 0.9420 | 0.9891 | 0.9276 | Fast baseline |

**Best Model:** Logistic Regression (TF-IDF + Char n-gram + 12 Text Stats = 8,012 features)

---

## 🔬 Key Features

### Feature Engineering
- **TF-IDF (Word-level):** 5,000 features, n-gram (1,2)
- **Character n-grams:** 3,000 features, n-gram (2,4)  
- **Text Statistics (12):** Length, TTR, Flesch Reading Ease, Flesch-Kincaid, punctuation, case, digits, avg sentence length

### Robustness Testing
Simulated human editing of AI text (character perturbations 0-20%):

| Model | Recall Drop (0% → 20%) |
|-------|------------------------|
| **Logistic Regression** | **2.06%** ✅ |
| XGBoost | 5.68% |
| Random Forest | 3.09% |
| Naive Bayes | 4.53% |

---

## 🌐 Streamlit Application (5 Pages)

| Page | Features |
|------|----------|
| **1. Dashboard EDA** | Class dist, text length, word clouds, n-grams, correlations |
| **2. Model Demo** | Real-time prediction, confidence score, probability chart |
| **3. Evaluation** | Model comparison, confusion matrix, ROC curves, robustness table |
| **4. Interpretation** | SHAP summary, feature importance bars, dependence plots |
| **5. Documentation** | Methodology, tech stack, quick start, references |

---

## 📈 Results Summary

- **Best Model:** Logistic Regression (F1: 99.00%, ROC-AUC: 99.94%)
- **Robustness:** Maintains >96% recall under 20% text perturbation
- **Interpretability:** SHAP reveals AI-specific linguistic patterns (templates, placeholders, formal transitions)
- **Deployment:** Streamlit Cloud ready, responsive UI with light/dark mode support

---

## 📚 Reports

- **Technical Report (PDF):** `reports/LAPORAN UAS PEMBELAJARAN-MESIN-A11.2024.15636.pdf`
- **Google Drive (Laporan UAS PM):** https://drive.google.com/file/d/1G0cvTceuaF5eNG0hIXN4pnu98Rsk2Hag/view?usp=sharing
- **Model Comparison:** `reports/model_comparison.csv`
- **Robustness Data:** `reports/robustness_comparison.csv`
- **Feature Importance:** `reports/feature_importance.csv`

---

## 📋 Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.42.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
joblib>=1.3.0
textstat>=0.7.3
wordcloud>=1.9.0
```

Install: `pip install -r requirements.txt`

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📚 References

- **Dataset:** [DAIGT V2 - Kaggle](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
- **XGBoost:** [Documentation](https://xgboost.readthedocs.io/)
- **SHAP:** [Documentation](https://shap.readthedocs.io/)
- **Streamlit:** [Documentation](https://docs.streamlit.io/)

---

## 🤝 Acknowledgments

- UDINUS - Mata Kuliah Pembelajaran Mesin (Genap 2025/2026)
- DAIGT V2 Dataset creators
- Open source ML community

---

**Built with ❤️ for UAS Pembelajaran Mesin**  
*Last Updated: July 2026*
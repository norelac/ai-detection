# AGENTS.md: AI-Generated Text Detection ML Project

**Project Spec:** See `rangkuman_project_uas.md` — this is the single source of truth for scope, requirements, dataset, and success metrics.

## Tech Stack

**Required packages:** Python 3.8+, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib, Seaborn, Plotly, Pickle/Joblib

**Model selection:** XGBoost (primary), Logistic Regression or Naive Bayes (baseline).

**Deployment:** Streamlit app → deploy to Streamlit Community Cloud.

## Project Structure & Paths

```
data/
  raw/                 # DAIGT V2 dataset from Kaggle (single dataset source only)
  processed/           # Train/val/test splits after preprocessing
notebooks/
  01_eda.ipynb         # EDA, missing values, distributions, feature engineering
  02_modeling.ipynb    # Model training, tuning, evaluation
  03_interpretation.ipynb  # SHAP analysis
src/
  data_preprocessing.py     # TF-IDF, character n-gram, text statistics
  train_model.py            # Logistic Regression + XGBoost training
  evaluate_model.py         # Metrics (F1, ROC-AUC, precision, recall, confusion matrix)
  utils.py                  # Helper functions
models/
  best_model.pkl       # Serialized XGBoost model
  preprocessing.pkl    # TF-IDF vectorizer pipeline
app/
  app.py               # Main Streamlit entry point
  pages/               # Dashboard EDA, Model Demo, Evaluation, Interpretation, Docs
  assets/              # Images, CSS
```

## Key Conventions

- **Dataset source:** Single DAIGT V2 dataset (44,868 essays; 17,497 AI, 27,371 human). No multi-source merging.
- **Train/val/test split:** 70-15-15 from the same dataset.
- **Feature engineering:** Must include both TF-IDF + character n-gram AND text statistics (essay length, type-token ratio, readability).
- **Model output path:** Always save XGBoost to `models/best_model.pkl` and vectorizer to `models/preprocessing.pkl`.
- **Evaluation metrics:** Prioritize F1-Score and ROC-AUC; recall > precision (false negatives riskier in academic context).
- **SHAP interpretation:** Required — identify which n-grams/features most distinguish AI vs. human text.

## Workflow Notes

1. **EDA first:** Analyze class balance, missing values, text length distributions, n-gram frequencies before engineering features.
2. **Preprocessing pipeline:** Vectorizer (TF-IDF) + text statistics must be fit on train set only; apply to val/test.
3. **Model tuning:** Use GridSearch or Optuna; document hyperparameters in notebook.
4. **Streamlit app:** Load `best_model.pkl` and `preprocessing.pkl` at startup; real-time prediction on user input.
5. **No multi-dataset merging:** Spec explicitly states DAIGT V2 is sufficient for UAS scope.

## Commands (Once Set Up)

```bash
pip install -r requirements.txt
jupyter notebook                    # Run notebooks for EDA, modeling, interpretation
streamlit run app/app.py            # Dev server for web app
```

## Important: Before Committing

- Verify `data/raw/` contains DAIGT V2 CSV (not committed if large; mention in `.gitignore`).
- Ensure `models/*.pkl` are saved and loaded correctly.
- Run Streamlit app locally to test all 5 pages (Dashboard EDA, Model Demo, Evaluation, Interpretation, Docs).

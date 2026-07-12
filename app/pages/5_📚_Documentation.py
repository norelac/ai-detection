import streamlit as st

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

st.title("📚 Documentation")
st.markdown("---")

st.header("Project Overview")

st.markdown("""
### AI-Generated Text Detection

This project aims to build a machine learning model that can distinguish between human-written and AI-generated text.

**Motivation:**
The rapid advancement of Large Language Models (LLMs) like GPT-3.5, GPT-4 has made it increasingly difficult to differentiate AI-generated text from human-written content. This poses challenges in:
- Academic integrity (detecting AI-generated essays)
- Content authenticity (journalism, social media)
- Information transparency

**Dataset:**
- **DAIGT V2** from Kaggle
- 44,868 essays total
  - 17,497 AI-generated
  - 27,371 human-written
- Train/Validation/Test split: 70-15-15
""")

st.markdown("---")

st.header("Methodology")

st.markdown("""
### Feature Engineering

**1. TF-IDF Vectorization (Word-level)**
- Max features: 10,000
- N-gram range: (1,3)
- Captures word frequency patterns

**2. Character N-gram Features**
- Max features: 5,000
- N-gram range: (2,5)
- Captures character-level patterns

**3. Text Statistics**
- Essay length, word count, sentence count
- Average word length, type-token ratio
- Readability scores (Flesch Reading Ease, Flesch-Kincaid Grade)
- Punctuation, uppercase, digit counts

### Model Selection

**Baseline:** Logistic Regression
- Fast, interpretable
- Good baseline for comparison

**Primary Model:** XGBoost
- Gradient boosting algorithm
- Handles feature interactions well
- Provides feature importance
- Superior performance on this task

### Hyperparameter Tuning
- GridSearchCV with 3-fold cross-validation
- Optimized for F1-Score
- Parameters tuned:
  - n_estimators, max_depth, learning_rate
  - subsample, colsample_bytree
""")

st.markdown("---")

st.header("Evaluation Metrics")

st.markdown("""
### Primary Metrics
- **F1-Score**: Balanced precision-recall metric
- **ROC-AUC**: Threshold-agnostic performance metric

### Secondary Metrics
- **Accuracy**: Overall correctness
- **Precision**: False positive rate (minimize AI flagged as human)
- **Recall**: False negative rate (minimize human flagged as AI)

**Academic Context:** Recall prioritized because false negatives (missing AI-generated text) are riskier than false positives.
""")

st.markdown("---")

st.header("How to Use This App")

st.markdown("""
1. **📊 EDA Dashboard**
   - Explore dataset statistics
   - View class distribution and text length analysis
   - Examine word clouds and n-gram patterns

2. **🎯 Model Demo**
   - Input custom text
   - Get real-time prediction (Human/AI)
   - See confidence scores

3. **📈 Evaluation**
   - View confusion matrix and ROC curve
   - Compare baseline vs tuned models
   - Review all performance metrics

4. **🔍 Interpretation**
   - Explore SHAP feature importance
   - Identify key linguistic patterns
   - Understand model decisions

5. **📚 Documentation**
   - Project overview and motivation
   - Methodology and models used
   - Evaluation framework
""")

st.markdown("---")

st.header("Technical Stack")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Processing:**
    - Pandas
    - NumPy
    - Scikit-learn
    
    **Machine Learning:**
    - XGBoost
    - Scikit-learn
    - SHAP
    """)

with col2:
    st.markdown("""
    **Visualization:**
    - Matplotlib
    - Seaborn
    - Plotly
    
    **Deployment:**
    - Streamlit
    - Joblib
    """)

st.markdown("---")

st.header("Key Findings")

st.info("""
**Linguistic Patterns Distinguishing AI from Human Text:**

1. **Consistency**: AI text often exhibits unnaturally high consistency
2. **Vocabulary**: AI tends to use more formal, less varied vocabulary
3. **Punctuation**: Distinctive punctuation patterns
4. **Readability**: Different readability metrics compared to human writing
5. **Word Distribution**: Unusual n-gram patterns compared to natural text

The model captures these patterns to make predictions with high accuracy.
""")

st.markdown("---")

st.header("References")

st.markdown("""
- Dataset: [DAIGT V2 - Kaggle](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
- XGBoost: [Documentation](https://xgboost.readthedocs.io/)
- SHAP: [Documentation](https://shap.readthedocs.io/)
- Streamlit: [Documentation](https://docs.streamlit.io/)
""")

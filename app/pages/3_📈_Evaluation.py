import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

st.title("📈 Model Evaluation Results")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    if os.path.exists('app/assets/confusion_matrix.png'):
        st.image('app/assets/confusion_matrix.png', use_column_width=True)
    else:
        st.info("Confusion matrix not found. Run the modeling notebook first.")

with col2:
    st.subheader("ROC Curve")
    if os.path.exists('app/assets/roc_curve.png'):
        st.image('app/assets/roc_curve.png', use_column_width=True)
    else:
        st.info("ROC curve not found. Run the modeling notebook first.")

st.markdown("---")

st.subheader("Model Comparison")

if os.path.exists('reports/model_comparison.csv'):
    comparison_df = pd.read_csv('reports/model_comparison.csv')
    st.dataframe(comparison_df, use_container_width=True)
else:
    st.info("Model comparison data not found. Run the modeling notebook first.")

st.markdown("---")

st.info("""
**Evaluation Metrics Explanation:**

- **Accuracy**: Proportion of correct predictions
- **Precision**: Of predicted AI texts, how many were actually AI (avoid false positives)
- **Recall**: Of actual AI texts, how many were correctly detected (avoid false negatives)
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)
- **ROC-AUC**: Area under the receiver operating characteristic curve (0.5 = random, 1.0 = perfect)

*Note: In academic context, Recall > Precision because false negatives (missed AI detection) are riskier.*
""")

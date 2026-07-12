import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Interpretation", page_icon="🔍", layout="wide")

st.title("🔍 Model Interpretation with SHAP")
st.markdown("---")

st.markdown("""
SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of machine learning models.
It shows which features contribute most to distinguishing AI-generated text from human-written text.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Importance")
    if os.path.exists('app/assets/feature_importance.png'):
        st.image('app/assets/feature_importance.png', use_column_width=True)
    else:
        st.info("Feature importance plot not found. Run the interpretation notebook first.")

with col2:
    st.subheader("SHAP Summary")
    if os.path.exists('app/assets/shap_summary.png'):
        st.image('app/assets/shap_summary.png', use_column_width=True)
    else:
        st.info("SHAP summary plot not found. Run the interpretation notebook first.")

st.markdown("---")

st.subheader("Top Features for AI Detection")

if os.path.exists('reports/feature_importance.csv'):
    features_df = pd.read_csv('reports/feature_importance.csv')
    st.dataframe(features_df.head(20), use_container_width=True)
else:
    st.info("Feature importance data not found. Run the interpretation notebook first.")

st.markdown("---")

st.subheader("SHAP Dependence Plot")
if os.path.exists('app/assets/shap_dependence.png'):
    st.image('app/assets/shap_dependence.png', use_column_width=True)
else:
    st.info("Dependence plot not found. Run the interpretation notebook first.")

st.markdown("---")

st.info("""
**Key Insights:**
- SHAP values reveal which linguistic patterns distinguish AI from human text
- Positive SHAP values push prediction towards "AI"
- Negative SHAP values push prediction towards "Human"
- Feature importance shows which features have highest average impact
""")

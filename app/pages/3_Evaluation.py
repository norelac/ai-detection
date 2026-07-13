import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Evaluation", page_icon="📈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: var(--background-color);
    }
    
    .page-header {
        color: var(--text-color);
        font-weight: 700;
        font-size: 2em;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1em;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .section-title {
        color: var(--text-color);
        font-size: 1.4em;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2563eb;
    }
    
    .stat-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    
    .image-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--border-color);
    }
    
    .info-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #2563eb;
        color: var(--text-color);
        margin: 1rem 0;
    }
    
    .info-title {
        color: var(--text-color);
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .info-content {
        color: var(--text-color);
        opacity: 0.8;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-header'>📈 Model Evaluation</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-subtitle'>Metrik performa model dan perbandingan algoritma</p>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🎯 Model Performance (XGBoost)</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.metric("F1-Score", "98.52%", "+2.4% vs Baseline")

with col2:
    st.metric("ROC-AUC", "99.88%", "+1.3% vs Baseline")

with col3:
    st.metric("Accuracy", "98.86%", "+1.8% vs Baseline")

with col4:
    st.metric("Recall", "98.52%", "+1.9% vs Baseline")

st.markdown("<h2 class='section-title'>📊 Confusion Matrix & ROC Curve</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.write("**Confusion Matrix**")
    if os.path.exists('app/assets/confusion_matrix.png'):
        st.image('app/assets/confusion_matrix.png', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.write("**ROC Curve**")
    if os.path.exists('app/assets/roc_curve.png'):
        st.image('app/assets/roc_curve.png', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📈 Model Comparison</h2>", unsafe_allow_html=True)

if os.path.exists('reports/model_comparison.csv'):
    comparison_df = pd.read_csv('reports/model_comparison.csv')
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.write("**Performance Comparison**")
        st.dataframe(comparison_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-title'>🏆 Model Selection</div>
            <div class='info-content'>
            <strong>XGBoost</strong> dipilih sebagai model utama karena:<br><br>
            1. <strong>Superior Performance</strong>: F1 98.52% vs 98.21% (LogReg)<br>
            2. <strong>Better Robustness</strong>: Menangani noise data lebih baik<br>
            3. <strong>Feature Importance</strong>: SHAP interpretability<br>
            4. <strong>Non-linearity</strong>: Capture complex patterns
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📝 Key Insights</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class='stat-card'>
        <p style='font-weight: 700; margin-bottom: 15px; color: var(--text-color);'>✅ Kelebihan Model</p>
        <ul style='margin-left: 20px; line-height: 1.8; color: var(--text-color); opacity: 0.8;'>
            <li>F1-Score sangat tinggi (98.52%)</li>
            <li>ROC-AUC luar biasa (99.88%)</li>
            <li>Recall tinggi mengurangi false negatives</li>
            <li>Konsisten di seluruh metrik evaluasi</li>
            <li>Robust terhadap variasi teks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='stat-card'>
        <p style='font-weight: 700; margin-bottom: 15px; color: var(--text-color);'>⚠️ Limitasi Model</p>
        <ul style='margin-left: 20px; line-height: 1.8; color: var(--text-color); opacity: 0.8;'>
            <li>Proses training memerlukan resources lebih</li>
            <li>Sensitif terhadap hyperparameter tuning</li>
            <li>Kurang akurat pada teks sangat pendek (<50 karakter)</li>
            <li>Kemungkinan misklasifikasi pada teks campuran (hybrid)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='info-card'>
    <div class='info-title'>🔍 Catatan Evaluasi</div>
    <div class='info-content'>
    <strong>Dataset Split:</strong> 70% Train, 15% Validation, 15% Test<br>
    <strong>Primary Metric:</strong> F1-Score (balanced precision & recall)<br>
    <strong>Threshold:</strong> 0.5 untuk binary classification<br>
    <strong>Random Seed:</strong> 42 untuk reproducibility<br>
    <strong>Cross-validation:</strong> 3-fold untuk hyperparameter tuning
    </div>
</div>
""", unsafe_allow_html=True)

if not os.path.exists('reports/model_comparison.csv'):
    st.info("⚠️ Data perbandingan model tidak ditemukan. Jalankan modeling notebook terlebih dahulu.")

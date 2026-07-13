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

st.markdown("<h1 class='page-header'>📈 Model Evaluation & Robustness</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-subtitle'>Metrik performa model, perbandingan algoritma, dan uji ketahanan (robustness)</p>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📊 Model Performance Comparison</h2>", unsafe_allow_html=True)

if os.path.exists('reports/model_comparison.csv'):
    comparison_df = pd.read_csv('reports/model_comparison.csv')
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.write("**Test Set Performance (TF-IDF + Char N-gram + Text Statistics)**")
        st.dataframe(comparison_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-title'>🏆 Model Selection</div>
            <div class='info-content'>
            Model terbaik dipilih berdasarkan **F1-Score** tertinggi pada data test.<br><br>
            <strong>Features Used:</strong><br>
            1. TF-IDF (5,000 features)<br>
            2. Character N-grams (3,000 features)<br>
            3. Text Statistics (12 features)<br><br>
            <strong>Total:</strong> 8,012 features
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📊 Confusion Matrix & ROC Curve</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.write("**Confusion Matrix (Best Model)**")
    if os.path.exists('app/assets/confusion_matrix.png'):
        st.image('app/assets/confusion_matrix.png', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.write("**ROC Curve Comparison**")
    if os.path.exists('app/assets/roc_curve.png'):
        st.image('app/assets/roc_curve.png', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🛡️ Robustness Test (AI Polishing Simulation)</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='info-card'>
    <div class='info-title'>💡 Apa itu Robustness Test?</div>
    <div class='info-content'>
    Kami menguji kehandalan model pada teks AI yang sudah "dihaluskan" oleh manusia. 
    Ini disimulasikan dengan menyuntikkan kesalahan ketik (typo) atau mengubah karakter pada tingkat tertentu (0% hingga 20%).
    Model yang baik seharusnya tetap bisa mendeteksi teks AI meskipun ada sedikit modifikasi manusia.
    </div>
</div>
""", unsafe_allow_html=True)

if os.path.exists('reports/robustness_comparison.csv'):
    robustness_df = pd.read_csv('reports/robustness_comparison.csv')
    
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.write("**Recall (AI Detection) vs Perturbation Rate**")
        st.dataframe(robustness_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-title'>🔍 Analisis Hasil</div>
            <div class='info-content'>
            **Logistic Regression & XGBoost** menunjukkan ketahanan terbaik. 
            Bahkan dengan modifikasi teks sebesar 20%, model LogReg tetap mempertahankan recall > 96%, 
            yang sangat krusial untuk mencegah AI lolos deteksi di dunia akademik.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class='info-card'>
    <div class='info-title'>🔍 Catatan Evaluasi</div>
    <div class='info-content'>
    <strong>Dataset Split:</strong> 70% Train, 15% Validation, 15% Test<br>
    <strong>Primary Metric:</strong> F1-Score (balanced precision & recall)<br>
    <strong>Robustness Metric:</strong> Recall on Perturbed AI Texts<br>
    <strong>Random Seed:</strong> 42 untuk reproducibility
    </div>
</div>
""", unsafe_allow_html=True)

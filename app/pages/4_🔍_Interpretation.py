import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Interpretation", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: #f8f9fa;
    }
    
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    .page-header {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 2em;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        color: #666666;
        font-size: 1em;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .section-title {
        color: #1a1a1a;
        font-size: 1.4em;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2563eb;
    }
    
    .info-card {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-title {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-text {
        color: #4a4a4a;
        line-height: 1.6;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e8e8;
    }
    
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e8e8;
    }
    
    .feature-bar {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
        border: 1px solid #e8e8e8;
    }
    
    .bar-fill {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        border-radius: 4px;
        height: 8px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-header'>🔍 Model Interpretation with SHAP</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-subtitle'>Memahami fitur-fitur yang paling berpengaruh dalam deteksi AI vs Human text</p>", unsafe_allow_html=True)

st.markdown("""
<div class='info-card'>
    <div class='info-title'>💡 Apa itu SHAP?</div>
    <div class='info-text'>
    SHAP (SHapley Additive exPlanations) adalah pendekatan berbasis game theory untuk menjelaskan prediksi model machine learning. SHAP menunjukkan fitur mana yang paling berkontribusi dalam membedakan teks AI dari teks manusia.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📊 Feature Importance Analysis</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.write("**Top 20 Most Important Features**")
    if os.path.exists('app/assets/feature_importance.png'):
        st.image('app/assets/feature_importance.png', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
    st.write("**SHAP Summary Plot**")
    if os.path.exists('app/assets/shap_summary.png'):
        st.image('app/assets/shap_summary.png', use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📝 Feature Impact Details</h2>", unsafe_allow_html=True)

if os.path.exists('reports/feature_importance.csv'):
    features_df = pd.read_csv('reports/feature_importance.csv')
    
    st.write("**Top 15 Most Important Features:**")
    
    for i in range(min(15, len(features_df))):
        row = features_df.iloc[i]
        feature_name = row['feature']
        importance = row['importance']
        
        bar_width = min(100, (importance / features_df['importance'].max()) * 100)
        
        st.markdown(f"""
        <div class='feature-bar'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                <span style='color: #1a1a1a; font-weight: 600;'>{feature_name}</span>
                <span style='color: #2563eb; font-weight: 700;'>{importance:.4f}</span>
            </div>
            <div style='background: #e8e8e8; border-radius: 4px; height: 8px;'>
                <div class='bar-fill' style='width: {bar_width}%;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📈 SHAP Dependence Plot</h2>", unsafe_allow_html=True)

st.markdown("<div class='image-card'>", unsafe_allow_html=True)
st.write("**Dependence Plot (Top Feature)**")
if os.path.exists('app/assets/shap_dependence.png'):
    st.image('app/assets/shap_dependence.png', use_column_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>💡 Key Insights</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class='stat-card'>
        <p style='font-weight: 700; margin-bottom: 15px; color: #1a1a1a;'>🎯 High Impact Features</p>
        <ul style='margin-left: 20px; line-height: 1.8; color: #4a4a4a;'>
            <li><strong>N-grams</strong> paling penting untuk deteksi</li>
            <li><strong>Word combinations</strong> menunjukkan pattern AI</li>
            <li><strong>Punctuation patterns</strong> berbeda AI vs manusia</li>
            <li><strong>Content-specific features</strong> sangat diskriminatif</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='stat-card'>
        <p style='font-weight: 700; margin-bottom: 15px; color: #1a1a1a;'>🔍 Model Logic</p>
        <ul style='margin-left: 20px; line-height: 1.8; color: #4a4a4a;'>
            <li>AI text lebih <strong>konsisten</strong> dalam struktur</li>
            <li>Human text lebih <strong>bervariasi</strong> dalam gaya</li>
            <li>AI lebih formal dengan <strong>pola repetitif</strong></li>
            <li>Human writing lebih <strong>spontan</strong> dan natural</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='info-card'>
    <div class='info-title'>ℹ️ Cara Membaca SHAP Plots</div>
    <div class='info-text'>
    <strong>Positive SHAP values</strong> → Mendorong prediksi ke arah "AI"<br>
    <strong>Negative SHAP values</strong> → Mendorong prediksi ke arah "Human"<br>
    <strong>Bar width</strong> → Menunjukkan kekuatan pengaruh fitur<br>
    <strong>Color (red/blue)</strong> → Nilai fitur tinggi (merah) atau rendah (biru)
    </div>
</div>
""", unsafe_allow_html=True)

if not os.path.exists('reports/feature_importance.csv'):
    st.warning("⚠️ Data feature importance tidak ditemukan. Jalankan interpretation notebook terlebih dahulu.")

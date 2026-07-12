import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: #f8f9fa;
        padding: 2rem;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #f8f9fa;
    }
    
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    .main-header {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 2.5em;
        margin-bottom: 0.3rem;
    }
    
    .sub-header {
        color: #666666;
        font-size: 1.1em;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e8e8;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #2563eb;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #666666;
        font-size: 0.95em;
        font-weight: 500;
    }
    
    .metric-icon {
        font-size: 2.5em;
        margin-bottom: 0.5rem;
    }
    
    .section-title {
        color: #1a1a1a;
        font-size: 1.5em;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2563eb;
    }
    
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .info-box-title {
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.1em;
        margin-bottom: 0.8rem;
    }
    
    .info-box-content {
        color: #4a4a4a;
        line-height: 1.8;
    }
    
    .stMetric {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e8e8e8;
    }
    
    .stMetric label {
        color: #666666 !important;
        font-weight: 500 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    
    .divider {
        height: 1px;
        background: #e0e0e0;
        margin: 2rem 0;
    }
    
    .stButton>button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/best_model.pkl')
        preprocessing = joblib.load('models/preprocessing.pkl')
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    st.markdown("<h1 class='main-header'>🔍 AI Text Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Sistem deteksi teks AI-generated dengan machine learning dan akurasi tinggi</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center;'>
                <div class='metric-icon'>📊</div>
                <div class='metric-value'>44,868</div>
                <div class='metric-label'>Dataset Essays</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center;'>
                <div class='metric-icon'>🤖</div>
                <div class='metric-value'>XGBoost</div>
                <div class='metric-label'>Model Utama</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='text-align: center;'>
                <div class='metric-icon'>⚡</div>
                <div class='metric-value'>98.86%</div>
                <div class='metric-label'>Akurasi Test</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-title'>Performa Model</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        st.metric("F1-Score", "98.52%", "+1.2%")
    
    with col2:
        st.metric("ROC-AUC", "99.88%", "+0.5%")
    
    with col3:
        st.metric("Precision", "98.52%", "+0.8%")
    
    with col4:
        st.metric("Recall", "98.52%", "+1.0%")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <div class='info-box-title'>💡 Cara Menggunakan Aplikasi</div>
        <div class='info-box-content'>
        <strong>1. Dashboard</strong> - Lihat analisis data dan statistik lengkap<br>
        <strong>2. Model Demo</strong> - Input teks custom dan dapatkan prediksi real-time<br>
        <strong>3. Evaluasi</strong> - Review performa model dengan confusion matrix dan ROC curve<br>
        <strong>4. Interpretasi</strong> - Pahami fitur penting menggunakan SHAP analysis<br>
        <strong>5. Dokumentasi</strong> - Baca metodologi dan penjelasan teknis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #1a1a1a; margin-bottom: 1rem;'>📈 Dataset Information</h3>
            <table style='width: 100%; color: #4a4a4a;'>
                <tr style='border-bottom: 1px solid #e8e8e8;'>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>Total Samples</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>44,868</td>
                </tr>
                <tr style='border-bottom: 1px solid #e8e8e8;'>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>Human Text</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>27,371 (61.0%)</td>
                </tr>
                <tr style='border-bottom: 1px solid #e8e8e8;'>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>AI Text</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>17,497 (39.0%)</td>
                </tr>
                <tr>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>Source</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>DAIGT V2 (Kaggle)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #1a1a1a; margin-bottom: 1rem;'>🛠️ Tech Stack</h3>
            <table style='width: 100%; color: #4a4a4a;'>
                <tr style='border-bottom: 1px solid #e8e8e8;'>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>ML Framework</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>XGBoost, Scikit-learn</td>
                </tr>
                <tr style='border-bottom: 1px solid #e8e8e8;'>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>Features</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>TF-IDF + Char n-grams</td>
                </tr>
                <tr style='border-bottom: 1px solid #e8e8e8;'>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>Interpretation</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>SHAP</td>
                </tr>
                <tr>
                    <td style='padding: 0.5rem 0; font-weight: 500;'>Deployment</td>
                    <td style='padding: 0.5rem 0; text-align: right;'>Streamlit Cloud</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    model, preprocessing = load_models()
    main()

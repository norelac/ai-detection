import streamlit as st

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

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
    
    .doc-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e8e8e8;
        margin: 1.5rem 0;
    }
    
    .doc-title {
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.2em;
        margin-bottom: 1rem;
    }
    
    .doc-text {
        color: #4a4a4a;
        line-height: 1.8;
    }
    
    .tech-badge {
        background: #2563eb;
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9em;
        display: inline-block;
        margin: 0.3rem;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        text-align: center;
        padding: 1rem;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #2563eb;
    }
    
    .metric-label {
        color: #666666;
        font-size: 0.9em;
        margin-top: 0.3rem;
    }
    
    .code-box {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 1.5rem;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    
    .info-section {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-header'>📚 Project Documentation</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-subtitle'>Panduan lengkap sistem AI Text Detection</p>", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🎯 Project Overview</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='doc-card'>
    <h3 class='doc-title'>AI-Generated Text Detection System</h3>
    <p class='doc-text'>
    Sistem machine learning untuk mendeteksi apakah sebuah teks ditulis oleh manusia atau dihasilkan oleh AI (Large Language Models). Dengan akurasi <strong>98.86%</strong>, sistem ini membantu menjaga integritas akademik dan transparansi konten digital.
    </p>
    <div style='margin-top: 1.5rem;'>
        <span class='tech-badge'>Machine Learning</span>
        <span class='tech-badge'>NLP</span>
        <span class='tech-badge'>Binary Classification</span>
        <span class='tech-badge'>XGBoost</span>
        <span class='tech-badge'>SHAP</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📊 Dataset Information</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class='doc-card'>
        <h4 class='doc-title'>📁 DAIGT V2 Dataset</h4>
        <p style='margin: 0.5rem 0; color: #4a4a4a;'><strong>Total Samples:</strong> 44,868 essays</p>
        <p style='margin: 0.5rem 0; color: #4a4a4a;'><strong>Human Written:</strong> 27,371 (61.0%)</p>
        <p style='margin: 0.5rem 0; color: #4a4a4a;'><strong>AI Generated:</strong> 17,497 (39.0%)</p>
        <p style='margin: 0.5rem 0; color: #4a4a4a;'><strong>Split:</strong> 70% Train, 15% Val, 15% Test</p>
        <p style='margin: 0.5rem 0; color: #4a4a4a;'><strong>Source:</strong> <a href='https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset' target='_blank' style='color: #2563eb;'>Kaggle - DAIGT V2</a></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='doc-card'>
        <h4 class='doc-title'>🔍 Data Characteristics</h4>
        <ul style='margin-left: 20px; line-height: 1.8; color: #4a4a4a;'>
            <li>Multiple AI sources (GPT, Claude, dll)</li>
            <li>Diverse essay topics</li>
            <li>Varied writing styles</li>
            <li>Real-world academic context</li>
            <li>No missing values or duplicates</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>⚙️ Methodology</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='doc-card'>
    <h4 class='doc-title'>Feature Engineering Pipeline</h4>
    <div class='info-section'>
        <p style='margin: 0.8rem 0; color: #1a1a1a; font-weight: 600;'>1. TF-IDF Vectorization (Word-level)</p>
        <ul style='margin-left: 20px; color: #4a4a4a;'>
            <li>5,000 features | N-grams (1-2)</li>
            <li>Captures word frequency patterns</li>
            <li>Stop words removed</li>
        </ul>
        
        <p style='margin: 0.8rem 0; color: #1a1a1a; font-weight: 600;'>2. Character N-grams</p>
        <ul style='margin-left: 20px; color: #4a4a4a;'>
            <li>3,000 features | N-grams (2-4)</li>
            <li>Captures character-level patterns</li>
            <li>Robust to typos and variations</li>
        </ul>
        
        <p style='margin: 0.8rem 0; color: #1a1a1a; font-weight: 600;'>3. Total: 8,000 Combined Features</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🤖 Model Architecture</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class='doc-card'>
        <h4 class='doc-title'>📊 Baseline: Logistic Regression</h4>
        <ul style='margin-left: 20px; line-height: 1.8; color: #4a4a4a;'>
            <li>Fast training & inference</li>
            <li>Highly interpretable</li>
            <li>F1-Score: 99.21%</li>
            <li>ROC-AUC: 99.93%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='doc-card'>
        <h4 class='doc-title'>🏆 Primary: XGBoost</h4>
        <ul style='margin-left: 20px; line-height: 1.8; color: #4a4a4a;'>
            <li>Gradient boosting algorithm</li>
            <li>Handles non-linearity well</li>
            <li>F1-Score: 98.52%</li>
            <li>ROC-AUC: 99.88%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📈 Performance Metrics</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='doc-card'>
    <div class='metric-grid'>
        <div class='metric-item'>
            <div class='metric-value'>98.52%</div>
            <div class='metric-label'>F1-Score</div>
        </div>
        <div class='metric-item'>
            <div class='metric-value'>99.88%</div>
            <div class='metric-label'>ROC-AUC</div>
        </div>
        <div class='metric-item'>
            <div class='metric-value'>98.86%</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        <div class='metric-item'>
            <div class='metric-value'>98.52%</div>
            <div class='metric-label'>Recall</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🛠️ Tech Stack</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class='doc-card'>
        <h4 class='doc-title'>🐍 Python Libraries</h4>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Pandas, NumPy</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Scikit-learn</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• XGBoost</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• SHAP</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='doc-card'>
        <h4 class='doc-title'>📊 Visualization</h4>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Matplotlib</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Seaborn</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Plotly</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• WordCloud</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='doc-card'>
        <h4 class='doc-title'>🚀 Deployment</h4>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Streamlit</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Joblib</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• GitHub</p>
        <p style='color: #4a4a4a; margin: 0.3rem 0;'>• Streamlit Cloud</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>🚀 Quick Start Guide</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='doc-card'>
    <h4 class='doc-title'>Installation & Setup</h4>
    <div class='code-box'>
git clone https://github.com/username/ai-text-detection.git<br>
cd ai-text-detection<br>
pip install -r requirements.txt<br>
streamlit run app/app.py
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 class='section-title'>📞 Resources</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='doc-card'>
    <h4 class='doc-title'>📚 References & Links</h4>
    <p style='margin: 0.8rem 0; color: #4a4a4a;'><strong>Dataset:</strong> <a href='https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset' target='_blank' style='color: #2563eb;'>DAIGT V2 - Kaggle</a></p>
    <p style='margin: 0.8rem 0; color: #4a4a4a;'><strong>XGBoost:</strong> <a href='https://xgboost.readthedocs.io/' target='_blank' style='color: #2563eb;'>Documentation</a></p>
    <p style='margin: 0.8rem 0; color: #4a4a4a;'><strong>SHAP:</strong> <a href='https://shap.readthedocs.io/' target='_blank' style='color: #2563eb;'>Documentation</a></p>
    <p style='margin: 0.8rem 0; color: #4a4a4a;'><strong>Streamlit:</strong> <a href='https://docs.streamlit.io/' target='_blank' style='color: #2563eb;'>Documentation</a></p>
</div>
""", unsafe_allow_html=True)

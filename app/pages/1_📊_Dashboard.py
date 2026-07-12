import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Dashboard EDA", page_icon="📊", layout="wide")

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
    
    .stMetric {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e8e8e8;
    }
    
    .insight-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    
    .insight-title {
        color: #1a1a1a;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .insight-content {
        color: #4a4a4a;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-header'>📊 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-subtitle'>Analisis mendalam tentang struktur dan karakteristik dataset DAIGT V2</p>", unsafe_allow_html=True)

@st.cache_data
def load_eda_data():
    try:
        df = pd.read_csv('data/processed/train.csv')
        return df
    except:
        return None

df = load_eda_data()

if df is not None:
    st.markdown("<h2 class='section-title'>📈 Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.metric("📁 Total Samples", f"{len(df):,}")
    
    with col2:
        human_count = (df['label'] == 0).sum()
        st.metric("👤 Human Text", f"{human_count:,}", f"{human_count/len(df)*100:.1f}%")
    
    with col3:
        ai_count = (df['label'] == 1).sum()
        st.metric("🤖 AI Text", f"{ai_count:,}", f"{ai_count/len(df)*100:.1f}%")
    
    st.markdown("<h2 class='section-title'>📊 Visualisasi Data</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.write("**Distribusi Class**")
        if os.path.exists('app/assets/class_distribution.png'):
            st.image('app/assets/class_distribution.png', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.write("**Analisis Panjang Text**")
        if os.path.exists('app/assets/text_length_analysis.png'):
            st.image('app/assets/text_length_analysis.png', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-title'>🎨 Pola Linguistik</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.write("**Word Cloud Comparison**")
        if os.path.exists('app/assets/wordcloud.png'):
            st.image('app/assets/wordcloud.png', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.write("**N-gram Analysis**")
        if os.path.exists('app/assets/ngram_analysis.png'):
            st.image('app/assets/ngram_analysis.png', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if os.path.exists('app/assets/source_distribution.png'):
        st.markdown("<h2 class='section-title'>📍 Distribusi Sumber Data</h2>", unsafe_allow_html=True)
        st.markdown("<div class='image-card'>", unsafe_allow_html=True)
        st.image('app/assets/source_distribution.png', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='section-title'>💡 Key Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>✍️ Human Written Text</div>
            <div class='insight-content'>
            • Lebih variatif dalam pemilihan kosakata<br>
            • Gaya penulisan kurang konsisten (lebih natural)<br>
            • Rata-rata 2,349 karakter per essay<br>
            • Lebih banyak fluktuasi dalam panjang kalimat<br>
            • Lebih banyak kesalahan grammar yang natural
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='insight-box'>
            <div class='insight-title'>🤖 AI Generated Text</div>
            <div class='insight-content'>
            • Konsistensi tinggi dalam struktur dan panjang<br>
            • Kosakata lebih formal dengan pola repetitif<br>
            • Rata-rata 2,009 karakter per essay<br>
            • N-gram patterns yang distinctive<br>
            • Grammar sempurna dan penulisan terstruktur
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("⚠️ Data tidak ditemukan. Jalankan EDA notebook terlebih dahulu.")

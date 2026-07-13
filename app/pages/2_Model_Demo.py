import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
import plotly.graph_objects as go

st.set_page_config(page_title="Model Demo", page_icon="🎯", layout="wide")

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
        margin-bottom: 1.5rem;
    }
    
    .section-title {
        color: var(--text-color);
        font-size: 1.3em;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    .info-box {
        background-color: rgba(37, 99, 235, 0.1);
        border-left: 4px solid #2563eb;
        padding: 1.2rem;
        border-radius: 8px;
        color: var(--text-color);
        margin: 1rem 0;
    }
    
    .info-title {
        color: var(--text-color);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-text {
        color: var(--text-color);
        opacity: 0.9;
        font-size: 0.95em;
    }
    
    .result-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--border-color);
        text-align: center;
        margin: 1.5rem 0;
        color: var(--text-color);
    }
    
    .result-human {
        border-top: 4px solid #16a34a;
    }
    
    .result-ai {
        border-top: 4px solid #ea580c;
    }
    
    .result-label {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
    }
    
    .result-title {
        font-size: 1.8em;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-human .result-title {
        color: #16a34a;
    }
    
    .result-ai .result-title {
        color: #ea580c;
    }
    
    .confidence-box {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--border-color);
        text-align: center;
        color: var(--text-color);
    }
    
    .confidence-label {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.9em;
        margin-bottom: 0.5rem;
    }
    
    .confidence-value {
        font-size: 2.2em;
        font-weight: 700;
        color: #2563eb;
    }
    
    .stButton>button {
        background: #2563eb;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.8rem 1.5rem;
    }
    
    .stButton>button:hover {
        background: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='page-header'>🎯 Real-time Prediction Demo</h1>", unsafe_allow_html=True)
st.markdown("<p class='page-subtitle'>Masukkan teks dan dapatkan prediksi apakah ditulis oleh manusia atau AI</p>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/best_model.pkl')
        preprocessing = joblib.load('models/preprocessing.pkl')
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, preprocessing = load_models()

if model and preprocessing:
    st.markdown("""
    <div class='info-box'>
        <div class='info-title'>💡 Cara Menggunakan</div>
        <div class='info-text'>
        Copy-paste essay, artikel, atau teks apapun. Model akan menganalisis pola linguistik seperti struktur kalimat, variasi kosakata, dan n-gram untuk menentukan apakah teks tersebut ditulis oleh manusia atau AI.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='section-title'>✍️ Input Text</h3>", unsafe_allow_html=True)
    
    user_text = st.text_area(
        "Masukkan teks untuk dianalisis:",
        placeholder="Ketik atau paste teks Anda di sini... (minimal 50 karakter untuk akurasi optimal)",
        height=250,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analisis Text", use_container_width=True)
    
    if analyze_button:
        if user_text.strip():
            if len(user_text) < 50:
                st.warning("⚠️ Text terlalu pendek. Masukkan minimal 50 karakter untuk hasil yang lebih akurat.")
            else:
                try:
                    tfidf = preprocessing['word_tfidf']
                    char_tfidf = preprocessing['char_tfidf']
                    
                    X_tfidf = tfidf.transform([user_text])
                    X_char = char_tfidf.transform([user_text])
                    X_combined = hstack([X_tfidf, X_char])
                    
                    prediction = model.predict(X_combined)[0]
                    probability = model.predict_proba(X_combined)[0]
                    
                    st.markdown("<h3 class='section-title'>📊 Hasil Analisis</h3>", unsafe_allow_html=True)
                    
                    if prediction == 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class='result-card result-human'>
                                <div class='result-label'>Hasil Prediksi</div>
                                <div class='result-title'>✅ HUMAN WRITTEN</div>
                                <div style='opacity: 0.9; font-size: 0.95em;'>Teks ini kemungkinan besar ditulis oleh manusia</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            confidence = probability[0] * 100
                            st.markdown(f"""
                            <div class='confidence-box'>
                                <div class='confidence-label'>Confidence</div>
                                <div class='confidence-value'>{confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class='result-card result-ai'>
                                <div class='result-label'>Hasil Prediksi</div>
                                <div class='result-title'>🤖 AI GENERATED</div>
                                <div style='opacity: 0.9; font-size: 0.95em;'>Teks ini kemungkinan besar dihasilkan oleh AI</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            confidence = probability[1] * 100
                            st.markdown(f"""
                            <div class='confidence-box'>
                                <div class='confidence-label'>Confidence</div>
                                <div class='confidence-value'>{confidence:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("<h3 class='section-title'>📈 Probability Distribution</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2, gap="medium")
                    
                    with col1:
                        st.metric("👤 Human", f"{probability[0]*100:.2f}%")
                    
                    with col2:
                        st.metric("🤖 AI", f"{probability[1]*100:.2f}%")
                    
                    fig = go.Figure(data=[
                        go.Bar(x=['Human', 'AI'], y=[probability[0]*100, probability[1]*100], 
                               marker_color=['#16a34a', '#ea580c'],
                               text=[f'{probability[0]*100:.1f}%', f'{probability[1]*100:.1f}%'],
                               textposition='outside')
                    ])
                    
                    # Adapt chart color to theme (dark or light grid lines)
                    fig.update_layout(
                        height=300,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='gray', family='Inter'),
                        yaxis_title="Probability (%)",
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class='info-box'>
                        <div class='info-title'>ℹ️ Catatan Penting</div>
                        <div class='info-text'>
                        Model ini dilatih dengan 44,868 essay dengan akurasi 98.86%. Prediksi dapat dipengaruhi oleh gaya penulisan, grammar, dan struktur kalimat. Hasil terbaik pada text dengan panjang >500 karakter.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")

else:
    st.error("❌ Model tidak berhasil dimuat. Pastikan file model sudah tersimpan dengan benar.")

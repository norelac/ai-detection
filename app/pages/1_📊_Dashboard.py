import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis Dashboard")
st.markdown("---")

@st.cache_data
def load_eda_data():
    try:
        df = pd.read_csv('data/processed/train.csv')
        return df
    except:
        return None

df = load_eda_data()

if df is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    
    with col2:
        human_count = (df['label'] == 0).sum()
        st.metric("Human Text", f"{human_count:,} ({human_count/len(df)*100:.1f}%)")
    
    with col3:
        ai_count = (df['label'] == 1).sum()
        st.metric("AI Text", f"{ai_count:,} ({ai_count/len(df)*100:.1f}%)")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        if os.path.exists('app/assets/class_distribution.png'):
            st.image('app/assets/class_distribution.png', use_column_width=True)
    
    with col2:
        st.subheader("Text Length Analysis")
        if os.path.exists('app/assets/text_length_analysis.png'):
            st.image('app/assets/text_length_analysis.png', use_column_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word Cloud")
        if os.path.exists('app/assets/wordcloud.png'):
            st.image('app/assets/wordcloud.png', use_column_width=True)
    
    with col2:
        st.subheader("N-gram Analysis")
        if os.path.exists('app/assets/ngram_analysis.png'):
            st.image('app/assets/ngram_analysis.png', use_column_width=True)
    
    if os.path.exists('app/assets/source_distribution.png'):
        st.markdown("---")
        st.subheader("Source Distribution")
        st.image('app/assets/source_distribution.png', use_column_width=True)

else:
    st.warning("Data not found. Please run the EDA notebook first.")

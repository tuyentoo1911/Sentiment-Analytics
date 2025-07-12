#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from sentiment_analysis_system import SentimentAnalysisSystem

# Cấu hình trang
st.set_page_config(
    page_title="Phân Tích Cảm Xúc Đánh Giá",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_system():
    """Load hệ thống phân tích cảm xúc (cached)"""
    try:
        system = SentimentAnalysisSystem()
        return system
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None

def render_sentiment_result(result):
    """Hiển thị kết quả phân tích"""
    if 'error' in result:
        st.error(f"Lỗi: {result['error']}")
        return
    
    # Tạo layout columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_color = "positive" if result['sentiment_label'] == ' Positive' else "negative" if result['sentiment_label'] == ' Negative' else "neutral"
        sentiment_percent = ((result['sentiment_score'] + 1) / 2) * 100  
        sentiment_emoji = "😊" if result['sentiment_label'] == 'Positive' else "😢" if result['sentiment_label'] == 'Negative' else "😐"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Cảm Xúc</h4>
            <h2 class="{sentiment_color}">{sentiment_emoji} {result['sentiment_label']}</h2>
            <p>Điểm: {sentiment_percent:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Rating Dự Đoán</h4>
            <h2>⭐ {result['predicted_rating']}</h2>
            <p>Cường độ: {result['emotion_intensity'] * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Điểm Hữu Ích</h4>
            <h2>📊 {result['helpfulness_score'] * 100:.1f}%</h2>
            <p>Độ dài: {result['text_length']} ký tự</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chi tiết sentiment
    st.subheader("Chi Tiết Phân Tích")
    detailed = result['detailed_sentiment']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**VADER Sentiment:**")
        st.write(f" ☺️ Positive: {detailed['vader_positive'] * 100:.1f}%")
        st.write(f" 😥 Negative: {detailed['vader_negative'] * 100:.1f}%")
        st.write(f" 😐 Neutral: {detailed['vader_neutral'] * 100:.1f}%")
        st.write(f" 🤔 Compound: {((detailed['vader_compound'] + 1) / 2) * 100:.1f}%")
    
    with col2:
        st.write("**TextBlob Analysis:**")
        st.write(f" 💭 Polarity: {((detailed['textblob_polarity'] + 1) / 2) * 100:.1f}%")
        st.write(f" 🧠 Subjectivity: {detailed['textblob_subjectivity'] * 100:.1f}%")
    
    # Biểu đồ
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('VADER Components', 'TextBlob Analysis'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # VADER chart - chuyển sang %
    fig.add_trace(
        go.Bar(
            x=['Positive', 'Negative', 'Neutral'],
            y=[detailed['vader_positive'] * 100, detailed['vader_negative'] * 100, detailed['vader_neutral'] * 100],
            name='VADER',
            marker_color=['green', 'red', 'gray'],
            text=[f"{detailed['vader_positive'] * 100:.1f}%", 
                  f"{detailed['vader_negative'] * 100:.1f}%", 
                  f"{detailed['vader_neutral'] * 100:.1f}%"],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # TextBlob chart - chuyển sang %
    fig.add_trace(
        go.Bar(
            x=['Polarity', 'Subjectivity'],
            y=[((detailed['textblob_polarity'] + 1) / 2) * 100, detailed['textblob_subjectivity'] * 100],
            name='TextBlob',
            marker_color=['blue', 'orange'],
            text=[f"{((detailed['textblob_polarity'] + 1) / 2) * 100:.1f}%", 
                  f"{detailed['textblob_subjectivity'] * 100:.1f}%"],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_batch_results(results):
    """Hiển thị kết quả phân tích batch"""
    if not results:
        return
    
    # Tạo DataFrame
    df = pd.DataFrame(results)
    
    # Thống kê tổng hợp
    st.subheader("Thống Kê Tổng Hợp")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng Số Text", len(df))
    
    with col2:
        avg_sentiment = df['sentiment_score'].mean()
        avg_sentiment_percent = ((avg_sentiment + 1) / 2) * 100
        st.metric("Sentiment Trung Bình", f"{avg_sentiment_percent:.1f}%")
    
    with col3:
        avg_rating = df['predicted_rating'].mean()
        st.metric("Rating Trung Bình", f"{avg_rating:.1f} ⭐")
    
    with col4:
        avg_helpfulness = df['helpfulness_score'].mean()
        st.metric("Điểm Hữu Ích TB", f"{avg_helpfulness * 100:.1f}%")
    
    # Phân phối sentiment
    st.subheader("Phân Phối Cảm Xúc")
    
    sentiment_counts = df['sentiment_label'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Phân Phối Sentiment"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_hist = px.histogram(
            df, x='sentiment_score',
            title="Phân Phối Điểm Sentiment",
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Bảng kết quả
    st.subheader("Kết Quả Chi Tiết")
    
    display_df = df[['text', 'sentiment_label', 'sentiment_score', 'predicted_rating', 'helpfulness_score']].copy()
    display_df['text'] = display_df['text'].str[:100] + "..."
    
    # Chuyển đổi các điểm số thành %
    display_df['sentiment_score'] = ((display_df['sentiment_score'] + 1) / 2 * 100).round(1).astype(str) + "%"
    display_df['predicted_rating'] = display_df['predicted_rating'].astype(str) + " ⭐"
    display_df['helpfulness_score'] = (display_df['helpfulness_score'] * 100).round(1).astype(str) + "%"
    
    display_df.columns = ['Text', 'Sentiment', 'Điểm Sentiment', 'Rating', 'Điểm Hữu Ích']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label=" Tải Kết Quả CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )

def main():
    """Hàm chính của web app"""
    
    # Header
    st.markdown('<h1 class="main-header"> Hệ Thống Phân Tích Cảm Xúc Đánh Giá</h1>', unsafe_allow_html=True)
    
    # Load hệ thống
    system = load_sentiment_system()
    
    if not system or not system.pipeline:
        st.error(" Không thể load model. Vui lòng kiểm tra:")
        st.write("1. Đã chạy notebook huấn luyện")
        st.write("2. Thư mục `saved_models/` có file `complete_pipeline.pkl`")
        return
    
    # Sidebar
    st.sidebar.title(" Thông Tin Model")
    st.sidebar.write(f"**Model:** {system.pipeline['model_name']}")
    st.sidebar.write(f"**Accuracy:** {system.pipeline['accuracy']:.4f}")
    st.sidebar.write("**Tính năng:**")
    st.sidebar.write("- Phân tích cảm xúc VADER + TextBlob")
    st.sidebar.write("- Dự đoán rating")
    st.sidebar.write("- Tính điểm hữu ích")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Phân Tích Đơn Lẻ", "📊 Phân Tích Batch", "📁 Upload File"])
    
    with tab1:
        st.header("Phân Tích Text Đơn Lẻ")
        
        # Text input
        text_input = st.text_area(
            "Nhập text cần phân tích:",
            placeholder="Ví dụ: This product is amazing! Great quality and fast delivery.",
            height=100
        )
        
        # Analyze button
        if st.button("🔍 Phân Tích", type="primary"):
            if text_input.strip():
                with st.spinner("Đang phân tích..."):
                    result = system.predict_single_text(text_input)
                    render_sentiment_result(result)
            else:
                st.warning("Vui lòng nhập text cần phân tích!")
        
        # Examples
        st.subheader("Ví Dụ Mẫu")
        examples = [
            "This product is amazing! Great quality and fast delivery. I love it!",
            "Terrible product. Poor quality and bad customer service. Very disappointed.",
            "The product is okay, nothing special but does the job.",
            "Excellent coffee! Rich flavor and perfect aroma. Highly recommend!",
            "Bad taste, too expensive for what you get. Not worth it."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Ví dụ {i}: {example[:50]}..."):
                with st.spinner("Đang phân tích..."):
                    result = system.predict_single_text(example)
                    render_sentiment_result(result)
    
    with tab2:
        st.header("Phân Tích Batch")
        
        # Text input for multiple texts
        batch_input = st.text_area(
            "Nhập các text cần phân tích (mỗi dòng một text):",
            placeholder="Dòng 1: Text thứ nhất\nDòng 2: Text thứ hai\n...",
            height=200
        )
        
        if st.button("🔍 Phân Tích Batch", type="primary"):
            if batch_input.strip():
                texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
                
                if texts:
                    with st.spinner(f"Đang phân tích {len(texts)} text..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(texts):
                            result = system.predict_single_text(text)
                            results.append(result)
                            progress_bar.progress((i + 1) / len(texts))
                        
                        progress_bar.empty()
                        render_batch_results(results)
                else:
                    st.warning("Vui lòng nhập ít nhất một text!")
            else:
                st.warning("Vui lòng nhập text cần phân tích!")
    
    with tab3:
        st.header("Upload File CSV")
        
        # File uploader
        uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Xem Trước Dữ Liệu")
                st.dataframe(df.head(), use_container_width=True)
                
                # Select text column
                text_column = st.selectbox(
                    "Chọn cột chứa text:",
                    options=df.columns
                )
                
                # Limit rows for demo
                max_rows = st.slider("Số dòng tối đa để phân tích (demo):", 1, min(100, len(df)), 10)
                
                if st.button("🔍 Phân Tích File", type="primary"):
                    df_sample = df.head(max_rows)
                    
                    with st.spinner(f"Đang phân tích {len(df_sample)} dòng..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, row in df_sample.iterrows():
                            text = row[text_column]
                            result = system.predict_single_text(text)
                            result['original_index'] = i
                            results.append(result)
                            progress_bar.progress((i + 1) / len(df_sample))
                        
                        progress_bar.empty()
                        render_batch_results(results)
                        
            except Exception as e:
                st.error(f"Lỗi khi đọc file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Hệ Thống Phân Tích Cảm Xúc** - Powered by Streamlit")

if __name__ == "__main__":
    main() 
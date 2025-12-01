"""
Interactive Dashboard for E-commerce Sentiment Analysis
========================================================
A professional Streamlit dashboard showcasing sentiment analysis and aspect extraction results.

Features:
- Overall sentiment statistics
- Aspect-based insights
- Platform comparison
- Interactive visualizations
- Custom review analyzer

Project: Multilingual Customer Intelligence Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import sys

# Page configuration
st.set_page_config(
    page_title="E-commerce Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .critical-box {
        background-color: #ffe8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #e74c3c;
    }
    .success-box {
        background-color: #e8f8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load the main dataset with aspects"""
    try:
        df = pd.read_csv('processed_data/dataset_with_aspects.csv', encoding='utf-8')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please run the aspect extraction script first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

@st.cache_resource
def load_models():
    """Load the trained sentiment model and vectorizer"""
    try:
        with open('models/sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.warning(f"⚠️ Could not load models: {str(e)}")
        return None, None

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">📊 E-commerce Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multilingual Customer Intelligence Platform</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        model, vectorizer = load_models()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        ["📊 Overview", "🎯 Aspect Analysis", "🏪 Platform Insights", "💬 Review Analyzer", "📈 About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.metric("Total Reviews", f"{len(df):,}")
    st.sidebar.metric("Model Accuracy", "76.71%")
    st.sidebar.metric("Aspects Analyzed", "5")
    
    # Route to pages
    if page == "📊 Overview":
        show_overview(df)
    elif page == "🎯 Aspect Analysis":
        show_aspect_analysis(df)
    elif page == "🏪 Platform Insights":
        show_platform_insights(df)
    elif page == "💬 Review Analyzer":
        show_review_analyzer(model, vectorizer)
    elif page == "📈 About":
        show_about()

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

def show_overview(df):
    st.header("📊 Sentiment Analysis Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    sentiment_counts = df['sentiment'].value_counts()
    total = len(df)
    
    with col1:
        pos_pct = (sentiment_counts.get('positive', 0) / total * 100)
        st.metric("😊 Positive", f"{pos_pct:.1f}%", f"{sentiment_counts.get('positive', 0):,} reviews")
    
    with col2:
        neg_pct = (sentiment_counts.get('negative', 0) / total * 100)
        st.metric("😞 Negative", f"{neg_pct:.1f}%", f"{sentiment_counts.get('negative', 0):,} reviews")
    
    with col3:
        neu_pct = (sentiment_counts.get('neutral', 0) / total * 100)
        st.metric("😐 Neutral", f"{neu_pct:.1f}%", f"{sentiment_counts.get('neutral', 0):,} reviews")
    
    with col4:
        st.metric("📊 Total Reviews", f"{total:,}", "20,000")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'},
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, config={'responsive': True})
    
    with col2:
        st.subheader("Rating Distribution")
        rating_counts = df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating (Stars)', 'y': 'Number of Reviews'},
            color=rating_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, config={'responsive': True})
    
    # Language Mix
    st.subheader("📝 Language Distribution")
    lang_counts = df['language_mix'].value_counts()
    fig = px.bar(
        x=lang_counts.index,
        y=lang_counts.values,
        labels={'x': 'Language Type', 'y': 'Number of Reviews'},
        color=lang_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, config={'responsive': True})
    
    # Category Distribution
    st.subheader("📦 Category Distribution")
    category_counts = df['category'].value_counts().head(10)
    fig = px.bar(
        x=category_counts.values,
        y=category_counts.index,
        labels={'x': 'Number of Reviews', 'y': 'Category'},
        orientation='h',
        color=category_counts.values,
        color_continuous_scale='Teal'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, config={'responsive': True})

# ============================================================================
# PAGE 2: ASPECT ANALYSIS
# ============================================================================

def show_aspect_analysis(df):
    st.header("🎯 Aspect-Based Sentiment Analysis")
    
    aspects = ['product_quality', 'delivery', 'packaging', 'price', 'customer_service']
    aspect_names = ['Product Quality', 'Delivery', 'Packaging', 'Price', 'Customer Service']
    
    # Summary Cards
    st.subheader("📊 Aspect Performance Summary")
    
    cols = st.columns(5)
    
    for idx, (aspect, name) in enumerate(zip(aspects, aspect_names)):
        aspect_col = f'{aspect}_sentiment'
        aspect_data = df[df[aspect_col].notna()]
        
        if len(aspect_data) > 0:
            sentiment_counts = aspect_data[aspect_col].value_counts()
            total = len(aspect_data)
            pos_pct = (sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0
            
            with cols[idx]:
                # Determine status
                if pos_pct >= 60:
                    status = "✅"
                    color = "#2ecc71"
                elif pos_pct >= 40:
                    status = "⚠️"
                    color = "#f39c12"
                else:
                    status = "❌"
                    color = "#e74c3c"
                
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {color};">
                    <h4 style="margin: 0;">{status} {name}</h4>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {color};">{pos_pct:.1f}%</p>
                    <p style="margin: 0; font-size: 0.8rem;">{total:,} mentions</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Heatmap
    st.subheader("📊 Aspect Sentiment Heatmap")
    
    # Prepare data for heatmap
    heatmap_data = []
    for aspect, name in zip(aspects, aspect_names):
        aspect_col = f'{aspect}_sentiment'
        aspect_data = df[df[aspect_col].notna()]
        
        if len(aspect_data) > 0:
            sentiment_counts = aspect_data[aspect_col].value_counts()
            total = len(aspect_data)
            
            heatmap_data.append({
                'Aspect': name,
                'Positive': (sentiment_counts.get('positive', 0) / total * 100),
                'Neutral': (sentiment_counts.get('neutral', 0) / total * 100),
                'Negative': (sentiment_counts.get('negative', 0) / total * 100)
            })
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('Aspect')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=['Positive', 'Neutral', 'Negative'],
        y=heatmap_df.index,
        colorscale='RdYlGn',
        text=heatmap_df.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 12},
        colorbar=dict(title="Percentage")
    ))
    
    fig.update_layout(height=400, xaxis_title="Sentiment", yaxis_title="Aspect")
    st.plotly_chart(fig, config={'responsive': True})
    
    # Critical Insights
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Strengths</h4>
            <ul>
                <li><strong>Product Quality:</strong> 69.9% positive - Customers love your products!</li>
                <li><strong>Price:</strong> 61.7% positive - Good value for money perception</li>
            </ul>
            <p><strong>Action:</strong> Maintain current standards and use as marketing advantage.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="critical-box">
            <h4>❌ Critical Issues</h4>
            <ul>
                <li><strong>Customer Service:</strong> 36.8% positive (52.8% negative) - URGENT!</li>
                <li><strong>Packaging:</strong> 39.3% positive (52.3% negative) - Fix now!</li>
                <li><strong>Delivery:</strong> 45.1% positive (48.1% negative) - Needs attention</li>
            </ul>
            <p><strong>Action:</strong> Implement improvements within 1-3 months.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: PLATFORM INSIGHTS
# ============================================================================

def show_platform_insights(df):
    st.header("🏪 Platform-wise Analysis")
    
    # Platform sentiment breakdown
    platform_sentiment = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
    
    fig = go.Figure()
    
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in platform_sentiment.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=platform_sentiment.index,
                y=platform_sentiment[sentiment],
                marker_color={'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}[sentiment]
            ))
    
    fig.update_layout(
        barmode='stack',
        title="Sentiment Distribution by Platform",
        xaxis_title="Platform",
        yaxis_title="Percentage (%)",
        height=500
    )
    
    st.plotly_chart(fig, config={'responsive': True})
    
    # Platform statistics
    st.subheader("📊 Platform Statistics")
    
    platform_stats = df.groupby('platform').agg({
        'review_text': 'count',
        'rating': 'mean'
    }).round(2)
    
    platform_stats.columns = ['Total Reviews', 'Average Rating']
    platform_stats = platform_stats.sort_values('Total Reviews', ascending=False)
    
    st.dataframe(platform_stats, width='stretch')

# ============================================================================
# PAGE 4: REVIEW ANALYZER
# ============================================================================

def show_review_analyzer(model, vectorizer):
    st.header("💬 Real-Time Review Analyzer")
    
    if model is None or vectorizer is None:
        st.warning("⚠️ Model not loaded. Please ensure the sentiment model is trained and saved.")
        return
    
    st.markdown("""
    **Try it out!** Enter a review in English, Hindi, or mixed language to see the sentiment prediction.
    """)
    
    # Input
    review_text = st.text_area(
        "Enter review:",
        placeholder="e.g., Product bahut achha hai! Quality amazing aur delivery on time. Highly recommend!",
        height=100
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_text.strip():
            # Preprocess
            import re
            cleaned = review_text.lower()
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Predict
            try:
                X = vectorizer.transform([cleaned])
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                
                # Display result
                sentiment_emoji = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                sentiment_color = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                
                st.markdown(f"""
                <div style="background-color: {sentiment_color[prediction]}20; padding: 2rem; border-radius: 1rem; border-left: 6px solid {sentiment_color[prediction]}; text-align: center;">
                    <h2 style="margin: 0;">{sentiment_emoji[prediction]} {prediction.upper()}</h2>
                    <p style="font-size: 1.2rem; margin-top: 1rem;">Confidence: {max(probabilities)*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probabilities
                st.subheader("Confidence Breakdown")
                prob_df = pd.DataFrame({
                    'Sentiment': model.classes_,
                    'Probability': probabilities * 100
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Sentiment',
                    y='Probability',
                    color='Sentiment',
                    color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, config={'responsive': True})
                
            except Exception as e:
                st.error(f"❌ Error analyzing review: {str(e)}")
        else:
            st.warning("⚠️ Please enter a review to analyze.")
    
    # Sample reviews
    st.markdown("---")
    st.subheader("📝 Try These Sample Reviews")
    
    samples = [
        "Product quality amazing hai aur delivery on time Price reasonable hai",
        "Terrible service. Product kharab tha. Not satisfied at all",
        "Product okay hai. Nothing special. Average quality"
    ]
    
    for sample in samples:
        if st.button(sample, key=sample):
            st.text_area("Review:", value=sample, height=100, key=f"display_{sample}")

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

def show_about():
    st.header("📈 About This Project")
    
    st.markdown("""
    ### 🎯 Multilingual Customer Intelligence Platform
    
    
    
    This dashboard presents insights from analyzing **20,000 customer reviews** across major Indian 
    e-commerce platforms in English, Hindi, and code-mixed (Hinglish) languages.
    
    ---
    
    ### 📊 Key Achievements
    
    - ✅ **76.71% Model Accuracy** - Random Forest classifier
    - ✅ **5 Business Aspects** analyzed - Product Quality, Delivery, Packaging, Price, Customer Service
    - ✅ **20,000 Reviews** processed from Amazon, Flipkart, Myntra, Nykaa, Swiggy, Zomato
    - ✅ **Multilingual Support** - English, Hindi, Hinglish (code-mixed)
    
    ---
    
    ### 🚀 Technology Stack
    
    - **Language:** Python 3.9+
    - **ML Framework:** Scikit-learn (Random Forest, TF-IDF)
    - **Dashboard:** Streamlit + Plotly
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib, Seaborn, Plotly
    
    ---
    
    ### 💡 Business Value
    
    **ROI Projection:** 200-300% in first year
    
    **Investment Required:** ₹10-16 Lakhs
    
    **Key Insights:**
    - Customer Service & Packaging need urgent improvement (< 40% positive)
    - Product Quality is a major strength (70% positive)
    - Delivery improvements can boost customer satisfaction by 10%
    
    ---
    
    ### 📞 Contact
    
    **Platform:** Multilingual Customer Intelligence for Indian E-commerce  
    **Date:** November 2025
    """)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()

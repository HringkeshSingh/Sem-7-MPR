#!/usr/bin/env python3
"""
Data Explorer Page - Modern interface for data exploration with RAG insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import os

# Add paths
frontend_path = Path(__file__).parent.parent
sys.path.insert(0, str(frontend_path))

from components.api_client import APIClient

# Page config
st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide"
)

# API client
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
api_client = APIClient(API_BASE_URL)

# Modern styling
st.markdown("""
<style>
    .explorer-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="explorer-header">
    <h1 style="color: white; margin: 0;">📊 Healthcare Data Explorer</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Interactive data exploration with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Data Source")
    
    data_source = st.radio(
        "Select data source:",
        ["Generated Data", "Upload CSV", "Generate Sample"],
        help="Choose where to get your data from"
    )
    
    df = None
    
    if data_source == "Generated Data":
        if 'generated_data' in st.session_state and st.session_state.generated_data:
            if isinstance(st.session_state.generated_data, dict):
                data = st.session_state.generated_data.get("data", [])
                if data:
                    df = pd.DataFrame(data)
                    st.success(f"✅ Loaded {len(df)} records")
                else:
                    st.info("No data available yet")
            else:
                st.info("No data available yet")
        else:
            st.info("👈 Generate data first in the main tab")
    
    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV file", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df)} records")
    
    elif data_source == "Generate Sample":
        sample_size = st.number_input("Sample Size", 10, 200, 50)
        if st.button("Generate Sample", type="primary"):
            with st.spinner("Generating sample data..."):
                result = api_client.generate_data(
                    f"Generate {sample_size} diverse patients",
                    num_patients=sample_size
                )
                if result.get("success"):
                    df = pd.DataFrame(result.get("data", []))
                    st.success(f"✅ Generated {len(df)} records")
                    st.session_state.generated_data = result

# Main content
if df is not None and not df.empty:
    # Overview metrics
    st.subheader("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        missing_pct = (missing / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory:.1f} MB")
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Custom Analysis"])
    
    with tab1:
        st.subheader("Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                selected_num = st.selectbox("Select numeric column:", numeric_cols)
                if selected_num:
                    fig = px.histogram(df, x=selected_num, title=f"Distribution of {selected_num}", nbins=30)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                selected_cat = st.selectbox("Select categorical column:", categorical_cols)
                if selected_cat:
                    value_counts = df[selected_cat].value_counts().head(10)
                    fig = px.bar(x=value_counts.values, y=value_counts.index, 
                               orientation='h', title=f"Top 10 values in {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Correlations")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", 
                          title="Correlation Matrix", color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    with tab3:
        st.subheader("Custom Analysis")
        
        plot_type = st.selectbox("Plot Type:", ["Scatter", "Box", "Violin", "Bar"])
        
        if plot_type == "Scatter":
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", df.columns)
            with col2:
                y_col = st.selectbox("Y-axis:", df.columns)
            
            if x_col != y_col:
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box":
            y_col = st.selectbox("Value column:", numeric_cols if numeric_cols else df.columns)
            x_col = st.selectbox("Group by (optional):", [None] + list(df.columns))
            fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot: {y_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Download
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button("📊 Download CSV", csv, 
                         f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button("📋 Download JSON", json_data,
                         f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")

else:
    st.info("👆 Select a data source from the sidebar to begin exploration")
    
    # Show RAG insights if available
    if 'rag_info' in st.session_state and st.session_state.rag_info:
        st.subheader("🧠 RAG Insights Available")
        rag_info = st.session_state.rag_info
        st.write(f"**Confidence:** {rag_info.get('confidence', 0):.1%}")
        st.write(f"**Documents Found:** {rag_info.get('num_documents', 0)}")
        
        summary = rag_info.get('summary', '')
        if summary:
            with st.expander("View RAG Summary"):
                st.markdown(summary)

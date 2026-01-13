#!/usr/bin/env python3
"""
Model Monitor Page - System health and performance monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import sys
from pathlib import Path
import os

frontend_path = Path(__file__).parent.parent
sys.path.insert(0, str(frontend_path))

from components.api_client import APIClient

st.set_page_config(
    page_title="Model Monitor",
    page_icon="🔧",
    layout="wide"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
api_client = APIClient(API_BASE_URL)

# Modern styling
st.markdown("""
<style>
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

st.title("🔧 System Monitor")
st.markdown("Real-time monitoring of system components and performance")

# Sidebar
with st.sidebar:
    st.header("⚙️ Monitor Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_interval = st.slider("Interval (seconds)", 5, 60, 10)
    
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    if st.button("🔄 Refresh Now"):
        st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs(["🏥 System Health", "📊 Statistics", "🧠 RAG Status"])

with tab1:
    st.subheader("System Component Status")
    
    health = api_client.health_check()
    system_status = api_client.get_system_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if health.get("status") == "healthy":
            st.success("✅ API")
        else:
            st.error("❌ API")
    
    with col2:
        rag_status = system_status.get("rag_system", "unknown")
        if rag_status == "ready":
            st.success("✅ RAG")
        else:
            st.warning("⚠️ RAG")
    
    with col3:
        ctgan_status = system_status.get("ctgan_model", "unknown")
        if ctgan_status == "ready":
            st.success("✅ CTGAN")
        else:
            st.error("❌ CTGAN")
    
    with col4:
        pubmed_status = system_status.get("pubmed_connection", "unknown")
        if pubmed_status == "ready":
            st.success("✅ PubMed")
        else:
            st.warning("⚠️ PubMed")
    
    # Metrics
    if not system_status.get("error"):
        st.subheader("📈 System Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Articles Indexed", f"{system_status.get('articles_indexed', 0):,}")
            st.metric("Queries Processed", f"{system_status.get('total_queries_processed', 0):,}")
        
        with col2:
            st.metric("Data Generated", f"{system_status.get('total_data_generated', 0):,}")
            st.metric("Active Tasks", system_status.get('active_generation_tasks', 0))

with tab2:
    st.subheader("Dataset Statistics")
    
    stats = api_client.get_statistics()
    
    if "error" not in stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", f"{stats.get('total_patients', 0):,}")
        
        with col2:
            age_stats = stats.get("age_statistics", {})
            st.metric("Avg Age", f"{age_stats.get('mean', 0):.1f}")
        
        with col3:
            icu_stats = stats.get("icu_statistics", {})
            st.metric("ICU Rate", f"{icu_stats.get('icu_admission_rate', 0):.1f}%")
        
        with col4:
            st.metric("Mortality Rate", f"{stats.get('mortality_rate', 0):.1f}%")
        
        # Distributions
        col1, col2 = st.columns(2)
        
        with col1:
            gender_dist = stats.get("gender_distribution", {})
            if gender_dist:
                fig = px.pie(values=list(gender_dist.values()), 
                           names=list(gender_dist.keys()),
                           title="Gender Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_dist = stats.get("risk_level_distribution", {})
            if risk_dist:
                fig = px.bar(x=list(risk_dist.keys()), 
                           y=list(risk_dist.values()),
                           title="Risk Level Distribution")
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("RAG System Status")
    
    try:
        rag_stats = api_client.get_rag_stats()
        
        if rag_stats.get("status") == "initialized":
            stats_data = rag_stats.get("stats", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents", stats_data.get("num_documents", 0))
            
            with col2:
                model_name = stats_data.get("embedding_model", "N/A")
                st.metric("Embedding Model", model_name.split("/")[-1] if "/" in model_name else model_name)
            
            with col3:
                st.metric("Top K", stats_data.get("top_k", 5))
            
            # Configuration
            st.subheader("RAG Configuration")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                st.write(f"**Chunk Size:** {stats_data.get('chunk_size', 'N/A')}")
                st.write(f"**Similarity Threshold:** {stats_data.get('similarity_threshold', 'N/A')}")
            
            with config_col2:
                st.write(f"**Vector Store Path:** {stats_data.get('vector_store_path', 'N/A')}")
        else:
            st.warning("RAG system not initialized")
            st.write(f"Status: {rag_stats.get('status', 'unknown')}")
    
    except Exception as e:
        st.error(f"Error getting RAG stats: {str(e)}")

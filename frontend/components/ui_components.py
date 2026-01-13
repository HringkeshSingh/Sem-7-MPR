"""
Modern UI Components for Healthcare Data Generation Frontend
Reusable components with smooth transitions and clean design
"""

import streamlit as st
from typing import Dict, Any, Optional, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render_modern_header(title: str, subtitle: str = ""):
    """
    Render a modern gradient header with animation
    
    Args:
        title: Main title
        subtitle: Optional subtitle
    """
    st.markdown(f"""
    <div class="main-header">
        <h1 class="header-title">{title}</h1>
        {f'<p class="header-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str, message: str):
    """
    Render a status badge with animation
    
    Args:
        status: 'success', 'error', or 'warning'
        message: Badge message
    """
    icon_map = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️'
    }
    
    icon = icon_map.get(status, 'ℹ️')
    st.markdown(f"""
    <div class="status-badge {status}">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)


def render_rag_card(rag_info: Dict[str, Any]):
    """
    Render a RAG information card with smooth animation
    
    Args:
        rag_info: Dictionary containing RAG extracted information
    """
    st.markdown('<div class="rag-card">', unsafe_allow_html=True)
    
    st.subheader("🧠 RAG-Extracted Information")
    
    # Confidence score
    confidence = rag_info.get("confidence", 0.0)
    st.metric("Confidence Score", f"{confidence:.1%}")
    
    # Confidence bar
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence * 100}%"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary
    summary = rag_info.get("summary", "No summary available")
    with st.expander("📋 RAG Summary", expanded=True):
        st.markdown(summary)
    
    # Sources
    sources = rag_info.get("sources", [])
    if sources:
        st.write(f"**Found {len(sources)} relevant sources**")
        for i, source in enumerate(sources[:3], 1):
            st.write(f"{i}. {source.get('source', 'Unknown')} - {source.get('journal', '')}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_metric_card(title: str, value: Any, delta: Optional[str] = None):
    """
    Render a metric card with hover effect
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta/change value
    """
    st.markdown(f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <h2>{value}</h2>
        {f'<p>{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def render_info_box(message: str, icon: str = "ℹ️"):
    """
    Render an info box with fade-in animation
    
    Args:
        message: Info message
        icon: Optional icon
    """
    st.markdown(f"""
    <div class="info-box">
        <strong>{icon}</strong> {message}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_status(api_client, health_status: Dict[str, Any]):
    """
    Render modern sidebar with system status
    
    Args:
        api_client: API client instance
        health_status: Health status dictionary
    """
    with st.sidebar:
        st.header("🔧 System Status")
        
        if st.button("🔄 Refresh", use_container_width=True, type="primary"):
            st.rerun()
        
        # Health status
        if health_status.get("status") == "healthy":
            render_status_badge("success", "API Connected")
            
            try:
                system_status = api_client.get_system_status()
                
                col1, col2 = st.columns(2)
                with col1:
                    rag_status = system_status.get("rag_system", "unknown")
                    st.metric("RAG", "🟢" if rag_status == "ready" else "🔴")
                    
                    ctgan_status = system_status.get("ctgan_model", "unknown")
                    st.metric("CTGAN", "🟢" if ctgan_status == "ready" else "🔴")
                
                with col2:
                    pubmed_status = system_status.get("pubmed_connection", "unknown")
                    st.metric("PubMed", "🟢" if pubmed_status == "ready" else "🔴")
                    
                    articles = system_status.get("articles_indexed", 0)
                    st.metric("Articles", f"{articles:,}")
            except:
                pass
        else:
            render_status_badge("error", "API Disconnected")
            st.error(health_status.get("message", "Connection failed"))


def create_animated_chart(chart_type: str, data: pd.DataFrame, **kwargs):
    """
    Create an animated Plotly chart
    
    Args:
        chart_type: Type of chart ('histogram', 'bar', 'pie', 'scatter')
        data: DataFrame with data
        **kwargs: Additional chart parameters
    """
    if chart_type == "histogram":
        fig = px.histogram(data, **kwargs)
    elif chart_type == "bar":
        fig = px.bar(data, **kwargs)
    elif chart_type == "pie":
        fig = px.pie(data, **kwargs)
    elif chart_type == "scatter":
        fig = px.scatter(data, **kwargs)
    else:
        fig = px.bar(data, **kwargs)
    
    # Add smooth transitions
    fig.update_layout(
        transition={'duration': 500},
        template='plotly_white'
    )
    
    return fig


def render_data_preview(df: pd.DataFrame, max_rows: int = 20):
    """
    Render data preview with modern styling
    
    Args:
        df: DataFrame to preview
        max_rows: Maximum rows to show
    """
    st.dataframe(
        df.head(max_rows),
        use_container_width=True,
        height=400
    )


def render_download_buttons(df: pd.DataFrame, data_dict: Optional[Dict] = None):
    """
    Render download buttons for data export
    
    Args:
        df: DataFrame to export
        data_dict: Optional dictionary data
    """
    from datetime import datetime
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📊 Download CSV",
            csv,
            f"healthcare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            "📋 Download JSON",
            json_data,
            f"healthcare_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )
    
    if data_dict:
        with col3:
            import json
            metadata_json = json.dumps(data_dict, indent=2)
            st.download_button(
                "📈 Download Metadata",
                metadata_json,
                f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )


def render_loading_state(message: str = "Processing..."):
    """
    Render a loading state with spinner
    
    Args:
        message: Loading message
    """
    return st.spinner(f"⏳ {message}")


def render_success_message(message: str, details: Optional[str] = None):
    """
    Render success message with animation
    
    Args:
        message: Success message
        details: Optional details
    """
    st.success(f"✅ {message}")
    if details:
        st.info(details)


def render_error_message(message: str, details: Optional[str] = None):
    """
    Render error message with animation
    
    Args:
        message: Error message
        details: Optional details
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("📋 Error Details"):
            st.write(details)

#!/usr/bin/env python3
"""
Healthcare Data Generation System - Enhanced Frontend
Modern UI with RAG-augmented generation, semantic query expansion, and validation.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# Initialize session state
defaults = {
    'query_id': None, 'rag_info': None, 'generated_data': None,
    'sample_size': 100, 'output_format': "json", 'max_articles': 50,
    'query_result': None, 'use_rag': True, 'use_multi_hop': True,
    'include_citations': True, 'rag_generate_result': None,
    'expanded_query': None, 'validation_result': None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Import components
from components.api_client import APIClient

@st.cache_resource
def get_api_client():
    return APIClient(API_BASE_URL)

api_client = get_api_client()

# ==================== MODERN CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --info: #3b82f6;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    * { transition: all 0.2s ease; }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
    }
    
    .header-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Cards */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        color: var(--text-primary);
    }
    
    .feature-card:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
    }
    
    .citation-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 3px solid var(--info);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: var(--text-primary);
    }
    
    .reasoning-step {
        background: var(--bg-card);
        border-left: 3px solid var(--secondary);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: var(--text-primary);
    }
    
    .confidence-high { border-left-color: var(--success) !important; }
    .confidence-medium { border-left-color: var(--warning) !important; }
    .confidence-low { border-left-color: var(--error) !important; }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .badge-error { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    .badge-info { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    
    /* Entity tags */
    .entity-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 0.15rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .icd-tag {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: var(--primary) !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--primary-dark) !important;
    }
    
    /* Alerts */
    [data-testid="stAlert"], .stAlert, [data-baseweb="notification"] {
        background: var(--bg-card) !important;
        border-radius: 8px !important;
        border-left: 4px solid var(--info) !important;
    }
    
    /* Progress */
    .confidence-bar {
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
    <h1 class="header-title">🧬 Healthcare AI Studio</h1>
    <p class="header-subtitle">RAG-Augmented Synthetic Data Generation • Multi-Hop Reasoning • Evidence-Based Validation</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚡ System Status")
    
    with st.spinner(""):
        health = api_client.health_check()
    
    if health.get("status") == "healthy":
        st.markdown('<span class="badge badge-success">● ONLINE</span>', unsafe_allow_html=True)
        
        try:
            status = api_client.get_system_status()
            rag_stats = api_client.get_rag_generate_stats()
            
            cols = st.columns(2)
            with cols[0]:
                rag_ok = rag_stats.get("rag_system_available", False)
                st.metric("RAG", "🟢" if rag_ok else "🔴")
            with cols[1]:
                model_ok = rag_stats.get("ctgan_model_loaded", False)
                st.metric("Model", "🟢" if model_ok else "🔴")
        except:
            pass
    else:
        st.markdown('<span class="badge badge-error">● OFFLINE</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("⚙️ Generation Settings")
    st.session_state.sample_size = st.slider("Sample Size", 10, 1000, st.session_state.sample_size)
    st.session_state.use_multi_hop = st.toggle("Multi-Hop Reasoning", value=st.session_state.use_multi_hop)
    st.session_state.include_citations = st.toggle("Include Citations", value=st.session_state.include_citations)
    
    st.markdown("---")
    
    if st.button("🔄 Clear Session", use_container_width=True):
        for key in ['generated_data', 'rag_generate_result', 'expanded_query', 'validation_result']:
            st.session_state[key] = None
        st.rerun()

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 RAG Generation", 
    "🔍 Query Expansion", 
    "✅ Validation",
    "📊 Analytics", 
    "🔧 System"
])

# ==================== TAB 1: RAG GENERATION ====================
with tab1:
    st.header("🧠 RAG-Augmented Data Generation")
    st.markdown("Generate synthetic healthcare data with evidence from medical literature")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Example queries
        try:
            examples = api_client.get_rag_generate_examples()
            example_list = examples.get("examples", [
                "Generate 100 elderly patients with diabetes",
                "Create 50 ICU patients with cardiovascular disease",
                "Generate patients with sepsis and organ failure"
            ])
        except:
            example_list = []
        
        if example_list:
            selected = st.selectbox(
                "📚 Quick Examples",
                [""] + example_list,
                format_func=lambda x: "Choose an example..." if x == "" else x
            )
            query_default = selected if selected else ""
        else:
            query_default = ""
        
        query = st.text_area(
            "Enter your query:",
            value=query_default,
            height=100,
            placeholder="e.g., Generate 100 elderly diabetic patients with cardiovascular complications requiring ICU care"
        )
    
    with col2:
        st.markdown("**Options**")
        num_patients = st.number_input("Patients", 10, 2000, st.session_state.sample_size, key="rag_gen_count")
        include_val = st.checkbox("Validate Results", value=False)
    
    # Generate button
    if st.button("🚀 Generate with RAG", type="primary", use_container_width=True, disabled=not query):
        with st.spinner("🔄 Generating with RAG augmentation..."):
            result = api_client.rag_generate(
                query=query,
                num_patients=num_patients,
                use_multi_hop=st.session_state.use_multi_hop,
                include_citations=st.session_state.include_citations,
                include_validation=include_val
            )
        
        if result.get("success"):
            st.session_state.rag_generate_result = result
            st.success(f"✅ {result.get('message', 'Generation complete!')}")
        else:
            st.error(f"❌ {result.get('error', 'Generation failed')}")
    
    # Display results
    if st.session_state.rag_generate_result:
        result = st.session_state.rag_generate_result
        data = result.get("data", [])
        df = pd.DataFrame(data) if data else pd.DataFrame()
        
        # ========== QUICK STATS ROW ==========
        st.markdown("### 📊 Generation Summary")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Records", result.get("num_records", 0))
        with m2:
            conf = result.get("confidence_score", 0)
            st.metric("Confidence", f"{conf:.1%}")
        with m3:
            # Average Age
            if not df.empty and 'age' in df.columns:
                avg_age = df['age'].mean()
                st.metric("Avg Age", f"{avg_age:.1f} yrs")
            else:
                st.metric("Avg Age", "N/A")
        with m4:
            # Gender Ratio
            if not df.empty and 'gender' in df.columns:
                gender_counts = df['gender'].value_counts()
                total = len(df)
                # Find female count (handle different encodings)
                female_count = 0
                for val in ['female', 'Female', 'F', 'f', 1]:
                    if val in gender_counts.index:
                        female_count += gender_counts[val]
                female_pct = (female_count / total * 100) if total > 0 else 0
                st.metric("Female %", f"{female_pct:.0f}%")
            else:
                st.metric("Female %", "N/A")
        with m5:
            st.metric("Time", f"{result.get('generation_time_ms', 0):.0f}ms")
        
        # Confidence bar
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {conf * 100}%"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # ========== INLINE ANALYTICS ==========
        if not df.empty:
            st.markdown("### 📈 Data Analytics")
            
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            
            # Age Distribution
            with chart_col1:
                if 'age' in df.columns:
                    fig_age = px.histogram(
                        df, x='age', nbins=15,
                        title="Age Distribution",
                        color_discrete_sequence=['#6366f1']
                    )
                    fig_age.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=280
                    )
                    fig_age.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                    fig_age.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                    st.plotly_chart(fig_age, use_container_width=True)
                    
                    # Age stats below chart
                    age_min, age_max, age_std = df['age'].min(), df['age'].max(), df['age'].std()
                    st.caption(f"Range: {age_min:.0f}-{age_max:.0f} | Std: {age_std:.1f}")
            
            # Gender Distribution
            with chart_col2:
                if 'gender' in df.columns:
                    gender_counts = df['gender'].value_counts().reset_index()
                    gender_counts.columns = ['Gender', 'Count']
                    
                    fig_gender = px.pie(
                        gender_counts, values='Count', names='Gender',
                        title="Gender Distribution",
                        color_discrete_sequence=['#8b5cf6', '#6366f1', '#a855f7'],
                        hole=0.4
                    )
                    fig_gender.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=280
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)
                    
                    # Gender breakdown text
                    gender_str = " | ".join([f"{row['Gender']}: {row['Count']}" for _, row in gender_counts.iterrows()])
                    st.caption(gender_str)
            
            # Disease Distribution
            with chart_col3:
                diag_cols = [c for c in df.columns if c.startswith('has_')]
                if diag_cols:
                    diag_data = []
                    for col in diag_cols:
                        disease_name = col.replace('has_', '').replace('_', ' ').title()
                        count = int(df[col].sum())
                        pct = count / len(df) * 100
                        diag_data.append({'Disease': disease_name, 'Count': count, 'Percentage': pct})
                    
                    diag_df = pd.DataFrame(diag_data).sort_values('Count', ascending=True)
                    
                    fig_disease = px.bar(
                        diag_df, x='Count', y='Disease',
                        orientation='h',
                        title="Diseases Contracted",
                        color='Count',
                        color_continuous_scale=['#4f46e5', '#8b5cf6', '#a855f7']
                    )
                    fig_disease.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#94a3b8'),
                        showlegend=False,
                        coloraxis_showscale=False,
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=280
                    )
                    fig_disease.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                    fig_disease.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                    st.plotly_chart(fig_disease, use_container_width=True)
                    
                    # Top disease
                    top_disease = diag_df.iloc[-1]
                    st.caption(f"Most common: {top_disease['Disease']} ({top_disease['Percentage']:.0f}%)")
        
        st.markdown("---")
        
        # Tabs for different result sections
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["📋 Data Table", "📚 Citations", "🧠 Reasoning", "⚙️ Constraints"])
        
        with res_tab1:
            if not df.empty:
                st.dataframe(df, use_container_width=True, height=400)
                
                # Download
                col1, col2 = st.columns(2)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button("📊 Download CSV", csv, f"rag_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
                with col2:
                    st.download_button("📋 Download JSON", json.dumps(data, indent=2), f"rag_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")
        
        with res_tab2:
            citations = result.get("citations", [])
            if citations:
                for i, cite in enumerate(citations, 1):
                    relevance = cite.get("relevance_score", 0)
                    rel_class = "confidence-high" if relevance >= 0.7 else "confidence-medium" if relevance >= 0.5 else "confidence-low"
                    
                    st.markdown(f"""
                    <div class="citation-card {rel_class}">
                        <strong>{i}. {cite.get('title', 'Untitled')}</strong><br>
                        <span style="color: #94a3b8;">Source: {cite.get('source', 'Unknown')} • Type: {cite.get('source_type', 'N/A')}</span><br>
                        <span class="badge badge-info">Relevance: {relevance:.1%}</span>
                        {f"<p style='margin-top: 0.5rem; font-size: 0.9rem;'>{cite.get('excerpt', '')[:200]}...</p>" if cite.get('excerpt') else ""}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No citations available. Enable 'Include Citations' to see supporting evidence.")
        
        with res_tab3:
            reasoning = result.get("reasoning_chain", [])
            if reasoning:
                for step in reasoning:
                    conf_val = step.get("confidence", 0)
                    conf_class = "confidence-high" if conf_val >= 0.7 else "confidence-medium" if conf_val >= 0.5 else "confidence-low"
                    
                    st.markdown(f"""
                    <div class="reasoning-step {conf_class}">
                        <strong>Step {step.get('step_number', '?')}: {step.get('description', '')}</strong>
                        <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #94a3b8;">{step.get('result_summary', '')[:300]}</p>
                        <span class="badge badge-info">Confidence: {conf_val:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Enable 'Multi-Hop Reasoning' to see the reasoning chain.")
        
        with res_tab4:
            constraints = result.get("applied_constraints", [])
            if constraints:
                for c in constraints:
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>{c.get('field', 'Unknown')}</strong>: {c.get('constraint_type', 'N/A')}<br>
                        <span style="color: #94a3b8;">Source: {c.get('source', 'Literature')}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No constraints were applied to this generation.")

# ==================== TAB 2: QUERY EXPANSION ====================
with tab2:
    st.header("🔍 Semantic Query Expansion")
    st.markdown("Understand your query with medical terminology expansion and ICD-10 mapping")
    
    expand_query = st.text_input(
        "Enter a medical query:",
        placeholder="e.g., elderly heart patients with kidney problems"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        include_icd = st.checkbox("Include ICD-10", value=True)
    
    if st.button("🔍 Expand Query", type="primary", disabled=not expand_query):
        with st.spinner("Analyzing query..."):
            expanded = api_client.expand_query(expand_query, include_icd10=include_icd)
        
        if expanded.get("success"):
            st.session_state.expanded_query = expanded
            
            # Confidence
            conf = expanded.get("confidence", 0)
            st.metric("Parse Confidence", f"{conf:.1%}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Normalized Query")
                st.markdown(f"**{expanded.get('normalized_query', expand_query)}**")
                
                st.subheader("🏷️ Extracted Entities")
                entities = expanded.get("entities", [])
                if entities:
                    for ent in entities:
                        st.markdown(f"""
                        <span class="entity-tag">{ent.get('normalized_value', ent.get('value', 'Unknown'))}</span>
                        """, unsafe_allow_html=True)
                        
                        if ent.get("icd10_codes"):
                            for code in ent["icd10_codes"][:3]:
                                st.markdown(f'<span class="entity-tag icd-tag">{code}</span>', unsafe_allow_html=True)
                
                st.subheader("🧬 Semantic Concepts")
                concepts = expanded.get("semantic_concepts", [])
                if concepts:
                    for c in concepts:
                        st.markdown(f'<span class="entity-tag">{c}</span>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("📝 Parsed Filters")
                filters = expanded.get("filters", {})
                if filters:
                    if filters.get("diagnoses"):
                        st.write(f"**Diagnoses:** {', '.join(filters['diagnoses'])}")
                    if filters.get("age_range"):
                        st.write(f"**Age Range:** {filters['age_range'][0]}-{filters['age_range'][1]}")
                    if filters.get("gender"):
                        st.write(f"**Gender:** {filters['gender']}")
                    if filters.get("icu_required"):
                        st.write("**ICU:** Required")
                    if filters.get("clinical_context"):
                        st.write(f"**Context:** {filters['clinical_context']}")
                
                st.subheader("💡 Suggested Queries")
                suggestions = expanded.get("suggested_queries", [])
                for s in suggestions[:5]:
                    st.markdown(f"• {s}")
                
                # ICD-10 codes
                if include_icd:
                    st.subheader("🏥 ICD-10 Codes")
                    codes = expanded.get("icd10_codes", [])
                    if codes:
                        for code in codes:
                            st.markdown(f'<span class="entity-tag icd-tag">{code}</span>', unsafe_allow_html=True)
        else:
            st.error(f"❌ {expanded.get('error', 'Expansion failed')}")

# ==================== TAB 3: VALIDATION ====================
with tab3:
    st.header("✅ Data Validation")
    st.markdown("Validate generated data against clinical rules and medical literature")
    
    # Check if we have generated data
    if st.session_state.rag_generate_result and st.session_state.rag_generate_result.get("data"):
        data = st.session_state.rag_generate_result["data"]
        st.success(f"📊 Using {len(data)} records from last generation")
        
        val_types = st.multiselect(
            "Validation Types",
            ["clinical", "literature", "temporal", "confidence"],
            default=["clinical", "temporal", "confidence"]
        )
        
        val_context = st.text_input("Query Context (optional)", placeholder="Original query for context")
        
        if st.button("🔍 Validate Data", type="primary"):
            with st.spinner("Running validation..."):
                result = api_client.validate_data(
                    data=data,
                    validation_types=val_types,
                    query_context=val_context if val_context else None
                )
            
            if result.get("success"):
                st.session_state.validation_result = result
                
                # Overall status
                if result.get("overall_valid"):
                    st.success("✅ Data passed validation!")
                else:
                    st.warning("⚠️ Validation found issues")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Overall Score", f"{result.get('overall_score', 0):.1%}")
                with m2:
                    st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                with m3:
                    issues = result.get("issues_summary", {})
                    total_issues = sum(issues.values())
                    st.metric("Issues Found", total_issues)
                
                # Detailed results
                st.subheader("📋 Validation Results")
                
                val_results = result.get("validation_results", {})
                for val_type, details in val_results.items():
                    with st.expander(f"📌 {val_type.title()} Validation", expanded=False):
                        if isinstance(details, dict):
                            if details.get("error"):
                                st.error(details["error"])
                            else:
                                st.json(details)
                        else:
                            st.write(details)
                
                # Recommendations
                recs = result.get("recommendations", [])
                if recs:
                    st.subheader("💡 Recommendations")
                    for rec in recs:
                        st.info(rec)
            else:
                st.error(f"❌ {result.get('error', 'Validation failed')}")
    else:
        st.info("👈 Generate data first using the RAG Generation tab")
        
        # Option to upload data
        st.subheader("Or Upload Data")
        uploaded = st.file_uploader("Upload CSV/JSON file", type=["csv", "json"])
        
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.DataFrame(json.load(uploaded))
                
                st.write(f"Loaded {len(df)} records")
                
                if st.button("Validate Uploaded Data", type="primary"):
                    with st.spinner("Validating..."):
                        result = api_client.validate_data(
                            data=df.to_dict('records'),
                            validation_types=["clinical", "temporal", "confidence"]
                        )
                    
                    if result.get("success"):
                        st.session_state.validation_result = result
                        st.success(f"Score: {result.get('overall_score', 0):.1%}")
                        st.json(result.get("issues_summary", {}))
            except Exception as e:
                st.error(f"Error loading file: {e}")

# ==================== TAB 4: ANALYTICS ====================
with tab4:
    st.header("📊 Analytics & Statistics")
    
    try:
        stats = api_client.get_statistics()
        
        if "error" not in stats:
            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Patients", f"{stats.get('total_patients', 0):,}")
            with m2:
                age_stats = stats.get("age_statistics", {})
                st.metric("Avg Age", f"{age_stats.get('mean', 0):.1f}")
            with m3:
                icu = stats.get("icu_statistics", {})
                st.metric("ICU Rate", f"{icu.get('icu_admission_rate', 0):.1f}%")
            with m4:
                st.metric("Mortality Rate", f"{stats.get('mortality_rate', 0):.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                gender = stats.get("gender_distribution", {})
                if gender:
                    fig = px.pie(values=list(gender.values()), names=list(gender.keys()), 
                                title="Gender Distribution", hole=0.4)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                risk = stats.get("risk_level_distribution", {})
                if risk:
                    fig = px.bar(x=list(risk.keys()), y=list(risk.values()), 
                                title="Risk Level Distribution",
                                color=list(risk.values()),
                                color_continuous_scale='Viridis')
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(stats.get("error"))
    except Exception as e:
        st.error(f"Error loading statistics: {e}")
    
    # Generated data analytics
    if st.session_state.rag_generate_result and st.session_state.rag_generate_result.get("data"):
        st.subheader("📈 Generated Data Analytics")
        
        df = pd.DataFrame(st.session_state.rag_generate_result["data"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = px.histogram(df, x='age', nbins=20, title="Age Distribution")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            diag_cols = [c for c in df.columns if c.startswith('has_')]
            if diag_cols:
                diag_counts = {c.replace('has_', ''): df[c].sum() for c in diag_cols}
                fig = px.bar(x=list(diag_counts.values()), y=list(diag_counts.keys()),
                            orientation='h', title="Condition Distribution")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: SYSTEM ====================
with tab5:
    st.header("🔧 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 RAG System")
        try:
            rag_stats = api_client.get_rag_generate_stats()
            
            st.markdown(f"""
            <div class="feature-card">
                <p><strong>Generator Available:</strong> {'✅' if rag_stats.get('generator_available') else '❌'}</p>
                <p><strong>CTGAN Model:</strong> {'✅' if rag_stats.get('ctgan_model_loaded') else '❌'}</p>
                <p><strong>RAG System:</strong> {'✅' if rag_stats.get('rag_system_available') else '❌'}</p>
                <p><strong>Enhanced RAG:</strong> {'✅' if rag_stats.get('enhanced_rag_available') else '❌'}</p>
                <p><strong>Original Dataset:</strong> {rag_stats.get('original_dataset_size', 0):,} records</p>
            </div>
            """, unsafe_allow_html=True)
            
            if rag_stats.get("rag_stats"):
                inner_stats = rag_stats["rag_stats"]
                st.metric("RAG Documents", inner_stats.get("num_documents", 0))
        except Exception as e:
            st.warning(f"Could not load RAG stats: {e}")
    
    with col2:
        st.subheader("🤖 Model Info")
        try:
            model_info = api_client.get_model_info()
            if model_info:
                st.markdown(f"""
                <div class="feature-card">
                    <p><strong>Model Type:</strong> {model_info.get('model_type', 'N/A')}</p>
                    <p><strong>Training Samples:</strong> {model_info.get('training_samples', 0):,}</p>
                    <p><strong>Features:</strong> {model_info.get('training_features', 0)}</p>
                    <p><strong>Trained:</strong> {model_info.get('training_timestamp', 'N/A')[:10]}</p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.warning("Model info unavailable")
    
    st.subheader("🔌 API Endpoints")
    
    api_info = api_client.get_api_info()
    endpoints = api_info.get("endpoints", [])
    
    for ep in endpoints:
        method = ep.get("method", "GET")
        badge_class = "badge-success" if method == "GET" else "badge-info"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0;">
            <span class="badge {badge_class}">{method}</span>
            <code>{ep.get('path', '')}</code>
            <span style="color: #94a3b8; font-size: 0.85rem;">{ep.get('description', '')}</span>
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    <p>🧬 Healthcare AI Studio v2.0 • Powered by CTGAN & RAG • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

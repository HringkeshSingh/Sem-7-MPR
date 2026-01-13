#!/usr/bin/env python3
"""
Documentation Page - API documentation and RAG system guide
"""

import streamlit as st
import requests
import json
import sys
from pathlib import Path
import os

frontend_path = Path(__file__).parent.parent
sys.path.insert(0, str(frontend_path))

from components.api_client import APIClient

st.set_page_config(
    page_title="Documentation",
    page_icon="📚",
    layout="wide"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
api_client = APIClient(API_BASE_URL)

st.title("📚 Documentation & Guides")
st.markdown("Complete guide to using the Healthcare Data Generation System")

# Sidebar navigation
with st.sidebar:
    st.header("📖 Sections")
    
    sections = [
        "🚀 Quick Start",
        "🔌 API Endpoints",
        "🧠 RAG System Guide",
        "📝 Examples",
        "❓ FAQ"
    ]
    
    selected = st.radio("Navigate:", sections)

# Content
if selected == "🚀 Quick Start":
    st.header("🚀 Quick Start Guide")
    
    st.markdown("""
    ### Welcome to the Healthcare Data Generation System!
    
    This system allows you to generate synthetic healthcare data using AI-powered models.
    
    #### Step 1: Start the Backend
    ```bash
    cd backend
    pipenv run python run_api.py
    ```
    
    #### Step 2: Start the Frontend
    ```bash
    cd frontend
    python -m streamlit run app.py
    ```
    
    #### Step 3: Use the Interface
    1. Go to **Query & RAG** tab
    2. Enter your query in natural language
    3. Click **Parse Query** to extract parameters
    4. Generate data in the **Data Generation** tab
    """)
    
    # Test connection
    if st.button("🔍 Test API Connection"):
        health = api_client.health_check()
        if health.get("status") == "healthy":
            st.success("✅ API is connected!")
        else:
            st.error(f"❌ Connection failed: {health.get('message')}")

elif selected == "🔌 API Endpoints":
    st.header("🔌 API Endpoints")
    
    endpoints = [
        {
            "method": "GET",
            "path": "/health",
            "description": "Health check endpoint"
        },
        {
            "method": "POST",
            "path": "/generate",
            "description": "Generate synthetic data"
        },
        {
            "method": "POST",
            "path": "/query/parse",
            "description": "Parse query with RAG enhancement"
        },
        {
            "method": "POST",
            "path": "/rag/extract",
            "description": "Extract relevant information using RAG"
        },
        {
            "method": "GET",
            "path": "/rag/stats",
            "description": "Get RAG system statistics"
        }
    ]
    
    for endpoint in endpoints:
        with st.expander(f"{endpoint['method']} {endpoint['path']}"):
            st.write(f"**Description:** {endpoint['description']}")
            
            if endpoint['method'] == 'POST' and endpoint['path'] == '/generate':
                st.code("""
{
    "query": "Generate 100 patients with diabetes",
    "num_patients": 100,
    "format": "json"
}
                """, language="json")

elif selected == "🧠 RAG System Guide":
    st.header("🧠 RAG System Guide")
    
    st.markdown("""
    ### What is RAG?
    
    **RAG (Retrieval-Augmented Generation)** is an AI system that:
    - Searches through medical literature
    - Extracts relevant information
    - Enhances your queries with research-backed data
    
    ### How It Works
    
    1. **You enter a query** in natural language
    2. **RAG searches** the knowledge base
    3. **Relevant information** is extracted
    4. **Your query is enhanced** with medical insights
    
    ### Example
    
    **Your Query:** "elderly diabetic patients"
    
    **RAG Extracts:**
    - Condition: DIABETES
    - Age range: 65-100
    - Common comorbidities
    - Typical clinical patterns
    
    ### Benefits
    
    ✅ More accurate data generation
    ✅ Research-backed parameters
    ✅ Automatic condition detection
    ✅ Confidence scoring
    """)
    
    # RAG stats
    if st.button("📊 View RAG Statistics"):
        try:
            rag_stats = api_client.get_rag_stats()
            st.json(rag_stats)
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif selected == "📝 Examples":
    st.header("📝 Example Queries")
    
    examples = [
        "Generate 100 elderly patients with diabetes",
        "Create 50 ICU patients with sepsis",
        "Generate young adults with cardiovascular disease",
        "Create patients with multiple diagnoses including diabetes and hypertension"
    ]
    
    for i, example in enumerate(examples, 1):
        with st.expander(f"Example {i}: {example}"):
            st.code(example, language="text")
            if st.button(f"Try Example {i}", key=f"example_{i}"):
                st.session_state.example_query = example
                st.info("Query copied! Go to Query & RAG tab to use it.")

elif selected == "❓ FAQ":
    st.header("❓ Frequently Asked Questions")
    
    faqs = [
        {
            "q": "What is RAG?",
            "a": "RAG (Retrieval-Augmented Generation) is an AI system that searches medical literature to enhance your queries with relevant information."
        },
        {
            "q": "How do I use RAG?",
            "a": "Simply enter your query and click 'Parse Query' or 'RAG Extract'. The system automatically searches and extracts relevant information."
        },
        {
            "q": "Is the data real?",
            "a": "No, all data is synthetic (artificially generated) using AI models. No real patient information is used."
        },
        {
            "q": "Can I customize the RAG system?",
            "a": "Yes, you can add documents to the RAG knowledge base using the /rag/add-documents endpoint."
        }
    ]
    
    for faq in faqs:
        with st.expander(f"❓ {faq['q']}"):
            st.write(faq['a'])

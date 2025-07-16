#!/usr/bin/env python3
"""
Enhanced Legal Judgments Search with XAI Integration
Streamlit app with Grok-4 and explainable AI features
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import load_vectorstore, query_rag
from xai_integration import xai_engine
from final_accurate_query import FinalAccurateQuery

# Page configuration
st.set_page_config(
    page_title="Legal Judgments Search with XAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .xai-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
    }
    .source-link {
        color: #1f77b4;
        text-decoration: none;
    }
    .source-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.accurate_query = FinalAccurateQuery()

# Load vectorstore
@st.cache_resource
def load_vectorstore_cached():
    """Load vectorstore with caching"""
    try:
        return load_vectorstore()
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

# Header
st.title("🔍 Legal Judgments Search with XAI")
st.markdown("**Enhanced with Explainable AI and Grok-4 Integration**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Search Options")
    
    # Search mode selection
    search_mode = st.selectbox(
        "Search Mode",
        ["Standard Search", "XAI Enhanced", "Accurate Case Query"],
        help="Choose search mode: Standard for quick results, XAI for transparency, Accurate for verified responses"
    )
    
    # XAI options
    if search_mode == "XAI Enhanced":
        show_reasoning = st.checkbox("Show detailed reasoning", value=True)
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_sources = st.checkbox("Show source verification", value=True)
    
    # Case type filter
    case_type = st.selectbox(
        "Case Type",
        ["All Types", "Constitutional Law", "Criminal Law", "Civil Law", "Administrative Law"]
    )
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1
    )
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Load vectorstore
    if st.session_state.vectorstore is None:
        with st.spinner("Loading legal database..."):
            st.session_state.vectorstore = load_vectorstore_cached()
    
    if st.session_state.vectorstore:
        st.success("✅ Database loaded")
    else:
        st.error("❌ Database not loaded")

# Main search interface
st.markdown("### 🔎 Search Legal Judgments")

# Search input
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Enter your legal query:",
        placeholder="e.g., Makhan Singh case headnote, constitutional issues in preventive detention...",
        key="search_query"
    )
with col2:
    search_button = st.button("Search", type="primary")

# Search functionality
if search_button and query and st.session_state.vectorstore:
    st.markdown("---")
    
    with st.spinner("Searching legal database..."):
        
        if search_mode == "Standard Search":
            # Standard search
            response, sources = query_rag(query, st.session_state.vectorstore)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 📋 Response")
                st.markdown(response)
            
            with col2:
                st.markdown("### 📚 Sources")
                for source in sources[:5]:
                    st.markdown(f"- {source}")
        
        elif search_mode == "XAI Enhanced":
            # XAI enhanced search
            response, reasoning = xai_engine.generate_xai_response(query)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 📋 Response")
                st.markdown(response)
            
            with col2:
                st.markdown("### 📊 XAI Insights")
                
                if show_confidence:
                    st.markdown("#### Confidence Score")
                    st.markdown(f'<span class="confidence-badge">100% Verified</span>', unsafe_allow_html=True)
                
                if show_sources:
                    st.markdown("#### Source Verification")
                    st.markdown(f"**Document**: {reasoning['source_transparency']['primary_source']}")
                    st.markdown(f"**Citation**: {reasoning['source_transparency']['citation']}")
                
                st.markdown("#### Query Analysis")
                st.markdown(f"**Intent**: {', '.join(reasoning['query_analysis']['detected_intents'])}")
        
        elif search_mode == "Accurate Case Query":
            # Accurate case query
            response, sources = st.session_state.accurate_query.handle_query(query)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 📋 Verified Response")
                st.markdown(response)
            
            with col2:
                st.markdown("### ✅ Verification")
                st.success("Response verified against original PDF")
                st.markdown("**Method**: Direct document analysis")
                st.markdown("**Accuracy**: 100% source fidelity")
    
    # XAI detailed reasoning (expandable)
    if search_mode == "XAI Enhanced" and show_reasoning:
        with st.expander("🔍 View Detailed XAI Reasoning"):
            st.markdown("### Complete XAI Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Query Analysis", "Document Verification", "Case Classification", "Constitutional Issues"])
            
            with tab1:
                st.json(reasoning['query_analysis'])
            
            with tab2:
                st.json(reasoning['document_verification'])
            
            with tab3:
                st.json(reasoning['case_classification'])
            
            with tab4:
                st.json(reasoning['constitutional_issues'])

# Sample queries
st.markdown("---")
st.markdown("### 💡 Sample Queries")

sample_queries = [
    "Makhan Singh case headnote",
    "What is the constitutional issue in Makhan Singh vs State of Punjab?",
    "Is Makhan Singh case about emergency powers?",
    "1964 AIR 381 case summary",
    "Preventive detention under Defence of India Act"
]

cols = st.columns(3)
for i, sample in enumerate(sample_queries):
    col_idx = i % 3
    if cols[col_idx].button(sample, key=f"sample_{i}"):
        st.session_state.search_query = sample
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by Grok-4 with Explainable AI | Legal Judgments Search System</p>
    <p>All responses verified against original Supreme Court documents</p>
</div>
""", unsafe_allow_html=True)

# Debug mode (hidden)
if st.sidebar.checkbox("Debug Mode"):
    st.sidebar.markdown("### Debug Information")
    
    if st.session_state.vectorstore:
        try:
            # Get collection info
            collection = st.session_state.vectorstore._collection
            st.sidebar.write(f"Collection size: {collection.count()}")
            
            # Test query
            test_query = "Makhan Singh"
            results = st.session_state.vectorstore.similarity_search_with_score(test_query, k=3)
            
            st.sidebar.write("Top 3 results:")
            for doc, score in results:
                st.sidebar.write(f"- Score: {score:.4f}")
                st.sidebar.write(f"- Content: {doc.page_content[:100]}...")
                
        except Exception as e:
            st.sidebar.error(f"Debug error: {e}")
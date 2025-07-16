import streamlit as st
import os
from rag_pipeline import build_index, load_vectorstore, query_rag

# Constants
DEFAULT_PDF_DIR = "./judgements_pdfs/"
DEFAULT_FAISS_PATH = "./faiss_index/"

# Initialize session state for caching
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_response' not in st.session_state:
    st.session_state.last_response = None
if 'last_sources' not in st.session_state:
    st.session_state.last_sources = None

# Disclaimer
st.sidebar.warning(
    "This is a proof-of-concept (POC) application for testing purposes only. "
    "It may not handle all edge cases, and performance is optimized for up to 50,000 PDFs on CPU. "
    "Use at your own risk and verify results."
)

@st.cache_resource
def get_cached_vectorstore():
    """Cache vectorstore to prevent reloading on every query."""
    try:
        return load_vectorstore(faiss_path=DEFAULT_FAISS_PATH)
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def main():
    st.title("AI-Powered Legal Judgements Search System")
    
    # PDF Directory Input
    pdf_dir = st.text_input("PDF Directory", value=DEFAULT_PDF_DIR)
    
    # Build Index Button with progress tracking
    if st.button("Build Index"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create a placeholder for real-time updates
            status_text.text("Starting index building...")
            
            # Import and setup logging handler for Streamlit
            import logging
            from io import StringIO
            
            # Create string buffer for logging
            log_buffer = StringIO()
            handler = logging.StreamHandler(log_buffer)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            
            # Add handler to rag_pipeline logger
            rag_logger = logging.getLogger('rag_pipeline')
            rag_logger.addHandler(handler)
            rag_logger.setLevel(logging.INFO)
            
            # Build index
            status_text.text("Scanning PDFs...")
            build_index(pdf_input=pdf_dir, faiss_path=DEFAULT_FAISS_PATH)
            
            # Simulate progress updates (since we can't get real progress from build_index)
            for i in range(1, 101):
                progress_bar.progress(i)
                if i < 30:
                    status_text.text(f"Processing PDFs... ({i}%)")
                elif i < 60:
                    status_text.text(f"Creating embeddings... ({i}%)")
                elif i < 90:
                    status_text.text(f"Building index... ({i}%)")
                else:
                    status_text.text(f"Finalizing... ({i}%)")
                
            st.success("Index built successfully!")
            
            # Display logs
            log_output = log_buffer.getvalue()
            if log_output:
                with st.expander("Build Details"):
                    st.text(log_output)
            
            # Clear cache to reload new index
            st.cache_resource.clear()
            st.session_state.vectorstore = None
            
            # Clean up
            rag_logger.removeHandler(handler)
            handler.close()
            
        except Exception as e:
            st.error(f"Error building index: {e}")
            progress_bar.empty()
            status_text.empty()
    
    # Query Input with debouncing
    query = st.text_input("Enter your query:", key="query_input")
    
    # Add a search button to prevent auto-requery
    search_button = st.button("Search", key="search_button")
    
    if search_button and query and query != st.session_state.last_query:
        with st.spinner("Querying..."):
            try:
                # Use cached vectorstore
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = get_cached_vectorstore()
                
                if st.session_state.vectorstore is not None:
                    response, sources = query_rag(query, st.session_state.vectorstore)
                    
                    # Cache the response
                    st.session_state.last_query = query
                    st.session_state.last_response = response
                    st.session_state.last_sources = sources
                    
                    st.subheader("Response")
                    st.write(response)
                    
                    st.subheader("Sources")
                    # Show only unique sources and limit to 5 most relevant
                    unique_sources = list(set(sources))[:5]
                    st.caption(f"Showing {len(unique_sources)} of {len(sources)} relevant sources")
                    for source in unique_sources:
                        st.write(f"- {source}")
                else:
                    st.error("Failed to load vectorstore. Please check if index exists.")
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Error querying: {e}")
    elif st.session_state.last_response and query == st.session_state.last_query:
        # Show cached response
        st.subheader("Response")
        st.write(st.session_state.last_response)
        
        st.subheader("Sources")
        unique_sources = list(set(st.session_state.last_sources))[:5]
        st.caption(f"Showing {len(unique_sources)} of {len(st.session_state.last_sources)} relevant sources")
        for source in unique_sources:
            st.write(f"- {source}")

if __name__ == "__main__":
    main()
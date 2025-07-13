import streamlit as st
import os
from rag_pipeline import build_index, load_vectorstore, query_rag

# Constants
DEFAULT_PDF_DIR = "./judgements_pdfs/"
DEFAULT_FAISS_PATH = "./faiss_index/"

# Disclaimer
st.sidebar.warning(
    "This is a proof-of-concept (POC) application for testing purposes only. "
    "It may not handle all edge cases, and performance is optimized for up to 50,000 PDFs on CPU. "
    "Use at your own risk and verify results."
)

def main():
    st.title("AI-Powered Legal Judgements Search System")
    
    # PDF Directory Input
    pdf_dir = st.text_input("PDF Directory", value=DEFAULT_PDF_DIR)
    
    # Build Index Button
    if st.button("Build Index"):
        with st.spinner("Building index... This may take a while."):
            try:
                build_index(pdf_dir=pdf_dir, faiss_path=DEFAULT_FAISS_PATH)
                st.success("Index built successfully!")
            except Exception as e:
                st.error(f"Error building index: {e}")
    
    # Query Input
    query = st.text_input("Enter your query:")
    
    if query:
        with st.spinner("Querying..."):
            try:
                vectorstore = load_vectorstore(faiss_path=DEFAULT_FAISS_PATH)
                response, sources = query_rag(query, vectorstore)
                st.subheader("Response")
                st.write(response)
                
                st.subheader("Sources")
                for source in sources:
                    st.write(f"- {source}")
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Error querying: {e}")

if __name__ == "__main__":
    main()